from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from beam_search_benchmark import BeamState, Proposal
from dataset import RuleMetadata
from gpu_proposals import (
    GpuRuleIndex,
    SelectedProposalTensors,
    build_gpu_proposals,
    materialize_selected_proposals,
)
from hierarchical_actions import hierarchical_paged_matches
from lazy_rollout_benchmark import lazy_child
from paged_rollout_benchmark import advance_selected
from ppo_core import (
    clipped_ppo_objective,
    generalized_advantages,
    shaped_transition_reward,
)
from train_paged_ppo import (
    EpisodeMetrics,
    batched_proposal_features,
    initialize_episode_batch,
    pad_batched_policy_inputs,
    refresh_speculative_runtimes,
)


ExactStateKey = tuple[int, int]
ExactActionKey = tuple[int, tuple[int, ...]]
RejectedActionCache = dict[ExactStateKey, set[ExactActionKey]]


@dataclass
class HierarchicalPPOTransition:
    state_features: torch.Tensor
    prefix_state: torch.Tensor
    node_features: torch.Tensor
    node_mask: torch.Tensor
    candidate_features: torch.Tensor
    matcher_logits: torch.Tensor
    candidate_nodes: torch.Tensor
    candidate_mask: torch.Tensor
    action_index: int
    old_log_prob: float
    old_value: float
    reward: float
    done: bool
    legal: bool
    xfer_id: int
    matcher_probability: float
    history_depth: int = 0
    committed_action: bool = False
    repeated_state: bool = False
    candidate_count: int = 0
    policy_entropy: float = 0.0
    previous_gate_count: int = 0
    next_gate_count: int = 0
    advantage: float = 0.0
    return_value: float = 0.0


def proposal_action_key(xfer_id: int, binding: tuple[int, ...]) -> ExactActionKey:
    return int(xfer_id), tuple(int(slot) for slot in binding if int(slot) >= 0)


def exact_state_key(runtime) -> ExactStateKey | None:
    state = runtime.state
    if state.depth != state.exact_checkpoint_depth:
        return None
    return (
        int(state.exact_graph_checkpoint.hash()),
        int(state.topology_index.fingerprint),
    )


def remember_rejected_action(
    cache: RejectedActionCache | None,
    state_key: ExactStateKey | None,
    action_key: ExactActionKey,
) -> bool:
    if cache is None or state_key is None:
        return False
    actions = cache.setdefault(state_key, set())
    before = len(actions)
    actions.add(action_key)
    return len(actions) != before


def mask_rejected_proposals(
    candidate_mask: torch.Tensor,
    candidate_flat_indices: torch.Tensor,
    proposals: SelectedProposalTensors,
    state_keys: list[ExactStateKey | None],
    cache: RejectedActionCache | None,
) -> int:
    """Mask deterministic failures already audited at the same exact state."""
    if not cache:
        return 0
    masked = 0
    for parent_index, state_key in enumerate(state_keys):
        rejected = cache.get(state_key) if state_key is not None else None
        if not rejected:
            continue
        positions = candidate_mask[parent_index].nonzero(as_tuple=False).flatten()
        flat_indices = candidate_flat_indices[parent_index, positions]
        xfers = proposals.xfer_ids.index_select(0, flat_indices).cpu().tolist()
        bindings = proposals.bindings.index_select(0, flat_indices).cpu().tolist()
        blocked = [
            offset
            for offset, (xfer_id, binding) in enumerate(zip(xfers, bindings))
            if proposal_action_key(xfer_id, tuple(binding)) in rejected
        ]
        if blocked:
            blocked_positions = positions[
                torch.tensor(blocked, dtype=torch.long, device=positions.device)
            ]
            candidate_mask[parent_index, blocked_positions] = False
            masked += len(blocked)
    return masked


def proposal_node_positions(
    proposals: SelectedProposalTensors,
    selected_nodes: torch.Tensor,
    selected_node_mask: torch.Tensor,
) -> torch.Tensor:
    parent_nodes = selected_nodes.index_select(0, proposals.parent_ids)
    parent_mask = selected_node_mask.index_select(0, proposals.parent_ids)
    matches = parent_mask & parent_nodes.eq(proposals.anchor_slots.unsqueeze(1))
    if not bool(matches.any(1).all()):
        raise RuntimeError("an expanded proposal anchor is absent from node Top-K")
    if bool(matches.sum(1).ne(1).any()):
        raise RuntimeError("selected node slots must be unique within a state")
    return matches.to(torch.int64).argmax(1)


def pad_proposal_node_positions(
    flat_node_positions: torch.Tensor,
    candidate_flat_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    padded = torch.zeros_like(candidate_flat_indices)
    padded[candidate_mask] = flat_node_positions[
        candidate_flat_indices[candidate_mask]
    ]
    return padded


def finalize_hierarchical_episode(
    transitions: list[HierarchicalPPOTransition],
    gamma: float,
    gae_lambda: float,
) -> None:
    if not transitions:
        return
    transitions[-1].done = True
    advantages, returns = generalized_advantages(
        torch.tensor([row.reward for row in transitions]),
        torch.tensor([row.old_value for row in transitions]),
        torch.tensor([row.done for row in transitions]),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    for transition, advantage, return_value in zip(
        transitions, advantages.tolist(), returns.tolist()
    ):
        transition.advantage = float(advantage)
        transition.return_value = float(return_value)


def collate_hierarchical_transitions(
    transitions: list[HierarchicalPPOTransition],
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = [transitions[int(index)] for index in indices]
    max_candidates = max(row.candidate_features.shape[0] for row in rows)
    policy_width = rows[0].candidate_features.shape[1]
    candidate_features = torch.zeros(
        (len(rows), max_candidates, policy_width), dtype=torch.float
    )
    matcher_logits = torch.zeros((len(rows), max_candidates), dtype=torch.float)
    candidate_nodes = torch.zeros(
        (len(rows), max_candidates), dtype=torch.long
    )
    candidate_mask = torch.zeros(
        (len(rows), max_candidates), dtype=torch.bool
    )
    for index, row in enumerate(rows):
        count = row.candidate_features.shape[0]
        candidate_features[index, :count] = row.candidate_features
        matcher_logits[index, :count] = row.matcher_logits
        candidate_nodes[index, :count] = row.candidate_nodes
        candidate_mask[index, :count] = row.candidate_mask
    return {
        "state_features": torch.stack([row.state_features for row in rows]).to(device),
        "prefix_states": torch.stack([row.prefix_state for row in rows]).to(device),
        "node_features": torch.stack([row.node_features for row in rows]).to(device),
        "node_mask": torch.stack([row.node_mask for row in rows]).to(device),
        "candidate_features": candidate_features.to(device),
        "matcher_logits": matcher_logits.to(device),
        "candidate_nodes": candidate_nodes.to(device),
        "candidate_mask": candidate_mask.to(device),
        "actions": torch.tensor([row.action_index for row in rows], device=device),
        "old_log_probs": torch.tensor(
            [row.old_log_prob for row in rows], device=device
        ),
        "old_values": torch.tensor([row.old_value for row in rows], device=device),
        "advantages": torch.tensor([row.advantage for row in rows], device=device),
        "returns": torch.tensor([row.return_value for row in rows], device=device),
    }


def hierarchical_ppo_update(
    actor_critic,
    optimizer: torch.optim.Optimizer,
    transitions: list[HierarchicalPPOTransition],
    *,
    device: torch.device,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
    target_kl: float,
    max_grad_norm: float,
    seed: int,
) -> dict[str, float | int | bool]:
    if not transitions:
        raise ValueError("PPO update requires at least one transition")
    advantages = torch.tensor([row.advantage for row in transitions])
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-6)
    for transition, advantage in zip(transitions, advantages.tolist()):
        transition.advantage = float(advantage)

    generator = torch.Generator().manual_seed(seed)
    totals = defaultdict(float)
    batches = 0
    completed_epochs = 0
    early_stopped = False
    actor_critic.train()
    for _ in range(epochs):
        epoch_kl = 0.0
        epoch_batches = 0
        order = torch.randperm(len(transitions), generator=generator)
        for begin in range(0, len(transitions), minibatch_size):
            batch = collate_hierarchical_transitions(
                transitions, order[begin : begin + minibatch_size], device
            )
            policy = actor_critic.policy(
                batch["node_features"],
                batch["node_mask"],
                batch["candidate_features"],
                batch["matcher_logits"],
                batch["candidate_nodes"],
                batch["candidate_mask"],
                batch["prefix_states"],
                batch["state_features"],
                include_stop=False,
            )
            distribution = torch.distributions.Categorical(logits=policy.log_probs)
            new_log_probs = distribution.log_prob(batch["actions"])
            new_values = actor_critic.state_values(
                batch["state_features"], prefix_states=batch["prefix_states"]
            )
            objective = clipped_ppo_objective(
                new_log_probs,
                batch["old_log_probs"],
                batch["advantages"],
                new_values,
                batch["old_values"],
                batch["returns"],
                distribution.entropy(),
                clip_epsilon=clip_epsilon,
                value_coefficient=value_coefficient,
                entropy_coefficient=entropy_coefficient,
            )
            optimizer.zero_grad(set_to_none=True)
            objective.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                actor_critic.parameters(), max_grad_norm
            )
            optimizer.step()
            for key, value in (
                ("loss", objective.loss),
                ("policy_loss", objective.policy_loss),
                ("value_loss", objective.value_loss),
                ("entropy", objective.entropy),
                ("approximate_kl", objective.approximate_kl),
                ("clip_fraction", objective.clip_fraction),
            ):
                totals[key] += float(value.detach())
            totals["grad_norm"] += float(grad_norm)
            batches += 1
            epoch_kl += float(objective.approximate_kl.detach())
            epoch_batches += 1
        completed_epochs += 1
        if target_kl and epoch_kl / max(1, epoch_batches) > 1.5 * target_kl:
            early_stopped = True
            break
    return {
        **{key: value / max(1, batches) for key, value in totals.items()},
        "batches": batches,
        "completed_epochs": completed_epochs,
        "target_kl_early_stopped": early_stopped,
    }


def _empty_policy_inputs(
    encoded: torch.Tensor, actor_critic
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = encoded.shape[0]
    return (
        encoded.new_zeros((batch_size, 0, actor_critic.policy_feature_dim)),
        encoded.new_zeros((batch_size, 0)),
        torch.zeros((batch_size, 0), dtype=torch.bool, device=encoded.device),
        torch.empty((batch_size, 0), dtype=torch.long, device=encoded.device),
    )


def collect_hierarchical_episode_batch(
    qasm,
    batch_size: int,
    *,
    context,
    quartz,
    xfers,
    rules: RuleMetadata,
    source_patterns,
    destination_patterns,
    rule_index: GpuRuleIndex,
    model,
    actor_critic,
    device: torch.device,
    threshold_config: dict,
    source_vectors: torch.Tensor,
    source_representations: torch.Tensor | None,
    max_steps: int,
    node_k: int,
    pattern_k: int,
    max_actions: int,
    invalid_reward: float,
    cycle_reward: float,
    step_penalty: float,
    max_rejected_actions_per_step: int,
    terminate_on_improvement: bool,
    gamma: float,
    gae_lambda: float,
    page_size: int,
    initial_gate_bias: float,
    greedy: bool,
    best_by_circuit: dict[str, dict],
    replay_pool: dict[str, dict],
    replay_capacity_per_circuit: int,
    refresh_interval: int,
    topology_audit_interval: int = 1,
    rejected_action_cache: RejectedActionCache | None = None,
    max_exact_rejections_per_episode: int = 8,
    start_from_best: bool = False,
    best_start_probability: float = 1.0,
    use_replay_starts: bool = False,
    replay_start_probability: float = 0.0,
    collector_timing: dict[str, float] | None = None,
) -> tuple[list[HierarchicalPPOTransition], list[EpisodeMetrics]]:
    if batch_size < 1 or refresh_interval < 1:
        raise ValueError("batch size and refresh interval must be positive")
    timing = defaultdict(float)
    started = time.perf_counter()
    active, states, live, gate_types, arena, handles = initialize_episode_batch(
        qasm,
        batch_size,
        context=context,
        quartz=quartz,
        model=model,
        device=device,
        max_steps=max_steps,
        page_size=page_size,
        initialization_backend="deduplicated",
        start_from_best=start_from_best,
        use_replay_starts=use_replay_starts,
        replay_start_probability=replay_start_probability,
        best_by_circuit=best_by_circuit,
        replay_pool=replay_pool,
        best_start_probability=best_start_probability,
    )
    all_runtimes = list(active)
    exact_rejection_counts = {id(runtime): 0 for runtime in all_runtimes}

    while active:
        current_states = [runtime.state for runtime in active]
        matched = hierarchical_paged_matches(
            current_states,
            states,
            live,
            gate_types,
            handles,
            arena,
            model,
            actor_critic,
            device,
            threshold_config,
            source_vectors,
            microbatch=len(active),
            node_k=node_k,
            pattern_k=pattern_k,
        )
        timing["hierarchical_match_seconds"] += matched.elapsed_seconds

        stage_started = time.perf_counter()
        _, proposal_metrics, _, proposal_tensors = build_gpu_proposals(
            matched.candidates,
            current_states,
            rule_index,
            per_parent_cap=max_actions,
            global_cap=max_actions * len(active),
            ranking_mode="probability",
            preselect_matches=False,
            return_selected_tensors=True,
            materialize_python_proposals=False,
        )
        timing["proposal_seconds"] += time.perf_counter() - stage_started
        timing["eligible_actions"] += int(proposal_metrics["eligible_actions"])
        timing["selected_actions"] += int(proposal_metrics.get("selected_actions", 0))

        stage_started = time.perf_counter()
        if proposal_tensors is not None and proposal_tensors.parent_ids.numel():
            flat_features, flat_logits, flat_parent_ids = batched_proposal_features(
                model,
                matched.encoded_states,
                live,
                current_states,
                proposals=None,
                rules=rules,
                device=device,
                initial_gate_bias=initial_gate_bias,
                ordered_roles=False,
                proposal_tensors=proposal_tensors,
                source_representations=source_representations,
            )
            (
                policy_features,
                matcher_logits,
                candidate_mask,
                _,
                candidate_flat_indices,
            ) = pad_batched_policy_inputs(
                flat_features,
                flat_logits,
                proposals=None,
                batch_size=len(active),
                parent_ids=flat_parent_ids,
                backend="tensorized",
            )
            flat_node_positions = proposal_node_positions(
                proposal_tensors,
                matched.selected_nodes,
                matched.selected_node_mask,
            )
            candidate_nodes = pad_proposal_node_positions(
                flat_node_positions, candidate_flat_indices, candidate_mask
            )
        else:
            policy_features, matcher_logits, candidate_mask, candidate_nodes = (
                _empty_policy_inputs(matched.encoded_states, actor_critic)
            )
            candidate_flat_indices = torch.empty_like(candidate_nodes)

        state_cache_keys = [exact_state_key(runtime) for runtime in active]
        if proposal_tensors is not None and proposal_tensors.parent_ids.numel():
            stage_started = time.perf_counter()
            masked_rejections = mask_rejected_proposals(
                candidate_mask,
                candidate_flat_indices,
                proposal_tensors,
                state_cache_keys,
                rejected_action_cache,
            )
            timing["rejected_action_mask_seconds"] += (
                time.perf_counter() - stage_started
            )
            timing["rejected_action_cache_hits"] += masked_rejections

        with torch.no_grad():
            old_values = actor_critic.state_values(
                matched.state_features,
                prefix_states=matched.prefix_states,
            )
        saved_state_features = matched.state_features.float().cpu()
        saved_prefix_states = matched.prefix_states.float().cpu()
        saved_node_features = matched.selected_node_features.float().cpu()
        saved_node_mask = matched.selected_node_mask.cpu()
        saved_policy_features = policy_features.float().cpu()
        saved_matcher_logits = matcher_logits.float().cpu()
        saved_candidate_nodes = candidate_nodes.cpu()
        saved_candidate_mask = candidate_mask.cpu()
        saved_old_values = old_values.float().cpu().tolist()
        timing["policy_preparation_seconds"] += time.perf_counter() - stage_started

        advance_records: dict[int, tuple[BeamState, Proposal]] = {}
        exact_refresh_actions: dict[
            int, tuple[ExactStateKey, ExactActionKey, BeamState, int]
        ] = {}
        refresh_indices = set()
        candidate_presence = candidate_mask.any(1).tolist()
        for parent_index, runtime in enumerate(active):
            if not candidate_presence[parent_index]:
                runtime.terminated_reason = "no_candidates"
                runtime.stopped = True
                if runtime.pending_transition_indices:
                    refresh_indices.add(parent_index)

        unresolved = [
            index for index, present in enumerate(candidate_presence) if present
        ]
        rejected_counts = {index: 0 for index in unresolved}
        while unresolved:
            unresolved_tensor = torch.tensor(
                unresolved, dtype=torch.long, device=device
            )
            stage_started = time.perf_counter()
            with torch.no_grad():
                policy = actor_critic.policy(
                    matched.selected_node_features.index_select(0, unresolved_tensor),
                    matched.selected_node_mask.index_select(0, unresolved_tensor),
                    policy_features.index_select(0, unresolved_tensor),
                    matcher_logits.index_select(0, unresolved_tensor),
                    candidate_nodes.index_select(0, unresolved_tensor),
                    candidate_mask.index_select(0, unresolved_tensor),
                    matched.prefix_states.index_select(0, unresolved_tensor),
                    matched.state_features.index_select(0, unresolved_tensor),
                    include_stop=False,
                )
                distribution = torch.distributions.Categorical(logits=policy.log_probs)
                actions = (
                    distribution.logits.argmax(-1)
                    if greedy
                    else distribution.sample()
                )
                log_probs = distribution.log_prob(actions)
                entropies = distribution.entropy()
            actor_outputs = torch.stack(
                (actions.float(), log_probs, entropies), dim=1
            ).cpu().tolist()
            chosen_flat_indices = candidate_flat_indices[
                unresolved_tensor, actions
            ]
            chosen_proposals = materialize_selected_proposals(
                proposal_tensors, chosen_flat_indices
            )
            timing["policy_sample_seconds"] += time.perf_counter() - stage_started

            retry = []
            for row_index, parent_index in enumerate(unresolved):
                runtime = active[parent_index]
                action_index = int(actor_outputs[row_index][0])
                proposal = chosen_proposals[row_index]
                action_key = proposal_action_key(
                    proposal.xfer_id, tuple(proposal.binding or ())
                )
                parent_state_key = state_cache_keys[parent_index]
                local_mask = candidate_mask[parent_index]
                candidate_count = int(local_mask.sum().item())

                stage_started = time.perf_counter()
                child, fingerprint, duplicate = lazy_child(
                    runtime.state,
                    proposal,
                    source_patterns,
                    destination_patterns,
                    structural_recheck=False,
                    dedup_mode="raw",
                    seen=runtime.topology_hashes,
                    topology_backend="indexed",
                )
                timing["lazy_apply_seconds"] += time.perf_counter() - stage_started
                rejected = child is None
                can_retry = (
                    rejected
                    and rejected_counts[parent_index] < max_rejected_actions_per_step
                    and candidate_count > 1
                )
                legal = bool(duplicate or child is not None)
                reward = float(invalid_reward)
                if duplicate:
                    reward = float(cycle_reward)
                elif child is not None:
                    reward = shaped_transition_reward(
                        runtime.state.gate_count,
                        child.gate_count,
                        repeated_state=False,
                        step_penalty=step_penalty,
                        cycle_reward=cycle_reward,
                    )
                next_gate_count = (
                    child.gate_count if child is not None else runtime.state.gate_count
                )
                transition = HierarchicalPPOTransition(
                    state_features=saved_state_features[parent_index],
                    prefix_state=saved_prefix_states[parent_index],
                    node_features=saved_node_features[parent_index],
                    node_mask=saved_node_mask[parent_index],
                    candidate_features=saved_policy_features[parent_index],
                    matcher_logits=saved_matcher_logits[parent_index],
                    candidate_nodes=saved_candidate_nodes[parent_index],
                    candidate_mask=saved_candidate_mask[parent_index].clone(),
                    action_index=action_index,
                    old_log_prob=float(actor_outputs[row_index][1]),
                    old_value=float(saved_old_values[parent_index]),
                    reward=reward,
                    done=rejected and not can_retry,
                    legal=legal,
                    xfer_id=proposal.xfer_id,
                    matcher_probability=proposal.probability,
                    history_depth=runtime.state.depth + 1,
                    committed_action=child is not None,
                    repeated_state=bool(duplicate),
                    candidate_count=candidate_count,
                    policy_entropy=float(actor_outputs[row_index][2]),
                    previous_gate_count=runtime.state.gate_count,
                    next_gate_count=next_gate_count,
                )
                runtime.transitions.append(transition)
                if rejected:
                    if remember_rejected_action(
                        rejected_action_cache, parent_state_key, action_key
                    ):
                        timing["rejected_action_cache_inserts"] += 1
                    candidate_mask[parent_index, action_index] = False
                    saved_candidate_mask[parent_index, action_index] = False
                    if can_retry:
                        rejected_counts[parent_index] += 1
                        retry.append(parent_index)
                        continue
                    runtime.terminated_reason = (
                        "cycle_actions_exhausted"
                        if duplicate
                        else "rejected_actions_exhausted"
                    )
                    runtime.stopped = True
                    if runtime.pending_transition_indices:
                        refresh_indices.add(parent_index)
                    continue

                runtime.pending_transition_indices.append(
                    len(runtime.transitions) - 1
                )
                if parent_state_key is not None:
                    exact_refresh_actions[parent_index] = (
                        parent_state_key,
                        action_key,
                        runtime.state,
                        int(fingerprint),
                    )
                runtime.state = child
                runtime.topology_hashes.add(int(fingerprint))
                advance_records[parent_index] = (child, proposal)
                if child.depth >= max_steps:
                    runtime.terminated_reason = "horizon"
                    runtime.stopped = True
                elif terminate_on_improvement and child.gate_count < runtime.initial_gate_count:
                    runtime.terminated_reason = "improvement"
                    runtime.stopped = True
                improved_global = (
                    child.gate_count
                    < best_by_circuit[runtime.circuit]["gate_count"]
                )
                if (
                    child.depth - child.exact_checkpoint_depth >= refresh_interval
                    or runtime.stopped
                    or improved_global
                ):
                    refresh_indices.add(parent_index)
            unresolved = retry

        stage_started = time.perf_counter()
        refresh_speculative_runtimes(
            [active[index] for index in sorted(refresh_indices)],
            context=context,
            quartz=quartz,
            xfers=xfers,
            invalid_reward=invalid_reward,
            cycle_reward=cycle_reward,
            replay_capacity_per_circuit=replay_capacity_per_circuit,
            best_by_circuit=best_by_circuit,
            replay_pool=replay_pool,
            topology_audit_interval=topology_audit_interval,
            profile_timing=timing,
            profile_counts=timing,
        )
        exact_retry_indices = set()
        for parent_index, (
            state_key,
            action_key,
            parent_state,
            child_fingerprint,
        ) in exact_refresh_actions.items():
            runtime = active[parent_index]
            if runtime.terminated_reason not in {
                "exact_cycle",
                "exact_refresh_failure",
                "exact_topology_mismatch",
            }:
                continue
            if remember_rejected_action(
                rejected_action_cache, state_key, action_key
            ):
                timing["rejected_action_cache_inserts"] += 1
            rejection_count = exact_rejection_counts[id(runtime)]
            can_retry_exact = (
                rejected_action_cache is not None
                and rejection_count < max_exact_rejections_per_episode
            )
            if not can_retry_exact:
                continue
            exact_rejection_counts[id(runtime)] = rejection_count + 1
            if runtime.transitions:
                runtime.transitions[-1].done = False
            runtime.state = parent_state
            runtime.topology_hashes.discard(child_fingerprint)
            runtime.stopped = False
            runtime.final_gate_count = None
            runtime.terminated_reason = "horizon"
            advance_records.pop(parent_index, None)
            exact_retry_indices.add(parent_index)
            timing["exact_rejection_retries"] += 1
        timing["refresh_seconds"] += time.perf_counter() - stage_started

        continuing_indices = []
        accepted_indices = []
        continuing_records = []
        continuing_runtimes = []
        for parent_index, runtime in enumerate(active):
            record = advance_records.get(parent_index)
            if runtime.stopped:
                continue
            if record is None and parent_index not in exact_retry_indices:
                continue
            continuing_indices.append(parent_index)
            continuing_runtimes.append(runtime)
            if record is not None:
                accepted_indices.append(parent_index)
                continuing_records.append(record)
        if not continuing_indices:
            for handle in handles:
                arena.release(handle)
            break
        if not continuing_records:
            stopped_indices = set(range(len(active))) - set(continuing_indices)
            for index in stopped_indices:
                arena.release(handles[index])
            selected = torch.tensor(
                continuing_indices, dtype=torch.long, device=device
            )
            states = states.index_select(0, selected)
            live = live.index_select(0, selected)
            gate_types = gate_types.index_select(0, selected)
            handles = [handles[index] for index in continuing_indices]
        else:
            for index in exact_retry_indices:
                arena.retain(handles[index])
            (
                advanced_states,
                advanced_live,
                advanced_types,
                advanced_handles,
                advance_seconds,
                _,
            ) = advance_selected(
                states,
                live,
                gate_types,
                handles,
                continuing_records,
                rules,
                arena,
                model,
                device,
                microbatch=len(continuing_records),
                trusted_paged_inputs=True,
                source_representations=source_representations,
            )
            target_slots = advanced_states.shape[1]
            if states.shape[1] < target_slots:
                extra = target_slots - states.shape[1]
                states = F.pad(states, (0, 0, 0, extra))
                live = F.pad(live, (0, extra), value=False)
                gate_types = F.pad(gate_types, (0, extra), value=-1)
            accepted_rows = {
                parent_index: row_index
                for row_index, parent_index in enumerate(accepted_indices)
            }
            next_states = []
            next_live = []
            next_types = []
            next_handles = []
            for parent_index in continuing_indices:
                advanced_row = accepted_rows.get(parent_index)
                if advanced_row is None:
                    next_states.append(states[parent_index : parent_index + 1])
                    next_live.append(live[parent_index : parent_index + 1])
                    next_types.append(gate_types[parent_index : parent_index + 1])
                    next_handles.append(handles[parent_index])
                else:
                    next_states.append(
                        advanced_states[advanced_row : advanced_row + 1]
                    )
                    next_live.append(advanced_live[advanced_row : advanced_row + 1])
                    next_types.append(
                        advanced_types[advanced_row : advanced_row + 1]
                    )
                    next_handles.append(advanced_handles[advanced_row])
            states = torch.cat(next_states)
            live = torch.cat(next_live)
            gate_types = torch.cat(next_types)
            handles = next_handles
            timing["cache_advance_seconds"] += advance_seconds
        active = continuing_runtimes

    transitions = []
    episode_metrics = []
    for runtime in all_runtimes:
        finalize_hierarchical_episode(runtime.transitions, gamma, gae_lambda)
        transitions.extend(runtime.transitions)
        legal_actions = sum(row.legal for row in runtime.transitions)
        cycle_actions = sum(row.repeated_state for row in runtime.transitions)
        accepted_rewrites = sum(
            row.committed_action and row.legal and not row.repeated_state
            for row in runtime.transitions
        )
        best_gate_count = min(
            [runtime.initial_gate_count]
            + [
                row.next_gate_count
                for row in runtime.transitions
                if row.committed_action and row.legal
            ]
        )
        episode_metrics.append(
            EpisodeMetrics(
                circuit=runtime.circuit,
                steps=len(runtime.transitions),
                legal_actions=legal_actions,
                invalid_actions=len(runtime.transitions) - legal_actions,
                cycle_actions=cycle_actions,
                accepted_rewrites=accepted_rewrites,
                total_reward=sum(row.reward for row in runtime.transitions),
                initial_gate_count=runtime.initial_gate_count,
                final_gate_count=(
                    runtime.final_gate_count
                    if runtime.final_gate_count is not None
                    else runtime.state.gate_count
                ),
                best_gate_count=best_gate_count,
                mean_candidates=(
                    sum(row.candidate_count for row in runtime.transitions)
                    / max(1, len(runtime.transitions))
                ),
                mean_entropy=(
                    sum(row.policy_entropy for row in runtime.transitions)
                    / max(1, len(runtime.transitions))
                ),
                started_from_replay=runtime.started_from_replay,
                terminated_reason=runtime.terminated_reason,
                exact_refreshes=runtime.exact_refreshes,
                exact_replay_actions=runtime.exact_replay_actions,
                exact_refresh_seconds=runtime.exact_refresh_seconds,
            )
        )
    timing["total_seconds"] = time.perf_counter() - started
    if collector_timing is not None:
        collector_timing.update(timing)
    return transitions, episode_metrics
