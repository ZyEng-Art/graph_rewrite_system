from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from beam_search_benchmark import BeamState, Proposal
from ppo_core import build_policy_features, segmented_log_softmax
from threshold_inference import CandidateTensors


@dataclass(frozen=True)
class GpuRuleIndex:
    xfer_offsets: torch.Tensor
    xfer_counts: torch.Tensor
    xfer_ids: torch.Tensor
    gate_deltas: torch.Tensor
    all_xfer_counts: torch.Tensor
    best_xfer_ids: torch.Tensor
    best_gate_deltas: torch.Tensor
    min_gate_delta: int
    max_gate_delta: int
    num_xfers: int

    @classmethod
    def build(
        cls,
        source_to_xfers: dict[int, list[int]],
        gate_deltas: list[int],
        num_sources: int,
        max_gate_increase: int,
        device: torch.device,
    ) -> "GpuRuleIndex":
        allowed: list[int] = []
        offsets = [0]
        counts = []
        all_counts = []
        best_xfers = []
        best_deltas = []
        for source in range(num_sources):
            source_xfers = source_to_xfers.get(source, [])
            selected = [
                xfer
                for xfer in source_xfers
                if gate_deltas[xfer] <= max_gate_increase
            ]
            allowed.extend(selected)
            counts.append(len(selected))
            all_counts.append(len(source_xfers))
            if selected:
                best_xfer = min(selected, key=lambda xfer: (gate_deltas[xfer], xfer))
                best_xfers.append(best_xfer)
                best_deltas.append(gate_deltas[best_xfer])
            else:
                best_xfers.append(-1)
                best_deltas.append(torch.iinfo(torch.int64).max)
            offsets.append(len(allowed))
        return cls(
            xfer_offsets=torch.tensor(offsets[:-1], device=device),
            xfer_counts=torch.tensor(counts, device=device),
            xfer_ids=torch.tensor(allowed, device=device),
            gate_deltas=torch.tensor(gate_deltas, device=device),
            all_xfer_counts=torch.tensor(all_counts, device=device),
            best_xfer_ids=torch.tensor(best_xfers, device=device),
            best_gate_deltas=torch.tensor(best_deltas, device=device),
            min_gate_delta=min(gate_deltas),
            max_gate_delta=max(gate_deltas),
            num_xfers=len(gate_deltas),
        )


@dataclass(frozen=True)
class SelectedProposalTensors:
    parent_ids: torch.Tensor
    xfer_ids: torch.Tensor
    source_ids: torch.Tensor
    anchor_slots: torch.Tensor
    bindings: torch.Tensor
    probabilities: torch.Tensor
    gate_deltas: torch.Tensor
    next_gate_counts: torch.Tensor
    value_scores: torch.Tensor


def materialize_selected_proposals(
    tensors: SelectedProposalTensors,
    indices: torch.Tensor | None = None,
) -> list[Proposal]:
    if indices is None:
        indices = torch.arange(
            tensors.parent_ids.numel(),
            dtype=torch.long,
            device=tensors.parent_ids.device,
        )
    else:
        indices = indices.to(device=tensors.parent_ids.device, dtype=torch.long)
    metadata = torch.stack(
        (
            tensors.parent_ids[indices],
            tensors.xfer_ids[indices],
            tensors.anchor_slots[indices],
            tensors.next_gate_counts[indices],
        ),
        dim=1,
    ).cpu()
    bindings = tensors.bindings[indices].cpu()
    scores = torch.stack(
        (tensors.probabilities[indices], tensors.value_scores[indices]), dim=1
    ).cpu()
    proposals = []
    for row, binding, score in zip(
        metadata.tolist(), bindings.tolist(), scores.tolist()
    ):
        parent, xfer_id, anchor, next_gate_count = row
        proposals.append(
            Proposal(
                parent=parent,
                xfer_id=xfer_id,
                anchor_slot=anchor,
                binding=tuple(slot for slot in binding if slot >= 0),
                probability=score[0],
                next_gate_count=next_gate_count,
                value_score=score[1],
            )
        )
    return proposals


def _stable_lexsort(
    tensors: list[tuple[torch.Tensor, bool]],
) -> torch.Tensor:
    """Stable lexicographic order; keys are listed most-significant first."""
    if not tensors:
        raise ValueError("at least one sort key is required")
    order = torch.arange(tensors[0][0].numel(), device=tensors[0][0].device)
    for values, descending in reversed(tensors):
        local = torch.argsort(
            values[order], descending=descending, stable=True
        )
        order = order[local]
    return order


def _preselect_match_rows(
    candidates: CandidateTensors,
    candidate_rows: torch.Tensor,
    rule_index: GpuRuleIndex,
    parent_gate_counts: torch.Tensor,
    per_parent_cap: int,
    ranking_mode: str,
) -> torch.Tensor:
    """Return every match that can still contribute a per-parent top action."""
    parents = candidates.batch_ids[candidate_rows]
    sources = candidates.sources[candidate_rows]
    probabilities = candidates.probabilities[candidate_rows]
    counts = torch.bincount(parents, minlength=parent_gate_counts.numel())
    max_matches = int(counts.max().item())
    positions = torch.arange(parents.numel(), device=parents.device)
    new_parent = torch.ones(parents.numel(), dtype=torch.bool, device=parents.device)
    new_parent[1:] = parents[1:] != parents[:-1]
    group_starts = torch.where(new_parent, positions, 0)
    group_starts = torch.cummax(group_starts, dim=0).values
    offsets = positions - group_starts
    padded_rows = torch.full(
        (parent_gate_counts.numel(), max_matches),
        -1,
        dtype=torch.long,
        device=parents.device,
    )
    padded_rows[parents, offsets] = candidate_rows
    valid = padded_rows.ge(0)
    safe_rows = padded_rows.clamp_min(0)
    padded_sources = candidates.sources[safe_rows]
    best_next_gate_counts = (
        parent_gate_counts.unsqueeze(1)
        + rule_index.best_gate_deltas[padded_sources]
    )
    best_xfer_ids = rule_index.best_xfer_ids[padded_sources]
    padded_probabilities = candidates.probabilities[safe_rows]
    probability_rank = (
        padded_probabilities.float().contiguous().view(torch.int32).to(torch.int64)
    )
    delta_rank = rule_index.max_gate_delta - (
        best_next_gate_counts - parent_gate_counts.unsqueeze(1)
    )
    xfer_rank = rule_index.num_xfers - 1 - best_xfer_ids
    local_rank = max_matches - 1 - torch.arange(
        max_matches, device=parents.device
    ).unsqueeze(0)
    probability_bits = 31
    delta_bits = max(
        1, (rule_index.max_gate_delta - rule_index.min_gate_delta).bit_length()
    )
    xfer_bits = max(1, (rule_index.num_xfers - 1).bit_length())
    local_bits = max(1, (max_matches - 1).bit_length())
    required_bits = probability_bits + delta_bits + xfer_bits + local_bits
    if required_bits > 62:
        raise RuntimeError("packed match ranking exceeds signed int64 capacity")
    if ranking_mode in {"gate", "ppo"}:
        rank_score = (delta_rank << probability_bits) | probability_rank
    else:
        rank_score = (probability_rank << delta_bits) | delta_rank
    rank_score = (rank_score << xfer_bits) | xfer_rank
    rank_score = (rank_score << local_bits) | local_rank
    rank_score = rank_score.masked_fill(~valid, -1)
    count = min(per_parent_cap, max_matches)
    _, order = rank_score.topk(count, dim=1, largest=True, sorted=True)
    selected = padded_rows.gather(1, order)
    return selected[selected.ge(0)]


def _match_set_policy_logits(
    actor_critic,
    candidate_features: torch.Tensor,
    matcher_logits: torch.Tensor,
    parent_ids: torch.Tensor,
    prefix_states: torch.Tensor,
    state_features: torch.Tensor,
) -> torch.Tensor:
    """Run a padded candidate-set actor and restore the flattened row order."""
    active_parents, parent_rows = torch.unique(
        parent_ids, sorted=True, return_inverse=True
    )
    order = torch.argsort(parent_rows, stable=True)
    sorted_parent_rows = parent_rows[order]
    positions = torch.arange(order.numel(), device=order.device)
    new_parent = torch.ones(
        order.numel(), dtype=torch.bool, device=order.device
    )
    new_parent[1:] = sorted_parent_rows[1:] != sorted_parent_rows[:-1]
    group_starts = torch.where(new_parent, positions, 0)
    group_starts = torch.cummax(group_starts, dim=0).values
    sorted_offsets = positions - group_starts
    offsets = torch.empty_like(sorted_offsets)
    offsets[order] = sorted_offsets
    max_candidates = int(
        torch.bincount(parent_rows, minlength=active_parents.numel()).max().item()
    )

    padded_features = candidate_features.new_zeros(
        (active_parents.numel(), max_candidates, candidate_features.shape[-1])
    )
    padded_logits = matcher_logits.new_zeros(
        (active_parents.numel(), max_candidates)
    )
    candidate_mask = torch.zeros(
        (active_parents.numel(), max_candidates),
        dtype=torch.bool,
        device=parent_ids.device,
    )
    padded_features[parent_rows, offsets] = candidate_features
    padded_logits[parent_rows, offsets] = matcher_logits
    candidate_mask[parent_rows, offsets] = True
    logits = actor_critic.policy_logits(
        padded_features,
        padded_logits,
        candidate_mask,
        prefix_states.index_select(0, active_parents),
        state_features.index_select(0, active_parents),
    )
    return logits[parent_rows, offsets]


@torch.no_grad()
def build_gpu_proposals(
    candidates: CandidateTensors,
    beam: list[BeamState],
    rule_index: GpuRuleIndex,
    *,
    per_parent_cap: int,
    global_cap: int,
    ranking_mode: str = "gate",
    ranking_seed: int = 0,
    action_value_model=None,
    action_value_states: torch.Tensor | None = None,
    action_value_live: torch.Tensor | None = None,
    action_value_weight: float = 0.0,
    value_increase_cap: int = 0,
    value_exploration_fraction: float = 0.0,
    action_value_microbatch: int = 16384,
    ppo_actor_critic=None,
    ppo_model=None,
    ppo_states: torch.Tensor | None = None,
    ppo_live: torch.Tensor | None = None,
    ppo_prefix_states: torch.Tensor | None = None,
    ppo_state_features: torch.Tensor | None = None,
    ppo_initial_gate_bias: float = 1.0,
    ppo_policy_weight: float = 0.25,
    preselect_matches: bool = False,
    return_selected_tensors: bool = False,
    materialize_python_proposals: bool = True,
    profile_stages: bool = False,
) -> tuple[
    list[Proposal] | None,
    dict[str, float | int],
    dict[str, float],
    SelectedProposalTensors | None,
]:
    """Expand and rank actions, optionally deferring their compact D2H copy."""
    if ranking_mode not in {"gate", "probability", "stochastic", "value", "ppo"}:
        raise ValueError(f"unknown proposal ranking mode: {ranking_mode}")
    if ranking_mode == "value" and (
        action_value_model is None
        or action_value_states is None
        or action_value_live is None
        or action_value_weight <= 0
    ):
        raise ValueError("value ranking requires model states and a positive weight")
    if ranking_mode == "ppo" and (
        ppo_actor_critic is None
        or ppo_model is None
        or ppo_states is None
        or ppo_live is None
    ):
        raise ValueError("PPO ranking requires actor, model, and encoded states")
    if (
        ranking_mode == "ppo"
        and getattr(ppo_actor_critic, "match_set_aware", False)
        and (ppo_prefix_states is None or ppo_state_features is None)
    ):
        raise ValueError("match-set PPO ranking requires prefix and state features")
    if ppo_policy_weight < 0:
        raise ValueError("PPO policy weight must be nonnegative")
    if value_increase_cap < 0 or value_increase_cap > per_parent_cap:
        raise ValueError("value increase cap must be within the per-parent cap")
    if value_increase_cap and ranking_mode != "value":
        raise ValueError("value increase cap is only valid for value ranking")
    if not 0.0 <= value_exploration_fraction <= 1.0:
        raise ValueError("value exploration fraction must be within [0, 1]")
    if value_exploration_fraction and ranking_mode != "value":
        raise ValueError("value exploration is only valid for value ranking")
    if not materialize_python_proposals and not return_selected_tensors:
        raise ValueError("deferred proposals require selected GPU tensors")
    device = candidates.sources.device
    timing: dict[str, float] = {}

    def finish_timing(name: str, started: float) -> None:
        if not profile_stages:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - started

    stage_started = time.perf_counter()
    source_counts = rule_index.xfer_counts[candidates.sources]
    candidate_count = candidates.sources.numel()
    expanded_count = int(source_counts.sum().item())
    predicted_actions = int(
        rule_index.all_xfer_counts[candidates.sources].sum().item()
    )
    if not expanded_count:
        finish_timing("gpu_action_expansion_seconds", stage_started)
        return (
            [] if materialize_python_proposals else None,
            {
                "predicted_actions": predicted_actions,
                "eligible_actions": 0,
            },
            timing,
            None,
        )
    parent_gate_counts = torch.tensor(
        [state.gate_count for state in beam], device=device
    )
    candidate_rows_to_expand = torch.arange(candidate_count, device=device)
    if preselect_matches and ranking_mode in {"gate", "probability", "ppo"}:
        eligible_matches = source_counts.gt(0)
        candidate_rows_to_expand = candidate_rows_to_expand[eligible_matches]
        candidate_rows_to_expand = _preselect_match_rows(
            candidates,
            candidate_rows_to_expand,
            rule_index,
            parent_gate_counts,
            per_parent_cap,
            ranking_mode,
        )
    finish_timing("gpu_match_preselection_seconds", stage_started)

    stage_started = time.perf_counter()
    materialized_source_counts = source_counts[candidate_rows_to_expand]
    materialized_count = int(materialized_source_counts.sum().item())
    candidate_rows = torch.repeat_interleave(
        candidate_rows_to_expand,
        materialized_source_counts,
        output_size=materialized_count,
    )
    group_starts = (
        torch.cumsum(materialized_source_counts, dim=0) - materialized_source_counts
    )
    within_source = torch.arange(
        materialized_count, device=device
    ) - torch.repeat_interleave(
        group_starts,
        materialized_source_counts,
        output_size=materialized_count,
    )
    source_ids = candidates.sources[candidate_rows]
    xfer_ids = rule_index.xfer_ids[
        rule_index.xfer_offsets[source_ids] + within_source
    ]
    parents = candidates.batch_ids[candidate_rows]
    anchors = candidates.anchors[candidate_rows]
    bindings = candidates.bindings[candidate_rows]
    probabilities = candidates.probabilities[candidate_rows]
    next_gate_counts = (
        parent_gate_counts[parents] + rule_index.gate_deltas[xfer_ids]
    )
    random_priorities = None
    if ranking_mode == "stochastic" or value_exploration_fraction:
        generator = torch.Generator(device=device)
        generator.manual_seed(ranking_seed)
        random_priorities = torch.rand(
            materialized_count, device=device, generator=generator
        )
    finish_timing("gpu_action_expansion_seconds", stage_started)

    stage_started = time.perf_counter()
    if ranking_mode in {"gate", "value", "ppo"}:
        rank_keys = [(next_gate_counts, False), (probabilities, True)]
    elif ranking_mode == "probability":
        rank_keys = [(probabilities, True), (next_gate_counts, False)]
    else:
        rank_keys = [(random_priorities, True), (next_gate_counts, False)]
    parent_order = _stable_lexsort(
        [(parents, False), *rank_keys, (xfer_ids, False)]
    )
    ordered_parents = parents[parent_order]
    positions = torch.arange(materialized_count, device=device)
    new_parent = torch.ones(materialized_count, dtype=torch.bool, device=device)
    new_parent[1:] = ordered_parents[1:] != ordered_parents[:-1]
    group_start_positions = torch.where(new_parent, positions, 0)
    group_start_positions = torch.cummax(group_start_positions, dim=0).values
    parent_rank = positions - group_start_positions
    if ranking_mode == "value" and value_increase_cap:
        base_cap = per_parent_cap - value_increase_cap
        base_rows = parent_order[parent_rank < base_cap]
        increasing_rows = torch.where(rule_index.gate_deltas[xfer_ids] > 0)[0]
        if increasing_rows.numel():
            increase_order = increasing_rows[
                _stable_lexsort(
                    [
                        (parents[increasing_rows], False),
                        (probabilities[increasing_rows], True),
                        (next_gate_counts[increasing_rows], False),
                        (xfer_ids[increasing_rows], False),
                    ]
                )
            ]
            ordered_increase_parents = parents[increase_order]
            increase_positions = torch.arange(
                increase_order.numel(), device=device
            )
            new_increase_parent = torch.ones(
                increase_order.numel(), dtype=torch.bool, device=device
            )
            new_increase_parent[1:] = (
                ordered_increase_parents[1:] != ordered_increase_parents[:-1]
            )
            increase_group_starts = torch.where(
                new_increase_parent, increase_positions, 0
            )
            increase_group_starts = torch.cummax(
                increase_group_starts, dim=0
            ).values
            increase_rank = increase_positions - increase_group_starts
            increase_rows = increase_order[increase_rank < value_increase_cap]
            selected_mask = torch.zeros(
                materialized_count, dtype=torch.bool, device=device
            )
            selected_mask[base_rows] = True
            selected_mask[increase_rows] = True
            parent_order = torch.where(selected_mask)[0]
        else:
            parent_order = base_rows
    else:
        parent_order = parent_order[parent_rank < per_parent_cap]
    finish_timing("gpu_per_parent_rank_seconds", stage_started)

    action_values = None
    if ranking_mode == "value":
        stage_started = time.perf_counter()
        value_chunks = []
        for begin in range(0, parent_order.numel(), action_value_microbatch):
            selected_rows = parent_order[begin : begin + action_value_microbatch]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                values = action_value_model.action_values(
                    action_value_states,
                    action_value_live,
                    xfer_ids[selected_rows],
                    source_ids[selected_rows],
                    bindings[selected_rows],
                    parents[selected_rows],
                )
            value_chunks.append(values.float())
        action_values = torch.cat(value_chunks)
        value_mean = action_values.mean()
        value_std = action_values.std(unbiased=False).clamp_min(1e-6)
        action_values = (action_values - value_mean) / value_std
        finish_timing("gpu_action_value_seconds", stage_started)

    stage_started = time.perf_counter()
    if ranking_mode == "value":
        ranking_cost = (
            next_gate_counts[parent_order].float()
            - action_value_weight * action_values
        )
        global_rank_keys = [
            (ranking_cost, False),
            (next_gate_counts[parent_order], False),
            (probabilities[parent_order], True),
        ]
    elif ranking_mode in {"gate", "ppo"}:
        global_rank_keys = [
            (next_gate_counts[parent_order], False),
            (probabilities[parent_order], True),
        ]
    elif ranking_mode == "probability":
        global_rank_keys = [
            (probabilities[parent_order], True),
            (next_gate_counts[parent_order], False),
        ]
    else:
        global_rank_keys = [
            (random_priorities[parent_order], True),
            (next_gate_counts[parent_order], False),
        ]
    global_order = _stable_lexsort(
        [
            *global_rank_keys,
            (parent_gate_counts[parents[parent_order]], False),
        ]
    )
    selection_count = min(global_cap, global_order.numel())
    selected_value_exploration = 0
    if value_exploration_fraction and selection_count > 1:
        selected_value_exploration = min(
            selection_count - 1,
            max(1, round(selection_count * value_exploration_fraction)),
        )
        selected_value_count = selection_count - selected_value_exploration
        value_indices = global_order[:selected_value_count]
        remaining = torch.ones(
            parent_order.numel(), dtype=torch.bool, device=device
        )
        remaining[value_indices] = False
        exploration_candidates = torch.where(remaining)[0]
        exploration_order = torch.argsort(
            random_priorities[parent_order[exploration_candidates]],
            descending=True,
            stable=True,
        )
        exploration_indices = exploration_candidates[
            exploration_order[:selected_value_exploration]
        ]
        selected_indices = torch.empty(
            selection_count, dtype=torch.long, device=device
        )
        exploration_positions = torch.div(
            (torch.arange(selected_value_exploration, device=device) * 2 + 1)
            * selection_count,
            2 * selected_value_exploration,
            rounding_mode="floor",
        )
        exploration_mask = torch.zeros(
            selection_count, dtype=torch.bool, device=device
        )
        exploration_mask[exploration_positions] = True
        selected_indices[exploration_mask] = exploration_indices
        selected_indices[~exploration_mask] = value_indices
    else:
        selected_indices = global_order[:selection_count]
    selected = parent_order[selected_indices]
    selected_value_scores = (
        action_values[selected_indices]
        if action_values is not None
        else torch.zeros(selection_count, device=device)
    )
    finish_timing("gpu_global_proposal_rank_seconds", stage_started)

    if ranking_mode == "ppo":
        stage_started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            candidate_features = ppo_model.candidate_features(
                ppo_states,
                ppo_live,
                xfer_ids[selected],
                source_ids[selected],
                bindings[selected],
                parents[selected],
                ordered_roles=getattr(
                    ppo_actor_critic, "match_set_aware", False
                ),
            )
            policy_features, matcher_logits = build_policy_features(
                candidate_features,
                probabilities[selected],
                rule_index.gate_deltas[xfer_ids[selected]],
                initial_gate_bias=ppo_initial_gate_bias,
            )
            if getattr(ppo_actor_critic, "match_set_aware", False):
                policy_logits = _match_set_policy_logits(
                    ppo_actor_critic,
                    policy_features,
                    matcher_logits,
                    parents[selected],
                    ppo_prefix_states,
                    ppo_state_features,
                ).float()
            else:
                policy_logits = ppo_actor_critic.policy_logits(
                    policy_features, matcher_logits
                ).float()
        policy_scores = segmented_log_softmax(
            policy_logits, parents[selected], len(beam)
        )
        policy_mean = policy_scores.mean()
        policy_std = policy_scores.std(unbiased=False).clamp_min(1e-6)
        standardized_policy_scores = (policy_scores - policy_mean) / policy_std
        ranking_cost = next_gate_counts[selected].float() - (
            ppo_policy_weight * standardized_policy_scores
        )
        ppo_order = _stable_lexsort(
            [
                (ranking_cost, False),
                (next_gate_counts[selected], False),
                (policy_scores, True),
                (probabilities[selected], True),
            ]
        )
        selected = selected[ppo_order]
        selected_value_scores = policy_scores[ppo_order]
        finish_timing("gpu_ppo_policy_seconds", stage_started)

    proposal_tensors = SelectedProposalTensors(
        parent_ids=parents[selected],
        xfer_ids=xfer_ids[selected],
        source_ids=source_ids[selected],
        anchor_slots=anchors[selected],
        bindings=bindings[selected],
        probabilities=probabilities[selected],
        gate_deltas=rule_index.gate_deltas[xfer_ids[selected]],
        next_gate_counts=next_gate_counts[selected],
        value_scores=selected_value_scores,
    )
    selected_tensors = proposal_tensors if return_selected_tensors else None

    stage_started = time.perf_counter()
    proposals = (
        materialize_selected_proposals(proposal_tensors)
        if materialize_python_proposals
        else None
    )
    finish_timing("proposal_device_to_host_and_pack_seconds", stage_started)
    metrics: dict[str, float | int] = {
        "predicted_actions": predicted_actions,
        "eligible_actions": expanded_count,
        "selected_actions": int(selected.numel()),
        "value_increase_candidates_after_parent_cap": (
            int(
                (rule_index.gate_deltas[xfer_ids[parent_order]] > 0)
                .sum()
                .item()
            )
            if ranking_mode == "value"
            else 0
        ),
        "selected_value_exploration_proposals": selected_value_exploration,
    }
    if preselect_matches:
        metrics["materialized_actions"] = materialized_count
    if action_values is not None:
        metrics.update(
            {
                "action_value_candidates": int(action_values.numel()),
                "selected_action_value_mean": float(
                    selected_value_scores.mean().item()
                ),
                "selected_action_value_std": float(
                    selected_value_scores.std(unbiased=False).item()
                ),
            }
        )
    if ranking_mode == "ppo":
        metrics.update(
            {
                "ppo_policy_candidates": int(selected_value_scores.numel()),
                "selected_ppo_score_mean": float(
                    selected_value_scores.mean().item()
                ),
                "ppo_policy_score_std": float(policy_std.item()),
            }
        )
    return proposals, metrics, timing, selected_tensors
