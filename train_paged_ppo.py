from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import ctypes.util
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
import types

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch
import torch.nn.functional as F

from beam_search_benchmark import BeamState, Proposal, snapshot, update_slots
from dataset import RuleMetadata
from gpu_proposals import GpuRuleIndex, build_gpu_proposals
from incremental_graph import parse_pattern
from lazy_rollout_benchmark import indexed_topology, lazy_child, replay_state
from model_factory import build_model
from paged_cache import PagedKVCache
from paged_rollout_benchmark import (
    advance_selected,
    initial_batch,
    paged_model_matches,
)
from ppo_core import (
    PagedPPOActorCritic,
    build_policy_features,
    build_state_features,
    clipped_ppo_objective,
    generalized_advantages,
    masked_policy_distribution,
    shaped_transition_reward,
)
from threshold_inference import load_threshold_config
from train import autocast_context, move_batch


@dataclass
class PPOTransition:
    state_features: torch.Tensor
    candidate_features: torch.Tensor
    matcher_logits: torch.Tensor
    candidate_mask: torch.Tensor
    action_index: int
    old_log_prob: float
    old_value: float
    reward: float
    done: bool
    legal: bool
    xfer_id: int
    matcher_probability: float
    advantage: float = 0.0
    return_value: float = 0.0


@dataclass
class EpisodeMetrics:
    circuit: str
    steps: int
    legal_actions: int
    invalid_actions: int
    cycle_actions: int
    accepted_rewrites: int
    total_reward: float
    initial_gate_count: int
    final_gate_count: int
    best_gate_count: int
    mean_candidates: float
    mean_entropy: float
    started_from_replay: bool
    terminated_reason: str


def make_replay_bucket(graph) -> dict:
    graph_hash = int(graph.hash())
    return {
        "states": [
            {
                "graph_hash": graph_hash,
                "gate_count": int(graph.gate_count),
                "qasm": graph.to_qasm_str(),
            }
        ],
        "retained_hashes": {graph_hash},
        "unique_states_seen": 1,
    }


def retain_replay_state(
    bucket: dict,
    graph,
    *,
    capacity: int,
) -> bool:
    graph_hash = int(graph.hash())
    if graph_hash in bucket["retained_hashes"]:
        return False
    bucket["unique_states_seen"] += 1
    states = bucket["states"]
    if len(states) < capacity:
        replacement = len(states)
    else:
        replacement = random.randrange(bucket["unique_states_seen"])
        if replacement >= capacity:
            return False
        bucket["retained_hashes"].remove(states[replacement]["graph_hash"])
    row = {
        "graph_hash": graph_hash,
        "gate_count": int(graph.gate_count),
        "qasm": graph.to_qasm_str(),
    }
    if replacement == len(states):
        states.append(row)
    else:
        states[replacement] = row
    bucket["retained_hashes"].add(graph_hash)
    return True


def replay_pool_metrics(replay_pool: dict[str, dict]) -> dict:
    return {
        circuit: {
            "retained_states": len(bucket["states"]),
            "unique_states_seen": bucket["unique_states_seen"],
            "min_gate_count": min(row["gate_count"] for row in bucket["states"]),
            "max_gate_count": max(row["gate_count"] for row in bucket["states"]),
        }
        for circuit, bucket in replay_pool.items()
    }


def initialize_episode(
    qasm: Path,
    *,
    initial_qasm_override: str | None,
    context,
    quartz,
    model,
    device: torch.device,
    max_steps: int,
    page_size: int,
) -> tuple[
    BeamState,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PagedKVCache,
    list,
    str,
]:
    graph = (
        quartz.PyGraph.from_qasm_str(
            context=context, qasm_str=initial_qasm_override
        )
        if initial_qasm_override is not None
        else quartz.PyGraph.from_qasm(context=context, filename=str(qasm))
    )
    initial_qasm = graph.to_qasm_str()
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    initial_snapshot = snapshot(graph, guid_to_slot)
    state = BeamState(
        graph=None,
        snapshot=initial_snapshot,
        guid_to_slot={},
        next_slot=next_slot,
        last_touched={},
        rewrite_distance={int(row[0]): 5 for row in initial_snapshot["nodes"]},
        previous_preferred=set(),
        local_streak=0,
        gate_count=int(graph.gate_count),
        depth=0,
        history=(),
        topology_index=indexed_topology(initial_snapshot),
        exact_graph_checkpoint=graph,
        exact_slot_checkpoint=dict(guid_to_slot),
        exact_checkpoint_depth=0,
    )
    with torch.no_grad(), autocast_context(device):
        states, live, gate_types = model.initialize_incremental(
            move_batch(initial_batch(initial_snapshot), device)
        )
    cache_pages = math.ceil(max_steps / page_size) + 4
    arena = PagedKVCache(
        layers=model.action_layers_count,
        capacity=cache_pages,
        page_size=page_size,
        heads=model.action_heads,
        head_width=model.width // model.action_heads,
        model_width=model.width,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else states.dtype,
        gather_backend="vectorized",
    )
    return state, states, live, gate_types, arena, [arena.empty_handle()], initial_qasm


@torch.no_grad()
def proposal_features(
    model,
    encoded: torch.Tensor,
    live: torch.Tensor,
    state: BeamState,
    proposals: list[Proposal],
    rules: RuleMetadata,
    device: torch.device,
    initial_gate_bias: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(proposals)
    xfer_ids = torch.tensor(
        [proposal.xfer_id for proposal in proposals], device=device
    )
    source_ids = torch.tensor(
        [rules.xfer_to_source[proposal.xfer_id] for proposal in proposals],
        device=device,
    )
    bindings = torch.full(
        (count, model.max_pattern), -1, dtype=torch.long, device=device
    )
    for index, proposal in enumerate(proposals):
        binding = proposal.binding or ()
        bindings[index, : len(binding)] = torch.tensor(binding, device=device)
    base = model.candidate_features(
        encoded,
        live,
        xfer_ids,
        source_ids,
        bindings,
        torch.zeros(count, dtype=torch.long, device=device),
    )
    probabilities = torch.tensor(
        [proposal.probability for proposal in proposals], device=device
    )
    gate_deltas = torch.tensor(
        [proposal.next_gate_count - state.gate_count for proposal in proposals],
        device=device,
    )
    return build_policy_features(
        base,
        probabilities,
        gate_deltas,
        initial_gate_bias=initial_gate_bias,
    )


def finalize_episode(transitions: list[PPOTransition], gamma: float, gae_lambda: float) -> None:
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


def collect_episode(
    qasm: Path,
    *,
    context,
    quartz,
    xfers,
    rules: RuleMetadata,
    source_patterns,
    destination_patterns,
    rule_index: GpuRuleIndex,
    model,
    actor_critic: PagedPPOActorCritic,
    device: torch.device,
    threshold_config: dict,
    source_vectors: torch.Tensor,
    max_steps: int,
    max_source_matches: int,
    max_actions: int,
    invalid_reward: float,
    cycle_reward: float,
    step_penalty: float,
    max_rejected_actions_per_step: int,
    terminate_on_improvement: bool,
    start_from_best: bool,
    use_replay_starts: bool,
    replay_start_probability: float,
    replay_capacity_per_circuit: int,
    gamma: float,
    gae_lambda: float,
    page_size: int,
    initial_gate_bias: float,
    greedy: bool,
    best_by_circuit: dict[str, dict],
    replay_pool: dict[str, dict],
) -> tuple[list[PPOTransition], EpisodeMetrics]:
    start_qasm = best_by_circuit[qasm.name]["qasm"] if start_from_best else None
    started_from_replay = False
    if (
        use_replay_starts
        and len(replay_pool[qasm.name]["states"]) > 1
        and random.random() < replay_start_probability
    ):
        start_qasm = random.choice(replay_pool[qasm.name]["states"])["qasm"]
        started_from_replay = True
    (
        state,
        states,
        live,
        gate_types,
        arena,
        handles,
        initial_qasm,
    ) = initialize_episode(
        qasm,
        initial_qasm_override=start_qasm,
        context=context,
        quartz=quartz,
        model=model,
        device=device,
        max_steps=max_steps,
        page_size=page_size,
    )
    initial_gate_count = state.gate_count
    best_gate_count = state.gate_count
    transitions: list[PPOTransition] = []
    entropies = []
    candidate_counts = []
    legal_actions = invalid_actions = cycle_actions = accepted_rewrites = 0
    terminated_reason = "horizon"
    exact_hashes = {int(state.exact_graph_checkpoint.hash())}
    stop_episode = False

    for step in range(max_steps):
        candidates, _, _, encoded = paged_model_matches(
            [state],
            states,
            live,
            gate_types,
            handles,
            arena,
            model,
            device,
            threshold_config,
            source_vectors,
            microbatch=1,
            max_candidates=max_source_matches,
            state_batch_backend="legacy",
            candidate_backend="gpu",
            return_encoded_states=True,
        )
        proposals, _, _ = build_gpu_proposals(
            candidates,
            [state],
            rule_index,
            per_parent_cap=max_actions,
            global_cap=max_actions,
            ranking_mode="gate",
        )
        if not proposals:
            terminated_reason = "no_candidates"
            break

        policy_features, matcher_logits = proposal_features(
            model,
            encoded,
            live,
            state,
            proposals,
            rules,
            device,
            initial_gate_bias,
        )
        state_features = build_state_features(
            encoded, live, torch.tensor([state.gate_count], device=device)
        )
        candidate_mask = torch.ones(
            (1, len(proposals)), dtype=torch.bool, device=device
        )
        accepted = False
        for rejected_count in range(max_rejected_actions_per_step + 1):
            with torch.no_grad():
                distribution = masked_policy_distribution(
                    actor_critic,
                    policy_features.unsqueeze(0),
                    matcher_logits.unsqueeze(0),
                    candidate_mask,
                )
                action = (
                    distribution.logits.argmax(-1)
                    if greedy
                    else distribution.sample()
                )
                action_index = int(action.item())
                old_log_prob = float(distribution.log_prob(action).item())
                old_value = float(actor_critic.state_values(state_features).item())
                entropies.append(float(distribution.entropy().item()))
            proposal = proposals[action_index]
            candidate_counts.append(int(candidate_mask.sum().item()))

            child, _, duplicate = lazy_child(
                state,
                proposal,
                source_patterns,
                destination_patterns,
                structural_recheck=False,
                dedup_mode="none",
                seen=set(),
                topology_backend="legacy",
            )
            if duplicate:
                raise RuntimeError("deduplication is disabled for PPO")
            exact_graph = exact_slots = None
            topology_ok = False
            if child is not None:
                exact_graph, _, topology_ok, exact_slots = replay_state(
                    child,
                    context,
                    quartz.PyGraph,
                    xfers,
                    initial_qasm,
                    return_checkpoint=True,
                )
            legal = exact_graph is not None and topology_ok
            repeated_state = legal and int(exact_graph.hash()) in exact_hashes
            rejected = not legal or repeated_state
            can_retry = (
                rejected
                and rejected_count < max_rejected_actions_per_step
                and int(candidate_mask.sum().item()) > 1
            )
            improved_initial = legal and child.gate_count < initial_gate_count
            done = (not can_retry and rejected) or (
                not rejected
                and (
                    step + 1 == max_steps
                    or (terminate_on_improvement and improved_initial)
                )
            )
            reward = invalid_reward
            if legal:
                reward = shaped_transition_reward(
                    state.gate_count,
                    child.gate_count,
                    repeated_state=repeated_state,
                    step_penalty=step_penalty,
                    cycle_reward=cycle_reward,
                )
            transitions.append(
                PPOTransition(
                    state_features=state_features[0].float().cpu(),
                    candidate_features=policy_features.float().cpu(),
                    matcher_logits=matcher_logits.float().cpu(),
                    candidate_mask=candidate_mask[0].cpu().clone(),
                    action_index=action_index,
                    old_log_prob=old_log_prob,
                    old_value=old_value,
                    reward=reward,
                    done=done,
                    legal=legal,
                    xfer_id=proposal.xfer_id,
                    matcher_probability=proposal.probability,
                )
            )
            if legal:
                legal_actions += 1
            else:
                invalid_actions += 1
            if repeated_state:
                cycle_actions += 1
            if rejected:
                candidate_mask[0, action_index] = False
                if can_retry:
                    continue
                terminated_reason = "rejected_actions_exhausted"
                stop_episode = True
                break

            accepted = True
            accepted_rewrites += 1
            child.exact_graph_checkpoint = exact_graph
            child.exact_slot_checkpoint = exact_slots
            child.exact_checkpoint_depth = len(child.history)
            exact_hashes.add(int(exact_graph.hash()))
            retain_replay_state(
                replay_pool[qasm.name],
                exact_graph,
                capacity=replay_capacity_per_circuit,
            )
            best_gate_count = min(best_gate_count, child.gate_count)
            circuit_best = best_by_circuit[qasm.name]
            if child.gate_count < circuit_best["gate_count"]:
                circuit_best["gate_count"] = child.gate_count
                circuit_best["qasm"] = exact_graph.to_qasm_str()
                circuit_best["episode_depth"] = step + 1
            if done:
                if improved_initial:
                    terminated_reason = "improvement"
                state = child
                stop_episode = True
                break

            states, live, gate_types, handles, _, _ = advance_selected(
                states,
                live,
                gate_types,
                handles,
                [(child, proposal)],
                rules,
                arena,
                model,
                device,
                microbatch=1,
            )
            state = child
            break
        if stop_episode:
            break
        if not accepted:
            raise RuntimeError("PPO action retry loop exited without a successor")

    for handle in handles:
        arena.release(handle)
    finalize_episode(transitions, gamma, gae_lambda)
    total_reward = sum(row.reward for row in transitions)
    final_gate_count = state.gate_count
    return transitions, EpisodeMetrics(
        circuit=qasm.name,
        steps=len(transitions),
        legal_actions=legal_actions,
        invalid_actions=invalid_actions,
        cycle_actions=cycle_actions,
        accepted_rewrites=accepted_rewrites,
        total_reward=total_reward,
        initial_gate_count=initial_gate_count,
        final_gate_count=final_gate_count,
        best_gate_count=best_gate_count,
        mean_candidates=sum(candidate_counts) / max(1, len(candidate_counts)),
        mean_entropy=sum(entropies) / max(1, len(entropies)),
        started_from_replay=started_from_replay,
        terminated_reason=terminated_reason,
    )


def collate_transitions(
    transitions: list[PPOTransition], indices: torch.Tensor, device: torch.device
) -> dict:
    rows = [transitions[int(index)] for index in indices]
    max_candidates = max(row.candidate_features.shape[0] for row in rows)
    policy_width = rows[0].candidate_features.shape[1]
    candidate_features = torch.zeros(
        (len(rows), max_candidates, policy_width), dtype=torch.float
    )
    matcher_logits = torch.zeros((len(rows), max_candidates), dtype=torch.float)
    candidate_mask = torch.zeros((len(rows), max_candidates), dtype=torch.bool)
    for index, row in enumerate(rows):
        count = row.candidate_features.shape[0]
        candidate_features[index, :count] = row.candidate_features
        matcher_logits[index, :count] = row.matcher_logits
        candidate_mask[index, :count] = row.candidate_mask
    return {
        "state_features": torch.stack([row.state_features for row in rows]).to(device),
        "candidate_features": candidate_features.to(device),
        "matcher_logits": matcher_logits.to(device),
        "candidate_mask": candidate_mask.to(device),
        "actions": torch.tensor(
            [row.action_index for row in rows], device=device
        ),
        "old_log_probs": torch.tensor(
            [row.old_log_prob for row in rows], device=device
        ),
        "old_values": torch.tensor(
            [row.old_value for row in rows], device=device
        ),
        "advantages": torch.tensor(
            [row.advantage for row in rows], device=device
        ),
        "returns": torch.tensor(
            [row.return_value for row in rows], device=device
        ),
    }


def ppo_update(
    actor_critic: PagedPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[PPOTransition],
    *,
    device: torch.device,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_grad_norm: float,
    seed: int,
) -> dict:
    advantages = torch.tensor([row.advantage for row in transitions])
    normalized = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-6)
    for row, advantage in zip(transitions, normalized.tolist()):
        row.advantage = float(advantage)

    generator = torch.Generator().manual_seed(seed)
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    actor_critic.train()
    for _ in range(epochs):
        order = torch.randperm(len(transitions), generator=generator)
        for begin in range(0, len(transitions), minibatch_size):
            batch = collate_transitions(
                transitions, order[begin : begin + minibatch_size], device
            )
            distribution = masked_policy_distribution(
                actor_critic,
                batch["candidate_features"],
                batch["matcher_logits"],
                batch["candidate_mask"],
            )
            new_log_probs = distribution.log_prob(batch["actions"])
            new_values = actor_critic.state_values(batch["state_features"])
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
    return {key: value / max(1, batches) for key, value in totals.items()} | {
        "batches": batches
    }


def aggregate_episodes(rows: list[EpisodeMetrics]) -> dict:
    transitions = sum(row.steps for row in rows)
    legal = sum(row.legal_actions for row in rows)
    invalid = sum(row.invalid_actions for row in rows)
    cycles = sum(row.cycle_actions for row in rows)
    accepted_rewrites = sum(row.accepted_rewrites for row in rows)
    return {
        "episodes": len(rows),
        "transitions": transitions,
        "legal_actions": legal,
        "invalid_actions": invalid,
        "cycle_actions": cycles,
        "accepted_rewrites": accepted_rewrites,
        "selected_action_legality": legal / max(1, legal + invalid),
        "selected_action_rejection_rate": (cycles + invalid)
        / max(1, transitions),
        "repeated_state_rate": cycles / max(1, legal),
        "mean_episode_reward": sum(row.total_reward for row in rows)
        / max(1, len(rows)),
        "mean_episode_steps": transitions / max(1, len(rows)),
        "mean_accepted_rewrites": accepted_rewrites / max(1, len(rows)),
        "mean_candidates": sum(row.mean_candidates for row in rows)
        / max(1, len(rows)),
        "mean_policy_entropy": sum(row.mean_entropy for row in rows)
        / max(1, len(rows)),
        "replay_start_episodes": sum(row.started_from_replay for row in rows),
        "best_gate_count_by_circuit": {
            circuit: min(row.best_gate_count for row in rows if row.circuit == circuit)
            for circuit in sorted({row.circuit for row in rows})
        },
        "termination_reasons": {
            reason: sum(row.terminated_reason == reason for row in rows)
            for reason in sorted({row.terminated_reason for row in rows})
        },
    }


def evaluate_policy(
    qasms: list[Path],
    episodes_per_circuit: int,
    **episode_kwargs,
) -> dict:
    episode_kwargs["actor_critic"].eval()
    episode_kwargs["use_replay_starts"] = False
    rows = []
    for qasm in qasms:
        for _ in range(episodes_per_circuit):
            _, metrics = collect_episode(qasm, greedy=True, **episode_kwargs)
            rows.append(metrics)
    return {
        **aggregate_episodes(rows),
        "episodes_detail": [asdict(row) for row in rows],
    }


def serialized_args(args: argparse.Namespace) -> dict:
    return {
        key: [str(item) for item in value]
        if isinstance(value, list) and value and isinstance(value[0], Path)
        else str(value)
        if isinstance(value, Path)
        else value
        for key, value in vars(args).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes-per-iteration", type=int, default=32)
    parser.add_argument("--evaluation-episodes-per-circuit", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--max-source-matches", type=int, default=2048)
    parser.add_argument("--max-actions", type=int, default=128)
    parser.add_argument("--max-gate-increase", type=int, default=3)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--initial-gate-bias", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--invalid-reward", type=float, default=-1.0)
    parser.add_argument("--cycle-reward", type=float, default=-1.0)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--max-rejected-actions-per-step", type=int, default=8)
    parser.add_argument("--replay-start-probability", type=float, default=0.5)
    parser.add_argument("--replay-capacity-per-circuit", type=int, default=256)
    parser.add_argument(
        "--continue-after-improvement",
        dest="terminate_on_improvement",
        action="store_false",
        help="continue to the horizon after beating the episode's input circuit",
    )
    parser.set_defaults(terminate_on_improvement=True)
    parser.add_argument(
        "--no-best-root-curriculum",
        dest="start_from_best",
        action="store_false",
        help="always start episodes from the original input QASM",
    )
    parser.set_defaults(start_from_best=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if args.iterations < 1 or args.episodes_per_iteration < 1:
        parser.error("iterations and episodes per iteration must be positive")
    if args.evaluation_episodes_per_circuit < 0:
        parser.error("evaluation episodes per circuit must be nonnegative")
    if args.max_steps < 1 or args.max_actions < 1:
        parser.error("max steps and max actions must be positive")
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        parser.error("gamma and GAE lambda must be within [0, 1]")
    if args.step_penalty < 0:
        parser.error("step penalty must be nonnegative")
    if args.max_rejected_actions_per_step < 0:
        parser.error("maximum rejected actions per step must be nonnegative")
    if not 0 <= args.replay_start_probability <= 1:
        parser.error("replay start probability must be within [0, 1]")
    if args.replay_capacity_per_circuit < 1:
        parser.error("replay capacity per circuit must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]
    if model_args.get("architecture") != "paged_action":
        raise ValueError("PPO requires a paged_action checkpoint")
    model = build_model(rules, len(rules.xfer_to_source), model_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.readout_attention_backend = "paged"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    actor_critic = PagedPPOActorCritic(
        model.width, hidden_size=args.hidden_size
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(actor_critic.policy_norm.parameters())
                + list(actor_critic.policy.parameters()),
                "lr": args.actor_learning_rate,
            },
            {
                "params": list(actor_critic.value_norm.parameters())
                + list(actor_critic.value.parameters()),
                "lr": args.critic_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )
    start_iteration = 0
    resume_best_by_circuit = None
    resume_replay_pool = None
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("format") != "paged-ppo-v1":
            raise ValueError("unsupported PPO checkpoint")
        if int(resume["width"]) != model.width:
            raise ValueError("resume PPO width differs from the base model")
        if int(resume["hidden_size"]) != actor_critic.hidden_size:
            raise ValueError("resume PPO hidden size differs from --hidden-size")
        if Path(resume["base_checkpoint"]).name != args.checkpoint.name:
            raise ValueError("resume PPO was trained on a different base model")
        actor_critic.load_state_dict(resume["actor_critic"])
        optimizer.load_state_dict(resume["optimizer"])
        start_iteration = int(resume["iteration"]) + 1
        resume_best_by_circuit = resume.get("best_by_circuit")
        resume_replay_pool = resume.get("replay_pool")

    threshold_config = load_threshold_config(
        args.calibration, args.target_recall
    )
    with torch.no_grad(), autocast_context(device):
        source_vectors = model.retrieval_source(model.source_representations())
    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    if context.num_xfers != len(rules.xfer_to_source):
        raise RuntimeError("dataset and Quartz context have different xfer counts")
    xfers = [
        context.get_xfer_from_id(id=index) for index in range(context.num_xfers)
    ]
    source_to_xfers: dict[int, list[int]] = defaultdict(list)
    for xfer_id, source_id in enumerate(rules.xfer_to_source):
        source_to_xfers[source_id].append(xfer_id)
    gate_deltas = [
        len(rules.destination_gate_types[index])
        - len(rules.source_gate_types[rules.xfer_to_source[index]])
        for index in range(len(rules.xfer_to_source))
    ]
    rule_index = GpuRuleIndex.build(
        source_to_xfers,
        gate_deltas,
        len(rules.source_gate_types),
        args.max_gate_increase,
        device,
    )
    source_patterns = tuple(parse_pattern(row) for row in rules.xfer_sources)
    destination_patterns = tuple(
        parse_pattern(row) for row in rules.xfer_destinations
    )

    best_by_circuit = {}
    replay_pool = {}
    for qasm in args.qasm:
        graph = quartz.PyGraph.from_qasm(context=context, filename=str(qasm))
        best_by_circuit[qasm.name] = {
            "gate_count": int(graph.gate_count),
            "qasm": graph.to_qasm_str(),
            "episode_depth": 0,
        }
        replay_pool[qasm.name] = make_replay_bucket(graph)
    if resume_best_by_circuit is not None:
        for circuit, best in resume_best_by_circuit.items():
            if circuit in best_by_circuit:
                best_by_circuit[circuit] = best
    if resume_replay_pool is not None:
        for circuit, bucket in resume_replay_pool.items():
            if circuit in replay_pool:
                replay_pool[circuit] = bucket
    training_log = {
        "format": "paged-ppo-training-v1",
        "args": serialized_args(args),
        "base_checkpoint": str(args.checkpoint),
        "frozen_base_model": True,
        "iterations": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    episode_kwargs = {
        "context": context,
        "quartz": quartz,
        "xfers": xfers,
        "rules": rules,
        "source_patterns": source_patterns,
        "destination_patterns": destination_patterns,
        "rule_index": rule_index,
        "model": model,
        "actor_critic": actor_critic,
        "device": device,
        "threshold_config": threshold_config,
        "source_vectors": source_vectors,
        "max_steps": args.max_steps,
        "max_source_matches": args.max_source_matches,
        "max_actions": args.max_actions,
        "invalid_reward": args.invalid_reward,
        "cycle_reward": args.cycle_reward,
        "step_penalty": args.step_penalty,
        "max_rejected_actions_per_step": args.max_rejected_actions_per_step,
        "terminate_on_improvement": args.terminate_on_improvement,
        "start_from_best": args.start_from_best,
        "use_replay_starts": True,
        "replay_start_probability": args.replay_start_probability,
        "replay_capacity_per_circuit": args.replay_capacity_per_circuit,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "page_size": args.page_size,
        "initial_gate_bias": args.initial_gate_bias,
        "best_by_circuit": best_by_circuit,
        "replay_pool": replay_pool,
    }
    if args.evaluation_episodes_per_circuit:
        training_log["initial_evaluation"] = evaluate_policy(
            args.qasm,
            args.evaluation_episodes_per_circuit,
            **episode_kwargs,
        )

    for iteration in range(start_iteration, start_iteration + args.iterations):
        collection_started = time.perf_counter()
        transitions = []
        episode_rows = []
        for episode in range(args.episodes_per_iteration):
            qasm = args.qasm[episode % len(args.qasm)]
            episode_transitions, metrics = collect_episode(
                qasm,
                greedy=False,
                **episode_kwargs,
            )
            transitions.extend(episode_transitions)
            episode_rows.append(metrics)
        collection_seconds = time.perf_counter() - collection_started
        if not transitions:
            raise RuntimeError("PPO collection produced no transitions")

        update_started = time.perf_counter()
        update_metrics = ppo_update(
            actor_critic,
            optimizer,
            transitions,
            device=device,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            clip_epsilon=args.clip_epsilon,
            value_coefficient=args.value_coefficient,
            entropy_coefficient=args.entropy_coefficient,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed + iteration,
        )
        update_seconds = time.perf_counter() - update_started
        collection_metrics = aggregate_episodes(episode_rows)
        evaluation_metrics = (
            evaluate_policy(
                args.qasm,
                args.evaluation_episodes_per_circuit,
                **episode_kwargs,
            )
            if args.evaluation_episodes_per_circuit
            else None
        )
        row = {
            "iteration": iteration,
            "collection_seconds": collection_seconds,
            "collection_transitions_per_second": len(transitions)
            / collection_seconds,
            "update_seconds": update_seconds,
            "update_samples_per_second": (
                len(transitions) * args.ppo_epochs / update_seconds
            ),
            "collection": collection_metrics,
            "update": update_metrics,
            "evaluation": evaluation_metrics,
            "replay_pool": replay_pool_metrics(replay_pool),
            "episodes": [asdict(metrics) for metrics in episode_rows],
            "best_so_far": {
                key: {name: value for name, value in row.items() if name != "qasm"}
                for key, row in best_by_circuit.items()
            },
        }
        training_log["iterations"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        torch.save(
            {
                "format": "paged-ppo-v1",
                "iteration": iteration,
                "actor_critic": actor_critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "width": model.width,
                "hidden_size": actor_critic.hidden_size,
                "base_checkpoint": str(args.checkpoint),
                "best_by_circuit": best_by_circuit,
                "replay_pool": replay_pool,
                "args": serialized_args(args),
            },
            args.output,
        )
        args.output.with_suffix(".training.json").write_text(
            json.dumps(training_log, indent=2, sort_keys=True) + "\n"
        )
        for circuit, best in best_by_circuit.items():
            args.output.with_name(
                f"{args.output.stem}_{Path(circuit).stem}_best.qasm"
            ).write_text(best["qasm"])

    print(
        json.dumps(
            {
                "output": str(args.output),
                "iterations": args.iterations,
                "best_gate_count_by_circuit": {
                    key: row["gate_count"] for key, row in best_by_circuit.items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
