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
from gpu_proposals import (
    GpuRuleIndex,
    SelectedProposalTensors,
    build_gpu_proposals,
    materialize_selected_proposals,
)
from incremental_graph import parse_pattern
from lazy_rollout_benchmark import (
    ExactReplayCacheEntry,
    indexed_topology,
    lazy_child,
    replay_state,
    shared_replay_prefixes,
)
from model_factory import build_model
from paged_cache import PagedKVCache
from paged_rollout_benchmark import (
    advance_selected,
    initial_batch,
    paged_model_matches,
)
from ppo_core import (
    build_actor_critic,
    build_policy_features,
    build_state_features,
    categorical_reference_kl,
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
    prefix_state: torch.Tensor | None = None
    history_depth: int = 0
    committed_action: bool = False
    repeated_state: bool = False
    candidate_count: int = 0
    policy_entropy: float = 0.0
    previous_gate_count: int = 0
    next_gate_count: int = 0
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
    exact_refreshes: int = 0
    exact_replay_actions: int = 0
    exact_refresh_seconds: float = 0.0


@dataclass
class EpisodeRuntime:
    circuit: str
    state: BeamState
    initial_qasm: str
    initial_gate_count: int
    started_from_replay: bool
    transitions: list[PPOTransition]
    pending_transition_indices: list[int]
    exact_hashes: set[int]
    topology_hashes: set[int]
    terminated_reason: str = "horizon"
    stopped: bool = False
    final_gate_count: int | None = None
    exact_refreshes: int = 0
    exact_replay_actions: int = 0
    exact_refresh_seconds: float = 0.0


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
    graph_hash: int | None = None,
) -> bool:
    graph_hash = int(graph.hash()) if graph_hash is None else int(graph_hash)
    if graph_hash in bucket["retained_hashes"]:
        return False
    bucket["unique_states_seen"] += 1
    states = bucket["states"]
    if len(states) < capacity:
        replacement = len(states)
    else:
        minimum_index = min(
            range(len(states)), key=lambda index: states[index]["gate_count"]
        )
        minimum_gate_count = states[minimum_index]["gate_count"]
        if int(graph.gate_count) < minimum_gate_count:
            replacement = max(
                range(len(states)), key=lambda index: states[index]["gate_count"]
            )
        else:
            replacement = random.randrange(bucket["unique_states_seen"])
            if replacement >= capacity:
                return False
            if replacement == minimum_index:
                return False
        bucket["retained_hashes"].remove(states[replacement]["graph_hash"])
    row = {
        "graph_hash": graph_hash,
        "gate_count": int(graph.gate_count),
        "qasm": None,
        "graph": graph,
    }
    if replacement == len(states):
        states.append(row)
    else:
        states[replacement] = row
    bucket["retained_hashes"].add(graph_hash)
    return True


def replay_state_qasm(row: dict) -> str:
    qasm = row.get("qasm")
    if qasm is None:
        qasm = row["graph"].to_qasm_str()
        row["qasm"] = qasm
    return qasm


def serializable_replay_pool(replay_pool: dict[str, dict]) -> dict[str, dict]:
    return {
        circuit: {
            "states": [
                {
                    "graph_hash": row["graph_hash"],
                    "gate_count": row["gate_count"],
                    "qasm": replay_state_qasm(row),
                }
                for row in bucket["states"]
            ],
            "retained_hashes": bucket["retained_hashes"],
            "unique_states_seen": bucket["unique_states_seen"],
        }
        for circuit, bucket in replay_pool.items()
    }


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


def initial_batch_many(snapshot_rows: list[dict]) -> dict:
    """Pack initial graphs without materializing a current-graph batch."""
    rows = [initial_batch(snapshot_row) for snapshot_row in snapshot_rows]
    max_slots = max(row["initial_types"].shape[1] for row in rows)
    initial_types = torch.cat(
        [
            F.pad(
                row["initial_types"],
                (0, max_slots - row["initial_types"].shape[1]),
                value=-1,
            )
            for row in rows
        ]
    )
    edge_batches = []
    edge_sources = []
    edge_destinations = []
    edge_relations = []
    for batch_index, row in enumerate(rows):
        edge_batches.append(
            torch.full_like(row["edge_batch"], batch_index)
        )
        edge_sources.append(row["edge_src"])
        edge_destinations.append(row["edge_dst"])
        edge_relations.append(row["edge_relation"])
    return {
        "initial_types": initial_types,
        "edge_batch": torch.cat(edge_batches),
        "edge_src": torch.cat(edge_sources),
        "edge_dst": torch.cat(edge_destinations),
        "edge_relation": torch.cat(edge_relations),
    }


def group_episode_starts(
    starts: list[str | None], backend: str
) -> tuple[list[str | None], list[int]]:
    if backend == "duplicated":
        return starts, list(range(len(starts)))
    if backend != "deduplicated":
        raise ValueError(f"unknown episode initialization backend: {backend}")
    unique_starts: list[str | None] = []
    template_by_start: dict[str | None, int] = {}
    template_indices = []
    for start in starts:
        template_index = template_by_start.get(start)
        if template_index is None:
            template_index = len(unique_starts)
            unique_starts.append(start)
            template_by_start[start] = template_index
        template_indices.append(template_index)
    return unique_starts, template_indices


def initialize_episode_batch(
    qasm: Path,
    batch_size: int,
    *,
    context,
    quartz,
    model,
    device: torch.device,
    max_steps: int,
    page_size: int,
    initialization_backend: str,
    start_from_best: bool,
    use_replay_starts: bool,
    replay_start_probability: float,
    best_by_circuit: dict[str, dict],
    replay_pool: dict[str, dict],
    best_start_probability: float = 1.0,
) -> tuple[
    list[EpisodeRuntime],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PagedKVCache,
    list,
]:
    if not 0.0 <= best_start_probability <= 1.0:
        raise ValueError("best start probability must be in [0, 1]")
    episode_starts = []
    for _ in range(batch_size):
        use_best = start_from_best and (
            best_start_probability == 1.0
            or random.random() < best_start_probability
        )
        start_qasm = best_by_circuit[qasm.name]["qasm"] if use_best else None
        started_from_replay = False
        if (
            use_replay_starts
            and len(replay_pool[qasm.name]["states"]) > 1
            and random.random() < replay_start_probability
        ):
            start_qasm = replay_state_qasm(
                random.choice(replay_pool[qasm.name]["states"])
            )
            started_from_replay = True
        episode_starts.append((start_qasm, started_from_replay))

    template_starts, template_indices = group_episode_starts(
        [start for start, _ in episode_starts], initialization_backend
    )
    templates = []
    snapshots = []
    for start_qasm in template_starts:
        graph = (
            quartz.PyGraph.from_qasm_str(context=context, qasm_str=start_qasm)
            if start_qasm is not None
            else quartz.PyGraph.from_qasm(context=context, filename=str(qasm))
        )
        initial_qasm = graph.to_qasm_str()
        guid_to_slot: dict[int, int] = {}
        next_slot = update_slots(graph, guid_to_slot, 0)
        initial_snapshot = snapshot(graph, guid_to_slot)
        topology = indexed_topology(initial_snapshot)
        templates.append(
            {
                "graph": graph,
                "initial_qasm": initial_qasm,
                "guid_to_slot": guid_to_slot,
                "next_slot": next_slot,
                "snapshot": initial_snapshot,
                "topology": topology,
            }
        )
        snapshots.append(initial_snapshot)

    runtimes = []
    for template_index, (_, started_from_replay) in zip(
        template_indices, episode_starts
    ):
        template = templates[template_index]
        graph = template["graph"]
        initial_qasm = template["initial_qasm"]
        guid_to_slot = template["guid_to_slot"]
        next_slot = template["next_slot"]
        initial_snapshot = template["snapshot"]
        topology = template["topology"]
        state = BeamState(
            graph=None,
            snapshot=initial_snapshot,
            guid_to_slot={},
            next_slot=next_slot,
            last_touched={},
            rewrite_distance={
                int(row[0]): 5 for row in initial_snapshot["nodes"]
            },
            previous_preferred=set(),
            local_streak=0,
            gate_count=int(graph.gate_count),
            depth=0,
            history=(),
            topology_index=topology,
            exact_graph_checkpoint=graph,
            exact_slot_checkpoint=dict(guid_to_slot),
            exact_checkpoint_depth=0,
        )
        runtimes.append(
            EpisodeRuntime(
                circuit=qasm.name,
                state=state,
                initial_qasm=initial_qasm,
                initial_gate_count=state.gate_count,
                started_from_replay=started_from_replay,
                transitions=[],
                pending_transition_indices=[],
                exact_hashes={int(graph.hash())},
                topology_hashes={int(topology.fingerprint)},
            )
        )
    with torch.no_grad(), autocast_context(device):
        states, live, gate_types = model.initialize_incremental(
            move_batch(initial_batch_many(snapshots), device)
        )
        if template_indices != list(range(batch_size)):
            episode_indices = torch.tensor(template_indices, device=device)
            states = states.index_select(0, episode_indices)
            live = live.index_select(0, episode_indices)
            gate_types = gate_types.index_select(0, episode_indices)
    cache_pages = batch_size * (math.ceil(max_steps / page_size) + 2) + 4
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
    return (
        runtimes,
        states,
        live,
        gate_types,
        arena,
        [arena.empty_handle() for _ in runtimes],
    )


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
    ordered_roles: bool = False,
    source_representations: torch.Tensor | None = None,
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
        ordered_roles=ordered_roles,
        source_representations=source_representations,
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


@torch.no_grad()
def batched_proposal_features(
    model,
    encoded: torch.Tensor,
    live: torch.Tensor,
    states: list[BeamState],
    proposals: list[Proposal] | None,
    rules: RuleMetadata,
    device: torch.device,
    initial_gate_bias: float,
    ordered_roles: bool = False,
    proposal_tensors: SelectedProposalTensors | None = None,
    source_representations: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if proposal_tensors is not None:
        count = proposal_tensors.parent_ids.numel()
        if proposals is not None and len(proposals) != count:
            raise ValueError("selected proposal tensor count differs from proposals")
        parent_ids = proposal_tensors.parent_ids
        xfer_ids = proposal_tensors.xfer_ids
        source_ids = proposal_tensors.source_ids
        bindings = proposal_tensors.bindings
        probabilities = proposal_tensors.probabilities
        gate_deltas = proposal_tensors.gate_deltas
    else:
        if proposals is None:
            raise ValueError("proposal features require Python or GPU proposals")
        count = len(proposals)
        parent_ids = torch.tensor(
            [proposal.parent for proposal in proposals], device=device
        )
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
        probabilities = torch.tensor(
            [proposal.probability for proposal in proposals], device=device
        )
        gate_deltas = torch.tensor(
            [
                proposal.next_gate_count
                - states[proposal.parent].gate_count
                for proposal in proposals
            ],
            device=device,
        )
    base = model.candidate_features(
        encoded,
        live,
        xfer_ids,
        source_ids,
        bindings,
        parent_ids,
        ordered_roles=ordered_roles,
        source_representations=source_representations,
    )
    features, logits = build_policy_features(
        base,
        probabilities,
        gate_deltas,
        initial_gate_bias=initial_gate_bias,
    )
    return features, logits, parent_ids


def pad_batched_policy_inputs(
    features: torch.Tensor,
    logits: torch.Tensor,
    proposals: list[Proposal] | None,
    batch_size: int,
    *,
    parent_ids: torch.Tensor | None = None,
    backend: str = "loop",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[list[Proposal]],
    torch.Tensor,
]:
    if backend not in {"loop", "tensorized"}:
        raise ValueError(f"unknown policy padding backend: {backend}")
    if backend == "tensorized" and parent_ids is None:
        raise ValueError("tensorized policy padding requires GPU parent IDs")
    if backend == "loop" and proposals is None:
        raise ValueError("loop policy padding requires Python proposals")

    grouped: list[list[tuple[int, Proposal]]] = [[] for _ in range(batch_size)]
    if proposals is not None:
        for flat_index, proposal in enumerate(proposals):
            grouped[proposal.parent].append((flat_index, proposal))
    grouped_proposals = [
        [proposal for _, proposal in rows] for rows in grouped
    ]
    if backend == "tensorized":
        counts = torch.bincount(parent_ids, minlength=batch_size)
        max_candidates = int(counts.max().item())
        order = torch.argsort(parent_ids, stable=True)
        sorted_parents = parent_ids[order]
        positions = torch.arange(order.numel(), device=features.device)
        new_parent = torch.ones(
            order.numel(), dtype=torch.bool, device=features.device
        )
        new_parent[1:] = sorted_parents[1:] != sorted_parents[:-1]
        group_starts = torch.where(new_parent, positions, 0)
        group_starts = torch.cummax(group_starts, dim=0).values
        sorted_offsets = positions - group_starts
        offsets = torch.empty_like(sorted_offsets)
        offsets[order] = sorted_offsets
        padded_features = features.new_zeros(
            (batch_size, max_candidates, features.shape[-1])
        )
        padded_logits = logits.new_zeros((batch_size, max_candidates))
        mask = torch.zeros(
            (batch_size, max_candidates),
            dtype=torch.bool,
            device=features.device,
        )
        padded_features[parent_ids, offsets] = features
        padded_logits[parent_ids, offsets] = logits
        mask[parent_ids, offsets] = True
        flat_indices = torch.full(
            (batch_size, max_candidates),
            -1,
            dtype=torch.long,
            device=features.device,
        )
        flat_indices[parent_ids, offsets] = torch.arange(
            parent_ids.numel(), device=features.device
        )
        return (
            padded_features,
            padded_logits,
            mask,
            grouped_proposals,
            flat_indices,
        )

    max_candidates = max((len(rows) for rows in grouped), default=0)
    padded_features = features.new_zeros(
        (batch_size, max_candidates, features.shape[-1])
    )
    padded_logits = logits.new_zeros((batch_size, max_candidates))
    mask = torch.zeros(
        (batch_size, max_candidates), dtype=torch.bool, device=features.device
    )
    flat_indices = torch.full(
        (batch_size, max_candidates),
        -1,
        dtype=torch.long,
        device=features.device,
    )
    for parent, rows in enumerate(grouped):
        if not rows:
            continue
        indices = torch.tensor(
            [flat_index for flat_index, _ in rows], device=features.device
        )
        count = len(rows)
        padded_features[parent, :count] = features.index_select(0, indices)
        padded_logits[parent, :count] = logits.index_select(0, indices)
        mask[parent, :count] = True
        flat_indices[parent, :count] = indices
    return padded_features, padded_logits, mask, grouped_proposals, flat_indices


def current_prefix_states(
    encoded: torch.Tensor,
    live: torch.Tensor,
    arena: PagedKVCache,
    handles: list,
) -> torch.Tensor:
    """Use the final causal token, with graph pooling as the root BOS state."""
    last_actions, present = arena.last_actions(handles)
    live_float = live.unsqueeze(-1)
    root_states = (encoded * live_float).sum(1)
    root_states = root_states / live_float.sum(1).clamp_min(1)
    last_actions = last_actions.to(root_states.dtype)
    return torch.where(present.unsqueeze(-1), last_actions, root_states)


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
    actor_critic: torch.nn.Module,
    device: torch.device,
    threshold_config: dict,
    source_vectors: torch.Tensor,
    source_representations: torch.Tensor | None,
    max_steps: int,
    max_source_matches: int,
    source_microbatch: int,
    source_grouping: str,
    proposal_expansion: str,
    transition_transfer_backend: str,
    proposal_tensor_backend: str,
    proposal_materialization_backend: str,
    policy_padding_backend: str,
    episode_initialization_backend: str,
    advance_input_backend: str,
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
    collector_timing: dict[str, float] | None = None,
) -> tuple[list[PPOTransition], EpisodeMetrics]:
    start_qasm = best_by_circuit[qasm.name]["qasm"] if start_from_best else None
    started_from_replay = False
    if (
        use_replay_starts
        and len(replay_pool[qasm.name]["states"]) > 1
        and random.random() < replay_start_probability
    ):
        start_qasm = replay_state_qasm(
            random.choice(replay_pool[qasm.name]["states"])
        )
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
        candidates, model_match_seconds, _, encoded = paged_model_matches(
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
            source_microbatch=source_microbatch,
            source_grouping=source_grouping,
            state_batch_backend="legacy",
            candidate_backend="gpu",
            return_encoded_states=True,
        )
        if collector_timing is not None:
            collector_timing["model_match_seconds"] += model_match_seconds
        proposal_started = time.perf_counter()
        proposals, proposal_metrics, _, _ = build_gpu_proposals(
            candidates,
            [state],
            rule_index,
            per_parent_cap=max_actions,
            global_cap=max_actions,
            ranking_mode="gate",
            preselect_matches=proposal_expansion == "preselect",
        )
        if collector_timing is not None:
            collector_timing["proposal_seconds"] += (
                time.perf_counter() - proposal_started
            )
            eligible_actions = int(proposal_metrics["eligible_actions"])
            collector_timing["eligible_actions"] += eligible_actions
            collector_timing["materialized_actions"] += int(
                proposal_metrics.get("materialized_actions", eligible_actions)
            )
            selected_actions = int(proposal_metrics.get("selected_actions", 0))
            collector_timing["selected_proposals"] += selected_actions
            collector_timing["python_proposals_materialized"] += selected_actions
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
            ordered_roles=actor_critic.match_set_aware,
            source_representations=source_representations,
        )
        state_features = build_state_features(
            encoded, live, torch.tensor([state.gate_count], device=device)
        )
        prefix_states = current_prefix_states(
            encoded, live, arena, handles
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
                    prefix_states=prefix_states,
                    state_features=state_features,
                )
                action = (
                    distribution.logits.argmax(-1)
                    if greedy
                    else distribution.sample()
                )
                action_index = int(action.item())
                old_log_prob = float(distribution.log_prob(action).item())
                old_value = float(
                    actor_critic.state_values(
                        state_features,
                        policy_features.unsqueeze(0),
                        candidate_mask,
                        prefix_states,
                    ).item()
                )
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
                    prefix_state=prefix_states[0].float().cpu(),
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

            (
                states,
                live,
                gate_types,
                handles,
                advance_seconds,
                _,
            ) = advance_selected(
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
                trusted_paged_inputs=advance_input_backend == "trusted",
                source_representations=source_representations,
            )
            if collector_timing is not None:
                collector_timing["cache_advance_seconds"] += advance_seconds
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


def refresh_speculative_runtimes(
    runtimes: list[EpisodeRuntime],
    *,
    context,
    quartz,
    xfers,
    invalid_reward: float,
    cycle_reward: float,
    replay_capacity_per_circuit: int,
    best_by_circuit: dict[str, dict],
    replay_pool: dict[str, dict],
    profile_timing: dict[str, float] | None = None,
    profile_counts: dict[str, float] | None = None,
) -> None:
    """Validate pending action suffixes and advance exact checkpoints."""
    def add_seconds(name: str, started: float) -> None:
        if profile_timing is not None:
            profile_timing[name] = (
                profile_timing.get(name, 0.0) + time.perf_counter() - started
            )

    def add_count(name: str, amount: int = 1) -> None:
        if profile_counts is not None:
            profile_counts[name] = profile_counts.get(name, 0.0) + amount

    refresh_started = time.perf_counter()
    grouping_started = time.perf_counter()
    grouped: dict[tuple[str, int, int], list[EpisodeRuntime]] = defaultdict(list)
    for runtime in runtimes:
        if not runtime.pending_transition_indices:
            continue
        state = runtime.state
        checkpoint_hash = int(state.exact_graph_checkpoint.hash())
        grouped[
            (runtime.circuit, state.exact_checkpoint_depth, checkpoint_hash)
        ].append(runtime)
    add_seconds("refresh_grouping_seconds", grouping_started)
    add_count("refresh_runtime_count", sum(map(len, grouped.values())))
    add_count("refresh_group_count", len(grouped))

    for group in grouped.values():
        group_states = [runtime.state for runtime in group]
        prefix_started = time.perf_counter()
        replay_prefixes = shared_replay_prefixes(group_states)
        add_seconds("refresh_prefix_analysis_seconds", prefix_started)
        add_count("refresh_shared_prefix_count", len(replay_prefixes))
        replay_cache: dict[tuple, ExactReplayCacheEntry] = {}
        for runtime in group:
            state = runtime.state
            replay_counts: dict[str, int] = {}
            replay_timing: dict[str, float] = {}
            started = time.perf_counter()
            exact_graph, failure_step, topology_ok, exact_slots = replay_state(
                state,
                context,
                quartz.PyGraph,
                xfers,
                runtime.initial_qasm,
                return_checkpoint=True,
                profile_timing=replay_timing,
                profile_counts=replay_counts,
                replay_cache=replay_cache,
                replay_cache_prefixes=replay_prefixes,
            )
            replay_seconds = time.perf_counter() - started
            runtime.exact_refresh_seconds += replay_seconds
            runtime.exact_refreshes += 1
            runtime.exact_replay_actions += replay_counts.get(
                "actions_attempted", 0
            )
            if profile_timing is not None:
                profile_timing["refresh_replay_total_seconds"] = (
                    profile_timing.get("refresh_replay_total_seconds", 0.0)
                    + replay_seconds
                )
                for name, seconds in replay_timing.items():
                    key = f"refresh_replay_{name}"
                    profile_timing[key] = profile_timing.get(key, 0.0) + seconds
            if profile_counts is not None:
                for name, count in replay_counts.items():
                    key = f"refresh_replay_{name}"
                    profile_counts[key] = profile_counts.get(key, 0.0) + count

            result_started = time.perf_counter()
            if failure_step is not None or exact_graph is None:
                failed_depth = int(failure_step or len(state.history))
                failed_index = next(
                    (
                        index
                        for index in runtime.pending_transition_indices
                        if runtime.transitions[index].history_depth == failed_depth
                    ),
                    runtime.pending_transition_indices[-1],
                )
                failed = runtime.transitions[failed_index]
                failed.legal = False
                failed.committed_action = False
                failed.reward = float(invalid_reward)
                failed.done = True
                del runtime.transitions[failed_index + 1 :]
                runtime.pending_transition_indices.clear()
                runtime.final_gate_count = failed.previous_gate_count
                runtime.terminated_reason = "exact_refresh_failure"
                runtime.stopped = True
                add_seconds("refresh_result_processing_seconds", result_started)
                continue

            if not topology_ok:
                failed_index = runtime.pending_transition_indices[-1]
                failed = runtime.transitions[failed_index]
                failed.legal = False
                failed.committed_action = False
                failed.reward = float(invalid_reward)
                failed.done = True
                runtime.pending_transition_indices.clear()
                runtime.final_gate_count = failed.previous_gate_count
                runtime.terminated_reason = "exact_topology_mismatch"
                runtime.stopped = True
                add_seconds("refresh_result_processing_seconds", result_started)
                continue

            hash_started = time.perf_counter()
            graph_hash = int(exact_graph.hash())
            add_seconds("refresh_result_graph_hash_seconds", hash_started)
            if graph_hash in runtime.exact_hashes:
                cycle_index = runtime.pending_transition_indices[-1]
                cycle = runtime.transitions[cycle_index]
                cycle.repeated_state = True
                cycle.committed_action = False
                cycle.reward = float(cycle_reward)
                cycle.done = True
                runtime.pending_transition_indices.clear()
                runtime.final_gate_count = cycle.next_gate_count
                runtime.terminated_reason = "exact_cycle"
                runtime.stopped = True
                add_seconds("refresh_result_processing_seconds", result_started)
                continue

            state.exact_graph_checkpoint = exact_graph
            state.exact_slot_checkpoint = exact_slots
            state.exact_checkpoint_depth = len(state.history)
            runtime.exact_hashes.add(graph_hash)
            runtime.pending_transition_indices.clear()
            runtime.final_gate_count = state.gate_count
            archive_started = time.perf_counter()
            retain_replay_state(
                replay_pool[runtime.circuit],
                exact_graph,
                capacity=replay_capacity_per_circuit,
                graph_hash=graph_hash,
            )
            add_seconds("refresh_result_archive_seconds", archive_started)
            circuit_best = best_by_circuit[runtime.circuit]
            if state.gate_count < circuit_best["gate_count"]:
                best_started = time.perf_counter()
                circuit_best["gate_count"] = state.gate_count
                circuit_best["qasm"] = exact_graph.to_qasm_str()
                circuit_best["episode_depth"] = state.depth
                add_seconds("refresh_result_best_qasm_seconds", best_started)
            add_seconds("refresh_result_processing_seconds", result_started)
    add_seconds("refresh_profiled_total_seconds", refresh_started)


def collect_episode_batch(
    qasm: Path,
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
    actor_critic: torch.nn.Module,
    device: torch.device,
    threshold_config: dict,
    source_vectors: torch.Tensor,
    source_representations: torch.Tensor | None,
    max_steps: int,
    max_source_matches: int,
    source_microbatch: int,
    source_grouping: str,
    proposal_expansion: str,
    transition_transfer_backend: str,
    proposal_tensor_backend: str,
    proposal_materialization_backend: str,
    policy_padding_backend: str,
    episode_initialization_backend: str,
    advance_input_backend: str,
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
    refresh_interval: int,
    collector_timing: dict[str, float] | None = None,
) -> tuple[list[PPOTransition], list[EpisodeMetrics]]:
    if batch_size < 1:
        raise ValueError("collector batch size must be positive")
    if refresh_interval < 1:
        raise ValueError("PPO refresh interval must be positive")
    (
        active,
        states,
        live,
        gate_types,
        arena,
        handles,
    ) = initialize_episode_batch(
        qasm,
        batch_size,
        context=context,
        quartz=quartz,
        model=model,
        device=device,
        max_steps=max_steps,
        page_size=page_size,
        initialization_backend=episode_initialization_backend,
        start_from_best=start_from_best,
        use_replay_starts=use_replay_starts,
        replay_start_probability=replay_start_probability,
        best_by_circuit=best_by_circuit,
        replay_pool=replay_pool,
    )
    all_runtimes = list(active)

    while active:
        current_states = [runtime.state for runtime in active]
        candidates, model_match_seconds, _, encoded = paged_model_matches(
            current_states,
            states,
            live,
            gate_types,
            handles,
            arena,
            model,
            device,
            threshold_config,
            source_vectors,
            microbatch=len(active),
            max_candidates=max_source_matches,
            source_microbatch=source_microbatch,
            source_grouping=source_grouping,
            state_batch_backend="tensorized",
            candidate_backend="gpu",
            return_encoded_states=True,
        )
        if collector_timing is not None:
            collector_timing["model_match_seconds"] += model_match_seconds
        proposal_started = time.perf_counter()
        proposals, proposal_metrics, _, selected_proposal_tensors = (
            build_gpu_proposals(
                candidates,
                current_states,
                rule_index,
                per_parent_cap=max_actions,
                global_cap=max_actions * len(active),
                ranking_mode="gate",
                preselect_matches=proposal_expansion == "preselect",
                return_selected_tensors=(
                    proposal_tensor_backend == "reuse"
                    or proposal_materialization_backend == "deferred"
                ),
                materialize_python_proposals=(
                    proposal_materialization_backend == "eager"
                ),
            )
        )
        if collector_timing is not None:
            collector_timing["proposal_seconds"] += (
                time.perf_counter() - proposal_started
            )
            eligible_actions = int(proposal_metrics["eligible_actions"])
            collector_timing["eligible_actions"] += eligible_actions
            collector_timing["materialized_actions"] += int(
                proposal_metrics.get("materialized_actions", eligible_actions)
            )
            selected_actions = int(proposal_metrics.get("selected_actions", 0))
            collector_timing["selected_proposals"] += selected_actions
            if proposal_materialization_backend == "eager":
                collector_timing["python_proposals_materialized"] += selected_actions
        policy_preparation_started = time.perf_counter()
        has_proposals = (
            bool(proposals)
            if proposal_materialization_backend == "eager"
            else selected_proposal_tensors is not None
            and bool(selected_proposal_tensors.parent_ids.numel())
        )
        if has_proposals:
            (
                flat_features,
                flat_logits,
                flat_parent_ids,
            ) = batched_proposal_features(
                model,
                encoded,
                live,
                current_states,
                proposals,
                rules,
                device,
                initial_gate_bias,
                ordered_roles=actor_critic.match_set_aware,
                proposal_tensors=selected_proposal_tensors,
                source_representations=source_representations,
            )
            (
                policy_features,
                matcher_logits,
                candidate_mask,
                grouped_proposals,
                candidate_flat_indices,
            ) = pad_batched_policy_inputs(
                flat_features,
                flat_logits,
                proposals,
                len(active),
                parent_ids=flat_parent_ids,
                backend=policy_padding_backend,
            )
        else:
            policy_features = encoded.new_zeros(
                (len(active), 0, actor_critic.policy_feature_dim)
            )
            matcher_logits = encoded.new_zeros((len(active), 0))
            candidate_mask = torch.zeros(
                (len(active), 0), dtype=torch.bool, device=device
            )
            grouped_proposals = [[] for _ in active]
            candidate_flat_indices = torch.empty(
                (len(active), 0), dtype=torch.long, device=device
            )

        state_features = build_state_features(
            encoded,
            live,
            torch.tensor(
                [runtime.state.gate_count for runtime in active], device=device
            ),
        )
        prefix_states = current_prefix_states(encoded, live, arena, handles)
        with torch.no_grad():
            if has_proposals:
                value_mask = candidate_mask.clone()
                value_mask[~value_mask.any(1), 0] = True
                old_values = actor_critic.state_values(
                    state_features,
                    policy_features,
                    value_mask,
                    prefix_states,
                )
            else:
                old_values = state_features.new_zeros(len(active))

        transition_state_features = None
        transition_policy_features = None
        transition_matcher_logits = None
        transition_candidate_mask = None
        transition_prefix_states = None
        transition_old_values = None
        if transition_transfer_backend == "batched":
            transfer_started = time.perf_counter()
            transition_state_features = state_features.float().cpu()
            transition_policy_features = policy_features.float().cpu()
            transition_matcher_logits = matcher_logits.float().cpu()
            transition_candidate_mask = candidate_mask.cpu()
            transition_prefix_states = prefix_states.float().cpu()
            transition_old_values = old_values.float().cpu().tolist()
            if collector_timing is not None:
                collector_timing["transition_transfer_seconds"] += (
                    time.perf_counter() - transfer_started
                )
        if collector_timing is not None:
            collector_timing["policy_preparation_seconds"] += (
                time.perf_counter() - policy_preparation_started
            )

        advance_records: dict[int, tuple[BeamState, Proposal]] = {}
        refresh_indices = set()
        candidate_presence = (
            transition_candidate_mask.any(1).tolist()
            if transition_candidate_mask is not None
            else candidate_mask.any(1).tolist()
        )
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
            with torch.no_grad():
                distribution = masked_policy_distribution(
                    actor_critic,
                    policy_features.index_select(0, unresolved_tensor),
                    matcher_logits.index_select(0, unresolved_tensor),
                    candidate_mask.index_select(0, unresolved_tensor),
                    prefix_states=prefix_states.index_select(
                        0, unresolved_tensor
                    ),
                    state_features=state_features.index_select(
                        0, unresolved_tensor
                    ),
                )
                actions = (
                    distribution.logits.argmax(-1)
                    if greedy
                    else distribution.sample()
                )
                log_probs = distribution.log_prob(actions)
                entropies = distribution.entropy()
            transfer_started = time.perf_counter()
            if transition_transfer_backend == "batched":
                actor_outputs = torch.stack(
                    (actions.float(), log_probs, entropies), dim=1
                ).cpu().tolist()
            else:
                actor_outputs = [
                    (
                        int(actions[index].item()),
                        float(log_probs[index].item()),
                        float(entropies[index].item()),
                    )
                    for index in range(len(unresolved))
                ]
            if collector_timing is not None:
                collector_timing["transition_transfer_seconds"] += (
                    time.perf_counter() - transfer_started
                )
            chosen_proposals = None
            if proposal_materialization_backend == "deferred":
                materialization_started = time.perf_counter()
                chosen_flat_indices = candidate_flat_indices[
                    unresolved_tensor, actions
                ]
                chosen_proposals = materialize_selected_proposals(
                    selected_proposal_tensors, chosen_flat_indices
                )
                if collector_timing is not None:
                    collector_timing["proposal_materialization_seconds"] += (
                        time.perf_counter() - materialization_started
                    )
                    collector_timing["python_proposals_materialized"] += len(
                        chosen_proposals
                    )
            retry = []
            for row_index, parent_index in enumerate(unresolved):
                runtime = active[parent_index]
                action_index = int(actor_outputs[row_index][0])
                proposal = (
                    chosen_proposals[row_index]
                    if chosen_proposals is not None
                    else grouped_proposals[parent_index][action_index]
                )
                old_log_prob = float(actor_outputs[row_index][1])
                entropy = float(actor_outputs[row_index][2])
                local_mask = candidate_mask[parent_index]
                transfer_started = time.perf_counter()
                if transition_transfer_backend == "batched":
                    local_transition_mask = transition_candidate_mask[parent_index]
                    candidate_count = int(local_transition_mask.sum().item())
                    saved_state_features = transition_state_features[parent_index]
                    saved_policy_features = transition_policy_features[parent_index]
                    saved_matcher_logits = transition_matcher_logits[parent_index]
                    saved_candidate_mask = local_transition_mask.clone()
                    saved_prefix_state = transition_prefix_states[parent_index]
                    saved_old_value = float(transition_old_values[parent_index])
                else:
                    candidate_count = int(local_mask.sum().item())
                    saved_state_features = state_features[parent_index].float().cpu()
                    saved_policy_features = policy_features[parent_index].float().cpu()
                    saved_matcher_logits = matcher_logits[parent_index].float().cpu()
                    saved_candidate_mask = local_mask.cpu().clone()
                    saved_prefix_state = prefix_states[parent_index].float().cpu()
                    saved_old_value = float(old_values[parent_index].item())
                if collector_timing is not None:
                    collector_timing["transition_transfer_seconds"] += (
                        time.perf_counter() - transfer_started
                    )
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
                rejected = child is None
                can_retry = (
                    rejected
                    and rejected_counts[parent_index]
                    < max_rejected_actions_per_step
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
                transition = PPOTransition(
                    state_features=saved_state_features,
                    candidate_features=saved_policy_features,
                    matcher_logits=saved_matcher_logits,
                    candidate_mask=saved_candidate_mask,
                    action_index=action_index,
                    old_log_prob=old_log_prob,
                    old_value=saved_old_value,
                    reward=reward,
                    done=rejected and not can_retry,
                    legal=legal,
                    xfer_id=proposal.xfer_id,
                    matcher_probability=proposal.probability,
                    prefix_state=saved_prefix_state,
                    history_depth=runtime.state.depth + 1,
                    committed_action=child is not None,
                    repeated_state=bool(duplicate),
                    candidate_count=candidate_count,
                    policy_entropy=entropy,
                    previous_gate_count=runtime.state.gate_count,
                    next_gate_count=next_gate_count,
                )
                runtime.transitions.append(transition)
                if rejected:
                    candidate_mask[parent_index, action_index] = False
                    if transition_candidate_mask is not None:
                        transition_candidate_mask[parent_index, action_index] = False
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
                runtime.state = child
                runtime.topology_hashes.add(int(fingerprint))
                advance_records[parent_index] = (child, proposal)
                improved_initial = child.gate_count < runtime.initial_gate_count
                improved_global = (
                    child.gate_count
                    < best_by_circuit[runtime.circuit]["gate_count"]
                )
                if child.depth >= max_steps:
                    runtime.terminated_reason = "horizon"
                    runtime.stopped = True
                elif terminate_on_improvement and improved_initial:
                    runtime.terminated_reason = "improvement"
                    runtime.stopped = True
                if (
                    child.depth - child.exact_checkpoint_depth >= refresh_interval
                    or runtime.stopped
                    or improved_global
                ):
                    refresh_indices.add(parent_index)
            unresolved = retry

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
        )

        continuing_records = []
        continuing_runtimes = []
        for parent_index, runtime in enumerate(active):
            record = advance_records.get(parent_index)
            if runtime.stopped or record is None:
                continue
            continuing_records.append(record)
            continuing_runtimes.append(runtime)
        if not continuing_records:
            for handle in handles:
                arena.release(handle)
            break
        (
            states,
            live,
            gate_types,
            handles,
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
            trusted_paged_inputs=advance_input_backend == "trusted",
            source_representations=source_representations,
        )
        if collector_timing is not None:
            collector_timing["cache_advance_seconds"] += advance_seconds
        active = continuing_runtimes

    transitions = []
    episode_metrics = []
    for runtime in all_runtimes:
        finalize_episode(runtime.transitions, gamma, gae_lambda)
        transitions.extend(runtime.transitions)
        legal_actions = sum(row.legal for row in runtime.transitions)
        invalid_actions = len(runtime.transitions) - legal_actions
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
                invalid_actions=invalid_actions,
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
                mean_candidates=sum(
                    row.candidate_count for row in runtime.transitions
                )
                / max(1, len(runtime.transitions)),
                mean_entropy=sum(
                    row.policy_entropy for row in runtime.transitions
                )
                / max(1, len(runtime.transitions)),
                started_from_replay=runtime.started_from_replay,
                terminated_reason=runtime.terminated_reason,
                exact_refreshes=runtime.exact_refreshes,
                exact_replay_actions=runtime.exact_replay_actions,
                exact_refresh_seconds=runtime.exact_refresh_seconds,
            )
        )
    return transitions, episode_metrics


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
        "prefix_states": torch.stack(
            [
                row.prefix_state
                if row.prefix_state is not None
                else torch.zeros((policy_width - 2) // 4)
                for row in rows
            ]
        ).to(device),
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
        "legal": torch.tensor(
            [row.legal for row in rows], dtype=torch.float, device=device
        ),
    }


def ppo_update(
    actor_critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions: list[PPOTransition],
    *,
    device: torch.device,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
    legality_coefficient: float,
    reference_kl_coefficient: float,
    target_kl: float,
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
    legality_rate = sum(row.legal for row in transitions) / len(transitions)
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    completed_epochs = 0
    early_stopped = False
    actor_critic.train()
    for _ in range(epochs):
        epoch_kl = 0.0
        epoch_batches = 0
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
                prefix_states=batch["prefix_states"],
                state_features=batch["state_features"],
            )
            new_log_probs = distribution.log_prob(batch["actions"])
            new_values = actor_critic.state_values(
                batch["state_features"],
                batch["candidate_features"],
                batch["candidate_mask"],
                batch["prefix_states"],
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
            reference_kl = categorical_reference_kl(
                distribution,
                batch["matcher_logits"],
                batch["candidate_mask"],
            )
            legality_logits = actor_critic.candidate_legality_logits(
                batch["candidate_features"],
                batch["candidate_mask"],
                batch["prefix_states"],
                batch["state_features"],
            )
            if legality_logits is None:
                legality_loss = objective.loss.new_zeros(())
                legality_accuracy = objective.loss.new_zeros(())
            else:
                selected_legality = legality_logits.gather(
                    1, batch["actions"].unsqueeze(1)
                ).squeeze(1)
                legality_losses = F.binary_cross_entropy_with_logits(
                    selected_legality, batch["legal"], reduction="none"
                )
                if 0.0 < legality_rate < 1.0:
                    legality_weights = torch.where(
                        batch["legal"].bool(),
                        0.5 / legality_rate,
                        0.5 / (1.0 - legality_rate),
                    )
                    legality_loss = (legality_losses * legality_weights).mean()
                else:
                    legality_loss = legality_losses.mean()
                legality_accuracy = (
                    selected_legality.ge(0).eq(batch["legal"].bool())
                ).float().mean()
                legal_predictions = selected_legality.ge(0)
                totals["legality_legal_correct"] += float(
                    (legal_predictions & batch["legal"].bool()).sum()
                )
                totals["legality_legal_count"] += float(batch["legal"].sum())
                totals["legality_invalid_correct"] += float(
                    (~legal_predictions & ~batch["legal"].bool()).sum()
                )
                totals["legality_invalid_count"] += float(
                    (~batch["legal"].bool()).sum()
                )
            loss = (
                objective.loss
                + legality_coefficient * legality_loss
                + reference_kl_coefficient * reference_kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                actor_critic.parameters(), max_grad_norm
            )
            optimizer.step()
            for key, value in (
                ("loss", loss),
                ("ppo_loss", objective.loss),
                ("policy_loss", objective.policy_loss),
                ("value_loss", objective.value_loss),
                ("legality_loss", legality_loss),
                ("legality_accuracy", legality_accuracy),
                ("reference_kl", reference_kl),
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
    legal_count = totals.pop("legality_legal_count", 0.0)
    legal_correct = totals.pop("legality_legal_correct", 0.0)
    invalid_count = totals.pop("legality_invalid_count", 0.0)
    invalid_correct = totals.pop("legality_invalid_correct", 0.0)
    result = {key: value / max(1, batches) for key, value in totals.items()}
    if legal_count and invalid_count:
        result["legality_legal_recall"] = legal_correct / legal_count
        result["legality_invalid_recall"] = invalid_correct / invalid_count
        result["legality_balanced_accuracy"] = 0.5 * (
            result["legality_legal_recall"]
            + result["legality_invalid_recall"]
        )
    result["legality_label_rate"] = legality_rate
    result["batches"] = batches
    result["completed_epochs"] = completed_epochs
    result["target_kl_early_stopped"] = early_stopped
    return result


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
        "exact_refreshes": sum(row.exact_refreshes for row in rows),
        "exact_replay_actions": sum(row.exact_replay_actions for row in rows),
        "exact_refresh_seconds": sum(row.exact_refresh_seconds for row in rows),
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
    *,
    collector_batch_size: int = 1,
    refresh_interval: int = 1,
    **episode_kwargs,
) -> dict:
    episode_kwargs["actor_critic"].eval()
    episode_kwargs["use_replay_starts"] = False
    rows = []
    for qasm in qasms:
        remaining = episodes_per_circuit
        while remaining:
            current_batch = min(collector_batch_size, remaining)
            if collector_batch_size == 1 and refresh_interval == 1:
                _, metrics = collect_episode(
                    qasm, greedy=True, **episode_kwargs
                )
                rows.append(metrics)
            else:
                _, batch_rows = collect_episode_batch(
                    qasm,
                    current_batch,
                    greedy=True,
                    refresh_interval=refresh_interval,
                    **episode_kwargs,
                )
                rows.extend(batch_rows)
            remaining -= current_batch
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
    parser.add_argument(
        "--source-microbatch",
        type=int,
        default=0,
        help=(
            "score this many source patterns at once during PPO collection; "
            "zero keeps each first-gate group whole"
        ),
    )
    parser.add_argument(
        "--source-grouping",
        choices=("none", "first_gate"),
        default="none",
        help="skip source products whose first gate differs from the anchor",
    )
    parser.add_argument(
        "--proposal-expansion",
        choices=("full", "preselect"),
        default="full",
        help="preselect top matches before materializing their xfers",
    )
    parser.add_argument(
        "--transition-transfer-backend",
        choices=("rowwise", "batched"),
        default="rowwise",
        help="batch PPO transition tensor copies from GPU to CPU",
    )
    parser.add_argument(
        "--proposal-tensor-backend",
        choices=("rebuild", "reuse"),
        default="rebuild",
        help="reuse selected GPU proposal tensors for PPO candidate features",
    )
    parser.add_argument(
        "--proposal-materialization-backend",
        choices=("eager", "deferred"),
        default="eager",
        help="materialize only actor-selected Python proposals",
    )
    parser.add_argument(
        "--source-representation-backend",
        choices=("recompute", "cached"),
        default="recompute",
        help="reuse frozen source-pattern representations during collection",
    )
    parser.add_argument(
        "--policy-padding-backend",
        choices=("loop", "tensorized"),
        default="loop",
        help="pack per-parent PPO candidate sets with one GPU scatter",
    )
    parser.add_argument(
        "--episode-initialization-backend",
        choices=("duplicated", "deduplicated"),
        default="duplicated",
        help="share parsing, topology, and initial encoding across equal starts",
    )
    parser.add_argument(
        "--advance-input-backend",
        choices=("checked", "trusted"),
        default="checked",
        help="skip per-step GPU syncs for caller-validated paged inputs",
    )
    parser.add_argument("--max-actions", type=int, default=128)
    parser.add_argument("--max-gate-increase", type=int, default=3)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument(
        "--actor-critic",
        choices=("legacy", "match_set"),
        default="legacy",
    )
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--set-heads", type=int, default=4)
    parser.add_argument(
        "--collector-batch-size",
        type=int,
        default=1,
        help="episodes advanced together by the paged speculative collector",
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=1,
        help="actions between exact Quartz checks (1 preserves the exact collector)",
    )
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--initial-gate-bias", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument(
        "--skip-ppo-updates",
        action="store_true",
        help=(
            "collect fixed-policy search trajectories without running PPO updates; "
            "useful for candidate-cap and replay-policy audits"
        ),
    )
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--legality-coefficient", type=float, default=0.1)
    parser.add_argument("--reference-kl-coefficient", type=float, default=0.0)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.0,
        help="stop PPO epochs when mean old-policy KL exceeds 1.5x this value",
    )
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
    if args.source_microbatch < 0:
        parser.error("source microbatch must be nonnegative")
    if args.collector_batch_size < 1 or args.refresh_interval < 1:
        parser.error("collector batch size and refresh interval must be positive")
    if args.set_layers < 0 or args.set_heads < 1:
        parser.error("set layers must be nonnegative and set heads positive")
    if args.legality_coefficient < 0:
        parser.error("legality coefficient must be nonnegative")
    if args.reference_kl_coefficient < 0 or args.target_kl < 0:
        parser.error("KL coefficient and target must be nonnegative")
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
    if (
        args.proposal_materialization_backend == "deferred"
        and args.proposal_tensor_backend != "reuse"
    ):
        parser.error("deferred proposals require --proposal-tensor-backend reuse")
    if (
        args.proposal_materialization_backend == "deferred"
        and args.policy_padding_backend != "tensorized"
    ):
        parser.error("deferred proposals require tensorized policy padding")

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
    if args.max_steps > model.max_sequence_length:
        parser.error("max steps exceed the base model sequence length")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    actor_critic = build_actor_critic(
        args.actor_critic,
        model.width,
        hidden_size=args.hidden_size,
        set_layers=args.set_layers,
        set_heads=args.set_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(actor_critic.actor_parameters()),
                "lr": args.actor_learning_rate,
            },
            {
                "params": list(actor_critic.critic_parameters()),
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
        if resume.get("format") not in {"paged-ppo-v1", "paged-ppo-v2"}:
            raise ValueError("unsupported PPO checkpoint")
        resume_architecture = resume.get("actor_critic_architecture", "legacy")
        if resume_architecture != args.actor_critic:
            raise ValueError(
                "resume actor/critic architecture differs from --actor-critic"
            )
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
        cached_source_representations = model.source_representations()
        source_vectors = model.retrieval_source(cached_source_representations)
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
        "collector": (
            "exact"
            if args.collector_batch_size == 1 and args.refresh_interval == 1
            else "batched_speculative"
        ),
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
        "source_representations": (
            cached_source_representations
            if args.source_representation_backend == "cached"
            else None
        ),
        "max_steps": args.max_steps,
        "max_source_matches": args.max_source_matches,
        "source_microbatch": args.source_microbatch,
        "source_grouping": args.source_grouping,
        "proposal_expansion": args.proposal_expansion,
        "transition_transfer_backend": args.transition_transfer_backend,
        "proposal_tensor_backend": args.proposal_tensor_backend,
        "proposal_materialization_backend": args.proposal_materialization_backend,
        "policy_padding_backend": args.policy_padding_backend,
        "episode_initialization_backend": args.episode_initialization_backend,
        "advance_input_backend": args.advance_input_backend,
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
            collector_batch_size=args.collector_batch_size,
            refresh_interval=args.refresh_interval,
            **episode_kwargs,
        )

    for iteration in range(start_iteration, start_iteration + args.iterations):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        collection_started = time.perf_counter()
        transitions = []
        episode_rows = []
        collection_timing = {
            "model_match_seconds": 0.0,
            "proposal_seconds": 0.0,
            "proposal_materialization_seconds": 0.0,
            "transition_transfer_seconds": 0.0,
            "policy_preparation_seconds": 0.0,
            "cache_advance_seconds": 0.0,
            "eligible_actions": 0,
            "materialized_actions": 0,
            "selected_proposals": 0,
            "python_proposals_materialized": 0,
        }
        circuit_counts = {
            qasm: sum(
                args.qasm[episode % len(args.qasm)] == qasm
                for episode in range(args.episodes_per_iteration)
            )
            for qasm in args.qasm
        }
        for qasm, episode_count in circuit_counts.items():
            remaining = episode_count
            while remaining:
                current_batch = min(args.collector_batch_size, remaining)
                if args.collector_batch_size == 1 and args.refresh_interval == 1:
                    episode_transitions, metrics = collect_episode(
                        qasm,
                        greedy=False,
                        collector_timing=collection_timing,
                        **episode_kwargs,
                    )
                    transitions.extend(episode_transitions)
                    episode_rows.append(metrics)
                else:
                    batch_transitions, batch_metrics = collect_episode_batch(
                        qasm,
                        current_batch,
                        greedy=False,
                        refresh_interval=args.refresh_interval,
                        collector_timing=collection_timing,
                        **episode_kwargs,
                    )
                    transitions.extend(batch_transitions)
                    episode_rows.extend(batch_metrics)
                remaining -= current_batch
        collection_seconds = time.perf_counter() - collection_started
        collection_timing["unattributed_seconds"] = max(
            0.0,
            collection_seconds
            - collection_timing["model_match_seconds"]
            - collection_timing["proposal_seconds"]
            - collection_timing["transition_transfer_seconds"],
        )
        collection_timing["model_match_fraction"] = (
            collection_timing["model_match_seconds"] / collection_seconds
        )
        collection_timing["proposal_fraction"] = (
            collection_timing["proposal_seconds"] / collection_seconds
        )
        collection_timing["proposal_materialization_fraction"] = (
            collection_timing["proposal_materialization_seconds"]
            / collection_seconds
        )
        collection_timing["transition_transfer_fraction"] = (
            collection_timing["transition_transfer_seconds"]
            / collection_seconds
        )
        collection_peak_cuda_allocated_gib = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        collection_peak_cuda_reserved_gib = (
            torch.cuda.max_memory_reserved(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        if not transitions:
            raise RuntimeError("PPO collection produced no transitions")

        update_started = time.perf_counter()
        if args.skip_ppo_updates:
            update_metrics = {
                "skipped": True,
                "legality_label_rate": sum(row.legal for row in transitions)
                / len(transitions),
                "batches": 0,
                "completed_epochs": 0,
                "target_kl_early_stopped": False,
            }
        else:
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
                legality_coefficient=args.legality_coefficient,
                reference_kl_coefficient=args.reference_kl_coefficient,
                target_kl=args.target_kl,
                max_grad_norm=args.max_grad_norm,
                seed=args.seed + iteration,
            )
        update_seconds = time.perf_counter() - update_started
        collection_metrics = aggregate_episodes(episode_rows)
        evaluation_metrics = (
            evaluate_policy(
                args.qasm,
                args.evaluation_episodes_per_circuit,
                collector_batch_size=args.collector_batch_size,
                refresh_interval=args.refresh_interval,
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
            "collection_timing": collection_timing,
            "collection_peak_cuda_allocated_gib": (
                collection_peak_cuda_allocated_gib
            ),
            "collection_peak_cuda_reserved_gib": (
                collection_peak_cuda_reserved_gib
            ),
            "update_seconds": update_seconds,
            "update_samples_per_second": (
                0.0
                if args.skip_ppo_updates
                else len(transitions) * args.ppo_epochs / update_seconds
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
                "format": (
                    "paged-ppo-v2"
                    if args.actor_critic == "match_set"
                    else "paged-ppo-v1"
                ),
                "iteration": iteration,
                "actor_critic": actor_critic.state_dict(),
                "actor_critic_architecture": args.actor_critic,
                "optimizer": optimizer.state_dict(),
                "width": model.width,
                "hidden_size": actor_critic.hidden_size,
                "set_layers": args.set_layers,
                "set_heads": args.set_heads,
                "base_checkpoint": str(args.checkpoint),
                "best_by_circuit": best_by_circuit,
                "replay_pool": serializable_replay_pool(replay_pool),
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
