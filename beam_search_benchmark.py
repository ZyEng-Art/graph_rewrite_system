from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import ctypes
import ctypes.util
from dataclasses import dataclass, replace
import gc
import hashlib
import heapq
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import types
from typing import Any

# Required by CUDA deterministic-algorithm mode before the first cuBLAS handle
# is created.  It has no effect unless deterministic algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch

from circuit_identity import ExactGraphRegistry, QuartzHashRegistry, exact_graph_key
from dataset import (
    _local_streak_bucket,
    _touch_age_bucket,
    compact_live_slots,
    RuleMetadata,
)
from gpu_proposals import GpuRuleIndex, build_gpu_proposals
from incremental_graph import parse_pattern
from model_factory import build_model
from search_types import BeamState, Proposal
from search_survivor import gate_priority, path_best_gate_count, select_survivors
from search_feedback import SearchFeedbackRegistry
from search_widening import select_widening_revisits
from widening_candidate_cache import WideningCandidateCache
from successor_fingerprint import (
    FingerprintAudit,
    build_wire_trace_profile,
    successor_fingerprint,
)
from threshold_inference import (
    CandidateTensors,
    load_threshold_config,
    threshold_candidates,
    threshold_candidate_tensors,
)
from train import autocast_context, move_batch
from train_neural_successor_prefilter import NeuralSuccessorPrefilter


NATIVE_APPLY_PROFILE_NAMES = (
    "native_guid_lookup",
    "native_source_match",
    "native_input_validation",
    "native_destination_creation",
    "native_output_validation",
    "native_graph_rewrite",
    "native_loop_check",
    "native_trace",
    "native_rotation_elimination",
    "native_unmatch_cleanup",
)
NATIVE_GRAPH_REWRITE_PROFILE_NAMES = (
    "native_graph_allocate",
    "native_graph_copy_constants",
    "native_graph_copy_special_guid",
    "native_graph_copy_qubit_map",
    "native_graph_copy_in_edges",
    "native_graph_copy_out_edges",
    "native_graph_reconnect_outputs",
    "native_graph_remove_source_ops",
    "native_graph_add_destination_ops",
    "native_graph_rebuild_qubit_index",
)
NATIVE_APPLY_RESULT_NAMES = {
    0: "success",
    1: "source_match_rejected",
    2: "input_qubit_alias_rejected",
    3: "destination_creation_rejected",
    4: "output_boundary_rejected",
    5: "cycle_rejected",
    6: "guid_lookup_rejected",
}
NATIVE_TRANSACTION_PROFILE_NAMES = (
    "transaction_guid_lookup",
    "transaction_source_match",
    "transaction_input_validation",
    "transaction_destination_creation",
    "transaction_output_validation",
    "transaction_rewrite",
    "transaction_loop_check",
    "transaction_binding_export",
    "transaction_rotation_elimination",
    "transaction_exact_key",
    "transaction_registry",
    "transaction_novel_clone",
    "transaction_rollback",
    "transaction_unmatch",
    "transaction_cpp_total",
)
NATIVE_TRANSACTION_RESULT_NAMES = {
    0: "novel",
    1: "exact_duplicate",
    2: "source_match_rejected",
    3: "input_qubit_alias_rejected",
    4: "destination_creation_rejected",
    5: "output_boundary_rejected",
    6: "cycle_rejected",
    7: "guid_lookup_rejected",
}
NATIVE_TRANSACTION_COUNTER_NAMES = (
    "transaction_captured_in_entries",
    "transaction_captured_out_entries",
    "transaction_captured_constant_entries",
    "transaction_captured_position_entries",
    "transaction_incremental_position_updates",
    "transaction_graph_copies",
)
NATIVE_TRANSACTION_DELTA_COUNTER_NAMES = (
    "transaction_removed_nodes",
    "transaction_added_nodes",
    "transaction_removed_edges",
    "transaction_added_edges",
)


def add_profile_ns(profile: Counter, name: str, started_ns: int) -> None:
    profile[f"{name}_ns"] += time.perf_counter_ns() - started_ns


def rendered_apply_profile(profile: Counter) -> dict:
    seconds = {
        key.removesuffix("_ns"): value / 1e9
        for key, value in sorted(profile.items())
        if key.endswith("_ns")
    }
    counts = {
        key.removesuffix("_count"): int(value)
        for key, value in sorted(profile.items())
        if key.endswith("_count")
    }
    return {"seconds": seconds, "counts": counts}


@dataclass(frozen=True)
class AppliedRewrite:
    """A Quartz successor before expensive BeamState metadata is materialized."""

    graph: Any
    source_guids: tuple[int, ...]
    destination_guids: tuple[int, ...]
    exact_identity: Any | None = None
    transaction_status: int | None = None
    structural_delta: dict[str, Any] | None = None


def update_slots(graph, guid_to_slot: dict[int, int], next_slot: int, preferred=()):
    live_guids = {int(node.guid) for node in graph.nodes}
    for raw_guid in preferred:
        guid = int(raw_guid)
        if guid not in live_guids:
            raise RuntimeError("destination GUID missing after rewrite")
        if guid not in guid_to_slot:
            guid_to_slot[guid] = next_slot
            next_slot += 1
    for node in graph.nodes:
        guid = int(node.guid)
        if guid not in guid_to_slot:
            guid_to_slot[guid] = next_slot
            next_slot += 1
    return next_slot


def snapshot(
    graph, guid_to_slot: dict[int, int], *, profile: Counter | None = None
) -> dict:
    started = time.perf_counter_ns() if profile is not None else 0
    nodes = list(graph.nodes)
    if profile is not None:
        add_profile_ns(profile, "child_snapshot_nodes", started)
    started = time.perf_counter_ns() if profile is not None else 0
    node_rows = sorted(
        (guid_to_slot[int(node.guid)], int(node.gate_tp), int(node.guid))
        for node in nodes
    )
    if profile is not None:
        add_profile_ns(profile, "child_snapshot_node_rows", started)
    started = time.perf_counter_ns() if profile is not None else 0
    raw_edges = graph.all_edges()
    if profile is not None:
        add_profile_ns(profile, "child_snapshot_native_edges", started)
    started = time.perf_counter_ns() if profile is not None else 0
    edge_rows = sorted(
        (
            guid_to_slot[int(nodes[int(src)].guid)],
            guid_to_slot[int(nodes[int(dst)].guid)],
            int(src_port),
            int(dst_port),
        )
        for src, dst, src_port, dst_port in raw_edges
    )
    if profile is not None:
        add_profile_ns(profile, "child_snapshot_edge_rows", started)
    return {"nodes": node_rows, "edges": edge_rows}


def graph_delta(before: dict, after: dict) -> tuple[set[int], set[tuple[int, ...]]]:
    before_nodes = {int(row[0]) for row in before["nodes"]}
    after_nodes = {int(row[0]) for row in after["nodes"]}
    before_edges = set(map(tuple, before["edges"]))
    after_edges = set(map(tuple, after["edges"]))
    return before_nodes - after_nodes, before_edges.symmetric_difference(after_edges)


def incremental_snapshot_from_delta(
    parent: BeamState,
    applied: AppliedRewrite,
    *,
    profile: Counter | None = None,
) -> tuple[
    dict,
    dict[int, int],
    int,
    tuple[int, ...],
    set[int],
    set[tuple[int, ...]],
]:
    """Apply Quartz's exact local node/edge delta to cached search metadata."""

    started = time.perf_counter_ns() if profile is not None else 0
    delta = applied.structural_delta
    if delta is None:
        raise ValueError("incremental snapshot requires a structural delta")
    removed_guids = set(map(int, delta["removed_node_guids"]))
    added_nodes = [
        (int(guid), int(gate_type)) for guid, gate_type in delta["added_nodes"]
    ]
    added_types = dict(added_nodes)
    guid_to_slot = dict(parent.guid_to_slot)
    next_slot = parent.next_slot

    surviving_destination_guids = tuple(
        guid for guid in applied.destination_guids if guid in added_types
    )
    for guid in surviving_destination_guids:
        if guid not in guid_to_slot:
            guid_to_slot[guid] = next_slot
            next_slot += 1
    for guid, _gate_type in added_nodes:
        if guid not in guid_to_slot:
            guid_to_slot[guid] = next_slot
            next_slot += 1

    node_rows = [
        tuple(map(int, row))
        for row in parent.snapshot["nodes"]
        if int(row[2]) not in removed_guids
    ]
    node_rows.extend(
        (guid_to_slot[guid], gate_type, guid)
        for guid, gate_type in added_nodes
    )
    node_rows.sort()

    def slot_edge(raw_edge) -> tuple[int, int, int, int]:
        src_guid, dst_guid, src_port, dst_port = map(int, raw_edge)
        try:
            return (
                guid_to_slot[src_guid],
                guid_to_slot[dst_guid],
                src_port,
                dst_port,
            )
        except KeyError as error:
            raise RuntimeError(
                f"transaction delta references unknown GUID {error.args[0]}"
            ) from error

    removed_edges = set(map(slot_edge, delta["removed_edges"]))
    added_edges = set(map(slot_edge, delta["added_edges"]))
    after_edges = set(map(tuple, parent.snapshot["edges"]))
    if not removed_edges.issubset(after_edges):
        raise RuntimeError("transaction delta removes an absent cached edge")
    after_edges.difference_update(removed_edges)
    after_edges.update(added_edges)
    after = {"nodes": node_rows, "edges": sorted(after_edges)}
    removed_slots = {
        parent.guid_to_slot[guid]
        for guid in removed_guids
        if guid in parent.guid_to_slot
    }
    changed_edges = removed_edges | added_edges
    if profile is not None:
        add_profile_ns(profile, "child_incremental_snapshot", started)
        profile["child_incremental_snapshot_count"] += 1
        profile["child_incremental_removed_nodes_count"] += len(removed_guids)
        profile["child_incremental_added_nodes_count"] += len(added_nodes)
        profile["child_incremental_removed_edges_count"] += len(removed_edges)
        profile["child_incremental_added_edges_count"] += len(added_edges)
    return (
        after,
        guid_to_slot,
        next_slot,
        surviving_destination_guids,
        removed_slots,
        changed_edges,
    )


def graph_collection_identity_digest(graphs) -> str:
    """Compact deterministic checksum for exact A/B beam comparisons."""

    rows = sorted(
        repr(exact_graph_key(graph)).encode("utf-8") for graph in graphs
    )
    digest = hashlib.sha256()
    digest.update(len(rows).to_bytes(8, "little"))
    for row in rows:
        digest.update(len(row).to_bytes(8, "little"))
        digest.update(row)
    return digest.hexdigest()


def distances_from_core(snapshot_row: dict, core: set[int]) -> dict[int, int]:
    live = {int(row[0]) for row in snapshot_row["nodes"]}
    adjacency = {slot: set() for slot in live}
    for src, dst, _, _ in snapshot_row["edges"]:
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    result = {slot: 0 for slot in core if slot in live}
    queue = deque(result)
    while queue:
        slot = queue.popleft()
        for neighbor in adjacency[slot]:
            if neighbor not in result:
                result[neighbor] = result[slot] + 1
                queue.append(neighbor)
    return {slot: min(result.get(slot, 5), 4) if slot in result else 5 for slot in live}


def collate_states(states: list[BeamState]) -> dict:
    max_slots = max(max((row[0] for row in state.snapshot["nodes"]), default=-1) + 1 for state in states)
    batch_size = len(states)
    current_types = torch.full((batch_size, max_slots), -1, dtype=torch.long)
    rewrite_distance = torch.full((batch_size, max_slots), 5, dtype=torch.long)
    touch_age = torch.full((batch_size, max_slots), 7, dtype=torch.long)
    streak = torch.zeros(batch_size, dtype=torch.long)
    edge_batch = []
    edge_src = []
    edge_dst = []
    edge_relation = []
    for batch_index, state in enumerate(states):
        for slot, gate_type, _ in state.snapshot["nodes"]:
            current_types[batch_index, slot] = gate_type
            rewrite_distance[batch_index, slot] = state.rewrite_distance.get(slot, 5)
            if slot in state.last_touched:
                age = state.depth - 1 - state.last_touched[slot]
                touch_age[batch_index, slot] = _touch_age_bucket(age)
        streak[batch_index] = _local_streak_bucket(bool(state.depth), state.local_streak)
        for src, dst, src_port, dst_port in state.snapshot["edges"]:
            edge_batch.append(batch_index)
            edge_src.append(src)
            edge_dst.append(dst)
            edge_relation.append(src_port * 4 + dst_port)
    return {
        "current_types": current_types,
        "current_live_slots": compact_live_slots(current_types),
        "current_edge_batch": torch.tensor(edge_batch, dtype=torch.long),
        "current_edge_src": torch.tensor(edge_src, dtype=torch.long),
        "current_edge_dst": torch.tensor(edge_dst, dtype=torch.long),
        "current_edge_relation": torch.tensor(edge_relation, dtype=torch.long),
        "current_rewrite_distance": rewrite_distance,
        "current_touch_age": touch_age,
        "current_local_streak": streak,
    }


def collate_matcher_states(states: list[BeamState], *, paged_action: bool) -> dict:
    """Collate exact current graphs for either matcher architecture.

    The paged matcher is deliberately rebased on each exact Quartz graph in
    this benchmark.  That keeps the CPU and model modes semantically aligned:
    both observe the same post-normalization circuit at every depth, while the
    comparison changes only candidate enumeration.
    """
    batch = collate_states(states)
    if not paged_action:
        return batch
    batch_size = batch["current_types"].shape[0]
    empty_actions = torch.empty((batch_size, 0), dtype=torch.long)
    empty_bindings = torch.empty((batch_size, 0, 0), dtype=torch.long)
    batch.update(
        {
            "initial_types": batch["current_types"].clone(),
            "edge_batch": batch["current_edge_batch"],
            "edge_src": batch["current_edge_src"],
            "edge_dst": batch["current_edge_dst"],
            "edge_relation": batch["current_edge_relation"],
            "action_xfers": empty_actions,
            "action_sources": empty_actions.clone(),
            "binding_slots": empty_bindings,
            "destination_slots": empty_bindings.clone(),
            "destination_types": empty_bindings.clone(),
        }
    )
    return batch


def collate_exact_states(
    states: list[BeamState],
) -> tuple[dict, torch.Tensor, dict[str, int]]:
    """Collate only each state's live exact graph, with dense local slots.

    Beam states retain monotonically assigned slots so Quartz bindings and
    diagnostics remain stable across a trajectory.  Those historical holes
    have no meaning to a state-only matcher, so this path compacts every exact
    graph independently and returns a map back to the persistent beam slots.
    """
    if not states:
        raise ValueError("cannot collate an empty state list")
    live_counts = [len(state.snapshot["nodes"]) for state in states]
    max_live = max(live_counts)
    if not max_live:
        raise ValueError("cannot match an empty circuit")
    batch_size = len(states)
    type_rows = []
    distance_rows = []
    touch_rows = []
    slot_rows = []
    live_slot_rows = []
    streak_rows = []
    edge_batches = []
    edge_sources = []
    edge_destinations = []
    edge_relations = []
    max_persistent_slots = 0
    for batch_index, state in enumerate(states):
        nodes = sorted(state.snapshot["nodes"], key=lambda row: int(row[0]))
        persistent_slots = [int(row[0]) for row in nodes]
        slot_to_dense = {
            persistent_slot: dense_slot
            for dense_slot, persistent_slot in enumerate(persistent_slots)
        }
        slot_rows.append(torch.tensor(persistent_slots, dtype=torch.long))
        type_rows.append(
            torch.tensor([int(row[1]) for row in nodes], dtype=torch.long)
        )
        distance_rows.append(
            torch.tensor(
                [
                    state.rewrite_distance.get(persistent_slot, 5)
                    for persistent_slot in persistent_slots
                ],
                dtype=torch.long,
            )
        )
        touch_rows.append(
            torch.tensor(
                [
                    _touch_age_bucket(
                        state.depth - 1 - state.last_touched[persistent_slot]
                    )
                    if persistent_slot in state.last_touched
                    else 7
                    for persistent_slot in persistent_slots
                ],
                dtype=torch.long,
            )
        )
        live_slot_rows.append(torch.arange(len(nodes), dtype=torch.long))
        max_persistent_slots = max(
            max_persistent_slots,
            max(slot_to_dense, default=-1) + 1,
        )
        streak_rows.append(
            _local_streak_bucket(bool(state.depth), state.local_streak)
        )
        edges = state.snapshot["edges"]
        if edges:
            edge_batches.append(
                torch.full((len(edges),), batch_index, dtype=torch.long)
            )
            edge_sources.append(
                torch.tensor(
                    [slot_to_dense[int(row[0])] for row in edges],
                    dtype=torch.long,
                )
            )
            edge_destinations.append(
                torch.tensor(
                    [slot_to_dense[int(row[1])] for row in edges],
                    dtype=torch.long,
                )
            )
            edge_relations.append(
                torch.tensor(
                    [int(row[2]) * 4 + int(row[3]) for row in edges],
                    dtype=torch.long,
                )
            )
    pad = torch.nn.utils.rnn.pad_sequence
    current_types = pad(type_rows, batch_first=True, padding_value=-1)
    rewrite_distance = pad(distance_rows, batch_first=True, padding_value=5)
    touch_age = pad(touch_rows, batch_first=True, padding_value=7)
    dense_to_slot = pad(slot_rows, batch_first=True, padding_value=-1)
    current_live_slots = pad(
        live_slot_rows, batch_first=True, padding_value=-1
    )
    empty = torch.empty(0, dtype=torch.long)
    batch = {
        "current_types": current_types,
        "current_live_slots": current_live_slots,
        "current_edge_batch": (
            torch.cat(edge_batches) if edge_batches else empty
        ),
        "current_edge_src": (
            torch.cat(edge_sources) if edge_sources else empty.clone()
        ),
        "current_edge_dst": (
            torch.cat(edge_destinations) if edge_destinations else empty.clone()
        ),
        "current_edge_relation": (
            torch.cat(edge_relations) if edge_relations else empty.clone()
        ),
        "current_rewrite_distance": rewrite_distance,
        "current_touch_age": touch_age,
        "current_local_streak": torch.tensor(streak_rows, dtype=torch.long),
    }
    return batch, dense_to_slot, {
        "live_nodes": sum(live_counts),
        "padded_dense_slots": batch_size * max_live,
        "padded_persistent_slots": batch_size * max_persistent_slots,
        "max_dense_slots": max_live,
        "max_persistent_slots": max_persistent_slots,
    }


def remap_candidate_slots(
    candidates: CandidateTensors,
    dense_to_slot: torch.Tensor,
    *,
    batch_offset: int,
) -> CandidateTensors:
    """Map state-local dense matcher slots back to persistent Quartz slots."""
    if not candidates.batch_ids.numel():
        return CandidateTensors(
            batch_ids=candidates.batch_ids + batch_offset,
            sources=candidates.sources,
            anchors=candidates.anchors,
            bindings=candidates.bindings,
            probabilities=candidates.probabilities,
        )
    if dense_to_slot.device != candidates.batch_ids.device:
        dense_to_slot = dense_to_slot.to(candidates.batch_ids.device)
    anchors = dense_to_slot[candidates.batch_ids, candidates.anchors]
    binding_present = candidates.bindings.ge(0)
    safe_bindings = candidates.bindings.clamp_min(0)
    binding_rows = candidates.batch_ids.unsqueeze(1).expand_as(safe_bindings)
    bindings = dense_to_slot[binding_rows, safe_bindings]
    bindings = torch.where(binding_present, bindings, -1)
    return CandidateTensors(
        batch_ids=candidates.batch_ids + batch_offset,
        sources=candidates.sources,
        anchors=anchors,
        bindings=bindings,
        probabilities=candidates.probabilities,
    )


def frozen_candidate_features(
    model,
    states: torch.Tensor,
    live: torch.Tensor,
    proposal_tensors,
    source_representations: torch.Tensor,
) -> torch.Tensor:
    """Reuse matcher embeddings without adding parameters to its checkpoint."""

    binding_slots = proposal_tensors.bindings
    batch_ids = proposal_tensors.parent_ids
    binding_mask = binding_slots.ge(0)
    safe_bindings = binding_slots.clamp_min(0)
    num_slots = states.shape[1]
    flat_indices = batch_ids.unsqueeze(1) * num_slots + safe_bindings
    bound_states = states.reshape(-1, model.width)[flat_indices]
    bound_states = bound_states.masked_fill(
        ~binding_mask.unsqueeze(-1), 0
    )
    bound_pool = bound_states.sum(1)
    bound_pool = bound_pool / binding_mask.sum(1, keepdim=True).clamp_min(1)
    live_states = states.masked_fill(~live.unsqueeze(-1), 0)
    graph_pool = live_states.sum(1)
    graph_pool = graph_pool / live.sum(1, keepdim=True).clamp_min(1)
    return torch.cat(
        (
            model.xfer_embedding(proposal_tensors.xfer_ids),
            source_representations.index_select(
                0, proposal_tensors.source_ids
            ),
            bound_pool,
            graph_pool.index_select(0, batch_ids),
        ),
        dim=-1,
    )


def load_neural_successor_prefilter(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "neural_successor_prefilter_v1":
        raise ValueError(f"unsupported neural prefilter checkpoint: {path}")
    train_args = checkpoint.get("args", {})
    model = NeuralSuccessorPrefilter(
        int(checkpoint["input_width"]),
        int(checkpoint["hidden_width"]),
        int(checkpoint["embedding_width"]),
        float(train_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    thresholds = checkpoint["thresholds"]
    return model, {
        "valid": float(thresholds["valid"]),
        "duplicate": float(thresholds["duplicate"]),
    }


@torch.no_grad()
def neural_prefilter_scores(
    prefilter: NeuralSuccessorPrefilter,
    frozen_features: torch.Tensor,
    proposal_tensors,
    beam: list[BeamState],
    step: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score proposal legality and duplicate propensity entirely on device."""

    parent_gate_counts = torch.tensor(
        [state.gate_count for state in beam],
        device=device,
        dtype=torch.float32,
    ).index_select(0, proposal_tensors.parent_ids)
    auxiliary = torch.stack(
        (
            proposal_tensors.probabilities.float(),
            proposal_tensors.gate_deltas.float() / 8.0,
            parent_gate_counts / 512.0,
            torch.full_like(parent_gate_counts, (step + 1) / 64.0),
        ),
        dim=1,
    )
    inputs = torch.cat((frozen_features.float(), auxiliary), dim=1)
    valid_scores = []
    duplicate_scores = []
    for begin in range(0, inputs.shape[0], batch_size):
        with autocast_context(device):
            valid, duplicate, _ = prefilter(inputs[begin : begin + batch_size])
        valid_scores.append(valid.float().sigmoid())
        duplicate_scores.append(duplicate.float().sigmoid())
    return torch.cat(valid_scores), torch.cat(duplicate_scores)


def exact_actions(
    state: BeamState, context
) -> list[tuple[int, int, tuple[int, ...] | None, float]]:
    """Enumerate actions through Quartz's original xfer-at-anchor API."""
    rows = []
    for node in state.graph.nodes:
        anchor_slot = state.guid_to_slot[int(node.guid)]
        for xfer_id in state.graph.available_xfers_parallel(context=context, node=node):
            rows.append((int(xfer_id), anchor_slot, None, 1.0))
    return rows


@torch.no_grad()
def model_matches(
    states: list[BeamState],
    model,
    device,
    threshold_config,
    microbatch: int,
    max_candidates: int,
    *,
    paged_action: bool = False,
) -> tuple[list[list[tuple[int, int, tuple[int, ...], float]]], float]:
    output = []
    started = time.perf_counter()
    with autocast_context(device):
        source_vectors = model.retrieval_source(model.source_representations())
    for begin in range(0, len(states), microbatch):
        selected = states[begin : begin + microbatch]
        batch = move_batch(
            collate_matcher_states(selected, paged_action=paged_action), device
        )
        with autocast_context(device):
            encoded, live, gate_types = model.encode(batch)
            logits, eligible = model.match_logits(
                encoded, live, gate_types, source_vectors=source_vectors
            )
        output.extend(
            threshold_candidates(
                model,
                batch,
                logits,
                eligible,
                threshold_config,
                max_candidates_per_state=max_candidates,
            )
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - started


@torch.no_grad()
def state_only_candidate_tensors(
    states: list[BeamState],
    model,
    device: torch.device,
    threshold_config: dict,
    source_vectors: torch.Tensor,
    microbatch: int,
    max_candidates: int,
    *,
    return_encoded_states: bool = False,
):
    """Predict all retained matches from exact graphs without action tensors."""
    if not hasattr(model, "encode_current_graph"):
        raise ValueError(
            "state-only inference requires a model with encode_current_graph"
        )
    chunks = []
    encoded_chunks = []
    encoded_live_chunks = []
    max_persistent_slots = max(
        max((int(row[0]) for row in state.snapshot["nodes"]), default=-1) + 1
        for state in states
    )
    collation = {
        "live_nodes": 0,
        "padded_dense_slots": 0,
        "padded_persistent_slots": 0,
        "max_dense_slots": 0,
        "max_persistent_slots": 0,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for begin in range(0, len(states), microbatch):
        selected = states[begin : begin + microbatch]
        cpu_batch, dense_to_slot, batch_stats = collate_exact_states(selected)
        for name in ("live_nodes", "padded_dense_slots", "padded_persistent_slots"):
            collation[name] += batch_stats[name]
        for name in ("max_dense_slots", "max_persistent_slots"):
            collation[name] = max(collation[name], batch_stats[name])
        batch = move_batch(cpu_batch, device)
        with autocast_context(device):
            encoded, live, gate_types = model.encode_current_graph(batch)
            logits, eligible = model.match_logits(
                encoded,
                live,
                gate_types,
                source_vectors=source_vectors,
            )
        if return_encoded_states:
            slot_map = dense_to_slot.to(device)
            mapped = slot_map.ge(0)
            persistent = encoded.new_zeros(
                encoded.shape[0], max_persistent_slots, encoded.shape[-1]
            )
            persistent_live = torch.zeros(
                encoded.shape[0],
                max_persistent_slots,
                dtype=torch.bool,
                device=device,
            )
            dense_rows = (
                torch.arange(encoded.shape[0], device=device)
                .unsqueeze(1)
                .expand_as(slot_map)
            )
            dense_columns = (
                torch.arange(encoded.shape[1], device=device)
                .unsqueeze(0)
                .expand_as(slot_map)
            )
            parent_rows = dense_rows[mapped]
            persistent_slots = slot_map[mapped]
            persistent[parent_rows, persistent_slots] = encoded[
                parent_rows, dense_columns[mapped]
            ]
            persistent_live[parent_rows, persistent_slots] = True
            encoded_chunks.append(persistent)
            encoded_live_chunks.append(persistent_live)
        local_candidates = threshold_candidate_tensors(
            model,
            batch,
            logits,
            eligible,
            threshold_config,
            max_candidates_per_state=max_candidates,
        )
        chunks.append(
            remap_candidate_slots(
                local_candidates,
                dense_to_slot,
                batch_offset=begin,
            )
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = (
        CandidateTensors.cat(chunks),
        time.perf_counter() - started,
        collation,
    )
    if not return_encoded_states:
        return result
    return (
        *result,
        (torch.cat(encoded_chunks), torch.cat(encoded_live_chunks)),
    )


def rank_proposals(
    beam: list[BeamState],
    action_rows: list[list[tuple[int, int, tuple[int, ...] | None, float]]],
    gate_deltas: list[int],
    *,
    beam_size: int,
    max_actions_per_parent: int,
    proposal_factor: int,
    max_gate_increase: int,
    ranking_mode: str = "gate",
    ranking_seed: int = 0,
) -> tuple[list[Proposal], int]:
    """Select stable bounded top-k actions before creating Proposal objects.

    Large circuits can expose millions of legal source/xfer combinations per
    layer, while the search consumes only a small per-parent and global prefix.
    Selecting raw tuples first preserves the old stable sort order but avoids
    allocating and sorting a Python dataclass for every discarded action.
    """
    if ranking_mode not in {"gate", "probability", "stochastic"}:
        raise ValueError(f"unknown proposal ranking mode: {ranking_mode}")
    generator = random.Random(ranking_seed)
    effective_parent_cap = max(
        max_actions_per_parent,
        math.ceil(beam_size / max(1, len(beam))) * 2,
    )
    proposals = []
    total_action_candidates = 0
    for parent_index, (state, rows) in enumerate(zip(beam, action_rows)):
        eligible_rows = [
            row for row in rows if gate_deltas[row[0]] <= max_gate_increase
        ]
        total_action_candidates += len(eligible_rows)
        decorated_rows = [(row, generator.random()) for row in eligible_rows]

        def parent_rank(item):
            row, random_priority = item
            next_gate_count = state.gate_count + gate_deltas[row[0]]
            if ranking_mode == "probability":
                return (-row[3], next_gate_count, row[0])
            if ranking_mode == "stochastic":
                return (-random_priority, next_gate_count, row[0])
            return (next_gate_count, -row[3], row[0])

        selected_rows = heapq.nsmallest(
            effective_parent_cap,
            decorated_rows,
            key=parent_rank,
        )
        proposals.extend(
            Proposal(
                parent=parent_index,
                xfer_id=row[0],
                anchor_slot=row[1],
                binding=row[2],
                probability=row[3],
                next_gate_count=state.gate_count + gate_deltas[row[0]],
                value_score=random_priority if ranking_mode == "stochastic" else 0.0,
            )
            for row, random_priority in selected_rows
        )

    def global_rank(row: Proposal):
        if ranking_mode == "probability":
            return (-row.probability, row.next_gate_count, beam[row.parent].gate_count)
        if ranking_mode == "stochastic":
            return (-row.value_score, row.next_gate_count, beam[row.parent].gate_count)
        return (row.next_gate_count, -row.probability, beam[row.parent].gate_count)

    proposals = heapq.nsmallest(
        beam_size * proposal_factor,
        proposals,
        key=global_rank,
    )
    return proposals, total_action_candidates


def is_direct_inverse_proposal(
    parent: BeamState,
    proposal: Proposal,
    inverse_xfer_ids: tuple[int, ...],
) -> bool:
    """Recognize an immediate rewrite followed by its unique reverse rule."""

    previous = parent.last_xfer_id
    return (
        proposal.binding is not None
        and 0 <= previous < len(inverse_xfer_ids)
        and 0 <= proposal.xfer_id < len(inverse_xfer_ids)
        and inverse_xfer_ids[previous] == proposal.xfer_id
        and inverse_xfer_ids[proposal.xfer_id] == previous
        and proposal.binding == parent.last_destination_slots
    )


def apply_rewrite(
    parent: BeamState,
    proposal: Proposal,
    xfers,
    *,
    eliminate_rotation: bool = False,
    binding_backend: str = "auto",
    profile: Counter | None = None,
    transactional_registry: Any | None = None,
) -> AppliedRewrite | None:
    """Apply one proposal without constructing metadata for duplicate children.

    A patched Quartz can validate the model's complete ordered source binding
    directly.  This avoids re-running anchor-based subgraph matching merely to
    rediscover a binding the model already supplied.  Original Quartz actions
    contain only an anchor and continue to use the original exact API.
    """
    started = time.perf_counter_ns() if profile is not None else 0
    slot_to_guid = {
        int(slot): int(guid) for slot, _, guid in parent.snapshot["nodes"]
    }
    if profile is not None:
        profile["attempted_count"] += 1
        add_profile_ns(profile, "python_slot_to_guid", started)
    guid_direct_method = getattr(
        parent.graph, "apply_xfer_with_guid_binding", None
    )
    node_direct_method = getattr(
        parent.graph, "apply_xfer_with_node_id_binding", None
    )
    use_direct = proposal.binding is not None and (
        binding_backend in ("direct", "guid_direct", "node_direct")
        or (
            binding_backend == "auto"
            and (guid_direct_method is not None or node_direct_method is not None)
        )
    )
    if use_direct:
        if guid_direct_method is None and node_direct_method is None:
            raise RuntimeError(
                "direct binding apply was requested, but the loaded Quartz "
                "extension does not provide a direct binding API"
            )
        try:
            started = time.perf_counter_ns() if profile is not None else 0
            source_guids = [slot_to_guid[slot] for slot in proposal.binding]
        except KeyError:
            if profile is not None:
                add_profile_ns(profile, "python_binding_to_guid", started)
                profile["python_missing_slot_count"] += 1
            return None
        if profile is not None:
            add_profile_ns(profile, "python_binding_to_guid", started)
        prefer_guid = binding_backend != "node_direct"
        if guid_direct_method is not None and prefer_guid:
            call_started = time.perf_counter_ns() if profile is not None else 0
            if transactional_registry is not None:
                transactional_method = getattr(
                    parent.graph,
                    "apply_xfer_with_guid_binding_transactional",
                    None,
                )
                if transactional_method is None:
                    raise RuntimeError(
                        "transactional apply requires the patched Quartz API"
                    )
                (
                    graph,
                    destination_guids,
                    native_key,
                    status,
                    native_values,
                    structural_delta,
                ) = transactional_method(
                    xfer=xfers[proposal.xfer_id],
                    source_node_guids=source_guids,
                    registry=transactional_registry,
                    eliminate_rotation=eliminate_rotation,
                )
                if profile is not None:
                    native_wall_ns = time.perf_counter_ns() - call_started
                    profile["transaction_call_wall_ns"] += native_wall_ns
                    for name, value in zip(
                        NATIVE_TRANSACTION_PROFILE_NAMES,
                        native_values[:15],
                    ):
                        profile[f"{name}_ns"] += int(value)
                    result_name = NATIVE_TRANSACTION_RESULT_NAMES.get(
                        int(status), f"unknown_{int(status)}"
                    )
                    profile[f"transaction_result_{result_name}_count"] += 1
                    for name, value in zip(
                        NATIVE_TRANSACTION_COUNTER_NAMES,
                        native_values[16:22],
                    ):
                        profile[f"{name}_count"] += int(value)
                    if len(native_values) > 22:
                        profile["transaction_delta_export_ns"] += int(
                            native_values[22]
                        )
                    for name, value in zip(
                        NATIVE_TRANSACTION_DELTA_COUNTER_NAMES,
                        native_values[23:27],
                    ):
                        profile[f"{name}_count"] += int(value)
                    cpp_ns = int(native_values[14])
                    profile["transaction_wrapper_unattributed_ns"] += max(
                        0, native_wall_ns - cpp_ns
                    )
                if int(status) >= 2:
                    return None
                exact_identity = (
                    "quartz_wire_trace_v1",
                    bytes(native_key),
                )
                return AppliedRewrite(
                    graph=graph,
                    source_guids=tuple(source_guids),
                    destination_guids=tuple(map(int, destination_guids)),
                    exact_identity=exact_identity,
                    transaction_status=int(status),
                    structural_delta=structural_delta,
                )
            if profile is not None:
                profiled_method = getattr(
                    parent.graph,
                    "apply_xfer_with_guid_binding_profiled",
                    None,
                )
                if profiled_method is None:
                    raise RuntimeError(
                        "detailed apply profiling requires the profiled "
                        "Quartz direct-binding API"
                    )
                graph, destination_guids, native_values = profiled_method(
                    xfer=xfers[proposal.xfer_id],
                    source_node_guids=source_guids,
                    eliminate_rotation=eliminate_rotation,
                )
                native_wall_ns = time.perf_counter_ns() - call_started
                profile["native_call_wall_ns"] += native_wall_ns
                for name, value in zip(
                    NATIVE_APPLY_PROFILE_NAMES, native_values[:10]
                ):
                    profile[f"{name}_ns"] += int(value)
                for name, value in zip(
                    NATIVE_GRAPH_REWRITE_PROFILE_NAMES,
                    native_values[12:22],
                ):
                    profile[f"{name}_ns"] += int(value)
                result_code = int(native_values[10])
                result_name = NATIVE_APPLY_RESULT_NAMES.get(
                    result_code, f"unknown_{result_code}"
                )
                profile[f"native_result_{result_name}_count"] += 1
                native_cpp_ns = int(native_values[11])
                profile["native_cpp_total_ns"] += native_cpp_ns
                profile["native_wrapper_unattributed_ns"] += max(
                    0, native_wall_ns - native_cpp_ns
                )
            else:
                graph, destination_guids = guid_direct_method(
                    xfer=xfers[proposal.xfer_id],
                    source_node_guids=source_guids,
                    eliminate_rotation=eliminate_rotation,
                )
            if graph is None:
                return None
            return AppliedRewrite(
                graph=graph,
                source_guids=tuple(source_guids),
                destination_guids=tuple(map(int, destination_guids)),
            )

        if node_direct_method is None:
            raise RuntimeError(
                "node-ID direct binding apply was selected, but the loaded "
                "Quartz extension does not provide that API"
            )
        nodes = list(parent.graph.nodes)
        guid_to_node_id = {
            int(node.guid): node_id for node_id, node in enumerate(nodes)
        }
        try:
            source_node_ids = [guid_to_node_id[guid] for guid in source_guids]
        except KeyError:
            return None
        result = node_direct_method(
            xfer=xfers[proposal.xfer_id],
            source_node_ids=source_node_ids,
            eliminate_rotation=eliminate_rotation,
        )
    else:
        if proposal.anchor_slot not in slot_to_guid:
            return None
        nodes = list(parent.graph.nodes)
        guid_to_node_id = {
            int(node.guid): node_id for node_id, node in enumerate(nodes)
        }
        node = parent.graph.get_node_from_id(
            id=guid_to_node_id[slot_to_guid[proposal.anchor_slot]]
        )
        result = parent.graph.apply_xfer_with_binding_trace(
            xfer=xfers[proposal.xfer_id],
            node=node,
            eliminate_rotation=eliminate_rotation,
            predecessor_layers=1,
        )
    if result is None or result[0] is None:
        return None
    graph, _, source_guids, destination_guids = result
    source_slots = tuple(parent.guid_to_slot[int(guid)] for guid in source_guids)
    # Model proposals carry the complete source binding, so this is also the
    # ground-truth validity check. Original Quartz actions only carry an anchor.
    if proposal.binding is not None and source_slots != proposal.binding:
        return None

    return AppliedRewrite(
        graph=graph,
        source_guids=tuple(map(int, source_guids)),
        destination_guids=tuple(map(int, destination_guids)),
    )


def materialize_child(
    parent: BeamState,
    proposal: Proposal,
    applied: AppliedRewrite,
    *,
    eliminate_rotation: bool = False,
    profile: Counter | None = None,
) -> BeamState:
    """Construct model/search metadata only after exact dedup accepts a graph."""
    total_started = time.perf_counter_ns() if profile is not None else 0
    graph = applied.graph
    started = time.perf_counter_ns() if profile is not None else 0
    source_slots = tuple(
        parent.guid_to_slot[guid] for guid in applied.source_guids
    )
    if profile is not None:
        profile["materialized_children_count"] += 1
        add_profile_ns(profile, "child_source_slots", started)

    if applied.structural_delta is not None:
        (
            after,
            guid_to_slot,
            next_slot,
            surviving_destination_guids,
            removed,
            changed_edges,
        ) = incremental_snapshot_from_delta(parent, applied, profile=profile)
    else:
        started = time.perf_counter_ns() if profile is not None else 0
        live_guids = {int(node.guid) for node in graph.nodes}
        if profile is not None:
            add_profile_ns(profile, "child_materialize_nodes", started)
        started = time.perf_counter_ns() if profile is not None else 0
        surviving_destination_guids = tuple(
            guid for guid in applied.destination_guids if guid in live_guids
        )
        if profile is not None:
            add_profile_ns(profile, "child_filter_destinations", started)
        started = time.perf_counter_ns() if profile is not None else 0
        guid_to_slot = dict(parent.guid_to_slot)
        if profile is not None:
            add_profile_ns(profile, "child_copy_slot_map", started)
        started = time.perf_counter_ns() if profile is not None else 0
        next_slot = update_slots(
            graph,
            guid_to_slot,
            parent.next_slot,
            surviving_destination_guids,
        )
        if profile is not None:
            add_profile_ns(profile, "child_update_slots", started)
        after = snapshot(graph, guid_to_slot, profile=profile)
        started = time.perf_counter_ns() if profile is not None else 0
        removed, changed_edges = graph_delta(parent.snapshot, after)
        if profile is not None:
            add_profile_ns(profile, "child_graph_delta", started)
    started = time.perf_counter_ns() if profile is not None else 0
    live = {int(row[0]) for row in after["nodes"]}
    ordered_destination_slots = tuple(
        guid_to_slot[guid] for guid in surviving_destination_guids
    )
    destination_slots = set(ordered_destination_slots)
    core = set(destination_slots)
    for src, dst, _, _ in changed_edges:
        if src in live:
            core.add(src)
        if dst in live:
            core.add(dst)
    last_touched = {
        slot: touched
        for slot, touched in parent.last_touched.items()
        if slot in live and slot not in removed
    }
    for slot in core:
        last_touched[slot] = parent.depth

    source_set = set(source_slots)
    predecessors = {
        src for src, dst, _, _ in parent.snapshot["edges"] if dst in source_set
    }
    previous_preferred = (destination_slots | predecessors) & live
    continued = proposal.anchor_slot in parent.previous_preferred
    local_streak = parent.local_streak + 1 if continued else 0
    gate_count = len(after["nodes"])
    parent_path_best = path_best_gate_count(parent)
    child_path_best = min(parent_path_best, gate_count)
    stagnation_steps = (
        0
        if gate_count < parent_path_best
        else int(getattr(parent, "stagnation_steps", 0)) + 1
    )
    exploration_ancestor = bool(
        getattr(parent, "exploration_ancestor", False)
        or getattr(parent, "survivor_lane", "root") == "exploration"
    )
    recovered_after_exploration = bool(
        exploration_ancestor and gate_count < parent_path_best
    )
    if profile is not None:
        add_profile_ns(profile, "child_local_metadata", started)
    if not eliminate_rotation and gate_count != proposal.next_gate_count:
        raise RuntimeError(
            f"gate delta mismatch: expected {proposal.next_gate_count}, got {gate_count}"
        )
    started = time.perf_counter_ns() if profile is not None else 0
    rewrite_distance = distances_from_core(after, core)
    if profile is not None:
        add_profile_ns(profile, "child_rewrite_distance", started)
    started = time.perf_counter_ns() if profile is not None else 0
    child = BeamState(
        graph=graph,
        snapshot=after,
        guid_to_slot=guid_to_slot,
        next_slot=next_slot,
        last_touched=last_touched,
        rewrite_distance=rewrite_distance,
        previous_preferred=previous_preferred,
        local_streak=local_streak,
        gate_count=gate_count,
        depth=parent.depth + 1,
        history=parent.history + ((proposal.xfer_id, proposal.anchor_slot),),
        last_xfer_id=proposal.xfer_id,
        last_source_slots=source_slots,
        last_destination_slots=ordered_destination_slots,
        path_best_gate_count=child_path_best,
        stagnation_steps=stagnation_steps,
        survivor_lane="unselected",
        exploration_ancestor=exploration_ancestor,
        recovered_after_exploration=recovered_after_exploration,
        expansion_round=0,
        last_action_parent_rank=proposal.parent_rank,
        widening_ancestor=(
            parent.widening_ancestor or parent.expansion_round > 0
        ),
        widened_action_trace=(
            parent.widened_action_trace
            + (
                (
                    parent.depth + 1,
                    proposal.parent_rank,
                    proposal.xfer_id,
                    proposal.anchor_slot,
                ),
            )
            if parent.expansion_round > 0
            else parent.widened_action_trace
        ),
    )
    if profile is not None:
        add_profile_ns(profile, "child_state_construction", started)
        add_profile_ns(profile, "child_total", total_started)
    return child


def make_child(
    parent: BeamState,
    proposal: Proposal,
    context,
    xfers,
    *,
    eliminate_rotation: bool = False,
    binding_backend: str = "auto",
) -> BeamState | None:
    """Compatibility helper for callers that need a fully materialized child."""
    del context  # Kept in the public signature for existing replay utilities.
    applied = apply_rewrite(
        parent,
        proposal,
        xfers,
        eliminate_rotation=eliminate_rotation,
        binding_backend=binding_backend,
    )
    if applied is None:
        return None
    return materialize_child(
        parent,
        proposal,
        applied,
        eliminate_rotation=eliminate_rotation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("model", "quartz"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--target-recall", type=float, default=0.97)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--max-source-matches", type=int, default=2048)
    parser.add_argument("--max-actions-per-parent", type=int, default=128)
    parser.add_argument("--proposal-factor", type=int, default=8)
    parser.add_argument(
        "--survivor-policy",
        choices=("gate", "dual_lane"),
        default="gate",
        help=(
            "historical gate-count beam, or reserve fixed beam slots for "
            "bounded no-improvement paths"
        ),
    )
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--exploration-max-stagnation", type=int, default=8)
    parser.add_argument("--exploration-max-detour", type=int, default=2)
    parser.add_argument("--exploration-seed", type=int, default=73)
    parser.add_argument(
        "--locality-action-reserve",
        type=int,
        default=0,
        help=(
            "per-parent action slots reserved for bindings overlapping the "
            "previous rewrite neighborhood"
        ),
    )
    parser.add_argument(
        "--survivor-candidate-factor",
        type=float,
        default=1.25,
        help=(
            "novel children collected before fixed-size survivor selection; "
            "gate policy always uses 1.0 for historical equivalence"
        ),
    )
    parser.add_argument(
        "--max-total-attempted-actions",
        type=int,
        default=0,
        help="global exact-apply budget; zero means unlimited",
    )
    parser.add_argument(
        "--progressive-widening",
        choices=("off", "on"),
        default="off",
        help=(
            "reserve beam slots to revisit exact parent graphs and expand "
            "successive disjoint per-parent action-rank bands"
        ),
    )
    parser.add_argument("--widening-revisit-fraction", type=float, default=0.25)
    parser.add_argument("--widening-max-expansions", type=int, default=4)
    parser.add_argument(
        "--widening-candidate-cache",
        choices=("off", "on"),
        default="off",
        help=(
            "reuse state-only matcher and structural-decode candidates when an "
            "exact parent is revisited for a later action-rank band"
        ),
    )
    parser.add_argument(
        "--widening-min-actions-per-parent",
        type=int,
        default=4,
        help=(
            "minimum globally selected proposals per live parent when "
            "progressive widening is enabled"
        ),
    )
    parser.add_argument("--widening-seed", type=int, default=73)
    parser.add_argument(
        "--deterministic-search",
        action="store_true",
        help=(
            "require deterministic CUDA kernels so repeated feedback search "
            "does not diverge from atomic GNN aggregation roundoff"
        ),
    )
    parser.add_argument(
        "--widening-policy",
        choices=("round_robin", "feedback"),
        default="round_robin",
        help=(
            "choose revisit parents with the historical seeded round-robin "
            "policy or deterministic online exact-search feedback"
        ),
    )
    parser.add_argument(
        "--proposal-ranking",
        choices=("gate", "probability", "stochastic"),
        default="gate",
        help="rank selected model actions by immediate cost, probability, or a seeded random key",
    )
    parser.add_argument(
        "--proposal-ranking-seed",
        type=int,
        default=73,
        help="base seed for reproducible stochastic proposal ranking",
    )
    parser.add_argument("--max-gate-increase", type=int, default=1)
    parser.add_argument("--refresh-interval", type=int, default=0)
    parser.add_argument("--refresh-count", type=int, default=100)
    parser.add_argument(
        "--target-gate-count",
        type=int,
        help="stop after the historical best gate count is reached",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-qasm", type=Path)
    parser.add_argument(
        "--apply-profile",
        choices=("off", "detailed"),
        default="off",
        help=(
            "collect Python and native C++ timings for every direct-binding "
            "apply stage; intended for profiling runs only"
        ),
    )
    parser.add_argument(
        "--transactional-apply",
        choices=("off", "on"),
        default="off",
        help=(
            "temporarily rewrite the parent, compute exact identity, roll "
            "back duplicates, and clone only novel successors"
        ),
    )
    parser.add_argument(
        "--neural-audit-output",
        type=Path,
        help=(
            "optional torch payload of frozen matcher candidate features and "
            "exact invalid/duplicate/unique labels"
        ),
    )
    parser.add_argument(
        "--neural-prefilter-checkpoint",
        type=Path,
        help=(
            "optional learned legality/duplicate checkpoint; it is used only "
            "to prioritize exact Quartz applies"
        ),
    )
    parser.add_argument(
        "--neural-prefilter-mode",
        choices=("off", "shadow", "defer"),
        default="off",
        help=(
            "shadow only scores proposals; defer moves high-confidence "
            "invalid/duplicate proposals behind the other exact applies"
        ),
    )
    parser.add_argument(
        "--neural-prefilter-batch-size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        help=(
            "optional trajectory payload whose first trajectory is checked "
            "for Quartz-hash retention after every beam layer"
        ),
    )
    parser.add_argument(
        "--stop-on-reference-loss",
        action="store_true",
        help="stop as soon as the expected reference prefix leaves the beam",
    )
    parser.add_argument(
        "--dedup-identity",
        choices=("exact", "quartz_hash"),
        default="exact",
        help=(
            "collision-safe physical-wire identity, or Quartz's legacy lossy "
            "integer hash for controlled historical comparisons"
        ),
    )
    parser.add_argument(
        "--preapply-fingerprint",
        choices=("off", "shadow", "filter"),
        default="off",
        help=(
            "incrementally predict successor wire traces before Quartz apply; "
            "shadow audits exactness without pruning, filter skips repeated "
            "fingerprints"
        ),
    )
    parser.add_argument(
        "--preapply-direct-inverse",
        choices=("off", "shadow", "filter"),
        default="off",
        help=(
            "detect an action immediately consuming the prior action's "
            "destination through its unique reverse xfer"
        ),
    )
    parser.add_argument(
        "--preapply-fingerprint-kind",
        choices=(
            "conservative",
            "parameter_transfer",
            "xfer_guarded",
            "topology",
        ),
        default="conservative",
        help=(
            "xfer-specific symbolic parameters, one-to-one parameter "
            "transfer, xfer-guarded transfer, or parameter-blind topology"
        ),
    )
    parser.add_argument(
        "--preapply-fingerprint-backend",
        choices=("auto", "python", "native"),
        default="auto",
        help=(
            "use Quartz's compact native successor profile when available, "
            "or force the Python reference implementation for A/B"
        ),
    )
    parser.add_argument(
        "--preapply-fingerprint-native-batch-size",
        type=int,
        default=512,
        help=(
            "number of proposal positions staged per native fingerprint "
            "batch; zero disables batching for controlled A/B"
        ),
    )
    parser.add_argument(
        "--preapply-fingerprint-representatives",
        type=int,
        default=1,
        help=(
            "number of valid proposals retained per fingerprint before "
            "filtering; use at least two with parameter_transfer"
        ),
    )
    parser.add_argument(
        "--preapply-fingerprint-min-proposals-per-gate",
        type=float,
        default=0.0,
        help=(
            "skip fingerprint construction for low-reuse parents unless "
            "their selected proposal count is at least this fraction of "
            "their gate count; zero fingerprints every parent"
        ),
    )
    parser.add_argument(
        "--eliminate-rotation",
        action="store_true",
        help="fold parameter expressions and remove zero rotations after each Quartz rewrite",
    )
    parser.add_argument(
        "--model-apply-binding",
        choices=("auto", "anchor", "direct"),
        default="auto",
        help=(
            "apply model proposals through Quartz's direct ordered-binding API "
            "when available, or force the original anchor rematch for A/B"
        ),
    )
    parser.add_argument(
        "--model-pipeline",
        choices=("compat_host", "state_only_gpu"),
        default="state_only_gpu",
        help=(
            "compat_host preserves the previous empty-action-prefix and Python "
            "proposal path; state_only_gpu encodes only the exact current graph "
            "and selects expanded actions on GPU"
        ),
    )
    args = parser.parse_args()
    if args.deterministic_search:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    if args.mode == "model" and (args.checkpoint is None or args.calibration is None):
        parser.error("model mode requires --checkpoint and --calibration")
    if args.neural_audit_output is not None and (
        args.mode != "model" or args.model_pipeline != "state_only_gpu"
    ):
        parser.error(
            "--neural-audit-output requires model state_only_gpu mode"
        )
    if args.neural_audit_output is not None and (
        args.preapply_fingerprint == "filter"
        or args.preapply_direct_inverse == "filter"
    ):
        parser.error(
            "neural audit collection requires all proposals to reach Quartz; "
            "use off or shadow pre-apply modes"
        )
    if (
        args.neural_audit_output is not None
        and args.dedup_identity != "exact"
    ):
        parser.error("neural audit labels require --dedup-identity exact")
    if args.neural_prefilter_mode != "off" and (
        args.neural_prefilter_checkpoint is None
        or args.mode != "model"
        or args.model_pipeline != "state_only_gpu"
    ):
        parser.error(
            "learned prefiltering requires --neural-prefilter-checkpoint and "
            "model state_only_gpu mode"
        )
    if (
        args.neural_prefilter_checkpoint is not None
        and args.neural_prefilter_mode == "off"
    ):
        parser.error(
            "--neural-prefilter-checkpoint requires shadow or defer mode"
        )
    if args.neural_prefilter_batch_size < 1:
        parser.error("--neural-prefilter-batch-size must be positive")
    if not 0.0 <= args.exploration_fraction < 1.0:
        parser.error("--exploration-fraction must be in [0, 1)")
    if args.exploration_max_stagnation < 1:
        parser.error("--exploration-max-stagnation must be positive")
    if args.exploration_max_detour < 0:
        parser.error("--exploration-max-detour must be nonnegative")
    if not 0 <= args.locality_action_reserve <= args.max_actions_per_parent:
        parser.error(
            "--locality-action-reserve must be within the per-parent action cap"
        )
    if args.survivor_candidate_factor < 1.0:
        parser.error("--survivor-candidate-factor must be at least 1")
    if args.max_total_attempted_actions < 0:
        parser.error("--max-total-attempted-actions must be nonnegative")
    if not 0.0 <= args.widening_revisit_fraction < 1.0:
        parser.error("--widening-revisit-fraction must be in [0, 1)")
    if args.widening_max_expansions < 1:
        parser.error("--widening-max-expansions must be positive")
    if args.widening_min_actions_per_parent < 1:
        parser.error("--widening-min-actions-per-parent must be positive")
    if args.progressive_widening == "on" and (
        args.mode != "model" or args.model_pipeline != "state_only_gpu"
    ):
        parser.error(
            "progressive widening requires model state_only_gpu mode"
        )
    if args.progressive_widening == "on" and args.proposal_ranking == "stochastic":
        parser.error(
            "progressive widening requires a stable gate or probability ranking"
        )
    if args.progressive_widening == "on" and args.locality_action_reserve:
        parser.error(
            "progressive widening cannot be combined with locality action reserve"
        )
    if args.progressive_widening == "on" and args.reference_data is not None:
        parser.error(
            "reference trajectory layers assume one action per search layer; "
            "use candidate audit rather than --reference-data with widening"
        )
    if args.widening_candidate_cache == "on" and args.progressive_widening != "on":
        parser.error("widening candidate cache requires progressive widening")
    if args.widening_candidate_cache == "on" and (
        args.neural_audit_output is not None or args.neural_prefilter_mode != "off"
    ):
        parser.error(
            "widening candidate cache does not yet support neural candidate features"
        )
    if args.apply_profile == "detailed" and args.dedup_identity != "exact":
        parser.error("detailed apply profiling requires exact graph identity")
    if args.transactional_apply == "on" and args.dedup_identity != "exact":
        parser.error("transactional apply requires --dedup-identity exact")
    if (
        args.neural_audit_output is not None
        and args.neural_prefilter_mode != "off"
    ):
        parser.error(
            "neural audit collection and neural prefiltering cannot be combined"
        )
    if (
        args.mode == "model"
        and args.model_pipeline == "state_only_gpu"
        and args.refresh_interval
    ):
        parser.error(
            "state_only_gpu does not use periodic CPU matching; set "
            "--refresh-interval 0"
        )
    if (
        args.preapply_fingerprint == "filter"
        and args.preapply_fingerprint_kind == "topology"
    ):
        parser.error(
            "parameter-blind topology fingerprints are shadow-only; use "
            "--preapply-fingerprint shadow or conservative filtering"
        )
    if args.preapply_fingerprint_min_proposals_per_gate < 0:
        parser.error(
            "--preapply-fingerprint-min-proposals-per-gate must be nonnegative"
        )
    if args.preapply_fingerprint_representatives < 1:
        parser.error("--preapply-fingerprint-representatives must be positive")
    if args.preapply_fingerprint_native_batch_size < 0:
        parser.error(
            "--preapply-fingerprint-native-batch-size must be nonnegative"
        )
    if (
        args.preapply_fingerprint == "filter"
        and args.preapply_fingerprint_kind == "parameter_transfer"
        and args.preapply_fingerprint_representatives < 2
    ):
        parser.error(
            "parameter_transfer filtering requires at least two "
            "representatives per fingerprint"
        )

    # Quartz imports these optional conversion packages unconditionally, while
    # this benchmark uses only the compiled graph API.
    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    reference_hashes = None
    if args.reference_data is not None:
        reference_payload = (
            payload
            if args.reference_data.resolve() == args.data.resolve()
            else torch.load(args.reference_data, map_location="cpu", weights_only=False)
        )
        reference_trajectories = (
            reference_payload["train_trajectories"]
            + reference_payload["test_trajectories"]
        )
        if not reference_trajectories:
            raise RuntimeError("reference payload contains no trajectories")
        reference_trajectory = reference_trajectories[0]
        reference_hashes = [
            int(step["graph_hash"]) for step in reference_trajectory["steps"]
        ] + [int(reference_trajectory["terminal_graph_hash"])]
    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    if context.num_xfers != len(payload["xfer_to_source"]):
        raise RuntimeError("dataset and Quartz context have different xfer counts")
    xfers = [context.get_xfer_from_id(id=index) for index in range(context.num_xfers)]
    for index, xfer in enumerate(xfers):
        if xfer.src_str.strip() != payload["xfer_sources"][index].strip():
            raise RuntimeError(f"xfer source mismatch at {index}")
    source_to_xfers: dict[int, list[int]] = defaultdict(list)
    for xfer_id, source_id in enumerate(payload["xfer_to_source"]):
        source_to_xfers[int(source_id)].append(xfer_id)
    gate_deltas = [int(xfer.dst_gate_count - xfer.src_gate_count) for xfer in xfers]
    source_patterns = tuple(parse_pattern(pattern) for pattern in rules.xfer_sources)
    destination_patterns = tuple(
        parse_pattern(pattern) for pattern in rules.xfer_destinations
    )
    inverse_xfer_ids = rules.unique_inverse_xfer_ids()
    native_fingerprint_kinds = {
        "conservative": 0,
        "parameter_transfer": 1,
        "xfer_guarded": 2,
        "topology": 3,
    }

    model = None
    threshold_config = None
    source_vectors = None
    candidate_source_states = None
    gpu_rule_index = None
    neural_prefilter = None
    neural_prefilter_thresholds = None
    if args.mode == "model":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        train_args = checkpoint["args"]
        model = build_model(
            rules, len(payload["xfer_to_source"]), train_args
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        threshold_config = load_threshold_config(args.calibration, args.target_recall)
        if args.model_pipeline == "state_only_gpu":
            if not hasattr(model, "encode_current_graph"):
                raise ValueError(
                    "checkpoint architecture does not support state-only inference"
                )
            with torch.no_grad(), autocast_context(device):
                candidate_source_states = model.source_representations()
                source_vectors = model.retrieval_source(candidate_source_states)
            gpu_rule_index = GpuRuleIndex.build(
                source_to_xfers,
                gate_deltas,
                model.num_sources,
                args.max_gate_increase,
                device,
            )
    if args.neural_prefilter_mode != "off":
        neural_prefilter, neural_prefilter_thresholds = (
            load_neural_successor_prefilter(
                args.neural_prefilter_checkpoint, device
            )
        )

    graph = quartz.PyGraph.from_qasm(context=context, filename=str(args.qasm))
    guid_direct_binding_available = hasattr(
        graph, "apply_xfer_with_guid_binding"
    )
    node_direct_binding_available = hasattr(
        graph, "apply_xfer_with_node_id_binding"
    )
    direct_binding_available = (
        guid_direct_binding_available or node_direct_binding_available
    )
    if (
        args.mode == "model"
        and args.model_apply_binding == "direct"
        and not direct_binding_available
    ):
        raise RuntimeError(
            "--model-apply-binding direct requires the patched Quartz extension"
        )
    if (
        args.mode == "model"
        and args.model_apply_binding != "anchor"
        and guid_direct_binding_available
    ):
        model_apply_backend = "guid_direct"
    elif (
        args.mode == "model"
        and args.model_apply_binding != "anchor"
        and node_direct_binding_available
    ):
        model_apply_backend = "node_direct"
    else:
        model_apply_backend = "anchor"
    if args.apply_profile == "detailed" and (
        model_apply_backend != "guid_direct"
        or not hasattr(graph, "apply_xfer_with_guid_binding_profiled")
    ):
        raise RuntimeError(
            "detailed apply profiling requires the patched Quartz GUID "
            "direct-binding profiling API"
        )
    if args.transactional_apply == "on" and (
        model_apply_backend != "guid_direct"
        or not hasattr(graph, "apply_xfer_with_guid_binding_transactional")
        or not hasattr(quartz, "PyExactKeyRegistry")
    ):
        raise RuntimeError(
            "transactional apply requires patched Quartz and GUID-direct "
            "model binding"
        )
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    initial_snapshot = snapshot(graph, guid_to_slot)
    use_search_feedback = (
        args.progressive_widening == "on"
        and args.widening_policy == "feedback"
    )
    search_feedback = (
        SearchFeedbackRegistry(exact_graph_key(graph), int(graph.gate_count))
        if use_search_feedback
        else None
    )
    beam = [
        BeamState(
            graph=graph,
            snapshot=initial_snapshot,
            guid_to_slot=guid_to_slot,
            next_slot=next_slot,
            last_touched={},
            rewrite_distance={int(row[0]): 5 for row in initial_snapshot["nodes"]},
            previous_preferred=set(),
            local_streak=0,
            gate_count=int(graph.gate_count),
            depth=0,
            history=(),
            path_best_gate_count=int(graph.gate_count),
            stagnation_steps=0,
            survivor_lane="root",
            exploration_ancestor=False,
            recovered_after_exploration=False,
            search_node_id=(
                search_feedback.root_id if search_feedback is not None else -1
            ),
            search_identity_order=(
                search_feedback.nodes[search_feedback.root_id].identity_order
                if search_feedback is not None
                else ""
            ),
        )
    ]
    if args.mode == "model" and args.model_pipeline == "compat_host":
        model_matches(
            beam,
            model,
            device,
            threshold_config,
            args.microbatch,
            args.max_source_matches,
            paged_action=train_args.get("architecture") == "paged_action",
        )
    elif args.mode == "model":
        warm_candidates, _, _ = state_only_candidate_tensors(
            beam,
            model,
            device,
            threshold_config,
            source_vectors,
            args.microbatch,
            args.max_source_matches,
        )
        build_gpu_proposals(
            warm_candidates,
            beam,
            gpu_rule_index,
            per_parent_cap=max(args.max_actions_per_parent, args.beam_size * 2),
            global_cap=args.beam_size * args.proposal_factor,
            ranking_mode="gate",
        )
    else:
        graph.available_xfers_parallel(context=context, node=graph.nodes[0])
    del payload
    gc.collect()
    gc.disable()
    initial_gate_count = beam[0].gate_count
    if reference_hashes is not None and int(graph.hash()) != reference_hashes[0]:
        raise RuntimeError(
            "input QASM does not match the first state of the reference trajectory"
        )
    best_state = beam[0]
    collect_neural_audit = args.neural_audit_output is not None
    need_candidate_features = (
        collect_neural_audit or args.neural_prefilter_mode != "off"
    )
    widening_candidate_cache = WideningCandidateCache(
        enabled=args.widening_candidate_cache == "on"
    )
    widening_cache_totals = Counter()
    widening_cache_peak_parents = 0
    widening_cache_peak_rows = 0
    widening_cache_peak_bytes = 0
    neural_feature_chunks = []
    neural_outcome_chunks = []
    neural_group_chunks = []
    neural_xfer_chunks = []
    neural_source_chunks = []
    neural_probability_chunks = []
    neural_gate_delta_chunks = []
    neural_parent_gate_chunks = []
    neural_step_chunks = []
    if collect_neural_audit:
        successor_groups = {exact_graph_key(graph): 0}
        next_successor_group = 1
    best_first_seen_step = 0
    best_first_seen_seconds = 0.0
    improvement_trace = [
        {
            "step": 0,
            "gate_count": initial_gate_count,
            "seconds": 0.0,
            "exploration_ancestor": False,
            "recovered_after_exploration": False,
            "widening_ancestor": False,
            "action_depth": 0,
        }
    ]
    seen = (
        ExactGraphRegistry.seeded(graph)
        if args.dedup_identity == "exact"
        else QuartzHashRegistry.seeded(graph)
    )
    transactional_native_registry = None
    if args.transactional_apply == "on":
        transactional_native_registry = quartz.PyExactKeyRegistry()
        if not transactional_native_registry.insert(graph.exact_key()):
            raise RuntimeError("failed to seed native exact-key registry")
    diagnostic_seen_hashes = (
        {int(graph.hash())} if reference_hashes is not None else None
    )
    fingerprint_audit = FingerprintAudit.create(
        mode=args.preapply_fingerprint,
        kind=args.preapply_fingerprint_kind,
        representatives=args.preapply_fingerprint_representatives,
    )
    fingerprint_profile_builds = Counter()
    direct_inverse_totals = {
        "candidates": 0,
        "skipped_before_apply": 0,
        "shadow_exact_duplicates": 0,
        "shadow_novel_successors": 0,
        "shadow_invalid": 0,
    }
    neural_prefilter_totals = {
        "candidates": 0,
        "deferred": 0,
        "scored_seconds": 0.0,
        "deferred_scanned": 0,
        "deferred_invalid": 0,
        "deferred_duplicates": 0,
        "deferred_novel": 0,
    }
    apply_profile_totals = Counter()
    step_rows = []
    reference_retention = []
    total_attempted_actions = 0
    total_started = time.perf_counter()
    for step in range(args.depth):
        step_started = time.perf_counter()
        input_beam = beam
        reference_parent_indices = []
        reference_target_seen_before = False
        if reference_hashes is not None and step + 1 < len(reference_hashes):
            reference_parent_indices = [
                index
                for index, state in enumerate(input_beam)
                if int(state.graph.hash()) == reference_hashes[step]
            ]
            reference_target_seen_before = (
                reference_hashes[step + 1] in diagnostic_seen_hashes
            )
        model_seconds = exact_seconds = proposal_seconds = 0.0
        parent_feedback_step = [Counter() for _ in input_beam]
        parent_best_child_gate: list[int | None] = [None] * len(input_beam)
        exact_refresh_actions_added = 0
        action_rows = None
        source_binding_candidates = 0
        proposal_metrics = {}
        collation_metrics = {}
        proposal_feature_rows = None
        selected_proposal_tensors = None
        parent_candidate_rows = None
        widening_cache_step: dict[str, Any] = {
            "enabled": widening_candidate_cache.enabled
        }
        if args.mode == "model" and args.model_pipeline == "state_only_gpu":
            if widening_candidate_cache.enabled:
                miss_indices = widening_candidate_cache.miss_indices(beam)
                fresh_candidates = None
                if miss_indices:
                    fresh_candidates, model_seconds, collation_metrics = (
                        state_only_candidate_tensors(
                            [beam[index] for index in miss_indices],
                            model,
                            device,
                            threshold_config,
                            source_vectors,
                            args.microbatch,
                            args.max_source_matches,
                        )
                    )
                else:
                    collation_metrics = {
                        "live_nodes": 0,
                        "padded_dense_slots": 0,
                        "padded_persistent_slots": 0,
                        "max_dense_slots": 0,
                        "max_persistent_slots": 0,
                    }
                candidates, parent_candidate_rows, widening_cache_step = (
                    widening_candidate_cache.resolve(
                        beam,
                        miss_indices=miss_indices,
                        fresh_candidates=fresh_candidates,
                    )
                )
                encoded_context = None
            else:
                candidate_result = state_only_candidate_tensors(
                    beam,
                    model,
                    device,
                    threshold_config,
                    source_vectors,
                    args.microbatch,
                    args.max_source_matches,
                    return_encoded_states=need_candidate_features,
                )
                if need_candidate_features:
                    (
                        candidates,
                        model_seconds,
                        collation_metrics,
                        encoded_context,
                    ) = candidate_result
                else:
                    candidates, model_seconds, collation_metrics = candidate_result
                    encoded_context = None
            source_binding_candidates = int(candidates.sources.numel())
            effective_parent_cap = (
                args.max_actions_per_parent
                if args.progressive_widening == "on"
                else max(
                    args.max_actions_per_parent,
                    math.ceil(args.beam_size / max(1, len(beam))) * 2,
                )
            )
            parent_rank_offsets = (
                [
                    int(state.expansion_round) * args.max_actions_per_parent
                    for state in beam
                ]
                if args.progressive_widening == "on"
                else None
            )
            proposal_started = time.perf_counter()
            (
                proposals,
                proposal_metrics,
                _,
                selected_proposal_tensors,
            ) = build_gpu_proposals(
                candidates,
                beam,
                gpu_rule_index,
                per_parent_cap=effective_parent_cap,
                global_cap=args.beam_size * args.proposal_factor,
                ranking_mode=args.proposal_ranking,
                ranking_seed=args.proposal_ranking_seed + step,
                locality_action_reserve=min(
                    args.locality_action_reserve, effective_parent_cap
                ),
                parent_rank_offsets=parent_rank_offsets,
                preserve_parent_best=args.progressive_widening == "on",
                parent_diversity_actions=(
                    args.widening_min_actions_per_parent
                    if args.progressive_widening == "on"
                    else 1
                ),
                return_selected_tensors=need_candidate_features,
            )
            proposal_seconds = time.perf_counter() - proposal_started
            if proposals is None:
                raise RuntimeError("state-only GPU proposal materialization failed")
            if need_candidate_features:
                if selected_proposal_tensors is None or encoded_context is None:
                    raise RuntimeError("neural candidate feature context is missing")
                encoded_states, encoded_live = encoded_context
                with torch.no_grad(), autocast_context(device):
                    proposal_feature_rows = frozen_candidate_features(
                        model,
                        encoded_states,
                        encoded_live,
                        selected_proposal_tensors,
                        candidate_source_states,
                    )
                if collect_neural_audit:
                    proposal_feature_rows = (
                        proposal_feature_rows.to(torch.float16).cpu()
                    )
            total_action_candidates = int(proposal_metrics["eligible_actions"])
            matched_action_count = int(proposal_metrics["predicted_actions"])
        elif args.mode == "model":
            predicted, model_seconds = model_matches(
                beam,
                model,
                device,
                threshold_config,
                args.microbatch,
                args.max_source_matches,
                paged_action=train_args.get("architecture") == "paged_action",
            )
            action_rows: list[
                list[tuple[int, int, tuple[int, ...] | None, float]]
            ] = []
            for rows in predicted:
                expanded = []
                for source, anchor, binding, probability in rows:
                    expanded.extend(
                        (xfer_id, anchor, binding, probability)
                        for xfer_id in source_to_xfers[source]
                    )
                action_rows.append(expanded)
            refresh = (
                args.refresh_interval > 0
                and step > 0
                and step % args.refresh_interval == 0
            )
            if refresh:
                for parent_index in range(min(args.refresh_count, len(beam))):
                    started = time.perf_counter()
                    exact_rows = exact_actions(beam[parent_index], context)
                    exact_seconds += time.perf_counter() - started
                    # Keep the model's ranking for candidates it already found.
                    # Exact refresh only adds missed xfer-at-anchor actions, at
                    # lower tie-break priority within the same gate-count band.
                    present = {
                        (int(xfer_id), int(anchor))
                        for xfer_id, anchor, _, _ in action_rows[parent_index]
                    }
                    additions = [
                        (xfer_id, anchor, binding, 0.0)
                        for xfer_id, anchor, binding, _ in exact_rows
                        if (int(xfer_id), int(anchor)) not in present
                    ]
                    exact_refresh_actions_added += len(additions)
                    action_rows[parent_index].extend(additions)
        else:
            action_rows = []
            for state in beam:
                started = time.perf_counter()
                action_rows.append(exact_actions(state, context))
                exact_seconds += time.perf_counter() - started

        if action_rows is not None:
            proposal_started = time.perf_counter()
            proposals, total_action_candidates = rank_proposals(
                beam,
                action_rows,
                gate_deltas,
                beam_size=args.beam_size,
                max_actions_per_parent=args.max_actions_per_parent,
                proposal_factor=args.proposal_factor,
                max_gate_increase=args.max_gate_increase,
                ranking_mode=args.proposal_ranking,
                ranking_seed=args.proposal_ranking_seed + step,
            )
            proposal_seconds = time.perf_counter() - proposal_started
            matched_action_count = sum(map(len, action_rows))
            source_binding_candidates = (
                sum(map(len, predicted)) if args.mode == "model" else 0
            )

        neural_prefilter_step = {
            "mode": args.neural_prefilter_mode,
            "candidates": 0,
            "deferred": 0,
            "scored_seconds": 0.0,
            "deferred_scanned": 0,
            "deferred_invalid": 0,
            "deferred_duplicates": 0,
            "deferred_novel": 0,
        }
        proposal_is_deferred = [False] * len(proposals)
        if args.neural_prefilter_mode != "off":
            if proposal_feature_rows is None or selected_proposal_tensors is None:
                raise RuntimeError(
                    "learned prefilter proposal features are missing: "
                    f"features={proposal_feature_rows is not None}, "
                    f"tensors={selected_proposal_tensors is not None}, "
                    f"need={need_candidate_features}"
                )
            neural_started = time.perf_counter()
            valid_scores, duplicate_scores = neural_prefilter_scores(
                neural_prefilter,
                proposal_feature_rows,
                selected_proposal_tensors,
                beam,
                step,
                device,
                args.neural_prefilter_batch_size,
            )
            defer_mask = valid_scores.lt(
                neural_prefilter_thresholds["valid"]
            ) | duplicate_scores.gt(neural_prefilter_thresholds["duplicate"])
            neural_seconds = time.perf_counter() - neural_started
            deferred = int(defer_mask.sum())
            neural_prefilter_step.update(
                {
                    "candidates": len(proposals),
                    "deferred": deferred,
                    "scored_seconds": neural_seconds,
                }
            )
            neural_prefilter_totals["candidates"] += len(proposals)
            neural_prefilter_totals["deferred"] += deferred
            neural_prefilter_totals["scored_seconds"] += neural_seconds
            mask_rows = defer_mask.cpu().tolist()
            proposal_is_deferred = mask_rows
            if args.neural_prefilter_mode == "defer" and deferred:
                order = [
                    index for index, is_deferred in enumerate(mask_rows)
                    if not is_deferred
                ] + [
                    index for index, is_deferred in enumerate(mask_rows)
                    if is_deferred
                ]
                proposals = [proposals[index] for index in order]
                proposal_is_deferred = [mask_rows[index] for index in order]

        fingerprint_before = fingerprint_audit.stats()
        fingerprint_profiles = {}
        native_fingerprint_cache = {}
        minimum_fingerprint_reuse = (
            args.preapply_fingerprint_min_proposals_per_gate
        )
        proposals_per_parent = Counter(
            proposal.parent for proposal in proposals
        )
        fingerprint_seconds = 0.0
        direct_inverse_step = {
            key: 0 for key in direct_inverse_totals
        }
        apply_started = time.perf_counter()
        children = []
        attempted = invalid = duplicates = proposals_scanned = 0
        scanned_parent_ranks = []
        attempted_parent_ranks = []
        scanned_widened_actions = 0
        attempted_widened_actions = 0
        applied_proposal_positions = set()
        fingerprint_skipped_positions = set()
        direct_inverse_skipped_positions = set()
        neural_step_outcomes = []
        neural_step_groups = []
        apply_profile_step = Counter()
        detailed_profile = (
            apply_profile_step if args.apply_profile == "detailed" else None
        )

        def get_fingerprint_profile(parent_index: int):
            if parent_index in fingerprint_profiles:
                return fingerprint_profiles[parent_index]
            parent_state = beam[parent_index]
            profile_started = (
                time.perf_counter_ns() if detailed_profile is not None else 0
            )
            native_builder = getattr(
                parent_state.graph,
                "successor_fingerprint_profile",
                None,
            )
            if (
                args.preapply_fingerprint_backend != "python"
                and native_builder is not None
            ):
                profile = native_builder(parent_state.snapshot["nodes"])
                profile_backend = "native"
            elif args.preapply_fingerprint_backend == "native":
                raise RuntimeError(
                    "native successor fingerprinting was requested, but "
                    "the loaded Quartz extension does not provide it"
                )
            else:
                profile = build_wire_trace_profile(
                    parent_state.graph,
                    parent_state.guid_to_slot,
                )
                profile_backend = "python"
            fingerprint_profiles[parent_index] = (profile_backend, profile)
            fingerprint_profile_builds[profile_backend] += 1
            if detailed_profile is not None:
                add_profile_ns(
                    detailed_profile,
                    f"fingerprint_{profile_backend}_profile_build",
                    profile_started,
                )
                detailed_profile[
                    f"fingerprint_{profile_backend}_profile_build_count"
                ] += 1
            return profile_backend, profile

        candidate_capacity = (
            args.beam_size
            if args.survivor_policy == "gate"
            else max(
                args.beam_size,
                math.ceil(args.beam_size * args.survivor_candidate_factor),
            )
        )
        for proposal_position, proposal in enumerate(proposals):
            if len(children) >= candidate_capacity:
                break
            if (
                args.max_total_attempted_actions
                and total_attempted_actions >= args.max_total_attempted_actions
            ):
                break
            proposals_scanned += 1
            parent_feedback_step[proposal.parent]["scanned"] += 1
            if proposal.parent_rank >= 0:
                scanned_parent_ranks.append(proposal.parent_rank)
            if beam[proposal.parent].expansion_round > 0:
                scanned_widened_actions += 1
            neural_deferred = proposal_is_deferred[proposal_position]
            fingerprint = None
            direct_inverse = (
                args.preapply_direct_inverse != "off"
                and is_direct_inverse_proposal(
                    beam[proposal.parent], proposal, inverse_xfer_ids
                )
            )
            if direct_inverse:
                direct_inverse_step["candidates"] += 1
                direct_inverse_totals["candidates"] += 1
                if args.preapply_direct_inverse == "filter":
                    direct_inverse_step["skipped_before_apply"] += 1
                    direct_inverse_totals["skipped_before_apply"] += 1
                    direct_inverse_skipped_positions.add(proposal_position)
                    continue
            if (
                args.preapply_fingerprint != "off"
                and proposal.binding is not None
            ):
                fingerprint_started = time.perf_counter()
                minimum_parent_proposals = max(
                    1,
                    math.ceil(
                        beam[proposal.parent].gate_count
                        * args.preapply_fingerprint_min_proposals_per_gate
                    ),
                )
                if (
                    proposals_per_parent[proposal.parent]
                    < minimum_parent_proposals
                ):
                    fingerprint_audit.observe_bypassed()
                    fingerprint_seconds += (
                        time.perf_counter() - fingerprint_started
                    )
                else:
                    profile_backend, profile = get_fingerprint_profile(
                        proposal.parent
                    )
                    if profile is not None:
                        if profile_backend == "native":
                            if (
                                args.preapply_fingerprint_native_batch_size
                                and proposal_position
                                not in native_fingerprint_cache
                            ):
                                batch_end = min(
                                    len(proposals),
                                    proposal_position
                                    + args.preapply_fingerprint_native_batch_size,
                                )
                                grouped_positions = defaultdict(list)
                                for batch_position in range(
                                    proposal_position, batch_end
                                ):
                                    if (
                                        batch_position
                                        in native_fingerprint_cache
                                    ):
                                        continue
                                    batch_proposal = proposals[batch_position]
                                    if batch_proposal.binding is None:
                                        continue
                                    minimum_batch_parent_proposals = max(
                                        1,
                                        math.ceil(
                                            beam[
                                                batch_proposal.parent
                                            ].gate_count
                                            * minimum_fingerprint_reuse
                                        ),
                                    )
                                    if (
                                        proposals_per_parent[
                                            batch_proposal.parent
                                        ]
                                        < minimum_batch_parent_proposals
                                    ):
                                        continue
                                    grouped_positions[
                                        batch_proposal.parent
                                    ].append(batch_position)
                                for batch_parent, positions in (
                                    grouped_positions.items()
                                ):
                                    batch_backend, batch_profile = (
                                        get_fingerprint_profile(batch_parent)
                                    )
                                    if (
                                        batch_backend != "native"
                                        or batch_profile is None
                                    ):
                                        continue
                                    batch_method = getattr(
                                        batch_profile,
                                        "successor_fingerprints",
                                        None,
                                    )
                                    if batch_method is None:
                                        continue
                                    batch_proposals = [
                                        proposals[position]
                                        for position in positions
                                    ]
                                    compute_started = (
                                        time.perf_counter_ns()
                                        if detailed_profile is not None
                                        else 0
                                    )
                                    batch_fingerprints = batch_method(
                                        [
                                            xfers[row.xfer_id]
                                            for row in batch_proposals
                                        ],
                                        [
                                            row.binding
                                            for row in batch_proposals
                                        ],
                                        [
                                            row.xfer_id
                                            for row in batch_proposals
                                        ],
                                        native_fingerprint_kinds[
                                            args.preapply_fingerprint_kind
                                        ],
                                    )
                                    if detailed_profile is not None:
                                        add_profile_ns(
                                            detailed_profile,
                                            "fingerprint_native_compute",
                                            compute_started,
                                        )
                                        detailed_profile[
                                            "fingerprint_native_compute_count"
                                        ] += len(batch_proposals)
                                    for position, batch_fingerprint in zip(
                                        positions, batch_fingerprints
                                    ):
                                        native_fingerprint_cache[position] = (
                                            batch_fingerprint
                                        )
                            if proposal_position in native_fingerprint_cache:
                                native_fingerprint = native_fingerprint_cache[
                                    proposal_position
                                ]
                            else:
                                compute_started = (
                                    time.perf_counter_ns()
                                    if detailed_profile is not None
                                    else 0
                                )
                                native_fingerprint = profile.successor_fingerprint(
                                    xfer=xfers[proposal.xfer_id],
                                    source_slots=proposal.binding,
                                    xfer_id=proposal.xfer_id,
                                    kind=native_fingerprint_kinds[
                                        args.preapply_fingerprint_kind
                                    ],
                                )
                                if detailed_profile is not None:
                                    add_profile_ns(
                                        detailed_profile,
                                        "fingerprint_native_compute",
                                        compute_started,
                                    )
                                    detailed_profile[
                                        "fingerprint_native_compute_count"
                                    ] += 1
                            if native_fingerprint is not None:
                                fingerprint = (
                                    "quartz_native_successor_v1",
                                    *native_fingerprint,
                                )
                        else:
                            compute_started = (
                                time.perf_counter_ns()
                                if detailed_profile is not None
                                else 0
                            )
                            fingerprint = successor_fingerprint(
                                profile,
                                source_patterns[proposal.xfer_id],
                                destination_patterns[proposal.xfer_id],
                                proposal.binding,
                                xfer_id=proposal.xfer_id,
                                kind=args.preapply_fingerprint_kind,
                            )
                            if detailed_profile is not None:
                                add_profile_ns(
                                    detailed_profile,
                                    "fingerprint_python_compute",
                                    compute_started,
                                )
                                detailed_profile[
                                    "fingerprint_python_compute_count"
                                ] += 1
                    registry_started = (
                        time.perf_counter_ns()
                        if detailed_profile is not None
                        else 0
                    )
                    should_skip = fingerprint_audit.should_skip(fingerprint)
                    if detailed_profile is not None:
                        add_profile_ns(
                            detailed_profile,
                            "fingerprint_registry",
                            registry_started,
                        )
                        detailed_profile["fingerprint_registry_count"] += 1
                    fingerprint_seconds += (
                        time.perf_counter() - fingerprint_started
                    )
                    if should_skip:
                        fingerprint_skipped_positions.add(proposal_position)
                        continue
            attempted += 1
            total_attempted_actions += 1
            parent_feedback_step[proposal.parent]["attempted"] += 1
            if proposal.parent_rank >= 0:
                attempted_parent_ranks.append(proposal.parent_rank)
            if beam[proposal.parent].expansion_round > 0:
                attempted_widened_actions += 1
            applied_proposal_positions.add(proposal_position)
            if neural_deferred:
                neural_prefilter_step["deferred_scanned"] += 1
                neural_prefilter_totals["deferred_scanned"] += 1
            applied = apply_rewrite(
                beam[proposal.parent],
                proposal,
                xfers,
                eliminate_rotation=args.eliminate_rotation,
                binding_backend=model_apply_backend,
                profile=detailed_profile,
                transactional_registry=transactional_native_registry,
            )
            if applied is None:
                invalid += 1
                parent_feedback_step[proposal.parent]["invalid"] += 1
                if neural_deferred:
                    neural_prefilter_step["deferred_invalid"] += 1
                    neural_prefilter_totals["deferred_invalid"] += 1
                if collect_neural_audit:
                    neural_step_outcomes.append(0)
                    neural_step_groups.append(-1)
                fingerprint_audit.observe_invalid(fingerprint)
                if direct_inverse and args.preapply_direct_inverse == "shadow":
                    direct_inverse_step["shadow_invalid"] += 1
                    direct_inverse_totals["shadow_invalid"] += 1
                continue
            if applied.exact_identity is not None:
                exact_identity = applied.exact_identity
            elif detailed_profile is not None:
                identity_started = time.perf_counter_ns()
                exact_identity = exact_graph_key(applied.graph)
                add_profile_ns(
                    detailed_profile, "exact_graph_key", identity_started
                )
                detailed_profile["exact_graph_key_count"] += 1
            else:
                exact_identity = (
                    exact_graph_key(applied.graph)
                    if collect_neural_audit
                    or use_search_feedback
                    or (
                        args.preapply_fingerprint == "shadow"
                        and fingerprint is not None
                    )
                    else None
                )
            if diagnostic_seen_hashes is not None and applied.graph is not None:
                diagnostic_seen_hashes.add(int(applied.graph.hash()))
            if applied.transaction_status is not None:
                is_new_successor = seen.record_prechecked_native_key(
                    exact_identity,
                    is_new=applied.transaction_status == 0,
                )
            elif detailed_profile is not None:
                registry_started = time.perf_counter_ns()
                is_new_successor = seen.register_native_key(exact_identity)
                add_profile_ns(
                    detailed_profile, "exact_registry", registry_started
                )
                detailed_profile["exact_registry_count"] += 1
            else:
                is_new_successor = seen.register(applied.graph)
            if collect_neural_audit:
                successor_group = successor_groups.get(exact_identity)
                if successor_group is None:
                    successor_group = next_successor_group
                    successor_groups[exact_identity] = successor_group
                    next_successor_group += 1
                neural_step_outcomes.append(2 if is_new_successor else 1)
                neural_step_groups.append(successor_group)
            audit_started = (
                time.perf_counter_ns() if detailed_profile is not None else 0
            )
            fingerprint_audit.observe_valid(fingerprint, exact_identity)
            if detailed_profile is not None:
                add_profile_ns(
                    detailed_profile,
                    "post_apply_fingerprint_audit",
                    audit_started,
                )
            if direct_inverse and args.preapply_direct_inverse == "shadow":
                inverse_result = (
                    "shadow_novel_successors"
                    if is_new_successor
                    else "shadow_exact_duplicates"
                )
                direct_inverse_step[inverse_result] += 1
                direct_inverse_totals[inverse_result] += 1
            if not is_new_successor:
                duplicates += 1
                parent_feedback_step[proposal.parent]["valid"] += 1
                parent_feedback_step[proposal.parent]["duplicate"] += 1
                if (
                    search_feedback is not None
                    and search_feedback.has_identity(exact_identity)
                ):
                    search_feedback.add_parent_edge(
                        exact_identity,
                        beam[proposal.parent].search_node_id,
                        step=step + 1,
                    )
                if neural_deferred:
                    neural_prefilter_step["deferred_duplicates"] += 1
                    neural_prefilter_totals["deferred_duplicates"] += 1
                continue
            if neural_deferred:
                neural_prefilter_step["deferred_novel"] += 1
                neural_prefilter_totals["deferred_novel"] += 1
            child = materialize_child(
                beam[proposal.parent],
                proposal,
                applied,
                eliminate_rotation=args.eliminate_rotation,
                profile=detailed_profile,
            )
            if search_feedback is not None:
                child.search_node_id = search_feedback.add_node(
                    exact_identity,
                    gate_count=child.gate_count,
                    depth=child.depth,
                    parent_id=beam[proposal.parent].search_node_id,
                    step=step + 1,
                )
                child.search_identity_order = search_feedback.nodes[
                    child.search_node_id
                ].identity_order
            parent_feedback_step[proposal.parent]["valid"] += 1
            parent_feedback_step[proposal.parent]["unique"] += 1
            if child.gate_count < beam[proposal.parent].gate_count:
                parent_feedback_step[proposal.parent]["improving"] += 1
            previous_best = parent_best_child_gate[proposal.parent]
            if previous_best is None or child.gate_count < previous_best:
                parent_best_child_gate[proposal.parent] = child.gate_count
            children.append(child)
        apply_seconds = time.perf_counter() - apply_started
        if search_feedback is not None:
            for parent_index, state in enumerate(input_beam):
                counters = parent_feedback_step[parent_index]
                if not counters["scanned"]:
                    continue
                search_feedback.observe_expansion(
                    state.search_node_id,
                    attempted=counters["attempted"],
                    valid=counters["valid"],
                    unique=counters["unique"],
                    duplicate=counters["duplicate"],
                    invalid=counters["invalid"],
                    improving=counters["improving"],
                    best_child_gate=parent_best_child_gate[parent_index],
                    step=step + 1,
                )
        if detailed_profile is not None:
            apply_profile_totals.update(apply_profile_step)
        if collect_neural_audit:
            if proposal_feature_rows is None or selected_proposal_tensors is None:
                raise RuntimeError("neural audit proposal features are missing")
            if len(neural_step_outcomes) != proposals_scanned:
                raise RuntimeError(
                    "neural audit labels do not align with scanned proposals"
                )
            selected_rows = slice(0, proposals_scanned)
            neural_feature_chunks.append(proposal_feature_rows[selected_rows])
            neural_outcome_chunks.append(
                torch.tensor(neural_step_outcomes, dtype=torch.int8)
            )
            neural_group_chunks.append(
                torch.tensor(neural_step_groups, dtype=torch.long)
            )
            neural_xfer_chunks.append(
                selected_proposal_tensors.xfer_ids[selected_rows].cpu()
            )
            neural_source_chunks.append(
                selected_proposal_tensors.source_ids[selected_rows].cpu()
            )
            neural_probability_chunks.append(
                selected_proposal_tensors.probabilities[selected_rows]
                .to(torch.float16)
                .cpu()
            )
            neural_gate_delta_chunks.append(
                selected_proposal_tensors.gate_deltas[selected_rows]
                .to(torch.int16)
                .cpu()
            )
            parent_rows = selected_proposal_tensors.parent_ids[
                selected_rows
            ].cpu()
            neural_parent_gate_chunks.append(
                torch.tensor(
                    [state.gate_count for state in input_beam],
                    dtype=torch.int16,
                ).index_select(0, parent_rows)
            )
            neural_step_chunks.append(
                torch.full(
                    (proposals_scanned,), step + 1, dtype=torch.int16
                )
            )
        fingerprint_after = fingerprint_audit.stats()
        fingerprint_step = {
            key: fingerprint_after[key] - fingerprint_before[key]
            for key in (
                "candidates",
                "unavailable",
                "bypassed_low_reuse",
                "hits",
                "skipped_before_apply",
                "shadow_valid_hits",
                "shadow_invalid_hits",
                "shadow_exact_duplicate_hits",
                "shadow_collision_hits",
            )
        }
        audited_fingerprint_hits = (
            fingerprint_step["shadow_exact_duplicate_hits"]
            + fingerprint_step["shadow_collision_hits"]
        )
        fingerprint_step.update(
            {
                "mode": args.preapply_fingerprint,
                "kind": args.preapply_fingerprint_kind,
                "seconds": fingerprint_seconds,
                "shadow_precision": (
                    fingerprint_step["shadow_exact_duplicate_hits"]
                    / audited_fingerprint_hits
                    if audited_fingerprint_hits
                    else 1.0
                ),
            }
        )
        if args.progressive_widening == "on":
            revisit_target = int(
                math.floor(args.beam_size * args.widening_revisit_fraction)
            )
            widening_selection = select_widening_revisits(
                input_beam,
                slots=revisit_target,
                max_expansions=args.widening_max_expansions,
                seed=args.widening_seed,
                step=step + 1,
                policy=args.widening_policy,
                feedback=(
                    search_feedback.nodes
                    if search_feedback is not None
                    else None
                ),
            )
            widening_revisits = [
                replace(
                    input_beam[index],
                    expansion_round=input_beam[index].expansion_round + 1,
                    survivor_lane="widening",
                )
                for index in widening_selection.indices
            ]
            if widening_candidate_cache.enabled:
                if parent_candidate_rows is None:
                    raise RuntimeError("widening cache is missing per-parent candidates")
                widening_cache_step.update(
                    widening_candidate_cache.retain(
                        input_beam,
                        parent_candidate_rows,
                        widening_selection.indices,
                    )
                )
                for name in (
                    "parent_hits",
                    "parent_misses",
                    "candidate_rows_reused",
                    "candidate_rows_generated",
                ):
                    widening_cache_totals[name] += int(widening_cache_step[name])
                widening_cache_peak_parents = max(
                    widening_cache_peak_parents,
                    int(widening_cache_step["resident_parents_after"]),
                )
                widening_cache_peak_rows = max(
                    widening_cache_peak_rows,
                    int(widening_cache_step["resident_rows_after"]),
                )
                widening_cache_peak_bytes = max(
                    widening_cache_peak_bytes,
                    int(widening_cache_step["resident_bytes_after"]),
                )
        else:
            widening_revisits = []
            widening_selection = select_widening_revisits(
                [], slots=0, max_expansions=1, step=step + 1
            )
        if not children and not widening_revisits:
            break
        reference_candidate_indices = []
        if reference_hashes is not None and step + 1 < len(reference_hashes):
            expected_hash = reference_hashes[step + 1]
            reference_candidate_indices = [
                index
                for index, state in enumerate(children)
                if int(state.graph.hash()) == expected_hash
            ]
        child_beam_capacity = args.beam_size - len(widening_revisits)
        survivor_selection = select_survivors(
            children,
            beam_size=child_beam_capacity,
            exploration_fraction=(
                args.exploration_fraction
                if args.survivor_policy == "dual_lane"
                else 0.0
            ),
            exploration_max_stagnation=args.exploration_max_stagnation,
            exploration_max_detour=args.exploration_max_detour,
            seed=args.exploration_seed,
            step=step + 1,
        )
        selected_children = survivor_selection.states
        beam = selected_children + widening_revisits
        beam.sort(key=gate_priority)
        reference_lost = False
        if reference_hashes is not None and step + 1 < len(reference_hashes):
            expected_hash = reference_hashes[step + 1]
            retained_indices = [
                index
                for index, state in enumerate(beam)
                if int(state.graph.hash()) == expected_hash
            ]
            reference_lost = not retained_indices
            retention_row = {
                "step": step + 1,
                "expected_quartz_hash": expected_hash,
                "parent_beam_indices": reference_parent_indices,
                "target_seen_before_layer": reference_target_seen_before,
                "candidate_indices_before_survivor": reference_candidate_indices,
                "retained": not reference_lost,
                "beam_indices": retained_indices,
                "retained_lanes": [
                    beam[index].survivor_lane for index in retained_indices
                ],
            }
            if (
                reference_lost
                and reference_parent_indices
                and action_rows is not None
            ):
                def proposal_key(proposal):
                    return (
                        int(proposal.parent),
                        int(proposal.xfer_id),
                        int(proposal.anchor_slot),
                        None
                        if proposal.binding is None
                        else tuple(map(int, proposal.binding)),
                    )

                matching_candidates = []
                for parent_index in reference_parent_indices:
                    state = input_beam[parent_index]
                    for xfer_id, anchor, binding, probability in action_rows[
                        parent_index
                    ]:
                        if gate_deltas[xfer_id] > args.max_gate_increase:
                            continue
                        candidate = Proposal(
                            parent=parent_index,
                            xfer_id=xfer_id,
                            anchor_slot=anchor,
                            binding=binding,
                            probability=probability,
                            next_gate_count=state.gate_count + gate_deltas[xfer_id],
                        )
                        candidate_result = apply_rewrite(
                            state,
                            candidate,
                            xfers,
                            eliminate_rotation=args.eliminate_rotation,
                            binding_backend=model_apply_backend,
                        )
                        if (
                            candidate_result is not None
                            and int(candidate_result.graph.hash()) == expected_hash
                        ):
                            matching_candidates.append(candidate)

                effective_parent_cap = max(
                    args.max_actions_per_parent,
                    math.ceil(args.beam_size / max(1, len(input_beam))) * 2,
                )
                parent_selected_keys = set()
                for parent_index in reference_parent_indices:
                    state = input_beam[parent_index]
                    eligible_rows = [
                        row
                        for row in action_rows[parent_index]
                        if gate_deltas[row[0]] <= args.max_gate_increase
                    ]
                    selected_rows = heapq.nsmallest(
                        effective_parent_cap,
                        eligible_rows,
                        key=lambda row: (
                            state.gate_count + gate_deltas[row[0]],
                            -row[3],
                            row[0],
                        ),
                    )
                    parent_selected_keys.update(
                        (
                            parent_index,
                            int(xfer_id),
                            int(anchor),
                            None if binding is None else tuple(map(int, binding)),
                        )
                        for xfer_id, anchor, binding, _ in selected_rows
                    )
                global_positions = {
                    proposal_key(proposal): index
                    for index, proposal in enumerate(proposals)
                }
                matching_keys = [proposal_key(row) for row in matching_candidates]
                parent_selected_matches = [
                    key for key in matching_keys if key in parent_selected_keys
                ]
                global_match_positions = [
                    global_positions[key]
                    for key in parent_selected_matches
                    if key in global_positions
                ]
                attempted_match_positions = [
                    index
                    for index in global_match_positions
                    if index in applied_proposal_positions
                ]
                if not matching_candidates:
                    exclusion_stage = "predicted_candidates"
                elif not parent_selected_matches:
                    exclusion_stage = "per_parent_cap"
                elif not global_match_positions:
                    exclusion_stage = "global_proposal_cap"
                elif any(
                    index in direct_inverse_skipped_positions
                    for index in global_match_positions
                ):
                    exclusion_stage = "preapply_direct_inverse"
                elif any(
                    index in fingerprint_skipped_positions
                    for index in global_match_positions
                ):
                    exclusion_stage = "preapply_fingerprint"
                elif not attempted_match_positions:
                    exclusion_stage = "beam_filled_before_proposal"
                elif reference_candidate_indices:
                    exclusion_stage = "survivor_selection"
                elif reference_target_seen_before:
                    exclusion_stage = "global_exact_dedup"
                else:
                    exclusion_stage = "attempted_but_not_retained"
                retention_row["loss_detail"] = {
                    "exclusion_stage": exclusion_stage,
                    "matching_predicted_candidates": len(matching_candidates),
                    "matching_parent_selected_candidates": len(parent_selected_matches),
                    "matching_global_proposal_positions": global_match_positions,
                    "matching_attempted_proposal_positions": attempted_match_positions,
                    "attempted_proposals": attempted,
                    "proposals_scanned": proposals_scanned,
                }
            elif reference_lost and reference_parent_indices:
                matching_positions = []
                for proposal_index, candidate in enumerate(proposals):
                    if candidate.parent not in reference_parent_indices:
                        continue
                    candidate_result = apply_rewrite(
                        input_beam[candidate.parent],
                        candidate,
                        xfers,
                        eliminate_rotation=args.eliminate_rotation,
                        binding_backend=model_apply_backend,
                    )
                    if (
                        candidate_result is not None
                        and int(candidate_result.graph.hash()) == expected_hash
                    ):
                        matching_positions.append(proposal_index)
                attempted_positions = [
                    index
                    for index in matching_positions
                    if index in applied_proposal_positions
                ]
                retention_row["loss_detail"] = {
                    "exclusion_stage": (
                        "gpu_match_or_proposal_cap"
                        if not matching_positions
                        else "preapply_direct_inverse"
                        if any(
                            index in direct_inverse_skipped_positions
                            for index in matching_positions
                        )
                        else "preapply_fingerprint"
                        if any(
                            index in fingerprint_skipped_positions
                            for index in matching_positions
                        )
                        else "beam_filled_before_proposal"
                        if not attempted_positions
                        else "survivor_selection"
                        if reference_candidate_indices
                        else "global_exact_dedup"
                        if reference_target_seen_before
                        else "attempted_but_not_retained"
                    ),
                    "matching_global_proposal_positions": matching_positions,
                    "matching_attempted_proposal_positions": attempted_positions,
                    "attempted_proposals": attempted,
                    "proposals_scanned": proposals_scanned,
                }
            reference_retention.append(retention_row)
        elapsed = time.perf_counter() - step_started
        cumulative_seconds = time.perf_counter() - total_started
        if beam[0].gate_count < best_state.gate_count:
            best_state = beam[0]
            best_first_seen_step = step + 1
            best_first_seen_seconds = cumulative_seconds
            improvement_trace.append(
                {
                    "step": best_first_seen_step,
                    "gate_count": best_state.gate_count,
                    "seconds": best_first_seen_seconds,
                    "exploration_ancestor": bool(
                        best_state.exploration_ancestor
                    ),
                    "recovered_after_exploration": bool(
                        best_state.recovered_after_exploration
                    ),
                    "widening_ancestor": bool(best_state.widening_ancestor),
                    "action_depth": int(best_state.depth),
                    "last_action_parent_rank": int(
                        best_state.last_action_parent_rank
                    ),
                }
            )
        match_seconds = model_seconds + exact_seconds
        successful_applies = attempted - invalid
        row = {
            "step": step + 1,
            "input_states": len(input_beam),
            "input_max_action_depth": max(
                (state.depth for state in input_beam), default=0
            ),
            "input_survivor_lanes": dict(
                Counter(state.survivor_lane for state in input_beam)
            ),
            "output_states": len(beam),
            "output_max_action_depth": max(
                (state.depth for state in beam), default=0
            ),
            "output_survivor_lanes": dict(
                Counter(state.survivor_lane for state in beam)
            ),
            "best_gate_count": beam[0].gate_count,
            "global_best_gate_count": best_state.gate_count,
            "cumulative_seconds": cumulative_seconds,
            "predicted_or_exact_actions": matched_action_count,
            "source_binding_candidates": source_binding_candidates,
            "eligible_actions_before_parent_cap": total_action_candidates,
            "proposals_after_caps": len(proposals),
            "proposals_scanned": proposals_scanned,
            "attempted_actions": attempted,
            "accepted_actions": len(selected_children),
            "survivor_selection": survivor_selection.metrics,
            "progressive_widening": widening_selection.metrics,
            "widening_candidate_cache": widening_cache_step,
            "search_feedback": {
                "parents_scanned": sum(
                    counters["scanned"] > 0
                    for counters in parent_feedback_step
                ),
                "scanned_actions": sum(
                    counters["scanned"] for counters in parent_feedback_step
                ),
                "attempted_actions": sum(
                    counters["attempted"] for counters in parent_feedback_step
                ),
                "valid_actions": sum(
                    counters["valid"] for counters in parent_feedback_step
                ),
                "unique_children": sum(
                    counters["unique"] for counters in parent_feedback_step
                ),
                "duplicate_children": sum(
                    counters["duplicate"] for counters in parent_feedback_step
                ),
                "invalid_actions": sum(
                    counters["invalid"] for counters in parent_feedback_step
                ),
                "improving_children": sum(
                    counters["improving"] for counters in parent_feedback_step
                ),
                "registry_nodes": (
                    len(search_feedback.nodes)
                    if search_feedback is not None
                    else 0
                ),
                "registry_nodes_with_descendant_gain": sum(
                    stats.descendant_gain > 0
                    for stats in (
                        search_feedback.nodes.values()
                        if search_feedback is not None
                        else ()
                    )
                ),
            },
            "widening_action_usage": {
                "scanned_widened_actions": scanned_widened_actions,
                "attempted_widened_actions": attempted_widened_actions,
                "scanned_parent_rank_min": (
                    min(scanned_parent_ranks) if scanned_parent_ranks else None
                ),
                "scanned_parent_rank_max": (
                    max(scanned_parent_ranks) if scanned_parent_ranks else None
                ),
                "attempted_parent_rank_min": (
                    min(attempted_parent_ranks) if attempted_parent_ranks else None
                ),
                "attempted_parent_rank_max": (
                    max(attempted_parent_ranks) if attempted_parent_ranks else None
                ),
            },
            "output_expansion_rounds": dict(
                Counter(state.expansion_round for state in beam)
            ),
            "candidate_path_improvements": sum(
                state.stagnation_steps == 0 for state in children
            ),
            "candidate_recoveries_after_exploration": sum(
                state.recovered_after_exploration for state in children
            ),
            "selected_recoveries_after_exploration": sum(
                state.recovered_after_exploration for state in beam
            ),
            "invalid_model_actions": invalid,
            "duplicate_successors": duplicates,
            "successor_metadata_skipped": duplicates,
            "preapply_fingerprint": fingerprint_step,
            "preapply_direct_inverse": {
                "mode": args.preapply_direct_inverse,
                **direct_inverse_step,
            },
            "neural_prefilter": neural_prefilter_step,
            "apply_profile": (
                rendered_apply_profile(apply_profile_step)
                if detailed_profile is not None
                else None
            ),
            "model_match_seconds": model_seconds,
            "quartz_exact_match_seconds": exact_seconds,
            "exact_refresh_actions_added": exact_refresh_actions_added,
            "proposal_seconds": proposal_seconds,
            "proposal_pipeline_metrics": proposal_metrics,
            "state_only_collation": collation_metrics,
            "quartz_apply_seconds": apply_seconds,
            "total_seconds": elapsed,
            "accepted_actions_per_second": len(selected_children) / elapsed,
            "match_states_per_second": len(input_beam)
            / max(1e-12, match_seconds),
            "matched_actions_per_second": matched_action_count
            / max(1e-12, match_seconds),
            "successful_apply_actions": successful_applies,
            "successful_apply_actions_per_second": successful_applies
            / max(1e-12, apply_seconds),
        }
        step_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if (
            args.target_gate_count is not None
            and best_state.gate_count <= args.target_gate_count
        ):
            break
        if args.stop_on_reference_loss and reference_lost:
            break
        if (
            args.max_total_attempted_actions
            and total_attempted_actions >= args.max_total_attempted_actions
        ):
            break

    total_seconds = time.perf_counter() - total_started
    total_accepted = sum(row["accepted_actions"] for row in step_rows)
    total_match_seconds = sum(
        row["model_match_seconds"] + row["quartz_exact_match_seconds"]
        for row in step_rows
    )
    total_matched_actions = sum(
        row["predicted_or_exact_actions"] for row in step_rows
    )
    total_input_states = sum(row["input_states"] for row in step_rows)
    total_apply_seconds = sum(row["quartz_apply_seconds"] for row in step_rows)
    total_fingerprint_seconds = sum(
        row["preapply_fingerprint"]["seconds"] for row in step_rows
    )
    total_successful_applies = sum(
        row["successful_apply_actions"] for row in step_rows
    )
    neural_audit_rows = 0
    if collect_neural_audit:
        neural_audit_rows = sum(row.shape[0] for row in neural_feature_chunks)
        if not neural_audit_rows:
            raise RuntimeError("neural audit collection produced no rows")
        audit_payload = {
            "format": "frozen_candidate_successor_v1",
            "qasm": str(args.qasm),
            "checkpoint": str(args.checkpoint),
            "feature_width": int(neural_feature_chunks[0].shape[1]),
            "outcome_labels": {
                "invalid": 0,
                "exact_duplicate": 1,
                "new_successor": 2,
            },
            "root_successor_group": 0,
            "features": torch.cat(neural_feature_chunks),
            "outcomes": torch.cat(neural_outcome_chunks),
            "successor_groups": torch.cat(neural_group_chunks),
            "xfer_ids": torch.cat(neural_xfer_chunks),
            "source_ids": torch.cat(neural_source_chunks),
            "probabilities": torch.cat(neural_probability_chunks),
            "gate_deltas": torch.cat(neural_gate_delta_chunks),
            "parent_gate_counts": torch.cat(neural_parent_gate_chunks),
            "steps": torch.cat(neural_step_chunks),
        }
        args.neural_audit_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(audit_payload, args.neural_audit_output)
    result = {
        "mode": args.mode,
        "model_apply_backend": model_apply_backend,
        "direct_binding_available": direct_binding_available,
        "dedup_before_child_materialization": True,
        "transactional_apply": args.transactional_apply,
        "transactional_native_registry_size": (
            None
            if transactional_native_registry is None
            else int(transactional_native_registry.size)
        ),
        "proposal_selection_backend": (
            "gpu_full_action_topk"
            if args.mode == "model" and args.model_pipeline == "state_only_gpu"
            else "stable_bounded_topk"
        ),
        "model_pipeline": args.model_pipeline if args.mode == "model" else None,
        "action_sequence_input": (
            False
            if args.mode == "model" and args.model_pipeline == "state_only_gpu"
            else None
            if args.mode == "quartz"
            else "zero_length_compatibility_tensors"
        ),
        "input_kind": "qasm_initial",
        "rule_metadata": str(args.data),
        "rule_metadata_role": "rewrite vocabulary only; not a search state",
        "eliminate_rotation": args.eliminate_rotation,
        "qasm": str(args.qasm),
        "beam_size": args.beam_size,
        "microbatch": args.microbatch,
        "max_source_matches": args.max_source_matches,
        "max_actions_per_parent": args.max_actions_per_parent,
        "proposal_factor": args.proposal_factor,
        "proposal_ranking": args.proposal_ranking,
        "proposal_ranking_seed": args.proposal_ranking_seed,
        "survivor_policy": args.survivor_policy,
        "exploration_fraction": (
            args.exploration_fraction
            if args.survivor_policy == "dual_lane"
            else 0.0
        ),
        "exploration_max_stagnation": args.exploration_max_stagnation,
        "exploration_max_detour": args.exploration_max_detour,
        "exploration_seed": args.exploration_seed,
        "locality_action_reserve": args.locality_action_reserve,
        "survivor_candidate_factor": (
            args.survivor_candidate_factor
            if args.survivor_policy == "dual_lane"
            else 1.0
        ),
        "max_total_attempted_actions": args.max_total_attempted_actions,
        "total_attempted_actions": total_attempted_actions,
        "progressive_widening": args.progressive_widening,
        "widening_revisit_fraction": (
            args.widening_revisit_fraction
            if args.progressive_widening == "on"
            else 0.0
        ),
        "widening_max_expansions": args.widening_max_expansions,
        "widening_rank_stride": args.max_actions_per_parent,
        "widening_min_actions_per_parent": (
            args.widening_min_actions_per_parent
            if args.progressive_widening == "on"
            else 0
        ),
        "widening_seed": args.widening_seed,
        "widening_policy": args.widening_policy,
        "widening_candidate_cache": {
            "enabled": widening_candidate_cache.enabled,
            "parent_hits": int(widening_cache_totals["parent_hits"]),
            "parent_misses": int(widening_cache_totals["parent_misses"]),
            "candidate_rows_reused": int(
                widening_cache_totals["candidate_rows_reused"]
            ),
            "candidate_rows_generated": int(
                widening_cache_totals["candidate_rows_generated"]
            ),
            "peak_resident_parents": widening_cache_peak_parents,
            "peak_resident_rows": widening_cache_peak_rows,
            "peak_resident_bytes": widening_cache_peak_bytes,
        },
        "deterministic_search": args.deterministic_search,
        "search_feedback": (
            search_feedback.rendered_summary()
            if search_feedback is not None
            else None
        ),
        "max_gate_increase": args.max_gate_increase,
        "requested_depth": args.depth,
        "completed_depth": len(step_rows),
        "maximum_action_depth": max(
            (state.depth for state in beam), default=0
        ),
        "initial_gate_count": initial_gate_count,
        "best_gate_count": best_state.gate_count,
        "best_first_seen_step": best_first_seen_step,
        "best_first_seen_seconds": best_first_seen_seconds,
        "best_action_depth": int(best_state.depth),
        "best_last_action_parent_rank": int(
            best_state.last_action_parent_rank
        ),
        "best_has_widening_ancestor": bool(best_state.widening_ancestor),
        "best_history": [list(map(int, action)) for action in best_state.history],
        "best_widened_action_trace": [
            list(map(int, action)) for action in best_state.widened_action_trace
        ],
        "best_has_exploration_ancestor": bool(
            best_state.exploration_ancestor
        ),
        "best_recovered_after_exploration": bool(
            best_state.recovered_after_exploration
        ),
        "improvement_trace": improvement_trace,
        "target_gate_count": args.target_gate_count,
        "target_reached": (
            args.target_gate_count is not None
            and best_state.gate_count <= args.target_gate_count
        ),
        "final_beam_size": len(beam),
        "final_beam_exact_identity_digest": graph_collection_identity_digest(
            state.graph for state in beam
        ),
        "best_graph_exact_identity_digest": graph_collection_identity_digest(
            (best_state.graph,)
        ),
        "dedup_identity": args.dedup_identity,
        "preapply_fingerprint": fingerprint_audit.stats(),
        "preapply_fingerprint_backend": {
            "requested": args.preapply_fingerprint_backend,
            "native_batch_size": (
                args.preapply_fingerprint_native_batch_size
            ),
            "profile_builds": dict(fingerprint_profile_builds),
        },
        "preapply_fingerprint_seconds": total_fingerprint_seconds,
        "neural_audit_output": (
            str(args.neural_audit_output)
            if args.neural_audit_output is not None
            else None
        ),
        "neural_audit_rows": neural_audit_rows,
        "neural_prefilter": {
            "mode": args.neural_prefilter_mode,
            "checkpoint": (
                str(args.neural_prefilter_checkpoint)
                if args.neural_prefilter_checkpoint is not None
                else None
            ),
            **neural_prefilter_totals,
        },
        "apply_profile": {
            "mode": args.apply_profile,
            **(
                rendered_apply_profile(apply_profile_totals)
                if args.apply_profile == "detailed"
                else {"seconds": {}, "counts": {}}
            ),
        },
        "preapply_fingerprint_min_proposals_per_gate": (
            args.preapply_fingerprint_min_proposals_per_gate
        ),
        "preapply_direct_inverse": {
            "mode": args.preapply_direct_inverse,
            **direct_inverse_totals,
        },
        "unique_graphs_seen": len(seen),
        "dedup_registry": seen.stats(),
        "total_seconds": total_seconds,
        "accepted_actions": total_accepted,
        "accepted_actions_per_second": total_accepted / total_seconds,
        "match_states_per_second": total_input_states
        / max(1e-12, total_match_seconds),
        "matched_actions_per_second": total_matched_actions
        / max(1e-12, total_match_seconds),
        "successful_apply_actions_per_second": total_successful_applies
        / max(1e-12, total_apply_seconds),
        "refresh_interval": args.refresh_interval,
        "refresh_count": args.refresh_count,
        "target_recall": args.target_recall if args.mode == "model" else None,
        "reference_data": (
            str(args.reference_data) if args.reference_data is not None else None
        ),
        "reference_retention": reference_retention,
        "steps": step_rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    if args.best_qasm is not None:
        args.best_qasm.parent.mkdir(parents=True, exist_ok=True)
        best_state.graph.to_qasm(filename=str(args.best_qasm))


if __name__ == "__main__":
    main()
