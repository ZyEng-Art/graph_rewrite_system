from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import ctypes
import ctypes.util
from dataclasses import dataclass
import gc
import hashlib
import heapq
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types
from typing import Any

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


@dataclass(frozen=True)
class AppliedRewrite:
    """A Quartz successor before expensive BeamState metadata is materialized."""

    graph: Any
    source_guids: tuple[int, ...]
    destination_guids: tuple[int, ...]


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


def snapshot(graph, guid_to_slot: dict[int, int]) -> dict:
    nodes = list(graph.nodes)
    return {
        "nodes": sorted(
            (guid_to_slot[int(node.guid)], int(node.gate_tp), int(node.guid))
            for node in nodes
        ),
        "edges": sorted(
            (
                guid_to_slot[int(nodes[int(src)].guid)],
                guid_to_slot[int(nodes[int(dst)].guid)],
                int(src_port),
                int(dst_port),
            )
            for src, dst, src_port, dst_port in graph.all_edges()
        ),
    }


def graph_delta(before: dict, after: dict) -> tuple[set[int], set[tuple[int, ...]]]:
    before_nodes = {int(row[0]) for row in before["nodes"]}
    after_nodes = {int(row[0]) for row in after["nodes"]}
    before_edges = set(map(tuple, before["edges"]))
    after_edges = set(map(tuple, after["edges"]))
    return before_nodes - after_nodes, before_edges.symmetric_difference(after_edges)


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
) -> tuple[CandidateTensors, float, dict[str, int]]:
    """Predict all retained matches from exact graphs without action tensors."""
    if not hasattr(model, "encode_current_graph"):
        raise ValueError(
            "state-only inference requires a model with encode_current_graph"
        )
    chunks = []
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
    return CandidateTensors.cat(chunks), time.perf_counter() - started, collation


def rank_proposals(
    beam: list[BeamState],
    action_rows: list[list[tuple[int, int, tuple[int, ...] | None, float]]],
    gate_deltas: list[int],
    *,
    beam_size: int,
    max_actions_per_parent: int,
    proposal_factor: int,
    max_gate_increase: int,
) -> tuple[list[Proposal], int]:
    """Select stable bounded top-k actions before creating Proposal objects.

    Large circuits can expose millions of legal source/xfer combinations per
    layer, while the search consumes only a small per-parent and global prefix.
    Selecting raw tuples first preserves the old stable sort order but avoids
    allocating and sorting a Python dataclass for every discarded action.
    """
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
        selected_rows = heapq.nsmallest(
            effective_parent_cap,
            eligible_rows,
            key=lambda row: (
                state.gate_count + gate_deltas[row[0]],
                -row[3],
                row[0],
            ),
        )
        proposals.extend(
            Proposal(
                parent=parent_index,
                xfer_id=xfer_id,
                anchor_slot=anchor,
                binding=binding,
                probability=probability,
                next_gate_count=state.gate_count + gate_deltas[xfer_id],
            )
            for xfer_id, anchor, binding, probability in selected_rows
        )

    proposals = heapq.nsmallest(
        beam_size * proposal_factor,
        proposals,
        key=lambda row: (
            row.next_gate_count,
            -row.probability,
            beam[row.parent].gate_count,
        ),
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
) -> AppliedRewrite | None:
    """Apply one proposal without constructing metadata for duplicate children.

    A patched Quartz can validate the model's complete ordered source binding
    directly.  This avoids re-running anchor-based subgraph matching merely to
    rediscover a binding the model already supplied.  Original Quartz actions
    contain only an anchor and continue to use the original exact API.
    """
    slot_to_guid = {
        int(slot): int(guid) for slot, _, guid in parent.snapshot["nodes"]
    }
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
            source_guids = [slot_to_guid[slot] for slot in proposal.binding]
        except KeyError:
            return None
        prefer_guid = binding_backend != "node_direct"
        if guid_direct_method is not None and prefer_guid:
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
) -> BeamState:
    """Construct model/search metadata only after exact dedup accepts a graph."""
    graph = applied.graph
    source_slots = tuple(
        parent.guid_to_slot[guid] for guid in applied.source_guids
    )

    live_guids = {int(node.guid) for node in graph.nodes}
    surviving_destination_guids = tuple(
        guid for guid in applied.destination_guids if guid in live_guids
    )
    guid_to_slot = dict(parent.guid_to_slot)
    next_slot = update_slots(
        graph,
        guid_to_slot,
        parent.next_slot,
        surviving_destination_guids,
    )
    after = snapshot(graph, guid_to_slot)
    removed, changed_edges = graph_delta(parent.snapshot, after)
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
    gate_count = int(graph.gate_count)
    if not eliminate_rotation and gate_count != proposal.next_gate_count:
        raise RuntimeError(
            f"gate delta mismatch: expected {proposal.next_gate_count}, got {gate_count}"
        )
    return BeamState(
        graph=graph,
        snapshot=after,
        guid_to_slot=guid_to_slot,
        next_slot=next_slot,
        last_touched=last_touched,
        rewrite_distance=distances_from_core(after, core),
        previous_preferred=previous_preferred,
        local_streak=local_streak,
        gate_count=gate_count,
        depth=parent.depth + 1,
        history=parent.history + ((proposal.xfer_id, proposal.anchor_slot),),
        last_xfer_id=proposal.xfer_id,
        last_source_slots=source_slots,
        last_destination_slots=ordered_destination_slots,
    )


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
    if args.mode == "model" and (args.checkpoint is None or args.calibration is None):
        parser.error("model mode requires --checkpoint and --calibration")
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

    model = None
    threshold_config = None
    source_vectors = None
    gpu_rule_index = None
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
                source_vectors = model.retrieval_source(
                    model.source_representations()
                )
            gpu_rule_index = GpuRuleIndex.build(
                source_to_xfers,
                gate_deltas,
                model.num_sources,
                args.max_gate_increase,
                device,
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
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    initial_snapshot = snapshot(graph, guid_to_slot)
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
    best_first_seen_step = 0
    best_first_seen_seconds = 0.0
    improvement_trace = [
        {"step": 0, "gate_count": initial_gate_count, "seconds": 0.0}
    ]
    seen = (
        ExactGraphRegistry.seeded(graph)
        if args.dedup_identity == "exact"
        else QuartzHashRegistry.seeded(graph)
    )
    diagnostic_seen_hashes = (
        {int(graph.hash())} if reference_hashes is not None else None
    )
    fingerprint_audit = FingerprintAudit.create(
        mode=args.preapply_fingerprint,
        kind=args.preapply_fingerprint_kind,
        representatives=args.preapply_fingerprint_representatives,
    )
    direct_inverse_totals = {
        "candidates": 0,
        "skipped_before_apply": 0,
        "shadow_exact_duplicates": 0,
        "shadow_novel_successors": 0,
        "shadow_invalid": 0,
    }
    step_rows = []
    reference_retention = []
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
        exact_refresh_actions_added = 0
        action_rows = None
        source_binding_candidates = 0
        proposal_metrics = {}
        collation_metrics = {}
        if args.mode == "model" and args.model_pipeline == "state_only_gpu":
            candidates, model_seconds, collation_metrics = (
                state_only_candidate_tensors(
                    beam,
                    model,
                    device,
                    threshold_config,
                    source_vectors,
                    args.microbatch,
                    args.max_source_matches,
                )
            )
            source_binding_candidates = int(candidates.sources.numel())
            effective_parent_cap = max(
                args.max_actions_per_parent,
                math.ceil(args.beam_size / max(1, len(beam))) * 2,
            )
            proposal_started = time.perf_counter()
            proposals, proposal_metrics, _, _ = build_gpu_proposals(
                candidates,
                beam,
                gpu_rule_index,
                per_parent_cap=effective_parent_cap,
                global_cap=args.beam_size * args.proposal_factor,
                ranking_mode="gate",
            )
            proposal_seconds = time.perf_counter() - proposal_started
            if proposals is None:
                raise RuntimeError("state-only GPU proposal materialization failed")
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
            )
            proposal_seconds = time.perf_counter() - proposal_started
            matched_action_count = sum(map(len, action_rows))
            source_binding_candidates = (
                sum(map(len, predicted)) if args.mode == "model" else 0
            )

        fingerprint_before = fingerprint_audit.stats()
        fingerprint_profiles = {}
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
        applied_proposal_positions = set()
        fingerprint_skipped_positions = set()
        direct_inverse_skipped_positions = set()
        for proposal_position, proposal in enumerate(proposals):
            if len(children) >= args.beam_size:
                break
            proposals_scanned += 1
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
                    if proposal.parent not in fingerprint_profiles:
                        profile = build_wire_trace_profile(
                            beam[proposal.parent].graph,
                            beam[proposal.parent].guid_to_slot,
                        )
                        fingerprint_profiles[proposal.parent] = profile
                    else:
                        profile = fingerprint_profiles[proposal.parent]
                    if profile is not None:
                        fingerprint = successor_fingerprint(
                            profile,
                            source_patterns[proposal.xfer_id],
                            destination_patterns[proposal.xfer_id],
                            proposal.binding,
                            xfer_id=proposal.xfer_id,
                            kind=args.preapply_fingerprint_kind,
                        )
                    should_skip = fingerprint_audit.should_skip(fingerprint)
                    fingerprint_seconds += (
                        time.perf_counter() - fingerprint_started
                    )
                    if should_skip:
                        fingerprint_skipped_positions.add(proposal_position)
                        continue
            attempted += 1
            applied_proposal_positions.add(proposal_position)
            applied = apply_rewrite(
                beam[proposal.parent],
                proposal,
                xfers,
                eliminate_rotation=args.eliminate_rotation,
                binding_backend=model_apply_backend,
            )
            if applied is None:
                invalid += 1
                fingerprint_audit.observe_invalid(fingerprint)
                if direct_inverse and args.preapply_direct_inverse == "shadow":
                    direct_inverse_step["shadow_invalid"] += 1
                    direct_inverse_totals["shadow_invalid"] += 1
                continue
            exact_identity = (
                exact_graph_key(applied.graph)
                if args.preapply_fingerprint == "shadow"
                and fingerprint is not None
                else None
            )
            if diagnostic_seen_hashes is not None:
                diagnostic_seen_hashes.add(int(applied.graph.hash()))
            is_new_successor = seen.register(applied.graph)
            fingerprint_audit.observe_valid(fingerprint, exact_identity)
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
                continue
            child = materialize_child(
                beam[proposal.parent],
                proposal,
                applied,
                eliminate_rotation=args.eliminate_rotation,
            )
            children.append(child)
        apply_seconds = time.perf_counter() - apply_started
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
        if not children:
            break
        children.sort(key=lambda state: (state.gate_count, len(state.history)))
        beam = children[: args.beam_size]
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
                "retained": not reference_lost,
                "beam_indices": retained_indices,
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
                }
            )
        match_seconds = model_seconds + exact_seconds
        successful_applies = attempted - invalid
        row = {
            "step": step + 1,
            "input_states": len(input_beam),
            "output_states": len(beam),
            "best_gate_count": beam[0].gate_count,
            "global_best_gate_count": best_state.gate_count,
            "cumulative_seconds": cumulative_seconds,
            "predicted_or_exact_actions": matched_action_count,
            "source_binding_candidates": source_binding_candidates,
            "eligible_actions_before_parent_cap": total_action_candidates,
            "proposals_after_caps": len(proposals),
            "proposals_scanned": proposals_scanned,
            "attempted_actions": attempted,
            "accepted_actions": len(beam),
            "invalid_model_actions": invalid,
            "duplicate_successors": duplicates,
            "successor_metadata_skipped": duplicates,
            "preapply_fingerprint": fingerprint_step,
            "preapply_direct_inverse": {
                "mode": args.preapply_direct_inverse,
                **direct_inverse_step,
            },
            "model_match_seconds": model_seconds,
            "quartz_exact_match_seconds": exact_seconds,
            "exact_refresh_actions_added": exact_refresh_actions_added,
            "proposal_seconds": proposal_seconds,
            "proposal_pipeline_metrics": proposal_metrics,
            "state_only_collation": collation_metrics,
            "quartz_apply_seconds": apply_seconds,
            "total_seconds": elapsed,
            "accepted_actions_per_second": len(beam) / elapsed,
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
    result = {
        "mode": args.mode,
        "model_apply_backend": model_apply_backend,
        "direct_binding_available": direct_binding_available,
        "dedup_before_child_materialization": True,
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
        "max_gate_increase": args.max_gate_increase,
        "requested_depth": args.depth,
        "completed_depth": len(step_rows),
        "initial_gate_count": initial_gate_count,
        "best_gate_count": best_state.gate_count,
        "best_first_seen_step": best_first_seen_step,
        "best_first_seen_seconds": best_first_seen_seconds,
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
        "preapply_fingerprint_seconds": total_fingerprint_seconds,
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
