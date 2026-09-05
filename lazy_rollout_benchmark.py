from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import ctypes.util
from dataclasses import dataclass
from functools import lru_cache
import gc
import hashlib
import heapq
import json
import math
import numpy as np
from pathlib import Path
import struct
import time
from typing import Any

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch

from beam_search_benchmark import (
    BeamState,
    Proposal,
    model_matches,
    snapshot,
    update_slots,
)
from dataset import RuleMetadata
from incremental_graph import IncrementalCircuit, parse_pattern, structural_binding
from model import S0ActionBindingModel
from threshold_inference import load_threshold_config


@dataclass(frozen=True)
class LazyAction:
    xfer_id: int
    source_slots: tuple[int, ...]
    destination_slots: tuple[int, ...]


@dataclass(frozen=True)
class ExactReplayCacheEntry:
    graph: Any | None
    guid_to_slot: dict[int, int] | None
    slot_to_guid: dict[int, int] | None
    failure_step: int | None


@dataclass(frozen=True)
class IndexedTopology:
    nodes: dict[int, int]
    edges: frozenset[tuple[int, int, int, int]]
    out_edge: dict[tuple[int, int], tuple[int, int]]
    in_edge: dict[tuple[int, int], tuple[int, int]]
    adjacency: dict[int, tuple[int, ...]]
    fingerprint: int


@dataclass(frozen=True)
class IndexedRewrite:
    fingerprint: int
    removed_edges: frozenset[tuple[int, int, int, int]]
    added_edges: frozenset[tuple[int, int, int, int]]


def _component_hash(kind: str, values: tuple) -> int:
    return hash((kind, *values))


def indexed_topology(snapshot_row: dict) -> IndexedTopology:
    nodes = {
        int(slot): int(gate_type)
        for slot, gate_type, _ in snapshot_row["nodes"]
    }
    edges = frozenset(tuple(map(int, edge)) for edge in snapshot_row["edges"])
    out_edge = {
        (src, src_port): (dst, dst_port)
        for src, dst, src_port, dst_port in edges
    }
    in_edge = {
        (dst, dst_port): (src, src_port)
        for src, dst, src_port, dst_port in edges
    }
    adjacency_lists = {slot: [] for slot in nodes}
    for src, dst, _, _ in edges:
        adjacency_lists[src].append(dst)
        adjacency_lists[dst].append(src)
    adjacency = {
        slot: tuple(neighbors) for slot, neighbors in adjacency_lists.items()
    }
    fingerprint = 0
    for item in nodes.items():
        fingerprint ^= _component_hash("node", item)
    for edge in edges:
        fingerprint ^= _component_hash("edge", edge)
    return IndexedTopology(nodes, edges, out_edge, in_edge, adjacency, fingerprint)


@lru_cache(maxsize=None)
def _pattern_wires(pattern) -> dict[int, tuple[tuple[int, int], ...]]:
    wires: dict[int, list[tuple[int, int]]] = {}
    for operation_index, operation in enumerate(pattern):
        for port, qubit in enumerate(operation.qubits):
            wires.setdefault(qubit, []).append((operation_index, port))
    return {qubit: tuple(wire) for qubit, wire in wires.items()}


def plan_indexed_rewrite(
    parent: IndexedTopology,
    source,
    destination,
    source_slots: tuple[int, ...],
    destination_slots: tuple[int, ...],
) -> IndexedRewrite:
    if len(source) != len(source_slots):
        raise ValueError("source pattern/binding length mismatch")
    if len(destination) != len(destination_slots):
        raise ValueError("destination pattern/binding length mismatch")
    removed = set(source_slots)
    if not removed.issubset(parent.nodes):
        raise ValueError("source binding contains a non-live node")
    if any(slot in parent.nodes for slot in destination_slots):
        raise ValueError("destination binding reuses a live slot")

    source_wires = _pattern_wires(source)
    destination_wires = _pattern_wires(destination)
    boundaries = {}
    for qubit, wire in source_wires.items():
        first_index, first_port = wire[0]
        last_index, last_port = wire[-1]
        predecessor = parent.in_edge.get(
            (source_slots[first_index], first_port)
        )
        successor = parent.out_edge.get(
            (source_slots[last_index], last_port)
        )
        if predecessor is not None and predecessor[0] in removed:
            predecessor = None
        if successor is not None and successor[0] in removed:
            successor = None
        boundaries[qubit] = (predecessor, successor)

    removed_edges = set()
    for slot in removed:
        for port in range(4):
            outgoing = parent.out_edge.get((slot, port))
            if outgoing is not None:
                removed_edges.add((slot, outgoing[0], port, outgoing[1]))
            incoming = parent.in_edge.get((slot, port))
            if incoming is not None:
                removed_edges.add((incoming[0], slot, incoming[1], port))

    requested_edges = set()
    for qubit in set(source_wires) | set(destination_wires):
        destination_wire = destination_wires.get(qubit, ())
        predecessor, successor = boundaries.get(qubit, (None, None))
        if destination_wire:
            for (left_index, left_port), (right_index, right_port) in zip(
                destination_wire, destination_wire[1:]
            ):
                requested_edges.add(
                    (
                        destination_slots[left_index],
                        destination_slots[right_index],
                        left_port,
                        right_port,
                    )
                )
            first_index, first_port = destination_wire[0]
            last_index, last_port = destination_wire[-1]
            if predecessor is not None:
                requested_edges.add(
                    (
                        predecessor[0],
                        destination_slots[first_index],
                        predecessor[1],
                        first_port,
                    )
                )
            if successor is not None:
                requested_edges.add(
                    (
                        destination_slots[last_index],
                        successor[0],
                        last_port,
                        successor[1],
                    )
                )
        elif predecessor is not None and successor is not None:
            requested_edges.add(
                (predecessor[0], successor[0], predecessor[1], successor[1])
            )

    added_edges = {
        edge
        for edge in requested_edges
        if edge in removed_edges or edge not in parent.edges
    }
    fingerprint = parent.fingerprint
    for slot in source_slots:
        gate_type = parent.nodes[slot]
        fingerprint ^= _component_hash("node", (slot, gate_type))
    for slot, operation in zip(destination_slots, destination):
        fingerprint ^= _component_hash("node", (slot, operation.gate_type))
    for edge in removed_edges:
        fingerprint ^= _component_hash("edge", edge)
    for edge in added_edges:
        fingerprint ^= _component_hash("edge", edge)

    return IndexedRewrite(
        fingerprint,
        frozenset(removed_edges),
        frozenset(added_edges),
    )


def materialize_indexed_rewrite(
    parent: IndexedTopology,
    destination,
    source_slots: tuple[int, ...],
    destination_slots: tuple[int, ...],
    rewrite: IndexedRewrite,
) -> IndexedTopology:
    child_nodes = dict(parent.nodes)
    for slot in source_slots:
        del child_nodes[slot]
    for slot, operation in zip(destination_slots, destination):
        child_nodes[slot] = operation.gate_type
    child_edges = frozenset(
        parent.edges.difference(rewrite.removed_edges).union(
            rewrite.added_edges
        )
    )
    out_edge = dict(parent.out_edge)
    in_edge = dict(parent.in_edge)
    adjacency = dict(parent.adjacency)
    mutable_neighbors: dict[int, list[int]] = {}

    def neighbors(slot: int) -> list[int]:
        if slot not in mutable_neighbors:
            mutable_neighbors[slot] = list(adjacency.get(slot, ()))
        return mutable_neighbors[slot]

    for src, dst, src_port, dst_port in rewrite.removed_edges:
        if out_edge.get((src, src_port)) == (dst, dst_port):
            del out_edge[(src, src_port)]
        if in_edge.get((dst, dst_port)) == (src, src_port):
            del in_edge[(dst, dst_port)]
        if src in child_nodes:
            neighbors(src).remove(dst)
        if dst in child_nodes:
            neighbors(dst).remove(src)
    for src, dst, src_port, dst_port in rewrite.added_edges:
        out_edge[(src, src_port)] = (dst, dst_port)
        in_edge[(dst, dst_port)] = (src, src_port)
        neighbors(src).append(dst)
        neighbors(dst).append(src)
    for slot in source_slots:
        adjacency.pop(slot, None)
    for slot in destination_slots:
        mutable_neighbors.setdefault(slot, [])
    adjacency.update(
        (slot, tuple(slot_neighbors))
        for slot, slot_neighbors in mutable_neighbors.items()
    )
    return IndexedTopology(
        child_nodes,
        child_edges,
        out_edge,
        in_edge,
        adjacency,
        rewrite.fingerprint,
    )


def distances_from_index(
    topology: IndexedTopology, core: set[int]
) -> dict[int, int]:
    slot_distance = distance_slots_from_index(topology, core)
    return {
        slot: min(slot_distance[slot], 4) if slot_distance[slot] >= 0 else 5
        for slot in topology.nodes
    }


def distance_slots_from_index(
    topology: IndexedTopology, core: set[int]
) -> list[int]:
    # Persistent slots are allocated densely and never reused.  A short list is
    # substantially cheaper than a Python hash table in the per-child BFS.
    slot_distance = [-1] * (max(topology.nodes, default=-1) + 1)
    queue = []
    for slot in core:
        if slot in topology.nodes:
            slot_distance[slot] = 0
            queue.append(slot)
    cursor = 0
    while cursor < len(queue):
        slot = queue[cursor]
        cursor += 1
        next_distance = slot_distance[slot] + 1
        for neighbor in topology.adjacency[slot]:
            if slot_distance[neighbor] < 0:
                slot_distance[neighbor] = next_distance
                queue.append(neighbor)
    return slot_distance


def dense_distances_from_index(
    topology: IndexedTopology, core: set[int]
) -> np.ndarray:
    slot_distance = np.asarray(
        distance_slots_from_index(topology, core), dtype=np.int16
    )
    return np.where(slot_distance >= 0, np.minimum(slot_distance, 4), 5).astype(
        np.int8, copy=False
    )


def dense_last_touched(
    parent: BeamState,
    removed: set[int],
    core: set[int],
    slots: int,
) -> np.ndarray:
    touched = np.full(slots, -1, dtype=np.int16)
    if isinstance(parent.last_touched, np.ndarray):
        touched[: len(parent.last_touched)] = parent.last_touched
    else:
        for slot, step in parent.last_touched.items():
            touched[int(slot)] = int(step)
    if removed:
        touched[np.fromiter(removed, dtype=np.int64)] = -1
    if core:
        touched[np.fromiter(core, dtype=np.int64)] = parent.depth
    return touched


def indexed_topology_signature(topology: IndexedTopology) -> tuple:
    return (
        tuple(sorted(topology.nodes.items())),
        tuple(sorted(topology.edges)),
    )


def snapshot_signature(row: dict) -> tuple:
    return (
        tuple(sorted((int(slot), int(gate_type)) for slot, gate_type, _ in row["nodes"])),
        tuple(sorted(tuple(map(int, edge)) for edge in row["edges"])),
    )


def graph_topology_signature(graph, guid_to_slot: dict[int, int]) -> tuple:
    nodes = list(graph.nodes)
    return (
        tuple(
            sorted(
                (guid_to_slot[int(node.guid)], int(node.gate_tp))
                for node in nodes
            )
        ),
        tuple(
            sorted(
                (
                    guid_to_slot[int(nodes[int(src)].guid)],
                    guid_to_slot[int(nodes[int(dst)].guid)],
                    int(src_port),
                    int(dst_port),
                )
                for src, dst, src_port, dst_port in graph.all_edges()
            )
        ),
    )


def shared_replay_prefixes(states: list[BeamState]) -> set[tuple[LazyAction, ...]]:
    counts: dict[tuple[LazyAction, ...], int] = defaultdict(int)
    for state in states:
        checkpoint_depth = (
            state.exact_checkpoint_depth
            if state.exact_graph_checkpoint is not None
            else 0
        )
        if checkpoint_depth > len(state.history):
            raise RuntimeError("exact checkpoint is ahead of speculative history")
        for depth in range(checkpoint_depth + 1, len(state.history) + 1):
            counts[state.history[:depth]] += 1
    return {prefix for prefix, count in counts.items() if count > 1}


def raw_topology_hash(row: dict) -> int:
    """Cheap process-local hash; catches duplicates with the same persistent slots."""
    return raw_topology_hash_components(
        {int(slot): int(gate_type) for slot, gate_type, _ in row["nodes"]},
        row["edges"],
    )


def raw_topology_hash_components(nodes: dict[int, int], edges) -> int:
    """Order-independent raw hash without sorting child topology first."""
    fingerprint = 0
    for item in nodes.items():
        fingerprint ^= _component_hash("node", item)
    for edge in edges:
        fingerprint ^= _component_hash("edge", tuple(edge))
    return fingerprint


def distances_from_components(nodes, edges, core: set[int]) -> dict[int, int]:
    """Distance buckets directly from an IncrementalCircuit representation."""
    live = set(nodes)
    adjacency = {slot: set() for slot in live}
    for src, dst, _, _ in edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    result = {slot: 0 for slot in core if slot in live}
    queue = list(result)
    cursor = 0
    while cursor < len(queue):
        slot = queue[cursor]
        cursor += 1
        next_distance = result[slot] + 1
        for neighbor in adjacency[slot]:
            if neighbor not in result:
                result[neighbor] = next_distance
                queue.append(neighbor)
    return {
        slot: min(result.get(slot, 5), 4) if slot in result else 5
        for slot in live
    }


def lazy_child_indexed(
    parent: BeamState,
    proposal: Proposal,
    source,
    destination,
    seen: set,
) -> tuple[BeamState | None, int | None, bool]:
    parent_topology = (
        parent.topology_index
        if parent.topology_index is not None
        else indexed_topology(parent.snapshot)
    )
    destination_slots = tuple(
        range(parent.next_slot, parent.next_slot + len(destination))
    )
    try:
        rewrite = plan_indexed_rewrite(
            parent_topology,
            source,
            destination,
            proposal.binding,
            destination_slots,
        )
    except ValueError:
        return None, None, False
    fingerprint = rewrite.fingerprint
    if fingerprint in seen:
        return None, fingerprint, True

    child_topology = materialize_indexed_rewrite(
        parent_topology,
        destination,
        proposal.binding,
        destination_slots,
        rewrite,
    )
    removed = set(proposal.binding)
    changed_edges = rewrite.removed_edges.symmetric_difference(
        rewrite.added_edges
    )
    live = set(child_topology.nodes)
    destination_set = set(destination_slots)
    core = set(destination_set)
    for src, dst, _, _ in changed_edges:
        if src in live:
            core.add(src)
        if dst in live:
            core.add(dst)
    next_slot = parent.next_slot + len(destination)
    last_touched = dense_last_touched(
        parent, removed, core, next_slot
    )
    predecessors = set()
    for slot in proposal.binding:
        for port in range(4):
            incoming = parent_topology.in_edge.get((slot, port))
            if incoming is not None:
                predecessors.add(incoming[0])
    previous_preferred = (destination_set | predecessors) & live
    continued = proposal.anchor_slot in parent.previous_preferred
    local_streak = parent.local_streak + 1 if continued else 0
    if len(child_topology.nodes) != proposal.next_gate_count:
        raise RuntimeError(
            f"gate delta mismatch: expected {proposal.next_gate_count}, "
            f"got {len(child_topology.nodes)}"
        )
    action = LazyAction(
        xfer_id=proposal.xfer_id,
        source_slots=proposal.binding,
        destination_slots=destination_slots,
    )
    return (
        BeamState(
            graph=None,
            # Tensorized paged collation consumes topology_index directly.  Do
            # not rebuild and sort a full duplicate snapshot for every child.
            snapshot=None,
            guid_to_slot={},
            next_slot=next_slot,
            last_touched=last_touched,
            rewrite_distance=dense_distances_from_index(child_topology, core),
            previous_preferred=previous_preferred,
            local_streak=local_streak,
            gate_count=proposal.next_gate_count,
            depth=parent.depth + 1,
            history=parent.history + (action,),
            topology_index=child_topology,
            exact_graph_checkpoint=parent.exact_graph_checkpoint,
            exact_slot_checkpoint=parent.exact_slot_checkpoint,
            exact_checkpoint_depth=parent.exact_checkpoint_depth,
        ),
        fingerprint,
        False,
    )


def topology_digest(row: dict) -> bytes:
    """A slot-renumbering-tolerant DAG digest used for speculative deduplication."""
    nodes = {int(slot): int(gate_type) for slot, gate_type, _ in row["nodes"]}
    incoming: dict[int, list[tuple[int, int, int]]] = {slot: [] for slot in nodes}
    outgoing: dict[int, list[tuple[int, int, int]]] = {slot: [] for slot in nodes}
    indegree = {slot: 0 for slot in nodes}
    for src, dst, src_port, dst_port in row["edges"]:
        src = int(src)
        dst = int(dst)
        src_port = int(src_port)
        dst_port = int(dst_port)
        outgoing[src].append((dst, src_port, dst_port))
        incoming[dst].append((src, src_port, dst_port))
        indegree[dst] += 1

    canonical: dict[int, int] = {}
    ready = []

    def ready_key(slot: int) -> tuple:
        predecessors = tuple(
            sorted(
                (canonical[src], src_port, dst_port)
                for src, src_port, dst_port in incoming[slot]
            )
        )
        future = tuple(
            sorted(
                (
                    src_port,
                    dst_port,
                    nodes[dst],
                    len(incoming[dst]),
                    len(outgoing[dst]),
                )
                for dst, src_port, dst_port in outgoing[slot]
            )
        )
        return nodes[slot], predecessors, future

    for slot, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, (ready_key(slot), slot))
    while ready:
        _, slot = heapq.heappop(ready)
        canonical[slot] = len(canonical)
        for dst, _, _ in outgoing[slot]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                heapq.heappush(ready, (ready_key(dst), dst))
    if len(canonical) != len(nodes):
        raise RuntimeError("incremental circuit is not a DAG")

    ordered_types = [0] * len(nodes)
    for slot, index in canonical.items():
        ordered_types[index] = nodes[slot]
    normalized_edges = sorted(
        (
            canonical[int(src)],
            canonical[int(dst)],
            int(src_port),
            int(dst_port),
        )
        for src, dst, src_port, dst_port in row["edges"]
    )
    digest = hashlib.blake2b(digest_size=16)
    digest.update(struct.pack("<I", len(ordered_types)))
    digest.update(bytes(ordered_types))
    digest.update(struct.pack("<I", len(normalized_edges)))
    for edge in normalized_edges:
        digest.update(struct.pack("<IIHH", *edge))
    return digest.digest()


def lazy_child(
    parent: BeamState,
    proposal: Proposal,
    source_patterns,
    destination_patterns,
    structural_recheck: bool,
    dedup_mode: str,
    seen: set,
    topology_backend: str = "legacy",
) -> tuple[BeamState | None, bytes | int | None, bool]:
    if proposal.binding is None:
        raise ValueError("lazy rollout requires a complete predicted binding")
    source = source_patterns[proposal.xfer_id]
    destination = destination_patterns[proposal.xfer_id]
    if (
        topology_backend == "indexed"
        and dedup_mode == "raw"
        and not structural_recheck
    ):
        return lazy_child_indexed(
            parent, proposal, source, destination, seen
        )
    circuit = IncrementalCircuit(parent.snapshot)
    before_nodes = set(circuit.nodes)
    before_edges = circuit.edges
    if structural_recheck and structural_binding(
        circuit, source, proposal.anchor_slot
    ) != proposal.binding:
        return None, None, False
    destination_slots = tuple(
        range(parent.next_slot, parent.next_slot + len(destination))
    )
    try:
        circuit.apply(source, destination, proposal.binding, destination_slots)
    except ValueError:
        return None, None, False
    if dedup_mode == "canonical":
        after = {
            "nodes": sorted(
                (slot, gate_type, -1) for slot, gate_type in circuit.nodes.items()
            ),
            "edges": sorted(circuit.edges),
        }
        fingerprint = topology_digest(after)
    elif dedup_mode == "raw":
        fingerprint = raw_topology_hash_components(circuit.nodes, circuit.edges)
    else:
        fingerprint = None
    # Most rejected proposals are duplicates. Avoid graph-delta, locality BFS,
    # and history construction for them.
    if fingerprint is not None and fingerprint in seen:
        return None, fingerprint, True
    after = {
        "nodes": sorted(
            (slot, gate_type, -1) for slot, gate_type in circuit.nodes.items()
        ),
        "edges": sorted(circuit.edges),
    }
    removed = before_nodes - set(circuit.nodes)
    changed_edges = before_edges.symmetric_difference(circuit.edges)
    live = set(circuit.nodes)
    destination_set = set(destination_slots)
    core = set(destination_set)
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
    source_set = set(proposal.binding)
    predecessors = {src for src, dst, _, _ in before_edges if dst in source_set}
    previous_preferred = (destination_set | predecessors) & live
    continued = proposal.anchor_slot in parent.previous_preferred
    local_streak = parent.local_streak + 1 if continued else 0
    if len(circuit.nodes) != proposal.next_gate_count:
        raise RuntimeError(
            f"gate delta mismatch: expected {proposal.next_gate_count}, "
            f"got {len(circuit.nodes)}"
        )
    action = LazyAction(
        xfer_id=proposal.xfer_id,
        source_slots=proposal.binding,
        destination_slots=destination_slots,
    )
    return (
        BeamState(
            graph=None,
            snapshot=after,
            guid_to_slot={},
            next_slot=parent.next_slot + len(destination),
            last_touched=last_touched,
            rewrite_distance=distances_from_components(
                circuit.nodes, circuit.edges, core
            ),
            previous_preferred=previous_preferred,
            local_streak=local_streak,
            gate_count=proposal.next_gate_count,
            depth=parent.depth + 1,
            history=parent.history + (action,),
            exact_graph_checkpoint=parent.exact_graph_checkpoint,
            exact_slot_checkpoint=parent.exact_slot_checkpoint,
            exact_checkpoint_depth=parent.exact_checkpoint_depth,
        ),
        fingerprint,
        False,
    )


def replay_state(
    state: BeamState,
    context,
    pygraph_cls,
    xfers,
    initial_qasm: str,
    *,
    return_checkpoint: bool = False,
    ignore_checkpoint: bool = False,
    profile_timing: dict[str, float] | None = None,
    profile_counts: dict[str, int] | None = None,
    replay_cache: dict[tuple[LazyAction, ...], ExactReplayCacheEntry] | None = None,
    replay_cache_prefixes: set[tuple[LazyAction, ...]] | None = None,
    prefer_direct_binding: bool = True,
    eliminate_rotation: bool = False,
):
    """Materialize one speculative trajectory in Quartz for an out-of-band audit."""
    def lookup_source_node_ids(action: LazyAction) -> list[int] | None:
        needed = set(action.source_slots)
        node_id_by_slot: dict[int, int] = {}
        for index, graph_node in enumerate(graph.nodes):
            slot = guid_to_slot.get(int(graph_node.guid))
            if slot in needed:
                node_id_by_slot[slot] = index
                if len(node_id_by_slot) == len(needed):
                    break
        if len(node_id_by_slot) != len(needed):
            return None
        return [node_id_by_slot[slot] for slot in action.source_slots]

    def lookup_source_guids(action: LazyAction) -> list[int] | None:
        source_guids = [slot_to_guid.get(slot) for slot in action.source_slots]
        if any(guid is None for guid in source_guids):
            return None
        return [int(guid) for guid in source_guids]

    def now() -> float:
        return time.perf_counter() if profile_timing is not None else 0.0

    def add_seconds(name: str, started: float) -> None:
        if profile_timing is not None:
            profile_timing[name] = (
                profile_timing.get(name, 0.0) + time.perf_counter() - started
            )

    def add_count(name: str, amount: int = 1) -> None:
        if profile_counts is not None:
            profile_counts[name] = profile_counts.get(name, 0) + amount

    add_count("replay_states_total")
    init_started = now()
    if ignore_checkpoint or state.exact_graph_checkpoint is None:
        graph = pygraph_cls.from_qasm_str(context=context, qasm_str=initial_qasm)
        guid_to_slot: dict[int, int] = {}
        update_slots(graph, guid_to_slot, 0)
        slot_to_guid = {slot: guid for guid, slot in guid_to_slot.items()}
        checkpoint_depth = 0
        add_seconds("init_from_qasm_seconds", init_started)
        add_count("init_from_qasm_states")
    else:
        graph = state.exact_graph_checkpoint
        guid_to_slot = dict(state.exact_slot_checkpoint)
        slot_to_guid = {slot: guid for guid, slot in guid_to_slot.items()}
        checkpoint_depth = state.exact_checkpoint_depth
        if checkpoint_depth > len(state.history):
            raise RuntimeError("exact checkpoint is ahead of speculative history")
        add_seconds("init_from_checkpoint_seconds", init_started)
        add_count("init_from_checkpoint_states")
    history = state.history
    add_count("actions_planned", len(history) - checkpoint_depth)
    failure_step = None
    direct_binding_method = None
    if prefer_direct_binding:
        if hasattr(graph, "apply_xfer_with_guid_binding"):
            direct_binding_method = "apply_xfer_with_guid_binding"
        elif hasattr(graph, "apply_xfer_with_node_id_binding"):
            direct_binding_method = "apply_xfer_with_node_id_binding"
        elif hasattr(graph, "apply_xfer_with_node_id_binding_trace"):
            direct_binding_method = "apply_xfer_with_node_id_binding_trace"
    step = checkpoint_depth + 1
    while step <= len(history):
        if replay_cache is not None:
            cache_lookup_started = now()
            cached_depth = None
            cached_entry = None
            for candidate_depth in range(len(history), step - 1, -1):
                entry = replay_cache.get(history[:candidate_depth])
                if entry is not None:
                    cached_depth = candidate_depth
                    cached_entry = entry
                    break
            add_seconds("replay_cache_lookup_seconds", cache_lookup_started)
            if cached_entry is not None:
                add_count("replay_cache_hits")
                add_count("actions_reused_from_cache", cached_depth - step + 1)
                if cached_entry.failure_step is not None:
                    failure_step = cached_entry.failure_step
                    break
                if (
                    cached_entry.graph is None
                    or cached_entry.guid_to_slot is None
                    or cached_entry.slot_to_guid is None
                ):
                    raise RuntimeError("corrupt exact replay cache entry")
                graph = cached_entry.graph
                guid_to_slot = dict(cached_entry.guid_to_slot)
                slot_to_guid = dict(cached_entry.slot_to_guid)
                step = cached_depth + 1
                continue
            add_count("replay_cache_misses")
        action = history[step - 1]
        add_count("actions_attempted")
        lookup_started = now()
        if direct_binding_method == "apply_xfer_with_guid_binding":
            source_guids = lookup_source_guids(action)
            add_seconds("source_guid_lookup_seconds", lookup_started)
            if source_guids is None:
                add_count("source_guid_missing_failures")
                failure_step = step
                if (
                    replay_cache is not None
                    and replay_cache_prefixes is not None
                    and history[:step] in replay_cache_prefixes
                ):
                    replay_cache[history[:step]] = ExactReplayCacheEntry(
                        None, None, None, failure_step
                    )
                    add_count("replay_cache_entries")
                break
            apply_started = now()
            result = graph.apply_xfer_with_guid_binding(
                xfer=xfers[action.xfer_id],
                source_node_guids=source_guids,
                eliminate_rotation=eliminate_rotation,
            )
            add_seconds("quartz_apply_seconds", apply_started)
            add_count("quartz_apply_calls")
            add_count("quartz_direct_guid_apply_calls")
        elif direct_binding_method is not None:
            source_node_ids = lookup_source_node_ids(action)
            add_seconds("source_node_lookup_seconds", lookup_started)
            if source_node_ids is None:
                add_count("source_node_missing_failures")
                failure_step = step
                if (
                    replay_cache is not None
                    and replay_cache_prefixes is not None
                    and history[:step] in replay_cache_prefixes
                ):
                    replay_cache[history[:step]] = ExactReplayCacheEntry(
                        None, None, None, failure_step
                    )
                    add_count("replay_cache_entries")
                break
            apply_started = now()
            if direct_binding_method == "apply_xfer_with_node_id_binding":
                result = graph.apply_xfer_with_node_id_binding(
                    xfer=xfers[action.xfer_id],
                    source_node_ids=source_node_ids,
                    eliminate_rotation=eliminate_rotation,
                )
            else:
                result = graph.apply_xfer_with_node_id_binding_trace(
                    xfer=xfers[action.xfer_id],
                    source_node_ids=source_node_ids,
                    eliminate_rotation=eliminate_rotation,
                    predecessor_layers=1,
                )
            add_seconds("quartz_apply_seconds", apply_started)
            add_count("quartz_apply_calls")
            add_count("quartz_direct_apply_calls")
        else:
            anchor = action.source_slots[0]
            anchor_node_id = None
            for index, graph_node in enumerate(graph.nodes):
                if guid_to_slot.get(int(graph_node.guid)) == anchor:
                    anchor_node_id = index
                    break
            if anchor_node_id is None:
                add_seconds("anchor_node_lookup_seconds", lookup_started)
                add_count("anchor_missing_failures")
                failure_step = step
                if (
                    replay_cache is not None
                    and replay_cache_prefixes is not None
                    and history[:step] in replay_cache_prefixes
                ):
                    replay_cache[history[:step]] = ExactReplayCacheEntry(
                        None, None, None, failure_step
                    )
                    add_count("replay_cache_entries")
                break
            node = graph.get_node_from_id(id=anchor_node_id)
            add_seconds("anchor_node_lookup_seconds", lookup_started)
            apply_started = now()
            result = graph.apply_xfer_with_binding_trace(
                xfer=xfers[action.xfer_id],
                node=node,
                eliminate_rotation=eliminate_rotation,
                predecessor_layers=1,
            )
            add_seconds("quartz_apply_seconds", apply_started)
            add_count("quartz_apply_calls")
            add_count("quartz_anchor_apply_calls")
        if result is None or result[0] is None:
            add_count("quartz_apply_failures")
            failure_step = step
            if (
                replay_cache is not None
                and replay_cache_prefixes is not None
                and history[:step] in replay_cache_prefixes
            ):
                replay_cache[history[:step]] = ExactReplayCacheEntry(
                    None, None, None, failure_step
                )
                add_count("replay_cache_entries")
            break
        validation_started = now()
        if direct_binding_method == "apply_xfer_with_guid_binding":
            next_graph, destination_guids = result
        else:
            next_graph, _, source_guids, destination_guids = result
        actual_binding = tuple(guid_to_slot[int(guid)] for guid in source_guids)
        if actual_binding != action.source_slots or len(destination_guids) != len(
            action.destination_slots
        ):
            add_seconds("binding_validation_seconds", validation_started)
            add_count("binding_mismatch_failures")
            failure_step = step
            if (
                replay_cache is not None
                and replay_cache_prefixes is not None
                and history[:step] in replay_cache_prefixes
            ):
                replay_cache[history[:step]] = ExactReplayCacheEntry(
                    None, None, None, failure_step
                )
                add_count("replay_cache_entries")
            break
        add_seconds("binding_validation_seconds", validation_started)
        update_started = now()
        live_guids = {int(node.guid) for node in next_graph.nodes}
        surviving_destination_pairs = tuple(
            (int(guid), int(slot))
            for guid, slot in zip(destination_guids, action.destination_slots)
            if int(guid) in live_guids
        )
        for guid, slot in surviving_destination_pairs:
            guid_to_slot[int(guid)] = int(slot)
            slot_to_guid[int(slot)] = int(guid)
        graph = next_graph
        add_seconds("slot_update_and_graph_swap_seconds", update_started)
        add_count("actions_applied")
        if (
            replay_cache is not None
            and replay_cache_prefixes is not None
            and history[:step] in replay_cache_prefixes
        ):
            replay_cache[history[:step]] = ExactReplayCacheEntry(
                graph, dict(guid_to_slot), dict(slot_to_guid), None
            )
            add_count("replay_cache_entries")
        step += 1
    if failure_step is not None:
        add_count("failed_states")
        result = (None, failure_step, False)
        return (*result, None) if return_checkpoint else result
    signature_started = now()
    exact_signature = graph_topology_signature(graph, guid_to_slot)
    expected_signature = (
        indexed_topology_signature(state.topology_index)
        if state.topology_index is not None
        else snapshot_signature(state.snapshot)
    )
    topology_matches = exact_signature == expected_signature
    add_seconds("topology_signature_compare_seconds", signature_started)
    add_count("valid_states")
    if not topology_matches:
        add_count("topology_mismatch_states")
    result = (graph, None, topology_matches)
    return (*result, guid_to_slot) if return_checkpoint else result


def parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    result = [int(item) for item in value.split(",")]
    if any(item <= 0 for item in result):
        raise ValueError("batch sizes must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.97)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--max-source-matches", type=int, default=2048)
    parser.add_argument("--max-actions-per-parent", type=int, default=128)
    parser.add_argument("--proposal-factor", type=int, default=16)
    parser.add_argument("--max-gate-increase", type=int, default=1)
    parser.add_argument(
        "--dedup-mode", choices=("none", "raw", "canonical"), default="raw"
    )
    parser.add_argument("--structural-recheck", action="store_true")
    parser.add_argument("--audit-count", type=int, default=1000)
    parser.add_argument("--batch-sweep", default="1,2,4,8,16,32,64")
    parser.add_argument("--batch-sweep-repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-qasm", type=Path)
    parser.add_argument(
        "--eliminate-rotation",
        action="store_true",
        help=(
            "use Quarl-compatible zero-rotation normalization during exact replay; "
            "the lazy speculative topology remains an upper-bound approximation"
        ),
    )
    args = parser.parse_args()

    import quartz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    if context.num_xfers != len(rules.xfer_to_source):
        raise RuntimeError("dataset and Quartz context have different xfer counts")
    xfers = [context.get_xfer_from_id(id=index) for index in range(context.num_xfers)]
    source_to_xfers: dict[int, list[int]] = defaultdict(list)
    for xfer_id, source_id in enumerate(rules.xfer_to_source):
        source_to_xfers[source_id].append(xfer_id)
    gate_deltas = [
        len(rules.destination_gate_types[index])
        - len(rules.source_gate_types[rules.xfer_to_source[index]])
        for index in range(len(rules.xfer_to_source))
    ]
    source_patterns = tuple(parse_pattern(pattern) for pattern in rules.xfer_sources)
    destination_patterns = tuple(
        parse_pattern(pattern) for pattern in rules.xfer_destinations
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    model = S0ActionBindingModel(
        rules,
        num_xfers=len(rules.xfer_to_source),
        width=train_args["width"],
        retrieval_width=train_args["retrieval_width"],
        graph_layers=train_args["graph_layers"],
        current_graph_layers=train_args["current_graph_layers"],
        use_action_history=not train_args.get("state_only", False),
        use_locality_features=train_args.get("locality_features", False),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    threshold_config = load_threshold_config(args.calibration, args.target_recall)

    graph = quartz.PyGraph.from_qasm(context=context, filename=str(args.qasm))
    initial_qasm = graph.to_qasm_str()
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    initial_snapshot = snapshot(graph, guid_to_slot)
    initial_gate_count = int(graph.gate_count)
    beam = [
        BeamState(
            graph=None,
            snapshot=initial_snapshot,
            guid_to_slot={},
            next_slot=next_slot,
            last_touched={},
            rewrite_distance={int(row[0]): 5 for row in initial_snapshot["nodes"]},
            previous_preferred=set(),
            local_streak=0,
            gate_count=initial_gate_count,
            depth=0,
            history=(),
            exact_graph_checkpoint=graph,
            exact_slot_checkpoint=dict(guid_to_slot),
            exact_checkpoint_depth=0,
        )
    ]
    model_matches(
        beam,
        model,
        device,
        threshold_config,
        args.microbatch,
        args.max_source_matches,
    )
    del payload, graph
    gc.collect()
    gc.disable()

    if args.dedup_mode == "canonical":
        seen = {topology_digest(initial_snapshot)}
    elif args.dedup_mode == "raw":
        seen = {raw_topology_hash(initial_snapshot)}
    else:
        seen = set()
    step_rows = []
    total_started = time.perf_counter()
    for step in range(args.depth):
        step_started = time.perf_counter()
        predicted, model_seconds = model_matches(
            beam,
            model,
            device,
            threshold_config,
            args.microbatch,
            args.max_source_matches,
        )
        action_rows = []
        for rows in predicted:
            expanded = []
            for source, anchor, binding, probability in rows:
                expanded.extend(
                    (xfer_id, anchor, binding, probability)
                    for xfer_id in source_to_xfers[source]
                )
            action_rows.append(expanded)

        proposal_started = time.perf_counter()
        proposals = []
        effective_parent_cap = max(
            args.max_actions_per_parent,
            math.ceil(args.beam_size / max(1, len(beam))) * 2,
        )
        eligible_actions = 0
        for parent_index, (state, rows) in enumerate(zip(beam, action_rows)):
            parent_proposals = []
            for xfer_id, anchor, binding, probability in rows:
                delta = gate_deltas[xfer_id]
                if delta > args.max_gate_increase:
                    continue
                parent_proposals.append(
                    Proposal(
                        parent=parent_index,
                        xfer_id=xfer_id,
                        anchor_slot=anchor,
                        binding=binding,
                        probability=probability,
                        next_gate_count=state.gate_count + delta,
                    )
                )
            eligible_actions += len(parent_proposals)
            parent_proposals.sort(
                key=lambda row: (row.next_gate_count, -row.probability, row.xfer_id)
            )
            proposals.extend(parent_proposals[:effective_parent_cap])
        proposals.sort(
            key=lambda row: (
                row.next_gate_count,
                -row.probability,
                beam[row.parent].gate_count,
            )
        )
        proposals = proposals[: args.beam_size * args.proposal_factor]
        proposal_seconds = time.perf_counter() - proposal_started

        update_started = time.perf_counter()
        children = []
        attempted = invalid = duplicates = 0
        for proposal in proposals:
            if len(children) >= args.beam_size:
                break
            attempted += 1
            child, fingerprint, is_duplicate = lazy_child(
                beam[proposal.parent],
                proposal,
                source_patterns,
                destination_patterns,
                args.structural_recheck,
                args.dedup_mode,
                seen,
            )
            if is_duplicate:
                duplicates += 1
                continue
            if child is None:
                invalid += 1
                continue
            if fingerprint is not None:
                seen.add(fingerprint)
            children.append(child)
        update_seconds = time.perf_counter() - update_started
        if not children:
            break
        children.sort(key=lambda state: (state.gate_count, len(state.history)))
        beam = children[: args.beam_size]
        elapsed = time.perf_counter() - step_started
        matched_actions = sum(map(len, action_rows))
        row = {
            "step": step + 1,
            "input_states": len(action_rows),
            "output_states": len(beam),
            "best_speculative_gate_count": beam[0].gate_count,
            "predicted_actions": matched_actions,
            "eligible_actions_before_parent_cap": eligible_actions,
            "proposals_after_caps": len(proposals),
            "attempted_actions": attempted,
            "accepted_actions": len(beam),
            "invalid_structural_actions": invalid,
            "duplicate_speculative_successors": duplicates,
            "model_match_seconds": model_seconds,
            "proposal_seconds": proposal_seconds,
            "lazy_update_and_hash_seconds": update_seconds,
            "total_seconds": elapsed,
            "match_states_per_second": len(action_rows) / max(model_seconds, 1e-12),
            "predicted_actions_per_second": matched_actions
            / max(model_seconds, 1e-12),
            "accepted_actions_per_second": len(beam) / max(elapsed, 1e-12),
            "lazy_attempts_per_second": attempted / max(update_seconds, 1e-12),
        }
        step_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    search_seconds = time.perf_counter() - total_started

    sweep_rows = []
    for microbatch in parse_int_list(args.batch_sweep):
        measurements = []
        action_counts = []
        for _ in range(args.batch_sweep_repeats):
            rows, seconds = model_matches(
                beam,
                model,
                device,
                threshold_config,
                microbatch,
                args.max_source_matches,
            )
            measurements.append(seconds)
            action_counts.append(sum(map(len, rows)))
        best_seconds = min(measurements)
        row = {
            "microbatch": microbatch,
            "states": len(beam),
            "seconds_best": best_seconds,
            "seconds_all": measurements,
            "source_matches": max(action_counts),
            "states_per_second": len(beam) / best_seconds,
            "source_matches_per_second": max(action_counts) / best_seconds,
        }
        sweep_rows.append(row)
        print(json.dumps({"batch_sweep": row}, sort_keys=True), flush=True)

    audit_started = time.perf_counter()
    audited = min(args.audit_count, len(beam))
    valid = topology_matches = 0
    exact_hashes = set()
    failure_steps: dict[int, int] = defaultdict(int)
    best_valid_gate_count = None
    best_valid_graph = None
    for state in beam[:audited]:
        exact_graph, failure_step, topology_ok = replay_state(
            state,
            context,
            quartz.PyGraph,
            xfers,
            initial_qasm,
            eliminate_rotation=args.eliminate_rotation,
        )
        if exact_graph is None:
            failure_steps[int(failure_step)] += 1
            continue
        valid += 1
        topology_matches += int(topology_ok)
        exact_hashes.add(int(exact_graph.hash()))
        gate_count = int(exact_graph.gate_count)
        if best_valid_gate_count is None or gate_count < best_valid_gate_count:
            best_valid_gate_count = gate_count
            best_valid_graph = exact_graph
    audit_seconds = time.perf_counter() - audit_started
    audit = {
        "audited_states": audited,
        "valid_trajectories": valid,
        "valid_trajectory_rate": valid / max(1, audited),
        "exact_topology_matches": topology_matches,
        "unique_exact_graph_hashes": len(exact_hashes),
        "failure_steps": dict(sorted(failure_steps.items())),
        "best_valid_gate_count": best_valid_gate_count,
        "audit_seconds": audit_seconds,
    }
    total_accepted = sum(row["accepted_actions"] for row in step_rows)
    total = {
        "mode": "lazy_model",
        "eliminate_rotation": args.eliminate_rotation,
        "speculative_normalization": False,
        "qasm": str(args.qasm),
        "beam_size": args.beam_size,
        "requested_depth": args.depth,
        "completed_depth": len(step_rows),
        "initial_gate_count": initial_gate_count,
        "best_speculative_gate_count": min(state.gate_count for state in beam),
        "final_beam_size": len(beam),
        "dedup_mode": args.dedup_mode,
        "speculative_graphs_seen": len(seen) if seen else None,
        "search_seconds_excluding_audit_and_sweep": search_seconds,
        "accepted_actions": total_accepted,
        "accepted_actions_per_search_second": total_accepted / search_seconds,
        "target_recall": args.target_recall,
        "microbatch": args.microbatch,
        "steps": step_rows,
        "batch_sweep": sweep_rows,
        "quartz_replay_audit": audit,
    }
    rendered = json.dumps(total, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    if args.best_qasm is not None and best_valid_graph is not None:
        args.best_qasm.parent.mkdir(parents=True, exist_ok=True)
        best_valid_graph.to_qasm(filename=str(args.best_qasm))


if __name__ == "__main__":
    main()
