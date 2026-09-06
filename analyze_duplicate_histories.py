from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys
import types
from typing import Any, Iterable

import torch

from beam_search_benchmark import update_slots
from dataset import RuleMetadata


@dataclass(frozen=True, order=True)
class Action:
    xfer_id: int
    source_slots: tuple[int, ...]
    destination_slots: tuple[int, ...]


@dataclass
class ReplayRecord:
    graph: Any
    guid_to_slot: dict[int, int]
    slot_to_guid: dict[int, int]
    graph_hash: int
    qasm: str
    canonical_qasm: tuple[str, ...]
    canonical_qasm_without_parameters: tuple[str, ...]


_QUBIT_RE = re.compile(r"q\[(\d+)\]")
_GATE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\((.*)\))?\s+(.+);$")


def parse_action(row: dict) -> Action:
    return Action(
        xfer_id=int(row["xfer_id"]),
        source_slots=tuple(map(int, row["source_slots"])),
        destination_slots=tuple(map(int, row["destination_slots"])),
    )


def canonical_qasm_operations(
    qasm: str, *, include_parameters: bool
) -> tuple[str, ...]:
    """Canonicalize a QASM circuit modulo ordering of independent operations.

    Quartz emits a sequential QASM listing even when two operations are unrelated.
    The per-qubit predecessor DAG recovers the relevant partial order.  A stable
    lexical topological traversal then gives identical independent-order variants
    the same representation while retaining gate parameters and physical qubits.
    """
    labels: list[str] = []
    predecessors: list[set[int]] = []
    successors: list[list[int]] = []
    last_on_qubit: dict[int, int] = {}
    for raw_line in qasm.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg")):
            continue
        match = _GATE_RE.match(line)
        if match is None:
            raise ValueError(f"unsupported QASM instruction: {line!r}")
        gate, parameters, operands = match.groups()
        qubits = tuple(map(int, _QUBIT_RE.findall(operands)))
        if not qubits:
            raise ValueError(f"instruction has no q[] operand: {line!r}")
        parameter_text = parameters.strip() if parameters and include_parameters else ""
        label = f"{gate}({parameter_text})@{','.join(map(str, qubits))}"
        index = len(labels)
        labels.append(label)
        deps = {last_on_qubit[q] for q in qubits if q in last_on_qubit}
        predecessors.append(deps)
        successors.append([])
        for predecessor in deps:
            successors[predecessor].append(index)
        for qubit in qubits:
            last_on_qubit[qubit] = index

    remaining = [len(row) for row in predecessors]
    ready: list[tuple[str, int]] = [
        (labels[index], index)
        for index, degree in enumerate(remaining)
        if degree == 0
    ]
    heapq.heapify(ready)
    result: list[str] = []
    while ready:
        label, index = heapq.heappop(ready)
        result.append(label)
        for successor in successors[index]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                heapq.heappush(ready, (labels[successor], successor))
    if len(result) != len(labels):
        raise ValueError("QASM dependency graph is cyclic")
    return tuple(result)


def make_record(graph, guid_to_slot: dict[int, int]) -> ReplayRecord:
    qasm = graph.to_qasm_str()
    return ReplayRecord(
        graph=graph,
        guid_to_slot=guid_to_slot,
        slot_to_guid={slot: guid for guid, slot in guid_to_slot.items()},
        graph_hash=int(graph.hash()),
        qasm=qasm,
        canonical_qasm=canonical_qasm_operations(qasm, include_parameters=True),
        canonical_qasm_without_parameters=canonical_qasm_operations(
            qasm, include_parameters=False
        ),
    )


def slot_topology_signature(record: ReplayRecord) -> tuple:
    nodes = list(record.graph.nodes)
    return (
        tuple(
            sorted(
                (record.guid_to_slot[int(node.guid)], int(node.gate_tp))
                for node in nodes
            )
        ),
        tuple(
            sorted(
                (
                    record.guid_to_slot[int(nodes[int(src)].guid)],
                    record.guid_to_slot[int(nodes[int(dst)].guid)],
                    int(src_port),
                    int(dst_port),
                )
                for src, dst, src_port, dst_port in record.graph.all_edges()
            )
        ),
    )


def apply_action(parent: ReplayRecord, action: Action, xfers) -> ReplayRecord:
    source_guids = [parent.slot_to_guid.get(slot) for slot in action.source_slots]
    if any(guid is None for guid in source_guids):
        raise RuntimeError(f"source slot is absent during replay: {action}")
    result = parent.graph.apply_xfer_with_guid_binding(
        xfer=xfers[action.xfer_id],
        source_node_guids=list(map(int, source_guids)),
        eliminate_rotation=True,
    )
    if result is None or result[0] is None:
        raise RuntimeError(f"Quartz rejected exported action: {action}")
    graph, destination_guids = result
    if len(destination_guids) != len(action.destination_slots):
        raise RuntimeError(
            "destination count changed during replay: "
            f"{len(destination_guids)} != {len(action.destination_slots)}"
        )
    guid_to_slot = dict(parent.guid_to_slot)
    live_guids = {int(node.guid) for node in graph.nodes}
    for guid, slot in zip(destination_guids, action.destination_slots):
        if int(guid) in live_guids:
            guid_to_slot[int(guid)] = int(slot)
    return make_record(graph, guid_to_slot)


def signature_digest(signature: tuple[str, ...]) -> str:
    payload = "\n".join(signature).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def action_token(action: Action) -> tuple[int, tuple[int, ...]]:
    # Destination slots are allocation metadata, not the matched location.
    return action.xfer_id, action.source_slots


def multiset_key(values: Iterable[Any]) -> tuple[tuple[Any, int], ...]:
    return tuple(sorted(Counter(values).items()))


def normalized_pattern(value: str) -> str:
    return "".join(value.split()).rstrip(";")


def describe_action(action: Action, rules: RuleMetadata) -> dict:
    return {
        "xfer_id": action.xfer_id,
        "source_slots": list(action.source_slots),
        "destination_slots": list(action.destination_slots),
        "source_pattern": rules.xfer_sources[action.xfer_id],
        "destination_pattern": rules.xfer_destinations[action.xfer_id],
    }


def pairwise_duplicate_statistics(
    histories: list[tuple[Action, ...]],
    groups: dict[Any, list[int]],
) -> dict:
    total_pairs = 0
    same_action_multiset = 0
    same_xfer_multiset = 0
    same_xfer_sequence = 0
    different_xfer_multiset = 0
    states_with_observed_order_variant: set[int] = set()
    example_pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for left, right in itertools.combinations(indices, 2):
            total_pairs += 1
            left_history = histories[left]
            right_history = histories[right]
            left_tokens = tuple(map(action_token, left_history))
            right_tokens = tuple(map(action_token, right_history))
            if multiset_key(left_tokens) == multiset_key(right_tokens):
                same_action_multiset += 1
                if left_tokens != right_tokens:
                    states_with_observed_order_variant.update((left, right))
                    if len(example_pairs) < 5:
                        example_pairs.append((left, right))
            left_xfers = tuple(action.xfer_id for action in left_history)
            right_xfers = tuple(action.xfer_id for action in right_history)
            if left_xfers == right_xfers:
                same_xfer_sequence += 1
            if multiset_key(left_xfers) == multiset_key(right_xfers):
                same_xfer_multiset += 1
            else:
                different_xfer_multiset += 1
    return {
        "duplicate_pairs": total_pairs,
        "pairs_same_concrete_action_multiset": same_action_multiset,
        "pairs_same_xfer_multiset": same_xfer_multiset,
        "pairs_same_xfer_sequence": same_xfer_sequence,
        "pairs_different_xfer_multiset": different_xfer_multiset,
        "states_with_observed_order_variant": len(states_with_observed_order_variant),
        "order_variant_example_pairs": [list(row) for row in example_pairs],
    }


def group_summary(groups: dict[Any, list[int]]) -> dict:
    sizes = sorted((len(rows) for rows in groups.values()), reverse=True)
    repeated = [size for size in sizes if size > 1]
    return {
        "unique": len(groups),
        "duplicate_excess": sum(size - 1 for size in sizes),
        "duplicate_groups": len(repeated),
        "largest_group": max(sizes, default=0),
        "largest_group_sizes": sizes[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Quartz imports these conversion helpers unconditionally even though this
    # analyzer only needs the compiled graph API.
    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    payload = json.loads(args.histories.read_text())
    histories = [
        tuple(parse_action(action) for action in state["history"])
        for state in payload["states"]
    ]
    rule_payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(rule_payload)
    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    if context.num_xfers != len(rules.xfer_to_source):
        raise RuntimeError("dataset and Quartz context have different xfer counts")
    xfers = [context.get_xfer_from_id(id=i) for i in range(context.num_xfers)]

    initial_graph = quartz.PyGraph.from_qasm_str(
        context=context, qasm_str=payload["initial_qasm"]
    )
    guid_to_slot: dict[int, int] = {}
    update_slots(initial_graph, guid_to_slot, 0)
    records: dict[tuple[Action, ...], ReplayRecord] = {
        (): make_record(initial_graph, guid_to_slot)
    }
    def get_record(history: tuple[Action, ...]) -> ReplayRecord:
        if history not in records:
            records[history] = apply_action(get_record(history[:-1]), history[-1], xfers)
        return records[history]

    initial_next_slot = max(guid_to_slot.values(), default=-1) + 1

    def next_slot_after(history: tuple[Action, ...]) -> int:
        return initial_next_slot + sum(
            len(action.destination_slots) for action in history
        )

    def swapped_adjacent_prefix(
        history: tuple[Action, ...], offset: int
    ) -> tuple[Action, ...] | None:
        """Return a slot-rebased adjacent swap when Quartz proves it commutes."""
        left = history[offset]
        right = history[offset + 1]
        # A right action that consumes the left action's output is dependent.
        if set(right.source_slots) & set(left.destination_slots):
            return None
        parent = history[:offset]
        next_slot = next_slot_after(parent)
        swapped_right = Action(
            xfer_id=right.xfer_id,
            source_slots=right.source_slots,
            destination_slots=tuple(
                range(next_slot, next_slot + len(right.destination_slots))
            ),
        )
        next_slot += len(right.destination_slots)
        swapped_left = Action(
            xfer_id=left.xfer_id,
            source_slots=left.source_slots,
            destination_slots=tuple(
                range(next_slot, next_slot + len(left.destination_slots))
            ),
        )
        swapped = parent + (swapped_right, swapped_left)
        try:
            swapped_record = get_record(swapped)
        except RuntimeError:
            return None
        original_record = get_record(history[: offset + 2])
        if swapped_record.canonical_qasm != original_record.canonical_qasm:
            return None
        return swapped

    for depth in range(1, max(map(len, histories), default=0) + 1):
        prefixes = sorted({history[:depth] for history in histories})
        for prefix in prefixes:
            get_record(prefix)

    final_records = [records[history] for history in histories]
    by_hash: dict[int, list[int]] = defaultdict(list)
    by_qasm: dict[str, list[int]] = defaultdict(list)
    by_canonical: dict[tuple[str, ...], list[int]] = defaultdict(list)
    by_canonical_without_parameters: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(final_records):
        by_hash[record.graph_hash].append(index)
        by_qasm[record.qasm].append(index)
        by_canonical[record.canonical_qasm].append(index)
        by_canonical_without_parameters[
            record.canonical_qasm_without_parameters
        ].append(index)

    hash_collision_groups = []
    hash_parameter_collision_groups = 0
    hash_structural_collision_groups = 0
    for graph_hash, indices in by_hash.items():
        canonical = {final_records[index].canonical_qasm for index in indices}
        no_parameters = {
            final_records[index].canonical_qasm_without_parameters
            for index in indices
        }
        if len(canonical) > 1:
            if len(no_parameters) < len(canonical):
                hash_parameter_collision_groups += 1
            if len(no_parameters) > 1:
                hash_structural_collision_groups += 1
            hash_collision_groups.append(
                {
                    "graph_hash": graph_hash,
                    "states": len(indices),
                    "canonical_circuits": len(canonical),
                    "canonical_without_parameters": len(no_parameters),
                    "example_indices": indices[:8],
                    "example_digests": [
                        signature_digest(final_records[index].canonical_qasm)
                        for index in indices[:8]
                    ],
                }
            )
    hash_collision_groups.sort(
        key=lambda row: (row["canonical_circuits"], row["states"]), reverse=True
    )

    depth_summary = []
    root_signature = records[()].canonical_qasm
    cycle_trajectories = 0
    cycle_to_root = 0
    cycle_examples = []
    inverse_adjacent_trajectories = 0
    inverse_examples = []
    direct_undo_trajectories = 0
    direct_undo_occurrences = 0
    exact_two_step_cycle_trajectories = 0
    exact_two_step_cycle_occurrences = 0
    shortest_equivalent_subsequence = Counter()
    shortest_by_index: dict[int, int] = {}
    self_redundant_examples = []
    normalized_sources = list(map(normalized_pattern, rules.xfer_sources))
    normalized_destinations = list(map(normalized_pattern, rules.xfer_destinations))
    inverse_pairs = {
        (left, right)
        for left in range(len(normalized_sources))
        for right in range(len(normalized_sources))
        if normalized_sources[left] == normalized_destinations[right]
        and normalized_destinations[left] == normalized_sources[right]
    }
    for index, history in enumerate(histories):
        signatures = [records[history[:depth]].canonical_qasm for depth in range(len(history) + 1)]
        if signatures[-1] in signatures[:-1]:
            cycle_trajectories += 1
            if signatures[-1] == root_signature:
                cycle_to_root += 1
            if len(cycle_examples) < 5:
                cycle_examples.append(index)
        has_inverse = any(
            (history[offset].xfer_id, history[offset + 1].xfer_id) in inverse_pairs
            for offset in range(len(history) - 1)
        )
        if has_inverse:
            inverse_adjacent_trajectories += 1
            if len(inverse_examples) < 5:
                inverse_examples.append(index)
        direct_undos = sum(
            (history[offset].xfer_id, history[offset + 1].xfer_id) in inverse_pairs
            and history[offset + 1].source_slots
            == history[offset].destination_slots
            for offset in range(len(history) - 1)
        )
        direct_undo_occurrences += direct_undos
        direct_undo_trajectories += direct_undos > 0
        two_step_cycles = sum(
            records[history[:offset]].canonical_qasm
            == records[history[: offset + 2]].canonical_qasm
            for offset in range(len(history) - 1)
        )
        exact_two_step_cycle_occurrences += two_step_cycles
        exact_two_step_cycle_trajectories += two_step_cycles > 0

        final_signature = records[history].canonical_qasm
        shortest = len(history)
        shortest_positions: tuple[int, ...] | None = None
        for size in range(len(history)):
            for positions in itertools.combinations(range(len(history)), size):
                subsequence = tuple(history[position] for position in positions)
                try:
                    subsequence_record = get_record(subsequence)
                except RuntimeError:
                    continue
                if subsequence_record.canonical_qasm == final_signature:
                    shortest = size
                    shortest_positions = positions
                    break
            if shortest_positions is not None:
                break
        shortest_equivalent_subsequence[shortest] += 1
        shortest_by_index[index] = shortest
        if shortest < len(history) and len(self_redundant_examples) < 10:
            self_redundant_examples.append(
                {
                    "index": index,
                    "history_length": len(history),
                    "shortest_equivalent_subsequence_length": shortest,
                    "kept_action_positions": list(shortest_positions or ()),
                }
            )

    max_depth = max(map(len, histories), default=0)
    for depth in range(max_depth + 1):
        prefixes = sorted({history[:depth] for history in histories})
        canonical_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        hash_groups: dict[int, list[int]] = defaultdict(list)
        slot_topology_groups: dict[tuple, list[int]] = defaultdict(list)
        for index, prefix in enumerate(prefixes):
            record = records[prefix]
            canonical_groups[record.canonical_qasm].append(index)
            hash_groups[record.graph_hash].append(index)
            slot_topology_groups[slot_topology_signature(record)].append(index)
        parent_child_pairs = {
            (prefix[:-1], records[prefix].canonical_qasm)
            for prefix in prefixes
        } if depth else {((), records[()].canonical_qasm)}
        canonical_parent_child_pairs = {
            (
                records[prefix[:-1]].canonical_qasm,
                records[prefix].canonical_qasm,
            )
            for prefix in prefixes
        } if depth else {(records[()].canonical_qasm, records[()].canonical_qasm)}
        pairwise_at_depth = pairwise_duplicate_statistics(prefixes, canonical_groups)
        commutable_last_pair = 0
        noncanonical_commutable_last_pair = 0
        commutable_last_pair_examples = []
        if depth >= 2:
            for prefix in prefixes:
                swapped = swapped_adjacent_prefix(prefix, depth - 2)
                if swapped is None:
                    continue
                commutable_last_pair += 1
                left_key = action_token(prefix[-2])
                right_key = action_token(prefix[-1])
                if left_key > right_key:
                    noncanonical_commutable_last_pair += 1
                if len(commutable_last_pair_examples) < 5:
                    commutable_last_pair_examples.append(
                        {
                            "original": [
                                describe_action(action, rules)
                                for action in prefix[-2:]
                            ],
                            "swapped": [
                                describe_action(action, rules)
                                for action in swapped[-2:]
                            ],
                            "original_keys": [
                                repr(action_token(prefix[-2])),
                                repr(action_token(prefix[-1])),
                            ],
                        }
                    )
        depth_summary.append(
            {
                "depth": depth,
                "unique_action_prefixes": len(prefixes),
                "slot_sensitive_exact_topologies": group_summary(slot_topology_groups),
                "quartz_hash": group_summary(hash_groups),
                "canonical_qasm_dag": group_summary(canonical_groups),
                "same_exact_parent_same_child_excess": (
                    len(prefixes) - len(parent_child_pairs)
                ),
                "unique_canonical_parent_child_transitions": len(
                    canonical_parent_child_pairs
                ),
                "duplicate_pair_analysis": pairwise_at_depth,
                "quartz_proven_commutable_last_pair_prefixes": (
                    commutable_last_pair
                ),
                "commutable_last_pair_noncanonical_by_source_xfer_key": (
                    noncanonical_commutable_last_pair
                ),
                "commutable_last_pair_examples": commutable_last_pair_examples,
            }
        )

    pairwise = pairwise_duplicate_statistics(histories, by_canonical)
    concrete_sequence_counts = Counter(
        tuple(map(action_token, history)) for history in histories
    )
    xfer_sequence_counts = Counter(
        tuple(action.xfer_id for action in history) for history in histories
    )
    concrete_multiset_counts = Counter(
        multiset_key(map(action_token, history)) for history in histories
    )
    xfer_multiset_counts = Counter(
        multiset_key(action.xfer_id for action in history) for history in histories
    )
    canonical_and_concrete_multiset = {
        (
            final_records[index].canonical_qasm,
            multiset_key(map(action_token, history)),
        )
        for index, history in enumerate(histories)
    }
    canonical_and_xfer_multiset = {
        (
            final_records[index].canonical_qasm,
            multiset_key(action.xfer_id for action in history),
        )
        for index, history in enumerate(histories)
    }
    irreducible_indices = [
        index
        for index, history in enumerate(histories)
        if shortest_by_index[index] == len(history)
    ]
    irreducible_canonical_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in irreducible_indices:
        irreducible_canonical_groups[final_records[index].canonical_qasm].append(index)
    xfer_frequency = Counter(
        action.xfer_id for history in histories for action in history
    )
    redundant_xfer_frequency = Counter(
        action.xfer_id
        for index, history in enumerate(histories)
        if shortest_by_index[index] < len(history)
        for action in history
    )
    adjacent_xfer_pairs = Counter(
        (history[offset].xfer_id, history[offset + 1].xfer_id)
        for history in histories
        for offset in range(len(history) - 1)
    )
    commutable_adjacent_occurrences = 0
    commutable_adjacent_trajectories = 0
    noncanonical_commutable_adjacent_occurrences = 0
    noncanonical_commutable_adjacent_trajectories = 0
    for history in histories:
        history_commutable = 0
        history_noncanonical = 0
        for offset in range(len(history) - 1):
            if swapped_adjacent_prefix(history, offset) is None:
                continue
            history_commutable += 1
            if action_token(history[offset]) > action_token(history[offset + 1]):
                history_noncanonical += 1
        commutable_adjacent_occurrences += history_commutable
        commutable_adjacent_trajectories += history_commutable > 0
        noncanonical_commutable_adjacent_occurrences += history_noncanonical
        noncanonical_commutable_adjacent_trajectories += history_noncanonical > 0

    order_examples = []
    for left, right in pairwise["order_variant_example_pairs"]:
        order_examples.append(
            {
                "indices": [left, right],
                "final_digest": signature_digest(final_records[left].canonical_qasm),
                "left": [describe_action(action, rules) for action in histories[left]],
                "right": [describe_action(action, rules) for action in histories[right]],
            }
        )

    largest_duplicate_examples = []
    for signature, indices in sorted(
        by_canonical.items(), key=lambda row: len(row[1]), reverse=True
    )[:10]:
        if len(indices) < 2:
            continue
        largest_duplicate_examples.append(
            {
                "canonical_digest": signature_digest(signature),
                "states": len(indices),
                "indices": indices[:10],
                "histories": [
                    [describe_action(action, rules) for action in histories[index]]
                    for index in indices[:3]
                ],
            }
        )

    result = {
        "input": {
            "histories": str(args.histories),
            "states": len(histories),
            "history_depths": dict(sorted(Counter(map(len, histories)).items())),
            "unique_serialized_histories": len(set(histories)),
        },
        "final_state_uniqueness": {
            "quartz_graph_hash": group_summary(by_hash),
            "raw_qasm": group_summary(by_qasm),
            "canonical_qasm_dag": group_summary(by_canonical),
            "canonical_qasm_dag_without_parameters": group_summary(
                by_canonical_without_parameters
            ),
        },
        "quartz_hash_collision_audit": {
            "source_limitation": (
                "Graph::hash ignores constant parameters, seeds all input qubits "
                "identically, and sums node hashes"
            ),
            "hash_groups_containing_multiple_canonical_circuits": len(
                hash_collision_groups
            ),
            "groups_with_parameter_distinctions": hash_parameter_collision_groups,
            "groups_with_structural_or_qubit_distinctions": (
                hash_structural_collision_groups
            ),
            "canonical_circuits_minus_hashes": len(by_canonical) - len(by_hash),
            "largest_collision_groups": hash_collision_groups[:20],
        },
        "prefix_uniqueness": depth_summary,
        "action_sequence_uniqueness": {
            "concrete_sequences": len(concrete_sequence_counts),
            "xfer_id_sequences": len(xfer_sequence_counts),
            "concrete_action_multisets": len(concrete_multiset_counts),
            "xfer_id_multisets": len(xfer_multiset_counts),
            "same_final_same_concrete_multiset_excess": (
                len(histories) - len(canonical_and_concrete_multiset)
            ),
            "same_final_same_xfer_multiset_excess": (
                len(histories) - len(canonical_and_xfer_multiset)
            ),
            "top_xfer_ids": [list(row) for row in xfer_frequency.most_common(20)],
            "top_xfer_ids_in_self_redundant_histories": [
                list(row) for row in redundant_xfer_frequency.most_common(20)
            ],
            "top_adjacent_xfer_pairs": [
                [list(pair), count]
                for pair, count in adjacent_xfer_pairs.most_common(20)
            ],
        },
        "canonical_duplicate_pair_analysis": pairwise,
        "independent_action_ordering": {
            "quartz_proven_commutable_adjacent_occurrences": (
                commutable_adjacent_occurrences
            ),
            "trajectories_with_commutable_adjacent_pair": (
                commutable_adjacent_trajectories
            ),
            "noncanonical_commutable_occurrences_by_concrete_key": (
                noncanonical_commutable_adjacent_occurrences
            ),
            "trajectories_with_noncanonical_commutable_pair": (
                noncanonical_commutable_adjacent_trajectories
            ),
            "observed_same_final_same_concrete_multiset_excess": (
                len(histories) - len(canonical_and_concrete_multiset)
            ),
            "observed_same_final_same_concrete_multiset_pairs": (
                pairwise["pairs_same_concrete_action_multiset"]
            ),
            "observed_states_with_order_variant": (
                pairwise["states_with_observed_order_variant"]
            ),
        },
        "cycles": {
            "final_equals_an_earlier_prefix": cycle_trajectories,
            "final_equals_root": cycle_to_root,
            "example_indices": cycle_examples,
            "adjacent_inverse_pattern_pair": inverse_adjacent_trajectories,
            "inverse_example_indices": inverse_examples,
            "inverse_xfer_pairs_in_ecc": len(inverse_pairs),
            "direct_inverse_on_previous_destination_trajectories": (
                direct_undo_trajectories
            ),
            "direct_inverse_on_previous_destination_occurrences": (
                direct_undo_occurrences
            ),
            "exact_two_step_cycle_trajectories": exact_two_step_cycle_trajectories,
            "exact_two_step_cycle_occurrences": exact_two_step_cycle_occurrences,
            "shortest_equivalent_subsequence_length": dict(
                sorted(shortest_equivalent_subsequence.items())
            ),
            "self_redundant_trajectory_count": sum(
                count
                for length, count in shortest_equivalent_subsequence.items()
                if length < max_depth
            ),
            "canonical_uniqueness_after_removing_self_redundant_paths": (
                group_summary(irreducible_canonical_groups)
            ),
            "self_redundant_examples": self_redundant_examples,
        },
        "observed_order_examples": order_examples,
        "largest_canonical_duplicate_examples": largest_duplicate_examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
