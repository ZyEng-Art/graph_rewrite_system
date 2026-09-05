from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from dataset import GATE_NAME_TO_ID


@dataclass(frozen=True)
class PatternOperation:
    gate_type: int
    qubits: tuple[int, ...]


@lru_cache(maxsize=None)
def parse_pattern(pattern: str) -> tuple[PatternOperation, ...]:
    operations = []
    for instruction in pattern.split(";"):
        fields = instruction.strip().split()
        if not fields:
            continue
        name = fields[0]
        if name not in GATE_NAME_TO_ID:
            raise ValueError(f"unknown gate {name}")
        operations.append(
            PatternOperation(
                gate_type=GATE_NAME_TO_ID[name],
                qubits=tuple(map(int, fields[1:])),
            )
        )
    return tuple(operations)


class IncrementalCircuit:
    """In-place slot graph updated only from rewrite id and ordered bindings."""

    def __init__(self, initial_graph: dict):
        self.nodes = {
            int(slot): int(gate_type)
            for slot, gate_type, _ in initial_graph["nodes"]
        }
        self.edges = {tuple(map(int, edge)) for edge in initial_graph["edges"]}

    def apply(
        self,
        source: tuple[PatternOperation, ...],
        destination: tuple[PatternOperation, ...],
        source_slots: tuple[int, ...],
        destination_slots: tuple[int, ...],
    ) -> None:
        if len(source) != len(source_slots):
            raise ValueError("source pattern/binding length mismatch")
        if len(destination) != len(destination_slots):
            raise ValueError("destination pattern/binding length mismatch")
        removed = set(source_slots)
        if not removed.issubset(self.nodes):
            raise ValueError("source binding contains a non-live node")

        source_wires: dict[int, list[tuple[int, int]]] = {}
        destination_wires: dict[int, list[tuple[int, int]]] = {}
        for operation_index, operation in enumerate(source):
            for port, qubit in enumerate(operation.qubits):
                source_wires.setdefault(qubit, []).append((operation_index, port))
        for operation_index, operation in enumerate(destination):
            for port, qubit in enumerate(operation.qubits):
                destination_wires.setdefault(qubit, []).append((operation_index, port))

        boundaries: dict[int, tuple[tuple[int, int] | None, tuple[int, int] | None]] = {}
        for qubit, wire in source_wires.items():
            first_index, first_port = wire[0]
            last_index, last_port = wire[-1]
            first_slot = source_slots[first_index]
            last_slot = source_slots[last_index]
            incoming = [
                edge
                for edge in self.edges
                if edge[1] == first_slot and edge[3] == first_port and edge[0] not in removed
            ]
            outgoing = [
                edge
                for edge in self.edges
                if edge[0] == last_slot and edge[2] == last_port and edge[1] not in removed
            ]
            if len(incoming) > 1 or len(outgoing) > 1:
                raise ValueError("a circuit wire has multiple boundary edges")
            predecessor = (incoming[0][0], incoming[0][2]) if incoming else None
            successor = (outgoing[0][1], outgoing[0][3]) if outgoing else None
            boundaries[qubit] = (predecessor, successor)

        self.edges = {
            edge for edge in self.edges if edge[0] not in removed and edge[1] not in removed
        }
        for slot in source_slots:
            del self.nodes[slot]
        for slot, operation in zip(destination_slots, destination):
            if slot in self.nodes:
                raise ValueError("destination binding reuses a live slot")
            self.nodes[slot] = operation.gate_type

        all_wires = set(source_wires) | set(destination_wires)
        for qubit in all_wires:
            destination_wire = destination_wires.get(qubit, [])
            predecessor, successor = boundaries.get(qubit, (None, None))
            if destination_wire:
                for (left_index, left_port), (right_index, right_port) in zip(
                    destination_wire, destination_wire[1:]
                ):
                    self.edges.add(
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
                    self.edges.add(
                        (
                            predecessor[0],
                            destination_slots[first_index],
                            predecessor[1],
                            first_port,
                        )
                    )
                if successor is not None:
                    self.edges.add(
                        (
                            destination_slots[last_index],
                            successor[0],
                            last_port,
                            successor[1],
                        )
                    )
            elif predecessor is not None and successor is not None:
                self.edges.add(
                    (predecessor[0], successor[0], predecessor[1], successor[1])
                )

    def apply_delta(self, delta: dict) -> None:
        """Apply an exact saved graph delta, including Quartz normalization."""
        removed_slots = {int(slot) for slot in delta["removed_slots"]}
        if not removed_slots.issubset(self.nodes):
            raise ValueError("graph delta removes a non-live slot")
        removed_edges = {tuple(map(int, edge)) for edge in delta["removed_edges"]}
        added_edges = {tuple(map(int, edge)) for edge in delta["added_edges"]}
        if not removed_edges.issubset(self.edges):
            raise ValueError("graph delta removes a non-live edge")

        self.edges.difference_update(removed_edges)
        for slot in removed_slots:
            del self.nodes[slot]
        for raw_slot, raw_gate_type, _ in delta["added_nodes"]:
            slot = int(raw_slot)
            if slot in self.nodes:
                raise ValueError("graph delta reuses a live destination slot")
            self.nodes[slot] = int(raw_gate_type)
        self.edges.update(added_edges)

        live = set(self.nodes)
        if any(src not in live or dst not in live for src, dst, _, _ in self.edges):
            raise ValueError("graph delta leaves an edge incident to a dead slot")

    def snapshot_without_guids(self) -> tuple[dict[int, int], set[tuple[int, ...]]]:
        return dict(self.nodes), set(self.edges)


def structural_binding(
    circuit: IncrementalCircuit,
    source: tuple[PatternOperation, ...],
    anchor_slot: int,
) -> tuple[int, ...] | None:
    """Recover the ordered binding by following source-pattern port edges."""
    if not source or circuit.nodes.get(anchor_slot) != source[0].gate_type:
        return None
    pattern_edges = []
    wires: dict[int, list[tuple[int, int]]] = {}
    for operation_index, operation in enumerate(source):
        for port, qubit in enumerate(operation.qubits):
            wires.setdefault(qubit, []).append((operation_index, port))
    for wire in wires.values():
        for (src_index, src_port), (dst_index, dst_port) in zip(wire, wire[1:]):
            pattern_edges.append((src_index, dst_index, src_port, dst_port))

    by_output = {(edge[0], edge[2]): (edge[1], edge[3]) for edge in circuit.edges}
    by_input = {(edge[1], edge[3]): (edge[0], edge[2]) for edge in circuit.edges}
    mapping = {0: anchor_slot}
    changed = True
    while changed:
        changed = False
        for src_index, dst_index, src_port, dst_port in pattern_edges:
            if src_index in mapping and dst_index not in mapping:
                candidate = by_output.get((mapping[src_index], src_port))
                if candidate is None or candidate[1] != dst_port:
                    return None
                mapping[dst_index] = candidate[0]
                changed = True
            elif dst_index in mapping and src_index not in mapping:
                candidate = by_input.get((mapping[dst_index], dst_port))
                if candidate is None or candidate[1] != src_port:
                    return None
                mapping[src_index] = candidate[0]
                changed = True
            elif src_index in mapping and dst_index in mapping:
                if by_output.get((mapping[src_index], src_port)) != (
                    mapping[dst_index],
                    dst_port,
                ):
                    return None
    if len(mapping) != len(source) or len(set(mapping.values())) != len(source):
        return None
    for operation_index, operation in enumerate(source):
        if circuit.nodes.get(mapping[operation_index]) != operation.gate_type:
            return None
    return tuple(mapping[index] for index in range(len(source)))
