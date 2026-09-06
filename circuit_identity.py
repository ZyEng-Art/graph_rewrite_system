from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Hashable


_QREG_RE = re.compile(r"^qreg\s+[A-Za-z_][A-Za-z0-9_]*\[(\d+)\];$")
_QUBIT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\[(\d+)\]")
_GATE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\((.*)\))?\s+(.+);$")

OperationKey = tuple[str, str, tuple[int, ...]]
CanonicalCircuitKey = tuple[int, tuple[tuple[OperationKey, ...], ...]]


def canonical_qasm_key(qasm: str) -> CanonicalCircuitKey:
    """Return a circuit key invariant to serialization of independent gates.

    Physical qubit ids, gate parameters, operand order, and each qubit's gate
    dependency chain are retained.  Quartz can emit unrelated gates in different
    textual orders after commuting rewrites; recording each physical wire's
    operation trace makes those serializations share a key without treating
    qubit permutations or distinct rotation parameters as equal.
    """
    num_qubits: int | None = None
    wire_operations: list[list[OperationKey]] | None = None
    for raw_line in qasm.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("OPENQASM", "include", "creg")):
            continue
        qreg_match = _QREG_RE.match(line)
        if qreg_match is not None:
            parsed_qubits = int(qreg_match.group(1))
            if num_qubits is not None and num_qubits != parsed_qubits:
                raise ValueError("multiple incompatible qreg declarations")
            num_qubits = parsed_qubits
            wire_operations = [[] for _ in range(num_qubits)]
            continue
        match = _GATE_RE.match(line)
        if match is None:
            raise ValueError(f"unsupported QASM instruction: {line!r}")
        gate, parameters, operands = match.groups()
        qubits = tuple(map(int, _QUBIT_RE.findall(operands)))
        if not qubits:
            raise ValueError(f"instruction has no qubit operand: {line!r}")
        if wire_operations is None:
            raise ValueError("instruction appears before the qreg declaration")
        if any(qubit < 0 or qubit >= len(wire_operations) for qubit in qubits):
            raise ValueError(f"qubit operand is outside the qreg: {line!r}")
        parameter_text = "" if parameters is None else "".join(parameters.split())
        label = (gate.lower(), parameter_text, qubits)
        for qubit in qubits:
            wire_operations[qubit].append(label)

    if num_qubits is None or wire_operations is None:
        raise ValueError("QASM is missing a qreg declaration")
    return num_qubits, tuple(tuple(row) for row in wire_operations)


def exact_graph_key(graph: Any) -> Hashable:
    native_exact_key = getattr(graph, "exact_key", None)
    if native_exact_key is not None:
        return ("quartz_wire_trace_v1", bytes(native_exact_key()))
    return canonical_qasm_key(graph.to_qasm_str())


def register_exact_graph(
    graph: Any, seen: set[Hashable]
) -> bool:
    """Register a materialized graph, returning false for an exact duplicate."""
    key = exact_graph_key(graph)
    if key in seen:
        return False
    seen.add(key)
    return True


@dataclass
class ExactGraphRegistry:
    """Collision-safe registry with a byte-identical QASM fast path.

    A new serialization is represented by its operation sequence on every
    physical qubit.  This is a complete dependency-trace representation:
    swapping independent gates leaves it unchanged, while gate parameters,
    operand roles, physical qubits, and every dependent ordering remain.  It is
    constructed in one linear pass, without a graph-wide topological sort.
    """

    raw_qasm: set[str] = field(default_factory=set)
    canonical_keys: set[Hashable] = field(default_factory=set)
    registrations: int = 0
    raw_duplicates: int = 0
    reordered_duplicates: int = 0
    canonicalized_serializations: int = 0
    native_identity_calls: int = 0
    native_duplicates: int = 0

    @classmethod
    def seeded(cls, graph: Any) -> "ExactGraphRegistry":
        registry = cls()
        registry.register(graph)
        return registry

    def register(self, graph: Any) -> bool:
        self.registrations += 1
        native_exact_key = getattr(graph, "exact_key", None)
        if native_exact_key is not None:
            self.native_identity_calls += 1
            key = ("quartz_wire_trace_v1", bytes(native_exact_key()))
            if key in self.canonical_keys:
                self.native_duplicates += 1
                return False
            self.canonical_keys.add(key)
            return True

        qasm = graph.to_qasm_str()
        if qasm in self.raw_qasm:
            self.raw_duplicates += 1
            return False
        self.raw_qasm.add(qasm)
        key = canonical_qasm_key(qasm)
        self.canonicalized_serializations += 1
        if key in self.canonical_keys:
            self.reordered_duplicates += 1
            return False
        self.canonical_keys.add(key)
        return True

    def __len__(self) -> int:
        return len(self.canonical_keys)

    def stats(self) -> dict[str, int | str]:
        return {
            "mode": "exact",
            "registrations": self.registrations,
            "unique_identities": len(self.canonical_keys),
            "raw_serializations": len(self.raw_qasm),
            "raw_duplicates": self.raw_duplicates,
            "reordered_duplicates": self.reordered_duplicates,
            "canonicalized_serializations": self.canonicalized_serializations,
            "native_identity_calls": self.native_identity_calls,
            "native_duplicates": self.native_duplicates,
        }


@dataclass
class QuartzHashRegistry:
    """Legacy coarse registry retained only for controlled A/B measurements."""

    hashes: set[int] = field(default_factory=set)
    registrations: int = 0
    duplicates: int = 0

    @classmethod
    def seeded(cls, graph: Any) -> "QuartzHashRegistry":
        registry = cls()
        registry.register(graph)
        return registry

    def register(self, graph: Any) -> bool:
        self.registrations += 1
        key = int(graph.hash())
        if key in self.hashes:
            self.duplicates += 1
            return False
        self.hashes.add(key)
        return True

    def __len__(self) -> int:
        return len(self.hashes)

    def stats(self) -> dict[str, int | str]:
        return {
            "mode": "quartz_hash_legacy_unsafe",
            "registrations": self.registrations,
            "unique_hashes": len(self.hashes),
            "duplicates": self.duplicates,
        }
