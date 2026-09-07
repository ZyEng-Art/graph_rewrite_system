from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import re
import struct
from typing import Any, Hashable

from dataset import GATE_NAME_TO_ID
from incremental_graph import PatternOperation


_QREG_RE = re.compile(r"^qreg\s+[A-Za-z_][A-Za-z0-9_]*\[(\d+)\];$")
_QUBIT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\[(\d+)\]")
_GATE_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)(?:\((.*)\))?\s+(.+);$"
)


@dataclass(frozen=True)
class OperationToken:
    gate_type: int
    parameters: bytes
    qubits: tuple[int, ...]


@dataclass(frozen=True)
class WireTraceProfile:
    """Physical-wire trace plus the persistent slot of every operation.

    Quartz's exact identity is the operation sequence on each physical wire.
    Keeping that sequence with slots lets a candidate rewrite splice only the
    affected wire intervals instead of cloning and rewriting the whole graph.
    """

    wire_rows: tuple[tuple[tuple[int, OperationToken], ...], ...]
    slot_tokens: dict[int, OperationToken]
    slot_wire_positions: dict[int, tuple[tuple[int, int], ...]]
    exact_prefix_hashes: tuple[tuple[int, ...], ...]
    topology_prefix_hashes: tuple[tuple[int, ...], ...]


_HASH_MASK = (1 << 128) - 1
_HASH_BASE = 0x1000000000000000000013B
_PARAMETERIZED_GATE_TYPES = frozenset(
    GATE_NAME_TO_ID[name]
    for name in (
        "rx",
        "ry",
        "rz",
        "p",
        "pdg",
        "u1",
        "u2",
        "u3",
        "rxx1",
        "rxx3",
    )
)


@lru_cache(maxsize=16384)
def _operation_payload(token: OperationToken, *, include_parameters: bool) -> bytes:
    payload = bytearray()
    payload.extend(struct.pack("<II", token.gate_type, len(token.qubits)))
    for qubit in token.qubits:
        payload.extend(struct.pack("<I", qubit))
    if include_parameters:
        payload.extend(struct.pack("<I", len(token.parameters)))
        payload.extend(token.parameters)
    return bytes(payload)


@lru_cache(maxsize=32768)
def _token_hash(token: OperationToken, *, include_parameters: bool) -> int:
    payload = _operation_payload(token, include_parameters=include_parameters)
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "little")


def _prefix_hashes(
    tokens: tuple[OperationToken, ...] | list[OperationToken],
    *,
    include_parameters: bool,
) -> tuple[int, ...]:
    result = [0]
    value = 0
    for token in tokens:
        value = (
            value * _HASH_BASE
            + _token_hash(token, include_parameters=include_parameters)
        ) & _HASH_MASK
        result.append(value)
    return tuple(result)


@lru_cache(maxsize=4096)
def _hash_power(length: int) -> int:
    return pow(_HASH_BASE, length, 1 << 128)


def _slice_hash(prefix: tuple[int, ...], begin: int, end: int) -> int:
    return (
        prefix[end] - prefix[begin] * _hash_power(end - begin)
    ) & _HASH_MASK


def _append_hash(left: int, right: int, right_length: int) -> int:
    return (left * _hash_power(right_length) + right) & _HASH_MASK


def _circuit_fingerprint(
    wire_rows: tuple[tuple[tuple[int, OperationToken], ...], ...],
    prefix_hashes: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (len(row), prefix[-1]) for row, prefix in zip(wire_rows, prefix_hashes)
    )


def _parse_qasm_operations(qasm: str) -> tuple[int, tuple[OperationToken, ...]]:
    num_qubits: int | None = None
    operations = []
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
            continue
        match = _GATE_RE.match(line)
        if match is None:
            raise ValueError(f"unsupported QASM instruction: {line!r}")
        gate_name, parameters, operands = match.groups()
        gate_name = gate_name.lower()
        if gate_name not in GATE_NAME_TO_ID:
            raise ValueError(f"unknown QASM gate: {gate_name}")
        qubits = tuple(map(int, _QUBIT_RE.findall(operands)))
        if not qubits:
            raise ValueError(f"instruction has no qubit operand: {line!r}")
        operations.append(
            OperationToken(
                gate_type=GATE_NAME_TO_ID[gate_name],
                parameters=(
                    b""
                    if parameters is None
                    else "".join(parameters.split()).encode("utf-8")
                ),
                qubits=qubits,
            )
        )
    if num_qubits is None:
        raise ValueError("QASM is missing a qreg declaration")
    return num_qubits, tuple(operations)


def build_wire_trace_profile(
    graph: Any, guid_to_slot: dict[int, int]
) -> WireTraceProfile | None:
    """Build a slot-addressable wire trace, or return none if APIs disagree.

    ``PyGraph.nodes`` and ``to_qasm_str`` are produced by Quartz's same
    topological traversal.  The explicit checks below turn any future Quartz
    ordering/API change into a disabled fingerprint rather than unsafe pruning.
    """

    native_profile = getattr(graph, "wire_trace_profile", None)
    try:
        nodes = list(graph.nodes)
        if native_profile is None:
            num_qubits, operations = _parse_qasm_operations(graph.to_qasm_str())
            operation_rows = tuple(
                (int(node.guid), token) for node, token in zip(nodes, operations)
            )
        else:
            num_qubits, native_rows = native_profile()
            num_qubits = int(num_qubits)
            operation_rows = tuple(
                (
                    int(guid),
                    OperationToken(
                        int(gate_type),
                        b"".join(struct.pack("<d", float(value)) for value in parameters),
                        tuple(map(int, qubits)),
                    ),
                )
                for guid, gate_type, qubits, parameters in native_rows
            )
            operations = tuple(token for _, token in operation_rows)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if len(nodes) != len(operations) or len(operation_rows) != len(operations):
        return None

    wire_rows: list[list[tuple[int, OperationToken]]] = [
        [] for _ in range(num_qubits)
    ]
    slot_tokens: dict[int, OperationToken] = {}
    for node, (record_guid, token) in zip(nodes, operation_rows):
        try:
            guid = int(node.guid)
            slot = int(guid_to_slot[guid])
            gate_type = int(node.gate_tp)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if (
            guid != record_guid
            or gate_type != token.gate_type
            or slot in slot_tokens
        ):
            return None
        if any(qubit < 0 or qubit >= num_qubits for qubit in token.qubits):
            return None
        slot_tokens[slot] = token
        for qubit in token.qubits:
            wire_rows[qubit].append((slot, token))

    frozen_rows = tuple(tuple(row) for row in wire_rows)
    positions: dict[int, list[tuple[int, int]]] = {
        slot: [] for slot in slot_tokens
    }
    for qubit, row in enumerate(frozen_rows):
        for index, (slot, _) in enumerate(row):
            positions[slot].append((qubit, index))
    exact_prefixes = tuple(
        _prefix_hashes(
            tuple(token for _, token in row), include_parameters=True
        )
        for row in frozen_rows
    )
    topology_prefixes = tuple(
        _prefix_hashes(
            tuple(token for _, token in row), include_parameters=False
        )
        for row in frozen_rows
    )
    return WireTraceProfile(
        wire_rows=frozen_rows,
        slot_tokens=slot_tokens,
        slot_wire_positions={slot: tuple(rows) for slot, rows in positions.items()},
        exact_prefix_hashes=exact_prefixes,
        topology_prefix_hashes=topology_prefixes,
    )


def profile_fingerprint(
    profile: WireTraceProfile, *, kind: str
) -> tuple[tuple[int, int], ...]:
    if kind in ("conservative", "parameter_transfer", "xfer_guarded"):
        return _circuit_fingerprint(
            profile.wire_rows, profile.exact_prefix_hashes
        )
    if kind == "topology":
        return _circuit_fingerprint(
            profile.wire_rows, profile.topology_prefix_hashes
        )
    raise ValueError(f"unknown fingerprint kind: {kind}")


def _destination_tokens(
    profile: WireTraceProfile,
    source: tuple[PatternOperation, ...],
    destination: tuple[PatternOperation, ...],
    source_slots: tuple[int, ...],
    local_to_physical: dict[int, int],
    *,
    xfer_id: int,
    kind: str,
) -> tuple[OperationToken, ...] | None:
    if kind == "topology":
        return tuple(
            OperationToken(
                operation.gate_type,
                b"",
                tuple(local_to_physical[qubit] for qubit in operation.qubits),
            )
            for operation in destination
        )

    parameter_rows: dict[int, list[bytes]] = {}
    destination_counts: dict[int, int] = {}
    if kind in ("parameter_transfer", "xfer_guarded"):
        for operation, slot in zip(source, source_slots):
            token = profile.slot_tokens[slot]
            if token.parameters:
                parameter_rows.setdefault(operation.gate_type, []).append(
                    token.parameters
                )
        for operation in destination:
            destination_counts[operation.gate_type] = (
                destination_counts.get(operation.gate_type, 0) + 1
            )
    parameter_offsets: dict[int, int] = {}

    # Compact ECC strings omit parameter expressions.  The conservative mode
    # never guesses them: it records an xfer-specific symbolic value derived
    # from every concrete source token.  ``parameter_transfer`` is a more
    # aggressive audited mode that carries a one-to-one same-gate parameter;
    # callers must retain multiple representatives when using it to filter.
    effect_tag: bytes | None = None

    result = []
    for operation_index, operation in enumerate(destination):
        physical_qubits = tuple(
            local_to_physical[qubit] for qubit in operation.qubits
        )
        source_parameters = parameter_rows.get(operation.gate_type, ())
        offset = parameter_offsets.get(operation.gate_type, 0)
        if (
            kind in ("parameter_transfer", "xfer_guarded")
            and source_parameters
            and len(source_parameters)
            == destination_counts[operation.gate_type]
        ):
            parameters = source_parameters[offset]
            parameter_offsets[operation.gate_type] = offset + 1
        elif operation.gate_type in _PARAMETERIZED_GATE_TYPES:
            if effect_tag is None:
                effect_digest = hashlib.blake2b(digest_size=16)
                effect_digest.update(struct.pack("<I", xfer_id))
                for slot in source_slots:
                    effect_digest.update(
                        _operation_payload(
                            profile.slot_tokens[slot], include_parameters=True
                        )
                    )
                effect_tag = b"effect:" + effect_digest.digest()
            parameters = effect_tag + struct.pack("<I", operation_index)
        else:
            parameters = b""
        result.append(
            OperationToken(operation.gate_type, parameters, physical_qubits)
        )
    return tuple(result)


def successor_fingerprint(
    profile: WireTraceProfile,
    source: tuple[PatternOperation, ...],
    destination: tuple[PatternOperation, ...],
    source_slots: tuple[int, ...],
    *,
    xfer_id: int,
    kind: str = "conservative",
) -> Hashable | None:
    """Predict a successor identity by splicing only affected wire traces.

    The conservative form retains exact existing parameters and assigns an
    xfer/source-derived symbolic value to parameterized destination gates.  It
    never guesses Quartz's hidden parameter expression.  The topology form is
    intentionally parameter-blind and is intended only for shadow upper-bound
    audits.
    """

    if kind not in (
        "conservative",
        "parameter_transfer",
        "xfer_guarded",
        "topology",
    ):
        raise ValueError(f"unknown fingerprint kind: {kind}")
    if len(source) != len(source_slots) or len(set(source_slots)) != len(source_slots):
        return None
    if any(slot not in profile.slot_tokens for slot in source_slots):
        return None

    local_to_physical: dict[int, int] = {}
    physical_to_local: dict[int, int] = {}
    for operation, slot in zip(source, source_slots):
        token = profile.slot_tokens[slot]
        if operation.gate_type != token.gate_type:
            return None
        if len(operation.qubits) != len(token.qubits):
            return None
        for local_qubit, physical_qubit in zip(operation.qubits, token.qubits):
            previous_physical = local_to_physical.setdefault(
                local_qubit, physical_qubit
            )
            previous_local = physical_to_local.setdefault(
                physical_qubit, local_qubit
            )
            if previous_physical != physical_qubit or previous_local != local_qubit:
                return None
    if any(
        local_qubit not in local_to_physical
        for operation in destination
        for local_qubit in operation.qubits
    ):
        return None

    destination_tokens = _destination_tokens(
        profile,
        source,
        destination,
        source_slots,
        local_to_physical,
        xfer_id=xfer_id,
        kind=kind,
    )
    if destination_tokens is None:
        return None

    destination_by_wire: dict[int, list[OperationToken]] = {}
    for token in destination_tokens:
        for physical_qubit in token.qubits:
            destination_by_wire.setdefault(physical_qubit, []).append(token)

    affected_wires = set(local_to_physical.values()) | set(destination_by_wire)
    prefix_hashes = (
        profile.exact_prefix_hashes
        if kind in ("conservative", "parameter_transfer", "xfer_guarded")
        else profile.topology_prefix_hashes
    )
    child_fingerprint = list(
        _circuit_fingerprint(profile.wire_rows, prefix_hashes)
    )
    positions_by_wire: dict[int, list[int]] = {
        physical_qubit: [] for physical_qubit in affected_wires
    }
    for slot in source_slots:
        for physical_qubit, position in profile.slot_wire_positions[slot]:
            positions_by_wire[physical_qubit].append(position)
    for physical_qubit in affected_wires:
        row = profile.wire_rows[physical_qubit]
        removed_indices = sorted(positions_by_wire[physical_qubit])
        if not removed_indices:
            return None
        # A valid Quartz source match occupies a contiguous interval on every
        # affected physical wire.  Refuse to predict if that invariant is not
        # visible instead of producing an unsafe splice.
        if removed_indices != list(
            range(removed_indices[0], removed_indices[-1] + 1)
        ):
            return None
        first = removed_indices[0]
        after = removed_indices[-1] + 1
        destination_wire = destination_by_wire.get(physical_qubit, ())
        destination_prefix = _prefix_hashes(
            destination_wire,
            include_parameters=kind
            in ("conservative", "parameter_transfer", "xfer_guarded"),
        )
        left_hash = prefix_hashes[physical_qubit][first]
        destination_hash = destination_prefix[-1]
        suffix_hash = _slice_hash(
            prefix_hashes[physical_qubit], after, len(row)
        )
        combined = _append_hash(
            left_hash, destination_hash, len(destination_wire)
        )
        combined = _append_hash(combined, suffix_hash, len(row) - after)
        child_fingerprint[physical_qubit] = (
            first + len(destination_wire) + len(row) - after,
            combined,
        )
    result = tuple(child_fingerprint)
    if kind == "xfer_guarded" and any(
        operation.gate_type in _PARAMETERIZED_GATE_TYPES
        for operation in destination
    ):
        return (xfer_id, result)
    return result


@dataclass
class FingerprintAudit:
    mode: str
    kind: str
    representatives: int
    identities: dict[Hashable, set[Hashable]]
    fingerprints: set[Hashable]
    observations: dict[Hashable, int]
    candidates: int = 0
    unavailable: int = 0
    bypassed_low_reuse: int = 0
    hits: int = 0
    skipped: int = 0
    valid_hits: int = 0
    invalid_hits: int = 0
    exact_duplicate_hits: int = 0
    collision_hits: int = 0

    @classmethod
    def create(
        cls, *, mode: str, kind: str, representatives: int = 1
    ) -> "FingerprintAudit":
        if mode not in ("off", "shadow", "filter"):
            raise ValueError(f"unknown fingerprint mode: {mode}")
        if kind not in (
            "conservative",
            "parameter_transfer",
            "xfer_guarded",
            "topology",
        ):
            raise ValueError(f"unknown fingerprint kind: {kind}")
        if representatives < 1:
            raise ValueError("fingerprint representatives must be positive")
        return cls(
            mode=mode,
            kind=kind,
            representatives=representatives,
            identities={},
            fingerprints=set(),
            observations={},
        )

    def should_skip(self, fingerprint: Hashable | None) -> bool:
        if fingerprint is None:
            self.unavailable += 1
            return False
        self.candidates += 1
        if fingerprint not in self.fingerprints:
            return False
        self.hits += 1
        if (
            self.mode == "filter"
            and self.observations.get(fingerprint, 0) >= self.representatives
        ):
            self.skipped += 1
            return True
        return False

    def observe_bypassed(self) -> None:
        self.bypassed_low_reuse += 1

    def observe_invalid(self, fingerprint: Hashable | None) -> None:
        if (
            self.mode == "shadow"
            and fingerprint is not None
            and fingerprint in self.fingerprints
        ):
            self.invalid_hits += 1

    def observe_valid(
        self, fingerprint: Hashable | None, exact_identity: Hashable | None = None
    ) -> None:
        if fingerprint is None:
            return
        was_seen = fingerprint in self.fingerprints
        if self.mode == "shadow":
            if exact_identity is None:
                raise ValueError("shadow auditing requires an exact identity")
            known = self.identities.setdefault(fingerprint, set())
            if was_seen:
                self.valid_hits += 1
                if exact_identity in known:
                    self.exact_duplicate_hits += 1
                else:
                    self.collision_hits += 1
            known.add(exact_identity)
        self.fingerprints.add(fingerprint)
        self.observations[fingerprint] = self.observations.get(fingerprint, 0) + 1

    def stats(self) -> dict[str, int | float | str]:
        audited_hits = self.exact_duplicate_hits + self.collision_hits
        return {
            "mode": self.mode,
            "kind": self.kind,
            "representatives": self.representatives,
            "candidates": self.candidates,
            "unavailable": self.unavailable,
            "bypassed_low_reuse": self.bypassed_low_reuse,
            "fingerprints": len(self.fingerprints),
            "hits": self.hits,
            "skipped_before_apply": self.skipped,
            "shadow_valid_hits": self.valid_hits,
            "shadow_invalid_hits": self.invalid_hits,
            "shadow_exact_duplicate_hits": self.exact_duplicate_hits,
            "shadow_collision_hits": self.collision_hits,
            "shadow_precision": (
                self.exact_duplicate_hits / audited_hits if audited_hits else 1.0
            ),
            "shadow_collision_fingerprints": sum(
                len(identities) > 1 for identities in self.identities.values()
            ),
            "shadow_max_exact_identities_per_fingerprint": max(
                map(len, self.identities.values()), default=0
            ),
        }
