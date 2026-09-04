from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import torch
from torch.utils.data import Dataset


GATE_NAME_TO_ID = {
    "h": 0,
    "x": 1,
    "y": 2,
    "rx": 3,
    "ry": 4,
    "rz": 5,
    "cx": 6,
    "ccx": 7,
    "add": 8,
    "neg": 9,
    "z": 10,
    "s": 11,
    "sdg": 12,
    "t": 13,
    "tdg": 14,
    "ch": 15,
    "swap": 16,
    "p": 17,
    "pdg": 18,
    "u1": 19,
    "u2": 20,
    "u3": 21,
    "ccz": 22,
    "cz": 23,
    "rx1": 24,
    "rx3": 25,
    "input_qubit": 26,
    "input_param": 27,
    "ry1": 28,
    "ry3": 29,
    "rxx1": 30,
    "rxx3": 31,
    "sx": 32,
}


def pattern_gate_types(pattern: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    result = []
    for instruction in pattern.split(";"):
        instruction = instruction.strip()
        if not instruction:
            continue
        gate_name = instruction.split()[0]
        if gate_name not in GATE_NAME_TO_ID:
            raise ValueError(f"unknown Quartz gate in pattern: {gate_name}")
        result.append(GATE_NAME_TO_ID[gate_name])
    if not result and not allow_empty:
        raise ValueError("empty Quartz pattern")
    return tuple(result)


@dataclass(frozen=True)
class RuleMetadata:
    source_patterns: tuple[str, ...]
    source_gate_types: tuple[tuple[int, ...], ...]
    destination_gate_types: tuple[tuple[int, ...], ...]
    xfer_to_source: tuple[int, ...]
    xfer_sources: tuple[str, ...]
    xfer_destinations: tuple[str, ...]
    num_gate_types: int = 33

    @classmethod
    def from_payload(cls, payload: dict) -> "RuleMetadata":
        return cls(
            source_patterns=tuple(payload["source_patterns"]),
            source_gate_types=tuple(
                pattern_gate_types(pattern) for pattern in payload["source_patterns"]
            ),
            destination_gate_types=tuple(
                pattern_gate_types(pattern, allow_empty=True)
                for pattern in payload["xfer_destinations"]
            ),
            xfer_to_source=tuple(map(int, payload["xfer_to_source"])),
            xfer_sources=tuple(payload["xfer_sources"]),
            xfer_destinations=tuple(payload["xfer_destinations"]),
        )


def validate_trajectory(trajectory: dict, rules: RuleMetadata) -> None:
    live = {
        int(slot): int(gate_type)
        for slot, gate_type, _ in trajectory["initial_graph"]["nodes"]
    }
    for step in trajectory["steps"]:
        action = step["action"]
        source_slots = tuple(map(int, action["binding_slots"]))
        destination_slots = tuple(map(int, action["dst_slots"]))
        destination_types = rules.destination_gate_types[int(action["xfer_id"])]
        if len(destination_slots) != len(destination_types):
            raise ValueError("destination slot/type length mismatch")
        if sorted(source_slots) != list(map(int, step["delta"]["removed_slots"])):
            raise ValueError("source binding does not equal removed slots")
        for slot in source_slots:
            if slot not in live:
                raise ValueError("action consumes a non-live slot")
            del live[slot]
        for slot, gate_type in zip(destination_slots, destination_types):
            if slot in live:
                raise ValueError("action reuses a live destination slot")
            live[slot] = gate_type
        expected_added = {
            int(slot): int(gate_type)
            for slot, gate_type, _ in step["delta"]["added_nodes"]
        }
        if {slot: live[slot] for slot in destination_slots} != expected_added:
            raise ValueError("rule-derived destination types differ from Quartz")


class PrefixDataset(Dataset):
    """Each item is s0 plus an action prefix; no materialized s_t is returned."""

    def __init__(
        self,
        trajectories: Iterable[dict],
        rules: RuleMetadata,
        *,
        include_terminal: bool = False,
        terminal_only_repeat: int = 1,
    ):
        self.trajectories = list(trajectories)
        self.rules = rules
        if terminal_only_repeat < 1:
            raise ValueError("terminal_only_repeat must be positive")
        self.indices: list[tuple[int, int]] = []
        for trajectory_id, trajectory in enumerate(self.trajectories):
            validate_trajectory(trajectory, rules)
            if not trajectory.get("terminal_only_supervision", False):
                self.indices.extend(
                    (trajectory_id, prefix_length)
                    for prefix_length in range(len(trajectory["steps"]))
                )
            if include_terminal and "terminal_matches" in trajectory:
                repeat = (
                    terminal_only_repeat
                    if trajectory.get("terminal_only_supervision", False)
                    else 1
                )
                self.indices.extend(
                    [(trajectory_id, len(trajectory["steps"]))] * repeat
                )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        trajectory_id, prefix_length = self.indices[index]
        trajectory = self.trajectories[trajectory_id]
        terminal = prefix_length == len(trajectory["steps"])
        target_step = trajectory["steps"][-1] if terminal else trajectory["steps"][prefix_length]
        previous_step = (
            trajectory["steps"][prefix_length - 1] if prefix_length > 0 else None
        )
        return {
            "initial_graph": trajectory["initial_graph"],
            "actions": [
                step["action"] for step in trajectory["steps"][:prefix_length]
            ],
            "matches": (
                trajectory["terminal_matches"] if terminal else target_step["matches"]
            ),
            "local_streak": int(target_step["local_streak"]),
            "trajectory_id": int(trajectory["trajectory_id"]),
            "prefix_length": prefix_length,
            "previous_action": (
                previous_step["action"] if previous_step is not None else None
            ),
            "previous_delta": (
                previous_step["delta"] if previous_step is not None else None
            ),
            "previous_local_streak": (
                int(previous_step["local_streak"])
                if previous_step is not None
                else None
            ),
        }


def _max_slot(sample: dict) -> int:
    slots = [int(row[0]) for row in sample["initial_graph"]["nodes"]]
    for action in sample["actions"]:
        slots.extend(map(int, action["binding_slots"]))
        slots.extend(map(int, action["dst_slots"]))
    return max(slots, default=-1)


def _touch_age_bucket(age: int) -> int:
    if age <= 3:
        return age
    if age <= 7:
        return 4
    if age <= 15:
        return 5
    return 6


def _local_streak_bucket(has_actions: bool, streak: int) -> int:
    if not has_actions:
        return 0
    if streak == 0:
        return 1
    if streak == 1:
        return 2
    if streak <= 3:
        return 3
    if streak <= 7:
        return 4
    return 5


def replay_with_locality(sample: dict, rules: RuleMetadata):
    """Derive current graph and rewrite-locality metadata from s0 + actions only."""
    from incremental_graph import IncrementalCircuit, parse_pattern

    circuit = IncrementalCircuit(sample["initial_graph"])
    last_touched: dict[int, int] = {}
    last_core: set[int] = set()
    previous_preferred: set[int] = set()
    local_streak = 0
    for action_index, action in enumerate(sample["actions"]):
        before_edges = set(circuit.edges)
        xfer_id = int(action["xfer_id"])
        source_slots = tuple(map(int, action["binding_slots"]))
        source_set = set(source_slots)
        surviving_predecessors = {
            src for src, dst, _, _ in before_edges if dst in source_set
        }
        destination_slots = tuple(map(int, action["dst_slots"]))
        circuit.apply(
            parse_pattern(rules.xfer_sources[xfer_id]),
            parse_pattern(rules.xfer_destinations[xfer_id]),
            source_slots,
            destination_slots,
        )
        live_slots = set(circuit.nodes)
        changed_edges = before_edges.symmetric_difference(circuit.edges)
        last_core = {slot for slot in destination_slots if slot in live_slots}
        for src, dst, _, _ in changed_edges:
            if src in live_slots:
                last_core.add(src)
            if dst in live_slots:
                last_core.add(dst)
        for slot in last_core:
            last_touched[slot] = action_index

        anchor = int(action.get("anchor_slot", action["binding_slots"][0]))
        continued_local = anchor in previous_preferred
        local_streak = local_streak + 1 if continued_local else 0
        previous_preferred = {
            slot for slot in destination_slots if slot in live_slots
        }
        previous_preferred.update(surviving_predecessors & live_slots)

    distances: dict[int, int] = {}
    if sample["actions"]:
        adjacency: dict[int, set[int]] = {
            int(slot): set() for slot in circuit.nodes
        }
        for src, dst, _, _ in circuit.edges:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
        distances = {slot: 0 for slot in last_core}
        queue = deque(last_core)
        while queue:
            slot = queue.popleft()
            for neighbor in adjacency[slot]:
                if neighbor not in distances:
                    distances[neighbor] = distances[slot] + 1
                    queue.append(neighbor)

    final_step = len(sample["actions"]) - 1
    rewrite_distance = {
        slot: min(distances.get(slot, 5), 4) if slot in distances else 5
        for slot in circuit.nodes
    }
    touch_age = {
        slot: (
            _touch_age_bucket(final_step - last_touched[slot])
            if slot in last_touched
            else 7
        )
        for slot in circuit.nodes
    }
    return (
        circuit,
        rewrite_distance,
        touch_age,
        _local_streak_bucket(bool(sample["actions"]), local_streak),
    )


def compact_live_slots(current_types: torch.Tensor) -> torch.Tensor:
    """Return padded slot ids so attention can skip historical dead slots."""
    rows = [row.ge(0).nonzero(as_tuple=False).flatten() for row in current_types]
    max_live = max((int(row.numel()) for row in rows), default=0)
    result = torch.full(
        (current_types.shape[0], max_live), -1, dtype=torch.long
    )
    for batch_index, row in enumerate(rows):
        result[batch_index, : row.numel()] = row
    return result


def collate_current_graphs(samples: list[dict], rules: RuleMetadata) -> dict:
    """Minimal collation for the exact discrete-state model."""
    batch_size = len(samples)
    max_slots = max(_max_slot(sample) + 1 for sample in samples)
    current_types = torch.full((batch_size, max_slots), -1, dtype=torch.long)
    current_rewrite_distance = torch.full(
        (batch_size, max_slots), 5, dtype=torch.long
    )
    current_touch_age = torch.full((batch_size, max_slots), 7, dtype=torch.long)
    current_local_streak = torch.zeros(batch_size, dtype=torch.long)
    edge_batch: list[int] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_relation: list[int] = []
    positives = []
    positive_near = []
    for batch_index, sample in enumerate(samples):
        circuit, rewrite_distance, touch_age, streak_bucket = replay_with_locality(
            sample, rules
        )
        for slot, gate_type in circuit.nodes.items():
            current_types[batch_index, slot] = gate_type
            current_rewrite_distance[batch_index, slot] = rewrite_distance[slot]
            current_touch_age[batch_index, slot] = touch_age[slot]
        current_local_streak[batch_index] = streak_bucket
        for src, dst, src_port, dst_port in circuit.edges:
            edge_batch.append(batch_index)
            edge_src.append(src)
            edge_dst.append(dst)
            edge_relation.append(src_port * 4 + dst_port)
        rows = [
            (int(match["source_id"]), tuple(map(int, match["binding_slots"])))
            for match in sample["matches"]
        ]
        positives.append(rows)
        positive_near.append(
            [
                min((rewrite_distance.get(slot, 5) for slot in binding), default=5)
                <= 2
                for _, binding in rows
            ]
        )
    return {
        "current_types": current_types,
        "current_live_slots": compact_live_slots(current_types),
        "current_edge_batch": torch.tensor(edge_batch, dtype=torch.long),
        "current_edge_src": torch.tensor(edge_src, dtype=torch.long),
        "current_edge_dst": torch.tensor(edge_dst, dtype=torch.long),
        "current_edge_relation": torch.tensor(edge_relation, dtype=torch.long),
        "current_rewrite_distance": current_rewrite_distance,
        "current_touch_age": current_touch_age,
        "current_local_streak": current_local_streak,
        "has_previous_rewrite": torch.tensor(
            [bool(sample["actions"]) for sample in samples], dtype=torch.bool
        ),
        "positives": positives,
        "positive_near": positive_near,
        "local_streak": torch.tensor(
            [sample["local_streak"] for sample in samples], dtype=torch.long
        ),
        "prefix_length": torch.tensor(
            [sample["prefix_length"] for sample in samples], dtype=torch.long
        ),
        "previous_action": [sample["previous_action"] for sample in samples],
        "previous_delta": [sample["previous_delta"] for sample in samples],
        "previous_local_streak": [
            sample["previous_local_streak"] for sample in samples
        ],
    }


def collate_prefixes(samples: list[dict], rules: RuleMetadata) -> dict:
    from incremental_graph import IncrementalCircuit, parse_pattern

    batch_size = len(samples)
    max_slots = max(_max_slot(sample) + 1 for sample in samples)
    max_actions = max(len(sample["actions"]) for sample in samples)
    max_pattern = max(
        max(map(len, rules.source_gate_types)),
        max(map(len, rules.destination_gate_types)),
    )
    initial_types = torch.full((batch_size, max_slots), -1, dtype=torch.long)
    action_xfers = torch.full((batch_size, max_actions), -1, dtype=torch.long)
    action_sources = torch.full((batch_size, max_actions), -1, dtype=torch.long)
    binding_slots = torch.full(
        (batch_size, max_actions, max_pattern), -1, dtype=torch.long
    )
    destination_slots = torch.full_like(binding_slots, -1)
    destination_types = torch.full_like(binding_slots, -1)
    edge_batch: list[int] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_relation: list[int] = []
    current_edge_batch: list[int] = []
    current_edge_src: list[int] = []
    current_edge_dst: list[int] = []
    current_edge_relation: list[int] = []
    current_types = torch.full((batch_size, max_slots), -1, dtype=torch.long)
    current_rewrite_distance = torch.full(
        (batch_size, max_slots), 5, dtype=torch.long
    )
    current_touch_age = torch.full((batch_size, max_slots), 7, dtype=torch.long)
    current_local_streak = torch.zeros(batch_size, dtype=torch.long)
    positives: list[list[tuple[int, tuple[int, ...]]]] = []
    positive_near: list[list[bool]] = []
    incremental_circuits = []

    for batch_index, sample in enumerate(samples):
        for slot, gate_type, _ in sample["initial_graph"]["nodes"]:
            initial_types[batch_index, int(slot)] = int(gate_type)
        for src, dst, src_port, dst_port in sample["initial_graph"]["edges"]:
            edge_batch.append(batch_index)
            edge_src.append(int(src))
            edge_dst.append(int(dst))
            edge_relation.append(int(src_port) * 4 + int(dst_port))
        for action_index, action in enumerate(sample["actions"]):
            xfer_id = int(action["xfer_id"])
            action_xfers[batch_index, action_index] = xfer_id
            action_sources[batch_index, action_index] = int(action["source_id"])
            src_slots = tuple(map(int, action["binding_slots"]))
            dst_slots = tuple(map(int, action["dst_slots"]))
            dst_types = rules.destination_gate_types[xfer_id]
            binding_slots[batch_index, action_index, : len(src_slots)] = torch.tensor(
                src_slots
            )
            destination_slots[
                batch_index, action_index, : len(dst_slots)
            ] = torch.tensor(dst_slots)
            destination_types[
                batch_index, action_index, : len(dst_types)
            ] = torch.tensor(dst_types)
        circuit, rewrite_distance, touch_age, local_streak = replay_with_locality(
            sample, rules
        )
        for slot, gate_type in circuit.nodes.items():
            current_types[batch_index, slot] = gate_type
            current_rewrite_distance[batch_index, slot] = rewrite_distance.get(
                slot, 5
            )
            current_touch_age[batch_index, slot] = touch_age.get(slot, 7)
        current_local_streak[batch_index] = local_streak
        for src, dst, src_port, dst_port in circuit.edges:
            current_edge_batch.append(batch_index)
            current_edge_src.append(src)
            current_edge_dst.append(dst)
            current_edge_relation.append(src_port * 4 + dst_port)
        incremental_circuits.append(circuit)
        rows = [
            (int(match["source_id"]), tuple(map(int, match["binding_slots"])))
            for match in sample["matches"]
        ]
        positives.append(rows)
        positive_near.append(
            [
                min((rewrite_distance.get(slot, 5) for slot in binding), default=5)
                <= 2
                for _, binding in rows
            ]
        )

    return {
        "initial_types": initial_types,
        "edge_batch": torch.tensor(edge_batch, dtype=torch.long),
        "edge_src": torch.tensor(edge_src, dtype=torch.long),
        "edge_dst": torch.tensor(edge_dst, dtype=torch.long),
        "edge_relation": torch.tensor(edge_relation, dtype=torch.long),
        "current_types": current_types,
        "current_live_slots": compact_live_slots(current_types),
        "current_rewrite_distance": current_rewrite_distance,
        "current_touch_age": current_touch_age,
        "current_local_streak": current_local_streak,
        "current_edge_batch": torch.tensor(current_edge_batch, dtype=torch.long),
        "current_edge_src": torch.tensor(current_edge_src, dtype=torch.long),
        "current_edge_dst": torch.tensor(current_edge_dst, dtype=torch.long),
        "current_edge_relation": torch.tensor(
            current_edge_relation, dtype=torch.long
        ),
        "action_xfers": action_xfers,
        "action_sources": action_sources,
        "binding_slots": binding_slots,
        "destination_slots": destination_slots,
        "destination_types": destination_types,
        "positives": positives,
        "positive_near": positive_near,
        "incremental_circuits": incremental_circuits,
        "has_previous_rewrite": torch.tensor(
            [bool(sample["actions"]) for sample in samples], dtype=torch.bool
        ),
        "local_streak": torch.tensor(
            [sample["local_streak"] for sample in samples], dtype=torch.long
        ),
        "prefix_length": torch.tensor(
            [sample["prefix_length"] for sample in samples], dtype=torch.long
        ),
        # Evaluation-only audit metadata. The paged model never consumes these
        # fields; they are used to compare action-derived locality with the
        # recorded previous Quartz transition.
        "previous_action": [sample["previous_action"] for sample in samples],
        "previous_delta": [sample["previous_delta"] for sample in samples],
        "previous_local_streak": [
            sample["previous_local_streak"] for sample in samples
        ],
    }


def load_datasets(
    path: Path,
    *,
    include_terminal: bool = False,
    include_train_terminal: bool = False,
    train_terminal_repeat: int = 1,
) -> tuple[dict, RuleMetadata, PrefixDataset, PrefixDataset]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "quartz-full-binding-v2":
        raise ValueError("expected quartz-full-binding-v2 dataset")
    rules = RuleMetadata.from_payload(payload)
    return (
        payload,
        rules,
        PrefixDataset(
            payload["train_trajectories"],
            rules,
            include_terminal=include_terminal or include_train_terminal,
            terminal_only_repeat=train_terminal_repeat,
        ),
        PrefixDataset(
            payload["test_trajectories"], rules, include_terminal=include_terminal
        ),
    )


class EpochRandomSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: Dataset, seed: int):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)
        self.epoch += 1
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
