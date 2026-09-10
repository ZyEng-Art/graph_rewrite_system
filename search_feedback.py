from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Hashable


def identity_order_key(identity: Hashable) -> str:
    """Return a process- and seed-independent ordering key for an exact graph."""
    return hashlib.blake2b(
        repr(identity).encode("utf-8"), digest_size=16
    ).hexdigest()


@dataclass
class SearchNodeStats:
    node_id: int
    identity_order: str
    gate_count: int
    depth: int
    origin_parent_id: int | None = None
    origin_continuation_score: float | None = None
    parent_ids: set[int] = field(default_factory=set)
    attempted_actions: int = 0
    valid_actions: int = 0
    unique_children: int = 0
    duplicate_children: int = 0
    invalid_actions: int = 0
    improving_children: int = 0
    observed_expansions: int = 0
    last_attempted_actions: int = 0
    last_valid_actions: int = 0
    last_unique_children: int = 0
    last_duplicate_children: int = 0
    last_invalid_actions: int = 0
    last_improving_children: int = 0
    last_best_child_gate: int | None = None
    last_expansion_step: int = -1
    best_child_gate: int | None = None
    best_descendant_gate: int | None = None
    last_improvement_step: int = -1

    @property
    def novel_yield(self) -> float:
        return self.unique_children / max(1, self.valid_actions)

    @property
    def valid_yield(self) -> float:
        return self.valid_actions / max(1, self.attempted_actions)

    @property
    def last_unique_yield(self) -> float:
        return self.last_unique_children / max(1, self.last_attempted_actions)

    @property
    def last_valid_yield(self) -> float:
        return self.last_valid_actions / max(1, self.last_attempted_actions)

    @property
    def last_improving_yield(self) -> float:
        return self.last_improving_children / max(1, self.last_attempted_actions)

    @property
    def descendant_gain(self) -> int:
        best = (
            self.gate_count
            if self.best_descendant_gate is None
            else self.best_descendant_gate
        )
        return max(0, self.gate_count - best)


class SearchFeedbackRegistry:
    """Persistent online outcomes for exact states in one search invocation."""

    def __init__(self, root_identity: Hashable, root_gate_count: int) -> None:
        self.nodes: dict[int, SearchNodeStats] = {}
        # Exact dedup remains authoritative elsewhere. This compact digest map
        # only attaches feedback edges to known nodes and avoids retaining a
        # second copy of every potentially large exact circuit key.
        self.identity_to_node: dict[str, int] = {}
        self._next_node_id = 0
        self.root_id = self.add_node(
            root_identity, gate_count=root_gate_count, depth=0
        )

    def add_node(
        self,
        identity: Hashable,
        *,
        gate_count: int,
        depth: int,
        parent_id: int | None = None,
        origin_continuation_score: float | None = None,
        step: int = 0,
    ) -> int:
        node_id = self._next_node_id
        order_key = identity_order_key(identity)
        if order_key in self.identity_to_node:
            raise ValueError("exact search node already exists")
        self._next_node_id += 1
        row = SearchNodeStats(
            node_id=node_id,
            identity_order=order_key,
            gate_count=int(gate_count),
            depth=int(depth),
            origin_parent_id=(
                int(parent_id) if parent_id is not None else None
            ),
            origin_continuation_score=(
                float(origin_continuation_score)
                if origin_continuation_score is not None
                else None
            ),
            best_descendant_gate=int(gate_count),
        )
        if parent_id is not None:
            row.parent_ids.add(int(parent_id))
        self.nodes[node_id] = row
        self.identity_to_node[order_key] = node_id
        if parent_id is not None:
            self._propagate_best(parent_id, int(gate_count), step=step)
        return node_id

    def add_parent_edge(
        self, identity: Hashable, parent_id: int, *, step: int
    ) -> int:
        node_id = self.identity_to_node[identity_order_key(identity)]
        row = self.nodes[node_id]
        row.parent_ids.add(int(parent_id))
        best = (
            row.gate_count
            if row.best_descendant_gate is None
            else row.best_descendant_gate
        )
        self._propagate_best(int(parent_id), int(best), step=step)
        return node_id

    def has_identity(self, identity: Hashable) -> bool:
        return identity_order_key(identity) in self.identity_to_node

    def observe_expansion(
        self,
        node_id: int,
        *,
        attempted: int,
        valid: int,
        unique: int,
        duplicate: int,
        invalid: int,
        improving: int,
        best_child_gate: int | None,
        step: int,
    ) -> None:
        row = self.nodes[node_id]
        row.observed_expansions += 1
        row.last_attempted_actions = int(attempted)
        row.last_valid_actions = int(valid)
        row.last_unique_children = int(unique)
        row.last_duplicate_children = int(duplicate)
        row.last_invalid_actions = int(invalid)
        row.last_improving_children = int(improving)
        row.last_best_child_gate = (
            int(best_child_gate) if best_child_gate is not None else None
        )
        row.last_expansion_step = int(step)
        row.attempted_actions += int(attempted)
        row.valid_actions += int(valid)
        row.unique_children += int(unique)
        row.duplicate_children += int(duplicate)
        row.invalid_actions += int(invalid)
        row.improving_children += int(improving)
        if best_child_gate is not None and (
            row.best_child_gate is None
            or int(best_child_gate) < row.best_child_gate
        ):
            row.best_child_gate = int(best_child_gate)
            row.last_improvement_step = int(step)
            self._propagate_best(node_id, int(best_child_gate), step=step)

    def _propagate_best(self, node_id: int, gate_count: int, *, step: int) -> None:
        pending = [int(node_id)]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            row = self.nodes[current]
            if (
                row.best_descendant_gate is not None
                and gate_count >= row.best_descendant_gate
            ):
                continue
            row.best_descendant_gate = int(gate_count)
            row.last_improvement_step = int(step)
            pending.extend(row.parent_ids)

    def rendered_summary(self) -> dict[str, Any]:
        return {
            "nodes": len(self.nodes),
            "attempted_actions": sum(
                row.attempted_actions for row in self.nodes.values()
            ),
            "valid_actions": sum(row.valid_actions for row in self.nodes.values()),
            "unique_children": sum(
                row.unique_children for row in self.nodes.values()
            ),
            "duplicate_children": sum(
                row.duplicate_children for row in self.nodes.values()
            ),
            "invalid_actions": sum(
                row.invalid_actions for row in self.nodes.values()
            ),
            "nodes_with_descendant_gain": sum(
                row.descendant_gain > 0 for row in self.nodes.values()
            ),
        }
