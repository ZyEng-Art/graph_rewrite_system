from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BeamState:
    graph: Any
    snapshot: dict | None
    guid_to_slot: dict[int, int]
    next_slot: int
    last_touched: Any
    rewrite_distance: Any
    previous_preferred: set[int]
    local_streak: int
    gate_count: int
    depth: int
    history: tuple[tuple[int, int], ...]
    topology_index: Any = None
    exact_graph_checkpoint: Any = None
    exact_slot_checkpoint: dict[int, int] | None = None
    exact_checkpoint_depth: int = 0


@dataclass(frozen=True)
class Proposal:
    parent: int
    xfer_id: int
    anchor_slot: int
    binding: tuple[int, ...] | None
    probability: float
    next_gate_count: int
    value_score: float = 0.0
