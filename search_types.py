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
    last_xfer_id: int = -1
    last_source_slots: tuple[int, ...] = ()
    last_destination_slots: tuple[int, ...] = ()
    path_best_gate_count: int | None = None
    stagnation_steps: int = 0
    survivor_lane: str = "root"
    exploration_ancestor: bool = False
    recovered_after_exploration: bool = False
    expansion_round: int = 0
    last_action_parent_rank: int = -1
    widening_ancestor: bool = False
    widened_action_trace: tuple[tuple[int, int, int, int], ...] = ()
    search_node_id: int = -1
    search_identity_order: str = ""
    origin_continuation_score: float | None = None


@dataclass(frozen=True)
class Proposal:
    parent: int
    xfer_id: int
    anchor_slot: int
    binding: tuple[int, ...] | None
    probability: float
    next_gate_count: int
    value_score: float = 0.0
    parent_rank: int = -1
