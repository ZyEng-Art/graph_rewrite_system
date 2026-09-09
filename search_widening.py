from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import math
from typing import Any

from search_feedback import SearchNodeStats


@dataclass(frozen=True)
class WideningSelection:
    indices: list[int]
    metrics: dict[str, Any]


def _stable_tiebreak(state: Any, *, seed: int, step: int) -> int:
    payload = (
        seed,
        step,
        int(getattr(state, "last_xfer_id", -1)),
        tuple(map(int, getattr(state, "last_source_slots", ()))),
        tuple(map(int, getattr(state, "last_destination_slots", ()))),
        tuple(tuple(map(int, row)) for row in state.history[-4:]),
    )
    return int.from_bytes(
        hashlib.blake2b(repr(payload).encode("utf-8"), digest_size=8).digest(),
        "little",
    )


def select_widening_revisits(
    states: list[Any],
    *,
    slots: int,
    max_expansions: int,
    seed: int = 73,
    step: int = 0,
    policy: str = "round_robin",
    feedback: dict[int, SearchNodeStats] | None = None,
) -> WideningSelection:
    """Choose exact parent states whose next disjoint rank band is expanded.

    Expansion round zero is the first visit. A state is eligible while its
    next round remains below ``max_expansions``. Eligible parents are
    round-robin interleaved across current expansion rounds; otherwise a
    continuous influx of round-zero children would permanently starve deeper
    rank bands.
    """
    if slots < 0:
        raise ValueError("widening slots must be nonnegative")
    if max_expansions < 1:
        raise ValueError("max expansions must be positive")
    if policy not in {"round_robin", "feedback"}:
        raise ValueError("unknown widening policy")
    if policy == "feedback" and feedback is None:
        raise ValueError("feedback policy requires node statistics")
    eligible = [
        index
        for index, state in enumerate(states)
        if int(getattr(state, "expansion_round", 0)) + 1 < max_expansions
    ]
    if policy == "feedback":
        return _select_feedback_revisits(
            states,
            eligible=eligible,
            slots=slots,
            feedback=feedback or {},
            step=step,
        )

    by_round: dict[int, list[int]] = defaultdict(list)
    for index in eligible:
        by_round[int(getattr(states[index], "expansion_round", 0))].append(index)
    queues = []
    for expansion_round, indices in sorted(by_round.items()):
        indices.sort(
            key=lambda index: (
                int(states[index].gate_count),
                len(states[index].history),
                _stable_tiebreak(states[index], seed=seed, step=step),
            )
        )
        queues.append((expansion_round, deque(indices)))
    selected = []
    while queues and len(selected) < slots:
        next_round = []
        for expansion_round, queue in queues:
            selected.append(queue.popleft())
            if len(selected) == slots:
                break
            if queue:
                next_round.append((expansion_round, queue))
        else:
            queues = next_round
            continue
        break
    return WideningSelection(
        indices=selected,
        metrics={
            "eligible_parents": len(eligible),
            "target_revisits": slots,
            "selected_revisits": len(selected),
            "selected_current_rounds": [
                int(getattr(states[index], "expansion_round", 0))
                for index in selected
            ],
            "selected_next_rounds": [
                int(getattr(states[index], "expansion_round", 0)) + 1
                for index in selected
            ],
        },
    )


def _feedback_row(
    states: list[Any], index: int, feedback: dict[int, SearchNodeStats]
) -> tuple[Any, SearchNodeStats]:
    state = states[index]
    node_id = int(getattr(state, "search_node_id", -1))
    if node_id not in feedback:
        raise ValueError(f"missing feedback for search node {node_id}")
    return state, feedback[node_id]


def _deterministic_key(state: Any, stats: SearchNodeStats) -> tuple:
    return (
        stats.identity_order,
        int(getattr(state, "last_xfer_id", -1)),
        tuple(map(int, getattr(state, "last_source_slots", ()))),
        tuple(map(int, getattr(state, "last_destination_slots", ()))),
    )


def _select_feedback_revisits(
    states: list[Any],
    *,
    eligible: list[int],
    slots: int,
    feedback: dict[int, SearchNodeStats],
    step: int,
) -> WideningSelection:
    """Allocate four deterministic lanes using observed exact-search outcomes."""
    if not eligible or slots == 0:
        return WideningSelection(
            indices=[],
            metrics={
                "policy": "feedback",
                "eligible_parents": len(eligible),
                "target_revisits": slots,
                "selected_revisits": 0,
                "lane_counts": {},
            },
        )

    total_expansions = 1 + sum(
        feedback[int(getattr(states[index], "search_node_id", -1))]
        .observed_expansions
        for index in eligible
    )

    def improvement_key(index: int) -> tuple:
        state, stats = _feedback_row(states, index, feedback)
        return (
            -stats.descendant_gain,
            int(state.gate_count),
            int(state.expansion_round),
            _deterministic_key(state, stats),
        )

    def novelty_key(index: int) -> tuple:
        state, stats = _feedback_row(states, index, feedback)
        return (
            -stats.novel_yield,
            -stats.valid_yield,
            int(state.gate_count),
            int(state.expansion_round),
            _deterministic_key(state, stats),
        )

    def exploration_key(index: int) -> tuple:
        state, stats = _feedback_row(states, index, feedback)
        ucb = math.sqrt(
            math.log1p(total_expansions) / (1 + stats.observed_expansions)
        )
        return (
            -ucb,
            int(state.expansion_round),
            int(state.gate_count),
            _deterministic_key(state, stats),
        )

    best_gate = min(int(states[index].gate_count) for index in eligible)

    def detour_key(index: int) -> tuple:
        state, stats = _feedback_row(states, index, feedback)
        path_best = int(
            getattr(state, "path_best_gate_count", state.gate_count)
        )
        detour = int(state.gate_count) - path_best
        return (
            0 if 0 < detour <= 3 else 1,
            -int(state.expansion_round),
            -int(getattr(state, "depth", len(state.history))),
            abs(int(state.gate_count) - best_gate),
            _deterministic_key(state, stats),
        )

    lanes = (
        ("improvement", improvement_key),
        ("novelty", novelty_key),
        ("exploration", exploration_key),
        ("detour_depth", detour_key),
    )
    ordered = [(name, sorted(eligible, key=key)) for name, key in lanes]
    selected: list[int] = []
    selected_set: set[int] = set()
    lane_counts: dict[str, int] = defaultdict(int)
    cursors = [0] * len(ordered)
    while len(selected) < min(slots, len(eligible)):
        progressed = False
        for lane_index, (name, indices) in enumerate(ordered):
            while (
                cursors[lane_index] < len(indices)
                and indices[cursors[lane_index]] in selected_set
            ):
                cursors[lane_index] += 1
            if cursors[lane_index] >= len(indices):
                continue
            index = indices[cursors[lane_index]]
            cursors[lane_index] += 1
            selected.append(index)
            selected_set.add(index)
            lane_counts[name] += 1
            progressed = True
            if len(selected) == min(slots, len(eligible)):
                break
        if not progressed:
            break

    selected_rows = []
    for index in selected:
        state, stats = _feedback_row(states, index, feedback)
        selected_rows.append(
            {
                "beam_index": index,
                "node_id": stats.node_id,
                "identity_order": stats.identity_order,
                "gate_count": int(state.gate_count),
                "action_depth": int(getattr(state, "depth", len(state.history))),
                "current_round": int(state.expansion_round),
                "next_round": int(state.expansion_round) + 1,
                "descendant_gain": stats.descendant_gain,
                "novel_yield": stats.novel_yield,
                "valid_yield": stats.valid_yield,
                "observed_expansions": stats.observed_expansions,
            }
        )
    return WideningSelection(
        indices=selected,
        metrics={
            "policy": "feedback",
            "eligible_parents": len(eligible),
            "target_revisits": slots,
            "selected_revisits": len(selected),
            "lane_counts": dict(lane_counts),
            "selected_current_rounds": [
                int(states[index].expansion_round) for index in selected
            ],
            "selected_next_rounds": [
                int(states[index].expansion_round) + 1 for index in selected
            ],
            "selected": selected_rows,
            "step": int(step),
        },
    )
