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


def continuation_revisit_shadow_rows(
    states: list[Any],
    *,
    max_expansions: int,
    slots: int,
    step: int,
    feedback: dict[int, SearchNodeStats],
    actually_selected: list[int],
) -> list[dict[str, Any]]:
    """Rank eligible scored branches without changing the widening selection."""

    if slots < 1:
        return []
    eligible = [
        index
        for index, state in enumerate(states)
        if int(getattr(state, "expansion_round", 0)) + 1 < max_expansions
        and getattr(state, "origin_continuation_score", None) is not None
    ]
    eligible.sort(
        key=lambda index: (
            -float(states[index].origin_continuation_score),
            int(states[index].gate_count),
            int(states[index].expansion_round),
            _deterministic_key(*_feedback_row(states, index, feedback)),
        )
    )
    actual = set(actually_selected)
    rows = []
    for rank, index in enumerate(eligible):
        state, stats = _feedback_row(states, index, feedback)
        best_before = (
            stats.gate_count
            if stats.best_descendant_gate is None
            else int(stats.best_descendant_gate)
        )
        rows.append(
            {
                "selection_step": int(step),
                "beam_index": int(index),
                "node_id": int(stats.node_id),
                "continuation_score": float(state.origin_continuation_score),
                "shadow_rank": int(rank),
                "shadow_selected": bool(rank < slots),
                "feedback_selected": bool(index in actual),
                "gate_count": int(state.gate_count),
                "action_depth": int(getattr(state, "depth", len(state.history))),
                "expansion_round_before": int(state.expansion_round),
                "observed_expansions_before": int(stats.observed_expansions),
                "attempted_actions_before": int(stats.attempted_actions),
                "best_descendant_gate_before": best_before,
                "descendant_gain_before": int(stats.descendant_gain),
                "novel_yield_before": float(stats.novel_yield),
                "valid_yield_before": float(stats.valid_yield),
                "last_attempted_actions_before": int(stats.last_attempted_actions),
                "last_improving_children_before": int(stats.last_improving_children),
                "last_improving_yield_before": float(stats.last_improving_yield),
                "last_unique_yield_before": float(stats.last_unique_yield),
                "last_valid_yield_before": float(stats.last_valid_yield),
                "last_best_child_gate_before": stats.last_best_child_gate,
                "last_expansion_step_before": int(stats.last_expansion_step),
            }
        )
    return rows


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
    if policy not in {
        "round_robin",
        "feedback",
        "feedback_balanced",
        "feedback_marginal",
        "probe_halving",
        "feedback_ucb",
    }:
        raise ValueError("unknown widening policy")
    if policy in {
        "feedback",
        "feedback_balanced",
        "feedback_marginal",
        "probe_halving",
        "feedback_ucb",
    } and feedback is None:
        raise ValueError("feedback policy requires node statistics")
    eligible = [
        index
        for index, state in enumerate(states)
        if int(getattr(state, "expansion_round", 0)) + 1 < max_expansions
    ]
    if policy == "feedback_balanced":
        return _select_balanced_feedback_revisits(
            states,
            eligible=eligible,
            slots=slots,
            feedback=feedback or {},
            step=step,
        )
    if policy == "feedback":
        return _select_feedback_revisits(
            states,
            eligible=eligible,
            slots=slots,
            feedback=feedback or {},
            step=step,
        )
    if policy == "feedback_ucb":
        return _select_feedback_revisits(
            states,
            eligible=eligible,
            slots=slots,
            feedback=feedback or {},
            step=step,
            confidence_yield=True,
        )
    if policy == "feedback_marginal":
        return _select_feedback_revisits(
            states,
            eligible=eligible,
            slots=slots,
            feedback=feedback or {},
            step=step,
            marginal_gain=True,
        )
    if policy == "probe_halving":
        return _select_probe_halving_revisits(
            states,
            eligible=eligible,
            slots=slots,
            max_expansions=max_expansions,
            seed=seed,
            step=step,
            feedback=feedback or {},
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


def _select_balanced_feedback_revisits(
    states: list[Any],
    *,
    eligible: list[int],
    slots: int,
    feedback: dict[int, SearchNodeStats],
    step: int,
) -> WideningSelection:
    """Round-robin surviving siblings before filling normal feedback lanes.

    This policy is intended for continuation-label collection. It gives sibling
    children created from the same exact parent comparable direct expansion
    exposure, while retaining the normal feedback policy as a deterministic
    fallback when sibling cohorts cannot fill the revisit budget.
    """

    cohorts: dict[int, list[int]] = defaultdict(list)
    for index in eligible:
        _, stats = _feedback_row(states, index, feedback)
        if stats.origin_parent_id is not None:
            cohorts[int(stats.origin_parent_id)].append(index)
    queues = []
    for parent_id, indices in cohorts.items():
        if len(indices) < 2:
            continue
        indices.sort(
            key=lambda index: (
                _feedback_row(states, index, feedback)[1].observed_expansions,
                int(states[index].expansion_round),
                _deterministic_key(
                    *_feedback_row(states, index, feedback)
                ),
            )
        )
        queues.append((parent_id, deque(indices)))
    queues.sort(
        key=lambda row: (
            min(
                _feedback_row(states, index, feedback)[1].observed_expansions
                for index in row[1]
            ),
            row[0],
        )
    )
    selected = []
    while queues and len(selected) < min(slots, len(eligible)):
        next_round = []
        for parent_id, queue in queues:
            selected.append(queue.popleft())
            if queue:
                next_round.append((parent_id, queue))
            if len(selected) == min(slots, len(eligible)):
                break
        queues = next_round

    selected_set = set(selected)
    remaining = [index for index in eligible if index not in selected_set]
    fallback = _select_feedback_revisits(
        states,
        eligible=remaining,
        slots=max(0, slots - len(selected)),
        feedback=feedback,
        step=step,
    )
    selected.extend(fallback.indices)
    lane_counts = {"balanced_sibling": len(selected_set)}
    for name, count in fallback.metrics.get("lane_counts", {}).items():
        lane_counts[name] = int(count)
    return WideningSelection(
        indices=selected,
        metrics={
            "policy": "feedback_balanced",
            "eligible_parents": len(eligible),
            "eligible_sibling_groups": sum(
                len(indices) >= 2 for indices in cohorts.values()
            ),
            "target_revisits": slots,
            "selected_revisits": len(selected),
            "lane_counts": lane_counts,
            "selected_current_rounds": [
                int(states[index].expansion_round) for index in selected
            ],
            "selected_next_rounds": [
                int(states[index].expansion_round) + 1 for index in selected
            ],
            "step": int(step),
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


def _select_probe_halving_revisits(
    states: list[Any],
    *,
    eligible: list[int],
    slots: int,
    max_expansions: int,
    seed: int,
    step: int,
    feedback: dict[int, SearchNodeStats],
) -> WideningSelection:
    """Keep a round-robin safety lane and race the top half of sibling probes."""
    target = min(slots, len(eligible))
    if target == 0:
        return WideningSelection(
            indices=[],
            metrics={
                "policy": "probe_halving",
                "eligible_parents": len(eligible),
                "target_revisits": slots,
                "selected_revisits": 0,
                "lane_counts": {},
            },
        )

    safety_slots = (target + 1) // 2
    safety = select_widening_revisits(
        states,
        slots=safety_slots,
        max_expansions=max_expansions,
        seed=seed,
        step=step,
        policy="round_robin",
    ).indices
    selected = list(safety)
    selected_set = set(selected)

    def recent_probe_key(index: int) -> tuple:
        state, stats = _feedback_row(states, index, feedback)
        best_delta = (
            0
            if stats.last_best_child_gate is None
            else max(0, int(state.gate_count) - stats.last_best_child_gate)
        )
        return (
            0 if stats.last_attempted_actions > 0 else 1,
            stats.last_improving_children,
            stats.last_improving_yield,
            best_delta,
            stats.last_unique_yield,
            stats.last_valid_yield,
            -stats.observed_expansions,
            -int(state.gate_count),
        )

    cohorts: dict[int, list[int]] = defaultdict(list)
    for index in eligible:
        _, stats = _feedback_row(states, index, feedback)
        if stats.origin_parent_id is not None:
            cohorts[int(stats.origin_parent_id)].append(index)

    promoted_queues = []
    for parent_id, indices in cohorts.items():
        if len(indices) < 2:
            continue
        ordered = sorted(
            indices,
            key=lambda index: (
                tuple(-value for value in recent_probe_key(index)),
                _deterministic_key(*_feedback_row(states, index, feedback)),
            ),
        )
        promoted = [
            index
            for index in ordered[: (len(ordered) + 1) // 2]
            if index not in selected_set
        ]
        if promoted:
            minimum_exposure = min(
                _feedback_row(states, index, feedback)[1].observed_expansions
                for index in promoted
            )
            promoted_queues.append(
                (minimum_exposure, parent_id, deque(promoted))
            )
    promoted_queues.sort(key=lambda row: (row[0], row[1]))

    promoted_count = 0
    while promoted_queues and len(selected) < target:
        next_queues = []
        for exposure, parent_id, queue in promoted_queues:
            index = queue.popleft()
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
                promoted_count += 1
            if queue:
                next_queues.append((exposure, parent_id, queue))
            if len(selected) == target:
                break
        promoted_queues = next_queues

    fallback_count = 0
    if len(selected) < target:
        fallback = select_widening_revisits(
            states,
            slots=len(eligible),
            max_expansions=max_expansions,
            seed=seed,
            step=step,
            policy="round_robin",
        ).indices
        for index in fallback:
            if index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            fallback_count += 1
            if len(selected) == target:
                break

    return WideningSelection(
        indices=selected,
        metrics={
            "policy": "probe_halving",
            "eligible_parents": len(eligible),
            "eligible_sibling_cohorts": sum(
                len(indices) >= 2 for indices in cohorts.values()
            ),
            "target_revisits": slots,
            "selected_revisits": len(selected),
            "lane_counts": {
                "round_robin_safety": len(safety),
                "probe_promoted": promoted_count,
                "round_robin_fallback": fallback_count,
            },
            "selected_current_rounds": [
                int(states[index].expansion_round) for index in selected
            ],
            "selected_next_rounds": [
                int(states[index].expansion_round) + 1 for index in selected
            ],
            "step": int(step),
        },
    )


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
    confidence_yield: bool = False,
    marginal_gain: bool = False,
) -> WideningSelection:
    """Allocate four deterministic lanes using observed exact-search outcomes."""
    if not eligible or slots == 0:
        return WideningSelection(
            indices=[],
            metrics={
                "policy": (
                    "feedback_ucb"
                    if confidence_yield
                    else "feedback_marginal" if marginal_gain else "feedback"
                ),
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
            stats.descendant_gain if marginal_gain else -stats.descendant_gain,
            int(state.gate_count),
            int(state.expansion_round),
            _deterministic_key(state, stats),
        )

    def novelty_key(index: int) -> tuple:
        state, stats = _feedback_row(states, index, feedback)
        if confidence_yield:
            yield_score = _wilson_upper_bound(
                stats.unique_children, stats.attempted_actions
            )
        else:
            yield_score = stats.novel_yield
        return (
            -yield_score,
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
        ("useful_yield_ucb" if confidence_yield else "novelty", novelty_key),
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
            "policy": (
                "feedback_ucb"
                if confidence_yield
                else "feedback_marginal" if marginal_gain else "feedback"
            ),
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


def _wilson_upper_bound(successes: int, trials: int, *, z: float = 1.96) -> float:
    """95% Wilson upper bound for a bounded useful-child yield."""
    trials = int(trials)
    if trials <= 0:
        return 1.0
    successes = min(max(0, int(successes)), trials)
    probability = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = probability + z_squared / (2.0 * trials)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return (center + radius) / denominator
