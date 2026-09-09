from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable


@dataclass(frozen=True)
class SurvivorSelection:
    states: list[Any]
    metrics: dict[str, Any]


def path_best_gate_count(state: Any) -> int:
    value = getattr(state, "path_best_gate_count", None)
    return int(state.gate_count if value is None else value)


def gate_detour(state: Any) -> int:
    return int(state.gate_count) - path_best_gate_count(state)


def is_exploration_candidate(
    state: Any,
    *,
    max_stagnation: int,
    max_detour: int,
) -> bool:
    stagnation = int(getattr(state, "stagnation_steps", 0))
    return (
        0 < stagnation <= max_stagnation
        and 0 <= gate_detour(state) <= max_detour
    )


def gate_priority(state: Any) -> tuple[int, int, str]:
    # Feedback-search states carry a digest of their exact Quartz identity.
    # Using it only as a final ordering key makes equal-cost survivor cuts
    # independent of CUDA candidate materialization order.  Historical states
    # have an empty key, preserving their former stable-input ordering.
    return (
        int(state.gate_count),
        len(state.history),
        str(getattr(state, "search_identity_order", "")),
    )


def _stable_tiebreak(state: Any, seed: int, step: int) -> int:
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


def exploration_signature(state: Any) -> tuple[int, int, int]:
    """Cheap structural/action bucket used only for diversity, never identity."""
    stagnation = int(getattr(state, "stagnation_steps", 0))
    return (
        int(getattr(state, "last_xfer_id", -1)),
        min(3, gate_detour(state)),
        min(3, max(0, stagnation - 1) // 2),
    )


def _diverse_exploration_order(
    states: Iterable[Any], *, seed: int, step: int
) -> list[Any]:
    groups: dict[tuple[int, int, int], list[Any]] = defaultdict(list)
    for state in states:
        groups[exploration_signature(state)].append(state)
    queues = []
    for signature, rows in groups.items():
        rows.sort(
            key=lambda state: (
                gate_detour(state),
                int(getattr(state, "stagnation_steps", 0)),
                _stable_tiebreak(state, seed, step),
            )
        )
        random_key = hashlib.blake2b(
            repr((seed, step, signature)).encode("utf-8"), digest_size=8
        ).digest()
        group_key = (
            gate_detour(rows[0]),
            int(getattr(rows[0], "stagnation_steps", 0)),
            random_key,
        )
        queues.append((group_key, deque(rows)))
    queues.sort(key=lambda row: row[0])
    ordered = []
    while queues:
        next_round = []
        for key, queue in queues:
            ordered.append(queue.popleft())
            if queue:
                next_round.append((key, queue))
        queues = next_round
    return ordered


def select_survivors(
    children: list[Any],
    *,
    beam_size: int,
    exploration_fraction: float = 0.0,
    exploration_max_stagnation: int = 8,
    exploration_max_detour: int = 2,
    seed: int = 73,
    step: int = 0,
) -> SurvivorSelection:
    """Select a fixed-size beam with an optional no-improvement survival lane.

    A zero exploration fraction is deliberately identical to the historical
    `(gate_count, history_length)` selection. The exploration lane is filled
    only from candidates outside the exploitation prefix, so it represents a
    real reservation rather than relabeling already-selected low-cost states.
    """
    if beam_size < 1:
        raise ValueError("beam_size must be positive")
    if not 0.0 <= exploration_fraction < 1.0:
        raise ValueError("exploration_fraction must be in [0, 1)")
    if exploration_max_stagnation < 1:
        raise ValueError("exploration_max_stagnation must be positive")
    if exploration_max_detour < 0:
        raise ValueError("exploration_max_detour must be nonnegative")

    ranked = sorted(children, key=gate_priority)
    if not ranked:
        return SurvivorSelection([], {
            "policy": "gate" if exploration_fraction == 0.0 else "dual_lane",
            "candidate_states": 0,
            "exploitation_target": beam_size,
            "exploration_target": 0,
            "exploitation_selected": 0,
            "exploration_selected": 0,
            "exploration_eligible_outside_prefix": 0,
            "fallback_selected": 0,
        })

    exploration_target = min(
        beam_size - 1,
        int(math.floor(beam_size * exploration_fraction)),
    )
    exploitation_target = beam_size - exploration_target
    exploitation = ranked[:exploitation_target]
    selected_ids = {id(state) for state in exploitation}

    exploration_pool = [
        state
        for state in ranked[exploitation_target:]
        if is_exploration_candidate(
            state,
            max_stagnation=exploration_max_stagnation,
            max_detour=exploration_max_detour,
        )
    ]
    exploration = _diverse_exploration_order(
        exploration_pool, seed=seed, step=step
    )[:exploration_target]
    selected_ids.update(id(state) for state in exploration)

    fallback_target = beam_size - len(exploitation) - len(exploration)
    fallback = [
        state for state in ranked if id(state) not in selected_ids
    ][:fallback_target]
    for state in exploitation:
        state.survivor_lane = "exploitation"
    for state in exploration:
        state.survivor_lane = "exploration"
    for state in fallback:
        state.survivor_lane = "fallback"

    states = exploitation + exploration + fallback
    states.sort(key=gate_priority)
    return SurvivorSelection(
        states=states,
        metrics={
            "policy": "gate" if exploration_target == 0 else "dual_lane",
            "candidate_states": len(children),
            "exploitation_target": exploitation_target,
            "exploration_target": exploration_target,
            "exploitation_selected": len(exploitation),
            "exploration_selected": len(exploration),
            "exploration_eligible_outside_prefix": len(exploration_pool),
            "fallback_selected": len(fallback),
            "exploration_gate_counts": [
                int(state.gate_count) for state in exploration
            ],
            "exploration_stagnation_steps": [
                int(state.stagnation_steps) for state in exploration
            ],
        },
    )
