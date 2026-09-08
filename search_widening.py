from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
from typing import Any


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
    eligible = [
        index
        for index, state in enumerate(states)
        if int(getattr(state, "expansion_round", 0)) + 1 < max_expansions
    ]
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
