from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Sequence

import torch

from search_types import Proposal


def matcher_logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def bounded_continuation_order(
    proposals: Sequence[Proposal],
    continuation_scores: torch.Tensor,
    *,
    max_matcher_logit_gap: float,
    min_continuation_score_margin: float,
    max_promotions_per_parent: int,
) -> tuple[list[int], dict[str, int | float]]:
    """Swap at most a bounded number of near-tied same-parent actions.

    Every group retains exactly the same global queue positions. Consequently,
    reranking cannot move an action across a parent or immediate gate-count
    allocation boundary; it only changes which sibling occupies an existing
    position.
    """

    if continuation_scores.ndim != 1 or len(continuation_scores) != len(proposals):
        raise ValueError("continuation scores must align with proposals")
    if max_matcher_logit_gap < 0:
        raise ValueError("max matcher logit gap must be nonnegative")
    if min_continuation_score_margin < 0:
        raise ValueError("minimum continuation margin must be nonnegative")
    if max_promotions_per_parent < 1:
        raise ValueError("max promotions per parent must be positive")

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for position, proposal in enumerate(proposals):
        groups[(int(proposal.parent), int(proposal.next_gate_count))].append(position)

    order = list(range(len(proposals)))
    parent_promotions: Counter[int] = Counter()
    eligible_groups = eligible_candidates = promotions = 0
    score_values = continuation_scores.detach().float().cpu().tolist()
    for (parent, _), positions in sorted(groups.items(), key=lambda row: row[1][0]):
        if len(positions) < 2 or parent_promotions[parent] >= max_promotions_per_parent:
            continue
        matcher_leader = min(
            positions,
            key=lambda position: (-float(proposals[position].probability), position),
        )
        leader_logit = matcher_logit(proposals[matcher_leader].probability)
        eligible = [
            position
            for position in positions
            if leader_logit - matcher_logit(proposals[position].probability)
            <= max_matcher_logit_gap
        ]
        if len(eligible) < 2:
            continue
        eligible_groups += 1
        eligible_candidates += len(eligible)
        winner = max(eligible, key=lambda position: (score_values[position], -position))
        if winner == matcher_leader:
            continue
        score_margin = score_values[winner] - score_values[matcher_leader]
        if score_margin < min_continuation_score_margin:
            continue
        order[matcher_leader], order[winner] = order[winner], order[matcher_leader]
        parent_promotions[parent] += 1
        promotions += 1

    moved_rows = sum(index != selected for index, selected in enumerate(order))
    return order, {
        "groups": len(groups),
        "eligible_groups": eligible_groups,
        "eligible_candidates": eligible_candidates,
        "promotions": promotions,
        "moved_rows": moved_rows,
        "parents_promoted": len(parent_promotions),
        "max_matcher_logit_gap": float(max_matcher_logit_gap),
        "min_continuation_score_margin": float(min_continuation_score_margin),
        "max_promotions_per_parent": int(max_promotions_per_parent),
    }
