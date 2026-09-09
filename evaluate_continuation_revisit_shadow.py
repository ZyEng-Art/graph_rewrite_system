from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
from typing import Any

import torch


def comparable_pairs(
    rows: list[dict[str, Any]], *, min_additional_expansions: int
) -> list[tuple[int, int, tuple[int, int]]]:
    """Return (preferred, rejected, opportunity) under equal exposure."""

    groups: dict[tuple[int, int, int, int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        additional = int(row["additional_observed_expansions"])
        if additional < min_additional_expansions:
            continue
        key = (
            int(row.get("source_id", -1)),
            int(row["selection_step"]),
            int(row["gate_count"]),
            int(row["expansion_round_before"]),
            int(row["observed_expansions_before"]),
            additional,
        )
        groups[key].append(index)
    pairs = []
    for key, indices in groups.items():
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                left_gain = int(rows[left]["future_descendant_gain"])
                right_gain = int(rows[right]["future_descendant_gain"])
                if left_gain == right_gain:
                    continue
                preferred, rejected = (
                    (left, right) if left_gain > right_gain else (right, left)
                )
                pairs.append((preferred, rejected, (key[0], key[1])))
    return pairs


def tie_aware_pair_metrics(
    rows: list[dict[str, Any]],
    pairs: list[tuple[int, int, tuple[int, int]]],
    field: str,
    *,
    higher_is_better: bool = True,
) -> dict[str, float | int]:
    if not pairs:
        return {"pairs": 0, "accuracy": math.nan, "group_macro_accuracy": math.nan}
    scores = []
    by_opportunity: dict[tuple[int, int], list[float]] = defaultdict(list)
    for preferred, rejected, opportunity in pairs:
        left = float(rows[preferred][field])
        right = float(rows[rejected][field])
        if not higher_is_better:
            left, right = -left, -right
        value = 1.0 if left > right else 0.0 if left < right else 0.5
        scores.append(value)
        by_opportunity[opportunity].append(value)
    return {
        "pairs": len(pairs),
        "opportunities": len(by_opportunity),
        "accuracy": sum(scores) / len(scores),
        "group_macro_accuracy": sum(
            sum(values) / len(values) for values in by_opportunity.values()
        )
        / len(by_opportunity),
    }


def selection_outcomes(rows: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    selected = [row for row in rows if row[field]]
    exposed = [row for row in selected if row["additional_observed_expansions"] > 0]
    improved = [row for row in exposed if row["future_descendant_gain"] > 0]
    return {
        "selected_rows": len(selected),
        "selected_with_additional_exposure": len(exposed),
        "selected_with_future_gain": len(improved),
        "future_gain_rate_given_exposure": len(improved) / len(exposed) if exposed else 0.0,
        "mean_future_gain_given_exposure": (
            sum(float(row["future_descendant_gain"]) for row in exposed) / len(exposed)
            if exposed
            else 0.0
        ),
    }


def evaluate_rows(
    rows: list[dict[str, Any]], *, min_additional_expansions: int
) -> dict[str, Any]:
    pairs = comparable_pairs(
        rows, min_additional_expansions=min_additional_expansions
    )
    return {
        "rows": len(rows),
        "selection_opportunities": len(
            {(row.get("source_id", -1), row["selection_step"]) for row in rows}
        ),
        "rows_with_additional_exposure": sum(
            row["additional_observed_expansions"] >= min_additional_expansions
            for row in rows
        ),
        "comparable_pairs": len(pairs),
        "continuation_score": tie_aware_pair_metrics(
            rows, pairs, "continuation_score"
        ),
        "feedback_descendant_gain_before": tie_aware_pair_metrics(
            rows, pairs, "descendant_gain_before"
        ),
        "feedback_novel_yield_before": tie_aware_pair_metrics(
            rows, pairs, "novel_yield_before"
        ),
        "feedback_valid_yield_before": tie_aware_pair_metrics(
            rows, pairs, "valid_yield_before"
        ),
        "feedback_selected": tie_aware_pair_metrics(
            rows, pairs, "feedback_selected"
        ),
        "continuation_shadow_selected": tie_aware_pair_metrics(
            rows, pairs, "shadow_selected"
        ),
        "selection_outcomes": {
            "feedback": selection_outcomes(rows, "feedback_selected"),
            "continuation_shadow": selection_outcomes(rows, "shadow_selected"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate continuation scores at branch revisit opportunities."
    )
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-additional-expansions", type=int, default=1)
    args = parser.parse_args()
    if args.min_additional_expansions < 1:
        parser.error("minimum additional expansions must be positive")

    paths = sorted(
        {Path(match) for pattern in args.audits for match in glob.glob(pattern)}
    )
    if not paths:
        raise ValueError("no audit files matched")
    all_rows = []
    sources = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        shadow = payload.get("continuation_revisit_shadow")
        if not shadow:
            raise ValueError(f"audit lacks continuation revisit shadow rows: {path}")
        rows = [dict(row, source_id=len(sources)) for row in shadow["rows"]]
        all_rows.extend(rows)
        sources.append(
            {
                "path": str(path.resolve()),
                "qasm": payload.get("qasm"),
                "evaluation": evaluate_rows(
                    rows,
                    min_additional_expansions=args.min_additional_expansions,
                ),
            }
        )
    result = {
        "min_additional_expansions": args.min_additional_expansions,
        "sources": sources,
        "aggregate": evaluate_rows(
            all_rows,
            min_additional_expansions=args.min_additional_expansions,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
