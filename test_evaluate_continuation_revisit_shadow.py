from __future__ import annotations

import unittest

from evaluate_continuation_revisit_shadow import (
    comparable_pairs,
    evaluate_rows,
)


def row(
    score: float,
    future_gain: int,
    *,
    node: int,
    additional: int = 1,
) -> dict:
    return {
        "selection_step": 4,
        "node_id": node,
        "continuation_score": score,
        "shadow_selected": node == 0,
        "feedback_selected": node == 1,
        "gate_count": 10,
        "expansion_round_before": 0,
        "observed_expansions_before": 1,
        "additional_observed_expansions": additional,
        "future_descendant_gain": future_gain,
        "descendant_gain_before": 0,
        "novel_yield_before": float(node),
        "valid_yield_before": 1.0,
    }


class EvaluateContinuationRevisitShadowTest(unittest.TestCase):
    def test_pairs_require_equal_exposure_and_non_tied_future_gain(self) -> None:
        rows = [
            row(3.0, 2, node=0),
            row(2.0, 0, node=1),
            row(1.0, 1, node=2, additional=2),
        ]
        self.assertEqual(
            comparable_pairs(rows, min_additional_expansions=1),
            [(0, 1, (-1, 4))],
        )

    def test_reports_model_and_selection_accuracy(self) -> None:
        metrics = evaluate_rows(
            [row(3.0, 2, node=0), row(2.0, 0, node=1)],
            min_additional_expansions=1,
        )
        self.assertEqual(metrics["comparable_pairs"], 1)
        self.assertEqual(metrics["continuation_score"]["accuracy"], 1.0)
        self.assertEqual(metrics["feedback_selected"]["accuracy"], 0.0)
        self.assertEqual(
            metrics["continuation_shadow_selected"]["accuracy"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
