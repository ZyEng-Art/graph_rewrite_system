from __future__ import annotations

import unittest

import torch

from train_continuation_revisit_ranker import (
    build_pair_tensors,
    feature_row,
    pair_metrics,
)


class ContinuationRevisitRankerTest(unittest.TestCase):
    def test_feature_row_uses_only_selection_time_fields(self) -> None:
        row = {
            "continuation_score": 8.0,
            "action_depth": 32,
            "attempted_actions_before": 512,
            "descendant_gain_before": 4,
            "novel_yield_before": 0.75,
            "valid_yield_before": 0.5,
            "future_descendant_gain": 99,
        }
        self.assertEqual(feature_row(row), [1.0, 0.5, 0.5, 0.5, 0.75, 0.5])

    def test_group_weighting_equalizes_opportunities(self) -> None:
        base = {
            "source_id": 0,
            "selection_step": 1,
            "gate_count": 10,
            "expansion_round_before": 0,
            "observed_expansions_before": 1,
            "additional_observed_expansions": 1,
        }
        rows = [
            dict(base, future_descendant_gain=2),
            dict(base, future_descendant_gain=1),
            dict(base, future_descendant_gain=0),
        ]
        pairs = build_pair_tensors(rows, min_additional_expansions=1)
        self.assertEqual(len(pairs["preferred"]), 3)
        self.assertAlmostEqual(float(pairs["weights"].mean()), 1.0)

    def test_pair_metrics_are_tie_aware(self) -> None:
        pairs = {
            "preferred": torch.tensor([0, 2]),
            "rejected": torch.tensor([1, 3]),
            "source_ids": torch.tensor([0, 0]),
            "steps": torch.tensor([1, 1]),
        }
        metrics = pair_metrics(torch.tensor([2.0, 1.0, 0.0, 0.0]), pairs)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["group_macro_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
