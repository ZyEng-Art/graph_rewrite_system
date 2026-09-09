from __future__ import annotations

import unittest

import torch

from evaluate_continuation_revisit_ranker import score_rows


class EvaluateContinuationRevisitRankerTest(unittest.TestCase):
    def test_score_rows_replays_checkpoint_normalization(self) -> None:
        checkpoint = {
            "feature_names": ("action_depth", "novel_yield_before"),
            "feature_mean": torch.tensor([0.25, 0.5]),
            "feature_scale": torch.tensor([0.25, 0.25]),
            "model": {"weight": torch.tensor([[2.0, -1.0]])},
        }
        row = {
            "continuation_score": 0.0,
            "action_depth": 32,
            "attempted_actions_before": 0,
            "descendant_gain_before": 0,
            "novel_yield_before": 0.75,
            "valid_yield_before": 0.0,
        }
        self.assertEqual(float(score_rows(checkpoint, [row])[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
