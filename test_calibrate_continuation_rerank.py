from __future__ import annotations

import unittest

import torch

from calibrate_continuation_rerank import intervention_metrics, wilson_lower


class CalibrateContinuationRerankTest(unittest.TestCase):
    def test_interventions_count_corrections_and_harm(self) -> None:
        # Last eight columns are the auxiliary continuation inputs. Probability
        # and scaled gate delta occupy the first two auxiliary positions.
        inputs = torch.zeros((4, 8))
        inputs[:, 0] = torch.tensor([0.49, 0.51, 0.51, 0.49])
        inputs[:, 1] = 0.0
        corpus = {"inputs": inputs, "input_width": 8}
        pairs = {
            "preferred": torch.tensor([0, 2]),
            "rejected": torch.tensor([1, 3]),
        }
        metrics = intervention_metrics(
            corpus,
            pairs,
            torch.tensor([2.0, 0.0, 0.0, 2.0]),
            max_matcher_logit_gap=0.25,
            min_continuation_score_margin=0.5,
        )
        self.assertEqual(metrics["interventions"], 2)
        self.assertEqual(metrics["corrected"], 1)
        self.assertEqual(metrics["harmed"], 1)
        self.assertEqual(metrics["net_corrected"], 0)

    def test_gate_delta_and_margin_filters_abstain(self) -> None:
        inputs = torch.zeros((2, 8))
        inputs[:, 0] = torch.tensor([0.49, 0.51])
        inputs[:, 1] = torch.tensor([0.0, 0.125])
        corpus = {"inputs": inputs, "input_width": 8}
        pairs = {"preferred": torch.tensor([0]), "rejected": torch.tensor([1])}
        metrics = intervention_metrics(
            corpus,
            pairs,
            torch.tensor([2.0, 0.0]),
            max_matcher_logit_gap=0.25,
            min_continuation_score_margin=0.5,
        )
        self.assertEqual(metrics["interventions"], 0)

    def test_wilson_lower_is_conservative(self) -> None:
        self.assertLess(wilson_lower(8, 10), 0.8)
        self.assertEqual(wilson_lower(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
