from __future__ import annotations

import unittest

import torch

from verify_continuation_shadow import pair_order_metrics, values_equal


class VerifyContinuationShadowTest(unittest.TestCase):
    def test_values_equal_handles_nested_tensors(self) -> None:
        left = {"rows": [torch.tensor([1, 2]), {"value": 3}]}
        self.assertTrue(values_equal(left, {"rows": [torch.tensor([1, 2]), {"value": 3}]}))
        self.assertFalse(values_equal(left, {"rows": [torch.tensor([1, 3]), {"value": 3}]}))

    def test_pair_metrics_detect_order_changes(self) -> None:
        preferences = {
            "train_pairs": [
                {"preferred_row": 0, "rejected_row": 1},
                {"preferred_row": 2, "rejected_row": 3},
            ],
            "test_pairs": [],
        }
        metrics = pair_order_metrics(
            preferences,
            torch.tensor([2.0, 1.0, 1.0, 2.0]),
            torch.tensor([2.1, 1.0, 3.0, 2.0]),
        )
        self.assertEqual(metrics["pairs"], 2)
        self.assertEqual(metrics["order_disagreements"], 1)
        self.assertEqual(metrics["online_accuracy"], 0.5)
        self.assertEqual(metrics["offline_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
