from __future__ import annotations

import unittest

import torch

from sibling_continuation_ranker import (
    SiblingContinuationRanker,
    continuation_ranker_inputs,
)
from train_sibling_continuation_ranker import tie_aware_accuracy


class SiblingContinuationRankerTest(unittest.TestCase):
    def test_inputs_exclude_post_search_labels(self) -> None:
        payload = {
            "features": torch.ones(3, 5),
            "probabilities": torch.ones(3),
            "gate_deltas": torch.zeros(3),
            "parent_gate_counts": torch.full((3,), 20),
            "steps": torch.arange(3),
            "action_parent_ranks": torch.arange(3),
            "parent_expansion_rounds": torch.zeros(3),
            "parent_stagnation_steps": torch.zeros(3),
            "parent_action_depths": torch.zeros(3),
            "descendant_labels": {"continuation_gains": torch.full((3,), 99)},
        }
        inputs = continuation_ranker_inputs(payload)
        self.assertEqual(inputs.shape, (3, 13))
        model = SiblingContinuationRanker(13, hidden_width=8, dropout=0.0)
        self.assertEqual(model(inputs).shape, (3,))

    def test_tie_aware_accuracy(self) -> None:
        self.assertAlmostEqual(
            tie_aware_accuracy(
                torch.tensor([2.0, 1.0, 1.0]),
                torch.tensor([1.0, 1.0, 2.0]),
            ),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
