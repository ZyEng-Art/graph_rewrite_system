from __future__ import annotations

import unittest

import torch

from sibling_continuation_ranker import (
    SiblingContinuationRanker,
    continuation_ranker_inputs,
)
from train_sibling_continuation_ranker import tie_aware_accuracy
from train_sibling_continuation_ranker import prefix_tensors


class SiblingContinuationRankerTest(unittest.TestCase):
    def test_inputs_exclude_post_search_labels(self) -> None:
        payload = {
            "features": torch.ones(3, 5),
            "probabilities": torch.tensor([0.2, 0.5, 0.8]),
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
        self.assertTrue(
            torch.allclose(model(inputs), torch.logit(payload["probabilities"]))
        )

    def test_tie_aware_accuracy(self) -> None:
        self.assertAlmostEqual(
            tie_aware_accuracy(
                torch.tensor([2.0, 1.0, 1.0]),
                torch.tensor([1.0, 1.0, 2.0]),
            ),
            0.5,
        )

    def test_prefix_encoder_is_zero_residual_at_initialization(self) -> None:
        payload = {
            "parent_node_ids": torch.tensor([3, 4, 3]),
            "parent_histories": [
                {"node_id": 3, "history": [[1, 8], [2, 9], [3, 10]]},
                {"node_id": 4, "history": []},
            ],
        }
        tokens, lengths = prefix_tensors(payload, 2)
        self.assertEqual(tokens.tolist(), [[3, 4], [0, 0], [3, 4]])
        self.assertEqual(lengths.tolist(), [2, 0, 2])
        inputs = torch.zeros(3, 10)
        inputs[:, 2] = torch.tensor([0.2, 0.5, 0.8])
        model = SiblingContinuationRanker(
            10,
            hidden_width=8,
            dropout=0.0,
            base_probability_index=2,
            num_xfers=4,
            prefix_width=4,
        )
        self.assertTrue(
            torch.allclose(
                model(inputs, tokens, lengths), torch.logit(inputs[:, 2])
            )
        )
        unseen = tokens.clone()
        unseen[0, 0] = 999
        self.assertEqual(model(inputs, unseen, lengths).shape, (3,))

    def test_row_aligned_prefixes_override_node_history(self) -> None:
        payload = {
            "parent_node_ids": torch.tensor([3, 3]),
            "parent_histories": [
                {"node_id": 3, "history": [[1, 8]]},
            ],
            "parent_history_xfer_ids": [[4, 5], [6]],
        }
        tokens, lengths = prefix_tensors(payload, 2)
        self.assertEqual(tokens.tolist(), [[5, 6], [7, 0]])
        self.assertEqual(lengths.tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
