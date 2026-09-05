from __future__ import annotations

import math
import unittest

import torch

from ppo_core import (
    HierarchicalPPOActorCritic,
    hierarchical_policy_log_probs,
)


class HierarchicalPolicyTest(unittest.TestCase):
    def test_factorized_policy_is_normalized(self) -> None:
        result = hierarchical_policy_log_probs(
            node_logits=torch.tensor([[2.0, 1.0, -3.0]]),
            node_mask=torch.tensor([[True, True, True]]),
            candidate_logits=torch.tensor([[0.0, 1.0, -1.0, 4.0]]),
            candidate_nodes=torch.tensor([[0, 0, 1, 2]]),
            candidate_mask=torch.tensor([[True, True, True, False]]),
            stop_logits=torch.tensor([math.log(0.25 / 0.75)]),
        )
        probabilities = result.log_probs.exp()
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(1)))
        self.assertAlmostEqual(float(probabilities[0, 0]), 0.25, places=6)
        # Node 2 has no retained candidate and receives no continue mass.
        self.assertTrue(torch.isneginf(result.node_log_probs[0, 2]))
        node_zero = probabilities[0, 1] + probabilities[0, 2]
        node_one = probabilities[0, 3]
        self.assertAlmostEqual(
            float(node_zero / node_one), math.exp(1.0), places=5
        )
        self.assertAlmostEqual(
            float(probabilities[0, 2] / probabilities[0, 1]),
            math.exp(1.0),
            places=5,
        )

    def test_stop_is_only_action_for_empty_candidate_row(self) -> None:
        result = hierarchical_policy_log_probs(
            node_logits=torch.zeros(2, 3),
            node_mask=torch.ones(2, 3, dtype=torch.bool),
            candidate_logits=torch.tensor([[1.0], [2.0]]),
            candidate_nodes=torch.zeros(2, 1, dtype=torch.long),
            candidate_mask=torch.tensor([[False], [True]]),
            stop_logits=torch.zeros(2),
        )
        self.assertTrue(torch.equal(result.mask[:, 0], torch.ones(2, dtype=torch.bool)))
        self.assertTrue(torch.isneginf(result.log_probs[0, 1]))
        # A state with no continuation candidates is deterministically stopped.
        normalized = result.log_probs.masked_fill(~result.mask, -torch.inf)
        probabilities = torch.softmax(normalized, dim=1)
        self.assertAlmostEqual(float(probabilities[0, 0]), 1.0, places=6)

    def test_actor_outputs_have_gradients_and_expected_shapes(self) -> None:
        torch.manual_seed(3)
        actor = HierarchicalPPOActorCritic(width=8, hidden_size=12)
        node_features = torch.randn(2, 4, 8)
        node_mask = torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        )
        candidate_features = torch.randn(2, 3, actor.policy_feature_dim)
        matcher_logits = torch.randn(2, 3)
        candidate_nodes = torch.tensor([[0, 0, 2], [0, 1, 0]])
        candidate_mask = torch.tensor([[True, True, True], [True, True, False]])
        prefix = torch.randn(2, 8)
        state = torch.randn(2, actor.state_feature_dim)
        policy = actor.policy(
            node_features,
            node_mask,
            candidate_features,
            matcher_logits,
            candidate_nodes,
            candidate_mask,
            prefix,
            state,
        )
        self.assertEqual(policy.log_probs.shape, (2, 4))
        self.assertEqual(policy.mask.shape, (2, 4))
        loss = -policy.log_probs[policy.mask].mean()
        loss.backward()
        self.assertIsNotNone(actor.node_output.weight.grad)
        self.assertIsNotNone(actor.pattern[-1].weight.grad)
        self.assertIsNotNone(actor.stop[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
