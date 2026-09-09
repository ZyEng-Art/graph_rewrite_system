from __future__ import annotations

import unittest

import torch

from gpu_proposals import SelectedProposalTensors
from hierarchical_ppo_rollout import (
    HierarchicalPPOTransition,
    hierarchical_ppo_update,
    mask_rejected_proposals,
    pad_proposal_node_positions,
    proposal_node_positions,
)
from ppo_core import HierarchicalPPOActorCritic


class HierarchicalRolloutTest(unittest.TestCase):
    def test_exact_rejection_cache_masks_full_action_only(self) -> None:
        proposals = SelectedProposalTensors(
            parent_ids=torch.tensor([0, 0, 1]),
            xfer_ids=torch.tensor([8, 9, 8]),
            source_ids=torch.tensor([2, 2, 2]),
            anchor_slots=torch.tensor([7, 7, 7]),
            bindings=torch.tensor([[7, 8, -1], [7, 8, -1], [7, 8, -1]]),
            probabilities=torch.ones(3),
            gate_deltas=torch.zeros(3, dtype=torch.long),
            next_gate_counts=torch.full((3,), 10),
            value_scores=torch.zeros(3),
            parent_ranks=torch.tensor([0, 1, 0]),
        )
        candidate_indices = torch.tensor([[0, 1], [2, 0]])
        candidate_mask = torch.tensor([[True, True], [True, False]])
        state_keys = [(100, 200), (100, 201)]
        cache = {(100, 200): {(8, (7, 8))}}

        masked = mask_rejected_proposals(
            candidate_mask,
            candidate_indices,
            proposals,
            state_keys,
            cache,
        )

        self.assertEqual(masked, 1)
        self.assertTrue(
            torch.equal(
                candidate_mask,
                torch.tensor([[False, True], [True, False]]),
            )
        )

    def test_expanded_actions_map_back_to_selected_node_branch(self) -> None:
        proposals = SelectedProposalTensors(
            parent_ids=torch.tensor([0, 0, 1]),
            xfer_ids=torch.tensor([8, 9, 10]),
            source_ids=torch.tensor([2, 2, 3]),
            anchor_slots=torch.tensor([7, 7, 4]),
            bindings=torch.tensor([[7, 8], [7, 8], [4, 5]]),
            probabilities=torch.tensor([0.9, 0.9, 0.8]),
            gate_deltas=torch.tensor([0, 1, -1]),
            next_gate_counts=torch.tensor([10, 11, 9]),
            value_scores=torch.zeros(3),
            parent_ranks=torch.tensor([0, 1, 0]),
        )
        positions = proposal_node_positions(
            proposals,
            selected_nodes=torch.tensor([[3, 7, 9], [4, 6, 8]]),
            selected_node_mask=torch.tensor(
                [[True, True, True], [True, True, False]]
            ),
        )
        self.assertTrue(torch.equal(positions, torch.tensor([1, 1, 0])))

        flat_indices = torch.tensor([[0, 1], [2, -1]])
        mask = flat_indices.ge(0)
        padded = pad_proposal_node_positions(positions, flat_indices, mask)
        self.assertTrue(torch.equal(padded, torch.tensor([[1, 1], [0, 0]])))

    def test_hierarchical_ppo_update_changes_policy_parameters(self) -> None:
        torch.manual_seed(11)
        actor = HierarchicalPPOActorCritic(width=8, hidden_size=12)
        node_features = torch.randn(2, 3, 8)
        node_mask = torch.ones(2, 3, dtype=torch.bool)
        candidate_features = torch.randn(2, 4, actor.policy_feature_dim)
        matcher_logits = torch.randn(2, 4)
        candidate_nodes = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])
        candidate_mask = torch.ones(2, 4, dtype=torch.bool)
        prefix = torch.randn(2, 8)
        state = torch.randn(2, actor.state_feature_dim)
        actions = torch.tensor([0, 2])
        with torch.no_grad():
            policy = actor.policy(
                node_features,
                node_mask,
                candidate_features,
                matcher_logits,
                candidate_nodes,
                candidate_mask,
                prefix,
                state,
                include_stop=False,
            )
            distribution = torch.distributions.Categorical(logits=policy.log_probs)
            old_log_probs = distribution.log_prob(actions)
        transitions = []
        for index in range(2):
            transitions.append(
                HierarchicalPPOTransition(
                    state_features=state[index],
                    prefix_state=prefix[index],
                    node_features=node_features[index],
                    node_mask=node_mask[index],
                    candidate_features=candidate_features[index],
                    matcher_logits=matcher_logits[index],
                    candidate_nodes=candidate_nodes[index],
                    candidate_mask=candidate_mask[index],
                    action_index=int(actions[index]),
                    old_log_prob=float(old_log_probs[index]),
                    old_value=0.0,
                    reward=1.0 if index == 0 else -1.0,
                    done=True,
                    legal=index == 0,
                    xfer_id=index,
                    matcher_probability=0.5,
                    advantage=1.0 if index == 0 else -1.0,
                    return_value=1.0 if index == 0 else -1.0,
                )
            )
        before = actor.pattern[-1].weight.detach().clone()
        metrics = hierarchical_ppo_update(
            actor,
            torch.optim.Adam(actor.parameters(), lr=1e-3),
            transitions,
            device=torch.device("cpu"),
            epochs=1,
            minibatch_size=2,
            clip_epsilon=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            target_kl=0.0,
            max_grad_norm=1.0,
            seed=4,
        )
        self.assertEqual(metrics["batches"], 1)
        self.assertFalse(torch.equal(before, actor.pattern[-1].weight))


if __name__ == "__main__":
    unittest.main()
