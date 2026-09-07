from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from dataset import RuleMetadata
from ppo_core import HierarchicalPPOActorCritic
from train_hierarchical_action import (
    WeightedActionPreferenceDataset,
    collate_action_preferences,
    gate_deltas,
    pair_scores,
)


class HierarchicalActionTrainingTest(unittest.TestCase):
    @staticmethod
    def preference_row(rejected_xfer: int = 1) -> dict:
        return {
            "initial_graph": {"nodes": [(0, 0, 0)], "edges": []},
            "actions": [],
            "preferred": {
                "xfer_id": 0,
                "source_id": 0,
                "anchor_slot": 0,
                "binding_slots": (0,),
            },
            "rejected": {
                "xfer_id": rejected_xfer,
                "source_id": 0,
                "anchor_slot": 0,
                "binding_slots": (0,),
            },
            "advantage": 1,
            "future_best_reduction": 20,
            "circuit": "test",
            "source_history": "/tmp/test/path",
            "prefix_length": 0,
        }

    def test_collate_encodes_a_repeated_state_once(self) -> None:
        rules = RuleMetadata(
            source_patterns=("x 0",),
            source_gate_types=((0,),),
            destination_gate_types=((0,), (0,)),
            xfer_to_source=(0, 0),
            xfer_sources=("x 0", "x 0"),
            xfer_destinations=("x 0", "x 0"),
        )
        dataset = WeightedActionPreferenceDataset(
            [self.preference_row(), self.preference_row(0)],
            return_weight=0.5,
            max_sample_weight=4.0,
        )
        batch = collate_action_preferences([dataset[0], dataset[1]], rules)
        self.assertEqual(batch["initial_types"].shape[0], 1)
        self.assertTrue(
            torch.equal(batch["preference_state_inverse"], torch.tensor([0, 0]))
        )
        self.assertEqual(batch["preferred_xfers"].shape[0], 2)

    def test_gate_deltas_use_each_transfer_source(self) -> None:
        rules = SimpleNamespace(
            xfer_to_source=(1, 0, 1),
            source_gate_types=((0, 1), (2, 3, 4)),
            destination_gate_types=((5,), (6, 7, 8), (9, 10, 11, 12)),
        )
        self.assertTrue(
            torch.equal(gate_deltas(rules), torch.tensor([-2.0, 1.0, 1.0]))
        )

    def test_return_weight_is_bounded(self) -> None:
        dataset = WeightedActionPreferenceDataset(
            [self.preference_row()], return_weight=0.5, max_sample_weight=4.0
        )
        self.assertEqual(dataset[0]["action_sample_weight"], 4.0)

    def test_pair_scores_include_history_context(self) -> None:
        actor = HierarchicalPPOActorCritic(width=4, hidden_size=4)
        with torch.no_grad():
            actor.pattern[0].weight.zero_()
            actor.pattern[0].bias.zero_()
            actor.pattern[2].weight.fill_(1.0)
            actor.pattern[2].bias.zero_()
            actor.node_prefix_projection.weight.copy_(torch.eye(4))
            actor.node_state_projection.weight.zero_()
        feature = torch.zeros(2, actor.policy_feature_dim)
        prefixes = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
        )
        states = torch.zeros(2, actor.state_feature_dim)
        preferred, rejected = pair_scores(
            actor,
            feature,
            torch.ones(2),
            feature,
            torch.zeros(2),
            prefixes,
            states,
        )
        self.assertTrue(torch.allclose(preferred - rejected, torch.ones(2)))
        self.assertGreater(float(preferred[1]), float(preferred[0]))


if __name__ == "__main__":
    unittest.main()
