import unittest

import torch

from dataset import apply_snapshot_delta, rebase_prefix_sample
from train import matcher_refresh_consistency_loss


class RefreshRebaseTest(unittest.TestCase):
    def test_rebase_preserves_current_graph_and_target(self) -> None:
        first_delta = {
            "removed_slots": [0],
            "added_nodes": [(2, 6, 12)],
            "removed_edges": [(0, 1, 0, 0)],
            "added_edges": [(2, 1, 0, 0)],
        }
        second_delta = {
            "removed_slots": [1],
            "added_nodes": [(3, 5, 13)],
            "removed_edges": [(2, 1, 0, 0)],
            "added_edges": [(2, 3, 0, 0)],
        }
        actions = [
            {
                "xfer_id": 1,
                "source_id": 1,
                "binding_slots": (0,),
                "dst_slots": (2,),
                "effective_delta": first_delta,
            },
            {
                "xfer_id": 2,
                "source_id": 2,
                "binding_slots": (1,),
                "dst_slots": (3,),
                "effective_delta": second_delta,
            },
        ]
        target = {"source_id": 7, "binding_slots": (2, 3)}
        sample = {
            "initial_graph": {
                "nodes": [(0, 6, 10), (1, 5, 11)],
                "edges": [(0, 1, 0, 0)],
            },
            "actions": actions,
            "matches": [(7, (2, 3))],
            "target_action": target,
            "local_streak": 0,
            "trajectory_id": 4,
            "prefix_length": 2,
            "previous_action": actions[-1],
            "previous_delta": second_delta,
            "previous_local_streak": 0,
        }

        rebased = rebase_prefix_sample(sample, 1)
        original_current = sample["initial_graph"]
        for action in actions:
            original_current = apply_snapshot_delta(
                original_current, action["effective_delta"]
            )
        rebased_current = apply_snapshot_delta(
            rebased["initial_graph"], rebased["actions"][0]["effective_delta"]
        )

        self.assertEqual(rebased_current, original_current)
        self.assertEqual(rebased["target_action"], target)
        self.assertEqual(rebased["matches"], sample["matches"])
        self.assertEqual(rebased["prefix_length"], 1)
        self.assertEqual(rebased["actions"], actions[-1:])


class RefreshConsistencyLossTest(unittest.TestCase):
    def test_identical_views_have_zero_loss(self) -> None:
        logits = torch.tensor([[[1.0, -2.0], [3.0, 0.5]]])
        eligible = torch.ones_like(logits, dtype=torch.bool)
        loss = matcher_refresh_consistency_loss(
            logits,
            logits.clone(),
            eligible,
            eligible,
            [[(0, (0,))]],
            max_hard_pairs=2,
        )
        self.assertEqual(float(loss.detach()), 0.0)

    def test_difference_produces_gradients_in_both_views(self) -> None:
        base = torch.tensor(
            [[[1.0, -2.0], [3.0, 0.5]]], requires_grad=True
        )
        refreshed = torch.tensor(
            [[[-1.0, -2.0], [0.0, 0.5]]], requires_grad=True
        )
        eligible = torch.ones_like(base, dtype=torch.bool)
        loss = matcher_refresh_consistency_loss(
            base,
            refreshed,
            eligible,
            eligible,
            [[(0, (0,))]],
            max_hard_pairs=2,
        )
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(float(base.grad.abs().sum().detach()), 0.0)
        self.assertGreater(float(refreshed.grad.abs().sum().detach()), 0.0)

    def test_rejects_misaligned_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical matcher shapes"):
            matcher_refresh_consistency_loss(
                torch.zeros((1, 2, 3)),
                torch.zeros((1, 3, 3)),
                torch.ones((1, 2, 3), dtype=torch.bool),
                torch.ones((1, 3, 3), dtype=torch.bool),
                [[]],
                max_hard_pairs=1,
            )


if __name__ == "__main__":
    unittest.main()
