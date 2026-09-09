from __future__ import annotations

import unittest

import torch

from build_sibling_continuation_preferences import collect_sibling_pairs


class BuildSiblingContinuationPreferencesTest(unittest.TestCase):
    def test_pairs_use_unique_children_and_best_descendant(self) -> None:
        payload = {
            "format": "frozen_candidate_successor_descendant_v2",
            "features": torch.zeros(5, 3),
            "outcomes": torch.tensor([2, 2, 2, 2, 0], dtype=torch.int8),
            "sibling_group_ids": torch.tensor([7, 7, 7, 7, 8]),
            "parent_node_ids": torch.tensor([7, 7, 7, 7, 8]),
            "child_node_ids": torch.tensor([10, 11, 11, 12, -1]),
            "action_parent_ranks": torch.tensor([8, 3, 9, 4, 0]),
            "parent_expansion_rounds": torch.zeros(5, dtype=torch.int16),
            "parent_stagnation_steps": torch.zeros(5, dtype=torch.int32),
            "parent_action_depths": torch.zeros(5, dtype=torch.int32),
            "descendant_labels": {
                "best_descendant_gate_counts": torch.tensor([19, 17, 17, 20, -1]),
                "continuation_gains": torch.tensor([1, 3, 3, 0, 0]),
                "parent_total_gains": torch.tensor([1, 3, 3, 0, 0]),
                "time_to_observed_best_descendant": torch.tensor([2, 4, 4, -1, -1]),
                "right_censored": torch.tensor([False, False, False, True, False]),
                "remaining_search_steps": torch.tensor([20, 20, 20, 20, 20]),
                "child_observed_expansions": torch.tensor([2, 3, 3, 4, 0]),
                "child_attempted_actions": torch.tensor([32, 48, 48, 64, 0]),
            },
        }
        pairs, stats = collect_sibling_pairs(
            payload, min_remaining_steps=16, max_rejected_per_group=8
        )
        self.assertEqual(stats["duplicate_child_rows_removed"], 1)
        self.assertEqual(stats["groups_with_signal"], 1)
        self.assertEqual([row["preferred_row"] for row in pairs], [1, 1])
        self.assertEqual(
            {row["rejected_row"] for row in pairs}, {0, 3}
        )
        self.assertEqual(
            {row["advantage"] for row in pairs}, {2, 3}
        )

    def test_exposure_filter_rejects_unexplored_and_unbalanced_pairs(self) -> None:
        payload = {
            "format": "frozen_candidate_successor_descendant_v2",
            "features": torch.zeros(3, 2),
            "outcomes": torch.full((3,), 2, dtype=torch.int8),
            "sibling_group_ids": torch.tensor([4, 4, 4]),
            "parent_node_ids": torch.tensor([4, 4, 4]),
            "child_node_ids": torch.tensor([10, 11, 12]),
            "action_parent_ranks": torch.tensor([0, 1, 2]),
            "parent_expansion_rounds": torch.zeros(3),
            "parent_stagnation_steps": torch.zeros(3),
            "parent_action_depths": torch.zeros(3),
            "descendant_labels": {
                "best_descendant_gate_counts": torch.tensor([10, 12, 13]),
                "continuation_gains": torch.tensor([3, 1, 0]),
                "parent_total_gains": torch.tensor([3, 1, 0]),
                "time_to_observed_best_descendant": torch.tensor([2, 2, -1]),
                "right_censored": torch.tensor([False, False, True]),
                "remaining_search_steps": torch.tensor([20, 20, 20]),
                "child_observed_expansions": torch.tensor([2, 3, 0]),
                "child_attempted_actions": torch.tensor([32, 48, 0]),
            },
        }
        pairs, stats = collect_sibling_pairs(
            payload,
            min_remaining_steps=1,
            max_rejected_per_group=8,
            min_child_expansions=1,
            max_child_expansion_gap=1,
        )
        self.assertEqual(stats["pairs"], 1)
        self.assertEqual(pairs[0]["preferred_row"], 0)
        self.assertEqual(pairs[0]["rejected_row"], 1)


if __name__ == "__main__":
    unittest.main()
