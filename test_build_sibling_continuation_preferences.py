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


if __name__ == "__main__":
    unittest.main()
