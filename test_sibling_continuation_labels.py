from __future__ import annotations

import unittest

import torch

from search_feedback import SearchFeedbackRegistry
from sibling_continuation_labels import descendant_label_tensors


class SiblingContinuationLabelsTest(unittest.TestCase):
    def test_labels_invalid_censored_and_delayed_gain(self) -> None:
        registry = SearchFeedbackRegistry("root", 20)
        first = registry.add_node(
            "first", gate_count=20, depth=1, parent_id=registry.root_id, step=1
        )
        second = registry.add_node(
            "second", gate_count=19, depth=1, parent_id=registry.root_id, step=1
        )
        registry.add_node("grandchild", gate_count=17, depth=2, parent_id=first, step=4)
        labels = descendant_label_tensors(
            registry,
            torch.tensor([-1, first, second]),
            torch.tensor([20, 20, 20]),
            torch.tensor([1, 1, 1]),
            observation_end_step=8,
        )
        self.assertEqual(labels["child_gate_counts"].tolist(), [-1, 20, 19])
        self.assertEqual(labels["best_descendant_gate_counts"].tolist(), [-1, 17, 19])
        self.assertEqual(labels["continuation_gains"].tolist(), [0, 3, 0])
        self.assertEqual(labels["parent_total_gains"].tolist(), [0, 3, 1])
        self.assertEqual(
            labels["time_to_observed_best_descendant"].tolist(), [-1, 3, -1]
        )
        self.assertEqual(labels["right_censored"].tolist(), [False, False, True])
        self.assertEqual(labels["remaining_search_steps"].tolist(), [7, 7, 7])


if __name__ == "__main__":
    unittest.main()
