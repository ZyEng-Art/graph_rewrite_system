from __future__ import annotations

import unittest

from train_hierarchical_node import LengthBucketBatchSampler, trajectory_gate_counts


class HierarchicalNodeTrainingTest(unittest.TestCase):
    def test_trajectory_gate_counts_follow_exact_deltas(self) -> None:
        trajectory = {
            "initial_graph": {"nodes": [(0, 0, 0), (1, 0, 1), (2, 0, 2)]},
            "steps": [
                {
                    "delta": {
                        "removed_slots": [0],
                        "added_nodes": [(3, 0, 3), (4, 0, 4)],
                    }
                },
                {
                    "delta": {
                        "removed_slots": [1, 2],
                        "added_nodes": [(5, 0, 5)],
                    }
                },
            ],
        }
        self.assertEqual(trajectory_gate_counts(trajectory), [3, 4, 3])

    def test_length_bucket_sampler_covers_each_state_once(self) -> None:
        lengths = [0, 1, 7, 8, 9, 15, 16]
        sampler = LengthBucketBatchSampler(
            lengths, batch_size=2, bucket_width=8, seed=5
        )
        batches = list(sampler)
        self.assertEqual(
            sorted(index for batch in batches for index in batch), list(range(7))
        )
        for batch in batches:
            selected = [lengths[index] for index in batch]
            self.assertLess(max(selected) - min(selected), 8)


if __name__ == "__main__":
    unittest.main()
