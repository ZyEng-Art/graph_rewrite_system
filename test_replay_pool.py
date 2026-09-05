from __future__ import annotations

import random
import unittest

from train_paged_ppo import make_replay_bucket, retain_replay_state


class FakeGraph:
    def __init__(self, graph_hash: int, gate_count: int) -> None:
        self._hash = graph_hash
        self.gate_count = gate_count

    def hash(self) -> int:
        return self._hash

    def to_qasm_str(self) -> str:
        return f"// graph {self._hash} gates {self.gate_count}"


class ReplayPoolTest(unittest.TestCase):
    def test_reservoir_never_evicts_its_lowest_gate_state(self) -> None:
        random.seed(12)
        bucket = make_replay_bucket(FakeGraph(1, 10))
        retain_replay_state(bucket, FakeGraph(2, 12), capacity=3)
        retain_replay_state(bucket, FakeGraph(3, 11), capacity=3)
        retain_replay_state(bucket, FakeGraph(4, 9), capacity=3)
        self.assertIn(4, bucket["retained_hashes"])
        for graph_hash in range(5, 200):
            retain_replay_state(
                bucket,
                FakeGraph(graph_hash, 10 + graph_hash % 7),
                capacity=3,
            )
        self.assertIn(4, bucket["retained_hashes"])
        self.assertEqual(
            min(row["gate_count"] for row in bucket["states"]), 9
        )

    def test_new_record_low_replaces_a_worse_state(self) -> None:
        bucket = make_replay_bucket(FakeGraph(1, 10))
        retain_replay_state(bucket, FakeGraph(2, 11), capacity=2)
        retained = retain_replay_state(bucket, FakeGraph(3, 8), capacity=2)
        self.assertTrue(retained)
        self.assertEqual(
            sorted(row["gate_count"] for row in bucket["states"]), [8, 10]
        )


if __name__ == "__main__":
    unittest.main()
