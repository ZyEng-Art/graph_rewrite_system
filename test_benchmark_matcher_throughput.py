from types import SimpleNamespace
import unittest

from benchmark_matcher_throughput import InitialGraphDataset, snapshot_qasm_graph
from beam_search_benchmark import BeamState, collate_matcher_states


class RawQasmBenchmarkTest(unittest.TestCase):
    def test_qasm_snapshot_and_action_free_dataset(self) -> None:
        graph = SimpleNamespace(
            nodes=[
                SimpleNamespace(guid=101, gate_tp=7),
                SimpleNamespace(guid=305, gate_tp=9),
            ],
            all_edges=lambda: [(0, 1, 2, 3)],
        )

        snapshot = snapshot_qasm_graph(graph)
        self.assertEqual(
            snapshot,
            {
                "nodes": [(0, 7, 101), (1, 9, 305)],
                "edges": [(0, 1, 2, 3)],
            },
        )

        dataset = InitialGraphDataset(snapshot)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["initial_graph"], snapshot)
        self.assertEqual(dataset[0]["actions"], [])
        self.assertEqual(dataset[0]["matches"], [])
        with self.assertRaises(IndexError):
            dataset[1]

    def test_paged_matcher_rebases_exact_current_graph(self) -> None:
        state = BeamState(
            graph=None,
            snapshot={"nodes": [(0, 7, 101)], "edges": []},
            guid_to_slot={},
            next_slot=1,
            last_touched={},
            rewrite_distance={0: 5},
            previous_preferred=set(),
            local_streak=0,
            gate_count=1,
            depth=4,
            history=(),
        )

        batch = collate_matcher_states([state], paged_action=True)
        self.assertEqual(batch["initial_types"].tolist(), [[7]])
        self.assertEqual(batch["current_types"].tolist(), [[7]])
        self.assertEqual(batch["action_xfers"].shape, (1, 0))
        self.assertEqual(batch["binding_slots"].shape, (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
