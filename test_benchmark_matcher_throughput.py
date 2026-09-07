from types import SimpleNamespace
import unittest

import torch

from benchmark_matcher_throughput import InitialGraphDataset, snapshot_qasm_graph
from beam_search_benchmark import (
    BeamState,
    Proposal,
    apply_rewrite,
    collate_exact_states,
    collate_matcher_states,
    remap_candidate_slots,
    rank_proposals,
)
from threshold_inference import CandidateTensors


class FakeApplyGraph:
    def __init__(self) -> None:
        self.nodes = [
            SimpleNamespace(guid=101),
            SimpleNamespace(guid=305),
        ]
        self.direct_calls = []
        self.guid_direct_calls = []
        self.anchor_calls = []
        self.successor = SimpleNamespace()

    def apply_xfer_with_node_id_binding(self, **kwargs):
        self.direct_calls.append(kwargs)
        return self.successor, [], [101, 305], [900]

    def apply_xfer_with_guid_binding(self, **kwargs):
        self.guid_direct_calls.append(kwargs)
        return self.successor, [900]

    def get_node_from_id(self, *, id):
        return self.nodes[id]

    def apply_xfer_with_binding_trace(self, **kwargs):
        self.anchor_calls.append(kwargs)
        return self.successor, [], [101, 305], [901]


class RawQasmBenchmarkTest(unittest.TestCase):
    @staticmethod
    def apply_state(graph) -> BeamState:
        return BeamState(
            graph=graph,
            snapshot={
                "nodes": [(4, 7, 101), (9, 8, 305)],
                "edges": [],
            },
            guid_to_slot={101: 4, 305: 9},
            next_slot=10,
            last_touched={},
            rewrite_distance={},
            previous_preferred=set(),
            local_streak=0,
            gate_count=2,
            depth=0,
            history=(),
        )

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

    def test_state_only_collation_compacts_and_restores_persistent_slots(self) -> None:
        first = BeamState(
            graph=None,
            snapshot={
                "nodes": [(4, 7, 101), (9, 8, 305)],
                "edges": [(4, 9, 2, 3)],
            },
            guid_to_slot={101: 4, 305: 9},
            next_slot=10,
            last_touched={9: 3},
            rewrite_distance={4: 5, 9: 1},
            previous_preferred={9},
            local_streak=2,
            gate_count=2,
            depth=4,
            history=(),
        )
        second = BeamState(
            graph=None,
            snapshot={"nodes": [(12, 6, 700)], "edges": []},
            guid_to_slot={700: 12},
            next_slot=13,
            last_touched={},
            rewrite_distance={12: 4},
            previous_preferred=set(),
            local_streak=0,
            gate_count=1,
            depth=1,
            history=(),
        )

        batch, dense_to_slot, stats = collate_exact_states([first, second])

        self.assertTrue(
            {
                "initial_types",
                "action_xfers",
                "action_sources",
                "binding_slots",
                "destination_slots",
                "destination_types",
            }.isdisjoint(batch)
        )
        self.assertEqual(batch["current_types"].tolist(), [[7, 8], [6, -1]])
        self.assertEqual(dense_to_slot.tolist(), [[4, 9], [12, -1]])
        self.assertEqual(batch["current_edge_src"].tolist(), [0])
        self.assertEqual(batch["current_edge_dst"].tolist(), [1])
        self.assertEqual(batch["current_rewrite_distance"].tolist(), [[5, 1], [4, 5]])
        self.assertEqual(stats["padded_dense_slots"], 4)
        self.assertEqual(stats["padded_persistent_slots"], 26)

        candidates = CandidateTensors(
            batch_ids=torch.tensor([0, 1]),
            sources=torch.tensor([3, 4]),
            anchors=torch.tensor([1, 0]),
            bindings=torch.tensor([[0, 1, -1], [0, -1, -1]]),
            probabilities=torch.tensor([0.8, 0.7]),
        )
        restored = remap_candidate_slots(
            candidates,
            dense_to_slot,
            batch_offset=5,
        )
        self.assertEqual(restored.batch_ids.tolist(), [5, 6])
        self.assertEqual(restored.anchors.tolist(), [9, 12])
        self.assertEqual(restored.bindings.tolist(), [[4, 9, -1], [12, -1, -1]])

    def test_model_binding_uses_direct_quartz_api_when_available(self) -> None:
        graph = FakeApplyGraph()
        proposal = Proposal(
            parent=0,
            xfer_id=7,
            anchor_slot=4,
            binding=(4, 9),
            probability=0.8,
            next_gate_count=2,
        )

        applied = apply_rewrite(
            self.apply_state(graph),
            proposal,
            [None] * 8,
            eliminate_rotation=True,
        )

        self.assertIs(applied.graph, graph.successor)
        self.assertEqual(applied.source_guids, (101, 305))
        self.assertEqual(applied.destination_guids, (900,))
        self.assertEqual(
            graph.guid_direct_calls[0]["source_node_guids"], [101, 305]
        )
        self.assertTrue(graph.guid_direct_calls[0]["eliminate_rotation"])
        self.assertEqual(graph.direct_calls, [])
        self.assertEqual(graph.anchor_calls, [])

    def test_anchor_backend_remains_available_for_ab_and_cpu_actions(self) -> None:
        graph = FakeApplyGraph()
        proposal = Proposal(
            parent=0,
            xfer_id=7,
            anchor_slot=4,
            binding=(4, 9),
            probability=0.8,
            next_gate_count=2,
        )

        applied = apply_rewrite(
            self.apply_state(graph),
            proposal,
            [None] * 8,
            binding_backend="anchor",
        )

        self.assertIs(applied.graph, graph.successor)
        self.assertEqual(applied.destination_guids, (901,))
        self.assertEqual(len(graph.anchor_calls), 1)
        self.assertEqual(graph.direct_calls, [])
        self.assertEqual(graph.guid_direct_calls, [])

    def test_bounded_topk_preserves_full_stable_sort_order(self) -> None:
        beam = [
            SimpleNamespace(gate_count=10),
            SimpleNamespace(gate_count=12),
        ]
        action_rows = [
            [
                (0, 9, (9,), 0.6),
                (1, 8, (8,), 0.9),
                (2, 7, (7,), 0.2),
                (1, 6, (6,), 0.9),
                (3, 5, (5,), 1.0),
            ],
            [
                (2, 4, (4,), 0.8),
                (0, 3, (3,), 0.7),
                (1, 2, (2,), 0.5),
            ],
        ]
        gate_deltas = [0, -1, -2, 4]

        actual, eligible = rank_proposals(
            beam,
            action_rows,
            gate_deltas,
            beam_size=2,
            max_actions_per_parent=2,
            proposal_factor=2,
            max_gate_increase=3,
        )

        reference = []
        for parent_index, (state, rows) in enumerate(zip(beam, action_rows)):
            parent_rows = [
                Proposal(
                    parent=parent_index,
                    xfer_id=xfer_id,
                    anchor_slot=anchor,
                    binding=binding,
                    probability=probability,
                    next_gate_count=state.gate_count + gate_deltas[xfer_id],
                )
                for xfer_id, anchor, binding, probability in rows
                if gate_deltas[xfer_id] <= 3
            ]
            parent_rows.sort(
                key=lambda row: (
                    row.next_gate_count,
                    -row.probability,
                    row.xfer_id,
                )
            )
            reference.extend(parent_rows[:2])
        reference.sort(
            key=lambda row: (
                row.next_gate_count,
                -row.probability,
                beam[row.parent].gate_count,
            )
        )

        self.assertEqual(actual, reference[:4])
        self.assertEqual(eligible, 7)


if __name__ == "__main__":
    unittest.main()
