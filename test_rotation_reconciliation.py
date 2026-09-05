from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from beam_search_benchmark import BeamState
from hierarchical_ppo_rollout import reconcile_paged_topology_rows
from lazy_rollout_benchmark import LazyAction, indexed_topology
from train_paged_ppo import reconcile_rotation_normalized_state


class FakeNode:
    def __init__(self, guid: int, gate_type: int) -> None:
        self.guid = guid
        self.gate_tp = gate_type


class FakeGraph:
    def __init__(self, nodes: list[FakeNode], edges: list[tuple[int, ...]]) -> None:
        self.nodes = nodes
        self._edges = edges
        self.gate_count = len(nodes)

    def all_edges(self):
        return list(self._edges)


class RotationStateReconciliationTest(unittest.TestCase):
    def test_exact_rotation_contraction_replaces_static_lazy_topology(self) -> None:
        static_snapshot = {
            "nodes": [(2, 0, -1), (3, 5, -1), (4, 5, -1)],
            "edges": [(2, 3, 0, 0), (3, 4, 0, 0)],
        }
        static_topology = indexed_topology(static_snapshot)
        state = BeamState(
            graph=None,
            snapshot=None,
            guid_to_slot={},
            next_slot=5,
            last_touched={2: 0, 3: 0, 4: 0},
            rewrite_distance={2: 1, 3: 0, 4: 0},
            previous_preferred={2, 3, 4},
            local_streak=0,
            gate_count=3,
            depth=1,
            history=(LazyAction(7, (0, 1), (3, 4)),),
            topology_index=static_topology,
        )
        transition = SimpleNamespace(
            previous_gate_count=3,
            next_gate_count=3,
            reward=-0.02,
        )
        runtime = SimpleNamespace(
            state=state,
            transitions=[transition],
            pending_transition_indices=[0],
            topology_hashes={int(static_topology.fingerprint)},
            paged_topology_changed=False,
            final_gate_count=None,
        )
        exact_graph = FakeGraph([FakeNode(30, 0)], [])

        changed = reconcile_rotation_normalized_state(
            runtime,
            exact_graph,
            {30: 2},
            step_penalty=0.02,
        )

        self.assertTrue(changed)
        self.assertEqual(state.gate_count, 1)
        self.assertEqual(state.topology_index.nodes, {2: 0})
        self.assertEqual(state.previous_preferred, {2})
        self.assertEqual(state.last_touched, {2: 0})
        self.assertEqual(transition.next_gate_count, 1)
        self.assertAlmostEqual(transition.reward, 1.98)
        self.assertTrue(runtime.paged_topology_changed)
        self.assertEqual(runtime.final_gate_count, 1)
        self.assertIn(
            int(state.topology_index.fingerprint), runtime.topology_hashes
        )
        self.assertNotIn(int(static_topology.fingerprint), runtime.topology_hashes)

    def test_unchanged_gate_count_skips_exact_topology_materialization(self) -> None:
        state = SimpleNamespace(gate_count=2)
        runtime = SimpleNamespace(state=state)
        exact_graph = FakeGraph([FakeNode(20, 5), FakeNode(21, 5)], [])

        changed = reconcile_rotation_normalized_state(
            runtime,
            exact_graph,
            {},
            step_penalty=0.02,
        )

        self.assertFalse(changed)

    def test_periodic_audit_reconciles_equal_count_topology_mismatch(self) -> None:
        static_topology = indexed_topology(
            {
                "nodes": [(0, 5, -1), (1, 5, -1)],
                "edges": [(0, 1, 0, 0)],
            }
        )
        state = BeamState(
            graph=None,
            snapshot=None,
            guid_to_slot={},
            next_slot=2,
            last_touched={0: 0, 1: 0},
            rewrite_distance={0: 0, 1: 0},
            previous_preferred={0, 1},
            local_streak=0,
            gate_count=2,
            depth=1,
            history=(LazyAction(7, (0,), (1,)),),
            topology_index=static_topology,
        )
        transition = SimpleNamespace(
            previous_gate_count=2,
            next_gate_count=2,
            reward=-0.02,
        )
        runtime = SimpleNamespace(
            state=state,
            transitions=[transition],
            pending_transition_indices=[0],
            topology_hashes={int(static_topology.fingerprint)},
            paged_topology_changed=False,
            final_gate_count=None,
        )
        exact_graph = FakeGraph([FakeNode(20, 5), FakeNode(21, 5)], [])

        changed = reconcile_rotation_normalized_state(
            runtime,
            exact_graph,
            {20: 0, 21: 1},
            step_penalty=0.02,
            topology_mismatch=True,
        )

        self.assertTrue(changed)
        self.assertEqual(state.topology_index.edges, frozenset())
        self.assertTrue(runtime.paged_topology_changed)

    def test_paged_reconciliation_masks_only_changed_rows(self) -> None:
        exact_topology = indexed_topology(
            {
                "nodes": [(1, 5, -1), (3, 6, -1)],
                "edges": [(1, 3, 0, 0)],
            }
        )
        runtimes = [
            SimpleNamespace(
                paged_topology_changed=True,
                state=SimpleNamespace(next_slot=4, topology_index=exact_topology),
            ),
            SimpleNamespace(
                paged_topology_changed=False,
                state=SimpleNamespace(next_slot=4, topology_index=exact_topology),
            ),
        ]
        states = torch.ones(2, 4, 3)
        live = torch.ones(2, 4, dtype=torch.bool)
        gate_types = torch.zeros(2, 4, dtype=torch.long)

        states, live, gate_types, changed = reconcile_paged_topology_rows(
            states, live, gate_types, runtimes
        )

        self.assertEqual(changed, 1)
        self.assertEqual(live[0].tolist(), [False, True, False, True])
        self.assertEqual(gate_types[0].tolist(), [-1, 5, -1, 6])
        self.assertFalse(bool(states[0, 0].any()))
        self.assertTrue(bool(states[0, 1].all()))
        self.assertTrue(bool(live[1].all()))
        self.assertTrue(bool(states[1].all()))
        self.assertFalse(runtimes[0].paged_topology_changed)


if __name__ == "__main__":
    unittest.main()
