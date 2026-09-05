from __future__ import annotations

import unittest

from dataset import (
    RuleMetadata,
    action_destination_types,
    action_with_effective_delta,
    replay_with_locality,
    validate_trajectory,
)
from incremental_graph import IncrementalCircuit


def rules() -> RuleMetadata:
    return RuleMetadata(
        source_patterns=("rz 0; rz 0;",),
        source_gate_types=((5, 5),),
        destination_gate_types=((8, 5),),
        xfer_to_source=(0,),
        xfer_sources=("rz 0; rz 0;",),
        xfer_destinations=("add; rz 0;",),
    )


def zero_rotation_step() -> dict:
    delta = {
        "removed_slots": [0, 1],
        "added_nodes": [],
        "removed_edges": [(0, 1, 0, 0)],
        "added_edges": [],
    }
    return {
        "index": 0,
        "local_streak": 0,
        "matches": [{"source_id": 0, "binding_slots": (0, 1)}],
        "action": {
            "xfer_id": 0,
            "source_id": 0,
            "anchor_slot": 0,
            "binding_slots": (0, 1),
            "dst_slots": (),
            "dst_types": (),
            "declared_dst_guids": (20, 21),
            "normalized_away_dst_guids": (20, 21),
        },
        "delta": delta,
    }


class NormalizedActionTest(unittest.TestCase):
    def test_zero_rotation_destination_can_contract_to_empty(self) -> None:
        step = zero_rotation_step()
        trajectory = {
            "trajectory_id": 0,
            "initial_graph": {
                "nodes": [(0, 5, 10), (1, 5, 11)],
                "edges": [(0, 1, 0, 0)],
            },
            "steps": [step],
        }
        validate_trajectory(trajectory, rules())
        self.assertEqual(action_destination_types(step["action"], rules()), ())

    def test_history_replay_uses_authoritative_effective_delta(self) -> None:
        step = zero_rotation_step()
        history_action = action_with_effective_delta(step)
        sample = {
            "initial_graph": {
                "nodes": [(0, 5, 10), (1, 5, 11), (2, 0, 12)],
                "edges": [(0, 1, 0, 0), (1, 2, 0, 0)],
            },
            "actions": [
                {
                    **history_action,
                    "effective_delta": {
                        **step["delta"],
                        "removed_edges": [(0, 1, 0, 0), (1, 2, 0, 0)],
                        "added_edges": [],
                    },
                }
            ],
        }
        circuit, _, _, _ = replay_with_locality(sample, rules())
        self.assertEqual(circuit.nodes, {2: 0})
        self.assertEqual(circuit.edges, set())

    def test_apply_delta_rejects_dangling_edges(self) -> None:
        circuit = IncrementalCircuit(
            {
                "nodes": [(0, 5, 10), (1, 0, 11)],
                "edges": [(0, 1, 0, 0)],
            }
        )
        with self.assertRaisesRegex(ValueError, "dead slot"):
            circuit.apply_delta(
                {
                    "removed_slots": [0],
                    "added_nodes": [],
                    "removed_edges": [],
                    "added_edges": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
