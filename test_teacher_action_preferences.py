from __future__ import annotations

import unittest

from collect_teacher_action_preferences import (
    base_trajectory_path,
    collect_preferences,
    hard_negative_actions,
    split_bucket,
)
from dataset import RuleMetadata


class TeacherActionPreferenceTest(unittest.TestCase):
    def test_windows_share_one_path_split(self):
        self.assertEqual(
            base_trajectory_path("/tmp/gf/370_2#window=64:128"),
            "/tmp/gf/370_2",
        )
        self.assertEqual(
            base_trajectory_path("/tmp/gf/370_2#segment=2#window=0:64"),
            "/tmp/gf/370_2",
        )

    def test_prefers_immediate_reduction_hard_negatives(self):
        rules = RuleMetadata(
            source_patterns=("", ""),
            source_gate_types=((0,), (1,)),
            destination_gate_types=((0, 0), (), (1,)),
            xfer_to_source=(0, 0, 1),
            xfer_sources=("", "", ""),
            xfer_destinations=("", "", ""),
        )
        step = {
            "action": {
                "xfer_id": 0,
                "source_id": 0,
                "anchor_slot": 4,
                "binding_slots": (4,),
            },
            "matches": [
                {
                    "source_id": 0,
                    "anchor_slot": 4,
                    "binding_slots": (4,),
                    "xfer_ids": [0, 1],
                },
                {
                    "source_id": 1,
                    "anchor_slot": 7,
                    "binding_slots": (7,),
                    "xfer_ids": [2],
                },
            ],
        }
        negatives = hard_negative_actions(step, rules, max_negatives=2)
        self.assertEqual([row["xfer_id"] for row in negatives], [1, 2])
        self.assertEqual(negatives[0]["binding_slots"], (4,))

    def test_can_force_and_emphasize_a_teacher_path(self):
        action = {
            "xfer_id": 0,
            "source_id": 0,
            "anchor_slot": 4,
            "binding_slots": (4,),
        }
        rejected = {
            "source_id": 0,
            "anchor_slot": 4,
            "binding_slots": (4,),
            "xfer_ids": [0, 1],
        }
        payload = {
            "source_patterns": ("x 0",),
            "xfer_to_source": (0, 0),
            "xfer_sources": ("x 0", "x 0"),
            "xfer_destinations": ("x 0;x 0", ""),
            "source_gate_types": ((0,),),
            "destination_gate_types": ((0, 0), ()),
            "train_trajectories": [
                {
                    "trajectory_id": 0,
                    "source_path": "/tmp/barenco_tof_3/38_3",
                    "initial_graph": {"gate_types": [0]},
                    "steps": [
                        {
                            "index": 0,
                            "action": action,
                            "matches": [rejected],
                        }
                    ],
                }
            ],
            "test_trajectories": [],
        }
        split_remainder = split_bucket("/tmp/barenco_tof_3/38_3", 2)
        train, test, metadata = collect_preferences(
            payload,
            max_negatives=1,
            test_modulo=2,
            test_remainder=split_remainder,
            force_train_suffixes=("barenco_tof_3/38_3",),
            emphasis_suffixes=("barenco_tof_3/38_3",),
            emphasis_repeat=3,
        )
        self.assertIn(split_remainder, (0, 1))
        self.assertEqual(len(train), 3)
        self.assertEqual(test, [])
        self.assertEqual(metadata["train_preferences"], 3)
        self.assertEqual(train[0]["teacher_action_gate_delta"], 1)
        self.assertEqual(train[0]["future_best_reduction"], -1)
        self.assertEqual(
            metadata["paths"]["/tmp/barenco_tof_3/38_3"]["split"],
            "train",
        )


if __name__ == "__main__":
    unittest.main()
