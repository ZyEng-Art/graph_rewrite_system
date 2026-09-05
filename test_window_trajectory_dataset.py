import unittest

from window_trajectory_dataset import apply_delta, window_trajectory


def make_step(index: int) -> dict:
    return {
        "index": index,
        "matches": [{"source_id": index, "binding_slots": (index,)}],
        "action": {
            "xfer_id": index,
            "source_id": index,
            "binding_slots": (index,),
            "dst_slots": (index + 10,),
        },
        "delta": {
            "removed_slots": [index],
            "added_nodes": [(index + 10, index + 1, index + 100)],
            "removed_edges": [],
            "added_edges": [],
        },
    }


class WindowTrajectoryDatasetTest(unittest.TestCase):
    def test_apply_delta_and_windows(self) -> None:
        trajectory = {
            "initial_graph": {
                "nodes": [(0, 0, 10), (1, 1, 11), (2, 2, 12)],
                "edges": [],
            },
            "steps": [make_step(0), make_step(1), make_step(2)],
            "terminal_matches": [{"source_id": 9, "binding_slots": (12,)}],
            "source_path": "sample",
        }
        windows = window_trajectory(trajectory, 2)
        self.assertEqual([len(window["steps"]) for window in windows], [2, 1])
        self.assertEqual(
            [row[0] for row in windows[1]["initial_graph"]["nodes"]],
            [2, 10, 11],
        )
        self.assertEqual(
            windows[0]["terminal_matches"], trajectory["steps"][2]["matches"]
        )
        self.assertEqual(
            windows[1]["terminal_matches"], trajectory["terminal_matches"]
        )
        self.assertEqual(
            [step["index"] for step in windows[0]["steps"]], [0, 1]
        )
        self.assertEqual([step["index"] for step in windows[1]["steps"]], [0])


if __name__ == "__main__":
    unittest.main()
