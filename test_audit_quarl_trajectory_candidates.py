from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from audit_quarl_trajectory_candidates import (
    parse_trajectory_directory,
    summarize_actions,
)


class QuarlCandidateAuditTest(unittest.TestCase):
    def test_parse_trajectory_directory_orders_numeric_steps(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for name in (
                "2_8_0_0_0.qasm",
                "0_10_-1_4_20.qasm",
                "1_9_1_3_21.qasm",
            ):
                (path / name).write_text(name)

            rows = parse_trajectory_directory(path)
            self.assertEqual([row.step for row in rows], [0, 1])
            self.assertEqual(rows[0].cost, 10)
            self.assertEqual(rows[0].reward, -1)
            self.assertEqual(rows[0].node_id, 4)
            self.assertEqual(rows[0].xfer_id, 20)
            self.assertEqual(rows[0].next_qasm.name, "1_9_1_3_21.qasm")

    def test_parse_trajectory_directory_rejects_noncontiguous_steps(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "0_10_0_1_2.qasm").write_text("")
            (path / "2_9_0_0_0.qasm").write_text("")
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                parse_trajectory_directory(path)

    def test_summarize_actions_distinguishes_candidate_caps(self) -> None:
        rows = [
            {
                "step": 0,
                "gate_rank": 3,
                "exact_action_available": True,
                "exact_transition_matches": True,
                "source_match_covered": True,
            },
            {
                "step": 1,
                "gate_rank": 100,
                "exact_action_available": True,
                "exact_transition_matches": True,
                "source_match_covered": True,
            },
            {
                "step": 2,
                "gate_rank": None,
                "exact_action_available": True,
                "exact_transition_matches": True,
                "source_match_covered": False,
            },
        ]

        summary = summarize_actions(rows, [64, 128])
        self.assertEqual(summary["caps"]["64"]["covered"], 1)
        self.assertEqual(summary["caps"]["64"]["missing_steps"], [1, 2])
        self.assertEqual(summary["caps"]["128"]["covered"], 2)
        self.assertFalse(summary["caps"]["128"]["whole_trajectory_covered"])
        self.assertEqual(summary["gate_rank"]["max"], 100)


if __name__ == "__main__":
    unittest.main()
