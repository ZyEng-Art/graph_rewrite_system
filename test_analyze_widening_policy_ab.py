from __future__ import annotations

import unittest

from analyze_widening_policy_ab import SUMMARY_FIELDS, compare


def row(gate: int, step: int, seconds: float, digest: str) -> dict:
    values = {
        "best_gate_count": gate,
        "best_first_seen_step": step,
        "best_first_seen_seconds": seconds / 2,
        "best_action_depth": step - 1,
        "unique_graphs_seen": 100,
        "total_seconds": seconds,
        "best_graph_exact_identity_digest": digest,
    }
    assert all(field in values for field in SUMMARY_FIELDS)
    return values


class AnalyzeWideningPolicyAbTest(unittest.TestCase):
    def test_compare_uses_lower_gate_count_as_a_win(self) -> None:
        result = compare(
            {"a": row(40, 10, 4.0, "x"), "b": row(38, 8, 2.0, "z")},
            {"a": row(38, 9, 3.0, "y"), "b": row(38, 8, 2.0, "z")},
        )
        self.assertEqual(
            result["gate_count_wins_ties_losses"],
            {"wins": 1, "ties": 1, "losses": 0},
        )
        self.assertEqual(result["means"]["best_gate_count"]["delta"], -1.0)
        self.assertFalse(result["rows"][0]["best_graph_digest_same"])

    def test_compare_rejects_different_circuit_sets(self) -> None:
        with self.assertRaises(ValueError):
            compare({"a": row(1, 1, 1.0, "x")}, {"b": row(1, 1, 1.0, "x")})


if __name__ == "__main__":
    unittest.main()
