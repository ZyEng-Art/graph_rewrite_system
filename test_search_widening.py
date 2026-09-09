from __future__ import annotations

from dataclasses import dataclass
import unittest

from search_widening import select_widening_revisits
from search_feedback import SearchNodeStats


@dataclass
class State:
    name: str
    gate_count: int
    expansion_round: int
    history: tuple[tuple[int, int], ...] = ((0, 0),)
    last_xfer_id: int = 0
    last_source_slots: tuple[int, ...] = ()
    last_destination_slots: tuple[int, ...] = ()
    depth: int = 1
    path_best_gate_count: int = 10
    search_node_id: int = -1
    search_identity_order: str = ""


class ProgressiveWideningSelectionTest(unittest.TestCase):
    def test_prefers_lower_expansion_round_before_gate_count(self) -> None:
        rows = [
            State("third-visit-low-gate", 5, 2),
            State("first-revisit", 10, 0),
            State("second-revisit", 8, 1),
        ]
        result = select_widening_revisits(
            rows, slots=2, max_expansions=4, seed=7, step=3
        )
        self.assertEqual(result.indices, [1, 2])
        self.assertEqual(result.metrics["selected_next_rounds"], [1, 2])

    def test_excludes_states_after_last_rank_band(self) -> None:
        rows = [State("done", 5, 2), State("eligible", 6, 1)]
        result = select_widening_revisits(
            rows, slots=4, max_expansions=3
        )
        self.assertEqual(result.indices, [1])
        self.assertEqual(result.metrics["eligible_parents"], 1)

    def test_new_round_zero_states_do_not_starve_deeper_band(self) -> None:
        rows = [State(f"new-{index}", 10 + index, 0) for index in range(8)]
        rows.extend(
            [State("round-one", 20, 1), State("round-two", 30, 2)]
        )
        result = select_widening_revisits(
            rows, slots=3, max_expansions=5, seed=11, step=4
        )
        self.assertEqual(
            sorted(rows[index].expansion_round for index in result.indices),
            [0, 1, 2],
        )

    def test_zero_slots_is_empty(self) -> None:
        result = select_widening_revisits(
            [State("eligible", 5, 0)], slots=0, max_expansions=2
        )
        self.assertEqual(result.indices, [])
        self.assertEqual(result.metrics["selected_revisits"], 0)

    def test_selection_is_reproducible(self) -> None:
        left = [State(str(index), 10, 0) for index in range(8)]
        right = [State(str(index), 10, 0) for index in range(8)]
        a = select_widening_revisits(
            left, slots=4, max_expansions=3, seed=91, step=8
        )
        b = select_widening_revisits(
            right, slots=4, max_expansions=3, seed=91, step=8
        )
        self.assertEqual(a.indices, b.indices)

    def test_feedback_selection_does_not_depend_on_seed(self) -> None:
        rows = [
            State(
                str(index),
                10 + (index % 3),
                index % 2,
                search_node_id=index,
                search_identity_order=f"id-{index}",
            )
            for index in range(12)
        ]
        feedback = {
            index: SearchNodeStats(
                node_id=index,
                identity_order=f"id-{index}",
                gate_count=row.gate_count,
                depth=row.depth,
                valid_actions=10,
                unique_children=index,
                observed_expansions=row.expansion_round + 1,
                best_descendant_gate=row.gate_count - (index == 9),
            )
            for index, row in enumerate(rows)
        }
        a = select_widening_revisits(
            rows,
            slots=8,
            max_expansions=4,
            seed=7,
            step=5,
            policy="feedback",
            feedback=feedback,
        )
        b = select_widening_revisits(
            rows,
            slots=8,
            max_expansions=4,
            seed=999,
            step=5,
            policy="feedback",
            feedback=feedback,
        )
        self.assertEqual(a.indices, b.indices)
        self.assertEqual(a.metrics["policy"], "feedback")
        self.assertEqual(sum(a.metrics["lane_counts"].values()), 8)

    def test_feedback_lanes_cover_observed_gain_and_novelty(self) -> None:
        rows = [
            State("gain", 12, 0, search_node_id=0),
            State("novel", 10, 0, search_node_id=1),
            State("plain", 10, 0, search_node_id=2),
            State(
                "detour",
                12,
                0,
                depth=8,
                path_best_gate_count=10,
                search_node_id=3,
            ),
        ]
        feedback = {
            0: SearchNodeStats(0, "a", 12, 1, best_descendant_gate=9),
            1: SearchNodeStats(
                1,
                "b",
                10,
                1,
                valid_actions=10,
                unique_children=10,
                best_descendant_gate=10,
            ),
            2: SearchNodeStats(2, "c", 10, 1, best_descendant_gate=10),
            3: SearchNodeStats(3, "d", 12, 8, best_descendant_gate=12),
        }
        result = select_widening_revisits(
            rows,
            slots=4,
            max_expansions=3,
            policy="feedback",
            feedback=feedback,
        )
        self.assertEqual(set(result.indices), {0, 1, 2, 3})
        self.assertEqual(
            result.metrics["lane_counts"],
            {
                "improvement": 1,
                "novelty": 1,
                "exploration": 1,
                "detour_depth": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
