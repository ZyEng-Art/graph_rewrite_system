from __future__ import annotations

from dataclasses import dataclass, field
import unittest

from search_survivor import select_survivors


@dataclass
class State:
    gate_count: int
    name: str
    path_best_gate_count: int
    stagnation_steps: int
    last_xfer_id: int
    history: tuple[tuple[int, int], ...] = ((0, 0),)
    last_source_slots: tuple[int, ...] = ()
    last_destination_slots: tuple[int, ...] = ()
    survivor_lane: str = field(default="unselected")


class SurvivorSelectionTest(unittest.TestCase):
    def test_zero_fraction_matches_historical_gate_order(self) -> None:
        rows = [
            State(11, "c", 10, 1, 2, ((0, 0),)),
            State(9, "a", 9, 0, 1, ((0, 0), (1, 0))),
            State(9, "b", 9, 0, 3, ((0, 0),)),
        ]
        selected = select_survivors(rows, beam_size=2)
        self.assertEqual([row.name for row in selected.states], ["b", "a"])
        self.assertEqual(selected.metrics["policy"], "gate")

    def test_dual_lane_reserves_non_improving_state_outside_prefix(self) -> None:
        rows = [
            State(8, "best", 8, 0, 1),
            State(9, "second", 9, 0, 2),
            State(10, "third_improving", 10, 0, 3),
            State(11, "plateau", 11, 3, 7),
            State(12, "detour", 11, 2, 8),
            State(13, "too_far", 10, 1, 9),
        ]
        selected = select_survivors(
            rows,
            beam_size=3,
            exploration_fraction=1 / 3,
            exploration_max_detour=2,
        )
        self.assertEqual(
            {row.name for row in selected.states},
            {"best", "second", "plateau"},
        )
        self.assertEqual(selected.metrics["exploration_selected"], 1)
        self.assertEqual(
            next(row for row in selected.states if row.name == "plateau").survivor_lane,
            "exploration",
        )

    def test_shortage_falls_back_without_shrinking_beam(self) -> None:
        rows = [
            State(8, "a", 8, 0, 1),
            State(9, "b", 9, 0, 2),
            State(10, "c", 10, 0, 3),
            State(11, "d", 11, 0, 4),
        ]
        selected = select_survivors(
            rows, beam_size=4, exploration_fraction=0.5
        )
        self.assertEqual(len(selected.states), 4)
        self.assertEqual(selected.metrics["exploration_selected"], 0)
        self.assertEqual(selected.metrics["fallback_selected"], 2)

    def test_selection_is_reproducible(self) -> None:
        rows_a = [State(10, str(i), 10, 2, i % 2) for i in range(12)]
        rows_b = [State(10, str(i), 10, 2, i % 2) for i in range(12)]
        left = select_survivors(
            rows_a, beam_size=6, exploration_fraction=0.5, seed=91, step=4
        )
        right = select_survivors(
            rows_b, beam_size=6, exploration_fraction=0.5, seed=91, step=4
        )
        self.assertEqual(
            [row.name for row in left.states],
            [row.name for row in right.states],
        )


if __name__ == "__main__":
    unittest.main()
