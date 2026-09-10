from __future__ import annotations

from dataclasses import dataclass
import unittest

from search_widening import (
    _wilson_upper_bound,
    continuation_revisit_shadow_rows,
    select_widening_revisits,
)
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
    origin_continuation_score: float | None = None
    probe_level: int = 0


class ProgressiveWideningSelectionTest(unittest.TestCase):
    def test_wilson_upper_bound_rewards_uncertain_useful_yield(self) -> None:
        self.assertEqual(_wilson_upper_bound(0, 0), 1.0)
        self.assertGreater(_wilson_upper_bound(8, 10), _wilson_upper_bound(80, 100))
        self.assertGreater(_wilson_upper_bound(9, 10), _wilson_upper_bound(8, 10))

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

    def test_feedback_ucb_uses_useful_child_confidence_lane(self) -> None:
        rows = [
            State("small-sample", 10, 0, search_node_id=0),
            State("large-sample", 10, 0, search_node_id=1),
            State("known-gain", 10, 0, search_node_id=2),
        ]
        feedback = {
            0: SearchNodeStats(
                0, "a", 10, 1, attempted_actions=10, unique_children=8
            ),
            1: SearchNodeStats(
                1, "b", 10, 1, attempted_actions=100, unique_children=80
            ),
            2: SearchNodeStats(2, "c", 10, 1, best_descendant_gate=9),
        }
        result = select_widening_revisits(
            rows,
            slots=2,
            max_expansions=3,
            policy="feedback_ucb",
            feedback=feedback,
        )
        self.assertEqual(result.metrics["policy"], "feedback_ucb")
        self.assertEqual(result.metrics["lane_counts"]["useful_yield_ucb"], 1)
        self.assertEqual(result.indices[1], 0)

    def test_feedback_marginal_prefers_unspent_descendant_headroom(self) -> None:
        rows = [
            State("spent", 10, 0, search_node_id=0),
            State("headroom", 10, 0, search_node_id=1),
        ]
        feedback = {
            0: SearchNodeStats(0, "a", 10, 1, best_descendant_gate=6),
            1: SearchNodeStats(1, "b", 10, 1, best_descendant_gate=10),
        }
        result = select_widening_revisits(
            rows,
            slots=1,
            max_expansions=3,
            policy="feedback_marginal",
            feedback=feedback,
        )
        self.assertEqual(result.indices, [1])
        self.assertEqual(result.metrics["policy"], "feedback_marginal")

    def test_probe_halving_keeps_safety_and_promotes_recent_sibling_winners(self) -> None:
        rows = [
            State(str(index), 10, 0, search_node_id=index)
            for index in range(8)
        ]
        feedback = {}
        for index in range(8):
            feedback[index] = SearchNodeStats(
                index,
                f"id-{index}",
                10,
                1,
                origin_parent_id=0 if index < 4 else 1,
                observed_expansions=1,
                last_attempted_actions=16,
                last_valid_actions=16,
                last_unique_children=15 if index in {3, 7} else index % 4,
                last_improving_children=1 if index in {3, 7} else 0,
                last_best_child_gate=9 if index in {3, 7} else 10,
            )
        result = select_widening_revisits(
            rows,
            slots=4,
            max_expansions=3,
            policy="probe_halving",
            feedback=feedback,
        )
        self.assertEqual(result.metrics["policy"], "probe_halving")
        self.assertEqual(result.metrics["lane_counts"]["round_robin_safety"], 3)
        self.assertEqual(result.metrics["lane_counts"]["probe_promoted"], 1)
        self.assertEqual(result.probe_promoted_indices, (3,))
        self.assertIn(3, result.indices)

    def test_probe_halving_only_races_the_highest_persistent_level(self) -> None:
        rows = [
            State(str(index), 10, 0, search_node_id=index, probe_level=index // 2)
            for index in range(4)
        ]
        feedback = {
            index: SearchNodeStats(
                index,
                f"id-{index}",
                10,
                1,
                origin_parent_id=9,
                last_attempted_actions=16,
                last_unique_children=15 if index == 3 else 1,
            )
            for index in range(4)
        }
        result = select_widening_revisits(
            rows,
            slots=4,
            max_expansions=3,
            policy="probe_halving",
            feedback=feedback,
        )
        self.assertEqual(result.probe_promoted_indices, (3,))

    def test_balanced_feedback_round_robins_sibling_cohorts(self) -> None:
        rows = [
            State(str(index), 10, 0, search_node_id=index)
            for index in range(6)
        ]
        feedback = {
            index: SearchNodeStats(
                index,
                f"id-{index}",
                10,
                1,
                origin_parent_id=0 if index < 3 else 1,
                observed_expansions=(2 if index in {0, 3} else 0),
                best_descendant_gate=10,
            )
            for index in range(6)
        }
        result = select_widening_revisits(
            rows,
            slots=4,
            max_expansions=3,
            policy="feedback_balanced",
            feedback=feedback,
        )
        self.assertEqual(set(result.indices), {1, 2, 4, 5})
        self.assertEqual(result.metrics["lane_counts"], {"balanced_sibling": 4})
        self.assertEqual(result.metrics["eligible_sibling_groups"], 2)

    def test_continuation_revisit_shadow_does_not_change_feedback_selection(self) -> None:
        rows = [
            State(
                str(index),
                10,
                0,
                search_node_id=index,
                origin_continuation_score=float(index),
            )
            for index in range(4)
        ]
        feedback = {
            index: SearchNodeStats(
                index,
                f"id-{index}",
                10,
                1,
                observed_expansions=index,
                best_descendant_gate=10,
            )
            for index in range(4)
        }
        actual = select_widening_revisits(
            rows,
            slots=2,
            max_expansions=4,
            policy="feedback",
            feedback=feedback,
        )
        before = list(actual.indices)
        shadow = continuation_revisit_shadow_rows(
            rows,
            max_expansions=4,
            slots=2,
            step=3,
            feedback=feedback,
            actually_selected=actual.indices,
        )
        self.assertEqual(actual.indices, before)
        self.assertEqual(
            [row["node_id"] for row in shadow if row["shadow_selected"]],
            [3, 2],
        )
        self.assertTrue(all(row["selection_step"] == 3 for row in shadow))


if __name__ == "__main__":
    unittest.main()
