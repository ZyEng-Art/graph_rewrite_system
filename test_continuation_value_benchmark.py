from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from build_continuation_manifest import (
    TrajectorySpec,
    build_manifest,
    future_labels,
    read_trajectory,
    uniform_indices,
)
from continuation_value_benchmark import (
    Job,
    aggregate_runs,
    group_budget_aggregates,
    job_directory,
    normalize_result,
    ranking_evaluations,
    truncate_result_payload,
    validate_manifest,
    validate_runner_args,
)
from beam_search_benchmark import rank_proposals


class ContinuationManifestTest(unittest.TestCase):
    def make_trajectory(self, root: Path, costs=(10, 11, 9, 8)) -> Path:
        trajectory = root / "trajectory"
        trajectory.mkdir()
        for step, cost in enumerate(costs):
            node = 0 if step + 1 == len(costs) else step + 3
            xfer = 0 if step + 1 == len(costs) else 100 + step
            (trajectory / f"{step}_{cost}_0_{node}_{xfer}.qasm").write_text(
                f"// state {step}\n", encoding="utf-8"
            )
        return trajectory

    def test_uniform_indices_are_deterministic_and_include_endpoints(self) -> None:
        self.assertEqual(uniform_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(uniform_indices(3, 8), [0, 1, 2])
        self.assertEqual(uniform_indices(5, 1), [2])

    def test_future_labels_preserve_plateau_before_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            states = read_trajectory(self.make_trajectory(Path(temporary)))
            labels = future_labels(states, 0, [1, 2, 8])
        self.assertEqual(labels["1"]["improvement"], 0)
        self.assertEqual(labels["2"]["improvement"], 1)
        self.assertEqual(labels["2"]["first_improvement_step"], 2)
        self.assertEqual(labels["8"]["improvement"], 2)
        self.assertEqual(labels["8"]["observed_steps"], 3)

    def test_manifest_has_digests_and_excludes_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trajectory = self.make_trajectory(Path(temporary))
            payload = build_manifest(
                [TrajectorySpec("source", "circuit", "positive", trajectory)],
                states_per_trajectory=10,
                horizons=[2],
            )
            validated = validate_manifest(payload)
        self.assertEqual(len(validated), 3)
        self.assertEqual([row["trajectory_step"] for row in validated], [0, 1, 2])
        self.assertTrue(all(len(row["qasm_sha256"]) == 64 for row in validated))


class ContinuationRunnerTest(unittest.TestCase):
    @staticmethod
    def state(state_id="state/one") -> dict:
        return {
            "id": state_id,
            "circuit": "demo",
            "kind": "positive",
            "source_id": "source",
            "trajectory_step": 4,
            "qasm": "/tmp/demo.qasm",
            "initial_gate_count": 10,
            "teacher_future": {"8": {"improvement": 2}},
        }

    def test_job_directory_sanitizes_state_id(self) -> None:
        job = Job(self.state(), budget=8, seed=73, gpu="0")
        path = job_directory(Path("root"), job)
        self.assertEqual(
            path,
            Path("root/jobs/state_one/budget-0008/seed-73"),
        )

    def test_controlled_runner_arguments_cannot_be_overridden(self) -> None:
        with self.assertRaises(ValueError):
            validate_runner_args(["--beam-size", "16", "--depth", "99"])

    def test_stochastic_proposal_ranking_is_seeded_and_changes_selection(self) -> None:
        state = type("State", (), {"gate_count": 10})()
        action_rows = [
            [(index, index, (index,), 0.5) for index in range(20)]
        ]
        gate_deltas = [0] * 20

        def selected(seed: int) -> list[int]:
            proposals, total = rank_proposals(
                [state],
                action_rows,
                gate_deltas,
                beam_size=1,
                max_actions_per_parent=3,
                proposal_factor=3,
                max_gate_increase=0,
                ranking_mode="stochastic",
                ranking_seed=seed,
            )
            self.assertEqual(total, 20)
            return [proposal.xfer_id for proposal in proposals]

        self.assertEqual(selected(73), selected(73))
        self.assertNotEqual(selected(73), selected(74))

    def test_paged_result_is_normalized_with_search_diagnostics(self) -> None:
        job = Job(self.state("s0"), budget=8, seed=73, gpu=None)
        payload = {
            "initial_gate_count": 10,
            "best_exact_gate_count": 8,
            "completed_depth": 2,
            "search_seconds_excluding_audit": 1.5,
            "steps": [
                {
                    "predicted_actions": 100,
                    "attempted_actions": 20,
                    "accepted_actions": 12,
                    "invalid_structural_actions": 3,
                    "duplicate_speculative_successors": 5,
                    "exact_refresh_exact_duplicates": 2,
                },
                {
                    "predicted_actions": 80,
                    "attempted_actions": 10,
                    "accepted_actions": 8,
                    "invalid_structural_actions": 1,
                    "duplicate_speculative_successors": 1,
                    "exact_refresh_exact_duplicates": 0,
                },
            ],
        }
        row = normalize_result(job, payload, wall_seconds=2.0)
        self.assertEqual(row["improvement"], 2)
        self.assertEqual(row["predicted_actions"], 180)
        self.assertEqual(row["attempted_actions"], 30)
        self.assertEqual(row["accepted_actions"], 20)
        self.assertAlmostEqual(row["unique_accept_rate"], 2 / 3)
        self.assertAlmostEqual(row["invalid_rate"], 4 / 30)
        self.assertAlmostEqual(row["duplicate_rate"], 8 / 32)

    def test_beam_result_field_aliases_are_normalized(self) -> None:
        job = Job(self.state("s1"), budget=4, seed=73, gpu=None)
        payload = {
            "initial_gate_count": 10,
            "best_gate_count": 9,
            "steps": [
                {
                    "predicted_or_exact_actions": 40,
                    "attempted_actions": 20,
                    "accepted_actions": 8,
                    "invalid_model_actions": 5,
                    "duplicate_successors": 7,
                }
            ],
        }
        row = normalize_result(job, payload, wall_seconds=1.0)
        self.assertEqual(row["predicted_actions"], 40)
        self.assertEqual(row["invalid_actions"], 5)
        self.assertEqual(row["speculative_duplicates"], 7)
        self.assertAlmostEqual(row["invalid_rate"], 0.25)
        self.assertAlmostEqual(row["duplicate_rate"], 0.35)

    def test_long_run_can_supply_exact_shorter_budget_prefixes(self) -> None:
        payload = {
            "initial_gate_count": 10,
            "best_gate_count": 5,
            "completed_depth": 4,
            "steps": [
                {
                    "global_best_gate_count": 10,
                    "attempted_actions": 10,
                    "accepted_actions": 5,
                    "cumulative_seconds": 0.2,
                },
                {
                    "global_best_gate_count": 9,
                    "attempted_actions": 20,
                    "accepted_actions": 8,
                    "cumulative_seconds": 0.5,
                },
                {
                    "global_best_gate_count": 7,
                    "attempted_actions": 30,
                    "accepted_actions": 9,
                    "cumulative_seconds": 0.9,
                },
                {
                    "global_best_gate_count": 5,
                    "attempted_actions": 40,
                    "accepted_actions": 10,
                    "cumulative_seconds": 1.4,
                },
            ],
        }
        prefix = truncate_result_payload(payload, 2)
        self.assertEqual(prefix["completed_depth"], 2)
        self.assertEqual(prefix["best_gate_count"], 9)
        self.assertEqual(prefix["search_seconds_excluding_audit"], 0.5)
        job = Job(self.state("prefix"), budget=2, seed=73, gpu=None)
        normalized = normalize_result(job, prefix, wall_seconds=70.0)
        self.assertEqual(normalized["improvement"], 1)
        self.assertEqual(normalized["attempted_actions"], 30)

    def test_aggregation_and_probe_ranking_report_lift(self) -> None:
        runs = []
        for state_id, probe, target in (
            ("good", 2, 4),
            ("medium", 1, 2),
            ("bad-a", 0, 0),
            ("bad-b", 0, 0),
        ):
            for budget, improvement in ((8, probe), (64, target)):
                runs.append(
                    {
                        "status": "completed",
                        "state_id": state_id,
                        "circuit": "demo",
                        "kind": "mixed",
                        "source_id": "source",
                        "trajectory_step": 0,
                        "budget": budget,
                        "seed": 1,
                        "initial_gate_count": 10,
                        "best_gate_count": 10 - improvement,
                        "improvement": improvement,
                        "search_seconds": 1.0,
                        "wall_seconds": 1.0,
                        "unique_accept_rate": 0.5,
                        "invalid_rate": 0.1,
                        "duplicate_rate": 0.2,
                        "attempted_actions": 20,
                        "teacher_future": {},
                    }
                )
        aggregates = aggregate_runs(runs)
        grouped = group_budget_aggregates(aggregates)
        self.assertEqual(grouped[0]["states"], 4)
        self.assertEqual(grouped[0]["improvement_success_rate"], 0.5)
        rankings = ranking_evaluations(aggregates, top_fractions=(0.25,))
        selected = next(
            row
            for row in rankings
            if row["score"] == "improvement_mean"
            and row["target_budget"] == 64
        )
        self.assertEqual(selected["selected"], 1)
        self.assertEqual(selected["selected_successes"], 1)
        self.assertAlmostEqual(selected["lift_over_random"], 2.0)


if __name__ == "__main__":
    unittest.main()
