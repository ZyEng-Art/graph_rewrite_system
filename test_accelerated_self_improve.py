from argparse import Namespace
import json
from pathlib import Path
import tempfile

from accelerated_self_improve import adopt_exact_candidate, build_rollout_command
from collect_action_preferences import (
    archive_history_paths,
    trajectory_future_residuals,
    unique_paths,
)


def test_archive_is_monotonic() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidate = root / "candidate.qasm"
        candidate.write_text("OPENQASM 2.0;\n")
        archive = {"best_gate_count": None, "best_qasm": None, "improvements": []}

        improved, best = adopt_exact_candidate(archive, candidate, 63, 0, root)
        assert improved and best.exists()
        assert archive["best_gate_count"] == 63

        improved, unchanged = adopt_exact_candidate(archive, candidate, 65, 1, root)
        assert not improved and unchanged == best
        assert archive["best_gate_count"] == 63

        improved, best = adopt_exact_candidate(archive, candidate, 62, 2, root)
        assert improved and best.exists()
        assert archive["best_gate_count"] == 62
        assert [row["gate_count"] for row in archive["improvements"]] == [63, 62]


def test_resident_rounds_share_one_rollout_process() -> None:
    args = Namespace(
        python=Path("python"),
        rollout_script=Path("paged_rollout_benchmark.py"),
        data=Path("data.pt"),
        checkpoint=Path("model.pt"),
        calibration=Path("calibration.json"),
        target_recall=0.95,
        ecc_file=Path("ecc.json"),
        beam_size=1000,
        depth=16,
        rounds=3,
        stop_after_stale_refreshes=2,
        microbatch=512,
        page_size=8,
        readout_attention_backend="paged",
        proposal_ranking="stochastic",
        proposal_ranking_seed=73,
        max_source_matches=2048,
        max_actions_per_parent=128,
        proposal_factor=16,
        max_gate_increase=1,
        refresh_interval=8,
        refresh_factor=2,
        dedup_mode="raw",
        audit_count=64,
        action_value_weight=0.25,
        value_increase_actions_per_parent=16,
        value_exploration_fraction=0.0,
        exploration_checkpoint=None,
        exploration_calibration=None,
        exploration_actions_per_parent=0,
        exploration_until_depth=0,
        profile_stages=False,
    )
    command = build_rollout_command(
        args,
        Path("root.qasm"),
        Path("result.json"),
        Path("best.qasm"),
        Path("final_histories.json"),
        Path("refresh_histories"),
        4,
    )
    seed_index = command.index("--proposal-ranking-seed") + 1
    qasm_index = command.index("--qasm") + 1
    depth_index = command.index("--depth") + 1
    assert command[seed_index] == "265"
    assert command[qasm_index] == "root.qasm"
    assert command[depth_index] == "48"
    assert command.count("--restart-from-best-at-refresh") == 1
    restart_index = command.index("--best-root-restart-interval") + 1
    assert command[restart_index] == "16"
    assert command.count("--stop-after-stale-refreshes") == 1


def test_archive_exposes_completed_segment_histories() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first.json"
        second = root / "second.json"
        archive_path = root / "archive.json"
        archive_path.write_text(
            json.dumps(
                {
                    "format": "accelerated-self-improve-v1",
                    "rounds": [
                        {"refresh_history": str(first)},
                        {"refresh_history": str(second)},
                    ],
                }
            )
        )
        histories = archive_history_paths([archive_path])
        assert histories == [first, second]
        assert unique_paths(histories + [first]) == histories


def test_best_prefix_target_ignores_forced_final_regression() -> None:
    history = ((0,), (1,), (2,), (3,))
    deltas = (2, -1, -3, 4)
    assert trajectory_future_residuals(
        10, history, deltas, "final-residual"
    ) == [0, 1, 4, 0]
    assert trajectory_future_residuals(
        10, history, deltas, "best-prefix-residual"
    ) == [-4, -3, 0, 0]


if __name__ == "__main__":
    test_archive_is_monotonic()
    test_resident_rounds_share_one_rollout_process()
    test_archive_exposes_completed_segment_histories()
    test_best_prefix_target_ignores_forced_final_regression()
