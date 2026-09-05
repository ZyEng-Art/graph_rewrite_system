from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def adopt_exact_candidate(
    archive: dict,
    candidate_qasm: Path,
    candidate_gate_count: int,
    round_index: int,
    archive_dir: Path,
) -> tuple[bool, Path]:
    previous = archive.get("best_gate_count")
    improved = previous is None or candidate_gate_count < int(previous)
    if improved:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_qasm = archive_dir / (
            f"best_g{candidate_gate_count}_round_{round_index:04d}.qasm"
        )
        shutil.copy2(candidate_qasm, archived_qasm)
        archive["best_gate_count"] = candidate_gate_count
        archive["best_qasm"] = str(archived_qasm.resolve())
        archive["best_round"] = round_index
        archive.setdefault("improvements", []).append(
            {
                "round": round_index,
                "gate_count": candidate_gate_count,
                "qasm": str(archived_qasm.resolve()),
            }
        )
        return True, archived_qasm
    return False, Path(archive["best_qasm"])


def build_rollout_command(
    args: argparse.Namespace,
    root_qasm: Path,
    result_path: Path,
    candidate_qasm: Path,
    final_histories_path: Path,
    refresh_histories_dir: Path,
    invocation_index: int,
) -> list[str]:
    command = [
        str(args.python),
        str(args.rollout_script),
        "--data",
        str(args.data),
        "--checkpoint",
        str(args.checkpoint),
        "--calibration",
        str(args.calibration),
        "--target-recall",
        str(args.target_recall),
        "--ecc-file",
        str(args.ecc_file),
        "--qasm",
        str(root_qasm),
        "--beam-size",
        str(args.beam_size),
        "--depth",
        str(args.depth * args.rounds),
        "--microbatch",
        str(args.microbatch),
        "--page-size",
        str(args.page_size),
        "--readout-attention-backend",
        args.readout_attention_backend,
        "--state-batch-backend",
        "tensorized",
        "--proposal-backend",
        "gpu",
        "--proposal-ranking",
        args.proposal_ranking,
        "--proposal-ranking-seed",
        str(args.proposal_ranking_seed + invocation_index * args.depth * args.rounds),
        "--max-source-matches",
        str(args.max_source_matches),
        "--max-actions-per-parent",
        str(args.max_actions_per_parent),
        "--proposal-factor",
        str(args.proposal_factor),
        "--max-gate-increase",
        str(args.max_gate_increase),
        "--lazy-topology-backend",
        "indexed",
        "--refresh-interval",
        str(args.refresh_interval),
        "--refresh-factor",
        str(args.refresh_factor),
        "--dedup-mode",
        args.dedup_mode,
        "--audit-count",
        str(args.audit_count),
        "--output",
        str(result_path),
        "--best-qasm",
        str(candidate_qasm),
        "--dump-beam-histories",
        str(final_histories_path),
        "--dump-refresh-histories-dir",
        str(refresh_histories_dir),
        "--restart-from-best-at-refresh",
        "--best-root-restart-interval",
        str(args.depth),
    ]
    if args.stop_after_stale_refreshes:
        command.extend(
            [
                "--stop-after-stale-refreshes",
                str(args.stop_after_stale_refreshes),
            ]
        )
    if args.proposal_ranking == "value":
        command.extend(
            [
                "--action-value-weight",
                str(args.action_value_weight),
                "--value-increase-actions-per-parent",
                str(args.value_increase_actions_per_parent),
                "--value-exploration-fraction",
                str(args.value_exploration_fraction),
            ]
        )
    if args.exploration_checkpoint is not None:
        command.extend(
            [
                "--exploration-checkpoint",
                str(args.exploration_checkpoint),
                "--exploration-calibration",
                str(args.exploration_calibration),
                "--exploration-actions-per-parent",
                str(args.exploration_actions_per_parent),
                "--exploration-until-depth",
                str(args.exploration_until_depth),
            ]
        )
    if args.profile_stages:
        command.append("--profile-stages")
    return command


def new_archive(args: argparse.Namespace, copied_input: Path) -> dict:
    return {
        "format": "accelerated-self-improve-v1",
        "input_qasm": str(Path(args.qasm).resolve()),
        "copied_input_qasm": str(copied_input.resolve()),
        "initial_gate_count": None,
        "best_gate_count": None,
        "best_qasm": str(copied_input.resolve()),
        "best_round": -1,
        "improvements": [],
        "rounds": [],
        "runs": [],
        "next_round": 0,
        "config": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "proposal_ranking": args.proposal_ranking,
            "action_value_weight": args.action_value_weight,
            "value_increase_actions_per_parent": (
                args.value_increase_actions_per_parent
            ),
            "value_exploration_fraction": args.value_exploration_fraction,
            "beam_size": args.beam_size,
            "depth_per_round": args.depth,
            "rounds_per_invocation": args.rounds,
            "refresh_interval": args.refresh_interval,
            "refresh_factor": args.refresh_factor,
            "max_gate_increase": args.max_gate_increase,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated paged searches, persist the exact best circuit, and "
            "relocate the next search root only after a Quartz-confirmed improvement."
        )
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--rollout-script",
        type=Path,
        default=Path(__file__).with_name("paged_rollout_benchmark.py"),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--exploration-checkpoint", type=Path)
    parser.add_argument("--exploration-calibration", type=Path)
    parser.add_argument("--exploration-actions-per-parent", type=int, default=0)
    parser.add_argument("--exploration-until-depth", type=int, default=0)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument(
        "--stop-after-stale-refreshes",
        "--stop-after-stale",
        dest="stop_after_stale_refreshes",
        type=int,
        default=0,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--beam-size", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument(
        "--readout-attention-backend",
        choices=("eager", "sdpa", "paged"),
        default="paged",
    )
    parser.add_argument(
        "--proposal-ranking",
        choices=("gate", "probability", "stochastic", "value"),
        default="stochastic",
    )
    parser.add_argument("--action-value-weight", type=float, default=0.25)
    parser.add_argument("--value-increase-actions-per-parent", type=int, default=16)
    parser.add_argument("--value-exploration-fraction", type=float, default=0.0)
    parser.add_argument("--proposal-ranking-seed", type=int, default=73)
    parser.add_argument("--max-source-matches", type=int, default=2048)
    parser.add_argument("--max-actions-per-parent", type=int, default=128)
    parser.add_argument("--proposal-factor", type=int, default=16)
    parser.add_argument("--max-gate-increase", type=int, default=1)
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument("--refresh-factor", type=int, default=2)
    parser.add_argument("--dedup-mode", choices=("none", "raw"), default="raw")
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--profile-stages", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.stop_after_stale_refreshes < 0:
        parser.error("--stop-after-stale must be nonnegative")
    if args.refresh_interval < 1:
        parser.error("an exact archive requires --refresh-interval >= 1")
    if args.depth % args.refresh_interval:
        parser.error("--depth must end on a Quartz refresh boundary")
    if args.proposal_ranking == "value" and args.action_value_weight <= 0:
        parser.error("value ranking requires --action-value-weight > 0")
    if not 0 <= args.value_increase_actions_per_parent <= args.max_actions_per_parent:
        parser.error("value increase action quota must be within the parent cap")
    if not 0.0 <= args.value_exploration_fraction <= 1.0:
        parser.error("value exploration fraction must be within [0, 1]")
    if args.value_exploration_fraction and args.proposal_ranking != "value":
        parser.error("value exploration fraction requires value ranking")
    if (args.exploration_checkpoint is None) != (
        args.exploration_calibration is None
    ):
        parser.error(
            "--exploration-checkpoint and --exploration-calibration are paired"
        )
    if args.exploration_actions_per_parent and args.exploration_checkpoint is None:
        parser.error("an exploration quota requires an exploration checkpoint")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    archive_path = output_dir / "archive.json"
    runs_dir = output_dir / "runs"
    roots_dir = output_dir / "roots"
    exact_dir = output_dir / "exact_archive"
    roots_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"archive already exists: {archive_path}; pass --resume to continue"
            )
        archive = json.loads(archive_path.read_text())
        archive.setdefault("runs", [])
        root_qasm = Path(archive["best_qasm"])
        start_round = int(archive["next_round"])
    else:
        copied_input = roots_dir / Path(args.qasm).name
        shutil.copy2(args.qasm, copied_input)
        archive = new_archive(args, copied_input)
        root_qasm = copied_input
        start_round = 0
        atomic_write_json(archive_path, archive)

    invocation_index = len(archive["runs"])
    run_dir = runs_dir / f"run_{invocation_index:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "rollout.json"
    candidate_qasm = run_dir / "best_exact.qasm"
    final_histories_path = run_dir / "final_beam_histories.json"
    refresh_histories_dir = run_dir / "refresh_histories"
    log_path = run_dir / "rollout.log"
    command = build_rollout_command(
        args,
        root_qasm,
        result_path,
        candidate_qasm,
        final_histories_path,
        refresh_histories_dir,
        invocation_index,
    )
    started = time.perf_counter()
    with log_path.open("w") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    wall_seconds = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"resident rollout failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )

    result = json.loads(result_path.read_text())
    candidate_gate_count = int(result["best_exact_gate_count"])
    root_gate_count = int(result["initial_gate_count"])
    if archive["best_gate_count"] is None:
        archive["initial_gate_count"] = root_gate_count
        archive["best_gate_count"] = root_gate_count
    if root_gate_count != int(archive["best_gate_count"]):
        raise RuntimeError(
            "rollout root differs from archived best: "
            f"root={root_gate_count}, archive={archive['best_gate_count']}"
        )
    best_depth = int(result["best_exact_depth"])
    best_local_round = max(0, (best_depth - 1) // args.depth)
    improved, root_qasm = adopt_exact_candidate(
        archive,
        candidate_qasm,
        candidate_gate_count,
        start_round + best_local_round,
        exact_dir,
    )

    refresh_rows = [
        row for row in result["steps"] if bool(row["segment_completed"])
    ]
    if not refresh_rows:
        raise RuntimeError("resident rollout completed without an exact refresh")
    trace_best = root_gate_count
    for offset, row in enumerate(refresh_rows):
        round_best = int(row["best_exact_gate_count_so_far"])
        round_improved = round_best < trace_best
        trace_best = min(trace_best, round_best)
        round_record = {
            "round": start_round + offset,
            "root_gate_count": int(row["segment_root_gate_count"]),
            "round_best_exact_gate_count": round_best,
            "best_gate_count_so_far": round_best,
            "improved": round_improved,
            "stale_refreshes": int(row["stale_refreshes"]),
            "refresh_history": row["refresh_history"],
            "search_step": int(row["step"]),
            "restarted_from_best": bool(row["restarted_from_best"]),
        }
        archive["rounds"].append(round_record)
        print(json.dumps(round_record, sort_keys=True), flush=True)

    archive["next_round"] = start_round + len(refresh_rows)
    archive["stale_refreshes"] = int(refresh_rows[-1]["stale_refreshes"])
    run_record = {
        "run": invocation_index,
        "start_round": start_round,
        "completed_rounds": len(refresh_rows),
        "root_gate_count": root_gate_count,
        "best_gate_count": candidate_gate_count,
        "improved": improved,
        "result": str(result_path.resolve()),
        "best_qasm": str(candidate_qasm.resolve()),
        "refresh_histories_dir": str(refresh_histories_dir.resolve()),
        "final_histories": str(final_histories_path.resolve()),
        "log": str(log_path.resolve()),
        "search_seconds": float(result["search_seconds_excluding_audit"]),
        "wall_seconds": wall_seconds,
        "model_loads": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "proposal_ranking": args.proposal_ranking,
        "proposal_ranking_seed": args.proposal_ranking_seed,
        "action_value_weight": args.action_value_weight,
        "value_increase_actions_per_parent": args.value_increase_actions_per_parent,
        "value_exploration_fraction": args.value_exploration_fraction,
        "max_gate_increase": args.max_gate_increase,
        "command": command,
    }
    archive["runs"].append(run_record)
    atomic_write_json(archive_path, archive)

    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "initial_gate_count": archive["initial_gate_count"],
                "best_gate_count": archive["best_gate_count"],
                "best_qasm": archive["best_qasm"],
                "completed_rounds": len(archive["rounds"]),
                "runs": len(archive["runs"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
