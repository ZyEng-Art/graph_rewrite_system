from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import threading
import time
from typing import Any, Iterable


FORBIDDEN_RUNNER_ARGS = {
    "--qasm",
    "--depth",
    "--output",
    "--best-qasm",
    "--proposal-ranking-seed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected a non-empty comma-separated integer list")
    return sorted(set(values))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any, lock: threading.Lock) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(payload: dict, *, check_files: bool = True) -> list[dict]:
    if payload.get("format") != "continuation-value-manifest-v1":
        raise ValueError("unsupported continuation manifest format")
    states = payload.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("manifest must contain a non-empty states list")
    ids = []
    for row in states:
        state_id = row.get("id")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("every manifest state needs a non-empty id")
        ids.append(state_id)
        qasm = Path(row.get("qasm", ""))
        if check_files:
            if not qasm.is_file():
                raise ValueError(f"missing QASM for {state_id}: {qasm}")
            expected_digest = row.get("qasm_sha256")
            if expected_digest and file_sha256(qasm) != expected_digest:
                raise ValueError(f"QASM digest mismatch for {state_id}: {qasm}")
        if int(row.get("initial_gate_count", -1)) < 0:
            raise ValueError(f"invalid initial_gate_count for {state_id}")
    if len(ids) != len(set(ids)):
        raise ValueError("manifest state ids are not unique")
    return states


def validate_runner_args(values: Iterable[str]) -> list[str]:
    result = list(map(str, values))
    for value in result:
        if value in FORBIDDEN_RUNNER_ARGS:
            raise ValueError(
                f"runner argument {value} is controlled by the continuation harness"
            )
    return result


def safe_component(value: str) -> str:
    rendered = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    ).strip("._")
    if not rendered:
        raise ValueError(f"cannot form a safe path component from {value!r}")
    return rendered


@dataclass(frozen=True)
class Job:
    state: dict
    budget: int
    seed: int
    gpu: str | None

    @property
    def key(self) -> str:
        return f"{self.state['id']}|b{self.budget}|s{self.seed}"


def job_directory(root: Path, job: Job) -> Path:
    return (
        root
        / "jobs"
        / safe_component(job.state["id"])
        / f"budget-{job.budget:04d}"
        / f"seed-{job.seed}"
    )


def sum_step_field(payload: dict, name: str) -> int:
    return sum(int(row.get(name, 0)) for row in payload.get("steps", []))


def sum_step_aliases(payload: dict, *names: str) -> int:
    total = 0
    for row in payload.get("steps", []):
        total += int(next((row[name] for name in names if name in row), 0))
    return total


def normalize_result(job: Job, payload: dict, wall_seconds: float) -> dict:
    initial = int(payload.get("initial_gate_count", job.state["initial_gate_count"]))
    best = payload.get("best_exact_gate_count")
    if best is None:
        best = payload.get("best_gate_count")
    if best is None:
        step_best = [
            row.get("best_exact_gate_count_so_far", row.get("best_gate_count"))
            for row in payload.get("steps", [])
        ]
        step_best = [int(value) for value in step_best if value is not None]
        best = min(step_best, default=initial)
    best = int(best)
    attempted = sum_step_field(payload, "attempted_actions")
    accepted = sum_step_field(payload, "accepted_actions")
    invalid = sum_step_aliases(
        payload, "invalid_structural_actions", "invalid_model_actions"
    )
    speculative_duplicates = sum_step_aliases(
        payload, "duplicate_speculative_successors", "duplicate_successors"
    )
    exact_duplicates = sum_step_field(payload, "exact_refresh_exact_duplicates")
    predicted = sum_step_aliases(
        payload, "predicted_actions", "predicted_or_exact_actions"
    )
    completed_depth = int(payload.get("completed_depth", len(payload.get("steps", []))))
    search_seconds = float(
        payload.get(
            "search_seconds_excluding_audit",
            sum(float(row.get("total_seconds", 0.0)) for row in payload.get("steps", [])),
        )
    )
    return {
        "format": "continuation-value-run-v1",
        "state_id": job.state["id"],
        "circuit": job.state.get("circuit"),
        "kind": job.state.get("kind"),
        "source_id": job.state.get("source_id"),
        "trajectory_step": job.state.get("trajectory_step"),
        "budget": job.budget,
        "seed": job.seed,
        "gpu": job.gpu,
        "initial_gate_count": initial,
        "best_gate_count": best,
        "improvement": max(0, initial - best),
        "completed_depth": completed_depth,
        "search_seconds": search_seconds,
        "wall_seconds": wall_seconds,
        "predicted_actions": predicted,
        "attempted_actions": attempted,
        "accepted_actions": accepted,
        "invalid_actions": invalid,
        "speculative_duplicates": speculative_duplicates,
        "exact_refresh_duplicates": exact_duplicates,
        "unique_accept_rate": accepted / max(1, attempted),
        "invalid_rate": invalid / max(1, attempted),
        "duplicate_rate": (
            speculative_duplicates + exact_duplicates
        )
        / max(1, attempted + exact_duplicates),
        "teacher_future": job.state.get("teacher_future", {}),
    }


def truncate_result_payload(payload: dict, budget: int) -> dict:
    """Derive an exact prefix result from one longer beam-search invocation."""

    steps = list(payload.get("steps", []))[:budget]
    truncated = dict(payload)
    truncated["steps"] = steps
    truncated["completed_depth"] = len(steps)
    initial = int(payload["initial_gate_count"])
    best_values = [initial]
    for row in steps:
        for name in (
            "best_exact_gate_count_so_far",
            "global_best_gate_count",
            "best_gate_count",
        ):
            if row.get(name) is not None:
                best_values.append(int(row[name]))
                break
    best = min(best_values)
    truncated.pop("best_exact_gate_count", None)
    truncated["best_gate_count"] = best
    if steps and steps[-1].get("cumulative_seconds") is not None:
        truncated["search_seconds_excluding_audit"] = float(
            steps[-1]["cumulative_seconds"]
        )
    else:
        truncated["search_seconds_excluding_audit"] = sum(
            float(row.get("total_seconds", 0.0)) for row in steps
        )
    return truncated


def run_job(
    job: Job,
    *,
    output_root: Path,
    python_bin: Path,
    runner_script: Path,
    runner_args: list[str],
    workdir: Path,
    base_environment: dict[str, str],
    timeout_seconds: float | None,
    resume: bool,
    events_path: Path,
    event_lock: threading.Lock,
) -> dict:
    directory = job_directory(output_root, job)
    directory.mkdir(parents=True, exist_ok=True)
    normalized_path = directory / "normalized.json"
    result_path = directory / "result.json"
    best_qasm_path = directory / "best.qasm"
    runner_log_path = directory / "runner.log"
    job_path = directory / "job.json"
    if resume and result_path.is_file():
        old_metadata = load_json(job_path) if job_path.is_file() else {}
        wall_seconds = float(old_metadata.get("wall_seconds", 0.0))
        payload = normalize_result(job, load_json(result_path), wall_seconds)
        payload.update(
            {
                "status": "completed",
                "result": str(result_path),
                "runner_log": str(runner_log_path),
            }
        )
        atomic_json(normalized_path, payload)
        append_jsonl(
            events_path,
            {"event": "job_renormalized", "time": utc_now(), "job": job.key},
            event_lock,
        )
        return payload

    command = [
        str(python_bin),
        str(runner_script),
        *runner_args,
        "--qasm",
        str(job.state["qasm"]),
        "--depth",
        str(job.budget),
        "--proposal-ranking-seed",
        str(job.seed),
        "--output",
        str(result_path),
        "--best-qasm",
        str(best_qasm_path),
    ]
    environment = dict(base_environment)
    if job.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = job.gpu
    job_metadata = {
        "format": "continuation-value-job-v1",
        "job": job.key,
        "state": job.state,
        "budget": job.budget,
        "seed": job.seed,
        "gpu": job.gpu,
        "command": command,
        "workdir": str(workdir),
        "runner_log": str(runner_log_path),
        "started_at": utc_now(),
    }
    atomic_json(job_path, job_metadata)
    append_jsonl(
        events_path,
        {"event": "job_started", "time": utc_now(), "job": job.key, "gpu": job.gpu},
        event_lock,
    )
    started = time.perf_counter()
    with runner_log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                command,
                cwd=workdir,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = 124
    wall_seconds = time.perf_counter() - started
    job_metadata.update(
        {
            "finished_at": utc_now(),
            "wall_seconds": wall_seconds,
            "return_code": return_code,
        }
    )
    atomic_json(job_path, job_metadata)
    if return_code != 0 or not result_path.is_file():
        failure = {
            "format": "continuation-value-run-v1",
            "state_id": job.state["id"],
            "budget": job.budget,
            "seed": job.seed,
            "gpu": job.gpu,
            "status": "failed",
            "return_code": return_code,
            "wall_seconds": wall_seconds,
            "runner_log": str(runner_log_path),
        }
        atomic_json(normalized_path, failure)
        append_jsonl(
            events_path,
            {"event": "job_failed", "time": utc_now(), "job": job.key, **failure},
            event_lock,
        )
        return failure
    normalized = normalize_result(job, load_json(result_path), wall_seconds)
    normalized["status"] = "completed"
    normalized["result"] = str(result_path)
    normalized["runner_log"] = str(runner_log_path)
    atomic_json(normalized_path, normalized)
    append_jsonl(
        events_path,
        {
            "event": "job_completed",
            "time": utc_now(),
            "job": job.key,
            "improvement": normalized["improvement"],
            "best_gate_count": normalized["best_gate_count"],
            "wall_seconds": wall_seconds,
        },
        event_lock,
    )
    return normalized


def run_job_guarded(job: Job, gpu_lock: threading.Lock | None, **kwargs) -> dict:
    """Prevent two subprocesses from being assigned to the same GPU concurrently."""

    if gpu_lock is None:
        return run_job(job, **kwargs)
    with gpu_lock:
        return run_job(job, **kwargs)


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / max(1, len(rows))


def aggregate_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in runs:
        if row.get("status") == "completed":
            grouped[(row["state_id"], int(row["budget"]))].append(row)
    output = []
    for (state_id, budget), rows in sorted(grouped.items()):
        first = rows[0]
        improvements = [int(row["improvement"]) for row in rows]
        output.append(
            {
                "state_id": state_id,
                "circuit": first.get("circuit"),
                "kind": first.get("kind"),
                "source_id": first.get("source_id"),
                "trajectory_step": first.get("trajectory_step"),
                "budget": budget,
                "seeds": len(rows),
                "initial_gate_count": int(first["initial_gate_count"]),
                "best_gate_count": min(int(row["best_gate_count"]) for row in rows),
                "improvement_mean": mean(improvements),
                "improvement_max": max(improvements),
                "improvement_success_rate": mean(value > 0 for value in improvements),
                "search_seconds_mean": mean(float(row["search_seconds"]) for row in rows),
                "wall_seconds_mean": mean(float(row["wall_seconds"]) for row in rows),
                "unique_accept_rate_mean": mean(
                    float(row["unique_accept_rate"]) for row in rows
                ),
                "invalid_rate_mean": mean(float(row["invalid_rate"]) for row in rows),
                "duplicate_rate_mean": mean(
                    float(row["duplicate_rate"]) for row in rows
                ),
                "attempted_actions_mean": mean(
                    int(row["attempted_actions"]) for row in rows
                ),
                "teacher_future": first.get("teacher_future", {}),
            }
        )
    return output


def ranking_evaluations(aggregates: list[dict], top_fractions=(0.1, 0.25, 0.5)) -> list[dict]:
    by_budget: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in aggregates:
        by_budget[int(row["budget"])][row["state_id"]] = row
    budgets = sorted(by_budget)
    if len(budgets) < 2:
        return []
    probe_budget = budgets[0]
    output = []
    score_fields = (
        "improvement_mean",
        "unique_accept_rate_mean",
        "attempted_actions_mean",
    )
    for target_budget in budgets[1:]:
        common = sorted(set(by_budget[probe_budget]) & set(by_budget[target_budget]))
        if not common:
            continue
        target_successes = sum(
            by_budget[target_budget][state_id]["improvement_max"] > 0
            for state_id in common
        )
        prevalence = target_successes / len(common)
        for score_field in score_fields:
            ranked = sorted(
                common,
                key=lambda state_id: (
                    by_budget[probe_budget][state_id][score_field],
                    -by_budget[probe_budget][state_id]["duplicate_rate_mean"],
                    state_id,
                ),
                reverse=True,
            )
            for fraction in top_fractions:
                selected_count = max(1, round(len(ranked) * fraction))
                selected = ranked[:selected_count]
                selected_successes = sum(
                    by_budget[target_budget][state_id]["improvement_max"] > 0
                    for state_id in selected
                )
                precision = selected_successes / selected_count
                recall = selected_successes / max(1, target_successes)
                output.append(
                    {
                        "probe_budget": probe_budget,
                        "target_budget": target_budget,
                        "score": score_field,
                        "top_fraction": fraction,
                        "states": len(common),
                        "selected": selected_count,
                        "target_successes": target_successes,
                        "selected_successes": selected_successes,
                        "target_success_prevalence": prevalence,
                        "precision": precision,
                        "recall": recall,
                        "lift_over_random": precision / prevalence if prevalence else None,
                        "target_improvement_mean": mean(
                            by_budget[target_budget][state_id]["improvement_mean"]
                            for state_id in selected
                        ),
                    }
                )
    return output


def group_budget_aggregates(aggregates: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in aggregates:
        grouped[(str(row.get("circuit")), str(row.get("kind")), int(row["budget"]))].append(row)
    output = []
    for (circuit, kind, budget), rows in sorted(grouped.items()):
        output.append(
            {
                "circuit": circuit,
                "kind": kind,
                "budget": budget,
                "states": len(rows),
                "improvement_success_rate": mean(
                    float(row["improvement_max"]) > 0 for row in rows
                ),
                "improvement_mean": mean(
                    float(row["improvement_mean"]) for row in rows
                ),
                "improvement_max": max(
                    float(row["improvement_max"]) for row in rows
                ),
                "duplicate_rate_mean": mean(
                    float(row["duplicate_rate_mean"]) for row in rows
                ),
                "invalid_rate_mean": mean(
                    float(row["invalid_rate_mean"]) for row in rows
                ),
                "unique_accept_rate_mean": mean(
                    float(row["unique_accept_rate_mean"]) for row in rows
                ),
                "search_seconds_mean": mean(
                    float(row["search_seconds_mean"]) for row in rows
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [key for key in rows[0] if key != "teacher_future"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_runner_config(path: Path) -> tuple[list[str], dict[str, str]]:
    payload = load_json(path)
    if payload.get("format") != "continuation-runner-config-v1":
        raise ValueError("unsupported runner config format")
    runner_args = validate_runner_args(payload.get("args", []))
    environment = {str(key): str(value) for key, value in payload.get("environment", {}).items()}
    return runner_args, environment


def git_revision(workdir: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_is_dirty(workdir: Path) -> bool | None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible fixed-budget continuations from QASM states."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runner-config", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="8,32,64")
    parser.add_argument(
        "--budget-mode",
        choices=("independent", "nested"),
        default="independent",
        help=(
            "nested runs only the largest depth and derives exact shorter beam prefixes; "
            "independent starts a fresh process for every budget"
        ),
    )
    parser.add_argument("--seeds", default="73")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    budgets = parse_int_csv(args.budgets)
    seeds = parse_int_csv(args.seeds)
    if any(value <= 0 for value in budgets):
        parser.error("--budgets must contain only positive integers")
    if any(value < 0 for value in seeds):
        parser.error("--seeds must contain only nonnegative integers")
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    manifest = load_json(args.manifest)
    states = validate_manifest(manifest, check_files=not args.dry_run)
    runner_args, configured_environment = load_runner_config(args.runner_config)
    workdir = args.workdir.resolve()
    runner_script = args.runner_script.resolve()
    python_bin = args.python_bin.resolve()
    output_root = args.output_dir.resolve()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    physical_budgets = [max(budgets)] if args.budget_mode == "nested" else budgets
    jobs = []
    ordinal = 0
    for state in states:
        for budget in physical_budgets:
            for seed in seeds:
                gpu = gpus[ordinal % len(gpus)] if gpus else None
                jobs.append(Job(state=state, budget=budget, seed=seed, gpu=gpu))
                ordinal += 1
    metadata = {
        "format": "continuation-value-benchmark-v1",
        "created_at": utc_now(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "git_revision": git_revision(workdir),
        "git_dirty": git_is_dirty(workdir),
        "harness_sha256": file_sha256(Path(__file__).resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "runner_config": str(args.runner_config.resolve()),
        "runner_config_sha256": file_sha256(args.runner_config),
        "runner_script": str(runner_script),
        "runner_script_sha256": file_sha256(runner_script),
        "python_bin": str(python_bin),
        "workdir": str(workdir),
        "output_dir": str(output_root),
        "budgets": budgets,
        "budget_mode": args.budget_mode,
        "physical_budgets": physical_budgets,
        "seeds": seeds,
        "gpus": gpus,
        "parallel_jobs": args.jobs,
        "timeout_seconds": args.timeout_seconds,
        "resume": args.resume,
        "physical_job_count": len(jobs),
        "logical_run_count": len(states) * len(budgets) * len(seeds),
        "states": len(states),
        "runner_args": runner_args,
        "configured_environment": configured_environment,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "metadata.json", metadata)
    if args.dry_run:
        preview = {
            **metadata,
            "jobs": [
                {
                    "key": job.key,
                    "qasm": job.state["qasm"],
                    "budget": job.budget,
                    "seed": job.seed,
                    "gpu": job.gpu,
                }
                for job in jobs
            ],
        }
        atomic_json(output_root / "dry_run.json", preview)
        print(json.dumps(preview, indent=2, sort_keys=True))
        return

    base_environment = dict(os.environ)
    base_environment.update(configured_environment)
    events_path = output_root / "events.jsonl"
    event_lock = threading.Lock()
    gpu_locks = {gpu: threading.Lock() for gpu in gpus}
    physical_runs = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                run_job_guarded,
                job,
                gpu_locks.get(job.gpu),
                output_root=output_root,
                python_bin=python_bin,
                runner_script=runner_script,
                runner_args=runner_args,
                workdir=workdir,
                base_environment=base_environment,
                timeout_seconds=args.timeout_seconds,
                resume=args.resume,
                events_path=events_path,
                event_lock=event_lock,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            row = future.result()
            physical_runs.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    runs = []
    if args.budget_mode == "nested":
        for physical in physical_runs:
            if physical.get("status") != "completed":
                for budget in budgets:
                    runs.append({**physical, "budget": budget, "physical_budget": max(budgets)})
                continue
            raw_payload = load_json(Path(physical["result"]))
            state = next(row for row in states if row["id"] == physical["state_id"])
            for budget in budgets:
                logical_job = Job(
                    state=state,
                    budget=budget,
                    seed=int(physical["seed"]),
                    gpu=physical.get("gpu"),
                )
                logical = normalize_result(
                    logical_job,
                    truncate_result_payload(raw_payload, budget),
                    float(physical["wall_seconds"]),
                )
                logical.update(
                    {
                        "status": "completed",
                        "result": physical["result"],
                        "runner_log": physical["runner_log"],
                        "physical_budget": max(budgets),
                        "shared_physical_run": True,
                    }
                )
                runs.append(logical)
    else:
        runs = physical_runs
    runs.sort(key=lambda row: (row["state_id"], row["budget"], row["seed"]))
    aggregates = aggregate_runs(runs)
    grouped_aggregates = group_budget_aggregates(aggregates)
    rankings = ranking_evaluations(aggregates)
    summary = {
        "format": "continuation-value-summary-v1",
        "created_at": utc_now(),
        "metadata": metadata,
        "completed_physical_jobs": sum(
            row.get("status") == "completed" for row in physical_runs
        ),
        "failed_physical_jobs": sum(
            row.get("status") != "completed" for row in physical_runs
        ),
        "completed_jobs": sum(row.get("status") == "completed" for row in runs),
        "failed_jobs": sum(row.get("status") != "completed" for row in runs),
        "runs": runs,
        "state_budget_aggregates": aggregates,
        "group_budget_aggregates": grouped_aggregates,
        "probe_ranking_evaluations": rankings,
    }
    atomic_json(output_root / "summary.json", summary)
    write_csv(output_root / "state_budget_aggregates.csv", aggregates)
    write_csv(output_root / "group_budget_aggregates.csv", grouped_aggregates)
    write_csv(output_root / "probe_ranking_evaluations.csv", rankings)
    print(
        json.dumps(
            {
                "summary": str((output_root / "summary.json").resolve()),
                "completed_jobs": summary["completed_jobs"],
                "failed_jobs": summary["failed_jobs"],
            },
            sort_keys=True,
        )
    )
    if summary["failed_physical_jobs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
