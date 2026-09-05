from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


COLLECT_RE = re.compile(r"Data for iter (\d+) collected in ([0-9.eE+-]+) s")
METRIC_RE = re.compile(r"^\s{4}([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$")
TIMING_RE = re.compile(
    r"Timing: rollout ([0-9.]+)s, learn ([0-9.]+)s, "
    r"iter ([0-9.]+)s, rollout/iter ([0-9.]+)"
)
IMPROVEMENT_RE = re.compile(
    r"Agent (\d+)\s*:\s*([^:]+):\s*(\d+)\s*->\s*(\d+)\s*!"
)


def parse_number(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_quarl_log(path: Path) -> dict:
    iterations: list[dict] = []
    improvements: list[dict] = []
    pending_improvements: list[dict] = []
    current: dict | None = None

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        improvement_match = IMPROVEMENT_RE.search(line)
        if improvement_match:
            pending_improvements.append(
                {
                    "line": line_number,
                    "agent": int(improvement_match.group(1)),
                    "circuit": improvement_match.group(2).strip(),
                    "previous_gate_count": int(improvement_match.group(3)),
                    "new_gate_count": int(improvement_match.group(4)),
                }
            )
            continue

        collect_match = COLLECT_RE.search(line)
        if collect_match:
            current = {
                "iteration": int(collect_match.group(1)),
                "rollout_seconds_exact": float(collect_match.group(2)),
                "improvements": pending_improvements,
            }
            for improvement in pending_improvements:
                improvement["iteration"] = current["iteration"]
                improvements.append(improvement)
            pending_improvements = []
            iterations.append(current)
            continue

        if current is None:
            continue

        metric_match = METRIC_RE.match(line)
        if metric_match:
            current[metric_match.group(1)] = parse_number(metric_match.group(2))
            continue

        timing_match = TIMING_RE.search(line)
        if timing_match:
            current.update(
                {
                    "rollout_seconds": float(timing_match.group(1)),
                    "learn_seconds": float(timing_match.group(2)),
                    "iteration_seconds": float(timing_match.group(3)),
                    "rollout_wall_fraction": float(timing_match.group(4)),
                }
            )

    complete_iterations = [row for row in iterations if "iteration_seconds" in row]
    total_transitions = sum(int(row.get("num_exps", 0)) for row in complete_iterations)
    total_rollout_seconds = sum(
        float(row["rollout_seconds_exact"]) for row in complete_iterations
    )
    total_learn_seconds = sum(
        float(row.get("learn_seconds", 0.0)) for row in complete_iterations
    )
    total_iteration_seconds = sum(
        float(row["iteration_seconds"]) for row in complete_iterations
    )
    best_suffix = "_best_graph_gate_count"
    circuits = sorted(
        {
            key[: -len(best_suffix)]
            for row in complete_iterations
            for key in row
            if key.endswith(best_suffix)
        }
    )
    initial_best = {
        circuit: int(complete_iterations[0][circuit + best_suffix])
        for circuit in circuits
        if circuit + best_suffix in complete_iterations[0]
    }
    final_best = {
        circuit: int(complete_iterations[-1][circuit + best_suffix])
        for circuit in circuits
        if circuit + best_suffix in complete_iterations[-1]
    }
    initial_before_rollout = dict(initial_best)
    for improvement in improvements:
        circuit = str(improvement["circuit"])
        if circuit not in initial_before_rollout:
            initial_before_rollout[circuit] = int(improvement["previous_gate_count"])
        else:
            initial_before_rollout[circuit] = max(
                initial_before_rollout[circuit],
                int(improvement["previous_gate_count"]),
            )

    return {
        "schema": "quarl-training-log-v1",
        "source_log": path.name,
        "completed_iterations": len(complete_iterations),
        "partial_iterations": len(iterations) - len(complete_iterations),
        "iterations": complete_iterations,
        "improvements": improvements,
        "unassigned_improvements_after_last_complete_iteration": pending_improvements,
        "aggregate": {
            "initial_best_before_rollout_by_circuit": initial_before_rollout,
            "initial_logged_best_gate_count_by_circuit": initial_best,
            "final_best_gate_count_by_circuit": final_best,
            "total_transitions": total_transitions,
            "total_rollout_seconds": total_rollout_seconds,
            "total_learn_seconds": total_learn_seconds,
            "total_iteration_seconds": total_iteration_seconds,
            "transition_throughput_per_second": (
                total_transitions / total_rollout_seconds
                if total_rollout_seconds
                else None
            ),
            "rollout_wall_fraction": (
                total_rollout_seconds / total_iteration_seconds
                if total_iteration_seconds
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse original Quarl PPO logs")
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = parse_quarl_log(args.log)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
