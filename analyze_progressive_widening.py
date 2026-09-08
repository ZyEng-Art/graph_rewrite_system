from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps", [])
    usage = [step.get("widening_action_usage", {}) for step in steps]
    return {
        "result": str(path),
        "initial_gate_count": int(payload["initial_gate_count"]),
        "best_gate_count": int(payload["best_gate_count"]),
        "best_first_seen_step": int(payload["best_first_seen_step"]),
        "best_action_depth": payload.get("best_action_depth"),
        "best_last_action_parent_rank": payload.get(
            "best_last_action_parent_rank"
        ),
        "best_has_widening_ancestor": payload.get(
            "best_has_widening_ancestor"
        ),
        "best_history": payload.get("best_history"),
        "best_widened_action_trace": payload.get(
            "best_widened_action_trace"
        ),
        "scheduling_layers": int(payload["completed_depth"]),
        "maximum_action_depth": payload.get("maximum_action_depth"),
        "attempted_actions": int(payload["total_attempted_actions"]),
        "search_seconds": float(payload["total_seconds"]),
        "unique_graphs_seen": int(payload["unique_graphs_seen"]),
        "max_scanned_parent_rank": max(
            (
                int(row["scanned_parent_rank_max"])
                for row in usage
                if row.get("scanned_parent_rank_max") is not None
            ),
            default=-1,
        ),
        "max_attempted_parent_rank": max(
            (
                int(row["attempted_parent_rank_max"])
                for row in usage
                if row.get("attempted_parent_rank_max") is not None
            ),
            default=-1,
        ),
        "attempted_widened_actions": sum(
            int(row.get("attempted_widened_actions", 0)) for row in usage
        ),
        "selected_revisits": sum(
            int(step.get("progressive_widening", {}).get("selected_revisits", 0))
            for step in steps
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = (
        [args.root]
        if args.root.is_file()
        else sorted(args.root.rglob("result.json"))
    )
    runs = [summarize(path) for path in paths]
    report = {
        "format": "progressive-widening-analysis-v1",
        "runs": runs,
        "run_count": len(runs),
        "improved_runs": sum(
            run["best_gate_count"] < run["initial_gate_count"] for run in runs
        ),
        "mean_search_seconds": (
            mean(run["search_seconds"] for run in runs) if runs else 0.0
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
