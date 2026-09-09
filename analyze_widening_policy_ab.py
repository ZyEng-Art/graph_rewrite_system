from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


SUMMARY_FIELDS = (
    "best_gate_count",
    "best_first_seen_step",
    "best_first_seen_seconds",
    "best_action_depth",
    "unique_graphs_seen",
    "total_seconds",
)


def load_results(directory: Path, budget: int) -> dict[str, dict]:
    suffix = f"_apply{budget}.json"
    results = {}
    for raw_path in glob.glob(str(directory / f"*{suffix}")):
        path = Path(raw_path)
        stem = path.name.removesuffix(suffix)
        results[stem] = json.loads(path.read_text())
    if not results:
        raise ValueError(f"no budget-{budget} results found in {directory}")
    return results


def compare(baseline: dict[str, dict], candidate: dict[str, dict]) -> dict:
    if baseline.keys() != candidate.keys():
        raise ValueError(
            "result sets differ: "
            f"baseline_only={sorted(baseline.keys() - candidate.keys())}, "
            f"candidate_only={sorted(candidate.keys() - baseline.keys())}"
        )
    rows = []
    wins = ties = losses = 0
    for circuit in sorted(baseline):
        left = baseline[circuit]
        right = candidate[circuit]
        gate_delta = right["best_gate_count"] - left["best_gate_count"]
        wins += gate_delta < 0
        ties += gate_delta == 0
        losses += gate_delta > 0
        rows.append(
            {
                "circuit": circuit,
                "baseline": {field: left[field] for field in SUMMARY_FIELDS},
                "candidate": {field: right[field] for field in SUMMARY_FIELDS},
                "candidate_minus_baseline": {
                    field: right[field] - left[field] for field in SUMMARY_FIELDS
                },
                "best_graph_digest_same": (
                    left["best_graph_exact_identity_digest"]
                    == right["best_graph_exact_identity_digest"]
                ),
            }
        )
    means = {}
    for field in SUMMARY_FIELDS:
        baseline_mean = sum(row["baseline"][field] for row in rows) / len(rows)
        candidate_mean = sum(row["candidate"][field] for row in rows) / len(rows)
        means[field] = {
            "baseline": baseline_mean,
            "candidate": candidate_mean,
            "delta": candidate_mean - baseline_mean,
            "relative_delta_percent": (
                100.0 * (candidate_mean / baseline_mean - 1.0)
                if baseline_mean
                else None
            ),
        }
    return {
        "circuits": len(rows),
        "gate_count_wins_ties_losses": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "means": means,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two widening-policy corpora.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        load_results(args.baseline_dir, args.budget),
        load_results(args.candidate_dir, args.budget),
    )
    result["baseline_dir"] = str(args.baseline_dir.resolve())
    result["candidate_dir"] = str(args.candidate_dir.resolve())
    result["budget"] = args.budget
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
