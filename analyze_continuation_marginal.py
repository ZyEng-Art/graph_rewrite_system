from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def mean(values):
    rows = list(values)
    return sum(rows) / max(1, len(rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(payload: dict) -> dict:
    transitions = payload["continuation_transition_aggregates"]
    by_pair = defaultdict(list)
    for row in transitions:
        by_pair[(row["probe_budget"], row["target_budget"])].append(row)

    overall = []
    for (probe_budget, target_budget), rows in sorted(by_pair.items()):
        seed_rows = [seed for row in rows for seed in row["seed_outcomes"]]
        class_counts = Counter(row["outcome_class"] for row in seed_rows)
        overall.append(
            {
                "probe_budget": probe_budget,
                "target_budget": target_budget,
                "states": len(rows),
                "seed_outcomes": len(seed_rows),
                "states_with_any_additional_gain": sum(
                    row["additional_improvement_max"] > 0 for row in rows
                ),
                "additional_success_rate": mean(
                    row["additional_improvement_max"] > 0 for row in rows
                ),
                "additional_improvement_mean": mean(
                    row["additional_improvement_mean"] for row in rows
                ),
                "additional_improvement_max": max(
                    row["additional_improvement_max"] for row in rows
                ),
                "additional_improvement_per_1000_attempts_mean": mean(
                    row["additional_improvement_per_1000_attempts"] for row in rows
                ),
                "additional_search_seconds_mean": mean(
                    row["additional_search_seconds_mean"] for row in rows
                ),
                "seed_outcome_classes": dict(sorted(class_counts.items())),
                "observed_frontier_detour_seed_outcomes": sum(
                    row["observed_frontier_peak_increase"] > 0 for row in seed_rows
                ),
            }
        )

    rankings = []
    for row in payload["marginal_ranking_evaluations"]:
        if row["scope_type"] not in {"all", "circuit"}:
            continue
        if row["top_fraction"] != 0.25:
            continue
        rankings.append(row)

    by_teacher_class = defaultdict(list)
    for row in transitions:
        teacher_class = str(row.get("behavior_stratum")).split(":", 1)[0]
        key = (
            row["circuit"],
            teacher_class,
            row["probe_budget"],
            row["target_budget"],
        )
        by_teacher_class[key].append(row)
    teacher_classes = []
    for (circuit, teacher_class, probe_budget, target_budget), rows in sorted(
        by_teacher_class.items()
    ):
        teacher_classes.append(
            {
                "circuit": circuit,
                "teacher_class": teacher_class,
                "probe_budget": probe_budget,
                "target_budget": target_budget,
                "states": len(rows),
                "actual_additional_success_rate": mean(
                    row["additional_improvement_max"] > 0 for row in rows
                ),
                "actual_additional_improvement_mean": mean(
                    row["additional_improvement_mean"] for row in rows
                ),
            }
        )

    seed_variation = Counter()
    for row in transitions:
        outcomes = {seed["additional_improvement"] for seed in row["seed_outcomes"]}
        seed_variation["same_additional_improvement"] += len(outcomes) == 1
        seed_variation["different_additional_improvement"] += len(outcomes) > 1

    return {
        "format": "continuation-marginal-analysis-v1",
        "completed_physical_jobs": payload["completed_physical_jobs"],
        "failed_physical_jobs": payload["failed_physical_jobs"],
        "completed_logical_jobs": payload["completed_jobs"],
        "failed_logical_jobs": payload["failed_jobs"],
        "overall_transitions": overall,
        "teacher_stratum_transitions": payload["group_transition_aggregates"],
        "teacher_class_transitions": teacher_classes,
        "top_25_percent_rankings": rankings,
        "seed_variation_across_state_transitions": dict(seed_variation),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    report = analyze(payload)
    report["summary_sha256"] = sha256(args.summary)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
