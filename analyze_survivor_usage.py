from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files = sorted(args.result_root.rglob("result.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    steps = [step for result in results for step in result.get("steps", [])]
    selections = [step.get("survivor_selection", {}) for step in steps]
    lane_counts = Counter()
    for step in steps:
        lane_counts.update(step.get("output_survivor_lanes", {}))

    reference_stages = Counter()
    for result in results:
        for row in result.get("reference_retention", []):
            detail = row.get("loss_detail")
            if detail:
                reference_stages[str(detail.get("exclusion_stage"))] += 1

    report = {
        "format": "survivor-usage-analysis-v1",
        "result_files": len(files),
        "steps": len(steps),
        "policy_counts": dict(
            Counter(result.get("survivor_policy", "legacy") for result in results)
        ),
        "output_lane_state_counts": dict(lane_counts),
        "mean_exploration_selected_per_step": (
            mean(int(row.get("exploration_selected", 0)) for row in selections)
            if selections else 0.0
        ),
        "mean_exploration_target_per_step": (
            mean(int(row.get("exploration_target", 0)) for row in selections)
            if selections else 0.0
        ),
        "steps_filling_exploration_target": sum(
            int(row.get("exploration_selected", 0))
            == int(row.get("exploration_target", 0))
            and int(row.get("exploration_target", 0)) > 0
            for row in selections
        ),
        "steps_with_exploration_shortage": sum(
            int(row.get("exploration_selected", 0))
            < int(row.get("exploration_target", 0))
            for row in selections
        ),
        "mean_candidate_pool_per_step": (
            mean(int(row.get("candidate_states", 0)) for row in selections)
            if selections else 0.0
        ),
        "reference_loss_stages": dict(reference_stages),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
