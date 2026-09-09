from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RUNS = (
    "fixed_top128",
    "round_robin_s73",
    "round_robin_s170",
    "feedback_s73",
    "feedback_s170",
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(root: Path, name: str) -> dict[str, Any]:
    result_path = root / f"{name}.json"
    payload = json.loads(result_path.read_text())
    wall_path = root / f"{name}.wall_seconds"
    improvement_structure = [
        {key: value for key, value in row.items() if key != "seconds"}
        for row in payload["improvement_trace"]
    ]
    decision_trace = []
    for row in payload.get("steps", []):
        widening = row.get("progressive_widening", {})
        decision_trace.append(
            {
                "step": row["step"],
                "best_gate_count": row["best_gate_count"],
                "global_best_gate_count": row["global_best_gate_count"],
                "input_max_action_depth": row["input_max_action_depth"],
                "output_max_action_depth": row["output_max_action_depth"],
                "proposals_scanned": row["proposals_scanned"],
                "attempted_actions": row["attempted_actions"],
                "accepted_actions": row["accepted_actions"],
                "feedback": row.get("search_feedback"),
                "widening_lane_counts": widening.get("lane_counts"),
                "widening_selected": [
                    {
                        "identity_order": selected["identity_order"],
                        "gate_count": selected["gate_count"],
                        "action_depth": selected["action_depth"],
                        "current_round": selected["current_round"],
                        "next_round": selected["next_round"],
                    }
                    for selected in widening.get("selected", [])
                ],
            }
        )
    decision_trace_sha256 = hashlib.sha256(
        json.dumps(
            decision_trace, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "name": name,
        "initial_gate_count": payload["initial_gate_count"],
        "best_gate_count": payload["best_gate_count"],
        "best_first_seen_step": payload["best_first_seen_step"],
        "best_first_seen_seconds": payload["best_first_seen_seconds"],
        "best_action_depth": payload["best_action_depth"],
        "total_attempted_actions": payload["total_attempted_actions"],
        "completed_depth": payload["completed_depth"],
        "maximum_action_depth": payload["maximum_action_depth"],
        "widening_policy": payload.get("widening_policy"),
        "search_feedback": payload.get("search_feedback"),
        "best_history": payload["best_history"],
        "best_widened_action_trace": payload["best_widened_action_trace"],
        "improvement_trace": payload["improvement_trace"],
        "improvement_trace_structure": improvement_structure,
        "decision_trace_sha256": decision_trace_sha256,
        "best_qasm_sha256": sha256(root / f"{name}.best.qasm"),
        "wall_seconds": (
            float(wall_path.read_text().strip())
            if wall_path.is_file()
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [load_run(args.input_dir, name) for name in RUNS]
    by_name = {row["name"]: row for row in rows}
    left = by_name["feedback_s73"]
    right = by_name["feedback_s170"]
    determinism_fields = (
        "best_gate_count",
        "best_first_seen_step",
        "best_action_depth",
        "total_attempted_actions",
        "completed_depth",
        "maximum_action_depth",
        "best_history",
        "best_widened_action_trace",
        "improvement_trace_structure",
        "decision_trace_sha256",
        "best_qasm_sha256",
    )
    comparison = {
        field: left[field] == right[field] for field in determinism_fields
    }
    payload = {
        "format": "feedback-widening-ab-summary-v1",
        "runs": rows,
        "feedback_seed_independence": {
            "all_compared_fields_equal": all(comparison.values()),
            "fields": comparison,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
