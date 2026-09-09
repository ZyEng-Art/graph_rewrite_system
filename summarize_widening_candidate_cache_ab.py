from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TIMING_KEYS = {
    "accepted_actions_per_second",
    "cumulative_seconds",
    "match_states_per_second",
    "matched_actions_per_second",
    "model_match_seconds",
    "proposal_seconds",
    "quartz_apply_seconds",
    "successful_apply_actions_per_second",
    "total_seconds",
}
EXPECTED_CACHE_DIFFERENCES = {
    "state_only_collation",
    "widening_candidate_cache",
}


def _sum_steps(payload: dict[str, Any], key: str) -> float:
    return sum(float(step.get(key, 0.0)) for step in payload["steps"])


def _structural_projection(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _structural_projection(child)
            for key, child in value.items()
            if key not in TIMING_KEYS
            and not key.endswith("_seconds")
            and not key.endswith("_per_second")
            and key not in EXPECTED_CACHE_DIFFERENCES
        }
    if isinstance(value, list):
        return [_structural_projection(child) for child in value]
    return value


def _variant(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    wall_path = path.with_suffix(".wall_seconds")
    return {
        "payload": payload,
        "summary": {
            "best_gate_count": payload["best_gate_count"],
            "best_action_depth": payload["best_action_depth"],
            "completed_depth": payload["completed_depth"],
            "total_attempted_actions": payload["total_attempted_actions"],
            "final_beam_exact_identity_digest": payload[
                "final_beam_exact_identity_digest"
            ],
            "search_seconds": float(payload["total_seconds"]),
            "wall_seconds": float(wall_path.read_text().strip()),
            "model_match_seconds": _sum_steps(payload, "model_match_seconds"),
            "proposal_seconds": _sum_steps(payload, "proposal_seconds"),
            "quartz_apply_seconds": _sum_steps(payload, "quartz_apply_seconds"),
            "candidate_cache": payload["widening_candidate_cache"],
        },
    }


def _relative_reduction(before: float, after: float) -> float:
    return (before - after) / before if before else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_off = _variant(args.input_dir / "cache_off.json")
    cache_on = _variant(args.input_dir / "cache_on.json")
    off_summary = cache_off["summary"]
    on_summary = cache_on["summary"]
    structural_equal = _structural_projection(
        cache_off["payload"]
    ) == _structural_projection(cache_on["payload"])
    result = {
        "structural_equal_excluding_timing_cache_and_collation": structural_equal,
        "cache_off": off_summary,
        "cache_on": on_summary,
        "relative_reduction": {
            "model_match_seconds": _relative_reduction(
                off_summary["model_match_seconds"], on_summary["model_match_seconds"]
            ),
            "search_seconds": _relative_reduction(
                off_summary["search_seconds"], on_summary["search_seconds"]
            ),
            "wall_seconds": _relative_reduction(
                off_summary["wall_seconds"], on_summary["wall_seconds"]
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not structural_equal:
        raise SystemExit("cache-on and cache-off structural search outputs differ")


if __name__ == "__main__":
    main()
