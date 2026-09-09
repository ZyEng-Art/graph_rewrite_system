from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TIMING_KEYS = {
    "seconds",
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
    "widening_action_cache",
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


def _variant(path: Path, cache_key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    wall_path = path.with_suffix(".wall_seconds")
    quartz_apply_seconds = _sum_steps(payload, "quartz_apply_seconds")
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
            "non_apply_search_seconds": (
                float(payload["total_seconds"]) - quartz_apply_seconds
            ),
            "wall_seconds": float(wall_path.read_text().strip()),
            "model_match_seconds": _sum_steps(payload, "model_match_seconds"),
            "proposal_seconds": _sum_steps(payload, "proposal_seconds"),
            "quartz_apply_seconds": quartz_apply_seconds,
            "cache": payload[cache_key],
        },
    }


def _relative_reduction(before: float, after: float) -> float:
    return (before - after) / before if before else 0.0


def _mean(left: float, right: float) -> float:
    return (left + right) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-key",
        choices=("widening_candidate_cache", "widening_action_cache"),
        default="widening_candidate_cache",
    )
    args = parser.parse_args()

    cache_off_before = _variant(
        args.input_dir / "cache_off_before.json", args.cache_key
    )
    cache_on = _variant(args.input_dir / "cache_on.json", args.cache_key)
    cache_off_after = _variant(
        args.input_dir / "cache_off_after.json", args.cache_key
    )
    off_before_summary = cache_off_before["summary"]
    on_summary = cache_on["summary"]
    off_after_summary = cache_off_after["summary"]
    structural = [
        _structural_projection(variant["payload"])
        for variant in (cache_off_before, cache_on, cache_off_after)
    ]
    structural_equal = structural[0] == structural[1] == structural[2]
    off_mean = {
        key: _mean(off_before_summary[key], off_after_summary[key])
        for key in (
            "model_match_seconds",
            "non_apply_search_seconds",
            "proposal_seconds",
            "quartz_apply_seconds",
            "search_seconds",
            "wall_seconds",
        )
    }
    result = {
        "structural_equal_excluding_timing_cache_and_collation": structural_equal,
        "cache_off_before": off_before_summary,
        "cache_on": on_summary,
        "cache_off_after": off_after_summary,
        "cache_off_timing_mean": off_mean,
        "relative_reduction": {
            "model_match_seconds": _relative_reduction(
                off_mean["model_match_seconds"], on_summary["model_match_seconds"]
            ),
            "non_apply_search_seconds": _relative_reduction(
                off_mean["non_apply_search_seconds"],
                on_summary["non_apply_search_seconds"],
            ),
            "search_seconds": _relative_reduction(
                off_mean["search_seconds"], on_summary["search_seconds"]
            ),
            "wall_seconds": _relative_reduction(
                off_mean["wall_seconds"], on_summary["wall_seconds"]
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not structural_equal:
        raise SystemExit("cache-on and cache-off structural search outputs differ")


if __name__ == "__main__":
    main()
