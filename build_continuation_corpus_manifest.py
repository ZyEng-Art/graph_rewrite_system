from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from build_continuation_manifest import (
    future_labels,
    parse_int_csv,
    qasm_sha256,
    read_trajectory,
    trajectory_behavior,
)


@dataclass(frozen=True)
class TreeSpec:
    prefix: str
    circuit: str
    kind: str
    path: Path


def parse_tree_spec(value: str) -> TreeSpec:
    fields = value.split("|", 3)
    if len(fields) != 4 or any(not field.strip() for field in fields):
        raise ValueError("tree specifications must be PREFIX|CIRCUIT|KIND|PATH")
    prefix, circuit, kind, raw_path = (field.strip() for field in fields)
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"trajectory tree does not exist: {path}")
    return TreeSpec(prefix, circuit, kind, path)


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def behavior_stratum(behavior: dict) -> str:
    suffix = "detour" if behavior["observed_increase_then_gain"] else "flat"
    return f"{behavior['class']}:{suffix}"


def discover_candidates(
    specs: list[TreeSpec],
    horizons: list[int],
    probe_horizon: int,
    target_horizon: int,
) -> tuple[list[dict], list[dict]]:
    candidates = []
    sources = []
    seen_source_ids = set()
    for spec in specs:
        for directory in sorted(path for path in spec.path.rglob("*") if path.is_dir()):
            if not any(directory.glob("*.qasm")):
                continue
            try:
                states = read_trajectory(directory)
            except ValueError:
                continue
            relative = directory.relative_to(spec.path).as_posix().replace("/", "-")
            source_id = f"{spec.prefix}-{relative}"
            if source_id in seen_source_ids:
                raise ValueError(f"duplicate discovered source id: {source_id}")
            seen_source_ids.add(source_id)
            sources.append(
                {
                    "source_id": source_id,
                    "circuit": spec.circuit,
                    "kind": spec.kind,
                    "path": str(directory),
                    "actions": len(states) - 1,
                }
            )
            for index, state in enumerate(states[:-1]):
                behavior = trajectory_behavior(
                    states, index, probe_horizon, target_horizon
                )
                digest = qasm_sha256(state.path)
                candidates.append(
                    {
                        "id": f"{source_id}-step-{state.step:04d}",
                        "circuit": spec.circuit,
                        "kind": spec.kind,
                        "source_id": source_id,
                        "source_trajectory": str(directory),
                        "trajectory_step": state.step,
                        "qasm": str(state.path),
                        "qasm_sha256": digest,
                        "initial_gate_count": state.gate_count,
                        "teacher_action": {
                            "xfer_id": state.xfer_id,
                            "node_id": state.node_id,
                            "reward": state.reward,
                        },
                        "teacher_future": future_labels(states, index, horizons),
                        "teacher_behavior": behavior,
                        "behavior_stratum": behavior_stratum(behavior),
                    }
                )
    return candidates, sources


def deduplicate_candidates(candidates: list[dict]) -> tuple[list[dict], int]:
    """Merge identical states while retaining all observed teacher behaviors.

    A circuit state can occur in several Quarl trajectories and have different
    continuations.  Search it only once, but do not arbitrarily discard the
    fact that at least one saved path was delayed or took an observed detour.
    """

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[(row["circuit"], row["qasm_sha256"])].append(row)
    priority = {
        "delayed_gain:detour": 0,
        "delayed_gain:flat": 1,
        "continued_gain:detour": 2,
        "continued_gain:flat": 3,
        "saturated_after_probe:detour": 4,
        "saturated_after_probe:flat": 5,
        "no_gain:detour": 6,
        "no_gain:flat": 7,
        "censored_after_probe:detour": 8,
        "censored_after_probe:flat": 9,
        "censored_no_gain:detour": 10,
        "censored_no_gain:flat": 11,
    }
    selected = []
    for rows in grouped.values():
        canonical = min(
            rows,
            key=lambda row: (priority.get(row["behavior_stratum"], 99), row["id"]),
        )
        merged = dict(canonical)
        behavior_counts = Counter(row["behavior_stratum"] for row in rows)
        merged["exact_qasm_occurrences"] = len(rows)
        merged["teacher_behavior_variants"] = {
            "stratum_counts": dict(sorted(behavior_counts.items())),
            "source_ids": sorted({row["source_id"] for row in rows}),
            "ambiguous": len(behavior_counts) > 1,
        }
        selected.append(merged)
    return selected, len(candidates) - len(selected)


def sample_candidates(
    candidates: list[dict],
    states_per_stratum: int,
    max_per_source_stratum: int,
    min_step_gap: int,
) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[(row["circuit"], row["behavior_stratum"])].append(row)

    selected = []
    statistics = []
    for (circuit, stratum), rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda row: stable_key(row["id"]))
        source_counts = Counter()
        source_steps: dict[str, list[int]] = defaultdict(list)
        chosen = []
        chosen_ids = set()
        # Fill one state per source before allowing a second from any source.
        for source_limit in range(1, max_per_source_stratum + 1):
            for row in ranked:
                source = row["source_id"]
                step = int(row["trajectory_step"])
                if row["id"] in chosen_ids:
                    continue
                if source_counts[source] >= source_limit:
                    continue
                if any(abs(step - old) < min_step_gap for old in source_steps[source]):
                    continue
                chosen.append(row)
                chosen_ids.add(row["id"])
                source_counts[source] += 1
                source_steps[source].append(step)
                if len(chosen) >= states_per_stratum:
                    break
            if len(chosen) >= states_per_stratum:
                break
        selected.extend(chosen)
        statistics.append(
            {
                "circuit": circuit,
                "behavior_stratum": stratum,
                "available_states_after_exact_qasm_dedup": len(rows),
                "available_sources": len({row["source_id"] for row in rows}),
                "selected_states": len(chosen),
                "selected_sources": len(source_counts),
            }
        )
    return sorted(selected, key=lambda row: row["id"]), statistics


def build_corpus_manifest(
    specs: list[TreeSpec],
    horizons: list[int],
    probe_horizon: int,
    target_horizon: int,
    states_per_stratum: int,
    max_per_source_stratum: int,
    min_step_gap: int,
    include_censored: bool = False,
) -> dict:
    candidates, sources = discover_candidates(
        specs, horizons, probe_horizon, target_horizon
    )
    deduplicated, duplicate_qasm_states = deduplicate_candidates(candidates)
    eligible = [
        row
        for row in deduplicated
        if include_censored
        or not str(row["teacher_behavior"]["class"]).startswith("censored_")
    ]
    selected, strata = sample_candidates(
        eligible,
        states_per_stratum,
        max_per_source_stratum,
        min_step_gap,
    )
    return {
        "format": "continuation-value-manifest-v1",
        "horizons": horizons,
        "selection": {
            "method": "cross_trajectory_behavior_stratified",
            "probe_horizon": probe_horizon,
            "target_horizon": target_horizon,
            "states_per_stratum": states_per_stratum,
            "max_per_source_stratum": max_per_source_stratum,
            "min_step_gap": min_step_gap,
            "raw_candidate_states": len(candidates),
            "exact_qasm_duplicates_removed": duplicate_qasm_states,
            "candidate_states_after_dedup": len(deduplicated),
            "censored_candidates_excluded": len(deduplicated) - len(eligible),
            "include_censored": include_censored,
            "selected_states": len(selected),
            "strata": strata,
        },
        "sources": sources,
        "states": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a diverse continuation corpus across trajectory trees."
    )
    parser.add_argument(
        "--trajectory-tree",
        action="append",
        required=True,
        help="PREFIX|CIRCUIT|KIND|PATH; recursively discovers normalized trajectories",
    )
    parser.add_argument("--horizons", default="4,16,64")
    parser.add_argument("--probe-horizon", type=int, default=4)
    parser.add_argument("--target-horizon", type=int, default=64)
    parser.add_argument("--states-per-stratum", type=int, default=16)
    parser.add_argument("--max-per-source-stratum", type=int, default=2)
    parser.add_argument("--min-step-gap", type=int, default=8)
    parser.add_argument("--include-censored", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.states_per_stratum <= 0 or args.max_per_source_stratum <= 0:
        parser.error("sampling limits must be positive")
    if args.min_step_gap < 0:
        parser.error("--min-step-gap must be nonnegative")
    if args.probe_horizon <= 0 or args.target_horizon <= args.probe_horizon:
        parser.error("target horizon must be greater than the positive probe horizon")
    try:
        manifest = build_corpus_manifest(
            [parse_tree_spec(value) for value in args.trajectory_tree],
            parse_int_csv(args.horizons),
            args.probe_horizon,
            args.target_horizon,
            args.states_per_stratum,
            args.max_per_source_stratum,
            args.min_step_gap,
            args.include_censored,
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sources": len(manifest["sources"]),
                "states": len(manifest["states"]),
                **{
                    key: manifest["selection"][key]
                    for key in (
                        "raw_candidate_states",
                        "exact_qasm_duplicates_removed",
                        "candidate_states_after_dedup",
                    )
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
