from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


QASM_ROW = re.compile(
    r"^(?P<step>\d+)_(?P<cost>-?\d+)_(?P<reward>-?\d+)_"
    r"(?P<node>\d+)_(?P<xfer>\d+)\.qasm$"
)


@dataclass(frozen=True)
class TrajectorySpec:
    source_id: str
    circuit: str
    kind: str
    path: Path


@dataclass(frozen=True)
class QasmState:
    step: int
    gate_count: int
    reward: int
    node_id: int
    xfer_id: int
    path: Path


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected a non-empty comma-separated list of positive integers")
    return sorted(set(values))


def parse_trajectory_spec(value: str) -> TrajectorySpec:
    fields = value.split("|", 3)
    if len(fields) != 4 or any(not field.strip() for field in fields):
        raise ValueError(
            "trajectory specifications must be SOURCE_ID|CIRCUIT|KIND|PATH"
        )
    source_id, circuit, kind, raw_path = (field.strip() for field in fields)
    return TrajectorySpec(
        source_id=source_id,
        circuit=circuit,
        kind=kind,
        path=Path(raw_path).expanduser().resolve(),
    )


def read_trajectory(path: Path) -> list[QasmState]:
    if not path.is_dir():
        raise ValueError(f"trajectory directory does not exist: {path}")
    states = []
    for qasm in path.glob("*.qasm"):
        match = QASM_ROW.fullmatch(qasm.name)
        if match is None:
            raise ValueError(f"unexpected trajectory filename: {qasm}")
        states.append(
            QasmState(
                step=int(match["step"]),
                gate_count=int(match["cost"]),
                reward=int(match["reward"]),
                node_id=int(match["node"]),
                xfer_id=int(match["xfer"]),
                path=qasm.resolve(),
            )
        )
    states.sort(key=lambda state: state.step)
    if len(states) < 2:
        raise ValueError(f"trajectory needs at least one action and a terminal state: {path}")
    expected_steps = list(range(len(states)))
    actual_steps = [state.step for state in states]
    if actual_steps != expected_steps:
        raise ValueError(
            f"trajectory steps are not contiguous in {path}: "
            f"expected 0..{len(states) - 1}"
        )
    if states[-1].node_id != 0 or states[-1].xfer_id != 0:
        raise ValueError(f"trajectory lacks a terminal 0/0 sentinel: {path}")
    return states


def uniform_indices(length: int, count: int) -> list[int]:
    """Choose deterministic, endpoint-inclusive indices without duplication."""

    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted(
        {
            round(index * (length - 1) / (count - 1))
            for index in range(count)
        }
    )


def future_labels(
    states: list[QasmState], index: int, horizons: Iterable[int]
) -> dict[str, dict[str, int | None]]:
    initial = states[index].gate_count
    labels = {}
    for horizon in horizons:
        future = states[index + 1 : min(len(states), index + horizon + 1)]
        if not future:
            labels[str(horizon)] = {
                "observed_steps": 0,
                "best_gate_count": initial,
                "improvement": 0,
                "first_improvement_step": None,
            }
            continue
        best = min(state.gate_count for state in future)
        first_improvement = next(
            (
                offset
                for offset, state in enumerate(future, start=1)
                if state.gate_count < initial
            ),
            None,
        )
        labels[str(horizon)] = {
            "observed_steps": len(future),
            "best_gate_count": best,
            "improvement": max(0, initial - best),
            "first_improvement_step": first_improvement,
        }
    return labels


def trajectory_behavior(
    states: list[QasmState], index: int, probe_horizon: int, target_horizon: int
) -> dict[str, int | str | bool | None]:
    """Describe teacher-trajectory behavior without claiming causal necessity."""

    initial = states[index].gate_count
    target_states = states[index + 1 : min(len(states), index + target_horizon + 1)]
    probe_states = target_states[:probe_horizon]
    probe_best = min((state.gate_count for state in probe_states), default=initial)
    target_best = min((state.gate_count for state in target_states), default=initial)
    probe_gain = max(0, initial - probe_best)
    target_gain = max(0, initial - target_best)
    right_censored = len(target_states) < target_horizon
    first_improvement = next(
        (
            offset
            for offset, state in enumerate(target_states, start=1)
            if state.gate_count < initial
        ),
        None,
    )
    observed_until_gain = (
        target_states[:first_improvement]
        if first_improvement is not None
        else target_states
    )
    peak_increase = max(
        [0, *(state.gate_count - initial for state in observed_until_gain)]
    )
    if probe_gain == 0 and target_gain > 0:
        behavior_class = "delayed_gain"
    elif probe_gain > 0 and target_gain > probe_gain:
        behavior_class = "continued_gain"
    elif probe_gain > 0 and right_censored:
        behavior_class = "censored_after_probe"
    elif probe_gain > 0:
        behavior_class = "saturated_after_probe"
    elif right_censored:
        behavior_class = "censored_no_gain"
    else:
        behavior_class = "no_gain"
    return {
        "class": behavior_class,
        "probe_horizon": probe_horizon,
        "target_horizon": target_horizon,
        "observed_probe_steps": len(probe_states),
        "observed_target_steps": len(target_states),
        "right_censored": right_censored,
        "probe_improvement": probe_gain,
        "target_improvement": target_gain,
        "additional_improvement": max(0, probe_best - target_best),
        "first_improvement_step": first_improvement,
        "plateau_steps": (
            first_improvement - 1
            if first_improvement is not None
            else len(target_states)
        ),
        "observed_peak_increase_before_improvement": peak_increase,
        "observed_increase_then_gain": peak_increase > 0 and target_gain > 0,
    }


def qasm_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    specs: list[TrajectorySpec],
    states_per_trajectory: int,
    horizons: list[int],
    *,
    selection_method: str = "uniform",
    probe_horizon: int = 4,
    target_horizon: int = 64,
    states_per_stratum: int | None = None,
) -> dict:
    if selection_method not in {"uniform", "behavior_stratified"}:
        raise ValueError(f"unknown selection method: {selection_method}")
    if probe_horizon <= 0 or target_horizon <= probe_horizon:
        raise ValueError("target horizon must be greater than the positive probe horizon")
    if states_per_stratum is not None and states_per_stratum <= 0:
        raise ValueError("states_per_stratum must be positive")
    rows = []
    source_selection = []
    for spec in specs:
        states = read_trajectory(spec.path)
        # The last file is a terminal state and has no continuation action.
        candidate_count = len(states) - 1
        behaviors = {
            index: trajectory_behavior(
                states, index, probe_horizon, target_horizon
            )
            for index in range(candidate_count)
        }
        available_counts = {
            name: sum(row["class"] == name for row in behaviors.values())
            for name in (
                "delayed_gain",
                "continued_gain",
                "saturated_after_probe",
                "no_gain",
                "censored_after_probe",
                "censored_no_gain",
            )
        }
        if selection_method == "uniform":
            selected_indices = uniform_indices(candidate_count, states_per_trajectory)
        else:
            quota = states_per_stratum or states_per_trajectory
            selected_indices = []
            for name in available_counts:
                bucket = [
                    index
                    for index, behavior in behaviors.items()
                    if behavior["class"] == name
                ]
                selected_indices.extend(
                    bucket[position]
                    for position in uniform_indices(len(bucket), quota)
                )
            selected_indices = sorted(set(selected_indices))
        selected_counts = {
            name: sum(behaviors[index]["class"] == name for index in selected_indices)
            for name in available_counts
        }
        source_selection.append(
            {
                "source_id": spec.source_id,
                "available_behavior_counts": available_counts,
                "selected_behavior_counts": selected_counts,
                "observed_increase_then_gain_available": sum(
                    bool(row["observed_increase_then_gain"])
                    for row in behaviors.values()
                ),
                "observed_increase_then_gain_selected": sum(
                    bool(behaviors[index]["observed_increase_then_gain"])
                    for index in selected_indices
                ),
            }
        )
        for index in selected_indices:
            state = states[index]
            rows.append(
                {
                    "id": f"{spec.source_id}-step-{state.step:04d}",
                    "circuit": spec.circuit,
                    "kind": spec.kind,
                    "source_id": spec.source_id,
                    "source_trajectory": str(spec.path),
                    "trajectory_step": state.step,
                    "qasm": str(state.path),
                    "qasm_sha256": qasm_sha256(state.path),
                    "initial_gate_count": state.gate_count,
                    "teacher_action": {
                        "xfer_id": state.xfer_id,
                        "node_id": state.node_id,
                        "reward": state.reward,
                    },
                    "teacher_future": future_labels(states, index, horizons),
                    "teacher_behavior": behaviors[index],
                }
            )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("state ids are not unique; use distinct SOURCE_ID values")
    return {
        "format": "continuation-value-manifest-v1",
        "horizons": horizons,
        "selection": {
            "method": (
                "uniform_endpoint_inclusive"
                if selection_method == "uniform"
                else "teacher_behavior_stratified"
            ),
            "states_per_trajectory": states_per_trajectory,
            "states_per_stratum": states_per_stratum,
            "probe_horizon": probe_horizon,
            "target_horizon": target_horizon,
            "source_counts": source_selection,
        },
        "sources": [
            {
                "source_id": spec.source_id,
                "circuit": spec.circuit,
                "kind": spec.kind,
                "path": str(spec.path),
            }
            for spec in specs
        ],
        "states": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reproducible fixed-budget continuation state manifest."
    )
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        help="SOURCE_ID|CIRCUIT|KIND|PATH; repeat for every trajectory",
    )
    parser.add_argument("--states-per-trajectory", type=int, default=16)
    parser.add_argument(
        "--selection",
        choices=("uniform", "behavior_stratified"),
        default="uniform",
    )
    parser.add_argument("--states-per-stratum", type=int)
    parser.add_argument("--probe-horizon", type=int, default=4)
    parser.add_argument("--target-horizon", type=int, default=64)
    parser.add_argument("--horizons", default="8,32,64,128")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.states_per_trajectory <= 0:
        parser.error("--states-per-trajectory must be positive")
    try:
        specs = [parse_trajectory_spec(value) for value in args.trajectory]
        horizons = parse_int_csv(args.horizons)
        manifest = build_manifest(
            specs,
            args.states_per_trajectory,
            horizons,
            selection_method=args.selection,
            probe_horizon=args.probe_horizon,
            target_horizon=args.target_horizon,
            states_per_stratum=args.states_per_stratum,
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sources": len(specs),
                "states": len(manifest["states"]),
                "horizons": horizons,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
