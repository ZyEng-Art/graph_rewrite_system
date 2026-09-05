from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from dataset import RuleMetadata


def action_key(action: dict) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    return (
        int(action["xfer_id"]),
        tuple(map(int, action["source_slots"])),
        tuple(map(int, action["destination_slots"])),
    )


def normalized_action(action: tuple, rules: RuleMetadata) -> dict:
    xfer_id, source_slots, destination_slots = action
    return {
        "xfer_id": xfer_id,
        "source_id": rules.xfer_to_source[xfer_id],
        "anchor_slot": source_slots[0],
        "binding_slots": source_slots,
        "dst_slots": destination_slots,
    }


def gate_deltas(rules: RuleMetadata) -> tuple[int, ...]:
    return tuple(
        len(rules.destination_gate_types[xfer_id])
        - len(rules.source_gate_types[rules.xfer_to_source[xfer_id]])
        for xfer_id in range(len(rules.xfer_to_source))
    )


def archive_history_paths(paths: list[Path]) -> list[Path]:
    histories = []
    for path in paths:
        archive = json.loads(path.read_text())
        if archive.get("format") != "accelerated-self-improve-v1":
            raise ValueError(f"unsupported self-improvement archive: {path}")
        for row in archive["rounds"]:
            history = row.get("refresh_history")
            if history is not None:
                histories.append(Path(history))
    return histories


def unique_paths(paths: list[Path]) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


def trajectory_future_residuals(
    initial_gate_count: int,
    history: tuple,
    deltas: tuple[int, ...],
    objective: str,
) -> list[int]:
    gate_counts = []
    gate_count = initial_gate_count
    for action in history:
        gate_count += deltas[action[0]]
        gate_counts.append(gate_count)
    if objective == "final-residual":
        return [gate_counts[-1] - child_count for child_count in gate_counts]
    if objective == "best-prefix-residual":
        return [
            min(gate_counts[depth:]) - child_count
            for depth, child_count in enumerate(gate_counts)
        ]
    raise ValueError(f"unknown preference objective: {objective}")


def collect_file(
    path: Path,
    rules: RuleMetadata,
    deltas: tuple[int, ...],
    *,
    min_remaining_depth: int,
    min_descendants: int,
    objective: str,
) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text())
    completed_depth = int(payload["completed_depth"])
    child_outcomes: dict[tuple, dict[tuple, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    prefix_actions: dict[tuple, tuple] = {}
    initial_gate_counts = set()

    for state in payload["states"]:
        history = tuple(action_key(action) for action in state["history"])
        final_gate_count = int(state["gate_count"])
        initial_gate_count = final_gate_count - sum(
            deltas[action[0]] for action in history
        )
        initial_gate_counts.add(initial_gate_count)
        future_residuals = trajectory_future_residuals(
            initial_gate_count, history, deltas, objective
        )
        parent_gate_count = initial_gate_count
        for depth, child in enumerate(history):
            prefix = history[:depth]
            child_gate_count = parent_gate_count + deltas[child[0]]
            future_residual = future_residuals[depth]
            child_outcomes[prefix][child].append(future_residual)
            prefix_actions.setdefault(prefix, prefix)
            parent_gate_count = child_gate_count

    if len(initial_gate_counts) != 1:
        raise ValueError(f"inconsistent initial gate counts in {path}")
    if "initial_snapshot" not in payload:
        raise ValueError(f"beam history lacks initial_snapshot: {path}")

    circuit = Path(payload["qasm"]).name
    preferences = []
    groups_considered = groups_with_signal = 0
    skipped_short_horizon = 0
    for prefix, children in child_outcomes.items():
        remaining_after_action = completed_depth - len(prefix) - 1
        if remaining_after_action < min_remaining_depth:
            skipped_short_horizon += 1
            continue
        if len(children) < 2:
            continue
        groups_considered += 1
        best_by_child = {
            child: min(outcomes)
            for child, outcomes in children.items()
            if len(outcomes) >= min_descendants
        }
        if len(best_by_child) < 2:
            continue
        preferred_residual = min(best_by_child.values())
        rejected = [
            child
            for child, outcome in best_by_child.items()
            if outcome > preferred_residual
        ]
        if not rejected:
            continue
        groups_with_signal += 1
        preferred = min(
            (
                child
                for child, outcome in best_by_child.items()
                if outcome == preferred_residual
            ),
            key=lambda child: (-len(children[child]), child),
        )
        actions = [normalized_action(action, rules) for action in prefix_actions[prefix]]
        for rejected_child in sorted(rejected):
            rejected_residual = best_by_child[rejected_child]
            preferences.append(
                {
                    "circuit": circuit,
                    "initial_graph": payload["initial_snapshot"],
                    "actions": actions,
                    "preferred": normalized_action(preferred, rules),
                    "rejected": normalized_action(rejected_child, rules),
                    "preferred_future_residual": preferred_residual,
                    "rejected_future_residual": rejected_residual,
                    "advantage": rejected_residual - preferred_residual,
                    "prefix_length": len(prefix),
                    "remaining_after_action": remaining_after_action,
                    "preferred_descendants": len(children[preferred]),
                    "rejected_descendants": len(children[rejected_child]),
                    "source_history": str(path),
                }
            )

    stats = {
        "circuit": circuit,
        "completed_depth": completed_depth,
        "final_states": len(payload["states"]),
        "initial_gate_count": initial_gate_counts.pop(),
        "groups_considered": groups_considered,
        "groups_with_signal": groups_with_signal,
        "preferences": len(preferences),
        "skipped_short_horizon_groups": skipped_short_horizon,
    }
    return preferences, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histories", type=Path, nargs="*", default=[])
    parser.add_argument("--archives", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--validation-histories", type=Path, nargs="*", default=[]
    )
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-marker", default="_s75_")
    parser.add_argument("--min-remaining-depth", type=int, default=3)
    parser.add_argument("--min-descendants", type=int, default=2)
    parser.add_argument(
        "--objective",
        choices=("final-residual", "best-prefix-residual"),
        default="best-prefix-residual",
    )
    args = parser.parse_args()
    if args.min_remaining_depth < 1:
        parser.error("--min-remaining-depth must be positive")
    if args.min_descendants < 1:
        parser.error("--min-descendants must be positive")
    histories = unique_paths(
        args.histories
        + archive_history_paths(args.archives)
        + args.validation_histories
    )
    if not histories:
        parser.error("provide --histories, --archives, or --validation-histories")
    explicit_validation = {
        path.resolve() for path in args.validation_histories
    }

    reference = torch.load(
        args.reference_data, map_location="cpu", weights_only=False
    )
    rules = RuleMetadata.from_payload(reference)
    deltas = gate_deltas(rules)
    train_preferences = []
    test_preferences = []
    file_stats = []
    for path in histories:
        preferences, stats = collect_file(
            path,
            rules,
            deltas,
            min_remaining_depth=args.min_remaining_depth,
            min_descendants=args.min_descendants,
            objective=args.objective,
        )
        split = (
            "test"
            if path.resolve() in explicit_validation
            or args.validation_marker in path.name
            else "train"
        )
        stats["split"] = split
        file_stats.append(stats)
        if split == "test":
            test_preferences.extend(preferences)
        else:
            train_preferences.extend(preferences)

    result = {
        key: reference[key]
        for key in (
            "source_patterns",
            "xfer_to_source",
            "xfer_sources",
            "xfer_destinations",
        )
    }
    result.update(
        {
            "format": "quartz-action-preference-v1",
            "train_preferences": train_preferences,
            "test_preferences": test_preferences,
            "metadata": {
                "objective": (
                    "minimum descendant prefix_gate - child_gate"
                    if args.objective == "best-prefix-residual"
                    else "minimum descendant final_gate - child_gate"
                ),
                "objective_mode": args.objective,
                "validation_marker": args.validation_marker,
                "min_remaining_depth": args.min_remaining_depth,
                "min_descendants": args.min_descendants,
                "archive_files": [str(path) for path in args.archives],
                "history_files": [str(path) for path in histories],
                "validation_history_files": [
                    str(path) for path in args.validation_histories
                ],
                "files": file_stats,
                "train_preferences": len(train_preferences),
                "test_preferences": len(test_preferences),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(result["metadata"], indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["metadata"], indent=2, sort_keys=True))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
