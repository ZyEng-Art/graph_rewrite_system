from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch

from dataset import RuleMetadata


def split_bucket(source_path: str, modulo: int) -> int:
    digest = hashlib.sha256(source_path.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % modulo


def candidate_action(match: dict, xfer_id: int) -> dict:
    return {
        "xfer_id": int(xfer_id),
        "source_id": int(match["source_id"]),
        "anchor_slot": int(match["anchor_slot"]),
        "binding_slots": tuple(map(int, match["binding_slots"])),
    }


def action_key(action: dict) -> tuple[int, tuple[int, ...]]:
    return int(action["xfer_id"]), tuple(map(int, action["binding_slots"]))


def hard_negative_actions(
    step: dict,
    rules: RuleMetadata,
    *,
    max_negatives: int,
) -> list[dict]:
    preferred = step["action"]
    preferred_key = action_key(preferred)
    preferred_delta = (
        len(rules.destination_gate_types[int(preferred["xfer_id"])])
        - len(rules.source_gate_types[int(preferred["source_id"])])
    )
    unique = {}
    for match in step["matches"]:
        for xfer_id in match["xfer_ids"]:
            action = candidate_action(match, int(xfer_id))
            key = action_key(action)
            if key != preferred_key:
                unique.setdefault(key, action)

    def rank(action: dict) -> tuple:
        xfer_id = int(action["xfer_id"])
        source_id = int(action["source_id"])
        delta = (
            len(rules.destination_gate_types[xfer_id])
            - len(rules.source_gate_types[source_id])
        )
        # Immediate reductions are precisely the alternatives that a
        # gate-sorted beam prefers over a temporarily uphill teacher action.
        return (
            delta > preferred_delta,
            delta,
            source_id != int(preferred["source_id"]),
            xfer_id,
            tuple(action["binding_slots"]),
        )

    return sorted(unique.values(), key=rank)[:max_negatives]


def collect_preferences(
    payload: dict,
    *,
    max_negatives: int,
    test_modulo: int,
    test_remainder: int,
    force_train_suffixes: tuple[str, ...] = (),
    emphasis_suffixes: tuple[str, ...] = (),
    emphasis_repeat: int = 1,
    emphasis_max_negatives: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    rules = RuleMetadata.from_payload(payload)
    train = []
    test = []
    seen_steps = set()
    paths = defaultdict(lambda: {"steps": 0, "pairs": 0, "split": None})
    for trajectory in payload["train_trajectories"] + payload["test_trajectories"]:
        source_path = str(trajectory.get("source_path", trajectory["trajectory_id"]))
        base_path = source_path.split("#segment=", 1)[0]
        force_train = any(
            base_path.replace("\\", "/").endswith(suffix.replace("\\", "/"))
            for suffix in force_train_suffixes
        )
        emphasized = any(
            base_path.replace("\\", "/").endswith(suffix.replace("\\", "/"))
            for suffix in emphasis_suffixes
        )
        split = (
            "test"
            if not force_train
            and split_bucket(base_path, test_modulo) == test_remainder
            else "train"
        )
        destination = test if split == "test" else train
        history = []
        for step_index, step in enumerate(trajectory["steps"]):
            unique_key = (source_path, int(step.get("index", step_index)))
            if unique_key in seen_steps:
                history.append(dict(step["action"]))
                continue
            seen_steps.add(unique_key)
            negatives = hard_negative_actions(
                step,
                rules,
                max_negatives=(
                    emphasis_max_negatives
                    if emphasized and emphasis_max_negatives is not None
                    else max_negatives
                ),
            )
            preferred = {
                key: step["action"][key]
                for key in (
                    "xfer_id",
                    "source_id",
                    "anchor_slot",
                    "binding_slots",
                )
            }
            for rejected in negatives:
                row = {
                    "circuit": Path(base_path).parts[-2]
                    if len(Path(base_path).parts) >= 2
                    else Path(base_path).name,
                    "initial_graph": trajectory["initial_graph"],
                    "actions": [dict(action) for action in history],
                    "preferred": preferred,
                    "rejected": rejected,
                    "preferred_future_residual": -1,
                    "rejected_future_residual": 0,
                    "advantage": 1,
                    "prefix_length": len(history),
                    "remaining_after_action": len(trajectory["steps"])
                    - step_index
                    - 1,
                    "preferred_descendants": 1,
                    "rejected_descendants": 1,
                    "source_history": source_path,
                }
                destination.extend(
                    dict(row)
                    for _ in range(emphasis_repeat if emphasized else 1)
                )
            paths[base_path]["steps"] += 1
            paths[base_path]["pairs"] += len(negatives) * (
                emphasis_repeat if emphasized else 1
            )
            paths[base_path]["split"] = split
            history.append(dict(step["action"]))
    metadata = {
        "kind": "teacher_action_vs_legal_hard_negatives",
        "max_negatives_per_action": max_negatives,
        "test_modulo": test_modulo,
        "test_remainder": test_remainder,
        "force_train_suffixes": list(force_train_suffixes),
        "emphasis_suffixes": list(emphasis_suffixes),
        "emphasis_repeat": emphasis_repeat,
        "emphasis_max_negatives": emphasis_max_negatives,
        "unique_teacher_steps": len(seen_steps),
        "train_preferences": len(train),
        "test_preferences": len(test),
        "paths": dict(sorted(paths.items())),
    }
    return train, test, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build pairwise behavior-cloning preferences from exact teacher "
            "actions and competing legal matcher actions."
        )
    )
    parser.add_argument("--trajectory-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-negatives", type=int, default=32)
    parser.add_argument("--test-modulo", type=int, default=5)
    parser.add_argument("--test-remainder", type=int, default=0)
    parser.add_argument("--force-train-suffix", action="append", default=[])
    parser.add_argument("--emphasis-suffix", action="append", default=[])
    parser.add_argument("--emphasis-repeat", type=int, default=1)
    parser.add_argument("--emphasis-max-negatives", type=int)
    args = parser.parse_args()
    if args.max_negatives < 1:
        parser.error("--max-negatives must be positive")
    if args.test_modulo < 2:
        parser.error("--test-modulo must be at least two")
    if not 0 <= args.test_remainder < args.test_modulo:
        parser.error("--test-remainder must be in [0, test-modulo)")
    if args.emphasis_repeat < 1:
        parser.error("--emphasis-repeat must be positive")
    if args.emphasis_max_negatives is not None and args.emphasis_max_negatives < 1:
        parser.error("--emphasis-max-negatives must be positive")

    source = torch.load(
        args.trajectory_data, map_location="cpu", weights_only=False
    )
    train, test, metadata = collect_preferences(
        source,
        max_negatives=args.max_negatives,
        test_modulo=args.test_modulo,
        test_remainder=args.test_remainder,
        force_train_suffixes=tuple(args.force_train_suffix),
        emphasis_suffixes=tuple(args.emphasis_suffix),
        emphasis_repeat=args.emphasis_repeat,
        emphasis_max_negatives=args.emphasis_max_negatives,
    )
    if not train or not test:
        raise ValueError("path-level split produced an empty preference split")
    result = {
        key: source[key]
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
            "train_preferences": train,
            "test_preferences": test,
            "metadata": metadata,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    rendered = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    args.output.with_suffix(".metadata.json").write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
