from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch


def state_rows(payload: dict) -> list[dict]:
    rows = []
    for trajectory in payload["train_trajectories"] + payload["test_trajectories"]:
        source_path = trajectory.get("source_path", "")
        for index, step in enumerate(trajectory["steps"]):
            action = step["action"]
            rows.append(
                {
                    "graph_hash": int(step["graph_hash"]),
                    "source_path": source_path,
                    "step": index,
                    "terminal": False,
                    "xfer_id": int(action["xfer_id"]),
                    "source_id": int(action["source_id"]),
                }
            )
        terminal_hash = trajectory.get("terminal_graph_hash")
        if terminal_hash is not None:
            rows.append(
                {
                    "graph_hash": int(terminal_hash),
                    "source_path": source_path,
                    "step": len(trajectory["steps"]),
                    "terminal": True,
                    "xfer_id": None,
                    "source_id": None,
                }
            )
    return rows


def grouped(rows: list[dict]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        result[row["graph_hash"]].append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--holdout-data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-state-overlap", action="store_true")
    parser.add_argument("--fail-on-supervised-state-overlap", action="store_true")
    parser.add_argument("--fail-on-action-overlap", action="store_true")
    args = parser.parse_args()

    train = torch.load(args.train_data, map_location="cpu", weights_only=False)
    holdout = torch.load(args.holdout_data, map_location="cpu", weights_only=False)
    train_rows = state_rows(train)
    holdout_rows = state_rows(holdout)
    train_by_hash = grouped(train_rows)
    holdout_by_hash = grouped(holdout_rows)
    overlap_hashes = sorted(set(train_by_hash) & set(holdout_by_hash))
    train_supervised_hashes = {
        row["graph_hash"] for row in train_rows if not row["terminal"]
    }
    holdout_supervised_hashes = {
        row["graph_hash"] for row in holdout_rows if not row["terminal"]
    }
    supervised_overlap_hashes = sorted(
        train_supervised_hashes & holdout_supervised_hashes
    )

    overlaps = []
    action_overlap_count = 0
    train_paths = set()
    for graph_hash in overlap_hashes:
        train_state_rows = train_by_hash[graph_hash]
        holdout_state_rows = holdout_by_hash[graph_hash]
        train_action_keys = {
            (row["xfer_id"], row["source_id"])
            for row in train_state_rows
            if not row["terminal"]
        }
        holdout_action_keys = {
            (row["xfer_id"], row["source_id"])
            for row in holdout_state_rows
            if not row["terminal"]
        }
        action_keys = sorted(train_action_keys & holdout_action_keys)
        action_overlap_count += len(action_keys)
        train_paths.update(row["source_path"] for row in train_state_rows)
        overlaps.append(
            {
                "graph_hash": graph_hash,
                "shared_action_keys": action_keys,
                "train": train_state_rows,
                "holdout": holdout_state_rows,
            }
        )

    result = {
        "train_data": str(args.train_data),
        "holdout_data": str(args.holdout_data),
        "train_state_rows": len(train_rows),
        "holdout_state_rows": len(holdout_rows),
        "overlapping_graph_hashes": len(overlap_hashes),
        "overlapping_supervised_graph_hashes": len(supervised_overlap_hashes),
        "shared_action_keys": action_overlap_count,
        "train_paths_with_overlap": sorted(train_paths),
        "overlaps": overlaps,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if args.fail_on_state_overlap and overlap_hashes:
        raise SystemExit(3)
    if args.fail_on_supervised_state_overlap and supervised_overlap_hashes:
        raise SystemExit(4)
    if args.fail_on_action_overlap and action_overlap_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
