from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import RuleMetadata, validate_trajectory


METADATA_KEYS = (
    "format",
    "source_patterns",
    "xfer_to_source",
    "xfer_sources",
    "xfer_destinations",
)


def apply_delta(snapshot: dict, delta: dict) -> dict:
    nodes = {
        int(slot): (int(gate_type), int(guid))
        for slot, gate_type, guid in snapshot["nodes"]
    }
    edges = {tuple(map(int, edge)) for edge in snapshot["edges"]}
    for slot in delta["removed_slots"]:
        nodes.pop(int(slot))
    for slot, gate_type, guid in delta["added_nodes"]:
        nodes[int(slot)] = (int(gate_type), int(guid))
    edges.difference_update(tuple(map(int, edge)) for edge in delta["removed_edges"])
    edges.update(tuple(map(int, edge)) for edge in delta["added_edges"])
    return {
        "nodes": sorted(
            (slot, gate_type, guid)
            for slot, (gate_type, guid) in nodes.items()
        ),
        "edges": sorted(edges),
    }


def window_trajectory(trajectory: dict, max_actions: int) -> list[dict]:
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    steps = trajectory["steps"]
    if not steps:
        return []
    snapshots = [trajectory["initial_graph"]]
    for step in steps:
        snapshots.append(apply_delta(snapshots[-1], step["delta"]))

    output = []
    for begin in range(0, len(steps), max_actions):
        end = min(len(steps), begin + max_actions)
        terminal_matches = (
            trajectory["terminal_matches"]
            if end == len(steps)
            else steps[end]["matches"]
        )
        output.append(
            {
                **{
                    key: value
                    for key, value in trajectory.items()
                    if key
                    not in {
                        "initial_graph",
                        "steps",
                        "terminal_matches",
                        "terminal_match_ms",
                        "trajectory_id",
                    }
                },
                "source_path": f"{trajectory.get('source_path', '')}#window={begin}:{end}",
                "initial_graph": snapshots[begin],
                "steps": [dict(step, index=index) for index, step in enumerate(steps[begin:end])],
                "terminal_matches": terminal_matches,
                "requested_history_length": end - begin,
                "valid_history_length": end - begin,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split long exact trajectories into independently replayable windows."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, default=64)
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    train = []
    test = []
    input_max = 0
    for split_name, destination in (
        ("train_trajectories", train),
        ("test_trajectories", test),
    ):
        for trajectory in payload[split_name]:
            input_max = max(input_max, len(trajectory["steps"]))
            for window in window_trajectory(trajectory, args.max_actions):
                window["trajectory_id"] = len(destination)
                validate_trajectory(window, rules)
                destination.append(window)

    result = {key: payload[key] for key in METADATA_KEYS}
    result.update(
        {
            "train_trajectories": train,
            "test_trajectories": test,
            "metadata": {
                "kind": "windowed_trajectory_dataset",
                "input": str(args.input),
                "input_metadata": payload.get("metadata"),
                "max_actions": args.max_actions,
                "input_max_actions": input_max,
                "train_windows": len(train),
                "test_windows": len(test),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(
        f"saved={args.output} train_windows={len(train)} "
        f"test_windows={len(test)} input_max_actions={input_max} "
        f"output_max_actions={max((len(t['steps']) for t in train + test), default=0)}"
    )


if __name__ == "__main__":
    main()
