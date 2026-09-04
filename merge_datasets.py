from __future__ import annotations

import argparse
from pathlib import Path

import torch


METADATA_KEYS = (
    "format",
    "source_patterns",
    "xfer_to_source",
    "xfer_sources",
    "xfer_destinations",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    reference = payloads[0]
    for payload in payloads[1:]:
        for key in METADATA_KEYS:
            if payload[key] != reference[key]:
                raise ValueError(f"dataset metadata differs at {key}")
    train = []
    test = []
    for dataset_index, payload in enumerate(payloads):
        for split_name, destination in (
            ("train_trajectories", train),
            ("test_trajectories", test),
        ):
            for trajectory in payload[split_name]:
                trajectory = dict(trajectory)
                trajectory["source_dataset"] = dataset_index
                trajectory["trajectory_id"] = len(destination)
                destination.append(trajectory)
    merged = {key: reference[key] for key in METADATA_KEYS}
    merged.update(
        {
            "train_trajectories": train,
            "test_trajectories": test,
            "metadata": {
                "merged_inputs": [str(path) for path in args.inputs],
                "component_metadata": [payload["metadata"] for payload in payloads],
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(
        f"merged train_states={sum(len(t['steps']) for t in train)} "
        f"test_states={sum(len(t['steps']) for t in test)} "
        f"train_trajectories={len(train)} test_trajectories={len(test)}"
    )


if __name__ == "__main__":
    main()
