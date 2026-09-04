from __future__ import annotations

import argparse
from pathlib import Path

import torch

from incremental_graph import IncrementalCircuit, parse_pattern, structural_binding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    sources = [parse_pattern(pattern) for pattern in payload["source_patterns"]]
    correct = 0
    total = 0
    disconnected = 0
    for trajectory in payload["train_trajectories"] + payload["test_trajectories"]:
        circuit = IncrementalCircuit(trajectory["initial_graph"])
        for step in trajectory["steps"]:
            for match in step["matches"]:
                source_id = int(match["source_id"])
                expected = tuple(map(int, match["binding_slots"]))
                actual = structural_binding(circuit, sources[source_id], expected[0])
                correct += int(actual == expected)
                disconnected += int(actual is None)
                total += 1
            action = step["action"]
            xfer_id = int(action["xfer_id"])
            circuit.apply(
                parse_pattern(payload["xfer_sources"][xfer_id]),
                parse_pattern(payload["xfer_destinations"][xfer_id]),
                tuple(map(int, action["binding_slots"])),
                tuple(map(int, action["dst_slots"])),
            )
    print(
        f"structural binding: correct={correct}/{total} ({correct / total:.4%}) "
        f"failed={disconnected}"
    )


if __name__ == "__main__":
    main()
