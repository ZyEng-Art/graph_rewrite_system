from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from incremental_graph import IncrementalCircuit, parse_pattern


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    sources = [parse_pattern(pattern) for pattern in payload["xfer_sources"]]
    destinations = [parse_pattern(pattern) for pattern in payload["xfer_destinations"]]
    trajectories = payload["train_trajectories"] + payload["test_trajectories"]
    elapsed = 0.0
    actions = 0
    for _ in range(args.repeats):
        for trajectory in trajectories:
            circuit = IncrementalCircuit(trajectory["initial_graph"])
            for step in trajectory["steps"]:
                action = step["action"]
                xfer_id = int(action["xfer_id"])
                started = time.perf_counter()
                circuit.apply(
                    sources[xfer_id],
                    destinations[xfer_id],
                    tuple(map(int, action["binding_slots"])),
                    tuple(map(int, action["dst_slots"])),
                )
                elapsed += time.perf_counter() - started
                actions += 1
    print(
        f"incremental apply: actions={actions} "
        f"ms/action={1000.0 * elapsed / actions:.6f}"
    )


if __name__ == "__main__":
    main()
