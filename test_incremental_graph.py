from __future__ import annotations

import argparse
from pathlib import Path

import torch

from incremental_graph import IncrementalCircuit, parse_pattern


def apply_teacher_delta(nodes: dict[int, int], edges: set[tuple], delta: dict):
    for slot in delta["removed_slots"]:
        nodes.pop(int(slot))
    for slot, gate_type, _ in delta["added_nodes"]:
        nodes[int(slot)] = int(gate_type)
    edges.difference_update(map(tuple, delta["removed_edges"]))
    edges.update(map(tuple, delta["added_edges"]))
    live = set(nodes)
    edges.intersection_update(
        edge for edge in edges if edge[0] in live and edge[1] in live
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    sources = [parse_pattern(pattern) for pattern in payload["xfer_sources"]]
    destinations = [parse_pattern(pattern) for pattern in payload["xfer_destinations"]]
    checked = 0
    for trajectory in payload["train_trajectories"] + payload["test_trajectories"]:
        circuit = IncrementalCircuit(trajectory["initial_graph"])
        expected_nodes = {
            int(slot): int(gate_type)
            for slot, gate_type, _ in trajectory["initial_graph"]["nodes"]
        }
        expected_edges = set(map(tuple, trajectory["initial_graph"]["edges"]))
        for step in trajectory["steps"]:
            action = step["action"]
            xfer_id = int(action["xfer_id"])
            if "dst_types" in action:
                circuit.apply_delta(step["delta"])
            else:
                circuit.apply(
                    sources[xfer_id],
                    destinations[xfer_id],
                    tuple(map(int, action["binding_slots"])),
                    tuple(map(int, action["dst_slots"])),
                )
            apply_teacher_delta(expected_nodes, expected_edges, step["delta"])
            actual_nodes, actual_edges = circuit.snapshot_without_guids()
            if actual_nodes != expected_nodes or actual_edges != expected_edges:
                missing = expected_edges - actual_edges
                extra = actual_edges - expected_edges
                raise AssertionError(
                    f"incremental graph mismatch at xfer {xfer_id}: "
                    f"missing={list(missing)[:4]} extra={list(extra)[:4]}"
                )
            checked += 1
    print(f"incremental graph ok: exact transitions={checked}")


if __name__ == "__main__":
    main()
