from __future__ import annotations

import argparse
from pathlib import Path

import torch

from incremental_graph import IncrementalCircuit, parse_pattern
from lazy_rollout_benchmark import (
    distances_from_index,
    indexed_topology,
    materialize_indexed_rewrite,
    plan_indexed_rewrite,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    sources = tuple(parse_pattern(row) for row in payload["xfer_sources"])
    destinations = tuple(parse_pattern(row) for row in payload["xfer_destinations"])
    checked = 0
    trajectories = payload["train_trajectories"] + payload["test_trajectories"]
    for trajectory in trajectories:
        topology = indexed_topology(trajectory["initial_graph"])
        circuit = IncrementalCircuit(trajectory["initial_graph"])
        for step in trajectory["steps"]:
            action = step["action"]
            xfer_id = int(action["xfer_id"])
            source_slots = tuple(map(int, action["binding_slots"]))
            destination_slots = tuple(map(int, action["dst_slots"]))
            rewrite = plan_indexed_rewrite(
                topology,
                sources[xfer_id],
                destinations[xfer_id],
                source_slots,
                destination_slots,
            )
            topology = materialize_indexed_rewrite(
                topology,
                destinations[xfer_id],
                source_slots,
                destination_slots,
                rewrite,
            )
            circuit.apply(
                sources[xfer_id],
                destinations[xfer_id],
                source_slots,
                destination_slots,
            )
            assert topology.nodes == circuit.nodes
            assert topology.edges == circuit.edges
            rebuilt = indexed_topology(
                {
                    "nodes": [
                        (slot, gate_type, -1)
                        for slot, gate_type in circuit.nodes.items()
                    ],
                    "edges": list(circuit.edges),
                }
            )
            assert topology.out_edge == rebuilt.out_edge
            assert topology.in_edge == rebuilt.in_edge
            assert {
                slot: tuple(sorted(neighbors))
                for slot, neighbors in topology.adjacency.items()
            } == {
                slot: tuple(sorted(neighbors))
                for slot, neighbors in rebuilt.adjacency.items()
            }
            assert topology.fingerprint == rebuilt.fingerprint
            core = set(destination_slots)
            changed_edges = rewrite.removed_edges.symmetric_difference(
                rewrite.added_edges
            )
            for src, dst, _, _ in changed_edges:
                if src in topology.nodes:
                    core.add(src)
                if dst in topology.nodes:
                    core.add(dst)
            expected_distances = {}
            queue = list(slot for slot in core if slot in rebuilt.nodes)
            expected_distances.update((slot, 0) for slot in queue)
            cursor = 0
            while cursor < len(queue):
                slot = queue[cursor]
                cursor += 1
                for neighbor in rebuilt.adjacency[slot]:
                    if neighbor not in expected_distances:
                        expected_distances[neighbor] = expected_distances[slot] + 1
                        queue.append(neighbor)
            expected_distances = {
                slot: min(expected_distances.get(slot, 5), 4)
                if slot in expected_distances
                else 5
                for slot in rebuilt.nodes
            }
            assert distances_from_index(topology, core) == expected_distances
            checked += 1
            if checked >= args.limit:
                print(f"indexed topology matches legacy transitions: {checked}")
                return
    print(f"indexed topology matches legacy transitions: {checked}")


if __name__ == "__main__":
    main()
