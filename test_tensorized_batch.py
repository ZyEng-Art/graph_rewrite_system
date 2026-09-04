from __future__ import annotations

import numpy as np
import torch

from beam_search_benchmark import BeamState, collate_states
from lazy_rollout_benchmark import indexed_topology
from tensorized_batch import collate_paged_states


def _state(index: int) -> BeamState:
    nodes = [(0, 1, -1), (2 + index, 2, -1), (5 + index, 3, -1)]
    edges = [(0, 2 + index, 0, 1), (2 + index, 5 + index, 1, 0)]
    return BeamState(
        graph=None,
        snapshot={"nodes": nodes, "edges": edges},
        guid_to_slot={},
        next_slot=6 + index,
        last_touched={0: index, 2 + index: max(0, index - 4)},
        rewrite_distance={0: 0, 2 + index: 1, 5 + index: 5},
        previous_preferred=set(),
        local_streak=index,
        gate_count=3,
        depth=index + 1,
        history=(),
    )


def main() -> None:
    states = [_state(0), _state(1), _state(4)]
    expected = collate_states(states)
    for state in states:
        state.topology_index = indexed_topology(state.snapshot)
        state.snapshot = None
    slots = expected["current_types"].shape[1] + 3
    current_types = torch.nn.functional.pad(
        expected["current_types"], (0, 3), value=-1
    )
    actual = collate_paged_states(states, current_types)
    assert torch.equal(actual["current_types"], current_types)
    for name in (
        "current_rewrite_distance",
        "current_touch_age",
    ):
        padded = torch.nn.functional.pad(
            expected[name], (0, slots - expected[name].shape[1]), value=5 if name.endswith("distance") else 7
        )
        assert torch.equal(actual[name], padded), name
    for name in (
        "current_edge_batch",
        "current_edge_src",
        "current_edge_dst",
        "current_edge_relation",
        "current_local_streak",
    ):
        assert torch.equal(actual[name], expected[name]), name
    for state in states:
        dense_distance = np.full(state.next_slot, 5, dtype=np.int8)
        for slot, distance in state.rewrite_distance.items():
            dense_distance[slot] = distance
        dense_touched = np.full(state.next_slot, -1, dtype=np.int16)
        for slot, step in state.last_touched.items():
            dense_touched[slot] = step
        state.rewrite_distance = dense_distance
        state.last_touched = dense_touched
    dense_actual = collate_paged_states(states, current_types)
    for name, value in actual.items():
        assert torch.equal(dense_actual[name], value), f"dense {name}"
    print("tensorized paged batch matches legacy collation")


if __name__ == "__main__":
    main()
