from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from beam_search_benchmark import BeamState
from dataset import _local_streak_bucket


def _touch_age_buckets(ages: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of dataset._touch_age_bucket."""
    return np.where(
        ages <= 3,
        ages,
        np.where(ages <= 7, 4, np.where(ages <= 15, 5, 6)),
    )


def collate_paged_states(
    states: Sequence[BeamState],
    current_types: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build only metadata not already maintained by the paged GPU state.

    ``current_types`` is the authoritative action-derived tensor.  The legacy
    collation path rebuilt it (and compact live slots) with hundreds of
    thousands of Python-to-Torch scalar assignments at every search level.
    This path reuses that tensor and fills locality/edge arrays through NumPy
    bulk operations before the single H2D move performed by the caller.

    It is intended for direct paged readout, which does not consume
    ``current_live_slots``.
    """
    batch_size, num_slots = current_types.shape
    if len(states) != batch_size:
        raise ValueError("state list and current_types batch differ")

    rewrite_distance = np.full((batch_size, num_slots), 5, dtype=np.int64)
    touch_age = np.full((batch_size, num_slots), 7, dtype=np.int64)
    local_streak = np.empty(batch_size, dtype=np.int64)
    state_edges = [
        state.topology_index.edges
        if state.topology_index is not None
        else state.snapshot["edges"]
        for state in states
    ]
    edge_counts = np.fromiter(
        (len(edges) for edges in state_edges),
        dtype=np.int64,
        count=batch_size,
    )

    for batch_index, state in enumerate(states):
        if isinstance(state.rewrite_distance, np.ndarray):
            if len(state.rewrite_distance) > num_slots:
                raise RuntimeError("locality slot is absent from the paged state")
            rewrite_distance[
                batch_index, : len(state.rewrite_distance)
            ] = state.rewrite_distance
        elif state.rewrite_distance:
            slots = np.fromiter(
                state.rewrite_distance.keys(),
                dtype=np.int64,
                count=len(state.rewrite_distance),
            )
            if slots.size and int(slots.max()) >= num_slots:
                raise RuntimeError("locality slot is absent from the paged state")
            rewrite_distance[batch_index, slots] = np.fromiter(
                state.rewrite_distance.values(),
                dtype=np.int64,
                count=len(state.rewrite_distance),
            )
        if isinstance(state.last_touched, np.ndarray):
            if len(state.last_touched) > num_slots:
                raise RuntimeError("touched slot is absent from the paged state")
            touched_slots = np.flatnonzero(state.last_touched >= 0)
            touched_steps = state.last_touched[touched_slots]
            ages = state.depth - 1 - touched_steps
            touch_age[batch_index, touched_slots] = _touch_age_buckets(ages)
        elif state.last_touched:
            touched_slots = np.fromiter(
                state.last_touched.keys(),
                dtype=np.int64,
                count=len(state.last_touched),
            )
            if touched_slots.size and int(touched_slots.max()) >= num_slots:
                raise RuntimeError("touched slot is absent from the paged state")
            touched_steps = np.fromiter(
                state.last_touched.values(),
                dtype=np.int64,
                count=len(state.last_touched),
            )
            ages = state.depth - 1 - touched_steps
            touch_age[batch_index, touched_slots] = _touch_age_buckets(ages)
        local_streak[batch_index] = _local_streak_bucket(
            bool(state.depth), state.local_streak
        )

    total_edges = int(edge_counts.sum())
    if total_edges:
        flat_edges = np.asarray(
            [edge for edges in state_edges for edge in edges],
            dtype=np.int64,
        )
        if flat_edges.shape != (total_edges, 4):
            raise RuntimeError("unexpected edge array shape")
        edge_batch = np.repeat(
            np.arange(batch_size, dtype=np.int64), edge_counts
        )
        # Preserve the canonical per-state edge order used by the original
        # snapshot path, but sort the whole packed array in NumPy rather than
        # invoking Python's sorted() once for every beam state.
        order = np.lexsort(
            (
                flat_edges[:, 3],
                flat_edges[:, 2],
                flat_edges[:, 1],
                flat_edges[:, 0],
                edge_batch,
            )
        )
        flat_edges = flat_edges[order]
        edge_batch = edge_batch[order]
        edge_src = flat_edges[:, 0]
        edge_dst = flat_edges[:, 1]
        edge_relation = flat_edges[:, 2] * 4 + flat_edges[:, 3]
    else:
        edge_batch = np.empty(0, dtype=np.int64)
        edge_src = np.empty(0, dtype=np.int64)
        edge_dst = np.empty(0, dtype=np.int64)
        edge_relation = np.empty(0, dtype=np.int64)

    return {
        "current_types": current_types,
        "current_edge_batch": torch.from_numpy(edge_batch),
        "current_edge_src": torch.from_numpy(edge_src),
        "current_edge_dst": torch.from_numpy(edge_dst),
        "current_edge_relation": torch.from_numpy(edge_relation),
        "current_rewrite_distance": torch.from_numpy(rewrite_distance),
        "current_touch_age": torch.from_numpy(touch_age),
        "current_local_streak": torch.from_numpy(local_streak),
    }
