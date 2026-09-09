from __future__ import annotations

from dataclasses import dataclass
import unittest

import torch

from threshold_inference import CandidateTensors
from widening_candidate_cache import (
    WideningCandidateCache,
    candidate_tensor_bytes,
    split_candidate_tensors,
)


@dataclass
class State:
    graph: object


def candidates(batch_ids: list[int]) -> CandidateTensors:
    count = len(batch_ids)
    values = torch.arange(count, dtype=torch.long)
    return CandidateTensors(
        batch_ids=torch.tensor(batch_ids, dtype=torch.long),
        sources=values + 10,
        anchors=values + 20,
        bindings=torch.stack((values + 30, values + 40), dim=1),
        probabilities=(values + 1).float() / 10,
    )


class WideningCandidateCacheTest(unittest.TestCase):
    def test_split_preserves_parent_row_order(self) -> None:
        rows = candidates([0, 1, 0, 2, 1])
        split = split_candidate_tensors(rows, 3)
        self.assertEqual(split[0].sources.tolist(), [10, 12])
        self.assertEqual(split[1].sources.tolist(), [11, 14])
        self.assertEqual(split[2].sources.tolist(), [13])
        self.assertEqual(split[0].batch_ids.tolist(), [0, 0])

    def test_revisit_reuses_rows_and_restores_beam_batch_ids(self) -> None:
        graphs = [object(), object(), object()]
        initial = [State(graph) for graph in graphs]
        cache = WideningCandidateCache(enabled=True)
        first, retained_candidates, first_metrics = cache.resolve(
            initial,
            miss_indices=[0, 1, 2],
            fresh_candidates=candidates([0, 0, 1, 2, 2]),
        )
        self.assertEqual(first.batch_ids.tolist(), [0, 0, 1, 2, 2])
        self.assertEqual(first_metrics["parent_hits"], 0)
        retained = cache.retain(initial, retained_candidates, [0, 2])
        self.assertEqual(retained["resident_parents_after"], 2)

        child_graph = object()
        next_states = [State(graphs[2]), State(child_graph), State(graphs[0])]
        self.assertEqual(cache.miss_indices(next_states), [1])
        combined, retained_candidates, metrics = cache.resolve(
            next_states,
            miss_indices=[1],
            fresh_candidates=candidates([0, 0]),
        )
        self.assertEqual(combined.batch_ids.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(combined.sources.tolist(), [13, 14, 10, 11, 10, 11])
        self.assertEqual(metrics["parent_hits"], 2)
        self.assertEqual(metrics["candidate_rows_reused"], 4)
        self.assertEqual(metrics["candidate_rows_generated"], 2)

        cache.retain(next_states, retained_candidates, [1])
        self.assertEqual(cache.miss_indices([State(graphs[0]), State(child_graph)]), [0])

    def test_entry_owns_graph_and_reports_tensor_bytes(self) -> None:
        graph = object()
        cache = WideningCandidateCache(enabled=True)
        rows = candidates([0, 0])
        _, retained_candidates, _ = cache.resolve(
            [State(graph)], miss_indices=[0], fresh_candidates=rows
        )
        metrics = cache.retain([State(graph)], retained_candidates, [0])
        self.assertEqual(metrics["resident_rows_after"], 2)
        self.assertEqual(
            metrics["resident_bytes_after"], candidate_tensor_bytes(rows)
        )

    def test_rejects_inconsistent_fresh_batch(self) -> None:
        cache = WideningCandidateCache(enabled=True)
        with self.assertRaisesRegex(ValueError, "exactly for cache misses"):
            cache.resolve([State(object())], miss_indices=[0], fresh_candidates=None)

    def test_parent_with_no_candidates_round_trips_through_cache(self) -> None:
        graphs = [object(), object()]
        states = [State(graph) for graph in graphs]
        cache = WideningCandidateCache(enabled=True)
        combined, retained_candidates, _ = cache.resolve(
            states, miss_indices=[0, 1], fresh_candidates=candidates([1, 1])
        )
        self.assertEqual(combined.batch_ids.tolist(), [1, 1])
        cache.retain(states, retained_candidates, [0])

        reused, _, metrics = cache.resolve(
            [State(graphs[0])], miss_indices=[], fresh_candidates=None
        )
        self.assertEqual(reused.sources.numel(), 0)
        self.assertEqual(metrics["parent_hits"], 1)


if __name__ == "__main__":
    unittest.main()
