from __future__ import annotations

import unittest

import torch

from threshold_inference import (
    threshold_candidate_tensors,
    threshold_candidate_tensors_chunked,
    threshold_candidate_tensors_grouped,
)


class _ChunkedMatchModel:
    def __init__(
        self,
        logits: torch.Tensor,
        eligible: torch.Tensor,
        source_first_types: torch.Tensor | None = None,
    ):
        self.logits = logits
        self.eligible = eligible
        self.num_sources = logits.shape[-1]
        self.max_pattern = 2
        if source_first_types is None:
            source_first_types = torch.zeros(self.num_sources, dtype=torch.long)
        ordered = []
        groups = []
        for gate_type in source_first_types.unique(sorted=True).tolist():
            source_ids = source_first_types.eq(gate_type).nonzero().squeeze(1).tolist()
            begin = len(ordered)
            ordered.extend(source_ids)
            groups.append((gate_type, begin, len(ordered)))
        self.source_first_gate_order = torch.tensor(ordered)
        self.source_first_gate_groups = tuple(groups)

    def match_logits_from_node_vectors(
        self,
        node_vectors,
        live,
        gate_types,
        source_vectors,
        *,
        source_begin,
        source_end,
    ):
        return (
            self.logits[:, :, source_begin:source_end],
            self.eligible[:, :, source_begin:source_end],
        )

    def structural_decode(
        self, batch, gate_types, live, batch_ids, source_ids, anchor_slots
    ):
        bindings = torch.stack((anchor_slots, source_ids), dim=1)
        return bindings, source_ids.remainder(3).ne(1)

    def match_logits_for_sources(
        self, node_vectors, source_vectors, source_ids
    ):
        batch_ids = node_vectors[..., 0].long()
        anchor_slots = node_vectors[..., 1].long()
        return self.logits[batch_ids, anchor_slots].index_select(2, source_ids)


class ChunkedThresholdTest(unittest.TestCase):
    def test_matches_full_global_topk_after_structural_decode(self):
        generator = torch.Generator().manual_seed(91)
        logits = torch.randn((3, 5, 11), generator=generator)
        eligible = torch.rand((3, 5, 11), generator=generator).gt(0.25)
        model = _ChunkedMatchModel(logits, eligible)
        batch = {
            "current_types": torch.zeros((3, 5), dtype=torch.long),
            "current_rewrite_distance": torch.tensor(
                [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 4, 1, 3, 2]]
            ),
        }
        config = {
            "groups": {
                "near": {"scale": 0.8, "bias": 0.1, "raw_threshold": -0.4},
                "far": {"scale": 1.2, "bias": -0.2, "raw_threshold": -0.1},
            }
        }
        full = threshold_candidate_tensors(
            model,
            batch,
            logits,
            eligible,
            config,
            max_candidates_per_state=7,
            batch_offset=13,
        )
        chunked = threshold_candidate_tensors_chunked(
            model,
            batch,
            torch.empty((3, 5, 2)),
            batch["current_types"].ge(0),
            batch["current_types"],
            torch.empty((11, 2)),
            config,
            source_chunk_size=4,
            max_candidates_per_state=7,
            batch_offset=13,
        )
        for field in ("batch_ids", "sources", "anchors", "bindings"):
            self.assertTrue(
                torch.equal(getattr(full, field), getattr(chunked, field)), field
            )
        torch.testing.assert_close(full.probabilities, chunked.probabilities)

    def test_first_gate_grouping_matches_dense_eligible_topk(self):
        generator = torch.Generator().manual_seed(117)
        logits = torch.randn((3, 6, 13), generator=generator)
        source_first_types = torch.tensor(
            [0, 1, 0, 2, 1, 2, 0, 1, 2, 2, 0, 1, 0]
        )
        gate_types = torch.tensor(
            [[0, 1, 2, 0, 1, -1], [2, 2, 1, 0, -1, -1], [1, 0, 1, 2, 0, 2]]
        )
        live = gate_types.ge(0)
        eligible = live.unsqueeze(-1) & gate_types.unsqueeze(-1).eq(
            source_first_types
        )
        model = _ChunkedMatchModel(logits, eligible, source_first_types)
        batch = {
            "current_types": gate_types,
            "current_rewrite_distance": torch.tensor(
                [[0, 1, 2, 3, 4, 5], [4, 3, 2, 1, 5, 5], [0, 4, 1, 3, 2, 4]]
            ),
        }
        config = {
            "groups": {
                "near": {"scale": 0.8, "bias": 0.1, "raw_threshold": -0.4},
                "far": {"scale": 1.2, "bias": -0.2, "raw_threshold": -0.1},
            }
        }
        full = threshold_candidate_tensors(
            model,
            batch,
            logits,
            eligible,
            config,
            max_candidates_per_state=9,
        )
        batch_ids = torch.arange(gate_types.shape[0]).unsqueeze(1).expand_as(gate_types)
        anchor_slots = torch.arange(gate_types.shape[1]).expand_as(gate_types)
        node_vectors = torch.stack((batch_ids, anchor_slots), dim=-1).float()
        grouped = threshold_candidate_tensors_grouped(
            model,
            batch,
            node_vectors,
            live,
            gate_types,
            torch.empty((source_first_types.numel(), 2)),
            config,
            source_chunk_size=4,
            max_candidates_per_state=9,
        )
        for field in ("batch_ids", "sources", "anchors", "bindings"):
            self.assertTrue(
                torch.equal(getattr(full, field), getattr(grouped, field)), field
            )
        torch.testing.assert_close(full.probabilities, grouped.probabilities)

    def test_first_gate_grouping_allows_a_batch_without_eligible_anchors(self):
        logits = torch.zeros((2, 3, 2))
        source_first_types = torch.tensor([0, 1])
        gate_types = torch.full((2, 3), 2, dtype=torch.long)
        live = torch.ones_like(gate_types, dtype=torch.bool)
        model = _ChunkedMatchModel(
            logits, torch.zeros_like(logits, dtype=torch.bool), source_first_types
        )
        batch = {
            "current_types": gate_types,
            "current_rewrite_distance": torch.zeros_like(gate_types),
        }
        config = {
            "groups": {
                "near": {"scale": 1.0, "bias": 0.0, "raw_threshold": 0.0},
                "far": {"scale": 1.0, "bias": 0.0, "raw_threshold": 0.0},
            }
        }
        grouped = threshold_candidate_tensors_grouped(
            model,
            batch,
            torch.zeros((2, 3, 2)),
            live,
            gate_types,
            torch.zeros((2, 2)),
            config,
            source_chunk_size=1,
        )
        self.assertEqual(grouped.batch_ids.numel(), 0)
        self.assertEqual(grouped.bindings.shape, (0, model.max_pattern))


if __name__ == "__main__":
    unittest.main()
