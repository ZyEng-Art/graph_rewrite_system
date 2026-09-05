from __future__ import annotations

import unittest

import torch

from threshold_inference import (
    threshold_candidate_tensors,
    threshold_candidate_tensors_chunked,
)


class _ChunkedMatchModel:
    def __init__(self, logits: torch.Tensor, eligible: torch.Tensor):
        self.logits = logits
        self.eligible = eligible
        self.num_sources = logits.shape[-1]

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


if __name__ == "__main__":
    unittest.main()
