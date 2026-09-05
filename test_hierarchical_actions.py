from __future__ import annotations

import unittest

import torch

from hierarchical_actions import candidates_for_selected_nodes


class FakeMatcher:
    max_pattern = 3
    source_first_gate_groups = ((0, 0, 2), (1, 2, 3))
    source_first_gate_order = torch.tensor([0, 1, 2])

    def __init__(self, logits: torch.Tensor, eligible: torch.Tensor) -> None:
        self.logits = logits
        self.eligible = eligible

    def match_logits_from_node_vectors(
        self,
        node_vectors: torch.Tensor,
        live: torch.Tensor,
        gate_types: torch.Tensor,
        source_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.logits.to(node_vectors.device), self.eligible.to(node_vectors.device)

    def match_logits_for_sources(
        self,
        node_vectors: torch.Tensor,
        source_vectors: torch.Tensor,
        source_ids: torch.Tensor,
    ) -> torch.Tensor:
        node_positions = node_vectors[:, 0, 0].long()
        rows = self.logits[0].to(node_vectors.device).index_select(0, node_positions)
        return rows.index_select(1, source_ids).unsqueeze(1)

    def structural_decode(
        self,
        batch: dict,
        gate_types: torch.Tensor,
        live: torch.Tensor,
        batch_ids: torch.Tensor,
        source_ids: torch.Tensor,
        anchor_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bindings = torch.full(
            (batch_ids.numel(), self.max_pattern),
            -1,
            dtype=torch.long,
            device=batch_ids.device,
        )
        bindings[:, 0] = anchor_slots
        # Exercise filtering after Top-N without changing node-position alignment.
        return bindings, source_ids.ne(1)


def threshold_config(near: float, far: float) -> dict:
    return {
        "groups": {
            "near": {"scale": 1.0, "bias": 0.0, "raw_threshold": near},
            "far": {"scale": 1.0, "bias": 0.0, "raw_threshold": far},
        }
    }


class HierarchicalCandidateTest(unittest.TestCase):
    def test_topn_is_applied_per_selected_node_before_structural_decode(self) -> None:
        model = FakeMatcher(
            logits=torch.tensor([[[3.0, 2.0, 1.0], [1.0, 0.8, 0.7]]]),
            eligible=torch.ones(1, 2, 3, dtype=torch.bool),
        )
        batch = {
            "current_rewrite_distance": torch.tensor([[5, 5, 5, 5, 1]]),
            "current_types": torch.tensor([[0, 0, 0, 0, 0]]),
        }
        candidates, node_positions = candidates_for_selected_nodes(
            model,
            batch,
            node_vectors=torch.zeros(1, 2, 4),
            selected_nodes=torch.tensor([[4, 1]]),
            selected_node_mask=torch.tensor([[True, True]]),
            selected_gate_types=torch.tensor([[0, 0]]),
            source_vectors=torch.zeros(3, 4),
            threshold_config=threshold_config(near=2.5, far=0.75),
            pattern_k=2,
            batch_offset=7,
        )
        self.assertTrue(torch.equal(candidates.batch_ids, torch.tensor([7, 7])))
        self.assertTrue(torch.equal(candidates.sources, torch.tensor([0, 0])))
        self.assertTrue(torch.equal(candidates.anchors, torch.tensor([4, 1])))
        self.assertTrue(torch.equal(node_positions, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(candidates.bindings[:, 0], candidates.anchors))

    def test_first_gate_grouping_preserves_global_sources_and_node_positions(self) -> None:
        model = FakeMatcher(
            logits=torch.tensor([[[3.0, 2.0, 1.0], [1.0, 0.8, 0.9]]]),
            eligible=torch.ones(1, 2, 3, dtype=torch.bool),
        )
        candidates, node_positions = candidates_for_selected_nodes(
            model,
            {
                "current_rewrite_distance": torch.tensor([[5, 5, 5, 5, 1]]),
                "current_types": torch.zeros(1, 5, dtype=torch.long),
            },
            node_vectors=torch.tensor([[[0.0], [1.0]]]),
            selected_nodes=torch.tensor([[4, 1]]),
            selected_node_mask=torch.tensor([[True, True]]),
            selected_gate_types=torch.tensor([[0, 1]]),
            source_vectors=torch.zeros(3, 1),
            threshold_config=threshold_config(near=2.5, far=0.75),
            pattern_k=2,
            source_grouping="first_gate",
        )
        self.assertTrue(torch.equal(candidates.sources, torch.tensor([0, 2])))
        self.assertTrue(torch.equal(candidates.anchors, torch.tensor([4, 1])))
        self.assertTrue(torch.equal(node_positions, torch.tensor([0, 1])))

    def test_empty_threshold_result_has_stable_shapes(self) -> None:
        model = FakeMatcher(
            logits=torch.zeros(2, 2, 3),
            eligible=torch.ones(2, 2, 3, dtype=torch.bool),
        )
        candidates, node_positions = candidates_for_selected_nodes(
            model,
            {
                "current_rewrite_distance": torch.full((2, 3), 5),
                "current_types": torch.zeros(2, 3, dtype=torch.long),
            },
            node_vectors=torch.zeros(2, 2, 4),
            selected_nodes=torch.tensor([[0, 1], [1, 2]]),
            selected_node_mask=torch.tensor([[True, True], [True, False]]),
            selected_gate_types=torch.zeros(2, 2, dtype=torch.long),
            source_vectors=torch.zeros(3, 4),
            threshold_config=threshold_config(near=1.0, far=1.0),
            pattern_k=2,
        )
        self.assertEqual(candidates.bindings.shape, (0, 3))
        self.assertEqual(candidates.batch_ids.numel(), 0)
        self.assertEqual(node_positions.numel(), 0)


if __name__ == "__main__":
    unittest.main()
