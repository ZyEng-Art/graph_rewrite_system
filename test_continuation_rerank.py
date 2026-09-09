from __future__ import annotations

import unittest

import torch

from continuation_rerank import bounded_continuation_order
from gpu_proposals import SelectedProposalTensors, select_proposal_tensor_rows
from search_types import Proposal


def proposal(parent: int, gate: int, probability: float, xfer_id: int) -> Proposal:
    return Proposal(
        parent=parent,
        xfer_id=xfer_id,
        anchor_slot=xfer_id,
        binding=(),
        probability=probability,
        next_gate_count=gate,
    )


class BoundedContinuationRerankTest(unittest.TestCase):
    def test_promotes_one_near_tied_sibling_without_moving_bucket_positions(self) -> None:
        proposals = [
            proposal(0, 10, 0.52, 0),
            proposal(1, 10, 0.80, 1),
            proposal(0, 10, 0.50, 2),
            proposal(0, 11, 0.90, 3),
            proposal(0, 10, 0.49, 4),
        ]
        order, metrics = bounded_continuation_order(
            proposals,
            torch.tensor([0.0, 0.0, 2.0, 0.0, 0.1]),
            max_matcher_logit_gap=0.25,
            min_continuation_score_margin=0.5,
            max_promotions_per_parent=1,
        )
        self.assertEqual(order, [2, 1, 0, 3, 4])
        self.assertEqual(metrics["promotions"], 1)
        self.assertEqual(metrics["moved_rows"], 2)
        self.assertEqual(
            [(proposals[index].parent, proposals[index].next_gate_count) for index in order],
            [(row.parent, row.next_gate_count) for row in proposals],
        )

    def test_respects_matcher_and_continuation_margins(self) -> None:
        proposals = [proposal(0, 10, 0.9, 0), proposal(0, 10, 0.5, 1)]
        order, metrics = bounded_continuation_order(
            proposals,
            torch.tensor([0.0, 10.0]),
            max_matcher_logit_gap=0.25,
            min_continuation_score_margin=0.5,
            max_promotions_per_parent=1,
        )
        self.assertEqual(order, [0, 1])
        self.assertEqual(metrics["eligible_groups"], 0)

        near_ties = [proposal(0, 10, 0.52, 0), proposal(0, 10, 0.50, 1)]
        order, metrics = bounded_continuation_order(
            near_ties,
            torch.tensor([1.0, 1.49]),
            max_matcher_logit_gap=0.25,
            min_continuation_score_margin=0.5,
            max_promotions_per_parent=1,
        )
        self.assertEqual(order, [0, 1])
        self.assertEqual(metrics["promotions"], 0)

    def test_rejects_misaligned_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            bounded_continuation_order(
                [proposal(0, 10, 0.5, 0)],
                torch.tensor([]),
                max_matcher_logit_gap=0.25,
                min_continuation_score_margin=0.5,
                max_promotions_per_parent=1,
            )

    def test_tensor_rows_follow_the_same_permutation(self) -> None:
        rows = torch.arange(3)
        tensors = SelectedProposalTensors(
            parent_ids=rows,
            xfer_ids=rows + 10,
            source_ids=rows + 20,
            anchor_slots=rows + 30,
            bindings=torch.stack((rows, rows + 1), dim=1),
            probabilities=rows.float() + 0.1,
            gate_deltas=rows - 1,
            next_gate_counts=rows + 40,
            value_scores=rows.float() + 0.2,
            parent_ranks=rows + 50,
        )
        selected = select_proposal_tensor_rows(tensors, torch.tensor([2, 0, 1]))
        for name in SelectedProposalTensors.__dataclass_fields__:
            expected = getattr(tensors, name).index_select(0, torch.tensor([2, 0, 1]))
            self.assertTrue(torch.equal(getattr(selected, name), expected), name)


if __name__ == "__main__":
    unittest.main()
