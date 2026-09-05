from __future__ import annotations

import unittest

import torch

from audit_hierarchical_actions import selected_source_ranks, stable_descending_ranks


class HierarchicalActionAuditTest(unittest.TestCase):
    def test_stable_descending_ranks_break_ties_by_column(self) -> None:
        scores = torch.tensor([[1.0, 3.0, 3.0, -1.0], [4.0, 2.0, 5.0, 3.0]])
        self.assertTrue(
            torch.equal(
                stable_descending_ranks(scores),
                torch.tensor([[3, 1, 2, 4], [2, 4, 1, 3]]),
            )
        )

    def test_selected_source_ranks_ignore_ineligible_columns(self) -> None:
        logits = torch.tensor(
            [
                [
                    [5.0, 3.0, 3.0, 8.0],
                    [1.0, 7.0, 2.0, 0.0],
                ]
            ]
        )
        eligible = torch.tensor(
            [
                [
                    [True, True, True, False],
                    [True, True, True, True],
                ]
            ]
        )
        ranks = selected_source_ranks(
            logits,
            eligible,
            torch.tensor([0, 0, 0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([1, 2, 2]),
        )
        self.assertTrue(torch.equal(ranks, torch.tensor([2, 3, 2])))


if __name__ == "__main__":
    unittest.main()
