from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from beam_search_benchmark import continuation_ranker_scores
from sibling_continuation_ranker import SiblingContinuationRanker


class ContinuationRankerShadowTest(unittest.TestCase):
    def test_shadow_scores_align_with_proposal_parents(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ranker = SiblingContinuationRanker(
            12,
            hidden_width=8,
            dropout=0.0,
            base_probability_index=4,
            num_xfers=4,
            prefix_width=4,
        ).to(device)
        proposals = SimpleNamespace(
            parent_ids=torch.tensor([1, 0], device=device),
            probabilities=torch.tensor([0.8, 0.2], device=device),
            gate_deltas=torch.tensor([1, -1], device=device),
            parent_ranks=torch.tensor([7, 3], device=device),
        )
        beam = [
            SimpleNamespace(
                gate_count=20,
                expansion_round=1,
                stagnation_steps=2,
                depth=3,
                history=((2, 8),),
            ),
            SimpleNamespace(
                gate_count=21,
                expansion_round=2,
                stagnation_steps=4,
                depth=5,
                history=((999, 9), (3, 10)),
            ),
        ]
        scores = continuation_ranker_scores(
            ranker,
            torch.ones((2, 4), device=device),
            proposals,
            beam,
            step=6,
            device=device,
            batch_size=1,
            prefix_max_length=2,
        )
        self.assertTrue(
            torch.allclose(
                scores.cpu(),
                torch.logit(
                    proposals.probabilities.to(torch.float16).float()
                ).cpu(),
            )
        )


if __name__ == "__main__":
    unittest.main()
