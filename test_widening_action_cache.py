from __future__ import annotations

import unittest

import torch

from gpu_proposals import GpuRuleIndex, build_gpu_proposals
from search_types import BeamState
from threshold_inference import CandidateTensors
from widening_action_cache import (
    WideningActionCache,
    build_ranked_parent_entries,
    select_rank_bands,
)


def state(gate_count: int, expansion_round: int = 0) -> BeamState:
    return BeamState(
        graph=object(),
        snapshot={"nodes": [], "edges": []},
        guid_to_slot={},
        next_slot=0,
        last_touched={},
        rewrite_distance={},
        previous_preferred=set(),
        local_streak=0,
        gate_count=gate_count,
        depth=0,
        history=(),
        expansion_round=expansion_round,
    )


def proposal_key(proposal):
    return (
        proposal.parent,
        proposal.xfer_id,
        proposal.anchor_slot,
        proposal.binding,
        round(proposal.probability, 6),
        proposal.next_gate_count,
        proposal.parent_rank,
    )


class WideningActionCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beam = [state(20), state(18, 1), state(21)]
        self.candidates = CandidateTensors(
            batch_ids=torch.tensor(
                [0, 0, 0, 1, 1, 1, 2, 2, 2], device=self.device
            ),
            sources=torch.tensor(
                [0, 1, 2, 0, 1, 2, 0, 1, 2], device=self.device
            ),
            anchors=torch.tensor(
                [2, 4, 6, 1, 3, 5, 7, 8, 9], device=self.device
            ),
            bindings=torch.tensor(
                [[2, 3], [4, -1], [6, -1], [1, 5], [3, 7], [5, -1],
                 [7, -1], [8, -1], [9, -1]],
                device=self.device,
            ),
            probabilities=torch.tensor(
                [0.7, 0.9, 0.4, 0.8, 0.6, 0.3, 0.95, 0.5, 0.2],
                device=self.device,
            ),
        )
        self.rule_index = GpuRuleIndex.build(
            {0: [0, 1], 1: [2], 2: [3, 4]},
            [0, -1, 1, 2, -2],
            num_sources=3,
            max_gate_increase=2,
            device=self.device,
        )

    def test_cached_rank_bands_match_direct_gpu_selection(self) -> None:
        ranked_pool = []
        direct, direct_metrics, _, _ = build_gpu_proposals(
            self.candidates,
            self.beam,
            self.rule_index,
            per_parent_cap=2,
            global_cap=6,
            parent_rank_offsets=[0, 2, 0],
            preserve_parent_best=True,
            parent_diversity_actions=2,
            ranked_pool_cap=4,
            ranked_pool_output=ranked_pool,
        )
        self.assertEqual(len(ranked_pool), 1)
        entries = build_ranked_parent_entries(
            self.beam, self.candidates, ranked_pool[0], self.rule_index
        )
        cached, cached_metrics, _ = select_rank_bands(
            self.beam,
            entries,
            per_parent_cap=2,
            global_cap=6,
            parent_diversity_actions=2,
        )
        self.assertEqual(list(map(proposal_key, direct)), list(map(proposal_key, cached)))
        for key in (
            "predicted_actions",
            "eligible_actions",
            "selected_actions",
            "parent_rank_offset_min",
            "parent_rank_offset_max",
            "selected_parent_rank_min",
            "selected_parent_rank_max",
            "rank_band_candidates",
            "selected_actions_from_widened_parents",
            "selected_parent_best_actions",
        ):
            self.assertEqual(direct_metrics[key], cached_metrics[key], key)

    def test_cache_retains_only_scheduled_exact_parents(self) -> None:
        ranked_pool = []
        build_gpu_proposals(
            self.candidates,
            self.beam,
            self.rule_index,
            per_parent_cap=2,
            global_cap=6,
            ranked_pool_cap=4,
            ranked_pool_output=ranked_pool,
        )
        fresh = build_ranked_parent_entries(
            self.beam, self.candidates, ranked_pool[0], self.rule_index
        )
        cache = WideningActionCache(enabled=True)
        entries, metrics = cache.resolve_entries(self.beam, [0, 1, 2], fresh)
        self.assertEqual(metrics["parent_hits"], 0)
        retained = cache.retain(self.beam, entries, [0, 2])
        self.assertEqual(retained["resident_parents_after"], 2)
        revisits = [self.beam[2], state(19), self.beam[0]]
        self.assertEqual(cache.miss_indices(revisits), [1])


if __name__ == "__main__":
    unittest.main()
