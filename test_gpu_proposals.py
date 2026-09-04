from __future__ import annotations

import torch

from beam_search_benchmark import BeamState
from gpu_proposals import GpuRuleIndex, build_gpu_proposals
from paged_rollout_benchmark import build_legacy_proposals
from threshold_inference import CandidateTensors


def _beam(gate_count: int) -> BeamState:
    return BeamState(
        graph=None,
        snapshot={"nodes": [], "edges": []},
        guid_to_slot={},
        next_slot=0,
        last_touched={},
        rewrite_distance={},
        previous_preferred=set(),
        local_streak=0,
        gate_count=gate_count,
        depth=1,
        history=(),
    )


class _XferValueModel:
    def action_values(
        self, states, live, xfer_ids, source_ids, bindings, batch_ids
    ):
        return xfer_ids.float()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beam = [_beam(20), _beam(18), _beam(21)]
    source_to_xfers = {0: [0, 1], 1: [2], 2: [3, 4]}
    gate_deltas = [0, -1, 1, 2, -2]
    rows = [
        [(0, 2, (2, 3), 0.7), (1, 4, (4,), 0.9)],
        [(2, 1, (1, 5), 0.8), (0, 3, (3, 7), 0.6)],
        [(1, 6, (6,), 0.95), (2, 8, (8,), 0.5)],
    ]
    legacy, _ = build_legacy_proposals(
        beam,
        rows,
        None,
        source_to_xfers,
        gate_deltas,
        beam_size=2,
        max_actions_per_parent=2,
        exploration_actions_per_parent=0,
        proposal_factor=2,
        max_gate_increase=1,
    )
    flat = [
        (batch, source, anchor, binding, probability)
        for batch, batch_rows in enumerate(rows)
        for source, anchor, binding, probability in batch_rows
    ]
    max_pattern = max(len(row[3]) for row in flat)
    bindings = torch.full(
        (len(flat), max_pattern), -1, dtype=torch.long, device=device
    )
    for index, row in enumerate(flat):
        bindings[index, : len(row[3])] = torch.tensor(row[3], device=device)
    candidates = CandidateTensors(
        batch_ids=torch.tensor([row[0] for row in flat], device=device),
        sources=torch.tensor([row[1] for row in flat], device=device),
        anchors=torch.tensor([row[2] for row in flat], device=device),
        bindings=bindings,
        probabilities=torch.tensor([row[4] for row in flat], device=device),
    )
    rule_index = GpuRuleIndex.build(
        source_to_xfers,
        gate_deltas,
        num_sources=3,
        max_gate_increase=1,
        device=device,
    )
    actual, metrics, _ = build_gpu_proposals(
        candidates,
        beam,
        rule_index,
        per_parent_cap=2,
        global_cap=4,
    )
    simplify = lambda proposal: (
        proposal.parent,
        proposal.xfer_id,
        proposal.anchor_slot,
        proposal.binding,
        round(proposal.probability, 5),
        proposal.next_gate_count,
    )
    assert list(map(simplify, actual)) == list(map(simplify, legacy))
    assert metrics == {"predicted_actions": 10, "eligible_actions": 8}

    probability_legacy, _ = build_legacy_proposals(
        beam,
        rows,
        None,
        source_to_xfers,
        gate_deltas,
        beam_size=2,
        max_actions_per_parent=2,
        exploration_actions_per_parent=0,
        proposal_factor=2,
        max_gate_increase=1,
        ranking_mode="probability",
    )
    probability_actual, _, _ = build_gpu_proposals(
        candidates,
        beam,
        rule_index,
        per_parent_cap=2,
        global_cap=4,
        ranking_mode="probability",
    )
    assert list(map(simplify, probability_actual)) == list(
        map(simplify, probability_legacy)
    )
    assert [row.probability for row in probability_actual] == sorted(
        (row.probability for row in probability_actual), reverse=True
    )

    value_actual, value_metrics, _ = build_gpu_proposals(
        candidates,
        beam,
        rule_index,
        per_parent_cap=8,
        global_cap=4,
        ranking_mode="value",
        action_value_model=_XferValueModel(),
        action_value_states=torch.zeros((3, 9, 1), device=device),
        action_value_live=torch.ones((3, 9), dtype=torch.bool, device=device),
        action_value_weight=10.0,
    )
    assert value_actual[0].xfer_id == 4
    assert value_metrics["action_value_candidates"] == 8
    assert value_actual[0].value_score > value_actual[-1].value_score
    print("GPU proposal expansion/ranking matches legacy semantics")


if __name__ == "__main__":
    main()
