from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from beam_search_benchmark import BeamState, Proposal
from threshold_inference import CandidateTensors


@dataclass(frozen=True)
class GpuRuleIndex:
    xfer_offsets: torch.Tensor
    xfer_counts: torch.Tensor
    xfer_ids: torch.Tensor
    gate_deltas: torch.Tensor
    all_xfer_counts: torch.Tensor

    @classmethod
    def build(
        cls,
        source_to_xfers: dict[int, list[int]],
        gate_deltas: list[int],
        num_sources: int,
        max_gate_increase: int,
        device: torch.device,
    ) -> "GpuRuleIndex":
        allowed: list[int] = []
        offsets = [0]
        counts = []
        all_counts = []
        for source in range(num_sources):
            source_xfers = source_to_xfers.get(source, [])
            selected = [
                xfer
                for xfer in source_xfers
                if gate_deltas[xfer] <= max_gate_increase
            ]
            allowed.extend(selected)
            counts.append(len(selected))
            all_counts.append(len(source_xfers))
            offsets.append(len(allowed))
        return cls(
            xfer_offsets=torch.tensor(offsets[:-1], device=device),
            xfer_counts=torch.tensor(counts, device=device),
            xfer_ids=torch.tensor(allowed, device=device),
            gate_deltas=torch.tensor(gate_deltas, device=device),
            all_xfer_counts=torch.tensor(all_counts, device=device),
        )


def _stable_lexsort(
    tensors: list[tuple[torch.Tensor, bool]],
) -> torch.Tensor:
    """Stable lexicographic order; keys are listed most-significant first."""
    if not tensors:
        raise ValueError("at least one sort key is required")
    order = torch.arange(tensors[0][0].numel(), device=tensors[0][0].device)
    for values, descending in reversed(tensors):
        local = torch.argsort(
            values[order], descending=descending, stable=True
        )
        order = order[local]
    return order


@torch.no_grad()
def build_gpu_proposals(
    candidates: CandidateTensors,
    beam: list[BeamState],
    rule_index: GpuRuleIndex,
    *,
    per_parent_cap: int,
    global_cap: int,
    profile_stages: bool = False,
) -> tuple[list[Proposal], dict[str, int], dict[str, float]]:
    """Expand, cap, and globally rank actions before one compact D2H copy."""
    device = candidates.sources.device
    timing: dict[str, float] = {}

    def finish_timing(name: str, started: float) -> None:
        if not profile_stages:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - started

    stage_started = time.perf_counter()
    source_counts = rule_index.xfer_counts[candidates.sources]
    candidate_count = candidates.sources.numel()
    expanded_count = int(source_counts.sum().item())
    predicted_actions = int(
        rule_index.all_xfer_counts[candidates.sources].sum().item()
    )
    if not expanded_count:
        finish_timing("gpu_action_expansion_seconds", stage_started)
        return [], {
            "predicted_actions": predicted_actions,
            "eligible_actions": 0,
        }, timing
    candidate_rows = torch.repeat_interleave(
        torch.arange(candidate_count, device=device),
        source_counts,
        output_size=expanded_count,
    )
    group_starts = torch.cumsum(source_counts, dim=0) - source_counts
    within_source = torch.arange(expanded_count, device=device) - torch.repeat_interleave(
        group_starts, source_counts, output_size=expanded_count
    )
    source_ids = candidates.sources[candidate_rows]
    xfer_ids = rule_index.xfer_ids[
        rule_index.xfer_offsets[source_ids] + within_source
    ]
    parents = candidates.batch_ids[candidate_rows]
    anchors = candidates.anchors[candidate_rows]
    bindings = candidates.bindings[candidate_rows]
    probabilities = candidates.probabilities[candidate_rows]
    parent_gate_counts = torch.tensor(
        [state.gate_count for state in beam], device=device
    )
    next_gate_counts = (
        parent_gate_counts[parents] + rule_index.gate_deltas[xfer_ids]
    )
    finish_timing("gpu_action_expansion_seconds", stage_started)

    stage_started = time.perf_counter()
    parent_order = _stable_lexsort(
        [
            (parents, False),
            (next_gate_counts, False),
            (probabilities, True),
            (xfer_ids, False),
        ]
    )
    ordered_parents = parents[parent_order]
    positions = torch.arange(expanded_count, device=device)
    new_parent = torch.ones(expanded_count, dtype=torch.bool, device=device)
    new_parent[1:] = ordered_parents[1:] != ordered_parents[:-1]
    group_start_positions = torch.where(new_parent, positions, 0)
    group_start_positions = torch.cummax(group_start_positions, dim=0).values
    parent_rank = positions - group_start_positions
    parent_order = parent_order[parent_rank < per_parent_cap]
    finish_timing("gpu_per_parent_rank_seconds", stage_started)

    stage_started = time.perf_counter()
    global_order = _stable_lexsort(
        [
            (next_gate_counts[parent_order], False),
            (probabilities[parent_order], True),
            (parent_gate_counts[parents[parent_order]], False),
        ]
    )
    selected = parent_order[global_order[:global_cap]]
    finish_timing("gpu_global_proposal_rank_seconds", stage_started)

    stage_started = time.perf_counter()
    metadata = torch.stack(
        (
            parents[selected],
            xfer_ids[selected],
            anchors[selected],
            next_gate_counts[selected],
        ),
        dim=1,
    ).cpu()
    selected_bindings = bindings[selected].cpu()
    selected_probabilities = probabilities[selected].cpu()
    proposals = []
    for row, binding, probability in zip(
        metadata.tolist(),
        selected_bindings.tolist(),
        selected_probabilities.tolist(),
    ):
        parent, xfer_id, anchor, next_gate_count = row
        # Structural decoding pads complete bindings with -1.
        clean_binding = tuple(slot for slot in binding if slot >= 0)
        proposals.append(
            Proposal(
                parent=parent,
                xfer_id=xfer_id,
                anchor_slot=anchor,
                binding=clean_binding,
                probability=probability,
                next_gate_count=next_gate_count,
            )
        )
    finish_timing("proposal_device_to_host_and_pack_seconds", stage_started)
    return proposals, {
        "predicted_actions": predicted_actions,
        "eligible_actions": expanded_count,
    }, timing
