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
    ranking_mode: str = "gate",
    ranking_seed: int = 0,
    action_value_model=None,
    action_value_states: torch.Tensor | None = None,
    action_value_live: torch.Tensor | None = None,
    action_value_weight: float = 0.0,
    value_increase_cap: int = 0,
    value_exploration_fraction: float = 0.0,
    action_value_microbatch: int = 16384,
    profile_stages: bool = False,
) -> tuple[list[Proposal], dict[str, float | int], dict[str, float]]:
    """Expand, cap, and globally rank actions before one compact D2H copy."""
    if ranking_mode not in {"gate", "probability", "stochastic", "value"}:
        raise ValueError(f"unknown proposal ranking mode: {ranking_mode}")
    if ranking_mode == "value" and (
        action_value_model is None
        or action_value_states is None
        or action_value_live is None
        or action_value_weight <= 0
    ):
        raise ValueError("value ranking requires model states and a positive weight")
    if value_increase_cap < 0 or value_increase_cap > per_parent_cap:
        raise ValueError("value increase cap must be within the per-parent cap")
    if value_increase_cap and ranking_mode != "value":
        raise ValueError("value increase cap is only valid for value ranking")
    if not 0.0 <= value_exploration_fraction <= 1.0:
        raise ValueError("value exploration fraction must be within [0, 1]")
    if value_exploration_fraction and ranking_mode != "value":
        raise ValueError("value exploration is only valid for value ranking")
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
    random_priorities = None
    if ranking_mode == "stochastic" or value_exploration_fraction:
        generator = torch.Generator(device=device)
        generator.manual_seed(ranking_seed)
        random_priorities = torch.rand(
            expanded_count, device=device, generator=generator
        )
    finish_timing("gpu_action_expansion_seconds", stage_started)

    stage_started = time.perf_counter()
    if ranking_mode in {"gate", "value"}:
        rank_keys = [(next_gate_counts, False), (probabilities, True)]
    elif ranking_mode == "probability":
        rank_keys = [(probabilities, True), (next_gate_counts, False)]
    else:
        rank_keys = [(random_priorities, True), (next_gate_counts, False)]
    parent_order = _stable_lexsort(
        [(parents, False), *rank_keys, (xfer_ids, False)]
    )
    ordered_parents = parents[parent_order]
    positions = torch.arange(expanded_count, device=device)
    new_parent = torch.ones(expanded_count, dtype=torch.bool, device=device)
    new_parent[1:] = ordered_parents[1:] != ordered_parents[:-1]
    group_start_positions = torch.where(new_parent, positions, 0)
    group_start_positions = torch.cummax(group_start_positions, dim=0).values
    parent_rank = positions - group_start_positions
    if ranking_mode == "value" and value_increase_cap:
        base_cap = per_parent_cap - value_increase_cap
        base_rows = parent_order[parent_rank < base_cap]
        increasing_rows = torch.where(rule_index.gate_deltas[xfer_ids] > 0)[0]
        if increasing_rows.numel():
            increase_order = increasing_rows[
                _stable_lexsort(
                    [
                        (parents[increasing_rows], False),
                        (probabilities[increasing_rows], True),
                        (next_gate_counts[increasing_rows], False),
                        (xfer_ids[increasing_rows], False),
                    ]
                )
            ]
            ordered_increase_parents = parents[increase_order]
            increase_positions = torch.arange(
                increase_order.numel(), device=device
            )
            new_increase_parent = torch.ones(
                increase_order.numel(), dtype=torch.bool, device=device
            )
            new_increase_parent[1:] = (
                ordered_increase_parents[1:] != ordered_increase_parents[:-1]
            )
            increase_group_starts = torch.where(
                new_increase_parent, increase_positions, 0
            )
            increase_group_starts = torch.cummax(
                increase_group_starts, dim=0
            ).values
            increase_rank = increase_positions - increase_group_starts
            increase_rows = increase_order[increase_rank < value_increase_cap]
            selected_mask = torch.zeros(
                expanded_count, dtype=torch.bool, device=device
            )
            selected_mask[base_rows] = True
            selected_mask[increase_rows] = True
            parent_order = torch.where(selected_mask)[0]
        else:
            parent_order = base_rows
    else:
        parent_order = parent_order[parent_rank < per_parent_cap]
    finish_timing("gpu_per_parent_rank_seconds", stage_started)

    action_values = None
    if ranking_mode == "value":
        stage_started = time.perf_counter()
        value_chunks = []
        for begin in range(0, parent_order.numel(), action_value_microbatch):
            selected_rows = parent_order[begin : begin + action_value_microbatch]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                values = action_value_model.action_values(
                    action_value_states,
                    action_value_live,
                    xfer_ids[selected_rows],
                    source_ids[selected_rows],
                    bindings[selected_rows],
                    parents[selected_rows],
                )
            value_chunks.append(values.float())
        action_values = torch.cat(value_chunks)
        value_mean = action_values.mean()
        value_std = action_values.std(unbiased=False).clamp_min(1e-6)
        action_values = (action_values - value_mean) / value_std
        finish_timing("gpu_action_value_seconds", stage_started)

    stage_started = time.perf_counter()
    if ranking_mode == "value":
        ranking_cost = (
            next_gate_counts[parent_order].float()
            - action_value_weight * action_values
        )
        global_rank_keys = [
            (ranking_cost, False),
            (next_gate_counts[parent_order], False),
            (probabilities[parent_order], True),
        ]
    elif ranking_mode == "gate":
        global_rank_keys = [
            (next_gate_counts[parent_order], False),
            (probabilities[parent_order], True),
        ]
    elif ranking_mode == "probability":
        global_rank_keys = [
            (probabilities[parent_order], True),
            (next_gate_counts[parent_order], False),
        ]
    else:
        global_rank_keys = [
            (random_priorities[parent_order], True),
            (next_gate_counts[parent_order], False),
        ]
    global_order = _stable_lexsort(
        [
            *global_rank_keys,
            (parent_gate_counts[parents[parent_order]], False),
        ]
    )
    selection_count = min(global_cap, global_order.numel())
    selected_value_exploration = 0
    if value_exploration_fraction and selection_count > 1:
        selected_value_exploration = min(
            selection_count - 1,
            max(1, round(selection_count * value_exploration_fraction)),
        )
        selected_value_count = selection_count - selected_value_exploration
        value_indices = global_order[:selected_value_count]
        remaining = torch.ones(
            parent_order.numel(), dtype=torch.bool, device=device
        )
        remaining[value_indices] = False
        exploration_candidates = torch.where(remaining)[0]
        exploration_order = torch.argsort(
            random_priorities[parent_order[exploration_candidates]],
            descending=True,
            stable=True,
        )
        exploration_indices = exploration_candidates[
            exploration_order[:selected_value_exploration]
        ]
        selected_indices = torch.empty(
            selection_count, dtype=torch.long, device=device
        )
        exploration_positions = torch.div(
            (torch.arange(selected_value_exploration, device=device) * 2 + 1)
            * selection_count,
            2 * selected_value_exploration,
            rounding_mode="floor",
        )
        exploration_mask = torch.zeros(
            selection_count, dtype=torch.bool, device=device
        )
        exploration_mask[exploration_positions] = True
        selected_indices[exploration_mask] = exploration_indices
        selected_indices[~exploration_mask] = value_indices
    else:
        selected_indices = global_order[:selection_count]
    selected = parent_order[selected_indices]
    selected_value_scores = (
        action_values[selected_indices]
        if action_values is not None
        else torch.zeros(selection_count, device=device)
    )
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
    selected_value_scores = selected_value_scores.cpu()
    proposals = []
    for row, binding, probability, value_score in zip(
        metadata.tolist(),
        selected_bindings.tolist(),
        selected_probabilities.tolist(),
        selected_value_scores.tolist(),
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
                value_score=value_score,
            )
        )
    finish_timing("proposal_device_to_host_and_pack_seconds", stage_started)
    metrics: dict[str, float | int] = {
        "predicted_actions": predicted_actions,
        "eligible_actions": expanded_count,
        "value_increase_candidates_after_parent_cap": (
            int(
                (rule_index.gate_deltas[xfer_ids[parent_order]] > 0)
                .sum()
                .item()
            )
            if ranking_mode == "value"
            else 0
        ),
        "selected_value_exploration_proposals": selected_value_exploration,
    }
    if action_values is not None:
        metrics.update(
            {
                "action_value_candidates": int(action_values.numel()),
                "selected_action_value_mean": float(
                    selected_value_scores.mean().item()
                ),
                "selected_action_value_std": float(
                    selected_value_scores.std(unbiased=False).item()
                ),
            }
        )
    return proposals, metrics, timing
