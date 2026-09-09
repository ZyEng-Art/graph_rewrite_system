from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from ppo_core import build_policy_features, segmented_log_softmax
from search_types import BeamState, Proposal
from threshold_inference import CandidateTensors


@dataclass(frozen=True)
class GpuRuleIndex:
    xfer_offsets: torch.Tensor
    xfer_counts: torch.Tensor
    xfer_ids: torch.Tensor
    gate_deltas: torch.Tensor
    all_xfer_counts: torch.Tensor
    best_xfer_ids: torch.Tensor
    best_gate_deltas: torch.Tensor
    min_gate_delta: int
    max_gate_delta: int
    num_xfers: int

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
        best_xfers = []
        best_deltas = []
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
            if selected:
                best_xfer = min(selected, key=lambda xfer: (gate_deltas[xfer], xfer))
                best_xfers.append(best_xfer)
                best_deltas.append(gate_deltas[best_xfer])
            else:
                best_xfers.append(-1)
                best_deltas.append(torch.iinfo(torch.int64).max)
            offsets.append(len(allowed))
        return cls(
            xfer_offsets=torch.tensor(offsets[:-1], device=device),
            xfer_counts=torch.tensor(counts, device=device),
            xfer_ids=torch.tensor(allowed, device=device),
            gate_deltas=torch.tensor(gate_deltas, device=device),
            all_xfer_counts=torch.tensor(all_counts, device=device),
            best_xfer_ids=torch.tensor(best_xfers, device=device),
            best_gate_deltas=torch.tensor(best_deltas, device=device),
            min_gate_delta=min(gate_deltas),
            max_gate_delta=max(gate_deltas),
            num_xfers=len(gate_deltas),
        )


@dataclass(frozen=True)
class SelectedProposalTensors:
    parent_ids: torch.Tensor
    xfer_ids: torch.Tensor
    source_ids: torch.Tensor
    anchor_slots: torch.Tensor
    bindings: torch.Tensor
    probabilities: torch.Tensor
    gate_deltas: torch.Tensor
    next_gate_counts: torch.Tensor
    value_scores: torch.Tensor
    parent_ranks: torch.Tensor


def select_proposal_tensor_rows(
    tensors: SelectedProposalTensors, indices: torch.Tensor
) -> SelectedProposalTensors:
    indices = indices.to(device=tensors.parent_ids.device, dtype=torch.long)
    return SelectedProposalTensors(
        **{
            name: getattr(tensors, name).index_select(0, indices)
            for name in SelectedProposalTensors.__dataclass_fields__
        }
    )


def materialize_selected_proposals(
    tensors: SelectedProposalTensors,
    indices: torch.Tensor | None = None,
) -> list[Proposal]:
    if indices is None:
        indices = torch.arange(
            tensors.parent_ids.numel(),
            dtype=torch.long,
            device=tensors.parent_ids.device,
        )
    else:
        indices = indices.to(device=tensors.parent_ids.device, dtype=torch.long)
    metadata = torch.stack(
        (
            tensors.parent_ids[indices],
            tensors.xfer_ids[indices],
            tensors.anchor_slots[indices],
            tensors.next_gate_counts[indices],
            tensors.parent_ranks[indices],
        ),
        dim=1,
    ).cpu()
    bindings = tensors.bindings[indices].cpu()
    scores = torch.stack(
        (tensors.probabilities[indices], tensors.value_scores[indices]), dim=1
    ).cpu()
    proposals = []
    for row, binding, score in zip(
        metadata.tolist(), bindings.tolist(), scores.tolist()
    ):
        parent, xfer_id, anchor, next_gate_count, parent_rank = row
        proposals.append(
            Proposal(
                parent=parent,
                xfer_id=xfer_id,
                anchor_slot=anchor,
                binding=tuple(slot for slot in binding if slot >= 0),
                probability=score[0],
                next_gate_count=next_gate_count,
                value_score=score[1],
                parent_rank=parent_rank,
            )
        )
    return proposals


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


def _preselect_match_rows(
    candidates: CandidateTensors,
    candidate_rows: torch.Tensor,
    rule_index: GpuRuleIndex,
    parent_gate_counts: torch.Tensor,
    per_parent_cap: int,
    ranking_mode: str,
) -> torch.Tensor:
    """Return every match that can still contribute a per-parent top action."""
    parents = candidates.batch_ids[candidate_rows]
    sources = candidates.sources[candidate_rows]
    probabilities = candidates.probabilities[candidate_rows]
    best_next_gate_counts = (
        parent_gate_counts[parents] + rule_index.best_gate_deltas[sources]
    )
    best_xfer_ids = rule_index.best_xfer_ids[sources]
    if ranking_mode in {"gate", "ppo"}:
        rank_keys = [(best_next_gate_counts, False), (probabilities, True)]
    else:
        rank_keys = [(probabilities, True), (best_next_gate_counts, False)]

    # CUDA compaction does not promise a stable order for equal matcher rows.
    # A local-index tie-break therefore makes the selected match set vary even
    # when logits and aggregate counts are identical.  Sort by the complete
    # structural match description instead: anchor followed by every binding
    # slot.  These are ordering keys only; Quartz still validates each action.
    structural_keys: list[tuple[torch.Tensor, bool]] = [
        (candidates.anchors[candidate_rows], False),
    ]
    selected_bindings = candidates.bindings[candidate_rows]
    structural_keys.extend(
        (selected_bindings[:, column], False)
        for column in range(selected_bindings.shape[1])
    )
    order = _stable_lexsort(
        [(parents, False), *rank_keys, (best_xfer_ids, False), *structural_keys]
    )
    ordered_parents = parents[order]
    positions = torch.arange(order.numel(), device=order.device)
    new_parent = torch.ones(order.numel(), dtype=torch.bool, device=order.device)
    new_parent[1:] = ordered_parents[1:] != ordered_parents[:-1]
    group_starts = torch.where(new_parent, positions, 0)
    group_starts = torch.cummax(group_starts, dim=0).values
    parent_rank = positions - group_starts
    return candidate_rows[order[parent_rank < per_parent_cap]]


def _match_set_policy_logits(
    actor_critic,
    candidate_features: torch.Tensor,
    matcher_logits: torch.Tensor,
    parent_ids: torch.Tensor,
    prefix_states: torch.Tensor,
    state_features: torch.Tensor,
) -> torch.Tensor:
    """Run a padded candidate-set actor and restore the flattened row order."""
    active_parents, parent_rows = torch.unique(
        parent_ids, sorted=True, return_inverse=True
    )
    order = torch.argsort(parent_rows, stable=True)
    sorted_parent_rows = parent_rows[order]
    positions = torch.arange(order.numel(), device=order.device)
    new_parent = torch.ones(
        order.numel(), dtype=torch.bool, device=order.device
    )
    new_parent[1:] = sorted_parent_rows[1:] != sorted_parent_rows[:-1]
    group_starts = torch.where(new_parent, positions, 0)
    group_starts = torch.cummax(group_starts, dim=0).values
    sorted_offsets = positions - group_starts
    offsets = torch.empty_like(sorted_offsets)
    offsets[order] = sorted_offsets
    max_candidates = int(
        torch.bincount(parent_rows, minlength=active_parents.numel()).max().item()
    )

    padded_features = candidate_features.new_zeros(
        (active_parents.numel(), max_candidates, candidate_features.shape[-1])
    )
    padded_logits = matcher_logits.new_zeros(
        (active_parents.numel(), max_candidates)
    )
    candidate_mask = torch.zeros(
        (active_parents.numel(), max_candidates),
        dtype=torch.bool,
        device=parent_ids.device,
    )
    padded_features[parent_rows, offsets] = candidate_features
    padded_logits[parent_rows, offsets] = matcher_logits
    candidate_mask[parent_rows, offsets] = True
    logits = actor_critic.policy_logits(
        padded_features,
        padded_logits,
        candidate_mask,
        prefix_states.index_select(0, active_parents),
        state_features.index_select(0, active_parents),
    )
    return logits[parent_rows, offsets]


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
    preserve_parent_best: bool = False,
    parent_diversity_actions: int = 1,
    parent_diversity_parent_cap: int = 0,
    locality_action_reserve: int = 0,
    parent_rank_offsets: list[int] | torch.Tensor | None = None,
    action_value_microbatch: int = 16384,
    ppo_actor_critic=None,
    ppo_model=None,
    ppo_states: torch.Tensor | None = None,
    ppo_live: torch.Tensor | None = None,
    ppo_prefix_states: torch.Tensor | None = None,
    ppo_state_features: torch.Tensor | None = None,
    ppo_initial_gate_bias: float = 1.0,
    ppo_policy_weight: float = 0.25,
    preselect_matches: bool = False,
    return_selected_tensors: bool = False,
    materialize_python_proposals: bool = True,
    profile_stages: bool = False,
    ranked_pool_cap: int = 0,
    ranked_pool_output: list[SelectedProposalTensors] | None = None,
    ranked_pool_only: bool = False,
) -> tuple[
    list[Proposal] | None,
    dict[str, float | int],
    dict[str, float],
    SelectedProposalTensors | None,
]:
    """Expand and rank actions, optionally deferring their compact D2H copy."""
    if ranking_mode not in {"gate", "probability", "stochastic", "value", "ppo"}:
        raise ValueError(f"unknown proposal ranking mode: {ranking_mode}")
    if ranking_mode == "value" and (
        action_value_model is None
        or action_value_states is None
        or action_value_live is None
        or action_value_weight <= 0
    ):
        raise ValueError("value ranking requires model states and a positive weight")
    if ranking_mode == "ppo" and (
        ppo_actor_critic is None
        or ppo_model is None
        or ppo_states is None
        or ppo_live is None
    ):
        raise ValueError("PPO ranking requires actor, model, and encoded states")
    if (
        ranking_mode == "ppo"
        and getattr(ppo_actor_critic, "match_set_aware", False)
        and (ppo_prefix_states is None or ppo_state_features is None)
    ):
        raise ValueError("match-set PPO ranking requires prefix and state features")
    if ppo_policy_weight < 0:
        raise ValueError("PPO policy weight must be nonnegative")
    if value_increase_cap < 0 or value_increase_cap > per_parent_cap:
        raise ValueError("value increase cap must be within the per-parent cap")
    if value_increase_cap and ranking_mode != "value":
        raise ValueError("value increase cap is only valid for value ranking")
    if not 0.0 <= value_exploration_fraction <= 1.0:
        raise ValueError("value exploration fraction must be within [0, 1]")
    if value_exploration_fraction and ranking_mode != "value":
        raise ValueError("value exploration is only valid for value ranking")
    if not materialize_python_proposals and not return_selected_tensors:
        raise ValueError("deferred proposals require selected GPU tensors")
    if ranked_pool_cap < 0:
        raise ValueError("ranked pool cap must be nonnegative")
    if ranked_pool_cap and ranked_pool_output is None:
        raise ValueError("ranked pool output is required when its cap is positive")
    if ranked_pool_cap and ranking_mode != "gate":
        raise ValueError("ranked pool capture currently requires gate ranking")
    if ranked_pool_only and not ranked_pool_cap:
        raise ValueError("ranked-pool-only mode requires ranked pool capture")
    if parent_diversity_actions < 1:
        raise ValueError("parent diversity actions must be positive")
    if parent_diversity_parent_cap < 0:
        raise ValueError("parent diversity parent cap must be nonnegative")
    if locality_action_reserve < 0 or locality_action_reserve > per_parent_cap:
        raise ValueError("locality action reserve must be within the per-parent cap")
    if parent_rank_offsets is not None and locality_action_reserve:
        raise ValueError(
            "parent rank offsets cannot be combined with locality action reserve"
        )
    device = candidates.sources.device
    if parent_rank_offsets is None:
        rank_offsets = torch.zeros(len(beam), dtype=torch.long, device=device)
    else:
        rank_offsets = torch.as_tensor(
            parent_rank_offsets, dtype=torch.long, device=device
        )
        if rank_offsets.ndim != 1 or rank_offsets.numel() != len(beam):
            raise ValueError("parent rank offsets must have one value per beam state")
        if bool(rank_offsets.lt(0).any().item()):
            raise ValueError("parent rank offsets must be nonnegative")
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
        if ranked_pool_cap:
            empty = candidates.sources[:0]
            ranked_pool_output.append(
                SelectedProposalTensors(
                    parent_ids=candidates.batch_ids[:0],
                    xfer_ids=empty,
                    source_ids=empty,
                    anchor_slots=candidates.anchors[:0],
                    bindings=candidates.bindings[:0],
                    probabilities=candidates.probabilities[:0],
                    gate_deltas=empty,
                    next_gate_counts=empty,
                    value_scores=candidates.probabilities[:0],
                    parent_ranks=empty,
                )
            )
        finish_timing("gpu_action_expansion_seconds", stage_started)
        return (
            [] if materialize_python_proposals else None,
            {
                "predicted_actions": predicted_actions,
                "eligible_actions": 0,
            },
            timing,
            None,
        )
    parent_gate_counts = torch.tensor(
        [state.gate_count for state in beam], device=device
    )
    candidate_rows_to_expand = torch.arange(candidate_count, device=device)
    if preselect_matches and ranking_mode in {"gate", "probability", "ppo"}:
        eligible_matches = source_counts.gt(0)
        candidate_rows_to_expand = candidate_rows_to_expand[eligible_matches]
        candidate_rows_to_expand = _preselect_match_rows(
            candidates,
            candidate_rows_to_expand,
            rule_index,
            parent_gate_counts,
            per_parent_cap,
            ranking_mode,
        )
    finish_timing("gpu_match_preselection_seconds", stage_started)

    stage_started = time.perf_counter()
    materialized_source_counts = source_counts[candidate_rows_to_expand]
    materialized_count = int(materialized_source_counts.sum().item())
    candidate_rows = torch.repeat_interleave(
        candidate_rows_to_expand,
        materialized_source_counts,
        output_size=materialized_count,
    )
    group_starts = (
        torch.cumsum(materialized_source_counts, dim=0) - materialized_source_counts
    )
    within_source = torch.arange(
        materialized_count, device=device
    ) - torch.repeat_interleave(
        group_starts,
        materialized_source_counts,
        output_size=materialized_count,
    )
    source_ids = candidates.sources[candidate_rows]
    xfer_ids = rule_index.xfer_ids[
        rule_index.xfer_offsets[source_ids] + within_source
    ]
    parents = candidates.batch_ids[candidate_rows]
    anchors = candidates.anchors[candidate_rows]
    bindings = candidates.bindings[candidate_rows]
    probabilities = candidates.probabilities[candidate_rows]
    next_gate_counts = (
        parent_gate_counts[parents] + rule_index.gate_deltas[xfer_ids]
    )
    random_priorities = None
    if ranking_mode == "stochastic" or value_exploration_fraction:
        generator = torch.Generator(device=device)
        generator.manual_seed(ranking_seed)
        random_priorities = torch.rand(
            materialized_count, device=device, generator=generator
        )
    local_continuations = torch.zeros(
        materialized_count, dtype=torch.bool, device=device
    )
    if locality_action_reserve and bindings.numel():
        max_slot = max(
            max((int(state.next_slot) for state in beam), default=0),
            int(bindings.max().item()) + 1,
        )
        if max_slot:
            preferred = torch.zeros(
                (len(beam), max_slot), dtype=torch.bool, device=device
            )
            for parent_index, state in enumerate(beam):
                slots = sorted(
                    int(slot)
                    for slot in state.previous_preferred
                    if 0 <= int(slot) < max_slot
                )
                if slots:
                    preferred[parent_index, slots] = True
            valid_bindings = bindings.ge(0)
            local_continuations = (
                preferred[
                    parents.unsqueeze(1), bindings.clamp_min(0)
                ]
                & valid_bindings
            ).any(dim=1)
    finish_timing("gpu_action_expansion_seconds", stage_started)

    stage_started = time.perf_counter()
    if ranking_mode in {"gate", "value", "ppo"}:
        rank_keys = [(next_gate_counts, False), (probabilities, True)]
    elif ranking_mode == "probability":
        rank_keys = [(probabilities, True), (next_gate_counts, False)]
    else:
        rank_keys = [(random_priorities, True), (next_gate_counts, False)]
    structural_action_keys = [
        (anchors, False),
        *((bindings[:, column], False) for column in range(bindings.shape[1])),
    ]
    parent_order = _stable_lexsort(
        [
            (parents, False),
            *rank_keys,
            (xfer_ids, False),
            *structural_action_keys,
        ]
    )
    ordered_parents = parents[parent_order]
    positions = torch.arange(materialized_count, device=device)
    new_parent = torch.ones(materialized_count, dtype=torch.bool, device=device)
    new_parent[1:] = ordered_parents[1:] != ordered_parents[:-1]
    group_start_positions = torch.where(new_parent, positions, 0)
    group_start_positions = torch.cummax(group_start_positions, dim=0).values
    parent_rank = positions - group_start_positions
    rank_by_materialized_row = torch.empty_like(parent_rank)
    rank_by_materialized_row[parent_order] = parent_rank
    if ranked_pool_cap:
        pool_rows = parent_order[parent_rank < ranked_pool_cap]
        ranked_pool_output.append(
            SelectedProposalTensors(
                parent_ids=parents[pool_rows],
                xfer_ids=xfer_ids[pool_rows],
                source_ids=source_ids[pool_rows],
                anchor_slots=anchors[pool_rows],
                bindings=bindings[pool_rows],
                probabilities=probabilities[pool_rows],
                gate_deltas=rule_index.gate_deltas[xfer_ids[pool_rows]],
                next_gate_counts=next_gate_counts[pool_rows],
                value_scores=torch.zeros(
                    pool_rows.numel(),
                    dtype=probabilities.dtype,
                    device=device,
                ),
                parent_ranks=rank_by_materialized_row[pool_rows],
            )
        )
        if ranked_pool_only:
            finish_timing("gpu_per_parent_rank_seconds", stage_started)
            return (
                None,
                {
                    "predicted_actions": predicted_actions,
                    "eligible_actions": expanded_count,
                },
                timing,
                None,
            )
    if bool(rank_offsets.any().item()):
        lower_ranks = rank_offsets[ordered_parents]
        parent_order = parent_order[
            (parent_rank >= lower_ranks)
            & (parent_rank < lower_ranks + per_parent_cap)
        ]
    elif locality_action_reserve:
        local_rows = torch.where(local_continuations)[0]
        if local_rows.numel():
            local_rank_keys = [
                (values[local_rows], descending)
                for values, descending in rank_keys
            ]
            local_order = local_rows[
                _stable_lexsort(
                    [
                        (parents[local_rows], False),
                        *local_rank_keys,
                        (xfer_ids[local_rows], False),
                        (anchors[local_rows], False),
                        *((bindings[local_rows, column], False)
                          for column in range(bindings.shape[1])),
                    ]
                )
            ]
            ordered_local_parents = parents[local_order]
            local_positions = torch.arange(
                local_order.numel(), device=device
            )
            new_local_parent = torch.ones(
                local_order.numel(), dtype=torch.bool, device=device
            )
            new_local_parent[1:] = (
                ordered_local_parents[1:] != ordered_local_parents[:-1]
            )
            local_starts = torch.where(
                new_local_parent, local_positions, 0
            )
            local_starts = torch.cummax(local_starts, dim=0).values
            local_rank = local_positions - local_starts
            local_rows = local_order[
                local_rank < locality_action_reserve
            ]
        local_selected_mask = torch.zeros(
            materialized_count, dtype=torch.bool, device=device
        )
        local_selected_mask[local_rows] = True
        selected_local_counts = torch.bincount(
            parents[local_rows], minlength=len(beam)
        )

        # Fill every parent's remaining capacity from its original ranking.
        # Excluding reserved local rows first avoids losing width when a local
        # row was already present in the ordinary top-k prefix.
        remaining_order = parent_order[
            ~local_selected_mask[parent_order]
        ]
        remaining_parents = parents[remaining_order]
        remaining_positions = torch.arange(
            remaining_order.numel(), device=device
        )
        new_remaining_parent = torch.ones(
            remaining_order.numel(), dtype=torch.bool, device=device
        )
        new_remaining_parent[1:] = (
            remaining_parents[1:] != remaining_parents[:-1]
        )
        remaining_starts = torch.where(
            new_remaining_parent, remaining_positions, 0
        )
        remaining_starts = torch.cummax(remaining_starts, dim=0).values
        remaining_rank = remaining_positions - remaining_starts
        remaining_caps = per_parent_cap - selected_local_counts
        fill_rows = remaining_order[
            remaining_rank < remaining_caps[remaining_parents]
        ]
        selected_mask = local_selected_mask.clone()
        selected_mask[fill_rows] = True
        parent_order = torch.where(selected_mask)[0]
    elif ranking_mode == "value" and value_increase_cap:
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
                        (anchors[increasing_rows], False),
                        *((bindings[increasing_rows, column], False)
                          for column in range(bindings.shape[1])),
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
                materialized_count, dtype=torch.bool, device=device
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
    elif ranking_mode in {"gate", "ppo"}:
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
            (parents[parent_order], False),
            (xfer_ids[parent_order], False),
            (anchors[parent_order], False),
            *((bindings[parent_order, column], False)
              for column in range(bindings.shape[1])),
        ]
    )
    parent_diverse_count = 0
    if preserve_parent_best and global_order.numel():
        # A purely global beam can discard every continuation from a useful
        # prefix even when that prefix's best action is ranked first locally.
        # Move the best-ranked action from every live parent to the front,
        # retaining the configured global order within both partitions.
        globally_ranked_parents = parents[parent_order[global_order]]
        positions = torch.arange(global_order.numel(), device=device)
        first_positions = torch.full(
            (len(beam),),
            global_order.numel(),
            dtype=torch.long,
            device=device,
        )
        first_positions.scatter_reduce_(
            0,
            globally_ranked_parents,
            positions,
            reduce="amin",
            include_self=True,
        )
        first_round = torch.sort(
            first_positions[first_positions < global_order.numel()]
        ).values
        if parent_diversity_parent_cap:
            first_round = first_round[:parent_diversity_parent_cap]
        diverse_positions_by_round = [first_round]
        selected_parents = globally_ranked_parents[first_round]
        available = torch.zeros(
            global_order.numel(), dtype=torch.bool, device=device
        )
        available_parents = torch.zeros(
            len(beam), dtype=torch.bool, device=device
        )
        available_parents[selected_parents] = True
        available = available_parents[globally_ranked_parents]
        available[first_round] = False
        for _ in range(1, parent_diversity_actions):
            next_positions = torch.full(
                (len(beam),),
                global_order.numel(),
                dtype=torch.long,
                device=device,
            )
            next_positions.scatter_reduce_(
                0,
                globally_ranked_parents[available],
                positions[available],
                reduce="amin",
                include_self=True,
            )
            next_round = torch.sort(
                next_positions[selected_parents][
                    next_positions[selected_parents] < global_order.numel()
                ]
            ).values
            if not next_round.numel():
                break
            diverse_positions_by_round.append(next_round)
            available[next_round] = False
        diverse_positions = torch.cat(diverse_positions_by_round)
        remaining = torch.ones(
            global_order.numel(), dtype=torch.bool, device=device
        )
        remaining[diverse_positions] = False
        reordered_positions = torch.cat(
            (diverse_positions, torch.where(remaining)[0])
        )
        global_order = global_order[reordered_positions]
        parent_diverse_count = min(
            int(diverse_positions.numel()),
            global_cap,
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

    if ranking_mode == "ppo":
        stage_started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            candidate_features = ppo_model.candidate_features(
                ppo_states,
                ppo_live,
                xfer_ids[selected],
                source_ids[selected],
                bindings[selected],
                parents[selected],
                ordered_roles=getattr(
                    ppo_actor_critic, "match_set_aware", False
                ),
            )
            policy_features, matcher_logits = build_policy_features(
                candidate_features,
                probabilities[selected],
                rule_index.gate_deltas[xfer_ids[selected]],
                initial_gate_bias=ppo_initial_gate_bias,
            )
            if getattr(ppo_actor_critic, "match_set_aware", False):
                policy_logits = _match_set_policy_logits(
                    ppo_actor_critic,
                    policy_features,
                    matcher_logits,
                    parents[selected],
                    ppo_prefix_states,
                    ppo_state_features,
                ).float()
            else:
                policy_logits = ppo_actor_critic.policy_logits(
                    policy_features, matcher_logits
                ).float()
        policy_scores = segmented_log_softmax(
            policy_logits, parents[selected], len(beam)
        )
        policy_mean = policy_scores.mean()
        policy_std = policy_scores.std(unbiased=False).clamp_min(1e-6)
        standardized_policy_scores = (policy_scores - policy_mean) / policy_std
        ranking_cost = next_gate_counts[selected].float() - (
            ppo_policy_weight * standardized_policy_scores
        )
        ppo_order = _stable_lexsort(
            [
                (ranking_cost, False),
                (next_gate_counts[selected], False),
                (policy_scores, True),
                (probabilities[selected], True),
                (parents[selected], False),
                (xfer_ids[selected], False),
                (anchors[selected], False),
                *((bindings[selected, column], False)
                  for column in range(bindings.shape[1])),
            ]
        )
        selected = selected[ppo_order]
        selected_value_scores = policy_scores[ppo_order]
        finish_timing("gpu_ppo_policy_seconds", stage_started)

    proposal_tensors = SelectedProposalTensors(
        parent_ids=parents[selected],
        xfer_ids=xfer_ids[selected],
        source_ids=source_ids[selected],
        anchor_slots=anchors[selected],
        bindings=bindings[selected],
        probabilities=probabilities[selected],
        gate_deltas=rule_index.gate_deltas[xfer_ids[selected]],
        next_gate_counts=next_gate_counts[selected],
        value_scores=selected_value_scores,
        parent_ranks=rank_by_materialized_row[selected],
    )
    selected_tensors = proposal_tensors if return_selected_tensors else None

    stage_started = time.perf_counter()
    proposals = (
        materialize_selected_proposals(proposal_tensors)
        if materialize_python_proposals
        else None
    )
    finish_timing("proposal_device_to_host_and_pack_seconds", stage_started)
    metrics: dict[str, float | int] = {
        "predicted_actions": predicted_actions,
        "eligible_actions": expanded_count,
        "selected_actions": int(selected.numel()),
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
    if parent_rank_offsets is not None:
        selected_parent_ranks = rank_by_materialized_row[selected]
        metrics.update(
            {
                "parent_rank_offset_min": int(rank_offsets.min().item()),
                "parent_rank_offset_max": int(rank_offsets.max().item()),
                "selected_parent_rank_min": (
                    int(selected_parent_ranks.min().item())
                    if selected_parent_ranks.numel()
                    else -1
                ),
                "selected_parent_rank_max": (
                    int(selected_parent_ranks.max().item())
                    if selected_parent_ranks.numel()
                    else -1
                ),
                "rank_band_candidates": int(parent_order.numel()),
                "selected_actions_from_widened_parents": int(
                    rank_offsets[parents[selected]].gt(0).sum().item()
                ),
            }
        )
    if preselect_matches:
        metrics["materialized_actions"] = materialized_count
    if locality_action_reserve:
        metrics.update(
            {
                "locality_action_reserve": locality_action_reserve,
                "local_continuation_candidates": int(
                    local_continuations.sum().item()
                ),
                "selected_local_continuations": int(
                    local_continuations[selected].sum().item()
                ),
            }
        )
    if preserve_parent_best:
        metrics["selected_parent_best_actions"] = parent_diverse_count
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
    if ranking_mode == "ppo":
        metrics.update(
            {
                "ppo_policy_candidates": int(selected_value_scores.numel()),
                "selected_ppo_score_mean": float(
                    selected_value_scores.mean().item()
                ),
                "ppo_policy_score_std": float(policy_std.item()),
            }
        )
    return proposals, metrics, timing, selected_tensors
