from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gpu_proposals import (
    GpuRuleIndex,
    SelectedProposalTensors,
    materialize_selected_proposals,
)
from threshold_inference import CandidateTensors


_TENSOR_FIELDS = (
    "parent_ids",
    "xfer_ids",
    "source_ids",
    "anchor_slots",
    "bindings",
    "probabilities",
    "gate_deltas",
    "next_gate_counts",
    "value_scores",
    "parent_ranks",
)


def _slice_actions(
    actions: SelectedProposalTensors, start: int, stop: int
) -> SelectedProposalTensors:
    return SelectedProposalTensors(
        **{name: getattr(actions, name)[start:stop] for name in _TENSOR_FIELDS}
    )


def _select_actions(
    actions: SelectedProposalTensors, rows: torch.Tensor
) -> SelectedProposalTensors:
    return SelectedProposalTensors(
        **{
            name: getattr(actions, name).index_select(0, rows)
            for name in _TENSOR_FIELDS
        }
    )


def _clone_actions(actions: SelectedProposalTensors) -> SelectedProposalTensors:
    return SelectedProposalTensors(
        **{name: getattr(actions, name).clone() for name in _TENSOR_FIELDS}
    )


def _cat_actions(
    chunks: list[SelectedProposalTensors],
) -> SelectedProposalTensors:
    if not chunks:
        raise ValueError("at least one ranked action chunk is required")
    return SelectedProposalTensors(
        **{
            name: torch.cat([getattr(chunk, name) for chunk in chunks])
            for name in _TENSOR_FIELDS
        }
    )


def _remap_parent(
    actions: SelectedProposalTensors, parent: int
) -> SelectedProposalTensors:
    return SelectedProposalTensors(
        parent_ids=torch.full_like(actions.parent_ids, parent),
        **{
            name: getattr(actions, name)
            for name in _TENSOR_FIELDS
            if name != "parent_ids"
        },
    )


def _stable_lexsort(
    tensors: list[tuple[torch.Tensor, bool]],
) -> torch.Tensor:
    order = torch.arange(tensors[0][0].numel(), device=tensors[0][0].device)
    for values, descending in reversed(tensors):
        local = torch.argsort(values[order], descending=descending, stable=True)
        order = order[local]
    return order


def action_tensor_bytes(actions: SelectedProposalTensors) -> int:
    return sum(
        getattr(actions, name).numel() * getattr(actions, name).element_size()
        for name in _TENSOR_FIELDS
    )


@dataclass(frozen=True)
class RankedParentActions:
    graph: Any
    actions: SelectedProposalTensors
    source_binding_candidates: int
    predicted_actions: int
    eligible_actions: int


def build_ranked_parent_entries(
    states: list[Any],
    candidates: CandidateTensors,
    ranked_pool: SelectedProposalTensors,
    rule_index: GpuRuleIndex,
) -> list[RankedParentActions]:
    """Split one miss-only ranked pool into exact-parent cache entries."""
    parent_count = len(states)
    device = candidates.batch_ids.device
    source_rows = torch.bincount(candidates.batch_ids, minlength=parent_count)
    eligible = torch.zeros(parent_count, dtype=torch.long, device=device)
    predicted = torch.zeros(parent_count, dtype=torch.long, device=device)
    if candidates.sources.numel():
        eligible.scatter_add_(
            0,
            candidates.batch_ids,
            rule_index.xfer_counts[candidates.sources],
        )
        predicted.scatter_add_(
            0,
            candidates.batch_ids,
            rule_index.all_xfer_counts[candidates.sources],
        )
    pool_counts = torch.bincount(ranked_pool.parent_ids, minlength=parent_count)
    source_rows_cpu = source_rows.cpu().tolist()
    eligible_cpu = eligible.cpu().tolist()
    predicted_cpu = predicted.cpu().tolist()
    pool_counts_cpu = pool_counts.cpu().tolist()
    offsets = [0]
    for count in pool_counts_cpu:
        offsets.append(offsets[-1] + int(count))
    return [
        RankedParentActions(
            graph=state.graph,
            actions=_slice_actions(ranked_pool, offsets[index], offsets[index + 1]),
            source_binding_candidates=int(source_rows_cpu[index]),
            predicted_actions=int(predicted_cpu[index]),
            eligible_actions=int(eligible_cpu[index]),
        )
        for index, state in enumerate(states)
    ]


def select_rank_bands(
    states: list[Any],
    entries: list[RankedParentActions],
    *,
    per_parent_cap: int,
    global_cap: int,
    parent_diversity_actions: int,
) -> tuple[list[Any], dict[str, int], SelectedProposalTensors]:
    """Reproduce gate-ranked global selection from cached local action orders."""
    chunks = []
    rank_offsets = []
    for parent, (state, entry) in enumerate(zip(states, entries)):
        lower = int(state.expansion_round) * per_parent_cap
        upper = lower + per_parent_cap
        rank_offsets.append(lower)
        chunks.append(_remap_parent(_slice_actions(entry.actions, lower, upper), parent))
    combined = _cat_actions(chunks)
    count = int(combined.parent_ids.numel())
    if not count:
        return [], {
            "predicted_actions": sum(entry.predicted_actions for entry in entries),
            "eligible_actions": sum(entry.eligible_actions for entry in entries),
            "selected_actions": 0,
            "value_increase_candidates_after_parent_cap": 0,
            "selected_value_exploration_proposals": 0,
            "parent_rank_offset_min": min(rank_offsets, default=0),
            "parent_rank_offset_max": max(rank_offsets, default=0),
            "selected_parent_rank_min": -1,
            "selected_parent_rank_max": -1,
            "rank_band_candidates": 0,
            "selected_actions_from_widened_parents": 0,
            "selected_parent_best_actions": 0,
        }, combined

    device = combined.parent_ids.device
    parent_gate_counts = torch.tensor(
        [state.gate_count for state in states], dtype=torch.long, device=device
    )
    order = _stable_lexsort(
        [
            (combined.next_gate_counts, False),
            (combined.probabilities, True),
            (parent_gate_counts[combined.parent_ids], False),
            (combined.parent_ids, False),
            (combined.xfer_ids, False),
            (combined.anchor_slots, False),
            *(
                (combined.bindings[:, column], False)
                for column in range(combined.bindings.shape[1])
            ),
        ]
    )

    globally_ranked_parents = combined.parent_ids[order]
    positions = torch.arange(order.numel(), device=device)
    first_positions = torch.full(
        (len(states),), order.numel(), dtype=torch.long, device=device
    )
    first_positions.scatter_reduce_(
        0,
        globally_ranked_parents,
        positions,
        reduce="amin",
        include_self=True,
    )
    first_round = torch.sort(
        first_positions[first_positions < order.numel()]
    ).values
    diverse_positions_by_round = [first_round]
    selected_parents = globally_ranked_parents[first_round]
    available_parents = torch.zeros(len(states), dtype=torch.bool, device=device)
    available_parents[selected_parents] = True
    available = available_parents[globally_ranked_parents]
    available[first_round] = False
    for _ in range(1, parent_diversity_actions):
        next_positions = torch.full(
            (len(states),), order.numel(), dtype=torch.long, device=device
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
                next_positions[selected_parents] < order.numel()
            ]
        ).values
        if not next_round.numel():
            break
        diverse_positions_by_round.append(next_round)
        available[next_round] = False
    diverse_positions = torch.cat(diverse_positions_by_round)
    remaining = torch.ones(order.numel(), dtype=torch.bool, device=device)
    remaining[diverse_positions] = False
    order = order[
        torch.cat((diverse_positions, torch.where(remaining)[0]))[:global_cap]
    ]
    selected = _select_actions(combined, order)
    proposals = materialize_selected_proposals(selected)
    selected_ranks = selected.parent_ranks
    offset_tensor = torch.tensor(rank_offsets, dtype=torch.long, device=device)
    metrics = {
        "predicted_actions": sum(entry.predicted_actions for entry in entries),
        "eligible_actions": sum(entry.eligible_actions for entry in entries),
        "selected_actions": int(order.numel()),
        "value_increase_candidates_after_parent_cap": 0,
        "selected_value_exploration_proposals": 0,
        "parent_rank_offset_min": min(rank_offsets, default=0),
        "parent_rank_offset_max": max(rank_offsets, default=0),
        "selected_parent_rank_min": int(selected_ranks.min().item()),
        "selected_parent_rank_max": int(selected_ranks.max().item()),
        "rank_band_candidates": count,
        "selected_actions_from_widened_parents": int(
            offset_tensor[selected.parent_ids].gt(0).sum().item()
        ),
        "selected_parent_best_actions": min(
            int(diverse_positions.numel()), global_cap
        ),
    }
    return proposals, metrics, selected


class WideningActionCache:
    """Bounded exact-parent cache of stable, locally ranked action tensors."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._entries: dict[int, RankedParentActions] = {}

    @staticmethod
    def _key(state: Any) -> int:
        return id(state.graph)

    def miss_indices(self, states: list[Any]) -> list[int]:
        missing = []
        for index, state in enumerate(states):
            entry = self._entries.get(self._key(state))
            if entry is None or entry.graph is not state.graph:
                missing.append(index)
        return missing

    def resolve_entries(
        self,
        states: list[Any],
        miss_indices: list[int],
        fresh_entries: list[RankedParentActions],
    ) -> tuple[list[RankedParentActions], dict[str, int | float | bool]]:
        if len(fresh_entries) != len(miss_indices):
            raise ValueError("fresh ranked entries must align with cache misses")
        fresh = dict(zip(miss_indices, fresh_entries))
        entries = []
        cached_rows = fresh_rows = 0
        for index, state in enumerate(states):
            if index in fresh:
                entry = fresh[index]
                fresh_rows += int(entry.actions.parent_ids.numel())
            else:
                entry = self._entries.get(self._key(state))
                if entry is None or entry.graph is not state.graph:
                    raise RuntimeError("widening action cache changed during resolution")
                cached_rows += int(entry.actions.parent_ids.numel())
            entries.append(entry)
        hits = len(states) - len(miss_indices)
        return entries, {
            "enabled": True,
            "parent_hits": hits,
            "parent_misses": len(miss_indices),
            "parent_hit_rate": hits / max(1, len(states)),
            "ranked_action_rows_reused": cached_rows,
            "ranked_action_rows_generated": fresh_rows,
            "resident_parents_before": len(self._entries),
            "resident_rows_before": self.resident_rows,
            "resident_bytes_before": self.resident_bytes,
        }

    def retain(
        self,
        states: list[Any],
        entries: list[RankedParentActions],
        selected_indices: list[int],
    ) -> dict[str, int]:
        retained = {}
        for index in selected_indices:
            state = states[index]
            entry = entries[index]
            old = self._entries.get(self._key(state))
            if old is not None and old.graph is state.graph and old is entry:
                retained[self._key(state)] = old
            else:
                retained[self._key(state)] = RankedParentActions(
                    graph=state.graph,
                    actions=_clone_actions(entry.actions),
                    source_binding_candidates=entry.source_binding_candidates,
                    predicted_actions=entry.predicted_actions,
                    eligible_actions=entry.eligible_actions,
                )
        self._entries = retained
        return {
            "resident_parents_after": len(self._entries),
            "resident_rows_after": self.resident_rows,
            "resident_bytes_after": self.resident_bytes,
        }

    @property
    def resident_rows(self) -> int:
        return sum(
            int(entry.actions.parent_ids.numel()) for entry in self._entries.values()
        )

    @property
    def resident_bytes(self) -> int:
        return sum(action_tensor_bytes(entry.actions) for entry in self._entries.values())
