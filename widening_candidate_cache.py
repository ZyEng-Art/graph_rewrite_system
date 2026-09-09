from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from threshold_inference import CandidateTensors


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def candidate_tensor_bytes(candidates: CandidateTensors) -> int:
    return sum(
        _tensor_bytes(getattr(candidates, name))
        for name in (
            "batch_ids",
            "sources",
            "anchors",
            "bindings",
            "probabilities",
        )
    )


def _select_rows(candidates: CandidateTensors, rows: torch.Tensor) -> CandidateTensors:
    return CandidateTensors(
        **{
            name: getattr(candidates, name).index_select(0, rows)
            for name in (
                "batch_ids",
                "sources",
                "anchors",
                "bindings",
                "probabilities",
            )
        }
    )


def split_candidate_tensors(
    candidates: CandidateTensors, num_parents: int
) -> list[CandidateTensors]:
    """Split a compact candidate batch while preserving every parent's row order."""
    if num_parents < 0:
        raise ValueError("num_parents must be nonnegative")
    if candidates.batch_ids.numel() and (
        bool(candidates.batch_ids.lt(0).any().item())
        or bool(candidates.batch_ids.ge(num_parents).any().item())
    ):
        raise ValueError("candidate batch id is outside the parent batch")
    result = []
    for parent in range(num_parents):
        rows = torch.nonzero(candidates.batch_ids.eq(parent), as_tuple=False).flatten()
        selected = _select_rows(candidates, rows)
        result.append(
            CandidateTensors(
                batch_ids=torch.zeros_like(selected.batch_ids),
                sources=selected.sources,
                anchors=selected.anchors,
                bindings=selected.bindings,
                probabilities=selected.probabilities,
            )
        )
    return result


def remap_parent_candidates(
    candidates: CandidateTensors, parent: int
) -> CandidateTensors:
    return CandidateTensors(
        batch_ids=torch.full_like(candidates.batch_ids, parent),
        sources=candidates.sources,
        anchors=candidates.anchors,
        bindings=candidates.bindings,
        probabilities=candidates.probabilities,
    )


@dataclass(frozen=True)
class CachedParentCandidates:
    graph: Any
    candidates: CandidateTensors


class WideningCandidateCache:
    """Invocation-local GPU candidate cache for exact-parent widening revisits.

    A cache entry is owned by the concrete Quartz graph object.  Widening creates
    a dataclass replacement of a parent while deliberately retaining that graph
    object, so object identity is sufficient and avoids rebuilding an exact key.
    Entries retain the graph reference, preventing Python object-id reuse.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._entries: dict[int, CachedParentCandidates] = {}

    @staticmethod
    def _key(state: Any) -> int:
        return id(state.graph)

    def miss_indices(self, states: list[Any]) -> list[int]:
        if not self.enabled:
            return list(range(len(states)))
        missing = []
        for index, state in enumerate(states):
            entry = self._entries.get(self._key(state))
            if entry is None or entry.graph is not state.graph:
                missing.append(index)
        return missing

    def resolve(
        self,
        states: list[Any],
        *,
        miss_indices: list[int],
        fresh_candidates: CandidateTensors | None,
    ) -> tuple[CandidateTensors, list[CandidateTensors], dict[str, int | float | bool]]:
        """Combine cached rows and a compact batch generated for cache misses."""
        if not self.enabled:
            raise ValueError("cannot resolve candidates with a disabled cache")
        if len(set(miss_indices)) != len(miss_indices) or any(
            index < 0 or index >= len(states) for index in miss_indices
        ):
            raise ValueError("invalid cache-miss indices")
        if bool(miss_indices) != (fresh_candidates is not None):
            raise ValueError("fresh candidates must be supplied exactly for cache misses")

        fresh_by_parent = (
            split_candidate_tensors(fresh_candidates, len(miss_indices))
            if fresh_candidates is not None
            else []
        )
        fresh_lookup = {
            parent_index: fresh_by_parent[compact_index]
            for compact_index, parent_index in enumerate(miss_indices)
        }
        miss_set = set(miss_indices)
        per_parent = []
        hit_rows = 0
        miss_rows = 0
        for index, state in enumerate(states):
            if index in miss_set:
                rows = fresh_lookup[index]
                miss_rows += int(rows.sources.numel())
            else:
                entry = self._entries.get(self._key(state))
                if entry is None or entry.graph is not state.graph:
                    raise RuntimeError("widening candidate cache changed during resolution")
                rows = entry.candidates
                hit_rows += int(rows.sources.numel())
            per_parent.append(rows)

        combined = CandidateTensors.cat(
            [remap_parent_candidates(rows, index) for index, rows in enumerate(per_parent)]
        )
        hits = len(states) - len(miss_indices)
        return combined, per_parent, {
            "enabled": True,
            "parent_hits": hits,
            "parent_misses": len(miss_indices),
            "parent_hit_rate": hits / max(1, len(states)),
            "candidate_rows_reused": hit_rows,
            "candidate_rows_generated": miss_rows,
            "resident_parents_before": len(self._entries),
            "resident_rows_before": self.resident_rows,
            "resident_bytes_before": self.resident_bytes,
        }

    def retain(
        self,
        states: list[Any],
        per_parent: list[CandidateTensors],
        selected_indices: list[int],
    ) -> dict[str, int]:
        """Retain only parents explicitly scheduled for a later rank band."""
        if not self.enabled:
            return {
                "resident_parents_after": 0,
                "resident_rows_after": 0,
                "resident_bytes_after": 0,
            }
        if len(states) != len(per_parent):
            raise ValueError("candidate rows must align with states")
        entries: dict[int, CachedParentCandidates] = {}
        for index in selected_indices:
            if index < 0 or index >= len(states):
                raise ValueError("selected cache index is outside the state batch")
            state = states[index]
            entries[self._key(state)] = CachedParentCandidates(
                graph=state.graph, candidates=per_parent[index]
            )
        self._entries = entries
        return {
            "resident_parents_after": len(self._entries),
            "resident_rows_after": self.resident_rows,
            "resident_bytes_after": self.resident_bytes,
        }

    @property
    def resident_rows(self) -> int:
        return sum(int(entry.candidates.sources.numel()) for entry in self._entries.values())

    @property
    def resident_bytes(self) -> int:
        return sum(candidate_tensor_bytes(entry.candidates) for entry in self._entries.values())
