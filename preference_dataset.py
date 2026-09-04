from __future__ import annotations

import torch
from torch.utils.data import Dataset

from dataset import RuleMetadata, collate_prefixes


class ActionPreferenceDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        actions = row["actions"]
        return {
            "initial_graph": row["initial_graph"],
            "actions": actions,
            "matches": [],
            "local_streak": 0,
            "trajectory_id": index,
            "prefix_length": len(actions),
            "previous_action": actions[-1] if actions else None,
            "previous_delta": None,
            "previous_local_streak": None,
            "preferred": row["preferred"],
            "rejected": row["rejected"],
            "advantage": int(row["advantage"]),
            "circuit": row["circuit"],
        }


class PrefixLengthBatchSampler:
    """Batch equal-length prefixes so padded inactive actions cannot leak NaNs."""

    def __init__(self, dataset: ActionPreferenceDataset, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        buckets: dict[int, list[int]] = {}
        for index, row in enumerate(dataset.rows):
            buckets.setdefault(len(row["actions"]), []).append(index)
        self.buckets = buckets

    def __iter__(self):
        for prefix_length in sorted(self.buckets):
            indices = self.buckets[prefix_length]
            for begin in range(0, len(indices), self.batch_size):
                yield indices[begin : begin + self.batch_size]

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.buckets.values()
        )


def collate_preferences(samples: list[dict], rules: RuleMetadata) -> dict:
    batch = collate_prefixes(samples, rules)
    max_pattern = max(
        max(map(len, rules.source_gate_types)),
        max(map(len, rules.destination_gate_types)),
    )

    def action_tensors(key: str):
        xfers = torch.tensor(
            [int(sample[key]["xfer_id"]) for sample in samples],
            dtype=torch.long,
        )
        sources = torch.tensor(
            [int(sample[key]["source_id"]) for sample in samples],
            dtype=torch.long,
        )
        bindings = torch.full(
            (len(samples), max_pattern), -1, dtype=torch.long
        )
        for index, sample in enumerate(samples):
            slots = tuple(map(int, sample[key]["binding_slots"]))
            bindings[index, : len(slots)] = torch.tensor(slots)
        return xfers, sources, bindings

    (
        batch["preferred_xfers"],
        batch["preferred_sources"],
        batch["preferred_bindings"],
    ) = action_tensors("preferred")
    (
        batch["rejected_xfers"],
        batch["rejected_sources"],
        batch["rejected_bindings"],
    ) = action_tensors("rejected")
    batch["preference_advantage"] = torch.tensor(
        [sample["advantage"] for sample in samples], dtype=torch.float
    )
    batch["preference_circuits"] = [sample["circuit"] for sample in samples]
    return batch
