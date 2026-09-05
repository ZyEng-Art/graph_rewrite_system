from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import collate_prefixes, load_datasets
from paged_cache import PagedKVCache
from paged_model import PagedActionBindingModel
from train import move_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    payload, rules, train, _ = load_datasets(args.data)
    candidates = [train[index] for index in range(min(128, len(train)))]
    trajectory = max(candidates, key=lambda row: len(row["actions"]))
    batch = collate_prefixes([trajectory], rules)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = move_batch(batch, device)
    model = PagedActionBindingModel(
        rules,
        num_xfers=len(payload["xfer_to_source"]),
        width=48,
        retrieval_width=32,
        graph_layers=1,
        action_layers=2,
        action_heads=6,
        max_sequence_length=64,
        dropout=0.0,
    ).to(device).eval()
    model.readout_attention_backend = "paged"
    source_representations = model.source_representations()
    states, live, gate_types = model.initialize_incremental(batch)
    arena = PagedKVCache(
        layers=model.action_layers_count,
        capacity=32,
        page_size=4,
        heads=model.action_heads,
        head_width=model.width // model.action_heads,
        model_width=model.width,
        device=device,
        dtype=states.dtype,
    )
    handle = arena.empty_handle()
    checked_steps = min(8, batch["action_xfers"].shape[1])
    assert checked_steps
    with torch.no_grad():
        for action_index in range(checked_steps):
            source_slots = batch["binding_slots"][:, action_index]
            destination_slots = batch["destination_slots"][:, action_index]
            referenced = torch.cat((source_slots, destination_slots), dim=1)
            required_slots = int(referenced.max().item()) + 1
            if states.shape[1] < required_slots:
                extra = required_slots - states.shape[1]
                states = F.pad(states, (0, 0, 0, extra))
                live = F.pad(live, (0, extra), value=False)
                gate_types = F.pad(gate_types, (0, extra), value=-1)
            block_table, past_lengths = arena.block_table([handle])
            empty_actions = states.new_empty((1, 0, model.width))
            empty_mask = torch.empty((1, 0), dtype=torch.bool, device=device)
            advance_kwargs = {
                "xfer_ids": batch["action_xfers"][:, action_index],
                "source_ids": batch["action_sources"][:, action_index],
                "source_slots": source_slots,
                "destination_slots": destination_slots,
                "destination_types": batch["destination_types"][:, action_index],
                "paged_key_cache": arena.keys,
                "paged_value_cache": arena.values,
                "block_table": block_table,
                "past_lengths": past_lengths,
            }
            checked = model.advance_incremental(
                states,
                live,
                gate_types,
                None,
                None,
                empty_actions,
                empty_mask,
                **advance_kwargs,
            )
            trusted = model.advance_incremental(
                states,
                live,
                gate_types,
                None,
                None,
                empty_actions,
                empty_mask,
                **advance_kwargs,
                trusted_paged_inputs=True,
            )
            cached = model.advance_incremental(
                states,
                live,
                gate_types,
                None,
                None,
                empty_actions,
                empty_mask,
                **advance_kwargs,
                trusted_paged_inputs=True,
                source_representations=source_representations,
            )
            for expected, actual in zip(checked[:6], trusted[:6]):
                if expected.dtype in (torch.bool, torch.long):
                    assert torch.equal(expected, actual)
                else:
                    torch.testing.assert_close(actual, expected)
            for expected, actual in zip(trusted, cached):
                if expected.dtype in (torch.bool, torch.long):
                    assert torch.equal(expected, actual)
                else:
                    torch.testing.assert_close(actual, expected)
            candidate_kwargs = {
                "xfer_ids": advance_kwargs["xfer_ids"],
                "source_ids": advance_kwargs["source_ids"],
                "binding_slots": source_slots,
            }
            recomputed_features = model.candidate_features(
                states, live, **candidate_kwargs
            )
            cached_features = model.candidate_features(
                states,
                live,
                **candidate_kwargs,
                source_representations=source_representations,
            )
            torch.testing.assert_close(cached_features, recomputed_features)
            assert trusted[6].numel() == 0
            states, live, gate_types, keys, values, action, _ = trusted
            readout_keys, readout_values = model.project_action_readout_kv(action)
            child = arena.append_batch(
                [handle], keys, values, action, readout_keys, readout_values
            )[0]
            arena.release(handle)
            handle = child
    arena.release(handle)
    assert arena.allocated_pages == 0
    print(f"trusted paged advance matches checked output for {checked_steps} steps")


if __name__ == "__main__":
    main()
