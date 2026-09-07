from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import collate_prefixes, load_datasets
from paged_cache import PagedKVCache
from paged_model import PagedActionBindingModel
from train import move_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ordered-binding-roles", action="store_true")
    parser.add_argument("--readout-graph-layers", type=int, default=0)
    parser.add_argument(
        "--readout-graph-input",
        choices=("cached", "gate", "cached_gate"),
        default="cached",
    )
    parser.add_argument("--readout-locality-features", action="store_true")
    parser.add_argument("--identity-readout-prefix", type=int, default=0)
    args = parser.parse_args()
    payload, rules, train, _ = load_datasets(args.data)
    indices = [0, min(1, len(train) - 1), min(7, len(train) - 1), min(31, len(train) - 1)]
    batch = collate_prefixes([train[index] for index in indices], rules)
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
        ordered_binding_roles=args.ordered_binding_roles,
        readout_graph_layers=args.readout_graph_layers,
        readout_graph_input=args.readout_graph_input,
        readout_locality_features=args.readout_locality_features,
        identity_readout_prefix=args.identity_readout_prefix,
        dropout=0.0,
    ).to(device).eval()
    with torch.no_grad():
        states, live, gate_types = model.encode(batch)
        logits, eligible = model.match_logits(states, live, gate_types)
    assert states.shape[:2] == batch["current_types"].shape
    assert torch.equal(gate_types, batch["current_types"])
    assert torch.equal(live, batch["current_types"].ge(0))
    assert logits.shape == (*gate_types.shape, model.num_sources)
    assert eligible.shape == logits.shape

    # State-only exact inference must reproduce the historical compatibility
    # path that rebased the current graph as s0 and supplied empty actions.
    rebased = dict(batch)
    rebased.update(
        {
            "initial_types": batch["current_types"],
            "edge_batch": batch["current_edge_batch"],
            "edge_src": batch["current_edge_src"],
            "edge_dst": batch["current_edge_dst"],
            "edge_relation": batch["current_edge_relation"],
            "action_xfers": batch["action_xfers"][:, :0],
            "action_sources": batch["action_sources"][:, :0],
            "binding_slots": batch["binding_slots"][:, :0],
            "destination_slots": batch["destination_slots"][:, :0],
            "destination_types": batch["destination_types"][:, :0],
        }
    )
    with torch.no_grad():
        compatibility_states, compatibility_live, compatibility_types = (
            model.encode(rebased)
        )
        state_only_states, state_only_live, state_only_types = (
            model.encode_current_graph(batch)
        )
    assert torch.equal(compatibility_live, state_only_live)
    assert torch.equal(compatibility_types, state_only_types)
    assert torch.allclose(
        compatibility_states, state_only_states, atol=2e-5, rtol=2e-5
    )

    # The fused SDPA readout must preserve the eager implementation, including
    # a row with no valid history. Temporarily make the normally zero-initialized
    # output projection observable for this backend equivalence check.
    history_length = min(7, batch["action_xfers"].shape[1])
    history = torch.randn(
        states.shape[0], history_length, model.width, device=device
    )
    history_lengths = torch.tensor(
        [0, 2, 5, history_length], device=device
    ).clamp_max(history_length)
    history_mask = torch.arange(
        history_length, device=device
    ).unsqueeze(0) < history_lengths.unsqueeze(1)
    saved_output = model.node_action_output.weight.detach().clone()
    with torch.no_grad():
        model.node_action_output.weight.normal_(std=0.01)
        model.readout_attention_backend = "eager"
        eager_readout = model._fuse_action_history(
            states, live, history, history_mask
        )
        model.readout_attention_backend = "sdpa"
        sdpa_readout = model._fuse_action_history(
            states, live, history, history_mask
        )
        model.readout_attention_backend = "sdpa_live"
        compact_readout = model._fuse_action_history(
            states,
            live,
            history,
            history_mask,
            batch["current_live_slots"],
        )
        model.node_action_output.weight.copy_(saved_output)
    assert torch.allclose(eager_readout, sdpa_readout, atol=2e-4, rtol=2e-4)
    assert torch.allclose(eager_readout, compact_readout, atol=2e-4, rtol=2e-4)

    candidate_batch = []
    candidate_sources = []
    candidate_anchors = []
    targets = []
    for batch_index, rows in enumerate(batch["positives"]):
        for source, binding in rows[:8]:
            candidate_batch.append(batch_index)
            candidate_sources.append(source)
            candidate_anchors.append(binding[0])
            targets.append(tuple(binding))
    bindings, valid = model.structural_decode(
        batch,
        gate_types,
        live,
        torch.tensor(candidate_batch, device=device),
        torch.tensor(candidate_sources, device=device),
        torch.tensor(candidate_anchors, device=device),
    )
    for row, source, target, is_valid in zip(
        bindings.tolist(), candidate_sources, targets, valid.tolist()
    ):
        length = int(model.source_lengths[source])
        assert is_valid and tuple(row[:length]) == target

    # The paged one-token path must reproduce full-prefix encoding exactly.
    sequence_batch = move_batch(collate_prefixes([train[indices[-1]]], rules), device)
    with torch.no_grad():
        full_states, full_live, full_types = model.encode(sequence_batch)
        slot_states, live, gate_types = model.initialize_incremental(sequence_batch)
        arena = PagedKVCache(
            layers=model.action_layers_count,
            capacity=32,
            page_size=4,
            heads=model.action_heads,
            head_width=model.width // model.action_heads,
            model_width=model.width,
            device=device,
            dtype=slot_states.dtype,
        )
        handle = arena.empty_handle()
        for action_index in range(sequence_batch["action_xfers"].shape[1]):
            past_keys, past_values, past_actions, past_mask = arena.gather([handle])
            advance_kwargs = dict(
                xfer_ids=sequence_batch["action_xfers"][:, action_index],
                source_ids=sequence_batch["action_sources"][:, action_index],
                source_slots=sequence_batch["binding_slots"][:, action_index],
                destination_slots=sequence_batch["destination_slots"][:, action_index],
                destination_types=sequence_batch["destination_types"][:, action_index],
            )
            result = model.advance_incremental(
                slot_states,
                live,
                gate_types,
                past_keys if handle.length else None,
                past_values if handle.length else None,
                past_actions,
                past_mask,
                **advance_kwargs,
            )
            block_table, past_lengths = arena.block_table([handle])
            paged_result = model.advance_incremental(
                slot_states,
                live,
                gate_types,
                None,
                None,
                past_actions[:, :0],
                past_mask[:, :0],
                **advance_kwargs,
                paged_key_cache=arena.keys,
                paged_value_cache=arena.values,
                block_table=block_table,
                past_lengths=past_lengths,
            )
            for expected, actual in zip(result, paged_result):
                if expected.dtype == torch.bool or expected.dtype == torch.long:
                    assert torch.equal(expected, actual)
                else:
                    assert torch.allclose(expected, actual, atol=2e-5, rtol=2e-5)
            slot_states, live, gate_types, new_keys, new_values, action, _ = result
            readout_keys, readout_values = model.project_action_readout_kv(action)
            child = arena.append_batch(
                [handle],
                new_keys,
                new_values,
                action,
                readout_keys,
                readout_values,
            )[0]
            arena.release(handle)
            handle = child
        _, _, paged_actions, paged_mask = arena.gather([handle])
        incremental_states, incremental_live, incremental_types = (
            model.readout_incremental(
                slot_states,
                live,
                gate_types,
                paged_actions,
                paged_mask,
                sequence_batch,
            )
        )
        block_table, lengths = arena.block_table([handle])
        direct_paged_states, direct_paged_live, direct_paged_types = (
            model.readout_incremental(
                slot_states,
                live,
                gate_types,
                paged_actions[:, :0],
                paged_mask[:, :0],
                sequence_batch,
                readout_key_cache=arena.readout_keys,
                readout_value_cache=arena.readout_values,
                block_table=block_table,
                lengths=lengths,
            )
        )
        assert torch.equal(full_live, incremental_live)
        assert torch.equal(full_types, incremental_types)
        assert torch.allclose(full_states, incremental_states, atol=2e-5, rtol=2e-5)
        assert torch.equal(incremental_live, direct_paged_live)
        assert torch.equal(incremental_types, direct_paged_types)
        assert torch.allclose(
            incremental_states, direct_paged_states, atol=2e-5, rtol=2e-5
        )
        arena.release(handle)
        assert arena.allocated_pages == 0
    print(
        "paged model ok: "
        f"batch={len(indices)} max_actions={batch['action_xfers'].shape[1]} "
        f"slots={states.shape[1]} candidates={len(targets)} "
        f"incremental_equivalence={sequence_batch['action_xfers'].shape[1]}"
    )


if __name__ == "__main__":
    main()
