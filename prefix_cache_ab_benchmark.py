from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from dataset import RuleMetadata, collate_prefixes
from model_factory import build_model
from paged_cache import PagedKVCache, PrefixHandle
from threshold_inference import load_threshold_config, threshold_candidates
from train import autocast_context, move_batch


ActionKey = tuple[int, tuple[int, ...], tuple[int, ...]]
PrefixKey = tuple[ActionKey, ...]


@dataclass(frozen=True)
class PrefixNode:
    key: PrefixKey
    parent: PrefixKey | None
    action: ActionKey | None


def action_key(row: dict) -> ActionKey:
    return (
        int(row["xfer_id"]),
        tuple(map(int, row["source_slots"])),
        tuple(map(int, row["destination_slots"])),
    )


def build_prefix_levels(
    histories: list[dict], max_depth: int, max_final_states: int
) -> list[list[PrefixNode]]:
    eligible = [
        row for row in histories if len(row.get("history", ())) >= max_depth
    ][:max_final_states]
    if not eligible:
        raise ValueError(f"no history reaches requested depth {max_depth}")
    levels: list[OrderedDict[PrefixKey, PrefixNode]] = [OrderedDict()]
    root: PrefixKey = ()
    levels[0][root] = PrefixNode(root, None, None)
    for _ in range(max_depth):
        levels.append(OrderedDict())
    for row in eligible:
        parent = root
        for depth, raw_action in enumerate(row["history"][:max_depth], start=1):
            action = action_key(raw_action)
            key = parent + (action,)
            levels[depth].setdefault(key, PrefixNode(key, parent, action))
            parent = key
    return [list(level.values()) for level in levels]


def load_actual_beam_levels(
    path: Path, max_depth: int, max_states: int
) -> tuple[dict, list[list[PrefixNode]]]:
    metadata = None
    raw_levels: dict[int, list[dict]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "metadata":
            metadata = row
        elif row.get("type") == "beam_level":
            raw_levels[int(row["depth"])] = row["states"]
    if metadata is None or "initial_snapshot" not in metadata:
        raise ValueError("beam-level JSONL lacks initial_snapshot metadata")
    missing = [depth for depth in range(max_depth + 1) if depth not in raw_levels]
    if missing:
        raise ValueError(f"beam-level JSONL lacks depths: {missing}")
    levels = []
    for depth in range(max_depth + 1):
        unique: OrderedDict[PrefixKey, PrefixNode] = OrderedDict()
        for row in raw_levels[depth][:max_states]:
            key = tuple(action_key(action) for action in row["history"])
            if len(key) != depth:
                raise ValueError("beam-level history length differs from its depth")
            unique.setdefault(
                key,
                PrefixNode(
                    key,
                    key[:-1] if key else None,
                    key[-1] if key else None,
                ),
            )
        levels.append(list(unique.values()))
    for depth in range(1, max_depth + 1):
        parents = {node.key for node in levels[depth - 1]}
        absent = [node.parent for node in levels[depth] if node.parent not in parents]
        if absent:
            raise ValueError(
                f"depth {depth} has {len(absent)} children whose parent was truncated"
            )
    return metadata["initial_snapshot"], levels


def make_sample(
    initial_snapshot: dict, key: PrefixKey, rules: RuleMetadata, sample_id: int
) -> dict:
    actions = []
    for xfer_id, source_slots, destination_slots in key:
        actions.append(
            {
                "xfer_id": xfer_id,
                "source_id": int(rules.xfer_to_source[xfer_id]),
                "binding_slots": list(source_slots),
                "dst_slots": list(destination_slots),
                "anchor_slot": int(source_slots[0]),
            }
        )
    previous = actions[-1] if actions else None
    return {
        "initial_graph": initial_snapshot,
        "actions": actions,
        "matches": [],
        "local_streak": 0,
        "trajectory_id": sample_id,
        "prefix_length": len(actions),
        "previous_action": previous,
        "previous_delta": None,
        "previous_local_streak": None,
    }


def pad_current_batch(batch: dict, slots: int) -> dict:
    current_slots = batch["current_types"].shape[1]
    if current_slots > slots:
        raise RuntimeError("collated graph is wider than the incremental state")
    if current_slots == slots:
        return batch
    extra = slots - current_slots
    # `model.encode` allocates its slot state from initial_types. A microbatch
    # can omit a high-numbered dead slot that remains present in the persistent
    # paged tensor, so pad both the reconstructed current state and s0.
    if "initial_types" in batch:
        batch["initial_types"] = F.pad(
            batch["initial_types"], (0, extra), value=-1
        )
    batch["current_types"] = F.pad(batch["current_types"], (0, extra), value=-1)
    batch["current_rewrite_distance"] = F.pad(
        batch["current_rewrite_distance"], (0, extra), value=5
    )
    batch["current_touch_age"] = F.pad(
        batch["current_touch_age"], (0, extra), value=7
    )
    return batch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def candidate_keys(rows) -> list[list[tuple[int, int, tuple[int, ...]]]]:
    return [
        [(int(source), int(anchor), tuple(binding)) for source, anchor, binding, _ in row]
        for row in rows
    ]


def cache_bytes(arena: PagedKVCache) -> int:
    tensors = (
        arena.keys,
        arena.values,
        arena.actions,
        arena.readout_keys,
        arena.readout_values,
    )
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def cached_readout(
    model,
    states: torch.Tensor,
    live: torch.Tensor,
    gate_types: torch.Tensor,
    handles: list[PrefixHandle],
    arena: PagedKVCache,
    batch: dict,
):
    if model.readout_attention_backend == "paged":
        block_table, lengths = arena.block_table(handles)
        empty_actions = states.new_empty((states.shape[0], 0, model.width))
        empty_mask = torch.empty(
            (states.shape[0], 0), device=states.device, dtype=torch.bool
        )
        return model.readout_incremental(
            states,
            live,
            gate_types,
            empty_actions,
            empty_mask,
            batch,
            readout_key_cache=arena.readout_keys,
            readout_value_cache=arena.readout_values,
            block_table=block_table,
            lengths=lengths,
        )
    actions, action_mask = arena.gather_actions(handles)
    return model.readout_incremental(
        states, live, gate_types, actions, action_mask, batch
    )


@torch.no_grad()
def advance_level(
    nodes: list[PrefixNode],
    parent_nodes: list[PrefixNode],
    states: torch.Tensor,
    live: torch.Tensor,
    gate_types: torch.Tensor,
    handles: list[PrefixHandle],
    arena: PagedKVCache,
    model,
    rules: RuleMetadata,
    device: torch.device,
    microbatch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[PrefixHandle], float]:
    parent_index = {node.key: index for index, node in enumerate(parent_nodes)}
    parent_rows = [parent_index[node.parent] for node in nodes]
    max_referenced = max(
        (
            slot
            for node in nodes
            for slot in ((*node.action[1], *node.action[2]) if node.action else ())
        ),
        default=-1,
    )
    if max_referenced >= states.shape[1]:
        extra = max_referenced + 1 - states.shape[1]
        states = F.pad(states, (0, 0, 0, extra))
        live = F.pad(live, (0, extra), value=False)
        gate_types = F.pad(gate_types, (0, extra), value=-1)
    max_pattern = max(
        max(map(len, rules.source_gate_types)),
        max(map(len, rules.destination_gate_types)),
    )
    output_states = []
    output_live = []
    output_types = []
    output_handles: list[PrefixHandle] = []
    started = time.perf_counter()
    for begin in range(0, len(nodes), microbatch):
        end = min(len(nodes), begin + microbatch)
        chunk = nodes[begin:end]
        selected_parent_rows = parent_rows[begin:end]
        parent_tensor = torch.tensor(
            selected_parent_rows, dtype=torch.long, device=device
        )
        parent_handles = [handles[index] for index in selected_parent_rows]
        use_paged_attention = model.readout_attention_backend == "paged"
        if use_paged_attention:
            block_table, past_lengths = arena.block_table(parent_handles)
            past_keys = past_values = None
            past_actions = states.new_empty((len(chunk), 0, model.width))
            past_mask = torch.empty(
                (len(chunk), 0), device=device, dtype=torch.bool
            )
        else:
            past_keys, past_values, past_actions, past_mask = arena.gather(
                parent_handles
            )
        xfer_ids = torch.tensor(
            [node.action[0] for node in chunk], dtype=torch.long, device=device
        )
        source_ids = torch.tensor(
            [rules.xfer_to_source[index] for index in xfer_ids.tolist()],
            dtype=torch.long,
            device=device,
        )
        source_slots = torch.full(
            (len(chunk), max_pattern), -1, dtype=torch.long, device=device
        )
        destination_slots = torch.full_like(source_slots, -1)
        destination_types = torch.full_like(source_slots, -1)
        for row_index, node in enumerate(chunk):
            xfer_id, source, destination = node.action
            source_slots[row_index, : len(source)] = torch.tensor(
                source, dtype=torch.long, device=device
            )
            destination_slots[row_index, : len(destination)] = torch.tensor(
                destination, dtype=torch.long, device=device
            )
            types = rules.destination_gate_types[xfer_id]
            destination_types[row_index, : len(types)] = torch.tensor(
                types, dtype=torch.long, device=device
            )
        with autocast_context(device):
            result = model.advance_incremental(
                states.index_select(0, parent_tensor),
                live.index_select(0, parent_tensor),
                gate_types.index_select(0, parent_tensor),
                past_keys if past_mask.shape[1] else None,
                past_values if past_mask.shape[1] else None,
                past_actions,
                past_mask,
                xfer_ids=xfer_ids,
                source_ids=source_ids,
                source_slots=source_slots,
                destination_slots=destination_slots,
                destination_types=destination_types,
                paged_key_cache=arena.keys if use_paged_attention else None,
                paged_value_cache=arena.values if use_paged_attention else None,
                block_table=block_table if use_paged_attention else None,
                past_lengths=past_lengths if use_paged_attention else None,
            )
            readout_keys, readout_values = model.project_action_readout_kv(
                result[5]
            )
        next_states, next_live, next_types, new_keys, new_values, action, _ = result
        output_states.append(next_states)
        output_live.append(next_live)
        output_types.append(next_types)
        output_handles.extend(
            arena.append_batch(
                parent_handles,
                new_keys,
                new_values,
                action,
                readout_keys,
                readout_values,
            )
        )
    for handle in handles:
        arena.release(handle)
    synchronize(device)
    return (
        torch.cat(output_states),
        torch.cat(output_live),
        torch.cat(output_types),
        output_handles,
        time.perf_counter() - started,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict same-checkpoint A/B of full s0+action-prefix recomputation "
            "versus incremental paged state/KV reuse on one fixed beam tree."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    history_group = parser.add_mutually_exclusive_group(required=True)
    history_group.add_argument("--histories", type=Path)
    history_group.add_argument(
        "--beam-levels",
        type=Path,
        help="actual per-depth beam JSONL emitted by paged_rollout_benchmark.py",
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--max-final-states", type=int, default=1000)
    parser.add_argument("--microbatch", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument(
        "--cache-gather-backend",
        choices=("loop", "vectorized"),
        default="vectorized",
    )
    parser.add_argument(
        "--readout-attention-backend",
        choices=("eager", "sdpa", "sdpa_live", "paged"),
        default="sdpa",
    )
    parser.add_argument(
        "--shape-warmup-repeats",
        type=int,
        default=1,
        help="untimed full and paged prefix calls for each batch shape",
    )
    parser.add_argument("--target-recall", type=float, default=0.97)
    parser.add_argument("--near-target-recall", type=float)
    parser.add_argument("--far-target-recall", type=float)
    parser.add_argument("--max-source-matches", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.depths) <= 0:
        parser.error("depths must be positive")
    if args.shape_warmup_repeats < 0:
        parser.error("--shape-warmup-repeats must be nonnegative")
    requested_depths = sorted(set(args.depths))
    max_depth = max(requested_depths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    if train_args.get("architecture") != "paged_action":
        raise ValueError("checkpoint must use the paged_action architecture")
    if max_depth > int(train_args.get("max_sequence_length", 256)):
        raise ValueError("requested depth exceeds the checkpoint sequence limit")
    model = build_model(rules, len(rules.xfer_to_source), train_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.readout_attention_backend = args.readout_attention_backend
    threshold_config = load_threshold_config(
        args.calibration,
        args.target_recall,
        near_target_recall=args.near_target_recall,
        far_target_recall=args.far_target_recall,
    )
    with autocast_context(device):
        source_vectors = model.retrieval_source(model.source_representations())

    if args.beam_levels is not None:
        initial_snapshot, levels = load_actual_beam_levels(
            args.beam_levels, max_depth, args.max_final_states
        )
        workload_path = args.beam_levels
        workload_kind = "actual_per_depth_beams"
    else:
        history_payload = json.loads(args.histories.read_text())
        if "initial_snapshot" not in history_payload:
            raise ValueError(
                "history dump lacks initial_snapshot; regenerate it with the current "
                "paged_rollout_benchmark.py"
            )
        levels = build_prefix_levels(
            history_payload["states"], max_depth, args.max_final_states
        )
        initial_snapshot = history_payload["initial_snapshot"]
        workload_path = args.histories
        workload_kind = "final_survivor_prefix_tree"

    root_batch = move_batch(
        collate_prefixes(
            [make_sample(initial_snapshot, levels[0][0].key, rules, 0)], rules
        ),
        device,
    )
    with autocast_context(device):
        states, live, gate_types = model.initialize_incremental(root_batch)
    cache_pages = args.max_final_states * (
        math.ceil(max_depth / args.page_size) + 3
    )
    arena = PagedKVCache(
        layers=model.action_layers_count,
        capacity=cache_pages,
        page_size=args.page_size,
        heads=model.action_heads,
        head_width=model.width // model.action_heads,
        model_width=model.width,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else states.dtype,
        gather_backend=args.cache_gather_backend,
    )
    handles = [arena.empty_handle()]

    totals = {
        "full_encode_seconds": 0.0,
        "full_match_decode_seconds": 0.0,
        "paged_readout_seconds": 0.0,
        "paged_match_decode_seconds": 0.0,
        "paged_advance_seconds": 0.0,
    }
    comparisons = {
        "states_compared": 0,
        "candidate_sets_equal": 0,
        "candidate_rows_equal": 0,
        "candidate_key_intersection": 0,
        "candidate_key_union": 0,
        "full_candidate_keys": 0,
        "paged_candidate_keys": 0,
        "max_logit_abs_error": 0.0,
        "max_state_abs_error": 0.0,
        "live_type_mismatches": 0,
    }
    milestones = []
    per_step = []
    max_full_transient_bytes = 0
    max_paged_transient_bytes = 0

    # Warm both paths with the identical root state. This keeps CUDA context,
    # allocator, and lazy-kernel setup out of the depth-1 measurement.
    warm_batch = root_batch
    with autocast_context(device):
        warm_full_states, warm_full_live, warm_full_types = model.encode(warm_batch)
        warm_paged_states, warm_paged_live, warm_paged_types = cached_readout(
            model,
            states,
            live,
            gate_types,
            handles,
            arena,
            warm_batch,
        )
        model.match_logits(
            warm_full_states,
            warm_full_live,
            warm_full_types,
            source_vectors=source_vectors,
        )
        model.match_logits(
            warm_paged_states,
            warm_paged_live,
            warm_paged_types,
            source_vectors=source_vectors,
        )
    synchronize(device)
    # Candidate decoding creates millions of short-lived Python tuples. Cyclic
    # GC can otherwise pause whichever A/B arm happens to run second even though
    # these acyclic objects are reclaimed immediately by reference counting.
    gc.disable()

    for depth in range(max_depth):
        nodes = levels[depth]
        samples = [
            make_sample(initial_snapshot, node.key, rules, sample_id)
            for sample_id, node in enumerate(nodes)
        ]
        step = {key: 0.0 for key in totals}
        step_equal = 0
        step_set_equal = 0
        step_states = 0
        step_intersection = 0
        step_union = 0
        step_full_candidates = 0
        step_paged_candidates = 0
        step_max_logit_error = 0.0
        step_max_state_error = 0.0
        step_type_mismatches = 0

        for begin in range(0, len(nodes), args.microbatch):
            end = min(len(nodes), begin + args.microbatch)
            cpu_batch = collate_prefixes(samples[begin:end], rules)
            cpu_batch = pad_current_batch(cpu_batch, states.shape[1])
            batch = move_batch(cpu_batch, device)

            # SDPA selects/initializes a kernel for each new Q/K shape. Since
            # the strict A/B always times full recompute first, it would absorb
            # that one-time cost and make paged reuse look artificially fast.
            # Warm both paths for this exact B/N/T shape before either timer.
            for _ in range(args.shape_warmup_repeats):
                with autocast_context(device):
                    warm_full = model.encode(batch)
                    warm_paged = cached_readout(
                        model,
                        states[begin:end],
                        live[begin:end],
                        gate_types[begin:end],
                        handles[begin:end],
                        arena,
                        batch,
                    )
                synchronize(device)
                del warm_full, warm_paged

            baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with autocast_context(device):
                full_states, full_live, full_types = model.encode(batch)
            synchronize(device)
            step["full_encode_seconds"] += time.perf_counter() - started
            if device.type == "cuda":
                max_full_transient_bytes = max(
                    max_full_transient_bytes,
                    torch.cuda.max_memory_allocated(device) - baseline,
                )

            baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with autocast_context(device):
                paged_states, paged_live, paged_types = cached_readout(
                    model,
                    states[begin:end],
                    live[begin:end],
                    gate_types[begin:end],
                    handles[begin:end],
                    arena,
                    batch,
                )
            synchronize(device)
            step["paged_readout_seconds"] += time.perf_counter() - started
            if device.type == "cuda":
                max_paged_transient_bytes = max(
                    max_paged_transient_bytes,
                    torch.cuda.max_memory_allocated(device) - baseline,
                )

            if not torch.equal(full_live, paged_live) or not torch.equal(
                full_types, paged_types
            ):
                step_type_mismatches += end - begin
            state_error = float(
                (full_states.float() - paged_states.float()).abs().max().item()
            )
            step_max_state_error = max(step_max_state_error, state_error)

            started = time.perf_counter()
            with autocast_context(device):
                full_logits, full_eligible = model.match_logits(
                    full_states,
                    full_live,
                    full_types,
                    source_vectors=source_vectors,
                )
            full_candidates = threshold_candidates(
                model,
                batch,
                full_logits,
                full_eligible,
                threshold_config,
                max_candidates_per_state=args.max_source_matches,
            )
            synchronize(device)
            step["full_match_decode_seconds"] += time.perf_counter() - started

            started = time.perf_counter()
            with autocast_context(device):
                paged_logits, paged_eligible = model.match_logits(
                    paged_states,
                    paged_live,
                    paged_types,
                    source_vectors=source_vectors,
                )
            paged_candidates = threshold_candidates(
                model,
                batch,
                paged_logits,
                paged_eligible,
                threshold_config,
                max_candidates_per_state=args.max_source_matches,
            )
            synchronize(device)
            step["paged_match_decode_seconds"] += time.perf_counter() - started

            logit_error = float(
                (full_logits.float() - paged_logits.float()).abs().max().item()
            )
            step_max_logit_error = max(step_max_logit_error, logit_error)
            full_keys = candidate_keys(full_candidates)
            paged_keys = candidate_keys(paged_candidates)
            for full_row, paged_row in zip(full_keys, paged_keys):
                step_states += 1
                step_equal += int(full_row == paged_row)
                full_set = set(full_row)
                paged_set = set(paged_row)
                step_full_candidates += len(full_set)
                step_paged_candidates += len(paged_set)
                step_set_equal += int(full_set == paged_set)
                step_intersection += len(full_set & paged_set)
                step_union += len(full_set | paged_set)
            del (
                cpu_batch,
                batch,
                full_states,
                full_live,
                full_types,
                paged_states,
                paged_live,
                paged_types,
                full_logits,
                full_eligible,
                full_candidates,
                paged_logits,
                paged_eligible,
                paged_candidates,
                full_keys,
                paged_keys,
            )

        next_states, next_live, next_types, next_handles, advance_seconds = (
            advance_level(
                levels[depth + 1],
                nodes,
                states,
                live,
                gate_types,
                handles,
                arena,
                model,
                rules,
                device,
                args.microbatch,
            )
        )
        step["paged_advance_seconds"] = advance_seconds
        states, live, gate_types, handles = (
            next_states,
            next_live,
            next_types,
            next_handles,
        )
        for key in totals:
            totals[key] += step[key]
        comparisons["states_compared"] += step_states
        comparisons["candidate_sets_equal"] += step_set_equal
        comparisons["candidate_rows_equal"] += step_equal
        comparisons["candidate_key_intersection"] += step_intersection
        comparisons["candidate_key_union"] += step_union
        comparisons["full_candidate_keys"] += step_full_candidates
        comparisons["paged_candidate_keys"] += step_paged_candidates
        comparisons["max_logit_abs_error"] = max(
            comparisons["max_logit_abs_error"], step_max_logit_error
        )
        comparisons["max_state_abs_error"] = max(
            comparisons["max_state_abs_error"], step_max_state_error
        )
        comparisons["live_type_mismatches"] += step_type_mismatches
        logical_pages = sum(len(handle.blocks) for handle in handles)
        row = {
            "depth": depth + 1,
            "input_prefixes": len(nodes),
            "output_prefixes": len(levels[depth + 1]),
            **step,
            "candidate_row_agreement": step_equal / max(1, step_states),
            "candidate_set_agreement": step_set_equal / max(1, step_states),
            "candidate_key_jaccard": step_intersection / max(1, step_union),
            "full_candidate_keys": step_full_candidates,
            "paged_candidate_keys": step_paged_candidates,
            "max_logit_abs_error": step_max_logit_error,
            "max_state_abs_error": step_max_state_error,
            "allocated_cache_pages": arena.allocated_pages,
            "logical_cache_pages": logical_pages,
            "page_sharing_ratio": logical_pages / max(1, arena.allocated_pages),
        }
        per_step.append(row)
        if depth + 1 in requested_depths:
            full_total = (
                totals["full_encode_seconds"]
                + totals["full_match_decode_seconds"]
            )
            paged_total = (
                totals["paged_readout_seconds"]
                + totals["paged_match_decode_seconds"]
                + totals["paged_advance_seconds"]
            )
            # Both arms execute the same matching and structural decoder. Their
            # fixed execution order can shift allocator/cache warm-up between
            # arms, so use their mean as the shared downstream cost when
            # reporting the cache-only end-to-end comparison.
            common_match_seconds = 0.5 * (
                totals["full_match_decode_seconds"]
                + totals["paged_match_decode_seconds"]
            )
            full_prefix_seconds = totals["full_encode_seconds"]
            paged_prefix_seconds = (
                totals["paged_readout_seconds"] + totals["paged_advance_seconds"]
            )
            normalized_full_total = full_prefix_seconds + common_match_seconds
            normalized_paged_total = paged_prefix_seconds + common_match_seconds
            state_predictions = comparisons["states_compared"]
            milestone = {
                "depth": depth + 1,
                "prefixes_at_depth": len(levels[depth + 1]),
                **totals,
                "full_prediction_total_seconds": full_total,
                "paged_prediction_and_advance_total_seconds": paged_total,
                "speedup": full_total / max(paged_total, 1e-12),
                "common_match_decode_seconds": common_match_seconds,
                "normalized_full_total_seconds": normalized_full_total,
                "normalized_paged_total_seconds": normalized_paged_total,
                "normalized_speedup": (
                    normalized_full_total / max(normalized_paged_total, 1e-12)
                ),
                "prefix_state_compute_speedup": (
                    full_prefix_seconds / max(paged_prefix_seconds, 1e-12)
                ),
                "state_predictions": state_predictions,
                "full_state_predictions_per_second": (
                    state_predictions / max(full_total, 1e-12)
                ),
                "paged_state_predictions_per_second": (
                    state_predictions / max(paged_total, 1e-12)
                ),
                "full_candidate_keys": comparisons["full_candidate_keys"],
                "paged_candidate_keys": comparisons["paged_candidate_keys"],
                "full_candidate_keys_per_second": (
                    comparisons["full_candidate_keys"] / max(full_total, 1e-12)
                ),
                "paged_candidate_keys_per_second": (
                    comparisons["paged_candidate_keys"] / max(paged_total, 1e-12)
                ),
                "candidate_row_agreement": (
                    comparisons["candidate_rows_equal"]
                    / max(1, comparisons["states_compared"])
                ),
                "candidate_set_agreement": (
                    comparisons["candidate_sets_equal"]
                    / max(1, comparisons["states_compared"])
                ),
                "candidate_key_jaccard": (
                    comparisons["candidate_key_intersection"]
                    / max(1, comparisons["candidate_key_union"])
                ),
                "max_logit_abs_error": comparisons["max_logit_abs_error"],
                "max_state_abs_error": comparisons["max_state_abs_error"],
                "live_type_mismatches": comparisons["live_type_mismatches"],
                "allocated_cache_pages": arena.allocated_pages,
                "page_sharing_ratio": logical_pages / max(1, arena.allocated_pages),
            }
            milestones.append(milestone)
            print(json.dumps(milestone, sort_keys=True), flush=True)
        del samples
        # Full-prefix and paged tensors deliberately coexist inside a comparison
        # microbatch. Return their cached blocks between depths so progressively
        # wider prefixes cannot ratchet reserved memory up to an avoidable OOM.
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for handle in handles:
        arena.release(handle)
    gc.enable()
    result = {
        "mode": "strict_same_model_prefix_cache_ab",
        "checkpoint": str(args.checkpoint),
        "workload": str(workload_path),
        "workload_kind": workload_kind,
        "device": str(device),
        "requested_depths": requested_depths,
        "max_final_states": args.max_final_states,
        "microbatch": args.microbatch,
        "page_size": args.page_size,
        "cache_gather_backend": args.cache_gather_backend,
        "readout_attention_backend": args.readout_attention_backend,
        "shape_warmup_repeats": args.shape_warmup_repeats,
        "tree_widths": [len(level) for level in levels],
        "cache_capacity_pages": arena.capacity,
        "cache_storage_bytes": cache_bytes(arena),
        "max_full_transient_bytes": max_full_transient_bytes,
        "max_paged_transient_bytes": max_paged_transient_bytes,
        "milestones": milestones,
        "per_step": per_step,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
