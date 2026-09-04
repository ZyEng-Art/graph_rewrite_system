from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import ctypes.util
import gc
import importlib.util
import json
import math
import numpy as np
from pathlib import Path
import sys
import time
import types

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch
import torch.nn.functional as F

from beam_search_benchmark import (
    BeamState,
    Proposal,
    collate_states,
    snapshot,
    update_slots,
)
from dataset import RuleMetadata
from incremental_graph import parse_pattern
from lazy_rollout_benchmark import (
    ExactReplayCacheEntry,
    indexed_topology,
    lazy_child,
    raw_topology_hash,
    replay_state,
    shared_replay_prefixes,
    snapshot_signature,
    topology_digest,
)
from model_factory import build_model
from paged_cache import PagedKVCache, PrefixHandle
from gpu_proposals import GpuRuleIndex, build_gpu_proposals
from threshold_inference import (
    CandidateTensors,
    load_threshold_config,
    threshold_candidates,
    threshold_candidate_tensors,
)
from tensorized_batch import collate_paged_states
from train import autocast_context, move_batch


def initial_batch(snapshot_row: dict) -> dict:
    slots = max((int(row[0]) for row in snapshot_row["nodes"]), default=-1) + 1
    initial_types = torch.full((1, slots), -1, dtype=torch.long)
    for slot, gate_type, _ in snapshot_row["nodes"]:
        initial_types[0, int(slot)] = int(gate_type)
    edges = snapshot_row["edges"]
    return {
        "initial_types": initial_types,
        "edge_batch": torch.zeros(len(edges), dtype=torch.long),
        "edge_src": torch.tensor([row[0] for row in edges], dtype=torch.long),
        "edge_dst": torch.tensor([row[1] for row in edges], dtype=torch.long),
        "edge_relation": torch.tensor(
            [row[2] * 4 + row[3] for row in edges], dtype=torch.long
        ),
    }


def pad_current_batch(batch: dict, slots: int) -> dict:
    current_slots = batch["current_types"].shape[1]
    if current_slots > slots:
        raise RuntimeError("current graph uses a slot absent from the causal cache")
    if current_slots == slots:
        return batch
    extra = slots - current_slots
    batch["current_types"] = F.pad(batch["current_types"], (0, extra), value=-1)
    batch["current_rewrite_distance"] = F.pad(
        batch["current_rewrite_distance"], (0, extra), value=5
    )
    batch["current_touch_age"] = F.pad(
        batch["current_touch_age"], (0, extra), value=7
    )
    return batch


def expand_action_rows(predicted, source_to_xfers: dict[int, list[int]]):
    action_rows = []
    for rows in predicted:
        expanded = []
        for source, anchor, binding, probability in rows:
            expanded.extend(
                (xfer_id, anchor, binding, probability)
                for xfer_id in source_to_xfers[source]
            )
        action_rows.append(expanded)
    return action_rows


def build_legacy_proposals(
    beam: list[BeamState],
    predicted,
    exploration_predicted,
    source_to_xfers: dict[int, list[int]],
    gate_deltas: list[int],
    *,
    beam_size: int,
    max_actions_per_parent: int,
    exploration_actions_per_parent: int,
    proposal_factor: int,
    max_gate_increase: int,
):
    action_expand_started = time.perf_counter()
    action_rows = expand_action_rows(predicted, source_to_xfers)
    exploration_action_rows = (
        expand_action_rows(exploration_predicted, source_to_xfers)
        if exploration_predicted is not None
        else [[] for _ in action_rows]
    )
    action_expand_seconds = time.perf_counter() - action_expand_started

    proposal_started = time.perf_counter()
    proposals = []
    eligible_actions = 0
    exploration_eligible_actions = 0
    selected_exploration_proposals = 0
    effective_parent_cap = max(
        max_actions_per_parent,
        math.ceil(beam_size / max(1, len(beam))) * 2,
    )
    for parent_index, (state, rows, exploration_rows) in enumerate(
        zip(beam, action_rows, exploration_action_rows)
    ):
        def make_proposals(candidate_rows):
            result = []
            for xfer_id, anchor, binding, probability in candidate_rows:
                delta = gate_deltas[xfer_id]
                if delta <= max_gate_increase:
                    result.append(
                        Proposal(
                            parent=parent_index,
                            xfer_id=xfer_id,
                            anchor_slot=anchor,
                            binding=binding,
                            probability=probability,
                            next_gate_count=state.gate_count + delta,
                        )
                    )
            result.sort(
                key=lambda row: (
                    row.next_gate_count,
                    -row.probability,
                    row.xfer_id,
                )
            )
            return result

        primary_proposals = make_proposals(rows)
        secondary_proposals = make_proposals(exploration_rows)
        eligible_actions += len(primary_proposals)
        exploration_eligible_actions += len(secondary_proposals)
        quota = min(exploration_actions_per_parent, effective_parent_cap)
        primary_limit = effective_parent_cap - quota
        parent_proposals = primary_proposals[:primary_limit]
        selected_keys = {
            (row.xfer_id, row.anchor_slot, row.binding) for row in parent_proposals
        }
        for proposal in secondary_proposals:
            key = (proposal.xfer_id, proposal.anchor_slot, proposal.binding)
            if key in selected_keys:
                continue
            parent_proposals.append(proposal)
            selected_keys.add(key)
            selected_exploration_proposals += 1
            if len(parent_proposals) >= effective_parent_cap:
                break
        if len(parent_proposals) < effective_parent_cap:
            for proposal in primary_proposals[primary_limit:]:
                key = (proposal.xfer_id, proposal.anchor_slot, proposal.binding)
                if key in selected_keys:
                    continue
                parent_proposals.append(proposal)
                selected_keys.add(key)
                if len(parent_proposals) >= effective_parent_cap:
                    break
        parent_proposals.sort(
            key=lambda row: (row.next_gate_count, -row.probability, row.xfer_id)
        )
        proposals.extend(parent_proposals)
    proposals.sort(
        key=lambda row: (
            row.next_gate_count,
            -row.probability,
            beam[row.parent].gate_count,
        )
    )
    proposals = proposals[: beam_size * proposal_factor]
    proposal_seconds = time.perf_counter() - proposal_started
    return proposals, {
        "predicted_actions": sum(map(len, action_rows)),
        "exploration_predicted_actions": sum(
            map(len, exploration_action_rows)
        ),
        "eligible_actions": eligible_actions,
        "exploration_eligible_actions": exploration_eligible_actions,
        "selected_exploration_proposals": selected_exploration_proposals,
        "action_expansion_seconds": action_expand_seconds,
        "proposal_seconds": proposal_seconds,
    }


def serialized_history_state(state: BeamState) -> dict:
    return {
        "gate_count": state.gate_count,
        "history": [
            {
                "xfer_id": action.xfer_id,
                "source_slots": list(action.source_slots),
                "destination_slots": list(action.destination_slots),
            }
            for action in state.history
        ],
    }


def append_beam_level(path: Path, depth: int, beam: list[BeamState]) -> None:
    with path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "type": "beam_level",
                    "depth": depth,
                    "states": [serialized_history_state(state) for state in beam],
                },
                sort_keys=True,
            )
            + "\n"
        )


@torch.no_grad()
def paged_model_matches(
    beam: list[BeamState],
    slot_states: torch.Tensor,
    live: torch.Tensor,
    gate_types: torch.Tensor,
    handles: list[PrefixHandle],
    arena: PagedKVCache,
    model,
    device,
    threshold_config,
    source_vectors: torch.Tensor,
    microbatch: int,
    max_candidates: int,
    state_batch_backend: str = "legacy",
    candidate_backend: str = "legacy",
    profile_stages: bool = False,
) -> tuple[
    list[list[tuple[int, int, tuple[int, ...], float]]] | CandidateTensors,
    float,
    dict[str, float],
]:
    output = []
    candidate_chunks: list[CandidateTensors] = []
    timing: dict[str, float] = {}

    def finish_timing(name: str, stage_started: float) -> None:
        if not profile_stages:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - stage_started

    started = time.perf_counter()
    for begin in range(0, len(beam), microbatch):
        end = min(len(beam), begin + microbatch)
        selected_states = slot_states[begin:end]
        selected_live = live[begin:end]
        selected_types = gate_types[begin:end]

        stage_started = time.perf_counter()
        if state_batch_backend == "tensorized":
            if model.readout_attention_backend == "sdpa_live":
                raise ValueError(
                    "tensorized state batches do not build compact live-slot rows"
                )
            cpu_batch = collate_paged_states(
                beam[begin:end], selected_types
            )
        else:
            cpu_batch = pad_current_batch(
                collate_states(beam[begin:end]), selected_states.shape[1]
            )
        finish_timing("batch_collate_and_pad_seconds", stage_started)

        stage_started = time.perf_counter()
        batch = move_batch(cpu_batch, device)
        finish_timing("batch_host_to_device_seconds", stage_started)
        if not torch.equal(selected_types, batch["current_types"]):
            raise RuntimeError("paged live/type cache differs from lazy topology")

        stage_started = time.perf_counter()
        if model.readout_attention_backend == "paged":
            block_table, lengths = arena.block_table(handles[begin:end])
            empty_actions = selected_states.new_empty(
                (end - begin, 0, model.width)
            )
            empty_mask = torch.empty(
                (end - begin, 0), device=device, dtype=torch.bool
            )
        else:
            actions, action_mask = arena.gather_actions(handles[begin:end])
        finish_timing("readout_cache_metadata_or_gather_seconds", stage_started)

        stage_started = time.perf_counter()
        with autocast_context(device):
            if model.readout_attention_backend == "paged":
                encoded, _, _ = model.readout_incremental(
                    selected_states,
                    selected_live,
                    selected_types,
                    empty_actions,
                    empty_mask,
                    batch,
                    readout_key_cache=arena.readout_keys,
                    readout_value_cache=arena.readout_values,
                    block_table=block_table,
                    lengths=lengths,
                )
            else:
                actions, action_mask = arena.gather_actions(handles[begin:end])
                encoded, _, _ = model.readout_incremental(
                    selected_states,
                    selected_live,
                    selected_types,
                    actions,
                    action_mask,
                    batch,
                )
        finish_timing("incremental_graph_readout_seconds", stage_started)

        stage_started = time.perf_counter()
        with autocast_context(device):
            logits, eligible = model.match_logits(
                encoded, selected_live, selected_types, source_vectors=source_vectors
            )
        finish_timing("match_logits_seconds", stage_started)
        if candidate_backend == "gpu":
            candidate_chunks.append(
                threshold_candidate_tensors(
                    model,
                    batch,
                    logits,
                    eligible,
                    threshold_config,
                    max_candidates_per_state=max_candidates,
                    batch_offset=begin,
                    timing=timing if profile_stages else None,
                )
            )
        else:
            output.extend(
                threshold_candidates(
                    model,
                    batch,
                    logits,
                    eligible,
                    threshold_config,
                    max_candidates_per_state=max_candidates,
                    timing=timing if profile_stages else None,
                )
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if profile_stages:
        timing["model_match_unattributed_seconds"] = max(
            0.0, elapsed - sum(timing.values())
        )
    candidates = (
        CandidateTensors.cat(candidate_chunks)
        if candidate_backend == "gpu"
        else output
    )
    return candidates, elapsed, timing


@torch.no_grad()
def advance_selected(
    old_states: torch.Tensor,
    old_live: torch.Tensor,
    old_types: torch.Tensor,
    old_handles: list[PrefixHandle],
    records: list[tuple[BeamState, Proposal]],
    rules: RuleMetadata,
    arena: PagedKVCache,
    model,
    device,
    microbatch: int,
    profile_stages: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[PrefixHandle],
    float,
    dict[str, float],
]:
    timing: dict[str, float] = {}

    def finish_timing(name: str, stage_started: float) -> None:
        if not profile_stages:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - stage_started

    started = time.perf_counter()
    stage_started = time.perf_counter()
    max_slots = max(record[0].next_slot for record in records)
    if old_states.shape[1] < max_slots:
        extra = max_slots - old_states.shape[1]
        old_states = F.pad(old_states, (0, 0, 0, extra))
        old_live = F.pad(old_live, (0, extra), value=False)
        old_types = F.pad(old_types, (0, extra), value=-1)
    max_source = max(len(record[1].binding or ()) for record in records)
    max_destination = max(len(record[0].history[-1].destination_slots) for record in records)
    result_states = []
    result_live = []
    result_types = []
    result_handles = []
    finish_timing("advance_shape_setup_seconds", stage_started)

    for begin in range(0, len(records), microbatch):
        chunk = records[begin : begin + microbatch]
        stage_started = time.perf_counter()
        parent_indices = [record[1].parent for record in chunk]
        parents = torch.tensor(
            parent_indices, dtype=torch.long, device=device
        )
        parent_handles = [old_handles[index] for index in parent_indices]
        use_paged_attention = model.readout_attention_backend == "paged"
        if use_paged_attention:
            block_table, past_lengths = arena.block_table(parent_handles)
            past_keys = past_values = None
            past_actions = old_states.new_empty((len(chunk), 0, model.width))
            past_mask = torch.empty(
                (len(chunk), 0), device=device, dtype=torch.bool
            )
        else:
            past_keys, past_values, past_actions, past_mask = arena.gather(
                parent_handles
            )
        finish_timing("advance_cache_metadata_or_gather_seconds", stage_started)

        stage_started = time.perf_counter()
        batch_size = len(chunk)
        xfer_ids_cpu = np.fromiter(
            (record[1].xfer_id for record in chunk),
            dtype=np.int64,
            count=batch_size,
        )
        source_ids_cpu = np.fromiter(
            (rules.xfer_to_source[xfer_id] for xfer_id in xfer_ids_cpu),
            dtype=np.int64,
            count=batch_size,
        )
        source_slots_cpu = np.full(
            (batch_size, max_source), -1, dtype=np.int64
        )
        destination_slots_cpu = np.full(
            (batch_size, max_destination), -1, dtype=np.int64
        )
        destination_types_cpu = np.full_like(destination_slots_cpu, -1)
        for row_index, (child, proposal) in enumerate(chunk):
            binding = proposal.binding or ()
            destination = child.history[-1].destination_slots
            source_slots_cpu[row_index, : len(binding)] = binding
            destination_slots_cpu[row_index, : len(destination)] = destination
            types = rules.destination_gate_types[proposal.xfer_id]
            destination_types_cpu[row_index, : len(types)] = types
        xfer_ids = torch.from_numpy(xfer_ids_cpu).to(device)
        source_ids = torch.from_numpy(source_ids_cpu).to(device)
        source_slots = torch.from_numpy(source_slots_cpu).to(device)
        destination_slots = torch.from_numpy(destination_slots_cpu).to(device)
        destination_types = torch.from_numpy(destination_types_cpu).to(device)
        finish_timing("advance_action_tensor_pack_seconds", stage_started)

        stage_started = time.perf_counter()
        with autocast_context(device):
            advanced = model.advance_incremental(
                old_states.index_select(0, parents),
                old_live.index_select(0, parents),
                old_types.index_select(0, parents),
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
        finish_timing("causal_model_advance_seconds", stage_started)

        stage_started = time.perf_counter()
        with autocast_context(device):
            readout_keys, readout_values = model.project_action_readout_kv(
                advanced[5]
            )
        finish_timing("readout_kv_projection_seconds", stage_started)

        stage_started = time.perf_counter()
        states, live, types, new_keys, new_values, action, _ = advanced
        child_handles = arena.append_batch(
            parent_handles,
            new_keys,
            new_values,
            action,
            readout_keys,
            readout_values,
        )
        finish_timing("cache_append_and_cow_seconds", stage_started)
        result_states.append(states)
        result_live.append(live)
        result_types.append(types)
        result_handles.extend(child_handles)

    stage_started = time.perf_counter()
    for handle in old_handles:
        arena.release(handle)
    concatenated_states = torch.cat(result_states)
    concatenated_live = torch.cat(result_live)
    concatenated_types = torch.cat(result_types)
    finish_timing("cache_release_and_output_concat_seconds", stage_started)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if profile_stages:
        timing["cache_advance_unattributed_seconds"] = max(
            0.0, elapsed - sum(timing.values())
        )
    return (
        concatenated_states,
        concatenated_live,
        concatenated_types,
        result_handles,
        elapsed,
        timing,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--exploration-checkpoint", type=Path)
    parser.add_argument("--exploration-calibration", type=Path)
    parser.add_argument(
        "--exploration-actions-per-parent",
        type=int,
        default=0,
        help="reserve this many per-parent proposals for a second paged model",
    )
    parser.add_argument(
        "--exploration-until-depth",
        type=int,
        default=0,
        help="last depth using the second model (0 keeps it active throughout)",
    )
    parser.add_argument("--target-recall", type=float, default=0.97)
    parser.add_argument("--near-target-recall", type=float)
    parser.add_argument("--far-target-recall", type=float)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=64)
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
        "--state-batch-backend",
        choices=("legacy", "tensorized"),
        default="legacy",
        help="reuse paged GPU types and bulk-pack graph/locality metadata",
    )
    parser.add_argument(
        "--proposal-backend",
        choices=("legacy", "gpu"),
        default="legacy",
        help="rank and cap expanded actions on GPU before compact D2H",
    )
    parser.add_argument("--cache-pages", type=int)
    parser.add_argument("--max-source-matches", type=int, default=2048)
    parser.add_argument("--max-actions-per-parent", type=int, default=128)
    parser.add_argument("--proposal-factor", type=int, default=16)
    parser.add_argument("--max-gate-increase", type=int, default=1)
    parser.add_argument(
        "--dedup-mode", choices=("none", "raw", "canonical"), default="raw"
    )
    parser.add_argument(
        "--lazy-topology-backend",
        choices=("legacy", "indexed"),
        default="legacy",
        help="use indexed port adjacency and incremental raw fingerprints",
    )
    parser.add_argument("--structural-recheck", action="store_true")
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=0,
        help="Quartz-replay and filter the beam every this many actions (0 disables)",
    )
    parser.add_argument(
        "--refresh-factor",
        type=int,
        default=2,
        help="over-generate this many beam widths before an exact refresh",
    )
    parser.add_argument("--audit-count", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-audit-count",
        type=int,
        default=0,
        help=(
            "at each refresh, compare this many checkpoint-based replays against "
            "a full replay from s0 (0 disables)"
        ),
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help=(
            "synchronize at fine-grained stage boundaries and report an exclusive "
            "search-time breakdown"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-qasm", type=Path)
    parser.add_argument("--dump-beam-histories", type=Path)
    parser.add_argument(
        "--dump-beam-levels",
        type=Path,
        help="write every actual per-depth beam as JSONL for strict cache A/B",
    )
    args = parser.parse_args()
    if args.refresh_interval < 0:
        parser.error("--refresh-interval must be nonnegative")
    if args.refresh_factor < 1:
        parser.error("--refresh-factor must be at least one")
    if args.checkpoint_audit_count < 0:
        parser.error("--checkpoint-audit-count must be nonnegative")
    if (
        args.lazy_topology_backend == "indexed"
        and args.state_batch_backend != "tensorized"
    ):
        parser.error("indexed lazy topology requires tensorized state batches")
    if (args.exploration_checkpoint is None) != (
        args.exploration_calibration is None
    ):
        parser.error(
            "--exploration-checkpoint and --exploration-calibration must be used together"
        )
    if args.exploration_actions_per_parent < 0:
        parser.error("--exploration-actions-per-parent must be nonnegative")
    if args.exploration_until_depth < 0:
        parser.error("--exploration-until-depth must be nonnegative")
    if args.exploration_actions_per_parent and args.exploration_checkpoint is None:
        parser.error("an exploration checkpoint is required for an exploration quota")
    if args.exploration_actions_per_parent > args.max_actions_per_parent:
        parser.error(
            "--exploration-actions-per-parent cannot exceed --max-actions-per-parent"
        )
    if (
        args.state_batch_backend == "tensorized"
        and args.readout_attention_backend == "sdpa_live"
    ):
        parser.error("tensorized state batches do not support sdpa_live")

    # Quartz imports these packages unconditionally for conversion and DGL
    # helpers, while this benchmark uses only its compiled graph API.  Supply
    # empty placeholders only when the optional packages are absent.
    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    if train_args.get("architecture") != "paged_action":
        raise ValueError("paged rollout requires a paged_action checkpoint")
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
    with torch.no_grad(), autocast_context(device):
        source_vectors = model.retrieval_source(model.source_representations())

    exploration_model = None
    exploration_threshold_config = None
    exploration_source_vectors = None
    exploration_checkpoint = None
    if args.exploration_checkpoint is not None:
        exploration_checkpoint = torch.load(
            args.exploration_checkpoint, map_location="cpu", weights_only=False
        )
        exploration_args = exploration_checkpoint["args"]
        if exploration_args.get("architecture") != "paged_action":
            raise ValueError("exploration checkpoint must use paged_action")
        exploration_model = build_model(
            rules, len(rules.xfer_to_source), exploration_args
        ).to(device)
        exploration_model.load_state_dict(exploration_checkpoint["model"])
        exploration_model.eval()
        exploration_model.readout_attention_backend = (
            args.readout_attention_backend
        )
        if (
            exploration_model.width != model.width
            or exploration_model.action_layers_count != model.action_layers_count
            or exploration_model.action_heads != model.action_heads
        ):
            raise ValueError("primary and exploration paged-cache shapes differ")
        exploration_threshold_config = load_threshold_config(
            args.exploration_calibration,
            args.target_recall,
            near_target_recall=args.near_target_recall,
            far_target_recall=args.far_target_recall,
        )
        with torch.no_grad(), autocast_context(device):
            exploration_source_vectors = exploration_model.retrieval_source(
                exploration_model.source_representations()
            )

    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    if context.num_xfers != len(rules.xfer_to_source):
        raise RuntimeError("dataset and Quartz context have different xfer counts")
    xfers = [
        context.get_xfer_from_id(id=index) for index in range(context.num_xfers)
    ]
    source_to_xfers: dict[int, list[int]] = defaultdict(list)
    for xfer_id, source_id in enumerate(rules.xfer_to_source):
        source_to_xfers[source_id].append(xfer_id)
    gate_deltas = [
        len(rules.destination_gate_types[index])
        - len(rules.source_gate_types[rules.xfer_to_source[index]])
        for index in range(len(rules.xfer_to_source))
    ]
    gpu_rule_index = (
        GpuRuleIndex.build(
            source_to_xfers,
            gate_deltas,
            len(rules.source_gate_types),
            args.max_gate_increase,
            device,
        )
        if args.proposal_backend == "gpu"
        else None
    )
    source_patterns = tuple(parse_pattern(pattern) for pattern in rules.xfer_sources)
    destination_patterns = tuple(
        parse_pattern(pattern) for pattern in rules.xfer_destinations
    )

    graph = quartz.PyGraph.from_qasm(context=context, filename=str(args.qasm))
    initial_qasm = graph.to_qasm_str()
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    initial_snapshot = snapshot(graph, guid_to_slot)
    initial_gate_count = int(graph.gate_count)
    beam = [
        BeamState(
            graph=None,
            snapshot=initial_snapshot,
            guid_to_slot={},
            next_slot=next_slot,
            last_touched={},
            rewrite_distance={int(row[0]): 5 for row in initial_snapshot["nodes"]},
            previous_preferred=set(),
            local_streak=0,
            gate_count=initial_gate_count,
            depth=0,
            history=(),
            topology_index=(
                indexed_topology(initial_snapshot)
                if args.lazy_topology_backend == "indexed"
                else None
            ),
            exact_graph_checkpoint=graph,
            exact_slot_checkpoint=dict(guid_to_slot),
            exact_checkpoint_depth=0,
        )
    ]
    if args.dump_beam_levels is not None:
        args.dump_beam_levels.parent.mkdir(parents=True, exist_ok=True)
        args.dump_beam_levels.write_text(
            json.dumps(
                {
                    "type": "metadata",
                    "qasm": str(args.qasm),
                    "initial_qasm": initial_qasm,
                    "initial_snapshot": initial_snapshot,
                },
                sort_keys=True,
            )
            + "\n"
        )
        append_beam_level(args.dump_beam_levels, 0, beam)
    with torch.no_grad(), autocast_context(device):
        slot_states, live, gate_types = model.initialize_incremental(
            move_batch(initial_batch(initial_snapshot), device)
        )
    cache_pages = args.cache_pages or args.beam_size * (
        math.ceil(args.depth / args.page_size) + 3
    )
    arena = PagedKVCache(
        layers=model.action_layers_count,
        capacity=cache_pages,
        page_size=args.page_size,
        heads=model.action_heads,
        head_width=model.width // model.action_heads,
        model_width=model.width,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else slot_states.dtype,
        gather_backend=args.cache_gather_backend,
    )
    handles = [arena.empty_handle()]
    exploration_slot_states = None
    exploration_live = None
    exploration_gate_types = None
    exploration_arena = None
    exploration_handles = None
    if exploration_model is not None:
        with torch.no_grad(), autocast_context(device):
            (
                exploration_slot_states,
                exploration_live,
                exploration_gate_types,
            ) = exploration_model.initialize_incremental(
                move_batch(initial_batch(initial_snapshot), device)
            )
        exploration_arena = PagedKVCache(
            layers=exploration_model.action_layers_count,
            capacity=cache_pages,
            page_size=args.page_size,
            heads=exploration_model.action_heads,
            head_width=exploration_model.width // exploration_model.action_heads,
            model_width=exploration_model.width,
            device=device,
            dtype=(
                torch.bfloat16
                if device.type == "cuda"
                else exploration_slot_states.dtype
            ),
            gather_backend=args.cache_gather_backend,
        )
        exploration_handles = [exploration_arena.empty_handle()]
    del payload, graph, checkpoint, exploration_checkpoint
    gc.collect()
    gc.disable()

    if args.dedup_mode == "canonical":
        seen = {topology_digest(initial_snapshot)}
    elif args.dedup_mode == "raw":
        seen = {raw_topology_hash(initial_snapshot)}
    else:
        seen = set()

    step_rows = []
    total_started = time.perf_counter()
    for step in range(args.depth):
        step_started = time.perf_counter()
        if (
            exploration_model is not None
            and args.exploration_until_depth
            and step >= args.exploration_until_depth
        ):
            for handle in exploration_handles:
                exploration_arena.release(handle)
            exploration_handles = None
            exploration_slot_states = None
            exploration_live = None
            exploration_gate_types = None
            exploration_source_vectors = None
            exploration_model = None
        refresh_due = bool(
            args.refresh_interval and (step + 1) % args.refresh_interval == 0
        )
        input_state_count = len(beam)
        use_gpu_proposals = (
            args.proposal_backend == "gpu" and exploration_model is None
        )
        predicted, model_seconds, model_timing = paged_model_matches(
            beam,
            slot_states,
            live,
            gate_types,
            handles,
            arena,
            model,
            device,
            threshold_config,
            source_vectors,
            args.microbatch,
            args.max_source_matches,
            args.state_batch_backend,
            "gpu" if use_gpu_proposals else "legacy",
            args.profile_stages,
        )
        exploration_predicted = None
        exploration_model_seconds = 0.0
        exploration_model_timing: dict[str, float] = {}
        if exploration_model is not None:
            (
                exploration_predicted,
                exploration_model_seconds,
                exploration_model_timing,
            ) = paged_model_matches(
                beam,
                exploration_slot_states,
                exploration_live,
                exploration_gate_types,
                exploration_handles,
                exploration_arena,
                exploration_model,
                device,
                exploration_threshold_config,
                exploration_source_vectors,
                args.microbatch,
                args.max_source_matches,
                args.state_batch_backend,
                "legacy",
                args.profile_stages,
            )

        effective_parent_cap = max(
            args.max_actions_per_parent,
            math.ceil(args.beam_size / max(1, len(beam))) * 2,
        )
        gpu_proposal_timing: dict[str, float] = {}
        if use_gpu_proposals:
            proposals, proposal_metrics, gpu_proposal_timing = build_gpu_proposals(
                predicted,
                beam,
                gpu_rule_index,
                per_parent_cap=effective_parent_cap,
                global_cap=args.beam_size * args.proposal_factor,
                profile_stages=args.profile_stages,
            )
            predicted_action_count = proposal_metrics["predicted_actions"]
            eligible_actions = proposal_metrics["eligible_actions"]
            exploration_predicted_action_count = 0
            exploration_eligible_actions = 0
            selected_exploration_proposals = 0
            action_expand_seconds = gpu_proposal_timing.get(
                "gpu_action_expansion_seconds", 0.0
            )
            proposal_seconds = sum(gpu_proposal_timing.values()) - action_expand_seconds
        else:
            proposals, proposal_metrics = build_legacy_proposals(
                beam,
                predicted,
                exploration_predicted,
                source_to_xfers,
                gate_deltas,
                beam_size=args.beam_size,
                max_actions_per_parent=args.max_actions_per_parent,
                exploration_actions_per_parent=args.exploration_actions_per_parent,
                proposal_factor=args.proposal_factor,
                max_gate_increase=args.max_gate_increase,
            )
            predicted_action_count = proposal_metrics["predicted_actions"]
            exploration_predicted_action_count = proposal_metrics[
                "exploration_predicted_actions"
            ]
            eligible_actions = proposal_metrics["eligible_actions"]
            exploration_eligible_actions = proposal_metrics[
                "exploration_eligible_actions"
            ]
            selected_exploration_proposals = proposal_metrics[
                "selected_exploration_proposals"
            ]
            action_expand_seconds = proposal_metrics["action_expansion_seconds"]
            proposal_seconds = proposal_metrics["proposal_seconds"]

        update_started = time.perf_counter()
        records: list[tuple[BeamState, Proposal]] = []
        attempted = invalid = duplicates = 0
        accepted_target = args.beam_size * (
            args.refresh_factor if refresh_due else 1
        )
        for proposal in proposals:
            if len(records) >= accepted_target:
                break
            attempted += 1
            child, fingerprint, is_duplicate = lazy_child(
                beam[proposal.parent],
                proposal,
                source_patterns,
                destination_patterns,
                args.structural_recheck,
                args.dedup_mode,
                seen,
                args.lazy_topology_backend,
            )
            if is_duplicate:
                duplicates += 1
                continue
            if child is None:
                invalid += 1
                continue
            if fingerprint is not None:
                seen.add(fingerprint)
            records.append((child, proposal))
        lazy_update_seconds = time.perf_counter() - update_started
        if not records:
            break
        records.sort(key=lambda row: (row[0].gate_count, len(row[0].history)))
        records = records[:accepted_target]
        cache_result = advance_selected(
            slot_states,
            live,
            gate_types,
            handles,
            records,
            rules,
            arena,
            model,
            device,
            args.microbatch,
            args.profile_stages,
        )
        (
            slot_states,
            live,
            gate_types,
            handles,
            cache_seconds,
            cache_timing,
        ) = cache_result
        exploration_cache_seconds = 0.0
        exploration_cache_timing: dict[str, float] = {}
        if exploration_model is not None:
            exploration_cache_result = advance_selected(
                exploration_slot_states,
                exploration_live,
                exploration_gate_types,
                exploration_handles,
                records,
                rules,
                exploration_arena,
                exploration_model,
                device,
                args.microbatch,
                args.profile_stages,
            )
            (
                exploration_slot_states,
                exploration_live,
                exploration_gate_types,
                exploration_handles,
                exploration_cache_seconds,
                exploration_cache_timing,
            ) = exploration_cache_result
        beam = [record[0] for record in records]

        refresh_seconds = 0.0
        refresh_candidates = 0
        refresh_attempted = 0
        refresh_valid = 0
        refresh_early_stopped = False
        refresh_failures: dict[int, int] = defaultdict(int)
        refresh_profile_seconds: dict[str, float] = {}
        refresh_profile_counts: dict[str, int] = {}
        checkpoint_audited = 0
        checkpoint_mismatches = 0
        if refresh_due:
            refresh_started = time.perf_counter()
            refresh_candidates = len(beam)
            valid_indices = []
            refresh_checkpoints = {}
            refresh_replay_cache: dict[
                tuple, ExactReplayCacheEntry
            ] | None = None
            refresh_shared_prefixes = shared_replay_prefixes(beam)
            if refresh_shared_prefixes:
                refresh_replay_cache = {}
                refresh_profile_counts["replay_cache_shared_prefixes"] = len(
                    refresh_shared_prefixes
                )
            for state_index, state in enumerate(beam):
                refresh_attempted += 1
                (
                    exact_graph,
                    failure_step,
                    topology_ok,
                    exact_slots,
                ) = replay_state(
                    state,
                    context,
                    quartz.PyGraph,
                    xfers,
                    initial_qasm,
                    return_checkpoint=True,
                    profile_timing=(
                        refresh_profile_seconds if args.profile_stages else None
                    ),
                    profile_counts=(
                        refresh_profile_counts if args.profile_stages else None
                    ),
                    replay_cache=refresh_replay_cache,
                    replay_cache_prefixes=refresh_shared_prefixes,
                )
                if (
                    checkpoint_audited < args.checkpoint_audit_count
                    and state.exact_graph_checkpoint is not None
                ):
                    (
                        full_graph,
                        full_failure_step,
                        full_topology_ok,
                        full_slots,
                    ) = replay_state(
                        state,
                        context,
                        quartz.PyGraph,
                        xfers,
                        initial_qasm,
                        return_checkpoint=True,
                        ignore_checkpoint=True,
                        profile_timing=(
                            refresh_profile_seconds if args.profile_stages else None
                        ),
                        profile_counts=(
                            refresh_profile_counts if args.profile_stages else None
                        ),
                    )
                    checkpoint_audited += 1
                    checkpoint_result = (
                        failure_step,
                        topology_ok,
                        None
                        if exact_graph is None
                        else snapshot_signature(snapshot(exact_graph, exact_slots)),
                    )
                    full_result = (
                        full_failure_step,
                        full_topology_ok,
                        None
                        if full_graph is None
                        else snapshot_signature(snapshot(full_graph, full_slots)),
                    )
                    if checkpoint_result != full_result:
                        checkpoint_mismatches += 1
                        raise RuntimeError(
                            "checkpoint replay diverged from full s0 replay: "
                            f"state={state_index}, checkpoint={checkpoint_result[:2]}, "
                            f"full={full_result[:2]}"
                        )
                if exact_graph is None:
                    refresh_failures[int(failure_step)] += 1
                    continue
                if not topology_ok:
                    refresh_failures[-1] += 1
                    continue
                valid_indices.append(state_index)
                refresh_checkpoints[state_index] = (exact_graph, exact_slots)
                if len(valid_indices) >= args.beam_size:
                    refresh_early_stopped = state_index + 1 < len(beam)
                    break
            refresh_valid = len(valid_indices)
            keep_indices = valid_indices[: args.beam_size]
            for state_index in keep_indices:
                state = beam[state_index]
                exact_graph, exact_slots = refresh_checkpoints[state_index]
                state.exact_graph_checkpoint = exact_graph
                state.exact_slot_checkpoint = exact_slots
                state.exact_checkpoint_depth = len(state.history)
            refresh_seconds = time.perf_counter() - refresh_started
        else:
            keep_indices = list(range(min(args.beam_size, len(beam))))

        beam_prune_started = time.perf_counter()
        keep_set = set(keep_indices)
        for state_index, handle in enumerate(handles):
            if state_index not in keep_set:
                arena.release(handle)
        if exploration_handles is not None:
            for state_index, handle in enumerate(exploration_handles):
                if state_index not in keep_set:
                    exploration_arena.release(handle)
        if not keep_indices:
            beam = []
            handles = []
            break
        index = torch.tensor(keep_indices, dtype=torch.long, device=device)
        slot_states = slot_states.index_select(0, index)
        live = live.index_select(0, index)
        gate_types = gate_types.index_select(0, index)
        handles = [handles[state_index] for state_index in keep_indices]
        if exploration_model is not None:
            exploration_slot_states = exploration_slot_states.index_select(0, index)
            exploration_live = exploration_live.index_select(0, index)
            exploration_gate_types = exploration_gate_types.index_select(0, index)
            exploration_handles = [
                exploration_handles[state_index] for state_index in keep_indices
            ]
        beam = [beam[state_index] for state_index in keep_indices]
        if args.profile_stages and device.type == "cuda":
            torch.cuda.synchronize(device)
        beam_prune_seconds = time.perf_counter() - beam_prune_started
        logical_pages = sum(len(handle.blocks) for handle in handles)
        elapsed = time.perf_counter() - step_started
        stage_seconds: dict[str, float] = {}
        if args.profile_stages:
            stage_seconds.update(
                {f"primary_{name}": value for name, value in model_timing.items()}
            )
            stage_seconds.update(
                {
                    f"exploration_{name}": value
                    for name, value in exploration_model_timing.items()
                }
            )
            if use_gpu_proposals:
                stage_seconds.update(gpu_proposal_timing)
            else:
                stage_seconds["action_expansion_seconds"] = action_expand_seconds
                stage_seconds["proposal_build_and_sort_seconds"] = proposal_seconds
            stage_seconds["lazy_topology_and_hash_seconds"] = lazy_update_seconds
            stage_seconds.update(cache_timing)
            stage_seconds.update(
                {
                    f"exploration_{name}": value
                    for name, value in exploration_cache_timing.items()
                }
            )
            stage_seconds["exact_quartz_refresh_seconds"] = refresh_seconds
            stage_seconds["beam_prune_and_reindex_seconds"] = beam_prune_seconds
            stage_seconds["step_unattributed_seconds"] = max(
                0.0, elapsed - sum(stage_seconds.values())
            )
        row = {
            "step": step + 1,
            "input_states": input_state_count,
            "output_states": len(beam),
            "best_speculative_gate_count": beam[0].gate_count,
            "predicted_actions": predicted_action_count,
            "eligible_actions_before_parent_cap": eligible_actions,
            "proposals_after_caps": len(proposals),
            "attempted_actions": attempted,
            "accepted_actions": len(beam),
            "invalid_structural_actions": invalid,
            "duplicate_speculative_successors": duplicates,
            "model_match_seconds": model_seconds,
            "exploration_model_match_seconds": exploration_model_seconds,
            "proposal_seconds": proposal_seconds,
            "lazy_update_and_hash_seconds": lazy_update_seconds,
            "paged_cache_advance_seconds": cache_seconds,
            "exploration_paged_cache_advance_seconds": exploration_cache_seconds,
            "exploration_predicted_actions": exploration_predicted_action_count,
            "exploration_eligible_actions_before_parent_cap": (
                exploration_eligible_actions
            ),
            "selected_exploration_proposals": selected_exploration_proposals,
            "exact_refresh_seconds": refresh_seconds,
            "exact_refresh_candidates": refresh_candidates,
            "exact_refresh_attempted": refresh_attempted,
            "exact_refresh_valid": refresh_valid,
            "exact_refresh_early_stopped": refresh_early_stopped,
            "exact_refresh_failures": dict(sorted(refresh_failures.items())),
            "checkpoint_audited": checkpoint_audited,
            "checkpoint_mismatches": checkpoint_mismatches,
            "allocated_cache_pages": arena.allocated_pages,
            "logical_cache_pages": logical_pages,
            "page_sharing_ratio": logical_pages / max(1, arena.allocated_pages),
            "total_seconds": elapsed,
        }
        if args.profile_stages:
            row["stage_seconds"] = stage_seconds
            if refresh_due:
                refresh_profile_percent = {
                    name: 100.0 * value / max(refresh_seconds, 1e-12)
                    for name, value in refresh_profile_seconds.items()
                }
                row["exact_refresh_profile"] = {
                    "scope": "exact_refresh_seconds",
                    "seconds": dict(
                        sorted(
                            refresh_profile_seconds.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ),
                    "percent": dict(
                        sorted(
                            refresh_profile_percent.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ),
                    "counts": dict(sorted(refresh_profile_counts.items())),
                    "accounted_seconds": sum(refresh_profile_seconds.values()),
                    "accounted_percent": sum(refresh_profile_percent.values()),
                }
        step_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.dump_beam_levels is not None:
            append_beam_level(args.dump_beam_levels, step + 1, beam)

    search_seconds = time.perf_counter() - total_started
    stage_totals: dict[str, float] = {}
    stage_percentages: dict[str, float] = {}
    refresh_profile_totals: dict[str, float] = {}
    refresh_profile_counts: dict[str, int] = {}
    if args.profile_stages:
        for row in step_rows:
            for name, value in row["stage_seconds"].items():
                stage_totals[name] = stage_totals.get(name, 0.0) + value
            refresh_profile = row.get("exact_refresh_profile")
            if refresh_profile is not None:
                for name, value in refresh_profile["seconds"].items():
                    refresh_profile_totals[name] = (
                        refresh_profile_totals.get(name, 0.0) + value
                    )
                for name, value in refresh_profile["counts"].items():
                    refresh_profile_counts[name] = (
                        refresh_profile_counts.get(name, 0) + int(value)
                    )
        profiled_steps_seconds = sum(row["total_seconds"] for row in step_rows)
        loop_boundary_seconds = max(0.0, search_seconds - profiled_steps_seconds)
        stage_totals["search_loop_boundary_seconds"] = loop_boundary_seconds
        stage_percentages = {
            name: 100.0 * value / max(search_seconds, 1e-12)
            for name, value in stage_totals.items()
        }
    audit_started = time.perf_counter()
    audited = min(args.audit_count, len(beam))
    valid = topology_matches = 0
    exact_hashes = set()
    failure_steps: dict[int, int] = defaultdict(int)
    best_valid_gate_count = None
    best_valid_graph = None
    for state in beam[:audited]:
        exact_graph, failure_step, topology_ok = replay_state(
            state, context, quartz.PyGraph, xfers, initial_qasm
        )
        if exact_graph is None:
            failure_steps[int(failure_step)] += 1
            continue
        valid += 1
        topology_matches += int(topology_ok)
        exact_hashes.add(int(exact_graph.hash()))
        gate_count = int(exact_graph.gate_count)
        if best_valid_gate_count is None or gate_count < best_valid_gate_count:
            best_valid_gate_count = gate_count
            best_valid_graph = exact_graph
    audit = {
        "audited_states": audited,
        "valid_trajectories": valid,
        "valid_trajectory_rate": valid / max(1, audited),
        "exact_topology_matches": topology_matches,
        "unique_exact_graph_hashes": len(exact_hashes),
        "failure_steps": dict(sorted(failure_steps.items())),
        "best_valid_gate_count": best_valid_gate_count,
        "audit_seconds": time.perf_counter() - audit_started,
    }
    total = {
        "mode": "paged_action_lazy",
        "qasm": str(args.qasm),
        "beam_size": args.beam_size,
        "requested_depth": args.depth,
        "completed_depth": len(step_rows),
        "initial_gate_count": initial_gate_count,
        "best_speculative_gate_count": min(state.gate_count for state in beam),
        "final_beam_size": len(beam),
        "dedup_mode": args.dedup_mode,
        "lazy_topology_backend": args.lazy_topology_backend,
        "page_size": args.page_size,
        "cache_gather_backend": args.cache_gather_backend,
        "readout_attention_backend": args.readout_attention_backend,
        "state_batch_backend": args.state_batch_backend,
        "proposal_backend": args.proposal_backend,
        "allocated_cache_pages": arena.allocated_pages,
        "cache_capacity_pages": arena.capacity,
        "exploration_checkpoint": (
            str(args.exploration_checkpoint)
            if args.exploration_checkpoint is not None
            else None
        ),
        "exploration_actions_per_parent": args.exploration_actions_per_parent,
        "exploration_until_depth": args.exploration_until_depth,
        "exploration_allocated_cache_pages": (
            exploration_arena.allocated_pages
            if exploration_arena is not None
            else 0
        ),
        "search_seconds_excluding_audit": search_seconds,
        "profile_stages": args.profile_stages,
        "target_recall": args.target_recall,
        "target_recall_by_group": threshold_config["target_recall_by_group"],
        "microbatch": args.microbatch,
        "refresh_interval": args.refresh_interval,
        "refresh_factor": args.refresh_factor,
        "checkpoint_audit_count": args.checkpoint_audit_count,
        "steps": step_rows,
        "quartz_replay_audit": audit,
    }
    if args.profile_stages:
        total["stage_profile"] = {
            "scope": "search_seconds_excluding_audit",
            "timing_method": "wall clock with CUDA synchronization at stage boundaries",
            "seconds": dict(
                sorted(stage_totals.items(), key=lambda item: item[1], reverse=True)
            ),
            "percent": dict(
                sorted(
                    stage_percentages.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),
            "accounted_seconds": sum(stage_totals.values()),
            "accounted_percent": sum(stage_percentages.values()),
        }
        if refresh_profile_totals:
            refresh_seconds_total = stage_totals.get("exact_quartz_refresh_seconds", 0.0)
            refresh_profile_percent = {
                name: 100.0 * value / max(refresh_seconds_total, 1e-12)
                for name, value in refresh_profile_totals.items()
            }
            total["exact_refresh_profile"] = {
                "scope": "exact_quartz_refresh_seconds",
                "seconds": dict(
                    sorted(
                        refresh_profile_totals.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ),
                "percent": dict(
                    sorted(
                        refresh_profile_percent.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ),
                "counts": dict(sorted(refresh_profile_counts.items())),
                "accounted_seconds": sum(refresh_profile_totals.values()),
                "accounted_percent": sum(refresh_profile_percent.values()),
            }
    rendered = json.dumps(total, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    if args.dump_beam_histories is not None:
        history_payload = {
            "qasm": str(args.qasm),
            "initial_qasm": initial_qasm,
            "initial_snapshot": initial_snapshot,
            "completed_depth": len(step_rows),
            "states": [
                serialized_history_state(state) for state in beam
            ],
        }
        args.dump_beam_histories.parent.mkdir(parents=True, exist_ok=True)
        args.dump_beam_histories.write_text(
            json.dumps(history_payload, indent=2, sort_keys=True) + "\n"
        )
    if args.best_qasm is not None and best_valid_graph is not None:
        args.best_qasm.parent.mkdir(parents=True, exist_ok=True)
        best_valid_graph.to_qasm(filename=str(args.best_qasm))


if __name__ == "__main__":
    main()
