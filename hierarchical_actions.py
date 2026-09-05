from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from beam_search_benchmark import BeamState
from paged_cache import PagedKVCache, PrefixHandle
from ppo_core import build_state_features
from threshold_inference import CandidateTensors
from tensorized_batch import collate_paged_states
from train import autocast_context, move_batch


@dataclass(frozen=True)
class HierarchicalMatchResult:
    candidates: CandidateTensors
    candidate_node_positions: torch.Tensor
    encoded_states: torch.Tensor
    state_features: torch.Tensor
    prefix_states: torch.Tensor
    selected_nodes: torch.Tensor
    selected_node_features: torch.Tensor
    selected_node_mask: torch.Tensor
    elapsed_seconds: float
    timing: dict[str, float]
    fallback_state_count: int = 0
    fallback_state_mask: torch.Tensor | None = None


def _empty_candidates(model, device: torch.device) -> CandidateTensors:
    return CandidateTensors(
        batch_ids=torch.empty(0, dtype=torch.long, device=device),
        sources=torch.empty(0, dtype=torch.long, device=device),
        anchors=torch.empty(0, dtype=torch.long, device=device),
        bindings=torch.empty(
            (0, model.max_pattern), dtype=torch.long, device=device
        ),
        probabilities=torch.empty(0, device=device),
    )


def _select_candidates(
    candidates: CandidateTensors, selected: torch.Tensor
) -> CandidateTensors:
    return CandidateTensors(
        batch_ids=candidates.batch_ids[selected],
        sources=candidates.sources[selected],
        anchors=candidates.anchors[selected],
        bindings=candidates.bindings[selected],
        probabilities=candidates.probabilities[selected],
    )


def _finish_timing(
    timing: dict[str, float] | None,
    name: str,
    started: float,
    device: torch.device,
) -> None:
    if timing is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timing[name] = timing.get(name, 0.0) + time.perf_counter() - started


def current_prefix_states(
    encoded: torch.Tensor,
    live: torch.Tensor,
    arena: PagedKVCache,
    handles: list[PrefixHandle],
) -> torch.Tensor:
    """Use the last action token, or graph pooling for an empty prefix."""
    last_actions, present = arena.last_actions(handles)
    live_float = live.unsqueeze(-1)
    root_states = (encoded * live_float).sum(1)
    root_states = root_states / live_float.sum(1).clamp_min(1)
    last_actions = last_actions.to(root_states.dtype)
    return torch.where(present.unsqueeze(-1), last_actions, root_states)


@torch.no_grad()
def candidates_for_selected_nodes(
    model,
    batch: dict,
    node_vectors: torch.Tensor,
    selected_nodes: torch.Tensor,
    selected_node_mask: torch.Tensor,
    selected_gate_types: torch.Tensor,
    source_vectors: torch.Tensor,
    threshold_config: dict,
    *,
    pattern_k: int,
    source_grouping: str = "none",
    batch_offset: int = 0,
    timing: dict[str, float] | None = None,
) -> tuple[CandidateTensors, torch.Tensor]:
    """Match and structurally validate Top-N patterns at selected nodes only."""
    if pattern_k <= 0:
        raise ValueError("pattern_k must be positive")
    if source_grouping not in {"none", "first_gate"}:
        raise ValueError(f"unknown source grouping: {source_grouping}")
    if selected_nodes.shape != selected_node_mask.shape:
        raise ValueError("selected nodes and mask must have the same shape")
    if selected_gate_types.shape != selected_nodes.shape:
        raise ValueError("selected gate types must align with selected nodes")
    if node_vectors.shape[:2] != selected_nodes.shape:
        raise ValueError("node vectors must align with selected nodes")

    device = node_vectors.device
    stage_started = time.perf_counter()
    distances = batch["current_rewrite_distance"].gather(1, selected_nodes)
    near_anchor = distances.le(2)
    near = threshold_config["groups"]["near"]
    far = threshold_config["groups"]["far"]
    scale = torch.where(
        near_anchor,
        torch.as_tensor(near["scale"], device=device),
        torch.as_tensor(far["scale"], device=device),
    )
    bias = torch.where(
        near_anchor,
        torch.as_tensor(near["bias"], device=device),
        torch.as_tensor(far["bias"], device=device),
    )
    threshold = torch.where(
        near_anchor,
        torch.as_tensor(near["raw_threshold"], device=device),
        torch.as_tensor(far["raw_threshold"], device=device),
    )
    _finish_timing(timing, "selected_group_compaction_seconds", stage_started, device)

    if source_grouping == "none":
        stage_started = time.perf_counter()
        logits, eligible = model.match_logits_from_node_vectors(
            node_vectors,
            selected_node_mask,
            selected_gate_types,
            source_vectors,
        )
        _finish_timing(
            timing, "selected_match_logits_seconds", stage_started, device
        )

        stage_started = time.perf_counter()
        calibrated_logits = logits.float() * scale.unsqueeze(-1) + bias.unsqueeze(-1)
        retained = eligible & selected_node_mask.unsqueeze(-1)
        retained &= logits.ge(threshold.unsqueeze(-1))
        ranked_logits = calibrated_logits.masked_fill(~retained, -torch.inf)
        count = min(pattern_k, ranked_logits.shape[-1])
        scores, sources = ranked_logits.topk(count, dim=-1)
        present = scores.isfinite()
        batch_grid = (
            torch.arange(selected_nodes.shape[0], device=device)
            .view(-1, 1, 1)
            .expand_as(sources)
        )
        node_grid = (
            torch.arange(selected_nodes.shape[1], device=device)
            .view(1, -1, 1)
            .expand_as(sources)
        )
        anchor_grid = selected_nodes.unsqueeze(-1).expand_as(sources)
        batch_ids = batch_grid[present]
        node_positions = node_grid[present]
        sources = sources[present]
        anchors = anchor_grid[present]
        scores = scores[present]
        _finish_timing(
            timing, "selected_threshold_and_topn_seconds", stage_started, device
        )
    else:
        candidate_batches = []
        candidate_nodes = []
        candidate_sources = []
        candidate_scores = []
        candidate_anchors = []
        for gate_type, source_begin, source_end in model.source_first_gate_groups:
            stage_started = time.perf_counter()
            group_mask = selected_node_mask & selected_gate_types.eq(gate_type)
            group_batches, group_nodes = group_mask.nonzero(as_tuple=True)
            _finish_timing(
                timing, "selected_group_compaction_seconds", stage_started, device
            )
            if not group_batches.numel():
                continue
            source_ids = model.source_first_gate_order[source_begin:source_end]
            group_vectors = node_vectors[group_batches, group_nodes].unsqueeze(1)
            stage_started = time.perf_counter()
            logits = model.match_logits_for_sources(
                group_vectors, source_vectors, source_ids
            ).squeeze(1)
            _finish_timing(
                timing, "selected_match_logits_seconds", stage_started, device
            )

            stage_started = time.perf_counter()
            local_scale = scale[group_batches, group_nodes]
            local_bias = bias[group_batches, group_nodes]
            local_threshold = threshold[group_batches, group_nodes]
            calibrated_logits = logits.float() * local_scale.unsqueeze(1)
            calibrated_logits += local_bias.unsqueeze(1)
            ranked_logits = calibrated_logits.masked_fill(
                logits.lt(local_threshold.unsqueeze(1)), -torch.inf
            )
            count = min(pattern_k, ranked_logits.shape[1])
            scores, local_sources = ranked_logits.topk(count, dim=1)
            present = scores.isfinite()
            candidate_batches.append(
                group_batches.unsqueeze(1).expand_as(local_sources)[present]
            )
            candidate_nodes.append(
                group_nodes.unsqueeze(1).expand_as(local_sources)[present]
            )
            candidate_sources.append(source_ids[local_sources[present]])
            candidate_scores.append(scores[present])
            group_anchors = selected_nodes[group_batches, group_nodes]
            candidate_anchors.append(
                group_anchors.unsqueeze(1).expand_as(local_sources)[present]
            )
            _finish_timing(
                timing, "selected_threshold_and_topn_seconds", stage_started, device
            )
        if candidate_batches:
            batch_ids = torch.cat(candidate_batches)
            node_positions = torch.cat(candidate_nodes)
            sources = torch.cat(candidate_sources)
            scores = torch.cat(candidate_scores)
            anchors = torch.cat(candidate_anchors)
        else:
            batch_ids = torch.empty(0, dtype=torch.long, device=device)
            node_positions = torch.empty(0, dtype=torch.long, device=device)
            sources = torch.empty(0, dtype=torch.long, device=device)
            scores = torch.empty(0, device=device)
            anchors = torch.empty(0, dtype=torch.long, device=device)

    if not batch_ids.numel():
        return _empty_candidates(model, device), torch.empty(
            0, dtype=torch.long, device=device
        )

    stage_started = time.perf_counter()
    probabilities = scores.sigmoid()
    _finish_timing(timing, "selected_candidate_pack_seconds", stage_started, device)

    stage_started = time.perf_counter()
    bindings, structurally_valid = model.structural_decode(
        batch,
        batch["current_types"],
        batch["current_types"].ge(0),
        batch_ids,
        sources,
        anchors,
    )
    _finish_timing(timing, "selected_structural_decode_seconds", stage_started, device)
    return (
        CandidateTensors(
            batch_ids=batch_ids[structurally_valid] + batch_offset,
            sources=sources[structurally_valid],
            anchors=anchors[structurally_valid],
            bindings=bindings[structurally_valid],
            probabilities=probabilities[structurally_valid],
        ),
        node_positions[structurally_valid],
    )


@torch.no_grad()
def hierarchical_paged_matches(
    beam: list[BeamState],
    slot_states: torch.Tensor,
    live: torch.Tensor,
    gate_types: torch.Tensor,
    handles: list[PrefixHandle],
    arena: PagedKVCache,
    model,
    actor_critic,
    device: torch.device,
    threshold_config: dict,
    source_vectors: torch.Tensor,
    *,
    microbatch: int,
    node_k: int,
    pattern_k: int,
    fallback_threshold_config: dict | None = None,
    fallback_node_k: int | None = None,
    fallback_pattern_k: int | None = None,
    fallback_min_candidates: int = 1,
    force_fallback_mask: torch.Tensor | None = None,
    profile_stages: bool = False,
) -> HierarchicalMatchResult:
    """Generate exact rewrite candidates with a node-first policy."""
    if not beam:
        raise ValueError("beam must contain at least one state")
    if node_k <= 0:
        raise ValueError("node_k must be positive")
    if fallback_node_k is not None and fallback_node_k < node_k:
        raise ValueError("fallback node K cannot be smaller than primary node K")
    if fallback_pattern_k is not None and fallback_pattern_k < 1:
        raise ValueError("fallback pattern K must be positive")
    if fallback_min_candidates < 1:
        raise ValueError("fallback minimum candidates must be positive")
    fallback_enabled = (
        fallback_threshold_config is not None
        and fallback_node_k is not None
        and fallback_pattern_k is not None
    )
    if force_fallback_mask is not None:
        if not fallback_enabled:
            raise ValueError("forced fallback requires a fallback configuration")
        if force_fallback_mask.shape != (len(beam),):
            raise ValueError("forced fallback mask must align with the beam")
    if microbatch <= 0:
        raise ValueError("microbatch must be positive")
    if len(beam) != slot_states.shape[0] or len(handles) != len(beam):
        raise ValueError("beam, paged states, and prefix handles must align")

    timing: dict[str, float] | None = {} if profile_stages else None
    candidate_chunks: list[CandidateTensors] = []
    candidate_node_chunks: list[torch.Tensor] = []
    encoded_chunks = []
    state_feature_chunks = []
    prefix_chunks = []
    selected_node_chunks = []
    selected_feature_chunks = []
    selected_mask_chunks = []
    fallback_mask_chunks = []
    fallback_state_count = 0
    started = time.perf_counter()

    for begin in range(0, len(beam), microbatch):
        end = min(len(beam), begin + microbatch)
        selected_states = slot_states[begin:end]
        selected_live = live[begin:end]
        selected_types = gate_types[begin:end]

        stage_started = time.perf_counter()
        cpu_batch = collate_paged_states(beam[begin:end], selected_types)
        _finish_timing(timing, "batch_collate_seconds", stage_started, device)

        stage_started = time.perf_counter()
        batch = move_batch(cpu_batch, device)
        _finish_timing(timing, "batch_host_to_device_seconds", stage_started, device)
        if not torch.equal(selected_types, batch["current_types"]):
            raise RuntimeError("paged live/type cache differs from lazy topology")

        stage_started = time.perf_counter()
        if model.readout_attention_backend == "paged":
            block_table, lengths = arena.block_table(handles[begin:end])
            actions = selected_states.new_empty((end - begin, 0, model.width))
            action_mask = torch.empty(
                (end - begin, 0), device=device, dtype=torch.bool
            )
        else:
            actions, action_mask = arena.gather_actions(handles[begin:end])
            block_table = lengths = None
        _finish_timing(timing, "readout_cache_metadata_seconds", stage_started, device)

        stage_started = time.perf_counter()
        with autocast_context(device):
            encoded, _, _ = model.readout_incremental(
                selected_states,
                selected_live,
                selected_types,
                actions,
                action_mask,
                batch,
                readout_key_cache=(
                    arena.readout_keys
                    if model.readout_attention_backend == "paged"
                    else None
                ),
                readout_value_cache=(
                    arena.readout_values
                    if model.readout_attention_backend == "paged"
                    else None
                ),
                block_table=block_table,
                lengths=lengths,
            )
        _finish_timing(timing, "incremental_graph_readout_seconds", stage_started, device)

        stage_started = time.perf_counter()
        state_features = build_state_features(
            encoded, selected_live, selected_live.sum(1)
        )
        prefix_states = current_prefix_states(
            encoded, selected_live, arena, handles[begin:end]
        )
        with autocast_context(device):
            node_logits = actor_critic.node_policy_logits(
                encoded,
                selected_live,
                prefix_states,
                state_features,
            )
        _finish_timing(timing, "node_policy_seconds", stage_started, device)

        stage_started = time.perf_counter()
        retained_nodes = min(
            fallback_node_k if fallback_enabled else node_k,
            node_logits.shape[1],
        )
        node_scores, selected_nodes = node_logits.topk(retained_nodes, dim=1)
        selected_node_mask = node_scores.isfinite()
        feature_indices = selected_nodes.unsqueeze(-1).expand(
            -1, -1, encoded.shape[-1]
        )
        selected_node_features = encoded.gather(1, feature_indices)
        selected_gate_types = selected_types.gather(1, selected_nodes)
        with autocast_context(device):
            node_vectors = model.match_node_vectors(selected_node_features)
        _finish_timing(timing, "node_topk_and_projection_seconds", stage_started, device)

        primary_node_count = min(node_k, retained_nodes)
        forced_fallback = (
            force_fallback_mask[begin:end].to(device=device, dtype=torch.bool)
            if force_fallback_mask is not None
            else torch.zeros(end - begin, dtype=torch.bool, device=device)
        )
        primary_node_mask = selected_node_mask[:, :primary_node_count]
        if bool(forced_fallback.any()):
            primary_node_mask = primary_node_mask & ~forced_fallback.unsqueeze(1)
        candidates, candidate_node_positions = candidates_for_selected_nodes(
            model,
            batch,
            node_vectors[:, :primary_node_count],
            selected_nodes[:, :primary_node_count],
            primary_node_mask,
            selected_gate_types[:, :primary_node_count],
            source_vectors,
            threshold_config,
            pattern_k=pattern_k,
            source_grouping="first_gate",
            batch_offset=begin,
            timing=timing,
        )
        policy_node_mask = selected_node_mask
        if fallback_enabled:
            candidate_counts = torch.bincount(
                candidates.batch_ids - begin, minlength=end - begin
            )
            fallback_states = forced_fallback | candidate_counts.lt(
                fallback_min_candidates
            )
            fallback_state_count += int(fallback_states.sum().item())
            if candidates.batch_ids.numel():
                primary_kept = ~fallback_states[candidates.batch_ids - begin]
                candidates = _select_candidates(candidates, primary_kept)
                candidate_node_positions = candidate_node_positions[primary_kept]
            policy_node_mask = selected_node_mask.clone()
            policy_node_mask[~fallback_states, primary_node_count:] = False
            if bool(fallback_states.any()):
                fallback_mask = selected_node_mask & fallback_states.unsqueeze(1)
                (
                    fallback_candidates,
                    fallback_node_positions,
                ) = candidates_for_selected_nodes(
                    model,
                    batch,
                    node_vectors,
                    selected_nodes,
                    fallback_mask,
                    selected_gate_types,
                    source_vectors,
                    fallback_threshold_config,
                    pattern_k=fallback_pattern_k,
                    source_grouping="first_gate",
                    batch_offset=begin,
                    timing=timing,
                )
                candidates = CandidateTensors.cat(
                    [candidates, fallback_candidates]
                )
                candidate_node_positions = torch.cat(
                    [candidate_node_positions, fallback_node_positions]
                )
        else:
            fallback_states = forced_fallback
        fallback_mask_chunks.append(fallback_states)
        candidate_chunks.append(candidates)
        candidate_node_chunks.append(candidate_node_positions)
        encoded_chunks.append(encoded)
        state_feature_chunks.append(state_features)
        prefix_chunks.append(prefix_states)
        selected_node_chunks.append(selected_nodes)
        selected_feature_chunks.append(selected_node_features)
        selected_mask_chunks.append(policy_node_mask)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_timing = timing or {}
    if profile_stages:
        final_timing["hierarchical_match_unattributed_seconds"] = max(
            0.0, elapsed - sum(final_timing.values())
        )
    return HierarchicalMatchResult(
        candidates=CandidateTensors.cat(candidate_chunks),
        candidate_node_positions=torch.cat(candidate_node_chunks),
        encoded_states=torch.cat(encoded_chunks),
        state_features=torch.cat(state_feature_chunks),
        prefix_states=torch.cat(prefix_chunks),
        selected_nodes=torch.cat(selected_node_chunks),
        selected_node_features=torch.cat(selected_feature_chunks),
        selected_node_mask=torch.cat(selected_mask_chunks),
        elapsed_seconds=elapsed,
        timing=final_timing,
        fallback_state_count=fallback_state_count,
        fallback_state_mask=torch.cat(fallback_mask_chunks),
    )
