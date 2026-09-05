from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch


@dataclass(frozen=True)
class CandidateTensors:
    batch_ids: torch.Tensor
    sources: torch.Tensor
    anchors: torch.Tensor
    bindings: torch.Tensor
    probabilities: torch.Tensor

    @classmethod
    def cat(cls, chunks: list["CandidateTensors"]) -> "CandidateTensors":
        if not chunks:
            raise ValueError("cannot concatenate an empty candidate list")
        return cls(
            **{
                name: torch.cat([getattr(chunk, name) for chunk in chunks])
                for name in (
                    "batch_ids",
                    "sources",
                    "anchors",
                    "bindings",
                    "probabilities",
                )
            }
        )


def load_threshold_config(
    path: Path,
    target_recall: float,
    *,
    near_target_recall: float | None = None,
    far_target_recall: float | None = None,
) -> dict:
    payload = json.loads(path.read_text())
    target_by_group = {
        "near": target_recall if near_target_recall is None else near_target_recall,
        "far": target_recall if far_target_recall is None else far_target_recall,
    }
    groups = {}
    for name in ("near", "far"):
        group = payload["groups"][name]
        recall_key = f"{target_by_group[name]:.4f}"
        threshold = group["thresholds"][recall_key]
        groups[name] = {
            "scale": float(group["probability_scale"]),
            "bias": float(group["probability_bias"]),
            "raw_threshold": float(threshold["raw_logit_threshold"]),
            "probability_threshold": float(
                threshold["calibrated_probability_threshold"]
            ),
        }
    return {
        "target_recall": target_recall,
        "target_recall_by_group": target_by_group,
        "groups": groups,
    }


@torch.no_grad()
def threshold_candidates(
    model,
    batch: dict,
    logits: torch.Tensor,
    eligible: torch.Tensor,
    config: dict,
    *,
    max_candidates_per_state: int = 2048,
    timing: dict[str, float] | None = None,
) -> list[list[tuple[int, int, tuple[int, ...], float]]]:
    """Return structurally valid (source, anchor, binding, probability) rows."""
    device = logits.device

    def finish_timing(name: str, started: float) -> None:
        if timing is None:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - started

    stage_started = time.perf_counter()
    distance = batch["current_rewrite_distance"]
    near_anchor = distance.le(2)
    near = config["groups"]["near"]
    far = config["groups"]["far"]
    scale = torch.where(
        near_anchor,
        torch.tensor(near["scale"], device=device),
        torch.tensor(far["scale"], device=device),
    )
    bias = torch.where(
        near_anchor,
        torch.tensor(near["bias"], device=device),
        torch.tensor(far["bias"], device=device),
    )
    threshold = torch.where(
        near_anchor,
        torch.tensor(near["raw_threshold"], device=device),
        torch.tensor(far["raw_threshold"], device=device),
    )
    calibrated_logits = logits.float() * scale.unsqueeze(-1) + bias.unsqueeze(-1)
    selected = eligible & logits.ge(threshold.unsqueeze(-1))
    finish_timing("threshold_and_calibration_seconds", stage_started)

    stage_started = time.perf_counter()
    candidate_batch = []
    candidate_source = []
    candidate_anchor = []
    candidate_probability = []
    offsets = [0]
    for batch_index in range(logits.shape[0]):
        flat_positions = selected[batch_index].flatten().nonzero(
            as_tuple=False
        ).squeeze(1)
        if flat_positions.numel() > max_candidates_per_state:
            scores = calibrated_logits[batch_index].flatten()[flat_positions]
            order = scores.topk(max_candidates_per_state).indices
            flat_positions = flat_positions[order]
        else:
            scores = calibrated_logits[batch_index].flatten()[flat_positions]
            order = scores.argsort(descending=True)
            flat_positions = flat_positions[order]
        count = flat_positions.numel()
        anchors = torch.div(
            flat_positions, model.num_sources, rounding_mode="floor"
        )
        sources = flat_positions.remainder(model.num_sources)
        probabilities = calibrated_logits[batch_index, anchors, sources].sigmoid()
        candidate_batch.append(
            torch.full((count,), batch_index, dtype=torch.long, device=device)
        )
        candidate_anchor.append(anchors)
        candidate_source.append(sources)
        candidate_probability.append(probabilities)
        offsets.append(offsets[-1] + count)
    finish_timing("candidate_selection_and_ranking_seconds", stage_started)

    if not offsets[-1]:
        if timing is not None:
            for name in (
                "candidate_gpu_pack_seconds",
                "structural_binding_decode_seconds",
                "candidate_device_to_host_seconds",
                "candidate_python_pack_seconds",
            ):
                timing.setdefault(name, 0.0)
        return [[] for _ in range(logits.shape[0])]

    stage_started = time.perf_counter()
    all_batch = torch.cat(candidate_batch)
    all_sources = torch.cat(candidate_source)
    all_anchors = torch.cat(candidate_anchor)
    all_probabilities = torch.cat(candidate_probability)
    finish_timing("candidate_gpu_pack_seconds", stage_started)

    stage_started = time.perf_counter()
    bindings, valid = model.structural_decode(
        batch,
        batch["current_types"],
        batch["current_types"].ge(0),
        all_batch,
        all_sources,
        all_anchors,
    )
    finish_timing("structural_binding_decode_seconds", stage_started)

    stage_started = time.perf_counter()
    sources_cpu = all_sources.tolist()
    anchors_cpu = all_anchors.tolist()
    probabilities_cpu = all_probabilities.tolist()
    bindings_cpu = bindings.tolist()
    valid_cpu = valid.tolist()
    source_lengths = model.source_lengths.tolist()
    finish_timing("candidate_device_to_host_seconds", stage_started)

    stage_started = time.perf_counter()
    result = []
    for batch_index in range(logits.shape[0]):
        rows = []
        begin, end = offsets[batch_index : batch_index + 2]
        for row_index in range(begin, end):
            if not valid_cpu[row_index]:
                continue
            source = sources_cpu[row_index]
            length = source_lengths[source]
            rows.append(
                (
                    source,
                    anchors_cpu[row_index],
                    tuple(bindings_cpu[row_index][:length]),
                    probabilities_cpu[row_index],
                )
            )
        result.append(rows)
    finish_timing("candidate_python_pack_seconds", stage_started)
    return result


@torch.no_grad()
def threshold_candidate_tensors(
    model,
    batch: dict,
    logits: torch.Tensor,
    eligible: torch.Tensor,
    config: dict,
    *,
    max_candidates_per_state: int = 2048,
    batch_offset: int = 0,
    timing: dict[str, float] | None = None,
) -> CandidateTensors:
    """GPU-resident vectorized equivalent of ``threshold_candidates``."""
    device = logits.device

    def finish_timing(name: str, started: float) -> None:
        if timing is None:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - started

    stage_started = time.perf_counter()
    distance = batch["current_rewrite_distance"]
    near_anchor = distance.le(2)
    near = config["groups"]["near"]
    far = config["groups"]["far"]
    scale = torch.where(
        near_anchor,
        torch.tensor(near["scale"], device=device),
        torch.tensor(far["scale"], device=device),
    )
    bias = torch.where(
        near_anchor,
        torch.tensor(near["bias"], device=device),
        torch.tensor(far["bias"], device=device),
    )
    threshold = torch.where(
        near_anchor,
        torch.tensor(near["raw_threshold"], device=device),
        torch.tensor(far["raw_threshold"], device=device),
    )
    calibrated_logits = logits.float() * scale.unsqueeze(-1) + bias.unsqueeze(-1)
    selected = eligible & logits.ge(threshold.unsqueeze(-1))
    finish_timing("threshold_and_calibration_seconds", stage_started)

    stage_started = time.perf_counter()
    flat_scores = calibrated_logits.flatten(1).masked_fill(
        ~selected.flatten(1), -torch.inf
    )
    count = min(max_candidates_per_state, flat_scores.shape[1])
    scores, flat_positions = flat_scores.topk(count, dim=1)
    present = scores.isfinite()
    finish_timing("candidate_selection_and_ranking_seconds", stage_started)

    stage_started = time.perf_counter()
    batch_ids = (
        torch.arange(logits.shape[0], device=device)
        .unsqueeze(1)
        .expand_as(flat_positions)[present]
    )
    scores = scores[present]
    flat_positions = flat_positions[present]
    anchors = torch.div(
        flat_positions, model.num_sources, rounding_mode="floor"
    )
    sources = flat_positions.remainder(model.num_sources)
    probabilities = scores.sigmoid()
    finish_timing("candidate_gpu_pack_seconds", stage_started)

    stage_started = time.perf_counter()
    bindings, structurally_valid = model.structural_decode(
        batch,
        batch["current_types"],
        batch["current_types"].ge(0),
        batch_ids,
        sources,
        anchors,
    )
    finish_timing("structural_binding_decode_seconds", stage_started)
    if timing is not None:
        timing.setdefault("candidate_device_to_host_seconds", 0.0)
        timing.setdefault("candidate_python_pack_seconds", 0.0)
    return CandidateTensors(
        batch_ids=batch_ids[structurally_valid] + batch_offset,
        sources=sources[structurally_valid],
        anchors=anchors[structurally_valid],
        bindings=bindings[structurally_valid],
        probabilities=probabilities[structurally_valid],
    )


@torch.no_grad()
def threshold_candidate_tensors_chunked(
    model,
    batch: dict,
    node_vectors: torch.Tensor,
    live: torch.Tensor,
    gate_types: torch.Tensor,
    source_vectors: torch.Tensor,
    config: dict,
    *,
    source_chunk_size: int,
    max_candidates_per_state: int = 2048,
    batch_offset: int = 0,
    timing: dict[str, float] | None = None,
) -> CandidateTensors:
    """Select the exact global top-k without materializing all source logits."""
    if source_chunk_size <= 0:
        raise ValueError("source_chunk_size must be positive")
    device = node_vectors.device

    def finish_timing(name: str, started: float) -> None:
        if timing is None:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing[name] = timing.get(name, 0.0) + time.perf_counter() - started

    stage_started = time.perf_counter()
    distance = batch["current_rewrite_distance"]
    near_anchor = distance.le(2)
    near = config["groups"]["near"]
    far = config["groups"]["far"]
    scale = torch.where(
        near_anchor,
        torch.tensor(near["scale"], device=device),
        torch.tensor(far["scale"], device=device),
    )
    bias = torch.where(
        near_anchor,
        torch.tensor(near["bias"], device=device),
        torch.tensor(far["bias"], device=device),
    )
    threshold = torch.where(
        near_anchor,
        torch.tensor(near["raw_threshold"], device=device),
        torch.tensor(far["raw_threshold"], device=device),
    )
    finish_timing("threshold_and_calibration_seconds", stage_started)

    best_scores = None
    best_anchors = None
    best_sources = None
    for source_begin in range(0, model.num_sources, source_chunk_size):
        source_end = min(model.num_sources, source_begin + source_chunk_size)
        stage_started = time.perf_counter()
        logits, eligible = model.match_logits_from_node_vectors(
            node_vectors,
            live,
            gate_types,
            source_vectors,
            source_begin=source_begin,
            source_end=source_end,
        )
        finish_timing("match_logits_seconds", stage_started)

        stage_started = time.perf_counter()
        calibrated_logits = logits.float() * scale.unsqueeze(-1) + bias.unsqueeze(-1)
        selected = eligible & logits.ge(threshold.unsqueeze(-1))
        flat_scores = calibrated_logits.flatten(1).masked_fill(
            ~selected.flatten(1), -torch.inf
        )
        count = min(max_candidates_per_state, flat_scores.shape[1])
        chunk_scores, flat_positions = flat_scores.topk(count, dim=1)
        chunk_sources = flat_positions.remainder(source_end - source_begin)
        chunk_anchors = torch.div(
            flat_positions, source_end - source_begin, rounding_mode="floor"
        )
        chunk_sources = chunk_sources + source_begin
        if best_scores is None:
            best_scores = chunk_scores
            best_anchors = chunk_anchors
            best_sources = chunk_sources
        else:
            merged_scores = torch.cat((best_scores, chunk_scores), dim=1)
            merged_anchors = torch.cat((best_anchors, chunk_anchors), dim=1)
            merged_sources = torch.cat((best_sources, chunk_sources), dim=1)
            count = min(max_candidates_per_state, merged_scores.shape[1])
            best_scores, order = merged_scores.topk(count, dim=1)
            best_anchors = merged_anchors.gather(1, order)
            best_sources = merged_sources.gather(1, order)
        finish_timing("candidate_selection_and_ranking_seconds", stage_started)

    if best_scores is None or best_anchors is None or best_sources is None:
        raise RuntimeError("model has no source patterns")

    stage_started = time.perf_counter()
    present = best_scores.isfinite()
    batch_ids = (
        torch.arange(node_vectors.shape[0], device=device)
        .unsqueeze(1)
        .expand_as(best_scores)[present]
    )
    scores = best_scores[present]
    anchors = best_anchors[present]
    sources = best_sources[present]
    probabilities = scores.sigmoid()
    finish_timing("candidate_gpu_pack_seconds", stage_started)

    stage_started = time.perf_counter()
    bindings, structurally_valid = model.structural_decode(
        batch,
        batch["current_types"],
        batch["current_types"].ge(0),
        batch_ids,
        sources,
        anchors,
    )
    finish_timing("structural_binding_decode_seconds", stage_started)
    if timing is not None:
        timing.setdefault("candidate_device_to_host_seconds", 0.0)
        timing.setdefault("candidate_python_pack_seconds", 0.0)
    return CandidateTensors(
        batch_ids=batch_ids[structurally_valid] + batch_offset,
        sources=sources[structurally_valid],
        anchors=anchors[structurally_valid],
        bindings=bindings[structurally_valid],
        probabilities=probabilities[structurally_valid],
    )
