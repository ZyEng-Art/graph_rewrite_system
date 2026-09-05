from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    EpochRandomSampler,
    collate_current_graphs,
    collate_prefixes,
    load_datasets,
    rebase_prefix_sample,
)
from model_factory import build_model


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def matcher_refresh_consistency_loss(
    logits: torch.Tensor,
    refreshed_logits: torch.Tensor,
    eligible: torch.Tensor,
    refreshed_eligible: torch.Tensor,
    positives: list[list],
    *,
    max_hard_pairs: int,
) -> torch.Tensor:
    """Align matcher scores for two decompositions of the same graph state.

    Every exact-positive anchor/source pair is included.  The highest-scoring
    eligible pairs from either view are added so the loss also constrains the
    part of the negative distribution that controls a finite Top-N boundary.
    """

    if logits.shape != refreshed_logits.shape:
        raise ValueError("refresh views must have identical matcher shapes")
    if max_hard_pairs < 0:
        raise ValueError("max_hard_pairs must be nonnegative")
    losses = []
    for batch_index, rows in enumerate(positives):
        common = eligible[batch_index] & refreshed_eligible[batch_index]
        if not bool(common.any()):
            continue
        state_losses = []
        positive_positions = torch.empty(
            0, dtype=torch.long, device=logits.device
        )
        if rows:
            anchors = torch.tensor(
                [int(binding[0]) for _, binding in rows],
                dtype=torch.long,
                device=logits.device,
            )
            sources = torch.tensor(
                [int(source) for source, _ in rows],
                dtype=torch.long,
                device=logits.device,
            )
            positive_positions = torch.unique(
                anchors * logits.shape[-1] + sources
            )
            base_positive = (
                logits[batch_index]
                .flatten()[positive_positions]
                .float()
                .clamp(-20, 20)
            )
            refreshed_positive = (
                refreshed_logits[batch_index]
                .flatten()[positive_positions]
                .float()
                .clamp(-20, 20)
            )
            # Do not average a correctly high positive down toward a failed
            # refresh view.  The stronger view acts as a stop-gradient target,
            # so consistency can only raise the weaker positive score.
            positive_target = torch.maximum(
                base_positive, refreshed_positive
            ).detach()
            state_losses.append(
                0.5
                * (
                    F.smooth_l1_loss(
                        base_positive,
                        positive_target,
                        reduction="none",
                    )
                    + F.smooth_l1_loss(
                        refreshed_positive,
                        positive_target,
                        reduction="none",
                    )
                )
            )
        if max_hard_pairs:
            common_positions = common.flatten().nonzero(
                as_tuple=False
            ).squeeze(1)
            hard_count = min(max_hard_pairs, int(common_positions.numel()))
            with torch.no_grad():
                boundary_scores = torch.maximum(
                    logits[batch_index].float(),
                    refreshed_logits[batch_index].float(),
                ).flatten()[common_positions]
                hard = common_positions[boundary_scores.topk(hard_count).indices]
            if positive_positions.numel():
                hard = hard[~torch.isin(hard, positive_positions)]
            if hard.numel():
                base_hard = (
                    logits[batch_index].flatten()[hard].float().clamp(-20, 20)
                )
                refreshed_hard = (
                    refreshed_logits[batch_index]
                    .flatten()[hard]
                    .float()
                    .clamp(-20, 20)
                )
                state_losses.append(
                    F.smooth_l1_loss(
                        base_hard,
                        refreshed_hard,
                        reduction="none",
                    )
                )
        if not state_losses:
            continue
        losses.append(torch.cat(state_losses).mean())
    if not losses:
        return logits.sum() * 0
    return torch.stack(losses).mean()


def collate_refresh_views(
    samples: list[dict],
    rules,
    max_actions: int,
    min_actions: int | None = None,
) -> dict:
    """Collate aligned long- and short-prefix views of identical states."""

    base = collate_prefixes(samples, rules)
    if min_actions is None:
        retained_actions = [max_actions] * len(samples)
    else:
        if not 0 <= min_actions <= max_actions:
            raise ValueError("refresh action range must satisfy 0 <= min <= max")
        retained_actions = torch.randint(
            min_actions,
            max_actions + 1,
            (len(samples),),
        ).tolist()
    refresh = collate_prefixes(
        [
            rebase_prefix_sample(sample, retained)
            for sample, retained in zip(samples, retained_actions)
        ],
        rules,
    )
    slots = max(base["initial_types"].shape[1], refresh["initial_types"].shape[1])
    for batch in (base, refresh):
        extra = slots - batch["initial_types"].shape[1]
        if not extra:
            continue
        batch["initial_types"] = F.pad(
            batch["initial_types"], (0, extra), value=-1
        )
        batch["current_types"] = F.pad(
            batch["current_types"], (0, extra), value=-1
        )
        batch["current_rewrite_distance"] = F.pad(
            batch["current_rewrite_distance"], (0, extra), value=5
        )
        batch["current_touch_age"] = F.pad(
            batch["current_touch_age"], (0, extra), value=7
        )
    return {"base": base, "refresh": refresh}


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    rules,
    max_batches: int | None = None,
    candidate_multiplier: int = 1,
    include_neural_metrics: bool = False,
) -> dict:
    model.eval()
    totals = {
        "matches": 0,
        "class_hits": 0,
        "full_hits": 0,
        "neural_pointer_hits": 0,
        "binding_examples": 0,
        "binding_hits": 0,
        "local_matches": 0,
        "local_full_hits": 0,
        "deep_local_matches": 0,
        "deep_local_full_hits": 0,
        "near_matches": 0,
        "near_full_hits": 0,
        "non_near_matches": 0,
        "non_near_full_hits": 0,
        "states": 0,
        "emitted_matches": 0,
        "states_with_fewer_than_n": 0,
    }
    prefix_buckets = {
        "prefix_0_15": [0, 0],
        "prefix_16_31": [0, 0],
        "prefix_32_63": [0, 0],
        "prefix_64_plus": [0, 0],
    }
    inference_seconds = 0.0
    structural_seconds = 0.0
    source_lengths_cpu = model.source_lengths.tolist()
    for batch_index, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(cpu_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with autocast_context(device):
            states, live, gate_types = model.encode(batch)
            logits, eligible = model.match_logits(states, live, gate_types)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - started

        decode_started = time.perf_counter()
        candidate_batch = []
        candidate_source = []
        candidate_anchor = []
        candidate_offsets = [0]
        for sample_index, rows in enumerate(batch["positives"]):
            num_true = len(rows)
            flat = logits[sample_index].flatten()
            count = min(
                candidate_multiplier * num_true,
                int(eligible[sample_index].sum().item()),
            )
            indices = flat.topk(count).indices
            candidate_anchor.append(indices // model.num_sources)
            candidate_source.append(indices % model.num_sources)
            candidate_batch.append(
                torch.full((count,), sample_index, dtype=torch.long, device=device)
            )
            candidate_offsets.append(candidate_offsets[-1] + count)
        all_batch = torch.cat(candidate_batch)
        all_sources = torch.cat(candidate_source)
        all_anchors = torch.cat(candidate_anchor)
        structural_bindings, structural_valid = model.structural_decode(
            batch, gate_types, live, all_batch, all_sources, all_anchors
        )
        all_sources_cpu = all_sources.tolist()
        all_anchors_cpu = all_anchors.tolist()
        structural_bindings_cpu = structural_bindings.tolist()
        structural_valid_cpu = structural_valid.tolist()
        predicted_bindings = None
        if include_neural_metrics:
            predicted_bindings, _ = model.decode_bindings(
                states, live, gate_types, all_batch, all_sources, all_anchors
            )
            positive_batch = []
            positive_source = []
            positive_anchor = []
            positive_rows = []
            for sample_index, rows in enumerate(batch["positives"]):
                for source_id, binding in rows:
                    positive_batch.append(sample_index)
                    positive_source.append(source_id)
                    positive_anchor.append(binding[0])
                    positive_rows.append((source_id, binding))
            positive_predictions, _ = model.decode_bindings(
                states,
                live,
                gate_types,
                torch.tensor(positive_batch, device=device),
                torch.tensor(positive_source, device=device),
                torch.tensor(positive_anchor, device=device),
            )
            for row_index, (source_id, binding) in enumerate(positive_rows):
                length = int(model.source_lengths[source_id])
                predicted = tuple(
                    map(int, positive_predictions[row_index, :length].tolist())
                )
                totals["binding_examples"] += 1
                totals["binding_hits"] += int(
                    predicted == tuple(map(int, binding))
                )

        for sample_index, rows in enumerate(batch["positives"]):
            true_full = {
                (int(source), tuple(map(int, binding))) for source, binding in rows
            }
            true_class = {
                (int(source), int(binding[0])) for source, binding in rows
            }
            begin, end = candidate_offsets[sample_index : sample_index + 2]
            predicted_full = set()
            neural_pointer_full = set()
            predicted_class = set()
            for candidate_rank, row_index in enumerate(range(begin, end)):
                source_id = all_sources_cpu[row_index]
                anchor = all_anchors_cpu[row_index]
                length = source_lengths_cpu[source_id]
                binding = tuple(structural_bindings_cpu[row_index][:length])
                if structural_valid_cpu[row_index] and len(predicted_full) < len(rows):
                    predicted_full.add((source_id, binding))
                if candidate_rank < len(rows):
                    if include_neural_metrics:
                        length = source_lengths_cpu[source_id]
                        neural_binding = tuple(
                            map(
                                int,
                                predicted_bindings[row_index, :length].tolist(),
                            )
                        )
                        neural_pointer_full.add((source_id, neural_binding))
                    predicted_class.add((source_id, anchor))
            class_hits = len(true_class & predicted_class)
            full_hits = len(true_full & predicted_full)
            neural_pointer_hits = len(true_full & neural_pointer_full)
            num_true = len(rows)
            totals["states"] += 1
            totals["matches"] += num_true
            totals["emitted_matches"] += len(predicted_full)
            totals["states_with_fewer_than_n"] += int(
                len(predicted_full) < num_true
            )
            totals["class_hits"] += class_hits
            totals["full_hits"] += full_hits
            for (source, binding), near in zip(
                rows, batch["positive_near"][sample_index]
            ):
                row = (int(source), tuple(map(int, binding)))
                key = "near" if near else "non_near"
                totals[f"{key}_matches"] += 1
                totals[f"{key}_full_hits"] += int(row in predicted_full)
            totals["neural_pointer_hits"] += neural_pointer_hits
            streak = int(batch["local_streak"][sample_index])
            if streak >= 2:
                totals["local_matches"] += num_true
                totals["local_full_hits"] += full_hits
            if streak >= 4:
                totals["deep_local_matches"] += num_true
                totals["deep_local_full_hits"] += full_hits
            prefix_length = int(batch["prefix_length"][sample_index])
            if prefix_length <= 15:
                bucket = "prefix_0_15"
            elif prefix_length <= 31:
                bucket = "prefix_16_31"
            elif prefix_length <= 63:
                bucket = "prefix_32_63"
            else:
                bucket = "prefix_64_plus"
            prefix_buckets[bucket][0] += full_hits
            prefix_buckets[bucket][1] += num_true
        structural_seconds += time.perf_counter() - decode_started

    def ratio(numerator: str, denominator: str) -> float:
        return totals[numerator] / max(1, totals[denominator])

    metrics = {
        "states": totals["states"],
        "matches": totals["matches"],
        "emitted_matches": totals["emitted_matches"],
        "emitted_to_true_ratio": totals["emitted_matches"]
        / max(1, totals["matches"]),
        "states_with_fewer_than_n": totals["states_with_fewer_than_n"],
        "class_topn": ratio("class_hits", "matches"),
        "full_binding_topn": ratio("full_hits", "matches"),
        "neural_pointer_topn": (
            ratio("neural_pointer_hits", "matches")
            if include_neural_metrics
            else None
        ),
        "oracle_pair_binding_accuracy": (
            ratio("binding_hits", "binding_examples")
            if include_neural_metrics
            else None
        ),
        "streak_ge_2_full_binding_topn": ratio(
            "local_full_hits", "local_matches"
        ),
        "streak_ge_4_full_binding_topn": ratio(
            "deep_local_full_hits", "deep_local_matches"
        ),
        "near_full_binding_topn": ratio("near_full_hits", "near_matches"),
        "near_matches": totals["near_matches"],
        "non_near_full_binding_topn": ratio(
            "non_near_full_hits", "non_near_matches"
        ),
        "model_ms_per_state": 1000.0
        * inference_seconds
        / max(1, totals["states"]),
        "structural_decode_ms_per_state": 1000.0
        * structural_seconds
        / max(1, totals["states"]),
        "model_plus_decode_ms_per_state": 1000.0
        * (inference_seconds + structural_seconds)
        / max(1, totals["states"]),
        "candidate_multiplier": candidate_multiplier,
    }
    for bucket, (hits, matches) in prefix_buckets.items():
        metrics[f"{bucket}_full_binding_topn"] = (
            hits / matches if matches else None
        )
        metrics[f"{bucket}_matches"] = matches
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="background workers for replaying and collating training prefixes",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="training batches prefetched by each background worker",
    )
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--retrieval-width", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--current-graph-layers", type=int, default=5)
    parser.add_argument(
        "--architecture", choices=("legacy", "paged_action"), default="legacy"
    )
    parser.add_argument("--action-layers", type=int, default=4)
    parser.add_argument("--action-heads", type=int, default=6)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--ordered-binding-roles", action="store_true")
    parser.add_argument("--readout-graph-layers", type=int, default=0)
    parser.add_argument(
        "--readout-graph-input",
        choices=("cached", "gate", "cached_gate"),
        default="cached",
    )
    parser.add_argument("--readout-locality-features", action="store_true")
    parser.add_argument("--identity-readout-prefix", type=int, default=0)
    parser.add_argument(
        "--readout-attention-backend",
        choices=("eager", "sdpa", "sdpa_live", "paged"),
        default="sdpa",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--binding-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--include-train-terminal", action="store_true")
    parser.add_argument("--train-terminal-repeat", type=int, default=1)
    parser.add_argument("--structural-hard-negatives", action="store_true")
    parser.add_argument("--state-only", action="store_true")
    parser.add_argument("--locality-features", action="store_true")
    parser.add_argument("--locality-positive-weight", type=float, default=0.0)
    parser.add_argument("--locality-negative-weight", type=float, default=0.0)
    parser.add_argument(
        "--action-positive-weight",
        type=float,
        default=0.0,
        help="extra classification weight for the trajectory's chosen exact action",
    )
    parser.add_argument("--topn-boundary-weight", type=float, default=0.0)
    parser.add_argument("--topn-boundary-margin", type=float, default=0.0)
    parser.add_argument(
        "--refresh-augmentation-actions",
        type=int,
        default=0,
        help=(
            "also supervise each state after rebasing its prefix to at most this "
            "many actions; 0 disables refresh augmentation"
        ),
    )
    parser.add_argument(
        "--refresh-augmentation-min-actions",
        type=int,
        help=(
            "randomly retain between this many and --refresh-augmentation-actions "
            "for each state; omitted keeps the fixed-length behavior"
        ),
    )
    parser.add_argument(
        "--refresh-supervision-weight",
        type=float,
        default=1.0,
        help="relative supervised-loss weight of the rebased refresh view",
    )
    parser.add_argument(
        "--refresh-consistency-weight",
        type=float,
        default=0.0,
        help="weight for matching logits across original and rebased views",
    )
    parser.add_argument(
        "--refresh-consistency-pairs",
        type=int,
        default=2048,
        help="hard eligible anchor/source pairs per state in consistency loss",
    )
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--selection-metric",
        choices=("full_binding_topn", "near_full_binding_topn"),
        default="full_binding_topn",
    )
    args = parser.parse_args()

    for name in (
        "locality_positive_weight",
        "locality_negative_weight",
        "action_positive_weight",
        "topn_boundary_weight",
        "refresh_supervision_weight",
        "refresh_consistency_weight",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        parser.error("worker count must be nonnegative and prefetch must be positive")
    if args.locality_features and not args.state_only:
        parser.error("--locality-features currently requires --state-only")
    if args.architecture == "paged_action" and args.state_only:
        parser.error("paged_action consumes action prefixes and cannot be state-only")
    if args.architecture == "paged_action" and args.locality_features:
        parser.error("paged_action does not consume materialized locality features")
    if args.refresh_augmentation_actions < 0:
        parser.error("--refresh-augmentation-actions must be nonnegative")
    if args.refresh_augmentation_min_actions is not None and not (
        0
        <= args.refresh_augmentation_min_actions
        <= args.refresh_augmentation_actions
    ):
        parser.error(
            "--refresh-augmentation-min-actions must be between zero and "
            "--refresh-augmentation-actions"
        )
    if (
        args.refresh_augmentation_min_actions is not None
        and not args.refresh_augmentation_actions
    ):
        parser.error(
            "--refresh-augmentation-min-actions requires refresh augmentation"
        )
    if args.refresh_consistency_pairs < 0:
        parser.error("--refresh-consistency-pairs must be nonnegative")
    if args.refresh_consistency_weight and not args.refresh_augmentation_actions:
        parser.error("refresh consistency requires refresh augmentation")
    if args.refresh_augmentation_actions and args.state_only:
        parser.error("refresh augmentation requires action-prefix training")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    payload, rules, train_dataset, test_dataset = load_datasets(
        args.data,
        include_train_terminal=args.include_train_terminal,
        train_terminal_repeat=args.train_terminal_repeat,
    )
    collate = partial(
        collate_current_graphs if args.state_only else collate_prefixes,
        rules=rules,
    )
    if args.refresh_augmentation_actions:
        train_collate = partial(
            collate_refresh_views,
            rules=rules,
            max_actions=args.refresh_augmentation_actions,
            min_actions=args.refresh_augmentation_min_actions,
        )
    else:
        train_collate = collate
    sampler = EpochRandomSampler(train_dataset, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        **(
            {"prefetch_factor": args.prefetch_factor}
            if args.num_workers > 0
            else {}
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model(rules, len(payload["xfer_to_source"]), vars(args)).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        if args.architecture == "paged_action":
            target = model.state_dict()
            compatible = {
                key: value
                for key, value in initial["model"].items()
                if key in target and target[key].shape == value.shape
            }
            incompatible = model.load_state_dict(compatible, strict=False)
            print(
                f"initialized_from={args.init_checkpoint} "
                f"compatible_tensors={len(compatible)} "
                f"new_tensors={len(incompatible.missing_keys)}",
                flush=True,
            )
        else:
            incompatible = model.load_state_dict(initial["model"], strict=False)
            allowed_missing = (
                "rewrite_distance_embedding.",
                "touch_age_embedding.",
                "local_streak_embedding.",
                "locality_fusion.",
            )
            unexpected_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(allowed_missing)
            ]
            if incompatible.unexpected_keys or unexpected_missing:
                raise RuntimeError(
                    "incompatible initialization checkpoint: "
                    f"missing={unexpected_missing}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            print(
                f"initialized_from={args.init_checkpoint} "
                f"new_parameters={incompatible.missing_keys}",
                flush=True,
            )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} parameters={parameter_count:,} "
        f"train_states={len(train_dataset)} test_states={len(test_dataset)}",
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_class = 0.0
        epoch_binding = 0.0
        epoch_refresh = 0.0
        epoch_consistency = 0.0
        batches = 0
        started = time.perf_counter()
        for batch_index, cpu_batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            if args.refresh_augmentation_actions:
                batch = move_batch(cpu_batch["base"], device)
                refresh_batch = move_batch(cpu_batch["refresh"], device)
            else:
                batch = move_batch(cpu_batch, device)
                refresh_batch = None
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                states, live, gate_types = model.encode(batch)
                logits, eligible = model.match_logits(states, live, gate_types)
                class_loss = model.classification_loss(
                    logits,
                    eligible,
                    batch["positives"],
                    batch=batch,
                    live=live,
                    gate_types=gate_types,
                    structural_hard_negatives=args.structural_hard_negatives,
                    locality_positive_weight=args.locality_positive_weight,
                    locality_negative_weight=args.locality_negative_weight,
                    action_positive_weight=args.action_positive_weight,
                    topn_boundary_weight=args.topn_boundary_weight,
                    topn_boundary_margin=args.topn_boundary_margin,
                )
                if args.binding_weight > 0:
                    binding_loss = model.binding_loss(
                        states, live, gate_types, batch["positives"]
                    )
                else:
                    binding_loss = states.sum() * 0
                base_supervised_loss = class_loss + args.binding_weight * binding_loss
                refresh_supervised_loss = states.sum() * 0
                consistency_loss = states.sum() * 0
                if refresh_batch is not None:
                    (
                        refresh_states,
                        refresh_live,
                        refresh_gate_types,
                    ) = model.encode(refresh_batch)
                    refresh_logits, refresh_eligible = model.match_logits(
                        refresh_states, refresh_live, refresh_gate_types
                    )
                    refresh_class_loss = model.classification_loss(
                        refresh_logits,
                        refresh_eligible,
                        refresh_batch["positives"],
                        batch=refresh_batch,
                        live=refresh_live,
                        gate_types=refresh_gate_types,
                        structural_hard_negatives=args.structural_hard_negatives,
                        locality_positive_weight=args.locality_positive_weight,
                        locality_negative_weight=args.locality_negative_weight,
                        action_positive_weight=args.action_positive_weight,
                        topn_boundary_weight=args.topn_boundary_weight,
                        topn_boundary_margin=args.topn_boundary_margin,
                    )
                    if args.binding_weight > 0:
                        refresh_binding_loss = model.binding_loss(
                            refresh_states,
                            refresh_live,
                            refresh_gate_types,
                            refresh_batch["positives"],
                        )
                    else:
                        refresh_binding_loss = refresh_states.sum() * 0
                    refresh_supervised_loss = (
                        refresh_class_loss
                        + args.binding_weight * refresh_binding_loss
                    )
                    consistency_loss = matcher_refresh_consistency_loss(
                        logits,
                        refresh_logits,
                        eligible,
                        refresh_eligible,
                        batch["positives"],
                        max_hard_pairs=args.refresh_consistency_pairs,
                    )
                supervised_denominator = 1.0 + (
                    args.refresh_supervision_weight
                    if refresh_batch is not None
                    else 0.0
                )
                loss = (
                    base_supervised_loss
                    + args.refresh_supervision_weight * refresh_supervised_loss
                ) / supervised_denominator
                loss = loss + args.refresh_consistency_weight * consistency_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_class += float(class_loss.detach())
            epoch_binding += float(binding_loss.detach())
            epoch_refresh += float(refresh_supervised_loss.detach())
            epoch_consistency += float(consistency_loss.detach())
            batches += 1
        elapsed = time.perf_counter() - started
        print(
            f"epoch={epoch:03d} loss={epoch_loss / batches:.4f} "
            f"class={epoch_class / batches:.4f} binding={epoch_binding / batches:.4f} "
            f"refresh={epoch_refresh / batches:.4f} "
            f"consistency={epoch_consistency / batches:.4f} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(
                model,
                test_loader,
                device,
                rules,
                max_batches=args.max_eval_batches,
            )
            print("eval " + json.dumps(metrics, sort_keys=True), flush=True)
            selection_value = metrics[args.selection_metric]
            if selection_value > best_metric:
                best_metric = selection_value
                best_metrics = metrics
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "metrics": metrics,
                        "format": payload["format"],
                    },
                    args.output,
                )
                args.output.with_suffix(".metrics.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n"
                )
    print(
        "best " + json.dumps(best_metrics, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
