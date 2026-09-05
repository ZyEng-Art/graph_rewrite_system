from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import collate_current_graphs, collate_prefixes, load_datasets
from model_factory import build_model


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def stable_descending_ranks(scores: torch.Tensor) -> torch.Tensor:
    """Return one-indexed column ranks, preferring lower ids on exact ties."""
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    values = torch.arange(1, scores.shape[1] + 1, device=scores.device)
    ranks.scatter_(1, order, values.unsqueeze(0).expand_as(order))
    return ranks


def selected_source_ranks(
    logits: torch.Tensor,
    eligible: torch.Tensor,
    batch_ids: torch.Tensor,
    anchors: torch.Tensor,
    sources: torch.Tensor,
) -> torch.Tensor:
    """Rank selected sources at their anchors with deterministic tie handling."""
    if not batch_ids.numel():
        return torch.empty(0, dtype=torch.long, device=logits.device)
    rows = logits[batch_ids, anchors]
    row_eligible = eligible[batch_ids, anchors]
    selected = rows.gather(1, sources.unsqueeze(1))
    source_ids = torch.arange(logits.shape[-1], device=logits.device)
    ahead = rows.gt(selected)
    ties_ahead = rows.eq(selected) & source_ids.unsqueeze(0).lt(sources.unsqueeze(1))
    return 1 + ((ahead | ties_ahead) & row_eligible).sum(1)


def source_best_gate_reductions(rules) -> torch.Tensor:
    reductions = torch.full((len(rules.source_patterns),), -10_000, dtype=torch.long)
    for xfer_id, source_id in enumerate(rules.xfer_to_source):
        reduction = len(rules.source_gate_types[source_id]) - len(
            rules.destination_gate_types[xfer_id]
        )
        reductions[source_id] = max(int(reductions[source_id]), reduction)
    if bool(reductions.eq(-10_000).any()):
        raise ValueError("a source pattern has no associated rewrite")
    return reductions


def _new_scorer_counters(node_ks: tuple[int, ...], pattern_ks: tuple[int, ...]):
    return {
        "positive_node_hits": {str(k): 0 for k in node_ks},
        "states_with_positive_node": {str(k): 0 for k in node_ks},
        "target_action_node_hits": {str(k): 0 for k in node_ks},
        "best_gate_reduction_node_hits": {str(k): 0 for k in node_ks},
        "compatible_pairs_retained": {str(k): 0 for k in node_ks},
        "exact_match_hits": {
            str(k): {str(m): 0 for m in pattern_ks} for k in node_ks
        },
        "target_action_hits": {
            str(k): {str(m): 0 for m in pattern_ks} for k in node_ks
        },
        "structurally_valid_candidates": {
            str(k): {str(m): 0 for m in pattern_ks} for k in node_ks
        },
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


@torch.no_grad()
def audit(
    model,
    loader,
    device: torch.device,
    rules,
    node_ks: tuple[int, ...],
    pattern_ks: tuple[int, ...],
) -> dict:
    model.eval()
    scorer_names = ("max", "logsumexp")
    counters = {
        name: _new_scorer_counters(node_ks, pattern_ks) for name in scorer_names
    }
    totals = {
        "states": 0,
        "states_with_positives": 0,
        "exact_matches": 0,
        "target_actions": 0,
        "states_with_best_gate_reduction": 0,
        "compatible_node_source_pairs": 0,
    }
    model_seconds = 0.0
    structural_seconds = 0.0
    reductions_cpu = source_best_gate_reductions(rules)
    max_node_k = max(node_ks)
    max_pattern_k = max(pattern_ks)

    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with autocast_context(device):
            states, live, gate_types = model.encode(batch)
            logits, eligible = model.match_logits(states, live, gate_types)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_seconds += time.perf_counter() - started
        logits = logits.float()
        batch_size, num_slots, num_sources = logits.shape
        totals["states"] += batch_size
        totals["compatible_node_source_pairs"] += int(eligible.sum().item())

        positive_batch = []
        positive_sources = []
        positive_anchors = []
        positive_offsets = [0]
        best_anchor_sets: list[set[int]] = []
        for batch_index, rows in enumerate(batch["positives"]):
            totals["exact_matches"] += len(rows)
            totals["states_with_positives"] += bool(rows)
            for source_id, binding in rows:
                positive_batch.append(batch_index)
                positive_sources.append(int(source_id))
                positive_anchors.append(int(binding[0]))
            positive_offsets.append(len(positive_batch))
            if rows:
                best_reduction = max(int(reductions_cpu[source]) for source, _ in rows)
                best_anchor_sets.append(
                    {
                        int(binding[0])
                        for source, binding in rows
                        if int(reductions_cpu[source]) == best_reduction
                    }
                )
                totals["states_with_best_gate_reduction"] += 1
            else:
                best_anchor_sets.append(set())

        positive_batch_tensor = torch.tensor(
            positive_batch, dtype=torch.long, device=device
        )
        positive_sources_tensor = torch.tensor(
            positive_sources, dtype=torch.long, device=device
        )
        positive_anchors_tensor = torch.tensor(
            positive_anchors, dtype=torch.long, device=device
        )
        positive_pattern_ranks = selected_source_ranks(
            logits,
            eligible,
            positive_batch_tensor,
            positive_anchors_tensor,
            positive_sources_tensor,
        )

        target_batch = []
        target_anchor = []
        target_sources = []
        for batch_index, action in enumerate(batch["target_actions"]):
            if action is None:
                continue
            target_batch.append(batch_index)
            target_anchor.append(
                int(action.get("anchor_slot", action["binding_slots"][0]))
            )
            target_sources.append(int(action["source_id"]))
        totals["target_actions"] += len(target_batch)
        target_batch_tensor = torch.tensor(target_batch, dtype=torch.long, device=device)
        target_anchor_tensor = torch.tensor(
            target_anchor, dtype=torch.long, device=device
        )
        target_sources_tensor = torch.tensor(
            target_sources, dtype=torch.long, device=device
        )
        target_pattern_ranks = selected_source_ranks(
            logits,
            eligible,
            target_batch_tensor,
            target_anchor_tensor,
            target_sources_tensor,
        )

        masked_logits = logits.masked_fill(~eligible, -torch.inf)
        node_scores = {
            "max": masked_logits.max(-1).values,
            "logsumexp": torch.logsumexp(masked_logits, dim=-1),
        }
        compatible_per_node = eligible.sum(-1)

        for scorer_name, scores in node_scores.items():
            node_ranks = stable_descending_ranks(scores)
            node_order = torch.argsort(scores, dim=1, descending=True, stable=True)
            positive_node_ranks = (
                node_ranks[positive_batch_tensor, positive_anchors_tensor]
                if positive_batch_tensor.numel()
                else torch.empty(0, dtype=torch.long, device=device)
            )
            target_node_ranks = (
                node_ranks[target_batch_tensor, target_anchor_tensor]
                if target_batch_tensor.numel()
                else torch.empty(0, dtype=torch.long, device=device)
            )

            available_node_k = min(max_node_k, num_slots)
            available_pattern_k = min(max_pattern_k, num_sources)
            top_nodes = node_order[:, :available_node_k]
            selected_logits = logits.gather(
                1, top_nodes.unsqueeze(-1).expand(-1, -1, num_sources)
            )
            selected_eligible = eligible.gather(
                1, top_nodes.unsqueeze(-1).expand(-1, -1, num_sources)
            )
            selected_logits = selected_logits.masked_fill(
                ~selected_eligible, -torch.inf
            )
            top_pattern_scores, top_sources = selected_logits.topk(
                available_pattern_k, dim=-1
            )
            present = top_pattern_scores.isfinite()
            structural_batch = (
                torch.arange(batch_size, device=device)
                .view(-1, 1, 1)
                .expand_as(top_sources)
                .reshape(-1)
            )
            structural_anchors = (
                top_nodes.unsqueeze(-1).expand_as(top_sources).reshape(-1)
            )
            structural_sources = top_sources.reshape(-1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            structural_started = time.perf_counter()
            _, structurally_valid = model.structural_decode(
                batch,
                gate_types,
                live,
                structural_batch,
                structural_sources,
                structural_anchors,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            structural_seconds += time.perf_counter() - structural_started
            structurally_valid = structurally_valid.view_as(top_sources) & present

            for node_k in node_ks:
                key_k = str(node_k)
                effective_node_k = min(node_k, num_slots)
                counters[scorer_name]["positive_node_hits"][key_k] += int(
                    positive_node_ranks.le(effective_node_k).sum().item()
                )
                counters[scorer_name]["target_action_node_hits"][key_k] += int(
                    target_node_ranks.le(effective_node_k).sum().item()
                )
                selected_nodes = node_order[:, :effective_node_k]
                counters[scorer_name]["compatible_pairs_retained"][key_k] += int(
                    compatible_per_node.gather(1, selected_nodes).sum().item()
                )
                for batch_index in range(batch_size):
                    begin, end = positive_offsets[batch_index : batch_index + 2]
                    if begin != end and bool(
                        positive_node_ranks[begin:end].le(effective_node_k).any()
                    ):
                        counters[scorer_name]["states_with_positive_node"][key_k] += 1
                    if best_anchor_sets[batch_index]:
                        chosen = set(map(int, selected_nodes[batch_index].tolist()))
                        if chosen & best_anchor_sets[batch_index]:
                            counters[scorer_name]["best_gate_reduction_node_hits"][key_k] += 1
                for pattern_k in pattern_ks:
                    key_m = str(pattern_k)
                    effective_pattern_k = min(pattern_k, num_sources)
                    exact_hits = positive_node_ranks.le(effective_node_k)
                    exact_hits &= positive_pattern_ranks.le(effective_pattern_k)
                    counters[scorer_name]["exact_match_hits"][key_k][key_m] += int(
                        exact_hits.sum().item()
                    )
                    target_hits = target_node_ranks.le(effective_node_k)
                    target_hits &= target_pattern_ranks.le(effective_pattern_k)
                    counters[scorer_name]["target_action_hits"][key_k][key_m] += int(
                        target_hits.sum().item()
                    )
                    counters[scorer_name]["structurally_valid_candidates"][key_k][
                        key_m
                    ] += int(
                        structurally_valid[
                            :, : min(node_k, available_node_k), : min(pattern_k, available_pattern_k)
                        ]
                        .sum()
                        .item()
                    )

    rendered_scorers = {}
    for scorer_name, values in counters.items():
        rendered = {}
        for node_k in node_ks:
            key_k = str(node_k)
            rendered[key_k] = {
                "positive_anchor_recall": _safe_ratio(
                    values["positive_node_hits"][key_k], totals["exact_matches"]
                ),
                "states_with_any_positive_anchor_recall": _safe_ratio(
                    values["states_with_positive_node"][key_k],
                    totals["states_with_positives"],
                ),
                "target_action_anchor_recall": _safe_ratio(
                    values["target_action_node_hits"][key_k],
                    totals["target_actions"],
                ),
                "best_gate_reduction_anchor_recall": _safe_ratio(
                    values["best_gate_reduction_node_hits"][key_k],
                    totals["states_with_best_gate_reduction"],
                ),
                "compatible_pair_fraction": _safe_ratio(
                    values["compatible_pairs_retained"][key_k],
                    totals["compatible_node_source_pairs"],
                ),
                "pattern": {
                    str(pattern_k): {
                        "exact_match_recall": _safe_ratio(
                            values["exact_match_hits"][key_k][str(pattern_k)],
                            totals["exact_matches"],
                        ),
                        "target_action_recall": _safe_ratio(
                            values["target_action_hits"][key_k][str(pattern_k)],
                            totals["target_actions"],
                        ),
                        "structurally_valid_candidates": values[
                            "structurally_valid_candidates"
                        ][key_k][str(pattern_k)],
                        "mean_structurally_valid_candidates_per_state": _safe_ratio(
                            values["structurally_valid_candidates"][key_k][
                                str(pattern_k)
                            ],
                            totals["states"],
                        ),
                    }
                    for pattern_k in pattern_ks
                },
            }
        rendered_scorers[scorer_name] = rendered

    return {
        "totals": totals,
        "node_scorers": rendered_scorers,
        "timing": {
            "full_matcher_seconds": model_seconds,
            "audit_structural_decode_seconds": structural_seconds,
            "states_per_second_full_matcher": _safe_ratio(
                totals["states"], model_seconds
            ),
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--node-k", type=parse_int_list, default=(1, 2, 4, 8, 16))
    parser.add_argument(
        "--pattern-k", type=parse_int_list, default=(1, 4, 8, 16, 32, 64)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, rules, train_dataset, test_dataset = load_datasets(args.data)
    dataset = train_dataset if args.split == "train" else test_dataset
    if args.max_states is not None:
        dataset = Subset(dataset, range(min(args.max_states, len(dataset))))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    model = build_model(rules, len(payload["xfer_to_source"]), train_args).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_parameters = [
        name
        for name in incompatible.missing_keys
        if name in dict(model.named_parameters())
    ]
    if unexpected or missing_parameters:
        raise RuntimeError(
            f"incompatible checkpoint: unexpected={unexpected} "
            f"missing_parameters={missing_parameters}"
        )
    collate = collate_current_graphs if train_args.get("state_only", False) else collate_prefixes
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda samples: collate(samples, rules),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = audit(
        model,
        loader,
        device,
        rules,
        args.node_k,
        args.pattern_k,
    )
    elapsed = time.perf_counter() - started
    result = {
        "format": "hierarchical-action-audit-v1",
        "scope": {
            "split": args.split,
            "states": len(dataset),
            "node_k": args.node_k,
            "pattern_k": args.pattern_k,
            "teacher_node_scores": ["max", "logsumexp"],
            "note": (
                "Teacher node scores aggregate the full matcher and measure the "
                "factorization ceiling; they are not a deployable cheap node head."
            ),
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "peak_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        },
        "artifacts": {
            "data": str(args.data),
            "data_sha256": sha256(args.data),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
        },
        "elapsed_seconds": elapsed,
        **metrics,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
