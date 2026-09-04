from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import collate_current_graphs, collate_prefixes, load_datasets
from model_factory import build_model
from train import autocast_context, move_batch


def fit_affine_probability(
    positives: torch.Tensor,
    negatives: torch.Tensor,
    negative_weight: float,
) -> tuple[float, float]:
    values = torch.cat((positives, negatives)).double()
    labels = torch.cat((torch.ones_like(positives), torch.zeros_like(negatives))).double()
    weights = torch.cat(
        (
            torch.ones_like(positives),
            torch.full_like(negatives, negative_weight),
        )
    ).double()
    log_scale = torch.zeros((), dtype=torch.double, requires_grad=True)
    bias = torch.zeros((), dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (log_scale, bias), max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        calibrated = log_scale.exp() * values + bias
        loss = (
            F.binary_cross_entropy_with_logits(calibrated, labels, reduction="none")
            * weights
        ).sum() / weights.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_scale.detach().exp()), float(bias.detach())


def raw_threshold(values: torch.Tensor, target_recall: float) -> float:
    ordered = values.sort().values
    index = max(0, min(len(ordered) - 1, math.floor((1.0 - target_recall) * len(ordered))))
    return float(ordered[index])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trajectory-modulo", type=int, default=4)
    parser.add_argument("--trajectory-remainder", type=int, default=0)
    parser.add_argument("--max-states", type=int, default=512)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument(
        "--target-recalls", type=float, nargs="+", default=(0.95, 0.97, 0.98, 0.99)
    )
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, rules, _, test_dataset = load_datasets(args.data, include_terminal=True)
    selected = [
        index
        for index in range(len(test_dataset))
        if test_dataset[index]["trajectory_id"] % args.trajectory_modulo
        == args.trajectory_remainder
    ][: args.max_states]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    model = build_model(rules, len(payload["xfer_to_source"]), train_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(
        Subset(test_dataset, selected),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda samples: (
            collate_prefixes(samples, rules)
            if train_args.get("architecture") == "paged_action"
            else collate_current_graphs(samples, rules)
        ),
        num_workers=0,
    )

    positive_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    negative_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    total_negatives = defaultdict(int)
    sampled_negatives = defaultdict(int)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    state_count = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with autocast_context(device):
            states, live, gate_types = model.encode(batch)
            logits, eligible = model.match_logits(states, live, gate_types)
        for batch_index, rows in enumerate(batch["positives"]):
            target = torch.zeros_like(eligible[batch_index])
            anchors = torch.tensor(
                [binding[0] for _, binding in rows],
                dtype=torch.long,
                device=device,
            )
            sources = torch.tensor(
                [source for source, _ in rows], dtype=torch.long, device=device
            )
            target[anchors, sources] = True
            distances = batch["current_rewrite_distance"][batch_index]
            positive_is_near = distances[anchors].le(2)
            scores = logits[batch_index, anchors, sources].float()
            for name, mask in (("near", positive_is_near), ("far", ~positive_is_near)):
                if bool(mask.any()):
                    positive_scores[name].append(scores[mask].cpu())

            negative_mask = eligible[batch_index] & ~target
            negative_positions = negative_mask.flatten().nonzero(
                as_tuple=False
            ).squeeze(1)
            negative_anchors = torch.div(
                negative_positions, model.num_sources, rounding_mode="floor"
            )
            negative_is_near = distances[negative_anchors].le(2)
            for name, group_mask in (("near", negative_is_near), ("far", ~negative_is_near)):
                positions = negative_positions[group_mask]
                total_negatives[name] += positions.numel()
                positive_count = max(1, int((positive_is_near if name == "near" else ~positive_is_near).sum()))
                sample_count = min(
                    positions.numel(), args.negative_ratio * positive_count
                )
                if sample_count:
                    order = torch.randperm(
                        positions.numel(), generator=generator, device=device
                    )[:sample_count]
                    negative_scores[name].append(
                        logits[batch_index].flatten()[positions[order]].float().cpu()
                    )
                    sampled_negatives[name] += sample_count
            state_count += 1

    result = {
        "checkpoint": str(args.checkpoint),
        "calibration_states": state_count,
        "trajectory_modulo": args.trajectory_modulo,
        "trajectory_remainder": args.trajectory_remainder,
        "groups": {},
    }
    for name in ("near", "far"):
        positives = torch.cat(positive_scores[name])
        negatives = torch.cat(negative_scores[name])
        negative_weight = total_negatives[name] / sampled_negatives[name]
        scale, bias = fit_affine_probability(
            positives, negatives, negative_weight
        )
        rows = {}
        for recall in args.target_recalls:
            threshold = raw_threshold(positives, recall)
            probability = torch.sigmoid(torch.tensor(scale * threshold + bias)).item()
            true_positive = int(positives.ge(threshold).sum())
            estimated_false_positive = float(
                negatives.ge(threshold).sum() * negative_weight
            )
            rows[f"{recall:.4f}"] = {
                "raw_logit_threshold": threshold,
                "calibrated_probability_threshold": probability,
                "positive_recall": true_positive / len(positives),
                "estimated_precision_before_structural_decode": (
                    true_positive / max(1.0, true_positive + estimated_false_positive)
                ),
                "estimated_candidates_per_state_before_structural_decode": (
                    (true_positive + estimated_false_positive) / state_count
                ),
            }
        result["groups"][name] = {
            "positive_count": len(positives),
            "total_negative_count": total_negatives[name],
            "sampled_negative_count": sampled_negatives[name],
            "probability_scale": scale,
            "probability_bias": bias,
            "thresholds": rows,
        }

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
