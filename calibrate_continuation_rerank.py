from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from train_sibling_continuation_ranker import load_corpus
from verify_continuation_shadow import load_ranker


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = probability + z * z / (2.0 * total)
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


@torch.no_grad()
def score_corpus(
    model,
    corpus: dict,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    inputs = corpus["inputs"]
    prefixes = corpus["prefix_xfers"]
    lengths = corpus["prefix_lengths"]
    scores = []
    for begin in range(0, len(inputs), batch_size):
        end = begin + batch_size
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            scores.append(
                model(
                    inputs[begin:end].to(device),
                    prefixes[begin:end].to(device) if prefixes is not None else None,
                    lengths[begin:end].to(device) if lengths is not None else None,
                ).float()
            )
    return torch.cat(scores).cpu()


def intervention_metrics(
    corpus: dict,
    pairs: dict[str, torch.Tensor],
    scores: torch.Tensor,
    *,
    max_matcher_logit_gap: float,
    min_continuation_score_margin: float,
) -> dict[str, int | float]:
    preferred = pairs["preferred"]
    rejected = pairs["rejected"]
    inputs = corpus["inputs"].float()
    probability_index = corpus["input_width"] - 8
    gate_delta_index = corpus["input_width"] - 7
    preferred_probability = inputs[preferred, probability_index].clamp(1e-6, 1 - 1e-6)
    rejected_probability = inputs[rejected, probability_index].clamp(1e-6, 1 - 1e-6)
    matcher_margin = torch.logit(preferred_probability) - torch.logit(
        rejected_probability
    )
    continuation_margin = scores[preferred] - scores[rejected]
    equal_immediate_gate = inputs[preferred, gate_delta_index].eq(
        inputs[rejected, gate_delta_index]
    )
    disagree = matcher_margin.gt(0) != continuation_margin.gt(0)
    eligible = (
        equal_immediate_gate
        & disagree
        & matcher_margin.abs().le(max_matcher_logit_gap)
        & continuation_margin.abs().ge(min_continuation_score_margin)
    )
    corrected = eligible & matcher_margin.le(0) & continuation_margin.gt(0)
    harmed = eligible & matcher_margin.gt(0) & continuation_margin.le(0)
    interventions = int(eligible.sum())
    corrected_count = int(corrected.sum())
    harmed_count = int(harmed.sum())
    return {
        "pairs": int(preferred.numel()),
        "equal_immediate_gate_pairs": int(equal_immediate_gate.sum()),
        "interventions": interventions,
        "corrected": corrected_count,
        "harmed": harmed_count,
        "net_corrected": corrected_count - harmed_count,
        "intervention_precision": (
            corrected_count / interventions if interventions else 0.0
        ),
        "intervention_precision_wilson_lower_95pct": wilson_lower(
            corrected_count, interventions
        ),
        "coverage": interventions / max(1, int(preferred.numel())),
        "max_matcher_logit_gap": max_matcher_logit_gap,
        "min_continuation_score_margin": min_continuation_score_margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Choose conservative continuation-rerank abstention thresholds on training pairs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--min-train-interventions", type=int, default=10)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prefix_max_length = int(checkpoint.get("args", {}).get("prefix_max_length", 0))
    corpus = load_corpus(args.manifest, prefix_max_length=prefix_max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_ranker(checkpoint, device)
    scores = score_corpus(model, corpus, device, args.batch_size)
    grid = []
    for matcher_gap in (0.05, 0.1, 0.25, 0.5, 1.0):
        for continuation_margin in (0.1, 0.25, 0.5, 1.0, 2.0):
            grid.append(
                intervention_metrics(
                    corpus,
                    corpus["train"],
                    scores,
                    max_matcher_logit_gap=matcher_gap,
                    min_continuation_score_margin=continuation_margin,
                )
            )
    eligible = [
        row for row in grid if row["interventions"] >= args.min_train_interventions
    ]
    if not eligible:
        raise ValueError("no threshold setting has enough training interventions")
    recommended = max(
        eligible,
        key=lambda row: (
            row["intervention_precision_wilson_lower_95pct"],
            row["net_corrected"],
            -row["coverage"],
        ),
    )
    validation = intervention_metrics(
        corpus,
        corpus["test"],
        scores,
        max_matcher_logit_gap=float(recommended["max_matcher_logit_gap"]),
        min_continuation_score_margin=float(
            recommended["min_continuation_score_margin"]
        ),
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "device": str(device),
        "selection_data": "train_pairs_only",
        "min_train_interventions": args.min_train_interventions,
        "recommended": {
            "max_matcher_logit_gap": recommended["max_matcher_logit_gap"],
            "min_continuation_score_margin": recommended[
                "min_continuation_score_margin"
            ],
            "max_promotions_per_parent": 1,
        },
        "recommended_train_metrics": recommended,
        "recommended_validation_metrics": validation,
        "training_grid": grid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
