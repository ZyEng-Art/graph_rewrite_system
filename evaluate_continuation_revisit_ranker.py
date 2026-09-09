from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from train_continuation_revisit_ranker import (
    baseline_scores,
    build_pair_tensors,
    feature_row,
    load_rows,
    pair_metrics,
    paired_group_bootstrap_delta,
)


BASELINE_FIELDS = (
    "continuation_score",
    "descendant_gain_before",
    "novel_yield_before",
    "valid_yield_before",
)


def score_rows(checkpoint: dict, rows: list[dict]) -> torch.Tensor:
    feature_names = tuple(checkpoint["feature_names"])
    inputs = torch.tensor(
        [feature_row(row, feature_names) for row in rows], dtype=torch.float32
    )
    inputs = (inputs - checkpoint["feature_mean"]) / checkpoint["feature_scale"]
    model = nn.Linear(len(feature_names), 1, bias=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        return model(inputs).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen linear branch revisit ranker without retraining."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--min-additional-expansions", type=int, default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "continuation_revisit_linear_ranker_v1":
        raise ValueError("unsupported revisit-ranker checkpoint format")
    rows, sources = load_rows(args.audits)
    pairs = build_pair_tensors(
        rows, min_additional_expansions=args.min_additional_expansions
    )
    scores = score_rows(checkpoint, rows)
    baselines = {
        field: baseline_scores(rows, field) for field in BASELINE_FIELDS
    }
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_feature_names": list(checkpoint["feature_names"]),
        "sources": sources,
        "model": pair_metrics(scores, pairs),
        "baselines": {
            field: pair_metrics(values, pairs)
            for field, values in baselines.items()
        },
        "model_minus_baseline_group_bootstrap": {
            field: paired_group_bootstrap_delta(
                scores,
                values,
                pairs,
                iterations=args.bootstrap_iterations,
                seed=args.seed,
            )
            for field, values in baselines.items()
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    args.metrics.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
