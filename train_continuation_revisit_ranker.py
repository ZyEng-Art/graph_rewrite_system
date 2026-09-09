from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from evaluate_continuation_revisit_shadow import comparable_pairs


FEATURE_NAMES = (
    "origin_continuation_score",
    "action_depth",
    "attempted_actions_before",
    "descendant_gain_before",
    "novel_yield_before",
    "valid_yield_before",
)


def feature_row(row: dict) -> list[float]:
    return [
        float(row["continuation_score"]) / 8.0,
        float(row["action_depth"]) / 64.0,
        float(row["attempted_actions_before"]) / 1024.0,
        float(row["descendant_gain_before"]) / 8.0,
        float(row["novel_yield_before"]),
        float(row["valid_yield_before"]),
    ]


def load_rows(patterns: list[str]) -> tuple[list[dict], list[dict]]:
    paths = sorted({Path(path) for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise ValueError("no revisit audit files matched")
    rows = []
    sources = []
    for source_id, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        shadow = payload.get("continuation_revisit_shadow")
        if not shadow:
            raise ValueError(f"audit lacks revisit shadow data: {path}")
        begin = len(rows)
        rows.extend(dict(row, source_id=source_id) for row in shadow["rows"])
        sources.append(
            {
                "source_id": source_id,
                "path": str(path.resolve()),
                "qasm": payload.get("qasm"),
                "begin": begin,
                "end": len(rows),
            }
        )
    return rows, sources


def build_pair_tensors(
    rows: list[dict], *, min_additional_expansions: int
) -> dict[str, torch.Tensor]:
    pairs = comparable_pairs(
        rows, min_additional_expansions=min_additional_expansions
    )
    preferred = torch.tensor([row[0] for row in pairs], dtype=torch.long)
    rejected = torch.tensor([row[1] for row in pairs], dtype=torch.long)
    source_ids = torch.tensor([row[2][0] for row in pairs], dtype=torch.long)
    steps = torch.tensor([row[2][1] for row in pairs], dtype=torch.long)
    group_counts = Counter(zip(source_ids.tolist(), steps.tolist()))
    source_group_counts = Counter(source for source, _ in group_counts)
    weights = torch.tensor(
        [
            1.0
            / (
                group_counts[(int(source), int(step))]
                * source_group_counts[int(source)]
            )
            for source, step in zip(source_ids, steps)
        ],
        dtype=torch.float32,
    )
    if weights.numel():
        weights /= weights.mean()
    return {
        "preferred": preferred,
        "rejected": rejected,
        "source_ids": source_ids,
        "steps": steps,
        "weights": weights,
    }


def pair_metrics(
    scores: torch.Tensor,
    pairs: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    if not pairs["preferred"].numel():
        return {"pairs": 0}
    margins = scores[pairs["preferred"]] - scores[pairs["rejected"]]
    decisions = margins.gt(0).float() + 0.5 * margins.eq(0).float()
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_source: dict[int, list[float]] = defaultdict(list)
    per_source_groups: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for value, source, step in zip(
        decisions.tolist(), pairs["source_ids"].tolist(), pairs["steps"].tolist()
    ):
        grouped[(int(source), int(step))].append(float(value))
        per_source[int(source)].append(float(value))
        per_source_groups[int(source)][int(step)].append(float(value))
    return {
        "pairs": int(decisions.numel()),
        "accuracy": float(decisions.mean()),
        "group_macro_accuracy": sum(
            sum(values) / len(values) for values in grouped.values()
        )
        / len(grouped),
        "source_macro_accuracy": sum(
            sum(values) / len(values) for values in per_source.values()
        )
        / len(per_source),
        "sources": {
            str(source): {
                "pairs": len(values),
                "accuracy": sum(values) / len(values),
                "group_macro_accuracy": sum(
                    sum(group) / len(group)
                    for group in per_source_groups[source].values()
                )
                / len(per_source_groups[source]),
            }
            for source, values in sorted(per_source.items())
        },
    }


def paired_group_bootstrap_delta(
    model_scores: torch.Tensor,
    baseline_scores: torch.Tensor,
    pairs: dict[str, torch.Tensor],
    *,
    iterations: int = 4000,
    seed: int = 73,
) -> dict[str, float | list[float] | int]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (source, step) in enumerate(
        zip(pairs["source_ids"].tolist(), pairs["steps"].tolist())
    ):
        groups[(int(source), int(step))].append(index)
    if not groups:
        return {"groups": 0, "delta": float("nan"), "bootstrap_95pct": []}

    preferred = pairs["preferred"]
    rejected = pairs["rejected"]

    def decisions(scores: torch.Tensor) -> torch.Tensor:
        margins = scores[preferred] - scores[rejected]
        return margins.gt(0).float() + 0.5 * margins.eq(0).float()

    model_decisions = decisions(model_scores)
    baseline_decisions = decisions(baseline_scores)
    group_deltas = torch.tensor(
        [
            float(model_decisions[indices].mean() - baseline_decisions[indices].mean())
            for indices in groups.values()
        ]
    )
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(iterations):
        chosen = torch.randint(
            len(group_deltas), (len(group_deltas),), generator=generator
        )
        samples.append(float(group_deltas[chosen].mean()))
    interval = torch.tensor(samples).quantile(torch.tensor([0.025, 0.975]))
    return {
        "groups": len(group_deltas),
        "delta": float(group_deltas.mean()),
        "bootstrap_95pct": [float(interval[0]), float(interval[1])],
    }


def baseline_scores(rows: list[dict], field: str) -> torch.Tensor:
    return torch.tensor([float(row[field]) for row in rows], dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a circuit/group-balanced linear branch revisit ranker."
    )
    parser.add_argument("--train-audits", nargs="+", required=True)
    parser.add_argument("--validation-audits", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--min-additional-expansions", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_rows, train_sources = load_rows(args.train_audits)
    validation_rows, validation_sources = load_rows(args.validation_audits)
    train_pairs = build_pair_tensors(
        train_rows, min_additional_expansions=args.min_additional_expansions
    )
    validation_pairs = build_pair_tensors(
        validation_rows, min_additional_expansions=args.min_additional_expansions
    )
    if not train_pairs["preferred"].numel():
        raise ValueError("training audits contain no comparable revisit pairs")

    train_inputs = torch.tensor([feature_row(row) for row in train_rows])
    validation_inputs = torch.tensor([feature_row(row) for row in validation_rows])
    mean = train_inputs.mean(dim=0)
    scale = train_inputs.std(dim=0).clamp_min(1e-6)
    train_inputs = (train_inputs - mean) / scale
    validation_inputs = (validation_inputs - mean) / scale
    model = nn.Linear(len(FEATURE_NAMES), 1, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        scores = model(train_inputs).squeeze(1)
        margins = (
            scores[train_pairs["preferred"]] - scores[train_pairs["rejected"]]
        )
        loss = (F.softplus(-margins) * train_pairs["weights"]).mean()
        loss.backward()
        optimizer.step()
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 25 == 0:
            history.append({"epoch": epoch + 1, "loss": float(loss)})

    with torch.no_grad():
        train_scores = model(train_inputs).squeeze(1)
        validation_scores = model(validation_inputs).squeeze(1)

    def evaluate_split(rows, pairs, scores):
        baselines = {
            "continuation_score": baseline_scores(rows, "continuation_score"),
            "descendant_gain_before": baseline_scores(
                rows, "descendant_gain_before"
            ),
            "novel_yield_before": baseline_scores(rows, "novel_yield_before"),
            "valid_yield_before": baseline_scores(rows, "valid_yield_before"),
        }
        return {
            "model": pair_metrics(scores, pairs),
            "baselines": {
                name: pair_metrics(values, pairs)
                for name, values in baselines.items()
            },
            "model_minus_baseline_group_bootstrap": {
                name: paired_group_bootstrap_delta(
                    scores, values, pairs, seed=args.seed
                )
                for name, values in baselines.items()
            },
        }

    weights = model.weight.detach().squeeze(0) / scale
    checkpoint = {
        "format": "continuation_revisit_linear_ranker_v1",
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean,
        "feature_scale": scale,
        "model": model.state_dict(),
        "effective_raw_feature_weights": weights,
        "args": vars(args),
    }
    metrics = {
        "train_sources": train_sources,
        "validation_sources": validation_sources,
        "history": history,
        "effective_raw_feature_weights": {
            name: float(value) for name, value in zip(FEATURE_NAMES, weights)
        },
        "train": evaluate_split(train_rows, train_pairs, train_scores),
        "validation": evaluate_split(
            validation_rows, validation_pairs, validation_scores
        ),
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n"
    args.metrics.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
