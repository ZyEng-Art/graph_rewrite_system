from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from sibling_continuation_ranker import (
    SiblingContinuationRanker,
    continuation_ranker_inputs,
)
from train import autocast_context
from train_sibling_continuation_ranker import prefix_tensors


IGNORED_AUDIT_KEYS = {
    "checkpoint",
    "continuation_ranker_checkpoint",
    "continuation_shadow_scores",
    "qasm",
}


def values_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(values_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def load_ranker(checkpoint: dict, device: torch.device) -> tuple[SiblingContinuationRanker, int]:
    if checkpoint.get("format") != "sibling_continuation_ranker_v3":
        raise ValueError("unsupported continuation ranker checkpoint")
    train_args = checkpoint.get("args", {})
    if int(checkpoint["prefix_width"]) and checkpoint.get("prefix_alignment") != "row":
        raise ValueError("continuation prefix checkpoint is not row-aligned")
    model = SiblingContinuationRanker(
        int(checkpoint["input_width"]),
        int(checkpoint["hidden_width"]),
        float(train_args.get("dropout", 0.0)),
        base_probability_index=int(checkpoint["base_probability_index"]),
        num_xfers=int(checkpoint["num_xfers"]),
        prefix_width=int(checkpoint["prefix_width"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(train_args.get("prefix_max_length", 0))


@torch.no_grad()
def recompute_scores(
    audit: dict,
    checkpoint: dict,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model, prefix_max_length = load_ranker(checkpoint, device)
    inputs = continuation_ranker_inputs(audit).half().to(device).float()
    prefixes = lengths = None
    if model.prefix_width:
        prefixes, lengths = prefix_tensors(audit, prefix_max_length)
        prefixes = prefixes.to(device)
        lengths = lengths.to(device)
    scores = []
    for begin in range(0, inputs.shape[0], batch_size):
        end = begin + batch_size
        with autocast_context(device):
            scores.append(
                model(
                    inputs[begin:end],
                    prefixes[begin:end] if prefixes is not None else None,
                    lengths[begin:end] if lengths is not None else None,
                ).float()
            )
    return torch.cat(scores).cpu()


def pair_order_metrics(
    preferences: dict, online: torch.Tensor, offline: torch.Tensor
) -> dict[str, float | int]:
    rows = list(preferences.get("train_pairs", [])) + list(
        preferences.get("test_pairs", [])
    )
    preferred = torch.tensor([int(row["preferred_row"]) for row in rows])
    rejected = torch.tensor([int(row["rejected_row"]) for row in rows])
    online_margin = online[preferred] - online[rejected]
    offline_margin = offline[preferred] - offline[rejected]
    online_order = online_margin > 0
    offline_order = offline_margin > 0
    return {
        "pairs": len(rows),
        "online_accuracy": float(online_order.float().mean()) if rows else 0.0,
        "offline_accuracy": float(offline_order.float().mean()) if rows else 0.0,
        "order_disagreements": int(torch.count_nonzero(online_order != offline_order)),
        "minimum_absolute_online_margin": (
            float(online_margin.abs().min()) if rows else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that continuation shadow scoring preserves search data and offline ordering."
    )
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--shadow-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--score-atol", type=float, default=0.01)
    args = parser.parse_args()

    baseline = torch.load(args.baseline_audit, map_location="cpu", weights_only=False)
    shadow = torch.load(args.shadow_audit, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    preferences = torch.load(args.preferences, map_location="cpu", weights_only=False)
    if "continuation_shadow_scores" not in shadow:
        raise ValueError("shadow audit does not contain continuation scores")
    structural_differences = [
        key
        for key in sorted(set(baseline) | set(shadow))
        if key not in IGNORED_AUDIT_KEYS
        and not values_equal(baseline.get(key), shadow.get(key))
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offline = recompute_scores(shadow, checkpoint, device, args.batch_size)
    online = shadow["continuation_shadow_scores"].float()
    absolute_error = (offline - online).abs()
    pairs = pair_order_metrics(preferences, online, offline)
    report = {
        "baseline_audit": str(args.baseline_audit.resolve()),
        "shadow_audit": str(args.shadow_audit.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "preferences": str(args.preferences.resolve()),
        "device": str(device),
        "rows": int(online.numel()),
        "all_scores_finite": bool(torch.isfinite(online).all()),
        "exact_score_matches": int(torch.count_nonzero(absolute_error == 0)),
        "maximum_absolute_score_error": float(absolute_error.max()),
        "scores_within_tolerance": bool((absolute_error <= args.score_atol).all()),
        "structural_differences": structural_differences,
        "pair_order": pairs,
        "passed": bool(
            not structural_differences
            and torch.isfinite(online).all()
            and (absolute_error <= args.score_atol).all()
            and pairs["order_disagreements"] == 0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
