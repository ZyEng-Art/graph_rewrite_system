from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from sibling_continuation_ranker import (
    SiblingContinuationRanker,
    continuation_ranker_inputs,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prefix_tensors(payload: dict, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    histories = {
        int(row["node_id"]): [int(action[0]) + 1 for action in row["history"]]
        for row in payload["parent_histories"]
    }
    parent_ids = payload["parent_node_ids"].tolist()
    tokens = torch.zeros((len(parent_ids), max_length), dtype=torch.long)
    lengths = torch.zeros(len(parent_ids), dtype=torch.long)
    for row, parent_id in enumerate(parent_ids):
        history = histories[int(parent_id)][-max_length:]
        if history:
            tokens[row, : len(history)] = torch.tensor(history)
            lengths[row] = len(history)
    return tokens, lengths


def load_corpus(manifest_path: Path, *, prefix_max_length: int = 0) -> dict:
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    if manifest.get("format") != "frozen-sibling-continuation-preference-manifest-v1":
        raise ValueError("unsupported sibling preference manifest")
    sources = sorted(manifest["sources"], key=lambda row: int(row["source_id"]))
    if [int(row["source_id"]) for row in sources] != list(range(len(sources))):
        raise ValueError("source ids must be contiguous from zero")
    inputs = []
    offsets = []
    metadata = []
    prefixes = []
    prefix_lengths = []
    offset = 0
    width = None
    for source in sources:
        path = Path(source["path"])
        observed_sha = sha256(path)
        if observed_sha != source["sha256"]:
            raise ValueError(f"audit checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = continuation_ranker_inputs(payload).half()
        if width is None:
            width = int(rows.shape[1])
        elif int(rows.shape[1]) != width:
            raise ValueError("audit feature widths differ")
        offsets.append(offset)
        offset += int(rows.shape[0])
        inputs.append(rows)
        if prefix_max_length:
            source_prefixes, source_lengths = prefix_tensors(
                payload, prefix_max_length
            )
            prefixes.append(source_prefixes)
            prefix_lengths.append(source_lengths)
        metadata.append(
            {
                "rank": payload["action_parent_ranks"].long(),
                "gate_delta": payload["gate_deltas"].long(),
                "probability": payload["probabilities"].float(),
            }
        )

    def render_pairs(name: str) -> dict[str, torch.Tensor]:
        rows = manifest[name]
        preferred = torch.tensor(
            [offsets[int(row["source_id"])] + int(row["preferred_row"]) for row in rows],
            dtype=torch.long,
        )
        rejected = torch.tensor(
            [offsets[int(row["source_id"])] + int(row["rejected_row"]) for row in rows],
            dtype=torch.long,
        )
        advantages = torch.tensor(
            [int(row["advantage"]) for row in rows], dtype=torch.float32
        )
        source_ids = torch.tensor(
            [int(row["source_id"]) for row in rows], dtype=torch.long
        )
        preferred_local = torch.tensor(
            [int(row["preferred_row"]) for row in rows], dtype=torch.long
        )
        rejected_local = torch.tensor(
            [int(row["rejected_row"]) for row in rows], dtype=torch.long
        )
        return {
            "preferred": preferred,
            "rejected": rejected,
            "advantages": advantages,
            "source_ids": source_ids,
            "preferred_local": preferred_local,
            "rejected_local": rejected_local,
        }

    return {
        "manifest": manifest,
        "inputs": torch.cat(inputs),
        "metadata": metadata,
        "train": render_pairs("train_pairs"),
        "test": render_pairs("test_pairs"),
        "input_width": int(width or 0),
        "prefix_xfers": torch.cat(prefixes) if prefixes else None,
        "prefix_lengths": torch.cat(prefix_lengths) if prefix_lengths else None,
        "num_xfers": (
            int(max(int(tokens.max()) for tokens in prefixes))
            if prefixes
            else 0
        ),
    }


def tie_aware_accuracy(preferred: torch.Tensor, rejected: torch.Tensor) -> float:
    if not preferred.numel():
        return math.nan
    return float((preferred.gt(rejected).float() + 0.5 * preferred.eq(rejected)).mean())


@torch.no_grad()
def evaluate(
    model: SiblingContinuationRanker,
    corpus: dict,
    pairs: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    *,
    include_sources: bool = True,
) -> dict:
    count = int(pairs["preferred"].numel())
    if not count:
        return {"pairs": 0}
    model.eval()
    margins = []
    inputs = corpus["inputs"]
    prefix_xfers = corpus["prefix_xfers"]
    prefix_lengths = corpus["prefix_lengths"]
    for begin in range(0, count, batch_size):
        chosen = pairs["preferred"][begin : begin + batch_size]
        rejected = pairs["rejected"][begin : begin + batch_size]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            chosen_prefix = (
                prefix_xfers[chosen].to(device) if prefix_xfers is not None else None
            )
            rejected_prefix = (
                prefix_xfers[rejected].to(device)
                if prefix_xfers is not None
                else None
            )
            chosen_lengths = (
                prefix_lengths[chosen].to(device)
                if prefix_lengths is not None
                else None
            )
            rejected_lengths = (
                prefix_lengths[rejected].to(device)
                if prefix_lengths is not None
                else None
            )
            margin = model(
                inputs[chosen].to(device), chosen_prefix, chosen_lengths
            ) - model(
                inputs[rejected].to(device), rejected_prefix, rejected_lengths
            )
        margins.append(margin.float().cpu())
    margins = torch.cat(margins)
    advantages = pairs["advantages"]
    weights = advantages.sqrt()
    correct = margins.gt(0).float() + 0.5 * margins.eq(0)

    preferred_rank = []
    rejected_rank = []
    preferred_gate = []
    rejected_gate = []
    preferred_probability = []
    rejected_probability = []
    for source, left, right in zip(
        pairs["source_ids"].tolist(),
        pairs["preferred_local"].tolist(),
        pairs["rejected_local"].tolist(),
    ):
        row = corpus["metadata"][source]
        preferred_rank.append(row["rank"][left])
        rejected_rank.append(row["rank"][right])
        preferred_gate.append(row["gate_delta"][left])
        rejected_gate.append(row["gate_delta"][right])
        preferred_probability.append(row["probability"][left])
        rejected_probability.append(row["probability"][right])
    result = {
        "pairs": count,
        "accuracy": float(correct.mean()),
        "advantage_weighted_accuracy": float((correct * weights).sum() / weights.sum()),
        "mean_margin": float(margins.mean()),
        "positive_margin_p10": float(torch.quantile(margins, 0.1)),
        "baselines": {
            "lower_action_rank_accuracy": tie_aware_accuracy(
                -torch.stack(preferred_rank), -torch.stack(rejected_rank)
            ),
            "lower_gate_delta_accuracy": tie_aware_accuracy(
                -torch.stack(preferred_gate), -torch.stack(rejected_gate)
            ),
            "higher_match_probability_accuracy": tie_aware_accuracy(
                torch.stack(preferred_probability), torch.stack(rejected_probability)
            ),
        },
    }
    if include_sources:
        result["sources"] = []
        for source_id in pairs["source_ids"].unique(sorted=True).tolist():
            mask = pairs["source_ids"].eq(int(source_id))
            selected_pairs = {
                key: value[mask] for key, value in pairs.items()
            }
            source = corpus["manifest"]["sources"][int(source_id)]
            result["sources"].append(
                {
                    "source_id": int(source_id),
                    "path": source["path"],
                    **evaluate(
                        model,
                        corpus,
                        selected_pairs,
                        device,
                        batch_size,
                        include_sources=False,
                    ),
                }
            )
    return result


def train(args) -> tuple[SiblingContinuationRanker, dict]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = load_corpus(
        args.manifest, prefix_max_length=args.prefix_max_length
    )
    train_pairs = corpus["train"]
    if not train_pairs["preferred"].numel():
        raise ValueError("manifest contains no training pairs")
    model = SiblingContinuationRanker(
        corpus["input_width"],
        args.hidden_width,
        args.dropout,
        base_probability_index=corpus["input_width"] - 8,
        num_xfers=corpus["num_xfers"],
        prefix_width=args.prefix_width if args.prefix_max_length else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    inputs = corpus["inputs"]
    prefix_xfers = corpus["prefix_xfers"]
    prefix_lengths = corpus["prefix_lengths"]
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(train_pairs["preferred"].numel())
        totals = {"loss": 0.0, "pairs": 0}
        started = time.perf_counter()
        for begin in range(0, order.numel(), args.batch_size):
            selected = order[begin : begin + args.batch_size]
            preferred = train_pairs["preferred"][selected]
            rejected = train_pairs["rejected"][selected]
            weights = train_pairs["advantages"][selected].sqrt().to(device)
            weights = weights / weights.mean()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                preferred_inputs = inputs[preferred].to(device)
                rejected_inputs = inputs[rejected].to(device)
                preferred_prefix = (
                    prefix_xfers[preferred].to(device)
                    if prefix_xfers is not None
                    else None
                )
                rejected_prefix = (
                    prefix_xfers[rejected].to(device)
                    if prefix_xfers is not None
                    else None
                )
                preferred_lengths = (
                    prefix_lengths[preferred].to(device)
                    if prefix_lengths is not None
                    else None
                )
                rejected_lengths = (
                    prefix_lengths[rejected].to(device)
                    if prefix_lengths is not None
                    else None
                )
                preferred_score = model(
                    preferred_inputs, preferred_prefix, preferred_lengths
                )
                rejected_score = model(
                    rejected_inputs, rejected_prefix, rejected_lengths
                )
                margin = preferred_score - rejected_score
                ranking_loss = (F.softplus(-margin) * weights).mean()
                residual_loss = 0.5 * (
                    model.residual(
                        preferred_inputs, preferred_prefix, preferred_lengths
                    ).square().mean()
                    + model.residual(
                        rejected_inputs, rejected_prefix, rejected_lengths
                    ).square().mean()
                )
                loss = ranking_loss + args.residual_penalty * residual_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            totals["loss"] += float(loss.detach()) * selected.numel()
            totals["pairs"] += selected.numel()
        row = {
            "epoch": epoch + 1,
            "loss": totals["loss"] / totals["pairs"],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    metrics = {
        "device": str(device),
        "train": evaluate(model, corpus, train_pairs, device, args.eval_batch_size),
        "test": evaluate(model, corpus, corpus["test"], device, args.eval_batch_size),
        "history": history,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a same-parent future-descendant ranker."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--prefix-width", type=int, default=32)
    parser.add_argument("--prefix-max-length", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--residual-penalty",
        type=float,
        default=0.01,
        help="L2 penalty on deviations from the frozen matcher logit baseline",
    )
    parser.add_argument("--seed", type=int, default=907)
    args = parser.parse_args()
    if args.residual_penalty < 0:
        parser.error("--residual-penalty must be nonnegative")
    if args.prefix_width < 1 or args.prefix_max_length < 0:
        parser.error("prefix width must be positive and max length nonnegative")
    model, metrics = train(args)
    checkpoint = {
        "format": "sibling_continuation_ranker_v3",
        "args": vars(args),
        "input_width": model.input_width,
        "hidden_width": model.hidden_width,
        "base_probability_index": model.base_probability_index,
        "num_xfers": model.num_xfers,
        "prefix_width": model.prefix_width,
        "model": model.state_dict(),
        "metrics": metrics,
        "feature_spec": (
            "frozen matcher action features + probability/gate/step/rank/"
            "expansion/stagnation/depth; no post-search labels"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    args.metrics.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
