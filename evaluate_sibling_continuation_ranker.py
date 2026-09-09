from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sibling_continuation_ranker import SiblingContinuationRanker
from train_sibling_continuation_ranker import evaluate, load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen continuation ranker on another manifest."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != "sibling_continuation_ranker_v3":
        raise ValueError("evaluator requires a v3 continuation ranker")
    train_args = checkpoint["args"]
    prefix_max_length = int(train_args.get("prefix_max_length", 0))
    corpus = load_corpus(
        args.manifest, prefix_max_length=prefix_max_length
    )
    if corpus["input_width"] != int(checkpoint["input_width"]):
        raise ValueError("evaluation input width differs from checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiblingContinuationRanker(
        int(checkpoint["input_width"]),
        int(checkpoint["hidden_width"]),
        float(train_args.get("dropout", 0.0)),
        base_probability_index=int(checkpoint["base_probability_index"]),
        num_xfers=int(checkpoint["num_xfers"]),
        prefix_width=int(checkpoint["prefix_width"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "device": str(device),
        "evaluation": evaluate(
            model,
            corpus,
            corpus["test"],
            device,
            args.batch_size,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
