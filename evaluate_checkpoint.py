from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import collate_current_graphs, collate_prefixes, load_datasets
from model_factory import build_model
from train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate-multiplier", type=int, default=1)
    parser.add_argument("--include-neural-metrics", action="store_true")
    parser.add_argument("--include-terminal", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, rules, _, test_dataset = load_datasets(
        args.data, include_terminal=args.include_terminal
    )
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
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda samples: (
            collate_current_graphs(samples, rules)
            if train_args.get("state_only", False)
            else collate_prefixes(samples, rules)
        ),
        num_workers=0,
    )
    metrics = evaluate(
        model,
        loader,
        device,
        rules,
        candidate_multiplier=args.candidate_multiplier,
        include_neural_metrics=args.include_neural_metrics,
    )
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
