from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import collate_current_graphs, collate_prefixes, load_datasets
from model_factory import build_model
from threshold_inference import load_threshold_config, threshold_candidates
from train import autocast_context, move_batch


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--near-target-recall", type=float)
    parser.add_argument("--far-target-recall", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-candidates-per-state", type=int, default=2048)
    parser.add_argument("--trajectory-modulo", type=int, default=4)
    parser.add_argument("--excluded-remainder", type=int, default=0)
    parser.add_argument("--max-states", type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, rules, _, test_dataset = load_datasets(args.data, include_terminal=True)
    selected = [
        index
        for index in range(len(test_dataset))
        if test_dataset[index]["trajectory_id"] % args.trajectory_modulo
        != args.excluded_remainder
    ]
    if args.max_states is not None:
        selected = selected[: args.max_states]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    model = build_model(rules, len(payload["xfer_to_source"]), train_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    config = load_threshold_config(
        args.calibration,
        args.target_recall,
        near_target_recall=args.near_target_recall,
        far_target_recall=args.far_target_recall,
    )
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

    true_count = predicted_count = hits = 0
    per_state_candidates = []
    model_seconds = decode_seconds = 0.0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with autocast_context(device):
            states, live, gate_types = model.encode(batch)
            logits, eligible = model.match_logits(states, live, gate_types)
        torch.cuda.synchronize(device)
        model_seconds += time.perf_counter() - started
        started = time.perf_counter()
        predictions = threshold_candidates(
            model,
            batch,
            logits,
            eligible,
            config,
            max_candidates_per_state=args.max_candidates_per_state,
        )
        torch.cuda.synchronize(device)
        decode_seconds += time.perf_counter() - started
        for truth, predicted in zip(batch["positives"], predictions):
            truth_set = {
                (int(source), tuple(map(int, binding)))
                for source, binding in truth
            }
            predicted_set = {(source, binding) for source, _, binding, _ in predicted}
            true_count += len(truth_set)
            predicted_count += len(predicted_set)
            hits += len(truth_set & predicted_set)
            per_state_candidates.append(len(predicted_set))

    states = len(per_state_candidates)
    result = {
        "states": states,
        "true_matches": true_count,
        "predicted_matches": predicted_count,
        "hits": hits,
        "recall": hits / true_count,
        "precision_before_quartz_verification": hits / max(1, predicted_count),
        "mean_candidates_per_state": statistics.mean(per_state_candidates),
        "median_candidates_per_state": statistics.median(per_state_candidates),
        "max_candidates_per_state_observed": max(per_state_candidates),
        "model_ms_per_state": model_seconds * 1000.0 / states,
        "threshold_and_decode_ms_per_state": decode_seconds * 1000.0 / states,
        "total_ms_per_state": (model_seconds + decode_seconds) * 1000.0 / states,
        "target_recall": args.target_recall,
        "target_recall_by_group": config["target_recall_by_group"],
        "candidate_cap": args.max_candidates_per_state,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
