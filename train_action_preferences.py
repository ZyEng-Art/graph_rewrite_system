from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import RuleMetadata
from model_factory import build_model
from preference_dataset import (
    ActionPreferenceDataset,
    PrefixLengthBatchSampler,
    collate_preferences,
)
from train import autocast_context, move_batch


@torch.no_grad()
def encode_preferences(model, loader, device: torch.device) -> dict:
    model.eval()
    preferred_features = []
    rejected_features = []
    advantages = []
    circuits = []
    started = time.perf_counter()
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with autocast_context(device):
            states, live, _ = model.encode(batch)
            preferred = model.action_value_features(
                states,
                live,
                batch["preferred_xfers"],
                batch["preferred_sources"],
                batch["preferred_bindings"],
            )
            rejected = model.action_value_features(
                states,
                live,
                batch["rejected_xfers"],
                batch["rejected_sources"],
                batch["rejected_bindings"],
            )
        preferred_features.append(preferred.float().cpu())
        rejected_features.append(rejected.float().cpu())
        advantages.append(cpu_batch["preference_advantage"])
        circuits.extend(cpu_batch["preference_circuits"])
    return {
        "preferred": torch.cat(preferred_features),
        "rejected": torch.cat(rejected_features),
        "advantages": torch.cat(advantages),
        "circuits": circuits,
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate_head(model, encoded: dict, device: torch.device) -> dict:
    preferred = encoded["preferred"].to(device)
    rejected = encoded["rejected"].to(device)
    with autocast_context(device):
        preferred_scores = model.action_value_mlp(
            model.action_value_norm(preferred)
        ).squeeze(-1)
        rejected_scores = model.action_value_mlp(
            model.action_value_norm(rejected)
        ).squeeze(-1)
    margins = (preferred_scores - rejected_scores).float().cpu()
    by_circuit: dict[str, list[float]] = defaultdict(list)
    for circuit, margin in zip(encoded["circuits"], margins.tolist()):
        by_circuit[circuit].append(margin)
    return {
        "pairs": len(margins),
        "accuracy": float(margins.gt(0).float().mean()),
        "mean_margin": float(margins.mean()),
        "loss": float(F.softplus(-margins).mean()),
        "accuracy_by_circuit": {
            circuit: sum(value > 0 for value in values) / len(values)
            for circuit, values in sorted(by_circuit.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--score-l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--eval-every", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    if payload.get("format") != "quartz-action-preference-v1":
        raise ValueError("expected quartz-action-preference-v1 dataset")
    rules = RuleMetadata.from_payload(payload)
    initial = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    model_args = dict(initial["args"])
    model_args["action_value_head"] = True
    model = build_model(rules, len(rules.xfer_to_source), model_args).to(device)
    incompatible = model.load_state_dict(initial["model"], strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith("action_value_") for key in incompatible.missing_keys
    ):
        raise RuntimeError(
            "incompatible initialization checkpoint: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if name.startswith("action_value_"):
            parameter.requires_grad = True

    collate = lambda samples: collate_preferences(samples, rules)
    train_dataset = ActionPreferenceDataset(payload["train_preferences"])
    test_dataset = ActionPreferenceDataset(payload["test_preferences"])
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PrefixLengthBatchSampler(
            train_dataset, args.encode_batch_size
        ),
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=PrefixLengthBatchSampler(
            test_dataset, args.encode_batch_size
        ),
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    encoded_train = encode_preferences(model, train_loader, device)
    encoded_test = encode_preferences(model, test_loader, device)
    for split, encoded in (("train", encoded_train), ("test", encoded_test)):
        for key in ("preferred", "rejected"):
            if not bool(torch.isfinite(encoded[key]).all()):
                raise RuntimeError(f"non-finite {split} {key} features")
    print(
        f"device={device} train_pairs={len(encoded_train['preferred'])} "
        f"test_pairs={len(encoded_test['preferred'])} "
        f"encode_seconds={encoded_train['seconds'] + encoded_test['seconds']:.2f}",
        flush=True,
    )

    feature_dataset = TensorDataset(
        encoded_train["preferred"],
        encoded_train["rejected"],
        encoded_train["advantages"],
    )
    generator = torch.Generator().manual_seed(args.seed)
    feature_loader = DataLoader(
        feature_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial_metrics = evaluate_head(model, encoded_test, device)
    print(
        f"trainable_parameters={sum(p.numel() for p in trainable):,} "
        f"initial_test={json.dumps(initial_metrics, sort_keys=True)}",
        flush=True,
    )

    best_accuracy = -1.0
    best_metrics = None
    best_epoch = None
    training_log = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_pairs": len(encoded_train["preferred"]),
        "test_pairs": len(encoded_test["preferred"]),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "encode_seconds": encoded_train["seconds"] + encoded_test["seconds"],
        "initial_test": initial_metrics,
        "epochs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.action_value_norm.train()
        model.action_value_mlp.train()
        total_loss = total_rank = total_l2 = 0.0
        batches = 0
        started = time.perf_counter()
        for preferred_cpu, rejected_cpu, _ in feature_loader:
            preferred = preferred_cpu.to(device, non_blocking=True)
            rejected = rejected_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                preferred_scores = model.action_value_mlp(
                    model.action_value_norm(preferred)
                ).squeeze(-1)
                rejected_scores = model.action_value_mlp(
                    model.action_value_norm(rejected)
                ).squeeze(-1)
                rank_loss = F.softplus(-(preferred_scores - rejected_scores)).mean()
                l2_loss = (preferred_scores.square() + rejected_scores.square()).mean()
                loss = rank_loss + args.score_l2 * l2_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_rank += float(rank_loss.detach())
            total_l2 += float(l2_loss.detach())
            batches += 1
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.action_value_norm.eval()
            model.action_value_mlp.eval()
            metrics = evaluate_head(model, encoded_test, device)
            epoch_seconds = time.perf_counter() - started
            print(
                f"epoch={epoch:03d} loss={total_loss / batches:.4f} "
                f"rank={total_rank / batches:.4f} l2={total_l2 / batches:.4f} "
                f"seconds={epoch_seconds:.2f} "
                f"eval={json.dumps(metrics, sort_keys=True)}",
                flush=True,
            )
            training_log["epochs"].append(
                {
                    "epoch": epoch,
                    "loss": total_loss / batches,
                    "rank_loss": total_rank / batches,
                    "score_l2": total_l2 / batches,
                    "seconds": epoch_seconds,
                    "evaluation": metrics,
                }
            )
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_metrics = metrics
                best_epoch = epoch
                checkpoint_args = dict(model_args)
                checkpoint_args.update(
                    {
                        "preference_data": args.data,
                        "preference_init_checkpoint": args.init_checkpoint,
                        "preference_epochs": args.epochs,
                        "preference_learning_rate": args.learning_rate,
                        "preference_score_l2": args.score_l2,
                    }
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": checkpoint_args,
                        "metrics": initial.get("metrics"),
                        "preference_metrics": metrics,
                        "format": initial.get("format"),
                    },
                    args.output,
                )
                args.output.with_suffix(".metrics.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n"
                )
            training_log["best_epoch"] = best_epoch
            training_log["best_test"] = best_metrics
            args.output.with_suffix(".training.json").write_text(
                json.dumps(training_log, indent=2, sort_keys=True) + "\n"
            )
    print("best " + json.dumps(best_metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
