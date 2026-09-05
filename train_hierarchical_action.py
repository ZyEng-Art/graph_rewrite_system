from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dataset import RuleMetadata
from model_factory import build_model
from ppo_core import (
    HierarchicalPPOActorCritic,
    build_policy_features,
    build_state_features,
)
from preference_dataset import (
    ActionPreferenceDataset,
    PrefixLengthBatchSampler,
    collate_preferences,
)
from threshold_inference import load_threshold_config
from train import autocast_context, move_batch
from train_hierarchical_node import encode_with_prefix_state


@dataclass(frozen=True)
class ActionPairMetrics:
    pairs: int
    loss: float
    accuracy: float
    mean_margin: float
    same_anchor_pairs: int
    same_anchor_accuracy: float
    same_source_pairs: int
    same_source_accuracy: float
    uphill_preferred_pairs: int
    uphill_preferred_accuracy: float
    accuracy_by_circuit: dict[str, float]


class WeightedActionPreferenceDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        *,
        return_weight: float,
        max_sample_weight: float,
    ) -> None:
        self.base = ActionPreferenceDataset(rows)
        self.rows = rows
        self.return_weight = return_weight
        self.max_sample_weight = max_sample_weight

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        row = self.rows[index]
        future_reduction = max(
            0.0,
            float(
                row.get(
                    "future_best_reduction",
                    row.get("preferred_descendants", row.get("advantage", 1)),
                )
            ),
        )
        sample["action_sample_weight"] = min(
            self.max_sample_weight,
            1.0 + self.return_weight * future_reduction,
        )
        return sample


def collate_action_preferences(samples: list[dict], rules: RuleMetadata) -> dict:
    batch = collate_preferences(samples, rules)
    batch["action_sample_weight"] = torch.tensor(
        [sample["action_sample_weight"] for sample in samples],
        dtype=torch.float,
    )
    return batch


def gate_deltas(rules: RuleMetadata) -> torch.Tensor:
    return torch.tensor(
        [
            len(rules.destination_gate_types[xfer_id])
            - len(rules.source_gate_types[rules.xfer_to_source[xfer_id]])
            for xfer_id in range(len(rules.xfer_to_source))
        ],
        dtype=torch.float,
    )


@torch.no_grad()
def action_policy_inputs(
    model,
    states: torch.Tensor,
    live: torch.Tensor,
    batch: dict,
    *,
    action: str,
    source_vectors: torch.Tensor,
    all_gate_deltas: torch.Tensor,
    threshold_config: dict,
    initial_gate_bias: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xfers = batch[f"{action}_xfers"]
    sources = batch[f"{action}_sources"]
    bindings = batch[f"{action}_bindings"]
    anchors = bindings[:, 0]
    batch_ids = torch.arange(states.shape[0], device=states.device)
    base_features = model.candidate_features(
        states,
        live,
        xfers,
        sources,
        bindings,
        batch_ids,
    )

    node_vectors = model.match_node_vectors(states)
    anchor_vectors = node_vectors[batch_ids, anchors]
    selected_sources = source_vectors.index_select(0, sources)
    raw_logits = (anchor_vectors * selected_sources).sum(-1)
    raw_logits = raw_logits / math.sqrt(model.retrieval_width)
    raw_logits = raw_logits + model.source_bias.index_select(0, sources)

    near_mask = batch["current_rewrite_distance"][batch_ids, anchors].le(2)
    near = threshold_config["groups"]["near"]
    far = threshold_config["groups"]["far"]
    scale = torch.where(
        near_mask,
        torch.as_tensor(near["scale"], device=states.device),
        torch.as_tensor(far["scale"], device=states.device),
    )
    bias = torch.where(
        near_mask,
        torch.as_tensor(near["bias"], device=states.device),
        torch.as_tensor(far["bias"], device=states.device),
    )
    probabilities = (raw_logits.float() * scale + bias).sigmoid()
    deltas = all_gate_deltas.to(states.device).index_select(0, xfers)
    features, base_logits = build_policy_features(
        base_features,
        probabilities,
        deltas,
        initial_gate_bias=initial_gate_bias,
    )
    return features, base_logits, deltas


@torch.no_grad()
def encode_preferences(
    model,
    loader,
    device: torch.device,
    *,
    rules: RuleMetadata,
    threshold_config: dict,
    initial_gate_bias: float,
) -> dict:
    source_vectors = model.retrieval_source(model.source_representations())
    all_gate_deltas = gate_deltas(rules).to(device)
    encoded: dict[str, list] = {
        "preferred_features": [],
        "preferred_base_logits": [],
        "preferred_deltas": [],
        "rejected_features": [],
        "rejected_base_logits": [],
        "rejected_deltas": [],
        "prefix_states": [],
        "state_features": [],
        "sample_weights": [],
        "same_anchor": [],
        "same_source": [],
        "circuits": [],
    }
    started = time.perf_counter()
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        states, live, _, prefix_states = encode_with_prefix_state(model, batch)
        state_features = build_state_features(states, live, live.sum(1))
        for action in ("preferred", "rejected"):
            features, base_logits, deltas = action_policy_inputs(
                model,
                states,
                live,
                batch,
                action=action,
                source_vectors=source_vectors,
                all_gate_deltas=all_gate_deltas,
                threshold_config=threshold_config,
                initial_gate_bias=initial_gate_bias,
            )
            encoded[f"{action}_features"].append(features.float().cpu())
            encoded[f"{action}_base_logits"].append(base_logits.float().cpu())
            encoded[f"{action}_deltas"].append(deltas.float().cpu())
        encoded["prefix_states"].append(prefix_states.float().cpu())
        encoded["state_features"].append(state_features.float().cpu())
        encoded["sample_weights"].append(cpu_batch["action_sample_weight"])
        encoded["same_anchor"].append(
            cpu_batch["preferred_bindings"][:, 0].eq(
                cpu_batch["rejected_bindings"][:, 0]
            )
        )
        encoded["same_source"].append(
            cpu_batch["preferred_sources"].eq(cpu_batch["rejected_sources"])
        )
        encoded["circuits"].extend(cpu_batch["preference_circuits"])
    for key in tuple(encoded):
        if key != "circuits":
            encoded[key] = torch.cat(encoded[key])
    encoded["encode_seconds"] = time.perf_counter() - started
    return encoded


def pair_scores(
    actor: HierarchicalPPOActorCritic,
    preferred_features: torch.Tensor,
    preferred_base_logits: torch.Tensor,
    rejected_features: torch.Tensor,
    rejected_base_logits: torch.Tensor,
    prefix_states: torch.Tensor,
    state_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    preferred = actor.candidate_policy_logits(
        preferred_features.unsqueeze(1),
        preferred_base_logits.unsqueeze(1),
        prefix_states,
        state_features,
    ).squeeze(1)
    rejected = actor.candidate_policy_logits(
        rejected_features.unsqueeze(1),
        rejected_base_logits.unsqueeze(1),
        prefix_states,
        state_features,
    ).squeeze(1)
    return preferred, rejected


@torch.no_grad()
def evaluate(
    actor: HierarchicalPPOActorCritic,
    encoded: dict,
    device: torch.device,
    *,
    batch_size: int,
    rank_margin: float,
) -> ActionPairMetrics:
    actor.eval()
    margins = []
    for begin in range(0, len(encoded["preferred_features"]), batch_size):
        end = begin + batch_size
        preferred, rejected = pair_scores(
            actor,
            encoded["preferred_features"][begin:end].to(device),
            encoded["preferred_base_logits"][begin:end].to(device),
            encoded["rejected_features"][begin:end].to(device),
            encoded["rejected_base_logits"][begin:end].to(device),
            encoded["prefix_states"][begin:end].to(device),
            encoded["state_features"][begin:end].to(device),
        )
        margins.append((preferred - rejected).float().cpu())
    margins = torch.cat(margins)
    correct = margins.gt(0)
    same_anchor = encoded["same_anchor"]
    same_source = encoded["same_source"]
    uphill = encoded["preferred_deltas"].gt(encoded["rejected_deltas"])

    def subset_accuracy(mask: torch.Tensor) -> float:
        return float(correct[mask].float().mean()) if bool(mask.any()) else 0.0

    circuit_correct: dict[str, list[bool]] = {}
    for circuit, value in zip(encoded["circuits"], correct.tolist()):
        circuit_correct.setdefault(circuit, []).append(value)
    return ActionPairMetrics(
        pairs=len(margins),
        loss=float(F.softplus(rank_margin - margins).mean()),
        accuracy=float(correct.float().mean()),
        mean_margin=float(margins.mean()),
        same_anchor_pairs=int(same_anchor.sum()),
        same_anchor_accuracy=subset_accuracy(same_anchor),
        same_source_pairs=int(same_source.sum()),
        same_source_accuracy=subset_accuracy(same_source),
        uphill_preferred_pairs=int(uphill.sum()),
        uphill_preferred_accuracy=subset_accuracy(uphill),
        accuracy_by_circuit={
            circuit: sum(values) / len(values)
            for circuit, values in sorted(circuit_correct.items())
        },
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--node-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=2.0)
    parser.add_argument("--score-l2", type=float, default=1e-3)
    parser.add_argument("--return-weight", type=float, default=0.25)
    parser.add_argument("--max-sample-weight", type=float, default=4.0)
    parser.add_argument("--same-anchor-weight", type=float, default=2.0)
    parser.add_argument("--initial-gate-bias", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--max-test-pairs", type=int)
    parser.add_argument("--seed", type=int, default=980)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.encode_batch_size < 1:
        parser.error("epochs and batch sizes must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    if payload.get("format") != "quartz-action-preference-v1":
        raise ValueError("expected quartz-action-preference-v1 dataset")
    rules = RuleMetadata.from_payload(payload)
    base_checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = build_model(
        rules, len(rules.xfer_to_source), base_checkpoint["args"]
    ).to(device)
    incompatible = model.load_state_dict(base_checkpoint["model"], strict=False)
    missing_parameters = [
        key for key in incompatible.missing_keys if key in dict(model.named_parameters())
    ]
    if incompatible.unexpected_keys or missing_parameters:
        raise RuntimeError(
            f"incompatible base checkpoint: missing={missing_parameters} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    model.requires_grad_(False)

    node_checkpoint = torch.load(
        args.node_checkpoint, map_location="cpu", weights_only=False
    )
    actor = HierarchicalPPOActorCritic(
        model.width, hidden_size=int(node_checkpoint["hidden_size"])
    ).to(device)
    actor.load_state_dict(node_checkpoint["actor_critic"])
    actor.requires_grad_(False)
    trainable = list(actor.pattern_norm.parameters()) + list(actor.pattern.parameters())
    for parameter in trainable:
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    train_rows = payload["train_preferences"]
    test_rows = payload["test_preferences"]
    if args.max_train_pairs is not None:
        train_rows = train_rows[: args.max_train_pairs]
    if args.max_test_pairs is not None:
        test_rows = test_rows[: args.max_test_pairs]
    train_dataset = WeightedActionPreferenceDataset(
        train_rows,
        return_weight=args.return_weight,
        max_sample_weight=args.max_sample_weight,
    )
    test_dataset = WeightedActionPreferenceDataset(
        test_rows,
        return_weight=args.return_weight,
        max_sample_weight=args.max_sample_weight,
    )
    collate = lambda samples: collate_action_preferences(samples, rules)

    def encode(dataset):
        loader = DataLoader(
            dataset,
            batch_sampler=PrefixLengthBatchSampler(dataset, args.encode_batch_size),
            collate_fn=collate,
            num_workers=0,
        )
        return encode_preferences(
            model,
            loader,
            device,
            rules=rules,
            threshold_config=load_threshold_config(
                args.calibration, args.target_recall
            ),
            initial_gate_bias=args.initial_gate_bias,
        )

    encoded_train = encode(train_dataset)
    encoded_test = encode(test_dataset)
    feature_dataset = TensorDataset(
        encoded_train["preferred_features"],
        encoded_train["preferred_base_logits"],
        encoded_train["rejected_features"],
        encoded_train["rejected_base_logits"],
        encoded_train["prefix_states"],
        encoded_train["state_features"],
        encoded_train["sample_weights"],
        encoded_train["same_anchor"],
    )
    generator = torch.Generator().manual_seed(args.seed)
    feature_loader = DataLoader(
        feature_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )

    initial_test = evaluate(
        actor,
        encoded_test,
        device,
        batch_size=args.batch_size,
        rank_margin=args.rank_margin,
    )
    print(json.dumps({"epoch": 0, "test": asdict(initial_test)}, sort_keys=True))
    best_accuracy = initial_test.accuracy
    best_epoch = 0
    best_metrics = initial_test
    best_actor_state = copy.deepcopy(actor.state_dict())
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        actor.pattern_norm.train()
        actor.pattern.train()
        total_loss = total_rank = total_l2 = 0.0
        batches = 0
        for row in feature_loader:
            (
                preferred_features,
                preferred_base_logits,
                rejected_features,
                rejected_base_logits,
                prefix_states,
                state_features,
                sample_weights,
                same_anchor,
            ) = (tensor.to(device, non_blocking=True) for tensor in row)
            preferred, rejected = pair_scores(
                actor,
                preferred_features,
                preferred_base_logits,
                rejected_features,
                rejected_base_logits,
                prefix_states,
                state_features,
            )
            margins = preferred - rejected
            weights = sample_weights * torch.where(
                same_anchor,
                torch.as_tensor(args.same_anchor_weight, device=device),
                torch.ones((), device=device),
            )
            weights = weights / weights.mean().clamp_min(1e-6)
            rank_loss = (F.softplus(args.rank_margin - margins) * weights).mean()
            residual_l2 = (
                (preferred - preferred_base_logits).square()
                + (rejected - rejected_base_logits).square()
            ).mean()
            loss = rank_loss + args.score_l2 * residual_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach())
            total_rank += float(rank_loss.detach())
            total_l2 += float(residual_l2.detach())
            batches += 1

        test_metrics = evaluate(
            actor,
            encoded_test,
            device,
            batch_size=args.batch_size,
            rank_margin=args.rank_margin,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / batches,
            "train_rank_loss": total_rank / batches,
            "train_residual_l2": total_l2 / batches,
            "test": asdict(test_metrics),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if test_metrics.accuracy > best_accuracy:
            best_accuracy = test_metrics.accuracy
            best_epoch = epoch
            best_metrics = test_metrics
            best_actor_state = copy.deepcopy(actor.state_dict())

    result = {
        "format": "hierarchical-action-training-v1",
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
        },
        "artifacts": {
            "data": str(args.data),
            "data_sha256": sha256(args.data),
            "base_checkpoint": str(args.checkpoint),
            "base_checkpoint_sha256": sha256(args.checkpoint),
            "node_checkpoint": str(args.node_checkpoint),
            "node_checkpoint_sha256": sha256(args.node_checkpoint),
            "calibration": str(args.calibration),
        },
        "args": vars(args),
        "width": model.width,
        "hidden_size": actor.hidden_size,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "train_pairs": len(train_dataset),
        "test_pairs": len(test_dataset),
        "encode_seconds": {
            "train": encoded_train["encode_seconds"],
            "test": encoded_test["encode_seconds"],
        },
        "initial_test": asdict(initial_test),
        "history": history,
        "selection": {
            "metric": "test_pair_accuracy",
            "best_epoch": best_epoch,
            "best_accuracy": best_accuracy,
            "best_metrics": asdict(best_metrics),
        },
        "training_seconds": time.perf_counter() - training_started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **result,
            "actor_critic": best_actor_state,
            "final_actor_critic": actor.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        args.output,
    )
    metrics_path = args.metrics_output or args.output.with_suffix(".training.json")
    metrics_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
