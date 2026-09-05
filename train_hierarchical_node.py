from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dataset import collate_prefixes, load_datasets
from model_factory import build_model
from ppo_core import HierarchicalPPOActorCritic, build_state_features
from train import autocast_context, move_batch


@dataclass(frozen=True)
class NodeMetrics:
    states: int
    loss: float
    mean_future_best_reduction: float
    target_anchor_recall: dict[str, float]
    any_exact_anchor_recall: dict[str, float]


class ReturnWeightedPrefixes(Dataset):
    def __init__(
        self,
        dataset,
        *,
        return_weight: float,
        max_sample_weight: float,
    ) -> None:
        self.dataset = dataset
        self.return_weight = return_weight
        self.max_sample_weight = max_sample_weight
        self.future_reductions = []
        self.prefix_lengths = []
        for trajectory_id, prefix_length in dataset.indices:
            trajectory = dataset.trajectories[trajectory_id]
            counts = trajectory_gate_counts(trajectory)
            current = counts[prefix_length]
            self.future_reductions.append(current - min(counts[prefix_length:]))
            self.prefix_lengths.append(prefix_length)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        reduction = self.future_reductions[index]
        return {
            **sample,
            "future_best_reduction": reduction,
            "node_sample_weight": min(
                self.max_sample_weight,
                1.0 + self.return_weight * max(0, reduction),
            ),
        }


class LengthBucketBatchSampler:
    """Shuffle batches while keeping their maximum replay length bounded."""

    def __init__(
        self,
        lengths: list[int],
        *,
        batch_size: int,
        bucket_width: int,
        seed: int,
    ) -> None:
        if batch_size <= 0 or bucket_width <= 0:
            raise ValueError("batch size and bucket width must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.buckets: dict[int, list[int]] = {}
        for index, length in enumerate(lengths):
            self.buckets.setdefault(length // bucket_width, []).append(index)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for rows in self.buckets.values():
            order = torch.randperm(len(rows), generator=generator).tolist()
            shuffled = [rows[index] for index in order]
            batches.extend(
                shuffled[begin : begin + self.batch_size]
                for begin in range(0, len(shuffled), self.batch_size)
            )
        order = torch.randperm(len(batches), generator=generator).tolist()
        yield from (batches[index] for index in order)

    def __len__(self) -> int:
        return sum(
            (len(rows) + self.batch_size - 1) // self.batch_size
            for rows in self.buckets.values()
        )


def prefix_lengths(dataset) -> list[int]:
    if isinstance(dataset, Subset):
        return [dataset.dataset.prefix_lengths[index] for index in dataset.indices]
    return list(dataset.prefix_lengths)


def trajectory_gate_counts(trajectory: dict) -> list[int]:
    counts = [len(trajectory["initial_graph"]["nodes"])]
    for step in trajectory["steps"]:
        delta = step["delta"]
        counts.append(
            counts[-1]
            - len(delta["removed_slots"])
            + len(delta["added_nodes"])
        )
    return counts


def collate_node_samples(samples: list[dict], rules) -> dict:
    batch = collate_prefixes(samples, rules)
    batch["future_best_reduction"] = torch.tensor(
        [sample["future_best_reduction"] for sample in samples],
        dtype=torch.float,
    )
    batch["node_sample_weight"] = torch.tensor(
        [sample["node_sample_weight"] for sample in samples],
        dtype=torch.float,
    )
    return batch


@torch.no_grad()
def encode_with_prefix_state(model, batch: dict) -> tuple[torch.Tensor, ...]:
    action_outputs: list[torch.Tensor] = []

    def capture_action_output(_module, _inputs, output) -> None:
        action_outputs.append(output)

    hook = model.action_output_norm.register_forward_hook(capture_action_output)
    try:
        with autocast_context(batch["current_types"].device):
            states, live, gate_types = model.encode(batch)
    finally:
        hook.remove()

    live_float = live.unsqueeze(-1)
    graph_pool = (states * live_float).sum(1)
    graph_pool = graph_pool / live_float.sum(1).clamp_min(1)
    if not action_outputs:
        return states, live, gate_types, graph_pool

    history = torch.stack(action_outputs, dim=1)
    action_mask = batch["action_xfers"].ge(0)
    history = history * action_mask.unsqueeze(-1)
    lengths = action_mask.sum(1)
    last_indices = lengths.sub(1).clamp_min(0)
    last_actions = history[
        torch.arange(history.shape[0], device=history.device), last_indices
    ]
    prefix_states = torch.where(
        lengths.gt(0).unsqueeze(-1), last_actions, graph_pool
    )
    return states, live, gate_types, prefix_states


def target_anchors(target_actions: list[dict | None], device: torch.device) -> torch.Tensor:
    if any(action is None for action in target_actions):
        raise ValueError("node behavior cloning requires non-terminal target actions")
    return torch.tensor(
        [
            int(action.get("anchor_slot", action["binding_slots"][0]))
            for action in target_actions
        ],
        dtype=torch.long,
        device=device,
    )


def topk_hits(
    node_logits: torch.Tensor,
    targets: torch.Tensor,
    positives: list[list[tuple[int, tuple[int, ...]]]],
    node_ks: tuple[int, ...],
) -> tuple[dict[int, int], dict[int, int]]:
    available = min(max(node_ks), node_logits.shape[1])
    top_nodes = node_logits.topk(available, dim=1).indices
    target_hits = {}
    positive_hits = {}
    for node_k in node_ks:
        selected = top_nodes[:, : min(node_k, available)]
        target_hits[node_k] = int(selected.eq(targets.unsqueeze(1)).any(1).sum())
        states_with_positive = 0
        for batch_index, rows in enumerate(positives):
            anchors = {int(binding[0]) for _, binding in rows}
            chosen = set(map(int, selected[batch_index].tolist()))
            states_with_positive += bool(anchors & chosen)
        positive_hits[node_k] = states_with_positive
    return target_hits, positive_hits


def run_epoch(
    model,
    actor: HierarchicalPPOActorCritic,
    loader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    node_ks: tuple[int, ...],
    gradient_clip: float,
) -> NodeMetrics:
    training = optimizer is not None
    actor.train(training)
    total_states = 0
    total_loss = 0.0
    total_reduction = 0.0
    target_hit_totals = {node_k: 0 for node_k in node_ks}
    positive_hit_totals = {node_k: 0 for node_k in node_ks}

    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        states, live, _, prefix_states = encode_with_prefix_state(model, batch)
        state_features = build_state_features(states, live, live.sum(1))
        targets = target_anchors(batch["target_actions"], device)
        if not bool(live[torch.arange(live.shape[0], device=device), targets].all()):
            raise RuntimeError("a target action anchor is not live")

        with torch.set_grad_enabled(training), autocast_context(device):
            node_logits = actor.node_policy_logits(
                states.detach(),
                live,
                prefix_states.detach(),
                state_features.detach(),
            )
            losses = F.cross_entropy(node_logits.float(), targets, reduction="none")
            loss = (losses * batch["node_sample_weight"]).mean()
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.actor_parameters(), gradient_clip)
            optimizer.step()

        with torch.no_grad():
            target_hits, positive_hits = topk_hits(
                node_logits.float(), targets, batch["positives"], node_ks
            )
        batch_size = states.shape[0]
        total_states += batch_size
        total_loss += float(loss.item()) * batch_size
        total_reduction += float(batch["future_best_reduction"].sum().item())
        for node_k in node_ks:
            target_hit_totals[node_k] += target_hits[node_k]
            positive_hit_totals[node_k] += positive_hits[node_k]

    return NodeMetrics(
        states=total_states,
        loss=total_loss / max(1, total_states),
        mean_future_best_reduction=total_reduction / max(1, total_states),
        target_anchor_recall={
            str(node_k): target_hit_totals[node_k] / max(1, total_states)
            for node_k in node_ks
        },
        any_exact_anchor_recall={
            str(node_k): positive_hit_totals[node_k] / max(1, total_states)
            for node_k in node_ks
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--return-weight", type=float, default=0.25)
    parser.add_argument("--max-sample-weight", type=float, default=4.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--node-k", default="1,4,8,16,32")
    parser.add_argument("--max-train-states", type=int)
    parser.add_argument("--max-test-states", type=int)
    parser.add_argument(
        "--length-bucket-width",
        type=int,
        default=8,
        help="group training prefixes into this many adjacent lengths; zero disables",
    )
    parser.add_argument("--seed", type=int, default=940)
    args = parser.parse_args()
    node_ks = tuple(sorted({int(value) for value in args.node_k.split(",")}))
    if min(node_ks) <= 0:
        parser.error("--node-k values must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, rules, train_base, test_base = load_datasets(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(
        rules, len(payload["xfer_to_source"]), checkpoint["args"]
    ).to(device)
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
    model.eval()
    model.requires_grad_(False)

    hidden_size = args.hidden_size or model.width
    actor = HierarchicalPPOActorCritic(
        model.width, hidden_size=hidden_size
    ).to(device)
    optimizer = torch.optim.AdamW(
        actor.actor_parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    train_dataset = ReturnWeightedPrefixes(
        train_base,
        return_weight=args.return_weight,
        max_sample_weight=args.max_sample_weight,
    )
    test_dataset = ReturnWeightedPrefixes(
        test_base,
        return_weight=args.return_weight,
        max_sample_weight=args.max_sample_weight,
    )
    if args.max_train_states is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.max_train_states, len(train_dataset)))
        )
    if args.max_test_states is not None:
        test_dataset = Subset(
            test_dataset, range(min(args.max_test_states, len(test_dataset)))
        )
    collate = lambda samples: collate_node_samples(samples, rules)
    if args.length_bucket_width:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=LengthBucketBatchSampler(
                prefix_lengths(train_dataset),
                batch_size=args.batch_size,
                bucket_width=args.length_bucket_width,
                seed=args.seed,
            ),
            num_workers=0,
            collate_fn=collate,
        )
    else:
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collate,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    history = []
    started = time.perf_counter()
    initial_test = run_epoch(
        model,
        actor,
        test_loader,
        device,
        optimizer=None,
        node_ks=node_ks,
        gradient_clip=args.gradient_clip,
    )
    print(json.dumps({"epoch": 0, "test": asdict(initial_test)}, sort_keys=True))
    selection_k = str(
        max((value for value in node_ks if value <= 16), default=min(node_ks))
    )
    best_epoch = 0
    best_recall = initial_test.target_anchor_recall[selection_k]
    best_actor_state = copy.deepcopy(actor.state_dict())
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            actor,
            train_loader,
            device,
            optimizer=optimizer,
            node_ks=node_ks,
            gradient_clip=args.gradient_clip,
        )
        test_metrics = run_epoch(
            model,
            actor,
            test_loader,
            device,
            optimizer=None,
            node_ks=node_ks,
            gradient_clip=args.gradient_clip,
        )
        row = {
            "epoch": epoch,
            "train": asdict(train_metrics),
            "test": asdict(test_metrics),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        recall = test_metrics.target_anchor_recall[selection_k]
        if recall > best_recall:
            best_epoch = epoch
            best_recall = recall
            best_actor_state = copy.deepcopy(actor.state_dict())

    elapsed = time.perf_counter() - started
    result = {
        "format": "hierarchical-node-training-v1",
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
        },
        "args": vars(args),
        "width": model.width,
        "hidden_size": hidden_size,
        "node_ks": node_ks,
        "initial_test": asdict(initial_test),
        "history": history,
        "selection": {
            "metric": f"test_target_anchor_recall@{selection_k}",
            "best_epoch": best_epoch,
            "best_recall": best_recall,
        },
        "elapsed_seconds": elapsed,
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
