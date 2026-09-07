from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AuditRows:
    path: Path
    features: torch.Tensor
    outcomes: torch.Tensor
    groups: torch.Tensor
    xfer_ids: torch.Tensor
    source_ids: torch.Tensor
    probabilities: torch.Tensor
    gate_deltas: torch.Tensor
    parent_gate_counts: torch.Tensor
    steps: torch.Tensor

    @property
    def inputs(self) -> torch.Tensor:
        auxiliary = torch.stack(
            (
                self.probabilities.float(),
                self.gate_deltas.float() / 8.0,
                self.parent_gate_counts.float() / 512.0,
                self.steps.float() / 64.0,
            ),
            dim=1,
        )
        return torch.cat((self.features.float(), auxiliary), dim=1)


def load_audit(path: Path, *, group_offset: int = 0) -> AuditRows:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "frozen_candidate_successor_v1":
        raise ValueError(f"unsupported audit payload: {path}")
    outcomes = payload["outcomes"].long()
    groups = payload["successor_groups"].long()
    valid_groups = groups.ge(0)
    groups = groups.clone()
    groups[valid_groups] += group_offset
    row_count = int(payload["features"].shape[0])
    fields = (
        outcomes,
        groups,
        payload["xfer_ids"],
        payload["source_ids"],
        payload["probabilities"],
        payload["gate_deltas"],
        payload["parent_gate_counts"],
        payload["steps"],
    )
    if any(int(field.shape[0]) != row_count for field in fields):
        raise ValueError(f"misaligned audit columns: {path}")
    return AuditRows(
        path=path,
        features=payload["features"].half(),
        outcomes=outcomes,
        groups=groups,
        xfer_ids=payload["xfer_ids"].long(),
        source_ids=payload["source_ids"].long(),
        probabilities=payload["probabilities"].half(),
        gate_deltas=payload["gate_deltas"].short(),
        parent_gate_counts=payload["parent_gate_counts"].short(),
        steps=payload["steps"].short(),
    )


def load_many(paths: list[Path]) -> list[AuditRows]:
    rows = []
    group_offset = 0
    for path in paths:
        audit = load_audit(path, group_offset=group_offset)
        rows.append(audit)
        maximum = audit.groups.max().item()
        if maximum >= 0:
            group_offset = int(maximum) + 1
    return rows


def concatenate(rows: list[AuditRows]) -> AuditRows:
    if not rows:
        raise ValueError("cannot concatenate an empty audit list")
    return AuditRows(
        path=Path("<concatenated>"),
        features=torch.cat([row.features for row in rows]),
        outcomes=torch.cat([row.outcomes for row in rows]),
        groups=torch.cat([row.groups for row in rows]),
        xfer_ids=torch.cat([row.xfer_ids for row in rows]),
        source_ids=torch.cat([row.source_ids for row in rows]),
        probabilities=torch.cat([row.probabilities for row in rows]),
        gate_deltas=torch.cat([row.gate_deltas for row in rows]),
        parent_gate_counts=torch.cat(
            [row.parent_gate_counts for row in rows]
        ),
        steps=torch.cat([row.steps for row in rows]),
    )


class NeuralSuccessorPrefilter(nn.Module):
    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        embedding_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_width = input_width
        self.hidden_width = hidden_width
        self.embedding_width = embedding_width
        self.input_norm = nn.LayerNorm(input_width)
        self.trunk = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
        )
        self.valid_head = nn.Linear(hidden_width, 1)
        self.duplicate_head = nn.Linear(hidden_width, 1)
        self.successor_projection = nn.Linear(hidden_width, embedding_width)
        self.pair_scale_log = nn.Parameter(torch.tensor(math.log(10.0)))
        self.pair_bias = nn.Parameter(torch.tensor(-5.0))

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(self.input_norm(inputs))
        valid = self.valid_head(hidden).squeeze(-1)
        duplicate = self.duplicate_head(hidden).squeeze(-1)
        embedding = F.normalize(self.successor_projection(hidden), dim=-1)
        return valid, duplicate, embedding

    def pair_logits(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        scale = self.pair_scale_log.exp().clamp(max=100.0)
        return (left * right).sum(-1) * scale + self.pair_bias


def balanced_weights(labels: torch.Tensor) -> torch.Tensor:
    positive = labels.sum().clamp_min(1)
    negative = (labels.numel() - labels.sum()).clamp_min(1)
    return torch.where(
        labels.bool(),
        labels.numel() / (2.0 * positive),
        labels.numel() / (2.0 * negative),
    )


def build_pairs(
    rows: AuditRows, *, max_pairs: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_indices = torch.where(rows.outcomes.gt(0))[0].tolist()
    by_group: dict[int, list[int]] = {}
    by_xfer: dict[int, list[int]] = {}
    for index in valid_indices:
        group = int(rows.groups[index])
        by_group.setdefault(group, []).append(index)
        by_xfer.setdefault(int(rows.xfer_ids[index]), []).append(index)
    positives = []
    for indices in by_group.values():
        if len(indices) < 2:
            continue
        representative = indices[0]
        positives.extend((representative, index) for index in indices[1:])
    generator = random.Random(seed)
    generator.shuffle(positives)
    if max_pairs:
        positives = positives[:max_pairs]
    left = []
    right = []
    labels = []
    for representative, duplicate in positives:
        left.append(representative)
        right.append(duplicate)
        labels.append(1.0)
        pool = by_xfer.get(int(rows.xfer_ids[duplicate]), valid_indices)
        negative = None
        for _ in range(16):
            candidate = pool[generator.randrange(len(pool))]
            if rows.groups[candidate] != rows.groups[duplicate]:
                negative = candidate
                break
        while negative is None:
            candidate = valid_indices[generator.randrange(len(valid_indices))]
            if rows.groups[candidate] != rows.groups[duplicate]:
                negative = candidate
        left.append(duplicate)
        right.append(negative)
        labels.append(0.0)
    if not left:
        raise ValueError("audit data contains no repeated successor groups")
    return (
        torch.tensor(left, dtype=torch.long),
        torch.tensor(right, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def threshold_for_false_positive_rate(
    safe_scores: torch.Tensor,
    *,
    false_positive_rate: float,
    low_is_positive: bool,
) -> float:
    if not 0 <= false_positive_rate <= 1:
        raise ValueError("false-positive rate must lie in [0, 1]")
    scores = safe_scores.sort(descending=not low_is_positive).values
    allowed = min(
        scores.numel() - 1,
        math.floor(false_positive_rate * scores.numel()),
    )
    if allowed <= 0:
        boundary = scores[0]
        direction = -math.inf if low_is_positive else math.inf
        return math.nextafter(float(boundary), direction)
    return float(scores[allowed - 1])


@torch.no_grad()
def encode_rows(
    model: NeuralSuccessorPrefilter,
    rows: AuditRows,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = rows.inputs
    valid_logits = []
    duplicate_logits = []
    embeddings = []
    model.eval()
    for begin in range(0, inputs.shape[0], batch_size):
        selected = inputs[begin : begin + batch_size].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            valid, duplicate, embedding = model(selected)
        valid_logits.append(valid.float().cpu())
        duplicate_logits.append(duplicate.float().cpu())
        embeddings.append(embedding.float().cpu())
    return (
        torch.cat(valid_logits),
        torch.cat(duplicate_logits),
        torch.cat(embeddings),
    )


def binary_filter_metrics(
    positive_scores: torch.Tensor,
    positive_mask: torch.Tensor,
    safe_mask: torch.Tensor,
    threshold: float,
    *,
    low_is_positive: bool,
) -> dict[str, float | int]:
    predicted = (
        positive_scores.lt(threshold)
        if low_is_positive
        else positive_scores.gt(threshold)
    )
    true_positive = int((predicted & positive_mask).sum())
    false_positive = int((predicted & safe_mask).sum())
    return {
        "threshold": threshold,
        "positive_examples": int(positive_mask.sum()),
        "safe_examples": int(safe_mask.sum()),
        "filtered_positives": true_positive,
        "filtered_safe": false_positive,
        "positive_recall": true_positive / max(1, int(positive_mask.sum())),
        "safe_false_positive_rate": false_positive
        / max(1, int(safe_mask.sum())),
    }


@torch.no_grad()
def retrieval_rows(
    embeddings: torch.Tensor,
    rows: AuditRows,
    device: torch.device,
    query_batch_size: int,
) -> dict[str, torch.Tensor | int]:
    valid_indices = torch.where(rows.outcomes.gt(0))[0]
    first_by_group: dict[int, int] = {}
    for index in valid_indices.tolist():
        first_by_group.setdefault(int(rows.groups[index]), index)
    representative_indices = torch.tensor(
        list(first_by_group.values()), dtype=torch.long
    )
    representative_groups = rows.groups[representative_indices]
    representative_positions = representative_indices
    representative_embeddings = embeddings[representative_indices].to(device)
    nearest_scores = []
    nearest_groups = []
    query_positions = []
    query_groups = []
    query_outcomes = []
    for begin in range(0, valid_indices.numel(), query_batch_size):
        indices = valid_indices[begin : begin + query_batch_size]
        query = embeddings[indices].to(device)
        similarities = query @ representative_embeddings.T
        earlier = representative_positions.unsqueeze(0).to(device) < (
            indices.unsqueeze(1).to(device)
        )
        similarities.masked_fill_(~earlier, -torch.inf)
        scores, columns = similarities.max(dim=1)
        nearest_scores.append(scores.cpu())
        nearest_groups.append(representative_groups[columns.cpu()])
        query_positions.append(indices)
        query_groups.append(rows.groups[indices])
        query_outcomes.append(rows.outcomes[indices])
    scores = torch.cat(nearest_scores)
    predicted_groups = torch.cat(nearest_groups)
    groups = torch.cat(query_groups)
    outcomes = torch.cat(query_outcomes)
    positions = torch.cat(query_positions)
    has_prior_representative = scores.isfinite()
    return {
        "scores": scores,
        "predicted_groups": predicted_groups,
        "groups": groups,
        "outcomes": outcomes,
        "positions": positions,
        "has_prior_representative": has_prior_representative,
        "representatives": representative_indices.numel(),
    }


def evaluate(
    model: NeuralSuccessorPrefilter,
    rows_list: list[AuditRows],
    device: torch.device,
    batch_size: int,
    *,
    thresholds: dict[str, float] | None = None,
    false_positive_rate: float = 0.0,
) -> tuple[dict, dict[str, float]]:
    encoded = [encode_rows(model, rows, device, batch_size) for rows in rows_list]
    outcomes = torch.cat([rows.outcomes for rows in rows_list])
    valid_scores = torch.cat([item[0].sigmoid() for item in encoded])
    duplicate_scores = torch.cat([item[1].sigmoid() for item in encoded])
    if thresholds is None:
        valid_threshold = threshold_for_false_positive_rate(
            valid_scores[outcomes.gt(0)],
            false_positive_rate=false_positive_rate,
            low_is_positive=True,
        )
        duplicate_threshold = threshold_for_false_positive_rate(
            duplicate_scores[outcomes.eq(2)],
            false_positive_rate=false_positive_rate,
            low_is_positive=False,
        )
    else:
        valid_threshold = thresholds["valid"]
        duplicate_threshold = thresholds["duplicate"]
    result = {
        "rows": int(outcomes.numel()),
        "invalid_filter": binary_filter_metrics(
            valid_scores,
            outcomes.eq(0),
            outcomes.gt(0),
            valid_threshold,
            low_is_positive=True,
        ),
        "duplicate_propensity_filter": binary_filter_metrics(
            duplicate_scores,
            outcomes.eq(1),
            outcomes.eq(2),
            duplicate_threshold,
            low_is_positive=False,
        ),
        "circuits": [],
    }
    retrieval_safe_scores = []
    retrieval_duplicate_scores = []
    retrieval_correct = 0
    retrieval_duplicates = 0
    retrieval_rows_by_circuit = []
    for rows, (_, _, embeddings) in zip(rows_list, encoded):
        retrieval = retrieval_rows(
            embeddings, rows, device, max(256, batch_size // 4)
        )
        has_prior = retrieval["has_prior_representative"]
        query_outcomes = retrieval["outcomes"]
        scores = retrieval["scores"]
        duplicate = query_outcomes.eq(1) & has_prior
        unique = query_outcomes.eq(2) & has_prior
        correct = retrieval["predicted_groups"].eq(retrieval["groups"])
        retrieval_safe_scores.append(scores[unique])
        retrieval_duplicate_scores.append(scores[duplicate])
        retrieval_correct += int((correct & duplicate).sum())
        retrieval_duplicates += int(duplicate.sum())
        retrieval_rows_by_circuit.append(
            {
                "path": str(rows.path),
                "rows": int(rows.outcomes.numel()),
                "representatives": int(retrieval["representatives"]),
                "retrievable_duplicates": int(duplicate.sum()),
                "nearest_group_recall": int((correct & duplicate).sum())
                / max(1, int(duplicate.sum())),
            }
        )
    safe_scores = torch.cat(retrieval_safe_scores)
    duplicate_retrieval_scores = torch.cat(retrieval_duplicate_scores)
    if thresholds is None:
        retrieval_threshold = threshold_for_false_positive_rate(
            safe_scores,
            false_positive_rate=false_positive_rate,
            low_is_positive=False,
        )
    else:
        retrieval_threshold = thresholds["retrieval"]
    result["retrieval_filter"] = binary_filter_metrics(
        torch.cat((duplicate_retrieval_scores, safe_scores)),
        torch.cat(
            (
                torch.ones_like(duplicate_retrieval_scores, dtype=torch.bool),
                torch.zeros_like(safe_scores, dtype=torch.bool),
            )
        ),
        torch.cat(
            (
                torch.zeros_like(
                    duplicate_retrieval_scores, dtype=torch.bool
                ),
                torch.ones_like(safe_scores, dtype=torch.bool),
            )
        ),
        retrieval_threshold,
        low_is_positive=False,
    )
    result["retrieval_filter"]["nearest_group_recall"] = (
        retrieval_correct / max(1, retrieval_duplicates)
    )
    result["circuits"] = retrieval_rows_by_circuit
    return result, {
        "valid": valid_threshold,
        "duplicate": duplicate_threshold,
        "retrieval": retrieval_threshold,
    }


def train(args) -> tuple[NeuralSuccessorPrefilter, dict]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows_list = load_many(args.train_data)
    calibration_rows = load_many(args.calibration_data)
    test_rows = load_many(args.test_data)
    train_rows = concatenate(train_rows_list)
    train_inputs = train_rows.inputs
    input_width = int(train_inputs.shape[1])
    model = NeuralSuccessorPrefilter(
        input_width,
        args.hidden_width,
        args.embedding_width,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    pair_left, pair_right, pair_labels = build_pairs(
        train_rows, max_pairs=args.max_pairs, seed=args.seed
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        row_order = torch.randperm(train_inputs.shape[0])
        pair_order = torch.randperm(pair_labels.shape[0])
        steps = max(
            math.ceil(row_order.numel() / args.batch_size),
            math.ceil(pair_order.numel() / args.pair_batch_size),
        )
        totals = {"loss": 0.0, "valid": 0.0, "duplicate": 0.0, "pair": 0.0}
        started = time.perf_counter()
        for step in range(steps):
            row_begin = (step * args.batch_size) % row_order.numel()
            row_indices = row_order[row_begin : row_begin + args.batch_size]
            if row_indices.numel() < args.batch_size:
                row_indices = torch.cat(
                    (row_indices, row_order[: args.batch_size - row_indices.numel()])
                )
            pair_begin = (step * args.pair_batch_size) % pair_order.numel()
            pair_indices = pair_order[
                pair_begin : pair_begin + args.pair_batch_size
            ]
            if pair_indices.numel() < args.pair_batch_size:
                pair_indices = torch.cat(
                    (
                        pair_indices,
                        pair_order[: args.pair_batch_size - pair_indices.numel()],
                    )
                )
            selected_inputs = train_inputs[row_indices].to(device)
            row_outcomes = train_rows.outcomes[row_indices].to(device)
            left_inputs = train_inputs[pair_left[pair_indices]].to(device)
            right_inputs = train_inputs[pair_right[pair_indices]].to(device)
            labels = pair_labels[pair_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                valid_logits, duplicate_logits, _ = model(selected_inputs)
                valid_targets = row_outcomes.gt(0).float()
                valid_loss = F.binary_cross_entropy_with_logits(
                    valid_logits,
                    valid_targets,
                    weight=balanced_weights(valid_targets),
                )
                valid_rows = row_outcomes.gt(0)
                duplicate_targets = row_outcomes[valid_rows].eq(1).float()
                duplicate_loss = F.binary_cross_entropy_with_logits(
                    duplicate_logits[valid_rows],
                    duplicate_targets,
                    weight=balanced_weights(duplicate_targets),
                )
                _, _, left_embeddings = model(left_inputs)
                _, _, right_embeddings = model(right_inputs)
                pair_loss = F.binary_cross_entropy_with_logits(
                    model.pair_logits(left_embeddings, right_embeddings),
                    labels,
                )
                loss = (
                    args.valid_weight * valid_loss
                    + args.duplicate_weight * duplicate_loss
                    + args.pair_weight * pair_loss
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["valid"] += float(valid_loss.detach())
            totals["duplicate"] += float(duplicate_loss.detach())
            totals["pair"] += float(pair_loss.detach())
        epoch_row = {
            "epoch": epoch + 1,
            **{name: value / steps for name, value in totals.items()},
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)
    calibration_metrics, thresholds = evaluate(
        model,
        calibration_rows,
        device,
        args.eval_batch_size,
        false_positive_rate=args.calibration_false_positive_rate,
    )
    test_metrics, _ = evaluate(
        model,
        test_rows,
        device,
        args.eval_batch_size,
        thresholds=thresholds,
    )
    metrics = {
        "device": str(device),
        "train_rows": int(train_rows.outcomes.numel()),
        "train_invalid": int(train_rows.outcomes.eq(0).sum()),
        "train_duplicates": int(train_rows.outcomes.eq(1).sum()),
        "train_unique": int(train_rows.outcomes.eq(2).sum()),
        "train_pairs": int(pair_labels.numel()),
        "thresholds": thresholds,
        "calibration_false_positive_rate": (
            args.calibration_false_positive_rate
        ),
        "history": history,
        "calibration": calibration_metrics,
        "test": test_metrics,
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--calibration-data", type=Path, nargs="+", required=True
    )
    parser.add_argument("--test-data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pair-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--embedding-width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--valid-weight", type=float, default=1.0)
    parser.add_argument("--duplicate-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-pairs", type=int, default=500000)
    parser.add_argument(
        "--calibration-false-positive-rate", type=float, default=0.0
    )
    parser.add_argument("--seed", type=int, default=907)
    args = parser.parse_args()
    model, metrics = train(args)
    checkpoint = {
        "format": "neural_successor_prefilter_v1",
        "args": vars(args),
        "model": model.state_dict(),
        "input_width": model.input_width,
        "hidden_width": model.hidden_width,
        "embedding_width": model.embedding_width,
        "thresholds": metrics["thresholds"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
