from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import (
    _local_streak_bucket,
    collate_current_graphs,
    collate_prefixes,
    load_datasets,
)
from model_factory import build_model
from train import autocast_context, move_batch


def distance_bucket(distance: int | None) -> str:
    if distance is None:
        return "unreachable"
    if distance <= 2:
        return str(distance)
    return "3_plus"


def prefix_bucket(prefix: int) -> str:
    if prefix <= 15:
        return "0_15"
    if prefix <= 31:
        return "16_31"
    if prefix <= 63:
        return "32_63"
    return "64_plus"


def streak_buckets(streak: int) -> tuple[str, ...]:
    if streak == 0:
        return ("0",)
    if streak == 1:
        return ("1",)
    if streak < 4:
        return ("2_3", "ge_2")
    return ("4_plus", "ge_2", "ge_4")


def graph_distances(
    live_slots: set[int], edges: list[tuple[int, int, int, int]], core: set[int]
) -> dict[int, int]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for src, dst, _, _ in edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    distances = {slot: 0 for slot in core if slot in live_slots}
    queue = deque(distances)
    while queue:
        slot = queue.popleft()
        for neighbor in adjacency[slot]:
            if neighbor not in distances:
                distances[neighbor] = distances[slot] + 1
                queue.append(neighbor)
    return distances


def affected_core(
    live_slots: set[int], previous_action: dict, previous_delta: dict
) -> tuple[set[int], set[int]]:
    new_nodes = {
        int(slot)
        for slot in previous_action.get("dst_slots", ())
        if int(slot) in live_slots
    }
    core = set(new_nodes)
    for edge in previous_delta["added_edges"] + previous_delta["removed_edges"]:
        src, dst = int(edge[0]), int(edge[1])
        if src in live_slots:
            core.add(src)
        if dst in live_slots:
            core.add(dst)
    for slot, _, _ in previous_delta["added_nodes"]:
        if int(slot) in live_slots:
            core.add(int(slot))
    return core, new_nodes


def empty_counter():
    return {"true": 0, "predicted": 0, "hits": 0}


def finalize(groups: dict[str, dict[str, int]]) -> dict:
    result = {}
    for key, counts in sorted(groups.items()):
        row = dict(counts)
        row["recall"] = counts["hits"] / counts["true"] if counts["true"] else None
        row["precision"] = (
            counts["hits"] / counts["predicted"] if counts["predicted"] else None
        )
        result[key] = row
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, rules, _, test_dataset = load_datasets(
        args.data, include_terminal=True
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    model = build_model(rules, len(payload["xfer_to_source"]), train_args).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing_parameters = [
        name
        for name in incompatible.missing_keys
        if name in dict(model.named_parameters())
    ]
    if incompatible.unexpected_keys or missing_parameters:
        raise RuntimeError(
            f"incompatible checkpoint: missing={missing_parameters}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda samples: (
            collate_prefixes(samples, rules)
            if train_args.get("architecture") == "paged_action"
            else collate_current_graphs(samples, rules)
        ),
        num_workers=0,
    )

    binding_distance = defaultdict(empty_counter)
    anchor_distance = defaultdict(empty_counter)
    binding_near_by_prefix = defaultdict(empty_counter)
    binding_near_by_streak = defaultdict(empty_counter)
    overlap = defaultdict(empty_counter)
    processed_states = 0
    skipped_initial_states = 0
    streak_label_disagreements = 0
    source_lengths = model.source_lengths.tolist()

    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with autocast_context(device):
            states, live, gate_types = model.encode(batch)
            logits, eligible = model.match_logits(states, live, gate_types)

        candidate_batch = []
        candidate_source = []
        candidate_anchor = []
        offsets = [0]
        for sample_index, rows in enumerate(batch["positives"]):
            count = min(
                args.candidate_multiplier * len(rows),
                int(eligible[sample_index].sum()),
            )
            indices = logits[sample_index].flatten().topk(count).indices
            candidate_batch.append(
                torch.full((count,), sample_index, dtype=torch.long, device=device)
            )
            candidate_anchor.append(indices // model.num_sources)
            candidate_source.append(indices % model.num_sources)
            offsets.append(offsets[-1] + count)
        all_batch = torch.cat(candidate_batch)
        all_sources = torch.cat(candidate_source)
        all_anchors = torch.cat(candidate_anchor)
        bindings, valid = model.structural_decode(
            batch, gate_types, live, all_batch, all_sources, all_anchors
        )
        sources_cpu = all_sources.tolist()
        bindings_cpu = bindings.tolist()
        valid_cpu = valid.tolist()

        edge_batch = cpu_batch["current_edge_batch"].tolist()
        edge_src = cpu_batch["current_edge_src"].tolist()
        edge_dst = cpu_batch["current_edge_dst"].tolist()
        edge_rel = cpu_batch["current_edge_relation"].tolist()
        edges_by_sample: list[list[tuple[int, int, int, int]]] = [
            [] for _ in batch["positives"]
        ]
        for batch_id, src, dst, relation in zip(
            edge_batch, edge_src, edge_dst, edge_rel
        ):
            edges_by_sample[batch_id].append(
                (src, dst, relation // 4, relation % 4)
            )

        current_types = cpu_batch["current_types"]
        for sample_index, true_rows in enumerate(batch["positives"]):
            previous_action = batch["previous_action"][sample_index]
            previous_delta = batch["previous_delta"][sample_index]
            if previous_action is None:
                skipped_initial_states += 1
                continue
            processed_states += 1
            live_slots = set(
                current_types[sample_index].ge(0).nonzero(as_tuple=False).squeeze(1).tolist()
            )
            core, new_nodes = affected_core(
                live_slots, previous_action, previous_delta
            )
            derived_core = set(
                cpu_batch["current_rewrite_distance"][sample_index]
                .eq(0)
                .nonzero(as_tuple=False)
                .squeeze(1)
                .tolist()
            )
            if derived_core != core:
                raise RuntimeError(
                    "s0+actions locality core differs from the Quartz audit delta"
                )
            expected_streak_bucket = _local_streak_bucket(
                True, int(batch["previous_local_streak"][sample_index])
            )
            if int(cpu_batch["current_local_streak"][sample_index]) != expected_streak_bucket:
                streak_label_disagreements += 1
            distances = graph_distances(
                live_slots, edges_by_sample[sample_index], core
            )
            begin, end = offsets[sample_index : sample_index + 2]
            predicted = []
            for row_index in range(begin, end):
                if not valid_cpu[row_index] or len(predicted) >= len(true_rows):
                    continue
                source_id = sources_cpu[row_index]
                length = source_lengths[source_id]
                predicted.append(
                    (source_id, tuple(bindings_cpu[row_index][:length]))
                )
            predicted_set = set(predicted)
            true_set = {
                (int(source), tuple(map(int, binding)))
                for source, binding in true_rows
            }
            prefix = int(batch["prefix_length"][sample_index])
            prefix_name = prefix_bucket(prefix)
            previous_streak = int(batch["previous_local_streak"][sample_index])
            current_streak_buckets = streak_buckets(previous_streak)

            for source_id, binding in true_set:
                hit = int((source_id, binding) in predicted_set)
                binding_d = min(
                    (
                        distance
                        for distance in (distances.get(slot) for slot in binding)
                        if distance is not None
                    ),
                    default=None,
                )
                anchor_d = distances.get(binding[0])
                binding_distance[distance_bucket(binding_d)]["true"] += 1
                binding_distance[distance_bucket(binding_d)]["hits"] += hit
                anchor_distance[distance_bucket(anchor_d)]["true"] += 1
                anchor_distance[distance_bucket(anchor_d)]["hits"] += hit
                if binding_d is not None and binding_d <= 2:
                    binding_near_by_prefix[prefix_name]["true"] += 1
                    binding_near_by_prefix[prefix_name]["hits"] += hit
                    for streak_name in current_streak_buckets:
                        binding_near_by_streak[streak_name]["true"] += 1
                        binding_near_by_streak[streak_name]["hits"] += hit
                if set(binding) & core:
                    overlap["affected_core"]["true"] += 1
                    overlap["affected_core"]["hits"] += hit
                if set(binding) & new_nodes:
                    overlap["new_nodes"]["true"] += 1
                    overlap["new_nodes"]["hits"] += hit

            for source_id, binding in predicted_set:
                binding_d = min(
                    (
                        distance
                        for distance in (distances.get(slot) for slot in binding)
                        if distance is not None
                    ),
                    default=None,
                )
                anchor_d = distances.get(binding[0])
                binding_distance[distance_bucket(binding_d)]["predicted"] += 1
                anchor_distance[distance_bucket(anchor_d)]["predicted"] += 1
                if binding_d is not None and binding_d <= 2:
                    binding_near_by_prefix[prefix_name]["predicted"] += 1
                    for streak_name in current_streak_buckets:
                        binding_near_by_streak[streak_name]["predicted"] += 1
                if set(binding) & core:
                    overlap["affected_core"]["predicted"] += 1
                if set(binding) & new_nodes:
                    overlap["new_nodes"]["predicted"] += 1

    result = {
        "candidate_multiplier": args.candidate_multiplier,
        "processed_states": processed_states,
        "skipped_initial_states": skipped_initial_states,
        "derived_streak_vs_collector_label_disagreements": (
            streak_label_disagreements
        ),
        "affected_core_definition": (
            "live destination slots plus live endpoints of the previous rewrite's changed edges"
        ),
        "binding_distance": finalize(binding_distance),
        "anchor_distance": finalize(anchor_distance),
        "binding_within_2_by_prefix": finalize(binding_near_by_prefix),
        "binding_within_2_by_previous_local_streak": finalize(
            binding_near_by_streak
        ),
        "overlap": finalize(overlap),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
