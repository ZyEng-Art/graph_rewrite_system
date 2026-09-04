from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import collate_prefixes, load_datasets
from model import S0ActionBindingModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--include-terminal", action="store_true")
    args = parser.parse_args()
    payload, rules, train, test = load_datasets(
        args.data, include_terminal=args.include_terminal
    )
    if args.split == "train":
        dataset = train
    elif args.split == "test":
        dataset = test
    else:
        dataset = torch.utils.data.ConcatDataset((train, test))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda samples: collate_prefixes(samples, rules),
    )
    model = S0ActionBindingModel(
        rules,
        num_xfers=len(payload["xfer_to_source"]),
        width=32,
        retrieval_width=16,
        graph_layers=1,
        current_graph_layers=1,
    )
    correct = 0
    total = 0
    valid_count = 0
    failures = Counter()
    for batch in loader:
        batch_ids = []
        source_ids = []
        anchor_slots = []
        expected = []
        for batch_index, rows in enumerate(batch["positives"]):
            for source_id, binding in rows:
                batch_ids.append(batch_index)
                source_ids.append(source_id)
                anchor_slots.append(binding[0])
                expected.append(tuple(map(int, binding)))
        bindings, valid = model.structural_decode(
            batch,
            batch["current_types"],
            batch["current_types"].ge(0),
            torch.tensor(batch_ids),
            torch.tensor(source_ids),
            torch.tensor(anchor_slots),
        )
        for row_index, (source_id, target) in enumerate(zip(source_ids, expected)):
            length = int(model.source_lengths[source_id])
            predicted = tuple(map(int, bindings[row_index, :length].tolist()))
            correct += int(bool(valid[row_index]) and predicted == target)
            if not (bool(valid[row_index]) and predicted == target):
                failures[source_id] += 1
            valid_count += int(valid[row_index])
            total += 1
    print(
        f"tensor structural binding: correct={correct}/{total} "
        f"({correct / total:.4%}) valid={valid_count}"
    )
    for source_id, count in failures.most_common(10):
        print(
            source_id,
            count,
            rules.source_patterns[source_id],
            model.binding_child[source_id].tolist(),
            model.binding_parent[source_id].tolist(),
        )


if __name__ == "__main__":
    main()
