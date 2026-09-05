from __future__ import annotations

import argparse
from pathlib import Path

from dataset import collate_prefixes, load_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    _, rules, train, test = load_datasets(args.data)
    samples = [
        train[0],
        train[min(7, len(train) - 1)],
        test[0] if len(test) else train[min(31, len(train) - 1)],
    ]
    batch = collate_prefixes(samples, rules)
    assert batch["initial_types"].shape[0] == len(samples)
    assert len(batch["positives"]) == len(samples)
    assert batch["target_actions"] == [sample["target_action"] for sample in samples]
    for sample, rows in zip(samples, batch["positives"]):
        assert len(rows) == len(sample["matches"])
        assert all(binding[0] == match["anchor_slot"] for (_, binding), match in zip(rows, sample["matches"]))
    print(
        f"dataset ok: train={len(train)} test={len(test)} "
        f"sources={len(rules.source_gate_types)} batch_slots={batch['initial_types'].shape[1]}"
    )


if __name__ == "__main__":
    main()
