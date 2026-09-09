from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audit(payload: dict[str, Any]) -> int:
    audit_format = payload.get("format")
    if audit_format not in {
        "frozen_candidate_successor_descendant_v2",
        "frozen_candidate_successor_descendant_v3",
        "frozen_candidate_successor_descendant_v4",
    }:
        raise ValueError("audit does not contain descendant sibling labels")
    required = (
        "features",
        "outcomes",
        "sibling_group_ids",
        "parent_node_ids",
        "child_node_ids",
        "action_parent_ranks",
        "parent_expansion_rounds",
        "parent_stagnation_steps",
        "parent_action_depths",
    )
    rows = int(payload["features"].shape[0])
    for key in required[1:]:
        if payload[key].ndim != 1 or payload[key].numel() != rows:
            raise ValueError(f"unaligned sibling audit field: {key}")
    labels = payload.get("descendant_labels")
    if not isinstance(labels, dict):
        raise ValueError("audit is missing descendant_labels")
    if audit_format == "frozen_candidate_successor_descendant_v2":
        # V2 predates exposure accounting. It remains usable with the default
        # zero minimum, but cannot satisfy a positive expansion filter.
        labels.setdefault(
            "child_observed_expansions", torch.zeros(rows, dtype=torch.int16)
        )
        labels.setdefault(
            "child_attempted_actions", torch.zeros(rows, dtype=torch.int32)
        )
    for key in (
        "best_descendant_gate_counts",
        "continuation_gains",
        "parent_total_gains",
        "time_to_observed_best_descendant",
        "right_censored",
        "remaining_search_steps",
        "child_observed_expansions",
        "child_attempted_actions",
    ):
        if labels[key].ndim != 1 or labels[key].numel() != rows:
            raise ValueError(f"unaligned descendant label field: {key}")
    return rows


def collect_sibling_pairs(
    payload: dict[str, Any],
    *,
    min_remaining_steps: int,
    max_rejected_per_group: int,
    min_child_expansions: int = 0,
    max_child_expansion_gap: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = validate_audit(payload)
    groups: dict[int, list[int]] = defaultdict(list)
    child_ids = payload["child_node_ids"].tolist()
    outcomes = payload["outcomes"].tolist()
    remaining = payload["descendant_labels"]["remaining_search_steps"].tolist()
    expansions = payload["descendant_labels"][
        "child_observed_expansions"
    ].tolist()
    for row, (group, child_id, horizon) in enumerate(
        zip(payload["sibling_group_ids"].tolist(), child_ids, remaining)
    ):
        # Outcome 2 means this action created a unique child. Duplicate children
        # may already have been explored through another parent before this edge,
        # so their descendant label would leak future information into the row.
        if (
            int(outcomes[row]) == 2
            and int(child_id) >= 0
            and int(horizon) >= min_remaining_steps
            and int(expansions[row]) >= min_child_expansions
        ):
            groups[int(group)].append(row)

    best_gates = payload["descendant_labels"][
        "best_descendant_gate_counts"
    ].tolist()
    ranks = payload["action_parent_ranks"].tolist()
    continuation = payload["descendant_labels"]["continuation_gains"].tolist()
    delays = payload["descendant_labels"][
        "time_to_observed_best_descendant"
    ].tolist()
    pairs = []
    groups_considered = groups_with_signal = duplicate_child_rows = 0
    for group, group_rows in sorted(groups.items()):
        unique_children: dict[int, int] = {}
        for row in group_rows:
            child = int(child_ids[row])
            previous = unique_children.get(child)
            if previous is None or (int(ranks[row]), row) < (
                int(ranks[previous]),
                previous,
            ):
                if previous is not None:
                    duplicate_child_rows += 1
                unique_children[child] = row
            else:
                duplicate_child_rows += 1
        candidates = list(unique_children.values())
        if len(candidates) < 2:
            continue
        groups_considered += 1
        preferred = min(
            candidates,
            key=lambda row: (
                int(best_gates[row]),
                -int(continuation[row]),
                int(delays[row]) if int(delays[row]) >= 0 else 2**31 - 1,
                int(ranks[row]),
                row,
            ),
        )
        rejected = sorted(
            (
                row
                for row in candidates
                if int(best_gates[row]) > int(best_gates[preferred])
                and (
                    max_child_expansion_gap is None
                    or abs(int(expansions[row]) - int(expansions[preferred]))
                    <= max_child_expansion_gap
                )
            ),
            key=lambda row: (
                -int(best_gates[row]),
                int(ranks[row]),
                row,
            ),
        )[:max_rejected_per_group]
        if not rejected:
            continue
        groups_with_signal += 1
        for rejected_row in rejected:
            pairs.append(
                {
                    "sibling_group_id": group,
                    "parent_node_id": int(payload["parent_node_ids"][preferred]),
                    "preferred_row": preferred,
                    "rejected_row": rejected_row,
                    "preferred_child_node_id": int(child_ids[preferred]),
                    "rejected_child_node_id": int(child_ids[rejected_row]),
                    "preferred_best_descendant_gate": int(best_gates[preferred]),
                    "rejected_best_descendant_gate": int(best_gates[rejected_row]),
                    "advantage": (
                        int(best_gates[rejected_row]) - int(best_gates[preferred])
                    ),
                    "preferred_parent_rank": int(ranks[preferred]),
                    "rejected_parent_rank": int(ranks[rejected_row]),
                    "preferred_remaining_steps": int(remaining[preferred]),
                    "rejected_remaining_steps": int(remaining[rejected_row]),
                    "preferred_child_expansions": int(expansions[preferred]),
                    "rejected_child_expansions": int(expansions[rejected_row]),
                }
            )
    return pairs, {
        "rows": rows,
        "eligible_groups": len(groups),
        "groups_considered": groups_considered,
        "groups_with_signal": groups_with_signal,
        "pairs": len(pairs),
        "duplicate_child_rows_removed": duplicate_child_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pairwise same-parent continuation preferences."
    )
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-audits", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-remaining-steps", type=int, default=16)
    parser.add_argument("--max-rejected-per-group", type=int, default=8)
    parser.add_argument("--min-child-expansions", type=int, default=0)
    parser.add_argument("--max-child-expansion-gap", type=int)
    args = parser.parse_args()
    if args.min_remaining_steps < 1:
        parser.error("--min-remaining-steps must be positive")
    if args.max_rejected_per_group < 1:
        parser.error("--max-rejected-per-group must be positive")
    if args.min_child_expansions < 0:
        parser.error("--min-child-expansions must be nonnegative")
    if args.max_child_expansion_gap is not None and args.max_child_expansion_gap < 0:
        parser.error("--max-child-expansion-gap must be nonnegative")
    validation = {path.resolve() for path in args.validation_audits}
    paths = []
    seen = set()
    for path in [*args.audits, *args.validation_audits]:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            paths.append(path)

    sources = []
    train_pairs = []
    test_pairs = []
    for source_id, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        pairs, stats = collect_sibling_pairs(
            payload,
            min_remaining_steps=args.min_remaining_steps,
            max_rejected_per_group=args.max_rejected_per_group,
            min_child_expansions=args.min_child_expansions,
            max_child_expansion_gap=args.max_child_expansion_gap,
        )
        for pair in pairs:
            pair["source_id"] = source_id
        split = "test" if path.resolve() in validation else "train"
        (test_pairs if split == "test" else train_pairs).extend(pairs)
        sources.append(
            {
                "source_id": source_id,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "split": split,
                **stats,
            }
        )
    manifest = {
        "format": "frozen-sibling-continuation-preference-manifest-v1",
        "sources": sources,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "metadata": {
            "min_remaining_steps": args.min_remaining_steps,
            "max_rejected_per_group": args.max_rejected_per_group,
            "min_child_expansions": args.min_child_expansions,
            "max_child_expansion_gap": args.max_child_expansion_gap,
            "train_pairs": len(train_pairs),
            "test_pairs": len(test_pairs),
            "pair_objective": "lower observed best descendant gate within sibling group",
            "censoring": (
                "zero observed continuation gain remains right-censored in source audit"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, args.output)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {"sources": sources, **manifest["metadata"]}, indent=2, sort_keys=True
        )
        + "\n"
    )
    print(json.dumps(manifest["metadata"], indent=2, sort_keys=True))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
