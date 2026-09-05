from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def aggregate_group_counts(steps: list[dict], key: str) -> dict[int, dict[str, int]]:
    aggregate: dict[int, dict[str, int]] = defaultdict(
        lambda: {"selected": 0, "parent_valid": 0, "legal": 0}
    )
    for step in steps:
        audit = step.get("proposal_legality_audit")
        if audit is None:
            continue
        for raw_id, row in audit[key].items():
            group_id = int(raw_id)
            for field in ("selected", "parent_valid", "legal"):
                aggregate[group_id][field] += int(row[field])
    return dict(aggregate)


def count_summary(row: dict[str, int]) -> dict:
    parent_valid = row["parent_valid"]
    legal = row["legal"]
    return {
        **row,
        "current_action_invalid": parent_valid - legal,
        "conditional_action_precision": legal / max(1, parent_valid),
    }


def rule_metadata(reference_data: Path | None) -> dict | None:
    if reference_data is None:
        return None
    import torch

    from dataset import RuleMetadata

    payload = torch.load(reference_data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    return {
        "rules": rules,
        "gate_deltas": tuple(
            len(rules.destination_gate_types[xfer_id])
            - len(rules.source_gate_types[rules.xfer_to_source[xfer_id]])
            for xfer_id in range(len(rules.xfer_to_source))
        ),
    }


def summarize_result(path: Path, metadata: dict | None, top_xfers: int) -> dict:
    payload = json.loads(path.read_text())
    xfer_counts = aggregate_group_counts(payload["steps"], "by_xfer")
    source_counts = aggregate_group_counts(payload["steps"], "by_source")

    xfer_rows = []
    for xfer_id, counts in xfer_counts.items():
        row = {"xfer_id": xfer_id, **count_summary(counts)}
        if metadata is not None:
            rules = metadata["rules"]
            row.update(
                {
                    "source_id": int(rules.xfer_to_source[xfer_id]),
                    "gate_delta": int(metadata["gate_deltas"][xfer_id]),
                    "source_pattern": rules.xfer_sources[xfer_id],
                    "destination_pattern": rules.xfer_destinations[xfer_id],
                }
            )
        xfer_rows.append(row)
    xfer_rows.sort(
        key=lambda row: (
            -row["current_action_invalid"],
            row["conditional_action_precision"],
            -row["selected"],
        )
    )

    source_rows = []
    for source_id, counts in source_counts.items():
        row = {"source_id": source_id, **count_summary(counts)}
        if metadata is not None:
            row["source_pattern"] = metadata["rules"].source_patterns[source_id]
        source_rows.append(row)
    source_rows.sort(
        key=lambda row: (
            -row["current_action_invalid"],
            row["conditional_action_precision"],
            -row["selected"],
        )
    )

    depth_rows = []
    for step in payload["steps"]:
        audit = step.get("proposal_legality_audit")
        if audit is None:
            continue
        depth_rows.append(
            {
                "step": int(step["step"]),
                "input_states": int(step["input_states"]),
                "exact_refresh_valid": int(step["exact_refresh_valid"]),
                "topn": audit["topn"],
            }
        )

    return {
        "result": str(path),
        "qasm": payload["qasm"],
        "proposal_ranking": payload["proposal_ranking"],
        "requested_depth": int(payload["requested_depth"]),
        "completed_depth": int(payload["completed_depth"]),
        "initial_gate_count": int(payload["initial_gate_count"]),
        "best_exact_gate_count": int(payload["best_exact_gate_count"]),
        "final_beam_size": int(payload["final_beam_size"]),
        "topn": payload["proposal_legality_audit"]["topn"],
        "by_depth": depth_rows,
        "worst_xfers": xfer_rows[:top_xfers],
        "worst_sources": source_rows[:top_xfers],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-data", type=Path)
    parser.add_argument("--top-xfers", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_xfers < 1:
        parser.error("--top-xfers must be positive")

    metadata = rule_metadata(args.reference_data)
    report = {
        "format": "proposal-topn-legality-summary-v1",
        "results": [
            summarize_result(path, metadata, args.top_xfers)
            for path in args.results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} results={len(report['results'])}")


if __name__ == "__main__":
    main()
