from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--caps", default="128,256,512,1024,2048")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    caps = [int(value) for value in args.caps.split(",")]
    payload = json.loads(args.audit.read_text(encoding="utf-8"))
    end = args.start + args.length
    rows = [
        row
        for row in payload["actions"]
        if args.start <= int(row["step"]) < end
    ]
    ranks = sorted(
        int(row["gate_rank"])
        for row in rows
        if row.get("gate_rank") is not None
    )
    result = {
        "format": "candidate-audit-window-v1",
        "start": args.start,
        "end_exclusive": end,
        "actions": len(rows),
        "exact_action_available": sum(
            bool(row["exact_action_available"]) for row in rows
        ),
        "source_match_covered": sum(
            bool(row["source_match_covered"]) for row in rows
        ),
        "ranked_action_covered": len(ranks),
        "missing_rank_steps": [
            int(row["step"]) for row in rows if row.get("gate_rank") is None
        ],
        "rank_min": min(ranks) if ranks else None,
        "rank_median": median(ranks) if ranks else None,
        "rank_max": max(ranks) if ranks else None,
        "caps": {
            str(cap): {
                "covered": sum(
                    row.get("gate_rank") is not None
                    and int(row["gate_rank"]) <= cap
                    for row in rows
                ),
                "recall": sum(
                    row.get("gate_rank") is not None
                    and int(row["gate_rank"]) <= cap
                    for row in rows
                ) / max(1, len(rows)),
            }
            for cap in caps
        },
        "actions_detail": [
            {
                key: row.get(key)
                for key in (
                    "step",
                    "cost",
                    "reward",
                    "xfer_id",
                    "gate_delta",
                    "source_candidates",
                    "source_match_covered",
                    "gate_rank",
                )
            }
            for row in rows
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
