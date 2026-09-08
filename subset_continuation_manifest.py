from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-circuit", type=int, default=1)
    parser.add_argument("--state-id", action="append", default=[])
    args = parser.parse_args()
    if args.per_circuit < 1:
        parser.error("--per-circuit must be positive")

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    selected = []
    requested = set(args.state_id)
    for row in payload["states"]:
        if requested:
            if row["id"] in requested:
                selected.append(row)
            continue
        circuit = str(row.get("circuit"))
        if counts.get(circuit, 0) >= args.per_circuit:
            continue
        selected.append(row)
        counts[circuit] = counts.get(circuit, 0) + 1
    if requested:
        missing = requested - {row["id"] for row in selected}
        if missing:
            raise RuntimeError(f"state ids not found: {sorted(missing)}")
    if not selected:
        raise RuntimeError("manifest selection is empty")
    payload["states"] = selected
    payload["subset"] = {
        "source_manifest": str(args.manifest),
        "per_circuit": args.per_circuit,
        "state_ids": sorted(requested),
        "selected_states": len(selected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["subset"], sort_keys=True))


if __name__ == "__main__":
    main()
