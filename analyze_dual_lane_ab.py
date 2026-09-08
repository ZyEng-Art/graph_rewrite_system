from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


def load_runs(path: Path) -> dict[tuple[str, int, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in payload["runs"]:
        if row.get("status") != "completed":
            continue
        key = (str(row["state_id"]), int(row["budget"]), int(row["seed"]))
        rows[key] = row
    return rows


def analyze(gate_path: Path, dual_path: Path) -> dict:
    gate = load_runs(gate_path)
    dual = load_runs(dual_path)
    keys = sorted(set(gate) & set(dual))
    pairs = []
    wins = Counter()
    for key in keys:
        left = gate[key]
        right = dual[key]
        gate_improvement = int(left["improvement"])
        dual_improvement = int(right["improvement"])
        outcome = (
            "dual_win" if dual_improvement > gate_improvement
            else "gate_win" if gate_improvement > dual_improvement
            else "tie"
        )
        wins[outcome] += 1
        behavior_stratum = str(left.get("behavior_stratum") or "unknown")
        teacher_class = behavior_stratum.split(":", 1)[0]
        gate_first = left.get("first_improvement_depth")
        dual_first = right.get("first_improvement_depth")
        pairs.append(
            {
                "state_id": key[0],
                "budget": key[1],
                "seed": key[2],
                "circuit": left.get("circuit"),
                "behavior_stratum": behavior_stratum,
                "teacher_class": teacher_class,
                "gate_improvement": gate_improvement,
                "dual_improvement": dual_improvement,
                "improvement_delta": dual_improvement - gate_improvement,
                "gate_attempted_actions": int(left["attempted_actions"]),
                "dual_attempted_actions": int(right["attempted_actions"]),
                "attempted_action_delta": (
                    int(right["attempted_actions"])
                    - int(left["attempted_actions"])
                ),
                "gate_completed_depth": int(left["completed_depth"]),
                "dual_completed_depth": int(right["completed_depth"]),
                "gate_search_seconds": float(left["search_seconds"]),
                "dual_search_seconds": float(right["search_seconds"]),
                "gate_first_improvement_depth": gate_first,
                "dual_first_improvement_depth": dual_first,
                "gate_realized_delayed_gain": bool(
                    gate_improvement > 0
                    and gate_first is not None
                    and int(gate_first) > 4
                ),
                "dual_realized_delayed_gain": bool(
                    dual_improvement > 0
                    and dual_first is not None
                    and int(dual_first) > 4
                ),
                "outcome": outcome,
            }
        )

    by_circuit = {}
    circuits = sorted({str(row["circuit"]) for row in pairs})
    for circuit in circuits:
        rows = [row for row in pairs if str(row["circuit"]) == circuit]
        counts = Counter(row["outcome"] for row in rows)
        by_circuit[circuit] = {
            "pairs": len(rows),
            **dict(counts),
            "mean_improvement_delta": mean(
                row["improvement_delta"] for row in rows
            ),
            "mean_search_seconds_delta": mean(
                row["dual_search_seconds"] - row["gate_search_seconds"]
                for row in rows
            ),
            "mean_completed_depth_delta": mean(
                row["dual_completed_depth"] - row["gate_completed_depth"]
                for row in rows
            ),
            "gate_realized_delayed_gains": sum(
                row["gate_realized_delayed_gain"] for row in rows
            ),
            "dual_realized_delayed_gains": sum(
                row["dual_realized_delayed_gain"] for row in rows
            ),
        }

    by_teacher_class = {}
    for teacher_class in sorted({row["teacher_class"] for row in pairs}):
        rows = [row for row in pairs if row["teacher_class"] == teacher_class]
        counts = Counter(row["outcome"] for row in rows)
        by_teacher_class[teacher_class] = {
            "pairs": len(rows),
            **dict(counts),
            "mean_improvement_delta": mean(
                row["improvement_delta"] for row in rows
            ),
            "gate_realized_delayed_gains": sum(
                row["gate_realized_delayed_gain"] for row in rows
            ),
            "dual_realized_delayed_gains": sum(
                row["dual_realized_delayed_gain"] for row in rows
            ),
        }

    return {
        "format": "dual-lane-equal-apply-ab-v1",
        "paired_runs": len(pairs),
        "unpaired_gate_runs": len(set(gate) - set(dual)),
        "unpaired_dual_runs": len(set(dual) - set(gate)),
        "outcomes": dict(wins),
        "mean_improvement_delta": (
            mean(row["improvement_delta"] for row in pairs) if pairs else 0.0
        ),
        "mean_attempted_action_delta": (
            mean(row["attempted_action_delta"] for row in pairs)
            if pairs else 0.0
        ),
        "mean_completed_depth_delta": (
            mean(
                row["dual_completed_depth"] - row["gate_completed_depth"]
                for row in pairs
            )
            if pairs else 0.0
        ),
        "by_circuit": by_circuit,
        "by_teacher_class": by_teacher_class,
        "gate_realized_delayed_gains": sum(
            row["gate_realized_delayed_gain"] for row in pairs
        ),
        "dual_realized_delayed_gains": sum(
            row["dual_realized_delayed_gain"] for row in pairs
        ),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_summary", type=Path)
    parser.add_argument("dual_summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.gate_summary, args.dual_summary)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
