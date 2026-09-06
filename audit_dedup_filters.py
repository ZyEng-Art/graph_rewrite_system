from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import RuleMetadata
from lazy_rollout_benchmark import (
    LazyAction,
    action_conflict_slots,
    is_direct_inverse_action,
    violates_canonical_trace_order,
)


def stored_action(step: dict) -> LazyAction:
    action = step["action"]
    delta = step["delta"]
    source_slots = tuple(map(int, action["binding_slots"]))
    destination_slots = tuple(map(int, action["dst_slots"]))
    return LazyAction(
        xfer_id=int(action["xfer_id"]),
        source_slots=source_slots,
        destination_slots=destination_slots,
        conflict_slots=action_conflict_slots(
            source_slots,
            destination_slots,
            delta["removed_edges"],
            delta["added_edges"],
        ),
    )


def audit_payload(payload: dict) -> dict:
    rules = RuleMetadata.from_payload(payload)
    inverse_ids = rules.unique_inverse_xfer_ids()
    trajectories = payload.get("train_trajectories", []) + payload.get(
        "test_trajectories", []
    )
    direct_inverse_rows = []
    trace_order_rows = []
    total_steps = 0
    independent_adjacent_pairs = 0
    for trajectory in trajectories:
        history: tuple[LazyAction, ...] = ()
        source_path = str(trajectory.get("source_path", ""))
        for offset, step in enumerate(trajectory["steps"]):
            action = stored_action(step)
            if history and set(history[-1].conflict_slots).isdisjoint(
                action.conflict_slots
            ):
                independent_adjacent_pairs += 1
            if is_direct_inverse_action(
                history, action.xfer_id, action.source_slots, inverse_ids
            ):
                direct_inverse_rows.append(
                    {
                        "source_path": source_path,
                        "offset": offset,
                        "step": int(step.get("index", offset)),
                        "previous_xfer_id": history[-1].xfer_id,
                        "xfer_id": action.xfer_id,
                        "source_slots": list(action.source_slots),
                    }
                )
            if violates_canonical_trace_order(history, action):
                trace_order_rows.append(
                    {
                        "source_path": source_path,
                        "offset": offset,
                        "step": int(step.get("index", offset)),
                        "xfer_id": action.xfer_id,
                        "source_slots": list(action.source_slots),
                    }
                )
            history += (action,)
            total_steps += 1
    return {
        "trajectory_windows": len(trajectories),
        "steps": total_steps,
        "unique_inverse_xfers": sum(xfer_id >= 0 for xfer_id in inverse_ids),
        "independent_adjacent_pairs": independent_adjacent_pairs,
        "direct_inverse_rejections": len(direct_inverse_rows),
        "canonical_trace_order_rejections": len(trace_order_rows),
        "direct_inverse_rows": direct_inverse_rows,
        "canonical_trace_order_rows": trace_order_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit history filters against authoritative trajectory actions."
    )
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {}
    for path in args.data:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        result[str(path)] = audit_payload(payload)
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
