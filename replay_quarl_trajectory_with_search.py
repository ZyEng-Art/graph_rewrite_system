from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch

from beam_search_benchmark import (
    BeamState,
    Proposal,
    make_child,
    snapshot,
    update_slots,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a converted Quarl teacher trajectory through the exact "
            "beam-search child executor."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    trajectories = payload["train_trajectories"] + payload["test_trajectories"]
    trajectory = trajectories[args.trajectory_index]
    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    xfers = context.get_xfers()
    graph = quartz.PyGraph.from_qasm(context=context, filename=str(args.qasm))
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    initial_snapshot = snapshot(graph, guid_to_slot)
    state = BeamState(
        graph=graph,
        snapshot=initial_snapshot,
        guid_to_slot=guid_to_slot,
        next_slot=next_slot,
        last_touched={},
        rewrite_distance={int(row[0]): 5 for row in initial_snapshot["nodes"]},
        previous_preferred=set(),
        local_streak=0,
        gate_count=int(graph.gate_count),
        depth=0,
        history=(),
    )

    rows = []
    steps = trajectory["steps"]
    for index, step in enumerate(steps):
        current_hash = int(state.graph.hash())
        expected_current_hash = int(step["graph_hash"])
        if current_hash != expected_current_hash:
            raise AssertionError(
                f"step {index}: current graph hash {current_hash} != "
                f"teacher {expected_current_hash}"
            )
        action = step["action"]
        proposal = Proposal(
            parent=0,
            xfer_id=int(action["xfer_id"]),
            anchor_slot=int(action["anchor_slot"]),
            binding=tuple(map(int, action["binding_slots"])),
            probability=1.0,
            # Rotation elimination can contract the declared destination, so
            # make_child deliberately verifies the actual Quartz result instead.
            next_gate_count=0,
        )
        child = make_child(
            state,
            proposal,
            context,
            xfers,
            eliminate_rotation=True,
        )
        if child is None:
            raise AssertionError(f"step {index}: exact search executor rejected teacher action")
        if index + 1 < len(steps):
            expected_next_hash = int(steps[index + 1]["graph_hash"])
        else:
            expected_next_hash = int(trajectory["terminal_graph_hash"])
        next_hash = int(child.graph.hash())
        if next_hash != expected_next_hash:
            raise AssertionError(
                f"step {index}: successor hash {next_hash} != teacher {expected_next_hash}"
            )
        rows.append(
            {
                "step": index,
                "xfer_id": proposal.xfer_id,
                "anchor_slot": proposal.anchor_slot,
                "source_size": len(proposal.binding),
                "normalized_destination_nodes": len(action.get("dst_slots", ())),
                "declared_destination_nodes": len(
                    action.get("declared_dst_guids", action.get("dst_guids", ()))
                ),
                "gate_count": child.gate_count,
                "graph_hash": next_hash,
            }
        )
        state = child

    result = {
        "data": str(args.data),
        "trajectory_index": args.trajectory_index,
        "qasm": str(args.qasm),
        "steps": len(rows),
        "initial_gate_count": int(graph.gate_count),
        "final_gate_count": state.gate_count,
        "normalization_contracted_steps": sum(
            row["normalized_destination_nodes"] < row["declared_destination_nodes"]
            for row in rows
        ),
        "all_successor_hashes_exact": True,
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
