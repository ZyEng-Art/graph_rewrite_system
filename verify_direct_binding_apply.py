from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
from pathlib import Path


_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import quartz


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Quartz anchor-rematch and direct-GUID apply on legal "
            "bindings from an arbitrary input circuit."
        )
    )
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, default=512)
    args = parser.parse_args()

    if not hasattr(quartz.PyGraph, "apply_xfer_with_guid_binding"):
        raise RuntimeError("loaded Quartz extension lacks direct GUID binding apply")
    if not hasattr(quartz.PyGraph, "exact_key"):
        raise RuntimeError("loaded Quartz extension lacks collision-safe exact_key")

    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    graph = quartz.PyGraph.from_qasm(context=context, filename=str(args.qasm))
    xfers = context.get_xfers()

    # Prefer rule diversity: retain one binding for each xfer before filling
    # the remaining audit budget with additional concrete bindings.
    first_by_xfer = {}
    remaining = []
    for anchor in graph.nodes:
        for xfer_id, node_ids, node_guids in graph.available_xfer_bindings_parallel(
            context=context, node=anchor
        ):
            row = (int(xfer_id), tuple(map(int, node_ids)), tuple(map(int, node_guids)))
            if row[0] not in first_by_xfer:
                first_by_xfer[row[0]] = row
            else:
                remaining.append(row)
    actions = list(first_by_xfer.values())
    actions.extend(remaining[: max(0, args.max_actions - len(actions))])
    actions = actions[: args.max_actions]

    for index, (xfer_id, node_ids, node_guids) in enumerate(actions):
        anchor_graph, _, anchor_sources, _ = graph.apply_xfer_with_binding_trace(
            xfer=xfers[xfer_id],
            node=graph.get_node_from_id(id=node_ids[0]),
            eliminate_rotation=True,
            predecessor_layers=1,
        )
        direct_graph, _ = graph.apply_xfer_with_guid_binding(
            xfer=xfers[xfer_id],
            source_node_guids=node_guids,
            eliminate_rotation=True,
        )
        if anchor_graph is None or direct_graph is None:
            raise AssertionError(f"legal action {index} failed to apply")
        if tuple(map(int, anchor_sources)) != node_guids:
            raise AssertionError(f"anchor binding changed at action {index}")
        if anchor_graph.exact_key() != direct_graph.exact_key():
            raise AssertionError(f"successor identity mismatch at action {index}")
        if int(anchor_graph.gate_count) != int(direct_graph.gate_count):
            raise AssertionError(f"successor gate-count mismatch at action {index}")

    print(
        json.dumps(
            {
                "qasm": str(args.qasm),
                "initial_gate_count": int(graph.gate_count),
                "audited_actions": len(actions),
                "audited_xfers": len({row[0] for row in actions}),
                "all_exact_successors_equal": True,
                "rotation_elimination": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
