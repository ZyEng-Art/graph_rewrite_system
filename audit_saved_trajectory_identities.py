from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import ctypes.util
import json
from pathlib import Path

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import quartz

from circuit_identity import canonical_qasm_key, exact_graph_key
from collect_quarl_trajectories import parse_trajectory_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact identities along saved Quarl optimization paths."
    )
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )

    result = {}
    for trajectory_path in args.trajectory:
        rows = parse_trajectory_directory(trajectory_path)
        native_positions: dict[object, list[int]] = defaultdict(list)
        qasm_positions: dict[object, list[int]] = defaultdict(list)
        native_to_qasm: dict[object, set] = defaultdict(set)
        qasm_to_native: dict[object, set] = defaultdict(set)
        cost_mismatches = []
        for row in rows:
            graph = quartz.PyGraph.from_qasm(
                context=context, filename=str(row.qasm)
            )
            if int(graph.gate_count) != row.cost:
                cost_mismatches.append(
                    {
                        "step": row.step,
                        "saved_cost": row.cost,
                        "graph_gate_count": int(graph.gate_count),
                    }
                )
            native = exact_graph_key(graph)
            fallback = canonical_qasm_key(graph.to_qasm_str())
            native_positions[native].append(row.step)
            qasm_positions[fallback].append(row.step)
            native_to_qasm[native].add(fallback)
            qasm_to_native[fallback].add(native)
        result[str(trajectory_path)] = {
            "states": len(rows),
            "actions": len(rows) - 1,
            "native_unique_identities": len(native_positions),
            "qasm_unique_identities": len(qasm_positions),
            "native_duplicate_step_groups": [
                positions
                for positions in native_positions.values()
                if len(positions) > 1
            ],
            "qasm_duplicate_step_groups": [
                positions
                for positions in qasm_positions.values()
                if len(positions) > 1
            ],
            "native_qasm_partition_mismatches": sum(
                len(values) != 1 for values in native_to_qasm.values()
            )
            + sum(len(values) != 1 for values in qasm_to_native.values()),
            "cost_mismatches": cost_mismatches,
        }

    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
