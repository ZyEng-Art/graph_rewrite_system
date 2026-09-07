from __future__ import annotations

import argparse
import json
from pathlib import Path


NATIVE_STAGES = (
    "native_guid_lookup",
    "native_source_match",
    "native_input_validation",
    "native_destination_creation",
    "native_output_validation",
    "native_graph_rewrite",
    "native_loop_check",
    "native_trace",
    "native_rotation_elimination",
    "native_unmatch_cleanup",
    "native_wrapper_unattributed",
)
GRAPH_REWRITE_STAGES = (
    "native_graph_allocate",
    "native_graph_copy_constants",
    "native_graph_copy_special_guid",
    "native_graph_copy_qubit_map",
    "native_graph_copy_in_edges",
    "native_graph_copy_out_edges",
    "native_graph_reconnect_outputs",
    "native_graph_remove_source_ops",
    "native_graph_add_destination_ops",
    "native_graph_rebuild_qubit_index",
)
CHILD_STAGES = (
    "child_source_slots",
    "child_materialize_nodes",
    "child_filter_destinations",
    "child_copy_slot_map",
    "child_update_slots",
    "child_snapshot_nodes",
    "child_snapshot_node_rows",
    "child_snapshot_native_edges",
    "child_snapshot_edge_rows",
    "child_graph_delta",
    "child_local_metadata",
    "child_rewrite_distance",
    "child_state_construction",
)


def stage_rows(
    seconds: dict[str, float], names: tuple[str, ...], total: float
) -> list[dict[str, float | str]]:
    rows = []
    for name in names:
        value = float(seconds.get(name, 0.0))
        rows.append(
            {
                "stage": name,
                "seconds": value,
                "percent": 100.0 * value / max(total, 1e-12),
            }
        )
    return sorted(rows, key=lambda row: row["seconds"], reverse=True)


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text())
    profile = payload["apply_profile"]
    if profile["mode"] != "detailed":
        raise ValueError(f"not a detailed apply profile: {path}")
    seconds = profile["seconds"]
    counts = profile["counts"]
    apply_loop = sum(float(row["quartz_apply_seconds"]) for row in payload["steps"])
    fingerprint = float(payload["preapply_fingerprint_seconds"])
    python_binding = float(seconds.get("python_slot_to_guid", 0.0)) + float(
        seconds.get("python_binding_to_guid", 0.0)
    )
    top_level_values = {
        "preapply_fingerprint": fingerprint,
        "python_binding_preparation": python_binding,
        "native_apply_wall": float(seconds.get("native_call_wall", 0.0)),
        "exact_graph_key": float(seconds.get("exact_graph_key", 0.0)),
        "exact_registry": float(seconds.get("exact_registry", 0.0)),
        "post_apply_fingerprint_audit": float(
            seconds.get("post_apply_fingerprint_audit", 0.0)
        ),
        "accepted_child_metadata": float(seconds.get("child_total", 0.0)),
    }
    attributed = sum(top_level_values.values())
    top_level_values["python_loop_and_unattributed"] = max(
        0.0, apply_loop - attributed
    )
    top_level = [
        {
            "stage": name,
            "seconds": value,
            "percent_of_apply_loop": 100.0
            * value
            / max(apply_loop, 1e-12),
        }
        for name, value in top_level_values.items()
    ]
    top_level.sort(key=lambda row: row["seconds"], reverse=True)
    native_wall = float(seconds.get("native_call_wall", 0.0))
    graph_rewrite = float(seconds.get("native_graph_rewrite", 0.0))
    child_total = float(seconds.get("child_total", 0.0))
    attempted = int(counts.get("attempted", 0))
    exact_keys = int(counts.get("exact_graph_key", 0))
    children = int(counts.get("materialized_children", 0))
    return {
        "path": str(path),
        "qasm": payload["qasm"],
        "completed_depth": payload["completed_depth"],
        "best_gate_count": payload["best_gate_count"],
        "total_seconds": payload["total_seconds"],
        "apply_loop_seconds": apply_loop,
        "attempted_applies": attempted,
        "successful_applies": exact_keys,
        "materialized_children": children,
        "top_level": top_level,
        "native": stage_rows(seconds, NATIVE_STAGES, native_wall),
        "graph_rewrite": stage_rows(
            seconds, GRAPH_REWRITE_STAGES, graph_rewrite
        ),
        "child_metadata": stage_rows(seconds, CHILD_STAGES, child_total),
        "average_microseconds": {
            "native_apply_wall_per_attempt": 1e6
            * native_wall
            / max(attempted, 1),
            "exact_graph_key_per_success": 1e6
            * float(seconds.get("exact_graph_key", 0.0))
            / max(exact_keys, 1),
            "child_metadata_per_materialized_child": 1e6
            * child_total
            / max(children, 1),
            "fingerprint_profile_build": 1e6
            * float(seconds.get("fingerprint_native_profile_build", 0.0))
            / max(int(counts.get("fingerprint_native_profile_build", 0)), 1),
            "fingerprint_compute": 1e6
            * float(seconds.get("fingerprint_native_compute", 0.0))
            / max(int(counts.get("fingerprint_native_compute", 0)), 1),
        },
        "result_counts": {
            key: value
            for key, value in counts.items()
            if key.startswith("native_result_")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = [summarize(path) for path in args.profiles]
    rendered = json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
