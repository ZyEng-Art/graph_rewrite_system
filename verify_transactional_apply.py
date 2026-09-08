from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
from pathlib import Path
from typing import Any


_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import quartz


def graph_state(graph) -> tuple[Any, ...]:
    """Capture every Python-visible structural property of a parent graph."""

    return (
        bytes(graph.exact_key()),
        graph.to_qasm_str(),
        int(graph.hash()),
        int(graph.gate_count),
        tuple((int(node.guid), int(node.gate_tp)) for node in graph.nodes),
        tuple(sorted(tuple(map(int, edge)) for edge in graph.all_edges())),
    )


def guid_structure(graph) -> tuple[dict[int, int], set[tuple[int, ...]]]:
    nodes = list(graph.nodes)
    node_types = {int(node.guid): int(node.gate_tp) for node in nodes}
    edges = {
        (
            int(nodes[int(src)].guid),
            int(nodes[int(dst)].guid),
            int(src_port),
            int(dst_port),
        )
        for src, dst, src_port, dst_port in graph.all_edges()
    }
    return node_types, edges


def assert_delta_reconstructs(parent, child, delta, label: str) -> None:
    node_types, edges = guid_structure(parent)
    for guid in delta["removed_node_guids"]:
        node_types.pop(int(guid))
    for guid, gate_type in delta["added_nodes"]:
        node_types[int(guid)] = int(gate_type)
    edges.difference_update(tuple(map(int, row)) for row in delta["removed_edges"])
    edges.update(tuple(map(int, row)) for row in delta["added_edges"])
    if (node_types, edges) != guid_structure(child):
        raise AssertionError(f"{label}: incremental structural delta mismatch")


def actions(graph, context, limit: int):
    first_by_xfer = {}
    remainder = []
    for anchor in graph.nodes:
        for xfer_id, node_ids, node_guids in graph.available_xfer_bindings_parallel(
            context=context, node=anchor
        ):
            row = (
                int(xfer_id),
                tuple(map(int, node_ids)),
                tuple(map(int, node_guids)),
            )
            if row[0] not in first_by_xfer:
                first_by_xfer[row[0]] = row
            else:
                remainder.append(row)
    rows = list(first_by_xfer.values()) + remainder
    return rows[:limit]


def audit_single_step(graph, context, xfers, limit: int) -> dict[str, int]:
    rows = actions(graph, context, limit)
    if not rows:
        raise AssertionError("input graph has no legal rewrite")
    counters = {
        "actions": 0,
        "xfers": 0,
        "novel": 0,
        "identity_rewrites": 0,
        "duplicate_replays": 0,
        "graph_copies": 0,
        "captured_in_entries": 0,
        "captured_out_entries": 0,
        "captured_position_entries": 0,
        "incremental_position_updates": 0,
    }
    audited_xfers = set()
    for action_index, (xfer_id, _node_ids, node_guids) in enumerate(rows):
        before = graph_state(graph)
        reference, _ = graph.apply_xfer_with_guid_binding(
            xfer=xfers[xfer_id],
            source_node_guids=node_guids,
            eliminate_rotation=True,
        )
        if reference is None:
            raise AssertionError(f"reference apply rejected action {action_index}")
        reference_key = bytes(reference.exact_key())

        registry = quartz.PyExactKeyRegistry()
        assert registry.insert(before[0])
        child, _dst, native_key, status, profile, delta = (
            graph.apply_xfer_with_guid_binding_transactional(
                xfer=xfers[xfer_id],
                source_node_guids=node_guids,
                registry=registry,
                eliminate_rotation=True,
            )
        )
        expected_status = 1 if reference_key == before[0] else 0
        if status != expected_status:
            raise AssertionError(
                f"action {action_index}: status {status}, expected {expected_status}"
            )
        if bytes(native_key) != reference_key:
            raise AssertionError(f"action {action_index}: exact identity mismatch")
        if expected_status == 0:
            if child is None or bytes(child.exact_key()) != reference_key:
                raise AssertionError(f"action {action_index}: novel child mismatch")
            if int(child.gate_count) != int(reference.gate_count):
                raise AssertionError(f"action {action_index}: gate count mismatch")
            if delta is None:
                raise AssertionError(f"action {action_index}: missing graph delta")
            assert_delta_reconstructs(
                graph, child, delta, f"action {action_index}"
            )
            counters["novel"] += 1
        else:
            if child is not None:
                raise AssertionError(f"action {action_index}: identity rewrite cloned")
            counters["identity_rewrites"] += 1
        if graph_state(graph) != before:
            raise AssertionError(f"action {action_index}: parent was not restored")

        replay, _dst, replay_key, replay_status, replay_profile, replay_delta = (
            graph.apply_xfer_with_guid_binding_transactional(
                xfer=xfers[xfer_id],
                source_node_guids=node_guids,
                registry=registry,
                eliminate_rotation=True,
            )
        )
        if replay is not None or replay_status != 1:
            raise AssertionError(f"action {action_index}: replay was not deduplicated")
        if replay_delta is not None:
            raise AssertionError(f"action {action_index}: duplicate exported a delta")
        if bytes(replay_key) != reference_key or graph_state(graph) != before:
            raise AssertionError(f"action {action_index}: replay rollback mismatch")

        counters["actions"] += 1
        audited_xfers.add(xfer_id)
        counters["duplicate_replays"] += 1
        counters["graph_copies"] += int(profile[21]) + int(replay_profile[21])
        counters["captured_in_entries"] += int(profile[16])
        counters["captured_out_entries"] += int(profile[17])
        counters["captured_position_entries"] += int(profile[19])
        counters["incremental_position_updates"] += int(profile[20])
    counters["xfers"] = len(audited_xfers)
    if counters["graph_copies"] != counters["novel"]:
        raise AssertionError("transaction copied a duplicate or missed a novel child")
    return counters


def audit_multistep(graph, context, xfers, depth: int, action_limit: int) -> dict:
    registry = quartz.PyExactKeyRegistry()
    assert registry.insert(graph.exact_key())
    current = graph
    trace = []
    duplicate_candidates = 0
    for step in range(depth):
        selected = None
        for xfer_id, _node_ids, node_guids in actions(
            current, context, action_limit
        ):
            before = graph_state(current)
            reference, _ = current.apply_xfer_with_guid_binding(
                xfer=xfers[xfer_id],
                source_node_guids=node_guids,
                eliminate_rotation=True,
            )
            child, _dst, key, status, profile, delta = (
                current.apply_xfer_with_guid_binding_transactional(
                    xfer=xfers[xfer_id],
                    source_node_guids=node_guids,
                    registry=registry,
                    eliminate_rotation=True,
                )
            )
            if graph_state(current) != before:
                raise AssertionError(f"step {step}: parent rollback mismatch")
            if reference is None:
                if status < 2:
                    raise AssertionError(f"step {step}: invalid status mismatch")
                continue
            if bytes(key) != bytes(reference.exact_key()):
                raise AssertionError(f"step {step}: reference identity mismatch")
            if status == 1:
                if child is not None or int(profile[21]) != 0:
                    raise AssertionError(f"step {step}: duplicate was copied")
                duplicate_candidates += 1
                continue
            if status != 0 or child is None or int(profile[21]) != 1:
                raise AssertionError(f"step {step}: novel transaction mismatch")
            if delta is None:
                raise AssertionError(f"step {step}: missing graph delta")
            assert_delta_reconstructs(current, child, delta, f"step {step}")
            selected = (xfer_id, child, profile)
            break
        if selected is None:
            break
        xfer_id, current, profile = selected
        trace.append(
            {
                "step": step + 1,
                "xfer_id": xfer_id,
                "gate_count": int(current.gate_count),
                "captured_in_entries": int(profile[16]),
                "captured_out_entries": int(profile[17]),
                "captured_position_entries": int(profile[19]),
                "incremental_position_updates": int(profile[20]),
            }
        )
    return {
        "completed_steps": len(trace),
        "duplicate_candidates": duplicate_candidates,
        "native_registry_size": int(registry.size),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, action="append", required=True)
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for name in ("PyExactKeyRegistry", "PyGraph"):
        if not hasattr(quartz, name):
            raise RuntimeError(f"loaded Quartz extension lacks {name}")
    if not hasattr(quartz.PyGraph, "apply_xfer_with_guid_binding_transactional"):
        raise RuntimeError("loaded Quartz extension lacks transactional apply")

    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    xfers = context.get_xfers()
    results = []
    for qasm in args.qasm:
        graph = quartz.PyGraph.from_qasm(context=context, filename=str(qasm))
        results.append(
            {
                "qasm": str(qasm),
                "initial_gate_count": int(graph.gate_count),
                "single_step": audit_single_step(
                    graph, context, xfers, args.max_actions
                ),
                "multistep": audit_multistep(
                    graph, context, xfers, args.depth, args.max_actions
                ),
            }
        )
    payload = {
        "all_reference_successors_equal": True,
        "all_parents_restored": True,
        "duplicates_clone_free": True,
        "rotation_elimination": True,
        "circuits": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
