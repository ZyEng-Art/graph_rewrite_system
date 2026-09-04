from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import ctypes.util
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch


def update_slots(graph, guid_to_slot: dict[int, int], next_slot: int) -> int:
    for node in graph.nodes:
        guid = int(node.guid)
        if guid not in guid_to_slot:
            guid_to_slot[guid] = next_slot
            next_slot += 1
    return next_slot


def snapshot(graph, guid_to_slot: dict[int, int]) -> dict:
    nodes = list(graph.nodes)
    return {
        "nodes": sorted(
            (guid_to_slot[int(node.guid)], int(node.gate_tp), int(node.guid))
            for node in nodes
        ),
        "edges": sorted(
            (
                guid_to_slot[int(nodes[int(src)].guid)],
                guid_to_slot[int(nodes[int(dst)].guid)],
                int(src_port),
                int(dst_port),
            )
            for src, dst, src_port, dst_port in graph.all_edges()
        ),
    }


def graph_delta(before: dict, after: dict) -> dict:
    before_nodes = {
        int(slot): (int(gate_type), int(guid))
        for slot, gate_type, guid in before["nodes"]
    }
    after_nodes = {
        int(slot): (int(gate_type), int(guid))
        for slot, gate_type, guid in after["nodes"]
    }
    before_edges = set(map(tuple, before["edges"]))
    after_edges = set(map(tuple, after["edges"]))
    return {
        "removed_slots": sorted(set(before_nodes) - set(after_nodes)),
        "added_nodes": sorted(
            (slot, after_nodes[slot][0], after_nodes[slot][1])
            for slot in set(after_nodes) - set(before_nodes)
        ),
        "removed_edges": sorted(before_edges - after_edges),
        "added_edges": sorted(after_edges - before_edges),
    }


def enumerate_matches(graph, context, xfer_to_source, guid_to_slot):
    started = time.perf_counter()
    grouped = {}
    for anchor_id, node in enumerate(graph.nodes):
        for xfer_id, _, node_guids in graph.available_xfer_bindings_parallel(
            context=context, node=node
        ):
            binding_slots = tuple(guid_to_slot[int(guid)] for guid in node_guids)
            source_id = int(xfer_to_source[int(xfer_id)])
            key = (source_id, binding_slots)
            if key not in grouped:
                grouped[key] = {
                    "source_id": source_id,
                    "anchor_id": int(anchor_id),
                    "anchor_slot": int(binding_slots[0]),
                    "binding_slots": binding_slots,
                    "binding_guids": tuple(map(int, node_guids)),
                    "xfer_ids": [],
                }
            grouped[key]["xfer_ids"].append(int(xfer_id))
    rows = list(grouped.values())
    for row in rows:
        row["xfer_ids"].sort()
    rows.sort(
        key=lambda row: (
            row["anchor_slot"], row["source_id"], row["binding_slots"]
        )
    )
    return rows, (time.perf_counter() - started) * 1000.0


def history_key(history: list[dict]) -> tuple:
    return tuple(
        (
            int(action["xfer_id"]),
            tuple(map(int, action["source_slots"])),
            tuple(map(int, action["destination_slots"])),
        )
        for action in history
    )


def replay_history(
    *,
    quartz,
    context,
    xfers,
    xfer_to_source,
    initial_qasm: str,
    history: list[dict],
    trajectory_id: int,
    circuit_name: str,
    terminal_only: bool,
) -> dict | None:
    graph = quartz.PyGraph.from_qasm_str(context=context, qasm_str=initial_qasm)
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    trajectory = {
        "split": "train",
        "trajectory_id": trajectory_id,
        "circuit": circuit_name,
        "initial_graph": snapshot(graph, guid_to_slot),
        "steps": [],
        "terminal_only_supervision": terminal_only,
    }
    previous_preferred: set[int] = set()
    local_streak = 0
    for step_index, dumped_action in enumerate(history):
        before = snapshot(graph, guid_to_slot)
        if terminal_only:
            matches, match_ms = [], 0.0
        else:
            matches, match_ms = enumerate_matches(
                graph, context, xfer_to_source, guid_to_slot
            )
        source_slots = tuple(map(int, dumped_action["source_slots"]))
        destination_slots = tuple(map(int, dumped_action["destination_slots"]))
        xfer_id = int(dumped_action["xfer_id"])
        slot_to_guid = {
            guid_to_slot[int(node.guid)]: int(node.guid) for node in graph.nodes
        }
        if not source_slots or any(slot not in slot_to_guid for slot in source_slots):
            break
        guid_to_id = {
            int(node.guid): index for index, node in enumerate(graph.nodes)
        }
        anchor_node = graph.get_node_from_id(
            id=guid_to_id[slot_to_guid[source_slots[0]]]
        )
        apply_started = time.perf_counter()
        result = graph.apply_xfer_with_binding_trace(
            xfer=xfers[xfer_id],
            node=anchor_node,
            eliminate_rotation=False,
            predecessor_layers=1,
        )
        apply_ms = (time.perf_counter() - apply_started) * 1000.0
        if result is None or result[0] is None:
            break
        next_graph, _, source_guids, destination_guids = result
        actual_source_slots = tuple(
            guid_to_slot[int(guid)] for guid in source_guids
        )
        if actual_source_slots != source_slots:
            break
        if len(destination_guids) != len(destination_slots):
            break
        for guid, slot in zip(destination_guids, destination_slots):
            guid = int(guid)
            slot = int(slot)
            if guid in guid_to_slot and guid_to_slot[guid] != slot:
                raise RuntimeError("destination GUID changed its persistent slot")
            guid_to_slot[guid] = slot
        next_slot = max(next_slot, max(destination_slots, default=-1) + 1)
        next_slot = update_slots(next_graph, guid_to_slot, next_slot)
        after = snapshot(next_graph, guid_to_slot)
        delta = graph_delta(before, after)
        if delta["removed_slots"] != sorted(source_slots):
            raise RuntimeError("on-policy replay removed unexpected slots")
        if [row[0] for row in delta["added_nodes"]] != sorted(destination_slots):
            raise RuntimeError("on-policy replay added unexpected slots")

        continued = source_slots[0] in previous_preferred
        local_streak = local_streak + 1 if continued else 0
        source_set = set(source_slots)
        predecessors = {
            int(src)
            for src, dst, _, _ in before["edges"]
            if int(dst) in source_set
        }
        live = {int(row[0]) for row in after["nodes"]}
        previous_preferred = (set(destination_slots) | predecessors) & live
        action = {
            "xfer_id": xfer_id,
            "source_id": int(xfer_to_source[xfer_id]),
            "anchor_slot": int(source_slots[0]),
            "binding_slots": source_slots,
            "binding_guids": tuple(map(int, source_guids)),
            "dst_slots": destination_slots,
            "dst_guids": tuple(map(int, destination_guids)),
        }
        trajectory["steps"].append(
            {
                "index": step_index,
                "graph_hash": int(graph.hash()),
                "num_nodes": int(graph.num_nodes),
                "local_streak": local_streak,
                "continued_local": int(continued),
                "match_ms": match_ms,
                "apply_ms": apply_ms,
                "matches": matches,
                "action": action,
                "delta": delta,
            }
        )
        graph = next_graph

    if not trajectory["steps"]:
        return None
    terminal_matches, terminal_ms = enumerate_matches(
        graph, context, xfer_to_source, guid_to_slot
    )
    trajectory["terminal_matches"] = terminal_matches
    trajectory["terminal_match_ms"] = terminal_ms
    trajectory["requested_history_length"] = len(history)
    trajectory["valid_history_length"] = len(trajectory["steps"])
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histories", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-histories-per-file", type=int, default=256)
    parser.add_argument("--terminal-only", action="store_true")
    args = parser.parse_args()

    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    reference = torch.load(
        args.reference_data, map_location="cpu", weights_only=False
    )
    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    xfers = context.get_xfers()
    trajectories = []
    seen = set()
    attempted = invalid = 0
    match_seconds = 0.0
    for history_path in args.histories:
        payload = json.loads(history_path.read_text())
        accepted_from_file = 0
        for state in payload["states"]:
            key = history_key(state["history"])
            if key in seen:
                continue
            seen.add(key)
            attempted += 1
            started = time.perf_counter()
            trajectory = replay_history(
                quartz=quartz,
                context=context,
                xfers=xfers,
                xfer_to_source=reference["xfer_to_source"],
                initial_qasm=payload["initial_qasm"],
                history=state["history"],
                trajectory_id=len(trajectories),
                circuit_name=Path(payload["qasm"]).name,
                terminal_only=args.terminal_only,
            )
            match_seconds += time.perf_counter() - started
            if trajectory is None:
                invalid += 1
                continue
            trajectories.append(trajectory)
            accepted_from_file += 1
            if accepted_from_file >= args.max_histories_per_file:
                break
            if len(trajectories) % 25 == 0:
                print(
                    f"collected={len(trajectories)} attempted={attempted}",
                    flush=True,
                )

    result = {
        key: reference[key]
        for key in (
            "format",
            "source_patterns",
            "xfer_to_source",
            "xfer_sources",
            "xfer_destinations",
        )
    }
    result["train_trajectories"] = trajectories
    result["test_trajectories"] = []
    result["metadata"] = {
        "kind": "on_policy_beam_histories",
        "history_files": [str(path) for path in args.histories],
        "attempted_histories": attempted,
        "invalid_histories": invalid,
        "collection_seconds": match_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(
        f"saved={args.output} trajectories={len(trajectories)} "
        f"prefix_states={sum(len(row['steps']) for row in trajectories)} "
        f"terminal_states={len(trajectories)} invalid={invalid} "
        f"seconds={match_seconds:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
