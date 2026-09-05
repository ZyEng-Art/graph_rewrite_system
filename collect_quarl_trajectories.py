from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
import sys
import time
import types

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch

from collect_onpolicy_histories import (
    enumerate_matches,
    graph_delta,
    snapshot,
    update_slots,
)
from dataset import RuleMetadata, validate_trajectory


TRAJECTORY_FILE = re.compile(
    r"^(?P<step>\d+)_(?P<cost>-?\d+)_(?P<reward>-?\d+)_"
    r"(?P<node>\d+)_(?P<xfer>\d+)\.qasm$"
)


@dataclass(frozen=True)
class SavedStep:
    step: int
    cost: int
    reward: int
    node_id: int
    xfer_id: int
    qasm: Path


def parse_trajectory_directory(path: Path) -> list[SavedStep]:
    rows = []
    for qasm in path.glob("*.qasm"):
        match = TRAJECTORY_FILE.fullmatch(qasm.name)
        if match is None:
            raise ValueError(f"unexpected trajectory filename: {qasm}")
        rows.append(
            SavedStep(
                step=int(match["step"]),
                cost=int(match["cost"]),
                reward=int(match["reward"]),
                node_id=int(match["node"]),
                xfer_id=int(match["xfer"]),
                qasm=qasm,
            )
        )
    rows.sort(key=lambda row: row.step)
    if len(rows) < 2:
        raise ValueError(f"trajectory needs an action and terminal state: {path}")
    if [row.step for row in rows] != list(range(len(rows))):
        raise ValueError(f"trajectory steps are not contiguous: {path}")
    if (rows[-1].node_id, rows[-1].xfer_id) != (0, 0):
        raise ValueError(f"trajectory terminal sentinel is missing: {path}")
    return rows


def discover_trajectory_directories(roots: list[Path]) -> list[Path]:
    directories = {
        qasm.parent
        for root in roots
        for qasm in root.rglob("*.qasm")
    }
    return sorted(
        (
            path
            for path in directories
            if len(list(path.glob("*.qasm"))) >= 2
        ),
        key=lambda path: str(path),
    )


def excluded(path: Path, suffixes: list[str]) -> bool:
    normalized = path.as_posix()
    return any(normalized.endswith(suffix.replace("\\", "/")) for suffix in suffixes)


def resolve_transition(graph, target, xfer, declared_node_id: int):
    """Return the binding trace whose successor is the saved target graph."""
    target_hash = int(target.hash())
    nodes = list(graph.nodes)
    order = list(range(len(nodes)))
    if 0 <= declared_node_id < len(nodes):
        order.remove(declared_node_id)
        order.insert(0, declared_node_id)
    for node_id in order:
        result = graph.apply_xfer_with_binding_trace(
            xfer=xfer,
            node=graph.get_node_from_id(id=node_id),
            eliminate_rotation=True,
            predecessor_layers=1,
        )
        if result is not None and result[0] is not None:
            if int(result[0].hash()) == target_hash:
                return result, node_id
    return None, None


def convert_trajectory(
    path: Path,
    *,
    quartz,
    context,
    xfers,
    rules: RuleMetadata,
    trajectory_id: int,
    split_unrepresentable: bool = False,
) -> tuple[list[dict], list[dict]]:
    saved = parse_trajectory_directory(path)
    graph = quartz.PyGraph.from_qasm(context=context, filename=str(saved[0].qasm))
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    segments = []
    skipped_transitions = []
    segment_index = 0

    def new_segment(initial_graph) -> dict:
        return {
            "split": "train",
            "trajectory_id": trajectory_id,
            "circuit": path.parent.name + ".qasm",
            "source_path": f"{path}#segment={segment_index}",
            "initial_graph": snapshot(initial_graph, guid_to_slot),
            "steps": [],
            "terminal_only_supervision": False,
        }

    def finish_segment(trajectory: dict, terminal_graph) -> None:
        if not trajectory["steps"]:
            return
        terminal_matches, terminal_ms = enumerate_matches(
            terminal_graph, context, rules.xfer_to_source, guid_to_slot
        )
        trajectory["terminal_matches"] = terminal_matches
        trajectory["terminal_match_ms"] = terminal_ms
        trajectory["terminal_graph_hash"] = int(terminal_graph.hash())
        trajectory["requested_history_length"] = len(trajectory["steps"])
        trajectory["valid_history_length"] = len(trajectory["steps"])
        validate_trajectory(trajectory, rules)
        segments.append(trajectory)

    trajectory = new_segment(graph)
    previous_preferred: set[int] = set()
    local_streak = 0
    for index, current in enumerate(saved[:-1]):
        following = saved[index + 1]
        if int(graph.gate_count) != current.cost:
            raise ValueError(
                f"saved cost differs from graph at step {current.step}: {path}"
            )
        target = quartz.PyGraph.from_qasm(
            context=context, filename=str(following.qasm)
        )
        if current.cost - int(target.gate_count) != current.reward:
            raise ValueError(f"saved reward differs from graph delta: {path}")
        before = snapshot(graph, guid_to_slot)
        matches, match_ms = enumerate_matches(
            graph, context, rules.xfer_to_source, guid_to_slot
        )
        apply_started = time.perf_counter()
        result, resolved_node_id = resolve_transition(
            graph,
            target,
            xfers[current.xfer_id],
            current.node_id,
        )
        apply_ms = (time.perf_counter() - apply_started) * 1000.0
        if result is None:
            representation_error = (
                f"no anchor reproduces saved transition at step {current.step}: {path}"
            )
            next_graph = None
        else:
            next_graph, _, source_guids, destination_guids = result
            source_slots = tuple(guid_to_slot[int(guid)] for guid in source_guids)
            candidate_mapping = dict(guid_to_slot)
            candidate_next_slot = next_slot
            declared_destination_guids = tuple(map(int, destination_guids))
            live_destination_guids = {
                int(node.guid): int(node.gate_tp) for node in next_graph.nodes
            }
            surviving_destination_guids = tuple(
                guid
                for guid in declared_destination_guids
                if guid in live_destination_guids
            )
            normalized_away_destination_guids = tuple(
                guid
                for guid in declared_destination_guids
                if guid not in live_destination_guids
            )
            destination_slots = []
            representation_error = None
            for guid in surviving_destination_guids:
                if guid in candidate_mapping:
                    representation_error = (
                        "rewrite reused a live destination GUID at step "
                        f"{current.step}: {path}"
                    )
                    break
                candidate_mapping[guid] = candidate_next_slot
                destination_slots.append(candidate_next_slot)
                candidate_next_slot += 1
            destination_slots = tuple(destination_slots)
            if representation_error is None:
                candidate_next_slot = update_slots(
                    next_graph, candidate_mapping, candidate_next_slot
                )
                after = snapshot(next_graph, candidate_mapping)
                delta = graph_delta(before, after)
                if delta["removed_slots"] != sorted(source_slots):
                    representation_error = (
                        "rotation elimination removed nodes outside source at step "
                        f"{current.step}: {path}"
                    )
                elif {int(row[2]) for row in delta["added_nodes"]} != set(
                    surviving_destination_guids
                ):
                    representation_error = (
                        f"normalization created nodes outside the xfer destination at "
                        f"step {current.step}: {path}; "
                        f"surviving_destination_guids={surviving_destination_guids} "
                        f"normalized_away_destination_guids="
                        f"{normalized_away_destination_guids} "
                        f"added_nodes={delta['added_nodes']} "
                        f"expected_types={rules.destination_gate_types[current.xfer_id]}"
                    )

        if representation_error is not None:
            if not split_unrepresentable:
                raise ValueError(representation_error)
            finish_segment(trajectory, graph)
            skipped_transitions.append(
                {
                    "path": str(path),
                    "step": current.step,
                    "xfer_id": current.xfer_id,
                    "error": representation_error,
                }
            )
            # The saved successor is still an exact graph state.  Start a new
            # causal training segment there, omitting only the transition that
            # the incremental rule representation cannot express (typically a
            # rewrite followed by Quartz's automatic RZ elimination).
            graph = target
            guid_to_slot = {}
            next_slot = update_slots(graph, guid_to_slot, 0)
            segment_index += 1
            trajectory = new_segment(graph)
            previous_preferred = set()
            local_streak = 0
            continue

        guid_to_slot = candidate_mapping
        next_slot = candidate_next_slot

        source_set = set(source_slots)
        predecessors = {
            int(src) for src, dst, _, _ in before["edges"] if int(dst) in source_set
        }
        live = {int(row[0]) for row in after["nodes"]}
        destination_set = set(destination_slots)
        continued = source_slots[0] in previous_preferred
        local_streak = local_streak + 1 if continued else 0
        previous_preferred = (destination_set | predecessors) & live
        action = {
            "xfer_id": current.xfer_id,
            "source_id": int(rules.xfer_to_source[current.xfer_id]),
            "anchor_slot": int(source_slots[0]),
            "binding_slots": source_slots,
            "binding_guids": tuple(map(int, source_guids)),
            "dst_slots": destination_slots,
            "dst_guids": surviving_destination_guids,
            "dst_types": tuple(
                live_destination_guids[guid]
                for guid in surviving_destination_guids
            ),
            "declared_dst_guids": declared_destination_guids,
            "normalized_away_dst_guids": normalized_away_destination_guids,
            "effective_delta": delta,
        }
        trajectory["steps"].append(
            {
                "index": current.step,
                "graph_hash": int(graph.hash()),
                "num_nodes": int(graph.num_nodes),
                "local_streak": local_streak,
                "continued_local": int(continued),
                "match_ms": match_ms,
                "apply_ms": apply_ms,
                "saved_node_id": current.node_id,
                "resolved_node_id": resolved_node_id,
                "matches": matches,
                "action": action,
                "delta": delta,
            }
        )
        graph = next_graph

    finish_segment(trajectory, graph)
    return segments, skipped_transitions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert saved Quarl QASM paths into exact full-binding training data."
    )
    parser.add_argument("--trajectory-root", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-suffix",
        action="append",
        default=[],
        help="hold out a trajectory directory whose normalized path has this suffix",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat every accepted trajectory to upweight hard-prefix states",
    )
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--split-unrepresentable",
        action="store_true",
        help=(
            "keep exact representable segments on both sides of transitions "
            "that the incremental action format cannot encode"
        ),
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    reference = torch.load(args.reference_data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(reference)
    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    if context.num_xfers != len(rules.xfer_to_source):
        raise ValueError("ECC set and reference data have different xfer counts")
    xfers = context.get_xfers()
    directories = [
        path
        for path in discover_trajectory_directories(args.trajectory_root)
        if not excluded(path, args.exclude_suffix)
    ]
    if args.max_trajectories is not None:
        directories = directories[: args.max_trajectories]

    trajectories = []
    failures = []
    skipped_transitions = []
    accepted_paths = 0
    started = time.perf_counter()
    for path in directories:
        try:
            path_segments, path_skips = convert_trajectory(
                path,
                quartz=quartz,
                context=context,
                xfers=xfers,
                rules=rules,
                trajectory_id=len(trajectories),
                split_unrepresentable=args.split_unrepresentable,
            )
        except Exception as error:
            if args.strict:
                raise
            failures.append({"path": str(path), "error": str(error)})
            print(f"skipped={path} error={error}", flush=True)
            continue
        accepted_paths += 1
        skipped_transitions.extend(path_skips)
        for segment in path_segments:
            segment["trajectory_id"] = len(trajectories)
            trajectories.append(segment)
        print(
            f"collected_paths={accepted_paths}/{len(directories)} "
            f"segments={len(path_segments)} path={path} "
            f"steps={sum(len(row['steps']) for row in path_segments)} "
            f"omitted_transitions={len(path_skips)}",
            flush=True,
        )

    repeated = []
    for _ in range(args.repeat):
        for trajectory in trajectories:
            repeated.append(
                {
                    **trajectory,
                    "trajectory_id": len(repeated),
                }
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
    result["train_trajectories"] = repeated
    result["test_trajectories"] = []
    normalized_unique_actions = sum(
        bool(step["action"].get("normalized_away_dst_guids"))
        for trajectory in trajectories
        for step in trajectory["steps"]
    )
    result["metadata"] = {
        "kind": "quarl_saved_optimization_paths",
        "normalization_aware_actions": True,
        "rotation_elimination": True,
        "action_schema": "effective_delta_v1",
        "trajectory_roots": [str(path) for path in args.trajectory_root],
        "exclude_suffixes": args.exclude_suffix,
        "repeat": args.repeat,
        "discovered_trajectories": len(directories),
        "accepted_unique_paths": accepted_paths,
        "accepted_unique_trajectories": len(trajectories),
        "accepted_repeated_trajectories": len(repeated),
        "unique_prefix_states": sum(len(row["steps"]) for row in trajectories),
        "repeated_prefix_states": sum(len(row["steps"]) for row in repeated),
        "normalization_contracted_unique_actions": normalized_unique_actions,
        "failures": failures,
        "skipped_unrepresentable_transitions": skipped_transitions,
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(
        f"saved={args.output} unique_trajectories={len(trajectories)} "
        f"repeated_trajectories={len(repeated)} "
        f"prefix_states={result['metadata']['repeated_prefix_states']} "
        f"failures={len(failures)} seconds={result['metadata']['seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
