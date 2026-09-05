from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib.util
import random
from pathlib import Path
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


RZ_ANGLES = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 1.625, 1.75)


def random_qasm(rng: random.Random, qubits: int, gates: int) -> str:
    """Generate a rewrite-rich random circuit in the model's NAM gate set."""

    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{qubits}];",
    ]
    emitted = 0
    while emitted < gates:
        remaining = gates - emitted
        motif = rng.random()
        if motif < 0.12 and remaining >= 2:
            qubit = rng.randrange(qubits)
            gate = rng.choice(("h", "x"))
            lines.extend((f"{gate} q[{qubit}];", f"{gate} q[{qubit}];"))
            emitted += 2
        elif motif < 0.22 and remaining >= 2 and qubits >= 2:
            control, target = rng.sample(range(qubits), 2)
            line = f"cx q[{control}],q[{target}];"
            lines.extend((line, line))
            emitted += 2
        elif motif < 0.38 and remaining >= 2:
            qubit = rng.randrange(qubits)
            first = rng.choice(RZ_ANGLES)
            second = rng.choice(RZ_ANGLES)
            lines.extend(
                (
                    f"rz(pi*{first:.6f}) q[{qubit}];",
                    f"rz(pi*{second:.6f}) q[{qubit}];",
                )
            )
            emitted += 2
        else:
            choice = rng.random()
            if choice < 0.42 and qubits >= 2:
                control, target = rng.sample(range(qubits), 2)
                lines.append(f"cx q[{control}],q[{target}];")
            elif choice < 0.68:
                qubit = rng.randrange(qubits)
                lines.append(f"rz(pi*{rng.choice(RZ_ANGLES):.6f}) q[{qubit}];")
            elif choice < 0.86:
                lines.append(f"h q[{rng.randrange(qubits)}];")
            else:
                lines.append(f"x q[{rng.randrange(qubits)}];")
            emitted += 1
    return "\n".join(lines) + "\n"


def candidate_order(
    matches: list[dict],
    previous_preferred: set[int],
    local_probability: float,
    rng: random.Random,
) -> list[tuple[dict, int]]:
    rows = [
        (match, int(xfer_id))
        for match in matches
        for xfer_id in match["xfer_ids"]
    ]
    local = []
    distant = []
    for row in rows:
        destination = (
            local
            if int(row[0]["anchor_slot"]) in previous_preferred
            else distant
        )
        destination.append(row)
    rng.shuffle(local)
    rng.shuffle(distant)
    if local and rng.random() < local_probability:
        return local + distant
    return distant + local


def try_rewrite(
    *,
    graph,
    context,
    xfers,
    matches: list[dict],
    guid_to_slot: dict[int, int],
    next_slot: int,
    previous_preferred: set[int],
    local_probability: float,
    eliminate_rotation: bool,
    max_graph_gates: int,
    seen_hashes: set[int],
    max_trials: int,
    rng: random.Random,
):
    fallback = None
    for match, xfer_id in candidate_order(
        matches, previous_preferred, local_probability, rng
    )[:max_trials]:
        anchor = graph.get_node_from_id(id=int(match["anchor_id"]))
        result = graph.apply_xfer_with_binding_trace(
            xfer=xfers[xfer_id],
            node=anchor,
            eliminate_rotation=eliminate_rotation,
            predecessor_layers=1,
        )
        if result is None or result[0] is None:
            continue
        next_graph, _, source_guids, declared_destination_guids = result
        source_guids = tuple(map(int, source_guids))
        source_slots = tuple(guid_to_slot[guid] for guid in source_guids)
        if source_slots != tuple(map(int, match["binding_slots"])):
            continue
        if int(next_graph.gate_count) > max_graph_gates:
            continue

        declared_destination_guids = tuple(map(int, declared_destination_guids))
        live_destination_types = {
            int(node.guid): int(node.gate_tp) for node in next_graph.nodes
        }
        surviving_destination_guids = tuple(
            guid
            for guid in declared_destination_guids
            if guid in live_destination_types
        )
        normalized_away_destination_guids = tuple(
            guid
            for guid in declared_destination_guids
            if guid not in live_destination_types
        )
        candidate_mapping = dict(guid_to_slot)
        candidate_next_slot = next_slot
        destination_slots = []
        representable = True
        for guid in surviving_destination_guids:
            if guid in candidate_mapping:
                representable = False
                break
            candidate_mapping[guid] = candidate_next_slot
            destination_slots.append(candidate_next_slot)
            candidate_next_slot += 1
        if not representable:
            continue
        candidate_next_slot = update_slots(
            next_graph, candidate_mapping, candidate_next_slot
        )
        after = snapshot(next_graph, candidate_mapping)
        before = snapshot(graph, guid_to_slot)
        delta = graph_delta(before, after)
        if delta["removed_slots"] != sorted(source_slots):
            continue
        if {int(row[2]) for row in delta["added_nodes"]} != set(
            surviving_destination_guids
        ):
            continue

        destination_slots = tuple(destination_slots)
        graph_hash = int(next_graph.hash())
        choice = {
            "next_graph": next_graph,
            "mapping": candidate_mapping,
            "next_slot": candidate_next_slot,
            "source_slots": source_slots,
            "source_guids": source_guids,
            "destination_slots": destination_slots,
            "destination_guids": surviving_destination_guids,
            "normalized_away_destination_guids": (
                normalized_away_destination_guids
            ),
            "destination_types": tuple(
                live_destination_types[guid]
                for guid in surviving_destination_guids
            ),
            "xfer_id": xfer_id,
            "source_id": int(match["source_id"]),
            "delta": delta,
            "next_hash": graph_hash,
        }
        if graph_hash not in seen_hashes:
            return choice
        if fallback is None:
            fallback = choice
    return fallback


def collect_trajectory(
    *,
    quartz,
    context,
    xfers,
    rules: RuleMetadata,
    qasm: str,
    trajectory_id: int,
    target_actions: int,
    min_actions: int,
    local_probability: float,
    eliminate_rotation: bool,
    max_gate_growth: float,
    max_trials: int,
    rng: random.Random,
) -> dict | None:
    graph = quartz.PyGraph.from_qasm_str(context=context, qasm_str=qasm)
    initial_gates = int(graph.gate_count)
    max_graph_gates = max(initial_gates + 16, int(initial_gates * max_gate_growth))
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    trajectory = {
        "split": "train",
        "trajectory_id": trajectory_id,
        "circuit": f"random_{trajectory_id:06d}.qasm",
        "source_path": f"random://rewrite-rich/{trajectory_id:06d}",
        "initial_qasm": qasm,
        "initial_graph": snapshot(graph, guid_to_slot),
        "steps": [],
        "terminal_only_supervision": False,
    }
    previous_preferred: set[int] = set()
    local_streak = 0
    seen_hashes = {int(graph.hash())}
    for step_index in range(target_actions):
        before = snapshot(graph, guid_to_slot)
        matches, match_ms = enumerate_matches(
            graph, context, rules.xfer_to_source, guid_to_slot
        )
        if not matches:
            break
        apply_started = time.perf_counter()
        choice = try_rewrite(
            graph=graph,
            context=context,
            xfers=xfers,
            matches=matches,
            guid_to_slot=guid_to_slot,
            next_slot=next_slot,
            previous_preferred=previous_preferred,
            local_probability=local_probability,
            eliminate_rotation=eliminate_rotation,
            max_graph_gates=max_graph_gates,
            seen_hashes=seen_hashes,
            max_trials=max_trials,
            rng=rng,
        )
        apply_ms = (time.perf_counter() - apply_started) * 1000.0
        if choice is None:
            break

        source_slots = choice["source_slots"]
        destination_slots = choice["destination_slots"]
        continued = source_slots[0] in previous_preferred
        local_streak = local_streak + 1 if continued else 0
        source_set = set(source_slots)
        predecessors = {
            int(src)
            for src, dst, _, _ in before["edges"]
            if int(dst) in source_set
        }
        live = {
            int(row[0])
            for row in snapshot(choice["next_graph"], choice["mapping"])["nodes"]
        }
        previous_preferred = (set(destination_slots) | predecessors) & live
        action = {
            "xfer_id": choice["xfer_id"],
            "source_id": choice["source_id"],
            "anchor_slot": int(source_slots[0]),
            "binding_slots": source_slots,
            "binding_guids": choice["source_guids"],
            "dst_slots": destination_slots,
            "dst_guids": choice["destination_guids"],
            "dst_types": choice["destination_types"],
            "declared_dst_guids": (
                choice["destination_guids"]
                + choice["normalized_away_destination_guids"]
            ),
            "normalized_away_dst_guids": (
                choice["normalized_away_destination_guids"]
            ),
            "effective_delta": choice["delta"],
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
                "delta": choice["delta"],
            }
        )
        graph = choice["next_graph"]
        guid_to_slot = choice["mapping"]
        next_slot = choice["next_slot"]
        seen_hashes.add(choice["next_hash"])

    if len(trajectory["steps"]) < min_actions:
        return None
    terminal_matches, terminal_ms = enumerate_matches(
        graph, context, rules.xfer_to_source, guid_to_slot
    )
    trajectory["terminal_matches"] = terminal_matches
    trajectory["terminal_match_ms"] = terminal_ms
    trajectory["terminal_graph_hash"] = int(graph.hash())
    trajectory["requested_history_length"] = target_actions
    trajectory["valid_history_length"] = len(trajectory["steps"])
    validate_trajectory(trajectory, rules)
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random circuits and collect exact, locality-biased Quartz "
            "rewrite trajectories with complete match supervision."
        )
    )
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=256)
    parser.add_argument("--min-qubits", type=int, default=4)
    parser.add_argument("--max-qubits", type=int, default=12)
    parser.add_argument("--min-gates", type=int, default=48)
    parser.add_argument("--max-gates", type=int, default=192)
    parser.add_argument("--min-actions", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=48)
    parser.add_argument("--local-action-probability", type=float, default=0.8)
    parser.add_argument("--max-gate-growth", type=float, default=1.5)
    parser.add_argument("--max-trials-per-step", type=int, default=96)
    parser.add_argument("--max-attempts", type=int, default=2048)
    parser.add_argument("--eliminate-rotation", action="store_true")
    parser.add_argument("--seed", type=int, default=307)
    args = parser.parse_args()
    if not (
        0 < args.trajectories <= args.max_attempts
        and 1 <= args.min_qubits <= args.max_qubits
        and 1 <= args.min_gates <= args.max_gates
        and 1 <= args.min_actions <= args.max_actions
        and 0 <= args.local_action_probability <= 1
        and args.max_gate_growth >= 1
        and args.max_trials_per_step >= 1
    ):
        parser.error("invalid random trajectory generation limits")

    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    reference = torch.load(
        args.reference_data, map_location="cpu", weights_only=False
    )
    rules = RuleMetadata.from_payload(reference)
    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    xfers = context.get_xfers()
    rng = random.Random(args.seed)
    trajectories = []
    attempted = 0
    started = time.perf_counter()
    while len(trajectories) < args.trajectories and attempted < args.max_attempts:
        attempted += 1
        qubits = rng.randint(args.min_qubits, args.max_qubits)
        gates = rng.randint(args.min_gates, args.max_gates)
        actions = rng.randint(args.min_actions, args.max_actions)
        trajectory = collect_trajectory(
            quartz=quartz,
            context=context,
            xfers=xfers,
            rules=rules,
            qasm=random_qasm(rng, qubits, gates),
            trajectory_id=len(trajectories),
            target_actions=actions,
            min_actions=args.min_actions,
            local_probability=args.local_action_probability,
            eliminate_rotation=args.eliminate_rotation,
            max_gate_growth=args.max_gate_growth,
            max_trials=args.max_trials_per_step,
            rng=rng,
        )
        if trajectory is not None:
            trajectories.append(trajectory)
            if len(trajectories) % 10 == 0:
                print(
                    f"collected={len(trajectories)} attempted={attempted} "
                    f"states={sum(len(row['steps']) for row in trajectories)}",
                    flush=True,
                )

    if len(trajectories) < args.trajectories:
        raise RuntimeError(
            f"collected only {len(trajectories)}/{args.trajectories} trajectories "
            f"after {attempted} attempts"
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
        "kind": "random_rewrite_rich_exact_trajectories",
        "reference_data": str(args.reference_data),
        "ecc_file": str(args.ecc_file),
        "seed": args.seed,
        "attempted": attempted,
        "trajectories": len(trajectories),
        "states": sum(len(row["steps"]) for row in trajectories),
        "continued_local_states": sum(
            int(step["continued_local"])
            for row in trajectories
            for step in row["steps"]
        ),
        "arguments": vars(args),
        "collection_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(
        f"saved={args.output} trajectories={len(trajectories)} "
        f"states={result['metadata']['states']} "
        f"continued_local={result['metadata']['continued_local_states']} "
        f"attempted={attempted} seconds={result['metadata']['collection_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
