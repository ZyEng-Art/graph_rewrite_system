from __future__ import annotations

import argparse
from collections import defaultdict, deque
import ctypes
import ctypes.util
from dataclasses import dataclass
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types
from typing import Any

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch

from dataset import (
    _local_streak_bucket,
    _touch_age_bucket,
    compact_live_slots,
    RuleMetadata,
)
from model import S0ActionBindingModel
from threshold_inference import load_threshold_config, threshold_candidates
from train import autocast_context, move_batch


@dataclass
class BeamState:
    graph: Any
    snapshot: dict | None
    guid_to_slot: dict[int, int]
    next_slot: int
    last_touched: Any
    rewrite_distance: Any
    previous_preferred: set[int]
    local_streak: int
    gate_count: int
    depth: int
    history: tuple[tuple[int, int], ...]
    topology_index: Any = None
    exact_graph_checkpoint: Any = None
    exact_slot_checkpoint: dict[int, int] | None = None
    exact_checkpoint_depth: int = 0


@dataclass(frozen=True)
class Proposal:
    parent: int
    xfer_id: int
    anchor_slot: int
    binding: tuple[int, ...] | None
    probability: float
    next_gate_count: int
    value_score: float = 0.0


def update_slots(graph, guid_to_slot: dict[int, int], next_slot: int, preferred=()):
    live_guids = {int(node.guid) for node in graph.nodes}
    for raw_guid in preferred:
        guid = int(raw_guid)
        if guid not in live_guids:
            raise RuntimeError("destination GUID missing after rewrite")
        if guid not in guid_to_slot:
            guid_to_slot[guid] = next_slot
            next_slot += 1
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


def graph_delta(before: dict, after: dict) -> tuple[set[int], set[tuple[int, ...]]]:
    before_nodes = {int(row[0]) for row in before["nodes"]}
    after_nodes = {int(row[0]) for row in after["nodes"]}
    before_edges = set(map(tuple, before["edges"]))
    after_edges = set(map(tuple, after["edges"]))
    return before_nodes - after_nodes, before_edges.symmetric_difference(after_edges)


def distances_from_core(snapshot_row: dict, core: set[int]) -> dict[int, int]:
    live = {int(row[0]) for row in snapshot_row["nodes"]}
    adjacency = {slot: set() for slot in live}
    for src, dst, _, _ in snapshot_row["edges"]:
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    result = {slot: 0 for slot in core if slot in live}
    queue = deque(result)
    while queue:
        slot = queue.popleft()
        for neighbor in adjacency[slot]:
            if neighbor not in result:
                result[neighbor] = result[slot] + 1
                queue.append(neighbor)
    return {slot: min(result.get(slot, 5), 4) if slot in result else 5 for slot in live}


def collate_states(states: list[BeamState]) -> dict:
    max_slots = max(max((row[0] for row in state.snapshot["nodes"]), default=-1) + 1 for state in states)
    batch_size = len(states)
    current_types = torch.full((batch_size, max_slots), -1, dtype=torch.long)
    rewrite_distance = torch.full((batch_size, max_slots), 5, dtype=torch.long)
    touch_age = torch.full((batch_size, max_slots), 7, dtype=torch.long)
    streak = torch.zeros(batch_size, dtype=torch.long)
    edge_batch = []
    edge_src = []
    edge_dst = []
    edge_relation = []
    for batch_index, state in enumerate(states):
        for slot, gate_type, _ in state.snapshot["nodes"]:
            current_types[batch_index, slot] = gate_type
            rewrite_distance[batch_index, slot] = state.rewrite_distance.get(slot, 5)
            if slot in state.last_touched:
                age = state.depth - 1 - state.last_touched[slot]
                touch_age[batch_index, slot] = _touch_age_bucket(age)
        streak[batch_index] = _local_streak_bucket(bool(state.depth), state.local_streak)
        for src, dst, src_port, dst_port in state.snapshot["edges"]:
            edge_batch.append(batch_index)
            edge_src.append(src)
            edge_dst.append(dst)
            edge_relation.append(src_port * 4 + dst_port)
    return {
        "current_types": current_types,
        "current_live_slots": compact_live_slots(current_types),
        "current_edge_batch": torch.tensor(edge_batch, dtype=torch.long),
        "current_edge_src": torch.tensor(edge_src, dtype=torch.long),
        "current_edge_dst": torch.tensor(edge_dst, dtype=torch.long),
        "current_edge_relation": torch.tensor(edge_relation, dtype=torch.long),
        "current_rewrite_distance": rewrite_distance,
        "current_touch_age": touch_age,
        "current_local_streak": streak,
    }


def exact_actions(
    state: BeamState, context
) -> list[tuple[int, int, tuple[int, ...] | None, float]]:
    """Enumerate actions through Quartz's original xfer-at-anchor API."""
    rows = []
    for node in state.graph.nodes:
        anchor_slot = state.guid_to_slot[int(node.guid)]
        for xfer_id in state.graph.available_xfers_parallel(context=context, node=node):
            rows.append((int(xfer_id), anchor_slot, None, 1.0))
    return rows


@torch.no_grad()
def model_matches(
    states: list[BeamState],
    model,
    device,
    threshold_config,
    microbatch: int,
    max_candidates: int,
) -> tuple[list[list[tuple[int, int, tuple[int, ...], float]]], float]:
    output = []
    started = time.perf_counter()
    with autocast_context(device):
        source_vectors = model.retrieval_source(model.source_representations())
    for begin in range(0, len(states), microbatch):
        selected = states[begin : begin + microbatch]
        batch = move_batch(collate_states(selected), device)
        with autocast_context(device):
            encoded, live, gate_types = model.encode(batch)
            logits, eligible = model.match_logits(
                encoded, live, gate_types, source_vectors=source_vectors
            )
        output.extend(
            threshold_candidates(
                model,
                batch,
                logits,
                eligible,
                threshold_config,
                max_candidates_per_state=max_candidates,
            )
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - started


def make_child(
    parent: BeamState,
    proposal: Proposal,
    context,
    xfers,
    *,
    eliminate_rotation: bool = False,
) -> BeamState | None:
    slot_to_guid = {
        parent.guid_to_slot[int(node.guid)]: int(node.guid)
        for node in parent.graph.nodes
    }
    if proposal.anchor_slot not in slot_to_guid:
        return None
    guid_to_id = {int(node.guid): index for index, node in enumerate(parent.graph.nodes)}
    anchor_guid = slot_to_guid[proposal.anchor_slot]
    node = parent.graph.get_node_from_id(id=guid_to_id[anchor_guid])
    result = parent.graph.apply_xfer_with_binding_trace(
        xfer=xfers[proposal.xfer_id],
        node=node,
        eliminate_rotation=eliminate_rotation,
        predecessor_layers=1,
    )
    if result is None or result[0] is None:
        return None
    graph, _, source_guids, destination_guids = result
    source_slots = tuple(parent.guid_to_slot[int(guid)] for guid in source_guids)
    # Model proposals carry the complete source binding, so this is also the
    # ground-truth validity check. Original Quartz actions only carry an anchor.
    if proposal.binding is not None and source_slots != proposal.binding:
        return None

    live_guids = {int(node.guid) for node in graph.nodes}
    surviving_destination_guids = tuple(
        int(guid) for guid in destination_guids if int(guid) in live_guids
    )
    guid_to_slot = dict(parent.guid_to_slot)
    next_slot = update_slots(
        graph,
        guid_to_slot,
        parent.next_slot,
        surviving_destination_guids,
    )
    after = snapshot(graph, guid_to_slot)
    removed, changed_edges = graph_delta(parent.snapshot, after)
    live = {int(row[0]) for row in after["nodes"]}
    destination_slots = {
        guid_to_slot[guid] for guid in surviving_destination_guids
    }
    core = set(destination_slots)
    for src, dst, _, _ in changed_edges:
        if src in live:
            core.add(src)
        if dst in live:
            core.add(dst)
    last_touched = {
        slot: touched
        for slot, touched in parent.last_touched.items()
        if slot in live and slot not in removed
    }
    for slot in core:
        last_touched[slot] = parent.depth

    source_set = set(source_slots)
    predecessors = {
        src for src, dst, _, _ in parent.snapshot["edges"] if dst in source_set
    }
    previous_preferred = (destination_slots | predecessors) & live
    continued = proposal.anchor_slot in parent.previous_preferred
    local_streak = parent.local_streak + 1 if continued else 0
    gate_count = int(graph.gate_count)
    if not eliminate_rotation and gate_count != proposal.next_gate_count:
        raise RuntimeError(
            f"gate delta mismatch: expected {proposal.next_gate_count}, got {gate_count}"
        )
    return BeamState(
        graph=graph,
        snapshot=after,
        guid_to_slot=guid_to_slot,
        next_slot=next_slot,
        last_touched=last_touched,
        rewrite_distance=distances_from_core(after, core),
        previous_preferred=previous_preferred,
        local_streak=local_streak,
        gate_count=gate_count,
        depth=parent.depth + 1,
        history=parent.history + ((proposal.xfer_id, proposal.anchor_slot),),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("model", "quartz"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--target-recall", type=float, default=0.97)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--max-source-matches", type=int, default=2048)
    parser.add_argument("--max-actions-per-parent", type=int, default=128)
    parser.add_argument("--proposal-factor", type=int, default=8)
    parser.add_argument("--max-gate-increase", type=int, default=1)
    parser.add_argument("--refresh-interval", type=int, default=0)
    parser.add_argument("--refresh-count", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-qasm", type=Path)
    parser.add_argument(
        "--eliminate-rotation",
        action="store_true",
        help="fold parameter expressions and remove zero rotations after each Quartz rewrite",
    )
    args = parser.parse_args()
    if args.mode == "model" and (args.checkpoint is None or args.calibration is None):
        parser.error("model mode requires --checkpoint and --calibration")

    # Quartz imports these optional conversion packages unconditionally, while
    # this benchmark uses only the compiled graph API.
    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    if context.num_xfers != len(payload["xfer_to_source"]):
        raise RuntimeError("dataset and Quartz context have different xfer counts")
    xfers = [context.get_xfer_from_id(id=index) for index in range(context.num_xfers)]
    for index, xfer in enumerate(xfers):
        if xfer.src_str.strip() != payload["xfer_sources"][index].strip():
            raise RuntimeError(f"xfer source mismatch at {index}")
    source_to_xfers: dict[int, list[int]] = defaultdict(list)
    for xfer_id, source_id in enumerate(payload["xfer_to_source"]):
        source_to_xfers[int(source_id)].append(xfer_id)
    gate_deltas = [int(xfer.dst_gate_count - xfer.src_gate_count) for xfer in xfers]

    model = None
    threshold_config = None
    if args.mode == "model":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        train_args = checkpoint["args"]
        model = S0ActionBindingModel(
            rules,
            num_xfers=len(payload["xfer_to_source"]),
            width=train_args["width"],
            retrieval_width=train_args["retrieval_width"],
            graph_layers=train_args["graph_layers"],
            current_graph_layers=train_args["current_graph_layers"],
            use_action_history=not train_args.get("state_only", False),
            use_locality_features=train_args.get("locality_features", False),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        threshold_config = load_threshold_config(args.calibration, args.target_recall)

    graph = quartz.PyGraph.from_qasm(context=context, filename=str(args.qasm))
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    initial_snapshot = snapshot(graph, guid_to_slot)
    beam = [
        BeamState(
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
    ]
    if args.mode == "model":
        model_matches(
            beam,
            model,
            device,
            threshold_config,
            args.microbatch,
            args.max_source_matches,
        )
    else:
        graph.available_xfers_parallel(context=context, node=graph.nodes[0])
    del payload
    gc.collect()
    gc.disable()
    initial_gate_count = beam[0].gate_count
    seen = {int(graph.hash())}
    step_rows = []
    total_started = time.perf_counter()
    for step in range(args.depth):
        step_started = time.perf_counter()
        model_seconds = exact_seconds = 0.0
        exact_refresh_actions_added = 0
        if args.mode == "model":
            predicted, model_seconds = model_matches(
                beam,
                model,
                device,
                threshold_config,
                args.microbatch,
                args.max_source_matches,
            )
            action_rows: list[
                list[tuple[int, int, tuple[int, ...] | None, float]]
            ] = []
            for rows in predicted:
                expanded = []
                for source, anchor, binding, probability in rows:
                    expanded.extend(
                        (xfer_id, anchor, binding, probability)
                        for xfer_id in source_to_xfers[source]
                    )
                action_rows.append(expanded)
            refresh = (
                args.refresh_interval > 0
                and step > 0
                and step % args.refresh_interval == 0
            )
            if refresh:
                for parent_index in range(min(args.refresh_count, len(beam))):
                    started = time.perf_counter()
                    exact_rows = exact_actions(beam[parent_index], context)
                    exact_seconds += time.perf_counter() - started
                    # Keep the model's ranking for candidates it already found.
                    # Exact refresh only adds missed xfer-at-anchor actions, at
                    # lower tie-break priority within the same gate-count band.
                    present = {
                        (int(xfer_id), int(anchor))
                        for xfer_id, anchor, _, _ in action_rows[parent_index]
                    }
                    additions = [
                        (xfer_id, anchor, binding, 0.0)
                        for xfer_id, anchor, binding, _ in exact_rows
                        if (int(xfer_id), int(anchor)) not in present
                    ]
                    exact_refresh_actions_added += len(additions)
                    action_rows[parent_index].extend(additions)
        else:
            action_rows = []
            for state in beam:
                started = time.perf_counter()
                action_rows.append(exact_actions(state, context))
                exact_seconds += time.perf_counter() - started

        proposal_started = time.perf_counter()
        proposals = []
        effective_parent_cap = max(
            args.max_actions_per_parent,
            math.ceil(args.beam_size / max(1, len(beam))) * 2,
        )
        total_action_candidates = 0
        for parent_index, (state, rows) in enumerate(zip(beam, action_rows)):
            parent_proposals = []
            for xfer_id, anchor, binding, probability in rows:
                delta = gate_deltas[xfer_id]
                if delta > args.max_gate_increase:
                    continue
                parent_proposals.append(
                    Proposal(
                        parent=parent_index,
                        xfer_id=xfer_id,
                        anchor_slot=anchor,
                        binding=binding,
                        probability=probability,
                        next_gate_count=state.gate_count + delta,
                    )
                )
            total_action_candidates += len(parent_proposals)
            parent_proposals.sort(
                key=lambda row: (row.next_gate_count, -row.probability, row.xfer_id)
            )
            proposals.extend(parent_proposals[:effective_parent_cap])
        proposals.sort(
            key=lambda row: (
                row.next_gate_count,
                -row.probability,
                beam[row.parent].gate_count,
            )
        )
        proposals = proposals[: args.beam_size * args.proposal_factor]
        proposal_seconds = time.perf_counter() - proposal_started

        apply_started = time.perf_counter()
        children = []
        attempted = invalid = duplicates = 0
        for proposal in proposals:
            if len(children) >= args.beam_size:
                break
            attempted += 1
            child = make_child(
                beam[proposal.parent],
                proposal,
                context,
                xfers,
                eliminate_rotation=args.eliminate_rotation,
            )
            if child is None:
                invalid += 1
                continue
            graph_hash = int(child.graph.hash())
            if graph_hash in seen:
                duplicates += 1
                continue
            seen.add(graph_hash)
            children.append(child)
        apply_seconds = time.perf_counter() - apply_started
        if not children:
            break
        children.sort(key=lambda state: (state.gate_count, len(state.history)))
        beam = children[: args.beam_size]
        elapsed = time.perf_counter() - step_started
        matched_action_count = sum(map(len, action_rows))
        match_seconds = model_seconds + exact_seconds
        successful_applies = attempted - invalid
        row = {
            "step": step + 1,
            "input_states": len(action_rows),
            "output_states": len(beam),
            "best_gate_count": beam[0].gate_count,
            "predicted_or_exact_actions": matched_action_count,
            "eligible_actions_before_parent_cap": total_action_candidates,
            "proposals_after_caps": len(proposals),
            "attempted_actions": attempted,
            "accepted_actions": len(beam),
            "invalid_model_actions": invalid,
            "duplicate_successors": duplicates,
            "model_match_seconds": model_seconds,
            "quartz_exact_match_seconds": exact_seconds,
            "exact_refresh_actions_added": exact_refresh_actions_added,
            "proposal_seconds": proposal_seconds,
            "quartz_apply_seconds": apply_seconds,
            "total_seconds": elapsed,
            "accepted_actions_per_second": len(beam) / elapsed,
            "match_states_per_second": len(action_rows) / max(1e-12, match_seconds),
            "matched_actions_per_second": matched_action_count
            / max(1e-12, match_seconds),
            "successful_apply_actions": successful_applies,
            "successful_apply_actions_per_second": successful_applies
            / max(1e-12, apply_seconds),
        }
        step_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    total_seconds = time.perf_counter() - total_started
    total_accepted = sum(row["accepted_actions"] for row in step_rows)
    total_match_seconds = sum(
        row["model_match_seconds"] + row["quartz_exact_match_seconds"]
        for row in step_rows
    )
    total_matched_actions = sum(
        row["predicted_or_exact_actions"] for row in step_rows
    )
    total_input_states = sum(row["input_states"] for row in step_rows)
    total_apply_seconds = sum(row["quartz_apply_seconds"] for row in step_rows)
    total_successful_applies = sum(
        row["successful_apply_actions"] for row in step_rows
    )
    result = {
        "mode": args.mode,
        "eliminate_rotation": args.eliminate_rotation,
        "qasm": str(args.qasm),
        "beam_size": args.beam_size,
        "requested_depth": args.depth,
        "completed_depth": len(step_rows),
        "initial_gate_count": initial_gate_count,
        "best_gate_count": min(state.gate_count for state in beam),
        "final_beam_size": len(beam),
        "unique_graphs_seen": len(seen),
        "total_seconds": total_seconds,
        "accepted_actions": total_accepted,
        "accepted_actions_per_second": total_accepted / total_seconds,
        "match_states_per_second": total_input_states
        / max(1e-12, total_match_seconds),
        "matched_actions_per_second": total_matched_actions
        / max(1e-12, total_match_seconds),
        "successful_apply_actions_per_second": total_successful_applies
        / max(1e-12, total_apply_seconds),
        "refresh_interval": args.refresh_interval,
        "refresh_count": args.refresh_count,
        "target_recall": args.target_recall if args.mode == "model" else None,
        "steps": step_rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    if args.best_qasm is not None:
        args.best_qasm.parent.mkdir(parents=True, exist_ok=True)
        beam[0].graph.to_qasm(filename=str(args.best_qasm))


if __name__ == "__main__":
    main()
