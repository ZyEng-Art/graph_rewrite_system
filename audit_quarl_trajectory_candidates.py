from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import ctypes.util
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import types

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch

from beam_search_benchmark import BeamState, Proposal, snapshot, update_slots
from dataset import RuleMetadata
from gpu_proposals import GpuRuleIndex, build_gpu_proposals
from incremental_graph import parse_pattern
from lazy_rollout_benchmark import indexed_topology, lazy_child
from model_factory import build_model
from paged_cache import PagedKVCache
from paged_rollout_benchmark import advance_selected, paged_model_matches
from threshold_inference import load_threshold_config
from train import autocast_context, move_batch
from train_paged_ppo import initial_batch_many


TRAJECTORY_FILE = re.compile(
    r"^(?P<step>\d+)_(?P<cost>-?\d+)_(?P<reward>-?\d+)_"
    r"(?P<node>\d+)_(?P<xfer>\d+)\.qasm$"
)


@dataclass(frozen=True)
class TrajectoryAction:
    trajectory: str
    step: int
    cost: int
    reward: int
    node_id: int
    xfer_id: int
    state_qasm: Path
    next_qasm: Path


def parse_trajectory_directory(path: Path) -> list[TrajectoryAction]:
    rows: list[tuple[int, int, int, int, int, Path]] = []
    for qasm in path.glob("*.qasm"):
        match = TRAJECTORY_FILE.fullmatch(qasm.name)
        if match is None:
            raise ValueError(f"unexpected trajectory filename: {qasm}")
        rows.append(
            (
                int(match["step"]),
                int(match["cost"]),
                int(match["reward"]),
                int(match["node"]),
                int(match["xfer"]),
                qasm,
            )
        )
    rows.sort(key=lambda row: row[0])
    if len(rows) < 2:
        raise ValueError(f"trajectory needs at least one action and terminal state: {path}")
    steps = [row[0] for row in rows]
    if steps != list(range(len(rows))):
        raise ValueError(f"trajectory steps are not contiguous in {path}: {steps}")
    terminal = rows[-1]
    if terminal[3:] and (terminal[3], terminal[4]) != (0, 0):
        raise ValueError(
            f"terminal trajectory row must contain the (node=0, xfer=0) sentinel: "
            f"{terminal[-1]}"
        )
    return [
        TrajectoryAction(
            trajectory=str(path),
            step=row[0],
            cost=row[1],
            reward=row[2],
            node_id=row[3],
            xfer_id=row[4],
            state_qasm=row[5],
            next_qasm=rows[index + 1][5],
        )
        for index, row in enumerate(rows[:-1])
    ]


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_actions(rows: list[dict], caps: list[int]) -> dict:
    ranks = [int(row["gate_rank"]) for row in rows if row["gate_rank"] is not None]
    summary = {
        "actions": len(rows),
        "exact_action_available": sum(bool(row["exact_action_available"]) for row in rows),
        "exact_transition_matches": sum(bool(row["exact_transition_matches"]) for row in rows),
        "source_match_covered": sum(bool(row["source_match_covered"]) for row in rows),
        "ranked_action_covered": len(ranks),
        "missing_rank_steps": [
            int(row["step"]) for row in rows if row["gate_rank"] is None
        ],
        "gate_rank": {
            "min": min(ranks) if ranks else None,
            "median": percentile(ranks, 0.5),
            "p90": percentile(ranks, 0.9),
            "p95": percentile(ranks, 0.95),
            "max": max(ranks) if ranks else None,
        },
        "caps": {},
    }
    for cap in caps:
        covered = sum(
            row["gate_rank"] is not None and int(row["gate_rank"]) <= cap
            for row in rows
        )
        summary["caps"][str(cap)] = {
            "covered": covered,
            "recall": covered / max(1, len(rows)),
            "whole_trajectory_covered": covered == len(rows),
            "missing_steps": [
                int(row["step"])
                for row in rows
                if row["gate_rank"] is None or int(row["gate_rank"]) > cap
            ],
        }
    return summary


def graph_hash(graph) -> int:
    return int(graph.hash())


def resolve_transition_anchors(graph, target, context, xfer, node_id: int, xfer_id: int):
    """Recover anchors after QASM serialization has reordered independent gates."""
    target_hash = graph_hash(target)
    declared_node = graph.get_node_from_id(id=node_id)
    declared_available = xfer_id in {
        int(row)
        for row in graph.available_xfers_parallel(context=context, node=declared_node)
    }
    declared_result = (
        graph.apply_xfer(xfer=xfer, node=declared_node, eliminate_rotation=True)
        if declared_available
        else None
    )
    declared_matches = (
        declared_result is not None and graph_hash(declared_result) == target_hash
    )
    if declared_matches:
        return [declared_node], True, declared_available, True

    matching_nodes = []
    any_available = declared_available
    for node in graph.nodes:
        if int(node.guid) == int(declared_node.guid):
            continue
        available = graph.available_xfers_parallel(context=context, node=node)
        if xfer_id not in {int(row) for row in available}:
            continue
        any_available = True
        result = graph.apply_xfer(xfer=xfer, node=node, eliminate_rotation=True)
        if result is not None and graph_hash(result) == target_hash:
            matching_nodes.append(node)
    return matching_nodes, any_available, declared_available, False


def audit_sequence_conditioned(
    trajectory_paths,
    *,
    quartz,
    context,
    xfers,
    rules,
    rule_index,
    gate_deltas,
    model,
    device,
    threshold_config,
    source_vectors,
    source_patterns,
    destination_patterns,
    args,
    caps,
):
    """Force the saved actions while retaining the model's causal action prefix."""

    def initialize_sequence_state(current_graph, cache_capacity: int):
        slot_map: dict[int, int] = {}
        next_free_slot = update_slots(current_graph, slot_map, 0)
        initial = snapshot(current_graph, slot_map)
        current_state = BeamState(
            graph=None,
            snapshot=initial,
            guid_to_slot={},
            next_slot=next_free_slot,
            last_touched={},
            rewrite_distance={int(node[0]): 5 for node in initial["nodes"]},
            previous_preferred=set(),
            local_streak=0,
            gate_count=int(current_graph.gate_count),
            depth=0,
            history=(),
            topology_index=indexed_topology(initial),
            exact_graph_checkpoint=current_graph,
            exact_slot_checkpoint=dict(slot_map),
            exact_checkpoint_depth=0,
        )
        with torch.no_grad(), autocast_context(device):
            current_states, current_live, current_gate_types = (
                model.initialize_incremental(
                    move_batch(initial_batch_many([initial]), device)
                )
            )
        current_arena = PagedKVCache(
            layers=model.action_layers_count,
            capacity=cache_capacity,
            page_size=8,
            heads=model.action_heads,
            head_width=model.width // model.action_heads,
            model_width=model.width,
            device=device,
            dtype=(
                torch.bfloat16 if device.type == "cuda" else current_states.dtype
            ),
            gather_backend="vectorized",
        )
        return (
            current_state,
            slot_map,
            next_free_slot,
            current_states,
            current_live,
            current_gate_types,
            current_arena,
            [current_arena.empty_handle()],
        )

    output = {}
    for trajectory_path in trajectory_paths:
        actions = parse_trajectory_directory(trajectory_path)
        graph = quartz.PyGraph.from_qasm(
            context=context, filename=str(actions[0].state_qasm)
        )
        cache_capacity = max(
            8,
            min(len(actions), args.sequence_window or len(actions)) + 2,
        )
        (
            state,
            guid_to_slot,
            next_slot,
            states,
            live,
            gate_types,
            arena,
            handles,
        ) = initialize_sequence_state(graph, cache_capacity)
        details = []
        probe_details = []
        window_start = 0
        for action_index, action in enumerate(actions):
            if (
                args.sequence_window
                and action_index
                and action_index % args.sequence_window == 0
            ):
                for handle in handles:
                    arena.release(handle)
                (
                    state,
                    guid_to_slot,
                    next_slot,
                    states,
                    live,
                    gate_types,
                    arena,
                    handles,
                ) = initialize_sequence_state(graph, cache_capacity)
                window_start = action_index
            target = quartz.PyGraph.from_qasm(
                context=context, filename=str(action.next_qasm)
            )
            matching_nodes, available, declared_available, declared_matches = (
                resolve_transition_anchors(
                    graph,
                    target,
                    context,
                    xfers[action.xfer_id],
                    action.node_id,
                    action.xfer_id,
                )
            )
            if not matching_nodes:
                raise RuntimeError(
                    f"forced sequence action no longer reaches target at step {action.step}"
                )
            trace = graph.apply_xfer_with_binding_trace(
                xfer=xfers[action.xfer_id],
                node=matching_nodes[0],
                eliminate_rotation=True,
                predecessor_layers=1,
            )
            if trace is None or trace[0] is None:
                raise RuntimeError(f"missing binding trace at step {action.step}")
            next_graph, _, source_guids, destination_guids = trace
            if graph_hash(next_graph) != graph_hash(target):
                raise RuntimeError(f"binding trace reaches wrong graph at step {action.step}")
            source_slots = tuple(guid_to_slot[int(guid)] for guid in source_guids)
            if not source_slots:
                raise RuntimeError(f"empty source binding at step {action.step}")

            with torch.no_grad():
                candidates, _, _, encoded = paged_model_matches(
                    [state],
                    states,
                    live,
                    gate_types,
                    handles,
                    arena,
                    model,
                    device,
                    threshold_config,
                    source_vectors,
                    microbatch=args.microbatch,
                    max_candidates=args.max_source_matches,
                    near_source_reserve=args.near_source_reserve,
                    source_microbatch=args.source_microbatch,
                    source_grouping=args.source_grouping,
                    state_batch_backend="tensorized",
                    candidate_backend="gpu",
                    return_encoded_states=True,
                    profile_stages=False,
                )
                proposals, _, _, _ = build_gpu_proposals(
                    candidates,
                    [state],
                    rule_index,
                    per_parent_cap=args.max_action_rank,
                    global_cap=args.max_action_rank,
                    ranking_mode=args.ranking_mode,
                    action_value_model=(
                        model if args.ranking_mode == "value" else None
                    ),
                    action_value_states=(
                        encoded if args.ranking_mode == "value" else None
                    ),
                    action_value_live=(
                        live if args.ranking_mode == "value" else None
                    ),
                    action_value_weight=args.action_value_weight,
                )
            source_id = int(rules.xfer_to_source[action.xfer_id])
            anchor_slot = int(source_slots[0])
            with autocast_context(device):
                raw_logits, eligible = model.match_logits(
                    encoded, live, gate_types, source_vectors=source_vectors
                )
            raw_logit = float(raw_logits[0, anchor_slot, source_id].float().item())
            match_group = (
                "near" if state.rewrite_distance.get(anchor_slot, 5) <= 2 else "far"
            )
            threshold_group = threshold_config["groups"][match_group]
            calibrated_logit = (
                raw_logit * threshold_group["scale"] + threshold_group["bias"]
            )
            calibrated_probability = float(torch.sigmoid(torch.tensor(calibrated_logit)))
            raw_threshold = float(threshold_group["raw_threshold"])
            near_slots = torch.tensor(
                [
                    state.rewrite_distance.get(slot, 5) <= 2
                    for slot in range(raw_logits.shape[1])
                ],
                dtype=torch.bool,
                device=device,
            )
            near_group = threshold_config["groups"]["near"]
            far_group = threshold_config["groups"]["far"]
            slot_scales = torch.where(
                near_slots,
                torch.tensor(near_group["scale"], device=device),
                torch.tensor(far_group["scale"], device=device),
            )
            slot_biases = torch.where(
                near_slots,
                torch.tensor(near_group["bias"], device=device),
                torch.tensor(far_group["bias"], device=device),
            )
            all_calibrated_logits = (
                raw_logits.float() * slot_scales.view(1, -1, 1)
                + slot_biases.view(1, -1, 1)
            )
            target_logit = all_calibrated_logits[0, anchor_slot, source_id]
            unfiltered_probability_rank = int(
                (
                    all_calibrated_logits.masked_fill(~eligible, -torch.inf)
                    > target_logit
                )
                .sum()
                .item()
                + 1
            )
            candidate_keys = {
                (
                    int(source),
                    int(anchor),
                    tuple(int(slot) for slot in binding if int(slot) >= 0),
                )
                for source, anchor, binding in zip(
                    candidates.sources.cpu().tolist(),
                    candidates.anchors.cpu().tolist(),
                    candidates.bindings.cpu().tolist(),
                )
            }
            candidate_anchor_keys = {
                (source, anchor) for source, anchor, _ in candidate_keys
            }
            slot_to_guid = {
                slot: int(guid)
                for guid, slot in guid_to_slot.items()
                if any(int(node.guid) == int(guid) for node in graph.nodes)
            }
            guid_to_node_id = {
                int(node.guid): index for index, node in enumerate(graph.nodes)
            }
            for probe in args.probe_xfer_anchor:
                probe_xfer, probe_slot = map(int, probe.split(":"))
                probe_source = int(rules.xfer_to_source[probe_xfer])
                probe_live = probe_slot in slot_to_guid
                probe_available = False
                if probe_live:
                    probe_node = graph.get_node_from_id(
                        id=guid_to_node_id[slot_to_guid[probe_slot]]
                    )
                    probe_available = probe_xfer in {
                        int(value)
                        for value in graph.available_xfers_parallel(
                            context=context, node=probe_node
                        )
                    }
                probe_group = (
                    "near"
                    if state.rewrite_distance.get(probe_slot, 5) <= 2
                    else "far"
                )
                probe_config = threshold_config["groups"][probe_group]
                probe_raw = (
                    float(raw_logits[0, probe_slot, probe_source].float().item())
                    if probe_live
                    else None
                )
                probe_calibrated = (
                    probe_raw * probe_config["scale"] + probe_config["bias"]
                    if probe_raw is not None
                    else None
                )
                probe_details.append(
                    {
                        "before_step": action.step,
                        "cost": action.cost,
                        "xfer_id": probe_xfer,
                        "source_id": probe_source,
                        "anchor_slot": probe_slot,
                        "match_group": probe_group,
                        "quartz_available": probe_available,
                        "matcher_eligible": bool(
                            probe_live
                            and eligible[0, probe_slot, probe_source].item()
                        ),
                        "matcher_raw_logit": probe_raw,
                        "matcher_raw_threshold": float(
                            probe_config["raw_threshold"]
                        ),
                        "matcher_calibrated_probability": (
                            float(torch.sigmoid(torch.tensor(probe_calibrated)))
                            if probe_calibrated is not None
                            else None
                        ),
                        "source_match_covered": (
                            probe_source,
                            probe_slot,
                        )
                        in candidate_anchor_keys,
                    }
                )
            source_covered = any(
                int(parent) == 0
                and int(source) == source_id
                and int(anchor) == anchor_slot
                and tuple(int(slot) for slot in binding if int(slot) >= 0)
                == source_slots
                for parent, source, anchor, binding in zip(
                    candidates.batch_ids.cpu().tolist(),
                    candidates.sources.cpu().tolist(),
                    candidates.anchors.cpu().tolist(),
                    candidates.bindings.cpu().tolist(),
                )
            )
            rank = next(
                (
                    index
                    for index, proposal in enumerate(proposals, start=1)
                    if proposal.xfer_id == action.xfer_id
                    and proposal.anchor_slot == anchor_slot
                    and proposal.binding == source_slots
                ),
                None,
            )
            details.append(
                {
                    "trajectory": str(trajectory_path),
                    "step": action.step,
                    "sequence_window_start": window_start,
                    "cost": action.cost,
                    "reward": action.reward,
                    "node_id": action.node_id,
                    "anchor_slots": [anchor_slot],
                    "matching_anchor_count": len(matching_nodes),
                    "declared_anchor_available": declared_available,
                    "declared_anchor_matches": declared_matches,
                    "xfer_id": action.xfer_id,
                    "source_id": source_id,
                    "gate_delta": gate_deltas[action.xfer_id],
                    "source_candidates": int(candidates.sources.numel()),
                    "exact_action_available": available,
                    "exact_transition_matches": True,
                    "source_match_covered": source_covered,
                    "gate_rank": rank,
                    "match_group": match_group,
                    "matcher_raw_logit": raw_logit,
                    "matcher_raw_threshold": raw_threshold,
                    "matcher_above_threshold": raw_logit >= raw_threshold,
                    "matcher_calibrated_probability": calibrated_probability,
                    "matcher_eligible": bool(eligible[0, anchor_slot, source_id].item()),
                    "matcher_unfiltered_probability_rank": (
                        unfiltered_probability_rank
                    ),
                }
            )

            forced = Proposal(
                parent=0,
                xfer_id=action.xfer_id,
                anchor_slot=anchor_slot,
                binding=source_slots,
                probability=1.0,
                next_gate_count=int(next_graph.gate_count),
            )
            child, _, duplicate = lazy_child(
                state,
                forced,
                source_patterns,
                destination_patterns,
                structural_recheck=True,
                dedup_mode="none",
                seen=set(),
                topology_backend="indexed",
            )
            if child is None or duplicate:
                raise RuntimeError(f"lazy forced action failed at step {action.step}")
            destination_slots = child.history[-1].destination_slots
            if len(destination_slots) != len(destination_guids):
                raise RuntimeError(
                    f"destination binding length differs at step {action.step}"
                )
            child_slot_map = dict(guid_to_slot)
            for guid, slot in zip(destination_guids, destination_slots):
                child_slot_map[int(guid)] = int(slot)
            states, live, gate_types, handles, _, _ = advance_selected(
                states,
                live,
                gate_types,
                handles,
                [(child, forced)],
                rules,
                arena,
                model,
                device,
                microbatch=1,
            )
            state = child
            graph = next_graph
            guid_to_slot = child_slot_map
        for handle in handles:
            arena.release(handle)
        output[str(trajectory_path)] = {
            "summary": summarize_actions(details, caps),
            "actions": details,
            "probes": probe_details,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether retained matcher candidates cover Quarl actions."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-source-matches", type=int, default=8192)
    parser.add_argument(
        "--near-source-reserve",
        type=int,
        default=0,
        help=(
            "union the standard candidates with this many unthresholded "
            "near-anchor candidates before structural deduplication"
        ),
    )
    parser.add_argument("--max-action-rank", type=int, default=512)
    parser.add_argument("--caps", default="64,128,256,512")
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--source-microbatch", type=int, default=0)
    parser.add_argument(
        "--source-grouping", choices=("none", "first_gate"), default="first_gate"
    )
    parser.add_argument("--max-gate-increase", type=int, default=3)
    parser.add_argument(
        "--ranking-mode",
        choices=("gate", "probability", "value"),
        default="gate",
        help="ordering used before applying the candidate cap",
    )
    parser.add_argument("--action-value-weight", type=float, default=1.0)
    parser.add_argument(
        "--sequence-conditioned",
        action="store_true",
        help="also force each trajectory while retaining the causal action prefix",
    )
    parser.add_argument(
        "--sequence-window",
        type=int,
        default=0,
        help=(
            "reset the causal prefix every N forced actions; use the model's "
            "training window for trajectories longer than max sequence length"
        ),
    )
    parser.add_argument(
        "--probe-xfer-anchor",
        action="append",
        default=[],
        metavar="XFER:SLOT",
        help="record one fixed xfer/anchor score before every forced sequence step",
    )
    args = parser.parse_args()

    caps = sorted({int(value) for value in args.caps.split(",")})
    if not caps or caps[0] < 1:
        parser.error("candidate caps must be positive")
    if args.max_action_rank < caps[-1]:
        parser.error("--max-action-rank must be at least the largest candidate cap")
    if args.max_source_matches < 1 or args.microbatch < 1:
        parser.error("source-match and microbatch limits must be positive")
    if args.near_source_reserve < 0:
        parser.error("near source reserve must be nonnegative")
    if args.sequence_window < 0:
        parser.error("--sequence-window must be nonnegative")

    actions = [
        action
        for trajectory in args.trajectory
        for action in parse_trajectory_directory(trajectory)
    ]
    if not actions:
        parser.error("no trajectory actions were found")

    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    if train_args.get("architecture") != "paged_action":
        raise ValueError("trajectory audit requires a paged_action checkpoint")
    model = build_model(rules, len(rules.xfer_to_source), train_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.readout_attention_backend = "sdpa"
    threshold_config = load_threshold_config(
        args.calibration, args.target_recall
    )
    with torch.no_grad(), autocast_context(device):
        source_vectors = model.retrieval_source(model.source_representations())

    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    if context.num_xfers != len(rules.xfer_to_source):
        raise RuntimeError("dataset and Quartz context have different xfer counts")
    xfers = [context.get_xfer_from_id(id=index) for index in range(context.num_xfers)]
    gate_deltas = [
        len(rules.destination_gate_types[index])
        - len(rules.source_gate_types[rules.xfer_to_source[index]])
        for index in range(len(rules.xfer_to_source))
    ]
    source_patterns = tuple(parse_pattern(row) for row in rules.xfer_sources)
    destination_patterns = tuple(
        parse_pattern(row) for row in rules.xfer_destinations
    )
    source_to_xfers: dict[int, list[int]] = defaultdict(list)
    for xfer_id, source_id in enumerate(rules.xfer_to_source):
        source_to_xfers[source_id].append(xfer_id)
    rule_index = GpuRuleIndex.build(
        source_to_xfers,
        gate_deltas,
        len(rules.source_gate_types),
        args.max_gate_increase,
        device,
    )

    graphs = []
    snapshots = []
    beam = []
    anchor_slots: list[list[int]] = []
    exact_binding_slots: list[list[tuple[int, ...]]] = []
    exact_available = []
    exact_matches = []
    declared_anchor_available = []
    declared_anchor_matches = []
    load_started = time.perf_counter()
    for action in actions:
        graph = quartz.PyGraph.from_qasm(context=context, filename=str(action.state_qasm))
        target = quartz.PyGraph.from_qasm(context=context, filename=str(action.next_qasm))
        (
            matching_nodes,
            action_available,
            declared_available,
            declared_matches,
        ) = resolve_transition_anchors(
            graph,
            target,
            context,
            xfers[action.xfer_id],
            action.node_id,
            action.xfer_id,
        )
        exact_available.append(action_available)
        exact_matches.append(bool(matching_nodes))
        declared_anchor_available.append(declared_available)
        declared_anchor_matches.append(declared_matches)

        guid_to_slot: dict[int, int] = {}
        next_slot = update_slots(graph, guid_to_slot, 0)
        row = snapshot(graph, guid_to_slot)
        topology = indexed_topology(row)
        graphs.append(graph)
        snapshots.append(row)
        anchor_slots.append(
            [guid_to_slot[int(matching_node.guid)] for matching_node in matching_nodes]
        )
        row_bindings = []
        for matching_node in matching_nodes:
            trace = graph.apply_xfer_with_binding_trace(
                xfer=xfers[action.xfer_id],
                node=matching_node,
                eliminate_rotation=True,
                predecessor_layers=1,
            )
            if trace is None or trace[0] is None:
                raise RuntimeError(
                    f"missing binding trace at {action.trajectory} step {action.step}"
                )
            traced_graph, _, source_guids, _ = trace
            if graph_hash(traced_graph) != graph_hash(target):
                raise RuntimeError(
                    f"binding trace reaches wrong graph at {action.trajectory} "
                    f"step {action.step}"
                )
            row_bindings.append(
                tuple(guid_to_slot[int(guid)] for guid in source_guids)
            )
        exact_binding_slots.append(row_bindings)
        beam.append(
            BeamState(
                graph=None,
                snapshot=row,
                guid_to_slot={},
                next_slot=next_slot,
                last_touched={},
                rewrite_distance={int(node_row[0]): 5 for node_row in row["nodes"]},
                previous_preferred=set(),
                local_streak=0,
                gate_count=int(graph.gate_count),
                depth=0,
                history=(),
                topology_index=topology,
                exact_graph_checkpoint=graph,
                exact_slot_checkpoint=dict(guid_to_slot),
                exact_checkpoint_depth=0,
            )
        )
    load_seconds = time.perf_counter() - load_started

    with torch.no_grad(), autocast_context(device):
        states, live, gate_types = model.initialize_incremental(
            move_batch(initial_batch_many(snapshots), device)
        )
    arena = PagedKVCache(
        layers=model.action_layers_count,
        capacity=len(beam) + 4,
        page_size=8,
        heads=model.action_heads,
        head_width=model.width // model.action_heads,
        model_width=model.width,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else states.dtype,
        gather_backend="vectorized",
    )
    handles = [arena.empty_handle() for _ in beam]
    with torch.no_grad():
        candidates, match_seconds, match_timing, encoded = paged_model_matches(
            beam,
            states,
            live,
            gate_types,
            handles,
            arena,
            model,
            device,
            threshold_config,
            source_vectors,
            microbatch=args.microbatch,
            max_candidates=args.max_source_matches,
            near_source_reserve=args.near_source_reserve,
            source_microbatch=args.source_microbatch,
            source_grouping=args.source_grouping,
            state_batch_backend="tensorized",
            candidate_backend="gpu",
            return_encoded_states=args.ranking_mode == "value",
            profile_stages=True,
        )
        proposal_started = time.perf_counter()
        proposals, proposal_metrics, proposal_timing, _ = build_gpu_proposals(
            candidates,
            beam,
            rule_index,
            per_parent_cap=args.max_action_rank,
            global_cap=args.max_action_rank * len(beam),
            ranking_mode=args.ranking_mode,
            action_value_model=(model if args.ranking_mode == "value" else None),
            action_value_states=(
                encoded if args.ranking_mode == "value" else None
            ),
            action_value_live=(live if args.ranking_mode == "value" else None),
            action_value_weight=args.action_value_weight,
            profile_stages=True,
        )
        proposal_seconds = time.perf_counter() - proposal_started

    ranks_by_parent: list[dict[tuple[int, tuple[int, ...]], int]] = [
        dict() for _ in beam
    ]
    parent_offsets = [0 for _ in beam]
    for proposal in proposals:
        parent = int(proposal.parent)
        parent_offsets[parent] += 1
        key = (int(proposal.xfer_id), tuple(map(int, proposal.binding)))
        ranks_by_parent[parent].setdefault(key, parent_offsets[parent])

    source_counts = torch.bincount(
        candidates.batch_ids, minlength=len(beam)
    ).cpu().tolist()
    source_match_keys = {
        (
            int(parent),
            int(source),
            tuple(int(slot) for slot in binding if int(slot) >= 0),
        )
        for parent, source, binding in zip(
            candidates.batch_ids.cpu().tolist(),
            candidates.sources.cpu().tolist(),
            candidates.bindings.cpu().tolist(),
        )
    }
    detail = []
    for index, action in enumerate(actions):
        source_id = int(rules.xfer_to_source[action.xfer_id])
        matching_ranks = [
            ranks_by_parent[index][(action.xfer_id, binding)]
            for binding in exact_binding_slots[index]
            if (action.xfer_id, binding) in ranks_by_parent[index]
        ]
        rank = min(matching_ranks, default=None)
        source_covered = any(
            (index, source_id, binding) in source_match_keys
            for binding in exact_binding_slots[index]
        )
        detail.append(
            {
                "trajectory": action.trajectory,
                "step": action.step,
                "cost": action.cost,
                "reward": action.reward,
                "node_id": action.node_id,
                "anchor_slots": anchor_slots[index],
                "binding_slots": [list(row) for row in exact_binding_slots[index]],
                "matching_anchor_count": len(anchor_slots[index]),
                "declared_anchor_available": declared_anchor_available[index],
                "declared_anchor_matches": declared_anchor_matches[index],
                "xfer_id": action.xfer_id,
                "source_id": source_id,
                "gate_delta": gate_deltas[action.xfer_id],
                "source_candidates": int(source_counts[index]),
                "exact_action_available": exact_available[index],
                "exact_transition_matches": exact_matches[index],
                "source_match_covered": source_covered,
                "gate_rank": rank,
            }
        )

    by_trajectory: dict[str, list[dict]] = defaultdict(list)
    for row in detail:
        by_trajectory[row["trajectory"]].append(row)
    result = {
        "config": {
            "data": str(args.data),
            "checkpoint": str(args.checkpoint),
            "calibration": str(args.calibration),
            "ecc_file": str(args.ecc_file),
            "trajectories": [str(path) for path in args.trajectory],
            "target_recall": args.target_recall,
            "max_source_matches": args.max_source_matches,
            "near_source_reserve": args.near_source_reserve,
            "max_action_rank": args.max_action_rank,
            "caps": caps,
            "microbatch": args.microbatch,
            "source_microbatch": args.source_microbatch,
            "source_grouping": args.source_grouping,
            "max_gate_increase": args.max_gate_increase,
            "ranking_mode": args.ranking_mode,
            "action_value_weight": args.action_value_weight,
            "device": str(device),
        },
        "summary": summarize_actions(detail, caps),
        "trajectories": {
            name: summarize_actions(rows, caps) for name, rows in by_trajectory.items()
        },
        "timing": {
            "qasm_load_and_exact_validation_seconds": load_seconds,
            "model_match_seconds": match_seconds,
            "model_match_stages": match_timing,
            "proposal_seconds": proposal_seconds,
            "proposal_stages": proposal_timing,
        },
        "proposal_metrics": proposal_metrics,
        "actions": detail,
    }
    if args.sequence_conditioned:
        result["sequence_conditioned"] = audit_sequence_conditioned(
            args.trajectory,
            quartz=quartz,
            context=context,
            xfers=xfers,
            rules=rules,
            rule_index=rule_index,
            gate_deltas=gate_deltas,
            model=model,
            device=device,
            threshold_config=threshold_config,
            source_vectors=source_vectors,
            source_patterns=source_patterns,
            destination_patterns=destination_patterns,
            args=args,
            caps=caps,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": result["summary"], "timing": result["timing"]}, indent=2))


if __name__ == "__main__":
    main()
