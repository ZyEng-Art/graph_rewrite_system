from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch

from beam_search_benchmark import BeamState, snapshot, update_slots
from dataset import RuleMetadata
from hierarchical_actions import hierarchical_paged_matches
from lazy_rollout_benchmark import indexed_topology
from model_factory import build_model
from paged_cache import PagedKVCache
from paged_rollout_benchmark import initial_batch, paged_model_matches
from ppo_core import HierarchicalPPOActorCritic
from threshold_inference import load_threshold_config
from train import autocast_context, move_batch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def add_timings(total: dict[str, float], current: dict[str, float]) -> None:
    for name, seconds in current.items():
        total[name] = total.get(name, 0.0) + seconds


@torch.no_grad()
def measure(
    fn,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
    batch_size: int,
) -> dict:
    for _ in range(warmup):
        fn(False)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    candidate_counts = []
    for _ in range(iterations):
        candidate_count, _ = fn(False)
        candidate_counts.append(candidate_count)
    synchronize(device)
    seconds = time.perf_counter() - started
    _, stage_timing = fn(True)
    return {
        "seconds": seconds,
        "milliseconds_per_batch": seconds * 1000 / iterations,
        "states_per_second": batch_size * iterations / seconds,
        "mean_exact_candidates_per_state": (
            sum(candidate_counts) / len(candidate_counts) / batch_size
        ),
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "profiled_single_batch_seconds": stage_timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--node-checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=224)
    parser.add_argument("--microbatch", type=int, default=224)
    parser.add_argument("--node-ks", default="4,8,16,32")
    parser.add_argument("--pattern-k", type=int, default=16)
    parser.add_argument("--target-recall", type=float, default=0.97)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=930)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    node_ks = [int(value) for value in args.node_ks.split(",") if value]
    if not node_ks or min(node_ks) <= 0:
        parser.error("--node-ks must contain positive integers")

    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    if train_args.get("architecture") != "paged_action":
        raise ValueError("hierarchical matching requires a paged_action checkpoint")
    model = build_model(rules, len(rules.xfer_to_source), train_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.readout_attention_backend = "paged"
    node_checkpoint = None
    if args.node_checkpoint is not None:
        node_checkpoint = torch.load(
            args.node_checkpoint, map_location="cpu", weights_only=False
        )
        if node_checkpoint.get("format") != "hierarchical-node-training-v1":
            raise ValueError("unsupported hierarchical node checkpoint")
        if int(node_checkpoint["width"]) != model.width:
            raise ValueError("node checkpoint width differs from the base model")
    actor = HierarchicalPPOActorCritic(
        model.width,
        hidden_size=(
            int(node_checkpoint["hidden_size"])
            if node_checkpoint is not None
            else model.width
        ),
    ).to(device)
    if node_checkpoint is not None:
        actor.load_state_dict(node_checkpoint["actor_critic"])
    actor.eval()
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
    graph = quartz.PyGraph.from_qasm(context=context, filename=str(args.qasm))
    guid_to_slot: dict[int, int] = {}
    next_slot = update_slots(graph, guid_to_slot, 0)
    initial_snapshot = snapshot(graph, guid_to_slot)
    root = BeamState(
        graph=None,
        snapshot=initial_snapshot,
        guid_to_slot={},
        next_slot=next_slot,
        last_touched={},
        rewrite_distance={int(row[0]): 5 for row in initial_snapshot["nodes"]},
        previous_preferred=set(),
        local_streak=0,
        gate_count=int(graph.gate_count),
        depth=0,
        history=(),
        topology_index=indexed_topology(initial_snapshot),
        exact_graph_checkpoint=graph,
        exact_slot_checkpoint=dict(guid_to_slot),
        exact_checkpoint_depth=0,
    )
    with torch.no_grad(), autocast_context(device):
        initial_states, initial_live, initial_types = model.initialize_incremental(
            move_batch(initial_batch(initial_snapshot), device)
        )
    slot_states = initial_states.expand(args.batch_size, -1, -1).contiguous()
    live = initial_live.expand(args.batch_size, -1).contiguous()
    gate_types = initial_types.expand(args.batch_size, -1).contiguous()
    beam = [root] * args.batch_size
    arena = PagedKVCache(
        layers=model.action_layers_count,
        capacity=max(1, args.batch_size * 2),
        page_size=8,
        heads=model.action_heads,
        head_width=model.width // model.action_heads,
        model_width=model.width,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else slot_states.dtype,
        gather_backend="vectorized",
    )
    handles = [arena.empty_handle()] * args.batch_size

    max_candidates = max(node_ks) * args.pattern_k

    def full_match(profile: bool) -> tuple[int, dict[str, float]]:
        candidates, _, timing, _ = paged_model_matches(
            beam,
            slot_states,
            live,
            gate_types,
            handles,
            arena,
            model,
            device,
            threshold_config,
            source_vectors,
            args.microbatch,
            max_candidates,
            source_grouping="first_gate",
            state_batch_backend="tensorized",
            candidate_backend="gpu",
            return_encoded_states=False,
            profile_stages=profile,
        )
        return candidates.batch_ids.numel(), timing

    result = {
        "format": "hierarchical-matching-benchmark-v1",
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
        },
        "circuit": {
            "path": str(args.qasm),
            "gate_count": int(graph.gate_count),
            "slots": int(slot_states.shape[1]),
        },
        "shape": {
            "batch_size": args.batch_size,
            "microbatch": args.microbatch,
            "sources": model.num_sources,
            "width": model.width,
            "pattern_k": args.pattern_k,
            "node_ks": node_ks,
        },
        "protocol": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "baseline": "all live nodes, first-gate grouped source matcher",
            "hierarchical": (
                f"{'trained' if node_checkpoint is not None else 'untrained'} "
                "node head Top-K, then first-gate-compatible sources "
                "and per-node Top-N"
            ),
            "shared": "paged readout, calibrated threshold, exact structural decoder",
            "warning": "This benchmark measures compute-path speed, not policy quality.",
            "node_checkpoint": (
                str(args.node_checkpoint) if args.node_checkpoint is not None else None
            ),
        },
    }
    result["full_match"] = measure(
        full_match,
        device,
        warmup=args.warmup,
        iterations=args.iterations,
        batch_size=args.batch_size,
    )

    result["hierarchical"] = {}
    for node_k in node_ks:
        def hierarchical(profile: bool, retained: int = node_k):
            output = hierarchical_paged_matches(
                beam,
                slot_states,
                live,
                gate_types,
                handles,
                arena,
                model,
                actor,
                device,
                threshold_config,
                source_vectors,
                microbatch=args.microbatch,
                node_k=retained,
                pattern_k=args.pattern_k,
                profile_stages=profile,
            )
            return output.candidates.batch_ids.numel(), output.timing

        measured = measure(
            hierarchical,
            device,
            warmup=args.warmup,
            iterations=args.iterations,
            batch_size=args.batch_size,
        )
        measured["speedup_over_full_match"] = (
            result["full_match"]["seconds"] / measured["seconds"]
        )
        result["hierarchical"][str(node_k)] = measured

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
