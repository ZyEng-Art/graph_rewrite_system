from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
import ctypes.util
import importlib.util
import json
from pathlib import Path
import random
import sys
import types

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch

from dataset import RuleMetadata
from gpu_proposals import GpuRuleIndex
from hierarchical_ppo_rollout import collect_hierarchical_episode_batch
from incremental_graph import parse_pattern
from model_factory import build_model
from ppo_core import HierarchicalPPOActorCritic
from threshold_inference import load_threshold_config
from train import autocast_context
from train_paged_ppo import make_replay_bucket


def summarize_metrics(metrics, transitions, timing: dict[str, float]) -> dict:
    total_seconds = timing["total_seconds"]
    stages = {
        name: {
            "seconds": seconds,
            "fraction": seconds / total_seconds,
        }
        for name, seconds in timing.items()
        if name.endswith("_seconds") and name != "total_seconds"
    }
    accepted = sum(row.accepted_rewrites for row in metrics)
    transition_count = sum(row.steps for row in metrics)
    return {
        "episodes": len(metrics),
        "transitions": transition_count,
        "accepted_rewrites": accepted,
        "transitions_per_second": transition_count / total_seconds,
        "accepted_rewrites_per_second": accepted / total_seconds,
        "initial_gate_count": metrics[0].initial_gate_count,
        "minimum_best_gate_count": min(row.best_gate_count for row in metrics),
        "minimum_final_gate_count": min(row.final_gate_count for row in metrics),
        "mean_best_gate_count": sum(row.best_gate_count for row in metrics)
        / len(metrics),
        "legal_actions": sum(row.legal_actions for row in metrics),
        "invalid_actions": sum(row.invalid_actions for row in metrics),
        "cycle_actions": sum(row.cycle_actions for row in metrics),
        "exact_refreshes": sum(row.exact_refreshes for row in metrics),
        "exact_replay_actions": sum(row.exact_replay_actions for row in metrics),
        "exact_refresh_seconds_recorded": sum(
            row.exact_refresh_seconds for row in metrics
        ),
        "mean_candidates": sum(row.mean_candidates for row in metrics)
        / len(metrics),
        "mean_policy_entropy": sum(row.mean_entropy for row in metrics)
        / len(metrics),
        "termination_reasons": dict(
            sorted(
                (
                    reason,
                    sum(row.terminated_reason == reason for row in metrics),
                )
                for reason in {row.terminated_reason for row in metrics}
            )
        ),
        "invalid_xfers": dict(
            Counter(
                str(row.xfer_id) for row in transitions if not row.legal
            ).most_common()
        ),
        "timing": {**timing, "stages": stages},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--node-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--qasm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--warmup-episodes", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--node-k", type=int, default=16)
    parser.add_argument("--pattern-k", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-rejected-actions-per-step", type=int, default=4)
    parser.add_argument("--invalid-reward", type=float, default=-2.0)
    parser.add_argument("--cycle-reward", type=float, default=-1.0)
    parser.add_argument("--step-penalty", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--initial-gate-bias", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=950)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--terminate-on-improvement", action="store_true")
    args = parser.parse_args()

    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(
        rules, len(rules.xfer_to_source), checkpoint["args"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.readout_attention_backend = "paged"
    model.requires_grad_(False)

    node_checkpoint = torch.load(
        args.node_checkpoint, map_location="cpu", weights_only=False
    )
    actor = HierarchicalPPOActorCritic(
        model.width, hidden_size=int(node_checkpoint["hidden_size"])
    ).to(device)
    actor.load_state_dict(node_checkpoint["actor_critic"])
    actor.eval()
    threshold_config = load_threshold_config(
        args.calibration, args.target_recall
    )
    with torch.no_grad(), autocast_context(device):
        source_representations = model.source_representations()
        source_vectors = model.retrieval_source(source_representations)

    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    if context.num_xfers != len(rules.xfer_to_source):
        raise RuntimeError("dataset and Quartz context have different xfer counts")
    xfers = [context.get_xfer_from_id(id=index) for index in range(context.num_xfers)]
    source_to_xfers: dict[int, list[int]] = defaultdict(list)
    for xfer_id, source_id in enumerate(rules.xfer_to_source):
        source_to_xfers[source_id].append(xfer_id)
    gate_deltas = [
        len(rules.destination_gate_types[index])
        - len(rules.source_gate_types[rules.xfer_to_source[index]])
        for index in range(len(rules.xfer_to_source))
    ]
    rule_index = GpuRuleIndex.build(
        source_to_xfers,
        gate_deltas,
        len(rules.source_gate_types),
        max_gate_increase=1,
        device=device,
    )
    source_patterns = tuple(parse_pattern(pattern) for pattern in rules.xfer_sources)
    destination_patterns = tuple(
        parse_pattern(pattern) for pattern in rules.xfer_destinations
    )
    graph = quartz.PyGraph.from_qasm(context=context, filename=str(args.qasm))
    circuit_name = args.qasm.name
    best_by_circuit = {
        circuit_name: {
            "gate_count": int(graph.gate_count),
            "qasm": graph.to_qasm_str(),
            "episode_depth": 0,
        }
    }
    replay_pool = {circuit_name: make_replay_bucket(graph)}

    def collect(
        run_batch_size: int,
        run_max_steps: int,
        run_best: dict,
        run_replay: dict,
        run_timing: dict[str, float],
    ):
        return collect_hierarchical_episode_batch(
            args.qasm,
            run_batch_size,
            context=context,
            quartz=quartz,
            xfers=xfers,
            rules=rules,
            source_patterns=source_patterns,
            destination_patterns=destination_patterns,
            rule_index=rule_index,
            model=model,
            actor_critic=actor,
            device=device,
            threshold_config=threshold_config,
            source_vectors=source_vectors,
            source_representations=source_representations,
            max_steps=run_max_steps,
            node_k=args.node_k,
            pattern_k=args.pattern_k,
            max_actions=args.max_actions,
            invalid_reward=args.invalid_reward,
            cycle_reward=args.cycle_reward,
            step_penalty=args.step_penalty,
            max_rejected_actions_per_step=args.max_rejected_actions_per_step,
            terminate_on_improvement=args.terminate_on_improvement,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            page_size=args.page_size,
            initial_gate_bias=args.initial_gate_bias,
            greedy=args.greedy,
            best_by_circuit=run_best,
            replay_pool=run_replay,
            replay_capacity_per_circuit=64,
            refresh_interval=min(args.refresh_interval, run_max_steps),
            collector_timing=run_timing,
        )

    if args.warmup_episodes and args.warmup_steps:
        warmup_best = {
            circuit_name: {
                "gate_count": int(graph.gate_count),
                "qasm": graph.to_qasm_str(),
                "episode_depth": 0,
            }
        }
        collect(
            args.warmup_episodes,
            args.warmup_steps,
            warmup_best,
            {circuit_name: make_replay_bucket(graph)},
            {},
        )
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timing: dict[str, float] = {}
    transitions, metrics = collect(
        args.batch_size,
        args.max_steps,
        best_by_circuit,
        replay_pool,
        timing,
    )
    result = {
        "format": "hierarchical-ppo-rollout-benchmark-v1",
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "peak_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        },
        "artifacts": {
            "data": str(args.data),
            "checkpoint": str(args.checkpoint),
            "node_checkpoint": str(args.node_checkpoint),
            "calibration": str(args.calibration),
            "qasm": str(args.qasm),
        },
        "args": vars(args),
        "summary": summarize_metrics(metrics, transitions, timing),
        "best_so_far": best_by_circuit[circuit_name],
        "stored_transition_count": len(transitions),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
