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
from hierarchical_ppo_rollout import (
    collect_hierarchical_episode_batch,
    hierarchical_ppo_update,
)
from incremental_graph import parse_pattern
from model_factory import build_model
from ppo_core import HierarchicalPPOActorCritic
from threshold_inference import load_threshold_config
from train import autocast_context
from train_paged_ppo import (
    make_replay_bucket,
    replay_pool_metrics,
    serializable_replay_pool,
)


def allocate_circuit_episodes(
    circuits: list[Path], total_episodes: int, *, offset: int = 0
) -> list[tuple[Path, int]]:
    if total_episodes < 1:
        raise ValueError("total episodes must be positive")
    if not circuits:
        raise ValueError("at least one circuit is required")
    counts = [total_episodes // len(circuits)] * len(circuits)
    for index in range(total_episodes % len(circuits)):
        counts[(offset + index) % len(circuits)] += 1
    return list(zip(circuits, counts))


def merge_collector_timing(
    destination: dict[str, float], source: dict[str, float]
) -> None:
    for key, value in source.items():
        destination[key] = destination.get(key, 0.0) + value


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
    rewards = [float(row.total_reward) for row in metrics]
    initial_gate_counts = [int(row.initial_gate_count) for row in metrics]
    committed_deltas = [
        int(row.next_gate_count - row.previous_gate_count)
        for row in transitions
        if row.committed_action and row.legal and not row.repeated_state
    ]
    selected_xfers = Counter(str(row.xfer_id) for row in transitions)
    cycle_xfers = Counter(
        str(row.xfer_id) for row in transitions if row.repeated_state
    )
    return {
        "episodes": len(metrics),
        "transitions": transition_count,
        "accepted_rewrites": accepted,
        "transitions_per_second": transition_count / total_seconds,
        "accepted_rewrites_per_second": accepted / total_seconds,
        "initial_gate_count": metrics[0].initial_gate_count,
        "initial_gate_count_histogram": dict(
            sorted(Counter(str(count) for count in initial_gate_counts).items())
        ),
        "minimum_initial_gate_count": min(initial_gate_counts),
        "maximum_initial_gate_count": max(initial_gate_counts),
        "mean_initial_gate_count": sum(initial_gate_counts) / len(initial_gate_counts),
        "minimum_best_gate_count": min(row.best_gate_count for row in metrics),
        "minimum_final_gate_count": min(row.final_gate_count for row in metrics),
        "mean_best_gate_count": sum(row.best_gate_count for row in metrics)
        / len(metrics),
        "improved_episodes": sum(
            row.best_gate_count < row.initial_gate_count for row in metrics
        ),
        "mean_total_reward": sum(rewards) / len(rewards),
        "minimum_total_reward": min(rewards),
        "maximum_total_reward": max(rewards),
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
        "cycle_xfers": dict(cycle_xfers.most_common()),
        "selected_xfers_top20": dict(selected_xfers.most_common(20)),
        "unique_selected_xfers": len(selected_xfers),
        "committed_gate_delta_histogram": dict(
            sorted(Counter(str(delta) for delta in committed_deltas).items())
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
    parser.add_argument("--additional-qasm", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--warmup-episodes", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--node-k", type=int, default=16)
    parser.add_argument("--pattern-k", type=int, default=16)
    parser.add_argument("--fallback-target-recall", type=float, default=0.99)
    parser.add_argument("--fallback-node-k", type=int, default=64)
    parser.add_argument("--fallback-pattern-k", type=int, default=32)
    parser.add_argument("--fallback-min-candidates", type=int, default=16)
    parser.add_argument("--disable-candidate-fallback", action="store_true")
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument(
        "--eliminate-rotation",
        action="store_true",
        help=(
            "fold Quartz rotation parameters after every rewrite and reconcile "
            "the normalized exact topology back into the paged rollout"
        ),
    )
    parser.add_argument("--topology-audit-interval", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-rejected-actions-per-step", type=int, default=4)
    parser.add_argument("--max-exact-rejections-per-episode", type=int, default=8)
    parser.add_argument("--invalid-reward", type=float, default=-2.0)
    parser.add_argument("--cycle-reward", type=float, default=-1.0)
    parser.add_argument("--step-penalty", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--initial-gate-bias", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=950)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--terminate-on-improvement", action="store_true")
    parser.add_argument("--ppo-iterations", type=int, default=0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=128)
    parser.add_argument("--ppo-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-value-coefficient", type=float, default=0.5)
    parser.add_argument("--ppo-entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--ppo-target-kl", type=float, default=0.02)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ppo-output", type=Path)
    parser.add_argument("--disable-rejected-action-cache", action="store_true")
    parser.add_argument("--start-from-best", action="store_true")
    parser.add_argument("--best-start-probability", type=float, default=0.25)
    parser.add_argument("--use-replay-starts", action="store_true")
    parser.add_argument("--replay-start-probability", type=float, default=0.25)
    parser.add_argument("--resume-search-state", action="store_true")
    args = parser.parse_args()
    if args.eliminate_rotation and args.refresh_interval != 1:
        parser.error("--eliminate-rotation requires --refresh-interval 1")

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
    actor_checkpoint_format = node_checkpoint.get("format", "unknown")
    if actor_checkpoint_format not in {
        "hierarchical-node-training-v1",
        "hierarchical-action-training-v1",
        "hierarchical-ppo-v1",
    }:
        raise ValueError(
            "--node-checkpoint must contain a hierarchical node or PPO actor; "
            f"got {actor_checkpoint_format!r}"
        )
    actor = HierarchicalPPOActorCritic(
        model.width, hidden_size=int(node_checkpoint["hidden_size"])
    ).to(device)
    actor.load_state_dict(node_checkpoint["actor_critic"])
    actor.eval()
    threshold_config = load_threshold_config(
        args.calibration, args.target_recall
    )
    fallback_threshold_config = (
        None
        if args.disable_candidate_fallback
        else load_threshold_config(args.calibration, args.fallback_target_recall)
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
    qasm_paths = [args.qasm, *args.additional_qasm]
    circuit_names = [path.name for path in qasm_paths]
    if len(set(circuit_names)) != len(circuit_names):
        raise ValueError("QASM file names must be unique across circuits")
    graphs = {
        path.name: quartz.PyGraph.from_qasm(context=context, filename=str(path))
        for path in qasm_paths
    }
    saved_best_by_circuit = (
        node_checkpoint.get("best_so_far", {}) if args.resume_search_state else {}
    )
    best_by_circuit = {}
    for circuit_name, graph in graphs.items():
        saved_best = saved_best_by_circuit.get(circuit_name)
        best_by_circuit[circuit_name] = (
            dict(saved_best)
            if saved_best is not None
            and int(saved_best["gate_count"]) <= int(graph.gate_count)
            else {
                "gate_count": int(graph.gate_count),
                "qasm": graph.to_qasm_str(),
                "episode_depth": 0,
            }
        )
    saved_replay_pool = node_checkpoint.get("replay_pool", {})
    replay_pool = {}
    for circuit_name, graph in graphs.items():
        replay_pool[circuit_name] = (
            saved_replay_pool[circuit_name]
            if args.resume_search_state and circuit_name in saved_replay_pool
            else make_replay_bucket(graph)
        )

    def collect(
        run_qasm: Path,
        run_batch_size: int,
        run_max_steps: int,
        run_best: dict,
        run_replay: dict,
        run_timing: dict[str, float],
        run_greedy: bool,
        run_rejected_cache,
    ):
        return collect_hierarchical_episode_batch(
            run_qasm,
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
            fallback_threshold_config=fallback_threshold_config,
            fallback_node_k=(
                None if args.disable_candidate_fallback else args.fallback_node_k
            ),
            fallback_pattern_k=(
                None if args.disable_candidate_fallback else args.fallback_pattern_k
            ),
            fallback_min_candidates=args.fallback_min_candidates,
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
            greedy=run_greedy,
            best_by_circuit=run_best,
            replay_pool=run_replay,
            replay_capacity_per_circuit=64,
            refresh_interval=min(args.refresh_interval, run_max_steps),
            eliminate_rotation=args.eliminate_rotation,
            topology_audit_interval=args.topology_audit_interval,
            rejected_action_cache=run_rejected_cache,
            max_exact_rejections_per_episode=(
                args.max_exact_rejections_per_episode
            ),
            start_from_best=args.start_from_best,
            best_start_probability=args.best_start_probability,
            use_replay_starts=args.use_replay_starts,
            replay_start_probability=args.replay_start_probability,
            collector_timing=run_timing,
        )

    def collect_across_circuits(
        total_episodes: int,
        run_max_steps: int,
        run_best: dict,
        run_replay: dict,
        run_timing: dict[str, float],
        run_greedy: bool,
        run_rejected_cache,
        *,
        offset: int = 0,
    ):
        all_transitions = []
        all_metrics = []
        per_circuit = {}
        for run_qasm, episode_count in allocate_circuit_episodes(
            qasm_paths, total_episodes, offset=offset
        ):
            if not episode_count:
                continue
            circuit_timing: dict[str, float] = {}
            circuit_transitions, circuit_metrics = collect(
                run_qasm,
                episode_count,
                run_max_steps,
                run_best,
                run_replay,
                circuit_timing,
                run_greedy,
                run_rejected_cache,
            )
            all_transitions.extend(circuit_transitions)
            all_metrics.extend(circuit_metrics)
            merge_collector_timing(run_timing, circuit_timing)
            per_circuit[run_qasm.name] = summarize_metrics(
                circuit_metrics, circuit_transitions, circuit_timing
            )
        return all_transitions, all_metrics, per_circuit

    if args.warmup_episodes and args.warmup_steps:
        warmup_best = {
            circuit_name: {
                "gate_count": int(graph.gate_count),
                "qasm": graph.to_qasm_str(),
                "episode_depth": 0,
            }
            for circuit_name, graph in graphs.items()
        }
        collect_across_circuits(
            args.warmup_episodes,
            args.warmup_steps,
            warmup_best,
            {
                circuit_name: make_replay_bucket(graph)
                for circuit_name, graph in graphs.items()
            },
            {},
            args.greedy,
            {},
        )
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iteration_results = []
    optimizer = None
    rejected_action_cache = (
        None
        if args.disable_rejected_action_cache
        else (
            node_checkpoint.get("rejected_action_cache", {})
            if args.resume_search_state
            else {}
        )
    )
    if args.ppo_iterations:
        optimizer = torch.optim.AdamW(
            actor.parameters(), lr=args.ppo_learning_rate, weight_decay=1e-4
        )
        for iteration in range(1, args.ppo_iterations + 1):
            actor.eval()
            iteration_timing: dict[str, float] = {}
            (
                iteration_transitions,
                iteration_metrics,
                iteration_by_circuit,
            ) = collect_across_circuits(
                args.batch_size,
                args.max_steps,
                best_by_circuit,
                replay_pool,
                iteration_timing,
                args.greedy,
                rejected_action_cache,
                offset=iteration - 1,
            )
            update = hierarchical_ppo_update(
                actor,
                optimizer,
                iteration_transitions,
                device=device,
                epochs=args.ppo_epochs,
                minibatch_size=args.ppo_minibatch_size,
                clip_epsilon=args.ppo_clip_epsilon,
                value_coefficient=args.ppo_value_coefficient,
                entropy_coefficient=args.ppo_entropy_coefficient,
                target_kl=args.ppo_target_kl,
                max_grad_norm=args.ppo_max_grad_norm,
                seed=args.seed + iteration,
            )
            row = {
                "iteration": iteration,
                "rollout": summarize_metrics(
                    iteration_metrics, iteration_transitions, iteration_timing
                ),
                "rollout_by_circuit": iteration_by_circuit,
                "update": update,
                "best_gate_count": best_by_circuit[args.qasm.name]["gate_count"],
                "best_gate_count_by_circuit": {
                    name: row["gate_count"] for name, row in best_by_circuit.items()
                },
            }
            iteration_results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        actor.eval()
        timing = {}
        transitions, metrics, summary_by_circuit = collect_across_circuits(
            args.batch_size,
            args.max_steps,
            best_by_circuit,
            replay_pool,
            timing,
            args.greedy,
            rejected_action_cache,
        )
    else:
        timing = {}
        transitions, metrics, summary_by_circuit = collect_across_circuits(
            args.batch_size,
            args.max_steps,
            best_by_circuit,
            replay_pool,
            timing,
            args.greedy,
            rejected_action_cache,
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
            "actor_checkpoint_format": actor_checkpoint_format,
            "calibration": str(args.calibration),
            "qasm": str(args.qasm),
            "qasm_paths": [str(path) for path in qasm_paths],
        },
        "args": vars(args),
        "summary": summarize_metrics(metrics, transitions, timing),
        "summary_by_circuit": summary_by_circuit,
        "best_so_far": best_by_circuit[args.qasm.name],
        "best_by_circuit": best_by_circuit,
        "stored_transition_count": len(transitions),
        "rejected_action_cache": {
            "enabled": rejected_action_cache is not None,
            "states": len(rejected_action_cache or {}),
            "actions": sum(
                len(actions) for actions in (rejected_action_cache or {}).values()
            ),
        },
        "replay_pool": replay_pool_metrics(replay_pool),
        "training_iterations": iteration_results,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")
    if args.ppo_output is not None:
        args.ppo_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "hierarchical-ppo-v1",
                "actor_critic": actor.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "width": model.width,
                "hidden_size": actor.hidden_size,
                "base_checkpoint": str(args.checkpoint),
                "node_checkpoint": str(args.node_checkpoint),
                "args": vars(args),
                "training_iterations": iteration_results,
                "best_so_far": best_by_circuit,
                "replay_pool": serializable_replay_pool(replay_pool),
                "rejected_action_cache": rejected_action_cache,
            },
            args.ppo_output,
        )


if __name__ == "__main__":
    main()
