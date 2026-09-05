from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TIMING_RE = re.compile(
    r"Timing: rollout ([0-9.]+)s, learn ([0-9.]+)s, "
    r"iter ([0-9.]+)s, rollout/iter ([0-9.]+)"
)

GROUPS = {
    "setup_and_root_sampling": (
        "buffer.prepare",
        "setup.sample_roots",
        "setup.horizon_sync",
    ),
    "graph_to_model_input": (
        "inference.graph_to_dgl",
        "inference.dgl_batch_h2d",
    ),
    "neural_node_policy": (
        "inference.gnn",
        "inference.critic",
        "inference.node_sampling",
        "inference.actor",
    ),
    "xfer_legality_and_sampling": (
        "inference.available_xfers",
        "inference.xfer_sampling_transfer",
        "inference.result_merge",
    ),
    "quartz_rewrite_and_reward": (
        "environment.apply_xfer",
        "environment.reward_termination",
    ),
    "ppo_state_materialization": (
        "experience.next_state",
        "experience.current_state",
        "experience.append",
    ),
    "frontier_update_and_restart": (
        "buffer.update_and_best",
        "environment.restart_or_advance",
    ),
    "finalize": ("finalize.concatenate",),
}


def load_profile(path: Path) -> dict:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"expected one profile record in {path}, found {len(lines)}")
    row = json.loads(lines[0])
    if row.get("schema") != "original-quarl-rollout-profile-v1":
        raise RuntimeError(f"unexpected profile schema in {path}")
    return row


def load_iteration_timing(path: Path) -> dict:
    matches = TIMING_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one iteration timing in {path}, found {len(matches)}")
    rollout, learn, iteration, fraction = map(float, matches[0])
    return {
        "logged_rollout_seconds_rounded": rollout,
        "learn_seconds": learn,
        "iteration_seconds": iteration,
        "rollout_iteration_fraction": fraction,
    }


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize fine-grained original Quarl rollout profiles"
    )
    parser.add_argument("--profile", action="append", type=Path, required=True)
    parser.add_argument("--run-log", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    if len(args.profile) != len(args.run_log):
        raise RuntimeError("--profile and --run-log counts must match")

    runs = []
    for profile_path, log_path in zip(args.profile, args.run_log):
        profile = load_profile(profile_path)
        graph = profile["input_graphs"][0]
        grouped = {}
        for group_name, stage_names in GROUPS.items():
            seconds = sum(profile["stages"][name]["seconds"] for name in stage_names)
            grouped[group_name] = {
                "seconds": seconds,
                "rollout_fraction": seconds / profile["rollout_seconds"],
                "microseconds_per_transition": (
                    seconds * 1e6 / profile["transitions"]
                ),
            }
        grouped["residual"] = {
            "seconds": profile["residual_seconds"],
            "rollout_fraction": profile["residual_rollout_fraction"],
            "microseconds_per_transition": (
                profile["residual_seconds"] * 1e6 / profile["transitions"]
            ),
        }
        grouped_fraction = sum(row["rollout_fraction"] for row in grouped.values())
        if abs(grouped_fraction - 1.0) > 2e-6:
            raise RuntimeError(
                f"grouped stages for {graph['name']} sum to {grouped_fraction}"
            )
        runs.append(
            {
                "circuit": graph["name"],
                "input_gate_count": graph["input_gate_count"],
                "input_cx_count": graph["input_cx_count"],
                "input_depth": graph["input_depth"],
                "best_gate_count_after_rollout": graph["best_gate_count"],
                "buffer_size_after_rollout": graph["buffer_size"],
                "rollout_seconds": profile["rollout_seconds"],
                "transitions": profile["transitions"],
                "transitions_per_second": profile["transitions_per_second"],
                "peak_cuda_allocated_mib": profile["peak_cuda_allocated_bytes"]
                / 2**20,
                "accounted_rollout_fraction": profile["accounted_rollout_fraction"],
                **load_iteration_timing(log_path),
                "grouped_stages": grouped,
                "stages": profile["stages"],
                "profile_file": profile_path.name,
                "remote_run_log": str(log_path),
            }
        )
    runs.sort(key=lambda row: row["input_gate_count"])
    baseline = runs[0]
    for row in runs:
        row["rollout_slowdown_vs_smallest"] = (
            row["rollout_seconds"] / baseline["rollout_seconds"]
        )

    summary = {
        "schema": "original-quarl-rollout-size-profile-summary-v1",
        "machine": "h100-gpu5",
        "gpu": "NVIDIA H100 80GB HBM3",
        "environment": {
            "python": "3.12.13",
            "torch": "2.4.0+cu121",
            "dgl": "2.4.0+cu124",
            "nvidia_driver": "580.65.06",
            "gpu_memory_mib": 81559,
            "background_vllm_allocation_mib": "47892-47894",
            "background_gpu_utilization_percent_before_runs": 0,
        },
        "source": {
            "snapshot": "/SharedData/dengzy/Quarl/experiment/ppo-new/original_snapshot_compat_20260828",
            "actor_sha256": "bc0bc79503b49ed2a53f945d944dbe9c9d86bb45cc42a8a8a95e4b32d2e3d11e",
            "ppo_sha256": "a387e375ec50383b67d730b520b191b2818ed64f622fc6a8d9e6445a9604e38d",
            "instrumented_actor_sha256": "64e2a6ffee9214163aaed52d22fd339f1a73bb4fe5b6b768dbe32c41ab3c409e",
            "instrumented_ppo_sha256": "339748309011e3e2e6b0e8ab0952dc35b4be5dbe371f24a88e10aed0c845c9a2",
            "checkpoint": "/SharedData/dengzy/Quarl/experiment/ppo-new/outputs/h100_nam_pretrain_6small_cluster_20260816_183726/ckpts/iter_576.pt",
            "checkpoint_sha256": "db198874f5275351ed4215c764c4f839194bffc26f1008018b49610114185807",
        },
        "protocol": {
            "completed_iterations": 1,
            "episodes": 64,
            "fixed_horizon": 20,
            "transitions": 1280,
            "agent_batch_size": 64,
            "agent_collect": True,
            "subgraph_opt": True,
            "xfer_predecessor_layers": 1,
            "ppo_epochs": 1,
            "learning_rates": 0,
            "cuda_stage_boundary_synchronization": True,
        },
        "instrumentation_check": {
            "small_profile_vs_prior_uninstrumented_percent": 1.84,
            "large_profile_seconds": 39.859069805,
            "large_uninstrumented_seconds": 40.557513382,
            "large_profile_vs_uninstrumented_percent": -1.72,
            "interpretation": "No systematic overhead was visible outside roughly +/-2% run-to-run variation.",
        },
        "runs": runs,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    group_order = list(GROUPS) + ["residual"]
    lines = [
        "# Original Quarl rollout profile by circuit size",
        "",
        "## Protocol",
        "",
        "These are one-iteration H100 runs through original Quarl's exact `agent_collect` path.",
        "Every run uses 64 episodes, a fixed horizon of 20, batch size 64, 1,280",
        "transitions, the Nam `iter_576.pt` checkpoint, and zero learning rates. The",
        "profiled snapshot changes timing only; action selection, Quartz application, PPO",
        "state construction, graph-buffer insertion, and episode restart are unchanged.",
        "CUDA is synchronized at neural stage boundaries so asynchronous work is charged",
        "to GNN, critic, actor, or sampling rather than a later `.cpu()` call.",
        "An idle vLLM process retained about 47.9 GiB on every H100 but reported 0% GPU",
        "utilization before these sequential runs; peak CUDA below is Quarl's own allocation.",
        "",
        "## Scale",
        "",
        "| circuit | gates | rollout | transitions/s | rollout/iteration | peak CUDA | buffer after |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        lines.append(
            f"| `{row['circuit']}` | {row['input_gate_count']} | "
            f"{row['rollout_seconds']:.3f}s | {row['transitions_per_second']:.2f} | "
            f"{percent(row['rollout_iteration_fraction'])} | "
            f"{row['peak_cuda_allocated_mib']:.1f} MiB | "
            f"{row['buffer_size_after_rollout']} |"
        )

    lines += [
        "",
        "## Grouped rollout share",
        "",
        "All percentages are mutually exclusive and include a measured residual, so each",
        "row sums to 100%.",
        "",
        "| circuit | setup | graph input | neural node | xfer legal/sample | Quartz apply/reward | PPO subgraphs | buffer/restart | finalize/residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        groups = row["grouped_stages"]
        final_fraction = (
            groups["finalize"]["rollout_fraction"]
            + groups["residual"]["rollout_fraction"]
        )
        lines.append(
            f"| `{row['circuit']}` | "
            f"{percent(groups['setup_and_root_sampling']['rollout_fraction'])} | "
            f"{percent(groups['graph_to_model_input']['rollout_fraction'])} | "
            f"{percent(groups['neural_node_policy']['rollout_fraction'])} | "
            f"{percent(groups['xfer_legality_and_sampling']['rollout_fraction'])} | "
            f"{percent(groups['quartz_rewrite_and_reward']['rollout_fraction'])} | "
            f"{percent(groups['ppo_state_materialization']['rollout_fraction'])} | "
            f"{percent(groups['frontier_update_and_restart']['rollout_fraction'])} | "
            f"{percent(final_fraction)} |"
        )

    detailed_stages = [
        "inference.graph_to_dgl",
        "inference.dgl_batch_h2d",
        "inference.gnn",
        "inference.critic",
        "inference.node_sampling",
        "inference.actor",
        "inference.available_xfers",
        "inference.xfer_sampling_transfer",
        "environment.apply_xfer",
        "experience.next_state",
        "experience.current_state",
        "buffer.update_and_best",
        "environment.restart_or_advance",
    ]
    lines += [
        "",
        "## Detailed rollout share",
        "",
        "| stage | " + " | ".join(f"{row['input_gate_count']}g" for row in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    for stage in detailed_stages:
        lines.append(
            f"| `{stage}` | "
            + " | ".join(percent(row["stages"][stage]["rollout_fraction"]) for row in runs)
            + " |"
        )

    scaling_stages = [
        "inference.graph_to_dgl",
        "inference.gnn",
        "inference.available_xfers",
        "environment.apply_xfer",
        "experience.next_state",
        "experience.current_state",
        "buffer.update_and_best",
    ]
    lines += [
        "",
        "## Milliseconds per transition",
        "",
        "| stage | " + " | ".join(f"{row['input_gate_count']}g" for row in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    for stage in scaling_stages:
        lines.append(
            f"| `{stage}` | "
            + " | ".join(
                f"{row['stages'][stage]['microseconds_per_transition'] / 1000:.3f}"
                for row in runs
            )
            + " |"
        )

    lines += [
        "",
        "## Findings",
        "",
        "- Throughput falls from 254.26 transitions/s at 58 gates to 32.11 at 3,435 gates; the fixed-work rollout is 7.92x slower.",
        "- At 58-259 gates, constructing current/next DGL subgraphs for PPO training is the largest group (35-44%); exact Quartz apply is only 1-10%.",
        "- At 831 gates, exact Quartz apply reaches 28.3% and becomes the largest individual stage.",
        "- At 3,435 gates, Quartz apply is 38.0%, available-xfer checking/mask construction is 19.6%, and full graph-to-DGL conversion is 13.5%. Together they consume 71.1% of rollout.",
        "- GNN time rises in absolute terms but its share falls from 11.9% to 5.3% at the largest size because sequential Quartz and CPU graph work grows faster.",
        "- The largest run retained only 19 buffer states, so its 39.86s rollout cannot be attributed to a large persistent buffer in this iteration.",
        "- The profiled and uninstrumented 3,435-gate runs were 39.86s and 40.56s with identical transitions and outcomes; instrumentation variance was -1.7%.",
        "",
        "The raw JSON files retain all stage seconds, call counts, percentages, and",
        "microseconds per transition. Remote `run.log` paths are recorded in the summary JSON.",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
