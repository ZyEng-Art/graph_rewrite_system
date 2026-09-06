from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import ctypes.util
import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import torch

from collect_quarl_trajectories import parse_trajectory_directory
from dataset import PrefixDataset, RuleMetadata, collate_prefixes
from gpu_proposals import GpuRuleIndex, build_gpu_proposals
from threshold_inference import (
    CandidateTensors,
    load_threshold_config,
    threshold_candidates,
    threshold_candidate_tensors,
)
from train import autocast_context, build_model, move_batch


class CyclicDataset:
    """Repeat real trajectory states to a requested benchmark batch shape."""

    def __init__(self, base: PrefixDataset, length: int):
        if not len(base) or length < 1:
            raise ValueError("cyclic dataset requires positive base and target lengths")
        self.base = base
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        return self.base[index % len(self.base)]


def summarize_seconds(seconds: list[float], states: int) -> dict:
    median = statistics.median(seconds)
    return {
        "repeats": len(seconds),
        "seconds": seconds,
        "min_seconds": min(seconds),
        "median_seconds": median,
        "mean_seconds": statistics.mean(seconds),
        "states_per_second_from_median": states / max(median, 1e-12),
        "ms_per_state_from_median": 1000.0 * median / max(states, 1),
    }


def expected_graph_hashes(dataset: PrefixDataset) -> list[int]:
    result = []
    for trajectory_index, prefix_length in dataset.indices:
        trajectory = dataset.trajectories[trajectory_index]
        result.append(int(trajectory["steps"][prefix_length]["graph_hash"]))
    return result


def load_quartz_graphs(quartz, context, trajectory_dir: Path, hashes: list[int]):
    saved = parse_trajectory_directory(trajectory_dir)
    state_files = [row.qasm for row in saved[:-1]]
    if len(state_files) != len(hashes):
        raise ValueError(
            f"trajectory has {len(state_files)} nonterminal states but dataset has "
            f"{len(hashes)} states"
        )
    started = time.perf_counter()
    graphs = [
        quartz.PyGraph.from_qasm(context=context, filename=str(path))
        for path in state_files
    ]
    parse_seconds = time.perf_counter() - started
    actual_hashes = [int(graph.hash()) for graph in graphs]
    mismatches = [
        index
        for index, (actual, expected) in enumerate(zip(actual_hashes, hashes))
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "QASM states do not match dataset graph hashes; first mismatches: "
            + ", ".join(map(str, mismatches[:8]))
        )
    return graphs, parse_seconds


def enumerate_xfer_anchor_actions(graph, context) -> int:
    count = 0
    for node in graph.nodes:
        count += len(graph.available_xfers_parallel(context=context, node=node))
    return count


def enumerate_full_bindings(
    graph,
    context,
    xfer_to_source: list[int],
) -> tuple[int, int]:
    raw_xfer_bindings = 0
    source_bindings = set()
    for node in graph.nodes:
        bindings = graph.available_xfer_bindings_parallel(context=context, node=node)
        raw_xfer_bindings += len(bindings)
        for xfer_id, _, node_guids in bindings:
            source_bindings.add(
                (
                    int(xfer_to_source[int(xfer_id)]),
                    tuple(map(int, node_guids)),
                )
            )
    return raw_xfer_bindings, len(source_bindings)


def benchmark_cpu(
    graphs,
    context,
    xfer_to_source: list[int],
    repeats: int,
) -> dict:
    # Warm both original Quartz enumeration APIs before timing the full path.
    enumerate_xfer_anchor_actions(graphs[0], context)
    enumerate_full_bindings(graphs[0], context, xfer_to_source)

    anchor_seconds = []
    anchor_counts = []
    binding_seconds = []
    raw_binding_counts = []
    source_binding_counts = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            anchor_counts.append(
                sum(enumerate_xfer_anchor_actions(graph, context) for graph in graphs)
            )
            anchor_seconds.append(time.perf_counter() - started)

        for _ in range(repeats):
            raw_count = 0
            source_count = 0
            started = time.perf_counter()
            for graph in graphs:
                raw, grouped = enumerate_full_bindings(
                    graph, context, xfer_to_source
                )
                raw_count += raw
                source_count += grouped
            binding_seconds.append(time.perf_counter() - started)
            raw_binding_counts.append(raw_count)
            source_binding_counts.append(source_count)
    finally:
        if was_enabled:
            gc.enable()

    if len(set(anchor_counts)) != 1:
        raise RuntimeError("xfer-anchor action count changed between CPU repeats")
    if len(set(raw_binding_counts)) != 1 or len(set(source_binding_counts)) != 1:
        raise RuntimeError("full-binding count changed between CPU repeats")
    return {
        "original_xfer_anchor": {
            **summarize_seconds(anchor_seconds, len(graphs)),
            "actions": anchor_counts[0],
            "actions_per_state": anchor_counts[0] / len(graphs),
        },
        "full_binding_ground_truth": {
            **summarize_seconds(binding_seconds, len(graphs)),
            "raw_xfer_bindings": raw_binding_counts[0],
            "raw_xfer_bindings_per_state": raw_binding_counts[0] / len(graphs),
            "unique_source_bindings": source_binding_counts[0],
            "unique_source_bindings_per_state": source_binding_counts[0]
            / len(graphs),
        },
    }


def prepare_batches(dataset, rules, batch_size: int):
    batches = []
    started = time.perf_counter()
    for begin in range(0, len(dataset), batch_size):
        samples = [
            dataset[index]
            for index in range(begin, min(begin + batch_size, len(dataset)))
        ]
        batches.append(collate_prefixes(samples, rules))
    return batches, time.perf_counter() - started


@torch.no_grad()
def run_model_pass(
    *,
    batches,
    beams,
    model,
    device,
    threshold_config,
    source_vectors,
    max_source_matches: int,
    output_mode: str,
    xfers_per_source_cpu: list[int],
    xfers_per_source_device: torch.Tensor,
    gpu_rule_index: GpuRuleIndex,
    per_parent_cap: int,
    global_proposal_cap: int,
) -> tuple[float, dict[str, int]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    source_candidates = 0
    expanded_xfer_actions = 0
    expanded_xfer_actions_device = torch.zeros((), device=device, dtype=torch.long)
    eligible_actions = 0
    materialized_actions = 0
    selected_actions = 0
    proposal_candidates = []
    batch_offset = 0
    for cpu_batch, beam_chunk in zip(batches, beams):
        batch = move_batch(cpu_batch, device)
        with autocast_context(device):
            encoded, live, gate_types = model.encode(batch)
            logits, eligible = model.match_logits(
                encoded,
                live,
                gate_types,
                source_vectors=source_vectors,
            )
        if output_mode == "host_materialized":
            rows = threshold_candidates(
                model,
                batch,
                logits,
                eligible,
                threshold_config,
                max_candidates_per_state=max_source_matches,
            )
            source_candidates += sum(map(len, rows))
            expanded_xfer_actions += sum(
                xfers_per_source_cpu[int(source)]
                for state_rows in rows
                for source, _, _, _ in state_rows
            )
        else:
            candidates = threshold_candidate_tensors(
                model,
                batch,
                logits,
                eligible,
                threshold_config,
                max_candidates_per_state=max_source_matches,
                batch_offset=(
                    batch_offset
                    if output_mode
                    in {"gpu_proposals_full", "gpu_proposals_preselect"}
                    else 0
                ),
            )
            source_candidates += int(candidates.sources.numel())
            if output_mode == "gpu_resident" and candidates.sources.numel():
                expanded_xfer_actions_device += xfers_per_source_device[
                    candidates.sources
                ].sum()
            if output_mode in {
                "gpu_proposals_full",
                "gpu_proposals_preselect",
            }:
                proposal_candidates.append(candidates)
            batch_offset += len(beam_chunk)
    if proposal_candidates:
        candidates = CandidateTensors.cat(proposal_candidates)
        beam = [state for chunk in beams for state in chunk]
        proposals, metrics, _, _ = build_gpu_proposals(
            candidates,
            beam,
            gpu_rule_index,
            per_parent_cap=per_parent_cap,
            global_cap=global_proposal_cap,
            ranking_mode="gate",
            preselect_matches=output_mode == "gpu_proposals_preselect",
        )
        if proposals is None:
            raise RuntimeError("proposal benchmark requires final D2H rows")
        eligible_actions = int(metrics["eligible_actions"])
        expanded_xfer_actions = int(metrics["predicted_actions"])
        materialized_actions = int(
            metrics.get("materialized_actions", metrics["eligible_actions"])
        )
        selected_actions = len(proposals)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if output_mode == "gpu_resident":
        expanded_xfer_actions = int(expanded_xfer_actions_device.item())
    return elapsed, {
        "source_binding_candidates": source_candidates,
        "expanded_xfer_actions": expanded_xfer_actions,
        "eligible_actions": eligible_actions,
        "materialized_actions": materialized_actions,
        "selected_actions": selected_actions,
    }


def benchmark_model(
    *,
    dataset,
    rules,
    model,
    device,
    threshold_config,
    batch_size: int,
    max_source_matches: int,
    max_gate_increase: int,
    per_parent_cap: int,
    global_proposal_cap: int,
    repeats: int,
) -> dict:
    batches, preparation_seconds = prepare_batches(dataset, rules, batch_size)
    beams = [
        [
            types.SimpleNamespace(gate_count=int(gate_count))
            for gate_count in cpu_batch["current_types"].ge(0).sum(dim=1).tolist()
        ]
        for cpu_batch in batches
    ]
    with autocast_context(device):
        source_vectors = model.retrieval_source(model.source_representations())

    xfer_counts = [0] * len(rules.source_patterns)
    for source in rules.xfer_to_source:
        xfer_counts[int(source)] += 1
    xfers_per_source_device = torch.tensor(
        xfer_counts, device=device, dtype=torch.long
    )
    source_to_xfers: dict[int, list[int]] = defaultdict(list)
    for xfer_id, source_id in enumerate(rules.xfer_to_source):
        source_to_xfers[int(source_id)].append(xfer_id)
    gate_deltas = [
        len(rules.destination_gate_types[xfer_id])
        - len(rules.source_gate_types[int(source_id)])
        for xfer_id, source_id in enumerate(rules.xfer_to_source)
    ]
    gpu_rule_index = GpuRuleIndex.build(
        source_to_xfers,
        gate_deltas,
        len(rules.source_patterns),
        max_gate_increase,
        device,
    )

    results = {}
    for name in (
        "gpu_resident",
        "host_materialized",
        "gpu_proposals_full",
        "gpu_proposals_preselect",
    ):
        # One full warm-up pass covers all sequence lengths and the final short batch.
        run_model_pass(
            batches=batches,
            beams=beams,
            model=model,
            device=device,
            threshold_config=threshold_config,
            source_vectors=source_vectors,
            max_source_matches=max_source_matches,
            output_mode=name,
            xfers_per_source_cpu=xfer_counts,
            xfers_per_source_device=xfers_per_source_device,
            gpu_rule_index=gpu_rule_index,
            per_parent_cap=per_parent_cap,
            global_proposal_cap=global_proposal_cap,
        )
        seconds = []
        metric_rows = []
        for _ in range(repeats):
            elapsed, metrics = run_model_pass(
                batches=batches,
                beams=beams,
                model=model,
                device=device,
                threshold_config=threshold_config,
                source_vectors=source_vectors,
                max_source_matches=max_source_matches,
                output_mode=name,
                xfers_per_source_cpu=xfer_counts,
                xfers_per_source_device=xfers_per_source_device,
                gpu_rule_index=gpu_rule_index,
                per_parent_cap=per_parent_cap,
                global_proposal_cap=global_proposal_cap,
            )
            seconds.append(elapsed)
            metric_rows.append(metrics)
        metric_series = {
            key: [row[key] for row in metric_rows] for key in metric_rows[0]
        }
        median_metrics = {
            key: int(statistics.median(values))
            for key, values in metric_series.items()
        }
        results[name] = {
            **summarize_seconds(seconds, len(dataset)),
            **median_metrics,
            "source_binding_candidates_per_state": median_metrics[
                "source_binding_candidates"
            ]
            / len(dataset),
            "expanded_xfer_actions_per_state": median_metrics[
                "expanded_xfer_actions"
            ]
            / len(dataset),
            "metrics_by_repeat": metric_rows,
            "source_binding_candidates_by_repeat": metric_series[
                "source_binding_candidates"
            ],
            "source_binding_candidates_min": min(
                metric_series["source_binding_candidates"]
            ),
            "source_binding_candidates_max": max(
                metric_series["source_binding_candidates"]
            ),
            "expanded_xfer_actions_by_repeat": metric_series[
                "expanded_xfer_actions"
            ],
            "candidate_count_stable": len(
                set(metric_series["source_binding_candidates"])
            )
            == 1,
        }

    results["tensor_preparation"] = {
        "seconds": preparation_seconds,
        "ms_per_state": 1000.0 * preparation_seconds / len(dataset),
        "states_per_second": len(dataset) / max(preparation_seconds, 1e-12),
        "note": "PrefixDataset replay plus CPU collation; excluded from core matcher timings",
    }
    for name in (
        "gpu_resident",
        "host_materialized",
        "gpu_proposals_full",
        "gpu_proposals_preselect",
    ):
        total = results[name]["median_seconds"] + preparation_seconds
        results[name]["including_one_time_tensor_preparation"] = {
            "seconds": total,
            "states_per_second": len(dataset) / max(total, 1e-12),
            "ms_per_state": 1000.0 * total / len(dataset),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the selected GPU matcher with Quartz's original CPU "
            "xfer-at-anchor and full-binding enumeration on identical states."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.999)
    parser.add_argument("--ecc-file", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 8, 16))
    parser.add_argument(
        "--benchmark-states",
        type=int,
        help=(
            "cyclically repeat the real trajectory states to this total count; "
            "use 512 to reproduce the historical batch-512 throughput shape"
        ),
    )
    parser.add_argument("--max-source-matches", type=int, default=8192)
    parser.add_argument("--max-gate-increase", type=int, default=1)
    parser.add_argument("--per-parent-cap", type=int, default=128)
    parser.add_argument("--global-proposal-cap", type=int, default=8192)
    parser.add_argument("--cpu-repeats", type=int, default=1)
    parser.add_argument("--model-repeats", type=int, default=5)
    parser.add_argument(
        "--reuse-cpu-from",
        type=Path,
        help="reuse cpu_quartz results from a completed compatible output",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(size < 1 for size in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if args.cpu_repeats < 1 or args.model_repeats < 1:
        parser.error("repeat counts must be positive")
    if args.benchmark_states is not None and args.benchmark_states < 1:
        parser.error("benchmark state count must be positive")
    if args.per_parent_cap < 1 or args.global_proposal_cap < 1:
        parser.error("proposal caps must be positive")
    if not 0.0 < args.target_recall <= 1.0:
        parser.error("target recall must be within (0, 1]")

    for optional_module in ("qiskit", "dgl"):
        if importlib.util.find_spec(optional_module) is None:
            sys.modules[optional_module] = types.ModuleType(optional_module)
    import quartz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    trajectories = payload["train_trajectories"] + payload["test_trajectories"]
    base_dataset = PrefixDataset(trajectories, rules)
    dataset = (
        CyclicDataset(base_dataset, args.benchmark_states)
        if args.benchmark_states is not None
        else base_dataset
    )

    context = quartz.QuartzContext(
        gate_set=["h", "cx", "x", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=False,
    )
    if context.num_xfers != len(rules.xfer_to_source):
        raise ValueError("dataset and Quartz context have different xfer counts")

    if args.reuse_cpu_from is not None:
        previous = json.loads(args.reuse_cpu_from.read_text())
        previous_config = previous.get("config", {})
        previous_states = previous_config.get("states", previous.get("states"))
        if previous_states != len(dataset):
            raise ValueError("reused CPU result has a different state count")
        if "cpu_quartz" not in previous:
            raise ValueError("reused file does not contain cpu_quartz results")
        cpu = previous["cpu_quartz"]
        qasm_parse_seconds = previous_config.get(
            "qasm_parse_seconds_excluded",
            previous.get("qasm_parse_seconds_excluded"),
        )
    else:
        base_graphs, qasm_parse_seconds = load_quartz_graphs(
            quartz,
            context,
            args.trajectory_dir,
            expected_graph_hashes(base_dataset),
        )
        graphs = [base_graphs[index % len(base_graphs)] for index in range(len(dataset))]
        cpu = benchmark_cpu(
            graphs,
            context,
            rules.xfer_to_source,
            args.cpu_repeats,
        )
    # Preserve the expensive CPU baseline even if a later GPU configuration
    # fails. The completed result below replaces this checkpoint atomically at
    # the Python file-write level.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "cpu_complete",
                "states": len(dataset),
                "qasm_parse_seconds_excluded": qasm_parse_seconds,
                "cpu_quartz": cpu,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    if train_args.get("architecture") != "paged_action":
        raise ValueError("throughput benchmark requires a paged_action checkpoint")
    model = build_model(rules, len(rules.xfer_to_source), train_args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.readout_attention_backend = "sdpa"
    threshold_config = load_threshold_config(args.calibration, args.target_recall)

    model_results = {}
    for batch_size in args.batch_sizes:
        effective_size = min(batch_size, len(dataset))
        key = str(effective_size)
        if key in model_results:
            continue
        model_results[key] = benchmark_model(
            dataset=dataset,
            rules=rules,
            model=model,
            device=device,
            threshold_config=threshold_config,
            batch_size=effective_size,
            max_source_matches=args.max_source_matches,
            max_gate_increase=args.max_gate_increase,
            per_parent_cap=args.per_parent_cap,
            global_proposal_cap=args.global_proposal_cap,
            repeats=args.model_repeats,
        )

    cpu_anchor_throughput = cpu["original_xfer_anchor"][
        "states_per_second_from_median"
    ]
    cpu_binding_throughput = cpu["full_binding_ground_truth"][
        "states_per_second_from_median"
    ]
    speedups = {}
    for batch_size, rows in model_results.items():
        speedups[batch_size] = {}
        for mode in (
            "gpu_resident",
            "host_materialized",
            "gpu_proposals_full",
            "gpu_proposals_preselect",
        ):
            throughput = rows[mode]["states_per_second_from_median"]
            speedups[batch_size][mode] = {
                "vs_original_xfer_anchor": throughput / cpu_anchor_throughput,
                "vs_full_binding_ground_truth": throughput
                / cpu_binding_throughput,
            }

    result = {
        "config": {
            "data": str(args.data),
            "checkpoint": str(args.checkpoint),
            "calibration": str(args.calibration),
            "target_recall": args.target_recall,
            "max_source_matches": args.max_source_matches,
            "max_gate_increase": args.max_gate_increase,
            "per_parent_cap": args.per_parent_cap,
            "global_proposal_cap": args.global_proposal_cap,
            "ecc_file": str(args.ecc_file),
            "trajectory_dir": str(args.trajectory_dir),
            "device": str(device),
            "states": len(dataset),
            "unique_input_states": len(base_dataset),
            "states_cyclically_repeated": len(dataset) != len(base_dataset),
            "cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "qasm_parse_seconds_excluded": qasm_parse_seconds,
        },
        "cpu_quartz": cpu,
        "model": model_results,
        "speedup": speedups,
        "timing_boundaries": {
            "cpu_original_xfer_anchor": (
                "all graph nodes plus available_xfers_parallel; graphs/context "
                "are preloaded"
            ),
            "cpu_full_binding_ground_truth": (
                "all graph nodes plus available_xfer_bindings_parallel and "
                "source-binding deduplication; graphs/context are preloaded"
            ),
            "model_gpu_resident": (
                "CPU-to-GPU tensor transfer, encode, source-anchor scoring, r99.9 "
                "threshold, per-state cap, and structural decode; output remains "
                "as GPU tensors"
            ),
            "model_host_materialized": (
                "same logical matcher stages plus device-to-host copies and Python "
                "packing of source-anchor-binding-probability rows"
            ),
            "model_gpu_proposals_full": (
                "GPU matcher, full source-to-xfer expansion, gate/parent/global "
                "ranking and caps, then D2H packing of selected proposals only"
            ),
            "model_gpu_proposals_preselect": (
                "GPU matcher, exact per-parent match preselection before xfer "
                "expansion, gate/parent/global ranking and caps, then D2H packing "
                "of selected proposals only"
            ),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
