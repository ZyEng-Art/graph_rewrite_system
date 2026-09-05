from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from ppo_core import HierarchicalPPOActorCritic, MatchSetPPOActorCritic


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure(fn, device: torch.device, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        output = fn()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(iterations):
        output = fn()
    synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "milliseconds_per_batch": seconds * 1000 / iterations,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "normalization_max_error": float(
            (output.exp().sum(1) - 1).abs().max().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=224)
    parser.add_argument("--nodes", type=int, default=450)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=930)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hierarchical = HierarchicalPPOActorCritic(
        args.width, hidden_size=args.hidden_size
    ).to(device).eval()
    match_set = MatchSetPPOActorCritic(
        args.width, hidden_size=args.hidden_size
    ).to(device).eval()
    policy_width = hierarchical.policy_feature_dim
    state_width = hierarchical.state_feature_dim
    node_features = torch.randn(
        args.batch_size, args.nodes, args.width, device=device
    )
    node_mask = torch.ones(
        args.batch_size, args.nodes, dtype=torch.bool, device=device
    )
    candidate_features = torch.randn(
        args.batch_size, args.candidates, policy_width, device=device
    )
    matcher_logits = torch.randn(
        args.batch_size, args.candidates, device=device
    )
    candidate_mask = torch.ones(
        args.batch_size, args.candidates, dtype=torch.bool, device=device
    )
    candidate_nodes = torch.randint(
        args.nodes,
        (args.batch_size, args.candidates),
        device=device,
    )
    prefix_states = torch.randn(args.batch_size, args.width, device=device)
    state_features = torch.randn(args.batch_size, state_width, device=device)

    def run_hierarchical() -> torch.Tensor:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            policy = hierarchical.policy(
                node_features,
                node_mask,
                candidate_features,
                matcher_logits,
                candidate_nodes,
                candidate_mask,
                prefix_states,
                state_features,
            )
        return policy.log_probs.float()

    def run_match_set() -> torch.Tensor:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = match_set.policy_logits(
                candidate_features,
                matcher_logits,
                candidate_mask,
                prefix_states,
                state_features,
            )
        return torch.log_softmax(logits.float(), dim=-1)

    result = {
        "format": "hierarchical-policy-head-benchmark-v1",
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
        },
        "shape": {
            "batch_size": args.batch_size,
            "nodes": args.nodes,
            "candidates": args.candidates,
            "width": args.width,
            "hidden_size": args.hidden_size,
        },
        "protocol": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "autocast": "bfloat16" if device.type == "cuda" else "disabled",
            "scope": "policy heads only; candidate retrieval and graph update excluded",
        },
        "parameters": {
            "hierarchical": sum(p.numel() for p in hierarchical.parameters()),
            "match_set": sum(p.numel() for p in match_set.parameters()),
        },
        "hierarchical": measure(
            run_hierarchical, device, args.warmup, args.iterations
        ),
        "match_set": measure(run_match_set, device, args.warmup, args.iterations),
    }
    result["hierarchical"]["states_per_second"] = (
        args.batch_size
        * args.iterations
        / result["hierarchical"]["seconds"]
    )
    result["match_set"]["states_per_second"] = (
        args.batch_size * args.iterations / result["match_set"]["seconds"]
    )
    result["speedup_over_match_set"] = (
        result["match_set"]["seconds"] / result["hierarchical"]["seconds"]
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
