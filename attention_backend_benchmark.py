from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from dataset import RuleMetadata
from model_factory import build_model
from train import autocast_context


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated eager/SDPA/live-compacted node-history attention A/B"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--slots", type=int, default=381)
    parser.add_argument("--live-slots", type=int, default=253)
    parser.add_argument("--history", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.live_slots <= args.slots:
        parser.error("--live-slots must be between zero and --slots")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    rules = RuleMetadata.from_payload(payload)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = build_model(
        rules, len(rules.xfer_to_source), checkpoint["args"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    states = torch.randn(
        args.batch_size, args.slots, model.width, device=device
    )
    action_states = torch.randn(
        args.batch_size, args.history, model.width, device=device
    )
    live = torch.zeros(
        args.batch_size, args.slots, device=device, dtype=torch.bool
    )
    live[:, : args.live_slots] = True
    live_indices = torch.arange(args.live_slots, device=device).expand(
        args.batch_size, -1
    )
    action_mask = torch.ones(
        args.batch_size, args.history, device=device, dtype=torch.bool
    )

    outputs = {}
    rows = []
    for backend in ("eager", "sdpa", "sdpa_live"):
        model.readout_attention_backend = backend
        for _ in range(args.warmup):
            with autocast_context(device):
                output = model._fuse_action_history(
                    states,
                    live,
                    action_states,
                    action_mask,
                    live_indices if backend == "sdpa_live" else None,
                )
        synchronize(device)
        del output
        if device.type == "cuda":
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline = 0
        started = time.perf_counter()
        for _ in range(args.repeats):
            with autocast_context(device):
                output = model._fuse_action_history(
                    states,
                    live,
                    action_states,
                    action_mask,
                    live_indices if backend == "sdpa_live" else None,
                )
        synchronize(device)
        elapsed = time.perf_counter() - started
        peak = (
            torch.cuda.max_memory_allocated(device) - baseline
            if device.type == "cuda"
            else 0
        )
        outputs[backend] = output.float().cpu()
        rows.append(
            {
                "backend": backend,
                "milliseconds_per_call": 1000.0 * elapsed / args.repeats,
                "peak_incremental_bytes": peak,
            }
        )

    eager = outputs["eager"]
    for row in rows:
        candidate = outputs[row["backend"]]
        row["max_abs_error_vs_eager"] = float(
            (candidate - eager).abs().max().item()
        )
        row["mean_abs_error_vs_eager"] = float(
            (candidate - eager).abs().mean().item()
        )
    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "slots": args.slots,
        "live_slots": args.live_slots,
        "history": args.history,
        "width": model.width,
        "heads": model.action_heads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "backends": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
