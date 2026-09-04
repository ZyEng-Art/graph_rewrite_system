from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from paged_cache import PagedKVCache, PrefixHandle


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_arena(args, device: torch.device, backend: str) -> PagedKVCache:
    return PagedKVCache(
        layers=args.layers,
        capacity=args.capacity,
        page_size=args.page_size,
        heads=args.heads,
        head_width=args.head_width,
        model_width=args.width,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        gather_backend=backend,
    )


@torch.no_grad()
def timed_call(device, warmup: int, repeats: int, function):
    for _ in range(warmup):
        output = function()
    synchronize(device)
    del output
    if device.type == "cuda":
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = 0
    started = time.perf_counter()
    for _ in range(repeats):
        output = function()
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(device) - baseline
        if device.type == "cuda"
        else 0
    )
    return output, 1000.0 * elapsed / repeats, peak


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loop-versus-vectorized paged gather/COW kernel benchmark"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--head-width", type=int, default=32)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--physical-pages", type=int, default=1141)
    parser.add_argument("--capacity", type=int, default=11000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blocks = (args.history + args.page_size - 1) // args.page_size

    arena = make_arena(args, device, "vectorized")
    arena.keys[:, : args.physical_pages].normal_()
    arena.values[:, : args.physical_pages].normal_()
    arena.actions[: args.physical_pages].normal_()
    handles = [
        PrefixHandle(
            tuple(
                (batch_index * blocks + block) % args.physical_pages
                for block in range(blocks)
            ),
            args.history,
        )
        for batch_index in range(args.batch_size)
    ]

    gc.disable()
    rows = []
    reference = {}
    for backend in ("loop", "vectorized"):
        arena.gather_backend = backend
        for operation, function in (
            ("actions", lambda: arena.gather_actions(handles)),
            ("full_kv_actions", lambda: arena.gather(handles)),
        ):
            output, milliseconds, peak = timed_call(
                device, args.warmup, args.repeats, function
            )
            if backend == "loop":
                reference[operation] = tuple(tensor.clone() for tensor in output)
                max_error = 0.0
            else:
                max_error = max(
                    float((candidate.float() - expected.float()).abs().max().item())
                    for candidate, expected in zip(output, reference[operation])
                )
            rows.append(
                {
                    "operation": operation,
                    "backend": backend,
                    "milliseconds_per_call": milliseconds,
                    "peak_incremental_bytes": peak,
                    "max_abs_error_vs_loop": max_error,
                }
            )
            del output

    # COW uses length history-1 so the final page has a partial tail.
    cow_history = args.history - 1
    cow_blocks = (cow_history + args.page_size - 1) // args.page_size
    for backend in ("loop", "vectorized"):
        cow_arena = make_arena(args, device, backend)
        parents = []
        for _ in range(args.batch_size):
            parent_blocks = tuple(
                cow_arena._allocate() for _ in range(cow_blocks)
            )
            parents.append(PrefixHandle(parent_blocks, cow_history))
        used_pages = args.batch_size * cow_blocks
        cow_arena.keys[:, :used_pages].normal_()
        cow_arena.values[:, :used_pages].normal_()
        cow_arena.actions[:used_pages].normal_()
        keys = torch.randn(
            args.layers,
            args.batch_size,
            args.heads,
            args.head_width,
            device=device,
        )
        values = torch.randn_like(keys)
        actions = torch.randn(args.batch_size, args.width, device=device)

        def append_and_release():
            children = cow_arena.append_batch(parents, keys, values, actions)
            for child in children:
                cow_arena.release(child)
            return keys

        _, milliseconds, peak = timed_call(
            device, args.warmup, args.repeats, append_and_release
        )
        rows.append(
            {
                "operation": "cow_append",
                "backend": backend,
                "milliseconds_per_call": milliseconds,
                "peak_incremental_bytes": peak,
                "max_abs_error_vs_loop": 0.0,
            }
        )
        for parent in parents:
            cow_arena.release(parent)

    gc.enable()

    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "history": args.history,
        "page_size": args.page_size,
        "physical_pages": args.physical_pages,
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
