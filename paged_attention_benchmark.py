from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from paged_attention import paged_attention


def synchronize() -> None:
    torch.cuda.synchronize()


def measure(function, warmup: int, repeats: int) -> tuple[float, int]:
    for _ in range(warmup):
        output = function()
    synchronize()
    del output
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    started = time.perf_counter()
    for _ in range(repeats):
        output = function()
    synchronize()
    seconds = (time.perf_counter() - started) / repeats
    peak = torch.cuda.max_memory_allocated() - baseline
    del output
    return seconds, peak


def gather_pages(
    cache: torch.Tensor, block_table: torch.Tensor, length: int
) -> torch.Tensor:
    page_size = cache.shape[1]
    positions = torch.arange(length, device=cache.device)
    columns = torch.div(positions, page_size, rounding_mode="floor")
    pages = block_table.index_select(1, columns)
    indices = pages * page_size + positions.remainder(page_size)
    flat = cache.reshape(-1, cache.shape[2], cache.shape[3])
    return flat[indices].transpose(1, 2)


def benchmark_case(
    batch_size: int,
    *,
    history_length: int,
    query_length: int,
    page_size: int,
    heads: int,
    head_width: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed + batch_size + query_length)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    blocks = math.ceil(history_length / page_size)
    capacity = batch_size * blocks
    keys = torch.randn(
        capacity, page_size, heads, head_width, device=device, dtype=dtype
    )
    values = torch.randn_like(keys)
    block_table = torch.arange(
        capacity, device=device, dtype=torch.int32
    ).view(batch_size, blocks)
    lengths = torch.full(
        (batch_size,), history_length, device=device, dtype=torch.int32
    )
    query = torch.randn(
        batch_size,
        heads,
        query_length,
        head_width,
        device=device,
        dtype=dtype,
    )

    def contiguous_readout():
        contiguous_keys = gather_pages(keys, block_table, history_length)
        contiguous_values = gather_pages(values, block_table, history_length)
        return F.scaled_dot_product_attention(
            query, contiguous_keys, contiguous_values, dropout_p=0.0
        )

    def direct_readout():
        return paged_attention(
            query, keys, values, block_table, lengths
        )

    expected = contiguous_readout()
    actual = direct_readout()
    synchronize()
    error = float((actual.float() - expected.float()).abs().max().item())
    contiguous_seconds, contiguous_peak = measure(
        contiguous_readout, warmup, repeats
    )
    direct_seconds, direct_peak = measure(direct_readout, warmup, repeats)
    query_vectors = batch_size * query_length
    return {
        "batch_size": batch_size,
        "history_length": history_length,
        "query_length": query_length,
        "contiguous_gather_sdpa_ms": 1000 * contiguous_seconds,
        "direct_paged_ms": 1000 * direct_seconds,
        "speedup": contiguous_seconds / direct_seconds,
        "contiguous_query_vectors_per_second": query_vectors / contiguous_seconds,
        "direct_query_vectors_per_second": query_vectors / direct_seconds,
        "contiguous_peak_bytes": contiguous_peak,
        "direct_peak_bytes": direct_peak,
        "max_abs_error": error,
    }


def benchmark_causal_case(
    batch_size: int,
    *,
    history_length: int,
    page_size: int,
    heads: int,
    head_width: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed + batch_size)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    blocks = max(1, math.ceil(history_length / page_size))
    capacity = batch_size * blocks
    keys = torch.randn(
        capacity, page_size, heads, head_width, device=device, dtype=dtype
    )
    values = torch.randn_like(keys)
    block_table = torch.arange(
        capacity, device=device, dtype=torch.int32
    ).view(batch_size, blocks)
    lengths = torch.full(
        (batch_size,), history_length, device=device, dtype=torch.int32
    )
    query = torch.randn(
        batch_size, heads, 1, head_width, device=device, dtype=dtype
    )
    current_key = torch.randn(
        batch_size, heads, head_width, device=device, dtype=dtype
    )
    current_value = torch.randn_like(current_key)
    current_valid = torch.ones(batch_size, device=device, dtype=torch.bool)

    def contiguous_causal():
        contiguous_keys = gather_pages(keys, block_table, history_length)
        contiguous_values = gather_pages(values, block_table, history_length)
        all_keys = torch.cat((contiguous_keys, current_key.unsqueeze(2)), dim=2)
        all_values = torch.cat(
            (contiguous_values, current_value.unsqueeze(2)), dim=2
        )
        return F.scaled_dot_product_attention(
            query, all_keys, all_values, dropout_p=0.0
        )

    def direct_causal():
        return paged_attention(
            query,
            keys,
            values,
            block_table,
            lengths,
            current_key=current_key,
            current_value=current_value,
            current_valid=current_valid,
        )

    expected = contiguous_causal()
    actual = direct_causal()
    synchronize()
    error = float((actual.float() - expected.float()).abs().max().item())
    contiguous_seconds, contiguous_peak = measure(
        contiguous_causal, warmup, repeats
    )
    direct_seconds, direct_peak = measure(direct_causal, warmup, repeats)
    return {
        "batch_size": batch_size,
        "history_length": history_length,
        "contiguous_gather_sdpa_ms": 1000 * contiguous_seconds,
        "direct_paged_ms": 1000 * direct_seconds,
        "speedup": contiguous_seconds / direct_seconds,
        "contiguous_tokens_per_second": batch_size / contiguous_seconds,
        "direct_tokens_per_second": batch_size / direct_seconds,
        "contiguous_peak_bytes": contiguous_peak,
        "direct_peak_bytes": direct_peak,
        "max_abs_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 512, 1000])
    parser.add_argument("--history-length", type=int, default=64)
    parser.add_argument("--query-length", type=int, default=381)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--head-width", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    rows = {"readout": [], "causal_append": []}
    for batch_size in args.batch_sizes:
        readout = benchmark_case(
            batch_size,
            history_length=args.history_length,
            query_length=args.query_length,
            page_size=args.page_size,
            heads=args.heads,
            head_width=args.head_width,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
        causal = benchmark_causal_case(
            batch_size,
            history_length=args.history_length,
            page_size=args.page_size,
            heads=args.heads,
            head_width=args.head_width,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
        rows["readout"].append(readout)
        rows["causal_append"].append(causal)
        print(json.dumps({"readout": readout, "causal_append": causal}))
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "page_size": args.page_size,
        "heads": args.heads,
        "head_width": args.head_width,
        "warmup": args.warmup,
        "repeats": args.repeats,
        **rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
