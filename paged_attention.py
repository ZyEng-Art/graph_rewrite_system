from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU development and environments without Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _paged_attention_kernel(
        query,
        key_cache,
        value_cache,
        block_table,
        lengths,
        current_key,
        current_value,
        current_valid,
        output,
        stride_qb,
        stride_qh,
        stride_qq,
        stride_kp,
        stride_kt,
        stride_kh,
        stride_bt,
        stride_ckb,
        stride_ckh,
        stride_ob,
        stride_oh,
        stride_oq,
        num_heads: tl.constexpr,
        query_length,
        scale: tl.constexpr,
        has_current: tl.constexpr,
        page_size: tl.constexpr,
        head_width: tl.constexpr,
        block_q: tl.constexpr,
        block_t: tl.constexpr,
        block_d: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        query_block = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads

        offsets_q = query_block * block_q + tl.arange(0, block_q)
        offsets_t = tl.arange(0, block_t)
        offsets_d = tl.arange(0, block_d)
        query_mask = offsets_q < query_length
        width_mask = offsets_d < head_width

        query_ptrs = (
            query
            + batch * stride_qb
            + head * stride_qh
            + offsets_q[:, None] * stride_qq
            + offsets_d[None, :]
        )
        query_tile = tl.load(
            query_ptrs,
            mask=query_mask[:, None] & width_mask[None, :],
            other=0.0,
        )

        past_length = tl.load(lengths + batch)
        past_mask = offsets_t < past_length
        page_columns = offsets_t // page_size
        page_offsets = offsets_t - page_columns * page_size
        physical_pages = tl.load(
            block_table + batch * stride_bt + page_columns,
            mask=past_mask,
            other=0,
        )
        cache_offsets = (
            physical_pages[:, None] * stride_kp
            + page_offsets[:, None] * stride_kt
            + head * stride_kh
            + offsets_d[None, :]
        )
        key_tile = tl.load(
            key_cache + cache_offsets,
            mask=past_mask[:, None] & width_mask[None, :],
            other=0.0,
        )
        value_tile = tl.load(
            value_cache + cache_offsets,
            mask=past_mask[:, None] & width_mask[None, :],
            other=0.0,
        )

        if has_current:
            include_current = tl.load(current_valid + batch)
            current_key_ptrs = (
                current_key
                + batch * stride_ckb
                + head * stride_ckh
                + offsets_d
            )
            current_value_ptrs = (
                current_value
                + batch * stride_ckb
                + head * stride_ckh
                + offsets_d
            )
            current_key_row = tl.load(
                current_key_ptrs, mask=width_mask, other=0.0
            )
            current_value_row = tl.load(
                current_value_ptrs, mask=width_mask, other=0.0
            )
            current_position = offsets_t == past_length
            key_tile = tl.where(
                current_position[:, None],
                current_key_row[None, :],
                key_tile,
            )
            value_tile = tl.where(
                current_position[:, None],
                current_value_row[None, :],
                value_tile,
            )
            token_mask = past_mask | (current_position & include_current)
        else:
            token_mask = past_mask

        logits = tl.dot(query_tile, tl.trans(key_tile)) * scale
        logits = tl.where(token_mask[None, :], logits, -1.0e20)
        row_max = tl.max(logits, axis=1)
        probabilities = tl.exp(logits - row_max[:, None])
        probabilities = tl.where(
            token_mask[None, :], probabilities, 0.0
        )
        denominator = tl.sum(probabilities, axis=1)
        denominator = tl.maximum(denominator, 1.0)
        probabilities = probabilities / denominator[:, None]
        context = tl.dot(probabilities.to(tl.bfloat16), value_tile)

        output_ptrs = (
            output
            + batch * stride_ob
            + head * stride_oh
            + offsets_q[:, None] * stride_oq
            + offsets_d[None, :]
        )
        tl.store(
            output_ptrs,
            context,
            mask=query_mask[:, None] & width_mask[None, :],
        )


def _contiguous_from_pages(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """Reference gather: [pages, page, heads, width] -> [B, H, T, D]."""
    batch_size = block_table.shape[0]
    page_size = cache.shape[1]
    if not max_length:
        return cache.new_empty(
            batch_size, cache.shape[2], 0, cache.shape[3]
        )
    positions = torch.arange(max_length, device=cache.device)
    page_columns = torch.div(positions, page_size, rounding_mode="floor")
    pages = block_table.index_select(1, page_columns)
    token_indices = pages * page_size + positions.remainder(page_size)
    flat = cache.reshape(-1, cache.shape[2], cache.shape[3])
    return flat[token_indices].transpose(1, 2)


def paged_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    current_key: torch.Tensor | None = None,
    current_value: torch.Tensor | None = None,
    current_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialized PyTorch reference for tests and non-Triton devices."""
    max_past = int(lengths.max().item()) if lengths.numel() else 0
    keys = _contiguous_from_pages(key_cache, block_table, max_past)
    values = _contiguous_from_pages(value_cache, block_table, max_past)
    positions = torch.arange(max_past, device=query.device)
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    if current_key is not None:
        if current_value is None or current_valid is None:
            raise ValueError("current key/value/valid must be supplied together")
        keys = torch.cat((keys, current_key.unsqueeze(2)), dim=2)
        values = torch.cat((values, current_value.unsqueeze(2)), dim=2)
        mask = torch.cat((mask, current_valid.unsqueeze(1)), dim=1)
    if not keys.shape[2]:
        return torch.zeros_like(query)
    context = F.scaled_dot_product_attention(
        query,
        keys,
        values,
        attn_mask=mask[:, None, None, :],
        dropout_p=0.0,
    )
    return context * mask.any(1)[:, None, None, None]


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    current_key: torch.Tensor | None = None,
    current_value: torch.Tensor | None = None,
    current_valid: torch.Tensor | None = None,
    use_triton: bool = True,
) -> torch.Tensor:
    """Attend to physical pages without materializing a contiguous history."""
    has_current = current_key is not None
    if has_current != (current_value is not None and current_valid is not None):
        raise ValueError("current key/value/valid must be supplied together")
    if (
        not use_triton
        or triton is None
        or not query.is_cuda
        or query.requires_grad
    ):
        return paged_attention_reference(
            query,
            key_cache,
            value_cache,
            block_table,
            lengths,
            current_key=current_key,
            current_value=current_value,
            current_valid=current_valid,
        )
    if query.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16:
        return paged_attention_reference(
            query,
            key_cache,
            value_cache,
            block_table,
            lengths,
            current_key=current_key,
            current_value=current_value,
            current_valid=current_valid,
        )
    batch_size, heads, query_length, head_width = query.shape
    if not query_length:
        return torch.zeros_like(query)
    max_tokens = block_table.shape[1] * key_cache.shape[1] + int(has_current)
    block_t = max(16, triton.next_power_of_2(max_tokens))
    block_q = 16 if query_length <= 16 else 32
    block_d = max(16, triton.next_power_of_2(head_width))
    output = torch.empty_like(query)
    if not has_current:
        current_key = query.new_empty((1, 1, 1))
        current_value = current_key
        current_valid = torch.zeros(1, device=query.device, dtype=torch.bool)
    grid = (batch_size * heads, triton.cdiv(query_length, block_q))
    _paged_attention_kernel[grid](
        query,
        key_cache,
        value_cache,
        block_table,
        lengths,
        current_key,
        current_value,
        current_valid,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        block_table.stride(0),
        current_key.stride(0),
        current_key.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads=heads,
        query_length=query_length,
        scale=1.0 / math.sqrt(head_width),
        has_current=has_current,
        page_size=key_cache.shape[1],
        head_width=head_width,
        block_q=block_q,
        block_t=block_t,
        block_d=block_d,
        num_warps=4,
    )
    return output
