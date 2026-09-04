from __future__ import annotations

import torch

from paged_attention import paged_attention, paged_attention_reference


def _case(device: torch.device, *, with_current: bool) -> None:
    torch.manual_seed(7)
    batch_size = 5
    capacity = 12
    page_size = 8
    heads = 6
    head_width = 32
    query_length = 37
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    keys = torch.randn(
        capacity, page_size, heads, head_width, device=device, dtype=dtype
    )
    values = torch.randn_like(keys)
    # Rows intentionally share physical pages. The final padded column must
    # never be observed for the shorter rows.
    block_table = torch.tensor(
        [[0, 6, 9], [1, 6, 8], [2, 7, 9], [0, 6, 8], [3, 4, 5]],
        device=device,
        dtype=torch.int32,
    )
    lengths = torch.tensor([0, 3, 8, 9, 19], device=device, dtype=torch.int32)
    query = torch.randn(
        batch_size,
        heads,
        query_length,
        head_width,
        device=device,
        dtype=dtype,
    )
    kwargs = {}
    if with_current:
        kwargs = {
            "current_key": torch.randn(
                batch_size, heads, head_width, device=device, dtype=dtype
            ),
            "current_value": torch.randn(
                batch_size, heads, head_width, device=device, dtype=dtype
            ),
            "current_valid": torch.tensor(
                [True, True, False, True, True], device=device
            ),
        }
    expected = paged_attention_reference(
        query, keys, values, block_table, lengths, **kwargs
    )
    actual = paged_attention(
        query, keys, values, block_table, lengths, **kwargs
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    # A zero-length row with no appended token has no valid attention source.
    if not with_current:
        assert torch.count_nonzero(actual[0]) == 0


def test_paged_attention_history_only() -> None:
    _case(torch.device("cuda" if torch.cuda.is_available() else "cpu"), with_current=False)


def test_paged_attention_with_current_token() -> None:
    _case(torch.device("cuda" if torch.cuda.is_available() else "cpu"), with_current=True)


if __name__ == "__main__":
    test_paged_attention_history_only()
    test_paged_attention_with_current_token()
    print("paged attention tests passed")
