from __future__ import annotations

import torch

from paged_cache import PagedKVCache


def token(value: float, layers: int, heads: int, head_width: int, width: int):
    keys = torch.full((layers, 1, heads, head_width), value, dtype=torch.float32)
    values = torch.full_like(keys, value + 0.25)
    action = torch.full((1, width), value + 0.5)
    readout_key = torch.full((1, heads, head_width), value + 0.75)
    readout_value = torch.full_like(readout_key, value + 1.0)
    return keys, values, action, readout_key, readout_value


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers, heads, head_width, width = 2, 2, 4, 8
    arena = PagedKVCache(
        layers=layers,
        capacity=16,
        page_size=4,
        heads=heads,
        head_width=head_width,
        model_width=width,
        device=device,
        dtype=torch.float32,
    )
    handle = arena.empty_handle()
    chain = [handle]
    for value in range(1, 5):
        tensors = token(value, layers, heads, head_width, width)
        child = arena.append_batch([handle], *[part.to(device) for part in tensors])[0]
        chain.append(child)
        handle = child
    assert handle.length == 4 and len(handle.blocks) == 1

    left_tensors = token(5, layers, heads, head_width, width)
    right_tensors = token(9, layers, heads, head_width, width)
    left, right = arena.append_batch(
        [handle, handle],
        torch.cat((left_tensors[0], right_tensors[0]), dim=1).to(device),
        torch.cat((left_tensors[1], right_tensors[1]), dim=1).to(device),
        torch.cat((left_tensors[2], right_tensors[2]), dim=0).to(device),
        torch.cat((left_tensors[3], right_tensors[3]), dim=0).to(device),
        torch.cat((left_tensors[4], right_tensors[4]), dim=0).to(device),
    )
    assert left.blocks[0] == right.blocks[0] == handle.blocks[0]
    assert left.blocks[-1] != right.blocks[-1]
    keys, values, actions, mask = arena.gather([left, right])
    assert tuple(keys.shape) == (layers, 2, heads, 5, head_width)
    assert mask.all()
    assert torch.allclose(actions[0, :4], torch.tensor([1.5, 2.5, 3.5, 4.5], device=device)[:, None].expand(-1, width))
    assert torch.all(actions[0, 4] == 5.5)
    assert torch.all(actions[1, 4] == 9.5)
    assert torch.all(values[:, 0, :, 4] == 5.25)
    assert torch.all(values[:, 1, :, 4] == 9.25)
    block_table, lengths = arena.block_table([left, right])
    assert block_table.tolist() == [list(left.blocks), list(right.blocks)]
    assert lengths.tolist() == [5, 5]
    assert torch.all(arena.readout_keys[left.blocks[-1], 0] == 5.75)
    assert torch.all(arena.readout_values[right.blocks[-1], 0] == 10.0)

    # Ragged batches, including an empty prefix, must preserve zero padding and
    # agree between the vectorized action-only and full-KV gather paths.
    ragged = [chain[0], chain[2], left]
    ragged_keys, ragged_values, ragged_actions, ragged_mask = arena.gather(ragged)
    action_only, action_mask = arena.gather_actions(ragged)
    assert tuple(ragged_keys.shape) == (layers, 3, heads, 5, head_width)
    assert tuple(ragged_values.shape) == tuple(ragged_keys.shape)
    assert torch.equal(ragged_mask, action_mask)
    assert torch.equal(ragged_actions, action_only)
    assert not ragged_mask[0].any()
    assert ragged_mask[1, :2].all() and not ragged_mask[1, 2:].any()
    assert ragged_mask[2].all()
    assert not ragged_actions[~ragged_mask].any()

    arena.gather_backend = "loop"
    loop_keys, loop_values, loop_actions, loop_mask = arena.gather(ragged)
    assert torch.equal(ragged_keys, loop_keys)
    assert torch.equal(ragged_values, loop_values)
    assert torch.equal(ragged_actions, loop_actions)
    assert torch.equal(ragged_mask, loop_mask)

    for old in chain[1:]:
        arena.release(old)
    arena.release(left)
    arena.release(right)
    assert arena.allocated_pages == 0
    print("paged cache ok: shared full prefix, COW tail, gather, reclaim")


if __name__ == "__main__":
    main()
