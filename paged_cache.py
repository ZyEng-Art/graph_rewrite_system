from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefixHandle:
    """Logical ownership of one paged causal prefix."""

    blocks: tuple[int, ...]
    length: int


class PagedKVCache:
    """Fixed-size GPU KV/action pool with prefix sharing and COW tail pages."""

    def __init__(
        self,
        *,
        layers: int,
        capacity: int,
        page_size: int,
        heads: int,
        head_width: int,
        model_width: int,
        device: torch.device,
        dtype: torch.dtype,
        gather_backend: str = "vectorized",
    ):
        if min(layers, capacity, page_size, heads, head_width, model_width) <= 0:
            raise ValueError("all cache dimensions must be positive")
        if gather_backend not in {"loop", "vectorized"}:
            raise ValueError("gather_backend must be 'loop' or 'vectorized'")
        self.layers = layers
        self.capacity = capacity
        self.page_size = page_size
        self.heads = heads
        self.head_width = head_width
        self.model_width = model_width
        self.device = device
        self.dtype = dtype
        self.gather_backend = gather_backend
        self.keys = torch.empty(
            layers,
            capacity,
            page_size,
            heads,
            head_width,
            device=device,
            dtype=dtype,
        )
        self.values = torch.empty_like(self.keys)
        self.actions = torch.empty(
            capacity,
            page_size,
            model_width,
            device=device,
            dtype=dtype,
        )
        # Readout attention uses different projections from the causal action
        # decoder. Cache those projected K/V once when an action is appended so
        # the hot readout can consume physical pages directly.
        self.readout_keys = torch.empty(
            capacity,
            page_size,
            heads,
            head_width,
            device=device,
            dtype=dtype,
        )
        self.readout_values = torch.empty_like(self.readout_keys)
        self.refcounts = [0] * capacity
        self.free = list(range(capacity - 1, -1, -1))

    @property
    def allocated_pages(self) -> int:
        return self.capacity - len(self.free)

    @staticmethod
    def empty_handle() -> PrefixHandle:
        return PrefixHandle((), 0)

    def _allocate(self) -> int:
        if not self.free:
            raise RuntimeError("paged KV cache exhausted")
        page = self.free.pop()
        if self.refcounts[page] != 0:
            raise RuntimeError("free-list corruption")
        self.refcounts[page] = 1
        return page

    def _retain(self, page: int) -> None:
        if self.refcounts[page] <= 0:
            raise RuntimeError("retaining an unallocated page")
        self.refcounts[page] += 1

    def release(self, handle: PrefixHandle) -> None:
        for page in handle.blocks:
            self.refcounts[page] -= 1
            if self.refcounts[page] < 0:
                raise RuntimeError("negative page reference count")
            if self.refcounts[page] == 0:
                self.free.append(page)

    def append_batch(
        self,
        parents: list[PrefixHandle],
        keys: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        readout_keys: torch.Tensor | None = None,
        readout_values: torch.Tensor | None = None,
    ) -> list[PrefixHandle]:
        """Append one causal token to each parent.

        keys/values: [layers, batch, heads, head_width]
        actions: [batch, model_width]
        """
        batch_size = len(parents)
        expected = (self.layers, batch_size, self.heads, self.head_width)
        if tuple(keys.shape) != expected or tuple(values.shape) != expected:
            raise ValueError(f"expected KV shape {expected}")
        if tuple(actions.shape) != (batch_size, self.model_width):
            raise ValueError("invalid action-state shape")
        readout_shape = (batch_size, self.heads, self.head_width)
        if (readout_keys is None) != (readout_values is None):
            raise ValueError("readout keys and values must be supplied together")
        if readout_keys is not None and (
            tuple(readout_keys.shape) != readout_shape
            or tuple(readout_values.shape) != readout_shape
        ):
            raise ValueError(f"expected readout KV shape {readout_shape}")
        children = []
        write_pages = []
        write_offsets = []
        copy_sources = []
        copy_destinations = []
        for parent in parents:
            offset = parent.length % self.page_size
            if offset == 0:
                for page in parent.blocks:
                    self._retain(page)
                page = self._allocate()
                blocks = parent.blocks + (page,)
            else:
                # Parent remains live while its children branch. Full prefix pages
                # are shared; the mutable tail is copied exactly once per child.
                for shared in parent.blocks[:-1]:
                    self._retain(shared)
                old_tail = parent.blocks[-1]
                page = self._allocate()
                if self.gather_backend == "loop":
                    self.keys[:, page, :offset].copy_(
                        self.keys[:, old_tail, :offset]
                    )
                    self.values[:, page, :offset].copy_(
                        self.values[:, old_tail, :offset]
                    )
                    self.actions[page, :offset].copy_(
                        self.actions[old_tail, :offset]
                    )
                    if readout_keys is not None:
                        self.readout_keys[page, :offset].copy_(
                            self.readout_keys[old_tail, :offset]
                        )
                        self.readout_values[page, :offset].copy_(
                            self.readout_values[old_tail, :offset]
                        )
                else:
                    copy_sources.append(old_tail)
                    copy_destinations.append(page)
                blocks = parent.blocks[:-1] + (page,)
            children.append(PrefixHandle(blocks, parent.length + 1))
            write_pages.append(page)
            write_offsets.append(offset)

        # Copy all COW tail pages with three batched device operations instead
        # of issuing three tiny copies for every child. Bytes beyond the valid
        # tail offset are never observed because each handle carries its length.
        if copy_destinations:
            source_pages = torch.tensor(
                copy_sources, device=self.device, dtype=torch.long
            )
            destination_pages = torch.tensor(
                copy_destinations, device=self.device, dtype=torch.long
            )
            self.keys[:, destination_pages] = self.keys[:, source_pages]
            self.values[:, destination_pages] = self.values[:, source_pages]
            self.actions[destination_pages] = self.actions[source_pages]
            if readout_keys is not None:
                self.readout_keys[destination_pages] = self.readout_keys[
                    source_pages
                ]
                self.readout_values[destination_pages] = self.readout_values[
                    source_pages
                ]

        page_ids = torch.tensor(write_pages, device=self.device, dtype=torch.long)
        offsets = torch.tensor(write_offsets, device=self.device, dtype=torch.long)
        self.keys[:, page_ids, offsets] = keys.to(self.dtype)
        self.values[:, page_ids, offsets] = values.to(self.dtype)
        self.actions[page_ids, offsets] = actions.to(self.dtype)
        if readout_keys is not None:
            self.readout_keys[page_ids, offsets] = readout_keys.to(self.dtype)
            self.readout_values[page_ids, offsets] = readout_values.to(self.dtype)
        return children

    def block_table(
        self, handles: list[PrefixHandle]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded physical page ids and logical token lengths.

        The table always has at least one column. That permits a single Triton
        kernel specialization for an empty root prefix without ever reading the
        dummy page because its logical length is zero.
        """
        max_blocks = max((len(handle.blocks) for handle in handles), default=0)
        max_blocks = max(max_blocks, 1)
        rows = []
        lengths = []
        for handle in handles:
            required_blocks = (
                handle.length + self.page_size - 1
            ) // self.page_size
            if required_blocks != len(handle.blocks):
                raise RuntimeError("prefix handle length/block table mismatch")
            rows.append(
                (*handle.blocks, *((0,) * (max_blocks - len(handle.blocks))))
            )
            lengths.append(handle.length)
        return (
            torch.tensor(rows, device=self.device, dtype=torch.int32).reshape(
                len(handles), max_blocks
            ),
            torch.tensor(lengths, device=self.device, dtype=torch.int32),
        )

    def _token_indices(
        self, handles: list[PrefixHandle]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one device token-index table for a ragged handle batch."""
        batch_size = len(handles)
        max_length = max((handle.length for handle in handles), default=0)
        if not batch_size or not max_length:
            return (
                torch.empty(
                    (batch_size, max_length),
                    device=self.device,
                    dtype=torch.long,
                ),
                torch.zeros(
                    (batch_size, max_length),
                    device=self.device,
                    dtype=torch.bool,
                ),
            )
        block_table, length_tensor = self.block_table(handles)
        positions = torch.arange(max_length, device=self.device)
        block_columns = torch.div(
            positions, self.page_size, rounding_mode="floor"
        )
        pages = block_table.index_select(1, block_columns)
        token_indices = pages * self.page_size + positions.remainder(
            self.page_size
        ).unsqueeze(0)
        mask = positions.unsqueeze(0) < length_tensor.unsqueeze(1)
        return token_indices, mask

    def _gather_vectorized(
        self, handles: list[PrefixHandle]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_indices, mask = self._token_indices(handles)
        flat_keys = self.keys.reshape(
            self.layers,
            self.capacity * self.page_size,
            self.heads,
            self.head_width,
        )
        flat_values = self.values.reshape_as(flat_keys)
        keys = flat_keys[:, token_indices].permute(0, 1, 3, 2, 4)
        values = flat_values[:, token_indices].permute(0, 1, 3, 2, 4)
        kv_mask = mask[None, :, None, :, None]
        keys = keys.masked_fill(~kv_mask, 0)
        values = values.masked_fill(~kv_mask, 0)
        flat_actions = self.actions.reshape(
            self.capacity * self.page_size, self.model_width
        )
        actions = flat_actions[token_indices]
        actions = actions.masked_fill(~mask.unsqueeze(-1), 0)
        return keys, values, actions, mask

    def _gather_loop(
        self, handles: list[PrefixHandle]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference implementation retained for correctness/performance A/B."""
        batch_size = len(handles)
        max_length = max((handle.length for handle in handles), default=0)
        keys = torch.zeros(
            self.layers,
            batch_size,
            self.heads,
            max_length,
            self.head_width,
            device=self.device,
            dtype=self.dtype,
        )
        values = torch.zeros_like(keys)
        actions = torch.zeros(
            batch_size,
            max_length,
            self.model_width,
            device=self.device,
            dtype=self.dtype,
        )
        mask = torch.zeros(
            batch_size, max_length, device=self.device, dtype=torch.bool
        )
        for batch_index, handle in enumerate(handles):
            cursor = 0
            remaining = handle.length
            for page in handle.blocks:
                count = min(self.page_size, remaining)
                stop = cursor + count
                keys[:, batch_index, :, cursor:stop] = self.keys[
                    :, page, :count
                ].transpose(1, 2)
                values[:, batch_index, :, cursor:stop] = self.values[
                    :, page, :count
                ].transpose(1, 2)
                actions[batch_index, cursor:stop] = self.actions[page, :count]
                mask[batch_index, cursor:stop] = True
                cursor = stop
                remaining -= count
        return keys, values, actions, mask

    def gather(
        self, handles: list[PrefixHandle]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return contiguous KV/action tensors and a valid-token mask."""
        if self.gather_backend == "loop":
            return self._gather_loop(handles)
        return self._gather_vectorized(handles)

    def _gather_actions_loop(
        self, handles: list[PrefixHandle]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(handles)
        max_length = max((handle.length for handle in handles), default=0)
        actions = torch.zeros(
            batch_size,
            max_length,
            self.model_width,
            device=self.device,
            dtype=self.dtype,
        )
        mask = torch.zeros(
            batch_size, max_length, device=self.device, dtype=torch.bool
        )
        for batch_index, handle in enumerate(handles):
            cursor = 0
            remaining = handle.length
            for page in handle.blocks:
                count = min(self.page_size, remaining)
                actions[batch_index, cursor : cursor + count] = self.actions[
                    page, :count
                ]
                mask[batch_index, cursor : cursor + count] = True
                cursor += count
                remaining -= count
        return actions, mask

    def gather_actions(
        self, handles: list[PrefixHandle]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather only contextual action states for match-head readout."""
        if self.gather_backend == "loop":
            return self._gather_actions_loop(handles)
        token_indices, mask = self._token_indices(handles)
        flat_actions = self.actions.reshape(
            self.capacity * self.page_size, self.model_width
        )
        actions = flat_actions[token_indices]
        actions = actions.masked_fill(~mask.unsqueeze(-1), 0)
        return actions, mask
