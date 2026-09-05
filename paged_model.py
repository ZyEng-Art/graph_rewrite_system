from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from dataset import RuleMetadata
from model import InitialGraphLayer, S0ActionBindingModel
from paged_attention import paged_attention


class CausalActionLayer(nn.Module):
    """A decoder-only attention block with an explicit one-token cache API."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.input_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        token: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        key_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append one token; keys/values are [batch, heads, time, head_width]."""
        batch_size = token.shape[0]
        qkv = self.qkv(self.input_norm(token)).view(
            batch_size, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(1)
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
        if past_key is not None:
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        scores = torch.einsum("bhd,bhtd->bht", query, key)
        scores = scores / math.sqrt(self.head_width)
        scores = scores.masked_fill(~key_mask.unsqueeze(1), -1e4)
        attention = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attention = self.attention_dropout(attention)
        context = torch.einsum("bht,bhtd->bhd", attention, value)
        context = context.reshape(batch_size, self.width)
        token = token + self.residual_dropout(self.output(context))
        token = token + self.residual_dropout(self.ffn(self.ffn_norm(token)))
        return token, key, value

    def forward_step_paged(
        self,
        token: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        past_lengths: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append one token while attending directly to physical cache pages."""
        batch_size = token.shape[0]
        qkv = self.qkv(self.input_norm(token)).view(
            batch_size, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(1)
        context = paged_attention(
            query.unsqueeze(2),
            key_cache,
            value_cache,
            block_table,
            past_lengths,
            current_key=key,
            current_value=value,
            current_valid=active,
        ).squeeze(2)
        context = context.reshape(batch_size, self.width)
        token = token + self.residual_dropout(self.output(context))
        token = token + self.residual_dropout(self.ffn(self.ffn_norm(token)))
        return token, key, value


class PagedActionBindingModel(S0ActionBindingModel):
    """True s0 + causal-action model; current topology is not encoded by the model.

    The inherited structural decoder may still consume a graph derived from the
    action sequence after retrieval. This keeps topology validation outside the
    learned hot path while preserving the old output contract.
    """

    def __init__(
        self,
        rules: RuleMetadata,
        *,
        num_xfers: int,
        width: int = 192,
        retrieval_width: int = 128,
        graph_layers: int = 3,
        action_layers: int = 4,
        action_heads: int = 6,
        max_sequence_length: int = 256,
        ordered_binding_roles: bool = False,
        readout_graph_layers: int = 0,
        readout_graph_input: str = "cached",
        readout_locality_features: bool = False,
        identity_readout_prefix: int = 0,
        readout_attention_backend: str = "sdpa",
        action_value_head: bool = False,
        dropout: float = 0.05,
    ):
        super().__init__(
            rules,
            num_xfers=num_xfers,
            width=width,
            retrieval_width=retrieval_width,
            graph_layers=graph_layers,
            current_graph_layers=0,
            dropout=dropout,
            use_action_history=True,
            use_locality_features=readout_locality_features,
        )
        self.action_layers_count = action_layers
        self.action_heads = action_heads
        self.max_sequence_length = max_sequence_length
        self.ordered_binding_roles = ordered_binding_roles
        self.readout_graph_layers_count = readout_graph_layers
        self.readout_locality_features = readout_locality_features
        self.identity_readout_prefix = identity_readout_prefix
        if readout_attention_backend not in {
            "eager",
            "sdpa",
            "sdpa_live",
            "paged",
        }:
            raise ValueError(
                "readout_attention_backend must be 'eager', 'sdpa', or "
                "'sdpa_live', or 'paged'"
            )
        self.readout_attention_backend = readout_attention_backend
        self.has_action_value_head = action_value_head
        if not 0 <= identity_readout_prefix <= readout_graph_layers:
            raise ValueError("identity_readout_prefix exceeds readout graph depth")
        if readout_graph_input not in {"cached", "gate", "cached_gate"}:
            raise ValueError(
                "readout_graph_input must be 'cached', 'gate', or 'cached_gate'"
            )
        self.readout_graph_input = readout_graph_input
        if readout_graph_input == "cached_gate":
            # Begin exactly at the cached-state checkpoint, then let training
            # restore clean gate identity where repeated action updates drift.
            self.readout_gate_scale = nn.Parameter(torch.zeros(width))
        if ordered_binding_roles:
            self.binding_role_delta = nn.Sequential(
                nn.Linear(2 * width, 2 * width),
                nn.GELU(),
                nn.Linear(2 * width, width),
            )
            # Preserve the v2 function exactly at initialization so that a v2
            # checkpoint can be fine-tuned without an abrupt representation shift.
            nn.init.zeros_(self.binding_role_delta[-1].weight)
            nn.init.zeros_(self.binding_role_delta[-1].bias)
        self.action_position = nn.Embedding(max_sequence_length, width)
        self.causal_action_layers = nn.ModuleList(
            CausalActionLayer(width, action_heads, dropout)
            for _ in range(action_layers)
        )
        self.action_output_norm = nn.LayerNorm(width)
        self.node_action_query = nn.Linear(width, width, bias=False)
        self.node_action_key = nn.Linear(width, width, bias=False)
        self.node_action_value = nn.Linear(width, width, bias=False)
        self.node_action_output = nn.Linear(width, width, bias=False)
        self.node_action_norm = nn.LayerNorm(width)
        nn.init.zeros_(self.node_action_output.weight)
        if readout_graph_layers:
            self.readout_graph_layers = nn.ModuleList(
                InitialGraphLayer(width) for _ in range(readout_graph_layers)
            )
            for layer in self.readout_graph_layers[identity_readout_prefix:]:
                if identity_readout_prefix:
                    nn.init.zeros_(layer.update[-1].weight)
                    nn.init.zeros_(layer.update[-1].bias)
            self.readout_graph_output = nn.Linear(width, width, bias=False)
            nn.init.zeros_(self.readout_graph_output.weight)

        # Full current-graph fusion is intentionally absent from the new path.
        del self.current_graph_layers
        del self.current_fusion
        del self.current_norm
        if action_value_head:
            self.action_value_norm = nn.LayerNorm(4 * width)
            self.action_value_mlp = nn.Sequential(
                nn.Linear(4 * width, width),
                nn.GELU(),
                nn.Linear(width, 1),
            )

    def candidate_features(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        xfer_ids: torch.Tensor,
        source_ids: torch.Tensor,
        binding_slots: torch.Tensor,
        batch_ids: torch.Tensor | None = None,
        ordered_roles: bool = False,
    ) -> torch.Tensor:
        """Build candidate features from the current causal graph state."""
        if batch_ids is None:
            batch_ids = torch.arange(states.shape[0], device=states.device)
        binding_mask = binding_slots.ge(0)
        safe_bindings = binding_slots.clamp_min(0)
        num_slots = states.shape[1]
        flat_indices = batch_ids.unsqueeze(1) * num_slots + safe_bindings
        bound_states = states.reshape(-1, self.width)[flat_indices]
        bound_states = bound_states.masked_fill(~binding_mask.unsqueeze(-1), 0)
        if ordered_roles:
            bound_states = self._ordered_bound_states(
                bound_states, binding_mask
            )
        bound_pool = bound_states.sum(1)
        bound_pool = bound_pool / binding_mask.sum(1, keepdim=True).clamp_min(1)
        live_states = states.masked_fill(~live.unsqueeze(-1), 0)
        graph_pool = live_states.sum(1)
        graph_pool = graph_pool / live.sum(1, keepdim=True).clamp_min(1)
        graph_pool = graph_pool.index_select(0, batch_ids)
        source_states = self.source_representations().index_select(0, source_ids)
        xfer_states = self.xfer_embedding(xfer_ids)
        return torch.cat(
            (xfer_states, source_states, bound_pool, graph_pool), dim=-1
        )

    def action_value_features(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        xfer_ids: torch.Tensor,
        source_ids: torch.Tensor,
        binding_slots: torch.Tensor,
        batch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.has_action_value_head:
            raise RuntimeError("model checkpoint has no action-value head")
        return self.candidate_features(
            states,
            live,
            xfer_ids,
            source_ids,
            binding_slots,
            batch_ids,
        )

    def action_values(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        xfer_ids: torch.Tensor,
        source_ids: torch.Tensor,
        binding_slots: torch.Tensor,
        batch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.action_value_features(
            states,
            live,
            xfer_ids,
            source_ids,
            binding_slots,
            batch_ids,
        )
        return self.action_value_mlp(self.action_value_norm(features)).squeeze(-1)

    def _ordered_bound_states(
        self, bound_states: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        """Associate each bound node with its ordered source-pattern role."""
        if not self.ordered_binding_roles:
            return bound_states
        positions = self.pattern_position.weight[: bound_states.shape[1]]
        positions = positions.unsqueeze(0).expand(bound_states.shape[0], -1, -1)
        delta = self.binding_role_delta(torch.cat((bound_states, positions), dim=-1))
        return (bound_states + delta) * source_mask.unsqueeze(-1)

    def _update_live_nodes(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        bound_states: torch.Tensor,
        source_mask: torch.Tensor,
        action_context: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Increment the cached node state without a current-graph GNN pass."""
        queries = self.node_query(states)
        keys = self.binding_key(bound_states)
        attention_logits = torch.einsum("bsd,bkd->bsk", queries, keys)
        attention_logits = attention_logits / math.sqrt(self.width)
        attention_logits = attention_logits.masked_fill(
            ~source_mask.unsqueeze(1), -1e4
        )
        attention = attention_logits.softmax(-1)
        local_context = torch.einsum(
            "bsk,bkd->bsd", attention, self.binding_value(bound_states)
        )
        affinity = attention_logits.max(-1).values.sigmoid().unsqueeze(-1)
        expanded = action_context.unsqueeze(1).expand(-1, states.shape[1], -1)
        delta = self.action_delta(
            torch.cat((states, local_context, expanded), dim=-1)
        )
        update_mask = live & active.unsqueeze(-1)
        updated = self.action_norm(states + affinity * delta)
        return torch.where(update_mask.unsqueeze(-1), updated, states)

    def _initial_state(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial_types = batch["initial_types"]
        live = initial_types.ge(0)
        gate_types = initial_types.clone()
        states = self.gate_embedding(initial_types.clamp_min(0)) + self.initial_marker
        states = states * live.unsqueeze(-1)
        for layer in self.graph_layers:
            states = layer(
                states,
                live,
                batch["edge_batch"],
                batch["edge_src"],
                batch["edge_dst"],
                batch["edge_relation"],
            )
        return states, live, gate_types

    def _fuse_action_history(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        action_states: torch.Tensor,
        action_mask: torch.Tensor,
        live_slot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_states.shape[1] == 0:
            return states * live.unsqueeze(-1)
        batch_size, num_slots, _ = states.shape
        compact_live = (
            self.readout_attention_backend == "sdpa_live"
            and live_slot_indices is not None
        )
        if compact_live:
            if live_slot_indices.shape[0] != batch_size:
                raise ValueError("live-slot index batch differs from state batch")
            live_slot_mask = live_slot_indices.ge(0)
            if not live_slot_indices.shape[1]:
                return torch.zeros_like(states)
            safe_live_slots = live_slot_indices.clamp_min(0)
            attention_states = torch.gather(
                states,
                1,
                safe_live_slots.unsqueeze(-1).expand(-1, -1, self.width),
            )
            attention_states = attention_states * live_slot_mask.unsqueeze(-1)
        else:
            attention_states = states
        heads = self.action_heads
        head_width = self.width // heads
        query_slots = attention_states.shape[1]
        query = self.node_action_query(attention_states).view(
            batch_size, query_slots, heads, head_width
        ).transpose(1, 2)
        key = self.node_action_key(action_states).view(
            batch_size, -1, heads, head_width
        ).transpose(1, 2)
        value = self.node_action_value(action_states).view(
            batch_size, -1, heads, head_width
        ).transpose(1, 2)
        has_history = action_mask.any(1)[:, None, None, None]
        if self.readout_attention_backend in {"sdpa", "sdpa_live", "paged"}:
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=action_mask[:, None, None, :],
                dropout_p=0.0,
            )
            context = context * has_history
        else:
            scores = torch.einsum("bhnd,bhtd->bhnt", query, key)
            scores = scores / math.sqrt(head_width)
            scores = scores.masked_fill(~action_mask[:, None, None, :], -1e4)
            attention = F.softmax(scores.float(), dim=-1).to(scores.dtype)
            attention = attention * has_history
            context = torch.einsum("bhnt,bhtd->bhnd", attention, value)
        context = context.transpose(1, 2).reshape(
            batch_size, query_slots, self.width
        )
        attention_states = self.node_action_norm(
            attention_states + self.node_action_output(context)
        )
        if compact_live:
            attention_states = attention_states * live_slot_mask.unsqueeze(-1)
            states = torch.zeros_like(states).scatter_add(
                1,
                safe_live_slots.unsqueeze(-1).expand(-1, -1, self.width),
                attention_states,
            )
        else:
            states = attention_states
        return states * live.unsqueeze(-1)

    def project_action_readout_kv(
        self, action_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a new contextual action once for paged node readout."""
        batch_size = action_states.shape[0]
        shape = (batch_size, self.action_heads, self.width // self.action_heads)
        return (
            self.node_action_key(action_states).view(shape),
            self.node_action_value(action_states).view(shape),
        )

    def _fuse_action_history_paged(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        readout_key_cache: torch.Tensor,
        readout_value_cache: torch.Tensor,
        block_table: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Node-to-action attention without gathering action history."""
        batch_size, num_slots, _ = states.shape
        heads = self.action_heads
        head_width = self.width // heads
        query = self.node_action_query(states).view(
            batch_size, num_slots, heads, head_width
        ).transpose(1, 2)
        context = paged_attention(
            query,
            readout_key_cache,
            readout_value_cache,
            block_table,
            lengths,
        )
        context = context.transpose(1, 2).reshape(
            batch_size, num_slots, self.width
        )
        updated = self.node_action_norm(
            states + self.node_action_output(context)
        )
        states = torch.where(lengths[:, None, None].gt(0), updated, states)
        return states * live.unsqueeze(-1)

    def _fuse_readout_graph(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        gate_types: torch.Tensor,
        batch: dict | None,
    ) -> torch.Tensor:
        """Fuse the lightweight action-derived graph only at match readout."""
        if not self.readout_graph_layers_count:
            return states
        if batch is None:
            raise ValueError("readout graph layers require an action-derived graph")
        if self.readout_graph_input == "gate":
            graph_states = (
                self.gate_embedding(gate_types.clamp_min(0))
                + self.current_graph_marker
            ) * live.unsqueeze(-1)
        else:
            graph_states = states
            if self.readout_graph_input == "cached_gate":
                clean_gate_states = (
                    self.gate_embedding(gate_types.clamp_min(0))
                    + self.current_graph_marker
                ) * live.unsqueeze(-1)
                graph_states = (
                    graph_states
                    + self.readout_gate_scale * clean_gate_states
                )
        if self.readout_locality_features:
            streak = self.local_streak_embedding(batch["current_local_streak"])
            streak = streak.unsqueeze(1).expand(-1, gate_types.shape[1], -1)
            locality = self.locality_fusion(
                torch.cat(
                    (
                        self.rewrite_distance_embedding(
                            batch["current_rewrite_distance"]
                        ),
                        self.touch_age_embedding(batch["current_touch_age"]),
                        streak,
                    ),
                    dim=-1,
                )
            )
            graph_states = graph_states + locality * live.unsqueeze(-1)
        for layer in self.readout_graph_layers:
            graph_states = layer(
                graph_states,
                live,
                batch["current_edge_batch"],
                batch["current_edge_src"],
                batch["current_edge_dst"],
                batch["current_edge_relation"],
            )
        return (
            states + self.readout_graph_output(graph_states)
        ) * live.unsqueeze(-1)

    def initialize_incremental(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode s0 once for incremental/paged decoding."""
        return self._initial_state(batch)

    def advance_incremental(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        gate_types: torch.Tensor,
        past_keys: torch.Tensor | None,
        past_values: torch.Tensor | None,
        past_action_states: torch.Tensor,
        past_mask: torch.Tensor,
        *,
        xfer_ids: torch.Tensor,
        source_ids: torch.Tensor,
        source_slots: torch.Tensor,
        destination_slots: torch.Tensor,
        destination_types: torch.Tensor,
        paged_key_cache: torch.Tensor | None = None,
        paged_value_cache: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        past_lengths: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Append exactly one action and return only its new per-layer KV.

        Past tensors may come from `PagedKVCache.gather`; no earlier action is
        reprojected or passed through the causal layers again.
        """
        batch_size = states.shape[0]
        active = xfer_ids.ge(0)
        paged_inputs = (
            paged_key_cache,
            paged_value_cache,
            block_table,
            past_lengths,
        )
        use_paged_attention = any(value is not None for value in paged_inputs)
        if use_paged_attention and not all(value is not None for value in paged_inputs):
            raise ValueError("all paged attention inputs must be supplied together")
        next_position = (
            past_lengths.long() if use_paged_attention else past_mask.sum(1)
        )
        if bool(next_position.ge(self.max_sequence_length).any()):
            raise ValueError("incremental prefix exceeds max_sequence_length")

        referenced = torch.cat((source_slots, destination_slots), dim=1)
        max_referenced = int(referenced.max().item()) if referenced.numel() else -1
        if max_referenced >= states.shape[1]:
            extra = max_referenced + 1 - states.shape[1]
            states = F.pad(states, (0, 0, 0, extra))
            live = F.pad(live, (0, extra), value=False)
            gate_types = F.pad(gate_types, (0, extra), value=-1)

        source_ids = source_ids.clamp_min(0)
        source_mask = source_slots.ge(0) & active.unsqueeze(-1)
        safe_source_slots = source_slots.clamp_min(0)
        bound_states = torch.gather(
            states,
            1,
            safe_source_slots.unsqueeze(-1).expand(-1, -1, self.width),
        )
        bound_states = self._ordered_bound_states(bound_states, source_mask)
        bound_pool = (bound_states * source_mask.unsqueeze(-1)).sum(1)
        bound_pool = bound_pool / source_mask.sum(1, keepdim=True).clamp_min(1)
        source_representations = self.source_representations()
        raw_context = self.action_context(
            torch.cat(
                (
                    self.xfer_embedding(xfer_ids.clamp_min(0)),
                    source_representations[source_ids],
                    bound_pool,
                ),
                dim=-1,
            )
        )
        token = raw_context + self.action_position(next_position)
        token = token * active.unsqueeze(-1)
        if use_paged_attention:
            max_past = int(past_lengths.max().item()) if batch_size else 0
            positions = torch.arange(max_past, device=states.device)
            history_mask = positions.unsqueeze(0) < past_lengths.unsqueeze(1)
            next_mask = torch.cat((history_mask, active.unsqueeze(1)), dim=1)
        else:
            next_mask = torch.cat((past_mask, active.unsqueeze(1)), dim=1)
        new_keys = []
        new_values = []
        for layer_index, layer in enumerate(self.causal_action_layers):
            if use_paged_attention:
                token, key, value = layer.forward_step_paged(
                    token,
                    paged_key_cache[layer_index],
                    paged_value_cache[layer_index],
                    block_table,
                    past_lengths,
                    active,
                )
            else:
                layer_past_key = (
                    None if past_keys is None else past_keys[layer_index]
                )
                layer_past_value = (
                    None if past_values is None else past_values[layer_index]
                )
                token, key, value = layer.forward_step(
                    token, layer_past_key, layer_past_value, next_mask
                )
            token = token * active.unsqueeze(-1)
            new_keys.append(key if use_paged_attention else key[:, :, -1])
            new_values.append(value if use_paged_attention else value[:, :, -1])
        token = self.action_output_norm(token) * active.unsqueeze(-1)

        states = self._update_live_nodes(
            states,
            live,
            bound_states,
            source_mask,
            raw_context,
            active,
        )

        next_states = states.clone()
        next_live = live.clone()
        next_gate_types = gate_types.clone()
        batch_ids = torch.arange(batch_size, device=states.device)
        for position in range(source_slots.shape[1]):
            valid = source_mask[:, position]
            if bool(valid.any()):
                rows = batch_ids[valid]
                slots = source_slots[valid, position]
                next_live[rows, slots] = False
                next_gate_types[rows, slots] = -1
                next_states[rows, slots] = 0
        destination_mask = destination_slots.ge(0) & active.unsqueeze(-1)
        for position in range(destination_slots.shape[1]):
            valid = destination_mask[:, position]
            if not bool(valid.any()):
                continue
            rows = batch_ids[valid]
            slots = destination_slots[valid, position]
            types = destination_types[valid, position]
            pattern_position = self.pattern_position.weight[position].expand(
                rows.numel(), -1
            )
            created = self.destination_state(
                torch.cat(
                    (
                        raw_context[valid],
                        self.gate_embedding(types),
                        pattern_position,
                    ),
                    dim=-1,
                )
            )
            next_states[rows, slots] = created.to(next_states.dtype)
            next_live[rows, slots] = True
            next_gate_types[rows, slots] = types
        return (
            next_states,
            next_live,
            next_gate_types,
            torch.stack(new_keys),
            torch.stack(new_values),
            token,
            next_mask,
        )

    def readout_incremental(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        gate_types: torch.Tensor,
        action_states: torch.Tensor,
        action_mask: torch.Tensor,
        batch: dict | None = None,
        *,
        readout_key_cache: torch.Tensor | None = None,
        readout_value_cache: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        paged_inputs = (
            readout_key_cache,
            readout_value_cache,
            block_table,
            lengths,
        )
        if any(value is not None for value in paged_inputs):
            if not all(value is not None for value in paged_inputs):
                raise ValueError("all paged readout inputs must be supplied together")
            states = self._fuse_action_history_paged(
                states,
                live,
                readout_key_cache,
                readout_value_cache,
                block_table,
                lengths,
            )
        else:
            states = self._fuse_action_history(
                states,
                live,
                action_states,
                action_mask,
                None if batch is None else batch.get("current_live_slots"),
            )
        states = self._fuse_readout_graph(states, live, gate_types, batch)
        return states, live, gate_types

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, live, gate_types = self._initial_state(batch)
        source_representations = self.source_representations()
        max_actions = batch["action_xfers"].shape[1]
        if max_actions > self.max_sequence_length:
            raise ValueError(
                f"prefix length {max_actions} exceeds {self.max_sequence_length}"
            )
        layer_keys: list[torch.Tensor | None] = [None] * self.action_layers_count
        layer_values: list[torch.Tensor | None] = [None] * self.action_layers_count
        action_mask = torch.empty(
            (states.shape[0], 0), dtype=torch.bool, device=states.device
        )
        action_outputs = []
        batch_ids = torch.arange(states.shape[0], device=states.device)

        for action_index in range(max_actions):
            xfer_ids = batch["action_xfers"][:, action_index]
            active = xfer_ids.ge(0)
            source_ids = batch["action_sources"][:, action_index].clamp_min(0)
            source_slots = batch["binding_slots"][:, action_index]
            source_mask = source_slots.ge(0) & active.unsqueeze(-1)
            safe_source_slots = source_slots.clamp_min(0)
            bound_states = torch.gather(
                states,
                1,
                safe_source_slots.unsqueeze(-1).expand(-1, -1, self.width),
            )
            bound_states = self._ordered_bound_states(bound_states, source_mask)
            bound_pool = (bound_states * source_mask.unsqueeze(-1)).sum(1)
            bound_pool = bound_pool / source_mask.sum(1, keepdim=True).clamp_min(1)
            raw_context = self.action_context(
                torch.cat(
                    (
                        self.xfer_embedding(xfer_ids.clamp_min(0)),
                        source_representations[source_ids],
                        bound_pool,
                    ),
                    dim=-1,
                )
            )
            token = raw_context + self.action_position.weight[action_index]
            token = token * active.unsqueeze(-1)
            action_mask = torch.cat((action_mask, active.unsqueeze(1)), dim=1)
            for layer_index, layer in enumerate(self.causal_action_layers):
                token, key, value = layer.forward_step(
                    token,
                    layer_keys[layer_index],
                    layer_values[layer_index],
                    action_mask,
                )
                token = token * active.unsqueeze(-1)
                layer_keys[layer_index] = key
                layer_values[layer_index] = value
            token = self.action_output_norm(token) * active.unsqueeze(-1)
            action_outputs.append(token)

            states = self._update_live_nodes(
                states,
                live,
                bound_states,
                source_mask,
                raw_context,
                active,
            )

            next_states = states.clone()
            next_live = live.clone()
            next_gate_types = gate_types.clone()
            for position in range(source_slots.shape[1]):
                valid = source_mask[:, position]
                if bool(valid.any()):
                    rows = batch_ids[valid]
                    slots = source_slots[valid, position]
                    next_live[rows, slots] = False
                    next_gate_types[rows, slots] = -1
                    next_states[rows, slots] = 0

            destination_slots = batch["destination_slots"][:, action_index]
            destination_types = batch["destination_types"][:, action_index]
            destination_mask = destination_slots.ge(0) & active.unsqueeze(-1)
            for position in range(destination_slots.shape[1]):
                valid = destination_mask[:, position]
                if not bool(valid.any()):
                    continue
                rows = batch_ids[valid]
                slots = destination_slots[valid, position]
                types = destination_types[valid, position]
                pattern_position = self.pattern_position.weight[position].expand(
                    rows.numel(), -1
                )
                created = self.destination_state(
                    torch.cat(
                        (
                            raw_context[valid],
                            self.gate_embedding(types),
                            pattern_position,
                        ),
                        dim=-1,
                    )
                )
                next_states[rows, slots] = created.to(next_states.dtype)
                next_live[rows, slots] = True
                next_gate_types[rows, slots] = types
            states, live, gate_types = next_states, next_live, next_gate_types

        if not torch.equal(gate_types, batch["current_types"]):
            raise RuntimeError("action-derived live slots differ from current graph")
        if action_outputs:
            history = torch.stack(action_outputs, dim=1)
        else:
            history = states.new_empty((states.shape[0], 0, self.width))
        states = self._fuse_action_history(
            states,
            live,
            history,
            action_mask,
            batch.get("current_live_slots"),
        )
        states = self._fuse_readout_graph(states, live, gate_types, batch)
        return states, live, gate_types
