from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from dataset import RuleMetadata


class InitialGraphLayer(nn.Module):
    def __init__(self, width: int, num_relations: int = 16):
        super().__init__()
        self.forward_message = nn.Linear(width, width, bias=False)
        self.backward_message = nn.Linear(width, width, bias=False)
        self.relation = nn.Embedding(2 * num_relations, width)
        self.update = nn.Sequential(
            nn.Linear(2 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.norm = nn.LayerNorm(width)
        self.num_relations = num_relations

    def forward(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_relation: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_slots, width = states.shape
        flat = states.reshape(batch_size * num_slots, width)
        aggregation = torch.zeros_like(flat)
        degree = torch.zeros((batch_size * num_slots, 1), device=states.device)
        if edge_batch.numel():
            src_index = edge_batch * num_slots + edge_src
            dst_index = edge_batch * num_slots + edge_dst
            forward = self.forward_message(flat[src_index]) + self.relation(edge_relation)
            backward = self.backward_message(flat[dst_index]) + self.relation(
                edge_relation + self.num_relations
            )
            aggregation.index_add_(0, dst_index, forward)
            aggregation.index_add_(0, src_index, backward)
            ones = torch.ones((src_index.numel(), 1), device=states.device)
            degree.index_add_(0, dst_index, ones)
            degree.index_add_(0, src_index, ones)
        aggregation = (aggregation / degree.clamp_min(1.0)).view_as(states)
        states = self.norm(states + self.update(torch.cat((states, aggregation), dim=-1)))
        return states * live.unsqueeze(-1)


class S0ActionBindingModel(nn.Module):
    def __init__(
        self,
        rules: RuleMetadata,
        *,
        num_xfers: int,
        width: int = 192,
        retrieval_width: int = 128,
        graph_layers: int = 3,
        current_graph_layers: int = 5,
        dropout: float = 0.05,
        use_action_history: bool = True,
        use_locality_features: bool = False,
    ):
        super().__init__()
        self.width = width
        self.retrieval_width = retrieval_width
        self.use_action_history = use_action_history
        self.use_locality_features = use_locality_features
        self.num_gate_types = rules.num_gate_types
        self.num_sources = len(rules.source_gate_types)
        self.max_pattern = max(map(len, rules.source_gate_types))
        source_types = torch.full(
            (self.num_sources, self.max_pattern), -1, dtype=torch.long
        )
        source_lengths = torch.zeros(self.num_sources, dtype=torch.long)
        for source_id, gate_types in enumerate(rules.source_gate_types):
            source_types[source_id, : len(gate_types)] = torch.tensor(gate_types)
            source_lengths[source_id] = len(gate_types)
        self.register_buffer("source_types", source_types)
        self.register_buffer("source_lengths", source_lengths)
        grouped_source_ids = []
        source_first_gate_groups = []
        for gate_type in range(rules.num_gate_types):
            source_ids = [
                source_id
                for source_id, types in enumerate(rules.source_gate_types)
                if types[0] == gate_type
            ]
            if not source_ids:
                continue
            begin = len(grouped_source_ids)
            grouped_source_ids.extend(source_ids)
            source_first_gate_groups.append(
                (gate_type, begin, len(grouped_source_ids))
            )
        self.source_first_gate_groups = tuple(source_first_gate_groups)
        self.register_buffer(
            "source_first_gate_order",
            torch.tensor(grouped_source_ids, dtype=torch.long),
            persistent=False,
        )
        from incremental_graph import parse_pattern

        pattern_edge_rows = []
        max_pattern_edges = 0
        for pattern in rules.source_patterns:
            operations = parse_pattern(pattern)
            wires: dict[int, list[tuple[int, int]]] = {}
            for operation_index, operation in enumerate(operations):
                for port, qubit in enumerate(operation.qubits):
                    wires.setdefault(qubit, []).append((operation_index, port))
            edges = []
            for wire in wires.values():
                for (src_index, src_port), (dst_index, dst_port) in zip(
                    wire, wire[1:]
                ):
                    edges.append((src_index, dst_index, src_port, dst_port))
            pattern_edge_rows.append(edges)
            max_pattern_edges = max(max_pattern_edges, len(edges))
        binding_child = torch.full(
            (self.num_sources, self.max_pattern), -1, dtype=torch.long
        )
        binding_parent = torch.full_like(binding_child, -1)
        binding_parent_port = torch.full_like(binding_child, -1)
        binding_child_port = torch.full_like(binding_child, -1)
        binding_direction = torch.zeros_like(binding_child)
        pattern_edges = torch.full(
            (self.num_sources, max_pattern_edges, 4), -1, dtype=torch.long
        )
        for source_id, edges in enumerate(pattern_edge_rows):
            if edges:
                pattern_edges[source_id, : len(edges)] = torch.tensor(edges)
            mapped = {0}
            for step in range(int(source_lengths[source_id]) - 1):
                for src_index, dst_index, src_port, dst_port in edges:
                    if src_index in mapped and dst_index not in mapped:
                        binding_child[source_id, step] = dst_index
                        binding_parent[source_id, step] = src_index
                        binding_parent_port[source_id, step] = src_port
                        binding_child_port[source_id, step] = dst_port
                        binding_direction[source_id, step] = 1
                        mapped.add(dst_index)
                        break
                    if dst_index in mapped and src_index not in mapped:
                        binding_child[source_id, step] = src_index
                        binding_parent[source_id, step] = dst_index
                        binding_parent_port[source_id, step] = dst_port
                        binding_child_port[source_id, step] = src_port
                        binding_direction[source_id, step] = -1
                        mapped.add(src_index)
                        break
        self.register_buffer("binding_child", binding_child)
        self.register_buffer("binding_parent", binding_parent)
        self.register_buffer("binding_parent_port", binding_parent_port)
        self.register_buffer("binding_child_port", binding_child_port)
        self.register_buffer("binding_direction", binding_direction)
        self.register_buffer("pattern_edges", pattern_edges)

        self.gate_embedding = nn.Embedding(rules.num_gate_types, width)
        self.source_embedding = nn.Embedding(self.num_sources, width)
        self.xfer_embedding = nn.Embedding(num_xfers, width)
        self.pattern_position = nn.Embedding(self.max_pattern, width)
        self.initial_marker = nn.Parameter(torch.zeros(width))
        self.pattern_composer = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.graph_layers = nn.ModuleList(
            InitialGraphLayer(width) for _ in range(graph_layers)
        )
        self.current_graph_marker = nn.Parameter(torch.zeros(width))
        if use_locality_features:
            self.rewrite_distance_embedding = nn.Embedding(6, width)
            self.touch_age_embedding = nn.Embedding(8, width)
            self.local_streak_embedding = nn.Embedding(6, width)
            self.locality_fusion = nn.Sequential(
                nn.Linear(3 * width, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            nn.init.zeros_(self.locality_fusion[-1].weight)
            nn.init.zeros_(self.locality_fusion[-1].bias)
        self.current_graph_layers = nn.ModuleList(
            InitialGraphLayer(width) for _ in range(current_graph_layers)
        )
        self.current_fusion = nn.Sequential(
            nn.Linear(2 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.current_norm = nn.LayerNorm(width)

        self.action_context = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.node_query = nn.Linear(width, width, bias=False)
        self.binding_key = nn.Linear(width, width, bias=False)
        self.binding_value = nn.Linear(width, width, bias=False)
        self.action_delta = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.action_norm = nn.LayerNorm(width)
        self.destination_state = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

        self.retrieval_node = nn.Linear(width, retrieval_width, bias=False)
        self.retrieval_source = nn.Linear(width, retrieval_width, bias=False)
        self.source_bias = nn.Parameter(torch.zeros(self.num_sources))
        self.pointer_node = nn.Linear(width, retrieval_width, bias=False)
        self.pointer_query = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, retrieval_width),
        )

    def source_representations(self) -> torch.Tensor:
        positions = self.pattern_position.weight[: self.max_pattern]
        safe_types = self.source_types.clamp_min(0)
        tokens = self.gate_embedding(safe_types) + positions.unsqueeze(0)
        mask = self.source_types.ge(0).unsqueeze(-1)
        composed = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.source_embedding.weight + self.pattern_composer(composed)

    def current_graph_tokens(
        self, batch: dict, gate_types: torch.Tensor, live: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.gate_embedding(gate_types.clamp_min(0)) + self.current_graph_marker
        if self.use_locality_features:
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
            tokens = tokens + locality
        return tokens * live.unsqueeze(-1)

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_action_history:
            gate_types = batch["current_types"]
            live = gate_types.ge(0)
            states = self.current_graph_tokens(batch, gate_types, live)
            for layer in self.current_graph_layers:
                states = layer(
                    states,
                    live,
                    batch["current_edge_batch"],
                    batch["current_edge_src"],
                    batch["current_edge_dst"],
                    batch["current_edge_relation"],
                )
            return states, live, gate_types

        initial_types = batch["initial_types"]
        batch_size, num_slots = initial_types.shape
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

        source_representations = self.source_representations()
        max_actions = batch["action_xfers"].shape[1]
        for action_index in range(max_actions):
            xfer_ids = batch["action_xfers"][:, action_index]
            active = xfer_ids.ge(0)
            if not bool(active.any()):
                continue
            source_ids = batch["action_sources"][:, action_index].clamp_min(0)
            source_slots = batch["binding_slots"][:, action_index]
            source_mask = source_slots.ge(0) & active.unsqueeze(-1)
            safe_source_slots = source_slots.clamp_min(0)
            bound_states = torch.gather(
                states,
                1,
                safe_source_slots.unsqueeze(-1).expand(-1, -1, self.width),
            )
            bound_pool = (bound_states * source_mask.unsqueeze(-1)).sum(1)
            bound_pool = bound_pool / source_mask.sum(1, keepdim=True).clamp_min(1)
            context = self.action_context(
                torch.cat(
                    (
                        self.xfer_embedding(xfer_ids.clamp_min(0)),
                        source_representations[source_ids],
                        bound_pool,
                    ),
                    dim=-1,
                )
            )

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
            expanded_context = context.unsqueeze(1).expand(-1, num_slots, -1)
            delta = self.action_delta(
                torch.cat((states, local_context, expanded_context), dim=-1)
            )
            update_mask = live & active.unsqueeze(-1)
            updated = self.action_norm(states + affinity * delta)
            states = torch.where(update_mask.unsqueeze(-1), updated, states)

            next_states = states.clone()
            next_live = live.clone()
            next_gate_types = gate_types.clone()
            for position in range(source_slots.shape[1]):
                valid = source_mask[:, position]
                if bool(valid.any()):
                    batch_ids = valid.nonzero(as_tuple=False).squeeze(1)
                    slots = source_slots[batch_ids, position]
                    next_live[batch_ids, slots] = False
                    next_gate_types[batch_ids, slots] = -1
                    next_states[batch_ids, slots] = 0

            destination_slots = batch["destination_slots"][:, action_index]
            destination_types = batch["destination_types"][:, action_index]
            destination_mask = destination_slots.ge(0) & active.unsqueeze(-1)
            for position in range(destination_slots.shape[1]):
                valid = destination_mask[:, position]
                if not bool(valid.any()):
                    continue
                batch_ids = valid.nonzero(as_tuple=False).squeeze(1)
                slots = destination_slots[batch_ids, position]
                types = destination_types[batch_ids, position]
                position_embedding = self.pattern_position.weight[position].expand(
                    batch_ids.numel(), -1
                )
                created = self.destination_state(
                    torch.cat(
                        (
                            context[batch_ids],
                            self.gate_embedding(types),
                            position_embedding,
                        ),
                        dim=-1,
                    )
                )
                next_states[batch_ids, slots] = created.to(next_states.dtype)
                next_live[batch_ids, slots] = True
                next_gate_types[batch_ids, slots] = types
            states, live, gate_types = next_states, next_live, next_gate_types
        if not torch.equal(gate_types, batch["current_types"]):
            raise RuntimeError("action-derived live slots differ from incremental graph")
        current_states = self.current_graph_tokens(batch, gate_types, live)
        for layer in self.current_graph_layers:
            current_states = layer(
                current_states,
                live,
                batch["current_edge_batch"],
                batch["current_edge_src"],
                batch["current_edge_dst"],
                batch["current_edge_relation"],
            )
        states = self.current_norm(
            states + self.current_fusion(torch.cat((states, current_states), dim=-1))
        ) * live.unsqueeze(-1)
        return states, live, gate_types

    def match_logits(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        gate_types: torch.Tensor,
        source_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_vectors = self.match_node_vectors(states)
        if source_vectors is None:
            source_vectors = self.retrieval_source(self.source_representations())
        return self.match_logits_from_node_vectors(
            node_vectors,
            live,
            gate_types,
            source_vectors,
        )

    def match_node_vectors(self, states: torch.Tensor) -> torch.Tensor:
        return self.retrieval_node(states)

    def match_logits_from_node_vectors(
        self,
        node_vectors: torch.Tensor,
        live: torch.Tensor,
        gate_types: torch.Tensor,
        source_vectors: torch.Tensor,
        *,
        source_begin: int = 0,
        source_end: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_end is None:
            source_end = self.num_sources
        selected_sources = source_vectors[source_begin:source_end]
        logits = torch.einsum("bsd,vd->bsv", node_vectors, selected_sources)
        logits = (
            logits / math.sqrt(self.retrieval_width)
            + self.source_bias[source_begin:source_end]
        )
        first_types = self.source_types[source_begin:source_end, 0]
        eligible = live.unsqueeze(-1) & gate_types.unsqueeze(-1).eq(first_types)
        return logits.masked_fill(~eligible, -1e4), eligible

    def match_logits_for_sources(
        self,
        node_vectors: torch.Tensor,
        source_vectors: torch.Tensor,
        source_ids: torch.Tensor,
    ) -> torch.Tensor:
        selected_sources = source_vectors.index_select(0, source_ids)
        logits = torch.einsum("bsd,vd->bsv", node_vectors, selected_sources)
        return (
            logits / math.sqrt(self.retrieval_width)
            + self.source_bias.index_select(0, source_ids)
        )

    @torch.no_grad()
    def structural_decode(
        self,
        batch: dict,
        gate_types: torch.Tensor,
        live: torch.Tensor,
        batch_ids: torch.Tensor,
        source_ids: torch.Tensor,
        anchor_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized port-following decoder for complete ordered bindings."""
        batch_size, num_slots = gate_types.shape
        max_ports = 4
        shape = (batch_size, num_slots, max_ports)
        out_node = torch.full(shape, -1, dtype=torch.long, device=gate_types.device)
        out_port = torch.full_like(out_node, -1)
        in_node = torch.full_like(out_node, -1)
        in_port = torch.full_like(out_node, -1)
        edge_batch = batch["current_edge_batch"]
        edge_src = batch["current_edge_src"]
        edge_dst = batch["current_edge_dst"]
        relations = batch["current_edge_relation"]
        src_ports = torch.div(relations, 4, rounding_mode="floor")
        dst_ports = relations.remainder(4)
        out_node[edge_batch, edge_src, src_ports] = edge_dst
        out_port[edge_batch, edge_src, src_ports] = dst_ports
        in_node[edge_batch, edge_dst, dst_ports] = edge_src
        in_port[edge_batch, edge_dst, dst_ports] = src_ports

        num_rows = batch_ids.numel()
        bindings = torch.full(
            (num_rows, self.max_pattern),
            -1,
            dtype=torch.long,
            device=gate_types.device,
        )
        bindings[:, 0] = anchor_slots
        valid = (
            live[batch_ids, anchor_slots]
            & gate_types[batch_ids, anchor_slots].eq(self.source_types[source_ids, 0])
        )
        row_ids = torch.arange(num_rows, device=gate_types.device)
        for step in range(self.max_pattern - 1):
            active = self.source_lengths[source_ids].gt(step + 1)
            child_positions = self.binding_child[source_ids, step]
            parent_positions = self.binding_parent[source_ids, step]
            valid &= ~active | (parent_positions.ge(0) & child_positions.ge(0))
            safe_parent_positions = parent_positions.clamp_min(0)
            parent_slots = bindings[row_ids, safe_parent_positions].clamp_min(0)
            parent_ports = self.binding_parent_port[source_ids, step].clamp_min(0)
            child_ports = self.binding_child_port[source_ids, step]
            forward = self.binding_direction[source_ids, step].gt(0)
            forward_node = out_node[batch_ids, parent_slots, parent_ports]
            forward_port = out_port[batch_ids, parent_slots, parent_ports]
            backward_node = in_node[batch_ids, parent_slots, parent_ports]
            backward_port = in_port[batch_ids, parent_slots, parent_ports]
            candidate = torch.where(forward, forward_node, backward_node)
            candidate_port = torch.where(forward, forward_port, backward_port)
            position_valid = candidate.ge(0) & candidate_port.eq(child_ports)
            safe_candidate = candidate.clamp_min(0)
            position_valid &= live[batch_ids, safe_candidate]
            expected_types = self.source_types[
                source_ids, child_positions.clamp_min(0)
            ]
            position_valid &= gate_types[batch_ids, safe_candidate].eq(
                expected_types
            )
            position_valid &= ~bindings.eq(candidate.unsqueeze(1)).any(1)
            valid &= ~active | position_valid
            active_rows = active.nonzero(as_tuple=False).squeeze(1)
            if active_rows.numel():
                bindings[
                    active_rows, child_positions[active_rows]
                ] = candidate[active_rows]

        for edge_index in range(self.pattern_edges.shape[1]):
            edges = self.pattern_edges[source_ids, edge_index]
            active = edges[:, 0].ge(0)
            src_positions = edges[:, 0].clamp_min(0)
            dst_positions = edges[:, 1].clamp_min(0)
            src_slots = bindings[row_ids, src_positions].clamp_min(0)
            dst_slots = bindings[row_ids, dst_positions]
            src_port = edges[:, 2].clamp_min(0)
            dst_port = edges[:, 3]
            edge_valid = out_node[batch_ids, src_slots, src_port].eq(dst_slots)
            edge_valid &= out_port[batch_ids, src_slots, src_port].eq(dst_port)
            valid &= ~active | edge_valid
        return bindings, valid

    def classification_loss(
        self,
        logits: torch.Tensor,
        eligible: torch.Tensor,
        positives: list[list],
        *,
        batch: dict | None = None,
        live: torch.Tensor | None = None,
        gate_types: torch.Tensor | None = None,
        structural_hard_negatives: bool = False,
        locality_positive_weight: float = 0.0,
        locality_negative_weight: float = 0.0,
        action_positive_weight: float = 0.0,
        topn_boundary_weight: float = 0.0,
        topn_boundary_margin: float = 0.0,
    ) -> torch.Tensor:
        sample_rows = []
        pool_batch = []
        pool_sources = []
        pool_anchors = []
        pool_logits = []
        pool_weights = []
        pool_offset = 0
        for batch_index, rows in enumerate(positives):
            target = torch.zeros_like(eligible[batch_index])
            anchors = torch.tensor(
                [binding[0] for _, binding in rows], device=logits.device
            )
            sources = torch.tensor([source for source, _ in rows], device=logits.device)
            target[anchors, sources] = True
            positive_logits = logits[batch_index, anchors, sources]
            positive_weights = torch.ones_like(positive_logits)
            if locality_positive_weight > 0:
                if batch is None or "positive_near" not in batch:
                    raise ValueError("locality weighting needs positive_near labels")
                near = torch.tensor(
                    batch["positive_near"][batch_index],
                    dtype=torch.bool,
                    device=logits.device,
                )
                positive_weights = positive_weights + locality_positive_weight * near
            if action_positive_weight > 0:
                if batch is None or "target_actions" not in batch:
                    raise ValueError("action-positive weighting needs target actions")
                target_action = batch["target_actions"][batch_index]
                if target_action is not None:
                    target_source = int(target_action["source_id"])
                    target_binding = tuple(
                        map(int, target_action["binding_slots"])
                    )
                    chosen_rows = [
                        int(source) == target_source
                        and tuple(map(int, binding)) == target_binding
                        for source, binding in rows
                    ]
                    if not any(chosen_rows):
                        raise ValueError(
                            "target action is absent from exact positive matches"
                        )
                    chosen = torch.tensor(
                        chosen_rows,
                        dtype=positive_weights.dtype,
                        device=positive_weights.device,
                    )
                    positive_weights = (
                        positive_weights + action_positive_weight * chosen
                    )
            negative_mask = eligible[batch_index] & ~target
            negative_logits = logits[batch_index][negative_mask]
            if positive_logits.numel() == 0 or negative_logits.numel() == 0:
                continue
            pool_count = min(
                negative_logits.numel(), max(512, 4 * positive_logits.numel())
            )
            negative_positions = negative_mask.flatten().nonzero(
                as_tuple=False
            ).squeeze(1)
            pool_order = negative_logits.topk(pool_count).indices
            flat_positions = negative_positions[pool_order]
            anchors_pool = torch.div(
                flat_positions, self.num_sources, rounding_mode="floor"
            )
            sources_pool = flat_positions.remainder(self.num_sources)
            begin = pool_offset
            pool_logits.append(logits[batch_index].flatten()[flat_positions])
            negative_weights = torch.ones(
                pool_count, dtype=logits.dtype, device=logits.device
            )
            if locality_negative_weight > 0:
                if batch is None or "current_rewrite_distance" not in batch:
                    raise ValueError("local negative weighting needs distances")
                near_negative = batch["current_rewrite_distance"][
                    batch_index, anchors_pool
                ].le(2)
                negative_weights = negative_weights + (
                    locality_negative_weight * near_negative
                )
            pool_weights.append(negative_weights)
            pool_batch.append(
                torch.full(
                    (pool_count,), batch_index, dtype=torch.long, device=logits.device
                )
            )
            pool_anchors.append(anchors_pool)
            pool_sources.append(sources_pool)
            pool_offset += pool_count
            end = pool_offset
            sample_rows.append((positive_logits, positive_weights, begin, end))

        if not sample_rows:
            return logits.sum() * 0
        structural_valid = None
        if structural_hard_negatives:
            if batch is None or live is None or gate_types is None:
                raise ValueError("structural hard negatives need the current graph")
            _, structural_valid = self.structural_decode(
                batch,
                gate_types,
                live,
                torch.cat(pool_batch),
                torch.cat(pool_sources),
                torch.cat(pool_anchors),
            )

        losses = []
        stacked_pool_logits = torch.cat(pool_logits)
        stacked_pool_weights = torch.cat(pool_weights)
        for positive_logits, positive_weights, begin, end in sample_rows:
            candidates = stacked_pool_logits[begin:end]
            candidate_weights = stacked_pool_weights[begin:end]
            if structural_valid is not None:
                validity = structural_valid[begin:end]
                candidates = torch.cat((candidates[validity], candidates[~validity]))
                candidate_weights = torch.cat(
                    (candidate_weights[validity], candidate_weights[~validity])
                )
            hard_count = min(
                candidates.numel(), max(128, 2 * positive_logits.numel())
            )
            hard_negatives = candidates[:hard_count]
            hard_negative_weights = candidate_weights[:hard_count]
            pair_count = min(positive_logits.numel(), hard_negatives.numel())
            permutation = torch.randperm(
                positive_logits.numel(), device=logits.device
            )[:pair_count]
            ranking = F.softplus(
                hard_negatives[:pair_count] - positive_logits[permutation]
            )
            ranking_weights = (
                positive_weights[permutation]
                * hard_negative_weights[:pair_count]
            )
            ranking = (ranking * ranking_weights).sum() / ranking_weights.sum()
            positive_loss = F.softplus(-positive_logits)
            positive_loss = (
                positive_loss * positive_weights
            ).sum() / positive_weights.sum()
            sample_loss = (
                positive_loss
                + (
                    F.softplus(hard_negatives) * hard_negative_weights
                ).sum()
                / hard_negative_weights.sum()
                + ranking
            )
            if topn_boundary_weight > 0:
                # Recall@N is controlled by the weakest true rows and the
                # strongest false rows around the output boundary.  The old
                # random pairing rarely trained that boundary when a state had
                # hundreds of true matches.  Sort both sides so every weak
                # positive receives a direct signal from a hard negative.  If
                # structural filtering is enabled, structurally valid false
                # rows stay ahead of invalid rows and are therefore preferred.
                boundary_count = min(positive_logits.numel(), candidates.numel())
                positive_order = positive_logits.argsort(descending=False)
                boundary_positive = positive_logits[positive_order[:boundary_count]]
                boundary_positive_weights = positive_weights[
                    positive_order[:boundary_count]
                ]
                boundary_negative = candidates[:boundary_count]
                boundary_negative_weights = candidate_weights[:boundary_count]
                boundary_ranking = F.softplus(
                    boundary_negative
                    - boundary_positive
                    + topn_boundary_margin
                )
                boundary_ranking = (
                    boundary_ranking
                    * boundary_positive_weights
                    * boundary_negative_weights
                ).sum() / (
                    boundary_positive_weights * boundary_negative_weights
                ).sum()
                sample_loss = sample_loss + topn_boundary_weight * boundary_ranking
            losses.append(sample_loss)
        return torch.stack(losses).mean()

    def _pointer_scores(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        gate_types: torch.Tensor,
        batch_ids: torch.Tensor,
        source_ids: torch.Tensor,
        anchor_slots: torch.Tensor,
        previous_states: torch.Tensor,
        position: int,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        anchor_states = states[batch_ids, anchor_slots]
        source_states = self.source_representations()[source_ids]
        query = self.pointer_query(
            torch.cat((anchor_states, source_states, previous_states), dim=-1)
        )
        keys = self.pointer_node(states[batch_ids])
        scores = torch.einsum("pd,psd->ps", query, keys) / math.sqrt(
            self.retrieval_width
        )
        expected_types = self.source_types[source_ids, position]
        mask = live[batch_ids] & gate_types[batch_ids].eq(expected_types.unsqueeze(1))
        for old_position in range(selected.shape[1]):
            mask.scatter_(1, selected[:, old_position : old_position + 1], False)
        return scores.masked_fill(~mask, -1e4)

    def binding_loss(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        gate_types: torch.Tensor,
        positives: list[list],
        max_per_sample: int = 256,
    ) -> torch.Tensor:
        batch_rows = []
        source_rows = []
        binding_rows = []
        for batch_index, rows in enumerate(positives):
            if len(rows) > max_per_sample:
                chosen = torch.randperm(len(rows))[:max_per_sample].tolist()
                rows = [rows[index] for index in chosen]
            for source_id, binding in rows:
                if len(binding) > 1:
                    batch_rows.append(batch_index)
                    source_rows.append(source_id)
                    binding_rows.append(binding)
        if not binding_rows:
            return states.sum() * 0
        device = states.device
        batch_ids = torch.tensor(batch_rows, device=device)
        source_ids = torch.tensor(source_rows, device=device)
        bindings = torch.full(
            (len(binding_rows), self.max_pattern), -1, dtype=torch.long, device=device
        )
        for row_index, binding in enumerate(binding_rows):
            bindings[row_index, : len(binding)] = torch.tensor(binding, device=device)
        anchors = bindings[:, 0]
        previous_sum = states[batch_ids, anchors]
        selected = anchors.unsqueeze(1)
        losses = []
        for position in range(1, self.max_pattern):
            active = self.source_lengths[source_ids].gt(position)
            if not bool(active.any()):
                continue
            ids = active.nonzero(as_tuple=False).squeeze(1)
            previous = previous_sum[ids] / selected.shape[1]
            scores = self._pointer_scores(
                states,
                live,
                gate_types,
                batch_ids[ids],
                source_ids[ids],
                anchors[ids],
                previous,
                position,
                selected[ids],
            )
            targets = bindings[ids, position]
            losses.append(F.cross_entropy(scores, targets))
            teacher_states = states[batch_ids[ids], targets]
            next_previous_sum = previous_sum.clone()
            next_previous_sum[ids] = next_previous_sum[ids] + teacher_states
            previous_sum = next_previous_sum
            selected = torch.cat((selected, bindings[:, position : position + 1]), dim=1)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def decode_bindings(
        self,
        states: torch.Tensor,
        live: torch.Tensor,
        gate_types: torch.Tensor,
        batch_ids: torch.Tensor,
        source_ids: torch.Tensor,
        anchor_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_rows = batch_ids.numel()
        bindings = torch.full(
            (num_rows, self.max_pattern), -1, dtype=torch.long, device=states.device
        )
        bindings[:, 0] = anchor_slots
        previous_sum = states[batch_ids, anchor_slots]
        confidence = torch.zeros(num_rows, device=states.device)
        selected = anchor_slots.unsqueeze(1)
        for position in range(1, self.max_pattern):
            active = self.source_lengths[source_ids].gt(position)
            if not bool(active.any()):
                continue
            ids = active.nonzero(as_tuple=False).squeeze(1)
            scores = self._pointer_scores(
                states,
                live,
                gate_types,
                batch_ids[ids],
                source_ids[ids],
                anchor_slots[ids],
                previous_sum[ids] / selected.shape[1],
                position,
                selected[ids],
            )
            log_probabilities = scores.log_softmax(-1)
            probability, prediction = log_probabilities.max(-1)
            bindings[ids, position] = prediction
            confidence[ids] += probability
            next_previous_sum = previous_sum.clone()
            next_previous_sum[ids] += states[batch_ids[ids], prediction]
            previous_sum = next_previous_sum
            selected = torch.cat((selected, bindings[:, position : position + 1]), dim=1)
        return bindings, confidence
