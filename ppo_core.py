from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PPOObjective:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


@dataclass(frozen=True)
class HierarchicalPolicy:
    log_probs: torch.Tensor
    mask: torch.Tensor
    node_log_probs: torch.Tensor
    conditional_candidate_log_probs: torch.Tensor


class PagedPPOActorCritic(nn.Module):
    """Small PPO heads over frozen paged-model state and action features."""

    def __init__(self, width: int, hidden_size: int | None = None) -> None:
        super().__init__()
        self.match_set_aware = False
        hidden_size = width if hidden_size is None else hidden_size
        self.width = width
        self.hidden_size = hidden_size
        self.policy_feature_dim = 4 * width + 2
        self.state_feature_dim = 2 * width + 1
        self.policy_norm = nn.LayerNorm(self.policy_feature_dim)
        self.policy = nn.Sequential(
            nn.Linear(self.policy_feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_norm = nn.LayerNorm(self.state_feature_dim)
        self.value = nn.Sequential(
            nn.Linear(self.state_feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        # Start from the calibrated matcher policy and a neutral critic.
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    def policy_logits(
        self,
        candidate_features: torch.Tensor,
        matcher_logits: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        prefix_states: torch.Tensor | None = None,
        state_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.policy(self.policy_norm(candidate_features)).squeeze(-1)
        return matcher_logits + residual

    def state_values(
        self,
        state_features: torch.Tensor,
        candidate_features: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        prefix_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.value(self.value_norm(state_features)).squeeze(-1)

    def candidate_legality_logits(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        prefix_states: torch.Tensor | None = None,
        state_features: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        return None

    def actor_parameters(self):
        yield from self.policy_norm.parameters()
        yield from self.policy.parameters()

    def critic_parameters(self):
        yield from self.value_norm.parameters()
        yield from self.value.parameters()


class CandidateSetBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(
        self, candidates: torch.Tensor, candidate_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.attention_norm(candidates)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~candidate_mask,
            need_weights=False,
        )
        candidates = candidates + attended
        candidates = candidates + self.ffn(self.ffn_norm(candidates))
        return candidates * candidate_mask.unsqueeze(-1)


class MatchSetPPOActorCritic(nn.Module):
    """PPO policy/value heads conditioned on prefix and the full match set."""

    def __init__(
        self,
        width: int,
        hidden_size: int | None = None,
        *,
        set_layers: int = 2,
        set_heads: int = 4,
    ) -> None:
        super().__init__()
        self.match_set_aware = True
        hidden_size = width if hidden_size is None else hidden_size
        if width % set_heads:
            raise ValueError("model width must be divisible by set attention heads")
        self.width = width
        self.hidden_size = hidden_size
        self.set_layers = set_layers
        self.set_heads = set_heads
        self.policy_feature_dim = 4 * width + 2
        self.state_feature_dim = 2 * width + 1

        self.actor_candidate_norm = nn.LayerNorm(self.policy_feature_dim)
        self.actor_candidate_projection = nn.Linear(
            self.policy_feature_dim, width
        )
        self.actor_prefix_projection = nn.Linear(width, width, bias=False)
        self.actor_state_projection = nn.Linear(
            self.state_feature_dim, width, bias=False
        )
        self.actor_set_blocks = nn.ModuleList(
            CandidateSetBlock(width, set_heads) for _ in range(set_layers)
        )
        self.actor_output_norm = nn.LayerNorm(width)
        self.policy = nn.Sequential(
            nn.Linear(width, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.legality = nn.Sequential(
            nn.Linear(width, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.legality_policy_weight = nn.Parameter(torch.zeros(()))

        self.critic_candidate_norm = nn.LayerNorm(self.policy_feature_dim)
        self.critic_candidate_projection = nn.Linear(
            self.policy_feature_dim, width
        )
        self.critic_prefix_projection = nn.Linear(width, width, bias=False)
        self.critic_state_projection = nn.Linear(
            self.state_feature_dim, width, bias=False
        )
        self.critic_set_blocks = nn.ModuleList(
            CandidateSetBlock(width, set_heads) for _ in range(set_layers)
        )
        self.critic_query = nn.Linear(2 * width, width, bias=False)
        self.critic_pool = nn.MultiheadAttention(
            width, set_heads, batch_first=True
        )
        self.value_norm = nn.LayerNorm(3 * width)
        self.value = nn.Sequential(
            nn.Linear(3 * width, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)
        nn.init.zeros_(self.legality[-1].weight)
        nn.init.zeros_(self.legality[-1].bias)
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    @staticmethod
    def _require_context(
        candidate_mask: torch.Tensor | None,
        prefix_states: torch.Tensor | None,
        state_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_mask is None or prefix_states is None or state_features is None:
            raise ValueError("match-set actor/critic requires mask, prefix, and state")
        return candidate_mask, prefix_states, state_features

    def _actor_candidates(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        prefix_states: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        candidates = self.actor_candidate_projection(
            self.actor_candidate_norm(candidate_features)
        )
        context = self.actor_prefix_projection(prefix_states)
        context = context + self.actor_state_projection(state_features)
        candidates = (
            candidates + context.unsqueeze(1)
        ) * candidate_mask.unsqueeze(-1)
        for block in self.actor_set_blocks:
            candidates = block(candidates, candidate_mask)
        return self.actor_output_norm(candidates)

    def policy_logits(
        self,
        candidate_features: torch.Tensor,
        matcher_logits: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        prefix_states: torch.Tensor | None = None,
        state_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        candidate_mask, prefix_states, state_features = self._require_context(
            candidate_mask, prefix_states, state_features
        )
        candidates = self._actor_candidates(
            candidate_features, candidate_mask, prefix_states, state_features
        )
        residual = self.policy(candidates).squeeze(-1)
        legality_logits = self.legality(candidates).squeeze(-1)
        legality_prior = self.legality_policy_weight * F.logsigmoid(
            legality_logits
        )
        return matcher_logits + residual + legality_prior

    def candidate_legality_logits(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        prefix_states: torch.Tensor | None = None,
        state_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        candidate_mask, prefix_states, state_features = self._require_context(
            candidate_mask, prefix_states, state_features
        )
        candidates = self._actor_candidates(
            candidate_features, candidate_mask, prefix_states, state_features
        )
        return self.legality(candidates).squeeze(-1)

    def state_values(
        self,
        state_features: torch.Tensor,
        candidate_features: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        prefix_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            candidate_features is None
            or candidate_mask is None
            or prefix_states is None
        ):
            raise ValueError("match-set critic requires candidates, mask, and prefix")
        candidates = self.critic_candidate_projection(
            self.critic_candidate_norm(candidate_features)
        )
        prefix = self.critic_prefix_projection(prefix_states)
        state = self.critic_state_projection(state_features)
        candidates = (
            candidates + (prefix + state).unsqueeze(1)
        ) * candidate_mask.unsqueeze(-1)
        for block in self.critic_set_blocks:
            candidates = block(candidates, candidate_mask)
        query = self.critic_query(torch.cat((prefix, state), dim=-1)).unsqueeze(1)
        pooled, _ = self.critic_pool(
            query,
            candidates,
            candidates,
            key_padding_mask=~candidate_mask,
            need_weights=False,
        )
        value_input = torch.cat((prefix, state, pooled[:, 0]), dim=-1)
        return self.value(self.value_norm(value_input)).squeeze(-1)

    def actor_parameters(self):
        modules = (
            self.actor_candidate_norm,
            self.actor_candidate_projection,
            self.actor_prefix_projection,
            self.actor_state_projection,
            self.actor_set_blocks,
            self.actor_output_norm,
            self.policy,
            self.legality,
        )
        for module in modules:
            yield from module.parameters()
        yield self.legality_policy_weight

    def critic_parameters(self):
        modules = (
            self.critic_candidate_norm,
            self.critic_candidate_projection,
            self.critic_prefix_projection,
            self.critic_state_projection,
            self.critic_set_blocks,
            self.critic_query,
            self.critic_pool,
            self.value_norm,
            self.value,
        )
        for module in modules:
            yield from module.parameters()


def hierarchical_policy_log_probs(
    node_logits: torch.Tensor,
    node_mask: torch.Tensor,
    candidate_logits: torch.Tensor,
    candidate_nodes: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    stop_logits: torch.Tensor | None = None,
) -> HierarchicalPolicy:
    """Normalize stop, node, and node-conditional candidate decisions.

    Candidates may cover a truncated node set. Nodes without a retained
    candidate are removed from the node distribution, making the returned
    policy exact over the supplied support.
    """
    # Policy normalization stays in fp32 under autocast, matching the existing
    # PPO distribution path and avoiding precision-sensitive segmented sums.
    node_logits = node_logits.float()
    candidate_logits = candidate_logits.float()
    if stop_logits is not None:
        stop_logits = stop_logits.float()
    if node_logits.ndim != 2 or node_mask.shape != node_logits.shape:
        raise ValueError("node logits and mask must be aligned matrices")
    if (
        candidate_logits.ndim != 2
        or candidate_nodes.shape != candidate_logits.shape
        or candidate_mask.shape != candidate_logits.shape
        or candidate_logits.shape[0] != node_logits.shape[0]
    ):
        raise ValueError("candidate tensors must be aligned padded matrices")
    if candidate_nodes.numel():
        valid_nodes = candidate_nodes[candidate_mask]
        if valid_nodes.numel() and (
            bool(valid_nodes.lt(0).any())
            or bool(valid_nodes.ge(node_logits.shape[1]).any())
        ):
            raise ValueError("candidate node is outside the node tensor")

    batch_size, num_nodes = node_logits.shape
    node_has_candidate = torch.zeros_like(node_mask)
    if candidate_mask.any():
        batch_ids = (
            torch.arange(batch_size, device=node_logits.device)
            .unsqueeze(1)
            .expand_as(candidate_nodes)
        )
        node_has_candidate[batch_ids[candidate_mask], candidate_nodes[candidate_mask]] = True
    effective_node_mask = node_mask & node_has_candidate
    has_candidate = candidate_mask.any(1)
    if stop_logits is None and not bool(has_candidate.all()):
        raise ValueError("a policy without stop must retain a candidate in every row")
    if bool(has_candidate.any()) and not bool(
        effective_node_mask[has_candidate].any(1).all()
    ):
        raise ValueError("candidate nodes must refer to enabled policy nodes")

    masked_node_logits = node_logits.masked_fill(~effective_node_mask, -torch.inf)
    node_log_probs = torch.full_like(node_logits, -torch.inf)
    node_log_probs[has_candidate] = F.log_softmax(
        masked_node_logits[has_candidate], dim=-1
    )

    conditional = torch.full_like(candidate_logits, -torch.inf)
    if bool(candidate_mask.any()):
        batch_ids = (
            torch.arange(batch_size, device=node_logits.device)
            .unsqueeze(1)
            .expand_as(candidate_nodes)
        )
        flat_batch = batch_ids[candidate_mask]
        flat_nodes = candidate_nodes[candidate_mask]
        segment_ids = flat_batch * num_nodes + flat_nodes
        flat_log_probs = segmented_log_softmax(
            candidate_logits[candidate_mask], segment_ids, batch_size * num_nodes
        )
        conditional[candidate_mask] = flat_log_probs
        candidate_node_log_probs = node_log_probs[flat_batch, flat_nodes]
    else:
        candidate_node_log_probs = candidate_logits.new_empty(0)

    if stop_logits is None:
        candidate_log_probs = torch.full_like(candidate_logits, -torch.inf)
        candidate_log_probs[candidate_mask] = (
            candidate_node_log_probs + conditional[candidate_mask]
        )
        return HierarchicalPolicy(
            candidate_log_probs,
            candidate_mask,
            node_log_probs,
            conditional,
        )

    if stop_logits.shape != (batch_size,):
        raise ValueError("stop logits must contain one scalar per state")
    stop_log_probs = F.logsigmoid(stop_logits)
    stop_log_probs = torch.where(
        has_candidate, stop_log_probs, torch.zeros_like(stop_log_probs)
    )
    continue_log_probs = F.logsigmoid(-stop_logits)
    candidate_log_probs = torch.full_like(candidate_logits, -torch.inf)
    candidate_log_probs[candidate_mask] = (
        continue_log_probs[flat_batch]
        + candidate_node_log_probs
        + conditional[candidate_mask]
    )
    all_log_probs = torch.cat((stop_log_probs.unsqueeze(1), candidate_log_probs), dim=1)
    all_mask = torch.cat(
        (
            torch.ones((batch_size, 1), dtype=torch.bool, device=node_mask.device),
            candidate_mask,
        ),
        dim=1,
    )
    return HierarchicalPolicy(all_log_probs, all_mask, node_log_probs, conditional)


class HierarchicalPPOActorCritic(nn.Module):
    """Sequence-conditioned stop/node/pattern policy over frozen model features."""

    def __init__(self, width: int, hidden_size: int | None = None) -> None:
        super().__init__()
        self.match_set_aware = False
        self.hierarchical = True
        hidden_size = width if hidden_size is None else hidden_size
        self.width = width
        self.hidden_size = hidden_size
        self.policy_feature_dim = 4 * width + 2
        self.state_feature_dim = 2 * width + 1

        self.node_norm = nn.LayerNorm(width)
        self.node_projection = nn.Linear(width, hidden_size)
        self.node_prefix_projection = nn.Linear(width, hidden_size, bias=False)
        self.node_state_projection = nn.Linear(
            self.state_feature_dim, hidden_size, bias=False
        )
        self.node_output = nn.Linear(hidden_size, 1)

        self.pattern_norm = nn.LayerNorm(self.policy_feature_dim)
        self.pattern = nn.Sequential(
            nn.Linear(self.policy_feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.stop_norm = nn.LayerNorm(width + self.state_feature_dim)
        self.stop = nn.Sequential(
            nn.Linear(width + self.state_feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_norm = nn.LayerNorm(width + self.state_feature_dim)
        self.value = nn.Sequential(
            nn.Linear(width + self.state_feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        # Preserve the matcher ranking until the hierarchy is distilled.
        nn.init.zeros_(self.pattern[-1].weight)
        nn.init.zeros_(self.pattern[-1].bias)
        nn.init.zeros_(self.stop[-1].weight)
        nn.init.constant_(self.stop[-1].bias, -4.0)
        nn.init.zeros_(self.value[-1].weight)
        nn.init.zeros_(self.value[-1].bias)

    @staticmethod
    def _context(
        prefix_states: torch.Tensor | None,
        state_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prefix_states is None or state_features is None:
            raise ValueError("hierarchical actor requires prefix and state features")
        return prefix_states, state_features

    def node_policy_logits(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        prefix_states: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        context = self.node_prefix_projection(prefix_states)
        context = context + self.node_state_projection(state_features)
        hidden = self.node_projection(self.node_norm(node_features))
        hidden = F.gelu(hidden + context.unsqueeze(1))
        return self.node_output(hidden).squeeze(-1).masked_fill(
            ~node_mask, -torch.inf
        )

    def candidate_policy_logits(
        self,
        candidate_features: torch.Tensor,
        matcher_logits: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.pattern(self.pattern_norm(candidate_features)).squeeze(-1)
        return matcher_logits + residual

    def stop_policy_logits(
        self, prefix_states: torch.Tensor, state_features: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat((prefix_states, state_features), dim=-1)
        return self.stop(self.stop_norm(features)).squeeze(-1)

    def policy(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        matcher_logits: torch.Tensor,
        candidate_nodes: torch.Tensor,
        candidate_mask: torch.Tensor,
        prefix_states: torch.Tensor,
        state_features: torch.Tensor,
        *,
        include_stop: bool = True,
    ) -> HierarchicalPolicy:
        node_logits = self.node_policy_logits(
            node_features, node_mask, prefix_states, state_features
        )
        candidate_logits = self.candidate_policy_logits(
            candidate_features, matcher_logits
        )
        stop_logits = (
            self.stop_policy_logits(prefix_states, state_features)
            if include_stop
            else None
        )
        return hierarchical_policy_log_probs(
            node_logits,
            node_mask,
            candidate_logits,
            candidate_nodes,
            candidate_mask,
            stop_logits=stop_logits,
        )

    def state_values(
        self,
        state_features: torch.Tensor,
        candidate_features: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        prefix_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prefix_states, state_features = self._context(prefix_states, state_features)
        features = torch.cat((prefix_states, state_features), dim=-1)
        return self.value(self.value_norm(features)).squeeze(-1)

    def actor_parameters(self):
        modules = (
            self.node_norm,
            self.node_projection,
            self.node_prefix_projection,
            self.node_state_projection,
            self.node_output,
            self.pattern_norm,
            self.pattern,
            self.stop_norm,
            self.stop,
        )
        for module in modules:
            yield from module.parameters()

    def critic_parameters(self):
        yield from self.value_norm.parameters()
        yield from self.value.parameters()


def build_actor_critic(
    architecture: str,
    width: int,
    *,
    hidden_size: int | None = None,
    set_layers: int = 2,
    set_heads: int = 4,
) -> nn.Module:
    if architecture == "legacy":
        return PagedPPOActorCritic(width, hidden_size=hidden_size)
    if architecture == "match_set":
        return MatchSetPPOActorCritic(
            width,
            hidden_size=hidden_size,
            set_layers=set_layers,
            set_heads=set_heads,
        )
    if architecture == "hierarchical":
        return HierarchicalPPOActorCritic(width, hidden_size=hidden_size)
    raise ValueError(f"unknown actor/critic architecture: {architecture}")


def build_policy_features(
    candidate_features: torch.Tensor,
    matcher_probabilities: torch.Tensor,
    gate_deltas: torch.Tensor,
    *,
    initial_gate_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = matcher_probabilities.float().clamp(1e-6, 1 - 1e-6)
    confidence_logits = torch.logit(probabilities)
    policy_base_logits = (
        probabilities.log() - initial_gate_bias * gate_deltas.float()
    )
    features = torch.cat(
        (
            candidate_features.float(),
            confidence_logits.unsqueeze(-1),
            gate_deltas.float().unsqueeze(-1) / 4.0,
        ),
        dim=-1,
    )
    return features, policy_base_logits


def build_state_features(
    states: torch.Tensor,
    live: torch.Tensor,
    gate_counts: torch.Tensor,
) -> torch.Tensor:
    live_float = live.unsqueeze(-1)
    mean_pool = (states.float() * live_float).sum(1)
    mean_pool = mean_pool / live_float.sum(1).clamp_min(1)
    max_pool = states.float().masked_fill(~live.unsqueeze(-1), -torch.inf).amax(1)
    max_pool = torch.where(
        torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool)
    )
    scaled_gate_count = torch.log1p(gate_counts.float()).unsqueeze(-1) / 8.0
    return torch.cat((mean_pool, max_pool, scaled_gate_count), dim=-1)


def shaped_transition_reward(
    previous_gate_count: int,
    next_gate_count: int,
    *,
    repeated_state: bool,
    step_penalty: float,
    cycle_reward: float,
) -> float:
    """Use exact gate reduction and reject within-episode cycles."""
    if repeated_state:
        return float(cycle_reward)
    return float(previous_gate_count - next_gate_count) - step_penalty


def masked_policy_distribution(
    actor_critic: nn.Module,
    candidate_features: torch.Tensor,
    matcher_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    prefix_states: torch.Tensor | None = None,
    state_features: torch.Tensor | None = None,
) -> torch.distributions.Categorical:
    logits = actor_critic.policy_logits(
        candidate_features,
        matcher_logits,
        candidate_mask,
        prefix_states,
        state_features,
    )
    logits = logits.masked_fill(~candidate_mask, -torch.inf)
    if not bool(candidate_mask.any(1).all()):
        raise ValueError("every PPO state must contain at least one candidate")
    return torch.distributions.Categorical(logits=logits)


def categorical_reference_kl(
    distribution: torch.distributions.Categorical,
    reference_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Exact mean KL from the current policy to a masked reference policy."""
    reference = torch.distributions.Categorical(
        logits=reference_logits.masked_fill(~candidate_mask, -torch.inf)
    )
    return torch.distributions.kl_divergence(distribution, reference).mean()


def segmented_log_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Normalize flattened candidate logits independently for each state."""
    if logits.ndim != 1 or segment_ids.shape != logits.shape:
        raise ValueError("logits and segment IDs must be aligned vectors")
    maxima = torch.full(
        (num_segments,), -torch.inf, dtype=logits.dtype, device=logits.device
    )
    maxima.scatter_reduce_(
        0, segment_ids, logits, reduce="amax", include_self=True
    )
    shifted = logits - maxima.index_select(0, segment_ids)
    denominators = torch.zeros_like(maxima)
    denominators.index_add_(0, segment_ids, shifted.exp())
    return shifted - denominators.index_select(0, segment_ids).log()


def generalized_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        rewards.ndim != 1
        or values.shape != rewards.shape
        or dones.shape != rewards.shape
    ):
        raise ValueError("rewards, values, and dones must be aligned vectors")
    advantages = torch.zeros_like(rewards)
    next_value = rewards.new_zeros(())
    next_advantage = rewards.new_zeros(())
    for index in range(rewards.numel() - 1, -1, -1):
        nonterminal = (~dones[index]).to(rewards.dtype)
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        next_advantage = (
            delta + gamma * gae_lambda * nonterminal * next_advantage
        )
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values


def clipped_ppo_objective(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    *,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> PPOObjective:
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    clipped_values = old_values + (new_values - old_values).clamp(
        -clip_epsilon, clip_epsilon
    )
    value_loss = 0.5 * torch.maximum(
        (new_values - returns).square(),
        (clipped_values - returns).square(),
    ).mean()
    mean_entropy = entropy.mean()
    loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * mean_entropy
    )
    approximate_kl = ((ratio - 1) - log_ratio).mean()
    clip_fraction = (ratio.sub(1).abs() > clip_epsilon).float().mean()
    return PPOObjective(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=mean_entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )
