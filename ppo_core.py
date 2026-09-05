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


class PagedPPOActorCritic(nn.Module):
    """Small PPO heads over frozen paged-model state and action features."""

    def __init__(self, width: int, hidden_size: int | None = None) -> None:
        super().__init__()
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
    ) -> torch.Tensor:
        residual = self.policy(self.policy_norm(candidate_features)).squeeze(-1)
        return matcher_logits + residual

    def state_values(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.value(self.value_norm(state_features)).squeeze(-1)


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
    actor_critic: PagedPPOActorCritic,
    candidate_features: torch.Tensor,
    matcher_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.distributions.Categorical:
    logits = actor_critic.policy_logits(candidate_features, matcher_logits)
    logits = logits.masked_fill(~candidate_mask, -torch.inf)
    if not bool(candidate_mask.any(1).all()):
        raise ValueError("every PPO state must contain at least one candidate")
    return torch.distributions.Categorical(logits=logits)


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
