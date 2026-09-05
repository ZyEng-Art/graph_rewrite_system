from __future__ import annotations

import math

import torch

from ppo_core import (
    PagedPPOActorCritic,
    build_policy_features,
    build_state_features,
    clipped_ppo_objective,
    generalized_advantages,
    masked_policy_distribution,
    segmented_log_softmax,
    shaped_transition_reward,
)


def main() -> None:
    rewards = torch.tensor([1.0, 2.0, 4.0])
    values = torch.zeros(3)
    dones = torch.tensor([False, True, True])
    advantages, returns = generalized_advantages(
        rewards,
        values,
        dones,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.equal(advantages, torch.tensor([3.0, 2.0, 4.0]))
    assert torch.equal(returns, advantages)

    objective = clipped_ppo_objective(
        new_log_probs=torch.tensor([math.log(1.5), math.log(0.5)]),
        old_log_probs=torch.zeros(2),
        advantages=torch.tensor([1.0, -1.0]),
        new_values=torch.zeros(2),
        old_values=torch.zeros(2),
        returns=torch.zeros(2),
        entropy=torch.zeros(2),
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
    )
    # Positive advantages cap an over-large increase at 1.2. For a negative
    # advantage, an over-large decrease is also capped at 0.8.
    assert torch.isclose(objective.policy_loss, torch.tensor(-0.2))
    assert torch.isclose(objective.clip_fraction, torch.tensor(1.0))

    width = 4
    actor_critic = PagedPPOActorCritic(width)
    base_features = torch.randn(3, 4 * width)
    policy_features, matcher_logits = build_policy_features(
        base_features,
        torch.tensor([0.2, 0.8, 0.6]),
        torch.zeros(3),
    )
    candidate_mask = torch.tensor([[True, True, False]])
    distribution = masked_policy_distribution(
        actor_critic,
        policy_features.unsqueeze(0),
        matcher_logits.unsqueeze(0),
        candidate_mask,
    )
    expected = torch.tensor([0.2, 0.8])
    expected = expected / expected.sum()
    assert torch.allclose(distribution.probs[0, :2], expected)
    assert distribution.probs[0, 2] == 0
    flattened_logits = torch.tensor([1.0, 2.0, -2.0, 0.0, 2.0])
    parent_ids = torch.tensor([0, 0, 1, 1, 1])
    normalized = segmented_log_softmax(flattened_logits, parent_ids, 2)
    shifted_normalized = segmented_log_softmax(
        flattened_logits + torch.tensor([10.0, 10.0, -7.0, -7.0, -7.0]),
        parent_ids,
        2,
    )
    assert torch.allclose(normalized, shifted_normalized)
    assert torch.allclose(normalized[:2].exp().sum(), torch.tensor(1.0))
    assert torch.allclose(normalized[2:].exp().sum(), torch.tensor(1.0))
    _, gate_biased_logits = build_policy_features(
        base_features[:2],
        torch.tensor([0.5, 0.5]),
        torch.tensor([-1, 1]),
        initial_gate_bias=1.0,
    )
    assert torch.isclose(
        gate_biased_logits[0] - gate_biased_logits[1], torch.tensor(2.0)
    )

    states = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [100.0, 100.0, 100.0, 100.0],
            ]
        ]
    )
    state_features = build_state_features(
        states,
        torch.tensor([[True, True, False]]),
        torch.tensor([58]),
    )
    assert state_features.shape == (1, 2 * width + 1)
    assert torch.allclose(state_features[0, :width], torch.full((width,), 2.5))
    assert torch.equal(
        state_features[0, width : 2 * width],
        torch.tensor([4.0, 3.0, 3.0, 4.0]),
    )
    assert actor_critic.state_values(state_features).item() == 0.0

    assert shaped_transition_reward(
        58,
        57,
        repeated_state=False,
        step_penalty=0.01,
        cycle_reward=-1.0,
    ) == 0.99
    assert shaped_transition_reward(
        57,
        58,
        repeated_state=False,
        step_penalty=0.01,
        cycle_reward=-1.0,
    ) == -1.01
    assert shaped_transition_reward(
        60,
        58,
        repeated_state=True,
        step_penalty=0.01,
        cycle_reward=-1.0,
    ) == -1.0
    print("PPO GAE, clipping, masking, and neutral initialization are correct")


if __name__ == "__main__":
    main()
