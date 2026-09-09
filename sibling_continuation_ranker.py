from __future__ import annotations

import torch
from torch import nn


class SiblingContinuationRanker(nn.Module):
    """Small head over frozen matcher/action features.

    A larger score means that allocating later search budget to the proposed
    child is expected to produce a better descendant than its siblings.
    """

    def __init__(
        self,
        input_width: int,
        hidden_width: int = 256,
        dropout: float = 0.05,
        base_probability_index: int | None = None,
    ) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.base_probability_index = (
            int(base_probability_index)
            if base_probability_index is not None
            else input_width - 8
        )
        self.input_norm = nn.LayerNorm(input_width)
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probability = inputs[:, self.base_probability_index].float().clamp(
            1e-6, 1 - 1e-6
        )
        return torch.logit(probability) + self.residual(inputs)

    def residual(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(self.input_norm(inputs)).squeeze(-1)


def continuation_ranker_inputs(payload: dict) -> torch.Tensor:
    """Build inference-available inputs without descendant-label leakage."""

    auxiliary = torch.stack(
        (
            payload["probabilities"].float(),
            payload["gate_deltas"].float() / 8.0,
            payload["parent_gate_counts"].float() / 512.0,
            payload["steps"].float() / 64.0,
            payload["action_parent_ranks"].float() / 1024.0,
            payload["parent_expansion_rounds"].float() / 8.0,
            payload["parent_stagnation_steps"].float() / 64.0,
            payload["parent_action_depths"].float() / 64.0,
        ),
        dim=1,
    )
    return torch.cat((payload["features"].float(), auxiliary), dim=1)
