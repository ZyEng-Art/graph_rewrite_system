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
        num_xfers: int = 0,
        prefix_width: int = 0,
    ) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.base_probability_index = (
            int(base_probability_index)
            if base_probability_index is not None
            else input_width - 8
        )
        self.num_xfers = int(num_xfers)
        self.prefix_width = int(prefix_width)
        if (self.num_xfers > 0) != (self.prefix_width > 0):
            raise ValueError("num_xfers and prefix_width must both be enabled")
        if self.prefix_width:
            self.prefix_embedding = nn.Embedding(
                self.num_xfers + 1, self.prefix_width, padding_idx=0
            )
            self.prefix_encoder = nn.GRU(
                self.prefix_width, self.prefix_width, batch_first=True
            )
        else:
            self.prefix_embedding = None
            self.prefix_encoder = None
        combined_width = input_width + self.prefix_width
        self.input_norm = nn.LayerNorm(combined_width)
        self.network = nn.Sequential(
            nn.Linear(combined_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        inputs: torch.Tensor,
        prefix_xfers: torch.Tensor | None = None,
        prefix_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probability = inputs[:, self.base_probability_index].float().clamp(
            1e-6, 1 - 1e-6
        )
        return torch.logit(probability) + self.residual(
            inputs, prefix_xfers, prefix_lengths
        )

    def residual(
        self,
        inputs: torch.Tensor,
        prefix_xfers: torch.Tensor | None = None,
        prefix_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        combined = inputs
        if self.prefix_width:
            if prefix_xfers is None or prefix_lengths is None:
                raise ValueError("enabled prefix encoder requires tokens and lengths")
            prefix_xfers = prefix_xfers.clamp_max(self.num_xfers)
            embedded = self.prefix_embedding(prefix_xfers)
            encoded, _ = self.prefix_encoder(embedded)
            positions = (prefix_lengths - 1).clamp_min(0)
            prefix = encoded[
                torch.arange(encoded.shape[0], device=encoded.device), positions
            ]
            prefix = prefix.masked_fill(prefix_lengths.eq(0).unsqueeze(1), 0)
            combined = torch.cat((inputs, prefix), dim=1)
        return self.network(self.input_norm(combined)).squeeze(-1)


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
