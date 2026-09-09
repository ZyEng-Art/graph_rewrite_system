from __future__ import annotations

import torch

from search_feedback import SearchFeedbackRegistry


def descendant_label_tensors(
    registry: SearchFeedbackRegistry,
    child_node_ids: torch.Tensor,
    parent_gate_counts: torch.Tensor,
    edge_steps: torch.Tensor,
    *,
    observation_end_step: int,
) -> dict[str, torch.Tensor]:
    """Render post-search descendant outcomes for attempted sibling actions.

    Invalid actions use child id and gate labels of -1. A zero continuation gain
    is right-censored: the observed search did not prove that the child can never
    improve under a larger or differently allocated budget.
    """
    if not (
        child_node_ids.ndim == parent_gate_counts.ndim == edge_steps.ndim == 1
        and child_node_ids.numel()
        == parent_gate_counts.numel()
        == edge_steps.numel()
    ):
        raise ValueError("sibling descendant label inputs must be aligned vectors")
    child_gates = []
    best_descendant_gates = []
    continuation_gains = []
    parent_total_gains = []
    time_to_best = []
    right_censored = []
    remaining_steps = []
    child_observed_expansions = []
    child_attempted_actions = []
    for child_id, parent_gate, edge_step in zip(
        child_node_ids.tolist(),
        parent_gate_counts.tolist(),
        edge_steps.tolist(),
    ):
        remaining_steps.append(max(0, int(observation_end_step) - int(edge_step)))
        if int(child_id) < 0:
            child_gates.append(-1)
            best_descendant_gates.append(-1)
            continuation_gains.append(0)
            parent_total_gains.append(0)
            time_to_best.append(-1)
            right_censored.append(False)
            child_observed_expansions.append(0)
            child_attempted_actions.append(0)
            continue
        node = registry.nodes[int(child_id)]
        best = (
            node.gate_count
            if node.best_descendant_gate is None
            else int(node.best_descendant_gate)
        )
        continuation_gain = max(0, int(node.gate_count) - best)
        child_gates.append(int(node.gate_count))
        best_descendant_gates.append(best)
        continuation_gains.append(continuation_gain)
        parent_total_gains.append(max(0, int(parent_gate) - best))
        time_to_best.append(
            max(0, int(node.last_improvement_step) - int(edge_step))
            if continuation_gain > 0
            else -1
        )
        right_censored.append(continuation_gain == 0)
        child_observed_expansions.append(int(node.observed_expansions))
        child_attempted_actions.append(int(node.attempted_actions))
    return {
        "child_gate_counts": torch.tensor(child_gates, dtype=torch.int32),
        "best_descendant_gate_counts": torch.tensor(
            best_descendant_gates, dtype=torch.int32
        ),
        "continuation_gains": torch.tensor(continuation_gains, dtype=torch.int16),
        "parent_total_gains": torch.tensor(parent_total_gains, dtype=torch.int16),
        "time_to_observed_best_descendant": torch.tensor(
            time_to_best, dtype=torch.int32
        ),
        "right_censored": torch.tensor(right_censored, dtype=torch.bool),
        "remaining_search_steps": torch.tensor(remaining_steps, dtype=torch.int32),
        "child_observed_expansions": torch.tensor(
            child_observed_expansions, dtype=torch.int16
        ),
        "child_attempted_actions": torch.tensor(
            child_attempted_actions, dtype=torch.int32
        ),
    }
