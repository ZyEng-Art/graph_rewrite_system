from __future__ import annotations

import torch

from beam_search_benchmark import Proposal
from train_paged_ppo import pad_batched_policy_inputs


def _proposal(parent: int, index: int) -> Proposal:
    return Proposal(
        parent=parent,
        xfer_id=index,
        anchor_slot=100 + index,
        binding=(index, index + 1),
        probability=0.1 * (index + 1),
        next_gate_count=20 - index,
    )


def _assert_backends_match(
    features: torch.Tensor,
    logits: torch.Tensor,
    proposals: list[Proposal],
    batch_size: int,
) -> None:
    parent_ids = torch.tensor(
        [proposal.parent for proposal in proposals],
        dtype=torch.long,
        device=features.device,
    )
    expected = pad_batched_policy_inputs(
        features, logits, proposals, batch_size, backend="loop"
    )
    actual = pad_batched_policy_inputs(
        features,
        logits,
        proposals,
        batch_size,
        parent_ids=parent_ids,
        backend="tensorized",
    )
    for expected_tensor, actual_tensor in zip(expected[:3], actual[:3]):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    assert actual[3] == expected[3]
    torch.testing.assert_close(actual[4], expected[4])

    deferred = pad_batched_policy_inputs(
        features,
        logits,
        None,
        batch_size,
        parent_ids=parent_ids,
        backend="tensorized",
    )
    for expected_tensor, deferred_tensor in zip(expected[:3], deferred[:3]):
        torch.testing.assert_close(deferred_tensor, expected_tensor)
    assert deferred[3] == [[] for _ in range(batch_size)]
    torch.testing.assert_close(deferred[4], expected[4])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parents = [2, 0, 1, 0, 2, 1, 0]
    proposals = [_proposal(parent, index) for index, parent in enumerate(parents)]
    features = torch.arange(
        len(proposals) * 5, dtype=torch.float32, device=device
    ).reshape(len(proposals), 5)
    logits = torch.linspace(-1.0, 1.0, len(proposals), device=device)
    _assert_backends_match(features, logits, proposals, batch_size=4)

    _assert_backends_match(
        torch.empty((0, 5), device=device),
        torch.empty(0, device=device),
        [],
        batch_size=3,
    )

    try:
        pad_batched_policy_inputs(
            features, logits, proposals, 4, backend="tensorized"
        )
    except ValueError as error:
        assert "requires GPU parent IDs" in str(error)
    else:
        raise AssertionError("tensorized padding accepted missing parent IDs")

    print(f"PPO policy padding backends agree on {device}")


if __name__ == "__main__":
    main()
