from __future__ import annotations

import torch

from beam_search_benchmark import Proposal
from train_paged_ppo import initial_batch_many, pad_batched_policy_inputs


def main() -> None:
    packed = initial_batch_many(
        [
            {
                "nodes": [(0, 1, -1), (1, 2, -1)],
                "edges": [(0, 1, 0, 1)],
            },
            {
                "nodes": [(0, 3, -1)],
                "edges": [],
            },
        ]
    )
    assert torch.equal(
        packed["initial_types"], torch.tensor([[1, 2], [3, -1]])
    )
    assert torch.equal(packed["edge_batch"], torch.tensor([0]))
    assert torch.equal(packed["edge_src"], torch.tensor([0]))
    assert torch.equal(packed["edge_dst"], torch.tensor([1]))
    assert torch.equal(packed["edge_relation"], torch.tensor([1]))

    proposals = [
        Proposal(1, 10, 3, (3,), 0.2, 12),
        Proposal(0, 11, 4, (4,), 0.7, 8),
        Proposal(1, 12, 5, (5,), 0.4, 11),
    ]
    features = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    logits = torch.tensor([0.1, 0.2, 0.3])
    padded, padded_logits, mask, grouped = pad_batched_policy_inputs(
        features, logits, proposals, batch_size=3
    )
    assert padded.shape == (3, 2, 2)
    assert torch.equal(padded[0, 0], features[1])
    assert torch.equal(padded[1, 0], features[0])
    assert torch.equal(padded[1, 1], features[2])
    assert torch.equal(padded_logits[1], torch.tensor([0.1, 0.3]))
    assert torch.equal(
        mask,
        torch.tensor(
            [[True, False], [True, True], [False, False]]
        ),
    )
    assert [row.xfer_id for row in grouped[0]] == [11]
    assert [row.xfer_id for row in grouped[1]] == [10, 12]
    assert grouped[2] == []
    print("batched PPO initial collation and candidate grouping are correct")


if __name__ == "__main__":
    main()
