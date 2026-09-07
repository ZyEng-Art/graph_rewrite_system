from pathlib import Path
from types import SimpleNamespace

import torch

from beam_search_benchmark import (
    frozen_candidate_features,
    neural_prefilter_scores,
)
from train_neural_successor_prefilter import (
    AuditRows,
    NeuralSuccessorPrefilter,
    build_pairs,
    threshold_for_false_positive_rate,
)


class _FeatureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.width = 2
        self.xfer_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )


def test_frozen_candidate_features_pool_exact_rows() -> None:
    model = _FeatureModel()
    states = torch.tensor(
        [
            [[2.0, 0.0], [4.0, 2.0], [0.0, 0.0]],
            [[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]],
        ]
    )
    live = torch.tensor(
        [[True, True, False], [True, True, True]]
    )
    proposals = SimpleNamespace(
        parent_ids=torch.tensor([0, 1]),
        xfer_ids=torch.tensor([1, 0]),
        source_ids=torch.tensor([0, 1]),
        bindings=torch.tensor([[0, 1], [1, -1]]),
    )
    sources = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    features = frozen_candidate_features(
        model, states, live, proposals, sources
    )

    assert features.shape == (2, 8)
    assert torch.equal(features[0, :2], torch.tensor([3.0, 4.0]))
    assert torch.equal(features[0, 2:4], torch.tensor([10.0, 20.0]))
    assert torch.equal(features[0, 4:6], torch.tensor([3.0, 1.0]))
    assert torch.equal(features[0, 6:8], torch.tensor([3.0, 1.0]))
    assert torch.equal(features[1, 4:6], torch.tensor([5.0, 7.0]))
    assert torch.equal(features[1, 6:8], torch.tensor([5.0, 7.0]))


def test_zero_false_positive_threshold_is_strictly_outside_safe_scores() -> None:
    safe = torch.tensor([0.2, 0.4, 0.8])
    low_threshold = threshold_for_false_positive_rate(
        safe, false_positive_rate=0.0, low_is_positive=True
    )
    high_threshold = threshold_for_false_positive_rate(
        safe, false_positive_rate=0.0, low_is_positive=False
    )

    assert low_threshold < float(safe.min())
    assert not safe.lt(low_threshold).any()
    assert high_threshold > float(safe.max())
    assert not safe.gt(high_threshold).any()


def test_pair_builder_emits_balanced_exact_group_pairs() -> None:
    rows = AuditRows(
        path=Path("synthetic.pt"),
        features=torch.zeros(6, 2),
        outcomes=torch.tensor([2, 1, 2, 1, 2, 0]),
        groups=torch.tensor([10, 10, 20, 20, 30, -1]),
        xfer_ids=torch.tensor([1, 1, 2, 2, 1, 3]),
        source_ids=torch.zeros(6, dtype=torch.long),
        probabilities=torch.zeros(6),
        gate_deltas=torch.zeros(6, dtype=torch.short),
        parent_gate_counts=torch.ones(6, dtype=torch.short),
        steps=torch.ones(6, dtype=torch.short),
    )

    left, right, labels = build_pairs(rows, max_pairs=100, seed=7)

    assert labels.tolist().count(1.0) == 2
    assert labels.tolist().count(0.0) == 2
    for left_index, right_index, label in zip(left, right, labels):
        same_group = rows.groups[left_index] == rows.groups[right_index]
        assert bool(same_group) == bool(label)


def test_neural_scores_are_batched_and_finite() -> None:
    prefilter = NeuralSuccessorPrefilter(
        input_width=12,
        hidden_width=8,
        embedding_width=4,
        dropout=0.0,
    )
    frozen = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    proposals = SimpleNamespace(
        parent_ids=torch.tensor([0, 1, 1]),
        probabilities=torch.tensor([0.1, 0.2, 0.3]),
        gate_deltas=torch.tensor([-1, 0, 1]),
    )
    beam = [SimpleNamespace(gate_count=20), SimpleNamespace(gate_count=30)]

    valid, duplicate = neural_prefilter_scores(
        prefilter,
        frozen,
        proposals,
        beam,
        step=3,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert valid.shape == (3,)
    assert duplicate.shape == (3,)
    assert torch.isfinite(valid).all()
    assert torch.isfinite(duplicate).all()
    assert ((0 <= valid) & (valid <= 1)).all()
    assert ((0 <= duplicate) & (duplicate <= 1)).all()
