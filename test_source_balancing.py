from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from audit_matcher_generalization import (
    exclude_trajectory_partition,
    prefix_training_state_count,
    source_frequency_bucket,
)
from dataset import RuleMetadata, TargetSourceBalancedSampler, inverse_frequency_weights
from model import S0ActionBindingModel
from train import (
    configure_source_adapter_only,
    configure_source_topology_only,
    register_source_adapter_frequency_mask,
)


class TinyPrefixDataset:
    def __init__(self, target_sources: list[int]):
        self.target_sources = target_sources

    def __len__(self) -> int:
        return len(self.target_sources)

    def target_source_id(self, index: int) -> int:
        return self.target_sources[index]


class SourceBalancingTest(unittest.TestCase):
    def test_source_frequency_prior_shrinks_only_memorized_id_terms(self):
        rules = RuleMetadata.from_payload(
            {
                "source_patterns": ("cx 0 1; h 0", "cx 0 1; h 1"),
                "xfer_to_source": (0, 1),
                "xfer_sources": ("cx 0 1; h 0", "cx 0 1; h 1"),
                "xfer_destinations": ("h 0", "h 1"),
            }
        )
        model = S0ActionBindingModel(
            rules,
            num_xfers=2,
            width=12,
            retrieval_width=8,
            graph_layers=0,
            current_graph_layers=0,
            source_topology_layers=1,
            source_id_frequency_prior=1.0,
        )
        model.source_bias.data.copy_(torch.tensor([2.0, 4.0]))

        model.set_source_frequency_prior(torch.tensor([0, 3]))

        torch.testing.assert_close(
            model.source_id_weight, torch.tensor([0.0, 0.75])
        )
        torch.testing.assert_close(
            model.source_bias_values(), torch.tensor([0.0, 3.0])
        )
        self.assertEqual(model.source_edge_relation.tolist(), [0, 4])
        self.assertTrue(bool(model.source_representations().isfinite().all()))

    def test_source_adapter_only_freezes_shared_parameters(self):
        model = nn.Module()
        model.source_embedding = nn.Embedding(3, 4)
        model.source_bias = nn.Parameter(torch.zeros(3))
        model.source_topology_output = nn.Linear(4, 4)
        model.shared = nn.Linear(4, 4)

        trainable = configure_source_adapter_only(model)

        self.assertEqual(trainable, 35)
        self.assertTrue(model.source_embedding.weight.requires_grad)
        self.assertTrue(model.source_bias.requires_grad)
        self.assertTrue(model.source_topology_output.weight.requires_grad)
        self.assertTrue(model.source_topology_output.bias.requires_grad)
        self.assertFalse(model.shared.weight.requires_grad)
        self.assertFalse(model.shared.bias.requires_grad)

    def test_source_adapter_frequency_mask_blocks_common_and_unseen_rows(self):
        model = nn.Module()
        model.num_sources = 3
        model.source_embedding = nn.Embedding(3, 2)
        model.source_bias = nn.Parameter(torch.zeros(3))
        register_source_adapter_frequency_mask(
            model, torch.tensor([False, True, False])
        )

        (model.source_embedding.weight.sum() + model.source_bias.sum()).backward()

        torch.testing.assert_close(
            model.source_embedding.weight.grad,
            torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        )
        torch.testing.assert_close(
            model.source_bias.grad, torch.tensor([0.0, 1.0, 0.0])
        )

    def test_source_topology_only_preserves_id_parameters(self):
        model = nn.Module()
        model.source_embedding = nn.Embedding(3, 4)
        model.source_bias = nn.Parameter(torch.zeros(3))
        model.source_topology_layers = nn.ModuleList([nn.Linear(4, 4)])
        model.source_topology_output = nn.Linear(4, 4)
        model.shared = nn.Linear(4, 4)

        trainable = configure_source_topology_only(model)

        self.assertEqual(trainable, 40)
        self.assertFalse(model.source_embedding.weight.requires_grad)
        self.assertFalse(model.source_bias.requires_grad)
        self.assertTrue(model.source_topology_layers[0].weight.requires_grad)
        self.assertTrue(model.source_topology_output.weight.requires_grad)
        self.assertFalse(model.shared.weight.requires_grad)

    def test_source_frequency_buckets_have_stable_boundaries(self):
        self.assertEqual(
            [
                source_frequency_bucket(value)
                for value in (0, 1, 2, 7, 8, 63, 64, 511, 512)
            ],
            [
                "unseen",
                "1",
                "2-7",
                "2-7",
                "8-63",
                "8-63",
                "64-511",
                "64-511",
                "512+",
            ],
        )

    def test_training_state_count_matches_terminal_sampling_rules(self):
        trajectories = [
            {"steps": [{}, {}], "terminal_matches": []},
            {
                "steps": [{}, {}, {}],
                "terminal_matches": [],
                "terminal_only_supervision": True,
            },
        ]
        self.assertEqual(
            prefix_training_state_count(
                trajectories,
                include_terminal=False,
                terminal_only_repeat=4,
            ),
            2,
        )
        self.assertEqual(
            prefix_training_state_count(
                trajectories,
                include_terminal=True,
                terminal_only_repeat=4,
            ),
            7,
        )

    def test_calibration_partition_is_excluded_by_trajectory_id(self):
        trajectories = [{"trajectory_id": value} for value in range(8)]
        retained, excluded = exclude_trajectory_partition(
            trajectories, modulo=4, remainder=0
        )
        self.assertEqual(excluded, 2)
        self.assertEqual(
            [trajectory["trajectory_id"] for trajectory in retained],
            [1, 2, 3, 5, 6, 7],
        )

    def test_inverse_frequency_is_bounded_and_leaves_unseen_sources_alone(self):
        counts = torch.tensor([100, 25, 1, 0])
        weights = inverse_frequency_weights(counts, power=0.5, cap=6.0)
        torch.testing.assert_close(weights, torch.tensor([1.0, 2.0, 6.0, 1.0]))

    def test_rare_positive_receives_more_classification_gradient(self):
        model = SimpleNamespace(num_sources=3)
        logits = torch.tensor([[[0.0, 0.0, 2.0]]], requires_grad=True)
        positives = [[(0, (0,)), (1, (0,))]]
        source_weights = torch.tensor([1.0, 8.0, 1.0])
        torch.manual_seed(11)
        loss = S0ActionBindingModel.classification_loss(
            model,
            logits,
            torch.ones_like(logits, dtype=torch.bool),
            positives,
            source_positive_weights=source_weights,
        )
        loss.backward()
        self.assertGreater(abs(float(logits.grad[0, 0, 1])), abs(float(logits.grad[0, 0, 0])))

    def test_interleaved_binding_receives_more_classification_gradient(self):
        model = SimpleNamespace(num_sources=4)
        logits = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]], requires_grad=True)
        positives = [[(0, (0,)), (1, (0, 10))]]
        torch.manual_seed(13)
        loss = S0ActionBindingModel.classification_loss(
            model,
            logits,
            torch.ones_like(logits, dtype=torch.bool),
            positives,
            interleaving_positive_weight=4.0,
            interleaving_positive_scale=4.0,
        )
        loss.backward()
        self.assertGreater(
            abs(float(logits.grad[0, 0, 1])),
            abs(float(logits.grad[0, 0, 0])),
        )

    def test_hard_ceiling_does_not_reweight_an_already_strong_rare_positive(self):
        model = SimpleNamespace(num_sources=3)
        positives = [[(0, (0,)), (1, (0,))]]
        source_weights = torch.tensor([1.0, 8.0, 1.0])
        ceilings = torch.tensor([-1.0, -1.0])
        baseline_logits = torch.tensor([[[0.0, 4.0, 2.0]]], requires_grad=True)
        gated_logits = baseline_logits.detach().clone().requires_grad_(True)
        torch.manual_seed(19)
        baseline = S0ActionBindingModel.classification_loss(
            model,
            baseline_logits,
            torch.ones_like(baseline_logits, dtype=torch.bool),
            positives,
        )
        torch.manual_seed(19)
        gated = S0ActionBindingModel.classification_loss(
            model,
            gated_logits,
            torch.ones_like(gated_logits, dtype=torch.bool),
            positives,
            batch={"positive_near": [[False, False]]},
            source_positive_weights=source_weights,
            source_positive_hard_ceilings=ceilings,
        )
        torch.testing.assert_close(gated, baseline)

    def test_hard_ceiling_reweights_only_the_weak_rare_positive(self):
        model = SimpleNamespace(num_sources=3)
        positives = [[(0, (0,)), (1, (0,))]]
        source_weights = torch.tensor([1.0, 8.0, 1.0])
        logits = torch.tensor([[[0.0, -2.0, 2.0]]], requires_grad=True)
        torch.manual_seed(23)
        loss = S0ActionBindingModel.classification_loss(
            model,
            logits,
            torch.ones_like(logits, dtype=torch.bool),
            positives,
            batch={"positive_near": [[False, False]]},
            source_positive_weights=source_weights,
            source_positive_hard_ceilings=torch.tensor([-1.0, -1.0]),
        )
        loss.backward()
        self.assertGreater(
            abs(float(logits.grad[0, 0, 1])),
            abs(float(logits.grad[0, 0, 0])),
        )

    def test_target_sampler_is_deterministic_and_upsamples_rare_sources(self):
        dataset = TinyPrefixDataset([0] * 100 + [1])
        sampler_a = TargetSourceBalancedSampler(
            dataset,
            2,
            17,
            power=1.0,
            cap=100.0,
            fraction=1.0,
        )
        sampler_b = TargetSourceBalancedSampler(
            dataset,
            2,
            17,
            power=1.0,
            cap=100.0,
            fraction=1.0,
        )
        first = list(sampler_a)
        self.assertEqual(first, list(sampler_b))
        rare_count = sum(index == 100 for index in first)
        self.assertGreater(rare_count, 20)


if __name__ == "__main__":
    unittest.main()
