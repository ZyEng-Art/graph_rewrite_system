from types import SimpleNamespace
import unittest

import torch

from model import S0ActionBindingModel


def classification_loss(
    logits: torch.Tensor,
    positives: list[list[tuple[int, tuple[int, ...]]]],
    target_action: dict,
    action_positive_weight: float,
) -> torch.Tensor:
    # classification_loss only needs num_sources when structural decoding is off.
    model = SimpleNamespace(num_sources=logits.shape[-1])
    return S0ActionBindingModel.classification_loss(
        model,
        logits,
        torch.ones_like(logits, dtype=torch.bool),
        positives,
        batch={"target_actions": [target_action]},
        action_positive_weight=action_positive_weight,
    )


class ActionPositiveLossTest(unittest.TestCase):
    def test_emphasizes_the_weak_chosen_match(self) -> None:
        # Both rows are exact positive matches, but the trajectory actually
        # chooses source 1. Its deliberately weak logit should matter more.
        positives = [[(0, (0,)), (1, (0,))]]
        target = {"source_id": 1, "binding_slots": (0,)}
        logits = torch.tensor([[[4.0, -4.0, 1.0, 0.0]]], requires_grad=True)

        torch.manual_seed(7)
        baseline = classification_loss(logits, positives, target, 0.0)
        torch.manual_seed(7)
        emphasized = classification_loss(logits, positives, target, 4.0)

        self.assertTrue(torch.isfinite(emphasized))
        self.assertGreater(float(emphasized), float(baseline))

    def test_rejects_an_inexact_target(self) -> None:
        logits = torch.zeros((1, 1, 3))
        with self.assertRaisesRegex(
            ValueError, "absent from exact positive matches"
        ):
            classification_loss(
                logits,
                [[(0, (0,))]],
                {"source_id": 1, "binding_slots": (0,)},
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
