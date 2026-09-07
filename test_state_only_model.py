import unittest

import torch

from dataset import RuleMetadata
from model import S0ActionBindingModel


def rules() -> RuleMetadata:
    return RuleMetadata(
        source_patterns=("x 0",),
        source_gate_types=((1,),),
        destination_gate_types=((),),
        xfer_to_source=(0,),
        xfer_sources=("x 0",),
        xfer_destinations=("",),
    )


def chain_batch(last_type: int = 1) -> dict:
    return {
        "current_types": torch.tensor([[1, 1, 1, last_type]]),
        "current_edge_batch": torch.tensor([0, 0, 0]),
        "current_edge_src": torch.tensor([0, 1, 2]),
        "current_edge_dst": torch.tensor([1, 2, 3]),
        "current_edge_relation": torch.tensor([0, 0, 0]),
        "current_rewrite_distance": torch.full((1, 4), 5),
        "current_touch_age": torch.full((1, 4), 7),
        "current_local_streak": torch.zeros(1, dtype=torch.long),
    }


class StateOnlyCurrentGraphTest(unittest.TestCase):
    def model(self, *, depth: int, identity_prefix: int = 0):
        return S0ActionBindingModel(
            rules(),
            num_xfers=1,
            width=24,
            retrieval_width=16,
            graph_layers=1,
            current_graph_layers=depth,
            use_action_history=False,
            identity_current_prefix=identity_prefix,
            dropout=0.0,
        ).eval()

    def test_state_only_encode_is_exact_current_graph_entrypoint(self) -> None:
        model = self.model(depth=2)
        batch = chain_batch()
        with torch.no_grad():
            expected = model.encode_current_graph(batch)
            actual = model.encode(batch)
        self.assertTrue(torch.equal(expected[1], actual[1]))
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertTrue(torch.equal(expected[0], actual[0]))

    def test_appended_hops_begin_as_an_exact_identity_extension(self) -> None:
        torch.manual_seed(19)
        shallow = self.model(depth=2)
        deep = self.model(depth=4, identity_prefix=2)
        compatible = {
            key: value
            for key, value in shallow.state_dict().items()
            if key in deep.state_dict() and deep.state_dict()[key].shape == value.shape
        }
        deep.load_state_dict(compatible, strict=False)

        base = chain_batch(last_type=1)
        distant_change = chain_batch(last_type=0)
        with torch.no_grad():
            shallow_states = shallow.encode_current_graph(base)[0]
            deep_states = deep.encode_current_graph(base)[0]
            before = deep.encode_current_graph(base)[0][:, 0]
            changed_before = deep.encode_current_graph(distant_change)[0][:, 0]

        self.assertTrue(torch.equal(shallow_states, deep_states))
        # The changed gate is three edges away, outside the active H=2 prefix.
        self.assertTrue(torch.equal(before, changed_before))

        with torch.no_grad():
            for scale in deep.current_graph_layer_scales:
                scale.fill_(1.0)
            after = deep.encode_current_graph(base)[0][:, 0]
            changed_after = deep.encode_current_graph(distant_change)[0][:, 0]
        # Enabling the appended hops makes the distance-three topology visible.
        self.assertFalse(torch.allclose(after, changed_after, atol=1e-7, rtol=1e-7))


if __name__ == "__main__":
    unittest.main()
