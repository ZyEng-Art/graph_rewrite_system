import unittest

from dataset import RuleMetadata
from lazy_rollout_benchmark import (
    LazyAction,
    is_direct_inverse_action,
    violates_canonical_trace_order,
)


def rules() -> RuleMetadata:
    return RuleMetadata(
        source_patterns=("h 0;", "x 0;", "rz 0; cx 0 1;", "cx 0 1; rz 0;"),
        source_gate_types=((0,), (1,), (5, 6), (6, 5)),
        destination_gate_types=((1,), (0,), (6, 5), (5, 6)),
        xfer_to_source=(0, 1, 2, 3),
        xfer_sources=("h 0;", "x 0;", "rz 0; cx 0 1;", "cx 0 1; rz 0;"),
        xfer_destinations=("x 0;", "h 0;", "cx 0 1; rz 0;", "rz 0; cx 0 1;"),
    )


class ActionTraceDedupTest(unittest.TestCase):
    def test_unique_inverse_map(self) -> None:
        self.assertEqual(rules().unique_inverse_xfer_ids(), (1, 0, 3, 2))

    def test_ambiguous_inverse_is_disabled(self) -> None:
        metadata = rules()
        ambiguous = RuleMetadata(
            **{
                **metadata.__dict__,
                "xfer_sources": metadata.xfer_sources + ("x 0;",),
                "xfer_destinations": metadata.xfer_destinations + ("h 0;",),
                "xfer_to_source": metadata.xfer_to_source + (1,),
                "destination_gate_types": metadata.destination_gate_types
                + ((0,),),
            }
        )
        inverse = ambiguous.unique_inverse_xfer_ids()
        self.assertEqual(inverse[0], -1)
        self.assertEqual(inverse[1], 0)

    def test_direct_inverse_must_consume_previous_destination(self) -> None:
        previous = LazyAction(2, (6, 7), (39, 40), (5, 6, 7, 39, 40))
        inverse = rules().unique_inverse_xfer_ids()

        self.assertTrue(
            is_direct_inverse_action((previous,), 3, (39, 40), inverse)
        )
        self.assertFalse(is_direct_inverse_action((previous,), 3, (8, 9), inverse))

    def test_orders_only_independent_suffix(self) -> None:
        high_key = LazyAction(3, (8, 9), (39, 40), (8, 9, 20, 39, 40))
        lower_independent = LazyAction(2, (6, 7), (41, 42), (6, 7, 21, 41, 42))
        lower_conflicting = LazyAction(2, (6, 7), (41, 42), (7, 9, 41, 42))

        self.assertTrue(
            violates_canonical_trace_order((high_key,), lower_independent)
        )
        self.assertFalse(
            violates_canonical_trace_order((high_key,), lower_conflicting)
        )

    def test_legacy_action_without_footprint_is_barrier(self) -> None:
        legacy = LazyAction(3, (8, 9), (39, 40))
        candidate = LazyAction(2, (6, 7), (41, 42), (6, 7, 41, 42))

        self.assertFalse(violates_canonical_trace_order((legacy,), candidate))


if __name__ == "__main__":
    unittest.main()
