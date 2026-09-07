from __future__ import annotations

from dataclasses import dataclass
import unittest

from beam_search_benchmark import is_direct_inverse_proposal
from incremental_graph import parse_pattern
from search_types import BeamState, Proposal
from successor_fingerprint import (
    FingerprintAudit,
    build_wire_trace_profile,
    profile_fingerprint,
    successor_fingerprint,
)


@dataclass(frozen=True)
class FakeNode:
    guid: int
    gate_tp: int


class FakeGraph:
    def __init__(self, qasm: str, nodes: list[FakeNode]):
        self._qasm = qasm
        self.nodes = nodes

    def to_qasm_str(self) -> str:
        return self._qasm


def graph(qasm_body: str, gate_types: list[int], *, guid_start: int = 100):
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
        + qasm_body
    )
    nodes = [
        FakeNode(guid_start + index, gate_type)
        for index, gate_type in enumerate(gate_types)
    ]
    return FakeGraph(qasm, nodes), {
        node.guid: index + guid_start * 10 for index, node in enumerate(nodes)
    }


class SuccessorFingerprintTest(unittest.TestCase):
    def test_parameterized_rule_is_stable_without_guessing_output_value(self):
        parent, parent_slots = graph(
            "rz(pi*0.250000) q[0];\ncx q[0],q[1];\n", [5, 6]
        )
        equivalent_parent, equivalent_slots = graph(
            "rz(pi*0.250000) q[0];\ncx q[0],q[1];\n",
            [5, 6],
            guid_start=200,
        )
        parent_profile = build_wire_trace_profile(parent, parent_slots)
        equivalent_profile = build_wire_trace_profile(
            equivalent_parent, equivalent_slots
        )
        self.assertIsNotNone(parent_profile)
        self.assertIsNotNone(equivalent_profile)

        left = successor_fingerprint(
            parent_profile,
            parse_pattern("rz 0; cx 0 1;"),
            parse_pattern("cx 0 1; rz 0;"),
            tuple(parent_slots.values()),
            xfer_id=38,
            kind="conservative",
        )
        right = successor_fingerprint(
            equivalent_profile,
            parse_pattern("rz 0; cx 0 1;"),
            parse_pattern("cx 0 1; rz 0;"),
            tuple(equivalent_slots.values()),
            xfer_id=38,
            kind="conservative",
        )
        self.assertEqual(left, right)
        expected, expected_slots = graph(
            "cx q[0],q[1];\nrz(pi*0.250000) q[0];\n",
            [6, 5],
            guid_start=300,
        )
        expected_profile = build_wire_trace_profile(expected, expected_slots)
        # The symbolic destination parameter prevents unsafe equality with an
        # actual circuit until Quartz has evaluated the hidden ECC expression.
        self.assertNotEqual(
            left,
            profile_fingerprint(expected_profile, kind="conservative"),
        )

    def test_xfer_guard_separates_parameter_transformations(self):
        parent, parent_slots = graph(
            "rz(pi*0.250000) q[0];\ncx q[0],q[1];\n", [5, 6]
        )
        profile = build_wire_trace_profile(parent, parent_slots)
        arguments = (
            profile,
            parse_pattern("rz 0; cx 0 1;"),
            parse_pattern("cx 0 1; rz 0;"),
            tuple(parent_slots.values()),
        )
        self.assertNotEqual(
            successor_fingerprint(
                *arguments, xfer_id=38, kind="xfer_guarded"
            ),
            successor_fingerprint(
                *arguments, xfer_id=39, kind="xfer_guarded"
            ),
        )

    def test_fingerprint_is_independent_of_guids_and_persistent_slots(self):
        left, left_slots = graph("h q[0];\ncx q[0],q[1];\n", [0, 6])
        right, right_slots = graph(
            "h q[0];\ncx q[0],q[1];\n", [0, 6], guid_start=900
        )
        left_profile = build_wire_trace_profile(left, left_slots)
        right_profile = build_wire_trace_profile(right, right_slots)
        left_fp = successor_fingerprint(
            left_profile,
            parse_pattern("h 0;"),
            parse_pattern("x 0;"),
            (next(iter(left_slots.values())),),
            xfer_id=7,
        )
        right_fp = successor_fingerprint(
            right_profile,
            parse_pattern("h 0;"),
            parse_pattern("x 0;"),
            (next(iter(right_slots.values())),),
            xfer_id=7,
        )
        self.assertEqual(left_fp, right_fp)

    def test_topology_kind_exposes_parameter_collision_for_shadow_audit(self):
        left, left_slots = graph("rz(pi*0.250000) q[0];\n", [5])
        right, right_slots = graph(
            "rz(pi*0.500000) q[0];\n", [5], guid_start=300
        )
        left_profile = build_wire_trace_profile(left, left_slots)
        right_profile = build_wire_trace_profile(right, right_slots)
        self.assertEqual(
            profile_fingerprint(left_profile, kind="topology"),
            profile_fingerprint(right_profile, kind="topology"),
        )
        self.assertNotEqual(
            profile_fingerprint(left_profile, kind="conservative"),
            profile_fingerprint(right_profile, kind="conservative"),
        )

    def test_noncontiguous_source_is_not_predicted(self):
        parent, parent_slots = graph(
            "rz(pi*0.250000) q[0];\nh q[0];\ncx q[0],q[1];\n",
            [5, 0, 6],
        )
        slots = tuple(parent_slots.values())
        profile = build_wire_trace_profile(parent, parent_slots)
        self.assertIsNone(
            successor_fingerprint(
                profile,
                parse_pattern("rz 0; cx 0 1;"),
                parse_pattern("cx 0 1; rz 0;"),
                (slots[0], slots[2]),
                xfer_id=38,
            )
        )

    def test_shadow_distinguishes_true_hits_from_collisions(self):
        audit = FingerprintAudit.create(mode="shadow", kind="conservative")
        fingerprint = b"same predicted successor"
        self.assertFalse(audit.should_skip(fingerprint))
        audit.observe_valid(fingerprint, b"exact-a")
        self.assertFalse(audit.should_skip(fingerprint))
        audit.observe_valid(fingerprint, b"exact-a")
        self.assertFalse(audit.should_skip(fingerprint))
        audit.observe_valid(fingerprint, b"exact-b")
        stats = audit.stats()
        self.assertEqual(stats["shadow_exact_duplicate_hits"], 1)
        self.assertEqual(stats["shadow_collision_hits"], 1)
        self.assertEqual(stats["shadow_precision"], 0.5)

    def test_filter_skips_only_after_a_valid_observation(self):
        audit = FingerprintAudit.create(mode="filter", kind="conservative")
        fingerprint = b"candidate"
        self.assertFalse(audit.should_skip(fingerprint))
        audit.observe_invalid(fingerprint)
        self.assertFalse(audit.should_skip(fingerprint))
        audit.observe_valid(fingerprint)
        self.assertTrue(audit.should_skip(fingerprint))
        self.assertEqual(audit.stats()["skipped_before_apply"], 1)

    def test_filter_can_preserve_two_collision_representatives(self):
        audit = FingerprintAudit.create(
            mode="filter", kind="parameter_transfer", representatives=2
        )
        fingerprint = b"parameter approximation"
        self.assertFalse(audit.should_skip(fingerprint))
        audit.observe_valid(fingerprint)
        self.assertFalse(audit.should_skip(fingerprint))
        audit.observe_valid(fingerprint)
        self.assertTrue(audit.should_skip(fingerprint))

    def test_direct_inverse_requires_mutual_unique_reverse_and_destination(self):
        parent = BeamState(
            graph=None,
            snapshot=None,
            guid_to_slot={},
            next_slot=0,
            last_touched={},
            rewrite_distance={},
            previous_preferred=set(),
            local_streak=0,
            gate_count=2,
            depth=1,
            history=((38, 4),),
            last_xfer_id=38,
            last_source_slots=(4, 5),
            last_destination_slots=(8, 9),
        )
        proposal = Proposal(0, 39, 8, (8, 9), 1.0, 2)
        inverse = [-1] * 40
        inverse[38] = 39
        inverse[39] = 38
        self.assertTrue(
            is_direct_inverse_proposal(parent, proposal, tuple(inverse))
        )
        self.assertFalse(
            is_direct_inverse_proposal(
                parent,
                Proposal(0, 39, 8, (8, 10), 1.0, 2),
                tuple(inverse),
            )
        )


if __name__ == "__main__":
    unittest.main()
