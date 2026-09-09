from __future__ import annotations

import unittest

from search_feedback import SearchFeedbackRegistry, identity_order_key


class SearchFeedbackRegistryTest(unittest.TestCase):
    def test_identity_order_is_stable(self) -> None:
        identity = ("exact", b"abc")
        self.assertEqual(identity_order_key(identity), identity_order_key(identity))
        self.assertNotEqual(
            identity_order_key(identity), identity_order_key(("exact", b"abd"))
        )

    def test_descendant_improvement_propagates_to_all_ancestors(self) -> None:
        registry = SearchFeedbackRegistry(("root", b"0"), 10)
        child = registry.add_node(
            ("child", b"1"), gate_count=12, depth=1, parent_id=0
        )
        grandchild = registry.add_node(
            ("grandchild", b"2"), gate_count=8, depth=2, parent_id=child
        )
        self.assertEqual(registry.nodes[grandchild].best_descendant_gate, 8)
        self.assertEqual(registry.nodes[child].best_descendant_gate, 8)
        self.assertEqual(registry.nodes[0].best_descendant_gate, 8)
        self.assertEqual(registry.nodes[0].descendant_gain, 2)

    def test_exact_duplicate_adds_a_transposition_parent(self) -> None:
        registry = SearchFeedbackRegistry(("root", b"0"), 10)
        left = registry.add_node(
            ("left", b"1"), gate_count=10, depth=1, parent_id=0
        )
        right = registry.add_node(
            ("right", b"2"), gate_count=10, depth=1, parent_id=0
        )
        shared_identity = ("shared", b"3")
        shared = registry.add_node(
            shared_identity, gate_count=8, depth=2, parent_id=left
        )
        registry.add_parent_edge(shared_identity, right, step=3)
        self.assertEqual(registry.nodes[shared].parent_ids, {left, right})
        self.assertEqual(registry.nodes[right].best_descendant_gate, 8)

    def test_expansion_outcomes_accumulate(self) -> None:
        registry = SearchFeedbackRegistry(("root", b"0"), 10)
        registry.observe_expansion(
            0,
            attempted=10,
            valid=8,
            unique=3,
            duplicate=5,
            invalid=2,
            improving=1,
            best_child_gate=9,
            step=1,
        )
        row = registry.nodes[0]
        self.assertEqual(row.observed_expansions, 1)
        self.assertAlmostEqual(row.novel_yield, 3 / 8)
        self.assertAlmostEqual(row.valid_yield, 0.8)
        self.assertEqual(row.best_descendant_gate, 9)


if __name__ == "__main__":
    unittest.main()
