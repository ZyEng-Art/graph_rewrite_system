import unittest

from paged_rollout_benchmark import register_exact_graph_hash


class FakeGraph:
    def __init__(self, graph_hash: int) -> None:
        self.graph_hash = graph_hash
        self.hash_calls = 0

    def hash(self) -> int:
        self.hash_calls += 1
        return self.graph_hash


class ExactRefreshDedupTest(unittest.TestCase):
    def test_registers_unique_hash_once(self) -> None:
        seen = {10}
        graph = FakeGraph(20)

        is_unique = register_exact_graph_hash(graph, seen)

        self.assertTrue(is_unique)
        self.assertEqual(seen, {10, 20})
        self.assertEqual(graph.hash_calls, 1)

    def test_rejects_duplicate_without_mutating_seen(self) -> None:
        seen = {10, 20}
        graph = FakeGraph(20)

        is_unique = register_exact_graph_hash(graph, seen)

        self.assertFalse(is_unique)
        self.assertEqual(seen, {10, 20})
        self.assertEqual(graph.hash_calls, 1)

    def test_distinct_graph_objects_with_same_hash_are_duplicates(self) -> None:
        seen: set[int] = set()

        first_unique = register_exact_graph_hash(FakeGraph(31), seen)
        second_unique = register_exact_graph_hash(FakeGraph(31), seen)

        self.assertTrue(first_unique)
        self.assertFalse(second_unique)
        self.assertEqual(seen, {31})


if __name__ == "__main__":
    unittest.main()
