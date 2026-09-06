import unittest

from circuit_identity import (
    ExactGraphRegistry,
    QuartzHashRegistry,
    canonical_qasm_key,
)
from paged_rollout_benchmark import register_exact_graph_hash


class FakeGraph:
    def __init__(self, qasm: str, graph_hash: int = 31) -> None:
        self.qasm = qasm
        self.graph_hash = graph_hash
        self.hash_calls = 0
        self.qasm_calls = 0

    def hash(self) -> int:
        self.hash_calls += 1
        return self.graph_hash

    def to_qasm_str(self) -> str:
        self.qasm_calls += 1
        return self.qasm


class FakeNativeGraph:
    def __init__(self, key: bytes) -> None:
        self.key = key
        self.key_calls = 0

    def exact_key(self) -> bytes:
        self.key_calls += 1
        return self.key

    def to_qasm_str(self) -> str:
        raise AssertionError("native identity must bypass QASM serialization")


HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'


class ExactRefreshDedupTest(unittest.TestCase):
    def test_registers_unique_identity_once(self) -> None:
        seen = set()
        graph = FakeGraph(HEADER + "h q[0];\n")

        is_unique = register_exact_graph_hash(graph, seen)

        self.assertTrue(is_unique)
        self.assertEqual(len(seen), 1)
        self.assertEqual(graph.hash_calls, 0)
        self.assertEqual(graph.qasm_calls, 1)

    def test_rejects_same_circuit_with_independent_qasm_order(self) -> None:
        seen = set()
        first = FakeGraph(HEADER + "h q[0];\nx q[2];\n")
        reordered = FakeGraph(HEADER + "x q[2];\nh q[0];\n")

        self.assertTrue(register_exact_graph_hash(first, seen))
        self.assertFalse(register_exact_graph_hash(reordered, seen))
        self.assertEqual(len(seen), 1)

    def test_dependent_gate_order_remains_distinct(self) -> None:
        forward = canonical_qasm_key(HEADER + "h q[0];\nx q[0];\n")
        reversed_order = canonical_qasm_key(HEADER + "x q[0];\nh q[0];\n")

        self.assertNotEqual(forward, reversed_order)

    def test_same_quartz_hash_does_not_merge_distinct_wiring(self) -> None:
        seen = set()
        first = FakeGraph(HEADER + "cx q[0],q[1];\n", graph_hash=31)
        distinct = FakeGraph(HEADER + "cx q[0],q[2];\n", graph_hash=31)

        self.assertTrue(register_exact_graph_hash(first, seen))
        self.assertTrue(register_exact_graph_hash(distinct, seen))
        self.assertEqual(len(seen), 2)

    def test_rotation_parameters_are_part_of_identity(self) -> None:
        quarter = canonical_qasm_key(HEADER + "rz(pi*0.25) q[0];\n")
        half = canonical_qasm_key(HEADER + "rz(pi*0.5) q[0];\n")

        self.assertNotEqual(quarter, half)

    def test_registry_fast_paths_text_and_catches_reordering(self) -> None:
        first = FakeGraph(HEADER + "h q[0];\nx q[2];\n")
        exact_text = FakeGraph(HEADER + "h q[0];\nx q[2];\n")
        reordered = FakeGraph(HEADER + "x q[2];\nh q[0];\n")
        registry = ExactGraphRegistry.seeded(first)

        self.assertFalse(registry.register(exact_text))
        self.assertFalse(registry.register(reordered))
        self.assertEqual(
            registry.stats(),
            {
                "mode": "exact",
                "registrations": 3,
                "unique_identities": 1,
                "raw_serializations": 2,
                "raw_duplicates": 1,
                "reordered_duplicates": 1,
                "canonicalized_serializations": 2,
                "native_identity_calls": 0,
                "native_duplicates": 0,
            },
        )

    def test_registry_prefers_native_exact_key(self) -> None:
        first = FakeNativeGraph(b"same")
        duplicate = FakeNativeGraph(b"same")
        distinct = FakeNativeGraph(b"different")
        registry = ExactGraphRegistry.seeded(first)

        self.assertFalse(registry.register(duplicate))
        self.assertTrue(registry.register(distinct))
        self.assertEqual(len(registry), 2)
        self.assertEqual(registry.stats()["native_identity_calls"], 3)
        self.assertEqual(registry.stats()["native_duplicates"], 1)

    def test_legacy_quartz_hash_registry_exposes_false_merge(self) -> None:
        first = FakeGraph(HEADER + "cx q[0],q[1];\n", graph_hash=31)
        distinct = FakeGraph(HEADER + "cx q[0],q[2];\n", graph_hash=31)
        registry = QuartzHashRegistry.seeded(first)

        self.assertFalse(registry.register(distinct))
        self.assertEqual(registry.stats()["mode"], "quartz_hash_legacy_unsafe")


if __name__ == "__main__":
    unittest.main()
