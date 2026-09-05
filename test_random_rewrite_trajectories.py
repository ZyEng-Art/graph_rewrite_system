import random
import unittest

from collect_random_rewrite_trajectories import candidate_order, random_qasm


class RandomCircuitTest(unittest.TestCase):
    def test_random_qasm_emits_exact_gate_count(self) -> None:
        qasm = random_qasm(random.Random(41), qubits=5, gates=53)
        lines = qasm.strip().splitlines()
        self.assertEqual(lines[:3], [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "qreg q[5];",
        ])
        self.assertEqual(len(lines[3:]), 53)
        self.assertTrue(all(line.endswith(";") for line in lines))

    def test_candidate_order_can_force_local_rows_first(self) -> None:
        matches = [
            {"anchor_slot": 2, "xfer_ids": [7, 8]},
            {"anchor_slot": 9, "xfer_ids": [10]},
        ]
        rows = candidate_order(
            matches,
            previous_preferred={9},
            local_probability=1.0,
            rng=random.Random(42),
        )
        self.assertEqual(int(rows[0][0]["anchor_slot"]), 9)
        self.assertEqual({xfer for _, xfer in rows}, {7, 8, 10})


if __name__ == "__main__":
    unittest.main()
