from __future__ import annotations

from pathlib import Path

from benchmark_hierarchical_rollout import (
    allocate_circuit_episodes,
    merge_collector_timing,
)


def main() -> None:
    circuits = [Path("a.qasm"), Path("b.qasm"), Path("c.qasm")]
    first = allocate_circuit_episodes(circuits, 8)
    assert [count for _, count in first] == [3, 3, 2]
    rotated = allocate_circuit_episodes(circuits, 8, offset=2)
    assert [count for _, count in rotated] == [3, 2, 3]
    sparse = allocate_circuit_episodes(circuits, 2, offset=1)
    assert [count for _, count in sparse] == [0, 1, 1]

    timing = {"total_seconds": 1.5, "refresh_seconds": 0.5}
    merge_collector_timing(
        timing,
        {"total_seconds": 2.0, "refresh_seconds": 0.25, "actions": 3},
    )
    assert timing == {
        "total_seconds": 3.5,
        "refresh_seconds": 0.75,
        "actions": 3.0,
    }
    print("multi-circuit episode allocation and timing aggregation are correct")


if __name__ == "__main__":
    main()
