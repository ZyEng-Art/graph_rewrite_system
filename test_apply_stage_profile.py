from collections import Counter
from dataclasses import dataclass

from beam_search_benchmark import rendered_apply_profile, snapshot


@dataclass(frozen=True)
class _Node:
    guid: int
    gate_tp: int


class _Graph:
    def __init__(self) -> None:
        self.nodes = [_Node(10, 2), _Node(11, 3)]

    def all_edges(self):
        return [(0, 1, 2, 1)]


def test_snapshot_profiling_preserves_snapshot_and_records_substages() -> None:
    profile = Counter()

    actual = snapshot(_Graph(), {10: 7, 11: 8}, profile=profile)

    assert actual == {
        "nodes": [(7, 2, 10), (8, 3, 11)],
        "edges": [(7, 8, 2, 1)],
    }
    assert {
        "child_snapshot_nodes_ns",
        "child_snapshot_node_rows_ns",
        "child_snapshot_native_edges_ns",
        "child_snapshot_edge_rows_ns",
    } <= profile.keys()


def test_rendered_apply_profile_separates_seconds_and_counts() -> None:
    rendered = rendered_apply_profile(
        Counter(
            {
                "native_graph_rewrite_ns": 1_500_000_000,
                "native_result_success_count": 17,
            }
        )
    )

    assert rendered == {
        "seconds": {"native_graph_rewrite": 1.5},
        "counts": {"native_result_success": 17},
    }
