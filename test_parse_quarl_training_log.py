from __future__ import annotations

import tempfile
from pathlib import Path

from parse_quarl_training_log import parse_quarl_log


def main() -> None:
    log = """
  Data for iter 577 collected in 5.25 s .
    demo_best_graph_gate_count : 58
    demo_max_epslen_global : 5
    num_exps : 1280
  Timing: rollout 5.25s, learn 1.00s, iter 6.30s, rollout/iter 0.833
Agent 0 : demo: 58 -> 56 ! Seq saved to ignored .
  Data for iter 578 collected in 4.0 s .
    demo_best_graph_gate_count : 56
    demo_max_epslen_global : 12
    num_exps : 1920
  Timing: rollout 4.00s, learn 2.00s, iter 6.00s, rollout/iter 0.667
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "run.log"
        path.write_text(log, encoding="utf-8")
        result = parse_quarl_log(path)

    assert result["completed_iterations"] == 2
    assert result["partial_iterations"] == 0
    assert result["improvements"] == [
        {
            "line": 7,
            "agent": 0,
            "circuit": "demo",
            "previous_gate_count": 58,
            "new_gate_count": 56,
            "iteration": 578,
        }
    ]
    aggregate = result["aggregate"]
    assert aggregate["initial_best_before_rollout_by_circuit"] == {"demo": 58}
    assert aggregate["initial_logged_best_gate_count_by_circuit"] == {"demo": 58}
    assert aggregate["final_best_gate_count_by_circuit"] == {"demo": 56}
    assert aggregate["total_transitions"] == 3200
    assert aggregate["total_rollout_seconds"] == 9.25
    assert aggregate["transition_throughput_per_second"] == 3200 / 9.25


if __name__ == "__main__":
    main()
