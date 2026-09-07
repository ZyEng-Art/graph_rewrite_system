from __future__ import annotations

from types import SimpleNamespace

from train_paged_ppo import topology_audit_required


def main() -> None:
    best = {"circuit": {"gate_count": 100}}
    runtime = SimpleNamespace(
        circuit="circuit",
        stopped=False,
        state=SimpleNamespace(depth=3, gate_count=100),
    )
    assert topology_audit_required(runtime, best, 1)
    assert not topology_audit_required(runtime, best, 8)

    runtime.state.depth = 8
    assert topology_audit_required(runtime, best, 8)
    runtime.state.depth = 3
    runtime.state.gate_count = 99
    assert topology_audit_required(runtime, best, 8)
    runtime.state.gate_count = 100
    runtime.stopped = True
    assert topology_audit_required(runtime, best, 8)
    print("topology audits are forced at intervals, termination, and new bests")


if __name__ == "__main__":
    main()
