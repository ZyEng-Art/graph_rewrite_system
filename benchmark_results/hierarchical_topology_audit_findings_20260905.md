# Periodic Topology Audit with Per-Step Exact Apply

## Design

`refresh_interval=1` still applies every selected action immediately to the
current exact Quartz graph. Binding validation, graph hashing, exact cycle
detection, replay retention, and best-QASM capture also remain per-step.

The new `--topology-audit-interval` controls only the redundant whole-graph
comparison between the incremental indexed topology and the exact Quartz
topology. Its default is 1. With a larger value, the full node/edge comparison
is still forced when:

- the configured interval is reached;
- an episode terminates; or
- a state would become a new global best.

Thus an invalid Quartz rewrite is still rejected in the same step. The risk
introduced by a value larger than 1 is limited to temporarily using an
incorrect incremental topology if the local rewrite implementation has a bug;
the next forced audit stops that trajectory before it can become a best result.

## H100 A/B

The A/B used `gf2^6_mult`, actor `s1002`, seed 1004, 16 episodes, horizon 16,
and exact apply interval 1. Only the topology audit interval changed from 1
to 8.

| Metric | Audit every step | Audit every 8 steps | Change |
|---|---:|---:|---:|
| Exact refreshes | 258 | 258 | unchanged |
| Full topology audits | 258 | 37 | -85.7% |
| Skipped redundant audits | 0 | 221 | +221 |
| Refresh time | 0.4143 s | 0.2564 s | -38.1% |
| Total rollout time | 1.0294 s | 0.7970 s | -22.6% |
| Transitions/s | 250.6 | 323.7 | +29.2% |

Behavior was identical: 258 transitions, 256 accepted rewrites, 0 invalid
actions, 2 exact cycles, best/final gate count 485, and mean reward 8.3675.
Quartz apply time was unchanged at 0.121 seconds, confirming that exact graphs
were still advanced every step.

Artifact: `hierarchical_refresh_audit8_gf6_b16_s16_s1004.json`.

