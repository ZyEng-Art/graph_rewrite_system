# Strict Refresh Breakdown and Optimization

## Protocol

- Circuit: `gf2^6_mult`, 495 gates.
- Actor: shared multi-circuit PPO checkpoint `s1002`.
- H100 physical GPU 6; 16 episodes; horizon 16; seed 1004.
- Strict refresh interval 1, so every committed speculative action is checked
  against Quartz before the next action.
- The before and after runs produced the same 258 transitions, 256 accepted
  rewrites, 0 invalid actions, 2 cycles, best gate count 485, and mean reward
  8.3675.

## Baseline Breakdown

The baseline spent 0.4769 seconds in refresh out of 1.1736 seconds total.

| Refresh component | Seconds | Refresh fraction |
|---|---:|---:|
| Exact replay total | 0.3131 | 65.7% |
| Result/archive processing | 0.1576 | 33.1% |
| Full topology signature and equality | 0.1184 | 24.8% |
| Quartz apply | 0.1140 | 23.9% |
| Slot update and graph swap | 0.0665 | 13.9% |
| Checkpoint map copy | 0.0054 | 1.1% |
| Prefix analysis | 0.0019 | 0.4% |
| Source GUID lookup | 0.0005 | 0.1% |

The rows under exact replay are subcomponents and therefore overlap with the
exact replay total. Source-node lookup is negligible; the expensive work is
whole-graph validation, Quartz graph construction/hash, and eager replay
archive serialization.

## Changes

1. When rotation elimination is disabled, destination GUIDs returned by the
   exact Quartz rewrite are already live. Refresh now updates only those GUIDs
   instead of scanning every node in the new graph.
2. A replay state retains the immutable Quartz graph and defers `to_qasm_str()`
   until that state is sampled or a serializable checkpoint is requested.
3. The graph hash already computed for exact-cycle detection is passed to the
   replay reservoir instead of being recomputed.
4. Refresh now exports grouping, prefix-cache, Quartz apply, binding, slot,
   topology, graph-hash, archive, and best-QASM timings and operation counts.

Checkpoint serialization converts every retained graph to a QASM string and
omits the live graph object. A real checkpoint smoke test loaded 8 serialized
states and verified that every state contained a QASM string and no graph
object.

## A/B Result

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Refresh time | 0.4769 s | 0.4143 s | -13.1% |
| Total rollout time | 1.1736 s | 1.0294 s | -12.3% |
| Transitions/s | 219.8 | 250.6 | +14.0% |
| Slot update/graph swap | 0.0665 s | 0.0016 s | -97.6% |
| Result/archive processing | 0.1576 s | 0.0869 s | -44.8% |

The remaining refresh floor is dominated by three full-graph C++/Python
operations: topology signature construction, `PyGraph.hash()`, and Quartz
apply. Removing those requires a Quartz API that carries an incremental hash
and exact topology delta out of `apply_xfer_with_guid_binding`; a neural
replacement would not preserve the exact legality/equivalence check.

## Artifacts

- Baseline: `hierarchical_refresh_profile_gf6_b16_s16_s1004.json`.
- Optimized: `hierarchical_refresh_optimized_final_r2_gf6_b16_s16_s1004.json`.

