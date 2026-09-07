# State-only exact-rewrite search: implementation and benchmark

Date: 2026-09-07

Machine: `h100-gpu3`, one H100 GPU per run

Implementation branch: `state-only-exact-rewrite-20260907`

Unified parent branch: `unified-graph-rewrite-20260907`

## Outcome

The model search path no longer serializes an action prefix as neural-network
input.  Every search layer starts from the exact Quartz successor graph,
predicts source bindings/actions for that graph, ranks retained actions on the
GPU, and then uses Quartz to apply each selected full binding exactly.  Exact
graph identity is used for successor deduplication before child metadata is
materialized.

The `BeamState.history` list remains only as an output/debug trajectory.  It is
not collated into tensors and is never consumed by `encode_current_graph`.

At depth 8, where both implementations follow the same optimization result,
the new path is 1.81x faster end to end on Barenco and 2.27x faster on GF.  At
depth 128, it reaches the identical 38-gate Barenco circuit at the identical
search step in 47.44 seconds instead of 88.07 seconds.  It completes the whole
Barenco run in 203.81 seconds instead of 374.77 seconds.

For GF, fixed-depth behavior is not bitwise trajectory-equivalent: the new path
finishes depth 128 at 475 gates while the archived current path reaches 474.
The source-binding formulas are equivalent, but H100 BF16 parallel graph
reductions can move scores immediately around the calibrated threshold.  The
fixed-depth speed number (3.36x) must therefore not be presented as an
equal-quality comparison.  A separate target-474 run reached 473 gates at
depth 133 in 579.42 seconds, giving a conservative equal-or-better-quality
speedup of 3.22x over the archived current path's 474-gate result.

## Branch consolidation

No historical branch was deleted.  `unified-graph-rewrite-20260907` was made
from the more recent refresh/model line and merges the independent Barenco
trajectory/rewrite line.  The following tips are all ancestors of its merge
commit `3562560`:

- `main` (`49a9097`)
- `barenco-trajectory-repro-20260905` (`73bc5e8`)
- `barenco-trajectory-repro-clean-20260905` (`c6bba2d`)
- `refresh-consistency-model-20260906` (`c6b48ad`)

The three merge conflicts were in matcher-generalization audit files.  The
refresh/model versions were retained because they contain the functional and
reporting superset.  Both `unified-graph-rewrite-20260907` and the implementation
branch are pushed to `origin`.

## Runtime pipeline

For each beam layer:

1. `collate_exact_states` compacts live persistent Quartz slots into a dense
   batch containing current gate types, current edges, edge relations, and
   current locality features only.
2. `PagedActionBindingModel.encode_current_graph` applies the gate embedding,
   current-graph message passing, and readout graph layers.  It does not read
   initial-graph, action-xfer, action-source, binding-slot, or destination-slot
   fields.
3. Source logits and structural binding decoding produce GPU candidate tensors.
   The dense slots are mapped back to their exact persistent Quartz slot IDs on
   the GPU.
4. Static source-to-xfer metadata expands a binding into every compatible
   rewrite action.  Per-parent Top-K and global Top-K are both performed on the
   GPU.  Only selected complete actions cross to the CPU.
5. Quartz applies the selected `(xfer, source, complete binding)` exactly.  The
   successor is assigned an exact graph key, duplicates are rejected, and only
   distinct children enter the next layer.
6. The next layer starts by encoding those exact successor graphs from scratch;
   it never reconstructs topology by replaying a predicted action sequence.

`--model-pipeline compat_host` preserves the prior exact-rebase/zero-action
model input and Python proposal path for A/B measurements.
`--model-pipeline state_only_gpu` is now the default.  The latter deliberately
requires `--refresh-interval 0`: every selected rewrite is already exact, while
periodically adding CPU-matched candidates would change the candidate source
and invalidate the matcher-throughput boundary.

## Controlled depth-8 A/B

Both sides used raw QASM, beam 1000, depth 8, microbatch 512, target recall
99.9%, at most 10240 source matches, at most 128 actions per parent, proposal
factor 16, maximum gate increase 3, rotation elimination, direct complete
binding, and exact graph deduplication.

| Circuit | Pipeline | Best gates | Unique graphs | Total (s) | Match states/s | Accepted actions/s |
|---|---|---:|---:|---:|---:|---:|
| Barenco | `compat_host` | 55 | 7,116 | 19.1725 | 795.33 | 371.11 |
| Barenco | `state_only_gpu` | 55 | 7,116 | 10.5652 | 4,258.40 | 673.44 |
| GF | `compat_host` | 485 | 7,886 | 84.2099 | 162.37 | 93.64 |
| GF | `state_only_gpu` | 485 | 7,886 | 37.0165 | 1,163.21 | 213.01 |

| Circuit | End-to-end speedup | Model/match speedup | Proposal speedup | Best-trace equality | Best-QASM equality |
|---|---:|---:|---:|---|---|
| Barenco | 1.8147x | 5.3543x | 8.3709x | yes | yes |
| GF | 2.2749x | 7.1641x | 15.0478x | yes | yes |

The complete best-gate traces agree at every step.  The paired best QASM SHA256
values are:

- Barenco: `f96cfb697f93ccad71087e64a45cecfe46bb79664f90406e02ebdb18f51054f0`
- GF: `99771cdbc7d741becb3e089099d7d0f9815324f5944de9f313acebaf74adf3ea`

## Depth-128 end-to-end comparison

| Circuit | Pipeline | Best gates | First-best step | First-best (s) | Total (s) | Match states/s |
|---|---|---:|---:|---:|---:|---:|
| Barenco | archived current | 38 | 34 | 88.0700 | 374.7722 | 798.28 |
| Barenco | `state_only_gpu` | 38 | 34 | 47.4415 | 203.8111 | 7,399.11 |
| GF | archived current | 474 | 128 | 1,865.9686 | 1,865.9716 | 119.35 |
| GF | `state_only_gpu` | 475 | 128 | 555.5763 | 555.5779 | 1,150.73 |

Barenco is an equal-search-step and equal-circuit comparison:

- 1.8388x faster for the complete 128-layer run;
- 1.8564x faster to the 38-gate result;
- 9.2688x higher matching throughput;
- 1.8987x higher accepted-successor throughput;
- the two 38-gate QASM files have SHA256
  `89914672f7b83d10378b5283c6e9cf4bd6f9084505a4d37573118d5b63edb627`.

GF at fixed depth is 3.3586x faster and has 9.6414x higher matching
throughput, but it is one gate worse, so this is a throughput/coverage result
rather than an equal-quality speedup.

### GF equal-or-better-quality run

The separate state-only run used the same settings, increased the depth ceiling
to 160, and stopped as soon as `--target-gate-count 474` was satisfied.  The
winning rewrite reduced 475 directly to 473 gates at step 133, so there is no
separately observed 474-gate state in this run.

| Metric | Archived current | `state_only_gpu` target run | Ratio/result |
|---|---:|---:|---:|
| Best gates | 474 | 473 | state-only is one gate better |
| Step | 128 | 133 | +5 search layers |
| Time to result (s) | 1,865.9686 | 579.4202 | 3.2204x faster |
| Total process time (s) | 1,865.9716 | 579.4219 | 3.2204x faster |
| Match states/s | 119.35 | 1,146.22 | 9.6035x |
| Accepted actions/s | 68.54 | 229.34 | 3.3463x |
| Unique graphs | 127,886 | 132,886 | +5,000 |

The 473-gate QASM SHA256 is
`85a66226b70d58b77919b5a8bdd4468862c2f0ef2c09c4bd20f6b21d5fd3b815`.
This is the fairest end-to-end result: the new implementation does five more
beam layers and still returns a strictly smaller circuit in less than one third
of the old time.

## Where time goes now

| Circuit/depth | Pipeline | Model (s) | Proposal (s) | Quartz apply (s) | Apply share of new total |
|---|---|---:|---:|---:|---:|
| Barenco/8 | `compat_host` | 7.6899 | 1.8865 | 8.7259 | - |
| Barenco/8 | `state_only_gpu` | 1.4362 | 0.2254 | 8.7223 | 82.56% |
| GF/8 | `compat_host` | 42.4099 | 4.3557 | 30.0566 | - |
| GF/8 | `state_only_gpu` | 5.9198 | 0.2895 | 29.7596 | 80.40% |
| Barenco/128 | archived current | 147.4438 | 34.4036 | 175.7253 | - |
| Barenco/128 | `state_only_gpu` | 16.4458 | 4.3561 | 179.4775 | 88.06% |
| GF/128 | archived current | 1,063.1107 | 127.0732 | 417.8778 | - |
| GF/128 | `state_only_gpu` | 110.2657 | 6.7691 | 411.8379 | 74.13% |
| GF/target 473 | `state_only_gpu` | 115.0619 | 7.0760 | 429.4513 | 74.12% |

Model matching is now roughly 9-10x faster in the long runs, and proposal
selection is roughly 8-19x faster.  Exact Quartz apply time is intentionally
almost unchanged.  It is now the dominant cost, so further large end-to-end
gains require batching/caching/parallelizing exact rewrites or reducing how
many low-value selected proposals reach Quartz without sacrificing coverage.

## Search-quality interpretation

This change removes action-prefix encoding and its topology-maintenance burden;
it does not change the learned ranking objective.  Therefore it does not make
Barenco reach 36 gates: both long Barenco paths reach the same 38-gate result at
step 34, and the known reference route is still lost by gate-first proposal
ranking before the later profitable continuation.  That is a search-depth/
ranking issue, not an inability of the new path to apply the exact binding.

The short deterministic-equivalence tests use identical formulas.  On the
actual BF16 H100 checkpoint, repeated parallel `index_add` reductions are not
bitwise deterministic.  Scores very close to the 99.9%-recall threshold can
therefore include or exclude a few boundary candidates between processes.  The
identical depth-8 traces/QASM and identical Barenco depth-128 trace/QASM show
that the remapping and exact rewrite semantics are correct; the one-gate GF
fixed-depth difference is explicitly retained rather than hidden.

## Validation

The following passed against the patched Quartz build:

- `test_benchmark_matcher_throughput.py`: 6 tests, including proof that the
  state-only batch contains no initial/action/binding/destination history fields;
- `test_gpu_proposals.py`: GPU and CPU, including equality with legacy proposal
  expansion/ranking semantics;
- `test_paged_model.py`: including current-graph encoding equivalence to exact
  rebase plus a zero-length action prefix;
- `test_exact_refresh_dedup.py`: 9 tests;
- `test_tensorized_batch.py`;
- `test_batched_ppo_collector.py`;
- `test_ppo_policy_padding.py`.

The raw JSON files retain all step records, candidate counts, duplicate counts,
stage timings, exact-dedup metadata, and improvement traces.
