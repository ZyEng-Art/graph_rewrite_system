# Exact-apply hot-path optimization (H100, 2026-09-06)

## Outcome

The model search from the untouched 58-gate `barenco_tof_3.qasm` now reaches
the same 38-gate QASM in 78.87 seconds. The prior implementation required
161.23 seconds, so this implementation-only change is a 2.04x speedup without
changing the checkpoint, calibration, candidate set, beam policy, or result.

On the same `h100-gpu1` host, CPU Quartz with its best measured thread setting
reaches that identical 38-gate QASM in 741.38 seconds. The optimized model path
therefore has a 9.40x end-to-end time-to-result speedup. Both searches start
from the original QASM; neither is initialized from a supplied optimized state.

This is a runtime optimization, not a search-policy claim. Both modes still
stop at 38 rather than the supplied 35-gate target under this beam policy.

## General changes

No circuit name, gate pattern, Barenco rule, or trajectory step is special
cased. The following changes apply to every circuit and rewrite:

1. **Deduplicate before child materialization.** Quartz produces the successor
   graph, and the collision-safe exact registry checks it immediately. Only a
   new circuit receives a Python snapshot, GUID/slot maps, graph delta,
   distance map, locality metadata, and extended history.
2. **Apply complete bindings directly.** A model candidate already contains
   the complete ordered source binding. When the patched Quartz API is loaded,
   source slots are translated to persistent node GUIDs and Quartz validates
   and applies that binding directly. This avoids anchor-based subgraph
   rematching. `--model-apply-binding anchor` preserves the old route for A/B;
   `auto` safely falls back to anchor rematching on an unpatched runtime.
3. **Select bounded top-k before Python object allocation.** Per-parent and
   global `heapq.nsmallest` selection runs on compact tuples. It retains the
   old stable ordering keys exactly, while avoiding creation and full sorting
   of all discarded `Proposal` objects.

The fastest path uses `exact_graph_key.patch`,
`direct_node_binding_apply.patch`, and `lazy_refresh_graph_binding.patch`.
Correct fallback behavior remains available without direct binding; the exact
identity implementation already has its slower QASM fallback.

## Controlled depth-8 decomposition

All stages use beam 1000, proposal factor 16, per-parent cap 128, maximum gate
increase 3, R99.9 model candidates, rotation elimination after every Quartz
apply, and native collision-safe exact identity. Runs were sequential on H100
GPU 4. The baseline uses the old Python workflow with the same native exact-key
runtime, isolating the hot-path changes from the earlier identity work.

| Input | Old workflow | Dedup before metadata, anchor | GUID direct apply | Final bounded top-k | Total speedup | Result |
|---|---:|---:|---:|---:|---:|---:|
| Barenco, 58 gates | 24.176 s | 18.909 s | 15.255 s | 14.180 s | 1.705x | 56 gates |
| GF, 495 gates | 98.863 s | 91.786 s | 81.632 s | 67.578 s | 1.463x | 485 gates |

For Barenco, old-to-final proposal time drops from 2.343 to 1.714 seconds
(1.37x), and apply/dedup/materialization drops from 16.130 to 6.864 seconds
(2.35x). For GF, proposal time drops from 17.709 to 3.610 seconds (4.91x), and
apply/dedup/materialization drops from 44.334 to 27.249 seconds (1.63x).
Matching time is effectively unchanged, as expected.

The output is invariant across all four stages: each Barenco run visits 7,116
unique circuits and writes the same best-QASM SHA-256
`658c2dcc80857b5d4edcd965c5bfffc63cc5cc37dd7dcbf245bf5479dba6eebb`;
each GF run visits 7,864 unique circuits and writes
`99771cdbc7d741becb3e089099d7d0f9815324f5944de9f313acebaf74adf3ea`.

The improvement generalizes differently according to circuit shape. Barenco
benefits most from avoiding metadata and anchor rematching because 83.48% of
validly applied depth-8 successors are duplicates. GF has fewer duplicates
(45.03%) but a much larger candidate list, so bounded top-k supplies most of
its gain. This is the intended evidence that the implementation is not tuned
to one case.

## End-to-end Barenco comparison

The shared protocol is beam 1000, proposal factor 16, per-parent cap 128,
maximum gate increase 3, rotation elimination after every apply, and exact
deduplication. Model search uses microbatch 512 and R99.9. CPU Quartz enumerates
the full legal match set with 32 OpenMP threads. Both ran sequentially on the
same host and stopped immediately after first reaching 38 gates.

| Mode | First 38 | Step | Exact match/model match | Proposal | Apply/dedup/materialization | Unique circuits |
|---|---:|---:|---:|---:|---:|---:|
| Prior model implementation | 161.23 s | 36 | 32.53 s | 13.72 s | 109.64 s | 34,544 through step 36 |
| Optimized model | 78.87 s | 36 | 27.23 s | 9.55 s | 36.83 s | 34,544 |
| Same-host CPU Quartz, OMP 32 | 741.38 s | 34 | 675.27 s | 6.81 s | 57.19 s | 33,116 |

The optimized model and CPU runs write the same byte-for-byte 38-gate QASM,
SHA-256
`89914672f7b83d10378b5283c6e9cf4bd6f9084505a4d37573118d5b63edb627`.
Thus the 9.40x number compares equal-quality, identical-output time to result.

Duplicate successors are still generated; exact deduplication prevents them
from entering the beam. The optimization makes rejection cheaper rather than
claiming to eliminate all commutation-equivalent action sequences. In the
optimized model run, 202,143 of 236,686 validly applied successors (85.41%) are
duplicates and skip all child metadata. In CPU Quartz the corresponding ratio
is 229,320 of 262,435 (87.38%). A future action-order canonicalization may
reduce generation further, but it must prove that it does not discard useful
dependent paths; it is intentionally not mixed into this exact runtime change.

The remaining optimized-model time is 36.83 seconds in apply/dedup/materialize,
27.23 seconds in learned matching, 9.55 seconds in proposal selection, and
about 5.27 seconds elsewhere. Apply and model matching are now the next general
bottlenecks; Python proposal construction is no longer dominant on GF.

## CPU thread selection

A same-host Barenco depth-3 sweep measured 106.65, 49.26, 35.69, 30.11, and
30.01 seconds at 1, 4, 8, 16, and 32 OpenMP threads. The long CPU comparison
therefore uses 32 threads, rather than an artificially slow single-thread
baseline. Exact matching accounts for 26.38 of the 30.01 seconds at OMP 32.

## Correctness checks

- The complete unit suite passes: 55/55 tests.
- Stable bounded top-k is checked against the former full stable sort,
  including ties and gate-increase filtering.
- Direct GUID apply is checked against the original anchor-rematch API after
  rotation elimination on every legal initial Barenco action: 248 actions
  spanning 49 xfers, all exact successor keys and gate counts equal.
- The same audit passes on 512 GF actions spanning 97 xfers.
- The staged end-to-end A/B produces byte-identical best QASM files on both
  circuits, and the long CPU/model comparison produces the same 38-gate QASM.

## Artifacts

- `exact_apply_hotpath_optimization_summary_20260906.json`
- `generalopt_{baseline_native,deferred_anchor,guid_native,topk_guid_native}_barenco_b1000_d8_r1.json`
- `generalopt_{baseline_native,deferred_anchor,guid_native,topk_guid_native}_gf2_6_b1000_d8_r1.json`
- Corresponding staged `_best.qasm` files.
- `generalopt_topk_guid_native_barenco_b1000_target38_r1.json` and `_best.qasm`.
- `generalopt_cpu_quartz_barenco_b1000_target38_omp32_r1.json` and `_best.qasm`.
- `generalopt_cpu_quartz_barenco_b1000_d3_omp{1,4,8,16,32}_r1.json`.
- `verify_direct_binding_apply.py` is the reusable integration audit.
