# Transactional in-place rewrite and incremental metadata results

Date: 2026-09-08
Branch: `transactional-inplace-rewrite-20260908`

## Outcome

The search now rewrites a parent Quartz graph in place, computes the exact
physical-wire identity, and rolls the parent back before returning. Exact
duplicates and cyclic rewrites allocate no child graph. A novel successor is
copied exactly once, after the native exact-key registry accepts it.

The accepted transaction also exports its exact local node/edge delta. The
beam cache applies that delta incrementally instead of enumerating
`graph.nodes` and `graph.all_edges()` for every new child. Logical-qubit
positions inside Quartz are maintained incrementally as replacement edges are
inserted; the propagation stops when it reaches an unchanged position.

## Correctness and rollback audit

`verify_transactional_apply.py` compared the old copy-first implementation
with the new transaction while rotation elimination was enabled.

| Circuit | Audited legal actions | Distinct xfers | Duplicate replays | Continuous transaction depth |
|---|---:|---:|---:|---:|
| `barenco_tof_3.qasm` | 248 | 49 | 248 | 24 |
| `gf2^6_mult.qasm` | 256 | 97 | 256 | 24 |

For every audited action:

- the old and new successors had byte-identical native exact keys and equal
  gate counts;
- applying the same action a second time returned exact-duplicate status and
  performed zero graph copies;
- the parent graph was identical before and after the call in exact key, QASM,
  Quartz hash, gate count, node GUID/type rows, and edge rows;
- applying the exported node/edge delta to the parent reconstructed the full
  materialized child exactly.

The undo invariant is local and deterministic: the first write to an affected
`inEdges`, `outEdges`, constant-parameter, or logical-position entry records
both its previous value and whether it existed. All transactional add/remove/
constant-fold operations use those instrumented mutation functions. Rollback
then restores each recorded entry or erases entries that were absent before
the transaction. The RAII destructor repeats rollback only if an early return
or exception has not already done so.

## H100 performance

All runs used `beam=1000`, the state-only GPU matcher, direct GUID bindings,
exact identity, and rotation elimination. Times are the benchmark's measured
search time and exclude process startup/ECC loading.

### Raw dedup pressure (`depth=3`, prefilters off)

| Circuit | Baseline total | Transaction + incremental | Total speedup | Baseline apply | New apply | Apply speedup | Exact duplicates / valid applies | Graph copies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| barenco | 2.189 s | 1.639 s | **1.335×** | 1.949 s | 1.421 s | **1.372×** | 8,841 / 10,956 (80.7%) | 2,115 |
| GF(2^6) | 10.580 s | 7.909 s | **1.338×** | 9.395 s | 6.749 s | **1.392×** | 3,306 / 5,771 (57.3%) | 2,465 |

The native graph-copy counter equals the number of novel children exactly.
No exact duplicate is copied. Cyclic rewrites are also rolled back without a
copy (171 barenco and 184 GF rewrites in these runs).

### Current hybrid pipeline

This is the existing native conservative fingerprint filter plus learned
defer, so many easy repeats have already been removed before Quartz. The
remaining gain therefore measures what the transaction adds to the actual
pipeline rather than an intentionally unfiltered workload.

| Circuit/run | Baseline total | New total | Total speedup | Baseline apply | New apply | Apply speedup | Novel children | Best gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| barenco, depth 16 | 13.020 s | 11.382 s | **1.144×** | 10.267 s | 8.318 s | **1.234×** | 15,115 | 48 |
| GF(2^6), depth 8 | 29.502 s | 23.529 s | **1.254×** | 22.782 s | 16.892 s | **1.349×** | 7,465 | 485 |

The unique-identity counts and best exact identities match their baselines.
The final barenco beam digest is also identical. The GF best digest is
identical, but the final beam digest differs between the two independent GPU
inference runs; the action/duplicate/novel counts are identical, so this A/B
does not claim bit-for-bit final-beam determinism for GF.

Incremental child metadata reduced its own accumulated time from 2.393 s to
1.365 s on barenco (1.75×) and from 8.661 s to 4.270 s on GF (2.03×). Exporting
all native deltas cost only 0.038 s and 0.031 s respectively.

## Updated bottleneck

The transaction eliminated full graph copies and full child graph walks for
duplicates, but it deliberately retains exact work. In the hybrid GF run the
largest native stages are:

| Stage | GF time | barenco time |
|---|---:|---:|
| Exact wire key | 3.223 s | 1.303 s |
| Cycle check | 2.363 s | 0.933 s |
| Rotation/constant elimination | 2.265 s | 0.871 s |
| Novel child clone | 1.014 s | 0.268 s |
| Transaction rewrite | 0.189 s | 0.563 s |
| Rollback | 0.038 s | 0.117 s |

The next substantial apply-side gain would require an exact incrementally
maintained wire identity and a local cycle/rotation analysis. The rollback
itself is no longer a material hotspot. Those changes should be audited
separately because they replace exact full-graph checks rather than merely
moving them before allocation.

## Reproduction

Apply `quartz_patches/transactional_inplace_rewrite.patch` after the Quartz
patch set on `detailed-apply-profiling-20260907`, rebuild the C++ runtime and
Cython extension, then use:

```bash
./run_transactional_apply_ab.sh 3 barenco_tof_3.qasm 3 on result_tag
./run_detailed_apply_profile.sh 5 barenco_tof_3.qasm 16 result_tag on
```

`--transactional-apply off` retains the previous copy-first path for A/B.
`--transactional-apply on` requires exact identity and the patched direct-GUID
Quartz API. The implementation is safe for the current synchronous apply loop;
parallel calls that mutate the same parent graph would require per-parent
serialization or private graph ownership.

## Tests

- Quartz C++ runtime and Cython extension: built successfully.
- Focused registry/search tests: 20 passed.
- All top-level project tests with CUDA hidden: 97 passed, 1 existing warning.
- The GPU-visible full test run had the same 95 passing tests plus two Triton
  launcher failures because that Python 3.10 environment lacks `Python.h`;
  rerunning those tests on their CPU fallback passed.

Raw measurements and the correctness audit are stored in
`benchmark_results/transactional_*_20260908.json` and the detailed profile
files committed with this branch.
