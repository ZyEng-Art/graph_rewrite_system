# Native successor-fingerprint adaptation (2026-09-07)

## Outcome

The general successor fingerprint is now implemented inside Quartz C++ and
exposed through both single-candidate and batched Cython calls.  The search
uses the native implementation automatically when the patched Quartz module
is loaded and retains the previous Python implementation as an explicit
reference/fallback backend.

On one idle H100, the native `xfer_guarded` filter turned the previous Python
prototype's apply reduction into an end-to-end speedup:

| circuit and search | mode | Quartz applies | fingerprint time | total time | best |
|---|---|---:|---:|---:|---:|
| Barenco, beam 1000, depth 64 | off | 629,277 | 0 s | 89.861 s | 38 at depth 34 |
| Barenco, beam 1000, depth 64 | native filter | 402,290 | 6.894 s | 68.299 s | 38 at depth 34 |
| GF, beam 1000, depth 8 | off | 15,668 | 0 s | 36.112 s | 485 |
| GF, beam 1000, depth 8 | native filter | 10,897 | 2.126 s | 32.531 s | 485 |

The adjacent Barenco comparison is 24.0% faster end to end and avoids
227,028 applies, or 36.08% of the baseline.  The adjacent GF comparison is
9.9% faster and avoids 4,771 applies, or 30.45% of the baseline.  Absolute
times on this shared machine vary between runs, so the apply counts and the
paired run order should be retained with timing comparisons.

## Safety audit

A full Barenco depth-64 `shadow` run computed fingerprints but did not filter
anything.  Every proposal still went through Quartz and its resulting graph
was compared using the collision-safe physical-wire identity:

- 516,470 representable candidates;
- 226,949 fingerprint hits;
- 226,893 hits returning an already-seen exact circuit;
- 56 hits on actions that Quartz rejected as invalid;
- zero hits joining two distinct valid successor circuits;
- 112,697 unsupported candidates falling back to normal Quartz apply.

The native shadow fingerprint work took 7.584 seconds, versus 26.889 seconds
for the Python reference artifact, a 3.55x reduction.  In the shorter current
Barenco depth-8 comparison, native batching took 0.504 seconds versus 2.218
seconds for Python, a 4.40x reduction.  The batched and unbatched native runs
had the same final exact-beam digest.

`xfer_guarded` remains an audited probabilistic filter rather than a formal
symbolic-equivalence proof: it preserves exact concrete parameters and adds
the rewrite id when a destination produces parameters whose expression is not
available in the compact ECC string.  A 128-bit key also has a theoretical
collision probability.  Unsupported structure returns `None` and follows the
authoritative Quartz path.  `shadow` should be rerun when the gate set, ECC
set, or parameter representation changes.

## Implementation

`quartz_patches/native_successor_fingerprint.patch` adds
`SuccessorFingerprintProfile` to Quartz.  One profile is built directly from
the exact graph and stores, entirely in C++:

- physical-wire operation order;
- persistent action slots and their positions on each wire;
- gate type, physical operands and exact parameter bit patterns;
- exact and parameter-blind polynomial prefix hashes.

For each `(xfer, ordered source binding)` candidate, Quartz validates the
source wiring, constructs the destination wiring from `GraphXfer`, verifies
that the removed operations are contiguous on every affected wire, and
splices prefix/destination/suffix hashes.  It returns only two `uint64_t`
words.  It neither copies the graph nor applies the rewrite.

The native patch is an incremental patch against the repository's existing
Quartz patch stack.  Apply `exact_graph_key.patch`,
`direct_node_binding_apply.patch` and `wire_trace_profile.patch` first; the
tested remote build also contains `lazy_refresh_graph_binding.patch`.

The Cython wrapper exposes:

- `PyGraph.successor_fingerprint_profile(slot_guid_rows)`;
- `PySuccessorFingerprintProfile.successor_fingerprint(...)`;
- `PySuccessorFingerprintProfile.successor_fingerprints(...)`.

`beam_search_benchmark.py` adds:

- `--preapply-fingerprint-backend auto|python|native`;
- `--preapply-fingerprint-native-batch-size N` (default 512, zero for the
  single-candidate native path).

The batched path stages up to 512 proposal positions, groups them by parent,
and crosses the Python/Cython boundary once per parent group.  Profile
construction is still lazy and only reaches at most one staged block beyond
the point where the beam fills.

Recommended experimental invocation:

```text
--preapply-fingerprint filter \
--preapply-fingerprint-kind xfer_guarded \
--preapply-fingerprint-backend auto \
--preapply-fingerprint-native-batch-size 512
```

Use `--preapply-fingerprint-backend native` in benchmarks that must fail
instead of silently falling back to Python when the Quartz patch is missing.

## Reproduction artifacts

- `benchmark_results/native_fp_batch512_shadow_barenco_b1000_d64_20260907.json`
- `benchmark_results/native_fp_batch512_filter_barenco_b1000_d64_20260907.json`
- `benchmark_results/native_fp_current_off_barenco_b1000_d64_20260907.json`
- `benchmark_results/native_fp_batch512_shadow_barenco_b1000_d8_20260907.json`
- `benchmark_results/native_fp_shadow_barenco_b1000_d8_20260907.json`
- `benchmark_results/python_fp_shadow_current_barenco_b1000_d8_20260907.json`
- `benchmark_results/native_fp_batch512_filter_gf_b1000_d8_20260907.json`
- `benchmark_results/native_fp_current_off_gf_b1000_d8_20260907.json`

## Remaining bottleneck

The native depth-64 run still built roughly 51,000 parent profiles.  The
batch interface removes most call overhead, but profile construction is
linear in the full parent graph.  The next general optimization is to store a
profile in every accepted `BeamState` and derive the child profile by applying
the already-validated affected-wire splice after Quartz accepts the rewrite.
That would replace repeated full-profile construction with work proportional
to the rewrite neighborhood.

## Verification

- Quartz C++ rebuilt successfully with CMake on `h100-gpu5`.
- The Cython extension rebuilt successfully against that library.
- The native single and batch APIs were exercised by the H100 searches above.
- The repository suite passed all 80 tests.
- `git apply --check --reverse` verified that the exported native patch exactly
  describes the tested Quartz source changes.
