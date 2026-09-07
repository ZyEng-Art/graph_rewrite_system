# Incremental successor fingerprint results (2026-09-07)

> Update: the native Quartz/Cython adaptation is now implemented and produces
> a net 24.0% Barenco depth-64 speedup in the adjacent H100 A/B.  See
> `NATIVE_SUCCESSOR_FINGERPRINT_RESULTS_20260907.md`.  The measurements below
> document the preceding Python prototype and motivated the native work.

## Outcome

The implementation has two pre-apply filters:

1. A general incremental physical-wire fingerprint predicts the successor by
   splicing only the rewritten intervals into the parent's wire traces.  Its
   selected `xfer_guarded` form removed 36.1% of Quartz applies on the
   Barenco depth-64 search and produced the exact same final beam, but the
   current Python implementation made the run 4.1% slower.  The idea works;
   profile construction and per-candidate Python dispatch are now the
   bottleneck.
2. An O(1) immediate-inverse fingerprint uses the previous rewrite id and its
   ordered destination slots.  It removed about 4.5% of Quartz applies and
   improved Barenco depth-64 wall time by about 2% in the repeated single-GPU
   comparison.  Every one of its 28,326 shadow hits was an exact duplicate,
   so this is the version worth enabling now.

All modes default to `off`.  Recommended current invocation:

```text
--preapply-direct-inverse filter
```

The general filter remains experimental until its batched native
implementation is faster:

```text
--preapply-direct-inverse filter \
--preapply-fingerprint filter \
--preapply-fingerprint-kind xfer_guarded \
--preapply-fingerprint-min-proposals-per-gate 0.4
```

## Implementation

`successor_fingerprint.py` builds an exact physical-wire trace containing the
gate type, physical operands and exact double parameter bytes for each live
operation.  For one proposal it:

1. maps the matched pattern's local qubits to physical qubits;
2. verifies that the source occupies a contiguous interval on every affected
   wire;
3. removes those intervals and inserts the destination operations logically;
4. combines prefix, destination and suffix hashes with a 128-bit rolling hash;
5. probes the global predicted-successor registry before Quartz copies or
   rewrites the graph.

The native Quartz patch `quartz_patches/wire_trace_profile.patch` exposes the
operation records directly from C++, avoiding QASM serialization, text
formatting and parsing.  The Python path retains a QASM fallback for an
unpatched Quartz build.  If a binding cannot be represented conservatively,
the fingerprint returns `None` and the candidate takes the original exact
Quartz path.

The beam benchmark supports:

- `--preapply-fingerprint off|shadow|filter`;
- `--preapply-fingerprint-kind conservative|parameter_transfer|xfer_guarded|topology`;
- `--preapply-fingerprint-representatives N`;
- `--preapply-fingerprint-min-proposals-per-gate R`;
- `--preapply-direct-inverse off|shadow|filter`.

`shadow` never filters.  It applies the proposal, obtains the authoritative
exact graph identity, and records whether a fingerprint hit was a real
duplicate, a collision, or an invalid action.  `topology` is prohibited in
filter mode.  The aggressive unguarded parameter-transfer mode requires at
least two representatives in filter mode.

The immediate-inverse filter is deliberately strict: the two ECC rules must
be each other's unique normalized reverse, and the new action's full ordered
source binding must be exactly the previous action's surviving destination
binding.  It does not reject merely nearby or syntactically similar actions.

## Barenco depth-64 A/B

Configuration: original 58-gate QASM, model search, beam 1000, depth 64,
128 actions per parent, proposal factor 16, microbatch 512, exact physical-wire
deduplication, rotation elimination, one idle H100 GPU.

### Cheap immediate-inverse filter

The most recent baseline was run immediately after the two filter runs on the
same H100.  Timing variance across the two baseline/filter repetitions is
roughly one to two seconds, so the apply reduction is the more stable metric.

| mode | Quartz applies | apply change | total time | time change | best |
|---|---:|---:|---:|---:|---:|
| off, repeat 2 | 629,259 | - | 74.987 s | - | 38 at step 34 |
| immediate inverse | 600,865 | -4.51% | 73.447 s | -2.05% | 38 at step 34 |
| immediate inverse + gated general fingerprint | 464,432 | -26.19% | 74.787 s | -0.27% | 38 at step 34 |

An earlier shadow/filter pair gave 78.057 s versus 75.669 s, also favoring
the immediate-inverse filter by 3.06%.  In shadow mode all 28,326 detected
inverse candidates materialized as already-seen exact circuits; none was new
and none was invalid.  The best graph identity was identical in every run:
`0c860d6ec5e5a1ff2277db6d79117ed026b12d96193813328caafd25bacd7f41`.

Artifacts:

- `benchmark_results/fingerprint_single_off_barenco_b1000_d64_r2_20260907.json`
- `benchmark_results/fingerprint_single_direct_inverse_barenco_b1000_d64_20260907.json`
- `benchmark_results/fingerprint_single_combined_r04_barenco_b1000_d64_20260907.json`
- `benchmark_results/direct_inverse_shadow_barenco_b1000_d64_20260907.json`
- `benchmark_results/direct_inverse_filter_barenco_b1000_d64_20260907.json`

### Full general wire fingerprint

In the sequential full-coverage comparison:

| mode | Quartz applies | apply change | fingerprint time | total time | best |
|---|---:|---:|---:|---:|---:|
| off | 629,311 | - | 0 s | 77.086 s | 38 at step 34 |
| `xfer_guarded` filter | 402,254 | -36.08% | 24.069 s | 80.273 s | 38 at step 34 |

The two final beams have exactly the same physical identity-set digest:
`613a7a1e6b4835e8a43f14724d92bcef293b8ba1d04404b9b5657b507467892a`.
Thus the 227,057 avoided applies did not alter the retained final circuits,
but the Python fingerprint work cost more than the avoided Quartz copy/apply
work.

The corresponding depth-64 shadow audit observed:

- 516,316 representable candidates;
- 226,973 fingerprint hits;
- 226,967 valid exact-duplicate hits;
- 6 hits whose Quartz action was invalid;
- zero valid successor collisions;
- at most one exact identity per fingerprint.

Artifacts:

- `benchmark_results/fingerprint_single_off_barenco_b1000_d64_20260907.json`
- `benchmark_results/fingerprint_single_xferguard_barenco_b1000_d64_20260907.json`
- `benchmark_results/fingerprint_xferguard_shadow_barenco_b1000_d64_20260907.json`

## Short-run Barenco and GF checks

| circuit / mode | Quartz applies | total time | best | exact quality check |
|---|---:|---:|---:|---|
| Barenco depth 8, off | 33,438 | 7.139 s | 55 | baseline |
| Barenco depth 8, `xfer_guarded` filter | 19,807 | 6.771 s | 55 | same final exact beam digest as shadow |
| GF depth 8, off | 15,662 | 31.793 s | 485 | baseline |
| GF depth 8, gated `xfer_guarded` filter | 14,822 | 32.211 s | 485 | same unique-state count |
| GF depth 8, inverse shadow | 15,668 | 31.734 s | 485 | 41/41 hits exact duplicates |
| GF depth 8, inverse filter | 15,622 | 31.640 s | 485 | same best exact identity |

The full general fingerprint helps the small Barenco graph but is not yet a
net win on the much larger GF graph.  A reuse gate of 0.2 reduced the GF
fingerprint work to 0.120 s, but the 0.418 s total-time difference from the
baseline is within run variance and slightly unfavorable.

Relevant artifacts:

- `benchmark_results/fingerprint_strict_ab_off_barenco_b1000_d8_20260907.json`
- `benchmark_results/fingerprint_xferguard_shadow_barenco_b1000_d8_20260907.json`
- `benchmark_results/fingerprint_xferguard_filter_barenco_b1000_d8_20260907.json`
- `benchmark_results/fingerprint_strict_ab_off_gf_b1000_d8_20260907.json`
- `benchmark_results/fingerprint_xferguard_shadow_gf_b1000_d8_20260907.json`
- `benchmark_results/fingerprint_xferguard_filter_r02_gf_b1000_d8_20260907.json`
- `benchmark_results/direct_inverse_shadow_gf_b1000_d8_20260907.json`
- `benchmark_results/direct_inverse_filter_gf_b1000_d8_20260907.json`

## Parameter safety

Compact ECC pattern strings do not contain the rule's parameter expressions,
so blindly copying same-type RZ parameters is not an exact symbolic evaluator.
The aggressive `parameter_transfer` shadow audit at Barenco depth 64 found
403,797 exact-duplicate hits but also 738 real collisions.  It must not be
used as a one-representative filter.

`xfer_guarded` additionally keys parameter-producing destinations by rewrite
id.  It gives up some cross-rule deduplication in exchange for zero observed
collisions on the Barenco depth-64 audit and both Barenco/GF depth-8 audits.
The conservative mode gives every parameter-producing rewrite a source-derived
symbolic effect token; it is safer but catches fewer duplicates.  Neither
observed zero collisions nor a 128-bit hash constitutes a formal proof, so the
new mode stays opt-in and retains shadow auditing.

Collision evidence:

- `benchmark_results/fingerprint_final_transfer_shadow_barenco_b1000_d64_20260907.json`

## High-quality trajectory audit

The authoritative Barenco 58-to-36 trajectory has 116 actions and contains no
immediate inverse.  The GF 495-to-371 trajectory has 271 actions and contains
26 immediate inverse actions.  Every such action returns to the immediately
preceding exact circuit, so exact global graph deduplication would discard it
after Quartz apply anyway.  Removing the two-action round trip shortens the GF
walk without removing any unique reachable state; it only means the saved
trajectory cannot be replayed literally step for step with this filter on.

Artifact:

- `benchmark_results/incremental_fingerprint_reference_audit_20260907.json`

## Verification

- The patched native `PyGraph.wire_trace_profile()` was built and exercised
  against the Barenco graph on `h100-gpu5`.
- The selected repository test suite passed: 80 tests, zero failures.
- General-filter quality was checked with exact physical-wire graph identities,
  not Quartz's lossy integer hash.
- Unknown/non-contiguous matches and unavailable native data fall back to the
  original exact Quartz apply path.

## Next optimization

The next useful change is not a larger learned similarity model.  The shadow
audit shows that successor prediction already identifies a large fraction of
the duplicates.  The remaining problem is execution cost.  The general
fingerprint should move into a batched C++/CUDA primitive that accepts one
parent profile and many `(xfer, binding)` rows, computes all affected-wire
splices in one call, and returns compact 128-bit keys.  A second option is to
carry a persistent wire profile in each accepted beam state and update only
the rewritten wires after the exact apply.  Either removes the Python object
construction that currently accounts for about 24 seconds at depth 64.
