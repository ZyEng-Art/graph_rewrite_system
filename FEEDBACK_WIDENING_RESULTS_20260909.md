# Deterministic feedback widening results (2026-09-09)

## Scope

All formal runs used an exact-apply budget of 100,000, beam size 256,
per-parent action cap 128, and the calibrated `r=0.999` matcher.  Runs were
made on `gpu1`; the excluded `h100-15` host was not used.

The feedback scheduler assigns revisit capacity deterministically across four
lanes: observed improvement, novel-child yield, UCB exploration, and bounded
detour/depth.  Exact Quartz identity remains authoritative for deduplication.

## Reproducibility fix

The first implementation ignored `widening_seed`, but two raw Barenco runs
still ended at 38 and 40 gates.  Their first divergence occurred at layer 3:
the preceding aggregate counts were equal, while the next matcher invocation
returned 59,947 versus 59,948 source-binding candidates.  The cause was CUDA
floating-point nondeterminism in GNN aggregation / cuBLAS, amplified by the
threshold and top-k cuts, rather than randomness in the feedback policy.

The fix has two parts:

1. Equal-score match, action, global-proposal, and survivor cuts use stable
   structural keys `(exact parent identity, xfer, anchor, binding)`.
2. `--deterministic-search` enables deterministic PyTorch CUDA algorithms and
   uses `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

A 3,000-apply diagnostic repeated with widening seeds 73 and 170 had identical
non-timing fields on every one of its four layers.  The 100,000-apply formal
repeat also had zero structural step differences across all 95 layers:

| Field | seed 73 | seed 170 |
|---|---:|---:|
| Best gate count | 38 | 38 |
| First-best layer | 77 | 77 |
| Best action depth | 69 | 69 |
| Maximum explored action depth | 86 | 86 |
| Attempted exact applies | 100,000 | 100,000 |
| Unique children | 24,085 | 24,085 |
| Duplicate children | 67,726 | 67,726 |
| Invalid actions | 8,189 | 8,189 |
| Wall time | 94.76 s | 93.94 s |

The action history, widening ancestry, feedback counters, final QASM SHA-256,
and every non-timing per-layer log field were identical.  Time to first see 38
was 16.54/16.53 seconds of measured search time; only timing noise differed.

## Search-quality A/B observations

### Raw Barenco, 58 gates

Before deterministic CUDA was enabled, fixed top-128 reached 38 gates at action
depth 36 in 81.99 seconds wall time.  Feedback widening reached deeper states
but was nondeterministic (38 at depth 52 for seed 73, 40 at depth 76 for seed
170).  After the reproducibility fix, both feedback repeats reached the same
38-gate circuit at action depth 69, with maximum explored action depth 86.

Thus feedback widening is demonstrably increasing depth, but 100,000 applies
still do not recover the known 36-gate result from the original 58-gate input.
The deterministic mode itself adds no material wall-time regression relative
to the earlier feedback run (94.76 versus 94.23 seconds); feedback scheduling
as a whole is about 15.6% slower than fixed top-128 in this end-to-end setup.

### Barenco reference suffix, 38 gates

All five policies reached 36 gates at action depth 7.  Fixed top-128 used
19,518 applies; feedback used 14,572 applies.  Both feedback repeats were
structurally identical.  This shows that once the search is placed at reference
step 98, the remaining optimization is well covered, but it does not explain
how to reach that basin from the raw 58-gate circuit.

### GF reference state, 372 gates

None of the five 100,000-apply runs found 371 gates.  Fixed top-128 reached
maximum action depth 69 in 181.97 seconds wall time.  Feedback reached maximum
depth 145/147 in 292.47/305.93 seconds, so the depth mechanism works, but this
extra depth did not produce an improvement.  The two old feedback runs also
had different structural traces; they predate deterministic CUDA mode and
should not be used as a reproducibility claim.

## Conclusion

Seed sensitivity is now removed under `--deterministic-search`: repeated runs
make the same decisions and return the same circuit.  The remaining issue is
search quality rather than reproducibility.  Feedback widening roughly doubles
reachable trajectory depth on raw Barenco and GF, but a generic value signal
is still needed to concentrate that depth into the narrow 36-gate/371-gate
basins instead of spending it on long neutral detours.
