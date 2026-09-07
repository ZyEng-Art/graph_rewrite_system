# Learned successor prefilter experiment (2026-09-07)

## Result

A small learned model can cheaply prioritize likely novel, legal successors,
but it is not accurate enough to replace Quartz apply plus exact graph identity.
It is therefore optional and disabled by default.  The safe implementation mode
is `defer`: high-confidence invalid/duplicate proposals are moved to the end of
the apply queue and are still tried if the beam cannot otherwise be filled.

On held-out Barenco-3 at depth 16, learned deferral and the native incremental
fingerprint are complementary.  Their combination reduced apply calls by 43.8%
and search time by 24.3%, while producing the same exact final beam as the
unfiltered run.  On held-out GF(2^6), however, learned deferral alone saved only
2.0% of apply calls and 0.5% of time.  GF final beams were not identical across
the four modes, although all modes found the same best gate count.  This is a
useful heuristic, not an exact deduplicator.

## Model and exact labels

Candidate features reuse the frozen current-graph matcher, so the extra model
does not run another GNN:

- xfer embedding: 192 dimensions
- source-pattern representation: 192 dimensions
- mean representation of bound nodes: 192 dimensions
- mean representation of the parent graph: 192 dimensions
- matcher probability, gate delta, parent gate count, and search depth: 4 scalars

The prefilter is a `772 -> 256 -> 256` MLP with three outputs: legality logit,
duplicate-propensity logit, and a normalized 64-dimensional successor embedding.
Training uses weighted legality and duplicate BCE losses plus an exact-successor
pair contrastive loss.

Every training target is obtained after an actual Quartz rewrite:

- `0`: Quartz rejected the binding;
- `1`: Quartz produced a graph already present in the exact graph registry;
- `2`: Quartz produced a new exact graph identity.

The four training circuits were Barenco-4, Tof-4, GF(2^4), and HWB6, each at
depth 8.  They contributed 103,269 rows: 2,141 invalid, 71,634 duplicate, and
29,494 novel.  Barenco-5 and GF(2^5), 47,593 rows total, were used only to set
thresholds.  Barenco-3 at depth 16 and GF(2^6) at depth 8, 108,313 rows total,
were held out from both training and calibration.

## Cross-circuit classification

Thresholds were selected so that no novel/valid candidate was filtered on the
calibration circuits.  That property did not transfer perfectly to held-out
circuits:

| Head | Calibration recall | Calibration false positives | Held-out recall | Held-out false positives |
| --- | ---: | ---: | ---: | ---: |
| Invalid binding | 85.35% (897/1,051) | 0/46,542 | 35.65% (4,314/12,101) | 1/96,212 |
| Duplicate propensity | 18.36% (5,823/31,714) | 0/14,828 | 19.66% (14,478/73,632) | 39/22,580 |
| Successor embedding retrieval | 0% at zero-FP threshold | 0/14,826 | 0% at the transferred threshold | 0/22,578 |

The successor embedding's nearest-group accuracy was only 28.32% on held-out
duplicates.  Neural successor equivalence is consequently unsuitable for a hard
dedup decision.  A model calibrated to zero observed errors also does not give a
formal zero-error guarantee on another circuit.

## End-to-end H100 A/B

All runs used one otherwise idle H100 on `h100-gpu1`, batch size 512, beam size
1,000, proposal factor 16, exact graph identity, direct ordered-binding apply,
and zero-rotation elimination.  Modes were run serially on the same GPU.  The
reported time is the benchmark's search time and excludes one-time Quartz
context construction.

### Barenco-3, depth 16

| Mode | Apply calls | Apply reduction | Search time | Speedup | Best gates | Exact final beam |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Off | 92,634 | - | 15.873 s | 1.00x | 48 | reference |
| Neural defer | 73,789 | 20.34% | 13.638 s | 1.16x | 48 | identical |
| Native fingerprint | 63,132 | 31.85% | 12.889 s | 1.23x | 48 | identical |
| Native + neural | 52,099 | 43.76% | 12.022 s | 1.32x | 48 | identical |

The learned model consumed only 47.8 ms in neural-only mode and 70.0 ms in the
hybrid run.  The hybrid also spent 916 ms on native fingerprints.

### GF(2^6), depth 8

| Mode | Apply calls | Apply reduction | Search time | Speedup | Best gates | Exact final beam |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Off | 15,665 | - | 31.377 s | 1.00x | 485 | reference |
| Neural defer | 15,356 | 1.97% | 31.225 s | 1.00x | 485 | different |
| Native fingerprint | 11,500 | 26.59% | 28.982 s | 1.08x | 485 | different |
| Native + neural | 11,459 | 26.85% | 29.084 s | 1.08x | 485 | different |

Neural scoring took 70.5 ms in neural-only mode and 43.0 ms in hybrid mode, but
the GF duplicate distribution provided almost no additional candidates that
could safely be delayed before the beam filled.

## Operational behavior

`--neural-prefilter-mode shadow` computes scores without changing proposal
order.  `--neural-prefilter-mode defer` stably partitions proposals: candidates
below the calibrated legality threshold or above the duplicate threshold are
placed after all other proposals.  Both partitions retain their original order.
Quartz still applies every proposal that is reached, and the exact registry
still decides whether its result is a duplicate.

If the first partition does not fill the beam, deferred candidates are consumed
automatically.  If it does fill the beam, deferred candidates are not applied;
this is where the speedup comes from and also why a neural false positive may
change the selected beam.  Hard neural filtering is intentionally not exposed.

Reproduce data collection with `run_neural_audit_collection.sh`, train with
`train_neural_successor_prefilter.py`, and reproduce the four end-to-end modes
with `run_neural_prefilter_benchmark.sh`.  The trained checkpoint and full JSON
metrics/results are under `benchmark_results/neural_*`.
