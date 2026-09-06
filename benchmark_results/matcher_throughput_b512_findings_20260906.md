# Matcher throughput at the historical batch-512 shape (2026-09-06)

## Historical benchmark convention

The earlier headline comparison used `hwb6`, beam 1000, and microbatch 512. At the full third search layer it reported approximately 423 matched states/s for the model and 9.72 states/s for original CPU Quartz, or 43.5x matcher throughput. Later cross-circuit beam-256 runs used microbatch 256 and are a different benchmark shape.

The current comparison therefore uses one batch of exactly 512 states. Because the held-out reference paths contain only 271 GF states and 16 Barenco states, their real states are cyclically repeated to fill the batch. The benchmark records both the unique-state count and the repetition; this is a throughput-shape measurement, not a claim that 512 independent reference states exist.

## Result

All measurements use the selected positive-safe checkpoint, recall-0.999 calibration, an 8192 source-anchor cap, one H100 80 GB, and the same 4700-xfer Quartz context. CPU graphs and the Quartz context are preloaded. QASM parsing and model loading are excluded.

Throughput is states/s. Speedup is relative to Quartz's original CPU `available_xfers_parallel` xfer-at-anchor enumeration on the same 512-state workload.

| Timed path | GF throughput | GF speedup | Barenco throughput | Barenco speedup |
| --- | ---: | ---: | ---: | ---: |
| Original CPU Quartz xfer-anchor | 6.1113 | 1.00x | 67.5398 | 1.00x |
| CPU Quartz full-binding ground truth | 5.7237 | 0.94x | 65.9774 | 0.98x |
| Model: GPU-resident through structural decode | 1990.5587 | **325.72x** | 10764.9436 | **159.39x** |
| Model: D2H plus Python candidate rows | 158.4863 | **25.93x** | 1256.4592 | **18.60x** |
| Model: GPU-resident plus prefix replay/tensor preparation | 119.3754 | **19.53x** | 1183.9949 | **17.53x** |
| Model: host rows plus prefix replay/tensor preparation | 70.5007 | **11.54x** | 646.1654 | **9.57x** |

## Complete GPU proposal path

The host-materialized row above is deliberately pessimistic: it copies every
retained source binding to Python.  The actual GPU proposal backend does not do
that.  It expands source bindings to xfers, applies the gate-increase filter,
ranks and caps actions per parent and globally on the GPU, and copies only the
final proposals that the Quartz successor stage will try.  A follow-up run
times that complete path with a per-parent cap of 128 and a global cap of 8192.

| Complete timed path | GF throughput | GF speedup | Barenco throughput | Barenco speedup |
| --- | ---: | ---: | ---: | ---: |
| GPU full expansion/ranking + selected-only D2H | 1812.5074 | **296.58x** | 8232.8190 | **121.90x** |
| GPU match preselection + expansion/ranking + selected-only D2H | 1843.6496 | **301.68x** | 8030.5233 | **118.90x** |

Thus the deployable matcher/proposal boundary is not the earlier 19--26x
all-row host-materialization result.  With only final proposals transferred,
it retains a 119--302x advantage over original CPU Quartz enumeration in this
batch-512 comparison.  The remaining CPU Quartz rewrite application is outside
both matcher timings.

The number of GPU action rows and final host objects explains the difference:

| 512-state workload | Eligible xfer rows | Xfer rows after optional preselection | Final D2H proposals |
| --- | ---: | ---: | ---: |
| GF, full | 2,059,253 | 2,059,253 | 8,192 |
| GF, preselect | 2,059,254 | 128,467 | 8,192 |
| Barenco, full | 115,360 | 115,360 | 8,192 |
| Barenco, preselect | 115,360 | 112,320 | 8,192 |

Preselection is exact for gate/probability ranking, but is not automatically
faster.  It reduced GF intermediate rows by 16.0x and was 1.7% faster in this
matcher benchmark; Barenco had almost no removable rows and was 2.5% slower.
The production default therefore remains full GPU expansion, with
`--proposal-expansion preselect` available for very dense workloads.

The GPU-resident timing includes H2D, graph/action encoding, source-anchor logits, r99.9 calibration and thresholding, the per-state 8192 cap, and complete-binding structural decode. Its output stays as GPU tensors.

The host-materialized timing adds device-to-host copies and Python packing of every `(source, anchor, binding, probability)` row. The final row additionally charges `PrefixDataset` action-prefix replay and CPU collation once for the 512 states. That reconstruction is conservative for a live optimizer that already maintains incremental current-state tensors, but it exposes the same class of state-materialization bottleneck identified in the historical depth-64 stage profile.

## Candidate volume

| Candidate count per state | GF | Barenco |
| --- | ---: | ---: |
| Exact CPU Quartz xfer-anchor actions | 3285.58 | 125.06 |
| r99.9 model source-binding rows after structural decode | 2837.28 | 196.38 |
| Model rows expanded back to xfer actions | 4843.35 | 307.31 |

The model can emit more expanded xfer actions than CPU Quartz because r99.9 intentionally retains false-positive source bindings. This is why moving and packing every candidate reduces the apparent acceleration from 159-326x in the GPU core to 19-26x, and prefix/tensor preparation reduces the conservative full matcher path further to 9.6-11.5x.

The appropriate integration conclusion is therefore conditional:

- keep filtering, structural decode, downstream ranking, and top-k selection GPU-resident to preserve the hundred-fold matcher-core advantage;
- if every r99.9 candidate is immediately materialized as a Python row for CPU Quartz application, expect approximately a ten-fold matcher-path advantage on these two workloads, not the core 159-326x figure;
- batch-1 is not representative of the historical throughput claim, especially for the 39-gate Barenco state where kernel-launch overhead can make CPU Quartz slightly faster.

## Validation and artifacts

Before cyclic repetition, every QASM graph hash is checked against the corresponding dataset state. CPU full-binding enumeration returns the same exact ground-truth sets used by the recall audit: 559524 source bindings across the 271 GF states and 1600 across the 16 Barenco states.

The GF GPU candidate count varies by at most three rows among roughly 1.45 million rows over five repeats due to BF16/SDPA boundary ties. This is approximately two parts per million and is recorded rather than hidden. Barenco is exactly stable over ten repeats.

- Driver: `benchmark_matcher_throughput.py`
- Remote GF result: `benchmark_results/matcher_throughput_gf370_2_b512_r999_s8192_h100_cpu_20260906.json`
  - SHA-256: `5c0a38ed9a6c27fcd88036eac21f6c70067581f37f2bd3e419b72dd29652e902`
- Remote Barenco result: `benchmark_results/matcher_throughput_barenco38_3_b512_r999_s8192_h100_cpu_20260906.json`
  - SHA-256: `f1732780b4368d388c9eee781521bec5f9a295f259ce0c0266b848835aff09bd`
- Complete GPU-proposal GF result: `benchmark_results/matcher_throughput_gf370_2_b512_gpu_proposals_h100_20260906.json`
  - SHA-256: `e85b17a473cbc8272d447384b80605d371e1a18b1a10e400e9ead75c8a2e7c06`
- Complete GPU-proposal Barenco result: `benchmark_results/matcher_throughput_barenco38_3_b512_gpu_proposals_h100_20260906.json`
  - SHA-256: `914ac92d8caaa12010f474f643e5af8368a3d923da1df817104bcec3956a9132`

## Search-level equivalence check

A separate beam-256, depth-8 H100 A/B started from the first state of each
high-quality trajectory.  Full and preselected expansion retained identical
accepted-state counts at every depth.  Barenco ended with 130 states and GF
with 256; every retained state passed Quartz replay and exact-topology checks.
Both modes reported the same best exact gate counts (39 for this Barenco start,
371 for this GF start).

Preselection reduced the GF search's materialized action rows from 6,779,897
to 424,511 (16.0x) while leaving the 29,184 globally capped proposals and all
1,918 accepted successors unchanged.  Barenco reduced only 270,848 to 269,294
rows and left 28,797 capped proposals and all 1,709 accepted successors
unchanged.  At this smaller beam shape the proposal-stage time did not improve,
which is why preselection is kept opt-in rather than forced globally.

The four raw search results are
`barenco38_3_b256_d8_proposal_{full,preselect}_final_20260906.json` and
`gf370_2_b256_d8_proposal_{full,preselect}_final_20260906.json` in
`benchmark_results/`.

## End-to-end follow-up

The matcher/proposal numbers above are not optimizer-level speedups. A later
beam-1000, depth-3, microbatch-512 comparison includes proposal processing,
successor construction, deduplication, cache advance, and final exact Quartz
refresh. Its median end-to-end speedup is 88.80x on GF but only 1.27x on the
small Barenco start; the latter requires 64x proposal overgeneration to refill
a 1,000-state beam after exact validation. Full timing boundaries, three GPU
repetitions, legality audits, and the exact-diversity caveat are in
`benchmark_results/end_to_end_throughput_findings_20260906.md`.
