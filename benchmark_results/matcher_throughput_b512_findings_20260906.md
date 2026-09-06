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
