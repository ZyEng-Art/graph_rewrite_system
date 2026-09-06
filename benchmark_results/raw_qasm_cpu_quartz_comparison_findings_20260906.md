# Raw-QASM CPU Quartz comparison (H100, 2026-09-06)

## Scope correction

Every search in this report starts from the untouched circuit file:

- `barenco_tof_3.qasm`: 58 gates, known supplied target 35.
- `gf2^6_mult.qasm`: 495 gates, known supplied target 369.

No 38/39-gate Barenco state and no 371-gate GF state is used as an input.
The file named `quarl_barenco38_3_normalized_w8_20260906.pt` supplies only the
rewrite vocabulary and source-pattern metadata required to construct the model.
The benchmark result now records this explicitly as
`rule_metadata_role: rewrite vocabulary only; not a search state`.

## Comparable protocol

The direct CPU/model end-to-end runs use beam 1000, per-parent cap 128,
proposal factor 16, maximum static gate increase 3, and collision-safe exact
circuit identity. Every selected successor is created by Quartz itself with
rotation elimination enabled, so both modes observe the actual post-RZ-folding
gate count after every action. Candidate support and equal-cost ordering differ
as follows:

- CPU mode enumerates all Quartz xfer-at-anchor matches.
- Model mode uses the R99.9 learned source candidates, expands them to concrete
  xfers, and uses model probability for equal-gate-count ordering.

This exact-apply benchmark supersedes the earlier lazy paged raw-QASM trial.
The lazy path used static ECC gate deltas between refreshes and therefore did
not reproduce Quartz's automatic rotation contraction after every action; its
end-to-end gate result is not used in the CPU comparison below.

## Batch-512 candidate throughput

The input batch is 512 repetitions of the untouched raw QASM state, not 512
trajectory states.

| Raw input | CPU Quartz | GPU full proposal core | Core speedup | GPU host-materialized | Materialized speedup | GPU including one-time tensor preparation | Inclusive speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Barenco, 58 gates | 46.04 states/s | 15,695.98 states/s | 340.89x | 2,399.88 states/s | 52.12x | 923.32 states/s | 20.05x |
| GF, 495 gates | 5.16 states/s | 4,262.51 states/s | 826.18x | 229.69 states/s | 44.52x | 124.25 states/s | 24.08x |

The 340.89x/826.18x numbers are matcher/proposal-core speedups and must not be
reported as optimizer speedups. Host materialization and initial tensor
construction reduce them substantially before Quartz apply, sorting, and
deduplication are counted.

## Direct end-to-end comparison, depth 8

| Raw input | CPU Quartz result | CPU time | Model result | Model time | Result interpretation |
|---|---:|---:|---:|---:|---|
| Barenco, 58 gates | 55 | 156.28s | 56 | 31.57s | 4.95x less wall time, but one gate worse |
| GF, 495 gates | 485 | 1,445.62s | 485 | 133.06s | equal quality, 10.86x end-to-end speedup |

For a same-quality Barenco checkpoint, the model first reaches 56 gates in
12.36s and CPU Quartz first reaches 56 in 56.82s, a 4.60x speedup. CPU then
finds 55 at step 8 while the model beam does not. For GF, both first reach 485
at step 5: 82.80s for the model and 821.52s for CPU Quartz, a 9.92x speedup to
the same result.

Inside the GF depth-8 search, candidate matching improves from 5.03 to 161.50
states/s (32.10x), while full optimization improves by 10.86x. The remaining
time is dominated by proposal materialization/sorting, Quartz apply, and exact
deduplication.

## Long search from the original circuits

Barenco was run for the same 128-depth budget in both modes:

| Mode | Raw start -> best | First best | Full 128-depth time | Unique circuits | Reached 35? |
|---|---:|---:|---:|---:|---:|
| CPU Quartz | 58 -> 38 | step 34, 755.25s | 2,554.60s | 127,093 | no |
| Model candidates | 58 -> 38 | step 36, 161.23s | 699.17s | 122,886 | no |

The model is 4.68x faster to the same 38-gate best and 3.65x faster over the
full budget. Neither complete Quartz matching nor model matching reaches the
supplied 35-gate target under this beam policy. Therefore the failure to reach
35 is not caused solely by probability thresholding: the current immediate
gate-count ordering and branch-retention policy also fail to preserve the
necessary preparatory sequence. The one-gate depth-8 difference still shows
that model support/ranking can alter which useful branch survives.

The model-only GF depth-128 reachability run visits 127,886 unique circuits and
improves 495 -> 468 in 2,754.37s. It does not reach the supplied 369-gate
target. A CPU depth-128 GF run was not performed; the exact depth-8 run already
takes 1,445.62s, and the direct CPU/model claim for GF is deliberately limited
to that shared budget.

## Conclusion

The learned matcher provides a real and large candidate-generation speedup,
and a 3.65x-10.86x measured end-to-end speedup when quality is comparable.
It does not by itself reproduce the two supplied optimum trajectories from the
raw circuits. The next search branch should spend its effort on long-horizon
branch retention (including commutation-equivalent action orders and neutral
preparatory moves), while treating the matcher as a high-throughput support
generator. Model work remains relevant for the Barenco depth-8 branch that is
lost, but increasing R99.9 alone cannot fix a failure that also occurs with the
complete CPU Quartz candidate set.

## Artifacts

- `matcher_throughput_raw_barenco58_b512_r999_s906.json`
- `matcher_throughput_raw_gf2_6_495_b512_r999_s906.json`
- `e2e_raw_barenco_cpu_quartz_b1000_d8_inc3_s906.json`
- `e2e_raw_barenco_gpu_model_exactapply_b1000_d8_inc3_s906.json`
- `e2e_raw_gf2_6_cpu_quartz_b1000_d8_inc3_s906.json`
- `e2e_raw_gf2_6_gpu_model_exactapply_b1000_d8_inc3_s906.json`
- `e2e_raw_barenco_cpu_quartz_b1000_d128_inc3_s906.json`
- `e2e_raw_barenco_gpu_model_exactapply_b1000_d128_inc3_s906.json`
- `e2e_raw_gf2_6_gpu_model_exactapply_b1000_d128_inc3_s906.json`
- Each search JSON has a corresponding `_best.qasm` file.
