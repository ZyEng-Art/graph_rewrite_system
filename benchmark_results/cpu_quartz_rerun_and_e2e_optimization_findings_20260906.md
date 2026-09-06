# CPU Quartz rerun and end-to-end optimization (2026-09-06)

## Result first

The batch-512 matcher/proposal hot path is still much faster than original CPU
Quartz: **319.37x on GF `370_2`** and **129.62x on Barenco `38_3`**. Those are
matching/proposal throughput numbers, not optimizer-level speedups.

At an equal, fully specified search budget:

| Circuit | Search budget | CPU Quartz | H100 median | Search speedup | Confirmed result |
|---|---:|---:|---:|---:|---:|
| GF `370_2` | beam 1000, depth 3 | 278.836 s | 6.632 s | **42.04x** | both 371 gates; both return 1000 exact-unique states |
| Barenco `38_3` | beam 1000, depth 16 | 253.891 s | 6.311 s | **40.23x** | both reach 38 gates, but the GPU history-diverse beam has only 113 exact-unique final states |

The extra independent full replay audit is deliberately outside the GPU search
timer because the search already performs exact Quartz refresh validation. If it
is charged a second time, the conservative speedups are 19.39x for GF and
29.66x for Barenco.

The two long optimization outcomes are therefore:

- Barenco: **39 -> 38 reproduced** at depth 16. All 790 returned histories are
  Quartz-valid and topology-exact, but they collapse to 113 exact circuits.
- GF: a complete autonomous 271-step run finished, but stayed **371 -> 371**.
  All 1000 final histories are valid and topology-exact; there are 492 exact
  final circuits. The saved reference's 370-gate endpoint was not found.

## Batch-512 CPU versus H100 matcher throughput

The 512 states are the real saved trajectory states, cyclically repeated: 271
unique inputs for GF and 16 for Barenco. Both paths use the same 4700 rewrites.
The CPU number calls original `available_xfers_parallel` at every node. The H100
number includes model matching, r99.9 filtering, full xfer expansion/ranking,
per-parent cap 128, global cap 8192, and selected-only device-to-host transfer.
Dataset/checkpoint loading and QASM parsing are excluded. GPU medians use five
repetitions; CPU is one complete 512-state pass.

| Circuit | CPU seconds / states/s | H100 full proposal seconds / states/s | Speedup | GPU-resident-only speedup |
|---|---:|---:|---:|---:|
| GF `370_2` | 83.276 / 6.148 | 0.26075 / 1963.54 | **319.37x** | 351.94x |
| Barenco `38_3` | 8.1626 / 62.725 | 0.06297 / 8130.37 | **129.62x** | 174.96x |

If the one-time conversion of all 512 already-materialized CPU graphs into GPU
tensors is charged to a single batch, the ratios are 17.16x (GF) and 14.51x
(Barenco). In the paged optimizer that conversion is not repeated: current
state tensors and KV pages remain resident and are incrementally advanced, so
the hot-path comparison is the relevant steady-state number.

GF has a tiny BF16 threshold-boundary variation across repetitions (at most
seven source bindings out of about 1.45 million). Barenco's candidate count is
identical in all five repetitions. This does not change the selected cap or the
reported conclusion.

## Equal-budget end-to-end details

### GF, beam 1000, depth 3, maximum single-step increase 2

The rerun uses collision-safe physical-wire/parameter identity in both paths.
The CPU run enumerates and applies real Quartz graphs. The GPU run performs
model matching, GPU proposal selection, lazy successor construction, paged KV
advance, exact Quartz refresh at depth 3, and exact deduplication while refilling
the final beam.

- CPU: 278.836 s, 371 gates, 1000 final states, 1000 exact identities.
- GPU search: 6.632, 7.204, and 6.476 s; median 6.632 s.
- Each GPU run: 371 gates, 1000/1000 valid and topology-exact, 1000 exact
  identities.
- Primary speedup: **42.04x**.
- Median-run second audit: 7.747 s; search plus redundant audit is 14.379 s,
  or **19.39x**.

This clean rerun agrees with the earlier strict-unique depth-3 result (roughly
46x); the small difference is normal run-to-run and configuration variation.

### Barenco, beam 1000, depth 16, maximum single-step increase 2

The `+2` allowance is necessary: the saved 16-action route begins 39 -> 41 and
reaches 38 only at its final action. CPU Quartz uses exact global state dedup.
The H100 optimization uses the Barenco-specific action-value model, r99 matcher
filtering, value weight 4, refresh every four actions, and full final replay.

- CPU: 253.891 s, final best 38, 1000 exact-unique final states. The first
  38-gate beam appears at depth 4 after 37.829 s.
- GPU search repetitions: 13.315, 6.311, and 5.600 s; median 6.311 s. Every run
  confirms 38 gates at depth 16 and returns the same 790 valid histories / 113
  exact circuits.
- Search-time ratio: **40.23x**. Including the median run's 2.249 s independent
  audit gives **29.66x**.

The 40.23x number is a throughput-to-result comparison, not an equal-diversity
claim. With exact dedup restricted to the current refresh level and an 8x
refill pool, the H100 run keeps 839 different, valid final circuits in 10.812 s
but does not reach 38; its confirmed best remains 39. Thus there is currently
no honest quality-and-diversity-equivalent GPU speedup claim for Barenco.

## Full 271-step GF optimization

The checkpoint was trained with a maximum action context of 64. A new opt-in
rebase path therefore re-encodes every surviving exact circuit at refreshes 64,
128, 192, and 256 and resets only the model KV history. The physical circuit,
full action history, gate count, and Quartz replay checkpoint keep advancing.
A beam-16 depth-65 smoke test confirms the boundary: the rebase takes 0.043 s
and all 16 final paths pass independent replay. In the beam-1000 run the four
rebases total 10.56 s without refresh dedup and 8.97 s with level dedup.

Two complete runs were made:

| Mode | Search | Independent audit | Best | Final valid | Final exact identities |
|---|---:|---:|---:|---:|---:|
| preserve multiple histories | 321.682 s | 422.927 s | 371 | 1000/1000 | 492 |
| exact dedup within each refresh level | 391.317 s | 504.592 s | 371 | 1000/1000 | 492 after the final seven unrefreshed steps |

In the level-dedup run, 33 refreshes replay 66,000 legal candidates, filter
38,072 within-level duplicates, and retain 27,928 exact identities in total
(about 846 per refresh). Exact identity construction itself takes 19.686 s.
At the last refresh, depth 264, 832 of 2000 candidates are different circuits.
Seven later speculative steps refill the beam to 1000; the final independent
audit finds 492 distinct exact endpoints.

For comparison, a global exact registry is too strong for the current
history-conditioned policy: it reaches depth 48 and then all 4000 replay-valid
candidates are circuits seen at earlier refreshes, so the beam becomes empty.
Level scope solves that termination problem while still removing the duplicate
circuits caused by permutations of independent actions inside a beam.

## Why GF still does not reach 370

This is now a policy/ranking failure, not a matcher-throughput or legality
failure.

- The selected checkpoint is trained to retrieve applicable source patterns;
  it has no GF-specific long-horizon value head.
- Gate-count ranking prefers immediate 371-gate continuations. The reference
  path does not improve until action 271 and temporarily rises as high as 375,
  so its useful actions are almost indistinguishable from millions of neutral
  or locally worse alternatives under this ranking.
- Two depth-64 stochastic controls also fail: their best confirmed graph remains
  the initial 371, while their final beams drift to 422 and 429 gates.
- Re-encoding at 64 steps makes the requested horizon executable, but discards
  policy history at each boundary. It cannot supply the missing long-term value
  signal.

The next modeling step should be a GF-specific value/PPO objective trained on
loop-erased high-quality paths plus diverse random circuits, with evaluation on
held-out circuits. Increasing matcher recall or the proposal pool alone is not
supported by these results: candidate production is already fast and the exact
refreshes show that the selected GF actions are legal; the search simply lacks
a score that distinguishes the rare 271-step useful route.

## Code changes and artifacts

- `beam_search_benchmark.py` now defaults CPU Quartz deduplication to the same
  collision-safe exact circuit identity; `--dedup-identity quartz_hash` retains
  the historical lossy A/B path.
- `paged_rollout_benchmark.py` adds
  `--rebase-model-history-at-refresh` for horizons beyond the learned 64-token
  context and `--refresh-dedup-scope level` for same-level exact dedup without
  globally suppressing later policy histories.
- The rebase packing helper has a unit test. A depth-65 H100 smoke test and all
  full-run final replay audits pass with no topology mismatch.

The compact machine-readable rollup is
`cpu_quartz_rerun_and_e2e_optimization_summary_20260906.json`. Raw JSON files
beside this report contain every step timing, refresh count, identity statistic,
and final audit result.
