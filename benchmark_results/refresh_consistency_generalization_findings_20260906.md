# Refresh-consistent matcher generalization (2026-09-06)

## Outcome

The selected matcher and calibration retain every teacher action from both held-out high-quality optimization trajectories under strict thresholding and a global per-source cap of 8192:

| Circuit / model view | Teacher actions retained | All exact matches retained | Exact-match recall |
| --- | ---: | ---: | ---: |
| `gf2^6_mult/370_2`, rolling window 8 | 271 / 271 | 559297 / 559524 | 0.9995942980 |
| `gf2^6_mult/370_2`, rolling window 64 | 271 / 271 | 559467 / 559524 | 0.9998981277 |
| `barenco/38_3`, rolling window 8 | 16 / 16 | 1600 / 1600 | 1.0000000000 |
| `barenco/38_3`, rolling window 64 | 16 / 16 | 1600 / 1600 | 1.0000000000 |

This establishes candidate-level replay: every action in both reference trajectories survives model scoring, calibrated filtering, the 8192 cap, and structural decoding. It does **not** establish that an autonomous beam or stochastic search will rank and select those actions; search policy work can be performed independently on the retained candidates.

The strict audit does not use a `near reserve` or any unthresholded fallback. The labels `near` and `far` in the calibration and audit files are calibration partitions only. Both partitions pass through their learned thresholds and the same global cap.

## Selected artifacts

- Checkpoint: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/paged_action_refreshrand07_complex_possafe_c015_s311.pt`
  - SHA-256: `d24313ad152448d0bf3ca4b18a405ed945be613c42d142d0a12a56c16e557d0c`
- Recall-0.999 calibration: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/paged_action_refreshrand07_complex_possafe_c015_s311_calibration_r999.json`
  - SHA-256: `8a803b8e26485f2ca5db633863002d6a13e12ad85d13273de1333f91f94763ff`
- Training data: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/data/binding_longmix_randomrefresh_complex_holdout_20260906.pt`
  - SHA-256: `e9f03f64f856d911ed846a6e3a3ee7dc1eb8030757d212a13d508c3bdaba1202`
- Remote audit directory: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/experiment/refresh_consistency_model_20260906/benchmark_results`
  - `refreshrand_possafe_c015_gf_w8_r999_s8192_norsv.json`
  - `refreshrand_possafe_c015_gf_w64_r999_s8192_norsv.json`
  - `refreshrand_possafe_c015_barenco_w8_r999_s8192_norsv.json`
  - `refreshrand_possafe_c015_barenco_w64_r999_s8192_norsv.json`

The recall-0.999 calibration was fitted on 512 held-out states. Its raw-logit thresholds and estimated pre-decode candidate counts are:

| Calibration partition | Raw-logit threshold | Estimated candidates/state |
| --- | ---: | ---: |
| `far` | -5.5378713608 | 4946.7417 |
| `near` | -6.9639654160 | 3553.0540 |

## Complex random rewrite data

Two independently generated datasets were added without using the target circuits:

| Dataset | Circuit range | Rewrite states | Continued-local states | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `random_rewrite_rich_exact_n128_s307_20260906.pt` | 128 circuits, 4-12 qubits, 48-192 initial gates, 12-48 actions | 3624 | 2775 | `c9c9dccce44b3bbbec0a223bdfa60868f6fb594980437fc139d2461a67e0165f` |
| `random_rewrite_large_exact_n128_s309_20260906.pt` | 128 circuits, 8-24 qubits, 192-448 initial gates, 16-56 actions | 4364 | 3369 | `7f29d71d7952f908016d12cfe98b3ed4e5718b94b264625dc6743c3bc89239d6` |

Together they add 7988 usable states, of which 6144 (76.91%) continue a spatially local rewrite sequence. Quartz enumerates the complete exact match set for every collected state; the random walk chooses a valid rewrite with locality bias, rather than fabricating action labels.

The merged training set contains 34912 usable training states and 2048 held-out evaluation states. Before merging, it contained 26924 usable training states.

Leakage audits against both `gf2^6_mult/370_2` and `barenco/38_3` found:

- zero initial/final graph-hash overlap;
- zero supervised-state overlap;
- zero shared supervised action keys.

Thus the target result is a generalization test, not target-trajectory memorization.

## Model-side changes

1. Each training state can be represented by two causally equivalent views: a short rolling history and a longer retained prefix ending in the same current circuit.
2. The retained prefix is sampled uniformly from 0 through 7 actions, covering all phases of an eight-step refresh cycle instead of training only one fixed offset.
3. Positive consistency is one-sided and safe: for every exact match, only the weaker view is aligned upward toward the detached stronger score. The stronger positive is never pulled downward.
4. Hard-negative consistency remains symmetric so the two views do not acquire incompatible false-positive distributions.
5. Random rewrite-rich circuits supply many local multi-step changes at circuit sizes substantially larger than the earlier synthetic set.

Unit and smoke validation performed on the H100 environment:

- 11/11 focused tests passed (refresh rebasing, random trajectory generation/order, windowing, and threshold inference);
- random collector smoke test passed;
- merged-data two-batch training smoke test passed with four workers;
- paired short/long views were checked to have identical current gate types, edge sets, exact match sets, and teacher actions.

## Ablations and why recall 0.999 is selected

The full-binding Top-N metric on the ordinary held-out set improved from about 0.881 to about 0.898 after adding the complex random data and refresh-consistent objective:

| Training variant | Held-out full-binding Top-N | Observation |
| --- | ---: | --- |
| Original baseline | 0.8808759 | No refresh-consistency training |
| Fixed-window refresh augmentation only | 0.8807443 | GF teacher recall was 270/271 in each view at r99 |
| Symmetric positive consistency | 0.8851180 | Pulled some strong positive scores down; rejected |
| Random 0-7 prefix, positive-safe consistency 0.05 | 0.8978725 | Strong general improvement; GF 269/271 at r99 |
| Random 0-7 prefix, positive-safe consistency 0.15 | 0.8984060 | Selected model; best critical-action ranks before calibration |
| Same model family, boundary weight 2 | 0.8985251 | r99 target retention regressed despite a slightly higher aggregate metric |
| Same model family, boundary weight 4 | 0.8967902 | Aggregate metric regressed; no expensive target audit run |

Strict teacher-action retention at recall-0.99 calibration was:

| Variant | GF w8 | GF w64 | Barenco w8 | Barenco w64 |
| --- | ---: | ---: | ---: | ---: |
| Fixed-window augmentation | 270/271 | 270/271 | 16/16 | 16/16 |
| Random-prefix positive-safe 0.05 | 269/271 | 269/271 | 16/16 | 16/16 |
| Random-prefix positive-safe 0.15 | 267/271 | 267/271 | 15/16 | 15/16 |
| Selected model, recalibrated to r99.9 | **271/271** | **271/271** | **16/16** | **16/16** |

This is not a contradiction between the model and calibration. An r99 threshold is explicitly allowed to discard about 1% of positives on its calibration distribution. For a 271-action reference path, zero omissions are therefore not the expected guarantee. Increasing boundary-loss weight changed which rare positives failed but did not make r99 a zero-loss operating point. The selected model with r99.9 is the first tested configuration that meets the actual requirement—do not filter any reference action—in all four strict audits without a reserve path.

## Recommended integration

Use the selected checkpoint with the separate r99.9 calibration, `max_source_matches=8192`, and no reserve. A refresh period of eight is supported: the model was trained across every 0-7 prefix phase and was audited on both rolling-8 and rolling-64 reconstructions. The downstream search branch should treat this matcher as a high-recall candidate generator and apply its own ranking/value logic after structural decoding.

If a production workload has substantially larger eligible-pair counts than these audits, re-fit r99.9 calibration on a representative held-out set and re-run the exact-match audit before reducing the 8192 cap.
