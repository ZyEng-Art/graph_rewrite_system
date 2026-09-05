# Hierarchical node actor pretraining (H100, 2026-09-05)

## Training objective

The paged base model is frozen. Only the hierarchical actor's node-scoring
modules are trained. Each prefix is supervised with the anchor of the exact
next action from the saved trajectory. Its cross-entropy weight is:

```text
min(4, 1 + 0.25 * max(0, current gates - future best gates))
```

This is behavior cloning with a best-so-far return weight. It prioritizes the
early actions on paths that eventually reduce the circuit, including uphill
actions whose immediate gate delta is positive. It is not yet an on-policy PPO
return target.

The training corpus combines general random/local trajectories with converted
Quarl Barenco and GF optimization paths. It has 26,924 training action states
and 2,048 held-out action states. The encoder checkpoint is
`paged_action_quarl_all_holdout_r4_w64_aw4_lw2_bw1_m1_lr2e5_bs32_s274.pt`.

## Held-out target-anchor recall

| Epoch | Top-1 | Top-4 | Top-8 | Top-16 | Top-32 | Test loss |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (random head) | 1.32% | 4.05% | 6.93% | 10.94% | 16.41% | 5.7849 |
| 1 | 22.80% | 64.31% | 75.78% | 80.32% | 85.06% | 3.1608 |
| 2 | 23.58% | 65.14% | 77.25% | 81.54% | 85.84% | 3.0949 |
| 3 | 22.61% | 65.67% | 77.39% | 82.32% | 86.62% | 3.0980 |
| 4 | 21.14% | 60.50% | 75.54% | **82.57%** | **87.06%** | 3.2299 |

Training Top-16 at epoch 4 is 84.54%, only 1.97 points above held-out. K=16 is
the practical default: K=8 has begun to overfit, while K=32 buys another 4.49
points at a larger candidate set. The exact-positive metric reaches nearly
100% even at Top-1 because most nodes anchor some legal identity or local rule;
it is not a useful policy-quality metric. Target-action anchor recall is the
relevant number.

The four-epoch run took 1,555.3 seconds (25.9 minutes), versus the roughly
eight-hour per-circuit Quarl fine-tuning target that this work is intended to
replace. This one node model is shared across all circuits in the combined
corpus and held-out split.

## Trained-head candidate check

On a batch of 224 identical `barenco_tof_10` root states (450 gates), the
untrained head previously found no exact candidate at K=8. The trained head
finds 40, 80, and 161 exact source-binding candidates per state at K=8, 16,
and 32. At K=16:

| Path | States/s | Peak allocated GiB | Candidates/state |
|---|---:|---:|---:|
| Full all-node matcher | 3,470 | 1.761 | 512 (cap) |
| Trained node-first K=16, pattern Top-16 | 3,848 | 0.867 | 80 |

This is a 1.109x candidate-generation speedup and a 50.8% reduction in peak
allocated memory. On the 58-gate `barenco_tof_3`, K=16 is effectively
throughput-neutral (1.014x) while reducing peak allocation from 0.290 to 0.204
GiB. Python topology collation remains the dominant runtime stage.

## Prefix-length bucketing

The offline encoder executes up to the longest prefix in each batch. Randomly
mixing lengths therefore makes almost every training batch run all 64 action
steps. A length-bucket batch sampler groups prefixes in width-eight ranges,
then shuffles samples and batches deterministically.

H100 A/B with the same 4,096 train states, 512 test states, batch 32 and seed
941:

| Training batches | Total train/eval seconds | Test Top-16 |
|---|---:|---:|
| Random | 39.281 | 64.65% |
| Length bucket width 8 | 31.456 | 65.23% |

Bucketing is 1.249x faster with equivalent held-out quality.

## Remaining risk

A held-out per-step Top-16 target recall of 82.57% is not enough to reproduce a
long optimization path action by action. The next stage must use this head as
initialization, retain K=32 or an exploration quota where needed, and optimize
the factorized node/action policy with on-policy PPO returns. The candidate
path still runs exact structural decoding and periodic Quartz refresh, so a
node miss reduces exploration but cannot introduce an invalid rewrite.

Remote checkpoint:

```text
/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/
  hierarchical_node_allpaths_rtg025_e4_s940.pt
```

Raw logs:

- `hierarchical_node_allpaths_rtg025_e4_s940.training.json`
- `hierarchical_node_batching_random_4096_512.json`
- `hierarchical_node_batching_bucket8_4096_512.json`
- `hierarchical_matching_trained_barenco_tof_3_b224_h100.json`
- `hierarchical_matching_trained_barenco_tof_10_b224_h100.json`
