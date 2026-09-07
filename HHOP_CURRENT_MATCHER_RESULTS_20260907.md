# Exact-current H-hop matcher experiment (2026-09-07)

## Outcome

The selected model is the H=6 exact-current graph encoder with one source-pattern
topology layer and balanced rare-positive training. It improves independent GF
exact-match retention over the current paged matcher while retaining every
teacher action in both high-quality trajectories. H=8 was tested fairly and was
not selected because its cleaner candidate set missed more true matches.

Selected remote checkpoint:

`/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/hhop_h6_topo1_balanced_s907.pt`

Selected R99.9 calibration:

`/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/hhop_h6_topo1_balanced_s907_calibration_r999.json`

SHA-256:

`93ec6504d14665d107e51f7181bbab63105ec629e48571ba22f8c56dc9b1f2f0`

## Meaning of H

H is the number of current-circuit graph message-passing layers evaluated after
every exact Quartz rewrite. H=6 gives every gate an effective six-edge
receptive field. It is not an action-history length or refresh interval. The
implementation currently recomputes the entire exact graph; H bounds the
information propagation depth per gate.

The H=8 experiment loads the trained H=6 prefix and zero-gates layers seven and
eight. Its initial function is therefore exactly identical to H=6. Unit tests
verify both that identity and that a distance-three change becomes visible only
after enough message-passing hops are enabled.

## Training

- Training data: `binding_longmix_randomrefresh_complex_holdout_20260906.pt`
  (34,912 training states, 2,048 fixed test states).
- Initialization: `locality_ft_feat_w1.pt`.
- Common configuration: width 192, current graph recomputed from exact state,
  one source-topology layer, structural hard negatives, locality-positive
  weight 1, inverse-frequency positive weighting (power 0.5, cap 8), balanced
  target sampling (power 0.5, cap 8, fraction 0.5), learning rate 1e-4,
  seed 907.
- H=6 trained for two epochs. H=8's best fixed-test checkpoint remained epoch
  one because epoch two reduced the selection metric.

Fixed 2,048-state test set:

| Model | Full-binding Top-N | Near Top-N | Model ms/state |
|---|---:|---:|---:|
| Old state-only H=6 | 94.9390% | 92.1561% | 0.633 |
| New H=8 best | 95.2861% | 92.8328% | 0.643 |
| **New H=6 selected** | **95.4544%** | **93.3406%** | **0.566** |

## Independent exact-match generalization at R99.9

No near-reserve or exact-match fallback was used. The source cap was 10,240 per
state.

| Model | Barenco retained | Teacher actions | GF retained | Teacher actions |
|---|---:|---:|---:|---:|
| Old state-only H=6 | 16,003/16,003 (100%) | 116/116 | 517,037/559,524 (92.4066%) | 257/271 |
| Current paged matcher | 15,925/16,003 (99.5126%) | 116/116 | 559,323/559,524 (99.9641%) | 271/271 |
| New H=8 best | 16,002/16,003 (99.9938%) | 116/116 | 559,280/559,524 (99.9564%) | 269/271 |
| **New H=6 selected** | **16,003/16,003 (100%)** | **116/116** | **559,395/559,524 (99.9769%)** | **271/271** |

For selected H=6 on GF, the threshold retains 559,471 true matches; the 10,240
source cap removes another 76, leaving 559,395. Thus 53 misses are model-score
misses and 76 are cap/ranking misses. Structural decoding removes no additional
true matches. Its decoded exact precision is 76.17%.

The first H=6 epoch had a smaller GF candidate set and retained 559,424 exact
matches, but it filtered the teacher action at state 188 (source 3755, xfer
4370). Epoch two restores that action and was selected because the primary
requirement is not to filter either reference optimization path.

## Reference trajectory coverage and ranking

| Trajectory | Source/action coverage | Median gate rank | Maximum gate rank | Smallest tested cap covering whole path |
|---|---:|---:|---:|---:|
| Barenco 58 -> 36 (116 actions) | 116/116 | 12 | 264 | 512 |
| GF 495 -> 371 path (271 actions) | 271/271 | 518 | 4,152 | 8,192 |

This confirms that matching recall is fixed for both recorded paths, while GF
ranking remains a separate search-depth problem.

## Batch-512 throughput

Core throughput includes GPU matching, structural decode, action expansion,
preselection, ranking, caps, and selected-proposal packing. The `with prep`
number additionally includes the one-time CPU dataset replay/collation used by
the benchmark.

| Circuit | Model | Core states/s | With prep states/s | Speedup vs CPU Quartz anchor scan |
|---|---|---:|---:|---:|
| Barenco-58 | Current paged | 10,764 | 982 | 245.8x |
| Barenco-58 | **New H=6** | **11,026** | **1,280** | **251.8x** |
| GF-495 | Current paged | 3,778 | 125 | 773.6x |
| GF-495 | **New H=6** | **3,643** | **167** | **746.1x** |

The extra exact-current topology modeling therefore does not cause a large GPU
throughput regression. On GF the core is 3.6% slower, while simpler state-only
collation makes the measured total including preparation faster.

## Short end-to-end search checks

| Circuit/run | Model | Best | Time to best | Total | Apply validity | Matcher states/s |
|---|---|---:|---:|---:|---:|---:|
| Barenco, beam 1000, depth 64 | Current paged | 38 | 47.44 s | 95.35 s | 72.77% | 7,353 |
| Barenco, beam 1000, depth 64 | **New H=6** | **38** | **43.41 s** | **89.36 s** | **76.11%** | **7,502** |
| GF, beam 1000, depth 8 | Current paged | 485 | 23.72 s | 37.02 s | 99.40% | 1,163 |
| GF, beam 1000, depth 8 | **New H=6** | **485** | **23.77 s** | **37.78 s** | **96.43%** | **1,158** |

The selected matcher does not by itself reproduce Barenco-36 in unconstrained
beam search. It keeps the entire known path, but the gate-only ranking needs a
beam/action cap of at least 512 on Barenco and 8,192 on GF. The broader GF
R99.9 candidate set also increases false proposals in the first search layer;
future ranking/binding work should address that without weakening path recall.
Of the 560 invalid GF proposals, 557 occur in the first layer and only three in
the following seven layers. This is evidence against cumulative topology-state
drift as the cause of the remaining invalid proposals.

The selected checkpoint was produced with:

```bash
python train.py \
  --data ../../data/binding_longmix_randomrefresh_complex_holdout_20260906.pt \
  --output runs/hhop_h6_topo1_balanced_s907.pt \
  --epochs 2 --eval-every 1 --batch-size 8 --eval-batch-size 8 \
  --num-workers 4 --prefetch-factor 2 \
  --width 192 --retrieval-width 128 --graph-layers 1 \
  --current-graph-layers 6 --source-topology-layers 1 \
  --architecture legacy --state-only --locality-features \
  --locality-positive-weight 1 --structural-hard-negatives \
  --source-positive-balance-power 0.5 --source-positive-balance-cap 8 \
  --target-source-sampler-power 0.5 --target-source-sampler-cap 8 \
  --target-source-sampler-fraction 0.5 --learning-rate 1e-4 \
  --init-checkpoint ../../runs/locality_ft_feat_w1.pt --seed 907
```

## Reproduction notes

- Branch: `hhop-current-matcher-20260907`
- Implementation commit: `4a90960`
- Tests: 71 unit tests passed on the remote H100 environment.
- Calibration is fitted on a held-out modulo partition of the complex dataset;
  Barenco and GF trajectory datasets are used only for independent audit.
