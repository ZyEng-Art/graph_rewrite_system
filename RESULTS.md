# s0 + action-binding experiment

## Task and leakage boundary

Input is the complete initial graph `s0` plus an action prefix. Every action has
the rewrite id and ordered source binding. Destination slots are allocated in
destination-pattern order, so they are derived action state rather than a Quartz
teacher label. Neither a materialized `s_t` nor the recorded Quartz graph delta is
fed to the model.

The predicted match identity is:

```text
(source_pattern_id, anchor_slot, ordered_binding_slots)
```

The source pattern contains the gate and qubit-port graph, so the ordered binding
also fixes the complete pattern-node/port-to-circuit-node/port mapping. Applicable
destination rewrites are recovered from the source-to-xfer table.

## Data and training

- Exact Quartz vocabulary: 4700 rewrites and 3855 distinct source patterns.
- Train: 4096 states from `barenco_tof_3`, `mod5_4`, `tof_4`, and `vbe_adder_3`.
- Held-out test: 512 states from `gf2^4_mult` and `hwb6`.
- Trajectory length: 16; local-continuation probability: 0.9.
- Test labels: 486,631 complete source bindings.
- Model: 5,782,607 parameters; 20 epochs on one H100; about 24 minutes.

The in-place discrete updater reproduced all nodes and port-labelled edges for all
4608 dataset transitions. The structural binding decoder reproduced all 1,889,859
observed true ordered bindings exactly.

## Held-out result

`TopN` is micro recall after emitting N predictions, where N is the exact number
of matches in that state.

| retrieval policy | complete TopN | streak >= 2 | streak >= 4 | model + binding decode |
|---|---:|---:|---:|---:|
| retrieve N | 89.28% | 89.13% | 88.98% | 9.31 ms/state |
| retrieve 2N, retain first N structurally valid | **93.41%** | **93.26%** | **93.10%** | 12.63 ms/state |

The held-out Quartz global exact-binding enumeration averaged 114.28 ms/state.
The new in-place action update averaged 0.0425 ms/action. Thus the 2N pipeline is
about 9.0x faster than exact global matching on this test, before charging Quartz
for graph copy/replacement.

The 2N policy uses only internal gate/port connectivity as its cheap structural
filter; it does not call Quartz exact pattern matching.

## Artifacts

- Remote data: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/data/binding_4096_512_v2.pt`
- Remote checkpoint: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/s0_binding_4096_discrete.pt`
- Remote metrics: `runs/s0_binding_4096_final_{x1,x2}_metrics.json`

Train:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ../../data/binding_4096_512_v2.pt \
  --output ../../runs/s0_binding_4096_discrete.pt \
  --epochs 20 --batch-size 16 --eval-batch-size 4 \
  --width 192 --retrieval-width 128 \
  --graph-layers 2 --current-graph-layers 6 \
  --binding-weight 1.5
```

Evaluate the 2N policy:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_checkpoint.py \
  --data ../../data/binding_4096_512_v2.pt \
  --checkpoint ../../runs/s0_binding_4096_discrete.pt \
  --batch-size 4 --candidate-multiplier 2
```

## Current boundary

The experiment establishes the requested oracle-N TopN result. A production
replacement still needs match-count/threshold calibration because rollout does
not know the exact N in advance, plus end-to-end integration into Quarl's batched
rollout and an exact fallback for low-confidence states.

## Long-sequence 16K run

A second run targets the requested longer rollouts and removes the unnecessary
continuous action replay. The current graph is maintained exactly, in place, from
`s0 + actions`; the neural network encodes that derived graph and trains against
structurally valid hard negatives.

- Regular component: 8192/1024 states, 32 actions, local probability 0.80.
- Local component: 8192/1024 states, 64 actions, local probability 0.98.
- Merged train/test: 16,384/2048 non-terminal states.
- Test with terminal states: 2096 states and 2,143,604 complete bindings.
- Approximately 6.54 million complete training bindings.
- Model: 5,480,591 parameters, 15 epochs, one H100, about 30 minutes.
- Dataset size: 745 MB; checkpoint size: 23 MB.

All 18,432 long-trajectory graph transitions reproduced the exact Quartz nodes
and port-labelled edges. The tensor structural decoder reproduced all 2,143,604
held-out true bindings, including terminal states, exactly.

| policy | all | prefix 0–15 | prefix 16–31 | prefix 32–63 | full 64 actions | streak >= 4 | total time |
|---|---:|---:|---:|---:|---:|---:|---:|
| retrieve N | **94.79%** | 95.19% | 94.76% | 94.35% | 94.34% | 94.66% | 3.14 ms/state |
| retrieve 2N, retain N structurally valid | **96.62%** | 97.04% | 96.49% | 96.30% | 96.19% | 96.55% | 6.46 ms/state |

The two exact Quartz test components average 105.85 ms/state. This corresponds
to about 33.8x speedup for direct N and 16.4x for the higher-accuracy 2N policy.
The measured in-place state transition remains 0.0425 ms/action.

Long-run remote artifacts:

- Dataset: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/data/binding_longmix_16384_2048_v2.pt`
- Checkpoint: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/s0_binding_longmix_16k_stateonly_hard.pt`
- Metrics: `runs/s0_binding_longmix_final_{x1,x2}_terminal_metrics.json`

Training command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data ../../data/binding_longmix_16384_2048_v2.pt \
  --output ../../runs/s0_binding_longmix_16k_stateonly_hard.pt \
  --epochs 15 --batch-size 32 --eval-batch-size 8 \
  --width 192 --retrieval-width 128 \
  --current-graph-layers 6 --binding-weight 0 \
  --structural-hard-negatives --state-only
```

## Accuracy around the previous rewrite

This audit excludes the 48 initial states because they have no previous rewrite.
It covers 2048 held-out states and 2,100,308 true complete bindings. The affected
core is the previous rewrite's still-live destination nodes plus the still-live
endpoints of its changed boundary edges. Distances are undirected graph distance
in the current circuit. The recorded Quartz delta is used only to define this
evaluation region; it is not a model input.

The complete-binding distance is the minimum distance over every circuit node in
the ordered binding. Recall and precision are micro-averaged over matches.

| complete-binding distance | true matches | N recall | N precision | 2N→N recall | 2N→N precision |
|---|---:|---:|---:|---:|---:|
| overlaps affected core (0) | 123,345 | 92.15% | 94.64% | 95.23% | 91.45% |
| 1 hop | 64,572 | 90.15% | 94.43% | 93.65% | 92.16% |
| 2 hops | 84,234 | 91.33% | 95.45% | 94.62% | 93.77% |
| 3+ hops | 1,828,157 | 95.27% | 98.29% | 96.89% | 97.26% |
| aggregate within 2 hops | 272,151 | 91.42% | 94.84% | 94.66% | 92.32% |

The 1-hop band is the hardest. Under direct-N retrieval its 9.85% miss rate is
2.08x the 4.73% miss rate at 3+ hops. Retrieving 2N candidates and retaining N
structurally valid candidates raises within-2-hop recall by 3.24 percentage points,
but a 2.24-point near/far recall gap remains.

For within-2-hop bindings, grouping by the local streak of the action that just
produced the current state gives:

| previous local streak | true matches | N recall | 2N→N recall |
|---|---:|---:|---:|
| 0 | 34,953 | 92.01% | 94.96% |
| 1 | 28,638 | 91.98% | 94.98% |
| 2–3 | 43,613 | 91.87% | 94.77% |
| 4+ | 164,947 | 91.08% | 94.52% |

Thus repeated local rewriting causes an additional measurable degradation: about
0.94 point for direct N and 0.45 point for 2N→N between streak 0 and streak 4+.
The dominant issue, however, is proximity to the most recent rewrite itself.

Locality metrics: `runs/s0_binding_longmix_locality_{x1,x2}.json`.

## Locality-aware fine-tuning

The state-only model now optionally embeds three features derived while replaying
`s0 + actions`:

- current-graph distance to the most recent rewrite core;
- age since each live node was last created or touched by a rewrite boundary;
- observed consecutive locality of the action sequence.

No recorded Quartz delta or future match is used to construct these features. On
all 2048 held-out post-action states, the core reconstructed from the action and
the incremental graph exactly matched the core computed from the recorded Quartz
delta. The feature model has 5,632,271 parameters, 151,680 more than the baseline.

Five six-epoch fine-tunes were run from the long-sequence checkpoint on separate
H100s. They compared features alone, loss weighting alone, and feature-plus-near-
positive weights of 0.5, 1.0, and 2.0. The balanced checkpoint uses weight 1.0.

| policy | overall before | overall after | within-2 recall before | within-2 recall after | within-2 precision before | within-2 precision after |
|---|---:|---:|---:|---:|---:|---:|
| retrieve N | 94.79% | **94.93%** | 91.42% | **92.15%** | 94.84% | **95.26%** |
| retrieve 2N, retain N valid | 96.62% | **96.71%** | 94.66% | **95.15%** | 92.32% | **92.98%** |

For the balanced checkpoint, the direct-N 1-hop recall rises from 90.15% to
91.12%, and the 2N-to-N 1-hop recall rises from 93.65% to 94.41%. Within-2-hop
recall after a previous local streak of 4+ rises from 91.08% to 91.99% for direct
N and from 94.52% to 95.10% for 2N-to-N.

An optional direct-N-specialized checkpoint uses weight 2.0. It reaches 94.97%
overall, 92.33% within two hops, 91.39% at exactly one hop, and 92.15% within two
hops after streak 4+. Its 2N-to-N global result is weaker, so it is not the
balanced default.

Isolated balanced-checkpoint timing is 3.98 ms/state for direct N and 6.76
ms/state for 2N-to-N. Against the 105.85 ms/state Quartz exact matcher, these are
about 26.6x and 15.7x faster respectively.

Locality-aware remote artifacts:

- Balanced checkpoint: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/locality_ft_feat_w1.pt`
- Direct-N-specialized checkpoint: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/locality_ft_feat_w2.pt`
- Balanced terminal metrics: `runs/locality_ft_feat_w1_terminal_x{1,2}_isolated.json`
- Detailed distance metrics: `runs/locality_ft_feat_w1_locality_x{1,2}.json`

## Oracle-N-free threshold inference

Near- and far-rewrite logits were independently calibrated on 512 held-out
states. Thresholds targeting 97% calibration recall were then evaluated on a
disjoint 1572-state split. This inference path never reads the true match count
N and has a per-state cap of 2048 source matches.

| states | actual complete-binding recall | precision before Quartz validation | mean predictions/state | model + threshold + decode |
|---:|---:|---:|---:|---:|
| 1572 | **96.53%** | **96.44%** | 1049.98 | 2.41 ms/state |

The configured 97% value is a target recall used to select thresholds, not a
claim that every emitted candidate has 97% probability. The observed recall on
the disjoint evaluation split is 96.53%.

Threshold artifacts are in `benchmark_results/locality_ft_feat_w1_{calibration,threshold_r0p97}.json`.

## Buffer search and original-Quartz action throughput

`beam_search_benchmark.py` implements the proposed search prototype:

1. predict every `(source pattern, anchor, complete ordered binding)` above the
   calibrated threshold for every circuit in the current buffer;
2. expand each source pattern to all applicable destination rewrites, covering
   all 4700 Quartz xfers;
3. rank candidates using the known source/destination gate-count delta before
   materializing a successor;
4. apply only top proposals through Quartz, validate the predicted complete
   binding, deduplicate by Quartz graph hash, and retain the best bounded buffer;
5. optionally run original exact matching on the best K states every L actions,
   adding only actions missed by the model.

Both modes share proposal caps, Quartz application, graph copying, hashing, and
deduplication. The only benchmarked difference is action matching: model mode
uses the neural predictor, while the baseline calls Quartz's original
`available_xfers_parallel` for every node. Measurements used one H100 and 32
OpenMP CPU threads.

| circuit / search | matcher | matching states/s | enumerated actions/s | end-to-end accepted actions/s | wall time | best gates |
|---|---|---:|---:|---:|---:|---:|
| `hwb6`, buffer 1000, depth 3 | model | **281.83** | **400,483** | **109.28** | **20.98 s** | 259 -> 253 |
| `hwb6`, buffer 1000, depth 3 | original Quartz | 9.72 | 13,765 | 15.27 | 151.05 s | 259 -> 253 |
| `gf2^4_mult`, buffer 256, depth 3 | model | **225.62** | **244,570** | **146.61** | **4.49 s** | 225 -> 219 |
| `gf2^4_mult`, buffer 256, depth 3 | original Quartz | 11.90 | 12,997 | 17.04 | 42.90 s | 225 -> 219 |

On `hwb6`, match enumeration is 29.0x faster and the complete three-layer
search is 7.20x faster by wall time. On `gf2^4_mult`, the corresponding numbers
are 18.9x and 9.56x. Model and exact search reached the same best gate count in
both controlled runs. Among the top proposals actually attempted by model mode,
the exact Quartz application rejected 41/10,007 on `hwb6` and 6/1813 on
`gf2^4_mult`; no proposal was rejected after the first layer in either run.

The application stage itself is now the bottleneck: both modes sustain about
690--790 successful Quartz copies/rewrites per second. In the full `hwb6` third
layer, model matching took 3.56 seconds, while applying candidates and rejecting
duplicate successor graphs took 8.54 seconds. Further speedup therefore requires
reducing duplicate materializations or replacing per-candidate Quartz graph copy,
not only improving matcher throughput.

### Periodic exact refresh

On `hwb6` with buffer 1000, exact refresh after four actions on the best 100
states added 3810 missed actions, or 38.1 per refreshed state. It cost 10.44
seconds. The corrected merge keeps model confidence for already-predicted
actions and assigns exact matching only a recovery role. At depth 6 it reached
252 gates, identical to no refresh; total time was 59.91 seconds versus 49.15
seconds without refresh through the same depth.

This supports sparse refresh as a recall safety net, but refreshing 10% of a
1000-state buffer every four actions is too expensive for the default setting.
The benchmark leaves refresh disabled unless `--refresh-interval` is supplied.

Local benchmark artifacts are under `benchmark_results/`; remote copies and
best-QASM outputs are under
`/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/`.

## Lazy sequence rollout and GPU batching

`lazy_rollout_benchmark.py` implements the sequence-state version of the search.
There is no Quartz graph in the ordinary rollout loop. A child is represented by
appending `(xfer_id, complete source binding, deterministic destination slots)`;
its gate count is updated from the rule delta. Because the current checkpoint's
binding decoder follows graph ports, the prototype also maintains a lightweight
gate-type/port topology. It does not create or copy a Quartz graph and does not
track rotation parameters during speculative rollout.

The cheap `raw` dedup mode hashes that lightweight state. Duplicate proposals are
rejected before constructing locality features or a complete child state. Quartz
is used only after timed search to replay trajectories, validate applicability,
compare the resulting exact topology, calculate exact graph hashes, and export a
valid best QASM.

### Three-layer end-to-end comparison

The model/hybrid rows below include batched model prediction plus real Quartz
copy/apply. Lazy time excludes the explicitly reported out-of-band replay audit.
All three modes use the same gate-delta proposal ranking and reach the same best
gate count.

| circuit / buffer | original Quartz | batched hybrid | lazy rollout | Quartz / lazy | best gates |
|---|---:|---:|---:|---:|---:|
| `hwb6` / 1000 | 151.05 s | 19.73 s | **8.88 s** | **17.0x** | 259 -> 253 |
| `gf2^4_mult` / 256 | 42.90 s | 3.78 s | **2.04 s** | **21.0x** | 225 -> 219 |

Removing per-action Quartz copy/apply adds another 2.22x over hybrid on `hwb6`
and 1.86x on `gf2^4_mult`. On the full third `hwb6` layer, the measured times are
113.33 seconds for original Quartz, 12.61 seconds for hybrid, and 6.00 seconds
for lazy, so the steady-layer Quartz/lazy speedup is 18.9x.

The exact audit results were:

| run | audited | Quartz-valid | exact topology agreement among valid | unique exact hashes | replay time |
|---|---:|---:|---:|---:|---:|
| `hwb6`, depth 3 | 1000 | 943 (94.3%) | 943/943 | 920 | 3.95 s |
| `gf2^4_mult`, depth 3 | 256 | 251 (98.0%) | 251/251 | 117 | 0.91 s |
| `hwb6`, depth 8 | 1000 | 1000 (100%) | 1000/1000 | 786 | 8.51 s |
| `hwb6`, depth 32 | 1000 | 1000 (100%) | 1000/1000 | 607 | 29.59 s |
| `hwb6`, depth 64 | 1000 | 1000 (100%) | 1000/1000 | 595 | 47.20 s |

All depth-3 failures came from a false first action. On `hwb6`, those speculative
branches were naturally removed by later ranking: the depth-8 final buffer is
fully valid and reaches 252 gates in 34.46 seconds of search. Including one full
depth-8 replay checkpoint gives 42.97 seconds total, or about 1.06 seconds of
amortized replay cost per layer.

The longer `hwb6` runs remain stable. Depth 32 searches in 181.75 seconds and
depth 64 searches in 263.32 seconds; both retain a full 1000-state beam, replay
every final trajectory successfully, agree exactly with the replayed Quartz
topology, and keep the best result at 252 gates. The depth-64 run was measured
on an otherwise idle H100 in `h100-gpu5`, so its wall time should not be compared
directly with the depth-32 run on a more heavily shared host. Its steady-layer
model throughput does not degrade with the longer action history.

The exact-hash counts expose an important remaining limitation. The lightweight
hash does not include Quartz rotation parameters and cannot identify every pair
of different-slot sequences that materialize the same circuit. With dedup
disabled, an append-only three-layer `hwb6` run retained only 32 distinct exact graphs
among 1000 trajectories. Thus production search should keep cheap dedup enabled,
oversample before refresh, and perform exact parameter-aware dedup at checkpoints.

### Batch-size sweep

The static 3855-source rule vectors are computed once per rollout call. On the
same 1000 `hwb6` states, including graph encoding, thresholding, and complete
binding decode, H100 throughput is:

| microbatch | states/s | relative to batch 1 |
|---:|---:|---:|
| 1 | 111.8 | 1.00x |
| 2 | 175.8 | 1.57x |
| 4 | 245.1 | 2.19x |
| 8 | 304.1 | 2.72x |
| 16 | 345.0 | 3.09x |
| 32 | 370.5 | 3.31x |
| 64 | 386.2 | 3.46x |
| 128 | 393.9 | 3.52x |
| 256 | 396.5 | 3.55x |
| 512 | **400.6** | **3.58x** |

The curve largely saturates after microbatch 128. Against the original Quartz
full-buffer matcher's 9.72 states/s, the batch-512 model matching path is about
41.2x faster. End-to-end speedup is lower because Python proposal sorting and
lightweight successor/dedup work are now the main costs.

Lazy artifacts:

- `benchmark_results/lazy_hwb6_b1000_d3_mb512_fastdedup.json`
- `benchmark_results/lazy_hwb6_b1000_d8_mb512_fastdedup.json`
- `benchmark_results/lazy_hwb6_b1000_d32_mb512_fastdedup.json`
- `benchmark_results/lazy_hwb6_b1000_d64_mb512_fastdedup_gpu5.json`
- `benchmark_results/lazy_gf2_4_b256_d3_mb256_fastdedup.json`
- `benchmark_results/lazy_hwb6_b1000_batch_sweep_cached.json`
