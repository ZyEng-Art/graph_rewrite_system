# Paged causal-action results

## Baseline audit

The previous rollout checkpoint `locality_ft_feat_w1.pt` declares
`state_only=True`. Its reported accuracy and rollout validity are current-graph
results, not action-token/KV-cache results. The previous implementation is
preserved unchanged in `../s0_binding_worldmodel`.

## Correctness tests

- Mixed prefix lengths 0, 1, 7, and 31 reconstruct exactly the same live slots
  and gate types as the graph derived independently from the actions.
- Every sampled true `(source, ordered binding)` is reproduced by the structural
  decoder.
- Full-prefix causal encoding and 31 successive one-token updates agree within
  `atol=rtol=2e-5`; live slots and gate types agree exactly.
- GPU page tests cover full-page sharing, copy-on-write tails, contiguous gather,
  reference-count reclamation, and zero leaked pages.

## Beam integration smoke test

The first end-to-end test uses a deliberately undertrained pilot checkpoint,
beam 32, page size 4, and depth 5. It is a plumbing test, not an optimization
quality result.

- 32/32 final trajectories replay successfully in Quartz.
- 32/32 lazy topologies equal the replayed exact topologies.
- At depth 5, 64 logical prefix pages occupy 40 physical pages: a 1.60x sharing
  ratio after the first complete four-action page.
- Search takes 0.697 seconds; exact replay audit takes 0.151 seconds.

Remote artifact:
`/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/paged_action_smoke_rollout_d5.json`.

## Training experiments

The pure final-cross-attention v1 plateaued early and was stopped after epoch 3:

| epoch | overall TopN | prefix 32--63 TopN | within-2-hop TopN |
|---:|---:|---:|---:|
| 1 | 76.12% | 72.48% | 52.36% |
| 2 | 77.49% | 74.11% | 55.56% |
| 3 | 77.92% | 74.36% | 57.45% |

The v2 model retains causal action K/V and additionally updates a cached node
state once per new action. It still does not run or consume a current-graph GNN.

| epoch | overall TopN | prefix 32--63 TopN | within-2-hop TopN |
|---:|---:|---:|---:|
| 1 | 77.33% | 73.56% | 57.78% |
| 2 | 78.63% | 75.36% | 60.95% |
| 3 | 78.95% | 75.66% | 62.46% |
| 4 | 79.28% | 76.22% | 63.40% |
| 5 | **79.36%** | **76.17%** | **63.41%** |

The final v2 checkpoint is `paged_action_recurrent_v2.pt`; epoch 4 is also
frozen separately as `paged_action_recurrent_v2_epoch4.pt` for the v3 warm
start. Five epochs took about 1,098 seconds of training time on one H100, plus
held-out evaluation after each epoch.

Over-retrieval followed by exact lightweight structural decoding helps, but
does not close the gap:

| retrieved before filtering | retained output | overall recall | within-2-hop recall | model + decode |
|---:|---:|---:|---:|---:|
| N | N | 79.36% | 63.41% | 6.23 ms/state |
| 2N | N | 83.04% | 74.52% | 9.74 ms/state |
| 4N | N | 84.78% | 78.63% | 10.54 ms/state |
| 8N | N | 85.46% | 80.18% | 17.19 ms/state |

Therefore candidate oversampling alone cannot honestly claim the requested 90%
Top-N result.

The v3 experiment adds an ordered pattern-role residual to every source binding
before pooling and recurrent node updates. Its final projection is zero
initialized, so loading v2 reproduces v2 exactly before fine-tuning. Enable it
with `--ordered-binding-roles`.

V3 was stopped after epoch 2 because it reached only 79.44% overall, 63.94%
within two hops, and 76.44% at prefix lengths 32--63. The change is retained as
an ablation checkpoint (`paged_action_ordered_v3.pt`), but the gain over v2 is
only 0.08 percentage points.

V4 retains the causal cache but adds two current-light-graph message-passing
layers at readout. The graph is deterministically maintained from actions; this
path never calls Quartz and does not copy/apply a Quartz graph. After one epoch
it reaches 83.32% overall, 74.98% within two hops, and 81.47% at prefix lengths
32--63. It was stopped after epoch 3 at 84.53% overall, 78.79% within two hops,
and 82.69% at prefix lengths 32--63. A four-layer v5 was warm-started from the
frozen v4 epoch-1 checkpoint to measure the accuracy/throughput tradeoff. Its
fourth epoch reaches 86.16% overall, 81.49% within two hops, and 84.80% for the
long-prefix bucket. The model-only time is 4.01 ms/state and total model plus
structural decode is 6.27 ms/state at evaluation batch size 32.

For the v5 epoch-4 checkpoint, retrieving 2N neural candidates and retaining
the first N structurally valid complete bindings reaches 89.68% overall,
87.72% within two hops, and 88.79% for prefix lengths 32--63. The final output
size is capped at N. The lower-learning-rate continuation was stopped after its
fifth epoch; the best direct-N checkpoint occurred at continuation epoch 4,
but the best 2N-to-N checkpoint remained continuation epoch 2.

After the first continuation epoch, direct N reaches 86.48%. The 2N-to-N result
is 89.904% overall, 88.19% within two hops, and 89.11% at prefix lengths 32--63.
This is still below the requested 90% and is not rounded up.

After the second continuation epoch, the held-out 2N-filtered result crosses the
target: **90.10%** overall on 2,048 states and 2,090,005 true complete bindings.
It emits 99.866% of N overall; 255 states have slightly fewer than N valid rows
inside the first 2N candidates. Recall is 90.65% for prefix lengths 0--15,
90.19% for 16--31, 89.26% for 32--63, and 88.40% within two hops of the latest
rewrite. The exact milestone checkpoint is frozen as
`paged_action_localgraph4_v5_cont_epoch2.pt`; its metrics are in
`paged_action_localgraph4_v5_cont_epoch2_x2.json`.

Increasing the internal candidate pool to 5N fills every state exactly: all
2,090,005 output slots are populated and `states_with_fewer_than_n=0`. This
strict N-output policy reaches **90.17%** overall, 88.50% within two hops, and
89.32% for prefix lengths 32--63. Model plus structural filtering takes 15.34
ms/state at batch size 32. The intermediate 3N and 4N settings miss 80 and one
output slots respectively, so 5N is the first measured strictly full-N policy.

The final checkpoint's staged path uses 16,384 training states (about 6.54
million exact complete bindings), trajectory lengths 32--64, 6.93 million
parameters, and about 44 minutes of H100 training kernels in total; held-out
evaluation after each stage is additional wall time.

The complete locality audit (including terminal depth-64 states) shows where
the remaining errors occur:

| binding distance from latest rewrite | recall | precision |
|---|---:|---:|
| overlaps affected core | 91.36% | 77.41% |
| 1 hop | 86.18% | 81.03% |
| 2 hops | 85.61% | 85.56% |
| 3+ hops | 90.32% | 91.79% |

For within-two-hop bindings after a local streak of 4 or more, recall is 88.62%
and precision is 77.78%. Bindings overlapping newly created nodes have 93.69%
recall. Thus the main residual error is one/two-hop ranking and local false
positives, not identity of newly allocated nodes. Full metrics are in
`paged_action_localgraph4_v5_cont_epoch2_locality_x2.json`.

V6 keeps the action/KV branch but feeds clean current gate-type embeddings,
rather than the evolving cached action states, into the four topology layers.
This isolates exact local connectivity from accumulated latent-state error. It
uses only the deterministic lightweight graph derived from the action prefix.
It was stopped after epoch 3 at 86.08% direct-N because it trailed the cached-
state continuation; the evolving action state contains useful graph inputs.

V7 applies a small positive-loss weight to bindings within two hops. Its final
2N-to-N result is 89.993% overall and 88.65% within two hops. It improves local
recall by 0.25 point but narrowly misses the global 90% requirement, so it is
retained as a locality-oriented ablation rather than the default.

## Oracle-N-free probability threshold

Thresholds were fitted on 512 held-out states and evaluated on a disjoint 1,572
states. At calibration target 95%, inference reads no true N and obtains 92.06%
recall, 85.90% precision before Quartz verification, and 1,124 predicted
complete bindings per state on average. Model plus threshold/structural decode
takes 10.25 ms/state at batch size 32.

## Paged beam search and exact refresh

On `hwb6`, beam 1000, target-recall 0.95, microbatch 512:

| run | search time | final exact replay | best gates | page sharing |
|---|---:|---:|---:|---:|
| depth 8, no refresh | 28.74 s | 890/1000 valid | 259 -> 253 | 1.00x at page boundary |
| depth 8, refresh every 4 | 52.26 s | 1000/1000 valid and topology-exact | 259 -> 253 | 1.00x at page boundary |
| depth 16, refresh every 4 | 138.77 s | 1000/1000 valid and topology-exact | 259 -> 253 | **1.92x** |

At depth 16 the 1000 beams hold 2000 logical eight-action pages in only 1042
physical pages. The final independent audit takes 12.90 seconds and finds 56
unique exact Quartz graph hashes.

For the first three layers of the depth-16 run, the paged search accepts 2317
children in 7.36 seconds. The preserved original-Quartz batch baseline accepts
2307 in 151.05 seconds, giving a **20.5x end-to-end speedup**. At the full third
layer, paged model matching processes about 423 states/s versus 9.72 states/s
for original Quartz matching, or **43.5x matcher throughput**. Exact refresh is
charged in the depth-8/depth-16 search times above.

## On-policy correction and one-step exploration (v13--v15)

The no-refresh failure is a closed-loop distribution-shift problem. The base
checkpoint still scores well on the fixed held-out set, but 110 of its final
1000 `hwb6` depth-8 trajectories first fail exact Quartz replay at action 4 or
5. Expanding the neural candidate pool from 5N to 16N does not improve recall,
and a TopN boundary loss, six topology layers, and a gated clean-topology branch
do not remove this failure.

An exact DAgger-style collector was therefore added. It searches only the four
training circuits, replays sampled model histories in Quartz, and records every
complete source-pattern binding at the terminal state. The first collection
contains 512 terminal states and 157,030 exact matches. Neither `hwb6` nor
`gf2^4_mult` is used for this fine-tuning. With terminal repeat 8, one H100
epoch sees 20,864 training states and takes 275.2 seconds of training kernels;
held-out evaluation is additional wall time.

The epoch-1 on-policy checkpoint improves direct-N recall from 86.50% to 86.60%
and near-rewrite recall from 82.13% to 82.70%. Its strict 5N-to-N recall is
89.998%, slightly below the base checkpoint's 90.166%. More importantly, its
closed-loop `hwb6` depth-8 audit is 1000/1000 valid, compared with 890/1000 for
the base checkpoint. The stable checkpoint alone finds 255 gates rather than
253, so it is not a complete replacement for the base model.

V15 uses the stable on-policy checkpoint as the primary model and reserves a
few per-parent proposals from the base checkpoint. The exploration model is
enabled only at depth 1. Both models encode `s0` once and use their own paged
action cache; no Quartz graph is copied or applied in the search hot path. The
second forward is negligible because depth 1 has a single parent state.

| `hwb6`, beam 1000, depth 8 | search time | exact replay | best gates |
|---|---:|---:|---:|
| base v5, no refresh | 28.74 s | 890/1000 (89.0%) | 259 -> 253 |
| stable v13, no exploration | 28.72 s | 1000/1000 (100%) | 259 -> 255 |
| **v15, 4 exploration proposals at depth 1** | **28.56 s** | **962/1000 (96.2%)** | **259 -> 253** |
| base v5, exact refresh every 4 | 52.26 s | 1000/1000 (100%) | 259 -> 253 |

The final v15 run reaches 253 gates at depth 3. Its first three layers take
7.38 seconds, versus 151.05 seconds for the preserved original-Quartz batch
baseline, a 20.5x end-to-end speedup. It then continues to depth 8 in 28.56
seconds. The final audit is deliberately outside the reported search time and
checks all 1000 trajectories.

At depth 16 without recovery, v15 takes 66.21 seconds and remains at 253 gates,
but only 823/1000 final trajectories replay exactly; the new failures are
concentrated at actions 14--16. Refreshing at actions 8 and 16 takes 106.18
seconds, returns 1000/1000 exact-valid trajectories, and retains 253 gates.
This is 23.5% faster than the earlier every-four-actions policy at depth 16
(138.77 seconds). The final page-sharing ratio is 1.85x. Therefore the measured
policy is no refresh through depth 8, then exact recovery every 8 actions for
longer trajectories.

On the independent `gf2^4_mult` circuit, beam 256 and depth 8, the base, stable,
and v15 policies all produce 225 -> 219 gates with 256/256 valid exact replays.
Their search times are 7.03, 6.66, and 6.53 seconds respectively, so the
one-step exploration policy causes no measured regression on this circuit.

Local artifacts for the final policy are:

- primary checkpoint: `benchmark_results/paged_action_onpolicy_v13_r8_lr5e5_epoch1.pt`;
- primary calibration: `benchmark_results/paged_action_onpolicy_v13_r8_lr5e5_epoch1_calibration.json`;
- exploration checkpoint and calibration: the recommended base v5 artifacts;
- `hwb6` audit: `benchmark_results/paged_action_dual_v15_q4_until1.json`;
- depth-16 audits: `benchmark_results/paged_action_dual_v15_q4_until1_hwb6_b1000_d16.json`
  and `benchmark_results/paged_action_dual_v15_q4_until1_hwb6_b1000_d16_refresh8.json`;
- independent-circuit audit: `benchmark_results/paged_action_dual_v15_gf2_b256_d8.json`.

## Strict full-prefix versus paged-cache A/B

The earlier state-only and Quartz-apply timings do not isolate the cache. A new
strict A/B therefore uses one `paged_action` checkpoint for both arms and the
same actual `hwb6` beam at every depth:

- full recompute calls `encode(s0, a0, ..., a[t-1])`; this loops through every
  previous action and updates the live node embeddings again;
- paged reuse keeps the node state and per-action K/V, calls the same readout,
  and advances only the new action;
- both arms use the same threshold configuration, current lightweight graph,
  source vectors, complete-binding structural decoder, microbatch 512, and
  1000-state beam after depth 2;
- both arms run under `torch.no_grad()` after CUDA warm-up; Python cyclic GC is
  disabled in the timed candidate decoder, as in the search benchmark.

The matching/structural decoder is common to both arms. Because the fixed A/B
execution order can give its second invocation a warm allocator/cache, the
normalized totals below use the mean of the two measured matching times as the
same downstream cost for both arms. This removes a non-cache order effect.

| depth | state-prefix predictions | full prefix state | paged readout + advance | prefix speedup | normalized full total | normalized paged total | normalized speedup | page sharing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6,328 | 0.97 s | 1.26 s | 0.77x | 6.77 s | 7.06 s | 0.96x | 1.00x |
| 16 | 14,328 | 3.71 s | 2.93 s | 1.27x | 19.29 s | 18.52 s | 1.04x | 1.83x |
| 32 | 30,328 | 12.06 s | 7.44 s | 1.62x | 45.89 s | 41.27 s | 1.11x | 3.94x |
| 64 | 62,328 | 34.57 s | 21.09 s | **1.64x** | 112.04 s | 98.57 s | **1.14x** | **7.01x** |

This validates the user's observation: at depth 8, large-batch full-prefix
recompute is efficient enough that page gather/COW/advance exceeds the saved
work. The cache crosses over by depth 16 and becomes material at depth 64, but
the common matching and complete-binding decode still take 77.48 seconds at
depth 64. Removing prefix replay alone therefore cannot reproduce the much
larger speedup over original Quartz matching.

Structural state agrees exactly: there are zero live-slot/gate-type mismatches
across 62,328 compared state-prefixes. With the runtime BF16 cache, maximum
node-state and logit absolute differences are 0.015625 and 0.5. At depth 64,
the two complete candidate sets have 99.9947% global Jaccard agreement (3,364
candidate-key differences among about 63.4 million); 95.31% of individual
states are exactly set-identical. These small boundary differences explain why
the two searches should not be expected to remain bit-for-bit identical after
beam sorting, even though their discrete circuit state is identical.

An earlier run incorrectly omitted `torch.no_grad()` around the benchmark main
loop. It retained autograd activations through the recurrent full-prefix replay
and consequently reported a false 80 GB OOM and 39.57 GB transient peak. Those
figures are superseded. The corrected microbatch-512 run completes depth 64.
Peak incremental prefix allocations are 1,647,508,992 bytes (1.647 GB) for full
recompute and 1,347,304,448 bytes (1.347 GB) for paged readout. The preallocated
paged arena adds 304,128,000 bytes (0.304 GB), making the paged-specific sum
1,651,432,448 bytes (1.651 GB). These component figures exclude common model,
input, matching, and structural-decode allocations. At depth 64, 8,000 logical
pages occupy 1,141 live physical pages.

Artifacts:

- normalized summary: `benchmark_results/prefix_cache_strict_ab_hwb6_actual_beams_b1000_d64_summary.json`;
- corrected A/B result: `benchmark_results/prefix_cache_strict_ab_hwb6_actual_beams_b1000_d64_nograd_mb512.json`;
- superseded autograd-enabled result: `benchmark_results/prefix_cache_strict_ab_hwb6_actual_beams_b1000_d64_mb256_gcfix.json`;
- 64-layer beam export run: `benchmark_results/paged_action_dual_v15_hwb6_b1000_d64_levels.json`;
- benchmark driver: `prefix_cache_ab_benchmark.py`.

## Vectorized page operations and SDPA

The current runtime replaces the per-handle/per-page CUDA assignments with one
device token-index table and batched indexing. Partial-tail COW still preserves
the ownership semantics above, but copies all child tails in three batched
operations. The old `loop` backend remains selectable for strict A/B and
rollback. Node-to-action readout now uses PyTorch SDPA by default; the eager
implementation also remains selectable.

An isolated H100 benchmark at the actual depth-64 shape (`B=512`, `T=64`, page
size 8, four layers, six heads, head width 32) gives:

| operation | loop | vectorized | speedup | output error |
|---|---:|---:|---:|---:|
| action-state gather | 55.06 ms | 7.65 ms | **7.20x** | 0 |
| full KV + action gather | 137.36 ms | 2.23 ms | **61.66x** | 0 |
| partial-tail COW append | 15.16 ms | 0.90 ms | **16.84x** | 0 |

The isolated node-history attention comparison (`N=381`, 253 live slots) shows
that SDPA primarily saves memory rather than time at this short history:

| backend | time | incremental peak | max error vs eager |
|---|---:|---:|---:|
| eager | 3.098 ms | 1,000.56 MB | 0 |
| SDPA | 3.088 ms | 627.46 MB | 0.00213 |
| SDPA with live-slot packing | 3.592 ms | 675.63 MB | 0.00213 |

Live-slot packing reduces the attention matrix but its gather/scatter overhead
is larger than the saved compute at `381 -> 253` slots. It is retained as the
`sdpa_live` experimental backend, while `sdpa` is the production default.

With exact-shape warm-up on the same actual 1000-state beam tree, the optimized
prefix timings are:

| depth | full prefix | optimized paged prefix | cache speedup | normalized total speedup |
|---:|---:|---:|---:|---:|
| 8 | 0.49 s | 0.91 s | 0.54x | 0.94x |
| 16 | 1.56 s | 1.66 s | 0.94x | 0.99x |
| 32 | 5.48 s | 3.18 s | 1.73x | 1.06x |
| 64 | 21.15 s | 6.31 s | **3.35x** | **1.17x** |

The production search comparison is more important because it includes page
management, candidate generation, proposal selection, lazy topology updates,
and beam divergence from BF16 threshold-boundary differences:

| `hwb6`, beam 1000 | previous runtime | vectorized + SDPA | speedup |
|---|---:|---:|---:|
| depth-64 search | 331.11 s | 308.22 s | **1.074x** |
| accumulated model/match | 178.98 s | 166.73 s | **1.073x** |
| accumulated cache advance | 15.90 s | 4.65 s | **3.42x** |
| best speculative gates | 253 | 253 | same |

At depth 8, exact Quartz replay accepts 960/1000 optimized trajectories versus
962/1000 previously; both runs reach 253 gates. The 0.2 percentage-point change
comes from SDPA/BF16 rounding at calibrated threshold boundaries rather than a
structural-state mismatch.

For applications that require the old eager arithmetic, vectorized page
operations can be selected with `--readout-attention-backend eager`. The
depth-64 search then takes 315.28 seconds (1.050x over the previous runtime),
with cache advance reduced from 15.90 to 4.50 seconds (3.54x). Its depth-8
Quartz audit exactly retains the previous 962/1000 valid trajectories and the
same failure counts. SDPA remains the default because it is another 1.023x
faster end to end, cuts isolated attention peak allocation by 37.3%, and the
observed validity difference is only 0.2 percentage points.

Optimization artifacts:

- `benchmark_results/cache_kernels_hwb6_shape_b512_t64_gcfix.json`;
- `benchmark_results/attention_backends_hwb6_shape_b512_n381_l253_t64.json`;
- `benchmark_results/prefix_cache_strict_ab_hwb6_b1000_d64_vec_sdpa_warm_mb512.json`;
- `benchmark_results/paged_action_dual_v15_hwb6_b1000_d64_vec_sdpa.json`;
- `benchmark_results/paged_action_dual_v15_hwb6_b1000_d8_vec_sdpa_audit.json`;
- `benchmark_results/paged_action_dual_v15_hwb6_b1000_d64_vec_eager.json`;
- `benchmark_results/paged_action_dual_v15_hwb6_b1000_d8_vec_eager_audit.json`;
- `cache_kernel_benchmark.py` and `attention_backend_benchmark.py`.

## Direct block-table attention and batch scaling

The earlier "paged" implementation paged storage and ownership, but gathered
each history into a contiguous tensor before attention. The new experimental
`paged` attention backend adds a Triton kernel that follows the block table
inside the attention operation. It is used in two places:

- one-query causal attention over the action-prefix KV cache, with the current
  action included without first appending a mutable page;
- node-to-action readout over separately cached readout K/V projections.

The projected readout K/V adds 6,144 BF16 bytes per physical page at page size
8. With capacity 11,000, arena storage therefore rises from 304.128 MB to
371.712 MB. The strict fixed-beam test still has zero live/type mismatches. Its
maximum state error versus full recomputation is 0.02190, and the depth-8 full
candidate-set global Jaccard is 99.9479%. Exact Quartz replay accepts 959/1000
depth-8 trajectories and retains the 253-gate best result (the SDPA path accepts
960/1000).

On the fixed beam, direct block-table attention changes the prefix timings as
follows. These figures include page table creation, readout, action advance,
projection insertion, and COW:

| depth | gather+SDPA readout+advance | direct paged | direct/gather speedup | full-prefix/direct speedup |
|---:|---:|---:|---:|---:|
| 8 | 0.913 s | 0.824 s | 1.108x | 0.60x |
| 16 | 1.659 s | 1.551 s | 1.070x | 1.01x |
| 32 | 3.177 s | 3.030 s | 1.049x | 1.81x |
| 64 | 6.315 s | 6.047 s | **1.044x** | **3.50x** |

The modest integrated gain is consistent with the kernel-level result: direct
paging accelerates the attention operation, but attention is only a small part
of `advance_incremental` and an even smaller part of the full search. The real
depth-64 run takes 310.60 seconds versus 308.22 seconds for vectorized gather +
SDPA, with the same 253-gate best speculative result. Run-to-run candidate
decode and beam divergence are larger than the roughly 0.27-second prefix
saving on the fixed beam, so SDPA remains the production default.

Batch scaling was measured both at the kernel and complete-search levels. At
`T=64`, `N=381`, processing 1000 readout sequences takes approximately 0.424,
0.424, and 0.415 ms using microbatches 256, 512, and 1000 respectively; the
readout is effectively saturated by 256. Causal append still improves from
about 0.089 to 0.045 to 0.040 ms for the same 1000 sequences. In the actual
depth-8 search, however, microbatch 256/512/1000 takes 28.26/28.59/29.15
seconds. The remaining
matching/binding decoder, proposal sorting, and topology hashing do not become
faster when only the neural microbatch is enlarged.

New artifacts:

- `paged_attention.py` and `test_paged_attention.py`;
- `paged_attention_benchmark.py`;
- `benchmark_results/paged_attention_h100_batch_sweep_b256_b512_b1000_t64_n381.json`;
- `benchmark_results/paged_attention_batch_search_summary.json`;
- `benchmark_results/prefix_cache_paged_attention_state_only_b1000_d64_mb512.json`;
- `benchmark_results/paged_attention_dynamic_search_hwb6_b1000_d64_mb512.json`;
- `benchmark_results/paged_attention_search_hwb6_b1000_d8_mb512_audit.json`.

## Fine-grained end-to-end stage profile

`paged_rollout_benchmark.py --profile-stages` now synchronizes CUDA at the
boundaries of otherwise asynchronous stages and records an exclusive breakdown.
The flag is off by default, so it adds no synchronization to normal searches.
The percentages below use `search_seconds_excluding_audit` as the denominator;
the leaf times, per-step unattributed time, and loop-boundary time add to exactly
100% without counting parent stages twice.

The main `hwb6`, beam-1000, depth-64, microbatch-512 direct-paged run takes
309.939 seconds, completes all 64 levels, and retains the 253-gate best result.
Its top-level breakdown is:

| exclusive stage | seconds | search time |
|---|---:|---:|
| model matching, including batch/candidate materialization | 172.684 | 55.72% |
| proposal construction and sorting | 58.760 | 18.96% |
| lazy topology update and raw hash | 38.603 | 12.46% |
| source-match to xfer-action expansion | 29.403 | 9.49% |
| residual step/loop overhead | 5.712 | 1.84% |
| paged-cache action advance, including COW | 4.745 | 1.53% |
| beam prune and GPU reindex | 0.032 | 0.01% |
| exact Quartz refresh (disabled in this run) | 0.000 | 0.00% |

The 172.684-second model-matching parent breaks down as follows. The last
column is relative to the whole search, not merely model matching:

| exclusive matching substage | seconds | search time |
|---|---:|---:|
| CPU `collate_states` and pad | 103.669 | 33.45% |
| candidate tensors device-to-host | 22.361 | 7.21% |
| valid candidates packed into Python rows | 17.382 | 5.61% |
| thresholded candidate selection and per-state ranking | 9.252 | 2.98% |
| consistency checks and matching-loop residual | 8.361 | 2.70% |
| match logits | 5.756 | 1.86% |
| threshold/calibration tensors | 2.267 | 0.73% |
| incremental graph plus paged-prefix readout | 2.243 | 0.72% |
| complete-binding structural decode | 1.152 | 0.37% |
| batch H2D, candidate GPU pack, and cache metadata | 0.242 | 0.08% |

Thus the apparent 55.72% "model matching" bottleneck is mostly state and
candidate materialization. The actual incremental readout plus match logits is
only 8.0 seconds, or 2.58% of the end-to-end search. Paged-cache advance is also
small: 4.745 seconds total, and its largest child is Python/CUDA action-tensor
packing (3.246 seconds); causal model advance itself is about 1.212 seconds.
Increasing the neural batch or further tuning attention cannot materially
change the dominant 33.45% batch construction, 18.96% proposal sorting, 12.82%
candidate D2H/Python packing, 12.46% topology/hash, and 9.49% action expansion.

A second current-code run covers the intended recovery policy: depth 16 with
exact Quartz replay at depths 8 and 16 and refresh factor 2. It takes 106.896
seconds and retains 253 gates. The two refreshes take 13.597 and 24.796 seconds,
respectively, and keep 1888/2000 then 1731/2000 replay-valid states before
pruning to beam 1000.

| exclusive stage, refresh-every-8 run | seconds | search time |
|---|---:|---:|
| exact Quartz replay and refresh | 38.393 | 35.92% |
| model matching | 35.656 | 33.36% |
| proposal construction and sorting | 13.695 | 12.81% |
| lazy topology update and raw hash | 11.436 | 10.70% |
| action expansion | 5.354 | 5.01% |
| paged-cache action advance | 1.360 | 1.27% |
| residual overhead and beam reindex | 1.001 | 0.94% |

Profile artifacts:

- `benchmark_results/paged_attention_dynamic_search_hwb6_b1000_d64_mb512_stage_profile.json`;
- `benchmark_results/paged_attention_refresh8_hwb6_b1000_d16_mb512_stage_profile.json`.

## Tensorized rollout pipeline and indexed topology

The CPU-heavy rollout stages above were replaced without reintroducing Quartz
copy/apply in the speculative hot path:

- `tensorized_batch.py` reuses the authoritative paged GPU gate-type state and
  bulk-packs locality and graph metadata.  Canonical edge ordering is retained
  with one packed NumPy lexicographic sort so the graph reductions match the
  previous path;
- `threshold_candidate_tensors` keeps thresholding, structural binding decode,
  and per-state candidate selection on the GPU;
- `gpu_proposals.py` expands source matches to allowed xfers and applies the
  per-parent/global caps on the GPU.  Only the final bounded proposal set is
  copied to the CPU;
- indexed topology stores port adjacency, undirected adjacency, and an
  incremental XOR fingerprint.  Duplicate successors are rejected before a
  child topology is copied; accepted children use array-based BFS locality and
  dense touch/distance metadata;
- action tensors are packed in bulk rather than creating one CUDA tensor per
  proposal;
- each exact refresh becomes a new Quartz checkpoint.  The next refresh replays
  only the suffix after that checkpoint instead of replaying from `s0` again.

All fast paths are explicit CLI choices:
`--state-batch-backend tensorized --proposal-backend gpu
--lazy-topology-backend indexed`.  The legacy implementations remain available
as their default choices for isolated A/B tests.

On H100 GPU4 of `h100-gpu1`, the v15 `hwb6` beam-1000 depth-8 audit gives:

| pipeline | search time | exact Quartz replay | best gates |
|---|---:|---:|---:|
| indexed GPU pipeline before adjacency/metadata work | 5.659 s | 959/1000 | 259 -> 253 |
| final tensorized/indexed pipeline | **3.919 s** | **959/1000** | **259 -> 253** |

The exact failure distribution is unchanged at 15 first failures on action 4
and 26 on action 5.  The first three search levels accept 2327 children in
1.803 seconds.  The preserved original-Quartz batch baseline accepts 2307 in
151.05 seconds, so this measured end-to-end prefix is **83.8x faster**.  At the
third level, neural matching processes 4338 states/s versus the preserved
Quartz matcher at 9.72 states/s, a **446x matcher-throughput ratio**.  These
ratios compare processing the complete batch, not one isolated action.

The full direct-paged depth-64 run now takes **29.412 seconds**, completes all
64 levels, retains 253 gates, and is **10.54x faster** than the profiled
309.939-second pipeline at the start of this optimization.  Its largest
exclusive stages are:

| exclusive stage | seconds | search time |
|---|---:|---:|
| indexed lazy topology, locality, and hash | 9.311 | 31.66% |
| tensorized graph/locality collation | 7.426 | 25.25% |
| incremental graph plus paged-prefix readout | 2.355 | 8.01% |
| match logits | 2.099 | 7.14% |
| GPU candidate selection and ranking | 1.605 | 5.46% |
| final proposal D2H and Python packing | 1.503 | 5.11% |
| complete-binding structural decode | 1.146 | 3.90% |
| causal action-model advance | 0.925 | 3.15% |

The refresh-every-8 depth-16 run takes **37.178 seconds**, versus the original
profiled 106.896 seconds (**2.88x**).  Exact Quartz recovery is now 29.535
seconds, or 79.44% of the remaining runtime.  Refreshes at depths 8 and 16 take
14.766 and 14.769 seconds and retain 1888/2000 and 1731/2000 valid over-generated
states before pruning to the final 1000-state beam.  A separate dual-path audit
compared checkpoint-suffix replay with full replay from `s0` for 64 depth-16
states; legality, failure step, and exact topology have zero mismatches.

Final artifacts:

- `tensorized_batch.py`, `gpu_proposals.py`, and `test_gpu_proposals.py`;
- `test_tensorized_batch.py` and `test_indexed_topology.py`;
- `benchmark_results/gpu_pipeline_densemetadata_dual_hwb6_b1000_d8_mb512_profile_audit_gpu1.json`;
- `benchmark_results/gpu_pipeline_densemetadata_dual_hwb6_b1000_d64_mb512_profile_gpu1.json`;
- `benchmark_results/gpu_pipeline_densemetadata_checkpoint_refresh8_dual_hwb6_b1000_d16_mb512_profile_gpu1.json`;
- `benchmark_results/gpu_pipeline_checkpoint_audit64_refresh8_hwb6_b1000_d16_mb512_profile_gpu1.json`.

## Cross-circuit audit of the final pipeline

The final tensorized/indexed pipeline was also measured on five non-`hwb6`
circuits with beam 256 and depth 8.  `vbe_adder_3` occurs in the training set,
`gf2^4_mult` is a held-out test circuit, and the larger `gf2^5_mult`, `adder_8`,
and `grover_5` circuits are absent from both training and held-out trajectories.

| circuit | relation to training data | gates | search | exact-valid/topology-exact |
|---|---|---:|---:|---:|
| `vbe_adder_3` | train | 150 -> 146 | 1.424 s | 256/256 |
| `gf2^4_mult` | held-out test | 225 -> 219 | 1.812 s | 256/256 |
| `gf2^5_mult` | unseen, larger | 347 -> 339 | 1.760 s | 256/256 |
| `adder_8` | unseen, larger | 900 -> 900 | 3.031 s | 256/256 |
| `grover_5` | unseen, larger | 831 -> 825 | 2.879 s | 256/256 |

Thus the implementation-level claim generalizes across these circuits: all
1,280 audited final trajectories replay legally and all 1,280 lightweight
topologies equal Quartz exactly.  The optimization-quality claim does not hold
uniformly: no gate reduction was found on `adder_8` at this search budget.

Original CPU Quartz depth-3 runs give the following same-circuit comparison.
The end-to-end ratio compares complete three-level search time; matcher ratio
compares exact-match state throughput against model matching at level 3.

| circuit | predicted depth-3 | CPU Quartz depth-3 | end-to-end | matcher throughput |
|---|---:|---:|---:|---:|
| `vbe_adder_3` | 0.979 s | 23.964 s | **24.5x** | **367x** |
| `gf2^4_mult` | 1.129 s | 42.897 s | **38.0x** | **428x** |
| `gf2^5_mult` | 1.108 s | 62.093 s | **56.0x** | **427x** |
| `grover_5` | 1.520 s | 191.133 s | **125.8x** | **515x** |

The earlier `hwb6` 83.8x result is therefore not a circuit-independent
constant.  At beam 256 the measured end-to-end range is 24.5x--125.8x; larger
circuits benefit more because Quartz exact matching grows with graph size,
whereas the batched model remains GPU efficient.

Recovery is also circuit dependent.  At depths 8 and 16, both `gf2^4_mult`
and `gf2^5_mult` have 512/512 valid over-generated states at both refreshes.
Nevertheless Quartz refresh consumes 71.8% and 75.9% of their respective
depth-16 runtimes.  This demonstrates that fixed 2x over-generation and a
uniform eight-step refresh are unnecessarily expensive on these circuits;
risk-adaptive recovery is preferable to a global schedule.

The machine-readable aggregate is
`benchmark_results/cross_circuit_final_summary.json`; the individual
`cross_circuit_final_*`, `cross_circuit_cpu_quartz_*`, and
`cross_circuit_refresh8_*` JSON files retain all per-step counters.

## Deferred Quartz graph materialization

Exact refresh previously materialized every intermediate `PyGraph` node list
and computed its whole-graph Quartz hash after each replayed action.  Refresh
only needs stable source identities and destination GUIDs between actions, so
the direct binding now accepts source GUIDs and defers node topology and hash
materialization until a caller actually requests them.

On `h100-gpu5` GPU 6 with `hwb6`, beam 1000, depth 16, and refresh interval 8,
the same profiled search changed as follows:

| metric | eager node/hash binding | deferred GUID binding | change |
|---|---:|---:|---:|
| search excluding audit | 13.153 s | 11.148 s | -15.2% |
| exact refresh | 4.051 s | 2.547 s | -37.1% |
| Quartz apply wrapper | 3.058 s | 1.383 s | -54.8% |
| best gate count | 253 | 253 | unchanged |

The final audit now explicitly ignores refresh checkpoints and independently
replays every trajectory from the input circuit through the legacy anchor
binding path.  All 1000 trajectories are valid and all 1000 reconstructed
topologies match; this independent audit takes 11.708 seconds.  The profile and
audit are retained in `current_gpu6_*_lazygraph*.json`.

## Batched speculative PPO collection

The original PPO collector advanced one episode at a time and performed an
exact Quartz replay after every selected action. The new collector directly
reuses the paged causal cache and indexed lightweight topology from beam search,
advances many episodes in one GPU batch, and commits pending PPO transitions
only after periodic exact replay. The exact B=1/R=1 path remains available as
the compatibility baseline.

On `h100-gpu5` GPU 6, 64 `barenco_tof_3` episodes with maximum depth 16 and
seed 773 give 77.36 transitions/s for the legacy collector, 217.89
transitions/s for B=64/R=1, and 214.88 transitions/s for B=64/R=8. Accepted
rewrite throughput rises from 67.76/s to 209.91/s at B=64/R=8. Exact selected
action legality is 98.12%, 98.72%, and 97.69%, respectively. The same 58-gate
best is retained in this one-iteration collector test.

The near tie between R=1 and R=8 on this small circuit shows that the measured
2.78x transition-throughput gain comes from batched neural collection rather
than delayed Quartz checks. The batched collector terminates deferred paths at
an exact invalid action or cycle, while the legacy collector retries at the
same state, so the transition counts are not trajectory-identical. Full data
and caveats are in
`benchmark_results/ppo_batched_collector_findings_20260905.md`.

## Offline action-value pilot

The matcher is trained to recover legal source-pattern bindings, not to choose
the rewrite with the best long-term optimization return.  An offline
preference pilot tested whether a small value head could fill that gap without
target-circuit PPO or changing matcher recall.

Training data was collected on the four training circuits only:
`barenco_tof_3`, `mod5_4`, `tof_4`, and `vbe_adder_3`.  Three reproducible
stochastic beam runs (seeds 73--75, beam 1000, depth 16, refresh every 8)
provided sibling actions from common prefixes.  The target subtracts the
current rewrite delta and compares only future return:
`minimum descendant final_gate - child_gate`.  Prefixes require at least three
remaining actions and both preferred/rejected children require at least two
surviving descendants.  Seeds 73/74 produce 687 training pairs; seed 75 is
held out as 342 validation pairs.

The value head concatenates xfer, source-pattern, bound-node, and whole-graph
features.  It has 149,377 trainable parameters.  All 147 tensors from the
6.9M-parameter matcher checkpoint remain byte-for-byte unchanged.  Prefixes
are bucketed by length during frozen feature encoding to avoid inactive-action
padding in the causal attention path.  Feature encoding takes 3.160 seconds on
H100; the selected checkpoint is epoch 5 at learning rate `5e-5`.

| preference validation | accuracy |
|---|---:|
| random baseline | 50.00% |
| unfiltered single-descendant labels | 57.82% |
| support-2 value head | **61.70%** |

At inference, every parent is first capped to 128 proposals with the existing
gate-first rule.  The retained actions are scored in GPU microbatches, globally
standardized, and ranked by
`next_gate_count - action_value_weight * standardized_value`.  The first-step
exploration matcher and quota are identical in the gate/value A/B.

The learned preference accuracy did **not** improve zero-shot circuit quality:

| held-out search | gate-first | value weights 0.1/0.25/0.5/1.0 | result |
|---|---:|---:|---|
| `hwb6`, beam 1000, depth 16 | 253 gates, 10.807 s | 253 gates, 11.655--12.217 s | no quality gain |
| `hwb6`, beam 1000, depth 64 | 253 gates, 48.94 s | 253 gates, 51.44--57.36 s | no quality gain |
| `gf2^4_mult`, beam 1000, depth 16 | 219 gates, 10.03 s | 219 gates, 9.62--10.94 s | no quality gain |

Every reported configuration has 64/64 valid independent Quartz replays and
64/64 exact topology matches.  Because quality is unchanged and value scoring
adds work, `gate` remains the default proposal ranking.  `value` is retained as
an explicit research mode, not a production optimization.  The result suggests
that minimum outcomes from stochastic descendants remain too noisy; the next
training data should evaluate each sibling with a shared continuation policy
and multiple short rollouts or a stronger teacher, then aggregate returns
before fitting the value head.

Reproducibility artifacts:

- `run_preference_beams.sh` and `collect_action_preferences.py`;
- `preference_dataset.py` and `train_action_preferences.py`;
- `run_action_value_ab.sh`;
- `benchmark_results/action_preferences_stochastic_s73_75_d16_support2_v2.metadata.json`;
- `benchmark_results/action_value_d16_support2_v2_lr5e5.training.json`;
- `benchmark_results/action_value_v2_ab_*`.

## Exact best-so-far tracking

Offline preference training did not maintain an optimization archive, and the
rollout output previously derived `--best-qasm` only from the final independent
audit prefix.  This could miss a better circuit confirmed at an earlier refresh
or later in the final beam, especially when value ranking is not ordered by gate
count.

Paged rollout now maintains a separate monotonic `best_exact_gate_count` and
`best_exact_depth`.  The archive starts with the input graph and is updated only
by a Quartz refresh or independent replay whose topology matches the lazy state.
Each step reports `best_exact_gate_count_so_far`; `--best-qasm` exports this
confirmed archive even when `--audit-count 0`.  Speculative minima remain a
separate metric and can never become a future training root without refresh.

An H100 smoke run on `mod5_4` with beam 256, depth 8, and refresh interval 8
records 63 gates through the speculative depths, then confirms and exports 62
gates at depth 8.  Parsing the exported QASM independently with Quartz also
returns 62 gates.  Search excluding audit takes 1.566 seconds.  The full result
and exported circuit are retained as
`benchmark_results/best_exact_tracking_smoke_mod5_b256_d8.json` and
`benchmark_results/best_exact_tracking_smoke_mod5_b256_d8_best.qasm`.

## Resident best-root self-improvement

`accelerated_self_improve.py` adds a persistent, resumable archive around the
paged search.  More importantly, `--restart-from-best-at-refresh` can relocate
the search root inside the existing process.  The model, rule tensors, Quartz
context, and CUDA allocations remain resident.  Only the selected exact graph,
incremental graph tensors, dedup set, and empty paged-cache handles are reset.
`--best-root-restart-interval` separates correctness refreshes from root
relocation, so a 16-action training trajectory can still refresh at action 8.

Every refresh beam can be written with `--dump-refresh-histories-dir`.  Each
file contains the exact segment root snapshot and action histories relative to
that root, making the files directly consumable by the preference collector.
The archive records the root gate count, monotonic exact best, history path,
stale-refresh count, and whether the next segment restarted from the best.

On H100, three `mod5_4` rounds with beam 256 and eight actions per round compare
as follows:

| execution | model loads | wall time | exact best trace |
|---|---:|---:|---|
| one Python process per round | 3 | 112.57 s | 63 -> 62 -> 62 -> 62 |
| resident model/CUDA process | 1 | **37.89 s** | 63 -> 62 -> 62 -> 62 |

Resident execution reduces wall time by 66.3% (2.97x).  A separate two-round,
16-action test refreshes every eight actions and relocates only at actions 16
and 32.  It records roots 63 then 62, keeps the exact best at 62, independently
replays 64/64 final states successfully with 64/64 topology matches, spends
3.376 seconds in search, and takes 37.781 seconds including the single process
and model startup.  The artifacts are:

- `benchmark_results/process_per_round_archive_mod5_b256_r3d8.json`;
- `benchmark_results/resident_archive_mod5_b256_r3d8.json`;
- `benchmark_results/resident_best_archive_mod5_b256_r2d16.json`;
- `benchmark_results/resident_best_archive_mod5_b256_r2d16_rollout.json`;
- `benchmark_results/resident_best_archive_mod5_b256_r2d16_best.qasm`.

## Accelerated online self-improvement

The resident archive is now connected to a self-contained online training
loop in `run_accelerated_online_iteration.sh`; it does not invoke Quarl or PPO.
Four training circuits search concurrently on four H100s.  Each process keeps
the matcher, Quartz context, and CUDA state resident for three 16-action
segments, archives only Quartz-confirmed best circuits, then contributes valid
refresh trajectories to the next action-value update.  Completion markers make
an iteration resumable per circuit, so one failed worker does not repeat the
other three searches.

The preference target was changed from final trajectory residual to the best
future prefix residual.  This avoids teaching a good early action that it is
bad merely because forced later actions regress before the fixed depth.  The
collector reads resident archives directly and keeps seed-75 histories as a
fixed validation set.  Preference magnitude can weight the pairwise loss, and
epoch 0 is now a checkpoint candidate: an online update cannot overwrite the
input policy unless it improves fixed validation accuracy.

The first best-prefix model uses 1,011 training and 155 fixed validation pairs.
Validation accuracy rises from 44.52% for the old head under the new target to
69.68% at epoch 16.  Two subsequent online fits do not exceed 69.68%, so the
epoch-0 guard correctly retains the existing policy.  This also shows why pair
accuracy alone is insufficient: pure value search can concentrate the whole
beam into model false positives and produce no Quartz-valid final leaf.

Value proposal pre-capping now reserves 16 of 128 per-parent positions for
matcher-ranked gate-increasing actions, allowing short-term regressions that
may unlock a lower circuit.  At global ranking, 25% of proposals can be
reserved for deterministic stochastic exploration and interleaved with value
proposals.  The selected exploration count is reported per step.  If every
leaf fails a Quartz refresh, resident search records the empty refresh and
restarts from the verified best at the next available step; an empty final beam
still exports that exact best instead of crashing.

A controlled H100 A/B uses the same original circuit, checkpoint, seed, beam
1000, depth 16, refresh interval 8, and maximum single-action gate increase 3:

| circuit | pure value best | 25% mixed best | final valid leaves, pure -> mixed | search, pure -> mixed |
|---|---:|---:|---:|---:|
| `barenco_tof_3` | 58 | 58 | 0 -> 145 | 5.305 -> 5.134 s |
| `mod5_4` | 62 | 62 | 8 -> 15 | 5.416 -> 4.682 s |
| `tof_4` | 75 | **74** | 1000 -> 435 | 5.686 -> 6.113 s |
| `vbe_adder_3` | 148 | **144** | 1000 -> 1000 | 8.063 -> 7.190 s |

The mixed policy improves two of four best circuits, restores a usable final
beam on `barenco_tof_3`, and reduces aggregate search time from 24.47 to 23.12
seconds in this run.  Independently parsing every exported QASM with Quartz
reproduces the reported gate count.

Across 12 archived self-improvement segments per circuit, the monotonic exact
best values are now 58 -> 56 for `barenco_tof_3`, 63 -> 62 for `mod5_4`,
75 -> 71 for `tof_4`, and 150 -> 140 for `vbe_adder_3`.  These are verified
search results, not model-predicted counts.  The archives and exact best QASM
files are retained as `benchmark_results/online_self_improve_*`; controlled
A/B outputs are `benchmark_results/*_mix_*.json`, and dataset/training logs are
`benchmark_results/action_preferences_online_v1_iter*` and
`benchmark_results/action_value_online_v1_iter*`.

## Exact on-policy PPO pilot

`train_paged_ppo.py` implements clipped PPO rather than another preference
loss.  The frozen paged matcher supplies the candidate set and graph/action
features; trainable policy-residual and critic heads are optimized from stored
old log probabilities, GAE returns, clipped policy and value objectives, and
an entropy bonus.  Every sampled `(xfer, binding)` is immediately executed by
Quartz.  An illegal action receives a negative transition, is masked at the
same exact graph, and the policy selects again.  No speculative invalid graph
is used as the next state.

Within an episode, predictions still use the initial full-graph encoding plus
the verified action sequence and paged KV state.  A new episode selected from
the exact best archive or replay pool starts with a full graph and calls
`initialize_incremental` again, so no action history crosses episode roots.
The bounded per-circuit replay pool is important: it lets the policy learn a
gate-reducing inverse action from an elevated graph without accepting a cycle
inside the trajectory.  The exact best QASM is a separate monotonic archive;
after an improvement, later episodes default to that best graph.

The H100 pilot trained one shared head on four circuits for 15 iterations and
96 episodes per iteration.  It collected 24,691 exact transitions in 317.28
seconds (77.82 transitions/s weighted), while all PPO updates took 9.02
seconds.  There were 36 replay-root improvements.  The selected-action Quartz
legality increased from 94.05% in iteration 0 to 98.97% in iteration 14.

| training circuit | input | exact PPO best |
|---|---:|---:|
| `barenco_tof_3` | 58 | 58 |
| `mod5_4` | 63 | **62** |
| `tof_4` | 75 | 75 |
| `vbe_adder_3` | 150 | **144** |

The exported QASMs were independently parsed by Quartz and reproduced
58/62/75/144 gates.  This run takes minutes rather than an eight-hour
per-circuit fine-tune, but it is not yet evidence that the learned policy beats
gate-first search on unseen circuits.

For inference, PPO proposal scoring is fused into `build_gpu_proposals`: it
reuses resident candidate tensors, normalizes logits per parent, and performs
only the final compact device-to-host copy.  On a profiled `vbe_adder_3` run,
the fused PPO policy stage is 0.073 seconds, or 1.01% of 7.23 seconds search
time.  The earlier Python repack prototype spent about five seconds per run in
policy ranking.

A controlled beam-1000, depth-16 A/B uses an immediate-gate cost corrected by
`0.25 * standardized PPO score`.  PPO and gate-first produce identical exact
best counts on all six circuits, including held-out `hwb6` and
`gf2^4_mult`: 58/62/75/148/255/219.  Aggregate search time is 44.52 seconds
for PPO versus 43.76 seconds for gate-first (+1.7%).  Both configurations have
384/384 valid independent Quartz replays and 384/384 exact topology matches.
The pilot therefore establishes a correct, fast PPO path and non-regressing
held-out behavior, but no held-out quality gain yet.

Reproducibility artifacts are `ppo_core.py`, `train_paged_ppo.py`,
`run_paged_ppo_training.sh`, `run_ppo_policy_ab.sh`,
`benchmark_results/paged_ppo_replay_v1*`, and
`benchmark_results/ppo_v3_*`.

## Original Quarl rollout audit

A fresh H100 run on `barenco_tof_3` used Quarl's original exact-graph actor,
persistent graph buffer, dynamic episode horizon, and six-circuit pretrained
checkpoint.  Fourteen completed fine-tuning iterations collected 81,344 exact
transitions in 261.24 rollout seconds (311.37 transitions/s) and changed the
best-so-far from 58 to 38 gates.  A same-seed control that executed the same PPO
training loop with all learning rates set to zero reached 36 gates in 59,584
transitions.  Thus the observed short-run descent is primarily evidence for
Quarl's exact graph-buffer search and restart curriculum, not by itself evidence
that online PPO is improving the policy.

The complete curves, exact QASM audits, configurations, source hashes, and raw
logs are recorded in
`benchmark_results/quarl_original_rollout_findings_20260905.md`.

## Sequence-conditioned match-set PPO

The PPO head now consumes the information already maintained by the paged
world model instead of scoring each action independently. The actor attends
over the full set of retained complete bindings, including ordered binding
roles, and conditions them on the current graph summary plus the final causal
action hidden state from `PagedKVCache`. The latter already represents the
preceding action sequence. The critic uses a separate candidate-set encoder and
attention pool to produce one state value. No action-prefix replay or full-graph
regeneration was added.

The actor starts exactly at the calibrated matcher plus gate-delta prior and
the critic starts at zero. An auxiliary selected-action legality head uses a
class-balanced loss and reports per-class recall; raw accuracy was misleading
because 97.38% of collected actions were legal.

On H100 GPU 6, `barenco_tof_3`, 64 episodes, `B=64`, `R=8`, and 64 candidates,
two match-set runs measured 242.25 and 274.09 transitions/s. The old batched MLP
reference measured 214.88 transitions/s, so the observed range is 1.13x to
1.28x faster. Cross-episode actor batching offsets the larger attention head.
The one-epoch run stayed at 58 gates and is an implementation benchmark, not a
policy-quality result.

`paged_rollout_benchmark.py` now loads both `paged-ppo-v1` and
`paged-ppo-v2`, reconstructs per-parent match sets entirely on GPU, and supplies
the same prefix and graph context used during training. A beam-32, depth-2 smoke
run produced 8/8 valid Quartz replays and 8/8 exact topology matches. Full logs
and configurations are in
`benchmark_results/ppo_matchset_actor_findings_20260905.md` and
`benchmark_results/ppo_matchset_actor_summary_20260905.json`.

A subsequent 15-iteration shared-policy run on four circuits collected 18,573
transitions in 54.36 seconds (341.64 transitions/s) and spent 14.18 seconds in
PPO updates. Exact training bests were 58/62/75/146 from inputs 58/63/75/150.
The curve stopped improving after iteration 2. Legality balanced accuracy rose
from 76.85% to a 95.36% peak, but final per-iteration PPO KL reached 0.0414.

The six-circuit beam A/B initially appeared to improve held-out `hwb6` from
gate-first 255 to PPO 253. A neutral-actor control also reached 253 and passed
64/64 exact replay audits, proving this gain came from the frozen matcher/gate
prior rather than learned PPO residuals. The trained actor's depth-16 beam was
entirely rejected at the final refresh, although its exported depth-8 exact
best independently parses to 253 gates. Thus this run is not evidence of
learned zero-shot quality; it motivates explicit reference-policy KL control
and a broader circuit training distribution. Full evidence is in
`benchmark_results/ppo_matchset_v1_training_findings_20260905.md`.

The PPO update now has both controls. `--reference-kl-coefficient` applies an
exact categorical KL to the frozen matcher/gate prior, analogous to reference
policy regularization in large-model RL, and `--target-kl` stops remaining PPO
epochs after excessive movement from the rollout policy. An H100 forced-stop
smoke completed one of four requested epochs and logged both KL values; see
`benchmark_results/ppo_reference_kl_findings_20260905.md`.

The regularized broad run trained one policy across 14 circuits and collected
46,945 transitions at 293.66 transitions/s. Training best-so-far improved five
circuits, including `csla_mux_3` from 170 to 159 and `rc_adder_6` from 200 to
192. Across seven held-out circuits, trained PPO produced 3846 total gates,
versus 3848 for a neutral actor and 3855 for gate-first. Only the `grover_5`
result is attributable to the learned residual (811 trained versus 813 neutral
and 817 gate-first); the other gains come from the shared matcher/gate prior.
All 448 trained-PPO audit replays were Quartz-valid with exact topology. Full
configuration, curves, attribution, and hashes are in
`benchmark_results/ppo_matchset_broad_kl_findings_20260905.md`.

## Source-pattern chunking

The match stage no longer needs to materialize the full `[state, slot, source]`
tensor. `--source-microbatch` projects graph nodes once, scores source patterns
in bounded chunks, and maintains the exact global per-state Top-K with a
streaming GPU merge before one structural decode. A deterministic unit test
compares all retained source IDs, anchors, bindings, and probabilities against
the original full-logit path.

On one H100, `grover_5`, beam 1000, depth 16, and gate-first ranking, source
chunks of 256 reduced peak allocated memory from 8.224 GiB to 4.638 GiB and
peak reserved memory from 30.389 GiB to 13.590 GiB at state microbatch 128.
Search time was unchanged within run noise (23.482 versus 23.155 seconds), and
both runs found the same exact 817-gate best from 831 gates. State microbatch
512, which previously OOMed while allocating the full logits, now completes in
24.035 seconds with 5.551 GiB peak allocated memory and the same 817-gate best.
The measured JSON logs are `benchmark_results/source_chunk_grover5_*_h100.json`.

Source retrieval can additionally group anchors and source patterns by their
first gate type. The rule set contains 2661 `cx`, 491 `x`, 452 `h`, 243 `rz`,
and 8 `add` sources. `--source-grouping first_gate` compacts the corresponding
anchor slots and computes only type-compatible dot products instead of
computing every pair and masking most of them afterward.

| Circuit | State batch | Match before | Match grouped | Search before | Search grouped |
|---|---:|---:|---:|---:|---:|
| `grover_5` | 128 | 9.914s | 8.258s (-16.7%) | 23.155s | 22.021s (-4.9%) |
| `gf2^5_mult` | 512 | 4.564s | 3.825s (-16.2%) | 11.120s | 10.663s (-4.1%) |
| `qcla_mod_7` | 512 | 11.375s | 9.162s (-19.4%) | 27.629s | 25.730s (-6.9%) |

All A/B pairs found the same exact best gate counts (817, 339, and 884), and
every grouped run passed 64/64 Quartz replay and topology audits. Peak memory
was unchanged because source chunking had already bounded the dense matcher
activation; first-gate grouping is a compute optimization. The raw A/B logs
are `benchmark_results/source_group_*_h100.json` and the existing
`source_chunk_grover5_chunk256_b128_h100.json` baseline.

## Match preselection before xfer expansion

The GPU proposal path previously expanded every retained structural match into
all compatible xfers before applying the per-parent action cap. For gate,
probability, and PPO ranking, only the top `K` matches ranked by their best
xfer can possibly contribute to the top `K` actions: every later match already
has `K` better best actions ahead of it. `--proposal-expansion preselect` uses
that bound, packs the existing lexicographic keys into one signed 64-bit key,
selects matches with a row-wise GPU Top-K, and expands only those matches.

On `grover_5`, beam 1000, depth 16, first-gate source grouping, and state batch
128, the number of materialized action rows fell from 40,707,322 to 1,160,108
(35.1x fewer). The measured proposal stage fell from 0.582 to 0.548 seconds
(5.8%), while end-to-end time was within run noise and slightly higher (21.607
versus 21.928 seconds); matcher and refresh variation dominate this small
stage. Both searches found the same exact 817-gate best and passed all 64
Quartz audits. A trained-PPO run retained its 811-gate best and passed 64/64
audits while materializing 1,313,543 of 40,788,395 eligible actions. The raw
logs are `benchmark_results/proposal_*_grover5_b128_h100.json`.

### Selected-only proposal transfer at batch 512

A follow-up comparison against original CPU Quartz now measures the complete
GPU proposal boundary rather than materializing every retained source binding
on the host.  It includes matcher inference, r99.9 thresholding, structural
decode, source-to-xfer expansion, the `max_gate_increase=1` filter,
per-parent-128 and global-8192 ranking/caps, and Python packing of only those
8192 final proposals.  The 512-state input shape matches the historical
microbatch-512 comparison.

| Workload | Full GPU proposals | Preselected GPU proposals | Original CPU Quartz |
|---|---:|---:|---:|
| GF trajectory states | 1812.51 states/s (296.58x) | 1843.65 states/s (301.68x) | 6.111 states/s |
| Barenco trajectory states | 8232.82 states/s (121.90x) | 8030.52 states/s (118.90x) | 67.540 states/s |

Only final proposals cross to the host.  The earlier all-source-row D2H
control measured 153.34 states/s on GF and 1170.30 states/s on Barenco, so it
understates the integrated GPU proposal throughput by 7--12x.  Preselection
reduced GF intermediate xfer rows from 2.059 million to 128,467, but Barenco
only fell from 115,360 to 112,320 and was slower; full expansion therefore
remains the stable GPU default and preselection remains opt-in.

A beam-256, depth-8 full/preselect search A/B retained identical accepted
counts at every depth on both trajectory starts.  All 130 Barenco and 256 GF
final states passed Quartz replay and exact-topology audits.  The driver,
six raw result files, hashes, and exact timing boundaries are recorded in
`benchmark_results/matcher_throughput_b512_findings_20260906.md` and
`benchmark_results/matcher_throughput_b512_summary_20260906.json`.

## First-gate matcher grouping in PPO collection

PPO collection previously used the full source-pattern matrix even though the
online beam path could already skip source patterns whose first gate type did
not match an anchor. The collector now accepts `--source-grouping first_gate`
and `--source-microbatch`. A zero source microbatch executes one matrix product
per gate-type group; this is important for the small per-circuit batches used
by broad training. The training launcher enables unchunked first-gate grouping
by default.

Two one-iteration H100 A/B runs used the same 14-circuit distribution, 224
episodes, batch cap 64 (16 active episodes per circuit), depth 16, refresh 8,
and 64 actor candidates. With seed 902, unchunked grouping reduced measured
matcher time from 3.220 to 2.672 seconds (-17.0%), collection time from 13.006
to 12.264 seconds (-5.7%), and increased collection throughput from 261.6 to
277.7 transitions/s (+6.1%). Incremental collection peak allocation fell from
0.512 to 0.212 GiB, and both runs found identical per-circuit best gate counts.
Seed 901 independently reduced collection time by 4.8% and raised throughput
by 4.4%.

Chunking each source group at 256 was also measured on seed 901 and regressed
collection time by 1.8% versus unchunked full matching because the actual batch
was only 16 and the extra small kernels dominated. Source chunking remains an
explicit memory control for large rollout batches, while unchunked grouping is
the PPO default. Raw logs are
`benchmark_results/ppo_training_matcher_ab_*_h100.json`; the compact result is
`benchmark_results/ppo_training_matcher_grouping_summary_20260905.json`.

PPO proposal preselection was tested separately and is not enabled by the
training launcher. It reduced materialized actions from 2,917,852 to 412,367
(7.1x) but increased proposal time from a 0.693-second mean across two full
runs to 0.767 seconds (+10.7%). Training advances only 16 same-circuit states
at once in this broad setup and averages about 858 eligible actions per state,
so directly sorting the expanded rows is cheaper than constructing a padded
match matrix and running another Top-K. The opt-in `--proposal-expansion
preselect` remains useful for much wider beam search, where the measured
materialization reduction was 35.1x. PPO A/B logs and the decision are in
`benchmark_results/ppo_training_proposal_ab_*_h100.json` and
`benchmark_results/ppo_training_proposal_preselection_summary_20260905.json`.

## Batched PPO transition transfer

The batched collector previously copied state features, all 64 candidate
features and logits, the candidate mask, prefix state, value, sampled action,
log probability, and entropy from GPU to CPU separately for every transition.
This preserved batching for inference but introduced thousands of small CUDA
synchronizations. `--transition-transfer-backend batched` copies immutable
transition tensors once per active collector batch and transfers actor outputs
once per retry round. Candidate masks are cloned on CPU before a rejected
action is removed, preserving the exact PPO observation stored at each retry.

Two H100 A/B pairs used the same 14-circuit, 224-episode protocol as the
matcher experiment, with seed 904 in rowwise-first order and seed 905 in
batched-first order. Averaged over both pairs, transition transfer fell from
0.869 to 0.413 seconds (-52.4%), collection time fell from 12.460 to 11.688
seconds (-6.2%), and throughput rose from 273.9 to 291.8 transitions/s (+6.5%).
Each seed produced identical per-circuit best gate counts across its A/B pair,
and selected-action legality differed by less than 0.04 percentage points.
The training launcher now selects the batched backend; the rowwise backend is
retained for regression comparison. Raw logs and a compact summary are in
`benchmark_results/ppo_training_transfer_ab_*_h100.json` and
`benchmark_results/ppo_training_transfer_summary_20260905.json`.

## Reusing selected GPU proposal tensors

After GPU proposal ranking, the collector previously converted selected
actions into Python `Proposal` objects and then immediately rebuilt parent,
xfer, source, binding, probability, and gate-delta tensors on the GPU for PPO
candidate features. The binding path was especially expensive because it
issued one Python-side tensor construction per proposal. The proposal builder
can now return a `SelectedProposalTensors` payload in the exact selected order,
and `--proposal-tensor-backend reuse` feeds it directly into candidate feature
construction. Python proposals are still retained for indexed lazy rewrites.

Two reversed-order H100 A/B pairs used the 14-circuit, 224-episode protocol,
unchunked first-gate matching, full proposal expansion, and batched transition
transfer. Averaged across seeds 906 and 907, policy preparation fell from
5.154 to 1.091 seconds (-78.8%), collection time fell from 12.254 to 7.678
seconds (-37.3%), and throughput rose from 275.5 to 440.3 transitions/s
(+59.8%). Selected-action legality remained within 0.06 percentage points in
each stochastic pair. A GPU test verifies that every returned parent, xfer,
source, binding, probability, and gate delta agrees with the corresponding
Python proposal. The launcher now enables tensor reuse by default. Raw logs
and the compact summary are
`benchmark_results/ppo_training_tensor_reuse_ab_*_h100.json` and
`benchmark_results/ppo_training_tensor_reuse_summary_20260905.json`.

## Optimized broad PPO rerun

The complete 15-iteration, 14-circuit KL-regularized training protocol was
rerun with unchunked first-gate matching, batched transition transfer, and
selected proposal tensor reuse. It collected 46,922 transitions in 97.867
seconds (479.4 transitions/s), versus 46,945 in 159.862 seconds (293.7/s) in
the original broad run. Collection time fell 38.8%, throughput rose 63.3%, and
collection plus PPO update time fell from 195.445 to 133.590 seconds (-31.6%).
PPO update time itself was unchanged at 35.7 seconds, as expected.

Training best-so-far reached `mod5_4=62`, `mod_red_21=276`,
`vbe_adder_3=146`, `csla_mux_3=164`, and `rc_adder_6=198`; the other nine
circuits stayed at their inputs. The old stochastic run happened to find
`csla_mux_3=159` and `rc_adder_6=192`, so faster execution did not improve the
sampled training archive by itself.

Zero-shot beam evaluation on seven held-out circuits produced 3,849 total
gates: 253, 219, 441, 813, 339, 884, and 900 on `hwb6`, `gf2^4_mult`,
`qcla_com_7`, `grover_5`, `gf2^5_mult`, `qcla_mod_7`, and `adder_8`. All 448
audited trajectories replayed successfully and matched exact topology. This
is six gates better than gate-first (3,855), but one gate worse than the
neutral actor (3,848) and three worse than the prior trained PPO (3,846).
Therefore the optimized checkpoint validates training throughput, not better
generalization, and does not replace the prior model as the quality baseline.
The full training log, held-out logs, hashes, and compact comparison are in
`benchmark_results/ppo_matchset_broad_optimized_s271_h100.training.json`,
`benchmark_results/optimized_broad_*_ppo_h100.json`, and
`benchmark_results/ppo_broad_optimized_summary_20260905.json`.

## Tensorized PPO policy padding

The batched PPO collector previously grouped its flat selected proposals with
one Python loop and one GPU `index_select` per active episode. The tensorized
backend performs a stable parent sort, derives each proposal's within-parent
offset, and scatters features, matcher logits, and masks into the padded
policy batch in one path. Proposal order within every parent is unchanged, so
sampled action indices still address the same Python proposal list. The loop
backend remains available through `--policy-padding-backend loop`; the training
launcher now selects `tensorized` by default.

Two reversed-order H100 A/B pairs used the same 14-circuit, 224-episode
protocol as the earlier collector optimizations, with selected proposal tensor
reuse and batched transition transfer enabled. Averaged across seeds 909 and
910, policy preparation fell from 1.118 to 1.005 seconds (-10.1%), collection
time fell from 7.941 to 7.609 seconds (-4.2%), and throughput rose from 427.1
to 445.7 transitions/s (+4.4%). Peak allocated CUDA memory was unchanged at
about 0.212 GiB. Every A/B pair produced the same per-circuit best gate counts;
the seed-909 pair also produced identical transition and legality counts. A
GPU unit test checks exact padded feature, logit, mask, and proposal-order
agreement, including interleaved parents and empty candidate sets. Raw logs and
the compact result are in
`benchmark_results/ppo_training_policy_padding_ab_*_h100.json` and
`benchmark_results/ppo_training_policy_padding_summary_20260905.json`.

## Deduplicated PPO episode initialization

Each broad-training circuit contributes 16 episodes to a collector call. The
old initializer independently parsed the same starting QASM, rebuilt its
snapshot and indexed topology, and ran the initial graph encoder 16 times.
`--episode-initialization-backend deduplicated` preserves every episode's
replay-start sampling first, groups equal QASM strings, constructs and encodes
each unique starting graph once, and expands the encoded rows back into the
original episode order. Initial Quartz checkpoints and topology objects are
read-only; accepted rewrites return new graph and topology objects.

An isolated six-round H100 benchmark initialized 14 circuits with 16 episodes
per circuit. Deduplication reduced mean initialization time from 0.380 to 0.040
seconds (-89.4%). Initial model states were bitwise equal with zero maximum
error, and live masks, gate types, topology fingerprints, and root hashes all
matched. Raw snapshot dictionaries differ only because independent Quartz
parses allocate different GUIDs; GUID-independent persistent-slot topology is
identical.

Two reversed-order end-to-end A/B pairs used the optimized tensorized policy
padding path. Averaged across seeds 911 and 912, collection time fell from
7.666 to 7.243 seconds (-5.5%) and throughput rose from 441.2 to 466.1
transitions/s (+5.6%). All paired runs retained the same per-circuit best gate
counts, selected-action legality stayed within 0.06 percentage points, and
peak CUDA allocation remained about 0.212 GiB. The launcher now enables
deduplicated initialization by default, while `duplicated` remains available
for regression tests. Raw and summarized results are in
`benchmark_results/ppo_training_init_ab_*_h100.json`,
`benchmark_results/ppo_episode_initialization_microbenchmark_h100.json`, and
`benchmark_results/ppo_training_episode_initialization_summary_20260905.json`.

## Trusted paged-model advance

The general incremental model validates sequence bounds and referenced slot
capacity with GPU scalar reads, reconstructs a contiguous history mask from
paged lengths, and checks every padded source/destination column with
`bool(valid.any())`. These checks synchronize the CPU with the GPU repeatedly
on every action. PPO already bounds its horizon by the checkpoint's maximum
sequence length, `advance_selected` pads states to every selected child's
`next_slot`, direct paged attention does not consume a contiguous history mask,
and padded tensor operations are valid on empty rows.

`--advance-input-backend trusted` uses those caller guarantees to remove the
redundant scalar reads and returns an empty unused history mask. The checked
backend retains all validation. A focused GPU regression advanced the same
trajectory through both paths for eight steps and matched states, live masks,
gate types, causal keys/values, and action states at every step. The paged-cache
share, copy-on-write, gather, and reclaim test also passes.

Two reversed-order H100 A/B pairs used the 14-circuit, 224-episode optimized
collector. The directly measured cache/model advance stage improved in both
pairs: 1.370 to 1.044 seconds (-23.8%) and 1.011 to 0.952 seconds (-5.9%). The
two-pair mean fell from 1.191 to 0.998 seconds (-16.2%). End-to-end collection
was inconclusive: matcher variation exceeded the saved advance time, giving a
7.314-second checked mean and a 7.359-second trusted mean. All paired runs
retained the same best gate counts, and peak CUDA allocation was unchanged.
The launcher enables trusted inputs, while checked remains the diagnostic
fallback. Raw logs and exact stage accounting are in
`benchmark_results/ppo_training_advance_ab_*_h100.json` and
`benchmark_results/ppo_training_trusted_advance_summary_20260905.json`.

## Deferred PPO proposal materialization

The PPO collector previously converted every retained candidate into a CPU
`Proposal` object even though the actor executes only one candidate per active
episode. With 64 candidates, this copied and packed roughly 64 Python objects
for every sampled action. `--proposal-materialization-backend deferred` keeps
parent, xfer, anchor, binding, probability, and gate-count metadata on the GPU.
Tensorized policy padding retains each candidate's flat GPU index, and only the
actor-selected indices are copied and packed immediately before the indexed
lazy rewrite. Invalid or cyclic choices are re-sampled against the updated mask
and materialized from their new indices. The eager backend remains available
for regression comparison.

A counted 14-circuit, 224-episode H100 run reduced Python proposal construction
from 222,400 objects to 3,472, exactly 64x. Proposal construction plus deferred
selected-item materialization fell from 1.395 to 0.559 seconds (-60.0%). Under
the same high host load, collection time fell from 13.882 to 13.535 seconds
(-2.5%) and throughput rose from 246.9 to 253.1 transitions/s (+2.5%). Both
runs found the same per-circuit best gate counts and used about 0.212 GiB peak
CUDA allocation.

An earlier clean pair independently reduced the targeted stage from 0.727 to
0.320 seconds (-55.9%) and produced exactly the same 3,441 transitions, 13
invalid actions, legality, and per-circuit best counts. Its total collection
time was statistically flat at 7.398 versus 7.449 seconds because policy,
matcher, advance, and refresh variation exceeded the saved wall time. A third
pair is retained but excluded from comparison because a concurrent trajectory
collector expanded to about 75 CPU cores between its two runs. Across the two
usable pairs, the directly targeted stage fell by 57.9%; the end-to-end effect
ranged from -0.7% to +2.5%.

GPU tests verify eager/deferred proposal field equality and Python-backed versus
tensor-only padding equality. An end-to-end smoke completed all 32 sampled
actions legally, including continued execution after a cycle rejection. The
training launcher now enables deferred materialization by default. Raw logs and
the load-qualified comparison are in
`benchmark_results/ppo_training_materialization_ab_*_h100.json` and
`benchmark_results/ppo_training_deferred_materialization_summary_20260905.json`.

## Cached frozen source representations

The frozen paged model previously recomputed every source pattern embedding in
both PPO candidate feature construction and every incremental action advance.
These embeddings depend only on frozen base-model parameters and static source
patterns, but a profiled 14-circuit collection still called
`source_representations()` 435 times. The collector now computes the source
representations once alongside the already cached retrieval vectors and passes
that tensor through candidate feature and paged advance APIs.
`--source-representation-backend recompute` preserves the old path; `cached` is
enabled by the training launcher.

Four reversed-order H100 A/B pairs used 224 episodes, depth 16, and all prior
collector optimizations. Caching reduced mean policy preparation from 0.938 to
0.863 seconds (-8.0%) and mean cache advance from 0.993 to 0.978 seconds
(-1.5%). The two affected stages together fell 4.7%, including after
normalizing by transition count. Mean collection time fell from 6.754 to 6.694
seconds (-0.9%) and throughput rose from 502.5 to 507.0 transitions/s (+0.9%).
Three pairs found identical per-circuit best counts; in the fourth stochastic
pair the cached run improved `mod5_4` from 63 to 62. Peak CUDA allocation rose
by 1.27 MiB for the retained source tensor.

A focused eight-step regression checks equal candidate features, states, live
masks, gate types, causal keys/values, and action states between recomputed and
cached inputs. A follow-up cProfile reduced source representation calls from
435 to one. Raw A/B logs and the compact result are in
`benchmark_results/ppo_training_source_cache_ab_*_h100.json`,
`benchmark_results/ppo_source_representation_profile_counts_20260905.txt`, and
`benchmark_results/ppo_training_source_representation_cache_summary_20260905.json`.

## Original Quarl rollout scaling profile

Quarl's audited original `agent_collect` path was profiled for one fixed-work
iteration on five Nam circuits from 58 to 3,435 gates. Every run loaded the
same `iter_576.pt` checkpoint, used 64 episodes of exactly 20 steps (1,280
transitions), batch size 64, one PPO epoch, and zero learning rates. Timing-only
instrumentation synchronizes CUDA at model-stage boundaries and accounts for
99.24-99.67% of rollout wall time. The profiled 3,435-gate run took 39.86
seconds versus 40.56 seconds for the byte-identical uninstrumented source with
the same seed and outcome, within normal run variation.

| circuit | gates | rollout | transitions/s | rollout/iteration |
|---|---:|---:|---:|---:|
| `barenco_tof_3` | 58 | 5.034s | 254.26 | 89.8% |
| `vbe_adder_3` | 150 | 6.188s | 206.86 | 84.5% |
| `hwb6` | 259 | 6.889s | 185.81 | 92.4% |
| `grover_5` | 831 | 10.754s | 119.03 | 93.9% |
| `gf2_16_mult` | 3,435 | 39.859s | 32.11 | 98.1% |

For 58-259 gates, materializing the current and next DGL subgraphs needed by
the original PPO update is the largest group at 35-44% of rollout. Exact
Quartz apply grows from 1.2% at 58 gates to 28.3% at 831 and 38.0% at 3,435.
At 3,435 gates, `apply_xfer_with_local_state_tracking`, selected-node
`available_xfers_parallel` plus mask construction, and full graph-to-DGL
conversion consume 38.0%, 19.6%, and 13.5%, respectively. These three stages
account for 71.1% of rollout; GNN inference is only 5.3% there. The largest run
retained only 19 buffer states, so this slowdown is not a large-buffer effect.

The instrumenter validates original source hashes before applying, and the run
script fixes the checkpoint, horizon, episode count, batch size, and no-learning
protocol. Detailed mutually exclusive stage percentages, absolute time per
transition, source/environment identity, and remote raw-log paths are in
`benchmark_results/quarl_original_rollout_size_profile_summary_20260905.json`
and `benchmark_results/quarl_original_rollout_size_profile_findings_20260905.md`.

## CPU Quartz versus GPU paged search end-to-end throughput

The historical batch-512 matcher comparison has now been followed by an actual
beam-search A/B at beam 1000, depth 3, and GPU microbatch 512. The CPU timer
includes exact Quartz matching, candidate selection, real graph copy/apply, and
exact hash deduplication. The GPU timer includes model matching, full GPU
proposal expansion/ranking, selected-only D2H, lazy topology/update/hash,
paged-cache advance, and the final exact Quartz refresh.

On the 371-gate GF trajectory start, CPU Quartz required 276.195 seconds versus
a 3.110-second median over three GPU runs, an end-to-end **88.80x** speedup.
On the 39-gate Barenco start, CPU Quartz required 4.076 seconds versus a
3.212-second GPU median, only **1.27x**. Every GPU repeat returned 1,000/1,000
Quartz-replayable and topology-exact trajectories. Counting a redundant
second replay audit of all final states gives conservative speedups of 32.97x
and 1.09x, respectively.

Barenco needed proposal factor 64 and refresh factor 32 to fill the final beam;
the default factor-8 run retained only 162 valid states after refresh. Also,
the final 1,000 GPU states contain only 524 unique exact graph hashes for GF and
312 for Barenco because raw speculative states can merge after Quartz
normalization. Therefore these results measure time to a full legal beam with
the same best gate count, not identical beam diversity or search distribution.
See `benchmark_results/end_to_end_throughput_findings_20260906.md` and
`benchmark_results/end_to_end_throughput_summary_20260906.json` for stage
timings, raw-result checksums, and interpretation limits.
