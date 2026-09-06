# Long-tail matcher and source-topology findings (2026-09-07)

## Outcome

The remaining Barenco failure was a learned-representation failure, but the
evidence does not support increasing the whole model indiscriminately.  The old
source representation pooled gate types and positions and added a memorized
source-ID embedding; the pattern's port-aware internal topology was available
to the hard structural decoder only *after* source retrieval.  A one-layer
port-aware source-pattern encoder fixes that omission.

The selected checkpoint trains only the new source-topology residual, then
scales its zero-origin output to `0.6875`.  It adds 339,072 parameters to the
6,930,383-parameter control model (7,269,455 total, +4.89%); all pre-existing
matcher parameters remain frozen during the topology training epoch.  The
selected operating point is recall 0.999 calibration, no near reserve, and a
source-anchor cap of 10,240 for maximum-recall trajectory replay.

All teacher actions in both saved high-quality trajectories survive scoring,
calibrated thresholding, source-anchor capping, and exact structural decoding:

| Circuit and causal input | Teacher actions retained | Exact matches retained | Exact recall |
| --- | ---: | ---: | ---: |
| GF `370_2`, rolling window 8 | 271 / 271 | 559,323 / 559,524 | 0.9996407661 |
| GF `370_2`, rolling window 64 | 271 / 271 | 559,451 / 559,524 | 0.9998695320 |
| Barenco full `36_0`, rolling window 8 | 116 / 116 | 15,925 / 16,003 | 0.9951259139 |
| Barenco full `36_0`, rolling window 64 | 116 / 116 | 16,001 / 16,003 | 0.9998750234 |

The Barenco trajectory was reconstructed from all 117 saved QASM snapshots.
All 116 transitions converted successfully with authoritative
`effective_delta` records; no transition was skipped.  This is important
because direct lazy replay cannot represent parameter-dependent Quartz RZ
normalization and failed with a gate-count delta mismatch.  The windowed audit
therefore tests the model inputs used by an actual refresh-8/refresh-64
optimizer rather than treating every intermediate QASM as an unrelated
initial graph.

For clarity, the S412 control already retains all Barenco teacher actions when
refresh windows are aligned exactly at steps `0, 8, 16, ...`: it retains
15,910/16,003 exact matches for window 8 and 16,000/16,003 for window 64.  The
selected topology model raises those totals by 15 and 1 respectively.  The
specific step-20 failure occurs when that exact QASM state is presented at a
refresh boundary with no preceding action tokens.  The topology model removes
that refresh-alignment sensitivity; this is stronger than merely showing that
one fixed refresh schedule happens to carry enough recent history.

This result establishes candidate coverage, not autonomous search-policy
reproduction.  Action ordering, depth allocation, and selection from the
retained set remain the responsibility of the search branch.

## Why the old model failed

The missed Barenco action is step 20, xfer 1525, source 1467, with binding
slots `(28, 40, 42, 43, 45)`.  Its source and destination patterns are:

```text
source:      cx 2 1; cx 1 0; rz 0; cx 2 0; cx 2 1;
destination: cx 1 0; cx 0 2; rz 2; cx 0 2;
```

The five matched nodes are adjacent along their circuit wires, but many
unrelated global slots occur between them.  The old selected matcher assigned
the source-anchor pair raw logit `-6.237612`, below its far threshold
`-5.537871`; increasing the downstream action cap could not restore an item
already removed at this stage.

This is not explained by raw source frequency alone.  Source 1467 is a positive
in 289 training states and has 292 concrete training matches.  Their binding
slot spans have minimum/median/90th-percentile/maximum `4/4/10/173`; only 27
have span at least 17 and only one has span exactly 17.  Thus the source ID is
not rare, but Barenco-like interleaved contexts are a small minority of its
examples.

The larger corpus also makes the representation bottleneck visible:

- 3,855 source classes exist;
- 958 have no positive state in the training split, and 954 of those are
  five-gate sources;
- 907 of the 958 unseen sources share a gate-type sequence with an observed
  source, but none shares the same complete port-aware topology signature.

The old source composer could share gate-type/position statistics across these
classes, but it could not use their differing wire/port connections until the
post-retrieval exact decoder.  A wider generic MLP would still receive the same
incomplete structural input.  The added graph layer gives the retrieval model
the missing information directly.

## Independent generalization audit

The strict audit uses 1,536 held-out states after excluding the 512-state
calibration trajectory partition.  It contains 574,851,236 eligible
source-anchor pairs and 1,607,788 exact bindings.  Neither Barenco nor GF is in
this training or strict-audit partition.

| Metric at recall 0.999, cap 8192 | Control S412 | Selected topology 0.6875 | Change |
| --- | ---: | ---: | ---: |
| Exact bindings finally retained | 1,598,001 | 1,598,033 | +32 |
| Exact-binding recall | 0.9939127547 | 0.9939326578 | +0.00199 percentage points |
| Teacher actions retained | 1,528 / 1,536 | 1,528 / 1,536 | unchanged |
| Pairs after the global source cap | 11,841,065 | 12,073,928 | +1.97% |
| Structurally decoded candidates | 1,981,966 | 1,935,580 | -2.34% |
| States retaining every exact binding | 334 | 324 | -10 |

The aggregate gain is small because common sources dominate the 1.6 million
exact matches.  The frequency and pattern-length slices show where the new
representation helps:

| Training source-frequency bucket | Control recall | Topology recall |
| --- | ---: | ---: |
| unseen | 0.0238095 | 0.0297619 |
| exactly 1 state | 0.0025000 | 0.0037500 |
| 2--7 states | 0.5864407 | 0.6508475 |
| 8--63 states | 0.7787409 | 0.7823905 |
| 64--511 states | 0.9244840 | 0.9416009 |
| 512+ states | 0.9954045 | 0.9952782 |

Five-gate exact recall improves from `0.9572797` to `0.9623999`.  The slight
regression in the most common bucket is why the topology residual is scaled
rather than used at full strength.  Importantly, the strict teacher total does
not regress and overall exact recall is slightly higher than the control.

The ordinary 2,048-state held-out full-binding Top-N metric is `0.9016634`,
versus `0.9020189` for the control: a decrease of 0.03555 percentage points.
This small aggregate tradeoff is retained in the result rather than hidden.

## Candidate cap and the GF boundary

At cap 8192, the selected topology model retains 270/271 GF actions for the
rolling-8 representation.  The missing step 251 is not a threshold miss:

```text
source_id                    3759
xfer_id                      4382
binding                      (968, 971)
raw logit                    -5.808431
near raw threshold           -7.218452
rank among above-threshold   9306
```

The pair is confidently above threshold but loses a global ranking tie-break
against a dense candidate set.  Raising the source-anchor cap by 25%, from
8192 to 10240, retains it and restores 271/271.  At the larger cap the rolling-8
GF exact-match total is 559,323/559,524, exactly the control model's total at
cap 8192.  Rolling-64 GF already retains 271/271 at cap 8192.

This is deliberately not called another model-learning failure: the matcher
has classified the pair as a candidate, and the user-facing requirement here
is candidate retention rather than final ordering.  The downstream search can
still apply a smaller action budget after structural decoding and learned or
heuristic ranking.  No circuit-specific reserve or special pattern rule is
used.

For Barenco step 20, by contrast, the selected topology checkpoint changes the
raw logit to `-5.579756` against threshold `-5.675974`, a positive raw margin of
`0.096218`; its unfiltered probability rank is 737.  A weaker residual of
0.625 also passes, but with only `0.010577` raw margin, so 0.6875 is selected
for robustness.

## Ablations

| Variant | Ordinary held-out full-binding Top-N | Barenco full teacher coverage | Decision |
| --- | ---: | ---: | --- |
| Control continuation S412 | 0.9020189 | 115 / 116 | baseline |
| Source-ID/bias adapter only | 0.897576 | 115 / 116 | frequency memorization does not fix the context |
| Interleaving-positive loss, weight 4 | 0.900601 | 115 / 116 | slot span alone is not the missing invariant |
| Source-balanced full-model interpolation 0.6875 | not selected | 116 / 116 | strict recall fell to 0.9924922 and teacher to 1525/1536 |
| Topology-only, residual 0.625 | 0.9017170 | 116 / 116 | good strict recall, but Barenco margin only 0.0106 |
| Topology-only, residual 0.6875 | 0.9016634 | 116 / 116 | selected |
| Topology-only, residual 0.75 | 0.9014720 | 116 / 116 | no additional coverage benefit |

Shrinking rare/unseen source-ID terms with a frequency prior was also rejected:
it increased calibrated candidate counts sharply and reduced the ordinary
held-out metric.  This further supports learning shared topology rather than
merely suppressing or boosting memorized source IDs.

## Implementation

The branch adds the following reusable mechanisms:

1. A static, port-aware graph representation for every ECC source pattern.
   Edge relations encode `(source_port, destination_port)`.  The topology
   output layer is zero-initialized, so loading an old checkpoint starts with
   exactly the old behavior.
2. `--source-topology-only` training, which freezes the original matcher and
   trains only the shared topology branch.
3. Optional bounded inverse-frequency positive weighting, hard-positive
   threshold gating, deterministic target-source-balanced sampling, and
   interleaved-binding weighting.  These remain experimental controls; they
   are not silently enabled in the selected model.
4. Strict audit splitting that excludes calibration trajectories and reports
   recall by source frequency and source length.
5. A checkpoint utility that scales only the zero-origin topology residual.

Source topology is static for an ECC set.  The optimizer already computes and
caches source representations once before batched rollout; the state-dependent
source-anchor logit tensor and downstream decoder shapes are unchanged.  The
new graph layer therefore does not add a topology pass for every search state.
Candidate decode work can rise because the selected calibration intentionally
retains more source-anchor pairs; that effect is separately bounded by the
10240 cap.

Focused validation on the H100 environment passes 20/20 unit tests, including
source-frequency weighting, deterministic balanced sampling, topology-only
freezing, interleaving loss, hard-positive gating, refresh rebasing, and action
loss behavior.  All changed Python files also pass `py_compile`, and
`git diff --check` is clean.

## Selected artifacts

- Checkpoint:
  `/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/paged_action_matcher_control_s412_sourcetopo1_only_lr1e4_a06875_s414.pt`
  - SHA-256: `a27088b0bafccc0c43c37598814f676555b7eed3c2cbdc9d37413d968f178879`
- Recall-0.999 calibration:
  `/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/paged_action_matcher_control_s412_sourcetopo1_only_lr1e4_a06875_s414_calibration_r999.json`
  - SHA-256: `50bfebabe4bf7dfc591662592506cef41f9277922930eca2aaa9582fb5a04025`
- Training data:
  `/SharedData/dengzy/quarl_matchformer_fresh_20260902/data/binding_longmix_randomrefresh_complex_holdout_20260906.pt`
  - SHA-256: `e9f03f64f856d911ed846a6e3a3ee7dc1eb8030757d212a13d508c3bdaba1202`
- Normalized Barenco full trajectory:
  `/SharedData/dengzy/quarl_matchformer_fresh_20260902/data/quarl_barenco_fullseq36_normalized_20260907.pt`
  - SHA-256: `b3df0caf10cf09986038689d2bc70354c1967ce635d4d2c7687c1c17f26d2604`

The compact machine-readable comparison is
`matcher_long_tail_topology_summary_20260907.json`.  The selected raw trajectory
audits and calibration are stored beside this report.  Full strict-audit JSONs
remain in the remote `runs` directory because each contains per-state records
and is about 2 MB; the compact summary records the comparison needed for the
branch.

## Remaining limitations

- Eight of 1,536 strict-audit teacher actions are still absent at the selected
  operating point, exactly the same count as the control.  The change solves
  the identified Barenco source-pattern failure; it is not a claim of perfect
  recall on every possible circuit.
- Unseen-source recall improves but remains very low.  No supervised objective
  can learn the behavior of all 958 unseen classes from positive examples that
  do not exist.  Further improvement should generate circuits that cover
  distinct source topology signatures, especially five-gate patterns, rather
  than only increasing examples for already-common source IDs.
- The topology scale was selected after checking Barenco/GF coverage.  The
  topology weights themselves were trained without either target trajectory,
  and the disjoint strict audit provides independent evidence, but the final
  scalar selection is not a fully blind benchmark choice.
- A 10,240 source cap increases the worst-case pre-decode budget by 25% over
  8192.  It is appropriate for the high-recall matcher boundary; the search
  branch should reduce the much smaller expanded action set using its own
  width/depth policy.
