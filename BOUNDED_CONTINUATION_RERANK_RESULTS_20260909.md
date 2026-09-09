# Bounded continuation live-rerank experiment

Date: 2026-09-09

Branch: `deterministic-feedback-widening-20260909`

Implementation commits: `c49a0b9`, `df5c35d`, `cdf5801`

## Outcome

The continuation ranker can now influence live search behind a non-default
`bounded` mode. It is deterministic, preserves the global parent/gate allocation,
and did not worsen final gate count in the evaluated circuits. It also did not
improve final gate count on the five prospectively frozen 58-gate holdouts.

The mode therefore remains experimental and off by default. The result suggests
that swapping near-tied actions is too local to solve branch-depth allocation:
under a large apply budget, both actions are often attempted anyway. The next
useful integration point is bounded prioritization of which materialized child
branch receives a later widening revisit, first in shadow/audit form.

## Safety boundary

`--continuation-ranker-mode bounded` may only swap candidates that:

- come from the same exact parent;
- have the same immediate next gate count;
- differ in matcher logit by no more than the calibrated threshold;
- differ in continuation score by at least the calibrated threshold.

Each parent receives at most one promotion per layer. A bucket keeps exactly the
same occupied positions in the global proposal queue, so the ranker cannot move
an action across another parent or immediate gate-count bucket. The mode is
disabled unless a compatible checkpoint is explicitly provided.

## Threshold calibration

The threshold grid was selected using the 553 balanced-search training pairs
only. The recommended setting was then evaluated, without retuning, on the 308
frozen natural-search validation pairs:

```text
maximum matcher logit gap:       0.05
minimum continuation margin:     0.10
maximum promotions/parent/layer: 1
```

| Split | Pairs | Interventions | Corrected matcher errors | Harmed matcher decisions |
|---|---:|---:|---:|---:|
| Training | 553 | 40 | 40 | 0 |
| Frozen validation | 308 | 37 | 37 | 0 |

The training intervention precision Wilson lower bound was 91.24%; the frozen
validation lower bound was 90.59%. These pairwise figures establish a sensible
abstention boundary, but do not prove end-to-end search improvement.

## Frozen validation live smoke at 20k actions

Circuits 110-116 were rerun with the no-prefix ranker in bounded mode. All seven
had the same final best gate count and the same best-first-seen step as their
shadow-off baselines. The ranker made 1,999 promotions in total (255-337 per
circuit). Circuit 114's maximum action depth changed from 12 to 13; all other
maximum depths were unchanged.

These circuits reach a 36-gate floor quickly and have little power to distinguish
long continuation behavior, so they are safety checks rather than evidence of
quality gain.

## Raw58 at 20k and 100k actions

At 20k actions, bounded mode made 442 promotions. Both modes reached the exact
same 48-gate best graph at step 24, with the same best history and action depth.
The final beam differed, as expected after live reranking.

At 100k actions, both modes reached 38 gates, but bounded mode found its best
earlier:

| Metric | Off | Bounded | Difference |
|---|---:|---:|---:|
| Best gate count | 38 | 38 | 0 |
| Best-first-seen step | 77 | 66 | -11 |
| Best-first-seen seconds | 16.6833 | 14.0723 | -2.6110 |
| Best action depth | 69 | 56 | -13 |
| Maximum action depth | 86 | 90 | +4 |
| Accepted actions | 18,111 | 19,239 | +1,128 |
| Unique graphs seen | 24,086 | 25,546 | +1,460 |
| Search seconds | 21.3865 | 22.8357 | +1.4491 |

Bounded mode made 1,804 promotions and spent 0.3726 seconds in the continuation
scorer. A second deterministic bounded run reproduced the best graph digest,
final beam digest, best history, accepted/unique counts, promotion count, and the
complete improvement trace after removing timing fields.

This is a real deterministic path improvement, but raw58 had already been
examined during earlier development and is not a prospective holdout.

## Prospective five-circuit 58-gate holdout at 100k actions

The file `sibling_continuation_live_holdout_qasms_20260909.txt` was committed
before any bounded result was observed. None of its circuits participated in
continuation training or threshold calibration.

| Circuit | Off best | Bounded best | Off best step | Bounded best step | Step delta | Promotions |
|---|---:|---:|---:|---:|---:|---:|
| `1_58_0_21_4409` | 38 | 38 | 54 | 59 | +5 | 1,987 |
| `2_58_0_28_120` | 38 | 38 | 51 | 51 | 0 | 1,840 |
| `3_58_0_29_336` | 38 | 38 | 60 | 52 | -8 | 1,933 |
| `4_58_0_23_45` | 38 | 38 | 43 | 42 | -1 | 1,915 |
| `5_58_0_12_38` | 38 | 38 | 62 | 63 | +1 | 1,672 |

Aggregate results:

- final gate wins/ties/losses: **0 / 5 / 0**;
- time-to-best step: two earlier, one equal, two later;
- mean best-step delta: -0.6 step;
- mean best-seconds delta: +0.3407 seconds (bounded slower);
- mean unique-graph delta: -514.6;
- mean search-time delta: +0.5534 seconds;
- mean promotions: 1,869.4 per circuit.

There is no prospective evidence of improved final solution quality. The mixed
time-to-best and graph-coverage changes also show that perfect accuracy on a
small subset of labeled sibling pairs does not translate directly into globally
better beam allocation.

## Reproducibility

The A/B runner records the Git commit, host, GPU, Python binary, QASM list,
checkpoint, budget, thresholds, per-run stdout/stderr, JSON output, and wall time:

```text
./run_continuation_bounded_ab.sh \
  sibling_continuation_live_holdout_qasms_20260909.txt \
  "$OUTPUT_DIR" 1 "$CHECKPOINT" 100000 0.05 0.1 1
```

The relevant regression suite passed 32/32 tests. Full experiment artifacts are
retained at:

```text
/SharedData/dengzy/quarl_matchformer_fresh_20260902/experiment/
  sibling_continuation_audit_v3_20260909/
    v4_bounded_rerank/
    v4_bounded_live_holdout5_100k/
    v4_balanced8_natural7_noseq_rerank_calibration.json
    v4_balanced8_natural7_noseq_rerank_calibration.log
```

Machine-readable headline results are in
`benchmark_results/bounded_continuation_rerank_20260909.json`.

## Next gate

Do not enable bounded action reranking by default. Preserve it as a diagnostic
control and move the continuation signal to a branch-level shadow audit:

1. attach the generating candidate's continuation score to each novel child;
2. measure whether that score predicts later improvement conditional on equal
   child exposure;
3. compare it with the existing observed-gain, novelty, gate, and expansion-round
   lanes used by `feedback` widening;
4. only if it adds held-out predictive value, reserve a very small bounded revisit
   lane without removing the existing feedback lanes;
5. evaluate final best gate and worst-circuit regression under equal 100k+ action
   budgets.
