# Balanced continuation supervision and shadow validation

Date: 2026-09-09

Branch: `deterministic-feedback-widening-20260909`

Implementation range: `2c9c743..f96f4a2`

## Decision

Keep the continuation ranker in shadow mode. Balanced sibling exposure produces
a statistically positive held-out ranking result, and the online scorer now
reproduces offline pair decisions without changing the search. The evidence is
not yet broad enough to let the ranker control widening: the independent raw58
result is negative and demonstrates material circuit shift.

The no-prefix model is the current deployment candidate. Once histories are
correctly aligned per candidate row, the GRU prefix encoder has the same held-out
accuracy as the simpler model, while adding latency and more ways to overfit.

## Why balanced collection was needed

Natural feedback search rapidly concentrates budget on a few branches. That is
efficient for optimization but gives poor counterfactual supervision: many
sibling branches are observed only once. At an equal 20,000 attempted-action
budget on raw58, `feedback_balanced` changed coverage as follows:

| Quantity | Natural feedback | Balanced feedback | Change |
|---|---:|---:|---:|
| Unique children | 7,562 | 8,822 | +16.7% |
| Expanded unique children | 5,631 | 6,556 | +16.4% |
| Children expanded at least twice | 861 | 1,825 | +112.0% |
| Exposure-1 sibling pairs | 234 | 308 | +31.6% |
| Expansion-gap-1 sibling pairs | 179 | 282 | +57.5% |

Balanced feedback is a data-collection policy, not the proposed production
scheduler. It round-robins sibling cohorts before falling back to the normal
feedback lanes, giving labels a more comparable observation budget.

## Frozen v4 evaluation

Eight balanced-search circuits (`101`-`105`, `107`-`109`) supplied 553 training
pairs from 445 sibling groups. Seven natural-search circuits (`110`-`116`) were
frozen as evaluation-only before this run and supplied 308 pairs from 245 sibling
groups. Each search used the same deterministic 20,000 attempted-action budget.

The ranker is a zero-initialized residual over the frozen matcher logit. It can
learn continuation evidence without having to relearn the already useful matcher
score.

| Ranker | Pair accuracy | Group macro accuracy | Matcher group macro | Delta vs matcher | Group bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| No rewrite prefix | 84.42% | 83.54% | 78.44% | +5.10 pp | [+1.94, +8.06] pp |
| Row-aligned GRU prefix | 84.42% | 83.54% | 78.44% | +5.10 pp | [+2.14, +8.06] pp |

`group macro accuracy` first averages decisions within each parent sibling group,
then averages parents. It prevents a few parents with many candidate pairs from
dominating the result. The bootstrap resamples sibling groups rather than rows.

The gain is not uniform across circuits. Six of the seven frozen validation
circuits have a non-negative aggregate delta, while circuit 116 is negative. More
importantly, the no-prefix model scored 35.38% group accuracy on the independent
raw58 audit versus 43.47% for the matcher, a delta of -8.09 pp with a clustered
95% interval of [-13.12, -2.89] pp. That failure is why live reranking remains
disabled despite the positive frozen-set aggregate.

## Audit correctness fixes

Two mismatches were found before enabling control:

1. An exact graph node can be reached through multiple rewrite histories. The v3
   audit stored one history per node id, so a sequence model could receive the
   wrong path. The v4 format stores `parent_history_xfer_ids` for every candidate
   row. Sequence training and inference reject legacy, non-row-aligned data.
2. Audit features are serialized as FP16, but the first shadow implementation fed
   live full-precision values to the ranker. Live scoring now applies the same
   FP16 round trip as corpus loading.

The earlier v3 sequence result is superseded and must not be used. The v3
no-prefix result did not consume the faulty histories, but all headline metrics
above were regenerated from v4 audits.

## Online shadow A/B

Circuit `115_38_2_11_37` was rerun with the v4 prefix checkpoint in shadow mode.
The scorer evaluated 59,238 proposals; the audit retained the same 20,000
attempted-action rows as the shadow-off baseline.

| Check | Result |
|---|---:|
| Non-timing audit structural differences | 0 |
| Finite stored scores | 20,000 / 20,000 |
| Scores exactly reproduced offline | 19,938 / 20,000 |
| Maximum absolute score difference | 0.00390625 |
| Exposure-matched sibling pairs | 62 |
| Online / offline pair accuracy | 83.87% / 83.87% |
| Online/offline pair-order disagreements | 0 |
| Minimum absolute online pair margin | 0.013671875 |

The 62 small score differences are expected FP16 GEMM batch-shape rounding: live
proposal batches and compact offline audit batches have different shapes. They
are below 0.01 and did not change any labeled pair decision. The reusable
`verify_continuation_shadow.py` command now enforces structural equality, finite
scores, a numerical tolerance, and zero pair-order disagreements.

The scorer used 0.2676 seconds, 7.38% of the measured search loop. Search time
changed from 3.2899 to 3.6270 seconds. End-to-end wall time, dominated by audit
materialization and serialization, changed from 76.2117 to 76.6898 seconds
(+0.4781 seconds, +0.63%). These are single-run engineering measurements, not a
multi-run latency confidence interval.

## Reproducibility and retained artifacts

The combined regression suite passed all 25 tests covering continuation labels,
preference construction, prefix alignment, OOV rewrite ids, shadow scoring,
balanced widening, action caching, and candidate caching.

Machine-readable headline results are in
`benchmark_results/balanced_continuation_shadow_20260909.json`. Full audits,
checkpoints, manifests, process logs, wall times, and the verifier output are
retained at:

```text
/SharedData/dengzy/quarl_matchformer_fresh_20260902/experiment/
  sibling_continuation_audit_v3_20260909/
    v4_balanced_corpus8/
    v4_validation7/
    v4_balanced8_natural7_exposure1_preferences.pt
    v4_balanced8_natural7_noseq_ranker.pt
    v4_balanced8_natural7_seq_ranker.pt
    v4_shadow_validation115/
```

The parent directory retains its historical `v3` name; the valid audit payloads
inside the listed directories declare `frozen_candidate_successor_descendant_v4`.

## Next experiment gate

1. Expand balanced training across circuit families and sizes; report
   leave-one-circuit-out results rather than selecting against raw58.
2. Calibrate a confidence/abstention rule on held-out circuits. The ranker should
   only be allowed to influence near-tied matcher candidates when its estimated
   benefit is supported out of distribution.
3. Add a deterministic, bounded rerank mode behind a non-default flag and compare
   best gate count and time-to-improvement under exactly equal action budgets.
4. Promote it only if aggregate gains survive per-circuit analysis and no circuit
   family shows a severe regression. Until then, `feedback` remains authoritative
   and continuation scoring remains observational.
