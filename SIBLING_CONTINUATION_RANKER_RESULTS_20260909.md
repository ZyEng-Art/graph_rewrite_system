# Sibling continuation labels and ranker

Date: 2026-09-09

Branch: `deterministic-feedback-widening-20260909`

Implementation commits: `f758bb1`, `d5fcdc0`

## Outcome

The search can now export supervision for the actual branch-allocation question:
among actions attempted from the same exact parent, which child later reaches the
best observed descendant gate count? The end-to-end H100 smoke produced real
non-tied sibling preferences, and a compact pairwise ranker trains successfully.

The first cross-circuit result is not strong enough to control live widening. On
an isolated raw58 circuit, the ranker achieved 63.41% pair accuracy. That beats
the current lower-action-rank heuristic (59.15%) but loses to simply preferring
the higher matcher probability (74.09%). The ranker and labels therefore remain
offline-only; no default search behavior changed.

## What was added

- `--neural-descendant-labels on`, used with `--neural-audit-output`, augments
  frozen candidate rows after search with parent/child search-node ids, sibling
  groups, action rank, expansion round, stagnation, depth, prefix history, best
  descendant gate, continuation gain, time to the observed best, remaining
  observation window, and right-censoring.
- Invalid attempts receive no descendant label. Duplicate children are retained
  for audit completeness but excluded from preference training because their
  descendants may have been explored before the current edge, which would leak
  future information.
- `build_sibling_continuation_preferences.py` constructs same-parent pairs,
  deduplicates exact children, verifies audit checksums, and keeps validation
  circuits separate.
- `train_sibling_continuation_ranker.py` trains a small pairwise head on frozen
  matcher/action features and inference-time parent context. It verifies source
  SHA-256 values and reports matcher-probability, gate-delta, and action-rank
  baselines.

## Label semantics and limitation

These are observational search labels, not complete counterfactual values. A
positive continuation gain proves that a child led to a better observed
descendant. A zero is marked right-censored because a larger or differently
allocated budget might still improve it. The training inputs exclude all
post-search outcomes and remaining-horizon fields.

The current 3,000-action smoke used `--min-remaining-steps 1` to verify the data
path. A production corpus should use a materially longer common horizon or forced
per-sibling rollouts; otherwise early-created branches receive more opportunity
than late-created branches.

## H100 validation

The combined regression suite ran 16 tests successfully, including exact cached
rank bands, progressive widening, raw candidate caching, descendant-label
alignment, and pair construction. Four additional continuation/ranker tests also
passed.

The raw58 smoke contained:

| Quantity | Count |
|---|---:|
| Attempted candidate rows | 3,000 |
| Invalid / duplicate / unique | 328 / 1,913 / 759 |
| Unique children with observed later gain | 188 |
| Unique right-censored children | 571 |
| Sibling groups | 448 |
| Non-tied preference pairs | 164 |
| Audit size | 4.665 MiB |

For the cross-circuit check, circuits `100_38_0_17_44` and
`10_56_0_15_45` supplied 243 training pairs; `0_58_0_8_121` supplied 164
held-out pairs. Circuit `106_41_1_11_4356` had no non-tied pair and was excluded.

| Selector | Train accuracy | Held-out raw58 accuracy |
|---|---:|---:|
| Learned continuation ranker | 89.71% | 63.41% |
| Higher matcher probability | 81.28% | 74.09% |
| Lower action rank | 70.37% | 59.15% |
| Lower immediate gate delta | 48.56% | 48.78% |

The large train/held-out gap is evidence of overfitting and circuit shift, not a
reason to tune against this one held-out circuit.

Machine-readable results are in
`benchmark_results/sibling_continuation_ranker_20260909.json`. Audits, manifests,
checkpoints, JSON outputs, metadata, and complete process logs are retained at:

```text
/SharedData/dengzy/quarl_matchformer_fresh_20260902/experiment/
  sibling_continuation_audit_20260909/
```

## Next gate before live branch control

1. Collect many circuits with a fixed post-child exact-apply horizon, stratified
   across Barenco and GF sizes. Keep entire circuits and rewrite families out of
   training for calibration/test.
2. Train as a regularized residual over matcher probability so the model must
   demonstrate incremental continuation information rather than relearn a strong
   existing signal from a few hundred pairs.
3. Add the exported prefix histories through a compact sequence encoder. The
   current smoke ranker deliberately tests the action/parent features first and
   does not yet claim sequence conditioning.
4. Calibrate against censoring (or use forced sibling rollouts) and require both
   higher held-out pair accuracy than matcher probability and improved
   best-gate/time-to-improvement under an equal exact-apply budget.
5. Only then attach the score to feedback-widening revisit lanes in shadow mode,
   followed by deterministic A/B. Until those gates pass, the current feedback
   scheduler remains authoritative.
