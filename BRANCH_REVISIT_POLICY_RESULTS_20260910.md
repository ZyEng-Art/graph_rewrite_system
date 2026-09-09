# Branch revisit policy results (2026-09-10)

## Outcome

Use deterministic `round_robin` for production branch revisits. Do not enable the learned continuation-revisit ranker, `feedback_ucb`, or `feedback_marginal` based on the current evidence. The main CLI already defaults to `round_robin`; `run_sibling_continuation_corpus.sh` selects feedback policies only when explicitly requested for audit-label collection.

On the prospectively frozen five-circuit 56/58-gate holdout at 100,000 attempted actions, `round_robin` versus `feedback` produced one better final gate count, four ties, and no losses. Mean best gate count improved from 38.6 to 38.2; mean first-best step fell from 66.4 to 57.8; and mean runtime fell from 22.08 s to 16.25 s. The winning circuit improved from 40 gates to 38 gates.

## Why the learned branch ranker was rejected

The first lightweight linear ranker used only fields available at revisit-selection time: origin continuation score, action depth, attempted actions, accumulated descendant gain, novel yield, and valid yield. Training pairs were restricted to the same circuit, selection step, gate count, expansion round, prior exposure count, and future additional exposure. Pair weights equalized circuits and selection opportunities.

Results initially looked strong on the 38-gate-area split:

- Balanced eight-circuit training group-macro accuracy: 77.60%, versus 64.96% for novel yield.
- Seven whole-circuit validation group-macro accuracy: 81.98%, versus 66.30% for novel yield; paired opportunity bootstrap delta +15.68 percentage points, 95% interval +11.39 to +19.79 points.
- Removing `origin_continuation_score` retained 81.40% validation accuracy and a +15.10-point delta (95% interval +10.91 to +19.22). The old continuation network was therefore not the useful signal.

The prospective 58-gate diagnostic set invalidated the apparent generalization:

- No-origin model: 52.58% group-macro versus 50.03% novel yield. The +2.55-point bootstrap interval was +0.40 to +4.78, but absolute prediction was near chance.
- Including the old continuation score was worse at 52.10%; continuation score alone was 48.16%.
- Within-selection percentile features retained 81.94% on the old validation split but reached only 51.61% on the 58-gate diagnostic set; its delta interval crossed zero.
- Leave-one-circuit-out training on four natural 58-gate traces and validation on the fifth averaged 42.30%, below the 50.16% novel-yield baseline.
- Collecting balanced sibling exposure did not repair this. Training on four balanced 58-gate traces and validating on the fifth circuit's natural feedback trace averaged 47.18%, again below 50.16%.

The correct interpretation is that aggregate online statistics and observational future-gain labels are policy-confounded and circuit-specific. The 81% figure was not deployable cross-scale branch-value accuracy.

## Rejected deterministic feedback variants

Two bounded, opt-in policies were implemented so their hypotheses could be tested without changing defaults.

`feedback_ucb` replaces only the novelty lane with a 95% Wilson upper confidence bound on exact-unique children per attempted action. At 20k on the five-circuit development set it had identical best gate, best step, and best digest on all five circuits, with only +0.20% unique graphs and +0.057 s mean time. At 100k it had 0 wins, 5 gate ties, and 0 losses; mean first-best step worsened from 54.0 to 57.2, unique graphs increased 4.48%, and runtime increased 1.72%. More coverage did not improve quality.

`feedback_marginal` reverses the accumulated-descendant-gain preference for one lane, testing whether already-realized gain represents exhausted headroom. At 100k it produced 0 wins, 4 ties, and 1 loss; mean best gate worsened from 38.0 to 38.2 and mean first-best step worsened by 10. It was rejected.

Both policies remain explicit experimental options and are off by default.

## Round-robin live evidence

On the same five 58-gate development circuits at 100k, `round_robin` versus `feedback` had 0 wins, 5 ties, and 0 losses in final gate count. It reached the best result three steps earlier on average (54.0 to 51.0) and reduced mean runtime by 27.72% (22.46 s to 16.23 s). It visited 17.89% fewer unique graphs, showing that feedback's extra coverage was not productive.

The final holdout list was committed at `da4e85a` before these circuits were run:

- `0_58_0_8_121`
- `6_58_2_24_3320`
- `7_56_0_16_44`
- `8_56_0_27_1593`
- `9_56_0_9_121`

At 100k, the per-circuit final gate counts for feedback versus round-robin were 38/38, 38/38, 40/38, 38/38, and 39/39. The corresponding first-best steps were 77/68, 79/46, 73/44, 44/48, and 59/83. Thus gate quality was 1 win / 4 ties / 0 losses; three circuits reached their best earlier and two later. Mean first-best wall time fell 30.04%, and total runtime fell 26.39%.

The winning `7_56_0_16_44` round-robin run was repeated independently. Best gate (38), first-best step (44), best action depth (38), best graph digest, final beam digest, full best history, attempted actions, unique graphs (17,947), and accepted actions (13,503) were all exactly equal. Only wall time varied, from 14.78 s to 15.44 s.

Machine-readable results are in `benchmark_results/round_robin_revisit_holdout_20260910.json`. Raw remote corpora are under `$AUDIT_ROOT/v6_*` through `$AUDIT_ROOT/v10_*`; each directory includes its environment manifest, per-circuit JSON, audit tensor, log, and wall-time file.

## Recommendation and next boundary

Keep `round_robin` as the production default and keep feedback policies confined to labeled-data collection or explicit experiments. This is not a learned ability to identify a uniquely promising branch; it is stronger evidence that, with current observable features, attempting to make that prediction is worse than deterministic fair allocation.

If a later iteration must outperform round-robin rather than merely retain it, the next experiment should add active counterfactual probes: give competing branches the same small next-band budget, record recent band-level improving/unique outcomes (not lifetime aggregates), and only then allocate additional bands using a predeclared successive-halving rule. That requires new causal data; further fitting of the current audit rows is not justified.
