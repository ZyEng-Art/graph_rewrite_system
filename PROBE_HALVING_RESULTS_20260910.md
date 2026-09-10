# Deterministic probe-halving experiment (2026-09-10)

## Decision

Reject `probe_halving` for production and retain deterministic `round_robin` as the production branch-revisit policy. Neither probe version met the predeclared development gate of zero final-gate regressions, so no 100k run and no external holdout were performed.

The policy remains an explicit, off-by-default experiment so its implementation and negative result are reproducible. It is not selected by the main CLI default.

## What was implemented

`SearchNodeStats` now records both lifetime feedback and the most recent actually scanned rank band:

- attempted, valid, unique, duplicate, invalid, and immediately improving action counts;
- useful, valid, and improving yields for the last band;
- best immediate child gate count and last expansion step.

`probe_halving` groups surviving branches by their exact origin parent and ranks them only with selection-time, last-band outcomes. Tie-breaking remains deterministic. Insufficient sibling cohorts fall back to round-robin, so the revisit beam is never intentionally shrunk.

Version 1 reserved 50% of revisit slots for round-robin safety and used 50% for the current top half of each sibling cohort. It recomputed the top half on every selection. Version 2 added a persistent `probe_level` to each `BeamState`: only branches at the cohort's highest tournament level can race for the next level. It also increased the safety allocation to 75%, leaving 25% for promoted probes.

The runner gained `COLLECT_NEURAL_AUDIT=off` for live A/B tests. This disables neural audit tensors and descendant labels while preserving the full search result JSON, log, environment manifest, and wall-time file. The default remains `on`, so existing label-collection workflows are unchanged.

## Development protocol

Both candidates were compared against clean `round_robin` runs on the existing five 58-gate development circuits, with beam size 256, deterministic search, and a 20,000 attempted-action budget. Continuation scoring and neural-audit collection were disabled for both sides. The promotion rule and safety fraction were fixed before each version ran.

The stop condition was strict: any worse final gate count prevents escalation to 100k or an untouched holdout. This avoids spending the external holdout by tuning on it.

## Version 1 result

Version 1 produced 0 wins, 3 ties, and 2 losses in final gate count. Mean best gate count worsened from 46.4 to 47.0; mean first-best step worsened from 24.2 to 25.8; runtime increased 26.91%; and unique graphs increased 11.98%.

The targeted lane was active: 4,725 probe promotions, 4,776 round-robin safety selections, and only 41 fallback selections. The failure was therefore not caused by silently falling back to the baseline. Recomputing cohort membership every step allowed noisy short-term winners to enter and leave repeatedly, so this was not strict successive halving.

## Version 2 result

Persistent levels and a larger safety lane improved the win/tie/loss count to 1/2/2, but still failed the zero-regression gate:

| Circuit | Round-robin gate | Probe gate | Round-robin best step | Probe best step |
|---|---:|---:|---:|---:|
| `1_58_0_21_4409` | 48 | 46 | 21 | 31 |
| `2_58_0_28_120` | 46 | 48 | 28 | 25 |
| `3_58_0_29_336` | 46 | 48 | 29 | 25 |
| `4_58_0_23_45` | 46 | 46 | 22 | 20 |
| `5_58_0_12_38` | 46 | 46 | 21 | 23 |

Aggregate changes were:

- mean best gate: 46.4 to 46.8, worse by 0.4;
- mean first-best step: 24.2 to 24.8, worse by 0.6;
- mean first-best time: +18.88%;
- mean total search time: +24.73%;
- mean unique graphs: +7.64%.

The v2 lane totals were 6,824 round-robin safety selections, 2,255 persistent tournament promotions, and 15 fallbacks. Persistent halving therefore operated as designed, but its short-band reward was not a reliable proxy for eventual gate reduction.

## Interpretation

This experiment falsifies a stronger claim than the earlier observational ranker tests. Even when the policy actively assigns probe bands, preserves tournament state, and limits learned/heuristic allocation to one quarter of revisit slots, immediate improvement and unique yield can redirect enough budget to lose two gates on unseen paths within the same development circuit.

The one winning circuit also matters: active probes can discover a better route. The problem is variance and asymmetric damage, not a total absence of signal. However, the current signal is too unreliable for production.

Further tuning of safety fractions, feature weights, or promotion thresholds on these five circuits would overfit the development set. A justified next experiment would need a different observable signal—most plausibly a bounded lookahead that measures the best gate reachable after a fixed multi-band probe, rather than one-band immediate yield—and a newly frozen development/holdout protocol. Until then, `round_robin` remains the evidence-backed answer.

Machine-readable results are in `benchmark_results/probe_halving_dev_results_20260910.json`. Raw results and environment manifests are retained under `$AUDIT_ROOT/v11_*` and `$AUDIT_ROOT/v12_*`.
