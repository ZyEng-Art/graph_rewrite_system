# Proposal Top-N Legality Audit

## Scope

- No training was run.
- H100 inference used beam size 1000, target recall 0.95, per-parent cap 128,
  max gate increase 3, and exact Quartz replay for audited proposals.
- `conditional` means that the parent action prefix was valid in Quartz and only
  the newly proposed action is being judged.
- `sequence` additionally counts proposals whose parent prefix was already
  invalid. Conditional precision is therefore the one-step action metric.

## Depth 8 conditional action precision

| circuit | ranking | Top-1 | Top-8 | Top-32 | Top-128 | Top-1000 | final beam |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| barenco_tof_3 | probability | 100.00% | 100.00% | 100.00% | 100.00% | 99.86% | 1000 |
| barenco_tof_3 | gate | 100.00% | 100.00% | 100.00% | 100.00% | 99.86% | 1000 |
| barenco_tof_3 | value | 100.00% | 100.00% | 96.62% | 89.68% | 93.13% | 1000 |
| mod5_4 | probability | 100.00% | 100.00% | 100.00% | 100.00% | 99.87% | 1000 |
| mod5_4 | gate | 100.00% | 100.00% | 100.00% | 100.00% | 99.79% | 1000 |
| mod5_4 | value | 100.00% | 100.00% | 100.00% | 98.50% | 97.39% | 1000 |
| tof_4 | probability | 100.00% | 100.00% | 100.00% | 100.00% | 99.83% | 1000 |
| tof_4 | gate | 100.00% | 100.00% | 100.00% | 100.00% | 99.83% | 1000 |
| tof_4 | value | 100.00% | 100.00% | 100.00% | 99.02% | 99.55% | 1000 |
| vbe_adder_3 | probability | 100.00% | 100.00% | 100.00% | 100.00% | 99.70% | 1000 |
| vbe_adder_3 | gate | 100.00% | 100.00% | 100.00% | 100.00% | 99.70% | 1000 |
| vbe_adder_3 | value | 100.00% | 100.00% | 100.00% | 100.00% | 99.70% | 1000 |

The small circuits are somewhat more sensitive to value ranking, but the base
matcher does not run out of legal actions: probability and gate ranking retain
approximately 99.8% Top-1000 conditional precision on all four circuits.

## Barenco depth 16

| ranking | Top-1000 conditional precision | final beam |
| --- | ---: | ---: |
| probability | 99.93% | 1000 |
| gate | 99.93% | 1000 |
| value | 68.16% | 0 |

The value rollout is healthy through depth 6, falls to 60.21% at depth 8,
and falls to 27.80% at depth 9. Depth 9 starts from the 1000 exact states
materialized by the depth-8 refresh, so this failure is not inherited lazy-state
drift. Removing the positive-gate-delta quota produces the same collapse.

## Score separation

Across the depth-16 value run's Top-1000 proposals:

| group | mean matcher probability | mean normalized value |
| --- | ---: | ---: |
| legal action | 0.9647 | 1.217 |
| current action invalid | 0.2593 | 1.723 |

The matcher already separates these groups, but value ranking reverses them.
Matcher probability is only a tie-break after
`next_gate_count - 0.25 * normalized_value`, so a high value can promote a
low-confidence matcher candidate.

The two dominant failure rules are xfer 4358 and 4361, inverse two-CNOT
commutation rules. They account for 1616 and 1478 current-action failures in the
audited Top-1000 sets. Their conditional precision is 75.26% and 34.37%.

## Value data diagnosis

- The action-value validation set has only 155 preference pairs.
- Overall preference accuracy is 69.68%; `barenco_tof_3` is 57.89% and `mod5_4`
  is 50.00%.
- The retained checkpoint is epoch 0. All 20 new epochs failed to improve total
  validation accuracy, and Barenco accuracy eventually fell to 50.00%.
- The training set has 1011 pairs, including 288 Barenco pairs. Xfers 4358 and
  4361 occur only 10 and 15 times respectively across preferred/rejected roles.
- Preference pairs contain valid actions only. They teach relative downstream
  quality, not action legality.

The evidence does not support "small circuits have too few legal leaves" as the
primary cause. The immediate cause is an undertrained long-horizon value head
overriding a well-separated matcher confidence signal without having an
illegality objective.

## Artifacts

- `topn_legality_summary.json` contains per-depth and per-xfer summaries with
  source/destination patterns.
- `barenco_tof_3_value_d16_scores.json` contains the detailed score-group audit.
- The remaining circuit/ranking JSON files contain the raw depth-8 and depth-16
  Quartz replay audits.
