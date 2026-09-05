# Hierarchical Full-Action Pretraining

Date: 2026-09-05

## Change

The hierarchical actor previously behavior-cloned only the target anchor node.
Its pattern head was untrained and did not receive the causal action-prefix
state. This change:

- injects the existing prefix/state context into the pattern hidden state;
- keeps the matcher and graph encoder frozen;
- trains the pattern residual on exact teacher `(xfer_id, binding_slots)`
  actions against legal hard negatives;
- retains the matcher probability and immediate gate delta as base policy
  logits;
- upweights same-anchor pairs, where the conditional pattern decision is the
  only factor that can distinguish the two actions;
- supports bounded future-reduction weights when the preference corpus
  provides them.

The pattern trainer has 149,765 trainable parameters. The node policy remains
fixed at its return-weighted behavior-cloning checkpoint.

## Leakage-Controlled Training

The formal run uses
`quarl_barenco_teacher_action_preferences_supervisedzeroleak38_3_n32_20260905.pt`.
The target `barenco_tof_3/38_3` action states are excluded from training.

| Metric | Before | Best epoch 16 |
| --- | ---: | ---: |
| Test legal-pair accuracy | 51.33% | 91.47% |
| Same-anchor accuracy | 68.40% | 83.75% |
| Same-source accuracy | 70.37% | 92.59% |
| Uphill-teacher accuracy | 8.68% | 92.73% |
| Mean preferred margin | 0.146 | 5.510 |

There are 23,104 training pairs and 4,960 path-held-out test pairs. Frozen
feature encoding took 51.16 seconds for train and 9.90 seconds for test;
30 epochs over cached features took 16.04 seconds on H100.

The checkpoint is stored remotely at
`/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/hierarchical_action_barenco_zeroleak_n32_m2_e30_s981.pt`.

## Exact Rollout Audit

The policy was evaluated without PPO updates. `barenco58` is the original
58-gate circuit. `barenco39` is the held-out 39-gate initial state of `38_3`.

| Start | Refresh | Transitions | Transition/s | Accepted | Invalid | Cycles | Improved episodes | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 58 | 1 | 369 | 421.0 | 305 | 30 | 34 | 0/64 | 58 |
| 58 | 8 | 610 | 425.3 | 560 | 49 | 1 | 0/64 | 58 |
| 39 | 1 | 77 | 594.6 | 13 | 58 | 6 | 0/64 | 39 |
| 39 | 8 | 137 | 118.6 | 74 | 63 | 0 | 0/64 | 39 |

Unlike the node-only PPO checkpoint, the full-action policy does not collapse
to zero-delta rewrites. On the strict 58-gate run, accepted actions contain 26
`-1`, four `-2`, 183 `+1`, and 92 zero gate-delta rewrites. The policy has
therefore learned the intended preference for some temporarily uphill actions
and later local reductions.

It still does not produce a net improvement. The exact validator rejects many
actions on unseen states, and cycles terminate the remaining episodes before
long-horizon credit is available. Pairwise behavior cloning solves action
ranking on the supervised distribution; it does not by itself solve on-policy
state shift or exact Quartz legality.

## Strict PPO Follow-Up

A ten-iteration `refresh=1` PPO run started from the full-action checkpoint,
using 64 episodes, horizon 32, learning rate `1e-5`, invalid reward `-3`, and
cycle reward `-2`. Invalid actions fell from 30 in iteration 1 to 17 in
iteration 10 and 15 in the final audit. Exact cycles rose from 34 to 47 and
then 49 in the final audit. No episode improved below 58 gates.

This is not a useful PPO endpoint. Because each exact invalid/cycle terminates
the episode, the same deterministic bad action can consume another episode in
the next batch. The next collector change must cache rejected full actions per
exact graph and mask them before sampling. PPO should learn from the negative
transition once, while search immediately continues over unexplored actions.
