# Hierarchical node/pattern action audit

## Protocol

The audit uses the frozen `paged_action_onpolicy_v13_r8_lr5e5_epoch1.pt`
matcher on all 2,048 held-out prefixes in
`binding_longmix_16384_2048_v2.pt`.  The states contain 2,090,005 exact
complete bindings and 747,635,928 gate-type-compatible node/source pairs.

Two teacher-only node scores are measured: the maximum source-pattern logit at
each node and the log-sum-exp of all compatible source-pattern logits at that
node.  Both scores require the existing full match matrix.  They establish the
factorization ceiling and generate distillation targets; they are not the
future cheap node head and do not demonstrate hierarchical runtime yet.

For each retained node, patterns are ranked by the frozen match logit and then
decoded with the existing exact lightweight structural decoder.  Exact match
recall counts all held-out positive bindings, while target-action recall refers
to the single recorded trajectory action at each prefix.

## Results

The maximum-logit node teacher gives:

| node K | compatible pairs retained | exact positive anchors | target-action anchors | best gate-delta anchors | valid candidates/state at pattern 16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.098% | 0.256% | 0.146% | 0.049% | 2.93 |
| 4 | 0.395% | 0.956% | 0.342% | 0.635% | 11.53 |
| 8 | 0.765% | 1.896% | 0.488% | 1.172% | 23.44 |
| 16 | 1.455% | 3.807% | 1.123% | 2.637% | 48.48 |

Log-sum-exp produces nearly the same result.  At node K=8 it retains 0.782%
of compatible pairs, 1.891% of exact positive anchors, 0.586% of target-action
anchors, and 1.172% of best gate-delta anchors.

Every state has at least one exact match under node K=1 for both teachers.  The
low all-positive recall is expected because each state has about 1,021 exact
bindings spread across many nodes; a fixed K-node policy is intended to choose
one useful action, not reproduce the complete legal match set.

Pattern selection is not the bottleneck once a node is retained.  With the
maximum-logit teacher, pattern K=16 retains 99.18% of the exact matches whose
anchors are in node K=8.  All recorded target actions whose anchors are in the
top eight also have their source pattern in pattern K=16.  Increasing pattern K
from 16 to 64 raises unconditional exact recall only from 1.881% to 1.896%.

The full matcher took 22.58 seconds, or 90.70 states/s, on one H100 and peaked
at 0.765 GiB allocated CUDA memory.  Structural decoding for both node teachers
and the complete K/M grid took 1.21 seconds.

## Decision

The current matcher cannot be used directly as a node actor.  It was trained
to recover legal bindings, so its aggregated node score has no reason to rank
the trajectory action or the best gate-reducing action.  This is confirmed by
only 1.12% target-action node recall at K=16 for max aggregation.

The next implementation should train a separate sequence-conditioned node head
from return-aware action labels or from a joint PPO-policy teacher aggregated
by anchor.  Pattern retrieval can initially use K=16 because its conditional
coverage is already above 99%.  A node-first rollout speed claim must wait for
that cheap head; these results support the factorization but deliberately do
not time a deployable hierarchical policy.

Machine-readable counters, hashes, timing, and every K/M combination are in
`hierarchical_action_audit_test2048.json`.
