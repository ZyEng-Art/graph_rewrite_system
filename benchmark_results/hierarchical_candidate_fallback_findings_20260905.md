# Adaptive Hierarchical Candidate Fallback

## Problem

The primary hierarchical matcher uses the actor's top 16 nodes and a
95%-recall pattern threshold. This is fast on the training circuits, but it
returned no candidates for all 16 root states of the held-out `grover_5`
circuit. A candidate sweep showed that this was candidate pruning rather than
an absence of legal Quartz rewrites:

| Node K, pattern K | Mean candidates/state | Match states/s |
|---|---:|---:|
| 16, 32 | 1 | 1278.4 |
| 32, 32 | 2 | 1274.1 |
| 64, 32 | 47 | 1291.1 |
| 128, 32 | 187 | 1234.0 |

All sweep rows used the 99%-recall threshold. `K=64` is therefore the smallest
tested setting that restores a useful action set without expanding to all
2386 matches per state.

## Implementation

The collector now starts each episode with the primary `K=16`, 95%-recall
candidate set. If exact structural decoding yields fewer than 16 candidates,
that state is regenerated with `K=64`, pattern `K=32`, and the 99%-recall
threshold. The episode remembers that decision and goes directly to the wide
candidate path on later steps. Normal episodes continue to expose only the
primary 16 node branches to the PPO policy.

This is a candidate-recall fallback, not a random-action fallback. Every
returned action still has a model score, an exact structural binding, and the
same immediate Quartz validation used by the strict rollout.

## H100 A/B

The A/B used `grover_5`, shared actor `s1002`, seed 1012, 16 episodes, horizon
16, exact refresh every step, and topology audit interval 8.

| Metric | Narrow then wide every step | Episode fallback memory | Fixed wide baseline |
|---|---:|---:|---:|
| Transitions | 230 | 229 | 229 |
| Accepted rewrites | 212 | 212 | 212 |
| Best gate count | 824 | 824 | 824 |
| Invalid actions | 2 | 2 | 2 |
| Hierarchical match time | 0.9171 s | 0.3240 s | 0.3326 s |
| Total rollout time | 1.6551 s | 1.0549 s | 1.0877 s |
| Transitions/s | 139.0 | 217.1 | 210.5 |
| Accepted rewrites/s | 128.1 | 201.0 | 194.9 |

The episode-memory result reproduces the fixed-wide path exactly: 229
transitions, 212 accepted rewrites, 2 invalid actions, 15 cycles, 13 improved
episodes, and best gate count 824. Relative to recomputing primary and fallback
matches every step, match time falls 64.7% and end-to-end throughput rises
56.2%.

Artifacts:

- `hierarchical_grover_candidate_sweep_r99_s1010.json`
- `hierarchical_grover_adaptive_hysteresis_b16_s16_s1012.json`
