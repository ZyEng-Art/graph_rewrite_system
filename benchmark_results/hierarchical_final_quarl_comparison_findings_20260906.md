# Final Hierarchical Rollout Comparison with Original Quarl

## Protocol

All measurements ran on one NVIDIA H100 80 GB GPU. The hierarchical collector
uses the shared actor checkpoint `hierarchical_multicircuit_ppo_i20_b64_s16_s1002.pt`,
16 episodes per circuit, a 16-step horizon, exact Quartz apply after every
action, exact rejection retry/cache, and a full topology audit every 8 steps.
It does not fine-tune on any evaluation circuit and does not resume a saved
best or replay pool.

Original Quarl uses its unmodified node-then-xfer sampling and exact graph
buffer semantics. Each profile has 64 episodes, horizon 20, and 1280
transitions. Consequently, throughput is directly comparable as work per
second, but best gate count is not an equal-search-budget comparison: Quarl
gets about five times as many transitions and can restart from new graph-buffer
states during the rollout.

The hierarchical action actor was supervised on Barenco/GF trajectories and
then shared-PPO-trained for 20 iterations on `barenco_tof_3`,
`barenco_tof_4`, `gf2^4_mult`, and `gf2^6_mult`. The VBE, HWB, Grover, and
GF16 evaluations receive no circuit-specific actor training. The frozen base
matching encoder was trained separately on a broader binding corpus.

## Results

| Circuit | Hierarchical transitions | Hierarchical input -> best | Hierarchical t/s | Quarl transitions | Quarl input -> best | Quarl t/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `barenco_tof_3` | 278 | 58 -> 58 | 496.7 | 1280 | 58 -> 58 | 254.3 | 1.95x |
| `gf2^4_mult` | 258 | 225 -> 219 | 475.4 | 1280 | 225 -> 219 | 194.5 | 2.44x |
| `gf2^6_mult` | 259 | 495 -> 485 | 356.5 | 1280 | 495 -> 485 | 154.5 | 2.31x |
| `vbe_adder_3` | 273 | 150 -> 150 | 419.1 | 1280 | 150 -> 146 | 206.9 | 2.03x |
| `hwb6` | 293 | 259 -> 256 | 391.4 | 1280 | 259 -> 253 | 185.8 | 2.11x |
| `grover_5` | 222 | 831 -> 825 | 157.1 | 1280 | 831 -> 791 | 119.0 | 1.32x |
| `gf2^16_mult` | 267 | 3435 -> 3435 | 71.3 | 1280 | 3435 -> 3435 | 32.1 | 2.22x |

The joint held-out run schedules three 16-episode circuit batches in one
process. A standalone Grover run with the same actor and strict settings gets
`831 -> 824` at 217.1 transitions/s, or 1.82x Quarl throughput. Both results
are retained: the joint number is the conservative multi-circuit measurement,
while the standalone number isolates Grover without neighboring circuit work.

Across the four PPO training circuits, the final clean run executes 1057
transitions in 2.274 seconds, or 464.8 transitions/s, with 1024 accepted exact
rewrites. Candidate fallback is never triggered on those four circuits. The
held-out VBE/HWB/Grover run executes 788 transitions in 2.813 seconds; adaptive
fallback is used for Grover and eliminates the previous zero-candidate
termination.

## What This Establishes

- The node-first paged collector is 1.32x to 2.44x faster per complete strict
  transition than original Quarl across 58 to 3435 gates in these measured
  configurations.
- A single shared actor immediately optimizes GF4, GF6, HWB6, and Grover
  without an eight-hour per-circuit fine-tune.
- On the two GF circuits used by shared PPO, the short hierarchical rollout
  reaches the same best gate counts as the 1280-transition Quarl profile with
  roughly one fifth of the transitions.
- This does not yet reproduce Quarl's quality on actor-unseen VBE, HWB, or
  Grover. Quarl reaches 146, 253, and 791 gates respectively, versus 150, 256,
  and 824/825 here. The remaining gap is search depth and persistent frontier
  reuse, not an inability to produce legal actions.

## Remaining Bottleneck

On the 3435-gate circuit, hierarchical matching is 19.1% of total rollout,
while strict refresh is 55.4%. Refresh contains 26.3% Quartz apply, 17.7%
graph hashing, and 5.8% periodic topology comparison. The exact graph remains
the correctness oracle, so the next useful system change is incremental exact
hash/archive maintenance and longer low-cost-frontier search, not replacing
refresh with an unverified neural graph prediction.

## Artifacts

- `hierarchical_multicircuit_adaptive_final_audit8_b64_s16_s1006.json`
- `hierarchical_unseen_adaptive_final_audit8_b48_s16_s1009.json`
- `hierarchical_unseen_gf16_adaptive_audit8_b16_s16_s1014.json`
- `hierarchical_grover_adaptive_hysteresis_b16_s16_s1012.json`
- `quarl_original_rollout_profile_gf2_4_mult_h100.jsonl`
- `quarl_original_rollout_profile_gf2_6_mult_h100.jsonl`
- `quarl_original_rollout_profile_{barenco,vbe,hwb6,grover,gf16}_*.json`
