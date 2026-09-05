# Hierarchical PPO policy-head benchmark

The new policy factors one action into stop, node, and node-conditional
pattern/xfer decisions.  It returns exact joint log probabilities so PPO can
store and recompute one action probability without treating the hierarchy as
independent losses.  Nodes without retained candidates receive no probability
mass, and stop is the deterministic fallback for an empty candidate row.

On one H100, a fixed synthetic rollout shape of 224 states, 450 live slots, 64
candidates, and width 192 gives:

| policy head | parameters | time/batch | states/s | peak allocated |
|---|---:|---:|---:|---:|
| hierarchical | 522,828 | 1.963 ms | 114,138 | 0.304 GiB |
| match-set reference | 2,893,836 | 1.276 ms | 175,606 | 0.237 GiB |

The hierarchical head is 1.54x slower in isolation because it additionally
scores all 450 nodes.  Its absolute cost is only 8.76 microseconds per state,
so the intended end-to-end gain must come from avoiding the full
node-by-source match matrix and proposal set, not from making the final head
itself faster.  Maximum probability-normalization error was `4.77e-7`.

The benchmark excludes graph encoding, candidate retrieval, rewrite, refresh,
and PPO update.  Full counters and environment details are in
`hierarchical_policy_head_b224_n450_c64_h100.json`.
