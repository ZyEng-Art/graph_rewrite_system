# Sequence-conditioned match-set PPO actor/critic

## Implementation

- The actor preserves the ordered roles of every complete rewrite binding and
  applies self-attention across all retained matches for the current state.
- Every candidate is conditioned on the current graph summary and the final
  causal action hidden state already stored in `PagedKVCache`. That hidden state
  attends to the preceding action tokens, so reading it does not replay or copy
  the action sequence. At depth zero, graph pooling supplies the root state.
- The critic has separate parameters and attention-pools the current match set
  with a query formed from the prefix and graph state. It predicts one value for
  the state rather than one value per action.
- Final actor and critic layers are zero-initialized. Before PPO updates, action
  probabilities exactly match the calibrated matcher plus gate-delta prior and
  values are zero.
- A legality head learns from the actions subsequently checked by Quartz. Its
  BCE is class-balanced because 97.38% of sampled labels in the benchmark are
  legal. Logs include legal recall, invalid recall, and balanced accuracy in
  addition to raw accuracy.
- PPO collection batches all unresolved episodes in each retry round. Training
  and beam-search inference use the same padded match-set representation.

## H100 result

Protocol: `barenco_tof_3`, 64 episodes, at most 16 accepted actions, 64
candidates per state, `B=64`, `R=8`, no replay starts, seed 773, and one PPO
epoch. GPU 6 on `h100-gpu5` also held an idle approximately 47.9 GiB vLLM
allocation.

| actor | transitions/s | accepted/s | exact legality | best |
|---|---:|---:|---:|---:|
| legacy MLP reference | 214.88 | 209.91 | 97.69% | 58 |
| match-set repeat 1 | 274.09 | 266.90 | 97.38% | 58 |
| match-set repeat 2 | 242.25 | 235.89 | 97.38% | 58 |

The two match-set runs span a 1.13x to 1.28x throughput improvement over the
legacy batched reference, with pooled throughput of 250.44 transitions/s. The
larger network is faster here because actor calls from separate episodes are
combined into one GPU attention batch; the previous collector invoked the
policy separately for each parent retry.

The balanced legality metric is 49.18% after only one update epoch, with legal
recall 52.53% and invalid recall 45.83%. This correctly shows that the model has
not learned legality yet; raw 97% selected-action legality is a data/policy
property, not classifier accuracy.

An end-to-end beam-32, depth-2 rollout loaded the new `paged-ppo-v2` checkpoint
and used GPU PPO proposal ranking. Independent Quartz replay found 8/8 valid
trajectories and 8/8 exact topology matches. This run validates architecture,
throughput, checkpoint loading, and inference consistency. It is not a quality
claim: the one-iteration policy did not improve the 58-gate input.
