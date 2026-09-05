# Hierarchical PPO rollout integration (H100, 2026-09-05)

## Implementation

The batch collector now executes the complete factorized action path:

1. The pretrained node actor selects Top-K graph slots from the current paged
   readout.
2. The matcher scores only first-gate-compatible source patterns at those
   slots and exact structural decoding retains complete bindings.
3. Source matches expand to concrete Quartz transfers. Every expanded proposal
   maps back to its selected node branch through its stable anchor slot.
4. The policy samples from the exact truncated distribution
   `P(node) * P(action | node)`.
5. A rejected lazy rewrite is masked and resampled at the same state. Accepted
   rewrites alone update the graph state and paged causal cache.
6. Quartz replay validates pending suffixes at the refresh interval. Invalid
   tails are removed before GAE; global best-so-far changes only after exact
   replay.

Transitions store only K node features rather than every graph node. The new
collator and PPO update recompute the same factorized distribution and use the
existing clipped policy/value objective.

## H100 rollout

Protocol: `barenco_tof_3`, 64 parallel episodes, horizon 16, node K=16,
pattern Top-16, action cap 256, stochastic policy, H100 GPU 6. A short 8x2
warmup is excluded. The base matcher and node actor are the all-path models
from the preceding experiment.

| Refresh | Time | Transitions | Transitions/s | Exact accepted | Accepted/s | Invalid | Cycles | Full horizon |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.014s | 727 | 716.6 | 687 | 677.2 | 37 | 3 | 24/64 |
| 1 | 0.612s | 347 | 567.4 | 285 | 466.0 | 14 | 48 | 2/64 |

Every rewrite counted in `Exact accepted` survived Quartz replay. No
speculative best was reported; both runs remain at 58 gates.

The previous full-node batched collector with the same 64-episode, 16-step
shape measured 217.9 transitions/s and 196.8 accepted rewrites/s at refresh 1,
and 214.9 / 209.9 at refresh 8. The experiments use different actor seeds,
candidate caps, and a newer base checkpoint, so this is a system-throughput
comparison rather than a trajectory-identical microbenchmark. The measured
hierarchical gains are 2.60x transitions and 2.37x accepted rewrites at strict
refresh 1, and 3.33x / 3.23x at refresh 8.

Original Quarl's earlier `barenco_tof_3` profile was 254.3 transitions/s. The
hierarchical collector is 2.23x faster under exact-every-action refresh and
2.82x faster with refresh 8. Its refresh-8 exact accepted throughput is 2.66x
Quarl's raw transition throughput, though their policies and stopping behavior
are not trajectory-equivalent.

## Refresh-8 stage shares

| Stage | Time | Share |
|---|---:|---:|
| Candidate feature construction and CPU transition transfer | 0.322s | 31.8% |
| Hierarchical readout and matching | 0.176s | 17.4% |
| Paged cache advance | 0.120s | 11.9% |
| Policy sampling and selected proposal materialization | 0.093s | 9.1% |
| Quartz refresh | 0.091s | 9.0% |
| Indexed lazy apply | 0.084s | 8.3% |
| GPU proposal expansion/cap | 0.062s | 6.2% |

The bottleneck has moved away from all-node matcher logits. Candidate feature
construction plus transition transfer is now the largest measured stage. The
next memory/performance optimization is to retain rollout features on GPU in
an iteration arena or transfer only sampled-transition tensors, rather than
copying every active state's padded candidate set to CPU before selection.

## Correctness and policy gap

The factorization itself does not create invalid bindings: every candidate
passes the exact source structural decoder. Nevertheless, 14 of 347 selected
transfers fail Quartz under refresh 1. The failures are concentrated in a
small group of concrete xfers (1482 occurs three times and 810 twice); source
structural validity is not sufficient for every transfer sharing that source
representation. PPO therefore needs the concrete action feature and exact
invalid reward, not only a pattern matcher.

Refresh 1 also detects 48 exact cycles immediately. Refresh 8 amortizes Quartz
and permits longer suffixes, but 37/64 episodes eventually terminate on a
deferred invalid action. The PPO pilot should train the pattern/action residual
and critic from these labels, while the pretrained node head supplies the
initial proposal support. A learned legality auxiliary head remains a useful
follow-up if PPO reward alone suppresses invalid actions too slowly.

Raw logs:

- `hierarchical_rollout_barenco_b64_s16_k16_h100.json`
- `hierarchical_rollout_barenco_b64_s16_k16_refresh1_h100.json`
