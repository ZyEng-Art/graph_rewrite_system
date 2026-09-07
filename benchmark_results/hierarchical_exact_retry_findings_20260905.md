# Exact Rejection Cache and Rollback

Date: 2026-09-05

## Problem

Strict `refresh=1` PPO correctly labels Quartz-invalid and exact-cycle actions,
but the collector previously terminated the whole episode at the first such
action. A deterministic bad action could then be sampled again from the same
root in every later PPO batch. In a ten-iteration pilot, invalid actions fell
from 30 to 15 while exact cycles rose from 34 to 49; no trajectory reached the
32-step horizon and no gate improvement was found.

## Rejection Cache

The collector now keys rejected actions by:

```text
(exact Quartz graph hash, slot-sensitive topology fingerprint)
    -> (xfer_id, complete binding slots)
```

The topology fingerprint prevents a binding expressed in one persistent-slot
layout from being reused against an incompatible layout of the same exact
graph. The full binding prevents one invalid match from suppressing other
matches of the same transfer.

Before policy sampling, known rejected actions are removed from the candidate
mask. The negative transition remains in the PPO batch, so the model still
learns the general legality/cycle signal.

With a frozen policy (`learning_rate=0`) over five repeated batches, cache-on
ended with 432 transitions and 370 accepted rewrites, versus 323 and 259 with
cache-off. This is +33.7% transitions and +42.9% accepted rewrites. The final
cache-on batch masked 237 already-audited actions; masking took 22.3 ms of a
590.3 ms rollout (3.8%).

Caching alone did not solve episode truncation: each newly encountered bad
action still ended its current trajectory.

## Exact Rollback

For strict refresh states, the collector now retains the exact parent state
and paged prefix handle until validation finishes. On exact failure or cycle:

1. the action receives its negative reward and is inserted into the cache;
2. the speculative topology and state are rolled back to the exact parent;
3. the failed transition is marked nonterminal;
4. the policy runs again at the same parent with the failed full action masked;
5. a bounded per-episode retry count prevents unbounded rejection loops.

Paged handles for mixed batches are reference-counted: accepted rows advance,
retry rows retain their old handle, and stopped rows release theirs. This
avoids copying the prefix cache during rollback.

## Fixed-Policy A/B

Both runs use the same action-BC checkpoint, seed 990, 64 episodes, horizon 16,
and strict `refresh=1`.

| Mode | Transitions | Accepted | Transition/s | Invalid | Cycles | Full horizons | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Terminate on rejection | 340 | 279 | 542.2 | 23 | 38 | 3/64 | 58 |
| Cache + rollback, max 8 retries | 1292 | 961 | 561.7 | 174 | 157 | 52/64 | 58 |

Rollback produces 3.44x as many strictly accepted rewrites and turns 49
prematurely terminated episodes into full trajectories. Raw transition
throughput is also 3.6% higher despite performing 319 exact rollback retries.
The larger invalid/cycle totals are expected: they are retained training
examples that no longer terminate the trajectory.

## Strict PPO Result

The formal run starts from leakage-controlled full-action behavior cloning and
uses 30 PPO iterations, 64 episodes, horizon 32, strict refresh, learning rate
`1e-5`, and equal `-3` rewards for invalid and cycle actions.

| Iteration | Accepted | Invalid | Cycles | Full horizons | Cache hits | Global best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1567 | 267 | 250 | 21 | 691 | 58 |
| 10 | 1642 | 273 | 258 | 20 | 1208 | 58 |
| 15 | 1836 | 232 | 236 | 31 | 1121 | 58 |
| 20 | 1827 | 248 | 200 | 37 | 1184 | 58 |
| 25 | 1842 | 240 | 179 | 37 | 1151 | 57 |
| 30 | 1872 | 240 | 162 | 41 | 1116 | 57 |

The first exact 57-gate graph was found in iteration 21 at accepted depth 15.
It is stored in `best_so_far` with its exact QASM. This is the first run in the
new hierarchical pipeline that improves the original 58-gate Barenco circuit
without forcing a saved trajectory.

The final independent rollout processed 2329 transitions at 936.8
transitions/s, accepted 1961 rewrites, and ran 50/64 trajectories to depth 32.
Its accepted actions included 261 `-1`, 42 `-2`, 839 `+1`, and 819 zero-delta
rewrites. It did not rediscover 57 in that one batch; the verified global
archive retained the iteration-21 result.

The remaining limitation is initialization. Every PPO batch still starts from
the original 58-gate graph, so the newly found 57-gate frontier is saved but
not used to continue optimization. The next collector change should mix root,
verified-best, and diverse verified replay starts across circuits.
