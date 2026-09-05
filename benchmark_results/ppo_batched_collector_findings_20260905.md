# Batched speculative PPO collector

The PPO collector now reuses the existing paged causal model, indexed lazy
topology, stable action slots, exact checkpoints, and shared replay-prefix
cache. The legacy one-episode, exact-every-action implementation remains the
default behavior for `--collector-batch-size 1 --refresh-interval 1`.

## Implementation

- Initial circuits in one collector group are encoded in a single batch.
- Match retrieval, complete-binding decode, actor/value inference, and causal
  action advancement operate on all active episodes together.
- Ordinary actions update `IndexedTopology` and append to `PagedKVCache`; they
  do not invoke Quartz.
- Every `--refresh-interval` accepted actions, pending suffixes are replayed
  from their latest exact checkpoint. Shared action prefixes reuse exact replay
  results.
- Transitions before the first failing action are retained, the failing action
  receives the invalid reward and terminates, and the speculative suffix after
  it is discarded before GAE.
- A newly predicted global best forces an immediate exact refresh, so
  best-so-far is never updated from a speculative gate count.

## H100 result

Protocol: `barenco_tof_3`, 64 episodes, maximum 16 accepted actions, 64
candidates, no replay starts, seed 773, and one PPO epoch. The runs used GPU 6
on `h100-gpu5`; an idle vLLM allocation occupied approximately 47.9 GiB on the
GPU in every run.

| collector | collection | transitions/s | accepted/s | exact legality | best |
|---|---:|---:|---:|---:|---:|
| legacy exact, B=1/R=1 | 15.111 s | 77.36 | 67.76 | 98.12% | 58 |
| batched, B=64/R=1 | 2.510 s | 217.89 | 196.78 | 98.72% | 58 |
| batched speculative, B=64/R=8 | 4.226 s | 214.88 | 209.91 | 97.69% | 58 |

The B=64/R=8 collector is 2.78x faster by retained transition throughput and
3.10x faster by accepted rewrite throughput. R=1 and R=8 have nearly identical
transition throughput on this small circuit, showing that batching is the
source of this speedup. Deferred refresh is still needed for larger circuits
and beam-like branching, where graph materialization and shared exact prefixes
matter more.

The run is an implementation/performance validation, not evidence that the
untrained PPO head improves `barenco_tof_3`: all three runs remain at 58 gates.
The legacy collector retries exact cycles and invalid actions at the same
state, while the new collector terminates a deferred trajectory at such an
event. Therefore transition counts and sampled paths are not expected to be
identical even with the same random seed.

Machine-readable results are in
`ppo_batched_collector_summary_20260905.json`; the three `*.training.json`
files retain full arguments, episode metrics, update metrics, and best-so-far.
