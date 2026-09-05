# Original Quarl rollout audit (H100, 2026-09-05)

## Scope and implementation identity

These runs use Quarl's original exact-graph rollout and graph-buffer algorithm,
not the paged matcher or the new PPO implementation in this repository.  The
source is the Python 3.12-compatible snapshot at:

`/SharedData/dengzy/Quarl/experiment/ppo-new/original_snapshot_compat_20260828`

Its `actor.py` is byte-identical to the untouched copy in
`quarl_barenco_tof3_20260816_001809` (SHA-256
`bc0bc79503b49ed2a53f945d944dbe9c9d86bb45cc42a8a8a95e4b32d2e3d11e`).
The compatibility changes remove the unavailable `numpy.str0`, make the wandb
dataclass hashable, and permit the existing `agent_collect` path to run with a
single DDP process.  They do not change action selection, graph application,
the graph buffer, reward, or PPO loss.

All runs load the six-circuit Nam pretraining checkpoint `iter_576.pt`
(SHA-256 `db198874f5275351ed4215c764c4f839194bffc26f1008018b49610114185807`).
They ran on `h100-gpu5`; an unrelated idle vLLM worker occupied about 47.9 GiB
on each H100.  The main run was stopped after iteration 590, after seven
consecutive completed iterations at gate count 38, to avoid exhausting the
remaining GPU memory as Quarl's dynamic horizon continued to grow.

## Results

| run | completed iterations | transitions | rollout seconds | rollout throughput | learn seconds | input -> final best |
|---|---:|---:|---:|---:|---:|---:|
| Quarl fine-tune, batch 64, dynamic horizon, 5 PPO epochs | 14 | 81,344 | 261.244 | 311.37 transitions/s | 49.12 | 58 -> 38 |
| Same run with all learning rates set to zero | 14 | 59,584 | 189.537 | 314.37 transitions/s | 34.98 | 58 -> 36 |
| Fixed horizon 20, batch 768, 1 PPO epoch, seed 98766 | 3 | 46,080 | 159.457 | 288.98 transitions/s | 14.35 | 58 -> 41 |

The fixed-horizon run overlapped the batch-64 run on another GPU and therefore
is a capacity point, not a clean batch-size A/B.  It nevertheless confirms that
the exact Quarl environment sustains roughly 300 complete rewrite transitions
per second on this small circuit.

The per-iteration best-so-far curves (each value is logged after rollout and
before that iteration's PPO update) are:

```text
PPO:     58, 52, 52, 48, 46, 46, 45, 38, 38, 38, 38, 38, 38, 38
LR=0:   58, 56, 48, 46, 46, 38, 38, 38, 36, 36, 36, 36, 36, 36
fixed20: 51, 46, 41
```

The fixed-horizon curve starts at 51 because it found `58 -> 51` during its
first rollout; its input circuit was also 58 gates.

Quartz independently parsed the exported QASM files as:

| artifact | gates | CX | depth |
|---|---:|---:|---:|
| input `barenco_tof_3.qasm` | 58 | 24 | 45 |
| fine-tuned run best | 38 | 16 | 36 |
| zero-learning-rate run best | 36 | 14 | 34 |
| fixed-horizon run best | 41 | 19 | 39 |

## Why best-so-far falls

Quarl performs much more than an on-policy PPO trajectory from a fixed root:

1. Every action is applied by Quartz with
   `apply_xfer_with_local_state_tracking`, producing an exact next graph.
2. Legal non-NOP states no worse than their episode root are inserted into a
   persistent, hash-deduplicated graph buffer.  In greedy mode, insertion is
   further restricted to states within six gates of the current best.
3. Cost levels are sampled with weight `1 / (cost - min_cost + 0.2)`, strongly
   preferring the current low-cost frontier.  Within a cost level, later states
   receive slightly larger sampling weights.
4. NOP, invalid, over-budget, and horizon-ended episodes restart from that
   updated buffer while the same rollout batch is still running.  A newly found
   best can therefore seed more exploration immediately.
5. The next rollout horizon grows from the largest completed episode by 1.5x
   below 40 steps and 1.2x afterwards, capped at 600.

This mechanism explains both observations.  The best circuit can fall rapidly
because successful exact states become new roots, while rollout wall time grows
because Quarl deliberately collects more transitions.  In the fine-tune run,
the horizon-generated transition count rose from 1,280 to 13,440 per iteration;
rollout time rose from 5.00 to 43.48 seconds, but aggregate per-transition
throughput remained 311.37/s.

## PPO attribution

This one-seed audit does not show that online PPO caused the short-run quality
gain.  Freezing all parameters reached 36 gates with fewer transitions, whereas
updating PPO reached 38.  The trajectories diverge after the first update, so
this is not enough to claim that PPO is generally harmful, but it is enough to
reject the claim that the monotonic best curve itself demonstrates online
learning.  The dominant mechanism in this run is exact stochastic exploration
with persistent low-cost restarts.

Our previous paged PPO run collected 24,691 exact transitions over four
circuits at 77.82 transitions/s and left `barenco_tof_3` at 58 gates.  It did
not reproduce Quarl's search pressure: Quarl devoted 59,584-81,344 transitions
to this one circuit, retained thousands of exact intermediate roots, and grew
the horizon beyond 100.  Model architecture is therefore not the first gap to
fix; the accelerated system needs the same persistent frontier and restart
semantics while avoiding Quarl's full-graph memory and training costs.

## Artifacts

- `quarl_original_finetune_b64_s98765_20260905.{log,json,config.yaml,time.txt}`
- `quarl_original_nolearn_b64_s98765_20260905.{log,json,config.yaml,time.txt}`
- `quarl_original_fixed20_b768_s98766_20260905.{log,json,config.yaml,time.txt}`
- `quarl_original_finetune_b64_s98765_20260905_best_g38.qasm`
- `quarl_original_nolearn_b64_s98765_20260905_best_g36.qasm`
- `quarl_original_fixed20_b768_s98766_20260905_best_g41.qasm`

The JSON files are generated from the raw logs by
`parse_quarl_training_log.py`.
