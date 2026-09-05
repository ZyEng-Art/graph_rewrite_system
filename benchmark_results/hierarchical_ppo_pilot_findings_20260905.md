# Hierarchical PPO Pilot on H100

Date: 2026-09-05

## Configuration

- Circuit: `barenco_tof_3.qasm`, 58 initial gates.
- Actor initialization: return-weighted node behavior cloning checkpoint.
- Frozen graph/matcher model: all-path holdout checkpoint.
- Hierarchy: node Top-16, pattern Top-16, at most 256 expanded actions.
- Rollout: 64 episodes x 16 steps, exact refresh every 8 accepted rewrites.
- PPO: 30 iterations, 2 epochs per iteration, minibatch 128, learning rate
  `5e-5`, clip epsilon `0.2`, target KL `0.02`.
- H100 peak allocated memory: 0.460 GiB. The resident vLLM allocation is not
  included in this PyTorch process metric.

The full training record is
`hierarchical_ppo_barenco_i30_b64_s16_k16_s960.training.json`. The actor
checkpoint is stored remotely at
`/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/hierarchical_ppo_barenco_i30_b64_s16_k16_s960.pt`.

## Training Curve

| Iteration | Transitions | Transition/s | Accepted | Invalid | Delayed cycles | Full horizons | Entropy | PPO KL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 638 | 490.6 | 593 | 44 | 1 | 19 | 3.796 | 0.01024 |
| 7 | 899 | 1597.5 | 872 | 14 | 13 | 37 | 2.411 | 0.00364 |
| 13 | 973 | 1629.8 | 962 | 7 | 4 | 53 | 2.417 | 0.00524 |
| 25 | 1024 | 1810.6 | 1024 | 0 | 0 | 64 | 2.174 | 0.00136 |
| 30 | 1024 | 1823.7 | 1024 | 0 | 0 | 64 | 2.245 | 0.00317 |

PPO clearly learned the validity signal visible under delayed refresh. It
removed the observed invalid actions and increased the fraction of episodes
reaching the nominal horizon. This also raised measured rollout throughput,
because fewer episodes terminated early and the GPU batches stayed full.

It did not optimize the circuit. `best_so_far` remained 58 in every iteration.

## Independent Exact-Refresh Audit

The final policy was evaluated in fresh processes with different random seeds.
The benchmark now logs rewards, gate deltas, selected transfers, cycle
transfers, and actor checkpoint format.

| Refresh | Transitions | Transition/s | Accepted/s | Invalid | Exact cycles | Full horizons | Improved episodes | Best gates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 308 | 298.5 | 236.5 | 0 | 64 | 0 | 0 | 58 |
| 8 | 1024 | 1118.0 | 1118.0 | 0 | 0 | 64 | 0 | 58 |

The strict `refresh=1` result changes the interpretation of the training
curve. Every episode terminated in an exact cycle. All 244 accepted rewrites
had gate delta zero, and the cycle-ending transfers were `38`, `39`, `44`, and
`45`. Under `refresh=8`, all 1024 rewrites also had gate delta zero. The delayed
audit only compares the graph at the end of an eight-action suffix with prior
checkpoints, so it can hide shorter exact cycles inside that suffix.

The apparent final throughput of the training run is therefore not evidence
of a useful optimizer. It is the throughput of a low-entropy, legal,
zero-gate-delta policy whose short cycles are not visible at the selected
refresh interval.

## Diagnosis

1. Node-only behavior cloning gives the policy plausible rewrite locations,
   but the pattern/action residual starts untrained. PPO initially receives a
   much denser legality signal than an optimization signal and learns legality
   first.
2. The original 58-gate graph is absent from the collected Quarl trajectory
   ancestry. The known reproducible `38_3` path starts at 39 gates and reaches
   38 after 16 actions. This pilot has no teacher bridge from 58 to that basin.
3. A 16-step rollout from 58 gates produced no reducing action. With no
   improvement and nearly all accepted actions having zero gate delta, the
   critic has no useful long-horizon optimization target.
4. Refresh interval 8 is valid as an execution optimization only after cycle
   detection remains semantically equivalent to strict replay. It is not a
   reliable training validator in the current form.

## Required Next Changes

1. Pretrain the complete hierarchical action policy, not only its node head,
   using exact `(xfer_id, binding_slots)` Quarl actions against legal hard
   negatives. Weight the teacher action by future best reduction so temporary
   increases can receive a positive long-horizon target.
2. Run PPO over multiple circuits and start a controlled fraction of episodes
   from exact replay/archive states. Keep strict refresh during the initial
   legality/cycle phase, then relax refresh only after a strict audit agrees.
3. Detect exact hashes for every action in a replayed suffix, or train a
   separate semantic-cycle/legality head from strict refresh labels. A final
   suffix hash alone cannot rule out internal cycles.
4. Preserve exact best/frontier states across PPO batches. The archive should
   retain both low-gate states and diverse uphill states that later lead to a
   lower return, rather than restart all episodes from the same root.

This pilot validates the high-throughput hierarchical PPO data path, but it
also rejects the current policy as an optimizer checkpoint.
