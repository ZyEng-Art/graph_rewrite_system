# Shared Multi-Circuit PPO Results

## Training Run

- Machine: `h100-gpu5`, NVIDIA H100 80 GB HBM3, physical GPU 6.
- Initial actor: mixed-trajectory hierarchical action checkpoint `s996`.
- Circuits: `barenco_tof_3`, `barenco_tof_4`, `gf2^4_mult`, and
  `gf2^6_mult`.
- 20 PPO iterations, 64 episodes per iteration, 16-step horizon.
- One shared actor; episodes divided equally between the four circuits.
- Strict Quartz refresh interval 1; exact rejection retry/cache enabled.
- Per-circuit best and replay starts each sampled with probability 0.25.
- PPO learning rate: `5e-5`; 4 epochs; minibatch size 128.

The 20 rollout phases took 62.83 seconds in total. The observed end-to-end
command time, including model startup, warmup, PPO updates, checkpoint save,
and final evaluation, was approximately 96 seconds.

## Archive Progress

| Circuit | Initial archive | Final archive | First new best iteration |
|---|---:|---:|---:|
| `barenco_tof_3` | 58 | 58 | - |
| `barenco_tof_4` | 114 | 114 | - |
| `gf2^4_mult` | 221 | 219 | 6 |
| `gf2^6_mult` | 488 | 485 | 2; final value at iteration 6 |

Across training, rollout throughput rose from 301.6 to 388.0 transitions/s,
invalid actions fell from 124 to 52 per 64-episode batch, and mean reward rose
from -14.47 to -0.40. Entropy fell from 3.121 to 2.723, so further training
should retain an entropy or diversity constraint to prevent collapse onto a
small set of zero-delta rewrites.

## Clean Root A/B

Both actors were evaluated with seed 1003, 16 episodes per circuit, strict
refresh interval 1, and no best/replay starts. This separates actor learning
from archive reuse.

| Circuit | Best before | Best after | Improved episodes before | Improved episodes after | Invalid before | Invalid after |
|---|---:|---:|---:|---:|---:|---:|
| `barenco_tof_3` | 58 | 58 | 0/16 | 0/16 | 36 | 7 |
| `barenco_tof_4` | 114 | 114 | 0/16 | 0/16 | 48 | 2 |
| `gf2^4_mult` | 221 | 219 | 7/16 | 16/16 | 26 | 0 |
| `gf2^6_mult` | 491 | 485 | 8/16 | 16/16 | 20 | 0 |

The trained actor improved every GF episode from the original circuit. Mean
reward changed from -13.37 to +5.24 on GF4 and from -11.19 to +9.56 on GF6.
Accepted rewrite throughput also increased from 280.7 to 381.0/s on GF4 and
from 213.1 to 276.4/s on GF6 because the actor stopped spending attempts on
invalid actions.

This is model improvement, not only best-so-far retention: the clean run
loaded actor weights but did not resume the saved archive or replay pool.

## Artifacts

- Training log: `hierarchical_multicircuit_ppo_i20_b64_s16_s1002.json`.
- Pre-PPO clean evaluation: `hierarchical_multicircuit_preppo_clean_eval_b64_s16_s1003.json`.
- Post-PPO clean evaluation: `hierarchical_multicircuit_ppo_i20_clean_eval_b64_s16_s1003.json`.
- Remote checkpoint: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/hierarchical_multicircuit_ppo_i20_b64_s16_s1002.pt`.

