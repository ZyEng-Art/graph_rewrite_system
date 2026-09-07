# General Hierarchical Policy: Strict Zero-Shot Rollout

## Question

Can one hierarchical node-then-pattern actor, trained once on mixed Quarl
trajectories, improve circuits without circuit-specific PPO or fine-tuning?

## Protocol

- Machine: `h100-gpu5`, NVIDIA H100 80 GB HBM3, physical GPU 6.
- Actor: `hierarchical_action_allpaths_dedup_pathsplit_n16_m2_rtg025_e30_s996.pt`.
- The train/test split is by original trajectory path. Windows from one path
  cannot leak across the split.
- Strict Quartz refresh interval: 1. Every accepted speculative rewrite is
  replayed and checked against a complete Quartz graph before the next action.
- Node Top-K: 16; pattern Top-K: 16; exact rejection cache enabled.
- No per-circuit PPO update, search-state resume, best start, or replay start.

## Results

| Circuit | Episodes | Root gates | Exact best | Improved episodes | Transitions/s | Accepted/s | Horizon completed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `barenco_tof_3` | 64 | 58 | 58 | 0 | 713.5 | 575.0 | 62/64 |
| `barenco_tof_4` | 32 | 114 | 114 | 0 | 522.3 | 412.1 | 28/32 |
| `gf2^4_mult` | 32 | 225 | 221 | 10 | 411.5 | 339.1 | 30/32 |
| `gf2^6_mult` | 16 | 495 | 488 | 8 | 223.7 | 196.2 | 16/16 |

The shared actor therefore found exact zero-shot reductions of 4 gates on
`gf2^4_mult` and 7 gates on `gf2^6_mult`. The two Barenco circuits did not
improve in these short rollouts, but the policy did complete almost all
horizons instead of collapsing into invalid actions.

On the exact same 58-gate `barenco_tof_3`, the instrumented original Quarl
rollout produced 254.3 transitions/s. The strict hierarchical rollout
produced 713.5 transitions/s, or 2.81x the raw throughput. Even after counting
only exact accepted rewrites, 575.0/s is 2.26x Quarl's transition rate.

## Cost Shift

| Circuit | Match | Policy preparation | Strict refresh | Cache advance |
|---|---:|---:|---:|---:|
| `barenco_tof_3` | 18.7% | 30.1% | 16.5% | 8.7% |
| `barenco_tof_4` | 19.6% | 23.2% | 22.6% | 12.4% |
| `gf2^4_mult` | 20.5% | 18.6% | 32.2% | 7.7% |
| `gf2^6_mult` | 22.3% | 9.7% | 41.8% | 8.4% |

Strict refresh is now the largest stage on the larger tested circuits. It
grows with the exact graph size, not with the neural actor's logits. This
confirms that the next systems target is incremental exact graph maintenance;
replacing refresh with another neural prediction would remove the correctness
oracle and would not be an equivalent optimization.

## Artifacts

- `hierarchical_general_barenco3_b64_s16_s997.json`
- `hierarchical_general_barenco4_b32_s16_s998.json`
- `hierarchical_general_gf4_b32_s16_s999.json`
- `hierarchical_general_gf6_b16_s16_s1000.json`

