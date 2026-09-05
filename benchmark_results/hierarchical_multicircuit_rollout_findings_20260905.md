# Hierarchical Multi-Circuit Rollout

## Change

`benchmark_hierarchical_rollout.py` now accepts repeated `--additional-qasm`
arguments. One actor is shared across all circuits, while each circuit keeps
an independent exact best-so-far entry and replay reservoir.

The requested total episode count is divided as evenly as possible. The
remainder rotates between circuits on successive PPO iterations, so no circuit
is permanently favored when the batch size is not divisible by the number of
circuits. Collector timings and transitions are aggregated for the PPO update,
and the JSON log also records per-circuit rollout summaries.

## H100 Smoke Test

The strict-refresh smoke test used one shared pretrained actor over
`barenco_tof_3`, `gf2^4_mult`, and `gf2^6_mult`, with 2 episodes per circuit
and a 4-step horizon.

| Circuit | Episodes | Root gates | Exact best | Transitions |
|---|---:|---:|---:|---:|
| `barenco_tof_3` | 2 | 58 | 58 | 10 |
| `gf2^4_mult` | 2 | 225 | 225 | 8 |
| `gf2^6_mult` | 2 | 495 | 493 | 8 |

All three per-circuit summaries and archive entries were emitted in one run.
The short smoke test already found an exact 2-gate reduction on `gf2^6_mult`.

Artifact: `hierarchical_multicircuit_smoke_s1001.json`.

