# PPO reference-policy and target-KL control

The PPO update now computes the exact categorical KL from the current actor to
the frozen calibrated matcher plus immediate gate-delta prior over every
candidate set. `--reference-kl-coefficient` adds this reference loss, while
`--target-kl` stops additional PPO epochs when the completed epoch's mean KL
from the rollout policy exceeds 1.5 times the target. Both default to zero in
the Python CLI for checkpoint compatibility; the training launcher selects
0.05 reference coefficient and 0.015 target KL.

An H100 smoke run deliberately set `--target-kl 1e-9` to exercise the stop.
It collected 936 transitions at 246.73 transitions/s, completed one of four
requested PPO epochs, set `target_kl_early_stopped=true`, and reported
old-policy KL `2.799e-5` and reference-policy KL `2.922e-5`. The complete
configuration and metrics are in `ppo_matchset_kl_smoke_s991.training.json`.
