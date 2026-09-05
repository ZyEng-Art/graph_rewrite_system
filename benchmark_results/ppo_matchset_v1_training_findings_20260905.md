# Match-set PPO v1 training and zero-shot audit

## Training

One shared match-set actor/critic was trained for 15 PPO iterations on
`barenco_tof_3`, `mod5_4`, `tof_4`, and `vbe_adder_3`. Each iteration used 96
episodes, maximum depth 16, 64 candidates, batched collection, refresh interval
8, four PPO epochs, and replay-root probability 0.75.

The run collected 18,573 retained transitions in 54.36 seconds, or 341.64
transitions/s weighted across iterations. PPO updates took another 14.18
seconds. Original Quarl measured 311.37 transitions/s in the separate audit,
so collection is 1.10x faster numerically, but this is not identical semantics:
Quarl executes each transition on an exact graph while this collector verifies
speculative suffixes periodically.

| circuit | input | exact best | first improvement |
|---|---:|---:|---:|
| `barenco_tof_3` | 58 | 58 | none |
| `mod5_4` | 63 | 62 | iteration 2 |
| `tof_4` | 75 | 75 | none |
| `vbe_adder_3` | 150 | 146 | iteration 0 |

The best curve is flat after iteration 2. These improvements are exact Quartz
best-so-far results, but they do not establish that PPO caused them: replay and
stochastic collection can discover them before a useful policy is learned.

The selected-action legality head does learn a nontrivial classifier. Balanced
accuracy rises from 76.85% to a peak of 95.36% and ends at 91.84%; final legal
and invalid recall are 91.58% and 92.11%. The final per-iteration PPO
approximate KL is 0.0414 and policy entropy falls from 4.135 to 3.662, indicating
a substantial cumulative policy shift.

## Beam A/B and attribution

Beam-1000, depth-16 search compared gate-first ranking with the trained PPO
score on the four training circuits and held-out `hwb6` and `gf2^4_mult`.

| circuit | split | input | gate-first | trained PPO |
|---|---|---:|---:|---:|
| `barenco_tof_3` | train | 58 | 58 | 58 |
| `mod5_4` | train | 63 | 62 | 62 |
| `tof_4` | train | 75 | 75 | 75 |
| `vbe_adder_3` | train | 150 | 148 | 148 |
| `hwb6` | held out | 259 | 255 | 253 |
| `gf2^4_mult` | held out | 225 | 219 | 219 |

The apparent held-out gain on `hwb6` is not attributable to learning. A
control checkpoint with the actor residual and legality policy weight set to
zero also reaches 253, retains a full final beam, and passes 64/64 independent
Quartz replays with 64/64 topology matches. The trained actor reaches the same
exact 253-gate best at refresh depth 8, but all depth-16 suffixes fail the final
refresh and the final beam is empty. Independently parsing its exported exact
best QASM returns 253 gates.

Therefore v1 validates fast PPO training and learns legality, but it does not
yet demonstrate a learned zero-shot optimization gain. The next training run
needs broader circuit diversity and explicit reference-policy/target-KL
control so the learned residual cannot erase the strong frozen prior.
