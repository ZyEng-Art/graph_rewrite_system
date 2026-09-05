# Broad KL-regularized match-set PPO

## Run

The shared actor/critic was trained once on 14 circuits, not fine-tuned per
held-out circuit. The run used 15 iterations, 224 episodes per iteration,
16-step episodes, 64 retained actions per state, state batches of 64, PPO
minibatches of 128, four PPO epochs, actor LR `1e-4`, critic LR `3e-4`, reference
KL coefficient `0.05`, target old-policy KL `0.015`, and seed 271.

It collected 46,945 transitions in 159.862 seconds (293.659 transitions/s) and
spent 35.583 seconds updating. The final old-policy approximate KL was 0.01271,
the final exact reference-policy KL was 0.12303, and final legality balanced
accuracy was 87.86%.

## Training best-so-far

| Circuit | Input | Best |
|---|---:|---:|
| `tof_3/4/5/10` | 45 / 75 / 105 / 255 | 45 / 75 / 105 / 255 |
| `barenco_tof_3/4/5/10` | 58 / 114 / 170 / 450 | 58 / 114 / 170 / 450 |
| `mod5_4` | 63 | 62 |
| `mod_mult_55` | 119 | 119 |
| `mod_red_21` | 278 | 276 |
| `vbe_adder_3` | 150 | 146 |
| `csla_mux_3` | 170 | 159 |
| `rc_adder_6` | 200 | 192 |

The last new best occurred at iteration 14 (`csla_mux_3`, 160 to 159), so the
archive continued to improve instead of becoming completely flat after the
first few iterations.

## Held-out attribution

All tests used beam 1000, depth 16, exact refresh every eight actions, and 64
independent final Quartz audits. The neutral actor has the same architecture
and frozen matcher/gate prior but zero learned residual, which separates policy
learning from search changes.

| Circuit | Input | Gate-first | Neutral | Trained PPO |
|---|---:|---:|---:|---:|
| `hwb6` | 259 | 255 | 253 | 253 |
| `gf2^4_mult` | 225 | 219 | 219 | 219 |
| `qcla_com_7` | 443 | 441 | 440 | 440 |
| `grover_5` | 831 | 817 | 813 | **811** |
| `gf2^5_mult` | 347 | 339 | 339 | 339 |
| `qcla_mod_7` | 884 | 884 | 884 | 884 |
| `adder_8` | 900 | 900* | 900 | 900 |
| **Total** | **3889** | **3855** | **3848** | **3846** |

`adder_8` gate-first exhausted its speculative beam at the depth-8 refresh;
the reported 900 is the retained root best, not a surviving final beam. Both
neutral and trained PPO completed depth 16 with 1000 states and 64/64 valid
audits, so that stability comes from the matcher/gate prior rather than learned
PPO residuals.

The trained residual has one attributable zero-shot win: two gates over the
neutral actor on `grover_5`. The other seven-gate aggregate gain over gate-first
comes from the common matcher/gate prior. This is positive evidence that the
shared policy can generalize, but one circuit and two gates are not enough to
claim a generally superior optimizer.

## Artifacts

The checkpoint is
`/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/paged_ppo_matchset_broad_kl_v2_s271.pt`
(SHA-256 `e23ab6a34fe31a83d798197768de64b36b4b7df9fab143de6a19d8f166396494`).
The full training record is the adjacent `.training.json` (SHA-256
`cf47ed00f90886951479df3bd81ca433528370ed27f8e1c4a28f01c244b18bac`).
The compact machine-readable results are in
`ppo_matchset_broad_kl_summary_20260905.json`.
