# Dual-lane depth-search results — 2026-09-08

## Bottom line

The implemented dual-lane survivor policy is mechanically correct, stays
within a fixed exact-apply budget, and retains long plateau paths. It did not
improve any paired result. The failure is no longer attributable to insufficient
search depth or a lack of exploration survivors: the necessary teacher
actions are present after matcher/source retrieval, but most are outside the
per-parent action cap and the current model provides no signal for composing
them into a coherent long sequence.

The policy therefore remains opt-in. It is useful instrumentation and a safe
search scaffold, but it is not presented as a production speedup.

## Implemented changes

- Fixed-size exploitation/exploration survivor lanes.
- Per-lineage best gate count and stagnation tracking.
- Bounded cumulative detour and configurable stagnation horizon.
- Stable diversity interleaving across rewrite/detour/stagnation buckets.
- A global attempted exact-apply budget for fair A/B comparisons.
- Reference-trajectory loss stages through exact child and survivor selection.
- Exploration ancestry and recovery logging for newly improved paths.
- Optional per-parent locality action reservation inside the existing cap.
- Correct propagation of `--proposal-ranking` and its seed through the
  `state_only_gpu` proposal path. That path previously hard-coded `gate` even
  when the run configuration requested `stochastic`.

All options are disabled under the default historical gate policy.

## 16-state equal-apply corpus

The corpus contains eight Barenco and eight GF states, with seeds 73 and 170.
Each policy received exactly 60,000 attempted exact applies per state/seed.

| Metric | Result |
| --- | ---: |
| Paired runs | 32 |
| Dual wins | 0 |
| Gate wins | 0 |
| Ties | 32 |
| Mean improvement delta, dual minus gate | 0 |
| Mean attempted-apply delta | 0 |
| Realized delayed gains, gate | 0 |
| Realized delayed gains, dual | 0 |
| Mean completed-depth delta | -3.375 layers |

The depth cost differed by circuit: -2.25 layers for Barenco and -4.5 layers
for GF. The first dual setting used a 1.25x pre-survivor child pool and an
eight-step stagnation limit.

These corpus jobs requested stochastic proposal ranking, but an audit found
that the `state_only_gpu` path still hard-coded gate ranking. Both sides of
every pair therefore used the same gate ranking, so the survivor comparison
remains paired, but these artifacts must not be interpreted as stochastic
search. The hard-coded path is fixed in the submitted implementation.

Across 970 dual-search layers, it selected 24,364 exploration states, averaging
25.12 per layer. It filled all 64 reserved slots in 348 layers and had too few
eligible candidates in 622 layers. The eight-step limit was too short for the
known GF teacher suffix whose first improvement occurs after 61 actions.

## Long-plateau controlled test

The follow-up used the known Barenco 42-step and GF 61-step teacher-delayed
roots. It increased the stagnation horizon to 64, reduced the candidate pool to
1.10x, and gave both policies exactly 100,000 attempted applies.

For GF, the gate baseline completed 67/70 layers across the two seeds, while
dual completed 64/65. Both therefore reached or exceeded the teacher's
61-action first-improvement depth. Neither found any reduction: all four
gate/dual GF results remained at 372.

The dual policy was active rather than starved:

- 197 total layers across the four Barenco/GF runs;
- 11,620 selected exploration states;
- 58.98 selected exploration states per layer on average;
- all 64 slots filled in 179/197 layers.

The Barenco best paths had exploration ancestors, demonstrating that the lane
could feed a later reduction, but their final result tied the baseline. GF had
no recovery from an exploration ancestor.

Adding 64 locality-reserved action positions inside the unchanged per-parent
cap also produced four ties at 100,000 attempted applies. After correcting an
overlap case so every parent fills all available positions up to the cap, the
mean depth cost was 2.25 layers. GF remained at 372 after 64/65 dual layers
versus 67/68 baseline layers. Mean measured search-time deltas were +0.17 s for
both Barenco and GF; locality selection was not a material runtime cost.

The corrected locality runs selected 11,824 exploration states over 197
layers, averaging 60.02 of the 64 reserved slots per layer and filling all 64
slots in 183/197 layers.

## Teacher-action audit

The complete `gf2^6_mult/370_2` trajectory contains 271 actions:

- exact action structurally available: 271/271;
- exact next transition validated: 271/271;
- source match retained at r99.9: 271/271;
- ranked action available by rank 2048: 235/271;
- median teacher gate rank: 486;
- p95 teacher gate rank: 1137;
- maximum teacher gate rank: 2019.

For the exact 64-action suffix starting at trajectory step 68:

| Per-parent cap | Teacher-action coverage |
| ---: | ---: |
| 128 | 12/64 = 18.75% |
| 256 | 18/64 = 28.13% |
| 512 | 26/64 = 40.63% |
| 1024 | 54/64 = 84.38% |
| 2048 | 57/64 = 89.06% |

All 64 source matches were retained. The teacher action's median rank was 525.
Seven steps had no rank even by 2048. The suffix repeatedly uses zero-cost
actions ranked around 500--700 and intersperses temporarily increasing actions
that are absent or ranked near 2000. The final reduction at step 128 has rank 1,
but it is reachable only after the preceding low-ranked sequence.

This explains why state-level exploration did not help: it retained many
states, but the correct next teacher action usually never entered that state's
128-action proposal set. Merely increasing the state beam or its survival time
cannot repair an action-composition failure.

## Consequence

The next model target should be a sequence-conditioned local action scorer,
not a binary state-level "continue searching" classifier. Training examples
must include the difficult teacher suffixes and random counterfactual actions,
with a pairwise objective that ranks the trajectory-consistent action above
other zero-cost rewrites in the same local neighborhood. Search should retain a
small model-independent exploration quota until held-out sequence coverage is
measured.

The immediate acceptance criterion is teacher-action coverage at the unchanged
per-parent cap of 128, reported on held-out trajectories. End-to-end deep search
should be rerun only after the 64-action GF window improves materially over its
current 12/64 coverage; otherwise a long search is still combinatorially
incapable of reproducing the suffix.

## Artifacts

Shared result roots:

```text
benchmark_results/dual_lane_equal_apply_corpus16_v1
benchmark_results/dual_lane_long_plateau_v1
benchmark_results/locality_dual_long_plateau_v2_fixed
```

Important SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| 16-state manifest | `370eed90be4189034593485087bc3d70f839fd8ba1919db8d2a817cec6694f75` |
| 16-state gate summary | `60ed8f72912878a0bf6fa3069a24369720751656ff2f3ced87e0eaca8c325393` |
| 16-state dual summary | `719babf47d6c935c4a5c761244ed3d98c6977f34a662a461d0f984250a6bb998` |
| 16-state paired analysis | `be63db2f15eb4bafe3235741ab2f75f07f0679017567634fa2a7a69964c47222` |
| Full GF candidate audit | `4e4b16b0c6bb1e30ce23e29aab0966f48d835a916eaeabee80d6f5d9f60f2458` |
| GF 64-action window | `9b0f18fa250153fcf33edd93a84399aea1bc1c2de72ff5086d357cb33a52d3ae` |
| Locality gate summary | `cd3fbdf79425ef3ddad64427da6042c96247335514a31e984d285c4549e9d47d` |
| Locality dual summary | `dbfbef86d710e3afb5140a2d694c5fc89525a7d9651eac5a16bbbde91467ed90` |
| Locality paired analysis | `ca927a34db0b4c8a086de5d90a8922dc3add9b3b0c77e3d81305ad7e4171fa9b` |
| Locality survivor usage | `5c1c12e39bdcdb5f6cd921e327c63ea24185366acf063899c86db2d1ebcb73c9` |

The result hashes describe the completed runs. Later source-only logging changes
do not rewrite raw job results.
