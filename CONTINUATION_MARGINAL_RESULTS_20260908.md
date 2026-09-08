# Marginal Continuation Corpus Results — 2026-09-08

## Bottom line

The corrected experiment does not support using short-term improvement as a
general depth signal. It predicts additional improvement made by the current
beam search, but the search produced no delayed-gain outcome at all: a state
that had not improved by the probe budget never began improving after more
depth. Consequently, a scheduler based only on recent improvement would speed
up the behavior the current search already prefers, while still failing to
recover Quarl-style plateau or detour paths.

## Corpus construction

The input contained 133 normalized Quarl hard-path trajectories from
`barenco_tof_3` and `gf2_6_mult`:

- 3,513 nonterminal trajectory states before deduplication;
- 2,048 states after byte-identical QASM deduplication;
- 1,465 exact QASM duplicates removed (41.7%);
- 260 right-censored negative candidates excluded from sampling because fewer than 64
  saved future steps could not establish a 64-step no-gain label;
- 80 selected states from 38 source trajectories;
- 29 Barenco and 51 GF states;
- 23 selected states had more than one saved teacher-behavior class for the
  same exact QASM content.

Sampling was stratified by circuit, teacher continuation class, and whether
the saved path showed an increase before its first reduction. It maximized
distinct source trajectories before taking a second state from one trajectory
and required an eight-step within-source separation. Teacher information was
used only to diversify the roots; it was not provided to model search.

The 23 ambiguous states are important: a state does not have one deterministic
"teacher continuation." Its observed future depends on which action sequence
was sampled. Teacher-path class is therefore coverage metadata, not a direct
state-value ground truth.

## Fixed-budget run

- Budgets: nested depths 4, 16, and 64.
- Seeds: 73 and 170.
- Physical jobs: 160 depth-64 searches.
- Logical results: 480 exact prefixes.
- GPUs: H100 devices 0, 5, 6, and 7.
- Search: beam 256, microbatch 512, H-hop state-only checkpoint, r99.9
  calibration, exact graph deduplication, and transactional exact apply.
- Completion: 160/160 physical and 480/480 logical jobs, zero failures.
- Run interval: 2026-09-08 10:05:44 to 11:40:16 UTC (94 minutes 32 seconds).
- Mean physical job wall time: 140.59 seconds.
- Sum of physical job wall time: 22,494.13 seconds.

Every short-budget result is the literal prefix of its depth-64 search, so a
marginal result cannot change because a shorter process was restarted with a
different random trajectory.

## Correct marginal outcomes

`Additional improvement` excludes all reduction already found at the probe.
For example, a state that improves by four gates at depth 4 and is unchanged at
depth 16 has zero `4 -> 16` continuation value.

| Transition | States with any added gain | Success rate | Mean added improvement | Maximum | Mean added gain per 1,000 attempted applies | Mean added search time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 -> 16 | 37/80 | 46.25% | 2.125 | 8 | 0.12796 | 8.48 s |
| 16 -> 64 | 18/80 | 22.50% | 0.950 | 10 | 0.00756 | 58.86 s |

Depth 16 to 64 was much less compute-efficient: it used about seven times the
added search time but its mean added improvement was less than half as large.
This average alone is not a stopping rule because a minority of roots still
obtained large reductions.

Across both seeds, the observed search outcome classes were:

| Transition | Continued gain | Saturated after probe | No gain | Delayed gain |
| ---| ---: | ---: | ---: | ---: | ---: |
| 4 -> 16 | 74 | 66 | 20 | **0** |
| 16 -> 64 | 33 | 107 | 20 | **0** |

These are 160 seed-level outcomes per transition. Ten `4 -> 16` outcomes and
32 `16 -> 64` outcomes showed a temporary increase in the best active frontier
gate count before the next global-best reduction, but all of them had already
improved before the probe. This is observed beam behavior, not proof that the
increase was required.

The two proposal-ranking seeds produced the same added improvement in 156 of
160 state/transition pairs. Only four pairs differed. The present beam is
therefore largely insensitive to this stochastic proposal ordering once its
gate-count survivor selection is applied.

## Teacher-path class versus model search

Combining detour and flat teacher subclasses gives:

| Circuit | Teacher class | Transition | States | Actual added-success rate | Actual mean added improvement |
| --- | --- | --- | ---: | ---: | ---: |
| Barenco | continued gain | 4 -> 16 | 16 | 81.25% | 4.562 |
| Barenco | delayed gain | 4 -> 16 | 13 | 84.62% | 4.154 |
| Barenco | continued gain | 16 -> 64 | 16 | 43.75% | 3.281 |
| Barenco | delayed gain | 16 -> 64 | 13 | 23.08% | 1.308 |
| GF | continued gain | 4 -> 16 | 16 | 50.00% | 1.875 |
| GF | delayed gain | 4 -> 16 | 16 | 25.00% | 0.750 |
| GF | no gain | 4 -> 16 | 8 | 0.00% | 0.000 |
| GF | saturated | 4 -> 16 | 11 | 9.09% | 0.091 |
| GF | continued gain | 16 -> 64 | 16 | 37.50% | 0.312 |
| GF | delayed gain | 16 -> 64 | 16 | 12.50% | 0.094 |
| GF | no gain | 16 -> 64 | 8 | 0.00% | 0.000 |
| GF | saturated | 16 -> 64 | 11 | 0.00% | 0.000 |

The Barenco model search usually found an alternative early reduction even for
roots selected from a delayed teacher path. GF is the harder case: roots from
teacher-delayed paths were substantially less likely to gain than roots from
teacher-continued paths. Crucially, those GF roots still did not become actual
delayed-gain outcomes—the current search either reduced early or never recovered
within depth 64.

## Probe-signal evaluation

At a 25% continuation allocation, using cumulative probe improvement only as a
ranking feature produced:

| Scope | Transition | Precision | Lift over random | Selected mean added improvement |
| --- | --- | ---: | ---: | ---: |
| All states | 4 -> 16 | 90.0% | 1.946x | 4.250 |
| All states | 16 -> 64 | 70.0% | 3.111x | 3.300 |
| Barenco | 4 -> 16 | 85.7% | 1.036x | 4.714 |
| Barenco | 16 -> 64 | 57.1% | 1.657x | 4.429 |
| GF | 4 -> 16 | 76.9% | 3.018x | 2.923 |
| GF | 16 -> 64 | 46.2% | 2.942x | 0.385 |

The circuit-stratified rows matter. Part of the aggregate signal comes from
mixing a small Barenco graph, where reductions are larger and cheaper, with a
large GF graph. Within GF, recent improvement remains predictive, but its
absolute `16 -> 64` gain is only 0.385 gates in the selected quarter.

Other proposed heuristics did not generalize:

- overall top-quartile unique-accept rate had 0.108x lift for `4 -> 16`;
- it was useful within Barenco but ineffective within GF, demonstrating severe
  circuit confounding;
- attempted-action count had 1.622x aggregate lift for `4 -> 16` but only
  0.667x for `16 -> 64`; within Barenco it selected no `16 -> 64` success.

Therefore neither unique-successor rate nor raw search activity should be used
as a circuit-independent continuation rule.

## Consequence for optimizer design

This experiment gives two separate conclusions:

1. Recent reduction can allocate compute efficiently among paths that the
   current search already knows how to improve.
2. It cannot rescue delayed paths, because the current beam did not realize a
   single delayed improvement in this corpus.

The second result prevents using recent improvement as a hard filter. Doing so
would make the optimizer faster at the same biased search while preserving the
failure mode that excludes plateau and detour sequences. MCTS supplied with
this signal alone would have the same problem.

The next optimizer change must target path retention or a learned long-horizon
value conditioned on topology. Its evaluation must explicitly count recovered
teacher-delayed roots on held-out trajectories and must reserve exploration for
states with no immediate reduction. It should then be compared under equal
exact-apply budgets, separately per circuit, rather than using raw depth or a
mixed-circuit average.

## Logs and integrity

Artifacts are stored at:

```text
/SharedData/dengzy/quarl_matchformer_fresh_20260902/experiment/
refresh_consistency_model_20260906/benchmark_results/
continuation_marginal_corpus_v4/run_nested
```

There are exactly 160 each of `job.json`, `runner.log`, `result.json`,
`normalized.json`, and `best.qasm`. `events.jsonl` preserves the original run
events and the later schema-v2 renormalization events. Raw runner logs and
results were not rerun or replaced during renormalization.

| Artifact | SHA-256 |
| --- | --- |
| Manifest | `712cfcfe876ff09c5fe58c05e156506ce633f335129d9f775cd4ead8c450c960` |
| Metadata | `4b24684b88ab47a80f0591cac11c8fae1c0e2aa0b3e0127c972b61dc5147a2e4` |
| Event log | `058c218960abc17b49e0abab8e34f0181f12f3758b49a1d12357fa40bec245ed` |
| Full summary | `275b1dece54820212cd28d64177d9dd460ed70550dccfeacae3c63d9eba165d1` |
| Compact analysis | `7867e064e523d9663bd18c7113835e3823bc65ed30af115af30f6da3244e8329` |
| Original harness log | `24648c5a448e029469324039e18179b5767f88f78dd4a2c3be8ffd56bebadb86` |
| Renormalization log | `4112d648d0ed72bbd658decc4c08108dbc418d80e7ef2c0bbf7ae4d5d1933ef7` |

The executable provenance recorded in metadata is:

- harness: `c9c180e131ea7a20de65b5bcaf7095b28be1894f26ae47ccd07db8e20ab64eed`;
- beam runner: `191c9ea8c56724f2ddccb24cd838e69224b107b03a3de2712c0ab9cd3d17a625`;
- runner config: `336d9d857deb27205729f42d4c77d9ba0fb01cc0132424748eda5190681c077b`.

The content hashes above identify both the raw search executable and the final
normalization code even though the shared execution mirror does not contain a
`.git` directory.
