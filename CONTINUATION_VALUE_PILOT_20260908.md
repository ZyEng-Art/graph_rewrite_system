# Continuation-Value Pilot — 2026-09-08

This pilot validates the fixed-budget continuation harness and provides an
initial answer to the width-versus-depth scheduling question. It is a pilot,
not a generalization claim: there are only 16 saved states from four selected
Quarl trajectories.

## Reproducible setup

- Circuits: `barenco_tof_3` and `gf2_6_mult`.
- Sources: one high-quality and one ordinary trajectory per circuit.
- States: four deterministic, endpoint-inclusive nonterminal samples per
  source (16 total).
- Search budgets: depth 4 and depth 16.
- Proposal-ranking seeds: 73 and 170.
- Budget mode: nested. Every depth-4 result is the exact prefix of the
  corresponding depth-16 run.
- Search: beam 256, microbatch 512, H-hop state-only model, r99.9 calibrated
  candidate filter, exact graph deduplication, and transactional exact apply.
- Physical work: 32 depth-16 jobs; 64 logical `(state, budget, seed)` results.
- Completion: 32/32 physical jobs and 64/64 logical results; zero failures.

The sampled high-quality states were:

| Source | Trajectory steps | Gate counts |
| --- | --- | --- |
| Barenco 58 -> 36 | 0, 38, 77, 115 | 58, 46, 41, 38 |
| GF 371 -> 370 | 0, 90, 180, 270 | 371, 372, 374, 371 |

The ordinary sources contributed Barenco steps 0/12/25/37 with gate counts
40/42/41/40, and GF steps 0/29/59/88 with gate counts 384/386/384/384.

## Outcomes

Values below are averages over sampled states and both seeds. `Success` means
that at least one seed found a lower exact gate count. `Unique` is accepted
nonduplicate successors divided by attempted exact applies.

| Circuit | Source kind | Depth | Success | Mean improvement | Max improvement | Unique | Duplicate | Invalid | Mean search seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Barenco | high quality | 4 | 100% | 1.75 | 2 | 13.3% | 66.3% | 20.4% | 0.654 |
| Barenco | high quality | 16 | 100% | 6.25 | 10 | 11.4% | 65.8% | 22.8% | 3.338 |
| Barenco | ordinary | 4 | 100% | 2.25 | 3 | 10.9% | 60.9% | 28.1% | 0.610 |
| Barenco | ordinary | 16 | 100% | 2.75 | 4 | 8.6% | 60.2% | 31.2% | 3.337 |
| GF | high quality | 4 | 50% | 1.00 | 3 | 46.7% | 52.7% | 0.6% | 2.219 |
| GF | high quality | 16 | 50% | 1.00 | 3 | 31.9% | 68.0% | 0.1% | 13.900 |
| GF | ordinary | 4 | 100% | 2.25 | 3 | 50.1% | 49.9% | 0.0% | 2.123 |
| GF | ordinary | 16 | 100% | 2.50 | 4 | 39.5% | 60.5% | 0.0% | 12.375 |

Fourteen of 16 states improved by depth 16. The same 14 states had already
improved by depth 4. In this sample there were no delayed-success states of the
form “no improvement by depth 4, improvement by depth 16.” The two states that
did not improve at depth 4 also did not improve at depth 16.

At a 25% continuation allocation (top four of 16 states):

| Depth-4 ranking signal | Precision at depth 16 | Recall | Lift over random | Selected mean depth-16 improvement |
| --- | ---: | ---: | ---: | ---: |
| Improvement | 100% | 28.6% | 1.143x | 3.50 |
| Attempted actions | 100% | 28.6% | 1.143x | 4.50 |
| Unique-accept rate | 75% | 21.4% | 0.857x | 1.50 |

The maximum possible success lift is small because the depth-16 success
prevalence is already 87.5%. The selected mean improvement is therefore more
informative than binary precision in this pilot. Short-probe improvement and
search activity are useful hypotheses; unique-accept rate alone is not a good
continuation signal here.

## Interpretation and next experiment

This result does not prove that deep scheduling is unnecessary. The state
sample is small and comes only from successful saved trajectories; it contains
no failed, loop-heavy, or deliberately plateaued on-policy states. It also
tests depth 16 rather than the 64+ actions needed to expose long detours.

The next statistically meaningful run should therefore:

1. sample at least 128 states per stratum from high-quality, ordinary, failed,
   loop-heavy, and randomized on-policy searches;
2. use nested probe/target depths 4/16/64 (and 128 where affordable);
3. reserve circuits or trajectory families for a held-out evaluation;
4. rank continuation by future gate improvement, not just binary success;
5. compare short-probe scheduling against random allocation and uniform beam
   expansion under the same exact-apply budget.

Only if those held-out data show delayed improvements or useful continuation
lift should a learned value head or DAG-PUCT scheduler be added. MCTS by itself
does not create a depth signal; it needs either measured short-probe evidence or
an out-of-sample value estimate.

## Logs and integrity

The full raw artifacts remain on the shared H100 filesystem at:

```text
/SharedData/dengzy/quarl_matchformer_fresh_20260902/experiment/
refresh_consistency_model_20260906/benchmark_results/
continuation_value_pilot_v1/run_nested
```

That directory contains 32 `runner.log` files, per-job raw and normalized JSON,
best QASM files, 96 append-only events, CSV aggregates, metadata, and the full
summary. Important SHA-256 digests are:

| Artifact | SHA-256 |
| --- | --- |
| Manifest | `e90f4f3ef295ac96d2690730688337bf351163a4d3a055fb023b0f5cc5932d30` |
| Metadata | `4b92e90c0bf563a379d4451e53b22cf0b1f936914d1eda52c03d69ab635efd4c` |
| Event log | `4b4460b64457cb947afd23528ba6e11dbd0bfbc9607dae895e2d540490ce8ab7` |
| Summary | `aae9b84f4ff3b58058f9e2c5ea8f387e737ffe9beb9e6dffdd070d08de1027f5` |
| State aggregates | `5a6335cbb597cde44c8d61f8694d6c3d28b9a31e38ad0d5884cee3af272a382f` |
| Group aggregates | `249116b318eabb05711eb4b5f552fbe814e0da5ccb0f10318e19f54803eb4dca` |
| Ranking evaluation | `17163ef4637ba2bdd28c8286f1e9aaa1a202be4f00ed7b91a3593fdb174fbcbf` |

The recorded harness, beam runner, and runner-configuration digests match the
files committed with this report. The execution mirror did not contain `.git`,
so its `git_revision` field is null; the content hashes provide the executable
provenance for this run.
