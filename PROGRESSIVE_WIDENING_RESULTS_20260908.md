# Progressive action-rank widening — 2026-09-08

## Result

Progressive widening recovered a 371-gate GF circuit without training a new
model. The fixed-budget gate baseline stayed at 372 for both seeds. With the
widening policy, seed 73 stayed at 372 and seed 170 reached 371. Barenco stayed
at 38 for both policies and both seeds.

This is evidence that the one-shot per-parent Top-128 decision was excluding
useful actions. It is not evidence that widening is already the best default:
the GF widening runs used about twice the search time of the gate baseline and
one of two configured seeds did not improve.

These runs start from the previously extracted difficult intermediate states,
not from the original full circuits. The GF result reproduces the target gate
count, not necessarily the same exact graph or action sequence as the recorded
Quarl trajectory.

## Mechanism

Every exact beam state now carries an `expansion_round`. Its first expansion
uses parent ranks 0--127. When that same exact graph receives another widening
slot, later expansions use disjoint 128-rank bands:

```text
round 0: ranks   0--127
round 1: ranks 128--255
round 2: ranks 256--383
...
round 7: ranks 896--1023
```

The final beam remains fixed. In the tested beam of 256, 64 slots are reserved
for exact parent-graph revisits and 192 for new children. Revisit selection is
round-robin across expansion rounds, preventing a continuous influx of new
round-zero states from starving the deeper bands.

The global proposal cap remains 4096. The test reserves 16 globally selected
proposals per live parent, so it samples each selected rank band without
claiming to exhaust all 128 actions in that band. Exact apply and exact graph
identity remain authoritative. The default policy is unchanged and widening
is opt-in.

## Fixed-apply comparison

All rows below received 100,000 attempted exact applies. The widening runs use
512 scheduling layers only as a safety ceiling; a parent revisit is a
scheduling event, not a graph rewrite, so scheduling depth and action depth
are logged separately.

The supplied runner configurations are
`continuation_runner_hhop_r999_widening_gate.json` and
`continuation_runner_hhop_r999_progressive_widening.json`. Use continuation
budget 512 for this comparison; the exact-apply limit remains 100,000.

| Circuit/state | Gate baseline | Progressive widening |
| --- | --- | --- |
| Barenco step 8, seed 73 | 38 | 38 |
| Barenco step 8, seed 170 | 38 | 38 |
| GF step 68, seed 73 | 372 | 372 |
| GF step 68, seed 170 | 372 | **371** |

The successful batch GF run first reached 371 at scheduling layer 287. Two
separate single-GPU replays with the same configured seed also reached 371,
but through different exact output circuits:

- replay 1: 27,351 attempted applies, 51.83 search seconds, and a 16-action
  best path;
- replay 2: a 29-action best path first seen at scheduling layer 166.

The two replay QASM SHA-256 values differ, as do their exact-identity digests.
This indicates that widening did not merely replay one memorized teacher
sequence. The 16-action path has a widening ancestor; its final reducing
action is parent rank 0, meaning an earlier widened action created a state in
which a normally top-ranked reduction became available.

The exact widened-action trace was added to result logging after these two
replays. Future results record `(action depth, parent rank, xfer id, anchor)`
for every widened action on the best path.

## Cost and limitations

- Widening materially reduces repeated-successor pressure for GF because it
  visits new rank bands, but it performs many more full-state model passes.
- Reserving revisit slots slows ordinary action-depth growth. A run must not
  report scheduling layers as rewrite depth.
- Rank bands require stable parent ordering. Widening currently accepts gate
  or probability ranking and rejects stochastic ranking.
- Widening is currently incompatible with the locality reserve and reference
  prefix audit. The latter assumes one trajectory action per scheduling layer.
- A band is subject to the global proposal cap and the early novel-child stop;
  it is not an exhaustive enumeration of every action in that band.
- One of the two GF seeds still failed, so this is an opportunity mechanism,
  not a deterministic solution.

## Artifacts

Shared result roots:

```text
benchmark_results/progressive_widening_long_plateau_v3_min16
benchmark_results/progressive_widening_replay_v1
benchmark_results/progressive_widening_replay_v2
```

SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Four-run widening summary | `6e541071dfbd7e3bcb43edcdf48548e099efa27835af79877032f38fc06df5a4` |
| Four-run widening analysis | `2a9dbb7b616178827616c1e3204fc86713c0ef7d0652e9e63ca6b8ad1c86a158` |
| Replay 1 result | `4cd9255eb41b51d1034e42bdf880daed588eb02ff5071185950ba24fddf12e66` |
| Replay 1 best QASM | `9ec8aa2c1ef538fe7dcc5d3762d3a2dee28564a17825ff2f4fda3a214aa563c2` |
| Replay 2 result | `e2bd518773ff66c3d54d6615fa6d661616f256e033080715f2cca3d33432dfd5` |
| Replay 2 best QASM | `82f24faa2e44a929f84de954e11c54984e5731b9ee77d555f47967db51401875` |

## Interpretation

The experiment changes the conclusion from "a learned deep-search value is
required" to a narrower statement: a model-independent rank-widening budget
can recover a useful path that the fixed Top-128 search misses. The next
optimization target is therefore scheduling and caching: retain the recovery
while avoiding repeated full-graph model evaluation and reducing the loss of
ordinary action depth.
