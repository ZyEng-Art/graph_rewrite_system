# Fixed-budget dual-lane depth search

## Purpose

The continuation corpus showed that the historical gate-count beam never
realized a delayed gain: every state that improved after a probe had already
improved before it. Increasing uniform depth therefore extended paths already
favored by the beam instead of protecting plateau or bounded-detour paths.

The dual-lane policy changes survivor allocation without assuming that a
deterministic "worth searching" rule exists. It is an experimental policy and
is disabled by default.

## Search state

Every exact child now carries three search-only fields:

- `path_best_gate_count`: smallest gate count on that child's own lineage;
- `stagnation_steps`: consecutive rewrites since that lineage last established
  a strictly smaller gate count;
- `survivor_lane`: `exploitation`, `exploration`, or `fallback`.

These fields do not alter graph matching, exact apply, or graph identity.

## Survivor policy

`--survivor-policy gate` is the default and preserves the historical ordering
by `(gate_count, history_length)`. Its candidate capacity remains exactly one
beam, irrespective of dual-lane tuning flags.

`--survivor-policy dual_lane` collects a bounded candidate pool and divides the
same final beam capacity into:

- an exploitation prefix selected by historical gate priority;
- an exploration reservation selected from candidates outside that prefix;
- a gate-priority fallback when too few exploration candidates exist.

An exploration candidate must:

- have at least one stagnating step;
- not exceed `--exploration-max-stagnation`;
- be at most `--exploration-max-detour` gates above its own path best.

Exploration candidates are round-robin interleaved across recent rewrite,
detour, and stagnation buckets. A stable seeded digest resolves ties. This is a
cheap diversity mechanism, not a claim that the bucket is a learned value or
an exact topology identity. Exact graph deduplication remains authoritative.

Recommended initial controlled setting:

```text
--survivor-policy dual_lane
--exploration-fraction 0.25
--exploration-max-stagnation 8
--exploration-max-detour 2
--survivor-candidate-factor 1.25
```

With beam 256 this reserves up to 64 output slots. The final beam is still 256;
the 1.25 candidate factor may perform more applies per layer, so depth alone is
not a fair comparison.

## Equal-apply comparison

`--max-total-attempted-actions` supplies a global exact-apply attempt budget.
The runner completes the partially filled layer when that budget is reached,
then stops. Both supplied A/B configurations use 60,000 attempted actions and
depth 128 only as a safety ceiling:

- `continuation_runner_hhop_r999_equal_apply_gate.json`
- `continuation_runner_hhop_r999_equal_apply_dual.json`

Run the paired corpus benchmark with:

```bash
./run_dual_lane_equal_apply_ab.sh MANIFEST OUTPUT_DIR 0,5,6,7 73,170
```

The wrapper runs gate and dual policies separately to avoid GPU contention and
writes `analysis.json` using `analyze_dual_lane_ab.py`. Every pair reports both
attempt counts; exhausted searches are therefore visible instead of silently
being treated as equal-budget runs.

## Survivor audit

When `--reference-data` is supplied, every layer now records:

- reference parent indices;
- whether the target graph was already seen;
- target indices in the exact child pool before survivor selection;
- retained beam indices and retained lane;
- an explicit `survivor_selection` exclusion stage when exact application made
  the teacher child but the fixed beam did not retain it.

Every normal step log also records input/output lane counts and complete
survivor-selection metrics. This separates matcher/proposal loss, invalid
apply, exact duplicate, and survivor loss.

`--locality-action-reserve N` additionally reserves up to `N` positions inside
each fixed per-parent action cap for bindings overlapping that parent's prior
rewrite neighborhood. It does not enlarge the cap. Proposal metrics report the
eligible and finally selected local-continuation counts. This option addresses
action-sequence continuity; the survivor lane alone cannot recover a path when
the necessary next action never passes the per-parent cap.

## Interpretation

A dual-lane win does not prove that its heuristic predicts long-horizon value.
It shows only that a fixed exploration reservation recovered a better result
under the same apply budget. A loss is equally informative: it quantifies the
cost of protecting non-improving paths. Learned value should be added only
after held-out A/B results show which retained paths later pay back their
budget.
