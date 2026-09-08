# Fixed-Budget Continuation Value Benchmark

This benchmark asks a search-scheduling question that immediate action accuracy
cannot answer: if two current circuits receive the same additional search
budget, which one is more likely to improve?

The benchmark deliberately separates three concerns:

1. `build_continuation_manifest.py` selects immutable QASM states and records
   their SHA-256 digests plus teacher-trajectory future improvements.
2. `continuation_value_benchmark.py` runs the existing exact search executable
   for fixed depths and reproducible proposal-ranking seeds.
3. The summary compares short-probe signals with longer-budget outcomes and
   reports their precision, recall, and lift over random state selection.

No teacher action is supplied to the continuation search. Teacher-future rows
are offline labels only.

## Manifest construction

Trajectory arguments have the form `SOURCE_ID|CIRCUIT|KIND|PATH`. State
selection is deterministic, uniform over all nonterminal trajectory states,
and endpoint-inclusive.

```bash
python build_continuation_manifest.py \
  --trajectory 'barenco36|barenco_tof_3|high_quality|/path/to/barenco_trace' \
  --trajectory 'gf370|gf2_6_mult|high_quality|/path/to/gf_trace' \
  --trajectory 'barenco38|barenco_tof_3|ordinary|/path/to/ordinary_trace' \
  --states-per-trajectory 16 \
  --horizons 8,32,64,128 \
  --output benchmark_results/continuation_value/manifest.json
```

Each state records:

- source trajectory and step;
- initial gate count;
- QASM path and SHA-256 digest;
- the saved teacher action for auditing;
- best teacher-trajectory improvement and first-improvement time at each
  requested horizon.

The terminal sentinel is never selected because it has no continuation action.

## Runner configuration

`continuation_runner_hhop_r999.json` freezes the current H-hop checkpoint,
calibration, exact deduplication mode, transactional apply backend, candidate
limits, and environment. The harness owns these arguments and rejects attempts
to override QASM, depth, output, best-QASM path, or ranking seed through the
configuration file.

The state-only beam runner now accepts:

```text
--proposal-ranking gate|probability|stochastic
--proposal-ranking-seed SEED
```

The default remains `gate`, preserving previous benchmark behavior. The
`stochastic` mode assigns reproducible random priorities before both the
per-parent and global proposal caps. It is intended for repeated fixed-budget
outcome measurements, not as a claim that random ranking is the final search
policy.

## Running on H100

```bash
bash run_continuation_value_pilot.sh \
  benchmark_results/continuation_value/manifest.json \
  benchmark_results/continuation_value/run \
  5,6,7 \
  8,32,64 \
  73,170,267
```

The wrapper uses nested budgets by default. A depth-64 process is run once for
each `(state, seed)`, and the exact depth-8 and depth-32 prefixes are derived
from the same beam-search step log. This avoids loading the model three times
and ensures the shorter observation is literally a prefix of the longer
outcome. Use `--budget-mode independent` when independent restarts are the
experimental variable.

One lock is maintained per GPU, so different executor threads cannot
accidentally place two runner processes on the same device.

## Logs and provenance

The output directory contains:

```text
metadata.json
events.jsonl
summary.json
state_budget_aggregates.csv
group_budget_aggregates.csv
probe_ranking_evaluations.csv
jobs/<state>/budget-<depth>/seed-<seed>/
  job.json
  runner.log
  result.json
  normalized.json
  best.qasm
```

`metadata.json` includes the Git revision and dirty flag, Python version,
hostname, manifest/config/harness/runner SHA-256 digests, logical and physical
job counts, device assignments, budgets, seeds, and the complete runner
argument list.

`events.jsonl` is append-only and records starts, completions, failures, and
resume-time re-normalization. Raw runner output is never replaced by the
normalizer. A result generated with an earlier summary schema can therefore be
re-normalized without repeating the expensive model load.

## Metrics

Every normalized run reports:

- initial and best exact gate count;
- improvement and completed depth;
- model-search and full process wall time;
- predicted, attempted, accepted, invalid, and duplicate action counts;
- unique-accept, invalid, and duplicate rates;
- teacher-future labels for comparison only.

Results are aggregated across seeds for every `(state, budget)`, then across
`(circuit, kind, budget)` groups. For every longer target budget, the smallest
budget is treated as a probe. The current summary evaluates these directly
observable probe scores:

- short-budget improvement;
- unique accepted successor rate;
- attempted action count.

For the top 10%, 25%, and 50% of states under each score, the report includes:

- precision: fraction of selected states that improve under the longer budget;
- recall: fraction of all improving states captured by the selected subset;
- lift over random: precision divided by the global improvement prevalence;
- mean longer-budget improvement of the selected states.

This is a falsifiable gate before implementing DAG-PUCT. If short-probe or
learned state values do not provide out-of-sample lift over random selection,
MCTS would only organize a lottery rather than solve depth allocation.

## Interpretation constraints

- A small pilot validates plumbing and produces hypotheses; it is not enough
  to claim generalization.
- High-quality and ordinary saved Quarl trajectories are both selected search
  outcomes. Failed and loop-heavy on-policy states must be added before
  training a circuit-value head.
- Process wall time currently includes checkpoint loading for every physical
  `(state, seed)` job. Search time is reported separately. Nested budgets remove
  repeated loading across horizons; a future resident multi-root worker can
  also remove repeated loading across states.
- Immediate gate reduction is intentionally not the only target. Long plateau
  states require future-improvement and first-passage labels at horizons up to
  at least 64 or 128 actions.
