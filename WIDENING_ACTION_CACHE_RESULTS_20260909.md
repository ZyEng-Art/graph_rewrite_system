# Ranked widening-action cache

Date: 2026-09-09

Branch: `deterministic-feedback-widening-20260909`

Implementation commits: `9d18a41`, `dfa8b1c`

## Outcome

The ranked-action cache is exact-search-equivalent but slower on the formal
Barenco run. It remains experimental, defaults to `off`, and should not be used
for production benchmarks. No GF long run was spent after the bracketed Barenco
result established that the approach is structurally unfavorable.

The cache stores the stable per-parent Top-1024 action order and slices later
128-action widening bands from it. This removes repeated matcher work, but a miss
must rank 1024 actions for every expanded parent before the scheduler knows which
64 parents it will retain. In the formal run, roughly 75% of miss-parent pools
were discarded. The extra proposal materialization and sorting cost exceeded the
saved matcher work.

## Correctness and tests

- The option is `--widening-action-cache on`; the default is `off`.
- It is restricted to deterministic feedback widening with gate ranking and is
  mutually exclusive with raw candidate caching and neural feature modes.
- A redundant global sort was removed before the final measurement.
- Cache-on/off runs produced identical non-timing search structure, final gate
  count, depth, and exact identities.
- The final remote suite included ranked-band equality and bounded-retention
  tests; the later combined suite ran 16 tests successfully on the H100 host.

## Formal Barenco result

Input: `data/fullseq_36_0_forward/0_58_0_8_121.qasm`

All variants used a 100,000 exact-apply budget, beam 256, deterministic feedback
widening, 128 actions per band, and at most eight bands. Timing order was off,
on, off to bracket shared-host drift.

| Metric (seconds) | Off before | Cache on | Off after | On vs off mean |
|---|---:|---:|---:|---:|
| Model match | 3.783 | 3.224 | 3.494 | 11.39% less |
| Proposal | 1.750 | 2.404 | 1.661 | 40.95% more |
| Search excluding apply | 6.900 | 7.297 | 6.401 | 9.71% more |
| Search total | 25.275 | 24.053 | 22.799 | 0.07% more |
| Process wall | 105.437 | 99.259 | 95.466 | 1.19% less (host noise) |

The cache recorded 5,923 hits and 17,920 misses. It generated 5,699,109 ranked
action rows, reused 1,892,007, and peaked at 25,005 rows / 64 parents /
2,800,560 bytes.

Machine-readable results are in
`benchmark_results/widening_action_cache_20260909.json`. Full run JSON and logs
are retained under:

```text
/SharedData/dengzy/quarl_matchformer_fresh_20260902/experiment/
  widening_action_cache_results_20260909/barenco_raw58_apply100k_v2/
```

## Decision

Keep the earlier raw candidate cache as the only opt-in cache worth retaining;
it reduced the matcher-heavy GF non-apply path by 8.74%. Do not pursue broader
ranked-pool caching unless proposal generation is first changed to rank only the
parents selected by the scheduler.
