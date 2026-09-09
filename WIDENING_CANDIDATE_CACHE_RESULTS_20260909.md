# Deterministic widening candidate cache

Date: 2026-09-09

Branch: `deterministic-feedback-widening-20260909`

Experiment commit: `a7aa576832c90938982dd34c1dfefe262a5ebe94`

## Outcome

The exact-parent candidate cache is search-equivalent and bounded. It removes
repeated state-only matcher and structural-decode work when progressive widening
revisits the same Quartz graph at a deeper action-rank band. It remains opt-in
(`--widening-candidate-cache on`) and defaults to `off`.

The cache improves the candidate-generation portion of deep widening, especially
on the 372-gate GF input. It does **not** decide which branch deserves deep search:
all cache-on and cache-off runs intentionally follow the same branch schedule.
Branch judgment therefore remains the next modeling problem.

## Implementation and safety properties

- Entries are keyed by the concrete Quartz graph object used by a widening
  revisit. The entry retains a strong graph reference, preventing Python object-id
  reuse from producing a false hit.
- Only parents selected for the next widening revisit are retained. With beam 256
  and revisit fraction 0.25, both formal runs peaked at 64 cached parents.
- Fresh matcher rows and cached rows are merged in original beam-parent order, so
  deterministic structural tie breaking is unchanged.
- A single `bincount` supplies parent row boundaries; the cache copies only the
  rows owned by selected revisit parents.
- Cache use is rejected with neural candidate audit/prefilter modes because those
  modes also need an encoded context that this cache does not retain.
- The default is off, so existing search commands are unchanged.

## Validation

Remote host: `gpu1`, NVIDIA H100 80GB HBM3, CUDA-visible device 1.

Python environment:
`/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python`
(PyTorch 2.4.0+cu121).

Tests:

```text
python3 -m unittest -v test_search_widening.py test_widening_candidate_cache.py
Ran 12 tests in 0.055s
OK
```

The formal timing order was cache-off, cache-on, cache-off. This brackets the
cache run because the shared host showed material timing drift. Each variant used
beam 256, deterministic feedback widening, 128 actions per parent, 8 rank bands,
25% revisit slots, and a 100,000 exact-apply budget.

The reproducible entry point is:

```bash
./run_widening_candidate_cache_ab.sh QASM OUTPUT_DIR GPU 0 512 100000
```

`summarize_widening_candidate_cache_ab.py` fails if any non-timing structural
search result differs among the three runs.

## Formal results

### Barenco raw 58-gate input

Input:
`data/fullseq_36_0_forward/0_58_0_8_121.qasm`

All variants reached 38 gates at action depth 69, completed depth 95, attempted
100,000 actions, and produced final beam digest
`6330582e4454d1d50822b91fffbd85b6f5ba889dba92da42a0fbc148b12fc59c`.

| Metric (seconds) | Off before | Cache on | Off after | On vs off mean |
|---|---:|---:|---:|---:|
| Model match | 3.330 | 2.836 | 3.542 | -17.47% |
| Search excluding Quartz apply | 6.079 | 6.098 | 6.466 | -2.79% |
| Quartz apply | 15.404 | 15.915 | 17.211 | timing drift |
| Search total | 21.483 | 22.013 | 23.676 | -2.51% |
| Process wall | 95.262 | 97.301 | 104.541 | -2.60% |

The cache reused 1,303,320 candidate rows and generated 3,933,648 rows. Peak
resident state was 16,975 rows, 64 parents, and 1,290,100 bytes.

### GF step-68 372-gate input

Input:
`data/quarl_hard_paths_20260905/gf2^6_mult/370_2/68_372_0_47_1591.qasm`

All variants remained at 372 gates, completed depth 275, attempted 100,000
actions, and produced final beam digest
`dcd482c989ab4d2aed2421b86578abb520c529ecbb1aceb6969cd73257457386`.

| Metric (seconds) | Off before | Cache on | Off after | On vs off mean |
|---|---:|---:|---:|---:|
| Model match | 41.502 | 33.231 | 42.366 | -20.75% |
| Search excluding Quartz apply | 69.319 | 63.835 | 70.580 | -8.74% |
| Quartz apply | 168.392 | 178.648 | 171.651 | timing drift |
| Search total | 237.710 | 242.483 | 242.231 | +1.05% |
| Process wall | 314.138 | 323.568 | 325.747 | +1.13% |

The cache reused 49,194,810 candidate rows and generated 147,926,689 rows. Peak
resident state was 183,552 rows, 64 parents, and 13,949,952 bytes.

The cache-on GF run had substantially slower Quartz apply time despite identical
actions and final identities. Therefore the raw total/wall comparison is not a
clean cache measurement. The isolated non-apply path shows the useful result:
8.74% less time than the mean of the bracketing off runs.

Machine-readable numbers are in
`benchmark_results/widening_candidate_cache_20260909.json`. Full JSON, QASM, wall
files, and stdout logs are retained on the shared filesystem under:

```text
/SharedData/dengzy/quarl_matchformer_fresh_20260902/experiment/
  widening_candidate_cache_results_20260909/
```

## Decision and next implementation

Keep this cache opt-in. It is correct and useful on matcher-heavy large graphs,
but raw candidate caching leaves proposal expansion/ranking repeated and gives
only a small Barenco benefit.

The next efficiency change should cache **fully ranked per-parent action tensors**
for the exact graph and slice disjoint 128-action bands from that order. That
eliminates both repeated matching and repeated proposal ranking, while retaining
Quartz as the exact legality/apply authority.

The next branch-quality change should be a sequence-conditioned action ranker,
trained before adding a more elaborate tree scheduler:

1. For each exact parent, collect candidates through rank 1024 and retain the
   rewrite prefix, current gate count, detour/stagnation features, and exact graph
   identity.
2. Label each candidate by the best descendant gate reduction reached within a
   fixed exact-apply horizon, plus time-to-first-improvement. Group labels by the
   same parent so training learns which sibling deserves budget.
3. Train a listwise/pairwise ranker on state embedding, prefix embedding, action
   embedding, matcher score, gate delta, and action-rank band. The primary offline
   gate is teacher-suffix Top-128 coverage; the current GF audit was only 12/64,
   while Top-1024 was 54/64.
4. A/B under equal exact-apply budgets. Require better best-gate count or earlier
   first improvement on held-out Barenco/GF paths; throughput alone is not a
   branch-quality success.
5. Only after the action ranker improves sibling ordering should its calibrated
   continuation value drive DAG-PUCT/branch-budget allocation. Otherwise a more
   complicated scheduler merely deepens incorrectly ranked branches faster.

This ordering separates two problems cleanly: cached ranked actions make deep
bands cheap, and the sequence-conditioned ranker decides which branch merits
those bands.
