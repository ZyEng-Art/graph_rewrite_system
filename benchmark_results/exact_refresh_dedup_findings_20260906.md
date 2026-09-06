# Exact graph-hash deduplication at refresh (2026-09-06)

## Implementation

Exact deduplication is now enabled by default at every Quartz refresh. Normal
speculative layers retain the existing incremental raw-topology fingerprint;
they do not materialize a Quartz graph or compute a Quartz hash. Once
`replay_state()` has already produced a valid, topology-exact `PyGraph`, the
refresh path calls `graph.hash()` once (and `PyGraph` caches that result),
rejects hashes already in a separate `seen_exact` set, and continues scanning
the existing over-generated pool until either the exact-unique beam is full or
the pool is exhausted.

`seen_exact` is initialized with the root graph and persists across refreshes
and best-root restarts. It therefore prevents refreshed leaves from returning
to a previously retained exact graph. It does not reproduce CPU Quartz's still
stricter per-layer semantics because speculative intermediate states are not
materialized between refreshes. `--no-refresh-exact-dedup` preserves the old
behavior for controlled A/B measurements.

The implementation records replay-valid states, exact-unique accepted states,
exact duplicates, hash time, and the size of the persistent exact-hash set.
The exact hash is also included in the optional refresh stage profile. Memory
overhead is one host integer per distinct refreshed graph.

## Correctness

Three unit tests check one-call hash registration, duplicate rejection, and
duplicate detection across distinct graph objects. A profiled beam-32 smoke
enabled exact dedup by default: 25 replay-valid leaves collapsed to seven
unique leaves, the final independent audit reported seven valid, topology-exact
and exact-unique graphs, and the profile counters recorded 25 hash calls and 18
duplicates.

All larger H100 results below independently replay the final beam. Every final
state is Quartz-valid and topology-exact, and every audited final hash is
unique. Thus `final_beam_size == unique_exact_graph_hashes` whenever exact
deduplication is enabled.

## Batch-1000, depth-3 A/B

The off and on runs use the same selected checkpoint, r99.9 calibration, H100,
GPU microbatch 512, full GPU proposals, raw speculative deduplication, indexed
lazy topology, gate ranking, per-parent cap 128, and exact refresh at depth 3.
Search time already includes the exact refresh and excludes the redundant
independent final audit.

### Same candidate-pool diagnostic

| Circuit/config | Exact dedup | Median search | Final beam | Unique exact hashes | Median refresh | Median hash time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GF, proposal 8x / refresh 2x | off | 3.1012 s | 1,000 | 524 | 1.2474 s | 0 |
| GF, proposal 8x / refresh 2x | on | 4.9978 s (one run) | 972 | 972 | 3.1352 s | 0.5871 s |
| Barenco, proposal 64x / refresh 32x | off | 3.3959 s | 1,000 | 312 | 0.2863 s | 0 |
| Barenco, proposal 64x / refresh 32x | on | 3.5042 s | 445 | 445 | 0.4121 s | 0.0372 s |

With the unchanged Barenco pool, exact dedup increases median total search time
by only 3.2%. It scans all 17,689--17,690 refreshed candidates instead of
stopping after the first 1,000 replay-valid trajectories, increasing exact
coverage from 312 to 445 graphs. Hash computation itself is 1.06% of search.

The unchanged GF 2x refresh pool contains only 972 non-root exact graphs, so it
cannot fill 1,000. Scanning all 2,000 replay-valid candidates costs more than
stopping after 1,000, and exact hashing is 11.7% of the one-run search time.

### Full exact-unique beam

GF needs only a refresh-factor increase from 2 to 3. Barenco needs proposal
factor 128 and refresh factor 64; with the per-parent cap of 128 this is nearly
the complete ranked proposal set.

| Circuit | CPU Quartz | Exact-dedup GPU median | Exact-unique beam | GPU/CPU | Median hash time | Hash share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GF | 276.1948 s | 5.3274 s | 1,000 / 1,000 | **51.84x faster** | 0.6036 s | 11.33% |
| Barenco | 4.0757 s | 6.7068 s | 1,000 / 1,000 | **1.65x slower** | 0.0676 s | 1.01% |

All three GF runs replayed 2,054 valid candidates, rejected 1,054 exact
duplicates, and stopped at 1,000 unique graphs. At 371 gates, one Quartz hash
cost about 294 microseconds. The strict result remains more than fifty times
faster than CPU Quartz even after exact uniqueness is enforced.

All three full-beam Barenco runs attempted 127,722 speculative proposals and
examined 37,187 refreshed candidates. Of 2,550 replay-valid trajectories,
1,550 were exact duplicates and 1,000 were unique. At 39 gates, one hash cost
about 27 microseconds. Median depth-3 stage time was dominated by lazy
successor construction (4.2265 seconds) and paged-cache advance (0.7767
seconds); exact refresh was 0.8323 seconds, of which hashing was only 0.0676
seconds. The strict Barenco slowdown is therefore caused by the very large,
mostly invalid or convergent candidate pool required for 1,000 unique graphs,
not by graph hashing.

## Two-refresh regression

A beam-256, depth-16 search with refresh interval 8 verifies that exact hashes
remain global across refreshes:

| Circuit | Search | Step-8 replay-valid / new unique / duplicate | Step-16 replay-valid / new unique / duplicate | Final audit |
| --- | ---: | ---: | ---: | ---: |
| GF | 4.6114 s | 332 / 256 / 76 | 359 / 256 / 103 | 256 valid, topology-exact, unique |
| Barenco | 6.0908 s | 734 / 40 / 694 | 2,694 / 109 / 2,585 | 109 valid, topology-exact, unique |

GF maintains a full exact-unique beam at both refreshes. Barenco does not: even
with proposal factor 128 and refresh factor 64, the first refresh finds only 40
new exact graphs and the second only 109. This is direct evidence that
Barenco's remaining limitation is candidate validity/diversity and exact-state
convergence. Keeping duplicate histories previously hid this beam collapse.

## Performance interpretation

The low-overhead part of the request is satisfied at the hash boundary:

- non-refresh layers add no Quartz construction or exact-hash work;
- each successful exact replay calls `PyGraph.hash()` once, after which that
  graph object retains the cached result;
- hash time is measured separately and accounts for about 1% of strict
  Barenco search and 11% of strict GF search;
- the persistent host set is tiny relative to model and paged-cache storage.

The larger total-time increase required to obtain 1,000 exact-unique states is
not hash overhead and should not be optimized away by retaining duplicates. It
is the cost of scanning more speculative candidates and performing more Quartz
replays. Improving Barenco now requires higher model legality and more diverse
candidate ranking, or accepting an honestly smaller beam; exact duplicates
must no longer be used to make the beam appear full.

Raw results and SHA-256 checksums are listed in
`benchmark_results/exact_refresh_dedup_summary_20260906.json`.
