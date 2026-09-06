# End-to-end CPU Quartz versus GPU paged search (2026-09-06)

> Historical note: this benchmark predates exact graph-hash filtering at
> refresh and therefore permits several trajectories to occupy the beam with
> the same exact Quartz graph. The strict exact-unique follow-up supersedes its
> final-beam speedup interpretation; see
> `benchmark_results/exact_refresh_dedup_findings_20260906.md`.

## Why this benchmark was needed

The earlier batch-512 result timed candidate matching and, in the follow-up,
GPU proposal expansion/ranking through selected-only D2H. It did not include
successor construction, lazy state/cache updates, graph deduplication, or the
periodic exact Quartz refresh. The beam-256 depth-8 check established search
legality/equivalence of full versus preselected GPU proposal expansion, but it
was not an end-to-end comparison against original CPU Quartz. Consequently,
the earlier 119--302x figures must not be described as optimizer-level speedup.

This follow-up measures the historical search shape: beam 1000, depth 3, and a
GPU microbatch of 512. There was no technical blocker to running it; omitting
it previously was a benchmark-scoping mistake.

## Timing boundary and configuration

Both paths start from the first circuit in the corresponding held-out
high-quality trajectory and use the same 4,700-xfer Quartz context, gate-count
ranking, per-parent action cap 128, maximum gate increase 1, rotation
elimination, beam 1000, and depth 3. QASM parsing, checkpoint loading, dataset
loading, and Quartz-context construction are excluded from both timers.

- CPU `quartz` mode performs original Quartz exact matching on every input
  graph, ranks/caps candidates, copies and applies real Quartz graphs, and uses
  exact graph-hash deduplication. Quartz's CPU API has no GPU-like microbatch
  knob; the batch-512 convention applies to GPU inference.
- GPU `paged_action_lazy` mode includes tensor preparation, model matching,
  r99.9 filtering, full GPU action expansion/ranking, selected-proposal D2H,
  indexed lazy topology/update/hash work, paged-cache advance, and the exact
  Quartz refresh at depth 3.
- The primary GPU time is `search_seconds_excluding_audit`. Despite the field
  name, it already includes the final exact Quartz refresh that validates the
  returned beam. `quartz_replay_audit.audit_seconds` is a second, independent
  replay of every returned trajectory, so adding it is a deliberately
  conservative double-validation number rather than required search work.

GF used proposal factor 8 and refresh factor 2. Barenco initially used those
same values but only 162 of 1,000 returned speculative states survived exact
refresh. To compare full final beams rather than 1,000 CPU states against 162
GPU states, the reported Barenco run uses proposal factor 64 and refresh factor
32. At depth 3 this ranks 64,000 proposals and exact-replays candidates until
1,000 valid trajectories are obtained.

## Wall-clock result

GPU search was repeated three times. The table uses the median search time;
the CPU exact run is a single run whose matching rates agree with the prior
independent batch-512 measurements.

| Circuit/start | CPU Quartz | GPU search median | End-to-end speedup | GPU + independent audit | Conservative speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| GF(2^6)-mult `370_2`, 371 gates | 276.1948 s | 3.1104 s | **88.80x** | 8.3766 s | **32.97x** |
| Barenco `38_3`, 39 gates | 4.0757 s | 3.2124 s | **1.27x** | 3.7534 s | **1.09x** |

The three GPU search times were 3.1100, 3.1767, and 3.1104 seconds for GF;
and 3.1784, 3.2124, and 3.2375 seconds for Barenco. Every repeat returned
1,000 states, and every independent audit reported 1,000/1,000 valid
trajectories and 1,000/1,000 exact topology matches.

Both CPU and GPU retained the starting best gate count at depth 3 (371 for GF,
39 for Barenco). This short run measures optimizer throughput at the
historical benchmark shape; it is not evidence that either search reproduced
the later gate-count improvements in the reference trajectories.

## Where the time went

The following stage totals are from the first retained run for each path.

| Circuit/path | Exact/model match | Proposal | Apply or lazy update/hash | Exact refresh | Total search |
| --- | ---: | ---: | ---: | ---: | ---: |
| GF CPU | 256.4127 s | 4.4971 s | 14.6682 s | included in apply | 276.1948 s |
| GF GPU | 0.7864 s | 0.0890 s | 0.8014 s | 1.2557 s | 3.1100 s |
| Barenco CPU | 2.1084 s | 0.0147 s | 1.9477 s | included in apply | 4.0757 s |
| Barenco GPU | 0.3838 s | 0.1140 s | 1.9626 s | 0.2786 s | 3.1784 s |

GF remains matcher-bound on CPU, so replacing exact matching produces a large
optimizer-level gain even after lazy successor work and exact refresh. The
39-gate Barenco circuit is different: exact CPU matching is already cheap,
while the GPU run's 64x overgeneration makes lazy update/hash (1.963 seconds)
roughly as expensive as the entire CPU graph-application stage (1.948
seconds). Thus the matcher-level 121.9x Barenco advantage collapses to a real
end-to-end 1.27x at the full validated-beam boundary.

## Output-quality limitation

The returned beams are legal but are not distributionally or structurally
identical:

| Circuit | CPU exact unique graphs seen | GPU final states | GPU valid/topology-exact | GPU unique exact hashes |
| --- | ---: | ---: | ---: | ---: |
| GF | 2,656 over the search | 1,000 | 1,000 / 1,000 | 524 |
| Barenco | 1,139 over the search | 1,000 | 1,000 / 1,000 | 312 |

The GPU path deduplicates speculative raw states before automatic Quartz
normalization. Multiple distinct speculative histories can therefore collapse
to the same exact graph during refresh. CPU Quartz deduplicates by exact graph
hash throughout. The speedups above are valid for reaching a full beam of
legal, exactly replayable states with the same best gate count; they are not a
claim that the two methods return the same 1,000 unique graphs or explore the
same search distribution. Exact-hash deduplication plus refill at refresh is
the remaining requirement for a strict equal-diversity benchmark.

## Artifacts

The compact machine-readable result is
`benchmark_results/end_to_end_throughput_summary_20260906.json`. Raw CPU and
all three GPU repetitions and the Barenco factor-8 diagnostic are stored beside
it as `benchmark_results/e2e_{gf370_2,barenco38_3}_*.json`; their SHA-256
checksums are recorded in the summary.
