# Detailed apply-path profiling (2026-09-08)

## Result

The current end-to-end bottleneck is graph-size-dependent CPU work after the
GPU matcher has produced proposals.  The model is not the dominant cost, and
neither is the hash-table lookup used by exact deduplication.

For Barenco-3 at depth 16, the apply loop consumed 10.267 s of a 13.020 s
instrumented run.  Native Quartz apply was 38.42% of that loop, accepted-child
metadata construction was 23.31%, exact-key construction was 14.51%, and the
pre-apply fingerprint was 9.74%.

For GF(2^6) at depth 8, the apply loop consumed 22.782 s of a 29.502 s run.
Accepted-child metadata was the largest component at 38.02%, followed by native
Quartz apply at 34.10%, exact-key construction at 14.18%, and the pre-apply
fingerprint at 8.84%.

The immediate general-purpose optimization target is therefore to carry an
incremental native graph/snapshot representation through apply and construct
Python metadata only for the final beam.  Within Quartz, the largest target is
to stop copying and rescanning the complete graph for a local rewrite.

## Configuration and scope

Both runs used GPU 3 of `h100-gpu1`, serially, with the current hybrid pipeline:

- beam size 1,000, model microbatch 512;
- maximum 128 actions per parent and proposal factor 16;
- exact graph identity and deduplication before child materialization;
- native conservative successor fingerprint plus neural deferral;
- direct ordered-binding Quartz apply;
- automatic constant/RZ elimination after every successful rewrite;
- Barenco-3 depth 16 and GF(2^6) depth 8.

The new `--apply-profile detailed` mode records nanosecond counters at four
boundaries:

1. Python proposal binding and fingerprint work;
2. individual native apply phases;
3. individual phases inside Quartz `create_new_graph`;
4. exact identity, registry lookup, and accepted-child metadata construction.

Normal runs retain the existing uninstrumented native apply path.  The two
Quartz patch files only add a separate profiled API, so this measurement code is
not on the default path.

## End-to-end observation overhead and result check

The control runs used the same patched binary and arguments, but left
`--apply-profile` at `off`.

| Circuit | Profiled time | Control time | Total overhead | Profiled apply loop | Control apply loop | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Barenco-3 d16 | 13.020 s | 12.091 s | 7.68% | 10.267 s | 9.359 s | 48 |
| GF(2^6) d8 | 29.502 s | 28.947 s | 1.92% | 22.782 s | 22.313 s | 485 |

Barenco's profiled and control runs have the same exact final-beam digest and
the same best-circuit digest.  GF has multiple tied choices: its final-beam
digest differs between repeated runs, but both runs selected the same exact
485-gate best circuit.  Consequently, the percentages below describe where the
instrumented apply-loop time went; the control times above are the production
throughput measurements.

## Exclusive apply-loop breakdown

These rows are mutually exclusive at the Python apply-loop level.  Native
subphases and child subphases in later tables are nested within their respective
rows and must not be added again.

| Stage | Barenco-3 seconds | Barenco share | GF(2^6) seconds | GF share |
| --- | ---: | ---: | ---: | ---: |
| Native apply wall time | 3.944 | 38.42% | 7.768 | 34.10% |
| Accepted-child metadata | 2.393 | 23.31% | 8.661 | 38.02% |
| Exact graph-key construction | 1.490 | 14.51% | 3.230 | 14.18% |
| Pre-apply fingerprint total | 1.000 | 9.74% | 2.015 | 8.84% |
| Python loop/unattributed | 0.922 | 8.98% | 0.527 | 2.31% |
| Python slot/GUID binding preparation | 0.372 | 3.63% | 0.455 | 2.00% |
| Exact registry membership/insert | 0.093 | 0.91% | 0.109 | 0.48% |
| Post-apply fingerprint audit | 0.052 | 0.50% | 0.017 | 0.08% |

The exact dedup registry itself is only 0.5--0.9% of apply time.  Replacing the
set lookup with a neural similarity model cannot materially improve throughput.
The expensive part of exact dedup is producing the complete graph key, not
looking the key up.

## Candidate flow and legality

| Circuit | Apply attempts | Native success | Native reject | Accepted unique children | Duplicate successful graphs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Barenco-3 | 52,085 | 45,336 | 6,749 | 15,115 | 30,221 |
| GF(2^6) | 11,456 | 10,896 | 560 | 7,465 | 3,431 |

Barenco's rejects were 3,743 input-qubit-alias failures and 3,006 cycle
failures.  GF's were 376 aliases and 184 cycles.  No other native rejection
category occurred in these runs.  Exact dedup discarded 66.7% of Barenco's
successful applies and 31.5% of GF's, while metadata was already postponed
until after that exact decision.

Average costs make the graph-size scaling visible:

| Operation | Barenco-3 | GF(2^6) | GF/Barenco |
| --- | ---: | ---: | ---: |
| Native apply per attempt | 75.7 us | 678.0 us | 8.95x |
| Exact key per successful apply | 32.9 us | 296.4 us | 9.02x |
| Child metadata per accepted child | 158.3 us | 1,160.3 us | 7.33x |
| Fingerprint parent-profile build | 61.6 us | 545.9 us | 8.86x |
| Batched fingerprint per candidate | 2.49 us | 3.49 us | 1.40x |

The fingerprint's batched candidate kernel scales well.  Its graph-wide parent
profile construction does not: on GF it consumed 1.873 s of the 2.015 s total
fingerprint time.  The useful optimization is to derive that profile from state
already maintained by the search, rather than replacing the cheap batched
candidate operation.

## Native Quartz apply breakdown

| Native phase | Barenco-3 | Share of native wall | GF(2^6) | Share of native wall |
| --- | ---: | ---: | ---: | ---: |
| Graph rewrite/create | 1.979 s | 50.16% | 3.498 s | 45.03% |
| Constant/RZ elimination | 0.856 s | 21.71% | 2.138 s | 27.52% |
| Full cycle check | 0.708 s | 17.95% | 1.718 s | 22.12% |
| GUID lookup | 0.116 s | 2.95% | 0.295 s | 3.80% |
| Source-pattern validation | 0.103 s | 2.61% | 0.036 s | 0.46% |
| Cython/wrapper unattributed | 0.080 s | 2.03% | 0.032 s | 0.41% |

The remaining native validation, trace, destination construction, and unmatch
phases are individually below 1% of native wall time.  The sequential GUID
lookup previously suspected as an apply bottleneck is measurable, but it is not
the main issue.

### Inside graph rewrite/create

| `create_new_graph` phase | Barenco-3 | Graph-rewrite share | GF(2^6) | Graph-rewrite share |
| --- | ---: | ---: | ---: | ---: |
| Rebuild logical-qubit position index | 1.008 s | 50.93% | 2.234 s | 63.86% |
| Copy input and output edge maps | 0.554 s | 27.99% | 1.027 s | 29.37% |
| Remove source operations | 0.182 s | 9.20% | 0.060 s | 1.72% |
| Add destination operations | 0.081 s | 4.08% | 0.023 s | 0.65% |
| Copy constant parameter map | 0.062 s | 3.11% | 0.114 s | 3.25% |
| Reconnect outputs | 0.056 s | 2.81% | 0.021 s | 0.61% |

For GF, 93.2% of graph creation is just rebuilding the logical-qubit index and
copying both complete edge maps.  The local rewrite itself is small.  This is
the strongest evidence that a native persistent/copy-on-write graph, or an
incremental rewrite result, will have much greater value than another proposal
classifier.

## Accepted-child metadata breakdown

| Child phase | Barenco-3 | Child share | GF(2^6) | Child share |
| --- | ---: | ---: | ---: | ---: |
| Materialize all native nodes as Python objects | 0.453 s | 18.93% | 2.082 s | 24.04% |
| Recompute rewrite-distance BFS | 0.490 s | 20.49% | 1.992 s | 23.00% |
| Convert/sort all edge rows | 0.292 s | 12.21% | 1.211 s | 13.98% |
| Diff full before/after snapshots | 0.254 s | 10.63% | 1.016 s | 11.73% |
| Extract all native edges | 0.200 s | 8.37% | 0.696 s | 8.04% |
| Convert/sort all node rows | 0.190 s | 7.96% | 0.614 s | 7.09% |
| Local metadata/set construction | 0.192 s | 8.04% | 0.423 s | 4.88% |
| Update persistent slot mapping | 0.139 s | 5.81% | 0.443 s | 5.11% |

There is no single expensive Python function here.  The cost comes from several
complete graph traversals for each accepted child.  A useful redesign should
return the removed/source GUIDs, new destination GUIDs, changed native edges,
and affected-hop frontier directly from Quartz.  With that delta, the search
can update slots and local embeddings without materializing and comparing two
complete snapshots.

## Optimization order implied by the profile

1. Add a native incremental rewrite result and carry native adjacency/slot
   state between depths.  Defer Python node/edge snapshot materialization until
   the final beam or until a consumer explicitly needs it.
2. Update `pos_2_logical_qubit` only on affected wires and avoid copying the
   complete `inEdges` and `outEdges` maps for every candidate.  A persistent
   base graph plus a small rewrite delta is the natural representation.
3. Localize or fuse the three remaining full-graph passes: cycle detection,
   constant/RZ elimination, and exact-key construction.  The rewrite delta
   gives each pass a small affected frontier.
4. Cache the native fingerprint parent profile in `BeamState`, deriving it from
   the same maintained adjacency instead of rebuilding it on first reuse.
5. Only after these changes revisit neural invalid/duplicate prediction.  In
   the current profile neural scoring is 0.114 s for Barenco and 0.042 s for GF,
   while exact registry membership is also negligible.

This order is circuit-general: every proposed change removes an observed
whole-graph pass whose cost grows sharply from the 58-gate Barenco input to the
495-gate GF input.  It does not encode either circuit or any particular rewrite
trajectory.

## Reproduction and artifacts

Apply both Quartz patches, rebuild the native library and Cython extension, and
run `run_detailed_apply_profile.sh`.  For example:

```bash
bash run_detailed_apply_profile.sh 3 barenco_tof_3.qasm 16 profile_barenco_d16
bash run_detailed_apply_profile.sh 3 'gf2^6_mult.qasm' 8 profile_gf6_d8
python summarize_apply_profile.py \
  benchmark_results/profile_barenco_d16.json \
  benchmark_results/profile_gf6_d8.json
```

Committed artifacts include:

- `quartz_patches/detailed_apply_profile.patch`;
- `quartz_patches/detailed_graph_rewrite_profile.patch`;
- both full detailed JSON profiles and their uninstrumented controls;
- `benchmark_results/detailed_apply_profile_summary_20260908.json`, a compact
  machine-readable hierarchy derived from the full profiles.
