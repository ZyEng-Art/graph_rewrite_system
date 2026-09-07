# Source-topology matcher throughput A/B (H100, 2026-09-07)

## Conclusion

The source-topology change fixes the identified Barenco candidate miss without
materially changing matcher throughput at the same source-anchor cap.  At cap
8192, replacing the 6,930,383-parameter control with the 7,269,455-parameter
topology model changes the complete GPU proposal rate by `+0.92%` on GF and
`-3.52%` on Barenco.  The static source-topology representation is computed
once per loaded matcher/ECC set and reused across every state and refresh.

The maximum-recall operating point also raises the pre-structural-decode cap
from 8192 to 10240 so that GF rolling-window step 251 is not truncated.  With
both changes enabled, the complete production-like GPU proposal path changes
by `-0.22%` on GF and `-3.15%` on Barenco relative to the old model at cap
8192.  It remains `305.46x` and `105.09x`, respectively, faster than original
CPU Quartz xfer-at-anchor enumeration at this batch-512 matcher boundary.

This is a matcher/proposal throughput comparison, not an end-to-end optimizer
speedup.  CPU successor construction, exact graph application, deduplication,
and the search width/depth policy are outside the timed boundary.

## Controlled setup

Each circuit's three configurations were run sequentially on the same otherwise
idle H100 80 GB.  The batch contains 512 real trajectory states, cyclically
repeated when the saved trajectory has fewer than 512 states.  All GPU numbers
are medians of ten repetitions.  The recall target is 0.999, the per-parent
proposal cap is 128, and the final global proposal cap is 8192.  QASM parsing,
model loading, and static source-vector construction are excluded from the
per-state hot path.  The unchanged CPU numbers are reused from the earlier
identical-state benchmark.

The first comparison holds the source-anchor cap fixed, isolating the cost of
the added topology encoder:

| Circuit | Timed path | Control, cap 8192 | Topology, cap 8192 | Change |
| --- | --- | ---: | ---: | ---: |
| GF | GPU matcher through structural decode | 2123.76 states/s | 2104.74 states/s | -0.90% |
| GF | Complete GPU expansion/ranking + selected-only D2H | 1870.88 states/s | 1888.11 states/s | +0.92% |
| Barenco | GPU matcher through structural decode | 9624.18 states/s | 9728.03 states/s | +1.08% |
| Barenco | Complete GPU expansion/ranking + selected-only D2H | 7328.81 states/s | 7070.62 states/s | -3.52% |

The final comparison includes the cap increase needed for 271/271 GF
rolling-window-8 teacher coverage:

| Circuit | Timed path | Control 8192 | Topology 10240 | Change | Versus CPU Quartz |
| --- | --- | ---: | ---: | ---: | ---: |
| GF | GPU matcher through structural decode | 2123.76 states/s | 2075.68 states/s | -2.26% | 339.65x |
| GF | Complete GPU proposal path | 1870.88 states/s | 1866.72 states/s | **-0.22%** | **305.46x** |
| Barenco | GPU matcher through structural decode | 9624.18 states/s | 7790.12 states/s | -19.06% | 115.34x |
| Barenco | Complete GPU proposal path | 7328.81 states/s | 7097.89 states/s | **-3.15%** | **105.09x** |

The Barenco matcher-core row exposes the cost of testing up to 25% more
above-threshold source-anchor pairs before structural decoding.  It does not
translate into a 19% proposal-boundary regression because expansion, ranking,
and the fixed 8192 final output dominate the small-circuit complete path.  GF's
denser downstream workload almost completely amortizes the larger source cap.

The all-row Python host-materialization diagnostic falls by 11.28% on GF and
10.28% on Barenco in the final configuration.  That path copies every retained
candidate and is intentionally not the deployment boundary; the complete GPU
proposal path copies only the final globally capped proposals.

## Static source cache

The ECC set has 3,855 source patterns.  After five warmups, 30 synchronized
measurements on the same H100 give:

| Matcher | Source cache shape / size | Median build time | P90 |
| --- | ---: | ---: | ---: |
| Control S412 | 3855 x 128 / 986,880 bytes | 0.195 ms | 0.205 ms |
| Topology 0.6875 S414 | 3855 x 128 / 986,880 bytes | 0.780 ms | 0.796 ms |

The added topology layer therefore costs about 0.585 ms once when source
vectors are built.  `paged_rollout_benchmark.py` constructs these vectors
outside the search loop, so refreshes recompute the current circuit encoding
but reuse the same source cache.  The output cache shape and memory footprint
are unchanged.

## Interpretation and remaining limit

For the two target trajectories, the candidate-coverage issue is resolved at
the selected operating point: GF retains 271/271 teacher actions and full
Barenco retains 116/116 for both rolling-8 and rolling-64 causal inputs.  This
does not mean the general matcher is perfect.  On the strict disjoint audit it
still retains 1528/1536 teacher actions and 99.3933% of all exact bindings.
The result should therefore be described as fixing the identified target miss
and improving the missing structural representation, not eliminating every
possible long-tail miss.

The six raw A/B files are:

- `matcher_throughput_ab_gf_control_s412_cap8192_20260907.json`
- `matcher_throughput_ab_gf_topo_a06875_cap8192_20260907.json`
- `matcher_throughput_ab_gf_topo_a06875_cap10240_20260907.json`
- `matcher_throughput_ab_barenco_control_s412_cap8192_20260907.json`
- `matcher_throughput_ab_barenco_topo_a06875_cap8192_20260907.json`
- `matcher_throughput_ab_barenco_topo_a06875_cap10240_20260907.json`

Their SHA-256 values, in the same order, are
`bc0ebba0e95accd6c91492b060d378e6d4acdac9e89c7db94a6760b91d0bcef1`,
`b6919d0139593e3da8b5f6ad76f743556b2617f62cc8e6df9a4c7f024cf74a1a`,
`b6eb894317afbac7804b48f2aa788c0c289aa98db8ed59eb89b2641dc897d179`,
`8bc7c2accd232a5f6f2ec89679bc1a85b2e8474c965646f91d53718b57a5948b`,
`83914f7e699329563d0a1b2f1ae310dbdf2e596ba876af36102dd2913e0d4000`,
and `fb4ccb695a0d8477bae691c66dba9d8a17ac246babe9e66d201b581aff0e211d`.
