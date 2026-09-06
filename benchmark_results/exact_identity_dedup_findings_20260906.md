# Low-overhead exact circuit deduplication (2026-09-06)

## Decision

The refresh path now defaults to deterministic, exact circuit-state identity.
It does not use a classifier to guess whether two action sequences are
equivalent, and it does not rely on Quartz `Graph.hash()` as if that value were
an exact identity.

The key is the ordered operation trace on every physical qubit. Each operation
token contains its gate type, exact parameters, and ordered physical-qubit
operands. Therefore:

- exchanging two independent actions produces the same key;
- inverse cycles that return to the same complete circuit produce the same key;
- dependent gate order, control/target roles, physical wiring, and rotation
  parameters remain distinct;
- equality compares the full byte string, not a truncated hash.

`quartz_patches/exact_graph_key.patch` implements the key in C++ and exposes it
as `PyGraph.exact_key() -> bytes`. `circuit_identity.py` automatically uses a
linear-time QASM implementation when Quartz has not yet been rebuilt with the
patch. That fallback is exact for the Quartz QASM emitted in this workload, but
the native path avoids QASM formatting/parsing and is substantially faster.

The exact registry is global across refresh boundaries. A state encountered at
an earlier global depth has at least as much remaining search depth as a later
return to the same circuit, so the later circuit-level path is redundant. The
proposal model is history-conditioned, however, so two representatives can
receive different learned scores even when the circuit and legal continuations
are identical. The implementation keeps the first candidate encountered in
ranked beam order; this is a deliberate search-diversity tradeoff, not a claim that
the neural cache states are equal.

## Why sequence-equivalence prediction was not added

A learned equivalence model would introduce inference cost and false-positive
risk precisely where a complete state is already available during refresh.
Exact state identity answers the relevant question directly. It also handles
long inverse cycles and arbitrary permutations of independent actions without
requiring pairwise sequence comparisons.

Two cheaper action-level ideas were implemented as diagnostics and audited:

- reject an apparent reverse xfer immediately after its forward xfer;
- impose a canonical ordering on a suffix of apparently independent actions.

They remain disabled by default. The compact ECC metadata omits parameter maps,
so a reverse source/destination string is not a proof of a concrete inverse.
More importantly, the authoritative `gf2^6_mult/370_2` trajectory would lose 23
actions under the apparent-inverse filter and 11 actions under the canonical
order filter. These shortcuts are therefore not used in the production path.
When disabled, the per-candidate conflict-footprint work is not performed.

## Duplicate diagnosis

Barenco `38_3` was rerun with exact refresh deduplication disabled and every
final history replayed in Quartz.

| Search | Histories | Exact final states | Duplicate excess | Main evidence |
|---|---:|---:|---:|---|
| beam 1000, depth 3 | 1000 | 320 | 680 | 582 Quartz-proven commutable adjacent pairs in 464 histories |
| beam 256, depth 8 | 256 | 11 | 245 | 926 direct reverse-pattern occurrences; all 256 histories affected |

At depth 3, raw QASM text has 380 distinct serializations but the per-wire exact
identity has only 320. Thus 60 duplicate states are caused solely by different
serializations/orders of independent operations and would be missed by raw
QASM string equality. The depth-8 population is dominated by xfers 38 and 39,
which account for 1,922 of 2,048 actions and repeatedly walk around the same
small state set.

The saved high-quality paths contain the same phenomenon:

| Reference path | Saved states | Actions | Exact unique states | Exact repeats |
|---|---:|---:|---:|---:|
| Barenco `38_3` | 17 | 16 | 15 | 2 |
| `gf2^6_mult/370_2` | 272 | 271 | 136 | 136 |

Barenco repeats states at steps 4/6 and 11/13. The GF path contains many longer
return-to-state groups. Native keys and independently computed QASM keys induce
exactly the same partition on both paths, and every stored cost matches the
reconstructed Quartz gate count. Removing these loops does not remove a circuit
state or a legal continuation: the suffix can start from the earlier occurrence
of the identical circuit.

## H100 identity A/B

The native exact key and legacy `Graph.hash()` were compared on the same H100,
checkpoint, calibration, beam 1000, depth 3, refresh at depth 3, and microbatch
512. There were three runs per mode; the table reports medians, and search time
excludes the final diagnostic replay audit.

| Circuit | Identity | Identity time | Time/call | Search time |
|---|---|---:|---:|---:|
| GF `370_2` | native exact | 0.558 s | 271.5 us | 6.059 s |
| GF `370_2` | legacy hash | 0.691 s | 336.3 us | 6.198 s |
| Barenco `38_3` | native exact | 0.0647 s | 26.2 us | 7.262 s |
| Barenco `38_3` | legacy hash | 0.0773 s | 30.3 us | 7.755 s |

Despite carrying the full identity, the native key is faster than the existing
Quartz hash: identity time drops 19.3% on GF and 16.3% in total on Barenco.
Median end-to-end search time drops 2.25% and 6.36%, respectively. On Barenco,
the legacy hash also has to replay roughly 85 additional candidates while
trying to fill the beam because it falsely merges distinct states.

The collision problem is measurable, not hypothetical. In the audited
Barenco 1000-state exact beam, there are 1000 native/QASM identities but only
950 Quartz hash values. In the depth-16 final beam, 112 exact identities map to
only 110 Quartz hashes. The legacy mode is retained only behind
`--refresh-dedup-identity quartz_hash` for reproducible A/B measurements.

## Depth-16 regression

With beam 256, depth 16, exact refresh every 8 actions, and native exact keys:

| Circuit | Search | Valid replays | Exact duplicates removed | Accepted exact states | Final audit |
|---|---:|---:|---:|---:|---:|
| GF `370_2` | 4.645 s | 691 | 179 | 512 | 256/256 |
| Barenco `38_3` | 7.752 s | 3348 | 3196 | 152 | 112/112 |

Barenco's final beam has 112 rather than 256 entries because the candidate pool
is exhausted after exact filtering: 95.46% of its valid refresh replays are
states already seen. This is the direct reason duplicate generation was making
Barenco spend large replay effort without adding search-state coverage. The
identity computation itself takes only 0.088 s, or 1.13% of search time.

## Validation

- 49/49 Python unit tests passed in the H100 environment.
- The patched native extension passed independent-order, dependent-order,
  parameter, and bytes-interface integration checks.
- Every transition in both authoritative paths replayed exactly in Quartz:
  Barenco 16/16 and GF 271/271.
- Indexed speculative topology replay also passed Barenco 16/16 and GF 271/271.
- Native and QASM exact-identity partitions have zero mismatches in the saved
  path audit and all final-beam audits.
- Tests explicitly show that different physical wiring and different RZ
  parameters are not merged even when `Graph.hash()` is identical.

## Quartz patch and runtime use

Apply the patch from the root of the matching Quartz checkout and rebuild both
the runtime and Python extension according to that checkout's normal build
procedure:

```bash
git apply /path/to/graph_rewrite_system/quartz_patches/exact_graph_key.patch
```

No rollout flag is required after rebuilding: exact refresh identity is the
default. If the loaded extension does not expose `exact_key`, the Python QASM
fallback is selected automatically. The unsafe compatibility comparison is
explicit:

```bash
python paged_rollout_benchmark.py ... \
  --refresh-exact-dedup \
  --refresh-dedup-identity quartz_hash
```

The structured aggregate is
`benchmark_results/exact_identity_dedup_summary_20260906.json`. Raw H100 runs
are the `dedup_native_final_*` and `dedup_hash_ab_*` JSON files; the no-dedup
diagnosis is in `duplicate_analysis_barenco_*`; authoritative-path checks are
in `dedup_filter_reference_audit_20260906.json` and
`dedup_saved_trajectory_identity_audit_20260906.json`.
