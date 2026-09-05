# Rotation No-Contraction Fast Path

Date: 2026-09-06

## Change

The exact replay now computes the nominal successor gate count from the current
Quartz graph and the selected action. When Quartz returns that count, all
declared destination nodes survived rotation normalization, so replay:

- updates only the action's source and destination GUID mappings;
- skips the full `next_graph.nodes` GUID scan;
- skips exact snapshot and topology construction outside periodic audits;
- retains the existing topology audit interval to catch an equal-count
  structural mismatch;
- still takes the full scan and reconciliation path whenever the exact gate
  count differs from the nominal count.

Removing known source GUIDs also keeps each exact checkpoint mapping bounded by
the live graph instead of allowing stale mappings to accumulate with history.

## H100 A/B

Configuration: `gf2^4_mult`, batch 16, horizon 16, seed 1015, exact refresh
every step, topology audit interval 8. All runs produced 258 transitions and
the same best result, 225 to 219 gates. No selected action contracted in this
sample.

| Metric | Rotation off | Initial rotation | Fast path |
| --- | ---: | ---: | ---: |
| Transitions/s | 442.8 | 347.1 | 425.1 |
| Total time | 0.583 s | 0.743 s | 0.607 s |
| Refresh time | 0.117 s | 0.275 s | 0.138 s |
| Exact replay | 0.076 s | 0.116 s | 0.096 s |
| Slot/GUID update | 1.06 ms | 33.99 ms | 1.96 ms |
| Result processing | 37.57 ms | 154.80 ms | 38.23 ms |
| Periodic topology audits | 35 | 0 | 35 |

Relative to the initial correctness implementation, refresh time fell 49.7%
and total throughput increased 22.5%. Relative to rotation-off, the remaining
refresh overhead is 21 ms across the 224 Quartz apply calls, predominantly the
actual Quartz rotation-folding pass. End-to-end throughput is 4.0% lower.

One additional run was affected by unrelated matcher/cache timing variance and
reached 288.9 transitions/s, while its refresh time remained 0.139 s. The table
uses the repeat whose non-refresh matcher/cache times match the rotation-off
baseline; the optimization claim is based primarily on the stable refresh
substage measurements.

## Contraction guard

The saved Barenco xfer `3322` case was replayed again after this optimization.
It recorded one `refresh_replay_rotation_live_guid_scans`, reconciled 42 nominal
gates to the exact 40-gate target, retained exactly 40 GUID/slot mappings, and
matched the saved target graph hash. Therefore the fast path does not bypass
the dynamic-destination case it is intended to preserve.
