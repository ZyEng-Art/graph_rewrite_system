# Hierarchical Rotation Reconciliation

Date: 2026-09-06

## Remote branch audit

- `origin/barenco-trajectory-repro-clean-20260905` adds an optional
  unthresholded near-source reserve to the full paged matcher. It recovers one
  known GF teacher action, but evaluates 16,384 extra rough candidates for that
  state and was tuned against that holdout. It is not enabled in the
  hierarchical node-first matcher.
- `origin/refresh-consistency-model-20260906` adds refresh-view supervision and
  a consistency loss. This changes matcher training and needs a newly trained
  checkpoint; it is not a drop-in runtime fix.
- Existing beam, lazy, and full paged entry points exposed rotation
  elimination. The hierarchical PPO collector did not, and its static lazy
  topology could not continue after Quartz removed normalized-away RZ nodes.

## Implementation

`--eliminate-rotation` now gives Quartz authority over the graph after every
hierarchical rewrite. It requires `--refresh-interval 1` so no later action is
predicted against a stale topology. A successful exact refresh:

1. removes GUID/slot mappings for gates eliminated by Quartz;
2. replaces the lazy nodes, edges, gate count, locality, and topology hash with
   the exact normalized topology;
3. corrects the PPO transition's next gate count and shaped reward;
4. masks eliminated paged slots and restores exact live gate types before the
   next actor/matcher call;
5. preserves the causal action-prefix cache, because the selected rewrite
   action remains part of policy history even when its declared destination is
   normalized away.

## Real contraction audit

The saved Barenco `40_2` transition applies xfer `3322` to a 42-gate graph.
The static rewrite has two source and two destination RZ operations, so its
nominal result remains 42 gates. Quartz folds both declared destinations to a
zero rotation and removes them, producing the saved 40-gate successor.

| Check | Result |
| --- | ---: |
| Exact Quartz target hash | match |
| Lazy nominal / exact gates | 42 / 40 |
| Declared / surviving destination gates | 2 / 0 |
| Corrected reward, step penalty 0.02 | 1.98 |
| Exact paged live slots | 40 |
| Rotation topology reconciliations | 1 |
| Profiled refresh time | 0.528 ms |

The machine-readable record is
`hierarchical_rotation_contraction_xfer3322_h100.json`.

## Ordinary rollout A/B

On `gf2^4_mult`, batch 16, horizon 16, seed 1015, both modes produced 258
transitions and the same 225 to 219 best result. This sample did not select a
contracting rewrite: all 258 rotation refreshes reported unchanged topology.

| Mode | Transitions/s | Total time | Refresh time |
| --- | ---: | ---: | ---: |
| Rotation disabled | 442.8 | 0.583 s | 0.117 s |
| Rotation enabled, initial implementation | 347.1 | 0.743 s | 0.275 s |

The initial correctness path scans the live Quartz GUIDs and builds an exact
snapshot every step. This makes rotation mode opt-in and motivates a separate
no-contraction fast-path optimization. These numbers do not establish a search
quality gain because no rotation contraction occurred in this A/B.
