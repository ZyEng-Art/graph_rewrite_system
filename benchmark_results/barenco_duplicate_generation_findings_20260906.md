# Barenco duplicate-generation analysis (2026-09-06)

## Scope

This analysis re-ran Barenco `38_3` with exact refresh deduplication disabled,
exported the retained action histories, replayed every unique prefix with Quartz,
and compared four notions of identity:

1. the slot-sensitive topology used by speculative raw deduplication;
2. Quartz `Graph::hash()`;
3. raw Quartz QASM output;
4. a parameter- and physical-qubit-aware QASM dependency-DAG key that ignores
   only the textual ordering of independent operations.

The dependency-DAG key is a syntactic circuit identity, not a unitary-equivalence
test. Equal keys are the same per-qubit gate DAG; different keys may still be
semantically equivalent through an ECC rule.

## Main finding

The duplicate explosion is primarily a search-state and reversible-action
problem, not a matcher-accuracy problem.

For the depth-8, beam-256 sample:

- all 256 retained histories and all 256 slot-sensitive topologies were unique;
- they represented only **11** parameter/qubit-aware circuit DAGs;
- the largest one circuit DAG occupied **47** beam slots;
- xfers `38` and `39` occupied **1,922 / 2,048 actions (93.8%)**;
- `39` implements `cx 0 1; rz 0; -> rz 0; cx 0 1;`, while `38` implements the
  reverse direction;
- all **256 / 256** histories contained an exact two-action cycle;
- there were **926** exact two-action cycles among the 1,792 adjacent action
  pairs, and all 926 were an inverse applied directly to the previous action's
  newly allocated destination slots;
- 158 histories had a proper subsequence producing the same final circuit;
- grouping by final circuit plus the multiset of xfer ids leaves only 15 groups,
  so 241 of the 256 retained paths differ only in the ordering/placement of
  essentially the same small rule collection.

The representative failure mode is:

```text
xfer 39: cx; rz -> rz; cx
xfer 38: rz; cx -> cx; rz   # consumes xfer 39's new destination: exact undo
... repeat on the same or another matching location ...
```

The rules are legal and sometimes useful: commuting an RZ through a CX can expose
a later merge or cancellation. The problem is that gate-count ranking assigns no
cost to making this move and immediately reversing it. Each rewrite allocates new
persistent slots, so raw deduplication sees a new state even when the exact circuit
has returned to an ancestor.

## Independent-action order is a separate source of duplicates

Independent action permutations must be distinguished from inverse cycles. For
every adjacent action pair in the exported histories, the analyzer now constructs
the swapped order, rebases the two destination-slot ranges, and asks Quartz to
apply it from the same prefix. The pair is called commutable only when both orders
are legal and their parameter/qubit-aware final circuit DAGs are identical.

For depth 3, beam 1000:

- 582 adjacent action occurrences are Quartz-proven commutable;
- 464 / 1000 histories contain at least one such pair;
- 313 occurrences, spanning 306 histories, are in the noncanonical order under
  the provisional concrete-action key;
- the retained beam contains 150 pairs of same-final histories with the exact
  same concrete-action multiset in a different order, involving 192 states;
- choosing one observed order representative per final-circuit/action-multiset
  group would remove 106 states.

The duplication is already visible after two actions. Among the 147 distinct
depth-2 prefixes of the final beam, 32 have a Quartz-proven commutable last pair.
Nine pairs, covering 18 states, contain both `AB` and `BA` in the retained set.

One real example is:

```text
A = xfer 38 on source slots (6, 7)
B = xfer 39 on source slots (8, 9)

AB destinations: A -> (39, 40), B -> (41, 42)
BA destinations: B -> (39, 40), A -> (41, 42)
```

Quartz accepts both orders and obtains the same circuit DAG, but the action
histories and slot-labelled topologies differ. Raw dedup therefore retains both.

For depth 8, 241 / 256 histories contain a commutable pair and 419 commutable
occurrences are in the provisional noncanonical order. Only 20 explicit `AB/BA`
pairs are both present in the final beam; this is a lower bound caused by beam
ranking retaining one permutation while other permutations may have been cut
earlier. The dominant direct inverse pair is not classified as independent,
because its second action consumes the first action's destination.

This confirms two distinct controls are needed:

1. inverse/ancestor-cycle pruning for dependent `A -> A^-1` backtracking;
2. partial-order reduction for independent `AB == BA` paths.

## Where the multiplicity starts

The broader depth-3, beam-1000 sample has 1,000 distinct serialized histories and
1,000 distinct slot-sensitive topologies, but only 320 circuit DAGs:

| Depth | Unique action prefixes | Slot-sensitive topologies | Circuit DAGs |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 1 |
| 1 | 11 | 11 | 11 |
| 2 | 147 | 147 | 106 |
| 3 | 1,000 | 1,000 | 320 |

At depth 3, the 680 duplicate excess paths split as follows:

- `1,000 - 680 = 320` repeat a canonical parent-to-child transition already
  represented through another slot/history representation;
- `680 - 320 = 360` are distinct canonical parent-to-child transitions that
  converge to an already represented child circuit.

There are 165 self-redundant depth-3 trajectories: 13 reduce to the root and 152
reduce to a one-action subsequence. Removing all of them still leaves 835 paths
and only 319 circuit DAGs, so inverse cycles are a large contributor but not the
only one. Different ECC paths and already-duplicated parents also converge.

An ordering over the exact same concrete action multiset could remove at most 106
of the 680 depth-3 duplicate excess paths observed here. Therefore action ordering
is useful, but global ordering alone is not sufficient.

## Quartz hash is not exact identity

The current refresh implementation uses `PyGraph.hash()`. Quartz's C++ source has
a `TODO: add constant parameters`; it also initializes every input-qubit node with
the same hash and sums all node hashes. It is a coarse transposition key, not a
collision-safe exact graph identity.

In the depth-3 sample:

- Quartz hash: 312 unique values;
- parameter/qubit-aware circuit DAG: 320 unique values;
- eight Quartz hash buckets each contain two structurally/qubit-distinct circuit
  DAGs;
- this sample has no collision caused by RZ parameter differences, although the
  implementation is generally exposed to that case because parameters are
  omitted.

Consequently, the previous `1000 -> 312` number consists of 680 confirmed
same-circuit duplicates plus at least eight false merges from the coarse Quartz
hash. A hash bucket must be followed by a collision-safe circuit comparison, and
that comparison must include wiring and parameters.

## Recommended controls

### 1. Cheap direct-inverse tabu

Precompute exact inverse xfer ids. Reject action `b` when it is the inverse of the
last action and `b.source_slots == last.destination_slots`. This is an O(1) test,
does not ban either direction globally, and directly targets all 926 depth-8
two-step cycles. A more general ancestor-state check should catch longer cycles.

### 2. Canonical state transposition table after every action

Use a key independent of persistent slot allocation and independent-operation
serialization. It must include physical input-qubit identity and RZ parameters.
Keep the fast raw key as a first-level filter, compute the stronger key only for
surviving/top-ranked proposals, and retain the exact refresh as validation.

The existing speculative `topology_digest()` is not yet suitable as the final
key: it lacks parameters and physical input labels, resolves canonicalization
ties using slot ids, and currently raises on some cyclic speculative candidates.
Those candidates should be rejected as invalid rather than aborting the search.

### 3. Partial-order reduction for independent actions

Do not require every action or xfer id to be globally increasing. That can remove
necessary dependent local sequences, including the Barenco reference trajectory.
Instead, canonicalize only commuting actions:

```text
append action b:
    scan backward over the maximal suffix whose actions are independent of b
    if one of those actions has a stable key greater than key(b): reject b
    otherwise keep b
```

This keeps the lexicographically minimal representative obtainable by swapping
adjacent independent actions. Independence should be conservative: two actions
must have disjoint source/destination lineage and disjoint rewrite boundary or
qubit-dependency footprints. The stable key should use physical qubits, canonical
topological position, and canonical rule id, not newly allocated slot ids.

### 4. Search scoring and diversity

- Penalize returning to a recent ancestor and repeated zero-delta commuting moves.
- Apply a small budget for zero-delta actions per parent unless they create a
  novel canonical state.
- Preserve proposals from distinct canonical parent states instead of allowing
  one reversible-rule family to fill the beam.
- Use adaptive early refresh when the online duplicate estimate spikes.

These are search-policy controls. Retraining the matcher is not the primary fix:
xfers 38/39 are real legal matches, and the failure is selecting legal but
non-progressing trajectories repeatedly.

## Safe rollout order

1. Replace the coarse Quartz-hash-only equality with a collision-safe exact key.
2. Add the direct-inverse tabu and run it in shadow mode on both saved high-quality
   GF and Barenco trajectories; it must reject zero reference actions.
3. Measure Barenco depth 8 with the tabu enabled.
4. Add parameter-aware per-step canonical transposition filtering.
5. Add conservative partial-order reduction and again replay both reference
   trajectories before enabling it by default.

Track unique circuit DAGs per beam, direct/ancestor cycle rejection, candidates
scanned, model/cache/replay time, peak cache pages, and best exact gate count. The
goal is to reduce duplicate work without losing either known high-quality path.
