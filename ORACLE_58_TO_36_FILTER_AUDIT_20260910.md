# 58-to-36 oracle path and filtering audit (2026-09-10)

## Outcome

The saved `fullseq_36_0_forward` directory is a valid 116-action Quartz
trajectory from 58 gates to 36 gates. It is not merely a collection of
independent 58-gate inputs: all 116 transitions replay exactly to the next saved
QASM, with no unrepresentable or skipped transition.

The selected H-hop matcher is not the source of the loss. It retains the exact
source binding for all 116 oracle actions at the deployed R99.9 threshold and
the 10,240 source-match cap. Every oracle rewrite is also available in Quartz
and reproduces the saved successor.

The path is lost by the downstream gate-first search policy:

1. In a beam-256, proposal-factor-16 fixed-beam retention run, oracle states 1
   and 2 are retained. The third oracle successor is absent before survivor
   selection. The recorded exclusion stage is `gpu_match_or_proposal_cap`.
2. The independent candidate audit proves that the step-2 oracle action is
   matcher-covered and has per-parent gate rank 34, below the 128-action parent
   cap. Therefore this first loss is specifically the 4,096-action global
   proposal truncation across 115 live parents, not matcher thresholding,
   Quartz apply, exact deduplication, or survivor selection.
3. Even if that early global loss were avoided, eight later required actions
   have per-parent gate ranks 133--264 and are outside the current rank-128
   first expansion. All eight temporarily increase gate count. Gate-first
   ranking therefore suppresses precisely the detours used by the oracle path.

This is direct evidence for a branch-selection failure, rather than a matcher
recall failure.

## Path shape

The filename schema is
`STEP_CURRENT_COST_REWARD_NODE_XFER.qasm`, where the next cost is
`current_cost - reward`. The terminal file is
`116_36_0_0_0.qasm`.

The 116 actions consist of:

| Reward | Meaning | Count |
|---:|---|---:|
| `+2` | remove two gates | 11 |
| `+1` | remove one gate | 18 |
| `0` | equal-cost rewrite | 74 |
| `-1` | add one gate | 8 |
| `-2` | add two gates | 5 |

The gross reductions remove 40 gates, while the required detours add 18 gates,
for a net reduction of 22 gates. Only 29/116 actions improve gate count
immediately; 87/116 do not. The first strictly better costs are reached at:

```text
step 0: 58    step 7: 56    step 17: 54   step 21: 53
step 23: 51   step 24: 50   step 25: 48   step 28: 46
step 54: 45   step 56: 43   step 57: 42   step 58: 41
step 63: 39   step 86: 38   step 113: 37  terminal: 36
```

This includes a 26-step interval before improving from 46 to 45, a 23-step
interval before improving from 39 to 38, and a 27-step interval before
improving from 38 to 37. A policy based primarily on immediate gate count is
not aligned with this trajectory.

## Exact replay and matcher coverage

Strict conversion accepted one path containing all 116 actions:

```text
accepted paths:                  1 / 1
accepted trajectory segments:   1
exact representable actions:    116 / 116
omitted transitions:            0
```

The action audit reports:

| Check | Result |
|---|---:|
| Quartz action available | 116/116 |
| Exact successor hash reproduced | 116/116 |
| Matcher source binding retained | 116/116 |
| Expanded oracle action ranked | 116/116 |
| Median per-parent gate rank | 12 |
| P90 rank | 74 |
| P95 rank | 154 |
| Maximum rank | 264 |

Coverage under local action caps is:

| Per-parent cap | Covered | Whole path covered |
|---:|---:|:---:|
| 16 | 69/116 (59.48%) | no |
| 128 | 108/116 (93.10%) | no |
| 256 | 114/116 (98.28%) | no |
| 512 | 116/116 (100%) | yes |

The eight actions beyond rank 128 are:

| Step | Transition | Xfer | Per-parent rank |
|---:|---:|---:|---:|
| 34 | 46 -> 47 | 1492 | 133 |
| 38 | 46 -> 47 | 1482 | 154 |
| 66 | 39 -> 41 | 42 | 208 |
| 69 | 40 -> 42 | 101 | 264 |
| 79 | 41 -> 43 | 36 | 247 |
| 86 | 38 -> 40 | 40 | 193 |
| 90 | 39 -> 40 | 1594 | 135 |
| 105 | 39 -> 41 | 42 | 258 |

Raising the cap to 256 is still insufficient because steps 69 and 105 rank
264 and 258. A cap of 512 covers this particular path, but that is an oracle
diagnostic rather than evidence that a blanket cap increase is a general or
efficient search solution.

## First actual beam loss

The fixed-beam reference retention run used the production matcher/search
settings where applicable: beam 256, gate ranking, 10,240 source matches,
128 actions per parent, proposal factor 16, exact identity deduplication,
transactional direct-binding apply, and rotation elimination.

| Search layer | Oracle parent in beam | Oracle successor generated | Retained |
|---:|:---:|:---:|:---:|
| 1 | yes, index 0 | yes, candidate 14 | yes, beam index 14 |
| 2 | yes, index 14 | yes, candidate 116 | yes, beam index 116 |
| 3 | yes, index 116 | no | no |

At layer 3 there are 115 input parents. The search chooses at most 4,096 global
proposals and scans 829 before filling the 256-child beam. No proposal capable
of creating the expected oracle hash is present in the globally selected
proposal list. The independent audit for this exact action (trajectory step 2,
`58 -> 58`, xfer 120) gives local rank 34. This isolates the first failure to
cross-parent global proposal competition.

The reference-retention facility intentionally rejects progressive widening,
because widening can revisit a state without advancing one oracle action per
search layer. Consequently, this run diagnoses the fixed gate-first pipeline's
first loss exactly. The separate candidate audit diagnoses the later rank-band
requirements that progressive widening would have to recover. It does not claim
that a particular widening seed loses the oracle at exactly layer 3.

## Implication for the next search change

The next change should target branch survival and deferred detour selection,
not retrain the matcher or merely lower its threshold. The evidence defines two
separate requirements:

1. Reserve a bounded per-parent/global lane so a locally plausible equal-cost
   continuation is not erased by competition from all other parents. This is
   needed as early as trajectory step 2.
2. Revisit surviving parents deeply enough to expose ranks beyond 128, while
   admitting selected `+1/+2`-gate actions. For this oracle, ranks through 264
   and detours of up to two gates are necessary.

Using the known path to force these choices would leak the answer and would not
be general. The path should instead be used as a positive-control audit: any
generic continuation score or allocation rule can be checked for whether it
promotes these actions, then accepted only if it also improves frozen
cross-circuit search under the same apply budget.

## Artifacts

- `benchmark_results/oracle_58_to_36_reference_20260910.pt`: converted exact
  reference trajectory used for hash-retention checks.
- `benchmark_results/oracle_58_to_36_candidate_audit_20260910.json`: full
  116-action Quartz/matcher/rank audit.
- `benchmark_results/oracle_58_to_36_fixed_loss_20260910.json`: first-loss beam
  retention result.
- `benchmark_results/oracle_58_to_36_fixed_loss_20260910.best.qasm`: unchanged
  58-gate best graph from the deliberately stopped-on-loss run.

SHA-256:

```text
EC61431F8FF45376A8341E08803BCCA227E644A5155AF63829F666F3CD71C402  oracle_58_to_36_candidate_audit_20260910.json
35650B0278F83837246B5006667417EC9E8DACC39199EE89C01CFA16B43894E9  oracle_58_to_36_fixed_loss_20260910.json
0DCB094E39BC516D7F428405E56AD8ED0842825876308E381FDA8AA56F3A1FC4  oracle_58_to_36_fixed_loss_20260910.best.qasm
FDCFEA034422C128B1DD620CE59770DB6AD56EAF055C5CBDC77E769C0F001260  oracle_58_to_36_reference_20260910.pt
```

## Reproduction

Convert and strictly verify the saved path:

```bash
python collect_quarl_trajectories.py \
  --trajectory-root ../../data/fullseq_36_0_forward \
  --reference-data ../../data/binding_longmix_randomrefresh_complex_holdout_20260906.pt \
  --ecc-file ../../quarl/experiment/ecc_set/nam_ecc.json \
  --output benchmark_results/oracle_58_to_36_reference_20260910.pt \
  --strict
```

Audit exact actions and their local gate ranks:

```bash
python audit_quarl_trajectory_candidates.py \
  --data ../../data/binding_longmix_randomrefresh_complex_holdout_20260906.pt \
  --checkpoint ../../runs/hhop_h6_topo1_balanced_s907.pt \
  --calibration ../../runs/hhop_h6_topo1_balanced_s907_calibration_r999.json \
  --target-recall 0.999 \
  --ecc-file ../../quarl/experiment/ecc_set/nam_ecc.json \
  --trajectory ../../data/fullseq_36_0_forward \
  --output benchmark_results/oracle_58_to_36_candidate_audit_20260910.json \
  --max-source-matches 10240 \
  --max-action-rank 8192 \
  --caps 16,128,256,512,1024,2048,4096,8192 \
  --microbatch 512 \
  --max-gate-increase 3 \
  --ranking-mode gate
```

The exact fixed-beam command is recorded in the `config`-equivalent top-level
fields of `oracle_58_to_36_fixed_loss_20260910.json`; it additionally used
`--reference-data ... --stop-on-reference-loss` and requested depth 116.

The deployed checkpoint is `state_only`, so it has no causal action-prefix
input. A sequence-conditioned audit is therefore neither supported nor needed
for this matcher.
