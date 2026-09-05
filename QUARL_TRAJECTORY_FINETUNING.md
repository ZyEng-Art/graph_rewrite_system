# Quarl trajectory hard-positive fine-tuning

> **Superseded audit note (2026-09-05):** the earlier action-rank tables in
> this file compared `xfer_id + anchor_slot` and can overstate full-action
> coverage when one anchor admits multiple bindings. The normalization-aware,
> full-binding audit and the successful 39→38 free-search result are documented
> in `NORMALIZED_TRAJECTORY_REPRODUCTION_20260905.md`.

Date: 2026-09-05

## Outcome

The original matcher does not fail because transfer `940` is absent from the
rule set or globally rare. It fails on a history-conditioned, locally created
binding in the held-out Barenco path `38_3`. The binding is created by the
first three forced rewrites, but the base matcher scores it below the candidate
threshold before the fourth rewrite.

Adding exact saved optimization paths and moderately upweighting the action
actually chosen by the trajectory fixes candidate coverage. With `38_3` held
out, the segmented Barenco model covers all 16 forced actions at target recall
0.95 and action cap 256. A real beam search from the original 58-gate circuit
still does not discover the optimization, so proposal selection and global
search remain a separate bottleneck.

## Baseline diagnosis

The held-out path begins at 39 gates rather than at the original 58-gate
Barenco circuit. Its first four actions are:

| Step | Cost before action | Transfer | Cost change |
| ---: | ---: | ---: | ---: |
| 0 | 39 | 40 | +2 |
| 1 | 41 | 896 | 0 |
| 2 | 41 | 4676 | -2 |
| 3 | 39 | 940 | +1 |

All four rewrites occur in a related local region. Transfer `940` at persistent
slot 2 is not available before step 0, becomes available after step 0, and is
still available when required after step 2. The base model nevertheless gives
its source/binding row raw logit `-16.737` (calibrated probability about
`0.000449`) and probability rank about 1636. It is therefore removed before
the later rollout stage can sample it.

The source pattern is:

```text
rz 0; cx 2 0; cx 2 1;
```

The original training corpus contains 80,204 positive bindings for this source
and about 13,176 on Barenco, so this is not ordinary source-class scarcity.
The exact local prefix `[40, 896, 4676]` is absent. This supports the hypothesis
that the missing feature is the distribution of consecutive local edits.

## Implementation

- `collect_quarl_trajectories.py` parses saved QASM paths, resolves each saved
  transition by exact successor hash, enumerates all exact matches, records
  persistent binding slots, and validates the incremental trajectory.
- `--split-unrepresentable` omits only an action whose combined Quartz rewrite
  and automatic RZ elimination cannot be represented by one incremental rule.
  It restarts an exact causal segment from the saved successor state, preserving
  valid data on both sides without inventing labels.
- `window_trajectory_dataset.py` splits histories longer than the model's
  64-action context into independently replayable exact windows.
- `PrefixDataset` and both collators now expose `target_action` /
  `target_actions`.
- `--action-positive-weight` gives extra classification weight to the exact
  source/binding row selected by the saved trajectory. It first verifies that
  the row is an exact positive match.
- `audit_quarl_trajectory_candidates.py` supports independent-state and causal
  sequence audits, gate/probability rankings, fixed source probes, and resetting
  a long forced sequence with `--sequence-window`.
- Mixed prefix-zero/nonempty training uncovered a pre-existing all-masked SDPA
  row that produced NaNs. `paged_model.py` now supplies a harmless key for the
  empty row and explicitly zeros its attention result.

## Converted data

`38_3` is excluded from every Barenco training set. The GF holdout is `370_2`.

| Dataset | Unique paths | Exact segments/windows | Unique action states | Repeated action states | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Initial Barenco conversion | 19 | 19 | 328 | 2,624 (8x) | Entire path skipped on an unrepresentable RZ transition |
| Segmented Barenco conversion | 51/51 | 65 | 884 | 7,072 (8x) | Only 35 transitions omitted; no path failures |
| Barenco + GF long-path conversion | 99 | 460 windows | 2,635 | 10,540 (4x) | 64-action windows; older strict Barenco conversion |

The segmented Barenco corpus contains ten distinct occurrences of transfer
`940` before repetition, compared with one in the initial strict conversion.
It also contains local streaks up to 11 consecutive related actions.

## H100 experiments

All models initialize from
`paged_action_onpolicy_v13_r8_lr5e5.pt`. Fine-tuning uses learning rate
`2e-5`, locality-positive weight 2, top-N boundary weight 1 and margin 1. The
default action-positive weight is 4 unless the row says otherwise.

| Model | Full-binding recall | Near recall | Non-near recall | Held-out `940` raw logit | `38_3`, recall 0.95 / cap 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 86.835% | 82.942% | 87.402% | -16.737 | 15/16 |
| Initial Barenco data, action weight 4 | 87.025% | 82.248% | 87.719% | -11.174 | 15/16 (16/16 at recall 0.98) |
| Initial Barenco data, action weight 16 | 87.016% | 82.275% | 87.706% | -12.424 | 15/16 |
| Segmented Barenco data, action weight 4 | 86.794% | 81.801% | 87.520% | **-6.924** | **16/16** |
| Barenco + GF windows, action weight 4 | **88.088%** | 82.512% | **88.899%** | -7.956 | 15/16 |

For the segmented Barenco model, the recalibrated far threshold at target
recall 0.95 is `-8.389`. The held-out `940` row is therefore retained. Its
unfiltered probability rank is 265, but after thresholding, structural binding
decode, action expansion and gate ordering, its action rank is 87. The complete
held-out path has maximum action rank 181 and is covered by cap 256; cap 128
covers 14/16.

Increasing the selected-action loss from 4 to 16 by itself does not help and is
not recommended. Expanding the relevant trajectory distribution is the
effective intervention. The all-path model has the best general held-out
metric, while the segmented Barenco model has the best Barenco local-prefix
coverage. The all-path model also reaches 16/16 at recall 0.99 and cap 256,
but that is a substantially more expensive threshold than the segmented
model's 16/16 at recall 0.95. A balanced combined dataset should retain both
advantages.

### Long GF holdout

The 271-action `gf2^6_mult/370_2` path is audited as five independent causal
windows of at most 64 actions, matching the training context. At target recall
0.95:

| Matcher | Source rows retained | Complete actions in Top512 |
| --- | ---: | ---: |
| Base | 253/271 (93.36%) | 215/271 (79.34%) |
| Barenco + GF model | **266/271 (98.15%)** | 189/271 (69.74%) |

The model learns the long-path source distribution, but the fixed action cap
becomes more congested: every additional retained source/binding expands into
one or more transfers. Thus source recall improves by 13 steps while complete
action Top512 coverage decreases. Five target source rows remain below the
95% thresholds; the other missing actions are mostly ranked beyond 512 after
expansion. For a 271-step path, even independent 98.15% per-step coverage would
give only about 0.64% probability that every step is present. Long-path
reproduction therefore needs near-perfect critical-action retention and an
action policy, not merely a good average matcher metric.

Expanding the all-path model's action cap gives 189/271 at 512, 238/271 at
1024, 254/271 at 2048, and 265/271 at 4096. The largest retained target rank is
3708. One already-recalled `+2` action is still below Top4096 because thousands
of lower-immediate-cost actions sort ahead of it, and five source rows remain
below the matcher threshold. Thus Top4096 nearly removes the action-cap loss,
but it cannot repair source misses and is too broad to substitute for a learned
long-horizon proposal policy.

## Actual search from the 58-gate circuit

Candidate coverage is necessary but not sufficient. The following exact-refresh
searches all finish at 58 gates:

| Matcher/search | Beam | Per-parent actions | Depth | Best exact | Time excluding audit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial hard model, gate order with best-root restart | 1,000 | 256 | 64 | 58 | 17.4 s |
| Initial hard model, stochastic order | 1,000 | 256 | 64 | 58 | 25.8 s |
| Segmented model, gate order | 1,000 | 256 | 64 | 58 | 22.0 s |
| Segmented model, probability order | 1,000 | 512 | 64 | 58 | 30.9 s |

The reason is now downstream of the matcher. Gate ordering globally prunes the
required temporary increases. Probability or random ordering preserves too
many unrelated branches and has no action-value signal telling it that the
specific local chain will later pay off. The saved `38_3` path also begins at a
39-gate graph, so the archive ancestry that first moves the original circuit
from 58 to that basin is not present in this path directory.

## Recommended next model/search change

1. Build the next combined corpus with segmented Barenco paths plus all GF
   windows, use a smaller Barenco repeat or a fixed general:hard batch ratio,
   and keep action-positive weight near 4.
2. Train the proposal policy/action-value head with behavior cloning on the
   saved `xfer_id + full binding`, not only matcher classification. The matcher
   probability says whether a source pattern matches; it does not distinguish
   which transfer sharing that source should be selected.
3. Add a return-to-go or eventual gate-reduction target so the first `+2` and
   fourth `+1` actions receive positive long-horizon value despite their
   immediate cost.
4. Preserve a Quartz-verified global graph archive across restarts, with a
   reserved quota for uphill/local-continuation proposals. A single globally
   gate-sorted beam repeatedly deletes the needed branch.
5. Re-run the 58-gate search with the behavior-cloned/value policy. The current
   matcher no longer blocks the known 39-to-38 local sequence, but it is not yet
   evidence that the whole 58-to-38 route is discoverable.

## Remote artifacts

The most useful remote files are:

```text
/SharedData/dengzy/quarl_matchformer_fresh_20260902/data/
  quarl_barenco_segments_holdout38_3_r8_20260905.pt
  binding_longmix_quarl_barenco_segments_holdout_r8_20260905.pt
  quarl_hard_paths_holdout_r4_w64_20260905.pt
  binding_longmix_quarl_all_holdout_r4_w64_20260905.pt

/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/
  paged_action_quarl_barenco_segments_holdout_r8_aw4_lw2_bw1_m1_lr2e5_s173.pt
  paged_action_quarl_barenco_segments_holdout_r8_aw4_lw2_bw1_m1_lr2e5_s173_calibration.json
  paged_action_quarl_all_holdout_r4_w64_aw4_lw2_bw1_m1_lr2e5_bs32_s274.pt
  paged_action_quarl_all_holdout_r4_w64_aw4_lw2_bw1_m1_lr2e5_bs32_s274_calibration.json
```

## Verification

- All converted trajectories and every generated 64-action window pass
  `validate_trajectory`.
- The selected-action loss tests cover emphasis of a weak chosen match and
  rejection of an inexact target row.
- Parser/discovery, windowing, causal audit summary and paged-model incremental
  tests pass in the remote PyTorch/Quartz environment.
- The mixed-prefix forward/backward smoke test has finite states, loss and
  gradients after the SDPA fix.
