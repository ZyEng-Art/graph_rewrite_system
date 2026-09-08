# Graph Rewrite System: paged `s0` + causal-action world model

This repository predicts complete Quartz rewrite matches from an initial circuit
`s0` and an ordered action prefix.  Its beam-search hot path uses paged causal
K/V state, GPU candidate/proposal processing, and an indexed lightweight graph;
Quartz graph matching and copy/apply are reserved for exact refresh and audit.

The implementation was developed independently from the earlier state-only
prototype.  Hashes of the frozen files used to establish that baseline boundary
are recorded in `BASELINE_SHA256.txt`.

## Current snapshot

- fixed held-out complete-binding Top-N recall: **90.17%** with strict N output;
- `hwb6`, beam 1000, depth 64: **29.412 s**, 259 -> 253 gates;
- `hwb6`, beam 1000, depth 16 with exact refresh every 8 actions:
  **37.178 s**;
- same-circuit depth-3 speedup over CPU Quartz: **24.5x--125.8x** across the
  measured cross-circuit suite;
- depth-8 cross-circuit audit: **1280/1280** trajectories Quartz-valid and
  topology-exact on `vbe_adder_3`, `gf2^4_mult`, `gf2^5_mult`, `adder_8`, and
  `grover_5`;
- 2026-09-06 batch-512 rerun: H100 full-proposal throughput is **319.37x**
  CPU Quartz on GF `370_2` and **129.62x** on Barenco `38_3`;
- current long-path optimization: Barenco reproduces 39 -> 38, while GF
  completes 271 actions but remains 371 -> 371.
- raw-QASM exact-apply optimization: Barenco reaches 58 -> 38 in **78.87 s**
  versus **741.38 s** for same-host 32-thread CPU Quartz (**9.40x**), with the
  same final QASM; the model implementation is **2.04x** faster than its prior
  161.23 s path to 38.

See `PAGED_RESULTS.md` for the final stage profiles and cross-circuit results,
`RESULTS.md` for the earlier model/data experiments, and
`QUARL_TRAJECTORY_FINETUNING.md` for the held-out Barenco/GF trajectory
hard-positive experiment.  The two recommended checkpoints and their
calibration files are under `benchmark_results/`.
The current CPU/H100 rerun, full Barenco/GF searches, and their limitations are
reported in
`benchmark_results/cpu_quartz_rerun_and_e2e_optimization_findings_20260906.md`.

## Repository layout

- model and training: `model.py`, `paged_model.py`, `train.py`, `dataset.py`;
- rollout: `paged_rollout_benchmark.py`, `lazy_rollout_benchmark.py`;
- GPU runtime: `paged_attention.py`, `paged_cache.py`, `gpu_proposals.py`,
  `tensorized_batch.py`;
- evaluation and profiling: `evaluate_*.py`, `*_benchmark.py`;
- regression tests: `test_*.py`;
- checkpoints, calibration, QASM outputs, and machine-readable measurements:
  `benchmark_results/`.
- recommended checkpoint and summary checksums: `ARTIFACT_SHA256.txt`.

The Python environment requires PyTorch, NumPy, and Triton; Quartz itself must
be built separately.  The measured environment used Python 3.12 and PyTorch
2.1.2 on Linux/H100.  Dataset files and the Quartz checkout are intentionally
external because the training dataset and compiled Quartz tree are not part of
this repository.

## Exact state-only search

`beam_search_benchmark.py` now defaults model mode to
`--model-pipeline state_only_gpu`.  This path does not serialize or replay an
action prefix in the neural model.  At every layer it:

1. densely collates each live, exact Quartz graph;
2. predicts and structurally decodes source bindings from that graph only;
3. expands source bindings to rewrite actions and applies the parent/global
   Top-K caps on GPU;
4. copies only selected actions to the host;
5. applies each complete binding through Quartz, deduplicates the exact
   successor, and encodes that successor in the next layer.

The old `initial graph + zero-length action tensors + Python proposal` path is
retained as `--model-pipeline compat_host` for controlled A/B tests.  The
state-only path deliberately rejects periodic CPU match refreshes: exact
Quartz *rewrite and identity checking* still occur for every selected action,
but candidate matching stays model-only so the measured boundary is clear.

On H100 with beam 1000 and microbatch 512, controlled depth-8 runs preserve
the exact best trace/QASM and improve end-to-end time by 1.81x on Barenco and
2.27x on GF.  In long runs, Barenco reaches the same 38-gate circuit at the
same step in 47.44 seconds rather than 88.07 seconds, and GF reaches a
strictly better 473-gate result in 579.42 seconds versus the previous
474-gate result in 1865.97 seconds (3.22x faster despite five more layers).
See `benchmark_results/state_only_exact_rewrite_findings_20260907.md` for the
Barenco/GF quality and throughput comparison.

## Corrected baseline boundary

The high-accuracy checkpoint used by the previous lazy rollout,
`locality_ft_feat_w1.pt`, has `state_only=True`. Its graph is derived exactly
from `s0 + actions`, but the neural forward consumes that materialized light
graph rather than the action prefix. The old rollout measurements remain valid;
they must not be described as causal-sequence/KV-cache measurements.

The new `paged_action` core instead:

- encodes the initial `s0` graph once;
- represents each rewrite as a causal action token;
- reconstructs live slots only from ordered source/destination bindings;
- incrementally updates cached node states without replaying the prefix;
- appends one immutable K/V entry per causal layer and action;
- never constructs, copies, or applies a Quartz graph in the ordinary hot path.

The pure action-only v2 reaches 79.36% direct Top-N. The recommended v5 adds
four message-passing layers over the lightweight port graph already maintained
for structural decoding. This graph is derived deterministically from the
actions and costs no Quartz match/copy/apply. The hybrid reaches 90.17% while
emitting exactly N structurally valid complete bindings after retrieving 5N
cheap neural candidates. The faster 2N policy reaches 90.10% recall but emits
99.866% of N because 255 states do not contain N valid candidates in the first
2N.

`PagedKVCache` stores several actions per GPU page. Full pages are shared by
beam states with a common prefix, the partial tail uses copy-on-write, and
reference counts reclaim pages after beam pruning. The current PyTorch prototype
gathers pages into contiguous tensors; a fused paged-attention kernel can later
consume the same block tables directly.

Core files:

- `paged_model.py`: causal action model and full-prefix/one-token APIs;
- `paged_cache.py`: GPU page arena, COW prefix sharing, gather and reclamation;
- `paged_rollout_benchmark.py`: lazy beam search with per-beam paged state;
- `tensorized_batch.py`: bulk graph/locality collation that reuses GPU state;
- `gpu_proposals.py`: GPU source-to-xfer expansion and bounded proposal ranking;
- `prefix_cache_ab_benchmark.py`: same-checkpoint full-prefix versus paged A/B;
- `collect_onpolicy_histories.py`: exact labels for model-generated trajectories;
- `interpolate_checkpoints.py`: compatible-checkpoint ablation utility;
- `test_paged_model.py`: full-prefix versus incremental numerical equivalence;
- `test_paged_cache.py`: page sharing/COW/reclamation invariants.

## Recommended artifact

- local checkpoint: `benchmark_results/paged_action_localgraph4_v5_cont_epoch2.pt`;
- strict-N metrics: `benchmark_results/paged_action_localgraph4_v5_cont_epoch2_x5.json`;
- faster 2N metrics: `benchmark_results/paged_action_localgraph4_v5_cont_epoch2_x2.json`;
- remote checkpoint: `/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/paged_action_localgraph4_v5_cont_epoch2.pt`;
- SHA-256: `7c1a0da00bc186523f0daa5c1a035481eeae65b86470c1224d2ff9c9298e6da9`.

The held-out set has 2,048 states and 2,090,005 exact complete bindings. Strict
5N-to-N recall is 90.17%; see `PAGED_RESULTS.md` for long-prefix, locality,
threshold-calibration, rollout, and ablation results.

## Recommended search policy

For maximum fixed-set matching recall, keep the v5 checkpoint above as the
default: it is the measured strict-N 90.17% checkpoint. For long no-refresh
search, use the v15 two-route policy:

- primary: `benchmark_results/paged_action_onpolicy_v13_r8_lr5e5_epoch1.pt`;
- exploration: the v5 checkpoint above;
- reserve four exploration proposals per parent at depth 1 only.

On `hwb6` with beam 1000 and depth 8 this policy takes 28.56 seconds, finds
259 -> 253 gates, and has 962/1000 exact-valid final trajectories. The primary
alone is 1000/1000 valid but finds 255 gates; v5 alone finds 253 gates but is
only 890/1000 valid. On independent `gf2^4_mult`, beam 256 and depth 8, v15 is
256/256 valid and produces 225 -> 219 gates in 6.53 seconds.

For trajectories longer than eight actions, enable exact recovery every eight
actions. On `hwb6`, depth 16 with `--refresh-interval 8 --refresh-factor 2`
takes 106.18 seconds and finishes 1000/1000 exact-valid at 253 gates. Without
recovery it takes 66.21 seconds but is only 823/1000 valid; the older four-step
recovery policy takes 138.77 seconds.

Exact refreshes now deduplicate by a complete per-physical-qubit operation
trace rather than `Graph.hash()`. The identity is invariant to serialization
order of independent actions while retaining gate parameters, operand roles,
physical wiring, and dependent order. A native Quartz patch is provided at
`quartz_patches/exact_graph_key.patch`; without that patch the rollout uses a
correct but slower QASM fallback. The Barenco/GF duplicate diagnosis, H100 A/B,
reference-trajectory safety audit, and rebuild instructions are in
`benchmark_results/exact_identity_dedup_findings_20260906.md`.

The exact-apply beam search also uses three circuit-independent hot-path
optimizations. It selects the stable bounded top-k before allocating Python
`Proposal` objects, applies a model-provided complete binding directly through
Quartz GUIDs when the patched API is available, and checks exact identity
before constructing snapshots, slot maps, distance maps, and history for a
child. The last change is especially important when independent rewrite orders
converge to the same circuit: at the Barenco 38-gate stopping point, 85.4% of
validly applied successors were exact duplicates, so all of their child
metadata is now skipped. These changes do not suppress or approximate any
rewrite and contain no circuit-specific rules. `--model-apply-binding anchor`
retains the original rematching path for controlled A/B tests, while `auto`
falls back to it when the direct-binding Quartz patch is unavailable. Results,
component timings, cross-circuit checks, and patch requirements are in
`benchmark_results/exact_apply_hotpath_optimization_findings_20260906.md`.

For history-conditioned models, `--refresh-dedup-scope level` removes exact
duplicates inside each refreshed beam without permanently rejecting a circuit
that reappears at a later depth. `--rebase-model-history-at-refresh` permits a
search to exceed a checkpoint's learned action-context limit by re-encoding
each Quartz-verified current circuit as a new model root; the physical state,
full replay history, and exact checkpoint are retained. The GF depth-271
benchmark uses both options and charges all re-encoding time to the search.

A strict same-checkpoint cache A/B on the actual 1000-state `hwb6` beams
confirms why short searches looked similar. With both paths at microbatch 512
and inference under `torch.no_grad()`, depth 8 is effectively tied (paged is
0.96x after normalizing their common matching cost). At depth 16/32/64,
normalized prediction-path speedups are 1.04x, 1.11x, and 1.14x; the
prefix-state computation alone reaches 1.64x at depth 64. Matching and
complete-binding structural decode remain the dominant cost, so paged prefix
reuse is useful for long histories but is not by itself a large end-to-end
acceleration. See `PAGED_RESULTS.md` for the exact protocol, accuracy agreement,
memory, and page-sharing results.

The optimized runtime now defaults to vectorized page-table gather/COW and
SDPA readout. On the same depth-64 workload, the paged prefix path drops from
21.09 to 6.31 seconds and is 3.35x faster than full-prefix replay. In the real
64-step search, accumulated cache advance drops from 15.90 to 4.65 seconds and
total search time drops from 331.11 to 308.22 seconds, while retaining the same
253-gate best result. The optional `sdpa_live` backend is numerically valid but
slower at this shape, so plain `sdpa` remains the default. Setting
`--readout-attention-backend eager` preserves the previous depth-8 Quartz audit
exactly and still reduces the depth-64 search to 315.28 seconds.

`--readout-attention-backend paged` now enables a true Triton block-table
attention path. Both the causal action decoder and node-to-action readout load
physical cache pages directly; no contiguous KV/action history is gathered.
Each action's readout K/V projection is stored once beside its causal KV, and
shared prefixes reuse those pages with the same COW ownership rules. On the
fixed depth-64 beam, direct paging reduces readout+advance from 6.315 to 6.047
seconds (1.044x over the vectorized-gather backend) and raises the prefix versus
full-recompute speedup from 3.35x to 3.50x. It is still experimental rather than
the default because the complete search is 310.60 seconds versus 308.22 seconds
for SDPA; matching/binding decode and CPU proposal/hash work dominate.

Increasing the microbatch does not fix that bottleneck. At depth 8, real-search
times for microbatch 256/512/1000 are 28.26/28.59/29.15 seconds. An isolated
`B=256/512/1000`, `T=64`, `N=381` sweep shows the 1000-state readout workload
already plateaus by microbatch 512, while causal append still benefits modestly
from 1000. Keep 256-512 for this workload unless the downstream candidate
decoder is also moved off its per-state/Python path.

Train:

```bash
CUDA_VISIBLE_DEVICES=4 python train.py \
  --architecture paged_action \
  --data ../../data/binding_longmix_16384_2048_v2.pt \
  --output ../../runs/paged_action_recurrent_v2.pt \
  --epochs 5 --batch-size 128 --eval-batch-size 32 \
  --width 192 --retrieval-width 128 --graph-layers 2 \
  --action-layers 4 --action-heads 6 --max-sequence-length 64 \
  --learning-rate 2e-4 --binding-weight 0 --structural-hard-negatives \
  --init-checkpoint ../../runs/s0_binding_4096_discrete.pt
```

Fine-tune the ordered-binding v3 without changing the v2 checkpoint contract:

```bash
CUDA_VISIBLE_DEVICES=5 python train.py \
  --architecture paged_action --ordered-binding-roles \
  --data ../../data/binding_longmix_16384_2048_v2.pt \
  --output ../../runs/paged_action_ordered_v3.pt \
  --epochs 5 --batch-size 128 --eval-batch-size 32 \
  --width 192 --retrieval-width 128 --graph-layers 2 \
  --action-layers 4 --action-heads 6 --max-sequence-length 64 \
  --learning-rate 1e-4 --binding-weight 0 --structural-hard-negatives \
  --init-checkpoint ../../runs/paged_action_recurrent_v2_epoch4.pt
```

The optional v4 readout consumes only the lightweight graph derived from the
action prefix; it does not invoke Quartz or materialize/copy a Quartz graph.
Two layers give an explicit two-hop topology correction while retaining the
same causal KV/node cache:

```bash
CUDA_VISIBLE_DEVICES=4 python train.py \
  --architecture paged_action --readout-graph-layers 2 \
  --data ../../data/binding_longmix_16384_2048_v2.pt \
  --output ../../runs/paged_action_localgraph2_v4.pt \
  --epochs 5 --batch-size 128 --eval-batch-size 32 \
  --width 192 --retrieval-width 128 --graph-layers 2 \
  --action-layers 4 --action-heads 6 --max-sequence-length 64 \
  --learning-rate 2e-4 --binding-weight 0 --structural-hard-negatives \
  --init-checkpoint ../../runs/paged_action_recurrent_v2.pt
```

Use clean live gate tokens as the topology-branch input (v6) while retaining
the action/KV branch as the residual state:

```bash
CUDA_VISIBLE_DEVICES=4 python train.py \
  --architecture paged_action --readout-graph-layers 4 \
  --readout-graph-input gate \
  --data ../../data/binding_longmix_16384_2048_v2.pt \
  --output ../../runs/paged_action_gategraph4_v6.pt \
  --epochs 5 --batch-size 128 --eval-batch-size 32 \
  --width 192 --retrieval-width 128 --graph-layers 2 \
  --action-layers 4 --action-heads 6 --max-sequence-length 64 \
  --learning-rate 2e-4 --binding-weight 0 --structural-hard-negatives \
  --init-checkpoint ../../runs/paged_action_localgraph4_v5_epoch2.pt
```

Paged search needs the Quartz checkout's current Python binding only for
periodic replay/audit:

```bash
export PYTHONPATH=../../quarl/python
export LD_LIBRARY_PATH=../../quarl/build:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=5 python paged_rollout_benchmark.py \
  --data ../../data/binding_longmix_16384_2048_v2.pt \
  --checkpoint ../../runs/paged_action_localgraph4_v5_cont_epoch2.pt \
  --calibration ../../runs/paged_action_localgraph4_v5_cont_epoch2_calibration.json \
  --target-recall 0.95 --ecc-file ../../quarl/experiment/ecc_set/nam_ecc.json \
  --qasm ../../quarl/experiment/circs/nam_circs/hwb6.qasm \
  --beam-size 1000 --depth 16 --microbatch 512 --page-size 8 \
  --refresh-interval 4 --refresh-factor 2 --dedup-mode raw \
  --audit-count 1000 --output ../../runs/paged_hwb6_d16_refresh4.json
```

The optimized v15 search keeps the legacy implementations available but opts
into the tensorized state, GPU proposal, and indexed-topology paths explicitly:

```bash
export PYTHONPATH=../../quarl/python
export LD_LIBRARY_PATH=../../quarl/build:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=4 python paged_rollout_benchmark.py \
  --data ../../data/binding_longmix_16384_2048_v2.pt \
  --checkpoint ../../runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1.pt \
  --calibration ../../runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1_calibration.json \
  --exploration-checkpoint ../../runs/paged_action_localgraph4_v5_cont_epoch2.pt \
  --exploration-calibration ../../runs/paged_action_localgraph4_v5_cont_epoch2_calibration.json \
  --exploration-actions-per-parent 4 --exploration-until-depth 1 \
  --target-recall 0.95 --ecc-file ../../quarl/experiment/ecc_set/nam_ecc.json \
  --qasm ../../quarl/experiment/circs/nam_circs/hwb6.qasm \
  --beam-size 1000 --depth 16 --microbatch 512 --page-size 8 \
  --readout-attention-backend paged --state-batch-backend tensorized \
  --proposal-backend gpu --lazy-topology-backend indexed --dedup-mode raw \
  --refresh-interval 8 --refresh-factor 2 --audit-count 0 \
  --output ../../runs/paged_hwb6_d16_refresh8_optimized.json
```

Measured on `h100-gpu1` GPU4, the optimized depth-64 no-refresh search is
29.412 seconds (253 best gates), and depth 16 with refresh every eight actions
is 37.178 seconds.  See `PAGED_RESULTS.md` for the full stage breakdown and
Quartz correctness audits.

The final path was additionally audited on `vbe_adder_3`, `gf2^4_mult`,
`gf2^5_mult`, `adder_8`, and `grover_5`: all 1,280 beam-256 depth-8 outputs are
Quartz-valid and topology-exact.  Same-circuit depth-3 speedups over CPU Quartz
range from 24.5x to 125.8x, so the `hwb6` 83.8x result should not be treated as
a universal constant.  Full results are in
`benchmark_results/cross_circuit_final_summary.json`.

## Fixed-budget continuation value

`build_continuation_manifest.py` and `continuation_value_benchmark.py` measure
whether a short search probe can identify circuits that deserve a larger depth
budget. The experiment records immutable QASM digests, exact fixed-budget
outcomes, duplicate/invalid/unique-successor rates, complete runner logs, and
top-subset lift over random selection. The state-only beam runner also supports
reproducible stochastic proposal ranking for repeated outcomes while preserving
the historical gate-ranking default. See
`CONTINUATION_VALUE_BENCHMARK.md` for the H100 workflow and interpretation
constraints. The first logged pilot and its artifact digests are recorded in
`CONTINUATION_VALUE_PILOT_20260908.md`.

## Legacy copied documentation

This directory is an independent implementation of the corrected task:

- input: the complete initial circuit `s0` and a prefix of actions;
- every action contains the rewrite id and the ordered source-node binding;
- output: every applicable source pattern and its complete ordered node binding.

It does not read or import any of the earlier anchor-only matcher code. Quartz is
used only to generate exact supervision and to validate source/destination node
identity at rewrite time.

The implementation combines an LLM-style action-conditioned graph memory with a
discrete in-place circuit state. The retrieval head predicts `(anchor, source)`;
a vectorized port-following decoder recovers the complete binding in source-pattern
order. See `RESULTS.md` for the held-out H100 result and reproduction commands.

The rollout prototype in `beam_search_benchmark.py` implements calibrated,
oracle-N-free match prediction, source-to-xfer expansion, gate-delta ranking,
a bounded best-circuit buffer, exact Quartz application/validation, graph-hash
deduplication, and optional periodic exact-match refresh. Its `quartz` mode uses
Quartz's original `available_xfers_parallel` action enumeration as the baseline.

The same benchmark can avoid many duplicate Quartz copies/applies with the
optional native successor fingerprint.  After the existing exact-key,
direct-binding and wire-profile Quartz patches, apply
`quartz_patches/native_successor_fingerprint.patch`, rebuild Quartz and use
`--preapply-fingerprint filter --preapply-fingerprint-kind xfer_guarded`.
The backend defaults to `auto`: patched Quartz uses the compact C++/Cython
batch path, while an unpatched build falls back to the Python reference path.
See `NATIVE_SUCCESSOR_FINGERPRINT_RESULTS_20260907.md` for the shadow safety
audit and H100 end-to-end results.

`lazy_rollout_benchmark.py` removes Quartz graph copy/apply from ordinary rollout.
It appends the predicted action, allocates destination slots deterministically,
and updates only a lightweight gate/port graph needed by the current structural
decoder. Quartz is invoked out of band to replay and audit shortlisted sequences.
On `hwb6`, full 1000-state audits at depths 32 and 64 replayed successfully with
100% lazy/exact topology agreement; see `RESULTS.md` for timing and graph-hash
diversity results.
