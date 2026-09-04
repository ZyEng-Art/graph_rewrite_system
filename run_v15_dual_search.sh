#!/usr/bin/env bash
set -euo pipefail

root=${QUARL_MATCHFORMER_ROOT:-/SharedData/dengzy/quarl_matchformer_fresh_20260902}
work=$root/experiment/s0_binding_paged_worldmodel_20260903
python_bin=${QUARL_PYTHON:-/SharedData/dengzy/micromamba/envs/quarl-torch212/bin/python}
gpu_id=${GPU_ID:-4}

export PYTHONPATH=$root/quarl/python:$work${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$root/quarl/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

CUDA_VISIBLE_DEVICES=$gpu_id "$python_bin" "$work/paged_rollout_benchmark.py" \
  --data "$root/data/binding_longmix_onpolicy512_v3.pt" \
  --checkpoint "$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1.pt" \
  --calibration "$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1_calibration.json" \
  --exploration-checkpoint "$root/runs/paged_action_localgraph4_v5_cont_epoch2.pt" \
  --exploration-calibration "$root/runs/paged_action_localgraph4_v5_cont_epoch2_calibration.json" \
  --exploration-actions-per-parent 4 \
  --exploration-until-depth 1 \
  --target-recall 0.95 \
  --ecc-file "$root/quarl/experiment/ecc_set/nam_ecc.json" \
  --qasm "$root/quarl/experiment/circs/nam_circs/hwb6.qasm" \
  --beam-size 1000 \
  --depth 8 \
  --microbatch 512 \
  --page-size 8 \
  --cache-pages 4000 \
  --max-actions-per-parent 128 \
  --proposal-factor 16 \
  --max-gate-increase 1 \
  --dedup-mode raw \
  --audit-count 1000 \
  --output "$root/runs/paged_action_dual_v15_q4_until1.json" \
  --best-qasm "$root/runs/paged_action_dual_v15_q4_until1_best.qasm"
