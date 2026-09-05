#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 SNAPSHOT CIRCUIT GPU OUTPUT_DIR [SEED]" >&2
  exit 2
fi

snapshot="$1"
circuit="$2"
circuit_label="${circuit//^/_}"
gpu="$3"
output_dir="$4"
seed="${5:-99001}"
quarl_root="${QUARL_ROOT:-/SharedData/dengzy/Quarl}"
python_bin="${QUARL_PYTHON:-/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python}"
checkpoint="${QUARL_CHECKPOINT:-${quarl_root}/experiment/ppo-new/outputs/h100_nam_pretrain_6small_cluster_20260816_183726/ckpts/iter_576.pt}"
qasm="${quarl_root}/experiment/circs/nam_circs/${circuit}.qasm"
profile_output="${output_dir}/rollout_profile.jsonl"

mkdir -p "${output_dir}"
rm -f "${profile_output}"
profile_qasm="${output_dir}/${circuit_label}.qasm"
ln -sfn "${qasm}" "${profile_qasm}"
export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export LD_PRELOAD="${QUARL_GOMP:-/SharedData/lintc/miniforge3/envs/torch212/lib/libgomp.so.1.0.0}"
export LD_LIBRARY_PATH="${QUARL_LIB:-/SharedData/dengzy/usr_local/lib}:${LD_LIBRARY_PATH:-}"
export QUARL_ROLLOUT_PROFILE_OUTPUT="${profile_output}"
export QUARL_ROLLOUT_PROFILE_SYNC_CUDA="${QUARL_ROLLOUT_PROFILE_SYNC_CUDA:-1}"

cd "${quarl_root}/experiment/ppo-new"
"${python_bin}" "${snapshot}/ppo.py" c=nam_ft \
  c.resume=true \
  c.resume_optimizer=false \
  c.ckpt_path="${checkpoint}" \
  c.gpus='[0]' \
  c.ddp_port="${QUARL_DDP_PORT:-$((34000 + gpu * 100 + seed % 97))}" \
  c.seed="${seed}" \
  c.agent_collect=true \
  c.obs_per_agent=0 \
  c.wandb.en=false \
  c.max_iterations=578 \
  c.k_epochs=1 \
  c.lr_gnn=0 \
  c.lr_actor=0 \
  c.lr_critic=0 \
  c.lr_scheduler=none \
  c.mini_batch_size=4800 \
  c.num_eps_per_iter=64 \
  c.agent_batch_size=64 \
  c.dyn_eps_len=false \
  c.min_eps_len=20 \
  c.max_eps_len=20 \
  c.gnn_num_layers=6 \
  c.num_gate_types=40 \
  "c.input_graphs=[{name: ${circuit_label}, path: ${profile_qasm}}]" \
  "hydra.run.dir=${output_dir}" \
  2>&1 | tee "${output_dir}/run.log"
