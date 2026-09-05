#!/usr/bin/env bash
set -euo pipefail

root=${QUARL_ROOT:-/SharedData/dengzy/quarl_matchformer_fresh_20260902}
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=${PYTHON_BIN:-/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python}
output=${PPO_OUTPUT:-$root/runs/paged_ppo_replay_v1.pt}
export PYTHONPATH="$root/quarl/python"
export LD_LIBRARY_PATH="$root/quarl/build:${LD_LIBRARY_PATH:-}"

data="$root/data/binding_longmix_16384_2048_v2.pt"
checkpoint="$root/runs/paged_action_onpolicy_v13_r8_lr5e5.pt"
calibration="$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1_calibration.json"
ecc="$root/quarl/experiment/ecc_set/nam_ecc.json"
circuit_dir="$root/quarl/experiment/circs/nam_circs"

"$python_bin" "$workdir/train_paged_ppo.py" \
    --data "$data" \
    --checkpoint "$checkpoint" \
    --calibration "$calibration" \
    --ecc-file "$ecc" \
    --qasm \
        "$circuit_dir/barenco_tof_3.qasm" \
        "$circuit_dir/mod5_4.qasm" \
        "$circuit_dir/tof_4.qasm" \
        "$circuit_dir/vbe_adder_3.qasm" \
    --output "$output" \
    --iterations "${PPO_ITERATIONS:-15}" \
    --episodes-per-iteration "${PPO_EPISODES:-96}" \
    --evaluation-episodes-per-circuit 1 \
    --max-steps "${PPO_MAX_STEPS:-16}" \
    --max-source-matches 2048 \
    --max-actions "${PPO_MAX_ACTIONS:-64}" \
    --collector-batch-size "${PPO_COLLECTOR_BATCH_SIZE:-64}" \
    --refresh-interval "${PPO_REFRESH_INTERVAL:-8}" \
    --actor-critic "${PPO_ACTOR_CRITIC:-match_set}" \
    --set-layers "${PPO_SET_LAYERS:-2}" \
    --set-heads "${PPO_SET_HEADS:-4}" \
    --initial-gate-bias "${PPO_GATE_BIAS:-1.0}" \
    --entropy-coefficient "${PPO_ENTROPY_COEFFICIENT:-0.02}" \
    --replay-start-probability "${PPO_REPLAY_START_PROBABILITY:-0.75}" \
    --replay-capacity-per-circuit "${PPO_REPLAY_CAPACITY:-256}" \
    --ppo-epochs 4 \
    --minibatch-size 128 \
    --seed "${PPO_SEED:-173}"
