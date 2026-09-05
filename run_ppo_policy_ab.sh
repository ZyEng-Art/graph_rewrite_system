#!/usr/bin/env bash
set -euo pipefail

root=${QUARL_ROOT:-/SharedData/dengzy/quarl_matchformer_fresh_20260902}
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=${PYTHON_BIN:-/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python}
ppo_checkpoint=${PPO_CHECKPOINT:-$root/runs/paged_ppo_replay_v1.pt}
output_dir=${OUTPUT_DIR:-$root/runs/ppo_policy_ab}
export PYTHONPATH="$root/quarl/python"
export LD_LIBRARY_PATH="$root/quarl/build:${LD_LIBRARY_PATH:-}"

data="$root/data/binding_longmix_16384_2048_v2.pt"
checkpoint="$root/runs/paged_action_onpolicy_v13_r8_lr5e5.pt"
calibration="$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1_calibration.json"
ecc="$root/quarl/experiment/ecc_set/nam_ecc.json"
circuit_dir="$root/quarl/experiment/circs/nam_circs"
circuits=(
    barenco_tof_3.qasm
    mod5_4.qasm
    tof_4.qasm
    vbe_adder_3.qasm
    hwb6.qasm
    'gf2^4_mult.qasm'
)
gpus=(2 3 4 5 6 7)
mkdir -p "$output_dir"

for ranking in gate ppo; do
    pids=()
    for index in "${!circuits[@]}"; do
        circuit=${circuits[$index]}
        stem=${circuit%.qasm}
        extra=()
        if [[ "$ranking" == ppo ]]; then
            extra=(--ppo-checkpoint "$ppo_checkpoint")
        fi
        env CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$python_bin" \
            "$workdir/paged_rollout_benchmark.py" \
            --data "$data" \
            --checkpoint "$checkpoint" \
            --calibration "$calibration" \
            --ecc-file "$ecc" \
            --qasm "$circuit_dir/$circuit" \
            --beam-size 1000 \
            --depth 16 \
            --microbatch 512 \
            --state-batch-backend tensorized \
            --proposal-backend gpu \
            --proposal-ranking "$ranking" \
            "${extra[@]}" \
            --max-source-matches 2048 \
            --max-actions-per-parent 64 \
            --proposal-factor 16 \
            --max-gate-increase 3 \
            --lazy-topology-backend indexed \
            --refresh-interval 8 \
            --refresh-factor 2 \
            --dedup-mode raw \
            --audit-count 64 \
            --output "$output_dir/${stem}_${ranking}.json" \
            --best-qasm "$output_dir/${stem}_${ranking}_best.qasm" \
            > "$output_dir/${stem}_${ranking}.log" 2>&1 &
        pids+=("$!")
    done
    status=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            printf 'completed ranking=%s circuit=%s\n' "$ranking" "${circuits[$index]}"
        else
            printf 'failed ranking=%s circuit=%s\n' "$ranking" "${circuits[$index]}" >&2
            status=1
        fi
    done
    (( status == 0 )) || exit "$status"
done
