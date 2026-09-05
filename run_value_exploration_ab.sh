#!/usr/bin/env bash
set -euo pipefail

root=/SharedData/dengzy/quarl_matchformer_fresh_20260902
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python
checkpoint=${VALUE_CHECKPOINT:-$root/runs/action_value_online_v1_iter1.pt}
output_dir=${OUTPUT_DIR:-$root/runs/value_exploration_ab}
export PYTHONPATH="$root/quarl/python"
export LD_LIBRARY_PATH="$root/quarl/build:${LD_LIBRARY_PATH:-}"

data="$root/data/binding_longmix_16384_2048_v2.pt"
calibration="$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1_calibration.json"
ecc="$root/quarl/experiment/ecc_set/nam_ecc.json"
circuit_dir="$root/quarl/experiment/circs/nam_circs"
circuits=(barenco_tof_3.qasm mod5_4.qasm tof_4.qasm vbe_adder_3.qasm)
gpus=(4 5 6 7)
mkdir -p "$output_dir"

for fraction in 0.0 0.25; do
    label=${fraction/./p}
    pids=()
    for index in "${!circuits[@]}"; do
        circuit=${circuits[$index]}
        stem=${circuit%.qasm}
        seed=$((2235 + index * 97))
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
            --proposal-ranking value \
            --proposal-ranking-seed "$seed" \
            --action-value-weight 0.25 \
            --value-increase-actions-per-parent 16 \
            --value-exploration-fraction "$fraction" \
            --max-source-matches 2048 \
            --max-actions-per-parent 128 \
            --proposal-factor 16 \
            --max-gate-increase 3 \
            --lazy-topology-backend indexed \
            --refresh-interval 8 \
            --refresh-factor 2 \
            --dedup-mode raw \
            --audit-count 64 \
            --output "$output_dir/${stem}_mix_${label}.json" \
            --best-qasm "$output_dir/${stem}_mix_${label}_best.qasm" \
            > "$output_dir/${stem}_mix_${label}.log" 2>&1 &
        pids+=("$!")
    done
    status=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            printf 'completed fraction=%s circuit=%s\n' "$fraction" "${circuits[$index]}"
        else
            printf 'failed fraction=%s circuit=%s\n' "$fraction" "${circuits[$index]}" >&2
            status=1
        fi
    done
    (( status == 0 )) || exit "$status"
done
