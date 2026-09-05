#!/usr/bin/env bash
set -euo pipefail

root=/SharedData/dengzy/quarl_matchformer_fresh_20260902
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python
checkpoint=${VALUE_CHECKPOINT:-$root/runs/action_value_online_v1_iter1.pt}
output_dir=${OUTPUT_DIR:-$root/runs/topn_legality_audit}
depth=${DEPTH:-8}
value_increase_cap=${VALUE_INCREASE_CAP:-16}
output_suffix=${OUTPUT_SUFFIX:-}
export PYTHONPATH="$root/quarl/python"
export LD_LIBRARY_PATH="$root/quarl/build:${LD_LIBRARY_PATH:-}"

data="$root/data/binding_longmix_16384_2048_v2.pt"
calibration="$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1_calibration.json"
ecc="$root/quarl/experiment/ecc_set/nam_ecc.json"
circuit_dir="$root/quarl/experiment/circs/nam_circs"
circuits=(barenco_tof_3.qasm mod5_4.qasm tof_4.qasm vbe_adder_3.qasm)
gpus=(4 5 6 7)
if [[ -n "${CIRCUIT:-}" ]]; then
    circuits=("$CIRCUIT")
fi
mkdir -p "$output_dir"

rankings=(probability gate value)
if [[ -n "${RANKING:-}" ]]; then
    rankings=("$RANKING")
fi
for ranking in "${rankings[@]}"; do
    pids=()
    for index in "${!circuits[@]}"; do
        circuit=${circuits[$index]}
        stem=${circuit%.qasm}
        value_args=()
        if [[ "$ranking" == value ]]; then
            value_args=(
                --action-value-weight 0.25
                --value-increase-actions-per-parent "$value_increase_cap"
            )
        fi
        env CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$python_bin" \
            "$workdir/paged_rollout_benchmark.py" \
            --data "$data" \
            --checkpoint "$checkpoint" \
            --calibration "$calibration" \
            --target-recall 0.95 \
            --ecc-file "$ecc" \
            --qasm "$circuit_dir/$circuit" \
            --beam-size 1000 \
            --depth "$depth" \
            --microbatch 512 \
            --readout-attention-backend paged \
            --state-batch-backend tensorized \
            --proposal-backend gpu \
            --proposal-ranking "$ranking" \
            --proposal-ranking-seed 73 \
            --max-source-matches 2048 \
            --max-actions-per-parent 128 \
            --proposal-factor 16 \
            --max-gate-increase 3 \
            --lazy-topology-backend indexed \
            --dedup-mode raw \
            --refresh-interval 8 \
            --refresh-factor 2 \
            --audit-count 0 \
            --audit-proposal-topn 1,8,32,128,1000 \
            --output "$output_dir/${stem}_${ranking}_d${depth}${output_suffix}.json" \
            "${value_args[@]}" \
            > "$output_dir/${stem}_${ranking}_d${depth}${output_suffix}.log" 2>&1 &
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
