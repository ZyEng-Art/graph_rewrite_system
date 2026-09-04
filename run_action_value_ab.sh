#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 CIRCUIT.qasm [DEPTH]" >&2
    exit 2
fi

root=/SharedData/dengzy/quarl_matchformer_fresh_20260902
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python
export PYTHONPATH="$root/quarl/python"
export LD_LIBRARY_PATH="$root/quarl/build:${LD_LIBRARY_PATH:-}"

circuit="$1"
depth=${2:-16}
circuit_stem=${circuit%.qasm}
base_checkpoint="$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1.pt"
value_checkpoint="$root/runs/action_value_d16_support2_v2_lr5e5.pt"
calibration="$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1_calibration.json"
exploration_checkpoint="$root/runs/paged_action_localgraph4_v5_cont_epoch2.pt"
exploration_calibration="$root/runs/paged_action_localgraph4_v5_cont_epoch2_calibration.json"
labels=(gate value_w010 value_w025 value_w050 value_w100)
rankings=(gate value value value value)
weights=(0 0.10 0.25 0.50 1.00)
gpus=(3 4 5 6 7)
pids=()

for index in "${!labels[@]}"; do
    checkpoint="$value_checkpoint"
    if [[ "${rankings[$index]}" == gate ]]; then
        checkpoint="$base_checkpoint"
    fi
    output_stem="action_value_v2_ab_${circuit_stem}_b1000_d${depth}_${labels[$index]}"
    value_args=()
    if [[ "${rankings[$index]}" == value ]]; then
        value_args=(--action-value-weight "${weights[$index]}")
    fi
    env CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$python_bin" \
        "$workdir/paged_rollout_benchmark.py" \
        --data "$root/data/binding_longmix_16384_2048_v2.pt" \
        --checkpoint "$checkpoint" \
        --calibration "$calibration" \
        --exploration-checkpoint "$exploration_checkpoint" \
        --exploration-calibration "$exploration_calibration" \
        --exploration-actions-per-parent 4 \
        --exploration-until-depth 1 \
        --target-recall 0.95 \
        --ecc-file "$root/quarl/experiment/ecc_set/nam_ecc.json" \
        --qasm "$root/quarl/experiment/circs/nam_circs/$circuit" \
        --beam-size 1000 \
        --depth "$depth" \
        --microbatch 512 \
        --page-size 8 \
        --readout-attention-backend paged \
        --state-batch-backend tensorized \
        --proposal-backend gpu \
        --proposal-ranking "${rankings[$index]}" \
        "${value_args[@]}" \
        --lazy-topology-backend indexed \
        --refresh-interval 8 \
        --refresh-factor 2 \
        --dedup-mode raw \
        --audit-count 64 \
        --output "$root/runs/${output_stem}.json" \
        --best-qasm "$root/runs/${output_stem}_best.qasm" \
        > "$root/runs/${output_stem}.log" 2>&1 &
    pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'completed label=%s\n' "${labels[$index]}"
    else
        printf 'failed label=%s\n' "${labels[$index]}" >&2
        status=1
    fi
done
exit "$status"
