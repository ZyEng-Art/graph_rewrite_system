#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 ITERATION [ROUNDS]" >&2
    exit 2
fi

iteration=$1
rounds=${2:-3}
root=/SharedData/dengzy/quarl_matchformer_fresh_20260902
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python
export PYTHONPATH="$root/quarl/python"
export LD_LIBRARY_PATH="$root/quarl/build:${LD_LIBRARY_PATH:-}"

base_value_checkpoint="$root/runs/action_value_d16_support2_v2_lr5e5.pt"
if (( iteration == 0 )); then
    input_checkpoint="$base_value_checkpoint"
    proposal_ranking=stochastic
    value_exploration_fraction=0.0
else
    previous=$((iteration - 1))
    input_checkpoint="$root/runs/action_value_online_v1_iter${previous}.pt"
    proposal_ranking=value
    value_exploration_fraction=0.25
fi
if [[ ! -f "$input_checkpoint" ]]; then
    echo "missing input checkpoint: $input_checkpoint" >&2
    exit 1
fi

calibration="$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1_calibration.json"
data="$root/data/binding_longmix_16384_2048_v2.pt"
ecc="$root/quarl/experiment/ecc_set/nam_ecc.json"
circuit_dir="$root/quarl/experiment/circs/nam_circs"
run_root="$root/runs/accelerated_online_v1"
iteration_dir="$run_root/iterations/iter${iteration}"
mkdir -p "$iteration_dir"

circuits=(barenco_tof_3.qasm mod5_4.qasm tof_4.qasm vbe_adder_3.qasm)
gpus=(4 5 6 7)
pids=()
active_circuits=()
completion_markers=()
archives=()
for index in "${!circuits[@]}"; do
    circuit=${circuits[$index]}
    stem=${circuit%.qasm}
    output_dir="$run_root/$stem"
    completion_marker="$iteration_dir/${stem}.rollout.complete"
    archives+=("$output_dir/archive.json")
    if [[ -f "$completion_marker" ]]; then
        printf 'skipped completed circuit=%s\n' "$circuit"
        continue
    fi
    resume_args=()
    if [[ -f "$output_dir/archive.json" ]]; then
        resume_args=(--resume)
    fi
    env CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$python_bin" \
        "$workdir/accelerated_self_improve.py" \
        --python "$python_bin" \
        --data "$data" \
        --checkpoint "$input_checkpoint" \
        --calibration "$calibration" \
        --ecc-file "$ecc" \
        --qasm "$circuit_dir/$circuit" \
        --output-dir "$output_dir" \
        --rounds "$rounds" \
        --beam-size 1000 \
        --depth 16 \
        --microbatch 512 \
        --proposal-ranking "$proposal_ranking" \
        --proposal-ranking-seed "$((73 + iteration * 1009 + index * 97))" \
        --action-value-weight 0.25 \
        --value-increase-actions-per-parent 16 \
        --value-exploration-fraction "$value_exploration_fraction" \
        --max-gate-increase 3 \
        --refresh-interval 8 \
        --refresh-factor 2 \
        --audit-count 64 \
        "${resume_args[@]}" \
        > "$iteration_dir/${stem}_driver.log" 2>&1 &
    pids+=("$!")
    active_circuits+=("$circuit")
    completion_markers+=("$completion_marker")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        touch "${completion_markers[$index]}"
        printf 'completed circuit=%s\n' "${active_circuits[$index]}"
    else
        printf 'failed circuit=%s\n' "${active_circuits[$index]}" >&2
        status=1
    fi
done
if (( status != 0 )); then
    exit "$status"
fi

train_histories=(
    "$root/runs/preference_train_stochastic_s73_b1000_d16_refresh8_barenco_tof_3_histories.json"
    "$root/runs/preference_train_stochastic_s73_b1000_d16_refresh8_mod5_4_histories.json"
    "$root/runs/preference_train_stochastic_s73_b1000_d16_refresh8_tof_4_histories.json"
    "$root/runs/preference_train_stochastic_s73_b1000_d16_refresh8_vbe_adder_3_histories.json"
    "$root/runs/preference_train_stochastic_s74_b1000_d16_refresh8_barenco_tof_3_histories.json"
    "$root/runs/preference_train_stochastic_s74_b1000_d16_refresh8_mod5_4_histories.json"
    "$root/runs/preference_train_stochastic_s74_b1000_d16_refresh8_tof_4_histories.json"
    "$root/runs/preference_train_stochastic_s74_b1000_d16_refresh8_vbe_adder_3_histories.json"
)
validation_histories=(
    "$root/runs/preference_train_stochastic_s75_b1000_d16_refresh8_barenco_tof_3_histories.json"
    "$root/runs/preference_train_stochastic_s75_b1000_d16_refresh8_mod5_4_histories.json"
    "$root/runs/preference_train_stochastic_s75_b1000_d16_refresh8_tof_4_histories.json"
    "$root/runs/preference_train_stochastic_s75_b1000_d16_refresh8_vbe_adder_3_histories.json"
)
preference_data="$root/data/action_preferences_online_v1_iter${iteration}.pt"
"$python_bin" "$workdir/collect_action_preferences.py" \
    --histories "${train_histories[@]}" \
    --archives "${archives[@]}" \
    --validation-histories "${validation_histories[@]}" \
    --reference-data "$data" \
    --output "$preference_data" \
    --min-remaining-depth 3 \
    --min-descendants 2 \
    --objective best-prefix-residual \
    > "$iteration_dir/preference_collection.log" 2>&1

output_checkpoint="$root/runs/action_value_online_v1_iter${iteration}.pt"
"$python_bin" "$workdir/train_action_preferences.py" \
    --data "$preference_data" \
    --init-checkpoint "$input_checkpoint" \
    --output "$output_checkpoint" \
    --epochs 20 \
    --batch-size 128 \
    --encode-batch-size 128 \
    --learning-rate 2e-5 \
    --weight-decay 1e-2 \
    --score-l2 1e-3 \
    --advantage-weight-power 0.5 \
    --seed "$((73 + iteration))" \
    --eval-every 1 \
    > "$iteration_dir/training.log" 2>&1

printf 'iteration=%s ranking=%s checkpoint=%s\n' \
    "$iteration" "$proposal_ranking" "$output_checkpoint"
