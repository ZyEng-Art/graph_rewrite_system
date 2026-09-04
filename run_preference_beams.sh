#!/usr/bin/env bash
set -euo pipefail

root=/SharedData/dengzy/quarl_matchformer_fresh_20260902
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python
export PYTHONPATH="$root/quarl/python"
export LD_LIBRARY_PATH="$root/quarl/build:${LD_LIBRARY_PATH:-}"

checkpoint="$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1.pt"
calibration="$root/runs/paged_action_onpolicy_v13_r8_lr5e5_epoch1_calibration.json"
exploration_checkpoint="$root/runs/paged_action_localgraph4_v5_cont_epoch2.pt"
exploration_calibration="$root/runs/paged_action_localgraph4_v5_cont_epoch2_calibration.json"
proposal_ranking=${PROPOSAL_RANKING:-probability}
proposal_seed=${PROPOSAL_SEED:-73}
run_stem=preference_train_${proposal_ranking}_s${proposal_seed}_b1000_d16_refresh8

launch() {
    local gpu="$1"
    local circuit="$2"
    local stem=${circuit%.qasm}
    local exploration_args=(
        --exploration-checkpoint "$exploration_checkpoint"
        --exploration-calibration "$exploration_calibration"
        --exploration-actions-per-parent 4
        --exploration-until-depth 1
    )
    if [[ "$proposal_ranking" == stochastic ]]; then
        exploration_args=()
    fi
    env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
        "$workdir/paged_rollout_benchmark.py" \
        --data "$root/data/binding_longmix_16384_2048_v2.pt" \
        --checkpoint "$checkpoint" \
        --calibration "$calibration" \
        "${exploration_args[@]}" \
        --target-recall 0.95 \
        --ecc-file "$root/quarl/experiment/ecc_set/nam_ecc.json" \
        --qasm "$root/quarl/experiment/circs/nam_circs/$circuit" \
        --beam-size 1000 \
        --depth 16 \
        --microbatch 512 \
        --page-size 8 \
        --readout-attention-backend paged \
        --state-batch-backend tensorized \
        --proposal-backend gpu \
        --proposal-ranking "$proposal_ranking" \
        --proposal-ranking-seed "$proposal_seed" \
        --lazy-topology-backend indexed \
        --refresh-interval 8 \
        --refresh-factor 2 \
        --dedup-mode raw \
        --audit-count 0 \
        --output "$root/runs/${run_stem}_${stem}.json" \
        --dump-beam-histories "$root/runs/${run_stem}_${stem}_histories.json" \
        > "$root/runs/${run_stem}_${stem}.log" 2>&1
}

pids=()
circuits=(barenco_tof_3.qasm mod5_4.qasm tof_4.qasm vbe_adder_3.qasm)
for index in "${!circuits[@]}"; do
    launch "$((index + 4))" "${circuits[$index]}" &
    pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'completed circuit=%s\n' "${circuits[$index]}"
    else
        printf 'failed circuit=%s\n' "${circuits[$index]}" >&2
        status=1
    fi
done
exit "$status"
