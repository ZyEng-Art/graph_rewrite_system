#!/usr/bin/env bash
set -euo pipefail

root=/SharedData/dengzy/quarl_matchformer_fresh_20260902
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=/SharedData/dengzy/micromamba/envs/quarl-torch212/bin/python
data="$root/data/binding_longmix_onpolicy512_v3.pt"
initial="$root/runs/paged_action_localgraph4_v5_cont_epoch2.pt"

launch() {
    local gpu="$1"
    local name="$2"
    local repeat="$3"
    local learning_rate="$4"
    shift 4
    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$workdir/train.py" \
        --data "$data" --output "$root/runs/$name.pt" \
        --epochs 3 --batch-size 128 --eval-batch-size 32 \
        --width 192 --retrieval-width 128 --graph-layers 2 \
        --architecture paged_action --action-layers 4 --action-heads 6 \
        --max-sequence-length 64 --readout-graph-layers 4 \
        --readout-graph-input cached --learning-rate "$learning_rate" \
        --weight-decay 1e-2 --binding-weight 0 \
        --structural-hard-negatives --eval-every 1 \
        --init-checkpoint "$initial" --include-train-terminal \
        --train-terminal-repeat "$repeat" "$@" \
        > "$root/runs/$name.log" 2>&1 &
    printf '%s gpu=%s pid=%s\n' "$name" "$gpu" "$!"
}

launch 4 paged_action_onpolicy_v13_r1_lr5e5 1 5e-5
launch 5 paged_action_onpolicy_v13_r8_lr5e5 8 5e-5
launch 6 paged_action_onpolicy_locality_v13_r8_lr5e5 8 5e-5 \
    --readout-locality-features
launch 7 paged_action_onpolicy_v13_r8_lr2e5 8 2e-5
