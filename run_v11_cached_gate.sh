#!/usr/bin/env bash
set -euo pipefail

root=/SharedData/dengzy/quarl_matchformer_fresh_20260902
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=/SharedData/dengzy/micromamba/envs/quarl-torch212/bin/python
name=paged_action_cachedgate_locality_v11

nohup env CUDA_VISIBLE_DEVICES=7 "$python_bin" "$workdir/train.py" \
    --data "$root/data/binding_longmix_16384_2048_v2.pt" \
    --output "$root/runs/$name.pt" \
    --epochs 3 --batch-size 128 --eval-batch-size 32 \
    --width 192 --retrieval-width 128 --graph-layers 2 \
    --architecture paged_action --action-layers 4 --action-heads 6 \
    --max-sequence-length 64 --readout-graph-layers 4 \
    --readout-graph-input cached_gate --readout-locality-features \
    --learning-rate 5e-5 --weight-decay 1e-2 --binding-weight 0 \
    --structural-hard-negatives --eval-every 1 \
    --init-checkpoint "$root/runs/paged_action_localgraph4_v5_cont_epoch2.pt" \
    --locality-positive-weight 1.0 \
    --selection-metric near_full_binding_topn \
    > "$root/runs/$name.log" 2>&1 &
printf '%s gpu=7 pid=%s\n' "$name" "$!"
