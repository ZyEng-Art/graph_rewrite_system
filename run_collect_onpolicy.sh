#!/usr/bin/env bash
set -euo pipefail

root=/SharedData/dengzy/quarl_matchformer_fresh_20260902
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=/SharedData/dengzy/micromamba/envs/quarl-torch212/bin/python
export PYTHONPATH="$root/quarl/python"
export LD_LIBRARY_PATH="$root/quarl/build:${LD_LIBRARY_PATH:-}"

launch() {
    local stem="$1"
    nohup "$python_bin" "$workdir/collect_onpolicy_histories.py" \
        --histories "$root/runs/onpolicy_${stem}_b256_d4_histories.json" \
        --reference-data "$root/data/binding_longmix_16384_2048_v2.pt" \
        --ecc-file "$root/quarl/experiment/ecc_set/nam_ecc.json" \
        --max-histories-per-file 128 \
        --terminal-only \
        --output "$root/data/onpolicy_${stem}_b128_d4_labels.pt" \
        > "$root/runs/onpolicy_${stem}_b128_d4_collect.log" 2>&1 &
    printf '%s pid=%s\n' "$stem" "$!"
}

launch barenco_tof_3
launch mod5_4
launch tof_4
launch vbe_adder_3
