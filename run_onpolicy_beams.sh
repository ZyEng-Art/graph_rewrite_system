#!/usr/bin/env bash
set -euo pipefail

root=/SharedData/dengzy/quarl_matchformer_fresh_20260902
workdir="$root/experiment/s0_binding_paged_worldmodel_20260903"
python_bin=/SharedData/dengzy/micromamba/envs/quarl-torch212/bin/python
export PYTHONPATH="$root/quarl/python"
export LD_LIBRARY_PATH="$root/quarl/build:${LD_LIBRARY_PATH:-}"

launch() {
    local gpu="$1"
    local circuit="$2"
    local stem=${circuit%.qasm}
    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
        "$workdir/paged_rollout_benchmark.py" \
        --data "$root/data/binding_longmix_16384_2048_v2.pt" \
        --checkpoint "$root/runs/paged_action_localityreadout_v9_local1.pt" \
        --calibration "$root/runs/paged_action_localityreadout_v9_local1_calibration.json" \
        --target-recall 0.95 --near-target-recall 0.80 --far-target-recall 0.95 \
        --ecc-file "$root/quarl/experiment/ecc_set/nam_ecc.json" \
        --qasm "$root/quarl/experiment/circs/nam_circs/$circuit" \
        --beam-size 256 --depth 4 --microbatch 256 --page-size 4 \
        --dedup-mode raw --audit-count 0 \
        --output "$root/runs/onpolicy_${stem}_b256_d4.json" \
        --dump-beam-histories "$root/runs/onpolicy_${stem}_b256_d4_histories.json" \
        > "$root/runs/onpolicy_${stem}_b256_d4.log" 2>&1 &
    printf '%s gpu=%s pid=%s\n' "$circuit" "$gpu" "$!"
}

launch 4 barenco_tof_3.qasm
launch 5 mod5_4.qasm
launch 6 tof_4.qasm
launch 7 vbe_adder_3.qasm
