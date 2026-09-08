#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 MANIFEST OUTPUT_DIR [GPUS] [SEEDS]" >&2
    exit 2
fi

manifest=$1
output_dir=$2
gpus=${3:-0,5,6,7}
seeds=${4:-73,170}
workdir=$(cd "$(dirname "$0")" && pwd)
python_bin=/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python
gpu_count=$(awk -F, '{print NF}' <<<"$gpus")

run_policy() {
    local policy=$1
    local config=$2
    "$python_bin" continuation_value_benchmark.py \
        --manifest "$manifest" \
        --runner-config "$config" \
        --runner-script beam_search_benchmark.py \
        --python-bin "$python_bin" \
        --workdir "$workdir" \
        --output-dir "$output_dir/$policy" \
        --budgets 128 \
        --budget-mode independent \
        --seeds "$seeds" \
        --gpus "$gpus" \
        --jobs "$gpu_count" \
        --resume
}

cd "$workdir"
run_policy gate continuation_runner_hhop_r999_equal_apply_gate.json
run_policy dual continuation_runner_hhop_r999_equal_apply_dual.json
"$python_bin" analyze_dual_lane_ab.py \
    "$output_dir/gate/summary.json" \
    "$output_dir/dual/summary.json" \
    --output "$output_dir/analysis.json"
