#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
    echo "usage: $0 MANIFEST OUTPUT_DIR [GPUS] [BUDGETS] [SEEDS]" >&2
    exit 2
fi

manifest=$1
output_dir=$2
gpus=${3:-0,5,6,7}
budgets=${4:-4,16,64}
seeds=${5:-73,170}
workdir=$(cd "$(dirname "$0")" && pwd)
python_bin=/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python
gpu_count=$(awk -F, '{print NF}' <<<"$gpus")

cd "$workdir"
"$python_bin" continuation_value_benchmark.py \
    --manifest "$manifest" \
    --runner-config continuation_runner_hhop_r999.json \
    --runner-script beam_search_benchmark.py \
    --python-bin "$python_bin" \
    --workdir "$workdir" \
    --output-dir "$output_dir" \
    --budgets "$budgets" \
    --budget-mode nested \
    --seeds "$seeds" \
    --gpus "$gpus" \
    --jobs "$gpu_count" \
    --resume
