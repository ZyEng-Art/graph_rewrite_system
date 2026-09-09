#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
    echo "usage: $0 QASM OUTPUT_DIR GPU [TARGET_GATE] [DEPTH] [APPLY_BUDGET]" >&2
    exit 2
fi

qasm=$(realpath "$1")
mkdir -p "$2"
output_dir=$(realpath "$2")
gpu=$3
target_gate=${4:-0}
depth=${5:-512}
apply_budget=${6:-100000}
python_bin=${QUARL_PYTHON:-/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python}
host=$(hostname)
workdir=$(cd "$(dirname "$0")" && pwd)

if [[ "$host" =~ (^|[-_])h?100[-_]?15$ || "$host" == "h100-15" ]]; then
    echo "refusing to run on excluded host h100-15" >&2
    exit 3
fi
if [[ ! -f "$qasm" ]]; then
    echo "missing input QASM: $qasm" >&2
    exit 4
fi

cd "$workdir"
export CUDA_VISIBLE_DEVICES=$gpu
export LD_LIBRARY_PATH=./quartz_exact_key/build
export OMP_NUM_THREADS=16
export PYTHONPATH=./quartz_exact_key/python

common=(
    --mode model
    --data ../../data/binding_longmix_randomrefresh_complex_holdout_20260906.pt
    --checkpoint ../../runs/hhop_h6_topo1_balanced_s907.pt
    --calibration ../../runs/hhop_h6_topo1_balanced_s907_calibration_r999.json
    --target-recall 0.999
    --ecc-file ../../quarl/experiment/ecc_set/nam_ecc.json
    --qasm "$qasm"
    --beam-size 256
    --microbatch 512
    --proposal-ranking gate
    --max-source-matches 10240
    --max-actions-per-parent 128
    --proposal-factor 16
    --max-gate-increase 3
    --model-pipeline state_only_gpu
    --model-apply-binding direct
    --dedup-identity exact
    --transactional-apply on
    --eliminate-rotation
    --survivor-policy gate
    --progressive-widening on
    --widening-policy feedback
    --widening-revisit-fraction 0.25
    --widening-max-expansions 8
    --widening-min-actions-per-parent 16
    --widening-seed 73
    --deterministic-search
    --max-total-attempted-actions "$apply_budget"
    --depth "$depth"
    --apply-profile off
)
if (( target_gate > 0 )); then
    common+=(--target-gate-count "$target_gate")
fi

run_one() {
    local name=$1
    local cache_mode=$2
    local started_ns
    local finished_ns
    started_ns=$(date +%s%N)
    "$python_bin" beam_search_benchmark.py \
        "${common[@]}" \
        --widening-candidate-cache "$cache_mode" \
        --output "$output_dir/$name.json" \
        --best-qasm "$output_dir/$name.best.qasm" \
        >"$output_dir/$name.log" 2>&1
    finished_ns=$(date +%s%N)
    awk -v start="$started_ns" -v finish="$finished_ns" \
        'BEGIN { printf "%.9f\n", (finish - start) / 1000000000 }' \
        >"$output_dir/$name.wall_seconds"
}

{
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'hostname=%s\n' "$host"
    printf 'cuda_visible_device=%s\n' "$gpu"
    printf 'python=%s\n' "$python_bin"
    printf 'qasm=%s\n' "$qasm"
    printf 'depth=%s\n' "$depth"
    printf 'apply_budget=%s\n' "$apply_budget"
    nvidia-smi --query-gpu=index,name,driver_version \
        --format=csv,noheader | sed -n "$((gpu + 1))p"
} >"$output_dir/environment.txt"

run_one cache_off_before off
run_one cache_on on
run_one cache_off_after off

"$python_bin" summarize_widening_candidate_cache_ab.py \
    --input-dir "$output_dir" --output "$output_dir/summary.json"
