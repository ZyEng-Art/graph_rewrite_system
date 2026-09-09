#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
    echo "usage: $0 QASM_LIST OUTPUT_DIR GPU [APPLY_BUDGET] [DEPTH] [POLICY]" >&2
    exit 2
fi

list=$(realpath "$1")
mkdir -p "$2"
output_dir=$(realpath "$2")
gpu=$3
apply_budget=${4:-20000}
depth=${5:-256}
policy=${6:-feedback}
if [[ "$policy" != "feedback" && "$policy" != "feedback_balanced" ]]; then
    echo "policy must be feedback or feedback_balanced" >&2
    exit 6
fi
python_bin=${QUARL_PYTHON:-/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python}
qasm_root=${QASM_ROOT:-/SharedData/dengzy/quarl_matchformer_fresh_20260902/data/fullseq_36_0_forward}
host=$(hostname)
workdir=$(cd "$(dirname "$0")" && pwd)

if [[ "$host" =~ (^|[-_])h?100[-_]?15$ || "$host" == "h100-15" ]]; then
    echo "refusing to run on excluded host h100-15" >&2
    exit 3
fi
if [[ ! -f "$list" ]]; then
    echo "missing QASM list: $list" >&2
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
    --widening-policy "$policy"
    --widening-revisit-fraction 0.25
    --widening-max-expansions 8
    --widening-min-actions-per-parent 16
    --widening-seed 73
    --deterministic-search
    --max-total-attempted-actions "$apply_budget"
    --depth "$depth"
    --apply-profile off
    --widening-candidate-cache off
    --widening-action-cache off
    --neural-descendant-labels on
)

{
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'hostname=%s\n' "$host"
    printf 'cuda_visible_device=%s\n' "$gpu"
    printf 'python=%s\n' "$python_bin"
    printf 'qasm_list=%s\n' "$list"
    printf 'qasm_root=%s\n' "$qasm_root"
    printf 'depth=%s\n' "$depth"
    printf 'apply_budget=%s\n' "$apply_budget"
    printf 'widening_policy=%s\n' "$policy"
    nvidia-smi --query-gpu=index,name,driver_version \
        --format=csv,noheader | sed -n "$((gpu + 1))p"
} >"$output_dir/environment.txt"

while IFS= read -r entry || [[ -n "$entry" ]]; do
    entry=${entry%%#*}
    entry=${entry%$'\r'}
    [[ -z "${entry//[[:space:]]/}" ]] && continue
    if [[ "$entry" = /* ]]; then
        qasm=$entry
    else
        qasm="$qasm_root/$entry"
    fi
    if [[ ! -f "$qasm" ]]; then
        echo "missing input QASM: $qasm" >&2
        exit 5
    fi
    stem=$(basename "$qasm" .qasm)
    audit="$output_dir/${stem}_apply${apply_budget}_audit.pt"
    if [[ -f "$audit" ]]; then
        echo "SKIP existing $stem"
        continue
    fi
    started_ns=$(date +%s%N)
    "$python_bin" beam_search_benchmark.py \
        "${common[@]}" \
        --qasm "$qasm" \
        --neural-audit-output "$audit" \
        --output "$output_dir/${stem}_apply${apply_budget}.json" \
        >"$output_dir/${stem}_apply${apply_budget}.log" 2>&1
    finished_ns=$(date +%s%N)
    awk -v start="$started_ns" -v finish="$finished_ns" \
        'BEGIN { printf "%.9f\n", (finish - start) / 1000000000 }' \
        >"$output_dir/${stem}_apply${apply_budget}.wall_seconds"
    echo "DONE $stem"
done <"$list"
