#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU_ID QASM_NAME DEPTH OUTPUT_TAG" >&2
  exit 2
fi

gpu_id=$1
qasm_name=$2
depth=$3
output_tag=$4
python_bin=/SharedData/dengzy/quarl_barenco_tof3_20260816_001809/.venv_torch212/bin/python

CUDA_VISIBLE_DEVICES="$gpu_id" \
PYTHONPATH=./quartz_exact_key/python \
LD_LIBRARY_PATH=./quartz_exact_key/build \
"$python_bin" beam_search_benchmark.py \
  --mode model \
  --data ../../data/binding_longmix_randomrefresh_complex_holdout_20260906.pt \
  --checkpoint ../../runs/hhop_h6_topo1_balanced_s907.pt \
  --calibration ../../runs/hhop_h6_topo1_balanced_s907_calibration_r999.json \
  --target-recall 0.999 \
  --ecc-file ../../quarl/experiment/ecc_set/nam_ecc.json \
  --qasm "../../quarl/experiment/circs/nam_circs/$qasm_name" \
  --beam-size 1000 \
  --depth "$depth" \
  --microbatch 512 \
  --max-source-matches 10240 \
  --max-actions-per-parent 128 \
  --proposal-factor 16 \
  --max-gate-increase 3 \
  --model-pipeline state_only_gpu \
  --model-apply-binding direct \
  --dedup-identity exact \
  --eliminate-rotation \
  --preapply-fingerprint off \
  --neural-audit-output "benchmark_results/${output_tag}.pt" \
  --output "benchmark_results/${output_tag}.json"
