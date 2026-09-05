# Hierarchical node-first matching A/B (H100, 2026-09-05)

## Scope

The benchmark compares candidate generation on identical root-state batches:

- baseline: all live nodes with the existing first-gate-grouped matcher;
- hierarchy: an untrained node actor selects Top-K nodes, then the matcher scores
  only first-gate-compatible source patterns and retains Top-16 per node;
- shared: paged graph readout, the same calibrated 95% threshold, and the exact
  vectorized structural binding decoder.

The node actor is intentionally untrained in this benchmark. These numbers
measure the compute path and memory scaling, not search quality or node recall.
All runs used batch 224, microbatch 224, five warmup iterations, 50 measured
iterations, bfloat16 autocast, and an NVIDIA H100 80GB HBM3 (PyTorch 2.4.0).

## Results

| Circuit | gates | Full states/s | K=4 states/s | K=8 states/s | K=16 states/s | Full peak GiB | K=8 peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `barenco_tof_3` | 58 | 17,725 | 19,526 | 18,027 | 17,773 | 0.290 | 0.182 |
| `barenco_tof_5` | 170 | 8,424 | 10,218 | 9,133 | 8,797 | 0.711 | 0.377 |
| `barenco_tof_10` | 450 | 3,476 | 4,025 | 4,014 | 3,839 | 1.761 | 0.867 |

At K=8, peak allocated memory falls by 37.4%, 47.0%, and 50.8% as the
circuit grows from 58 to 450 gates. Candidate-generation speedup is only 1.02x,
1.08x, and 1.15x because matcher work is no longer the dominant stage.

## Stage diagnosis

For the 450-gate circuit, the full baseline's profiled batch takes about
64.4ms:

| Stage | Time | Share |
|---|---:|---:|
| Python topology to tensor batch | 39.85ms | 61.8% |
| Incremental graph readout | 12.27ms | 19.0% |
| Candidate ranking | 5.00ms | 7.8% |
| Structural decode | 2.39ms | 3.7% |
| Full grouped matcher products | 1.87ms | 2.9% |

The node-first K=8 matcher product itself is only about 0.11ms, but it adds a
1.21ms node actor and cannot remove the 40ms topology collation or 12ms graph
readout. This means further matcher-only optimization cannot materially improve
end-to-end rollout until graph tensors are maintained incrementally with the
topology state.

## Quality warning

The random node head returns only 2 exact candidates per state on
`barenco_tof_5` at K=8 and none on `barenco_tof_10` at K=8. This is expected and
is consistent with the earlier held-out audit: conditional pattern selection is
high-recall once the correct node is retained, while node selection is the hard
decision. A return-aware node actor must be distilled or pretrained before this
path is used for PPO rollout.

Raw results:

- `hierarchical_matching_barenco_b224_h100.json`
- `hierarchical_matching_barenco_tof_5_b224_h100.json`
- `hierarchical_matching_barenco_tof_10_b224_h100.json`
