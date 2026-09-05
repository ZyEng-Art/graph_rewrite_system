# Original Quarl rollout profile by circuit size

## Protocol

These are one-iteration H100 runs through original Quarl's exact `agent_collect` path.
Every run uses 64 episodes, a fixed horizon of 20, batch size 64, 1,280
transitions, the Nam `iter_576.pt` checkpoint, and zero learning rates. The
profiled snapshot changes timing only; action selection, Quartz application, PPO
state construction, graph-buffer insertion, and episode restart are unchanged.
CUDA is synchronized at neural stage boundaries so asynchronous work is charged
to GNN, critic, actor, or sampling rather than a later `.cpu()` call.
An idle vLLM process retained about 47.9 GiB on every H100 but reported 0% GPU
utilization before these sequential runs; peak CUDA below is Quarl's own allocation.

## Scale

| circuit | gates | rollout | transitions/s | rollout/iteration | peak CUDA | buffer after |
|---|---:|---:|---:|---:|---:|---:|
| `barenco_tof_3` | 58 | 5.034s | 254.26 | 89.8% | 69.2 MiB | 211 |
| `vbe_adder_3` | 150 | 6.188s | 206.86 | 84.5% | 102.7 MiB | 261 |
| `hwb6` | 259 | 6.889s | 185.81 | 92.4% | 141.3 MiB | 237 |
| `grover_5` | 831 | 10.754s | 119.03 | 93.9% | 324.5 MiB | 1059 |
| `gf2_16_mult` | 3435 | 39.859s | 32.11 | 98.1% | 1273.1 MiB | 19 |

## Grouped rollout share

All percentages are mutually exclusive and include a measured residual, so each
row sums to 100%.

| circuit | setup | graph input | neural node | xfer legal/sample | Quartz apply/reward | PPO subgraphs | buffer/restart | finalize/residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `barenco_tof_3` | 4.7% | 6.4% | 13.8% | 20.2% | 1.2% | 44.3% | 8.8% | 0.5% |
| `vbe_adder_3` | 3.9% | 5.6% | 19.2% | 17.7% | 2.4% | 41.2% | 9.5% | 0.5% |
| `hwb6` | 3.4% | 9.3% | 16.8% | 17.9% | 10.0% | 35.0% | 6.7% | 0.8% |
| `grover_5` | 2.3% | 11.4% | 13.3% | 8.7% | 28.9% | 25.5% | 9.1% | 0.7% |
| `gf2_16_mult` | 0.6% | 13.9% | 5.6% | 19.8% | 39.2% | 9.7% | 10.8% | 0.3% |

## Detailed rollout share

| stage | 58g | 150g | 259g | 831g | 3435g |
|---|---:|---:|---:|---:|---:|
| `inference.graph_to_dgl` | 5.8% | 4.9% | 8.3% | 10.7% | 13.5% |
| `inference.dgl_batch_h2d` | 0.6% | 0.7% | 1.0% | 0.7% | 0.4% |
| `inference.gnn` | 11.9% | 17.7% | 15.5% | 12.4% | 5.3% |
| `inference.critic` | 0.7% | 0.5% | 0.4% | 0.3% | 0.1% |
| `inference.node_sampling` | 1.1% | 1.0% | 0.9% | 0.6% | 0.2% |
| `inference.actor` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `inference.available_xfers` | 18.1% | 16.1% | 16.5% | 7.8% | 19.6% |
| `inference.xfer_sampling_transfer` | 2.1% | 1.5% | 1.4% | 0.9% | 0.2% |
| `environment.apply_xfer` | 1.2% | 2.4% | 9.9% | 28.3% | 38.0% |
| `experience.next_state` | 24.7% | 24.7% | 19.2% | 13.5% | 6.9% |
| `experience.current_state` | 19.5% | 16.4% | 15.7% | 12.0% | 2.8% |
| `buffer.update_and_best` | 1.8% | 3.7% | 4.3% | 8.7% | 5.9% |
| `environment.restart_or_advance` | 7.1% | 5.8% | 2.5% | 0.4% | 4.8% |

## Milliseconds per transition

| stage | 58g | 150g | 259g | 831g | 3435g |
|---|---:|---:|---:|---:|---:|
| `inference.graph_to_dgl` | 0.228 | 0.238 | 0.447 | 0.899 | 4.212 |
| `inference.gnn` | 0.470 | 0.854 | 0.832 | 1.046 | 1.651 |
| `inference.available_xfers` | 0.712 | 0.779 | 0.886 | 0.657 | 6.091 |
| `environment.apply_xfer` | 0.047 | 0.115 | 0.535 | 2.379 | 11.841 |
| `experience.next_state` | 0.970 | 1.196 | 1.033 | 1.132 | 2.158 |
| `experience.current_state` | 0.767 | 0.793 | 0.846 | 1.009 | 0.868 |
| `buffer.update_and_best` | 0.069 | 0.178 | 0.230 | 0.734 | 1.846 |

## Findings

- Throughput falls from 254.26 transitions/s at 58 gates to 32.11 at 3,435 gates; the fixed-work rollout is 7.92x slower.
- At 58-259 gates, constructing current/next DGL subgraphs for PPO training is the largest group (35-44%); exact Quartz apply is only 1-10%.
- At 831 gates, exact Quartz apply reaches 28.3% and becomes the largest individual stage.
- At 3,435 gates, Quartz apply is 38.0%, available-xfer checking/mask construction is 19.6%, and full graph-to-DGL conversion is 13.5%. Together they consume 71.1% of rollout.
- GNN time rises in absolute terms but its share falls from 11.9% to 5.3% at the largest size because sequential Quartz and CPU graph work grows faster.
- The largest run retained only 19 buffer states, so its 39.86s rollout cannot be attributed to a large persistent buffer in this iteration.
- The profiled and uninstrumented 3,435-gate runs were 39.86s and 40.56s with identical transitions and outcomes; instrumentation variance was -1.7%.

The raw JSON files retain all stage seconds, call counts, percentages, and
microseconds per transition. Remote `run.log` paths are recorded in the summary JSON.
