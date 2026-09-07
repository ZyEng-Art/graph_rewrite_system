# Current source-topology matcher: raw-QASM end-to-end comparison

Date: 2026-09-07

## Outcome

The current source-topology matcher does **not** autonomously reproduce the
known Barenco 36-gate result from the untouched 58-gate QASM. With beam 1000,
depth 128, r99.9 candidate calibration, direct Quartz binding, rotation
elimination, and exact graph deduplication, it first reaches 38 gates at step
34 in 88.07 seconds and remains at 38 through step 128 (374.77 seconds total).

Original CPU Quartz under the same Barenco search budget also reaches 38 at
step 34, but in 684.34 seconds, and remains at 38 through step 128 (2277.92
seconds total). The current model is therefore 7.77x faster in time-to-38 and
6.08x faster for the complete equal-depth run, without a quality difference.

On the untouched 495-gate GF input, the model reaches 474 gates after 128
steps in 1865.97 seconds and does not reproduce the saved 369-gate result. A
full CPU depth-128 run would require hours at the measured rate, so the direct
quality-equivalent comparison uses the first eight layers: both methods reach
485 gates, the model in 84.33 seconds and CPU Quartz in 1404.38 seconds. This
is a 16.65x equal-depth end-to-end speedup. Comparing first arrival at 485 gives
15.80x (50.66 seconds for the model versus 800.68 seconds for CPU Quartz).

## Why the known Barenco path is lost

The 116-action teacher trajectory can be replayed exactly to 36, and the
current matcher recalls its continuation on every teacher state. That does not
mean an autonomous beam retains the teacher prefix.

The retention audit identifies the first loss after the fourth action:

| Teacher action count | Beam position of expected prefix |
| ---: | ---: |
| 1 | 10 |
| 2 | 90 |
| 3 | 417 |
| 4 | absent |

The lost fourth action is xfer 336 / source 323 and keeps the circuit at 58
gates. Four model candidates reproduce the expected successor. All four pass
the per-parent top-128 cap, but none enters the global top-16000 across the
whole beam. The layer contains many branches with an immediate 58-to-56 gate
decrease, so the current next-gate-count-first ordering removes this neutral
teacher step before Quartz apply. The loss is therefore neither a matcher miss
nor exact-dedup removal; it is a long-horizon ranking failure.

## Matcher throughput at batch 512

This test cyclically repeats each untouched original QASM to exactly 512 input
states. QASM parsing is excluded. “Full proposal” includes GPU matching,
source-to-xfer expansion, gate/parent/global ranking and caps, and transfer of
the selected proposals back to the host. It does not include Quartz rewrite
application or successor deduplication.

| Circuit | CPU Quartz xfer-at-anchor | Model full proposal | Core speedup | Model incl. one-time tensor preparation | Speedup incl. preparation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Barenco 58 | 43.79 states/s | 10,757.76 states/s | 245.66x | 982.22 states/s | 22.43x |
| GF 495 | 4.88 states/s | 3,764.41 states/s | 770.86x | 125.40 states/s | 25.68x |

The batch benchmark demonstrates that the matcher itself is fast. The smaller
end-to-end speedups arise from exact Quartz apply, invalid/duplicate successor
handling, graph materialization, and—most importantly for quality—the global
ranking policy.

## Protocol and artifacts

- Inputs are the untouched barenco_tof_3.qasm (58 gates) and gf2^6_mult.qasm
  (495 gates).
- Beam size is 1000; model depth is 128; CPU Barenco depth is 128; CPU GF depth
  is 8.
- Candidate settings are r99.9, 10,240 source matches per state, 128 actions
  per parent, global proposal factor 16 (16,000 proposals), and maximum
  declared gate increase 3.
- Every accepted successor is produced by Quartz and uses exact physical-wire
  graph identity for deduplication.
- Model runs and CPU GF ran on h100-gpu3; CPU Barenco ran on h100-gpu1.
  Both hosts expose 192 logical CPUs and H100 80 GB GPUs. The batch-512 CPU and
  GPU measurements ran sequentially on h100-gpu3.

Machine-readable values and artifact names are collected in
current_sourcetopo_raw_e2e_summary_20260907.json.
