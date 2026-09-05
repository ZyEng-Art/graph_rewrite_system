# Verified Best and Replay Starts

Date: 2026-09-05

## Change

The hierarchical collector can now mix three exact episode starts:

- the original input circuit;
- the globally verified best QASM;
- a bounded reservoir of Quartz-verified replay states.

The best and replay probabilities are independent CLI controls. Search state
can be restored from a hierarchical PPO checkpoint, including the actor,
verified best, replay pool, and exact rejection cache. Mixed-start metrics log
the complete initial gate-count histogram instead of reporting only the first
episode's gate count.

The replay reservoir now protects its minimum-gate state. A new record low
replaces a worst retained state; ordinary reservoir replacement is not allowed
to evict the protected minimum. The remaining entries retain stochastic
diversity, including uphill frontier states.

## 57-to-56 Continuation

The first run resumes the strict PPO checkpoint whose exact best is 57. It
uses 30 iterations, 64 total episodes per iteration, horizon 32, 25% best
starts, and 25% replay starts.

- Iteration 1 starts 14/64 episodes at 57 gates.
- Iteration 4 first finds an exact 56-gate graph.
- Every later iteration reaches 56 again; it is not a one-off audit result.
- The final independent batch starts 13 episodes at 56 and has six episodes
  whose best is 56.
- Final throughput is 891.1 transitions/s with 1785 accepted rewrites.
- The exact best QASM is stored at gate count 56 and accepted depth 32.

The run before minimum protection retained 64 replay states from 53,080 unique
verified states, but its replay minimum was 57 even though the global best was
56. This measurement motivated the protected-minimum change.

## Horizon-64 Continuation

A second 30-iteration run resumes from 56 with 50% best starts, 20% replay
starts, and horizon 64. The protected replay pool correctly retains a 56-gate
state after seeing 122,410 unique verified states. It does not find 55.

The longer horizon does not improve effective search depth under the current
retry budget. With at most eight exact rejections per episode, only 1--5 of 64
episodes usually reach depth 64; the final batch has four full horizons. The
policy averages roughly 33 accepted rewrites per final episode before exhausting
its rejection budget.

This rules out simply increasing horizon as the next optimization. Further
progress requires broader multi-circuit training and a better learned
legality/cycle prior, or a retry budget that adapts to circuit size. Repeated
single-circuit PPO would recreate Quarl's per-circuit fine-tuning problem.
