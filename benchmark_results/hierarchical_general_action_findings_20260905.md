# General Hierarchical Action Pretraining

Date: 2026-09-05

## Corpus Split Fix

The combined trajectory corpus contains 99 Barenco and GF optimization paths,
many represented as multiple 64-action windows. The old preference splitter
hashed each `#window=` path independently. This could put windows from the same
original long trajectory on both sides of the train/test split.

The collector now removes both `#segment=` and `#window=` suffixes before
hashing. All windows of one original trajectory therefore share one split. The
new corpus contains:

| Item | Count |
| --- | ---: |
| Original paths | 99 |
| Train/test paths | 78 / 21 |
| Unique teacher action states | 2635 |
| Train/test action pairs | 34,160 / 8,000 |
| Hard negatives per teacher action | 16 |

Each preference also records the realized teacher gate delta and future best
reduction from the exact trajectory. The action trainer uses the latter for
bounded return weighting, including temporary uphill actions that later pay
off.

## General Actor Accuracy

One history-conditioned pattern actor is trained over both circuit families.
The graph/matcher model and return-weighted node actor remain frozen.

| Test metric | Initial | Best epoch 24 |
| --- | ---: | ---: |
| All held-out paths | 45.46% | 88.27% |
| Barenco paths | 44.30% | 83.62% |
| GF paths | 45.68% | 89.15% |
| Same-anchor pairs | 52.85% | 74.60% |
| Same-source pairs | 52.51% | 82.88% |
| Uphill teacher preferred | 12.72% | 94.95% |

The target path split is by original trajectory, not by individual states or
windows. These are pairwise action-ranking metrics, not end-to-end circuit
optimization rates.

The formal checkpoint is stored remotely at
`/SharedData/dengzy/quarl_matchformer_fresh_20260902/runs/hierarchical_action_allpaths_dedup_pathsplit_n16_m2_rtg025_e30_s996.pt`.

## State-Encoding Deduplication

Each teacher state contributes up to 16 preference pairs. The first trainer
version replayed and encoded the same graph/action prefix once per pair. On the
full corpus, frozen feature encoding took 361.1 seconds for train and 77.0
seconds for test; the actual 30-epoch head training took only 22.8 seconds.

The collator now groups preference rows by exact source history and prefix
length, encodes each state once, and expands only its frozen features to pair
rows. It encodes 2135 unique train states and 500 unique test states instead of
42,160 pair rows.

| Encoding implementation | Train + test time | Speedup |
| --- | ---: | ---: |
| One state encode per pair | 438.1 s | 1.00x |
| Deduplicated state encode | 80.7 s | 5.43x |

The optimized full run, including 30 training epochs, completes its measured
encoding and head-training work in about 104.5 seconds. Its best held-out
accuracy is slightly higher than the original run (88.27% versus 88.25%), so
the speedup does not trade away the reported policy metric.
