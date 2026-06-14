# Candidate failure report (train split)

- Proposed **0 edits** from 10000 candidate sites.
- Train Edge Accuracy (the fitness, = 100 - %Split - %Omit - %Merged; higher is better): baseline 85.4327 -> candidate 85.4327.
- Merge-error component (the repair target): %Merged Edges baseline 9.0921 -> candidate 9.0921; # Merges baseline 5.93 -> candidate 5.93.
- Over-split watchdog: %Split Edges baseline 0.1724 -> candidate 0.1724 (a merge repair that drives this UP is over-splitting a real neuron).


## Per-skeleton delta (candidate - baseline)

| GT skeleton | dEdgeAcc | d%MergedEdges | d#Merges | d%SplitEdges | d#Splits | d%OmitEdges |
|---|---|---|---|---|---|---|
| N005-789202-SP | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N006-789202-JG | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N007-789202-PP | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N010-789202-JT | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N011-789202-IG | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N020-789202-PP | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N022-789202-JG | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N023-789202-SP | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |

## Skeletons the candidate made WORSE

_none — every skeleton improved or held._


## Baseline merge errors (split_label repair targets)

_none — no raw label spans >=2 GT neurons in this split._


## Edits proposed

_none — the policy proposed no edits._


## Merges attributed to edits (the ones to fix)

_no merge was caused by an edit (any merges are pre-existing in the baseline segmentation)._

