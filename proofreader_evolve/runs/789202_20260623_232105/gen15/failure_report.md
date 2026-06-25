# Candidate failure report (train split)

- Proposed **972 edits** from 5000 candidate sites.
- **Split-repair score (THE FITNESS the gate keeps on) = correct - false = 99 - 0 = 99** (train-classified merges; unscored=873). The gate accepts only if this BEATS the parent AND false == 0 — a single false merge (fusing two different neurons) rejects the candidate outright.
- Edge Accuracy (secondary diagnostic, NOT the bar; = 100 - %Split - %Omit - %Merged): baseline 85.0156 -> candidate 81.5840.
- Merge-error component: %Merged Edges baseline 11.7259 -> candidate 15.1621; # Merges baseline 7.88 -> candidate 7.88.
- Over-split watchdog: %Split Edges baseline 0.3121 -> candidate 0.3078 (a merge repair that drives this UP is over-splitting a real neuron).

- Largest fused class = **3 raw labels** (0 class(es) fuse >5 labels). OK (no chaining)


## ENUM_PARAMS rail sensitivity (candidate-stream knobs)

The candidate stream you reason over is shaped by `ENUM_PARAMS` (module-level dict in heuristics.py). Each knob is CLAMPED to a safety rail; `requested` is what your policy asked for ('—' = not set, framework default used), `in_effect` is what actually shaped THIS run's stream. If a knob is `clamped`, your request had NO effect past the rail — stop tuning it. If it is `[AT … rail]`, the stream is already at its widening/narrowing limit in that direction.

| knob | requested | in_effect | rail [lo, hi] | status |
|---|---|---|---|---|
| max_gap_um | 15.0 | 15.0 | [1.0, 40.0] | ok |
| split_max_sites | — | 5000 | [100, 50000] | default |
| tip_to_shaft | True | True | — | ok |
| min_arm_cable_um | 10.0 | 10.0 | [2.0, 50.0] | ok |
| seed_depth_um | 8.0 | 8.0 | [2.0, 30.0] | ok |
| merge_max_sites | — | 5000 | [100, 50000] | default |
| max_per_label | 8 | 8 | [1, 100] | ok |

## Per-skeleton delta (candidate - baseline)

| GT skeleton | dEdgeAcc | d%MergedEdges | d#Merges | d%SplitEdges | d#Splits | d%OmitEdges |
|---|---|---|---|---|---|---|
| N006-789202-JG | -1.240 | +1.244 | +0 | -0.002 | -18 | +0.000 |
| N011-789202-IG | +0.030 | +0.000 | +0 | -0.015 | -19 | -0.012 |
| N016-789202-HP | -10.890 | +10.886 | +0 | -0.002 | -15 | +0.000 |
| N020-789202-PP | -0.610 | +0.606 | +0 | -0.002 | -5 | +0.000 |
| N022-789202-JG | -0.610 | +0.620 | +0 | -0.006 | -33 | -0.003 |
| N023-789202-SP | -2.130 | +2.134 | +0 | -0.004 | -9 | -0.004 |

_Note: a positive `d#Splits` paired with a negative `d%OmitEdges` is the footprint of your MERGES healing background (omit) nodes into labels (the skeleton then crosses more distinct labels) — not over-splitting by you. `# Splits` / Edge Accuracy are GT-side diagnostics, NOT the gate._


## Train-side FALSE merges (precision alert — NOT a gate reject)

_no train-side false merges — your merges did not fuse two different neurons on any train skeleton._

## Skeletons with an Edge-Accuracy dip (secondary diagnostic)

N006-789202-JG, N016-789202-HP, N020-789202-PP, N022-789202-JG, N023-789202-SP


## SplitSite audit: REAL splits vs FALSE joins, crossed with your decision

Each enumerated SplitSite is classified by whether its two fragment labels belong to the SAME train GT neuron (a REAL split your `merge_labels` SHOULD repair — the RECALL signal) or DIFFERENT neurons (a FALSE join that would CREATE a merge — the precision signal), AND crossed with whether YOUR policy accepted (emitted a merge) or rejected it. So:
  • **rejected REAL = your false NEGATIVES** (reachable real splits you MISSED — raise recall here);
  • **accepted FALSE = your false POSITIVES** (over-merges that fail the gate — tighten here);
  • accepted REAL = correct repairs; rejected FALSE = correct refusals.
Labels whose dominant neuron is held-out are omitted (leak-free).

Each row carries cheap, GT-free fragment geometry so you can pick a GENERALIZABLE accept/reject rule (gap alone barely separates the buckets):
  • `colinear_cos` — straightness of armA→gap→armB: ~+1 a clean colinear continuation (the segmentation broke ONE neuron — merge), ~0 a right-angle join, <0 the arms double back (likely a FALSE join). The single strongest precision feature here.
  • `deg_a`/`deg_b` — endpoint graph degree (A is always a tip=1; B = 1 tip / 2 shaft / 3+ branch). Joining INTO a branch/shaft is riskier than tip-to-tip.
  • `rad_a`/`rad_b`/`rad_ratio` — neurite radius at each end and their max/min. A ratio far from 1.0 = two different cable calibers (less likely one broken neuron).
Find the colinear_cos / rad_ratio cutoff that separates your MISSES from your hits below.

Confusion summary: REAL 535 (accepted 99 / MISSED 436); FALSE 3 (WRONGLY accepted 0 / correctly rejected 3).


**MISSED real splits (rejected REAL — raise recall)** (436 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 0.00 | — | 1 | 1 | 1.67 | 0.75 | 2.24 | rejected |
| 0.79 | -0.16 | 1 | 2 | 0.75 | 1.45 | 1.94 | rejected |
| 0.85 | 0.05 | 1 | 2 | 1.00 | 1.22 | 1.22 | rejected |
| 1.00 | 0.15 | 1 | 1 | 1.00 | 0.75 | 1.34 | rejected |
| 1.00 | 0.06 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.00 | 0.18 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.00 | 0.06 | 1 | 1 | 1.50 | 0.75 | 2.00 | rejected |
| 1.00 | 0.08 | 1 | 1 | 1.46 | 0.75 | 1.95 | rejected |
| 1.00 | -0.43 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.22 | 0.03 | 1 | 2 | 0.75 | 1.14 | 1.52 | rejected |
| 1.32 | 0.43 | 1 | 2 | 0.75 | 1.40 | 1.88 | rejected |
| 1.33 | 0.35 | 1 | 2 | 0.75 | 1.80 | 2.41 | rejected |
| 1.35 | 0.07 | 1 | 2 | 0.75 | 1.36 | 1.82 | rejected |
| 1.37 | 0.50 | 1 | 2 | 1.00 | 1.94 | 1.94 | rejected |
| 1.38 | 0.10 | 1 | 2 | 1.25 | 1.44 | 1.15 | rejected |
| 1.41 | -0.06 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.41 | -0.87 | 1 | 1 | 0.75 | 1.25 | 1.67 | rejected |
| 1.41 | -0.64 | 1 | 1 | 1.06 | 1.00 | 1.06 | rejected |
| 1.41 | 0.02 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.41 | -0.23 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.41 | -0.11 | 1 | 1 | 1.06 | 1.50 | 1.41 | rejected |
| 1.41 | -0.40 | 1 | 1 | 0.75 | 1.95 | 2.60 | rejected |
| 1.41 | -0.21 | 1 | 1 | 1.00 | 1.25 | 1.25 | rejected |
| 1.41 | -0.71 | 1 | 1 | 0.75 | 1.25 | 1.67 | rejected |
| 1.41 | -0.34 | 1 | 1 | 0.75 | 1.46 | 1.95 | rejected |
| 1.41 | 0.37 | 1 | 1 | 1.00 | 0.75 | 1.34 | rejected |
| 1.41 | -0.68 | 1 | 1 | 0.75 | 1.50 | 2.00 | rejected |
| 1.47 | -0.06 | 1 | 2 | 0.75 | 1.75 | 2.34 | rejected |
| 1.49 | 0.10 | 1 | 2 | 1.00 | 1.54 | 1.54 | rejected |
| 1.51 | 0.45 | 1 | 2 | 0.75 | 1.94 | 2.60 | rejected |
| 1.52 | -0.05 | 1 | 2 | 1.06 | 1.80 | 1.70 | rejected |
| 1.53 | 0.42 | 1 | 2 | 1.06 | 1.70 | 1.61 | rejected |
| 1.60 | 0.24 | 1 | 2 | 0.75 | 1.97 | 2.64 | rejected |
| 1.62 | -0.17 | 1 | 2 | 0.75 | 1.91 | 2.56 | rejected |
| 1.65 | 0.07 | 1 | 2 | 0.75 | 1.46 | 1.95 | rejected |
| 1.65 | 0.78 | 1 | 2 | 0.75 | 2.01 | 2.69 | rejected |
| 1.68 | 0.34 | 1 | 2 | 1.00 | 2.01 | 2.01 | rejected |
| 1.71 | -0.11 | 1 | 2 | 0.75 | 1.49 | 1.99 | rejected |
| 1.72 | 0.41 | 1 | 2 | 1.06 | 1.62 | 1.53 | rejected |
| 1.73 | -0.67 | 1 | 1 | 0.75 | 1.25 | 1.67 | rejected |
| 1.73 | 0.50 | 1 | 1 | 1.00 | 1.80 | 1.80 | rejected |
| 1.73 | -0.11 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.73 | 0.28 | 1 | 1 | 1.95 | 1.06 | 1.84 | rejected |
| 1.73 | -0.76 | 1 | 1 | 0.75 | 1.06 | 1.41 | rejected |
| 1.73 | -0.29 | 1 | 1 | 0.75 | 1.80 | 2.41 | rejected |
| 1.73 | -0.12 | 1 | 1 | 1.06 | 1.06 | 1.00 | rejected |
| 1.75 | 0.10 | 1 | 2 | 1.00 | 1.26 | 1.26 | rejected |
| 1.76 | 0.15 | 1 | 2 | 1.00 | 1.84 | 1.84 | rejected |
| 1.77 | 0.19 | 1 | 2 | 1.00 | 1.79 | 1.79 | rejected |
| 1.80 | -0.05 | 1 | 2 | 0.75 | 1.83 | 2.45 | rejected |
| 1.80 | 0.29 | 1 | 2 | 1.25 | 1.56 | 1.25 | rejected |
| 1.82 | 0.86 | 1 | 2 | 1.25 | 1.86 | 1.49 | rejected |
| 1.88 | 0.29 | 1 | 2 | 0.75 | 1.93 | 2.58 | rejected |
| 1.89 | 0.02 | 1 | 2 | 0.75 | 1.27 | 1.70 | rejected |
| 1.92 | 0.20 | 1 | 2 | 1.00 | 1.81 | 1.81 | rejected |
| 1.96 | 0.57 | 1 | 2 | 0.75 | 1.87 | 2.50 | rejected |
| 1.98 | 0.13 | 1 | 2 | 0.75 | 1.91 | 2.56 | rejected |
| 1.98 | 0.30 | 1 | 2 | 1.00 | 1.90 | 1.90 | rejected |
| 1.98 | 0.04 | 1 | 2 | 0.75 | 1.64 | 2.20 | rejected |
| 2.00 | 0.70 | 1 | 1 | 1.25 | 1.80 | 1.44 | rejected |

…and 376 more.

**WRONGLY accepted false joins (accepted FALSE — over-merges to kill)** (0 sites):
_none._

**Correctly repaired (accepted REAL)** (99 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 0.61 | 0.78 | 1 | 2 | 1.46 | 1.67 | 1.14 | accepted |
| 1.13 | 0.57 | 1 | 2 | 0.75 | 1.79 | 2.40 | accepted |
| 1.17 | 0.74 | 1 | 2 | 1.06 | 1.63 | 1.54 | accepted |
| 1.41 | 0.80 | 1 | 1 | 1.06 | 1.00 | 1.06 | accepted |
| 1.41 | 0.63 | 1 | 1 | 1.25 | 1.25 | 1.00 | accepted |
| 1.43 | 0.70 | 1 | 2 | 1.67 | 1.43 | 1.17 | accepted |
| 1.65 | 0.87 | 1 | 2 | 0.75 | 1.97 | 2.63 | accepted |
| 1.65 | 0.92 | 1 | 2 | 1.46 | 1.46 | 1.01 | accepted |
| 1.67 | 0.76 | 1 | 2 | 0.75 | 1.51 | 2.01 | accepted |
| 1.73 | 0.77 | 1 | 1 | 0.75 | 2.24 | 3.00 | accepted |
| 1.73 | 0.72 | 1 | 1 | 0.75 | 1.50 | 2.00 | accepted |
| 1.73 | 0.90 | 1 | 1 | 0.75 | 1.95 | 2.60 | accepted |
| 1.82 | 0.79 | 1 | 2 | 1.25 | 1.74 | 1.39 | accepted |
| 1.86 | 0.81 | 1 | 2 | 1.25 | 1.27 | 1.02 | accepted |
| 1.86 | 0.78 | 1 | 2 | 1.00 | 1.72 | 1.72 | accepted |
| 1.86 | 0.85 | 1 | 2 | 0.75 | 1.91 | 2.55 | accepted |
| 1.87 | 0.67 | 1 | 2 | 0.75 | 1.69 | 2.26 | accepted |
| 1.97 | 0.63 | 1 | 2 | 0.75 | 1.85 | 2.48 | accepted |
| 2.00 | 0.88 | 1 | 1 | 1.50 | 1.67 | 1.12 | accepted |
| 2.00 | 0.91 | 1 | 1 | 1.06 | 1.25 | 1.18 | accepted |
| 2.00 | 0.81 | 1 | 1 | 2.13 | 1.80 | 1.19 | accepted |
| 2.06 | 0.64 | 1 | 2 | 0.75 | 1.55 | 2.08 | accepted |
| 2.21 | 0.72 | 1 | 2 | 0.75 | 1.71 | 2.29 | accepted |
| 2.23 | 0.66 | 1 | 2 | 0.75 | 1.21 | 1.62 | accepted |
| 2.24 | 0.91 | 1 | 1 | 1.80 | 2.12 | 1.18 | accepted |
| 2.24 | 0.85 | 1 | 1 | 1.50 | 1.46 | 1.03 | accepted |
| 2.24 | 0.77 | 1 | 1 | 1.67 | 1.46 | 1.15 | accepted |
| 2.24 | 0.94 | 1 | 1 | 1.50 | 1.67 | 1.12 | accepted |
| 2.24 | 0.64 | 1 | 1 | 1.06 | 1.46 | 1.38 | accepted |
| 2.24 | 0.97 | 1 | 1 | 1.95 | 0.75 | 2.60 | accepted |
| 2.32 | 0.64 | 1 | 2 | 1.06 | 1.97 | 1.86 | accepted |
| 2.35 | 0.87 | 1 | 2 | 0.75 | 1.56 | 2.09 | accepted |
| 2.45 | 1.00 | 1 | 1 | 1.80 | 0.75 | 2.41 | accepted |
| 2.45 | 0.67 | 1 | 1 | 1.67 | 1.95 | 1.16 | accepted |
| 2.45 | 0.85 | 1 | 1 | 1.46 | 1.25 | 1.17 | accepted |
| 2.48 | 0.87 | 1 | 2 | 1.80 | 1.84 | 1.02 | accepted |
| 2.58 | 0.67 | 1 | 2 | 1.25 | 1.58 | 1.27 | accepted |
| 2.61 | 0.69 | 1 | 2 | 0.75 | 1.69 | 2.25 | accepted |
| 2.68 | 0.71 | 1 | 2 | 1.67 | 1.38 | 1.21 | accepted |
| 2.68 | 0.78 | 1 | 2 | 1.67 | 1.92 | 1.15 | accepted |
| 2.70 | 0.78 | 1 | 2 | 1.06 | 1.93 | 1.82 | accepted |
| 2.80 | 0.80 | 1 | 2 | 1.25 | 1.99 | 1.59 | accepted |
| 2.81 | 0.97 | 1 | 2 | 1.00 | 1.82 | 1.82 | accepted |
| 2.83 | 0.97 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 2.83 | 0.81 | 1 | 1 | 1.46 | 0.75 | 1.95 | accepted |
| 2.83 | 0.65 | 1 | 1 | 0.75 | 2.00 | 2.67 | accepted |
| 2.89 | 0.62 | 1 | 2 | 1.25 | 1.81 | 1.45 | accepted |
| 2.89 | 0.74 | 1 | 2 | 1.46 | 1.94 | 1.33 | accepted |
| 2.91 | 0.61 | 1 | 2 | 1.25 | 1.70 | 1.36 | accepted |
| 2.98 | 0.70 | 1 | 2 | 1.06 | 2.05 | 1.94 | accepted |
| 3.00 | 0.61 | 1 | 1 | 0.75 | 1.95 | 2.60 | accepted |
| 3.00 | 0.96 | 1 | 1 | 1.50 | 1.46 | 1.03 | accepted |
| 3.00 | 0.77 | 1 | 1 | 1.95 | 1.67 | 1.16 | accepted |
| 3.00 | 0.89 | 1 | 1 | 1.00 | 1.67 | 1.67 | accepted |
| 3.00 | 0.96 | 1 | 1 | 1.67 | 1.67 | 1.00 | accepted |
| 3.00 | 0.67 | 1 | 1 | 2.34 | 1.80 | 1.30 | accepted |
| 3.00 | 0.97 | 1 | 1 | 1.50 | 2.00 | 1.34 | accepted |
| 3.00 | 0.93 | 1 | 1 | 1.50 | 1.67 | 1.12 | accepted |
| 3.07 | 0.92 | 1 | 2 | 1.95 | 1.62 | 1.20 | accepted |
| 3.12 | 0.65 | 1 | 2 | 0.75 | 1.48 | 1.98 | accepted |

…and 39 more.

**Correctly refused (rejected FALSE)** (3 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 2.20 | 0.44 | 1 | 2 | 1.00 | 1.97 | 1.97 | rejected |
| 3.00 | 0.06 | 1 | 1 | 1.00 | 1.67 | 1.67 | rejected |
| 3.16 | 0.67 | 1 | 1 | 1.25 | 2.34 | 1.87 | rejected |

_(4462 SplitSite(s) omitted: a label's dominant neuron is held-out, so no train-derivable verdict — excluded to keep the signal leak-free.)_


## Image evidence the policy read (gap_bridge_evidence), by GT class

Bridge statistics from `gap_bridge_evidence` calls the policy ALREADY made, split by whether the SplitSite's two labels are the SAME train neuron. A HIGH `bridge_ratio` (signal stays bright across the gap) is the merge tell; pick a `merge_labels` image threshold that ACCEPTS the REAL splits below and REJECTS the false joins.

**REAL splits — SHOULD merge** (90 read):
| bridge_ratio | bridge_min | endpoint_mean | bridge_pos |
|---|---|---|---|
| 1.00 | 29.00 | 29.0 | 0.00 |
| 1.00 | 50.00 | 50.0 | 0.00 |
| 0.98 | 26.00 | 26.5 | 0.75 |
| 0.99 | 53.00 | 53.5 | 0.00 |
| 0.94 | 25.00 | 26.5 | 0.50 |
| 0.97 | 18.00 | 18.5 | 0.00 |
| 0.84 | 32.00 | 38.0 | 0.75 |
| 1.00 | 23.00 | 23.0 | 0.00 |
| 0.97 | 17.00 | 17.5 | 0.00 |
| 0.98 | 119.00 | 121.2 | 1.00 |
| 0.85 | 30.00 | 35.2 | 1.00 |
| 1.00 | 15.00 | 15.0 | 0.00 |
| 1.00 | 12.00 | 12.0 | 0.00 |
| 0.97 | 86.00 | 89.0 | 0.00 |
| 0.98 | 21.00 | 21.5 | 0.00 |
| 0.98 | 91.00 | 92.5 | 0.00 |
| 0.97 | 56.00 | 58.0 | 0.75 |
| 0.97 | 34.00 | 35.0 | 0.00 |
| 1.00 | 11.00 | 11.0 | 0.00 |
| 0.98 | 108.00 | 110.2 | 1.00 |
| 1.02 | 165.00 | 161.8 | 1.00 |
| 0.97 | 53.00 | 54.5 | 1.00 |
| 1.00 | 33.00 | 33.0 | 0.00 |
| 0.96 | 34.00 | 35.5 | 0.50 |
| 1.02 | 22.00 | 21.5 | 1.00 |
| 1.00 | 41.00 | 41.0 | 0.00 |
| 0.97 | 35.00 | 36.0 | 0.75 |
| 0.99 | 22.00 | 22.2 | 0.25 |
| 1.00 | 28.00 | 28.0 | 0.00 |
| 1.00 | 24.00 | 24.0 | 1.00 |
| 1.02 | 88.00 | 86.2 | 1.00 |
| 1.01 | 35.00 | 34.5 | 1.00 |
| 0.95 | 62.00 | 65.5 | 0.00 |
| 0.98 | 80.00 | 81.5 | 1.00 |
| 0.98 | 22.00 | 22.5 | 0.50 |
| 1.00 | 34.00 | 34.0 | 0.00 |
| 0.98 | 27.00 | 27.5 | 0.00 |
| 0.97 | 233.00 | 239.2 | 1.00 |
| 1.04 | 27.00 | 26.0 | 1.00 |
| 1.00 | 33.00 | 33.0 | 0.00 |

**FALSE joins — must NOT merge** (0 read):
_none read on a cross-neuron SplitSite this generation._


## Edits proposed

972 edits: 972× merge_labels

merge(801978891, 802051384), merge(565614391, 565469412), merge(366536142, 202641797), merge(375899928, 521318114), merge(452805935, 438163157), merge(487951931, 473236675), merge(710871816, 755235300), merge(389720140, 330642703), merge(307033194, 307033204), merge(341308216, 620607676), merge(363570665, 363498394), merge(901511095, 886868105), merge(724783573, 724783574), merge(640352064, 625854266), merge(331654259, 316938994), merge(741179881, 726537103), merge(801699872, 787129586), merge(136997205, 137069487), merge(817567298, 817567299), merge(731361673, 731289401), merge(623092676, 623092456), merge(539069160, 567702317), merge(684932360, 685004630), merge(446836433, 432049117), merge(524939720, 524867231), merge(110629212, 125343127), merge(334276135, 334276133), merge(454026860, 349999232), merge(80447246, 80374753), merge(512971368, 512899102), merge(497535396, 497535183), merge(607567103, 607495051), merge(434240656, 434312917), merge(699466729, 699394241), merge(675046259, 660403263), merge(158781357, 158708648), merge(641847175, 627059211), merge(332034661, 346750587), merge(700580573, 685937578), merge(546602979, 620179976)

…and 932 more.


## Merges attributed to edits (the ones to fix)

Each row is a merge that an edit *created* (its class fuses >=2 raw labels). Guard these label pairs in the policy.

| GT skeleton | fused labels | merge location (world) |
|---|---|---|
| N020-789202-PP | 245469874+404943484 | (4781.96, 15262.19, 19668.0) |
| N023-789202-SP | 861787341+891579445 | (11444.4, 17603.43, 34770.0) |
| N022-789202-JG | 375815390+510142981+539429632 | (9686.6, 18381.35, 34773.0) |
| N022-789202-JG | 802201821+816917309 | (10493.69, 15797.76, 35954.0) |
| N006-789202-JG | 375899928+419836153+521318114 | (5029.55, 16824.02, 41598.0) |
| N016-789202-HP | 156096417+97165705 | (2618.0, 16578.67, 18204.0) |




## Image warm-start probe: does bridge_ratio separate REAL from FALSE splits?

HARNESS-measured (NOT your policy, NOT scored, never held-out): the harness read `gap_bridge_evidence` on a balanced, budget-capped sample of train SplitSites in the `gap_um` band where REAL and FALSE OVERLAP — i.e. where geometry alone CANNOT separate them, so image has the most decision value. Use this to decide, BEFORE writing any image rule, whether `bridge_ratio` is worth gating on, and roughly where the threshold sits. A high `bridge_ratio` (signal stays bright across the gap) should mark REAL splits you SHOULD merge.

**Separability: bridge_ratio AUC = 0.86** (1.0 = REAL always brighter than FALSE; 0.5 = no separation). Read: strong — image is worth using.

**REAL splits — SHOULD merge** (12 probed):
| gap_um | bridge_ratio | bridge_min | endpoint_mean |
|---|---|---|---|
| 2.63 | 0.98 | 23.00 | 23.5 |
| 2.62 | 1.00 | 73.00 | 73.0 |
| 2.61 | 1.01 | 35.00 | 34.5 |
| 2.65 | 1.01 | 266.00 | 263.5 |
| 2.60 | 0.99 | 91.00 | 91.5 |
| 2.66 | 0.98 | 23.00 | 23.5 |
| 2.66 | 1.00 | 26.00 | 26.0 |
| 2.59 | 0.98 | 27.00 | 27.5 |
| 2.59 | 0.98 | 45.00 | 46.0 |
| 2.59 | 1.00 | 22.00 | 22.0 |
| 2.67 | 0.98 | 68.00 | 69.5 |
| 2.58 | 1.02 | 88.00 | 86.2 |

**FALSE joins — must NOT merge** (3 probed):
| gap_um | bridge_ratio | bridge_min | endpoint_mean |
|---|---|---|---|
| 3.00 | 0.97 | 259.00 | 265.8 |
| 2.20 | 0.97 | 173.00 | 179.2 |
| 3.16 | 0.99 | 172.00 | 174.5 |