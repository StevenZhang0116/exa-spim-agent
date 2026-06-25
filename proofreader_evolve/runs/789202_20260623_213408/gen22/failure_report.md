# Candidate failure report (train split)

- Proposed **1866 edits** from 5000 candidate sites.
- **Split-repair score (THE FITNESS the gate keeps on) = correct - false = 190 - 0 = 190** (train-classified merges; unscored=1676). The gate accepts only if this BEATS the parent AND false == 0 — a single false merge (fusing two different neurons) rejects the candidate outright.
- Edge Accuracy (secondary diagnostic, NOT the bar; = 100 - %Split - %Omit - %Merged): baseline 90.0449 -> candidate 85.3701.
- Merge-error component: %Merged Edges baseline 5.0241 -> candidate 9.7378; # Merges baseline 4.97 -> candidate 5.43.
- Over-split watchdog: %Split Edges baseline 0.4476 -> candidate 0.4335 (a merge repair that drives this UP is over-splitting a real neuron).

- Largest fused class = **5 raw labels** (0 class(es) fuse >5 labels). OK (no chaining)


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
| N005-789202-SP | -1.560 | +1.620 | +1 | -0.017 | -57 | -0.044 |
| N010-789202-JT | -0.280 | +0.328 | +0 | -0.015 | -31 | -0.027 |
| N011-789202-IG | +0.040 | +0.000 | +0 | -0.021 | -29 | -0.019 |
| N020-789202-PP | -18.650 | +18.667 | +0 | -0.013 | -16 | +0.000 |
| N021-789202-HP | -9.150 | +9.186 | +1 | -0.008 | -30 | -0.025 |
| N023-789202-SP | +0.020 | +0.000 | +0 | -0.013 | -27 | -0.011 |

_Note: a positive `d#Splits` paired with a negative `d%OmitEdges` is the footprint of your MERGES healing background (omit) nodes into labels (the skeleton then crosses more distinct labels) — not over-splitting by you. `# Splits` / Edge Accuracy are GT-side diagnostics, NOT the gate._


## Train-side FALSE merges (precision alert — NOT a gate reject)

_no train-side false merges — your merges did not fuse two different neurons on any train skeleton._

## Skeletons with an Edge-Accuracy dip (secondary diagnostic)

N005-789202-SP, N010-789202-JT, N020-789202-PP, N021-789202-HP


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

Confusion summary: REAL 490 (accepted 190 / MISSED 300); FALSE 1 (WRONGLY accepted 0 / correctly rejected 1).


**MISSED real splits (rejected REAL — raise recall)** (300 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 1.51 | -0.28 | 1 | 2 | 0.75 | 1.25 | 1.68 | rejected |
| 1.52 | 0.06 | 1 | 2 | 1.25 | 1.24 | 1.01 | rejected |
| 1.58 | -0.01 | 1 | 2 | 1.00 | 1.70 | 1.70 | rejected |
| 1.60 | 0.24 | 1 | 2 | 0.75 | 1.97 | 2.64 | rejected |
| 1.62 | 0.23 | 1 | 2 | 0.75 | 1.93 | 2.58 | rejected |
| 1.62 | -0.17 | 1 | 2 | 0.75 | 1.91 | 2.56 | rejected |
| 1.64 | 0.16 | 1 | 2 | 0.75 | 1.85 | 2.48 | rejected |
| 1.65 | 0.78 | 1 | 2 | 0.75 | 2.01 | 2.69 | rejected |
| 1.69 | 0.26 | 1 | 2 | 1.00 | 1.78 | 1.78 | rejected |
| 1.69 | 0.57 | 1 | 2 | 0.75 | 1.42 | 1.90 | rejected |
| 1.71 | 0.40 | 1 | 2 | 1.00 | 1.72 | 1.72 | rejected |
| 1.77 | 0.19 | 1 | 2 | 1.00 | 1.79 | 1.79 | rejected |
| 1.77 | 0.10 | 1 | 2 | 1.00 | 1.97 | 1.97 | rejected |
| 1.87 | 0.67 | 1 | 2 | 0.75 | 1.69 | 2.26 | rejected |
| 1.89 | 0.02 | 1 | 2 | 0.75 | 1.27 | 1.70 | rejected |
| 1.90 | 0.23 | 1 | 2 | 0.75 | 1.78 | 2.37 | rejected |
| 1.91 | 0.55 | 1 | 2 | 0.75 | 1.86 | 2.49 | rejected |
| 1.92 | 0.20 | 1 | 2 | 1.00 | 1.81 | 1.81 | rejected |
| 1.98 | 0.30 | 1 | 2 | 1.00 | 1.90 | 1.90 | rejected |
| 1.98 | 0.04 | 1 | 2 | 0.75 | 1.64 | 2.20 | rejected |
| 1.99 | 0.00 | 1 | 2 | 1.00 | 1.76 | 1.76 | rejected |
| 2.02 | 0.60 | 1 | 2 | 1.06 | 2.02 | 1.91 | rejected |
| 2.03 | 0.00 | 1 | 2 | 0.75 | 1.83 | 2.45 | rejected |
| 2.05 | 0.56 | 1 | 2 | 0.75 | 1.92 | 2.57 | rejected |
| 2.06 | -0.18 | 1 | 2 | 0.75 | 1.70 | 2.27 | rejected |
| 2.07 | 0.69 | 1 | 2 | 0.75 | 2.07 | 2.77 | rejected |
| 2.08 | 0.65 | 1 | 2 | 1.25 | 2.10 | 1.68 | rejected |
| 2.08 | 0.07 | 1 | 2 | 0.75 | 1.65 | 2.21 | rejected |
| 2.09 | 0.04 | 1 | 2 | 0.75 | 1.68 | 2.25 | rejected |
| 2.10 | -0.03 | 1 | 2 | 1.00 | 1.63 | 1.63 | rejected |
| 2.11 | 0.05 | 1 | 2 | 0.75 | 2.03 | 2.71 | rejected |
| 2.15 | 0.62 | 1 | 2 | 0.75 | 1.92 | 2.56 | rejected |
| 2.16 | 0.22 | 1 | 2 | 0.75 | 2.05 | 2.74 | rejected |
| 2.18 | 0.03 | 1 | 2 | 1.50 | 1.88 | 1.26 | rejected |
| 2.18 | 0.08 | 1 | 2 | 1.00 | 1.83 | 1.83 | rejected |
| 2.19 | 0.48 | 1 | 2 | 1.00 | 1.96 | 1.96 | rejected |
| 2.19 | -0.14 | 1 | 2 | 1.00 | 1.52 | 1.52 | rejected |
| 2.20 | 0.44 | 1 | 2 | 1.00 | 1.97 | 1.97 | rejected |
| 2.20 | 0.09 | 1 | 2 | 0.75 | 1.51 | 2.01 | rejected |
| 2.20 | 0.78 | 1 | 2 | 1.25 | 1.75 | 1.40 | rejected |
| 2.21 | 0.29 | 1 | 2 | 0.75 | 1.79 | 2.40 | rejected |
| 2.24 | -0.40 | 1 | 1 | 1.46 | 0.75 | 1.95 | rejected |
| 2.24 | 0.17 | 1 | 1 | 0.75 | 1.06 | 1.41 | rejected |
| 2.24 | 0.10 | 1 | 1 | 1.50 | 1.50 | 1.00 | rejected |
| 2.24 | 0.08 | 1 | 1 | 1.25 | 1.00 | 1.25 | rejected |
| 2.24 | 0.32 | 1 | 1 | 1.50 | 1.25 | 1.20 | rejected |
| 2.24 | -0.33 | 1 | 1 | 1.00 | 1.00 | 1.00 | rejected |
| 2.24 | -0.59 | 1 | 1 | 1.46 | 0.75 | 1.95 | rejected |
| 2.24 | 0.35 | 1 | 1 | 1.00 | 0.75 | 1.34 | rejected |
| 2.24 | 0.19 | 1 | 1 | 1.50 | 0.75 | 2.00 | rejected |
| 2.24 | 0.68 | 1 | 1 | 1.67 | 0.75 | 2.24 | rejected |
| 2.24 | -0.21 | 1 | 1 | 0.75 | 2.13 | 2.85 | rejected |
| 2.24 | 0.33 | 1 | 1 | 1.25 | 1.46 | 1.17 | rejected |
| 2.24 | 0.45 | 1 | 1 | 1.50 | 1.67 | 1.12 | rejected |
| 2.24 | 0.66 | 1 | 1 | 1.25 | 1.25 | 1.00 | rejected |
| 2.24 | 0.34 | 1 | 1 | 1.06 | 1.00 | 1.06 | rejected |
| 2.24 | 0.53 | 1 | 1 | 1.46 | 1.25 | 1.17 | rejected |
| 2.24 | -0.60 | 1 | 1 | 0.75 | 1.50 | 2.00 | rejected |
| 2.24 | 0.37 | 1 | 1 | 1.46 | 1.46 | 1.00 | rejected |
| 2.24 | 0.22 | 1 | 1 | 1.67 | 1.00 | 1.67 | rejected |

…and 240 more.

**WRONGLY accepted false joins (accepted FALSE — over-merges to kill)** (0 sites):
_none._

**Correctly repaired (accepted REAL)** (190 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 0.18 | 0.02 | 1 | 2 | 0.75 | 1.80 | 2.40 | accepted |
| 0.56 | 0.59 | 1 | 2 | 1.00 | 1.58 | 1.58 | accepted |
| 0.66 | 0.11 | 1 | 2 | 1.06 | 1.55 | 1.47 | accepted |
| 0.73 | 0.18 | 1 | 2 | 1.06 | 2.04 | 1.92 | accepted |
| 0.94 | 0.21 | 1 | 2 | 0.75 | 1.53 | 2.04 | accepted |
| 1.00 | 0.36 | 1 | 1 | 0.75 | 1.25 | 1.67 | accepted |
| 1.00 | -0.31 | 1 | 1 | 0.75 | 1.46 | 1.95 | accepted |
| 1.00 | -0.49 | 1 | 1 | 0.75 | 1.80 | 2.41 | accepted |
| 1.00 | 0.18 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.00 | 0.06 | 1 | 1 | 1.50 | 0.75 | 2.00 | accepted |
| 1.00 | -0.43 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.09 | 0.03 | 1 | 2 | 0.75 | 1.55 | 2.08 | accepted |
| 1.12 | 0.08 | 1 | 2 | 1.67 | 1.87 | 1.12 | accepted |
| 1.17 | 0.74 | 1 | 2 | 1.06 | 1.63 | 1.54 | accepted |
| 1.17 | 0.94 | 1 | 2 | 0.75 | 1.79 | 2.40 | accepted |
| 1.17 | -0.25 | 1 | 2 | 1.25 | 1.44 | 1.15 | accepted |
| 1.19 | 0.84 | 1 | 2 | 1.25 | 1.68 | 1.35 | accepted |
| 1.22 | 0.03 | 1 | 2 | 0.75 | 1.14 | 1.52 | accepted |
| 1.24 | 0.45 | 1 | 2 | 0.75 | 1.49 | 1.99 | accepted |
| 1.25 | -0.03 | 1 | 2 | 1.00 | 1.76 | 1.76 | accepted |
| 1.32 | 0.43 | 1 | 2 | 0.75 | 1.40 | 1.88 | accepted |
| 1.33 | -0.18 | 1 | 2 | 1.00 | 1.16 | 1.16 | accepted |
| 1.35 | 0.07 | 1 | 2 | 0.75 | 1.36 | 1.82 | accepted |
| 1.41 | -0.41 | 1 | 1 | 1.67 | 0.75 | 2.24 | accepted |
| 1.41 | 0.34 | 1 | 1 | 1.46 | 1.06 | 1.38 | accepted |
| 1.41 | -0.06 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.41 | -0.11 | 1 | 1 | 1.06 | 0.75 | 1.41 | accepted |
| 1.41 | 0.20 | 1 | 1 | 1.80 | 0.75 | 2.41 | accepted |
| 1.41 | -0.23 | 1 | 1 | 1.06 | 0.75 | 1.41 | accepted |
| 1.41 | -0.21 | 1 | 1 | 1.00 | 1.25 | 1.25 | accepted |
| 1.41 | -0.73 | 1 | 1 | 1.00 | 0.75 | 1.34 | accepted |
| 1.41 | 0.01 | 1 | 1 | 1.50 | 0.75 | 2.00 | accepted |
| 1.41 | -0.68 | 1 | 1 | 0.75 | 1.50 | 2.00 | accepted |
| 1.41 | -0.25 | 1 | 1 | 0.75 | 1.00 | 1.34 | accepted |
| 1.41 | -0.10 | 1 | 1 | 1.67 | 0.75 | 2.24 | accepted |
| 1.41 | -0.05 | 1 | 1 | 1.06 | 0.75 | 1.41 | accepted |
| 1.44 | 0.12 | 1 | 2 | 0.75 | 1.86 | 2.49 | accepted |
| 1.45 | 0.60 | 1 | 2 | 2.00 | 1.51 | 1.32 | accepted |
| 1.46 | 0.78 | 1 | 2 | 1.25 | 1.68 | 1.35 | accepted |
| 1.49 | 0.28 | 1 | 2 | 0.75 | 1.57 | 2.10 | accepted |
| 1.56 | 0.32 | 1 | 2 | 0.75 | 1.88 | 2.51 | accepted |
| 1.65 | 0.87 | 1 | 2 | 0.75 | 1.97 | 2.63 | accepted |
| 1.73 | -0.67 | 1 | 1 | 0.75 | 1.25 | 1.67 | accepted |
| 1.73 | 0.62 | 1 | 1 | 1.46 | 0.75 | 1.95 | accepted |
| 1.73 | -0.11 | 1 | 1 | 1.06 | 0.75 | 1.41 | accepted |
| 1.73 | -0.05 | 1 | 1 | 1.67 | 0.75 | 2.24 | accepted |
| 1.73 | 0.56 | 1 | 1 | 0.75 | 1.46 | 1.95 | accepted |
| 1.73 | 0.90 | 1 | 1 | 0.75 | 1.95 | 2.60 | accepted |
| 1.73 | 0.71 | 1 | 1 | 1.06 | 1.67 | 1.58 | accepted |
| 1.73 | -0.12 | 1 | 1 | 1.06 | 1.06 | 1.00 | accepted |
| 1.75 | 0.87 | 1 | 2 | 1.00 | 1.79 | 1.79 | accepted |
| 1.91 | 0.82 | 1 | 2 | 1.25 | 1.49 | 1.19 | accepted |
| 2.00 | 0.91 | 1 | 1 | 1.06 | 1.25 | 1.18 | accepted |
| 2.00 | -0.55 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 2.00 | 0.81 | 1 | 1 | 2.13 | 1.80 | 1.19 | accepted |
| 2.00 | -0.33 | 1 | 1 | 0.75 | 1.00 | 1.34 | accepted |
| 2.00 | 0.12 | 1 | 1 | 1.95 | 0.75 | 2.60 | accepted |
| 2.00 | 0.52 | 1 | 1 | 1.67 | 1.06 | 1.58 | accepted |
| 2.24 | 0.98 | 1 | 1 | 1.50 | 0.75 | 2.00 | accepted |
| 2.24 | 0.84 | 1 | 1 | 2.12 | 1.50 | 1.41 | accepted |

…and 130 more.

**Correctly refused (rejected FALSE)** (1 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 3.12 | 0.55 | 1 | 2 | 1.25 | 1.99 | 1.59 | rejected |

_(4509 SplitSite(s) omitted: a label's dominant neuron is held-out, so no train-derivable verdict — excluded to keep the signal leak-free.)_


## Image evidence the policy read (gap_bridge_evidence), by GT class

Bridge statistics from `gap_bridge_evidence` calls the policy ALREADY made, split by whether the SplitSite's two labels are the SAME train neuron. A HIGH `bridge_ratio` (signal stays bright across the gap) is the merge tell; pick a `merge_labels` image threshold that ACCEPTS the REAL splits below and REJECTS the false joins.

**REAL splits — SHOULD merge** (158 read):
| bridge_ratio | bridge_min | endpoint_mean | bridge_pos |
|---|---|---|---|
| 1.00 | 39.00 | 39.0 | 0.00 |
| 0.98 | 133.00 | 135.5 | 0.00 |
| 0.97 | 30.00 | 31.0 | 0.50 |
| 0.97 | 33.00 | 34.0 | 0.50 |
| 1.00 | 66.00 | 66.0 | 0.00 |
| 0.98 | 24.00 | 24.5 | 0.75 |
| 1.00 | 18.00 | 18.0 | 0.00 |
| 0.98 | 49.00 | 50.0 | 0.75 |
| 1.00 | 20.00 | 20.0 | 0.00 |
| 0.98 | 44.50 | 45.2 | 0.00 |
| 0.97 | 24.50 | 25.2 | 0.75 |
| 1.00 | 22.00 | 22.0 | 0.00 |
| 1.00 | 23.00 | 23.0 | 0.00 |
| 1.00 | 25.00 | 25.0 | 0.00 |
| 1.00 | 44.00 | 44.0 | 0.00 |
| 1.00 | 20.00 | 20.0 | 0.00 |
| 1.00 | 14.00 | 14.0 | 0.00 |
| 1.00 | 18.00 | 18.0 | 0.00 |
| 1.00 | 29.00 | 29.0 | 1.00 |
| 1.00 | 33.00 | 33.0 | 0.00 |
| 1.00 | 21.00 | 21.0 | 0.00 |
| 1.02 | 165.00 | 161.8 | 1.00 |
| 1.00 | 43.00 | 43.0 | 0.00 |
| 0.98 | 30.00 | 30.5 | 0.00 |
| 1.00 | 21.00 | 21.0 | 0.00 |
| 1.00 | 41.00 | 41.0 | 0.00 |
| 1.00 | 53.00 | 53.0 | 0.00 |
| 1.00 | 27.00 | 27.0 | 0.00 |
| 1.02 | 23.00 | 22.5 | 1.00 |
| 1.01 | 53.00 | 52.5 | 1.00 |
| 0.99 | 35.00 | 35.5 | 0.00 |
| 1.02 | 88.00 | 86.2 | 1.00 |
| 1.00 | 68.50 | 68.2 | 1.00 |
| 1.00 | 44.00 | 44.0 | 0.00 |
| 0.98 | 27.00 | 27.5 | 0.00 |
| 1.01 | 35.00 | 34.5 | 1.00 |
| 0.98 | 23.00 | 23.5 | 0.00 |
| 1.02 | 24.00 | 23.5 | 1.00 |
| 0.96 | 27.00 | 28.0 | 0.00 |
| 1.03 | 18.00 | 17.5 | 0.00 |

**FALSE joins — must NOT merge** (0 read):
_none read on a cross-neuron SplitSite this generation._


## Edits proposed

1866 edits: 1866× merge_labels

merge(406128277, 406200992), merge(525270419, 525198142), merge(416983886, 402196132), merge(1095213747, 1095213755), merge(608373682, 608735686), merge(433216823, 418574265), merge(801978891, 802051384), merge(565614391, 565469412), merge(502450387, 502377899), merge(348274169, 333558927), merge(366536142, 202641797), merge(375899928, 521318114), merge(99080420, 99080424), merge(8683133, 8610643), merge(642002447, 346312585), merge(493129531, 493129532), merge(670722771, 685366197), merge(136997232, 166646552), merge(110629650, 110702127), merge(452805935, 438163157), merge(552116304, 626199623), merge(300561522, 300489036), merge(257372087, 287019871), merge(364798597, 364798596), merge(509140844, 551401497), merge(301953917, 316596916), merge(169436145, 169435928), merge(487951931, 473236675), merge(649248717, 663746735), merge(937821511, 923178736), merge(671087406, 656372132), merge(710871816, 755235300), merge(491750708, 491823196), merge(671077108, 671367497), merge(389720140, 330642703), merge(350520945, 365308435), merge(307033194, 307033204), merge(701275453, 701203401), merge(285715073, 285714856), merge(341308216, 620607676)

…and 1826 more.


## Merges attributed to edits (the ones to fix)

Each row is a merge that an edit *created* (its class fuses >=2 raw labels). Guard these label pairs in the policy.

| GT skeleton | fused labels | merge location (world) |
|---|---|---|
| N005-789202-SP | 582071132+612006453 | (8356.66, 16226.36, 52629.0) |
| N005-789202-SP | 449778262+480151161+494648954 | (6396.9, 18194.35, 51728.0) |
| N005-789202-SP | 464488053+553504107+568075271+597428505 | (6787.35, 19507.84, 46619.0) |
| N005-789202-SP | 554079644+568722855 | (7386.5, 20176.55, 43409.0) |
| N021-789202-HP | 817134991+846132124 | (10937.26, 15541.94, 37416.0) |
| N021-789202-HP | 557479832+675277741+849092788 | (8735.14, 27285.54, 39506.0) |
| N010-789202-JT | 408366016+408583287 | (5298.08, 23203.71, 32973.0) |
| N023-789202-SP | 861714850+861787341 | (11444.4, 17603.43, 34770.0) |
| N023-789202-SP | 463095869+492382520+492455008+492527280+507169623 | (6911.52, 14460.34, 32114.0) |




## Image warm-start probe: does bridge_ratio separate REAL from FALSE splits?

HARNESS-measured (NOT your policy, NOT scored, never held-out): the harness read `gap_bridge_evidence` on a balanced, budget-capped sample of train SplitSites in the `gap_um` band where REAL and FALSE OVERLAP — i.e. where geometry alone CANNOT separate them, so image has the most decision value. Use this to decide, BEFORE writing any image rule, whether `bridge_ratio` is worth gating on, and roughly where the threshold sits. A high `bridge_ratio` (signal stays bright across the gap) should mark REAL splits you SHOULD merge.

**Separability: bridge_ratio AUC = 0.83** (1.0 = REAL always brighter than FALSE; 0.5 = no separation). Read: strong — image is worth using.

**REAL splits — SHOULD merge** (12 probed):
| gap_um | bridge_ratio | bridge_min | endpoint_mean |
|---|---|---|---|
| 2.81 | 0.95 | 37.00 | 39.0 |
| 2.81 | 1.01 | 85.00 | 84.2 |
| 2.80 | 1.01 | 283.00 | 280.5 |
| 2.80 | 1.00 | 34.00 | 34.0 |
| 2.82 | 1.01 | 38.00 | 37.8 |
| 2.82 | 0.95 | 42.50 | 44.8 |
| 2.79 | 1.00 | 31.00 | 31.0 |
| 2.82 | 1.00 | 30.00 | 30.0 |
| 2.79 | 1.00 | 24.00 | 24.0 |
| 2.83 | 1.00 | 31.00 | 31.0 |
| 2.83 | 1.00 | 16.00 | 16.0 |
| 2.83 | 1.03 | 37.00 | 36.0 |

**FALSE joins — must NOT merge** (1 probed):
| gap_um | bridge_ratio | bridge_min | endpoint_mean |
|---|---|---|---|
| 3.12 | 1.00 | 91.50 | 91.8 |