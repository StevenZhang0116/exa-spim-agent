# Candidate failure report (train split)

- Proposed **2193 edits** from 15378 candidate sites.
- **Split-repair score (THE FITNESS the gate keeps on) = correct - false = 507 - 0 = 507** (train-classified merges; unscored=1686). The gate accepts only if this BEATS the parent AND false == 0 — a single false merge (fusing two different neurons) rejects the candidate outright.
- Edge Accuracy (secondary diagnostic, NOT the bar; = 100 - %Split - %Omit - %Merged): baseline 79.4094 -> candidate 77.9354.
- Merge-error component: %Merged Edges baseline 17.6169 -> candidate 19.1272; # Merges baseline 9.88 -> candidate 10.03.
- Over-split watchdog: %Split Edges baseline 0.3108 -> candidate 0.2976 (a merge repair that drives this UP is over-splitting a real neuron).


### Split-repair attribution — most discriminative feature axes (where your repairs vs. mistakes come from)

**deg_b** (quantile buckets) — correct / false / unscored (n):

  - deg_b [-inf,2): 158 / 0 / 474  (n=632)

  - deg_b [2,inf): 349 / 0 / 1212  (n=1561)

**gap_um** (quantile buckets) — correct / false / unscored (n):

  - gap_um [-inf,1.58535): 177 / 0 / 554  (n=731)

  - gap_um [1.58535,2.03288): 165 / 0 / 566  (n=731)

  - gap_um [2.03288,inf): 165 / 0 / 566  (n=731)

**rad_ratio** (quantile buckets) — correct / false / unscored (n):

  - rad_ratio [-inf,1.42924): 157 / 0 / 574  (n=731)

  - rad_ratio [1.42924,2.01175): 176 / 0 / 555  (n=731)

  - rad_ratio [2.01175,inf): 174 / 0 / 557  (n=731)

Read: a bucket with high correct and zero false is a regime worth EXTENDING; one with few correct but >0 false (or mostly unscored) is a regime to TIGHTEN or prune — do not spend more generations micro-tuning a low-yield region of feature space.

- Largest fused class = **6 raw labels** (16 class(es) fuse >5 labels). **MEGA-MERGE: a single class fused 6 fragments** — almost certainly an over-merge chaining distinct neurons, not one repair. Cap it (ENUM_PARAMS/policy) before raising recall.


## ENUM_PARAMS rail sensitivity (candidate-stream knobs)

The candidate stream you reason over is shaped by `ENUM_PARAMS` (module-level dict in heuristics.py). Each knob is CLAMPED to a safety rail; `requested` is what your policy asked for ('—' = not set, framework default used), `in_effect` is what actually shaped THIS run's stream. If a knob is `clamped`, your request had NO effect past the rail — stop tuning it. If it is `[AT … rail]`, the stream is already at its widening/narrowing limit in that direction.

| knob | requested | in_effect | rail [lo, hi] | status |
|---|---|---|---|---|
| max_gap_um | 15.0 | 15.0 | [1.0, 40.0] | ok |
| split_max_sites | 25000 | 25000 | [100, 50000] | ok |
| tip_to_shaft | True | True | — | ok |
| split_alt_per_pair | — | 1 | [1, 10] | default [AT lo rail] |
| min_arm_cable_um | 10.0 | 10.0 | [2.0, 50.0] | ok |
| seed_depth_um | 8.0 | 8.0 | [2.0, 30.0] | ok |
| merge_max_sites | — | 5000 | [100, 50000] | default |
| max_per_label | 8 | 8 | [1, 100] | ok |

## Per-skeleton delta (candidate - baseline)

| GT skeleton | dEdgeAcc | d%MergedEdges | d#Merges | d%SplitEdges | d#Splits | d%OmitEdges |
|---|---|---|---|---|---|---|
| N005-789202-SP | -2.000 | +2.078 | +0 | -0.024 | -61 | -0.051 |
| N006-789202-JG | -2.300 | +2.311 | +0 | -0.011 | -50 | -0.005 |
| N007-789202-PP | +0.130 | +0.000 | +0 | -0.027 | -39 | -0.102 |
| N010-789202-JT | -0.280 | +0.328 | +0 | -0.020 | -39 | -0.024 |
| N011-789202-IG | +0.050 | +0.000 | +0 | -0.022 | -30 | -0.028 |
| N013-789202-KS | -2.250 | +2.326 | +0 | -0.011 | -47 | -0.066 |
| N016-789202-HP | -0.310 | +0.316 | +0 | -0.004 | -28 | -0.001 |
| N018-789202-JM | -1.490 | +1.513 | +1 | -0.009 | -46 | -0.011 |
| N020-789202-PP | +0.010 | +0.000 | +0 | -0.011 | -16 | +0.000 |
| N021-789202-HP | -2.770 | +2.782 | +0 | -0.008 | -27 | -0.001 |
| N022-789202-JG | -1.510 | +1.536 | +0 | -0.015 | -61 | -0.012 |
| N023-789202-SP | -2.670 | +2.703 | +0 | -0.019 | -34 | -0.019 |

_Note: a positive `d#Splits` paired with a negative `d%OmitEdges` is the footprint of your MERGES healing background (omit) nodes into labels (the skeleton then crosses more distinct labels) — not over-splitting by you. `# Splits` / Edge Accuracy are GT-side diagnostics, NOT the gate._


## Train-side FALSE merges (precision alert — NOT a gate reject)

_no train-side false merges — your merges did not fuse two different neurons on any train skeleton._

## Skeletons with an Edge-Accuracy dip (secondary diagnostic)

N005-789202-SP, N006-789202-JG, N010-789202-JT, N013-789202-KS, N016-789202-HP, N018-789202-JM, N021-789202-HP, N022-789202-JG, N023-789202-SP


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

Confusion summary: REAL 2723 (accepted 507 / MISSED 2216); FALSE 42 (WRONGLY accepted 0 / correctly rejected 42).


**MISSED real splits (rejected REAL — raise recall)** (2216 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 2.15 | 0.31 | 1 | 2 | 1.00 | 1.90 | 1.90 | rejected |
| 2.15 | 0.67 | 1 | 2 | 0.75 | 1.77 | 2.37 | rejected |
| 2.15 | 0.29 | 1 | 2 | 0.75 | 1.93 | 2.58 | rejected |
| 2.15 | -0.01 | 1 | 2 | 1.06 | 1.27 | 1.20 | rejected |
| 2.16 | 0.05 | 1 | 2 | 0.75 | 1.42 | 1.90 | rejected |
| 2.16 | -0.16 | 1 | 2 | 1.00 | 2.01 | 2.01 | rejected |
| 2.16 | 0.56 | 1 | 2 | 0.75 | 2.03 | 2.72 | rejected |
| 2.17 | 0.51 | 1 | 2 | 1.06 | 1.91 | 1.81 | rejected |
| 2.17 | 0.00 | 1 | 2 | 0.75 | 1.39 | 1.86 | rejected |
| 2.17 | 0.50 | 1 | 2 | 0.75 | 2.02 | 2.70 | rejected |
| 2.17 | -0.06 | 1 | 2 | 0.75 | 1.86 | 2.49 | rejected |
| 2.18 | 0.24 | 1 | 2 | 0.75 | 1.51 | 2.02 | rejected |
| 2.18 | 0.28 | 1 | 2 | 1.06 | 2.05 | 1.94 | rejected |
| 2.18 | 0.07 | 1 | 2 | 0.75 | 1.77 | 2.36 | rejected |
| 2.19 | 0.24 | 1 | 2 | 1.25 | 1.49 | 1.20 | rejected |
| 2.19 | 0.27 | 1 | 2 | 1.00 | 1.95 | 1.95 | rejected |
| 2.19 | 0.17 | 1 | 2 | 1.46 | 1.92 | 1.32 | rejected |
| 2.19 | 0.23 | 1 | 2 | 1.50 | 1.84 | 1.23 | rejected |
| 2.19 | 0.14 | 1 | 2 | 1.67 | 1.54 | 1.08 | rejected |
| 2.20 | -0.02 | 1 | 2 | 0.75 | 2.08 | 2.79 | rejected |
| 2.20 | -0.01 | 1 | 2 | 1.25 | 1.46 | 1.17 | rejected |
| 2.20 | 0.20 | 1 | 2 | 1.00 | 2.02 | 2.02 | rejected |
| 2.20 | 0.28 | 1 | 2 | 0.75 | 1.68 | 2.24 | rejected |
| 2.20 | 0.19 | 1 | 2 | 0.75 | 2.04 | 2.73 | rejected |
| 2.20 | 0.38 | 1 | 2 | 0.75 | 1.92 | 2.57 | rejected |
| 2.21 | 0.20 | 1 | 2 | 0.75 | 1.74 | 2.33 | rejected |
| 2.21 | 0.19 | 1 | 2 | 0.75 | 1.85 | 2.48 | rejected |
| 2.21 | 0.28 | 1 | 2 | 1.67 | 1.92 | 1.15 | rejected |
| 2.21 | 0.49 | 1 | 2 | 0.75 | 1.93 | 2.58 | rejected |
| 2.22 | -0.21 | 1 | 2 | 0.75 | 1.08 | 1.45 | rejected |
| 2.22 | 0.35 | 1 | 2 | 0.75 | 1.91 | 2.55 | rejected |
| 2.22 | 0.12 | 1 | 2 | 0.75 | 1.83 | 2.45 | rejected |
| 2.22 | 0.26 | 1 | 2 | 0.75 | 1.29 | 1.72 | rejected |
| 2.22 | 0.04 | 1 | 2 | 1.00 | 1.79 | 1.79 | rejected |
| 2.22 | 0.09 | 1 | 2 | 0.75 | 1.91 | 2.55 | rejected |
| 2.23 | -0.18 | 1 | 2 | 0.75 | 1.43 | 1.91 | rejected |
| 2.23 | 0.40 | 1 | 2 | 0.75 | 1.66 | 2.22 | rejected |
| 2.23 | 0.19 | 1 | 2 | 1.00 | 1.98 | 1.98 | rejected |
| 2.23 | -0.14 | 1 | 2 | 1.25 | 1.40 | 1.12 | rejected |
| 2.23 | 0.33 | 1 | 2 | 0.75 | 1.96 | 2.63 | rejected |
| 2.24 | 0.19 | 1 | 1 | 1.06 | 1.00 | 1.06 | rejected |
| 2.24 | 0.23 | 1 | 1 | 1.50 | 1.25 | 1.20 | rejected |
| 2.24 | 0.16 | 1 | 1 | 0.75 | 1.06 | 1.41 | rejected |
| 2.24 | 0.01 | 1 | 1 | 1.50 | 1.50 | 1.00 | rejected |
| 2.24 | 0.01 | 1 | 1 | 1.46 | 0.75 | 1.95 | rejected |
| 2.24 | 0.31 | 1 | 1 | 1.50 | 1.25 | 1.20 | rejected |
| 2.24 | -0.29 | 1 | 1 | 0.75 | 1.67 | 2.24 | rejected |
| 2.24 | -0.05 | 1 | 1 | 1.50 | 0.75 | 2.00 | rejected |
| 2.24 | 0.30 | 1 | 1 | 1.00 | 0.75 | 1.34 | rejected |
| 2.24 | 0.19 | 1 | 1 | 1.50 | 0.75 | 2.00 | rejected |
| 2.24 | 0.05 | 1 | 1 | 1.00 | 1.80 | 1.80 | rejected |
| 2.24 | 0.38 | 1 | 1 | 1.00 | 1.46 | 1.46 | rejected |
| 2.24 | -0.24 | 1 | 1 | 0.75 | 1.06 | 1.41 | rejected |
| 2.24 | 0.42 | 1 | 1 | 1.67 | 1.95 | 1.16 | rejected |
| 2.24 | -0.28 | 1 | 1 | 0.75 | 1.06 | 1.41 | rejected |
| 2.24 | 0.25 | 1 | 1 | 1.06 | 1.00 | 1.06 | rejected |
| 2.24 | 0.23 | 1 | 1 | 2.12 | 1.46 | 1.45 | rejected |
| 2.24 | 0.40 | 1 | 1 | 1.95 | 1.00 | 1.95 | rejected |
| 2.24 | 0.14 | 1 | 1 | 1.25 | 1.80 | 1.44 | rejected |
| 2.24 | -0.18 | 1 | 1 | 1.80 | 1.00 | 1.80 | rejected |

…and 2156 more.

**WRONGLY accepted false joins (accepted FALSE — over-merges to kill)** (0 sites):
_none._

**Correctly repaired (accepted REAL)** (507 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 0.00 | — | 1 | 1 | 1.67 | 0.75 | 2.24 | accepted |
| 0.00 | — | 1 | 1 | 1.00 | 1.25 | 1.25 | accepted |
| 0.19 | -0.08 | 1 | 2 | 1.46 | 1.36 | 1.07 | accepted |
| 0.20 | 0.01 | 1 | 2 | 0.75 | 1.77 | 2.36 | accepted |
| 0.33 | 0.24 | 1 | 2 | 0.75 | 1.38 | 1.85 | accepted |
| 0.37 | 0.66 | 1 | 2 | 0.75 | 1.92 | 2.56 | accepted |
| 0.43 | 0.01 | 1 | 2 | 0.75 | 1.79 | 2.40 | accepted |
| 0.46 | 0.04 | 1 | 2 | 0.75 | 2.02 | 2.70 | accepted |
| 0.48 | 0.74 | 1 | 2 | 0.75 | 1.80 | 2.40 | accepted |
| 0.49 | -0.17 | 1 | 2 | 1.25 | 1.51 | 1.21 | accepted |
| 0.63 | 0.78 | 1 | 2 | 1.46 | 1.41 | 1.04 | accepted |
| 0.71 | 0.69 | 1 | 2 | 1.25 | 1.80 | 1.44 | accepted |
| 0.71 | 0.28 | 1 | 2 | 0.75 | 1.42 | 1.90 | accepted |
| 0.73 | 0.26 | 1 | 2 | 0.75 | 1.88 | 2.51 | accepted |
| 0.77 | 0.59 | 1 | 2 | 1.00 | 1.14 | 1.14 | accepted |
| 0.79 | 0.41 | 1 | 2 | 1.00 | 1.52 | 1.52 | accepted |
| 0.81 | 0.24 | 1 | 2 | 1.00 | 1.56 | 1.56 | accepted |
| 0.82 | 0.08 | 1 | 2 | 0.75 | 1.77 | 2.37 | accepted |
| 0.84 | 0.74 | 1 | 2 | 0.75 | 1.37 | 1.84 | accepted |
| 0.84 | 0.10 | 1 | 2 | 1.46 | 1.66 | 1.14 | accepted |
| 0.85 | 0.44 | 1 | 2 | 1.25 | 1.68 | 1.35 | accepted |
| 0.88 | 0.33 | 1 | 2 | 1.46 | 1.20 | 1.22 | accepted |
| 0.91 | 0.85 | 1 | 2 | 0.75 | 1.43 | 1.92 | accepted |
| 0.91 | 0.24 | 1 | 2 | 0.75 | 1.90 | 2.53 | accepted |
| 0.93 | -0.09 | 1 | 2 | 1.06 | 1.55 | 1.46 | accepted |
| 0.94 | 0.23 | 1 | 2 | 0.75 | 1.79 | 2.39 | accepted |
| 0.94 | 0.60 | 1 | 2 | 1.50 | 1.48 | 1.01 | accepted |
| 0.95 | 0.12 | 1 | 2 | 0.75 | 1.87 | 2.50 | accepted |
| 0.95 | 0.03 | 1 | 2 | 0.75 | 1.46 | 1.95 | accepted |
| 0.96 | 0.13 | 1 | 2 | 0.75 | 1.42 | 1.89 | accepted |
| 0.96 | 0.64 | 1 | 2 | 1.06 | 1.98 | 1.87 | accepted |
| 0.96 | 0.42 | 1 | 2 | 1.46 | 1.23 | 1.19 | accepted |
| 0.97 | 0.03 | 1 | 2 | 0.75 | 1.91 | 2.55 | accepted |
| 0.98 | 0.23 | 1 | 2 | 0.75 | 1.49 | 2.00 | accepted |
| 0.99 | 0.42 | 1 | 2 | 0.75 | 1.19 | 1.59 | accepted |
| 1.00 | 0.11 | 1 | 2 | 0.75 | 1.92 | 2.56 | accepted |
| 1.00 | 0.39 | 1 | 1 | 0.75 | 1.25 | 1.67 | accepted |
| 1.00 | -0.33 | 1 | 1 | 0.75 | 1.46 | 1.95 | accepted |
| 1.00 | 0.13 | 1 | 1 | 1.00 | 0.75 | 1.34 | accepted |
| 1.00 | -0.04 | 1 | 1 | 1.06 | 0.75 | 1.41 | accepted |
| 1.00 | -0.48 | 1 | 1 | 0.75 | 1.80 | 2.41 | accepted |
| 1.00 | 0.16 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.00 | 0.09 | 1 | 1 | 1.46 | 0.75 | 1.95 | accepted |
| 1.00 | -0.43 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.00 | -0.10 | 1 | 1 | 1.25 | 0.75 | 1.67 | accepted |
| 1.00 | 0.33 | 1 | 2 | 0.75 | 1.67 | 2.23 | accepted |
| 1.01 | 0.59 | 1 | 2 | 1.46 | 1.96 | 1.34 | accepted |
| 1.04 | 0.38 | 1 | 2 | 0.75 | 1.68 | 2.25 | accepted |
| 1.04 | 0.19 | 1 | 2 | 1.06 | 1.99 | 1.89 | accepted |
| 1.04 | 0.06 | 1 | 2 | 0.75 | 1.46 | 1.95 | accepted |
| 1.04 | 0.08 | 1 | 2 | 1.06 | 2.04 | 1.92 | accepted |
| 1.05 | -0.16 | 1 | 2 | 0.75 | 1.39 | 1.86 | accepted |
| 1.06 | 0.24 | 1 | 2 | 1.95 | 1.53 | 1.27 | accepted |
| 1.09 | -0.20 | 1 | 2 | 1.25 | 1.43 | 1.15 | accepted |
| 1.10 | 0.62 | 1 | 2 | 0.75 | 1.39 | 1.86 | accepted |
| 1.10 | 0.13 | 1 | 2 | 1.50 | 1.34 | 1.12 | accepted |
| 1.10 | 0.04 | 1 | 2 | 0.75 | 1.90 | 2.54 | accepted |
| 1.11 | 0.05 | 1 | 2 | 0.75 | 1.66 | 2.21 | accepted |
| 1.11 | 0.13 | 1 | 2 | 0.75 | 1.93 | 2.58 | accepted |
| 1.12 | 0.73 | 1 | 2 | 0.75 | 1.61 | 2.15 | accepted |

…and 447 more.

**Correctly refused (rejected FALSE)** (42 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 2.19 | 0.39 | 1 | 2 | 1.00 | 1.97 | 1.97 | rejected |
| 2.24 | -0.01 | 1 | 1 | 1.25 | 1.00 | 1.25 | rejected |
| 2.45 | 0.17 | 1 | 1 | 1.95 | 1.25 | 1.56 | rejected |
| 2.71 | 0.10 | 1 | 2 | 1.67 | 2.01 | 1.20 | rejected |
| 3.12 | 0.15 | 1 | 2 | 0.75 | 2.02 | 2.70 | rejected |
| 3.13 | 0.55 | 1 | 2 | 1.25 | 1.99 | 1.59 | rejected |
| 3.16 | 0.62 | 1 | 1 | 1.25 | 2.34 | 1.87 | rejected |
| 3.40 | 0.30 | 1 | 2 | 1.00 | 1.92 | 1.92 | rejected |
| 3.67 | 0.79 | 1 | 2 | 1.06 | 1.95 | 1.84 | rejected |
| 3.74 | 0.43 | 1 | 1 | 0.75 | 1.67 | 2.24 | rejected |
| 3.76 | 0.10 | 1 | 2 | 0.75 | 2.05 | 2.75 | rejected |
| 4.46 | 0.64 | 1 | 2 | 1.25 | 1.95 | 1.56 | rejected |
| 4.52 | 0.38 | 1 | 2 | 1.25 | 1.95 | 1.56 | rejected |
| 4.64 | 0.36 | 1 | 2 | 1.00 | 2.00 | 2.00 | rejected |
| 4.66 | 0.37 | 1 | 2 | 1.67 | 2.09 | 1.25 | rejected |
| 5.26 | 0.35 | 1 | 2 | 1.25 | 2.04 | 1.63 | rejected |
| 5.71 | -0.14 | 1 | 2 | 0.75 | 2.04 | 2.72 | rejected |
| 6.00 | 0.25 | 1 | 2 | 1.25 | 1.97 | 1.58 | rejected |
| 6.40 | 0.17 | 1 | 2 | 1.80 | 2.16 | 1.20 | rejected |
| 6.46 | 0.25 | 1 | 2 | 1.46 | 2.12 | 1.46 | rejected |
| 6.50 | 0.03 | 1 | 2 | 1.67 | 1.92 | 1.15 | rejected |
| 7.00 | 0.82 | 1 | 3 | 1.46 | 1.95 | 1.34 | rejected |
| 7.00 | 0.46 | 1 | 2 | 1.25 | 2.01 | 1.61 | rejected |
| 7.01 | -0.12 | 1 | 2 | 2.13 | 2.01 | 1.06 | rejected |
| 7.10 | 0.05 | 1 | 2 | 0.75 | 1.92 | 2.57 | rejected |
| 7.20 | 0.08 | 1 | 2 | 1.80 | 2.01 | 1.12 | rejected |
| 7.87 | 0.70 | 1 | 1 | 1.80 | 1.25 | 1.44 | rejected |
| 8.34 | -0.20 | 1 | 2 | 0.75 | 1.86 | 2.49 | rejected |
| 8.42 | 0.18 | 1 | 2 | 2.24 | 2.04 | 1.10 | rejected |
| 8.58 | 0.40 | 1 | 2 | 1.67 | 1.92 | 1.15 | rejected |
| 8.68 | 0.23 | 1 | 2 | 1.46 | 2.03 | 1.39 | rejected |
| 9.22 | -0.06 | 1 | 2 | 1.00 | 1.86 | 1.86 | rejected |
| 9.75 | -0.10 | 1 | 2 | 2.00 | 1.95 | 1.03 | rejected |
| 9.88 | 0.29 | 1 | 2 | 1.80 | 2.04 | 1.13 | rejected |
| 9.95 | 0.48 | 1 | 1 | 1.80 | 1.50 | 1.20 | rejected |
| 10.35 | 0.30 | 1 | 2 | 1.95 | 1.85 | 1.05 | rejected |
| 10.83 | 0.82 | 1 | 2 | 1.67 | 1.92 | 1.15 | rejected |
| 12.08 | 0.52 | 1 | 3 | 1.46 | 1.80 | 1.24 | rejected |
| 12.97 | -0.09 | 1 | 2 | 1.95 | 2.03 | 1.04 | rejected |
| 13.08 | 0.23 | 1 | 2 | 0.75 | 1.92 | 2.56 | rejected |
| 14.84 | 0.45 | 1 | 2 | 1.50 | 1.87 | 1.25 | rejected |
| 14.90 | -0.05 | 1 | 2 | 1.67 | 2.03 | 1.22 | rejected |

_(12613 SplitSite(s) omitted: a label's dominant neuron is held-out, so no train-derivable verdict — excluded to keep the signal leak-free.)_


## SplitSite enumeration recall ceiling (what no policy change can reach)

Across **12** train neurons broken into ≥2 fragments (10042 fragments total), **10030** `merge_labels` repairs are needed to make them whole. Of those, **2614** (26.1%) are REACHABLE — the enumerator produced at least one REAL SplitSite bridging them, so a good policy CAN make them; **7416** are NOT — no enumerated SplitSite connects the pieces, so NO accept/reject change can repair them (an enumeration limit, like the MergeSite detector recall gap, not a policy bug). This 26.1% is your achievable-recall CEILING on train; the SplitSite audit's MISSED bucket lives BELOW it.

- **12** of those neurons have a fragment no enumerated site reaches; **6852** fragments are fully ISOLATED (no same-neuron SplitSite at all). To pull these into reach you must WIDEN the candidate stream via `ENUM_PARAMS` (raise `max_gap_um` for long true gaps; set `tip_to_shaft=True` if a partner is mid-shaft; raise `split_max_sites` if truncation is dropping them — see below), NOT tune your thresholds.


### Candidate-stream truncation (`split_max_sites` cap)

The scan found **15378** label pairs, all returned (cap `split_max_sites`=25000 not hit) — truncation is NOT limiting recall this run.



## Edits proposed

2193 edits: 2193× merge_labels

merge(406128277, 406200992), merge(525270419, 525198142), merge(416983886, 402196132), merge(1095213747, 1095213755), merge(608373682, 608735686), merge(789301408, 818876918), merge(684932360, 685004630), merge(140129350, 125342259), merge(685936914, 685864655), merge(304781224, 319423782), merge(634750481, 649321210), merge(744853823, 744853605), merge(817567298, 817567299), merge(597066931, 582133543), merge(433216823, 418574265), merge(680201300, 665340617), merge(713855477, 713855481), merge(715142096, 729711944), merge(110629650, 110702127), merge(169436145, 169435928), merge(522391432, 522391653), merge(541036216, 540963738), merge(481611665, 466896182), merge(478123885, 478123661), merge(375020421, 389735470), merge(212491811, 227207467), merge(541250420, 556037944), merge(349435747, 320004994), merge(168549199, 168549198), merge(491750708, 491823196), merge(502450387, 502377899), merge(584739861, 570024606), merge(140057309, 140130004), merge(979872360, 994659896), merge(211839807, 211912307), merge(612006453, 582071132), merge(98544748, 113260018), merge(493129531, 493129532), merge(110629212, 125343127), merge(359072841, 373860377)

…and 2153 more.


## Merges attributed to edits (the ones to fix)

Each row is a merge that an edit *created* (its class fuses >=2 raw labels). Guard these label pairs in the policy.

| GT skeleton | fused labels | merge location (world) |
|---|---|---|
| N010-789202-JT | 408366016+408583287 | (5298.08, 23203.71, 32973.0) |
| N022-789202-JG | 375815390+510142981+539429632+759141385 | (9686.6, 18381.35, 34773.0) |
| N022-789202-JG | 802201821+816917309 | (10493.69, 15797.76, 35954.0) |
| N022-789202-JG | 213369075+257803516+302456963+317316771 | (4105.77, 17394.74, 29998.0) |
| N023-789202-SP | 861787341+862294326+891579445+920791852+920792071 | (11444.4, 17603.43, 34770.0) |
| N023-789202-SP | 346243395+419820811+463095869 | (6911.52, 14460.34, 32114.0) |
| N021-789202-HP | 508189513+625187625 | (7398.47, 14971.22, 39323.0) |
| N021-789202-HP | 600610352+645191523 | (8310.28, 24930.09, 35830.0) |
| N021-789202-HP | 715142096+729711944+744428308 | (9755.42, 18128.53, 36164.0) |
| N006-789202-JG | 553856478+568354495+640988476+684410054 | (9329.8, 16202.43, 39438.0) |
| N006-789202-JG | 553856478+568354495+640988476+684410054 | (9513.81, 16355.77, 39492.0) |
| N006-789202-JG | 375827439+375899928+493195242+521318114+550458696+669774491 | (5029.55, 16824.02, 41598.0) |
| N006-789202-JG | 390767620+390914130 | (5188.88, 17227.19, 49467.0) |
| N006-789202-JG | 612144424+625914492+627583713 | (8084.38, 18302.81, 44231.0) |
| N006-789202-JG | 714349752+743562608 | (9421.81, 16655.72, 39851.0) |
| N018-789202-JM | 649466627+649466629+649612042+664399357 | (8537.67, 4214.23, 35047.0) |
| N018-789202-JM | 341308216+576607729+605684583+620607676+634823406+649538015 | (11160.91, 3921.76, 32145.0) |
| N018-789202-JM | 341308216+576607729+605684583+620607676+634823406+649538015 | (7541.34, 4986.17, 33434.0) |
| N016-789202-HP | 553856478+568354495+640988476+684410054 | (9156.27, 16277.23, 39252.0) |
| N016-789202-HP | 272636190+272636191+435019631 | (4556.07, 17546.58, 10412.0) |
| N016-789202-HP | 346657744+434584481 | (5151.48, 17541.35, 13882.0) |
| N016-789202-HP | 346657744+434584481 | (4998.14, 16061.8, 11923.0) |
| N016-789202-HP | 156096417+183991840+243436246 | (2618.0, 16578.67, 18204.0) |
| N016-789202-HP | 154846150+184203969 | (2510.29, 16055.82, 8992.0) |
| N005-789202-SP | 655484521+655557011+670272497+670272500 | (8680.54, 17026.72, 36444.0) |
| N005-789202-SP | 582071132+582071350+612006453 | (8356.66, 16226.36, 52629.0) |
| N005-789202-SP | 464488053+479058563+553504107+568075271+583008228 | (6787.35, 19507.84, 46619.0) |
| N013-789202-KS | 716209065+730852059 | (9419.56, 20733.06, 16872.0) |
| N013-789202-KS | 553856478+568354495+640988476+684410054 | (9139.06, 16247.31, 39273.0) |
| N013-789202-KS | 553856478+568354495+640988476+684410054 | (9272.96, 16243.57, 39402.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (8937.85, 19686.61, 16418.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (8696.25, 19948.41, 16661.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (8655.11, 19894.56, 18801.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (9250.52, 20015.73, 17155.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (8893.72, 20381.5, 17446.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (8454.64, 21719.68, 16549.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (8201.82, 21851.32, 16427.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (8486.81, 21775.03, 16578.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (9133.83, 20454.81, 19629.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (9612.55, 20828.06, 17618.0) |
| N013-789202-KS | 375149638+450969847+612405031+687068923+716208403+730488744 | (8147.22, 21306.03, 16546.0) |
| N013-789202-KS | 432801166+432873873 | (5633.19, 12738.44, 38887.0) |
| N013-789202-KS | 686490761+730780009 | (9679.87, 20528.86, 17257.0) |
| N013-789202-KS | 659384254+659529018 | (8539.17, 25325.78, 22632.0) |
| N013-789202-KS | 113185131+228287360 | (3651.74, 19187.7, 18865.0) |
| N013-789202-KS | 330642703+389720140 | (5292.1, 15072.2, 17806.0) |
| N013-789202-KS | 330642703+389720140 | (4443.12, 14749.81, 19213.0) |
| N013-789202-KS | 642632510+658438831+671917186+672716974+673080517+686778969 | (8519.72, 21315.76, 16503.0) |
| N013-789202-KS | 642632510+658438831+671917186+672716974+673080517+686778969 | (8693.26, 20765.23, 15765.0) |




## Image warm-start probe: does bridge_ratio separate REAL from FALSE splits?

HARNESS-measured (NOT your policy, NOT scored, never held-out): the harness read `gap_bridge_evidence` on a balanced, budget-capped sample of train SplitSites in the `gap_um` band where REAL and FALSE OVERLAP — i.e. where geometry alone CANNOT separate them, so image has the most decision value. Use this to decide, BEFORE writing any image rule, whether `bridge_ratio` is worth gating on, and roughly where the threshold sits. A high `bridge_ratio` (signal stays bright across the gap) should mark REAL splits you SHOULD merge.

**Separability: bridge_ratio AUC = 0.52** (1.0 = REAL always brighter than FALSE; 0.5 = no separation). Read: weak — bridge_ratio alone may not separate; do not over-invest.

**REAL splits — SHOULD merge** (12 probed):
| gap_um | bridge_ratio | bridge_min | endpoint_mean |
|---|---|---|---|
| 2.67 | 1.00 | 68.50 | 68.8 |
| 2.67 | 0.98 | 57.00 | 58.2 |
| 2.66 | 0.96 | 58.00 | 60.5 |
| 2.66 | 0.99 | 40.00 | 40.5 |
| 2.67 | 0.97 | 29.00 | 30.0 |
| 2.67 | 1.02 | 302.50 | 297.0 |
| 2.68 | 0.98 | 24.00 | 24.5 |
| 2.68 | 0.98 | 26.00 | 26.5 |
| 2.66 | 1.00 | 58.00 | 58.0 |
| 2.65 | 1.02 | 41.00 | 40.0 |
| 2.69 | 0.99 | 122.00 | 123.5 |
| 2.69 | 0.95 | 95.00 | 100.2 |

**FALSE joins — must NOT merge** (7 probed):
| gap_um | bridge_ratio | bridge_min | endpoint_mean |
|---|---|---|---|
| 2.71 | 1.00 | 271.00 | 271.0 |
| 2.45 | 0.99 | 227.00 | 229.5 |
| 2.24 | 0.98 | 26.00 | 26.5 |
| 3.12 | 0.92 | 92.00 | 100.5 |
| 3.13 | 1.00 | 91.50 | 91.8 |
| 2.19 | 0.97 | 173.00 | 179.2 |
| 3.16 | 0.99 | 172.00 | 174.5 |