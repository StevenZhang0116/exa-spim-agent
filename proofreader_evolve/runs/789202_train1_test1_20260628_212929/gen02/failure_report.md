# Candidate failure report (train split)

- Proposed **218 edits** from 5000 candidate sites.
- **Split-repair score (THE FITNESS the gate keeps on) = correct - false = 61 - 0 = 61** (train-classified merges; unscored=157). The gate accepts only if this BEATS the parent AND false == 0 — a single false merge (fusing two different neurons) rejects the candidate outright.
- Edge Accuracy (secondary diagnostic, NOT the bar; = 100 - %Split - %Omit - %Merged): baseline 79.4094 -> candidate 79.1307.
- Merge-error component: %Merged Edges baseline 17.6169 -> candidate 17.8966; # Merges baseline 9.88 -> candidate 9.88.
- Over-split watchdog: %Split Edges baseline 0.3108 -> candidate 0.3095 (a merge repair that drives this UP is over-splitting a real neuron).


### Split-repair attribution — most discriminative feature axes (where your repairs vs. mistakes come from)

**deg_b** (quantile buckets) — correct / false / unscored (n):

  - deg_b [1,inf): 61 / 0 / 157  (n=218)

**gap_um** (quantile buckets) — correct / false / unscored (n):

  - gap_um [-inf,1.41421): 11 / 0 / 32  (n=43)

  - gap_um [1.41421,1.73205): 28 / 0 / 70  (n=98)

  - gap_um [1.73205,inf): 22 / 0 / 55  (n=77)

**rad_ratio** (quantile buckets) — correct / false / unscored (n):

  - rad_ratio [-inf,1.41384): 20 / 0 / 49  (n=69)

  - rad_ratio [1.41384,1.94648): 16 / 0 / 51  (n=67)

  - rad_ratio [1.94648,inf): 25 / 0 / 57  (n=82)

Read: a bucket with high correct and zero false is a regime worth EXTENDING; one with few correct but >0 false (or mostly unscored) is a regime to TIGHTEN or prune — do not spend more generations micro-tuning a low-yield region of feature space.

- Largest fused class = **5 raw labels** (0 class(es) fuse >5 labels). OK (no chaining)


## ENUM_PARAMS rail sensitivity (candidate-stream knobs)

The candidate stream you reason over is shaped by `ENUM_PARAMS` (module-level dict in heuristics.py). Each knob is CLAMPED to a safety rail; `requested` is what your policy asked for ('—' = not set, framework default used), `in_effect` is what actually shaped THIS run's stream. If a knob is `clamped`, your request had NO effect past the rail — stop tuning it. If it is `[AT … rail]`, the stream is already at its widening/narrowing limit in that direction.

| knob | requested | in_effect | rail [lo, hi] | status |
|---|---|---|---|---|
| max_gap_um | 15.0 | 15.0 | [1.0, 40.0] | ok |
| split_max_sites | — | 5000 | [100, 50000] | default |
| tip_to_shaft | True | True | — | ok |
| split_alt_per_pair | — | 1 | [1, 10] | default [AT lo rail] |
| min_arm_cable_um | 10.0 | 10.0 | [2.0, 50.0] | ok |
| seed_depth_um | 8.0 | 8.0 | [2.0, 30.0] | ok |
| merge_max_sites | — | 5000 | [100, 50000] | default |
| max_per_label | 8 | 8 | [1, 100] | ok |

## Per-skeleton delta (candidate - baseline)

| GT skeleton | dEdgeAcc | d%MergedEdges | d#Merges | d%SplitEdges | d#Splits | d%OmitEdges |
|---|---|---|---|---|---|---|
| N005-789202-SP | -0.090 | +0.095 | +0 | -0.002 | -7 | -0.001 |
| N006-789202-JG | -0.640 | +0.641 | +0 | -0.001 | -7 | -0.000 |
| N007-789202-PP | +0.000 | +0.000 | +0 | -0.003 | -4 | +0.000 |
| N010-789202-JT | +0.000 | +0.000 | +0 | -0.002 | -5 | -0.001 |
| N011-789202-IG | +0.000 | +0.000 | +0 | -0.003 | -4 | -0.003 |
| N013-789202-KS | +0.000 | +0.000 | +0 | -0.000 | -4 | +0.000 |
| N016-789202-HP | -0.030 | +0.033 | +0 | -0.001 | -4 | +0.000 |
| N018-789202-JM | -0.460 | +0.458 | +0 | -0.001 | -7 | +0.000 |
| N020-789202-PP | +0.000 | +0.000 | +0 | -0.004 | -3 | +0.000 |
| N021-789202-HP | +0.000 | +0.000 | +0 | -0.001 | -3 | +0.000 |
| N022-789202-JG | -1.130 | +1.134 | +0 | -0.002 | -9 | -0.001 |
| N023-789202-SP | +0.000 | +0.000 | +0 | -0.001 | -4 | -0.007 |

_Note: a positive `d#Splits` paired with a negative `d%OmitEdges` is the footprint of your MERGES healing background (omit) nodes into labels (the skeleton then crosses more distinct labels) — not over-splitting by you. `# Splits` / Edge Accuracy are GT-side diagnostics, NOT the gate._


## Train-side FALSE merges (precision alert — NOT a gate reject)

_no train-side false merges — your merges did not fuse two different neurons on any train skeleton._

## Skeletons with an Edge-Accuracy dip (secondary diagnostic)

N005-789202-SP, N006-789202-JG, N016-789202-HP, N018-789202-JM, N022-789202-JG


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

Confusion summary: REAL 1126 (accepted 61 / MISSED 1065); FALSE 7 (WRONGLY accepted 0 / correctly rejected 7).


**MISSED real splits (rejected REAL — raise recall)** (1065 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 0.19 | -0.08 | 1 | 2 | 1.46 | 1.36 | 1.07 | rejected |
| 0.20 | 0.01 | 1 | 2 | 0.75 | 1.77 | 2.36 | rejected |
| 0.33 | 0.24 | 1 | 2 | 0.75 | 1.38 | 1.85 | rejected |
| 0.37 | 0.66 | 1 | 2 | 0.75 | 1.92 | 2.56 | rejected |
| 0.43 | 0.01 | 1 | 2 | 0.75 | 1.79 | 2.40 | rejected |
| 0.46 | 0.04 | 1 | 2 | 0.75 | 2.02 | 2.70 | rejected |
| 0.48 | 0.74 | 1 | 2 | 0.75 | 1.80 | 2.40 | rejected |
| 0.49 | -0.17 | 1 | 2 | 1.25 | 1.51 | 1.21 | rejected |
| 0.63 | 0.78 | 1 | 2 | 1.46 | 1.41 | 1.04 | rejected |
| 0.71 | 0.69 | 1 | 2 | 1.25 | 1.80 | 1.44 | rejected |
| 0.71 | 0.28 | 1 | 2 | 0.75 | 1.42 | 1.90 | rejected |
| 0.73 | 0.26 | 1 | 2 | 0.75 | 1.88 | 2.51 | rejected |
| 0.77 | 0.59 | 1 | 2 | 1.00 | 1.14 | 1.14 | rejected |
| 0.79 | 0.41 | 1 | 2 | 1.00 | 1.52 | 1.52 | rejected |
| 0.81 | 0.24 | 1 | 2 | 1.00 | 1.56 | 1.56 | rejected |
| 0.82 | 0.08 | 1 | 2 | 0.75 | 1.77 | 2.37 | rejected |
| 0.84 | 0.74 | 1 | 2 | 0.75 | 1.37 | 1.84 | rejected |
| 0.84 | 0.10 | 1 | 2 | 1.46 | 1.66 | 1.14 | rejected |
| 0.85 | 0.44 | 1 | 2 | 1.25 | 1.68 | 1.35 | rejected |
| 0.88 | 0.33 | 1 | 2 | 1.46 | 1.20 | 1.22 | rejected |
| 0.91 | 0.85 | 1 | 2 | 0.75 | 1.43 | 1.92 | rejected |
| 0.91 | 0.24 | 1 | 2 | 0.75 | 1.90 | 2.53 | rejected |
| 0.93 | -0.09 | 1 | 2 | 1.06 | 1.55 | 1.46 | rejected |
| 0.94 | 0.23 | 1 | 2 | 0.75 | 1.79 | 2.39 | rejected |
| 0.94 | 0.60 | 1 | 2 | 1.50 | 1.48 | 1.01 | rejected |
| 0.95 | 0.12 | 1 | 2 | 0.75 | 1.87 | 2.50 | rejected |
| 0.95 | 0.03 | 1 | 2 | 0.75 | 1.46 | 1.95 | rejected |
| 0.96 | 0.13 | 1 | 2 | 0.75 | 1.42 | 1.89 | rejected |
| 0.96 | 0.64 | 1 | 2 | 1.06 | 1.98 | 1.87 | rejected |
| 0.96 | 0.42 | 1 | 2 | 1.46 | 1.23 | 1.19 | rejected |
| 0.97 | 0.03 | 1 | 2 | 0.75 | 1.91 | 2.55 | rejected |
| 0.98 | 0.23 | 1 | 2 | 0.75 | 1.49 | 2.00 | rejected |
| 0.99 | 0.42 | 1 | 2 | 0.75 | 1.19 | 1.59 | rejected |
| 1.00 | 0.11 | 1 | 2 | 0.75 | 1.92 | 2.56 | rejected |
| 1.00 | 0.33 | 1 | 2 | 0.75 | 1.67 | 2.23 | rejected |
| 1.01 | 0.59 | 1 | 2 | 1.46 | 1.96 | 1.34 | rejected |
| 1.04 | 0.38 | 1 | 2 | 0.75 | 1.68 | 2.25 | rejected |
| 1.04 | 0.19 | 1 | 2 | 1.06 | 1.99 | 1.89 | rejected |
| 1.04 | 0.06 | 1 | 2 | 0.75 | 1.46 | 1.95 | rejected |
| 1.04 | 0.08 | 1 | 2 | 1.06 | 2.04 | 1.92 | rejected |
| 1.05 | -0.16 | 1 | 2 | 0.75 | 1.39 | 1.86 | rejected |
| 1.06 | 0.24 | 1 | 2 | 1.95 | 1.53 | 1.27 | rejected |
| 1.09 | -0.20 | 1 | 2 | 1.25 | 1.43 | 1.15 | rejected |
| 1.10 | 0.62 | 1 | 2 | 0.75 | 1.39 | 1.86 | rejected |
| 1.10 | 0.13 | 1 | 2 | 1.50 | 1.34 | 1.12 | rejected |
| 1.10 | 0.04 | 1 | 2 | 0.75 | 1.90 | 2.54 | rejected |
| 1.11 | 0.05 | 1 | 2 | 0.75 | 1.66 | 2.21 | rejected |
| 1.11 | 0.13 | 1 | 2 | 0.75 | 1.93 | 2.58 | rejected |
| 1.12 | 0.73 | 1 | 2 | 0.75 | 1.61 | 2.15 | rejected |
| 1.12 | 0.46 | 1 | 2 | 1.25 | 1.48 | 1.18 | rejected |
| 1.13 | 0.36 | 1 | 2 | 1.46 | 1.72 | 1.18 | rejected |
| 1.14 | 0.13 | 1 | 2 | 0.75 | 1.68 | 2.24 | rejected |
| 1.17 | 0.63 | 1 | 2 | 1.00 | 1.47 | 1.47 | rejected |
| 1.17 | 0.55 | 1 | 2 | 0.75 | 2.01 | 2.69 | rejected |
| 1.17 | 0.19 | 1 | 2 | 0.75 | 1.65 | 2.21 | rejected |
| 1.18 | 0.69 | 1 | 2 | 0.75 | 1.97 | 2.63 | rejected |
| 1.19 | 0.08 | 1 | 2 | 0.75 | 1.50 | 2.01 | rejected |
| 1.19 | 0.71 | 1 | 2 | 1.00 | 1.47 | 1.47 | rejected |
| 1.21 | 0.43 | 1 | 2 | 1.25 | 1.85 | 1.48 | rejected |
| 1.21 | -0.05 | 1 | 2 | 1.06 | 1.63 | 1.54 | rejected |

…and 1005 more.

**WRONGLY accepted false joins (accepted FALSE — over-merges to kill)** (0 sites):
_none._

**Correctly repaired (accepted REAL)** (61 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 0.00 | — | 1 | 1 | 1.67 | 0.75 | 2.24 | accepted |
| 0.00 | — | 1 | 1 | 1.00 | 1.25 | 1.25 | accepted |
| 1.00 | 0.39 | 1 | 1 | 0.75 | 1.25 | 1.67 | accepted |
| 1.00 | -0.33 | 1 | 1 | 0.75 | 1.46 | 1.95 | accepted |
| 1.00 | 0.13 | 1 | 1 | 1.00 | 0.75 | 1.34 | accepted |
| 1.00 | -0.04 | 1 | 1 | 1.06 | 0.75 | 1.41 | accepted |
| 1.00 | -0.48 | 1 | 1 | 0.75 | 1.80 | 2.41 | accepted |
| 1.00 | 0.16 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.00 | 0.09 | 1 | 1 | 1.46 | 0.75 | 1.95 | accepted |
| 1.00 | -0.43 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.00 | -0.10 | 1 | 1 | 1.25 | 0.75 | 1.67 | accepted |
| 1.41 | -0.55 | 1 | 1 | 0.75 | 1.25 | 1.67 | accepted |
| 1.41 | -0.36 | 1 | 1 | 1.00 | 2.00 | 2.00 | accepted |
| 1.41 | -0.46 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.41 | -0.14 | 1 | 1 | 1.67 | 0.75 | 2.24 | accepted |
| 1.41 | 0.34 | 1 | 1 | 1.46 | 1.06 | 1.38 | accepted |
| 1.41 | -0.08 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.41 | -0.20 | 1 | 1 | 1.06 | 0.75 | 1.41 | accepted |
| 1.41 | -0.53 | 1 | 1 | 1.67 | 0.75 | 2.24 | accepted |
| 1.41 | 0.22 | 1 | 1 | 1.80 | 0.75 | 2.41 | accepted |
| 1.41 | 0.01 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.41 | -0.59 | 1 | 1 | 0.75 | 0.75 | 1.00 | accepted |
| 1.41 | -0.19 | 1 | 1 | 1.06 | 0.75 | 1.41 | accepted |
| 1.41 | -0.10 | 1 | 1 | 1.06 | 1.50 | 1.41 | accepted |
| 1.41 | -0.40 | 1 | 1 | 0.75 | 1.95 | 2.60 | accepted |
| 1.41 | 0.75 | 1 | 1 | 1.06 | 1.00 | 1.06 | accepted |
| 1.41 | -0.16 | 1 | 1 | 1.00 | 1.25 | 1.25 | accepted |
| 1.41 | -0.62 | 1 | 1 | 0.75 | 1.25 | 1.67 | accepted |
| 1.41 | -0.73 | 1 | 1 | 1.00 | 0.75 | 1.34 | accepted |
| 1.41 | 0.85 | 1 | 1 | 0.75 | 1.46 | 1.95 | accepted |
| 1.41 | 0.01 | 1 | 1 | 1.50 | 0.75 | 2.00 | accepted |
| 1.41 | -0.36 | 1 | 1 | 0.75 | 1.46 | 1.95 | accepted |
| 1.41 | 0.36 | 1 | 1 | 1.00 | 0.75 | 1.34 | accepted |
| 1.41 | -0.66 | 1 | 1 | 0.75 | 1.50 | 2.00 | accepted |
| 1.41 | 0.62 | 1 | 1 | 1.25 | 1.25 | 1.00 | accepted |
| 1.41 | -0.01 | 1 | 1 | 1.06 | 0.75 | 1.41 | accepted |
| 1.41 | -0.22 | 1 | 1 | 0.75 | 1.25 | 1.67 | accepted |
| 1.41 | -0.40 | 1 | 1 | 0.75 | 1.80 | 2.41 | accepted |
| 1.41 | -0.48 | 1 | 1 | 0.75 | 1.80 | 2.41 | accepted |
| 1.73 | 0.25 | 1 | 1 | 1.67 | 0.75 | 2.24 | accepted |
| 1.73 | 0.73 | 1 | 1 | 0.75 | 2.24 | 3.00 | accepted |
| 1.73 | -0.48 | 1 | 1 | 0.75 | 1.25 | 1.67 | accepted |
| 1.73 | 0.41 | 1 | 1 | 1.00 | 1.80 | 1.80 | accepted |
| 1.73 | -0.10 | 1 | 1 | 1.06 | 0.75 | 1.41 | accepted |
| 1.73 | 0.54 | 1 | 1 | 1.46 | 0.75 | 1.95 | accepted |
| 1.73 | 0.00 | 1 | 1 | 1.80 | 0.75 | 2.41 | accepted |
| 1.73 | 0.31 | 1 | 1 | 1.95 | 1.06 | 1.84 | accepted |
| 1.73 | 0.56 | 1 | 1 | 0.75 | 1.46 | 1.95 | accepted |
| 1.73 | 0.75 | 1 | 1 | 0.75 | 1.50 | 2.00 | accepted |
| 1.73 | -0.02 | 1 | 1 | 1.67 | 0.75 | 2.24 | accepted |
| 1.73 | -0.75 | 1 | 1 | 0.75 | 1.06 | 1.41 | accepted |
| 1.73 | 0.02 | 1 | 1 | 1.00 | 0.75 | 1.34 | accepted |
| 1.73 | -0.27 | 1 | 1 | 0.75 | 1.80 | 2.41 | accepted |
| 1.73 | 0.86 | 1 | 1 | 0.75 | 1.95 | 2.60 | accepted |
| 1.73 | 0.70 | 1 | 1 | 1.06 | 1.67 | 1.58 | accepted |
| 1.73 | -0.12 | 1 | 1 | 1.06 | 1.06 | 1.00 | accepted |
| 2.24 | 0.96 | 1 | 1 | 1.50 | 0.75 | 2.00 | accepted |
| 2.74 | 0.99 | 1 | 2 | 1.95 | 2.06 | 1.06 | accepted |
| 2.91 | 0.99 | 1 | 2 | 1.95 | 1.79 | 1.09 | accepted |
| 3.00 | 0.98 | 1 | 1 | 1.67 | 1.67 | 1.00 | accepted |

…and 1 more.

**Correctly refused (rejected FALSE)** (7 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 2.19 | 0.39 | 1 | 2 | 1.00 | 1.97 | 1.97 | rejected |
| 2.24 | -0.01 | 1 | 1 | 1.25 | 1.00 | 1.25 | rejected |
| 2.45 | 0.17 | 1 | 1 | 1.95 | 1.25 | 1.56 | rejected |
| 2.71 | 0.10 | 1 | 2 | 1.67 | 2.01 | 1.20 | rejected |
| 3.12 | 0.15 | 1 | 2 | 0.75 | 2.02 | 2.70 | rejected |
| 3.13 | 0.55 | 1 | 2 | 1.25 | 1.99 | 1.59 | rejected |
| 3.16 | 0.62 | 1 | 1 | 1.25 | 2.34 | 1.87 | rejected |

_(3867 SplitSite(s) omitted: a label's dominant neuron is held-out, so no train-derivable verdict — excluded to keep the signal leak-free.)_


## SplitSite enumeration recall ceiling (what no policy change can reach)

Across **12** train neurons broken into ≥2 fragments (10042 fragments total), **10030** `merge_labels` repairs are needed to make them whole. Of those, **1124** (11.2%) are REACHABLE — the enumerator produced at least one REAL SplitSite bridging them, so a good policy CAN make them; **8906** are NOT — no enumerated SplitSite connects the pieces, so NO accept/reject change can repair them (an enumeration limit, like the MergeSite detector recall gap, not a policy bug). This 11.2% is your achievable-recall CEILING on train; the SplitSite audit's MISSED bucket lives BELOW it.

- **12** of those neurons have a fragment no enumerated site reaches; **8294** fragments are fully ISOLATED (no same-neuron SplitSite at all). To pull these into reach you must WIDEN the candidate stream via `ENUM_PARAMS` (raise `max_gap_um` for long true gaps; set `tip_to_shaft=True` if a partner is mid-shaft; raise `split_max_sites` if truncation is dropping them — see below), NOT tune your thresholds.


### Candidate-stream truncation (`split_max_sites` cap)

The scan found **15378** label pairs within `max_gap_um`=15.0 µm but the `split_max_sites`=5000 cap kept only the **5000** closest — **10378** pairs (gaps 3.32–15.00 µm) were DROPPED before the policy saw them. Any real split among them is unreachable until you raise `split_max_sites` (rail 100–50000). Note the dropped pairs are the FARTHEST gaps (most likely false joins), so widen with care.



## Edits proposed

218 edits: 218× merge_labels

merge(406128277, 406200992), merge(525270419, 525198142), merge(416983886, 402196132), merge(1095213747, 1095213755), merge(608373682, 608735686), merge(140057309, 140130004), merge(688745595, 688600835), merge(99007273, 99007280), merge(873325354, 873325355), merge(244307425, 258950231), merge(363570665, 363498394), merge(437868394, 437868410), merge(862441493, 862513982), merge(806109660, 806109217), merge(901511095, 886868105), merge(615835231, 601192669), merge(349342888, 349270182), merge(464488053, 568075271), merge(685134715, 685206982), merge(864243426, 849527953), merge(438879945, 202641797), merge(718465639, 658952170), merge(539574392, 569440956), merge(555823993, 555824226), merge(358333280, 343618231), merge(245481485, 245408560), merge(612644619, 627287397), merge(227702185, 257277261), merge(483691138, 483618614), merge(724783573, 724783574), merge(815692040, 801049684), merge(878683409, 878755900), merge(1204978736, 1204906248), merge(571235882, 512592502), merge(640352064, 625854266), merge(307163495, 307163743), merge(712977938, 713124011), merge(360376330, 360376552), merge(846135634, 846135632), merge(642994955, 657710221)

…and 178 more.


## Merges attributed to edits (the ones to fix)

Each row is a merge that an edit *created* (its class fuses >=2 raw labels). Guard these label pairs in the policy.

| GT skeleton | fused labels | merge location (world) |
|---|---|---|
| N022-789202-JG | 375815390+510142981 | (9686.6, 18381.35, 34773.0) |
| N022-789202-JG | 213369075+257803516+302456963 | (4105.77, 17394.74, 29998.0) |
| N006-789202-JG | 553856478+684410054 | (9329.8, 16202.43, 39438.0) |
| N006-789202-JG | 553856478+684410054 | (9513.81, 16355.77, 39492.0) |
| N006-789202-JG | 375899928+493195242 | (5029.55, 16824.02, 41598.0) |
| N018-789202-JM | 341308216+605684583+649538015+678968768+708110009 | (11160.91, 3921.76, 32145.0) |
| N018-789202-JM | 341308216+605684583+649538015+678968768+708110009 | (7541.34, 4986.17, 33434.0) |
| N016-789202-HP | 553856478+684410054 | (9156.27, 16277.23, 39252.0) |
| N016-789202-HP | 346657744+434584481 | (5151.48, 17541.35, 13882.0) |
| N016-789202-HP | 346657744+434584481 | (4998.14, 16061.8, 11923.0) |
| N005-789202-SP | 464488053+553504107+568075271 | (6787.35, 19507.84, 46619.0) |
| N013-789202-KS | 553856478+684410054 | (9139.06, 16247.31, 39273.0) |
| N013-789202-KS | 553856478+684410054 | (9272.96, 16243.57, 39402.0) |
| N013-789202-KS | 642632510+658438831 | (8519.72, 21315.76, 16503.0) |
| N013-789202-KS | 642632510+658438831 | (8693.26, 20765.23, 15765.0) |




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