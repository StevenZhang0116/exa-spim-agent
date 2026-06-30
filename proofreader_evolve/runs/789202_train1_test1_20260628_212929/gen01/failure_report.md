# Candidate failure report (train split)

- Proposed **25 edits** from 5000 candidate sites.
- **Split-repair score (THE FITNESS the gate keeps on) = correct - false = 5 - 0 = 5** (train-classified merges; unscored=20). The gate accepts only if this BEATS the parent AND false == 0 — a single false merge (fusing two different neurons) rejects the candidate outright.
- Edge Accuracy (secondary diagnostic, NOT the bar; = 100 - %Split - %Omit - %Merged): baseline 79.4094 -> candidate 79.3695.
- Merge-error component: %Merged Edges baseline 17.6169 -> candidate 17.6572; # Merges baseline 9.88 -> candidate 9.88.
- Over-split watchdog: %Split Edges baseline 0.3108 -> candidate 0.3107 (a merge repair that drives this UP is over-splitting a real neuron).


### Split-repair attribution — most discriminative feature axes (where your repairs vs. mistakes come from)

**colinear_cos** (quantile buckets) — correct / false / unscored (n):

  - colinear_cos [-inf,0.955143): 0 / 0 / 8  (n=8)

  - colinear_cos [0.955143,0.975591): 2 / 0 / 6  (n=8)

  - colinear_cos [0.975591,inf): 3 / 0 / 6  (n=9)

**deg_b** (quantile buckets) — correct / false / unscored (n):

  - deg_b [1,2): 3 / 0 / 9  (n=12)

  - deg_b [2,inf): 2 / 0 / 11  (n=13)

**gap_um** (quantile buckets) — correct / false / unscored (n):

  - gap_um [-inf,2.23607): 0 / 0 / 5  (n=5)

  - gap_um [2.23607,2.82843): 2 / 0 / 9  (n=11)

  - gap_um [2.82843,inf): 3 / 0 / 6  (n=9)

Read: a bucket with high correct and zero false is a regime worth EXTENDING; one with few correct but >0 false (or mostly unscored) is a regime to TIGHTEN or prune — do not spend more generations micro-tuning a low-yield region of feature space.

- Largest fused class = **2 raw labels** (0 class(es) fuse >5 labels). OK (no chaining)


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
| N005-789202-SP | -0.060 | +0.058 | +0 | -0.000 | -1 | +0.000 |
| N006-789202-JG | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N007-789202-PP | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N010-789202-JT | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N011-789202-IG | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N013-789202-KS | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N016-789202-HP | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N018-789202-JM | +0.000 | +0.000 | +0 | -0.000 | -1 | +0.000 |
| N020-789202-PP | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N021-789202-HP | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N022-789202-JG | -0.330 | +0.335 | +0 | -0.001 | -2 | +0.000 |
| N023-789202-SP | +0.000 | +0.000 | +0 | -0.000 | -1 | +0.000 |

_Note: a positive `d#Splits` paired with a negative `d%OmitEdges` is the footprint of your MERGES healing background (omit) nodes into labels (the skeleton then crosses more distinct labels) — not over-splitting by you. `# Splits` / Edge Accuracy are GT-side diagnostics, NOT the gate._


## Train-side FALSE merges (precision alert — NOT a gate reject)

_no train-side false merges — your merges did not fuse two different neurons on any train skeleton._

## Skeletons with an Edge-Accuracy dip (secondary diagnostic)

N005-789202-SP, N022-789202-JG


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

Confusion summary: REAL 1126 (accepted 5 / MISSED 1121); FALSE 7 (WRONGLY accepted 0 / correctly rejected 7).


**MISSED real splits (rejected REAL — raise recall)** (1121 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 0.00 | — | 1 | 1 | 1.67 | 0.75 | 2.24 | rejected |
| 0.00 | — | 1 | 1 | 1.00 | 1.25 | 1.25 | rejected |
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
| 1.00 | 0.39 | 1 | 1 | 0.75 | 1.25 | 1.67 | rejected |
| 1.00 | -0.33 | 1 | 1 | 0.75 | 1.46 | 1.95 | rejected |
| 1.00 | 0.13 | 1 | 1 | 1.00 | 0.75 | 1.34 | rejected |
| 1.00 | -0.04 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.00 | -0.48 | 1 | 1 | 0.75 | 1.80 | 2.41 | rejected |
| 1.00 | 0.16 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.00 | 0.09 | 1 | 1 | 1.46 | 0.75 | 1.95 | rejected |
| 1.00 | -0.43 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.00 | -0.10 | 1 | 1 | 1.25 | 0.75 | 1.67 | rejected |
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

…and 1061 more.

**WRONGLY accepted false joins (accepted FALSE — over-merges to kill)** (0 sites):
_none._

**Correctly repaired (accepted REAL)** (5 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 2.24 | 0.96 | 1 | 1 | 1.50 | 0.75 | 2.00 | accepted |
| 2.74 | 0.99 | 1 | 2 | 1.95 | 2.06 | 1.06 | accepted |
| 2.91 | 0.99 | 1 | 2 | 1.95 | 1.79 | 1.09 | accepted |
| 3.00 | 0.98 | 1 | 1 | 1.67 | 1.67 | 1.00 | accepted |
| 3.00 | 0.96 | 1 | 1 | 1.95 | 1.67 | 1.16 | accepted |

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

25 edits: 25× merge_labels

merge(140057309, 140130004), merge(688745595, 688600835), merge(757203452, 713130579), merge(56123143, 56195631), merge(69416674, 69416660), merge(464488053, 553504107), merge(255606972, 255606989), merge(429604198, 414961425), merge(430185205, 430112718), merge(70576695, 70576914), merge(229880373, 244668127), merge(229519009, 229519230), merge(715005889, 685647174), merge(256258928, 256258484), merge(788209252, 817566640), merge(775734138, 760873676), merge(272729687, 273020514), merge(554954761, 554954763), merge(847144780, 847144564), merge(315791212, 315790994), merge(419654601, 434297150), merge(460993472, 460993470), merge(629152404, 629806994), merge(375815390, 510142981), merge(634605281, 649320993)


## Merges attributed to edits (the ones to fix)

Each row is a merge that an edit *created* (its class fuses >=2 raw labels). Guard these label pairs in the policy.

| GT skeleton | fused labels | merge location (world) |
|---|---|---|
| N022-789202-JG | 375815390+510142981 | (9686.6, 18381.35, 34773.0) |
| N005-789202-SP | 464488053+553504107 | (6787.35, 19507.84, 46619.0) |




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