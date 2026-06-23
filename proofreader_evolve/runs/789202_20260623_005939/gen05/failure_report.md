# Candidate failure report (train split)

- Proposed **906 edits** from 5000 candidate sites.
- **Split-repair score (THE FITNESS the gate keeps on) = correct - false = 91 - 0 = 91** (train-classified merges; unscored=815). The gate accepts only if this BEATS the parent AND false == 0 — a single false merge (fusing two different neurons) rejects the candidate outright.
- Edge Accuracy (secondary diagnostic, NOT the bar; = 100 - %Split - %Omit - %Merged): baseline 72.9691 -> candidate 72.3571.
- Merge-error component: %Merged Edges baseline 24.3451 -> candidate 25.3672; # Merges baseline 12.53 -> candidate 12.29.
- Over-split watchdog: %Split Edges baseline 0.0908 -> candidate 0.2049 (a merge repair that drives this UP is over-splitting a real neuron).


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
| N013-789202-KS | -1.400 | +1.615 | -1 | +0.114 | +385 | -0.326 |
| N016-789202-HP | +0.660 | -0.440 | +1 | +0.123 | +400 | -0.344 |
| N018-789202-JM | -1.950 | +2.108 | -1 | +0.052 | +233 | -0.207 |
| N020-789202-PP | -0.110 | +0.592 | +0 | +0.189 | +223 | -0.671 |
| N021-789202-HP | +0.380 | -0.042 | +0 | +0.121 | +294 | -0.466 |
| N023-789202-SP | -0.190 | +2.134 | +0 | +0.193 | +366 | -2.130 |

## What FAILED the gate (split-repair currency)

_no false merges — the no-new-merge guard is satisfied; improve by RAISING correct repairs (recall)._

## Skeletons with an Edge-Accuracy dip (secondary diagnostic)

N013-789202-KS, N018-789202-JM, N020-789202-PP, N023-789202-SP


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

Confusion summary: REAL 540 (accepted 91 / MISSED 449); FALSE 3 (WRONGLY accepted 0 / correctly rejected 3).


**MISSED real splits (rejected REAL — raise recall)** (449 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 0.18 | 0.02 | 1 | 2 | 0.75 | 1.80 | 2.40 | rejected |
| 0.96 | 0.01 | 1 | 2 | 1.25 | 1.47 | 1.18 | rejected |
| 1.00 | 0.06 | 1 | 1 | 1.50 | 0.75 | 2.00 | rejected |
| 1.00 | -0.14 | 1 | 1 | 1.25 | 0.75 | 1.67 | rejected |
| 1.09 | 0.03 | 1 | 2 | 0.75 | 1.55 | 2.08 | rejected |
| 1.17 | -0.25 | 1 | 2 | 1.25 | 1.44 | 1.15 | rejected |
| 1.25 | -0.03 | 1 | 2 | 1.00 | 1.76 | 1.76 | rejected |
| 1.35 | 0.07 | 1 | 2 | 0.75 | 1.36 | 1.82 | rejected |
| 1.41 | -0.54 | 1 | 1 | 0.75 | 1.25 | 1.67 | rejected |
| 1.41 | -0.47 | 1 | 1 | 1.00 | 2.00 | 2.00 | rejected |
| 1.41 | -0.38 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.41 | -0.06 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.41 | -0.51 | 1 | 1 | 1.67 | 0.75 | 2.24 | rejected |
| 1.41 | 0.02 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.41 | -0.51 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.41 | -0.23 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.41 | -0.21 | 1 | 1 | 1.00 | 1.25 | 1.25 | rejected |
| 1.41 | -0.73 | 1 | 1 | 1.00 | 0.75 | 1.34 | rejected |
| 1.41 | -0.68 | 1 | 1 | 0.75 | 1.50 | 2.00 | rejected |
| 1.41 | -0.25 | 1 | 1 | 0.75 | 1.25 | 1.67 | rejected |
| 1.41 | -0.38 | 1 | 1 | 0.75 | 1.80 | 2.41 | rejected |
| 1.41 | -0.41 | 1 | 1 | 0.75 | 1.80 | 2.41 | rejected |
| 1.44 | 0.12 | 1 | 2 | 0.75 | 1.86 | 2.49 | rejected |
| 1.45 | 0.06 | 1 | 2 | 0.75 | 1.75 | 2.34 | rejected |
| 1.48 | 0.19 | 1 | 2 | 0.75 | 1.71 | 2.28 | rejected |
| 1.51 | -0.28 | 1 | 2 | 0.75 | 1.25 | 1.68 | rejected |
| 1.51 | 0.14 | 1 | 2 | 0.75 | 1.49 | 1.99 | rejected |
| 1.55 | 0.62 | 1 | 2 | 1.06 | 1.86 | 1.76 | rejected |
| 1.55 | 0.13 | 1 | 2 | 1.00 | 1.86 | 1.86 | rejected |
| 1.59 | 0.00 | 1 | 2 | 1.00 | 1.37 | 1.37 | rejected |
| 1.62 | -0.17 | 1 | 2 | 0.75 | 1.91 | 2.56 | rejected |
| 1.63 | 0.03 | 1 | 2 | 1.25 | 1.51 | 1.21 | rejected |
| 1.66 | -0.06 | 1 | 2 | 1.06 | 1.64 | 1.55 | rejected |
| 1.69 | 0.57 | 1 | 2 | 0.75 | 1.42 | 1.90 | rejected |
| 1.70 | -0.28 | 1 | 2 | 0.75 | 2.06 | 2.75 | rejected |
| 1.71 | -0.14 | 1 | 2 | 0.75 | 1.55 | 2.08 | rejected |
| 1.72 | -0.03 | 1 | 2 | 0.75 | 1.50 | 2.00 | rejected |
| 1.73 | 0.32 | 1 | 1 | 1.67 | 0.75 | 2.24 | rejected |
| 1.73 | 0.77 | 1 | 1 | 0.75 | 2.24 | 3.00 | rejected |
| 1.73 | -0.11 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.73 | -0.06 | 1 | 1 | 0.75 | 1.46 | 1.95 | rejected |
| 1.73 | -0.05 | 1 | 1 | 1.67 | 0.75 | 2.24 | rejected |
| 1.73 | -0.76 | 1 | 1 | 0.75 | 1.06 | 1.41 | rejected |
| 1.73 | -0.12 | 1 | 1 | 1.06 | 1.06 | 1.00 | rejected |
| 1.77 | 0.10 | 1 | 2 | 1.00 | 1.97 | 1.97 | rejected |
| 1.79 | 0.10 | 1 | 2 | 0.75 | 1.95 | 2.60 | rejected |
| 1.80 | 0.29 | 1 | 2 | 1.25 | 1.56 | 1.25 | rejected |
| 1.86 | 0.03 | 1 | 2 | 1.00 | 1.54 | 1.54 | rejected |
| 1.87 | 0.25 | 1 | 2 | 0.75 | 1.66 | 2.22 | rejected |
| 1.88 | -0.01 | 1 | 2 | 1.95 | 1.30 | 1.49 | rejected |
| 1.88 | -0.06 | 1 | 2 | 0.75 | 1.94 | 2.60 | rejected |
| 1.89 | 0.53 | 1 | 2 | 0.75 | 1.63 | 2.18 | rejected |
| 1.89 | 0.02 | 1 | 2 | 0.75 | 1.27 | 1.70 | rejected |
| 1.93 | 0.39 | 1 | 2 | 1.00 | 1.72 | 1.72 | rejected |
| 1.93 | 0.32 | 1 | 2 | 1.50 | 1.65 | 1.11 | rejected |
| 1.94 | 0.76 | 1 | 2 | 1.00 | 1.72 | 1.72 | rejected |
| 1.94 | 0.07 | 1 | 2 | 0.75 | 1.24 | 1.66 | rejected |
| 1.94 | -0.06 | 1 | 2 | 0.75 | 1.73 | 2.31 | rejected |
| 1.95 | 0.36 | 1 | 2 | 1.00 | 1.88 | 1.88 | rejected |
| 1.96 | 0.57 | 1 | 2 | 0.75 | 1.87 | 2.50 | rejected |

…and 389 more.

**WRONGLY accepted false joins (accepted FALSE — over-merges to kill)** (0 sites):
_none._

**Correctly repaired (accepted REAL)** (91 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 0.73 | 0.18 | 1 | 2 | 1.06 | 2.04 | 1.92 | accepted |
| 0.91 | 0.71 | 1 | 2 | 0.75 | 1.87 | 2.50 | accepted |
| 0.94 | 0.90 | 1 | 2 | 1.46 | 1.22 | 1.19 | accepted |
| 0.97 | 0.60 | 1 | 2 | 1.50 | 1.48 | 1.01 | accepted |
| 1.32 | 0.43 | 1 | 2 | 0.75 | 1.40 | 1.88 | accepted |
| 1.34 | 0.38 | 1 | 2 | 0.75 | 1.33 | 1.78 | accepted |
| 1.38 | 0.52 | 1 | 2 | 0.75 | 1.67 | 2.23 | accepted |
| 1.41 | 0.68 | 1 | 2 | 1.25 | 1.84 | 1.47 | accepted |
| 1.41 | 0.20 | 1 | 1 | 1.80 | 0.75 | 2.41 | accepted |
| 1.41 | 0.63 | 1 | 1 | 1.25 | 1.25 | 1.00 | accepted |
| 1.44 | 0.97 | 1 | 2 | 1.00 | 1.22 | 1.22 | accepted |
| 1.49 | 0.16 | 1 | 2 | 1.00 | 1.85 | 1.85 | accepted |
| 1.49 | 0.26 | 1 | 2 | 1.46 | 1.51 | 1.04 | accepted |
| 1.65 | 0.78 | 1 | 2 | 0.75 | 2.01 | 2.69 | accepted |
| 1.83 | 0.94 | 1 | 2 | 0.75 | 1.70 | 2.27 | accepted |
| 1.87 | 0.67 | 1 | 2 | 0.75 | 1.69 | 2.26 | accepted |
| 1.91 | 0.82 | 1 | 2 | 1.25 | 1.49 | 1.19 | accepted |
| 1.95 | 0.81 | 1 | 2 | 0.75 | 1.59 | 2.12 | accepted |
| 2.00 | 0.93 | 1 | 1 | 2.13 | 0.75 | 2.85 | accepted |
| 2.00 | 0.81 | 1 | 1 | 2.13 | 1.80 | 1.19 | accepted |
| 2.00 | 0.82 | 1 | 1 | 2.24 | 1.46 | 1.54 | accepted |
| 2.16 | 0.92 | 1 | 2 | 0.75 | 1.46 | 1.95 | accepted |
| 2.17 | 0.80 | 1 | 2 | 1.06 | 1.80 | 1.70 | accepted |
| 2.22 | 0.75 | 1 | 2 | 1.50 | 1.39 | 1.07 | accepted |
| 2.22 | 0.66 | 1 | 2 | 1.95 | 1.41 | 1.38 | accepted |
| 2.22 | 0.93 | 1 | 2 | 0.75 | 1.83 | 2.44 | accepted |
| 2.24 | 0.91 | 1 | 1 | 1.80 | 2.12 | 1.18 | accepted |
| 2.24 | 0.85 | 1 | 1 | 1.50 | 1.46 | 1.03 | accepted |
| 2.24 | 0.74 | 1 | 1 | 2.24 | 1.50 | 1.50 | accepted |
| 2.24 | 0.77 | 1 | 1 | 1.67 | 1.46 | 1.15 | accepted |
| 2.24 | 0.87 | 1 | 1 | 0.75 | 1.46 | 1.95 | accepted |
| 2.35 | 0.68 | 1 | 2 | 0.75 | 1.61 | 2.15 | accepted |
| 2.35 | 0.87 | 1 | 2 | 0.75 | 1.56 | 2.09 | accepted |
| 2.45 | 0.69 | 1 | 1 | 1.25 | 1.46 | 1.17 | accepted |
| 2.45 | 0.76 | 1 | 1 | 0.75 | 1.50 | 2.00 | accepted |
| 2.45 | 0.68 | 1 | 1 | 1.00 | 1.67 | 1.67 | accepted |
| 2.45 | 0.87 | 1 | 1 | 1.06 | 1.50 | 1.41 | accepted |
| 2.45 | 0.93 | 1 | 1 | 0.75 | 2.12 | 2.83 | accepted |
| 2.45 | 0.94 | 1 | 1 | 1.95 | 0.75 | 2.60 | accepted |
| 2.45 | 0.94 | 1 | 1 | 1.25 | 2.12 | 1.69 | accepted |
| 2.45 | 0.81 | 1 | 1 | 1.06 | 1.25 | 1.18 | accepted |
| 2.49 | 0.72 | 1 | 2 | 0.75 | 2.03 | 2.72 | accepted |
| 2.60 | 0.91 | 1 | 2 | 0.75 | 1.90 | 2.54 | accepted |
| 2.68 | 0.78 | 1 | 2 | 1.67 | 1.92 | 1.15 | accepted |
| 2.72 | 0.66 | 1 | 2 | 0.75 | 1.94 | 2.60 | accepted |
| 2.75 | 0.67 | 1 | 2 | 0.75 | 1.63 | 2.18 | accepted |
| 2.80 | 0.84 | 1 | 2 | 1.80 | 1.53 | 1.18 | accepted |
| 2.83 | 0.79 | 1 | 1 | 1.06 | 1.80 | 1.70 | accepted |
| 2.83 | 0.72 | 1 | 1 | 0.75 | 1.95 | 2.60 | accepted |
| 2.83 | 0.95 | 1 | 1 | 1.80 | 1.50 | 1.20 | accepted |
| 2.83 | 0.81 | 1 | 1 | 1.46 | 0.75 | 1.95 | accepted |
| 2.83 | 0.82 | 1 | 1 | 1.46 | 1.80 | 1.24 | accepted |
| 2.84 | 0.88 | 1 | 2 | 0.75 | 1.41 | 1.88 | accepted |
| 2.92 | 0.61 | 1 | 2 | 0.75 | 1.86 | 2.49 | accepted |
| 2.93 | 0.84 | 1 | 2 | 1.46 | 1.35 | 1.08 | accepted |
| 3.00 | 0.83 | 1 | 1 | 1.67 | 1.67 | 1.00 | accepted |
| 3.00 | 0.69 | 1 | 1 | 2.24 | 1.06 | 2.12 | accepted |
| 3.00 | 0.77 | 1 | 1 | 1.95 | 1.67 | 1.16 | accepted |
| 3.00 | 0.75 | 1 | 1 | 1.25 | 2.12 | 1.69 | accepted |
| 3.00 | 0.81 | 1 | 1 | 2.00 | 1.50 | 1.34 | accepted |

…and 31 more.

**Correctly refused (rejected FALSE)** (3 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 2.20 | 0.44 | 1 | 2 | 1.00 | 1.97 | 1.97 | rejected |
| 2.45 | 0.19 | 1 | 1 | 1.95 | 1.25 | 1.56 | rejected |
| 3.00 | 0.06 | 1 | 1 | 1.00 | 1.67 | 1.67 | rejected |

_(4457 SplitSite(s) omitted: a label's dominant neuron is held-out, so no train-derivable verdict — excluded to keep the signal leak-free.)_


## Edits proposed

906 edits: 906× merge_labels

merge(801978891, 802051384), merge(565614391, 565469412), merge(502450387, 502377899), merge(348274169, 333558927), merge(366536142, 202641797), merge(375899928, 521318114), merge(99080420, 99080424), merge(136997232, 166646552), merge(452805935, 438163157), merge(552116304, 626199623), merge(300561522, 300489036), merge(364798597, 364798596), merge(487951931, 473236675), merge(649248717, 663746735), merge(937821511, 923178736), merge(671087406, 656372132), merge(710871816, 755235300), merge(389720140, 330642703), merge(350520945, 365308435), merge(307033194, 307033204), merge(341308216, 620607676), merge(363570665, 363498394), merge(437868394, 437868410), merge(901511095, 886868105), merge(349342888, 349270182), merge(438879945, 202641797), merge(724783573, 724783574), merge(483691138, 483618614), merge(878683409, 878755900), merge(571235882, 512592502), merge(640352064, 625854266), merge(801267792, 801267569), merge(331654259, 316938994), merge(741179881, 726537103), merge(801699872, 787129586), merge(241182298, 241254793), merge(302088384, 316803650), merge(136997205, 137069487), merge(817567298, 817567299), merge(731361673, 731289401)

…and 866 more.


## Merges attributed to edits (the ones to fix)

Each row is a merge that an edit *created* (its class fuses >=2 raw labels). Guard these label pairs in the policy.

| GT skeleton | fused labels | merge location (world) |
|---|---|---|
| N013-789202-KS | 245469874+404943484 | (4781.96, 15262.19, 19668.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (8937.85, 19686.61, 16418.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (8696.25, 19948.41, 16661.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (8655.11, 19894.56, 18801.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (9250.52, 20015.73, 17155.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (8893.72, 20381.5, 17446.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (8454.64, 21719.68, 16549.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (8201.82, 21851.32, 16427.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (8486.81, 21775.03, 16578.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (9133.83, 20454.81, 19629.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (9612.55, 20828.06, 17618.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (8147.22, 21306.03, 16546.0) |
| N013-789202-KS | 432801166+432873873 | (5633.19, 12738.44, 38887.0) |
| N013-789202-KS | 375149638+568620365+613127743+642413074+642994519+643212421+686490761+701420658+716208403+716718237+730780009+730924773 | (9679.87, 20528.86, 17257.0) |
| N013-789202-KS | 349999232+454026860+542461262+659384254+659601729+688454540 | (8539.17, 25325.78, 22632.0) |
| N013-789202-KS | 643649326+643649766 | (8805.46, 21209.54, 17779.0) |
| N013-789202-KS | 330642703+360218434+389720140 | (5292.1, 15072.2, 17806.0) |
| N013-789202-KS | 330642703+360218434+389720140 | (4443.12, 14749.81, 19213.0) |
| N013-789202-KS | 330642703+360218434+389720140 | (4830.58, 14850.04, 18670.0) |
| N013-789202-KS | 642632510+671917186 | (8519.72, 21315.76, 16503.0) |
| N013-789202-KS | 642632510+671917186 | (8693.26, 20765.23, 15765.0) |
| N023-789202-SP | 861787341+891579445 | (11444.4, 17603.43, 34770.0) |
| N018-789202-JM | 341308216+532670856+576607729+590604902+620035430+620607676+635177966+694120063+708109796+708254549 | (11160.91, 3921.76, 32145.0) |
| N018-789202-JM | 341308216+532670856+576607729+590604902+620035430+620607676+635177966+694120063+708109796+708254549 | (7541.34, 4986.17, 33434.0) |




## Image warm-start probe: does bridge_ratio separate REAL from FALSE splits?

HARNESS-measured (NOT your policy, NOT scored, never held-out): the harness read `gap_bridge_evidence` on a balanced, budget-capped sample of train SplitSites in the `gap_um` band where REAL and FALSE OVERLAP — i.e. where geometry alone CANNOT separate them, so image has the most decision value. Use this to decide, BEFORE writing any image rule, whether `bridge_ratio` is worth gating on, and roughly where the threshold sits. A high `bridge_ratio` (signal stays bright across the gap) should mark REAL splits you SHOULD merge.

**Separability: bridge_ratio AUC = 0.69** (1.0 = REAL always brighter than FALSE; 0.5 = no separation). Read: moderate.

**REAL splits — SHOULD merge** (12 probed):
| gap_um | bridge_ratio | bridge_min | endpoint_mean |
|---|---|---|---|
| 2.58 | 1.00 | 67.50 | 67.2 |
| 2.58 | 1.00 | 68.50 | 68.2 |
| 2.58 | 0.94 | 158.00 | 167.8 |
| 2.59 | 0.98 | 27.00 | 27.5 |
| 2.57 | 1.02 | 66.00 | 64.5 |
| 2.60 | 0.95 | 46.00 | 48.5 |
| 2.56 | 0.99 | 50.50 | 51.0 |
| 2.60 | 1.00 | 40.00 | 40.0 |
| 2.56 | 1.00 | 32.00 | 32.0 |
| 2.55 | 0.96 | 35.00 | 36.5 |
| 2.55 | 0.98 | 29.00 | 29.5 |
| 2.62 | 1.00 | 73.00 | 73.0 |

**FALSE joins — must NOT merge** (3 probed):
| gap_um | bridge_ratio | bridge_min | endpoint_mean |
|---|---|---|---|
| 2.45 | 0.99 | 227.00 | 229.5 |
| 2.20 | 0.97 | 173.00 | 179.2 |
| 3.00 | 0.97 | 259.00 | 265.8 |