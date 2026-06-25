# Candidate failure report (train split)

- Proposed **51 edits** from 5000 candidate sites.
- **Split-repair score (THE FITNESS the gate keeps on) = correct - false = 3 - 0 = 3** (train-classified merges; unscored=48). The gate accepts only if this BEATS the parent AND false == 0 — a single false merge (fusing two different neurons) rejects the candidate outright.
- Edge Accuracy (secondary diagnostic, NOT the bar; = 100 - %Split - %Omit - %Merged): baseline 90.0449 -> candidate 90.0316.
- Merge-error component: %Merged Edges baseline 5.0241 -> candidate 5.0370; # Merges baseline 4.97 -> candidate 4.97.
- Over-split watchdog: %Split Edges baseline 0.4476 -> candidate 0.4475 (a merge repair that drives this UP is over-splitting a real neuron).

- Largest fused class = **2 raw labels** (0 class(es) fuse >5 labels). OK (no chaining)


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
| N005-789202-SP | -0.060 | +0.058 | +0 | -0.000 | -2 | +0.000 |
| N010-789202-JT | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N011-789202-IG | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N020-789202-PP | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N021-789202-HP | +0.000 | +0.000 | +0 | +0.000 | +0 | +0.000 |
| N023-789202-SP | +0.000 | +0.000 | +0 | -0.000 | -1 | +0.000 |

_Note: a positive `d#Splits` paired with a negative `d%OmitEdges` is the footprint of your MERGES healing background (omit) nodes into labels (the skeleton then crosses more distinct labels) — not over-splitting by you. `# Splits` / Edge Accuracy are GT-side diagnostics, NOT the gate._


## Train-side FALSE merges (precision alert — NOT a gate reject)

_no train-side false merges — your merges did not fuse two different neurons on any train skeleton._

## Skeletons with an Edge-Accuracy dip (secondary diagnostic)

N005-789202-SP


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

Confusion summary: REAL 490 (accepted 3 / MISSED 487); FALSE 1 (WRONGLY accepted 0 / correctly rejected 1).


**MISSED real splits (rejected REAL — raise recall)** (487 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 0.18 | 0.02 | 1 | 2 | 0.75 | 1.80 | 2.40 | rejected |
| 0.56 | 0.59 | 1 | 2 | 1.00 | 1.58 | 1.58 | rejected |
| 0.66 | 0.11 | 1 | 2 | 1.06 | 1.55 | 1.47 | rejected |
| 0.73 | 0.18 | 1 | 2 | 1.06 | 2.04 | 1.92 | rejected |
| 0.94 | 0.21 | 1 | 2 | 0.75 | 1.53 | 2.04 | rejected |
| 1.00 | 0.36 | 1 | 1 | 0.75 | 1.25 | 1.67 | rejected |
| 1.00 | -0.31 | 1 | 1 | 0.75 | 1.46 | 1.95 | rejected |
| 1.00 | -0.49 | 1 | 1 | 0.75 | 1.80 | 2.41 | rejected |
| 1.00 | 0.18 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.00 | 0.06 | 1 | 1 | 1.50 | 0.75 | 2.00 | rejected |
| 1.00 | -0.43 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.09 | 0.03 | 1 | 2 | 0.75 | 1.55 | 2.08 | rejected |
| 1.12 | 0.08 | 1 | 2 | 1.67 | 1.87 | 1.12 | rejected |
| 1.17 | 0.74 | 1 | 2 | 1.06 | 1.63 | 1.54 | rejected |
| 1.17 | -0.25 | 1 | 2 | 1.25 | 1.44 | 1.15 | rejected |
| 1.19 | 0.84 | 1 | 2 | 1.25 | 1.68 | 1.35 | rejected |
| 1.22 | 0.03 | 1 | 2 | 0.75 | 1.14 | 1.52 | rejected |
| 1.24 | 0.45 | 1 | 2 | 0.75 | 1.49 | 1.99 | rejected |
| 1.25 | -0.03 | 1 | 2 | 1.00 | 1.76 | 1.76 | rejected |
| 1.32 | 0.43 | 1 | 2 | 0.75 | 1.40 | 1.88 | rejected |
| 1.33 | -0.18 | 1 | 2 | 1.00 | 1.16 | 1.16 | rejected |
| 1.35 | 0.07 | 1 | 2 | 0.75 | 1.36 | 1.82 | rejected |
| 1.41 | -0.41 | 1 | 1 | 1.67 | 0.75 | 2.24 | rejected |
| 1.41 | 0.34 | 1 | 1 | 1.46 | 1.06 | 1.38 | rejected |
| 1.41 | -0.06 | 1 | 1 | 0.75 | 0.75 | 1.00 | rejected |
| 1.41 | -0.11 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.41 | 0.20 | 1 | 1 | 1.80 | 0.75 | 2.41 | rejected |
| 1.41 | -0.23 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.41 | -0.21 | 1 | 1 | 1.00 | 1.25 | 1.25 | rejected |
| 1.41 | -0.73 | 1 | 1 | 1.00 | 0.75 | 1.34 | rejected |
| 1.41 | 0.01 | 1 | 1 | 1.50 | 0.75 | 2.00 | rejected |
| 1.41 | -0.68 | 1 | 1 | 0.75 | 1.50 | 2.00 | rejected |
| 1.41 | -0.25 | 1 | 1 | 0.75 | 1.00 | 1.34 | rejected |
| 1.41 | -0.10 | 1 | 1 | 1.67 | 0.75 | 2.24 | rejected |
| 1.41 | -0.05 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.44 | 0.12 | 1 | 2 | 0.75 | 1.86 | 2.49 | rejected |
| 1.45 | 0.60 | 1 | 2 | 2.00 | 1.51 | 1.32 | rejected |
| 1.46 | 0.78 | 1 | 2 | 1.25 | 1.68 | 1.35 | rejected |
| 1.49 | 0.28 | 1 | 2 | 0.75 | 1.57 | 2.10 | rejected |
| 1.51 | -0.28 | 1 | 2 | 0.75 | 1.25 | 1.68 | rejected |
| 1.52 | 0.06 | 1 | 2 | 1.25 | 1.24 | 1.01 | rejected |
| 1.56 | 0.32 | 1 | 2 | 0.75 | 1.88 | 2.51 | rejected |
| 1.58 | -0.01 | 1 | 2 | 1.00 | 1.70 | 1.70 | rejected |
| 1.60 | 0.24 | 1 | 2 | 0.75 | 1.97 | 2.64 | rejected |
| 1.62 | 0.23 | 1 | 2 | 0.75 | 1.93 | 2.58 | rejected |
| 1.62 | -0.17 | 1 | 2 | 0.75 | 1.91 | 2.56 | rejected |
| 1.64 | 0.16 | 1 | 2 | 0.75 | 1.85 | 2.48 | rejected |
| 1.65 | 0.87 | 1 | 2 | 0.75 | 1.97 | 2.63 | rejected |
| 1.65 | 0.78 | 1 | 2 | 0.75 | 2.01 | 2.69 | rejected |
| 1.69 | 0.26 | 1 | 2 | 1.00 | 1.78 | 1.78 | rejected |
| 1.69 | 0.57 | 1 | 2 | 0.75 | 1.42 | 1.90 | rejected |
| 1.71 | 0.40 | 1 | 2 | 1.00 | 1.72 | 1.72 | rejected |
| 1.73 | -0.67 | 1 | 1 | 0.75 | 1.25 | 1.67 | rejected |
| 1.73 | 0.62 | 1 | 1 | 1.46 | 0.75 | 1.95 | rejected |
| 1.73 | -0.11 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.73 | -0.05 | 1 | 1 | 1.67 | 0.75 | 2.24 | rejected |
| 1.73 | 0.56 | 1 | 1 | 0.75 | 1.46 | 1.95 | rejected |
| 1.73 | 0.90 | 1 | 1 | 0.75 | 1.95 | 2.60 | rejected |
| 1.73 | 0.71 | 1 | 1 | 1.06 | 1.67 | 1.58 | rejected |
| 1.73 | -0.12 | 1 | 1 | 1.06 | 1.06 | 1.00 | rejected |

…and 427 more.

**WRONGLY accepted false joins (accepted FALSE — over-merges to kill)** (0 sites):
_none._

**Correctly repaired (accepted REAL)** (3 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 1.17 | 0.94 | 1 | 2 | 0.75 | 1.79 | 2.40 | accepted |
| 2.24 | 0.98 | 1 | 1 | 1.50 | 0.75 | 2.00 | accepted |
| 3.32 | 0.98 | 1 | 1 | 2.24 | 1.80 | 1.25 | accepted |

**Correctly refused (rejected FALSE)** (1 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 3.12 | 0.55 | 1 | 2 | 1.25 | 1.99 | 1.59 | rejected |

_(4509 SplitSite(s) omitted: a label's dominant neuron is held-out, so no train-derivable verdict — excluded to keep the signal leak-free.)_


## Edits proposed

51 edits: 51× merge_labels

merge(684932360, 685004630), merge(699466729, 699394241), merge(731361018, 731361019), merge(481611665, 466896182), merge(98108279, 98180767), merge(540439897, 555154936), merge(756768080, 771628328), merge(22020889, 21948403), merge(226627164, 226626965), merge(244671404, 244671403), merge(464488053, 553504107), merge(567356524, 596859319), merge(255606972, 255606989), merge(168765382, 168765583), merge(395607515, 410033486), merge(429604198, 414961425), merge(430185205, 430112718), merge(370450109, 341237041), merge(858671195, 873458947), merge(375149638, 643212421), merge(815546148, 815619311), merge(522391432, 522391653), merge(494358345, 494575806), merge(656945254, 657234994), merge(639550965, 669342851), merge(272729687, 273020514), merge(905858684, 891143208), merge(672570693, 643212857), merge(445697853, 430982365), merge(375755183, 375610633), merge(760573865, 745931304), merge(170793545, 170794186), merge(346234843, 360877618), merge(629152404, 629806994), merge(375815390, 510142981), merge(536969830, 537259345), merge(920211284, 934927208), merge(454766649, 454694608), merge(634605281, 649320993), merge(226631933, 226559446)

…and 11 more.


## Merges attributed to edits (the ones to fix)

Each row is a merge that an edit *created* (its class fuses >=2 raw labels). Guard these label pairs in the policy.

| GT skeleton | fused labels | merge location (world) |
|---|---|---|
| N005-789202-SP | 464488053+553504107 | (6787.35, 19507.84, 46619.0) |




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