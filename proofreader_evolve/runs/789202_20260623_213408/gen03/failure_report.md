# Candidate failure report (train split)

- Proposed **294 edits** from 5000 candidate sites.
- **Split-repair score (THE FITNESS the gate keeps on) = correct - false = 28 - 0 = 28** (train-classified merges; unscored=266). The gate accepts only if this BEATS the parent AND false == 0 — a single false merge (fusing two different neurons) rejects the candidate outright.
- Edge Accuracy (secondary diagnostic, NOT the bar; = 100 - %Split - %Omit - %Merged): baseline 90.0449 -> candidate 89.7471.
- Merge-error component: %Merged Edges baseline 5.0241 -> candidate 5.3284; # Merges baseline 4.97 -> candidate 4.97.
- Over-split watchdog: %Split Edges baseline 0.4476 -> candidate 0.4458 (a merge repair that drives this UP is over-splitting a real neuron).

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
| N005-789202-SP | -1.130 | +1.147 | +0 | -0.002 | -9 | -0.016 |
| N010-789202-JT | -0.310 | +0.328 | +0 | -0.002 | -4 | -0.010 |
| N011-789202-IG | +0.000 | +0.000 | +0 | -0.005 | -8 | -0.001 |
| N020-789202-PP | +0.000 | +0.000 | +0 | -0.001 | -3 | +0.000 |
| N021-789202-HP | +0.000 | +0.000 | +0 | -0.000 | -1 | +0.000 |
| N023-789202-SP | +0.000 | +0.000 | +0 | -0.001 | -3 | -0.001 |

_Note: a positive `d#Splits` paired with a negative `d%OmitEdges` is the footprint of your MERGES healing background (omit) nodes into labels (the skeleton then crosses more distinct labels) — not over-splitting by you. `# Splits` / Edge Accuracy are GT-side diagnostics, NOT the gate._


## Train-side FALSE merges (precision alert — NOT a gate reject)

_no train-side false merges — your merges did not fuse two different neurons on any train skeleton._

## Skeletons with an Edge-Accuracy dip (secondary diagnostic)

N005-789202-SP, N010-789202-JT


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

Confusion summary: REAL 490 (accepted 28 / MISSED 462); FALSE 1 (WRONGLY accepted 0 / correctly rejected 1).


**MISSED real splits (rejected REAL — raise recall)** (462 sites):
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
| 1.65 | 0.78 | 1 | 2 | 0.75 | 2.01 | 2.69 | rejected |
| 1.69 | 0.26 | 1 | 2 | 1.00 | 1.78 | 1.78 | rejected |
| 1.69 | 0.57 | 1 | 2 | 0.75 | 1.42 | 1.90 | rejected |
| 1.71 | 0.40 | 1 | 2 | 1.00 | 1.72 | 1.72 | rejected |
| 1.73 | -0.67 | 1 | 1 | 0.75 | 1.25 | 1.67 | rejected |
| 1.73 | 0.62 | 1 | 1 | 1.46 | 0.75 | 1.95 | rejected |
| 1.73 | -0.11 | 1 | 1 | 1.06 | 0.75 | 1.41 | rejected |
| 1.73 | -0.05 | 1 | 1 | 1.67 | 0.75 | 2.24 | rejected |
| 1.73 | 0.56 | 1 | 1 | 0.75 | 1.46 | 1.95 | rejected |
| 1.73 | 0.71 | 1 | 1 | 1.06 | 1.67 | 1.58 | rejected |
| 1.73 | -0.12 | 1 | 1 | 1.06 | 1.06 | 1.00 | rejected |
| 1.77 | 0.19 | 1 | 2 | 1.00 | 1.79 | 1.79 | rejected |
| 1.77 | 0.10 | 1 | 2 | 1.00 | 1.97 | 1.97 | rejected |
| 1.87 | 0.67 | 1 | 2 | 0.75 | 1.69 | 2.26 | rejected |

…and 402 more.

**WRONGLY accepted false joins (accepted FALSE — over-merges to kill)** (0 sites):
_none._

**Correctly repaired (accepted REAL)** (28 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 1.17 | 0.94 | 1 | 2 | 0.75 | 1.79 | 2.40 | accepted |
| 1.19 | 0.84 | 1 | 2 | 1.25 | 1.68 | 1.35 | accepted |
| 1.65 | 0.87 | 1 | 2 | 0.75 | 1.97 | 2.63 | accepted |
| 1.73 | 0.90 | 1 | 1 | 0.75 | 1.95 | 2.60 | accepted |
| 1.75 | 0.87 | 1 | 2 | 1.00 | 1.79 | 1.79 | accepted |
| 2.00 | 0.91 | 1 | 1 | 1.06 | 1.25 | 1.18 | accepted |
| 2.24 | 0.98 | 1 | 1 | 1.50 | 0.75 | 2.00 | accepted |
| 2.24 | 0.84 | 1 | 1 | 2.12 | 1.50 | 1.41 | accepted |
| 2.24 | 0.85 | 1 | 1 | 1.50 | 1.46 | 1.03 | accepted |
| 2.45 | 0.87 | 1 | 1 | 0.75 | 1.95 | 2.60 | accepted |
| 2.45 | 0.94 | 1 | 1 | 1.06 | 1.95 | 1.84 | accepted |
| 2.48 | 0.87 | 1 | 2 | 1.80 | 1.84 | 1.02 | accepted |
| 2.56 | 0.94 | 1 | 2 | 0.75 | 1.61 | 2.15 | accepted |
| 2.95 | 0.87 | 1 | 2 | 1.00 | 1.86 | 1.86 | accepted |
| 3.00 | 0.96 | 1 | 1 | 1.50 | 1.46 | 1.03 | accepted |
| 3.00 | 0.77 | 1 | 1 | 1.95 | 1.67 | 1.16 | accepted |
| 3.00 | 0.89 | 1 | 1 | 1.00 | 1.67 | 1.67 | accepted |
| 3.00 | 0.93 | 1 | 1 | 1.50 | 1.67 | 1.12 | accepted |
| 3.07 | 0.92 | 1 | 2 | 1.95 | 1.62 | 1.20 | accepted |
| 3.11 | 0.87 | 1 | 2 | 0.75 | 1.90 | 2.54 | accepted |
| 3.13 | 0.94 | 1 | 2 | 1.50 | 1.59 | 1.06 | accepted |
| 3.16 | 0.71 | 1 | 1 | 1.50 | 2.13 | 1.43 | accepted |
| 3.16 | 0.80 | 1 | 1 | 1.67 | 1.00 | 1.67 | accepted |
| 3.19 | 0.95 | 1 | 2 | 1.46 | 1.73 | 1.19 | accepted |
| 3.32 | 0.98 | 1 | 1 | 2.24 | 1.80 | 1.25 | accepted |
| 3.32 | 0.87 | 1 | 1 | 1.95 | 1.46 | 1.34 | accepted |
| 3.46 | 0.81 | 1 | 1 | 0.75 | 1.67 | 2.24 | accepted |
| 3.54 | 0.92 | 1 | 2 | 1.00 | 1.89 | 1.89 | accepted |

**Correctly refused (rejected FALSE)** (1 sites):
| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | rad_ratio | your decision |
|---|---|---|---|---|---|---|---|
| 3.12 | 0.55 | 1 | 2 | 1.25 | 1.99 | 1.59 | rejected |

_(4509 SplitSite(s) omitted: a label's dominant neuron is held-out, so no train-derivable verdict — excluded to keep the signal leak-free.)_


## Image evidence the policy read (gap_bridge_evidence), by GT class

Bridge statistics from `gap_bridge_evidence` calls the policy ALREADY made, split by whether the SplitSite's two labels are the SAME train neuron. A HIGH `bridge_ratio` (signal stays bright across the gap) is the merge tell; pick a `merge_labels` image threshold that ACCEPTS the REAL splits below and REJECTS the false joins.

**REAL splits — SHOULD merge** (25 read):
| bridge_ratio | bridge_min | endpoint_mean | bridge_pos |
|---|---|---|---|
| 0.98 | 44.50 | 45.2 | 0.00 |
| 1.00 | 29.00 | 29.0 | 0.00 |
| 0.97 | 18.00 | 18.5 | 0.00 |
| 0.97 | 151.50 | 156.2 | 0.00 |
| 0.98 | 21.00 | 21.5 | 0.00 |
| 1.00 | 21.00 | 21.0 | 0.00 |
| 1.02 | 165.00 | 161.8 | 1.00 |
| 1.00 | 43.00 | 43.0 | 0.00 |
| 0.98 | 30.00 | 30.5 | 0.00 |
| 1.00 | 24.00 | 24.0 | 1.00 |
| 1.01 | 53.00 | 52.5 | 1.00 |
| 1.00 | 46.00 | 46.0 | 0.00 |
| 0.92 | 49.00 | 53.5 | 1.00 |
| 0.95 | 67.00 | 70.8 | 0.00 |
| 0.96 | 27.00 | 28.0 | 0.00 |
| 0.96 | 27.00 | 28.0 | 0.25 |
| 1.00 | 36.00 | 36.0 | 0.00 |
| 0.97 | 17.00 | 17.5 | 0.00 |
| 1.03 | 19.00 | 18.5 | 0.00 |
| 1.03 | 20.00 | 19.5 | 1.00 |
| 1.02 | 42.00 | 41.0 | 0.00 |
| 0.98 | 24.00 | 24.5 | 1.00 |
| 0.87 | 104.00 | 119.8 | 1.00 |
| 1.00 | 49.00 | 49.0 | 0.00 |
| 0.98 | 23.00 | 23.5 | 0.75 |

**FALSE joins — must NOT merge** (0 read):
_none read on a cross-neuron SplitSite this generation._


## Edits proposed

294 edits: 294× merge_labels

merge(801699872, 787129586), merge(136997205, 137069487), merge(684932360, 685004630), merge(524939720, 524867231), merge(334276135, 334276133), merge(80447246, 80374753), merge(497535396, 497535183), merge(699466729, 699394241), merge(331437670, 331365180), merge(757203452, 713130579), merge(477541784, 477541786), merge(741107830, 741107611), merge(551250377, 536607600), merge(704404743, 675046259), merge(555659522, 540944475), merge(731361018, 731361019), merge(612350501, 656640845), merge(481611665, 466896182), merge(738051252, 723263488), merge(541036216, 540963738), merge(467724214, 467651725), merge(478123885, 478123661), merge(727768758, 683619676), merge(375020421, 389735470), merge(494576247, 494576246), merge(346385732, 346313463), merge(408366016, 408583287), merge(700724889, 700652183), merge(98108279, 98180767), merge(993281509, 978711003), merge(375149638, 716208403), merge(98544748, 113260018), merge(540439897, 555154936), merge(576455743, 547024771), merge(349435747, 320004994), merge(756768080, 771628328), merge(555304067, 555304288), merge(391330663, 405684152), merge(599939559, 643432514), merge(789892082, 789892079)

…and 254 more.


## Merges attributed to edits (the ones to fix)

Each row is a merge that an edit *created* (its class fuses >=2 raw labels). Guard these label pairs in the policy.

| GT skeleton | fused labels | merge location (world) |
|---|---|---|
| N005-789202-SP | 582071132+612006453 | (8356.66, 16226.36, 52629.0) |
| N005-789202-SP | 464488053+553504107 | (6787.35, 19507.84, 46619.0) |
| N010-789202-JT | 408366016+408583287 | (5298.08, 23203.71, 32973.0) |




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