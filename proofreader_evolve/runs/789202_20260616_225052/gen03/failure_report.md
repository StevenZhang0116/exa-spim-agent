# Candidate failure report (train split)

- Proposed **51 edits** from 10000 candidate sites.
- Train Edge Accuracy (the fitness, = 100 - %Split - %Omit - %Merged; higher is better): baseline 86.7532 -> candidate 87.5273.
- Merge-error component (the repair target): %Merged Edges baseline 10.3709 -> candidate 10.2289; # Merges baseline 9.05 -> candidate 9.05.
- Over-split watchdog: %Split Edges baseline 0.1174 -> candidate 0.2794 (a merge repair that drives this UP is over-splitting a real neuron).


## Per-skeleton delta (candidate - baseline)

| GT skeleton | dEdgeAcc | d%MergedEdges | d#Merges | d%SplitEdges | d#Splits | d%OmitEdges |
|---|---|---|---|---|---|---|
| N005-789202-SP | +1.240 | +0.006 | +0 | +0.219 | +522 | -1.471 |
| N006-789202-JG | +0.950 | -0.074 | +0 | +0.176 | +727 | -1.047 |
| N007-789202-PP | +0.740 | -0.072 | +0 | +0.173 | +264 | -0.838 |
| N016-789202-HP | +0.660 | -0.440 | +0 | +0.125 | +411 | -0.344 |
| N020-789202-PP | +0.500 | -0.014 | +0 | +0.190 | +229 | -0.671 |
| N021-789202-HP | +0.380 | -0.042 | +0 | +0.123 | +300 | -0.466 |

## Skeletons the candidate made WORSE

_none — every skeleton improved or held._


## Baseline merge errors (split_label repair targets)

4 raw label(s) span >=2 GT neurons in this split — each is a merge a `split_label` edit should break apart (reduces %Merged Edges / #Merges). Locations are GT-frame centroids (µm).

| raw label | # GT neurons | nodes per GT | per-GT centroid (x,y,z) |
|---|---|---|---|
| 553856478 | 2 | N006-789202-JG:34141; N016-789202-HP:3035 | (9134.7,17309.9,38792.2) | (9202.1,16124.0,38987.8) |
| 272636191 | 2 | N016-789202-HP:21605; N020-789202-PP:3492 | (4889.2,17128.2,9881.6) | (4907.6,17301.2,11881.3) |
| 729711944 | 2 | N006-789202-JG:18525; N021-789202-HP:766 | (10048.9,18299.6,37460.8) | (9675.8,18324.9,35950.8) |
| 714349752 | 2 | N006-789202-JG:899; N016-789202-HP:70 | (9454.0,16641.3,39867.4) | (9398.6,16645.7,39814.9) |


## MergeSite features at TRUE merges (split_label targets)

Each row is a candidate MergeSite whose label IS a baseline merge (a real fusion of >=2 GT neurons). These are the geometry patterns a `split_label` SHOULD fire on. Key your policy on these columns, never the raw label.

| detector | angle_deg | radius_ratio | cable_a | cable_b | branch_degree | arms_reconverge |
|---|---|---|---|---|---|---|
| bridge | 107.65 | 1.00 | 29.7 | 29.6 | 2 | — |
| branch | 155.68 | 1.13 | 30.3 | 29.4 | 3 | — |
| branch | 139.17 | 1.04 | 29.0 | 28.8 | 3 | — |
| branch | 151.11 | 1.09 | 28.9 | 28.5 | 3 | — |
| bridge | 105.74 | 1.02 | 28.3 | 29.0 | 2 | — |
| bridge | 112.62 | 1.01 | 28.3 | 28.4 | 2 | — |
| branch | 99.32 | 1.05 | 28.4 | 28.2 | 3 | — |
| branch | 166.78 | 1.04 | 28.9 | 28.2 | 3 | — |
| branch | 120.29 | 1.00 | 28.1 | 28.0 | 3 | — |
| bridge | 112.51 | 1.00 | 27.8 | 30.1 | 2 | — |
| branch | 171.96 | 1.03 | 28.0 | 27.6 | 3 | — |
| bridge | 117.66 | 1.03 | 27.6 | 27.7 | 2 | — |
| bridge | 84.31 | 1.00 | 30.0 | 27.6 | 2 | — |


## MergeSite features at NON-merges (cutting here over-splits)

Same geometry for MergeSites whose label is NOT a baseline merge — a `split_label` here would cut a single real neuron (raises %Split Edges). Use these as the negative class: pick thresholds that separate the table above from this one.

| detector | angle_deg | radius_ratio | cable_a | cable_b | branch_degree | arms_reconverge |
|---|---|---|---|---|---|---|
| component | NaN | 1.02 | 10883.7 | 3783.2 | 0 | — |
| component | NaN | 1.01 | 3066.6 | 593.7 | 0 | — |
| component | NaN | 1.04 | 944.4 | 556.8 | 0 | — |
| component | NaN | 1.17 | 270.3 | 149.0 | 0 | — |
| component | NaN | 1.02 | 94.0 | 87.6 | 0 | — |
| branch | 3.89 | 1.00 | 38.6 | 34.2 | 3 | — |
| branch | 2.43 | 1.07 | 43.9 | 33.5 | 3 | — |
| branch | 89.84 | 1.09 | 35.1 | 32.2 | 3 | — |
| branch | 145.17 | 1.48 | 32.0 | 31.0 | 3 | — |
| bridge | 93.38 | 1.05 | 30.9 | 30.8 | 2 | — |
| branch | 101.64 | 1.06 | 31.4 | 30.7 | 3 | — |
| branch | 71.06 | 1.02 | 33.5 | 30.6 | 3 | — |
| branch | 119.62 | 1.07 | 31.0 | 30.3 | 3 | — |
| branch | 76.94 | 1.06 | 34.8 | 30.2 | 3 | — |
| branch | 84.75 | 1.03 | 30.2 | 30.2 | 3 | — |
| branch | 74.22 | 1.05 | 30.7 | 30.2 | 3 | — |
| branch | 91.31 | 1.05 | 30.7 | 30.1 | 3 | — |
| branch | 168.58 | 1.14 | 30.3 | 30.1 | 3 | — |
| branch | 140.48 | 1.04 | 31.7 | 30.1 | 3 | — |
| branch | 83.58 | 1.00 | 31.8 | 30.1 | 3 | — |
| branch | 145.84 | 1.01 | 30.8 | 30.0 | 3 | — |
| branch | 107.13 | 1.15 | 36.2 | 30.0 | 3 | — |
| branch | 164.56 | 1.05 | 30.0 | 29.9 | 3 | — |
| branch | 92.05 | 1.04 | 30.3 | 29.9 | 3 | — |
| branch | 157.92 | 1.00 | 52.5 | 29.9 | 3 | — |
| branch | 96.62 | 1.02 | 34.4 | 29.9 | 3 | — |
| branch | 116.01 | 1.06 | 30.4 | 29.9 | 3 | — |
| bridge | 71.19 | 1.13 | 29.8 | 30.3 | 2 | — |
| branch | 73.12 | 1.12 | 29.8 | 29.8 | 3 | — |
| branch | 33.02 | 1.05 | 30.4 | 29.8 | 3 | — |
| branch | 81.74 | 1.02 | 30.5 | 29.8 | 3 | — |
| branch | 102.22 | 1.01 | 29.9 | 29.8 | 3 | — |
| branch | 97.16 | 1.04 | 30.3 | 29.7 | 3 | — |
| branch | 59.02 | 1.02 | 30.7 | 29.7 | 3 | — |
| branch | 126.03 | 1.06 | 30.2 | 29.7 | 3 | — |
| branch | 98.25 | 1.03 | 30.4 | 29.7 | 3 | — |
| branch | 81.57 | 1.02 | 31.1 | 29.7 | 3 | — |
| branch | 59.56 | 1.04 | 30.3 | 29.7 | 3 | — |
| branch | 113.48 | 1.03 | 30.2 | 29.7 | 3 | — |
| branch | 100.49 | 1.00 | 30.7 | 29.7 | 3 | — |
| branch | 103.69 | 1.16 | 30.0 | 29.6 | 3 | — |
| branch | 122.49 | 1.12 | 29.8 | 29.6 | 3 | — |
| branch | 81.49 | 1.04 | 30.2 | 29.6 | 3 | — |
| branch | 48.61 | 1.29 | 29.7 | 29.6 | 3 | — |
| branch | 126.52 | 1.05 | 30.3 | 29.6 | 3 | — |
| branch | 158.36 | 1.22 | 30.1 | 29.6 | 3 | — |
| branch | 53.43 | 1.12 | 29.6 | 29.6 | 3 | — |
| branch | 149.69 | 1.02 | 30.9 | 29.6 | 3 | — |
| branch | 41.25 | 1.00 | 29.7 | 29.6 | 3 | — |
| branch | 105.06 | 1.02 | 29.9 | 29.6 | 3 | — |
| branch | 165.28 | 1.09 | 29.6 | 29.6 | 3 | — |
| branch | 88.72 | 1.06 | 30.9 | 29.6 | 3 | — |
| branch | 140.76 | 1.07 | 29.7 | 29.6 | 3 | — |
| branch | 169.73 | 1.01 | 29.8 | 29.5 | 3 | — |
| bridge | 109.37 | 1.00 | 29.5 | 29.8 | 2 | — |
| branch | 61.18 | 1.03 | 29.9 | 29.5 | 3 | — |
| branch | 102.43 | 1.04 | 29.6 | 29.5 | 3 | — |
| branch | 23.95 | 1.12 | 29.6 | 29.5 | 3 | — |
| branch | 127.22 | 1.01 | 30.6 | 29.5 | 3 | — |
| branch | 113.53 | 1.07 | 36.6 | 29.5 | 3 | — |

…and 4919 more non-merge sites.

_(8 MergeSite(s) omitted from BOTH tables: their label is a merge only visible via held-out GT, so it is neither a train-derivable positive nor a safe negative — excluded to keep the reviser's signal leak-free.)_


## Merge targets with NO candidate MergeSite (detector recall gap)

1 baseline merge label(s) have no enumerated MergeSite, so `propose_edits` can never reach them — a `candidate_merge_sites` detector limitation, not a policy bug: 714349752



## Edits proposed

51 edits: 51× merge_labels

merge(684932360, 685004630), merge(699466729, 699394241), merge(731361018, 731361019), merge(481611665, 466896182), merge(98108279, 98180767), merge(540439897, 555154936), merge(756768080, 771628328), merge(22020889, 21948403), merge(226627164, 226626965), merge(244671404, 244671403), merge(464488053, 553504107), merge(567356524, 596859319), merge(255606972, 255606989), merge(168765382, 168765583), merge(395607515, 410033486), merge(429604198, 414961425), merge(430185205, 430112718), merge(370450109, 341237041), merge(858671195, 873458947), merge(375149638, 643212421), merge(815546148, 815619311), merge(522391432, 522391653), merge(494358345, 494575806), merge(656945254, 657234994), merge(639550965, 669342851), merge(272729687, 273020514), merge(905858684, 891143208), merge(672570693, 643212857), merge(445697853, 430982365), merge(375755183, 375610633), merge(760573865, 745931304), merge(170793545, 170794186), merge(346234843, 360877618), merge(629152404, 629806994), merge(375815390, 510142981), merge(536969830, 537259345), merge(920211284, 934927208), merge(454766649, 454694608), merge(634605281, 649320993), merge(226631933, 226559446)

…and 11 more.


## Merges attributed to edits (the ones to fix)

Each row is a merge that an edit *created* (its class fuses >=2 raw labels). Guard these label pairs in the policy.

| GT skeleton | fused labels | merge location (world) |
|---|---|---|
| N005-789202-SP | 464488053+553504107 | (6787.35, 19507.84, 46619.0) |

