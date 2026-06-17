# Candidate failure report (train split)

- Proposed **51 edits** from 10000 candidate sites.
- Train Edge Accuracy (the fitness, = 100 - %Split - %Omit - %Merged; higher is better): baseline 79.9772 -> candidate 80.6817.
- Merge-error component (the repair target): %Merged Edges baseline 17.2204 -> candidate 17.1115; # Merges baseline 7.39 -> candidate 7.36.
- Over-split watchdog: %Split Edges baseline 0.1168 -> candidate 0.3039 (a merge repair that drives this UP is over-splitting a real neuron).


## Per-skeleton delta (candidate - baseline)

| GT skeleton | dEdgeAcc | d%MergedEdges | d#Merges | d%SplitEdges | d#Splits | d%OmitEdges |
|---|---|---|---|---|---|---|
| N005-789202-SP | +1.240 | +0.006 | +0 | +0.219 | +522 | -1.471 |
| N006-789202-JG | +0.950 | -0.074 | +0 | +0.176 | +727 | -1.047 |
| N010-789202-JT | +1.800 | -0.222 | +0 | +0.698 | +1065 | -2.275 |
| N016-789202-HP | +0.660 | -0.440 | +1 | +0.125 | +411 | -0.344 |
| N018-789202-JM | +0.360 | -0.212 | -1 | +0.056 | +260 | -0.204 |
| N020-789202-PP | +0.500 | -0.014 | +0 | +0.190 | +229 | -0.671 |
| N021-789202-HP | +0.380 | -0.042 | +0 | +0.123 | +300 | -0.466 |
| N022-789202-JG | +0.320 | +0.253 | +0 | +0.220 | +759 | -0.785 |

## Skeletons the candidate made WORSE

_none — every skeleton improved or held._


## Baseline merge errors (split_label repair targets)

10 raw label(s) span >=2 GT neurons in this split — each is a merge a `split_label` edit should break apart (reduces %Merged Edges / #Merges). Locations are GT-frame centroids (µm).

| raw label | # GT neurons | nodes per GT | per-GT centroid (x,y,z) |
|---|---|---|---|
| 553856478 | 2 | N006-789202-JG:34141; N016-789202-HP:3035 | (9134.7,17309.9,38792.2) | (9202.1,16124.0,38987.8) |
| 272636191 | 2 | N016-789202-HP:21605; N020-789202-PP:3492 | (4889.2,17128.2,9881.6) | (4907.6,17301.2,11881.3) |
| 729711944 | 2 | N006-789202-JG:18525; N021-789202-HP:766 | (10048.9,18299.6,37460.8) | (9675.8,18324.9,35950.8) |
| 670859198 | 2 | N018-789202-JM:6201; N005-789202-SP:985 | (9223.0,18060.4,41506.2) | (8888.6,18211.9,41607.6) |
| 802201821 | 2 | N022-789202-JG:4646; N021-789202-HP:1478 | (10497.8,16286.3,34815.1) | (10587.5,15894.9,36288.6) |
| 625476285 | 2 | N018-789202-JM:2887; N016-789202-HP:1798 | (8546.5,15772.8,38782.0) | (8543.1,15718.9,38406.4) |
| 272226639 | 2 | N021-789202-HP:2139; N022-789202-JG:554 | (3904.4,15349.6,28492.1) | (4078.9,15261.7,28444.4) |
| 330942292 | 2 | N022-789202-JG:1192; N021-789202-HP:998 | (4343.7,15224.0,28127.8) | (4359.6,15222.5,28147.3) |
| 758491181 | 2 | N022-789202-JG:1313; N021-789202-HP:115 | (9954.8,17135.5,36241.4) | (9901.3,17456.5,36152.4) |
| 714349752 | 2 | N006-789202-JG:899; N016-789202-HP:70 | (9454.0,16641.3,39867.4) | (9398.6,16645.7,39814.9) |


## Edits proposed

51 edits: 51× merge_labels

merge(684932360, 685004630), merge(699466729, 699394241), merge(731361018, 731361019), merge(481611665, 466896182), merge(98108279, 98180767), merge(540439897, 555154936), merge(756768080, 771628328), merge(22020889, 21948403), merge(226627164, 226626965), merge(244671404, 244671403), merge(464488053, 553504107), merge(567356524, 596859319), merge(255606972, 255606989), merge(168765382, 168765583), merge(395607515, 410033486), merge(429604198, 414961425), merge(430185205, 430112718), merge(370450109, 341237041), merge(858671195, 873458947), merge(375149638, 643212421), merge(815546148, 815619311), merge(522391432, 522391653), merge(494358345, 494575806), merge(656945254, 657234994), merge(639550965, 669342851), merge(272729687, 273020514), merge(905858684, 891143208), merge(672570693, 643212857), merge(445697853, 430982365), merge(375755183, 375610633), merge(760573865, 745931304), merge(170793545, 170794186), merge(346234843, 360877618), merge(629152404, 629806994), merge(375815390, 510142981), merge(536969830, 537259345), merge(920211284, 934927208), merge(454766649, 454694608), merge(634605281, 649320993), merge(226631933, 226559446)

…and 11 more.


## Merges attributed to edits (the ones to fix)

Each row is a merge that an edit *created* (its class fuses >=2 raw labels). Guard these label pairs in the policy.

| GT skeleton | fused labels | merge location (world) |
|---|---|---|
| N022-789202-JG | 375815390+510142981 | (9686.6, 18381.35, 34773.0) |
| N005-789202-SP | 464488053+553504107 | (6787.35, 19507.84, 46619.0) |

