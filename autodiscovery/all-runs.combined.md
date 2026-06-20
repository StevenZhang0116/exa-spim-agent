# AutoDiscovery Cross-Report Consolidation — All Runs

## Header

**Source reports ingested** (helper: `python agentic/collect_summaries.py`, `n_files = 2`, `n_entries_total = 40`):

- `ground-truth-error-annotations-revised-version_2026-06-17.summary.md` — run `ground-truth-error-annotations-revised-version_2026-06-17` — 20 entries (older).
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20.summary.md` — run `run-4--ground-truth-error-annotations-revised-version_2026-06-20` — 20 entries (newer).

**After clustering:** 26 distinct scientific findings remain (14 corroborated across both reports + 12 unique-to-one-report). Of the 12 unique findings, 11 appear only in the newer run-4 report and 1 appears only in the older 2026-06-17 report.

**Synthesis.** Both reports independently re-discover the same dominant theme: U-Net segmentation failures are non-random topological-spatial phenomena, with (a) **merge errors concentrated in structurally crowded / dense-branching neuropil**, (b) **split errors concentrated at topological branch points and on distal/thin processes**, and (c) **omit errors concentrated on terminal/leaf edges and forming contiguous streaks**. The clearest corroborated discriminators are angular alignment across split gaps (~153° true vs ~90° false, AUC ≈ 0.93 in both runs) and tight split-gap distance distributions (95th percentile ≈ 5.8 µm, 99th ≈ 6.5 µm). The most important unique-and-new findings come from the newer run-4 report: **(1) the headline confidence flip in favor of distance-only auto-reconnection (AUC 0.9979, F1-optimal threshold 6.84 µm)** (id 30); **(2) the retirement of the Z-axis anisotropy assumption** (ids 27 and 21, both reaching CRITICAL/MAJOR after correction); **(3) the "super-merges" finding that 3+ neuron fusions cover ~5× more cable than 2-neuron merges** (id 24); and **(4) the spatial co-localization of omit and split errors with merge sites** (ids 36 and 61), both of which were OVERTURNED on a cluster-correct test. Cross-report disagreements: the older report's H39 measures **merge-cut targeted severing** (~86% merge removal, +11pt accuracy), whereas the newer report's H39 measures **A\* split-repair** (86% per-split success, +0.42pt accuracy) — same id, different intervention, different effect. The older report's H35 declares fixed-radius split bridging "unworkable" (gaps ~4× larger than internal edges) while the newer report's H30/H63 declares distance-only bridging near-perfect against inter-neuron controls — the apparent contradiction is reconciled by the choice of control (intra-fragment internal edges vs inter-neuron neighbors).

---

## Unique & New Findings

### 1. (Priority 0.507 · Surprise 0.690) Euclidean gap distance alone separates true splits from inter-neuron neighbors nearly perfectly, flipping belief in favor of distance-only auto-reconnection.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 30) — only report to find this. Newest run.
- **Conclusion:** Across 6,805 true split gaps and 4,189 inter-neuron gaps within 20 µm, true-split distances peak tightly at ~4.5 µm (mostly 2–7 µm) while inter-neuron gaps rarely fall below 7 µm. A binary-classifier ROC AUC of 0.9979 with an F1-optimal threshold of 6.84 µm (max F1 = 0.9945) confirms distance alone is a viable, safe heuristic. The prior "Leaning False" (0.2917) was overturned to "Leaning True" (0.7344). Generalizes across all three brains: AUC ≥ 0.9889, thresholds 6.48–6.84 µm, F1 ≥ 0.9711.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES; Verdict OK.
- **Why unique/new:** Distance-only auto-reconnection was framed as unworkable in the older report (H35: gaps ~4× larger than internal edges). The newer report's reframing against an inter-neuron control (not intra-fragment internal edges) inverts that conclusion.
- **Caveats:** In-sample F1-optimal threshold selection introduces mild optimism; deployed agents would need to guard against false positives in regions of dense neuropil not represented in the test set.

### 2. (Priority 0.323 · Surprise 0.795) Z-axis-aligned neurites are NOT meaningfully more error-prone than XY-aligned ones — but a corrected GEE flips this to "Z is significantly PROTECTIVE."
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 27) — only report to find this. Newest run.
- **Conclusion:** Recorded mixed-effects logistic regression on a 20,000-edge subsample (of 1,160,529) returned a standardized z-alignment coefficient of −0.0661, p = 0.0582, OR = 0.8029 — non-significant and pointed slightly the "wrong" way, motivating the belief drop from Likely True (0.9167) to Uncertain (0.4062). Under the corrected cluster-robust GEE on the full 1.16M edges, the coefficient becomes −0.1229, p ≈ 0, OR = 0.6649 [0.6180, 0.7154]: Z-alignment is significantly PROTECTIVE, the opposite of the prior. Generalization PARTIAL: origin 789202 protective (OR 0.66, p≈0); 794495 protective (OR 0.51, p=5.7e-12); 794491 no significant effect (OR 1.12, CI [0.81, 1.55]).
- **Verdict carried over:** Reproduction REPRODUCED; Generalization PARTIAL; Verdict MAJOR; Post-correction verdict OVERTURNED.
- **Why unique/new:** Z-axis anisotropy was not tested in the older report.
- **Caveats:** Original 20k subsample wasted ~98% of available data; the "failed-to-reject = null is true" fallacy drove the original negative surprisal; the corrected protective effect is brain-specific (absent on 794491).

### 3. (Priority 0.266 · Surprise 0.568) Z-dominant vs XY-dominant edges show essentially identical error rates (3.70% vs 3.63%), independently confirming the no-anisotropy result — but flips sign across extra brains.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 21) — only report to find this. Newest run.
- **Conclusion:** Chi-square on 433,243 Z-dominant edges (3.70% error) vs 975,802 XY-dominant edges (3.63% error) yielded χ² = 3.5328, p = 0.0602 — failing significance. Belief dropped Likely True (0.8333) → Uncertain (0.4688). DOES-NOT-GENERALIZE: on 794491 Z-error rate is significantly HIGHER (χ²=166.12, p=5.21e-38), on 794495 it is significantly LOWER (χ²=395.76, p=4.62e-88). Corrected GEE OR(Z vs XY) = 1.0184 [0.9712, 1.0680], p = 0.45 on origin; UPHELD as "no effect" on origin only, with the two extras pointing in OPPOSITE directions.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization DOES-NOT-GENERALIZE; Verdict CRITICAL; Post-correction verdict UPHELD on origin only.
- **Why unique/new:** Companion to finding #2 (id 27); the older report did not test imaging-axis anisotropy at all.
- **Caveats:** Borderline p = 0.0602 misinterpreted as refuting H₁ ("failed-to-reject = null is true" fallacy); fails BH-FDR (cutoff 0.0444); independence-violation between edges from the same neuron inflates effective n.

### 4. (Priority 0.265 · Surprise 0.414) Centrifugal branch order predicts split errors after all, with deep-order branches spiking past 3% error rate — but sign flips on both extra brains.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 139) — only report to find this. Newest run.
- **Conclusion:** Logistic regression on 1,409,045 edges gave branch_order coef = +0.0194, z = 16.37, p < 0.001 on origin (deeper-order more split-prone). Generalization DOES-NOT-GENERALIZE: on 794491 coef = −0.0157 (p < 0.001) and on 794495 coef = −0.0063 (p < 0.001) — both significant in the OPPOSITE direction. Corrected GEE with neuron cluster gives coef = +0.01943, p = 0.038 on origin (barely significant; OR = 1.0196 [1.001, 1.039]), p = 0.034 with opposite sign on 794491, p = 0.58 (NS) on 794495.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization DOES-NOT-GENERALIZE; Verdict CRITICAL; Post-correction verdict WEAKENED on origin and OVERTURNED on extras.
- **Why unique/new:** Older report did not analyze centrifugal branch order.
- **Caveats:** `norm_thickness` had zero variance and was dropped, so the "independent of cable thickness" qualifier was never tested; deep-order spike rests on sparse high-order bins; mechanistic causal phrasing not supported by observational regression.

### 5. (Priority 0.253 · Surprise 0.284) "Super-merges" fusing 3+ GT neurons cover ~5× more GT cable than 2-neuron merges, confirming that worst merge errors are massive structures.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 24) — only report to find this. Newest run.
- **Conclusion:** 27 merging segments qualified (24 two-neuron, 3 super-merges). Median covered cable was 6.21 mm for 2-neuron vs 35.19 mm for super-merges (Mann-Whitney U = 0.0, p = 5.89e-03). Per-neuron, that is ~11.7 mm/neuron vs ~3.1 mm/neuron. Belief shifted Leaning True (0.7083) → Likely True (0.8906). Generalization PARTIAL: 794495 strongly confirms (n=24+2, p=6.15e-03, super-merge median 168.42 mm); 794491 has NO super-merges (test cannot run); origin single-brain rerun gives n=8+1 with p=0.222 (NS, U=0 is a floor effect).
- **Verdict carried over:** Reproduction DIVERGED (dataset scope, single-brain pkl has only 8+1); Generalization PARTIAL; Verdict MAJOR; Post-correction verdict WEAKENED.
- **Why unique/new:** Older report did not partition merges by number of fused neurons.
- **Caveats:** Severely underpowered (n=3 super-merges, U=0 is the floor); the test compares total cable (partly tautological — fusing more neurons mechanically spans more cable) rather than the per-neuron quantity the hypothesis literally proposed.

### 6. (Priority 0.253 · Surprise 0.284) Omission errors are ~4× more frequent at topological branch points than at linear cable nodes.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 32) — only report to find this. Newest run.
- **Conclusion:** Branch-node omit rate 11.95% (606/5,072) vs linear-node omit rate 2.76% (38,610/1,398,807), χ² = 1567.69, p ≈ 0. The model demonstrably drops fragments at complex junctions. Generalizes across all three brains: 794491 7.54% vs 3.41% (ratio 2.2×, χ²=193.63, p=5.13e-44); 794495 6.52% vs 1.72% (ratio 3.8×, χ²=974.86, p=5.24e-214). Belief shifted Leaning True (0.7083) → Likely True (0.8906).
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES; Verdict MINOR.
- **Why unique/new:** The older report tested OMIT concentration near LEAVES (H11, H64), not at branch points; SPLIT vs branch (H10) is a different error type. The newer report's id 32 is the first measurement of OMIT vs branch nodes.
- **Caveats:** Edges/nodes within the same neuron are not i.i.d. (clustering not modeled); branch nodes are far rarer (5,072) than linear (1.4M); causal phrasing "complex branching structure causes dropout" is observational only.

### 7. (Priority 0.253 · Surprise 0.284) Omitted cable is systematically closer to merge sites than correctly reconstructed cable — but OVERTURNED on cluster correction.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 36) — only report to find this. Newest run.
- **Conclusion:** Over 49,295 omit nodes vs a length-matched random sample of 49,295 correct nodes, median distance to the nearest of 67 merge sites was 1,812.73 µm (omit) vs 1,959.88 µm (correct), one-sided Mann-Whitney U = 1.138e9, p = 1.60e-66. Belief shifted Leaning True (0.7083) → Likely True (0.8906). Generalization DOES-NOT-GENERALIZE: on 794491 median omit 1103.24 µm > correct 842.11 µm (direction FLIPS, p=1.000 in hypothesized direction); 794495 confirms (1422.53 vs 1670.94 µm, p=9.78e-226). Corrected cluster-bootstrap by neuron: gap CI [−401.82, +91.98] µm crosses zero, p = 0.27 (NS).
- **Verdict carried over:** Reproduction REPRODUCED; Generalization DOES-NOT-GENERALIZE; Verdict CRITICAL; Post-correction verdict OVERTURNED.
- **Why unique/new:** Older report did not test omit-merge spatial co-location.
- **Caveats:** Tiny effect (~147 µm gap on ~1,900 µm baseline, ~7.7%); significance driven by treating 49k spatially correlated nodes as i.i.d.; mechanistic claim "model sacrifices thin adjacent processes when fusing" is overreach.

### 8. (Priority 0.253 · Surprise 0.284) A heuristic A\* on the fragments graph (angle + radius penalties) repairs 86% of split edges without inducing merges.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 39) — only report to find this. Newest run.
- **Conclusion:** On 6,805 targeted splits the agent found valid (no-merge) paths for 5,881 — an 86.42% per-split success rate, double the 40% threshold. End-to-end Edge Accuracy improved from 78.71% to 79.13% (+0.42% net gain). Belief shifted Leaning True (0.7083) → Likely True (0.8906). Generalizes across all three brains: 794491 success 84.72%, EA gain +1.18%; 794495 success 89.71%, EA gain +0.53%.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES; Verdict MINOR.
- **Why unique/new:** Companion to the older report's merge-cut H39 but addresses SPLITS via A\*, not MERGES via graph severing. Same id-number, different intervention, different error type.
- **Caveats:** No statistical test or CI on the success rate; "without inducing merges" relies on GT — at deployment without GT, no-merge cannot be guaranteed; +0.42% EA is modest; arbitrary >40% threshold easily cleared.

### 9. (Priority 0.253 · Surprise 0.284) Merging segments behave as "giant" components — mean cable length ~30× larger than non-merging segments.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 43) — only report to find this. Newest run.
- **Conclusion:** 64 merging segments had mean length ~15,449 µm (median ~3,099 µm) vs 8,273 non-merging at mean ~532 µm (median ~102 µm); Welch's t on log-cable-length = 16.54, p = 7.32e-25. Belief shifted Leaning True (0.7083) → Likely True (0.8906). Generalizes: 794491 23×, t=29.03, p≈0; 794495 34×, t=24.02, p≈0.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES; Verdict OK.
- **Why unique/new:** Older report measured merge segment branch-density (H53) but not their absolute cable length.
- **Caveats:** Partly tautological — a segment fusing multiple neurons must span them — but the ~30× magnitude is the informative claim; sample-size asymmetry 64 vs 8,273 handled by Welch's correction.

### 10. (Priority 0.253 · Surprise 0.284) Omit errors are >2× more likely at the extreme Z-depths of the imaged volume than at central depths.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 45) — only report to find this. Newest run.
- **Conclusion:** Extreme-Z omit rate 4.44% (2,283/51,369) vs central-Z 1.94% (11,342/585,909); Cochran-Mantel-Haenszel pooled OR = 2.3561, p ≈ 0. Belief shifted Leaning True (0.7083) → Likely True (0.8906). Generalization PARTIAL: 794495 OR=3.1672 (p≈0, confirms strongly); 794491 OR=1.0149 (p=0.80, NO effect). Corrected GEE with neuron cluster on origin gives OR = 2.3561 [0.4558, 12.1801], p = 0.31 (NS once clustering respected).
- **Verdict carried over:** Reproduction REPRODUCED; Generalization PARTIAL; Verdict MAJOR; Post-correction verdict WEAKENED.
- **Why unique/new:** Older report did not test depth-extreme volumetric effects.
- **Caveats:** "Stratification by brain ID" was a no-op on single-brain pkl; bin definitions (top/bottom 10% vs central 20%) are arbitrary; mechanistic "optical attenuation" claim contradicted by 794491.

### 11. (Priority 0.253 · Surprise 0.284) Short omission gaps are usually internal dropouts within a single predicted segment; long gaps are true fragment boundaries.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 58) — only report to find this. Newest run.
- **Conclusion:** 307 bridged omit paths (mean 18.71 µm, median 13.55 µm) vs 4,298 broken omit paths (mean 42.54 µm, median 20.16 µm); Mann-Whitney U = 458,554, p = 1.95e-19. Short omits are largely internal network dropouts, suggesting safe auto-filling targets. Belief shifted Leaning True (0.7083) → Likely True (0.8906). Generalizes across all three brains (medians 10.99–13.55 vs 15.23–20.16 µm; all p ≤ 2.75e-09).
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES; Verdict MINOR.
- **Why unique/new:** Older report did not partition omit gaps by bridged-vs-broken predicted-segment continuity.
- **Caveats:** Definition coupling — "bridged" requires sufficient non-omit neighbors on the same predicted segment, partially encoding path length already; median ratio is modest (~1.5×); significance partly from n=4,605 total.

### 12. (Priority 0.287 · Surprise 0.307) Neuron-level split rate and omit rate are strongly positively correlated, implying a shared upstream failure mode.
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 30) — only report to find this. Older run.
- **Conclusion:** Across 12 neurons exceeding the 50 µm length threshold, Pearson r = 0.65 (p = 0.022) and Spearman ρ = 0.881 (p = 1.53e-04), OLS R² = 0.422. Belief shifted Leaning True (0.7083) → Likely True (0.9327). Generalizes: 794491 (n=9) Spearman ρ=0.983, p=1.94e-06; 794495 (n=19) Spearman ρ=0.740, p=2.89e-04. Corrected: Spearman ρ promoted to headline (bootstrap CI [0.5596, 0.9923], permutation p=3.00e-04, Kendall τ=0.7273 p=4.99e-04). Post-correction verdict UPHELD.
- **Verdict carried over:** Reproduction REPRODUCED on stats; Generalization GENERALIZES; Verdict MINOR; Post-correction verdict UPHELD.
- **Why unique/new:** Only the older report performed a per-neuron correlation of split rate and omit rate. The newer report did not revisit this question.
- **Caveats:** n=12 neurons very small; Pearson sensitive to outliers (codeOutput flagged a high-leverage point near Y≈12.6); "implying a shared upstream failure mode" is mechanistic interpretation that the n=12 correlation cannot isolate.

---

## Corroborated Findings (consolidated)

### 13. (Priority 0.287 · Surprise 0.307) Merge errors concentrate in regions of unusually high local fragment-graph density / branchy "tangle" structure.
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (ids 3 [10 µm density], 23 [15 µm density], 49 [distance-to-nearest-fragment-branch], 53 [branch density of merge segment], 81 [volumetric density]); `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 84 [15 µm local branch density, ~28× at controls]) — 2 reports, 6 entries.
- **Conclusion:** Merge sites are surrounded by markedly denser fragment-graph node and branch structures than matched controls. Canonical numbers from the most strongly verified variant (older H3): mean local node density within 10 µm at 67 merge sites = 7.03 vs 4.39 at controls (Mann-Whitney U = 3931, p = 8.99e-15, ~60% elevation). At 15 µm (H23): 11.09 vs 6.33 (U = 4100.5, p = 8.95e-17, 75% elevation). Distance-to-nearest-branch (H49): merge sites median 4.48 µm vs random 179.76 µm (p = 2.21e-17); ~40% coincide with a branch exactly and ~86% lie within 10 µm. Merge segments themselves have ~2× higher branch density per 100 µm cable (H53: 0.108 vs 0.057, p = 2.91e-25). Volumetric formulation (H81): 0.001678 vs 0.001083 nodes/µm³ (Welch t = 8.93, p = 2.77e-14). Newer report's H84 paired-design: 1.13 vs 0.04 branches/15 µm at on-segment controls, paired t = 13.35 (p = 2.03e-20), AUC = 0.9236.
- **Agreement:** Both reports independently confirm with multiple metrics and radii (10 µm, 15 µm, volumetric, on-segment paired) on all three brains. Effect-size ratios are 41% to 84% elevation in node density (older); ~28× in branch density (newer paired). Corrected non-parametric Cliff's delta (older H81 correction) = 0.67 to 0.78 across all three brains.
- **Disagreement:** None on direction; the older report's H49 control construction (non-branch random nodes by construction) was flagged as borderline circular, but the merge-side effect is so large the conclusion survives.
- **Verdict carried over:** Reproduction REPRODUCED on all; Generalization GENERALIZES on all six entries across all three brains; Verdicts OK / MINOR; Post-correction verdict UPHELD (where corrected).
- **Caveats:** Modest n on the merge side (67 origin, 86, 105 extras); the older report's five density variants are not statistically independent (same merge sites, different radius/volume formulation) — count as ONE effective signal; newer H84's AUC is in-sample on paired controls.

### 14. (Priority 0.287 · Surprise 0.307) Split errors are concentrated near topological branch points (geodesic and Euclidean) — direction robust, magnitude attenuates on extra brains.
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (ids 13 [geodesic, deduped], 19 [Euclidean], 72 [topological]); `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 64 [geodesic, mean 516 vs 711 µm]) — 2 reports, 4 entries.
- **Conclusion:** Split edges sit significantly closer to GT branch points than correct edges. Canonical numbers (deduped origin H13): Mann-Whitney U = 2.98e9, p = 1.32e-197 on n = 1,109,034 correct vs 6,805 split; median shift = −184.19 µm. Euclidean (H19): mean 275.76 µm split vs 381.98 µm correct (p = 1.39e-229). Topological (H72): mean 518.39 vs 712.57 µm (p = 0). Newer report's H64 (geodesic on the same population): mean 516.30 vs 710.59 µm (p = 1.32e-197). After cluster-bootstrap correction (H13/H72/H64), Cliff's delta = −0.22 [−0.28, −0.13] origin, attenuating to −0.08 to −0.13 on extras; per-neuron paired Wilcoxon p ≤ 0.027 across all brains.
- **Agreement:** Direction (split closer to branch) preserved on all three brains and across geodesic, Euclidean, and topological distance formulations. Cluster-correct effect-size is "small but robust" (Cliff's delta 0.08–0.22).
- **Disagreement:** Magnitude. Older H13 verdict marked MAJOR for double-count bug (recorded n was 2× true n via duplicated glob); older H72 marked MAJOR/WEAKENED because the ~184 µm gap on origin collapses to ~39 µm on 794491 and ~30 µm on 794495 while floor-p stays at 0. Newer H64 marked MAJOR for the same n-driven inflation but corrected to UPHELD at "small effect" magnitude. Older H19 marked MINOR.
- **Verdict carried over:** Reproduction REPRODUCED (with the H13 double-count bug exposed); Generalization GENERALIZES directionally on all three brains across all entries; Verdicts MINOR / MAJOR; Post-correction verdict UPHELD with effect-size honesty (small Cliff's delta).
- **Caveats:** Edges within skeletons not independent — pooled p-values overstate evidence; "tightly constrained to branch-point regions" framing in newer H64 overstates the highly attenuated effect on 794491.

### 15. (Priority 0.287 · Surprise 0.307) Split errors are spatially clustered / cascade — observed inter-split distances are much shorter than null and split edges have ~10× more split neighbors than correct edges.
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 55 [KS vs uniform null, 9× shorter inter-split]); `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 37 [10× more split neighbors within 30 µm]) — 2 reports.
- **Conclusion:** Splits are not independent events but cluster locally. Older H55: median observed inter-split distance 25.58 µm vs null 236.98 µm (KS D = 0.454, p ≈ 0, ~9.3× shorter than uniform). Newer H37: split edges have mean 0.97 other-split neighbors within 30 µm vs 0.10 for matched correct edges (Mann-Whitney U = 3.47e7, p ≈ 0; ~9.7× ratio).
- **Agreement:** Both reports independently confirm ~10× local clustering of splits on origin; both generalize across all three brains (ratios 6.0×–14.3× for H55; 5.0×–11.5× for H37). Belief shift in both: Leaning True → Likely True. Cluster-correct (older H55): per-neuron median ratio obs/null = 0.07–0.13 on three brains; matched-pairs r_rb = −1.0 (every neuron exhibits obs<null); cluster-permutation p ≤ 0.031 on each brain.
- **Disagreement:** None on direction or order-of-magnitude effect.
- **Verdict carried over:** Reproduction REPRODUCED on both; Generalization GENERALIZES on both; Verdicts MINOR; Post-correction verdict (older H55) UPHELD.
- **Caveats:** Older H55's uniform null is a coarse choice (cable-density gradients absent); the causal "cascade" wording overreaches what spatial clustering establishes. Newer H37 is partly autocorrelative by design ("splits cluster with splits").

### 16. (Priority 0.287 · Surprise 0.307) Endpoint directional alignment across split gaps is a strong, exploitable discriminator (anti-parallel cosine / ~153° true continuation vs ~90° false).
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 67 [cosine similarity, AUC implicit]); `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 33 [angle in degrees, AUC = 0.9322]) — 2 reports.
- **Conclusion:** Split-gap endpoints exhibit strong directional alignment that distinguishes true continuations from false candidates. Older H67: 525 split pairs averaged cosine similarity −0.685 vs −0.511 for 2,689 spatially adjacent topologically unconnected controls (Mann-Whitney U = 594,814, p = 1.13e-08). Newer H33 on 13,582 split-node configurations: mean true-continuation angle = 152.96° vs 90.17° false (KS = 0.7460, p ≈ 0; ROC AUC = 0.9322).
- **Agreement:** Both reports confirm a sharp directional signature usable as a bridging heuristic; both generalize on all three brains. Newer H33's AUC ≥ 0.9281 across brains; older H67's effect strengthens on extras (split −0.593 to −0.791 vs control −0.333 to −0.527).
- **Disagreement:** Older H67 hedges as "discriminator, not crisp classifier" (overlap with controls at ~ −0.51), while newer H33 frames as "strong and reliable" (AUC 0.93 is unambiguous). The newer H33's tighter operational definition (true continuation vs false candidate from a different neuron) explains the cleaner AUC.
- **Verdict carried over:** Reproduction REPRODUCED on both; Generalization GENERALIZES on both; Verdicts OK on both.
- **Caveats:** Newer H33's "false candidate" is operationally defined as a nearby edge from a different neuron — the actionable claim depends on that operational definition being available at inference time without GT.

### 17. (Priority 0.287 · Surprise 0.307) Split errors are biased toward thinner / distal processes (closer to leaves than correct edges).
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 60 [distance-to-leaf, ~344 µm shorter on origin], id 65 [topological distance to leaf, 38% shorter on origin]) — only the older report explicitly tested split-vs-leaf; the newer report does not measure this directly. However, the older report's H17 (distance-to-leaf) and H65 (38% shorter median) describe the same property from two angles and corroborate each other.
- **Conclusion:** Split edges sit closer to GT leaf nodes than correct edges. Older H60: split mean 938.60 µm vs correct mean 1282.69 µm (Mann-Whitney p = 2.16e-134). Older H65: split median 417.22 µm vs correct median 671.61 µm (38% shorter, p = 0). Generalizes on all three brains directionally (gap 28–344 µm) but magnitude collapses to 13–15% on extras.
- **Agreement:** The two older entries (H60 and H65) are essentially the same measurement (distance-to-leaf) at different downsamplings; both confirm direction.
- **Disagreement:** None across reports because only one report tested it. Internally within the older report H60 and H65 are flagged as not statistically independent.
- **Verdict carried over:** Reproduction REPRODUCED (H60 exact, H65 with seedless downsample drift); Generalization GENERALIZES; Verdicts MINOR on both.
- **Caveats:** Distance-to-leaf is a proxy for radius/thickness, not a direct measurement; H65 downsampling without a fixed RNG seed makes the U statistic vary between runs; magnitude is brain-specific.

### 18. (Priority 0.287 · Surprise 0.307) Omit errors are concentrated on terminal / leaf-ending edges versus internal edges (~1.3×–1.8× ratio).
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 64 [terminal vs internal, 1.82× ratio]); `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 85 [terminal vs internal, 1.82× ratio]) — 2 reports.
- **Conclusion:** Terminal-edge omit rate ≈ 4.38% vs internal-edge omit rate ≈ 2.41% (χ² = 4189.94, p < 1e-300; ratio 1.82×). Both reports report numerically identical statistics on origin (same edge population and partition). Generalizes on all three brains: 794491 ratio 1.29× (p = 1.63e-94); 794495 ratio 1.36× (p = 1.66e-152). Belief shifted Leaning True → Likely True in both reports.
- **Agreement:** Byte-identical numbers across the two reports. Both characterize direction as robust. Older H64 corrected with cluster-bootstrap: rate ratio 1.81 [1.35, 2.44] origin, 1.29 [0.96, 1.89] 794491 (CI crosses 1!), 1.36 [1.14, 1.59] 794495.
- **Disagreement:** None on direction; the "nearly twice as likely" headline is brain-specific to origin and 794495 (drops to 1.3× on 794491); older H64's cluster-bootstrap CI on 794491 crosses 1.0.
- **Verdict carried over:** Reproduction REPRODUCED on both; Generalization GENERALIZES on both; Verdicts MINOR on both; Post-correction verdict (older) WEAKENED — direction holds but 794491 magnitude bound crosses 1.
- **Caveats:** Adjacent terminal edges share leaves and are not independent; mechanistic "distal thin neurites" framing not directly tested.

### 19. (Priority 0.287 · Surprise 0.307) Omit errors are topologically concentrated near terminal leaves of GT neurons (multi-source BFS distance).
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 11) — primary; corroborated by the newer report's id 85 at a different granularity (terminal vs internal edges). The older H11 measures continuous distance-to-leaf for individual omit edges; the newer H85 partitions into terminal/internal. Same underlying property.
- **Conclusion:** Omit edges have mean topological distance-to-leaf of 247.18 (median 130.5) vs 325.08 (median 168.5) for correct edges; one-sided Mann-Whitney U = 2.22e10, p ≈ 7.75e-311 (origin). Generalization PARTIAL: 794491 direction FLIPS in pooled MWU (omit 159.76 vs correct 138.76, p=1.000); 794495 confirms (omit 124.85 vs correct 172.81, p=6.10e-149). Corrected per-skeleton paired Wilcoxon on H11: 12/12 origin skeletons show omits closer (p=2.44e-04, CI [−90.51, −29.83] µm); 8/9 794491 skeletons (p=1.37e-02, CI [−21.45, −4.89] µm); 13/19 794495 skeletons (p=8.77e-03, CI [−28.92, −6.16] µm). Post-correction verdict reverses to UPHELD/GENERALIZES.
- **Agreement:** Older H11 and newer H85 (terminal-vs-internal partition) both implicate terminal/distal cable. Belief shift Leaning True → Likely True in both reports.
- **Disagreement:** Older H11's pooled MWU direction-flip on 794491 was originally marked MAJOR (DOES-NOT-GENERALIZE) but the test correction reverses it to GENERALIZES. The "distal-thin bias" framing is supported once skeleton-level dependence is respected.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization originally MAJOR / "DOES-NOT-GENERALIZE" but Post-correction verdict UPHELD and "GENERALIZES" under cluster-aware paired test.
- **Caveats:** Older H11's recorded p ≈ 0 floor is sample-size-driven; the corrected per-skeleton paired effect-size on extras is small (~12–17 µm shift).

### 20. (Priority 0.287 · Surprise 0.307) Split errors are ~3.4× more frequent on edges adjacent to branch points than on linear edges.
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 10) — only report to find this for SPLIT errors specifically. The newer report's id 32 measures the analogous OMIT effect (finding #6 above). Both implicate branch-adjacent edges as failure loci but for different error types — kept as separate findings (split vs omit). Older H10 is grouped here as corroborated because it is the direct categorical counterpart to all the older report's other split-near-branch findings (H13, H19, H72) and the newer H64 — i.e. it is the categorical / branching-edge formulation of finding #14.
- **Conclusion:** Branching edges have a 1.62% split rate (248/15,298) vs 0.47% for linear edges (6,557/1,393,747); χ² = 414.5, p = 3.9e-92. Generalizes on all three brains: 794491 2.07× ratio; 794495 2.00× ratio (direction never flips). Belief Leaning True → Likely True. Corrected cluster-aware test: rate ratio = 3.4458 [2.7788, 4.2931] origin (cluster-permutation p=1.996e-03); 2.0556 [1.7943, 2.4236] 794491; 2.0087 [1.6411, 2.4111] 794495.
- **Agreement:** Direction and magnitude robust on all three brains under both pooled and cluster-aware tests; corroborates finding #14 at the categorical (edge-class) level.
- **Disagreement:** None.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES; Verdict MINOR; Post-correction verdict UPHELD.
- **Caveats:** Edges sharing branch nodes are not strictly independent; the huge n makes p uninformative — cite the rate ratio (2.0×–3.4×) as the effect size.

### 21. (Priority 0.287 · Surprise 0.307) Inter-segment split gaps have a tight, characteristic length scale (~5.8 µm 95th-percentile, ~6.5 µm 99th-percentile, 100% under 15 µm).
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 35 [median 19.67 µm, gaps 3–4× larger than internal edges]); `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 63 [100% under 15 µm; 95% by ~5.85 µm; 99% by ~6.50 µm]) — 2 reports.
- **Conclusion:** Split-bridging gap lengths are short and tightly characterized. Older H35: 4,078 origin gaps, median 19.67 µm; 95th-percentile internal edge length only 5.77 µm (Mann-Whitney p ≈ 0, 3.4× ratio of gap-median to internal-95th-pct). Newer H63: 100.00% of 6,805 split gaps are under 15 µm; 95% covered by 5.85 µm, 99% by 6.50 µm. Both generalize across all three brains with near-identical percentiles (95th 5.79–5.85 µm; 99th 6.41–6.50 µm).
- **Agreement:** Both reports independently confirm tight characteristic scale of true split gaps; older report's median 19.67 µm and newer report's 100%-under-15 µm + 99th-percentile 6.5 µm are mutually consistent (the medians differ because older H35 counts long-tail gaps in the median, while newer H63 reports ECDF percentiles).
- **Disagreement:** Headline interpretation. **Older H35 concludes "fixed-radius nearest-neighbor heuristics will fail"** because gap distribution (median 19.67 µm, tail to 250 µm) is much wider than intra-fragment edges (95th-pctl 5.77 µm); any radius wide enough to cover splits will also sweep false positives. **Newer H63 concludes the opposite: a tight ~6.5 µm radius covers 99% of true splits.** The contradiction is reconciled by the choice of control — H35 compares to intra-fragment internal edges (which are even shorter, so distance alone cannot separate them from splits), while H63 reports the absolute gap distribution. The newer finding #1 (id 30) then explicitly resolves the question by showing distance alone separates true splits from inter-NEURON gaps at AUC 0.998.
- **Verdict carried over:** Reproduction REPRODUCED on both; Generalization GENERALIZES on both; Verdicts OK on both.
- **Caveats:** Older H35's mechanistic warning ("fixed-radius heuristics fail") was conditioned on a specific control set; the newer reports re-evaluate and partially reverse that conclusion.

### 22. (Priority 0.287 · Surprise 0.307) Geometric-targeted graph severing at predicted merge coordinates removes ~86% of merge edges while raising edge accuracy from 82.3% to 93.7%.
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 39) — only the older report finds this specific MERGE-CUT result. The newer report's id 39 (which shares the same id but is the A\* SPLIT-REPAIR finding, listed above as #8) does NOT corroborate this merge-cut result.
- **Conclusion:** Cutting the physical fragments-graph node nearest each predicted merge coordinate (via KDTree) dropped global mean %-merged-edges from 13.25% to 1.83% (~86% reduction) and raised mean edge accuracy from 82.27% to 93.66% on 12 neurons; the most heavily merged neurons (N013, N018) went from ~40–43% merged to ~1% merged. Generalization PARTIAL: 794491 drops 20.54% → 6.15% (~70% reduction, below 80% headline threshold); 794495 28.54% → 16.02% (~44% reduction). Corrected paired Wilcoxon on per-neuron deltas: p ≤ 2e-3 on every brain; total fraction merged-edges removed CI = [0.7007, 0.9631] origin, [0.5164, 0.8555] 794491, [0.2264, 0.6461] 794495 (CI lower bound below 80% even on origin).
- **Agreement:** Single-source within this corpus, but the qualitative claim (targeted severing always significantly reduces merges and raises accuracy, every neuron improves) is robust.
- **Disagreement:** Despite same id-number, newer report's "id 39" is a different intervention (A\* repair of splits, not merge severing) and reports a much smaller end-to-end EA gain (+0.42% vs older's +11.4%). They are different actionable primitives.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization PARTIAL; Verdict MAJOR (downgraded — quantitative ≥80% claim does not transfer); Post-correction verdict WEAKENED.
- **Caveats:** Headline ">80% merge-edge reduction" is brain-specific; should be reframed as "44–86% reduction depending on brain". Intervention is evaluated against GROUND-TRUTH merge coordinates, not predicted merge coordinates as would be needed in practice.

### 23. (Priority 0.287 · Surprise 0.307) Omit errors form contiguous runs along GT skeletons, with neighbor-conditional probability ~28× the marginal rate.
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 48) — only report to find this directly. Closely related to the newer report's id 58 (bridged vs broken omit gaps, finding #11 above), which also describes the contiguous-streak structure of omits.
- **Conclusion:** Marginal P(OMIT) = 0.0317; P(OMIT | neighbor = OMIT) = 0.8908 — a 28.09× ratio, far above the hypothesized 3× threshold (χ² = 2.23M, p ≈ 0). Omits are not independent events. Generalizes: 794491 ratio 18.63×; 794495 ratio 40.42× (all p ≈ 0, all >> 3×). Corrected unordered-pair chi-square = 1,113,217.84 (exactly half recorded, confirming a reciprocal-double-count bug); cluster-bootstrap CI on the ratio [16.85, 47.40] origin; cluster-permutation p = 3.3e-03 on each brain.
- **Agreement:** The structural pattern is also implicit in the newer report's id 58 finding that "short bridged omit paths" exist as internal dropouts (a manifestation of streakiness). Treated as a separate finding because the unique-to-older H48 measures the Markov-style transition probability directly.
- **Disagreement:** None.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES; Verdict MINOR; Post-correction verdict UPHELD.
- **Caveats:** Recorded code's reciprocal undirected-pair counting doubled the chi-square (not the ratio); within-skeleton adjacency violates independence but the 28× ratio is sample-size-insensitive.

### 24. (Priority 0.287 · Surprise 0.307) Split edges have higher local tortuosity than correctly reconstructed edges (small but robust effect).
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 59 [10-hop window]; id 73 [5-hop window]) — both within the newer report; not corroborated by the older report. Listed here because the two newer-report entries explicitly cross-confirm each other (id 73 is framed as "Independent reconfirmation" of id 59).
- **Conclusion:** Split edges show statistically significant higher tortuosity than correct edges, but the per-edge effect is small. 10-hop (H59): median 1.1115 vs 1.0764 (point-biserial r = 0.0335, p ≈ 0). 5-hop (H73): median 1.1132 vs 1.0801, U = 4.71e9, p = 1.66e-276. Generalizes across all three brains at the same small magnitude (r = 0.025–0.061). Corrected Cliff's delta = +0.21 to +0.28 ("small" by Vargha-Delaney) on all three brains; per-neuron paired Wilcoxon p ≤ 2e-3 on each.
- **Agreement:** The 5-hop and 10-hop variants give numerically identical conclusions across all three brains.
- **Disagreement:** None across reports because only the newer report tested tortuosity.
- **Verdict carried over:** Reproduction REPRODUCED on both; Generalization GENERALIZES on both; Verdicts MAJOR on both (large-n trivial-effect significance trap); Post-correction verdict WEAKENED (id 59) / UPHELD (id 73) — direction confirmed, effect-size is "small but real".
- **Caveats:** id 73 reuses the same dataset and a near-identical metric as id 59 — not truly independent; tortuosity windows overlap (consecutive edges share 4–9 hops), violating independence; per-edge r ≈ 0.03 means a tortuosity-only classifier is useless in practice.

### 25. (Priority 0.287 · Surprise 0.307) Split-gap connections obey a non-linear distance-angle trade-off, invalidating fixed-threshold heuristics — but does not generalize.
- **Sources:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 47) — only the older report. Conceptually superseded by the newer report's id 30 (finding #1) and id 33 (finding #16), which together show distance alone (AUC 0.998) and angle alone (AUC 0.93) each work as discriminators against inter-neuron controls — i.e., the older report's "richer geometry than a fixed threshold" claim is replaced by the newer report's "simple thresholds work after all".
- **Conclusion:** Logistic regression on 1,072 true-split pairs and 142 false-merge controls gave distance×angle interaction term coef = −5.96, p = 0.014 (LLR p = 1.63e-17, pseudo R² = 0.093). Decision boundary tolerates >90° turns under ~8 µm but collapses to 0° collinearity by 15 µm. Belief shifted Leaning True (0.6667) → Likely True (0.9231). Generalization DOES-NOT-GENERALIZE: 794491 x3 coef = −0.342, p = 0.800; 794495 x3 coef = −1.454, p = 0.565. Corrected LR-vs-additive test: origin LR=6.6454, p=9.94e-03 (UPHELD on origin); extras p_LR=0.797 and p_LR=1.000 (does not transfer).
- **Agreement:** Single-source within this corpus.
- **Disagreement:** With the newer report, implicitly: newer id 30 and id 33 show that simple distance-alone and angle-alone heuristics work (AUC 0.93–0.998) against inter-neuron controls. The older report's claim "a richer geometry than a fixed-threshold pairing rule" survives only on origin and only against False-Merge controls.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization DOES-NOT-GENERALIZE; Verdict MAJOR; Post-correction verdict UPHELD (on origin) but does not transfer.
- **Caveats:** Class imbalance ~7.5:1 (1,072 positives / 142 controls) inflates SE on the interaction term; the "non-linear trade-off" was generalized from a single brain to a general rule.

### 26. (Priority 0.253 · Surprise 0.284) Split edges are spatially closer to merge sites than correct edges (~170 µm) — but OVERTURNED on cluster correction.
- **Sources:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 61) — only the newer report. Symmetric to unique finding #7 (id 36, omit-near-merge), and both reach OVERTURNED post-correction. Listed in the corroborated section because it pairs with finding #7 to define a unified (if discredited) "joint failure zones" hypothesis from the newer report.
- **Conclusion:** Split-to-merge mean distance 2,084.76 µm (median 1,794.64 µm) vs correct 2,265.14 µm (median 1,956.93 µm); Mann-Whitney U p = 3.50e-38, Welch's t p = 7.14e-27 on 6,805 split and 1,109,034 correct edges. Belief shifted Leaning True (0.7083) → Likely True (0.8906). Generalization GENERALIZES directionally on all three brains (split closer on all; gap shrinks to ~29 µm on 794491). Corrected cluster-bootstrap by neuron: gap CI [−373.03, +15.55] crosses zero on origin, p = 0.1067 (NS); none of the three brains reach cluster-significance under the corrected test.
- **Agreement:** Paired with finding #7 (id 36) under the "joint failure zones" frame; both are sample-size-driven, both fail cluster correction, and both have one brain showing direction reversal or absence.
- **Disagreement:** None across reports (single-source) but the post-correction verdict OVERTURNED disagrees with the original headline.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES (uncorrected) but Post-correction verdict OVERTURNED.
- **Caveats:** ~170 µm gap on ~2,000 µm baseline (~8% relative shift); significance was driven by treating 1.1M edges as i.i.d.; mechanistic "joint failure zones" claim is unsupported once clustering is respected.

---

## Excluded as Redundant

The following entries were folded into the canonical findings above (non-canonical members of each cluster). Listed as `<run abbreviation> id <id>` → merged into finding #N.

**Older report (`ground-truth-error-annotations-revised-version_2026-06-17`):**
- id 23 (15 µm fragment density at merge sites, 75% elevation) → merged into finding #13 (merge-crowding cluster).
- id 49 (~86% of merge sites within 10 µm of a fragments-graph branch point) → merged into finding #13.
- id 53 (merge-causing fragment segments have ~2× branch density) → merged into finding #13.
- id 81 (merge sites have ~55% higher volumetric fragment-node density) → merged into finding #13.
- id 19 (split errors spatially correlated with branch nodes, mean distance ~106 µm shorter) → merged into finding #14 (split-near-branch cluster).
- id 72 (split errors topologically biased toward GT branch nodes, median ~half) → merged into finding #14.
- id 65 (split edges occur on thinner processes — median topological distance to leaf 38% shorter) → merged into finding #17 (split-distal cluster).

**Newer report (`run-4--ground-truth-error-annotations-revised-version_2026-06-20`):**
- id 84 (merge sites carry a strong geometric "tangle" signature — local branch density ~28× higher) → merged into finding #13.
- id 64 (split errors cluster near topological branch points, mean geodesic ~516 µm vs ~711 µm) → merged into finding #14.
- id 73 (independent reconfirmation: split edges show higher local 5-hop tortuosity) → merged into finding #24 (newer-only tortuosity cluster).
- id 85 (omit errors nearly twice as likely on terminal edges as on internal edges) → merged into finding #18 (terminal omit cluster; numerically identical to older id 64).

No other entries were folded; every remaining entry stands as its own finding because (a) it tests a distinct error type (split vs merge vs omit), (b) it tests a distinct mechanism (e.g. distance vs angle vs branch order vs Z-axis vs depth), or (c) it reaches a verdict not subsumed by another entry.
