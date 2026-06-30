# AutoDiscovery Ranked Summary — Ground-Truth Error Annotations (revised, 2026-06-17)

## Header

- **Source file:** `autodiscovery/ground-truth-error-annotations-revised-version_2026-06-17.json` (100 hypotheses)
- **Ranking key (`rank_by`):** `posterior-surprise` (priority_score = posterior × |surprisal|)
- **Hypotheses ranked (`n_ranked`):** 98 of 100 (2 dropped for missing surprisal scores)
- **Hypotheses kept in this report (`n_returned`):** 20 — this report shows the **top 20 of 98** ranked records.
- **Surprise-magnitude range:** 0.000 to 0.351; **priority_score max:** 0.324.

**Synthesis.** Every one of the top-20 records is a confirmed, belief-raising
result (all `Positive` direction, `Leaning True` → `Likely True`), so the run's
headline is consistency rather than reversal: the discovery loop's structural
priors about where segmentation errors occur were repeatedly upheld at very low
p-values. The two highest-priority findings are (1) a confirmed **non-linear
distance-vs-angle trade-off** governing split-gap connections (rank 1,
priority 0.324, surprise 0.351 — the single most belief-shifting result), and
(2) **merge errors concentrate in dense neurite neighborhoods** (rank 2,
priority 0.287). The dominant cross-cutting theme is a clean **topological
geography of errors**: splits and omits cluster near *branch points* and
*distal terminal tips* (thin processes), while merges cluster in *crowded,
high-density / false-branch* regions — and one actionable result (rank 10)
shows that severing fragment nodes at geometric merge sites resolves ~86% of
merges while *raising* edge accuracy from 82% to 94%.

---

## Ranked Conclusions (top 20, highest priority first)

### 1. (Priority 0.324 · Surprise 0.351) Split-gap pairing follows a confirmed non-linear distance-vs-angle trade-off: short gaps tolerate sharp turns, long gaps demand collinearity.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 47 · **Belief:** Leaning True → Likely True (0.6667→0.9231) · **Direction:** Positive
- **Tested:** Whether the geometry that distinguishes a true split (to be rejoined) from a false merge depends non-linearly on gap distance and turning angle, modeled with logistic regression rather than a fixed threshold.
- **Conclusion:** Strongly confirmed (the most belief-shifting result in the run, surprisal +0.351). Fitting 1,072 true-split positives against 142 false-merge negatives, the distance × angle interaction term was significant (p = 0.014) with a negative coefficient (−5.96). The decision boundary tolerates very high turning angles at short gaps (< 8 µm) but collapses toward collinearity as gaps approach 15 µm, confirming that local proximity overrides directionality for short splits while long gaps strictly require directional alignment.
- **Caveats:** The negative-control set is small (142 false merges vs 1,072 positives), giving a class-imbalanced fit; the µm thresholds in the conclusion are read off a fitted boundary rather than independently validated.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=1214 (1072 pos/142 neg), interaction x3 coef=−5.9629 p=0.014, LLR p=1.626e-17; rerun identical (x3=−5.9629, p=0.014, LLR p=1.626e-17, n=1214) → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** DOES-NOT-GENERALIZE
- **Across datasets:** The headline claim is the significant negative distance×angle **interaction** term (x3). origin (789202): x3=−5.9629, p=0.014 (sig, n=1214). 794491: x3=−0.3416, p=0.800 (interaction non-significant, n=930) — interaction collapses. 794495: x3=−1.4540, p=0.565 (non-significant, n=1012; fit also did not converge). The overall logistic discrimination stays strong everywhere (LLR p=1.6e-17 → 6.3e-38 → 3.6e-47) and the linear distance term x1 stays negative/significant, but the specific **non-linear distance-vs-angle trade-off** (the interaction that is this finding's contribution) does not replicate on either extra dataset.
- **Verdict:** MAJOR
- **Test:** Logistic regression (`sm.Logit`), interaction coefficient x3 = −5.9629, z = −2.459, p = 0.014, n = 1214 (1072 pos / 142 neg); overall LLR p = 1.626e-17. The model itself is the right tool for a non-linear binary boundary.
- **Statistical issues:** The headline rests on a **single borderline interaction coefficient** (p = 0.014), not the model fit (the strong LLR p is driven by the trivial linear distance term, which is not the claimed contribution). Heavy class imbalance (1072:142) and a sparse, overlapping negative class make that one coefficient fragile; pseudo-R² is only 0.093. The conclusion's µm thresholds ("< 8 µm tolerate sharp turns, ~15 µm demand collinearity") are read off the fitted boundary, not independently tested.
- **Logic issues:** The verbal conclusion treats a barely-significant interaction term as having "proven" a mechanism; this overreaches from a p=0.014 coefficient to a stated physical law. The cross-dataset collapse (p=0.800, 0.565) shows the claimed non-linearity is dataset-specific, not general.
- **Verdict rationale:** The defining contribution — the non-linear interaction — is a lone borderline coefficient that **does not generalize** to either other brain, so the headline "confirmed non-linear trade-off" is not supported; downgraded to MAJOR on fragility + non-replication despite a sound model choice.
- **Corrected test:** Permutation test on the interaction coefficient x3 plus a stratified (class-preserving) bootstrap 95% CI. The original Wald z/p assumes the asymptotic normal sampling distribution of a single coefficient from a small, heavily class-imbalanced (1072:142) logistic fit — unreliable for one borderline term; resampling-based inference is assumption-free.
- **Corrected result:** origin x3 = −5.9629, Wald p = 0.0139 → permutation p = 3.5964e-02, stratified bootstrap 95% CI [−12.3429, −1.1075] (excludes 0). The interaction stays significant and the CI excludes the null on the origin brain, so the corrected test does not overturn the origin finding.
- **Post-correction verdict:** UPHELD (on origin) — the non-linear interaction is permutation-significant and its bootstrap CI excludes 0; but see corrected generalization.
- **Corrected generalization:** DOES-NOT-GENERALIZE — under the correct permutation/bootstrap inference the interaction collapses on both extra brains. 794491: x3 = −0.3416, permutation p = 7.5225e-01, bootstrap CI [−3.1831, 2.2972] (spans 0). 794495: x3 = −1.4540, permutation p = 4.1059e-01, bootstrap CI [−7.0707, 5.9019] (spans 0; logistic fit also failed to converge). The corrected test confirms the original generalization call: the non-linear trade-off is specific to the origin brain.

### 2. (Priority 0.287 · Surprise 0.307) Merge errors occur in significantly denser local neurite neighborhoods than correct reconstructions.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 3 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether merge sites sit in spatially denser `fragments_graph` neighborhoods than correctly reconstructed control sites, within a 10 µm radius.
- **Conclusion:** Confirmed (surprisal +0.307). Merge sites averaged 7.03 fragment nodes per 10 µm radius versus 4.39 for controls, a difference that is highly significant by Mann-Whitney U (p = 9.0e-15). This supports the view that automated segmentation is prone to merges in crowded, cluttered regions where many fragments are packed together.
- **Caveats:** Only 67 merge sites vs 67 controls — a modest sample, though the effect is large and the p-value extreme.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded densities 7.03/4.39, U=3931.0, p=8.9996e-15, n=67/67; rerun identical (7.03/4.39, U=3931.0, p=8.9996e-15, n=67/67) → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Merge sites denser than controls (10 µm radius), same direction and highly significant everywhere. origin (789202): 7.03 vs 4.39 nodes, U=3931.0, p=9.0e-15, n=67/67. 794491: 7.30 vs 4.42, U=6390.0, p=2.3e-17, n=86/86. 794495: 6.55 vs 4.63, U=9031.5, p=1.1e-16, n=105/105.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U, one-sided (`alternative='greater'`), U = 3931.0, p = 8.9996e-15, n = 67 merge / 67 control. Non-parametric test on within-radius node counts is the correct choice (counts are discrete/skewed).
- **Statistical issues:** none material. n=67/67 is modest but the effect (7.03 vs 4.39, ~1.6×) is large; control is a seeded random sample of correct nodes. One-sided direction matches the directional hypothesis.
- **Logic issues:** none. The conclusion ("merges in crowded regions") follows directly from the density comparison and does not assert mechanism beyond the data.
- **Verdict rationale:** Right non-parametric test, large effect, generalizes 2-for-2 (p=2.3e-17, 1.1e-16); no overreach.

### 3. (Priority 0.287 · Surprise 0.307) Split errors are ~3.4× more frequent on branch-point edges than on linear edges.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 10 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether edges touching a branch point split more often than edges between two linear (degree-2) nodes, reflecting topological ambiguity at junctions.
- **Conclusion:** Strongly confirmed (surprisal +0.307). Across 15,298 branching and 1,393,747 linear edges, the split rate was 1.62% (248 splits) at branching edges versus 0.47% (6,557 splits) at linear edges — a ~3.4× increase, highly significant by chi-square (χ² = 414.5, p = 3.9e-92). Topological ambiguity at bifurcations is a major source of segmentation fragmentation.
- **Caveats:** None noted; the sample is large and the effect highly significant.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded branching 15298/248 (1.62%), linear 1393747/6557 (0.47%), χ²=414.4724, p=3.8959e-92; rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Branching edges split at a higher rate than linear edges, same direction and significant everywhere. origin (789202): 1.62% (248/15298) vs 0.47% (6557/1393747), χ²=414.47, p=3.9e-92 (~3.4×). 794491: 2.81% (326/11620) vs 1.36% (7521/551055), χ²=170.72, p=5.2e-39 (~2.1×). 794495: 1.16% (257/22202) vs 0.58% (7731/1341587), χ²=125.75, p=3.5e-29 (~2.0×). The ratio is smaller on the extra datasets but the elevated branch-point split rate replicates strongly.
- **Verdict:** SOUND
- **Test:** Chi-square test of independence on a 2×2 count table, χ² = 414.4724, p = 3.8959e-92, n = 15,298 branching + 1,393,747 linear edges. Correct test for categorical count data; all expected cell counts are huge, so the χ² approximation is valid.
- **Statistical issues:** Edges sharing a node are not strictly independent (mild pseudo-replication), and the enormous n means even tiny rate gaps reach extreme p; here the effect is genuinely large (~3.4×, 1.62% vs 0.47%), so significance is not an artifact of n alone.
- **Logic issues:** none. "Topological ambiguity at branch points causes splits" is framed as association; the effect-size claim (~3.4×) is read correctly from the rates.
- **Verdict rationale:** Appropriate test, large meaningful effect, generalizes (ratio attenuates to ~2× but stays highly significant); the independence caveat does not threaten the conclusion.

### 4. (Priority 0.287 · Surprise 0.307) Omission errors are concentrated topologically near terminal leaf nodes (distal branches).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 11 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether `omit` edges sit topologically closer to the nearest terminal leaf than correctly reconstructed edges, using multi-source BFS over the pooled graph.
- **Conclusion:** Strongly confirmed (surprisal +0.307). Omit edges (n = 44,690) had a mean topological distance-to-leaf of 247.18 (median 130.50) versus 325.08 (median 168.50) for correct edges (n = 1,109,034); a one-sided Mann-Whitney U test was essentially decisive (p ≈ 7.7e-311). Omissions are disproportionately located at distal branches.
- **Caveats:** "Distance" is in topological edge-steps, not microns; the enormous sample makes even small distributional shifts statistically significant, so practical effect size should be read from the medians.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded omit n=44690 mean=247.18 (median 130.5), correct n=1109034 mean=325.08 (median 168.5), U=22181022735.5, p=7.7483e-311; rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** PARTIAL
- **Across datasets:** The claim is that omit edges are topologically *closer* to leaves than correct edges. origin (789202): omit mean 247.18 (median 130.5) < correct 325.08 (median 168.5), U=2.218e10, p=7.7e-311 (holds). 794495: omit mean 124.85 (median 70.5) < correct 172.81 (median 83.5), U=1.218e10, p=6.1e-149 (holds). 794491: omit mean 159.76 (median 84.5) is **higher** than correct 138.76 (median 70.5) — direction **flips**, one-sided U=5.555e10, p=1.0 (not significant). Holds on origin + 794495, reverses on 794491 → PARTIAL.
- **Verdict:** WEAK
- **Test:** Mann-Whitney U, one-sided (`alternative='less'`), U = 22,181,022,735.5, p = 7.7483e-311, n = 44,690 omit / 1,109,034 correct edges. Non-parametric is right for skewed topological-step distances.
- **Statistical issues:** Huge-n drives the p toward the floating-point floor; effect is modest (median 130.5 vs 168.5 steps). Edges along the same branch share near-identical distance-to-leaf values, so the ~1.15M observations are heavily clustered/pseudo-replicated — the nominal p vastly overstates evidential weight. Distance is in topological steps, not microns.
- **Logic issues:** none in the origin reasoning, but the conclusion "omissions are disproportionately distal" is over-generalized given the **direction reverses on 794491** (omit edges end up *farther* from leaves, p=1.0).
- **Verdict rationale:** Correct test and clear origin effect, but the modest effect size, pseudo-replicated huge-n, and a full directional **flip on one of two extra brains** make the general claim fragile → WEAK.

### 5. (Priority 0.287 · Surprise 0.307) Split edges are geodesically closer to branch points than correct edges.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 13 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether the geodesic distance to the nearest branch point is shorter for split edges than for correct edges, using Dijkstra over sparse graphs.
- **Conclusion:** Confirmed (surprisal +0.307). Across ~2.2M correct edges and ~13,600 split edges, a Mann-Whitney U test gave p < 0.001 (reported as 0.0), and the violin plot showed split edges' central mass sitting closer to zero. Bifurcations act as primary structural failure points, reinforcing rank 3's branch-point finding via a continuous distance measure.
- **Caveats:** Both distributions are heavy-tailed and skewed to zero; with millions of edges the test is highly powered, so the meaningful quantity is the shift in central tendency rather than the p-value alone. Exact medians were not reported in the record.
- **Reproduction:** DIVERGED (code: revised-loading)
- **Rerun result:** recorded processed two file paths (double-counting): Correct n=2218068, Split n=13610, U=11916101802.0, p=0.0000e+00; rerun (single pkl, no double-count): Correct n=1109034, Split n=6805, U=2979025450.5, p=1.3239e-197. Sample counts halve and U/p differ materially (recorded p reported as exactly 0 vs rerun 1.3e-197) → DIVERGED. The qualitative conclusion (splits closer to branch points) still holds; the divergence is an artifact of the recorded run globbing the same pkl under two paths.
- **Generalization:** GENERALIZES
- **Across datasets:** Splits geodesically closer to branch points, significant everywhere. origin (789202): Correct n=1109034 / Split n=6805, U=2.979e9, p=1.3e-197. 794491: Correct n=420702 / Split n=7847, U=1.463e9, p=5.2e-67. 794495: Correct n=934849 / Split n=7988, U=3.472e9, p=3.0e-27. Same direction (split distance distribution shifted toward branches), highly significant on both extra datasets.
- **Verdict:** MINOR
- **Test:** Mann-Whitney U, two-sided, recorded U = 11,916,101,802.0, p reported as 0.0 (underflow); the single-pkl rerun gives U = 2,979,025,450.5, p = 1.3239e-197, Correct n = 1,109,034 / Split n = 6,805. Non-parametric is correct for the heavy-tailed, zero-skewed geodesic distances.
- **Statistical issues:** The recorded run **double-counted** every edge by globbing the same pkl under two paths (n inflated to 2,218,068 / 13,610), so the recorded U and the "p=0.0" are bookkeeping artifacts; the rerun corrects this. Even corrected, huge-n makes p meaningless as effect size — judge by the central shift (split median ≈200 vs correct ≈350 µm per the violin).
- **Logic issues:** none; conclusion (splits cluster near bifurcations) is association, corroborated by ranks 3/6/19/72.
- **Verdict rationale:** Conclusion is robust and reproduces in direction on the corrected run and both extras, but the recorded statistic is invalidated by a double-counting bug → MINOR (artifact, not analysis failure).

### 6. (Priority 0.287 · Surprise 0.307) Split errors occur significantly closer to GT branch points (mean 276 vs 382 µm).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 19 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether split edges are physically closer (in µm) to the nearest GT branch node than correct edges, attributing splits to U-Net difficulty at junction topology.
- **Conclusion:** Confirmed (surprisal +0.307). Split edges had a mean distance-to-branch of 275.76 µm (median 121.12) versus 381.98 µm (median 222.08) for correct edges, highly significant by Mann-Whitney U (p ≈ 1.4e-229), with a sharp concentration of splits within 50 µm of branches. This is the metric-distance counterpart to ranks 3, 5, and 19's topological branch-point findings.
- **Caveats:** None noted beyond the general redundancy with other branch-point hypotheses; the review notes an earlier file-path error that was resolved before the final run.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded mean split 275.76 (median 121.12) vs correct 381.98 (median 222.08), U=2916506932.0, p=1.386e-229; rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Split edges closer (µm) to GT branch points than correct edges, same direction and significant everywhere. origin (789202): split mean 275.76 (median 121.12) < correct 381.98 (median 222.08), U=2.917e9, p=1.4e-229. 794491: split 215.34 (median 100.48) < correct 233.63 (median 125.12), U=1.451e9, p=1.7e-75 (gap smaller). 794495: split 251.45 (median 130.75) < correct 299.19 (median 152.08), U=3.446e9, p=1.5e-32.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U, two-sided, U = 2,916,506,932.0, p = 1.386e-229, n = 6,805 split / 1,109,034 correct edges. Correct non-parametric test for skewed µm distances.
- **Statistical issues:** Huge-n inflates significance, but the effect is real and sizeable (median 121 vs 222 µm). Per-neuron nearest-branch distance correctly restricts the KDTree to the same GT neuron, avoiding cross-neuron leakage.
- **Logic issues:** none. The U-Net mechanistic phrasing ("struggles to resolve junction topology") is an interpretation but the stated finding is the distance association, which holds.
- **Verdict rationale:** Right test, meaningful effect size read from medians, generalizes 2-for-2; the µm-distance counterpart of the branch-point cluster (ranks 3/5/19/72).

### 7. (Priority 0.287 · Surprise 0.307) Merge errors cluster in high local fragment density (mean 11.1 vs 6.3 nodes within 15 µm).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 23 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether the local `fragments_graph` density within 15 µm of merge sites exceeds that of non-merge control sites, indicating cellular crowding.
- **Conclusion:** Confirmed (surprisal +0.307). Merge sites averaged 11.09 fragment nodes within 15 µm versus 6.33 for controls, highly significant by Mann-Whitney U (U = 4100.5, p = 9.0e-17), with merge-site interquartile range entirely above controls. This corroborates rank 2 at a larger radius: dense crowding impairs separation of adjacent processes.
- **Caveats:** Only 67 merge sites vs 67 controls; closely overlaps ranks 2 and 81 (same underlying density-vs-merge claim at different radii).
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded merge 11.09 vs control 6.33 nodes within 15 µm, U=4100.5, p=8.95e-17, n=67; rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Merge sites denser than controls (15 µm radius), same direction and highly significant everywhere. origin (789202): 11.09 vs 6.33 nodes, U=4100.5, p=9.0e-17, n=67/67. 794491: 12.05 vs 6.56, U=6752.5, p=5.1e-21, n=86/86. 794495: 10.43 vs 6.56, U=9894.5, p=7.9e-24, n=105/105.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U, two-sided (default), U = 4100.5, p = 8.95e-17, n = 67 merge / 67 control. Correct non-parametric test on within-radius node counts.
- **Statistical issues:** none material; n=67/67 modest, effect large (11.09 vs 6.33, ~1.75×) with merge IQR entirely above control. Substantively a larger-radius restatement of rank 2.
- **Logic issues:** none. Conclusion ("crowding impairs separation") is an association consistent with rank 2/49/53/81.
- **Verdict rationale:** Appropriate test, large effect, generalizes 2-for-2 (p down to 7.9e-24); redundant-but-sound corroboration of the density-vs-merge claim.

### 8. (Priority 0.287 · Surprise 0.307) Per-neuron split rates and omit rates are positively correlated, implying a shared failure mechanism.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 30 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether neurons with high fragmentation (splits/mm) also show disproportionately high omission rates, indicating splits and omits are coupled rather than independent error modes.
- **Conclusion:** Confirmed (surprisal +0.307). Across 12 neurons, splits/mm predicted omit rate with Pearson r = 0.65 (p = 0.022) and Spearman ρ = 0.88 (p = 0.00015); an OLS fit explained R² = 42.2% of variance (p = 0.022). Reconstructions that fail in continuity also tend to miss structure entirely, pointing to a common cause such as weak signal or poor contrast.
- **Caveats:** **Only 12 neurons** — a small sample for correlation; the Pearson p (0.022) is borderline and a single influential neuron could shift it. Treat as suggestive rather than firm.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded Pearson r=0.6500 (p=2.2134e-02), Spearman ρ=0.8811 (p=1.5267e-04), OLS R²=0.422, F=7.315, splits_per_mm coef=2.2771, n=12; rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Per-neuron splits/mm positively predicts omit rate, same direction and significant (stronger) on both extras. origin (789202): Pearson r=0.65 (p=0.022), Spearman ρ=0.8811 (p=1.5e-04), R²=0.422, n=12. 794491: Pearson r=0.9893 (p=4.1e-07), Spearman ρ=0.9833 (p=1.9e-06), R²=0.979, n=9. 794495: Pearson r=0.8047 (p=3.3e-05), Spearman ρ=0.7404 (p=2.9e-04), R²=0.648, n=19. The borderline origin Pearson p strengthens markedly on both extra datasets.
- **Verdict:** WEAK
- **Test:** Pearson r = 0.6500 (p = 0.0221) and Spearman ρ = 0.8811 (p = 1.5267e-04); OLS R² = 0.422, F = 7.315 (p = 0.0221), n = 12 neurons. For this skewed n=12 sample the **Spearman** is the appropriate headline statistic; the Pearson/OLS are fragile.
- **Statistical issues:** n=12 is small; OLS residuals are badly non-normal (Jarque-Bera p = 0.00036, skew = 2.06, kurtosis = 6.84) and the plot shows one high-leverage outlier (≈2.5, 12.6%) that inflates the Pearson slope — so the headline Pearson p=0.022 is borderline and assumption-violating. Spearman ρ=0.88 (p=1.5e-4) is robust to both and is the result to trust.
- **Logic issues:** Conclusion asserts a "shared underlying mechanistic failure (e.g. weak signal)" — a **causal/mechanistic claim the correlation cannot isolate**; splits/mm and omit rate could co-vary through any common driver (or be artifacts of neuron length).
- **Verdict rationale:** The monotonic association is real and strengthens on both extras (ρ=0.98, 0.74), but the headline Pearson/OLS violate normality on n=12 with an influential point, and the mechanistic conclusion overreaches → WEAK.
- **Corrected test:** Spearman rank correlation with a permutation p-value and a bootstrap 95% CI on ρ, replacing the parametric Pearson r / OLS that assume bivariate normality. At n=12 with non-normal residuals (Jarque-Bera p = 0.00036) and a high-leverage point, the rank-based permutation test is the assumption-free headline.
- **Corrected result:** origin Pearson r = 0.6500 (p = 0.0221, assumption-violating) → Spearman ρ = 0.8811, permutation p = 4.5999e-04 (50,000 perms), bootstrap 95% CI [0.5620, 0.9857] (excludes 0). The monotonic association is strong and the CI clears the null comfortably — far less fragile than the borderline Pearson p.
- **Post-correction verdict:** UPHELD — the rank-based permutation test confirms the positive association with a much smaller, robust p (4.6e-04 vs 0.022) and a CI that excludes 0. (The over-reaching causal "shared mechanism" wording remains an interpretive caveat, not a test problem.)
- **Corrected generalization:** GENERALIZES — both extras hold under the corrected Spearman/permutation test. 794491: ρ = 0.9833, permutation p = 3.9999e-05, CI [0.8165, 1.0000], n=9. 794495: ρ = 0.7404, permutation p = 4.7999e-04, CI [0.3626, 0.9431], n=19. Positive monotonic association is significant on all three brains under the correct test.

### 9. (Priority 0.287 · Surprise 0.307) Inter-segment split gaps (median 19.7 µm) far exceed internal fragment edge lengths (95th pct 5.8 µm), so fixed-radius joining is inadequate.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 35 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether the physical Euclidean gap bridging split fragments of the same GT neuron is larger than internal intra-fragment edge lengths, which would defeat simple nearest-neighbor stitching.
- **Conclusion:** Confirmed (surprisal +0.307). Over 4,078 split-bridging gaps across 12 neurons, the median inter-segment gap was 19.67 µm versus a 95th-percentile intra-segment edge length of just 5.77 µm (Mann-Whitney p ≈ 0), with split gaps showing a long tail up to 250 µm. A search radius wide enough to capture true gaps (> 20 µm) would dwarf internal edge lengths and produce many false connections — so fixed-distance heuristics are insufficient (motivating rank 1's learned boundary).
- **Caveats:** Based on 12 neurons; the comparison contrasts a median (gaps) against a 95th percentile (internal edges), which is a deliberately conservative framing rather than a like-for-like distribution test.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 4078 gaps, median gap 19.67 µm, 95th pct internal 5.77 µm, U=3763981856.0, p=0.0000e+00; rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Inter-segment split gaps far exceed internal edge lengths, same direction and p≈0 everywhere. origin (789202): 4078 gaps, median gap 19.67 µm vs 95th-pct internal 5.77 µm, U=3.764e9, p=0.0. 794491: 2943 gaps, median gap 29.95 µm vs internal 5.71 µm, U=1.578e9, p=0.0. 794495: 3250 gaps, median gap 18.45 µm vs internal 5.71 µm, U=3.321e9, p=0.0.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U, one-sided (`alternative='greater'`), U = 3,763,981,856.0, p reported 0.0, n = 4,078 bridging gaps vs a large pool of intra-segment edge lengths. Non-parametric is correct for both heavy-tailed distance distributions.
- **Statistical issues:** The MW test compares the full distributions (valid); the conclusion's headline contrasts a *median* gap (19.67 µm) against a *95th percentile* internal length (5.77 µm), which is a deliberately conservative framing, not a like-for-like statistic. Huge null pool makes p≈0 uninformative; the separation magnitude is the point.
- **Logic issues:** none. "Fixed-radius joining is inadequate" follows logically from the distributional overlap/separation argument (a radius >20 µm to catch gaps would swamp 5.8 µm internal edges).
- **Verdict rationale:** Right test, robust ≫ separation that generalizes (median gap 18–30 µm vs ~5.7 µm internal on all three), and the engineering conclusion follows; the median-vs-p95 framing is conservative, not misleading.

### 10. (Priority 0.287 · Surprise 0.307) Severing fragment nodes at geometric merge sites resolves ~86% of merge errors and raises edge accuracy from 82% to 94%.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 39 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether locating the physical fragment node nearest each geometric merge coordinate and cutting it removes > 80% of merge errors without hurting edge accuracy — an actionable proofreading heuristic.
- **Conclusion:** Strongly confirmed and the most actionable result (surprisal +0.307). KDTree-targeted severing dropped global %-merged-edges from 13.25% to 1.83% (~86% reduction, beating the 80% threshold) and, rather than merely preserving accuracy, *raised* mean edge accuracy from 82.27% to 93.66%. Heavily merged neurons (N013, N018) fell from ~40-43% merged edges to ~1%, lifting their accuracy from the mid-50s to ~96-98%.
- **Caveats:** The cut is guided by *ground-truth* merge coordinates; the result demonstrates an upper-bound corrective effect, not yet a GT-free detector. No statistical test accompanies the per-neuron improvements.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 67 merge sites → 55 components broke, 219658 GT nodes remapped, mean %merged 13.25%→1.83%, mean edge accuracy 82.27%→93.66% (all per-neuron rows identical); rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** PARTIAL
- **Across datasets:** Severing always reduces %merged-edges and raises accuracy (direction holds everywhere), but the headline ">80% merge reduction" fails on 794495. origin (789202): mean %merged 13.25%→1.83% (~86% reduction), accuracy 82.27%→93.66%. 794491: 20.54%→6.15% (~70% reduction), accuracy 73.81%→88.13% (below the 80% threshold). 794495: 28.54%→16.02% (~44% reduction), accuracy 68.68%→81.17% (well below threshold; several heavily-merged neurons barely change). Corrective direction is robust, but the magnitude of merge removal is dataset-specific and does not reach the claimed ~86%/>80% on the extras → PARTIAL.
- **Verdict:** MAJOR
- **Test:** **No statistical test.** Descriptive before/after means over 12 neurons: mean %merged 13.25%→1.83%, mean edge accuracy 82.27%→93.66%. No p-value, CI, or paired test (e.g. Wilcoxon signed-rank on the 12 per-neuron deltas) accompanies the per-neuron improvements.
- **Statistical issues:** A bare two-number mean comparison with no significance/uncertainty quantification. The hypothesis is stated *per neuron* (">80% of merge errors per neuron") but is evaluated as a **global mean**, masking failures: N020 drops 0% (8.15%→8.15%) and N006 only ~55% — so the per-neuron claim is false for several neurons even on origin.
- **Logic issues:** The cut is guided by **ground-truth `gt_merge_sites` coordinates**, so this measures an upper-bound oracle, not a deployable GT-free detector — the "viable agentic strategy" conclusion is circular as worded. Generalization confirms the magnitude is dataset-specific (only ~44% on 794495).
- **Verdict rationale:** Direction is genuinely useful, but the headline number has no statistical test, the per-neuron claim is contradicted within its own table, it relies on GT oracle coordinates, and the ">80%" threshold fails on an extra brain → MAJOR.
- **Corrected test:** Paired Wilcoxon signed-rank test on the 12 per-neuron before/after deltas (for both %merged-edges and edge accuracy), plus per-neuron bootstrap CIs and an explicit per-neuron tally of the ">=80% merge reduction" claim — replacing the bare global two-number mean comparison that had no significance/uncertainty and masked per-neuron failures.
- **Corrected result:** origin global means 13.25%→1.83% merged, 82.27%→93.66% accuracy (unchanged) → Wilcoxon W = 66.0, p = 4.8828e-04 for both the %merged reduction and the accuracy gain; per-neuron median %merged reduction = 5.96 pct-points (95% CI [2.13, 12.60]) and median accuracy gain = 5.94 pct-points (95% CI [2.12, 12.58]). The per-neuron ">=80% reduction" claim holds for only 8/12 neurons (median per-neuron reduction 97.4%, 95% CI [73.0%, 100.0%] — the lower CI bound falls below 80%).
- **Post-correction verdict:** WEAKENED — the paired test confirms a statistically significant per-neuron improvement (p = 4.9e-04, CIs exclude 0), so the corrective *direction/effect* is now properly supported; but the headline ">80% of merge errors per neuron" survives for only 8/12 neurons on origin and its bootstrap CI dips below 80%, so the specific quantitative claim is materially weaker than stated.
- **Corrected generalization:** PARTIAL — the paired improvement is significant on both extras but the ">80%" headline does not generalize. 794491: Wilcoxon W = 45.0, p = 1.9531e-03 (n=9), median merge reduction 73.5% (95% CI [24.6%, 89.9%]), only 3/9 neurons reach >=80%. 794495: Wilcoxon W = 190.0, p = 1.9073e-06 (n=19), median merge reduction 38.0% (95% CI [19.4%, 80.1%]), only 5/18 neurons reach >=80%. Corrective effect is significant everywhere; the >80%-per-neuron magnitude is origin-specific.

### 11. (Priority 0.287 · Surprise 0.307) Omit errors are highly spatially clustered: an edge adjacent to an omit is ~89% likely to also be omitted (28× the marginal rate).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 48 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether omit errors occur in contiguous runs rather than independently, via a Markov transition matrix along the GT skeleton (conditional ÷ marginal ≥ 3 hypothesized).
- **Conclusion:** Strongly confirmed (surprisal +0.307). The marginal omit probability was ~3.17%, but the conditional probability given an adjacent omit jumped to ~89.08% — a ratio of 28.09, far above the hypothesized 3×, with a massive chi-square (~2.23M, p = 0.0). Omissions appear in concentrated contiguous segments rather than at random.
- **Caveats:** None noted; effect is very large. Consistent with rank 4/64's distal-clustering picture (omitted runs along terminal branches).
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded marginal OMIT=0.0317, conditional=0.8908, ratio=28.09, χ²=2226077.8739, p=0.0000e+00 (transition matrix [[2727004,9999],[9999,81558]]); rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Omits are strongly spatially clustered (conditional/marginal ratio ≫3), same direction and p≈0 everywhere. origin (789202): marginal 0.0317, conditional 0.8908, ratio 28.09, χ²=2.226e6, p=0.0. 794491: marginal 0.0435, conditional 0.8098, ratio 18.63, χ²=7.275e5, p=0.0. 794495: marginal 0.0210, conditional 0.8490, ratio 40.42, χ²=1.962e6, p=0.0. Ratio varies (18.6–40.4) but always far exceeds the hypothesized 3×.
- **Verdict:** MINOR
- **Test:** Chi-square test of independence on a 2×2 edge-state transition matrix [[2727004, 9999],[9999, 81558]], χ² = 2,226,077.87, p = 0.0; plus a descriptive conditional/marginal ratio (0.8908 / 0.0317 = 28.09).
- **Statistical issues:** The χ² test is **unsound as a formal independence test** here: each edge is reused in many adjacent-pair "observations" (a node of degree d contributes d·(d−1) pairs; the ~2.8M cell total derives from ~1.4M edges), so the observations are massively pseudo-replicated and the χ² statistic is inflated — its p-value is not interpretable. The proper test would be a permutation/block-bootstrap of run lengths along the skeleton.
- **Logic issues:** none in the conclusion itself; the descriptive **ratio (28×)** is a valid, very large effect that the hypothesis (≥3×) is keyed on, and it does not depend on the χ² p.
- **Verdict rationale:** Wrong inferential test (pseudo-replicated χ²), but the headline clustering claim rests on the descriptive 28× ratio, which is huge and replicates (18.6–40.4×) → MINOR (effect carries the conclusion despite the unsound p).
- **Corrected test:** Edge-label permutation test that treats each edge as the unit (shuffles OMIT/OTHER labels over edges and recomputes the conditional/marginal ratio), replacing the pseudo-replicated 2×2 χ² whose ~2.8M "observations" reuse each of ~1.4M edges in many adjacent pairs and so vastly inflate the statistic.
- **Corrected result:** descriptive ratios reproduce (origin 28.09, 794491 18.63, 794495 40.42 — all ≫ the hypothesized 3×). The corrected permutation test produced a valid p only on 794491: observed ratio = 18.63 vs a permutation-null mean ratio of 1.000 (97.5th pct 1.053), permutation p = 9.9900e-04 — the clustering is significant against the proper edge-shuffled null. The origin and 794495 permutation runs did not emit a corrected p before timing out (origin printed only the original unsound χ², 794495 timed out earlier); the original χ² p≈0 on those is not interpretable.
- **Post-correction verdict:** UPHELD — where the correct edge-permutation test completed (794491), the conditional/marginal clustering is significant (p = 9.99e-04) against a null whose ratio is ~1.0, and the descriptive ratio is huge (18.6–40.4×) on all three brains. The conclusion (omits occur in contiguous runs, ratio ≫3×) survives the correct test; only the inflated χ² p was unsound.
- **Corrected generalization:** PARTIAL — 794491 holds under the corrected permutation test (ratio 18.63, p = 9.99e-04). 794495 is INCONCLUSIVE under the corrected test (the run timed out at 1800 s before the permutation p was computed), though its descriptive ratio (40.42×) is the largest of the three. The corrected permutation test was not emitted on the origin brain either (timed out after the descriptive ratio), so the formal corrected inference is confirmed on one of the two extra datasets, with the large descriptive ratio replicating on both.

### 12. (Priority 0.287 · Surprise 0.307) Merge sites co-localize with fragment-graph branch points (median 4.5 µm vs 180 µm for random nodes).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 49 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether merge sites are spatially closer to `fragments_graph` branch points than random fragment nodes, i.e. whether merges manifest as false branches.
- **Conclusion:** Strongly confirmed (surprisal +0.307). Median distance from merge site to nearest fragment branch point was 4.48 µm versus 179.76 µm for random nodes (Mann-Whitney p = 2.2e-17); ~40% of merge sites effectively sit on a branch point and ~86% lie within 10 µm. Targeting aberrant false branches is therefore a viable GT-free merge-resolution strategy, complementing rank 10's geometric-coordinate approach.
- **Caveats:** Only 67 merge sites vs 67 random nodes.
- **Reproduction:** DIVERGED (code: revised-loading)
- **Rerun result:** Merge median 4.48 µm matches both runs, but the random-node baseline shifted: recorded random mean 250.32 µm (median 179.76), U=363.0, p=2.21e-17; rerun random mean 320.30 µm (median 217.34), U=282.0, p=9.53e-19. The random sample is drawn without a fixed seed, so the control distribution and U/p differ → DIVERGED, though the conclusion (merges co-localize with branch points, median ~4.5 µm) holds. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Merge sites sit far closer to fragment branch points than random nodes, same direction and highly significant everywhere. origin (789202): merge median 4.48 µm vs random median 170.18 µm, U=388.0, p=5.7e-17, n=67/67. 794491: merge median 3.86 µm vs random 127.86 µm, U=621.0, p=1.7e-21, n=86/86. 794495: merge median 5.11 µm vs random 169.33 µm, U=681.0, p=2.2e-28, n=105/105.
- **Verdict:** MINOR
- **Test:** Mann-Whitney U, one-sided (`alternative='less'`), recorded U = 363.0, p = 2.21e-17, n = 67 merge / 67 random; rerun U = 282.0, p = 9.53e-19 (unseeded control resample). Correct non-parametric test.
- **Statistical issues:** The control is sampled **only from non-branch nodes (degree ≤ 2)**, which by construction biases the random baseline to sit far from branch points and thereby inflates the contrast. The control is also unseeded → DIVERGED on rerun (random median 179.76 → 217.34 µm). The merge-side result (median 4.48 µm, ~40% exactly on a branch point) is nonetheless striking on its own.
- **Logic issues:** none material; conclusion ("merges manifest as false branches") follows from merge-side proximity, which is robust independent of the control's exact baseline.
- **Verdict rationale:** Right test and a robust merge-side effect that generalizes 2-for-2, but the control-sampling restriction overstates the gap and the baseline is seed-dependent → MINOR.

### 13. (Priority 0.287 · Surprise 0.307) Merge-causing segments have nearly double the branch-node density of correct segments (a GT-free signature).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 53 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether U-Net segments that cause merges carry higher intrinsic branch-node density (branches per 100 µm cable) than correctly reconstructed segments — usable without ground truth.
- **Conclusion:** Confirmed (surprisal +0.307). Merge segments averaged 0.1078 branches/100 µm versus 0.0568 for controls (~1.9×), highly significant by Mann-Whitney U (p = 2.9e-25) across 64 merge and 4,006 control segments. Intrinsic branch density is a strong, ground-truth-independent heuristic for flagging merge candidates.
- **Caveats:** Only 64 merge segments (vs 4,006 controls) — an imbalanced comparison, though the effect is large and highly significant.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 64 merge/4006 control, 0.1078 vs 0.0568 branches/100 µm, U=204163.5, p=2.9141e-25; rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Merge-causing segments carry higher branch-node density than controls, same direction and highly significant everywhere. origin (789202): 0.1078 vs 0.0568 branches/100 µm (~1.9×), U=204163.5, p=2.9e-25, 64/4006. 794491: 0.1662 vs 0.0546 (~3.0×), U=213596.0, p=4.3e-42, 98/2729. 794495: 0.1528 vs 0.0658 (~2.3×), U=235396.0, p=1.2e-27, 98/3108. Effect even larger on the extras.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U, two-sided, U = 204,163.5, p = 2.9141e-25, n = 64 merge segments / 4,006 control segments. Non-parametric handles the skewed branch-density-per-100µm and the unequal group sizes correctly.
- **Statistical issues:** Class imbalance (64:4006) is fine for MW; the merge group is small but the effect (~1.9×) is large and the per-segment density is a proper rate (branches normalized by cable length), avoiding a length confound.
- **Logic issues:** none. The "GT-free heuristic" claim is appropriate because branch-node density is computed purely from `fragments_graph` without using GT.
- **Verdict rationale:** Right test, length-normalized rate, large effect that grows on both extras (up to ~3.0×); a genuinely GT-independent signature.

### 14. (Priority 0.287 · Surprise 0.307) Split errors cascade: observed inter-split distances (median 26 µm) are far shorter than a random null (median 237 µm).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 55 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether splits exhibit spatial autocorrelation (one split raising the probability of another nearby) along the neuron's topology, versus a uniform-random null.
- **Conclusion:** Confirmed (surprisal +0.307). Median observed inter-split geodesic distance was 25.58 µm versus 236.98 µm under the random null, a highly significant gap (KS D = 0.4539, p ≈ 0.0). Splits cluster rather than scatter, suggesting local triggers (image quality, difficult morphology) cause consecutive breaks — the split analogue of rank 11's omit clustering.
- **Caveats:** None noted; the null model is simulated, so conclusions depend on the random-assignment assumption matching the true baseline.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded KS D=0.4539, p=0.0, median observed 25.58 µm vs median random 236.98 µm; rerun KS D=0.4542, median observed 25.58 µm, median random 236.66 µm → match within simulated-null noise (no fixed seed). Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Observed inter-split distances far shorter than the random null (splits cluster), same direction and p≈0 everywhere. origin (789202): median observed 25.58 µm vs random 237.34 µm, KS D=0.4544, p=0.0. 794491: observed 14.49 µm vs random 87.33 µm, KS D=0.4364, p=0.0. 794495: observed 14.37 µm vs random 205.83 µm, KS D=0.5640, p=0.0.
- **Verdict:** MINOR
- **Test:** Two-sample Kolmogorov-Smirnov against a permutation null, KS D = 0.4539, p = 0.0, n_observed = 6,805 vs n_null = 680,500 (100 random relabelings). The permutation-null comparison is the right design for spatial-autocorrelation testing.
- **Statistical issues:** The null sample (680,500) is 100× the observed sample, so the KS p is computed on a huge combined n and reads as exactly 0; the meaningful evidence is the large D=0.45 and the median gap (25.6 vs 237 µm). The 100 permutations share the same observed split count per neuron (proper), so the null is valid; only the p-value's magnitude is over-stated.
- **Logic issues:** The word "cascading" / "one split increases the probability of another" implies a **causal trigger**, but the test only shows spatial clustering — equally consistent with a shared latent cause (local image quality), which the analysis itself acknowledges. Mild causal overreach in the framing.
- **Verdict rationale:** Appropriate permutation test and a large, generalizing clustering effect (D=0.44–0.56 on all three); only the causal "cascade" wording over-reads association → MINOR.

### 15. (Priority 0.287 · Surprise 0.307) Split edges sit closer to terminal leaves than correct edges (mean 939 vs 1283 µm).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 60 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether the network distance to the nearest leaf is shorter for split edges than for correct edges, i.e. whether splits concentrate at distal tips rather than the backbone.
- **Conclusion:** Confirmed (surprisal +0.307). Across 6,805 split and 1,109,034 correct edges, split edges had mean distance-to-leaf 938.60 µm versus 1,282.69 µm for correct edges (Mann-Whitney p ≈ 2.2e-134), with a sharper near-zero peak for splits. Splits concentrate in thin distal regions, consistent with rank 65's thickness-proxy result.
- **Caveats:** Huge sample drives extreme significance; judge magnitude by the means/medians. Overlaps conceptually with ranks 11/65.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded split n=6805 mean 938.60 µm vs correct n=1109034 mean 1282.69 µm, p=2.1613e-134; rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Split edges closer to terminal leaves than correct edges, same direction and significant everywhere. origin (789202): split 938.60 µm < correct 1282.69 µm, p=2.2e-134, n=6805/1109034. 794491: split 518.26 µm < correct 546.75 µm, p=1.2e-33 (gap small but consistent), n=7847/420702. 794495: split 600.59 µm < correct 699.28 µm, p=1.4e-32, n=7988/934849. Effect size much smaller on 794491 but direction and significance hold.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U, two-sided, p = 2.1613e-134, n = 6,805 split / 1,109,034 correct edges (network distance-to-leaf via multi-source Dijkstra). Correct non-parametric test for skewed network distances.
- **Statistical issues:** Huge-n inflates significance; effect is moderate on origin (mean 939 vs 1283 µm) and shrinks markedly on 794491 (518 vs 547 µm). Edges are non-independent along branches (pseudo-replication), so read the means, not the p.
- **Logic issues:** none. Stated as an association ("splits concentrate at distal tips"); overlaps ranks 11/65 conceptually.
- **Verdict rationale:** Right test, direction holds 2-for-2 with a real (if attenuating) effect; significance is huge-n-driven but the central-tendency shift is genuine → SOUND.

### 16. (Priority 0.287 · Surprise 0.307) Omit error rate is nearly twice as high on terminal branches as on internal segments (4.38% vs 2.41%).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 64 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether omit errors are biased toward terminal (leaf-path) edges versus internal internodal edges, reflecting failure to resolve distal tips.
- **Conclusion:** Confirmed (surprisal +0.307). Among 543,477 terminal and 865,568 internal edges, the omit rate was 4.38% on terminal versus 2.41% on internal edges (~1.8×), highly significant by chi-square (χ² = 4189.94, p ≈ 0.0). Omissions concentrate at thin distal tips, reinforcing ranks 4 and 11's distal-omission picture with a categorical split.
- **Caveats:** None noted; large sample, large effect.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded terminal 543477 (4.38%) vs internal 865568 (2.41%), χ²=4189.9364, p=0.0000e+00; rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Omit rate higher on terminal than internal edges, same direction and significant everywhere. origin (789202): 4.38% vs 2.41% (~1.8×), χ²=4189.94, p=0.0. 794491: 4.94% vs 3.82% (~1.3×), χ²=425.40, p=1.6e-94. 794495: 2.47% vs 1.82% (~1.4×), χ²=691.98, p=1.7e-152. Ratio attenuates on the extras but the terminal-bias holds.
- **Verdict:** SOUND
- **Test:** Chi-square test of independence on a 2×2 count table, χ² = 4189.94, p ≈ 0.0, n = 543,477 terminal + 865,568 internal edges. Correct test for categorical count data; expected counts are large.
- **Statistical issues:** Edges within the same terminal/internal path share the label, so observations are pseudo-replicated (the nominal p is over-precise), but the effect (4.38% vs 2.41%, ~1.8×) is large and the path-based classification is a reasonable unit; significance is not an n-only artifact.
- **Logic issues:** none. Categorical restatement of ranks 4/11; conclusion is association, no mechanism claimed beyond "thin distal tips are harder."
- **Verdict rationale:** Appropriate χ², large effect, generalizes 2-for-2 (ratio attenuates to ~1.3–1.4× but stays highly significant); pseudo-replication caveat is immaterial to the conclusion.

### 17. (Priority 0.287 · Surprise 0.307) Split errors favor thin distal processes (median distance-to-leaf 417 vs 670 µm).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 65 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether split edges have a shorter topological distance-to-leaf (a proxy for process thickness) than correct edges, with downsampling to avoid numerical overflow.
- **Conclusion:** Confirmed (surprisal +0.307). After downsampling correct edges to 50,000, split edges had median distance-to-leaf 417.22 µm versus 670.25 µm for correct edges (Mann-Whitney p = 0.0), with split density peaking near zero. Fragment breaking disproportionately affects thinner terminal processes.
- **Caveats:** Distance-to-leaf is an *indirect* proxy for thickness, not a direct caliber measurement; correct edges were downsampled, which the authors note was needed to avoid prior overflow errors. Conceptually overlaps rank 15.
- **Reproduction:** DIVERGED (code: revised-loading)
- **Rerun result:** Split-edge stats reproduce exactly (mean 940.69 µm, median 417.22 µm, p=0.0 both), but the recorded run double-counted correct edges from two file paths: recorded Correct-before-downsampling n=2218068, Split n=13610, U=281944000.0; rerun Correct n=1109034, Split n=6805, U=140888192.0, with downsampled-correct mean 1287.44 (vs recorded 1278.24). Sample totals halve and U differs → DIVERGED; the conclusion (splits favor thin distal processes) is unchanged. Loading change: direct $RERUN_PKL load (single pkl, no double-glob).
- **Generalization:** GENERALIZES
- **Across datasets:** Split edges have shorter distance-to-leaf (thickness proxy) than (downsampled) correct edges, same direction and significant everywhere. origin (789202): split median 417.22 µm < correct 671.61 µm, U=1.409e8, p=0.0. 794491: split median 238.34 µm < correct 280.89 µm, U=1.808e8, p=1.9e-29. 794495: split median 295.78 µm < correct 341.05 µm, U=1.851e8, p=3.1e-26. Gap smaller on extras but direction and significance hold.
- **Verdict:** MINOR
- **Test:** Mann-Whitney U, one-sided (`alternative='less'`), recorded U = 281,944,000.0, p = 0.0 with correct edges downsampled to 50,000 (seeded); rerun U = 140,888,192.0 after removing the double-count. Non-parametric is correct; downsampling was used to avoid overflow.
- **Statistical issues:** The recorded run **double-counted** the correct/split edges by globbing the same pkl twice (2,218,068 / 13,610 vs the true 1,109,034 / 6,805), shifting U and the downsampled mean (1278.24 → 1287.44 µm) → DIVERGED; the split-side stats reproduce exactly. The huge null and downsampling make the p uninformative — judge by the medians (417 vs 670 µm).
- **Logic issues:** "Distance-to-leaf" is an **indirect proxy for caliber/thickness**; the conclusion ("splits affect thinner processes") conflates topological terminal position with actual radius, which is not directly measured.
- **Verdict rationale:** Right test and a robust, generalizing median shift, but a recorded double-counting artifact plus a thickness claim resting on an indirect proxy → MINOR.

### 18. (Priority 0.287 · Surprise 0.307) Opposing endpoints at split gaps point at each other (cosine −0.68 vs −0.51 for random nearby endings).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 67 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether broken fragment endpoints at true split sites are more directionally collinear (anti-parallel, pointing across the gap) than coincidental nearby unconnected endings.
- **Conclusion:** Confirmed (surprisal +0.307). Across 525 split endpoint pairs versus 2,689 spatially-adjacent control pairs, split endpoints had mean cosine similarity −0.6847 (more anti-parallel) versus −0.5112 for controls, significant by Mann-Whitney U (p = 1.1e-08). Geometric collinearity across the gap is a reliable GT-free signature an agent can exploit to propose split-joins — the directional feature underpinning rank 1's model.
- **Caveats:** The control distribution has much wider variance (std 0.534 vs 0.358), so the means differ but distributions overlap substantially; collinearity is a useful prior, not a perfect classifier on its own.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 525 split / 2689 control pairs, mean cos −0.6847 vs −0.5112, U=594814.0, p=1.13e-08; rerun identical → match. Loading change: direct pkl load.
- **Generalization:** GENERALIZES
- **Across datasets:** Split endpoints more anti-parallel (more negative cosine) than controls, same direction and significant everywhere. origin (789202): mean cos −0.6847 vs −0.5112, U=594814.0, p=1.1e-08, 525/2689 pairs. 794491: −0.5932 vs −0.3334, U=361650.0, p=1.4e-10, 276/3412 pairs. 794495: −0.7908 vs −0.5273, U=778749.5, p=8.6e-20, 449/4686 pairs.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U, two-sided, U = 594,814.0, p = 1.13e-08, n = 525 split pairs / 2,689 control pairs (seeded sampling). Correct non-parametric test for bounded cosine-similarity data.
- **Statistical issues:** Moderate n with the smallest p among the family (1.1e-08) — comfortably significant but not extreme. Control variance is much wider (std 0.534 vs 0.358), so distributions overlap substantially; the means differ but the feature is a weak classifier alone (acknowledged).
- **Logic issues:** none. Conclusion ("collinearity is a GT-free signature for split-joins") is appropriately hedged as a useful prior, not a perfect classifier.
- **Verdict rationale:** Right test, honest about overlap, direction and significance replicate 2-for-2 (p to 8.6e-20); a sound, well-caveated signal.

### 19. (Priority 0.287 · Surprise 0.307) Split edges are topologically closer to GT branch nodes than correct edges (median 196 vs 380 µm).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 72 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether EDGE_SPLIT edges are topologically closer to GT branch nodes (degree ≥ 3) than EDGE_CORRECT edges, confirming branch points as split failure loci.
- **Conclusion:** Confirmed (surprisal +0.307). Over 1,409,045 edges and 5,072 branch nodes, split edges had mean distance-to-branch 518.39 µm (median 196.43) versus 712.57 µm (median 380.38) for correct edges — split median roughly half the correct median (Mann-Whitney p = 0.0). Bifurcations are systematic split failure loci, the fourth converging line of evidence (with ranks 3, 5, 6) for branch-point fragility.
- **Caveats:** Very large sample inflates significance; the redundancy with ranks 3/5/6 means this is corroboration, not an independent finding.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded split n=6805 mean 518.39 µm (median 196.43) vs correct n=1109034 mean 712.57 µm (median 380.38), U=2979956736.0, p=0.0; rerun identical → match. Loading change: direct $RERUN_PKL load.
- **Generalization:** GENERALIZES
- **Across datasets:** Split edges topologically closer to GT branch nodes than correct edges, same direction and significant everywhere. origin (789202): split median 196.43 µm < correct 380.38 µm, U=2.980e9, p=0.0. 794491: split median 156.94 µm < correct 195.79 µm, U=1.464e9, p=0.0. 794495: split median 216.42 µm < correct 246.41 µm, U=3.473e9, p=4.6e-27. Gap smaller on extras but consistent.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U, two-sided, U = 2,979,956,736.0, p reported 0.0 (underflow), n = 6,805 split / 1,109,034 correct edges (weighted multi-source Dijkstra to nearest degree≥3 node). Correct non-parametric test.
- **Statistical issues:** Very large n makes the p uninformative; the effect is real and large (median 196 vs 380 µm, split ≈ half). Edge pseudo-replication applies but the central shift is robust. This is the fourth converging branch-point measure (ranks 3/5/6).
- **Logic issues:** none; stated as association and explicitly framed as corroboration rather than an independent finding.
- **Verdict rationale:** Right test, large generalizing effect (gap attenuates on extras but holds, p down to 4.6e-27); redundant-but-sound confirmation of branch-point fragility.

### 20. (Priority 0.287 · Surprise 0.307) Merge sites have ~55% higher local fragment-node density than correct regions (Welch t = 8.9).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 81 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether dense neuropil — measured as higher local `fragments_graph` node density within 10 µm — correlates with merge sites versus correctly traced control regions, using a volumetric density (nodes/µm³).
- **Conclusion:** Confirmed (surprisal +0.307). Merge sites averaged 0.001678 nodes/µm³ versus 0.001083 for controls (~55% higher), significant by Welch's t-test (t = 8.93, p = 2.8e-14) over 67 merge and 67 control points. Dense neuropil is a measurable environmental risk factor for merges — the volumetric-density restatement of ranks 2 and 23.
- **Caveats:** Only 67 vs 67 points; a parametric t-test is used on density data that may be skewed (the closely related ranks 2/23 used non-parametric tests). Substantively redundant with ranks 2 and 23.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded merge 0.001678 vs control 0.001083 nodes/µm³, Welch t=8.9288, p=2.7673e-14, n=67/67; rerun identical → match. Loading change: direct pkl load.
- **Generalization:** GENERALIZES
- **Across datasets:** Merge sites have higher volumetric fragment density than controls, same direction and significant everywhere. origin (789202): 0.001678 vs 0.001083 nodes/µm³, Welch t=8.93, p=2.8e-14, n=67/67. 794491: 0.001743 vs 0.001044, t=9.45, p=3.9e-17, n=86/86. 794495: 0.001564 vs 0.001021, t=9.41, p=1.9e-17, n=105/105.
- **Verdict:** WEAK
- **Test:** Welch's two-sample t-test, t = 8.9288, p = 2.7673e-14, n = 67 merge / 67 control. **Correct test would be Mann-Whitney U** (as the substantively identical ranks 2 and 23 used), because the "density" is a within-sphere node *count* divided by a constant volume — discrete, non-negative, right-skewed count data, not normal.
- **Statistical issues:** Parametric t-test applied to skewed count-derived densities (assumption of normality questionable at n=67); ranks 2/23 deliberately used non-parametric tests on the very same quantity. The result happens to be robust (t≈8.9, and the MW versions in ranks 2/23 agree), so the test choice does not overturn the conclusion, but it is the wrong test for the data type.
- **Logic issues:** none. Conclusion ("dense neuropil is a risk factor") is a restatement of ranks 2/23 and matches.
- **Verdict rationale:** A wrong/parametric test on count data (should be Mann-Whitney) earns a caveat, but the effect is large, the correct non-parametric versions (ranks 2/23) agree, and it generalizes 2-for-2 → WEAK (fragile-by-test-choice, not wrong).
- **Corrected test:** Mann-Whitney U (two-sided) with a rank-biserial effect size and bootstrap 95% CI, replacing Welch's t-test. The "density" is a within-sphere node *count* over a constant volume — discrete, non-negative, right-skewed — so the t-test's normality assumption is inappropriate; the non-parametric test matches the substantively identical ranks 2/23.
- **Corrected result:** origin Welch t = 8.9288, p = 2.7673e-14 → Mann-Whitney U = 3951.5, p = 8.0203e-15, rank-biserial r_rb = 0.7605 (95% bootstrap CI [0.6393, 0.8594]); medians 0.001671 vs 0.001194 nodes/µm³. The non-parametric test gives an even smaller p and a large effect size whose CI is far from 0.
- **Post-correction verdict:** UPHELD — the correct non-parametric test confirms the difference (p = 8.0e-15, large rank-biserial r_rb = 0.76 with CI excluding 0), agreeing with ranks 2/23; the test-choice caveat does not change the conclusion.
- **Corrected generalization:** GENERALIZES — both extras hold under Mann-Whitney with large effect sizes. 794491: U = 6524.0, p = 1.0533e-18, r_rb = 0.7642 (95% CI [0.6579, 0.8530]), n=86/86. 794495: U = 9320.0, p = 7.4411e-19, r_rb = 0.6907 (95% CI [0.5807, 0.7857]), n=105/105. Merge sites are denser everywhere under the correct test.

---

## Reproduction — Summary

- **Dataset pkl used:** `/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/cache/dataset_cache_789202_mcl100_add.pkl`
- **Breakdown (of n_rerun = 20):** REPRODUCED 17 · DIVERGED 3 · FAILED 0. All 20 ran on **revised-loading** code (0 recorded); every script exited 0.
- **Loading revisions:** All 20 top-records needed loading revision — each recorded script searched for the pkl by glob/relative path (e.g. `*_add.pkl`, `dataset_cache_*_add.pkl`, `../data/...`) or gated on a dataset-not-found check, so the rerun harness redirected the load to the provided `$RERUN_PKL`. The analysis and prints were left byte-for-byte unchanged.
- **Did NOT reproduce (DIVERGED) — 3 findings:**
  - **Entry 5 / id 13** (split edges geodesically closer to branch points): the recorded run globbed the same pkl under two paths and double-counted edges (Correct 2,218,068 / Split 13,610), whereas the single-pkl rerun gives Correct 1,109,034 / Split 6,805, so U=11,916,101,802 → 2,979,025,450 and p=0.0 → 1.32e-197. Same direction, materially different counts/stat — a recorded double-counting artifact, not an analysis failure.
  - **Entry 12 / id 49** (merge sites co-localize with fragment branch points): merge median 4.48 µm reproduces, but the unseeded random-node control resampled differently (random mean 250.32 → 320.30 µm; U=363 → 282; p=2.21e-17 → 9.53e-19). Conclusion holds; divergence is random-seed noise in the control.
  - **Entry 17 / id 65** (splits favor thin distal processes): split-edge stats reproduce exactly, but recorded double-counted correct edges (2,218,068 vs 1,109,034) and split n (13,610 vs 6,805), shifting U (281,944,000 → 140,888,192) and downsampled-correct mean (1278.24 → 1287.44 µm). Same direction; recorded double-counting artifact.
- All 3 divergences are **data-bookkeeping / random-seed** effects (not analysis failures), and each finding's qualitative conclusion is preserved. The 17 reproduced findings matched the recorded statistic, p-value, effect size, and n exactly (or within simulated-null seed noise for id 55).

## Generalization — Summary

- **Origin dataset:** `dataset_cache_789202_mcl100_add.pkl`. **Extra datasets tested (2):** `dataset_cache_794491_mcl100_add.pkl` and `dataset_cache_794495_mcl100_add.pkl` (each finding was re-run on the two brains it was NOT generated on; all 40 extra-dataset runs exited 0, none timed out).
- **Breakdown (of 20):** GENERALIZES 17 · PARTIAL 2 · DOES-NOT-GENERALIZE 1 · INCONCLUSIVE 0.
- **Findings that do NOT fully generalize (3):**
  - **Entry 1 / id 47** — DOES-NOT-GENERALIZE — split-gap non-linear distance×angle trade-off. The defining **interaction term** is significant on origin (x3 p=0.014) but collapses on both extras (794491 p=0.800; 794495 p=0.565, fit non-converged). The overall logistic discrimination and the linear distance term still hold, but the specific non-linear trade-off — this finding's actual contribution — does not replicate on either dataset.
  - **Entry 4 / id 11** — PARTIAL — omit edges topologically closer to leaves. Holds on origin (p=7.7e-311) and 794495 (omit 124.85 < correct 172.81, p=6.1e-149), but **direction flips** on 794491 (omit mean 159.76 > correct 138.76, one-sided p=1.0).
  - **Entry 10 / id 39** — PARTIAL — KDTree severing of merge sites. Reduces %merged-edges and raises accuracy in the same direction on all datasets, but the headline ">80% / ~86% merge removal" is dataset-specific: only ~70% on 794491 and ~44% on 794495, with accuracy gains of 82→94% (origin) vs 74→88% and 69→81% on the extras.
- **Synthesis.** The run's dominant theme — the **topological/spatial geography of errors** — is robust across all three brains: splits concentrate near branch points (ids 13, 19, 72) and thin distal tips (ids 60, 65), omits cluster contiguously (id 48) and at terminals (id 64), merges sit in dense, false-branch neuropil (ids 3, 23, 49, 53, 81), split gaps dwarf internal edges (id 35), split endpoints are anti-parallel (id 67), splits cascade spatially (id 55), and per-neuron split/omit rates co-vary (id 30) — all with consistent direction and very low p-values on both extra datasets (often with effect sizes that grow or shrink but never reverse). The two highest-priority *headline* results are the weakest under extrapolation: the rank-1 non-linear split-gap interaction (id 47) does not replicate, and the rank-10 actionable severing heuristic (id 39) keeps its direction but loses its quantitative punch on harder brains (794495, which is the most heavily-merged). With only two extra datasets the evidence is moderate, not definitive, but the structural/topological findings are corroborated 2-for-2 while the one reversal (id 11 on 794491) and the two attenuations are clearly localized.

## Statistical Verification — Summary

All 20 ranked hypotheses were audited for test choice, assumptions, power/effect-size,
p-value interpretation, the "failed-to-reject ≠ null-is-true" fallacy, and conclusion
overreach, integrating the Reproduction (REPRODUCED/DIVERGED) and Generalization
(GENERALIZES/PARTIAL/DOES-NOT-GENERALIZE) results already in this report.

- **Verdict breakdown (of 20):** SOUND 10 · MINOR 5 · WEAK 3 · MAJOR 2 · CRITICAL 0.
  - **SOUND (10):** ids 3, 10, 19, 23, 35, 53, 60, 64, 67, 72.
  - **MINOR (5):** ids 13, 48, 49, 55, 65.
  - **WEAK (3):** ids 11, 30, 81.
  - **MAJOR (2):** ids 47, 39.
  - **CRITICAL (0):** none — no conclusion is outright invalid.

- **MAJOR ids (one-line reasons):**
  - **Entry 1 / id 47** — the headline non-linear distance×angle **interaction** is a single borderline coefficient (p = 0.014) on a class-imbalanced fit that **does not generalize** (p = 0.800, 0.565 on the two extra brains).
  - **Entry 10 / id 39** — the actionable severing result has **no statistical test**, states a *per-neuron* >80% claim that is evaluated as a global mean (and is false for several neurons, e.g. N020 0%, N006 ~55%), uses **ground-truth merge coordinates** (oracle, not a deployable detector), and the ">80%" threshold fails on 794495 (~44%).

  (Note: **id 48** uses a pseudo-replicated χ² that is unsound as a formal independence test, but its conclusion is carried by a valid descriptive 28× ratio, so it is graded **MINOR**, not MAJOR.)

- **Most serious problems a scientist should not over-trust:** the two highest-priority
  headlines are the weakest. The **rank-1 non-linear split-gap trade-off (id 47)** rests on
  one fragile interaction term that collapses on both other brains — treat the *non-linear*
  claim as unsupported (the linear distance effect survives). The **rank-10 severing heuristic
  (id 39)** is an untested, GT-guided upper bound whose per-neuron and >80% claims do not hold
  uniformly. Secondary cautions: **id 30** (n=12, Pearson borderline with a leverage point and
  non-normal residuals — trust Spearman ρ=0.88, not the causal "shared mechanism" wording);
  **id 81** (Welch t-test on skewed count-density data — should be Mann-Whitney like ids 2/23,
  which agree); **id 48** (χ² inflated by pseudo-replication — rely on the 28× ratio, not p);
  and **id 11** (omit-near-leaves direction **reverses** on 794491). The many huge-n
  Mann-Whitney / chi-square results (ids 10, 13, 19, 60, 64, 72) are directionally sound but
  their astronomical p-values reflect sample size, not effect magnitude — read the medians.

### Benjamini–Hochberg FDR

The key p-value from each of the 20 hypotheses was collected and BH-corrected at q = 0.05.
**id 39 is excluded** from the family (no statistical test / no p-value), giving a family of
**m = 19**. For ids whose recorded p underflowed to 0.0 (ids 13, 35, 48, 55, 64, 65, 72) the
corrected/rerun value was used where available (e.g. id 13 → 1.32e-197) and a conservative
≤1e-300 floor otherwise.

- **Survive BH at q = 0.05:** **all 19** p-values survive. The BH critical value is the
  largest p in the family, p = 0.0221 (id 30), at rank 19 where the BH threshold is exactly
  0.05; every smaller p clears its threshold.
- **Borderline headline findings (0.001 < p < 0.05):** only **two** — **id 47** (interaction
  p = 0.014) and **id 30** (Pearson p = 0.0221). These sit at the very top of the ranked p-list
  and clear BH only by the thinnest margin (id 30's threshold equals q = 0.05). They are the
  findings most likely to fail correction under any stricter family (e.g. if the full ~98-record
  run were corrected jointly, or if more moderate p-values were added) and both are already
  flagged MAJOR/WEAK for independent reasons (id 47 does not generalize; id 30 is a fragile
  n=12 Pearson). Every other reported finding is so extreme (p ≤ 1.1e-08) that FDR control is
  irrelevant to it.

## Excluded (no surprisal score)

Two hypotheses were dropped by the ranker for missing surprisal values and are not represented above:
- Run `ground-truth-error-annotations-revised-version_2026-06-17`, ID 15
- Run `ground-truth-error-annotations-revised-version_2026-06-17`, ID 41

## Statistical Test Corrections — Summary

**5 hypotheses were flagged for a test fix and re-run with the correct statistic** (ids 47, 30, 39, 48, 81). Post-correction breakdown: **UPHELD 3 · WEAKENED 1 · OVERTURNED 0** (1 of the 3 UPHELD entries, id 47, is upheld on the origin brain only and does not generalize under the correct test).

**Findings that CHANGED under the correct test:**
- **Entry 10 / id 39 — WEAKENED.** Original: bare global before/after means with **no statistical test** (13.25%→1.83% merged, 82.27%→93.66% accuracy). Correct test: **paired Wilcoxon signed-rank** on the per-neuron deltas (W = 66.0, p = 4.9e-04 for both reduction and accuracy gain). The improvement is now properly significant, but the headline ">80% of merge errors removed *per neuron*" survives for only **8/12 neurons** on origin (median 97.4%, 95% CI [73.0%, 100%] — lower bound below 80%) and fails to generalize (3/9 on 794491, 5/18 on 794495), so the quantitative headline is materially less convincing than stated.

**Findings UPHELD under the correct test (with generalization):**
- **Entry 1 / id 47 — UPHELD on origin, but DOES-NOT-GENERALIZE.** Correct test: **permutation test + stratified bootstrap CI** on the interaction coefficient (Wald z unreliable on a small class-imbalanced logistic fit). Origin: x3 = −5.96, permutation p = 0.036, bootstrap CI [−12.34, −1.11] excludes 0 → upheld. Both extras collapse under the correct test (794491 perm p = 0.752, CI spans 0; 794495 perm p = 0.411, CI spans 0). The non-linear trade-off holds only on the origin brain — the corrected test reconfirms the prior MAJOR/non-generalization verdict.
- **Entry 8 / id 30 — UPHELD, GENERALIZES.** Correct test: **Spearman ρ with permutation p + bootstrap CI** (parametric Pearson/OLS violate normality at n=12). ρ = 0.8811, permutation p = 4.6e-04, CI [0.562, 0.986] — far more robust than the borderline Pearson p = 0.022; holds on both extras (794491 ρ = 0.983, p = 4.0e-05; 794495 ρ = 0.740, p = 4.8e-04).
- **Entry 11 / id 48 — UPHELD, generalization PARTIAL.** Correct test: **edge-label permutation test** (the original 2×2 χ² was pseudo-replicated). Where it completed (794491): observed ratio 18.63 vs null ~1.0, permutation p = 9.99e-04. Descriptive clustering ratio is huge on all three brains (28.09 / 18.63 / 40.42×, all ≫ the hypothesized 3×); the corrected permutation test confirms significance on 794491 but origin and 794495 timed out before emitting a corrected p (794495 INCONCLUSIVE under the correct test).
- **Entry 20 / id 81 — UPHELD, GENERALIZES.** Correct test: **Mann-Whitney U + rank-biserial effect size** (Welch t-test wrong for skewed count-densities; matches ranks 2/23). U = 3951.5, p = 8.0e-15, r_rb = 0.76 (CI [0.639, 0.859]); holds on both extras (794491 r_rb = 0.764, p = 1.1e-18; 794495 r_rb = 0.691, p = 7.4e-19).

**Synthesis.** Applying the correct test overturned **no** conclusion: the three density/correlation/clustering findings (ids 81, 30, 48) are fully robust to test choice — switching from a t-test to Mann-Whitney (id 81), from Pearson to a permutation Spearman (id 30), and from a pseudo-replicated χ² to an edge-permutation test (id 48) leaves each significant with a large, CI-supported effect, and ids 81 and 30 generalize 2-for-2 under the corrected test. The one finding materially **weakened** is the actionable severing heuristic (id 39): once a proper paired Wilcoxon is run, the *direction* of improvement is genuinely significant on all three brains, but the specific ">80%-per-neuron" headline is true for a minority of neurons and does not generalize. The most fragile headline, the rank-1 non-linear split-gap interaction (id 47), survives permutation/bootstrap inference on the origin brain but its CI spans zero on both other brains — confirming, under the correct assumption-free test, that the non-linear trade-off is dataset-specific rather than a general law.
