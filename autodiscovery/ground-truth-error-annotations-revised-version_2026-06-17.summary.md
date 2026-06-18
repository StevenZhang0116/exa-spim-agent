# AutoDiscovery Run Summary — ground-truth-error-annotations-revised-version_2026-06-17

## Header

- **Source file:** `autodiscovery/ground-truth-error-annotations-revised-version_2026-06-17.json` (100 hypotheses)
- **Ranking key:** `posterior-surprise` (priority = posterior * |surprisal|)
- **Hypotheses ranked:** 98 (2 dropped for missing surprisal: ids 15, 41)
- **Hypotheses shown:** top 20 of 98
- **Surprise magnitude range:** 0.0000 to 0.3513
- **Max priority score:** 0.3243

**Synthesis.** The highest-priority belief flips in this run all upgrade prior
"Leaning True" hunches to confident posterior beliefs about *where* and *why*
the U-Net reconstruction fails. The single most belief-shifting finding (H47,
priority 0.324) is mechanistic: split gaps obey a non-linear distance-angle
trade-off (logistic interaction term p = 0.014, coefficient −5.96), invalidating
fixed-threshold heuristics. A dominant cross-cutting theme follows: nearly
every other top-ranked result converges on the same picture — **merge errors
cluster where the neuropil is structurally crowded** (high local fragment-graph
density, high branch-density of the offending fragment) while **split and omit
errors cluster at thin distal/terminal branches and at topological
bifurcations**. The single most actionable finding (H39) demonstrates that
geometrically targeted graph severing at predicted merge coordinates removes
~86% of merge edges and *raises* edge accuracy from 82.3% to 93.7%, making
"geometric-targeted cut" a directly deployable proofreading primitive. Across
the 20 entries the consistent message is that segmentation failures are
non-random topological-spatial phenomena with strong, exploitable signatures.

## Ranked Entries

### 1. (Priority 0.324 · Surprise 0.351) Split-gap connections obey a non-linear distance-angle trade-off, invalidating fixed-threshold heuristics.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 47 · **Belief:** Leaning True → Likely True (0.6667→0.9231) · **Direction:** Positive
- **Tested:** Whether split gaps tolerate large turning angles at short distances but require collinearity at longer distances, modeled with logistic regression on 1,072 true-split pairs and 142 false-merge controls.
- **Conclusion:** The distance × angle interaction term is statistically significant (p = 0.014) with a strongly negative coefficient (−5.96), so the trade-off is genuinely non-linear: the fitted decision boundary tolerates >90° turns under ~8 µm but collapses toward 0° (strict collinearity) as gaps approach 15 µm. The positive surprisal (+0.351) reflects a confident upgrade from "Leaning True" to "Likely True" — the data confirms a richer geometry than a fixed-threshold pairing rule can capture.
- **Caveats:** Negative-control set ("False Merge") is only 142 examples versus 1,072 positives, an ~7.5:1 imbalance that may inflate boundary confidence near sparse control regions.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=1214, x3 coef=−5.9629 / p=0.0139, LLR p=1.626e-17, Pseudo R²=0.09278; rerun n=1214, x3 coef=−5.9629 / p=0.0139, LLR p=1.626e-17, Pseudo R²=0.09278 → exact match. Loading revision: direct `$RERUN_PKL` load + user-site numpy>=2 install + statsmodels upgrade; in-script pip installs no-op'd (host already has all deps).
- **Generalization:** DOES-NOT-GENERALIZE
- **Across datasets:** Origin 789202 n=1214, x3 (dist×angle interaction) coef=−5.963, p=0.0139 (significant, ratifying the non-linear trade-off). ds_794491 n=930, x3 coef=−0.342, p=0.800 — interaction term is NOT significant (and the dominant predictor flips to x1 / x2 alone). ds_794495 n=1012, x3 coef=−1.454, p=0.565 — also NOT significant. The overall logistic model is highly significant on both extras (LLR p=6.33e-38 and 3.61e-47) but the specific distance × angle interaction effect that defines the discovery is dataset-specific to 789202.
- **Verdict:** MAJOR (downgraded — rerun shows DOES-NOT-GENERALIZE)
- **Test:** statsmodels Logit on n=1214 (1,072 positives / 142 controls), reported interaction-term `x3` coef=−5.96, p=0.0139; overall LLR p=1.63e-17, Pseudo R²=0.093.
- **Statistical issues:** Class imbalance ~7.5:1 with only 142 negative controls inflates the standard error on the interaction term; p=0.014 sits just below 0.05 and could easily not survive a more conservative encoding of "False Merge". No effect-size CI reported for the interaction; statistical separation in the boundary plot is the only effect-size proxy.
- **Logic issues:** The claim ("split gaps obey a non-linear distance-angle trade-off") was generalized from a *single brain* to a general "rule" for split-gap geometry; rerun on two extra brains shows x3 p=0.800 and p=0.565 — i.e. on two brains there is no detectable non-linear interaction. The conclusion sentence "the trade-off is genuinely non-linear" overreaches the single-dataset evidence.
- **Verdict rationale:** Test choice (logistic regression with an interaction term) is appropriate; on origin the test is barely significant (p=0.014) and the headline mechanism does not replicate on either extra brain. The mechanistic generalization is not supported.

### 2. (Priority 0.287 · Surprise 0.307) Merge errors concentrate in regions of unusually high local fragment-graph density.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 3 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether local node density of `fragments_graph` within a 10 µm radius is higher at the 67 merge sites than at 67 matched correctly-reconstructed control sites.
- **Conclusion:** Mean local density at merge sites was 7.03 nodes vs. 4.39 at controls (Mann-Whitney U = 3931, p = 8.99e-15), a ~60% elevation that strongly supports the crowding-failure mechanism. The positive surprisal (+0.307) reflects firm posterior support that structural crowding is a measurable environmental risk factor.
- **Caveats:** Only 67 merge sites; control sampling is random and may not match other latent covariates (e.g., depth, contrast).
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded merge mean=7.03, control mean=4.39, U=3931.0, p=8.9996e-15, n=67/67; rerun merge mean=7.03, control mean=4.39, U=3931.0, p=8.9996e-15, n=67/67 → exact byte-for-byte match. Loading revision: bootstrap that pre-loads numpy>=2 and patches `os.walk` to yield the provided pkl (recorded code had a "Dataset not found" gate that exited without loading).
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: merge mean=7.03, control mean=4.39 (60% elevation), U=3931, p=9.00e-15, n=67/67. ds_794491: merge mean=7.30, control mean=4.42 (65% elevation), U=6390, p=2.33e-17, n=86/86 — same direction, same significance level. ds_794495: merge mean=6.55, control mean=4.63 (41% elevation), U=9031.5, p=1.07e-16, n=105/105 — same direction, significant. Crowding-at-merge effect is robust across all three brains.
- **Verdict:** OK
- **Test:** Mann-Whitney U on n=67/67, U=3931, p=8.9996e-15. Effect: 60% mean elevation (7.03 vs 4.39).
- **Statistical issues:** Sample size is modest (67/67) but the effect size is large (~60% mean elevation) and confirmed by independent reruns on two more brains with larger n. No correction for matching on latent covariates (depth, contrast).
- **Logic issues:** Conclusion ("structural crowding is a measurable environmental risk factor") is properly hedged as an association; no causal overreach. Belief shift (Leaning True → Likely True, +0.31 surprisal) matches the strength of evidence.
- **Verdict rationale:** Right test for two-sample skewed count data; result holds direction and significance on both extra brains; passes BH-FDR (q=0.05).

### 3. (Priority 0.287 · Surprise 0.307) Split errors are ~3.4x more frequent on edges adjacent to branch points than on linear edges.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 10 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Edge-level split rate on 15,298 branching edges versus 1,393,747 linear edges across pooled datasets.
- **Conclusion:** Branching edges have a 1.62% split rate (248/15,298) versus 0.47% for linear edges (6,557/1,393,747); the Chi-square test gives χ² = 414.5, p = 3.9e-92 — a ~3.4x relative increase that identifies topological junctions as a primary failure locus.
- **Caveats:** None noted; large sample on both classes gives ample power.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded branching=15298 (248 split, 1.62%), linear=1393747 (6557 split, 0.47%), χ²=414.4724, p=3.8959e-92; rerun branching=15298 (248 split, 1.62%), linear=1393747 (6557 split, 0.47%), χ²=414.4724, p=3.8959e-92 → exact match. Loading revision: numpy>=2 + `glob.glob` patch to return `$RERUN_PKL`.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: branching split rate 1.62% vs linear 0.47% (3.4× ratio), χ²=414.5, p=3.90e-92. ds_794491: branching 2.81% vs linear 1.36% (2.07× ratio), χ²=170.7, p=5.16e-39 — direction holds, significance overwhelming, ratio attenuated. ds_794495: branching 1.16% vs linear 0.58% (2.00× ratio), χ²=125.7, p=3.50e-29 — direction holds, significant. The branch-edges-are-more-split-prone effect is robust; the effect size varies (2-3.4×) but the direction never flips.
- **Verdict:** MINOR
- **Test:** Pearson χ² (chi2_contingency) on 2×2 of branching (n=15,298) vs linear (n=1,393,747); χ²=414.47, p=3.90e-92.
- **Statistical issues:** Edges sharing branch nodes are not strictly independent (each branch contributes ≥2 adjacent edges that share the same "branching" status). Independence is mildly violated, which inflates χ² and shrinks p, but with a 3.4× relative-rate gap and replication on two extra brains the qualitative conclusion is unaffected. The huge n (~1.4M edges) makes p far below any meaningful threshold; cite the rate ratio (3.4×) as the effect size.
- **Logic issues:** Conclusion appropriately scopes to "topological junctions as a primary failure locus"; no causal language.
- **Verdict rationale:** Right test for 2×2 categorical; minor dependence between adjacent edges; effect direction and magnitude robust across all three brains.

### 4. (Priority 0.287 · Surprise 0.307) Omit errors are topologically concentrated near terminal leaves of ground-truth neurons.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 11 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Multi-source BFS distances from each edge to the nearest leaf node, compared between 44,690 omit edges and 1,109,034 correct edges.
- **Conclusion:** Omit edges have mean topological distance-to-leaf of 247.18 (median 130.5) versus 325.08 (median 168.5) for correct edges; one-sided Mann-Whitney U = 2.22e10, p ≈ 7.75e-311. The model preferentially drops distal branches rather than mid-arbor structure.
- **Caveats:** None noted; effect direction and magnitude are consistent across mean and median.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded omit n=44690 (mean 247.1843, median 130.5), correct n=1109034 (mean 325.0803, median 168.5), U=22181022735.5, p=7.7483e-311; rerun omit n=44690 (mean 247.1843, median 130.5), correct n=1109034 (mean 325.0803, median 168.5), U=22181022735.5, p=7.7483e-311 → exact match. Loading revision: numpy>=2 + glob/walk patch.
- **Generalization:** PARTIAL
- **Across datasets:** Origin 789202: omit mean 247.18 (median 130.5) vs correct mean 325.08 (median 168.5), U=2.22e10, p=7.75e-311 — omits CLOSER to leaves (distal bias). ds_794491: omit mean 159.76 (median 84.5) vs correct mean 138.76 (median 70.5) — direction FLIPS, omits are FARTHER from leaves than correct edges; one-sided U=5.55e9, p=1.000 (not significant in the hypothesized direction). ds_794495: omit mean 124.85 (median 70.5) vs correct mean 172.81 (median 83.5), U=1.22e10, p=6.10e-149 — direction holds, significant. So the distal-leaf concentration of omits replicates on 794495 but is contradicted on 794491.
- **Verdict:** MAJOR (downgraded — direction flips on one extra brain)
- **Test:** One-sided Mann-Whitney U on 44,690 omit vs 1,109,034 correct edges; U=2.22e10, p=7.75e-311 on origin.
- **Statistical issues:** Edges on the same skeleton are not independent (BFS distances of adjacent edges are correlated), so the effective sample size is much smaller than the nominal n; the absurdly small p (~1e-311) reflects huge n more than effect-size magnitude. The mean shift (~247 vs 325) is modest relative to the variance, and on ds_794491 the *direction* of the shift reverses entirely.
- **Logic issues:** The conclusion ("the model preferentially drops distal branches") was generalized from one brain to a behavioural claim about "the model"; ds_794491's reversal shows this is brain-specific. Not all "p≈0" findings are universal — here the directional finding is fragile.
- **Verdict rationale:** Test is mostly appropriate (non-parametric for skewed distances), but the very-near-zero p-value misrepresents the robustness of the directional effect: it flips on one of two extra brains. Conclusion overreaches given that reproduction.

### 5. (Priority 0.287 · Surprise 0.307) Split edges sit significantly closer to branch points than correct edges in geodesic distance.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 13 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Dijkstra-based geodesic distance from each edge to its nearest GT branch point, compared between ~13,600 split edges and 2.2M correct edges.
- **Conclusion:** The Mann-Whitney U test (statistic 1.19e10) returns p = 0.0, with split-edge distributions sharply mass-shifted toward zero relative to correct edges. Bifurcations are confirmed structural failure points for continuity.
- **Caveats:** None noted; large sample on both sides.
- **Reproduction:** DIVERGED — conclusion holds, sample sizes differ (code: revised-loading)
- **Rerun result:** recorded `Correct n=2218068, Split n=13610, U=11916101802.0, p=0.0e+00`; rerun `Correct n=1109034, Split n=6805, U=2979025450.5, p=1.3239e-197`. Recorded loop processed TWO pkl paths (`./` and `../data/`) so it counted each edge twice (2.2M / 13.6k = exactly 2x); the rerun's monkeypatched glob resolves both paths to the SAME provided pkl, so the loader dedupes to the single dataset (1.1M / 6.8k). The direction, effect, and conclusion all hold — split edges are closer to branch points, p << 0.001 — but the recorded U statistic is inflated by the double-count, so the absolute U-value cannot be matched. Loading revision: numpy>=2 + glob patch.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202 (rerun): Correct n=1109034, Split n=6805, U=2.98e9, p=1.32e-197 — split edges closer to branch points. ds_794491: Correct n=420702, Split n=7847, U=1.46e9, p=5.16e-67 — direction holds, significant. ds_794495: Correct n=934849, Split n=7988, U=3.47e9, p=2.97e-27 — direction holds, significant. Geodesic proximity of splits to branch points is robust across all three brains.
- **Verdict:** MAJOR (downgraded — recorded n is double-counted)
- **Test:** Mann-Whitney U on geodesic distance-to-branch; recorded U=1.19e10 with `p=0.0` floor (recorded n=2,218,068 / 13,610 was double-counted). True (dedup) test: U=2.98e9, p=1.32e-197 (n=1,109,034 / 6,805).
- **Statistical issues:** Recorded code processed each edge twice via a duplicated glob hit, so the headline n is exactly 2× the true n and the reported U statistic is artefactually inflated; the recorded `p=0.0` is a floor report that hides the true magnitude. Edges within a skeleton are also non-independent, so even the deduped p-value overstates effective evidence. Conclusion direction is, however, preserved at p=1.3e-197 after dedup and on both extra brains.
- **Logic issues:** The recorded analysis treats "p=0.0" as conclusive; the dedup rerun shows the true p is ~1e-197, still extreme. No causal overreach in the conclusion sentence ("Bifurcations are confirmed structural failure points for continuity").
- **Verdict rationale:** Code bug (double counting) materially miscalibrates the headline numbers; the directional finding survives dedup and replicates on both extras, but the recorded statistic should not be cited as-is.

### 6. (Priority 0.287 · Surprise 0.307) Split errors are spatially correlated with branch nodes with mean distance ~106 µm shorter than correct edges.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 19 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Mean and median Euclidean distance from each split edge versus correct edge to the nearest GT branch node.
- **Conclusion:** Mean distance-to-branch is 275.76 µm (median 121.12) for split edges versus 381.98 µm (median 222.08) for correct edges (Mann-Whitney p ≈ 1.39e-229), and the density histogram shows a sharp concentration of splits within 50 µm of a branch. Branch topology is repeatedly implicated as a failure trigger.
- **Caveats:** Largely replicates the H13 geodesic result on Euclidean distance — informative but not statistically independent of H13.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded Split mean=275.76 µm (median 121.12), Correct mean=381.98 µm (median 222.08), U=2916506932.0, p=1.3860e-229; rerun Split mean=275.76 µm (median 121.12), Correct mean=381.98 µm (median 222.08), U=2916506932.0, p=1.3860e-229 → exact match. Loading revision: numpy>=2 + direct `$RERUN_PKL` open via the runner's open-patch.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: Split mean 275.76 µm (median 121.12), Correct mean 381.98 µm (median 222.08), U=2.92e9, p=1.39e-229 — splits closer to branch nodes. ds_794491: Split mean 215.34 µm (median 100.48), Correct mean 233.63 µm (median 125.12), U=1.45e9, p=1.68e-75 — direction holds, significant. ds_794495: Split mean 251.45 µm (median 130.75), Correct mean 299.19 µm (median 152.08), U=3.45e9, p=1.50e-32 — direction holds, significant. The Euclidean distance gap shrinks (106 → 18 → 48 µm) but the direction and significance hold on all three.
- **Verdict:** MINOR
- **Test:** Mann-Whitney U on Euclidean distance-to-branch; U=2.92e9, p=1.39e-229 on origin; significant in same direction on both extras (p=1.68e-75 and p=1.50e-32).
- **Statistical issues:** Sample is not statistically independent of H13 (same edge population, distance computed Euclidean vs geodesic), so H19 should be viewed as a robustness check rather than confirmatory new evidence. Edges within skeletons are non-independent. The "106 µm shorter" headline is brain-specific (origin only); the absolute gap shrinks to 18 µm on ds_794491.
- **Logic issues:** "Branch topology is repeatedly implicated as a failure trigger" overstates the causal direction (it is correlation, not a trigger); but the descriptive claim of spatial concentration is supported.
- **Verdict rationale:** Right test; effect direction is robust across brains; the headline µm gap is dataset-specific.

### 7. (Priority 0.287 · Surprise 0.307) Merge sites have a 75% higher 15 µm fragment density than non-merge controls.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 23 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Number of fragment nodes within a 15 µm radius around 67 merge sites versus 67 random non-merge control sites.
- **Conclusion:** Merge sites averaged 11.09 fragment nodes versus 6.33 at controls (Mann-Whitney U = 4100.5, p = 8.95e-17); the merge-site IQR sits entirely above the control IQR. Reinforces H3 at a coarser spatial scale.
- **Caveats:** Only 67 merge sites; closely overlapping with H3 (different radius) so should be treated as a robustness check rather than independent evidence.
- **Reproduction:** REPRODUCED on stats; script raised post-stats (code: revised-loading)
- **Rerun result:** recorded merge mean=11.09, control mean=6.33, U=4100.5, p=8.95e-17, n=67; rerun merge mean=11.09, control mean=6.33, U=4100.5, p=8.95e-17, n=67 → analysis numbers match exactly. Script then crashed in the plotting block on `boxplot(labels=...)` — `TypeError: boxplot() got an unexpected keyword argument 'labels'` (matplotlib 3.11 renamed it to `tick_labels`); this is post-statistic and does not affect the reproduction of the test result. Loading revision: numpy>=2 + direct `$RERUN_PKL`.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: merge mean 11.09 vs control 6.33 (75% elevation), U=4100.5, p=8.95e-17, n=67. ds_794491: merge mean 12.05 vs control 6.56 (84% elevation), U=6752.5, p=5.10e-21, n=86. ds_794495: merge mean 10.43 vs control 6.56 (59% elevation), U=9894.5, p=7.93e-24, n=105. All three brains show the same direction with strong significance; the 15 µm fragment-density-at-merge effect is robust. (All three runs raised the post-stats matplotlib `labels` keyword error; stats were emitted before the crash.)
- **Verdict:** MINOR
- **Test:** Mann-Whitney U on local fragment counts within 15 µm radius; U=4100.5, p=8.95e-17, n=67/67.
- **Statistical issues:** Modest n (67/67) but very large effect size (75% mean elevation). Not statistically independent of H3 (different radius on the same merge sites); should be counted as one effective finding with H3, H49, H81.
- **Logic issues:** Conclusion appropriately framed as confirmation of H3 ("Reinforces H3 at a coarser spatial scale"); no overreach.
- **Verdict rationale:** Right non-parametric test for count data; result replicates on both extras; redundant with H3 but consistent.

### 8. (Priority 0.287 · Surprise 0.307) Neuron-level split rate and omit rate are strongly positively correlated, implying a shared upstream failure mode.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 30 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Per-neuron Pearson and Spearman correlation between splits per mm and omit-edge fraction across 12 neurons exceeding the 50 µm length threshold.
- **Conclusion:** Pearson r = 0.65 (p = 0.022) and Spearman ρ = 0.881 (p < 0.001), with an OLS R² of 0.422 — neurons that fragment also drop edges, consistent with a shared cause such as weak local signal or poor contrast.
- **Caveats:** N = 12 neurons is very small; the Pearson p-value sits just under 0.05 and the OLS would not survive multiple-comparison correction across many candidate predictors. Treat the effect size as suggestive rather than definitive.
- **Reproduction:** REPRODUCED on stats; script raised post-stats (code: revised-loading)
- **Rerun result:** recorded n=12 neurons, Pearson r=0.6500 (p=2.2134e-02), Spearman ρ=0.8811 (p=1.5267e-04), OLS R²=0.422, splits_per_mm coef=2.2771 (p=0.022); rerun n=12 neurons, Pearson r=0.6500 (p=2.2134e-02), Spearman ρ=0.8811 (p=1.5267e-04), OLS R²=0.422, splits_per_mm coef=2.2771 (p=0.022) → exact match for all statistical outputs. Script crashed afterwards on `plt.cm.get_cmap` (removed in matplotlib 3.11). Loading revision: numpy>=2 + glob/walk patch.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: n=12 neurons, Pearson r=0.6500 (p=0.022), Spearman ρ=0.881 (p=1.53e-04), OLS R²=0.422, splits_per_mm coef=2.277 (p=0.022). ds_794491: n=9, Pearson r=0.989 (p=4.13e-07), Spearman ρ=0.983 (p=1.94e-06), OLS R²=0.979, splits_per_mm coef=1.451 (p=4.13e-07) — direction holds, effect MUCH stronger. ds_794495: n=19, Pearson r=0.805 (p=3.26e-05), Spearman ρ=0.740 (p=2.89e-04), OLS R²=0.648, splits_per_mm coef=1.483 (p=3.26e-05). The split-rate ↔ omit-rate per-neuron correlation strengthens (origin r=0.65 was borderline) and is robust. (All three runs raised the post-stats `plt.cm.get_cmap` error; stats were emitted before the crash.)
- **Verdict:** MINOR
- **Test:** Pearson r=0.650 (p=0.0221), Spearman ρ=0.881 (p=1.53e-04), OLS R²=0.422 on n=12 neurons.
- **Statistical issues:** Pearson is sensitive to non-linearity and to high-leverage outliers (the codeOutput explicitly notes the trend line is pulled by an outlier at Y≈12.6) and heteroscedasticity (codeOutput notes variance increases at high splits/mm). With n=12 the Pearson p (0.022) sits just below 0.05 and would not survive within-study multiple-comparison correction across many candidate predictor pairs. Spearman is more robust and is much stronger (p=1.5e-04), so reliance on rank-based statistics is preferred.
- **Logic issues:** "Implying a shared upstream failure mode" is a causal-mechanistic interpretation of a correlation between two per-neuron rates — both quantities can also be driven independently by neuron length, branch density, etc. Conclusion overreaches what an n=12 correlation can establish.
- **Verdict rationale:** Both tests are appropriate; the Spearman correlation is robust and replicates strongly on both extras, but the mechanistic interpretation ("shared upstream failure") is not isolated by this analysis.

### 9. (Priority 0.287 · Surprise 0.307) Inter-segment split gaps are ~4x larger than internal fragment edge lengths, so fixed-radius nearest-neighbor heuristics will fail.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 35 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Euclidean gap distance for 4,078 split-bridging gaps across 12 GT neurons compared to intra-segment edge lengths over 1.19M fragment nodes.
- **Conclusion:** Median split gap is 19.67 µm versus a 95th-percentile internal edge length of only 5.77 µm (Mann-Whitney p ≈ 0), and gaps extend to a 250 µm tail. Any global search radius wide enough to capture typical splits will sweep in many false positives, so distance-only heuristics are unworkable.
- **Caveats:** Only 12 neurons contributed gap measurements; gap-tail distribution may be neuron-specific.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded gaps n=4078 (median 19.67 µm), 95th-pctl internal edge=5.77 µm, U=3763981856.0, p=0.0e+00, 1190203 fragments-graph nodes processed; rerun gaps n=4078 (median 19.67 µm), 95th-pctl internal edge=5.77 µm, U=3763981856.0, p=0.0e+00, 1190203 nodes → exact match. Loading revision: numpy>=2 + glob patch.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: median split gap=19.67 µm vs 95th-pct internal edge=5.77 µm (3.4× ratio), U=3.76e9, p=0. ds_794491: median gap=29.95 µm vs internal 95th-pct=5.71 µm (5.2× ratio), U=1.58e9, p=0, n=2943 gaps. ds_794495: median gap=18.45 µm vs internal 95th-pct=5.71 µm (3.2× ratio), U=3.32e9, p=0, n=3250 gaps. The gap/internal-edge length disparity (and hence the unworkability of fixed-radius search) replicates on both extras.
- **Verdict:** OK
- **Test:** Mann-Whitney U on 4,078 split gaps vs 1.19M internal edges; U=3.76e9, p=0.0 (floor).
- **Statistical issues:** "p=0.0" is a floor report (true p is below double-precision representable); the qualitative gap (median 19.67 µm vs 95th-pctl 5.77 µm internal) is the meaningful effect size, not p. Internal edges within a fragment are not independent, but with a 3-4× scale gap independence violations cannot reverse the direction.
- **Logic issues:** Conclusion ("distance-only heuristics are unworkable") follows directly from the descriptive scale gap and replicates on both extras; no overreach.
- **Verdict rationale:** Right non-parametric test; effect-size dominates and is robust; the floor p-value is cosmetic but the underlying claim is well-supported.

### 10. (Priority 0.287 · Surprise 0.307) Geometric-targeted graph severing at merge coordinates removes ~86% of merge edges while raising edge accuracy from 82.3% to 93.7%.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 39 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Whether cutting the physical `fragments_graph` node nearest to each predicted merge coordinate (via KDTree) resolves >80% of merge errors per neuron without harming edge accuracy.
- **Conclusion:** Global mean percent-merged-edges dropped from 13.25% to 1.83% (~86% reduction, exceeding the 80% threshold) and mean edge accuracy *rose* from 82.27% to 93.66%; the most heavily merged neurons (N013, N018) went from ~40-43% merged and mid-50s accuracy to ~1% merged and ~96-98% accuracy. This is the most actionable result in the run — a directly deployable proofreading primitive.
- **Caveats:** Evaluated on a fixed neuron set; generalization to larger volumes and to predicted (not ground-truth) merge coordinates needs separate validation.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded Baseline mean %merged=13.25%, after-cut=1.83% (~86% reduction), baseline mean edge-accuracy=82.27% → after-cut=93.66%; rerun Baseline mean %merged=13.25%, after-cut=1.83%, baseline edge-accuracy=82.27% → after-cut=93.66% → exact match across all 12 neurons in the per-neuron table. Loading revision: numpy>=2 + walk patch + suppress the script's own `pip install` that was downgrading numpy.
- **Generalization:** PARTIAL
- **Across datasets:** Origin 789202 (12 neurons): %merged 13.25% → 1.83% (~86% reduction), edge-accuracy 82.27% → 93.66%. ds_794491 (9 neurons): %merged 20.54% → 6.15% (~70% reduction), accuracy 73.81% → 88.13% — direction holds, but the reduction does NOT clear the hypothesis's stated 80% threshold (70% achieved). ds_794495 (19 neurons): %merged 28.54% → 16.02% (~44% reduction), accuracy 68.68% → 81.17% — direction holds but the reduction is only 44%, far below 80%. Targeted severing always helps and always raises accuracy, but the "removes ≥80% of merges" claim only holds on origin (789202); on the more heavily merged brains the residual merge fraction stays large.
- **Verdict:** MAJOR (downgraded — quantitative claim does not transfer)
- **Test:** Descriptive paired comparison across 12 neurons (no formal hypothesis test reported in recorded code/output); per-neuron means: baseline 13.25% → after-cut 1.83% (~86% reduction).
- **Statistical issues:** No inferential test is reported (no t/Wilcoxon on paired per-neuron deltas, no confidence interval on the reduction proportion); the headline 86% figure is a single point estimate from 12 neurons with no uncertainty quantification. Effect-size estimation is fragile when n=12 includes two extreme neurons (N013, N018 at ~40-43% merged) that drive the global mean.
- **Logic issues:** The "≥80% reduction" headline is presented as a generalisable property of the intervention; rerun on ds_794495 achieves only 44% reduction, on ds_794491 70% — i.e. the quantitative claim is brain-specific. Additionally, the intervention is evaluated against ground-truth merge coordinates, not against *predicted* merge coordinates as would be needed in practice — this is acknowledged in the caveat but not in the headline.
- **Verdict rationale:** Direction is robust (always helps); the quantitative ≥80% claim only holds on origin and would not survive a confidence interval that included ds_794495's 44%. Should be reframed as "reduces merges by 44-86% depending on brain".

### 11. (Priority 0.287 · Surprise 0.307) Omit errors form contiguous runs along GT skeletons, with neighbor-conditional probability ~28x the marginal rate.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 48 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Markov-style transition probability that an edge is OMIT given its adjacent edge is also OMIT, versus the marginal OMIT rate.
- **Conclusion:** Marginal omit probability is ~3.17%, but the conditional probability conditioned on an OMIT neighbor jumps to ~89.08% — a 28.09x ratio, far above the hypothesized 3x threshold (chi-square ≈ 2.23M, p ≈ 0). Omits are not independent events; they manifest in concentrated streaks.
- **Caveats:** Adjacency is undirected on the GT skeleton, so reciprocal counting may slightly inflate the conditional rate; the qualitative effect dwarfs this concern.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded marginal P(OMIT)=0.0317, P(OMIT|neighbor=OMIT)=0.8908, ratio=28.09, χ²=2226077.8739, p=0.0e+00; rerun marginal P(OMIT)=0.0317, P(OMIT|neighbor=OMIT)=0.8908, ratio=28.09, χ²=2226077.8739, p=0.0e+00 → exact match. Loading revision: numpy>=2 + glob/walk patch + pip-install no-op (the recorded code had a retry loop that downgraded numpy).
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: marginal P(OMIT)=0.0317, conditional=0.8908, ratio=28.09, χ²=2.23M, p=0. ds_794491: marginal=0.0435, conditional=0.8098, ratio=18.63, χ²=7.28e5, p=0. ds_794495: marginal=0.0210, conditional=0.8490, ratio=40.42, χ²=1.96M, p=0. All three pass the 3× threshold by an order of magnitude; omits cluster in streaks on every brain.
- **Test:** χ² of independence on a 2×2 transition table (2,727,004 / 9,999 / 9,999 / 81,558); χ²=2.23M, p=0.0 (floor).
- **Verdict:** MINOR
- **Statistical issues:** Each ordered edge-pair contributes twice (undirected adjacency means (A,B) and (B,A) both counted), exactly doubling the off-diagonal and diagonal counts in symmetric fashion; the χ² statistic is roughly doubled but the conditional/marginal *ratio* is unchanged. The 28.09× ratio dwarfs the 3× threshold so direction is bullet-proof; the χ²=2.23M (essentially infinite by the floor p) is statistically meaningless as an effect-size statement.
- **Logic issues:** Conclusion ("omits manifest in concentrated streaks") follows from the ratio, not the p-value; appropriately scoped.
- **Verdict rationale:** Right test, expected-cell counts trivially satisfied, conclusion driven by effect size not p; replicates on both extras with ratios 18-40×.

### 12. (Priority 0.287 · Surprise 0.307) ~86% of merge sites sit within 10 µm of a fragments-graph branch point, and ~40% coincide with one exactly.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 49 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Distance from each of 67 merge sites and 67 random non-branch control nodes to the nearest fragment-graph branch point.
- **Conclusion:** Median distance is 4.48 µm at merge sites versus 179.76 µm at controls (Mann-Whitney p = 2.21e-17); the CDF shows ~40% of merge sites coincide with a branch and ~86% lie within 10 µm. False-branch detection is therefore a viable proofreader trigger.
- **Caveats:** Only 67 merge sites; control sampling restricts to non-branch nodes by construction, which may exaggerate the distance gap.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded Merge mean=27.47 µm (median 4.48 µm), Random mean=250.32 µm (median 179.76 µm), U=363.0, p=2.21e-17; rerun Merge mean=27.47 µm (median 4.48 µm) [identical], Random mean=247.16 µm (median 100.96 µm), U=381.0, p=4.38e-17 → merge-side stats reproduce exactly; control numbers shift because the script samples non-branch random control nodes without a fixed seed (per-run resampling). p-value direction, magnitude, and conclusion all hold (p ~1e-17, ~40% merge-at-branch claim is from the merge-side CDF which matched). Loading revision: numpy>=2 + glob patch.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: Merge mean 27.47 µm (median 4.48), Random mean 269.63 µm (median 149.51), U=399, p=8.60e-17, n=67. ds_794491: Merge mean 22.63 µm (median 3.86), Random mean 158.46 µm (median 131.16), U=605, p=1.04e-21, n=86. ds_794495: Merge mean 25.48 µm (median 5.11), Random mean 197.37 µm (median 136.90), U=705, p=4.04e-28, n=105. All three brains show merges sit ~3-5 µm (median) from a fragment branch, vs ~130-180 µm for controls — robust and strongly significant.
- **Verdict:** MINOR
- **Test:** Mann-Whitney U on n=67/67 distances to nearest fragment branch; U=363, p=2.21e-17.
- **Statistical issues:** Control set is *constructed* to be non-branch nodes — this guarantees a non-zero baseline distance to the nearest branch and biases the comparison in favour of the hypothesis. The 40%/86% headline figures are descriptive (single-dataset proportions, no CI). RNG seed is not fixed (rerun U shifted from 363 → 381). 
- **Logic issues:** Constructing the control to *exclude* the very feature being tested for at the alternative ("at a branch") is borderline circular and inflates the effect; the conclusion ("false-branch detection is a viable proofreader trigger") is plausible but the comparison would be cleaner against unconstrained control nodes.
- **Verdict rationale:** Right non-parametric test; control selection inflates effect size but the merge-at-branch effect is so large (median 4.48 vs >100 µm) that the conclusion survives; replicates on both extras.

### 13. (Priority 0.287 · Surprise 0.307) Merge-causing fragment segments have nearly double the branch density of correctly-mapped segments.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 53 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Branch-node density per 100 µm of cable on 64 merge-causing fragment segments versus 4,006 correctly-reconstructed control segments.
- **Conclusion:** Merge segments averaged 0.108 branches/100 µm versus 0.057 for controls (Mann-Whitney p = 2.91e-25); intrinsic branch density therefore works as a ground-truth-independent flag for likely-merging segments.
- **Caveats:** Only 64 merge segments; control set is much larger (~4,000), giving potentially imbalanced variance estimates but not undermining the median difference.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded merge n=64, control n=4006, merge density=0.108 (rerun: 0.1078), control density=0.0568 vs 0.0568 (rerun match), U=204163.5, p=2.9141e-25; rerun n=64/4006, U=204163.5, p=2.9141e-25 → exact match. Loading revision: numpy>=2 + glob patch + pip-install no-op.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: merge n=64 density=0.108 vs control n=4006 density=0.057 (1.9× ratio), U=204163.5, p=2.91e-25. ds_794491: merge n=98 density=0.166 vs control n=2729 density=0.055 (3.0× ratio), U=213596, p=4.32e-42. ds_794495: merge n=98 density=0.153 vs control n=3108 density=0.066 (2.3× ratio), U=235396, p=1.19e-27. The branch-density elevation is robust across all three brains; if anything the ratio increases on the extra datasets.
- **Verdict:** OK
- **Test:** Mann-Whitney U on branches/100µm; U=204163.5, p=2.91e-25, n=64 merge vs 4,006 control.
- **Statistical issues:** Imbalanced sample sizes (~63×) don't bias Mann-Whitney U for direction; effect-size ratio (1.9-3.0×) is reported and replicates. Branch count per segment is the only test variable, not a derived statistic, so independence is well-satisfied.
- **Logic issues:** Conclusion ("intrinsic branch density therefore works as a ground-truth-independent flag") is a defensible inductive claim — branch density is a fragment-intrinsic property; replicates on both extras with comparable effect.
- **Verdict rationale:** Right test, modest n on the merge side but large effect; passes BH-FDR and replicates.

### 14. (Priority 0.287 · Surprise 0.307) Split errors cascade — observed inter-split geodesic distances are ~9x shorter than a uniform-random null model.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 55 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** KS comparison of observed nearest-neighbor inter-split distances against a uniform null along the topological skeleton.
- **Conclusion:** Median observed inter-split distance is 25.58 µm versus 236.98 µm under the null (KS D = 0.454, p ≈ 0), confirming strong spatial clustering: once a split occurs, additional splits are highly likely nearby, consistent with locally bad signal or hard morphology.
- **Caveats:** Null model assumes uniform deposition along topology; alternative nulls accounting for cable-density gradients could partially attenuate the effect.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded KS D=0.4539, p=0.0e+00, obs median=25.58 µm, null median=236.98 µm (mean 372.21 µm); rerun KS D=0.4540, p=0.0e+00, obs median=25.58 µm, null median=237.34 µm (mean 371.59 µm) → D differs by 0.0001 (random-null resampling), all other numbers and the conclusion match. n=6805 splits and 680,500 null samples. Loading revision: numpy>=2 + glob patch + pip-install no-op.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: KS D=0.454, p=0, obs median 25.58 µm vs null median 237.30 µm (9.3× shorter), n=6805 splits. ds_794491: KS D=0.436, p=0, obs median 14.49 µm vs null median 87.37 µm (6.0× shorter), n=7847. ds_794495: KS D=0.563, p=0, obs median 14.37 µm vs null median 205.21 µm (14.3× shorter), n=7988. The split-clustering effect (observed inter-split distances much shorter than uniform null) replicates strongly on both extras.
- **Verdict:** MINOR
- **Test:** Kolmogorov-Smirnov two-sample on observed vs uniform-null inter-split distances; D=0.454, p=0.0 (floor), n=6805 observed and 680,500 null samples.
- **Statistical issues:** Null is uniform along the GT skeleton — caveated correctly. Non-uniform null (matched to cable-density gradients) could absorb some of the effect, since splits and skeleton density are likely correlated. Inter-split nearest-neighbor distances are not independent observations (nearest-neighbor structure on a tree is correlated), so the KS p is over-confident; the 9× effect-size gap is the more reliable evidence and replicates across both extras.
- **Logic issues:** Causal language "once a split occurs, additional splits are highly likely nearby" is a Markov-style interpretation that the analysis does not directly establish (the analysis shows spatial clustering, not a sequential causal mechanism). Conclusion slightly overreaches.
- **Verdict rationale:** Right test for a distribution comparison; p-value is floor-effect dominated; effect size is real and replicates; the "cascade" causal language is interpretive.

### 15. (Priority 0.287 · Surprise 0.307) Split edges sit closer to leaf nodes than correct edges, indicating distal-thin bias.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 60 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Network distance from each of 6,805 split edges and 1,109,034 correct edges to the nearest GT leaf node.
- **Conclusion:** Split edges average 938.6 µm to nearest leaf versus 1,282.7 µm for correct edges (Mann-Whitney p = 2.16e-134), with a sharp split-edge density peak under 500 µm. Splits disproportionately affect distal arbor; correct edges retain a thicker tail into the backbone.
- **Caveats:** None noted; very large samples on both sides.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded Split mean=938.60 µm, Correct mean=1282.69 µm, p=2.1613e-134, n=6805/1109034; rerun Split mean=938.60 µm, Correct mean=1282.69 µm, p=2.1613e-134, n=6805/1109034 → exact match. Loading revision: numpy>=2 + glob/walk patch + pip-install no-op.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: Split mean 938.60 µm vs Correct mean 1282.69 µm (Δ=344 µm), p=2.16e-134, n=6805/1109034. ds_794491: Split mean 518.26 µm vs Correct mean 546.75 µm (Δ=28 µm), p=1.17e-33, n=7847/420702 — direction holds, significant, but effect size is dramatically reduced. ds_794495: Split mean 600.59 µm vs Correct mean 699.28 µm (Δ=99 µm), p=1.45e-32, n=7988/934849 — direction holds, significant. The "splits closer to leaves" sign and significance generalize, though absolute distances and effect magnitude depend heavily on brain.
- **Verdict:** MINOR
- **Test:** Mann-Whitney U on network distance-to-nearest-leaf; n=6,805 split vs 1,109,034 correct; p=2.16e-134 on origin.
- **Statistical issues:** Edges within a skeleton are non-independent; with n>10⁶ on one side the p-value is dominated by sample size. ds_794491 shows only Δ=28 µm — directionally consistent but the headline "344 µm shorter" is brain-specific. Effect size should be reported as % shift (~27% shorter on origin, 5% on ds_794491).
- **Logic issues:** "Distal-thin bias" is a defensible inductive label for the distance-to-leaf shift; redundant with H17/H65 (different distance proxies); no causal overreach.
- **Verdict rationale:** Right test, direction robust on all three brains, but the headline magnitude is dataset-specific; should be cited with the range, not just origin's number.

### 16. (Priority 0.287 · Surprise 0.307) Omit error rate on terminal branches is ~1.8x the rate on internal segments.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 64 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Chi-square comparison of OMIT rate between 543,477 terminal edges and 865,568 internal edges.
- **Conclusion:** Terminal OMIT rate is 4.38% versus 2.41% for internal edges (χ² = 4189.9, p ≈ 0). Reinforces H11 at the categorical (terminal/internal) level: thin distal tips are dropped more often.
- **Caveats:** Largely redundant with H11; offers a cleaner binary partition but is not statistically independent.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded Terminal n=543477 (OMIT rate 4.38%, 23792/543477), Internal n=865568 (OMIT 2.41%, 20898/865568), χ²=4189.9364, p=0.0e+00; rerun Terminal n=543477 (rate 4.38%, 23792/543477), Internal n=865568 (rate 2.41%, 20898/865568), χ²=4189.9364, p=0.0e+00 → exact match. Loading revision: numpy>=2 + glob patch + pip-install no-op.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: Terminal OMIT rate 4.38% vs Internal 2.41% (1.82× ratio), χ²=4189.9, p=0. ds_794491: Terminal 4.94% vs Internal 3.82% (1.29× ratio), χ²=425.4, p=1.63e-94 — direction holds, ratio attenuated. ds_794495: Terminal 2.47% vs Internal 1.82% (1.36× ratio), χ²=691.98, p=1.66e-152 — direction holds, ratio attenuated. The terminal>internal OMIT rate effect is significant on all three; the ~1.8× ratio is brain-specific and the extra datasets show a smaller (1.3-1.4×) but still significant gap.
- **Verdict:** MINOR
- **Test:** χ² of independence on 2×2 of terminal (n=543,477) vs internal (n=865,568); χ²=4190, p=0.0 (floor).
- **Statistical issues:** Adjacent terminal edges share leaves and are not independent (mild violation, doesn't reverse direction). Not statistically independent of H11 — same OMIT signal partitioned differently. p=0 is a floor, the effect is a 1.82× rate ratio.
- **Logic issues:** Conclusion ("thin distal tips are dropped more often") is descriptively supported; no causal claim.
- **Verdict rationale:** Right test for binary outcome × binary partition; effect direction robust on all three brains.

### 17. (Priority 0.287 · Surprise 0.307) Split edges occur on thinner processes — median topological distance to leaf is 38% shorter.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 65 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Distance-to-nearest-leaf as a proxy for process thickness, comparing split edges to a downsampled 50,000-edge correct-edge sample.
- **Conclusion:** Median distance is 417.22 µm for split edges versus 670.25 µm for correct edges (Mann-Whitney p = 0.0); splits cluster on thinner terminal processes, consistent with H60.
- **Caveats:** Distance-to-leaf is an indirect proxy for radius/thickness; the correct-edge group was downsampled to 50k to avoid overflow, which is reasonable but constrains effect-size estimation precision.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded Split edges n=6805 (mean 940.69 µm, median 417.22 µm), Correct edges (downsampled to 50000, mean 1278.24 µm, median 670.25 µm), U=281944000.0, p=0.0e+00; rerun Split edges n=6805 (mean 940.69 µm, median 417.22 µm — identical), Correct edges 50000 (mean 1287.44 µm, median 671.61 µm — different sample), U=140888192.0, p=0.0e+00. Correct-edge group is downsampled WITHOUT a fixed seed in the recorded code, so the downsampled subset differs between runs; the headline median-shift and p-value-of-zero match, the U statistic differs as expected. Conclusion holds. Loading revision: numpy>=2 + glob patch + pip-install no-op.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: Split median 417.22 µm vs Correct median 671.61 µm (38% shorter), U=1.41e8, p=0, n=6805/50000. ds_794491: Split median 238.34 µm vs Correct median 280.89 µm (15% shorter), U=1.81e8, p=1.87e-29, n=7847/50000 — direction holds, significant. ds_794495: Split median 295.78 µm vs Correct median 341.05 µm (13% shorter), U=1.85e8, p=3.10e-26, n=7988/50000 — direction holds, significant. The split-closer-to-leaves effect generalizes (sign and significance) although the relative gap shrinks on the extras.
- **Verdict:** MINOR
- **Test:** Mann-Whitney U on distance-to-leaf, 6,805 split vs 50,000 downsampled correct edges; p=0.0 (floor).
- **Statistical issues:** Downsampling without a fixed RNG seed means the test result is not byte-for-byte reproducible; downsample size (50,000) is arbitrary and inflates the U statistic relative to no downsampling. Distance-to-leaf is a proxy for thickness, not a direct measurement. Highly redundant with H60 (same distance proxy, different downsampling).
- **Logic issues:** Conclusion ("splits cluster on thinner terminal processes") chains an unmeasured causal step (proxy → "thinner") that is plausible but not directly demonstrated.
- **Verdict rationale:** Right test, fragile to RNG seed, redundant with H60; direction robust across all three brains.

### 18. (Priority 0.287 · Surprise 0.307) Opposing endpoints across split gaps are geometrically anti-parallel, providing an exploitable directional signature.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 67 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Cosine similarity of inward-pointing endpoint direction vectors for 525 split endpoint pairs versus 2,689 spatially adjacent but topologically unconnected control pairs.
- **Conclusion:** Split pairs averaged cosine similarity −0.685 (vs. −0.511 for controls; Mann-Whitney p = 1.13e-08), with much tighter variance. Collinearity across a gap is a statistically distinguishing local signature that an agent can use to propose split-joins without ground truth.
- **Caveats:** Control pairs are still mildly anti-parallel on average (−0.51), so the discriminator is real but not crisp — practical use will likely need a composite score.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded split-pairs n=525 (mean cos sim=−0.6847 ± 0.3581), control n=2689 (mean=−0.5112 ± 0.5342), U=594814.0, p=1.13e-08; rerun split-pairs n=525 (mean=−0.6847 ± 0.3581), control n=2689 (mean=−0.5112 ± 0.5342), U=594814.0, p=1.13e-08 → exact match. Loading revision: numpy>=2 + direct `$RERUN_PKL` open.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: split-pair mean cosine=−0.685 vs control=−0.511 (gap 0.17), U=594814, p=1.13e-08, n=525/2689. ds_794491: split=−0.593 vs control=−0.333 (gap 0.26), U=361650, p=1.38e-10, n=276/3412. ds_794495: split=−0.791 vs control=−0.527 (gap 0.26), U=778749.5, p=8.55e-20, n=449/4686. All three brains show split-pair cosine similarity significantly more negative (anti-parallel) than spatially-adjacent controls; effect is robust and significance is comparable or stronger on extras.
- **Verdict:** OK
- **Test:** Mann-Whitney U on cosine similarity; U=594814, p=1.13e-08, n=525 split / 2,689 control pairs.
- **Statistical issues:** Cosine similarity is bounded [−1,1] and non-normal, so non-parametric Mann-Whitney is the correct choice. Modest effect-size gap (~0.17 in mean) but with variance reduction in the split group; the discriminator is real but not crisp.
- **Logic issues:** Conclusion ("collinearity is a statistically distinguishing local signature") is appropriately framed as a discriminator, not a sufficient classifier; replicates on both extras.
- **Verdict rationale:** Right test for bounded non-normal data; effect direction robust on all three brains; caveat properly hedges modest effect size.

### 19. (Priority 0.287 · Surprise 0.307) Split errors are topologically biased toward GT branch nodes, with median distance about half that of correct edges.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 72 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Topological distance from each edge to the nearest GT branch node across 1,409,045 edges and 5,072 branch nodes.
- **Conclusion:** Mean split distance was 518.39 µm (median 196.43) versus 712.57 µm (median 380.38) for correct edges (Mann-Whitney p = 0.0). Independent corroboration of H13/H19 at the full-graph scale.
- **Caveats:** Heavily overlapping with H13 and H19 — treat as a consistency check rather than fresh evidence.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded U=2979956736.0, p=0.0e+00, Split mean=518.39 µm (median 196.43), Correct mean=712.57 µm (median 380.38); rerun U=2979956736.0, p=0.0e+00, Split mean=518.39 µm (median 196.43), Correct mean=712.57 µm (median 380.38) → exact match. Loading revision: numpy>=2 + `pathlib.Path.rglob` patch.
- **Generalization:** PARTIAL
- **Across datasets:** Origin 789202: Split mean 518.39 µm (median 196.43) vs Correct mean 712.57 µm (median 380.38), gap 194 µm in mean, U=2.98e9, p=0. ds_794491: Split mean 386.40 µm (median 156.94) vs Correct mean 391.45 µm (median 195.79) — direction in MEAN is preserved but the gap collapses to ~5 µm; median gap still ~39 µm. U=1.46e9, p=0 (still significant due to sample size). ds_794495: Split mean 425.78 µm (median 216.42) vs Correct mean 508.92 µm (median 246.41) — direction holds, gap 83 µm in mean, p=4.56e-27 (much weaker than origin's p=0 but still significant). Direction holds on both extras and remains formally significant, but the effect-size on 794491 nearly vanishes in mean — the "splits at half the distance" framing in the conclusion is brain-specific.
- **Verdict:** MAJOR (downgraded — PARTIAL generalization, magnitude claim doesn't transfer)
- **Test:** Mann-Whitney U on topological distance-to-branch over 1,409,045 edges (5,072 branch nodes); U=2.98e9, p=0.0 (floor).
- **Statistical issues:** Sample is non-independent (within-skeleton edges); n>10⁶ makes any nonzero mean shift detectable, so p=0 reflects sample size more than effect magnitude. Highly redundant with H13 and H19 (same edge population, same target). On ds_794491 the mean gap collapses from 194 µm to 5 µm while p stays at 0 — a textbook example of huge n giving "significance" for a trivial effect.
- **Logic issues:** The headline "median distance about half" is a single-dataset feature; framing it as a general property of split errors overreaches given ds_794491's near-equal means.
- **Verdict rationale:** Right non-parametric test; the direction generalizes (formally) but the *effect magnitude claim* of the headline does not. The huge n masks how small the effect can be on other brains.

### 20. (Priority 0.287 · Surprise 0.307) Merge sites have ~55% higher local fragment-node volumetric density than correctly-traced regions.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 81 · **Belief:** Leaning True → Likely True (0.7083→0.9327) · **Direction:** Positive
- **Tested:** Local fragment-node density (nodes/µm³) within 10 µm of 67 merge sites versus 67 control sites along correctly-reconstructed edges, using Welch's t-test.
- **Conclusion:** Merge sites averaged 0.001678 nodes/µm³ versus 0.001083 for controls (t = 8.93, p = 2.77e-14), a ~55% elevation. Adds a volumetric-density formulation to the crowding-merge story (H3, H23).
- **Caveats:** Only 67 sites per group; tightly overlapping with H3 and H23 — independent confirmation but not orthogonal evidence.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded merge density=0.001678 nodes/µm³, control density=0.001083 nodes/µm³, Welch t=8.9288, p=2.7673e-14, n=67/67; rerun merge density=0.001678 nodes/µm³, control density=0.001083 nodes/µm³, Welch t=8.9288, p=2.7673e-14, n=67/67 → exact match. Loading revision: numpy>=2 + `subprocess.getoutput('find ...')` patch (the recorded code calls `find / -name "dataset_cache_*_add.pkl"`) + pip-install no-op.
- **Generalization:** GENERALIZES
- **Across datasets:** Origin 789202: merge density=0.001678 vs control=0.001083 nodes/µm³ (55% elevation), Welch t=8.93, p=2.77e-14, n=67. ds_794491: merge=0.001743 vs control=0.001044 (67% elevation), t=9.45, p=3.86e-17, n=86. ds_794495: merge=0.001564 vs control=0.001021 (53% elevation), t=9.41, p=1.92e-17, n=105. The volumetric fragment-node density elevation at merge sites holds with comparable effect size and stronger significance on both extras.
- **Verdict:** MINOR
- **Test:** Welch's two-sample t-test on volumetric density (nodes/µm³); t=8.93, p=2.77e-14, n=67/67.
- **Statistical issues:** Welch's t-test assumes approximately normal sampling distribution of the mean; with n=67 and what is plausibly a right-skewed density distribution, a non-parametric test (Mann-Whitney U as used in H3/H23/H49) would be more conservative — though the same data passes Mann-Whitney in H3 at p=9e-15, so the t-test result is not artefactual. Effect is highly redundant with H3, H23, H49 (same merge sites, different radius/volume formulation); should be counted as one finding.
- **Logic issues:** Conclusion appropriately frames as "adds a volumetric-density formulation"; no overreach.
- **Verdict rationale:** A non-parametric test would have been a marginally safer default than Welch's t at n=67 for skewed density data, but the conclusion is consistent with three independent variants of the same underlying signal and replicates on both extras.

## Reproduction — Summary

- **Dataset pkl used:** `cache/dataset_cache_789202_mcl100_add.pkl` (633 MB, single-brain ground-truth + fragments graph dump for brain 789202, mcl=100, with `_add` post-processing).
- **Rerun count:** 20 of 20 top-ranked hypotheses re-executed (rank-by `posterior-surprise`).
- **Verdict counts:** **REPRODUCED 19** (95%), **DIVERGED 1** (5%), **FAILED 0** (0%). All 19 reproductions exactly match the recorded headline statistics within rounding; the one DIVERGED case (H13) has materially different absolute numbers but the same direction and conclusion. No top-20 finding failed to run.
- **Code source:** all 20 ran on the **revised-loading** scripts (`autodiscovery/ground-truth-error-annotations-revised-version_2026-06-17.json.rerun/hypo_<id>.py`); none ran on the recorded code as-is. The recorded code uniformly failed pass-1 reruns because (a) every script's recorded `pickle.load` raised `ModuleNotFoundError: No module named 'numpy._core.numeric'` (the pkl was written with NumPy 2 and the host's user-site numpy was 1.26.4), and (b) most scripts gated execution behind a "Dataset not found" check using `os.walk("..")` / `glob.glob("../*.pkl")` / `pathlib.Path('..').rglob('*.pkl')` / `subprocess.getoutput('find ...')` that did not see the provided pkl. The revised loading bootstrap (a) ensures `numpy>=2,<2.3` is available before any unpickling, (b) monkeypatches `os.walk`, `pathlib.Path.rglob`, and `subprocess.getoutput` to yield the provided pkl path so the dataset-search branch succeeds, and (c) no-ops in-script `pip install ...` calls because the host already provides every dependency and those installs were downgrading numpy back to 1.26.4 and corrupting the user-site numpy directory mid-run. **No statistical logic was altered in any revised script.**

### Findings that did NOT reproduce exactly

- **Entry 5 / id 13** (DIVERGED): recorded `Total Correct edges analyzed: 2218068, Total Split edges analyzed: 13610, U=1.19e10, p=0.0` vs rerun `Correct=1109034, Split=6805, U=2.98e9, p=1.32e-197`. The recorded script's glob pattern returned two distinct paths in the original environment (`./` and `../data/`) — both pointing at the same dataset content — and the loop processed each file independently, so each edge was counted **twice**. The rerun's glob patch resolves both patterns to a single pkl, so it processes the dataset once. The direction (split edges closer to branch points), the effect size relationship, and the conclusion (p far below any reasonable significance threshold) are unchanged. Treat the recorded U-statistic as inflated by the original code's double-counting bug, not as evidence against the underlying finding.

### Entries with minor non-reproduction artefacts (still REPRODUCED)

- **Entry 7 / id 23**: stats reproduce exactly (mean 11.09 vs 6.33, U=4100.5, p=8.95e-17). Script then raised `TypeError: boxplot() got an unexpected keyword argument 'labels'` (matplotlib 3.11 renamed it to `tick_labels`). Post-statistic plot crash only.
- **Entry 8 / id 30**: stats reproduce exactly (Pearson r=0.6500/p=0.0221, Spearman ρ=0.8811/p=1.53e-04, OLS R²=0.422). Script then raised `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'` (removed in mpl 3.11). Post-statistic plot crash only.
- **Entry 12 / id 49**: merge-side statistics reproduce exactly; the random control-node sample differs slightly between runs (no fixed RNG seed in the recorded code) so the random-control mean/median and the U value shift modestly (U=363→381, p=2.21e-17→4.38e-17). Conclusion holds.
- **Entry 14 / id 55**: KS D differs by 0.0001 (0.4539→0.4540) due to random-null resampling without a fixed seed; observed medians, null medians, p-value, and conclusion all match.
- **Entry 17 / id 65**: split-edge stats reproduce exactly; the 50,000-correct-edge downsample is drawn without a fixed seed in the recorded code, so the correct-edge mean and U value shift modestly (mean 1278.24→1287.44 µm, U=2.82e8→1.41e8). p=0.0 in both runs; conclusion holds.

### Records requiring loading revisions and why

- **All 20 records (47, 3, 10, 11, 13, 19, 23, 30, 35, 39, 48, 49, 53, 55, 60, 64, 65, 67, 72, 81)** needed the NumPy-2 unpickling fix (the pkl was written with NumPy 2; the host's user-site numpy was 1.26.4 and lacked `numpy._core.numeric`).
- **17 of 20** (all except ids 19, 23, 67 which use hardcoded relative paths the runner's `open`-patch already redirected) additionally needed the dataset-search patch (`os.walk` / `glob.glob` / `pathlib.Path.rglob` / `subprocess.getoutput('find ...')`) so the "Dataset not found" gate did not exit before loading.
- **All 20** received the `pip install` no-op patch because the recorded scripts run their own pip-install bootstrap, and the `agentic-neuron-proofreader` package pins `numpy<2` which would downgrade the host numpy mid-run and break the in-flight unpickle. Suppressing those reinstalls (the host already provides every dependency) was essential to a clean run.

## Generalization — Summary

- **Extra datasets used:** `cache/dataset_cache_794491_mcl100_add.pkl` and `cache/dataset_cache_794495_mcl100_add.pkl` (two whole-brain pkls structurally identical to the 789202 origin). All 20 revised hypothesis scripts were re-executed against each extra pkl via the existing rerun bootstrap; no further loading edits were required (the matplotlib-keyword post-stats failures in H23 and H30 occur identically on all three datasets and do NOT affect the statistical results, which are emitted before the plot crash).
- **Verdict counts (n=20):** **GENERALIZES 16** (80%), **PARTIAL 3** (15%), **DOES-NOT-GENERALIZE 1** (5%), **INCONCLUSIVE 0** (0%).

### Findings that DO NOT fully generalize

- **Entry 1 / id 47** (DOES-NOT-GENERALIZE): The distance × angle interaction term `x3` is significant on origin (coef=−5.96, p=0.014) but completely loses significance on both extras (ds_794491 coef=−0.34, p=0.800; ds_794495 coef=−1.45, p=0.565). The overall logistic model remains highly significant on both extras (LLR p<10⁻³⁷), but the specific *non-linear* (interaction) component that defines the discovery is unique to 789202; on the other brains a simpler distance-only (and distance + angle) model captures the same data.
- **Entry 4 / id 11** (PARTIAL): Omit-edges' topological distance to nearest leaf — origin shows omits significantly closer than correct edges (mean 247 vs 325, p=7.75e-311); ds_794495 reproduces the direction (omits 125 vs correct 173 µm, p=6.10e-149). On ds_794491 the direction FLIPS: omit edges sit at mean 159.76 / median 84.5 vs correct 138.76 / 70.5 (one-sided p=1.000 for the hypothesized direction). The distal-leaf concentration of omits is brain-dependent.
- **Entry 10 / id 39** (PARTIAL): Geometric-targeted graph severing always improves accuracy (origin 82.27% → 93.66%; ds_794491 73.81% → 88.13%; ds_794495 68.68% → 81.17%) and reduces merges, but only on origin does it clear the headline ">80% merge-edge reduction" threshold (origin ~86%; ds_794491 ~70%; ds_794495 ~44%). The intervention helps everywhere; the headline magnitude is origin-specific.
- **Entry 19 / id 72** (PARTIAL): Splits closer (geodesically) to GT branch nodes than correct edges — direction preserved on all three brains and p remains < 1e-26, but the effect size collapses on ds_794491 (split mean 386.4 vs correct mean 391.5 µm, only 5 µm gap; though median gap stays at ~39 µm). The "splits at half the distance" framing is brain-specific even though the directional finding generalizes.

### Synthesis

The dominant cross-cutting theme of the run — **merges cluster in structurally dense / branchy regions; splits and omits cluster at thin distal/terminal branches and at bifurcations** — is robust: every merge-density finding (H3, H23, H49, H53, H81), every split-cascading or split-clustering finding (H10, H13, H19, H35, H48, H55, H65, H67), and the OMIT-streakiness / terminal-vs-internal asymmetry (H48, H64) replicate on both extra brains, often with comparable or stronger significance and only modest effect-size variation. The two findings that depend on a specific *quantitative* feature of 789202 — the **non-linear interaction structure of split-gap merging** (H47) and the **headline ">80% merge-edge reduction"** of geometric severing (H39) — are partially dataset-specific: the underlying mechanism still operates, but the parametric claim does not transfer. One finding is genuinely brain-dependent in direction: **omits' topological closeness to leaves (H11) flips sign on ds_794491**, suggesting the distal-bias of omits is itself a property of how each volume was reconstructed rather than a universal model failure mode. With only two extra brains the evidence is mid-confidence — the 16/20 generalization rate is encouraging but should not be over-interpreted as universal applicability.

## Excluded (no surprisal score)

- ground-truth-error-annotations-revised-version_2026-06-17 · id 15
- ground-truth-error-annotations-revised-version_2026-06-17 · id 41

## Statistical Verification — Summary

### Verdict counts (top-20 audited)

- **OK:** 4 — H3 (id 3, entry 2), H35 (id 35, entry 9), H53 (id 53, entry 13), H67 (id 67, entry 18).
- **MINOR:** 11 — H10 (id 10, entry 3), H19 (id 19, entry 6), H23 (id 23, entry 7), H30 (id 30, entry 8), H48 (id 48, entry 11), H49 (id 49, entry 12), H55 (id 55, entry 14), H60 (id 60, entry 15), H64 (id 64, entry 16), H65 (id 65, entry 17), H81 (id 81, entry 20).
- **MAJOR:** 5 — H47 (id 47, entry 1, downgraded by DOES-NOT-GENERALIZE); H11 (id 11, entry 4, downgraded by sign-flip on ds_794491); H13 (id 13, entry 5, downgraded by double-count code bug); H39 (id 39, entry 10, downgraded by PARTIAL generalization of headline ">80%" reduction); H72 (id 72, entry 19, downgraded by PARTIAL generalization of "half-distance" magnitude claim).
- **CRITICAL:** 0.
- **Total:** 20.

### Multiple comparisons — file-wide BH-FDR analysis

The full run contains 100 hypotheses; 98 carry a `surprisal` score (ids 15 and 41 are excluded). From `analysis` + `codeOutput` text we could extract a headline p-value for **69 / 98** records (the remainder are descriptive findings, "Pivot Required" exit cases, or did not emit a numeric p-value). Treating literal `p = 0.0` floor reports as p = 1e-300, **48 of those 69 are nominally significant at uncorrected α = 0.05**.

Applying Benjamini-Hochberg FDR control at q = 0.05 across the 69 extracted p-values: **all 48 nominally-significant findings pass BH** (the BH cutoff is p ≤ 0.0108; the largest passing p is 0.0108 vs the next-larger reported p > 0.34, so the gap is wide and FDR control is not the binding constraint). **All 20 top-ranked hypotheses pass BH-FDR**, including H47 (p = 1.63e-17 from the LLR, with the interaction term at p = 0.0139 still passing its rank-threshold), H30 (Pearson p = 0.022 still passing its larger rank-threshold), and every "p = 0.0" floor-report (H13, H35, H48, H55, H64, H65, H72). No borderline-significant headline finding fails FDR. The BH analysis is therefore *not* a hidden weakness of the run; the much more serious issues are reproduction/generalization failures (H11, H47) and code-bug-driven statistic inflation (H13).

### Most serious problems found (CRITICAL and MAJOR)

- **H11 / Entry 4 (MAJOR — direction flips):** The "omits are topologically closer to leaves" claim has p ≈ 7.75e-311 on origin but **flips sign on ds_794491** (omits are *farther* from leaves; one-sided p = 1.000 in the hypothesized direction). The headline framing of a general "distal-thin bias" of the model's omit errors is not supported across brains; the conclusion should be qualified as brain-specific or as reflecting differences in the GT-vs-prediction baseline distribution.
- **H13 / Entry 5 (MAJOR — code bug inflates statistic):** The recorded `Correct n=2,218,068, Split n=13,610, U=1.19e10, p=0.0` is exactly 2× the deduped true counts; the script processed two glob hits pointing at the same pkl and counted each edge twice. The deduped rerun gives U=2.98e9, p=1.32e-197 — still extreme, but the recorded U statistic is artefactually doubled. Cite the deduped numbers.
- **H47 / Entry 1 (MAJOR — DOES-NOT-GENERALIZE):** The headline mechanism (non-linear distance × angle interaction in split-gap acceptance) was significant on origin only at p = 0.014; on both extra brains the interaction term is not detectable (p = 0.800 and p = 0.565). The headline insight is brain-specific even though the overall logistic model is significant everywhere. The single most belief-shifting finding in the run does not transfer.
- **H39 / Entry 10 (MAJOR — quantitative claim is brain-specific):** Geometric severing *always* helps, but the headline ">80% merge-edge reduction" only holds on origin (~86%); ds_794491 achieves ~70%, ds_794495 only ~44%. The recorded analysis reports no inferential test (no Wilcoxon on paired per-neuron deltas, no CI on the reduction proportion). The actionable framing should be reduced to a range, "removes 44-86% of merge edges depending on brain".
- **H72 / Entry 19 (MAJOR — effect size collapses):** "Median distance about half" of correct edges on origin (194 µm mean gap) collapses to a 5 µm mean gap on ds_794491, while p stays at the 0.0 floor purely from sample size (n>10⁶). Textbook example of huge-n p-value masking a vanishing effect.

A pervasive cross-cutting issue is that many high-n tests (H11, H13, H35, H48, H55, H64, H65, H72) report `p = 0.0` floor values that say nothing meaningful beyond "very significant"; **effect-size statements (ratios, mean shifts, U-ratios) are the load-bearing numbers**, not the p-values. Several findings (H3, H23, H49, H53, H81) are tightly correlated variants of the same merge-density-at-crowding signal and should be counted as one effective finding, not five. The cross-edge non-independence of skeleton-derived measurements (H10, H11, H13, H19, H35, H55, H60, H64, H65, H72) is uniformly unaccounted for; this inflates significance but does not generally reverse direction, except where the rerun on additional brains demonstrates that it does (notably H11). The "discovery loop"'s most serious failure mode in this run is conclusion *overreach* — generalizing single-brain quantitative claims (H39, H47, H72) and single-test causal claims (H30, H55) beyond what the data isolates.
