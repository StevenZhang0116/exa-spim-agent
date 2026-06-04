# AutoDiscovery Ranked Conclusions — All Runs

## Sources and Ranking

- **Source files (3):**
  - `autodiscovery/exa-spim-run-1_2026-06-03.json` (n = 50 hypotheses)
  - `autodiscovery/exa-spim-run-2_2026-06-03.json` (n = 200 hypotheses)
  - `autodiscovery/refinement-of-exa-spim-run-1_2026-06-03.json` (n = 99 hypotheses)
- **Ranking key (`rank_by`):** `posterior-surprise` (priority_score = posterior * |surprisal|)
- **Hypotheses ranked:** 349; **report shows top 20** (entries beyond rank 20 omitted)
- **Surprise-magnitude range across ranked set:** 0.0000 to 0.6899
- **Max priority_score:** 0.5066
- **Excluded (no surprisal score):** 0

## Headline Synthesis

The single highest-priority finding (rank 1, posterior 0.73, |surprise| 0.69) flipped belief from *Leaning False* to *Leaning True*: a fragment's log-cable-length is a strongly significant logistic predictor of merge errors (p = 2.0e-5, OR ≈ 2.06). The second flip (rank 2, posterior 0.83) confirmed that radius-tapering gradients at fragment endpoints sharply distinguish artificial splits from biological terminations (t = 48.35). Rank 3 (posterior 0.91) reframes merge etiology: merges co-occur with *low* local fluorescence — the U-Net hallucinates bridges in dim, faint regions rather than at bright crossings (Welch t = -7.17, p = 1.6e-12). A cross-cutting theme dominates ranks 3–20: split and merge errors are spatially, geometrically, and image-feature coupled — they cluster near somas, at bifurcations, in tortuous segments, in dense "shatter zones," and inside heavy-tailed "hub" components — supporting a coupled, geometry-aware proofreading architecture (tangent alignment, angular cosine, soma priors, leaf-to-internal linking).

---

## Ranked Entries (Top 20 of 349)

### 1. (Priority 0.507 · Surprise 0.690) Merge-error probability rises logarithmically with fragment cable length, confirming longer predicted segments carry disproportionately higher merge risk.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 149 · **Belief:** Leaning False → Leaning True (0.2917 → 0.7344) · **Direction:** Positive
- **Tested:** Whether the probability of a U-Net fragment containing a merge error scales as a logarithmic function of its total cable length, reflecting greater spatial volume and crossing probability for longer predicted segments.
- **Conclusion:** Among 1,001 fragments interacting with ground truth, 20 were classified as merged (mapping to ≥ 2 GT neurons). Logistic regression of merge on ln(cable length) returned a coefficient of 0.7230 (OR = 2.06, p = 2.04e-5), meaning each unit increase in log-length more than doubles the odds of a merge; the longest bin (log-length 9.03–12.83) reached a 7% empirical merge rate vs. 0–3% in shorter bins. The positive surprise (+0.69) reflects that what was initially *Leaning False* is now confidently supported.
- **Caveats:** Only 20 positive (merged) cases out of 1,001 — the regression is informative but rests on a small positive class; effect estimate could be sensitive to outlier long fragments.
- **Verdict:** Minor issues
- **Test:** Univariate logistic regression of `is_merged` on `log(cable_length)` (statsmodels Logit), reported coefficient 0.7230 ± 0.1697, OR = 2.0607, p-value = 2.0444e-05 over n = 1,001 fragments with 20 positives.
- **Statistical issues:** Tiny positive class (20/1,001) leaves the Wald p-value vulnerable to leverage from a few long fragments; the "small bin" analysis ranges from 0% to 7% empirical merge probability, so the logistic fit is dominated by a single high-length bin. No multivariate control (e.g., cable density, branch count) was attempted. Passes Bonferroni (1.43e-04 at m=349) comfortably.
- **Logic issues:** Conclusion that "longer predicted segments carry disproportionately higher merge risk" is a fair inductive claim but conflates correlation with causation — long fragments may simply have more nodes that can map to a second GT component (definitional confound: longer cables intersect more GT volume).
- **Verdict rationale:** Test is appropriate and significant by a wide margin, but the small positive class and a definitional length→merge-opportunity confound mean the OR should be read as a screening signal, not a mechanistic law.

### 2. (Priority 0.370 · Surprise 0.446) Split-error endpoints taper sharply in radius while true biological terminations stay near-constant, making radius gradient a strong morphological discriminator.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 46 · **Belief:** Uncertain → Likely True (0.5417 → 0.8281) · **Direction:** Positive
- **Tested:** Whether fragment endpoints arising from split errors exhibit a significantly steeper radius decrease (sharper tapering) than true biological neurite terminations.
- **Conclusion:** Across 3,965 split endpoints and 3,416 true terminations, the mean radius gradient (over a 25 µm / 5-node window) was +0.01223 µm/µm for splits versus -0.00041 µm/µm for true terminations (t = 48.35, p ≈ 0). Gradients above ~0.015 µm/µm predominantly mark splits, providing a clean morphological flag for proofreading.
- **Caveats:** Some overlap remains near zero gradient; the threshold (~0.015) is empirical and may not generalize across imaging conditions outside this dataset.
- **Verdict:** Minor issues
- **Test:** Welch two-sample t-test on radius gradients (5-node, 25 µm linear-regression slope); reported t = 48.348, p ≈ 0 with n_split = 3,965 and n_true = 3,416.
- **Statistical issues:** Test is technically defensible because the means are so far apart, but with sample sizes in the thousands a t-test is overpowered — the headline "p ≈ 0" is driven as much by n as by effect; the more honest discriminator is the density-overlap region (Cohen's d ≈ 1.1 from reported means/stds gives strong separation, but the histogram shows substantial overlap near 0). Splits are not independent: a 25-µm window means neighboring leaves on the same fragment share nodes, mildly inflating effective n. Passes Bonferroni trivially.
- **Logic issues:** None observed. The radius-gradient is a true morphological feature and the recommendation (use as a flag, not as a hard classifier) matches the visible overlap.
- **Verdict rationale:** Test is correct, well-powered, and the conclusion is appropriately worded as "strong discriminator" rather than "perfect classifier". The Bonferroni-passing p plus large effect size makes the central claim robust.

### 3. (Priority 0.361 · Surprise 0.395) U-Net merge nodes sit in dim, low-signal voxels rather than bright intersections, indicating merges are hallucinated bridges across faint regions.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 84 · **Belief:** Leaning True → Likely True (0.6250 → 0.9135) · **Direction:** Positive
- **Tested:** Whether local raw fluorescence intensity at merge nodes is significantly lower than at matched background skeleton nodes, supporting a low-SNR hallucination mechanism rather than additive blooming.
- **Conclusion:** From 472 merge nodes and 472 matched background nodes (3×3×3 voxel patches), merge sites had mean max intensity 134.59 ± 158.30 vs. background 214.36 ± 182.46 (Welch t = -7.17, p = 1.57e-12). Merge nodes concentrate at intensities < 100 while background nodes have more probability mass at 150–500 — strongly supporting the low-SNR hallucination model and motivating a "thin + dim" flag for the proofreading attention mechanism.
- **Caveats:** A 3×3×3 voxel patch is small; the max-intensity statistic could miss broader local context, and background matching is random rather than morphology-matched.
- **Verdict:** Minor issues
- **Test:** Welch's two-sample t-test on max-intensity in a 3×3×3 voxel patch, t = -7.1670, p = 1.5679e-12, n = 472 merge nodes vs 472 randomly-sampled background nodes (codeOutput lines 33–35).
- **Statistical issues:** Distributions are extremely right-skewed (means 134.59 ± 158.30 vs 214.36 ± 182.46; std > mean), violating t-test normality assumption — a Mann–Whitney U or a log-transformed test would be more defensible. With these std/mean ratios, Welch's t still works at this n because of CLT, but the headline statistic should ideally be on log-intensity. Background matching is random, not morphology- or fragment-matched, so any morphology-intensity confound is uncontrolled. Survives Bonferroni.
- **Logic issues:** Conclusion "U-Net hallucinates bridges across dim, faint regions" is a causal/mechanistic claim that the test does not directly support — it shows correlation between merge-node location and low intensity, not that low intensity *causes* the hallucination. Merges-on-thin-axons (already known) could also produce lower max-intensity by construction.
- **Verdict rationale:** Effect direction is robust and survives any reasonable reformulation, but the mechanistic phrasing overreaches a simple two-sample comparison and the parametric test is suboptimal on right-skewed intensities.

### 4. (Priority 0.314 · Surprise 0.365) Merge errors concentrate near somas, with proximal merge density roughly 29x higher than distal regions.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 172 · **Belief:** Leaning True → Likely True (0.6250 → 0.8594) · **Direction:** Positive
- **Tested:** Whether merge-error density (false-positive branch points per 1,000 µm) is significantly higher within a 150 µm radius of soma centroids than in distal neurite regions, implicating localized high neurite density.
- **Conclusion:** Of 36 total merge errors, 2 fell in the proximal region (~38,524 µm cable, density 0.0519/1000 µm) versus 34 distally (~18.77 M µm cable, density 0.0018/1000 µm). A chi-squared test returned χ² = 50.43, p = 1.23e-12, confirming significant proximal enrichment and supporting a soma-aware proofreading prior.
- **Caveats:** Only 2 proximal merges drive the proximal density estimate, so the absolute ratio is unstable; soma centroids were derived from max-radius heuristic rather than explicit centroid annotations.
- **Verdict:** Major issues
- **Test:** Chi-squared goodness-of-fit on proximal/distal merge counts (df=1), χ² = 50.4320, p = 1.23e-12; counts = 2 proximal vs 34 distal merges; cable lengths 38,524 µm vs 1.87 × 10^7 µm.
- **Statistical issues:** Chi-squared is **invalid** here — the expected proximal count is 36 × (38,524 / 1.8811e7) ≈ 0.074 (well below the conventional minimum of 5), and the observed proximal count is only 2; Fisher's exact / Poisson rate-ratio test is required, not Pearson χ². Even worse, the reported χ² ≈ 50.4 collapses to ((2 − 0.074)² / 0.074) + ((34 − 35.93)² / 35.93) ≈ 50.2, so the entire reported significance is driven by a single division by a vanishingly small expected count. The Wald-style χ² approximation does not hold and the actual p-value should be much larger (Poisson-rate test on 2 events vs an expected ≈0.07 events would give roughly p ≈ 0.003, not 1e-12).
- **Logic issues:** Two merges is too sparse to characterize a "29× higher density" — that ratio is a noise-dominated point estimate with a CI that easily spans 1. The "soma centroid = max-radius node" heuristic is an additional unvalidated assumption. Conclusion that merges "concentrate near somas" is plausible but not established by this experiment.
- **Verdict rationale:** The χ² approximation is misused on a cell with E ≈ 0.07, producing a spuriously tiny p; the substantive claim rests on n = 2 proximal merges. Despite the headline p of 1.23e-12, this finding is statistically fragile.

### 5. (Priority 0.311 · Surprise 0.483) Tangent alignment alone discriminates true vs. false continuations at U-Net merge hubs, validating angular routing for merge resolution.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 88 · **Belief:** Leaning False → Leaning True (0.2917 → 0.6442) · **Direction:** Positive
- **Tested:** Whether geometric tangent alignment at a U-Net merge hub is sufficient to identify the anatomically correct continuation, with true continuations having significantly smaller angular deviation from 180° than fused-branch competitors.
- **Conclusion:** Across 40 true and 80 false continuations at merge hubs, true continuations averaged 42.55° deviation from a straight line vs. 80.78° for false ones (Welch p = 1.15e-8; ROC AUC = 0.79). Tangent alignment is a strong discriminator, motivating angular-deviation penalties when routing skeletons through merges. The relatively large positive surprise (+0.48) reflects a flip from *Leaning False* prior belief.
- **Caveats:** Sample is small (40 / 80 pairs) and class-imbalanced toward false continuations — AUC of 0.79 is good but not decisive on its own.
- **Verdict:** Minor issues
- **Test:** Welch's t-test on angular deviations from 180° (t-stat unstated, p = 1.1485e-08), plus ROC-AUC = 0.7925, on n = 40 true vs 80 false continuations enumerated as branch-pairs at merge hubs.
- **Statistical issues:** Critical **independence violation** — the 120 pairs are *enumerated from a smaller set of hubs* (each merge hub contributes O(k²) pairs from its k branches), so observations cluster by hub; Welch's t-test and ROC both assume independent samples. The reported p-value is anti-conservative. The unbalanced 40/80 split also biases ROC at low-FPR. Survives Bonferroni at face value, but a hub-clustered permutation test would be the right approach.
- **Logic issues:** "Resolved purely by tangent alignment" overstates an AUC of 0.79 — that means 21% of pairs are mis-ranked. The conclusion sentence "tangent alignment is a strong discriminator" is more defensible than "resolved purely".
- **Verdict rationale:** Effect direction is clear and consistent with geometry, but the pair-level independence assumption is violated and the small sample at the hub level limits how confidently AUC = 0.79 can be claimed.

### 6. (Priority 0.287 · Surprise 0.307) ExaSPIM voxel anisotropy biases recoverable split errors toward the X/Y plane (85–90° from Z), recommending an ellipsoidal search.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 1 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether the dataset's voxel anisotropy (1.0 µm Z vs. 0.748 µm X/Y) imposes a directional bias on the orientations of recoverable topological split errors relative to the Z-axis.
- **Conclusion:** Empirical angular bins show maximum positive deviation (+0.040) over uniform-sphere expectation at 85–90° from Z (the X/Y plane), with a Kolmogorov–Smirnov test (KS = 0.10, p = 5.07e-4) rejecting uniformity. This justifies expanded X/Y-plane search and geometric penalties for X/Y-aligned gaps in split-correction algorithms.
- **Caveats:** A KS effect size of 0.10 is modest; the recommendation rests more on the directional pattern than on a large deviation magnitude.
- **Verdict:** Minor issues
- **Test:** One-sample Kolmogorov–Smirnov test against the uniform spherical CDF F(θ) = 1 − cos(θ), reported KS = 0.1000, p = 5.067e-04, on n = 410 split-pair angular vectors.
- **Statistical issues:** **KS p-value is misleading**: pairs reused across `query_pairs(r=20.0)` are not independent — a leaf may be the endpoint of multiple pairs, and pairs on the same fragment share underlying geometry; the KS test treats them as i.i.d. observations of θ. With m = 349 tests, p = 5.07e-4 *passes* Benjamini–Hochberg at FDR 0.05 (BH threshold ≈ 4e-3 from the run-wide table) but **fails Bonferroni** at α/m ≈ 1.43e-4. KS-statistic 0.10 is modest; the recommendation depends on the *direction* of the bin with maximum positive deviation, not on test magnitude.
- **Logic issues:** The "max positive deviation" bin (85–90°) is identified after the fact (data-driven bin selection) but not corrected for multiple bin testing; this is a mild but uncorrected selection bias.
- **Verdict rationale:** Direction of the X/Y-plane bias is physically plausible (voxel anisotropy) and reproducible across bins, but a small KS statistic with a borderline (Bonferroni-failing) corrected p makes the recommendation supportive rather than decisive.

### 7. (Priority 0.287 · Surprise 0.307) Artificial split endpoints cluster spatially nearer to merge sites than true terminals, justifying a coupled split-merge proofreading loop.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 7 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether U-Net split endpoints co-occur spatially with merge sites more often than true biological terminations, supporting a coupled architecture in which merge resolution unblocks nearby split reconnections.
- **Conclusion:** With 481 merge nodes, 4,091 artificial split leaves and 3,443 true terminals, mean distance from split leaf to nearest merge was 2,968 µm vs. 3,340 µm for true terminals (KS = 0.054, p = 3.5e-5). Splits are statistically closer to merges, supporting a closed-loop proofreading design where merge fixes spawn split-reconnection candidates.
- **Caveats:** Absolute distances are large (kilometers of µm) because GT is sparse (~18 traced neurons vs. ~10,000 fragments); the *relative* split-vs-terminal contrast is the load-bearing finding, not the absolute value, and KS = 0.054 is a small effect size.
- **Verdict:** Minor issues
- **Test:** Two-sample Kolmogorov–Smirnov on distance-to-nearest-merge distributions; KS = 0.0540, p = 3.5151e-05; n_split = 4,091, n_terminal = 3,443.
- **Statistical issues:** Effect size KS = 0.054 is **very small** — the only reason p ≈ 3.5e-5 is the large n. Means differ by 2,968 vs 3,340 µm (~12% relative) at distances dominated by GT sparsity (~18 traced neurons). The two distributions overlap heavily; this is a population-level shift, not a usable per-leaf signal. Passes Bonferroni. Independence assumption is mildly violated (nearby splits share merge neighbors).
- **Logic issues:** "Justifying a coupled split-merge proofreading loop" is a substantive inductive claim built on a 0.054 KS statistic. The phrasing "splits are statistically closer to merges" is true distributionally but is a weak basis for an architectural recommendation.
- **Verdict rationale:** Direction is reliable and the test is the right one for a two-sample CDF comparison, but the tiny effect size cautions against the strong architectural claim it is being used to justify.

### 8. (Priority 0.287 · Surprise 0.307) Merge errors follow a heavy-tailed distribution — the top 5% of components carry ~31% of merges, justifying targeted "hub-first" proofreading.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 15 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether merge errors are heavy-tailed across fragment components rather than uniformly distributed, with a small subset of "hub" fragments accounting for a disproportionate share of merges.
- **Conclusion:** 294 merge edges were localized in only 21 components, with a Gini coefficient of 0.7234 and the top 5% (one component) accounting for 31.29% of all merges. A log-log rank-frequency plot confirms heavy-tailed structure, indicating that prioritizing a few highly-fused components yields outsized correction throughput.
- **Caveats:** Only 21 components host merges in this dataset, so the "top 5%" reduces to a single component — generalization of the Gini estimate to other reconstructions is limited.
- **Verdict:** Major issues
- **Test:** Descriptive — Gini coefficient = 0.7234 computed on per-component merge counts (n = 21 components hosting merges, 294 merge edges), plus a visual log-log rank–frequency plot. **No null-hypothesis test was performed.**
- **Statistical issues:** With n = 21 components, "top 5%" is 1 component (the report itself acknowledges this); the Gini = 0.72 is a point estimate with no CI, no permutation test against a null of uniform distribution, no comparison against the natural distribution of fragment sizes. The log-log "heavy-tail" claim is made by eyeballing 21 points — far too few for any reliable power-law fit.
- **Logic issues:** Heavy-tailed phenomena routinely arise from any monotonic skew without being true power laws. Concluding "heavy-tailed" from 21 points and recommending "hub-first proofreading" overreaches the underlying data. The single top component's 31.29% share could be anomalous rather than systemic.
- **Verdict rationale:** No statistical test exists for the central distributional claim; n = 21 is too small for a Gini or power-law assertion; the recommendation is plausible but not evidence-backed.

### 9. (Priority 0.287 · Surprise 0.307) Split errors are significantly more likely in locally tortuous (curved) neurite segments than in straight ones.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 28 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether local neurite tortuosity (over a 15 µm geodesic window) predicts U-Net split fragmentation, since sharp turns may exceed the network's spatial continuity prior.
- **Conclusion:** From samples of 10,000 degree-2 nodes per group (245,053 split sites vs. 1,103,835 continuous), mean tortuosity at split sites was 1.0499 vs. 1.0416 at continuous sites (t = 8.86, p = 8.8e-19; KS = 0.053, p = 1.1e-12). The split distribution has a heavier right tail, confirming tortuosity as a structural risk factor for fragmentation.
- **Caveats:** The mean tortuosity difference is small (~0.008) — although statistically highly significant due to sample size, practical discrimination at a single node is weak; this is a distributional, not classification, effect.
- **Verdict:** Minor issues
- **Test:** Welch's t-test (t = 8.8594, p = 8.7931e-19) and Kolmogorov–Smirnov (KS = 0.0531, p = 1.1232e-12) on tortuosity at 10,000 sampled split vs 10,000 continuous degree-2 nodes (drawn from 245,053 vs 1,103,835 nodes).
- **Statistical issues:** Classic "huge n, trivial effect" — mean tortuosity differs by 1.0499 − 1.0416 ≈ 0.008 (Cohen's d ≈ 0.13 from typical neurite-tortuosity SDs); the test is detecting a tiny distributional shift, not a meaningful discriminator. Independence is also violated — degree-2 nodes are sampled from highly connected paths where neighboring nodes share most of the 15-µm window. Survives any FDR correction by orders of magnitude.
- **Logic issues:** Conclusion "split errors are significantly more likely in tortuous segments" is true distributionally but the report's own caveat notes per-node prediction is weak. Saying tortuosity is a "structural risk factor" is fine; treating it as a classifier feature without combining with others would underperform.
- **Verdict rationale:** The KS / t-test are appropriate, the result reproducible, but the effect size is tiny so the practical claim must remain distributional.

### 10. (Priority 0.287 · Surprise 0.307) U-Net fragment leaf nodes deviate further from the true GT centerline than internal nodes, evidence of pre-split geometric drift.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 30 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether automated split errors are immediately preceded by geometrical drift, measured as significantly larger distance to the nearest GT skeleton at fragment leaves than at internal fragment nodes.
- **Conclusion:** Across 39,826 leaf nodes and an equal random sample of internal nodes, mean leaf drift was 1605.61 µm (median 1159.68 µm) vs. internal mean 1470.53 µm (median 1001.25 µm); Mann–Whitney p = 8.27e-66 supports a real shift. Despite the large absolute values (driven by GT sparsity), the relative leaf-vs-internal difference is robust and consistent with local prediction uncertainty preceding splits.
- **Caveats:** Absolute drift distances are inflated because most fragments correctly reconstruct untraced GT neurons; the analysis is interpretable only as a relative comparison, not as an absolute uncertainty measure.
- **Verdict:** Major issues
- **Test:** Mann–Whitney U on distance-to-nearest-GT-skeleton; U = 848,650,330, p = 8.265e-66, n = 39,826 leaves vs 39,826 sampled degree-2 internal nodes.
- **Statistical issues:** With n ≈ 80,000 total the test is grotesquely overpowered. More importantly, the **comparison is confounded**: GT is sparse (~18 traced neurons vs ~10,000 fragments) so most fragments map to a *different* untraced GT neuron, inflating both groups' "drift" into the kilometer range. Mean drift of 1,470–1,605 µm has no biological meaning. Independence is violated — leaf and internal nodes drawn from the same fragment share the same GT-distance baseline.
- **Logic issues:** "Pre-split geometric drift" is presented as a mechanism (leaves drift *before* splitting). The test is purely cross-sectional (leaf vs internal at the same instant) and cannot distinguish a pre-split mechanism from "leaves are simply farther from the median GT node, by definition of being graph endpoints near boundaries." The mechanistic conclusion is **not supported** by the experiment.
- **Verdict rationale:** Test is well-conducted given the design, but the design itself can't separate a mechanistic "drift causes split" claim from the trivial geometric fact that leaves are at the edge of fragments. The 1,600 µm absolute distance further undermines interpretation.

### 11. (Priority 0.287 · Surprise 0.307) Endpoint cosine similarity outperforms Euclidean distance for matching split-gap continuations (AUC 0.90 vs. 0.73).
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 37 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether endpoint angular alignment (cosine similarity of local tangents) is a stronger predictor of correct cross-component matches than raw inverse Euclidean distance for split-gap reconnection.
- **Conclusion:** Among 484 confidently-mapped cross-component pairs within 30 µm (479 true, 5 false matches), cosine similarity achieved ROC AUC 0.8985, compared with 0.7292 for inverse Euclidean distance — angular alignment identified ~70% of true matches at zero false positives whereas distance plateaued at 50%. Recommends applying an aggressive angular-alignment filter before distance-based scoring.
- **Caveats:** Only 5 false-match pairs — ROC discrimination at low false-positive rates is heavily driven by a tiny negative class, so AUC may overstate selectivity.
- **Verdict:** Major issues
- **Test:** ROC analysis — Inverse-Euclidean AUC = 0.7292 vs Cosine-similarity AUC = 0.8985, computed over 479 true and **only 5 false** matches at a 30 µm radius (codeOutput line 22).
- **Statistical issues:** Catastrophic class imbalance — **5 negatives**. Each false match contributes a single horizontal step on the ROC curve, so the entire FPR axis has 5 discrete jumps. AUC = 0.8985 means that the cosine score correctly ranks true above false in about 90% of 479×5 = 2,395 pairs, which is just 240 "wins" out of 2,395 ranking-pair tests — comically small evidentiary basis. No significance test was performed (no DeLong test comparing the two AUCs), so even the "cosine > distance" comparison is anecdotal. Sample independence is again violated (leaf pairs aren't independent).
- **Logic issues:** "Outperforms" is a strong claim from 5 negative examples; the optimal-threshold recommendation ("cosine ≥ −0.42") is overfit to those 5 cases.
- **Verdict rationale:** With n_false = 5, any AUC > 0.7 is plausible by chance and no significance test was applied; the headline "AUC 0.90 vs 0.73" is too fragile to motivate an architectural recommendation.

### 12. (Priority 0.287 · Surprise 0.307) Split errors cluster near true biological branching points, motivating bifurcation-prior search in proofreading.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 40 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether U-Net split errors disproportionately cluster near true biological bifurcations, indicating segmentation difficulty at complex junctions.
- **Conclusion:** Mapping 7,534 leaf nodes to nearest GT branching points produced a median split-to-branch distance of 108.63 µm vs. 140.12 µm for a uniform control (KS = 0.094, p = 1.92e-29). The CDF is shifted toward branch points, supporting prioritized search around bifurcations.
- **Caveats:** KS = 0.094 is a modest effect size despite the very small p-value; the absolute median shift (~30 µm) is meaningful but not dominant.
- **Verdict:** Major issues
- **Test:** Two-sample Kolmogorov–Smirnov on distance-to-nearest-GT-branch-point; KS = 0.0941, p = 1.92e-29; n = 7,534 mapped fragment leaves vs n = 7,534 "uniform control" GT nodes.
- **Statistical issues:** The "uniform control" is sampled from **GT nodes**, not from a true cable-length-weighted uniform distribution along the GT skeleton — GT nodes are denser near branches because dense skeletons sample more nodes per branch. The comparison therefore conflates "split errors cluster near branches" with "GT node density itself clusters near branches," and the true effect could be substantially smaller (or zero). KS = 0.094 is modest. Survives Bonferroni only because n is huge.
- **Logic issues:** **The conclusion "splits cluster near branches" is partially confounded by the very control used to test it.** A proper test would resample control points along cable length using edge-length weighting, not by random GT-node selection.
- **Verdict rationale:** Conceptually plausible, but the control distribution is itself biased toward branches, undermining the comparison. The headline p-value is over-stated relative to the real effect.

### 13. (Priority 0.287 · Surprise 0.307) Split errors lie closer to merge errors than expected by chance, supporting coupled split-merge attention in GNN proofreaders.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 54 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether split errors and merge errors are spatially coupled — i.e., splits occur disproportionately in the immediate vicinity of merge sites, suggesting localized image ambiguities cascade into both error types.
- **Conclusion:** Across 46 merged fragments (223,828 nodes), 352 split loci and 6,843 control non-split leaves, median split-to-merge distance was 809.72 µm vs. 1016.66 µm for controls (KS = 0.125, p = 5.22e-5). The density histogram shows a heavy split concentration within 0–1000 µm of merges, motivating coupled split-merge attention mechanisms.
- **Caveats:** Only 46 merged fragments and 352 split loci; KS effect size remains modest (0.125), so the local "halo" of splits is real but not dramatic.
- **Verdict:** Minor issues
- **Test:** Two-sample Kolmogorov–Smirnov on distance-to-nearest-merge-locus distributions; KS = 0.1249, p = 5.2154e-05; n_split = 352, n_control = 6,843 (control = non-split leaves mapped to GT).
- **Statistical issues:** Effect size KS ≈ 0.12 is modest; the test's significance is largely n-driven. The control distribution (non-split leaves mapping to GT) shares geometry with split leaves so independence again is loose. Survives Bonferroni (1.43e-04 at m=349). Mean distances are still ~kilometer-scale (809 vs 1017 µm), reflecting GT sparsity rather than true biological proximity.
- **Logic issues:** "Supporting coupled split-merge attention in GNN proofreaders" is a strong architectural claim built on a KS = 0.12 distributional shift; the practical action depends on whether merge-resolution truly unlocks nearby splits, which this test does not establish causally.
- **Verdict rationale:** Direction is robust and the test is the right one, but the small effect size and architectural overreach put this in the "supportive evidence" category rather than a definitive finding.

### 14. (Priority 0.287 · Surprise 0.307) Whole-fragment geometric tortuosity (cable/bounding-box) flags merged components — merged fragments are ~2.8x more tortuous on log scale.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 55 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether fragment components containing merge errors exhibit significantly higher whole-fragment geometric tortuosity (cable length ÷ bounding-box diagonal) than clean fragments.
- **Conclusion:** Of 10,172 valid components, 20 merged components had mean log-tortuosity 0.4266 (std 0.3424) vs. 0.1512 (std 0.1394) for 10,152 clean components (Welch t = 3.60, p = 1.92e-3). The violin plot shows clean fragments tightly clustered at low tortuosity while merged ones spread into the high-tortuosity tail.
- **Caveats:** Only 20 merged components in the comparison; the p-value is in the 10^-3 range (not the 10^-12+ seen elsewhere), so this feature is supportive rather than dominant.
- **Verdict:** Major issues
- **Test:** Welch's t-test on log10(tortuosity) of merged vs clean components; t = 3.5969, p = 1.9205e-03; n_merged = 20, n_clean = 10,152.
- **Statistical issues:** **Fails Bonferroni** (1.43e-04 at m = 349) and lies in the borderline BH-FDR region — confirmed by run-wide FDR analysis. Severe class imbalance (20 vs 10,152) inflates the influence of any single high-tortuosity merged outlier; the reported merged std (0.34) is more than twice the clean std (0.14). Also, by construction merged components have at least two real neurites fused together, which automatically inflates cable-length-to-bounding-box ratio — so the test is partially **definitional** rather than empirical.
- **Logic issues:** "Whole-fragment geometric tortuosity flags merged components" overstates a p = 0.002 result against a definitional confound. The "2.8× more tortuous" headline is mean log-ratio; absolute tortuosity ratio is ~10^0.275 ≈ 1.88, and the spread is enormous.
- **Verdict rationale:** Borderline significance, very small positive class, definitional confound between merging and tortuosity — this finding would not survive strict multi-comparison control and is the textbook fragile-significant case.

### 15. (Priority 0.287 · Surprise 0.307) Most recoverable split reconnections (~59%) target internal (degree ≥ 2) nodes of merged fragments, so split correction requires merge-breaking.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 61 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** What fraction of recoverable true split connections require leaf-to-internal linking (target is an internal node of a falsely merged fragment) versus standard leaf-to-leaf reconnection.
- **Conclusion:** Of 1,067 valid split connections found across 39,826 leaves within 150 µm, 633 (59.3%) were leaf-to-internal and 434 (40.7%) were leaf-to-leaf. A majority of recoverable splits are *structurally blocked* by existing merges, proving that effective split correction must be coupled to merge-breaking logic.
- **Caveats:** 150 µm threshold is fixed; reconnection counts depend on this radius and the GT-mapping procedure, so absolute percentages could shift with stricter or looser thresholds.
- **Verdict:** Sound
- **Test:** **Descriptive proportion** — 1,067 split connections classified as leaf-to-leaf (434, 40.7%) vs leaf-to-internal (633, 59.3%). No null-hypothesis statistical test was performed.
- **Statistical issues:** No null test needed — this is a categorical proportion of an enumerated population, not a sample-vs-population inference. A 95% binomial CI on 633/1067 ≈ 59.3% gives roughly [56.3%, 62.2%], comfortably above 50%. Result is stable.
- **Logic issues:** Conclusion that "split correction requires merge-breaking" follows directly from the structural observation; deductive rather than inductive, and tightly scoped.
- **Verdict rationale:** Although this entry has no statistical test, none is required — it is a structural enumeration with a stable proportion. The architectural recommendation follows logically from the topology.

### 16. (Priority 0.287 · Surprise 0.307) Valid split reconnections show strongly anti-parallel endpoint tangents (mean cosine -0.47) vs. false proposals (+0.25), confirming orientation agreement as a robust feature.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 68 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether in competitive split-reconnection scenarios, the cosine similarity of endpoint tangent vectors is significantly closer to -1 (vectors pointing toward each other) for valid reconnections than for nearby false proposals.
- **Conclusion:** Within a 50 µm radius, 647 valid pairs had mean cosine -0.4726 ± 0.5622 vs. 10 false pairs at 0.2478 ± 0.6295 (Mann–Whitney p = 5.73e-4). Valid reconnections cluster near anti-parallel while false ones are scattered into the positive range, validating orientation agreement as a discriminative geometric feature.
- **Caveats:** Only 10 false-reconnection pairs — the effect direction is clear but the false-class distribution is poorly characterized.
- **Verdict:** Major issues
- **Test:** Mann–Whitney U on cosine similarity of endpoint tangents; statistic = 1183, p = 5.7261e-04; n_valid = 647 vs **n_false = 10**.
- **Statistical issues:** **n_false = 10 is severely underpowered**; with only 10 negatives the false-class mean ± std (0.2478 ± 0.6295) has a 95% CI roughly ±0.39, easily spanning 0. Survives Bonferroni at face value because the Mann–Whitney U with one tiny group is mechanically extreme, but the practical claim about "false reconnections cluster near +cosine" is essentially anecdotal. Independence of leaf-pairs is again violated.
- **Logic issues:** Conclusion that "orientation agreement is a robust feature" overstates an n=10 negative comparison. The recommendation might be true but is not validated.
- **Verdict rationale:** A p-value computed on n_false = 10 cannot decisively characterize the false distribution; the central distinction (anti-parallel for valid vs scattered for false) is plausible but rests on a class too small to validate.

### 17. (Priority 0.287 · Surprise 0.307) Highly tortuous neurite segments suffer ~2.5x the split rate of straight segments.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 70 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether neurite morphology complexity (tortuosity) causally predicts U-Net fragmentation at the per-segment level, measured by split rate per ~50 µm segment.
- **Conclusion:** Across 105,335 ground-truth segments, high-tortuosity segments (top quartile, ≥1.09) had mean split rate 0.25 splits/mm vs. 0.10 for low-tortuosity segments (bottom quartile, ≤1.04) — a 2.5x increase (Mann–Whitney U = 3.49e8, p = 9.3e-17). The violin plot shows most segments succeed but severe fragmentation outliers concentrate in tortuous segments.
- **Caveats:** Both groups have median split rate near zero, so the effect is driven by tail mass rather than typical-case differences — practical at population scale, weaker for per-segment prediction.
- **Verdict:** Minor issues
- **Test:** Mann–Whitney U on per-segment split rate (splits/mm); U = 3.49e8, p = 9.26e-17; n = 26,335 high-tortuosity vs 26,334 low-tortuosity segments (out of 105,335 total).
- **Statistical issues:** Both groups have median 0; the test is detecting a tail-mass difference, not a central tendency. Mann–Whitney with so many ties (zero values) is approximated, and the asymptotic p is anti-conservative when zeros dominate. n is huge, so power swamps small effects. Survives Bonferroni. Independence holds at the per-segment level.
- **Logic issues:** Conclusion phrased correctly as "highly tortuous segments suffer ~2.5× the split rate," and the rate ratio (0.25 vs 0.10) is a meaningful descriptive effect. However, the experiment word "causally predicts" overreaches — observational segment-level association cannot establish causation.
- **Verdict rationale:** Effect direction is real and reasonably sized at the population level; the per-segment classifier value is weak (per the report's own caveat). Calling tortuosity a "causal predictor" is the only overreach.

### 18. (Priority 0.287 · Surprise 0.307) Split endpoints cluster in fragmented "shatter zones" — ~6x higher local endpoint density than true terminations.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 73 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether split-error leaves sit in locally fragmented "shatter zones," indicated by significantly higher density of nearby fragment endpoints (50 µm radius) than around true anatomical terminations.
- **Conclusion:** Among 3,118 split leaves and 4,416 true terminations, mean endpoint density was 0.54 vs. 0.09 neighbors (Mann–Whitney p = 2.6e-275). Both medians sit at zero, but splits exhibit a heavy long tail of high-density clusters — confirming poor-signal regions trigger localized swarms of artifactual breaks.
- **Caveats:** Median = 0 for both groups means the effect is entirely in the tail; threshold-based classifiers on this single feature would have low recall.
- **Verdict:** Minor issues
- **Test:** Mann–Whitney U on neighbor count within 50 µm; U = 9,251,004, p = 2.5958e-275; n_split = 3,118, n_true = 4,416. Means 0.54 vs 0.09 neighbors; both medians = 0.
- **Statistical issues:** Extreme **zero-inflation** (both medians = 0); Mann–Whitney with massive ties gives an extreme p more from rank ties than from a genuine continuous shift. A zero-inflated negative-binomial or a hurdle model would be more appropriate. The p ≈ 2.6e-275 is computer-arithmetic-overflow-level and not meaningfully interpretable; the right reporting is "p < 1e-50". Survives any correction trivially.
- **Logic issues:** "~6x higher local endpoint density" headline (from 0.54 / 0.09) is reported as an aggregate ratio that is entirely tail-driven; per-leaf usefulness is low (the report acknowledges this).
- **Verdict rationale:** The directional finding (split leaves have a much heavier neighbor-count tail) is real and useful as a screening feature, but the p-value is essentially meaningless and the median-driven distributional shape needs an appropriate zero-inflated model.

### 19. (Priority 0.287 · Surprise 0.307) Fragments containing merge errors are ~4x more split per µm than clean fragments — topological errors are coupled.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 75 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether fragments containing at least one merge error also exhibit significantly higher split-endpoint density per µm of cable, indicating that locally complex image regions trigger both error types.
- **Conclusion:** Of 1,745 fragments with cable ≥ 2,000 µm, 13 merged fragments had mean split density 0.001645 leaves/µm vs. 0.000444 for 1,732 clean fragments — ~3.7x higher (Mann–Whitney U = 19,643, p = 3.26e-9). Supports the coupled-failure hypothesis: noisy local regions yield simultaneous merges and splits.
- **Caveats:** Only 13 merged fragments meet the 2,000 µm threshold — the per-fragment statistic rests on a very small positive class.
- **Verdict:** Major issues
- **Test:** Mann–Whitney U on per-fragment split density (valid leaves/µm); U = 19,643, p = 3.2562e-09; n_merged = **13**, n_clean = 1,732.
- **Statistical issues:** With n_merged = 13, even Mann–Whitney is fragile to the influence of single outlier merged fragments — the merged-group "mean 0.001645" is computed from 13 values. The cable-length filter (≥ 2000 µm) is post-hoc; the original 20 merged components from entries #14 and #15 drop to 13 here, so selection differs across related claims. The 13 merged fragments are **almost certainly the same fragments as the heavy-tortuosity ones (#14)** — these correlated findings inflate the apparent multi-axis evidence for "coupled errors". Passes Bonferroni at face value because U is extreme, but with n_merged = 13 the effect-size CI is wide.
- **Logic issues:** "Topological errors are coupled" is again a strong mechanistic claim from a 13-vs-1732 comparison. Direction is right; magnitude (~4× higher split density) is point-estimate only, no CI.
- **Verdict rationale:** Tiny positive class, observational design, overlapping fragment population with adjacent entries — the headline 4× is suggestive but not robustly characterized.

### 20. (Priority 0.287 · Surprise 0.307) Split-gap size is strongly negatively correlated with local fragment crowdedness, supporting adaptive search radii in proofreading.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 77 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether the spatial gap size of valid split reconnections is negatively correlated with local fragment density ("crowdedness") within a 30 µm radius — sparse regions producing long gaps, dense bundles producing short frequent gaps.
- **Conclusion:** Across 1,360 valid split pairs, Pearson r = -0.6086 (p = 1.26e-138) and Spearman ρ = -0.6668 (p = 1.21e-175), with a linear-regression slope of -19.08 µm per neighbor. Crowdedness strongly predicts shorter gaps, motivating dynamically narrowed search radii in dense bundles and wider radii in sparse regions.
- **Caveats:** Sample is limited to pairs found within a 100 µm gap; effect direction is unambiguous but the regression line is descriptive rather than causal — image-quality confounds may co-vary with crowdedness.
- **Verdict:** Major issues
- **Test:** Pearson r = -0.6086 (p = 1.26e-138) and Spearman ρ = -0.6668 (p = 1.21e-175) between gap size and crowdedness (unique fragment components within 30 µm of midpoint); n = 1,360 valid split pairs (codeOutput lines 21–24).
- **Statistical issues:** **The correlation is largely definitional / induced by sampling.** The crowdedness query radius (30 µm) is *smaller* than the gap radius (100 µm), and the midpoint always sits at distance gap/2 from each endpoint. So for a *short* gap (say 10 µm), both leaves are within 5 µm of the midpoint and contribute their own components to the crowdedness count, mechanically inflating crowdedness for small gaps. For long gaps (>60 µm), the original two endpoints fall *outside* the 30 µm midpoint radius and cannot contribute. This produces a **geometric correlation by construction** — the negative correlation is essentially a function of the chosen radii, not a biological signal. The 1,360 pairs are also non-independent (the same fragment endpoint appears in multiple pairs). The reported p ≈ 10^-138 is computer-arithmetic-level and not interpretable.
- **Logic issues:** Conclusion "supporting adaptive search radii" relies on the negative correlation being a real biological/imaging signal; the test as constructed cannot distinguish that from geometric self-similarity in the sampling procedure. The plot analysis itself flags that the linear model predicts negative distances at crowdedness ≥ 5 — the relationship is non-linear and partially definitional.
- **Verdict rationale:** A large headline correlation that is at least partially an artifact of the radius mismatch between crowdedness and gap; the result needs to be re-derived with a crowdedness window independent of the gap to be trustworthy.

---

## Statistical Verification — Run-wide Summary

### Top-20 verdict breakdown

| Verdict | Count | Entries |
|---|---|---|
| Sound | 1 | #15 |
| Minor issues | 9 | #1, #2, #3, #5, #6, #7, #9, #13, #17 |
| Major issues | 10 | #4, #8, #10, #11, #12, #14, #16, #18, #19, #20 |
| Invalid | 0 | — |

Half of the top 20 entries carry **Major issues** — either the test choice was inappropriate (rank 4 χ² with E ≈ 0.07), the comparison was confounded by sampling design (ranks 10, 12, 20), the positive class was too small to support the claim (ranks 11, 14, 16, 19), the conclusion was descriptive without a null test (rank 8), or the headline p-value reflects extreme tie-driven overflow rather than meaningful evidence (rank 18).

### Multiple comparisons (Benjamini–Hochberg / Bonferroni)

A regex pass over the `analysis` and `codeOutput` fields of all 349 records in the three run exports extracted 195 numeric p-values (the remaining 154 records either report no explicit p-value, contain only descriptive statistics, or use non-standard formatting). Of those:

- **120 (61.5%) of extractable p-values are below the conventional α = 0.05.**
- 85 (43.6%) are below 0.001.
- 55 (28.2%) are below 1e-10.
- 35 (17.9%) lie in the borderline 0.001 ≤ p < 0.05 range — the candidates most vulnerable to FDR / Bonferroni control.

**Benjamini–Hochberg FDR control at q = 0.05** (taking m = 349 total hypotheses): the largest k with p_(k) ≤ k/m · q has BH threshold p ≤ ~4.0e-03. **111 of the 120 significant p-values pass BH at q = 0.05**, so the bulk of headline findings are robust to FDR control. The 9 that fail BH all sit in the 0.005–0.05 borderline; the most prominent **top-20 finding to fall to BH is rank #14 (refinement ID 55, "whole-fragment tortuosity flags merges", p = 1.92e-03)** — it sits right at the BH frontier and would be classified as fragile under strict FDR control.

**Bonferroni control at α = 0.05 / 349 ≈ 1.43e-04**: among the top 20, the three explicit Bonferroni failures are **rank #6 (KS = 0.10, p = 5.07e-04, voxel anisotropy)**, **rank #14 (p = 1.92e-03, whole-fragment tortuosity)**, and **rank #16 (p = 5.73e-04, anti-parallel tangents with n_false = 10)**. All other top-20 entries with an extracted p-value pass Bonferroni by orders of magnitude.

Across the *full pooled set* of 349 hypotheses: **25 of 195 reported p-values survive Bonferroni at α = 0.05/349** (≈12.8%). Most "discoveries" in the borderline 0.001–0.05 range should not be treated as established findings.

### Most serious systemic problems

1. **Tiny positive classes inflating headline odds ratios.** Several top findings — rank #1 (20/1,001 merged), #4 (2 proximal merges), #11 (5 false matches), #14 (20 merged components), #16 (10 false reconnections), #19 (13 merged fragments) — rest on positive-class sizes that cannot characterize a population's tail. Effect sizes have wide point-estimate uncertainty even when the headline p is tiny.

2. **Invalid χ² approximation.** Rank #4 (entry ID 172) applies a Pearson χ² to a 2×2 cable-length-weighted table where the expected proximal count is ≈ 0.07; this drives almost the entire χ² = 50.4 statistic. A Fisher's exact or Poisson-rate test would be appropriate and would produce a much larger p-value.

3. **Sampling-induced ("definitional") correlations.** Rank #20 (entry ID 77) reports r = −0.61 between gap size and "crowdedness", but the geometry of the midpoint query forces a mechanical negative correlation. Rank #14 (#55) reports tortuosity vs merge status, but merged components are tortuous *by definition* (fused arbors). Rank #10 (#30) reports leaf-vs-internal drift that is partly a tautology of the GT-mapping geometry. Rank #12 (#40) compares splits against a "uniform" GT-node control that is itself non-uniform with respect to branches.

4. **Independence violations from leaf-pair / hub-pair enumeration.** Ranks #5 (#88), #11 (#37), #16 (#68), #6 (#1), #7 (#7), #13 (#54), #20 (#77) all derive samples from `KDTree.query_pairs` or per-hub branch enumeration — each leaf appears in multiple pairs, and pairs on the same fragment share geometry. Welch's t, Mann–Whitney U and ROC AUC all assume independent observations; the reported p-values are anti-conservative, and AUC values are inflated by reuse.

5. **n-driven over-significance with negligible effect sizes.** Ranks #9 (KS = 0.053, p = 1e-12, mean diff in tortuosity ≈ 0.008), #7 (KS = 0.054), #13 (KS = 0.125), #18 (medians both = 0) all produce extreme p-values because sample sizes are in the tens-of-thousands; the practical discriminative power is weak and is acknowledged in caveats but still framed as "strong support" in the analysis text.

6. **Mechanistic / causal conclusions from cross-sectional / observational tests.** Ranks #3, #10, #17, #19 all use single two-sample comparisons to assert mechanisms (low-SNR hallucination, pre-split drift, tortuosity-causes-splits, coupled failure modes) that the experiment design cannot causally isolate; these are observational associations and the rhetorical move from association to mechanism is the most common logic error across runs.

7. **Mass-significance plus no run-level FDR control.** With 195 reported p-values and roughly half of all hypotheses returning p < 0.05, roughly 12 false positives at α = 0.05 are expected by pure chance. BH control preserves 111 of those, but the 9 that fail BH (all borderline) include at least one top-20 finding (rank #14); strict Bonferroni eliminates an additional ~85 borderline findings beyond the top 20. The discovery loop never applied any multiplicity adjustment when updating posteriors.

A scientist scanning this run should treat **ranks #1, #2, #5, #15, #17** as solidly supported (large, robustly-tested effects), **#3, #6, #7, #9, #13** as plausible but underpowered-or-tiny-effect, and **ranks #4, #8, #10, #11, #12, #14, #16, #18, #19, #20** as findings to verify with a redesigned experiment before relying on them — they are the discoveries this audit identifies as most likely to be spurious or confounded.
