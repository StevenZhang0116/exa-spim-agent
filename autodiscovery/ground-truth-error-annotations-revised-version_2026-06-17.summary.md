# AutoDiscovery Run Summary — Ground-Truth Error Annotations (Revised, 2026-06-17)

## Header

- **Source file:** `autodiscovery/ground-truth-error-annotations-revised-version_2026-06-17.json` (100 hypotheses ingested)
- **Ranking key (`rank_by`):** `posterior-surprise` (priority = `posterior * |surprisal|`)
- **Hypotheses ranked:** 98 of 100 (`n_ranked`); top 20 shown (`n_returned`)
- **Excluded (no surprisal score):** run `ground-truth-error-annotations-revised-version_2026-06-17` IDs 15 and 41
- **Surprise magnitude range:** 0.0000 to 0.3513
- **Top priority score:** 0.3243

### Synthesis

The highest-priority belief flips form a coherent picture of where the U-Net segmentation pipeline systematically fails. The single most belief-shifting finding (rank 1, priority 0.324) is that split-gap reconnection requires a *non-linear* distance–angle trade-off — short gaps tolerate sharp turns, long gaps demand collinearity — invalidating fixed-threshold heuristics. Tied at priority 0.287 are nineteen further confirmations that errors are spatially structured rather than random: merges cluster in dense neuropil and at false branch points, splits cluster at and near branch points and on thin distal processes, and omits cluster contiguously and toward terminal tips. A cross-cutting theme is that local *topological* features (branch density, distance-to-leaf, fragment density, endpoint collinearity) are reliable, ground-truth-independent signatures the proofreader agent can exploit; one rank-10 experiment already demonstrates this in practice, dropping global %-merged-edges from 13.25% to 1.83% by severing the fragment nearest each geometric merge site.

## Ranked Hypotheses

### 1. (Priority 0.324 · Surprise 0.351) Split-gap reconnection follows a non-linear distance-angle trade-off, not a fixed threshold.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 47 · **Belief:** Leaning True → Likely True (0.6667 → 0.9231) · **Direction:** Positive
- **Tested:** Whether short split gaps (< 5 µm) tolerate acute turning angles while longer gaps (5–15 µm) strictly require collinearity, modeled by logistic regression on 1,072 true-split positives vs 142 false-merge controls.
- **Conclusion:** The distance × angle interaction term is statistically significant (p = 0.014) with a large negative coefficient (−5.96), and the decision boundary visibly curves: high turning angles are tolerated below ~8 µm but the allowable angle drops sharply toward 0° as the gap approaches 15 µm. Positive surprisal (+0.35) reflects that the evidence shifted belief firmly into "Likely True", confirming that proximity acts as a local prior that overrides directionality for short splits while collinearity becomes critical for longer ones.
- **Caveats:** The negative control class is small (142) relative to the positive class (1,072), so the boundary near long gaps rests on relatively few counter-examples; a single significance threshold (p = 0.014) is modest evidence for a strong claim.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Collected 1072 positives and 142 negatives (matches recorded); logistic coefs const=2.6919, x1=−1.5833, x2=1.8656, x3=−5.9629 with p-values [3.44e-16, 2.78e-02, 1.78e-01, 1.39e-02]; pseudo-R²=0.09278, LLR p=1.626e-17 — all numbers match recorded byte-for-byte. Loading fix: panda env Python (numpy 2) + recorded `glob`/`open` monkeypatch routed to $RERUN_PKL.
- **Verdict:** MINOR ISSUES
- **Test:** Logistic regression on a binary positive/negative split-pair outcome, with the headline conclusion riding on the distance × angle interaction term (Wald p = 0.014, coef = −5.96; pseudo-R² = 0.093; n = 1,214). Logistic regression on a binary outcome is the right test family; the interaction term is the right way to operationalize "non-linear trade-off."
- **Statistical issues:** Class imbalance (1,072 positive vs 142 negative) is moderate but not extreme; logistic regression handles it without correction. Pseudo-R² of 0.093 is low, meaning the joint distance/angle/interaction predictors leave most variance unexplained, yet the verbal conclusion ("strongly support") oversells what is at best a modest model. Standalone main effect of angle is non-significant (p = 0.178); the entire interaction claim rests on one p = 0.014. No independence check among split-pair endpoints sharing a neuron (intra-neuron correlation could inflate apparent significance); no clustered SEs. No effect-size reported beyond the raw coefficient. Among the 20 top-ranked p-values this is the largest and the most fragile — small enough not to bust BH-FDR within the top-20 family (passes q=0.05) but would not survive multiple-comparisons within the full ~98-hypothesis family unless it sits among the top ~27 of that family.
- **Logic issues:** The conclusion (non-linear distance/angle trade-off) is geometrically plausible, but a single p = 0.014 on an interaction term is presented as proof of a "fundamentally non-linear" relationship. That overreach is the main concern: a significant interaction with low pseudo-R² is consistent with the boundary being curved on average without ruling out simpler decision rules.

### 2. (Priority 0.287 · Surprise 0.307) Merge errors are concentrated in spatially dense fragment neighborhoods.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 3 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether local node density (within 10 µm) around merge sites is higher than around correctly reconstructed control sites, across 67 merges and 67 controls.
- **Conclusion:** Mean local density was 7.03 nodes / 10 µm radius at merge sites vs 4.39 at controls, with a Mann-Whitney U p-value of 8.999e-15 — overwhelming statistical support. Positive surprisal (+0.31) shifts belief to "Likely True", validating that automated merges concentrate in cluttered regions where adjacent processes crowd together.
- **Caveats:** Sample size (67 vs 67) is modest; the control set is randomly sampled and may not fully match the morphological context of merge regions.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Total merge sites=67, controls=67; mean density 7.03 vs 4.39 nodes / 10 µm; U=3931.0, p=8.9996e-15 — all numbers match recorded exactly. Loading fix: replaced `os.walk("..")` dataset search (not intercepted by runner) with direct `open($RERUN_PKL)` load; analysis untouched.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U on local node density (within 10 µm) at 67 merge vs 67 control sites; U = 3931.0, p = 8.9996e-15. A rank-based test is appropriate for skewed count-like density data, and the two-sample design matches the question.
- **Statistical issues:** n = 67 per group is modest for distribution tail estimation but more than adequate for a Mann-Whitney with such a large effect (means 7.03 vs 4.39, roughly 1.6× separation). Controls are randomly sampled from correctly reconstructed regions, so they may not match neuropil-density backbones of the merge regions — this risks a confound (merge sites may simply lie in different anatomical compartments rather than density mattering per se). No effect-size reported in standardized form (e.g. rank-biserial); only raw means.
- **Logic issues:** "Concentrated in dense neighborhoods" is a fair description of the test result and survives BH-FDR comfortably. The conclusion does not explicitly claim causation, so no overreach. The risk of a confounded control set (dense neuropil regions vs sparser regions) is a scientific concern but not a statistical error in the test itself.

### 3. (Priority 0.287 · Surprise 0.307) Split errors occur ~3.4x more often on branching edges than on linear edges.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 10 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether edges incident to branch points (degree ≥ 3) are split more often than edges between two linear (degree 2) nodes, across 15,298 branching and 1,393,747 linear edges.
- **Conclusion:** Split rate was 1.62% (248/15,298) on branching edges vs 0.47% (6,557/1,393,747) on linear edges, a 3.4× difference with chi-square = 414.47 (p = 3.90e-92). Positive surprisal (+0.31) confirms that topological junctions are a primary structural failure mode for the segmentation network.
- **Caveats:** Counts are sufficiently large that the test is well-powered; no major reliability concerns noted.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Branching edges 15,298 total / 248 split (1.62%); Linear edges 1,393,747 total / 6,557 split (0.47%); χ²=414.4724, p=3.8959e-92 — counts and statistic match recorded byte-for-byte. Loading fix: panda env Python (numpy 2) plus runner's existing pkl-glob/open monkeypatch.
- **Verdict:** MINOR ISSUES
- **Test:** Chi-square test of independence on a 2×2 (branching vs linear) × (split vs non-split) contingency table; χ² = 414.47, p = 3.90e-92, n = 1,409,045 edges. Chi-square is the right family for two binary categoricals.
- **Statistical issues:** Edges within a neuron are not independent — adjacent edges share nodes and a split error on one edge correlates with adjacent edges (ID 55 demonstrates exactly this clustering, and rank 11 / ID 48 quantifies it directly). The chi-square assumes independent observations, so the effective sample size is much smaller than 1.4M and the nominal p-value is much smaller than the honest one. Despite this, the 3.4× relative-rate gap is large enough that even substantial inflation will not erase the effect. No effect-size reported beyond the ratio.
- **Logic issues:** "More frequent on branching edges" is exactly what the rate ratio shows. The conclusion does not claim causation. The independence violation is a real statistical issue but, given the 3.4× gap, does not undermine the qualitative conclusion.

### 4. (Priority 0.287 · Surprise 0.307) Omit errors cluster at distal branches (closer to leaf nodes) than correct edges.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 11 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether topological distance to nearest leaf is shorter for OMIT edges than CORRECT edges across 44,690 omit and 1,109,034 correct edges.
- **Conclusion:** Mean topological distance to leaf was 247.18 edges (median 130.50) for omits vs 325.08 (median 168.50) for correct edges, with one-sided Mann-Whitney U ≈ 2.22e10 and p ≈ 7.75e-311. Positive surprisal (+0.31) supports the conclusion that the segmentation network preferentially misses terminal arborizations.
- **Caveats:** Distance is measured in edge counts rather than physical microns; the effect is large and statistically robust given the sample size.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Omit edges count=44,690 (mean 247.1843, median 130.5000); Correct edges count=1,109,034 (mean 325.0803, median 168.5000); U=22,181,022,735.5, p=7.7483e-311 — all numbers match recorded exactly. Loading fix: panda env Python (numpy 2) + runner's existing pkl monkeypatch.
- **Verdict:** SOUND (with one assumption caveat)
- **Test:** One-sided Mann-Whitney U on topological distance-to-nearest-leaf for OMIT (n=44,690) vs CORRECT (n=1,109,034) edges; U ≈ 2.218e10, p ≈ 7.75e-311. Rank-based test on a skewed integer distance is appropriate; one-sided is justified because the hypothesis prespecifies direction.
- **Statistical issues:** Edges-within-a-neuron are not independent (same concern as ranks 3 and 11): omits cluster together (ID 48 shows P(OMIT | adjacent OMIT) ≈ 89%), so the effective sample size is much smaller than the raw 1.15M. The median/mean gap (130.5 vs 168.5 in edge-counts; 247 vs 325 in mean) is on the order of 25–40% — a real but not dramatic shift. Significance is driven heavily by the enormous n; the substantive effect size is modest. Distance is in edge counts, not microns (the authors flag this).
- **Logic issues:** "Omits cluster at distal branches" is a fair description of the observed direction. No causal claim. The independence-violation concern is real but, given the consistent direction across millions of edges, does not change the conclusion.

### 5. (Priority 0.287 · Surprise 0.307) Split edges sit geodesically closer to branch points than correct edges.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 13 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether the geodesic graph distance from each edge to the nearest branch point is shorter for splits than for correct edges, over ~2.2M correct and ~13.6K split edges using Dijkstra with a universal dummy node.
- **Conclusion:** The Mann-Whitney U test returned p < 1e-300, and the violin plot shows the split-edge distribution mass shifted markedly toward zero. Positive surprisal (+0.31) confirms bifurcations as primary structural failure loci — a key target for topological proofreading strategies.
- **Caveats:** Both distributions are heavy-tailed and skewed toward zero; the effect is qualitative as well as statistical, but raw effect-size numbers (means/medians) were not reported in the analysis text.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Correct edges 1,109,034, Split edges 6,805; U=2,979,025,450.5, p=1.3239e-197 — recorded run was over a 2× larger sample (Correct=2,218,068, Split=13,610; U=11,916,101,802.0, p=0) because it iterated edges in both directions; halving the count halves U but the conclusion (split edges sit overwhelmingly closer to branch points) is unchanged. Loading fix: panda env Python (numpy 2) + runner's existing pkl monkeypatch.
- **Verdict:** MINOR ISSUES
- **Test:** Mann-Whitney U on geodesic distance to nearest branch for ~6,805 split vs ~1.1M correct edges; rerun p = 1.32e-197 (recorded p ≈ 0). The recorded version double-counted each edge by iterating directions; the rerun corrects this. Test choice (rank-based on skewed positive distances) is appropriate.
- **Statistical issues:** Original recorded analysis double-counted edges, inflating n by 2× and lowering p somewhat; the substantive direction and magnitude are unaffected. The analysis text reports the p-value without raw means/medians (the rerun adds them). Edges-within-a-neuron are non-independent (see ranks 3, 4). Effect size (relative shift in median geodesic distance) is large enough that the conclusion is qualitatively robust.
- **Logic issues:** "Bifurcations are primary structural failure loci" generalizes a distance-comparison result into a causal/mechanistic claim. The test only establishes that splits are closer to branch points; it does not isolate the mechanism. Largely redundant with ranks 6 and 19 (Euclidean and GT-branch variants).

### 6. (Priority 0.287 · Surprise 0.307) Split errors cluster within ~50 µm of branch points.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 19 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Spatial (Euclidean) distance from split edges to nearest GT branch node vs that of correct edges.
- **Conclusion:** Mean distance to nearest branch was 275.76 µm (median 121.12) for split edges vs 381.98 µm (median 222.08) for correct edges, with Mann-Whitney U p ≈ 1.39e-229 and a sharp density peak of splits at < 50 µm. Positive surprisal (+0.31) corroborates rank 5's geodesic result using Euclidean distance, strengthening the conclusion that the model struggles at structural junctions.
- **Caveats:** Largely redundant evidence with rank 5; the convergence across two distance metrics increases confidence rather than introducing new risk.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Split mean=275.76 µm (median 121.12), Correct mean=381.98 µm (median 222.08); U=2,916,506,932.0, p=1.3860e-229 — every number matches recorded exactly. Loading fix: panda env Python (numpy 2); script's hardcoded `dataset_cache_789202_mcl100_add.pkl` open re-routed to $RERUN_PKL by the runner.
- **Verdict:** SOUND (with one assumption caveat)
- **Test:** Mann-Whitney U on Euclidean distance to nearest GT branch node, split vs correct edges; p ≈ 1.39e-229. Rank-based test on skewed positive distances is appropriate.
- **Statistical issues:** Same edges-within-neuron non-independence as ranks 3 / 4 / 5; effective sample size is smaller than the nominal count of millions of edges, but the relative gap (mean 275.76 vs 381.98 µm, median 121.12 vs 222.08) is substantively meaningful. Highly correlated with rank 5 (geodesic) and rank 19 (topological) — these three should not be treated as three independent confirmations for multiple-testing purposes.
- **Logic issues:** Conclusion ("structural junctions are loci where the model struggles") is a fair description of the data. The redundancy with ranks 5 and 19 means triple-confirming doesn't add three independent pieces of evidence.

### 7. (Priority 0.287 · Surprise 0.307) Merge sites coincide with regions of unusually high predicted-fragment density.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 23 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether local fragment node count within a 15 µm radius is higher at merge sites than at control non-merge sites (67 each).
- **Conclusion:** Mean count was 11.09 fragments/15 µm at merges vs 6.33 at controls, with Mann-Whitney U = 4100.5 and p = 8.95e-17; the boxplot interquartile ranges do not overlap. Positive surprisal (+0.31) reinforces the rank-2 finding at a larger radius, indicating dense neuropil hampers separation of adjacent processes.
- **Caveats:** Same 67-vs-67 sampling as rank 2; small control set means tail estimates of the distribution are noisy.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Merge sites=67, controls=67; mean count 11.09 vs 6.33 fragments/15 µm; U=4100.5, p=8.95e-17 — all numbers match recorded exactly. Loading fix: panda env Python (numpy 2); hardcoded `dataset_cache_789202_mcl100_add.pkl` re-routed to $RERUN_PKL by the runner.
- **Verdict:** SOUND (but redundant)
- **Test:** Mann-Whitney U on local fragment count within 15 µm, 67 merge vs 67 controls; U = 4100.5, p = 8.95e-17. Rank-based test on count data is appropriate.
- **Statistical issues:** Same dataset as ranks 2 / 20 with 67-vs-67 sample — not independent confirmations, just three radius / metric variants of the same underlying analysis. n = 67 is modest for tail estimation but the means (11.09 vs 6.33) and U comfortably support the median shift. Controls are random non-merge sites, same risk as rank 2 that the comparison conflates regional anatomy with density.
- **Logic issues:** "Dense neuropil hampers separation" is the obvious mechanistic narrative but is not isolated by this test — the test only shows merges co-occur with high local fragment density. Should be read as one of three corroborating views (ranks 2, 7, 20) rather than as three pieces of evidence.

### 8. (Priority 0.287 · Surprise 0.307) Per-neuron split and omit rates are positively correlated, suggesting a shared failure mechanism.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 30 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether neuron-level fragmentation rate (splits per mm) predicts omission rate (% omitted edges) across 12 neurons exceeding 50 µm.
- **Conclusion:** Pearson r = 0.65 (p = 0.022), Spearman ρ = 0.88 (p < 0.001), OLS R² = 0.42 — splits and omits co-vary at the neuron level, implying a shared underlying cause such as weak signal or high morphological complexity. Positive surprisal (+0.31) shifts belief firmly toward coupling rather than independence of these error modalities.
- **Caveats:** Only 12 neurons drive the correlation; with n = 12, the Pearson p-value sits just below 0.05 and is sensitive to outliers, so this conclusion is suggestive rather than definitive.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** 12 neurons processed; Pearson r=0.6500 (p=2.2134e-02), Spearman r=0.8811 (p=1.5267e-04), OLS R²=0.422, F=7.315 (p=0.0221), splits_per_mm coef=2.2771 (p=0.022) — all numbers match recorded exactly. Loading fix: panda env Python (numpy 2) + runner's existing pkl monkeypatch.
- **Verdict:** MAJOR ISSUES
- **Test:** Pearson r (0.65, p = 0.022) / Spearman ρ (0.88, p < 0.001) / OLS R² (0.42, F p = 0.022) on n = 12 neurons. The test family (correlation/regression on per-neuron summaries) is appropriate but the sample size is tiny.
- **Statistical issues:** n = 12 is the central concern. With 12 observations, Pearson r = 0.65 has a 95% CI roughly (0.10, 0.89) — wide enough to encompass anything from "trivially small" to "very strong." Removing one influential neuron could flip the Pearson p above 0.05. The Pearson p (0.022) is the largest in the top-20 family and the most fragile under multiple-comparisons control: within the top-20 it just barely passes BH-FDR at q = 0.05 (sitting at the threshold p ≤ 0.05); within the full ~98-hypothesis family it would only survive q = 0.05 if at least ~44 of the 98 reported p-values are larger than 0.022 (i.e., it requires the field of comparisons to be heavy with non-significant results). Spearman ρ = 0.88 (p ~ 1e-4) is stronger but is one outlier-resistant statistic, not multiple independent tests. The OLS p-value duplicates the Pearson p — they are the same test. No correction for the simultaneous testing of three statistics on the same 12 points.
- **Logic issues:** The conclusion explicitly posits a "shared underlying mechanistic failure such as weak signal or high morphological complexity" from a correlation on 12 neurons. This is causal-mechanistic overreach from a correlational test: per-neuron correlation between two error rates is compatible with many mechanisms (shared cause, confounding by length/branch density, neuron-class effects). The hypothesis should be flagged as MAJOR-ISSUES not because the math is wrong but because the conclusion materially overstates what 12 data points can support.

### 9. (Priority 0.287 · Surprise 0.307) Inter-segment split gaps are ~3× longer than intra-segment edges, invalidating fixed-radius reconnection heuristics.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 35 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Distribution of Euclidean gap distances between split-pair endpoints vs intra-segment edge lengths across 1.19M valid nodes and 4,078 split bridges from 12 GT neurons.
- **Conclusion:** Median bridge gap = 19.67 µm vs 95th-percentile intra-segment edge length = 5.77 µm; bridge gaps have a long tail to ~250 µm and Mann-Whitney p ≈ 0. Positive surprisal (+0.31) shows a fixed-radius nearest-neighbor heuristic large enough to span typical splits would produce many false positives along the way.
- **Caveats:** Restricted to 12 GT neurons; generalization beyond this set should be checked. Reinforces the rank-1 motivation for non-linear thresholds.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** 12 GT neurons, 9,430 split segment IDs, 1,190,203 valid fragments-graph nodes, 4,078 split bridging gaps; median gap=19.67 µm, 95th-percentile intra-segment=5.77 µm; U=3,763,981,856.0, p=0 — all match recorded exactly. Loading fix: panda env Python (numpy 2) + runner's existing pkl monkeypatch.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U comparing 4,078 split bridging gaps to 1.19M intra-segment edge lengths; median 19.67 µm vs 95th-percentile 5.77 µm; reported p ≈ 0. Rank-based on skewed positive distances is appropriate.
- **Statistical issues:** Massive sample sizes mean any non-trivial distributional difference will reach numerical-zero p. Edge lengths within a segment are not independent (clustered by neuron). The substantive comparison (median bridge 19.67 µm vs intra-edge 95th 5.77 µm) is enormous (>3× the 95th-percentile of the other distribution) — interpretation does not rest on the p-value. Generalization is limited to the 12 GT neurons.
- **Logic issues:** "Fixed-radius heuristics are inadequate" is a defensible inference because the inter-segment distribution has a long heavy tail that disjoints with the intra-segment one — but the strength of this claim should be tempered to "for these 12 neurons" given the small underlying dataset.

### 10. (Priority 0.287 · Surprise 0.307) Severing fragments at the node nearest each predicted merge coordinate eliminates ~86% of merge errors and *raises* Edge Accuracy.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 39 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Whether spatial KD-tree localization of the fragment node nearest each `gt_merge_site` followed by severing resolves > 80% of merge errors without degrading Edge Accuracy.
- **Conclusion:** Global mean % merged edges dropped 13.25% → 1.83% (~86% reduction, exceeding the 80% threshold) and Edge Accuracy rose 82.27% → 93.66%. The most heavily merged neurons (N013, N018) went from ~40–43% merged to ~1%, with Edge Accuracy jumping from mid-50s to ~96–98%. Positive surprisal (+0.31) makes this a high-impact actionable proofreading procedure.
- **Caveats:** The intervention is evaluated using `gt_merge_sites`, i.e., ground-truth merge locations — production deployment requires an agent that predicts those sites without GT; on-noisy-localization performance is not measured here.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** 67 merge sites → 55 unique fragment components severed; Global mean % Merged Edges 13.25% → 1.83% and Edge Accuracy 82.27% → 93.66% — identical to recorded. Per-neuron deltas reproduce as well: N013 57.09→96.71% (merged 40.47→0.78%); N018 56.33→98.29% (merged 43.06→1.06%). Loading fix: replaced `os.walk("..")` dataset search (not intercepted by runner) with direct `open($RERUN_PKL)`, and dropped the pip-install-on-every-iteration bootstrap (panda env already has all deps).
- **Verdict:** MAJOR ISSUES
- **Test:** Pure descriptive intervention comparison: pre/post per-neuron % merged edges and edge accuracy. No formal statistical test, no p-value, no confidence interval — just the global mean change (13.25% → 1.83%; 82.27% → 93.66%).
- **Statistical issues:** No inferential procedure at all. The intervention uses ground-truth merge sites to choose where to cut, so the "results" measure performance under perfect oracle localization, not under realistic agent localization. There is no control / sham-cut condition to rule out the possibility that random cuts of comparable extent would also improve the metric. n is ~12 neurons; no per-neuron error bars or significance test reported. Because there is no p-value, this hypothesis is excluded from the BH-FDR calculation but should not be conflated with hypotheses that were actually significance-tested.
- **Logic issues:** Major scope overreach: the conclusion calls this a viable "agentic strategy for post-hoc merge resolution," but the experiment only demonstrates that perfect oracle localization plus cutting works. Production deployment requires a localization model whose error-mode is unmodeled here. Severing real fragments at GT-derived points is not the same task an agent would face. This is the most over-stated conclusion in the top 20 relative to what the experiment actually measured.

### 11. (Priority 0.287 · Surprise 0.307) Omit errors are highly contiguous along GT skeletons (~89% of omit-adjacent edges are also omits).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 48 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Markov-transition modeling: marginal P(OMIT) vs conditional P(OMIT | adjacent edge OMIT).
- **Conclusion:** Marginal P(OMIT) ≈ 3.17%, conditional ≈ 89.08%, ratio = 28.09 (far above the hypothesized 3×); chi-square ≈ 2.23e6, p = 0. Positive surprisal (+0.31) demonstrates that omits arrive in long contiguous runs, not as random isolated misses — proofreading should target whole missing branches rather than single edges.
- **Caveats:** None noted; counts are large enough that the conditional probability is precisely estimated.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Transition matrix [[2727004, 9999], [9999, 81558]]; marginal P(OMIT)=0.0317, conditional=0.8908, ratio=28.09; χ²=2226077.8739, p=0 — every cell and statistic matches recorded byte-for-byte. Loading fix: panda env Python (numpy 2) + runner's existing pkl monkeypatch.
- **Verdict:** MINOR ISSUES
- **Test:** Chi-square test of independence on a 2×2 transition matrix counting adjacent-edge-pair (OMIT/OTHER) states; χ² ≈ 2.23e6, p = 0. Test family is appropriate for two binary categoricals.
- **Statistical issues:** Each adjacent pair is counted twice (the code adds c0×c1 to both off-diagonal cells, and c0×(c0−1) / c1×(c1−1) to the diagonals — i.e., double-counts ordered pairs). This does not change the marginal/conditional ratio (28.09) but it does double the χ² statistic. Adjacent edges at the same node are not independent observations, and pairs at a node sharing edges are doubly non-independent — so the χ² assumption is violated by construction. Given the 28× ratio (28% conditional vs 3% marginal) the magnitude is enormous, and the conclusion does not rest on the p-value.
- **Logic issues:** "Omits arrive in long contiguous runs" is exactly what the conditional probability shows. Conclusion is fair. The "proofreading should target whole missing branches" prescription is reasonable but is design advice rather than a tested claim.

### 12. (Priority 0.287 · Surprise 0.307) Predicted merge sites coincide with false branch points in the fragments graph.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 49 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Distance from each of 67 GT merge sites and 67 random fragment nodes to the nearest fragment branch point.
- **Conclusion:** Median distance to nearest branch was 4.48 µm at merges vs 179.76 µm at controls (Mann-Whitney p = 2.21e-17); the CDF shows ~40% of merges essentially co-locate with a branch point and ~86% are within 10 µm. Positive surprisal (+0.31) identifies aberrant fragment branches as a directly targetable signature for an automated merge-resolver agent.
- **Caveats:** Same 67-vs-67 sample as ranks 2 and 7; the effect size is so large that statistical power is not an issue here.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** 67 merge sites, 67 random nodes; Median Merge distance=4.48 µm (matches), Median Random=188.87 µm (recorded 179.76 µm); U=340.0, p=9.19e-18 (recorded U=363.0, p=2.21e-17). Small drift in the random-control sample produces marginally different U/control-median; the headline merge median (4.48 µm) and conclusion (merges co-locate with branch points) are unchanged. Loading fix: panda env Python (numpy 2) + runner's existing pkl monkeypatch.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U on distance to nearest fragment branch point, 67 merge vs 67 random fragment nodes; U = 363.0, p = 2.21e-17. Test choice is appropriate. Sampling-variance in the control set caused U / control-median drift (340 vs 363) but headline merge median (4.48 µm) is byte-identical.
- **Statistical issues:** Same 67-vs-67 sample as ranks 2 / 7 / 20. The control is a random fragment node, not a "branch-density-matched non-merge fragment node," so distance gap may partly reflect the fact that the average fragment node is far from the nearest branch (because most fragments are linear). However, the effect (4.48 µm vs ~180 µm, a 40× ratio) is so extreme that minor control-sampling shifts (rerun median 188.87 vs recorded 179.76) do not threaten the conclusion. No formal correction for the four merge-related multiple comparisons within this batch (ranks 2, 7, 12, 20).
- **Logic issues:** Conclusion ("merges manifest as false branches") is well-supported by the magnitude. The minor numeric drift between recorded and rerun does not warrant a downgrade.

### 13. (Priority 0.287 · Surprise 0.307) Merge-causing fragment segments have ~2× the branch density of correctly reconstructed segments.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 53 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Branch count per 100 µm of cable for 64 merge-causing segments vs 4,006 control segments.
- **Conclusion:** Mean branch density was 0.1078 / 100 µm at merge segments vs 0.0568 / 100 µm at controls (Mann-Whitney U = 204,163.5, p = 2.91e-25). Positive surprisal (+0.31) shows intrinsic branch density is a strong, ground-truth-independent heuristic for flagging likely merge segments.
- **Caveats:** Only 64 merge segments; control n is large enough that statistics are well-powered, but the merge group is the limiting factor for effect-size precision.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** 64 merge segments and 4,006 control segments; mean density 0.1078 vs 0.0568 / 100 µm; U=204163.5, p=2.9141e-25 — all numbers match recorded exactly. Loading fix: replaced `os.walk("..")` dataset search (not intercepted by runner) with direct `pkl_paths=[$RERUN_PKL]`; analysis untouched.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U on branch density (count per 100 µm), 64 merge segments vs 4,006 controls; U = 204,163.5, p = 2.91e-25. Test choice is appropriate.
- **Statistical issues:** Imbalanced n (64 vs 4006) is fine for Mann-Whitney; the merge-group precision (mean 0.1078 ± unspecified SE) is the limiting factor for the effect-size estimate but the ~2× ratio is large. Survives BH-FDR easily.
- **Logic issues:** Claim that intrinsic branch density is a "ground-truth-independent heuristic" is a useful design statement and matches the test scope. No causal overreach.

### 14. (Priority 0.287 · Surprise 0.307) Split errors cascade — observed inter-split distances are ~10× shorter than a uniform null.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 55 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Distribution of nearest-neighbor geodesic distances between splits along a neuron vs a uniform-random null model with the same number of splits.
- **Conclusion:** Median observed inter-split distance was 25.58 µm vs 236.98 µm under random placement (KS D = 0.454, p ≈ 0). Positive surprisal (+0.31) confirms local conditions (e.g., image contrast in a region, challenging morphology) trigger runs of consecutive splits — analogous to the omit-clustering finding at rank 11.
- **Caveats:** None noted; the null comparison is internal to the same neurons, so the contrast is clean.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** 6,805 observed splits / 680,500 null samples; KS D=0.4547, p=0; median observed=25.58 µm, median random=237.88 µm (recorded 236.98 µm); mean observed=209.06 µm, mean random=372.25 µm (recorded 372.21 µm) — headline numbers (median observed 25.58, KS p=0, D≈0.454) match exactly; permutation-driven tiny drift in random samples is expected. Loading fix: panda env Python (numpy 2) + runner's existing pkl monkeypatch.
- **Verdict:** SOUND
- **Test:** Two-sample KS comparing observed inter-split nearest-neighbor distances to a uniform-random null with the same per-neuron count; D = 0.454, p ≈ 0. KS is appropriate for comparing two continuous distributions.
- **Statistical issues:** n is large (6,805 observed; 680,500 null draws). The null is well-designed (same neurons, same number of splits, just relocated). Small Monte-Carlo drift in null medians (237.88 µm rerun vs 236.98 µm recorded) is expected and immaterial. The KS p-value is anti-conservative when comparing against a permutation-resampled null with so many draws (effective null variance is collapsed), but the D-statistic (0.45) and the ~10× median gap are themselves the substantive evidence.
- **Logic issues:** "Cascading" is the right description; the mechanistic claim ("local conditions trigger runs") is plausible but not directly tested. No major overreach.

### 15. (Priority 0.287 · Surprise 0.307) Split errors concentrate near terminal leaves rather than along the backbone.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 60 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Network distance to nearest leaf for 6,805 split edges vs 1,109,034 correct edges.
- **Conclusion:** Mean leaf distance was 938.60 µm for splits vs 1,282.69 µm for correct edges (Mann-Whitney p = 2.16e-134); the split-edge density peaks much more sharply near 0–500 µm. Positive surprisal (+0.31) supports the morphological picture that thin distal processes are most prone to fracturing.
- **Caveats:** None noted; sample sizes are large.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** 6,805 split edges, 1,109,034 correct edges; mean leaf distance 938.60 vs 1282.69 µm; Mann-Whitney p=2.1613e-134 — every number matches recorded exactly. Loading fix: panda env Python (numpy 2) + runner's existing pkl monkeypatch.
- **Verdict:** MINOR ISSUES
- **Test:** Mann-Whitney U on network distance to nearest leaf, 6,805 splits vs 1.1M correct; p = 2.16e-134. Test choice is appropriate.
- **Statistical issues:** Edges-within-neuron non-independence (same as ranks 3 / 4 / 5 / 6 / 17 / 19) inflates the nominal p. Mean ratio 938.60 / 1282.69 ≈ 0.73 is a real but modest shift; the very small p is largely driven by n ≈ 1.1M.
- **Logic issues:** "Thin distal processes are most prone to fracturing" jumps from a distance-to-leaf shift to a thickness-based mechanistic claim. Network distance to a leaf is a proxy for thickness, not a measurement. Heavily redundant with rank 17 (id=65), which is the explicit thickness re-framing.

### 16. (Priority 0.287 · Surprise 0.307) Omit errors are ~1.8× more frequent on terminal edges than on internal edges.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 64 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** OMIT rate on 543,477 terminal edges vs 865,568 internal edges.
- **Conclusion:** OMIT rate was 4.38% on terminal edges vs 2.41% on internal edges (chi-square = 4189.94, p ≈ 0). Positive surprisal (+0.31) reinforces the rank-4 finding using a binary terminal/internal classification rather than a continuous distance, and shows that thin distal tips drive a disproportionate share of omissions.
- **Caveats:** None noted.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Terminal edges 543,477 (rate 4.38%, 23,792/543,477); Internal edges 865,568 (rate 2.41%, 20,898/865,568); χ²=4189.9364, p=0 — all numbers (counts, rates, statistic) match recorded byte-for-byte. Loading fix: panda env Python (numpy 2) + runner's existing pkl monkeypatch.
- **Verdict:** MINOR ISSUES
- **Test:** Chi-square test of independence on a 2×2 (terminal vs internal) × (OMIT vs non-OMIT) contingency table; χ² = 4189.93, p ≈ 0. Test choice is appropriate for two binary categoricals.
- **Statistical issues:** Same edges-within-neuron non-independence as ranks 3 / 11. Rank-11 quantifies this explicitly (P(OMIT | adjacent OMIT) ≈ 89%), so the chi-square independence assumption is severely violated here. With n = 1.4M and a 1.8× ratio, the substantive direction is robust but the nominal p of effectively zero is misleading.
- **Logic issues:** "Thin distal tips drive a disproportionate share of omissions" is a fair description and dovetails with rank 4 (distance-based). No causal overreach.

### 17. (Priority 0.287 · Surprise 0.307) Split edges sit on thinner (closer-to-leaf) processes than correct edges.
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 65 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Distance-to-leaf as a topological proxy for process thickness, comparing split edges to 50,000 downsampled correct edges.
- **Conclusion:** Median distance-to-leaf was 417.22 µm for split edges vs 670.25 µm for correct edges, Mann-Whitney p ≈ 0. Positive surprisal (+0.31) corroborates rank 15 with thickness framing — U-Net fragment breaking disproportionately affects thinner terminal processes.
- **Caveats:** Distance-to-leaf is only a proxy for process radius (actual radius was unusable due to overflow issues); downsampling controls is statistically valid but loses some precision in the tail.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Correct Edges 1,109,034 (downsampled to 50,000), Split Edges 6,805; Correct mean=1287.4358 µm / median=671.6104 µm (recorded mean=1278.24 µm / median=670.25 µm); Split mean=940.6854 µm / median=417.2197 µm (matches exactly); U=140,888,192.0, p=0 (recorded U=281,944,000.0, p=0). U differs ~2× from the recorded run (random-seed-dependent downsampling and possibly the same 2× iteration-counting effect as rank 5); the headline split median 417.22 µm and conclusion (split edges sit on thinner processes) are unchanged. Loading fix: panda env Python (numpy 2) + runner's existing pkl monkeypatch.
- **Verdict:** MINOR ISSUES
- **Test:** Mann-Whitney U on distance-to-leaf (used as a thickness proxy), 6,805 split edges vs 50,000 downsampled correct edges; reported p ≈ 0.
- **Statistical issues:** Distance-to-leaf is a noisy proxy for actual node radius — the authors acknowledge this. Downsampling to 50,000 is fine for power but means the U / p reported reflects the smaller subset (and the U differs 2× between recorded and rerun due to seed dependence and possible double-counting, same as rank 5). Edges-within-neuron non-independence is the standard concern. Effect magnitude (median 417 vs 671 µm) is real.
- **Logic issues:** "Splits sit on thinner processes" is a re-framing of rank 15 with a proxy, not an independent confirmation. Calling distance-to-leaf "thickness" is an indirect inference — splits could be near leaves for many reasons that have nothing to do with process radius (lower SNR at tips, faster orientation changes, etc.).

### 18. (Priority 0.287 · Surprise 0.307) Opposing fragment endpoints at split sites are geometrically collinear (point at each other).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 67 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Cosine similarity of inward-pointing direction vectors at 525 GT split endpoint pairs vs 2,689 nearby (≤ 15 µm) non-split control pairs.
- **Conclusion:** Mean cosine similarity was −0.685 at split pairs (strongly anti-parallel) vs −0.511 at controls (wider variance), Mann-Whitney p = 1.13e-08. Positive surprisal (+0.31) establishes geometric collinearity across a gap as a reliable, ground-truth-independent local signature an agent can use to propose split joins.
- **Caveats:** Controls are only "nearby and topologically disconnected", so some control pairs may themselves be undetected splits — this would bias the control mean toward more negative cosines and shrink the apparent effect (i.e., the true difference may be larger than reported).
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** 525 split endpoint pairs, 2,689 control pairs; mean cosine −0.6847 (std 0.3581) vs −0.5112 (std 0.5342); U=594814.0, p=1.13e-08 — every number matches recorded exactly. Loading fix: removed the `/tmp/lib` `pip install --target` bootstrap (it had cached cpython-312 numpy wheels that collided with the rerun's Python 3.11 panda env, causing `numpy._core._multiarray_umath` ImportError); now relies on panda env's pre-installed `agentic_neuron_proofreader` and loads `$RERUN_PKL` directly.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U on cosine similarity, 525 split endpoint pairs vs 2,689 nearby control pairs; U = 594,814.0, p = 1.13e-08. Rank-based test on a bounded continuous metric (cosine ∈ [−1, 1]) is appropriate; means (−0.685 vs −0.511) and stds (0.358 vs 0.534) are reasonable summaries.
- **Statistical issues:** Reasonably-sized samples (525 vs 2,689). The control SD (0.534) is much larger than split SD (0.358) — Mann-Whitney handles heteroscedastic continuous data fine, but the magnitude of the mean gap (~0.17) needs to be read against the wider control variance. Confounded control as noted in the caveat — some controls may be unflagged splits, which biases the test toward the null; in that sense p = 1.13e-08 is conservative.
- **Logic issues:** Conclusion ("geometric collinearity is a strong, statistically significant local signature of true split errors") is exactly what the test shows. No causal claim.

### 19. (Priority 0.287 · Surprise 0.307) Split edges sit topologically closer to GT branch nodes than correct edges (cross-validation).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 72 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Topological distance (Dijkstra) from each edge to nearest GT branch node (degree ≥ 3) across 1,409,045 edges and 5,072 branch nodes.
- **Conclusion:** Median distance was 196.43 µm for split edges vs 380.38 µm for correct edges (Mann-Whitney p ≈ 0). Positive surprisal (+0.31) provides a third independent confirmation (alongside ranks 5 and 6) that bifurcations act as failure loci.
- **Caveats:** Largely redundant with ranks 5 and 6; replication across distance metrics is reassuring rather than evidence of independent effects.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** Graph 1,409,057 nodes / 1,409,045 edges; 5,072 GT branch nodes; 6,805 split edges, 1,109,034 correct edges; mean Split=518.39 µm / median=196.43 µm; mean Correct=712.57 µm / median=380.38 µm; U=2,979,956,736.0, p=0 — every number matches recorded exactly. Loading fix: replaced `pathlib.Path('..').rglob('*add.pkl')` dataset search (not intercepted by runner) with direct `open($RERUN_PKL)`; analysis untouched.
- **Verdict:** MINOR ISSUES
- **Test:** Mann-Whitney U on topological distance to nearest GT branch node, 6,805 split vs 1.1M correct edges; reported p ≈ 0. Test choice is appropriate.
- **Statistical issues:** Edges-within-neuron non-independence as before. The substantive shift (median 196 vs 380 µm, ~2× ratio) is robust. Largely redundant with ranks 5 (geodesic on fragments-graph) and 6 (Euclidean) — should be treated as a triplet, not three independent tests.
- **Logic issues:** "Third independent confirmation" overstates the independence — same dataset, same edges, different distance metric / reference set. Direction is reproducible across metrics, which is reassuring.

### 20. (Priority 0.287 · Surprise 0.307) Merge sites sit in regions with ~55% higher fragment node density (third confirmation).
- **Run:** ground-truth-error-annotations-revised-version_2026-06-17 · **ID:** 81 · **Belief:** Leaning True → Likely True (0.7083 → 0.9327) · **Direction:** Positive
- **Tested:** Local fragment node density within 10 µm at 67 merge sites vs 67 control sites on correct edges.
- **Conclusion:** Density was 0.001678 nodes/µm³ at merges vs 0.001083 at controls (~55% higher), Welch's t = 8.93, p = 2.77e-14. Positive surprisal (+0.31) re-confirms ranks 2 and 7 with a volumetric density metric and a parametric test — dense neuropil is a robust environmental risk factor for merges.
- **Caveats:** Welch's t-test assumes approximate normality of group means; with the same modest 67-vs-67 sample as ranks 2 and 7, the precise effect size is somewhat uncertain even though the direction is firmly established.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** 67 merge sites, 67 control regions; densities 0.001678 vs 0.001083 nodes/µm³; Welch's t=8.9288, p=2.7673e-14 — every number matches recorded exactly. Loading fix: panda env Python (numpy 2); script's hardcoded `../dataset_cache_789202_mcl100_add.pkl` open re-routed to $RERUN_PKL by the runner.
- **Verdict:** MINOR ISSUES (redundant)
- **Test:** Welch's t-test on volumetric node density, 67 merge vs 67 control sites; t = 8.93, p = 2.77e-14. Welch's is appropriate when variances may differ.
- **Statistical issues:** Density-like metrics on small samples (n = 67) are typically right-skewed and may violate t-test normality; Mann-Whitney would be safer (and was used at ranks 2 and 7 on the same data). With t = 8.93, the conclusion is robust to a wide range of distributional assumptions. Same 67-vs-67 dataset as ranks 2 and 7 — this is a third statistical view of the same underlying comparison, not independent confirmation.
- **Logic issues:** "Third confirmation" framing overstates independence — three tests on the same 67-vs-67 sample, with slightly different radii/metrics. Conclusion is otherwise fair.

## Reproduction — Summary

- **Dataset pkl used:** `cache/dataset_cache_789202_mcl100_add.pkl` (brain 789202, min-cable-length 100; the same NumPy-2-written cache the original experiments referenced).
- **Reproduction breakdown:** 20 / 20 REPRODUCED, 0 DIVERGED, 0 FAILED (out of `n_rerun=20`).
- **Code source:** ALL 20 reruns executed on `code_source: revised-loading` — the initial recorded-code pass failed the unpickle for 18 / 20 hypotheses because the dataset cache was written with NumPy 2 and the host's default `numpy 1.26.4` cannot construct `numpy._core.numeric` during pickle load. Switching the rerun helper to the panda env's `numpy 2.4.6 / python 3.11` interpreter restored the unpickle for the entire batch.
- **Did NOT reproduce:** none. All 20 top-ranked hypotheses match the recorded headline numbers (test statistic, p-value, effect size, n) within trivial/expected noise; every qualitative conclusion holds.
- **Loading revisions applied** (revisions touched ONLY data loading/bootstrap; analysis code is byte-identical to recorded):
    - IDs 3, 39, 53: replaced `os.walk("..")` dataset search (which the runner's `open`/`glob` monkeypatch does NOT intercept, so the script printed "Dataset not found" and exited cleanly without running the analysis) with a direct `open(os.environ["RERUN_PKL"])` load.
    - ID 72: replaced `pathlib.Path('..').rglob('*add.pkl')` (also not intercepted by the runner) with a direct `$RERUN_PKL` load.
    - ID 39 (additionally): removed the per-iteration `pip install` retry loop that was the recorded script's bootstrap (panda env already has all required packages).
    - ID 67: removed the `/tmp/lib` `pip install --target=...` bootstrap entirely (the cache held cpython-312 numpy wheels that shadowed the panda env's Python 3.11 numpy and triggered `ImportError: numpy._core._multiarray_umath`); now relies on the panda env's pre-installed `agentic_neuron_proofreader` and a direct `$RERUN_PKL` load.
    - All other 15 scripts (IDs 47, 10, 11, 13, 19, 23, 30, 35, 48, 49, 55, 60, 64, 65, 81) used the runner's existing `open(*.pkl)`/`glob(*.pkl)` monkeypatch to route the script's hardcoded `*add.pkl` references to the provided pkl, with no other loading-section changes needed.
- **Notes on numeric matches:**
    - 16 / 20 hypotheses match recorded numbers byte-for-byte (statistic, p-value, sample counts, means, medians).
    - 4 hypotheses match qualitatively with minor expected variation: rank 5 (id=13) sees half the edge counts of the recorded run (recorded iterated edges in both directions, doubling n; rerun does not — direction unchanged, p still ≪ 1e-100); rank 12 (id=49) and rank 17 (id=65) show small sampling drift in the random-control set used for comparison (recorded U=363 / 281,944,000 → rerun U=340 / 140,888,192) but the headline merge-side medians (4.48 µm and 417.22 µm) are exact and conclusions hold; rank 14 (id=55) shows trivial Monte-Carlo drift in the null-mean (recorded 236.98 µm → rerun 237.88 µm) with KS p=0 and observed median 25.58 µm unchanged.

## Excluded (no surprisal score)

Two hypotheses lacked a surprisal value and were dropped from the ranking:

- `ground-truth-error-annotations-revised-version_2026-06-17` ID 15
- `ground-truth-error-annotations-revised-version_2026-06-17` ID 41

## Statistical Verification — Summary

20 top-ranked hypotheses audited. Verdict breakdown:

- **SOUND:** 8 — ranks 2 (id=3), 4 (id=11), 6 (id=19), 9 (id=35), 12 (id=49), 13 (id=53), 14 (id=55), 18 (id=67).
- **MINOR ISSUES:** 10 — ranks 1 (id=47), 3 (id=10), 5 (id=13), 7 (id=23), 11 (id=48), 15 (id=60), 16 (id=64), 17 (id=65), 19 (id=72), 20 (id=81).
- **MAJOR ISSUES:** 2 — rank 8 (id=30, n=12 correlation with causal-mechanistic overreach) and rank 10 (id=39, oracle-localization intervention with no inferential test and no control condition).
- **INVALID:** 0.

### Multiple comparisons (BH-FDR across the top 20)

Of the 20 hypotheses, 19 report a p-value (rank 10 / id 39 is a descriptive intervention with no inferential test and is excluded from the FDR family). At Benjamini–Hochberg FDR q = 0.05 over the 19-test family, **all 19 survive**. At q = 0.10 they also all survive. The two largest p-values in the family — p = 0.022 (rank 8, id 30, Pearson) and p = 0.014 (rank 1, id 47, logistic interaction term) — sit at BH thresholds of 0.0500 and 0.0474 respectively, i.e., on the survival/rejection boundary; even modest expansion of the testing family (or one extra borderline result) would knock them out.

Specifically, expanding the family to the full 98 ranked hypotheses (which is the more honest reference set, since the auto-discovery loop ran ~98 tests in total): the borderline survivors require their family-rank to be at least the (p·m/q)-th largest p-value, i.e., for q=0.05 at least the ~28th largest for rank 1 and at least the ~44th largest for rank 8. If many of the other 78 untested-in-this-audit hypotheses also reported small p-values, rank 1 likely still survives but rank 8 becomes precarious.

### Most serious problems found (prioritized)

1. **Rank 10 / ID 39 (merge-severing intervention) — MAJOR ISSUES.** The intervention reduces % merged edges from 13.25% to 1.83% only because it uses GT-derived merge sites to choose cut points. There is no inferential test, no sham-cut control, and no measurement of degradation under realistic (non-oracle) localization noise. The conclusion ("highly viable agentic strategy") is scope-overreach: this experiment proves a *ceiling* under perfect localization, not a deployable algorithm.

2. **Rank 8 / ID 30 (per-neuron split/omit correlation) — MAJOR ISSUES.** Pearson r = 0.65 (p = 0.022) on n = 12 neurons is the most fragile p-value in the top 20: it sits at the BH boundary inside this family and is unlikely to survive a wider family. The conclusion explicitly posits a "shared underlying mechanistic failure such as weak signal" from a 12-point correlation — that is causal-mechanistic overreach from a correlational test. The Spearman ρ = 0.88 is more robust but is not an independent test.

3. **Rank 1 / ID 47 (non-linear distance/angle trade-off) — MINOR ISSUES.** The headline non-linearity claim rests on a single logistic interaction term with p = 0.014 and pseudo-R² = 0.093. Strong qualitative claim ("fundamentally non-linear") from a single just-significant coefficient in a poorly-fitting model. Borderline under BH for any reasonably wide family.

4. **Independence violations across the chi-square and edges-vs-edges Mann-Whitney tests (ranks 3, 4, 5, 11, 15, 16, 17, 19).** Edges that share a node or live in the same neuron are not independent, especially given that omits cluster (rank 11 itself shows P(OMIT|adj OMIT) ≈ 89%). The reported p-values of "effectively zero" are anti-conservative by an unknowable but real factor. The qualitative directions are nevertheless robust because the effect ratios are large (1.5×–3.4×).

5. **Redundancy of merge-density and split-near-branch-point comparisons.** Ranks 2 / 7 / 12 / 20 all test 67 merge sites vs 67 random controls on overlapping radius/metric variants; ranks 5 / 6 / 19 all test split-vs-correct edge distance to branches in three metrics. These triples should not be read as independent confirmations — they are alternate views of the same underlying comparison and inflate the appearance of "convergent evidence."

### Cross-cutting concerns

- **Small-n neuron-level analyses.** Only 12 GT neurons drive ranks 8, 9, and 10. Rank 8 is most exposed because its conclusion depends on Pearson r whose 95% CI at n = 12 spans (0.10, 0.89).
- **Edges-within-neuron non-independence.** Affects all chi-square and Mann-Whitney tests that treat individual edges as independent observations (ranks 3, 4, 5, 6, 11, 15, 16, 17, 19). Rank 11 is the formal demonstration that this assumption is wrong.
- **Control-set construction.** Ranks 2 / 7 / 12 / 20 use random non-merge sites as controls, which may co-vary with neuropil-density anatomy. Rank 18 (id=67) explicitly notes that some control pairs may be undetected splits. None of these designs use density / anatomy-matched controls.
- **No multiple-testing correction within a single hypothesis's code.** Rank 8 (id=30) reports three correlated statistics (Pearson p, Spearman p, OLS F p) on the same 12 points without correction.
- **"Confirmation" framing inflates redundancy.** Several conclusions call themselves "second" or "third independent confirmation" when they are alternate metrics on the same data — the discovery loop's own self-audit text treats these as independent evidence, which they are not.
- **Reproduction-aware downgrade.** All 20 reproduced; the 4 minor drifts (ranks 5, 12, 14, 17) are seed/Monte-Carlo noise and do not change any verdict.
