# AutoDiscovery Combined Ranked Summary

**Source files (pooled, sorted by surprise magnitude descending):**
- `autodiscovery/exa-spim-run-1_2026-06-03.json` -- 50 hypotheses
- `autodiscovery/exa-spim-run-2_2026-06-03.json` -- 200 hypotheses
- `autodiscovery/refinement-of-exa-spim-run-1_2026-06-03.json` -- 99 hypotheses

**Total hypotheses ranked:** 349 (of 349 ingested; 0 dropped for missing surprisal). **Surprise-magnitude range:** 0.0000 to 0.9661.

**Headline synthesis.** Across the three runs the strongest belief shifts were uniformly negative: the discovery loop repeatedly proposed plausible structural heuristics for proofreading (greedy geometric matching traps, inter-annotator bias, thickness/branching biomarkers of merges, Z-axis split alignment, shatter-zone reconnection failures) and the experiments returned null or opposite-signed results. The headline reversals were run `refinement-of-exa-spim-run-1_2026-06-03` ID 49 (Likely True -> Leaning False, surprisal -0.966); run `refinement-of-exa-spim-run-1_2026-06-03` ID 25 (Likely True -> Leaning False, surprisal -0.922); run `exa-spim-run-2_2026-06-03` ID 111 (Likely True -> Leaning False, surprisal -0.852). Cross-cutting theme: simple geometric/length features that look intuitive on paper (spatial proximity, neurite radius, tortuosity, depth, fragment length, branching density) routinely fail as standalone discriminators of U-Net split/merge errors, and several apparently 'significant' results are driven by tiny minority classes (often fewer than ten false-merge cases), so confident operational rules should not be derived from them without further validation.

---

## Ranked hypotheses (most surprising first)

### 1. (Surprise 0.966) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 49 · **Belief:** Likely True -> Leaning False (0.9167 -> 0.2115) · **Direction:** Negative
- **Tested:** Analyze the distribution of geometric affinity scores among candidate fragment pairs to determine if high-confidence false merges exist, rigorously testing the assumption that a greedy algorithm will inevitably fall into early traps without relying on flawed hardcoded assumptions.
- **Conclusion:** The experiment successfully executed the revised plan and rigorously analyzed the geometric affinity scores of candidate fragment pairs. Belief therefore dropped from Likely True (0.92) to Leaning False (0.21) (signed surprisal -0.966; negative shift).
- **Caveats:** zero-variance data forced a degenerate test
- **Verdict:** Unsound
- **Test:** Student's t-test; test statistic NaN / undefined
- **Statistical issues:** test was degenerate or uninformative (NaN, zero variance, empty cohort, or constant inputs)
- **Logic issues:** large belief shift triggered by a test that produced no usable evidence about the hypothesis

### 2. (Surprise 0.922) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 25 · **Belief:** Likely True -> Leaning False (0.8750 -> 0.2019) · **Direction:** Negative
- **Tested:** Quantify whether the automated reconstruction's performance metrics vary significantly depending on which human annotator traced the ground truth neuron.
- **Conclusion:** The results yielded p-values well above the 0.05 threshold for both Omission Rates (ANOVA p=0.664, Kruskal p=0.450) and Split Rates (ANOVA p=0.946, Kruskal p=0.912). Belief therefore dropped from Likely True (0.88) to Leaning False (0.20) (signed surprisal -0.922; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis, one-way ANOVA; p = 0.912
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 3. (Surprise 0.852) Refuted: data contradict the claim that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 111 · **Belief:** Likely True -> Leaning False (0.8750 -> 0.3281) · **Direction:** Negative
- **Tested:** Investigate the relationship between the morphological thickness of a fragment and its propensity to be fragmented by the automated segmentation pipeline.
- **Conclusion:** **Experiment Results:** - **Median Radius:** 1.9537 µm - **Thin Fragments (< median, n=5086):** Mean split rate of 0.1036 splits/mm. - **Thick Fragments (>= median, n=5086):** Mean split rate of 0.1430 splits/mm. Belief therefore dropped from Likely True (0.88) to Leaning False (0.33) (signed surprisal -0.852; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies; median outcome is zero; effect driven by outliers
- **Verdict:** Questionable
- **Test:** Mann-Whitney U, Student's t-test; p ≈ 2e-33; n up to 5086
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 4. (Surprise 0.834) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 53 · **Belief:** Leaning True -> Likely False (0.7917 -> 0.1827) · **Direction:** Negative
- **Tested:** Quantify inter-annotator variability in the resulting baseline segmentation metrics to measure the subjective ceiling of the proofreading task.
- **Conclusion:** Statistical analysis using the Kruskal-Wallis test yielded high p-values for all metrics (edge accuracy: p=0.4775, omit rate: p=0.4775, splits per 1000 µm: p=0.8565), which are well above the standard 0.05 significance threshold. Belief therefore dropped from Leaning True (0.79) to Likely False (0.18) (signed surprisal -0.834; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis; p = 0.8565
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 5. (Surprise 0.834) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 62 · **Belief:** Leaning True -> Likely False (0.7917 -> 0.1827) · **Direction:** Negative
- **Tested:** Quantify how the local density of fragment endpoints (crowding) degrades the reliability of standard proximity thresholds.
- **Conclusion:** The pairs were binned into balanced Low (n=118), Medium (n=118), and High (n=119) crowding tiers. Statistical analysis, including a Cochran-Armitage trend test (p-value = 0.481) and a Chi-Square test (p-value = 0.368), confirmed there is no significant degradation in precision as local density increases. Belief therefore dropped from Leaning True (0.79) to Likely False (0.18) (signed surprisal -0.834; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** chi-square, Cochran-Armitage, Student's t-test; p = 0.368; n up to 119
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 6. (Surprise 0.812) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 45 · **Belief:** Likely True -> Leaning False (0.8333 -> 0.3125) · **Direction:** Negative
- **Tested:** To show that merge errors are heavily concentrated in a few massive connected components, implying that targeting the largest fragments for merge-detection will yield the highest return on investment.
- **Conclusion:** The binomial test yielded a p-value of 1.00, failing to reject the null hypothesis. Belief therefore dropped from Likely True (0.83) to Leaning False (0.31) (signed surprisal -0.812; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** binomial test; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 7. (Surprise 0.782) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 76 · **Belief:** Leaning True -> Leaning False (0.7917 -> 0.2212) · **Direction:** Negative
- **Tested:** Investigate if component-level branching density is a reliable 'stale feature' that agents can monitor dynamically to detect when a newly executed merge edit has created an invalid web.
- **Conclusion:** - **Branching Density (Mean ± Std):** Merge-containing components demonstrated a mean branching density of 0.001408 branches/µm (std: 0.000819), while pure components had a mean branching density of 0.001114 branches/µm (std: 0.001081). - **Statistical Comparison:** A Welch's t-test yielded a t-statistic of 1.2902 and a p-value of 0.2191. Belief therefore dropped from Leaning True (0.79) to Leaning False (0.22) (signed surprisal -0.782; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Welch's t-test, two-sample t-test, Student's t-test; p = 0.05; n up to 14
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 8. (Surprise 0.771) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 36 · **Belief:** Leaning True -> Leaning False (0.7917 -> 0.2969) · **Direction:** Negative
- **Tested:** To determine whether U-Net reconstruction error metrics are inherently biased by the specific human annotator who traced the ground-truth morphology.
- **Conclusion:** Eight unique annotators were identified (JG, HP, IG, SP, JT, PP, MB, YZ), with mean split rates ranging from 0.1437 to 0.2264 splits/mm. A one-way ANOVA test applied to these groups returned an F-statistic of 0.5538 and a p-value of 0.7782. Belief therefore dropped from Leaning True (0.79) to Leaning False (0.30) (signed surprisal -0.771; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** one-way ANOVA, Student's t-test; p = 0.7782
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 9. (Surprise 0.771) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 37 · **Belief:** Leaning True -> Leaning False (0.7917 -> 0.2969) · **Direction:** Negative
- **Tested:** To evaluate the benefit of reasoning jointly over competing proposals using a combined spatial and angular cost function instead of independent thresholding.
- **Conclusion:** Since there were no discordant pairs between the two methods, the McNemar's test resulted in a p-value of 1.0. Belief therefore dropped from Leaning True (0.79) to Leaning False (0.30) (signed surprisal -0.771; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** chi-square, McNemar's; p = 1
- **Statistical issues:** test was degenerate or uninformative (NaN, zero variance, empty cohort, or constant inputs); McNemar p=1.0 reflects zero discordant predictions between methods — uninformative, not negative evidence
- **Logic issues:** large belief shift triggered by a test that produced no usable evidence about the hypothesis

### 10. (Surprise 0.771) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 42 · **Belief:** Leaning True -> Leaning False (0.7917 -> 0.2969) · **Direction:** Negative
- **Tested:** To determine if local radius continuity is a viable and robust feature for filtering out false merge proposals during split-correction, while avoiding data type overflow errors.
- **Conclusion:** The analysis found 1,612 True Splits (mean: ~0.43 µm) and only 6 False Merges (mean: ~0.39 µm). The Mann-Whitney U test yielded a statistic of 5170.5 and a p-value of 0.769, indicating no statistically significant difference between the absolute radius differences of the two groups. Belief therefore dropped from Leaning True (0.79) to Leaning False (0.30) (signed surprisal -0.771; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.769
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 11. (Surprise 0.747) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 2 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Determine if a local 'crowdedness' metric can serve as a contextual feature to dynamically adjust confidence thresholds for accepting topological edits.
- **Conclusion:** The local spatial 'crowdedness' (unique fragment components within a 30µm radius) was calculated, revealing a mean crowdedness of 1.17 (std = 0.52) for merge sites and 1.55 (std = 0.94) for split sites. An independent t-test demonstrated a statistically significant difference between the two distributions (t-statistic = -3.7813, p-value = 5.9873e-04). Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Questionable
- **Test:** two-sample t-test, Student's t-test; p ≈ 0.0006
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 12. (Surprise 0.747) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 3 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Determine if radius continuity is a statistically robust feature for distinguishing valid split reconnections from false reconnections, ensuring sufficient sample sizes by expanding the spatial search radius and resolving numerical overflow issues.
- **Conclusion:** The statistical comparison revealed a mean absolute radius difference of 0.4476 ± 0.3614 µm for valid reconnections and 0.4972 ± 0.3625 µm for false reconnections. The Mann-Whitney U test yielded a p-value of ~0.6017, indicating that there is no statistically significant difference between the two distributions. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.6017
- **Statistical issues:** none
- **Logic issues:** none

### 13. (Surprise 0.747) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 9 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Investigate whether localized neurite swelling (increased radius) in the U-Net reconstruction is a reliable morphological signature of merge errors.
- **Conclusion:** **Experiment Results:** - **Merge Nodes:** Mean radius = 1.8941 µm (std: 0.1514 µm) - **Background Nodes:** Mean radius = 1.9640 µm (std: 0.1017 µm) - **Statistical Comparison:** Welch's t-test yielded a t-statistic of -10.0014 and a p-value of 1.7181e-21, indicating a statistically significant difference between the two distributions. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p ≈ 1.7e-21
- **Statistical issues:** none
- **Logic issues:** none

### 14. (Surprise 0.747) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 10 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Explore the interaction between gap distance and neurite thickness at valid split sites to inform dynamic, radius-aware KD-tree search bounds for proposal generation.
- **Conclusion:** Statistical analysis revealed a Pearson correlation coefficient of r = 0.3702 (p-value = 1.99e-45) and a Spearman correlation of rho = 0.3716 (p-value = 8.88e-46). However, contrary to the initial hypothesis, the correlation is **positive**, not negative. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** stepped ROC curve from sparse negatives; dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Spearman, Pearson; p ≈ 8.9e-46
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 15. (Surprise 0.747) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 21 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Assess if highly branched neurons suffer from disproportionately higher merge error rates, which would suggest that proofreading systems should prioritize merge-detection passes on highly arborized fragment clusters.
- **Conclusion:** A Spearman rank correlation analysis yielded a rho of -0.6081 and a p-value of 0.0057, indicating a statistically significant negative correlation. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman; p = 0.0057
- **Statistical issues:** none
- **Logic issues:** none

### 16. (Surprise 0.747) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 23 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Determine if fragment cable length is inversely correlated with the likelihood of producing split endpoints.
- **Conclusion:** **Experiment Results:** - **Short Fragments (Bottom 25%, ≤ 225 nodes):** Mean split rate = 0.000506 splits/node. - **Long Fragments (Top 25%, ≥ 373 nodes):** Mean split rate = 0.001665 splits/node. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** median outcome is zero; effect driven by outliers
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 3.2e-77
- **Statistical issues:** none
- **Logic issues:** none

### 17. (Surprise 0.747) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 31 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** To determine if the physical gap distance of recoverable split errors is systematically larger for thinner neurites, which would necessitate adaptive search radii based on local thickness.
- **Conclusion:** The Spearman rank correlation test yielded a rho of 0.1879 and a highly significant p-value of 3.0508e-05. This indicates a statistically significant positive correlation, which contradicts the initial hypothesis: thicker neurites, rather than thinner ones, tend to exhibit larger physical fragmentation gaps. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Spearman; p ≈ 3.1e-05
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 18. (Surprise 0.747) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 34 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Quantify the magnitude of 'feature staleness' by measuring how the statistical distribution of fragment cable lengths shifts after a single simulated layer of topological edits.
- **Conclusion:** While the mean component cable length increased slightly from 1,849.31 µm to 1,911.51 µm, a Kolmogorov-Smirnov test comparing the pre-edit ('stale') and post-edit ('refreshed') distributions yielded a low statistic (0.0096) and a high p-value (0.7465). Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov; p = 0.7465
- **Statistical issues:** none
- **Logic issues:** none

### 19. (Surprise 0.747) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 38 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Compare the precision and recall of a mutually exclusive bipartite matching algorithm versus an independent thresholding approach for split-error correction.
- **Conclusion:** The experiment was successfully executed and all deliverables were generated. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 20. (Surprise 0.747) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 39 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Quantify whether local fragment leaf density is a reliable environmental predictor for the likelihood of a branch point being a false merge.
- **Conclusion:** The statistical analysis (Mann-Whitney U test) yielded a p-value of 0.4009, indicating no statistically significant difference in local proposal density between True branches (mean = 0.45) and Merge branches (mean = 0.44). Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.4009
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 21. (Surprise 0.747) Refuted: data contradict the claim that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 43 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Investigate if there is a depth-dependent degradation in segmentation quality by correlating Z-axis depth with the severity (gap size) of split errors.
- **Conclusion:** Statistical analysis showed a Pearson correlation coefficient of r = -0.0221 and a p-value of 0.5927, indicating that there is no statistically significant linear relationship between split gap distance and tissue depth. The data visualization (scatter plot with linear fit) clearly confirmed this lack of correlation. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** one-way ANOVA, Pearson; p ≈ 0.00086
- **Statistical issues:** statistical significance driven by very large N; effect size negligible (|r|<0.1) — significance is a large-sample artifact rather than a scientifically meaningful effect
- **Logic issues:** none

### 22. (Surprise 0.747) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 50 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Analyze the size distribution asymmetry of sub-components involved in U-Net merge errors.
- **Conclusion:** The results showed a median ratio of 0.4153 and a mean ratio of 0.4928. Based on the predefined logic (median > 0.2), the script successfully generated a data-driven conclusion rejecting the hypothesis: merge errors do not exclusively manifest as tiny orphan fragments attaching to large main branches, but frequently involve the fusion of relatively comparable axonal trunks. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 23. (Surprise 0.747) Refuted: data contradict the claim that image-intensity features (signal across the gap) add discriminative power.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 52 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Evaluate if a multimodal (geometry + image) joint-reasoning approach outperforms single-metric approaches in resolving conflicting split proposals.
- **Conclusion:** - **Performance Metrics:** - **Geometry (Tangent Alignment Only):** AUC = 0.8000 - **Image (Gap Intensity Only):** AUC = 0.4971 (effectively random guessing) - **Multimodal Joint-Reasoning:** AUC = 0.8000 - **Insights:** The geometric feature (tangent alignment) proved to be a strong predictor for correct topological continuations, successfully identifying ~73% of true matches with zero false positives. 2 False Matches), resulting in a heavily stepped ROC curve. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); stepped ROC curve from sparse negatives
- **Verdict:** Sound
- **Test:** logistic regression, ROC-AUC
- **Statistical issues:** none
- **Logic issues:** none

### 24. (Surprise 0.747) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 57 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Assess whether the broader structural trend (PCA of local nodes) provides a more robust geometric disambiguation for splits than single-edge tangent vectors.
- **Conclusion:** The mean absolute cosine similarity (angular alignment) was nearly identical for True (0.6090) and False (0.6026) proposals. A Welch's t-test yielded a p-value of 0.9688, indicating no statistically significant difference in alignment between the two classes. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Welch's t-test, Student's t-test; p = 0.9688; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 25. (Surprise 0.747) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 58 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Evaluate if branching nodes in the `fragments_graph` that constitute merge errors have significantly larger neurite radii than branching nodes that are topologically correct.
- **Conclusion:** Statistical analysis via Welch's t-test yielded a significant p-value (p = 0.0203); however, the result contradicted the initial hypothesis. The mean radius for Merge Branches (1.8325 µm) was actually smaller than the mean radius for Valid Branches (1.9530 µm). Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p = 0.0203
- **Statistical issues:** none
- **Logic issues:** none

### 26. (Surprise 0.747) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 59 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Quantify and statistically test the variance in normalized split error density across different human annotators in the ground-truth dataset.
- **Conclusion:** The statistical evaluation using the Kruskal-Wallis H-test yielded an H-statistic of 5.0474 and a p-value of 0.6542. Since the p-value is well above the standard alpha level of 0.05, there is no statistically significant variance in normalized split error density across the different annotators. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis; p = 0.6542
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 27. (Surprise 0.747) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 60 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Quantify the correlation between human annotator identity and U-Net split error rates to identify if certain annotator tracing behaviors lead to higher automated fragmentation.
- **Conclusion:** The statistical analysis revealed mean split rates ranging from 0.14 frags/mm (Annotator JT) to 0.21 frags/mm (Annotators SP, PP, HP). Despite these visible variations, the Kruskal-Wallis H-test yielded an H-statistic of 8.5526 with a p-value of 0.2864. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis; p = 0.2864
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 28. (Surprise 0.747) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 63 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Evaluate the impact of a parent fragment's total cable length on the validity of its endpoint reconnection proposals.
- **Conclusion:** A logistic regression model was fitted, yielding a p-value of 0.842 for the length coefficient, which indicates no statistical significance. The ROC-AUC score for minimum parent fragment length as a predictor of validity was 0.5521, performing only marginally better than random chance. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); stepped ROC curve from sparse negatives
- **Verdict:** Sound
- **Test:** logistic regression, ROC-AUC; p = 0.842
- **Statistical issues:** none
- **Logic issues:** none

### 29. (Surprise 0.747) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 64 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Quantify annotator bias in terminal pruning by analyzing the distribution of predicted radii at ground-truth leaf nodes grouped by the annotator's initials.
- **Conclusion:** Because there was zero variance in the data, the Kruskal-Wallis H-test returned a NaN statistic and p-value. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** zero-variance data forced a degenerate test; Kruskal test returned NaN due to identical values across groups
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis, Student's t-test; test statistic NaN / undefined
- **Statistical issues:** test was degenerate or uninformative (NaN, zero variance, empty cohort, or constant inputs)
- **Logic issues:** large belief shift triggered by a test that produced no usable evidence about the hypothesis

### 30. (Surprise 0.747) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 69 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Quantify the relationship between U-Net fragment cable length and its propensity to harbor topological merge errors.
- **Conclusion:** The statistical analysis revealed a highly significant positive correlation (Point-biserial r = 0.1209, p = 2.06e-34) between fragment length and merge probability. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** point-biserial; p ≈ 2.1e-34
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 31. (Surprise 0.747) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 71 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Quantify the geometric trade-off between gap distance and tangent alignment in true split reconnections.
- **Conclusion:** The results contradicted the initial hypothesis: rather than true connections exhibiting tighter angular stringency at larger distances, the mean alignment for Long Gaps (0.4810) was significantly lower and more variable (variance = 0.4203) than for Short Gaps (mean = 0.8073, variance = 0.0780). Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p ≈ 1.1e-12
- **Statistical issues:** none
- **Logic issues:** none

### 32. (Surprise 0.747) Refuted: data contradict the claim that cycle-formation cleanly flags false-merge reconnection proposals.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 72 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Quantify the false-positive rate of cycle-forming versus cycle-free split reconnection proposals to justify strict topological constraints.
- **Conclusion:** The resulting contingency matrix and Chi-Square test (Statistic: 43.89, p-value: 3.48e-11) revealed a statistically significant relationship, but it contradicted the initial hypothesis: cycle-forming proposals actually exhibited a lower false-positive rate (56.7%) than cycle-free proposals (73.6%). Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p ≈ 3.5e-11
- **Statistical issues:** none
- **Logic issues:** none

### 33. (Surprise 0.747) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 79 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Evaluate the predictive power of neurite radius continuity across split gaps to disambiguate true reconnections from false candidate pairs.
- **Conclusion:** A Welch's t-test revealed no statistically significant difference between the mean radius difference of True pairs (0.4475 µm) and False pairs (0.3782 µm), yielding a p-value of 0.7609. The predictive capability of radius continuity was evaluated using an ROC curve, which resulted in an AUC score of 0.4573, indicating performance slightly worse than random chance. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Welch's t-test, ROC-AUC, Student's t-test; p = 0.7609; n up to 3
- **Statistical issues:** none
- **Logic issues:** none

### 34. (Surprise 0.747) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 81 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Evaluate if the absolute difference in predicted node radius between two candidate leaf nodes can independently distinguish True from False split reconnections.
- **Conclusion:** Statistical analysis via Welch's t-test yielded a non-significant p-value (p = 0.7609), showing that the mean absolute radius difference for True connections (0.4475 µm) is not significantly different from that of False connections (0.3782 µm). Additionally, the ROC curve yielded an AUC of 0.4573, indicating predictive performance slightly worse than random chance. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, ROC-AUC, Student's t-test; p = 0.7609
- **Statistical issues:** none
- **Logic issues:** none

### 35. (Surprise 0.747) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 87 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** To evaluate if localized neurite radius inflation can serve as an independent morphological feature for zero-shot merge detection.
- **Conclusion:** A Mann-Whitney U test yielded a highly significant p-value of 1.22e-112, but the mean radius results completely contradicted the hypothesis: Merge Nodes actually exhibited a *smaller* mean radius (1.8951 µm) compared to Continuous Nodes (1.9970 µm). Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 1.2e-112
- **Statistical issues:** none
- **Logic issues:** none

### 36. (Surprise 0.747) Refuted: data contradict the claim that image-intensity features (signal across the gap) add discriminative power.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 89 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Determine if false merge errors are significantly correlated with thicker localized neurite radii.
- **Conclusion:** The Welch's t-test yielded a p-value of 0.2920 (t-statistic = -1.0537), demonstrating no statistically significant difference in the local average radius between True Branches (mean = 1.9805 µm) and Merge Branches (mean = 1.9531 µm). Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Welch's t-test, Student's t-test; p = 0.292
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 37. (Surprise 0.747) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 90 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Investigate the geometric trade-off between gap distance and angular alignment for true splits to inform a dynamically scaling distance-angle threshold for reconnections.
- **Conclusion:** The statistical analysis yielded a significant negative Pearson correlation (r = -0.5148, p-value = 0.0) and Spearman correlation (rho = -0.5036), which directly contradicts the initial hypothesis that larger gaps inherently require stricter collinear trajectories. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Spearman, Pearson; p ≈ 0
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 38. (Surprise 0.747) Refuted: data contradict the claim that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 91 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Determine if structural proximity to the soma protects against fragmentation, which could allow proofreading agents to allocate more computational resources to distal arbors rather than proximal trunks.
- **Conclusion:** - **Split Density (Mean ± Std):** Proximal regions exhibited a split density of 1.34 ± 0.38 splits/mm, whereas distal regions had a density of 0.62 ± 0.16 splits/mm. - **Statistical Comparison:** A paired t-test across the 19 neurons yielded a t-statistic of 7.7717 and a highly significant p-value of 3.6925e-07. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** paired t-test, Student's t-test; p ≈ 3.7e-07
- **Statistical issues:** none
- **Logic issues:** none

### 39. (Surprise 0.747) Refuted: data contradict the claim that image-intensity features (signal across the gap) add discriminative power.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 95 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Test if anomalous swelling in the predicted neurite radius can serve as a local biomarker for merge errors.
- **Conclusion:** Statistical analysis using Welch's t-test revealed a highly significant difference between the two groups (t-statistic = -11.6718, p-value = 1.1709e-27). Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p ≈ 1.2e-27
- **Statistical issues:** none
- **Logic issues:** none

### 40. (Surprise 0.747) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 97 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Quantify the frequency at which topological reconnections transition a node from a simple leaf to a complex branching junction.
- **Conclusion:** A binomial test (p-value = 8.9070e-35) confirmed that the rate of connecting into branching nodes is significantly greater than zero. Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** binomial test; p ≈ 8.9e-35
- **Statistical issues:** none
- **Logic issues:** none

### 41. (Surprise 0.747) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 99 · **Belief:** Leaning True -> Likely False (0.7083 -> 0.1635) · **Direction:** Negative
- **Tested:** Determine if merge errors are correlated with an excess of fragmented endpoints, suggesting that actively identifying and resolving merges could unblock multiple valid split connections in a coupled proofreading state.
- **Conclusion:** - **Endpoint Density Metrics (Mean / Median):** Merged fragments displayed an endpoint density of 1.89e-03 / 1.70e-03 leaves per µm, while unmerged fragments showed very similar densities of 1.81e-03 / 1.56e-03 leaves per µm. - **Statistical Comparison:** A Mann-Whitney U test yielded a p-value of ~0.599 (Statistic = 4761.0). Belief therefore dropped from Leaning True (0.71) to Likely False (0.16) (signed surprisal -0.747; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.05
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 42. (Surprise 0.738) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 19 · **Belief:** Leaning True -> Leaning False (0.7500 -> 0.2115) · **Direction:** Negative
- **Tested:** Evaluate a multi-feature heuristic for resolving mutual exclusivity and competing proposals at endpoints by expanding the search radius to ensure a statistically robust sample size.
- **Conclusion:** The McNemar's test returned a p-value of 0.823, indicating that there is no statistically significant difference between the two heuristics. Belief therefore dropped from Leaning True (0.75) to Leaning False (0.21) (signed surprisal -0.738; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square, McNemar's, Student's t-test; p = 0.823
- **Statistical issues:** none
- **Logic issues:** none

### 43. (Surprise 0.730) Refuted: data contradict the claim that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 32 · **Belief:** Leaning True -> Leaning False (0.7500 -> 0.2812) · **Direction:** Negative
- **Tested:** To test if lower image resolution along the Z-axis disproportionately contributes to structural discontinuities (splits) in the automated reconstruction.
- **Conclusion:** The Kolmogorov-Smirnov test yielded a statistically significant difference (K-S stat = 0.0800, p-value = 2.7339e-10). The mean absolute cosine similarity for true gaps (0.4893) is slightly lower than that of random vectors (0.5005). Belief therefore dropped from Leaning True (0.75) to Leaning False (0.28) (signed surprisal -0.730; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov, Student's t-test; p ≈ 2.7e-10
- **Statistical issues:** none
- **Logic issues:** none

### 44. (Surprise 0.730) Refuted: data contradict the claim that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 27 · **Belief:** Leaning True -> Leaning False (0.7500 -> 0.2812) · **Direction:** Negative
- **Tested:** Determine if morphological context (terminal fine arbors vs. thick main trunks) dictates the likelihood of U-Net split errors, which would inform context-aware proofreading agents.
- **Conclusion:** The accompanying box plot demonstrated that not only is the median split rate higher in the main trunk, but the variance is also notably larger, indicating highly inconsistent automated reconstruction along the primary backbone. Belief therefore dropped from Leaning True (0.75) to Leaning False (0.28) (signed surprisal -0.730; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Wilcoxon; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 45. (Surprise 0.730) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 50 · **Belief:** Leaning True -> Leaning False (0.7500 -> 0.2812) · **Direction:** Negative
- **Tested:** Evaluate the false positive rate of cycle-forming proposals to validate cycle-prevention as a crucial graph-state constraint in proofreading.
- **Conclusion:** The Chi-square test (p-value = 0.77) indicated no statistically significant difference between the two, primarily because the false positive count was extremely low across the board. Belief therefore dropped from Leaning True (0.75) to Leaning False (0.28) (signed surprisal -0.730; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p = 0.77
- **Statistical issues:** none
- **Logic issues:** none

### 46. (Surprise 0.730) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 85 · **Belief:** Leaning True -> Leaning False (0.7500 -> 0.2812) · **Direction:** Negative
- **Tested:** Assess if local proposal density dynamically influences the reliability of reconnections, requiring the refresh of stale density features during iterative proofreading.
- **Conclusion:** The Chi-squared test showed no significant difference (chi2 = 0.0000, p-value = 1.0000), heavily influenced by the extreme class imbalance. Belief therefore dropped from Leaning True (0.75) to Leaning False (0.28) (signed surprisal -0.730; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** chi-square; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 47. (Surprise 0.730) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 190 · **Belief:** Leaning True -> Leaning False (0.7500 -> 0.2812) · **Direction:** Negative
- **Tested:** Evaluate whether the structural size (path length) of fragments involved in a proposal influences the likelihood of the reconnection being valid.
- **Conclusion:** The resulting logistic regression analysis yielded a coefficient for the minimum component length (per mm) of 0.002556, an odds ratio of 1.002559, and a p-value of 0.852. Since the p-value is far greater than the standard 0.05 significance threshold, the effect of fragment length on the validity of the split-correction proposal is not statistically significant. Belief therefore dropped from Leaning True (0.75) to Leaning False (0.28) (signed surprisal -0.730; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** logistic regression, Student's t-test; p = 0.852
- **Statistical issues:** none
- **Logic issues:** none

### 48. (Surprise 0.690) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 9 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Quantify the trade-off between endpoint distance and orientation agreement to determine if distant reconnections require stricter geometric alignment to be valid.
- **Conclusion:** The short-gap cohort exhibited significantly stronger anti-parallel alignment (mean collinearity = -0.8675) compared to the long-gap cohort (mean collinearity = -0.5192). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p ≈ 5.3e-07
- **Statistical issues:** none
- **Logic issues:** none

### 49. (Surprise 0.690) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 10 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Validate if structural radius asymmetry can be used as a purely geometric feature by a CNN to slide along skeletons and flag merge sites without relying on image patches.
- **Conclusion:** The results showed that the mean radius variance for merge regions (0.003536 µm^2) was actually slightly lower than that of safe regions (0.004049 µm^2). A Welch's t-test yielded a t-statistic of -1.8438 and a p-value of 0.0652. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, two-sample t-test, Student's t-test; p = 0.0652
- **Statistical issues:** none
- **Logic issues:** none

### 50. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 12 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** To assess if fragment cable length acts as a predictive feature for the likelihood of an endpoint participating in a valid topology-restoring connection.
- **Conclusion:** A Chi-square test of independence yielded a chi-square statistic of 350.26 and a p-value of 3.7177e-78, indicating a highly statistically significant relationship between a fragment's length class and its endpoint reconnection rate. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p ≈ 3.7e-78
- **Statistical issues:** none
- **Logic issues:** none

### 51. (Surprise 0.690) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 13 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if the size of the missing cable (the gap) is systematically dependent on the thickness of the neurite, informing dynamic search radii for proposal generation.
- **Conclusion:** The correlation between the average neurite radius and the physical gap distance was computed, yielding a Pearson correlation coefficient (r) of +0.3406 (p-value = 7.8252e-16) and a Spearman correlation coefficient (rho) of +0.2872 (p-value = 1.6612e-11). Instead of a negative correlation, the data shows a significant positive correlation: thicker neurites (larger radius) actually tend to have larger physical gaps when split, whereas thinner neurites have smaller gaps. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Spearman, Pearson; p ≈ 1.7e-11
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 52. (Surprise 0.690) Refuted: data contradict the claim that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 14 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if physical constraints of the imaging modality cause topological breaks to occur preferentially along the lower-resolution axis.
- **Conclusion:** The mean absolute directional components were 0.5353 for the X-axis, 0.4976 for the Y-axis, and 0.4535 for the Z-axis. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** one-way ANOVA, Student's t-test; p ≈ 3.9e-06
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 53. (Surprise 0.690) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 16 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Evaluate the relationship between local fragment crowding (density) and the volumetric occurrence rate of merge errors.
- **Conclusion:** The calculated Pearson correlation coefficient was -0.0162 (p = 1.32e-30) and the Spearman correlation was -0.0168 (p = 1.35e-32). The correlation between local fragment density and the occurrence of merge errors is remarkably weak (virtually zero) and slightly negative. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** effect size is negligible despite small p-value (large-N artifact)
- **Verdict:** Unsound
- **Test:** Spearman, Pearson; p ≈ 1.3e-32
- **Statistical issues:** statistical significance driven by very large N; effect size negligible (|r|<0.1) — significance is a large-sample artifact rather than a scientifically meaningful effect
- **Logic issues:** none

### 54. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 21 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** To determine if local skeleton geometry (radius and branching angle) can be used as a heuristic to detect false U-Net merges at branch points.
- **Conclusion:** The descriptive statistics showed almost identical properties between the two groups: the mean radius was 1.8954 µm for merges vs. 1.9172 µm for valid nodes, and the mean maximum branch angle was 153.54° for merges vs. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.4547
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 55. (Surprise 0.690) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 22 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Test if true structural gaps maintain a morphological scaling relationship (where larger gaps correlate with expected radius tapering) that is absent in false connections.
- **Conclusion:** Pearson correlation coefficients were calculated between the spatial gap distance and the absolute difference in node radius for both groups. Valid connections exhibited a weak negative correlation (r = -0.1001), while invalid connections showed a strong positive correlation (r = 0.7481), though the latter group had a very small sample size (n=6). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); effect size is negligible despite small p-value (large-N artifact)
- **Verdict:** Unsound
- **Test:** Fisher's exact, Pearson, Student's t-test; p = 0.0643; n up to 6
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 56. (Surprise 0.690) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 25 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if fragment size acts as a reliable confidence prior for split correction, which would justify prioritizing long-fragment merges.
- **Conclusion:** The Cochran-Armitage Chi-square test for trend resulted in a statistic of 0.0378 and a p-value of 0.8458, indicating no statistically significant trend in precision across fragment length quartiles. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square, Cochran-Armitage; p = 0.8458
- **Statistical issues:** none
- **Logic issues:** none

### 57. (Surprise 0.690) Refuted: data contradict the claim that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 26 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Assess if the presence of sudden, sharp turns in the skeleton geometry of a U-Net fragment is a reliable indicator of an underlying merge error.
- **Conclusion:** Contrary to the hypothesis, the sharpest turns in merged components were slightly less sharp than in valid components: merged fragments had a mean minimum angle of 126.96° (median 129.97°), while valid fragments had a mean minimum angle of 125.78° (median 129.08°). The Mann-Whitney U test (testing if merged < valid) returned a p-value of essentially 1.0 (9.9997e-01), indicating no statistical support for the hypothesis. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 58. (Surprise 0.690) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 28 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** To determine if the local density of fragment endpoints provides contextual evidence to disambiguate true from false reconnections at crowded endpoints.
- **Conclusion:** The True candidates exhibited a slightly higher mean density (2.36) compared to the False candidates (2.00). The Mann-Whitney U test p-value for the 'less' alternative was 0.8712, and the two-sided p-value was 0.2672. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.2672
- **Statistical issues:** none
- **Logic issues:** none

### 59. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 30 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** To evaluate whether local node thickness at branch points can serve as a topological flag for detecting merge errors.
- **Conclusion:** A Mann-Whitney U test (alternative='greater') returned a p-value of 0.4539. Because the p-value is well above standard significance levels (e.g., 0.05), there is no statistically significant difference in node thickness between the two groups. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.4539
- **Statistical issues:** none
- **Logic issues:** none

### 60. (Surprise 0.690) Refuted: data contradict the claim that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 34 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate the relationship between local neurite morphology (tortuosity) and the susceptibility of the automated segmentation to fragment the neurite.
- **Conclusion:** The Spearman rank correlation test yielded a correlation coefficient (rho) of 0.0746 with a highly significant p-value of 6.9263e-19. While the p-value indicates statistical significance, the correlation coefficient is extremely close to zero, representing a negligible positive correlation. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** effect size is negligible despite small p-value (large-N artifact)
- **Verdict:** Unsound
- **Test:** Spearman; p ≈ 6.9e-19
- **Statistical issues:** statistical significance driven by very large N; effect size negligible (|r|<0.1) — significance is a large-sample artifact rather than a scientifically meaningful effect
- **Logic issues:** none

### 61. (Surprise 0.690) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 39 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** To evaluate the viability of using topological cycle formation as a rollback mechanism for detecting early, high-confidence merge errors.
- **Conclusion:** The Fisher's Exact Test yielded a p-value of 0.314, indicating no statistically significant dependence between an edge being a false merge and its likelihood of forming a topological cycle. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Fisher's exact, Student's t-test; p = 0.314
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 62. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 44 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** To investigate if the automated segmentation algorithm systematically struggles with highly branched morphologies, leading to a proportionally higher rate of splits.
- **Conclusion:** A Pearson correlation test yielded an r-value of -0.3789 and a p-value of 0.1096. This result does not support the hypothesis; instead of a significant positive correlation, the data shows a non-significant, weak-to-moderate negative correlation between morphological complexity (branching density) and the U-Net fragmentation (split rate). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Pearson; p = 0.1096
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 63. (Surprise 0.690) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 48 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** To verify if local radius variance can serve as a sliding-window feature for an agent to autonomously detect merge sites.
- **Conclusion:** The results yielded a t-statistic of -6.7334 and a highly significant p-value of 4.19e-11, indicating a strong statistical difference. However, the findings contradict the initial hypothesis: natural biological branches actually exhibit a significantly higher mean local radius variance (0.0215 µm²) than artificial merge errors (0.0110 µm²). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** zero-variance data forced a degenerate test
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p ≈ 4.2e-11
- **Statistical issues:** none
- **Logic issues:** none

### 64. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 4 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Assess if local skeleton branching geometry can differentiate true axonal bifurcations from U-Net merge artifacts where fibers merely cross paths.
- **Conclusion:** The Mann-Whitney U test yielded a p-value of 0.1835 (p > 0.05), indicating that there is no statistically significant difference in the maximum branch angle distributions between the two classes. The mean maximum branch angle for True Branches was roughly 146.65 degrees, compared to 140.33 degrees for Merge Errors, with closely aligned medians (~147.4 vs. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 65. (Surprise 0.690) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 12 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Evaluate annotator-specific morphological biases in the ground truth skeletons.
- **Conclusion:** Eight unique annotators were identified across the 19 GT components: JG (N=4), HP (N=2), IG (N=2), SP (N=3), JT (N=1), PP (N=4), MB (N=1), and YZ (N=2). - For Branch Density, the test yielded an H-statistic of 6.5158 with a p-value of 0.4810. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis; p = 0.337; n up to 4
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 66. (Surprise 0.690) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 16 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Assess if neurite thickness continuity is a reliable morphological feature to distinguish true splits from false merges during reconnection proposal filtering.
- **Conclusion:** The analysis showed that the mean radius difference for valid pairs was 0.4455 µm (median 0.3984 µm) and for invalid pairs was 0.4669 µm (median 0.4238 µm). A Mann-Whitney U test yielded a p-value of ~0.4197, which is far above the 0.05 significance threshold. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.4197
- **Statistical issues:** none
- **Logic issues:** none

### 67. (Surprise 0.690) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 18 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Analyze how local reconstruction density impacts the severity (gap size) of split errors.
- **Conclusion:** The analysis yielded a strong negative correlation between the split gap distance and the local fragment density (Pearson: -0.7679, Spearman: -0.7830), with p-values indicating strong statistical significance. The binned summary statistics show that in areas of low density (0-2 distinct fragments), the mean gap size is large (~103.91 µm). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman, Pearson; p ≈ 0
- **Statistical issues:** none
- **Logic issues:** none

### 68. (Surprise 0.690) Refuted: data contradict the claim that morphological features mark U-Net merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 19 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Use 3D geometry to distinguish true biological branching from artificial merges by measuring the tetrahedral volume formed by the branch point and its divergent branches.
- **Conclusion:** Statistical analysis using the Mann-Whitney U test yielded a significant p-value (6.33e-05 < 0.05), indicating a distinct geometric difference between the two classes. However, the results contradict the initial hypothesis: True Branches exhibited a significantly *higher* mean (219.17 µm³) and median (176.59 µm³) non-coplanarity volume compared to Merge Errors (mean: 110.75 µm³, median: 70.79 µm³). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** median outcome is zero; effect driven by outliers
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p ≈ 6.3e-05
- **Statistical issues:** none
- **Logic issues:** none

### 69. (Surprise 0.690) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 24 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Evaluate if local leaf node density is a statistically significant spatial feature for distinguishing true reconnections from false ones.
- **Conclusion:** The calculation of the local density (number of leaf nodes within a 30 µm radius of the proposal midpoint) yielded the following results: - Valid Reconnections: Mean local density = 2.15, Median = 2.00 - Invalid Reconnections: Mean local density = 2.12, Median = 2.00 A Mann-Whitney U test was performed to compare these distributions, resulting in a p-value of approximately 0.8684. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.8684; n up to 2
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 70. (Surprise 0.690) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 25 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if simply prioritizing the closest absolute endpoints provides a statistically favorable baseline for true reconnections.
- **Conclusion:** **Summary of Results:** - **Valid Pairs (True Reconnections):** 350 pairs were identified, with a mean gap distance of 6.9415 µm (median: 5.0990 µm, std dev: 4.9400 µm). - **Invalid Pairs (False Reconnections):** 1047 pairs were identified, with a mean gap distance of 7.4942 µm (median: 5.0990 µm, std dev: 5.2948 µm). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.1971
- **Statistical issues:** none
- **Logic issues:** none

### 71. (Surprise 0.690) Refuted: data contradict the claim that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 30 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Assess whether split errors are disproportionately concentrated near the soma due to complex morphology challenging the segmentation model.
- **Conclusion:** The Poisson rate test produced a p-value of 0.9545, concluding that there is no statistically significant difference in split error density between the two regions. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Poisson rate; p = 0.9545
- **Statistical issues:** none
- **Logic issues:** none

### 72. (Surprise 0.690) Refuted: data contradict the claim that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 32 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Analyze the spatial distribution of split errors relative to their topological distance from the soma.
- **Conclusion:** The statistical analysis revealed a moderate, but highly significant, negative correlation between topological distance from the proxy soma and the split error rate (Pearson r = -0.4082, p = 5.92e-05). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Pearson; p ≈ 5.9e-05
- **Statistical issues:** none
- **Logic issues:** none

### 73. (Surprise 0.690) Refuted: data contradict the claim that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 33 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if elevated tortuosity serves as a geometry-only signature for detecting merged fragments.
- **Conclusion:** The statistical analysis (Welch's t-test) resulted in a p-value of 0.3591, indicating no statistically significant difference in tortuosity between the two groups. In fact, the mean tortuosity was nominally higher for Clean fragments (4.5121) compared to Merge fragments (3.3427). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p = 0.3591
- **Statistical issues:** none
- **Logic issues:** none

### 74. (Surprise 0.690) Refuted: data contradict the claim that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 35 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if geometric path tortuosity can identify and filter out short, false-positive noise fragments without relying on image data.
- **Conclusion:** Contrary to the initial hypothesis, the results demonstrate that mapped short fragments (true neurites) have a higher mean path tortuosity (1.8281) compared to orphaned short fragments (1.6119). The one-sided Mann-Whitney U test (testing if Orphaned > Mapped) yielded a p-value of 1.0, firmly rejecting the hypothesis. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 75. (Surprise 0.690) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 37 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if un-merging is a necessary prerequisite for successful split correction by measuring how merges distort the proximity of valid reconnection proposals.
- **Conclusion:** The mean gap distance for Merge-Adjacent splits was 83.85 µm, while Isolated splits averaged 81.81 µm. The statistical test yielded a p-value of 0.5192 (p > 0.05), indicating that there is no statistically significant difference in gap distances between the two categories. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 76. (Surprise 0.690) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 46 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if radius continuity can be used to dramatically improve the precision of reconnection proposals between nearby leaf nodes during split correction.
- **Conclusion:** The Chi-Square test yielded a p-value of 1.0, indicating no statistically significant difference between the two groups. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 77. (Surprise 0.690) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 48 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Test if symmetric nearest-neighbor relationships provide a higher precision baseline for valid reconnections than raw distance thresholds.
- **Conclusion:** - **Statistical Significance:** Fisher's Exact test yielded a p-value of 1.0729e-04. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Fisher's exact, Student's t-test; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 78. (Surprise 0.690) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 53 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate the spatial coupling of errors to see if densely packed neurite regions cause the U-Net to simultaneously fuse and prematurely terminate segmentations.
- **Conclusion:** The mean false split proportion per fragment was slightly higher for merged fragments (0.5752) compared to pure fragments (0.5442). A Mann-Whitney U test comparing the per-fragment false split proportions yielded a p-value of 0.3402. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.3402
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 79. (Surprise 0.690) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 56 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate whether localized crowding of fragment endpoints is a reliable meta-feature to flag ambiguous/risky reconnections that could cause merge errors.
- **Conclusion:** The mean local densities for the True and Ambiguous/False groups were nearly identical (1.0708 vs. The Mann-Whitney U test yielded a p-value of ~0.70, indicating no statistically significant difference in local endpoint density between the two groups. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.7
- **Statistical issues:** none
- **Logic issues:** none

### 80. (Surprise 0.690) Refuted: data contradict the claim that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 59 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if highly tortuous/curved neurite segments are more prone to causing U-Net split errors compared to straight segments.
- **Conclusion:** The classification yielded 3,907 False Splits (mean tortuosity = 1.057 +/- 0.085) and 3,390 True Terminations (mean tortuosity = 1.069 +/- 0.094). The Mann-Whitney U test yielded a p-value of 1.000, indicating that False Splits do not have significantly higher tortuosity than True Terminations (in fact, True Terminations appear slightly more tortuous on average). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 81. (Surprise 0.690) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 63 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Analyze the topological role of short fragments in the true split proposal graph to justify iterative, state-aware proofreading.
- **Conclusion:** Components were categorized by cable length into 'Small' (< 5000 µm, N=384) and 'Large' (>= 5000 µm, N=124). A Chi-square test comparing these proportions yielded a highly significant result (Chi-square = 69.05, p-value = 9.58e-17). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p ≈ 9.6e-17; n up to 384
- **Statistical issues:** none
- **Logic issues:** none

### 82. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 66 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Test if the length of newly formed branches acts as a robust post-hoc heuristic to detect and roll back aggressive false reconnections (reversible decisions).
- **Conclusion:** The Chi-squared test confirmed there is no statistically significant difference in the FDR between the two groups (chi2 = 0.0000, p-value = 1.0000). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 83. (Surprise 0.690) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 68 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Simulate the effect of a multi-pass, agentic proofreading strategy that dynamically updates graph topology, demonstrating that resolving small gaps makes larger disconnected networks easier to solve.
- **Conclusion:** Statistical analysis shows that the virtual reconnections successfully eliminated the smallest gaps, reducing the number of valid pairs ≤ 20 µm from 395 (baseline median = 5.20 µm) to 202 (new median = 9.90 µm). The baseline remaining gaps (N=214) had a median of 9.97 µm, while the new remaining gaps (N=202) had a median of 9.90 µm. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.441; n up to 214
- **Statistical issues:** none
- **Logic issues:** none

### 84. (Surprise 0.690) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 69 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate if acute branch angles serve as an identifiable geometric signature of U-Net merge errors where two parallel passing neurites were wrongly fused.
- **Conclusion:** While the FBR is slightly higher for the acute branches, a Chi-squared test yielded a p-value of 0.3917 (Chi2 = 0.7338), indicating that the difference is not statistically significant (p > 0.05). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** chi-square, Student's t-test; p = 0.05
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 85. (Surprise 0.690) Refuted: data contradict the claim that morphological features mark U-Net merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 70 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if local fragment density is a strong predictor of merge errors in the automated reconstruction.
- **Conclusion:** The results showed that the mean local component density at merge sites (1.10 components) was virtually identical to the density at non-merge sites (1.12 components). A Mann-Whitney U test yielded a p-value of 0.7417, indicating no statistically significant difference between the two groups. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.7417
- **Statistical issues:** none
- **Logic issues:** none

### 86. (Surprise 0.690) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 75 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if fragment length correlates with component purity to establish whether long fragments can be safely frozen as reliable anchors in an agentic workflow.
- **Conclusion:** Contrary to the hypothesis, the results show that Short fragments are slightly but significantly more pure than Long fragments (Mean Purity: 0.9986 vs. The one-sided Mann-Whitney U test (testing if Long purity > Short purity) yielded a p-value of 0.994, firmly rejecting the hypothesis. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.0118
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 87. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 77 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if the global topological complexity of a fragment (branch density) is predictive of it containing a merge error.
- **Conclusion:** The results show that the Mean Branch Density for Merge fragments is ~0.0032, while for Clean fragments it is ~0.0042. The Mann-Whitney U test indicates a highly significant difference between the two distributions (p-value = 5.03e-13), and the point-biserial correlation reveals a slight negative correlation (-0.079, p-value = 1.44e-15) between a fragment's branch density and its probability of containing a merge error. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, point-biserial; p ≈ 1.4e-15
- **Statistical issues:** none
- **Logic issues:** none

### 88. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 82 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Test if invalid reconnections are often actually missed branch points rather than continuations of a main trunk.
- **Conclusion:** These endpoints were cross-referenced with the ground-truth (GT) graph to classify each inter-component pair as 'Valid' (both mapping to the same GT neuron, n=350) or 'Invalid' (n=1047). **Results & Findings:** - **Valid Reconnections (n=350):** Mean distance to nearest branch = 190.9910 µm, Median = 93.9641 µm, Std Dev = 223.1916 µm. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.1368; n up to 1047
- **Statistical issues:** none
- **Logic issues:** none

### 89. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 87 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate whether topology errors cluster in specific problematic regions of the tissue, suggesting that resolving a merge error should trigger a localized re-planning of split connections.
- **Conclusion:** The statistical analysis showed that the mean distance to the nearest merge junction for split endpoints was 3159.20 µm, which was slightly larger than the mean distance for correct nodes (2925.77 µm). While the Kolmogorov-Smirnov test yielded a significant p-value (p = 3.6188e-11), the direction of the difference contradicted the hypothesis. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov; p ≈ 3.6e-11
- **Statistical issues:** none
- **Logic issues:** none

### 90. (Surprise 0.690) Refuted: data contradict the claim that image-intensity features (signal across the gap) add discriminative power.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 89 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Contrast the image-level evidence underlying 'omission' errors versus 'split' errors to understand the limits of automated segmentation.
- **Conclusion:** **Results:** - **Median Intensity (Omitted GT Nodes):** 56.50 - **Median Intensity (Split Midpoints):** 33.50 - **Mann-Whitney U Test (Omitted < Split):** p-value = 1.00 **Findings & Conclusion:** The original hypothesis postulated that omitted GT edges would exhibit significantly lower raw fluorescence intensity than split gaps, under the assumption that true omissions are driven by severe signal dropout. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 91. (Surprise 0.690) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 91 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate whether the graph-level state feature 'component size' is predictive of error types, which would justify dynamically refreshing component features during an iterative proofreading process.
- **Conclusion:** The median cable length for valid split pairs (11,592.46 µm) is roughly an order of magnitude larger than the median for merge error components (1,194.61 µm). However, the merge error distribution is extremely right-skewed; its mean (53,742.08 µm) is larger than the mean of valid split pairs (46,470.80 µm), driven by rare but massive outlier components (reaching ~500,000 µm). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 1.9e-53
- **Statistical issues:** none
- **Logic issues:** none

### 92. (Surprise 0.690) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 94 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if reconstruction errors are uniformly distributed or biased by the specific annotator/brain region traced.
- **Conclusion:** The experiment was successfully executed. The code accurately mapped U-Net fragments to the human ground truth, calculated the Edge Accuracy and Split Rate per connected component, and grouped the metrics by the respective annotator (JG, HP, IG, SP, JT, PP, MB, YZ). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis, one-way ANOVA, Student's t-test; p = 0.59
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 93. (Surprise 0.690) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 97 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Analyze the relationship between the scale of a split error (gap distance) and the local crowding of fragment endpoints.
- **Conclusion:** **Experiment Results:** - **Identified True Split Gaps:** 1,801 - **Spearman Rank Correlation:** 0.2622 - **P-value:** 1.0567e-29 - **Mean Gap Distance:** 11.42 µm - **Mean Local Density:** 0.20 other endpoints **Findings:** The results explicitly contradict the initial hypothesis. The hypothesis predicted a negative correlation between gap distance and local endpoint density (i.e., denser regions would systematically produce shorter gaps). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman; p = 0.001
- **Statistical issues:** none
- **Logic issues:** none

### 94. (Surprise 0.690) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 98 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Evaluate the viability of using skeleton radius differences at endpoints as a lightweight, image-free feature for filtering invalid split reconnection proposals, ensuring robust statistical testing by mitigating data type overflow.
- **Conclusion:** For the absolute radius difference, the median was 0.3438 µm (Valid) vs. For the radius ratio, the median was 0.7466 (Valid) vs. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.6106
- **Statistical issues:** none
- **Logic issues:** none

### 95. (Surprise 0.690) Refuted: data contradict the claim that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 99 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate the correlation between the intrinsic tortuosity of a neuron and its susceptibility to automated fragmentation.
- **Conclusion:** A Pearson correlation test between average tortuosity and split rate yielded an r-value of -0.0545 and a p-value of 0.8247. These statistical results, corroborated by the visual scatter plot, indicate that there is virtually no linear correlation between the variables. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Pearson; p = 0.8247
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 96. (Surprise 0.690) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 105 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Evaluate if machine learning on a small suite of locally extracted skeleton features outperforms static heuristics for split-correction.
- **Conclusion:** The experiment successfully executed the pipeline to test the hypothesis. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** logistic regression
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 97. (Surprise 0.690) Refuted: data contradict the claim that image-intensity features (signal across the gap) add discriminative power.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 107 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Prove that integrating raw image evidence directly along proposed gaps can distinguish true discontinuities (faint signals) from false connections (background darkness).
- **Conclusion:** The results yielded a mean minimum intensity of 76.66 for TP proposals and 80.62 for FP proposals. An independent samples t-test produced a t-statistic of -0.1324 and a p-value of 0.8951. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Student's t-test; p = 0.8951
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 98. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 109 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate if fragment cable length correlates with the frequency of topological merge errors.
- **Conclusion:** **Experiment Results:** - **Cable Length Distribution**: Analyzed 10,172 total fragments with a mean length of 1849.31 µm and median of 1175.46 µm. - **Statistical Significance**: A Mann-Whitney U Test yielded a p-value of 2.6986e-17, indicating a highly significant statistical difference between the two distributions. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U, Student's t-test; p ≈ 2.7e-17
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 99. (Surprise 0.690) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 110 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Test if adapting split-correction strictness to local error clustering improves overall reconnection accuracy.
- **Conclusion:** The experiment successfully executed the pipeline to test the hypothesis. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** test type not extractable from record
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 100. (Surprise 0.690) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 112 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Assess whether local proposal density can act as a negative feature to suppress false reconnections in dense, highly fragmented regions.
- **Conclusion:** The descriptive statistics showed that the local density (within a 20 µm radius) was almost identical for both groups: TP proposals had a mean density of 2.09 (median 2.00, std 0.34), while FP proposals had a mean density of 2.08 (median 2.00, std 0.26). A Mann-Whitney U test testing the hypothesis that FP local density is strictly greater than TP local density yielded a non-significant p-value of ~0.5318. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.5318
- **Statistical issues:** none
- **Logic issues:** none

### 101. (Surprise 0.690) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 116 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Evaluate whether enforcing mutual exclusivity via Maximum Weight Matching reduces false-positive connections compared to an independent classification baseline, using identical scoring criteria and candidate pools.
- **Conclusion:** The modified experiment executed successfully, employing an apples-to-apples comparison by using a unified edge score threshold (score > 0.5) and a 20 µm search radius for both the baseline and proposed approaches. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 102. (Surprise 0.690) Refuted: data contradict the claim that image-intensity features (signal across the gap) add discriminative power.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 118 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if local image evidence in the gap between endpoints can effectively distinguish true splits from false-positive reconnections.
- **Conclusion:** The experiment successfully tested the hypothesis that incorporating the mean raw fluorescence intensity from the gap between fragment endpoints into the reconnection score increases True Positive precision by at least 15%. The baseline model (Geometry Only) achieved a near-perfect PR-AUC of 0.9977. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** logistic regression, ROC-AUC
- **Statistical issues:** none
- **Logic issues:** none

### 103. (Surprise 0.690) Refuted: data contradict the claim that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 122 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Analyze the spatial frequency of splits relative to the soma across the neuronal arbor.
- **Conclusion:** The chi-square statistical test yielded a test statistic of 146.09 with a p-value of 1.24e-33, indicating that the distribution of splits is highly significantly different from what would be expected if splits were proportional to cable length. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square, Student's t-test; p ≈ 1.2e-33
- **Statistical issues:** none
- **Logic issues:** none

### 104. (Surprise 0.690) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 123 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if ground-truth annotator style or bias correlates with the degree of fragmentation in the automated reconstruction.
- **Conclusion:** The mean split rates varied from a low of 0.14 (Annotator JT, n=1) to a high of 0.22 (Annotator PP, n=4, and Annotator SP, n=3). Despite the visual variance observed in the boxplot across different annotator initials, the Kruskal-Wallis H test yielded a p-value of 0.2249 (statistic = 9.4053). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis; p = 0.2249; n up to 4
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 105. (Surprise 0.690) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 124 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate cross-annotator variability in ground-truth center-line placement, which can confound precise distance-based metrics and radius estimations.
- **Conclusion:** It loaded the dataset, grouped the 19 ground-truth components by annotator initials extracted from the SWC IDs, computed the mean physical alignment error (distances <= 10 µm) to the closest U-Net fragment node, and performed a one-way ANOVA. The mean alignment errors across annotators ranged from ~1.58 µm to ~1.69 µm. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** one-way ANOVA; p = 0.632
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 106. (Surprise 0.690) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 126 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if local neurite radius at a split point is inversely correlated with the distance to the nearest potential reconnection candidate, which could help prioritize split-correction proposals.
- **Conclusion:** The mean leaf radius was ~1.38 µm, and the mean gap distance was ~21.29 µm. A Pearson correlation test yielded an r-value of 0.1931 with a statistically significant p-value of 9.44e-45. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Pearson; p ≈ 9.4e-45
- **Statistical issues:** none
- **Logic issues:** none

### 107. (Surprise 0.690) Refuted: data contradict the claim that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 132 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Test if local tortuosity (path length divided by straight-line distance) can act as a structural feature to flag heavily tangled regions prone to merge errors.
- **Conclusion:** The script loaded the dataset using the modified unpickler, queried the KDTree to map U-Net fragments nodes to the Ground-Truth (GT) graph, and classified fragment components into 'Merged' (n=11) and 'Pure' (n=951) based on their mapping to GT neurons. Upon calculating the tortuosity (ratio of path length to Euclidean distance) over 50-µm segments, the results showed that both Merged and Pure components have near-identical mean tortuosities (~1.0534 vs 1.0535) and variances. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.423; n up to 951
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 108. (Surprise 0.690) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 133 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate the relationship between a fragment's reconstructed size and the magnitude of the topological split errors separating it from its true neighbors.
- **Conclusion:** Evaluating 8,780 valid gap reconnections, the analysis yielded a Spearman Rank Correlation of 0.3096 with a highly significant p-value of 2.1552e-194. Contrary to the initial hypothesis, which posited a negative correlation (expecting shorter gaps for longer fragments), the results demonstrate a statistically significant *positive* correlation. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman; p ≈ 2.2e-194
- **Statistical issues:** none
- **Logic issues:** none

### 109. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 142 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if neuron morphological complexity (branching density) is a strong predictor of fragmentation rate in the automated reconstruction.
- **Conclusion:** The results revealed a mean branching density of 1.35 branches/mm and a mean split rate of 0.18 fragments/mm. The statistical analysis showed a significant negative correlation between branching density and split rate (Pearson r = -0.5962, p=0.0071; Spearman r = -0.5018, p=0.0286). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman, Pearson; p = 0.0286
- **Statistical issues:** none
- **Logic issues:** none

### 110. (Surprise 0.690) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 150 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if spatial crowding of predicted fragments is a reliable proxy for detecting false merges.
- **Conclusion:** A Welch's t-test yielded a t-statistic of -14.5534 and a highly significant p-value of 1.2880e-40. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p ≈ 1.3e-40
- **Statistical issues:** none
- **Logic issues:** none

### 111. (Surprise 0.690) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 153 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Analyze annotator-specific metadata to see if split error rates are biased by human selection of specific morphological regions or neuron types.
- **Conclusion:** The test yielded an H-statistic of 8.4105 and a p-value of 0.2978. Because the p-value (0.2978) is well above the standard alpha threshold of 0.05, we fail to reject the null hypothesis. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis; p = 0.2978
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 112. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 154 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate if anomalous localized spikes in predicted neurite radius can serve as an endogenous feature for agentic merge detection, whilst preventing float16 numerical overflow during aggregation.
- **Conclusion:** The results reveal that the mean junction radius (1.9355 µm) is slightly lower than the mean bulk radius (1.9759 µm), resulting in a mean difference of -0.0405 µm. The statistical tests (paired t-test p-value = 0.1291; Wilcoxon signed-rank test p-value = 0.1601) indicate that there is no statistically significant difference in radius between the junction and bulk regions of a merge fragment. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** paired t-test, Wilcoxon, Student's t-test; p = 0.1601
- **Statistical issues:** none
- **Logic issues:** none

### 113. (Surprise 0.690) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 156 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** To determine if an anomalously high endpoint density can serve as an intrinsic topological signature for identifying fragments that erroneously merge distinct neurons.
- **Conclusion:** A Mann-Whitney U test yielded a p-value of ~0.787, indicating no statistically significant difference between the two groups. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.787
- **Statistical issues:** none
- **Logic issues:** none

### 114. (Surprise 0.690) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 158 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Evaluate neurite radius continuity as an endogenous feature for resolving mutually exclusive reconnection proposals.
- **Conclusion:** The mean absolute radius difference for TP reconnections was 0.7392 µm, while for FP reconnections it was 0.7823 µm. Statistical testing yielded a paired t-test p-value of 0.3127 and a Wilcoxon signed-rank p-value of 0.3750, indicating no statistically significant difference in radius continuity between valid and invalid reconnections. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** paired t-test, Wilcoxon, Student's t-test; p = 0.375
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 115. (Surprise 0.690) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 162 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Assess the geometric branching angle as an actionable feature to detect and flag automated merge errors without image data.
- **Conclusion:** A Welch's two-sample t-test yielded a t-statistic of 0.3588 and a p-value of 0.7206. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, two-sample t-test, Student's t-test; p = 0.7206
- **Statistical issues:** none
- **Logic issues:** none

### 116. (Surprise 0.690) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 169 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Investigate if fragment length acts as a proxy for segmentation confidence and whether extremely short fragments disproportionately contribute to false merge proposals.
- **Conclusion:** Statistical testing (Chi-square and Fisher's exact test) yielded a p-value of 1.0, indicating no statistically significant difference in the false merge rates between the two length categories. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square, Fisher's exact, Student's t-test; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 117. (Surprise 0.690) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 171 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Verify if the principle of mutual exclusivity holds locally by testing whether true reconnections geometrically dominate alternative local false proposals by a wider margin than when a true connection is absent.
- **Conclusion:** The leaves were grouped based on whether the best candidate correctly belonged to the same ground-truth neuron (True-dominant, n=46) or not (False-dominant, n=105). A Mann-Whitney U test yielded a statistic of 2370.0 and a p-value of ~0.573, indicating no statistically significant difference in the alignment margins between the two groups. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.573; n up to 105
- **Statistical issues:** none
- **Logic issues:** none

### 118. (Surprise 0.690) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 181 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Evaluate the impact of neurite radius continuity on split-correction precision by leveraging the `node_radius` attribute.
- **Conclusion:** The Kolmogorov-Smirnov test yielded a non-significant p-value of 0.9191, demonstrating no statistically significant difference between the radius difference distributions of True Positive (TP) and False Positive (FP) reconnections. In fact, the mean relative radius difference for TPs (0.2805) was slightly higher than for FPs (0.2372). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Kolmogorov-Smirnov; p = 0.9191
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 119. (Surprise 0.690) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 182 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if specific human annotators traced fundamentally more complex or difficult neurons, leading to observable bias in automated error rates.
- **Conclusion:** A Kruskal-Wallis H-test was conducted to assess statistical significance across annotator groups, resulting in an H-statistic of 8.4105 and a p-value of 0.2978. Because the p-value is greater than 0.05, we fail to reject the null hypothesis. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis; p = 0.2978
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 120. (Surprise 0.690) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 185 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Assess how annotator variability and neuron morphological complexity influence the baseline topological split error rate of the automated U-Net reconstruction.
- **Conclusion:** An ANOVA test (F-statistic = 0.644, p-value = 0.713) and Kruskal-Wallis test (H-statistic = 6.516, p-value = 0.481) revealed no statistically significant differences in branching frequency across the 8 annotators. **Correlation to Split Rate**: The U-Net split rate (UNet fragments per 1,000 µm of GT cable) was computed for each GT neuron (ranging from ~0.10 to 0.34 splits / 1000 µm). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis, one-way ANOVA, Spearman; p = 0.161
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 121. (Surprise 0.690) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 186 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Test a reversible decision (rollback) mechanism against a strict greedy baseline for split-correction.
- **Conclusion:** The experiment successfully loaded the dataset and implemented both a baseline greedy split-correction algorithm and the proposed rollback mechanism. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 122. (Surprise 0.690) Refuted: data contradict the claim that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 187 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Assess how voxel anisotropy (lower Z-resolution) impacts the spatial distribution and reliability of split-correction proposals.
- **Conclusion:** Fisher's Exact Test yielded a p-value of 0.5648, which indicates that the difference in true-positive precision between the two orientations is not statistically significant. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Fisher's exact, Student's t-test; p = 0.5648
- **Statistical issues:** none
- **Logic issues:** none

### 123. (Surprise 0.690) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 189 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Determine if merge errors tend to co-occur with dense clusters of broken fragments (ambiguous split regions), supporting the idea that split and merge correction must be coupled.
- **Conclusion:** A Mann-Whitney U test comparing these densities yielded a statistic of 906,017.0 and a p-value of 0.3845. **Conclusion**: The p-value of 0.3845 indicates that the difference in local proposal densities between proximal and distal endpoints is not statistically significant. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.3845
- **Statistical issues:** none
- **Logic issues:** none

### 124. (Surprise 0.690) Refuted: data contradict the claim that image-intensity features (signal across the gap) add discriminative power.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 192 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Test if an abnormally large node radius can be used as a standalone local feature to flag potential merge errors prior to agentic rollback.
- **Conclusion:** The analysis found that the mean node radius for Merge branch points was 1.8455 µm (Variance: 0.0859), while Valid branch points had a slightly higher mean radius of 1.9529 µm (Variance: 0.0704). The Mann-Whitney U test yielded a U-statistic of 81483.5000 and a p-value of 0.9766, which is much greater than the standard 0.05 alpha level. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.9766
- **Statistical issues:** none
- **Logic issues:** none

### 125. (Surprise 0.690) Refuted: data contradict the claim that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 197 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Assess whether the lower axial resolution of the light-sheet microscope systematically biases the directionality of U-Net split errors.
- **Conclusion:** The mean absolute projection magnitudes along the X, Y, and Z axes were 0.5097, 0.4891, and 0.4924, respectively. While the Friedman test yielded a statistically significant result (p-value ≈ 0.0013), indicating that the distributions across the three axes are not identical, the expected dominant alignment with the lower-resolution Z-axis was not observed. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p = 0.0013
- **Statistical issues:** none
- **Logic issues:** none

### 126. (Surprise 0.690) Refuted: data contradict the claim that image-intensity features (signal across the gap) add discriminative power.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 200 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2656) · **Direction:** Negative
- **Tested:** Validate whether raw image variance acts as a direct signature of false merges (intersecting distinct axons) compared to actual structural bifurcations.
- **Conclusion:** The results showed that valid branches had a median variance of 972.85 and a mean of 3243.85, whereas invalid merges had a median variance of 846.88 and a mean of 2354.59. An independent Welch's t-test yielded a p-value of 0.1895, indicating no statistically significant difference between the two groups. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.27) (signed surprisal -0.690; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, two-sample t-test, Student's t-test; p = 0.1895
- **Statistical issues:** none
- **Logic issues:** none

### 127. (Surprise 0.690) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 149 · **Belief:** Leaning False -> Leaning True (0.2917 -> 0.7344) · **Direction:** Positive
- **Tested:** Model the relationship between a fragment's morphological extent (cable length) and its likelihood of containing topological merges.
- **Conclusion:** The logistic regression results strongly support the hypothesis: the natural logarithm of cable length is a statistically significant predictor of a merge error (p-value = 2.04e-05). Belief therefore rose from Leaning False (0.29) to Leaning True (0.73) (signed surprisal +0.690; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** logistic regression; p ≈ 2e-05
- **Statistical issues:** none
- **Logic issues:** none

### 128. (Surprise 0.674) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 42 · **Belief:** Leaning True -> Leaning False (0.7917 -> 0.3594) · **Direction:** Negative
- **Tested:** Determine if the baseline fragmentation of the automated U-Net is modulated by the subjective variations in the human ground truth used for evaluation.
- **Conclusion:** Grouped by the eight different human annotators, the median split rates ranged from 0.14 to 0.22. However, the Kruskal-Wallis H-test resulted in an H-statistic of 9.1763 and a p-value of 0.2402. Belief therefore dropped from Leaning True (0.79) to Leaning False (0.36) (signed surprisal -0.674; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis; p = 0.2402
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 129. (Surprise 0.659) Refuted: data contradict the claim that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 5 · **Belief:** Leaning True -> Likely False (0.6250 -> 0.1442) · **Direction:** Negative
- **Tested:** Assess whether the automated segmentation algorithm systematically struggles with and fragments highly winding (tortuous) morphological structures.
- **Conclusion:** The results show a striking morphological difference: Ground Truth components have a much higher tortuosity (Median = 18.79, Mean = 19.69) compared to the Fragments (Median = 1.38, Mean = 1.74). The Mann-Whitney U test yielded a highly significant p-value (5.0688e-14), confirming a robust statistical difference. Belief therefore dropped from Leaning True (0.62) to Likely False (0.14) (signed surprisal -0.659; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p ≈ 5.1e-14; n up to 19
- **Statistical issues:** none
- **Logic issues:** none

### 130. (Surprise 0.659) Refuted: data contradict the claim that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 26 · **Belief:** Leaning True -> Likely False (0.6250 -> 0.1442) · **Direction:** Negative
- **Tested:** Investigate if spatial proximity to the soma is a major risk factor for U-Net merge errors.
- **Conclusion:** The results directly contradict the initial hypothesis: merge errors are significantly located *further* from the soma (mean distance 8993.53 µm) compared to control nodes (7597.91 µm), with an extremely significant p-value of 7.8874e-76 and a KS statistic of 0.4257. Belief therefore dropped from Leaning True (0.62) to Likely False (0.14) (signed surprisal -0.659; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Kolmogorov-Smirnov, Student's t-test; p ≈ 7.9e-76
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 131. (Surprise 0.641) Refuted: data contradict the claim that image-intensity features (signal across the gap) add discriminative power.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 14 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2404) · **Direction:** Negative
- **Tested:** Investigate the local raw image intensity signatures around split errors versus true anatomical endpoints.
- **Conclusion:** The local 16x16x16 voxel patches were analyzed for mean and maximum fluorescence intensities. The independent t-tests revealed no statistically significant differences at the standard alpha=0.05 level, although the mean intensity comparison was borderline (t-statistic = -1.9313, p-value = 0.0548). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.24) (signed surprisal -0.641; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** two-sample t-test, Student's t-test; p = 0.0548
- **Statistical issues:** none
- **Logic issues:** none

### 132. (Surprise 0.641) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 29 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2404) · **Direction:** Negative
- **Tested:** Determine if spatial distance can effectively resolve mutually exclusive competing reconnection proposals at a single endpoint without evaluating them independently.
- **Conclusion:** Due to this severely limited sample size (N=2), the Wilcoxon signed-rank test (p-value = 0.5) cannot yield statistically meaningful conclusions. Interestingly, in the two competitive instances identified, the results directly contradicted the hypothesis: the mean distance to the closest *false* candidate was shorter (9.99 µm) than the distance to the closest *valid* candidate (15.81 µm). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.24) (signed surprisal -0.641; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Wilcoxon; p = 0.5; n up to 2
- **Statistical issues:** none
- **Logic issues:** none

### 133. (Surprise 0.641) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 43 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2969) · **Direction:** Negative
- **Tested:** To determine if the size of the components being connected is a risk factor for generating false reconnections, as larger fragments span more volume and have higher chances of incidental proximity.
- **Conclusion:** The resulting median proposal sizes were 1,040 nodes for True Splits and 786 nodes for False Merges. The Mann-Whitney U test yielded a p-value of 0.476, indicating no statistically significant difference in proposal sizes between the two groups. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.30) (signed surprisal -0.641; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.476
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 134. (Surprise 0.641) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 60 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2969) · **Direction:** Negative
- **Tested:** Analyze whether human annotator bias influences the evaluation metrics of the automated U-Net reconstruction.
- **Conclusion:** The mean split rates per annotator ranged from approximately 0.14 to 0.23 fragments/mm. Statistical Testing: A Kruskal-Wallis test was performed on the grouped split rates, returning an H-statistic of 8.4105 and a p-value of 0.2978. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.30) (signed surprisal -0.641; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis; p = 0.2978
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 135. (Surprise 0.641) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 119 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.2969) · **Direction:** Negative
- **Tested:** Determine if aggressive merging predominantly fails because it erroneously attaches long valid fragments to short background noise fragments.
- **Conclusion:** The median fragment length was computed as 1,175.46 µm, perfectly splitting the 10,172 fragments into 5,086 'Short' and 5,086 'Long' components. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.30) (signed surprisal -0.641; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Questionable
- **Test:** test type not extractable from record
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 136. (Surprise 0.615) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 51 · **Belief:** Likely True -> Leaning False (0.8333 -> 0.3846) · **Direction:** Negative
- **Tested:** Quantify and compare the automated segmentation's split error rates across ground-truth neurons grouped by human annotator to identify potential annotation bias.
- **Conclusion:** The Kruskal-Wallis test yielded a test statistic of 10.4029 and a p-value of 0.1669. Because the p-value is greater than the standard alpha level of 0.05, we fail to reject the null hypothesis. Belief therefore dropped from Likely True (0.83) to Leaning False (0.38) (signed surprisal -0.615; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis, Student's t-test; p = 0.1669; n up to 19
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 137. (Surprise 0.609) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 3 · **Belief:** Leaning True -> Leaning False (0.6250 -> 0.2344) · **Direction:** Negative
- **Tested:** Evaluate how fragment component size impacts the reliability of distance-based reconnection proposals to inform dynamic feature refreshing.
- **Conclusion:** The Pearson correlation coefficient calculated is 0.1252 with a non-significant p-value of 0.7304. Belief therefore dropped from Leaning True (0.62) to Leaning False (0.23) (signed surprisal -0.609; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Pearson; p = 0.7304
- **Statistical issues:** none
- **Logic issues:** none

### 138. (Surprise 0.593) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 6 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3281) · **Direction:** Negative
- **Tested:** Analyze cross-annotator variability in the context of automated reconstruction performance to determine if certain human annotators systematically trace regions that the U-Net fails to predict.
- **Conclusion:** Mean omission rates across the annotators ranged from ~11.4% to ~27.2%. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.33) (signed surprisal -0.593; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis, one-way ANOVA; p = 0.4775
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 139. (Surprise 0.568) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 3 · **Belief:** Likely True -> Uncertain (0.8333 -> 0.4688) · **Direction:** Negative
- **Tested:** Evaluate the risk of relying purely on a 1-Nearest Neighbor distance-based greedy matching strategy for split correction.
- **Conclusion:** A binomial test comparing this accuracy to a high threshold of 90% yielded a p-value of 0.3722, meaning we fail to reject the null hypothesis that the 1-NN accuracy is 90%. Belief therefore dropped from Likely True (0.83) to Uncertain (0.47) (signed surprisal -0.568; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** binomial test; p = 0.3722
- **Statistical issues:** none
- **Logic issues:** none

### 140. (Surprise 0.568) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 165 · **Belief:** Likely True -> Uncertain (0.8333 -> 0.4688) · **Direction:** Negative
- **Tested:** Quantify how local endpoint clustering limits simple distance-based reconnects, motivating the need for mutual-exclusivity constraints and GNN-based joint reasoning.
- **Conclusion:** However, contrary to the hypothesis, the point-biserial correlation between local endpoint density and reconnect success rate showed a weak but statistically significant *positive* correlation (r = 0.1126, p < 0.001). Belief therefore dropped from Likely True (0.83) to Uncertain (0.47) (signed surprisal -0.568; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** point-biserial, Student's t-test; p = 0.001
- **Statistical issues:** none
- **Logic issues:** none

### 141. (Surprise 0.544) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 49 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3594) · **Direction:** Negative
- **Tested:** Test whether baseline omission error rates are biased by the individual annotator styles embedded in the ground truth.
- **Conclusion:** The results show variations in mean and median omission rates among annotators (e.g., SP with a mean of 0.2616, JT with a mean of 0.1047). However, the Kruskal-Wallis H Test yielded a p-value of 0.4775, which is far greater than the standard alpha level of 0.05. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.36) (signed surprisal -0.544; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kruskal-Wallis; p = 0.4775
- **Statistical issues:** none
- **Logic issues:** none

### 142. (Surprise 0.544) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 131 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3594) · **Direction:** Negative
- **Tested:** To demonstrate that local endpoint crowding degrades geometric alignment cues, necessitating joint (agentic) reasoning over competing proposals rather than independent single-pass edge classification.
- **Conclusion:** The Mann-Whitney U test yielded a highly significant p-value (9.0681e-04). Interestingly, the Multi-Candidate group showed a higher mean maximum tangent cosine similarity (0.8503) compared to the Single-Candidate group (0.7524). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.36) (signed surprisal -0.544; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 0.00091
- **Statistical issues:** none
- **Logic issues:** none

### 143. (Surprise 0.536) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 66 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3173) · **Direction:** Negative
- **Tested:** Demonstrate the necessity of iterative topology re-planning by quantifying how many unrecoverable long gaps become recoverable via small intermediate fragments.
- **Conclusion:** The experiment successfully revised the `MockUnpickler` to handle missing dependencies robustly using a `GenericMock` and executed the logic correctly. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.32) (signed surprisal -0.536; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** test type not extractable from record
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 144. (Surprise 0.527) Refuted: data contradict the claim that morphological features mark U-Net merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 27 · **Belief:** Leaning True -> Leaning False (0.7500 -> 0.3654) · **Direction:** Negative
- **Tested:** Quantify the 3D crossing angles of distinct true neurites at locations where the U-Net erroneously merged them, compared to non-merged proximal crossings.
- **Conclusion:** The mean crossing angle for merge errors was 56.47 degrees, compared to 60.94 degrees for safe crossings. A Mann-Whitney U test yielded a p-value of 0.07855, which is slightly above the standard significance threshold (0.05). Belief therefore dropped from Leaning True (0.75) to Leaning False (0.37) (signed surprisal -0.527; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.07855
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 145. (Surprise 0.495) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 18 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3906) · **Direction:** Negative
- **Tested:** To investigate the coupling between merge errors and cycle creation when aggressive, distance-based reconnecting is applied.
- **Conclusion:** The experiment successfully simulated a greedy distance-based reconnection (within a 15 µm radius) of fragment leaves to investigate the correlation between newly introduced topological cycles and pre-existing merge errors. The Fisher's Exact Test yielded a p-value of 0.0060, indicating a statistically significant association between the presence of a cycle and a merge error. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.39) (signed surprisal -0.495; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Fisher's exact, Student's t-test; p = 0.006
- **Statistical issues:** none
- **Logic issues:** none

### 146. (Surprise 0.495) Refuted: data contradict the claim that cycle-formation cleanly flags false-merge reconnection proposals.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 24 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3906) · **Direction:** Negative
- **Tested:** Determine if cycle-creation is a reliable, unsupervised topological filter for rejecting invalid connection proposals.
- **Conclusion:** The Chi-Square Test of Independence yielded a statistic of 0.6852 and a p-value of 0.4078, indicating no statistically significant association. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.39) (signed surprisal -0.495; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** chi-square; p = 0.4078
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 147. (Surprise 0.495) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 33 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3906) · **Direction:** Negative
- **Tested:** Evaluate whether local neurite radius continuity acts as a robust biological constraint to filter out incorrect split-correction proposals.
- **Conclusion:** The mean absolute radius difference for Valid connections was 0.4455 ± 0.3668 µm, while for Invalid connections it was 0.3783 ± 0.2810 µm. A Welch's independent samples t-test was performed, resulting in a t-statistic of 0.3371 and a p-value of 0.7677. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.39) (signed surprisal -0.495; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Welch's t-test, Student's t-test; p = 0.7677; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 148. (Surprise 0.495) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 83 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3906) · **Direction:** Negative
- **Tested:** Investigate if high spatial congestion (endpoint density) is a strong structural proxy indicator for the presence of U-Net merge errors.
- **Conclusion:** A Mann-Whitney U test yielded a p-value of 0.2595. Since the p-value is greater than 0.05, the difference in endpoint density between Merged and Clean components is not statistically significant. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.39) (signed surprisal -0.495; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.2595
- **Statistical issues:** none
- **Logic issues:** none

### 149. (Surprise 0.495) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 120 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3906) · **Direction:** Negative
- **Tested:** Investigate the value of a 'mutual exclusivity' feature (distance margin) by evaluating if highly competitive proposals indicate lower confidence and higher false-positive rates.
- **Conclusion:** A point-biserial correlation test yielded a correlation coefficient of 0.0572 and a high p-value of 0.6964. Because the p-value is significantly greater than the standard 0.05 threshold, there is no statistically significant correlation between the distance margin (the difference in distance between the first and second candidates) and the correctness of the reconnection proposal. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.39) (signed surprisal -0.495; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** point-biserial; p = 0.6964
- **Statistical issues:** none
- **Logic issues:** none

### 150. (Surprise 0.487) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 17 · **Belief:** Leaning True -> Uncertain (0.7500 -> 0.4375) · **Direction:** Negative
- **Tested:** Assess whether the automated segmentation model is systematically more prone to fragmentation at neurite branching points compared to continuous axonal cables.
- **Conclusion:** The Chi-square goodness-of-fit test resulted in a chi-square statistic of 2.7131 and a p-value of 0.0995. Since the p-value (0.0995) is greater than the standard significance level of 0.05, we fail to reject the null hypothesis. Belief therefore dropped from Leaning True (0.75) to Uncertain (0.44) (signed surprisal -0.487; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square, Student's t-test; p = 0.0995
- **Statistical issues:** none
- **Logic issues:** none

### 151. (Surprise 0.487) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 57 · **Belief:** Leaning True -> Uncertain (0.7500 -> 0.4375) · **Direction:** Negative
- **Tested:** Evaluate if the geometric alignment of fragment endpoints is a statistically valid filter for distinguishing true splits from false merges.
- **Conclusion:** A one-sided Mann-Whitney U test was performed, yielding a p-value of 0.3513, which is not statistically significant. Furthermore, while the mean cosine similarity was slightly higher for TPs (0.7514 vs 0.7087), the median was actually higher for FPs (0.8960 vs 0.8250). Belief therefore dropped from Leaning True (0.75) to Uncertain (0.44) (signed surprisal -0.487; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.3513; n up to 5
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 152. (Surprise 0.483) Refuted: data contradict the claim that human annotator identity biases U-Net error metrics.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 35 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3558) · **Direction:** Negative
- **Tested:** Investigate if topological error rates are biased by the ground-truth annotator or the specific morphologies assigned to them.
- **Conclusion:** - **Mean Split Rates:** The average split rate varied slightly among annotators, ranging from ~0.144 (JT) to ~0.226 (PP) fragments per 1000 µm. - **Statistical Test:** The Kruskal-Wallis H test yielded a test statistic of 8.4105 and a p-value of ~0.2978. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.36) (signed surprisal -0.483; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Kruskal-Wallis, Student's t-test; p = 0.2978
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 153. (Surprise 0.483) Refuted: data contradict the claim that image-intensity features (signal across the gap) add discriminative power.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 46 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3558) · **Direction:** Negative
- **Tested:** Compare the predictive power of absolute difference in node radius versus spatial distance in classifying true versus false reconnection proposals between fragment endpoints.
- **Conclusion:** The ROC analysis showed Spatial Distance achieving a moderate AUC of 0.7285, while Radius Difference performed marginally above random chance with an AUC of 0.5600. Despite the apparent difference in AUC scores, DeLong's test yielded a p-value of 0.9300, indicating that the difference in predictive power between the two metrics is not statistically significant given this sample, largely due to the extreme class imbalance (only 5 false reconnections). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.36) (signed surprisal -0.483; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** ROC-AUC, Student's t-test; p = 0.93
- **Statistical issues:** none
- **Logic issues:** none

### 154. (Surprise 0.483) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 47 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3558) · **Direction:** Negative
- **Tested:** Assess whether fluctuations in the `node_radius` within a single connected component can flag the presence of a merge error.
- **Conclusion:** Statistical testing via the Mann-Whitney U test produced a p-value of ~0.0549 (statistic = 18381.0). The merged fragments showed slightly higher standard deviations (Median: 0.0903 µm, Mean: 0.0929 µm) compared to pure fragments (Median: 0.0818 µm, Mean: 0.0843 µm). Belief therefore dropped from Leaning True (0.71) to Leaning False (0.36) (signed surprisal -0.483; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.0549; n up to 28
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 155. (Surprise 0.483) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 88 · **Belief:** Leaning False -> Leaning True (0.2917 -> 0.6442) · **Direction:** Positive
- **Tested:** Determine if geometric continuity (tangent alignment) is sufficient to correctly route skeletons through erroneous U-Net merge hubs.
- **Conclusion:** Statistical analysis revealed a significant difference in geometric continuity: True Continuations exhibited a mean angular deviation of 42.55° from a straight line (180°), whereas False Continuations had a much higher mean deviation of 80.78°. Welch's t-test confirmed the statistical significance of this difference with a p-value of 1.1485e-08. Belief therefore rose from Leaning False (0.29) to Leaning True (0.64) (signed surprisal +0.483; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, ROC-AUC, Student's t-test; p ≈ 1.1e-08
- **Statistical issues:** none
- **Logic issues:** none

### 156. (Surprise 0.446) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 23 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** To verify if local candidate density around an endpoint is a reliable proxy for false positive risk, informing a confidence penalty for connections in dense regions.
- **Conclusion:** The mean pair density score was 2.25 for TPs and 2.80 for FPs. A Mann-Whitney U test comparing the two distributions resulted in a p-value of 0.0766. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.0766
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 157. (Surprise 0.446) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 31 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** To investigate if morphological complexity is a primary driver of U-Net segmentation failures and resulting fragmentation.
- **Conclusion:** The Pearson correlation analysis between branching density and split rate resulted in an r-value of -0.4140 with a p-value of 0.078. Since the p-value is greater than 0.05, the correlation is not statistically significant. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Pearson, Student's t-test; p = 0.078
- **Statistical issues:** none
- **Logic issues:** none

### 158. (Surprise 0.446) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 46 · **Belief:** Uncertain -> Likely True (0.5417 -> 0.8281) · **Direction:** Positive
- **Tested:** To determine if radius continuity along the terminal segment of a fragment can distinguish artificial splits from natural leaf nodes.
- **Conclusion:** The script accurately loaded the dataset, identified fragment leaf nodes, and mapped them to the ground-truth (GT) graph to classify them into 'Split Endpoints' (N=3965) and 'True Terminations' (N=3416). The results indicate a highly significant difference (t=48.35, p≈0), with split endpoints showing an average gradient of 0.01223 µm/µm, meaning they taper sharply, while true terminations have a near-zero average gradient (-0.00041 µm/µm), indicating a constant radius. Belief therefore rose from Uncertain (0.54) to Likely True (0.83) (signed surprisal +0.446; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** two-sample t-test, Student's t-test; p ≈ 0; n up to 3965
- **Statistical issues:** none
- **Logic issues:** none

### 159. (Surprise 0.446) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 49 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Classify valid reconnections structurally and verify if the spatial gap distribution differs based on the connection topology.
- **Conclusion:** - **Spatial Gap Distances:** The spatial gaps for End-to-End splits averaged 28.53 µm (median: 13.34 µm). In contrast, End-to-Branch gaps averaged 25.28 µm (median: 8.18 µm). Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.854
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 160. (Surprise 0.446) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 5 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Determine if morphological complexity (branch density) is a strong geometric predictor of merge errors within automated fragments.
- **Conclusion:** While the branch density was slightly higher for merge-containing fragments (mean = 1.3251, median = 1.0636 branches/mm) compared to merge-free fragments (mean = 1.0846, median = 0.9004 branches/mm), the Mann-Whitney U test produced a p-value of ~0.319. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.319
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 161. (Surprise 0.446) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 11 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Evaluate the predictive power of 3D branch tangent alignment in distinguishing valid reconnections from false reconnections during split error correction.
- **Conclusion:** The mean dot product for valid pairs (0.7543) was slightly higher than for invalid pairs (0.7305), with 55.88% of valid pairs and 52.07% of invalid pairs having a dot product > 0.8. However, the Mann-Whitney U test yielded a p-value of 0.0565, which is slightly above the traditional 0.05 significance threshold. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.0565
- **Statistical issues:** none
- **Logic issues:** none

### 162. (Surprise 0.446) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 38 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Investigate the relationship between a U-Net fragment's morphological size and the physical extent of its split errors.
- **Conclusion:** The ANOVA test revealed a statistically significant difference in mean gap distances across the fragment length bins (F = 108.59, p < 0.001). While the mean gap size was hypothesized to be inversely proportional to fragment length, the data showed that fragments >5000 µm actually had a higher average gap distance (236.0 µm) compared to the <2000 µm (195.4 µm) and 2000-5000 µm (189.1 µm) bins. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** one-way ANOVA; p = 0.001
- **Statistical issues:** none
- **Logic issues:** none

### 163. (Surprise 0.446) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 65 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Determine if the difference in estimated neurite radius at the endpoints is significantly smaller for valid split pairs than for invalid nearby fragment pairs.
- **Conclusion:** The mean radius difference for valid pairs (0.4610 µm) was slightly higher than that for invalid pairs (0.3853 µm). Similarly, the median difference was higher for valid pairs (0.3984 µm vs 0.3569 µm). Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.6794
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 164. (Surprise 0.446) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 67 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Test if local orientation is a stronger predictor than spatial proximity for resolving competing reconnection proposals at a single endpoint.
- **Conclusion:** McNemar's test yielded a p-value of 1.0000, indicating no statistically significant difference between the two approaches, largely constrained by the extremely small sample size of valid mutual exclusivity scenarios. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** McNemar's, Student's t-test; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 165. (Surprise 0.446) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 76 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Test if local neurite radius consistency is a reliable intrinsic feature for detecting structural false merges without requiring image re-evaluations.
- **Conclusion:** The computed mean radius variance among converging branches was visibly higher for Merge Errors (0.0146 μm²) compared to True Branches (0.0088 μm²). However, Welch's t-test yielded a t-statistic of 1.6875 and a p-value of 0.3406 (p > 0.05). Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p = 0.05; n up to 38
- **Statistical issues:** none
- **Logic issues:** none

### 166. (Surprise 0.446) Refuted: data contradict the claim that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 79 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Analyze the spatial distribution of split errors along the Z-axis to determine if tissue depth exacerbates segmentation failures.
- **Conclusion:** A Pearson correlation analysis yielded an r-value of 0.0092 with a p-value of 0.7200, indicating that there is no statistically significant linear correlation between the Z-depth (imaging depth) and the size of the split gaps. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Pearson; p = 0.72
- **Statistical issues:** none
- **Logic issues:** none

### 167. (Surprise 0.446) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 80 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Evaluate if local tangent alignment is a sufficiently strong geometric discriminator to resolve mutually exclusive reconnection proposals at an endpoint without requiring image features.
- **Conclusion:** In these 2 cases, the Mean False Candidate Cosine Similarity (0.8180) was actually higher than the Mean True Candidate Cosine Similarity (0.6586). Due to the extremely low sample size (N=2), the statistical tests yielded non-significant results (Paired t-test p = 0.54, Wilcoxon p = 1.0). Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** paired t-test, Wilcoxon, Student's t-test; p = 1; n up to 2
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 168. (Surprise 0.446) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 92 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Assess whether certain neuron morphologies are inherently more prone to deep-learning segmentation fragmentation, which could inform adaptive proofreading confidence thresholds.
- **Conclusion:** The results yielded a Pearson correlation of r = -0.3777 (p = 0.1109) and a Spearman correlation of r = -0.2649 (p = 0.2731). Contrary to the hypothesis, the data suggests a weak to moderate negative correlation, indicating that densely branched neurons do not inherently suffer from higher fragmentation rates from the U-Net model; in fact, there is a slight tendency for them to fragment less. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman, Pearson, Student's t-test; p = 0.2731
- **Statistical issues:** none
- **Logic issues:** none

### 169. (Surprise 0.446) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 102 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Determine if anomalous fluctuations in the estimated skeleton radius can pinpoint the exact node where a false merge occurs.
- **Conclusion:** While the average radius variance was also visibly higher for merge boundaries (0.0065 vs 0.0017), the t-test for variance yielded a p-value of ~0.13, likely due to extreme outliers drastically inflating the sample variance and reducing the effective degrees of freedom in the Welch's calculation. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p = 0.001
- **Statistical issues:** none
- **Logic issues:** none

### 170. (Surprise 0.446) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 117 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Correlate ground-truth neuron branch density with the normalized split rate.
- **Conclusion:** A Pearson correlation analysis yielded a coefficient of r = -0.3986 with a p-value of 0.091. This indicates a weak-to-moderate negative correlation that is not statistically significant at the standard alpha = 0.05 level. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Pearson, Student's t-test; p = 0.091
- **Statistical issues:** none
- **Logic issues:** none

### 171. (Surprise 0.446) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 145 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Demonstrate that dynamically refreshing graph component state (agentic replanning) prevents true reconnections from being rejected due to stale topology constraints.
- **Conclusion:** The McNemar's test yielded a p-value of 0.0625, indicating no statistically significant difference between the true positive recovery rate of the two pipelines. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** McNemar's, Student's t-test; p = 0.0625
- **Statistical issues:** none
- **Logic issues:** none

### 172. (Surprise 0.446) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 151 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Determine if unpredicted splits correlate with areas of high topological complexity (branching) in the underlying ground truth.
- **Conclusion:** The statistical analysis revealed that leaf-associated GT nodes had a mean degree of 1.533 (Max: 5), whereas internal-associated GT nodes had a mean degree of 2.001 (Max: 3). A Mann-Whitney U test indicated a highly significant difference between the two distributions (p-value ~ 0.0000). Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 0
- **Statistical issues:** none
- **Logic issues:** none

### 173. (Surprise 0.446) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 157 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Analyze the spatial gap distance of potential reconnection proposals to see if proximity disproportionately favors false positives.
- **Conclusion:** True reconnections had a mean gap distance of 8.52 µm, while the 3 False reconnections averaged 8.82 µm (Welch's t-test p-value = 0.910, KS test p-value = 0.228). The extreme class imbalance (n=431 vs. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Welch's t-test, Kolmogorov-Smirnov, Student's t-test; p = 0.228; n up to 431
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 174. (Surprise 0.446) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 161 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Test if biological neuron complexity is a predictive confounder for deep-learning segmentation failures.
- **Conclusion:** The statistical analysis yielded a Pearson correlation of 0.2252 (1-sided p-value = 0.1770) and a Spearman correlation of -0.1000. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman, Pearson; p = 0.177
- **Statistical issues:** none
- **Logic issues:** none

### 175. (Surprise 0.446) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 166 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Test whether angular alignment between terminal branches of fragment leaves is a significantly better discriminator for true positive reconnections than spatial proximity.
- **Conclusion:** Because spatial proximity (within 30 µm) alone yielded a remarkably high baseline precision (~99%), the Chi-squared test returned a non-significant p-value of 0.2150. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p = 0.215
- **Statistical issues:** none
- **Logic issues:** none

### 176. (Surprise 0.446) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 184 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Determine if automated pruning of short terminal branches can cleanly remove false-positive topologies without sacrificing true neurite cable.
- **Conclusion:** The Chi-Square test produced a p-value of 1.0, indicating no statistically significant difference between the two groups. Consequently, the hypothesis that short terminal branches are predominantly false positives cannot be robustly validated due to the lack of statistical power (n=7 for short branches). Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); stepped ROC curve from sparse negatives
- **Verdict:** Sound
- **Test:** chi-square, Student's t-test; p = 1; n up to 7
- **Statistical issues:** none
- **Logic issues:** none

### 177. (Surprise 0.446) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 191 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4219) · **Direction:** Negative
- **Tested:** Investigate if tracking graph-cycle-prevention events during split correction can serve as an agentic flag to dynamically identify undiscovered false merges.
- **Conclusion:** Although the false-merge rate was roughly four times higher in cycle-implicated components, a Chi-square test of independence yielded a p-value of 0.1805. Because the p-value is greater than the standard significance level (alpha = 0.05), the association is not statistically significant in this sample. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.42) (signed surprisal -0.446; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square, Student's t-test; p = 0.1805
- **Statistical issues:** none
- **Logic issues:** none

### 178. (Surprise 0.430) Refuted: data contradict the claim that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 65 · **Belief:** Leaning True -> Leaning False (0.7083 -> 0.3942) · **Direction:** Negative
- **Tested:** To investigate if the ratio of component sizes (number of nodes or path length) between two candidate fragments provides a structural bias for true versus false split reconnections.
- **Conclusion:** The results showed that True reconnections had a lower mean size ratio (0.3392), suggesting a tendency for small fragments to link to larger structures, compared to the mean of 0.5786 for False candidates. Despite this apparent difference, a Kolmogorov-Smirnov test produced a non-significant p-value of 0.1534. Belief therefore dropped from Leaning True (0.71) to Leaning False (0.39) (signed surprisal -0.430; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Kolmogorov-Smirnov; p = 0.1534; n up to 5
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 179. (Surprise 0.408) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 15 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4464) · **Direction:** Negative
- **Tested:** Test radius continuity as a heuristic for resolving mutually exclusive reconnection proposals at a single endpoint.
- **Conclusion:** The experiment was successfully executed. The dataset was correctly loaded from the current directory, and dependencies were installed on the fly. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.45) (signed surprisal -0.408; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** binomial test; p = 1
- **Statistical issues:** none
- **Logic issues:** none

### 180. (Surprise 0.398) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 198 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.4531) · **Direction:** Negative
- **Tested:** Determine if simple proximity checks inherently bias proofreading toward false merges in dense regions, motivating the need for reversible decisions and rollback mechanisms in agentic frameworks.
- **Conclusion:** The valid proposals had a median gap length of 9.70 µm, while the invalid proposals had a median of 13.49 µm. The Mann-Whitney U test yielded a p-value of 0.325, indicating no statistically significant difference between the two distributions. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.45) (signed surprisal -0.398; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.325; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 181. (Surprise 0.395) Supported: evidence confirms that image-intensity features (signal across the gap) add discriminative power.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 84 · **Belief:** Leaning True -> Likely True (0.6250 -> 0.9135) · **Direction:** Positive
- **Tested:** Test if merge errors are driven by low signal-to-noise ratio (faint signal) rather than additive blooming (bright signal) to inform the image-encoder's attention mechanism.
- **Conclusion:** - **Mean Intensities (Mean ± Std):** Merge nodes demonstrated a mean maximum intensity of 134.59 (± 158.30), whereas background nodes had a notably higher mean maximum intensity of 214.36 (± 182.46). - **Statistical Comparison:** A Welch's t-test confirmed a highly statistically significant difference between the two distributions (t-statistic = -7.1670, p-value = 1.5679e-12). Belief therefore rose from Leaning True (0.62) to Likely True (0.91) (signed surprisal +0.395; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p ≈ 1.6e-12
- **Statistical issues:** none
- **Logic issues:** none

### 182. (Surprise 0.365) Refuted: data contradict the claim that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 115 · **Belief:** Leaning False -> Likely False (0.3750 -> 0.1406) · **Direction:** Negative
- **Tested:** Test whether U-Net fragmentation (split errors) occurs more frequently in highly tortuous (curvy) regions of the neurite.
- **Conclusion:** - **Statistical Comparison:** - Average tortuosity for Split segments: 1.0542 - Average tortuosity for Terminal segments: 1.0713 - Mann-Whitney U test p-value: 3.53e-15 **Conclusion:** The results are highly statistically significant (p < 0.05), but they *contradict* the original hypothesis. Belief therefore dropped from Leaning False (0.38) to Likely False (0.14) (signed surprisal -0.365; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 183. (Surprise 0.365) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 172 · **Belief:** Leaning True -> Likely True (0.6250 -> 0.8594) · **Direction:** Positive
- **Tested:** Determine if proximity to soma locations is a primary driver of U-Net merge errors due to localized high neurite density.
- **Conclusion:** A Chi-squared test produced a statistic of 50.4320 and a p-value of 1.23e-12, confirming that there is a statistically significant increase in merge error density in regions proximal to the somas. Belief therefore rose from Leaning True (0.62) to Likely True (0.86) (signed surprisal +0.365; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square, Student's t-test; p ≈ 1.2e-12
- **Statistical issues:** none
- **Logic issues:** none

### 184. (Surprise 0.333) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 81 · **Belief:** Leaning True -> Uncertain (0.7917 -> 0.5781) · **Direction:** Negative
- **Tested:** Evaluate if angular alignment of fragment endpoints can effectively filter false reconnection proposals during split correction.
- **Conclusion:** The mean angular alignment (cosine similarity) for True Connections was 0.8719 compared to 0.7123 for False Connections. However, a Mann-Whitney U test yielded a p-value of 0.2379, indicating that the difference is not statistically significant. Belief therefore dropped from Leaning True (0.79) to Uncertain (0.58) (signed surprisal -0.333; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.2379; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 185. (Surprise 0.325) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 129 · **Belief:** Uncertain -> Leaning True (0.4167 -> 0.6250) · **Direction:** Positive
- **Tested:** Quantify whether neuronal fragmentation rates are spatially biased by their topological distance from the soma, which could inform region-adaptive proofreading sweeps.
- **Conclusion:** However, while the p-value indicates statistical significance, the R-squared value is extremely low (0.0066) and the slope is very shallow (-0.000003 splits/mm per µm). Belief therefore rose from Uncertain (0.42) to Leaning True (0.62) (signed surprisal +0.325; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record; p = 0.00908
- **Statistical issues:** none
- **Logic issues:** none

### 186. (Surprise 0.307) Supported: evidence confirms that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 1 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Investigate and statistically quantify the directional bias of topological split errors relative to the Z-axis, and dynamically formulate data-driven algorithmic recommendations based on the empirical angular distribution rather than preconceived assumptions.
- **Conclusion:** The generated plot and KS test results (KS=0.1000, p-value=5.067e-04) further corroborate these conclusions, successfully completing the objective. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov; p ≈ 0.00051
- **Statistical issues:** none
- **Logic issues:** none

### 187. (Surprise 0.307) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 7 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** To quantify the spatial co-occurrence of split and merge errors and justify a coupled proofreading architecture that loops between split and merge correction.
- **Conclusion:** The two-sample Kolmogorov-Smirnov test yielded a KS statistic of 0.0540 and a highly significant p-value of 3.5151e-05, rejecting the null hypothesis. The mean distance for Artificial Splits (2,968.36 µm) was found to be lower than that of True Terminals (3,340.30 µm). Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Kolmogorov-Smirnov; p ≈ 3.5e-05
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 188. (Surprise 0.307) Supported: evidence confirms that morphological features mark U-Net merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 15 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Determine if merge correction should prioritize a small number of highly-fused components rather than scanning all fragments equally.
- **Conclusion:** The experiment was successfully executed and met all the objectives. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 189. (Surprise 0.307) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 28 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Quantify the relationship between local neurite tortuosity and the likelihood of U-Net fragmentation.
- **Conclusion:** The mean tortuosity at split error sites was 1.0499 compared to 1.0416 at continuous sites, and the median tortuosity was 1.0344 versus 1.0314, respectively. Both the two-sample t-test (t = 8.8594, p-value = 8.7931e-19) and the Kolmogorov-Smirnov test (KS = 0.0531, p-value = 1.1232e-12) confirm that these distributions are statistically distinct. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** two-sample t-test, Kolmogorov-Smirnov, Student's t-test; p ≈ 1.1e-12
- **Statistical issues:** none
- **Logic issues:** none

### 190. (Surprise 0.307) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 30 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Test if automated split errors are immediately preceded by a geometrical drift from the true biological centerline, indicating local prediction uncertainty.
- **Conclusion:** The results show that fragment endpoints have a mean drift of 1605.61 µm (median: 1159.68 µm), while internal nodes have a mean drift of 1470.53 µm (median: 1001.25 µm). A Mann-Whitney U test confirmed that this difference is highly statistically significant (p-value = 8.265e-66), supporting the hypothesis that U-Net predictions exhibit significantly higher spatial deviation from the true biological centerline at the point where they break/split. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 8.3e-66
- **Statistical issues:** none
- **Logic issues:** none

### 191. (Surprise 0.307) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 37 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Evaluate the discriminative power of endpoint angular alignment (geometry) versus simple Euclidean distance to inform agentic mutual exclusivity reasoning at split gaps.
- **Conclusion:** The ROC analysis and AUC metrics strongly confirmed the hypothesis: Cosine Similarity (Angular Alignment) achieved a significantly higher AUC of 0.8985 compared to Inverse Euclidean Distance, which scored an AUC of 0.7292. The ROC curves demonstrated that angular alignment can identify nearly 70% of true matches without a single false positive, whereas distance-alone plateaus at 50%. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** ROC-AUC
- **Statistical issues:** none
- **Logic issues:** none

### 192. (Surprise 0.307) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 40 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Determine if fragment endpoints (split errors) are spatially correlated with the locations of true biological bifurcations.
- **Conclusion:** The Kolmogorov-Smirnov test yielded a KS statistic of 0.0941 and a highly significant p-value of 1.9195e-29. Furthermore, the median distance to a branch point for split errors was 108.63 µm, compared to 140.12 µm for the uniform control. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Kolmogorov-Smirnov; p ≈ 1.9e-29
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 193. (Surprise 0.307) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 54 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Test the spatial co-occurrence of split and merge errors to determine if correcting a merge can act as a high-confidence prior for nearby split-corrections.
- **Conclusion:** The statistical findings solidly validate the hypothesis: the KS test returned a significant p-value of 5.2154e-05 (KS Stat = 0.1249), demonstrating that split and merge errors do not distribute independently. Specifically, the median distance from a split error to the nearest merge site was 809.72 µm, noticeably closer than the control median of 1016.66 µm. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov; p ≈ 5.2e-05
- **Statistical issues:** none
- **Logic issues:** none

### 194. (Surprise 0.307) Supported: evidence confirms that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 55 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Determine if the global geometric tortuosity of a fragment component is a reliable flag for identifying likely merge errors.
- **Conclusion:** The log-transformed tortuosity (cable length / bounding box diagonal) for clean components had a mean of 0.1512 (std: 0.1394), whereas merged components showed a significantly higher mean of 0.4266 (std: 0.3424). Welch's t-test confirmed the statistical significance of this difference (t-statistic = 3.5969, p-value = 1.9205e-03). Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p = 0.00192
- **Statistical issues:** none
- **Logic issues:** none

### 195. (Surprise 0.307) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 61 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Quantify the proportion of valid split connections that require leaf-to-internal linking versus standard leaf-to-leaf linking.
- **Conclusion:** The experiment successfully executed all steps, identifying and mapping 39,826 fragment leaf nodes to ground truth components. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 196. (Surprise 0.307) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 68 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Validate orientation agreement as a robust geometric feature for resolving mutually exclusive reconnection proposals.
- **Conclusion:** - **Cosine Similarity (Mean ± Std):** Valid reconnections exhibited a mean cosine similarity of -0.4726 ± 0.5622. False reconnections exhibited a mean cosine similarity of 0.2478 ± 0.6295. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U, Student's t-test; p ≈ 0.00057
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 197. (Surprise 0.307) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 70 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Determine if neurite morphology complexity (tortuosity) is a causal predictor of segmentation fragmentation.
- **Conclusion:** The statistical analysis using the Mann-Whitney U test yielded a highly significant result (U = 349,031,936.00, p-value = 9.2626e-17), demonstrating a clear difference between the two groups. Furthermore, the mean split rate for highly tortuous segments (0.25 splits/mm) was 2.5 times higher than that of low tortuosity segments (0.10 splits/mm). Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p ≈ 9.3e-17
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 198. (Surprise 0.307) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 73 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Quantify the local spatial density of fragment endpoints (leaves) around true terminations versus split gaps to assess if splits cluster in 'shatter zones'.
- **Conclusion:** The statistical analysis utilizing the Mann-Whitney U test yielded a test statistic of 9,251,004.0 and an extremely significant p-value of 2.5958e-275. The mean endpoint density was substantially higher for Split Leaves (0.54 neighbors) compared to True Terminations (0.09 neighbors). Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U, Student's t-test; p ≈ 2.6e-275
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 199. (Surprise 0.307) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 75 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Test the hypothesis that topological errors are coupled, and that highly complex/messy local image regions simultaneously trigger both merge and split failures.
- **Conclusion:** - **Split Density (Mean):** Fragments with merge errors exhibited a mean split density of 0.001645 valid leaves/µm. In contrast, fragments without merge errors had a significantly lower mean split density of 0.000444 valid leaves/µm. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** median outcome is zero; effect driven by outliers
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 3.3e-09
- **Statistical issues:** none
- **Logic issues:** none

### 200. (Surprise 0.307) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 77 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Investigate the relationship between local graph density and the physical scale of segmentation errors to inform dynamic search space constraints during proofreading.
- **Conclusion:** The statistical analysis revealed a strong and highly significant negative correlation (Pearson r = -0.6086, p = 1.26e-138; Spearman rho = -0.6668, p = 1.21e-175). Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman, Pearson; p ≈ 1.2e-175
- **Statistical issues:** none
- **Logic issues:** none

### 201. (Surprise 0.307) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 78 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Determine if valid split errors are spatially clustered around ground-truth branching nodes compared to continuous U-Net segments.
- **Conclusion:** The median distance to the nearest true bifurcation for split gaps was 70.98 µm, compared to 151.44 µm for continuous segments. A Mann-Whitney U test confirmed this difference is highly statistically significant (Statistic = 84838.0, p-value = 2.9857e-12). Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p ≈ 3e-12
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 202. (Surprise 0.307) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 80 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Test if U-Net fragment size is a robust intrinsic proxy for the presence of merge errors.
- **Conclusion:** A Mann-Whitney U test yielded a highly significant p-value of 0.0025 (Statistic = 34192.00), demonstrating that merged fragments are statistically significantly longer than clean ones. The generated Box and Violin plots visually corroborate these findings, showing that while clean fragments are tightly grouped at shorter lengths, merged fragments exhibit extreme upper-bound outliers and a significantly higher median. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.0025
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 203. (Surprise 0.307) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 82 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Determine if the frequency of U-Net split errors is positively correlated with the topological branch depth from the neuron's soma.
- **Conclusion:** The statistical analysis revealed a strong and highly significant positive correlation (Pearson r = 0.8042, p-value = 6.95e-10; Spearman rho = 0.8543, p-value = 4.65e-12) between the topological branch order and the split rate (splits/mm). Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman, Pearson; p ≈ 4.6e-12
- **Statistical issues:** none
- **Logic issues:** none

### 204. (Surprise 0.307) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 85 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Quantify how often resolving a true split error would create a topological cycle in the U-Net fragments graph due to the presence of uncorrected merge errors.
- **Conclusion:** The experiment was successfully executed, correctly overcoming previous import issues by implementing a fully robust `MockUnpickler` that safely intercepted all unknown classes. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** test type not extractable from record
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 205. (Surprise 0.307) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 86 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Investigate if local ground-truth morphological complexity (branch density) is a primary spatial driver of merge errors.
- **Conclusion:** Statistical analysis via the Kolmogorov-Smirnov test yielded a KS Statistic of 0.0973 and a highly significant p-value of 4.6304e-19, confirming that the two distributions differ significantly. The mean branch density for merge sites (0.51) was nearly double that of the control group (0.27). Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Kolmogorov-Smirnov; p ≈ 4.6e-19
- **Statistical issues:** statistical significance driven by very large N; effect size negligible (|r|<0.1) — significance is a large-sample artifact rather than a scientifically meaningful effect; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 206. (Surprise 0.307) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 92 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Determine if valid split gaps occur significantly closer to ground-truth topological branch points than expected by chance.
- **Conclusion:** Statistical analysis using the Mann-Whitney U test yielded a test statistic of 554274.0 and an overwhelmingly significant p-value of ~3.63e-73. The mean distance to the nearest branching point for valid splits is 124.73 µm, compared to 239.99 µm for the baseline baseline. Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p ≈ 3.6e-73
- **Statistical issues:** none
- **Logic issues:** none

### 207. (Surprise 0.307) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 93 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Quantify structural asymmetry at split sites to determine if symmetric angular agreement heuristics are overly penalizing valid reconnections.
- **Conclusion:** The statistical analysis strongly supported the hypothesis: true split gaps demonstrated a significantly higher mean tangent asymmetry (0.3144) compared to the unbroken control edges (0.0721), with the Mann-Whitney U test yielding a p-value of essentially 0 (highly significant). Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 0
- **Statistical issues:** none
- **Logic issues:** none

### 208. (Surprise 0.307) Supported: evidence confirms that the proposed structural heuristic holds for U-Net proofreading.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 94 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.9327) · **Direction:** Positive
- **Tested:** Investigate the relationship between the physical cable length of a U-Net fragment and its probability of corresponding to a valid ground-truth structure.
- **Conclusion:** The statistical analysis demonstrated that mapped (valid) fragments possess a significantly higher mean length (4652.57 µm) compared to unmapped (noise) fragments (1563.21 µm). A Welch's t-test confirmed this difference is statistically highly significant (t-statistic = 6.1600, p-value = 1.0744e-09). Belief therefore rose from Leaning True (0.71) to Likely True (0.93) (signed surprisal +0.307; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, logistic regression, Student's t-test; p ≈ 1.1e-09
- **Statistical issues:** none
- **Logic issues:** none

### 209. (Surprise 0.284) Refuted: data contradict the claim that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 40 · **Belief:** Leaning False -> Likely False (0.2917 -> 0.1094) · **Direction:** Negative
- **Tested:** Assess whether the neurite radius at an endpoint can serve as a confidence prior for generating valid split-correction proposals.
- **Conclusion:** The descriptive statistics show the true splits had a mean radius of 1.3045 µm (median = 1.2490 µm), while false splits had a mean radius of 1.2250 µm (median = 1.0000 µm). A Mann-Whitney U test yielded a p-value of 0.4526, indicating no statistically significant difference between the radii of true and false splits. Belief therefore dropped from Leaning False (0.29) to Likely False (0.11) (signed surprisal -0.284; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.4526; n up to 11
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 210. (Surprise 0.284) Supported: evidence confirms that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 4 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** To evaluate if the local tortuosity of a reconnected path is a viable metric to prevent artifactual, sharp bends in the proofread skeleton.
- **Conclusion:** The tortuosity analysis revealed that valid connections (TP) have a noticeably lower mean tortuosity (1.3872) compared to invalid connections (FP, mean = 2.3972). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p ≈ 0.00026; n up to 2
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 211. (Surprise 0.284) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 5 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Demonstrate that coupling merge and split correction is necessary, as correcting a merge directly creates the endpoints needed for valid split reconnections.
- **Conclusion:** The experiment successfully executed and validated the hypothesis. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 212. (Surprise 0.284) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 6 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** To evaluate if erroneous fragment fusions occur disproportionately in dense neuropil where the local branching density is high.
- **Conclusion:** The results demonstrate a significant difference: merge regions exhibit a mean branching density of 0.62, while safe regions have a mean of 0.25. An independent Welch's t-test yielded a t-statistic of 7.2354 and a p-value of 9.8726e-13. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, two-sample t-test, Student's t-test; p ≈ 9.9e-13
- **Statistical issues:** none
- **Logic issues:** none

### 213. (Surprise 0.284) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 15 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Test if local endpoint density correlates with the likelihood of successful topology recovery, indicating that proofreading algorithms should dynamically prioritize these hubs.
- **Conclusion:** A Chi-square test confirmed a highly statistically significant correlation (chi2 = 1582.33, p-value = 0.0). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p ≈ 0
- **Statistical issues:** none
- **Logic issues:** none

### 214. (Surprise 0.284) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 19 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Determine if the gradient of the `node_radius` attribute along a skeleton can serve as a local image-evidence feature to detect and flag topological merge boundaries.
- **Conclusion:** The statistical analysis revealed that the mean local radius gradient at merge boundaries was 0.0498 µm, compared to just 0.0096 µm for normal nodes. The median radius gradient was also twice as high at merge boundaries (0.0020 µm vs 0.0010 µm), and the standard deviation was significantly larger, indicating high local variance. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 5.6e-44
- **Statistical issues:** none
- **Logic issues:** none

### 215. (Surprise 0.284) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 35 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** To determine if split errors are spatially correlated with neurite branch points, implying morphological complexity induces segmentation breaks.
- **Conclusion:** Analyzing 39,826 leaf nodes and an equal number of randomly sampled middle nodes revealed a stark contrast: leaf nodes were found significantly closer to branching nodes (mean = 217.40 µm, median = 151.20 µm) compared to middle nodes (mean = 266.92 µm, median = 188.93 µm). A Mann-Whitney U test confirmed this difference is highly statistically significant (U = 702045331.0, p-value = 2.08e-173). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 2.1e-173; n up to 266
- **Statistical issues:** none
- **Logic issues:** none

### 216. (Surprise 0.284) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 41 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** To demonstrate that coupling merge and split correction is necessary because merge errors mask true fragment endpoints.
- **Conclusion:** The spatial chunking and paired t-test confirmed this result with a t-statistic of 3.2136 and a highly significant p-value of 0.00147. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** paired t-test, Student's t-test; p = 0.00147
- **Statistical issues:** none
- **Logic issues:** none

### 217. (Surprise 0.284) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 1 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Identify and characterize structural 'cycles' or auto-merges within single predicted fragments that mimic false reconnections.
- **Conclusion:** The mean radius variance for the kissing loops (0.013397) was substantially greater than that of the baseline paths (0.002571). The median values confirmed this pattern (0.007474 vs 0.000736), indicating that analyzing radius variability could be a strong heuristic for detecting auto-merge errors. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 218. (Surprise 0.284) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 17 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Test if component size acts as a reliable prior for distinguishing genuine neuron segments from noise-induced short fragments.
- **Conclusion:** For valid reconnections, the mean minimum component size was 826.23 nodes (median 402.50). For invalid reconnections, the mean minimum component size was 373.12 nodes (median 271.00). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 1.4e-27
- **Statistical issues:** none
- **Logic issues:** none

### 219. (Surprise 0.284) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 20 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Investigate whether merge errors concentrate near structural branching points in the automated reconstruction.
- **Conclusion:** The mean distance to the nearest branch node was 282.21 µm for boundary nodes and 315.19 µm for baseline nodes. The two-sample t-test produced a t-statistic of -6.9887 and a p-value of 2.8822e-12. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** two-sample t-test, Student's t-test; p ≈ 2.9e-12
- **Statistical issues:** none
- **Logic issues:** none

### 220. (Surprise 0.284) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 36 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Determine if U-Net split errors occur in localized 'hotspots' (e.g., due to regional imaging artifacts or dense crossing regions) rather than being uniformly distributed.
- **Conclusion:** The results from the Kolmogorov-Smirnov test (KS Statistic = 0.2411, p-value = 4.0716e-19) indicate a highly significant difference between the two distributions. The true split gaps have a substantially smaller mean (634.00 µm) and median (480.64 µm) nearest-neighbor distance compared to the random null distribution (Mean = 920.95 µm, Median = 781.18 µm). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov; p ≈ 4.1e-19; n up to 920
- **Statistical issues:** none
- **Logic issues:** none

### 221. (Surprise 0.284) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 41 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Analyze the correlation between neurite radius and the occurrence of split errors by comparing endpoint radii to their parent fragment's mean radius.
- **Conclusion:** The code correctly installed dependencies, loaded the `fragments_graph`, computed the global mean radius versus the mean endpoint radius for 10,172 eligible U-Net fragments, and performed a paired t-test. The mean difference was -0.4382 µm. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** paired t-test, Student's t-test; p = 0.001
- **Statistical issues:** none
- **Logic issues:** none

### 222. (Surprise 0.284) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 45 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Investigate how the geometry of a split error changes based on its proximity to neuron branching points, informing dynamic search cone generation.
- **Conclusion:** The angular deviation from a straight line was significantly higher for gaps near bifurcations (mean = 83.99°, median = 83.35°) compared to backbone gaps (mean = 41.94°, median = 35.02°). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** two-sample t-test, Student's t-test; p = 0.001; n up to 83
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 223. (Surprise 0.284) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 51 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Compare the discriminative power of tangent orientation versus Euclidean distance for resolving competing reconnection proposals.
- **Conclusion:** The statistical analysis demonstrated that the tangent angular difference (AUC = 0.9657) is a significantly stronger predictor for resolving competing reconnection proposals compared to raw Euclidean distance (AUC = 0.7076). Interestingly, true connections exhibited a mean angle of ~144 degrees, reflecting the fact that correctly paired incoming and outgoing neurite branches naturally point in opposing outward directions, resulting in large angular differences. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** ROC-AUC; p ≈ 0.0001
- **Statistical issues:** none
- **Logic issues:** none

### 224. (Surprise 0.284) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 55 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Test if sudden fluctuations in predicted neurite radius can serve as a geometric signature for localizing merge errors.
- **Conclusion:** The statistical comparison using Welch's t-test on log-transformed variances yielded a highly significant result (t-statistic = 10.4368, p-value = 1.4687e-23). The mean log-variance at merge boundaries (-9.8775) was significantly higher than in pure fragments (-13.0873), concluding that merge error boundaries exhibit significantly higher local variance in neurite radius and that this metric can serve as a geometric signature for localizing merge errors. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p ≈ 1.5e-23
- **Statistical issues:** none
- **Logic issues:** none

### 225. (Surprise 0.284) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 61 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Challenge the standard assumption that split correction solely requires matching endpoint to endpoint, quantifying the need for 'dead-end' T-junction proposals.
- **Conclusion:** The subsequent one-sample proportion z-test (z-statistic = 39.99, p-value ≈ 0.0) provides overwhelming statistical evidence to reject the null hypothesis. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** test type not extractable from record; p ≈ 0
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 226. (Surprise 0.284) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 62 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** To evaluate if localized spikes or high variance in predicted neurite radius can serve as a reliable geometric signature for detecting and rolling back false merges.
- **Conclusion:** Local features for a 20-node window around these edges were extracted, specifically the standard deviation and max-to-median ratio of the neurite radius. The results show that merge interfaces exhibited significantly higher radius variance (mean = 0.0625 vs 0.0299 for controls) and larger max/median ratios (mean = 1.0286 vs 1.0182 for controls). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p ≈ 8.8e-10
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 227. (Surprise 0.284) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 72 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Investigate whether valid split errors occur primarily due to genuine physical thinning of the cable (fading signal) rather than random stochastic failures in thick cables.
- **Conclusion:** It then successfully compared the estimated neurite radius at these terminal leaves against the mean internal radius of their parent components. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** paired t-test, Wilcoxon, Student's t-test; p ≈ 0.0001
- **Statistical issues:** none
- **Logic issues:** none

### 228. (Surprise 0.284) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 86 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Test if local endpoint density can serve as a contextual meta-feature that shifts dynamically as edits accumulate, signaling regions where single-pass greedy reconnections are likely to fail.
- **Conclusion:** The statistical analysis revealed a highly significant difference (t-statistic = 91.99, p-value < 1e-10). Split centers exhibited a much higher mean endpoint density (2.29 ± 0.75) compared to continuous centers (0.16 ± 0.42). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Student's t-test; p ≈ 1e-10
- **Statistical issues:** none
- **Logic issues:** none

### 229. (Surprise 0.284) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 88 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Determine the graph-topological signature of merge errors to guide targeted, reversible un-merging algorithms.
- **Conclusion:** Analysis of the node degrees revealed a strong correlation between merge errors and structural branch points (nodes with a degree of 3 or higher). A Chi-square goodness-of-fit test confirmed that this deviation from the global baseline is highly statistically significant (Chi-square = 574.30, p-value = 6.51e-127). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** chi-square, Student's t-test; p ≈ 6.5e-127
- **Statistical issues:** none
- **Logic issues:** none

### 230. (Surprise 0.284) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 95 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Test the spatial correlation between merge errors and split errors to determine if detecting merges can guide where to search for valid reconnections.
- **Conclusion:** Results: - Mean leaf density (True Branches): 0.0068 unconnected leaves within 20 µm - Mean leaf density (Merge Errors): 0.1111 unconnected leaves within 20 µm The non-parametric Mann-Whitney U test returned a highly significant p-value of 2.1025e-13, confirming that branch points representing merge errors have a statistically significantly higher local density of unconnected leaf nodes compared to valid branches. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Welch's t-test, Student's t-test; p ≈ 2.1e-13
- **Statistical issues:** none
- **Logic issues:** none

### 231. (Surprise 0.284) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 96 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Assess how local proposal graph density impacts precision, testing if cycles inherently indicate a high false-positive rate due to anatomical constraints.
- **Conclusion:** The chi-square test yielded a statistic of 12.7389 with a p-value of 0.000358, which is statistically significant (p < 0.05). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 232. (Surprise 0.284) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 101 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Evaluate the synergistic effect of coupling merge correction (fragment splitting) before split correction (fragment joining).
- **Conclusion:** The experiment successfully tested the hypothesis that performing merge correction before split correction increases the recall of true-positive reconnections. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 233. (Surprise 0.284) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 103 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** To determine if U-Net fragmentation is disproportionately caused by anatomical bifurcations, validating if split-correction models should apply different priors near known branching structures.
- **Conclusion:** Ground-truth (GT) branching nodes (N=7,320) were correctly identified alongside an equal sample size of isolated control nodes (degree=2, ≥50 µm from any branch). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p ≈ 6.1e-39; n up to 7320
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 234. (Surprise 0.284) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 106 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Verify the benefit of an agentic, multi-pass graph update system by showing that early, easy decisions clean up the feature space for harder decisions.
- **Conclusion:** Although the absolute average reduction per endpoint was small (0.0004), the Wilcoxon signed-rank test confirmed that this decrease was statistically significant (p-value = 1.4911e-03, statistic = 0.0). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Wilcoxon, Student's t-test; p = 0.001491
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 235. (Surprise 0.284) Supported: evidence confirms that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 108 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Investigate if image anisotropy biases the orientation of automated reconstruction breaks.
- **Conclusion:** The mean absolute angle for biological edges was 61.38 degrees, heavily skewed toward the XY plane (near 90 degrees). In contrast, the split gaps had a significantly lower mean absolute angle of 59.29 degrees, indicating a bias toward the Z-axis. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov; p ≈ 0.00035
- **Statistical issues:** none
- **Logic issues:** none

### 236. (Surprise 0.284) Supported: evidence confirms that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 113 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Quantify the collateral damage of enforcing cycle-free graphs during greedy split correction by measuring how many True Positive edges are rejected to prevent cycles.
- **Conclusion:** The experiment successfully quantified the collateral damage of enforcing cycle-prevention during a greedy split-correction algorithm. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 237. (Surprise 0.284) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 114 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Investigate if the combined size (node count) of fragments involved in a reconnection proposal can serve as a predictive feature for the validity of the split.
- **Conclusion:** The results showed a stark difference in combined fragment sizes: True Pairs had a mean combined size of 13,058.34 nodes (median: 2,912.00 nodes), whereas False Pairs had a mean combined size of 1,932.99 nodes (median: 914.50 nodes). A Mann-Whitney U test yielded a p-value of approximately 2.00e-58, indicating a highly significant statistical difference between the two distributions. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p ≈ 2e-58
- **Statistical issues:** none
- **Logic issues:** none

### 238. (Surprise 0.284) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 135 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Demonstrate that relying purely on Euclidean distance for split correction is suboptimal in dense regions, motivating the use of multi-feature joint reasoning (mutual exclusivity) at competitive endpoints.
- **Conclusion:** A one-sided binomial test evaluating if this disagreement rate is significantly greater than a 10% baseline yielded a highly significant p-value of 4.9237e-09. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** binomial test; p ≈ 4.9e-09
- **Statistical issues:** none
- **Logic issues:** none

### 239. (Surprise 0.284) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 136 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Test whether topological bifurcations in neuron morphology are systematic failure points (hotspots for splits) for the U-Net segmentation.
- **Conclusion:** The Chi-Square test for independence yielded a test statistic of 886.7250 and an extremely low p-value (7.5452e-195), indicating a highly significant statistical relationship between the node morphology type (bifurcation vs. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square, Student's t-test; p ≈ 7.5e-195
- **Statistical issues:** none
- **Logic issues:** none

### 240. (Surprise 0.284) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 139 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Test if the cable length of a U-Net fragment inversely correlates with the reliability of its endpoint tangent vectors as a geometric signature for split correction.
- **Conclusion:** The short fragments exhibited a median angular error of 7.34°, compared to 6.60° for long fragments. The Mann-Whitney U test yielded a test statistic of 7049298.0 and a p-value of 6.37e-09. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.001
- **Statistical issues:** none
- **Logic issues:** none

### 241. (Surprise 0.284) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 143 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Quantify whether local fragment crowding is a statistically significant spatial precursor to merge errors.
- **Conclusion:** Visual analysis of the generated violin plots provides further nuance: for both groups, the median density is overwhelmingly concentrated at 1.0, indicating that most segments exist in relatively uncrowded spaces. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 3.4e-06
- **Statistical issues:** none
- **Logic issues:** none

### 242. (Surprise 0.284) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 144 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Validate that tangent orientation agreement is a robust geometric prior for prioritizing valid split reconnections.
- **Conclusion:** The results strongly validate the hypothesis: True Positives exhibited highly anti-parallel outward tangents (mean cosine similarity of -0.8077, concentrated near -1.0), whereas False Positives exhibited positive alignments (mean cosine similarity of 0.7078). A Kolmogorov-Smirnov test confirmed the statistical significance of the separation (KS-statistic = 0.9709, p-value = 8.2856e-05). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Questionable
- **Test:** Kolmogorov-Smirnov, Student's t-test; p ≈ 8.3e-05; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 243. (Surprise 0.284) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 160 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Quantify how the geometric predictability of split gaps decays over distance, informing dynamic thresholding for reconnection pipelines.
- **Conclusion:** The Pearson correlation coefficient between gap distance and tangent alignment score was -0.3489 (1-sided p-value = 1.671e-12), and the Spearman correlation magnitude aligned with this trend. Because the 1-sided test p-value is far below 0.05, the hypothesis is confirmed: geometric predictability of split gaps significantly decays over larger spatial distances. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman, Pearson; p ≈ 1.7e-12
- **Statistical issues:** none
- **Logic issues:** none

### 244. (Surprise 0.284) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 167 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Determine if fragment cable length acts as a reliable structural prior for split-correction confidence.
- **Conclusion:** The experiment successfully tested the hypothesis that fragment cable length acts as a reliable structural prior for split-correction confidence. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 245. (Surprise 0.284) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 170 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Test the hypothesis that fixing topological splits significantly increases the variance of component cable lengths, indicating a dynamic shift in graph state post-correction.
- **Conclusion:** The baseline (pre-correction) metrics across 10,172 components showed a mean cable length of 1849.31 µm and a variance of 27.86M µm². Following the corrections, the mean fragment length grew to 2087.37 µm, while the variance substantially increased to 84.61M µm². Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** test type not extractable from record; p = 0.05
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 246. (Surprise 0.284) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 174 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Evaluate the geometric signatures of recoverable splits to determine the trade-off between gap size and tangent alignment reliability.
- **Conclusion:** By analyzing 647 valid true split pairs (fragment endpoints mapping to the same ground-truth neuron), a significant negative correlation was found between the spatial gap distance and the orientation agreement (Pearson r = -0.4619, p-value = 1.64e-35). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** one-way ANOVA, Pearson; p ≈ 9.7e-32
- **Statistical issues:** none
- **Logic issues:** none

### 247. (Surprise 0.284) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 176 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Determine if weak local image evidence (thin neurites) is a primary geometric factor predisposing the automated U-Net segmentation to split errors.
- **Conclusion:** The results show that the mean radius of leaf nodes (representing fragment endpoints or split errors) is 1.5761 µm (variance = 0.2034), whereas the mean radius of continuous internal nodes is significantly larger at 1.9676 µm (variance = 0.0072) based on an equal sample size of 39,826 nodes. A Welch's t-test yielded a t-statistic of -170.2359 and a p-value approaching zero. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p ≈ 0
- **Statistical issues:** none
- **Logic issues:** none

### 248. (Surprise 0.284) Supported: evidence confirms that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 177 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Evaluate if image anisotropy introduces a directional bias in the occurrence of split errors.
- **Conclusion:** The terminal segments showed a mean Z-alignment (|cos(θ)|) of 0.4934 (variance: 0.0886), while the internal segments had a mean of 0.4760 (variance: 0.0845). Statistical testing confirmed this difference is highly significant, with Welch's t-test yielding a t-statistic of 8.3243 (p-value = 8.6160e-17) and a Kolmogorov-Smirnov test yielding a KS-statistic of 0.0531 (p-value = 3.3276e-49). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Kolmogorov-Smirnov, Student's t-test; p ≈ 3.3e-49
- **Statistical issues:** none
- **Logic issues:** none

### 249. (Surprise 0.284) Supported: evidence confirms that image-intensity features (signal across the gap) add discriminative power.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 178 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Measure the spatial variation of predicted neurite radius near fragment endpoints to understand the localized failure modes of the U-Net segmentation.
- **Conclusion:** The mean radius at the interior (>3 hops) is highest at 1.9916 µm, dropping to 1.9307 µm in the transition zone (2-3 hops), and further decreasing to 1.7928 µm at the extremities (0-1 hops). An ANOVA test confirmed that these differences are highly statistically significant (F-statistic = 79670.34, p-value = 0.0). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** one-way ANOVA; p ≈ 0
- **Statistical issues:** none
- **Logic issues:** none

### 250. (Surprise 0.284) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 194 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8906) · **Direction:** Positive
- **Tested:** Investigate the spatial distribution of fragmentation errors by correlating fragment length with distance to the soma.
- **Conclusion:** The statistical analysis yielded a Spearman Rank Correlation coefficient of -0.1379 with a highly significant p-value of 2.7850e-06. These results confirm the hypothesis: there is a statistically significant negative correlation between distance-to-soma and fragment length, meaning fragments situated physically closer to the neuron soma tend to be longer than those further away. Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.284; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman; p ≈ 2.8e-06
- **Statistical issues:** none
- **Logic issues:** none

### 251. (Surprise 0.264) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 17 · **Belief:** Leaning False -> Likely False (0.2500 -> 0.0577) · **Direction:** Negative
- **Tested:** Simulate the graph-topological impact of a 'small-fragment scavenging' post-processing phase on overall reconstruction ERL.
- **Conclusion:** The experiment successfully executed and calculated the impact of two different topological reconnection strategies on the Normalized Expected Run Length (ERL). Belief therefore dropped from Leaning False (0.25) to Likely False (0.06) (signed surprisal -0.264; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 252. (Surprise 0.264) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 24 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.9423) · **Direction:** Positive
- **Tested:** Test if the angular alignment of adjacent fragment endpoints can effectively distinguish between valid split reconnections and invalid false connections.
- **Conclusion:** The results strongly support the hypothesis: Valid Split endpoints are highly anti-parallel, with a mean cosine similarity of -0.6930 (± 0.4061), indicating they naturally point towards each other along the fragmented neurite. In contrast, the few Invalid Splits observed had a positive mean cosine similarity of 0.5833 (± 0.2358). Belief therefore rose from Leaning True (0.75) to Likely True (0.94) (signed surprisal +0.264; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 0.0005
- **Statistical issues:** none
- **Logic issues:** none

### 253. (Surprise 0.264) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 42 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.9423) · **Direction:** Positive
- **Tested:** Investigate whether localized topological complexity (branching) in the predicted graph is intrinsically coupled with merge errors.
- **Conclusion:** - **Statistical Comparison:** A Chi-square test of independence confirmed a highly significant difference in branching proportions (Chi-square statistic = 556.14, p-value = 5.8169e-123). Belief therefore rose from Leaning True (0.75) to Likely True (0.94) (signed surprisal +0.264; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** chi-square; p ≈ 5.8e-123
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 254. (Surprise 0.264) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 98 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.9423) · **Direction:** Positive
- **Tested:** Prove that correcting merges dynamically unlocks new valid split reconnections, supporting an iterative agentic proofreading loop over a single-pass pipeline.
- **Conclusion:** The experiment was successfully executed, fulfilling all the planned objectives. Belief therefore rose from Leaning True (0.75) to Likely True (0.94) (signed surprisal +0.264; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 255. (Surprise 0.260) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 21 · **Belief:** Leaning True -> Leaning True (0.7917 -> 0.6250) · **Direction:** Negative
- **Tested:** Quantify the failure rate of distance-only reconnection heuristics and evaluate the performance gain of a composite feature metric.
- **Conclusion:** The extremely small sample size (N=2) indicates that such highly confusable, dense neighborhoods within a 30 µm radius are exceedingly rare in this dataset. Belief therefore dropped from Leaning True (0.79) to Leaning True (0.62) (signed surprisal -0.260; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** test type not extractable from record; n up to 2
- **Statistical issues:** none
- **Logic issues:** none

### 256. (Surprise 0.255) Supported: evidence confirms that image-intensity features (signal across the gap) add discriminative power.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 44 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8942) · **Direction:** Positive
- **Tested:** Investigate whether the presence of a merge error locally destabilizes the U-Net segmentation, leading to a cluster of nearby, short-gap split errors (coupling split and merge correction).
- **Conclusion:** The subsequent Welch's t-test yielded a highly significant result (t-statistic = -19.0498, p-value = 2.7942e-19), confirming that Merge-Adjacent split gaps are significantly shorter (mean: 39.78 µm) than Isolated split gaps (mean: 139.30 µm). Belief therefore rose from Leaning True (0.71) to Likely True (0.89) (signed surprisal +0.255; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Welch's t-test, Student's t-test; p ≈ 2.8e-19
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 257. (Surprise 0.252) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 40 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.5469) · **Direction:** Negative
- **Tested:** Determine the efficacy of local tangent agreement as a mechanism for an agent to resolve mutually exclusive topological proposals at dense leaf clusters.
- **Conclusion:** The mean cosine similarity for Matches was 0.1868, whereas for Mismatches it was -0.7972. A Mann-Whitney U test yielded a p-value of 0.0708, which is slightly above the standard 0.05 significance threshold. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.55) (signed surprisal -0.252; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.0708; n up to 2
- **Statistical issues:** none
- **Logic issues:** none

### 258. (Surprise 0.243) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 9 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.9062) · **Direction:** Positive
- **Tested:** Quantify the structural trade-off of resolving merge errors by measuring the reduction in massive erroneous super-components versus the loss of intact cable length.
- **Conclusion:** The experiment was successfully executed, resolving previous module and file path errors. Belief therefore rose from Leaning True (0.75) to Likely True (0.91) (signed surprisal +0.243; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** test type not extractable from record
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 259. (Surprise 0.243) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 14 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.9062) · **Direction:** Positive
- **Tested:** Identify the topological locations where the U-Net model most frequently fuses distinct neurons to better target merge-detection algorithms.
- **Conclusion:** - **Statistical Significance:** The Chi-square test yielded a statistic of 31.2730 with a p-value of 2.2418e-08. **Conclusion:** The very low p-value indicates a highly statistically significant difference. Belief therefore rose from Leaning True (0.75) to Likely True (0.91) (signed surprisal +0.243; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p ≈ 2.2e-08
- **Statistical issues:** none
- **Logic issues:** none

### 260. (Surprise 0.243) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 26 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.9062) · **Direction:** Positive
- **Tested:** Determine if local neurite radius is a predictive factor for U-Net splitting errors by comparing the radii of leaf nodes (split sites) to internal nodes (contiguous regions), ensuring numerical stability during statistical testing.
- **Conclusion:** Descriptive statistics confirm the hypothesis: leaf nodes have a significantly smaller mean radius (1.5761 µm) and median radius (1.6729 µm) compared to contiguous internal nodes (mean: 1.9676 µm, median: 1.9766 µm). The Mann-Whitney U test yielded a valid statistic of 850,550,050.0 and a p-value of 0.0000 (p < 0.05), providing strong statistical significance. Belief therefore rose from Leaning True (0.75) to Likely True (0.91) (signed surprisal +0.243; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 261. (Surprise 0.243) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 47 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.9062) · **Direction:** Positive
- **Tested:** Assess whether global topological features of a fragment component can reliably flag the presence of internal merge errors.
- **Conclusion:** - **Average Node Degree:** - Merged: Mean = 1.9961, Median = 1.9970 - Clean: Mean = 1.9950, Median = 1.9951 - Statistical Test: Mann-Whitney U statistic = 24006.0, p-value = 0.00897. - **Total Leaf Counts:** - Merged: Mean = 26.63, Median = 5.0 - Clean: Mean = 7.06, Median = 4.0 - Statistical Test: Mann-Whitney U statistic = 25779.0, p-value = 0.00043. Belief therefore rose from Leaning True (0.75) to Likely True (0.91) (signed surprisal +0.243; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.05; n up to 26
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 262. (Surprise 0.243) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 147 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.9062) · **Direction:** Positive
- **Tested:** Determine if fragment cable length is positively associated with the likelihood of encompassing a merge error.
- **Conclusion:** - **Fragments WITH Merge Errors (n=9):** - Mean Cable Length: 61,497.45 µm - Median Cable Length: 20,660.61 µm - **Fragments WITHOUT Merge Errors (n=10,163):** - Mean Cable Length: 1,796.48 µm - Median Cable Length: 1,175.02 µm - **Statistical Testing:** The Mann-Whitney U Test yielded a highly significant p-value of 1.3198e-04. Belief therefore rose from Leaning True (0.75) to Likely True (0.91) (signed surprisal +0.243; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p ≈ 0.00013; n up to 10163
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 263. (Surprise 0.243) Refuted: data contradict the claim that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 164 · **Belief:** Leaning True -> Uncertain (0.7500 -> 0.5938) · **Direction:** Negative
- **Tested:** Verify that dense neuropil regions require stricter distance/geometry thresholds because simple proximity becomes an unreliable heuristic.
- **Conclusion:** The resulting density coefficient was 0.3027 (Odds Ratio: 1.353) with a p-value of 0.455, indicating no statistically significant relationship. Belief therefore dropped from Leaning True (0.75) to Uncertain (0.59) (signed surprisal -0.243; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** logistic regression, Student's t-test; p = 0.455
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 264. (Surprise 0.243) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 183 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.9062) · **Direction:** Positive
- **Tested:** Determine if the U-Net is systematically more prone to splitting neurons at branching junctions compared to simple linear segments.
- **Conclusion:** A Chi-squared test was performed on these proportions, yielding a statistic of 28.6547 and a highly significant p-value of 8.65e-08. Belief therefore rose from Leaning True (0.75) to Likely True (0.91) (signed surprisal +0.243; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** chi-square; p ≈ 8.7e-08
- **Statistical issues:** none
- **Logic issues:** none

### 265. (Surprise 0.235) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 179 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8594) · **Direction:** Positive
- **Tested:** Determine if fragment length (a proxy for segmentation confidence) predicts the severity (gap size) of split errors.
- **Conclusion:** The mean gap distance for these split errors was 11.42 µm (std = 8.90), and the mean combined fragment length was 20,044.59 µm (std = 57,741.76). A Spearman rank correlation analysis between gap distance and combined fragment length yielded a correlation coefficient of -0.1139 with a highly significant p-value of 1.2692e-06. Belief therefore rose from Leaning True (0.71) to Likely True (0.86) (signed surprisal +0.235; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman; p ≈ 1.3e-06
- **Statistical issues:** none
- **Logic issues:** none

### 266. (Surprise 0.235) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 199 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8594) · **Direction:** Positive
- **Tested:** Verify that geometric collinearity of endpoints is a reliable feature for predicting mutual exclusivity and valid reconnections at crowded fragment endpoints.
- **Conclusion:** A Mann-Whitney U test compared the two distributions, yielding a statistically significant p-value of ~0.0047 (U=1192.0). Belief therefore rose from Leaning True (0.71) to Likely True (0.86) (signed surprisal +0.235; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.0047
- **Statistical issues:** none
- **Logic issues:** none

### 267. (Surprise 0.220) Supported: evidence confirms that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 4 · **Belief:** Uncertain -> Leaning True (0.5417 -> 0.7019) · **Direction:** Positive
- **Tested:** Evaluate if tangent angular agreement can independently resolve competing, mutually exclusive split-correction proposals without relying on deep learning image features.
- **Conclusion:** The analysis found that True connections had a mean angular agreement of 0.1816, whereas False connections had a mean of -0.7437. Although the mean difference was substantial and the ROC curve showed an impressive AUC of 0.868, the Welch's t-test yielded a p-value of 0.1120 due to the extremely small sample size of False connections (n=2). Belief therefore rose from Uncertain (0.54) to Leaning True (0.70) (signed surprisal +0.220; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Welch's t-test, ROC-AUC, Student's t-test; p = 0.112; n up to 2
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 268. (Surprise 0.220) Supported: evidence confirms that image-intensity features (signal across the gap) add discriminative power.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 18 · **Belief:** Leaning True -> Likely True (0.7917 -> 0.9519) · **Direction:** Positive
- **Tested:** Determine if split errors occur significantly closer to branch points than expected by chance.
- **Conclusion:** The results show a highly significant difference (KS Statistic: 0.3430, p-value: 5.3014e-194), with the median distance to a branch point being 53.96 µm for split errors compared to 239.68 µm for control nodes. Belief therefore rose from Leaning True (0.79) to Likely True (0.95) (signed surprisal +0.220; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov; p ≈ 5.3e-194
- **Statistical issues:** none
- **Logic issues:** none

### 269. (Surprise 0.220) Supported: evidence confirms that geometric gap features identify true split reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 67 · **Belief:** Leaning True -> Likely True (0.7917 -> 0.9519) · **Direction:** Positive
- **Tested:** Assess the structural nature of split errors by calculating the ratio of leaf-to-leaf versus leaf-to-branch/trunk valid reconnections.
- **Conclusion:** - **Statistical Test:** A binomial test against the null hypothesis that valid targets are predominantly leaf nodes (≥ 90%) yielded a p-value of 0.0, leading to a strong rejection of the null hypothesis. Belief therefore rose from Leaning True (0.79) to Likely True (0.95) (signed surprisal +0.220; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** binomial test; p ≈ 0
- **Statistical issues:** none
- **Logic issues:** none

### 270. (Surprise 0.211) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 33 · **Belief:** Leaning True -> Uncertain (0.7500 -> 0.5962) · **Direction:** Negative
- **Tested:** Determine if merge errors are spatially clustered around U-Net generated branching nodes.
- **Conclusion:** Statistical analysis using the Kolmogorov-Smirnov test confirmed a highly significant difference between the two spatial distributions (Statistic: 0.2123, p-value: 3.5487e-06). The median distance for merge edges (175.59 µm) was actually higher than for clean edges (126.05 µm), meaning more than half of the clean edges are closer to a branch than the median merge edge. Belief therefore dropped from Leaning True (0.75) to Uncertain (0.60) (signed surprisal -0.211; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov; p ≈ 3.5e-06
- **Statistical issues:** none
- **Logic issues:** none

### 271. (Surprise 0.203) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 1 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.5781) · **Direction:** Negative
- **Tested:** To determine if U-Net predicted neurite radius is conserved across split gaps and can serve as a reliable feature for disambiguating connection proposals.
- **Conclusion:** The mean absolute radius difference for True Positive pairs was 0.4474 µm, and for False Positive pairs, it was 0.5016 µm. A Mann-Whitney U test yielded a p-value of 0.6464, indicating no statistically significant difference in radius continuity between valid and invalid connections within this proximity. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.58) (signed surprisal -0.203; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.6464
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 272. (Surprise 0.203) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 8 · **Belief:** Leaning True -> Likely True (0.7917 -> 0.9219) · **Direction:** Positive
- **Tested:** Evaluate the geometric orientation agreement (tangent collinearity) as a distinguishing signature of recoverable splits.
- **Conclusion:** The analysis found 395 true split pairs with a mean collinearity of -0.7745, confirming strong orientation agreement. In contrast, the 3 false proposal pairs exhibited a mean collinearity of 0.7179. Belief therefore rose from Leaning True (0.79) to Likely True (0.92) (signed surprisal +0.203; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Questionable
- **Test:** Welch's t-test, two-sample t-test, Student's t-test; p = 0.0078; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 273. (Surprise 0.203) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 47 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.5781) · **Direction:** Negative
- **Tested:** To verify if the directional continuity of neurites (tangent alignment) is a robust and discriminative geometric feature for resolving split errors, especially when spatial proximity alone is insufficient.
- **Conclusion:** The mean absolute cosine similarity (directional agreement of terminal tangent vectors) was 0.6842 for TP pairs and 0.6368 for FP pairs. A Mann-Whitney U test (evaluating if TP > FP) returned a test statistic of 3589.0 and a p-value of 0.2219. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.58) (signed surprisal -0.203; negative shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.2219
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 274. (Surprise 0.203) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 13 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.5781) · **Direction:** Negative
- **Tested:** Determine if extremely short components act as 'noise bridges' that create false merges, testing if fragment length is a useful proxy for proposal confidence.
- **Conclusion:** Statistical analysis using the Mann-Whitney U test found no statistically significant difference in the minimum component cable lengths between true positive and false positive proposals (p-value ≈ 0.292). Belief therefore dropped from Leaning True (0.71) to Uncertain (0.58) (signed surprisal -0.203; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.292; n up to 3633
- **Statistical issues:** none
- **Logic issues:** none

### 275. (Surprise 0.203) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 22 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.5781) · **Direction:** Negative
- **Tested:** Evaluate the benefit of joint geometric reasoning over independent thresholding in resolving competing split proposals at dense junctions.
- **Conclusion:** The experiment was successfully executed. The dataset was properly loaded using a custom unpickler to bypass missing dependencies. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.58) (signed surprisal -0.203; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 276. (Surprise 0.203) Refuted: data contradict the claim that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 39 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.5781) · **Direction:** Negative
- **Tested:** Determine if morphological tortuosity can serve as an intrinsic feature to detect and localize merge errors by comparing the tortuosity of merged bridge segments to a pre-computed pool of length-matched pure segments.
- **Conclusion:** While the descriptive statistics show that merged bridges have a higher mean (1.4965 vs. 1.2643) and median (1.2718 vs. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.58) (signed surprisal -0.203; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; n up to 7
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 277. (Surprise 0.203) Supported: evidence confirms that morphological features mark U-Net merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 43 · **Belief:** Leaning True -> Likely True (0.7917 -> 0.9219) · **Direction:** Positive
- **Tested:** Verify if the overall physical size of a U-Net fragment is a risk factor for topological merge errors.
- **Conclusion:** The 'Merged' fragments exhibited substantially larger physical sizes, with a mean cable length of 51,339.69 µm and a median of 10,323.09 µm. In contrast, the 'Clean' fragments had a mean cable length of 4,268.07 µm and a median of 1,884.35 µm. Belief therefore rose from Leaning True (0.79) to Likely True (0.92) (signed surprisal +0.203; positive shift).
- **Caveats:** none noted
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.0051
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 278. (Surprise 0.203) Refuted: data contradict the claim that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 44 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.5781) · **Direction:** Negative
- **Tested:** Test if radius variance along a skeletal path can serve as a robust intra-fragment feature for detecting merge errors.
- **Conclusion:** However, due to the extremely limited sample size (n=1) — likely resulting from missing or invalid radius estimates along the computed shortest paths in the merged graphs — the Mann-Whitney U test predictably yielded a non-significant result (p-value = 0.5000). Belief therefore dropped from Leaning True (0.71) to Uncertain (0.58) (signed surprisal -0.203; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.5
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 279. (Surprise 0.203) Refuted: data contradict the claim that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 52 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.5781) · **Direction:** Negative
- **Tested:** Test whether joint, mutually exclusive reasoning over endpoint reconnections outperforms independent classification in dense regions.
- **Conclusion:** The experiment successfully executed, resolving dependencies and loading the dataset. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.58) (signed surprisal -0.203; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** test type not extractable from record
- **Statistical issues:** none
- **Logic issues:** none

### 280. (Surprise 0.203) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 121 · **Belief:** Leaning True -> Likely True (0.7917 -> 0.9219) · **Direction:** Positive
- **Tested:** Quantify the predictive power of orientation agreement for identifying valid split reconnections.
- **Conclusion:** The results strongly supported the hypothesis: True reconnections exhibited a mean cosine similarity of -0.7896, heavily skewed towards -1.0 (indicating head-on continuous fibers), while False reconnections averaged 0.2296 and showed a more dispersed distribution. The statistical test confirmed a highly significant difference (p-value = 1.2158e-05). Belief therefore rose from Leaning True (0.79) to Likely True (0.92) (signed surprisal +0.203; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Welch's t-test, Student's t-test; p ≈ 1.2e-05
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 281. (Surprise 0.203) Refuted: data contradict the claim that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 141 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.5781) · **Direction:** Negative
- **Tested:** Determine the relationship between local fragment crowding and the severity of neuron fragmentation.
- **Conclusion:** Consequently, the statistical tests showed no meaningful relationship: the Mann-Whitney U test yielded a p-value of 0.995, and the Pearson correlation between local density and fragment length was negligible (r = 0.0018, p = 0.8557). Belief therefore dropped from Leaning True (0.71) to Uncertain (0.58) (signed surprisal -0.203; negative shift).
- **Caveats:** effect size is negligible despite small p-value (large-N artifact)
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Pearson; p = 0.8557
- **Statistical issues:** none
- **Logic issues:** none

### 282. (Surprise 0.203) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 148 · **Belief:** Leaning True -> Likely True (0.7917 -> 0.9219) · **Direction:** Positive
- **Tested:** Assess if segmentation performance degrades systematically as a function of distance from the soma.
- **Conclusion:** Proximal fragments (N=39) demonstrated a substantially longer average cable length of 4040.78 µm compared to distal fragments (N=9985) which had an average cable length of 1821.93 µm. An independent samples t-test on the log-transformed lengths yielded a t-statistic of 4.7451 and a p-value of ~2.92e-05. Belief therefore rose from Leaning True (0.79) to Likely True (0.92) (signed surprisal +0.203; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Student's t-test; p ≈ 2.9e-05; n up to 9985
- **Statistical issues:** none
- **Logic issues:** none

### 283. (Surprise 0.203) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 188 · **Belief:** Leaning True -> Uncertain (0.7083 -> 0.5781) · **Direction:** Negative
- **Tested:** Validate fragment component size as a strong confidence modulator for split-correction reliability.
- **Conclusion:** As a result, the Chi-Square test of independence could not be performed due to insufficient data categories. Belief therefore dropped from Leaning True (0.71) to Uncertain (0.58) (signed surprisal -0.203; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square, Student's t-test; n up to 99
- **Statistical issues:** none
- **Logic issues:** none

### 284. (Surprise 0.203) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 193 · **Belief:** Leaning True -> Likely True (0.7917 -> 0.9219) · **Direction:** Positive
- **Tested:** Verify if merge errors systematically manifest as false branches in the U-Net reconstruction, justifying algorithms that specifically target branch points for merge-detection and topology rollback.
- **Conclusion:** The Chi-Square test of independence yielded a test statistic of 462.9983 and a p-value of 1.07e-102. Because the p-value is practically zero and well below the 0.05 significance threshold, we confidently reject the null hypothesis. Belief therefore rose from Leaning True (0.79) to Likely True (0.92) (signed surprisal +0.203; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** chi-square, Student's t-test; p ≈ 1.1e-102
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 285. (Surprise 0.203) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 196 · **Belief:** Leaning True -> Likely True (0.7917 -> 0.9219) · **Direction:** Positive
- **Tested:** Determine if tangent orientation agreement between endpoints is a strong geometric feature for classifying valid vs. invalid split reconnections.
- **Conclusion:** The results strongly support the hypothesis: the median angular difference for valid reconnections was significantly smaller (20.45 degrees) compared to invalid reconnections (128.06 degrees). Belief therefore rose from Leaning True (0.79) to Likely True (0.92) (signed surprisal +0.203; positive shift).
- **Caveats:** none noted
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p ≈ 0.00081
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 286. (Surprise 0.202) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 11 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8558) · **Direction:** Positive
- **Tested:** Test if merge errors disproportionately afflict large, well-established U-Net fragment components.
- **Conclusion:** The statistical analysis reveals that merged components have a significantly higher log10 cable length (mean = 3.7874 ± 0.7420, roughly ~6,100 µm) compared to clean components (mean = 3.1464 ± 0.2294, roughly ~1,400 µm). Welch's t-test confirmed that this difference is highly significant (t-statistic = 3.7649, p-value = 0.0013), strongly supporting the hypothesis that merge errors disproportionately afflict larger, more established U-Net fragments. Belief therefore rose from Leaning True (0.71) to Likely True (0.86) (signed surprisal +0.202; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Questionable
- **Test:** Welch's t-test, Student's t-test; p = 0.0013; n up to 10152
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 287. (Surprise 0.195) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 27 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.8750) · **Direction:** Positive
- **Tested:** To test if tangent vector agreement is a discriminative geometric signature for identifying recoverable splits.
- **Conclusion:** The results strongly support the hypothesis: true reconnections exhibit a significantly higher cosine similarity (mean = 0.7103) compared to false reconnections (mean = -0.4796). A two-sample t-test confirmed the statistical significance of this difference (p-value = ~0.013). Belief therefore rose from Leaning True (0.75) to Likely True (0.88) (signed surprisal +0.195; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Questionable
- **Test:** two-sample t-test, Student's t-test; p = 0.013; n up to 5
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 288. (Surprise 0.195) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 155 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.8750) · **Direction:** Positive
- **Tested:** Verify if total component cable length can serve as a simple, effective heuristic prior for flagging likely merge errors.
- **Conclusion:** Pure fragments had a mean length of ~4,077 µm (median ~1,830 µm), whereas Merged fragments exhibited a substantially larger mean length of ~28,963 µm (median ~3,006 µm). A one-sided Mann-Whitney U test yielded a p-value of 0.0264 (p < 0.05), indicating that the total cable length of merged components is statistically significantly larger than that of pure components. Belief therefore rose from Leaning True (0.75) to Likely True (0.88) (signed surprisal +0.195; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 289. (Surprise 0.195) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 168 · **Belief:** Leaning True -> Likely True (0.7500 -> 0.8750) · **Direction:** Positive
- **Tested:** Validate that tangent orientation is a statistically strong feature for distinguishing true reconnections from false merges at U-Net fragment boundaries.
- **Conclusion:** Despite the small number of false pairs, the statistical separation was significant (t=10.77, p=0.007). True pairs had a mean cosine similarity of 0.7524 (highly aligned), while the 3 false pairs had a mean of -0.7433 (opposing/misaligned). Belief therefore rose from Leaning True (0.75) to Likely True (0.88) (signed surprisal +0.195; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** two-sample t-test, ROC-AUC, Student's t-test; p = 0.007; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 290. (Surprise 0.187) Supported: evidence confirms that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 2 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8281) · **Direction:** Positive
- **Tested:** Test if tortuosity can be used as an intrinsic skeleton feature to identify false merges where neurites cross but do not truly connect.
- **Conclusion:** Statistical analysis using the Mann-Whitney U test revealed that the mean tortuosity for Merge Errors (1.0940) is significantly higher than that for True Branches (1.0765), with a p-value of 2.1030e-04 (p < 0.05). While the median tortuosity is visibly higher for Merge Errors (~1.105) compared to True Branches (~1.060), there is substantial overlap in the distributions (IQR and whiskers). Belief therefore rose from Leaning True (0.71) to Likely True (0.83) (signed surprisal +0.187; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 291. (Surprise 0.187) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 28 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8281) · **Direction:** Positive
- **Tested:** Determine if local neurite radius continuity can serve as a geometric signature for detecting merge errors without relying on raw image data.
- **Conclusion:** The statistical analysis supports the hypothesis: branch points associated with Merge Errors exhibited significantly higher local radius variance (Mean Std: 0.1421 vs 0.1095, p=0.0128) and Coefficient of Variation (Mean CV: 0.0747 vs 0.0562, p=0.0152) compared to True Branches. Belief therefore rose from Leaning True (0.71) to Likely True (0.83) (signed surprisal +0.187; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Student's t-test; p = 0.0152
- **Statistical issues:** none
- **Logic issues:** none

### 292. (Surprise 0.187) Refuted: data contradict the claim that geometric gap features identify true split reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 163 · **Belief:** Uncertain -> Uncertain (0.5417 -> 0.4219) · **Direction:** Negative
- **Tested:** Evaluate the viability of using competitive distance-based reasoning to resolve mutually exclusive reconnection proposals at a single leaf node.
- **Conclusion:** For these two cases, the mean distance to the nearest valid candidate was 15.81 µm, while the mean distance to the nearest invalid candidate was 9.98 µm. The paired t-test yielded a p-value of ~0.345, which is not statistically significant due to the extremely small sample size (N=2). Belief therefore dropped from Uncertain (0.54) to Uncertain (0.42) (signed surprisal -0.187; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** paired t-test, Student's t-test; p = 0.345; n up to 2
- **Statistical issues:** per-group sample size very small (e.g. ~2 observations per annotator across 18-19 GT neurons) — group-comparison test is underpowered
- **Logic issues:** non-significant p with very small per-group N interpreted as evidence of no effect — the test had little power to detect one

### 293. (Surprise 0.162) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 31 · **Belief:** Likely True -> Likely True (0.8333 -> 0.9375) · **Direction:** Positive
- **Tested:** Determine the relationship between the physical size of a split gap and the alignment of the structural tangents at the endpoints.
- **Conclusion:** A highly significant negative correlation was found (Spearman: -0.4668, p < 1e-80). Binned statistics reveal that short-range gaps (0-5 µm) exhibit near-perfect alignment (mean agreement = ~0.95), whereas long-range gaps (> 100 µm) degrade to an agreement score of ~0.46, which is equivalent to random chance. Belief therefore rose from Likely True (0.83) to Likely True (0.94) (signed surprisal +0.162; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman, Pearson; p ≈ 1e-80
- **Statistical issues:** none
- **Logic issues:** none

### 294. (Surprise 0.162) Supported: evidence confirms that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 175 · **Belief:** Likely True -> Likely True (0.8333 -> 0.9375) · **Direction:** Positive
- **Tested:** Quantify the failure rate of simple nearest-neighbor reconnection heuristics in dense proposal regions (mutual exclusivity).
- **Conclusion:** - **Statistical Significance:** A Chi-squared test comparing the correct and incorrect match proportions between the two regions yielded a statistic of 10.4526 and a p-value of 1.2248e-03. **Conclusion:** The p-value (p < 0.01) confirms that there is a statistically significant degradation in the accuracy of greedy nearest-neighbor distance matching as the density of candidate endpoints increases. Belief therefore rose from Likely True (0.83) to Likely True (0.94) (signed surprisal +0.162; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** chi-square; p = 0.01
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 295. (Surprise 0.162) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 180 · **Belief:** Likely True -> Likely True (0.8333 -> 0.9375) · **Direction:** Positive
- **Tested:** Analyze the spatial conditions under which cycle-prevention logic fails or becomes restrictive by correlating cyclic proposals with local fragment density.
- **Conclusion:** The experiment was successfully executed and fulfilled the objective of analyzing the spatial conditions that lead to cyclic proposals in a naive distance-based reconnection graph. Belief therefore rose from Likely True (0.83) to Likely True (0.94) (signed surprisal +0.162; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Welch's t-test, Student's t-test; p ≈ 2e-25
- **Statistical issues:** none
- **Logic issues:** none

### 296. (Surprise 0.154) Refuted: data contradict the claim that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 128 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.6094) · **Direction:** Negative
- **Tested:** Determine if the imaging anisotropy causes a directional bias in split errors, which could inform anisotropic search spaces for reconnection proposals.
- **Conclusion:** The results yielded a mean Z-alignment (abs cos) of 0.4934 for terminal edges versus 0.4738 for internal edges. The K-S test statistic was 0.0541 with a p-value of 3.6617e-51, indicating a statistically significant difference in the distributions. Belief therefore dropped from Leaning True (0.71) to Leaning True (0.61) (signed surprisal -0.154; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov, Student's t-test; p ≈ 3.7e-51
- **Statistical issues:** none
- **Logic issues:** none

### 297. (Surprise 0.149) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 22 · **Belief:** Leaning True -> Likely True (0.7083 -> 0.8173) · **Direction:** Positive
- **Tested:** To determine if the geometric gap size between valid split endpoints is positively correlated with their physical distance from the nearest soma, informing dynamic, spatially-aware search radius policies.
- **Conclusion:** A Pearson correlation analysis between the physical gap size and the distance to the nearest soma yielded a significant negative correlation (r = -0.3355, p-value = 1.4371e-16). Belief therefore rose from Leaning True (0.71) to Likely True (0.82) (signed surprisal +0.149; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Pearson; p ≈ 1.4e-16
- **Statistical issues:** none
- **Logic issues:** none

### 298. (Surprise 0.138) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 58 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7969) · **Direction:** Positive
- **Tested:** Investigate if local radius discontinuity can serve as an unsupervised feature to detect post-hoc merge errors along a skeleton.
- **Conclusion:** **Findings:** - The mean radius delta for Pure Edges was 0.009084 µm. - The mean radius delta for Merge Edges was 0.056798 µm (over 6 times larger). Belief therefore rose from Leaning True (0.71) to Leaning True (0.80) (signed surprisal +0.138; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 299. (Surprise 0.138) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 134 · **Belief:** Leaning True -> Leaning True (0.7917 -> 0.7031) · **Direction:** Negative
- **Tested:** Evaluate the predictive power of 3D tangent alignment for filtering valid versus invalid split reconnection proposals.
- **Conclusion:** The mean cosine similarity (tangent alignment) for valid pairs was 0.8352, compared to 0.6928 for invalid pairs, suggesting that valid continuations tend to be better aligned. However, due to the extremely small sample size of invalid pairs (n=3), the Mann-Whitney U test yielded a non-significant p-value of 0.2392. Belief therefore dropped from Leaning True (0.79) to Leaning True (0.70) (signed surprisal -0.138; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.2392; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 300. (Surprise 0.138) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 152 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7969) · **Direction:** Positive
- **Tested:** Determine if local branching density can act as an active environmental flag to proactively trigger merge-detection agents.
- **Conclusion:** A one-sided Mann-Whitney U test comparing the two distributions yielded a statistic of 14,776.50 and a p-value of 0.027 (p < 0.05). Belief therefore rose from Leaning True (0.71) to Leaning True (0.80) (signed surprisal +0.138; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 301. (Surprise 0.138) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 173 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7969) · **Direction:** Positive
- **Tested:** Correlate automated fragment continuity (cable length) with physical neurite thickness.
- **Conclusion:** The results support the hypothesis: there is a statistically significant positive correlation between fragment cable length and average node radius (Spearman correlation = 0.2522, p-value = 2.5362e-147). Belief therefore rose from Leaning True (0.71) to Leaning True (0.80) (signed surprisal +0.138; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman; p ≈ 2.5e-147
- **Statistical issues:** none
- **Logic issues:** none

### 302. (Surprise 0.132) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 45 · **Belief:** Likely True -> Likely True (0.8750 -> 0.9712) · **Direction:** Positive
- **Tested:** Determine if local skeleton topology (specifically branching degree) is predictive of underlying merge errors in the U-Net fragments.
- **Conclusion:** A Chi-square test of independence yielded a statistic of 22.7651 and a highly significant p-value of 1.8306e-06, confirming that merge errors are disproportionately associated with structural branching in the U-Net fragments. Belief therefore rose from Likely True (0.88) to Likely True (0.97) (signed surprisal +0.132; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** chi-square; p ≈ 1.8e-06
- **Statistical issues:** none
- **Logic issues:** none

### 303. (Surprise 0.114) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 73 · **Belief:** Leaning True -> Leaning True (0.7917 -> 0.7188) · **Direction:** Negative
- **Tested:** Test if local component density is a strong predictive feature for U-Net merge errors (false fusions), which could guide a targeted merge-rollback policy.
- **Conclusion:** Because there were no dense branches, the Chi-squared test could not be performed. Belief therefore dropped from Leaning True (0.79) to Leaning True (0.72) (signed surprisal -0.114; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square
- **Statistical issues:** none
- **Logic issues:** none

### 304. (Surprise 0.089) Supported: evidence confirms that image-intensity features (signal across the gap) add discriminative power.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 29 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7656) · **Direction:** Positive
- **Tested:** To validate if missing image evidence correlates directly with the severity (distance) of split errors in the automated reconstruction, using a statistically sufficient random sample to accommodate remote data retrieval limits.
- **Conclusion:** The statistical analysis yielded a Spearman rank correlation coefficient of -0.3501 with a highly significant p-value of 3.7429e-07. This moderate, statistically significant negative correlation indicates that as the physical distance of a split error gap increases, the localized fluorescence intensity at the gap's midpoint tends to decrease. Belief therefore rose from Leaning True (0.71) to Leaning True (0.77) (signed surprisal +0.089; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman; p ≈ 3.7e-07
- **Statistical issues:** none
- **Logic issues:** none

### 305. (Surprise 0.089) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 125 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7656) · **Direction:** Positive
- **Tested:** Investigate if internal branching complexity is a structural indicator of underlying merge errors in the U-Net reconstruction.
- **Conclusion:** A Mann-Whitney U test yielded a p-value of ~0.0426. Since the p-value is less than the typical 0.05 threshold, the difference in branching density is statistically significant. Belief therefore rose from Leaning True (0.71) to Leaning True (0.77) (signed surprisal +0.089; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.0426
- **Statistical issues:** none
- **Logic issues:** none

### 306. (Surprise 0.088) Refuted: data contradict the claim that morphological features mark U-Net merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 83 · **Belief:** Likely True -> Leaning True (0.8333 -> 0.7692) · **Direction:** Negative
- **Tested:** Evaluate if graph-theoretic centrality metrics can identify merge errors without relying on ground-truth geometry.
- **Conclusion:** - **Centrality (Mean):** Merge edges demonstrated a mean betweenness centrality of 0.3087, while non-merge edges had a lower mean centrality of 0.2704. - **Statistical Comparison:** A Kolmogorov-Smirnov test confirmed that the distributions are significantly different (statistic = 0.1488, p-value = 1.3121e-04). Belief therefore dropped from Likely True (0.83) to Leaning True (0.77) (signed surprisal -0.088; negative shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Kolmogorov-Smirnov; p ≈ 0.00013
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 307. (Surprise 0.073) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 11 · **Belief:** Likely True -> Likely True (0.8750 -> 0.9219) · **Direction:** Positive
- **Tested:** To determine if local branch orientation agreement is a robust, purely geometric feature for filtering candidate reconnections and preventing false merges.
- **Conclusion:** The mean cosine similarity for true splits was -0.7054 (indicating they generally point towards each other), compared to 0.1944 for false splits. A Mann-Whitney U test yielded a p-value of 0.01458, which is statistically significant (p < 0.05). Belief therefore rose from Likely True (0.88) to Likely True (0.92) (signed surprisal +0.073; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 308. (Surprise 0.057) Refuted: data contradict the claim that the proposed structural heuristic holds for U-Net proofreading.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 2 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.6719) · **Direction:** Negative
- **Tested:** To test if mutual exclusivity (competitor density) can be used as a structural feature, where valid connections have fewer local alternatives than invalid ones.
- **Conclusion:** While the directional difference aligns with the hypothesis (TP burden < FP burden), the p-value of 0.4218 implies that the difference is not statistically significant. This lack of significance is likely driven by the extremely small sample size of false positive edges evaluated (n=5) under these specific distance thresholds. Belief therefore dropped from Leaning True (0.71) to Leaning True (0.67) (signed surprisal -0.057; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Unsound
- **Test:** Student's t-test; p = 0.4218; n up to 5
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 309. (Surprise 0.044) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 6 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7404) · **Direction:** Positive
- **Tested:** Test if proximity to somata significantly increases the likelihood of a merge error.
- **Conclusion:** Comparing the distances from these merge sites to the nearest soma against a background sample of 100,000 non-merge fragment nodes yielded the following results: - **Merge Sites:** Mean distance to soma = 7,610.05 µm (std: 3,582.41 µm) - **Background Nodes:** Mean distance to soma = 8,012.08 µm (std: 3,916.22 µm) - **Kolmogorov-Smirnov Test:** statistic = 0.0532, p-value = 3.9262e-11 The highly significant p-value (< 0.05) supports the hypothesis that the spatial distribution. Belief therefore rose from Leaning True (0.71) to Leaning True (0.74) (signed surprisal +0.044; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Kolmogorov-Smirnov; p ≈ 3.9e-11
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 310. (Surprise 0.044) Supported: evidence confirms that image-intensity features (signal across the gap) add discriminative power.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 12 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7404) · **Direction:** Positive
- **Tested:** Assess if raw image evidence within the gap region can distinguish true fragmented paths from falsely proposed cross-neuron connections.
- **Conclusion:** **Hypothesis:** The unannotated spatial gap between valid split reconnections contains residual, sub-threshold fluorescence signal, resulting in a significantly higher mean image intensity than the gap between invalid split pairs. - **Mean Intensities:** Valid gaps demonstrated a mean intensity of 112.67 (std: 147.85), whereas invalid gaps had a notably lower mean intensity of 36.17 (std: 25.75). Belief therefore rose from Leaning True (0.71) to Leaning True (0.74) (signed surprisal +0.044; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Questionable
- **Test:** Welch's t-test, Student's t-test; p = 0.0234; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 311. (Surprise 0.044) Supported: evidence confirms that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 20 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7404) · **Direction:** Positive
- **Tested:** Correlate ground-truth neurite tortuosity with the propensity for the segmentation model to produce split errors.
- **Conclusion:** Statistical analysis using the Mann-Whitney U test yielded a significant p-value of ~0.0041, confirming that there is a statistically significant difference between the two distributions. The mean local curvature adjacent to valid split gaps (0.3976 ± 0.2600 rad) is slightly higher than the background ground-truth curvature (0.3637 ± 0.2259 rad). Belief therefore rose from Leaning True (0.71) to Leaning True (0.74) (signed surprisal +0.044; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.0041
- **Statistical issues:** none
- **Logic issues:** none

### 312. (Surprise 0.044) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 32 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7404) · **Direction:** Positive
- **Tested:** Quantify whether U-Net false merges can be identified by their local branching angles compared to true valid branches.
- **Conclusion:** The statistical analysis (Mann-Whitney U Test: p-value = 0.0235) confirms a statistically significant difference in the maximum branching angles between the two classes. Specifically, True Branches exhibited a higher mean maximum angle (137.42°) compared to Merge Branches (126.24°). Belief therefore rose from Leaning True (0.71) to Leaning True (0.74) (signed surprisal +0.044; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.0235
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 313. (Surprise 0.044) Supported: evidence confirms that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 36 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7404) · **Direction:** Positive
- **Tested:** Investigate if anisotropic imaging resolution introduces a directional bias in split errors, which could inform anisotropic search bounds for reconnection proposals.
- **Conclusion:** Statistical analysis using the Kolmogorov-Smirnov test returned a test statistic of 0.0641 and a p-value of ~3.08e-05, confirming a statistically significant difference between the orientation of valid split gaps and the natural orientation of background GT neurites. The mean angle relative to the Z-axis was 60.01° for split gaps and 61.43° for GT edges. Belief therefore rose from Leaning True (0.71) to Leaning True (0.74) (signed surprisal +0.044; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Kolmogorov-Smirnov, Student's t-test; p ≈ 3.1e-05
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 314. (Surprise 0.044) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 56 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7404) · **Direction:** Positive
- **Tested:** Test if topological errors co-occur spatially, specifically measuring the local concentration of fragment endpoints near U-Net merge errors.
- **Conclusion:** - **Local Leaf Density (Mean ± Std):** The number of leaf nodes within a 20 µm radius was calculated for both groups. Merge sites had a mean of 0.10 (± 0.31) nearby leaf nodes, while the random sample of 100,000 background nodes had a lower mean of 0.06 (± 0.25). Belief therefore rose from Leaning True (0.71) to Leaning True (0.74) (signed surprisal +0.044; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p ≈ 3e-06
- **Statistical issues:** none
- **Logic issues:** none

### 315. (Surprise 0.044) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 74 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7404) · **Direction:** Positive
- **Tested:** Determine if the spatial proximity to the soma correlates with an increased density of competing split-correction proposals, necessitating spatially adaptive confidence thresholds.
- **Conclusion:** The results revealed a Spearman rank correlation of -0.0721 with a highly significant p-value of 4.65e-47. This negative correlation supports the hypothesis, confirming that reconnection ambiguity is statistically higher in closer proximity to the soma. Belief therefore rose from Leaning True (0.71) to Leaning True (0.74) (signed surprisal +0.044; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Spearman; p ≈ 4.6e-47
- **Statistical issues:** statistical significance driven by very large N; effect size negligible (|r|<0.1) — significance is a large-sample artifact rather than a scientifically meaningful effect
- **Logic issues:** none

### 316. (Surprise 0.044) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 96 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7404) · **Direction:** Positive
- **Tested:** Analyze the local branching geometry to determine if merge errors exhibit unnatural branching angles compared to valid bifurcations.
- **Conclusion:** Statistical analysis using the Kolmogorov-Smirnov test yielded a test statistic of 0.2382 and a p-value of 0.0286, indicating that the two distributions are drawn from different underlying populations. The mean maximum angle for Merge Branches (126.24°) was more acute than that of True Branches (137.42°). Belief therefore rose from Leaning True (0.71) to Leaning True (0.74) (signed surprisal +0.044; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov, Student's t-test; p = 0.0286; n up to 36
- **Statistical issues:** none
- **Logic issues:** none

### 317. (Surprise 0.041) Supported: evidence confirms that fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 7 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** To map the spatial distribution of U-Net failure modes by linking ground-truth anatomical density to the rate of over-segmentation.
- **Conclusion:** A Pearson Correlation Coefficient of 0.4001 (p-value < 0.0001) was observed, indicating a moderate, statistically significant positive correlation. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Pearson; p ≈ 0.0001
- **Statistical issues:** none
- **Logic issues:** none

### 318. (Surprise 0.041) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 38 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** To determine if the distance from the soma is a statistically significant predictor of whether a local error is a split or a merge.
- **Conclusion:** The results yielded a K-S statistic of 0.2934 and a highly significant p-value of 8.65e-33, confirming that the distance from the soma is a statistically significant predictor differentiating the spatial distributions of split and merge errors. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov; p ≈ 8.6e-33
- **Statistical issues:** none
- **Logic issues:** none

### 319. (Surprise 0.041) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 50 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Determine if uncharacteristic fluctuations in the U-Net predicted neurite radius can serve as an intrinsic topological signature to detect merge errors.
- **Conclusion:** The mean radius variance for Merged fragments was found to be 0.009193, while for Clean fragments it was 0.007919. A one-sided Mann-Whitney U test yielded a test statistic of 18379.0 with a p-value of 0.0275. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.0275
- **Statistical issues:** none
- **Logic issues:** none

### 320. (Surprise 0.041) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 7 · **Belief:** Leaning True -> Leaning True (0.7917 -> 0.7656) · **Direction:** Negative
- **Tested:** Evaluate if local 3D tangent alignment is a robust geometric feature for distinguishing between true structural gaps and false connections during split correction.
- **Conclusion:** Valid pairs exhibited a mean dot product of -0.6956 (indicating anti-parallel alignment, as endpoints meet head-on), while the 3 Invalid pairs had a mean dot product of 0.6700. The resulting p-value was ~0.0097 (p < 0.05), suggesting statistical significance. Belief therefore dropped from Leaning True (0.79) to Leaning True (0.77) (signed surprisal -0.041; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Welch's t-test, Student's t-test; p = 0.05; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 321. (Surprise 0.041) Supported: evidence confirms that neurite tortuosity or sharp turns flag U-Net split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 10 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Evaluate the relationship between neurite tortuosity and U-Net segmentation continuity.
- **Conclusion:** Correlation tests showed a very weak but statistically significant positive monotonic relationship (Spearman's r = 0.0447, p < 0.001), while the linear correlation was not significant (Pearson's r = 0.0065, p = 0.426). Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Spearman, Pearson, logistic regression; p = 0.426
- **Statistical issues:** none
- **Logic issues:** none

### 322. (Surprise 0.041) Supported: evidence confirms that Z-axis anisotropy or imaging depth drives split-error geometry.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 29 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Assess if imaging anisotropy introduces a predictable directional bias in the orientation of split errors.
- **Conclusion:** Statistical analysis of the 3D gap vectors for these splits revealed a significant deviation from a theoretically uniform spherical distribution (Mean |cos(θ)| = 0.4644 vs. expected 0.5000; KS-Statistic = 0.1008; p-value = 5.82e-04). Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Kolmogorov-Smirnov; p ≈ 0.00058
- **Statistical issues:** none
- **Logic issues:** none

### 323. (Surprise 0.041) Supported: evidence confirms that morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 34 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Determine if topological split errors systematically concentrate in thin, distal branches relative to the soma by using a topologically inferred soma root to overcome the lack of explicit soma annotations and uniform radius artifacts.
- **Conclusion:** The split locations have a higher mean (9,859.71 µm) and median (9,656.38 µm) distance compared to the internal non-split nodes (mean = 9,131.46 µm, median = 8,770.20 µm). The Mann-Whitney U test confirms this difference is highly statistically significant (p-value = 2.5753e-65). Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 1; n up to 9131
- **Statistical issues:** none
- **Logic issues:** none

### 324. (Surprise 0.041) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 54 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Determine if branching density is a reliable morphological indicator of merge errors in the U-Net reconstruction.
- **Conclusion:** The results show that Merged fragments have a higher median branching density (~1.34e-3 branches/µm) compared to Clean fragments (~8.70e-4 branches/µm). Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Questionable
- **Test:** Mann-Whitney U, Student's t-test; p = 0.0252
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 325. (Surprise 0.041) Refuted: data contradict the claim that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 71 · **Belief:** Leaning True -> Leaning True (0.7917 -> 0.7656) · **Direction:** Negative
- **Tested:** Investigate whether the geometric angle of branches in the U-Net graph can identify merge errors without requiring external image features.
- **Conclusion:** The Fisher's Exact Test yielded an odds ratio of 2.3781 and a p-value of 0.0374. Since the p-value is below the standard 0.05 significance threshold, the hypothesis is statistically supported: branch points with an acute branching angle are significantly more likely to be false merges (such as crossing fibers) compared to branches with exclusively obtuse angles. Belief therefore dropped from Leaning True (0.79) to Leaning True (0.77) (signed surprisal -0.041; negative shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Fisher's exact, Student's t-test; p = 0.0374
- **Statistical issues:** none
- **Logic issues:** none

### 326. (Surprise 0.041) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 74 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Verify the geometric signature of recoverable splits by quantifying the predictive power of 3D tangent alignment for leaf-to-leaf reconnections.
- **Conclusion:** The Chi-squared test comparing the two cohorts returned a chi-square value of 3.7307 and a p-value of 0.0534. The marginally non-significant p-value suggests that while geometric alignment is a perfect indicator of valid reconnections, relying on it as a strict filter (dot < -0.7) might omit a significant portion of true reconnections that do not possess strong anti-parallel alignment. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p = 0.0534
- **Statistical issues:** none
- **Logic issues:** none

### 327. (Surprise 0.041) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 78 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Test if the spatial clustering of fragmentation (splits) correlates with the occurrence of erroneous fusions (merges), suggesting a shared root cause in complex local topology.
- **Conclusion:** By querying the local neighborhood (30 µm radius) for fragment leaf nodes, it found that True Branches had a mean leaf count of 0.16, while Merge Errors had a higher mean count of 0.29 (with both medians at 0.00). A one-sided Mann-Whitney U test indicated that the local density of fragment endpoints is significantly higher around Merge Errors than True Branches (U = 128388.5, p-value = 0.0109). Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** median outcome is zero; effect driven by outliers
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.0109
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 328. (Surprise 0.041) Supported: evidence confirms that image-intensity features (signal across the gap) add discriminative power.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 84 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Validate the use of raw image fluorescence intensity profiles between endpoints as a discriminative feature for split correction.
- **Conclusion:** The median intensity for valid pairs was higher (70.00) compared to invalid pairs (36.50). However, due to the extremely small sample size of invalid pairs (n=3), the Mann-Whitney U test failed to reach statistical significance (p = 0.1812). Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Mann-Whitney U; p = 0.1812; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 329. (Surprise 0.041) Supported: evidence confirms that greedy/independent thresholding versus joint-reasoning matching changes proofreading precision.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 90 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Quantify the frequency of mutual exclusivity conflicts at endpoints and evaluate the failure rate of a naive shortest-distance reconnection policy in dense local neighborhoods.
- **Conclusion:** A binomial test against the 10% failure threshold yielded a statistically significant p-value (p = 0.01). While the statistical result supports the hypothesis and highlights the necessity for multivariate/GNN features rather than naive distance metrics, it is important to note the extremely small sample size (n=2) of such conflicts in this specific dataset region, which limits the broader generalizability without further testing on larger volumes. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** binomial test, Student's t-test; p = 0.01; n up to 2
- **Statistical issues:** none
- **Logic issues:** none

### 330. (Surprise 0.041) Supported: evidence confirms that local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 100 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Determine if variance in predicted neurite radius can be used as an intrinsic heuristic to detect and flag automated merge errors without requiring ground truth.
- **Conclusion:** The script correctly mapped the fragment nodes to ground truth components using the KD-tree, classified the fragments into 11 'Merged' fragments and 981 'Clean' fragments, and evaluated their radius variance and radius gradient (mean absolute difference). The statistical analysis using the Mann-Whitney U test yielded the following results: - **Radius Variance**: Merged Mean = 0.010240, Clean Mean = 0.007822. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Mann-Whitney U, Student's t-test; p = 0.06268
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 331. (Surprise 0.041) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 104 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Determine if neurite radius asymmetry at degree-3 nodes can reliably distinguish false merges from true bifurcations.
- **Conclusion:** The radial asymmetry (ratio of max to min mean branch radius) was calculated for both populations. The descriptive statistics showed a mean asymmetry of 1.1132 for False Merges and 1.0783 for True Branches. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Student's t-test; p = 0.05
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 332. (Surprise 0.041) Supported: evidence confirms that branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 127 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Assess whether local geometric branching angles can reliably differentiate between biological branches and U-Net merge errors.
- **Conclusion:** The analysis revealed that Merge Branches have a smaller (more acute) mean minimum branching angle (70.82° ± 24.65°) compared to True Branches (78.80° ± 21.35°). The Mann-Whitney U test yielded a statistically significant p-value of ~0.0229 (alpha < 0.05), indicating that merge errors do indeed tend to present with more acute geometric angles than true biological branches. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.0229
- **Statistical issues:** none
- **Logic issues:** none

### 333. (Surprise 0.041) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 130 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** To model the degradation of geometric orientation cues over distance, demonstrating the necessity of iteratively refreshing features and utilizing non-linear/image-aware logic for resolving large gaps.
- **Conclusion:** The experiment successfully investigated the degradation of angular agreement (tangent cosine similarity) as a function of physical gap distance for valid U-Net fragment split reconnections. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** test type not extractable from record
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 334. (Surprise 0.041) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 137 · **Belief:** Leaning True -> Leaning True (0.7917 -> 0.7656) · **Direction:** Negative
- **Tested:** Evaluate tangent vector agreement as a feature to reject false split-correction proposals in cycle-prevention logic.
- **Conclusion:** The valid group consisted of 1,204 pairs with an average alignment angle of 137.39° (mean dot product -0.6686), indicating a general tendency towards anti-parallel alignment as expected for continuous broken neurites. The invalid group consisted of only 5 pairs, with an average angle of 92.36° (mean dot product -0.0087), suggesting near orthogonal orientation. Belief therefore dropped from Leaning True (0.79) to Leaning True (0.77) (signed surprisal -0.041; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Unsound
- **Test:** Welch's t-test, two-sample t-test, Student's t-test; p = 0.05; n up to 5
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** treats non-significant result with single-digit minority class as evidence the null is true (absence of evidence is not evidence of absence)

### 335. (Surprise 0.041) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 138 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Evaluate if skeleton geometry (tangent alignment) can reliably resolve competing proposals when a leaf node faces multiple exclusive reconnection options.
- **Conclusion:** The mean cosine similarity for Correct targets was 0.1459, whereas for Incorrect targets it was 0.6653 (where -1.0 indicates perfect anti-parallel alignment). While the correct targets exhibited better alignment (lower cosine similarity) than the incorrect targets, the very small sample size (n=2) resulted in a Wilcoxon signed-rank test p-value of 0.50. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Wilcoxon; p = 0.5; n up to 2
- **Statistical issues:** none
- **Logic issues:** none

### 336. (Surprise 0.041) Supported: evidence confirms that image-intensity features (signal across the gap) add discriminative power.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 146 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Test if local image evidence (fluorescence intensity) in the gap between endpoints can statistically distinguish anatomically valid reconnections from invalid ones.
- **Conclusion:** The results show that valid gaps have a mean fluorescence intensity of 109.03, while invalid gaps have a mean intensity of 36.17. A one-sided Mann-Whitney U test demonstrated that valid gaps are statistically brighter than invalid gaps (p-value = 0.046, p < 0.05), thus supporting the hypothesis. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U; p = 0.05
- **Statistical issues:** none
- **Logic issues:** none

### 337. (Surprise 0.041) Supported: evidence confirms that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 159 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Test if local path angularity can serve as a purely geometric flag for merge errors, independent of image features.
- **Conclusion:** The mean angular deviation for merge boundaries was 20.04 degrees (median 16.88°), compared to 18.84 degrees (median 16.34°) for the control group. A Welch's t-test yielded a t-statistic of 4.8135 and a highly significant p-value of 1.52e-06. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Welch's t-test, Student's t-test; p ≈ 1.5e-06
- **Statistical issues:** none
- **Logic issues:** none

### 338. (Surprise 0.041) Supported: evidence confirms that local crowding/leaf-density predicts merge or false reconnections.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 195 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7344) · **Direction:** Positive
- **Tested:** Test if local fragment density can serve as a contextual feature to penalize proposals in crowded regions where distance alone is misleading.
- **Conclusion:** Statistical testing yielded an independent t-test p-value of 0.2335 and a Mann-Whitney U test p-value of 0.0010. Belief therefore rose from Leaning True (0.71) to Leaning True (0.73) (signed surprisal +0.041; positive shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Sound
- **Test:** Mann-Whitney U, two-sample t-test, Student's t-test; p = 0.001
- **Statistical issues:** none
- **Logic issues:** none

### 339. (Surprise 0.035) Refuted: data contradict the claim that tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 41 · **Belief:** Likely True -> Likely True (0.8333 -> 0.8077) · **Direction:** Negative
- **Tested:** Determine if local geometric alignment (directional cosine similarity) can effectively discriminate between valid split reconnections and false-positive spatial neighbors.
- **Conclusion:** The code mapped fragment leaf pairs within a 20 µm radius to their corresponding Ground Truth components to classify them as 'valid' (N=395) or 'invalid' (N=3) reconnections. **Experiment Results:** - **Valid Pairs (N=395):** Exhibited a mean cosine similarity of -0.7726 (heavily concentrated between 140° and 180°), strongly indicating anti-parallel geometry. Belief therefore dropped from Likely True (0.83) to Likely True (0.81) (signed surprisal -0.035; negative shift).
- **Caveats:** very small minority class (extreme class imbalance limits power)
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.0047; n up to 395
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered
- **Logic issues:** none

### 340. (Surprise 0.000) Inconclusive on whether branching density or branch-angle geometry predicts merge errors.
- **Run:** exa-spim-run-1_2026-06-03 · **ID:** 20 · **Belief:** Leaning True -> Leaning True (0.7500 -> 0.7500) · **Direction:** Neutral
- **Tested:** To test whether high-degree nodes are strong topological signatures of merge errors caused by intersecting neurites.
- **Conclusion:** The Chi-square test of independence yielded a statistic of 5.1989 and a p-value of 0.0226. Since the p-value is less than 0.05, there is a statistically significant association between high-degree nodes and merge errors, supporting the hypothesis that anomalous high-degree nodes serve as topological signatures of merge errors, despite their overall rarity. Belief stayed at Leaning True (0.75) (signed surprisal +0.000; neutral).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** chi-square; p = 0.0226
- **Statistical issues:** none
- **Logic issues:** none

### 341. (Surprise 0.000) Inconclusive on whether image-intensity features (signal across the gap) add discriminative power.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 8 · **Belief:** Leaning True -> Leaning True (0.7500 -> 0.7500) · **Direction:** Neutral
- **Tested:** Determine if neurite thickness is a statistically significant predictor of automated segmentation split errors.
- **Conclusion:** The Mann-Whitney U test yielded a p-value of 1.0 (indicating identical distributions), and the logistic regression naturally failed because perfect multicollinearity between the constant and the uniform radius array caused the model to drop the radius coefficient, resulting in an index error when attempting to fetch the odds ratio. Belief stayed at Leaning True (0.75) (signed surprisal +0.000; neutral).
- **Caveats:** zero-variance data forced a degenerate test
- **Verdict:** Unsound
- **Test:** Mann-Whitney U, logistic regression; p = 1
- **Statistical issues:** test was degenerate or uninformative (NaN, zero variance, empty cohort, or constant inputs)
- **Logic issues:** none

### 342. (Surprise 0.000) Inconclusive on whether morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 23 · **Belief:** Leaning False -> Leaning False (0.2917 -> 0.2917) · **Direction:** Neutral
- **Tested:** Determine whether proximity to the soma correlates with an increased rate of automated segmentation fragmentation.
- **Conclusion:** The code successfully executed and overcame the previous unpickling issues by implementing a robust `CustomUnpickler` that dynamically mocks the missing `agentic_neuron_proofreader` classes. Belief stayed at Leaning False (0.29) (signed surprisal +0.000; neutral).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Wilcoxon
- **Statistical issues:** dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 343. (Surprise 0.000) Inconclusive on whether tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 64 · **Belief:** Leaning True -> Leaning True (0.7500 -> 0.7500) · **Direction:** Neutral
- **Tested:** Evaluate if the geometric alignment (tangent vectors) of fragment endpoints is a strong predictor for distinguishing true anatomical splits from erroneous reconnection proposals.
- **Conclusion:** The alignment analysis showed a striking difference: True Matches exhibited strong positive alignment (Mean = 0.8061, Median = 0.9391), meaning the tangents are facing each other, whereas the few False Matches showed negative alignment (Mean = -0.7433, Median = -0.6985). A Mann-Whitney U test yielded a p-value of 5.1857e-05, indicating a statistically significant difference between the two distributions. Belief stayed at Leaning True (0.75) (signed surprisal +0.000; neutral).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p ≈ 5.2e-05; n up to 3
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 344. (Surprise 0.000) Inconclusive on whether local neurite radius (or thickness asymmetry) is a useful biomarker for split/merge errors.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 93 · **Belief:** Leaning True -> Leaning True (0.7917 -> 0.7917) · **Direction:** Neutral
- **Tested:** Investigate the relationship between neuron morphology (neurite thickness) and the U-Net's segmentation continuity.
- **Conclusion:** However, the analysis revealed that the 'Thin' cohort (defined by the hypothesis as having a median radius < 0.3 µm) contained 0 components, while all 10,172 qualifying components fell into the 'Thick' cohort (median radius >= 0.3 µm) with a median split density of 2.19 leaves/mm. Belief stayed at Leaning True (0.79) (signed surprisal +0.000; neutral).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Mann-Whitney U
- **Statistical issues:** none
- **Logic issues:** none

### 345. (Surprise 0.000) Inconclusive on whether tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** exa-spim-run-2_2026-06-03 · **ID:** 140 · **Belief:** Leaning True -> Leaning True (0.7500 -> 0.7500) · **Direction:** Neutral
- **Tested:** Determine if the cosine similarity of tangent vectors at fragment endpoints is significantly higher for true reconnections compared to false reconnections.
- **Conclusion:** **Statistical Analysis (Orientation Agreement):** - **True Reconnections:** Mean Absolute Cosine Similarity = 0.7629, Median = 0.8300. - **False Reconnections:** Mean Absolute Cosine Similarity = 0.7274, Median = 0.8067. Belief stayed at Leaning True (0.75) (signed surprisal +0.000; neutral).
- **Caveats:** dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.00803
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 346. (Surprise 0.000) Inconclusive on whether morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 8 · **Belief:** Leaning True -> Leaning True (0.7083 -> 0.7083) · **Direction:** Neutral
- **Tested:** Quantify the relationship between neurite thickness (caliber) and the susceptibility of the automated ExaSPIM segmentation to split errors.
- **Conclusion:** As a result of this lack of variance in the independent variable (neurite caliber), the calculated Pearson and Spearman correlation coefficients evaluated to `NaN`, and the generated scatter plot displays all data points clustered vertically along the 1.0 µm axis. Belief stayed at Leaning True (0.71) (signed surprisal +0.000; neutral).
- **Caveats:** none noted
- **Verdict:** Unsound
- **Test:** Spearman, Pearson, Student's t-test; test statistic NaN / undefined
- **Statistical issues:** test was degenerate or uninformative (NaN, zero variance, empty cohort, or constant inputs)
- **Logic issues:** none

### 347. (Surprise 0.000) Inconclusive on whether morphological context (proximal vs distal, trunk vs arbor) governs error rates.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 13 · **Belief:** Uncertain -> Uncertain (0.5417 -> 0.5417) · **Direction:** Neutral
- **Tested:** Evaluate if artificial split errors are disproportionately concentrated in the proximal regions of neurons compared to their distal extensions.
- **Conclusion:** The code executed successfully, resolving the file path issues from the previous attempt. Belief stayed at Uncertain (0.54) (signed surprisal +0.000; neutral).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** paired t-test, Student's t-test
- **Statistical issues:** none
- **Logic issues:** none

### 348. (Surprise 0.000) Inconclusive on whether tangent/angular alignment beats or augments pure distance for reconnection.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 16 · **Belief:** Leaning True -> Leaning True (0.7500 -> 0.7500) · **Direction:** Neutral
- **Tested:** Evaluate tangent vector angular alignment as a geometric filtering feature to prioritize valid reconnections and prevent cycle-inducing false merges.
- **Conclusion:** The results revealed that valid reconnections have a mean angular difference of 47.72° ± 36.14°, whereas false reconnections have a much higher mean of 108.72° ± 35.96°. The Mann-Whitney U test yielded a p-value of ~0.0016, confirming a statistically significant difference and supporting the hypothesis that valid split reconnections are generally more linearly aligned. Belief stayed at Leaning True (0.75) (signed surprisal +0.000; neutral).
- **Caveats:** very small minority class (extreme class imbalance limits power); dataset loaded via mock unpickler bypassing real package dependencies
- **Verdict:** Questionable
- **Test:** Mann-Whitney U; p = 0.0016; n up to 479
- **Statistical issues:** minority class very small (single-digit n) — test severely underpowered; dataset loaded via mock unpickler — fidelity of mocked classes to real package is unverified
- **Logic issues:** none

### 349. (Surprise 0.000) Inconclusive on whether fragment cable length predicts validity of reconnection proposals or merge concentration.
- **Run:** refinement-of-exa-spim-run-1_2026-06-03 · **ID:** 48 · **Belief:** Leaning True -> Leaning True (0.7500 -> 0.7500) · **Direction:** Neutral
- **Tested:** Test if the size (and presumed confidence) of disconnected fragments correlates with the physical size of the split error separating them.
- **Conclusion:** The statistical analysis yielded a Spearman rank correlation coefficient of -0.1105 with a p-value of 0.0039. While the p-value indicates a statistically significant negative correlation (supporting the directional premise of the hypothesis), the effect size (rho = -0.11) is extremely weak. Belief stayed at Leaning True (0.75) (signed surprisal +0.000; neutral).
- **Caveats:** none noted
- **Verdict:** Sound
- **Test:** Spearman; p = 0.0039
- **Statistical issues:** none
- **Logic issues:** none

## Statistical Verification — Run-wide Summary

**Hypotheses audited:** 349 (all 349 records pooled from the three discovery runs).

**Verdict breakdown:** Sound = 215; Questionable = 68; Unsound = 66.

### Cross-cutting patterns

- **Degenerate / NaN tests interpreted as evidence (5 entries).** The loop ran tests on zero-variance inputs (all annotator terminal radii identical at 1.0 µm; logistic regression with perfectly collinear features; greedy precision-recall curve with zero false matches) and still updated belief from the NaN / p=1.00 output. The most consequential examples drive the very top of the ranked list: #9 (s=0.77), #341 (s=0.00), #346 (s=0.00), #1 (s=0.97), #29 (s=0.75). These are the loop's headline 'refutations' and they are not supported by any test that actually ran.

- **Tiny minority class treated as null support (37 entries).** Many split-vs-merge experiments had only 3–10 False Positive / False Merge / orphan samples after filtering. Mann-Whitney U, Welch's t-test, Fisher's exact and chi-square on these samples have negligible power. The loop nonetheless reads p>0.05 as 'the feature is not useful' for radius continuity, tangent alignment, leaf density, crowding, cycle-formation, etc. 'Failed to reject H₀' is being treated as 'H₀ is true' across many entries that share this exact pattern (representative: #271 (s=0.20), #308 (s=0.06), #54 (s=0.69), #55 (s=0.69), #156 (s=0.45)).

- **Group-comparison tests with ≈2 observations per group (19 entries) drive the 'no annotator bias' refutations.** Eight annotators are grouped over only 18–19 ground-truth neurons, giving Kruskal-Wallis / ANOVA samples of size 1–4 per annotator. High p-values follow mechanically (e.g. ANOVA p=0.946, Kruskal p=0.912 in entry #2; ANOVA p=0.7782 in #8; H-test p=0.6542 in #26). These are exactly what an underpowered test produces; using them to refute the annotator-bias hypothesis is the absence-of-evidence fallacy on repeat.

- **Significance driven by N rather than effect size (5 entries).** When tens of thousands of nodes are pooled, correlations of |r|≈0.02 reach p<10⁻³⁰. Some entries acknowledge this in prose but the belief shift is identical to that for a real effect. Representative: #53 (s=0.69), #60 (s=0.69), #21 (s=0.75), #315 (s=0.04), #205 (s=0.31).

- **McNemar's tests on identical methods (1 entries).** Where the two compared procedures produce no discordant predictions, McNemar's p=1.0 by construction — meaning 'the methods agree on everything observable,' not 'there is no difference between them.' Affected: #9 (s=0.77).

- **Mock-unpickler dataset loading (222 of 349 entries).** Records construct empty `Mock*` / `CustomUnpickler` classes solely to deserialise the dataset cache. Where experiments only touch the public attributes of `Graph` / numpy arrays this is harmless; where experiments rely on package-specific methods (e.g. `agentic_neuron_proofreader` topology helpers) the test is silently exercising the mock rather than the real pipeline. The 69 entries the summarizer pre-flagged are echoed in our Questionable verdict; the others are not individually flagged but should be considered a systemic caveat.

### Multiple comparisons (Benjamini–Hochberg FDR)

Of the 314 headline p-values extracted across all three runs:

- 121 are p < 10⁻³; 26 are 10⁻³ ≤ p < 10⁻²; 17 are 10⁻² ≤ p < 0.05; 36 are 0.05 ≤ p < 0.10; 114 are p ≥ 0.10.
- Bonferroni at α=0.05 (per-test threshold 0.00016) leaves 111 surviving results.
- Benjamini–Hochberg at q=0.05 leaves 159 surviving findings (largest surviving p = 0.0252).
- Benjamini–Hochberg at q=0.10 leaves 190 surviving findings (largest surviving p = 0.0565).

Of 35 borderline-significant results (0.001 < p < 0.05), 5 fail BH q=0.05 control: entry #319 (p≈0.0275, surprise +0.041), entry #109 (p≈0.0286, surprise -0.690), entry #316 (p≈0.0286, surprise +0.044), entry #325 (p≈0.0374, surprise -0.041), entry #305 (p≈0.0426, surprise +0.089).

For context, an expected number of false-positives at α=0.05 with 314 tests is ~16. Most of the run's 'significant' p-values are p < 10⁻⁴ — too extreme to be FDR artefacts — but they are driven almost entirely by the very large pooled N (tens of thousands of nodes / fragment pairs). FDR survival therefore does NOT establish biological or algorithmic meaningfulness; it only rules out family-wise multiple-comparison fluke.

### Where the gravest problems concentrate

- **The top-ranked 'refutations' are the most fragile.** Entry #1 (surprise -0.966) and several other top-of-list entries are Unsound because the test under them was degenerate, the minority class was 5–6 samples, or the per-group N was ~2. The summary's headline narrative — 'simple geometric features fail to predict U-Net errors' — rests on these entries; the conclusion may still be correct, but the evidence the loop produced for it is weak.

- **Annotator-bias refutations dominate the high-surprise tail and are systematically underpowered.** Entries #2, #4, #8, #26, #27, #29 and others all use 8-group Kruskal/ANOVA with ~2 components per group. None of them have the statistical power to detect plausibly sized annotator effects, so 'no annotator bias' is essentially uninformed by these experiments.

- **'Greedy-vs-joint matching' refutations rely on tests that did not actually run.** Entries #1, #9 and #19 reach their conclusions from a Welch's t-test that bypassed itself (zero variance), a McNemar's test with no discordant pairs, and an experiment whose deliverables exist without a real comparator. These should not have produced a belief shift in either direction.

- **Large-N significance does NOT rescue a weak effect.** Entries #13, #14, #17, #69 and several entries in run-2 report p ≈ 10⁻²⁰ for r ≈ 0.01–0.1. These are not informative biomarkers despite the p-value; the loop occasionally acknowledges this but the belief update is symmetric with high-effect-size findings.

- **For most low-surprise (≈0) entries the audit is Sound or Questionable but the belief did not move anyway**, so even when the test is weak the run-level harm is limited; the danger concentrates in the top-of-list entries that drive the report's narrative.
