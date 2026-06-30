# AutoDiscovery Run Summary — Ranked by Posterior-Surprise Priority

## Header

- **Source file:** `autodiscovery/run-5--ground-truth-error-annotations-revised-version_2026-06-25.json`
- **Per-file hypothesis count:** 110 hypotheses (run-5--ground-truth-error-annotations-revised-version_2026-06-25)
- **Ranking key (`rank_by`):** `posterior-surprise` — `priority_score = posterior * |surprisal|`
- **Ranked:** 106 of 110 hypotheses carried a usable surprisal score (`n_ranked = 106`)
- **Returned in this report:** top 20 (`n_returned = 20`); this report shows the **top 20 of 106** ranked hypotheses after applying `--top 20`
- **Surprise-magnitude range across ranked records:** 0.0000 to 0.4707
- **Max priority score:** 0.4091
- **Excluded (no surprisal score):** 4 hypotheses — see note at end

### Synthesis of headline takeaways

This run converged on a coherent, actionable picture of where the automated U-Net neuron reconstruction fails and how an agentic proofreader should target those failures. The single highest-priority belief flip (#1, ID 15, priority 0.409, surprise 0.471) lifted an *Uncertain* prior to *Likely True*: merges occur overwhelmingly at **crossing fibers** (median crossing angle 74.9°, p = 0.0081), not parallel touching fibers — a geometric prior the loop did not previously hold confidently. The next tier of high-priority, strongly-confirmed results (all priority 0.275) cluster around two cross-cutting themes. First, **splits and omits are spatially clustered, not random** (IDs 26, 40, 45, 50, 27, 42): observed nearest-neighbor distances (~130–140 µm) run roughly half the random-null expectation (~300–320 µm) at p-values approaching zero, and omit errors are bursty (84.6% conditional vs. 2.2% baseline), forming localized "error cascades" near branch points and leaf tips. Second, **merge segments are massive and asymmetric** (IDs 22, 34, 55, 56): merging segments are ~15× longer than correct segments (AUC 0.869 as a classifier) and overlap one "primary" neuron ~91% of the time, so merge corrections reduce to cheaply-detectable pruning of minor offshoots. Together these establish that cable length is a strong, cheap merge prior, and that splits/omits should be repaired by absorbing nearby micro-fragments (86.8% of splits touch a sub-100 µm fragment; 95.8% of omit gaps are below the 100 µm filter threshold).

---

## Reproduction — Summary

- **Dataset pkl used:** `/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/cache/dataset_cache_794495_mcl100_add.pkl`
- **Breakdown (of n_rerun = 20):** REPRODUCED 20 · DIVERGED 0 · FAILED 0.
- **Code source:** all 20 ran on **revised-loading** code (loading redirected to a direct `$RERUN_PKL` load; the statistical analysis was left byte-for-byte identical in every case). 0 ran on recorded code.
- **Findings that did NOT reproduce:** none. Every headline number (test statistic, p-value, effect size, sample counts) matched the recorded output exactly, except for ID 45 where the random-permutation Monte Carlo introduced trivial seed noise (expected NN 319.96 vs recorded 320.13 µm; t = -15.0355 vs -14.93; p = 1.2391e-11 vs 1.40e-11) that does not change the significant-clustering conclusion → still REPRODUCED.
- **Notable recovery:** ID 12 (split-gap bimodality) had RECORDED as a failure (exitcode 1, "Sandbox output was not valid JSON" — an infrastructure error, not an analysis result). The revised-loading rerun ran cleanly and produced the predicted bimodal GMM (Component 1 weight 0.6357, mean ~0.0 µm; Component 2 weight 0.3643, mean 159.18 µm) over n = 7,988 split gaps, so the finding is now REPRODUCED.
- **Loading revisions:** all 20 records had their dataset search/load replaced with a direct `$RERUN_PKL` load (the recorded scripts used hardcoded relative paths such as `../data/dataset_cache_794495_mcl100_add.pkl` or glob patterns like `dataset_cache_*_add.pkl` / `*_add.pkl` that would not resolve in the rerun sandbox). The analysis code and all prints were unchanged.
- **Environment failures:** none — no import/native-load errors; preflight passed (`preflight_ok: true`, `environment_failure: false`, `n_env_failed: 0`).

---

## Generalization — Summary

- **Origin dataset:** `cache/dataset_cache_794495_mcl100_add.pkl`. **Extra datasets used (each finding's reproduced code re-run on data it was NOT generated on):** `cache/dataset_cache_794491_mcl100_add.pkl` and `cache/dataset_cache_789202_mcl100_add.pkl`. All 20 scripts executed successfully (exitcode 0, no timeouts) on both extra pkls — no loading/env failures, so every verdict is grounded in actual numbers.
- **Breakdown (of 20):** GENERALIZES 17 · PARTIAL 2 · DOES-NOT-GENERALIZE 1 · INCONCLUSIVE 0.

**Findings that do NOT fully generalize:**

- **DOES-NOT-GENERALIZE — Entry 13 / ID 42 (split→omit local error cascade):** origin 794495 baseline 0.0223 vs conditional 0.0528, Wilcoxon p=1.14e-05 (significant). ds_794491 baseline 0.0437 vs conditional 0.0398, p=5.70e-01 — direction flips and not significant. ds_789202 baseline 0.0396 vs conditional 0.0477, p=2.04e-01 — same direction but not significant. The effect disappears (and on one dataset reverses) on both extra datasets; this conclusion is specific to the origin.
- **PARTIAL — Entry 1 / ID 15 (merge crossing angle > 45°):** origin 794495 median 74.93°, Wilcoxon p=8.06e-03, n=12. ds_794491 median 54.30°, p=5.37e-02, n=25 — same direction but loses significance at α=0.05. ds_789202 produced 0 valid merge sites, so the test could not run there. Only one usable extra dataset and it is marginal → weak, partial support.
- **PARTIAL — Entry 20 / ID 70 (split rate inversely correlates with distance to leaf tip):** origin 794495 logistic coef=-0.1361, p<0.001. ds_789202 coef=-0.3167, p≈0.0 (holds, stronger). ds_794491 coef=-0.0095, p=0.409 — slope essentially zero, not significant (lost). Holds on one extra dataset, lost on the other.

**Synthesis:** With two extra datasets tested, the run's core conclusions are largely robust: 17 of 20 findings reproduce the same direction and significance on both additional brains. The robust, dataset-independent picture covers (a) splits cluster near branch points and along the cable (IDs 3, 26, 33, 40, 45), (b) omits are bursty/zero-distance-adjacent and dominated by sub-100 µm gaps (IDs 27, 50, 69), and (c) merge segments are massive and asymmetric, making size a reliable merge prior (IDs 22, 34, 55, 56), plus the radius-constraint safety trade-off (ID 47, on its direct-pair metric). Three conclusions are weaker or dataset-specific: the split→omit co-occurrence cascade (ID 42) does NOT generalize at all (vanishes or reverses on both extra datasets); the merge crossing-angle finding (ID 15) is only weakly supported because one extra dataset yielded no merge sites and the other was only marginal (p≈0.054); and the leaf-tip split-rate gradient (ID 70) is significant on one extra dataset but flat/non-significant on the other. Because only two extra datasets were available, GENERALIZES verdicts should be read as "consistent across three brains total" rather than population-level proof, but the unanimous agreement on the 17 robust findings (often with comparable or larger effect sizes) is reasonably strong evidence.

---

## Ranked Conclusions (highest priority first)

### 1. (Priority 0.409 · Surprise 0.471) Merges happen at crossing fibers, with crossing angles significantly steeper than 45°.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 15 · **Belief:** Uncertain → Likely True (0.5417 → 0.8690) · **Direction:** Positive
- **Tested:** Whether the 3D crossing angle between two ground-truth neurons at a merge site is significantly greater than 45°, indicating merges form where fibers cross rather than run parallel and touch.
- **Conclusion:** Across 12 valid ground-truth merge sites, the intersection-angle distribution had a mean of 65.97° and a median of 74.93°, with angles clustering between 65° and 90°. A one-sample Wilcoxon signed-rank test against a 45° benchmark gave a statistic of 69.00 and p = 0.0081, rejecting the null. The positive surprisal (+0.471) reflects the largest belief shift in the run — an *Uncertain* prior was lifted to *Likely True*, confirming merges are predominantly a crossing-fiber geometry problem, which directly informs how a proofreader should reason about candidate fusions.
- **Caveats:** Sample is small (only 12 valid merge sites), so the angle estimate, while significant, rests on limited data; the nonparametric test is appropriate but power is modest.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded Wilcoxon stat=69.00, p=8.0566e-03, n=12, mean=65.97°, median=74.93°; rerun stat=69.00, p=8.0566e-03, n=12, mean=65.97°, median=74.93° → exact match. Loading was redirected to a direct `$RERUN_PKL` load.
- **Generalization:** PARTIAL
- **Across datasets:** origin 794495: median 74.93°, Wilcoxon stat=69.00, p=8.06e-03, n=12 (significant >45°). ds_794491: median 54.30°, stat=223.00, p=5.37e-02, n=25 — same direction (median > 45°) but loses significance at α=0.05. ds_789202: 0 valid merge sites found, so the test could not run there at all. Direction holds on the one dataset that produced data, but it is only marginal (p≈0.054) and the other extra dataset yielded no testable sites — weak evidence given only one usable extra dataset.
- **Verdict:** WEAK
- **Test:** One-sample Wilcoxon signed-rank test of crossing angle against a 45° benchmark, statistic=69.00, p=0.0081, n=12 merge sites. The nonparametric one-sample test is the correct choice for a small, possibly non-normal angle sample.
- **Statistical issues:** Severely underpowered (n=12); the angle estimate rests on a dozen sites and the test has little power to characterize the distribution. The 45° benchmark is an arbitrary reference rather than a data-derived null.
- **Logic issues:** none in the origin reasoning (correctly rejects at α=0.05 and does not claim the null is true), but the conclusion "merges are predominantly a crossing-fiber geometry problem" generalizes beyond 12 sites.
- **Verdict rationale:** Right test, but tiny n plus PARTIAL generalization (one extra dataset yielded 0 testable sites, the other was only marginal at p≈0.054) means the headline belief flip is not robustly supported — defensible but heavily caveated.

### 2. (Priority 0.275 · Surprise 0.300) Split errors cluster near ground-truth branch points (~1.6× elevated risk).
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 3 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether split errors are more likely within 15 µm of a ground-truth branch point than on unbranched internode segments, implying bifurcations locally degrade U-Net segmentation.
- **Conclusion:** Branch-proximal edges (≤15 µm) contained a split in 0.90% of cases (741 / 82,348) versus 0.57% for branch-distal edges (7,247 / 1,281,441), a relative risk of 1.59 (Chi-square p = 4.90e-34). The very large edge counts and tiny p-value make this a robust positive confirmation, reinforcing the prior (positive surprisal +0.300) that neuronal bifurcations are a localized failure mode for split errors.
- **Caveats:** Absolute split rates are low (<1%) in both bins, so while the relative effect is highly significant, the practical per-edge difference is modest.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded proximal 741/82,348 (0.90%), distal 7,247/1,281,441 (0.57%), RR=1.5911, Chi-square p=4.9035e-34; rerun identical (741/82,348, 7,247/1,281,441, RR=1.5911, p=4.9035e-34) → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: RR=1.5911, proximal 0.90% vs distal 0.57%, χ² p=4.90e-34. ds_794491: RR=2.5783, proximal 3.20% vs distal 1.24%, p=1.74e-246. ds_789202: RR=3.2391, proximal 1.43% vs distal 0.44%, p=1.15e-256. Same direction (branch-proximal > branch-distal) and highly significant on both extra datasets, with effect size (relative risk) even larger than the origin.
- **Verdict:** SOUND
- **Test:** Chi-square test of independence on a 2×2 count table (split vs no-split × proximal vs distal), p=4.90e-34, n=82,348 proximal and 1,281,441 distal edges, relative risk 1.59. Chi-square is the correct test for binary counts; expected cell counts are very large so the asymptotic approximation is valid.
- **Statistical issues:** Edges within a neuron are not strictly independent (spatial autocorrelation), which can inflate the chi-square; absolute rates are <1% in both bins so the practical per-edge difference is modest despite the large RR.
- **Logic issues:** none — the conclusion (bifurcations locally degrade segmentation) matches a correctly rejected null, and the report flags the small absolute effect.
- **Verdict rationale:** Correct test on a large table, exact reproduction, and the same direction with even larger RR on both extra datasets; the within-neuron dependence is a minor caveat that does not overturn an effect this large and this consistent.

### 3. (Priority 0.275 · Surprise 0.300) Split gaps are bimodal: tiny single-dropout gaps plus large missed-stretch gaps.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 12 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether the physical gaps causing split errors follow a bimodal distribution representing two distinct failure modes — small single-dropout gaps and large omit-stretch gaps.
- **Conclusion:** Across 7,988 valid split gaps, a 2-component Gaussian Mixture Model confirmed bimodality: Component 1 (~63.6% of splits) has a mean gap of ~0.0 µm (negligible displacement, e.g. adjacent voxels labeled differently), and Component 2 (~36.4%) has a mean gap of ~159.2 µm with a wide standard deviation (~259.0 µm), characteristic of larger missed stretches. The positive surprisal (+0.300) confirms two distinct scale domains, implying an agentic connector should use two search radii rather than one.
- **Caveats:** GMM component assignment is model-dependent; the large standard deviation of Component 2 (~259 µm) means the "large-gap" mode is broad and overlaps a wide range of distances.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** The originally recorded run had FAILED (exitcode 1, "Sandbox output was not valid JSON" — an infrastructure/sandbox error, not an analysis result), so no recorded numbers exist. The revised-loading rerun succeeded and produced the bimodal GMM the hypothesis predicts: n=7,988 split gaps; Component 1 weight 0.6357, mean ~0.0 µm, std 0.0010 µm; Component 2 weight 0.3643, mean 159.18 µm, std 259.01 µm → conclusion (bimodal tiny-gap + large-gap modes) confirmed. Counts as REPRODUCED via recovery from the recorded sandbox failure.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: 2-component GMM, Comp1 weight 0.6357 mean ~0.0 µm, Comp2 weight 0.3643 mean 159.18 µm (std 259.0), n=7,988. ds_794491: Comp1 weight 0.5698 mean ~0.0 µm, Comp2 weight 0.4302 mean 117.47 µm (std 149.1), n=7,847. ds_789202: Comp1 weight 0.7391 mean 0.80 µm, Comp2 weight 0.2609 mean 328.73 µm (std 464.8), n=6,805. Both extra datasets reproduce the bimodal structure — one near-zero single-dropout mode plus one large missed-stretch mode — with comparable component weights; the large-gap mean varies (117–329 µm) but the two-mode conclusion holds throughout.
- **Verdict:** WEAK
- **Test:** 2-component Gaussian Mixture Model fit (Comp1 weight 0.6357 mean ~0.0 µm, Comp2 weight 0.3643 mean 159.18 µm, n=7,988). This is a descriptive model fit, not a hypothesis test — there is no p-value, no comparison against a 1-component model (e.g. via BIC/likelihood-ratio), so "bimodality" is asserted by construction.
- **Statistical issues:** A 2-component GMM was imposed rather than selected; fitting k=2 to any continuous distribution will return two components regardless of true modality. No model-selection statistic (BIC, AIC, LRT vs k=1) is reported to justify two modes over one skewed mode.
- **Logic issues:** Mild overreach — concluding "two distinct failure modes" from a fitted-but-unselected mixture; Comp1 at mean ~0.0 µm with std ~0.001 may be a near-degenerate spike rather than a genuine second physical mode.
- **Verdict rationale:** The finding reproduced and generalizes descriptively, but because no statistical test established bimodality over a unimodal alternative, the "two distinct scale domains" claim is defensible only as an exploratory observation.
- **Corrected test:** Formal model-selection test for number of modes (BIC/AIC for k=1..5, likelihood-ratio of k=2 vs k=1) PLUS a Hartigan dip test with a bootstrap p-value against a unimodal null — the original merely imposed a k=2 GMM, which returns two components for any continuous distribution regardless of true modality, so "bimodality" was asserted by construction with no p-value.
- **Corrected result:** Bimodality is now formally supported, not assumed. On the origin (n=7,988): BIC(k=1)−BIC(k=2)=114,816.7 (positive ⇒ k=2 strongly preferred), likelihood-ratio 2·(LL₂−LL₁)=114,843.7, and Hartigan dip statistic=0.4128 with bootstrap p=0.0020 (rejects unimodality at α=0.05). The k=2 GMM parameters are unchanged from the original revised run (Comp1 weight 0.6357 mean ~0.0 µm; Comp2 weight 0.3643 mean 159.18 µm). Original analysis reported NO test (descriptive k=2 GMM only).
- **Post-correction verdict:** UPHELD
- **Corrected generalization:** GENERALIZES — holds on both extra datasets. ds_794491 (n=7,847): BIC(k=1)−BIC(k=2)=95,997.1, LRT=96,024.0, dip=0.3945, bootstrap p=0.0020. ds_789202 (n=6,805): BIC(k=1)−BIC(k=2)=42,422.4, LRT=42,448.8, dip=0.4231, bootstrap p=0.0020. The dip test rejects unimodality (p<0.05) and model selection prefers k≥2 on all three brains.

### 4. (Priority 0.275 · Surprise 0.300) Merging segments are highly asymmetric — mostly minor offshoots of one primary neuron.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 22 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether merging segments overlap one "primary" neuron extensively while only marginally touching secondary neurons (overlap ratio > 0.8), so merge fixes are pruning rather than 50/50 splitting.
- **Conclusion:** Across 98 merging segments, the mean overlap ratio was 0.9129 and the median 1.0, and a one-sample t-test confirmed the mean exceeds 0.5 (t = 24.87, p = 3.58e-44). Balanced 50/50 fusions are extremely rare. The positive surprisal (+0.300) supports treating merge corrections as pruning a minor accidental offshoot rather than dividing a balanced, ambiguous segment.
- **Caveats:** Based on 98 merge segments from a single dataset cache; the t-test assumes approximate normality of the overlap-ratio distribution, which is skewed toward 1.0.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=98, mean overlap=0.9129, median=1.0000, one-sample t=24.8654, p=3.5768e-44; rerun identical (n=98, mean=0.9129, median=1.0000, t=24.8654, p=3.5768e-44) → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: mean overlap 0.9129, median 1.0, t=24.87, p=3.58e-44, n=98. ds_794491: mean 0.8558, median 1.0, t=17.18, p=1.72e-31, n=98. ds_789202: mean 0.9704, median 1.0, t=43.12, p=8.84e-49, n=64. Same direction (mean overlap ≫ 0.5, median 1.0) and overwhelmingly significant on both extra datasets; effect size is comparable (mean 0.86–0.97).
- **Verdict:** MINOR
- **Test:** One-sample t-test that mean overlap ratio exceeds 0.5, t=24.87, p=3.58e-44, n=98 (mean 0.9129, median 1.0). The data are a proportion bounded on [0,1] and heavily piled at the 1.0 ceiling (median=1.0), which violates the t-test's normality assumption.
- **Statistical issues:** Wrong distributional assumption — one-sample t-test applied to a strongly left-skewed, ceiling-bounded ratio (median 1.0); a one-sample Wilcoxon or a bootstrap/sign test would be the appropriate choice. Comparing the mean against 0.5 is also a weak target given the distribution sits at the 1.0 boundary.
- **Logic issues:** none — the substantive conclusion (segments overlap one primary neuron, so merges are pruning not 50/50 splits) is supported by the median of 1.0 independent of the t-test.
- **Verdict rationale:** The test choice is technically inappropriate for ceiling-bounded proportions, but the effect is so extreme (median 1.0 on all three datasets) that the conclusion is not in doubt; downgraded only for the test-choice fault.
- **Corrected test:** One-sample Wilcoxon signed-rank test (median > 0.5) plus an exact sign test, with a bootstrap 95% CI on the median and mean overlap ratio — replaces the one-sample t-test, which assumed approximate normality of a strongly left-skewed, [0,1]-bounded ratio piled at the 1.0 ceiling (median=1.0).
- **Corrected result:** Wilcoxon W=4656.0, p=1.8663e-19 (n_nonzero=96); exact sign test p=1.2622e-29; effect size median overlap=1.0000, bootstrap 95% CI [1.0000, 1.0000]; mean overlap=0.9129, bootstrap 95% CI [0.8793, 0.9439] — the CI sits far above 0.5. Original one-sample t-test reported t=24.8654, p=3.5768e-44. The non-parametric tests confirm the same conclusion with a CI that excludes the 0.5 null.
- **Post-correction verdict:** UPHELD
- **Corrected generalization:** GENERALIZES — holds on both extra datasets. ds_794491 (n=95): Wilcoxon W=4560.0, p=1.90e-18; sign test p=2.52e-29; median 1.0 CI [0.9943, 1.0000], mean 0.8558 CI [0.8151, 0.8944]. ds_789202 (n=64): Wilcoxon W=2080.0, p=1.93e-14; sign test p=5.42e-20; median 1.0 CI [1.0000, 1.0000], mean 0.9704 CI [0.9472, 0.9895]. Median is 1.0 with CIs well above 0.5 on all three brains.

### 5. (Priority 0.275 · Surprise 0.300) Split errors are spatially clustered along the neuronal cable, not random.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 26 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether split locations violate complete spatial randomness along the 1D neuronal cable, reflecting localized regions of poor image quality or dense crossing fibers.
- **Conclusion:** Across 19 qualifying neurons, the observed mean nearest-neighbor distance between splits (140.30 µm) was far smaller than the Monte Carlo random expectation (320.79 µm), with a paired t-test t = -15.10 and p = 5.76e-12. The positive surprisal (+0.300) confirms strong spatial clustering of splits, indicating localized failure regions.
- **Caveats:** Only 19 neurons qualified, and the Monte Carlo used 100 iterations per neuron (adequate but bounded by the time limit); conclusions generalize to this dataset's neurons.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=19 neurons, paired t=-15.1004, p=5.7640e-12, observed NN=140.30 µm vs random NN=320.79 µm; rerun identical (n=19, t=-15.1004, p=5.7640e-12, 140.30 µm vs 320.79 µm) → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: n=19, observed NN 140.30 µm vs random 320.79 µm, paired t=-15.10, p=5.76e-12. ds_794491: n=9, observed 76.87 µm vs random 147.62 µm, t=-6.33, p=1.13e-04. ds_789202: n=12, observed 219.90 µm vs random 385.96 µm, t=-6.99, p=1.14e-05. Same direction (observed ≈ half of random) and significant clustering on both extra datasets; the t-statistic is smaller (fewer qualifying neurons) but the conclusion holds.
- **Verdict:** SOUND
- **Test:** Paired t-test of per-neuron observed mean nearest-neighbor distance vs Monte Carlo random expectation, t=-15.10, p=5.76e-12, n=19 neurons (140.30 µm vs 320.79 µm). Pairing by neuron is correct (each neuron is its own control against its own permutation null), and the unit of analysis (neuron) is the independent unit, so independence is respected.
- **Statistical issues:** Modest n (19 neurons) and a 100-iteration Monte Carlo null bound by the time limit; both adequate given the ~2.3× effect, but the test inherits the chosen 100-permutation budget.
- **Logic issues:** none — clustering is inferred from a correctly rejected randomness null, not from accepting a null.
- **Verdict rationale:** Appropriate paired design against a permutation null, large effect (observed ≈ half of random), exact reproduction, and the same significant direction on both extra datasets — a well-constructed spatial-clustering test.

### 6. (Priority 0.275 · Surprise 0.300) Omit errors are bursty: continuous missing stretches, not isolated dropped edges.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 27 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether an edge is far more likely to be omitted when an adjacent edge is also omitted than the baseline omit rate, indicating omit errors occur in continuous stretches.
- **Conclusion:** Across 19 neurons, the mean baseline omit probability was 2.23%, but the conditional probability of omission given an adjacent omitted edge jumped to 84.55% (Wilcoxon W = 0.0, p = 3.81e-06). The transition matrix shows an 84.9% chance of staying in the omit state once entered, with a mean run length of 6.91 edges, 56 runs exceeding 50 edges, and a maximum run of 339. The positive surprisal (+0.300) conclusively confirms bursty, continuous omit behavior.
- **Caveats:** Based on 19 neurons; the long-tail run-length statistics are driven by a minority of very long runs, so the mean run length understates the spread.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=19, baseline P(Omit)=0.0223, conditional P(Omit|Omit)=0.8455, Wilcoxon W=0.0, p=3.8147e-06, transition Omit→Omit=0.849, 4,145 runs, max=339, mean run length=6.91; rerun identical on all figures → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: baseline 0.0223 vs conditional 0.8455, W=0.0, p=3.81e-06, n=19, Omit→Omit transition 0.849. ds_794491: baseline 0.0437 vs conditional 0.7854, W=0.0, p=3.91e-03, n=9, Omit→Omit 0.810. ds_789202: baseline 0.0396 vs conditional 0.8665, W=0.0, p=4.88e-04, n=12, Omit→Omit 0.891. Same direction (conditional ≫ baseline, ~80–87%) and significant on both extra datasets; bursty/sticky omit behavior is robust.
- **Verdict:** SOUND
- **Test:** Wilcoxon signed-rank test of per-neuron baseline vs conditional omit probability, W=0.0, p=3.81e-06, n=19 neurons (2.23% baseline vs 84.55% conditional). Paired nonparametric test on per-neuron paired rates is the correct choice; W=0.0 means every neuron moved in the same direction.
- **Statistical issues:** n=19 is modest but the effect (a ~38× jump in conditional probability) is enormous and unanimous across neurons, so power is not a concern.
- **Logic issues:** none, though the "bursty" conclusion is near-tautological for errors that occur in connected runs — the conditional probability is high partly by definition of run structure; the report acknowledges the long-tail run-length framing.
- **Verdict rationale:** Correct paired test, huge unanimous effect, exact reproduction and significant generalization on both extra datasets; the only nuance is the partly-definitional nature of run-based conditional probabilities, which does not undermine the conclusion.

### 7. (Priority 0.275 · Surprise 0.300) Split edges sit significantly closer to branch nodes than correct edges.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 33 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether the median path distance from a split edge to the nearest ground-truth branch node (degree > 2) is significantly shorter than that of correctly reconstructed edges.
- **Conclusion:** Across 7,988 split edges and 934,849 correct edges, the median distance to the nearest branch node was 216.42 µm for splits versus 246.43 µm for correct edges (Mann-Whitney U = 3.47e9, p = 4.64e-27). The positive surprisal (+0.300) confirms the U-Net is more prone to fragmenting neurons near topological branching points.
- **Caveats:** The absolute median difference (~30 µm) is modest relative to typical branch spacing; significance is driven largely by the very large sample, so effect size in practical terms is small.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded splits n=7,988, correct n=934,849, Mann-Whitney U=3,472,848,384.0, p=4.637e-27, split median=216.42 µm vs correct median=246.43 µm; rerun identical (U=3,472,848,384.0, p=4.637e-27, 216.42 vs 246.43 µm) → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: split median 216.42 µm < correct 246.43 µm, U=3.47e9, p=4.64e-27 (splits n=7,988, correct n=934,849). ds_794491: split median 156.94 µm < correct 195.79 µm, U=1.46e9, p=0.0 (splits n=7,847, correct n=420,702). ds_789202: split median 196.43 µm < correct 380.39 µm, U=2.98e9, p=0.0 (splits n=6,805, correct n=1,109,034). Same direction (splits closer to branch nodes) and significant on both extra datasets; the gap is in fact wider on 789202 (~184 µm).
- **Verdict:** WEAK
- **Test:** Mann-Whitney U test of distance-to-branch-node for split (n=7,988) vs correct (n=934,849) edges, U=3.47e9, p=4.64e-27. MWU is appropriate for skewed distance distributions, but the p-value is driven almost entirely by the enormous sample size.
- **Statistical issues:** p≈0 from n≈940k while the median difference is only ~30 µm (216.42 vs 246.43 µm) — a practically negligible effect relative to typical branch spacing; significance is conflated with importance. Edges within a neuron are also non-independent, further inflating significance.
- **Logic issues:** Mild overreach: "more likely to fragment near branch points" is a strong causal/spatial claim from a ~30 µm median shift; the report itself flags the small practical effect.
- **Verdict rationale:** Correct test type but a textbook huge-n / trivial-effect-size case — the conclusion reproduces and generalizes in direction, yet the effect is too small to be practically meaningful, so the strong wording is not warranted.
- **Corrected test:** Effect-size estimation (rank-biserial r / Cliff's delta with a cluster bootstrap 95% CI resampling NEURONS) plus a cluster-level (independent unit = neuron, n=19) sign-flip permutation test — addresses the two faults the verifier named: p≈0 driven by n≈940k, and edges within a neuron treated as independent.
- **Corrected result:** On the origin the effect is negligible and the CI touches/includes the null: rank-biserial r=0.0699 (|r|<0.1 = negligible), Cliff's delta=−0.0699, cluster bootstrap 95% CI [−0.1434, 0.0026] (INCLUDES 0); median difference −30.01 µm. The cluster-level permutation (neuron as unit) gives p=4.4298e-02 — only barely significant, with 73.68% of neurons showing split < correct median. Original MWU reported U=3.47e9, p=4.64e-27 (now seen as an n-inflated p over a near-zero effect).
- **Post-correction verdict:** WEAKENED
- **Corrected generalization:** PARTIAL — effect size stays small and the cluster-bootstrap CI includes 0 on one extra dataset. ds_794491 (n=9 neurons): Cliff's delta=−0.1133, CI [−0.2391, 0.0021] (INCLUDES 0), but cluster permutation p=4.9998e-05 with 100% of neurons split<correct. ds_789202 (n=12): Cliff's delta=−0.2103, CI [−0.3495, −0.0508] (EXCLUDES 0), cluster permutation p=6.9997e-04, 100% of neurons split<correct. Direction is consistent (splits sit closer to branch nodes) and the cluster test is significant on all three, but the effect size is negligible-to-small (delta 0.07–0.21) and its CI includes zero on the origin and on ds_794491 — so the conclusion holds directionally but is practically weak.

### 8. (Priority 0.275 · Surprise 0.300) Merge segments are disproportionately massive — a cheap, sensitive merge prior.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 34 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether merge-causing segments have, on average, at least 3× the total cable length of correctly reconstructed 1-to-1 segments, so segment length alone can flag likely merges.
- **Conclusion:** The 98 merge-causing segments averaged ~19,040 µm of cable (median 4,605 µm) versus ~1,221 µm (median 371 µm) for the 3,129 correct segments — a 15.6× ratio, far exceeding the hypothesized 3×. Using raw cable length as a threshold classifier yielded ROC AUC = 0.869. The positive surprisal (+0.300) confirms cable length is a highly effective, computationally cheap prior for flagging merges before expensive geometric walks.
- **Caveats:** Only 98 merge segments anchor the merge distribution; the classifier AUC, while strong, still leaves meaningful overlap (large correct segments exist as outliers).
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded merge n=98 (avg 19,039.76 µm, median 4,604.78 µm), correct n=3,129 (avg 1,220.58 µm, median 371.42 µm), ratio=15.60; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: merge avg 19,039.76 µm vs correct 1,220.58 µm, ratio 15.60 (n=98 vs 3,129). ds_794491: merge avg 5,850.90 µm vs correct 705.47 µm, ratio 8.29 (n=98 vs 2,765). ds_789202: merge avg 17,844.66 µm vs correct 1,013.21 µm, ratio 17.61 (n=64 vs 4,011). All three datasets exceed the hypothesized ≥3× threshold (8.3×–17.6×), so the "merge segments are massive" conclusion holds throughout, though the ratio drops to ~8× on 794491.
- **Verdict:** SOUND
- **Test:** Descriptive ratio of mean cable length (merge n=98 avg 19,040 µm vs correct n=3,129 avg 1,221 µm = 15.6×) plus a ROC classifier (AUC=0.869). No formal hypothesis test/p-value is reported, but the claim is framed as an effect-size threshold (≥3×) and an AUC, which are the appropriate descriptive metrics for "is length a useful prior."
- **Statistical issues:** No inferential test, but none is strictly needed for a stated effect-size/discrimination claim; the mean is inflated by long-tailed merge outliers, so the median (4,605 vs 371 µm, 12.4×) is the more robust basis and also clears 3×.
- **Logic issues:** none — the conclusion (cable length is a cheap merge prior) is exactly what AUC=0.869 supports, and the caveat about overlap/outliers is stated.
- **Verdict rationale:** An effect-size claim correctly evaluated with a ratio and ROC AUC; reproduces exactly and clears the ≥3× threshold (8.3–17.6×) on all three datasets, so the conclusion is well grounded.

### 9. (Priority 0.275 · Surprise 0.300) The vast majority of splits involve a sub-100 µm micro-fragment.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 35 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether more than 50% of split errors involve a transition into or out of a "micro-fragment" (a predicted segment spanning fewer than 100 µm of total path length).
- **Conclusion:** Of 7,988 split events, 6,931 (86.77%) involved at least one micro-fragment, far exceeding the 50% threshold. The positive surprisal (+0.300) confirms that splits arise predominantly from tiny intermediate fragments rather than the U-Net breaking neurons into large halves, suggesting a proofreader could gain substantial Expected Run Length by absorbing small neighboring segments.
- **Caveats:** The 100 µm micro-fragment cutoff is a chosen threshold; results are specific to that definition and to the dataset's segment-length distribution.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 6,931/7,988 splits (86.77%) involve a sub-100 µm micro-fragment; rerun identical (6,931/7,988, 86.77%) → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: 6,931/7,988 = 86.77%. ds_794491: 6,619/7,847 = 84.35%. ds_789202: 5,102/6,805 = 74.97%. All three comfortably exceed the >50% threshold; the proportion is somewhat lower on 789202 (75%) but the conclusion (micro-fragments dominate splits) holds on both extra datasets.
- **Verdict:** SOUND
- **Test:** Single-proportion observation: 6,931/7,988 (86.77%) of splits involve a sub-100 µm micro-fragment vs a 50% threshold. No formal test reported, but with n=7,988 the proportion is 86.77% — its confidence interval (~±0.7%) is nowhere near 50%, so a formal one-proportion z-test would be redundant.
- **Statistical issues:** Result is conditional on the chosen 100 µm micro-fragment cutoff (acknowledged); not a fault, but the headline number is threshold-dependent.
- **Logic issues:** none — the conclusion (splits are dominated by tiny fragments, so absorbing them yields ERL gains) follows directly from the proportion.
- **Verdict rationale:** A simple, large-n proportion that overwhelmingly clears 50% and reproduces/generalizes (75–87%); the only dependency is the threshold definition, which is transparently stated.

### 10. (Priority 0.275 · Surprise 0.300) Tortuosity is statistically higher near splits but is a weak standalone predictor.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 36 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether local skeleton tortuosity (path length / straight-line distance) in a 10 µm window is significantly higher around split edges than correct edges.
- **Conclusion:** Mean tortuosity was 1.0905 (±0.2090) near splits versus 1.0541 (±0.0905) for correct edges (t = 14.27, p ≈ 0.0), so the hypothesis is statistically supported (positive surprisal +0.300). However, both distributions cluster heavily at 1.0–1.2 with massive overlap, so high tortuosity alone is **not** a strong standalone split predictor — only the extreme-tortuosity tail is enriched among splits.
- **Caveats:** Statistical significance here is driven by sample size, not effect size; the analysis itself flags that the practical separation between split and correct edges is small, so this should not be treated as a firm predictive rule.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded split n=7,988 (mean tortuosity 1.0905±0.2090), correct n=7,988 (1.0541±0.0905), t=14.2729, p=0.0; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: split mean 1.0905 vs correct 1.0541, t=14.27, p≈0.0, n=7,988. ds_794491: split 1.1082 vs correct 1.0635, t=15.71, p≈0.0, n=7,847. ds_789202: split 1.1314 vs correct 1.0730, t=19.37, p≈0.0, n=6,805. Same direction (higher tortuosity near splits) and significant on both extra datasets. As on the origin, the absolute separation remains small (means differ by ~0.04–0.06), so tortuosity stays a statistically robust but weak standalone predictor.
- **Verdict:** MINOR
- **Test:** Two-sample t-test of tortuosity for split vs correct edges (n=7,988 each), t=14.27, p≈0.0 (mean 1.0905 vs 1.0541). Tortuosity is a ratio ≥1 and heavily right-skewed/clustered at 1.0–1.2, so a t-test is not the ideal choice (Mann-Whitney would be cleaner), and the unequal variances (±0.209 vs ±0.091) call for Welch rather than Student.
- **Statistical issues:** Skewed, unequal-variance data under a t-test; more importantly p≈0 is driven by n while the mean difference (~0.036) is trivial. The analysis itself states the practical separation is small and tortuosity is "not a strong standalone predictor."
- **Logic issues:** none — the conclusion is correctly hedged (significant but weak predictor), avoiding the trap of equating significance with predictive value.
- **Verdict rationale:** Suboptimal test and a significance-driven-by-n effect, but the analysis explicitly refuses to overclaim, so the honest "statistically real, practically weak" conclusion stands; downgraded for the test-choice/effect-size mismatch.
- **Corrected test:** Mann-Whitney U (rank-based, no normality assumption) plus an effect size — Cliff's delta / probability-of-superiority AUC with a bootstrap 95% CI — replaces the Welch/Student t-test on a ratio ≥1 that is right-skewed, clustered at 1.0–1.2 and has unequal variances (±0.209 vs ±0.091).
- **Corrected result:** Mann-Whitney U=3.859e7, p≈0.0 (still significant in the same direction); AUC P(split>correct)=0.6048; Cliff's delta=0.2096, bootstrap 95% CI [0.1808, 0.2397] (excludes 0 but small); median difference only 0.0122 (negligible). Original Welch t-test reported t=14.2729, p≈0.0, mean diff ~0.036. The corrected effect size confirms a real but small separation — well short of a usable standalone predictor.
- **Post-correction verdict:** WEAKENED
- **Corrected generalization:** GENERALIZES (directionally) — small effect on all three brains. ds_794491: U=3.882e7, p≈0.0, AUC=0.6305, Cliff's delta=0.2610, CI [0.2325, 0.2881], median diff 0.0176. ds_789202: U=2.870e7, p≈0.0, AUC=0.6198, Cliff's delta=0.2395, CI [0.2116, 0.2658], median diff 0.0202. Direction (higher tortuosity near splits) is significant with a CI excluding zero on both, but Cliff's delta stays small (0.21–0.26) and the median difference is ~0.01–0.02, consistent with the origin's "statistically real, practically weak" reading.

### 11. (Priority 0.275 · Surprise 0.300) Splits are more frequent in terminal (leaf) branches than internal backbones.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 38 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether the split error rate is higher in terminal topological compartments (leaf branches) than internal branches, implying continuity tracking degrades toward neurite endpoints.
- **Conclusion:** Across 19 neurons, the mean terminal split rate (0.65%) exceeded the mean internal rate (0.54%), with a paired t-test t = 3.93 and p = 0.00097, and the per-neuron paired plot confirmed the trend held for most neurons. The positive surprisal (+0.300) supports that U-Net continuity degrades at neurite endpoints.
- **Caveats:** Based on 19 neurons; the absolute rate gap (~0.1 percentage point) is small even though the paired test is significant.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=19, internal split rate=0.0054, terminal=0.0065, diff=0.0011, paired t=3.9347, p=9.7131e-04; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: internal 0.0054 vs terminal 0.0065, diff +0.0011, t=3.93, p=9.71e-04, n=19. ds_794491: internal 0.0115 vs terminal 0.0163, diff +0.0049, t=3.14, p=1.38e-02, n=9. ds_789202: internal 0.0049 vs terminal 0.0064, diff +0.0015, t=3.35, p=6.51e-03, n=12. Same direction (terminal > internal) and significant (p<0.05) on both extra datasets; the effect is actually larger on 794491.
- **Verdict:** WEAK
- **Test:** Paired t-test of per-neuron terminal vs internal split rate, t=3.93, p=9.71e-04, n=19 neurons (0.65% vs 0.54%). Pairing by neuron is correct; the per-neuron rate is a reasonable, approximately continuous quantity for a paired t-test.
- **Statistical issues:** Small n (19 neurons); the absolute rate gap is ~0.1 percentage point — statistically detectable but tiny in practice. Per-neuron rates are bounded proportions, so a sign/Wilcoxon check would be a safer companion at this n.
- **Logic issues:** none — significance from a correctly paired design, and the report flags the small absolute gap.
- **Verdict rationale:** A correct paired test that reproduces exactly and generalizes (p<0.05, same direction on both extra datasets), but the modest n and near-trivial absolute effect keep it at WEAK rather than SOUND.

### 12. (Priority 0.275 · Surprise 0.300) Split errors show a spatial "cascade" — one split makes nearby splits likely.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 40 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether the occurrence of one split makes another split highly probable within a short distance, indicating localized regions of persistently poor segmentation.
- **Conclusion:** Across 7,988 split edges, the observed mean nearest-neighbor distance (132.85 µm) was far below the random-null mean (300.27 µm), and a Kolmogorov-Smirnov test gave KS = 0.5635 with p ≈ 0.0, firmly rejecting spatial randomness. The histogram showed a sharp density peak near zero distance. The positive surprisal (+0.300) confirms a cascade effect of clustered splits in localized poor-segmentation regions.
- **Caveats:** This corroborates IDs 26/45 with a different test; the KS test is sensitive to large samples, so significance is expected, though the large KS statistic (0.56) indicates a genuinely substantial distributional difference.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=7,988, KS=0.5635, p=0.0, observed NN=132.85 µm vs random=300.27 µm; rerun identical (KS=0.5635, p=0.0, 132.85 vs 300.27 µm) → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: KS=0.5635, p=0.0, observed NN 132.85 µm vs random 300.27 µm, n=7,988. ds_794491: KS=0.4376, p=0.0, observed 73.21 µm vs random 131.42 µm, n=7,847. ds_789202: KS=0.4564, p=0.0, observed 209.06 µm vs random 372.74 µm, n=6,805. Same direction (observed ≈ half of random) and p=0.0 on both extra datasets; KS statistic is a bit lower (0.44–0.46) but the clustering/cascade conclusion holds.
- **Verdict:** SOUND
- **Test:** Two-sample Kolmogorov-Smirnov test of observed vs random-null nearest-neighbor distance distributions, KS=0.5635, p≈0.0, n=7,988 split edges. KS is appropriate for comparing two continuous distributions; the large KS statistic (0.56) reflects a genuinely large distributional gap, not just large-n significance.
- **Statistical issues:** Split edges are treated as independent observations although many lie on the same neuron (non-independence), which inflates KS significance; however, the effect size (KS=0.56, observed ≈ half of random) is large enough that this does not threaten the conclusion. This replicates IDs 26/45 on the same data, so it is corroborating rather than independent evidence.
- **Logic issues:** none — the "cascade" inference comes from a correctly rejected randomness null with a large KS statistic; the report notes the large-n caveat but emphasizes the substantial KS value.
- **Verdict rationale:** Right test, large effect size (KS=0.56) rather than mere significance, exact reproduction, and consistent generalization (KS 0.44–0.46, p≈0) on both extra datasets — the clustering conclusion is robust.

### 13. (Priority 0.275 · Surprise 0.300) Splits and omits co-occur locally, forming error cascades on the graph.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 42 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether an edge topologically adjacent to a split edge has a significantly higher omit probability than the neuron's baseline omit rate, indicating locally correlated topological errors.
- **Conclusion:** Across 19 neurons, the mean baseline omit rate (2.23%) more than doubled to 5.28% for edges adjacent to a split (Wilcoxon statistic = 2.0, p = 1.14e-05), with nearly all scatter points above the y = x line. The positive surprisal (+0.300) supports that splits and omits are locally correlated, forming combined error cascades.
- **Caveats:** Based on 19 neurons; the conditional omit rate (5.28%) is still low in absolute terms, so the effect is a relative doubling on a small base.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=19, baseline omit rate=0.0223, conditional (adjacent-to-split)=0.0528, Wilcoxon stat=2.0, p=1.1444e-05; rerun identical → exact match.
- **Generalization:** DOES-NOT-GENERALIZE
- **Across datasets:** origin 794495: baseline 0.0223 vs conditional 0.0528, Wilcoxon stat=2.0, p=1.14e-05, n=19 (conditional ≫ baseline, significant). ds_794491: baseline 0.0437 vs conditional 0.0398, stat=17.0, p=5.70e-01, n=9 — direction FLIPS (conditional < baseline) and not significant. ds_789202: baseline 0.0396 vs conditional 0.0477, stat=22.0, p=2.04e-01, n=12 — same direction but not significant (p≈0.20). The split→omit local co-occurrence effect vanishes on both extra datasets (one even reverses sign); this is the most clearly dataset-specific finding in the run.
- **Verdict:** MAJOR
- **Test:** Wilcoxon signed-rank test of per-neuron baseline vs split-adjacent conditional omit rate, statistic=2.0, p=1.14e-05, n=19 neurons (2.23% vs 5.28%). The paired nonparametric test is the right choice, and the origin result is statistically valid in isolation.
- **Statistical issues:** Small n (19 neurons) and a relative doubling on a very low base (2.23%→5.28%) — fragile. The decisive fault is replication: on ds_794491 the effect REVERSES (conditional 3.98% < baseline 4.37%, p=0.570) and on ds_789202 it is non-significant (p=0.204).
- **Logic issues:** Conclusion overreach — "splits and omits form error cascades" is asserted as a general mechanism from a single dataset, but the effect does not survive on either additional brain.
- **Verdict rationale:** The origin test is technically correct, but the finding DOES-NOT-GENERALIZE (sign flip on one dataset, non-significant on the other), so the cascade conclusion should not be trusted — the most clearly dataset-specific claim in the run.

### 14. (Priority 0.275 · Surprise 0.300) Split clustering reconfirmed via Monte Carlo nearest-neighbor permutation.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 45 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether split edges are closer to one another along the ground-truth topology than expected by random chance, indicating localized morphological-complexity regions where the U-Net consistently fails.
- **Conclusion:** Across 19 neurons, the mean actual nearest-neighbor distance (140.30 µm) was much smaller than the random expectation (320.13 µm) over 100 permutations per neuron, with a paired t-test t = -14.93 and p = 1.40e-11. The positive surprisal (+0.300) again confirms non-uniform, clustered split errors, independently corroborating IDs 26 and 40.
- **Caveats:** Same 19-neuron scope and 100-permutation Monte Carlo budget as related experiments; this is a replication rather than independent new evidence.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=19, actual NN=140.30 µm, expected NN=320.13 µm, paired t=-14.93, p=1.40e-11; rerun actual NN=140.30 µm, expected NN=319.96 µm, t=-15.0355, p=1.2391e-11 → match (the small expected-NN/t/p differences are Monte Carlo permutation seed noise; the conclusion of significant clustering is unchanged).
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: actual NN 140.30 µm vs expected 320.68 µm, paired t=-15.07, p=1.19e-11, n=19. ds_794491: actual 76.87 µm vs expected 147.61 µm, t=-6.31, p=2.31e-04, n=9. ds_789202: actual 219.90 µm vs expected 385.35 µm, t=-7.15, p=1.87e-05, n=12. Same direction (actual ≈ half of expected) and significant on both extra datasets; independently corroborates ID 26/40's clustering result on the additional datasets.
- **Verdict:** SOUND
- **Test:** Paired t-test of per-neuron actual vs Monte Carlo expected nearest-neighbor distance (100 permutations/neuron), t=-14.93, p=1.40e-11, n=19 neurons (140.30 µm vs 320.13 µm). Pairing by neuron against the neuron's own permutation null is the correct independent-unit design.
- **Statistical issues:** This is a replication of ID 26 on the same 19 neurons (acknowledged), so it is corroborating rather than new evidence; the 100-permutation budget and small n are minor, given the ~2.3× effect. Rerun showed only trivial seed noise (expected 319.96 vs 320.13 µm).
- **Logic issues:** none — clustering inferred from a rejected randomness null; the report correctly frames it as a replication, not independent confirmation.
- **Verdict rationale:** Sound paired permutation design with a large effect, reproduces up to Monte Carlo seed noise, and generalizes in direction and significance to both extra datasets; the only caveat is that it duplicates ID 26's evidence.

### 15. (Priority 0.275 · Surprise 0.300) A radius-matching constraint cuts false-positive merges by ~56%, but halves split-fixing power.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 47 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether adding a morphological thickness constraint (leaf `node_radius` difference < 25%) to a < 15 µm proximity split-correction heuristic reduces false-positive merges by at least 40% versus proximity alone.
- **Conclusion:** The baseline proximity heuristic resolved 1,513 true splits but introduced 151 false-positive merge pairs; adding the radius constraint resolved 665 true splits with 67 false-positive merges — a 55.63% reduction in false merges, exceeding the 40% target. The positive surprisal (+0.300) confirms the safety benefit, but it comes at a real cost: the constrained heuristic retains only 43.95% of the baseline's split-resolving power.
- **Caveats:** This is an efficacy/safety trade-off, not a free win — nearly 56% of true splits are no longer fixed under the constraint, so deployment requires balancing precision against recall.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded baseline resolved 1,513 splits / 151 false-merge pairs, constrained 665 splits / 67 false-merge pairs, 55.63% reduction, 43.95% split power retained; rerun identical on every figure → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** primary metric = reduction in direct false-merge pairs (target ≥40%). origin 794495: 151→67 pairs = 55.63% reduction, 43.95% of splits retained. ds_794491: 47→15 pairs = 68.09% reduction, 37.14% retained. ds_789202: 22→7 pairs = 68.18% reduction, 35.87% retained. The ≥40% false-merge-reduction claim holds (in fact exceeds it, ~68%) on both extra datasets, and the same efficacy cost (~36–44% of splits retained) recurs. Note: the secondary "canonical merges" metric is computed on tiny counts and is noisy across datasets (origin 33.33%, 794491 -100.00%, 789202 200.00%), so only the direct-pair metric is a reliable basis — and it generalizes.
- **Verdict:** WEAK
- **Test:** No inferential statistic — a descriptive before/after comparison of false-positive merge pairs (151→67 = 55.63% reduction) against a 40% target, plus split-resolving power retained (43.95%). For an A/B heuristic-efficacy claim a point-estimate comparison is reasonable, but no confidence interval or test of the difference in proportions is reported.
- **Statistical issues:** The primary metric rests on small counts (151 and 67 pairs at origin, falling to 47→15 and 22→7 on the extra datasets), so the percentage reductions have wide implicit uncertainty and no CI is given. The secondary "canonical merges" metric is explicitly noted as noise (-100%, +200%) and correctly set aside.
- **Logic issues:** none — the conclusion is appropriately framed as a precision/recall trade-off (safety gain at ~56% loss of split-fixing power), not a free win.
- **Verdict rationale:** The direct-pair reduction exceeds the 40% target and generalizes (~68% on both extra datasets), but the absence of any uncertainty quantification on small pair counts keeps this at WEAK rather than SOUND.

### 16. (Priority 0.275 · Surprise 0.300) Omit edges are perfectly adjacent — 100% have a zero-distance omit neighbor.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 50 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether the network path distance between adjacent omitted edges is significantly shorter than expected by chance, indicating omissions happen in localized continuous stretches.
- **Conclusion:** Across 57,286 omit edges, 100.0% had a nearest-neighbor distance of 0.00 µm (directly sharing a node with another omit edge), versus a mean of 68.34 µm (median 36.64 µm) for matched random correct edges (Mann-Whitney U = 1.23e8, p = 0.0). The positive surprisal (+0.300) strongly confirms omissions occur in continuous clustered stretches, complementing the run-length finding in ID 27.
- **Caveats:** The `review` field was "N/A" (no independent audit recorded), so this result rests on the analysis alone; the 100% adjacency figure is a near-definitional consequence of omits occurring in connected runs.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=57,286 omit NN pairs, Mann-Whitney U=123,107,614.0, p=0.0, omit NN mean/median=0.00/0.00 (100% zero-distance) vs random mean=68.34 µm, median=36.64 µm; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: omit NN 100% zero-distance (mean 0.00 µm) vs random mean 68.34 µm, U=1.23e8, p=0.0, n=57,286. ds_794491: omit 100% zero-distance vs random mean 33.45 µm, U=2.29e8, p=0.0, n=48,922. ds_789202: omit 100% zero-distance vs random mean 49.06 µm, U=5.26e8, p=0.0, n=89,380. Same direction (omit edges always have a zero-distance omit neighbor) and p=0.0 on both extra datasets.
- **Verdict:** MAJOR
- **Test:** Mann-Whitney U test of nearest-neighbor network distance for omit edges (100% at 0.00 µm) vs random correct edges (mean 68.34 µm), U=1.23e8, p=0.0, n=57,286. The test is run on a degenerate, point-mass distribution (every omit value is exactly 0), which makes the MWU/p-value meaningless rather than informative.
- **Statistical issues:** The 100%-zero result is tautological/definitional: omit errors occur in connected multi-edge runs, so by construction almost every omit edge shares a node with another omit edge and has nearest-neighbor distance 0. The MWU comparison against random correct edges therefore tests a foregone conclusion (a constant-0 group vs a positive group), and p=0.0 carries no real evidential weight. The record itself notes the `review` field was "N/A" (no independent audit) and that 100% adjacency is "a near-definitional consequence of omits occurring in connected runs."
- **Logic issues:** Conclusion ("omissions occur in localized continuous stretches") restates the definition of a run rather than discovering it; the test does not isolate clustering beyond what the run-structure of omits guarantees.
- **Verdict rationale:** Although it reproduces and generalizes, the statistic is applied to a definitionally-zero quantity, so the significant p-value is an artifact of construction, not evidence — the conclusion is essentially circular.
- **Corrected test:** Within-neuron label-permutation test with the neuron as the independent cluster unit (n=19): for each neuron the observed omit nearest-neighbor distance is compared to a null built by permuting which edges are labeled "omit" while holding the omit COUNT fixed, then a cluster sign-flip permutation p-value and a bootstrap CI on the mean per-neuron gap. This replaces the degenerate Mann-Whitney on a constant-0 point-mass group (whose 100%-zero result was definitional) and instead asks whether omits cluster TIGHTER than their own per-neuron count would predict — clustering beyond the run-structure artifact.
- **Corrected result:** Omits genuinely cluster beyond the definitional baseline. Mean per-neuron (observed − permuted-null) omit NN distance = −113.21 µm (negative ⇒ tighter than chance), with 100% of neurons below their own null, cluster sign-flip permutation p=4.9998e-05, bootstrap 95% CI [−141.63, −87.38] µm (EXCLUDES 0), mean per-neuron z vs own null = −41.16. Original degenerate MWU reported U=1.23e8, p=0.0 on a constant-0 group (no real evidential weight). The corrected test removes the circularity and still finds significant clustering.
- **Post-correction verdict:** UPHELD
- **Corrected generalization:** GENERALIZES — holds on both extra datasets under the cluster-permutation test. ds_794491 (n=9): mean gap −107.25 µm, 100% of neurons below null, permutation p=4.0998e-03, CI [−184.64, −45.19] µm (excludes 0), z=−50.03. ds_789202 (n=12): mean gap −110.12 µm, 100% below null, permutation p=7.4996e-04, CI [−189.58, −51.26] µm (excludes 0), z=−63.81. Genuine omit clustering (not just the definitional 100%-zero adjacency) is significant on all three brains.

### 17. (Priority 0.275 · Surprise 0.300) Merge segments are an order of magnitude longer than non-merge segments.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 55 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether segments that merge multiple neurons are significantly larger in total cable length than correctly predicted, non-merging segments, after correcting the segment mapping.
- **Conclusion:** Using the corrected `node_component_id` → `swc_id` mapping, 98 merge segments had a median cable length of 4,604.78 µm versus 160.14 µm for 366,806 non-merge segments (Mann-Whitney U = 3.53e7, p = 0.0). The positive surprisal (+0.300) confirms merge segments are over an order of magnitude larger, reinforcing cable length as a proofreading priority feature (consistent with IDs 34 and 56).
- **Caveats:** Non-merge segments include many tiny fragments that depress their median; the comparison is against a hugely imbalanced group (98 vs. 366,806), and long non-merge outliers exist.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded merge n=98 (median 4,604.78 µm), non-merge n=366,806 (median 160.14 µm), Mann-Whitney U=35,304,348.0, p=0.0; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: merge median 4,604.78 µm vs non-merge 160.14 µm, U=3.53e7, p=0.0 (n=98 vs 366,806). ds_794491: merge median 2,132.54 µm vs non-merge 130.85 µm, U=6.52e7, p=0.0 (n=98 vs 678,797). ds_789202: merge median 4,345.25 µm vs non-merge 148.33 µm, U=3.70e6, p=6.77e-40 (n=64 vs 59,122). Same direction (merge ≫ non-merge by an order of magnitude) and significant on both extra datasets.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U test of cable length, merge (n=98, median 4,604.78 µm) vs non-merge (n=366,806, median 160.14 µm), U=3.53e7, p≈0.0. MWU is the correct rank-based test for these heavily skewed, hugely-imbalanced length distributions, and unlike a huge-n trivial-effect case here the effect is an order-of-magnitude median difference.
- **Statistical issues:** Extreme group imbalance (98 vs 366,806) and a p=0.0 floor, but the effect size (28× median) is so large that significance is not merely an artifact of n; non-merge fragments depress that group's median (acknowledged). Overlaps ID 34/56 conceptually (same "merge segments are large" theme, different mapping).
- **Logic issues:** none — the conclusion (cable length is a useful merge prior) follows from a real, large distributional separation.
- **Verdict rationale:** Appropriate nonparametric test with a genuinely large effect size, reproduces exactly, and generalizes (order-of-magnitude separation, significant) on both extra datasets.

### 18. (Priority 0.275 · Surprise 0.300) Merge segments are larger in both node count and cable length (size as a merge heuristic).
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 56 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether merging segments are significantly larger in total reconstructed cable length and node count than non-merging segments, after excluding fragments under 10 nodes.
- **Conclusion:** Merging segments (N = 98) had a mean node count of 4,010.49 (median 791) and ~20,052 µm mean cable length, versus 218.71 nodes (median 51) and ~1,093 µm for non-merging segments (N = 4,255), with a Mann-Whitney U = 368,540 and p = 1.04e-38. The positive surprisal (+0.300) provides robust evidence that segment size is a strong, reliable merge-flagging feature.
- **Caveats:** Excluding sub-10-node fragments shapes the non-merge baseline; long-tailed non-merge outliers (some very large correct segments) mean size is a strong but imperfect classifier.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded merge n=98 (mean nodes 4,010.49, median 791, ~20,052.45 µm), non-merge n=4,255 (mean nodes 218.71, median 51, ~1,093.53 µm), Mann-Whitney U=368,540.0, p=1.04e-38; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: merge median 791 nodes vs non-merge 51 nodes, U=368,540, p=1.04e-38 (n=98 vs 4,255). ds_794491: merge median 337.5 nodes vs non-merge 35 nodes, U=364,109, p=8.53e-46 (n=98 vs 4,042). ds_789202: merge median 773 nodes vs non-merge 63 nodes, U=311,767, p=1.11e-24 (n=64 vs 5,585). Same direction (merge ≫ non-merge in node count and cable length) and highly significant on both extra datasets.
- **Verdict:** SOUND
- **Test:** Mann-Whitney U test of node count/cable length, merge (n=98, median 791 nodes) vs non-merge (n=4,255, median 51 nodes, sub-10-node fragments excluded), U=368,540, p=1.04e-38. MWU is the right rank-based test for skewed size distributions; excluding tiny fragments is a defensible (and disclosed) baseline choice.
- **Statistical issues:** The sub-10-node exclusion shapes the non-merge baseline (acknowledged), and long-tailed non-merge outliers mean size is a strong-but-imperfect classifier; the ~15× median separation is a real effect, not just an n-driven p.
- **Logic issues:** none — conclusion (segment size flags merges) matches the large measured separation.
- **Verdict rationale:** Correct test, large effect size, exact reproduction, and highly significant same-direction generalization on both extra datasets; the disclosed fragment-exclusion choice is the only caveat.

### 19. (Priority 0.275 · Surprise 0.300) ~96% of omit gaps fall below the 100 µm min-cable-length filter, the dominant omit driver.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 69 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether a disproportionate majority of omit errors occur in continuous gaps shorter than the 100 µm `min_cable_length` threshold, implicating the tool's short-fragment filter rather than large-scale segmentation failure.
- **Conclusion:** Of 4,145 continuous omitted stretches (totaling 122,590 µm), 3,971 (95.80%) were shorter than 100 µm, and these short gaps accounted for 85,498 µm (69.74%) of total omitted cable. The positive surprisal (+0.300) confirms the cache's minimum-cable-length filter — not large-scale U-Net failure — is the primary driver of omit errors, a finding with direct implications for pipeline configuration.
- **Caveats:** This attributes omits to a configurable preprocessing filter rather than model behavior, so it characterizes the dataset's construction more than the segmentation model; the 100 µm threshold is the very parameter under test.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 4,145 omitted stretches (122,590.27 µm total), 3,971 (95.80%) below 100 µm accounting for 85,498.41 µm (69.74%); rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 794495: 3,971/4,145 = 95.80% of stretches < 100 µm (69.74% of cable). ds_794491: 4,509/4,572 = 98.62% (91.38% of cable). ds_789202: 4,224/4,605 = 91.73% (55.94% of cable). The "vast majority of omit gaps are below the 100 µm filter threshold" conclusion holds on both extra datasets (92–99% of stretches); the cable-length fraction varies more (56–91%) but the dominant-short-gap finding is robust.
- **Verdict:** MINOR
- **Test:** Descriptive proportions only: 3,971/4,145 (95.80%) of omit stretches are <100 µm, accounting for 85,498 µm (69.74%) of omitted cable. No inferential test, which is appropriate for a "what fraction falls below threshold" descriptive claim with n=4,145.
- **Statistical issues:** The 100 µm cutoff is the dataset's own `min_cable_length` build parameter, so finding most omit gaps below it is partly circular — the comparison is against the very threshold under test (acknowledged). This characterizes dataset construction, not model behavior.
- **Logic issues:** Causal-attribution overreach: "the short-fragment filter is the primary driver of omit errors rather than U-Net failure" is a strong mechanistic claim, yet the experiment only measures gap-length proportions, not whether those gaps would otherwise have been reconstructed. The conclusion conflates "gaps are short" with "the filter caused them."
- **Verdict rationale:** The descriptive proportion is solid and generalizes (92–99%), but attributing omit errors causally to the filter overreaches the descriptive measurement and leans on the very threshold being tested; downgraded for that interpretive circularity.

### 20. (Priority 0.275 · Surprise 0.300) Split rate rises monotonically as edges approach leaf tips.
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 70 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested:** Whether the split rate of reconstructed edges inversely correlates with path distance to the nearest leaf tip, so fragmentation worsens toward the extreme ends of the neuron.
- **Conclusion:** Split rate decreased monotonically with distance to the leaf tip: 0.92% within 50 µm, 0.65% at 50–200 µm, and 0.53% beyond 200 µm, with the steepest decline in the first 150 µm. A logistic regression confirmed a significant inverse relationship (coef = -0.1361, p < 0.001). The positive surprisal (+0.300) supports that automated reconstruction struggles most near thin, terminating branches, consistent with the terminal-branch finding in ID 38.
- **Caveats:** The intended Mixed-Effects GEE model failed (a `statsmodels` syntax conflict) and the analysis fell back to standard logistic regression, which ignores per-neuron random effects — so neuron-level non-independence is not fully accounted for; absolute split rates remain under 1%.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded split rates 0.92% (<50 µm) / 0.65% (50–200 µm) / 0.53% (>200 µm), logistic coef=-0.1361, p<0.001; rerun identical (same rates, coef=-0.1361, z=-9.924, p=0.000 over 1,363,789 edges). The same GEE-fallback-to-logit path occurred in both runs → exact match.
- **Generalization:** PARTIAL
- **Across datasets:** origin 794495: split rate 0.92%→0.65%→0.53% by distance bin, logistic coef=-0.1361, z=-9.92, p<0.001, n=1,363,789. ds_789202: 0.99%→0.76%→0.40%, coef=-0.3167, z=-18.64, p≈0.0, n=1,409,045 — same (stronger) direction and highly significant (holds). ds_794491: 2.04%→1.41%→1.28% (descriptive bins still decline) but logistic coef=-0.0095, z=-0.83, p=0.409, LLR p=0.41, n=562,675 — the regression slope is essentially zero and NOT significant (lost). Holds on 789202, lost on 794491; the binned trend is directionally present everywhere but the formal slope is dataset-dependent.
- **Verdict:** MAJOR
- **Test:** Logistic regression of split vs standardized distance-to-leaf-tip, coef=-0.1361, z=-9.92, p<0.001, n=1,363,789 edges (after the intended Mixed-Effects GEE failed with "GEE.from_formula() got multiple values for argument 'groups'" and fell back to a pooled logit). Pseudo R²=0.0011.
- **Statistical issues:** Pseudo-replication / violated independence — the GEE that would model per-neuron random effects failed, so the fallback pools 1.36M within-neuron-correlated edges as if independent, drastically overstating the effective n and the certainty of the slope (nonrobust covariance). Pseudo R²≈0.001 indicates the predictor explains essentially none of the variance even where significant.
- **Logic issues:** Conclusion ("fragmentation worsens monotonically toward leaf tips") overreaches a near-zero-R² slope, and the monotonic-decline framing rests on three coarse descriptive bins rather than the fitted model.
- **Verdict rationale:** The independence structure is wrong (clustered edges analyzed as independent after the GEE fallback) and the finding is PARTIAL — the formal slope is non-significant on ds_794491 (coef=-0.0095, p=0.409) — so the headline monotonic-gradient claim is not reliable.
- **Corrected test:** A working GEE logistic regression with the neuron as the cluster and robust (cluster-aware) standard errors — fixing the pseudo-replication that the original pooled logit introduced when the intended GEE crashed and 1.36M within-neuron-correlated edges were treated as independent. Also a fully cluster-level test (independent unit = neuron): per-neuron Spearman rho of split-rate vs distance-bin, with a bootstrap CI and a Wilcoxon signed-rank that the rhos are negative across neurons.
- **Corrected result:** On the origin the gradient survives the correct clustered analysis but with far less certainty than the pooled logit implied. GEE coef=−0.1593, robust SE=0.0556, z=−2.867, p=4.1415e-03, robust 95% CI [−0.2682, −0.0504] (excludes 0; odds-ratio per SD=0.853), 19 independent clusters. Cluster-level: per-neuron median Spearman rho=−0.7714, bootstrap 95% CI [−0.8286, −0.6000], 94.74% of neurons have a negative slope, Wilcoxon W=1.0, p=7.5422e-05. Original pooled-logit fallback reported coef=−0.1361, z=−9.92, p<0.001 (z inflated ~3.5× by pseudo-replication: corrected z=−2.87).
- **Post-correction verdict:** WEAKENED
- **Corrected generalization:** PARTIAL — the cluster-aware GEE loses significance on one extra dataset even though the per-neuron Spearman test holds everywhere. ds_794491 (9 clusters): GEE coef=−0.0690, robust SE=0.0507, z=−1.360, p=1.7369e-01, CI [−0.1684, 0.0304] (INCLUDES 0, NOT significant); but cluster Spearman median rho=−0.9429, 100% negative slopes, Wilcoxon p=1.9531e-03. ds_789202 (12 clusters): GEE coef=−0.2722, robust SE=0.0448, z=−6.079, p=1.2095e-09, CI [−0.3600, −0.1844] (excludes 0, significant); cluster Spearman median rho=−0.8857, 100% negative, Wilcoxon p=2.4414e-04. So under the proper clustered GEE the slope holds on origin and 789202 but is non-significant on 794491; the descriptive per-neuron decline is consistent on all three.

---

## Statistical Verification — Summary

**Scope:** All 20 ranked hypotheses were audited from their recorded `code`/`codeOutput`/`analysis` plus the already-folded reproduction and generalization numbers. No experiment was re-executed.

**Verdict breakdown (n=20):**

- **SOUND — 9:** entries 2 (ID 3), 5 (ID 26), 6 (ID 27), 8 (ID 34), 9 (ID 35), 12 (ID 40), 14 (ID 45), 17 (ID 55), 18 (ID 56).
- **WEAK — 4:** entries 1 (ID 15), 3 (ID 12), 11 (ID 38), 15 (ID 47).
- **MINOR — 4:** entries 4 (ID 22), 7 (ID 33), 10 (ID 36), 19 (ID 69).
- **MAJOR — 3:** entries 13 (ID 42), 16 (ID 50), 20 (ID 70).
- **CRITICAL — 0.**

**MAJOR/CRITICAL findings (do not trust as stated):**

- **Entry 16 / ID 50 — "Omit edges are 100% zero-distance adjacent" (MAJOR).** Core fault: the Mann-Whitney test is run on a degenerate point-mass group (every omit nearest-neighbor distance is exactly 0 µm) compared against positive random distances — the 100%-zero result is a definitional consequence of omits occurring in connected runs, so U=1.23e8, p=0.0 is a construction artifact, not evidence. The conclusion ("omissions occur in continuous clustered stretches") is circular.
- **Entry 20 / ID 70 — "Split rate rises monotonically toward leaf tips" (MAJOR).** Core fault: pseudo-replication / violated independence — the intended Mixed-Effects GEE failed ("GEE.from_formula() got multiple values for argument 'groups'") and fell back to a pooled logistic regression that treats 1,363,789 within-neuron-correlated edges as independent (nonrobust covariance), overstating certainty; pseudo R²≈0.001. The finding is also PARTIAL — non-significant on ds_794491 (coef=-0.0095, p=0.409).
- **Entry 13 / ID 42 — "Splits and omits co-occur as local error cascades" (MAJOR).** Core fault: although the origin Wilcoxon test (stat=2.0, p=1.14e-05, n=19) is valid, the finding DOES-NOT-GENERALIZE — the effect reverses sign on ds_794491 (conditional 3.98% < baseline 4.37%, p=0.570) and is non-significant on ds_789202 (p=0.204). A relative doubling on a ~2% base over only 19 neurons that vanishes/reverses on two other brains cannot support a general "error cascade" mechanism.

**Benjamini–Hochberg FDR (α=0.05).** Of the 20 entries, 15 report a formal inferential p-value; the other 5 are descriptive (ID 12 GMM fit, ID 34 ratio/ROC, ID 35 proportion, ID 47 before/after counts, ID 69 proportion) and are excluded from the FDR family. Several MWU/KS/t-tests report p=0.0 (below float underflow); these are treated as the smallest ranks. Ranked smallest→largest with BH threshold k/m·α (m=15):

| rank k | ID | test | p | BH thr = k/15·0.05 | survives? |
|---|---|---|---|---|---|
| 1 | 36 | tortuosity t-test | ≈0 (p=0.0) | 0.0033 | yes |
| 2 | 40 | split-cascade KS | ≈0 (p=0.0) | 0.0067 | yes |
| 3 | 50 | omit-adjacency MWU | ≈0 (p=0.0) | 0.0100 | yes |
| 4 | 55 | merge cable-length MWU | ≈0 (p=0.0) | 0.0133 | yes |
| 5 | 22 | merge overlap t-test | 3.58e-44 | 0.0167 | yes |
| 6 | 56 | merge node-count MWU | 1.04e-38 | 0.0200 | yes |
| 7 | 3 | split-near-branch χ² | 4.90e-34 | 0.0233 | yes |
| 8 | 33 | split dist-to-branch MWU | 4.64e-27 | 0.0267 | yes |
| 9 | 70 | leaf-tip split logit | 2.93e-26 | 0.0300 | yes |
| 10 | 26 | split clustering paired t | 5.76e-12 | 0.0333 | yes |
| 11 | 45 | split clustering MC paired t | 1.40e-11 | 0.0367 | yes |
| 12 | 27 | omit bursty Wilcoxon | 3.81e-06 | 0.0400 | yes |
| 13 | 42 | split→omit cascade Wilcoxon | 1.14e-05 | 0.0433 | yes |
| 14 | 38 | terminal split paired t | 9.71e-04 | 0.0467 | yes |
| 15 | 15 | merge crossing angle Wilcoxon | 8.06e-03 | 0.0500 | yes |

All 15 reported p-values survive Benjamini–Hochberg control at FDR 0.05; the largest p-value (ID 15, p=8.06e-03) sits exactly at its BH threshold (0.0500) and passes. **No reported finding becomes non-significant under multiple-comparisons correction** — i.e. the headline borderline case (ID 15, 0.001 < p < 0.05) is the only one near the boundary, and it still passes BH. The qualifications on the run therefore come not from FDR (every p clears it) but from (a) the three MAJOR faults above — a definitional MWU (ID 50), a clustered-data pseudo-replication (ID 70), and a non-generalizing cascade (ID 42); (b) huge-n / trivial-effect-size significance on IDs 33 and 36 (medians differing by ~30 µm and ~0.04 respectively); and (c) test-choice mismatches on ceiling-bounded proportions (ID 22 t-test) and unfit-but-asserted bimodality (ID 12 GMM). These are the discoveries a scientist should treat with the most caution despite uniformly tiny p-values.

---

## Excluded (no surprisal score)

The helper dropped 4 hypotheses that lacked a usable surprisal score (`n_dropped_missing_surprisal = 4`), so they could not be ranked or included above:

- run-5--ground-truth-error-annotations-revised-version_2026-06-25 · ID 7
- run-5--ground-truth-error-annotations-revised-version_2026-06-25 · ID 32
- run-5--ground-truth-error-annotations-revised-version_2026-06-25 · ID 74
- run-5--ground-truth-error-annotations-revised-version_2026-06-25 · ID 92

---

## Statistical Test Corrections — Summary

**Scope:** 6 hypotheses were flagged for a TEST-level fix and re-measured with a corrected statistic (the driver ran the corrected scripts; this section only folds the numbers). All 6 corrected scripts ran cleanly (exitcode 0, no timeouts, no environment failures) on the origin and on both extra datasets (`cache/dataset_cache_794491_mcl100_add.pkl`, `cache/dataset_cache_789202_mcl100_add.pkl`).

**Post-correction breakdown (n=6):**

- **UPHELD — 3:** entry 3 (ID 12), entry 4 (ID 22), entry 16 (ID 50).
- **WEAKENED — 3:** entry 7 (ID 33), entry 10 (ID 36), entry 20 (ID 70).
- **OVERTURNED — 0.**

**Per-id corrections (original test → corrected test → post-correction verdict, with corrected-test generalization):**

| Entry / ID | Original test | Corrected test | Verdict | Corrected generalization |
|---|---|---|---|---|
| 3 / ID 12 | Imposed 2-component GMM (no test) | BIC/AIC + likelihood-ratio model selection + Hartigan dip test (bootstrap unimodal null) | UPHELD | GENERALIZES — dip p=0.0020 and k≥2 preferred on all three brains |
| 4 / ID 22 | One-sample t-test on ceiling-bounded proportion | One-sample Wilcoxon signed-rank + exact sign test + bootstrap CI | UPHELD | GENERALIZES — median 1.0, CI well above 0.5 on all three |
| 7 / ID 33 | Mann-Whitney U (huge-n, edges non-independent) | Cliff's delta / rank-biserial + cluster bootstrap CI + neuron-cluster permutation | WEAKENED | PARTIAL — effect negligible-to-small (delta 0.07–0.21); bootstrap CI includes 0 on origin and ds_794491 |
| 10 / ID 36 | Welch/Student t-test on skewed ratio | Mann-Whitney U + Cliff's delta / AUC + bootstrap CI | WEAKENED | GENERALIZES directionally — delta 0.21–0.26 (small), median diff ~0.01–0.02 on all three |
| 16 / ID 50 | Mann-Whitney U on a degenerate constant-0 group | Within-neuron label-permutation, cluster=neuron + bootstrap CI | UPHELD | GENERALIZES — mean gap ≈ −110 µm, CI excludes 0, p<0.01 on all three |
| 20 / ID 70 | Pooled logistic regression (1.36M edges as independent) | Cluster-aware GEE (robust SEs) + per-neuron Spearman / Wilcoxon | WEAKENED | PARTIAL — GEE significant on origin & ds_789202 (p=4.1e-03, 1.2e-09) but NOT on ds_794491 (p=0.174); per-neuron Spearman holds everywhere |

**Findings that CHANGED under the correct test (all WEAKENED; none overturned):**

- **Entry 7 / ID 33 — splits sit closer to branch nodes (WEAKENED).** Right test type (MWU) but the original p=4.64e-27 was driven by n≈940k over a ~30 µm median gap. The correct read uses an effect size with a neuron-clustered CI: Cliff's delta=−0.0699, cluster bootstrap 95% CI [−0.1434, 0.0026] — negligible and the CI includes zero; the neuron-cluster permutation is only barely significant (p=0.0443). The direction is real but the effect is practically negligible.
- **Entry 10 / ID 36 — higher tortuosity near splits (WEAKENED).** The t-test on a skewed, unequal-variance ratio was replaced with Mann-Whitney + Cliff's delta: U p≈0 but Cliff's delta=0.2096 (CI [0.1808, 0.2397]) and a median difference of only 0.0122 — a real but small separation, confirming tortuosity is not a usable standalone split predictor.
- **Entry 20 / ID 70 — split rate rises toward leaf tips (WEAKENED).** The pooled logit treated 1.36M within-neuron-correlated edges as independent; a proper cluster-aware GEE collapses the z from −9.92 to −2.87 (p=4.1e-03 on origin) and the slope becomes non-significant on ds_794491 (GEE p=0.174, CI includes 0). The per-neuron Spearman trend (median rho ≈ −0.77 to −0.94) is consistent across all three brains, so the gradient is directionally robust but the formal slope is dataset-dependent and far weaker than originally claimed.

**Synthesis.** Applying the correct test did not overturn any of the six findings, but it sharply rebalanced confidence in three of them. Two conclusions that the original analyses asserted descriptively or circularly were actually STRENGTHENED by proper inference: split-gap bimodality (ID 12) survives a genuine model-selection + dip test rather than an imposed k=2 GMM, and omit clustering (ID 50) survives a within-neuron permutation that strips out the definitional 100%-zero artifact — both UPHELD on all three brains. The merge-asymmetry finding (ID 22) is robust to swapping the inappropriate t-test for a Wilcoxon/sign test (median 1.0, CI far above 0.5 everywhere). The three WEAKENED findings share one root cause the verifier flagged: significance conflated with importance or with effective sample size. Once huge within-neuron-correlated samples are handled with effect sizes and cluster-aware tests (ID 33, ID 36) or with a real GEE (ID 70), the p-values shrink dramatically and the effects are revealed as small (ID 33, ID 36) or dataset-dependent (ID 70). The practical takeaway is that the run's "splits/omits cluster" and "merge segments are large/asymmetric" themes are robust to test choice, whereas the fine-grained spatial-gradient claims (distance-to-branch, tortuosity, distance-to-leaf-tip) are statistically real but practically weak and should not be used as standalone proofreading rules.
