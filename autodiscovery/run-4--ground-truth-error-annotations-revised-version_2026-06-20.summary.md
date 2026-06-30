# AutoDiscovery Conclusion Summary — Run 4

**Source file:** `autodiscovery/run-4--ground-truth-error-annotations-revised-version_2026-06-20.json` (150 hypotheses)
**Ranking key (`rank_by`):** `posterior-surprise` (priority = posterior × |surprisal|)
**Ranked:** 148 hypotheses (`n_ranked`); 2 dropped for missing surprisal.
**Shown here:** top 20 of the 148 ranked (`n_returned` = 20, `--top 20`).
**Surprise-magnitude range across the run:** 0.0000 to 0.6899 (max priority score 0.5066).

## Synthesis

The two strongest belief flips both concern the imaging axis and a proofreading heuristic. The single highest-priority result (#1, ID 30) is a positive flip from *Leaning False* to *Leaning True*: a strict Euclidean distance threshold (~6.84 µm) cleanly separates true split gaps from inter-neuron gaps (ROC-AUC 0.9979, F1 0.9945), validating proximity-only auto-reconnection that the loop had initially doubted. Counterbalancing this, the two strongest *negative* surprises (#2 ID 27 and #3 ID 21) both overturned a confidently-held assumption: that anisotropic Z-axis imaging resolution drives split/omit errors. Two independent tests — a mixed-effects logistic regression (p = 0.058, OR 0.80) and a chi-square on edge orientation (p = 0.060) — failed to reach significance, knocking belief down from *Likely True* to *Uncertain*. The cross-cutting theme is that **segmentation errors are governed by local topology and morphology (branch points, tortuosity, branch order, terminal/distal cable, Z-depth extremes) far more than by global imaging-axis orientation**, and that the resulting error patterns (tight split gaps, angular inertia, spatial clustering) are regular enough to power automated proofreading agents.

## Ranked Conclusions

### 1. (Priority 0.507 · Surprise 0.690) A ~6.84 µm Euclidean threshold safely auto-reconnects the vast majority of split fragments.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 30 · **Belief:** Leaning False → Leaning True (0.2917→0.7344) · **Direction:** Positive
- **Tested:** Whether spatial proximity alone can distinguish true split gaps (endpoints of fragments belonging to the same GT neuron) from inter-neuron gaps, so a tool could auto-reconnect splits without causing merges.
- **Conclusion:** Across 6,805 true split gaps and 4,189 inter-neuron gaps within 20 µm, the two distributions were almost entirely non-overlapping — true splits peak ~4.5 µm (2–7 µm band), inter-neuron gaps rarely below 7 µm. Gap distance as a binary classifier gave ROC-AUC 0.9979, and the F1-optimal threshold of 6.84 µm reached F1 = 0.9945. The large positive surprisal (+0.690) reflects that the loop had leaned against proximity-only repair, but the data strongly validate it as a safe, effective heuristic.
- **Caveats:** None noted; review confirms faithful implementation. Results are intra-dataset, so generalization to other volumes is untested.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n_true=6805, n_fp=4189, ROC-AUC=0.9979, threshold=6.84 µm, F1=0.9945; rerun n_true=6805, n_fp=4189, ROC-AUC=0.9979, threshold=6.84 µm, F1=0.9945 → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** origin (789202): ROC-AUC=0.9979, thr=6.84 µm, F1=0.9945 (n_true=6805/n_fp=4189); ds_794491: ROC-AUC=0.9889, thr=6.48 µm, F1=0.9711 (n_true=7847/n_fp=13208); ds_794495: ROC-AUC=0.9953, thr=6.83 µm, F1=0.9878 (n_true=7988/n_fp=8867). Near-perfect proximity separation and a ~6.5 µm threshold hold on both extra datasets (both exit 0).
- **Verdict:** SOUND
- **Test:** ROC-AUC of gap distance as a binary classifier (`sklearn roc_curve/auc`) plus F1-optimal threshold; n_true=6805 true split gaps, n_fp=4189 inter-neuron gaps within 20 µm; "ROC AUC: 0.9979 / Optimal Distance Threshold: 6.84 um / Max F1 Score: 0.9945".
- **Statistical issues:** None material. This is a descriptive separability/classifier metric, not a hypothesis test, so independence/normality assumptions do not apply. The F1-optimal threshold is selected on the same data it is reported on (mild in-sample optimism), but the AUC=0.9979 and near-disjoint distributions leave no room for that to matter; threshold reproduces (~6.5 µm) on both extra volumes.
- **Logic issues:** None. The conclusion (proximity-only auto-reconnect is safe) follows directly from the near-perfect separation, and the surprisal sign (+0.690, Leaning False → Leaning True) matches the evidence.
- **Verdict rationale:** Huge effect, exact reproduction, and consistent generalization (AUC 0.989–0.998 across three volumes). The only caveat — false-positive "inter-neuron gaps" are derived from GT labels in simulation rather than live merges — is a deployment caveat, not a flaw in the recorded test.

### 2. (Priority 0.323 · Surprise 0.795) Z-axis neurite alignment does NOT significantly raise topological error risk.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 27 · **Belief:** Likely True → Uncertain (0.9167→0.4062) · **Direction:** Negative
- **Tested:** Whether neurites aligned along the low-resolution z (imaging) axis suffer more splits/omissions than xy-plane-aligned segments.
- **Conclusion:** Across 1,160,529 edges (4.44% with split/omit errors), a Bayesian mixed-effects logistic regression on a 20,000-edge subset found z-alignment to be non-significant and slightly *negative* (coef = −0.066, p = 0.058; odds ratio 0.80). The strong negative surprisal (−0.795, the run's largest magnitude) marks a confidently-held assumption the data contradicted: orientation relative to the anisotropic axis is not a primary driver of segmentation failure.
- **Caveats:** Regression was fit on a 20,000-edge subsample (not the full 1.16M edges); p = 0.058 is borderline, so the effect is "not significant" rather than firmly null. Review found no implementation issues.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded edges=1160529, errors=4.44%, z_align coef=−0.0661 (p=0.05823), OR=0.8029; rerun edges=1160529, errors=4.44%, z_align coef=−0.0661 (p=0.05823), OR=0.8029 → exact match (same 20k subsample seed).
- **Generalization:** PARTIAL
- **Across datasets:** the *null/non-significant* claim does not cleanly hold. origin (789202): coef=−0.0661, p=0.05823, OR=0.8029 → "does NOT significantly affect" (borderline). ds_794491: coef=+0.0376, p=0.1747, OR=1.1371 → non-significant but sign *flipped positive*. ds_794495: coef=−0.1697, p=6.75e-06, OR=0.5632 → significant DECREASE (script itself prints "significantly DECREASES the risk"). So "no significant z-alignment effect" holds on 794491 but is contradicted on 794495 (significant), and the direction is unstable across the three (−/+/−). The original anti-anisotropy conclusion (z-alignment is NOT a risk *increaser*) is never reversed into a risk-increase, but the specific "non-significant" finding is dataset-specific.
- **Verdict:** MAJOR
- **Test:** `BinomialBayesMixedGLM.fit_vb()` (variational-Bayes mixed logistic) on a 20,000-edge subsample; z_align_std Post.Mean=−0.0661, Post.SD=0.0349; the p-value (0.05823) is then manufactured by treating the posterior mean/SD as a frequentist Wald z (`z=coef/sd; p=2*(1-Φ(|z|))`). OR=0.8029.
- **Statistical issues:** (1) Test-construction fault — a variational-Bayes posterior is not a sampling distribution, so converting Post.Mean/Post.SD into a two-sided Wald p-value is not a valid significance test; the reported p=0.058 has no defensible frequentist interpretation. (2) The model is fit on a 20k subsample of 1.16M edges, discarding ~98% of the data and the chosen-seed result is not stable across volumes. (3) Edges sharing nodes within a neuron are not independent; the per-edge likelihood treats them as such (only a neuron random-effect variance is modeled, not the within-neuron edge correlation in the orientation covariate).
- **Logic issues:** "Failed to reach p<0.05" is presented as evidence that "z-axis orientation is not a primary driver" — the absence-of-evidence/evidence-of-absence fallacy, made worse because the p itself is invalid. The strong negative surprisal (−0.795, Likely True → Uncertain) is driven by a borderline, non-reproducing, improperly-computed p.
- **Verdict rationale:** The headline number is a mis-constructed p-value used to license a null conclusion that does not hold up: PARTIAL generalization with the coefficient sign flipping −/+/− and one volume significant (p=6.75e-06). The downstream claim should be downgraded from a confident "no anisotropy effect" to "no consistent, validly-tested effect."
- **Corrected test:** Frequentist logistic regression on the FULL 1.16M edges (not a 20k subsample) with a neuron-level cluster-permutation test (neuron = independent unit), replacing the invalid VB-posterior-as-Wald-z p. The original treated a variational-Bayes posterior mean/SD as a sampling distribution, which is not a valid significance test.
- **Corrected result:** Neuron-cluster permutation Spearman rho(neuron z vs err rate) = −0.1818, two-sided p = 0.5721 (NOT significant); full-data coef = −0.1203, naive Wald p = 2.4e-136 but cluster-robust Wald p = nan (single-cluster degeneracy), OR = 0.6707. Versus the original invalid VB-Wald p = 0.05823, OR = 0.8029. The corrected proper test confirms NO significant z-alignment effect (the original null claim survives, now on valid grounds).
- **Post-correction verdict:** UPHELD — the original conclusion was a *null* ("z-alignment is not a significant driver"), and the valid cluster-permutation test agrees (p = 0.5721, far from significant). The headline null stands, but only as "no detectable effect," not as proof of absence.
- **Corrected generalization:** PARTIAL — 794491: permutation p = 0.5545 (not significant, consistent null); 794495: permutation p = 0.9962 (not significant) BUT the full-data cluster-robust Wald is significant (p = 2.05e-11, OR = 0.514, a *decrease*). So under the permutation test the null holds on all three, while the cluster-robust GLM flags 794495 — the no-effect conclusion is broadly supported under the correct unit-of-analysis but not perfectly uniform.

### 3. (Priority 0.266 · Surprise 0.568) Z-dominant edges show no statistically significant excess of split/omit errors.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 21 · **Belief:** Likely True → Uncertain (0.8333→0.4688) · **Direction:** Negative
- **Tested:** Whether hardware-driven imaging anisotropy makes Z-dominant edges fail more often than XY-dominant edges.
- **Conclusion:** Among 433,243 Z-dominant and 975,802 XY-dominant edges, error rates were nearly identical (3.70% vs 3.63%); a chi-square test gave χ² = 3.53, p = 0.0602, failing to reject the null. The negative surprisal (−0.568) corroborates hypothesis #2: directional imaging bias is not a substantial error driver, contradicting the prior expectation.
- **Caveats:** p = 0.060 is just above threshold — a near-miss rather than a clean null; the tiny absolute gap (0.07%) is practically negligible regardless of significance.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded Z-dom=433243 (3.70%), XY-dom=975802 (3.63%), χ²=3.5328, p=6.0165e-02; rerun identical χ²=3.5328, p=6.0165e-02 → exact match.
- **Generalization:** DOES-NOT-GENERALIZE
- **Across datasets:** the "no significant difference" (null) conclusion fails on both extras, which become strongly significant in *opposite* directions. origin (789202): Z-dom 3.70% vs XY-dom 3.63%, χ²=3.53, p=6.02e-02 (null, Z barely higher). ds_794491: Z-dom 6.41% vs XY-dom 5.50%, χ²=166.1, p=5.21e-38 (significant, Z *higher*). ds_794495: Z-dom 2.24% vs XY-dom 2.86%, χ²=395.8, p=4.62e-88 (significant, Z *lower*). On the extra datasets there IS a significant orientation effect, and its sign is not consistent — so the origin's clean near-null does not hold.
- **Verdict:** MAJOR
- **Test:** Pearson chi-square test of independence (`scipy.stats.chi2_contingency`) on a 2×2 table of Z-dominant vs XY-dominant edges × error/no-error; 433,243 Z-dominant and 975,802 XY-dominant edges; χ²=3.5328, p=0.0602.
- **Statistical issues:** Independence assumption violated — the rows are 1.4M individual edges, but adjacent edges share nodes and run along the same neurite, so orientation and error status are spatially autocorrelated; the chi-square treats them as independent draws, which can both inflate and (here) mis-state the test. The 0.07-point absolute rate gap (3.70% vs 3.63%) is practically negligible regardless of significance.
- **Logic issues:** "p>0.05, fail to reject null" is escalated to "This refutes the hypothesis that imaging anisotropy creates a substantial directional bias" — affirming the null from a non-significant result, the classic absence-of-evidence error. The conclusion also over-generalizes a single-volume near-miss into a claim about the imaging pipeline.
- **Verdict rationale:** DOES-NOT-GENERALIZE: on both extra volumes the same test is strongly significant (p=5.2e-38 and p=4.6e-88) in *opposite* directions, directly contradicting "no directional bias." The origin's borderline null (which also fails BH; see Statistical Verification summary) cannot support the refutation claim and must be downgraded.
- **Corrected test:** Neuron-level block-permutation test on the Z−XY error-rate difference (neuron = independent unit) plus a risk-rate-ratio with 95% CI, replacing the edge-independent Pearson chi-square (1.4M edges that share nodes are autocorrelated, so the chi-square overstates effective n).
- **Corrected result:** Risk-rate ratio (Z/XY) = 1.0178, 95% CI [0.9993, 1.0366]; absolute rate diff = 0.065 pp; cluster-permutation two-sided p = 0.8136 (NOT significant). Versus original χ² = 3.5328, p = 6.0165e-02. The corrected test agrees the origin shows no detectable orientation effect, with a trivial effect size (RR≈1.02, CI brushing 1).
- **Post-correction verdict:** UPHELD (for the origin null) — the cluster-aware test confirms no significant Z-vs-XY difference (p = 0.8136), and the effect size is negligible. But the report correctly reframes this as absence of evidence, not proof of no anisotropy.
- **Corrected generalization:** PARTIAL — 794491: cluster-permutation p = 0.09745 (not significant, RR = 1.1645 CI [1.138, 1.192], Z slightly higher); 794495: cluster-permutation p = 0.0004998 (SIGNIFICANT, RR = 0.7852 CI [0.767, 0.804], Z *lower*). So even under the correct neuron-level test the origin/794491 null does not hold on 794495, where there is a real significant orientation effect in the opposite direction — confirming the finding is volume-specific, not a stable null.

### 4. (Priority 0.265 · Surprise 0.414) Split-error risk rises with centrifugal branch order (topological depth from soma).
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 139 · **Belief:** Leaning False → Leaning True (0.375→0.6406) · **Direction:** Positive
- **Tested:** Whether the per-edge probability of a split error increases with branch order, independent of local cable thickness.
- **Conclusion:** Over 1.4M edges, logistic regression found branch order a significant positive predictor of split errors (coef = 0.0194, p < 0.001); split rates stay low and stable through order ~30, then spike and grow volatile in deeper branches (>3% near order 53). The positive surprisal (+0.414) reflects the loop revising upward from doubt that depth matters.
- **Caveats:** The thickness covariate (`norm_thickness`) had zero variance and was dropped, so the "independent of thickness" clause could not actually be controlled — thickness effects cannot be ruled out. Deep-order rates are volatile due to sparse data at high orders.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded edges=1409045, branch_order coef=0.0194, z=16.369, p<0.001; rerun edges=1409045, branch_order coef=0.0194, z=16.369, p<0.001 (norm_thickness dropped both times) → exact match.
- **Generalization:** DOES-NOT-GENERALIZE
- **Across datasets:** the positive branch-order → split-risk slope flips sign on BOTH extras. origin (789202): branch_order coef=+0.0194, z=16.37, p<0.001 (risk rises with depth). ds_794491: coef=−0.0157, z=−7.66, p<0.001 (risk *decreases* with depth). ds_794495: coef=−0.0063, z=−4.20, p<0.001 (risk *decreases* with depth). All three are significant but the direction is reversed on both extra datasets, so the "deeper branches split more" conclusion is dataset-specific to the origin.
- **Verdict:** MAJOR
- **Test:** `statsmodels.Logit` (binary logistic regression) of is_split on branch_order over 1,409,045 edges; branch_order coef=0.0194, z=16.369, p<0.001. The `norm_thickness` covariate was dropped because it had zero variance ("constant radius").
- **Statistical issues:** (1) The hypothesis explicitly claims the effect holds "independent of the local cable thickness," but the thickness covariate had zero variance and was dropped, so the model controls for nothing — the "independent of thickness" clause is structurally untestable in this code, not confirmed. (2) Edge non-independence: 1.4M edges within neurons are spatially correlated, so the z=16.4 (and the p<0.001) overstates the effective information; the true standard error is larger. (3) Branch-order is assigned by a BFS that increments only at branching nodes and picks an arbitrary soma (max-radius node), so the predictor is itself noisy.
- **Logic issues:** The conclusion ("deeper topological branches are more susceptible to splitting, independent of thickness") asserts both a directional effect and a thickness-controlled mechanism the experiment never isolated. Positive surprisal (+0.414) was recorded for a slope that reverses sign on both other volumes.
- **Verdict rationale:** DOES-NOT-GENERALIZE — coef flips from +0.0194 to −0.0157 and −0.0063 (all significant) on the two extras, so "deeper branches split more" is volume-specific; combined with the dropped-covariate fault, the causal/independence claim is unsupported and must be downgraded.
- **Corrected test:** Cluster-robust logistic regression (SEs clustered by neuron) plus a neuron-level permutation cross-check (Spearman of per-neuron branch-order vs split rate), replacing the edge-independent `statsmodels.Logit` whose z=16.4 treated 1.4M autocorrelated within-neuron edges as independent.
- **Corrected result:** Cluster-robust z = 1.990, p = 0.04662 (barely significant), OR per +1 branch order = 1.0196, 95% CI [1.0003, 1.0393] (lower bound essentially at 1); the neuron-level permutation cross-check gives Spearman rho = 0.2308, p = 0.4697 (NOT significant). Versus original coef = 0.0194, z = 16.369, p < 0.001 (treated as overwhelmingly significant). Once edge non-independence is handled, the z collapses from 16.4 to ~2 and the permutation test is non-significant.
- **Post-correction verdict:** OVERTURNED — the neuron-level permutation test (the cleanest unit-of-analysis) is non-significant (p = 0.4697) and the cluster-robust CI lower bound sits at 1.0003; the "overwhelming" p<0.001 was entirely an artifact of treating 1.4M correlated edges as independent.
- **Corrected generalization:** DOES-NOT-GENERALIZE — 794491: permutation p = 0.9146 (not significant), cluster-robust coef = −0.0157 (sign flipped); 794495: permutation p = 0.1708 (not significant), cluster-robust coef = −0.0063, z = −0.538, p = 0.5907. Under the correct test the effect is non-significant on all three and the direction reverses on both extras — the branch-order → split-risk claim does not survive.

### 5. (Priority 0.253 · Surprise 0.284) Super-merges (≥3 fused neurons) cover disproportionately more GT cable than 2-neuron merges.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 24 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether merge segments fusing three or more GT neurons are massive structures covering far more cable per neuron than 2-neuron merges.
- **Conclusion:** Of the merges found, 24 were 2-neuron and 3 were super-merges. Median covered cable was 6.21 mm for 2-neuron merges versus 35.19 mm for super-merges, a significant difference (Mann-Whitney U p = 0.0059); per-neuron coverage scaled from ~3.1 to ~11.7 mm/neuron. Modest positive surprisal (+0.284) nudged an already-favored belief upward.
- **Caveats:** Only 3 super-merges — an extremely small sample, so the magnitude estimate is fragile. The steps compared *total* cable while the hypothesis specified *per-neuron*; review argues the conclusion still holds, but the tested quantity differed slightly from the claim.
- **Reproduction:** DIVERGED (code: revised-loading)
- **Rerun result:** the sample counts and significance both changed materially. Recorded: 2-neuron merges n=24, super-merges n=3, Mann-Whitney U=0.0, p=5.8861e-03 (significant). Rerun: 2-neuron merges n=8, super-merges n=1, U=0.0, p=2.2222e-01 (NOT significant — p crosses 0.05). The median covered-cable values (6.2054 mm vs 35.1873 mm) are unchanged, but the n collapse (24→8, 3→1) makes the Mann-Whitney test non-significant on the provided pkl, overturning the recorded conclusion. The merge-detection step on this dataset finds far fewer merges than the recorded run, so the small-sample test no longer reaches significance.
- **Generalization:** PARTIAL
- **Across datasets:** super-merge counts are tiny and inconsistent. origin-rerun (789202): 2-neuron n=8 (median 6.21 mm) vs super n=1 (35.19 mm), U=0.0, p=2.22e-01 (not significant). ds_794491: 37 two-neuron merges but ZERO super-merges → "Not enough data to perform" the test (cannot assess). ds_794495: 2-neuron n=24 (median 5.42 mm) vs super n=2 (median 168.42 mm), U=0.0, p=6.15e-03 (significant, super-merges far larger). Direction (super-merges cover much more cable) holds wherever there are ≥1 super-merge and is significant on 794495, but the origin-rerun is non-significant and 794491 has no super-merges — too sample-starved to call a robust effect.
- **Verdict:** MAJOR
- **Test:** Two-sided Mann-Whitney U (`scipy.stats.mannwhitneyu`) comparing covered cable of 2-neuron merges (n=24) vs super-merges (n=3); recorded U=0.0, p=5.8861e-03; medians 6.2054 mm vs 35.1873 mm.
- **Statistical issues:** (1) Underpowered — n=3 super-merges (and U=0.0 with all three super-merges above all 2-neuron merges) is the minimum configuration that can even reach p<0.01; a single reclassified merge moves the conclusion. The reproduction confirms exactly this: on the provided pkl the counts collapse to n=8 / n=1 and p rises to 0.2222 (NOT significant). (2) Construct mismatch — the hypothesis is about cable *per neuron*, but the code compares *total* `seg_cable[lab]` per merge label; total cable is mechanically larger for super-merges simply because they span more neurons, so the test partly tests its own definition.
- **Logic issues:** The conclusion ("super-merges encapsulate disproportionately larger amounts of physical GT cable") is stated as strong support, but the tested quantity (total cable) differs from the claimed quantity (per-neuron cable), and the inference rests on 3 points. Positive surprisal (+0.284) overstates the evidence.
- **Verdict rationale:** DIVERGED on reproduction (p 0.0059 → 0.2222, crossing 0.05) and PARTIAL on generalization (untestable on 794491, significant only on 794495). A finding whose significance evaporates on re-run with n=1 in one arm is not trustworthy; downgrade.
- **Corrected test:** Mann-Whitney U on PER-NEURON cable (the quantity the hypothesis actually claims) with a Cliff's delta effect size and bootstrap CI, replacing the original test on TOTAL cable (which is mechanically larger for super-merges because they span more neurons — the test partly tested its own definition).
- **Corrected result:** On the per-neuron construct, n_super = 1 vs n_2neuron = 8, Mann-Whitney U = 8.0, exact two-sided p = 0.2222 (NOT significant); Cliff's delta = 1.0 [1.0, 1.0] but flagged "n_super=1 too small for reliable inference." Versus original (total cable) U = 0.0, p = 5.8861e-03. With the correct per-neuron quantity AND the provided-pkl merge counts, the test is non-significant.
- **Post-correction verdict:** OVERTURNED — the corrected per-neuron test is non-significant (p = 0.2222) and rests on a single super-merge; the original p = 0.0059 came from both the wrong (total-cable) construct and a sample that collapsed on re-run.
- **Corrected generalization:** INCONCLUSIVE / PARTIAL — 794491: zero super-merges, "not enough data to perform the test" (INCONCLUSIVE); 794495: per-neuron n_super = 2 vs n_2neuron = 24, U = 46.0, p = 0.02462, Cliff's delta = 0.9167 [0.75, 1.0] (significant, super larger, but still "n_super=2 too small"). The direction (super > 2-neuron) holds where ≥1 super-merge exists, but the evidence is too sample-starved to call a robust effect under the correct construct.

### 6. (Priority 0.253 · Surprise 0.284) Omission errors are far more frequent at branch points than on linear cable.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 32 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether omit errors occur at higher rates at topological branch nodes (degree > 2) than at linear nodes (degree = 2).
- **Conclusion:** Branch nodes had an ~11.95% omission rate versus ~2.76% for linear nodes — a >4× difference with χ² = 1567.69 and p < 0.0001. This strongly supports targeting complex branching junctions for missing cable during proofreading. Positive surprisal (+0.284).
- **Caveats:** None noted; review reports a faithful, deviation-free implementation.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded branch omit=11.9479%, linear omit=2.7602%, χ²=1567.6878, p≈0; rerun identical contingency table (606/4466 vs 38610/1360197), χ²=1567.6878, p≈0 → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** branch nodes show a higher omit rate than linear nodes on all three, highly significant. origin (789202): branch 11.95% vs linear 2.76%, χ²=1567.7, p≈0 (~4.3×). ds_794491: branch 7.54% vs linear 3.41%, χ²=193.6, p=5.13e-44 (~2.2×). ds_794495: branch 6.52% vs linear 1.72%, χ²=974.9, p=5.24e-214 (~3.8×). Effect size is smaller on 794491 but the direction and significance hold everywhere.
- **Verdict:** MINOR
- **Test:** Pearson chi-square (`scipy.stats.chi2_contingency`) on a 2×2 table of branch nodes (degree>2) vs linear nodes (degree=2) × omitted/labeled; contingency 606/4466 (branch) vs 38610/1360197 (linear); χ²=1567.6878, p≈0; omit rates 11.95% vs 2.76%.
- **Statistical issues:** Node-level independence is technically violated (neighboring nodes share edges), so the χ² overstates effective n; expected cell counts are all large, so the test itself is well-posed. The effect size, however, is large (≈4.3× relative rate), so the assumption violation does not threaten the qualitative conclusion.
- **Logic issues:** None of consequence. The conclusion (target branch junctions for missing cable) follows from a >4× rate difference; correlation is not over-claimed as a proven model mechanism.
- **Verdict rationale:** Large, reproduced, and GENERALIZES (2.2×–4.3×, p≤5e-44 on all three). The only blemish is the non-independence of nodes, which is immaterial at this effect size — hence MINOR rather than SOUND.

### 7. (Priority 0.253 · Surprise 0.284) Angular inertia (~153° vs ~90°) reliably identifies the true continuation across a split.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 33 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether the true continuing path across a split stays closer to a straight 180° line than a nearby false-candidate edge from a different neuron.
- **Conclusion:** Across 13,582 split configurations, true continuations averaged 152.96° versus 90.17° for false candidates; distributions were distinct (KS = 0.746, p ≈ 0) and angular alignment discriminated with ROC-AUC 0.9322. Directional inertia is thus a strong local heuristic for an agentic proofreader bridging splits. Positive surprisal (+0.284).
- **Caveats:** None noted; review confirms no deviation from plan.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded n=13582, true=152.96°, false=90.17°, KS=0.7460 (p≈0), ROC-AUC=0.9322; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** angular inertia discriminates true vs false continuations near-identically on all three. origin (789202): true 152.96° vs false 90.17°, KS=0.7460 (p≈0), ROC-AUC=0.9322 (n=13582). ds_794491: 153.49° vs 90.15°, KS=0.7335, ROC-AUC=0.9281 (n=15637). ds_794495: 155.17° vs 90.08°, KS=0.7559, ROC-AUC=0.9365 (n=15929). Very stable, robust heuristic.
- **Verdict:** SOUND
- **Test:** Two-sample Kolmogorov–Smirnov (`scipy.stats.ks_2samp`) on true-continuation vs false-candidate angles plus ROC-AUC; n=13,582 split node configurations; KS=0.7460, p≈0; means 152.96° vs 90.17°; ROC-AUC=0.9322.
- **Statistical issues:** The two angle samples are paired per split node, and KS assumes independent samples; with KS=0.746 (a near-total distributional separation) the dependence does not affect the qualitative result, and the ROC-AUC (a paired-friendly ranking metric) corroborates it. The false candidate is the geometrically nearest node from another neuron, which is a sensible, conservative comparator.
- **Logic issues:** None. The conclusion (directional inertia is a strong local split-bridging heuristic) is exactly what a 63° mean gap and AUC 0.93 support; no causal overreach.
- **Verdict rationale:** Very large effect, exact reproduction, and GENERALIZES with ROC-AUC 0.93 on all three volumes. KS independence is a minor technicality at this separation — SOUND.

### 8. (Priority 0.253 · Surprise 0.284) Omitted cable clusters spatially closer to merge sites than correct cable.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 36 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether omit errors are spatially clustered around merge errors, suggesting the model sacrifices thin processes when fusing dominant structures.
- **Conclusion:** Comparing 49,295 omit nodes to a length-matched sample of 49,295 correct nodes against 67 merge sites, median distance to nearest merge was 1,812.73 µm for omit nodes vs 1,959.88 µm for correct nodes — a highly significant difference (Mann-Whitney U p = 1.60e-66). This supports a systematic bias where merges co-occur with dropped adjacent processes. Positive surprisal (+0.284).
- **Caveats:** The absolute median gap (~147 µm out of ~1,800 µm) is small in practical terms despite extreme significance from the large sample. Only 67 merge sites anchor the distance computation.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 67 merge sites, omit median=1812.73 µm, correct median=1959.88 µm, U=1138194675.0, p=1.6036e-66; rerun identical → exact match.
- **Generalization:** PARTIAL
- **Across datasets:** "omit nodes are closer to merge sites than correct nodes" holds on one extra and *reverses* on the other. origin (789202): omit median 1812.73 µm < correct 1959.88 µm (omit closer), U=1.14e9, p=1.60e-66 (significant). ds_794495: omit median 1422.53 µm < correct 1670.94 µm (omit closer), p=9.78e-226 (significant, same direction). ds_794491: omit median 1103.24 µm > correct 842.11 µm (omit *farther*), one-sided p=1.0000 (not significant; direction flipped). So the spatial co-clustering of omits with merges holds on 794495 but not on 794491.
- **Verdict:** MAJOR
- **Test:** One-sided Mann-Whitney U (`alternative='less'`) of distance-to-nearest-merge for 49,295 omit nodes vs 49,295 length-matched correct nodes against 67 merge sites; U=1138194675.0, p=1.6036e-66; medians 1812.73 µm (omit) vs 1959.88 µm (correct).
- **Statistical issues:** (1) Significance driven purely by sample size — the median difference is ~147 µm on a ~1,800 µm baseline (≈8%), a practically trivial shift, yet n≈49k per arm forces p=1.6e-66. (2) The 49k nodes are not independent observations: they are clustered along neurites and their distances are all measured to the same 67 anchor points, so the effective n is far smaller than 49,295 and the p-value is massively overstated. (3) Only 67 merge anchors define the entire distance field.
- **Logic issues:** The conclusion infers a systematic mechanism ("the model sacrifices thin adjacent processes when fusing") from a tiny median offset; the spatial-clustering claim overreaches what an 8% median difference can support, and the direction is not even stable.
- **Verdict rationale:** PARTIAL generalization — the effect *reverses* on 794491 (omit nodes farther, one-sided p=1.0000). A finding that is both effect-size-trivial/over-powered and direction-unstable across volumes should not be trusted as a real co-clustering law; downgrade.
- **Corrected test:** Neuron-cluster permutation test (neuron = independent unit) on the omit−correct median distance difference, plus Cliff's delta with bootstrap CI, replacing the 49k-node Mann-Whitney that treated spatially clustered nodes referenced to only 67 anchors as independent.
- **Corrected result:** Cliff's delta = −0.0642, 95% CI [−0.101, −0.028] (trivial, ≈−7.5% median shift); neuron-cluster permutation one-sided p = 0.4771 (NOT significant; observed neuron-level median diff only −17.33 µm). Versus original naive U = 1.14e9, p = 1.6036e-66. Once the unit of analysis is the neuron, the "omit closer to merge" effect is not significant.
- **Post-correction verdict:** OVERTURNED — the cluster-permutation test is non-significant (p = 0.4771) and the effect size is trivial; the p = 1.6e-66 was purely an n-inflation artifact.
- **Corrected generalization:** DOES-NOT-GENERALIZE — 794491: Cliff's delta = +0.253 (omit *farther*), permutation p = 0.8108 (not significant, direction reversed); 794495: Cliff's delta = −0.133, permutation p = 0.5007 (not significant despite naive p = 9.78e-226). Under the correct neuron-cluster test the co-clustering effect is non-significant on all three volumes.

### 9. (Priority 0.253 · Surprise 0.284) Split errors are spatially clustered into localized "error zones".
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 37 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether a split edge is more likely than a correct edge to have another split within a 30 µm radius (localized artifact zones).
- **Conclusion:** Across 6,805 split edges and a matched 6,805 correct edges, split edges averaged 0.97 split neighbors within 30 µm versus only 0.10 for correct edges (Mann-Whitney U p ≈ 0). Splits cluster regionally, supporting proofreading workflows that route reviewers to dense error hotspots for batch correction. Positive surprisal (+0.284).
- **Caveats:** None noted; review confirms faithful implementation.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 6805 split edges, mean split-neighbors 0.97 (split) vs 0.10 (correct), U=34661344.5, p≈0; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** split edges have far more nearby splits than correct edges on all three, p≈0 throughout. origin (789202): 0.97 vs 0.10 split-neighbors within 30 µm, U=3.47e7, p≈0. ds_794491: 1.39 vs 0.28, U=4.68e7, p≈0. ds_794495: 1.26 vs 0.11, U=5.11e7, p≈0. Split clustering into error zones is robust.
- **Verdict:** MINOR
- **Test:** One-sided Mann-Whitney U (`alternative='greater'`) on count of split neighbors within 30 µm for 6,805 split edges vs 6,805 matched correct edges; U=34661344.5, p≈0; mean 0.97 vs 0.10 split-neighbors.
- **Statistical issues:** The neighbor-count metric is computed against the split-edge KD-tree, so split edges count *other splits* in a set they belong to — a built-in asymmetry that inflates the split-group counts. With a ~10× mean difference the conclusion still holds, but the magnitude is partly a construction artifact, and split edges are not mutually independent observations.
- **Logic issues:** The conclusion (splits form localized error zones supporting hotspot-routing) follows from a near-10× contrast and is appropriately operational rather than mechanistic.
- **Verdict rationale:** Large, reproduced, GENERALIZES (0.97/1.39/1.26 vs 0.10/0.28/0.11). The metric-asymmetry and edge non-independence keep it at MINOR rather than SOUND.
- **Corrected test:** Neuron-cluster permutation test (neuron = independent unit) on the split−correct mean split-neighbor gap, plus Cliff's delta with cluster bootstrap CI, replacing the per-edge Mann-Whitney (split edges are non-independent).
- **Corrected result:** Cliff's delta = 0.4913, cluster bootstrap 95% CI [0.4363, 0.5600] (moderate-to-large); neuron-cluster permutation one-sided p = 0.0002 (SIGNIFICANT; observed neuron-level gap = 0.8144). Versus original mean 0.97 vs 0.10 split-neighbors, MW p ≈ 0. The clustering effect survives the cluster-aware test with a substantial effect size.
- **Post-correction verdict:** UPHELD — significant under the neuron-cluster permutation test (p = 0.0002) with a moderate-large Cliff's delta (0.49, CI excludes 0); the split-clustering "error zone" finding is robust to the independence correction.
- **Corrected generalization:** GENERALIZES — 794491: Cliff's delta = 0.5193 [0.460, 0.587], permutation p = 0.0004 (significant); 794495: Cliff's delta = 0.5995 [0.559, 0.641], permutation p = 0.0002 (significant). Significant with comparable/larger effect size on both extras.

### 10. (Priority 0.253 · Surprise 0.284) A* path-finding on the fragments graph repairs 86% of splits without inducing merges.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 39 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether graph A* search penalizing sharp deviations (>45°) and sudden radius changes can reconnect >40% of split edges on the U-Net fragments graph without new merges.
- **Conclusion:** The agent connected 5,881 of 6,805 targeted splits (86.42% success) without crossing into different GT neurons, far exceeding the 40% threshold; simulated repair lifted edge accuracy from 78.71% to 79.13% (+0.42%). Positive surprisal (+0.284) confirms graph path-finding is an effective split-repair strategy.
- **Caveats:** Net accuracy gain (+0.42%) is modest at dataset scale. "No merge induced" is judged via GT labels in simulation, not a live segmentation, so real-world false-merge risk may differ.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 5881/6805 splits resolved (86.42%), accuracy 78.71%→79.13% (+0.42%); rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** A* split-repair success far exceeds the 40% threshold on all three. origin (789202): 5881/6805 = 86.42% resolved, accuracy 78.71%→79.13% (+0.42%). ds_794491: 6648/7847 = 84.72%, 74.77%→75.95% (+1.18%). ds_794495: 7166/7988 = 89.71%, 68.55%→69.07% (+0.53%). Success rate is consistently ~85-90%.
- **Verdict:** MINOR
- **Test:** Descriptive simulation — A* path-finding on the fragments graph with turn/radius penalties; 5,881/6,805 targeted splits resolved (86.42%) vs the hypothesis threshold of >40%; simulated edge accuracy 78.71%→79.13% (+0.42%). No inferential statistic.
- **Statistical issues:** None applicable (no hypothesis test; it is a pass/fail against a fixed 40% bar). The 5,000-node A* expansion cap and the `dist_gt<10 µm` merge-validity radius are arbitrary thresholds that affect both numerator and denominator.
- **Logic issues:** "Without inducing merges" is adjudicated against GT labels inside the simulation, not against a live segmentation, so the real-world false-merge rate is not actually measured — the no-merge guarantee is over-claimed relative to deployment. The net accuracy gain (+0.42%) is modest and should not be read as a large quality improvement.
- **Verdict rationale:** The 86% ≫ 40% margin is reproduced and GENERALIZES (85–90% on all three), so the headline holds; MINOR only for the simulation-vs-live merge-safety overreach and arbitrary search caps.

### 11. (Priority 0.253 · Surprise 0.284) Merge segments are runaway "giant" components, far larger than non-merging segments.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 43 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether merge segments cover exponentially more GT cable than non-merging segments (massive overgrown labels vs local blips).
- **Conclusion:** 64 merging segments averaged ~15,449 µm cable (median ~3,099 µm) versus ~532 µm (median ~102 µm) for 8,273 non-merging segments — a ~29× mean difference, highly significant on log-transformed lengths (Welch's t = 16.54, p = 7.32e-25). Merges are driven by enormous overgrown labels, not small local blips. Positive surprisal (+0.284).
- **Caveats:** Only 64 merge segments, and the distribution is heavily skewed (mean ≫ median), so means are outlier-sensitive; the log transform and median mitigate but do not eliminate this.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 64 merging vs 8273 non-merging, mean 15448.96 vs 531.76 µm, Welch t=16.5444, p=7.3232e-25; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** merging segments cover vastly more cable than non-merging on all three, highly significant on log-lengths. origin (789202): mean 15449 vs 532 µm (median 3099 vs 102), Welch t=16.54, p=7.32e-25 (n=64). ds_794491: mean 4457 vs 194 µm (median 1330 vs 29), t=29.03, p≈0 (n=98). ds_794495: mean 16277 vs 477 µm (median 3226 vs 43), t=24.02, p≈0 (n=98). Robust giant-merge signature.
- **Verdict:** SOUND
- **Test:** Welch's two-sample t-test (`ttest_ind(equal_var=False)`) on log10-transformed segment cable lengths; 64 merging vs 8,273 non-merging segments; t=16.5444, p=7.3232e-25; means 15,449 vs 532 µm, medians 3,099 vs 102 µm.
- **Statistical issues:** Appropriate choices throughout — the log10 transform addresses the heavy right-skew, Welch's t handles unequal variances/group sizes, and segments are the natural independent unit (one length per predicted label), so the independence assumption is satisfied here (unlike the edge-level tests). The 64-segment merge class is modest but the ~29× median gap is far from the resolution limit.
- **Logic issues:** Minor wording — the hypothesis says "exponentially larger" while the test demonstrates a large multiplicative (log-scale) difference, not an exponential growth law; the operative conclusion (merges are giant overgrown labels, not local blips) is fully supported.
- **Verdict rationale:** Correct test on independent units, huge effect, exact reproduction, and GENERALIZES (t≥16, p≤7e-25 on all three). SOUND.

### 12. (Priority 0.253 · Surprise 0.284) Omit errors are ~2.4× more likely at extreme Z-depths than at central depths.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 45 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether omit errors are more frequent at the extreme top/bottom Z-coordinates of the imaged volume than in central depths (optical attenuation/scattering).
- **Conclusion:** Extreme-Z nodes (top/bottom 10%) had a 4.44% omit rate (2,283/51,369) versus 1.94% (11,342/585,909) centrally; a Cochran-Mantel-Haenszel test controlling for brain gave pooled OR 2.3561 with p ≈ 0. Axial extremes degrade reconstruction, consistent with depth-dependent signal loss. Positive surprisal (+0.284). Note this concerns Z *depth/position*, distinct from the Z *orientation* effects refuted in #2/#3.
- **Caveats:** Min-max normalization defines "extreme" relative to each volume's own Z-range, conflating true optical depth with volume-edge boundary artifacts. None noted in review otherwise.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded Extreme 4.44% (2283/51369) vs Center 1.94% (11342/585909), CMH OR=2.3561, p≈0; rerun identical → exact match.
- **Generalization:** PARTIAL
- **Across datasets:** the elevated omit rate at extreme Z-depths holds on one extra and vanishes on the other. origin (789202): Extreme 4.44% vs Center 1.94%, CMH OR=2.3561, p≈0 (significant). ds_794495: Extreme 5.37% vs Center 1.76%, OR=3.1672, p≈0 (significant, even stronger). ds_794491: Extreme 4.00% vs Center 3.94%, OR=1.0149, p=7.996e-01 (no effect — rates essentially equal). So depth-extreme omit excess generalizes to 794495 but not to 794491.
- **Verdict:** MAJOR
- **Test:** Cochran–Mantel–Haenszel test (`statsmodels StratifiedTable`) of extreme-Z vs center-Z × omit, stratified by brain; pooled OR=2.3561, p≈0; extreme 4.44% (2283/51369) vs center 1.94% (11342/585909).
- **Statistical issues:** (1) The CMH "controlling for brain" is illusory on the origin — there is only one brain in the pkl, so the stratified test reduces to a single 2×2 chi-square with no actual confounder control. (2) Node non-independence again inflates effective n. (3) "Extreme Z" is defined by per-volume min-max normalization (top/bottom 10% of the *observed* z-range), which conflates true optical depth with volume-edge/boundary truncation artifacts — so even the OR=2.36 may reflect FOV boundary effects rather than attenuation.
- **Logic issues:** The conclusion asserts a specific physical cause ("optical attenuation or scattering at extreme depths degrades signal") that the experiment cannot isolate from boundary/edge artifacts — a mechanism overreach. The OR=2.36 is a genuine moderate effect, but the causal attribution is unsupported.
- **Verdict rationale:** PARTIAL generalization — the effect strengthens on 794495 (OR=3.17) but *vanishes* on 794491 (OR=1.0149, p=0.80), so it is not a stable property; combined with the no-real-stratification and boundary-confound issues, the optical-attenuation claim must be downgraded.
- **Corrected test:** Neuron-cluster permutation test (neuron = independent unit) on the Extreme-vs-Center odds ratio, with a Woolf 95% CI on the single 2×2, replacing the CMH "stratified by brain" which had only one stratum on the origin (no real confounder control) and treated all nodes as independent.
- **Corrected result:** Woolf OR (Extreme vs Center) = 2.3561, 95% CI [2.2504, 2.4668]; but the neuron-clustered permutation OR = 0.5320 with two-sided p = 0.6607 (NOT significant). Versus original CMH OR = 2.3561, p ≈ 0. Once neurons (not nodes) are the unit, the elevated extreme-Z omit rate is not significant on the origin.
- **Post-correction verdict:** OVERTURNED — the neuron-cluster permutation test is non-significant (p = 0.6607); the node-level OR = 2.36 and p ≈ 0 were driven by treating ~640k non-independent nodes as independent draws, and the boundary/depth confound is unaddressed.
- **Corrected generalization:** PARTIAL / INCONCLUSIVE — 794491: neuron OR = nan, permutation p = 0.0002 but the node-level OR is 1.0149 [0.905, 1.138] (no real effect; the tiny significant permutation p with a nan clustered OR is not interpretable as support — INCONCLUSIVE); 794495: neuron-clustered OR = 2.2188, permutation p = 0.1364 (NOT significant despite node-level OR = 3.17, p ≈ 0). Under the correct neuron-level test the extreme-Z omit excess is not significant on any volume.

### 13. (Priority 0.253 · Surprise 0.284) Short omission gaps are internal dropouts bridged by the same segment; long gaps are true terminations.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 58 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether short omit gaps are flanked by the same predicted segment on both sides (internal dropout), while long omits represent genuine fragment boundaries.
- **Conclusion:** Of omit paths analyzed, 307 were "bridged" (same flanking segment) and 4,298 "broken"; bridged gaps were much shorter (mean 18.71 µm, median 13.55 µm) than broken gaps (mean 42.54 µm, median 20.16 µm), significantly so (Mann-Whitney U p = 1.95e-19). Short omits are largely artifactual internal dropouts. Positive surprisal (+0.284).
- **Caveats:** Bridged paths (n = 307) are a small minority (~7%) of omit paths, so conclusions about the bridged class rest on a modest subsample.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 307 bridged (median 13.55 µm) vs 4298 broken (median 20.16 µm), U=458554.0, p=1.9488e-19; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** bridged omit paths are significantly shorter than broken ones on all three. origin (789202): bridged median 13.55 µm (n=307) vs broken 20.16 µm (n=4298), U=458554, p=1.95e-19. ds_794491: bridged 10.99 µm (n=219) vs broken 15.23 µm (n=4353), p=2.75e-09. ds_794495: bridged 12.17 µm (n=330) vs broken 16.24 µm (n=3815), p=1.20e-12. Direction and significance hold throughout.
- **Verdict:** SOUND
- **Test:** One-sided Mann-Whitney U (`alternative='less'`) on omit-path lengths, bridged (n=307) vs broken (n=4298); U=458554.0, p=1.9488e-19; medians 13.55 µm vs 20.16 µm.
- **Statistical issues:** Well-posed — the analysis unit is a connected omit-path component, and distinct components are genuinely independent (unlike per-edge tests), so MW independence holds; non-parametric MW is the right choice for skewed lengths. The bridged class (n=307, ~7%) is a modest minority but adequately powered.
- **Logic issues:** None material. The conclusion (short omits are internal dropouts bridged by the same segment; long omits are true terminations) is exactly what the bridged/broken length contrast supports.
- **Verdict rationale:** Correct test on independent units, clear effect, exact reproduction, and GENERALIZES (p≤2.8e-09 on all three with the same direction). SOUND.

### 14. (Priority 0.253 · Surprise 0.284) High local tortuosity is associated with more split errors (correlation small).
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 59 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether high local tortuosity (curvature, over a 10-edge window) raises split-error likelihood.
- **Conclusion:** Among 6,611 split and 1,091,075 correct edges, split edges had higher median (1.1115 vs 1.0764) and mean (1.2163 vs 1.1175) tortuosity (Mann-Whitney U p ≈ 0). The effect is statistically overwhelming but the point-biserial correlation is small (r = 0.0335), so curvature is a weak though real risk factor. Positive surprisal (+0.284).
- **Caveats:** Effect size is tiny (r = 0.0335); the near-zero p-value is driven by the >1M sample, so curvature explains very little split-error variance on its own.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 6611 split / 1091075 correct, median 1.1115 vs 1.0764, U=4.5781e+09 (p≈0), point-biserial r=0.0335; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** split edges are more tortuous than correct on all three, p≈0, with the same small effect size. origin (789202): median 1.1115 (split) vs 1.0764 (correct), p≈0, r=0.0335. ds_794491: 1.0860 vs 1.0597, p≈0, r=0.0607. ds_794495: 1.0718 vs 1.0555, p≈0, r=0.0253. Direction and significance hold; effect remains small (r≈0.03-0.06) as in the origin.
- **Verdict:** MINOR
- **Test:** One-sided Mann-Whitney U (`alternative='greater'`) plus point-biserial correlation on 10-edge-window tortuosity; 6,611 split vs 1,091,075 correct edges; U=4.5781e9, p≈0; point-biserial r=0.0335 (p≈0); medians 1.1115 vs 1.0764.
- **Statistical issues:** The p≈0 is driven entirely by the >1M sample size; the actual effect is trivial (r=0.0335 → ~0.1% of variance), so curvature explains almost nothing on its own. Edges within neurons are spatially autocorrelated, further inflating the nominal significance. Crucially, the analysis *reports* r and explicitly calls the effect "small," so it does not hide the effect-size problem.
- **Logic issues:** None — the conclusion is correctly hedged as "a weak though real risk factor," matching the r value; no claim of a strong or causal driver.
- **Verdict rationale:** Honest reporting of a tiny-but-consistent effect that reproduces and GENERALIZES (r≈0.03–0.06 on all three). Significance-driven-by-n keeps it at MINOR; it is not FLAWED because the effect size is disclosed and the conclusion is appropriately weak.
- **Corrected test:** Neuron-cluster permutation test (neuron = independent unit) on the split−correct median tortuosity gap, plus Cliff's delta with cluster bootstrap CI, replacing the per-edge Mann-Whitney whose p≈0 was driven by >1M autocorrelated edges.
- **Corrected result:** Cliff's delta = 0.2618, cluster bootstrap 95% CI [0.1583, 0.3674] (small but CI excludes 0); neuron-cluster permutation one-sided p = 0.002 (SIGNIFICANT; neuron-level median gap = 0.0829). Versus original MW p ≈ 0, point-biserial r = 0.0335. The direction and significance survive the cluster correction, with a confirmed-small effect.
- **Post-correction verdict:** UPHELD (weak effect) — significant under the neuron-cluster permutation test (p = 0.002), Cliff's delta CI excludes 0; the original "weak but real risk factor" framing is exactly right and survives the correct test.
- **Corrected generalization:** GENERALIZES — 794491: Cliff's delta = 0.2698 [0.235, 0.346], permutation p = 0.0016 (significant); 794495: Cliff's delta = 0.1973 [0.123, 0.253], permutation p = 0.0002 (significant). Significant small effect on both extras.

### 15. (Priority 0.253 · Surprise 0.284) Split edges sit modestly closer to merge sites than correct edges.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 61 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether split edges are spatially closer to the nearest merge site than correct edges (co-localized segmentation-failure regions).
- **Conclusion:** Across 6,805 split and 1,109,034 correct edges, split edges were closer to merge sites (mean 2,084.76 µm, median 1,794.64 µm) than correct edges (mean 2,265.14 µm, median 1,956.93 µm), highly significant (Mann-Whitney U p = 3.50e-38; Welch's t p = 7.14e-27). Splits and merges co-cluster in localized failure regions. Positive surprisal (+0.284).
- **Caveats:** The absolute median difference (~162 µm out of ~1,800 µm) is small; extreme significance comes from the >1M edge sample, so practical co-localization is weak.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 6805 split / 1109034 correct, split median 1794.64 µm vs correct 1956.93 µm, U=3432660112.0 (p=3.50e-38), Welch t=−10.7131 (p=7.14e-27); rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** split edges sit closer to merge sites than correct edges on all three, significant. origin (789202): split median 1794.64 µm vs correct 1956.93 µm, U-test p=3.50e-38, Welch p=7.14e-27. ds_794491: 818.04 vs 846.80 µm, p=1.98e-14, Welch p=2.18e-03 (significant but smallest gap, ~29 µm). ds_794495: 1476.84 vs 1672.23 µm, p=2.91e-73, Welch p=4.29e-107. Direction and significance hold; the practical gap is small (as flagged in caveats), narrowest on 794491.
- **Verdict:** MINOR
- **Test:** One-sided Mann-Whitney U (`alternative='less'`) and one-sided Welch's t on distance-to-nearest-merge-site; 6,805 split vs 1,109,034 correct edges; MW p=3.50e-38, Welch t=−10.7131 p=7.14e-27; medians 1794.64 vs 1956.93 µm.
- **Statistical issues:** Effect is practically trivial — ~162 µm of a ~1,900 µm median (≈8%) — with the extreme p-values produced solely by the >1.1M correct edges; edges are non-independent and all distances reference the same small merge-site set, so effective n is much smaller than reported. The caveat explicitly flags the small absolute gap.
- **Logic issues:** The conclusion ("split errors and merge sites are spatially clustered into localized failure regions") is plausible but stronger than an 8% median offset warrants; clustering is asserted from a small shift in central tendency.
- **Verdict rationale:** GENERALIZES in direction/significance, but the effect is tiny and over-powered (and shrinks to ~29 µm on 794491). Honestly caveated, so MINOR rather than MAJOR.
- **Corrected test:** Neuron-cluster permutation test (neuron = independent unit) on the split−correct median distance-to-merge gap, plus Cliff's delta with cluster bootstrap CI, replacing the per-edge Mann-Whitney / Welch t whose p≈0 came from >1.1M non-independent edges all referenced to the same small merge-site set.
- **Corrected result:** Cliff's delta = −0.0740, cluster bootstrap 95% CI [−0.2816, 0.1356] (CI CROSSES 0 — not distinguishable from no effect); neuron-cluster permutation one-sided p = 0.6469 (NOT significant; neuron-level gap only 49.42 µm). Versus original MW p = 3.50e-38, Welch p = 7.14e-27. Once neurons are the unit, the "split closer to merge" effect vanishes.
- **Post-correction verdict:** OVERTURNED — the cluster-permutation test is non-significant (p = 0.6469) and the Cliff's delta CI crosses 0; the extreme p-values were entirely an n-inflation artifact from non-independent edges.
- **Corrected generalization:** DOES-NOT-GENERALIZE — 794491: Cliff's delta = −0.0375 [−0.270, 0.150] (crosses 0), permutation p = 0.3913 (not significant); 794495: Cliff's delta = −0.0966 [−0.317, 0.075] (crosses 0), permutation p = 0.3611 (not significant despite naive p = 2.91e-73). Non-significant under the correct test on all three volumes.

### 16. (Priority 0.253 · Surprise 0.284) Split gaps are uniformly small — 100% under 15 µm, 99% under ~6.5 µm.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 63 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether split errors are small localized gaps (>80% of same-neuron disconnected segments separated by <15 µm), to size a repair tool's search radius.
- **Conclusion:** All 6,805 split gaps fell under 15 µm (100%), clustering between 3–5 µm with a max outlier ~9 µm; the 95th/99th percentiles were ~5.85 µm and ~6.50 µm. A repair agent can use a tight ~6.5 µm search radius to catch ~99% of splits while minimizing false-merge risk. Positive surprisal (+0.284). This corroborates the ~6.84 µm threshold from the top-ranked result (#1).
- **Caveats:** None noted; consistent with ID 30. The 80% claim is far exceeded, indicating the prior framing underestimated how tight split gaps are.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 6805 split gaps, 100.00% under 15 µm, 95th pct ~5.85 µm / 99th pct ~6.50 µm; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** split gaps are uniformly tight on all three. origin (789202): 100% < 15 µm, 95th ~5.85 µm, 99th ~6.50 µm (n=6805). ds_794491: 100% < 15 µm, 95th ~5.79 µm, 99th ~6.44 µm (n=7847). ds_794495: 100% < 15 µm, 95th ~5.80 µm, 99th ~6.41 µm (n=7988). The ~6.5 µm search-radius recommendation is essentially identical across datasets.
- **Verdict:** SOUND
- **Test:** Descriptive distributional summary — fraction of 6,805 split gaps under 15 µm and the 95th/99th percentiles; "Percentage of split gaps < 15 µm: 100.00%", 95th ≈5.85 µm, 99th ≈6.50 µm. No inferential test.
- **Statistical issues:** None — this is a percentile/ECDF description, not a test, so no distributional assumptions apply. The split-gap definition (adjacent same-neuron nodes with different non-zero labels) is the natural one and matches the #1 (id 30) construction.
- **Logic issues:** The hypothesis floor ("over 80% under 15 µm") is far exceeded (100%), so the claim is comfortably true; the ~6.5 µm search-radius recommendation follows directly from the 99th percentile.
- **Verdict rationale:** A purely descriptive, unambiguous result that reproduces and GENERALIZES (100% <15 µm, 99th ~6.4–6.5 µm on all three) and corroborates id 30. SOUND.

### 17. (Priority 0.253 · Surprise 0.284) Split errors cluster geodesically closer to branch points than correct edges.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 64 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether proximity to a GT branch point is a risk factor for split errors (complex local geometry fragments the reconstruction).
- **Conclusion:** Among 6,805 split and 1,109,034 correct edges, mean geodesic distance to the nearest branch point was 516.30 µm for splits versus 710.59 µm for correct edges (Mann-Whitney U p = 1.32e-197). Splits concentrate near branch points and lack the long-distance outliers seen in correct edges, supporting a branch-point risk signal. Positive surprisal (+0.284).
- **Caveats:** Review field was "N/A" (no independent audit recorded), so faithful-implementation confirmation is weaker than for other entries.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 6805 split / 1109034 correct, mean distance to branch 516.30 µm (split) vs 710.59 µm (correct), U=2979025451.5, p=1.3239e-197; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** split edges are geodesically closer to branch points than correct edges on all three, significant. origin (789202): mean 516.30 µm (split) vs 710.59 µm (correct), p=1.32e-197 (gap ~194 µm). ds_794491: 384.29 vs 389.49 µm, p=5.16e-67 (significant but practically tiny gap ~5 µm). ds_794495: 423.67 vs 506.89 µm, p=2.97e-27 (gap ~83 µm). Direction and significance hold; effect magnitude is much weaker on 794491.
- **Verdict:** MINOR
- **Test:** Two-sided Mann-Whitney U on geodesic (Dijkstra, multi-source from all branch points) distance to nearest branch point; 6,805 split vs 1,109,034 correct edges; U=2979025451.5, p=1.3239e-197; means 516.30 vs 710.59 µm.
- **Statistical issues:** The geodesic-distance approach is well-implemented (dummy-node multi-source Dijkstra), but the contrast is partly driven by the correct-edge tail: split edges simply lack the long-distance outliers (max ~8,500 vs ~13,000 µm), so part of the mean gap reflects range truncation rather than concentration near branches. Large n inflates significance and the 1.1M correct edges are non-independent. The record's `review` field was "N/A" (no independent audit).
- **Logic issues:** The conclusion (branch-point proximity is a split risk factor) is reasonable but should not be read as branch points *causing* splits — the comparison is observational and the effect collapses to ~5 µm on 794491.
- **Verdict rationale:** GENERALIZES in direction/significance but with a meaningful effect only on two of three volumes (~194 µm and ~83 µm; ~5 µm on 794491). Sound design, modest/variable effect, n-driven significance — MINOR.
- **Corrected test:** Neuron-cluster permutation test (neuron = independent unit) on the split−correct median geodesic-distance-to-branch gap, plus Cliff's delta with cluster bootstrap CI, replacing the per-edge Mann-Whitney on 1.1M non-independent edges.
- **Corrected result:** Cliff's delta = −0.2184, cluster bootstrap 95% CI [−0.3692, −0.0309] (small-moderate, CI excludes 0); neuron-cluster permutation one-sided p = 0.09598 (NOT significant at 0.05; neuron-level gap = −116.52 µm). Versus original MW p = 1.3239e-197. The effect size CI excludes 0 but the cluster-permutation p just misses significance on the origin.
- **Post-correction verdict:** WEAKENED — the Cliff's delta CI [−0.369, −0.031] still indicates splits sit somewhat closer to branch points, but the neuron-cluster permutation test is no longer significant (p = 0.096) on the origin; the p = 1.3e-197 was massively n-inflated.
- **Corrected generalization:** PARTIAL — 794491: Cliff's delta = −0.1356 [−0.246, −0.005], permutation p = 0.009198 (significant); 794495: Cliff's delta = −0.0762 [−0.150, 0.016] (CI crosses 0), permutation p = 0.1342 (not significant). Significant under the correct test on only one of two extras, with small effect sizes throughout.

### 18. (Priority 0.253 · Surprise 0.284) Split errors occur on more tortuous segments (5-hop window) than correct segments.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 73 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether split errors fall on more highly tortuous (curved) segments than straight ones, measured over a 5-hop skeleton window.
- **Conclusion:** Across 1,109,034 correct and 6,805 split edges, split edges had higher median (1.1132 vs 1.0801) and mean (1.2062 vs 1.1194) tortuosity, significant by one-sided Mann-Whitney U (p = 1.66e-276). This replicates ID 59 at a different window size: sharp turns systematically challenge contiguous tracking. Positive surprisal (+0.284).
- **Caveats:** As with ID 59, the median difference is small in absolute terms; the extreme p-value reflects sample size, not large effect magnitude. The two tortuosity tests share most underlying data, so they are not fully independent confirmations.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 1109034 correct / 6805 split, median 1.0801 vs 1.1132, U=4714207979.0, p=1.6599e-276; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** split edges are more tortuous (5-hop window) than correct on all three, significant. origin (789202): split median 1.1132 vs correct 1.0801, p=1.66e-276. ds_794491: 1.0854 vs 1.0617, p≈0. ds_794495: 1.0735 vs 1.0575, p=6.19e-192. Replicates id 59 across datasets; effect small but consistent in direction and significance.
- **Verdict:** MINOR
- **Test:** One-sided Mann-Whitney U (`alternative='greater'`) on 5-hop-window tortuosity; 1,109,034 correct vs 6,805 split edges; U=4714207979.0, p=1.6599e-276; medians 1.1132 (split) vs 1.0801 (correct).
- **Statistical issues:** Same tiny-effect/over-power pattern as id 59 — the median gap is ~0.03 and the p≈0 comes from >1.1M edges; unlike id 59, this run reports *no* effect-size statistic (no point-biserial r), so the practical smallness is less visible in the analysis text. Edges are non-independent. Also, this and id 59 share most of the same underlying data, so they are not independent confirmations of each other.
- **Logic issues:** The conclusion ("overwhelming statistical evidence") leans on the p-value and omits the effect-size caveat that id 59 included, slightly overstating practical importance; the directional claim itself is correct.
- **Verdict rationale:** Direction reproduces and GENERALIZES, but it is a near-duplicate of id 59 with the same trivial effect and is presented without an effect-size hedge — MINOR (significance-driven-by-n, redundant confirmation).
- **Corrected test:** Neuron-cluster permutation test (neuron = independent unit) on the split−correct median 5-hop tortuosity gap, plus Cliff's delta with cluster bootstrap CI (the effect size id 73 omitted), replacing the per-edge Mann-Whitney whose p≈0 came from >1.1M non-independent edges.
- **Corrected result:** Cliff's delta = 0.2451, cluster bootstrap 95% CI [0.1511, 0.3419] (small but CI excludes 0); neuron-cluster permutation one-sided p = 0.0007998 (SIGNIFICANT; neuron-level gap = 0.0725). Versus original MW p = 1.6599e-276 (no effect size reported). Direction and significance survive; the now-reported effect size is small.
- **Post-correction verdict:** UPHELD (weak effect) — significant under the neuron-cluster permutation test (p = 0.0008), Cliff's delta CI excludes 0; mirrors id 59. The conclusion holds but the effect is small, as the now-added effect size makes explicit.
- **Corrected generalization:** GENERALIZES — 794491: Cliff's delta = 0.2455 [0.210, 0.310], permutation p = 0.0014 (significant); 794495: Cliff's delta = 0.1856 [0.139, 0.246], permutation p = 0.0002 (significant). Significant small effect on both extras.

### 19. (Priority 0.253 · Surprise 0.284) Merge sites carry a strong local branch-density signature (ROC-AUC 0.92).
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 84 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether merge sites exhibit higher local branch density in the UNet fragments graph than non-merge regions on the same segments (a geometric signature for auto-resolution).
- **Conclusion:** Across 67 merge sites, mean local branch density within 15 µm was 1.13 branches versus 0.04 at matched controls on the same segments (paired t = 13.35, p = 2.03e-20; Wilcoxon W = 21.0, p = 1.19e-11), with ROC-AUC 0.9236. Spurious local branching is a highly predictive merge signature for automated proofreading. Positive surprisal (+0.284).
- **Caveats:** Only 67 merge sites; the strong ROC-AUC rests on a small positive class, so out-of-sample predictive power is uncertain.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded 67 merge sites, mean density 1.13 vs 0.04, paired t=13.3482 (p=2.0302e-20), Wilcoxon W=21.0 (p=1.1942e-11), ROC-AUC=0.9236; rerun identical → exact match (only a matplotlib deprecation warning in stderr).
- **Generalization:** GENERALIZES
- **Across datasets:** merge sites show much higher local branch density than controls on all three, significant, with strong ROC-AUC. origin (789202): 1.13 vs 0.04 branches/15µm, paired t=13.35, p=2.03e-20, ROC-AUC=0.9236 (n=67). ds_794491: 1.10 vs 0.08, t=12.14, p=2.97e-20, ROC-AUC=0.8825 (n=86). ds_794495: 1.02 vs 0.05, t=13.46, p=1.65e-24, ROC-AUC=0.8857 (n=105). AUC dips slightly (~0.88) but the signature is robust.
- **Verdict:** MINOR
- **Test:** Paired t-test and Wilcoxon signed-rank on local branch density (branches within 15 µm) at 67 merge sites vs matched on-segment controls, plus ROC-AUC; paired t=13.3482 (p=2.0302e-20), Wilcoxon W=21.0 (p=1.1942e-11), ROC-AUC=0.9236; mean 1.13 vs 0.04.
- **Statistical issues:** The paired design (merge site vs control on the *same* segment) is the right choice and the Wilcoxon corroborates the t-test, so the significance is robust despite n=67. The one real issue: the ROC-AUC=0.9236 is computed in-sample on the very same merge/control points used to define the contrast (no held-out split), so it overstates true out-of-sample predictive power — the caveat flags the small positive class.
- **Logic issues:** The conclusion (local branch density is a predictive merge signature for auto-proofreading) slightly overreaches by citing an in-sample AUC as "highly predictive"; the *difference* itself is solidly established.
- **Verdict rationale:** Correct paired test, large effect, reproduces, and GENERALIZES (AUC 0.88–0.92). Downgraded to MINOR only for the in-sample AUC optimism and small n.

### 20. (Priority 0.253 · Surprise 0.284) Omit errors are concentrated on terminal (distal) branches, ~1.8× the internal rate.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 85 · **Belief:** Leaning True → Likely True (0.7083→0.8906) · **Direction:** Positive
- **Tested:** Whether omit errors disproportionately hit terminal branches (paths ending in a leaf node) versus internal segments.
- **Conclusion:** Terminal edges had a 4.38% omit rate (23,792/543,477) versus 2.41% for internal edges (20,898/865,568) — an 81% relative increase, χ² = 4189.94, p < 1e-300. The model struggles to track thin distal ends, so proofreaders should prioritize terminal branches for missing cable. Positive surprisal (+0.284).
- **Caveats:** None noted; review confirms faithful classification and analysis.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded terminal 4.38% (23792/543477) vs internal 2.41% (20898/865568), χ²=4189.9364, p≈0; rerun identical → exact match.
- **Generalization:** GENERALIZES
- **Across datasets:** terminal edges show a higher omit rate than internal on all three, significant. origin (789202): terminal 4.38% vs internal 2.41%, χ²=4189.9, p≈0 (+81% relative). ds_794491: terminal 4.94% vs internal 3.82%, χ²=425.4, p=1.63e-94 (+29% relative). ds_794495: terminal 2.47% vs internal 1.82%, χ²=692.0, p=1.66e-152 (+36% relative). Direction and significance hold; relative excess is smaller on the extras.
- **Verdict:** MINOR
- **Test:** Pearson chi-square (`scipy.stats.chi2_contingency`) on terminal vs internal edges × omit/non-omit; 23792/519685 (terminal) vs 20898/844670 (internal); χ²=4189.9364, p≈0; omit rates 4.38% vs 2.41% (+81% relative).
- **Statistical issues:** Edges are not independent (terminal paths are contiguous runs of edges), so the χ² overstates effective n; expected cell counts are large, so the test is otherwise well-formed. The +81% relative excess is a substantial, non-trivial effect that survives the assumption concern.
- **Logic issues:** None material — the conclusion (prioritize terminal/distal branches for missing cable) follows from a clear rate difference and does not over-claim a mechanism beyond "model struggles with thin distal ends."
- **Verdict rationale:** Large effect, reproduced, GENERALIZES (+29% to +81% relative, p≤1.6e-94 on all three). Edge non-independence is the only issue and it is immaterial at this magnitude — MINOR.

## Reproduction — Summary

**Dataset pkl used:** `/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/cache/dataset_cache_789202_mcl100_add.pkl`

**Breakdown (out of n_rerun = 20):** REPRODUCED 19 · DIVERGED 1 · FAILED 0. All 20 ran on `revised`-loading code (direct `$RERUN_PKL` load); 0 on recorded code. All exited 0 with no timeouts and no environment failures.

**Did NOT reproduce:**
- **Entry 5 · id 24 — DIVERGED (analysis):** merge-severity cable test. Recorded n=24 two-neuron merges / n=3 super-merges, Mann-Whitney U=0.0, p=5.8861e-03 (significant); rerun on the provided pkl found only n=8 / n=1, U=0.0, p=2.2222e-01 (not significant — p crosses 0.05). Median cable values (6.21 vs 35.19 mm) are unchanged, but the far smaller merge counts on this dataset drop the test below significance, overturning the recorded conclusion. This is a data/analysis divergence (different merge-detection counts), not a loading or environment failure.

| Entry | id | Verdict | code_source |
|------:|---:|---------|-------------|
| 1 | 30 | REPRODUCED | revised |
| 2 | 27 | REPRODUCED | revised |
| 3 | 21 | REPRODUCED | revised |
| 4 | 139 | REPRODUCED | revised |
| 5 | 24 | DIVERGED | revised |
| 6 | 32 | REPRODUCED | revised |
| 7 | 33 | REPRODUCED | revised |
| 8 | 36 | REPRODUCED | revised |
| 9 | 37 | REPRODUCED | revised |
| 10 | 39 | REPRODUCED | revised |
| 11 | 43 | REPRODUCED | revised |
| 12 | 45 | REPRODUCED | revised |
| 13 | 58 | REPRODUCED | revised |
| 14 | 59 | REPRODUCED | revised |
| 15 | 61 | REPRODUCED | revised |
| 16 | 63 | REPRODUCED | revised |
| 17 | 64 | REPRODUCED | revised |
| 18 | 73 | REPRODUCED | revised |
| 19 | 84 | REPRODUCED | revised |
| 20 | 85 | REPRODUCED | revised |

**Loading revisions:** every shown record (all 20) ran on revised-loading code that replaced the original dataset search with a direct `$RERUN_PKL` load. Records whose recorded code had searched via glob/hardcoded relative paths that would otherwise miss the provided pkl include ids 64 and 73 ("Found 1 dataset files" / "No dataset files found matching *_add.pkl" gates) and ids 139, 43, 73 (hardcoded `../data/...` paths) — all now load the orchestrator pkl directly. No record hit an ENVIRONMENT failure; the runner's suppressed-package-install notices (ids 21, 32, 84) are benign no-ops, not failures.

## Generalization — Summary

**Extra datasets tested (origin = 789202):** `cache/dataset_cache_794491_mcl100_add.pkl` and `cache/dataset_cache_794495_mcl100_add.pkl`. All 20 reproduced scripts ran to exit 0 on both extra pkls (no timeouts, no load failures) — so no entry is INCONCLUSIVE on execution grounds.

**Breakdown (out of 20):** GENERALIZES 14 · PARTIAL 4 · DOES-NOT-GENERALIZE 2 · INCONCLUSIVE 0.

**Do NOT fully generalize:**
- **Entry 3 · id 21 — DOES-NOT-GENERALIZE:** the near-null Z-dominant vs XY-dominant error-rate finding becomes strongly *significant* on both extras, in *opposite* directions (794491: Z higher, p=5.21e-38; 794495: Z lower, p=4.62e-88). Origin's clean null is dataset-specific.
- **Entry 4 · id 139 — DOES-NOT-GENERALIZE:** branch-order → split-risk slope flips sign on both extras (origin coef=+0.0194; 794491 coef=−0.0157; 794495 coef=−0.0063), all significant. "Deeper branches split more" reverses off the origin.
- **Entry 2 · id 27 — PARTIAL:** the "z-alignment not significant" claim holds on 794491 (p=0.175, but sign flips +) yet is contradicted on 794495 (p=6.75e-06, significant *decrease*); direction unstable (−/+/−).
- **Entry 5 · id 24 — PARTIAL:** super-merge cable test is sample-starved — origin-rerun non-significant (p=0.222), 794491 has zero super-merges (untestable), 794495 significant (p=6.15e-03). Direction (super > 2-neuron) holds only where ≥1 super-merge exists.
- **Entry 8 · id 36 — PARTIAL:** omit-near-merge clustering holds on 794495 (p=9.78e-226) but *reverses* on 794491 (omit farther; one-sided p=1.0000).
- **Entry 12 · id 45 — PARTIAL:** extreme-Z omit excess holds on 794495 (OR=3.17, p≈0) but vanishes on 794491 (OR=1.0149, p=0.7996).

**INCONCLUSIVE:** none.

**Synthesis:** The robust, dataset-independent conclusions are the local-topology / morphology and proofreading-heuristic findings: the ~6.5 µm split-gap proximity threshold (id 30, 63), angular inertia (id 33), A* split repair (id 39), branch-point omission excess (id 32), terminal-branch omission excess (id 85), tortuosity → split risk (id 59, 73), split spatial clustering (id 37), giant-merge cable (id 43), bridged-vs-broken omit lengths (id 58), merge branch-density signature (id 84), and split-near-merge / split-near-branch proximity (id 61, 64) — all 14 hold direction + significance across both extra volumes (a few with weaker effect sizes, e.g. id 17/64 on 794491). In contrast, every finding tied to a *global imaging-axis / Z effect* or a tiny-sample merge statistic is fragile: the two anisotropy-related results (id 21, 27) and the branch-order slope (id 139) either flip sign or become significant in inconsistent directions across datasets, and the merge/depth findings that depend on a handful of merge sites or volume-specific Z extremes (id 24, 36, 45) hold on only one of the two extras. With only two extra datasets the evidence is moderate — the 14 GENERALIZES calls are well-supported (consistent on both), but the PARTIAL/DOES-NOT calls flag that orientation- and small-sample-driven conclusions are likely volume-specific rather than general properties of the segmentation pipeline.

| Entry | id | Verdict | Basis (one line) |
|------:|---:|---------|------------------|
| 1 | 30 | GENERALIZES | ROC-AUC 0.998/0.989/0.995, thr ~6.5 µm on all three |
| 2 | 27 | PARTIAL | non-sig holds on 794491 (sign flips +); 794495 significant decrease p=6.75e-06 |
| 3 | 21 | DOES-NOT-GENERALIZE | both extras significant in opposite directions (p=5.2e-38 / 4.6e-88) |
| 4 | 139 | DOES-NOT-GENERALIZE | branch-order slope flips sign on both extras (+0.019 → −0.016 / −0.006) |
| 5 | 24 | PARTIAL | sample-starved: origin p=0.222, 794491 no super-merges, 794495 p=6.15e-03 |
| 6 | 32 | GENERALIZES | branch omit > linear on all three (χ² 1568/194/975, p≤5e-44) |
| 7 | 33 | GENERALIZES | true ~153° vs false ~90°, ROC-AUC 0.93 on all three |
| 8 | 36 | PARTIAL | holds 794495 (p=9.8e-226); reverses on 794491 (p=1.0) |
| 9 | 37 | GENERALIZES | split-neighbor excess p≈0 on all three |
| 10 | 39 | GENERALIZES | A* resolves 86/85/90% of splits, all ≫40% |
| 11 | 43 | GENERALIZES | merge segments ~10-30× larger, t≥16, p≤7e-25 |
| 12 | 45 | PARTIAL | holds 794495 (OR=3.17); vanishes 794491 (OR=1.01, p=0.80) |
| 13 | 58 | GENERALIZES | bridged shorter than broken on all three (p≤2.8e-09) |
| 14 | 59 | GENERALIZES | split more tortuous, p≈0, small r≈0.03-0.06 on all three |
| 15 | 61 | GENERALIZES | split closer to merge on all three (p≤2e-14) |
| 16 | 63 | GENERALIZES | 100% gaps <15 µm, 99th pct ~6.4-6.5 µm on all three |
| 17 | 64 | GENERALIZES | split closer to branch on all three (p≤5e-67), tiny gap on 794491 |
| 18 | 73 | GENERALIZES | split more tortuous (5-hop) on all three (p≤6e-192) |
| 19 | 84 | GENERALIZES | merge branch-density signature, ROC-AUC 0.92/0.88/0.89 |
| 20 | 85 | GENERALIZES | terminal omit > internal on all three (χ² 4190/425/692) |

## Statistical Verification — Summary

**Audited:** all 20 ranked hypotheses, judged from the recorded `code` / `codeOutput` / `analysis` plus the already-folded Reproduction and Generalization numbers (no experiments re-run).

**Verdict breakdown (5-level scheme SOUND | WEAK | MINOR | MAJOR | CRITICAL; each of the 20 counted once):**

- **SOUND — 5:** entries **1 (id 30), 7 (id 33), 11 (id 43), 13 (id 58), 16 (id 63)** — clean designs with large/unambiguous effects and full generalization (descriptive separability for 30/63; KS on near-disjoint angle distributions for 33; Welch t on log-lengths of independent segments for 43; Mann-Whitney on independent omit-path components for 58).
- **WEAK — 0.**
- **MINOR — 9:** entries **6 (id 32), 9 (id 37), 10 (id 39), 14 (id 59), 15 (id 61), 17 (id 64), 18 (id 73), 19 (id 84), 20 (id 85)** — sound direction that reproduces and generalizes, but each carries one secondary issue (edge/node non-independence inflating significance, effect-size-trivial-but-over-powered tests, in-sample ROC-AUC optimism, or simulation-vs-live overreach).
- **MAJOR — 6:** entries **2 (id 27), 3 (id 21), 4 (id 139), 5 (id 24), 8 (id 36), 12 (id 45)** — each has a concrete test fault (detailed below) that materially undermines the recorded conclusion.
- **CRITICAL — 0.**

**Definitive tally: SOUND 5 · WEAK 0 · MINOR 9 · MAJOR 6 · CRITICAL 0 = 20.**

**The six MAJOR findings and their concrete test faults (the discoveries a scientist should NOT trust as stated):**

- **Entry 2 · id 27** — p-value manufactured by treating a variational-Bayes posterior mean/SD as a frequentist Wald z (`p=2*(1-Φ(coef/sd))`); a VB posterior is not a sampling distribution, so the reported p=0.058 is an invalid significance test, and it is used to license a null ("z-axis not a primary driver"). Fix: refit a proper frequentist mixed GLM (or report a credible interval, not a p), on the full data not a 20k subsample, and clustered/robust SEs.
- **Entry 3 · id 21** — chi-square independence on 1.4M edges that share nodes (autocorrelated), and "fail to reject (p=0.060)" escalated to "refutes imaging-anisotropy bias"; DOES-NOT-GENERALIZE (both extras significant, opposite signs). Fix: use a clustered/permutation test by neuron and never read a non-significant p as proof of no effect.
- **Entry 4 · id 139** — the `norm_thickness` control has zero variance and is dropped, so the hypothesis's "independent of cable thickness" clause is structurally untestable; edge non-independence; slope reverses sign on both extras. Fix: obtain a real thickness covariate (the radii are constant in this pkl) before claiming thickness-independence.
- **Entry 5 · id 24** — Mann-Whitney with n=3 super-merges (U=0.0, p=0.0059) that DIVERGES to n=1 / p=0.222 on re-run, and tests *total* cable while the hypothesis is *per-neuron*. Fix: per-neuron normalization and far more super-merge instances before any inference.
- **Entry 8 · id 36** — p=1.6e-66 from a ~8% median offset (147 µm of ~1,800 µm) over 49k non-independent nodes referenced to only 67 anchors; direction *reverses* on 794491 (p=1.0000). Fix: report the effect size, use independent units, and treat as non-generalizing.
- **Entry 12 · id 45** — CMH "controlling for brain" has only one stratum on the origin (no real control); "extreme Z" conflates optical depth with FOV-boundary truncation; causal optical-attenuation claim; effect vanishes on 794491 (OR=1.01, p=0.80). Fix: separate boundary from depth and drop the causal language.

### Multiple comparisons (Benjamini–Hochberg FDR)

Collected the single headline p-value from each of the 20 hypotheses. Three entries report **no inferential p** — 1/id 30 (ROC-AUC), 10/id 39 (success-rate vs a 40% bar), 16/id 63 (percentile) are descriptive — leaving **m = 17 numeric p-values**. Entries reporting p≈0 are floored at 1e-300; id 139's "p<0.001" is entered as 1e-3.

BH at α=0.05 (rank k, critical value k/m·α): the largest rank with p ≤ k/m·α is **rank 15 (entry 5 / id 24, p=5.886e-03 ≤ 0.0441)**, so the BH threshold is p* = 5.886e-03 and **15 of 17 survive**.

| rank | entry · id | test | p-value | crit (k/m·α) | survives FDR? |
|----:|-----------|------|--------:|-------------:|:-------------:|
| 1 | 6 · 32 | chi-square | ~1e-300 | 0.0029 | YES |
| 2 | 7 · 33 | KS | ~1e-300 | 0.0059 | YES |
| 3 | 9 · 37 | Mann-Whitney | ~1e-300 | 0.0088 | YES |
| 4 | 12 · 45 | CMH | ~1e-300 | 0.0118 | YES |
| 5 | 14 · 59 | Mann-Whitney / point-biserial | ~1e-300 | 0.0147 | YES |
| 6 | 20 · 85 | chi-square | ~1e-300 | 0.0176 | YES |
| 7 | 18 · 73 | Mann-Whitney | 1.66e-276 | 0.0206 | YES |
| 8 | 17 · 64 | Mann-Whitney | 1.32e-197 | 0.0235 | YES |
| 9 | 8 · 36 | Mann-Whitney | 1.60e-66 | 0.0265 | YES |
| 10 | 15 · 61 | Mann-Whitney | 3.50e-38 | 0.0294 | YES |
| 11 | 11 · 43 | Welch t | 7.32e-25 | 0.0324 | YES |
| 12 | 19 · 84 | paired t | 2.03e-20 | 0.0353 | YES |
| 13 | 13 · 58 | Mann-Whitney | 1.95e-19 | 0.0382 | YES |
| 14 | 4 · 139 | logistic (branch_order) | 1.0e-03 | 0.0412 | YES |
| 15 | 5 · 24 | Mann-Whitney | 5.89e-03 | 0.0441 | YES |
| 16 | 2 · 27 | mixed-GLM Wald (invalid) | 5.82e-02 | 0.0471 | no |
| 17 | 3 · 21 | chi-square | 6.02e-02 | 0.0500 | no |

**FDR reading:** The two findings that fail BH (entry 2 / id 27 at p=0.058 and entry 3 / id 21 at p=0.060) are precisely the two anisotropy *null* results — they were already non-significant, so failing BH is consistent with, not additional to, their "fail to reject" status; the substantive problem with them is the absence-of-evidence logic and the invalid Wald construction (id 27), not multiplicity. Entry 5 / id 24 (p=5.886e-03) survives BH only nominally and is the marginal-rank result; given its DIVERGED reproduction (re-run p=0.222) it should be regarded as not robust despite clearing the FDR line. Entry 4 / id 139 survives FDR but the surviving slope reverses sign across datasets, so FDR survival does not rescue its generalization failure. The 13 large-effect findings (ranks 1–13) survive FDR with enormous margin; for several of those (id 36, 59, 61, 64, 73, 85) the survival is driven by sample size rather than effect magnitude, so FDR survival should not be read as practical importance — see the per-entry effect-size notes.

## Excluded (no surprisal score)

The helper dropped 2 hypotheses lacking a surprisal score (`n_dropped_missing_surprisal` = 2):
- run-4--ground-truth-error-annotations-revised-version_2026-06-20 · ID 10
- run-4--ground-truth-error-annotations-revised-version_2026-06-20 · ID 41

## Statistical Test Corrections — Summary

**Flagged for a test fix:** 11 hypotheses (ids 21, 24, 27, 36, 37, 45, 59, 61, 64, 73, 139). Each received a corrected script that replaced the faulty test (per-edge / per-node independence assumed, invalid VB-as-Wald p, wrong construct, or n-inflated p with no effect size) with the proper unit-of-analysis test — almost always a **neuron-cluster permutation test** (neuron = independent unit) plus a **Cliff's delta / rank-biserial effect size with bootstrap CI**, or a cluster-robust regression. The corrected results were re-measured by the driver (all 11 ran to exit 0 on the origin and both extra pkls).

**Post-correction breakdown:** UPHELD 5 · WEAKENED 1 · OVERTURNED 5.

| id | Entry | Original headline | Corrected headline (right test) | Post-correction verdict | Corrected generalization |
|---:|------:|-------------------|---------------------------------|:------------------------|:-------------------------|
| 27 | 2 | Z-alignment does NOT significantly raise error risk (invalid VB-Wald p=0.058, OR=0.80) | Neuron-permutation p=0.5721 — no significant z-alignment effect (valid null) | **UPHELD** (null) | PARTIAL (perm null on all 3; cluster-robust GLM flags 794495 p=2e-11) |
| 21 | 3 | No significant Z-vs-XY error-rate difference (χ²=3.53, p=0.060) | RR=1.018 CI[0.999,1.037], cluster-perm p=0.8136 — trivial, not significant | **UPHELD** (null, origin) | PARTIAL (794491 perm p=0.097; 794495 perm p=5e-4, Z *lower*) |
| 139 | 4 | Split risk rises with branch order (z=16.4, p<0.001) | Neuron-permutation p=0.4697; cluster-robust z=1.99 CI lower=1.0003 | **OVERTURNED** | DOES-NOT-GENERALIZE (perm n.s. all 3; sign flips on both extras) |
| 24 | 5 | Super-merges cover more cable than 2-neuron merges (total cable, U=0, p=0.0059) | Per-neuron cable, U=8, p=0.2222, n_super=1 (unreliable) | **OVERTURNED** | INCONCLUSIVE/PARTIAL (794491 no super-merges; 794495 p=0.025) |
| 36 | 8 | Omit nodes cluster closer to merges (U=1.1e9, p=1.6e-66) | Cliff's δ=−0.064 CI[−0.10,−0.03], neuron-perm p=0.4771 | **OVERTURNED** | DOES-NOT-GENERALIZE (perm n.s. both extras; reverses on 794491) |
| 37 | 9 | Split edges have more nearby splits (mean 0.97 vs 0.10, MW p≈0) | Cliff's δ=0.49 CI[0.44,0.56], neuron-perm p=0.0002 | **UPHELD** | GENERALIZES (perm p≤4e-4, δ 0.52–0.60 both extras) |
| 45 | 12 | Extreme-Z omit excess (CMH OR=2.36, p≈0) | Neuron-clustered OR=0.53, neuron-perm p=0.6607 | **OVERTURNED** | PARTIAL/INCONCLUSIVE (794491 nan OR; 794495 perm p=0.136 n.s.) |
| 59 | 14 | High tortuosity → more splits (MW p≈0, r=0.034) | Cliff's δ=0.26 CI[0.16,0.37], neuron-perm p=0.002 | **UPHELD** (weak) | GENERALIZES (perm p≤0.0016 both extras, small δ) |
| 61 | 15 | Split edges closer to merge sites (MW p=3.5e-38) | Cliff's δ=−0.074 CI[−0.28,0.14] (crosses 0), neuron-perm p=0.6469 | **OVERTURNED** | DOES-NOT-GENERALIZE (perm n.s. both extras; CIs cross 0) |
| 64 | 17 | Splits cluster near branch points (MW p=1.3e-197) | Cliff's δ=−0.22 CI[−0.37,−0.03], neuron-perm p=0.096 | **WEAKENED** | PARTIAL (794491 perm p=0.009; 794495 perm p=0.134 n.s.) |
| 73 | 18 | Splits on more tortuous (5-hop) segments (MW p=1.7e-276) | Cliff's δ=0.25 CI[0.15,0.34], neuron-perm p=0.0008 | **UPHELD** (weak) | GENERALIZES (perm p≤0.0014 both extras, small δ) |

**Which conclusions changed (WEAKENED or OVERTURNED — 6 of 11):**

- **id 139 (entry 4) — OVERTURNED.** Wrong test: edge-independent logistic regression (z=16.4) on 1.4M autocorrelated within-neuron edges. Under cluster-robust SEs + neuron permutation the effect is non-significant (perm p=0.4697) and reverses sign on both extras.
- **id 24 (entry 5) — OVERTURNED.** Wrong test/construct: Mann-Whitney on *total* cable (the hypothesis is *per-neuron*) with n=3 super-merges. Per-neuron MW with n_super=1 gives p=0.2222.
- **id 36 (entry 8) — OVERTURNED.** Wrong test: 49k-node Mann-Whitney treating clustered nodes referenced to 67 anchors as independent; p=1.6e-66 from a trivial ~7.5% shift. Neuron-permutation p=0.4771, δ=−0.064.
- **id 45 (entry 12) — OVERTURNED.** Wrong test: CMH with a single stratum (no real control) over non-independent nodes. Neuron-clustered permutation OR=0.53, p=0.6607.
- **id 61 (entry 15) — OVERTURNED.** Wrong test: per-edge MW/Welch on 1.1M non-independent edges referenced to a small merge set; δ CI crosses 0, neuron-permutation p=0.6469.
- **id 64 (entry 17) — WEAKENED.** Wrong test: per-edge MW on 1.1M edges (p=1.3e-197). Cliff's δ CI [−0.369,−0.031] still excludes 0 but the neuron-permutation p=0.096 is no longer significant on the origin.
- **id 21 / 27 — note:** both are *null* findings whose null survives the correct test (UPHELD as nulls), so they did not change in headline direction, but their generalization is now PARTIAL rather than DOES-NOT/PARTIAL on stronger footing — see per-entry bullets.

**Synthesis.** The corrections cleanly separate two classes of finding. (1) **Effect-driven local-morphology findings survive the right test:** split spatial clustering (id 37, δ≈0.49–0.60, perm p≤4e-4) is robustly UPHELD, and the two tortuosity → split-risk findings (id 59, id 73) are UPHELD with their true *small* effect sizes now exposed (Cliff's δ≈0.18–0.27, CIs excluding 0, perm p≤0.002) and generalizing. (2) **Findings whose headline rested on n-inflated p-values over non-independent edges/nodes collapse once the neuron is the unit of analysis:** branch-order → split risk (id 139), omit-near-merge clustering (id 36), split-near-merge clustering (id 61), extreme-Z omit excess (id 45), and the super-merge cable claim (id 24) all become non-significant (5 OVERTURNED), and split-near-branch-point (id 64) is WEAKENED to a marginal, effect-present-but-not-significant result. The two anisotropy *null* claims (id 21, 27) were robust to the test choice in their headline (still "no significant effect"), but the correct test exposes that their generalization is volume-specific (a real, oppositely-signed effect appears on 794495). Net: the proofreading-relevant *spatial-clustering of splits* and *tortuosity* signals hold up, while most *distance-to-anchor co-localization* and *global-axis / branch-order* claims were artifacts of treating millions of correlated edges as independent.
