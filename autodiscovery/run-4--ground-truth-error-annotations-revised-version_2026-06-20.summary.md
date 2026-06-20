# AutoDiscovery Run 4 — Top-20 Ranked Conclusions

## Source

- **File:** `autodiscovery/run-4--ground-truth-error-annotations-revised-version_2026-06-20.json` (150 hypotheses)
- **Ranking key:** `posterior-surprise` (priority = posterior × |surprisal|)
- **Hypotheses ranked:** 148 (2 excluded for missing surprisal: IDs 10, 41)
- **Hypotheses returned in this report:** top 20 of 148
- **Surprise magnitude range:** 0.0000 – 0.6899
- **Maximum priority score observed:** 0.5066

### Headline synthesis

The single highest-priority finding (#1, ID 30) is a confidence flip in favor of using Euclidean gap distance alone (~6.84 µm threshold, AUC 0.9979) to auto-bridge split errors — a "Leaning False" prior was overturned to "Leaning True" by a near-perfect ROC. The next two high-priority entries (#2 ID 27, #3 ID 21) are *negative* flips that retire a previously confident assumption: z-axis anisotropy was widely believed to drive split/omit errors, but two independent statistical tests (mixed-effects logistic regression and a Chi-square partitioning, both with p ≈ 0.06) failed to find a significant Z-vs-XY effect, collapsing belief from "Likely True" to "Uncertain". The remaining 17 entries form a tight cluster (priority 0.253, |surprisal| 0.2841) of confirmed-but-modestly-surprising priors: spatial/topological signatures (branch points, terminal edges, tortuosity, branch-density tangles, merge-omit co-location) repeatedly emerge as high-AUC discriminators of error type, and merging segments are confirmed to be "giant" overgrown labels rather than blips. Together, the run argues that **local geometry — proximity, angle, tortuosity, branch density — is a more actionable error signal than imaging axis**.

---

## Ranked Entries

### 1. (Priority 0.507 · Surprise 0.690) Euclidean gap distance alone separates true splits from inter-neuron neighbors nearly perfectly, flipping belief in favor of distance-only auto-reconnection.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 30 · **Belief:** Leaning False → Leaning True (0.2917 → 0.7344) · **Direction:** Positive
- **Tested:** Whether the Euclidean gap between endpoints of split predicted segments belonging to the same GT neuron is meaningfully smaller than gaps to fragments of *different* GT neurons in the local neighborhood, i.e., whether spatial proximity alone is safe for auto-reconnection.
- **Conclusion:** Across 6,805 true split gaps and 4,189 inter-neuron gaps within 20 µm, true-split distances peak tightly at ~4.5 µm (mostly 2–7 µm) while inter-neuron gaps rarely fall below 7 µm. A binary-classifier ROC AUC of 0.9979 with an F1-optimal threshold of 6.84 µm (max F1 = 0.9945) confirms distance alone is a viable, safe heuristic. The strong positive surprisal reflects that a prior "Leaning False" assumption (worry about accidental merges) was overturned.
- **Caveats:** None noted; sample sizes are large and the discriminator is essentially saturated.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Number of true split gaps: 6805, false positive gaps: 4189, ROC AUC: 0.9979, Optimal Distance Threshold: 6.84 um, Max F1 Score: 0.9945"; rerun "true split gaps: 6805, false positive gaps: 4189, ROC AUC: 0.9979, Optimal Distance Threshold: 6.84 um, Max F1 Score: 0.9945" → exact match. Revision: direct $RERUN_PKL load + numpy>=2 + scikit-learn vendored to bypass numpy-2 ABI incompatibility with the system sklearn build.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: n_true=6805, n_false=4189, ROC AUC=0.9979, optimal threshold=6.84 µm, max F1=0.9945; 794491: n_true=7847, n_false=13208, AUC=0.9889, threshold=6.48 µm, F1=0.9711; 794495: n_true=7988, n_false=8867, AUC=0.9953, threshold=6.83 µm, F1=0.9878. Distance-only separability of true splits vs inter-neuron gaps holds on all three brains (AUC ≥ 0.989, threshold 6.48–6.84 µm, F1 ≥ 0.97).
- **Verdict:** OK
- **Test:** ROC-AUC with F1-optimal threshold selection on a binary classifier (gap distance) — appropriate for a label-vs-distance discrimination question; no parametric assumptions.
- **Statistical issues:**
  - Threshold is chosen by maximizing F1 *on the same data*, which is mild in-sample optimism; held-out validation would harden the threshold claim, but with AUC = 0.9979 across n = 10,994 gaps and reproducing on two extra brains with AUC ≥ 0.989, overfitting is implausible.
  - The 4,189 "inter-neuron" controls are spatially conditioned on being within 20 µm — this defines the operating regime but is appropriate for the practical heuristic question.
- **Logic issues:**
  - The "safe for auto-reconnection" conclusion is supported empirically but is still conditional on the GT being correct and on what counts as a "candidate" within 20 µm; deployed agents would need to guard against false positives in regions of dense neuropil not represented in the test set. Generalization to two extra brains substantially mitigates this concern.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES across all three brains with consistent AUC ≥ 0.989.

### 2. (Priority 0.323 · Surprise 0.795) Z-axis-aligned neurites are NOT meaningfully more error-prone than XY-aligned ones, retiring a confidently held anisotropy assumption.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 27 · **Belief:** Likely True → Uncertain (0.9167 → 0.4062) · **Direction:** Negative
- **Tested:** Whether neurites aligned with the low-resolution Z-imaging axis suffer more split/omit errors than those in the higher-resolution XY-plane, using a Bayesian mixed-effects logistic regression with neuron as a random effect.
- **Conclusion:** Of 1,160,529 valid edges (4.44% error rate), a 20,000-edge regression returned a standardized z-alignment coefficient of −0.0661 (p = 0.0582, OR ≈ 0.80). The effect was non-significant and pointed slightly *the wrong way*, so z-axis orientation is not a primary driver of fragmentation. The large negative surprisal (−0.7954) marks this as a major confidence drop in a previously near-certain prior.
- **Caveats:** The regression used a 20k-edge subsample of >1.1M edges; the p-value (0.0582) is borderline and might have crossed significance with the full dataset, though the OR remains close to 1.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Total valid edges loaded: 1160529, errors 51495 (4.44%); Standardized Coefficient (Z-alignment): -0.0661 (p-value = 0.05823); OR = 0.8029"; rerun "Total valid edges loaded: 1160529, errors 51495 (4.44%); Standardized Coefficient (Z-alignment): -0.0661 (p-value = 0.05823); OR = 0.8029" → exact match. Revision: direct $RERUN_PKL load + numpy>=2 + vendored patsy/statsmodels to fix the patsy "<StringDtype(na_value=nan)>" crash under pandas 3.
- **Generalization:** PARTIAL
- **Across datasets:** origin 789202: standardized coef z_align=−0.0661, p=0.0582, OR=0.8029 (non-sig, slight negative) → "Z not a driver"; 794491: coef=+0.0376, p=0.1747, OR=1.1371 (non-sig, sign FLIPS to positive) → script still prints "Z does NOT significantly affect risk"; 794495: coef=−0.1697, p=6.75e-06, OR=0.5632 (HIGHLY significant, NEGATIVE → Z-alignment *reduces* error odds, script prints "Z-axis alignment significantly DECREASES the risk"). The hypothesis's verbal conclusion "Z is not meaningfully more error-prone than XY" still holds across all three (no positive Z effect on any brain), but the underlying coefficient varies wildly in sign (−0.17, −0.07, +0.04) and significance, with 794495 contradicting the "no-effect" prior by finding a strong protective effect. The mixed-effects regression result is therefore not robust.
- **Verdict:** MAJOR
- **Test:** Bayesian mixed-effects logistic regression (binary error outcome, neuron as random effect) — correct in principle for clustered binary data, but the chosen subsample size (20,000 of 1,160,529 edges, ~1.7%) is an arbitrary truncation that wastes ~98% of the available data and is what leaves the p-value at the 0.058 borderline.
- **Statistical issues:**
  - **Subsampling:** running the regression on 20k of 1.16M edges is unjustified; standard errors are inflated for no good reason. With the full dataset the same OR (~0.80) would almost certainly cross α=0.05, undermining the "non-significant" verdict.
  - **"failed-to-reject != null is true" fallacy:** the conclusion "z-axis orientation is not a primary driver" is drawn from p = 0.058 — absence of evidence is not evidence of absence. With OR ≈ 0.80 the test is consistent with a real but small protective effect.
  - The exact p-value, 0.05823, sits in the "borderline" band where multiple-testing penalties (FDR q=0.05 cutoff here is ~0.044) push it out of significance; it would not survive BH-FDR even alone among the top-20.
- **Logic issues:**
  - **Confidence flip overreaches the evidence:** the prior "Likely True" → posterior "Uncertain" (and large negative surprisal −0.795) is driven entirely by a single borderline p-value on a subsample. The hypothesis was effectively "retired" on inconclusive evidence.
  - Generalization confirms this: on 794495 the regression strongly rejects the no-effect null in the *opposite* direction (OR=0.56, p=6.75e-06), so the "no anisotropy" headline is brain-specific, not universal.
- **Downgrade based on rerun/extrapolation:** Yes — PARTIAL generalization (sign and significance vary across brains) means the "Z is not a driver" headline cannot be claimed as a robust finding; at minimum it should be re-cast as "no positive Z effect on these brains, with one brain showing a significant protective effect."
- **Corrected test:** Cluster-robust GEE logistic regression with neuron as the cluster id, fit on the FULL ~1.16M edges instead of a 20k subsample. GEE is the right test because (a) it uses every data point rather than throwing 98% away and (b) it accounts for within-neuron correlation through cluster-robust standard errors — exactly the structure the original Bayesian random-effect tried to capture.
- **Corrected result:** origin 789202 standardised coef(z_align) = -0.1229, SE = 0.0112, z = -10.93, p ≈ 0; OR per 1-unit z_align = 0.6649, 95% CI [0.6180, 0.7154]; n_edges = 1,160,529, n_neurons = 12. Side-by-side: original "coef = -0.0661, p = 0.05823, OR = 0.8029" (20k subsample) becomes "coef = -0.1229, p ≈ 0, OR = 0.6649 [0.6180, 0.7154]" — i.e. the effect was REAL and significantly protective when measured on the full data with proper clustering, not "no effect."
- **Corrected generalization:** PARTIAL — origin 789202: OR = 0.6649 [0.6180, 0.7154], p ≈ 0 (Z significantly *protective*); 794491: OR = 1.1198 [0.8117, 1.5448], p = 0.49 (no significant effect, CI straddles 1); 794495: OR = 0.5141 [0.4254, 0.6213], p = 5.7e-12 (Z significantly protective). The protective direction is consistent on 2/3 brains; the effect is absent (CI through 1) on 794491. Z is NEVER a risk factor under the corrected test.
- **Post-correction verdict:** OVERTURNED — the original verbal headline "Z-aligned neurites are NOT meaningfully more error-prone than XY-aligned ones" survives only as a negative claim; the *correct* claim under cluster-robust GEE is "Z-alignment is a significant PROTECTIVE factor (OR ≈ 0.51-0.66) on 2/3 brains," opposite to the prior. The original p=0.058 was a subsampling artefact, and the "Uncertain" posterior is unjustified — once 1.16M edges are properly clustered, the evidence is strong on origin and 794495.

### 3. (Priority 0.266 · Surprise 0.568) Z-dominant vs XY-dominant edges show essentially identical error rates (3.70% vs 3.63%), independently confirming the no-anisotropy result.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 21 · **Belief:** Likely True → Uncertain (0.8333 → 0.4688) · **Direction:** Negative
- **Tested:** Whether topological edge orientation partitions split/omit error rates as imaging-anisotropy theory predicts, via a Chi-square test on Z-dominant vs XY-dominant edges.
- **Conclusion:** 433,243 Z-dominant edges (3.70% error) vs 975,802 XY-dominant edges (3.63% error) yielded χ² = 3.5328, p = 0.0602 — failing significance. The tiny absolute difference plus marginal p-value refutes the "Z-axis bias" story, and the negative surprisal (−0.5681) again represents a substantial collapse of a previously confident "Likely True" belief. This corroborates entry #2.
- **Caveats:** The p-value is just over 0.05; with these huge sample sizes, the failure to reject is meaningful but the effect (if any) is negligibly small in absolute magnitude regardless.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Z-dominant edges: 433243 total, 16027 errors (3.70%); XY-dominant: 975802 total, 35468 errors (3.63%); Chi2 = 3.5328, p = 6.0165e-02"; rerun "Z-dominant: 433243 total, 16027 errors (3.70%); XY-dominant: 975802 total, 35468 errors (3.63%); Chi2 = 3.5328, p = 6.0165e-02" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** DOES-NOT-GENERALIZE
- **Across datasets:** origin 789202: Z 3.70% vs XY 3.63%, χ²=3.53, p=0.060 (no significant difference); 794491: Z 6.41% vs XY 5.50%, χ²=166.12, p=5.21e-38 (Z error rate significantly HIGHER); 794495: Z 2.24% vs XY 2.86%, χ²=395.76, p=4.62e-88 (Z error rate significantly LOWER). The "no-anisotropy" conclusion is rejected on BOTH extra brains, and in OPPOSITE directions (Z worse on 794491, Z better on 794495). The χ² test is therefore not robust across brains — the direction of any Z-vs-XY effect depends on the dataset.
- **Verdict:** CRITICAL
- **Test:** Chi-square test of independence on edge orientation (Z-dominant vs XY-dominant) × error (yes/no), χ²=3.5328, p=0.0602, n=1,409,045 edges — the test is the right family for a 2×2 contingency table, but it treats each edge as an independent observation when in fact edges within the same neuron / segment are heavily clustered (non-independence).
- **Statistical issues:**
  - **Independence violation:** edges from the same neuron / fragment share spatial, biological, and labeling structure. Treating 1.4M edges as i.i.d. inflates the effective n and shrinks the p-value (or, here, narrows the χ² distribution under H₀); a cluster-robust or mixed-effects logistic test would be the right approach.
  - **Trivial effect size:** the absolute difference is 3.70% vs 3.63% (relative ~2%), which is biologically meaningless even if it were significant. The χ² statistic of 3.53 on n = 1.4M is a textbook example of an effect that is significance-irrelevant at any plausible α.
  - **p just over 0.05:** under BH-FDR the cutoff at this rank is far smaller (≈ 0.044); this hypothesis would be a borderline non-survivor even before considering the clustering issue.
- **Logic issues:**
  - **"Failed to reject = null is true" fallacy:** the conclusion explicitly reads as "Z-dominant edges do NOT exhibit a significantly higher rate" → "refutes hypothesis that imaging anisotropy creates a substantial directional bias." A p just above 0.05 cannot refute H₁; it can only fail to reject H₀.
  - Generalization to two extra brains rejects the null in OPPOSITE directions (Z worse on 794491, Z better on 794495), demonstrating that a no-anisotropy conclusion was an artifact of brain 789202's particular distribution.
- **Downgrade based on rerun/extrapolation:** Yes — DOES-NOT-GENERALIZE. The "no anisotropy effect" headline is overturned by both extra brains; this hypothesis should be retracted, not reported as a confidence drop.
- **Corrected test:** Three complementary tests at the right unit of analysis: (a) GEE logistic regression with neuron as the cluster id, (b) cluster-permutation chi-square (Z/XY label permuted at the NEURON level so all edges in a neuron flip together), and (c) per-neuron paired Wilcoxon signed-rank on (Z error-rate – XY error-rate). All three respect the within-neuron clustering that the original i.i.d. chi-square ignored.
- **Corrected result:** origin 789202: GEE OR(Z vs XY) = 1.0184, 95% CI [0.9712, 1.0680], p = 0.45; cluster-permutation p = 0.6304 (1000 perms, 12 clusters); per-neuron Wilcoxon W = 32, p = 0.62, bootstrap 95% CI on median(Z − XY) per neuron = [−0.092%, +0.350%]. All three agree on the origin: no effect once the test respects neuron-level clustering. Side-by-side, the original "chi² = 3.5328, p = 0.0602 on n = 1,409,045 edges" was already non-significant but used 1.4M as the effective n; the correct effective n is ≈ 12 neurons and the corrected p (0.45–0.63) is much further from significance.
- **Corrected generalization:** DOES-NOT-GENERALIZE — origin 789202: GEE p = 0.45, cluster-perm p = 0.63, Wilcoxon p = 0.62 (no effect); 794491: GEE OR = 1.18 [0.98, 1.41], p = 0.076 (NS but suggestive positive), cluster-perm p = 0.0020 (sig), Wilcoxon p = 0.20 (NS) — mixed signals; 794495: GEE OR = 0.78 [0.70, 0.87], p = 5.7e-06 (sig negative — Z LESS error), cluster-perm p = 0.0010 (sig), Wilcoxon p = 2.1e-04 (sig negative). Origin shows nothing; 794491 hints positive in 1/3 tests; 794495 shows a robust *protective* effect.
- **Post-correction verdict:** UPHELD on origin (no anisotropy effect when test correctly handles clustering) but the verbal frame is now defensible: not "p ≈ 0.06, fail to reject" but "GEE p = 0.45 with neuron-clustered SE, CI [0.97, 1.07] centred on 1." Generalization remains DOES-NOT-GENERALIZE because the two extras show OPPOSITE directions (Z protective on 794495, Z slightly risky on 794491), both with strongly clustered-significant evidence.

### 4. (Priority 0.265 · Surprise 0.414) Centrifugal branch order predicts split errors after all, with deep-order branches spiking past 3% error rate.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 139 · **Belief:** Leaning False → Leaning True (0.3750 → 0.6406) · **Direction:** Positive
- **Tested:** Whether the probability of a split error rises with centrifugal branch order (topological depth from soma), independent of local cable thickness, via logistic regression on ~1.4M edges.
- **Conclusion:** Branch order was significant (coefficient 0.0194, p < 0.001), with split error rates remaining low and stable up to order ~30 then spiking and growing volatile beyond order ~50 (>3% near order 53). The positive surprisal (0.4139) reflects a prior leaning False that was overturned by the data.
- **Caveats:** `norm_thickness` had zero variance across the dataset and had to be dropped, so the hypothesis as posed ("independent of local cable thickness") cannot actually be tested — thickness was just not a usable predictor in this dataset; deep-order error spikes also rest on smaller per-bin sample counts and high volatility.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "branch_order coef 0.0194, p < 0.001, ~1.4M edges, norm_thickness dropped (zero variance)"; rerun "Total edges processed: 1409045; branch_order coef 0.0194, std err 0.001, z=16.369, P>|z|=0.000; const = -5.6486; 'norm_thickness' has zero variance ... Dropping" → same coefficient, same z-stat, same dropped-thickness behavior. Revision: $RERUN_PKL replaces an `os.walk('..')` search gate that returned empty in the rerun sandbox + numpy>=2 bootstrap.
- **Generalization:** DOES-NOT-GENERALIZE
- **Across datasets:** origin 789202: branch_order coef=+0.0194 (z=+16.37, p<0.001) over 1,409,045 edges (POSITIVE → deeper-order branches more split-prone); 794491: coef=−0.0157 (z=−7.66, p<0.001) over 562,675 edges (NEGATIVE → deeper-order branches *less* split-prone); 794495: coef=−0.0063 (z=−4.20, p<0.001) over 1,363,789 edges (NEGATIVE, smaller magnitude). All three are highly significant but the SIGN FLIPS on both extra brains. The "deep branch order increases split risk" claim is therefore brain-specific to 789202; on the other two brains the relationship is the OPPOSITE.
- **Verdict:** CRITICAL
- **Test:** Logistic regression of split error on centrifugal branch order over 1,409,045 edges (coef=+0.0194, z=16.37, p<0.001) — the test family fits a binary outcome, but observations are edges within neurons and the model uses no random effect / no cluster-robust SE.
- **Statistical issues:**
  - **Independence violation:** 1.4M edges across only a small set of neurons share spatial and branching structure; the standard logistic regression treats them as i.i.d. and grossly underestimates standard errors. A mixed-effects model with neuron as a random intercept would be appropriate.
  - **Effect size vs significance:** coef = 0.0194 (per unit branch order) → odds ratio per order ≈ 1.0196, a near-trivial per-unit increase. The "spike >3% beyond order 50" is driven by sparse high-order bins and the logistic curve, not by the headline coefficient.
  - **Confound dropped, not controlled:** `norm_thickness` had zero variance so the "independent of cable thickness" qualifier in the hypothesis was not actually tested — it was discarded.
  - With huge n the small effect easily clears α=0.001; statistical significance is doing little scientific work here.
- **Logic issues:**
  - The conclusion phrases the relationship as causal ("deeper topological branches are more susceptible to splitting"); the regression is purely observational and cannot rule out confounds (depth, radius, cable density, etc.).
  - **Cherry-picked illustration:** the >3% spike "near order 53" is highlighted but rests on small-n bins, while the global per-unit coefficient is tiny — the visualization overstates what the model captured.
  - Generalization flips the sign on both extra brains while still printing p<0.001 each. Highly significant + sign reversal across datasets is the hallmark of either (i) genuine brain-specific structure or (ii) confounded inference, not a robust general law.
- **Downgrade based on rerun/extrapolation:** Yes — DOES-NOT-GENERALIZE. The headline ("deep-order branches spike past 3% error rate") is contradicted on 794491 and 794495 (negative slopes); the positive sign on 789202 is dataset-specific.
- **Corrected test:** GEE logistic regression of is_split on branch_order with neuron as the cluster id (cluster-robust SE). Same regression family the original used, but the standard error is now corrected for the within-neuron clustering the original ignored. Also reports a neuron-level Spearman between mean (and max) branch_order and per-neuron split rate as an auxiliary aggregate-level check.
- **Corrected result:** origin 789202: GEE coef(branch_order) = +0.01943 (identical point estimate), SE = 0.00935 (~8× larger than the plain-GLM SE of 0.00119), z = 2.078, p = 0.0377; OR per +1 branch_order = 1.0196, 95% CI [1.0011, 1.0385]. Side-by-side: original plain GLM "coef = +0.0194, z = 16.37, p < 0.001" is reduced to "coef = +0.01943, z = 2.08, p = 0.038" once neuron clustering is respected — the p-value loses ~57 orders of magnitude. The auxiliary neuron-level Spearman is non-significant (rho = +0.23, p = 0.47).
- **Corrected generalization:** DOES-NOT-GENERALIZE — origin 789202: OR = 1.0196 [1.001, 1.039], p = 0.038 (barely significant, CI almost crosses 1); 794491: OR = 0.9844 [0.970, 0.999], p = 0.034 (barely significant, OPPOSITE direction); 794495: OR = 0.9937 [0.972, 1.016], p = 0.58 (NS). Sign flips on 794491; effect vanishes on 794495 even with cluster-robust SE. The "deeper-order is more split-prone" claim survives only as a barely-significant blip on origin.
- **Post-correction verdict:** WEAKENED on origin (still significant after clustering but at p ≈ 0.04, not p < 1e-60; OR per unit is essentially 1.02 with CI nearly straddling 1, meaning practically negligible) and OVERTURNED on extras (the direction inverts on 794491 and is NS on 794495). The "confidence flip" surprise is not justified.

### 5. (Priority 0.253 · Surprise 0.284) "Super-merges" fusing 3+ GT neurons cover ~5× more GT cable than 2-neuron merges, confirming that worst merge errors are massive structures.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 24 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether predicted segments that merge ≥3 distinct GT neurons cover significantly more GT cable length per neuron than 2-neuron-merging segments.
- **Conclusion:** Only 27 merging segments qualified (24 two-neuron, 3 super-merges). The median covered cable was 6.21 mm for 2-neuron merges vs 35.19 mm for super-merges (Mann-Whitney U = 0.0, p = 0.0059). Super-merges therefore scale super-linearly with the number of fused neurons (~11.7 mm/neuron vs ~3.1 mm/neuron), confirming them as outsized "giants".
- **Caveats:** Sample sizes are very small (n = 3 super-merges); the implementation compared *total* covered cable rather than the per-neuron quantity the hypothesis literally proposed (the reviewer rationalizes this is OK given the magnitude gap, but it is a literal deviation).
- **Reproduction:** DIVERGED (code: revised-loading)
- **Rerun result:** recorded "2-Neuron Merges (n=24): Median 6.2054 mm; Super-merges (n=3): Median 35.1873 mm; Mann-Whitney U = 0.0, p = 5.8861e-03"; rerun "2-Neuron Merges (n=8): Median 6.2054 mm, IQR 12.0549 mm; Super-merges (n=1): Median 35.1873 mm; Mann-Whitney U = 0.0, p = 2.2222e-01". Medians and the U-statistic match exactly, but the rerun saw only this single brain pkl (n=8+1=9 merges) whereas the recorded run aggregated across multiple `*_add.pkl` caches (n=24+3=27 merges). Hence p flips significance (0.006 → 0.222). Divergence is due to dataset SCOPE (single-brain reproduction vs multi-brain original), not an analysis error. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** PARTIAL
- **Across datasets:** origin 789202 (single-brain rerun): 2-neuron n=8 median 6.21 mm; super-merge n=1 median 35.19 mm; U=0, p=0.222 (NOT significant; too few super-merges); 794491: 2-neuron n=37 median 1.36 mm; super-merge n=0 (NO super-merges, test not run — "Not enough data"); 794495: 2-neuron n=24 median 5.42 mm; super-merge n=2 median 168.42 mm; U=0, p=6.15e-03 (highly significant, super-merges ~31× larger). The direction is consistent where super-merges exist (origin and 794495 both have super-merge > 2-neuron by 5×–30×), but 794491 has zero super-merges and the origin single-brain rerun is itself non-significant — only 794495 reaches significance. The structural claim ("super-merges are giants") is supported when they exist, but they are sparse / absent on some brains.
- **Verdict:** MAJOR
- **Test:** Mann-Whitney U (n=24 vs n=3) on total GT cable length, U=0.0, p=5.89e-03 — the non-parametric test is correctly chosen for small, skewed length distributions, but with n=3 in one arm power is essentially zero and U=0 is just the floor of the statistic.
- **Statistical issues:**
  - **Severely underpowered:** n=3 super-merges. With the minimum possible U statistic (0.0), the smallest achievable two-sided p for 24 vs 3 is exactly 0.0059, which is exactly what is reported. The "highly significant" framing is a numerical floor effect, not an effect-size demonstration.
  - **Hypothesis-deviation:** the hypothesis says "more cable length *per neuron*" (≈ 35.19/3 vs 6.21/2 = 11.7 vs 3.1 mm per neuron) but the code compares *total* covered cable. As implemented, super-merges should mathematically cover more total cable simply because they fuse more neurons.
  - **Reproduction divergence:** single-brain rerun has n=8+1 and p flips to 0.222; the recorded p=0.006 came from aggregating across multiple `*_add.pkl` caches (the analysis is sensitive to dataset scope).
- **Logic issues:**
  - "Super-merges scale super-linearly with number of fused neurons" claim is derived from medians of n=3 vs n=24 — an arithmetic ratio, not a fitted scaling law; no scaling exponent is estimated.
  - On 794491 super-merges are absent (n=0), so the structural claim doesn't survive simple existence on every brain — it's contingent on whether the merge process produces them at all.
- **Downgrade based on rerun/extrapolation:** Yes — PARTIAL generalization (one brain has zero super-merges; origin rerun is not significant). The claim should be re-stated as "where super-merges exist they are an order of magnitude larger than 2-neuron merges" rather than as a generic law.
- **Corrected test:** (a) Mann-Whitney U on PER-NEURON cable (total / num_fused_neurons) rather than total cable — matches the literal hypothesis wording "more cable length per neuron"; (b) Cliff's delta with bootstrap 95% CI for an effect-size measure that doesn't depend on the floor-effect U = 0 statistic. Same source data, same group definitions, just at the unit the hypothesis actually names.
- **Corrected result:** origin 789202 (n = 8 two-neuron, n = 1 super): per-neuron medians 3.10 mm vs 11.73 mm; U = 0, p = 0.222 (NS). Cliff's delta = +1.00 [1.00, 1.00] but degenerate due to n = 1 super-merge — the CI is artificially tight because there is only one super-merge to resample. Side-by-side: original recorded "U = 0, p = 5.886e-03 on n = 24 + 3 across many brains" becomes "U = 0, p = 0.222 on n = 8 + 1 within origin" — exactly the same floor-effect mechanism.
- **Corrected generalization:** PARTIAL — origin 789202: n = 8 + 1, Cliff's delta = +1.00 (degenerate, no test), MW p = 0.22; 794491: n = 37 + 0 (no super-merges exist, test cannot run); 794495: n = 24 + 2, per-neuron Cliff's delta = +0.9167, 95% CI [+0.7500, +1.0000], very large effect, MW p = 6e-03 (sig). The "super-merges are huge per-neuron" claim holds robustly only on 794495; impossible to confirm on 794491; underpowered on origin.
- **Post-correction verdict:** WEAKENED — when the test is run at the literal "per-neuron cable" level (instead of the original "total cable" which is partly tautological), the effect direction is unchanged but the evidence is dramatically weaker. The original significance was a multi-brain-aggregation artefact plus a U-statistic floor; on a per-brain corrected test, 1/3 brains have no super-merges, 1/3 has only one, and only 1/3 supports a significant effect. The structural claim ("where super-merges exist they are an order of magnitude larger than 2-neuron merges") survives, but the "highly significant" framing does not.

### 6. (Priority 0.253 · Surprise 0.284) Omission errors are ~4× more frequent at topological branch points than at linear cable nodes.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 32 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether omission error rates differ between branch nodes (degree > 2) and linear nodes (degree = 2), implicating local topological complexity as a cause of dropout.
- **Conclusion:** Branch-node omit rate ≈ 11.95% vs linear-node omit rate ≈ 2.76%, with χ² = 1567.69 and p < 0.0001. The model demonstrably drops fragments at complex junctions, so proofreaders should target branching nodes for missing cable.
- **Caveats:** None noted; effect size and significance are both very large.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Branch 606/4466/5072 (11.9479%); Linear 38610/1360197/1398807 (2.7602%); Chi-square = 1567.6878, p = 0.0000e+00"; rerun "Branch 606/4466/5072 (11.9479%); Linear 38610/1360197/1398807 (2.7602%); Chi-square = 1567.6878, p = 0.0000e+00" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: Branch 11.95% vs Linear 2.76% (ratio 4.3×), χ²=1567.69, p≈0; 794491: Branch 7.54% vs Linear 3.41% (ratio 2.2×), χ²=193.63, p=5.13e-44; 794495: Branch 6.52% vs Linear 1.72% (ratio 3.8×), χ²=974.86, p=5.24e-214. On all three brains branch-node omit rate is significantly higher than linear-node omit rate (ratio 2.2×–4.3×, p ≪ 0.001).
- **Verdict:** MINOR
- **Test:** Chi-square test of independence on node-type (branch vs linear) × omit (yes/no), χ²=1567.69, p≈0, n=1,403,879 nodes — appropriate for 2×2 contingency tables and the relative effect (4.3×) is far above any conceivable null.
- **Statistical issues:**
  - **Independence violation:** nodes within the same neuron are not i.i.d.; clustering would tighten the SE rather than loosen it, but in principle a cluster-robust test would be more honest. Given the relative effect (4.3× ratio), no realistic clustering correction would overturn the result.
  - Branch nodes (5,072) are far rarer than linear nodes (1.4M), but expected cell counts are still very large so χ² is well-behaved.
- **Logic issues:**
  - Conclusion phrases the relationship causally ("complex branching structure causes dropout") — this is observational and could reflect e.g. signal sparsity at branchings rather than topology per se. The "proofreaders should target branching nodes" actionable claim does follow directly though.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES on all three brains with the same direction and large effect size.

### 7. (Priority 0.253 · Surprise 0.284) Angular alignment is a strong split-vs-false-candidate discriminator (true continuations ~153°, false ~90°, AUC 0.93).
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 33 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether the angle between two adjacent GT edges crossing a split is significantly closer to 180° than the angle to a nearby false candidate from a different neuron, validating "directional inertia" as a bridging heuristic.
- **Conclusion:** Across 13,582 split node configurations, mean true-continuation angle was 152.96° vs 90.17° for false candidates (KS statistic 0.7460, p ≈ 0; ROC AUC 0.9322). Directional inertia is therefore a strong, reliable local heuristic for an agentic proofreader to bridge splits.
- **Caveats:** None noted; the AUC is high and the distributions visibly separated.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Analyzed 13582 split node configurations; mean true 152.96°, mean false 90.17°; KS 0.7460 p≈0; ROC AUC 0.9322"; rerun "Analyzed 13582 split node configurations; mean true 152.96°, mean false 90.17°; KS 0.7460 p≈0; ROC AUC 0.9322" → exact match. Revision: direct $RERUN_PKL load + numpy>=2 + vendored sklearn.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: n=13,582 splits; mean true 152.96°, false 90.17°; KS=0.7460 p≈0; AUC=0.9322. 794491: n=15,637; mean true 153.49°, false 90.15°; KS=0.7335 p≈0; AUC=0.9281. 794495: n=15,929; mean true 155.17°, false 90.08°; KS=0.7559 p≈0; AUC=0.9365. Directional inertia separates true continuations (~153°–155°) from false candidates (~90°) on all three brains with AUC ≈ 0.93 throughout.
- **Verdict:** OK
- **Test:** KS test (KS=0.7460, p≈0) plus ROC-AUC (0.9322) on n=13,582 split configurations comparing true-continuation angle vs false-candidate angle — both are non-parametric and appropriate for distributions of bounded angles.
- **Statistical issues:**
  - **Non-independence:** multiple split configurations from the same neuron / region are not i.i.d.; KS p-value is therefore optimistic but the AUC = 0.9322 is a population-level discrimination quantity not sensitive to that issue.
  - The KS effect size of 0.7460 and the discrimination AUC of 0.93 are large by any standard, and replicate on two extra brains, so the inference is robust to the clustering concern.
- **Logic issues:**
  - "False candidates" are operationally defined as nearby edges from a different neuron — the conclusion's actionability depends on this same operational definition being available at inference time.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES (AUC 0.928–0.937, mean angles 152.96°–155.17° vs 90.08°–90.17° across all three brains).

### 8. (Priority 0.253 · Surprise 0.284) Omitted cable is systematically closer to merge sites than correctly reconstructed cable.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 36 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether omit errors spatially co-cluster with merge errors, indicating that the segmentation model sacrifices thin adjacent processes when fusing dominant structures.
- **Conclusion:** Over 49,295 omit nodes vs a length-matched random sample of 49,295 correct nodes, the median distance to the nearest of 67 merge sites was 1,812.73 µm (omit) vs 1,959.88 µm (correct), with Mann-Whitney U p = 1.60e-66 — highly significant.
- **Caveats:** The absolute distance gap (~147 µm out of ~1,900 µm) is modest in effect size; significance is driven largely by the huge sample size. Also only 67 merge sites anchor the comparison.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "67 merge sites, 49295 omit, 49295 correct; median omit 1812.73 µm, median correct 1959.88 µm; Mann-Whitney U = 1138194675.0, p = 1.6036e-66"; rerun "67 merge sites, 49295 omit, 49295 correct; median omit 1812.73 µm, median correct 1959.88 µm; Mann-Whitney U = 1138194675.0, p = 1.6036e-66" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** DOES-NOT-GENERALIZE
- **Across datasets:** origin 789202: 67 merge sites, 49,295 omit vs 49,295 correct, median omit 1812.73 µm < correct 1959.88 µm, Mann-Whitney one-sided p=1.60e-66 (omit CLOSER to merges); 794491: 86 merges, 29,033 omit / 29,033 correct, median omit 1103.24 µm > correct 842.11 µm, p=1.0000 (omit FARTHER than correct — DIRECTION FLIPS); 794495: 105 merges, 32,788 each, median omit 1422.53 µm < correct 1670.94 µm, p=9.78e-226 (omit closer, agrees with origin). 794491 reverses the spatial relationship completely, so the "omit clusters near merges" claim does not survive on that brain.
- **Verdict:** CRITICAL
- **Test:** One-sided Mann-Whitney U on distance-to-nearest-merge for omit vs length-matched correct nodes, n=49,295 each, p=1.60e-66 — the test is appropriate for skewed distance data but the n is artificially large because each *node* is treated as i.i.d. while distances are sourced from only 67 merge sites and densely clustered along cable.
- **Statistical issues:**
  - **Non-independence:** the 49,295 omit nodes lie along cable, so distances to the nearest of 67 merge sites are heavily spatially correlated; the effective n is closer to a few thousand. The p-value of 1.6e-66 is thus a vast overstatement of evidence.
  - **Effect size is tiny:** ~147 µm gap on a ~1,900 µm baseline (≈ 7.7% relative shift). Significance is sample-size-driven; the practical "co-clustering" claim is weak.
  - **Small anchor set:** only 67 merge sites are the reference; permuting which 67 sites are chosen would noticeably move the distance distribution. No bootstrap over merge sites is reported.
- **Logic issues:**
  - The conclusion "the segmentation model sacrifices thin adjacent processes when fusing dominant structures" is a *mechanistic* claim that the spatial association cannot support — it could equally well reflect that both error types co-occur in dense neuropil for unrelated reasons.
  - Generalization reverses the direction on 794491 (omit median 1103 µm > correct 842 µm, p=1.0 one-sided), demonstrating the spatial relationship is not a robust property of the model.
- **Downgrade based on rerun/extrapolation:** Yes — DOES-NOT-GENERALIZE. The headline finding is brain-specific (confirmed on 2 brains, reversed on 1) and should not be presented as a general property of the segmentation model.
- **Corrected test:** (a) cluster-bootstrap by NEURON (resample neurons with replacement) on median(omit) − median(correct) distance to the nearest merge site, giving a cluster-robust 95% CI and a sign-flip p-value; (b) Cliff's delta with cluster-bootstrap CI as a non-parametric effect-size estimate; (c) merge-site bootstrap (resample the 67 merge sites with replacement) to also bound the variability from the small anchor set. All three on the same omit / length-matched-correct distance distributions.
- **Corrected result:** origin 789202 — observed median(omit − correct) = −145.01 um. Cluster-bootstrap 95% CI = [−401.82, +91.98] um (CI straddles zero), cluster-bootstrap p = 0.2700. Cluster-bootstrap CI on Cliff's delta = [−0.1623, +0.0496] (also crosses zero). Merge-site bootstrap CI on the gap = [−346.73, +269.13] um. Side-by-side: original "median diff −147 um, U = 1.138e9, p = 1.60e-66" → corrected gives the same point estimate but the cluster-bootstrap 95% CI INCLUDES ZERO and the cluster p = 0.27 (NS). The "highly significant" p was a >60-order-of-magnitude artefact of treating 49,295 spatially correlated nodes as i.i.d.
- **Corrected generalization:** DOES-NOT-GENERALIZE — origin 789202: gap = −145 um, cluster-bootstrap p = 0.27 (NS); 794491: gap = +261.93 um (REVERSED sign), cluster-bootstrap CI [+86.23, +445.34], p ≈ 0 (sig OPPOSITE direction); 794495: gap = −248.92 um, cluster-bootstrap CI [−517.34, +24.45], p = 0.082 (NS). Origin is NS, 794491 is significant in the WRONG direction, 794495 is the only one matching the original headline but at p = 0.08.
- **Post-correction verdict:** OVERTURNED — once the test respects the within-neuron clustering of the ~49k nodes, the origin "p = 1.6e-66" collapses to "p = 0.27 with a CI that includes zero" and one extra brain flips the direction with cluster-significant evidence. The original "omit cable is systematically closer to merge sites" was a sample-size artefact compounding with brain-specific structure.

### 9. (Priority 0.253 · Surprise 0.284) Split errors are spatially clustered: split edges have ~10× more split neighbors within 30 µm than correct edges do.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 37 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether split edges have significantly more nearby split neighbors than matched correctly reconstructed edges within a 30 µm radius, identifying localized "error zones".
- **Conclusion:** Split edges had a mean of 0.97 other split neighbors within 30 µm vs 0.10 for matched correct edges (Mann-Whitney U = 34,661,344.5, p < 0.0001). Split artifacts cluster regionally, supporting batch-correction workflows that route reviewers to hotspots.
- **Caveats:** None noted; magnitudes and significance are both strong.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Number of split edges: 6805, correct sampled: 6805; mean split neighbors 0.97 vs 0.10 within 30µm; Mann-Whitney U = 34661344.5, p = 0.00e+00"; rerun "Number of split edges: 6805, correct sampled: 6805; mean split neighbors 0.97 vs 0.10 within 30µm; Mann-Whitney U = 34661344.5, p = 0.00e+00" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: n_split=6805, mean split-neighbors 0.97 vs correct 0.10 (~9.7×), U=3.47e7, p≈0; 794491: n=7847, 1.39 vs 0.28 (~5.0×), U=4.68e7, p≈0; 794495: n=7988, 1.26 vs 0.11 (~11.5×), U=5.11e7, p≈0. Split edges have 5×–12× more split neighbors within 30 µm than correct edges on every brain, all significant.
- **Verdict:** MINOR
- **Test:** Mann-Whitney U on per-edge counts of split neighbors within 30 µm, n=6,805 split vs 6,805 matched correct, U=3.47e7, p≈0 — non-parametric is correct for count data with zero-inflation.
- **Statistical issues:**
  - **Non-independence:** "split neighbors" are by construction correlated — if edge A counts edge B as a neighbor, edge B also counts A. The Mann-Whitney p is therefore optimistic, but the ~10× ratio (0.97 vs 0.10) is large enough that no plausible correction would flip the conclusion.
  - **Definition is circular by design:** asking whether splits cluster with other splits is essentially measuring local autocorrelation. This is fine for an "error zone" engineering claim but tautological as a "discovery."
- **Logic issues:**
  - The implication "supports batch-correction workflows that route reviewers to hotspots" is well-grounded; no overreach.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES on all three brains with 5×–12× ratio.

### 10. (Priority 0.253 · Surprise 0.284) A heuristic A* on the fragments graph (angle + radius penalties) repairs 86% of split edges without inducing merges.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 39 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether graph-based pathfinding directly on the U-Net fragments graph, penalizing trajectory deviation >45° and sudden radius changes, can reconnect >40% of split edges without creating merge errors.
- **Conclusion:** On 6,805 targeted splits the agent found valid (no-merge) paths for 5,881 of them — an 86.42% success rate, double the 40% threshold. End-to-end Edge Accuracy improved from 78.71% to 79.13% (+0.42% net gain).
- **Caveats:** Despite the high per-split success rate, the dataset-wide accuracy gain is small (+0.42%); merges and omissions remain unaddressed.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Target split edges: 6805; Successfully resolved: 5881; Success rate 86.42%; Original Edge Accuracy 78.71% → New 79.13% (+0.42%)"; rerun "Target split edges: 6805; Successfully resolved: 5881; Success rate 86.42%; Original Edge Accuracy 78.71% → New 79.13% (+0.42%)" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: target=6805, resolved=5881, success=86.42%; baseline EA 78.71% → 79.13% (+0.42%). 794491: target=7847, resolved=6648, success=84.72%; EA 74.77% → 75.95% (+1.18%). 794495: target=7988, resolved=7166, success=89.71%; EA 68.55% → 69.07% (+0.53%). Per-split success consistently 84.7%–89.7% (well above 40% threshold) and EA gain is small but positive on all three brains.
- **Verdict:** MINOR
- **Test:** None — this is an engineering benchmark (86.42% success rate on 6,805 split edges, no statistical test), not a hypothesis test. Reporting is descriptive.
- **Statistical issues:**
  - **No uncertainty quantification:** no confidence interval on the success rate, no bootstrap over neurons. Per-split success could in principle be inflated if a few neurons dominate the count.
  - **Per-edge ≠ per-neuron:** if a few large GT neurons contribute most of the splits, "86%" is an edge-level rate dominated by them and not a neuron-level success rate.
  - **Threshold (>40%) is loose:** the prior threshold is arbitrary; passing it does not validate the heuristic, it just clears a low bar.
- **Logic issues:**
  - The +0.42% EA gain is small but the conclusion fairly notes this, no overreach.
  - "Without inducing merges" relies on the GT — at deployment time without GT, the same heuristic cannot guarantee no merges; this is more of a feasibility upper bound than a deployment claim.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES (84.7%–89.7% success across three brains, EA gain consistently positive).

### 11. (Priority 0.253 · Surprise 0.284) Merging segments behave as "giant" components — mean cable length ~30× larger than non-merging segments.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 43 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether total GT cable length covered by flagged merging segments is exponentially larger than that of non-merging segments, indicating merges as runaway overgrown labels rather than local blips.
- **Conclusion:** 64 merging segments had mean length ~15,449 µm (median ~3,099 µm) vs 8,273 non-merging segments at mean ~532 µm (median ~102 µm). Welch's t-test on log-transformed lengths: t = 16.54, p = 7.32e-25. Merge errors are dominated by massive runaway components.
- **Caveats:** None noted; effect size is enormous.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Number of merging segments: 64; non-merging: 8273; Merging mean 15448.96 µm, median 3099.27 µm; Non-merging mean 531.76 µm, median 101.74 µm; Welch's t=16.5444, p=7.3232e-25"; rerun "Number of merging segments: 64; non-merging: 8273; Merging mean 15448.96 µm, median 3099.27 µm; Non-merging mean 531.76 µm, median 101.74 µm; Welch's t=16.5444, p=7.3232e-25" → exact match. Revision: $RERUN_PKL replaces an `os.walk` search that returned empty in the sandbox + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: 64 merging vs 8273 non-merging, mean 15449 µm vs 532 µm (~29×); t=16.54, p=7.32e-25. 794491: 98 vs 8544, mean 4457 µm vs 194 µm (~23×); t=29.03, p≈0. 794495: 98 vs 7998, mean 16277 µm vs 477 µm (~34×); t=24.02, p≈0. Merging segments are 23×–34× longer than non-merging on every brain with the log-Welch t-test extremely significant throughout.
- **Verdict:** OK
- **Test:** Welch's two-sample t-test on log-transformed cable length (t=16.54, p=7.32e-25, n=64 vs 8,273) — log-transform handles the heavy-tailed cable-length distribution, and Welch's correction is right for unequal variances.
- **Statistical issues:**
  - **Sample-size asymmetry:** 64 vs 8,273 is heavily unbalanced, but Welch's t accommodates this and ~30× mean ratio is robust.
  - **Tautological flavor:** "merging segments are larger" is partly definitional — a segment that fuses multiple neurons must span them, so it has to be longer. The hypothesis is more informative because of the magnitude (~30×, not ~2×).
  - Multiple-testing-wise this would clearly survive any FDR.
- **Logic issues:**
  - "Driven by enormous overgrown labels rather than local blips" is supported by the magnitude; no overreach.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES (23×–34× ratio across all three brains).

### 12. (Priority 0.253 · Surprise 0.284) Omit errors are >2× more likely at the extreme Z-depths of the imaged volume than at central depths.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 45 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether omit error rate is higher at the top/bottom 10% of Z-coordinates than in the central 20%, attributable to optical attenuation/scattering.
- **Conclusion:** Extreme-Z omit rate 4.44% (2,283/51,369) vs central-Z 1.94% (11,342/585,909). A Cochran–Mantel–Haenszel test controlling for brain ID gave pooled OR = 2.3561, p ≈ 0, validating spatial anisotropy of omissions specifically at axial extremes.
- **Caveats:** This positive Z-extreme result coexists with the negative Z-orientation results (#2, #3) — the effect concerns depth extremes (boundary artifacts), not orientation-driven anisotropy along the cable.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Extreme 2283/51369 (4.44%); Center 11342/585909 (1.94%); Pooled Odds Ratio (Extreme vs Center) = 2.3561, p = 0.0000e+00"; rerun "Extreme 2283/51369 (4.44%); Center 11342/585909 (1.94%); Pooled Odds Ratio = 2.3561, p = 0.0000e+00" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** PARTIAL
- **Across datasets:** origin 789202: Extreme 4.44% vs Center 1.94%, pooled OR=2.3561, p≈0 (extreme Z significantly more error-prone). 794491: Extreme 4.00% vs Center 3.94%, pooled OR=1.0149, p=0.7996 (NO SIGNIFICANT EFFECT — extreme Z behaves like center). 794495: Extreme 5.37% vs Center 1.76%, pooled OR=3.1672, p≈0 (HIGHER than origin, holds strongly). The "extreme Z is worse" effect is robust on 789202 and 794495 (OR > 2.3) but absent on 794491 (OR ≈ 1, p = 0.80).
- **Verdict:** MAJOR
- **Test:** Cochran-Mantel-Haenszel test "controlling for brain ID", pooled OR=2.3561, p≈0, n=637,278 nodes — CMH is appropriate for stratified 2×2 tables, but in the single-brain reproduction there is effectively only one stratum so the "controlling for brain ID" framing is misleading.
- **Statistical issues:**
  - **Stratification framed as multi-brain but executed single-brain:** the recorded code says "controlling for brain variations" but the reproduction confirms this was computed on a single-brain pkl; the stratifier "brain ID" was a no-op for the reported result. The pooled OR collapses to a plain OR on this dataset.
  - **Independence violation:** nodes within the same neuron / region are not i.i.d.; CMH on stratified counts treats every node as independent. The p≈0 is overstated.
  - **Cherry-picked bins:** "extreme = top/bottom 10%, center = central 20%" is a specific bin definition; other bin choices may attenuate the effect. No sensitivity to bin choice is reported.
- **Logic issues:**
  - **Mechanism overreach:** the conclusion attributes the effect to "optical attenuation or scattering along the imaging axis" — the test only shows a spatial association, not the mechanism. Many alternative causes (annotation effort at boundaries, GT density near edges, neurite truncation at volume boundaries) would produce the same signal.
  - The PARTIAL generalization (one brain shows OR=1.01, p=0.80) materially weakens the "optical attenuation" story since the same imaging modality is used across brains; an imaging-physics explanation should generalize, but it doesn't.
- **Downgrade based on rerun/extrapolation:** Yes — PARTIAL generalization (effect absent on 794491). The "extreme Z is worse" claim is dataset-specific, not a generic imaging-physics consequence.
- **Corrected test:** GEE logistic regression with neuron as the cluster id, predicting is_omit from is_extreme (1 = top/bottom 10% Z, 0 = central 20% Z). This replaces the original CMH (which collapsed to a plain OR on a single-brain pkl and ignored within-neuron clustering) with the same OR but cluster-robust SE. Also a cluster-bootstrap CI on the OR by resampling neurons.
- **Corrected result:** origin 789202 — descriptives unchanged (extreme 4.44% vs centre 1.94% on 637,278 nodes). GEE OR(Extreme vs Centre) = 2.3561 (identical point estimate), 95% CI [0.4558, 12.1801], p = 0.31; cluster-bootstrap CI on OR = [0.5046, 22.9182]. Side-by-side: original "CMH pooled OR = 2.3561, p ≈ 0 on 637k nodes" → corrected "OR = 2.3561 with cluster-robust 95% CI = [0.46, 12.18], p = 0.31." The p-value loses ALL of its "≈ 0" significance once the 12 neurons clustering the 637k nodes is recognised.
- **Corrected generalization:** WEAKENED — origin 789202: OR = 2.36 [0.46, 12.18], p = 0.31 (NS); 794491: OR = 1.01 [0.12, 8.80], p = 0.99 (NS, CI spans 2 orders of magnitude); 794495: OR = 3.17 [2.11, 4.76], p = 2.8e-08 (sig, large). The "extreme Z worse" effect survives clustering only on 794495; on origin and 794491 the CI brackets 1 with a wide range.
- **Post-correction verdict:** WEAKENED — the original "pooled OR = 2.36, p ≈ 0" was driven by treating all 637k nodes within only 12 clusters as independent observations; once clustering is respected, the same OR point estimate has CI [0.46, 12.18] and p = 0.31 on origin. The effect remains directionally consistent (point OR > 1 on 2/3 brains) and is robustly significant on 794495, but the original generalisation claim ("imaging-physics effect") cannot be sustained: the effect is not significant on origin or 794491 once their non-independence is accounted for.

### 13. (Priority 0.253 · Surprise 0.284) Short omission gaps are usually internal dropouts within a single predicted segment; long gaps are true fragment boundaries.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 58 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether continuous omit paths whose flanking nodes share the same predicted segment (bridged) are shorter than those with mismatched flanking IDs (broken).
- **Conclusion:** 307 bridged omit paths (mean 18.71 µm, median 13.55 µm) vs 4,298 broken omit paths (mean 42.54 µm, median 20.16 µm); Mann-Whitney U p = 1.95e-19. Short omits are largely internal network dropouts and are good targets for safe auto-filling.
- **Caveats:** Bridged sample is much smaller (n = 307) than broken (n = 4,298), but separation is clear.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Total Bridged 307 (mean 18.71 µm, median 13.55 µm); Total Broken 4298 (mean 42.54 µm, median 20.16 µm); Mann-Whitney U = 458554.0, p = 1.9488e-19"; rerun "Total Bridged 307 (mean 18.71 µm, median 13.55 µm); Total Broken 4298 (mean 42.54 µm, median 20.16 µm); Mann-Whitney U = 458554.0, p = 1.9488e-19" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: bridged n=307 median 13.55 µm vs broken n=4298 median 20.16 µm, Mann-Whitney U=458554, p=1.95e-19. 794491: bridged n=219 median 10.99 µm vs broken n=4353 median 15.23 µm, U=365509, p=2.75e-09. 794495: bridged n=330 median 12.17 µm vs broken n=3815 median 16.24 µm, U=483296, p=1.20e-12. Bridged-shorter-than-broken holds on all three brains; medians and significance line up.
- **Verdict:** MINOR
- **Test:** Mann-Whitney U on continuous omit-path length, n=307 bridged vs 4,298 broken, U=458,554, p=1.95e-19 — correct non-parametric choice for heavy-tailed lengths.
- **Statistical issues:**
  - **Class-imbalance is fine for Mann-Whitney U:** the test handles n=307 vs 4,298 cleanly.
  - **Definition coupling:** "bridged" requires sufficient non-omit neighbors on the same predicted segment; "broken" includes both true terminations and structural breaks. The category label thus partially encodes path length already (a long broken span is less likely to be bridged by the same segment by chance), which inflates the apparent effect.
  - The median ratio (13.55 vs 20.16 µm) is moderate (~1.5×); significance comes partly from n=4,605 total.
- **Logic issues:**
  - "Short omits are largely internal dropouts and are good targets for safe auto-filling" is supported as a class-discrimination claim. The recommendation is operationally reasonable.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES on all three brains in the same direction.

### 14. (Priority 0.253 · Surprise 0.284) Split edges have higher local tortuosity than correctly reconstructed edges, though the correlation is small.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 59 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether high local tortuosity (path length / Euclidean distance over a 10-edge sliding window) increases the likelihood of a split error.
- **Conclusion:** 6,611 split vs 1,091,075 correct edges: median tortuosity 1.1115 vs 1.0764 (means 1.2163 vs 1.1175). Mann-Whitney U p ≈ 0; point-biserial r = 0.0335 (also p ≈ 0). Tortuosity is statistically associated with splits but the effect is small in magnitude.
- **Caveats:** The correlation (r = 0.0335) is tiny — significance comes from sample size; tortuosity alone is a weak per-edge predictor.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Split 6611, Correct 1091075; Median Tortuosity (Split) 1.1115, (Correct) 1.0764; Mean (Split) 1.2163, (Correct) 1.1175; Mann-Whitney U=4.5781e+09, p=0; Point-biserial r=0.0335, p=0"; rerun "Split 6611, Correct 1091075; Median Tortuosity (Split) 1.1115, (Correct) 1.0764; Mean (Split) 1.2163, (Correct) 1.1175; Mann-Whitney U=4.5781e+09, p=0.0000e+00; Point-biserial r=0.0335, p=0.0000e+00" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: split n=6611, correct n=1091075; median 1.1115 vs 1.0764; r=0.0335, p≈0. 794491: split n=7485, correct n=408632; median 1.0860 vs 1.0597; r=0.0607, p≈0. 794495: split n=7662, correct n=912147; median 1.0718 vs 1.0555; r=0.0253, p≈0. Same direction (split > correct), same modest effect-size (r 0.025–0.061), all p ≈ 0 across all brains.
- **Verdict:** MAJOR
- **Test:** Mann-Whitney U + point-biserial correlation, n=6,611 split vs 1,091,075 correct, U=4.58e9, p≈0; r=0.0335, p≈0 — both tests are appropriate in principle but the conclusion is driven entirely by the huge n.
- **Statistical issues:**
  - **Significance dominated by sample size:** with n ≈ 1.1M, point-biserial r = 0.0335 corresponds to a z-score around ~35; the result is unsurprising but the effect is essentially noise (r² ≈ 0.001 — tortuosity explains 0.1% of split-vs-correct variance).
  - **Non-independence:** edges within the same neuron and along the same cable are spatially correlated; the effective n is much smaller than 1.1M. The reported p≈0 vastly overstates evidence.
  - **Median gap of 0.035 in tortuosity** is biologically tiny; this is a textbook "large-n trivial-effect" significance trap.
- **Logic issues:**
  - The paper's verbal conclusion "tortuosity is statistically associated with splits but the effect is small in magnitude" is honest; the caveat in the report acknowledges the tiny r. So the inference is not overreaching — but the discovery loop still flagged this as confirmation of a "strong" structural property by virtue of significance.
  - No predictive value: r=0.03 means a tortuosity-only classifier would be useless in practice.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES but the effect size remains tiny (r=0.025–0.061) on all three brains. The hypothesis should be presented as "directionally consistent but practically negligible."
- **Corrected test:** (a) Cliff's delta (rank-biserial) computed split-vs-correct on tortuosity, with cluster-bootstrap 95% CI by resampling NEURONS — that turns a "significance-driven-by-1.1M" point r-statistic into an effect-size with a CI that respects the within-neuron correlation. (b) Per-neuron paired Wilcoxon on neuron-level median tortuosity (split − correct), the appropriate aggregate test when each neuron's edges are not independent.
- **Corrected result:** origin 789202 — Cliff's delta = +0.2562 with cluster-bootstrap 95% CI [+0.2104, +0.3324] (cluster p ≈ 0); per-neuron paired Wilcoxon W = 78, p = 2.44e-04, n_pairs = 12, bootstrap 95% CI on median(split − correct) per neuron = [+0.02389, +0.05565]. Side-by-side: original "U = 4.58e9, p ≈ 0; point-biserial r = 0.0335, p ≈ 0" — the corrected Cliff's delta of 0.26 is a SMALL but real effect by Vargha-Delaney standards (|delta| 0.147-0.33 = "small"), and the per-neuron paired Wilcoxon confirms the same direction at the right unit of analysis. The original r = 0.03 was a misleadingly small effect size because it was computed at the wrong unit (per-edge); per-cluster the effect is small-but-meaningful.
- **Corrected generalization:** GENERALIZES — origin 789202: Cliff's delta = +0.26 [+0.21, +0.33]; 794491: Cliff's delta = +0.28 [+0.23, +0.34], per-neuron Wilcoxon p = 1.95e-03; 794495: Cliff's delta = +0.21 [+0.14, +0.24], per-neuron Wilcoxon p = 1.91e-06. The direction and magnitude are consistent across all three brains.
- **Post-correction verdict:** WEAKENED — direction confirmed, effect-size much smaller than the p ≈ 0 originally implied but real (Cliff's delta in 0.21–0.28 range, "small" effect by Vargha-Delaney). Original framing was statistically misleading (point-biserial r on 1.1M edges); the corrected Cliff's delta and per-neuron Wilcoxon confirm a small but robust tortuosity-vs-split effect.

### 15. (Priority 0.253 · Surprise 0.284) Split edges are spatially closer to merge sites than correct edges (by ~170 µm), suggesting joint failure zones.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 61 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether the Euclidean distance from split edges to the nearest merge site is smaller than that for correct edges.
- **Conclusion:** Across 6,805 split and 1,109,034 correct edges, split-to-merge mean distance 2,084.76 µm (median 1,794.64 µm) vs correct 2,265.14 µm (median 1,956.93 µm). Mann-Whitney U p = 3.50e-38, Welch's t p = 7.14e-27. Splits and merges co-localize in shared failure zones.
- **Caveats:** Effect size is modest (~170 µm gap on ~2 mm baseline); significance again driven by sample size.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Split mean 2084.76 µm (median 1794.64); Correct mean 2265.14 µm (median 1956.93); Mann-Whitney U=3432660112.0, p=3.50e-38; Welch's t=-10.7131, p=7.14e-27"; rerun "n_split=6805, n_correct=1109034; Split mean 2084.76 µm (median 1794.64); Correct mean 2265.14 µm (median 1956.93); Mann-Whitney U=3432660112.0, p=3.50e-38; Welch's t=-10.7131, p=7.14e-27" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: split median 1794.64 µm vs correct 1956.93 µm, Mann-Whitney p=3.50e-38, t=−10.71, p=7.14e-27. 794491: split 818.04 vs correct 846.80 µm, p=1.98e-14, t=−2.85, p=2.18e-03 (effect much smaller but same direction). 794495: split 1476.84 vs correct 1672.23 µm, p=2.91e-73, t=−22.28, p=4.29e-107. Split-closer-to-merge holds on all three brains; gap shrinks on 794491 but direction and significance are preserved.
- **Verdict:** MAJOR
- **Test:** Mann-Whitney U + Welch's t (n=6,805 split, 1,109,034 correct), p=3.50e-38 / p=7.14e-27 — appropriate tests for the data shape, but the conclusion confuses statistical with practical significance.
- **Statistical issues:**
  - **Effect size small:** ~170 µm gap on a ~2,000 µm baseline (~8% relative shift). The colossal n drives the p-value, not the effect.
  - **Non-independence:** distances along cable are spatially correlated; treating each of 1.1M edges as independent inflates evidence. A bootstrap over neurons would yield a much tamer p-value.
  - **Sparse anchors:** only 67 merge sites anchor the comparison; permutation over merge sites is not reported.
  - **Effect attenuates substantially on 794491** (median gap shrinks from ~170 µm to ~29 µm, t drops from −10.71 to −2.85) — the same direction but practically a different magnitude regime.
- **Logic issues:**
  - "Joint failure zones" is a mechanistic frame the test cannot establish — it could equally reflect that both error types prefer the same dense neuropil for unrelated reasons, or any unmodeled common confounder.
  - The headline "split errors are closer to merge sites by ~170 µm" overstates the practical effect, given how small ~170 µm is relative to the baseline.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES directionally but the effect-size attenuation on 794491 should be highlighted; the report should add "magnitude varies substantially across brains."
- **Corrected test:** (a) cluster-bootstrap by NEURON on the median(split − correct) distance to nearest merge site, returning a 95% CI and a cluster-robust p-value; (b) Cliff's delta with cluster-bootstrap 95% CI; (c) merge-site bootstrap (resample the 67 merge anchors) to bound anchor-set variability. All on the same data.
- **Corrected result:** origin 789202 — observed median gap (split − correct) = −162.29 um. Cluster-bootstrap 95% CI = [−373.03, +15.55] um (CI crosses zero), cluster-p = 0.1067. Cluster-bootstrap Cliff's delta CI = [−0.1935, +0.0171], p = 0.1333. Merge-site bootstrap CI = [−302.78, +274.89] um. Side-by-side: original "U = 3.43e9, p = 3.5e-38; t = −10.71, p = 7.1e-27" → corrected "gap CI straddles zero, p = 0.11; Cliff's delta CI straddles zero, p = 0.13" — neither cluster nor merge-site bootstrap can reject zero at α = 0.05.
- **Corrected generalization:** DOES-NOT-GENERALIZE (cluster-corrected) — origin: gap CI [−373, +16], p = 0.11; 794491: gap CI [−126, +25], p = 0.18 (NS, gap collapses to ≈ −29 um); 794495: gap CI [−457, +24], p = 0.09 (NS but borderline; observed gap −195 um). None of the three brains reach cluster-significance under the corrected test, although all three point estimates are negative.
- **Post-correction verdict:** OVERTURNED — the original "split is closer to merge sites by 170 µm with p = 3.5e-38" relied on treating 1.1M edges as i.i.d.; once the within-neuron correlation is respected, the same point estimate has a 95% CI that includes zero on every brain. The directional consistency (negative on all three) is suggestive but not significant; the "joint failure zones" mechanistic claim is not supported by a cluster-correct test.

### 16. (Priority 0.253 · Surprise 0.284) Effectively 100% of GT split gaps are <15 µm, well below the 80% threshold posited.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 63 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether >80% of disconnected GT segments belonging to the same neuron are separated by <15 µm, to inform a path-finding agent's search radius.
- **Conclusion:** Across 6,805 split transitions, 100.00% of gaps fall under 15 µm, with the ECDF rising sharply between 3–5 µm. 95th percentile ≈ 5.85 µm, 99th ≈ 6.50 µm. A repair agent can use a tight ~6.5 µm radius to cover virtually all real splits while minimizing accidental merges.
- **Caveats:** None noted; this directly reinforces entry #1.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Total split gaps analyzed: 6805; % < 15 µm = 100.00%; ~5.85 µm covers 95%; ~6.50 µm covers 99%"; rerun "Total split gaps analyzed: 6805; % < 15 µm = 100.00%; ~5.85 µm (95%), ~6.50 µm (99%)" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: 6805 gaps, 100.00% < 15 µm; 95% covered by 5.85 µm, 99% by 6.50 µm. 794491: 7847 gaps, 100.00% < 15 µm; 95% by 5.79 µm, 99% by 6.44 µm. 794495: 7988 gaps, 100.00% < 15 µm; 95% by 5.80 µm, 99% by 6.41 µm. The "100% under 15 µm" claim and the ~6.5 µm 99-percentile threshold are nearly identical across all three brains.
- **Verdict:** OK
- **Test:** Descriptive ECDF and percentile thresholds over n=6,805 gaps; no formal hypothesis test reported. The claim "≥80% are under 15 µm" is trivially true given 100% are under 15 µm.
- **Statistical issues:**
  - **No formal test:** no CI on the 95th/99th percentiles is reported, but with n=6,805 these percentiles are tightly estimated and the result generalizes near-identically to two extra brains, so this is fine.
  - The 80% threshold posited in the hypothesis is far below the observed 100%, so the test is one-sided in a trivial direction; no power concern.
- **Logic issues:**
  - None — the description matches the conclusion. The generalization is striking: 95th-percentile is 5.79–5.85 µm across all three brains.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES essentially identically across all three brains.

### 17. (Priority 0.253 · Surprise 0.284) Split errors cluster near topological branch points (mean geodesic distance ~516 µm vs ~711 µm for correct edges).
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 64 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether split edges are closer (geodesically) to GT branch points than correct edges, implicating local geometric complexity in fragmentation.
- **Conclusion:** Across 6,805 split and 1,109,034 correct edges, mean geodesic distance to nearest branch point was 516.30 µm (split) vs 710.59 µm (correct), Mann-Whitney U p = 1.32e-197. Split edges are tightly constrained to branch-point regions and lack the long-distance tails of correct edges.
- **Caveats:** No formal review was provided for this hypothesis (review field "N/A"), so implementation faithfulness is unverified.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "split mean 516.30 µm, correct mean 710.59 µm; Mann-Whitney U p = 1.32e-197 over 6805 split & 1109034 correct edges"; rerun "Total split edges evaluated: 6,805; Total correct edges evaluated: 1,109,034; split mean 516.30 µm, correct mean 710.59 µm; Mann-Whitney U=2979025451.5, p=1.3239e-197" → exact match. Revision: $RERUN_PKL replaces a `Path.rglob` search that returned empty in the sandbox + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: split mean 516.30 µm vs correct 710.59 µm over 6805/1109034 edges, U=2.98e9, p=1.32e-197. 794491: split 384.29 µm vs correct 389.49 µm over 7847/420702 edges, U=1.46e9, p=5.16e-67 (much smaller effect size — only ~5 µm gap — but same direction, still highly significant). 794495: split 423.67 µm vs correct 506.89 µm over 7988/934849 edges, U=3.47e9, p=2.97e-27. Split-closer-to-branch-point holds on all three brains; effect size is largest on the origin and weakest on 794491.
- **Verdict:** MAJOR
- **Test:** Mann-Whitney U on geodesic distance to nearest branch point, n=6,805 split vs 1,109,034 correct, U=2.98e9, p=1.32e-197 — appropriate non-parametric for skewed distance data; reviewer field is "N/A" so faithfulness of implementation is unaudited.
- **Statistical issues:**
  - **Non-independence:** geodesic distance from one edge to a branch point is highly correlated with its neighbors' distances; effective n is much smaller than 1.1M.
  - **Effect-size attenuation:** on 794491 the median gap collapses from ~195 µm to ~5 µm yet p remains <1e-66. This is the signature of n-driven significance: direction is preserved, but the practical signal nearly vanishes on one brain.
  - **No review (review = "N/A")** — implementation faithfulness has not been audited; the experiment loop didn't generate a checker.
- **Logic issues:**
  - "Complex local geometry around branch points increases the risk of fragmentation" is mechanistically plausible but the observational test cannot rule out alternative explanations (signal density, GT density, soma proximity).
  - Headline overstates: "split edges are tightly constrained to branch-point regions" doesn't capture the highly attenuated effect on 794491.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES directionally on all three brains, though the report should be qualified with "effect size varies by ~30× across brains."
- **Corrected test:** (a) cluster-bootstrap by NEURON on the median(split − correct) geodesic distance to nearest branch point and on Cliff's delta; (b) per-neuron paired Wilcoxon signed-rank on neuron-level median distance. Both account for the within-neuron spatial autocorrelation the original Mann-Whitney treated as i.i.d.
- **Corrected result:** origin 789202 — observed median gap = −184.19 um, Cliff's delta = −0.2238 with cluster-bootstrap 95% CI = [−0.2847, −0.1346], cluster-p ≈ 0; per-neuron paired Wilcoxon W = 0, p = 2.44e-04, bootstrap CI on per-neuron median(split − correct) = [−178.03, −87.09] um. Side-by-side: original "U = 2.98e9, p = 1.32e-197" → corrected "Cliff's delta = −0.22 [−0.28, −0.13] (small effect by Vargha-Delaney), cluster-p ≈ 0; per-neuron Wilcoxon p = 2.4e-04." The direction is confirmed; the effect-size moves from a "p ≈ 0 sample-size artefact" to a "small but real" Cliff's delta.
- **Corrected generalization:** GENERALIZES — origin 789202: Cliff's delta = −0.22 [−0.28, −0.13], p ≈ 0; 794491: Cliff's delta = ≈ −0.13 [−0.18, −0.07], cluster-bootstrap CI on gap = [−56.90, −26.46] um, p ≈ 0; per-neuron Wilcoxon p = 1.95e-03. 794495: Cliff's delta = ≈ −0.08 [−0.13, −0.03], CI on gap = [−55.85, −9.74] um, p = 0.013; per-neuron Wilcoxon p = 0.027. The effect is consistent in direction and significant under the cluster-corrected test on all three brains, with effect size attenuating from "small" on origin to "negligible-to-small" on 794495.
- **Post-correction verdict:** UPHELD — the original conclusion ("split edges are closer to branch points") survives cluster correction on all three brains, but the practical magnitude is now properly quantified as a Cliff's delta of 0.08–0.22 (negligible-to-small), not the "p ≈ 1e-197" the original i.i.d. Mann-Whitney implied. Headline framing should be "small but robust spatial association" rather than "tightly constrained."

### 18. (Priority 0.253 · Surprise 0.284) Independent reconfirmation: split edges show significantly higher local 5-hop tortuosity than correct edges.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 73 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether 5-hop local skeleton tortuosity is higher for split edges than for correct edges (a finer-window replication of #14).
- **Conclusion:** 1,109,034 correct vs 6,805 split edges: median tortuosity 1.0801 vs 1.1132 (means 1.1194 vs 1.2062), one-sided Mann-Whitney p = 1.66e-276. The U-Net consistently struggles through sharp turns, but per-edge effect size is again small.
- **Caveats:** Same caveat as #14: the absolute median difference (~0.03) is modest; the result depends on the huge sample for significance. This is essentially a robustness check of #14.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Median Tortuosity CORRECT 1.080117, SPLIT 1.113188; Mean CORRECT 1.119395, SPLIT 1.206197; Mann-Whitney U=4714207979.0, p=1.6599e-276 over 1109034 correct & 6805 split edges"; rerun "Processed 1109034 correct edges and 6805 split edges; Median CORRECT 1.080117, SPLIT 1.113188; Mean CORRECT 1.119395, SPLIT 1.206197; Mann-Whitney U=4714207979.0, p=1.6599e-276" → exact match. Revision: $RERUN_PKL replaces a `Path.rglob` search that returned empty in the sandbox + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: correct median 1.0801, split median 1.1132 over 1.11M correct / 6805 split, U=4.71e9, p=1.66e-276. 794491: correct 1.0617, split 1.0854 over 420702/7485, U=2.09e9, p≈0. 794495: correct 1.0575, split 1.0735 over 912147/7662, U=4.45e9, p=6.19e-192. Same direction (split > correct), same modest magnitude (~0.02–0.03 median gap), all p extremely small. Independent replication of #14 holds on all brains.
- **Verdict:** MAJOR
- **Test:** One-sided Mann-Whitney U on 5-hop local tortuosity, n=6,805 split vs 1,109,034 correct, U=4.71e9, p=1.66e-276 — appropriate non-parametric choice; same statistical concerns as #14 (this is its sliding-window replication).
- **Statistical issues:**
  - **Tortuosity windows overlap:** consecutive edges share most of their 5-hop window, so per-edge tortuosity values are heavily autocorrelated. The independence assumption of Mann-Whitney is violated even more strongly than in #14.
  - **Effect size tiny:** median gap of ~0.03 in tortuosity; mean gap of ~0.09. The huge p-value is driven by n ~ 1.1M.
  - **Replication does not add evidence:** because #18 uses the same dataset and a near-identical metric to #14, it is mostly a robustness check rather than independent confirmation. Reporting it as a separate finding inflates the apparent number of confirmations.
- **Logic issues:**
  - "Overwhelming statistical evidence" is misleading framing: the median gap is tiny and indistinguishable from #14's median gap. The conclusion should emphasize the marginality of the effect, not its p-value.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES but same caveat as #14: practically negligible effect size on all brains.
- **Corrected test:** (a) cluster-bootstrap by NEURON on the median(split − correct) 5-hop tortuosity gap and on Cliff's delta — the 5-hop sliding windows OVERLAP by 4 edges so per-edge tortuosities are not independent, and the Mann-Whitney p-value of 1.66e-276 is the inflated consequence; (b) per-neuron paired Wilcoxon on neuron-level median tortuosity (split vs correct).
- **Corrected result:** origin 789202 — observed median gap = +0.0331 (essentially identical to the original ≈ +0.03), Cliff's delta = +0.2498 with cluster-bootstrap 95% CI = [+0.1926, +0.3110], cluster-p ≈ 0; per-neuron paired Wilcoxon W = 78, p = 2.44e-04, n_pairs = 12. Side-by-side: original "U = 4.71e9, p = 1.66e-276" → corrected "Cliff's delta = +0.25 [+0.19, +0.31] (small effect), cluster-p ≈ 0; per-neuron Wilcoxon p = 2.4e-04." Direction confirmed; effect-size moves from "significance-driven-by-n" to a "small but consistent" Cliff's delta.
- **Corrected generalization:** GENERALIZES — origin 789202: Cliff's delta = +0.25 [+0.19, +0.31]; 794491: Cliff's delta = +0.27 [+0.22, +0.32], per-neuron Wilcoxon p = 1.95e-03; 794495: Cliff's delta = +0.19 [+0.15, +0.24], per-neuron Wilcoxon p = 1.91e-06. Consistent direction and magnitude across all three brains.
- **Post-correction verdict:** UPHELD — direction confirmed and Cliff's delta in the 0.19–0.27 range across all three brains (small but real), matching the corrected effect-size for #14 (id 59) almost exactly (Cliff's delta 0.21–0.28 for 10-hop). This is genuine reconfirmation of the tortuosity-vs-split relationship but at a "small effect" magnitude rather than the "p ≈ 0" headline; the framing should be "small but robust" not "overwhelming statistical evidence."

### 19. (Priority 0.253 · Surprise 0.284) Merge sites carry a strong geometric "tangle" signature — local branch density ~28× higher than at non-merge controls.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 84 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether merge sites exhibit higher local branch density (within 15 µm) in the U-Net fragments graph than matched non-merge control sites on the same segments.
- **Conclusion:** Across 67 merge sites, mean local branch density was 1.13 vs 0.04 at controls; paired t = 13.35 (p = 2.03e-20), Wilcoxon W = 21.0 (p = 1.19e-11), ROC-AUC 0.9236. Spurious local branching is a strong geometric feature for detecting merge errors automatically.
- **Caveats:** Only 67 merge sites; reproducibility on larger merge populations would strengthen the result.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "67 merge sites; density merge 1.13 vs control 0.04; paired t=13.35 p=2.03e-20; Wilcoxon W=21.0 p=1.19e-11; ROC-AUC 0.9236"; rerun "Analyzed 67 merge sites; merge 1.13 vs control 0.04 branches/15µm; paired t=13.3482 p=2.0302e-20; Wilcoxon W=21.0000 p=1.1942e-11; ROC-AUC 0.9236" → all key numbers match exactly. The script exited 1 *after* printing all statistics because the post-analysis matplotlib `boxplot(..., labels=...)` keyword was renamed to `tick_labels` in matplotlib 3.11; this is a plotting-API quirk, not a reproduction failure. Revision: direct $RERUN_PKL load + numpy>=2 + vendored sklearn.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: 67 merge sites; merge density 1.13 vs control 0.04; paired t=13.35, p=2.03e-20; Wilcoxon W=21, p=1.19e-11; AUC=0.9236. 794491: 86 merges; 1.10 vs 0.08; t=12.14, p=2.97e-20; W=75, p=5.47e-13; AUC=0.8825. 794495: 105 merges; 1.02 vs 0.05; t=13.46, p=1.65e-24; W=109, p=7.95e-16; AUC=0.8857. (Same downstream matplotlib boxplot `labels` crash on all three runs after the statistics print — analysis itself reproduces cleanly on every brain.) Merge sites carry the dense-branching tangle signature on all three brains with AUC ≥ 0.88.
- **Verdict:** OK
- **Test:** Paired t-test (t=13.35, p=2.03e-20) + Wilcoxon signed-rank (W=21, p=1.19e-11) + ROC-AUC (0.9236) on 67 merge sites with on-segment matched controls — both tests are appropriate for paired data, and Wilcoxon (non-parametric) corroborates the parametric t-test against any normality concerns.
- **Statistical issues:**
  - **Small but adequate n:** 67 merge sites is modest, but the effect size is huge (1.13 vs 0.04, a ~28× ratio) so power is fine. Both parametric and non-parametric tests agree.
  - **Matched controls on the same segment** is a reasonable design that absorbs neuron-level confounders.
  - **Same metric used as the discoverable feature:** the AUC of 0.9236 is an in-sample-on-control-design quantity; out-of-sample performance on raw segments would be the relevant deployment number, but the structural claim (local-branch-density is high at merge sites) is solid.
- **Logic issues:**
  - "Strong geometric signature ... can serve as a powerful feature" is supported by AUC ≥ 0.88 on three brains, no overreach.
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES (AUC 0.88–0.94 on all three brains, paired t and Wilcoxon both significant).

### 20. (Priority 0.253 · Surprise 0.284) Omit errors are nearly twice as likely on terminal (leaf-ending) edges as on internal edges.
- **Run:** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID:** 85 · **Belief:** Leaning True → Likely True (0.7083 → 0.8906) · **Direction:** Positive
- **Tested:** Whether omit errors disproportionately occur on terminal branches (paths ending in a leaf) rather than internal segments of the GT neuron's topology.
- **Conclusion:** Terminal-edge omit rate 4.38% (23,792 / 543,477) vs internal-edge omit rate 2.41% (20,898 / 865,568); χ² = 4189.94, p < 1e-300. An 81% relative increase shows the segmentation model fails distal thin neurites at roughly double the internal rate.
- **Caveats:** None noted; sample sizes and effect size are both large.
- **Reproduction:** REPRODUCED (code: revised-loading)
- **Rerun result:** recorded "Terminal Omit=23792 / Non-Omit=519685 (4.38%); Internal Omit=20898 / Non-Omit=844670 (2.41%); Chi-square=4189.94, p ≈ 0"; rerun "Terminal Omit=23792 Non-Omit=519685 (4.38%); Internal Omit=20898 Non-Omit=844670 (2.41%); Chi-square=4189.9364, p=0.0000e+00" → exact match. Revision: direct $RERUN_PKL load + numpy>=2.
- **Generalization:** GENERALIZES
- **Across datasets:** origin 789202: terminal 4.38% vs internal 2.41% (ratio 1.82×), χ²=4189.94, p≈0. 794491: terminal 4.94% vs internal 3.82% (ratio 1.29×), χ²=425.40, p=1.63e-94. 794495: terminal 2.47% vs internal 1.82% (ratio 1.36×), χ²=691.98, p=1.66e-152. Terminal-omit-rate-greater-than-internal holds on all three brains, all p ≪ 0.001; relative magnitude is ~1.3×–1.8×.
- **Verdict:** MINOR
- **Test:** Chi-square test of independence on edge-type (terminal vs internal) × omit (yes/no), χ²=4189.94, p<1e-300, n=1,409,045 edges — appropriate for a 2×2 contingency table; expected cell counts are very large.
- **Statistical issues:**
  - **Independence violation:** edges within the same neuron are not i.i.d.; clustering at neuron level would tighten SE. With ratio 1.82× and χ² ≈ 4200, no realistic correction overturns the conclusion.
  - **Effect-size attenuation across brains:** ratio drops from 1.82× (origin) to 1.29× (794491) and 1.36× (794495). Direction is consistent but the "twice as likely" headline overstates two of three brains.
- **Logic issues:**
  - "Distal thin neurites" is a mechanistic conjecture not directly tested — terminal edges could be omitted for many reasons (signal sparsity, axon-vs-dendrite difference, GT-curation artifacts).
- **Downgrade based on rerun/extrapolation:** No — GENERALIZES directionally. Headline "nearly twice as likely" should be softened to "1.3× to 1.8× more likely depending on brain."

---

## Reproduction — Summary

- **Dataset pkl used:** `cache/dataset_cache_789202_mcl100_add.pkl` (single brain: 789202, mcl 100, with `_add` enrichment).
- **Breakdown over the top-20 reranked records:** **19 REPRODUCED**, **1 DIVERGED**, **0 FAILED** (n_rerun = 20). All 20 verdicts came from `code_source: revised` (every script needed a loading/env revision; 0 ran the recorded code unchanged).
- **Findings that did NOT reproduce:**
  - Entry 5 (id 24, "super-merges fuse 3+ neurons"): **DIVERGED — dataset scope, not analysis bug.** The recorded experiment aggregated across multiple `*_add.pkl` brain caches (n=24 two-neuron + 3 super-merges, U=0, p=5.89e-03); the rerun has only the single-brain pkl provided (n=8 + 1, U=0, p=2.22e-01). Medians (6.21 mm vs 35.19 mm) and the U statistic still match — only the sample size and therefore the p-value differ. The within-pkl analysis is faithful.
- **Loading revisions applied** (`autodiscovery/run-4--ground-truth-error-annotations-revised-version_2026-06-20.json.rerun/hypo_<id>.py`):
  - **NumPy-2 pkl unpicklable under the host's NumPy 1.x** (ids 21, 24, 27, 30, 32, 33, 36, 37, 39, 45, 58, 59, 61, 63, 84, 85): vendored a fresh `numpy>=2 / scipy / pandas / sklearn / patsy / statsmodels / matplotlib / networkx / psutil / agentic_neuron_proofreader` stack to `~/.local-numpy2`, prepended it on `sys.path`, removed `~/.local/lib` and the `/shared/utils...` site-packages from the path so the new stack wins.
  - **"Dataset not found" gates** (ids 64, 73, 139, 43): the recorded code searched `Path.rglob`/`os.walk('..')` for `*_add.pkl`, which the rerun helper's `glob`/`Path.exists` monkeypatch doesn't cover. The revision adds `Path.rglob` and `os.walk` monkeypatches in the bootstrap so any `.pkl` search yields the single `$RERUN_PKL`.
  - **`pip install` retry loops** in every script were defanged (replaced with no-op `subprocess`) to avoid re-installing per record at runtime.
  - **sklearn / patsy ABI incompatibilities** (ids 30, 27) were resolved by the same vendored numpy-2 stack (sklearn 1.9.0, patsy 1.0.2, statsmodels 0.14.6, pandas 3.0.3).
- **Note on entry 19 (id 84):** the rerun's exit code is 1 but ALL key numbers (paired t=13.3482 p=2.03e-20, Wilcoxon W=21 p=1.19e-11, ROC-AUC 0.9236, n=67 merge sites) printed BEFORE the crash, which happened in a `plt.boxplot(..., labels=...)` plotting call — `labels` was renamed to `tick_labels` in matplotlib 3.11. The analysis itself reproduced exactly; only the post-analysis plot rendering was hit by a downstream library API rename, so this is counted as REPRODUCED.

---

## Generalization — Summary

- **Extra datasets tested:** `cache/dataset_cache_794491_mcl100_add.pkl` and `cache/dataset_cache_794495_mcl100_add.pkl` (both single-brain caches with the same payload structure as the origin 789202 brain).
- **Verdict breakdown over the top 20:** **14 GENERALIZES**, **3 PARTIAL**, **3 DOES-NOT-GENERALIZE**, **0 INCONCLUSIVE**.
- **Per-entry verdicts:** #1 (id 30) GENERALIZES · #2 (id 27) PARTIAL · #3 (id 21) DOES-NOT-GENERALIZE · #4 (id 139) DOES-NOT-GENERALIZE · #5 (id 24) PARTIAL · #6 (id 32) GENERALIZES · #7 (id 33) GENERALIZES · #8 (id 36) DOES-NOT-GENERALIZE · #9 (id 37) GENERALIZES · #10 (id 39) GENERALIZES · #11 (id 43) GENERALIZES · #12 (id 45) PARTIAL · #13 (id 58) GENERALIZES · #14 (id 59) GENERALIZES · #15 (id 61) GENERALIZES · #16 (id 63) GENERALIZES · #17 (id 64) GENERALIZES · #18 (id 73) GENERALIZES · #19 (id 84) GENERALIZES · #20 (id 85) GENERALIZES.

**Findings that DO NOT fully generalize (with one-line reasons):**

- **#3 (id 21) — Z-dominant vs XY-dominant Chi-square — DOES-NOT-GENERALIZE.** Origin had no significant difference (p=0.060); 794491 shows Z significantly HIGHER error rate (p=5.2e-38), 794495 shows Z significantly LOWER (p=4.6e-88). Both extras reject the no-anisotropy story, in OPPOSITE directions.
- **#4 (id 139) — Branch order predicts splits (positive coefficient) — DOES-NOT-GENERALIZE.** Origin had coef=+0.0194 (p<0.001); 794491 coef=−0.0157 and 794495 coef=−0.0063 (both p<0.001). All three are significant, but the SIGN FLIPS on both extras.
- **#8 (id 36) — Omitted cable closer to merge sites — DOES-NOT-GENERALIZE.** Origin and 794495 confirm (omit < correct), but on 794491 the relationship reverses (omit 1103 µm > correct 842 µm, one-sided p=1.000). The spatial co-clustering is not robust across brains.
- **#2 (id 27) — Z-alignment mixed-effects regression — PARTIAL.** All three brains print a non-positive Z-coefficient (so the "Z not a driver" headline survives), but the standardized coef varies from +0.0376 (794491, p=0.17) to −0.0661 (origin, p=0.058) to −0.1697 (794495, highly significant, p=6.75e-06). 794495 finds a strong protective effect that the prior did not anticipate.
- **#5 (id 24) — Super-merges fuse 3+ neurons cover more cable — PARTIAL.** Origin single-brain rerun is not significant (n=8+1, p=0.222) — flagged DIVERGED already; 794491 has 0 super-merges so the test cannot run; 794495 strongly confirms (n=24+2, p=6.15e-03). The pattern holds where super-merges exist but they are rare on some brains.
- **#12 (id 45) — Extreme-Z omit rate vs central-Z — PARTIAL.** Origin OR=2.36 (p≈0) and 794495 OR=3.17 (p≈0) confirm; 794491 OR=1.01 (p=0.80) shows no effect at all. Edge-of-volume omission is dataset-specific.

**Synthesis.** With two extra brains tested, the local-geometry findings — distance-only split bridging (#1), branch-node omit rate (#6), angular alignment (#7), split clustering (#9), A* repair success (#10), merging-segments-are-giants (#11), bridged-vs-broken omit length (#13), tortuosity (#14, #18), split proximity to merges and branch points (#15, #17), 6.5 µm split-gap threshold (#16), merge tangle signature (#19), terminal-edge omit rate (#20) — replicate cleanly across all three brains and look like robust dataset-independent properties of the U-Net's failure modes. By contrast, every Z-axis/anisotropy claim (#2, #3, #12) is brain-specific or directionally inconsistent: the no-anisotropy story on the origin actually flips into significantly different (and contradictory) Z-effects on the other two brains, and the extreme-Z omission effect is present on 2/3 brains and absent on the third. The branch-order coefficient (#4) flips sign on both extras, so its positive direction on the origin is likely an artifact of brain 789202 rather than a general trend. The super-merge claim (#5) and the merge-omit spatial co-clustering claim (#8) hold on one extra brain but break on the other. Given only two extra datasets, statements about "robust" should be read as "consistent on 3 brains", which is good evidence but not conclusive proof of universality.

---

## Excluded (no surprisal score)

The helper dropped 2 hypotheses for missing surprisal:

- Run `run-4--ground-truth-error-annotations-revised-version_2026-06-20`, ID 10
- Run `run-4--ground-truth-error-annotations-revised-version_2026-06-20`, ID 41

---

## Statistical Verification — Summary

### Per-hypothesis severity (priority order)

**CRITICAL (3):**
- **#3 (id 21) — Z-dominant vs XY-dominant Chi-square.** Borderline p=0.0602 misinterpreted as "refutes Z-anisotropy"; "failed to reject = null is true" fallacy. Generalization reverses the null in opposite directions on the two extra brains.
- **#4 (id 139) — Centrifugal branch order logistic regression.** Independence violation (no neuron random effect), tiny per-unit OR ≈ 1.02 inflated by n = 1.4M, and the coefficient sign FLIPS on both extra brains while still printing p < 0.001 — the hallmark of significance without robustness.
- **#8 (id 36) — Omitted cable closer to merge sites.** Non-independence + small ~147 µm effect on ~2 mm baseline + reversal on 794491 (omit FARTHER than correct, p=1.000). Mechanism claim ("model sacrifices thin adjacent processes") is overreach.

**MAJOR (7):**
- **#2 (id 27) — Z-alignment mixed-effects logreg.** Arbitrary 20k subsample drops 98% of data and parks the p at the 0.058 borderline; conclusion "Z not a driver" is drawn from p>0.05 with the same null-acceptance fallacy. Generalization PARTIAL (sign and significance vary).
- **#5 (id 24) — Super-merges fuse 3+ neurons.** Underpowered (n=3 super-merges), test compares total cable not per-neuron cable (hypothesis deviation), and on 794491 super-merges don't exist at all.
- **#12 (id 45) — Extreme-Z omit rate.** "Stratification by brain ID" is a no-op on a single brain; non-independence understates p; mechanistic optical-attenuation claim contradicted by 794491 (OR=1.01, p=0.80).
- **#14 (id 59) — Local tortuosity for split detection.** r=0.0335 is significance-without-effect (r² = 0.001); non-independence; large-n trivial-effect trap.
- **#15 (id 61) — Split distance to merge sites.** ~170 µm gap on 2 mm baseline (~8% relative shift); effect attenuates ~6× on 794491.
- **#17 (id 64) — Split distance to branch points.** Effect attenuates ~30× on 794491 (195 µm → 5 µm gap) yet p stays <1e-66; non-independence on 1.1M edges. Review field is "N/A" so faithfulness is unaudited.
- **#18 (id 73) — 5-hop tortuosity replication.** Same significance-by-n trap as #14; not actually independent (same dataset, near-identical metric). Reports as if it adds evidence.

**MINOR (5):**
- **#6 (id 32) — Branch vs linear node omit rate.** Large effect (4.3×), p≈0; only concern is edge non-independence which doesn't overturn the result. GENERALIZES.
- **#9 (id 37) — Split clustering.** Partly tautological ("splits cluster with splits"), but 10× ratio is real and generalizes.
- **#10 (id 39) — A* repair.** Engineering benchmark with no statistical test, no CI on success rate.
- **#13 (id 58) — Bridged vs broken omit length.** Class-definition partially encodes length already; effect is moderate (~1.5× median ratio).
- **#20 (id 85) — Terminal vs internal omit rate.** Edge non-independence; ratio attenuates to 1.3× on two extra brains.

**OK (4):**
- **#1 (id 30) — Euclidean gap distance.** Distance-only AUC 0.9979 on 11k gaps reproducing to ≥0.989 on two extra brains; in-sample threshold optimism is mild.
- **#7 (id 33) — Angular alignment.** AUC 0.9322 with KS 0.7460 on 13,582 splits, reproducing AUC 0.93 on two extras.
- **#11 (id 43) — Merging segments are giants.** Welch t on log-cable-length, ~30× ratio across all brains.
- **#16 (id 63) — 100% of split gaps under 15 µm.** Descriptive, near-identical 95th-percentile thresholds (~5.8 µm) on all three brains.
- **#19 (id 84) — Merge-site branch density.** Paired t + Wilcoxon + AUC=0.92, with matched on-segment controls. Reproduces with AUC ≥ 0.88 on all three brains.

### Multiple comparisons (BH-FDR, q=0.05) over the top-20

- 18 of the 20 top-ranked findings report a usable p-value (#10 and #16 report descriptive statistics, no p; these are excluded from BH-FDR).
- Applying Benjamini–Hochberg with q=0.05 to those 18 p-values, the cutoff is k=16 with p ≤ k/N·q = 0.0444.
- **Survivors (16/18):** #1 (effective p ≈ 0, AUC-based), #4 (logreg p<0.001), #5 (Mann-Whitney p=0.0059), #6 (χ²≈0), #7 (KS≈0), #8 (p=1.6e-66), #9 (p≈0), #11 (p=7.3e-25), #12 (p≈0), #13 (p=1.9e-19), #14 (p≈0), #15 (p=3.5e-38), #17 (p=1.3e-197), #18 (p=1.7e-276), #19 (paired t p=2.0e-20), #20 (χ²≈0).
- **Do NOT survive (2/18):** **#2 (id 27) p=0.0582 > 0.0444**, **#3 (id 21) p=0.0602 > 0.0444**. Both are the borderline Z-axis "no-effect" results whose verbal conclusions are already discredited by generalization across brains. The BH-FDR result confirms that these borderline p-values would also fail run-wide correction even at this small panel size.
- For the full 150-hypothesis run (not just top-20), the BH cutoff would be considerably tighter; if one further considers the ~250-hypothesis discovery-loop family typical of these runs, ~12 spuriously significant results are expected by chance at α=0.05. The strong-effect findings (AUC > 0.9, ratios > 4×, p ≪ 1e-20) are insensitive to any reasonable correction; the borderline ones (#2, #3) are not.

### Cross-cutting issues

- **Non-independence of edges/nodes within neurons is endemic.** #3, #4, #6, #8, #9, #14, #15, #17, #18, #20 all run tests over ~1M edges treated as i.i.d. when in reality edges within the same neuron / cable / fragment are heavily correlated. With clustering, the effective n is one-to-several orders of magnitude smaller. This does not overturn the large-effect findings (#6 ratio 4.3×, #9 ratio 10×, #20 ratio 1.8×) but it makes the borderline / small-effect findings (#14 r=0.03, #15 ~170 µm gap, #17 attenuating to ~5 µm on 794491, #18 median gap 0.03) much weaker than the printed p-values suggest. None of the top-20 used a mixed-effects model or cluster-robust SE.
- **"Failed to reject = null is true" fallacy.** #2 and #3 both treat p just above 0.05 as confirming no effect; the prior → posterior swing on these is a major contributor to the run's "headline" surprise scores. This is the single most consequential logic error in the run.
- **Significance driven by sample size, not effect size.** #8 (~7% relative shift on ~1M edges, p=1.6e-66), #14 (r=0.0335 on 1.1M edges, p≈0), #15 (~8% relative shift, p=3.5e-38), #17 (5–195 µm gap, p=1.3e-197), #18 (Δmedian 0.03, p=1.7e-276), #20 (1.3×–1.8× ratio on 1.4M edges) — all of these would warrant a much smaller "effect size" claim if reported with bootstrap-over-neurons CIs.
- **Generalization-as-validation.** Three findings (#3, #4, #8) fully reverse on extra brains and three more (#2, #5, #12) are partial; the original surprise / belief shifts on these are unjustified once the extras are considered. They should be downgraded from "confidence flip" to "brain-specific."
- **Mechanistic overreach.** #4 ("deeper-order branches more susceptible"), #8 ("model sacrifices thin processes when fusing"), #12 ("optical attenuation at depth"), #15 ("joint failure zones"), #20 ("distal thin neurites") all draw causal/mechanistic conclusions from purely associational tests. The downstream implications matter more for actionable intervention design than for the reported surprise scores.

---

## Statistical Test Corrections — Summary

**Scope:** 10 of the top-20 hypotheses were flagged for a test-level fix and re-measured with corrected scripts in `autodiscovery/run-4--ground-truth-error-annotations-revised-version_2026-06-20.json.fixed/hypo_<id>.py`. The remaining 10 entries either ran the right test or had only a "test is fine, headline overreaches" caveat.

### Post-correction verdict breakdown

- **UPHELD (4):** #3 (id 21), #14 (id 59), #17 (id 64), #18 (id 73). On each, the direction the original claimed survives a cluster-correct test; the effect size is properly quantified (Cliff's delta or paired Wilcoxon at the neuron level) and is "small but real" rather than the "p ≈ 0" the original i.i.d. test inflated. #3 is UPHELD only at the origin (no anisotropy effect when neuron clusters are respected) but DOES-NOT-GENERALIZE — that part is unchanged.
- **WEAKENED (4):** #4 (id 139), #5 (id 24), #12 (id 45), #14 (id 59). The direction survives but the effect-size is much smaller than the original significance implied, or the result is significant on only 1/3 brains under the cluster-correct test. (#14 is listed in both UPHELD and WEAKENED because the direction generalises but the effect-size is small once properly estimated; see entry for details.)
- **OVERTURNED (3):** #2 (id 27), #8 (id 36), #15 (id 61). The corrected test either inverts the sign (#2: Z is significantly PROTECTIVE under cluster-robust GEE, opposite of "Z not a driver"; #8: gap REVERSES on 794491 under cluster bootstrap), or no brain reaches cluster-significance (#15: cluster-bootstrap CI on the gap includes zero on every brain).

### Tabular summary

| # | id | Original test | Original headline result | Corrected test | Corrected origin result | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 27 | mixed-effects logreg on 20k subsample | coef = -0.066, p = 0.058, OR = 0.80 | GEE on full 1.16M, cluster-robust SE | coef = -0.123, p ≈ 0, OR = 0.66 [0.62, 0.72] | OVERTURNED |
| 3 | 21 | chi-square on 1.4M edges (i.i.d.) | chi2 = 3.53, p = 0.060 | GEE + cluster-perm chi2 + per-neuron Wilcoxon | OR = 1.02 [0.97, 1.07], all p > 0.45 | UPHELD (origin) |
| 4 | 139 | plain logreg on 1.4M edges (i.i.d.) | coef = +0.019, z = 16.37, p < 0.001 | GEE with neuron cluster | coef = +0.019, p = 0.038, OR = 1.02 [1.001, 1.04] | WEAKENED |
| 5 | 24 | Mann-Whitney U on TOTAL cable (multi-brain) | U = 0, p = 5.9e-03 (floor) | MW on PER-NEURON cable + Cliff's delta + bootstrap CI | n=8 vs 1 on origin, p = 0.22 (NS) | WEAKENED |
| 8 | 36 | one-sided MW on 49k node distances | U = 1.138e9, p = 1.60e-66 | cluster-bootstrap by neuron + Cliff's delta + merge-site bootstrap | gap CI [-402, +92] um, p = 0.27 | OVERTURNED |
| 12 | 45 | CMH "stratified by brain" (single-brain) | OR = 2.36, p ≈ 0 | GEE with neuron cluster + cluster-bootstrap on OR | OR = 2.36 [0.46, 12.18], p = 0.31 | WEAKENED |
| 14 | 59 | MW + point-biserial on 1.1M edges | U = 4.58e9, p ≈ 0; r = 0.034 | cluster-bootstrap Cliff's delta + per-neuron Wilcoxon | delta = +0.26 [+0.21, +0.33] | WEAKENED |
| 15 | 61 | one-sided MW + Welch's t on 1.1M edges | U = 3.43e9, p = 3.5e-38 | cluster-bootstrap by neuron + Cliff's delta + merge-site bootstrap | gap CI [-373, +16] um, p = 0.11 | OVERTURNED |
| 17 | 64 | MW on 1.1M edge geodesic distances | U = 2.98e9, p = 1.32e-197 | cluster-bootstrap Cliff's delta + per-neuron Wilcoxon | delta = -0.22 [-0.28, -0.13], p ≈ 0 | UPHELD |
| 18 | 73 | one-sided MW on 1.1M edge 5-hop tortuosity | U = 4.71e9, p = 1.66e-276 | cluster-bootstrap Cliff's delta + per-neuron Wilcoxon | delta = +0.25 [+0.19, +0.31], p ≈ 0 | UPHELD |

### Patterns

- **Non-independence inflation is the dominant fault.** 8 of 10 corrected entries (#3, #4, #8, #12, #14, #15, #17, #18) had a test that treated within-neuron-correlated observations as i.i.d. Once a cluster-robust (GEE) or cluster-bootstrap (neuron-resampled) test is applied, three findings (#2, #8, #15) OVERTURN, two find the effect significant only on 1 of 3 brains (#4 on origin, #12 on 794495), and three (#14, #17, #18) retain the direction at "small effect" magnitude that the original "p ≈ 0" framing massively overstated.
- **The "failed-to-reject = null is true" fallacy compounds with under-sampling.** #2 (20k subsample, p = 0.058) and #3 (i.i.d. chi-square, p = 0.060) were the two entries whose ORIGINAL conclusions ("Z not a driver") were drawn from p > 0.05. Both are flipped under the corrected analysis — #2 becomes significantly *protective* (Z reduces error odds) on origin and 794495, and #3 becomes "no effect on origin but DOES-NOT-GENERALIZE" with one brain showing protective and one showing positive Z effect. The "Likely True → Uncertain" surprise on #2 and #3 is unjustified by the data.
- **Sample-size-driven significance becomes effect-size honesty.** #14, #17, #18 all retained the same direction under cluster-correct tests but with Cliff's delta in the "small" range (|delta| 0.08–0.28). The original p ≈ 0 / p ≈ 1e-200 numbers were artefacts of treating ~1M correlated edges as independent; the true evidence is consistent across brains but at "small effect" magnitude.
- **Floor effects and tautologies.** #5 (id 24) is a textbook small-sample floor effect: U = 0 with n = 3 super-merges yields the smallest achievable p of 0.006, but the per-neuron rerun on the origin brain alone gives p = 0.22; the "highly significant" claim is sustainable only on 794495 (n = 24 + 2 super-merges).

### Synthesis

Once the correct test is applied, the headline picture of the run shifts in three important ways. **First, the two confidence-flipping anisotropy findings (#2 id 27, #3 id 21) cannot be sustained as "no effect" results;** on origin, the cluster-robust GEE on the full 1.16M edges of #2 shows Z-alignment is significantly *protective* (OR = 0.66, p ≈ 0) — exactly the opposite of the prior's "Likely True → Uncertain" swing — while #3's "fail-to-reject" survives under correct clustering but DOES-NOT-GENERALIZE in opposite directions. **Second, the three "huge-n, tiny effect" findings (#14, #17, #18) become honest "small effect" findings:** the direction and across-brain robustness survives, but the effect-size is properly described as Cliff's delta ≈ 0.08–0.28 rather than "p ≈ 1e-200". **Third, two of the spatial co-clustering findings (#8 omit-near-merge, #15 split-near-merge) collapse under cluster correction:** the cluster-bootstrap 95% CI on the median gap includes zero on every brain for #15 and on 2/3 brains for #8, with one brain actually reversing the direction. The mechanistic claims about "joint failure zones" and "the model sacrifices thin processes when fusing" are no longer supported by a sample-size-honest test. The robustly UPHELD findings under correct tests are the local-geometry / topology ones — tortuosity (#14, #18) and branch-point proximity (#17) — at small but real effect sizes. The OVERTURNED or WEAKENED findings concentrate in (a) the Z-axis anisotropy story and (b) the spatial co-clustering claims, exactly the entries the prior cross-cutting analysis already flagged for "large-n trivial-effect significance" and "failed-to-reject fallacy."
