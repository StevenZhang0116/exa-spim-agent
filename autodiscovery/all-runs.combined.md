# AutoDiscovery Cross-Report Consolidated Summary

## Source reports and coverage

| Report (run) | Origin dataset | Entries |
| --- | --- | --- |
| `ground-truth-error-annotations-revised-version_2026-06-17` | 789202 | 20 |
| `run-4--ground-truth-error-annotations-revised-version_2026-06-20` | 789202 | 20 |
| `run-5--ground-truth-error-annotations-revised-version_2026-06-25` | 794495 | 20 |

- Reports ingested (`n_files`): 3
- Total entries ingested (`n_entries_total`): 60
- Distinct findings after consolidation: 24
- Findings unique to a single report: 12

### Synthesis

Across the three runs the dominant, repeatedly corroborated picture is a small set of robust spatial/topological error signatures: (1) **merge sites sit in dense, branchy neighborhoods** (high local fragment density and high branch-node density) — re-measured at 10 µm, 15 µm, volumetric density, and as branch-density ROC signatures, and it GENERALIZES on every brain; (2) **merge segments are giant, overgrown, asymmetric "runaway" labels** — confirmed as ~15–29× larger cable than non-merge segments, asymmetric overlap (median 1.0) with one primary neuron, with cable length / node count as a cheap, generalizing merge prior; (3) **split errors concentrate near branch points and at thin distal/terminal tips, and cluster spatially into "error zones"** — found by many distance/geodesic/topological tests, all GENERALIZING in direction (though several are huge-n / trivial-effect-size cases that WEAKEN under neuron-cluster correction); (4) **split gaps are tiny (≈4–7 µm, ~99% < 6.5 µm) while inter-neuron gaps are larger, so a ~6.5–6.84 µm proximity threshold plus angular/collinearity priors reconnects ~85–86% of splits safely**; and (5) **omit errors are bursty/contiguous and concentrated at distal terminal branches**, confirmed by Markov-transition and run-length analyses on all brains.

The most important **unique/new** findings come from run-4 and run-5: run-4's near-perfect proximity classifier (ROC-AUC 0.9979, 6.84 µm threshold), angular-inertia continuation heuristic (ROC-AUC 0.9322), A* graph repair of 86% of splits, and the two refuted Z-orientation/anisotropy hypotheses; run-5's brand-new crossing-fiber merge geometry (>45°), bimodal split-gap GMM (now formally model-selected), asymmetric merge-overlap pruning insight, the radius-matching precision/recall trade-off, and the attribution of ~96% of omit gaps to the 100 µm min-cable-length filter.

The clearest **cross-report disagreements** are about Z-axis effects and branch-order/distal gradients. Run-4's "z-alignment / Z-dominant orientation is NOT a risk driver" (#2/#3) carries DOES-NOT-GENERALIZE / PARTIAL — the same test becomes strongly significant in *opposite* directions on the two extra brains. Run-4's "split risk rises with branch order" (#139) was OVERTURNED (sign flips, neuron-cluster permutation non-significant). The 2026-06-17 report's "omissions are distal / closer to leaves" finding is only PARTIAL — direction flips on dataset 794491 — whereas the categorical terminal-vs-internal omit-rate restatement GENERALIZES cleanly. Several "splits/omits co-cluster with merges" and "splits cascade near merges" claims that looked significant per-edge were OVERTURNED or downgraded once the neuron was used as the unit of analysis, and run-5's split→omit local co-occurrence (#42) DOES-NOT-GENERALIZE (sign flip + non-significance on both extra brains).

---

## Unique & New Findings

### 1. (Priority 0.507 · Surprise 0.690) A ~6.84 µm Euclidean threshold safely auto-reconnects the vast majority of split fragments.
- **Sources:** run-4--...2026-06-20 (id 30) — only report to find this proximity-classifier result.
- **Conclusion:** Tested whether spatial proximity alone separates true split gaps from inter-neuron gaps. Across 6,805 true split gaps and 4,189 inter-neuron gaps within 20 µm, distributions were almost entirely non-overlapping (true splits peak ~4.5 µm; inter-neuron gaps rarely below 7 µm). Gap distance as a binary classifier gave ROC-AUC 0.9979, and the F1-optimal threshold of 6.84 µm reached F1 = 0.9945. The large positive surprisal (+0.690, Leaning False → Leaning True) reflects that the loop had leaned against proximity-only repair, but the data strongly validate it as a safe, effective heuristic.
- **Verdict carried over:** Reproduction REPRODUCED (exact match); Generalization GENERALIZES (ds_794491 ROC-AUC=0.9889, thr=6.48 µm, F1=0.9711; ds_794495 ROC-AUC=0.9953, thr=6.83 µm, F1=0.9878); Verdict SOUND.
- **Why unique/new:** Not found in the 2026-06-17 report; first appears in the newer run-4 as its top-ranked, highest-surprise result. (Run-5's id 63 corroborates the *gap-size* fact, folded into Corroborated #4; the ROC-AUC classifier construction is unique to run-4.)
- **Caveats:** The false-positive "inter-neuron gaps" are derived from GT labels in simulation rather than live merges, so this is a deployment caveat; the F1-optimal threshold is selected in-sample (mild optimism, immaterial given AUC=0.9979).

### 2. (Priority 0.409 · Surprise 0.471) Merges happen at crossing fibers, with crossing angles significantly steeper than 45°.
- **Sources:** run-5--...2026-06-25 (id 15) — only report to find this; the newest run.
- **Conclusion:** Tested whether the 3D crossing angle between two GT neurons at a merge site exceeds 45°. Across 12 valid merge sites the intersection-angle distribution had mean 65.97° and median 74.93° (clustering 65°–90°); a one-sample Wilcoxon signed-rank test against 45° gave statistic 69.00, p = 0.0081, rejecting the null. The positive surprisal (+0.471) was the largest belief shift in run-5, lifting an Uncertain prior to Likely True — merges are predominantly a crossing-fiber geometry problem.
- **Verdict carried over:** Reproduction REPRODUCED (exact match); Generalization PARTIAL (ds_794491 median 54.30°, stat=223.00, p=5.37e-02, n=25 — same direction but loses significance; ds_789202 produced 0 valid merge sites, untestable); Verdict WEAK.
- **Why unique/new:** First appears in the newest run-5; no Z-angle/crossing-fiber merge-geometry hypothesis exists in the two earlier reports.
- **Caveats:** Severely underpowered (n=12); the 45° benchmark is an arbitrary reference; only one of two extra datasets yielded testable sites, and that one was marginal (p≈0.054).

### 3. (Priority 0.275 · Surprise 0.300) Split gaps are bimodal: tiny single-dropout gaps plus large missed-stretch gaps.
- **Sources:** run-5--...2026-06-25 (id 12) — only report to model split-gap modality.
- **Conclusion:** A 2-component Gaussian Mixture Model on 7,988 split gaps found Component 1 (~63.6% of splits) at mean ~0.0 µm (negligible displacement) and Component 2 (~36.4%) at mean ~159.2 µm (std ~259.0 µm, large missed stretches), implying a connector should use two search radii. The post-correction model-selection test confirmed bimodality formally: BIC(k=1)−BIC(k=2)=114,816.7, likelihood-ratio 114,843.7, Hartigan dip statistic=0.4128 with bootstrap p=0.0020.
- **Verdict carried over:** Reproduction REPRODUCED (recovered from a recorded sandbox FAILED run); Generalization GENERALIZES (dip test rejects unimodality, p=0.0020, on both extras; large-gap mean varies 117–329 µm); Verdict WEAK → Post-correction verdict UPHELD.
- **Why unique/new:** First appears in the newest run-5; no earlier report fit a mixture model to split-gap distances.
- **Caveats:** GMM component assignment is model-dependent; Component 1 at mean ~0.0 µm with std ~0.001 µm may be a near-degenerate spike. Original analysis reported no model-selection test (bimodality was asserted by construction before correction).

### 4. (Priority 0.275 · Surprise 0.300) Merging segments are highly asymmetric — mostly minor offshoots of one primary neuron (overlap median 1.0).
- **Sources:** run-5--...2026-06-25 (id 22) — only report to measure merge overlap asymmetry.
- **Conclusion:** Across 98 merging segments the mean overlap ratio with the primary neuron was 0.9129 (median 1.0); a one-sample test that the mean exceeds 0.5 gave t = 24.87, p = 3.58e-44. Balanced 50/50 fusions are extremely rare, so merge corrections are pruning a minor accidental offshoot rather than dividing a balanced segment. The corrected non-parametric tests (Wilcoxon W=4656.0, p=1.87e-19; sign test p=1.26e-29; median 1.0, bootstrap CI [1.0, 1.0]) confirm the same conclusion.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES (ds_794491 mean 0.8558, t=17.18, p=1.72e-31; ds_789202 mean 0.9704, t=43.12, p=8.84e-49); Verdict MINOR → Post-correction verdict UPHELD.
- **Why unique/new:** First appears in the newest run-5; an actionable framing of merge fixes ("prune, don't split") absent from the two earlier reports.
- **Caveats:** Single-dataset origin cache for the headline; the original one-sample t-test was inappropriate for a [0,1]-bounded ratio piled at the 1.0 ceiling (test-choice fault, corrected without changing the conclusion).

### 5. (Priority 0.275 · Surprise 0.300) A radius-matching constraint cuts false-positive merges by ~56%, but halves split-fixing power.
- **Sources:** run-5--...2026-06-25 (id 47) — only report to test the thickness-constraint trade-off.
- **Conclusion:** Adding a leaf `node_radius` difference < 25% constraint to a < 15 µm proximity split-correction heuristic reduced false-positive merges by 55.63% (151→67 pairs, exceeding the 40% target) but retained only 43.95% of the baseline's split-resolving power (1,513→665 true splits resolved). This is a precision/recall trade-off, not a free win.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES (ds_794491 47→15 pairs = 68.09% reduction, 37.14% retained; ds_789202 22→7 pairs = 68.18% reduction, 35.87% retained); Verdict WEAK.
- **Why unique/new:** First appears in the newest run-5; an explicit safety/efficacy trade-off study not present in the earlier reports.
- **Caveats:** No inferential statistic or CI; the primary metric rests on small pair counts (151/67, falling to 47/15 and 22/7). The secondary "canonical merges" metric is explicit noise (origin 33.33%, 794491 −100.00%, 789202 200.00%) and was set aside.

### 6. (Priority 0.275 · Surprise 0.300) ~96% of omit gaps fall below the 100 µm min-cable-length filter — the dominant omit driver is the tool's short-fragment filter, not U-Net failure.
- **Sources:** run-5--...2026-06-25 (id 69) — only report to attribute omits to the build-time filter.
- **Conclusion:** Of 4,145 continuous omitted stretches (totaling 122,590 µm), 3,971 (95.80%) were shorter than 100 µm, accounting for 85,498 µm (69.74%) of total omitted cable. This implicates the cache's configurable `min_cable_length` preprocessing filter — not large-scale segmentation failure — as the primary omit driver, with direct pipeline-configuration implications.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES (ds_794491 98.62% of stretches / 91.38% of cable; ds_789202 91.73% / 55.94%); Verdict MINOR.
- **Why unique/new:** First appears in the newest run-5; a pipeline-configuration insight (vs model behavior) absent from earlier reports.
- **Caveats:** Causal-attribution overreach — the 100 µm cutoff is the dataset's own build parameter, so finding most gaps below it is partly circular; the experiment measures gap-length proportions, not whether those gaps would otherwise have been reconstructed.

### 7. (Priority 0.265 · Surprise 0.414) Split-error risk rises with centrifugal branch order (topological depth from soma) — OVERTURNED.
- **Sources:** run-4--...2026-06-20 (id 139) — only report to test branch order vs split risk.
- **Conclusion:** Logistic regression over 1.4M edges found branch order a significant positive predictor of split errors (coef = 0.0194, p < 0.001; split rates spike >3% near order 53). The positive surprisal (+0.414) reflected the loop revising upward. However, the correction showed the effect to be an artifact: cluster-robust z collapsed from 16.4 to 1.990 (p = 0.04662), and the neuron-level permutation Spearman rho = 0.2308 was non-significant (p = 0.4697).
- **Verdict carried over:** Reproduction REPRODUCED; Generalization DOES-NOT-GENERALIZE (coef flips to −0.0157 on ds_794491 and −0.0063 on ds_794495, both significant in the *opposite* direction); Verdict MAJOR → Post-correction verdict OVERTURNED.
- **Why unique/new:** Only run-4 tested centrifugal branch order; first and only appearance.
- **Caveats:** The `norm_thickness` covariate had zero variance and was dropped, so the "independent of thickness" clause was structurally untestable; edge non-independence inflated the original p; the conclusion does not survive on any brain under the correct neuron-level test.

### 8. (Priority 0.253 · Surprise 0.284) Angular inertia (~153° vs ~90°) reliably identifies the true continuation across a split.
- **Sources:** run-4--...2026-06-20 (id 33) — only report to test directional continuation across splits as a classifier.
- **Conclusion:** Across 13,582 split configurations, true continuations averaged 152.96° versus 90.17° for false candidates; distributions were distinct (KS = 0.746, p ≈ 0) and angular alignment discriminated with ROC-AUC 0.9322. Directional inertia is a strong local heuristic for an agentic proofreader bridging splits.
- **Verdict carried over:** Reproduction REPRODUCED (exact match); Generalization GENERALIZES (ds_794491 ROC-AUC=0.9281; ds_794495 ROC-AUC=0.9365); Verdict SOUND.
- **Why unique/new:** Only run-4 framed continuation angle as a ROC classifier. (Related to but distinct from the 2026-06-17 endpoint-cosine finding in Corroborated #11, which measures anti-parallel endpoint alignment, not continuation-vs-false-candidate discrimination.)
- **Caveats:** The two angle samples are paired per split node while KS assumes independence (immaterial at KS=0.746); the false candidate is the nearest node from another neuron, a sensible conservative comparator.

### 9. (Priority 0.253 · Surprise 0.284) A* path-finding on the fragments graph repairs 86% of splits without inducing merges.
- **Sources:** run-4--...2026-06-20 (id 39) — only report to test graph A* split repair.
- **Conclusion:** Graph A* search penalizing sharp deviations (>45°) and sudden radius changes connected 5,881 of 6,805 targeted splits (86.42% success, far exceeding the 40% threshold) without crossing into different GT neurons; simulated repair lifted edge accuracy from 78.71% to 79.13% (+0.42%).
- **Verdict carried over:** Reproduction REPRODUCED (exact match); Generalization GENERALIZES (ds_794491 84.72%; ds_794495 89.71%, consistently ~85–90%); Verdict MINOR.
- **Why unique/new:** Only run-4 implemented an A* repair simulation; not present in the other reports. (Note: this id 39 is run-4's A* repair, distinct from the 2026-06-17 id 39, which is a KDTree-sever merge-resolution heuristic — see Corroborated #12.)
- **Caveats:** "No merge induced" is adjudicated against GT labels inside the simulation, not a live segmentation, so the no-merge guarantee is over-claimed for deployment; the 5,000-node A* expansion cap and 10 µm merge-validity radius are arbitrary; net accuracy gain (+0.42%) is modest.

### 10. (Priority 0.253 · Surprise 0.284) Omit errors are ~2.4× more likely at extreme Z-depths than at central depths — OVERTURNED.
- **Sources:** run-4--...2026-06-20 (id 45) — only report to test axial-depth (position) omit excess.
- **Conclusion:** Extreme-Z nodes (top/bottom 10%) had a 4.44% omit rate (2,283/51,369) versus 1.94% (11,342/585,909) centrally; a Cochran-Mantel-Haenszel test gave pooled OR 2.3561, p ≈ 0. The positive surprisal (+0.284) suggested axial-extreme signal loss. The correction overturned it: the neuron-cluster permutation OR = 0.5320, p = 0.6607 (non-significant).
- **Verdict carried over:** Reproduction REPRODUCED; Generalization PARTIAL (ds_794495 OR=3.1672, p≈0, stronger; ds_794491 OR=1.0149, p=7.996e-01, vanishes); Verdict MAJOR → Post-correction verdict OVERTURNED.
- **Why unique/new:** Only run-4 tested Z *depth/position* (distinct from the Z *orientation* hypotheses in Corroborated #13). First and only appearance.
- **Caveats:** The CMH "controlling for brain" was illusory (one brain per origin pkl); "extreme Z" is per-volume min-max normalized, conflating optical depth with FOV-boundary artifacts; node non-independence inflated the original p.

### 11. (Priority 0.253 · Surprise 0.284) Short omission gaps are internal dropouts bridged by the same segment; long gaps are true terminations.
- **Sources:** run-4--...2026-06-20 (id 58) — only report to split omits into bridged vs broken classes.
- **Conclusion:** Of omit paths analyzed, 307 were "bridged" (same flanking segment) and 4,298 "broken"; bridged gaps were much shorter (mean 18.71 µm, median 13.55 µm) than broken gaps (mean 42.54 µm, median 20.16 µm), significant by one-sided Mann-Whitney U (p = 1.95e-19). Short omits are largely artifactual internal dropouts.
- **Verdict carried over:** Reproduction REPRODUCED (exact match); Generalization GENERALIZES (ds_794491 bridged 10.99 µm vs broken 15.23 µm, p=2.75e-09; ds_794495 12.17 µm vs 16.24 µm, p=1.20e-12); Verdict SOUND.
- **Why unique/new:** Only run-4 distinguished bridged vs broken omit paths; not present elsewhere.
- **Caveats:** Bridged paths (n=307, ~7% of omit paths) are a modest minority; the analysis unit is a connected omit-path component (genuinely independent), so MW independence holds.

### 12. (Priority 0.324 · Surprise 0.351) Split-gap pairing follows a non-linear distance-vs-angle trade-off (origin only) — does not generalize.
- **Sources:** ...2026-06-17 (id 47) — only report to fit the distance×angle interaction.
- **Conclusion:** Logistic regression on 1,072 true-split positives vs 142 false-merge negatives found a significant distance × angle interaction (coef = −5.96, p = 0.014; LLR p = 1.626e-17): short gaps (< 8 µm) tolerate sharp turns while long gaps (~15 µm) demand collinearity. On the origin brain the permutation test (p = 3.5964e-02) and stratified bootstrap CI [−12.3429, −1.1075] kept the interaction significant.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization DOES-NOT-GENERALIZE (the interaction collapses: ds_794491 x3=−0.3416, p=0.800; ds_794495 x3=−1.4540, p=0.565, fit did not converge); Verdict MAJOR → Post-correction verdict UPHELD on origin, Corrected generalization DOES-NOT-GENERALIZE.
- **Why unique/new:** Only the 2026-06-17 report fit this interaction model. (The underlying engineering motivation — fixed-radius joining is inadequate — is corroborated separately in Corroborated #6.)
- **Caveats:** Headline rests on a single borderline interaction coefficient (p=0.014) under heavy class imbalance (1072:142); pseudo-R² only 0.093; the µm thresholds are read off the fitted boundary, not independently validated; the non-linearity is specific to the origin brain.

---

## Corroborated Findings (consolidated)

### 13. (Priority 0.323 · Surprise 0.795) Z-axis neurite orientation/anisotropy is NOT a consistent driver of topological errors — dataset-specific null.
- **Sources:** run-4--...2026-06-20 (id 27, Z-alignment mixed-effects logistic; id 21, Z-dominant-edge chi-square) — 2 entries, 1 report.
- **Conclusion:** Run-4 tested two forms of the anisotropy hypothesis. Z-alignment: a Bayesian mixed-effects logistic on a 20,000-edge subset gave coef = −0.066, p = 0.058, OR 0.80 (non-significant, slightly negative); the strong negative surprisal (−0.795) marked a confidently-held assumption the data contradicted. Z-dominant edges: error rates were nearly identical (3.70% Z-dom vs 3.63% XY-dom; χ² = 3.53, p = 0.0602). The corrected neuron-cluster permutation tests confirm no detectable origin effect (z-alignment p = 0.5721; Z-dominant cluster-permutation p = 0.8136, RR≈1.02).
- **Agreement:** Both entries reach the same origin conclusion — orientation relative to the anisotropic axis is not a substantial error driver on the origin brain.
- **Disagreement:** Both fail to generalize and in conflicting ways. For Z-dominant edges (id 21) the same test becomes strongly significant in *opposite* directions on the extras: ds_794491 χ²=166.1, p=5.21e-38 (Z higher); ds_794495 χ²=395.8, p=4.62e-88 (Z lower). For z-alignment (id 27) the coefficient sign is unstable (−/+/−) and ds_794495 is significant (p=6.75e-06, a decrease). So the "no anisotropy effect" null is dataset-specific, not a stable property.
- **Verdict carried over:** Reproduction REPRODUCED (both, exact match); Verdict MAJOR (both); Post-correction verdict UPHELD as a null on origin but Generalization DOES-NOT-GENERALIZE (id 21) / PARTIAL (id 27, id 21 corrected).
- **Caveats:** Both originals committed the absence-of-evidence/evidence-of-absence fallacy; id 27's reported p was mis-constructed (VB posterior treated as a Wald z) and fit on a 20k subsample; edge non-independence inflated both; reframe as "no detectable effect," not proof of absence.

### 14. (Priority 0.287 · Surprise 0.307) Merge sites sit in significantly denser local neurite neighborhoods than correct/control regions.
- **Sources:** ...2026-06-17 (id 3, 10 µm radius; id 23, 15 µm radius; id 81, volumetric nodes/µm³; id 49, distance to fragment branch point; id 53, branch-node density per 100 µm), run-4--...2026-06-20 (id 84, local branch density ROC) — 6 entries, 2 reports.
- **Conclusion (canonical = id 3, SOUND, highest-confidence):** Merge sites averaged 7.03 fragment nodes per 10 µm radius versus 4.39 for controls, highly significant by Mann-Whitney U (U = 3931.0, p = 9.0e-15, n=67/67). The same crowding signature recurs at 15 µm (11.09 vs 6.33 nodes, U=4100.5, p=8.95e-17 — id 23), as volumetric density (0.001678 vs 0.001083 nodes/µm³, Welch t=8.93, p=2.8e-14 — id 81), as proximity to fragment branch points (median 4.48 µm vs 179.76 µm, p=2.2e-17 — id 49), as branch-node density per cable (0.1078 vs 0.0568 branches/100 µm, ~1.9×, p=2.9e-25 — id 53), and as a local-branch-density classifier (1.13 vs 0.04 branches/15 µm, paired t=13.35, ROC-AUC=0.9236 — run-4 id 84).
- **Agreement:** All six agree merge sites are in crowded, branchy neuropil and all GENERALIZE 2-for-2 (e.g. id 3: 794491 p=2.3e-17, 794495 p=1.1e-16; id 53 effect grows to ~3.0× on extras; run-4 id 84 ROC-AUC 0.88–0.92 across brains). The density-based forms (id 3/23/81) and the branch-density forms (id 49/53/84) are the same underlying "dense, branchy environment causes merges" claim measured different ways.
- **Disagreement:** None — all positive, all GENERALIZE.
- **Verdict carried over:** Best/most-conservative across the cluster — Verdict SOUND (id 3, id 23, id 53); id 81 WEAK (wrong parametric test) → Post-correction UPHELD; id 49 MINOR (unseeded control); run-4 id 84 MINOR (in-sample AUC).
- **Caveats:** Small positive class (n≈64–105 merge sites/segments) in most; id 81 used a t-test on count data (corrected to MW, conclusion unchanged); id 49's control was unseeded and restricted to non-branch nodes, inflating the gap (DIVERGED on rerun in baseline value, conclusion holds); id 84's ROC-AUC is in-sample.

### 15. (Priority 0.287 · Surprise 0.307) Merge segments are giant "runaway" overgrown labels, an order of magnitude larger than non-merge segments.
- **Sources:** run-4--...2026-06-20 (id 43, log-cable Welch t), run-5--...2026-06-25 (id 34, cable ratio + ROC; id 55, corrected-mapping cable MWU; id 56, node count + cable MWU), and id 24 (super-merge cable, overturned) — 5 entries, 2 reports.
- **Conclusion (canonical = run-5 id 56 / run-4 id 43, SOUND):** Merge-causing segments cover vastly more cable than non-merge segments: run-4 id 43 found 64 merging segments at ~15,449 µm mean (median 3,099 µm) vs ~532 µm (median 102 µm) for 8,273 non-merging (~29× mean, Welch t = 16.54 on log-lengths, p = 7.32e-25). Run-5 confirms with corrected mappings: id 34 (98 merge segments avg 19,040 µm vs 1,221 µm, 15.6× ratio, ROC-AUC=0.869), id 55 (median 4,604.78 µm vs 160.14 µm, U=3.53e7, p=0.0), id 56 (median 791 vs 51 nodes, U=368,540, p=1.04e-38). Cable length / node count is a cheap, sensitive merge prior.
- **Agreement:** All converge on "merges are massive overgrown labels, not local blips," GENERALIZING on all brains (id 43 t≥16 on all three; id 34 ratios 8.3–17.6×; id 56 p≤1.04e-38). id 24 (super-merges ≥3 neurons cover more cable than 2-neuron merges) is a finer sub-claim of the same theme.
- **Disagreement:** id 24's *significance* did not hold: it DIVERGED on rerun (p 0.0059 → 0.2222) and was OVERTURNED under the per-neuron construct (U=8.0, p=0.2222, n_super=1); the general "merges are giant" claim is unaffected, but the specific super-merge-vs-2-neuron comparison is too sample-starved to call.
- **Verdict carried over:** Verdict SOUND (id 43, id 34, id 55, id 56); id 24 MAJOR → Post-correction verdict OVERTURNED.
- **Caveats:** Merge classes are small (n=64–98) and right-skewed (means outlier-sensitive; medians more robust); run-5's non-merge baselines depend on fragment-exclusion/mapping choices; id 24 rests on n=1–3 super-merges.

### 16. (Priority 0.287 · Surprise 0.307) Split errors concentrate at / near branch points (split rate elevated and split edges sit closer to branch nodes).
- **Sources:** ...2026-06-17 (id 10, branch-edge split rate ~3.4×; id 13, geodesic distance to branch; id 19, µm distance to GT branch; id 72, topological distance to GT branch), run-4--...2026-06-20 (id 64, geodesic distance to branch), run-5--...2026-06-25 (id 3, ≤15 µm branch-proximal RR=1.59; id 33, distance to branch node) — 7 entries, 3 reports.
- **Conclusion (canonical = ...2026-06-17 id 10, SOUND):** Branching edges split far more than linear edges — 1.62% (248/15,298) vs 0.47% (6,557/1,393,747), ~3.4×, χ² = 414.5, p = 3.9e-92. The continuous-distance forms agree: split edges sit closer to branch points geodesically (id 13/72: median ~196 vs 380 µm), in µm (id 19: median 121 vs 222 µm), and run-5's branch-proximal edges carry RR=1.59 (0.90% vs 0.57%, p=4.90e-34). All three reports independently re-discover that bifurcations are systematic split-failure loci.
- **Agreement:** Direction GENERALIZES on every brain across all entries (id 10 ratio attenuates to ~2× but stays p≤3.5e-29; run-5 id 3 RR grows to 2.58–3.24× on extras; id 19/72 hold 2-for-2).
- **Disagreement:** None in direction. Magnitude/robustness vary: the huge-n distance forms (id 13, id 72, run-4 id 64, run-5 id 33) are flagged as effect-size-trivial under correction — run-4 id 64 WEAKENED (neuron-cluster permutation p=0.096 on origin, PARTIAL on extras) and run-5 id 33 WEAKENED (Cliff's delta 0.07, cluster CI includes 0 on origin/794491). The categorical rate forms (id 10, run-5 id 3) remain SOUND.
- **Verdict carried over:** Best = SOUND (id 10, id 19, id 72, run-5 id 3); weaker = MINOR (id 13 — recorded double-count bug; run-4 id 64 → WEAKENED); run-5 id 33 WEAK → WEAKENED.
- **Caveats:** Edge pseudo-replication inflates the per-edge p-values; id 13 and id 65 (see #18) had a recorded double-counting bug (DIVERGED, conclusion intact); the distance effects shrink markedly on 794491.

### 17. (Priority 0.287 · Surprise 0.307) Split errors are spatially clustered into localized "error zones" (one split raises the chance of nearby splits).
- **Sources:** ...2026-06-17 (id 55, KS vs random null), run-4--...2026-06-20 (id 37, split-neighbors within 30 µm), run-5--...2026-06-25 (id 26, Monte Carlo NN distance; id 40, KS NN distance; id 45, Monte Carlo NN replication) — 5 entries, 3 reports.
- **Conclusion (canonical = run-5 id 26, SOUND):** Observed inter-split nearest-neighbor distance is far below a random null on every brain — run-5 id 26: 140.30 µm observed vs 320.79 µm random (paired t=−15.10, p=5.76e-12, n=19 neurons). The 2026-06-17 id 55 (median 25.58 µm observed vs 236.98 µm random, KS D=0.4539), run-4 id 37 (0.97 vs 0.10 split-neighbors within 30 µm), and run-5 id 40 (KS=0.5635) / id 45 (Monte Carlo replication, t=−14.93) all confirm splits cluster rather than scatter.
- **Agreement:** All five GENERALIZE in direction (observed ≈ half of random across brains); run-4 id 37 UPHELD under neuron-cluster permutation (p=0.0002, Cliff's delta 0.49).
- **Disagreement:** None — universally positive and robust. Run-5 id 45 is explicitly a replication of id 26 on the same neurons.
- **Verdict carried over:** Verdict SOUND (run-5 id 26, id 40, id 45); ...2026-06-17 id 55 MINOR (causal "cascade" wording over-reads association); run-4 id 37 MINOR → Post-correction UPHELD.
- **Caveats:** "Cascade"/"one split causes another" is causal over-reach — the tests show clustering, consistent with a shared latent cause (local image quality); split edges are non-independent (handled by the neuron-cluster correction for run-4 id 37).

### 18. (Priority 0.287 · Surprise 0.307) Split errors favor thin distal / terminal processes (shorter distance-to-leaf than correct edges).
- **Sources:** ...2026-06-17 (id 60, mean distance-to-leaf 939 vs 1283 µm; id 65, median 417 vs 670 µm thickness proxy), run-5--...2026-06-25 (id 38, terminal vs internal split rate; id 70, monotonic split rate vs distance-to-leaf) — 4 entries, 2 reports.
- **Conclusion (canonical = ...2026-06-17 id 60, SOUND):** Split edges sit closer to terminal leaves than correct edges — mean 938.60 µm vs 1,282.69 µm (p=2.2e-134, n=6,805/1,109,034); id 65 restates this as a thickness proxy (median 417.22 vs 670.25 µm). Run-5 confirms terminal compartments split more than internal ones (id 38: 0.65% terminal vs 0.54% internal, paired t=3.93, p=9.71e-04) and that split rate declines monotonically with distance to the leaf tip (id 70: 0.92% < 50 µm → 0.53% > 200 µm, logistic coef=−0.1361, p<0.001).
- **Agreement:** Direction GENERALIZES across entries (id 60 holds 2-for-2; run-5 id 38 p<0.05 same direction on both extras).
- **Disagreement:** Magnitude/robustness vary. Run-5 id 70 is PARTIAL — the formal logistic slope is non-significant on ds_794491 (coef=−0.0095, p=0.409) and WEAKENED post-correction (GEE z drops 16.4→2.87 on origin), though the per-neuron Spearman decline is consistent everywhere. id 65 had a recorded double-counting bug (DIVERGED, conclusion intact).
- **Verdict carried over:** Best = SOUND (id 60); id 65 MINOR (double-count bug + indirect thickness proxy); run-5 id 38 WEAK (tiny ~0.1 pp absolute gap); run-5 id 70 MAJOR → Post-correction WEAKENED, PARTIAL generalization.
- **Caveats:** Distance-to-leaf is an *indirect* proxy for caliber, not a measured radius; huge-n inflates significance (read medians/effect sizes); run-5 id 70's intended GEE failed and fell back to a pooled logit, mishandling per-neuron non-independence.

### 19. (Priority 0.287 · Surprise 0.307) Omit errors are bursty/contiguous — concentrated in continuous stretches, not isolated dropped edges.
- **Sources:** ...2026-06-17 (id 48, Markov transition, 89% conditional / 28× ratio), run-5--...2026-06-25 (id 27, run-length transition matrix; id 50, 100% zero-distance omit neighbor) — 3 entries, 2 reports.
- **Conclusion (canonical = run-5 id 27, SOUND):** Omits occur in sticky runs. Run-5 id 27: baseline omit probability 2.23% jumps to 84.55% conditional on an adjacent omit (Wilcoxon W=0.0, p=3.81e-06; Omit→Omit transition 0.849, mean run length 6.91 edges, max 339). The 2026-06-17 id 48 found the same with a marginal 3.17% rising to conditional 89.08% (ratio 28.09, ≫ the hypothesized 3×). Run-5 id 50 (100% of omit edges have a zero-distance omit neighbor) is the degenerate-but-corrected version, UPHELD via a within-neuron permutation showing genuine clustering beyond the run-structure artifact (mean per-neuron gap −113.21 µm, p=4.9998e-05).
- **Agreement:** All GENERALIZE (id 48 ratio 18.6–40.4× on extras; id 27 conditional ~80–87% on all brains; id 50 corrected clustering significant on all three).
- **Disagreement:** None in conclusion.
- **Verdict carried over:** Verdict SOUND (run-5 id 27); ...2026-06-17 id 48 MINOR (pseudo-replicated χ²) → Post-correction UPHELD (PARTIAL: corrected permutation completed only on 794491, p=9.99e-04; origin/794495 timed out but descriptive ratios huge); run-5 id 50 MAJOR (degenerate point-mass MWU) → Post-correction UPHELD.
- **Caveats:** The high conditional probability is partly definitional for errors that occur in connected runs; the corrected tests (edge/neuron permutation) confirm clustering beyond that artifact; id 48's original χ² was pseudo-replicated and its corrected permutation timed out on two brains.

### 20. (Priority 0.287 · Surprise 0.307) Omit errors are more frequent at branch points / on terminal (distal) branches than on linear/internal cable.
- **Sources:** ...2026-06-17 (id 11, topological distance-to-leaf; id 64, terminal vs internal omit rate), run-4--...2026-06-20 (id 32, branch vs linear node omit rate; id 85, terminal vs internal omit rate) — 4 entries, 2 reports.
- **Conclusion (canonical = run-4 id 32 / ...2026-06-17 id 64, SOUND/MINOR):** Omits concentrate at complex/distal topology. Branch nodes had an ~11.95% omit rate vs ~2.76% for linear nodes (>4×, χ²=1567.69, p<0.0001 — run-4 id 32). Terminal edges had a 4.38% omit rate vs 2.41% internal (~1.8×, χ²=4189.94, p≈0 — id 64 and run-4 id 85, identical numbers). id 11 found omit edges topologically closer to leaves (mean 247.18 vs 325.08 steps).
- **Agreement:** The categorical rate forms (id 32, id 64, run-4 id 85) GENERALIZE cleanly (terminal/branch omit rate higher on all brains; ratios attenuate to ~1.3–3.8× but stay highly significant).
- **Disagreement:** id 11 (continuous distance-to-leaf) is PARTIAL — the direction *flips* on ds_794491 (omit mean 159.76 > correct 138.76, one-sided p=1.0), even though it holds on origin and 794495. So the "omits are distal" claim is robust as a categorical terminal-vs-internal rate but unstable as a continuous distance-to-leaf metric.
- **Verdict carried over:** Best = SOUND (id 64); run-4 id 32 MINOR, run-4 id 85 MINOR; id 11 WEAK (huge-n, pseudo-replication, directional flip on 794491).
- **Caveats:** Edge/node pseudo-replication inflates the categorical p-values (immaterial at these effect sizes); id 11's distance is topological steps, not microns, with a full directional flip on one extra brain.

### 21. (Priority 0.287 · Surprise 0.307) Per-neuron split rates and omit rates are positively correlated (a shared failure tendency).
- **Sources:** ...2026-06-17 (id 30) — 1 report, but carried here as a corroborated-across-datasets finding rather than unique, since it GENERALIZES strongly on both extra brains.
- **Conclusion:** Across 12 neurons, splits/mm predicted omit rate with Pearson r = 0.65 (p = 0.022), Spearman ρ = 0.88 (p = 0.00015), OLS R² = 42.2%. The corrected Spearman permutation (p = 4.5999e-04, bootstrap CI [0.5620, 0.9857]) is robust. Neurons that fail in continuity also tend to miss structure entirely.
- **Agreement:** GENERALIZES — ds_794491 ρ=0.9833 (p=3.9999e-05), ds_794495 ρ=0.7404 (p=4.7999e-04); the borderline origin Pearson strengthens markedly on both extras.
- **Disagreement:** None across the datasets within this run; no other report tested the split–omit per-neuron correlation, so this could equally sit under unique findings. It is placed here because its multi-dataset corroboration is the salient point.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES; Verdict WEAK → Post-correction verdict UPHELD.
- **Caveats:** Only 12 neurons; original Pearson/OLS violated normality (Jarque-Bera p=0.00036) with a high-leverage point — the Spearman is the trustworthy headline; the "shared mechanism" wording is an interpretive over-reach the correlation cannot isolate.

### 22. (Priority 0.287 · Surprise 0.307) Inter-segment split gaps are tiny (≈4–20 µm; ~99% under 6.5 µm) and far exceed internal fragment edge lengths — motivating a tight repair search radius.
- **Sources:** ...2026-06-17 (id 35, median gap 19.7 µm vs 95th-pct internal 5.8 µm), run-4--...2026-06-20 (id 63, 100% of split gaps < 15 µm, 99th ~6.5 µm), run-5--...2026-06-25 (id 35, 86.77% of splits involve a sub-100 µm micro-fragment) — 3 entries, 3 reports.
- **Conclusion (canonical = run-4 id 63, SOUND):** Split gaps are uniformly small — all 6,805 split gaps < 15 µm, clustering 3–5 µm, with 95th/99th percentiles ~5.85 µm and ~6.50 µm, so a ~6.5 µm search radius catches ~99% of splits. The 2026-06-17 id 35 frames the same fact comparatively: median inter-segment gap 19.67 µm vs a 95th-pct intra-segment edge length of 5.77 µm (p≈0), so a radius wide enough for true gaps dwarfs internal edges and a fixed radius alone over-connects. Run-5 id 35 adds that 86.77% of splits involve a sub-100 µm micro-fragment.
- **Agreement:** All GENERALIZE (id 63: 99th ~6.4–6.5 µm on all three; id 35-17: median gap 18–30 µm everywhere; run-5 id 35: 75–87% micro-fragment involvement). This is the descriptive basis for Unique Finding #1's 6.84 µm classifier.
- **Disagreement:** None.
- **Verdict carried over:** Verdict SOUND (run-4 id 63, ...2026-06-17 id 35); run-5 id 35 SOUND.
- **Caveats:** The 2026-06-17 id 35 compares a median (gaps) against a 95th percentile (internal edges) — a deliberately conservative framing, not a like-for-like distribution test; run-5 id 35's 100 µm micro-fragment cutoff is a chosen threshold.

### 23. (Priority 0.287 · Surprise 0.307) Opposing endpoints at split gaps point at each other (anti-parallel, cosine −0.68 vs −0.51).
- **Sources:** ...2026-06-17 (id 67) — 1 report; corroborated across datasets within the run and conceptually related to run-4's angular-inertia finding (#8).
- **Conclusion:** Across 525 split endpoint pairs vs 2,689 spatially-adjacent control pairs, split endpoints had mean cosine similarity −0.6847 (more anti-parallel) versus −0.5112 for controls, significant by Mann-Whitney U (p = 1.1e-08). Geometric collinearity across the gap is a GT-free signature for proposing split-joins.
- **Agreement:** GENERALIZES — ds_794491 −0.5932 vs −0.3334 (p=1.4e-10); ds_794495 −0.7908 vs −0.5273 (p=8.6e-20). Run-4's id 33 angular-inertia result (#8) is a stronger, classifier-form expression of the same "directional continuity bridges splits" idea, but measures continuation-vs-false-candidate angle rather than endpoint cosine, so it is kept separate.
- **Disagreement:** None.
- **Verdict carried over:** Reproduction REPRODUCED; Generalization GENERALIZES; Verdict SOUND.
- **Caveats:** The control distribution has much wider variance (std 0.534 vs 0.358), so distributions overlap substantially — collinearity is a useful prior, not a perfect standalone classifier.

### 24. (Priority 0.287 · Surprise 0.307) Local tortuosity / curvature is statistically higher near splits but is a weak standalone predictor.
- **Sources:** run-4--...2026-06-20 (id 59, 10-edge window, point-biserial r=0.0335; id 73, 5-hop window), run-5--...2026-06-25 (id 36, 10 µm window) — 3 entries, 2 reports.
- **Conclusion (canonical = run-4 id 59, MINOR, includes effect size):** Split edges are more tortuous than correct edges but the effect is tiny. Run-4 id 59: median 1.1115 vs 1.0764, p≈0 over >1M edges, point-biserial r = 0.0335 (curvature explains ~0.1% of variance). id 73 replicates at a 5-hop window (median 1.1132 vs 1.0801, p=1.66e-276). Run-5 id 36: mean 1.0905 vs 1.0541 (t=14.27, p≈0) with Cliff's delta 0.2096 — "statistically real, practically weak."
- **Agreement:** All GENERALIZE in direction (r≈0.03–0.06 / Cliff's delta 0.19–0.26 on all brains); run-4 id 59 and id 73 are UPHELD under neuron-cluster permutation (p=0.002 and p=0.0008) with confirmed-small effects.
- **Disagreement:** None — all agree the effect is consistent but small; all reports correctly hedge it as a weak standalone predictor.
- **Verdict carried over:** Verdict MINOR (run-4 id 59, id 73; run-5 id 36) → Post-correction WEAKENED (run-5 id 36) / UPHELD-weak (run-4 id 59, id 73).
- **Caveats:** Significance is huge-n-driven; edges are non-independent; only the extreme-tortuosity tail is enriched among splits; run-4 id 73 and id 59 share most underlying data, so they are not independent confirmations of each other.

---

## Excluded as Redundant

Non-canonical entries folded into the consolidated findings above:

**Into Finding #14 (merge sites in dense, branchy neighborhoods):**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 23 → merged into #14 (canonical id 3)
- `ground-truth-error-annotations-revised-version_2026-06-17` id 81 → merged into #14
- `ground-truth-error-annotations-revised-version_2026-06-17` id 49 → merged into #14
- `ground-truth-error-annotations-revised-version_2026-06-17` id 53 → merged into #14
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 84 → merged into #14

**Into Finding #15 (merge segments are giant overgrown labels):**
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 34 → merged into #15 (canonical id 56 / run-4 id 43)
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 55 → merged into #15
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 24 → merged into #15 (super-merge sub-claim, overturned)

**Into Finding #16 (splits at/near branch points):**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 13 → merged into #16 (canonical id 10)
- `ground-truth-error-annotations-revised-version_2026-06-17` id 19 → merged into #16
- `ground-truth-error-annotations-revised-version_2026-06-17` id 72 → merged into #16
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 64 → merged into #16
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 33 → merged into #16

**Into Finding #17 (splits cluster into spatial "error zones"):**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 55 → merged into #17 (canonical run-5 id 26)
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 37 → merged into #17
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 40 → merged into #17
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 45 → merged into #17

**Into Finding #18 (splits favor thin distal/terminal processes):**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 65 → merged into #18 (canonical id 60)
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 38 → merged into #18
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 70 → merged into #18

**Into Finding #19 (omits are bursty/contiguous):**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 48 → merged into #19 (canonical run-5 id 27)
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 50 → merged into #19

**Into Finding #20 (omits at branch points / terminal branches):**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 11 → merged into #20 (canonical id 64)
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 32 → merged into #20
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 85 → merged into #20

**Into Finding #22 (split gaps tiny, motivating tight search radius):**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 35 → merged into #22 (canonical run-4 id 63)
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 35 → merged into #22

**Into Finding #24 (tortuosity weak split predictor):**
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 73 → merged into #24 (canonical run-4 id 59)
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 36 → merged into #24

**Into Finding #13 (Z-orientation not a consistent driver):**
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 21 → merged into #13 (with canonical id 27)

**Folded away as overturned/non-generalizing duplicates of the split/omit co-clustering theme (reported in the synthesis, not promoted to a standalone trustworthy finding):**
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 36 (omit cable closer to merge sites) → OVERTURNED under neuron-cluster permutation (p=0.4771); DOES-NOT-GENERALIZE. Not retained as a finding.
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 61 (split edges closer to merge sites) → OVERTURNED (cluster-permutation p=0.6469, CI crosses 0); DOES-NOT-GENERALIZE. Not retained as a finding.
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 42 (split→omit local co-occurrence) → DOES-NOT-GENERALIZE (sign flip on 794491, non-significant on 789202); Verdict MAJOR. Not retained as a trustworthy finding.
