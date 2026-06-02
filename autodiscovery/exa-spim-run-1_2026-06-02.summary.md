# AutoDiscovery Run Summary — exa-spim-run-1 (2026-06-02)

**Source CSV:** `exa-spim-run-1_2026-06-02.csv`
**Hypotheses ranked:** 49 of 50 (1 dropped for missing surprisal score)
**Surprisal range:** 0.041 – 0.812 (higher = more belief-shifting)
**Ordering:** descending Surprisal, ties broken by ID.

## Headline takeaways

This run probed the ExaSPIM U-Net neuron-reconstruction proofreading problem — what drives split and merge errors, and which local features can disambiguate true reconnections from false ones. The dominant theme is a string of **confidently-held assumptions that the data contradicted**: the run's most surprising results are all `Negative`-direction belief downgrades (`Likely True`/`Maybe True` → `Maybe False`). The single most belief-shifting result (rank 1, ID 45, surprisal 0.812) refuted the intuitive "super-fragment" hypothesis — the largest fragments hold *fewer* merge errors per unit length than their size predicts, so targeting big components is not the high-yield strategy assumed. Equally striking, several proposed geometric/morphological merge-detection heuristics (radius continuity, branch-angle sharpness, local density) repeatedly failed, and the anisotropic-Z-resolution theory of split direction (IDs 32, 14) was *reversed*: gaps lie in the higher-resolution X-Y plane, not along Z. A recurring, important caveat undercuts much of the geometric-feature work: spatial proximity alone (within 20–50 µm of ground truth) is such a strong predictor of true connections that experiments find only a handful of false positives (n=3–11), leaving most discriminative-feature tests badly underpowered.

On the constructive side, the lower-surprisal `Positive` results form a coherent, actionable picture: **merge and split correction are coupled** (severing merges unblocks valid split reconnections — IDs 5, 41), **tangent collinearity is a robust split-reconnection signature** (IDs 8, 27, 11), **radius gradient/variance flags merges and distinguishes split endpoints** (IDs 19, 46, 50), and **signal dropout drives large gaps** (ID 29). These confirmations were unsurprising (they raised belief without flipping it), which is exactly why they rank low.

## Ranked conclusions (most surprising first)

### 1. (Surprisal 0.812) Merge errors are NOT concentrated in the largest "super-fragments"
- **ID:** 45 · **Belief:** Likely True → Maybe False · **Direction:** Negative
- **Tested:** Whether the top 1% of U-Net fragments by cable length hold a disproportionate share of merge errors (which would justify prioritizing big fragments for merge detection).
- **Conclusion:** Across 10,172 components, the top 1% (102 fragments) held 15.66% of total cable length (2.94M/18.81M µm) but only 8.91% of merge errors (404/4,536). The binomial test gave p=1.00 — large fragments are *under*-represented in merge errors relative to their length. A confidently-held ROI assumption was reversed.
- **Caveats:** None noted; large sample, faithfully implemented.

### 2. (Surprisal 0.771) Baseline split error rate does NOT differ by human annotator
- **ID:** 36 · **Belief:** Likely True → Maybe False · **Direction:** Negative
- **Tested:** Whether U-Net split rate (fragments/mm) varies systematically with which annotator traced the ground-truth neuron.
- **Conclusion:** 8 annotators, mean split rates 0.144–0.226 splits/mm; one-way ANOVA F=0.554, p=0.778. No annotator bias — reassuring for metric robustness, but contradicts the prior expectation of systematic tracing differences.
- **Caveats:** Only 19 ground-truth neurons spread across 8 annotators; small per-group counts limit power to detect modest effects.

### 3. (Surprisal 0.771) A composite distance+tangent metric does NOT beat distance alone for reconnection
- **ID:** 37 · **Belief:** Likely True → Maybe False · **Direction:** Negative
- **Tested:** Whether combining Euclidean distance with terminal-tangent cosine similarity improves top-1 reconnection accuracy over distance alone in dense neighborhoods.
- **Conclusion:** Among 102 competitive origins (multiple candidates within 40 µm), both methods scored identical 98.04% top-1 accuracy (100/102); no discordant pairs, McNemar p=1.0. Within 40 µm, distance is already an overwhelming predictor, leaving no room for orientation to help.
- **Caveats:** Only 102 competitive cases; ceiling effect (98% baseline) leaves little measurable headroom.

### 4. (Surprisal 0.771) Radius difference does NOT separate true splits from false merges
- **ID:** 42 · **Belief:** Likely True → Maybe False · **Direction:** Negative
- **Tested:** Whether true split reconnections have smaller absolute neurite-radius differences than false merge proposals (radius continuity as a filter).
- **Conclusion:** 1,612 true splits (mean ~0.43 µm) vs 6 false merges (mean ~0.39 µm); Mann-Whitney p=0.769. No discriminative power; distributions heavily overlap.
- **Caveats:** Only 6 false merges — severely underpowered; conclusion is weak. (A prior float overflow bug was fixed by casting to float64.)

### 5. (Surprisal 0.730) Split gaps do NOT align with the low-resolution Z-axis — they lie in the X-Y plane
- **ID:** 32 · **Belief:** Likely True → Maybe False · **Direction:** Negative
- **Tested:** Whether anisotropic Z-resolution biases split gaps to align with Z more than random.
- **Conclusion:** 1,800 true-positive gaps vs 100,000 random vectors; K-S stat=0.080, p=2.7e-10 (significant) but in the *opposite* direction — mean |cos| to Z was 0.489 vs 0.501 random, with a large spike at 0.0 (gaps orthogonal to Z, i.e., flat in high-res X-Y). The resolution-driven-split theory is rejected.
- **Caveats:** None on power; the significance is genuine but contradicts the hypothesis.

### 6. (Surprisal 0.690) Longer-gap true splits show WEAKER collinearity, not stronger
- **ID:** 9 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether true splits over 10–20 µm gaps have stronger anti-parallel tangent alignment than those under 10 µm.
- **Conclusion:** Short-gap (n=288) mean collinearity −0.868 vs long-gap (n=107) −0.519; Welch t=−5.30, p=5.3e-07. Alignment *degrades* with distance — likely because neurites bend over longer spans, making local tangents unreliable for distant reconnections.
- **Caveats:** None major; reasonable cohort sizes.

### 7. (Surprisal 0.690) Local radius variance does NOT flag merge transition regions
- **ID:** 10 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether merge sites show higher local radius variance (±10-node window) than safe segments.
- **Conclusion:** 4,506 merge nodes (mean variance 0.00354 µm²) vs 4,506 safe nodes (0.00405 µm²); Welch t=−1.84, p=0.065. Trend is opposite and non-significant — not a reliable geometric merge flag. (Contrast with ID 19/50, which find a signal using radius *gradient*/whole-fragment variance.)
- **Caveats:** None on power; borderline p-value, but effect is in the wrong direction.

### 8. (Surprisal 0.690) Shorter fragments reconnect MORE often than long fragments
- **ID:** 12 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether short fragments (noisy terminal branches) reconnect less than long main-axon fragments.
- **Conclusion:** Short fragments reconnected 32.95% (747/2,267) vs long 11.91% (369/3,097); chi-square=350.3, p=3.7e-78. Reversed: short fragments are often heavily-split main-axon pieces that critically need reconnection, not noise.
- **Caveats:** None; large, highly significant.

### 9. (Surprisal 0.690) Gap distance is POSITIVELY correlated with neurite radius (thicker = larger gaps)
- **ID:** 13 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether thinner neurites suffer larger split gaps (negative radius-gap correlation).
- **Conclusion:** 529 mutually-closest true split pairs; Pearson r=+0.341 (p=7.8e-16), Spearman ρ=+0.287 (p=1.7e-11). Significant but reversed — thicker neurites have larger gaps, so dynamic search radii should expand for *thick* neurites.
- **Caveats:** None; moderate effect, well-powered.

### 10. (Surprisal 0.690) Reconnection displacement aligns LEAST with Z (X-axis dominant)
- **ID:** 14 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether split reconnection vectors align more with the low-res Z-axis (anisotropy theory, displacement-vector version).
- **Conclusion:** 1,860 split pairs; mean |components| X=0.535, Y=0.498, Z=0.454; ANOVA F=35.9, p=3.2e-16; Z significantly < X (p=4.1e-17) and < Y (p=3.9e-06). Strong directional bias exists but is opposite to hypothesis — corroborates ID 32.
- **Caveats:** Review field marked "N/A" (no audit recorded); result internally consistent with ID 32.

### 11. (Surprisal 0.690) Local fragment crowding does NOT drive merge errors
- **ID:** 16 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether 3D fragment density (50³ µm bins) correlates with merge-error incidence.
- **Conclusion:** 500,839 bins; Pearson r=−0.016 (p=1.3e-30), Spearman ρ=−0.017. Significance is purely a large-N artifact; effect size is essentially zero. Merges are driven by specific local morphology, not bulk crowding.
- **Caveats:** Only 28 merge fragments among 1,111 mapped — the merge signal is sparse; "significant" p-values are not meaningful here.

### 12. (Surprisal 0.690) Branch radius and branching angle do NOT flag merges
- **ID:** 21 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether merge-associated branch nodes (degree ≥3) have larger radius and more orthogonal angles than error-free branch nodes.
- **Conclusion:** 188 merge vs 19,272 valid branch nodes; radius 1.895 vs 1.917 µm, max angle 153.5° vs 153.3°; Mann-Whitney p=0.455 (radius), 0.911 (angle). No discriminative power.
- **Caveats:** Only 188 merge branch nodes, but means are near-identical so the null is well-supported.

### 13. (Surprisal 0.690) Gap-vs-radius scaling does NOT distinguish valid from invalid connections
- **ID:** 22 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether valid connections show a stronger gap-distance/radius-difference correlation than invalid ones.
- **Conclusion:** 1,391 valid (r=−0.100) vs 6 invalid (r=+0.748); Fisher z difference=−1.85, p=0.064. Valid pairs show essentially no scaling relationship; difference not significant.
- **Caveats:** Only 6 invalid pairs — severely underpowered; the n=6 correlation is unstable.

### 14. (Surprisal 0.690) Fragment size is NOT an added confidence prior for reconnection
- **ID:** 25 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether reconnections joining larger fragments are more likely true positives.
- **Conclusion:** 469 mapped pairs; precision by min-length quartile 98.3%/99.2%/100%/98.3%; Cochran-Armitage trend stat=0.038, p=0.846. Proximity alone is already overwhelming (464 TP vs 5 FP), leaving no size effect to detect.
- **Caveats:** Only 5 false positives — no variance to model size as a risk factor.

### 15. (Surprisal 0.690) Sharp skeleton turns do NOT indicate merge errors
- **ID:** 26 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether merged fragments have sharper maximum curvature (lower min angle) than valid fragments.
- **Conclusion:** 2,286 merged vs 7,886 valid; mean min angle 126.96° vs 125.78° (merged slightly *less* sharp); Mann-Whitney p≈1.0. Curvature is not a merge indicator.
- **Caveats:** None; well-powered, clear null.

### 16. (Surprisal 0.690) Local proposal density does NOT disambiguate true vs false candidates
- **ID:** 28 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether true reconnection candidates sit in lower-density neighborhoods than false ones at crowded endpoints.
- **Conclusion:** 87 true (mean density 2.36) vs 4 false (2.00); Mann-Whitney p=0.871 (one-sided), 0.267 (two-sided). No effect, and trend is opposite.
- **Caveats:** Only 4 false candidates — essentially no power.

### 17. (Surprisal 0.690) Node thickness at branch points does NOT flag merges
- **ID:** 30 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether branch nodes in merged components have larger node_radius than in correctly segmented ones.
- **Conclusion:** 786 merged vs 5,460 non-merged branch nodes; mean radius 1.952 vs 1.951 µm; Mann-Whitney p=0.454. No difference.
- **Caveats:** Review "N/A" (no audit recorded); means near-identical, null well-supported.

### 18. (Surprisal 0.690) Neurite tortuosity does NOT predict split density
- **ID:** 34 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether tortuous (twisty) ground-truth segments suffer more U-Net splits than straight ones.
- **Conclusion:** 14,129 segments; Spearman ρ=0.075 (p=6.9e-19) — significant but negligible, and the scatter is L-shaped: high split density occurs almost only in *straight* segments. Tortuosity is not the driver; thinness/SNR likely are.
- **Caveats:** Significance is a large-N artifact; effect size near zero.

### 19. (Surprisal 0.690) Cycle formation does NOT identify false merges in greedy correction
- **ID:** 39 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether false merges introduce topological cycles more than true splits (cycle detection as a rollback heuristic).
- **Conclusion:** 2,198 true splits vs 11 false merges; 24.43% of true splits formed cycles vs 9.09% of false merges; Fisher exact p=0.314. No useful dependence.
- **Caveats:** Only 11 false merges — underpowered; cycle detection looks unviable as a merge filter.

### 20. (Surprisal 0.690) Branching density does NOT correlate with split rate
- **ID:** 44 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether neurons with higher branch density (branches/mm) have higher split rates (fragments/mm).
- **Conclusion:** 19 neurons; Pearson r=−0.379, p=0.110. Non-significant, weak *negative* trend — branching complexity is not the main fragmentation driver. (Note: appears to conflict with the constructive ID 35/6 results that splits/merges cluster near branch points; the per-neuron aggregate here washes out the local effect.)
- **Caveats:** Only 19 neurons — low power; treat the null cautiously.

### 21. (Surprisal 0.690) Merge radius discontinuity is REVERSED — smooth radii flag merges, not sharp ones
- **ID:** 48 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether merge sites show sharper radius discontinuities than natural bifurcations.
- **Conclusion:** 477 merge vs 5,583 natural branch nodes; Welch t=−6.73, p=4.2e-11 (significant) but reversed — natural branches have *higher* variance (0.0215 vs 0.0110 µm²). Suspiciously smooth/uniform radius is the merge signature, not discontinuity.
- **Caveats:** None on power; the feature is discriminative but in the opposite direction hypothesized.

### 22. (Surprisal 0.641) False merges do NOT involve larger components than true splits
- **ID:** 43 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether false merge proposals connect larger fragments than true splits.
- **Conclusion:** 1,801 true splits (median 1,040 nodes) vs 11 false merges (median 786 nodes); Mann-Whitney p=0.476. Size is not a false-merge risk factor.
- **Caveats:** Only 11 false merges — underpowered.

### 23. (Surprisal 0.568) Nearest-neighbor is usually correct even in dense environments
- **ID:** 3 · **Belief:** Likely True → Maybe False · **Direction:** Negative
- **Tested:** Whether the spatially closest candidate is frequently NOT the true connection in dense regions.
- **Conclusion:** 57 dense cases; 1-NN correct in 94.74% (54/57); binomial test vs 90% gives p=0.372 (cannot reject 90%). 1-NN is robust, contradicting the assumed unreliability — though the missed 5% could matter at full-brain scale.
- **Caveats:** Only 57 dense cases; the practical-significance note (5% miss at scale) is worth heeding.

### 24. (Surprisal 0.495) Greedy cycles appear MOSTLY in non-merge components
- **ID:** 18 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether greedy reconnection introduces cycles predominantly into pre-existing merge components.
- **Conclusion:** 1,382 new edges, 99 cycle-containing components; 36% merge vs 64% non-merge; Fisher p=0.006 (significant association) but cycles are *not* predominantly in merge components — so cycles flag aggressive reconnection broadly, not merges specifically.
- **Caveats:** None major; nuanced result (association exists but majority are non-merges).

### 25. (Surprisal 0.495) False positives do NOT form cycles more than true positives
- **ID:** 24 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether false reconnections form topological cycles more than true ones (cycle as unsupervised filter).
- **Conclusion:** 651 TP (26.4% form cycles) vs 5 FP (0% form cycles); chi-square=0.685, p=0.408. No association; worse, 26% of *valid* connections form cycles, so a cycle filter would discard many true reconnections.
- **Caveats:** Only 5 false positives — severely underpowered. Review "N/A".

### 26. (Surprisal 0.495) Radius difference does NOT filter false reconnections
- **ID:** 33 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether valid reconnections have smaller radius differences than false ones (within 20 µm).
- **Conclusion:** 408 valid (mean 0.446 µm) vs 3 invalid (0.378 µm); Welch t=0.337, p=0.768. No discriminative power.
- **Caveats:** Only 3 invalid pairs — essentially no statistical power; conclusion is provisional.

### 27. (Surprisal 0.487) Splits do NOT concentrate at branch points (endpoint-mapping version)
- **ID:** 17 · **Belief:** Likely True → Maybe False · **Direction:** Negative
- **Tested:** Whether fragment endpoints map to ground-truth branching nodes more than the baseline morphological rate.
- **Conclusion:** Baseline 0.540% branch nodes; 29/3,965 endpoints (0.731%) mapped to branch nodes; chi-square goodness-of-fit=2.71, p=0.0995. Not significant — splits occur along paths roughly in proportion to baseline. (Note tension with ID 35's distance-based finding; the endpoint-mapping framing detects no enrichment.)
- **Caveats:** Borderline p; only 29 branch-mapped endpoints.

### 28. (Surprisal 0.446) "Hub" endpoints do NOT disproportionately produce false positives
- **ID:** 23 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether false reconnections originate from high-candidate-density "hub" endpoints.
- **Conclusion:** 397 TP (mean density 2.25) vs 5 FP (2.80); Mann-Whitney p=0.077. No significant effect.
- **Caveats:** Only 5 false positives — underpowered.

### 29. (Surprisal 0.446) Branching density does NOT drive split rate (per-neuron, replicate of ID 44)
- **ID:** 31 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether neurons with higher branching density have disproportionately higher split rates.
- **Conclusion:** 19 neurons; branching density 0.92–2.09/mm, split rate 0.11–0.33/mm; Pearson r=−0.414, p=0.078. Non-significant weak negative trend; morphological complexity is not the primary split driver.
- **Caveats:** Only 19 neurons — low power; consistent with ID 44.

### 30. (Surprisal 0.446) [CONFIRMED] Split endpoints taper more sharply than true terminations
- **ID:** 46 · **Belief:** Maybe True → Likely True · **Direction:** Positive
- **Tested:** Whether split-error endpoints show steeper radius gradients than natural biological terminations.
- **Conclusion:** 3,965 split endpoints (mean gradient +0.01223 µm/µm) vs 3,416 true terminations (−0.00041); two-sample t=48.35, p≈0. Split endpoints taper sharply (thickening toward root); gradient >~0.015 is a strong discriminator. A rare belief *upgrade* among the high-surprisal block.
- **Caveats:** Some overlap near zero; large, well-powered.

### 31. (Surprisal 0.446) End-to-branch reconnections dominate but are NOT larger gaps
- **ID:** 49 · **Belief:** Maybe True → Maybe False · **Direction:** Negative
- **Tested:** Whether a large fraction of splits need end-to-branch (not end-to-end) reconnection, and whether those gaps are larger.
- **Conclusion:** 1,388 reconnections: 63.6% end-to-branch, 36.4% end-to-end (first claim confirmed). But end-to-branch gaps averaged 25.3 µm (median 8.2) vs end-to-end 28.5 µm (median 13.3); Mann-Whitney p=0.854 — end-to-branch gaps are *not* larger (slightly shorter). Mixed result; overall downgraded.
- **Caveats:** None major; the topology-classification half is robust.

### 32. (Surprisal 0.284) [CONFIRMED] Low local tortuosity distinguishes true reconnections
- **ID:** 4 · **Belief:** Maybe True → Likely True · **Direction:** Positive
- **Tested:** Whether true reconnections create lower-tortuosity local paths than false ones.
- **Conclusion:** 479 TP (mean tortuosity 1.387) vs 5 FP (2.397); Mann-Whitney p=0.000255. Local tortuosity (arc-chord ratio) is a viable filter against artifactual sharp bends.
- **Caveats:** Only 5 false positives — significance holds but rests on a tiny negative class.

### 33. (Surprisal 0.284) [CONFIRMED] Severing merges unblocks hidden split reconnections
- **ID:** 5 · **Belief:** Maybe True → Likely True · **Direction:** Positive
- **Tested:** Whether correcting false merges exposes new valid split-reconnection endpoints.
- **Conclusion:** Severing 292 merge edges created endpoints yielding 574 new true-positive split proposals; 288/292 merge sites (98.63%) unblocked at least one split. Strong evidence that merge and split correction are coupled.
- **Caveats:** None; clear, large effect.

### 34. (Surprisal 0.284) [CONFIRMED] Merges occur in high-branching-density regions
- **ID:** 6 · **Belief:** Maybe True → Likely True · **Direction:** Positive
- **Tested:** Whether merge errors concentrate where local branching density is high.
- **Conclusion:** 505 merge nodes (mean branching density 0.62 within 50 µm) vs safe nodes (0.25); Welch t=7.24, p=9.9e-13. Merges occur disproportionately in dense, complex neuropil.
- **Caveats:** Merge nodes drawn from only 23 merged fragments; otherwise well-powered.

### 35. (Surprisal 0.284) [CONFIRMED] Fragmentation hubs have far higher reconnection rates
- **ID:** 15 · **Belief:** Maybe True → Likely True · **Direction:** Positive
- **Tested:** Whether high-endpoint-density regions have higher valid reconnection rates than isolated endpoints.
- **Conclusion:** High-density endpoints (n=1,195) reconnected 78.74% vs low-density (n=6,186) 20.93%; chi-square=1582, p≈0. Clustered endpoints are complex fragmentation hubs worth prioritizing.
- **Caveats:** Review "N/A"; effect is large and clear.

### 36. (Surprisal 0.284) [CONFIRMED] Radius gradient flags merge boundaries
- **ID:** 19 · **Belief:** Maybe True → Likely True · **Direction:** Positive
- **Tested:** Whether high radius-gradient/variance marks merge boundaries.
- **Conclusion:** 475 merge boundary nodes (mean gradient 0.0498 µm) vs 10,000 normal (0.0096); Mann-Whitney p=5.6e-44. Radius gradient is a strong local merge-flag feature. (Note: stronger and more direct than the variance-window framing in ID 10, which failed.)
- **Caveats:** None; large, well-powered.

### 37. (Surprisal 0.284) [CONFIRMED] Split errors cluster near branch points (distance version)
- **ID:** 35 · **Belief:** Maybe True → Likely True · **Direction:** Positive
- **Tested:** Whether leaf (split) nodes sit closer to branch points than typical middle nodes.
- **Conclusion:** Leaf nodes mean 217.4 µm to nearest branch vs middle nodes 266.9 µm; Mann-Whitney p=2.1e-173 (n=39,826 each). Splits cluster near junctions. (Reconciles partly with the ID 17/44/31 nulls: the local distance signal is real even though per-neuron aggregates and endpoint-mapping framings miss it.)
- **Caveats:** None; very large sample.

### 38. (Surprisal 0.284) [CONFIRMED null→stronger] Endpoint radius does NOT predict true vs false splits
- **ID:** 40 · **Belief:** Maybe False → Likely False · **Direction:** Negative
- **Tested:** Whether true-split leaf nodes have larger radii than false-split/artifact leaves.
- **Conclusion:** 2,687 true (mean 1.305 µm) vs 11 false (1.225 µm); Mann-Whitney p=0.453. Radius is not a confidence prior; belief moved further toward False (an expected, unsurprising confirmation).
- **Caveats:** Only 11 false splits — underpowered.

### 39. (Surprisal 0.284) [CONFIRMED] Resolving merges increases the recoverable split pool
- **ID:** 41 · **Belief:** Maybe True → Likely True · **Direction:** Positive
- **Tested:** Whether simulated merge correction increases the number of valid split proposals.
- **Conclusion:** Valid split proposals rose from 1,391 (baseline) to 2,023 after removing 4,536 merge edges; paired t=3.21, p=0.00147 over spatial chunks. Confirms merge/split coupling (companion to ID 5).
- **Caveats:** None; faithful implementation.

### 40. (Surprisal 0.203) Radius continuity does NOT disambiguate connections (proximity dominates)
- **ID:** 1 · **Belief:** Maybe True → Maybe True · **Direction:** Negative
- **Tested:** Whether radius difference is smaller for true than false reconnections.
- **Conclusion:** 464 TP (mean 0.447 µm) vs 5 FP (0.502 µm); Mann-Whitney p=0.646. No effect; belief unchanged (hence low surprisal).
- **Caveats:** Only 5 false positives — explicitly flagged as severely limiting power.

### 41. (Surprisal 0.203) [CONFIRMED] Tangent collinearity is a strong split signature
- **ID:** 8 · **Belief:** Likely True → Likely True · **Direction:** Positive
- **Tested:** Whether same-neuron endpoint pairs show higher anti-parallel collinearity than different-neuron pairs.
- **Conclusion:** 395 true splits (mean collinearity −0.775) vs 3 false (+0.718); Welch t=−10.35, p=0.0078. Tangent collinearity strongly discriminates; belief held (unsurprising confirmation).
- **Caveats:** Only 3 false pairs — significant but rests on a tiny negative class.

### 42. (Surprisal 0.203) Tangent agreement does NOT separate TP from FP within 50 µm
- **ID:** 47 · **Belief:** Maybe True → Maybe True · **Direction:** Negative
- **Tested:** Whether true reconnections show higher tangent cosine similarity than false within 50 µm.
- **Conclusion:** 629 TP (mean |cos| 0.684) vs 10 FP (0.637); Mann-Whitney p=0.222. No significant difference at this radius; belief unchanged.
- **Caveats:** Only 10 false positives; proximity already near-perfectly filters, leaving too few negatives.

### 43. (Surprisal 0.195) [CONFIRMED] Tangent cosine similarity discriminates recoverable splits
- **ID:** 27 · **Belief:** Likely True → Likely True · **Direction:** Positive
- **Tested:** Whether true reconnections preserve branch orientation (higher tangent cosine similarity) than false.
- **Conclusion:** 464 TP (mean 0.710) vs 5 FP (−0.480); two-sample t-test p≈0.013. Confirms tangent alignment as a discriminative feature; belief held.
- **Caveats:** Only 5 false positives.

### 44. (Surprisal 0.089) [CONFIRMED] Larger split gaps coincide with lower fluorescence (signal dropout)
- **ID:** 29 · **Belief:** Maybe True → Likely True · **Direction:** Positive
- **Tested:** Whether gap distance inversely correlates with raw image intensity at the gap midpoint.
- **Conclusion:** 200 sampled gaps; Spearman ρ=−0.350, p=3.7e-07. Larger gaps occur where fluorescence is weaker — signal dropout is a primary cause of large splits.
- **Caveats:** Sampled 200 of 1,801 gaps (network-latency constraint), but still adequately powered.

### 45. (Surprisal 0.073) [CONFIRMED] Tangent alignment is a robust geometric reconnection filter
- **ID:** 11 · **Belief:** Likely True → Likely True · **Direction:** Positive
- **Tested:** Whether true splits show higher tangent alignment (cos near −1) than false candidate pairs.
- **Conclusion:** 1,391 true splits (mean cos −0.705) vs 6 false (+0.194); Mann-Whitney p=0.0146. Proximity + tangent alignment is a robust filter; belief held.
- **Caveats:** Only 6 false pairs.

### 46. (Surprisal 0.057) Competitor density does NOT significantly separate true from false (directionally consistent)
- **ID:** 2 · **Belief:** Maybe True → Maybe True · **Direction:** Negative
- **Tested:** Whether true connections have lower proposal-degree centrality (fewer competitors) than false.
- **Conclusion:** 479 TP (mean burden 2.36) vs 5 FP (2.80); t-test p=0.422. Direction matches the hypothesis but is not significant; belief unchanged.
- **Caveats:** Only 5 false edges — underpowered.

### 47. (Surprisal 0.041) [CONFIRMED] Denser ground-truth packing correlates with more fragmentation
- **ID:** 7 · **Belief:** Maybe True → Maybe True · **Direction:** Positive
- **Tested:** Whether the U-Net-fragments-to-GT-neuron ratio rises with ground-truth cable density.
- **Conclusion:** 24,360 blocks (200³ µm); Pearson r=0.400, p<0.0001. Moderate positive correlation — density is a major but not sole driver (high-fragmentation low-density blocks exist). Belief unchanged (expected).
- **Caveats:** High variance; complexity is one of several drivers.

### 48. (Surprisal 0.041) [CONFIRMED] Soma distance separates split vs merge error locations
- **ID:** 38 · **Belief:** Maybe True → Maybe True · **Direction:** Positive
- **Tested:** Whether merge errors occur nearer the soma and splits farther away.
- **Conclusion:** 3,965 split vs 475 merge nodes; K-S stat=0.293, p=8.7e-33. Distance-to-soma significantly distinguishes the two; merges cluster locally (with a suspicious spike at exactly 3,000 µm suggesting a tiling/algorithmic boundary artifact). Belief unchanged.
- **Caveats:** Soma locations were *approximated* from node radii; the 3,000 µm spike hints at a methodological/tiling artifact worth investigating.

### 49. (Surprisal 0.041) [CONFIRMED] Merged fragments show higher radius variance than clean ones
- **ID:** 50 · **Belief:** Maybe True → Maybe True · **Direction:** Positive
- **Tested:** Whether merged fragments have higher skeleton radius variance than clean fragments.
- **Conclusion:** 28 merged (mean variance 0.00919) vs 1,083 clean (0.00792); one-sided Mann-Whitney p=0.0275. Statistically significant; whole-fragment radius variance is a viable merge signature. Belief unchanged (low surprisal). Note the apparent tension with ID 48 (ID 48 = ID 48 above) — this whole-fragment framing succeeds where the ±10-node window framing (ID 10) failed.
- **Caveats:** Only 28 merged fragments — modest power; p just under 0.05, so treat as suggestive rather than firm.

## Excluded (no surprisal score)

One hypothesis was dropped by the ranking helper for a missing Surprisal value:
- **ID 20** — not included in the ranking above.
