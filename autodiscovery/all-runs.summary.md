# AutoDiscovery Collective Summary — All Runs

- Source files (2): `exa-spim-run-1_2026-06-03.json` (n=50), `exa-spim-run-2_2026-06-03.json` (n=200)
- Total hypotheses ranked: **250** (dropped for missing surprisal: 0)
- Surprise magnitude range: **0.000 … 0.852**

## Headline synthesis

Across both runs (250 hypotheses pooled), the most belief-shifting findings are uniformly *negative*: the discovery loop went in with `Likely True` / `Leaning True` priors and watched the data overturn them. The single biggest flip (Run 2, hypothesis 111, surprise 0.852) reverses a confident assumption that thin neurites are the U-Net's weak point — thicker fragments actually have a higher split rate per millimeter. The next two flips (Run 1, hypotheses 45 and 36/37/42 at surprise ≈ 0.77–0.81) similarly reject the ideas that merge errors concentrate in giant 'super-fragments' and that annotator identity systematically biases error rates. A cross-cutting theme spans both runs: many morphology- or geometry-based heuristics that 'should' help proofreading agents (radius continuity, tangent-orientation rescoring within local neighborhoods, shatter-zone density, Z-axis alignment of gaps) fail to beat a plain Euclidean-distance baseline or fail to find the predicted spatial bias. Where evidence does support hypotheses, it is more often modest in magnitude (Z-scale belief shifts ≈ 0.0–0.3), consistent with the loop confirming things it already mostly believed. Many of the surprising rejections carry caveats — heavy class imbalance, tiny numbers of false-merge events, or chi-square tests dominated by near-zero positives — so several headline 'rejections' are really 'no statistically detectable effect with the data on hand,' not strong negative evidence.

## Ranked findings (most surprising first)

### 1. (Surprise 0.852) Rejected: U-Net fragments with a smaller mean neurite radius (thin structures like distal axons) suffer from a significantly higher split error rate (splits per millimete...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 111 · **Belief:** Likely True → Leaning False (0.875→0.328) · **Direction:** Negative · **Surprisal:** −0.852
- **Tested:** Investigate the relationship between the morphological thickness of a fragment and its propensity to be fragmented by the automated segmentation pipeline.
- **Conclusion:** The experiment successfully investigated the relationship between fragment thickness (neurite radius) and split error rate. Experiment Results: - Median Radius: 1.9537 µm - Thin Fragments (< median, n=5086): Mean split rate of 0.1036 splits/mm. - Thick Fragments (>= median, n=5086): Mean split rate of 0.1430 splits/mm.
- **Caveats:** none noted

### 2. (Surprise 0.812) Rejected: Aggressive merge errors create disproportionately large 'super-fragments'.
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 45 · **Belief:** Likely True → Leaning False (0.833→0.312) · **Direction:** Negative · **Surprisal:** −0.812
- **Tested:** To show that merge errors are heavily concentrated in a few massive connected components, implying that targeting the largest fragments for merge-detection will yield the highest return on investment.
- **Conclusion:** The analysis mapped UNet fragment nodes to Ground-Truth (GT) components to identify merge edges and computed the physical cable length of all 10,172 U-Net connected components. The top 1% of fragments (102 components) accounted for 15.66% of the total network cable length (2.94M / 18.81M µm). However, they only contained 8.91% (404 / 4,536) of the total merge errors.
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 3. (Surprise 0.771) Rejected: Human annotators exhibit systematically different tracing behaviors, resulting in statistically significant differences in the baseline split error rate (fragme...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 36 · **Belief:** Leaning True → Leaning False (0.792→0.297) · **Direction:** Negative · **Surprisal:** −0.771
- **Tested:** To determine whether U-Net reconstruction error metrics are inherently biased by the specific human annotator who traced the ground-truth morphology.
- **Conclusion:** The code accurately loaded the dataset, parsed annotator initials from the ground-truth SWC identifiers, and computed the split rate (U-Net fragments per mm of cable length) for each ground-truth neuron. Eight unique annotators were identified (JG, HP, IG, SP, JT, PP, MB, YZ), with mean split rates ranging from 0.1437 to 0.2264 splits/mm. A one-way ANOVA test applied to these groups returned an F-statistic of 0.5538 and a p-value of 0.7782.
- **Caveats:** none noted

### 4. (Surprise 0.771) Rejected: In dense local neighborhoods where a single leaf node has multiple reconnection candidates, a composite geometric metric combining Euclidean distance and termin...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 37 · **Belief:** Leaning True → Leaning False (0.792→0.297) · **Direction:** Negative · **Surprisal:** −0.771
- **Tested:** To evaluate the benefit of reasoning jointly over competing proposals using a combined spatial and angular cost function instead of independent thresholding.
- **Conclusion:** It evaluated tangent vectors for 39,826 leaves and found 102 'competitive' origins (leaves with multiple candidate connections within 40 µm and at least one true positive matching the ground truth). The results showed that ranking candidate connections by Euclidean distance alone achieved a top-1 accuracy of 98.04% (100/102). Ranking by the composite score (incorporating both distance and terminal tangent cosine similarity) yielded the exact same accuracy (98.04%, 100/102).
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 5. (Surprise 0.771) Rejected: True split reconnections between fragment endpoints exhibit significantly smaller absolute differences in neurite radius than false merge proposals.
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 42 · **Belief:** Leaning True → Leaning False (0.792→0.297) · **Direction:** Negative · **Surprisal:** −0.771
- **Tested:** To determine if local radius continuity is a viable and robust feature for filtering out false merge proposals during split-correction, while avoiding data type overflow errors.
- **Conclusion:** By casting the `node_radius` differences to `float64`, the numerical overflow issue was resolved, and valid statistical results were generated. The analysis found 1,612 True Splits (mean: ~0.43 µm) and only 6 False Merges (mean: ~0.39 µm). The Mann-Whitney U test yielded a statistic of 5170.5 and a p-value of 0.769, indicating no statistically significant difference between the absolute radius differences of the two groups.
- **Caveats:** extreme class imbalance / very few positives

### 6. (Surprise 0.730) Rejected: Due to the anisotropic resolution of the ExaSPIM imaging (0.748 × 0.748 × 1.0 µm), split gaps exhibit a directional bias, aligning more frequently with the lowe...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 32 · **Belief:** Leaning True → Leaning False (0.750→0.281) · **Direction:** Negative · **Surprisal:** −0.730
- **Tested:** To test if lower image resolution along the Z-axis disproportionately contributes to structural discontinuities (splits) in the automated reconstruction.
- **Conclusion:** The code located the dataset and identified 1,800 True Positive split reconnections (gap pairs). It calculated their alignment with the Z-axis and compared it to a uniform random distribution of 100,000 vectors. The Kolmogorov-Smirnov test yielded a statistically significant difference (K-S stat = 0.0800, p-value = 2.7339e-10).
- **Caveats:** none noted

### 7. (Surprise 0.730) Rejected: The frequency of split errors (measured as splits per 1,000 µm of cable length) is not uniform across a neuron's morphology; it is significantly higher in termi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 27 · **Belief:** Leaning True → Leaning False (0.750→0.281) · **Direction:** Negative · **Surprisal:** −0.730
- **Tested:** Determine if morphological context (terminal fine arbors vs.
- **Conclusion:** The programmer computed node betweenness centrality for each ground-truth neuron, classifying the top 20% of high-centrality nodes as the 'Main Trunk' and the remaining 80% as 'Terminal Arbors'. By mapping the U-Net fragments to these ground-truth annotations, the split error rates (splits per 1,000 µm of cable length) were calculated for both zones across 19 validated neurons. The results revealed an average split rate of 0.174 splits / 1,000 µm in the main trunks compared to 0.127 splits / 1,000 µm in the terminal arbors.
- **Caveats:** none noted

### 8. (Surprise 0.730) Rejected: Naive nearest-neighbor distance proposals frequently create topologically invalid small cycles, meaning intra-component reconnection proposals yield a massively...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 50 · **Belief:** Leaning True → Leaning False (0.750→0.281) · **Direction:** Negative · **Surprisal:** −0.730
- **Tested:** Evaluate the false positive rate of cycle-forming proposals to validate cycle-prevention as a crucial graph-state constraint in proofreading.
- **Conclusion:** The experiment successfully executed the steps to evaluate the false positive rate of cycle-forming (intra-component) naive reconnection proposals compared to inter-component proposals. Using a 25 µm Euclidean radius, 1,889 total naive proposals were generated from 39,826 fragment leaves. After mapping to the Ground Truth (within a 15 µm tolerance), 448 proposals were classified as Inter-component and 140 as Intra-component.
- **Caveats:** extreme class imbalance / very few positives

### 9. (Surprise 0.730) Rejected: Split errors are spatially clustered into 'shatter zones'; consequently, putative leaf-to-leaf reconnections proposed in regions with high local leaf density ha...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 85 · **Belief:** Leaning True → Leaning False (0.750→0.281) · **Direction:** Negative · **Surprisal:** −0.730
- **Tested:** Assess if local proposal density dynamically influences the reliability of reconnections, requiring the refresh of stale density features during iterative proofreading.
- **Conclusion:** The script properly loaded the dataset, identified putative leaf-to-leaf reconnections, computed the local leaf density within 50 µm for each pair, and performed the statistical comparison against the ground truth. Out of 1,397 putative reconnections found, 398 were confidently mapped to the ground truth. The results completely contradict the hypothesis: the 'Shatter Zone' cohort (>= 5 nearby leaves) yielded only 2 reconnections, both of which were True Positives, resulting in a 0.00% False Discovery Rate (FDR).
- **Caveats:** extreme class imbalance / very few positives; small sample size; test driven to p=1.0 by near-zero positive counts; implementation deviated from plan

### 10. (Surprise 0.730) Rejected: Shorter U-Net fragments (based on total cable length) are significantly more likely to produce false-positive split-correction proposals than longer fragments,...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 190 · **Belief:** Leaning True → Leaning False (0.750→0.281) · **Direction:** Negative · **Surprisal:** −0.730
- **Tested:** Evaluate whether the structural size (path length) of fragments involved in a proposal influences the likelihood of the reconnection being valid.
- **Conclusion:** The experiment successfully tested the hypothesis that shorter U-Net fragments are significantly more likely to produce false-positive split-correction proposals. The path lengths of all fragment components were calculated, and candidate leaf pairs within 15 µm were mapped to the ground truth to label them as true positive (TP) or false positive (FP). The minimum path length between the two connected fragments was then used as the sole predictor in a logistic regression model to predict the TP status.
- **Caveats:** implementation deviated from plan

### 11. (Surprise 0.690) Rejected: True split reconnections spanning longer spatial gaps (10-20 µm) exhibit significantly stronger geometric orientation agreement (anti-parallel collinearity) tha...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 9 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Quantify the trade-off between endpoint distance and orientation agreement to determine if distant reconnections require stricter geometric alignment to be valid.
- **Conclusion:** By analyzing 39,826 U-Net fragment leaf nodes and calculating tangent collinearity, the code evaluated 395 valid true split pairs. These were divided into a 'Short Gap' cohort (< 10 µm) containing 288 pairs and a 'Long Gap' cohort (10-20 µm) containing 107 pairs. The results directly contradicted the initial hypothesis.
- **Caveats:** none noted

### 12. (Surprise 0.690) Belief dropped: Merge transition regions (where a single U-Net fragment erroneously bridges two distinct GT neurons) exhibit significantly higher local variance in neurite radi...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 10 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Validate if structural radius asymmetry can be used as a purely geometric feature by a CNN to slide along skeletons and flag merge sites without relying on image patches.
- **Conclusion:** The experiment executed successfully and completed the evaluation of the proposed hypothesis. The code successfully mapped U-Net fragment nodes to ground-truth (GT) components to identify 4,506 unique merge transition nodes. It then compared the local radius variance (within a +/- 10 node window) of these merge sites against 4,506 randomly sampled 'safe' nodes from correctly unmerged fragments.
- **Caveats:** none noted

### 13. (Surprise 0.690) Rejected: Shorter U-Net fragments represent noisy or broken terminal branches and have a significantly lower probability of reconnecting to other fragments than longer, m...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 12 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** To assess if fragment cable length acts as a predictive feature for the likelihood of an endpoint participating in a valid topology-restoring connection.
- **Conclusion:** The experiment successfully calculated fragment cable lengths and categorized them into Short (< 3000 µm) and Long (> 10000 µm) buckets. By mapping the fragment leaf nodes to the Ground Truth (GT) skeleton, the script computed the reconnection rates for each category. The results show that endpoints of Short fragments reconnected in 32.95% of valid cases (747 out of 2267), whereas endpoints of Long fragments reconnected in only 11.91% of valid cases (369 out of 3097).
- **Caveats:** none noted

### 14. (Surprise 0.690) Rejected: The physical gap distance between matching split fragment endpoints (true splits) is negatively correlated with their average estimated neurite radius, indicati...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 13 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if the size of the missing cable (the gap) is systematically dependent on the thickness of the neurite, informing dynamic search radii for proposal generation.
- **Conclusion:** By robustly mocking the missing dependencies, the script loaded the dataset and identified 529 mutually-closest true split endpoint pairs within a 50 µm search radius. The correlation between the average neurite radius and the physical gap distance was computed, yielding a Pearson correlation coefficient (r) of +0.3406 (p-value = 7.8252e-16) and a Spearman correlation coefficient (rho) of +0.2872 (p-value = 1.6612e-11). These results are statistically significant but contradict the initial hypothesis.
- **Caveats:** none noted

### 15. (Surprise 0.690) Rejected: Due to the anisotropic resolution of the ExaSPIM imaging (lower resolution along the Z-axis), split errors are directionally biased, with valid reconnection dis...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 14 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if physical constraints of the imaging modality cause topological breaks to occur preferentially along the lower-resolution axis.
- **Conclusion:** ### Summary of Findings - Hypothesis: It was hypothesized that due to the lower-resolution Z-axis of ExaSPIM imaging, topological breaks (split errors) would be directionally biased, meaning valid reconnection displacement vectors would align more strongly with the Z-axis. - Results: The experiment identified 1,860 valid split pairs and computed their normalized displacement vectors. The mean absolute directional components were 0.5353 for the X-axis, 0.4976 for the Y-axis, and 0.4535 for the Z-axis.
- **Caveats:** none noted

### 16. (Surprise 0.690) Rejected: The local 3D spatial density of U-Net fragments is strongly positively correlated with the incidence of merge errors, indicating that crowding of distinct neuri...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 16 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Evaluate the relationship between local fragment crowding (density) and the volumetric occurrence rate of merge errors.
- **Conclusion:** The script binned the nodes of the fragments graph into 50x50x50 µm volumetric bins and quantified the local density of total unique fragments versus the density of unique merge fragments (fragments mapped to multiple ground-truth components). ### Summary of Findings - Hypothesis: It was hypothesized that local 3D spatial density of U-Net fragments is strongly positively correlated with the incidence of merge errors (fusions), assuming that physical crowding of neurites drives segmentation mistakes. - Results: The analysis evaluated 500,839 populated spatial bins.
- **Caveats:** none noted

### 17. (Surprise 0.690) Rejected: Branching nodes (degree >= 3) in the U-Net fragments that are associated with merge errors exhibit a significantly higher average neurite radius and more orthog...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 21 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** To determine if local skeleton geometry (radius and branching angle) can be used as a heuristic to detect false U-Net merges at branch points.
- **Conclusion:** The code identified 19,460 branching nodes in the fragment graph. By matching the 10-node neighborhood to the ground truth, 188 branching nodes were classified as 'merges', while 19,272 were classified as 'valid'. The descriptive statistics showed almost identical properties between the two groups: the mean radius was 1.8954 µm for merges vs.
- **Caveats:** none noted

### 18. (Surprise 0.690) Rejected: Among candidate reconnection pairs, valid connections (same GT neuron) exhibit a significantly stronger correlation between their spatial gap distance and their...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 22 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Test if true structural gaps maintain a morphological scaling relationship (where larger gaps correlate with expected radius tapering) that is absent in false connections.
- **Conclusion:** The script successfully identified candidate reconnection pairs (fragment leaves within a 20 µm radius), categorizing them into 1,391 valid pairs (sharing the same ground-truth ID) and 6 invalid pairs. Valid connections exhibited a weak negative correlation (r = -0.1001), while invalid connections showed a strong positive correlation (r = 0.7481), though the latter group had a very small sample size (n=6). Fisher's z-transformation yielded a z-score difference of -1.8498 and a p-value of 0.0643.
- **Caveats:** small sample size

### 19. (Surprise 0.690) Belief dropped: Reconnection proposals connecting larger fragments (greater total cable length) are significantly more likely to be true positives than proposals involving shor...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 25 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if fragment size acts as a reliable confidence prior for split correction, which would justify prioritizing long-fragment merges.
- **Conclusion:** The experiment successfully executed and evaluated the hypothesis. Out of 39,826 leaf nodes, 2,209 pairs were identified within a 30 µm radius, and 469 of these pairs were successfully mapped to ground-truth traces (distance < 10 µm). The minimum fragment cable lengths of these pairs were divided into quartiles, yielding remarkably consistent and high precision across all bins: Q1 (98.31%), Q2 (99.15%), Q3 (100.00%), and Q4 (98.29%).
- **Caveats:** none noted

### 20. (Surprise 0.690) Rejected: U-Net fragments containing merge errors possess significantly sharper maximum curvatures (lower minimum angles between adjacent edges) than valid fragments, ind...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 26 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Assess if the presence of sudden, sharp turns in the skeleton geometry of a U-Net fragment is a reliable indicator of an underlying merge error.
- **Conclusion:** The experiment successfully tested the hypothesis that U-Net fragments containing merge errors possess significantly sharper maximum curvatures (lower minimum angles between adjacent edges) than valid fragments. The script accurately loaded the dataset, matched fragment components to ground-truth traces to classify them as 'merged' or 'valid', and calculated the minimum turn angle (sharpest turn) for each component. The results evaluated 2,286 merged components and 7,886 valid components.
- **Caveats:** none noted

### 21. (Surprise 0.690) Rejected: In local neighborhoods where a leaf node has multiple competing reconnection candidates within 20 µm, true candidates exhibit a significantly lower local propos...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 28 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** To determine if the local density of fragment endpoints provides contextual evidence to disambiguate true from false reconnections at crowded endpoints.
- **Conclusion:** The experiment successfully tested the hypothesis by calculating and comparing the local proposal densities of 'True' and 'False' candidate leaves in crowded local neighborhoods. The results yielded 87 True candidates and 4 False candidates under the specified criteria (>=2 leaves within 20 µm, and high-confidence GT mapping). The True candidates exhibited a slightly higher mean density (2.36) compared to the False candidates (2.00).
- **Caveats:** none noted

### 22. (Surprise 0.690) Rejected: Branching nodes (degree >= 3) in the U-Net fragments graph that belong to incorrectly merged components (spanning multiple ground-truth neurons) exhibit a signi...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 30 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** To evaluate whether local node thickness at branch points can serve as a topological flag for detecting merge errors.
- **Conclusion:** The experiment successfully tested the hypothesis by analyzing the local node thickness (`node_radius`) of branching nodes (degree >= 3) in merged versus non-merged components of the U-Net fragments graph. The code mapped fragment nodes to ground-truth nodes (within a 10 µm confidence threshold) to determine which fragments spanned multiple GT neurons (merged) versus single GT neurons (non-merged). The results yielded 786 branching nodes in merged components and 5,460 in non-merged components.
- **Caveats:** none noted

### 23. (Surprise 0.690) Rejected: Ground-truth neurite segments with high geometric tortuosity (curvature/twistedness) suffer from a significantly higher density of U-Net split errors compared t...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 34 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate the relationship between local neurite morphology (tortuosity) and the susceptibility of the automated segmentation to fragment the neurite.
- **Conclusion:** The experiment successfully analyzed the relationship between neurite tortuosity and U-Net split density to test the hypothesis. The script decomposed the ground-truth (GT) graph into 14,901 unbranched segments, filtering out those shorter than 20 µm to reduce noise, resulting in 14,129 valid segments for analysis. The Spearman rank correlation test yielded a correlation coefficient (rho) of 0.0746 with a highly significant p-value of 6.9263e-19.
- **Caveats:** none noted

### 24. (Surprise 0.690) Rejected: In a greedy split-correction strategy, false-positive connections (merges) are significantly more likely to introduce topological cycles than true-positive conn...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 39 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** To evaluate the viability of using topological cycle formation as a rollback mechanism for detecting early, high-confidence merge errors.
- **Conclusion:** The experiment executed successfully and completed the required steps to test the hypothesis. The script loaded the dataset using the mock unpickler, identified 2,209 proposed edges (pairs of fragment leaves within 30 µm), and categorized them into 2,198 True Splits and 11 False Merges. Applying a greedy distance-based strategy with a Union-Find data structure to track cycle formation, it found that 24.43% of True Splits caused topological cycles, compared to only 9.09% of False Merges.
- **Caveats:** none noted

### 25. (Surprise 0.690) Rejected: Neuron morphological complexity drives U-Net fragmentation: there is a significant positive correlation between a ground-truth neuron's branching density (branc...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 44 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** To investigate if the automated segmentation algorithm systematically struggles with highly branched morphologies, leading to a proportionally higher rate of splits.
- **Conclusion:** The programmer properly mocked the necessary dependencies to load the dataset and calculated the branching density and split rate for the 19 ground-truth neurons. A Pearson correlation test yielded an r-value of -0.3789 and a p-value of 0.1096. This result does not support the hypothesis; instead of a significant positive correlation, the data shows a non-significant, weak-to-moderate negative correlation between morphological complexity (branching density) and the U-Net fragmentation (split rate).
- **Caveats:** none noted

### 26. (Surprise 0.690) Rejected: Artificial merge errors exhibit sharp, unnatural discontinuities in predicted neurite radius at the exact point of fusion, distinguishing them from natural biol...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 48 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** To verify if local radius variance can serve as a sliding-window feature for an agent to autonomously detect merge sites.
- **Conclusion:** The script identified 477 merge boundary nodes and 5,583 natural branching nodes, computed the radius variance over a local 5-node window for each, and performed a Welch's t-test. The results yielded a t-statistic of -6.7334 and a highly significant p-value of 4.19e-11, indicating a strong statistical difference. However, the findings contradict the initial hypothesis: natural biological branches actually exhibit a significantly higher mean local radius variance (0.0215 µm²) than artificial merge errors (0.0110 µm²).
- **Caveats:** none noted

### 27. (Surprise 0.690) No effect found: False merge errors exhibit branch angles closer to 90 or 180 degrees (orthogonal or crossing fibers), whereas true biological bifurcations exhibit more acute, Y...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 4 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Assess if local skeleton branching geometry can differentiate true axonal bifurcations from U-Net merge artifacts where fibers merely cross paths.
- **Conclusion:** The pipeline identified 5,571 True Branches and 38 Merge Errors based on distance and component ID mappings to the ground truth. The Mann-Whitney U test yielded a p-value of 0.1835 (p > 0.05), indicating that there is no statistically significant difference in the maximum branch angle distributions between the two classes. The mean maximum branch angle for True Branches was roughly 146.65 degrees, compared to 140.33 degrees for Merge Errors, with closely aligned medians (~147.4 vs.
- **Caveats:** none noted

### 28. (Surprise 0.690) Belief dropped: Different human annotators exhibit systematic biases in their tracing behavior; the ground truth neurons traced by different annotators show statistically signi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 12 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Evaluate annotator-specific morphological biases in the ground truth skeletons.
- **Conclusion:** Eight unique annotators were identified across the 19 GT components: JG (N=4), HP (N=2), IG (N=2), SP (N=3), JT (N=1), PP (N=4), MB (N=1), and YZ (N=2). The Kruskal-Wallis H-test was performed to determine if there were statistically significant differences in branching density and baseline normalized ERL across these annotators. - For Branch Density, the test yielded an H-statistic of 6.5158 with a p-value of 0.4810.
- **Caveats:** small sample size

### 29. (Surprise 0.690) No effect found: The absolute difference in estimated neurite radius (`node_radius`) between two fragment endpoints is significantly smaller for valid reconnections (same ground...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 16 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Assess if neurite thickness continuity is a reliable morphological feature to distinguish true splits from false merges during reconnection proposal filtering.
- **Conclusion:** The experiment successfully tested the hypothesis regarding neurite thickness continuity. It isolated 39,826 leaf nodes from the fragment graph, found 1,618 pairs within a 20 µm radius, and classified inter-component pairs into 408 'Valid' and 989 'Invalid' reconnections based on ground-truth mapping. The analysis showed that the mean radius difference for valid pairs was 0.4455 µm (median 0.3984 µm) and for invalid pairs was 0.4669 µm (median 0.4238 µm).
- **Caveats:** none noted

### 30. (Surprise 0.690) Rejected: The physical gap distance between fragment endpoints that belong to the same true neuron is positively correlated with the local spatial density of unrelated fr...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 18 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Analyze how local reconstruction density impacts the severity (gap size) of split errors.
- **Conclusion:** The code successfully identified 1,612 valid split pairs mapping fragment leaf nodes to ground-truth neurons. The analysis yielded a strong negative correlation between the split gap distance and the local fragment density (Pearson: -0.7679, Spearman: -0.7830), with p-values indicating strong statistical significance. The binned summary statistics show that in areas of low density (0-2 distinct fragments), the mean gap size is large (~103.91 µm).
- **Caveats:** none noted

### 31. (Surprise 0.690) Rejected: True biological bifurcations are largely co-planar (Y-shaped), whereas U-Net merge errors (crossing fibers) exhibit significantly higher 3D non-coplanarity.
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 19 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Use 3D geometry to distinguish true biological branching from artificial merges by measuring the tetrahedral volume formed by the branch point and its divergent branches.
- **Conclusion:** The experiment successfully extracted degree-3 branch points from the skeleton fragments and classified them into True Branches (5,481) and Merge Errors (36). The 3D non-coplanarity was evaluated by computing the parallelepiped volume (scalar triple product of diverging branch vectors). Statistical analysis using the Mann-Whitney U test yielded a significant p-value (6.33e-05 < 0.05), indicating a distinct geometric difference between the two classes.
- **Caveats:** extreme class imbalance / very few positives

### 32. (Surprise 0.690) No effect found: False-positive split reconnections occur in significantly denser local graph environments (higher number of nearby fragment endpoints) compared to true-positive...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 24 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Evaluate if local leaf node density is a statistically significant spatial feature for distinguishing true reconnections from false ones.
- **Conclusion:** The code successfully loaded the dataset by employing a resilient mocking strategy to handle unpickling. It isolated the leaf nodes from the fragment graph, identified 1,397 inter-component pairs within a 20 µm radius, and classified them using the ground truth graph into 408 Valid and 989 Invalid pairs. The calculation of the local density (number of leaf nodes within a 30 µm radius of the proposal midpoint) yielded the following results: - Valid Reconnections: Mean local density = 2.15, Median = 2.00 - Invalid Reconnections: Mean local density = 2.12, Median = 2.00 A Mann-Whitney U test was performed to compare these distributions, resulting in a p-value of approximately 0.8684.
- **Caveats:** small sample size

### 33. (Surprise 0.690) No effect found: Valid reconnections between split fragment endpoints have a significantly shorter Euclidean gap distance than invalid reconnections within a local 20 µm search...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 25 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if simply prioritizing the closest absolute endpoints provides a statistically favorable baseline for true reconnections.
- **Conclusion:** Summary of Results: - Valid Pairs (True Reconnections): 350 pairs were identified, with a mean gap distance of 6.9415 µm (median: 5.0990 µm, std dev: 4.9400 µm). - Invalid Pairs (False Reconnections): 1047 pairs were identified, with a mean gap distance of 7.4942 µm (median: 5.0990 µm, std dev: 5.2948 µm). - Statistical Significance: The Mann-Whitney U test yielded a p-value of 0.1971.
- **Caveats:** small sample size; implementation deviated from plan

### 34. (Surprise 0.690) No effect found: The density of split errors (number of U-Net fragments per 1,000 µm of ground-truth cable) is significantly higher within a 150 µm spatial radius of the soma th...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 30 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Assess whether split errors are disproportionately concentrated near the soma due to complex morphology challenging the segmentation model.
- **Conclusion:** The experiment executed successfully and achieved its objective of comparing split error densities between proximal (<= 150 µm from the soma) and distal (> 150 µm) regions of the neurons. The results revealed that the split rate in the proximal region is 0.20 fragments per 1,000 µm (6 fragments over 29,358.42 µm of cable), while the split rate in the distal region is also 0.20 fragments per 1,000 µm (1105 fragments over 5,534,619.00 µm of cable). The Poisson rate test produced a p-value of 0.9545, concluding that there is no statistically significant difference in split error density between the two regions.
- **Caveats:** none noted

### 35. (Surprise 0.690) Rejected: The rate of split errors (number of U-Net fragments per unit length of ground-truth neuron) increases significantly as the topological distance from the soma in...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 32 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Analyze the spatial distribution of split errors relative to their topological distance from the soma.
- **Conclusion:** The topological distance for over 1.36 million GT nodes across 19 components was computed and binned into 500 µm intervals. In total, over 5.56 meters of cable length and 8,671 split boundaries were analyzed. The statistical analysis revealed a moderate, but highly significant, negative correlation between topological distance from the proxy soma and the split error rate (Pearson r = -0.4082, p = 5.92e-05).
- **Caveats:** none noted

### 36. (Surprise 0.690) No effect found: U-Net fragments containing merge errors possess significantly higher tortuosity than structurally correct fragments of matched cable lengths, because the errone...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 33 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if elevated tortuosity serves as a geometry-only signature for detecting merged fragments.
- **Conclusion:** The experiment successfully executed and tested the hypothesis that U-Net fragments containing merge errors possess significantly higher tortuosity than structurally correct (Clean) fragments of similar sizes. The code reliably identified 20 Merge fragments and 981 Clean fragments, subsampling the clean set to match the cable length distribution of the merge set (19 matched pairs). The statistical analysis (Welch's t-test) resulted in a p-value of 0.3591, indicating no statistically significant difference in tortuosity between the two groups.
- **Caveats:** none noted

### 37. (Surprise 0.690) Belief dropped: Short U-Net fragments (1000 µm to 5000 µm) that are orphaned (do not map to any ground-truth component) exhibit significantly higher path tortuosity than short...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 35 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if geometric path tortuosity can identify and filter out short, false-positive noise fragments without relying on image data.
- **Conclusion:** The dataset was loaded, and short fragments (1,000 to 5,000 µm) were extracted and classified based on their mapping to the ground truth. The analysis identified 658 mapped fragments and 6,399 orphaned fragments. Contrary to the initial hypothesis, the results demonstrate that mapped short fragments (true neurites) have a higher mean path tortuosity (1.8281) compared to orphaned short fragments (1.6119).
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 38. (Surprise 0.690) Rejected: U-Net false merges create topological 'blocks' that artificially widen the spatial gap between true biological continuations, resulting in split-error gaps near...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 37 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if un-merging is a necessary prerequisite for successful split correction by measuring how merges distort the proximity of valid reconnection proposals.
- **Conclusion:** The experiment successfully executed the data pipeline and statistical analysis to evaluate whether U-Net false merges artificially widen the spatial gaps between true biological continuations (True Splits). Out of 1,256 identified true splits, 89 were classified as 'Merge-Adjacent' (within 50 µm of a false merge along the skeleton) and 1,167 as 'Isolated'. The mean gap distance for Merge-Adjacent splits was 83.85 µm, while Isolated splits averaged 81.81 µm.
- **Caveats:** none noted

### 39. (Surprise 0.690) Rejected: Pairs of fragment leaf nodes that are spatially close (<15 µm) and have a similar local neurite radius (difference < 15%) are significantly more likely to belon...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 46 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if radius continuity can be used to dramatically improve the precision of reconnection proposals between nearby leaf nodes during split correction.
- **Conclusion:** The code loaded the dataset, extracted 39,826 leaf nodes, and identified 1,209 candidate leaf pairs from different components within a 15 µm distance. These pairs were mapped to the ground truth and classified based on their radius differences into 'Similar Radius' (< 15% difference) and 'Divergent Radius' (>= 15% difference). The analysis revealed that spatial proximity alone is a highly robust predictor for true connections: the Similar Radius group had a 98.70% true connection rate (76/77), and the Divergent Radius group had a 99.15% true connection rate (232/234).
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 40. (Surprise 0.690) Belief dropped: Valid reconnections between split fragment endpoints are significantly more likely to be mutually closest neighbors in 3D space than invalid reconnections.
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 48 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Test if symmetric nearest-neighbor relationships provide a higher precision baseline for valid reconnections than raw distance thresholds.
- **Conclusion:** The experiment successfully tested the hypothesis that mutually closest neighbors in 3D space yield a higher precision baseline for valid reconnections. The code extracted 39,826 leaf nodes, computed their absolute closest inter-component neighbors using a KD-tree, and grouped pairs into 'Mutual' (symmetric nearest neighbors) and 'Asymmetric' groups. Results & Findings: - Mutual Pairs: 9,690 total pairs identified, with 1,125 being valid (Precision: 11.61%).
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 41. (Surprise 0.690) Rejected: U-Net fragments containing at least one merge error have a significantly higher probability of also terminating in a false split error compared to topologically...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 53 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate the spatial coupling of errors to see if densely packed neurite regions cause the U-Net to simultaneously fuse and prematurely terminate segmentations.
- **Conclusion:** The experiment successfully executed and investigated the spatial coupling of merge and split errors in U-Net fragments. The dataset yielded 990 pure fragments and 12 merged fragments. Out of these, 965 pure fragments and 12 merged fragments possessed mappable endpoints for evaluation.
- **Caveats:** none noted

### 42. (Surprise 0.690) No effect found: U-Net fragment endpoints (leaves) that represent 'recoverable true splits' (capable of correctly rejoining a single GT neuron) exist in regions of significantly...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 56 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate whether localized crowding of fragment endpoints is a reliable meta-feature to flag ambiguous/risky reconnections that could cause merge errors.
- **Conclusion:** Out of 39,826 fragment leaf nodes, 36,269 were isolated (local density 0) within a 25 µm radius. The remaining leaves were classified into 961 True candidates and 2,596 Ambiguous/False candidates. The mean local densities for the True and Ambiguous/False groups were nearly identical (1.0708 vs.
- **Caveats:** none noted

### 43. (Surprise 0.690) Belief dropped: Endpoints of U-Net fragments that represent false splits (i.e., the neuron actually continues in the ground truth) have significantly higher local tortuosity (c...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 59 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if highly tortuous/curved neurite segments are more prone to causing U-Net split errors compared to straight segments.
- **Conclusion:** The code successfully isolated 39,826 fragment leaf nodes, of which 7,297 were mapped to a ground truth node within a 10 µm radius. The traversal logic accurately calculated the tortuosity for the last 25 µm of path length leading to these nodes. The classification yielded 3,907 False Splits (mean tortuosity = 1.057 +/- 0.085) and 3,390 True Terminations (mean tortuosity = 1.069 +/- 0.094).
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 44. (Surprise 0.690) Belief dropped: Small fragment components (< 5000 µm in cable length) act as critical 'missing link' chains in split correction; they are significantly more likely to require m...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 63 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Analyze the topological role of short fragments in the true split proposal graph to justify iterative, state-aware proofreading.
- **Conclusion:** The code identified 1,397 inter-component leaf pairs within 20 µm, filtering them down to 408 true split proposals verified against the ground truth. From these true proposals, a proposal graph comprising 508 components was built. Components were categorized by cable length into 'Small' (< 5000 µm, N=384) and 'Large' (>= 5000 µm, N=124).
- **Caveats:** none noted

### 45. (Surprise 0.690) Rejected: Reconnections between fragment endpoints that result in the creation of a topological branch where the newly formed minor branch is exceptionally short (< 25 µm...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 66 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Test if the length of newly formed branches acts as a robust post-hoc heuristic to detect and roll back aggressive false reconnections (reversible decisions).
- **Conclusion:** The experiment successfully executed and tested the proposed hypothesis. Putative reconnections were generated between fragment leaves and nearby internal nodes (< 15 µm). Out of 4,365 putative reconnections, 1,034 were confidently mapped to the ground truth and evaluated.
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 46. (Surprise 0.690) Rejected: The distribution of spatial gap distances for valid splits is clustered in chains, meaning that virtually applying a single pass of high-confidence short-range...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 68 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Simulate the effect of a multi-pass, agentic proofreading strategy that dynamically updates graph topology, demonstrating that resolving small gaps makes larger disconnected networks easier to solve.
- **Conclusion:** The experiment successfully simulated a multi-pass proofreading strategy by identifying valid split pairs and virtually reconnecting those separated by less than 5 µm. The code identified 181 pairs of valid endpoints separated by < 5 µm and added edges to join them, updating the graph topology. Statistical analysis shows that the virtual reconnections successfully eliminated the smallest gaps, reducing the number of valid pairs ≤ 20 µm from 395 (baseline median = 5.20 µm) to 202 (new median = 9.90 µm).
- **Caveats:** small sample size

### 47. (Surprise 0.690) Rejected: Branch nodes in the automated reconstruction where the outgoing minor branches form a highly acute angle (maximum cosine similarity > 0.8) are significantly mor...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 69 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate if acute branch angles serve as an identifiable geometric signature of U-Net merge errors where two parallel passing neurites were wrongly fused.
- **Conclusion:** The script identified 19,438 degree-3 branch nodes within the automated UNet fragments, successfully mapping 5,600 of them to the human-traced ground truth within a reliable 15 µm threshold. The results directly evaluate the hypothesis regarding acute branching patterns and merge errors: - Acute branches (max cosine similarity > 0.8): Out of 122 mapped nodes, only 2 were merge errors, resulting in a False Branch Rate (FBR) of 1.64%. - Non-Acute branches: Out of 5,478 mapped nodes, 33 were merge errors, resulting in an FBR of 0.60%.
- **Caveats:** none noted

### 48. (Surprise 0.690) Rejected: U-Net merge errors (fragments that incorrectly fuse multiple ground-truth neurons) are disproportionately located in spatial regions with a high density of uniq...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 70 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if local fragment density is a strong predictor of merge errors in the automated reconstruction.
- **Conclusion:** The script correctly mapped 23 U-Net fragments that falsely joined multiple ground-truth components and successfully identified 484 putative merge nodes where the ground-truth mapping switched. By querying a 30 µm radius around these merge nodes and comparing them to 484 randomly sampled non-merge nodes, the script calculated local component densities. The results showed that the mean local component density at merge sites (1.10 components) was virtually identical to the density at non-merge sites (1.12 components).
- **Caveats:** none noted

### 49. (Surprise 0.690) Rejected: Long U-Net fragments (total cable length > 5000 µm) are significantly more 'pure' (measured by the percentage of their mapped nodes belonging to a single majori...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 75 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if fragment length correlates with component purity to establish whether long fragments can be safely frozen as reliable anchors in an agentic workflow.
- **Conclusion:** The analysis identified 185 mapped 'Long' fragments (>5000 µm) and 703 mapped 'Short' fragments (1000-5000 µm). Contrary to the hypothesis, the results show that Short fragments are slightly but significantly more pure than Long fragments (Mean Purity: 0.9986 vs. 0.9863).
- **Caveats:** none noted

### 50. (Surprise 0.690) Rejected: Fragments containing merge errors are structurally more complex than clean fragments, exhibiting a significantly higher ratio of branching nodes (degree >= 3) t...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 77 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if the global topological complexity of a fragment (branch density) is predictive of it containing a merge error.
- **Conclusion:** The code accurately mapped fragment components to Ground Truth neurons to classify 2,119 fragments as 'Merge' and 8,053 as 'Clean' (after filtering out fragments with fewer than 10 nodes and using a 5% node mapping threshold). The results show that the Mean Branch Density for Merge fragments is ~0.0032, while for Clean fragments it is ~0.0042. The Mann-Whitney U test indicates a highly significant difference between the two distributions (p-value = 5.03e-13), and the point-biserial correlation reveals a slight negative correlation (-0.079, p-value = 1.44e-15) between a fragment's branch density and its probability of containing a merge error.
- **Caveats:** none noted

### 51. (Surprise 0.690) No effect found: Invalid split proposals (false reconnections) are located significantly closer to existing branching nodes (nodes with degree ≥ 3) in the fragment graph than va...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 82 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Test if invalid reconnections are often actually missed branch points rather than continuations of a main trunk.
- **Conclusion:** The experiment successfully tested the hypothesis and executed all steps without any significant deviation. Summary of Hypothesis & Experiment: - Hypothesis: Invalid split proposals (false reconnections) are located significantly closer to existing branching nodes (nodes with degree ≥ 3) in the fragment graph than valid split proposals. - Experiment Execution: The code isolated 39,826 degree-1 leaf nodes and 19,460 degree ≥ 3 branching nodes from the fragment graph.
- **Caveats:** implementation deviated from plan

### 52. (Surprise 0.690) Rejected: Split errors and merge errors are spatially coupled; the 3D distance from a true split endpoint to the nearest merge junction is significantly shorter than the...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 87 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate whether topology errors cluster in specific problematic regions of the tissue, suggesting that resolving a merge error should trigger a localized re-planning of split connections.
- **Conclusion:** The experiment was successfully executed, extracting the spatial distance metrics and performing the statistical comparison as requested. The code correctly mapped fragment nodes to the ground truth, identifying 295 merge junctions and 7444 valid split endpoints (leaves). The statistical analysis showed that the mean distance to the nearest merge junction for split endpoints was 3159.20 µm, which was slightly larger than the mean distance for correct nodes (2925.77 µm).
- **Caveats:** none noted

### 53. (Surprise 0.690) Rejected: Omitted ground-truth edges (areas missed entirely by the U-Net) exhibit significantly lower raw fluorescence intensity than the midpoints of valid split gaps, v...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 89 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Contrast the image-level evidence underlying 'omission' errors versus 'split' errors to understand the limits of automated segmentation.
- **Conclusion:** The script identified 243,180 omitted GT nodes and 1,391 valid split midpoints, subsequently sampling 200 from each group for intensity extraction from the Zarr image store. Results: - Median Intensity (Omitted GT Nodes): 56.50 - Median Intensity (Split Midpoints): 33.50 - Mann-Whitney U Test (Omitted < Split): p-value = 1.00 Findings & Conclusion: The original hypothesis postulated that omitted GT edges would exhibit significantly lower raw fluorescence intensity than split gaps, under the assumption that true omissions are driven by severe signal dropout. However, the experimental data contradicts this hypothesis.
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 54. (Surprise 0.690) Belief dropped: False merge errors disproportionately occur between fragment components with large overall cable lengths, whereas valid split endpoints typically connect smalle...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 91 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate whether the graph-level state feature 'component size' is predictive of error types, which would justify dynamically refreshing component features during an iterative proofreading process.
- **Conclusion:** The code accurately calculated the cable lengths for all connected components, identified 295 merge edges and 733 valid split component pairs, and performed a Mann-Whitney U test to compare their size distributions. The statistical test yielded a highly significant result (p = 1.87e-53), indicating a clear difference in the component sizes associated with these two topological events. Interestingly, the data reveals a more nuanced picture than the initial hypothesis suggested.
- **Caveats:** none noted

### 55. (Surprise 0.690) No effect found: The topological accuracy of the U-Net reconstruction (measured by Split Rate and Edge Accuracy) varies significantly depending on the human annotator who traced...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 94 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if reconstruction errors are uniformly distributed or biased by the specific annotator/brain region traced.
- **Conclusion:** The code accurately mapped U-Net fragments to the human ground truth, calculated the Edge Accuracy and Split Rate per connected component, and grouped the metrics by the respective annotator (JG, HP, IG, SP, JT, PP, MB, YZ). Statistical evaluations including Kruskal-Wallis and One-Way ANOVA were performed to test for annotator-specific bias. The results show performance variations between the annotators; for example, annotator 'JT' reported the highest Edge Accuracy (89.92%) and lowest Split Rate (0.15), while 'SP' exhibited the lowest Edge Accuracy (74.43%) and highest Split Rate (0.23).
- **Caveats:** none noted

### 56. (Surprise 0.690) Rejected: The physical distance of a true split gap is negatively correlated with the local spatial density of independent fragment endpoints, suggesting that in dense ne...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 97 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Analyze the relationship between the scale of a split error (gap distance) and the local crowding of fragment endpoints.
- **Conclusion:** The implementation successfully identified 1,801 true split gaps (distance < 30 µm, endpoints mapping to the same ground-truth neuron), calculated their exact physical distances, and counted the local endpoint density within a 30 µm radius for each gap's midpoint. Experiment Results: - Identified True Split Gaps: 1,801 - Spearman Rank Correlation: 0.2622 - P-value: 1.0567e-29 - Mean Gap Distance: 11.42 µm - Mean Local Density: 0.20 other endpoints Findings: The results explicitly contradict the initial hypothesis. The hypothesis predicted a negative correlation between gap distance and local endpoint density (i.e., denser regions would systematically produce shorter gaps).
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 57. (Surprise 0.690) Rejected: The absolute difference in estimated neurite radius (node_radius) between two reconnecting fragment endpoints is significantly smaller for true-positive split r...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 98 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Evaluate the viability of using skeleton radius differences at endpoints as a lightweight, image-free feature for filtering invalid split reconnection proposals, ensuring robust statistical testing by mitigating data type overflow.
- **Conclusion:** The experiment was successfully re-run with corrected data types (float64) and an expanded search radius of 50 µm. This yielded 629 valid pairs and 10 invalid pairs. The numerical overflow issue was resolved, and the Mann-Whitney U test successfully computed p-values for both metrics.
- **Caveats:** none noted

### 58. (Surprise 0.690) Rejected: Ground-truth neurons with higher morphological tortuosity (higher ratio of path length to Euclidean distance) experience a significantly higher rate of U-Net sp...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 99 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate the correlation between the intrinsic tortuosity of a neuron and its susceptibility to automated fragmentation.
- **Conclusion:** The code calculated the length-weighted average tortuosity for all 19 ground-truth neurons and compared it against the normalized split error rate ((fragments - 1) per mm of cable length). Split rates across the neurons ranged between 0.10 and 0.32 per mm, while tortuosity values ranged from 1.178 to 1.360. A Pearson correlation test between average tortuosity and split rate yielded an r-value of -0.0545 and a p-value of 0.8247.
- **Caveats:** none noted

### 59. (Surprise 0.690) Belief dropped: A multivariate Logistic Regression classifier trained on a combination of topological and geometric features achieves a higher F1 score for predicting true-posi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 105 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Evaluate if machine learning on a small suite of locally extracted skeleton features outperforms static heuristics for split-correction.
- **Conclusion:** The experiment successfully executed the pipeline to test the hypothesis. The findings reject the original hypothesis: the static rule baseline actually outperformed the Machine Learning model on the test set. The static rule achieved an F1 score of 0.7921, whereas the Logistic Regression model yielded an F1 score of 0.7526 (a delta of -0.0395).
- **Caveats:** implementation deviated from plan

### 60. (Surprise 0.690) Rejected: True-positive split reconnections have a significantly higher minimum raw fluorescence intensity along their straight-line bridging paths compared to false-posi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 107 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Prove that integrating raw image evidence directly along proposed gaps can distinguish true discontinuities (faint signals) from false connections (background darkness).
- **Conclusion:** The experiment successfully executed the pipeline to test the hypothesis that true-positive (TP) split reconnections have significantly higher minimum raw fluorescence intensity across their gaps compared to false-positive (FP) reconnections. A custom unpickler mocked the missing dependencies, and S3-based anonymous access was utilized to fetch the multi-terabyte 5D Zarr array lazily. The algorithm identified 1,209 fragment connection proposals within a 15 µm threshold.
- **Caveats:** none noted

### 61. (Surprise 0.690) Rejected: Short U-Net fragments (1000-5000 µm total path length) have a significantly higher rate of false-merge branch points per millimeter of cable than long fragments...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 109 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate if fragment cable length correlates with the frequency of topological merge errors.
- **Conclusion:** Experiment Results: - Cable Length Distribution: Analyzed 10,172 total fragments with a mean length of 1849.31 µm and median of 1175.46 µm. - Fragment Categorization: 7,057 fragments fell into the 'Short' category (1000-5000 µm) and 373 fragments in the 'Long' category (>5000 µm). - Merge Rates: Short fragments exhibited an average of 0.0004 false merges / mm (std: 0.0184), whereas Long fragments had a notably higher average of 0.0037 false merges / mm (std: 0.0350).
- **Caveats:** implementation deviated from plan

### 62. (Surprise 0.690) Rejected: A density-adaptive distance threshold—where the allowable search radius for reconnections is mathematically penalized in regions of high leaf density—yields a h...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 110 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Test if adapting split-correction strictness to local error clustering improves overall reconnection accuracy.
- **Conclusion:** The experiment successfully executed the pipeline to test the hypothesis. The dataset was correctly located and loaded by safely mocking dependencies, and 657 candidate leaf pairs (647 True Positives, 10 False Positives) within a 50 µm search radius were extracted based on their ground-truth mappings. The findings reject the hypothesis: the density-adaptive distance threshold did not yield a higher F1 score than the globally fixed distance threshold.
- **Caveats:** none noted

### 63. (Surprise 0.690) Rejected: The local density of candidate endpoints (leaves) within a 20 µm radius is significantly higher for false-positive split-correction proposals than for true-posi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 112 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Assess whether local proposal density can act as a negative feature to suppress false reconnections in dense, highly fragmented regions.
- **Conclusion:** The code accurately identified fragment leaf nodes, generated reconnection proposals within a 20 µm radius, mapped them to ground truth to classify True Positive (TP) and False Positive (FP) proposals, and computed the local density of candidate endpoints around the proposal midpoints. Out of 1,397 proposals connecting different components, 395 were classified as TP and 1,002 as FP. The descriptive statistics showed that the local density (within a 20 µm radius) was almost identical for both groups: TP proposals had a mean density of 2.09 (median 2.00, std 0.34), while FP proposals had a mean density of 2.08 (median 2.00, std 0.26).
- **Caveats:** none noted

### 64. (Surprise 0.690) Rejected: Formulating split-correction as a Maximum Weight Matching problem to enforce mutual exclusivity at endpoints reduces false-positive reconnections by at least 10...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 116 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Evaluate whether enforcing mutual exclusivity via Maximum Weight Matching reduces false-positive connections compared to an independent classification baseline, using identical scoring criteria and candidate pools.
- **Conclusion:** The modified experiment executed successfully, employing an apples-to-apples comparison by using a unified edge score threshold (score > 0.5) and a 20 µm search radius for both the baseline and proposed approaches. Out of 928 valid candidates, the baseline (independent classification) yielded 246 True Positives (TP) and 1 False Positive (FP). The proposed Maximum Weight Matching (MWM) approach yielded 242 TP and 1 FP, resulting in a 0.00% reduction in False Positives and a slight decrease (-1.63%) in True Positives.
- **Caveats:** none noted

### 65. (Surprise 0.690) Rejected: Incorporating mean raw fluorescence intensity from the gap between fragment endpoints into the reconnection score increases True Positive precision by at least...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 118 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if local image evidence in the gap between endpoints can effectively distinguish true splits from false-positive reconnections.
- **Conclusion:** The experiment successfully tested the hypothesis that incorporating the mean raw fluorescence intensity from the gap between fragment endpoints into the reconnection score increases True Positive precision by at least 15%. The script identified 347 valid candidate reconnection pairs within a 15 µm radius. By mapping these pairs to the ground-truth annotations, it found an overwhelmingly imbalanced candidate pool of 344 True Positives (TP) and 3 False Positives (FP).
- **Caveats:** none noted

### 66. (Surprise 0.690) Rejected: Split errors occur significantly less frequently near the thickest region of the neuron (soma proxy), and the spatial density of splits increases as a function...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 122 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Analyze the spatial frequency of splits relative to the soma across the neuronal arbor.
- **Conclusion:** The analysis generated the following key metrics: - Proximal Cable Length (<1000 µm): 222.82 mm, with 155.5 splits. - Distal Cable Length (>=1000 µm): 5340.01 mm, with 1852.0 splits. - Proximal Split Density: 0.70 splits/mm.
- **Caveats:** none noted

### 67. (Surprise 0.690) Rejected: The apparent automated reconstruction split rate (mapped fragment components per GT neuron) varies significantly depending on the human annotator who traced the...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 123 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if ground-truth annotator style or bias correlates with the degree of fragmentation in the automated reconstruction.
- **Conclusion:** The experiment was successfully executed, calculating the automated reconstruction split rate (mapped fragments per mm of GT cable) grouped by annotator. The dataset contained 19 ground-truth neurons, traced by 8 different annotators. The mean split rates varied from a low of 0.14 (Annotator JT, n=1) to a high of 0.22 (Annotator PP, n=4, and Annotator SP, n=3).
- **Caveats:** small sample size; implementation deviated from plan

### 68. (Surprise 0.690) No effect found: The mean physical distance (alignment error) between human ground-truth tracing nodes and the nearest automated U-Net fragment center-line varies significantly...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 124 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate cross-annotator variability in ground-truth center-line placement, which can confound precise distance-based metrics and radius estimations.
- **Conclusion:** It loaded the dataset, grouped the 19 ground-truth components by annotator initials extracted from the SWC IDs, computed the mean physical alignment error (distances <= 10 µm) to the closest U-Net fragment node, and performed a one-way ANOVA. The findings show that there are 8 distinct annotators (JG, HP, IG, SP, JT, PP, MB, YZ) who contributed to the 18 traced neurons (with one additional component perhaps accounting for a total of 19 tracked components). The mean alignment errors across annotators ranged from ~1.58 µm to ~1.69 µm.
- **Caveats:** none noted

### 69. (Surprise 0.690) Rejected: U-Net fragments terminating at leaf nodes with larger neurite radii represent abrupt breakages in thick trunks and will thus have significantly smaller spatial...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 126 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if local neurite radius at a split point is inversely correlated with the distance to the nearest potential reconnection candidate, which could help prioritize split-correction proposals.
- **Conclusion:** Out of 39,826 total leaf nodes, 5,185 pairs were found within a 50 µm distance upper bound. The mean leaf radius was ~1.38 µm, and the mean gap distance was ~21.29 µm. A Pearson correlation test yielded an r-value of 0.1931 with a statistically significant p-value of 9.44e-45.
- **Caveats:** none noted

### 70. (Surprise 0.690) Rejected: U-Net fragment components that erroneously merge multiple ground-truth neurons exhibit significantly higher local path tortuosity than fragment components that...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 132 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Test if local tortuosity (path length divided by straight-line distance) can act as a structural feature to flag heavily tangled regions prone to merge errors.
- **Conclusion:** The script loaded the dataset using the modified unpickler, queried the KDTree to map U-Net fragments nodes to the Ground-Truth (GT) graph, and classified fragment components into 'Merged' (n=11) and 'Pure' (n=951) based on their mapping to GT neurons. Upon calculating the tortuosity (ratio of path length to Euclidean distance) over 50-µm segments, the results showed that both Merged and Pure components have near-identical mean tortuosities (~1.0534 vs 1.0535) and variances. The Mann-Whitney U test yielded a p-value of 0.423 (U-statistic: 5409.0), which is not statistically significant.
- **Caveats:** none noted

### 71. (Surprise 0.690) Rejected: There is a significant negative correlation between the physical length of a U-Net fragment and the spatial gap distance to its nearest valid reconnection, mean...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 133 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate the relationship between a fragment's reconstructed size and the magnitude of the topological split errors separating it from its true neighbors.
- **Conclusion:** Evaluating 8,780 valid gap reconnections, the analysis yielded a Spearman Rank Correlation of 0.3096 with a highly significant p-value of 2.1552e-194. Contrary to the initial hypothesis, which posited a negative correlation (expecting shorter gaps for longer fragments), the results demonstrate a statistically significant *positive* correlation. This indicates that larger, longer fragments are actually separated by larger spatial gaps, whereas smaller fragments are typically separated by smaller gaps (likely representing closely shattered pieces of local noise or weak signal).
- **Caveats:** implementation deviated from plan

### 72. (Surprise 0.690) Rejected: Neurons with a higher ground-truth branching density (bifurcations per mm of cable) exhibit a significantly higher split rate (number of fragmented U-Net compon...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 142 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if neuron morphological complexity (branching density) is a strong predictor of fragmentation rate in the automated reconstruction.
- **Conclusion:** The experiment successfully executed and analyzed 19 Ground Truth (GT) components. It computed the branching density and split rate for each neuron. The results revealed a mean branching density of 1.35 branches/mm and a mean split rate of 0.18 fragments/mm.
- **Caveats:** none noted

### 73. (Surprise 0.690) Rejected: Merge errors in the automated reconstruction occur in regions with significantly higher local fragment node density than correctly reconstructed branching regio...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 150 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if spatial crowding of predicted fragments is a reliable proxy for detecting false merges.
- **Conclusion:** The script correctly mapped fragments to ground truth components using KDTrees, identifying 468 merge boundary nodes and 5,374 valid branching nodes. The local node density within a 30 µm radius was computed and averaged 18.35 nodes for merge boundaries and 23.72 nodes for valid branching nodes. A Welch's t-test yielded a t-statistic of -14.5534 and a highly significant p-value of 1.2880e-40.
- **Caveats:** none noted

### 74. (Surprise 0.690) Rejected: The topological fragmentation rate (number of broken UNet fragments per millimeter of ground-truth cable length) varies significantly across different human ann...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 153 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Analyze annotator-specific metadata to see if split error rates are biased by human selection of specific morphological regions or neuron types.
- **Conclusion:** The experiment successfully tested the hypothesis that topological fragmentation rates differ across human annotators due to selection bias. The script mapped automated fragments to the ground-truth neurons, calculated the fragmentation rate (fragments per millimeter of cable length) for each neuron, and grouped these rates by the 8 human annotators identified in the dataset metadata. Experiment Results: - Fragmentation Rates: The average fragmentation rate across annotators ranged from a minimum of 0.14 frags/mm (Annotator JT, 1 neuron) to a maximum of 0.23 frags/mm (Annotator PP, 4 neurons).
- **Caveats:** none noted

### 75. (Surprise 0.690) Rejected: Nodes at the junction of a merge error exhibit unnaturally inflated radii compared to the rest of the merged fragment, reflecting the U-Net fusing distinct adja...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 154 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate if anomalous localized spikes in predicted neurite radius can serve as an endogenous feature for agentic merge detection, whilst preventing float16 numerical overflow during aggregation.
- **Conclusion:** The experiment successfully corrected the numerical overflow issue by casting the `node_radius` array to `float64` before aggregation, enabling a valid evaluation of the hypothesis. A total of 23 merge fragments mapping to distinct Ground Truth (GT) components were analyzed. The results reveal that the mean junction radius (1.9355 µm) is slightly lower than the mean bulk radius (1.9759 µm), resulting in a mean difference of -0.0405 µm.
- **Caveats:** none noted

### 76. (Surprise 0.690) Rejected: Fragments containing U-Net merge errors (nodes mapping to multiple distinct ground-truth neurons) act as artificial topological hubs, possessing a significantly...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 156 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** To determine if an anomalously high endpoint density can serve as an intrinsic topological signature for identifying fragments that erroneously merge distinct neurons.
- **Conclusion:** The experiment was successfully executed and correctly implemented the steps to test the hypothesis. Out of the U-Net fragments matched to ground-truth nodes (within 5 µm), a total of 1,001 fragments were evaluated. Of these, 20 fragments were identified as 'merged' (erroneously mapping to multiple ground-truth components), while 981 were 'clean'.
- **Caveats:** none noted

### 77. (Surprise 0.690) Rejected: In ambiguous multi-way split scenarios, true-positive reconnections exhibit significantly higher radius continuity (lower difference in estimated neurite radius...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 158 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Evaluate neurite radius continuity as an endogenous feature for resolving mutually exclusive reconnection proposals.
- **Conclusion:** The experiment successfully executed, overcoming previous module loading errors by implementing a robust `MockingUnpickler` that dynamically creates dummy classes for missing dependencies. It evaluated the hypothesis that true-positive (TP) reconnections exhibit significantly higher radius continuity (lower radius difference) than false-positive (FP) reconnections in ambiguous split scenarios. The analysis identified 10 competitive split scenarios.
- **Caveats:** none noted

### 78. (Surprise 0.690) No effect found: Degree-3 branching nodes representing merge errors (where the incident edges belong to different true neurons) have a significantly larger minimum branching ang...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 162 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Assess the geometric branching angle as an actionable feature to detect and flag automated merge errors without image data.
- **Conclusion:** The experiment successfully identified and analyzed degree-3 branching nodes in the UNet fragments graph. A total of 19,438 degree-3 nodes were evaluated and mapped to ground-truth components. Results revealed 19,346 'True Branches' (where all nodes map to the same ground-truth neuron) and 90 'Merge Branches' (where the neighbors map to at least two different ground-truth neurons).
- **Caveats:** none noted

### 79. (Surprise 0.690) Rejected: Candidate reconnections involving at least one very short U-Net fragment (< 2000 µm) are significantly more likely to be false spatial merges than candidates wh...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 169 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Investigate if fragment length acts as a proxy for segmentation confidence and whether extremely short fragments disproportionately contribute to false merge proposals.
- **Conclusion:** The experiment successfully investigated whether candidate reconnections involving short U-Net fragments (< 2000 µm) are more likely to be false spatial merges compared to those where both fragments are long. The 'Contains Short' group had 233 True and 2 False connections (99.15% True), while the 'Both Long' group had 175 True and 1 False connection (99.43% True). Statistical testing (Chi-square and Fisher's exact test) yielded a p-value of 1.0, indicating no statistically significant difference in the false merge rates between the two length categories.
- **Caveats:** test driven to p=1.0 by near-zero positive counts; implementation deviated from plan

### 80. (Surprise 0.690) No effect found: For U-Net fragment endpoints with multiple nearby candidate reconnections, the margin of geometric alignment (difference in tangent cosine similarity between th...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 171 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Verify if the principle of mutual exclusivity holds locally by testing whether true reconnections geometrically dominate alternative local false proposals by a wider margin than when a true connection is absent.
- **Conclusion:** The experiment successfully tested the hypothesis that true anatomical reconnections at U-Net fragment endpoints geometrically dominate alternative false proposals by a wider margin than false-dominant sets. The code extracted 39,826 U-Net fragment leaves and identified 151 leaves with multiple candidate reconnections within a 25 µm radius. The leaves were grouped based on whether the best candidate correctly belonged to the same ground-truth neuron (True-dominant, n=46) or not (False-dominant, n=105).
- **Caveats:** none noted

### 81. (Surprise 0.690) Rejected: Candidate leaf pairs with a smaller relative difference in neurite radius have a higher probability of being True Positive reconnections than those with large d...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 181 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Evaluate the impact of neurite radius continuity on split-correction precision by leveraging the `node_radius` attribute.
- **Conclusion:** The script located the dataset, successfully applied the dependency mocks to retain networkx functionality, extracted candidate leaf pairs, computed the topological features and relative radius differences, and evaluated the metrics using both statistical testing and greedy split-correction benchmarks. The findings strongly reject the hypothesis. The Kolmogorov-Smirnov test yielded a non-significant p-value of 0.9191, demonstrating no statistically significant difference between the radius difference distributions of True Positive (TP) and False Positive (FP) reconnections.
- **Caveats:** none noted

### 82. (Surprise 0.690) No effect found: The baseline topological complexity of a neuron varies systematically by annotator, such that the U-Net split error rate (splits per millimeter) is significantl...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 182 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if specific human annotators traced fundamentally more complex or difficult neurons, leading to observable bias in automated error rates.
- **Conclusion:** The experiment was successfully executed, fulfilling the objective to evaluate if annotator-specific topological complexity introduces bias in the baseline U-Net split error rates. The script successfully extracted 8 distinct annotators ('HP', 'IG', 'JG', 'JT', 'MB', 'PP', 'SP', 'YZ') from the 18 ground truth neurons. The total skeleton cable lengths were computed, and fragments were accurately mapped to their corresponding GT components to determine split counts.
- **Caveats:** none noted

### 83. (Surprise 0.690) Rejected: The complexity of ground-truth neurons, measured by branching frequency, varies significantly across different human annotators and is strongly correlated with...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 185 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Assess how annotator variability and neuron morphological complexity influence the baseline topological split error rate of the automated U-Net reconstruction.
- **Conclusion:** The experiment was successfully executed, and the hypothesis was tested by evaluating ground-truth neuron complexity and its relationship with the U-Net split rate. 1. Annotator Variability: The branching frequency (branches per 1,000 µm) was calculated for each ground-truth neuron and grouped by the human annotators.
- **Caveats:** none noted

### 84. (Surprise 0.690) Rejected: Strict greedy cycle and degree constraints prematurely reject topologically valid reconnections; a 'rollback' mechanism that replaces an earlier greedy edge wit...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 186 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Test a reversible decision (rollback) mechanism against a strict greedy baseline for split-correction.
- **Conclusion:** Out of 1209 candidate leaf pairs, there were 308 true positive candidates. The strict greedy baseline achieved 300 True Positives (TPs) and 874 False Positives (FPs), yielding an F1 score of 0.4049. Implementing the rollback mechanism—which replaces greedy edges with mutually exclusive proposals having a tangent agreement score higher by at least 0.3—resulted in no net improvement.
- **Caveats:** implementation deviated from plan

### 85. (Surprise 0.690) Rejected: Due to the lower imaging resolution along the Z-axis of the ExaSPIM volume, false-positive split connections are disproportionately aligned along the Z-axis, re...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 187 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Assess how voxel anisotropy (lower Z-resolution) impacts the spatial distribution and reliability of split-correction proposals.
- **Conclusion:** The experiment successfully executed the pipeline to test the hypothesis regarding the impact of Z-axis anisotropy on candidate reconnection precision. It identified candidate leaf-to-leaf pairs within a 15 µm radius, computed the 3D connection vectors, and categorized them into Z-dominant and XY-dominant directions. Out of the GT-mapped candidate pairs, 95 were Z-dominant (95 TP, 0 FP) and 252 were XY-dominant (249 TP, 3 FP).
- **Caveats:** none noted

### 86. (Surprise 0.690) Rejected: Fragment endpoints located within 50 µm of a merge error have a significantly higher local density of competing reconnection proposals compared to endpoints loc...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 189 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Determine if merge errors tend to co-occur with dense clusters of broken fragments (ambiguous split regions), supporting the idea that split and merge correction must be coupled.
- **Conclusion:** The experiment was successfully executed, and the hypothesis was tested by evaluating the local density of fragment endpoints (competing reconnection proposals) near merge errors compared to those farther away. 1. Merge Identification: The analysis identified 92 merge nodes (branch points where immediate neighbors map to different ground-truth components) and a total of 39,826 fragment endpoints (leaf nodes) in the automated reconstruction.
- **Caveats:** none noted

### 87. (Surprise 0.690) Belief dropped: False-positive merge branch points in the U-Net fragments have a significantly larger predicted `node_radius` compared to valid biological branch points due to...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 192 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Test if an abnormally large node radius can be used as a standalone local feature to flag potential merge errors prior to agentic rollback.
- **Conclusion:** The experiment was successfully executed, and the previous numerical overflow issues were resolved by casting the float16 radii arrays to float64. The script successfully categorized the fragment branch nodes into 36 'Merge' points and 5,586 'Valid' points. The analysis found that the mean node radius for Merge branch points was 1.8455 µm (Variance: 0.0859), while Valid branch points had a slightly higher mean radius of 1.9529 µm (Variance: 0.0704).
- **Caveats:** none noted

### 88. (Surprise 0.690) Rejected: Because the ExaSPIM imaging resolution is anisotropic (Z-axis has poorer resolution than X/Y), the displacement vectors of true-positive split gaps (valid recon...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 197 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Assess whether the lower axial resolution of the light-sheet microscope systematically biases the directionality of U-Net split errors.
- **Conclusion:** The experiment successfully executed and analyzed 8,007 valid split pairs to determine if U-Net split errors exhibit a directional bias due to anisotropic imaging resolution. The mean absolute projection magnitudes along the X, Y, and Z axes were 0.5097, 0.4891, and 0.4924, respectively. While the Friedman test yielded a statistically significant result (p-value ≈ 0.0013), indicating that the distributions across the three axes are not identical, the expected dominant alignment with the lower-resolution Z-axis was not observed.
- **Caveats:** none noted

### 89. (Surprise 0.690) No effect found: The 3D raw fluorescence image patches centered on invalid merge nodes exhibit significantly higher voxel intensity variance than patches centered on valid biolo...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 200 · **Belief:** Leaning True → Leaning False (0.708→0.266) · **Direction:** Negative · **Surprisal:** −0.690
- **Tested:** Validate whether raw image variance acts as a direct signature of false merges (intersecting distinct axons) compared to actual structural bifurcations.
- **Conclusion:** The algorithm identified 5,555 valid branch nodes and 50 invalid merge nodes by mapping fragment branch nodes to the ground truth. A subset of 150 valid branches and all 50 invalid merges were sampled, extracting 32x32x32 voxel patches to calculate their intensity variance. The results showed that valid branches had a median variance of 972.85 and a mean of 3243.85, whereas invalid merges had a median variance of 846.88 and a mean of 2354.59.
- **Caveats:** implementation deviated from plan

### 90. (Surprise 0.690) Supported: The probability of a fragment containing a merge error increases logarithmically with its total cable length, reflecting the higher spatial volume and crossing-...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 149 · **Belief:** Leaning False → Leaning True (0.292→0.734) · **Direction:** Positive · **Surprisal:** +0.690
- **Tested:** Model the relationship between a fragment's morphological extent (cable length) and its likelihood of containing topological merges.
- **Conclusion:** The experiment successfully tested the hypothesis by mapping automated UNet fragments to ground-truth graphs and modeling the probability of a merge error based on the fragment's total cable length. Out of 1001 fragments that interacted with the ground truth, 20 were identified as merged (mapped to >= 2 ground truth neurons). The logistic regression results strongly support the hypothesis: the natural logarithm of cable length is a statistically significant predictor of a merge error (p-value = 2.04e-05).
- **Caveats:** none noted

### 91. (Surprise 0.674) Rejected: Different human annotators exhibit systematically different topological split rates (fragment discontinuities per millimeter of ground truth) due to varying per...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 42 · **Belief:** Leaning True → Leaning False (0.792→0.359) · **Direction:** Negative · **Surprisal:** −0.674
- **Tested:** Determine if the baseline fragmentation of the automated U-Net is modulated by the subjective variations in the human ground truth used for evaluation.
- **Conclusion:** The script successfully extracted the human annotators' initials from the SWC identifiers, calculated the ground-truth (GT) cable lengths, and accurately mapped the U-Net fragments to the 18 GT neurons using the KD-tree with a 5 µm distance threshold. The calculated split rates (fragments per millimeter) varied from 0.10 to 0.27 across individual neurons. Grouped by the eight different human annotators, the median split rates ranged from 0.14 to 0.22.
- **Caveats:** none noted

### 92. (Surprise 0.641) Rejected: False merge proposals between fragment endpoints involve significantly larger fragment components on average than true split proposals.
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 43 · **Belief:** Leaning True → Leaning False (0.708→0.297) · **Direction:** Negative · **Surprisal:** −0.641
- **Tested:** To determine if the size of the components being connected is a risk factor for generating false reconnections, as larger fragments span more volume and have higher chances of incidental proximity.
- **Conclusion:** The script correctly loaded the dataset, identified pairs of fragment leaves within 30 µm, mapped them to ground-truth nodes to classify them as True Splits (1,801 pairs) or False Merges (11 pairs), and calculated the proposal sizes (sum of the node counts of the connecting fragments). The resulting median proposal sizes were 1,040 nodes for True Splits and 786 nodes for False Merges. The Mann-Whitney U test yielded a p-value of 0.476, indicating no statistically significant difference in proposal sizes between the two groups.
- **Caveats:** none noted

### 93. (Surprise 0.641) Rejected: Different human annotators trace neurons with varying morphological priorities (e.g., stopping at thin axons vs.
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 60 · **Belief:** Leaning True → Leaning False (0.708→0.297) · **Direction:** Negative · **Surprisal:** −0.641
- **Tested:** Analyze whether human annotator bias influences the evaluation metrics of the automated U-Net reconstruction.
- **Conclusion:** Hypothesis: Different human annotators trace neurons with varying morphological priorities (e.g., stopping at thin axons vs. tracing fully), resulting in statistically significant differences in the U-Net split rate (splits per mm) across annotator groups. Experiment Results: 1.
- **Caveats:** none noted

### 94. (Surprise 0.641) Rejected: Over 80% of false-positive reconnections introduced by a relaxed distance-only split-correction involve at least one 'noisy' U-Net fragment from the bottom 50%...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 119 · **Belief:** Leaning True → Leaning False (0.708→0.297) · **Direction:** Negative · **Surprisal:** −0.641
- **Tested:** Determine if aggressive merging predominantly fails because it erroneously attaches long valid fragments to short background noise fragments.
- **Conclusion:** The experiment executed successfully and analyzed the hypothesis that over 80% of false-positive (FP) reconnections involve short U-Net fragments. The median fragment length was computed as 1,175.46 µm, perfectly splitting the 10,172 fragments into 5,086 'Short' and 5,086 'Long' components. Applying a relaxed, distance-only split-correction (15 µm radius) generated 1,174 connections.
- **Caveats:** extreme class imbalance / very few positives

### 95. (Surprise 0.609) Rejected: The precision of spatial reconnection proposals is inversely correlated with the combined cable length of the two candidate fragment components, suggesting that...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 3 · **Belief:** Leaning True → Leaning False (0.625→0.234) · **Direction:** Negative · **Surprisal:** −0.609
- **Tested:** Evaluate how fragment component size impacts the reliability of distance-based reconnection proposals to inform dynamic feature refreshing.
- **Conclusion:** The experiment successfully executed the steps to evaluate the hypothesis that reconnection proposal precision is inversely correlated with the combined cable length of the candidate fragments. Out of 1,209 inter-component leaf pairs within 15 µm, 347 proposals were successfully mapped to ground truth components for validation. The proposals were binned into deciles based on combined cable length ranging from roughly 2,500 µm to over 375,000 µm.
- **Caveats:** none noted

### 96. (Surprise 0.593) No effect found: The rate of U-Net omission errors (the fraction of ground-truth edges with no U-Net fragment nodes within 5 µm) varies significantly depending on the human anno...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 6 · **Belief:** Leaning True → Leaning False (0.708→0.328) · **Direction:** Negative · **Surprisal:** −0.593
- **Tested:** Analyze cross-annotator variability in the context of automated reconstruction performance to determine if certain human annotators systematically trace regions that the U-Net fails to predict.
- **Conclusion:** The experiment successfully calculated the U-Net omission rates for the ground-truth edges by computing edge midpoints and querying the KD-tree. The 18 ground-truth components were accurately mapped to their respective human annotators using the SWC IDs, yielding groupings for 8 unique annotators (JG, HP, IG, SP, JT, PP, MB, YZ). Mean omission rates across the annotators ranged from ~11.4% to ~27.2%.
- **Caveats:** none noted

### 97. (Surprise 0.568) Belief dropped: Due to the mutual exclusivity constraint at endpoints, the spatially closest candidate leaf node is frequently not the true positive connection in dense environ...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 3 · **Belief:** Likely True → Uncertain (0.833→0.469) · **Direction:** Negative · **Surprisal:** −0.568
- **Tested:** Evaluate the risk of relying purely on a 1-Nearest Neighbor distance-based greedy matching strategy for split correction.
- **Conclusion:** The experiment successfully executed and addressed the hypothesis. By analyzing leaf nodes with at least two candidates within a 30 µm radius, it identified 57 dense cases that successfully mapped to a ground-truth (GT) valid connection. The results showed that the 1-Nearest Neighbor (1-NN) approach correctly identified the True Positive connection in 94.74% (54 out of 57) of cases.
- **Caveats:** none noted

### 98. (Surprise 0.568) Belief dropped: The accuracy of a simple spatial nearest-neighbor heuristic for split correction (reconnecting a leaf to its absolute closest neighbor) is negatively correlated...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 165 · **Belief:** Likely True → Uncertain (0.833→0.469) · **Direction:** Negative · **Surprisal:** −0.568
- **Tested:** Quantify how local endpoint clustering limits simple distance-based reconnects, motivating the need for mutual-exclusivity constraints and GNN-based joint reasoning.
- **Conclusion:** A total of 7,053 fragment leaves successfully mapped to ground-truth (GT) components and found valid alternate-fragment nearest neighbors. The overall accuracy of the distance-based reconnect heuristic was surprisingly high at ~87.0%. However, contrary to the hypothesis, the point-biserial correlation between local endpoint density and reconnect success rate showed a weak but statistically significant *positive* correlation (r = 0.1126, p < 0.001).
- **Caveats:** none noted

### 99. (Surprise 0.544) No effect found: The proportion of ground truth cable that the U-Net completely fails to reconstruct (omitted edges) varies significantly depending on the human annotator who tr...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 49 · **Belief:** Leaning True → Leaning False (0.708→0.359) · **Direction:** Negative · **Surprisal:** −0.544
- **Tested:** Test whether baseline omission error rates are biased by the individual annotator styles embedded in the ground truth.
- **Conclusion:** The code extracted annotator initials from the ground truth component IDs, calculated the omission rate for each component, and performed a Kruskal-Wallis H test to compare omission rates across different annotators. The results show variations in mean and median omission rates among annotators (e.g., SP with a mean of 0.2616, JT with a mean of 0.1047). However, the Kruskal-Wallis H Test yielded a p-value of 0.4775, which is far greater than the standard alpha level of 0.05.
- **Caveats:** implementation deviated from plan

### 100. (Surprise 0.544) Belief dropped: Reconnection proposals for leaf nodes with multiple candidate partners (within 15 µm) exhibit significantly lower mean orientation agreement (tangent cosine sim...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 131 · **Belief:** Leaning True → Leaning False (0.708→0.359) · **Direction:** Negative · **Surprisal:** −0.544
- **Tested:** To demonstrate that local endpoint crowding degrades geometric alignment cues, necessitating joint (agentic) reasoning over competing proposals rather than independent single-pass edge classification.
- **Conclusion:** It extracted terminal tangent vectors for leaf nodes and used a KD-tree to identify candidates within a 15 µm radius, categorizing them into single-candidate (2,326 proposals) and multi-candidate (46 proposals) groups. The Mann-Whitney U test yielded a highly significant p-value (9.0681e-04). Interestingly, the Multi-Candidate group showed a higher mean maximum tangent cosine similarity (0.8503) compared to the Single-Candidate group (0.7524).
- **Caveats:** none noted

### 101. (Surprise 0.495) Belief dropped: Greedy split-correction without topological constraints will introduce cycles predominantly into fragment components that already contain merge errors.
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 18 · **Belief:** Leaning True → Leaning False (0.708→0.391) · **Direction:** Negative · **Surprisal:** −0.495
- **Tested:** To investigate the coupling between merge errors and cycle creation when aggressive, distance-based reconnecting is applied.
- **Conclusion:** The experiment successfully simulated a greedy distance-based reconnection (within a 15 µm radius) of fragment leaves to investigate the correlation between newly introduced topological cycles and pre-existing merge errors. The algorithm added 1,382 new edges between leaves. Among the updated connected components, 99 contained cycles.
- **Caveats:** none noted

### 102. (Surprise 0.495) Rejected: False positive split reconnections are significantly more likely to form topological cycles within the U-Net graph than true positive reconnections.
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 24 · **Belief:** Leaning True → Leaning False (0.708→0.391) · **Direction:** Negative · **Surprisal:** −0.495
- **Tested:** Determine if cycle-creation is a reliable, unsupervised topological filter for rejecting invalid connection proposals.
- **Conclusion:** It evaluated 656 candidate leaf pairs within a 30 µm radius that could be reliably mapped to the ground truth. The results showed 651 True Positive (TP) proposals, of which 172 (26.42%) would induce cycles if connected. In contrast, there were only 5 False Positive (FP) proposals, none of which (0.00%) induced cycles.
- **Caveats:** extreme class imbalance / very few positives

### 103. (Surprise 0.495) No effect found: Valid reconnections between split U-Net fragments have a significantly smaller absolute difference in their estimated neurite radii compared to spatially nearby...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 33 · **Belief:** Leaning True → Leaning False (0.708→0.391) · **Direction:** Negative · **Surprisal:** −0.495
- **Tested:** Evaluate whether local neurite radius continuity acts as a robust biological constraint to filter out incorrect split-correction proposals.
- **Conclusion:** The experiment successfully calculated the absolute difference in estimated neurite radii for candidate U-Net leaf node pairs located within a 20 µm distance from distinct fragments. The results found 408 Valid connections and only 3 Invalid connections within this spatial threshold. The mean absolute radius difference for Valid connections was 0.4455 ± 0.3668 µm, while for Invalid connections it was 0.3783 ± 0.2810 µm.
- **Caveats:** small sample size

### 104. (Surprise 0.495) No effect found: Fragment components that contain topological merge errors exist in spatial regions with significantly higher local fragment endpoint density than completely cle...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 83 · **Belief:** Leaning True → Leaning False (0.708→0.391) · **Direction:** Negative · **Surprisal:** −0.495
- **Tested:** Investigate if high spatial congestion (endpoint density) is a strong structural proxy indicator for the presence of U-Net merge errors.
- **Conclusion:** Fragment components were mapped to Ground Truth (GT) nodes; 23 components mapped to 2 or more GT components ('Merged'), and 1,042 components mapped to exactly 1 GT component ('Clean'). Leaf nodes in the fragment graph were isolated (39,826 leaves) to serve as endpoints. The local endpoint density within a 50 µm radius around each component's centroid was calculated.
- **Caveats:** none noted

### 105. (Surprise 0.495) Rejected: In reconnection proposals originating from a single fragment leaf node, a larger distance margin between the nearest and second-nearest candidate significantly...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 120 · **Belief:** Leaning True → Leaning False (0.708→0.391) · **Direction:** Negative · **Surprisal:** −0.495
- **Tested:** Investigate the value of a 'mutual exclusivity' feature (distance margin) by evaluating if highly competitive proposals indicate lower confidence and higher false-positive rates.
- **Conclusion:** The experiment successfully executed, extracting 49 competitive reconnection proposals mapped to the ground truth. A point-biserial correlation test yielded a correlation coefficient of 0.0572 and a high p-value of 0.6964. Because the p-value is significantly greater than the standard 0.05 threshold, there is no statistically significant correlation between the distance margin (the difference in distance between the first and second candidates) and the correctness of the reconnection proposal.
- **Caveats:** none noted

### 106. (Surprise 0.487) Rejected: Split errors disproportionately concentrate at structural bifurcations, resulting in fragment endpoints mapping to ground-truth branching nodes at a significant...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 17 · **Belief:** Leaning True → Uncertain (0.750→0.438) · **Direction:** Negative · **Surprisal:** −0.487
- **Tested:** Assess whether the automated segmentation model is systematically more prone to fragmentation at neurite branching points compared to continuous axonal cables.
- **Conclusion:** ### Summary of Findings - Hypothesis: It was hypothesized that split errors disproportionately occur at structural bifurcations, meaning fragment endpoints would map to ground-truth branching nodes more frequently than expected by baseline distribution. - Results: The baseline Ground-Truth (GT) graph is composed of 0.5397% branching nodes and 99.4603% path nodes. Out of 3,965 fragment endpoints that successfully mapped to GT nodes within a 15 µm threshold, 29 (0.7314%) mapped to branching nodes and 3,936 (99.2686%) mapped to path nodes.
- **Caveats:** none noted

### 107. (Surprise 0.487) Rejected: True-positive split reconnections (gap < 15 µm) exhibit significantly higher tangent angular alignment (cosine similarity) compared to false-positive reconnecti...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 57 · **Belief:** Leaning True → Uncertain (0.750→0.438) · **Direction:** Negative · **Surprisal:** −0.487
- **Tested:** Evaluate if the geometric alignment of fragment endpoints is a statistically valid filter for distinguishing true splits from false merges.
- **Conclusion:** The experiment successfully executed and evaluated the geometric alignment (cosine similarity) of fragment endpoint reconnections within a 15 µm radius. The analysis identified 1,204 True Positive (TP) and only 5 False Positive (FP) reconnection candidates. A one-sided Mann-Whitney U test was performed, yielding a p-value of 0.3513, which is not statistically significant.
- **Caveats:** extreme class imbalance / very few positives; small sample size

### 108. (Surprise 0.446) Belief dropped: False positive connection proposals within a 30 µm radius disproportionately originate from 'hub' endpoints (leaves with a high density of neighboring candidate...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 23 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** To verify if local candidate density around an endpoint is a reliable proxy for false positive risk, informing a confidence penalty for connections in dense regions.
- **Conclusion:** The experiment successfully executed and addressed the hypothesis. By extracting leaf nodes from the `fragments_graph` and identifying neighboring leaves from different components within a 30 µm radius, the experiment computed a 'pair density score' (the sum of valid candidates for both endpoints of a pair). Mapping these pairs to the ground-truth graph (`gt_graph`) yielded 397 True Positive (TP) pairs and 5 False Positive (FP) pairs.
- **Caveats:** none noted

### 109. (Surprise 0.446) Rejected: Ground-truth neurons with greater branching complexity (higher number of branching nodes per millimeter of cable) suffer from a disproportionately higher U-Net...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 31 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** To investigate if morphological complexity is a primary driver of U-Net segmentation failures and resulting fragmentation.
- **Conclusion:** The code processed all 19 ground-truth (GT) neurons, computing their total cable length, branch counts, and branching density (branches/mm). It then successfully mapped UNet fragments to these GT neurons using a 15 µm proximity threshold to calculate the split rate (fragments/mm) for each neuron. The results showed branching densities ranging from 0.92 to 2.09 branches/mm, and split rates ranging from 0.11 to 0.33 splits/mm.
- **Caveats:** none noted

### 110. (Surprise 0.446) Supported: Fragment endpoints resulting from split errors exhibit a significantly sharper tapering effect (steeper decrease in radius) than true biological terminations.
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 46 · **Belief:** Uncertain → Likely True (0.542→0.828) · **Direction:** Positive · **Surprisal:** +0.446
- **Tested:** To determine if radius continuity along the terminal segment of a fragment can distinguish artificial splits from natural leaf nodes.
- **Conclusion:** The script accurately loaded the dataset, identified fragment leaf nodes, and mapped them to the ground-truth (GT) graph to classify them into 'Split Endpoints' (N=3965) and 'True Terminations' (N=3416). The results indicate a highly significant difference (t=48.35, p≈0), with split endpoints showing an average gradient of 0.01223 µm/µm, meaning they taper sharply, while true terminations have a near-zero average gradient (-0.00041 µm/µm), indicating a constant radius. The generated density plot confirms that while there is some overlap, radius gradients above ~0.015 predominantly belong to split endpoints.
- **Caveats:** none noted

### 111. (Surprise 0.446) Rejected: A substantial fraction of split errors require 'end-to-branch' reconnections rather than 'end-to-end' reconnections, and these end-to-branch gaps are significan...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 49 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Classify valid reconnections structurally and verify if the spatial gap distribution differs based on the connection topology.
- **Conclusion:** ### Summary of Findings - Proportions of Split Types: The experiment found a total of 1388 valid reconnections. Of these, a majority (883 or 63.62%) were classified as End-to-Branch connections, while the remaining 505 (36.38%) were End-to-End connections. This strongly supports the first part of the hypothesis: a substantial fraction of topological split errors indeed require reconnecting an endpoint to an internal branch node, rather than merely joining two endpoints.
- **Caveats:** none noted

### 112. (Surprise 0.446) Rejected: Fragment components containing a high density of branching nodes (degree >= 3) are significantly more likely to contain merge errors than linear fragments, indi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 5 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Determine if morphological complexity (branch density) is a strong geometric predictor of merge errors within automated fragments.
- **Conclusion:** The experiment successfully executed and achieved its objective of testing whether branch density is a strong geometric predictor of merge errors in automated fragments. The code efficiently located and loaded the dataset, calculated node degrees and cable lengths, and successfully mapped the fragment nodes to ground truth (GT) to identify merge-containing and merge-free fragments. The analysis identified 14 merge-containing fragments and 1023 merge-free fragments.
- **Caveats:** small sample size

### 113. (Surprise 0.446) Belief dropped: Pairs of fragment endpoints (leaf nodes) that belong to the same ground-truth neuron (true-positive splits) exhibit significantly higher branch tangent alignmen...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 11 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Evaluate the predictive power of 3D branch tangent alignment in distinguishing valid reconnections from false reconnections during split error correction.
- **Conclusion:** The experiment successfully isolated leaf nodes and evaluated branch tangent alignment between pairs within a 20 µm radius. The results show 408 valid reconnections and 989 invalid reconnections (after excluding intra-component pairs). The mean dot product for valid pairs (0.7543) was slightly higher than for invalid pairs (0.7305), with 55.88% of valid pairs and 52.07% of invalid pairs having a dot product > 0.8.
- **Caveats:** small sample size; implementation deviated from plan

### 114. (Surprise 0.446) Rejected: The geometric distance from a fragment's endpoint to its true reconnection partner (gap size) is inversely proportional to the total cable length of the fragmen...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 38 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Investigate the relationship between a U-Net fragment's morphological size and the physical extent of its split errors.
- **Conclusion:** The ANOVA test revealed a statistically significant difference in mean gap distances across the fragment length bins (F = 108.59, p < 0.001). However, the findings partially contradict the initial hypothesis. While the mean gap size was hypothesized to be inversely proportional to fragment length, the data showed that fragments >5000 µm actually had a higher average gap distance (236.0 µm) compared to the <2000 µm (195.4 µm) and 2000-5000 µm (189.1 µm) bins.
- **Caveats:** none noted

### 115. (Surprise 0.446) Rejected: True split gaps exhibit significantly higher neurite radius continuity (smaller difference in endpoint radii) compared to false spatial reconnections, providing...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 65 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Determine if the difference in estimated neurite radius at the endpoints is significantly smaller for valid split pairs than for invalid nearby fragment pairs.
- **Conclusion:** The script identified 1,391 valid split pairs (true gaps belonging to the same ground-truth neuron) and 6 invalid pairs (spatial neighbors belonging to different ground-truth neurons) within a 20 µm radius. The results do not support the hypothesis. The mean radius difference for valid pairs (0.4610 µm) was slightly higher than that for invalid pairs (0.3853 µm).
- **Caveats:** extreme class imbalance / very few positives

### 116. (Surprise 0.446) Rejected: In scenarios where a single fragment leaf node has multiple competing reconnection candidates within a 15 µm radius (mutual exclusivity), the proposal with the...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 67 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Test if local orientation is a stronger predictor than spatial proximity for resolving competing reconnection proposals at a single endpoint.
- **Conclusion:** The experiment successfully executed and evaluated 14 scenarios where a fragment leaf node had multiple competing reconnection candidates within a 15 µm radius. Out of these, 13 scenarios had at least one True Positive (TP) candidate. The distance-based selection method identified the correct TP in 13 of the 14 scenarios (92.86% overall accuracy, 100% accuracy on solvable scenarios).
- **Caveats:** test driven to p=1.0 by near-zero positive counts; implementation deviated from plan

### 117. (Surprise 0.446) No effect found: False merge sites exhibit significantly higher variance in neurite radius among their converging branches compared to true biological bifurcations, because merg...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 76 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Test if local neurite radius consistency is a reliable intrinsic feature for detecting structural false merges without requiring image re-evaluations.
- **Conclusion:** Out of 19,460 branch points in the fragments graph, 5,571 were mapped to True Branches and 38 to Merge Errors based on ground-truth proximity. The computed mean radius variance among converging branches was visibly higher for Merge Errors (0.0146 μm²) compared to True Branches (0.0088 μm²). However, Welch's t-test yielded a t-statistic of 1.6875 and a p-value of 0.3406 (p > 0.05).
- **Caveats:** small sample size

### 118. (Surprise 0.446) Rejected: Split errors (frequency and gap size) exhibit significant variation along the Z-axis (imaging depth) due to the anisotropic resolution (0.748 x 0.748 x 1.0 µm)...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 79 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Analyze the spatial distribution of split errors along the Z-axis to determine if tissue depth exacerbates segmentation failures.
- **Conclusion:** The script identified 1,519 valid split pairs correctly mapped to the ground-truth neurons. A Pearson correlation analysis yielded an r-value of 0.0092 with a p-value of 0.7200, indicating that there is no statistically significant linear correlation between the Z-depth (imaging depth) and the size of the split gaps. The binned summary statistics reveal that the frequency of splits varies along the Z-axis, with the vast majority concentrated in the middle depths (from ~4,103 to 14,562 µm).
- **Caveats:** none noted

### 119. (Surprise 0.446) Belief dropped: In dense regions where a fragment endpoint has multiple potential reconnection candidates within a 20 µm radius, the true biological connection (True Split) can...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 80 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Evaluate if local tangent alignment is a sufficiently strong geometric discriminator to resolve mutually exclusive reconnection proposals at an endpoint without requiring image features.
- **Conclusion:** The experiment successfully executed to test the hypothesis that local tangent alignment (cosine similarity) can distinguish true biological reconnections from false ones in dense regions where mutually exclusive candidates exist within 20 µm. However, the spatial queries and GT-mapping isolated only 2 mutually exclusive endpoint scenarios that met all criteria. In these 2 cases, the Mean False Candidate Cosine Similarity (0.8180) was actually higher than the Mean True Candidate Cosine Similarity (0.6586).
- **Caveats:** extreme class imbalance / very few positives; small sample size

### 120. (Surprise 0.446) No effect found: The frequency of split errors (measured as U-Net fragment components per millimeter of GT cable length) is positively correlated with the morphological complexi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 92 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Assess whether certain neuron morphologies are inherently more prone to deep-learning segmentation fragmentation, which could inform adaptive proofreading confidence thresholds.
- **Conclusion:** The script correctly computed the morphological complexity (branching nodes per millimeter of cable length) for 19 valid human-traced neurons and quantified the U-Net fragment split rate by mapping automated fragments to the ground-truth skeletons using a spatial KDTree query. A summary table detailing the length, branch density, and split rate for each neuron was successfully generated. The statistical analysis evaluated the hypothesis that split frequency is positively correlated with neuron morphological complexity.
- **Caveats:** none noted

### 121. (Surprise 0.446) Belief dropped: The topological boundary where a single U-Net fragment incorrectly fuses two distinct GT neurons exhibits a statistically significant spike in neurite radius va...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 102 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Determine if anomalous fluctuations in the estimated skeleton radius can pinpoint the exact node where a false merge occurs.
- **Conclusion:** The analysis mapped fragment nodes to ground-truth components, effectively identifying 266 merge boundaries and successfully sampling 1,330 control windows. The findings support the hypothesis that topological boundaries where a U-Net incorrectly fuses distinct GT neurons exhibit anomalous fluctuations in neurite radius. Specifically, the average maximum radius gradient at merge boundaries (0.1418) was substantially higher than in control segments (0.0502), demonstrating a highly statistically significant difference (Welch's t-test: t = 6.2148, p < 0.001).
- **Caveats:** none noted

### 122. (Surprise 0.446) Rejected: Neurons with higher morphological complexity (higher branch density) suffer from a higher rate of split errors per millimeter of cable length, independent of th...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 117 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Correlate ground-truth neuron branch density with the normalized split rate.
- **Conclusion:** The experiment successfully analyzed the relationship between ground-truth (GT) neuron branch density and the normalized split rate (fragments per mm of cable length) across the 19 GT components. The data showed GT neurons ranging in total cable length from ~97.69 mm to ~560.73 mm, with branch densities between 0.92 and 2.09 branches per mm. Split rates (used as a proxy for reconstruction fragmentation) varied from 0.11 to 0.37 splits per mm.
- **Caveats:** none noted

### 123. (Surprise 0.446) Rejected: Continuously updating a fragment's component properties (like total cable length and degree) during an iterative reconnection process significantly reduces the...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 145 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Demonstrate that dynamically refreshing graph component state (agentic replanning) prevents true reconnections from being rejected due to stale topology constraints.
- **Conclusion:** The experiment successfully simulated and compared two pipelines (Static and Dynamic) for reconnecting fragmented neuron skeletons based on spatial proximity (<= 15 µm) and alignment. The Static pipeline applied cycle prevention via Union-Find but without node degree limits, while the Dynamic pipeline continuously updated graph components and enforced a strict degree limit (degree == 1) to avoid unnatural branching. Results showed that the Static pipeline had a slightly higher recall (95.93%) with 330 true positives (TPs) accepted, but 14 TPs were blocked by cycles.
- **Caveats:** none noted

### 124. (Surprise 0.446) Rejected: The segmentation model is more likely to falsely terminate a fragment at anatomically complex junctions; thus, the ground-truth nodes nearest to fragment leaf n...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 151 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Determine if unpredicted splits correlate with areas of high topological complexity (branching) in the underlying ground truth.
- **Conclusion:** It extracted 39,826 leaf nodes (degree 1) from the fragments graph and sampled an equal number of internal nodes (degree 2). Using a KD-tree with a 5 µm threshold, it matched these nodes to the ground-truth (GT) graph, finding 7,050 valid matches for leaf nodes and 9,270 for internal nodes. The statistical analysis revealed that leaf-associated GT nodes had a mean degree of 1.533 (Max: 5), whereas internal-associated GT nodes had a mean degree of 2.001 (Max: 3).
- **Caveats:** none noted

### 125. (Surprise 0.446) No effect found: False-positive leaf reconnections are, on average, spatially closer (smaller Euclidean gap) than true-positive leaf reconnections, exposing a vulnerability in g...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 157 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Analyze the spatial gap distance of potential reconnection proposals to see if proximity disproportionately favors false positives.
- **Conclusion:** The dataset was successfully loaded, and spatial queries identified 434 potential split reconnection proposals (leaf pairs from different fragment components) within a 25 µm Euclidean search radius. By projecting these proposals onto the ground-truth graph, the script classified them into 431 True reconnections and only 3 False reconnections. Statistical analysis showed no significant difference in Euclidean gap distance between the two classes.
- **Caveats:** extreme class imbalance / very few positives; small sample size

### 126. (Surprise 0.446) Rejected: Ground-truth neurons with high morphological complexity (measured by branching node density per millimeter of cable) suffer from a significantly higher rate of...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 161 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Test if biological neuron complexity is a predictive confounder for deep-learning segmentation failures.
- **Conclusion:** The experiment successfully tested the hypothesis that higher morphological complexity (branch density) in ground-truth neurons correlates with a higher rate of split errors in the automated U-Net reconstruction. The programmer resolved previous environment issues by dynamically installing the required packages. By calculating the branching density for each of the 19 ground-truth components and mapping U-Net fragments via a spatial KDTree with majority voting, the pipeline successfully computed the split rates (fragments per mm of cable).
- **Caveats:** none noted

### 127. (Surprise 0.446) Belief dropped: Putative reconnections between fragment leaves that exhibit high orientation agreement (i.e., their terminal tangent vectors are highly aligned) have a signific...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 166 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Test whether angular alignment between terminal branches of fragment leaves is a significantly better discriminator for true positive reconnections than spatial proximity.
- **Conclusion:** A total of 1,812 putative inter-component leaf pairs within 30 µm were identified, and 484 were confidently mapped to the ground truth and evaluated. The results show that terminal angular alignment does not provide a statistically significant improvement over spatial proximity alone for predicting valid reconnections in this specific range. The High Alignment group (> 150 degrees) achieved a True Positive Rate (TPR) of 100% (177 TP, 0 FP), while the Low Alignment group (<= 150 degrees) achieved a TPR of 98.37% (302 TP, 5 FP).
- **Caveats:** none noted

### 128. (Surprise 0.446) No effect found: Short terminal branches in the automated reconstruction (path length to the nearest branch point < 15 µm) are significantly more likely to be structural false p...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 184 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Determine if automated pruning of short terminal branches can cleanly remove false-positive topologies without sacrificing true neurite cable.
- **Conclusion:** The experiment executed successfully and evaluated the terminal branch lengths for 39,826 leaf nodes in the automated reconstruction. The analysis classified only 7 terminal branches as 'Short' (< 15 µm) and 39,819 as 'Long' (≥ 15 µm). Ground truth mapping using a 10 µm distance threshold revealed a false-positive rate of 85.71% for short branches (6 out of 7) and 81.68% for long branches (32,523 out of 39,819).
- **Caveats:** small sample size; test driven to p=1.0 by near-zero positive counts

### 129. (Surprise 0.446) Rejected: U-Net fragments that are implicated in cycle-violating reconnections during a greedy split-correction pass have a significantly higher probability of harboring...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 191 · **Belief:** Leaning True → Uncertain (0.708→0.422) · **Direction:** Negative · **Surprisal:** −0.446
- **Tested:** Investigate if tracking graph-cycle-prevention events during split correction can serve as an agentic flag to dynamically identify undiscovered false merges.
- **Conclusion:** The experiment successfully tested the hypothesis that U-Net fragments implicated in cycle-violating reconnections during split correction have a significantly higher probability of containing a false merge. The code simulated a greedy split-correction process (radius ≤ 15 µm), evaluating 1,209 valid proposals. It rejected 33 proposals to prevent topological cycles, thereby implicating 59 unique fragments.
- **Caveats:** none noted

### 130. (Surprise 0.408) Rejected: In competitive reconnection scenarios where a fragment leaf has multiple candidates within 20 µm, selecting the candidate that yields the most continuous neurit...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 15 · **Belief:** Leaning True → Uncertain (0.708→0.446) · **Direction:** Negative · **Surprisal:** −0.408
- **Tested:** Test radius continuity as a heuristic for resolving mutually exclusive reconnection proposals at a single endpoint.
- **Conclusion:** The code successfully identified competitive fragment leaves within 20 µm and mapped them to the Ground Truth (GT). However, due to the sparse nature of the GT (only 18 neurons), the strict filtering criteria (exactly one valid candidate mapping to the true GT neuron within 10 µm) yielded only 2 competitive reconnection scenarios. For these 2 cases, the radius continuity heuristic successfully resolved 0 of them, resulting in 0% accuracy against a random baseline of 50%.
- **Caveats:** small sample size

### 131. (Surprise 0.398) Rejected: In dense neurite regions, invalid (false-positive) reconnection proposals are spatially shorter, on average, than valid (true-positive) reconnection proposals,...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 198 · **Belief:** Leaning True → Uncertain (0.708→0.453) · **Direction:** Negative · **Surprisal:** −0.398
- **Tested:** Determine if simple proximity checks inherently bias proofreading toward false merges in dense regions, motivating the need for reversible decisions and rollback mechanisms in agentic frameworks.
- **Conclusion:** The algorithm identified 4,132 fragment leaf nodes within the top 5% highest density regions and found 535 unique leaf pairs within a 30 µm radius. By mapping to the ground truth, 121 pairs were classified as valid (true-positive) and 3 pairs as invalid (false-positive). The valid proposals had a median gap length of 9.70 µm, while the invalid proposals had a median of 13.49 µm.
- **Caveats:** extreme class imbalance / very few positives; small sample size

### 132. (Surprise 0.365) Rejected: The local tortuosity (ratio of path length to Euclidean distance) of neurite segments leading up to true-positive split endpoints is significantly higher than t...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 115 · **Belief:** Leaning False → Likely False (0.375→0.141) · **Direction:** Negative · **Surprisal:** −0.365
- **Tested:** Test whether U-Net fragmentation (split errors) occurs more frequently in highly tortuous (curvy) regions of the neurite.
- **Conclusion:** The experiment successfully executed and tested the hypothesis. The programmer resolved the previous dependency and file path issues by implementing a robust `CustomUnpickler` and dynamically checking for the dataset file. Experiment Execution & Findings: - Endpoint Isolation & Classification: The code identified 39,826 leaf nodes (endpoints) in the fragments graph.
- **Caveats:** none noted

### 133. (Surprise 0.365) Supported: The density of merge errors (number of false-positive branch points per 1,000 µm of cable length) is significantly higher within a 150 µm radius of soma centroi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 172 · **Belief:** Leaning True → Likely True (0.625→0.859) · **Direction:** Positive · **Surprisal:** +0.365
- **Tested:** Determine if proximity to soma locations is a primary driver of U-Net merge errors due to localized high neurite density.
- **Conclusion:** The dataset revealed a total of 36 merge errors. In the proximal region (<= 150 µm from a soma), the cable length was ~38,524 µm with 2 merge errors, resulting in a density of 0.0519 merges per 1,000 µm. In the distal region (> 150 µm), the cable length was ~18,772,622 µm with 34 merge errors, yielding a density of 0.0018 merges per 1,000 µm.
- **Caveats:** none noted

### 134. (Surprise 0.333) No effect found: True split-correction proposals (pairs of leaf nodes from different fragments that map to the same ground-truth neuron) have significantly higher angular alignm...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 81 · **Belief:** Leaning True → Uncertain (0.792→0.578) · **Direction:** Negative · **Surprisal:** −0.333
- **Tested:** Evaluate if angular alignment of fragment endpoints can effectively filter false reconnection proposals during split correction.
- **Conclusion:** The code correctly located and loaded the dataset, identified 39,826 leaf nodes, and found 1,397 candidate pairs from different components within a 20 µm radius. By mapping to the ground truth, it classified 350 pairs as True Connections and 3 pairs as False Connections. The mean angular alignment (cosine similarity) for True Connections was 0.8719 compared to 0.7123 for False Connections.
- **Caveats:** small sample size

### 135. (Surprise 0.325) Supported: The topological split rate (number of fragment breaks per millimeter of ground-truth cable) decreases significantly as the geodesic path distance from the soma...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 129 · **Belief:** Uncertain → Leaning True (0.417→0.625) · **Direction:** Positive · **Surprisal:** +0.325
- **Tested:** Quantify whether neuronal fragmentation rates are spatially biased by their topological distance from the soma, which could inform region-adaptive proofreading sweeps.
- **Conclusion:** The experiment was successfully executed and the code correctly implemented the required logic to test the hypothesis. The results show that there is a statistically significant negative linear relationship (p ≈ 0.009) between the geodesic distance from the soma and the topological split rate (splits per mm). This confirms the hypothesis that U-Net fragmentation rates decrease significantly in distal projections compared to the densely tangled neurite regions nearer to the cell body.
- **Caveats:** extreme class imbalance / very few positives

### 136. (Surprise 0.284) Rejected: Fragment leaf nodes that represent 'true splits' (reconnectable to the same ground-truth neuron) have significantly larger neurite radii than leaf nodes that ar...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 40 · **Belief:** Leaning False → Likely False (0.292→0.109) · **Direction:** Negative · **Surprisal:** −0.284
- **Tested:** Assess whether the neurite radius at an endpoint can serve as a confidence prior for generating valid split-correction proposals.
- **Conclusion:** The code identified 39,826 leaf nodes in the UNet fragments and successfully matched pairs located within a 20 µm radius. By comparing the local connectivity to the ground truth, 2,687 true splits (where both leaves belong to the same GT neuron) and 11 false splits (artifacts or erroneous pairings) were identified. The descriptive statistics show the true splits had a mean radius of 1.3045 µm (median = 1.2490 µm), while false splits had a mean radius of 1.2250 µm (median = 1.0000 µm).
- **Caveats:** extreme class imbalance / very few positives; small sample size

### 137. (Surprise 0.284) Supported: Connecting true split fragments (True Positives) creates local pathways with significantly lower tortuosity (arc-chord ratio) compared to falsely connecting unr...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 4 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** To evaluate if the local tortuosity of a reconnected path is a viable metric to prevent artifactual, sharp bends in the proofread skeleton.
- **Conclusion:** The script generated 479 True Positive (TP) and 5 False Positive (FP) connection proposals within a 30 µm radius. The tortuosity analysis revealed that valid connections (TP) have a noticeably lower mean tortuosity (1.3872) compared to invalid connections (FP, mean = 2.3972). A Mann-Whitney U test confirmed that this difference is statistically significant (p = 0.000255).
- **Caveats:** small sample size

### 138. (Surprise 0.284) Supported: Severing false U-Net connections (merge errors) creates new terminal nodes that immediately unblock previously hidden, valid true-positive reconnection proposal...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 5 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Demonstrate that coupling merge and split correction is necessary, as correcting a merge directly creates the endpoints needed for valid split reconnections.
- **Conclusion:** The experiment successfully executed and validated the hypothesis. By identifying and severing 292 false U-Net connections (merge edges) based on ground-truth mappings, the simulation created new terminal nodes. Querying neighbors within a 30 µm radius of these newly created leaves uncovered 574 new True Positive (valid) split-correction proposals that were previously inaccessible.
- **Caveats:** none noted

### 139. (Surprise 0.284) Supported: U-Net merge errors are statistically correlated with spatial regions of high structural complexity (high local branching density).
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 6 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** To evaluate if erroneous fragment fusions occur disproportionately in dense neuropil where the local branching density is high.
- **Conclusion:** By mapping the automated U-Net `fragments_graph` to the human-traced `gt_graph`, the script successfully isolated 23 U-Net fragments that erroneously bridge multiple distinct ground truth neurons. From these, 505 merge transition nodes were identified. To evaluate local structural complexity, the branching density (number of degree >= 3 nodes within a 50 µm spatial radius) was computed for both these merge regions and an equal-sized, unbiased sample of safe regions from non-merged fragments.
- **Caveats:** none noted

### 140. (Surprise 0.284) Supported: Regions with a high spatial density of fragment endpoints (fragmentation hubs) have a significantly higher rate of valid split reconnections than isolated endpo...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 15 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Test if local endpoint density correlates with the likelihood of successful topology recovery, indicating that proofreading algorithms should dynamically prioritize these hubs.
- **Conclusion:** The experiment successfully tested the hypothesis by calculating the local spatial density of fragment endpoints and comparing their topological reconnection rates. Endpoints were clustered based on a 30 µm radius. The threshold for 'High Density' was adjusted to ≥1 nearby endpoint due to the sparseness of the overall distribution, while 'Low Density' endpoints had 0 neighbors within that radius.
- **Caveats:** none noted

### 141. (Surprise 0.284) Supported: Sudden changes or high variance in neurite radius are predictive of merge errors, occurring at U-Net artifact sites where two closely passing distinct ground-tr...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 19 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Determine if the gradient of the `node_radius` attribute along a skeleton can serve as a local image-evidence feature to detect and flag topological merge boundaries.
- **Conclusion:** The experiment successfully tested the hypothesis that sudden changes in neurite radius correlate with merge errors. By mapping U-Net nodes to the ground truth using a KD-tree, the script identified 475 merge boundary nodes (points where a single U-Net fragment erroneously connects distinct GT neurons) and compared them against a random sample of 10,000 normal, non-boundary nodes. The statistical analysis revealed that the mean local radius gradient at merge boundaries was 0.0498 µm, compared to just 0.0096 µm for normal nodes.
- **Caveats:** implementation deviated from plan

### 142. (Surprise 0.284) Supported: U-Net split errors disproportionately occur near complex morphological structures like branch points.
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 35 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** To determine if split errors are spatially correlated with neurite branch points, implying morphological complexity induces segmentation breaks.
- **Conclusion:** The experiment successfully tested the hypothesis by comparing the spatial proximity of leaf nodes (representing U-Net split errors) and middle nodes (representing continuous segments) to the nearest branching nodes. Analyzing 39,826 leaf nodes and an equal number of randomly sampled middle nodes revealed a stark contrast: leaf nodes were found significantly closer to branching nodes (mean = 217.40 µm, median = 151.20 µm) compared to middle nodes (mean = 266.92 µm, median = 188.93 µm). A Mann-Whitney U test confirmed this difference is highly statistically significant (U = 702045331.0, p-value = 2.08e-173).
- **Caveats:** none noted

### 143. (Surprise 0.284) Supported: Resolving U-Net merge errors unblocks valid split reconnections: correcting false fusions exposes hidden topological endpoints and significantly increases the p...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 41 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** To demonstrate that coupling merge and split correction is necessary because merge errors mask true fragment endpoints.
- **Conclusion:** The programmer resolved the file path issue from the previous run, and the code executed to completion. The script correctly mapped UNet fragment nodes to Ground-Truth (GT) components and identified 4,536 merge edges. The baseline evaluation found 1,391 valid split proposals.
- **Caveats:** none noted

### 144. (Surprise 0.284) Supported: Fragment endpoint pairs that belong to the SAME fragment component, are separated by < 15 µm in Euclidean space, but > 100 µm in graph path length, represent 'k...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 1 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Identify and characterize structural 'cycles' or auto-merges within single predicted fragments that mimic false reconnections.
- **Conclusion:** The code identified 'kissing loop' auto-merges across fragment components by locating nodes within the same component that are physically close (< 15 µm Euclidean distance) but topologically distant (> 100 µm shortest path length). It compared the radius variance of the connecting paths for 500 such kissing loop pairs against 494 randomly sampled baseline non-looping paths. The results validate the hypothesis: the structural paths characterizing kissing loops exhibit significantly higher radius variance than typical straight pathways.
- **Caveats:** none noted

### 145. (Surprise 0.284) Supported: Reconnection proposals bridging large fragment components (measured by total node count) are significantly more likely to be valid than proposals involving smal...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 17 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Test if component size acts as a reliable prior for distinguishing genuine neuron segments from noise-induced short fragments.
- **Conclusion:** The experiment successfully installed the missing dependencies and executed the steps to test the hypothesis. It identified 39,826 leaf nodes and found 1,618 pairs within a 20 µm radius. The pairs were mapped to the ground-truth data, resulting in 408 valid and 989 invalid reconnections (after ignoring intra-component pairs).
- **Caveats:** none noted

### 146. (Surprise 0.284) Supported: Merge errors predominantly occur at complex topological junctions; therefore, U-Net nodes at the 'boundaries' of a merge are located significantly closer to U-N...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 20 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Investigate whether merge errors concentrate near structural branching points in the automated reconstruction.
- **Conclusion:** The experiment was successfully executed and yielded statistically significant results. The code correctly mapped fragments to ground truth, identified 2,119 merge fragments, and extracted 8,285 merge boundary nodes. The mean distance to the nearest branch node was 282.21 µm for boundary nodes and 315.19 µm for baseline nodes.
- **Caveats:** none noted

### 147. (Surprise 0.284) Supported: Split errors are spatially clustered, meaning the physical distance from one true split gap to the nearest neighboring true split gap is significantly smaller t...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 36 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Determine if U-Net split errors occur in localized 'hotspots' (e.g., due to regional imaging artifacts or dense crossing regions) rather than being uniformly distributed.
- **Conclusion:** A total of 395 true split gaps were mapped to the ground truth. The results from the Kolmogorov-Smirnov test (KS Statistic = 0.2411, p-value = 4.0716e-19) indicate a highly significant difference between the two distributions. The true split gaps have a substantially smaller mean (634.00 µm) and median (480.64 µm) nearest-neighbor distance compared to the random null distribution (Mean = 920.95 µm, Median = 781.18 µm).
- **Caveats:** none noted

### 148. (Surprise 0.284) Supported: The neurite radius at the endpoints of U-Net fragments is systematically smaller than the average radius of the entire fragment, indicating that U-Net fragmenta...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 41 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Analyze the correlation between neurite radius and the occurrence of split errors by comparing endpoint radii to their parent fragment's mean radius.
- **Conclusion:** The code correctly installed dependencies, loaded the `fragments_graph`, computed the global mean radius versus the mean endpoint radius for 10,172 eligible U-Net fragments, and performed a paired t-test. Experiment Results & Findings: - Summary Statistics: The global average neurite radius across the fragments was calculated as 1.9463 µm (std: 0.0529). In contrast, the average radius at the endpoints (leaf nodes) was substantially smaller at 1.5078 µm (std: 0.2686).
- **Caveats:** none noted

### 149. (Surprise 0.284) Supported: Split gaps located near structural bifurcations exhibit significantly lower angular alignment (larger deviation from a straight line) between their endpoints co...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 45 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Investigate how the geometry of a split error changes based on its proximity to neuron branching points, informing dynamic search cone generation.
- **Conclusion:** The experiment was successfully executed and the results strongly support the initial hypothesis. The code loaded the dataset using the dynamically generated mocked classes, successfully extracted fragment leaf nodes and ground-truth branching nodes, and computed the true split gaps within 15 µm. Out of 1204 identified true split gaps, 19 were located near a bifurcation (< 10 µm) and 1185 were along the backbone (>= 10 µm).
- **Caveats:** implementation deviated from plan

### 150. (Surprise 0.284) Supported: For mutually exclusive reconnection candidates originating from a single fragment endpoint, the angular difference in terminal branch orientation (tangent vecto...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 51 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Compare the discriminative power of tangent orientation versus Euclidean distance for resolving competing reconnection proposals.
- **Conclusion:** The experiment was successfully executed and robustly evaluated the hypothesis. The script correctly installed all missing dependencies and processed the dataset to extract 39,826 leaf nodes from the fragments graph. After mapping these to the ground truth (GT) within a 5 µm threshold, 7,050 valid leaves were retained.
- **Caveats:** none noted

### 151. (Surprise 0.284) Supported: Nodes within 10 µm of a merge error boundary in U-Net fragments exhibit significantly higher local variance in neurite radius than nodes in correctly reconstruc...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 55 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Test if sudden fluctuations in predicted neurite radius can serve as a geometric signature for localizing merge errors.
- **Conclusion:** The experiment successfully executed and validated the hypothesis. By mapping U-Net fragments to Ground Truth components, the script identified 23 merge fragments and 1042 pure (non-merged) fragments, locating 309 merge boundary edges. The local neurite radius variance within 10 µm topological neighborhoods was calculated for both merge boundary windows and equivalent random windows in pure fragments.
- **Caveats:** none noted

### 152. (Surprise 0.284) Supported: More than 30% of valid split reconnections (gap < 15 µm) require connecting a fragment leaf node to an internal node (degree > 1) of another fragment (a T-junct...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 61 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Challenge the standard assumption that split correction solely requires matching endpoint to endpoint, quantifying the need for 'dead-end' T-junction proposals.
- **Conclusion:** The script analyzed 39,826 fragment leaf nodes to find valid topological reconnections within a 15 µm radius. Out of 4,314 valid reconnections identified, 1,816 were leaf-to-leaf (endpoint) reconnections, while 2,498 were leaf-to-internal (T-junction) reconnections. This yielded a T-junction proportion of ~57.90%.
- **Caveats:** extreme class imbalance / very few positives

### 153. (Surprise 0.284) Supported: Nodes in the `fragments_graph` that constitute a merge error (where a single U-Net fragment fuses two different GT neurons) exhibit a significantly higher local...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 62 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** To evaluate if localized spikes or high variance in predicted neurite radius can serve as a reliable geometric signature for detecting and rolling back false merges.
- **Conclusion:** The programmer resolved the previous file path issue and correctly installed necessary missing packages. The script effectively mapped fragment nodes to ground-truth nodes using a 5 µm distance threshold, identifying 290 merge interfaces (where a single U-Net fragment incorrectly fused two distinct ground-truth components) and randomly sampled an equal number of control windows from cleanly reconstructed fragments. Local features for a 20-node window around these edges were extracted, specifically the standard deviation and max-to-median ratio of the neurite radius.
- **Caveats:** small sample size; implementation deviated from plan

### 154. (Surprise 0.284) Supported: The predicted neurite radius at fragment leaf nodes involved in valid split errors is significantly smaller than the average internal radius of the fragment, in...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 72 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Investigate whether valid split errors occur primarily due to genuine physical thinning of the cable (fading signal) rather than random stochastic failures in thick cables.
- **Conclusion:** The experiment successfully executed and effectively tested the hypothesis. The script correctly identified 687 True Positive (valid) split errors by matching UNet fragment leaf pairs (within 15 µm) to the same ground-truth (GT) component. It then successfully compared the estimated neurite radius at these terminal leaves against the mean internal radius of their parent components.
- **Caveats:** extreme class imbalance / very few positives

### 155. (Surprise 0.284) Supported: The local density of fragment endpoints is significantly higher around valid split gaps than around correctly reconstructed continuous backbone nodes, indicatin...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 86 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Test if local endpoint density can serve as a contextual meta-feature that shifts dynamically as edits accumulate, signaling regions where single-pass greedy reconnections are likely to fail.
- **Conclusion:** The experiment was successfully executed and robustly evaluated the hypothesis. The implementation correctly extracted 1,391 valid split gaps and sampled an equal number of continuous backbone nodes. Using a 50 µm search radius, the algorithm calculated the local density of fragment endpoints (leaf nodes).
- **Caveats:** none noted

### 156. (Surprise 0.284) Supported: Merge errors (erroneous fusions of distinct neurons) occur disproportionately at U-Net fragment branch points (nodes with degree >= 3) rather than along continu...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 88 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Determine the graph-topological signature of merge errors to guide targeted, reversible un-merging algorithms.
- **Conclusion:** By mapping the UNet fragment nodes to the human ground-truth (GT) graph with a 5 µm threshold, the algorithm successfully identified 290 merge edges involving 468 distinct nodes (where adjacent fragment nodes mapped to different GT components). Analysis of the node degrees revealed a strong correlation between merge errors and structural branch points (nodes with a degree of 3 or higher). Globally, branch nodes are rare, constituting only 0.45% of the entire fragments graph (19,460 out of ~4.28 million nodes).
- **Caveats:** implementation deviated from plan

### 157. (Surprise 0.284) Supported: Branch points in the automated reconstruction that represent merge errors (false fusions) have a significantly higher local density of unconnected leaf nodes wi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 95 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Test the spatial correlation between merge errors and split errors to determine if detecting merges can guide where to search for valid reconnections.
- **Conclusion:** The experiment was successfully executed and generated the required summary statistics. The script correctly identified 19,460 branch nodes, extracted their neighbors, and mapped them to the ground truth to classify 5,586 as True Branches and 36 as Merge Errors. Using a KDTree for leaf nodes, the local density of unconnected leaf nodes (within 20 µm) was queried for each branch class.
- **Caveats:** extreme class imbalance / very few positives; implementation deviated from plan

### 158. (Surprise 0.284) Supported: Reconnection proposals that form small spatial cycles (e.g., three fragment endpoints mutually within 20 µm forming a triangle in the proposal graph) have a sig...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 96 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Assess how local proposal graph density impacts precision, testing if cycles inherently indicate a high false-positive rate due to anatomical constraints.
- **Conclusion:** The script identified 1,397 total gap proposals within a 20 µm radius. By distinguishing between proposals participating in cyclic structures versus those forming linear (bridge) proposals, and mapping these to ground truth (with a 50 µm tolerance to capture more mappings), the script built a contingency table. The results revealed 21 cyclic proposals (19 valid, 2 invalid; 90.48% precision) and 395 linear proposals (394 valid, 1 invalid; 99.75% precision).
- **Caveats:** none noted

### 159. (Surprise 0.284) Supported: Breaking U-Net components at known false merge sites prior to split correction increases the recall of true-positive reconnections by at least 10% because it re...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 101 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Evaluate the synergistic effect of coupling merge correction (fragment splitting) before split correction (fragment joining).
- **Conclusion:** The experiment successfully tested the hypothesis that performing merge correction before split correction increases the recall of true-positive reconnections. The script loaded the dataset, mapped the U-Net fragments to ground-truth components, and applied a greedy distance-and-tangent-based split-correction algorithm to establish a baseline. The baseline yielded 294 true-positive (TP) reconnections and 881 false-positive (FP) reconnections.
- **Caveats:** implementation deviated from plan

### 160. (Surprise 0.284) Supported: Split errors in the U-Net reconstruction are significantly concentrated around complex topological junctions, specifically occurring more frequently within a 30...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 103 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** To determine if U-Net fragmentation is disproportionately caused by anatomical bifurcations, validating if split-correction models should apply different priors near known branching structures.
- **Conclusion:** Ground-truth (GT) branching nodes (N=7,320) were correctly identified alongside an equal sample size of isolated control nodes (degree=2, ≥50 µm from any branch). Fragment leaves mapping to non-leaf GT nodes were filtered to identify true split errors (3,907 errors mapped to 3,774 unique GT nodes). By evaluating a 30 µm path-length neighborhood around each target, the analysis revealed that branching regions contained an average of 0.0941 split errors, compared to just 0.0354 near control regions.
- **Caveats:** small sample size

### 161. (Surprise 0.284) Supported: Incrementally resolving the highest-confidence split reconnections (distance < 5 µm) alters the local graph topology such that the average proposal density (com...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 106 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Verify the benefit of an agentic, multi-pass graph update system by showing that early, easy decisions clean up the feature space for harder decisions.
- **Conclusion:** The algorithm identified 39,826 fragment leaf nodes. Out of these, 243 high-confidence pairs (distance < 5 µm, tangent alignment > 0.90) were identified and successfully merged using a greedy strategy and updated via union-find. For the remaining 39,340 endpoints, the average proposal density (competing endpoints within 15 µm) dropped from 0.0488 to 0.0484.
- **Caveats:** none noted

### 162. (Surprise 0.284) Supported: Due to the anisotropic resolution of ExaSPIM imaging (poorer resolution in the Z-axis), split errors are disproportionately aligned with the Z-axis compared to...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 108 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Investigate if image anisotropy biases the orientation of automated reconstruction breaks.
- **Conclusion:** The experiment successfully tested the hypothesis that split errors are disproportionately aligned with the Z-axis due to anisotropic resolution. The script loaded the dataset and extracted valid split-correction gaps (336 true-positive reconnections) and compared them against a massive baseline of true biological edges (~1.36 million GT edges). The mean absolute angle for biological edges was 61.38 degrees, heavily skewed toward the XY plane (near 90 degrees).
- **Caveats:** none noted

### 163. (Surprise 0.284) Contradicted: During a greedy, single-pass split-correction algorithm, strict cycle-prevention constraints erroneously block a significant percentage (at least 5%) of anatomi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 113 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Quantify the collateral damage of enforcing cycle-free graphs during greedy split correction by measuring how many True Positive edges are rejected to prevent cycles.
- **Conclusion:** The script correctly proposed connections between fragment leaves within 15 µm, classified them based on ground-truth mappings, and simulated a greedy, single-pass edge acceptance algorithm using a Union-Find structure. Out of 1,209 proposed inter-component connections, 1,176 were accepted. 33 edges were rejected specifically because they would form a cycle.
- **Caveats:** none noted

### 164. (Surprise 0.284) Supported: True reconnection proposals are predominantly formed between large, stable U-Net fragments, while false merge-inducing proposals disproportionately involve smal...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 114 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Investigate if the combined size (node count) of fragments involved in a reconnection proposal can serve as a predictive feature for the validity of the split.
- **Conclusion:** The code executed successfully and calculated the combined fragment size for candidate split-reconnection pairs within a 20 µm radius. Out of 1,397 valid candidate pairs, 395 were classified as True Pairs (correctly reconnecting to the same ground-truth neuron) and 1,002 were classified as False Pairs (ambiguous or erroneous reconnections). The results showed a stark difference in combined fragment sizes: True Pairs had a mean combined size of 13,058.34 nodes (median: 2,912.00 nodes), whereas False Pairs had a mean combined size of 1,932.99 nodes (median: 914.50 nodes).
- **Caveats:** none noted

### 165. (Surprise 0.284) Supported: In dense regions with competitive split-correction proposals (a fragment leaf node having >= 2 nearby leaves from different components within 20 µm), the propos...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 135 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Demonstrate that relying purely on Euclidean distance for split correction is suboptimal in dense regions, motivating the use of multi-feature joint reasoning (mutual exclusivity) at competitive endpoints.
- **Conclusion:** The experiment successfully executed and validated the hypothesis. By analyzing the fragments graph, the code identified 94 competitive leaf nodes (those with 2 or more candidates from different U-Net components within a 20 µm radius). For these competitive endpoints, the most angularly aligned candidate (highest tangent cosine similarity) differed from the spatially closest candidate in 30 cases (31.91%).
- **Caveats:** none noted

### 166. (Surprise 0.284) Supported: Nodes corresponding to anatomical branching points (degree >= 3) in the ground truth exhibit a significantly higher split error rate than standard continuation...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 136 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Test whether topological bifurcations in neuron morphology are systematic failure points (hotspots for splits) for the U-Net segmentation.
- **Conclusion:** The results strongly support the hypothesis: U-Net segmentation introduces split errors at anatomical branching points (nodes with degree >= 3) at a significantly higher rate than along continuous paths (degree == 2). Specifically: - Branching Nodes: Experienced a split error rate of ~1.48% (94 splits out of 6,339 branching nodes). - Path Nodes: Experienced a much lower split error rate of ~0.12% (1,359 splits out of 1,109,694 path nodes).
- **Caveats:** extreme class imbalance / very few positives

### 167. (Surprise 0.284) Supported: Shorter U-Net fragments exhibit significantly higher angular variance (misalignment) at their endpoints relative to the true ground-truth continuation compared...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 139 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Test if the cable length of a U-Net fragment inversely correlates with the reliability of its endpoint tangent vectors as a geometric signature for split correction.
- **Conclusion:** The experiment successfully tested the hypothesis by computing the cable lengths for U-Net fragments, identifying their leaf nodes, and calculating the angular alignment error of the fragment's leaf tangent relative to the ground-truth continuation. Out of 39,826 identified leaf nodes, 3,171 endpoints belonging to short fragments (< 5000 µm) and 4,126 endpoints belonging to long fragments (>= 5000 µm) were aligned and evaluated. The short fragments exhibited a median angular error of 7.34°, compared to 6.60° for long fragments.
- **Caveats:** none noted

### 168. (Surprise 0.284) Supported: False merge errors occur in regions of significantly higher local fragment density (crowding) compared to correctly reconstructed segments, reflecting U-Net seg...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 143 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Quantify whether local fragment crowding is a statistically significant spatial precursor to merge errors.
- **Conclusion:** The analysis successfully identified 468 merge boundary nodes and sampled 2,340 control nodes. The results support the hypothesis that false merge errors occur in regions with significantly higher local fragment density. The average local fragment density within a 30 µm radius was slightly higher at merge boundaries (1.10 fragments) compared to correctly traced control nodes (1.05 fragments).
- **Caveats:** none noted

### 169. (Surprise 0.284) Supported: For split reconnection proposals, the cosine similarity of the outward-facing tangents of connecting endpoints is strongly anti-parallel (approaching -1) for tr...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 144 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Validate that tangent orientation agreement is a robust geometric prior for prioritizing valid split reconnections.
- **Conclusion:** The algorithm identified 39,826 leaf nodes and 1,209 candidate pairs within a 15 µm radius. Out of these, 344 pairs were classified as True Positives (TP) and 3 pairs as False Positives (FP) based on the ground truth mapping. The results strongly validate the hypothesis: True Positives exhibited highly anti-parallel outward tangents (mean cosine similarity of -0.8077, concentrated near -1.0), whereas False Positives exhibited positive alignments (mean cosine similarity of 0.7078).
- **Caveats:** small sample size

### 170. (Surprise 0.284) Supported: For true-positive split reconnections, the directional agreement (tangent alignment) between fragment endpoints degrades significantly as the spatial gap distan...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 160 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Quantify how the geometric predictability of split gaps decays over distance, informing dynamic thresholding for reconnection pipelines.
- **Conclusion:** The experiment successfully tested the hypothesis by analyzing the true-positive split reconnections between fragment endpoints. Using a spatial radius threshold of 25 µm, 376 true-positive (TP) reconnection pairs were identified. The outward-facing tangent vectors were extracted for these endpoints to compute a tangent alignment score (where +1 represents ideal anti-parallel outward vectors, i.e., leaves perfectly facing each other).
- **Caveats:** none noted

### 171. (Surprise 0.284) Supported: Reconnection candidates where both participating U-Net fragments are extremely short (under 2,000 µm in total cable length) have a False Positive rate that is a...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 167 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Determine if fragment cable length acts as a reliable structural prior for split-correction confidence.
- **Conclusion:** The experiment successfully tested the hypothesis that fragment cable length acts as a reliable structural prior for split-correction confidence. The code correctly computed the total cable length for components and queried valid leaf-to-leaf pairs within a 20 µm radius, resulting in 1,397 candidates. These were appropriately categorized into the two bins: 'Short-Short' (both fragments < 2000 µm, yielding 437 candidates) and 'Long-Any' (at least one fragment > 5000 µm, yielding 537 candidates).
- **Caveats:** none noted

### 172. (Surprise 0.284) Supported: Applying true-positive split reconnections significantly increases the statistical variance of the fragment cable lengths, meaning split correction primarily el...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 170 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Test the hypothesis that fixing topological splits significantly increases the variance of component cable lengths, indicating a dynamic shift in graph state post-correction.
- **Conclusion:** The experiment successfully tested the hypothesis regarding the distributional impact of split correction on fragment cable lengths. The baseline (pre-correction) metrics across 10,172 components showed a mean cable length of 1849.31 µm and a variance of 27.86M µm². The script identified and simulated 1,157 valid True Positive (TP) reconnections.
- **Caveats:** none noted

### 173. (Surprise 0.284) Supported: For true split errors, the spatial gap distance between fragment endpoints is negatively correlated with their orientation agreement (cosine similarity of branc...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 174 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Evaluate the geometric signatures of recoverable splits to determine the trade-off between gap size and tangent alignment reliability.
- **Conclusion:** The experiment was successfully executed and robustly validates the hypothesis. By analyzing 647 valid true split pairs (fragment endpoints mapping to the same ground-truth neuron), a significant negative correlation was found between the spatial gap distance and the orientation agreement (Pearson r = -0.4619, p-value = 1.64e-35). Furthermore, the data binned by gap distance showed a clear degradation in alignment as the gap increased: mean orientation agreement was high (0.8320) for gaps of 0-10 µm, moderately high (0.7733) for 10-20 µm, and significantly dropped (0.5582) for 20-50 µm.
- **Caveats:** none noted

### 174. (Surprise 0.284) Supported: U-Net fragmentation (split errors) occurs disproportionately in regions of thin neurites, such that the estimated neurite radius at the leaf nodes of fragments...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 176 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Determine if weak local image evidence (thin neurites) is a primary geometric factor predisposing the automated U-Net segmentation to split errors.
- **Conclusion:** The experiment successfully tested the hypothesis regarding the geometry of U-Net split errors. By explicitly casting the `float16` node radii arrays to `float64`, numerical overflow was prevented, allowing for accurate statistical computation. The results show that the mean radius of leaf nodes (representing fragment endpoints or split errors) is 1.5761 µm (variance = 0.2034), whereas the mean radius of continuous internal nodes is significantly larger at 1.9676 µm (variance = 0.0072) based on an equal sample size of 39,826 nodes.
- **Caveats:** none noted

### 175. (Surprise 0.284) Supported: Due to the lower optical resolution along the Z-axis (1.0 µm/voxel) compared to the X/Y axes (0.748 µm/voxel), U-Net split errors occur disproportionately when...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 177 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Evaluate if image anisotropy introduces a directional bias in the occurrence of split errors.
- **Conclusion:** A total of 39,826 terminal segments (representing split points) and an equal number of randomly sampled internal segments (continuous pathways) were compared. The terminal segments showed a mean Z-alignment (|cos(θ)|) of 0.4934 (variance: 0.0886), while the internal segments had a mean of 0.4760 (variance: 0.0845). Statistical testing confirmed this difference is highly significant, with Welch's t-test yielding a t-statistic of 8.3243 (p-value = 8.6160e-17) and a Kolmogorov-Smirnov test yielding a KS-statistic of 0.0531 (p-value = 3.3276e-49).
- **Caveats:** none noted

### 176. (Surprise 0.284) Supported: The predicted neurite radius in U-Net fragments tapers off (systematically decreases) as the skeleton approaches a split endpoint compared to the interior of th...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 178 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Measure the spatial variation of predicted neurite radius near fragment endpoints to understand the localized failure modes of the U-Net segmentation.
- **Conclusion:** The experiment was successfully executed and robustly validates the hypothesis. By analyzing 1,745 large fragment components (length > 2000 µm), nodes were grouped by their hop distance to the nearest fragment extremity. The results demonstrate a clear and progressive tapering of the predicted neurite radius as the skeleton approaches an endpoint.
- **Caveats:** small sample size

### 177. (Surprise 0.284) Supported: U-Net fragments situated physically closer to the neuron soma are significantly longer on average than distal fragments, due to thicker and more easily traceabl...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 194 · **Belief:** Leaning True → Likely True (0.708→0.891) · **Direction:** Positive · **Surprisal:** +0.284
- **Tested:** Investigate the spatial distribution of fragmentation errors by correlating fragment length with distance to the soma.
- **Conclusion:** The pipeline evaluated 1,146 U-Net fragments, calculating their total cable length and minimum Euclidean distance to their corresponding ground-truth soma. The statistical analysis yielded a Spearman Rank Correlation coefficient of -0.1379 with a highly significant p-value of 2.7850e-06. These results confirm the hypothesis: there is a statistically significant negative correlation between distance-to-soma and fragment length, meaning fragments situated physically closer to the neuron soma tend to be longer than those further away.
- **Caveats:** none noted

### 178. (Surprise 0.260) Belief dropped: In dense neighborhoods where a fragment leaf has multiple reconnection candidates within 30 µm, the purely nearest-neighbor (Euclidean distance) candidate is fr...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 21 · **Belief:** Leaning True → Leaning True (0.792→0.625) · **Direction:** Negative · **Surprisal:** −0.260
- **Tested:** Quantify the failure rate of distance-only reconnection heuristics and evaluate the performance gain of a composite feature metric.
- **Conclusion:** However, the analysis revealed that only 2 fragment leaves met the strict 'dense' criteria (having both True and False candidates from different UNet fragments mapped to the ground truth within a 30 µm radius). For these two instances, the distance-only baseline completely failed to identify the true candidate (Top-1 Accuracy: 0.00%). Incorporating the composite metric (combining distance, radius difference, and orientation disagreement) did not improve the ranking, also resulting in a 0.00% Top-1 Accuracy.
- **Caveats:** small sample size

### 179. (Surprise 0.252) Belief dropped: In dense regions with competing reconnection proposals, the angular alignment (cosine similarity of branch tangents) between matching split endpoints is signifi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 40 · **Belief:** Leaning True → Uncertain (0.708→0.547) · **Direction:** Negative · **Surprisal:** −0.252
- **Tested:** Determine the efficacy of local tangent agreement as a mechanism for an agent to resolve mutually exclusive topological proposals at dense leaf clusters.
- **Conclusion:** The script extracted 39,826 leaf nodes, mapped 7,050 to ground truth, and isolated 196 leaves belonging to dense clusters (defined as 3 or more leaves within a 25 µm radius). In these dense clusters, the algorithm identified 41 "Matches" (true continuations) and 2 "Mismatches" (false proposals). The mean cosine similarity for Matches was 0.1868, whereas for Mismatches it was -0.7972.
- **Caveats:** small sample size

### 180. (Surprise 0.243) Supported: Rolling back (removing) true merge edges in the U-Net fragments dramatically reduces the maximum connected component size while causing only a marginal decrease...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 9 · **Belief:** Leaning True → Likely True (0.750→0.906) · **Direction:** Positive · **Surprisal:** +0.243
- **Tested:** Quantify the structural trade-off of resolving merge errors by measuring the reduction in massive erroneous super-components versus the loss of intact cable length.
- **Conclusion:** A total of 292 true merge edges were identified and removed. Removing these merge edges reduced the maximum component size from 85,321 nodes to 40,931 nodes, which is a massive 52.03% reduction. In contrast, the average component path length saw only a marginal decrease from 1849.31 µm to 1797.58 µm, a reduction of just 2.80%.
- **Caveats:** none noted

### 181. (Surprise 0.243) Supported: Predicted branching nodes (degree ≥ 3) in the fragment graph are significantly more likely to be the structural sites of merge errors than predicted path nodes...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 14 · **Belief:** Leaning True → Likely True (0.750→0.906) · **Direction:** Positive · **Surprisal:** +0.243
- **Tested:** Identify the topological locations where the U-Net model most frequently fuses distinct neurons to better target merge-detection algorithms.
- **Conclusion:** The script identified 19,460 branching nodes (degree >= 3) and sampled an equal number of path nodes (degree == 2) from the fragments graph. Results & Findings: - Branching Nodes: 36 merge sites out of 19,460 (0.1850%). - Path Nodes: 1 merge site out of 19,460 (0.0051%).
- **Caveats:** extreme class imbalance / very few positives; implementation deviated from plan

### 182. (Surprise 0.243) Supported: Automated segmentation (U-Net) is more likely to fail and create split errors at regions where the biological neurite is thinnest, resulting in fragment endpoin...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 26 · **Belief:** Leaning True → Likely True (0.750→0.906) · **Direction:** Positive · **Surprisal:** +0.243
- **Tested:** Determine if local neurite radius is a predictive factor for U-Net splitting errors by comparing the radii of leaf nodes (split sites) to internal nodes (contiguous regions), ensuring numerical stability during statistical testing.
- **Conclusion:** The experiment successfully executed after correcting the numerical overflow issue by casting the `node_radius` data to `float64`. The pipeline extracted 39,826 valid leaf nodes (split sites) and a random sample of 100,000 internal nodes (contiguous regions). Descriptive statistics confirm the hypothesis: leaf nodes have a significantly smaller mean radius (1.5761 µm) and median radius (1.6729 µm) compared to contiguous internal nodes (mean: 1.9676 µm, median: 1.9766 µm).
- **Caveats:** none noted

### 183. (Surprise 0.243) Supported: Fragment components containing at least one merge error exhibit a significantly higher average node degree and a larger total number of leaf nodes than clean fr...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 47 · **Belief:** Leaning True → Likely True (0.750→0.906) · **Direction:** Positive · **Surprisal:** +0.243
- **Tested:** Assess whether global topological features of a fragment component can reliably flag the presence of internal merge errors.
- **Conclusion:** The experiment was successfully executed, confirming the hypothesis that fragment components with internal merge errors exhibit different global topological features compared to clean fragments. Results & Findings: - Dataset breakdown: Out of the mapped components, 35 were identified as 'Merged' (mapped to 2 or more GT components) and 1111 as 'Clean' (mapped to exactly 1 GT component). - Average Node Degree: - Merged: Mean = 1.9961, Median = 1.9970 - Clean: Mean = 1.9950, Median = 1.9951 - Statistical Test: Mann-Whitney U statistic = 24006.0, p-value = 0.00897.
- **Caveats:** small sample size

### 184. (Surprise 0.243) Supported: U-Net fragment components that contain at least one merge error have a significantly greater total cable length than fragments with no merge errors, indicating...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 147 · **Belief:** Leaning True → Likely True (0.750→0.906) · **Direction:** Positive · **Surprisal:** +0.243
- **Tested:** Determine if fragment cable length is positively associated with the likelihood of encompassing a merge error.
- **Conclusion:** The experiment was successfully executed and the results directly address the hypothesis. The code successfully identified branch nodes acting as merge errors, mapped them to their parent U-Net fragment components, and compared the total cable lengths between fragments containing merge errors and those without. Experiment Results: - Identified Merge Errors: 36 merge error branch nodes were found and mapped to 9 distinct fragment components.
- **Caveats:** small sample size

### 185. (Surprise 0.243) Belief dropped: The local spatial density of fragment endpoints (count of leaves within a 50 µm radius) is positively correlated with the local probability of a false-positive...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 164 · **Belief:** Leaning True → Uncertain (0.750→0.594) · **Direction:** Negative · **Surprisal:** −0.243
- **Tested:** Verify that dense neuropil regions require stricter distance/geometry thresholds because simple proximity becomes an unreliable heuristic.
- **Conclusion:** The dataset was loaded, candidate fragment leaf pairs within 20 µm were identified, and they were mapped to Ground Truth (GT) components to establish true-positive (TP) and FP labels. The pipeline identified 409 TP pairs but only 3 FP pairs. A logistic regression model was fit using local endpoint density (radius of 50 µm) to predict FP probability.
- **Caveats:** small sample size

### 186. (Surprise 0.243) Supported: Ground-truth branch points are primary topological vulnerabilities for the automated U-Net; therefore, fragment endpoints (splits) occur at a significantly high...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 183 · **Belief:** Leaning True → Likely True (0.750→0.906) · **Direction:** Positive · **Surprisal:** +0.243
- **Tested:** Determine if the U-Net is systematically more prone to splitting neurons at branching junctions compared to simple linear segments.
- **Conclusion:** The experiment was successfully executed and properly evaluated the hypothesis. The dataset was successfully loaded, and the ground-truth graph was analyzed to identify 7,320 branch nodes and 1,348,888 linear nodes. By defining a proximity threshold of 5.0 µm to the U-Net fragment leaf nodes (endpoints/splits), it was found that 1.34% (98) of true branch nodes had a split nearby, compared to only 0.78% (10,511) of linear nodes.
- **Caveats:** none noted

### 187. (Surprise 0.235) Supported: The physical gap distance of a valid split error is inversely correlated with the combined cable length of the two fragments being connected, meaning that longe...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 179 · **Belief:** Leaning True → Likely True (0.708→0.859) · **Direction:** Positive · **Surprisal:** +0.235
- **Tested:** Determine if fragment length (a proxy for segmentation confidence) predicts the severity (gap size) of split errors.
- **Conclusion:** The script identified 1,801 valid split pairs (pairs of fragment leaves within 30 µm that map to the same ground-truth component but belong to different fragment components). The mean gap distance for these split errors was 11.42 µm (std = 8.90), and the mean combined fragment length was 20,044.59 µm (std = 57,741.76). A Spearman rank correlation analysis between gap distance and combined fragment length yielded a correlation coefficient of -0.1139 with a highly significant p-value of 1.2692e-06.
- **Caveats:** none noted

### 188. (Surprise 0.235) Supported: Valid split reconnections can be distinguished from invalid ones based on endpoint collinearity, such that pairs of fragment leaves mapping to the same GT neuro...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 199 · **Belief:** Leaning True → Likely True (0.708→0.859) · **Direction:** Positive · **Surprisal:** +0.235
- **Tested:** Verify that geometric collinearity of endpoints is a reliable feature for predicting mutual exclusivity and valid reconnections at crowded fragment endpoints.
- **Conclusion:** The experiment successfully executed, identifying 408 valid and 3 invalid reconnection pairs among fragment endpoints within a 20 µm radius. Endpoint collinearity was computed using reversed cosine similarity of tangent vectors (where 1.0 means endpoints face each other directly). A Mann-Whitney U test compared the two distributions, yielding a statistically significant p-value of ~0.0047 (U=1192.0).
- **Caveats:** none noted

### 189. (Surprise 0.203) No effect found: The difference in estimated neurite radius (`node_radius`) between true connecting endpoints is significantly smaller than the difference between spatially clos...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 1 · **Belief:** Leaning True → Uncertain (0.708→0.578) · **Direction:** Negative · **Surprisal:** −0.203
- **Tested:** To determine if U-Net predicted neurite radius is conserved across split gaps and can serve as a reliable feature for disambiguating connection proposals.
- **Conclusion:** The experiment successfully executed and evaluated the hypothesis. Out of 39,826 leaf nodes in the fragments graph, 2,209 pairs were found within a 30 µm radius. By mapping these pairs to the ground-truth components, they were classified into 464 True Positive (valid) reconnections and 5 False Positive (invalid) reconnections.
- **Caveats:** small sample size

### 190. (Surprise 0.203) Supported: Pairs of fragment endpoints belonging to the same ground-truth neuron (true splits) exhibit significantly higher collinearity (anti-parallel tangent vectors) th...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 8 · **Belief:** Leaning True → Likely True (0.792→0.922) · **Direction:** Positive · **Surprisal:** +0.203
- **Tested:** Evaluate the geometric orientation agreement (tangent collinearity) as a distinguishing signature of recoverable splits.
- **Conclusion:** The experiment successfully tested the hypothesis that pairs of U-Net fragment endpoints belonging to the same ground-truth neuron (true splits) demonstrate significantly higher collinearity (anti-parallel alignment) than nearby endpoints belonging to different neurons (false proposals). The script identified 39,826 fragment leaves and evaluated pairs located within a 20 µm spatial radius. By mapping these back to the ground-truth (GT) graph and applying a 10 µm distance threshold to avoid noise, 398 valid candidate pairs were isolated.
- **Caveats:** small sample size

### 191. (Surprise 0.203) No effect found: True positive split reconnections between leaf nodes exhibit significantly higher directional agreement (higher cosine similarity of their terminal tangent vect...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 47 · **Belief:** Leaning True → Uncertain (0.708→0.578) · **Direction:** Negative · **Surprisal:** −0.203
- **Tested:** To verify if the directional continuity of neurites (tangent alignment) is a robust and discriminative geometric feature for resolving split errors, especially when spatial proximity alone is insufficient.
- **Conclusion:** The experiment successfully executed and evaluated the hypothesis. Out of the leaf nodes in the fragments graph, candidate pairs within a 50 µm radius were extracted and mapped to the ground-truth graph, yielding 639 valid mappings. These were classified into 629 True Positive (TP) reconnections and 10 False Positive (FP) reconnections.
- **Caveats:** none noted

### 192. (Surprise 0.203) No effect found: False positive reconnection proposals (spatially close leaf nodes belonging to different true neurons) are significantly more likely to involve very short, frag...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 13 · **Belief:** Leaning True → Uncertain (0.708→0.578) · **Direction:** Negative · **Surprisal:** −0.203
- **Tested:** Determine if extremely short components act as 'noise bridges' that create false merges, testing if fragment length is a useful proxy for proposal confidence.
- **Conclusion:** The code successfully generated leaf node pairs within 20 µm that belonged to different fragments, resulting in 1397 inter-component leaf node pairs. These leaf nodes were then mapped to the ground truth (GT) structures to classify the reconnections as True Positives (408 proposals) and False Positives (3 proposals). Statistical analysis using the Mann-Whitney U test found no statistically significant difference in the minimum component cable lengths between true positive and false positive proposals (p-value ≈ 0.292).
- **Caveats:** none noted

### 193. (Surprise 0.203) Belief dropped: When an endpoint has multiple mutually exclusive reconnection candidates, a joint metric combining spatial distance and radius difference yields significantly h...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 22 · **Belief:** Leaning True → Uncertain (0.708→0.578) · **Direction:** Negative · **Surprisal:** −0.203
- **Tested:** Evaluate the benefit of joint geometric reasoning over independent thresholding in resolving competing split proposals at dense junctions.
- **Conclusion:** The script successfully identified 'competing scenarios' where a fragment leaf node had multiple candidates within a 20 µm radius, out of which exactly one candidate mapped to the correct Ground Truth (GT) neuron. Only 2 scenarios met these strict criteria. For these complex junctions, both the Baseline (spatial distance alone) and the Joint metric (spatial distance + 10 * radius difference) failed to rank the true topological connection as the top candidate (Top-1 Accuracy: 0.00% for both).
- **Caveats:** small sample size

### 194. (Surprise 0.203) Belief dropped: The bridge segment connecting two incorrectly merged neurons inside a single U-Net fragment exhibits significantly higher structural tortuosity than valid, unme...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 39 · **Belief:** Leaning True → Uncertain (0.708→0.578) · **Direction:** Negative · **Surprisal:** −0.203
- **Tested:** Determine if morphological tortuosity can serve as an intrinsic feature to detect and localize merge errors by comparing the tortuosity of merged bridge segments to a pre-computed pool of length-matched pure segments.
- **Conclusion:** The new approach generated a massive pool of over 530,000 pure control segments, ensuring robust length-matching. Seven merged bridge segments were successfully extracted from the 20 merged fragments and matched with pure control segments of almost identical average cable lengths (~3318 µm). While the descriptive statistics show that merged bridges have a higher mean (1.4965 vs.
- **Caveats:** small sample size

### 195. (Surprise 0.203) Supported: Merge errors are disproportionately concentrated in exceptionally long U-Net fragments, as accumulating a large contiguous predicted component naturally increas...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 43 · **Belief:** Leaning True → Likely True (0.792→0.922) · **Direction:** Positive · **Surprisal:** +0.203
- **Tested:** Verify if the overall physical size of a U-Net fragment is a risk factor for topological merge errors.
- **Conclusion:** The script mapped fragment nodes to ground-truth (GT) components using a 5 µm distance threshold, classified fragments into 'Merged' (mapped to ≥2 GT components) and 'Clean' (mapped to 1 GT component), and calculated the total cable length for each fragment. The results revealed 11 'Merged' fragments and 951 'Clean' fragments. The 'Merged' fragments exhibited substantially larger physical sizes, with a mean cable length of 51,339.69 µm and a median of 10,323.09 µm.
- **Caveats:** none noted

### 196. (Surprise 0.203) Belief dropped: The skeletal paths connecting erroneously merged neurons (merge bridges) exhibit significantly higher variance in neurite radius than contiguous paths of simila...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 44 · **Belief:** Leaning True → Uncertain (0.708→0.578) · **Direction:** Negative · **Surprisal:** −0.203
- **Tested:** Test if radius variance along a skeletal path can serve as a robust intra-fragment feature for detecting merge errors.
- **Conclusion:** The script successfully identified 20 merged fragments and 981 pure fragments, but only 1 merge bridge was successfully extracted with valid radius attributes. A pool of 25,000 pure paths was generated, allowing a precise length match for the single merge bridge (bridge length: 890.39 µm vs pure length: 890.49 µm). For this single pair, the merge bridge exhibited a notably higher radius coefficient of variation (CV = 0.0682) compared to the pure segment (CV = 0.0223), suggesting that merge errors involve artificial fusion points with abruptly changing thicknesses.
- **Caveats:** small sample size

### 197. (Surprise 0.203) Belief dropped: In dense clusters of fragment endpoints (>3 endpoints within a 20 µm radius), resolving reconnections using optimal bipartite matching (linear sum assignment) m...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 52 · **Belief:** Leaning True → Uncertain (0.708→0.578) · **Direction:** Negative · **Surprisal:** −0.203
- **Tested:** Test whether joint, mutually exclusive reasoning over endpoint reconnections outperforms independent classification in dense regions.
- **Conclusion:** It identified 39,826 fragment leaf nodes and built a proximity graph using a 20 µm radius. However, the strict definition of a 'dense cluster' (requiring a core node to have >= 3 connections to other components) resulted in only 1 such cluster being identified across the entire dataset. Within this single cluster, the maximum possible true positive reconnections was 1.
- **Caveats:** small sample size

### 198. (Surprise 0.203) Supported: True reconnections (valid splits recoverable by joining nearby fragment leaves) have a significantly higher orientation agreement (cosine similarity of tangent...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 121 · **Belief:** Leaning True → Likely True (0.792→0.922) · **Direction:** Positive · **Surprisal:** +0.203
- **Tested:** Quantify the predictive power of orientation agreement for identifying valid split reconnections.
- **Conclusion:** The programmer appropriately mocked missing dependencies to load the dataset and correctly identified 1,397 valid fragment leaf pairs within 20 µm from different components. Mapping these to the ground truth yielded 392 True reconnections and 19 False reconnections. The results strongly supported the hypothesis: True reconnections exhibited a mean cosine similarity of -0.7896, heavily skewed towards -1.0 (indicating head-on continuous fibers), while False reconnections averaged 0.2296 and showed a more dispersed distribution.
- **Caveats:** none noted

### 199. (Surprise 0.203) Rejected: Regions with a high spatial density of distinct U-Net fragments are significantly associated with shorter individual fragment lengths, indicating 'tangle' zones...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 141 · **Belief:** Leaning True → Uncertain (0.708→0.578) · **Direction:** Negative · **Surprisal:** −0.203
- **Tested:** Determine the relationship between local fragment crowding and the severity of neuron fragmentation.
- **Conclusion:** The experiment successfully loaded the dataset and calculated the local density and path lengths for 10,172 connected fragment components, exactly following the requested steps. However, the analysis revealed that a 50 µm radius around the component centroids is too small, or the centroid is a poor spatial representative for extended, non-linear structures. Specifically, both the 25th and 75th percentiles for local density were 0.0, indicating that the vast majority of fragments had no other component centroids within 50 µm.
- **Caveats:** none noted

### 200. (Surprise 0.203) Supported: Automated reconstructions of distal axons are significantly more fragmented (shorter average component lengths) than proximal neurites near the soma.
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 148 · **Belief:** Leaning True → Likely True (0.792→0.922) · **Direction:** Positive · **Surprisal:** +0.203
- **Tested:** Assess if segmentation performance degrades systematically as a function of distance from the soma.
- **Conclusion:** The experiment successfully loaded the dataset and calculated the distances of UNet fragments from the ground truth soma locations. Fragments were successfully classified as either proximal (< 300 µm) or distal (> 1000 µm). Proximal fragments (N=39) demonstrated a substantially longer average cable length of 4040.78 µm compared to distal fragments (N=9985) which had an average cable length of 1821.93 µm.
- **Caveats:** none noted

### 201. (Surprise 0.203) Belief dropped: Reconnection proposals between two large U-Net fragments (e.g., >50 nodes) have a significantly higher true-positive rate than proposals involving small fragmen...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 188 · **Belief:** Leaning True → Uncertain (0.708→0.578) · **Direction:** Negative · **Surprisal:** −0.203
- **Tested:** Validate fragment component size as a strong confidence modulator for split-correction reliability.
- **Conclusion:** It found 411 candidate pairs in the 'Large-Large' category (Precision = 99.27%, with 408 True Positives and 3 False Positives). This outcome aligns with the dataset metadata (from previous explorations), which stated that UNet fragments with a cable length shorter than 1000 µm were discarded when the cache was built. Because of this pre-filtering, small fragments (<= 50 nodes) are virtually non-existent in the provided graph.
- **Caveats:** none noted

### 202. (Surprise 0.203) Supported: In the automated reconstruction, predicted branching nodes (degree >= 3) are significantly more likely to be merge error boundaries than continuation nodes (deg...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 193 · **Belief:** Leaning True → Likely True (0.792→0.922) · **Direction:** Positive · **Surprisal:** +0.203
- **Tested:** Verify if merge errors systematically manifest as false branches in the U-Net reconstruction, justifying algorithms that specifically target branch points for merge-detection and topology rollback.
- **Conclusion:** The automated reconstruction's nodes were successfully extracted and categorized by degree into branching (degree >= 3) and continuation (degree == 2) nodes. A KD-tree spatial query mapped 5,609 branching nodes and 986,314 continuation nodes to the human ground truth with a 10 µm distance tolerance. The analysis found 38 merge boundaries at branching nodes and 429 at continuation nodes.
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 203. (Surprise 0.203) Supported: The angular difference between the tangent vectors of two reconnecting fragment endpoints is significantly smaller for true-positive split reconnections than fo...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 196 · **Belief:** Leaning True → Likely True (0.792→0.922) · **Direction:** Positive · **Surprisal:** +0.203
- **Tested:** Determine if tangent orientation agreement between endpoints is a strong geometric feature for classifying valid vs.
- **Conclusion:** The pipeline identified 464 valid pairs (true splits) and 5 invalid pairs (false splits) within the 30 µm search radius. The results strongly support the hypothesis: the median angular difference for valid reconnections was significantly smaller (20.45 degrees) compared to invalid reconnections (128.06 degrees). A Mann-Whitney U test confirmed that this difference is statistically significant (p = 0.0008113, which is < 0.05).
- **Caveats:** implementation deviated from plan

### 204. (Surprise 0.195) Supported: True structural reconnections at split errors preserve branch orientation better than false reconnections, exhibiting a higher cosine similarity of terminal tan...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 27 · **Belief:** Leaning True → Likely True (0.750→0.875) · **Direction:** Positive · **Surprisal:** +0.195
- **Tested:** To test if tangent vector agreement is a discriminative geometric signature for identifying recoverable splits.
- **Conclusion:** The code then extracted 39,826 valid leaf nodes and evaluated tangent vector alignment for 2,209 leaf pairs within a 30 µm search radius. By mapping these pairs to the ground-truth graph, the code identified 464 valid split reconnections (true positives) and 5 merge errors (false positives). The results strongly support the hypothesis: true reconnections exhibit a significantly higher cosine similarity (mean = 0.7103) compared to false reconnections (mean = -0.4796).
- **Caveats:** small sample size

### 205. (Surprise 0.195) Supported: Because merge errors erroneously fuse multiple biological neurons, fragment components containing a merge error have a significantly larger total cable length t...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 155 · **Belief:** Leaning True → Likely True (0.750→0.875) · **Direction:** Positive · **Surprisal:** +0.195
- **Tested:** Verify if total component cable length can serve as a simple, effective heuristic prior for flagging likely merge errors.
- **Conclusion:** The mapping identified 1,042 'Pure' fragments (mapping to exactly one Ground Truth component) and 23 'Merged' fragments (mapping to two or more Ground Truth components). Statistical analysis showed a stark difference in cable lengths. Pure fragments had a mean length of ~4,077 µm (median ~1,830 µm), whereas Merged fragments exhibited a substantially larger mean length of ~28,963 µm (median ~3,006 µm).
- **Caveats:** none noted

### 206. (Surprise 0.195) Supported: For pairs of nearby U-Net fragment leaf nodes (<= 20 µm apart), the geometric alignment (cosine similarity of their incident branch tangents) is significantly h...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 168 · **Belief:** Leaning True → Likely True (0.750→0.875) · **Direction:** Positive · **Surprisal:** +0.195
- **Tested:** Validate that tangent orientation is a statistically strong feature for distinguishing true reconnections from false merges at U-Net fragment boundaries.
- **Conclusion:** The experiment successfully validated the hypothesis that geometric alignment (tangent cosine similarity) is a strong feature for distinguishing true reconnections from false spatial merges at U-Net fragment boundaries. After correctly mocking missing dependencies and loading the dataset, the code extracted U-Net fragment leaves and found pairs within a 20 µm radius. Near the ground truth annotations, it identified 411 valid inter-component pairs.
- **Caveats:** extreme class imbalance / very few positives; small sample size

### 207. (Surprise 0.187) Supported: Local skeleton tortuosity (the ratio of path length to Euclidean distance within a 25 µm neighborhood) is significantly higher at U-Net false merge sites than a...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 2 · **Belief:** Leaning True → Likely True (0.708→0.828) · **Direction:** Positive · **Surprisal:** +0.187
- **Tested:** Test if tortuosity can be used as an intrinsic skeleton feature to identify false merges where neurites cross but do not truly connect.
- **Conclusion:** The experiment successfully executed the data pipeline and statistical analysis. A total of 19,460 branch points were identified in the fragments graph, out of which 5,546 were classified as 'True Branches' and 43 as 'Merge Errors' based on their proximity and mapping to ground-truth (GT) neurons. Statistical analysis using the Mann-Whitney U test revealed that the mean tortuosity for Merge Errors (1.0940) is significantly higher than that for True Branches (1.0765), with a p-value of 2.1030e-04 (p < 0.05).
- **Caveats:** none noted

### 208. (Surprise 0.187) Supported: Fragment branch points (degree >= 3) that represent U-Net merge errors (erroneously fused distinct neurons) exhibit a significantly higher local variance in `no...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 28 · **Belief:** Leaning True → Likely True (0.708→0.828) · **Direction:** Positive · **Surprisal:** +0.187
- **Tested:** Determine if local neurite radius continuity can serve as a geometric signature for detecting merge errors without relying on raw image data.
- **Conclusion:** The analysis identified 19,460 branch points in the U-Net fragments graph. By mapping 1-hop and 2-hop neighborhoods to the ground truth using a KD-tree, 5,572 branch points were classified as True Branches (mapped to a single GT component) and 39 as Merge Errors (mapped to multiple GT components). The statistical analysis supports the hypothesis: branch points associated with Merge Errors exhibited significantly higher local radius variance (Mean Std: 0.1421 vs 0.1095, p=0.0128) and Coefficient of Variation (Mean CV: 0.0747 vs 0.0562, p=0.0152) compared to True Branches.
- **Caveats:** none noted

### 209. (Surprise 0.187) Rejected: In ambiguous proofreading regions where a fragment endpoint has multiple nearby reconnection candidates, the distance to the nearest valid candidate is signific...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 163 · **Belief:** Uncertain → Uncertain (0.542→0.422) · **Direction:** Negative · **Surprisal:** −0.187
- **Tested:** Evaluate the viability of using competitive distance-based reasoning to resolve mutually exclusive reconnection proposals at a single leaf node.
- **Conclusion:** The code successfully identified 39,826 leaf nodes in the U-Net fragments. However, within the specified 30 µm search radius, it found only 2 'Ambiguous Leaves' (cases where a leaf has both a valid and an invalid candidate nearby). For these two cases, the mean distance to the nearest valid candidate was 15.81 µm, while the mean distance to the nearest invalid candidate was 9.98 µm.
- **Caveats:** small sample size

### 210. (Surprise 0.162) Supported: Branch tangent alignment (orientation agreement) degrades as the split gap size increases, making endpoint orientation a highly reliable feature for short-range...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 31 · **Belief:** Likely True → Likely True (0.833→0.938) · **Direction:** Positive · **Surprisal:** +0.162
- **Tested:** Determine the relationship between the physical size of a split gap and the alignment of the structural tangents at the endpoints.
- **Conclusion:** The code dynamically resolved the dataset path, identified 1,479 valid split pairs, and successfully correlated split gap size with endpoint tangent alignment. The results confirm the hypothesis: tangent alignment (orientation agreement) degrades significantly as split gap size increases. A highly significant negative correlation was found (Spearman: -0.4668, p < 1e-80).
- **Caveats:** implementation deviated from plan

### 211. (Surprise 0.162) Supported: In high-density regions where multiple fragment endpoints converge, single-pass greedy distance matching frequently fails because the spatially nearest neighbor...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 175 · **Belief:** Likely True → Likely True (0.833→0.938) · **Direction:** Positive · **Surprisal:** +0.162
- **Tested:** Quantify the failure rate of simple nearest-neighbor reconnection heuristics in dense proposal regions (mutual exclusivity).
- **Conclusion:** Experiment Results & Findings: - Low-Density Regions (1 candidate within 20 µm): Out of 751 true split proposals, 742 were correctly matched to the same ground-truth neuron, yielding a highly reliable nearest-neighbor accuracy of 98.80%. - High-Density Regions (≥ 2 candidates within 20 µm): Out of 28 true split proposals, 25 were matched correctly, resulting in a degraded accuracy of 89.29%. - Statistical Significance: A Chi-squared test comparing the correct and incorrect match proportions between the two regions yielded a statistic of 10.4526 and a p-value of 1.2248e-03.
- **Caveats:** none noted

### 212. (Surprise 0.162) Supported: In naive local reconnection graphs (connecting all fragment leaves within 15 µm), the presence of invalid structural cycles is significantly more frequent in re...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 180 · **Belief:** Likely True → Likely True (0.833→0.938) · **Direction:** Positive · **Surprisal:** +0.162
- **Tested:** Analyze the spatial conditions under which cycle-prevention logic fails or becomes restrictive by correlating cyclic proposals with local fragment density.
- **Conclusion:** By constructing a proposal graph where fragment leaf nodes within 15 µm were connected, the script identified 38,463 total connected components, of which 19 were classified as cyclic (edges >= nodes) and 38,444 were acyclic. The local fragment density (number of other leaf nodes within a 50 µm radius) was calculated for all nodes. Results showed a striking contrast: nodes in cyclic components had an average density of 2.37 nearby leaves per 50 µm, compared to just 0.21 for nodes in acyclic components.
- **Caveats:** none noted

### 213. (Surprise 0.154) Belief dropped: U-Net fragment endpoints (splits) are significantly more likely to occur on neurite segments traveling along the Z-axis than on segments traveling in the X-Y pl...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 128 · **Belief:** Leaning True → Leaning True (0.708→0.609) · **Direction:** Negative · **Surprisal:** −0.154
- **Tested:** Determine if the imaging anisotropy causes a directional bias in split errors, which could inform anisotropic search spaces for reconnection proposals.
- **Conclusion:** The results yielded a mean Z-alignment (abs cos) of 0.4934 for terminal edges versus 0.4738 for internal edges. The K-S test statistic was 0.0541 with a p-value of 3.6617e-51, indicating a statistically significant difference in the distributions. Plot analysis reveals that terminal edges have a strong tendency to align orthogonally to the Z-axis (cosine near 0), suggesting that split errors might be exacerbated by the lower resolution along the Z-axis.
- **Caveats:** none noted

### 214. (Surprise 0.138) Supported: Merge errors exhibit sudden, discontinuous jumps in local neurite radius at the exact site of fusion, making radius delta a reliable geometric signature for age...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 58 · **Belief:** Leaning True → Leaning True (0.708→0.797) · **Direction:** Positive · **Surprisal:** +0.138
- **Tested:** Investigate if local radius discontinuity can serve as an unsupervised feature to detect post-hoc merge errors along a skeleton.
- **Conclusion:** Hypothesis: Merge errors exhibit sudden, discontinuous jumps in local neurite radius at the exact site of fusion, making radius delta a reliable geometric signature for agentic false-merge detection. Experiment Results: The dataset was successfully loaded, and UNet fragment nodes were mapped to Ground Truth (GT) components using a KDTree with a 10 µm distance threshold. The script evaluated 997,617 Pure Edges and 292 Merge Edges.
- **Caveats:** none noted

### 215. (Surprise 0.138) Belief dropped: For pairs of closely located fragment leaf nodes (e.g., within 20 µm), those that belong to the same ground-truth neuron (valid split corrections) exhibit signi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 134 · **Belief:** Leaning True → Leaning True (0.792→0.703) · **Direction:** Negative · **Surprisal:** −0.138
- **Tested:** Evaluate the predictive power of 3D tangent alignment for filtering valid versus invalid split reconnection proposals.
- **Conclusion:** Leaf nodes from the fragment graph were mapped to ground truth, and pairs within 20 µm were classified as 'Valid' (same true neuron) or 'Invalid' (different true neurons). Out of 7,297 mapped leaves, the algorithm found 395 valid pairs and only 3 invalid pairs. The mean cosine similarity (tangent alignment) for valid pairs was 0.8352, compared to 0.6928 for invalid pairs, suggesting that valid continuations tend to be better aligned.
- **Caveats:** small sample size

### 216. (Surprise 0.138) Supported: Fragments containing merge errors are located in regions with a significantly higher local density of branching nodes (degree >= 3) compared to topologically co...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 152 · **Belief:** Leaning True → Leaning True (0.708→0.797) · **Direction:** Positive · **Surprisal:** +0.138
- **Tested:** Determine if local branching density can act as an active environmental flag to proactively trigger merge-detection agents.
- **Conclusion:** The experiment successfully tested the hypothesis by mapping automated UNet fragments to ground-truth graphs and evaluating the local branching node densities around topologically incorrect (merged) and correct (clean) fragments. Out of the mapped fragments, 23 were identified as 'Merged' (mapping to >= 2 ground truth neurons) and 1042 as 'Clean' (mapping to 1 ground truth neuron). A spatial KDTree was built using 19,460 branching nodes (degree >= 3) extracted from the fragments graph.
- **Caveats:** none noted

### 217. (Surprise 0.138) Supported: The U-Net's ability to maintain unbroken tracings is highly dependent on neurite thickness, such that longer U-Net fragment components exhibit a significantly h...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 173 · **Belief:** Leaning True → Leaning True (0.708→0.797) · **Direction:** Positive · **Surprisal:** +0.138
- **Tested:** Correlate automated fragment continuity (cable length) with physical neurite thickness.
- **Conclusion:** The script loaded the dataset, calculated the total cable length and average neurite radius for each of the 10,172 U-Net fragments, and performed the required statistical analyses. The results support the hypothesis: there is a statistically significant positive correlation between fragment cable length and average node radius (Spearman correlation = 0.2522, p-value = 2.5362e-147). The binned summary statistics further illustrate this trend, showing that longer fragment components consistently exhibit higher average radii: - <1000 µm: 1.9358 µm - 1000-2000 µm: 1.9427 µm - 2000-5000 µm: 1.9709 µm - >5000 µm: 1.9921 µm These findings suggest that the U-Net segmentation algorithm is more effective at tracing thicker neurites continuously over long distances, whereas thinner neurites are disproportionately broken up into shorter fragments.
- **Caveats:** small sample size

### 218. (Surprise 0.114) Belief dropped: Branch nodes located in regions of high fragment density (where multiple distinct fragment components intertwine) have a significantly higher rate of being merg...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 73 · **Belief:** Leaning True → Leaning True (0.792→0.719) · **Direction:** Negative · **Surprisal:** −0.114
- **Tested:** Test if local component density is a strong predictive feature for U-Net merge errors (false fusions), which could guide a targeted merge-rollback policy.
- **Conclusion:** The script correctly installed missing dependencies, loaded the dataset via the custom unpickler, mapped UNet branch nodes to the ground truth, and queried the local component density within a 50 µm radius. Out of 5,613 confidently mapped branch nodes, none met the criteria for a 'Dense' region (> 5 distinct fragment components). All 5,613 branch nodes were classified as 'Sparse' (≤ 5 components), containing 36 merge errors for a False Branch Rate of 0.64%.
- **Caveats:** none noted

### 219. (Surprise 0.089) Supported: The physical distance of the gap in true split errors is inversely correlated with the local raw fluorescence image intensity at the gap's midpoint, implying th...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 29 · **Belief:** Leaning True → Leaning True (0.708→0.766) · **Direction:** Positive · **Surprisal:** +0.089
- **Tested:** To validate if missing image evidence correlates directly with the severity (distance) of split errors in the automated reconstruction, using a statistically sufficient random sample to accommodate remote data retrieval limits.
- **Conclusion:** The experiment successfully tested the hypothesis by sampling 200 True Positive split reconnections (from a total of 1,801 identified valid gaps) and querying the 5x5x5 voxel patches around their midpoints from the raw ExaSPIM image volume. The statistical analysis yielded a Spearman rank correlation coefficient of -0.3501 with a highly significant p-value of 3.7429e-07. This moderate, statistically significant negative correlation indicates that as the physical distance of a split error gap increases, the localized fluorescence intensity at the gap's midpoint tends to decrease.
- **Caveats:** none noted

### 220. (Surprise 0.089) Supported: Fragment components containing topological merge errors (mapping to multiple ground-truth neurons) possess a significantly higher branching density (fraction of...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 125 · **Belief:** Leaning True → Leaning True (0.708→0.766) · **Direction:** Positive · **Surprisal:** +0.089
- **Tested:** Investigate if internal branching complexity is a structural indicator of underlying merge errors in the U-Net reconstruction.
- **Conclusion:** The script loaded the dataset using a custom unpickler to bypass missing dependencies, and successfully mapped all fragment nodes to the nearest ground-truth (GT) node using a 10 µm threshold. Fragment components were successfully classified as 'Merged' (mapping to ≥2 distinct GT components) or 'Clean' (mapping to exactly 1 GT component), identifying 23 merged components and 1,042 clean components. The branching density (fraction of nodes with degree ≥ 3) was calculated for each component.
- **Caveats:** implementation deviated from plan

### 221. (Surprise 0.073) Supported: True split reconnections (nearby fragment endpoints belonging to the same ground-truth neuron) exhibit significantly higher geometric tangent alignment (cosine...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 11 · **Belief:** Likely True → Likely True (0.875→0.922) · **Direction:** Positive · **Surprisal:** +0.073
- **Tested:** To determine if local branch orientation agreement is a robust, purely geometric feature for filtering candidate reconnections and preventing false merges.
- **Conclusion:** The experiment executed successfully and completed the evaluation of the proposed hypothesis. The script extracted leaf nodes from the U-Net fragments graph and mapped them to the ground-truth components to differentiate between 'True Splits' (reconnections belonging to the same GT neuron) and 'False Splits' (reconnections belonging to different GT neurons) within a 20 µm search radius. The analysis evaluated 1,391 true split pairs and 6 false split pairs.
- **Caveats:** extreme class imbalance / very few positives

### 222. (Surprise 0.057) No effect found: False connection proposals cluster in dense regions of the proposal graph, meaning that true connections have lower proposal degree centrality than false connec...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 2 · **Belief:** Leaning True → Leaning True (0.708→0.672) · **Direction:** Negative · **Surprisal:** −0.057
- **Tested:** To test if mutual exclusivity (competitor density) can be used as a structural feature, where valid connections have fewer local alternatives than invalid ones.
- **Conclusion:** The experiment successfully executed and tested the hypothesis that valid connections (True Positives) have lower proposal degree centrality (competitor burden) compared to invalid ones (False Positives). The implementation correctly used a `MockUnpickler` to load the dataset, built a proposal graph based on spatial proximity (< 30 µm) between UNet fragment leaf nodes, and mapped these leaves to Ground Truth (GT) components using a 15 µm threshold. The statistical analysis revealed an average competitor burden of 2.36 for the 479 TP edges, and 2.80 for the 5 FP edges.
- **Caveats:** small sample size

### 223. (Surprise 0.041) Supported: Regions with high ground-truth anatomical complexity (dense neural packing) cause exponentially higher segmentation failures, meaning the ratio of U-Net fragmen...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 7 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** To map the spatial distribution of U-Net failure modes by linking ground-truth anatomical density to the rate of over-segmentation.
- **Conclusion:** The experiment successfully calculated the relationship between ground-truth anatomical density and U-Net fragmentation count across 24,360 populated bounding blocks (200x200x200 µm). A Pearson Correlation Coefficient of 0.4001 (p-value < 0.0001) was observed, indicating a moderate, statistically significant positive correlation. This supports the hypothesis that denser neural packing leads to higher segmentation failure rates (more U-Net fragment components).
- **Caveats:** none noted

### 224. (Surprise 0.041) Supported: The spatial distribution of topological errors varies by type, with merge errors occurring closer to the soma (high density) and split errors occurring further...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 38 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** To determine if the distance from the soma is a statistically significant predictor of whether a local error is a split or a merge.
- **Conclusion:** The script correctly extracted approximate soma locations, mapped UNet fragment nodes to ground-truth nodes, and identified 3,965 split errors alongside 475 merge error nodes. The results yielded a K-S statistic of 0.2934 and a highly significant p-value of 8.65e-33, confirming that the distance from the soma is a statistically significant predictor differentiating the spatial distributions of split and merge errors. Furthermore, the generated Empirical Cumulative Distribution Function (ECDF) plot highlighted differing spatial profiles: split errors are distributed broadly up to 8,000 µm, while merge errors display a highly localized distribution with a severe, artifact-like spike at exactly 3,000 µm, likely indicating an algorithmic or tiling boundary in the segmentation process.
- **Caveats:** none noted

### 225. (Surprise 0.041) Supported: Merge errors in U-Net fragments (where multiple distinct neurons are falsely fused) contain local structural anomalies, exhibiting significantly higher radius v...
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 50 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Determine if uncharacteristic fluctuations in the U-Net predicted neurite radius can serve as an intrinsic topological signature to detect merge errors.
- **Conclusion:** The experiment successfully calculated and compared the neurite radius variance between Merged and Clean U-Net fragments. By mapping the fragments to the Ground Truth (GT) skeleton, the script identified 28 Merged fragments (fragments mapping to multiple distinct GT connected components) and 1083 Clean fragments (fragments mapping to exactly one GT connected component). The mean radius variance for Merged fragments was found to be 0.009193, while for Clean fragments it was 0.007919.
- **Caveats:** none noted

### 226. (Surprise 0.041) Belief dropped: Valid split reconnections (pairs of spatial endpoints belonging to the same ground-truth neuron) exhibit a significantly stronger negative dot product (anti-par...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 7 · **Belief:** Leaning True → Leaning True (0.792→0.766) · **Direction:** Negative · **Surprisal:** −0.041
- **Tested:** Evaluate if local 3D tangent alignment is a robust geometric feature for distinguishing between true structural gaps and false connections during split correction.
- **Conclusion:** The experiment successfully executed the data pipeline and statistical analysis. It identified 1,397 inter-component leaf pairs within a 20 µm radius in the fragments graph. After mapping these to the ground-truth (GT) graph (using a 10 µm threshold), 398 pairs were classified: 395 as 'Valid' (belonging to the same GT neuron) and 3 as 'Invalid' (belonging to different GT neurons).
- **Caveats:** small sample size

### 227. (Surprise 0.041) Supported: Ground-truth structural paths (unbranched segments between somas, branch points, or leaves) with higher geometric tortuosity have a higher likelihood of sufferi...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 10 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Evaluate the relationship between neurite tortuosity and U-Net segmentation continuity.
- **Conclusion:** Out of 14,901 extracted ground-truth structural paths, 14,814 were deemed valid for analysis. Only 3.47% of these paths experienced at least one split, with an average tortuosity of 1.1756. Correlation tests showed a very weak but statistically significant positive monotonic relationship (Spearman's r = 0.0447, p < 0.001), while the linear correlation was not significant (Pearson's r = 0.0065, p = 0.426).
- **Caveats:** none noted

### 228. (Surprise 0.041) Supported: Due to the anisotropic resolution of the ExaSPIM imaging (Z-axis is lower resolution), spatial gaps representing true split errors are predominantly aligned wit...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 29 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Assess if imaging anisotropy introduces a predictable directional bias in the orientation of split errors.
- **Conclusion:** The experiment successfully tested the hypothesis that imaging anisotropy introduces a directional bias in the orientation of true split errors. The script efficiently extracted 39,826 leaf nodes from the automated reconstruction, identified 1,812 inter-component pairs within 30 µm of each other, and mapped them to the ground truth to confirm 397 'True Split' pairs. Statistical analysis of the 3D gap vectors for these splits revealed a significant deviation from a theoretically uniform spherical distribution (Mean |cos(θ)| = 0.4644 vs.
- **Caveats:** implementation deviated from plan

### 229. (Surprise 0.041) Supported: The physical path distance from the soma is a primary driver of split errors: fragment endpoints (splits) that map to a given ground truth neuron occur at regio...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 34 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Determine if topological split errors systematically concentrate in thin, distal branches relative to the soma by using a topologically inferred soma root to overcome the lack of explicit soma annotations and uniform radius artifacts.
- **Conclusion:** Findings: 1. The split locations have a higher mean (9,859.71 µm) and median (9,656.38 µm) distance compared to the internal non-split nodes (mean = 9,131.46 µm, median = 8,770.20 µm). The Mann-Whitney U test confirms this difference is highly statistically significant (p-value = 2.5753e-65).
- **Caveats:** small sample size; test driven to p=1.0 by near-zero positive counts

### 230. (Surprise 0.041) Supported: U-Net fragment components that incorrectly merge distinct ground-truth neurons have a significantly higher branching density (branch nodes per micrometer of cab...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 54 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Determine if branching density is a reliable morphological indicator of merge errors in the U-Net reconstruction.
- **Conclusion:** Out of 10,172 valid fragments analyzed, 981 were classified as 'Clean' (mapping to exactly 1 ground-truth component) and 20 were classified as 'Merged' (mapping to 2 or more distinct ground-truth components). The results show that Merged fragments have a higher median branching density (~1.34e-3 branches/µm) compared to Clean fragments (~8.70e-4 branches/µm). A Mann-Whitney U test confirmed that this difference is statistically significant (U = 12654.5, p = 0.0252 < 0.05) with a moderate effect size (Rank-Biserial = 0.29).
- **Caveats:** none noted

### 231. (Surprise 0.041) Belief dropped: Topological branch points (nodes with degree >= 3) in the automated reconstruction that possess an acute branching angle (< 60 degrees) are significantly more l...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 71 · **Belief:** Leaning True → Leaning True (0.792→0.766) · **Direction:** Negative · **Surprisal:** −0.041
- **Tested:** Investigate whether the geometric angle of branches in the U-Net graph can identify merge errors without requiring external image features.
- **Conclusion:** The experiment was successfully executed and tested the proposed hypothesis. Out of the 19,460 branching nodes identified in the `fragments_graph`, local geometries were mapped and evaluated against the ground-truth annotations. For the acute group (containing at least one angle < 60 degrees), there were 14 'Merge' errors and 1,081 'Valid' branches.
- **Caveats:** implementation deviated from plan

### 232. (Surprise 0.041) Supported: The dot product of the unit tangent vectors of two disconnected leaf nodes is highly predictive of valid reconnections, with true reconnections overwhelmingly e...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 74 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Verify the geometric signature of recoverable splits by quantifying the predictive power of 3D tangent alignment for leaf-to-leaf reconnections.
- **Conclusion:** The code loaded the dataset, extracted putative leaf-to-leaf reconnections within 20 µm across different components (1,397 pairs), and mapped them to the ground truth, resulting in 398 confidently mapped pairs. According to the analysis, the Aligned cohort (dot product < -0.7, representing anti-parallel alignment) consisted of 272 pairs, all of which were True Positives, yielding a perfect precision of 100.00% (272 TP, 0 FP). The Unaligned cohort (dot product >= -0.7) contained 126 pairs and exhibited a precision of 97.62% (123 TP, 3 FP).
- **Caveats:** none noted

### 233. (Surprise 0.041) Supported: False merge errors occur disproportionately in regions of high fragment endpoint density (crowded regions), such that the local count of fragment leaves within...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 78 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Test if the spatial clustering of fragmentation (splits) correlates with the occurrence of erroneous fusions (merges), suggesting a shared root cause in complex local topology.
- **Conclusion:** The experiment was successfully executed, confirming the hypothesis that spatial clustering of fragmentation (split errors) correlates with the occurrence of erroneous fusions (merge errors). The script successfully mapped 19,460 branch points and 39,826 leaves from the fragments graph. After mapping these to ground truth, it classified 5,562 True Branches and 41 Merge Errors.
- **Caveats:** small sample size; implementation deviated from plan

### 234. (Surprise 0.041) Supported: The median raw fluorescence intensity along the 3D line segment connecting endpoints of true-positive splits is significantly higher than the median intensity c...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 84 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Validate the use of raw image fluorescence intensity profiles between endpoints as a discriminative feature for split correction.
- **Conclusion:** The algorithm identified 350 valid true-split pairs and 3 invalid false-split pairs within a 20 µm radius. For evaluation, 200 valid pairs and all 3 invalid pairs were sampled. The median intensity for valid pairs was higher (70.00) compared to invalid pairs (36.50).
- **Caveats:** small sample size

### 235. (Surprise 0.041) Supported: In spatially conflicted split proposals (where a single endpoint has multiple reconnection candidates within 30 µm), a strictly greedy distance-based policy wil...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 90 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Quantify the frequency of mutual exclusivity conflicts at endpoints and evaluate the failure rate of a naive shortest-distance reconnection policy in dense local neighborhoods.
- **Conclusion:** The experiment was successfully executed, confirming the hypothesis that a greedy distance-based policy fails in spatially conflicted split proposals. The code correctly mapped fragment leaf nodes to ground-truth neurons, identifying 2,209 leaf pairs within a 30 µm radius. Out of 896 leaves with ground-truth-mapped candidates, only 2 leaves were found to be 'conflicted' (possessing both a valid and an invalid connection candidate).
- **Caveats:** small sample size

### 236. (Surprise 0.041) Supported: Merge errors in U-Net fragments correspond to regions with anomalously high variations in predicted neurite radius, reflecting the unnatural fusion of distinct...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 100 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Determine if variance in predicted neurite radius can be used as an intrinsic heuristic to detect and flag automated merge errors without requiring ground truth.
- **Conclusion:** The script correctly mapped the fragment nodes to ground truth components using the KD-tree, classified the fragments into 11 'Merged' fragments and 981 'Clean' fragments, and evaluated their radius variance and radius gradient (mean absolute difference). The statistical analysis using the Mann-Whitney U test yielded the following results: - Radius Variance: Merged Mean = 0.010240, Clean Mean = 0.007822. The p-value was 0.0612.
- **Caveats:** small sample size

### 237. (Surprise 0.041) Supported: False-merge branch points exhibit significantly higher radial asymmetry among their outgoing branches compared to true anatomical branch points, providing a mor...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 104 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Determine if neurite radius asymmetry at degree-3 nodes can reliably distinguish false merges from true bifurcations.
- **Conclusion:** The experiment successfully tested the hypothesis that false-merge branch points exhibit higher radial asymmetry than true anatomical branch points. The script used a custom unpickler to load the dataset, identified 19,438 degree-3 branch points, and mapped their outgoing branches to the ground truth components. It identified 43 guaranteed False Merges and 5,582 True Branches.
- **Caveats:** none noted

### 238. (Surprise 0.041) Supported: False branching nodes in the automated reconstruction (where a single U-Net fragment erroneously fuses multiple distinct ground-truth neurons) possess significa...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 127 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Assess whether local geometric branching angles can reliably differentiate between biological branches and U-Net merge errors.
- **Conclusion:** The script identified 41 'Merge Branches' and 5,561 'True Branches' by cross-referencing automated U-Net branching nodes with the human ground-truth (GT) components. The analysis revealed that Merge Branches have a smaller (more acute) mean minimum branching angle (70.82° ± 24.65°) compared to True Branches (78.80° ± 21.35°). The Mann-Whitney U test yielded a statistically significant p-value of ~0.0229 (alpha < 0.05), indicating that merge errors do indeed tend to present with more acute geometric angles than true biological branches.
- **Caveats:** implementation deviated from plan

### 239. (Surprise 0.041) Supported: For valid split reconnections (where disconnected fragment leaves map to the same ground-truth neuron), angular agreement (tangent cosine similarity) decays exp...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 130 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** To model the degradation of geometric orientation cues over distance, demonstrating the necessity of iteratively refreshing features and utilizing non-linear/image-aware logic for resolving large gaps.
- **Conclusion:** The dataset unpickling was fixed using a custom `Dummy` class strategy to bypass the missing module error. Key Findings: - Sample Size: 464 valid gap pairs (disconnected U-Net fragment leaves belonging to the same ground-truth neuron and within 30 µm of each other) were identified. - Linear Fit: Yielded an equation of `y = -0.022768x + 0.830732` with an R² of 0.136.
- **Caveats:** none noted

### 240. (Surprise 0.041) No effect found: In valid split reconnections (where two different fragment leaves map to the same ground-truth neuron), the tangent vectors of the endpoints are significantly c...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 137 · **Belief:** Leaning True → Leaning True (0.792→0.766) · **Direction:** Negative · **Surprisal:** −0.041
- **Tested:** Evaluate tangent vector agreement as a feature to reject false split-correction proposals in cycle-prevention logic.
- **Conclusion:** Leaf nodes from separate U-Net fragments within 15 µm were grouped as 'Valid' (mapping to the same ground-truth neuron) or 'Invalid' (mapping to different neurons). The valid group consisted of 1,204 pairs with an average alignment angle of 137.39° (mean dot product -0.6686), indicating a general tendency towards anti-parallel alignment as expected for continuous broken neurites. The invalid group consisted of only 5 pairs, with an average angle of 92.36° (mean dot product -0.0087), suggesting near orthogonal orientation.
- **Caveats:** small sample size

### 241. (Surprise 0.041) Supported: In mutually exclusive reconnection scenarios (where one fragment endpoint has multiple nearby candidate endpoints), the anatomically correct target exhibits sig...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 138 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Evaluate if skeleton geometry (tangent alignment) can reliably resolve competing proposals when a leaf node faces multiple exclusive reconnection options.
- **Conclusion:** The script mapped fragment endpoints to the ground truth and identified 2 mutually exclusive scenarios (where one fragment endpoint has multiple nearby candidate endpoints, with exactly one mapping to the same GT component and at least one mapping to a different GT component). The mean cosine similarity for Correct targets was 0.1459, whereas for Incorrect targets it was 0.6653 (where -1.0 indicates perfect anti-parallel alignment). While the correct targets exhibited better alignment (lower cosine similarity) than the incorrect targets, the very small sample size (n=2) resulted in a Wilcoxon signed-rank test p-value of 0.50.
- **Caveats:** small sample size

### 242. (Surprise 0.041) Supported: The raw fluorescence intensity at the midpoint of a valid gap (a true split between fragments of the same neuron) is significantly higher than at the midpoint o...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 146 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Test if local image evidence (fluorescence intensity) in the gap between endpoints can statistically distinguish anatomically valid reconnections from invalid ones.
- **Conclusion:** The experiment was successfully executed and robustly tested the hypothesis. By querying fragment leaves within a 15 µm radius, the code identified 1,382 candidate gap pairs. Mapping these endpoints to ground truth structures successfully classified 344 as valid (same GT neuron) and 3 as invalid (different GT neurons) proposals.
- **Caveats:** none noted

### 243. (Surprise 0.041) Supported: The geometric path across a false merge boundary exhibits a sharper angular deviation (a larger 'kink') compared to correctly reconstructed continuous segments,...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 159 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Test if local path angularity can serve as a purely geometric flag for merge errors, independent of image features.
- **Conclusion:** Hypothesis: The geometric path across a false merge boundary exhibits a sharper angular deviation (a larger 'kink') compared to correctly reconstructed continuous segments, as falsely merged crossing fibers lack natural structural smoothness. Experiment Results: The code successfully identified 4,484 valid merge edges and sampled 22,420 control edges. For each edge, the local angular deviation from a straight line was computed using a 4-node sequence.
- **Caveats:** implementation deviated from plan

### 244. (Surprise 0.041) Supported: False-positive split reconnection candidates generated by spatial distance thresholds are predominantly located in regions of high local fragment density (crowd...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 195 · **Belief:** Leaning True → Leaning True (0.708→0.734) · **Direction:** Positive · **Surprisal:** +0.041
- **Tested:** Test if local fragment density can serve as a contextual feature to penalize proposals in crowded regions where distance alone is misleading.
- **Conclusion:** It evaluated the hypothesis that False Positive (FP) split reconnection candidates occur in regions of higher local fragment density compared to True Positive (TP) candidates. The code identified 344 True Positive candidates and 3 False Positive candidates. The average local density (measured as unique fragment components per 25 µm sphere) was 2.1047 for TPs and 2.6667 for FPs.
- **Caveats:** extreme class imbalance / very few positives; small sample size

### 245. (Surprise 0.000) Belief largely unchanged: Merge errors are disproportionately localized at anomalous high-degree nodes (degree >= 4) compared to natural bifurcations (degree = 3) in the fragments graph.
- **Run:** `exa-spim-run-1_2026-06-03` · **ID:** 20 · **Belief:** Leaning True → Leaning True (0.750→0.750) · **Direction:** Neutral · **Surprisal:** +0.000
- **Tested:** To test whether high-degree nodes are strong topological signatures of merge errors caused by intersecting neurites.
- **Conclusion:** The experiment successfully executed and tested the hypothesis that merge errors are disproportionately localized at anomalous high-degree nodes (degree >= 4) compared to natural bifurcations (degree = 3). The code accurately classified nodes in the fragments graph by degree and mapped their neighbors to ground-truth (GT) neurons to detect merge sites. The resulting contingency table shows 35 merges and 19,403 valid sites for degree 3 nodes (a merge rate of ~0.18%), and 1 merge and 21 valid sites for degree >= 4 nodes (a merge rate of ~4.55%).
- **Caveats:** none noted

### 246. (Surprise 0.000) Belief largely unchanged: Ground-truth neurite edges with a smaller average radius are significantly more likely to be affected by split errors (U-Net fragment discontinuities) than thic...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 8 · **Belief:** Leaning True → Leaning True (0.750→0.750) · **Direction:** Neutral · **Surprisal:** +0.000
- **Tested:** Determine if neurite thickness is a statistically significant predictor of automated segmentation split errors.
- **Conclusion:** The experiment successfully executed the required steps, extracting the ground-truth edges, mapping them to the U-Net fragments, and classifying them as split or intact. However, the results reveal that all edges in the ground-truth dataset have a uniform radius of exactly 1.0000 µm. Because there is no variance in neurite thickness across the dataset, the hypothesis cannot be tested.
- **Caveats:** test driven to p=1.0 by near-zero positive counts

### 247. (Surprise 0.000) Belief largely unchanged: Ground-truth neurite segments located proximal to the soma (within 200 µm) suffer from a significantly higher rate of split errors (fragment components per µm o...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 23 · **Belief:** Leaning False → Leaning False (0.292→0.292) · **Direction:** Neutral · **Surprisal:** +0.000
- **Tested:** Determine whether proximity to the soma correlates with an increased rate of automated segmentation fragmentation.
- **Conclusion:** However, the output reveals that "No GT neurons with both proximal and distal regions evaluated." This indicates that either the 18 ground-truth neurons lack annotated soma centroids (which the dataset metadata notes may be empty) or none of the neurons possessing an annotated soma contain sufficient nodes to establish measurable cable lengths in both proximal (< 200 µm) and distal (≥ 200 µm) regions. Consequently, the split rates could not be calculated, and the Wilcoxon signed-rank test could not be performed. The hypothesis cannot be evaluated on this dataset due to the absence of required structural annotations.
- **Caveats:** none noted

### 248. (Surprise 0.000) Belief largely unchanged: True split reconnections (fragment leaf pairs mapping to the same ground-truth neuron) exhibit significantly higher directional tangent alignment than false rec...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 64 · **Belief:** Leaning True → Leaning True (0.750→0.750) · **Direction:** Neutral · **Surprisal:** +0.000
- **Tested:** Evaluate if the geometric alignment (tangent vectors) of fragment endpoints is a strong predictor for distinguishing true anatomical splits from erroneous reconnection proposals.
- **Conclusion:** The code identified 1209 candidate leaf pairs from different components within 15 µm. After mapping to the sparse ground truth, 344 pairs were classified as True Matches and 3 pairs as False Matches. The alignment analysis showed a striking difference: True Matches exhibited strong positive alignment (Mean = 0.8061, Median = 0.9391), meaning the tangents are facing each other, whereas the few False Matches showed negative alignment (Mean = -0.7433, Median = -0.6985).
- **Caveats:** extreme class imbalance / very few positives; small sample size

### 249. (Surprise 0.000) Belief largely unchanged: Neurite caliber is inversely correlated with the frequency of split errors; thin neurite fragments (median radius < 0.3 µm) suffer from a significantly higher s...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 93 · **Belief:** Leaning True → Leaning True (0.792→0.792) · **Direction:** Neutral · **Surprisal:** +0.000
- **Tested:** Investigate the relationship between neuron morphology (neurite thickness) and the U-Net's segmentation continuity.
- **Conclusion:** However, the analysis revealed that the 'Thin' cohort (defined by the hypothesis as having a median radius < 0.3 µm) contained 0 components, while all 10,172 qualifying components fell into the 'Thick' cohort (median radius >= 0.3 µm) with a median split density of 2.19 leaves/mm. Because the Thin cohort was empty, the Mann-Whitney U test could not be performed. The results indicate that the predefined threshold of 0.3 µm is too low for this specific dataset, as neurite fragments in this automated reconstruction uniformly have a median radius greater than or equal to 0.3 µm.
- **Caveats:** none noted

### 250. (Surprise 0.000) Belief largely unchanged: Reconnections representing true splits exhibit high orientation agreement, meaning their tangent vectors are significantly more aligned than the tangent vectors...
- **Run:** `exa-spim-run-2_2026-06-03` · **ID:** 140 · **Belief:** Leaning True → Leaning True (0.750→0.750) · **Direction:** Neutral · **Surprisal:** +0.000
- **Tested:** Determine if the cosine similarity of tangent vectors at fragment endpoints is significantly higher for true reconnections compared to false reconnections.
- **Conclusion:** The experiment successfully loaded the dataset by correctly mocking the dependencies and achieved the objective of testing the orientation agreement hypothesis for candidate reconnections. Experiment Results: - Tangent vectors were successfully extracted for 39,826 valid fragment leaf nodes. - A total of 1,618 pairs of leaf nodes belonging to different U-Net fragments were found within a 20 µm search radius.
- **Caveats:** small sample size
