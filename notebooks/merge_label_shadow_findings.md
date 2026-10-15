# Cache merge labels vs canonical metrics: node-count threshold finding

Date: 2026-10-07. All numbers below were produced on n238 (panda env) from the
`dataset_cache_<brain>_mcl100_add.pkl` caches and the canonical `metrics_out/<brain>/*/`
results. No cache, feature table, context cache or detector was modified.

## Summary

- The cache's merge labels (`gt_merge_sites`, `gt_merge_labels`) recover only **46–89%**
  of the canonical merge sites (site recall @ 50 µm), with precision 0.83–1.00.
  The six older-microscope brains are the worst (0.46–0.74, except 754613).
- The main cause is a **unit mismatch**. `canonical_labeling` ports three merge
  thresholds from `segmentation_skeleton_metrics` as *GT node counts* (50 nodes). The
  canonical package reads raw GT SWCs (≈1.0 µm between nodes), so 50 nodes ≈ 50 µm. The
  cache's GT graph is resampled (≈4.0–4.4 µm between nodes), so the same 50 nodes ≈
  200–220 µm. Short genuine merges are therefore rejected as "pass-throughs".
- Expressing the thresholds as the same cable length (50 µm → 11–13 cache nodes) raises
  site recall to **0.80–0.94** on all 11 brains, with precision 0.83–0.96.
- Downstream impact on the frozen merge detector is **negligible**: positives grow by
  8–24%, but almost all new positives rank far down the detector's list. P@500 / P@2000
  change by ≤ 0.005, and the brain ordering is unchanged.
- **Recommendation:** do not rebuild the caches. Keep the current labels for now so that
  old and new brains share one label definition, and record the bias described here.
  Use the corrected labels if merge candidate generation is redesigned.

## 1. A verifier bug that had to be fixed first

`notebooks/verify_add_cache_merge_info.py` reported a hard FAIL ("cache appears to be in
'World reversed' frame") on every brain, including the old brain 789202. The bug was in
the verifier, not in the caches:

- In `merge_sites.csv`, `Voxel` is (x, y, z). `Voxel * anisotropy` is the (x, y, z) µm
  frame used by the cache.
- The package's `World` column equals `(Voxel_z*0.748, Voxel_y*0.748, Voxel_x*1.0)`. The
  axes are reversed but the anisotropy is applied in the original order, so this column
  is not a physical frame.

The verifier now uses `Voxel * anisotropy` as the reference and keeps the `World` column
only as a diagnostic candidate. With this fix all 11 brains PASS, and the same-segment
placement median is 2.4–4.5 µm.

| brain | cache sites | canonical sites | placement median (µm) | site recall @50µm | site precision @50µm |
|---|---|---|---|---|---|
| 747807 | 102 | 160 | 4.5 | 0.531 | 0.833 |
| 750318 | 109 | 162 | 3.5 | 0.593 | 0.899 |
| 751473 | 108 | 164 | 3.8 | 0.616 | 0.935 |
| 754610 | 14 | 26 | 3.1 | 0.462 | 0.857 |
| 754612 | 18 | 23 | 3.1 | 0.739 | 0.944 |
| 754613 | 9 | 9 | 2.4 | 0.889 | 1.000 |
| 789202 | 71 | 99 | 3.3 | 0.687 | 0.958 |
| 794491 | 104 | 148 | 3.5 | 0.669 | 0.942 |
| 794493 | 39 | 51 | 3.2 | 0.725 | 0.949 |
| 794495 (rebuilt cache, canonical of 2026-10-14) | 156 | 198 | 3.4 | 0.722 | 0.917 |
| 802449 | 945 | 1279 | 3.5 | 0.697 | 0.935 |

Coordinates agree; coverage does not. The merge sweep in
`notebooks/merge_candidate_pool_sweep_outputs_with_oldscope/` does **not** measure this gap,
because its denominator is the cache's own `gt_merge_sites`.

## 2. Diagnosis of the missing canonical sites

Each canonical site with no cache site within 50 µm was classified on five brains (the
794495 row is from the 2026-10-07 cache and the June canonical run; not redone):

| brain | canonical | matched | unmatched | segment present, not flagged as merge | flagged, no site here | segment absent (mcl filter) |
|---|---|---|---|---|---|---|
| 789202 | 99 | 68 | 31 | 17 | 11 | 3 |
| 754610 | 26 | 12 | 14 | 13 | 1 | 0 |
| 754612 | 23 | 17 | 6 | 6 | 0 | 0 |
| 750318 | 162 | 96 | 66 | 55 | 7 | 4 |
| 794495 | 168 | 125 | 43 | 22 | 20 | 1 |

What the "present, not flagged" sites look like:

- The merged arm exists in the cache. 98% of these segments have fragment nodes > 50 µm
  from all GT, and the segment passes within ~2 µm of the canonical site.
- The same-label GT component at these sites has a median of only 26–31 cache nodes
  (≈110–135 µm); 70–88% have fewer than 50 nodes. Matched sites have medians of 162–605.
- Measured node spacing: raw GT SWCs ≈ 1.0–1.06 µm (789202, 750318); cache GT graphs
  3.99–4.36 µm (all 11 brains).

The affected thresholds in `agentic_neuron_proofreader/data_modules/canonical_labeling.py`:

| constant | used by | canonical meaning | cache meaning |
|---|---|---|---|
| `MERGE_MIN_NODES = 50` | node-count merge rule (`merge_labels`), `summarize` | > 50 µm per neuron | > ~200 µm per neuron |
| `MERGE_PASSTHRU_MIN_CC = 50` | geometric walk pass-through test | component < 50 µm | component < ~200 µm |
| `min_gt_nodes=MERGE_MIN_NODES` | two-GT junction audit | 50 µm | ~200 µm |

`MERGE_MIN_NODES` is bound as a default argument in `merge_labels` and
`audit_junction_gt_connections`, so changing the module constant alone is not enough.

Smaller contributors, not addressed here:

- The cache walks against the union of all GT neurons; the canonical package walks per
  GT neuron.
- The canonical package snaps sites to a nearby branching node.
- The canonical package dedups using the mislabeled `World` column.
- Fragments shorter than the 100 µm MCL are absent from the cache.

## 3. Shadow relabel (stages 1–2)

`scripts/shadow_merge_relabel.py` recomputes merge labels and sites from each `_add.pkl`.
It reuses the stored GT node labels, makes no cloud reads and never writes the cache.
All three thresholds are set to `round(50 µm / median GT edge length)`. With
`--check-reproduce`, the original thresholds reproduced the stored sites and labels
**exactly** on all 11 brains.

| brain | nodes (scaled) | sites (stored → scaled) | canonical sites | site recall | site precision | canonical merge segments flagged |
|---|---|---|---|---|---|---|
| 747807 | 12 | 102 → 161 | 160 | 0.53 → **0.86** | 0.83 → 0.86 | 0.49 → 0.87 |
| 750318 | 11 | 109 → 171 | 162 | 0.59 → **0.90** | 0.90 → 0.87 | 0.56 → 0.91 |
| 751473 | 12 | 108 → 160 | 164 | 0.62 → **0.87** | 0.94 → 0.89 | 0.64 → 0.91 |
| 754610 | 12 | 14 → 24 | 26 | 0.46 → **0.81** | 0.86 → 0.83 | 0.43 → 0.86 |
| 754612 | 12 | 18 → 23 | 23 | 0.74 → **0.87** | 0.94 → 0.87 | 0.67 → 0.78 |
| 754613 | 12 | 9 → 9 | 9 | 0.89 → 0.89 | 1.00 → 1.00 | 1.00 → 1.00 |
| 789202 | 13 | 71 → 84 | 99 | 0.69 → **0.80** | 0.96 → 0.94 | 0.76 → 0.89 |
| 794491 | 12 | 104 → 132 | 148 | 0.67 → **0.83** | 0.94 → 0.93 | 0.70 → 0.87 |
| 794493 | 12 | 39 → 50 | 51 | 0.73 → **0.94** | 0.95 → 0.96 | 0.69 → 0.93 |
| 794495 (2026-10-14) | 12 | 156 → 177 | 198 | 0.72 → **0.82** | 0.92 → 0.92 | 0.75 → 0.92 |
| 802449 | 12 | 945 → 1143 | 1279 | 0.70 → **0.83** | 0.94 → 0.92 | 0.73 → 0.92 |

Cache-only `% Merged Edges` rises by only 0.03–1.2 points per brain. That metric is
dominated by long merges, so edge-level metrics are essentially unaffected. Split labels
depend only on GT node segment labels and are not affected by these thresholds (by code
inspection; not separately measured).

The older-microscope brains gain the most, which closes most of their gap to the old
brains. The remaining recall gap (0.80–0.94) comes from the smaller contributors listed in
section 2.

## 4. Downstream impact (stage 3)

`scripts/shadow_merge_impact.py` rebuilds the frozen merge detector's candidate universe
and labels with the detector's own `build_sample_universe`. The detector is
`merge-error-794495-mcl100_2026-08-04` with policy `junction|nms=20|r=150` (positive =
geodesic ≤ 20 µm). It runs once with stored sites and once with scaled sites, and scores
the table's frozen `detector_score` under both. Stored-site labels reproduced each table's
`evaluator_labels.npy` exactly. Only the five old brains have tables.

| brain | candidates | positives (stored → scaled) | flips 0→1 / 1→0 | AP | P@500 | P@2000 | new-positive median rank | new positives in top 2000 |
|---|---|---|---|---|---|---|---|---|
| 789202 | 43,737 | 62 → 71 | 9 / 0 | 0.012 → 0.012 | 0.012 → 0.012 | 0.016 → 0.017 | 5,853 | 2 |
| 794491 | 203,956 | 88 → 111 | 23 / 0 | 0.002 → 0.002 | 0.004 → 0.004 | 0.003 → 0.003 | 62,329 | 0 |
| 794493 | 74,051 | 29 → 33 | 4 / 0 | 0.004 → 0.003 | 0.006 → 0.006 | 0.005 → 0.005 | 10,159 | 0 |
| 794495 (2026-10-14) | 58,120 | 128 → 141 | 13 / 0 | 0.114 → 0.105 | 0.080 → 0.082 | 0.040 → 0.041 | 27,960 | 2 |
| 802449 | 299,339 | 825 → 973 | 149 / 1 | 0.045 → 0.044 | 0.096 → 0.098 | 0.070 → 0.075 | 75,051 | 10 |

Fraction of sites the frozen policy reaches (claim ≤ 150 µm / positive ≤ 20 µm):

| brain | stored sites | scaled sites |
|---|---|---|
| 789202 | 0.92 / 0.87 | 0.88 / 0.85 |
| 794491 | 0.91 / 0.84 | 0.92 / 0.83 |
| 794493 | 0.85 / 0.74 | 0.74 / 0.66 |
| 794495 (2026-10-14) | 0.91 / 0.82 | 0.89 / 0.80 |
| 802449 | 0.93 / 0.87 | 0.92 / 0.85 |

Interpretation:

- The recovered merges are mostly short merges away from junctions. The junction-anchored
  policy reaches them less often, and the detector ranks them low.
- Merge Precision@K, AP and the per-brain ordering are therefore essentially unchanged.
- Not measured: whether recent evolution `best_scorer`s change order under the corrected
  labels. Given the size of the P@K changes, this is unlikely.

## 5. Decisions and open items

- **No rebuild.** The cost would be: relabel 11 `_add.pkl`, re-record the image-alignment
  receipts, rebuild feature tables and context caches, and possibly refit the merge
  detector and re-run evolution baselines. This is not justified by the measured impact.
- **Known bias.** Cache merge labels are conservative relative to canonical (site recall
  0.46–0.89). About 10–20% of true merge sites are invisible to the current merge
  pipeline (candidate policy plus labels).
- **New brains.** Build feature tables and context caches from the current caches, so all
  11 brains share one label definition.
- **If the fix is adopted later:** make the thresholds length-based in `canonical_labeling`
  (keeping the old behaviour behind a flag), relabel all brains together, and add a dated
  entry to `proofreader_evolve/WORKFLOW_REVISION.md`.

## Reproduction

```bash
conda activate panda   # compute node, from exa-spim-agent/
python -u notebooks/verify_add_cache_merge_info.py --all --mcl 100
python -u scripts/shadow_merge_relabel.py --all --check-reproduce
python -u scripts/shadow_merge_impact.py --brains 789202 794491 794493 794495 802449
```

Outputs (gitignored):

- `notebooks/merge_label_shadow/<brain>_mcl100.json` and `summary.csv` (stages 1–2)
- `notebooks/merge_label_shadow/impact/` (stage 3)
- logs in `notebooks/log/shadow_merge_relabel_all.log` and `shadow_merge_impact.log`

## 6. Follow-up (2026-10-14): how many merge errors the pipeline sees, and how to reach the rest

Discussion notes to be revised as the measurements below are made. On 2026-10-14 the 794495
rows of sections 1, 3 and 4 and the pooled figures of 6.1 and 6.2 were refreshed against the
rebuilt cache (22 GT neurons) and the regenerated canonical reference (198 sites); section 2
and 6.3 keep the 2026-10-07 figures.

### 6.1 Pooled gap against canonical

| Quantity (11 brains) | Current thresholds | Length-based thresholds |
|---|---|---|
| Canonical merge sites | 2,319 | 2,319 |
| Sites in the cache labels | 1,675 | 2,134 |
| Matched sites (recall) | 1,557 (67%) | 1,953 (84%) |
| Per-brain recall | 0.46 to 0.89 | 0.80 to 0.94 |

About one third of the canonical sites are absent from the current labels; the rescaled
thresholds recover half of that, and about 16% remains (section 6.3).

### 6.2 What the candidate pool sees

For the junction policy `junction|nms=20|r=150` (positive = geodesic <= 20 um), the share
of canonical sites represented by a positive candidate is label recall times policy reach
(section 4 table):

| Brain | Canonical sites | Now: recall x reach | Rescaled: recall x reach | Positive candidates (now -> rescaled) |
|---|---|---|---|---|
| 794495 (2026-10-14) | 198 | 0.72 x 0.82 = 59% | 0.82 x 0.80 = 65% | 128 -> 141 |
| 802449 | 1,279 | 0.70 x 0.87 = 61% | 0.83 x 0.85 = 70% | 825 -> 973 |
| 794491 | 148 | 0.67 x 0.84 = 56% | 0.83 x 0.83 = 69% | 88 -> 111 |
| 789202 | 99 | 0.69 x 0.87 = 60% | 0.80 x 0.85 = 68% | 62 -> 71 |
| 794493 | 51 | 0.73 x 0.74 = 54% | 0.94 x 0.66 = 62% | 29 -> 33 |

**The product is wrong; measured directly on 2026-10-14 the picture changes.** The product
assumes that "missing from the labels" and "not reachable by a candidate" are independent;
they are not. `notebooks/merge_canonical_gap.py` measures both gaps against the canonical
sites themselves (cache gap: Euclidean distance to the nearest cache site; candidate gap:
the detector's own rule, snap the site to its segment within 50 um, geodesic cable distance
to the nearest junction candidate of the proofreader_evolve merge table). Results in
`notebooks/merge_canonical_gap/summary.csv` and `notebooks/merge_canonical_gap/merge_canonical_gap.svg`:

| Brain | Canonical sites | Cache site within 50 um | Candidate within 20 um | Candidate within 150 um | Candidate within 20 um of a cache site (sweep measure) |
|---|---|---|---|---|---|
| 802449 | 1,279 | 70% | 85% | 92% | 87% |
| 794495 | 198 | 72% | 81% | 89% | 82% |
| 751473 | 164 | 62% | 70% | 79% | 80% |
| 750318 | 162 | 59% | 77% | 84% | 70% |
| 747807 | 160 | 53% | 73% | 85% | 79% |
| 794491 | 148 | 67% | 81% | 88% | 84% |
| 789202 | 99 | 69% | 85% | 88% | 87% |
| 794493 | 51 | 73% | 65% | 73% | 74% |
| 754610 | 26 | 46% | 62% | 77% | 64% |
| 754612 | 23 | 74% | 87% | 91% | 89% |
| 754613 | 9 | 89% | 89% | 100% | 100% |
| pooled | 2,319 | 67% | 81% | 89% | 84% |

The junction rule has a candidate within 20 um of 81% of the canonical sites, almost the
same share as for the cache's own sites (84%), so the label gap and the candidate gap
overlap: most canonical merges absent from the labels sit next to a junction candidate and
are simply labeled 0 or NaN today. Relabeling therefore recovers most of the gap by itself;
the candidate rule's own ceiling is 19% of the canonical sites at 20 um (11% at 150 um),
the bridge-type sites of the census. The product approximation (54 to 63%) in the first
version of this section understated the pool by about 20 points and is kept only in the
`product_approximation_at_20` column of the summary for reference. The cross-check column
reproduces the sweep's `site_recall` at `junction|nms=20|r=20` for every brain (802449
0.868 against 0.8677), so the method agrees with the sweep where they measure the same
thing. The 794495 canonical reference was regenerated on 2026-10-14 from the 22-neuron GT
(`notebooks/evaluate_skeleton_metrics.py`; 198 sites, the June run with 168 sites is archived
under `metrics_out/archive/`); its cache precision rose from 74% to 92%, in line with the other
brains, so both sides now use the same GT. Cross-tabulating the 2,319 canonical sites by the
two conditions: cache site within 50 um and candidate within 20 um 58%, candidate only
23%, cache site only 9%, neither 10%.

Among the current label-0 candidates the merges hidden by the thresholds remain rare: 9 of
6,882 on 794495 (0.13%), 149 of 17,739 on 802449 (0.84%), so negatives handed to
AutoDiscovery are little affected. The frozen detector ranks the recovered positives at
median 5,900 to 75,000 (section 4): candidates exist for them, the scoring does not see
them.

### 6.3 The residual ~16%: label-definition differences, not thresholds

Three differences between `canonical_labeling` and `segmentation_skeleton_metrics` remain
after the threshold fix and belong to the relabel step (`label_gt_from_segmentation`), not
to candidate generation:

1. **Walk per GT neuron versus against the union.** The canonical package walks each GT
   neuron separately; the cache walks against the union of all GT neurons, so a segment
   crossing from one neuron to a nearby other neuron can look like a single GT patch and
   is not flagged.
2. **Site snapping.** The canonical package snaps a site to a nearby branching node before
   deduplication; the cache does not. The two sides can place the same event tens of um
   apart and fail the 50 um match although both flag it.
3. **MCL filter.** Fragments under the 100 um minimum cable length are absent from the
   cache; merges on them have no segment to flag (3 / 4 / 1 sites on 789202 / 750318 /
   794495 in section 2). This one is by design and is accepted.

Smaller: the canonical package deduplicates with the mislabeled `World` column, so its own
site list carries a little error. Canonical is a definition, not absolute truth (its
precision against the cache is 0.83 to 0.96); the goal of a fix is agreement and an
explainable bias, not matching one side exactly.

### 6.4 Options for reaching the unreachable sites (to be measured before choosing)

Step 1, diagnosis (read-only, 794495 and 802449): for each labeled-but-unreached site
measure the geodesic distance to the nearest junction, whether another segment's tip lies
within 50 um, local radius and direction change, and the segment length. This decides
which of the sources below is worth building. Part of it exists: the census in
`notebooks/merge_diagnosis/mcl100/` (2026-10-07, stored labels) classifies every cache
site by its nearest junction, `branch` within 30 um geodesic against `bridge` beyond it:
bridge sites are 13% (794495, refreshed 2026-10-14 on 156 sites), 13% (802449), 12% (789202),
16% (794491) and 26% (794493), and the branch fraction saturates between 20 and 30 um (0.85
to 0.87 on 794495), so widening the radius alone has a hard ceiling. The census has not been rerun on
the rescaled labels, which is where the newly recovered short merges would land.

Step 2, candidate sources, cheapest first; all are GT-blind. Given 6.2, this step is
second in priority to the relabel: the junction rule already reaches 81% of the canonical
sites at 20 um and 89% at 150 um, so new candidate families compete for the remaining
11 to 19% (bridge sites), while the labels lose 33%.

1. **Wider positive radius, same enumeration.** Claimable sites (<= 150 um) are 88 to 92%
   against 82 to 87% positive (<= 20 um). Raising the positive radius to 40 or 50 um may
   lift coverage into the high eighties at the cost of coarser localization and noisier
   labels. For the stored labels the whole radius series (5 to 150 um, nms 0 / 10 / 20)
   is already in `notebooks/merge_candidate_pool_sweep_outputs_with_oldscope/`
   `mcl100_dcbde4ab2a6f6c44/tables/all_dataset_configurations.csv` for all 11 brains
   (denominator: the cache's own sites); for rescaled labels it is one parameter in
   `scripts/shadow_merge_impact.py`.
2. **Tip-to-shaft candidates.** A merge usually leaves a split behind: neuron B's fragment
   ends where it was fused into A's segment, so B's tip sits on A's shaft. The split rule
   `tip_to_any_node|r=50|k=2` already enumerates these pairs; their partner nodes on the
   other segment, where the partner is not itself a tip, are natural merge-site candidates.
   Add them as a second candidate family with 20 um NMS. Reuses existing code, bounded
   count, aimed exactly at pass-through merges. Preferred option pending step 1.
3. **Dense sampling along skeletons** every 20 um: coverage near the label ceiling, but
   hundreds of thousands of candidates per brain and a positive rate in the tenths of a
   percent that dilutes both the detector and Precision@K. Last resort, restricted to long
   segments if used.
4. **Mid-shaft radius-bulge scan.** Recorded as an unevaluated second candidate family in
   the sweep's `recommended_policy.json` (`bridge_candidate_family_evaluated: false`): scan
   each skeleton for local radius bulges away from junctions, where two neurites fused
   side by side. Label-free; cost and count unknown until tried.
5. **Segment-level candidates** (one row per segment; the detector builder keeps this unit
   with `--merge-row-unit segment`): coverage 100% of flagged segments but no localization;
   usable as a coarse first stage, not a replacement for site-level candidates. The sweep
   also lists `segment_detector_gating_evaluated: false`.

Step 3, one yardstick for every option: canonical sites covered, rescaled-label sites
covered, candidate count and positive rate, and GT-judgeable rows under `near_gt_150`.
Note that the sweep's `site_recall` is relative to the cache's own `gt_merge_sites`, so
it measures reach, never the label gap of 6.1; the two factors multiply as in 6.2.

### 6.5 Sequencing

The detector will be refitted anyway. With the direct measurement of 6.2 the order is:
relabel first (length-based thresholds plus the per-neuron walk and snapping of 6.3; this
is where two thirds of the gap sits and the candidates for those sites already exist),
then decide on a second candidate family for the bridge sites (the 794495 canonical reference
was already rerun on 2026-10-14), and rebuild once: a changed pool or label set invalidates feature tables, sidecars, companions
and context caches for all brains, and the image-alignment receipts must be re-recorded
after any cache rebuild (see TODO.md, 794495 rebuild of 2026-10-13). Merge positives per
brain roughly double under the combined change, which also gives the merge Precision@K
more statistical weight. Until then the current labels stay, uniformly across the eleven
brains, and the merge feature specification tells AutoDiscovery that merge labels are
conservative.

## 7. The split side (2026-10-14): no label problem, a small candidate gap at omitted stretches

Same question for splits, measured by `notebooks/split_canonical_gap.py` (outputs in
`notebooks/split_canonical_gap/`, figure `split_canonical_gap.svg`). The canonical metrics export
no split sites, so the script labels the raw 1 um GT tracings with the segmentation exactly as
`evaluate_skeleton_metrics.py` does (`DataLoader.load_groundtruth`, misalignment fix included)
and derives split events: GT edges whose ends carry different non-zero segment labels (direct)
and the label pairs flanking a run of unlabeled GT nodes (flank, a split across an omitted
stretch). The same procedure on the cache's 4 um GT graph gives the cache's events. Events are
identified by their unordered segment pair, which is also the identity of a split candidate.

**Motivation.** `verify_add_cache_metrics.py` on the rebuilt 794495 (both sides on the 22-neuron
GT) shows the cache counting 9.8% fewer splits than canonical (11,440 against 12,687, every
neuron lower, r 0.996).

**Where the undercount comes from.** Of the 167,873 canonical split pairs over the eleven
brains, 83% involve a segment that is absent from the fragments graph: a fragment below the
100 um minimum cable length, i.e. a sliver of segmentation the GT crosses. Those pairs can never
be candidates by design (the pool is built from the fragments graph), and the cache still labels
86% of them because labeling reads the segmentation volume. Of the pairs missing from the cache,
95% to 98% are of this kind on every brain; genuine resampling misses (a label on at most four
raw GT nodes) are 13 pairs on 794495 and `other` 67, of which 63 are in the pool anyway. The
canonical `# Splits` is therefore dominated by sub-MCL slivers and measures something different
from what the split pool can address; the cache's count is the one aligned with the pool.

**The pipeline-relevant pairs** (both segments in the fragments graph; 27,949 pooled, 16.6% of
all canonical pairs):

| Brain | Present pairs (of all) | In cache labels | In the split pool | Direct pairs in pool | Flank pairs in pool |
|---|---|---|---|---|---|
| 802449 | 6,773 of 20,598 | 98.3% | 93.3% | 99.7% | 83.8% |
| 751473 | 5,113 of 33,928 | 96.9% | 98.9% | 99.1% | 97.9% |
| 789202 | 2,886 of 10,512 | 99.3% | 93.1% | 99.9% | 85.5% |
| 794495 | 2,657 of 13,563 | 97.0% | 91.0% | 99.4% | 82.7% |
| 794491 | 1,824 of 11,630 | 97.0% | 98.7% | 99.0% | 98.2% |
| 794493 | 1,779 of 13,155 | 96.7% | 98.9% | 99.2% | 98.3% |
| 754610 | 1,717 of 16,796 | 97.7% | 98.0% | 99.6% | 95.2% |
| 747807 | 1,562 of 16,334 | 96.9% | 94.4% | 98.8% | 86.0% |
| 754612 | 1,388 of 11,631 | 97.8% | 86.8% | 99.1% | 69.3% |
| 750318 | 1,180 of 10,190 | 94.3% | 97.7% | 99.4% | 95.8% |
| 754613 | 1,070 of 9,536 | 97.9% | 94.3% | 99.2% | 87.6% |
| pooled | 27,949 of 167,873 | 97.5% | 95.0% | 99.4% | 87.9% |

Cross-tabulated over the present pairs: in cache labels and in the pool 92.7%, candidate exists
but cache label missing 2.3%, cache label but no candidate 4.8%, neither 0.2%. The pool's
coverage of the cache's own pairs reproduces the split sweep (794495: 0.910 against the sweep's
0.907). Cache precision against canonical pairs is 92% to 96%.

**Reading.** Unlike merge, the split labels have no systematic gap: 97.5% of the relevant pairs
are labeled, the misses are scattered (2% to 6% per brain) and small. The candidate gap is 5%
pooled and sits almost entirely on flank pairs, splits across an omitted stretch where the two
fragment ends are farther apart than the 50 um pairing radius (754612 69%, 794495 83%, 802449
84%, against 99% for directly adjacent pairs). Raising `r` for the tip-to-any-node rule, or adding
a tip-to-tip family with a larger radius for ends that face each other across a gap, is the lever;
it costs candidates (the pool grows with r squared) and should be sized on these flank pairs.
Nothing on the split side calls for a relabel.
