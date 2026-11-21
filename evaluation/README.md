# CodeOcean capsule 6960813 on our data: source audit, capsule metrics, edits judged by them

This directory holds three pieces of work around
[CodeOcean capsule 6960813](../external/codeocean/capsule_6960813/README.md) (Arborist
evaluation, imported 2026-11-04). None of them changes a cache, a candidate pool or a file
of the capsule:

1. a data audit of brain **794495**: are the ground-truth tracings and the image that this
   repository uses the same data as the capsule's (2026-11-04);
2. the capsule's own skeleton metrics, computed on our inputs for the unproofread
   segmentation of 794495 (2026-11-18);
3. the first steps toward judging edits with those metrics (841260, 2026-11-18): the
   fragment graph of a labeled cache, written out and evaluated by the capsule; then the
   same graph cut at the merge candidates that ground truth labels as merges, and then
   joined at the split candidates that ground truth puts next to each other: two upper
   bounds that involve no policy. No policy's edits have been evaluated yet; the plan, its
   state and the open problems are in
   [Edits judged by the capsule: status and open problems](#edits-judged-by-the-capsule-status-and-open-problems).

None of them scores a model of this repository, and nothing here feeds an evolution run.

It is a different thing from [`proofreader_evolve/evaluation/`](../proofreader_evolve/evaluation/README.md),
which records how the final policy of a finished evolution run scores on 794495.

| Path | Content | In Git |
|---|---|---|
| `compare_794495_sources.py` | the source comparison script | yes |
| `run_capsule_original.py` | runs the capsule's `main.py` on our inputs, "original" step; `--fragments` evaluates another fragment set | yes |
| `export_cache_fragments.py` | writes fragments of a labeled cache as SWCs the capsule reads | yes |
| `cut_merge_sites.py` | cuts the cache's fragment graph at merge candidates (`--sites oracle`; rules `remove`, `detach`) and writes it for the capsule | yes |
| `join_split_candidates.py` | after those cuts, enumerates the split candidates again on the cut graph and joins the pairs ground truth names (`--joins oracle`; `--truth labels` or `graph`, `--min-support-um`) | yes |
| `compare_capsule_runs.py` | totals, per-neuron table and shared merge sites of two capsule outputs; `--cuts` relates the sites to the cuts made | yes |
| `README.md` | this file, with the results | yes |
| `results/794495_source_comparison_20261104/` | `comparison.json`, `report.md`, `patch_01..03.png` of the audit | no (`results/` is ignored) |
| `results/794495_capsule_original_20261118/` | the capsule's outputs, `run_info.json`, `run.log`, the per-neuron comparison | no |
| `results/841260_baseline_alignment_20261118/` | `raw_swcs/` and `cache_graph/` (two capsule outputs), `cache_fragments.zip`, `comparison.txt` | no |
| `results/841260_merge_oracle_20261118/` | `remove0/`, `remove12/`, `detach/`: the cut graph (`fragments.zip`, `.json`, `fragments_cuts.csv`), the capsule's output (`capsule/`), `comparison.txt` | no |
| `results/841260_split_oracle_20261118/` | ten variants (eight `<cut>_<truth>[_s50]`, `remove12_graph_across`, `none_graph`), each with the joined graph (`fragments.zip`, `.json`, `fragments_joins.csv`, `fragments_connections.txt`), `capsule/`, `comparison_with_cut.txt`, `comparison_with_unedited.txt` | no |

## Where 794495 stands

- It has no role inside an evolution run: not TRAIN, not selection / OOF, not
  development validation or promotion. Launch commands name their brains explicitly.
- Since 2026-11-14 (user decision) the final policy of a finished run is applied to it
  once, as post-analysis. Those figures and the rules of that step are in
  `proofreader_evolve/evaluation/README.md`; they are not fed back to any run.
- It is not an untouched brain. It was used in feature discovery and detector fitting,
  and as a TRAIN brain of evolution runs in 2026-10.
- A formal independent validation protocol has not been designed.

The two tools that read 794495 (the audit and the capsule metrics of the next two
sections) do not touch this: the audit compares data sources, and the capsule metrics
describe the segmentation itself. Both read the ground truth of 794495; neither reads a
label table, a feature table, a score or a model of this repository. The edit experiments
further down run on 841260, a development-validation brain, and do read its candidate
pools and evaluator labels; 794495 comes into them only as the last step of the plan, once.

## Source audit: repository vs capsule

### Why the two sources were compared

The capsule evaluates the Arborist proofreading pipeline on 794495 with its own data
paths. The repository reaches the same brain through different ones:

| | Repository | Capsule |
|---|---|---|
| GT tracings | `gs://allen-nd-goog/ground_truth_tracings/794495/voxel` (cache builder) | `gs://allen-nd-goog/ground_truth_tracings/794495/world/` (`code/main.py`) |
| Image | `s3://aind-open-data/exaSPIM_794495_2026-01-21_14-25-07_processed_2026-07-28_08-24-11/fusion/fused.zarr/` (`configs/image_alignment.json`) | `gs://allen-nd-goog/from_aind/exaSPIM_794495_2026-01-21_14-25-07_training-data/whole-brain-denoised/fused.zarr/0/` (`code/review_merges/build_review.py`) |

Before a figure from one side is set beside a figure from the other, the two have to
describe the same neurons and the same pixels.

### What the script does

`compare_794495_sources.py` makes a small comparison and changes nothing:

- Lists the `voxel/` GT directory and the `world/` GT directory and compares the file
  names.
- Samples three shared neuron files spread across the sorted inventory. Matches SWC
  node IDs, converts voxel xyz with the cache builder's default spacing
  `[0.748, 0.748, 1.0]` um, and compares coordinates, parents and node types.
- Compares the image metadata and one seeded random GT-node-centered patch per sampled
  neuron (48 um per side, level 0, channel and timepoint 0 by default). Pixels are
  compared only when the two grids match; the script fits no registration and
  resamples nothing.
- Reports exact pixel equality, absolute differences, RMSE, Pearson correlation and
  intensity summaries. The three-plane PNGs use common intensity limits for each pair,
  so brightness differences stay visible.

The repository image comes from `configs/image_alignment.json` through the usual
reviewed-source resolver. The capsule's image and GT paths are read from the imported
Python source by AST parsing; the capsule's programs are not executed.

### Run it

In the `panda` environment **on a compute node**, from the repository root:

```bash
/home/zihan.zhang/.conda/envs/panda/bin/python -u \
  evaluation/compare_794495_sources.py --samples 3 \
  --out evaluation/results/794495_source_comparison
```

For an existing allocation, prefix the command with
`srun --jobid=<JOB_ID> --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=2 --nodelist=<NODE>`.
Normal GCS and S3 read access is required. Existing credential configuration is reused;
no credentials are written into the reports.

Options: `--samples` (default 3), `--radius-um` (24), `--level` (0), `--seed` (0),
`--out`. Outputs are `comparison.json`, `report.md` and `patch_*.png` in the output
directory. The report records the source URIs, the sampled files, the GT objects'
generations and hashes, the image metadata hashes, the seed and the host. The image
metadata hashes describe the headers, **not** the image contents.

### What it cannot show

- It reads the current cloud sources. It does not examine an existing `_add.pkl` cache
  or the data assets attached to the capsule on CodeOcean.
- A few matching patches do not establish that two volumes are equal everywhere, and
  the patches are centered on neurites, not drawn from the whole volume.
- Different intensities do not by themselves prove a registration error.
- No dense segmentation is compared.

### Result of 2026-11-04

Run on **n06** with the defaults (three samples, seed 0, level 0, 24 um radius).

**GT: the same tracings.** Both directories held the same 22 SWC file names. The
sampled neurons N001, N011 and N023 have 920,902 nodes in total; node IDs, parent IDs
and node types matched, and the largest coordinate error after conversion was 0 um.

**Image: the same grid, similar structure, different pixel values.** Shapes, spacing,
translations and the two metadata hashes matched, and all three patches lay fully
inside both volumes (202,800 voxels each).

| GT neuron used as image anchor | Pearson | Exactly equal voxels | Mean absolute intensity difference |
|---|---:|---:|---:|
| N001 | 0.993053 | 10.36% | 2.9916 |
| N011 | 0.984458 | 20.01% | 1.5439 |
| N023 | 0.973548 | 14.53% | 1.5511 |

So the two sides trace the same neurons, and their images show the same structure
without being pixel-identical. The comparison does not say which processing step
produces the intensity difference, and three patches do not prove equivalence
everywhere. It gives no reason to replace the image URI that the repository's features
are computed from: image descriptors read pixel values, and these differ.

Synthetic coordinate and pixel checks of the script and `pyflakes` passed on the
compute node.

### Since the audit

- The comparison has not been rerun. `comparison.json` keeps the GT objects'
  generations and hashes of 2026-11-04, so a later run can tell whether a tracing
  changed.
- The repository's image source for 794495 is still the S3 volume above (checked in
  `configs/image_alignment.json` on 2026-11-18; the alignment receipt was re-recorded
  on 2026-11-11 after a label-only refresh of the cache, which does not touch the
  image).
- Recorded on 2026-11-09 ([docs/stage3_evolution.md](../docs/stage3_evolution.md), image
  sources): that S3 `fused.zarr` is itself a denoised fusion (AIND denoise-net on the
  tiles before flat-field correction and fusion). Both images are therefore processed
  products of the same acquisition, which fits what the patches show; which steps
  differ between them is still not established.

## Capsule metrics of the unproofread segmentation

The capsule's main program computes skeleton metrics for every step of an Arborist
proofreading result that it reads from S3: the segmentation as it was, then after each
correction. We have no Arborist result. We do have what the first step needs: the
ground truth, the fragment SWCs and the segmentation. `run_capsule_original.py` runs the
capsule's `code/main.py` unmodified on them, which gives the capsule's "original" step.
The corrected steps, and the capsule's two other tools, need the output of an Arborist
run and are not covered.

It differs from a run of the capsule on an Arborist result in four ways (details in the
script's docstring):

1. No list of the segments the pipeline kept (`segment_ids.txt`), so no segment is
   dropped.
2. No list of reviewed false-positive merge sites (`ignored_merges.swc`), so no merge
   site is ignored.
3. The fragment SWCs of the Google segmentation hold voxel indices; the capsule expects
   um. They are converted before the capsule's loader sees them. Without this every edge
   counts as omitted (seen in the first test).
4. Fragment SWCs are parsed only for the segment IDs that label a ground-truth node. By
   the code this cannot change a result; it was not checked against an unrestricted run.

Run it on a compute node in the capsule's environment `arborist_eval` (never in
`panda`), from the repository root:

```bash
conda activate /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/conda_envs/arborist_eval
python -u evaluation/run_capsule_original.py --brain 794495 \
    --out evaluation/results/794495_capsule_original
```

Run every ground-truth neuron together. A test with N001 alone counted 8 merges for it;
with all 22 neurons loaded the same neuron has 5. Counts of one neuron depend on which
other neurons are present.

### Result of 2026-11-18

n243, 19 minutes (4 of them labeling 7,094,284 ground-truth nodes, about 6 downloading
the 10,000 fragment archives). The ground-truth nodes fall in 12,941 segments, of which
3,980 have a fragment SWC. No archive failed.

Beside it, the repository's reference metrics of the same segmentation
(`metrics_out/794495/`, 2026-10-14), computed in `panda` with the older
`segmentation_skeleton_metrics` (commit `d76557d`) by `notebooks/evaluate_skeleton_metrics.py`:

| | capsule version (`478aede`) | reference (`d76557d`) |
|---|---:|---:|
| Merges, total | 210 | 198 |
| Splits, total | 4,656 | 12,687 |
| Truncations, total | 109 | not reported |
| % merged edges, mean per neuron | 27.65 | 29.48 |
| % omit edges | 2.92 | 1.86 |
| % split edges | 0.11 | 0.34 |
| Edge accuracy | 69.31 | 68.32 |

**Merge sites.** The totals are close, the sites less so. Matched one to one within
20 um (positions taken from the voxel columns; the reference file's `World` column has
its axes permuted), 147 sites are in both, 63 only in the capsule version and 51 only in
the reference. A matched pair is the same place: the median distance from a capsule
site to the nearest reference site is 1.7 um, and widening the radius to 100 um adds
four pairs. Per neuron the two merge counts correlate at 0.94 and are equal for 4 of 22
neurons. The merge search was rewritten between the two commits; this repository's
`length_v2` merge labels follow the older walk.

**Splits.** The capsule version reports about a third as many and has a separate count
of truncations. The two counts are not the same quantity.

**Edges.** For 19 neurons the shares of merged and omitted edges agree within about two
points. Omitted edges are higher throughout (by 1.1 points on average), as expected: the
capsule keeps a ground-truth node's label only when a skeleton node of the node's own
segment lies within 20 um, and two thirds of the segments under the ground truth have no
skeleton. Three neurons differ strongly in merged edges: N004 (3.2% against 30.6%), N019
(4.1% against 17.4%) and N017 (7.5% against 11.7%). The definition of a merged edge is
the same in both versions (a label shared by two neurons with more than 50 nodes each),
so the difference comes from the relabeling step; which nodes lost their label was not
traced. Until it is, do not read the edge-level figures of these three neurons as a
finding about the segmentation.

**Lengths.** Run length and ERL are not compared. The reference loaded voxel coordinates
without scaling, so its lengths are not in um (N001: 354,311 against 294,442).

The per-neuron table is `results/794495_capsule_original_20261118/comparison_with_reference_20261014.txt`.

## Edits judged by the capsule: status and open problems

Aim: apply the merge and split decisions of an evolved policy to a brain's fragment graph and
let the capsule's metrics judge the edited graph. Merges are cut first, and the split stage
works on the graph the cuts left (the user's rule, 2026-11-18). How many cuts and how many
joins to make is to be chosen on development brains and frozen; 794495 is then evaluated
once. Everything is stated for the user's scope of 2026-11-18, the fragments of the cache
(at least 100 um).

**No policy's edits have been evaluated yet.** What exists is the way to evaluate an edited
graph, and two upper bounds on 841260 in which ground truth chooses the edits.

| Step | What | State |
|---|---|---|
| 1 | The cache's fragment graph through the capsule, against the raw SWCs | done: [the cache's fragment graph](#the-caches-fragment-graph-as-the-capsule-sees-it) |
| 2a | Upper bound: cut at every merge candidate labeled as a merge | done: [merge cuts](#merge-cuts-at-the-candidates-that-ground-truth-labels-as-merges) |
| 2b | Upper bound: on the cut graph, join the split candidates that ground truth names | done: [joins](#joins-at-the-split-candidates-that-ground-truth-puts-next-to-each-other) |
| 3 | Merge cuts chosen by a policy's scores, at several depths | not started |
| 4 | Joins chosen by a policy's scores, on the cut graph | not started |
| 5 | 794495 once, with the frozen choices | not started |

### Open problems

All four come from one brain, 841260; the figures are those of the sections below.

1. **The merge labels cover about half of the capsule's merges.** Cutting every candidate
   labeled as a merge takes the capsule's count from 189 to 91. Of the 189 sites, 97 have
   such a candidate within 20 um, 60 have a candidate labeled 0 and 32 have no candidate.
   The labels and the capsule's count come from two versions of the merge search; which
   part of the rules differs was not traced. A policy that follows these labels perfectly
   stops at 91. Details: "The reach".
2. **About 7% of the pairs that the split truth names are joins the capsule counts as
   merges.** In 225 of 3,262 joins one of the two fragments labels less than 50 um of the
   neuron. Joining every named pair takes the merges from 91 to 173; without those 225 they
   end at 83. The split truth has no minimum length, while the capsule's merge count takes
   a fragment as part of a neuron only after 50 um. Whether such a fragment is a crossing
   fiber or a side branch that the ground truth does not trace was not looked at in the
   image.
3. **Joins enlarge the merges that are left.** The capsule counts every ground-truth edge
   of a fragment that has a merge as merged, and joined fragments count as one fragment.
   With the 3,037 joins of the 50 um condition the merge count does not rise (91 to 83),
   yet merged edges rise from 1.20% to 4.72% and edge accuracy returns to the unedited
   level (80.64 against 80.77), while ERL rises fourfold. This follows from problem 1: the
   more merges the cuts miss, the more a join spoils.
4. **What a policy's cuts cost is not measured.** The 131 cuts of the upper bound added 198
   splits. A wrong cut adds splits and removes no merge, and a policy's highest-ranked
   merge candidates are mostly not labeled merges (validation average precision 0.152 for
   the final merge policy of run `precision_20261115_234249_byxx989h`). Step 3 is to
   measure how deep a policy can cut before the cost outweighs the merges removed.

### Not decided

* **How an edit is judged.** ERL and the split count cannot be read alone (problem 3), and
  "the share of merged edges must not rise" fails even the upper bound. Suggested, not
  agreed: merge count, share of merged edges and edge accuracy together, ERL for reference.
* **Whether the labels change.** Nothing was changed. The merge labels (`length_v2`) and the
  split truth (`candidate_rules.split_truth`) are those the evolution runs were trained and
  measured on. Changing either means new labels and pool tables on every brain
  ([docs/dataset_artifact_dependencies.md](../docs/dataset_artifact_dependencies.md)), and
  the figures of earlier runs would no longer compare.

## The cache's fragment graph as the capsule sees it

The candidates of this repository point at nodes of the fragment
graph in a labeled cache, so an edit can only be made on that graph, and the capsule then has
to evaluate that graph before and after. The cache's graph is not the raw SWCs: it keeps
fragments of at least 100 um and resamples them at 5 um. The first question is therefore how
the capsule's picture of the cache's graph compares with its picture of the raw SWCs.

```bash
# 1. the raw SWCs (arborist_eval); also writes segments_under_gt.txt
python -u evaluation/run_capsule_original.py --brain 841260 --out <exp>/raw_swcs
# 2. the same segments from the cache (panda)
python -u evaluation/export_cache_fragments.py --brain 841260 \
    --segments <exp>/raw_swcs/segments_under_gt.txt --out <exp>/cache_fragments.zip
# 3. the exported graph (arborist_eval)
python -u evaluation/run_capsule_original.py --brain 841260 \
    --fragments <exp>/cache_fragments.zip --out <exp>/cache_graph
# 4. side by side (either environment)
python evaluation/compare_capsule_runs.py raw=<exp>/raw_swcs cache=<exp>/cache_graph --out <exp>/comparison.txt
```

### Result on 841260 (2026-11-18)

841260 is a development-validation brain of the evolution runs, chosen because its
segmentation method is the closest to 794495's (both UNet). 13 ground-truth neurons,
3,447,287 nodes, labeled by 28,792 segments. Of these, 24,073 have a raw SWC and 7,124 are in
the cache; the other 21,668 are shorter than the cache's 100 um or have no skeleton.

| | raw SWCs | cache graph |
|---|---:|---:|
| Merges, total | 202 | 189 |
| Splits, total | 43,439 | 7,185 |
| Truncations, total | 190 | 544 |
| % merged edges, unweighted mean over neurons | 3.18 | 3.47 |
| % omit edges | 8.27 | 15.36 |
| % split edges | 1.56 | 0.40 |
| Edge accuracy | 86.99 | 80.77 |

**Merges carry over.** Of the 202 merge sites on the raw SWCs, 187 are found again on the
cache graph within 20 um; 15 are lost and 2 are new. The share of merged edges moves by 0.3
points on average and by at most 1.5 for one neuron. A merge edit on the cache graph can be
judged with these metrics, against the cache-graph column as its starting point.

**Splits do not.** The cache graph shows a sixth of the splits. A ground-truth node whose
segment is too short for the cache has no fragment there, so it counts as omitted (plus 7.1
points) instead of sitting between two splits. A split edit on the cache graph can only
join fragments of at least 100 um; on this brain that leaves most of the fragmentation out
of reach, whatever the policy does. A cache with a lower minimum cable length would be
needed to reach it; none was built or evaluated here. The user's choice (2026-11-18): edits
are limited to the fragments of the cache, what is shorter is left out, and every result is
stated for that scope, against the cache-graph column. The 15% of omitted edges cannot
change under any edit.

The two brains differ in kind. On 794495 the damage is merges (27.7% of edges merged, 0.1%
split, capsule overview figures); on 841260 it is fragmentation (43,439 splits, under 4%
merged). How many merges to cut and how many splits to join cannot be set on one of them
and carried to the other.

## Merge cuts at the candidates that ground truth labels as merges

An upper bound, with no policy in it: cut the cache's fragment graph at every merge candidate
whose evaluator label is 1, and let the capsule evaluate the result. It answers how many of
the capsule's merges the candidates can remove at best, what a cut costs in splits, and which
cut rule to use. A candidate says where a merge is, not which arm belongs to the other
neuron, so `cut_merge_sites.py` has two rules:

* `remove`: delete the nodes within `--radius-um` of path length from the site, extended
  around every other branching node inside that neighbourhood. With 12 um this is the
  Arborist pipeline's own rule (`FragmentsGraph.remove_merge_sites` of neuron-proofreader);
  with 0 only the site's node goes. Every arm becomes a fragment of its own.
* `detach`: remove one edge, the one to the arm that continues no other arm (directions over
  30 um of path). The straightest pair of arms stays connected through the site.

```bash
# the cut graph (panda), once per rule
python -u evaluation/cut_merge_sites.py --brain 841260 --sites oracle --rule remove --radius-um 12 \
    --segments <base>/raw_swcs/segments_under_gt.txt --out <exp>/remove12/fragments.zip
# the capsule on it (arborist_eval)
python -u evaluation/run_capsule_original.py --brain 841260 \
    --fragments <exp>/remove12/fragments.zip --out <exp>/remove12/capsule
# against the unedited cache graph (either environment)
python evaluation/compare_capsule_runs.py cache=<base>/cache_graph remove12=<exp>/remove12/capsule \
    --cuts <exp>/remove12/fragments_cuts.csv --out <exp>/remove12/comparison.txt
```

The pieces of a cut fragment are written as SWCs of their own, named `<segment id>.<k>` with
numbers the segment has not used; the capsule assigns a ground-truth node to the nearest
piece of the node's segment. The script refuses a pool that was not built from the present
cache file, and a candidate whose node does not have the row's position, degree and segment.

### Result on 841260 (2026-11-18)

The merge pool of 841260 has 174,038 candidates, 131 of them labeled as merges (123
junctions, 8 shaft nodes next to another segment's tip), all on segments under ground truth.
All 131 were cut.

| | unedited | `remove`, 0 um | `remove`, 12 um | `detach` |
|---|---:|---:|---:|---:|
| Merges, total | 189 | 112 | 91 | 126 |
| Splits, total | 7,185 | 7,328 | 7,383 | 7,241 |
| % merged edges, unweighted mean over neurons | 3.47 | 1.46 | 1.20 | 1.80 |
| % omit edges | 15.36 | 15.36 | 15.36 | 15.36 |
| % split edges | 0.40 | 0.40 | 0.40 | 0.40 |
| Edge accuracy | 80.77 | 82.78 | 83.04 | 82.44 |
| ERL, um, mean over neurons | 1,087 | 1,159 | 1,161 | 1,172 |
| Merge sites gone / new (20 um) | | 77 / 0 | 98 / 0 | 65 / 2 |
| Splits added per merge removed | | 1.9 | 2.0 | 0.9 |
| Of the 97 sites with a cut within 20 um, gone | | 75 | 96 | 64 |

**The cut rule.** Where a cut is made, the Arborist rule removes the merge almost always (96
of 97); deleting only the site's node leaves 22 of them, and detaching one arm leaves 33:
the arm that continues no other arm is the foreign one in about two cases of three. The
price of `remove` is two new splits per merge removed, against one for `detach`; the omitted
share does not move although 1,209 nodes are deleted. After the joins of the next section
the Arborist rule stays ahead on everything but the split count.

**The reach.** The cuts remove about half of the capsule's merges, and the cut rule is not
what limits it. The 189 merge sites of the unedited graph, by what lies within 20 um:

| | sites | still there after `remove`, 12 um |
|---|---:|---:|
| a candidate labeled as a merge | 97 | 1 |
| a candidate of the same segment, labeled 0 (all judgeable) | 60 | 60 |
| no candidate of the segment | 32 | 30 |

* The 60 are places where a candidate exists and the cache's labels do not call it a merge:
  36 at a junction of the fragment, 24 at a shaft node. One of them has a merge site of the
  cache (`gt_merge_sites`) within 20 um. They are not at ground-truth tips (6 within 20 um
  of one).
* The 32 are all at a node of degree 2 of the fragment, where the candidate rule has nothing
  (16 on segments without any merge candidate).
* In the other direction, 34 of the 131 cuts have no merge site of the capsule within 20 um
  (30 within 100 um), and of the cache's 152 merge sites 110 have one.

So the cache's merge labels (`length_v2`, a port of the merge walk of the metric version in
`panda`) and the capsule's merge count (segmentation_skeleton_metrics 478aede: fragment nodes
more than 10 um from the ground truth that the fragment labels, in a component anchored by a
tip more than 100 um away) name about 110 common sites; 79 of the capsule's 189 are not in
the labels and 42 of the labels' 152 are not the capsule's. Which part of the two rules
differs was not traced. A policy trained and measured on these labels is aimed at the first
group only; relabeling by the capsule's rule would put a candidate within 20 um of 157 of the
189 sites.

Limits: one brain; ground truth chose the sites, so nothing here measures a policy.

## Joins at the split candidates that ground truth puts next to each other

The second upper bound, again with no policy in it. The split stage works on the graph that
the merge stage left (the user's rule, 2026-11-18): `join_split_candidates.py` applies the
merge cuts of the section above, enumerates the split candidates again on the cut graph with
the declared rule (`tip_to_any_node|r=80|k=2`, through `candidate_rules.pair_occurrences`),
and joins the candidates that ground truth names, closest first, never closing a loop. A cut
fragment is several pieces of one segment, so a candidate is a pair of pieces; the rule runs
on segment IDs as for the pools, so no join crosses a cut unless `--across-cuts` is given.
Two readings of "ground truth names the pair" (`--truth`):

* `labels`: the repository's split truth (`candidate_rules.split_truth`) carried to pieces: the
  two sides of an edge that the cache classes as a split, or two flanks of an unlabeled
  stretch. A segment that the cache has no fragment of is a label nothing can be joined to.
* `graph`: the same rule on the fragments that exist. A ground-truth node belongs to the
  nearest piece of its segment within 20 um and to nothing when the cache has no fragment of
  the segment, so two fragments on either side of a missing short one are a pair.

```bash
python -u evaluation/join_split_candidates.py --brain 841260 --merge-rule remove --radius-um 12 \
    --joins oracle --truth labels --min-support-um 50 \
    --segments <base>/raw_swcs/segments_under_gt.txt --out <exp>/<name>/fragments.zip      # panda
python -u evaluation/run_capsule_original.py --brain 841260 --fragments <exp>/<name>/fragments.zip \
    --connections <exp>/<name>/fragments_connections.txt --out <exp>/<name>/capsule       # arborist_eval
```

`--connections` hands the capsule the joined segment pairs in the form of a pipeline step's
`connections.txt`; its own label handler then gives a ground-truth node of either segment to
the joined fragment.

### Result on 841260 (2026-11-18)

After the Arborist cut (`remove`, 12 um). "50 um" is `--min-support-um 50`, explained below.

| | unedited | cut | `labels` | `labels`, 50 um | `graph` | `graph`, 50 um |
|---|---:|---:|---:|---:|---:|---:|
| Joins made | | | 3,262 | 3,037 | 5,580 | 5,219 |
| Merges, total | 189 | 91 | 173 | 83 | 183 | 67 |
| Merge sites not there before the joins (20 um) | | | 98 | 7 | 131 | 14 |
| Splits, total | 7,185 | 7,383 | 3,492 | 3,933 | 1,526 | 2,086 |
| % merged edges, unweighted mean over neurons | 3.47 | 1.20 | 11.37 | 4.72 | 36.74 | 21.42 |
| % omit edges | 15.36 | 15.36 | 14.34 | 14.40 | 11.11 | 11.34 |
| Edge accuracy | 80.77 | 83.04 | 74.07 | 80.64 | 52.06 | 67.12 |
| ERL, um, mean over neurons | 1,087 | 1,161 | 4,231 | 4,754 | 9,182 | 14,382 |

After the `detach` cut:

| | cut | `labels` | `labels`, 50 um | `graph` | `graph`, 50 um |
|---|---:|---:|---:|---:|---:|
| Joins made | | 3,262 | 3,045 | 5,579 | 5,230 |
| Merges, total | 126 | 212 | 117 | 221 | 98 |
| Splits, total | 7,241 | 3,355 | 3,787 | 1,387 | 1,934 |
| % merged edges | 1.80 | 13.87 | 7.86 | 40.85 | 28.80 |
| Edge accuracy | 82.44 | 71.57 | 77.51 | 47.96 | 59.76 |
| ERL, um | 1,172 | 3,807 | 4,294 | 7,784 | 12,050 |

Two more runs, both `graph` without the 50 um condition: joins on the uncut graph give 260
merges, 1,317 splits, 46.42% merged edges; joins after the Arborist cut with `--across-cuts`
give 202 merges (183 without it), 1,381 splits, 42.16% merged edges, 129 of the 5,705 joins
being between two pieces of one cut fragment.

**A pair that ground truth names is not yet a safe join.** Joining every named pair halves
the splits and nearly doubles the merges (91 to 173 under `labels`). Of the 98 merge sites
that were not there before the joins, 94 have a join within 50 um. In those joins the weaker
piece labels a median of 9 um of the neuron's ground truth and has a median of 438 um of
cable elsewhere; in the other joins the figures are 175 um and 28 um. They are fragments that
touch the neuron at a few ground-truth nodes and run on, and they are not at ground-truth
tips (3 of the 98 sites within 20 um of one). The repository's split truth takes any pair of
segments that meet along a neuron; the capsule's merge count takes a fragment as on the
neuron only after 50 um (`MergeCountMetric.min_intersection_length`). With that condition on
both pieces (`--min-support-um 50`) 225 of the 3,262 joins fall away, 7 new merge sites
remain, and the merge count no longer rises (91 to 83).

**What the candidates reach.** With the condition, the splits of the cut graph fall from
7,383 to 3,933 under the cache's labels (47%) and to 2,086 when two fragments around a missing
short one also count (72%). Without the condition, 3,319 of the 3,666 `labels` pairs between
existing pieces have a candidate, and 5,778 of the 8,835 `graph` pairs.

**Joins enlarge the merges that are left.** The share of merged edges rises although the
merge count does not: 1.20% to 4.72% (`labels`, 50 um) and to 21.42% (`graph`, 50 um). The
metric counts every ground-truth edge of a fragment that has a merge, so each of the 83 or 67
merges that the cuts did not reach now spoils a longer fragment. Edge accuracy falls back to
the unedited level (80.64) or below it (67.12) while ERL, which gives a merged fragment a run
length of zero, still rises four- to twelvefold. A split stage therefore pays for every
merge the merge stage left, in proportion to how much it joins; ERL and the split count alone
do not show this.

**Cut rule and cuts.** After the joins the Arborist cut stays ahead of `detach` on merges,
merged edges, edge accuracy and ERL, and behind by about 150 splits. Letting joins cross a
cut brings merges back (202 against 183), which supports keeping them apart.

Limits: one brain; ground truth chose the cuts and the joins, so nothing here measures a
policy; the fragments that label less than 50 um of a neuron were not looked at in the image,
so whether they are crossing fibers or side branches missing from the ground truth is open;
the rise of merged edges was not broken down by fragment; why some merge sites disappear
after joins (15 under `labels`, 50 um) was not traced; the ten outputs take 4.7 GB.
