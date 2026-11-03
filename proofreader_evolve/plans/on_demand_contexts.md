# On-demand descriptor contexts (2026-10-29)

Evolution now reads geometry and image crops on demand for any native candidate.
There is no precomputed context prerequisite. The default probe samples the full TRAIN
pool uniformly; `judgeable` (the production default, see below) covers the GT-labeled rows,
`all_candidates` covers the full pool, `pilot` is uniform, and `boundary` explicitly selects
detector-rank neighbours. No-score/raw production accepts `judgeable` or `all_candidates`.
Legacy `all_cached` calls and registries normalize to `all_candidates`; `--context-cache` is
ignored. `--no-descriptors` disables the tools.

Since 2026-11-03, `--descriptor-wall-seconds` defaults to 7200 seconds per TRAIN
compute call and per validation-brain descriptor computation (3600 from 2026-10-31,
1800 before), and the per-generation descriptor budget
`--descriptor-generation-wall-seconds` to 14400 seconds (5400 before); probe and
planning time also count toward it.

Probe, batch computation and frozen inference share descriptor_inputs. Legacy image
tier names resolve to physical requests when no offline tier exists. New descriptor
patches are temporary, while numeric descriptor-bank results remain reusable under
source/configuration/code identities. Source failures are explicit. The graph store
serializes cold loads so concurrent input batches share one graph. Validation logs
remain host-only; sampling and input assembly do not use labels.

The preparation script and preflight no longer build or require contexts. The human
error-context plot, fragment/image offset diagnostics and specification checker read live sources too. The old explicit
offline cache builder/reader remain compatibility utilities, outside the evolution
runtime. The derived `proofreader_evolve/context_cache/` data was deleted after
verification (3,707 files, 97.06 GiB); raw volumes, source graph caches, native tables and historical runs stay.

Full candidate pools are much larger than the former detector bands. Use the measured
planning estimates before expensive production calls; cold IO and graph loading count
toward descriptor wall time, and CPU count alone does not bound retrieval throughput.

## Judgeable production scope (2026-10-29, evening)

The full pools hold 2.31 M merge / 4.83 M split rows over the six TRAIN brains and
2.13 M / 5.14 M over the five validation brains, against about 354 k / 326 k in the former
band plus labeled extension. Measured on n283 with 754612: geometry 1.1 ms per row with no
gain from threads (the host assembles contexts under the GIL), level-1 image 0.03 s per row
at 8 threads, 23 s per graph load. A geometry-only `all_candidates` call therefore needs
42 to 89 minutes against the 1800 s per-call budget, image calls hours, and frozen
inference on 751473 split (1.99 M rows) alone exceeds its per-brain budget.

`judgeable` = `label_availability.labeled_rows(table.truth)`: the GT-labeled rows used by
`rank_metrics`, the bootstrap and supervised fitting. Every ranked row is selected for
descriptor computation while unlabeled rows keep NaN (about 49 k merge / 34 k split rows
over the TRAIN brains, 37 k / 27 k over validation). TRAIN, selection and validation use
one scope. Equivalence to `all_candidates` holds for row-wise formulas; for classifiers it
also requires fitting only labeled rows, frozen TRAIN-fitted preprocessing and row-wise
independent `predict` outputs. The arbitrary Python prediction interface does not enforce
these conditions. The section below describes paths that can see the uncomputed rows.
It is the default scope;
no-score/raw accept `judgeable` or `all_candidates`; `pilot`, `boundary` and the probes
are unchanged. Which rows carry a value is label-availability-derived (as the 2026-10-12
labeled extension already was); the values are not. Rows outside the scope do not directly
participate in metric ranking, but can affect the result indirectly through whole-column
statistics or unlabeled-data training. Deployment on an unlabeled brain requires full
descriptor computation; models with cross-row dependencies also need reevaluation under
that coverage.

Measured on n283 (2026-10-30 00:00, 24 workers, `../claude_scratch/time_judgeable_20261029.py`) on the
largest merge cell, 802449 (27,934 judgeable of 553,193 rows): the geometry descriptor took 177 s wall
including the 4.7 GB graph load (worker time 427 s over 110 batches), the level-1 image descriptor
854 s wall (worker time 329 s; about 0.03 s per row of host-side S3 reads, no faster than with
8 threads). A whole-TRAIN level-1 call is therefore about 25 min for merge (49 k rows) and 17 min
for split (34 k rows) plus six graph loads, close to the then-default 1800 s per-call budget.
The default was raised to 3600 seconds on 2026-10-31 for this loading margin. Geometry-only
calls take a few minutes. Frozen inference per validation brain stays well inside its budget
(largest cell 751473 merge, 13,374 rows).

Re-measured on n266 (2026-11-03 01:00, 600 GB job, 24 workers admitted,
`../claude_scratch/time_image_descriptor_20261103.py`) with the 12-column level-1 / 12 um intensity
descriptor the agent wrote in gen 3 of `precision_20261102_140641_3f9z8t3v`, on the same 27,934 rows.
(1) `plan_descriptor_run` had estimated 3,432 s for a call that took 727 s: the 64-row sample charged the
110 s graph load as 1.7 s per row, and its uniform rows scattered the reads. The plan now times the graph
load apart (`graph_load_seconds`, charged once per brain), samples a contiguous block of labeled rows and
credits image descriptors with at most 8 concurrent readers (`IMAGE_IO_PARALLELISM`); with the old
TensorStore settings it estimates 861 s for a 736 s call. The residual pessimism under the new settings
(843 s estimated, 299 s measured) is the worker's fixed start-up cost, which a 64-row sample amortizes over
four times fewer rows than a 256-row batch; it is left on the conservative side. (2) `ImageSource` opened
every array with its own 128 MiB chunk cache and a decode concurrency of 2, which is why 24 threads read no
faster than 8: one shared context (`image_context.tensorstore_context()`: 1 GiB cache, decode concurrency
min(16, CPUs), 64 S3/GCS requests) cut the call from 727-736 s to 299 s (image thread-seconds 14,113 to
4,098, 0.51 to 0.15 s per read, maximum about 1 s in both). A whole-TRAIN level-1 call is therefore about
10 min plus six graph loads, and the largest validation cell about 5 min; level-0 or wide crops need
the 7200 s per-call and 14400 s per-generation budgets, which are the defaults since 2026-11-03.

Patch size (2026-11-03 10:00, `claude_scratch/time_image_reads_20261103.py`, 802449 merge, a contiguous
block of judgeable rows, 8 reader threads through `CandidateImageStore.patch`): the read time is flat in
the patch volume, because the zarr chunks are 128x256x256 voxels (32 MiB raw, zstd) and every allowed
patch touches a few of them. On n266: 0.18 s per read for the gen-3 level-1 12 um patch (7 KB), 0.15 s
for the level-1 30 um tier (0.1 MB), 0.20 s for the level-0 16 um tier (0.11 MB), 0.15 s at level 1 /
60 um (0.75 MB), 0.25 s at level 1 / 100 um (3.4 MB). On n06 with the 256 um cube (radius 128): 0.22 s
at level 2 (0.9 MB), 0.39 s at level 1 (7.2 MB), 1.4 s at level 0 (57 MB, max 2.4 s), all far under the
45 s read timeout. What grows is bytes against the unchanged 128 MiB descriptor batch limit (a level-0
cube runs one row per worker batch, so worker start-ups dominate; the planner measures this) and the
agent's own `describe` cost on the larger array. End to end with the gen-3 descriptor on n266 (24
workers, 27,934 rows; `claude_scratch/time_image_scaling_20261103/`): level 1 / 12 um 299 s (891
worker-seconds), level 1 / 30 um 325 s (684), level 0 / 16 um 416 s (674), level 1 / 60 um 486 s
(2,627; 1.1 MB per patch, about three worker executions per 256-row batch), level 1 / 100 um 1,189 s
(12,012; 3.4 MB per patch, about eleven executions per batch, slowest single read 33 s against the 45 s
timeout). The planner estimated 843 / 609 / 730 / 775 / 1,645 s for these, 1.4-2.8x pessimistic. The former cap of 256 voxels per axis (level 0 at most
a 95 um radius) therefore bought nothing and was replaced by the physical bound
`image_coordinates.MAX_PATCH_EXTENT_UM` = 256 um per patch (radius 4..128 at any level), shared by the
descriptor contract, `LOCAL_IMAGE`, volume analysis and `inspect_candidate` (WORKFLOW_REVISION row
"2026-11-03 / image patches up to a 256 um cube").

The pre-2026-10-29 descriptor-bank entries (scope `all_cached`, 325 directories, 4.1 GB across
`descriptor_bank`, `descriptor_bank_no_score` and `descriptor_bank_raw`) were unreachable under the new
identity and were deleted on 2026-10-30; the per-generation `descriptor_runs/runNNN/result.json`
summaries remain, and any historical descriptor can be recomputed from its generation's `descriptor.py`.

## Scope-dependent paths and the per-scope inference estimate (2026-10-31)

A review of the judgeable entry found two gaps, both corrected on 2026-10-31.

1. The claim that the judgeable scope measures identically to a full computation was too
   strong. `fixed_pool_scoring.evaluate` hands the whole table to `score_candidates`, so a
   formula's whole-column statistics (ranks, quantiles, means, standard deviations of a
   `bank_agent_*` column) are taken over the computed rows only, and
   `classifier_training.fit_classifier` stages up to 100,000 NaN-label rows per brain as
   `X_unlabeled`, which carry NaN in those columns. Under `all_candidates` both paths receive
   computed descriptor values for the otherwise uncomputed rows, so results can differ.
   `predict` can also depend on other input rows even when fitting used only labeled rows.
   Equivalence requires row-wise formulas, or classifiers fitted only on labeled rows with
   frozen TRAIN-fitted preprocessing and row-wise independent predictions; the arbitrary
   Python interface does not enforce those conditions. One historical best scorer
   (`precision_20261011_234710_sssj3jpw`) uses `np.quantile`/`nanstd`, and generations 11/15/16/19/20 of
   `precision_20261025_193120_5to8nx13` declared `X_unlabeled` (their clustering keyed on a
   base `h11_` column, so they did not hit the gap). The measurement remains self-consistent
   because one scope is used on TRAIN, selection and validation; `descriptor_guide.md`,
   `classifier_guide.md` and WORKFLOW_REVISION now state the limitation, and
   `tests/test_descriptor_context.py` pins both paths. A scope that adds the deterministic
   `X_unlabeled` sample to the judgeable rows (about 600 k TRAIN rows: geometry minutes,
   images hours) is a possible follow-up (TODO.md P3), not implemented.
2. `plan_descriptor_run` reported one frozen-inference estimate labelled `judgeable`, while
   `ensure_descriptors` reuses the scope of the registered descriptor at submission. An
   `all_candidates` computation was therefore under-estimated by the pool-to-labeled ratio
   (validation 2.13 M / 5.14 M rows against 37 k / 27 k) and would fail the per-brain budget
   at submission. `DescriptorRuns.inference_rows` now counts every allowed scope on the
   validation tables host-side, the driver passes the dict as `frozen_inference_rows`, and the
   plan returns `frozen_inference[scope]` with the total rows, total and largest-brain seconds
   and `fits_per_brain_budget` against the per-call budget.
