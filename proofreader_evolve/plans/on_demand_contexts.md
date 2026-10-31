# On-demand descriptor contexts (2026-10-29)

Evolution now reads geometry and image crops on demand for any native candidate.
There is no precomputed context prerequisite. The default probe samples the full TRAIN
pool uniformly; `judgeable` (the production default, see below) covers the GT-labeled rows,
`all_candidates` covers the full pool, `pilot` is uniform, and `boundary` explicitly selects
detector-rank neighbours. No-score/raw production accepts `judgeable` or `all_candidates`.
Legacy `all_cached` calls and registries normalize to `all_candidates`; `--context-cache` is
ignored. `--no-descriptors` disables the tools.

Since 2026-10-31, `--descriptor-wall-seconds` defaults to 3600 seconds per TRAIN
compute call and per validation-brain descriptor computation. The per-generation
descriptor budget remains 5400 seconds; probe and planning time also count toward it.

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

`judgeable` = `label_availability.labeled_rows(table.truth)`: the GT-labeled rows, exactly
the rows `rank_metrics`, the bootstrap and every fit use, so every ranked row carries a
value while unlabeled rows keep NaN (about 49 k merge / 34 k split rows over the TRAIN
brains, 37 k / 27 k over validation). The measurement is self-consistent because TRAIN,
selection and validation use one scope; it equals an `all_candidates` computation only for
row-wise formulas and fits on the labeled rows (the section below lists the two paths that
see the uncomputed rows). It is the default scope;
no-score/raw accept `judgeable` or `all_candidates`; `pilot`, `boundary` and the probes
are unchanged. Which rows carry a value is label-availability-derived (as the 2026-10-12
labeled extension already was); the values are not, no column marks the set, and the
rows outside it never enter a measured number. Deployment on an unlabeled brain needs one
offline full computation.

Measured on n283 (2026-10-30 00:00, 24 workers, `../claude_scratch/time_judgeable_20261029.py`) on the
largest merge cell, 802449 (27,934 judgeable of 553,193 rows): the geometry descriptor took 177 s wall
including the 4.7 GB graph load (worker time 427 s over 110 batches), the level-1 image descriptor
854 s wall (worker time 329 s; about 0.03 s per row of host-side S3 reads, no faster than with
8 threads). A whole-TRAIN level-1 call is therefore about 25 min for merge (49 k rows) and 17 min
for split (34 k rows) plus six graph loads, close to the then-default 1800 s per-call budget.
The default was raised to 3600 seconds on 2026-10-31 for this loading margin. Geometry-only
calls take a few minutes. Frozen inference per validation brain stays well inside its budget
(largest cell 751473 merge, 13,374 rows).

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
   `X_unlabeled`, which carry NaN in those columns. Under `all_candidates` both paths see
   every row, so their results differ between the scopes; row-wise formulas and fits on the
   labeled rows do not. One historical best scorer (`precision_20261011_234710_sssj3jpw`)
   uses `np.quantile`/`nanstd`, and generations 11/15/16/19/20 of
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
