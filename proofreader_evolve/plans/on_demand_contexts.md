# On-demand descriptor contexts (2026-10-29)

Evolution now reads geometry and image crops on demand for any native candidate.
There is no precomputed context prerequisite. The default probe samples the full TRAIN
pool uniformly; `judgeable` (the production default, see below) covers the GT-labeled rows,
`all_candidates` covers the full pool, `pilot` is uniform, and `boundary` explicitly selects
detector-rank neighbours. No-score/raw production accepts `judgeable` or `all_candidates`.
Legacy `all_cached` calls and registries normalize to `all_candidates`; `--context-cache` is
ignored. `--no-descriptors` disables the tools.

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
the rows `rank_metrics`, the bootstrap and every fit use, so the outer measurement is
identical to a full computation while unlabeled rows keep NaN (about 49 k merge / 34 k
split rows over the TRAIN brains, 37 k / 27 k over validation). It is the default scope;
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
for split (34 k rows) plus six graph loads, close to the 1800 s per-call budget; raise
`--descriptor-wall-seconds` to 3600 for image descriptors or accept the margin. Geometry-only
calls take a few minutes. Frozen inference per validation brain stays well inside its budget
(largest cell 751473 merge, 13,374 rows).
