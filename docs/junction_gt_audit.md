# Junction GT Audit

## Separate Truth Levels

`gt_merge_labels` identifies merging **segments** using the node-count rule plus
the canonical geometric walk. Older caches contained only geometric-walk
locations in `gt_merge_sites`. Freshly generated or refreshed caches combine
these with two-GT junction locations in the same `gt_merge_sites` list; there
are no parallel label versions. The detector attaches `is_merge_site=1` near
these recorded sites. Coverage is still incomplete, so zero is not proof that
a junction is correct.

The merge plotter now displays segment and junction GT independently and records
`gt_segment_is_merge` (blank if unavailable) and `junction_gt_status` in its
manifest. Plotting alone does not change CSV labels, winner OOF scores, ranking
or GT-visible selection. After a cache refresh, regenerate detector labels and
train/evaluate again; old CSVs and models still reflect the old site set.

## Generate Or Refresh The Cache

The generator is `scripts/relabel_cache.py`, calling
`BrainDataset.label_gt_from_segmentation()`. Full generation labels GT nodes
from the segmentation and invokes `refresh_merge_labels()` to combine both
site sources. For existing `_add.pkl` files, use the cloud-free refresh:

```bash
srun --partition=aibs_debug --constraint=cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=100G --time=00:25:00 \
  "$HOME/.conda/envs/panda/bin/python" scripts/relabel_cache.py \
  --cache-dir cache --brain 794495 --mcl 100 --refresh-merge-sites
```

This atomically replaces the selected `_add.pkl`, reusing existing GT node
labels and preserving old geometric positives, image paths and extra payload
fields. `--dry-run` shows exactly which files would be replaced. The ordinary
command without `--refresh-merge-sites` still reads raw `.pkl` caches and
writes `_add.pkl`, including both site sources automatically.

`junction_merge_sites()` examines every raw degree>=3 junction on segments
meeting the original two-GT node-count criterion, before candidate NMS and
without reading scores. Supported junctions are consolidated within 30 um of
cable only for the same segment and GT-neuron set. Geometric sites are retained;
exact coincident same-segment records carry both sources. Distinct-source sites
can still describe the same biological error, so site count is not a validated
count of independent merge events.

New records use `source=two_gt_junction`, `node_id`, `gt_neurons` and
`branch_support`. Geometric records use `source=geometric_walk`.
`gt_neuron` remains a representative name for backward compatibility; use
`gt_neurons` for the multiple-neuron evidence. `gt_merge_site_metadata` stores
generation parameters and counts, not a second label set. Candidate labels
cached inside the payload are discarded on refresh.

The sweep now invalidates checkpoints when the input cache identity or sweep
source changes. Before a multi-brain sweep, refresh each intended brain with
the same rule; do not accidentally mix old geometric-only and combined caches.
Rebuild generated detector code from the updated template so the new GT
metadata is also excluded from feature extraction. Re-run label generation,
training and evaluation; do not just overwrite the labels in an old scored CSV
and present its scores as results of the updated training procedure.

## Build And Retrain With Current Labels

The detector-build workflow validates the sweep policy and aggregate hashes,
plus every scope brain's manifest-bound `per_brain/<brain>_meta.json`. The
recorded cache path, size, nanosecond mtime and inode must match the current
cache. Missing legacy identity records, changed caches, and scope mismatches
abort before output cleanup. Rerun the sweep when this gate rejects stale
evidence; do not edit hashes to bypass it. Moving only the bundle is supported
because metadata files are resolved relative to the policy file. The recorded
cache itself must still be available at the recorded path.

Fresh-build into a new directory to preserve older detector outputs. For example,
on a compute node in panda, from the repository root:

```bash
python agentic/run_detector_build_workflow.py \
  autodiscovery/merge-error-794495-mcl100_2026-08-04.json \
  --out-dir autodiscovery-application/merge-error-794495-mcl100_2026-08-04-refreshed \
  --merge-row-unit site \
  --merge-candidate-policy notebooks/merge_candidate_pool_sweep_outputs/mcl100_dcbde4ab2a6f6c44/recommended_policy.json \
  --merge-min-worst-brain-site-recall 0.80 \
  --merge-positive-label-radius-um 20
```

Choose a directory that does not already exist. Do not pass `--yes` to an old
experiment: it authorizes deleting that directory. `--keep-existing` can reuse
validated build stages but leaves historical training outputs untouched. Build
generates and smoke-checks code; it does not retrain the model. Follow the new
instance's `RUN_COMMANDS.md` for smoke, optional profiling, training, held-out
evaluation and result analysis. A new LLM build can change feature implementation
and candidate models, so it is not a controlled label-only comparison.

The runtime writes the same `data_provenance` object into model-selection JSON,
joblib and `<output.csv>.provenance.json`. The sidecar also binds the written
CSV with SHA-256. Training and held-out provenance are separate and include:

- the cache identity captured before loading; a change during loading/extraction
  aborts the run;
- the actual post-scope row count and positive count;
- `ordered_row_labels_sha256`, hashing ordered `(sample_display, binary_label)`
  records (a profiling run instead records its sampled rows);
- the full-universe audit, including merge-site `site_provenance`.

`site_provenance` records geometric-only, two-GT-only, shared-evidence and
untagged-legacy counts, plus a fresh `site_records_sha256` digest over segment,
XYZ, representative/associated neurons and source tags. Site order does not
affect that digest, but duplicate records do. Shared sites count once in the
total; the legacy count is a subset, not an additional source. The optional
`generation_metadata` snapshot preserves cached parameters/counts. A missing
site field is distinguished from an empty list. This is provenance for one
active label collection, not parallel label versions.

File identity is a fast change detector, not a cryptographic hash of the full
pickle. Record full input hashes when archiving an experiment. The site digest
is computed from actual loaded records instead of trusting a pre-existing
metadata digest. A filtered run's row digest covers the retained training rows;
its universe audit still describes the full candidate pool.

The agent prompts and generated documentation preserve both site sources,
distinguish the 20 um positive-label tolerance from the 150 um coverage radius,
and call unlabelled zero targets unverified negatives. `gt_junction_audit`,
`gt_merge_site_metadata` and provenance helpers are unavailable to feature math.
Old timing-bound selections must be regenerated for a changed detector; using
all features does not require a timing run. Winner OOF remains the primary
review score, while selector OOF evaluates the model-selection procedure.

Candidate-policy selection used all brains listed in its evidence scope. A
training-excluded brain in that scope is not an untouched end-to-end test set.
Junction-derived labels can favor junction-geometry features by construction;
report that limitation and validate localization with independent review.

## Generated Feature Validation

Assembly checks statically resolvable accumulator writes against their declared
`set()` signatures, including positional/keyword binding and literal registry
names. A definition expecting `(segment_id, feature_name, value)` cannot be
called as `(feature_name, segment_id, value)`. Both consistent conventions are
supported; explicit keywords reduce ambiguity. Dynamic dispatch and expanded
arguments still require executable extraction tests.

`--synthetic-smoke-test` covers synthetic model matrices and runtime contracts,
not every generated feature branch. A small synthetic-graph test should exercise
the actual extractor, accumulator writes and `to_frame()`, including undefined
values, a large integer identity, row ordering and enabled-group filtering.
It does not establish correctness on every real graph or validate feature
localization semantics. The 20260923-160112 rebuild has instance-bound tests in
`agentic/tests/test_rebuilt_merge_extraction.py`; they skip if that published
instance is absent. Generic assembler regressions are in
`agentic/tests/test_detector_accumulator_contract.py`.

## Opt-In Local Evidence

`canonical_labeling.audit_junction_gt_connections` adds a separate,
GT-dependent review audit. It does not change canonical metrics, segment labels,
site lists, candidate enumeration, model features, or binary training labels
when called on its own. The cache generator now uses its supported connections
to supplement the unified site list.

Default method `junction_gt_branch_support_v1`:

- Follow each arm to the next junction, leaf, or 60 um of cable.
- Exclude the central 10 um; sample the remaining cable at 2 um intervals.
- Match samples to GT nodes within 6 um whose canonical label is the candidate
  segment. That segment must label more than 50 nodes of the matched neuron.
- Reject samples with another GT neuron within 2 um of the best distance,
  including competing neurons assigned to other segment labels.
- Require at least 20 um of consecutive sampled support and 80% supported
  samples on each accepted arm. Lengths are sampled estimates, not exact bounds.
- Return `merge_supported` when separate, non-reconverging arms support at least
  two distinct GT neurons. Otherwise return `unknown`, never a certified negative.

Evidence is spatial and sampling-dependent, not manually verified truth.
A merge with an untraced partner can remain unknown, even if the
old site rule correctly labelled it positive. This supplements rather than
replaces the canonical rule. Nearby downstream junctions and sparse GT can also
limit coverage. Thresholds are explicit heuristic parameters, not calibrated
confidence levels.

## Run An Audit

From the exa-spim-agent root, run Python only on a compute node, using panda:

```bash
srun --partition=aibs_debug --constraint=cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=80G --time=00:10:00 \
  "$HOME/.conda/envs/panda/bin/python" scripts/audit_merge_junction_gt.py \
  --pkl cache/dataset_cache_794495_mcl100_add.pkl \
  --csv autodiscovery-application/merge-error-794495-mcl100_2026-08-04/merge_junction_detector_794495.csv \
  --candidate-id 42020 \
  --output figs/junction_audit_42020.json
```

The command refuses to overwrite an existing output. It loads trusted caches
only, makes no image requests, and does not train. Use `--top-k 20` instead of
`--candidate-id` to audit the highest finite winner OOF scores. This is not the
GT-visible plot subset. An explicit `--score-column` override is retained.
Rank-biased audits must not be reported as unbiased error-rate estimates.

The JSON includes schema/method versions, thresholds, input paths and file
size/mtime metadata, original CSV labels, segment truth, original score rank,
and per-arm evidence. Missing bounded site distances are JSON null. The nearest
recorded same-segment site distance is explicitly Euclidean, not geodesic.

`BrainDataset.audit_junction_gt_connections(node_ids, **parameters)` provides
the same optional audit for in-memory canonical-labelled data. Each call replaces
the prior audit with the requested node set. `save()` stores it independently
as `gt_junction_audit` and on `gt_graph.junction_gt_audit`; save to a **new**
cache path. Recomputing canonical labels clears an existing audit. Old caches
without this field still load. Future generated detectors block the field from
feature extraction; regenerate detector code from the updated template before
using an audit-bearing cache, rather than reusing old generated scripts.

## Initial Check

On brain 794495, candidate 42020 (node 15231401, segment 23809407743) has two
arms matching N007 for 50 um each and two matching N009 for 48 um each, with
100% sampled support outside the central exclusion zone. It returns
`merge_supported`. Before refreshing the cache, its CSV label was
`is_merge_site=0` and the only recorded same-segment site was about 1847 um away.
After the 794495/mcl100 refresh, it has a site at the junction itself and the
existing candidate adapter assigns label 1 at distance 0 um.

A targeted 12-candidate diagnostic initially returned one supported connection (42020)
and 11 unknowns, including five legacy site positives. This is a smoke check,
not validation of specificity or recall. Continue to review representative
samples, including random candidates and existing positives. The full raw
junction refresh audited 1259 junctions on 26 two-GT segments and consolidated
42 supported junctions into 29 additional sites. All 105 geometric sites were
preserved, giving 134 total. With the existing frozen candidate policy, all
58120 candidate identities stayed unchanged and positives increased from 85
to 114; all old positives were retained. These counts describe coverage of
rule-derived labels, not independently verified detector accuracy.

Unsupported audits do not invalidate geometric positives and are not fed as
`unknown` or `-1` into the binary pipeline. Unmatched candidates retain the
existing site-unlabelled zero target. A reliable-negative/unknown training
mask would be a separate change requiring its own validation.