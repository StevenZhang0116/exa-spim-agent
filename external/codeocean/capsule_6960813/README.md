# CodeOcean capsule 6960813: Arborist evaluation

This directory contains an imported source snapshot from **CodeOcean**, not code
originally developed in `exa-spim-agent`.

| Provenance | Value |
|---|---|
| Original capsule | [6960813](https://codeocean.allenneuraldynamics.org/capsule/6960813/tree) |
| Title | Arborist (neuron proofreading) - Evaluation |
| Author listed in the export | Anna Grim; see [metadata/metadata.yml](metadata/metadata.yml) |
| Source | User-supplied `capsule-6960813.zip` |
| Imported | 2026-11-04 |
| Export SHA-256 | `57c902913cbc591ee70557f8073c2bdf20a5941546520ae8dd29faef42906b20` |

The export does not identify a capsule revision or include a license file. The
archive hash identifies the supplied snapshot; it does not establish that this is
the latest version available on CodeOcean.

All 16 imported files are byte-for-byte identical to their ZIP entries. Their
original relative paths and author notices are preserved.
[UPSTREAM_SHA256SUMS](UPSTREAM_SHA256SUMS) records their hashes. This README,
the checksum manifest, and the local `.gitignore` were added during import.
Future local adaptations should be documented separately from upstream behavior;
retain the manifest as the baseline for this snapshot.

## Included files and scope

| Path | Purpose |
|---|---|
| `code/main.py`, `code/run` | Evaluate skeleton metrics at proofreading pipeline stages |
| `code/residual_merge_detection/` | Evaluate residual merge predictions; generate figures and an HTML slide |
| `code/review_merges/` | Build human-review pages with image projections, GT and fragment skeletons |
| `environment/` | Original Docker and dependency-install recipes |
| `.codeocean/app-panel.json` | Original named-parameter panel, including its upload default |
| `.codeocean/datasets.json` | Attached data-asset identifiers and mount names |
| `metadata/`, `REPRODUCING.md` | Original attribution and reproduction instructions |

The ZIP itself remains outside the repository. IDE history, logs, bytecode,
credentials, machine-local settings and the CodeOcean authentication helper were
not imported. Attached datasets and prediction results were not present in the ZIP.
The Docker recipe still references the omitted `git-ask-pass` helper; this is a
source snapshot, not a complete standalone Docker build context.

The code is not wired into `proofreader_evolve` or its tests. Importing it does not
change candidate labels, training, selection or promotion. Brain **794495** is
reserved for a future independent validation process whose protocol is not yet
designed; do not use this snapshot to add it to current TRAIN/OOF, tuning, testing
or evolution promotion. Historical use of this brain is not erased by that
reservation. No real 794495 evaluation was performed for this import.

## Entry points and runtime requirements

- `code/run` sets up GCS credentials and forwards arguments to `main.py`.
  The main program requires `--result_prefix` and `--segmentation_path`.
  It infers the brain, segmentation and result IDs from the final three path
  components of the result prefix. Neither path has a saved default in the
  exported app panel. Direct CLI execution defaults to `--upload 1`; the
  CodeOcean panel defaults to `0`. Set `--upload 0` explicitly for local evaluation.
- `code/residual_merge_detection/evaluate.py` reads prediction CSVs and existing
  pipeline evaluation outputs. It does not load a dense segmentation volume.
- `code/review_merges/build_review.py` reads local evaluation outputs and a cloud
  image volume to generate human-review HTML. It does not implement an LLM judge.

The original environment uses Python 3.12.4 and installs
`segmentation-skeleton-metrics` at `478aede` and
`neuron-proofreading-evaluation` at `f15bd8d`; see
[environment/postInstall](environment/postInstall). The dependency environment,
cloud access and data mounts have not been reproduced here. Any execution in this
workspace must use a compute node.

## Data paths and relationship to this repository

The main evaluator constructs its GT path as
`gs://allen-nd-goog/ground_truth_tracings/<brain_id>/world/`.
Our labeled-cache pipeline uses the same parent directory's `voxel/` variant.
The source-only import did not verify neuron identities or coordinates after
conversion. A subsequent user-requested sample comparison is documented in
[evaluation/README.md](../../../evaluation/README.md); it checks current cloud
sources, not historical cache or attached-data snapshots.

The review script fixes `BRAIN = "794495"` and reads this image:

```text
gs://allen-nd-goog/from_aind/exaSPIM_794495_2026-01-21_14-25-07_training-data/whole-brain-denoised/fused.zarr/0/
```

At import, this repository's reviewed image source for 794495 instead points to:

```text
s3://aind-open-data/exaSPIM_794495_2026-01-21_14-25-07_processed_2026-07-28_08-24-11/fusion/fused.zarr/
```

The acquisition identifier matches. The subsequent three-patch comparison found
matching image metadata and high pixel correlation, but different pixel values;
global registration and processing equivalence remain unverified. Neither image
URI is a segmentation URI; the main evaluator needs the actual segmentation
supplied separately.

## Known upstream limitations retained in this snapshot

- `REPRODUCING.md` shows obsolete positional arguments. The current main program
  requires the named arguments described above; its detailed merge/split PR
  branches are disabled by default.
- The review script expects `/root/capsule/data/aggs08_arborist_v2`, whereas the
  exported data mounts are `aggs08_old_pipeline` and `arborist_v2_agg08_eval`.
  Its data root and configuration names need checking before use.
- The residual merge evaluator raises `IndexError` when no GT merge site is
  reachable by a retained prediction. Multiple predictions near one GT merge can
  each count as a true positive; matching is not one-to-one. Both behaviors were
  confirmed with synthetic inputs on compute node n06 during the source review.
- The residual evaluator's headline recall uses reachable sites. All-site recall
  and candidate coverage must also be considered. Thresholds span 0 to 1, which
  cannot be applied unchanged to arbitrary evolution ranking scores.

These are review findings, not fixes included in this import. No upstream source
file was edited or dataset downloaded.

## Verify the imported snapshot

From this directory, run `sha256sum -c UPSTREAM_SHA256SUMS` to compare the imported
files with the recorded baseline. This is a file-integrity check, not an
end-to-end evaluation or a check of cloud data freshness.
