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
change candidate labels, training, selection or promotion. Brain **794495** has
no role inside an evolution run; since 2026-11-14 it is used only for post-analysis
of a finished run's final policy, scored with this repository's own tools
([record](../../../proofreader_evolve/evaluation/README.md)). Do not use this
snapshot to add it to TRAIN/OOF, tuning or evolution promotion. Historical use of
this brain is not erased by that. No real 794495 evaluation was performed for this
import. On 2026-11-18 the main program was run on this repository's own inputs for the
unproofread segmentation of 794495; see "Running on our own inputs" below.

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
[environment/postInstall](environment/postInstall). Since 2026-11-18 a local conda
environment holds these dependencies (next section). The S3 access and the data
mounts have not been reproduced here; GCS reads work with the repository's
credentials. Any execution in this workspace must use a compute node.

## Local environment `arborist_eval` (2026-11-18)

The capsule's programs do not run in `panda`: `neuron_proofreading_evaluation` is
absent there, and `panda`'s `segmentation_skeleton_metrics` (commit `d76557d`,
2026-05-21, with two local fixes) predates the method that `code/main.py` replaces.
Moving `panda` to the capsule's commit would break this repository's own scripts
(`data_handling` was renamed to `datamodules`, and the merge metric was restructured),
so the capsule has an environment of its own and `panda` is unchanged.

```text
/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/conda_envs/arborist_eval
```

Activate it by that path (`conda activate <path>`); it lies outside the home
directory, which had too little free space. Python 3.12.4, `torch` 2.6.0+cu124 from
the capsule's own install line, 119 distributions (`arborist_eval.freeze.txt` beside
the environment). The five first-party packages are fixed to commits:

| Package | Commit | Why this commit |
|---|---|---|
| `segmentation_skeleton_metrics` | `478aede` (2026-09-29) | fixed by the capsule |
| `neuron_proofreading_evaluation` | `f15bd8d` (2026-09-29) | fixed by the capsule |
| `arborist` | `c7a0258` (2026-09-23) | not fixed by the capsule; last commit on `main` up to 2026-09-30 |
| `neuron-proofreader` | `bb88768` (2026-09-28) | not fixed by the capsule; last commit on `main` up to 2026-09-30 |
| `aind-exaspim-dataset-utils` | `689b6ca` (2026-07-02) | dependency of `neuron-proofreader`; its `main` |

`neuron_proofreading_evaluation` declares `arborist`, `segmentation-skeleton-metrics`
and `neuron-proofreader` at their `main` branches, so which commits the capsule's
image really holds is not known. The three unfixed packages were taken from the same
weeks as the two fixed ones; the names the evaluation code imports exist at these
commits.

Recipe (compute node). The environment was built in two passes, because the user site
was ignored only in the second one; the commands below are the same steps in one pass
and were not run in exactly this form:

```bash
ENV=/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/conda_envs/arborist_eval
source /shared/utils.x86_64/anaconda3-2024.10/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 CONDA_PKGS_DIRS=$(dirname $ENV)/.pkgs
conda create -y -p $ENV python=3.12.4
conda env config vars set PYTHONNOUSERSITE=1 -p $ENV
PY=$ENV/bin/python
$PY -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
$PY -m pip install \
  aiobotocore==2.13.3 "boto3>=1.34,<2" botocore==1.34.162 einops fastremap fonttools \
    gcsfs google-cloud-storage matplotlib more_itertools networkx numpy pandas pillow \
    plotly psutil rustworkx s3fs==2024.12.0 scikit-image scikit-learn scipy \
    smartsheet-python-sdk tensorboard tensorstore==0.1.71 tifffile \
    torch_geometric==2.5.3 torcheval tqdm urllib3==1.26.15 zarr
G=git+https://github.com
$PY -m pip install --no-deps \
  "segmentation_skeleton_metrics @ $G/AllenNeuralDynamics/segmentation-skeleton-metrics.git@478aedeb6c072281f64c611c92e35f0fbde9343a" \
  "neuron_proofreading_evaluation @ $G/AllenNeuralDynamics/neuron-proofreading-evaluation.git@f15bd8d3f13fb5c4f007129a7af72b9a5034d7ae" \
  "arborist @ $G/AllenInstitute/arborist.git@c7a0258f5851cb226f33ea601be2fb1ea565cb33" \
  "neuron-proofreader @ $G/AllenInstitute/neuron-proofreader.git@bb887689524299fad9afe97eab659fb58276df2c" \
  "aind-exaspim-dataset-utils @ $G/AllenNeuralDynamics/aind-exaspim-dataset-utils.git@689b6ca78daf95cedabb27ad1f4340ed01945515"
$PY -m pip check
```

Two points of the recipe matter:

- `--no-deps` for the five first-party packages. With their own dependency lists pip
  would fetch `main` of three of them and replace the fixed commits. Their third-party
  requirements are the list in the second `pip install`, with `neuron-proofreader`'s
  exact pins.
- `PYTHONNOUSERSITE=1`, also set by the environment on activation. The home directory
  holds a user site for Python 3.12 (`~/.local/lib/python3.12`, 79 distributions of
  other projects), which Python puts before the environment's own packages. The first
  pass took nine requirements from there (numpy, pandas, scipy, matplotlib and others)
  and they were then installed into the environment. Calling `$ENV/bin/python`
  without activating the environment needs `-s` or the variable.

Checked on n243: `pip check` reports no broken requirement; the five commits above are
the installed ones; numpy, pandas and torch are imported from the environment;
`code/main.py` imports, and its replacement of `MergeCountMetric.record_merge_site`
takes effect; `main.py`, `residual_merge_detection/evaluate.py` and
`review_merges/build_review.py` print their `--help`. That check ran no evaluation and
opened no cloud path. The node has no GPU, so CUDA was not exercised.

### Running on our own inputs (2026-11-18)

`code/main.py` needs a proofreading result on S3, which this workspace does not have
(no Arborist run, no AWS credentials). Its first step, the segmentation before any
correction, needs only ground truth, fragment SWCs and the segmentation, which the
repository reads from GCS. [evaluation/run_capsule_original.py](../../../evaluation/run_capsule_original.py)
runs `main.py` unmodified on them and replaces the three inputs that come from a result
directory; the fragment SWCs of the Google segmentation hold voxel indices and are
converted to um for it. The differences from a run on an Arborist result, and the
figures for 794495 (210 merges, 4,656 splits, 109 truncations over 22 neurons), are in
[evaluation/README.md](../../../evaluation/README.md). The corrected steps and the two
other tools still need an Arborist result or the capsule's data assets. With `--fragments`
(and `--connections` for joined fragments) the same script evaluates another fragment set
against the same ground truth, which is how an edited fragment graph of a labeled cache is
judged; procedure and results are in that README.

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
