# Fixed-pool scorer evolution

`proofreader_evolve` has one workflow: improve native-label Precision@K on the
exact candidate pools of selected frozen AutoDiscovery detectors. It does not
edit graphs, expand candidates, retrain detectors or deploy scorers automatically.

## Pipeline

![Fixed-pool scorer evolution workflow](workflow_diagram.svg)

1. Select one frozen merge-junction detector and one split-site detector from
   `autodiscovery-application/`, with their matching fitted joblibs.
2. Prepare tables from matching `cache/dataset_cache_<brain>_mcl<N>_add.pkl`
   caches. Call each detector's original universe builder and feature extractor;
   preserve candidate identity and split occurrences without extra enumeration caps.
   Enable only the native analysis groups owning columns in that fitted model's
   `feature_order` (including definedness columns). Shared groups remain intact;
   unknown or ambiguous feature ownership fails instead of enabling all groups.
3. Store predictor features plus `detector_score` separately from candidate
   identities and evaluator labels. Validate table, source and model provenance.
4. Evaluate the seed scorer and retain a separate frozen-detector baseline.
5. In each generation, a fresh LLM session edits `scorer.py` and `rules.md`
   using TRAIN feedback only. There is one child per generation.
6. Require strict macro Precision@K improvement with no brain/kind regression on
   TRAIN, then apply the same gate on development validation. Reject invalid,
   nondeterministic or timed-out scorers; keep the previous accepted parent.
7. Save the accepted scorer and complete generation measurements/accounting.

## Commands

Use panda on an allocated compute node for table preparation:

```bash
python proofreader_evolve/prepare_feature_tables.py --brains 789202 794491 --mcl 100
python -m proofreader_evolve.cli.preflight --brains 789202 794491 --mcl 100
python -m proofreader_evolve.cli.run_evolution \
  --train-brains 789202 --validation-brains 794491 \
  --merge-k 100 --split-k 100 --generations 5
```

Preparation can scan whole graphs; it is not a smoke test. One brain is loaded
per subprocess, and the first failure stops the launcher. Matching tables are
reused. Evolution itself requires prepared tables and never builds them.
The preparation launcher accepts `--threads all` to use the process's available
CPU count, respecting CPU affinity, for numerical-library threads. It logs the
requested setting, resolved thread count and available CPUs. The default is 1;
this setting does not parallelize serial Python feature loops.
For other detector directories, use `cli.precompute_error_scores` and pass the
same `--merge-dir`/`--split-dir` selections to preflight and evolution.

`--generations 0` evaluates the seed/baseline without calling an LLM. Preflight
does not test API access unless explicitly given `--check-api`; that probe does
not validate the full reviser SDK/tool workflow. Live evolution requires
`ANTHROPIC_API_KEY`.

## Scorer contract

`score_candidates(features, ctx)` receives a copied feature-only DataFrame;
`ctx` contains only `kind`. Return one finite, deterministic score per row in
the original order. No candidate IDs, GT labels or brain identifiers are supplied
to the scorer. TRAIN feedback may contain bounded anonymous labeled examples;
validation examples and measurements are not provided to the reviser.

The harness takes exactly `min(K, pool size)` per brain/kind, with stable
candidate-ID tie breaking. It averages cell precision equally. Pool identity,
labels, budgets, frozen weights and feature definitions cannot change within a
run. Returning fewer rows cannot improve precision. Ties do not pass the gate.

## Modules and artifacts

- `cli/run_evolution.py`: lightweight entrypoint to `run_precision_evolution.py`.
- `cli/precompute_error_scores.py`, `harness/native_pool.py`: native tables,
  frozen detector inference and provenance.
- `harness/dataset.py`: trusted cache loading and brain/MCL checks only.
- `harness/fixed_pool_scoring.py`: ranking evaluation and acceptance.
- `harness/reviser_session.py`, `reviser_access.py`, `policy_runtime.py`:
  SDK configuration, exact file-tool allowlists and evaluation timeout.
- `harness/run_records.py`, `collect_run.py`, `plotting/`: precision-only
  record reading, summaries and plots.

Runs save `manifest.json`, `baseline.json`, `seed.json`, `final.json`,
`best_scorer.py`, `ledger.jsonl`, `tool_audit.jsonl`, and per-generation
`scorer.py`, `rules.md`, `train_feedback.json`, `evaluation.json`.
Costs are summed per fresh session; missing amounts remain unknown. Missing
evaluations are never converted into zero or retained-parent measurements.

## Removed workflow and retained data

The `--objective structural_repair` workflow and its compatibility exports,
graph-edit scoring, repair-site enumeration, detector query adapter, image repair
helpers, seeds, priors, structural tests and analysis notebooks were removed.
The CLI no longer accepts `--objective`, `--candidate-mode`, repair enumeration
flags or the old `--test-brains` alias. Use `--validation-brains`.

Historical `runs/`, feature tables, `log/` and `prepared_cache/` data were not
deleted. They are not silently migrated or reinterpreted: current summaries and
plots accept precision runs only. Old executable analysis requires its original
code version. A pre-removal source/notebook snapshot was saved outside the repo
at `/tmp/proofreader-structural-removal-E3rZk1/source-backup.tar.gz`; this is a
temporary recovery copy, not permanent archival storage.

## Integrity and interpretation limits

Source fingerprints are strict, so this cleanup can invalidate existing native
tables even though the detector candidate pool is unchanged. Rebuild if preflight
requests it; do not rename table directories or bypass provenance checks.
Source caches, detector scripts and joblibs are never rewritten by preparation.

The default models were fitted on 794495. Training/validation brains must be
disjoint, and a detector-fitted brain cannot serve as validation. Repeated
validation guides selection; it is not an untouched final test. Sparse native
annotations are not exhaustive biological truth, and ranking gains do not prove
safe graph repair. SDK file-tool guards are not an OS sandbox for generated
Python; execution assumes trusted code.

## Regression tests

```bash
python -m unittest discover -s proofreader_evolve/tests -t .
```

Tests use tiny fixtures, frozen-model parity checks, mocked evolution and local
plots. They do not load full brains or call a live LLM.
