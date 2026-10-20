# Stage 1. Discover: from AutoDiscovery exports to a validated report

*Part of the [exa-spim-agent](../README.md) documentation.*

## 1.1 AutoDiscovery runs

AutoDiscovery is an external autonomous-experimentation system. A run is pointed
at one labeled cache and one task specification from `markdowns/`
(currently `merge_features_for_proofreader_evolve.md` and
`split_features_for_proofreader_evolve.md`, one submission per error type, each with its
companion files; `markdowns/README.md` lists every document, and superseded versions,
including the earlier detector-discovery specifications, are in `markdowns/archive/`). The specification separates GT-informed
characterization from GT-blind detection and names the cache fields a detector
may never read. Each experiment the system performs records its hypothesis,
analysis code, numerical result and interpretation; the whole run is exported
as one JSON file saved as

```text
autodiscovery/<kind>-error-<brain>-mcl<N>[-<variant>]_<date>.json
```

for example `autodiscovery/merge-error-794495-mcl100_2026-08-04.json`. This
export is the raw experimental record. Nothing in it counts as evidence until
the validation workflow below has reproduced and audited it.

## 1.2 Validate one run

```bash
conda activate panda
python agentic/run_discovery_workflow.py \
    autodiscovery/<RUN>.json \
    --pkl cache/dataset_cache_<brain>_mcl<N>_add.pkl \
    --direction predictive \
    --extra-pkl cache/dataset_cache_<other-brain>_mcl<N>_add.pkl   # optional transfer test
```

Runs that follow `markdowns/{merge,split}_features_for_proofreader_evolve.md` also
open a companion file. Pass it with `--companion markdowns/rerank_companion_<brain>_<kind>.npz`
(the `.json` half is found next to it; repeat for each `--extra-pkl` brain). Each
companion is bound to the pkl its manifest names, and every re-run links it into the
script's working directory under the companion names the recorded code uses.

The driver alternates agent steps, which run in fresh Claude Agent SDK sessions
and operate only on files already on disk, with compute steps, which the driver
runs itself as blocking subprocesses. The phases, in order:

| Phase | Kind | What happens |
|---|---|---|
| summarize | compute + agent | The driver computes each companion's baseline card (`agentic/canonical_stats.py`) and scans every record for substituted data (`agentic/recorded_audit.py`); an agent selects the hypotheses to keep (the audit flags are evidence, not verdicts) and ranks them by evidence and surprise. `--direction predictive` keeps every hypothesis except explicit exclusions (invalid, constant, non-predictive, confounded); `positive` and `both` apply a top-K cut instead. |
| reproduce | compute + agent | Re-run every recorded analysis on the origin cache (up to `--jobs` scripts at once, every data read audited, known library-version skews rewritten), export editable scripts, let an agent repair data-loading failures without touching the logic, re-measure only changed scripts (up to 3 repair rounds), diff every recorded output against its rerun (`agentic/output_diff.py`) and fold `REPRODUCED` · `DIVERGED` · `FAILED` verdicts. |
| extrapolate | compute + agent | With `--extra-pkl`, run the reproduced code unchanged on another brain and fold `GENERALIZES` · `PARTIAL` · `DOES-NOT-GENERALIZE` · `INCONCLUSIVE`, each brain judged against its own baseline card. |
| verify | compute + agent | Write the verifier's slice of the export once; audit test choice, effect size against the baseline card, power and deployability in batches of 12 ids, grading `SOUND` · `WEAK` · `MINOR` · `MAJOR` · `CRITICAL` and marking each feature `BLIND-COMPUTABLE` or `GT-REFERENCING`; the driver runs Benjamini-Hochberg over the structured headline p-values and one synthesis turn writes the run-wide section. |
| fix-tests | agent + compute | Write corrected tests for the flagged findings, measured with `canonical_stats` against the raw baseline, re-measure them and fold `UPHELD` · `WEAKENED` · `OVERTURNED`. |
| feature-applicability | agent + driver | Split runs only: classify the node-role requirement of every rerun or fixed source and bind it to the source hash. |

Everything folds into one report with a fixed top-level order: Header, Ranked
Conclusions, Reproduction, Generalization, Statistical Verification, Statistical
Test Corrections, then the Excluded appendix. Each fold step also writes its
verdicts as JSON, which the driver checks against the report and merges into
`<RUN>.json.predictive.verdicts.json`. The driver writes, beside the input JSON:

| Artifact | Content |
|---|---|
| `<RUN>.summary.md` | the report every later stage reads |
| `<RUN>.predictive-selection.json` | the cached exclusion-only selection, reused while the run hash and policy version match |
| `<RUN>.json.predictive.rerun/hypo_<id>.py`, `.fixed/hypo_<id>.py` | the exact re-runnable hypothesis sources that Stage 2 inventories |
| `<RUN>.json.predictive.{reproduce,extrapolate,corrected}.json` | measured results per phase; `.outputs/` beside each holds the full per-record outputs, and `.reproduce.json.diff.json` the recorded-vs-rerun numeric diff |
| `<RUN>.json.predictive.verdicts.json` | the checked verdict tokens of every phase per hypothesis; fix-tests selection and Stage 2 read it first |
| `<RUN>.json.predictive.{baseline,recorded-audit,multiple-comparisons}.json` | the baseline cards, the recorded-data audit, and the Benjamini-Hochberg analysis |
| `<RUN>.split-feature-applicability.json` | split only: source-bound node-role requirements |
| `<RUN>.json.predictive.workflow.log.txt` | the full console log, ending in an `# OK/FAILED` footer |

Every compute step runs its helper again and reuses its per-work-item checkpoint;
there is no `--force`, so delete a checkpoint to recompute it. Agent-step timeouts
scale with the hypothesis count; compute steps get a 24-hour budget and each
experiment script a 1-hour cap. `--smoke` prints the selected hypothesis IDs and
stops without loading data. The `.claude/agents/discovery-*.md` files define the
subagents; `agentic/tests/` holds the unittest suite
(`python -m unittest discover -s agentic/tests`).

## 1.3 Consolidate across runs and rank

```bash
python agentic/run_consolidation_workflow.py        # -> autodiscovery/all-runs.combined.md (+ .zh.md)
python agentic/rank_by_surprise.py autodiscovery/<RUN>.json \
    --rank-by posterior-surprise --direction predictive \
    --predictive-manifest autodiscovery/<RUN>.predictive-selection.json
```

Consolidation clusters equivalent findings across finished reports, keeps one
canonical copy per cluster and lists what is unique and new; it re-runs nothing
and re-judges nothing. The ranking helper reproduces the summarizer's ordering
deterministically. `agentic/collect_summaries.py` gathers report entries across
runs and `agentic/summary_heatmap.py` renders a per-hypothesis verdict heatmap.
