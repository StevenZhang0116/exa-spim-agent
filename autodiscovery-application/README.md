# autodiscovery-application

Downstream **applications** of AutoDiscovery findings: code that takes the
hypotheses a run confirmed and turns them into a working detector, rather than
code that discovers or audits hypotheses.

- `autodiscovery/` — the runs themselves (JSON exports, `.summary.md` reports).
- `autodiscovery-application/` — *this* folder: what we build with a run's output.

## Layout convention

**One subfolder per originating AutoDiscovery run, named exactly after the run.**
Every application is downstream of a specific run's hypotheses, so the folder
name is the provenance:

```
autodiscovery-application/
└── <RUN>/                         # matches autodiscovery/<RUN>.json
    ├── README.md                  # which hypotheses → which features, results
    ├── <script>.py                # the application
    └── <generated outputs>        # CSVs etc. (rebuildable)
```

Current runs:

| Subfolder | Origin run | What it builds |
|---|---|---|
| `merge-error-794495-mcl100_2026-08-04/` | `autodiscovery/merge-error-794495-mcl100_2026-08-04.json` | Logistic-regression merge detector over the run's 10 confirmed features |

## Generating a subfolder

`agentic/run_detector_build_workflow.py` builds one of these from a finished
discovery run — it reads the run's `<RUN>.summary.md` and the reproducer's
`<RUN>.json.rerun/hypo_<id>.py` scripts, lifts each confirmed hypothesis's
feature computation into one consolidated script, runs it, audits the fit for
imputation confounds, and writes the README:

```bash
conda activate panda                     # on a COMPUTE NODE — the pkl needs >20 GB
python agentic/run_detector_build_workflow.py \
    autodiscovery/<RUN>.json \
    --pkl cache/dataset_cache_<brain>_mcl<N>_add.pkl
```

It writes into `autodiscovery-application/<RUN>/` by default, which is where the
naming convention above comes from. Steps: `inventory` → `generate` →
*(driver-owned compute)* → `audit`; resume with `--steps` / `--from`, and use
`--skip-compute` on a login node. The subagent it drives is
`.claude/agents/discovery-detector-builder.md`.

The `merge-error-794495-mcl100_2026-08-04/` subfolder was built by hand before
this workflow existed; the workflow encodes the same procedure, including the
audit step that caught the AUC inflation documented in its README.

## Why per-run subfolders

A run's hypotheses are tied to the brain and `mcl` level they were generated on,
and to that run's specific statistical verdicts (SOUND / WEAK / MINOR). A script
built on run A's features is not interchangeable with one built on run B's, even
if both detect merges — the feature definitions, thresholds, and imputation
defaults all come from the originating run. Keeping them separate means a
finding that later fails to replicate invalidates one subfolder, not the folder.

## Generated outputs

CSVs and other artifacts are written inside the run subfolder and are fully
rebuildable from the script plus the `_add.pkl` cache. They are small enough to
track (~1 MB) — see the repo `.gitignore` if that changes.
