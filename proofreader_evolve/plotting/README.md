# Run analysis

Use the panda environment. These commands read saved artifacts; performance and
search plots do not load brain caches, execute policies or call an LLM.

```bash
python -m proofreader_evolve.harness.collect_run <RUN_OR_PATH>
python -m proofreader_evolve.plotting.plot_run_performance <RUN_OR_PATH> --csv metrics.csv
python -m proofreader_evolve.plotting.plot_search_dynamics <RUN_OR_PATH> --md digest.md
python -m proofreader_evolve.cli.build_evolution_report <RUN_OR_PATH>
```

`build_evolution_report` uses the Python standard library only. Its default
output is `<RUN>/report.html`, an offline interactive report with English labels:
submitted/accepted curves, per-brain counts, clickable observed branch ancestry
with attempt details, the current specialist pool, and code diffs. Its three
sections are Run Overview, Exploration Branches, and Current TRAIN Candidate Pool.
Complete generation and event records remain in the run directory.
Removed sections' event payloads and generation/inspection previews are not
embedded in the HTML. The renderer reads bounded trajectory tails only to recover
run/generation status; attempt code and diffs remain in the branch view.
Use `--out report.html` to choose another HTML path or `--watch`
to rebuild every 10 seconds (browser refresh is manual). New evolution runs
automatically rebuild at generation boundaries and at shutdown.

Solid ancestry edges indicate the last observed measured/restored workspace
checkpoint, not inferred intellectual ancestry. Dashed edges indicate cached
measurements. Old records without explicit lineage show no invented edges.
Each parameter/model grid shares its starting checkpoint until the best result
is restored. A validation-rejected candidate may still be retained as a TRAIN
specialist. Candidate and accepted-policy curves are distinguished; missing
evaluations leave gaps. TRAIN champion precision is per kind and appears in the
pool table, rather than being mixed with full-policy macro precision.

Only JSON and allowlisted text artifacts are read, never model files or source
graphs. Partial JSONL records are skipped with warnings; missing evaluations leave
gaps in the curves. Source previews are bounded (32 KB per code file,
12 MiB shared preview budget); full originals stay
on disk. The report includes validation measurements and is for humans only.

Only fixed-pool precision runs are supported. Missing/failed evaluations remain
missing, not zero or the parent's retained score. TRAIN/validation curves require
the same policy hash on both sides. Repeated validation is not final testing.

Costs are summed across fresh per-generation sessions, even when the amounts
happen to increase monotonically. Missing recorded costs propagate as unknown.
There is no accounting-mode override.

Graph-edit analyses and their notebooks were removed with the structural-repair
workflow. Historical run data remain on disk, but these tools reject their
objectives instead of fabricating precision metrics.
