# Run analysis

Use the panda environment. These commands read saved artifacts; performance and
search plots do not load brain caches, execute policies or call an LLM.

```bash
python -m proofreader_evolve.harness.collect_run <RUN_OR_PATH>
python -m proofreader_evolve.plotting.plot_run_performance <RUN_OR_PATH> --csv metrics.csv
python -m proofreader_evolve.plotting.plot_search_dynamics <RUN_OR_PATH> --md digest.md
```

Only fixed-pool precision runs are supported. Missing/failed evaluations remain
missing, not zero or the parent's retained score. TRAIN/validation curves require
the same policy hash on both sides. Repeated validation is not final testing.

Costs are summed across fresh per-generation sessions, even when the amounts
happen to increase monotonically. Missing recorded costs propagate as unknown.
There is no accounting-mode override.

Graph-edit analyses and their notebooks were removed with the structural-repair
workflow. Historical run data remain on disk, but these tools reject their
objectives instead of fabricating precision metrics.
