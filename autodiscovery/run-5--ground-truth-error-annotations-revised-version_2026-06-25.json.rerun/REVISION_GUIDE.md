# Rerun revision guide

Each `hypo_<id>.py` here is one hypothesis's recorded experiment script. The
reruns fail mostly because the script cannot LOCATE the dataset (hardcoded paths
/ globs / "dataset not found" gates), not for analysis reasons. Revise ONLY the
data-loading part of each script; keep the statistical ANALYSIS and what it
prints byte-for-byte identical so the rerun is a faithful reproduction.

You MAY change, near the top of each script:
- Replace the dataset search (hardcoded paths, `os.path.exists`, `glob.glob`,
  "dataset not found" gates) with a DIRECT load of the provided pkl:

      import os, pickle
      with open(os.environ["RERUN_PKL"], "rb") as f:
          payload = pickle.load(f)

  Keep the SAME variable name the rest of the script uses (e.g. `payload`,
  `data`) and the same downstream keys (`fragments_graph`, `gt_graph`,
  `gt_edge_error`, `gt_node_canonical_label`, `gt_merge_sites`, ...).

The ENVIRONMENT is owned by the driver, NOT by your script. Do NOT manage
packages or interpreters from inside the script:
- Do NOT `pip install` / `apt install` / `conda install` anything, and do NOT
  add or keep install retry loops — the runner already turns every install
  command into a logged no-op, so they only waste time. The host env is
  pre-provisioned with numpy, pandas, scipy, statsmodels, sklearn, networkx,
  matplotlib, tensorstore and the proofreader package; just `import` them.
- Do NOT touch NumPy: never `pip install numpy...`, never `del sys.modules[...]`
  to reload it, and NEVER put a numpy/site-packages/source directory on
  `sys.path` (e.g. `sys.path.insert(0, "/tmp/np2")`). Importing numpy from a
  source tree raises "you should not try to import numpy from its source
  directory" and breaks the whole script. Just `import numpy as np`.
- If an import genuinely fails, that is an ENVIRONMENT problem for the driver to
  fix (provision the package once, globally) — not something to patch per
  script. Leave it; the driver reports it as an environment failure.

You MUST NOT change:
- The statistical test, its parameters, the sampling/grouping logic, the effect
  the script computes, or the lines it prints. The whole point is to re-measure
  the SAME analysis on the real data.

Print a clear `Loading dataset from: $RERUN_PKL` line so the log shows the fix
took effect. `MPLBACKEND=Agg` is already set (headless plotting is fine).
