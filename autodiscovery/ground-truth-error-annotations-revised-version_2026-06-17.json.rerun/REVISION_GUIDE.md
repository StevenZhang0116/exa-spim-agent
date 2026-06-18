# Rerun revision guide

Each `hypo_<id>.py` here is one hypothesis's recorded experiment script. The
reruns fail mostly for DATA-LOADING / ENVIRONMENT reasons, not analysis reasons.
Revise ONLY the data-loading and bootstrap part of each script; keep the
statistical ANALYSIS and what it prints byte-for-byte identical so the rerun is
a faithful reproduction.

You MAY change, near the top of each script:
- Replace the dataset search (hardcoded paths, `os.path.exists`, `glob.glob`,
  "dataset not found" gates) with a DIRECT load of the provided pkl:

      import os, pickle
      with open(os.environ["RERUN_PKL"], "rb") as f:
          payload = pickle.load(f)

  Keep the SAME variable name the rest of the script uses (e.g. `payload`,
  `data`) and the same downstream keys (`fragments_graph`, `gt_graph`,
  `gt_edge_error`, `gt_node_canonical_label`, `gt_merge_sites`, ...).
- Fix environment/version problems that block the load, e.g. a pkl written with
  NumPy 2 that won't unpickle under NumPy 1.x — pin/upgrade inside the script
  (`subprocess.check_call([sys.executable,"-m","pip","install","-q","numpy>=2"])`
  BEFORE importing numpy), or otherwise make the unpickle succeed.
- Remove `pip install` RETRY LOOPS that re-install on every iteration and cause
  timeouts; install each dependency at most once, quietly, up front.

You MUST NOT change:
- The statistical test, its parameters, the sampling/grouping logic, the effect
  the script computes, or the lines it prints. The whole point is to re-measure
  the SAME analysis on the real data.

Print a clear `Loading dataset from: $RERUN_PKL` line so the log shows the fix
took effect. `MPLBACKEND=Agg` is already set (headless plotting is fine).
