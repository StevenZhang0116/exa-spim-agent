# Central run-log mirror

Every `cli/run_evolution.py` run tees its full console output to TWO places:

- `runs/<run_id>/run.log` — beside the run's artifacts (ledger, gen snapshots);
- `log/<run_id>.log` — a byte-identical durable mirror in this folder, plus one
  line per run appended to `log/runs_index.tsv`
  (`started-at <TAB> run_id <TAB> argv`).

Why: run directories get pruned when experiments are cleaned up, and their logs
used to go with them. This folder keeps the run history (what was launched with
which flags, seed scores, per-generation fitness, accept/reject decisions,
crashes) readable after `runs/<id>/` is gone. The mirror is best-effort — if it
cannot be opened the run continues with only `runs/<id>/run.log`.

Files here are append-only text; prune manually when they outlive their use.
