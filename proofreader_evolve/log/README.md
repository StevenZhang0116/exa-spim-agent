# Console logs

`prepare_feature_tables.py` automatically mirrors console output here, including
stdout/stderr from precompute and preflight subprocesses. Each run creates
`prepare_feature_tables_<timestamp>_<pid>.log.txt` and flushes output as it arrives.
The log includes the invocation, start time, host, Python version, commands,
and final exit status and duration. A missing footer indicates an abrupt stop.
Use `--log-txt PATH` to append to a chosen file. `--dry-run` creates no log.
Generated `.log.txt` files in this directory are ignored by Git.

Scorer evolution saves live run/generation traces as `trajectory.txt` and
`trajectory.jsonl` under `runs/precision_*/` and each `genNNN/`. Generation
directories also retain prompts, final replies, parent snapshots and code diffs.
Console output includes progress and metric changes; redirect or tee it here
if you also want one combined console transcript.

This directory also contains retained log mirrors from the removed graph-edit
workflow. Nothing here is written by current scorer evolution.

Precision runs save manifest, baseline/seed/final measurements, generation
feedback/evaluations, ledger and best_scorer.py under runs/precision_*.
Redirect stdout explicitly if you need a console transcript. Historical logs
were not deleted or migrated by the code cleanup.
