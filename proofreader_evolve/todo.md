# proofreader_evolve — TODO / scratch notes

A running memo for pending work and ideas. Move finished items to "Done" with a date.

## TODO

- [ ] **Export the "repaired" artifact (the result of applying the final policy).**
  A finished run currently produces only the *policy*
  (`runs/<run>/artifacts/heuristics.py`'s `propose_edits`), not the *repaired
  result* of applying it. Each generation's `merge_labels` edits are applied only
  transiently to a graph copy inside `incremental_scoring.score_incremental` for
  scoring, then discarded — never written to disk. The only pkl in a run dir is
  `prepared_<brain>.pkl`, which is the *input cache* (the candidate-invariant
  PreparedBrain), **not** a repaired graph.
  - Need a new script: take `run_name` + `brain` → load that run's final
    `artifacts/heuristics.py` → run `propose_edits` on the brain to get all edits →
    **persist** the edits applied and export the result.
  - Output format TBD: edit-list JSON (lightweight, auditable) / a saved repaired
    graph pkl (directly usable downstream) / both. Decide, then implement.
  - Relevant code: `harness/candidate.run_candidate` (run policy → edits),
    `harness/edit_handler.EditHandler` (apply edits),
    `harness/incremental_scoring` (graph-copy application logic to reuse).

## Done
