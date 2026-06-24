# proofreader_evolve — TODO / scratch notes

A running memo for pending work and ideas. Move finished items to "Done" with a date.

## TODO

- [ ] **Materialize a repaired GRAPH/volume artifact (heavier export).**
  `notebooks/compare_proofreader_policy.ipynb` now exports the edit-list JSON
  (the lightweight, auditable artifact) and visualizes before/after, but it still
  does NOT write a repaired graph pkl or relabeled volume usable directly
  downstream. If a downstream consumer needs the applied result as data (not just
  the edit list), add an export that applies the EditHandler relabel and saves the
  resulting graph(s). Relevant code: `harness/edit_handler.EditHandler` (apply),
  `harness/incremental_scoring._edited_fragment_graphs` (graph-copy relabel logic).

## Done

- [x] **Apply the final policy to the whole brain & compare before/after**
  (2026-06-23). `notebooks/compare_proofreader_policy.ipynb`: loads the run's
  `gen<NN>/heuristics.accepted.py`, runs `propose_edits` on all 12 GT skeletons of
  789202, scores baseline vs edited (incremental scorer), shows a per-skeleton
  metric comparison table + plot, picks one repaired SplitSite and visualizes its
  receptive field (raw-image MIP + fragment nodes coloured baseline-labels vs
  edited-merged-class), and exports the edit-list JSON. Parameterized by
  RUN_NAME / GEN / BRAIN_ID / PATCH / SEED.
