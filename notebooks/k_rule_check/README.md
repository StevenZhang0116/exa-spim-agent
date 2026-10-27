# K rule check (2026-10-27, objective v5)

Evidence for `WORKFLOW_REVISION.md` "Uniform K = GT positives (2026-10-27, objective v5)".

- `selection_k_variants.py`: re-ranks every measured candidate's stored `selection_state.npz`
  (802449 merge OOF scores) of runs `precision_20261026_210812_yqeacq5s` and
  `precision_20261027_000457_5d0_pbkb` at K = 300, P, 1.5P, 2P, 3P and correlates the submitted
  candidate's score with its validation merge mean from the ledger. Output: `selection_k_variants.txt` (the log, kept as .txt because `*.log` is gitignored).
- `validation_k_variants.py`: re-scores the seed and the descriptor-free validated merge candidates on
  the five validation brains' merge tables (compute node, `panda`, run from the repository root;
  `bank_context_available` is attached from the context cache band) and reports Precision@K per brain
  at the same K variants. Outputs: `validation_k_variants.txt` (the log), `validation_k_variants_per_brain.txt` (per-brain hits at every K, derived from the gitignored `validation_k_variants.json`)
  (per-brain hits); the per-candidate score dumps (`val_scores_*.npz`) are not kept.

Both scripts read run directories under `proofreader_evolve/runs/`, which are not in git.
