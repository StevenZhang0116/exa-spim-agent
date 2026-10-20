# Verify in batches, driver-side BH, and the recorded-data audit

Status: implemented 2026-10-20 (`agentic/run_discovery_workflow.py`,
`agentic/recorded_audit.py`, `agentic/rank_by_surprise.py`, the verifier and selector
specs, tests). Not yet run on a full workflow.

## Why

- On 2026-10-19 the verifier audited 91 entries in one turn (2,790 s, $14) from a slice
  whose code and output were cut at 6,000 characters, and computed Benjamini-Hochberg by
  hand from p-values it extracted itself. One turn for everything also sat under the
  agent ceiling.
- Four records across the two runs (new-split-1 ids 46, 89, 90; new-merge-1 id 95)
  reported results for stand-in data (intensity from the gap, coordinate scatter named a
  Hessian), and only an agent reading the output noticed.

## Verify in batches

- `verify-slice` (compute) writes the ranking helper's `--include-code` slice once, with
  `--max-code-chars 20000` and a `recorded_audit` per record.
- `run_verify_batches`: the ranked ids in report order, `VERIFY_BATCH_SIZE` (12) per turn.
  Each batch instruction (`verify_batch_instruction`) names its ids, the slice, the full
  scripts, the baseline card, the canonical-stats reference and the recorded audit, and
  asks for the per-entry bullets plus the entries in the verify verdicts JSON (add or
  replace, keep others). The ledger (`.verify-ledger.json`) records ids whose entry
  carries a Verdict bullet, so a resumed run audits only the open ids.
- After the batches: `_check_phase_verdicts` on the whole file (one reconciliation turn,
  then abort), `multiple_comparisons` over `headline.p` (BH at `BH_Q`, survivors with m =
  extractable and with m = `n_total` of the export, borderline, floor count) written to
  `.multiple-comparisons.json`, and `verify_synthesis_instruction`: the run-wide section
  from the driver's verdict counts and BH text, with cross-entry problems; a verdict may
  be revised for a cross-entry reason if both bullet and JSON change.
- `_settle_verdicts` (the check-reconcile-merge of every fold step) runs after the
  synthesis, as for the other folds.

Expected cost: about 8 batch turns of 12 plus one synthesis turn; more wall time than
one turn, but no truncation, no ceiling risk, and BH from structured p-values.

## Recorded-data audit

- `recorded_audit.py`: per record, output lines naming a proxy / fallback / mock /
  simulation / dummy / unavailable source / missing module (glog lines aside), code lines
  naming a proxy / simulation / synthetic value (`code_substitutes`) or a mock / dummy /
  fallback (`code_mocks`, informational: usually an unpickling shim), and whether an
  image hypothesis shows an image reader in the code and read evidence in the output.
  `suspected` = an output marker, a code substitute, or an image hypothesis without a
  reader or without evidence.
- The driver runs it at startup (`compute_recorded_audit`, before predictive selection,
  also in `--smoke`), writes `.recorded-audit.json`, logs the flagged ids, and appends
  `audit_text` to the selection instruction and every verify batch instruction. The
  slice carries the per-record audit too.
- It is evidence, not a verdict: the selector excludes a flagged record only when its
  output shows the measured quantity is a stand-in; the verifier treats such a record as
  invalidly computed.

## Tests

- `test_recorded_audit.py`: a real image read with glog noise is clean; output marker +
  code substitute + no read evidence (id 46); image hypothesis without a reader (id 90);
  an unpickling mock alone is informational; the CLI and the instruction text.
- `test_discovery_smoke.py`: step shape (slice before verify, no inline instruction,
  shared verdicts file); batches then synthesis with the driver's BH figures; resume
  audits only the open ids; `multiple_comparisons` on known p-values; the audit text in
  the selection and batch instructions.
- `test_rank_by_surprise.py` still passes with the added `recorded_audit` field.
