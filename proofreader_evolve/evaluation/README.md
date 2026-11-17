# Post-analysis on the reserved brain 794495

Brain 794495 has no role inside an evolution run (not TRAIN, selection or development
validation). After a run has ended, its final policy is applied to 794495 once and the result
is recorded here. Nothing in this folder is fed back to an evolution run.

## Rules of this folder

- **One evaluation at a time.** The folder holds the artifacts of the latest evaluation only.
  Before a new evaluation is launched, everything here except this file is deleted (the
  earlier run directory `precision_*/`, its console log `eval_794495_*.log.txt` and
  `.descriptor_restore/`).
- **This file is the record.** It holds the latest result in full and one line per earlier
  evaluation, so the figures survive the deletion of the artifacts.
- **Git tracks this file only.** `.gitignore` ignores `proofreader_evolve/evaluation/*` and
  re-includes `README.md`.

## How an evaluation is run

No dedicated command exists. The driver is started with the finished run's policy as its
starting scorer, zero generations and 794495 in the validation position:

```bash
cd <repository root>

# 1. remove the previous evaluation's artifacts (everything except this file)
find proofreader_evolve/evaluation -mindepth 1 -maxdepth 1 ! -name README.md -exec rm -rf {} +

# 2. evaluate the final policy of <RUN>; check first that no other run uses <NODE>
REPO=$PWD
LOG="$REPO/proofreader_evolve/evaluation/eval_794495_$(date +%Y%m%d_%H%M%S).log.txt"
nohup setsid /home/zihan.zhang/.conda/envs/panda/bin/python -u \
  -m proofreader_evolve.cli.launch_evolution \
  --job-id <JOB> --node <NODE> --cpus 40 --runs-dir "$REPO/proofreader_evolve/evaluation" -- \
  --detector-prior raw \
  --train-brains 802449,747807,750318,754610,754612,754613 \
  --selection-brains all --selection-holdout-brains all \
  --validation-brains 794495 --generations 0 \
  --start-from "$REPO/proofreader_evolve/runs/<RUN>/best_scorer.py" \
  --objective average_precision --precision-margin 0.002 \
  --aggregation positive-weighted --min-positives 10 \
  --classifier-memory-mb 98304 --classifier-time-budget 3600 --policy-time-budget 2400 \
  --classifier-threads 32 --classifier-search-seconds 7200 --descriptor-workers auto \
  > "$LOG" 2>&1 < /dev/null &
```

Run the two steps as separate commands: an `&&` chain ending in `&` puts the whole chain in the
background and loses the variables.

The run's `descriptors.json` and `model_artifacts/` beside `best_scorer.py` are read
automatically. The driver also loads the six TRAIN brains and re-evaluates the policy on them;
scoring 794495 does not need that, it is what this entry point does. Descriptor values already
in `proofreader_evolve/descriptor_bank_raw/` are reused. The re-evaluation doubles as a check:
the TRAIN figures in the new `final.json` must equal those of the original run. No language
model is called.

Results: `final.json` (`validation` holds 794495), `baseline.json` (`raw_heuristics`) and
`validation_scores/seed.npz` (the GT-labeled rows' scores) in the new run directory.

794495 needs its raw candidate tables (`bash proofreader_evolve/rebuild_brains.sh 794495` after a
label refresh or cache rebuild; built 2026-11-14) and a current alignment receipt.

## Reading the figures

- Labels `length_v2` (cache of 2026-11-11). On 794495: merge 9,745 GT-labeled rows with 173
  positives, split 6,076 with 2,497. K = GT positives of the pool.
- "Raw heuristics" is the learning-free comparator of `harness/detector_prior.py`: split
  candidates ranked by the smallest gap, merge candidates by junction arity with the tip gap as
  tie-breaker. On 794495: merge average precision 0.0301 (10 of 173 in the top K), split
  0.8255 (1,956 of 2,497). A constant ranking has the base rate as average precision: merge
  0.018, split 0.411.
- Point estimates; no resampling interval is computed.
- 794495 is not an untouched brain: it was used in feature discovery and detector fitting and
  as a TRAIN brain of evolution runs in 2026-10. The policies evaluated here come from raw runs
  (no detector) whose TRAIN and development-validation brains exclude 794495, and no session of
  those runs saw it; the candidate rules and the feature specifications were designed with it
  among the brains.

## Latest evaluation: run `precision_20261115_234249_byxx989h` (2026-11-17)

Evaluation run `precision_20261117_113718_2y1gu39n` on n269, 49 minutes. Policy file
`best_scorer.py` of the run, SHA-256 `a55ced76b4ac30af...`, the policy the run's `final.json`
names (sixteen generations; development validation merge 0.1520, split 0.8694).

| kind | statistic | raw heuristics | final policy |
|---|---|---|---|
| merge | average precision | 0.0301 | 0.1173 |
| | AUROC | 0.6321 | 0.7934 |
| | hits in the top K | 10 of 173 | 32 of 173 |
| split | average precision | 0.8255 | 0.8419 |
| | AUROC | 0.8828 | 0.8952 |
| | hits in the top K | 1,956 of 2,497 | 1,955 of 2,497 |

Merge by candidate family: junction 166 positives, 32 in the pool's top K (average precision
0.1206, AUROC 0.7374 within the family); tip_to_shaft 7 positives, none in the pool's top K.

The same policy on the run's five development-validation brains: merge average precision
0.1520 pooled (0.022 to 0.256 per brain), split 0.8694 pooled (0.836 to 0.916 per brain).

Check of the loaded policy: on the six TRAIN brains the evaluation run reproduced the original
run's final figures exactly (merge average precision 0.210192, split 0.920354, the same top-K
hits in every pool).

## History

One line per evaluation; the artifacts of all but the latest are deleted. "Dev" is the same
policy's pooled average precision on the run's development-validation brains.

| evaluated | run | generations | merge AP (dev) | merge hits | split AP (dev) | split hits |
|---|---|---|---|---|---|---|
| 2026-11-15 | `precision_20261114_133715_jkc36yxa` | 10 | 0.1285 (0.1608) | 29 of 173 | 0.8438 (0.8661) | 1,962 of 2,497 |
| 2026-11-17 | `precision_20261115_234249_byxx989h` | 16 | 0.1173 (0.1520) | 32 of 173 | 0.8419 (0.8694) | 1,955 of 2,497 |
