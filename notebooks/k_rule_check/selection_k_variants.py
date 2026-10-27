"""Selection-side check: for every measured candidate of the two stopped runs, Precision@K on the
802449 merge OOF scores at K = 300 (current cap), P, 2P, 3P, and the relation to the validation
outcome of the submitted candidate (ledger)."""
import json, pickle, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from proofreader_evolve.harness import label_availability as la
from proofreader_evolve.harness.native_pool import candidate_keys
from proofreader_evolve.harness.fixed_pool_scoring import labeled_order

def _kind(d):
    m = json.loads((d / 'meta.json').read_text())
    return m.get('kind') or (m.get('provenance') or {}).get('kind') or (len(np.load(d / 'evaluator_labels.npy', mmap_mode='r')) == 553193 and 'merge')
TABLE = next(d for d in (REPO / 'proofreader_evolve/feature_tables/802449').iterdir() if d.is_dir() and not d.name.endswith('.availability') and _kind(d) == 'merge')
RUNS = REPO / 'proofreader_evolve/runs'
RUN_IDS = ['precision_20261026_210812_yqeacq5s', 'precision_20261027_000457_5d0_pbkb']

print('merge table dir:', TABLE.name)
truth_bin = np.load(TABLE / 'evaluator_labels.npy')
with open(TABLE / 'candidates.pkl', 'rb') as fh:
    rows = pickle.load(fh)
keys = candidate_keys('merge', rows)
with np.load(TABLE.parent / (TABLE.name + '.availability') / 'availability.npz', allow_pickle=False) as data:
    mask = data[la.DEFAULT_DEFINITION['merge']]
truth = la.three_valued(truth_bin, mask)
P = int(np.nansum(truth))
print(f'802449/merge rows={len(truth)} labeled={int(np.isfinite(truth).sum())} positives={P}')
KS = {'K300': 300, 'K=P': P, 'K=1.5P': int(np.ceil(1.5 * P)), 'K=2P': 2 * P, 'K=3P': 3 * P}

def metrics(scores):
    order = labeled_order(scores, truth, keys)
    hits = np.cumsum(truth[order])
    return {name: float(hits[min(k, len(order)) - 1] / min(k, len(order))) for name, k in KS.items()}

out = []
for run in RUN_IDS:
    rd = RUNS / run
    ledger = [json.loads(l) for l in (rd / 'ledger.jsonl').open()]
    seed = np.load(rd / 'train_seed/merge/selection_state.npz')['802449/merge:scores']
    seed_m = metrics(seed)
    print(f'\n=== {run}\nseed selection: ' + ' '.join(f'{k}={v:.4f}' for k, v in seed_m.items()))
    for row in ledger:
        g = row['generation']
        sub = row.get('submitted_experiment')
        val = row.get('validation') or {}
        cells = val.get('cells') or {}
        merge_mean = np.mean([c['precision'] for n, c in cells.items() if n.endswith('/merge')]) if cells else float('nan')
        merge_hits = sum(c['tp'] for n, c in cells.items() if n.endswith('/merge')) if cells else -1
        attempts = sorted((rd / f'gen{g:03d}/experiments').glob('attempt*/selection_state.npz'))
        per_attempt = {}
        for p in attempts:
            try:
                sc = np.load(p)['802449/merge:scores']
            except KeyError:
                continue
            per_attempt[p.parent.name] = metrics(sc)
        if not per_attempt:
            continue
        sub_name = sub.split('/')[-1] if sub else None
        sm = per_attempt.get(sub_name)
        spread = {k: (min(v[k] for v in per_attempt.values()), max(v[k] for v in per_attempt.values())) for k in KS}
        status = 'repeated' if row.get('submission_repeat') else ('accepted' if row.get('accepted') else 'rejected')
        print(f"gen{g:02d} {row['target_kind']:5} {status:8} submitted={sub_name} val_merge_mean={merge_mean:.4f} merge_hits={merge_hits} "
              + (' '.join(f'{k}={sm[k]:.4f}' for k in KS) if sm else '(no state for submitted)')
              + '  | attempts n=%d spread ' % len(per_attempt) + ' '.join(f'{k}:[{a:.3f},{b:.3f}]' for k, (a, b) in spread.items()))
        if sm and row['target_kind'] == 'merge' and cells and not row.get('submission_repeat'):
            out.append({'run': run, 'gen': g, 'val_merge_mean': merge_mean, 'merge_hits': merge_hits, 'val_mean': val.get('macro_precision'), **sm})

print('\n=== Spearman between selection metric of the submitted merge candidate and its validation merge mean (both runs pooled, n=%d)' % len(out))
for k in KS:
    x = [o[k] for o in out]; y = [o['val_merge_mean'] for o in out]
    r, p = spearmanr(x, y)
    print(f'{k:7} rho={r:+.3f} p={p:.3f}')
for run in RUN_IDS:
    sub = [o for o in out if o['run'] == run]
    print(f'--- {run} n={len(sub)}')
    for k in KS:
        r, p = spearmanr([o[k] for o in sub], [o['val_merge_mean'] for o in sub])
        print(f'{k:7} rho={r:+.3f} p={p:.3f}')
