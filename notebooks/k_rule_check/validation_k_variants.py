"""Validation-side check: re-score the seed and every validated merge candidate without
descriptor columns of the two stopped runs on the five validation brains (merge tables only),
then report Precision@K at K = 300, P, 1.5P, 2P, 3P per brain from the raw scores.

Run on a compute node inside the panda environment from the repository root."""
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from proofreader_evolve.cli import precompute_error_scores as pc
from proofreader_evolve.harness import native_pool, label_availability, fixed_pool_scoring as scoring
from proofreader_evolve.harness.fixed_pool_scoring import labeled_order

OUT = Path(__file__).resolve().parent
RUNS = REPO / 'proofreader_evolve/runs'
RUN_IDS = ['precision_20261026_210812_yqeacq5s', 'precision_20261027_000457_5d0_pbkb']
BRAINS = ['789202', '794493', '794491', '750318', '751473']
TIMEOUT, MEMORY_MB = 2400, 98304

def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)

selected = pc.resolve_detector_runs(None, None)
banks = {}
for brain in BRAINS:
    log(f'loading {brain}')
    bank = native_pool.ensure_native_tables(brain, native_pool.cache_path(brain, 100), selected, pc.DEFAULT_OUT,
                                            prepare=False, mcl=100)
    banks[brain] = bank
label_availability.apply(banks)
for bank in banks.values():
    bank.tables = {'merge': bank.tables['merge']}
from proofreader_evolve.harness.context_cache import ContextCache
from proofreader_evolve.harness.descriptor_runs import DescriptorRuns
cache = ContextCache(REPO / 'proofreader_evolve/context_cache', expose_band=True)
for b in BRAINS:
    DescriptorRuns.register_context_column(banks[b].tables['merge'], cache.entry(banks[b].tables['merge']))
log('bank_context_available attached from the context cache band')
positives = {b: int(np.nansum(banks[b].tables['merge'].truth)) for b in BRAINS}
log(f'merge positives: {positives}')

budgets = scoring.Budgets({'merge': None, 'split': None}, positive_ratio={'merge': 1.0, 'split': 1.0}, min_positives=20)

def k_variants(P):
    return {'K300': 300, 'K=P': P, 'K=1.5P': int(np.ceil(1.5 * P)), 'K=2P': 2 * P, 'K=3P': 3 * P}

def precision_table(scores, table):
    order = labeled_order(scores, table.truth, table.keys)
    hits = np.cumsum(table.truth[order])
    P = int(np.nansum(table.truth))
    out = {}
    for name, k in k_variants(P).items():
        kk = min(k, len(order))
        out[name] = {'k': kk, 'tp': int(hits[kk - 1]), 'precision': float(hits[kk - 1] / kk)}
    return out

candidates = [('seed', 'artifacts/scorer.py', None, (REPO / 'proofreader_evolve/artifacts/scorer.py').read_text(), None)]
for run in RUN_IDS:
    rd = RUNS / run
    for row in (json.loads(l) for l in (rd / 'ledger.jsonl').open()):
        if row.get('target_kind') != 'merge' or not row.get('validation') or row.get('submission_repeat'):
            continue
        if row.get('descriptors_referenced'):
            continue
        sub = row['submitted_experiment']
        src = (rd / sub.replace('/', '/experiments/', 1) / 'combined_scorer.py').read_text()
        status = 'accepted' if row.get('accepted') else 'rejected'
        merge_cells = {n: c for n, c in row['validation']['cells'].items() if n.endswith('/merge')}
        candidates.append((f'{run[-8:]}/gen{row["generation"]:02d}', sub, status, src,
                           {n: (c['tp'], c['precision']) for n, c in merge_cells.items()}))
log(f'{len(candidates)} candidates to score')

results = []
for label, sub, status, source, ledger_cells in candidates:
    store = None if label == 'seed' else RUNS / ('precision_20261026_210812_yqeacq5s' if label.startswith('yqeacq5s') else 'precision_20261027_000457_5d0_pbkb') / 'model_artifacts'
    state = {}
    started = time.monotonic()
    try:
        report = scoring.evaluate(source, banks, budgets, TIMEOUT, state=state, artifact_store=store, memory_mb=MEMORY_MB)
    except Exception as exc:  # noqa: BLE001
        log(f'{label}: FAILED {type(exc).__name__}: {str(exc)[:200]}')
        results.append({'label': label, 'submitted': sub, 'status': status, 'error': f'{type(exc).__name__}: {exc}'})
        continue
    per_brain = {b: precision_table(state[f'{b}/merge']['scores'], banks[b].tables['merge']) for b in BRAINS}
    # consistency with the ledger at K = 2P (the run's rule; cap 300 never binds on these brains)
    check = {b: (per_brain[b]['K=2P']['tp'], ledger_cells[f'{b}/merge'][0]) for b in BRAINS} if ledger_cells else None
    results.append({'label': label, 'submitted': sub, 'status': status, 'per_brain': per_brain, 'ledger_check_tp': check,
                    'wall_seconds': time.monotonic() - started})
    np.savez(OUT / f'val_scores_{label.replace("/", "_")}.npz', **{f'{b}/merge': state[f'{b}/merge']['scores'] for b in BRAINS})
    means = {name: float(np.mean([per_brain[b][name]['precision'] for b in BRAINS])) for name in k_variants(1)}
    hits = {name: int(sum(per_brain[b][name]['tp'] for b in BRAINS)) for name in k_variants(1)}
    log(f'{label} {status or ""}: ' + ' '.join(f'{n}={means[n]:.4f}({hits[n]})' for n in means)
        + (f'  ledger_check={check}' if check else '') + f'  [{time.monotonic() - started:.0f}s]')
    (OUT / 'validation_k_variants.json').write_text(json.dumps({'positives': positives, 'results': results}, indent=1))
log('done')
