"""Paired standardised effect (delta / paired-bootstrap SE) of rank statistics (2026-11-02).

Synthetic pools shaped like the five validation merge cells (positives 144/140/74/125/41 at
the 789202 labeled prevalence 74/5855), the 802449 OOF selection pool and the split
validation pools; Gaussian scores at AUROC 0.8 (0.9 for split). Four changes: a broad weak
new feature, a re-ranking confined to the top 3K, to ranks 3K..10K, to ranks below 10K
(re-ranking inside a block reuses the block's score values, so nothing crosses its
boundary), and label-independent jitter. The evidence behind the AUROC default objective
(WORKFLOW_REVISION 2026-11-02). Run: python notebooks/metric_signal_sim.py
The paired bootstrap keeps each pool's K at its original positive count, as the production
bootstrap does (until the 2026-11-02 review it recomputed K from the resampled positives;
the z values moved by at most 0.4 and the ordering of the statistics did not change).
"""
import numpy as np
rng = np.random.default_rng(1)

def make(KS, prev, auc_target):
    mu = np.sqrt(2) * {0.8: 0.8416, 0.9: 1.2816}[auc_target]
    cells = []
    for k in KS:
        n_neg = int(round(k / prev)) - k
        y = np.r_[np.ones(k), np.zeros(n_neg)].astype(int)
        cells.append((y, rng.normal(0, 1, y.size) + mu * y))
    return cells

def metrics(y, s, k=None):
    o = np.argsort(-s, kind='stable'); y = y[o]
    npos = int(y.sum()); nneg = y.size - npos; k = npos if k is None else k
    cum = np.cumsum(y); idx = np.flatnonzero(y); prec = cum[idx] / (idx + 1)
    ranks = np.arange(1, y.size + 1)
    fpr = np.cumsum(1 - y) / nneg; tpr = cum / npos
    def pauc(f):
        m = fpr <= f
        return float(np.trapezoid(np.r_[0, tpr[m]], np.r_[0, fpr[m]]) / f) if m.any() else 0.
    return dict(npos=npos, k=k, tp=y[:k].sum(),
        pk=y[:k].mean(),
        p3=np.mean([y[:int(np.ceil(k/2))].mean(), y[:k].mean(), y[:2*k].mean()]),
        ap=float(prec.mean()),
        ap2k=float(prec[idx < 2*k].sum() / npos),
        ap4k=float(prec[idx < 4*k].sum() / npos),
        pauc02=pauc(.02), pauc10=pauc(.10),
        auc=(np.sum(ranks[y == 0]) - nneg*(nneg+1)/2) / (npos*nneg))

NAMES = ('pk', 'p3', 'ap2k', 'ap4k', 'ap', 'pauc02', 'pauc10', 'auc')
def pooled(ys, ss, ks=None):
    acc = {m: 0. for m in NAMES}; tp = kk = w = 0
    for y, s, k in zip(ys, ss, ks or [None] * len(ys)):
        r = metrics(y, s, k); tp += r['tp']; kk += r['k']; w += r['npos']
        for m in NAMES: acc[m] += r[m] * r['npos']
    out = {m: acc[m] / w for m in NAMES}; out['pk'] = tp / kk
    return out

def paired_boot(cells, s0, s1, B=300):
    diffs, ks = [], [int(y.sum()) for y, _ in cells]  # K fixed per pool, as in production
    for _ in range(B):
        ys, a0, a1 = [], [], []
        for (y, _), x0, x1 in zip(cells, s0, s1):
            idx = rng.integers(0, y.size, y.size)
            ys.append(y[idx]); a0.append(x0[idx]); a1.append(x1[idx])
        m0, m1 = pooled(ys, a0, ks), pooled(ys, a1, ks)
        diffs.append({m: m1[m] - m0[m] for m in NAMES})
    return {m: np.std([d[m] for d in diffs]) for m in NAMES}

def rerank_block(y, s, lo, hi, a):
    """Re-order rows whose rank is in (lo, hi] by s + a*y, reusing the block's score values."""
    o = np.argsort(-s, kind='stable'); r = np.empty(s.size, int); r[o] = np.arange(1, s.size + 1)
    block = np.flatnonzero((r > lo) & (r <= hi))
    vals = np.sort(s[block])[::-1]
    order = block[np.argsort(-(s[block] + a * y[block]), kind='stable')]
    s2 = s.copy(); s2[order] = vals
    return s2

def variant(cells, kind, a):
    out = []
    for y, s in cells:
        k = int(y.sum())
        if kind == 'broad':
            out.append(s + a * (rng.normal(0, 1, s.size) + 0.8 * y))
        elif kind == 'head':          # inside top 3K only
            out.append(rerank_block(y, s, 0, 3*k, a))
        elif kind == 'mid':           # ranks 3K..10K only
            out.append(rerank_block(y, s, 3*k, 10*k, a))
        elif kind == 'tail':          # below 10K only
            out.append(rerank_block(y, s, 10*k, s.size, a))
        elif kind == 'degrade':       # label-independent jitter
            out.append(s + a * rng.normal(0, 1, s.size))
    return out

def run(label, KS, prev, auc_target):
    cells = make(KS, prev, auc_target); base = [s for _, s in cells]
    b = pooled([y for y, _ in cells], base)
    print(f'\n=== {label}: positives {sum(KS)}, prevalence {prev:.3f}; baseline ' +
          ' '.join(f'{m}={b[m]:.3f}' for m in NAMES))
    print('variant     ' + ''.join(f'{m:>9s}' for m in NAMES) + '   (z = delta / paired bootstrap SE)')
    for kind, a in (('broad', .3), ('head', .5), ('mid', .5), ('tail', .5), ('degrade', .3)):
        s1 = variant(cells, kind, a); p = pooled([y for y, _ in cells], s1)
        se = paired_boot(cells, base, s1)
        z = {m: (p[m] - b[m]) / se[m] if se[m] > 0 else 0. for m in NAMES}
        print(f'{kind:8s} a={a:<3}' + ''.join(f'{z[m]:+9.1f}' for m in NAMES))

run('merge validation', [144, 140, 74, 125, 41], 74/5855, 0.8)
run('merge selection 802449 OOF', [1038], 1038/27934, 0.8)
run('split validation', [1500, 5063, 2783, 1800, 1435], 2783/4975, 0.9)
