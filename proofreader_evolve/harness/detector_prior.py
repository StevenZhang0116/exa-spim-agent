"""Detector-prior ablation: run evolution without the frozen detector score.

``--detector-prior no-score`` asks how much of the final validation precision comes
from the frozen detector model. The candidate pools, the detector's feature columns
(computed quantities, not fitted), labels, K, brains and budgets stay as in a normal
run; only the frozen ranking is withheld from everything the reviser, its scorers,
training programs and descriptors can see:

- ``detector_score`` is dropped from every loaded ``table.features`` (in memory; table
  files and digests are unchanged) after the host has measured the frozen baseline;
- ``bank_context_available`` (1 inside the detector band) is not registered and
  descriptor contexts omit ``row_in_band``;
- descriptor runs use the ``all_cached`` scope only (``pilot`` and ``boundary`` are
  detector-rank windows), and every context-cache entry must cover all GT-labeled rows,
  otherwise "descriptor is finite" would mark the detector band among ranked rows;
- local and image row selection has no default column (the default is the score).

The host keeps the frozen score on ``table.host_detector_score`` for diagnostics only.
"""
import re

from .label_availability import labeled_rows

MODES = ('frozen', 'no-score')
SCORE_COLUMN = 'detector_score'
BAND_COLUMN = 'bank_context_available'
HIDDEN_NAMES = (SCORE_COLUMN, BAND_COLUMN, 'row_in_band')
NO_SCORE_DESCRIPTOR_SCOPES = ('all_cached',)

NOTICE = """

## Detector-prior ablation (this run)

This run measures evolution WITHOUT the frozen detector score. `detector_score` is not
a column of `features`, of descriptor contexts or of any TRAIN data, and
`bank_context_available` / `row_in_band` do not exist. Ignore every mention of them in
the guides. All other predictor columns (the detector's computed features) remain.
Local and image specs must set `selection_feature` to one of the remaining base
predictor columns. Descriptor runs accept only `{"scope": "all_cached"}`. The starting
scorer is a constant: rank from the remaining columns, your descriptors, or a
TRAIN-fitted model.
"""


def enabled(args):
    return getattr(args, 'detector_prior', 'frozen') == 'no-score'


def _tables(banks):
    seen = set()
    for brain, bank in banks.items():
        for kind, table in bank.tables.items():
            if id(table) not in seen:
                seen.add(id(table))
                yield brain, kind, table


def strip(banks):
    """Drop the frozen score from every table; returns the cells it was removed from."""
    cells = []
    for brain, kind, table in _tables(banks):
        if SCORE_COLUMN in table.features.columns:
            table.host_detector_score = table.features[SCORE_COLUMN].to_numpy(dtype=float, copy=True)
            table.features = table.features.drop(columns=[SCORE_COLUMN])
            cells.append(f'{brain}/{kind}')
    return cells


def check_source(text, what):
    """Refuse a starting source that reads the withheld columns."""
    found = sorted({m.group(0) for m in re.finditer(r'\b(?:%s)\b' % '|'.join(HIDDEN_NAMES), text or '')})
    if found:
        raise ValueError(f'{what} references {", ".join(found)}, which --detector-prior no-score withholds')


def check_context_coverage(cache, banks, kinds):
    """Every entry must hold all GT-labeled rows (band plus labeled extension)."""
    gaps = []
    for brain, kind, table in _tables(banks):
        if kind not in kinds:
            continue
        rows = labeled_rows(table.truth)
        missing = int((~cache.entry(table).has_rows(rows)).sum()) if len(rows) else 0
        if missing:
            gaps.append(f'{brain}/{kind} ({missing} of {len(rows)} labeled rows uncached)')
    if gaps:
        brains = ' '.join(sorted({gap.split('/')[0] for gap in gaps}))
        raise ValueError('--detector-prior no-score with --context-cache needs every GT-labeled row cached, or '
                         'descriptor coverage would mark the detector band: ' + '; '.join(gaps)
                         + '. Build the labeled extension first (compute node, one brain at a time): '
                         f'python -u -m proofreader_evolve.cli.precompute_context_cache --brains {brains} '
                         '--labeled-extension; or run without --context-cache')


def describe(args, cells=None):
    if not enabled(args):
        return {'mode': 'frozen', 'scope': 'detector_score is a predictor column and the default seed'}
    return {'mode': 'no-score', 'withheld': list(HIDDEN_NAMES), 'stripped_cells': cells or [],
            'descriptor_scopes': list(NO_SCORE_DESCRIPTOR_SCOPES),
            'baseline': 'Frozen detector baseline measured on the unstripped tables (host-only comparator)',
            'scope': 'Ablation: same pools, detector feature columns, labels, K, brains and budgets; '
                     'no frozen detector score'}
