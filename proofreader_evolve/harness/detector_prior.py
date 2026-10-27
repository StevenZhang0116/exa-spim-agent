"""Detector-prior ablations: run evolution without the frozen detector score, or without
the detector's discovered feature columns as well.

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

``--detector-prior raw`` (2026-10-27, ``plans/raw_prior_ablation.md``) withholds the
discovered feature columns too: every ``h<digits>_*`` column (the AutoDiscovery features the
detector build computed) is dropped with the score, and the feature frame is rebuilt from
the label-blind candidate attributes of ``candidate_rules`` (``RAW_COLUMNS``). The agent
then starts from the pool definition, those columns, the context cache and its own
descriptors. Coordinates and identifiers are never exposed. Everything the no-score mode
withholds stays withheld, and ``check_source`` also refuses ``h<digits>_`` names.

The host keeps the frozen score on ``table.host_detector_score`` (and, under ``raw``, the
dropped frame on ``table.host_detector_features``) for diagnostics only.
"""
import math
import re

import numpy as np
import pandas as pd

from .label_availability import labeled_rows
from .scorer_components import compose

MODES = ('frozen', 'no-score', 'raw')
SCORE_COLUMN = 'detector_score'
BAND_COLUMN = 'bank_context_available'
HIDDEN_NAMES = (SCORE_COLUMN, BAND_COLUMN, 'row_in_band')
NO_SCORE_DESCRIPTOR_SCOPES = ('all_cached',)
RAW_PREFIX = 'raw_'
DETECTOR_FEATURE_NAME = re.compile(r'\bh\d+_[A-Za-z0-9_]+')
RAW_COLUMNS = {
    'merge': ('raw_degree', 'raw_candidate_family', 'raw_tip_gap_um'),
    'split': ('raw_gap_um', 'raw_partner_rank', 'raw_anchor_side', 'raw_node_role_a', 'raw_node_role_b',
              'raw_occurrence_count', 'raw_occurrence_gap_min_um', 'raw_occurrence_gap_max_um'),
}
RAW_COLUMN_RULES = {
    'raw_degree': 'junction arity (candidate node degree)',
    'raw_candidate_family': 'rule that produced the row: junction = 0, tip_to_shaft = 1',
    'raw_tip_gap_um': 'tip-to-shaft gap in um; NaN on junction rows (no tip)',
    'raw_gap_um': 'anchor pair gap in um',
    'raw_partner_rank': 'rank of the partner among the anchor tip\'s neighbours (1 = nearest)',
    'raw_anchor_side': 'which side holds the anchor tip: a = 0, b = 1',
    'raw_node_role_a': 'node a is a tip (1) or not (0)',
    'raw_node_role_b': 'node b is a tip (1) or not (0)',
    'raw_occurrence_count': 'how many enumerations produced this pair',
    'raw_occurrence_gap_min_um': 'smallest gap over the pair\'s occurrences',
    'raw_occurrence_gap_max_um': 'largest gap over the pair\'s occurrences',
}
_FAMILY = {'junction': 0., 'tip_to_shaft': 1.}
_ROLE = {'tip': 1., 'non_tip': 0.}
_SIDE = {'a': 0., 'b': 1.}

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

NOTICE_RAW = """

## Raw-candidate ablation (this run)

This run measures evolution WITHOUT the frozen detector score AND WITHOUT the detector's
discovered feature columns. `detector_score`, `bank_context_available`, `row_in_band` and
every `h<number>_...` column do not exist: ignore every mention of them in the guides. The
only base predictor columns are label-blind candidate attributes:

- merge: `raw_degree` (junction arity), `raw_candidate_family` (junction = 0,
  tip_to_shaft = 1), `raw_tip_gap_um` (NaN on junction rows);
- split: `raw_gap_um`, `raw_partner_rank`, `raw_anchor_side` (a = 0, b = 1),
  `raw_node_role_a` / `raw_node_role_b` (tip = 1), `raw_occurrence_count`,
  `raw_occurrence_gap_min_um`, `raw_occurrence_gap_max_um`.

Every other feature must be built by you with `compute_descriptors` from the cached
fragment neighbourhoods and image patches (scope `{"scope": "all_cached"}` only), or
measured with `run_volume_analysis`. Local and image specs must set `selection_feature`
to one of the raw columns. The starting scorer is a constant: rank from the raw columns,
your descriptors, or a TRAIN-fitted model.
"""


def mode(args):
    return getattr(args, 'detector_prior', 'frozen')


def enabled(args):
    """True for both ablations (score withheld, band hidden, all_cached descriptors only)."""
    return mode(args) in ('no-score', 'raw')


def raw_enabled(args):
    return mode(args) == 'raw'


def notice(args):
    return NOTICE_RAW if raw_enabled(args) else NOTICE if enabled(args) else ''


def is_detector_feature(name):
    """Discovered-feature columns of the detector build: h<digits>_<name>."""
    return re.fullmatch(r'h\d+_[A-Za-z0-9_]+', str(name)) is not None


def _tables(banks):
    seen = set()
    for brain, bank in banks.items():
        for kind, table in bank.tables.items():
            if id(table) not in seen:
                seen.add(id(table))
                yield brain, kind, table


def _number(value):
    if value is None:
        return math.nan
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Raw candidate attributes must be finite or None')
    return value


def _code(value, table, what):
    try:
        return table[value]
    except KeyError:
        raise ValueError(f'Unknown {what} {value!r}; expected one of {sorted(table)}') from None


def raw_columns(kind, rows):
    """The raw predictor frame of one pool from its label-blind candidate rows."""
    if kind == 'merge':
        data = {'raw_degree': [float(int(r['degree'])) for r in rows],
                'raw_candidate_family': [_code(r['candidate_family'], _FAMILY, 'candidate_family') for r in rows],
                'raw_tip_gap_um': [_number(r.get('tip_gap_um')) for r in rows]}
    elif kind == 'split':
        occurrences = [tuple(r.get('occurrences') or ()) for r in rows]
        gaps = [[_number(o['gap_um']) for o in occ] for occ in occurrences]
        data = {'raw_gap_um': [_number(r['gap_um']) for r in rows],
                'raw_partner_rank': [float(int(r['partner_rank'])) for r in rows],
                'raw_anchor_side': [_code(r['anchor_side'], _SIDE, 'anchor_side') for r in rows],
                'raw_node_role_a': [_code(r['node_role_a'], _ROLE, 'node_role_a') for r in rows],
                'raw_node_role_b': [_code(r['node_role_b'], _ROLE, 'node_role_b') for r in rows],
                'raw_occurrence_count': [float(len(occ)) for occ in occurrences],
                'raw_occurrence_gap_min_um': [min(g) if g else math.nan for g in gaps],
                'raw_occurrence_gap_max_um': [max(g) if g else math.nan for g in gaps]}
    else:
        raise ValueError(f'No raw columns are defined for kind {kind!r}')
    frame = pd.DataFrame({name: np.asarray(data[name], dtype=float) for name in RAW_COLUMNS[kind]})
    if len(frame) != len(rows):
        raise ValueError('Raw columns must align with the candidate rows')
    return frame


def strip(banks):
    """Drop the frozen score from every table; returns the cells it was removed from."""
    cells = []
    for brain, kind, table in _tables(banks):
        if SCORE_COLUMN in table.features.columns:
            table.host_detector_score = table.features[SCORE_COLUMN].to_numpy(dtype=float, copy=True)
            table.features = table.features.drop(columns=[SCORE_COLUMN])
            cells.append(f'{brain}/{kind}')
    return cells


def strip_to_raw(banks):
    """Replace every table's feature frame by its raw columns; returns the cells changed.

    The score and the dropped frame stay on the table as host-only copies. A table whose
    frame already holds raw columns only is left alone, so shared tables convert once.
    """
    cells = []
    for brain, kind, table in _tables(banks):
        columns = list(table.features.columns)
        if columns and all(str(c).startswith(RAW_PREFIX) for c in columns):
            continue
        if SCORE_COLUMN in columns:
            table.host_detector_score = table.features[SCORE_COLUMN].to_numpy(dtype=float, copy=True)
        table.host_detector_features = table.features
        table.features = raw_columns(kind, table.candidates)
        cells.append(f'{brain}/{kind}')
    return cells


def check_source(text, what, *, raw=False):
    """Refuse a source that reads the withheld columns (and, under raw, discovered features)."""
    found = {m.group(0) for m in re.finditer(r'\b(?:%s)\b' % '|'.join(HIDDEN_NAMES), text or '')}
    if raw:
        found |= {m.group(0) for m in DETECTOR_FEATURE_NAME.finditer(text or '')}
    if found:
        raise ValueError(f'{what} references {", ".join(sorted(found))}, which --detector-prior '
                         f'{"raw" if raw else "no-score"} withholds')


def raw_heuristic_source():
    """Host-only comparators on the raw columns: split by the smallest gap, merge by junction
    arity with the tip gap as a tie-breaker. Never the seed, never shown to the agent."""
    return compose({
        'merge': ('import numpy as np\n\n\n'
                  'def score_candidates(features, ctx):\n'
                  '    degree = features["raw_degree"].fillna(0.).to_numpy(dtype=float)\n'
                  '    gap = features["raw_tip_gap_um"].fillna(0.).to_numpy(dtype=float)\n'
                  '    return degree - gap / 1000.\n'),
        'split': ('import numpy as np\n\n\n'
                  'def score_candidates(features, ctx):\n'
                  '    return -features["raw_gap_um"].fillna(1e6).to_numpy(dtype=float)\n')})


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
        raise ValueError('--detector-prior no-score/raw with --context-cache needs every GT-labeled row cached, or '
                         'descriptor coverage would mark the detector band: ' + '; '.join(gaps)
                         + '. Build the labeled extension first (compute node, one brain at a time): '
                         f'python -u -m proofreader_evolve.cli.precompute_context_cache --brains {brains} '
                         '--labeled-extension; or run without --context-cache')


def describe(args, cells=None):
    if not enabled(args):
        return {'mode': 'frozen', 'scope': 'detector_score is a predictor column and the default seed'}
    if raw_enabled(args):
        return {'mode': 'raw', 'withheld': [*HIDDEN_NAMES, 'h<digits>_* (discovered feature columns)'],
                'stripped_cells': cells or [], 'raw_columns': {k: list(v) for k, v in RAW_COLUMNS.items()},
                'raw_column_rules': dict(RAW_COLUMN_RULES),
                'descriptor_scopes': list(NO_SCORE_DESCRIPTOR_SCOPES),
                'baseline': 'Frozen detector baseline measured on the unstripped tables and raw-heuristic '
                            'baselines (split -gap, merge degree with the tip gap as tie-breaker) measured on '
                            'the raw columns (host-only comparators)',
                'scope': 'Ablation: same pools, labels, K, brains and budgets; no frozen detector score and no '
                         'discovered feature columns; predictors are the raw candidate attributes, the agent\'s '
                         'descriptors and TRAIN-fitted models'}
    return {'mode': 'no-score', 'withheld': list(HIDDEN_NAMES), 'stripped_cells': cells or [],
            'descriptor_scopes': list(NO_SCORE_DESCRIPTOR_SCOPES),
            'baseline': 'Frozen detector baseline measured on the unstripped tables (host-only comparator)',
            'scope': 'Ablation: same pools, detector feature columns, labels, K, brains and budgets; '
                     'no frozen detector score'}
