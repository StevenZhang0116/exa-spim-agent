"""Official TRAIN selection score for exploration branches.

``grouped_oof``: candidates are ranked by grouped out-of-fold precision at a
partitioned K on selection brains that the frozen detectors were not fitted on.
Auxiliary TRAIN brains only supply fitting rows and are never held out.
``in_sample``: the historical protocol (full-pool in-sample TRAIN ranking),
retained so attribution runs can compare the two selection signals.

Neither protocol reads validation brains or changes the outer promotion gate.
The selection score is queried repeatedly by the reviser, so it is a development
signal, not an unbiased estimate of generalization.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .classifier_contract import frozen_model
from .classifier_training import fit_classifier
from .fixed_pool_scoring import aggregate_precision, rank_metrics
from .image_features import subset_features
from .internal_validation import FOLDS, fold_assignments, grouped_folds
from .local_features import augmented_features
from .model_execution import predict_model


PROTOCOL_VERSION = 'grouped-cv-precision-v1'
MODES = ('grouped_oof', 'in_sample')


def fold_budgets(held_counts, k):
    """Largest-remainder split of K across folds, capped by each fold's held rows."""
    counts = [int(c) for c in held_counts]
    if any(c < 0 for c in counts) or type(k) is not int or k < 1:
        raise ValueError('Fold sizes must be nonnegative and K a positive integer')
    total = sum(counts)
    if total == 0:
        return [0] * len(counts)
    k = min(k, total)
    shares = [k * c / total for c in counts]
    budgets = [min(math.floor(s), c) for s, c in zip(shares, counts)]
    remaining = k - sum(budgets)
    order = sorted(range(len(counts)), key=lambda i: (-(shares[i] - math.floor(shares[i])), i))
    while remaining > 0:
        progressed = False
        for index in order:
            if remaining == 0:
                break
            if budgets[index] < counts[index]:
                budgets[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return budgets


def partitioned_rank_metrics(scores, truth, keys, fold_ids, k):
    """Rank inside each fold with its own budget; fold budgets sum to K.

    Fold models can emit differently scaled scores, so out-of-fold scores are
    never pooled into one global ranking. The same partition and budgets apply
    to every candidate and to the reference, including non-fitted formulas.
    """
    scores, truth = np.asarray(scores, dtype=float), np.asarray(truth)
    fold_ids = np.asarray(fold_ids)
    if type(k) is not int or k < 1:
        raise ValueError('K must be a positive integer')
    if (scores.shape != truth.shape or scores.ndim != 1 or len(keys) != len(scores) or not len(scores)
            or len(set(keys)) != len(keys) or not np.isfinite(scores).all() or not np.isin(truth, [0, 1]).all()):
        raise ValueError('Return one finite score per candidate in original row order; no dropping/abstention')
    if fold_ids.shape != scores.shape or not np.issubdtype(fold_ids.dtype, np.integer) or (fold_ids < 0).any():
        raise ValueError('Every selection row needs exactly one nonnegative fold assignment')
    n_folds = int(fold_ids.max()) + 1
    held_counts = [int(np.count_nonzero(fold_ids == fold)) for fold in range(n_folds)]
    budgets = fold_budgets(held_counts, k)
    keys = np.asarray(keys, dtype=str)
    parts, fold_tp = [], []
    for fold in range(n_folds):
        rows = np.flatnonzero(fold_ids == fold)
        order = rows[np.lexsort((keys[rows], -scores[rows]))]
        picked = order[:budgets[fold]]
        parts.append(picked)
        fold_tp.append(int(truth[picked].sum()))
    chosen = np.concatenate(parts) if parts else np.zeros(0, dtype=int)
    tp, positives, effective = int(sum(fold_tp)), int(truth.sum()), len(chosen)
    return {'requested_k': k, 'effective_k': effective, 'pool_size': len(scores),
            'tp': tp, 'fp': effective - tp, 'precision': tp / effective if effective else 0.,
            'recall': tp / positives if positives else None, 'positives': positives,
            'protocol': PROTOCOL_VERSION, 'fold_budgets': budgets, 'fold_tp': fold_tp,
            'fold_held_rows': held_counts}, chosen


def save_selection(directory, report, state):
    directory = Path(directory)
    (directory / 'selection_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez(directory / 'selection_state.npz', **{f'{cell}:{key}': values[key]
             for cell, values in state.items() for key in ('scores', 'chosen')})


def load_selection(directory):
    directory = Path(directory)
    report = json.loads((directory / 'selection_evaluation.json').read_text())
    with np.load(directory / 'selection_state.npz', allow_pickle=False) as data:
        state = {cell: {key: data[f'{cell}:{key}'] for key in ('scores', 'chosen')} for cell in report['cells']}
    return report, state


class SelectionProtocol:
    def __init__(self, train, budgets, *, kinds=None, selection_brains=None, mode='grouped_oof'):
        if mode not in MODES:
            raise ValueError(f'Unknown selection protocol {mode!r}; use one of {MODES}')
        if not train:
            raise ValueError('The selection protocol needs at least one TRAIN brain')
        self.train, self.budgets, self.mode = train, dict(budgets), mode
        available = set.intersection(*(set(bank.tables) for bank in train.values()))
        self.kinds = [k for k in (kinds or sorted(available)) if k in available and k in self.budgets]
        if mode == 'in_sample':
            selection = sorted(train)
        else:
            selection = sorted(str(b) for b in (selection_brains or ()))
            if not selection or set(selection) - set(train):
                raise ValueError('grouped_oof selection needs a nonempty subset of TRAIN brains as selection brains')
        self.selection_brains = selection
        self.auxiliary_brains = sorted(set(train) - set(selection))
        self.partitions = {}
        if mode == 'grouped_oof':
            for kind in self.kinds:
                self.partitions[kind] = grouped_folds(train, kind, selection_brains=selection)

    @property
    def requires_fold_fits(self):
        return self.mode == 'grouped_oof'

    @property
    def selection_train(self):
        return {brain: self.train[brain] for brain in self.selection_brains}

    def cells(self, kind):
        return [f'{brain}/{kind}' for brain in self.selection_brains]

    def fold_ids(self, kind, brain):
        size = len(self.train[brain].tables[kind].truth)
        return fold_assignments(self.partitions[kind][0], brain, size)

    def partition_sha256(self, kind):
        return self.partitions[kind][1]['partition_sha256'] if kind in self.partitions else None

    def describe(self, kind=None):
        grouped = self.mode == 'grouped_oof'
        info = {'version': PROTOCOL_VERSION, 'mode': self.mode,
                'selection_brains': list(self.selection_brains),
                'auxiliary_train_brains': list(self.auxiliary_brains),
                'folds': FOLDS if grouped else None,
                'metric': 'grouped_cv_precision' if grouped else 'in_sample_precision_at_k',
                'ranking': ('Per-fold ranking of out-of-fold scores with largest-remainder fold budgets '
                            'that sum to K; TP summed across folds and divided by K.' if grouped else
                            'Full-pool in-sample ranking at K (historical protocol).'),
                'formulas': ('Non-fitted formulas use the same partition and budgets; their result is a '
                             'descriptive diagnostic because their constants were chosen with TRAIN feedback.'
                             if grouped else 'Formulas and models share the same in-sample ranking.'),
                'auxiliary_role': ('Auxiliary brains join every fitting set and are never held out; their '
                                   'in-sample scores are diagnostics only.' if grouped else None),
                'scope': ('Development selection signal for TRAIN branches, used repeatedly by the reviser; '
                          'not an unbiased generalization estimate. The outer equal-weight promotion '
                          'gate is unchanged.')}
        if grouped:
            info['partitions'] = {}
            for name, (partitions, description) in self.partitions.items():
                if kind is not None and name != kind:
                    continue
                budgets = {}
                for brain in self.selection_brains:
                    ids = self.fold_ids(name, brain)
                    counts = [int(np.count_nonzero(ids == fold)) for fold in range(FOLDS)]
                    budgets[f'{brain}/{name}'] = fold_budgets(counts, self.budgets[name])
                info['partitions'][name] = {'partition_sha256': description['partition_sha256'],
                                            'grouping': description['grouping'],
                                            'fold_details': description['fold_details'],
                                            'fold_budgets': budgets}
        return info

    def score_report(self, kind, scores_by_cell, policy_sha256=None):
        """Apply the official ranking to one score vector per selection cell of this kind."""
        report = {'version': PROTOCOL_VERSION, 'mode': self.mode, 'target_kind': kind,
                  'policy_sha256': policy_sha256, 'cells': {}}
        state = {}
        for brain in self.selection_brains:
            cell = f'{brain}/{kind}'
            table = self.train[brain].tables[kind]
            scores = np.asarray(scores_by_cell[cell], dtype=float)
            if self.mode == 'in_sample':
                metrics, chosen = rank_metrics(scores, table.truth, table.keys, self.budgets[kind])
                metrics['protocol'] = 'in_sample'
            else:
                metrics, chosen = partitioned_rank_metrics(scores, table.truth, table.keys,
                                                           self.fold_ids(kind, brain), self.budgets[kind])
                metrics['partition_sha256'] = self.partition_sha256(kind)
            metrics['pool_sha256'] = table.meta['pool_sha256']
            metrics['labels_sha256'] = hashlib.sha256(np.asarray(table.truth, dtype=np.int8).tobytes()).hexdigest()
            report['cells'][cell] = metrics
            state[cell] = {'scores': scores, 'chosen': chosen}
        report.update(aggregate_precision(report['cells']))
        return report, state

    def measure_formula(self, kind, measured_state, policy_sha256=None):
        """A deterministic formula is scored once; the partition only sets the budgets."""
        return self.score_report(kind, {cell: measured_state[cell]['scores'] for cell in self.cells(kind)},
                                 policy_sha256)

    def measure_classifier(self, kind, program, config, directory, *, frames=None, fit_timeout=300,
                           score_timeout=120, memory_mb=8192, threads=1, include_auxiliary=True):
        """Fit the agent program per fold on fitting rows only; predict every held row once.

        Fold models stay under ``directory`` and never enter the resumable artifact
        store. Only out-of-fold scores of selection rows are returned.
        """
        if self.mode != 'grouped_oof':
            raise ValueError('Fold fits are only defined for the grouped_oof protocol')
        partitions, description = self.partitions[kind]
        if frames is None:
            frames = {brain: augmented_features(program, bank.tables[kind], fit_timeout, memory_mb)
                      for brain, bank in self.train.items()}
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        scores = {f'{brain}/{kind}': np.full(len(self.train[brain].tables[kind].truth), np.nan)
                  for brain in self.selection_brains}
        folds = []
        for fold, (fitting, held) in enumerate(partitions):
            fit_rows = dict(fitting)
            if not include_auxiliary:
                for brain in self.auxiliary_brains:
                    fit_rows[brain] = np.zeros(0, dtype=int)
            fold_dir = directory / f'fold{fold}'
            store = fold_dir / 'models'
            component, summary = fit_classifier(self.train, kind, config, fold_dir, program=program,
                                                artifact_store=store, timeout=fit_timeout, memory_mb=memory_mb,
                                                threads=threads, feature_frames=frames, fit_rows=fit_rows,
                                                random_seed=fold)
            model = frozen_model(component)
            for brain in self.selection_brains:
                rows = np.asarray(held[brain], dtype=int)
                if len(rows):
                    scores[f'{brain}/{kind}'][rows] = predict_model(
                        model, store, subset_features(frames[brain], rows), kind,
                        timeout=score_timeout, memory_mb=memory_mb)
            folds.append({'fold': fold, 'training_rows': summary['training_rows'],
                          'class_counts': summary.get('class_counts'), 'wall_seconds': summary.get('wall_seconds'),
                          'held_rows': {brain: int(len(held[brain])) for brain in self.selection_brains}})
        for cell, values in scores.items():
            if not np.isfinite(values).all():
                raise ValueError(f'Out-of-fold scores must cover every selection row of {cell} exactly once')
        report, state = self.score_report(kind, scores)
        report.update(fold_fits=folds, include_auxiliary=include_auxiliary,
                      partition_sha256=description['partition_sha256'],
                      fitting_brains=sorted(set(self.train) if include_auxiliary else set(self.selection_brains)))
        return report, state
