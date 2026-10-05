"""Paired, host-measured feature removal with unrestricted isolated agent models."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .classifier_contract import config_identity, frozen_model
from .classifier_training import fit_classifier
from .fixed_pool_scoring import rank_metrics
from .internal_validation import FOLDS, fold_assignments, grouped_folds
from .isolated_scoring import score
from .local_feature_contract import local_feature_spec
from .local_features import augmented_features
from .model_execution import predict_model
from .image_contract import image_spec
from .image_features import subset_features
from .image_selection import scoring_coverage


ABLATION_VERSION = 'paired-train-feature-ablation-v2'
# One evaluation unit per arm: a fold triple is one measurement, like a candidate's fold fits.
CLASSIFIER_UNITS = 2
FORMULA_UNITS = 2


def prepare_ablation(train, kind, source, columns, *, classifier, config,
                     timeout, memory_mb, partitions=None, selection_brains=None):
    frames = {brain: augmented_features(source, bank.tables[kind], timeout, memory_mb)
              for brain, bank in train.items()}
    if not columns:
        contract = local_feature_spec(source)
        image_contract = image_spec(source)
        columns = ([*contract[1]['feature_names'], 'local_context_available'] if contract else [])
        if image_contract:
            columns += [*image_contract[1]['feature_names'], 'image_available']
        if not columns:
            raise ValueError('Declare research.feature_columns, LOCAL_CONTEXT or LOCAL_IMAGE for a feature ablation')
    for brain, frame in frames.items():
        absent = set(columns) - set(frame.columns)
        if absent:
            raise ValueError(f'Unknown ablation input columns on TRAIN {brain}: {sorted(absent)}')
    split = ((partitions or grouped_folds(train, kind, selection_brains=selection_brains))
             if classifier else (None, None))
    fingerprints = {}
    for brain, frame in frames.items():
        fingerprint = hashlib.sha256()
        for start in range(0, len(frame), 65536):
            fingerprint.update(frame.iloc[start:start + 65536].to_numpy(dtype='<f8').tobytes())
        table = train[brain].tables[kind]
        fingerprints[brain] = {'pool': table.meta['pool_sha256'], 'columns': list(frame.columns),
                              'features': fingerprint.hexdigest(),
                              'labels': hashlib.sha256(np.asarray(table.truth, dtype=np.int8).tobytes()).hexdigest()}
        if 'image_context' in frame.attrs:
            fingerprints[brain]['image_inputs'] = frame.attrs['image_context']['feature_cache_key']
    identity = {'version': ABLATION_VERSION, 'kind': kind, 'classifier': classifier,
                'program_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'parameters_sha256': config_identity(config), 'feature_columns': sorted(columns),
                'split': split[1], 'tables': fingerprints,
                'implementation': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('feature_ablation.py', 'internal_validation.py', 'classifier_training.py',
                                 'classifier_train_worker.py', 'classifier_contract.py', 'model_execution.py',
                                 'model_sandbox.py', 'local_features.py', 'local_feature_contract.py',
                                 'image_context.py', 'image_features.py', 'image_contract.py', 'image_coordinates.py',
                                 'image_selection.py',
                                 'isolated_scoring.py', 'scorer_worker.py', 'fixed_pool_scoring.py')},
                'masking': 'constant_zero_in_fit_and_predict', 'fold_seeds': list(range(FOLDS)) if classifier else []}
    return frames, list(columns), split, identity


def measure_ablation(train, kind, source, config, frames, columns, split, identity,
                     budgets, directory, *, classifier, charge, fit_timeout=300,
                     score_timeout=120, memory_mb=8192, threads=1):
    """The identical program is refit for both arms; removed inputs become constants.

    Constant replacement preserves the input schema for arbitrary agent programs.
    It removes values and missingness; removing image_available also withholds
    raw patches. Both classifier arms refit after input removal.
    Diagnostic models cannot be submitted through the normal candidate registry.
    """
    started = time.monotonic()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'program.py').write_text(source)
    (directory / 'identity.json').write_text(json.dumps(identity, indent=2, allow_nan=False))
    predictions, timings = {}, {}
    for arm in ('full', 'without_features'):
        arm_frames = frames if arm == 'full' else {brain: frame.copy() for brain, frame in frames.items()}
        if arm == 'without_features':
            for frame in arm_frames.values():
                frame.loc[:, columns] = 0.
                if 'image_available' in columns and '_image_rows' in frame.attrs:
                    frame.attrs['_image_rows'] = {}
        arm_started = time.monotonic()
        if classifier:
            folds, _ = split
            # Only selection brains receive out-of-fold predictions; auxiliary
            # brains are fitting-only and have no honest score to rank.
            scored = sorted({brain for _, held in folds for brain, rows in held.items() if len(rows)})
            predicted = {brain: np.full(len(frames[brain]), np.nan) for brain in scored}
            charge()
            for fold, (fitting, held) in enumerate(folds):
                fold_dir = directory / arm / f'fold{fold}'
                # Keep internal models out of the resumable/final artifact store.
                store = fold_dir / 'models'
                component, _ = fit_classifier(train, kind, config, fold_dir, program=source,
                    artifact_store=store, timeout=fit_timeout, memory_mb=memory_mb, threads=threads,
                    feature_frames=arm_frames, fit_rows=fitting, random_seed=fold)
                model = frozen_model(component)
                for brain, indices in held.items():
                    if brain in predicted and len(indices):
                        predicted[brain][indices] = predict_model(model, store,
                            subset_features(arm_frames[brain], indices), kind,
                            timeout=score_timeout, memory_mb=memory_mb)
        else:
            charge()
            predicted = {brain: score(source, frame, kind, score_timeout) for brain, frame in arm_frames.items()}
        if not predicted or any(not np.isfinite(values).all() for values in predicted.values()):
            raise ValueError('Ablation must score every scored TRAIN candidate exactly once per arm')
        predictions[arm] = predicted
        timings[arm] = time.monotonic() - arm_started
    cells = {}
    for brain in sorted(predictions['full']):
        table = train[brain].tables[kind]
        if classifier:
            from .selection_protocol import partitioned_rank_metrics
            ids = fold_assignments(split[0], brain, len(table.truth))
            full, chosen = partitioned_rank_metrics(predictions['full'][brain], table.truth, table.keys, ids, budgets[kind])
            removed, control = partitioned_rank_metrics(predictions['without_features'][brain], table.truth,
                                                       table.keys, ids, budgets[kind])
        else:
            full, chosen = rank_metrics(predictions['full'][brain], table.truth, table.keys, budgets[kind])
            removed, control = rank_metrics(predictions['without_features'][brain], table.truth, table.keys, budgets[kind])
        gained = np.setdiff1d(chosen, control)
        lost = np.setdiff1d(control, chosen)
        cells[f'{brain}/{kind}'] = {
            'full': full, 'without_features': removed,
            'delta_precision': full['precision'] - removed['precision'],
            'gained_positives': int(table.truth[gained].sum()),
            'lost_positives': int(table.truth[lost].sum()),
            'top_k_overlap': len(np.intersect1d(chosen, control))}
        if 'image_context' in frames[brain].attrs:
            cells[f'{brain}/{kind}']['image_coverage'] = {
                'selection': frames[brain].attrs['image_context'].get('selection_coverage'),
                'full': scoring_coverage(frames[brain], chosen, table.truth, budgets[kind])}
    delta = sum(cell['delta_precision'] for cell in cells.values()) / len(cells)
    arrays = {f'{arm}:{brain}': values for arm, brains in predictions.items() for brain, values in brains.items()}
    np.savez(directory / 'predictions.npz', **arrays)
    image_contract = image_spec(source)
    image_columns = ([*image_contract[1]['feature_names'], 'image_available'] if image_contract else [])
    removed_images = bool(image_columns) and set(image_columns) <= set(columns)
    # Mixing image removal with other inputs measures their joint contribution.
    image_only = removed_images and set(columns) == set(image_columns)
    return {'version': ABLATION_VERSION, 'status': 'measured', 'target_kind': kind,
            'protocol': 'TRAIN_grouped_out_of_fold_partitioned_k' if classifier else 'TRAIN_formula_descriptive',
            'scored_brains': sorted(predictions['full']),
            'program_sha256': identity['program_sha256'], 'parameters_sha256': identity['parameters_sha256'],
            'feature_columns': columns, 'signature': config_identity(identity),
            'image_evidence': {'image_inputs_present': bool(image_columns),
                'all_image_inputs_removed_in_control': removed_images,
                'image_only_control': image_only,
                'conclusion': ('positive_image_increment' if delta > 1e-12 else
                               'negative_image_increment' if delta < -1e-12 else 'no_image_increment')
                              if image_only else 'image_increment_not_isolated',
                'scope': 'Exact program, parameters, image coverage and TRAIN diagnostic only'},
            'removal_method': 'Set declared input columns to zero in both fit and predict; schema preserved',
            'raw_image_removal': 'image_available in the removed group makes all raw patch lists empty',
            'split': split[1], 'cells': cells, 'delta_precision': delta,
            'conclusion': ('positive_increment_on_this_internal_measurement' if delta > 1e-12 else
                           'negative_increment_on_this_internal_measurement' if delta < -1e-12 else
                           'no_precision_increment_on_this_internal_measurement'),
            'scope': ('Paired TRAIN-only diagnosis; the same program/parameters and fold seeds are used in both arms. '
                      'Agent overrides and non-NumPy RNGs can still affect fit reproducibility. '
                      'Out-of-fold scores are ranked inside each fold with budgets summing to K, never pooled. '
                      'Auxiliary TRAIN brains are fitting-only and are not scored. '
                      'Frozen base predictors are reused; under grouped_oof the scored selection brains are '
                      'detector-naive while auxiliary fitting rows may come from the detector\'s own brain. '
                      'Adaptively inspected TRAIN folds are not an independent final test; formula results are descriptive. '
                      'No outer validation brain is accessed; this result is not a promotable scorer.'),
            'arm_seconds': timings, 'wall_seconds': time.monotonic() - started,
            'required_evaluations': CLASSIFIER_UNITS if classifier else FORMULA_UNITS}
