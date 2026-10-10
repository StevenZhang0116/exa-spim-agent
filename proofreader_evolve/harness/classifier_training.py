"""Host-owned TRAIN transport, code snapshots and immutable opaque model artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .classifier_contract import (MODEL_VERSION, normalize_config, render_model, validate_program,
                                  artifact_files, config_identity, verify_artifacts)
from .model_execution import ModelExecutionError, run_worker, DEFAULT_MEMORY_MB
from .local_features import augmented_features


class ClassifierTrainingError(ModelExecutionError):
    pass


# Unlabeled (NaN-label) rows passed to `fit(..., X_unlabeled=...)`, per TRAIN brain:
# a deterministic sample, so million-row pools stay transportable.
UNLABELED_ROWS_PER_BRAIN = 100_000


def declares_parameter(program, name):
    """True when the program's top-level `fit` names `name` as an explicit parameter."""
    import ast
    for node in ast.parse(program).body:
        if isinstance(node, ast.FunctionDef) and node.name == 'fit':
            arguments = node.args
            return name in {a.arg for a in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]}
    return False


def _unlabeled_sample(rows, seed_text):
    if len(rows) <= UNLABELED_ROWS_PER_BRAIN:
        return rows
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], 'big')
    return np.sort(np.random.default_rng(seed).choice(rows, UNLABELED_ROWS_PER_BRAIN, replace=False))


def fit_classifier(train, kind, config, output_dir, *, program, artifact_store,
                   timeout=300, memory_mb=DEFAULT_MEMORY_MB, threads=1, feature_frames=None,
                   fit_rows=None, random_seed=None):
    """Fit GT-labeled TRAIN rows; optional host-owned subsets support isolated internal folds.

    Labels are three-valued (1 / 0 / NaN): `X`/`y` hold only rows labeled 1 or 0. A
    program whose `fit` declares `X_unlabeled` also receives features (no labels, no
    images) of a deterministic sample of the NaN rows among the selected rows.
    No heldout features or labels enter the fit worker. These optional arguments
    are framework transport controls, not agent-configurable data paths.
    """
    validate_program(program)
    config = normalize_config(config)
    if not train:
        raise ClassifierTrainingError('Training requires at least one TRAIN brain')
    frames = (feature_frames if feature_frames is not None else
              {brain: augmented_features(program, bank.tables[kind], timeout, memory_mb)
               for brain, bank in train.items()})
    if set(frames) != set(train) or (fit_rows is not None and set(fit_rows) != set(train)):
        raise ClassifierTrainingError('Internal feature/subset transport must match the TRAIN brains')
    selections, unlabeled = {}, {}
    wants_unlabeled = declares_parameter(program, 'X_unlabeled')
    for brain, bank in train.items():
        table = bank.tables[kind]
        size = len(table.features)
        rows = np.asarray(fit_rows[brain]) if fit_rows is not None else np.arange(size)
        if (len(frames[brain]) != size or rows.ndim != 1 or rows.dtype.kind not in 'iu'
                or (rows < 0).any() or (rows >= size).any() or len(np.unique(rows)) != len(rows)):
            raise ClassifierTrainingError('Invalid host TRAIN row selection or feature alignment')
        truth = np.asarray(table.truth, dtype=float)
        if len(truth) != size:
            raise ClassifierTrainingError('Invalid TRAIN labels or row alignment')
        known = ~np.isnan(truth[rows])
        selections[brain] = rows[known]
        if wants_unlabeled:
            unlabeled[brain] = _unlabeled_sample(np.sort(rows[~known]), f"{brain}/{kind}/{table.meta['pool_sha256']}")
    first = frames[sorted(frames)[0]]
    columns = [name for name in first.columns if all(name in frame.columns for frame in frames.values())]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'training.py').write_text(program)
    with tempfile.TemporaryDirectory(prefix='classifier_train_') as tmp:
        root = Path(tmp)
        count = sum(len(rows) for rows in selections.values())
        if count < 1 or not columns:
            raise ClassifierTrainingError('Training requires nonempty TRAIN rows and predictor columns')
        x = np.lib.format.open_memmap(root / 'X.npy', mode='w+', dtype=np.float64, shape=(count, len(columns)))
        y = np.lib.format.open_memmap(root / 'y.npy', mode='w+', dtype=np.int8, shape=(count,))
        # Label-free brain membership per training row, so a program can weight or
        # block by source brain. Codes index group_names; no identities beyond that.
        group_names = sorted(train)
        groups = np.lib.format.open_memmap(root / 'groups.npy', mode='w+', dtype=np.int16, shape=(count,))
        provenance, offset, positives = [], 0, 0
        for brain in sorted(train):
            table = train[brain].tables[kind]
            if not np.isin(np.asarray(table.truth, dtype=float)[selections[brain]], [0., 1.]).all():
                raise ClassifierTrainingError('Invalid TRAIN labels or row alignment')
            feature_hash = hashlib.sha256()
            rows = selections[brain]
            for start in range(0, len(rows), 65536):
                stop = min(start + 65536, len(rows))
                values = frames[brain].iloc[rows[start:stop]][columns].to_numpy(dtype=float, copy=True)
                x[offset + start:offset + stop] = values
                feature_hash.update(values.astype('<f8', copy=False).tobytes())
            labels = np.asarray(table.truth, dtype=float)[rows].astype(np.int8)
            y[offset:offset + len(rows)] = labels
            groups[offset:offset + len(rows)] = group_names.index(brain)
            positives += int(labels.sum())
            provenance.append({'brain': brain, 'rows': len(rows), 'pool_sha256': table.meta['pool_sha256'],
                               'features_sha256': feature_hash.hexdigest(),
                               **({'fit_rows_sha256': hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()}
                                  if fit_rows is not None else {}),
                               'labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
                               **({'unlabeled_rows': len(unlabeled[brain]),
                                   'unlabeled_rows_sha256': hashlib.sha256(
                                       unlabeled[brain].astype('<i8').tobytes()).hexdigest()}
                                  if wants_unlabeled else {})})
            offset += len(rows)
        x.flush()
        y.flush()
        groups.flush()
        del x, y, groups
        unlabeled_count = sum(len(rows) for rows in unlabeled.values())
        if wants_unlabeled:
            xu = np.lib.format.open_memmap(root / 'X_unlabeled.npy', mode='w+', dtype=np.float64,
                                           shape=(unlabeled_count, len(columns)))
            start = 0
            for brain in sorted(train):
                for chunk in range(0, len(unlabeled[brain]), 65536):
                    part = unlabeled[brain][chunk:chunk + 65536]
                    xu[start:start + len(part)] = frames[brain].iloc[part][columns].to_numpy(dtype=float, copy=True)
                    start += len(part)
            xu.flush()
            del xu
        request = {'mode': 'fit', 'columns': columns, 'config': config, 'group_names': group_names,
                   'timeout': timeout, 'memory_mb': memory_mb, 'threads': threads,
                   **({'unlabeled_inputs': True} if wants_unlabeled else {}),
                   **({'random_seed': random_seed} if random_seed is not None else {})}
        from .image_contract import image_spec
        from .image_features import stage_images
        image_contract = image_spec(program)
        image_transport = None
        if image_contract is not None and image_contract[1]['raw_patches']:
            image_transport = stage_images(root, [(frames[brain], selections[brain]) for brain in sorted(train)])
            request['image_inputs'] = True
        fingerprint = config_identity({'columns': columns, 'tables': provenance,
                                       **({'image_transport': image_transport} if image_transport is not None else {})})
        (root / 'training.py').write_text(program)
        (root / 'artifacts').mkdir()
        (output_dir / 'request.json').write_text(json.dumps({**request, 'training_tables': provenance,
                    'training_fingerprint': fingerprint, 'image_transport': image_transport}, indent=2, allow_nan=False))
        status = run_worker(root, request, log_path=output_dir / 'worker.log')
        files = artifact_files(root / 'artifacts')
        model = {'version': MODEL_VERSION, 'kind': kind, 'config': config, 'program': program,
                 'training_brains': sorted(train), 'training_rows': count, 'training_fingerprint': fingerprint,
                 'files': files, 'artifact_sha256': config_identity(files), 'threads': threads,
                 # Exact input columns, so inference can reindex and registered descriptors are explicit.
                 'columns': list(columns)}
        source = render_model(model)
        store = Path(artifact_store)
        destination = store / model['artifact_sha256']
        store.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copytree(root / 'artifacts', destination)
        verify_artifacts(model, store)
        summary = {key: model[key] for key in ('config', 'training_brains', 'training_rows',
                    'training_fingerprint', 'artifact_sha256', 'threads')}
        summary.update(class_counts={'0': count - positives, '1': positives},
                       unlabeled_rows=unlabeled_count if wants_unlabeled else None,
                       image_transport=image_transport,
                       image_context={brain: frame.attrs['image_context'] for brain, frame in frames.items()
                                      if 'image_context' in frame.attrs},
                       input_columns=columns, artifact_files=files,
                       local_context={brain: frame.attrs['local_context'] for brain, frame in frames.items()
                                      if 'local_context' in frame.attrs},
                       program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                       component_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       wall_seconds=status['wall_seconds'], versions=status.get('versions', {}),
                       training_output_tail=status.get('training_output_tail', ''),
                       preprocessing='Defined by agent training.py; fitted state saved in artifacts',
                       # The full fit's own TRAIN metric is resubstitution; branch selection uses
                       # the run's selection protocol (see train_experiments / selection_protocol).
                       group_transport={'group_names': group_names,
                                        'rows': {brain: int(len(selections[brain])) for brain in group_names}},
                       training_metric_protocol='in_sample_resubstitution')
        summary['implementation_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).with_name('classifier_train_worker.py'),
            Path(__file__).with_name('classifier_contract.py'), Path(__file__).with_name('model_execution.py'),
            Path(__file__).with_name('model_sandbox.py'), Path(__file__).with_name('local_context.py'),
            Path(__file__).with_name('local_features.py'), Path(__file__).with_name('local_feature_contract.py'),
            Path(__file__).with_name('image_context.py'), Path(__file__).with_name('image_coordinates.py'),
            Path(__file__).with_name('image_features.py'), Path(__file__).with_name('image_contract.py'),
            Path(__file__).with_name('image_selection.py'))}
        (output_dir / 'model.json').write_text(json.dumps(model, allow_nan=False))
        (output_dir / 'scorer.py').write_text(source)
        (output_dir / 'status.json').write_text(json.dumps(status, indent=2))
        (output_dir / 'training_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        return source, summary
