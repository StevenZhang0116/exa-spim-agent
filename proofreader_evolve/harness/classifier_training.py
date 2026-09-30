"""Host-owned TRAIN transport, code snapshots and immutable opaque model artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .classifier_contract import (MODEL_VERSION, normalize_config, render_model, validate_program,
                                  artifact_files, config_identity, verify_artifacts)
from .model_execution import ModelExecutionError, run_worker


class ClassifierTrainingError(ModelExecutionError):
    pass


def available_features(train, kind):
    columns = [list(train[brain].tables[kind].features.columns) for brain in sorted(train)]
    return [name for name in columns[0] if all(name in other for other in columns[1:])]


def fit_classifier(train, kind, config, output_dir, *, program, artifact_store,
                   timeout=300, memory_mb=8192, threads=1):
    """All TRAIN rows enter fit; no validation argument, raw-data path or model whitelist."""
    validate_program(program)
    config = normalize_config(config)
    columns = available_features(train, kind)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'training.py').write_text(program)
    with tempfile.TemporaryDirectory(prefix='classifier_train_') as tmp:
        root = Path(tmp)
        count = sum(len(bank.tables[kind].features) for bank in train.values())
        if count < 1 or not columns:
            raise ClassifierTrainingError('Training requires nonempty TRAIN rows and predictor columns')
        x = np.lib.format.open_memmap(root / 'X.npy', mode='w+', dtype=np.float64, shape=(count, len(columns)))
        y = np.lib.format.open_memmap(root / 'y.npy', mode='w+', dtype=np.int8, shape=(count,))
        provenance, offset, positives = [], 0, 0
        for brain in sorted(train):
            table = train[brain].tables[kind]
            if len(table.truth) != len(table.features) or not np.isin(table.truth, [0, 1]).all():
                raise ClassifierTrainingError('Invalid TRAIN labels or row alignment')
            feature_hash = hashlib.sha256()
            for start in range(0, len(table.features), 65536):
                stop = min(start + 65536, len(table.features))
                values = table.features.iloc[start:stop][columns].to_numpy(dtype=float, copy=True)
                x[offset + start:offset + stop] = values
                feature_hash.update(values.astype('<f8', copy=False).tobytes())
            y[offset:offset + len(table.truth)] = table.truth
            positives += int(np.asarray(table.truth).sum())
            provenance.append({'brain': brain, 'rows': len(table.truth), 'pool_sha256': table.meta['pool_sha256'],
                               'features_sha256': feature_hash.hexdigest(),
                               'labels_sha256': hashlib.sha256(np.asarray(table.truth, dtype=np.int8).tobytes()).hexdigest()})
            offset += len(table.truth)
        x.flush()
        y.flush()
        del x, y
        fingerprint = config_identity({'columns': columns, 'tables': provenance})
        request = {'mode': 'fit', 'columns': columns, 'config': config,
                   'timeout': timeout, 'memory_mb': memory_mb, 'threads': threads}
        (root / 'training.py').write_text(program)
        (root / 'artifacts').mkdir()
        (output_dir / 'request.json').write_text(json.dumps({**request, 'training_tables': provenance,
                    'training_fingerprint': fingerprint}, indent=2, allow_nan=False))
        status = run_worker(root, request, log_path=output_dir / 'worker.log')
        files = artifact_files(root / 'artifacts')
        model = {'version': MODEL_VERSION, 'kind': kind, 'config': config, 'program': program,
                 'training_brains': sorted(train), 'training_rows': count, 'training_fingerprint': fingerprint,
                 'files': files, 'artifact_sha256': config_identity(files), 'threads': threads}
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
                       input_columns=columns, artifact_files=files,
                       program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                       component_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       wall_seconds=status['wall_seconds'], versions=status.get('versions', {}),
                       training_output_tail=status.get('training_output_tail', ''),
                       preprocessing='Defined by agent training.py; fitted state saved in artifacts',
                       training_metric_protocol='in_sample_resubstitution')
        summary['implementation_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).with_name('classifier_train_worker.py'),
            Path(__file__).with_name('classifier_contract.py'), Path(__file__).with_name('model_execution.py'),
            Path(__file__).with_name('model_sandbox.py'))}
        (output_dir / 'model.json').write_text(json.dumps(model, allow_nan=False))
        (output_dir / 'scorer.py').write_text(source)
        (output_dir / 'status.json').write_text(json.dumps(status, indent=2))
        (output_dir / 'training_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        return source, summary
