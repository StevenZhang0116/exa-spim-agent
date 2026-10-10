"""Geometry transport and run-local caching for isolated feature extraction."""
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .local_context import CONTEXT_VERSION, cache_identity
from .local_feature_contract import local_feature_spec
from .model_execution import ModelExecutionError, run_worker, DEFAULT_MEMORY_MB


MAX_CONTEXT_BYTES = 128 * 1024**2


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_rows(frame, spec):
    feature = spec['selection_feature']
    if feature not in frame.columns:
        raise ValueError('selection_feature must be an existing base predictor for geometry or image selection')
    values = frame[feature].to_numpy(dtype=float, copy=True)
    priority = -values if spec['selection_largest'] else values
    priority[~np.isfinite(priority)] = np.inf
    # Frozen row order breaks ties. No label or agent-chosen candidate IDs enter selection.
    return np.lexsort((np.arange(len(frame)), priority))[:spec['max_candidates']]


def augmented_features(source, table, timeout=120, memory_mb=DEFAULT_MEMORY_MB):
    from .image_features import augmented_image_features
    frame = _geometry_features(source, table, timeout, memory_mb)
    return augmented_image_features(source, table, frame, timeout, memory_mb)


def _geometry_features(source, table, timeout=120, memory_mb=DEFAULT_MEMORY_MB):
    started = time.monotonic()
    contract = local_feature_spec(source)
    if contract is None:
        return table.features
    program, spec = contract
    provider = getattr(table, 'local_context', None)
    if provider is None:
        raise ValueError('Local geometry is unavailable: this table has no host context provider')
    names = [*spec['feature_names'], 'local_context_available']
    if set(names) & set(table.features.columns):
        raise ValueError('Local features cannot overwrite existing predictor columns')
    provenance = table.meta['provenance']
    if cache_identity(provenance['source_cache']['path']) != provenance['source_cache']:
        raise ValueError('Fragment cache changed since native feature preparation')
    identity = {'version': CONTEXT_VERSION, 'spec': spec, 'program_sha256': hashlib.sha256(program.encode()).hexdigest(),
                'pool_sha256': table.meta['pool_sha256'], 'source_cache': provenance['source_cache'],
                'features_sha256': table.meta['files']['features.pkl'], 'kind': table.kind,
                'implementation': {name: _sha(Path(__file__).with_name(name)) for name in (
                    'local_features.py', 'local_context.py', 'local_feature_contract.py',
                    'classifier_train_worker.py', 'model_execution.py', 'model_sandbox.py')}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    store = provider.store.directory
    store.mkdir(parents=True, exist_ok=True)
    destination = store / key
    selected = select_rows(table.features, spec)
    cached = destination.is_dir()
    trace = provider.store.trace
    if trace:
        trace.emit('local_features_start', 'Loading or extracting local fragment features',
                   brain=provenance['brain'], kind=table.kind, selected_rows=len(selected),
                   pool_rows=len(table.features), cached=cached, feature_cache_key=key)
    if not cached:
        with tempfile.TemporaryDirectory(prefix='local_features_') as tmp:
            root = Path(tmp)
            (root / 'training.py').write_text(program)
            (root / 'artifacts').mkdir()
            byte_count, truncated_sites, occurrence_limited_rows = 0, 0, 0
            with (root / 'contexts.jsonl').open('w') as stream:
                for index in selected:
                    if time.monotonic() - started >= timeout:
                        raise ModelExecutionError('Local geometry preparation exceeded the feature time budget')
                    context, _ = provider.context(int(index), radius_um=spec['radius_um'],
                        max_nodes=spec['max_nodes'], max_occurrences=spec['max_occurrences'])
                    context['features'] = {str(k): float(v) if np.isfinite(v) else None
                                           for k, v in table.features.iloc[index].items()}
                    truncated_sites += sum(s['truncated'] for s in context['sites'])
                    occurrence_limited_rows += int(context['occurrences_truncated'])
                    line = json.dumps(context, allow_nan=False, separators=(',', ':')) + '\n'
                    byte_count += len(line.encode())
                    if byte_count > MAX_CONTEXT_BYTES:
                        raise ModelExecutionError('Local geometry exceeds 128 MiB; reduce nodes, occurrences or candidates')
                    stream.write(line)
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise ModelExecutionError('Local geometry preparation exceeded the feature time budget')
            request = {'mode': 'extract', 'feature_names': spec['feature_names'], 'rows': len(selected),
                       'timeout': remaining, 'memory_mb': memory_mb, 'threads': 1}
            logs = store / 'worker_logs'
            logs.mkdir(exist_ok=True)
            status = run_worker(root, request, log_path=logs / f'{key}.log')
            output = root / 'local_features.npy'
            if output.stat().st_size > len(selected) * len(spec['feature_names']) * 8 + 4096:
                raise ModelExecutionError('Local feature output exceeds the declared shape')
            values = np.load(output, allow_pickle=False)
            if (values.dtype != np.float64 or values.shape != (len(selected), len(spec['feature_names']))
                    or np.isinf(values).any()):
                raise ModelExecutionError('Invalid local feature output shape, dtype or infinite value')
            metadata = {'identity': identity, 'selected_rows': len(selected), 'context_bytes': byte_count,
                        'truncated_sites': truncated_sites, 'worker_seconds': status['wall_seconds'],
                        'occurrence_limited_rows': occurrence_limited_rows,
                        'versions': status.get('versions', {}),
                        'preparation_and_extraction_seconds': time.monotonic() - started}
            # Host-only, atomic cache; workers cannot read or write it.
            with tempfile.TemporaryDirectory(prefix='.staging_', dir=store) as staging:
                staged = Path(staging) / 'entry'
                staged.mkdir()
                np.save(staged / 'values.npy', values, allow_pickle=False)
                metadata['values_sha256'] = _sha(staged / 'values.npy')
                (staged / 'metadata.json').write_text(json.dumps(metadata, allow_nan=False))
                staged.rename(destination)
    metadata = json.loads((destination / 'metadata.json').read_text())
    if metadata['identity'] != identity or _sha(destination / 'values.npy') != metadata['values_sha256']:
        raise ValueError('Local feature cache identity/checksum mismatch')
    values = np.load(destination / 'values.npy', allow_pickle=False)
    if values.dtype != np.float64 or values.shape != (len(selected), len(spec['feature_names'])) or np.isinf(values).any():
        raise ValueError('Invalid cached local feature shape or values')
    frame = table.features.copy()
    for column, name in enumerate(spec['feature_names']):
        full = np.full(len(frame), np.nan)
        full[selected] = values[:, column]
        frame[name] = full
    available = np.zeros(len(frame))
    available[selected] = 1.
    frame['local_context_available'] = available
    frame.attrs['local_context'] = {'version': CONTEXT_VERSION, 'spec': spec, 'feature_cache_key': key,
                                   'selected_rows': len(selected), 'pool_rows': len(frame), 'cached': cached,
                                   'truncated_sites': metadata['truncated_sites'],
                                   'occurrence_limited_rows': metadata['occurrence_limited_rows'],
                                   'wall_seconds': time.monotonic() - started}
    if trace:
        trace.emit('local_features_ready', 'Local fragment features ready; fixed pool preserved',
                   brain=provenance['brain'], kind=table.kind, **frame.attrs['local_context'])
    return frame
