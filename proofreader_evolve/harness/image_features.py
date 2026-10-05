"""Image feature extraction and host-only patch transport for arbitrary models."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .image_contract import image_spec
from .image_context import digest_json, file_hash
from .image_selection import select_image_rows, selection_coverage
from .model_execution import run_worker

MAX_STAGE_BYTES = 1024**3


def subset_features(frame, indices, *, missing_row=False):
    """Keep private image-row transport aligned when a frame is subset/reset."""
    import pandas as pd
    indices = np.asarray(indices, dtype=int)
    result = frame.iloc[indices].copy().reset_index(drop=True)
    attrs = deepcopy(frame.attrs)
    rows = frame.attrs.get('_image_rows', {})
    if '_image_rows' in frame.attrs:
        attrs['_image_rows'] = {new: rows[int(old)] for new, old in enumerate(indices) if int(old) in rows}
    if missing_row:
        missing = pd.DataFrame(np.nan, index=[0], columns=frame.columns)
        missing['detector_score'] = 0.
        if 'image_available' in missing:
            missing['image_available'] = 0.
        result.attrs = {}
        result = pd.concat([result, missing], ignore_index=True)
    result.attrs = attrs
    return result


def stage_images(root, frames_and_rows):
    """Stage only requested rows' patches for fitting, prediction or 3D analysis."""
    directory = Path(root) / 'images'
    directory.mkdir()
    mapping, checksums, total, offset = {}, {}, 0, 0
    for frame, indices in frames_and_rows:
        if '_image_rows' not in frame.attrs:
            raise ValueError('Image inputs were not prepared for the requested rows')
        rows = frame.attrs['_image_rows']
        for local, original in enumerate(indices):
            files = []
            for occurrence, path in enumerate(rows.get(int(original), [])):
                path = Path(path)
                total += path.stat().st_size
                if total > MAX_STAGE_BYTES:
                    raise ValueError('Image transport exceeds 1 GiB; reduce selected candidates, radius or occurrences')
                name = f'{offset + local}_{occurrence}.npz'
                shutil.copyfile(path, directory / name)
                checksums[name] = file_hash(directory / name)
                files.append(name)
            if files:
                mapping[str(offset + local)] = files
        offset += len(indices)
    (directory / 'index.json').write_text(json.dumps({'rows': offset, 'patches': mapping}, allow_nan=False))
    return {'rows': offset, 'rows_with_images': len(mapping), 'bytes': total,
            'content_sha256': digest_json({'rows': offset, 'patches': mapping, 'checksums': checksums})}


def augmented_image_features(source, table, frame, timeout=120, memory_mb=8192):
    contract = image_spec(source)
    if contract is None:
        return frame
    program, spec = contract
    provider = getattr(table, 'image_context', None)
    if provider is None:
        raise ValueError('No host image provider is attached')
    names = [*spec['feature_names'], 'image_available']
    if set(names) & set(frame.columns):
        raise ValueError('Image feature names collide with existing predictors')
    started = time.monotonic()
    selected = select_image_rows(table.features, spec, getattr(table, 'keys', None))
    coverage = selection_coverage(table.features, spec, selected, getattr(table, 'keys', None))
    rows, manifests, total_bytes = {}, {}, 0
    batch_indices, batch_rows, batch_bytes = [], {}, 0
    values = np.empty((len(selected), len(spec['feature_names'])), dtype=float)
    batches, cached_batches, written_rows = 0, 0, 0
    batch_limit = min(spec['batch_max_mb'] * 1024**2, MAX_STAGE_BYTES)
    total_limit = spec['max_total_mb'] * 1024**2
    if spec['raw_patches']:
        # Arbitrary fit/predict code may need all its selected inputs together.
        # Only the stateless numeric extractor can be safely batched by the host.
        total_limit = min(total_limit, MAX_STAGE_BYTES)
    base_identity = {'version': 'batched-image-features-v2', 'program': digest_json(program), 'spec': spec,
        'pool': table.meta['pool_sha256'], 'features': table.meta.get('files', {}).get('features.pkl'),
        'implementation': {name: file_hash(Path(__file__).with_name(name)) for name in (
            'image_features.py', 'image_selection.py', 'local_features.py', 'image_context.py',
            'image_coordinates.py', 'image_contract.py', 'classifier_train_worker.py',
            'model_execution.py', 'model_sandbox.py')}}

    def remaining_time():
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Candidate image preparation/extraction exceeded its time budget')
        return remaining

    def flush_batch():
        nonlocal batches, cached_batches, written_rows, batch_bytes
        if not batch_indices:
            return
        remaining_time()
        identity = {**base_identity, 'rows': batch_indices,
                    'patches': {str(i): manifests[str(i)] for i in batch_indices}}
        key = digest_json(identity)
        destination = provider.directory / 'features' / key
        cached = destination.is_dir()
        if not cached:
            with tempfile.TemporaryDirectory(prefix='image_features_') as tmp:
                root = Path(tmp)
                base = table.features.iloc[batch_indices].reset_index(drop=True).copy()
                base.attrs['_image_rows'] = {i: batch_rows[row] for i, row in enumerate(batch_indices)}
                stage_images(root, [(base, np.arange(len(base)))])
                np.save(root / 'X.npy', base.to_numpy(dtype=float), allow_pickle=False)
                (root / 'training.py').write_text(program)
                (root / 'artifacts').mkdir()
                logs = provider.directory / 'worker_logs'
                logs.mkdir(parents=True, exist_ok=True)
                request = {'mode': 'extract_image', 'columns': list(base.columns),
                           'feature_names': spec['feature_names'], 'rows': len(base),
                           'config': {'parameters': {}}, 'image_inputs': True,
                           'timeout': remaining_time(), 'memory_mb': memory_mb, 'threads': 1}
                run_worker(root, request, log_path=logs / f'{key}.log')
                output = root / 'local_features.npy'
                if output.stat().st_size > len(base) * len(spec['feature_names']) * 8 + 4096:
                    raise ValueError('Image feature output exceeds its expected size')
                extracted = np.load(output, allow_pickle=False)
                if extracted.shape != (len(base), len(spec['feature_names'])) or np.isinf(extracted).any():
                    raise ValueError('Invalid image feature output')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='.staging_', dir=destination.parent) as staging:
                    staged = Path(staging) / 'entry'
                    staged.mkdir()
                    np.save(staged / 'values.npy', extracted, allow_pickle=False)
                    (staged / 'metadata.json').write_text(json.dumps({
                        'identity': identity, 'values_sha256': file_hash(staged / 'values.npy')}, allow_nan=False))
                    staged.rename(destination)
        metadata = json.loads((destination / 'metadata.json').read_text())
        if metadata['identity'] != identity or metadata['values_sha256'] != file_hash(destination / 'values.npy'):
            raise ValueError('Image feature cache integrity failure')
        extracted = np.load(destination / 'values.npy', allow_pickle=False)
        if extracted.shape != (len(batch_indices), len(spec['feature_names'])) or np.isinf(extracted).any():
            raise ValueError('Cached image features have an invalid shape/value')
        values[written_rows:written_rows + len(batch_indices)] = extracted
        written_rows += len(batch_indices)
        batches += 1
        cached_batches += int(cached)
        if provider.trace:
            provider.trace.emit('image_feature_batch', 'Image feature batch completed',
                brain=table.meta['provenance']['brain'], kind=table.kind, batch=batches,
                rows=len(batch_indices), staged_bytes=batch_bytes, cached=cached,
                processed_rows=written_rows, selected_rows=len(selected))
        batch_indices.clear()
        batch_rows.clear()
        batch_bytes = 0

    for index in selected:
        remaining_time()
        paths, identities = [], []
        row_bytes = 0
        for occurrence in range(spec['max_occurrences']):
            remaining_time()
            if occurrence and occurrence >= metadata['occurrences_total']:
                break
            path, metadata = provider.patch(table, int(index), spec, occurrence)
            remaining_time()
            size = path.stat().st_size
            total_bytes += size
            row_bytes += size
            if total_bytes > total_limit:
                raise ValueError('Selected images exceed max_total_mb or the 1 GiB raw-patch limit; '
                                 'reduce radius/occurrences, use a reviewed coarser level, or stage a smaller pilot')
            paths.append(str(path))
            identities.append(metadata['patch_sha256'])
        manifests[str(int(index))] = identities
        if spec['raw_patches']:
            rows[int(index)] = paths
        if spec['feature_names']:
            if row_bytes > batch_limit:
                raise ValueError('One candidate exceeds batch_max_mb; reduce radius/occurrences or raise that limit')
            if batch_indices and (len(batch_indices) >= spec['batch_size'] or batch_bytes + row_bytes > batch_limit):
                flush_batch()
            batch_indices.append(int(index))
            batch_rows[int(index)] = paths
            batch_bytes += row_bytes
    if spec['feature_names']:
        flush_batch()
    remaining_time()
    key = digest_json({**base_identity, 'patches': manifests, 'selected_rows': selected.tolist()})
    result = frame.copy()
    if spec['feature_names']:
        for column, name in enumerate(spec['feature_names']):
            result[name] = np.nan
            result.iloc[selected, result.columns.get_loc(name)] = values[:, column]
    result['image_available'] = 0.
    result.iloc[selected, result.columns.get_loc('image_available')] = 1.
    if spec['raw_patches']:
        result.attrs['_image_rows'] = rows
    result.attrs['image_context'] = {'spec': spec, 'selected_rows': len(selected), 'pool_rows': len(frame),
        'bytes': total_bytes, 'feature_cache_key': key,
        'input_sha256': digest_json({'patches': manifests, 'selected_rows': selected.tolist()}),
        'cached_features': bool(batches) and cached_batches == batches,
        'extraction_batches': batches, 'cached_batches': cached_batches,
        'selection_coverage': coverage,
        'input_path': 'raw_patches' if spec['raw_patches'] else 'numeric_features',
        'limits': {'batch_rows': spec['batch_size'], 'batch_bytes': batch_limit, 'total_bytes': total_limit},
        'wall_seconds': time.monotonic() - started, 'alignment': 'reviewed_source_and_cache_identity'}
    if provider.trace:
        provider.trace.emit('image_features_ready', 'Candidate image inputs prepared',
                            brain=table.meta['provenance']['brain'], kind=table.kind, **result.attrs['image_context'])
    return result
