"""Pool-scale execution of agent descriptors over cached contexts in parallel sandboxed workers.

The host fans batches of rows out to many single-threaded workers (the sandbox forbids
child processes), enforces one wall-clock deadline, caches results by descriptor code
hash so repeats cost nothing across generations and runs, and registers the resulting
columns as ordinary predictors. Labels are touched only by the summary the host builds
for TRAIN brains after the computation.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from .descriptor_contract import DESCRIPTOR_VERSION
from .image_context import file_hash
from .model_execution import ModelExecutionError, run_worker


RUN_VERSION = 'agent-descriptor-compute-v1'
SCOPES = ('pilot', 'boundary', 'all_cached')
PILOT_ROWS = 512
BOUNDARY_ROWS = 2000
BATCH_ROWS = 256
CONTEXT_COLUMN = 'bank_context_available'
IMPLEMENTATION_FILES = ('descriptor_runs.py', 'classifier_train_worker.py', 'model_execution.py', 'model_sandbox.py')


def available_workers(reserve=2):
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError, NotImplementedError):
        count = os.cpu_count() or 1
    return max(1, count - reserve)


def stage_batch(root, contexts, patches):
    """Worker transport: contexts JSON plus the ImageDataset layout for rows with patches."""
    root = Path(root)
    (root / 'volume_contexts.json').write_text(json.dumps(contexts, allow_nan=False))
    mapping = {}
    if any(patch is not None for patch in patches):
        images = root / 'images'
        images.mkdir()
        for index, patch in enumerate(patches):
            if patch is None:
                continue
            name = f'{index}_0.npz'
            np.savez(images / name, **patch)
            mapping[str(index)] = [name]
        (images / 'index.json').write_text(json.dumps({'rows': len(contexts), 'patches': mapping}, allow_nan=False))
    return bool(mapping)


class DescriptorRuns:
    def __init__(self, cache, bank_dir, *, workers='auto', memory_mb=8192, trace=None):
        self.cache, self.bank_dir, self.trace = cache, Path(bank_dir), trace
        self.workers = available_workers() if workers == 'auto' else max(1, int(workers))
        self.memory_mb = memory_mb
        self.implementation = {name: file_hash(Path(__file__).with_name(name)) for name in IMPLEMENTATION_FILES}

    def scope_rows(self, entry, scope, k):
        if scope not in SCOPES:
            raise ValueError(f'scope must be one of {SCOPES}')
        rows = entry.rows
        if scope == 'pilot':
            return rows[:PILOT_ROWS]
        if scope == 'boundary':
            k = min(int(k), len(rows))
            return rows[max(0, k - BOUNDARY_ROWS):k + BOUNDARY_ROWS]
        return rows

    def identity(self, table, spec, scope, rows):
        return {'version': RUN_VERSION, 'descriptor_version': DESCRIPTOR_VERSION,
                'code_sha256': spec['code_sha256'], 'inputs': spec['inputs'],
                'image_tier': spec['image_tier'] if spec['inputs'] != 'geometry' else None,
                'feature_names': list(spec['feature_names']), 'scope': scope,
                'rows_sha256': hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest(),
                'context_cache': self.cache.identity_key(table), 'implementation': self.implementation}

    def directory(self, table, spec, scope):
        from .context_cache import table_identity
        identity = table_identity(table)
        return self.bank_dir / identity['brain'] / table.kind / identity['pool_sha256'] / spec['code_sha256'] / scope

    def cached(self, table, spec, scope, k):
        entry = self.cache.entry(table)
        rows = self.scope_rows(entry, scope, k)
        directory = self.directory(table, spec, scope)
        manifest = directory / 'manifest.json'
        if not manifest.is_file():
            return False
        return json.loads(manifest.read_text()).get('identity') == self.identity(table, spec, scope, rows)

    def _run_batch(self, root, table, entry, program, spec, rows, deadline, log_path):
        contexts, patches = entry.contexts(table, rows, inputs=spec['inputs'], tier=spec['image_tier'])
        has_images = stage_batch(root, contexts, patches)
        (root / 'training.py').write_text(program)
        (root / 'artifacts').mkdir()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelExecutionError('Descriptor wall budget exhausted before the batch started')
        request = {'mode': 'describe', 'feature_names': list(spec['feature_names']), 'rows': len(rows),
                   'image_inputs': has_images, 'timeout': remaining, 'memory_mb': self.memory_mb, 'threads': 1}
        status = run_worker(root, request, log_path=log_path)
        output = root / 'descriptors.npy'
        count = len(spec['feature_names'])
        if output.stat().st_size > len(rows) * count * 8 + 4096:
            raise ModelExecutionError('Descriptor output exceeds the declared shape')
        values = np.load(output, allow_pickle=False)
        if values.dtype != np.float64 or values.shape != (len(rows), count) or np.isinf(values).any():
            raise ModelExecutionError('Invalid descriptor output shape, dtype or infinite value')
        return values, float(status['wall_seconds'])

    def compute(self, table, program, spec, scope, *, k, wall_budget, log_dir, force=False):
        """Compute (or load) descriptor values for one table; returns rows, values and provenance."""
        entry = self.cache.entry(table)
        if spec['inputs'] != 'geometry' and spec['image_tier'] not in entry.tiers():
            raise ValueError(f"Image tier {spec['image_tier']} is not cached for {entry.brain}/{entry.kind}")
        rows = np.asarray(self.scope_rows(entry, scope, k), dtype=np.int64)
        identity = self.identity(table, spec, scope, rows)
        directory = self.directory(table, spec, scope)
        if directory.is_dir() and not force:
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest.get('identity') == identity and file_hash(directory / 'values.npy') == manifest['values_sha256']:
                values = np.load(directory / 'values.npy', allow_pickle=False)
                return {'rows': rows, 'values': values, 'cached': True, 'wall_seconds': 0.,
                        'worker_seconds': 0., 'batches': 0, 'workers': 0, 'manifest': manifest}
        started = time.monotonic()
        deadline = started + float(wall_budget)
        batches = [rows[i:i + BATCH_ROWS] for i in range(0, len(rows), BATCH_ROWS)]
        results, errors, cancel = [None] * len(batches), [], threading.Event()
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        def job(index):
            if cancel.is_set():
                return
            with tempfile.TemporaryDirectory(prefix='descriptor_batch_') as tmp:
                try:
                    results[index] = self._run_batch(Path(tmp), table, entry, program, spec, batches[index],
                                                     deadline, log_dir / f'batch{index:04d}.log')
                except Exception as exc:
                    cancel.set()
                    errors.append((index, exc))

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(job, range(len(batches))))
        if errors:
            index, exc = sorted(errors, key=lambda item: item[0])[0]
            raise ModelExecutionError(f'Descriptor batch {index} failed: {exc}') from exc
        if any(result is None for result in results):
            raise ModelExecutionError('Descriptor computation was cancelled before every batch completed')
        values = np.concatenate([result[0] for result in results]) if results else np.zeros((0, len(spec['feature_names'])))
        worker_seconds = float(sum(result[1] for result in results))
        manifest = {'identity': identity, 'rows': int(len(rows)), 'batches': len(batches), 'workers': self.workers,
                    'wall_seconds': time.monotonic() - started, 'worker_seconds_total': worker_seconds,
                    'created_at': datetime.now().astimezone().isoformat(timespec='seconds')}
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.staging_', dir=directory.parent) as staging:
            staged = Path(staging) / 'entry'
            staged.mkdir()
            np.save(staged / 'values.npy', values, allow_pickle=False)
            np.save(staged / 'rows.npy', rows, allow_pickle=False)
            manifest['values_sha256'] = file_hash(staged / 'values.npy')
            (staged / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
            if directory.exists():
                import shutil
                shutil.rmtree(directory)
            staged.rename(directory)
        return {'rows': rows, 'values': values, 'cached': False, 'wall_seconds': manifest['wall_seconds'],
                'worker_seconds': worker_seconds, 'batches': len(batches), 'workers': self.workers,
                'manifest': manifest}

    def plan(self, table, program, spec, *, k, sample_rows=64, timeout=600):
        """Time one small single-worker batch and extrapolate to each scope at the configured width."""
        entry = self.cache.entry(table)
        rows = np.asarray(entry.rows[:min(int(sample_rows), len(entry.rows))], dtype=np.int64)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='descriptor_plan_') as tmp:
            values, worker_seconds = self._run_batch(Path(tmp), table, entry, program, spec, rows,
                                                     started + timeout, Path(tmp) / 'plan.log')
        wall = time.monotonic() - started
        per_row = wall / max(1, len(rows))
        overhead = max(0., wall - worker_seconds)
        estimates = {}
        for scope in SCOPES:
            count = len(self.scope_rows(entry, scope, k))
            batches = -(-count // BATCH_ROWS)
            serial = per_row * count + overhead * batches
            estimates[scope] = {'rows': int(count), 'batches': int(batches),
                                'estimated_seconds': float(serial / max(1, self.workers))}
        return {'sample_rows': int(len(rows)), 'sample_wall_seconds': wall, 'seconds_per_row_single_worker': per_row,
                'workers': self.workers, 'estimates': estimates,
                'finite_fraction': {name: float(np.isfinite(values[:, j]).mean())
                                    for j, name in enumerate(spec['feature_names'])}}

    @staticmethod
    def register(table, columns, rows, values, *, replace=False):
        frame = table.features
        rows = np.asarray(rows, dtype=int)
        for position, column in enumerate(columns):
            full = np.full(len(frame), np.nan)
            full[rows] = values[:, position]
            if column in frame.columns and not replace:
                if not np.array_equal(frame[column].to_numpy(dtype=float), full, equal_nan=True):
                    raise ValueError(f'{column} is already registered with different values in this run')
                continue
            frame[column] = full

    @staticmethod
    def register_context_column(table, entry):
        if CONTEXT_COLUMN in table.features.columns:
            return
        available = np.zeros(len(table.features))
        available[np.asarray(entry.rows, dtype=int)] = 1.
        table.features[CONTEXT_COLUMN] = available

    @staticmethod
    def summarize(table, columns, rows, values, *, labels=True):
        """Host-only distribution summary; TRAIN labels are used here and nowhere upstream."""
        rows = np.asarray(rows, dtype=int)
        truth = np.asarray(table.truth)
        result = {}
        for position, column in enumerate(columns):
            series = values[:, position]
            finite = np.isfinite(series)
            entry = {'computed_rows': int(len(rows)), 'pool_rows': int(len(table.features)),
                     'finite_fraction': float(finite.mean()) if len(series) else None}
            if labels:
                for label in (0, 1):
                    mask = truth[rows] == label
                    selected = series[finite & mask]
                    entry[str(label)] = {'count': int(mask.sum()), 'finite': int((finite & mask).sum()),
                                         'q25_q50_q75': [float(format(v, '.6g')) for v in np.quantile(selected, [.25, .5, .75])]
                                         if len(selected) else None}
            result[column] = entry
        return result
