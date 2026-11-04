"""Pool-scale execution of agent descriptors over host-provided contexts in parallel sandboxed workers.

The host fans batches of rows out to many single-threaded workers (the sandbox forbids
child processes), enforces one wall-clock deadline, caches results by descriptor code,
input and implementation identity across generations and runs, and registers the resulting
columns as ordinary predictors. Labels are touched only by the host: the `judgeable` scope
selects the GT-labeled rows (the rows every fit and ranking uses), and the summary for
TRAIN brains is built after the computation.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from .descriptor_contract import DESCRIPTOR_VERSION
from . import descriptor_inputs
from . import memory_budget
from .image_context import file_hash
from .label_availability import labeled_rows
from .model_execution import ModelExecutionError, run_worker, DEFAULT_MEMORY_MB


RUN_VERSION = 'agent-descriptor-compute-v3-on-demand'
SCOPES = ('judgeable', 'pilot', 'boundary', 'all_candidates')
PILOT_ROWS = 512
BOUNDARY_ROWS = 2000
BATCH_ROWS = 256
# S3 patch reads, not workers, bound image descriptors: 24 threads read like 8 before the shared
# TensorStore context and like about 14 after it (plans/on_demand_contexts.md, 2026-11-03), so plans
# credit image descriptors with at most this concurrency and stay on the conservative side.
IMAGE_IO_PARALLELISM = 8
CONTEXT_COLUMN = 'bank_context_available'
IMPLEMENTATION_FILES = ('descriptor_runs.py', 'descriptor_inputs.py', 'descriptor_contract.py',
                        'classifier_train_worker.py', 'model_execution.py', 'model_sandbox.py', 'memory_budget.py')


def normalize_scope(scope):
    return 'all_candidates' if scope == 'all_cached' else scope


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
    def __init__(self, contexts, bank_dir, *, workers='auto', memory_mb=DEFAULT_MEMORY_MB, trace=None):
        self.contexts, self.bank_dir, self.trace = contexts, Path(bank_dir), trace
        self.requested_workers = workers
        cpu_limit = available_workers()
        self.workers = cpu_limit if workers == 'auto' else min(cpu_limit, max(1, int(workers)))
        self.memory_mb = memory_mb
        self.implementation = {name: file_hash(Path(__file__).with_name(name)) for name in IMPLEMENTATION_FILES}
        self._graph_loads = {}

    def capacity(self, pending=0):
        return memory_budget.admission(self.memory_mb, self.workers, pending=pending)

    @staticmethod
    def estimated_seconds(rows, rate, workers):
        # The first batch runs alone (serial admission); only subsequent batches overlap. The
        # fragment graph load is timed apart by plan() and charged once per brain.
        first = min(rows, BATCH_ROWS)
        remaining_batches = -(-(rows - first) // BATCH_ROWS)
        return float(rate * (first + (rows - first) / max(1, min(workers, remaining_batches))))

    def scope_rows(self, entry, scope, k, truth=None):
        """Row subset of one scope; `judgeable` needs the table's three-valued labels (host-side)."""
        scope = normalize_scope(scope)
        if scope not in SCOPES:
            raise ValueError(f'scope must be one of {SCOPES}')
        if scope == 'judgeable':
            # The rows the outer measurement ranks (label 1 or 0), on TRAIN, selection and validation
            # brains alike; the same truth Budgets.k_for reads. Unlabeled rows keep NaN.
            if truth is None:
                raise ValueError('The judgeable scope needs the table labels')
            return labeled_rows(truth)
        if scope == 'pilot':
            rows = np.sort(entry.rows)
            return np.sort(np.random.default_rng(0).choice(rows, min(PILOT_ROWS, len(rows)), replace=False))
        if scope == 'boundary':
            band = entry.band
            k = min(int(k), len(band))
            return band[max(0, k - BOUNDARY_ROWS):k + BOUNDARY_ROWS]
        return entry.rows

    def inference_rows(self, tables, k_for):
        """Rows frozen inference would compute per allowed scope on other brains' tables.

        A registered descriptor is recomputed at submission with the scope it was computed
        with on TRAIN, so the plan estimates every allowed scope. Labels and detector rank
        stay host-side; only counts leave this method.
        """
        return {scope: [int(len(self.scope_rows(self.contexts.entry(table), scope, k_for(table), truth=table.truth)))
                        for table in tables]
                for scope in getattr(self, 'allowed_scopes', SCOPES)}

    def identity(self, table, spec, scope, rows):
        scope = normalize_scope(scope)
        inputs = descriptor_inputs.source_identity(table, self.contexts.entry(table), spec)
        return {'version': RUN_VERSION, 'descriptor_version': DESCRIPTOR_VERSION,
                'code_sha256': spec['code_sha256'], 'inputs': spec['inputs'],
                'image_tier': spec['image_tier'] if spec['inputs'] != 'geometry' else None,
                'image_request': inputs, 'feature_names': list(spec['feature_names']), 'scope': scope,
                'rows_sha256': hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest(),
                'context_source': self.contexts.identity_key(
                    table, tiers=(inputs['cached_tier'],) if inputs['cached_tier'] is not None else ()),
                'implementation': self.implementation}

    def directory(self, table, spec, scope):
        scope = normalize_scope(scope)
        from .descriptor_context import table_identity
        identity = table_identity(table)
        return self.bank_dir / identity['brain'] / table.kind / identity['pool_sha256'] / spec['code_sha256'] / scope

    def cached(self, table, spec, scope, k):
        entry = self.contexts.entry(table)
        rows = self.scope_rows(entry, scope, k, truth=table.truth)
        directory = self.directory(table, spec, scope)
        manifest = directory / 'manifest.json'
        if not manifest.is_file():
            return False
        metadata = json.loads(manifest.read_text())
        if metadata.get('identity') != self.identity(table, spec, scope, rows):
            return False
        # Budget planning and execution must agree about whether numeric output is reusable.
        try:
            return file_hash(directory / 'values.npy') == metadata['values_sha256']
        except (OSError, KeyError):
            return False

    def _execute_batch(self, root, program, spec, rows, contexts, patches, deadline, log_path):
        # Recheck after input preparation, especially the first lazy graph load.
        memory_budget.require_capacity(self.capacity())
        root.mkdir(parents=True, exist_ok=True)
        has_images = stage_batch(root, contexts, patches)
        (root / 'training.py').write_text(program)
        (root / 'artifacts').mkdir()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelExecutionError('Descriptor wall budget exhausted before the batch started')
        request = {'mode': 'describe', 'feature_names': list(spec['feature_names']), 'rows': len(rows),
                   'image_inputs': has_images, 'timeout': remaining, 'memory_mb': self.memory_mb, 'threads': 1}
        try:
            status = run_worker(root, request, log_path=log_path)
        except ModelExecutionError as exc:
            match = re.search(r'Descriptor row (\d+):', str(exc))
            if match and int(match[1]) < len(rows):
                exc.row = int(rows[int(match[1])])
            raise
        output = root / 'descriptors.npy'
        count = len(spec['feature_names'])
        if output.stat().st_size > len(rows) * count * 8 + 4096:
            raise ModelExecutionError('Descriptor output exceeds the declared shape')
        values = np.load(output, allow_pickle=False)
        if values.dtype != np.float64 or values.shape != (len(rows), count) or np.isinf(values).any():
            raise ModelExecutionError('Invalid descriptor output shape, dtype or infinite value')
        return values, float(status['wall_seconds'])

    def _run_batch(self, root, table, entry, program, spec, rows, deadline, log_path, *,
                   persist_images=False, observe=None, timings=None):
        """Stream prepared patches into byte-bounded worker inputs; one path for all callers."""
        started = time.monotonic()
        contexts, patches, selected, results = [], [], [], []
        used_bytes, worker_seconds, executions = 0, 0., 0

        def flush():
            nonlocal worker_seconds, executions
            import shutil
            log = log_path if executions == 0 else log_path.with_name(f'{log_path.stem}_part{executions:03d}.log')
            worker_root = root / f'worker{executions:03d}'
            try:
                values, seconds = self._execute_batch(worker_root, program, spec,
                                                      selected, contexts, patches, deadline, log)
            finally:
                if worker_root.exists():
                    shutil.rmtree(worker_root)
            results.append(values)
            worker_seconds += seconds
            executions += 1

        stats = {'image_reads': 0, 'image_seconds': 0., 'image_seconds_max': 0.}
        try:
            for row, context, patch in descriptor_inputs.iter_inputs(
                    table, entry, spec, rows, root, persist=persist_images, deadline=deadline, stats=stats):
                size = sum(a.nbytes for a in patch.values()) if patch is not None else 0
                if contexts and used_bytes + size > descriptor_inputs.MAX_BATCH_BYTES:
                    flush()
                    contexts, patches, selected, used_bytes = [], [], [], 0
                if observe is not None:
                    observe(row, context, patch)
                contexts.append(context)
                patches.append(patch)
                selected.append(row)
                used_bytes += size
            if contexts:
                flush()
        finally:
            if timings is not None:
                timings.update(worker_seconds=worker_seconds, worker_batches=executions,
                               preparation_seconds=max(0., time.monotonic() - started - worker_seconds), **stats)
        values = np.concatenate(results) if results else np.zeros((0, len(spec['feature_names'])))
        return values, worker_seconds

    @staticmethod
    def _image_summary(timings):
        return {'image_reads': int(sum(t.get('image_reads', 0) for t in timings)),
                'image_seconds': float(sum(t.get('image_seconds', 0.) for t in timings)),
                'image_seconds_max': float(max((t.get('image_seconds_max', 0.) for t in timings), default=0.))}

    def _trace_images(self, phase, table, rows, summary):
        """One host-only event per call: counts and seconds, no rows or labels."""
        if self.trace is None or not summary['image_reads']:
            return
        self.trace.emit('descriptor_images_read', f"{phase}: {summary['image_reads']} image patches for {len(rows)} rows",
                        phase=phase, brain=table.meta['provenance']['brain'], kind=table.kind, rows=int(len(rows)),
                        reads=summary['image_reads'], seconds=round(summary['image_seconds'], 3),
                        max_seconds=round(summary['image_seconds_max'], 3))

    def sample_rows(self, table, count=128, seed=0):
        """Uniform over candidate IDs, independent of detector rank or labels."""
        if type(count) is not int or not 1 <= count <= 2048 or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Use sample_per_brain 1..2048 and an integer seed in [0, 2**32)')
        rows = np.sort(self.contexts.entry(table).rows)
        return np.sort(np.random.default_rng(seed).choice(rows, min(count, len(rows)), replace=False))

    def probe(self, table, program, spec, rows, *, timeout=600., log_dir, observe=None):
        """Run the real descriptor without registering columns or writing the descriptor bank."""
        if timeout <= 0:
            raise ModelExecutionError('Descriptor wall budget exhausted before probe started')
        entry = self.contexts.entry(table)
        rows = np.asarray(rows, dtype=np.int64)
        if not entry.has_rows(rows).all():
            raise ValueError('Probe candidates must belong to the candidate pool')
        started = time.monotonic()
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        chunks, timings = [], []
        try:
            for start in range(0, len(rows), BATCH_ROWS):
                memory_budget.require_capacity(self.capacity())
                timing = {}
                with tempfile.TemporaryDirectory(prefix='descriptor_probe_') as tmp:
                    values, _ = self._run_batch(Path(tmp), table, entry, program, spec,
                        rows[start:start + BATCH_ROWS], started + timeout, log_dir / f'batch{start // BATCH_ROWS:04d}.log',
                        persist_images=False, observe=observe, timings=timing)
                chunks.append(values)
                timings.append(timing)
        finally:
            entry.release()
        images = self._image_summary(timings)
        self._trace_images('probe', table, rows, images)
        return {'rows': rows, 'values': np.concatenate(chunks) if chunks else np.zeros((0, len(spec['columns']))),
                'wall_seconds': time.monotonic() - started,
                **{k: sum(t[k] for t in timings) for k in ('preparation_seconds', 'worker_seconds', 'worker_batches')},
                **images}

    def compute(self, table, program, spec, scope, *, k, wall_budget, log_dir, force=False):
        """Compute (or load) descriptor values for one table; returns rows, values and provenance."""
        started = time.monotonic()
        deadline = started + float(wall_budget)
        entry = self.contexts.entry(table)
        rows = np.asarray(self.scope_rows(entry, scope, k, truth=table.truth), dtype=np.int64)
        identity = self.identity(table, spec, scope, rows)
        directory = self.directory(table, spec, scope)
        if directory.is_dir() and not force:
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest.get('identity') == identity and file_hash(directory / 'values.npy') == manifest['values_sha256']:
                values = np.load(directory / 'values.npy', allow_pickle=False)
                return {'rows': rows, 'values': values, 'cached': True, 'wall_seconds': 0.,
                        'worker_seconds': 0., 'batches': 0, 'workers': 0, 'manifest': manifest}
        batches = [rows[i:i + BATCH_ROWS] for i in range(0, len(rows), BATCH_ROWS)]
        results, errors, cancel = [None] * len(batches), [], threading.Event()
        timings = [None] * len(batches)
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        def job(index):
            if cancel.is_set():
                return
            with tempfile.TemporaryDirectory(prefix='descriptor_batch_') as tmp:
                try:
                    timing = {}
                    results[index] = self._run_batch(Path(tmp), table, entry, program, spec, batches[index],
                                                     deadline, log_dir / f'batch{index:04d}.log', timings=timing)
                    timings[index] = timing
                except Exception as exc:
                    cancel.set()
                    errors.append((index, exc))

        loads_before = entry.chunk_loads
        peak_workers, admission_records = 0, []

        def record(snapshot, active):
            nonlocal peak_workers
            peak_workers = max(peak_workers, active)
            # Bounded summaries, not a per-batch log or any candidate/validation content.
            if not admission_records or active != admission_records[-1]['active_workers']:
                admission_records.append({**snapshot, 'active_workers': active})
                if len(admission_records) > 32:
                    del admission_records[1]
            elif snapshot['headroom_bytes'] < admission_records[-1]['headroom_bytes']:
                admission_records[-1] = {**snapshot, 'active_workers': active}

        try:
            if batches:
                snapshot = self.capacity()
                memory_budget.require_capacity(snapshot)
                record(snapshot, 1)
                job(0)  # Load the host graph before admitting concurrent input preparation.
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                pending, next_index = set(), 1
                while not errors and (pending or next_index < len(batches)):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        cancel.set()
                        raise ModelExecutionError('Descriptor wall budget exhausted while waiting for batches/memory')
                    snapshot = self.capacity(len(pending))
                    count = min(snapshot['available_workers'], len(batches) - next_index)
                    if count:
                        record(snapshot, len(pending) + count)
                        for _ in range(count):
                            pending.add(pool.submit(job, next_index))
                            next_index += 1
                    elif not pending and next_index < len(batches):
                        memory_budget.require_capacity(snapshot)
                    if pending:
                        _, pending = wait(pending, timeout=min(.2, remaining), return_when=FIRST_COMPLETED)
        finally:
            cancel.set()
            entry.release()
        if errors:
            index, exc = sorted(errors, key=lambda item: item[0])[0]
            raise ModelExecutionError(f'Descriptor batch {index} failed: {type(exc).__name__}: {exc}') from exc
        if any(result is None for result in results):
            raise ModelExecutionError('Descriptor computation was cancelled before every batch completed')
        values = np.concatenate([result[0] for result in results]) if results else np.zeros((0, len(spec['feature_names'])))
        worker_seconds = float(sum(result[1] for result in results))
        images = self._image_summary([timing for timing in timings if timing])
        self._trace_images('compute', table, rows, images)
        manifest = {'identity': identity, 'rows': int(len(rows)), 'batches': len(batches), 'workers': peak_workers,
                    'worker_limit': self.workers, 'requested_workers': self.requested_workers,
                    'worker_memory_mb': self.memory_mb, 'memory_admission': admission_records,
                    'wall_seconds': time.monotonic() - started, 'worker_seconds_total': worker_seconds,
                    'chunk_loads': entry.chunk_loads - loads_before, **images,
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
                'worker_seconds': worker_seconds, 'batches': len(batches), 'workers': peak_workers,
                'manifest': manifest}

    def plan_rows(self, table, count=64):
        """A contiguous block of the rows compute would process: GT-labeled rows in table order
        (the judgeable scope) from the middle of the pool. Uniform random rows scatter image reads
        over the whole brain and overstated the per-row cost (2026-11-03)."""
        entry = self.contexts.entry(table)
        candidates = np.sort(np.asarray(entry.rows, dtype=np.int64))
        truth = getattr(table, 'truth', None)
        if truth is not None:
            labeled = candidates[np.isin(candidates, labeled_rows(truth))]
            if len(labeled) >= count:
                candidates = labeled
        start = max(0, (len(candidates) - count) // 2)
        return candidates[start:start + count]

    def graph_load_seconds(self, table):
        """Time the host fragment graph load a brain's first batch pays.

        The load is charged once per brain, apart from the per-row rate: a 64-row sample that
        includes a 100 s graph load reads as 1.5 s per row. A graph that is already resident
        costs nothing now, so the last measured load of that brain is kept instead."""
        store = getattr(getattr(table, 'local_context', None), 'store', None)
        if store is None:
            return 0.
        key = str(table.meta.get('provenance', {}).get('brain', id(table)))
        started = time.monotonic()
        store.graph(table.meta['provenance'])
        seconds = time.monotonic() - started
        # A preceding probe/compute may have loaded it; retain that cold cost for
        # planning and frozen inference, even when resident-first ordering saves IO now.
        from .local_context import FragmentStore
        if isinstance(store, FragmentStore):
            seconds = max(seconds, store.load_seconds)
        if seconds >= .5:
            self._graph_loads[key] = seconds
        return float(self._graph_loads.get(key, seconds))

    def plan(self, table, program, spec, *, k, sample_rows=64, timeout=600):
        """Estimate this brain only; the session aggregates every TRAIN brain's measured cost.

        Image descriptors are credited with at most IMAGE_IO_PARALLELISM concurrent readers: the
        S3 patch reads, not the workers, bound them (plans/on_demand_contexts.md).

        One deadline covers the graph load and the probe: the caller's allowance is checked before
        the load (an exhausted budget loads nothing) and the load's time is deducted from the probe."""
        if timeout <= 0:
            raise ModelExecutionError('Descriptor wall budget exhausted before plan started')
        deadline = time.monotonic() + float(timeout)
        entry = self.contexts.entry(table)
        load = self.graph_load_seconds(table)
        rows = self.plan_rows(table, sample_rows)
        with tempfile.TemporaryDirectory(prefix='descriptor_plan_') as tmp:
            measured = self.probe(table, program, spec, rows, timeout=deadline - time.monotonic(), log_dir=Path(tmp))
        values, wall = measured['values'], measured['wall_seconds']
        per_row = wall / max(1, len(rows))
        capacity = self.capacity()
        effective_workers = capacity['available_workers']
        workers = effective_workers if spec['inputs'] == 'geometry' else min(effective_workers, IMAGE_IO_PARALLELISM)
        estimates = {}
        for scope in getattr(self, 'allowed_scopes', SCOPES):
            count = len(self.scope_rows(entry, scope, k, truth=table.truth))
            batches = -(-count // BATCH_ROWS)
            # Never claim parallelism from more workers than actual row batches.
            seconds = load + self.estimated_seconds(count, per_row, workers) if count else 0.
            estimates[scope] = {'rows': int(count), 'batches': int(batches),
                                'estimated_seconds': float(seconds), 'memory_feasible': effective_workers > 0}
        return {'sample_rows': int(len(rows)), 'sample_wall_seconds': wall, 'seconds_per_row_single_worker': per_row,
                'graph_load_seconds': float(load),
                'preparation_seconds': measured['preparation_seconds'], 'worker_seconds': measured['worker_seconds'],
                'image_reads': int(measured.get('image_reads', 0)), 'image_seconds': float(measured.get('image_seconds', 0.)),
                'workers': workers, 'admitted_workers': effective_workers, 'worker_limit': self.workers,
                'requested_workers': self.requested_workers, 'memory': capacity, 'estimates': estimates,
                'finite_fraction': {name: float(np.isfinite(values[:, j]).mean()) if len(rows) else None
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
        """Legacy offline replay only; evolution never registers this band indicator."""
        if CONTEXT_COLUMN in table.features.columns:
            return
        available = np.zeros(len(table.features))  # 1 inside the detector band: a function of detector rank only
        available[np.asarray(entry.band, dtype=int)] = 1.
        table.features[CONTEXT_COLUMN] = available

    @staticmethod
    def summarize(table, columns, rows, values, *, labels=True):
        """Host-only distribution summary; TRAIN labels are used here and nowhere upstream."""
        rows = np.asarray(rows, dtype=int)
        truth = np.asarray(table.truth, dtype=float) if labels else None
        result = {}
        for position, column in enumerate(columns):
            series = values[:, position]
            finite = np.isfinite(series)
            present = series[finite]
            def number(value):
                return float(format(value, '.6g')) if np.isfinite(value) else None
            scale = float(np.max(np.abs(present))) if len(present) else 0.
            entry = {'computed_rows': int(len(rows)), 'pool_rows': int(len(table.features)),
                     'finite_count': int(finite.sum()), 'missing_count': int((~finite).sum()),
                     'finite_fraction': float(finite.mean()) if len(series) else None,
                     'nonzero_count': int(np.count_nonzero(present)),
                     'nonzero_fraction': float(np.count_nonzero(present) / len(present)) if len(present) else None,
                     'min': number(present.min()) if len(present) else None,
                     'max': number(present.max()) if len(present) else None,
                     'std': number(np.std(present / scale) * scale) if scale else (0. if len(present) else None),
                     'constant': bool(present.min() == present.max()) if len(present) else None,
                     'q25_q50_q75': [number(v) for v in np.quantile(present, [.25, .5, .75])] if len(present) else None}
            if labels:
                entry['unlabeled_count'] = int(np.isnan(truth[rows]).sum())
                for label in (0, 1):
                    mask = truth[rows] == label
                    selected = series[finite & mask]
                    entry[str(label)] = {'count': int(mask.sum()), 'finite': int((finite & mask).sum()),
                                         'q25_q50_q75': [number(v) for v in np.quantile(selected, [.25, .5, .75])]
                                         if len(selected) else None}
            result[column] = entry
        return result
