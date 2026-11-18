"""Host transport for arbitrary model programs; all executable artifacts stay opaque."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .classifier_contract import MAX_ARTIFACT_BYTES


DEFAULT_MEMORY_MB = 16384  # 8192 until 2026-10-08; raised with --classifier-memory-mb after 16-member ensembles were killed


class ModelExecutionError(RuntimeError):
    pass


def exit_detail(returncode, request, tail):
    """SIGKILL alone does not identify the cause; the message lists the worker's limits and what
    each does when it is reached, so that the reader does not chase the wrong one."""
    if returncode == -9:
        threads = max(1, int(request.get('threads', 1)))
        timeout = request.get('timeout')
        cpu = f'{math.ceil(timeout) * threads + 1} s' if timeout is not None else 'timeout x threads'
        return (f"killed by SIGKILL; cause unconfirmed. Worker limits: CPU time {cpu} summed over its {threads} "
                f"thread(s) (reaching it kills the worker like this); address space {request.get('memory_mb')} MiB "
                "(exceeding it raises MemoryError, it does not kill). Otherwise check Slurm accounting for an "
                "out-of-memory kill of the job or a cancellation. "
                + tail)
    return tail


# GPU mode (2026-11-17): a run started with --gpu lets fit and predict workers use the GPU of the
# allocation. Every other worker (descriptors, feature extraction, 3D analysis) stays on the
# CPU: dozens of them run side by side. One switch here covers every caller of run_worker:
# fold fits and predictions, the full fit, ablations, validation and restored candidates.
GPU_MODES = ('fit', 'predict')
_GPU = {'enabled': False}


def set_gpu(enabled):
    _GPU['enabled'] = bool(enabled)


def gpu_enabled():
    return _GPU['enabled']


def gpu_device():
    """The GPU this process may use, as the scheduler exposes it; raises when there is none."""
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('--gpu was requested but CUDA_VISIBLE_DEVICES is empty: no GPU is allocated to this step')
    try:
        listing = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=60, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f'--gpu was requested but nvidia-smi could not list a device: {exc}') from exc
    if not listing:
        raise ValueError('--gpu was requested but nvidia-smi lists no device')
    name, memory = (part.strip() for part in listing[0].rsplit(',', 1))
    return {'name': name, 'memory_mb': int(float(memory)), 'devices': len(listing),
            'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}


GPU_CHECK_PROGRAM = '''import json
import torch


def fit(X_train, y_train, artifact_dir, params):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available inside the sandboxed worker')
    device = torch.device('cuda')
    left = torch.arange(6., device=device).reshape(2, 3)
    value = float((left @ left.T).sum().cpu())
    torch.cuda.synchronize()
    if value != 83.:  # [[0, 1, 2], [3, 4, 5]] times its transpose sums to 83
        raise RuntimeError(f'CUDA returned {value} for a product that is 83')
    properties = torch.cuda.get_device_properties(0)
    (artifact_dir / 'device.json').write_text(json.dumps({
        'name': properties.name, 'memory_mb': int(properties.total_memory // 2**20),
        'torch': torch.__version__, 'cuda': torch.version.cuda}))


def predict(X, artifact_dir, params):
    return [0. for _ in range(len(X))]
'''


def gpu_worker_check(*, timeout=300., memory_mb=DEFAULT_MEMORY_MB, threads=1):
    """Run one small CUDA computation in a real sandboxed fit worker, with the limits a fit of
    this run gets, and return the device it used. `gpu_device` shows that the scheduler exposes
    a GPU; only this shows that CUDA works inside the sandbox on this node. Call it before any
    data is loaded; GPU mode must be on."""
    import tempfile
    import numpy as np
    if not _GPU['enabled']:
        raise ValueError('gpu_worker_check needs GPU mode (set_gpu(True))')
    with tempfile.TemporaryDirectory(prefix='gpu_check_') as tmp:
        root = Path(tmp)
        np.save(root / 'X.npy', np.zeros((2, 1)), allow_pickle=False)
        np.save(root / 'y.npy', np.array([0, 1], dtype=np.int8), allow_pickle=False)
        (root / 'training.py').write_text(GPU_CHECK_PROGRAM)
        (root / 'artifacts').mkdir()
        request = {'mode': 'fit', 'columns': ['x'], 'config': {'parameters': {}}, 'group_names': ['check'],
                   'timeout': float(timeout), 'memory_mb': int(memory_mb), 'threads': int(threads)}
        try:
            run_worker(root, request)
            return json.loads((root / 'artifacts' / 'device.json').read_text())
        except (ModelExecutionError, OSError, ValueError) as exc:
            log = root / 'worker.log'
            tail = log.read_text(errors='replace')[-1500:].strip() if log.is_file() else ''
            raise ValueError('--gpu was requested but a sandboxed worker could not use CUDA: '
                             f'{type(exc).__name__}: {exc}' + (f'\nworker log: {tail}' if tail else '')) from exc


def run_worker(root, request, *, log_path=None):
    request = dict(request)
    request['gpu'] = bool(_GPU['enabled'] and request.get('mode') in GPU_MODES)
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    request['threads'] = min(max(1, int(request.get('threads', 1))), max(1, cpus))
    root = Path(root).resolve()
    scratch = root / 'scratch'
    scratch.mkdir()
    (root / 'request.json').write_text(json.dumps(request, allow_nan=False))
    threads = str(request.get('threads', 1))
    env = {name: threads for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    env.update(HOME=str(scratch), TMPDIR=str(scratch), XDG_CACHE_HOME=str(scratch),
               CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '') if request['gpu'] else '',
               PYTHONDONTWRITEBYTECODE='1')
    worker = Path(__file__).with_name('classifier_train_worker.py')
    log_path = Path(log_path) if log_path else root / 'worker.log'
    started = time.monotonic()
    with log_path.open('wb') as log:
        process = subprocess.Popen([sys.executable, '-I', '-B', str(worker), str(root)], cwd=root,
                                   env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > request['timeout']:
                    raise ModelExecutionError(f"Model {request['mode']} exceeded {request['timeout']:g}s")
                total, count = 0, 0
                for directory in (root / 'artifacts', scratch):
                    for parent, dirs, files in os.walk(directory, followlinks=False):
                        count += len(dirs) + len(files)
                        if count > 2048:
                            raise ModelExecutionError('Model output/scratch has too many entries')
                        for name in files:
                            try:
                                total += (Path(parent) / name).lstat().st_size
                            except FileNotFoundError:  # Temporary files may be removed by a library.
                                pass
                        if total > 2 * MAX_ARTIFACT_BYTES:
                            raise ModelExecutionError('Model output/scratch exceeds 256 MiB')
                if log_path.stat().st_size > 4 * 1024**2:
                    raise ModelExecutionError('Model worker log exceeds 4 MiB')
                try:
                    process.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()  # seccomp forbids child processes; all worker threads exit here.
    try:
        with (root / 'status.json').open('rb') as stream:
            status = json.loads(stream.read(32_768))
    except (OSError, ValueError) as exc:
        with log_path.open('rb') as stream:
            detail = stream.read(2000).decode(errors='replace')
        raise ModelExecutionError(f'Model worker exited {process.returncode} without status: '
                                  f'{exit_detail(process.returncode, request, detail)}') from exc
    if process.returncode or status.get('status') != 'ok':
        raise ModelExecutionError(f"{status.get('error', 'Model worker failed')}; frames={status.get('frames', [])}")
    with log_path.open('rb') as stream:
        stream.seek(max(0, log_path.stat().st_size - 8000))
        status['training_output_tail'] = stream.read(8000).decode(errors='replace')
    if not status['training_output_tail']:
        # Nothing to read later; failures keep their log, where even an empty file shows the worker ran.
        log_path.unlink(missing_ok=True)
    return status


def predict_model(model, store, frame, kind, timeout=120, memory_mb=DEFAULT_MEMORY_MB):
    import tempfile
    import numpy as np
    from .classifier_contract import verify_artifacts
    if kind != model['kind']:
        raise ModelExecutionError('Trained-model kind does not match prediction request')
    artifacts = verify_artifacts(model, store)
    columns = model.get('columns')
    if columns is not None:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ModelExecutionError(f'Prediction input lacks {len(missing)} training columns, e.g. {missing[:3]}; '
                                      'registered descriptors must be computed for this table first')
        attrs = dict(frame.attrs)
        frame = frame[list(columns)]
        frame.attrs.update(attrs)
    with tempfile.TemporaryDirectory(prefix='model_predict_') as tmp:
        root = Path(tmp)
        shutil.copytree(artifacts, root / 'artifacts')
        # Check the staged bytes too; never deserialize a changed snapshot.
        from .classifier_contract import artifact_files
        if artifact_files(root / 'artifacts') != model['files']:
            raise ModelExecutionError('Staged prediction artifacts differ from the manifest')
        (root / 'training.py').write_text(model['program'])
        np.save(root / 'X.npy', frame.to_numpy(dtype=float), allow_pickle=False)
        request = {'mode': 'predict', 'columns': list(frame.columns), 'config': model['config'],
                   'timeout': timeout, 'memory_mb': memory_mb, 'threads': model.get('threads', 1)}
        from .image_contract import image_spec
        from .image_features import stage_images
        contract = image_spec(model['program'])
        if contract is not None and contract[1]['raw_patches']:
            stage_images(root, [(frame, np.arange(len(frame)))])
            request['image_inputs'] = True
        run_worker(root, request)
        if artifact_files(root / 'artifacts') != model['files']:
            raise ModelExecutionError('Prediction modified frozen model artifacts')
        path = root / 'scores.npy'
        if path.stat().st_size > len(frame) * 8 + 4096:
            raise ModelExecutionError('Prediction output exceeds expected size')
        values = np.load(path, allow_pickle=False)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ModelExecutionError('predict must return one finite score per candidate')
        return values
