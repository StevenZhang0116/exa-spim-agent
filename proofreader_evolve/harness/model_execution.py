"""Host transport for arbitrary model programs; all executable artifacts stay opaque."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .classifier_contract import MAX_ARTIFACT_BYTES


class ModelExecutionError(RuntimeError):
    pass


def run_worker(root, request, *, log_path=None):
    root = Path(root).resolve()
    scratch = root / 'scratch'
    scratch.mkdir()
    (root / 'request.json').write_text(json.dumps(request, allow_nan=False))
    threads = str(request.get('threads', 1))
    env = {name: threads for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    env.update(HOME=str(scratch), TMPDIR=str(scratch), XDG_CACHE_HOME=str(scratch),
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
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
        raise ModelExecutionError(f'Model worker exited {process.returncode} without status: {detail}') from exc
    if process.returncode or status.get('status') != 'ok':
        raise ModelExecutionError(f"{status.get('error', 'Model worker failed')}; frames={status.get('frames', [])}")
    with log_path.open('rb') as stream:
        stream.seek(max(0, log_path.stat().st_size - 8000))
        status['training_output_tail'] = stream.read(8000).decode(errors='replace')
    return status


def predict_model(model, store, frame, kind, timeout=120, memory_mb=8192):
    import tempfile
    import numpy as np
    from .classifier_contract import verify_artifacts
    if kind != model['kind']:
        raise ModelExecutionError('Trained-model kind does not match prediction request')
    artifacts = verify_artifacts(model, store)
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
