"""Feature-only worker transport with hard wall time and bounded output files."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


class ScorerExecutionError(ValueError):
    pass


def score(source, frame, kind, timeout=120, memory_mb=8192, *, artifact_store=None):
    from .classifier_contract import frozen_model, MODEL_VERSION
    from .model_execution import predict_model, ModelExecutionError
    try:
        model = frozen_model(source)
        if model is not None and model["version"] == MODEL_VERSION:
            return predict_model(model, artifact_store, frame, kind, timeout, memory_mb)
    except (ModelExecutionError, ValueError, SyntaxError) as exc:
        raise ScorerExecutionError(str(exc)) from exc
    worker = Path(__file__).with_name('scorer_worker.py')
    with tempfile.TemporaryDirectory(prefix='scorer_') as tmp:
        root = Path(tmp)
        (root / 'scorer.py').write_text(source)
        np.save(root / 'features.npy', frame.to_numpy(dtype=float), allow_pickle=False)
        (root / 'request.json').write_text(json.dumps({
            'columns': list(frame.columns), 'kind': kind, 'timeout': timeout,
            'memory_bytes': memory_mb * 1024 * 1024}))
        # Never inherit API keys, home settings, Python path or parent's file FDs.
        env = {name: '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                     'NUMEXPR_NUM_THREADS')}
        with (root / 'worker.log').open('wb') as log:
            try:
                result = subprocess.run([sys.executable, '-I', str(worker), str(root)],
                                        cwd=root, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=log, timeout=timeout, close_fds=True)
            except subprocess.TimeoutExpired as exc:
                raise ScorerExecutionError(f'Scorer exceeded {timeout:g}s wall-clock budget') from exc
        status_path = root / 'status.json'
        try:
            with status_path.open('rb') as stream:
                status = json.loads(stream.read(16384))
        except (OSError, ValueError):
            with (root / 'worker.log').open('rb') as stream:
                detail = stream.read(2000).decode(errors='replace')
            raise ScorerExecutionError(f'Scorer worker exited {result.returncode}: {detail}')
        if status['status'] != 'ok' or result.returncode:
            raise ScorerExecutionError(status.get('error', f'Worker exit {result.returncode}')
                                       + f"; frames={status.get('frames', [])}")
        scores = np.load(root / 'scores.npy', allow_pickle=False)
        if scores.shape != (len(frame),) or not np.isfinite(scores).all():
            raise ScorerExecutionError('Invalid worker scores')
        return scores


def preflight(source, frame, kind, timeout=120, *, artifact_store=None):
    """Real schema plus missing predictor values; no evaluator labels are sent."""
    import pandas as pd
    indices = np.linspace(0, len(frame) - 1, min(16, len(frame)), dtype=int)
    sample = frame.iloc[indices].copy()
    missing = pd.DataFrame(np.nan, index=[0], columns=frame.columns)
    missing['detector_score'] = 0.0
    sample = pd.concat([sample, missing], ignore_index=True)
    return score(source, sample, kind, timeout, artifact_store=artifact_store)
