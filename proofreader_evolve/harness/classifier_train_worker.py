"""Fit or predict agent programs after OS isolation; never receives validation labels."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback
import types


def main():
    root = Path(sys.argv[1]).resolve()
    request = json.loads((root / 'request.json').read_text())
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (request['memory_mb'] * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(request['timeout']) + 1,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2,) * 2)
    # Load only this trusted sandbox module before restricting filesystem access.
    spec = importlib.util.spec_from_file_location('_model_sandbox', Path(__file__).with_name('model_sandbox.py'))
    sandbox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sandbox)
    program = (root / 'training.py').read_text()
    status = (root / 'status.json').open('w')
    output = (root / 'scores.npy').open('wb') if request['mode'] == 'predict' else None
    artifacts, scratch = root / 'artifacts', root / 'scratch'
    started = time.monotonic()
    try:
        reads = [root / 'training.py', root / 'X.npy', artifacts]
        if request['mode'] == 'fit':
            reads.append(root / 'y.npy')
        sandbox.isolate(reads, [scratch, artifacts] if request['mode'] == 'fit' else [scratch])
        import numpy as np
        import pandas as pd
        x = pd.DataFrame(np.load(root / 'X.npy', allow_pickle=False), columns=request['columns'])
        params = request['config']['parameters']
        # Stable module name supports pickle/joblib of agent-defined classes.
        module = types.ModuleType('agent_model')
        module.__file__ = str(root / 'training.py')
        sys.modules['agent_model'] = module
        exec(compile(program, module.__file__, 'exec'), module.__dict__)
        if request['mode'] == 'fit':
            y = np.load(root / 'y.npy', allow_pickle=False)
            if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
                raise ValueError('Invalid TRAIN label alignment')
            module.fit(x, y, artifacts, copy.deepcopy(params))
        else:
            values = []
            for _ in range(2):
                prediction = np.asarray(module.predict(x.copy(deep=True), artifacts,
                                                       copy.deepcopy(params)), dtype=float).copy()
                if prediction.shape != (len(x),) or not np.isfinite(prediction).all():
                    raise ValueError('predict must return one finite score per row in original order')
                values.append(prediction)
            if not np.array_equal(*values):
                raise ValueError('predict must be deterministic with frozen artifacts')
            np.save(output, values[0], allow_pickle=False)
            output.flush()
        result = {'status': 'ok', 'wall_seconds': time.monotonic() - started,
                  'versions': {name: str(getattr(mod, '__version__', 'unknown'))[:100]
                               for name, mod in list(sys.modules.items())
                               if name in ('numpy', 'pandas', 'sklearn', 'torch', 'xgboost', 'lightgbm', 'scipy')}}
    except BaseException as exc:
        frames = traceback.extract_tb(exc.__traceback__, limit=12)
        result = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'[:4000],
                  'frames': [{'function': f.name, 'line': f.lineno} for f in frames],
                  'wall_seconds': time.monotonic() - started}
    json.dump(result, status, allow_nan=False)
    status.close()
    if output:
        output.close()
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
