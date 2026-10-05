"""Fit, predict, extract features or analyze 3D volumes after OS isolation."""
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


class ImageDataset:
    """Read-only, row-aligned local patches; no cloud access or external IDs."""
    def __init__(self, root, np):
        self.root, self.np = root / 'images', np
        self.index = json.loads((self.root / 'index.json').read_text())

    def __len__(self):
        return self.index['rows']

    def __getitem__(self, row):
        row = int(row)
        if not 0 <= row < len(self):
            raise IndexError(row)
        results = []
        for name in self.index['patches'].get(str(row), []):
            with self.np.load(self.root / name, allow_pickle=False) as data:
                patch = {key: data[key] for key in data.files}
            for values in patch.values():
                values.flags.writeable = False
            results.append(patch)
        return results


def extract_image(module, x, images, request, output, np):
    names = request['feature_names']
    values = np.empty((len(x), len(names)), dtype=np.float64)
    for index in range(len(x)):
        results = []
        for _ in range(2):
            context = {'features': {name: float(value) if np.isfinite(value) else None
                                   for name, value in x.iloc[index].items()}, 'patches': images[index]}
            result = module.extract_image_features(context)
            if not isinstance(result, dict) or set(result) != set(names):
                raise ValueError('extract_image_features must return exactly the declared feature_names')
            row = np.asarray([result[name] for name in names], dtype=np.float64)
            if row.shape != (len(names),) or np.isinf(row).any():
                raise ValueError('Image features must be scalars or NaN, not infinity')
            results.append(row)
        if not np.array_equal(*results, equal_nan=True):
            raise ValueError('Image feature extraction must be deterministic')
        values[index] = results[0]
    np.save(output, values, allow_pickle=False)
    output.flush()


def analyze_volume(module, contexts, images, request, output, np):
    """No projected image is involved; the program receives actual 3D arrays."""
    results = []
    for index, description in enumerate(contexts):
        patches = images[index]
        if len(patches) != 1:
            raise ValueError('A volume analysis context must contain one native occurrence patch')
        context = copy.deepcopy(description)
        context.update(patches[0])
        fragment = context['fragment']
        for name, dtype in (('xyz_um', float), ('nodes_zyx', float), ('radius_um', float),
                            ('degree', int), ('segment', int), ('edges', int), ('anchor_nodes', int),
                            ('outside_radius', bool), ('inside_patch', bool)):
            values = np.asarray(fragment[name], dtype=dtype)
            if name == 'edges':
                values = values.reshape(-1, 2)
            values.flags.writeable = False
            fragment[name] = values
        result = module.analyze(context)
        if not isinstance(result, dict):
            raise ValueError('analyze(context) must return a JSON object of findings or measurements')
        results.append(result)
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported analysis result type: {type(value).__name__}')
    encoded = json.dumps(results, default=convert, allow_nan=False).encode('utf-8')
    if len(encoded) > request['max_result_bytes']:
        raise ValueError('Analysis results exceed 24 KiB; return summaries, not volume arrays')
    output.write(encoded)
    output.flush()


def describe_rows(module, contexts, images, request, output, np):
    """Agent descriptor over cached contexts: one finite-or-NaN row per context, deterministic."""
    names = request['feature_names']
    values = np.empty((len(contexts), len(names)), dtype=np.float64)
    for index, description in enumerate(contexts):
        context = copy.deepcopy(description)
        if images is not None:
            patches = images[index]
            if len(patches) > 1:
                raise ValueError('A descriptor context carries at most one patch')
            if patches:
                context.update(patches[0])
        fragment = context['fragment']
        for name, dtype in (('xyz_um', float), ('radius_um', float), ('degree', int), ('segment', int),
                            ('edges', int), ('anchor_nodes', int), ('outside_radius', bool),
                            ('nodes_zyx', float), ('inside_patch', bool)):
            if name not in fragment:
                continue
            array = np.asarray(fragment[name], dtype=dtype)
            if name == 'edges':
                array = array.reshape(-1, 2)
            if name == 'nodes_zyx':
                array = array.reshape(-1, 3)
            array.flags.writeable = False
            fragment[name] = array
        results = []
        for _ in range(2):
            result = module.describe(copy.deepcopy(context))
            if not isinstance(result, dict) or set(result) != set(names):
                raise ValueError('describe(context) must return a dict with exactly the declared feature_names')
            row = np.asarray([result[name] for name in names], dtype=np.float64)
            if row.shape != (len(names),) or np.isinf(row).any():
                raise ValueError('Descriptor values must be finite scalars or NaN, never infinity')
            results.append(row)
        if not np.array_equal(*results, equal_nan=True):
            raise ValueError('describe(context) must be deterministic')
        values[index] = results[0]
    np.save(output, values, allow_pickle=False)
    output.flush()


def extract(module, root, request, output, np):
    names = request['feature_names']
    values = np.empty((request['rows'], len(names)), dtype=np.float64)
    count = 0
    with (root / 'contexts.jsonl').open() as stream:
        for index, line in enumerate(stream):
            if index >= len(values):
                raise ValueError('Too many local contexts')
            context = json.loads(line)
            results = []
            for _ in range(2):
                result = module.extract_local_features(copy.deepcopy(context))
                if not isinstance(result, dict) or set(result) != set(names):
                    raise ValueError('extract_local_features must return exactly the declared feature_names')
                row = np.asarray([result[name] for name in names], dtype=np.float64)
                if row.shape != (len(names),) or np.isinf(row).any():
                    raise ValueError('Local features must be scalar numbers or NaN, never infinity')
                results.append(row)
            if not np.array_equal(*results, equal_nan=True):
                raise ValueError('Local feature extraction must be deterministic')
            values[index] = results[0]
            count += 1
    if count != len(values):
        raise ValueError('Missing local contexts')
    np.save(output, values, allow_pickle=False)
    output.flush()


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
    output_name = {'predict': 'scores.npy', 'extract': 'local_features.npy',
                   'extract_image': 'local_features.npy', 'analyze_volume': 'analysis.json',
                   'describe': 'descriptors.npy'}.get(request['mode'])
    output = (root / output_name).open('wb') if output_name else None
    artifacts, scratch = root / 'artifacts', root / 'scratch'
    started = time.monotonic()
    try:
        if request['mode'] not in ('fit', 'predict', 'extract', 'extract_image', 'analyze_volume', 'describe'):
            raise ValueError('Unknown model worker mode')
        reads = ([root / 'training.py', root / 'contexts.jsonl'] if request['mode'] == 'extract'
                 else [root / 'training.py', root / 'X.npy', artifacts])
        if request['mode'] in ('analyze_volume', 'describe'):
            reads = [root / 'training.py', root / 'volume_contexts.json']
        if request['mode'] == 'fit':
            reads.append(root / 'y.npy')
            if (root / 'groups.npy').exists():
                reads.append(root / 'groups.npy')
        if request.get('image_inputs'):
            reads.append(root / 'images')
        sandbox.isolate(reads, [scratch, artifacts] if request['mode'] == 'fit' else [scratch])
        import numpy as np
        import pandas as pd
        if 'random_seed' in request:
            import random
            random.seed(request['random_seed'])
            np.random.seed(request['random_seed'])
        if request['mode'] in ('analyze_volume', 'describe'):
            contexts = json.loads((root / 'volume_contexts.json').read_text())
        elif request['mode'] != 'extract':
            x = pd.DataFrame(np.load(root / 'X.npy', allow_pickle=False), columns=request['columns'])
            params = request['config']['parameters']
        images = ImageDataset(root, np) if request.get('image_inputs') else None
        expected_rows = (len(contexts) if request['mode'] in ('analyze_volume', 'describe')
                         else (len(x) if images is not None else 0))
        if images is not None and len(images) != expected_rows:
            raise ValueError('Image inputs are not aligned with feature rows')
        # Stable module name supports pickle/joblib of agent-defined classes.
        module = types.ModuleType('agent_model')
        module.__file__ = str(root / 'training.py')
        sys.modules['agent_model'] = module
        exec(compile(program, module.__file__, 'exec'), module.__dict__)
        if request['mode'] == 'extract':
            extract(module, root, request, output, np)
        elif request['mode'] == 'extract_image':
            extract_image(module, x, images, request, output, np)
        elif request['mode'] == 'analyze_volume':
            analyze_volume(module, contexts, images, request, output, np)
        elif request['mode'] == 'describe':
            describe_rows(module, contexts, images, request, output, np)
        elif request['mode'] == 'fit':
            y = np.load(root / 'y.npy', allow_pickle=False)
            if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
                raise ValueError('Invalid TRAIN label alignment')
            extra = {'images': images} if images is not None else {}
            # Optional, label-free source-brain codes: passed only to programs that
            # declare a `groups` parameter, so existing fit signatures keep working.
            if 'group_names' in request and (root / 'groups.npy').exists():
                import inspect
                parameters = inspect.signature(module.fit).parameters
                if 'groups' in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
                    codes = np.load(root / 'groups.npy', allow_pickle=False)
                    if codes.shape != (len(x),):
                        raise ValueError('Invalid TRAIN group alignment')
                    names = np.asarray(request['group_names'], dtype=str)
                    extra['groups'] = pd.Series(names[codes], name='train_brain')
            module.fit(x, y, artifacts, copy.deepcopy(params), **extra)
        else:
            values = []
            for _ in range(2):
                prediction = np.asarray(module.predict(x.copy(deep=True), artifacts,
                    copy.deepcopy(params), **({'images': images} if images is not None else {})), dtype=float).copy()
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
