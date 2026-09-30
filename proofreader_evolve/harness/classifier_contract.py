"""Model-independent programs and artifact manifests; never deserialize models here."""
import ast
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat

MODEL_VERSION = 'agent-trained-model-v2'
MAX_PROGRAM_BYTES = 128_000
MAX_MODEL_SOURCE_BYTES = 2_000_000
MAX_ARTIFACT_BYTES = 128 * 1024**2
MAX_ARTIFACT_FILES = 256


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e100


def config_identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def normalize_config(spec, available_features=None):
    if not isinstance(spec, dict) or set(spec) - {'parameters'}:
        raise ValueError('classifier accepts only parameters; define your model and feature selection in training.py')
    parameters = spec.get('parameters', {})
    if not isinstance(parameters, dict) or any(not isinstance(k, str) for k in parameters):
        raise ValueError('classifier.parameters must be a JSON object')
    if len(json.dumps(parameters, allow_nan=False).encode()) > 32_000:
        raise ValueError('classifier.parameters exceeds 32 KB')
    return {'parameters': deepcopy(parameters)}


def validate_program(program):
    if not isinstance(program, str) or len(program.encode()) > MAX_PROGRAM_BYTES:
        raise ValueError('training.py must be Python source of at most 128 KB')
    tree = ast.parse(program)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not {'fit', 'predict'} <= names:
        raise ValueError('training.py must define fit(X_train, y_train, artifact_dir, params) '
                         'and predict(X, artifact_dir, params)')
    return tree


def classifier_info(config, program):
    tree = validate_program(program)
    config = normalize_config(config)
    numeric = {k: v for k, v in config['parameters'].items() if finite(v)}
    structure = {k: [k in numeric, None if k in numeric else v] for k, v in config['parameters'].items()}
    return {'candidate_type': 'classifier', 'classifier': config, 'parameters': numeric,
            'program_sha256': hashlib.sha256(program.encode()).hexdigest(),
            'formula_sha256': config_identity({'version': MODEL_VERSION,
                'program_ast': ast.dump(tree, include_attributes=False), 'parameters': structure})}


def classifier_configs(spec, grid, available_features=None):
    base = normalize_config(spec)
    numeric = {k for k, v in base['parameters'].items() if finite(v)}
    if set(grid) - numeric:
        raise ValueError('parameter_grid keys must be numeric entries in classifier.parameters')
    if math.prod(len(v) for v in grid.values()) > 64:
        raise ValueError('Classifier grid exceeds 64 combinations')
    configs = {}
    for values in itertools.product(*grid.values()):
        config = normalize_config({'parameters': {**base['parameters'], **dict(zip(grid, values))}})
        configs[config_identity(config)] = config
    return list(configs.values())


def model_info(model):
    if model['version'] == MODEL_VERSION:
        return classifier_info(model['config'], model['program'])
    from .legacy_classifier import classifier_info as legacy_info
    info = legacy_info(model['config'])
    info['parameters'] = {}  # Legacy exports are resumable; new fitting uses training.py.
    return info


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_model(model):
    if model.get('version') != MODEL_VERSION or model.get('kind') not in ('merge', 'split'):
        raise ValueError('Invalid trained-model version/kind')
    validate_program(model['program'])
    if normalize_config(model['config']) != model['config']:
        raise ValueError('Invalid trained-model parameters')
    if (not isinstance(model['training_brains'], list) or not model['training_brains']
            or any(not isinstance(b, str) or not b.isascii() or not b.isdigit() for b in model['training_brains'])
            or type(model['training_rows']) is not int or model['training_rows'] < 1
            or not _hex(model['training_fingerprint'])):
        raise ValueError('Invalid TRAIN provenance')
    if type(model.get('threads')) is not int or model['threads'] < 1:
        raise ValueError('Invalid numerical thread setting')
    files = model['files']
    if not isinstance(files, dict) or len(files) > MAX_ARTIFACT_FILES:
        raise ValueError('Invalid artifact file manifest')
    for name, entry in files.items():
        path = PurePosixPath(name)
        if (not name or not path.parts or path.is_absolute() or '..' in path.parts or str(path) != name
                or '\\' in name or not isinstance(entry, dict)
                or not _hex(entry.get('sha256')) or type(entry.get('bytes')) is not int or entry['bytes'] < 0):
            raise ValueError('Invalid artifact path/hash/size')
    if sum(e['bytes'] for e in files.values()) > MAX_ARTIFACT_BYTES:
        raise ValueError('Model artifacts exceed 128 MiB')
    if model['artifact_sha256'] != config_identity(files):
        raise ValueError('Artifact manifest digest differs')


def render_model(model):
    validate_model(model)
    source = ('# Harness-managed trained model; accompanying model_artifacts/ is required.\n'
              f'_TRAINED_MODEL = {model!r}\n'
              'def score_candidates(features, ctx):\n'
              '    raise RuntimeError("Load this model through the isolated model worker")\n')
    if len(source.encode()) > MAX_MODEL_SOURCE_BYTES:
        raise ValueError('Trained-model manifest is too large')
    return source


def frozen_model(source, tree=None):
    tree = ast.parse(source) if tree is None else tree
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_TRAINED_MODEL' for t in n.targets)]
    if not nodes:
        from .legacy_classifier import frozen_model as legacy_model
        return legacy_model(source, tree)
    if len(nodes) != 1:
        raise ValueError('Invalid trained-model manifest')
    model = ast.literal_eval(nodes[0].value)
    if render_model(model) != source:
        raise ValueError('Trained-model manifest was edited; call train_classifier() to create a new model')
    return model


def artifact_files(root):
    """Inspect opaque files without loading them; disallow links and special files."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Artifact root must be a regular directory')
    files, total, entries = {}, 0, 0
    for path in sorted(root.rglob('*')):
        entries += 1
        if entries > 1024:
            raise ValueError('Too many artifact directory entries')
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode) or path.stat().st_nlink != 1:
            raise ValueError('Model artifacts must be regular files, without links')
        size = path.stat().st_size
        total += size
        if total > MAX_ARTIFACT_BYTES or len(files) >= MAX_ARTIFACT_FILES:
            raise ValueError('Model artifacts exceed 128 MiB or 256 files')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                digest.update(chunk)
        files[path.relative_to(root).as_posix()] = {'sha256': digest.hexdigest(), 'bytes': size}
    return files


def verify_artifacts(model, store):
    validate_model(model)
    if store is None:
        raise ValueError('This trained model requires the run model_artifacts directory')
    root = Path(store) / model['artifact_sha256']
    if artifact_files(root) != model['files']:
        raise ValueError('Model artifact files differ from the measured snapshot')
    return root


def copy_artifacts(model, source_store, destination_store):
    source = verify_artifacts(model, source_store)
    destination = Path(destination_store) / model['artifact_sha256']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copytree(source, destination)
    verify_artifacts(model, destination_store)
