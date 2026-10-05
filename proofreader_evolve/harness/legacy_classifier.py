"""Read/validate v1 NumPy exports for resume; new fits use classifier_contract.

RUNTIME_PATH is the byte-preserved v1 source template: frozen_model compares the
entire historical export, including that template's comments and docstring.
"""

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


MODEL_VERSION = 'train-classifier-v1'
MAX_MODEL_SOURCE_BYTES = 2_000_000
DEFAULTS = {
    'logistic_regression': {'C': 1.0, 'max_iter': 500, 'class_weight': 'balanced'},
    'random_forest': {'n_estimators': 32, 'max_depth': 4, 'min_samples_leaf': 10,
                      'max_features': 'sqrt', 'class_weight': 'balanced'},
}
RUNTIME_PATH = Path(__file__).with_name('frozen_classifier_runtime.py')


def finite(value):
    return type(value) in (int, float) and abs(value) <= 1e100 and math.isfinite(value)


def normalize_config(spec):
    if not isinstance(spec, dict) or set(spec) - {'algorithm', 'features', 'parameters'}:
        raise ValueError('classifier must contain algorithm, optional features and parameters only')
    algorithm = spec.get('algorithm')
    if not isinstance(algorithm, str) or algorithm not in DEFAULTS:
        raise ValueError('classifier.algorithm must be logistic_regression or random_forest')
    features = spec.get('features')
    if features is not None:
        if (not isinstance(features, list) or not 1 <= len(features) <= 256
                or any(not isinstance(f, str) or not f or len(f) > 256 for f in features)
                or len(set(features)) != len(features)):
            raise ValueError('classifier.features must list 1–256 distinct predictor column names')
    supplied = spec.get('parameters', {})
    if not isinstance(supplied, dict) or set(supplied) - DEFAULTS[algorithm].keys():
        raise ValueError(f'Allowed {algorithm} parameters: {sorted(DEFAULTS[algorithm])}')
    parameters = {**DEFAULTS[algorithm], **supplied}
    if parameters['class_weight'] not in (None, 'balanced'):
        raise ValueError('class_weight must be null or "balanced"')
    bounds = ({'C': (1e-6, 1e6, False), 'max_iter': (50, 2000, True)} if algorithm == 'logistic_regression'
              else {'n_estimators': (1, 64, True), 'max_depth': (1, 6, True),
                    'min_samples_leaf': (1, 10000, True)})
    for key, (lower, upper, integer) in bounds.items():
        value = parameters[key]
        if not finite(value) or not lower <= value <= upper or (integer and int(value) != value):
            raise ValueError(f'{key} must be {"an integer" if integer else "a number"} in [{lower}, {upper}]')
        parameters[key] = int(value) if integer else float(value)
    if algorithm == 'random_forest':
        value = parameters['max_features']
        if value not in ('sqrt', 'log2') and not (finite(value) and 0 < value <= 1):
            raise ValueError('max_features must be "sqrt", "log2", or a fraction in (0, 1]')
        if finite(value):
            parameters['max_features'] = float(value)  # 1 means 100%, not sklearn's integer one-column setting.
    return {'algorithm': algorithm, 'features': deepcopy(features), 'parameters': parameters}


def config_identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def classifier_info(config):
    numeric = {k: v for k, v in config['parameters'].items() if finite(v)}
    structure = {**config, 'parameters': {k: '<numeric>' if k in numeric else v
                                         for k, v in config['parameters'].items()}}
    return {'candidate_type': 'classifier', 'classifier': deepcopy(config), 'parameters': numeric,
            'formula_sha256': config_identity({'classifier_version': MODEL_VERSION, **structure})}


def validate_model(model):
    if (not isinstance(model, dict) or model.get('version') != MODEL_VERSION
            or model.get('kind') not in ('merge', 'split')):
        raise ValueError('Invalid frozen classifier version/kind')
    config = normalize_config(model['config'])
    if config != model['config'] or config['features'] is None:
        raise ValueError('Frozen classifier needs canonical fitted feature names/configuration')
    brains = model['training_brains']
    if (not isinstance(brains, list) or not brains
            or any(not isinstance(b, str) or not b.isascii() or not b.isdigit() for b in brains)):
        raise ValueError('Frozen classifier must record its TRAIN brain provenance')
    if type(model['training_rows']) is not int or model['training_rows'] < 2:
        raise ValueError('Invalid frozen classifier training row count')
    fingerprint = model['training_fingerprint']
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
        raise ValueError('Invalid frozen classifier TRAIN fingerprint')
    width = len(config['features'])
    def vector(values, length):
        return isinstance(values, list) and len(values) == length and all(finite(v) for v in values)
    if not all(vector(model[key], width) for key in ('impute', 'mean', 'scale')) or any(v <= 0 for v in model['scale']):
        raise ValueError('Invalid frozen preprocessing arrays')
    if config['algorithm'] == 'logistic_regression':
        if not vector(model['coef'], width) or not finite(model['intercept']):
            raise ValueError('Invalid frozen logistic coefficients')
    else:
        trees = model['trees']
        if not isinstance(trees, list) or len(trees) != config['parameters']['n_estimators']:
            raise ValueError('Invalid frozen forest size')
        for tree in trees:
            n = len(tree['left'])
            if not 1 <= n <= 2 ** (config['parameters']['max_depth'] + 1) - 1:
                raise ValueError('Invalid frozen tree size')
            if not all(vector(tree[key], n) for key in ('left', 'right', 'feature', 'threshold', 'probability')):
                raise ValueError('Invalid frozen tree arrays')
            for i, (left, right, feature, probability) in enumerate(zip(
                    tree['left'], tree['right'], tree['feature'], tree['probability'])):
                if not 0 <= probability <= 1 or any(type(v) is not int for v in (left, right, feature)):
                    raise ValueError('Invalid tree index/probability')
                if not (left == right == -1 or (i < left < n and i < right < n and 0 <= feature < width)):
                    raise ValueError('Tree children must point forward; no cycles or invalid features')
    json.dumps(model, allow_nan=False)


def render_model(model):
    validate_model(model)
    source = ('# Frozen classifier fitted by the harness on TRAIN only. Do not edit fitted arrays.\n'
              f'_FROZEN_CLASSIFIER = {model!r}\n' + RUNTIME_PATH.read_text())
    if len(source.encode()) > MAX_MODEL_SOURCE_BYTES:
        raise ValueError('Frozen classifier exceeds the 2 MB source limit; reduce model size')
    return source


def frozen_model(source, tree=None):
    tree = tree if tree is not None else ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_FROZEN_CLASSIFIER' for t in node.targets)]
    if not nodes:
        return None
    if len(nodes) != 1:
        raise ValueError('Invalid frozen classifier source')
    model = ast.literal_eval(nodes[0].value)
    if render_model(model) != source:
        raise ValueError('Frozen classifier source was edited; refit using train_classifier()')
    return model
