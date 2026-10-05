"""Static opt-in image contract. Agent programs never choose cloud paths."""
import ast
import math
import re

from .classifier_contract import frozen_model

IMAGE_VERSION = 'candidate-image-v1'


def image_spec(source):
    model = frozen_model(source)
    program = model.get('program', '') if model is not None else source
    tree = ast.parse(program)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'LOCAL_IMAGE' for t in n.targets)]
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_image_features']
    if not nodes and not functions:
        return None
    if len(nodes) != 1:
        raise ValueError('Define one literal LOCAL_IMAGE dictionary')
    value = ast.literal_eval(nodes[0].value)
    defaults = {'radius_um': 40., 'level': 0, 'channel': 0, 'timepoint': 0,
                'max_candidates': 32, 'max_occurrences': 1, 'selection_feature': 'detector_score',
                'selection_largest': True, 'feature_names': [], 'raw_patches': False,
                'selection_mode': 'top', 'selection_k': None,
                'batch_size': 32, 'batch_max_mb': 256, 'max_total_mb': 4096}
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError('Unknown LOCAL_IMAGE setting; cloud paths and coordinate transforms are host-owned')
    spec = {**defaults, **value}
    radius = spec['radius_um']
    if type(radius) not in (int, float) or not math.isfinite(radius) or not 4 <= radius <= 100:
        raise ValueError('LOCAL_IMAGE.radius_um must be in [4, 100]')
    for name, low, high in [('level', 0, 6), ('channel', 0, 15), ('timepoint', 0, 15),
                           ('max_candidates', 1, 4096), ('max_occurrences', 1, 4),
                           ('batch_size', 1, 256), ('batch_max_mb', 1, 1024),
                           ('max_total_mb', 1, 8192)]:
        if type(spec[name]) is not int or not low <= spec[name] <= high:
            raise ValueError(f'LOCAL_IMAGE.{name} must be an integer in [{low}, {high}]')
    if type(spec['raw_patches']) is not bool or type(spec['selection_largest']) is not bool:
        raise ValueError('raw_patches and selection_largest must be booleans')
    if not isinstance(spec['selection_feature'], str):
        raise ValueError('selection_feature must name an existing base predictor')
    if spec['selection_mode'] not in ('top', 'top_k_boundary'):
        raise ValueError('selection_mode must be top or top_k_boundary')
    if spec['selection_mode'] == 'top_k_boundary':
        if type(spec['selection_k']) is not int or not 1 <= spec['selection_k'] <= 1_000_000:
            raise ValueError('top_k_boundary requires a literal selection_k in [1, 1000000]')
    elif spec['selection_k'] is not None:
        raise ValueError('selection_k is only used with top_k_boundary')
    names = spec['feature_names']
    if (not isinstance(names, (list, tuple)) or len(names) > 64 or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not re.fullmatch(r'image_[a-zA-Z0-9_]{1,64}', n)
                   or n == 'image_available' for n in names)):
        raise ValueError('feature_names requires unique image_* names; image_available is reserved')
    if bool(names) != (len(functions) == 1) or len(functions) > 1:
        raise ValueError('Declare feature_names and exactly one extract_image_features(context) together')
    if not names and not spec['raw_patches']:
        raise ValueError('LOCAL_IMAGE must request feature extraction or raw_patches')
    if spec['raw_patches']:
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        if not {'fit', 'predict'} <= defined:
            raise ValueError('raw_patches requires a fit/predict model program')
    spec['feature_names'] = list(names)
    return program, spec
