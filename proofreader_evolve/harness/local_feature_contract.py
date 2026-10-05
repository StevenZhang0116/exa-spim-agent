"""Static opt-in contract; never import or execute agent source in the host."""
import ast
import re

from .classifier_contract import frozen_model


def local_feature_spec(source):
    model = frozen_model(source)
    program = model.get('program', '') if model is not None else source
    tree = ast.parse(program)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'LOCAL_CONTEXT' for t in node.targets)]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'extract_local_features']
    if not assignments and not functions:
        return None
    if len(assignments) != 1 or len(functions) != 1:
        raise ValueError('Define one literal LOCAL_CONTEXT dict and extract_local_features(context)')
    value = ast.literal_eval(assignments[0].value)
    defaults = {'radius_um': 50., 'max_nodes': 128, 'max_occurrences': 1,
                'max_candidates': 5000, 'selection_feature': 'detector_score',
                'selection_largest': True}
    if not isinstance(value, dict) or set(value) - {*defaults, 'feature_names'}:
        raise ValueError('Unknown LOCAL_CONTEXT configuration field')
    spec = {**defaults, **value}
    names = spec.get('feature_names')
    if (not isinstance(names, (tuple, list)) or not 1 <= len(names) <= 64
            or any(not isinstance(n, str) or not re.fullmatch(r'local_[a-zA-Z0-9_]{1,64}', n)
                   or n == 'local_context_available' for n in names) or len(set(names)) != len(names)):
        raise ValueError('feature_names requires 1..64 unique local_* names; local_context_available is reserved')
    spec['feature_names'] = list(names)
    radius = spec['radius_um']
    if type(radius) not in (int, float) or not 1 <= radius <= 200:
        raise ValueError('radius_um must be a finite number in [1, 200]')
    for name, lower, upper in [('max_nodes', 2, 512), ('max_occurrences', 1, 8), ('max_candidates', 1, 20000)]:
        if type(spec[name]) is not int or not lower <= spec[name] <= upper:
            raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
    if type(spec['selection_largest']) is not bool or not isinstance(spec['selection_feature'], str):
        raise ValueError('selection_largest must be boolean and selection_feature a base feature name')
    return program, spec
