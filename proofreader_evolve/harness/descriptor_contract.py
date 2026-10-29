"""Static contract for agent-authored descriptors; never import or execute agent source here.

`descriptor.py` declares one literal DESCRIPTOR dict and one describe(context) function.
The host runs it over cached contexts in isolated workers; the agent chooses what to
measure, the host only fixes names, limits and which inputs are transported.
"""
import ast
import hashlib
import math
import re


DESCRIPTOR_VERSION = 'agent-descriptor-v2-image-request'
MAX_DESCRIPTOR_BYTES = 128_000
MAX_NAMES = 32  # programs reached 14-15 names in the 2026-10 runs
COLUMN_PREFIX = 'bank_agent_'
NAME_PATTERN = re.compile(r'[a-z][a-z0-9_]{1,48}')
INPUTS = ('geometry', 'image', 'both')
TIERS = ('level1', 'level0')
TIER_IMAGES = {'level1': {'level': 1, 'radius_um': 30.},
               'level0': {'level': 0, 'radius_um': 16.}}


def descriptor_spec(source):
    if not isinstance(source, str) or len(source.encode()) > MAX_DESCRIPTOR_BYTES:
        raise ValueError('descriptor.py must be Python source of at most 128 KB')
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'DESCRIPTOR' for t in node.targets)]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'describe']
    if len(assignments) != 1 or len(functions) != 1:
        raise ValueError('descriptor.py must define one literal DESCRIPTOR dict and one describe(context) function')
    try:
        value = ast.literal_eval(assignments[0].value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError('DESCRIPTOR must be a literal dictionary') from exc
    defaults = {'inputs': 'geometry', 'image_tier': 'level1'}
    if not isinstance(value, dict) or set(value) - {*defaults, 'kind', 'feature_names', 'image'}:
        raise ValueError('DESCRIPTOR accepts kind, inputs, image_tier, image and feature_names only')
    spec = {**defaults, **value}
    if spec.get('kind') not in ('merge', 'split'):
        raise ValueError('DESCRIPTOR.kind must be merge or split')
    if spec['inputs'] not in INPUTS:
        raise ValueError(f'DESCRIPTOR.inputs must be one of {INPUTS}')
    if spec['image_tier'] not in TIERS:
        raise ValueError(f'DESCRIPTOR.image_tier must be one of {TIERS}')
    if 'image' in value:
        image = value['image']
        if (not isinstance(image, dict) or set(image) != {'level', 'radius_um'}
                or type(image['level']) is not int or not 0 <= image['level'] <= 6
                or type(image['radius_um']) not in (int, float)
                or not math.isfinite(image['radius_um']) or not 4 <= image['radius_um'] <= 100):
            raise ValueError('DESCRIPTOR.image requires level (integer 0..6) and radius_um (4..100)')
        if spec['inputs'] == 'geometry':
            raise ValueError('DESCRIPTOR.image requires image or both inputs')
        if 'image_tier' in value and image != TIER_IMAGES[value['image_tier']]:
            raise ValueError('DESCRIPTOR.image conflicts with image_tier; declare only one')
        spec['image'] = {**image, 'radius_um': float(image['radius_um']), 'channel': 0, 'timepoint': 0}
        spec['image_tier'] = value.get('image_tier')
    names = spec.get('feature_names')
    if (not isinstance(names, (list, tuple)) or not 1 <= len(names) <= MAX_NAMES or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not NAME_PATTERN.fullmatch(n) for n in names)):
        raise ValueError(f'feature_names requires 1..{MAX_NAMES} unique lowercase names matching {NAME_PATTERN.pattern}')
    spec['feature_names'] = list(names)
    spec['columns'] = [COLUMN_PREFIX + n for n in names]
    spec['code_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    return source, spec


def referenced_columns(source):
    """Registered descriptor columns a formula or program mentions by name."""
    return sorted(set(re.findall(COLUMN_PREFIX + r'[a-z][a-z0-9_]{1,48}', source or '')))
