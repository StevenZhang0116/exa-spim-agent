"""Data-only proposals and AST-based numeric parameter substitution; never executes code."""

import ast
import hashlib
import itertools
import json
import math
import re

from .classifier_contract import frozen_model, model_info, normalize_config
from .hypothesis_memory import validate_research


MAX_PROPOSAL_BYTES = 128_000  # Resolved classifier feature lists can exceed a small formula proposal.


def _number(value):
    return type(value) in (int, float) and abs(value) <= 1e100 and math.isfinite(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate proposal field: {key}')
        result[key] = value
    return result


def read_proposal(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_PROPOSAL_BYTES:
        raise ValueError('Write a regular proposal.json (at most 128 KB) in this generation')
    try:
        proposal = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Fix proposal.json JSON syntax: {exc}') from exc
    if not isinstance(proposal, dict):
        raise ValueError('proposal.json must contain a JSON object')
    extra = set(proposal) - {'hypothesis', 'strategy', 'family', 'parameter_grid', 'classifier', 'research', 'handoff'}
    if extra:
        raise ValueError(f'Unknown proposal fields: {sorted(extra)}')
    for key in ('hypothesis', 'strategy'):
        value = proposal.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 1500:
            raise ValueError(f'Fill proposal.json.{key} with 1–1500 characters')
    family = proposal.get('family', 'unspecified')
    if not isinstance(family, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', family):
        raise ValueError('proposal.json.family must be a short name using letters, digits, _ or -')
    grid = proposal.get('parameter_grid', {})
    if not isinstance(grid, dict) or len(grid) > 32:
        raise ValueError('parameter_grid must be an object with at most 32 numeric parameter lists')
    for key, values in grid.items():
        if (not isinstance(values, list) or not 1 <= len(values) <= 64
                or not all(_number(v) for v in values)):
            raise ValueError(f'parameter_grid.{key} must be a nonempty finite numeric list (maximum 64)')
    if 'classifier' in proposal:
        proposal['classifier'] = normalize_config(proposal['classifier'])
    if 'research' in proposal:
        proposal['research'] = validate_research(proposal['research'])
    # The optional research handoff is advice for the next session, not part of the experiment:
    # it never enters attempt entries and the driver reads it from the file once
    # (hypothesis_memory.sanitize_handoff), so it can never make a proposal invalid.
    proposal.pop('handoff', None)
    return {**proposal, 'family': family, 'parameter_grid': grid}


def formula_info(source):
    """Describe a formula or frozen model, separating structure from tunable numbers.

    Formula identity ignores values only in its single literal PARAMS dictionary;
    frozen models delegate to the model program/configuration contract.
    """
    tree = ast.parse(source)
    model = frozen_model(source, tree)
    if model is not None:
        return model_info(model), None
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'PARAMS' for t in node.targets)]
    stores = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
              and node.id == 'PARAMS' and isinstance(node.ctx, ast.Store)]
    parameters = {}
    value_node = None
    if stores:
        if len(assignments) != 1 or len(stores) != 1 or len(assignments[0].targets) != 1:
            raise ValueError('Use exactly one top-level PARAMS = {"name": number} assignment')
        value_node = assignments[0].value
        if not isinstance(value_node, ast.Dict):
            raise ValueError('PARAMS must be a literal numeric dictionary')
        try:
            keys = [ast.literal_eval(key) for key in value_node.keys]
            parameters = ast.literal_eval(value_node)
        except (ValueError, TypeError) as exc:
            raise ValueError('PARAMS must be a literal numeric dictionary') from exc
        if (len(parameters) > 32 or len(keys) != len(set(keys))
                or any(not isinstance(k, str) or not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]{0,63}', k)
                       or not _number(v) for k, v in parameters.items())):
            raise ValueError('PARAMS must have unique identifier keys and finite numeric values (maximum 32)')
        # Sort keys so formatting and dictionary order do not create new formula families.
        assignments[0].value = ast.Dict(keys=[ast.Constant(k) for k in sorted(parameters)],
                                       values=[ast.Constant(0) for k in sorted(parameters)])
    identity = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    return {'candidate_type': 'formula', 'formula_sha256': identity, 'parameters': parameters}, value_node


def parameterize(source, updates):
    info, node = formula_info(source)
    if node is None or not updates or set(updates) - info['parameters'].keys():
        raise ValueError('Every grid key must exist in a nonempty top-level PARAMS dictionary')
    if not all(_number(value) for value in updates.values()):
        raise ValueError('Parameter values must be finite numbers')
    # AST columns are UTF-8 byte offsets, including when comments contain non-ASCII text.
    lines = source.encode().splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
    value = repr({**info['parameters'], **updates}).encode()
    return (source.encode()[:start] + value + source.encode()[end:]).decode()


def parameter_candidates(source, proposal):
    grid = proposal['parameter_grid']
    if not grid:
        raise ValueError('Fill proposal.json.parameter_grid before search_parameters()')
    if math.prod(len(values) for values in grid.values()) > 64:
        raise ValueError('Parameter grid exceeds 64 combinations; narrow the ranges')
    return list(dict.fromkeys(parameterize(source, dict(zip(grid, values)))
                              for values in itertools.product(*grid.values())))
