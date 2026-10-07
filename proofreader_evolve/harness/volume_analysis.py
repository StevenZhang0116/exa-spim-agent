"""TRAIN-only, agent-authored analysis of real 3D pixels and aligned fragments."""
import ast
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from .image_context import digest_json, file_hash
from .image_coordinates import ImageGeometry
from .image_features import stage_images
from .local_context import resolve_train_candidate
from .model_execution import run_worker

VOLUME_ANALYSIS_VERSION = 'train-volume-analysis-v1'
MAX_ANALYSES = 8
# Raised from 4 on 2026-10-06 so one exploratory call can compare groups (for example
# eight missed positives against eight selected label-0 rows) instead of a handful of sites.
MAX_CASES = 16
MAX_RESULT_BYTES = 64 * 1024


def reject_constant(value):
    raise ValueError(f'Nonfinite analysis value: {value}')


def read_analysis(gen_dir, train):
    """Validate files and every TRAIN handle before any graph or cloud access."""
    directory = Path(gen_dir).resolve()
    contents = {}
    for name, maximum in (('analysis.py', 128_000), ('analysis_request.json', 16_000)):
        path = directory / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
            raise ValueError(f'Write a regular {name} of at most {maximum} bytes in this generation')
        contents[name] = path.read_text()
    program = contents['analysis.py']
    tree = ast.parse(program)
    if sum(isinstance(n, ast.FunctionDef) and n.name == 'analyze' for n in tree.body) != 1:
        raise ValueError('analysis.py must define analyze(context)')
    value = json.loads(contents['analysis_request.json'])
    defaults = {'radius_um': 40., 'level': 0, 'channel': 0, 'timepoint': 0, 'max_nodes': 256}
    if not isinstance(value, dict) or set(value) - {*defaults, 'candidates'}:
        raise ValueError('Unknown analysis request field; image paths and registration are host-owned')
    request = {**defaults, **value}
    radius = request['radius_um']
    if type(radius) not in (int, float) or not math.isfinite(radius) or not 4 <= radius <= 100:
        raise ValueError('radius_um must be in [4, 100]')
    for name, lower, upper in (('level', 0, 6), ('channel', 0, 15), ('timepoint', 0, 15), ('max_nodes', 2, 512)):
        if type(request[name]) is not int or not lower <= request[name] <= upper:
            raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
    cases = request.get('candidates')
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError(f'Select 1..{MAX_CASES} TRAIN candidates in analysis_request.json')
    resolved, normalized, seen = [], [], set()
    for case in cases:
        if not isinstance(case, dict) or set(case) - {'candidate_ref', 'occurrence_index'}:
            raise ValueError('Each candidate requires candidate_ref and optional occurrence_index')
        occurrence = case.get('occurrence_index', 0)
        if type(occurrence) is not int or occurrence < 0:
            raise ValueError('occurrence_index must be a nonnegative integer')
        brain, table, index = resolve_train_candidate(train, case.get('candidate_ref'))
        from .local_context import _anchors
        _anchors(table.kind, table.candidates[index], occurrence)
        key = case['candidate_ref'], occurrence
        if key in seen:
            raise ValueError('Duplicate candidate/occurrence in one analysis request')
        seen.add(key)
        normalized.append({'candidate_ref': key[0], 'occurrence_index': occurrence})
        resolved.append((brain, table, index, occurrence))
    request['candidates'] = normalized
    return program, request, resolved


def aligned_context(table, index, occurrence, request, metadata, patch_path):
    """Whitelisted fragments in the patch's voxel frame, without absolute IDs/locations."""
    geometry, centers = table.local_context.context(index, radius_um=request['radius_um'],
        max_nodes=request['max_nodes'], max_occurrences=1, occurrence_index=occurrence)
    site = geometry['sites'][0]
    image_geometry = ImageGeometry(tuple(metadata['source_axes']), tuple(metadata['source_shape']),
        np.asarray(metadata['spacing_xyz_um']), np.asarray(metadata['translation_xyz_um']),
        np.asarray(metadata['graph_to_world_um']), metadata['dataset'])
    relative = np.asarray(site['xyz_um'], dtype=float)
    nodes = image_geometry.graph_to_voxel(relative + np.asarray(centers[0])) - np.asarray(metadata['origin_zyx'])
    with np.load(patch_path, allow_pickle=False) as patch:
        if not np.allclose(nodes[site['anchor_nodes']], patch['anchors_zyx'], atol=1e-6, rtol=0):
            raise ValueError('Fragment anchors and image anchors disagree')
        shape = np.asarray(patch['image_zyx'].shape)
    # Only explicit local geometry is transported; never copy a graph attribute dict.
    fragment = {name: site[name] for name in ('xyz_um', 'radius_um', 'degree', 'segment',
        'edges', 'anchor_nodes', 'outside_radius', 'truncated')}
    fragment.update(nodes_zyx=nodes.tolist(), inside_patch=((nodes >= 0) & (nodes < shape)).all(axis=1).tolist())
    features = {str(name): float(value) if np.isfinite(value) else None
                for name, value in table.features.iloc[index].items()}
    return {'kind': table.kind, 'features': features, 'fragment': fragment,
            'radius_um': request['radius_um'], 'level': request['level'],
            'channel': request['channel'], 'timepoint': request['timepoint'],
            'occurrence_index': occurrence, 'occurrences_total': geometry['occurrences_total'],
            'coordinates': {'array_axes': 'zyx', 'anchors_and_nodes': 'patch-relative voxel centers',
                'xyz_um': 'graph xyz microns relative to the candidate anchor midpoint',
                'spacing_zyx_um': 'physical image spacing in zyx order',
                'valid_zyx': 'in-domain mask, not proof of acquired signal in every chunk',
                'intensity': 'original fused-volume pixels; no preview normalization'}}


def execute_analysis(program, request, resolved, directory, *, timeout=120, memory_mb=8192, threads=1):
    """One isolated program, applied to each requested 3D context; never rank or fit a policy."""
    directory = Path(directory)
    started = time.monotonic()
    spec = {k: request[k] for k in ('radius_um', 'level', 'channel', 'timepoint')}
    contexts, inputs, rows = [], [], {}
    for position, (brain, table, index, occurrence) in enumerate(resolved):
        provider = getattr(table, 'image_context', None)
        if provider is None:
            raise ValueError('TRAIN volume access is not configured')
        path, metadata = provider.patch(table, index, spec, occurrence)
        contexts.append(aligned_context(table, index, occurrence, request, metadata, path))
        rows[position] = [str(path)]
        inputs.append({'candidate_ref': request['candidates'][position]['candidate_ref'],
                       'brain': brain, 'kind': table.kind, 'row': index, 'occurrence_index': occurrence,
                       'image_metadata': metadata})
    identity = {'version': VOLUME_ANALYSIS_VERSION, 'program_sha256': hashlib.sha256(program.encode()).hexdigest(),
                'request': request, 'inputs': inputs, 'contexts_sha256': digest_json(contexts),
                'implementation': {name: file_hash(Path(__file__).with_name(name)) for name in (
                    'volume_analysis.py', 'classifier_train_worker.py', 'model_execution.py',
                    'model_sandbox.py', 'local_context.py', 'image_features.py')}}
    (directory / 'input_manifest.json').write_text(json.dumps(identity, indent=2, allow_nan=False))
    preparation_seconds = time.monotonic() - started
    with tempfile.TemporaryDirectory(prefix='volume_analysis_') as tmp:
        root = Path(tmp)
        frame = pd.DataFrame(index=range(len(contexts)))
        frame.attrs['_image_rows'] = rows
        transport = stage_images(root, [(frame, np.arange(len(frame)))])
        (root / 'volume_contexts.json').write_text(json.dumps(contexts, allow_nan=False))
        (root / 'training.py').write_text(program)
        (root / 'artifacts').mkdir()
        worker_request = {'mode': 'analyze_volume', 'image_inputs': True, 'timeout': timeout,
            'memory_mb': memory_mb, 'threads': threads, 'max_result_bytes': MAX_RESULT_BYTES}
        status = run_worker(root, worker_request, log_path=directory / 'worker.log')
        output = root / 'analysis.json'
        if output.stat().st_size > MAX_RESULT_BYTES:
            raise ValueError(f'Analysis results exceed {MAX_RESULT_BYTES // 1024} KiB; return summaries, not volume arrays')
        results = json.loads(output.read_text(), parse_constant=reject_constant)
        if not isinstance(results, list) or len(results) != len(contexts) or any(not isinstance(r, dict) for r in results):
            raise ValueError('Analysis must return one JSON object per requested candidate')
    cases = []
    for selected, context, input_info, result in zip(request['candidates'], contexts, inputs, results):
        metadata = input_info['image_metadata']
        cases.append({**selected, 'kind': context['kind'], 'shape_zyx': metadata['shape_zyx'],
            'spacing_zyx_um': list(reversed(metadata['spacing_xyz_um'])),
            'valid_fraction': metadata['valid_fraction'], 'fragment_truncated': context['fragment']['truncated'],
            'result': result})
    return {'status': 'analyzed', 'version': VOLUME_ANALYSIS_VERSION,
            'program_sha256': identity['program_sha256'], 'input_signature': digest_json(identity),
            'cases': cases, 'preparation_seconds': preparation_seconds, 'worker_seconds': status['wall_seconds'],
            'wall_seconds': time.monotonic() - started, 'transport': transport,
            'scope': 'Agent-computed exploratory summaries on selected TRAIN 3D patches; '
                     'not a measured ranking improvement or a submitted policy. '
                     'Selection is independent of LOCAL_IMAGE scoring coverage.'}
