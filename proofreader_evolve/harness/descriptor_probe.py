"""TRAIN-only diagnostics around the same descriptor worker used for full computation."""
import base64
import json
import time

import numpy as np

from .descriptor_inputs import input_summary
from .descriptor_runs import BATCH_ROWS
from .local_context import candidate_ref, resolve_train_candidate


def diagnostics(runs, table, columns, rows, values, *, labels=False):
    result = runs.summarize(table, columns, rows, values, labels=labels)
    for index, column in enumerate(columns):
        data = values[:, index]
        finite = np.flatnonzero(np.isfinite(data))
        examples = {}
        if len(finite):
            examples.update(min=finite[np.argmin(data[finite])], max=finite[np.argmax(data[finite])])
        for name, mask in (('nonzero', np.isfinite(data) & (data != 0)), ('missing', ~np.isfinite(data))):
            matches = np.flatnonzero(mask)
            if len(matches):
                examples[name] = matches[0]
        result[column]['examples'] = {name: candidate_ref(table, rows[i]) for name, i in examples.items()}
    return result


def probe(session, candidate_refs=None, sample_per_brain=128, seed=0, previews=False):
    """No registrations or scoring charges; elapsed time includes preparation and failures."""
    if type(previews) is not bool:
        raise ValueError('previews must be a boolean')
    program, spec = session._read_descriptor(check_registry=False)
    runs, tables = session.memory.descriptor_runs, session._descriptor_tables()
    selected = {}
    if candidate_refs is not None:
        if not isinstance(candidate_refs, list) or not 1 <= len(candidate_refs) <= 16:
            raise ValueError('Use 1..16 explicit TRAIN candidate_refs, or omit them for uniform sampling')
        for reference in candidate_refs:
            brain, table, row = resolve_train_candidate(session.train, reference)
            if table.kind != session.target_kind or not runs.cache.entry(table).has_rows([row]).all():
                raise ValueError('Every probe candidate must have cached geometry for the current target kind')
            selected.setdefault(brain, []).append(row)
        selected = {brain: np.asarray(sorted(set(rows)), dtype=int) for brain, rows in selected.items()}
    else:
        selected = {brain: runs.sample_rows(table, sample_per_brain, seed) for brain, table in tables.items()}
    allowance = min(session.descriptor_budget['per_call_seconds'], session.descriptor_seconds_remaining)
    if allowance <= 0:
        raise ValueError('The per-generation descriptor wall budget is exhausted')
    base = session.gen_dir / 'descriptor_probes'
    number = len(list(base.glob('probe*'))) + 1
    directory = base / f'probe{number:03d}'
    directory.mkdir(parents=True)
    (directory / 'descriptor.py').write_text(program)
    (directory / 'spec.json').write_text(json.dumps(spec, indent=2))
    response = {'status': 'probed', 'probe': f'{session.gen_dir.name}/probe{number:03d}',
                'columns': spec['columns'], 'inputs': spec['inputs'], 'image': spec.get('image'),
                'image_tier': spec['image_tier'] if spec['inputs'] != 'geometry' else None,
                'code_sha256': spec['code_sha256'],
                'sampling': 'explicit_train_cases' if candidate_refs is not None else 'uniform_cached_rows',
                'seed': seed if candidate_refs is None else None, 'brains': {}, 'previews': []}
    # At most one sample preview per brain; explicit cases share the existing eight-image allowance.
    preview_keys = [(brain, int(row)) for brain, rows in sorted(selected.items())
                    for row in (rows if candidate_refs is not None else rows[:1])]
    preview_keys = set(preview_keys[:max(0, 8 - session.image_inspections_used)]) if previews else set()
    images, started = [], time.monotonic()
    active_brain = None
    try:
        for brain, rows in sorted(selected.items()):
            active_brain = brain
            table = tables[brain]
            details = []

            def observe(row, context, patch):
                detail = {'candidate_ref': candidate_ref(table, row), **input_summary(context, patch)}
                details.append(detail)
                if (brain, row) not in preview_keys or patch is None:
                    return
                from .image_preview import render_preview
                session.image_inspections_used += 1
                path = directory / f'preview{session.image_inspections_used:03d}.png'
                fragment = context['fragment']
                render_preview(path, patch, np.asarray(fragment['nodes_zyx']), fragment['edges'],
                               fragment['segment'], f'TRAIN {brain} / {table.kind} / descriptor input')
                response['previews'].append({**detail, 'file': path.name})
                images.append({'type': 'image', 'mimeType': 'image/png',
                               'data': base64.b64encode(path.read_bytes()).decode('ascii')})

            try:
                result = runs.probe(table, program, spec, rows, log_dir=directory / brain, observe=observe,
                                    timeout=allowance - (time.monotonic() - started))
            finally:
                (directory / f'{brain}_inputs.json').write_text(json.dumps(details, indent=2, allow_nan=False))
            np.savez(directory / f'{brain}_values.npz', rows=rows, values=result['values'])
            # Full numeric output lives in the TRAIN artifact; keep the tool response compact.
            cases = []
            for i in range(min(len(rows), 16 if candidate_refs is not None else 3)):
                cases.append({**details[i], 'values': {column: float(v) if np.isfinite(v) else None
                    for column, v in zip(spec['columns'], result['values'][i])}})
            response['brains'][brain] = {
                'rows': len(rows), 'cases': cases,
                'summaries': diagnostics(runs, table, spec['columns'], rows, result['values']),
                **{key: result[key] for key in ('wall_seconds', 'preparation_seconds', 'worker_seconds', 'worker_batches')}}
    except Exception as exc:
        response.update(status='execution_error', error=f'{type(exc).__name__}: {exc}', brain=active_brain)
        if active_brain is not None and hasattr(exc, 'row'):
            response['candidate_ref'] = candidate_ref(tables[active_brain], exc.row)
            response['failed_input'] = next((d for d in details if d['candidate_ref'] == response['candidate_ref']), None)
    finally:
        elapsed = time.monotonic() - started
        session.descriptor_seconds_used += elapsed
    response.update(wall_seconds=elapsed, descriptor_seconds_remaining=session.descriptor_seconds_remaining,
                    image_inspections_remaining=8 - session.image_inspections_used,
                    evaluations_charged=0, evaluations_used=session.evaluations_used,
                    evaluations_remaining=session.remaining,
                    note='Diagnostic only; no predictor columns were registered. '
                    'Input support does not identify which pixels arbitrary descriptor code uses.')
    path = directory / 'result.json'
    path.write_text(json.dumps(response, indent=2, allow_nan=False))
    session._record_research('probe_descriptor', session._descriptor_source(spec), 'observation',
        {'code_sha256': spec['code_sha256'], 'cases': {b: r.tolist() for b, r in selected.items()}},
        response['status'] == 'probed', path, outcome={'status': response['status']})
    session.trace.emit('descriptor_probe', f"{response['probe']}: {response['status']}; {elapsed:.1f}s",
                       status=response['status'], preview_count=len(images))
    return {'content': [{'type': 'text', 'text': json.dumps(response, allow_nan=False)}, *images]} if images else response


def plan(session):
    """Measure every TRAIN brain and compare the whole call with its wall allowance."""
    program, spec = session._read_descriptor(check_registry=False)
    runs, tables = session.memory.descriptor_runs, session._descriptor_tables()
    allowance = min(600., session.descriptor_budget['per_call_seconds'], session.descriptor_seconds_remaining)
    if allowance <= 0:
        raise ValueError('The per-generation descriptor wall budget is exhausted')
    measured, started = {}, time.monotonic()
    response = {'status': 'planned', 'code_sha256': spec['code_sha256'], 'columns': spec['columns'],
                'inputs': spec['inputs'], 'image': spec.get('image'),
                'image_tier': spec['image_tier'] if spec['inputs'] != 'geometry' else None}
    try:
        for brain, table in sorted(tables.items()):
            measured[brain] = runs.plan(table, program, spec, k=session._k(session.target_kind, table.truth),
                                       timeout=allowance - (time.monotonic() - started))
        scopes = next(iter(measured.values()))['estimates']
        estimates = {scope: {key: sum(p['estimates'][scope][key] for p in measured.values())
                            for key in ('rows', 'batches', 'estimated_seconds')} for scope in scopes}
        cached = {scope: {brain: runs.cached(t, spec, scope, session._k(session.target_kind, t.truth))
                          for brain, t in tables.items()} for scope in scopes}
        for scope, estimate in estimates.items():
            estimate['cached_brains'] = sorted(brain for brain, hit in cached[scope].items() if hit)
            estimate['estimated_uncached_seconds'] = sum(p['estimates'][scope]['estimated_seconds']
                for brain, p in measured.items() if not cached[scope][brain])
        response.update(timed_on_brains=sorted(measured), per_brain=measured, estimates=estimates,
                        cached={scope: all(hits.values()) for scope, hits in cached.items()})
    except Exception as exc:
        response.update(status='execution_error', error=f'{type(exc).__name__}: {exc}', per_brain=measured)
    finally:
        elapsed = time.monotonic() - started
        session.descriptor_seconds_used += elapsed
    available = min(session.descriptor_budget['per_call_seconds'], session.descriptor_seconds_remaining)
    inference = {'scope': 'all_cached', 'estimated_seconds': None,
                 'note': 'Additional frozen inference is outside the TRAIN estimate; no heldout images were read.'}
    sizes = session.descriptor_budget.get('frozen_inference_rows')
    if response['status'] == 'planned' and sizes is not None:
        rate = max(p['seconds_per_row_single_worker'] for p in measured.values())
        inference.update(estimated_seconds=sum(rate * count / max(1, min(runs.workers, -(-count // BATCH_ROWS)))
                                              for count in sizes),
                         basis='Full cached workload sizes and slowest measured TRAIN seconds/row; '
                               'extrapolation only, cold source IO can differ.')
    response.update(wall_seconds=elapsed,
        budget={'per_call_seconds': session.descriptor_budget['per_call_seconds'],
                'generation_seconds_remaining': session.descriptor_seconds_remaining,
                'evaluations_remaining': session.remaining},
        fits_budget={scope: item['estimated_uncached_seconds'] <= available
                     for scope, item in response.get('estimates', {}).items()},
        frozen_inference=inference,
        note='Full estimates include sampled preparation and execution on every TRAIN brain; '
             'fits_budget compares only uncached work with the remaining descriptor allowance. Worker scaling is approximate; '
             'cold retrieval and shared IO can make full runs slower. Planning consumes descriptor wall time, no evaluation units.')
    directory = session.gen_dir / 'descriptor_plans'
    directory.mkdir(exist_ok=True)
    path = directory / f'plan{len(list(directory.glob("plan*.json"))) + 1:03d}.json'
    path.write_text(json.dumps(response, indent=2, allow_nan=False))
    session._record_research('plan_descriptor_run', 'execution', 'planning',
                            {'code_sha256': spec['code_sha256']}, response['status'] == 'planned', path)
    session.trace.emit('descriptor_plan', f"{spec['columns']}: {response['status']} across {len(measured)} TRAIN brains",
                       **response)
    return response
