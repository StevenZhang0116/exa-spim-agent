"""Read a few real candidate patches and save registration evidence; never train."""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import socket

import numpy as np

from . import precompute_error_scores as pc
from ..harness.native_pool import cache_path, ensure_native_tables
from ..harness.local_context import attach_local_context, _anchors
from ..harness.image_context import attach_image_context, DEFAULT_ALIGNMENT
from ..harness.image_preview import fragment_patch_geometry, alignment_statistics, render_preview
from ..harness.trajectory import Trajectory


def representative_rows(table, graph, count):
    candidates = np.linspace(0, len(table.candidates) - 1, min(512, len(table.candidates)), dtype=int)
    centers = np.asarray([graph.node_xyz[_anchors(table.kind, table.candidates[int(i)], 0)[0]].mean(axis=0)
                          for i in candidates])
    selected = [int(np.argmin(np.linalg.norm(centers - np.median(centers, axis=0), axis=1)))]
    while len(selected) < min(count, len(candidates)):
        distances = np.min(np.linalg.norm(centers[:, None] - centers[selected][None], axis=2), axis=1)
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    return candidates[selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brains', nargs='+', required=True)
    parser.add_argument('--mcl', type=int, default=100)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-kind', type=int, default=3)
    parser.add_argument('--radius-um', type=float, default=40.)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--image-uri', help='Explicit host override for one brain; recorded, never guessed')
    parser.add_argument('--alignment', type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument('--record-reviewed', action='store_true', help='Record a completed visual review, without reading images')
    parser.add_argument('--review-note', default='')
    args = parser.parse_args()
    if args.image_uri and len(args.brains) != 1:
        parser.error('--image-uri requires exactly one brain')
    if not 1 <= args.per_kind <= 10 or not 4 <= args.radius_um <= 100:
        parser.error('per-kind must be 1..10 and radius-um 4..100')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.record_reviewed:
        if not args.review_note.strip():
            parser.error('--record-reviewed requires a factual --review-note after inspecting the saved overlays')
        registry = json.loads(args.alignment.read_text()) if args.alignment.exists() else {'version': 1, 'brains': {}}
        for brain in args.brains:
            report = json.loads((args.out / brain / 'alignment.json').read_text())
            if report['status'] != 'needs_visual_review' or any(c.get('error') for c in report['cases']):
                raise ValueError('An incomplete/failed alignment audit cannot be recorded as reviewed')
            registry['brains'][brain] = {k: report[k] for k in ('source_cache', 'image_identity', 'image_uri')}
            registry['brains'][brain].update(status='reviewed', review_note=args.review_note,
                graph_to_world_um=report['geometry']['graph_to_world_um'],
                evidence=str((args.out / brain / 'alignment.json').resolve()),
                reviewed_at=datetime.now().astimezone().isoformat(), checked_level=report['spec']['level'])
        args.alignment.parent.mkdir(parents=True, exist_ok=True)
        args.alignment.write_text(json.dumps(registry, indent=2, allow_nan=False))
        print(f'Saved reviewed alignment receipts: {args.alignment}', flush=True)
        return
    trace = Trajectory(args.out)
    trace.emit('alignment_audit_start', f'Host={socket.gethostname()}; no fitting, evolution, or GT-based selection')
    selected = pc.resolve_detector_runs()
    spec = {'radius_um': args.radius_um, 'level': args.level, 'channel': 0, 'timepoint': 0}
    for brain in args.brains:
        directory = args.out / brain
        directory.mkdir(exist_ok=True)
        report = {'brain': brain, 'host': socket.gethostname(), 'status': 'unavailable', 'spec': spec, 'cases': []}
        bank = fragment_store = store = source = graph = None
        try:
            trace.emit('alignment_brain', f'Loading fixed candidate tables and fragments for {brain}')
            bank = ensure_native_tables(brain, cache_path(brain, args.mcl), selected, pc.DEFAULT_OUT,
                                        prepare=False, mcl=args.mcl)
            fragment_store = attach_local_context({brain: bank}, directory / 'geometry', trace)
            store = attach_image_context({brain: bank}, directory / 'image_cache', args.alignment, trace)
            if args.image_uri:
                store.reviews[brain] = {'image_uri': args.image_uri}
            table = next(iter(bank.tables.values()))
            source, graph = store.source(table, reviewed=False)
            report.update(source_cache=table.meta['provenance']['source_cache'], image_identity=source.identity,
                          image_uri=source.root, geometry=source.level(args.level).summary())
            for kind, table in bank.tables.items():
                for index in representative_rows(table, graph, args.per_kind):
                    case = {'kind': kind, 'row': int(index), 'occurrence': 0}
                    try:
                        path, metadata = store.patch(table, int(index), spec, reviewed=False)
                        with np.load(path, allow_pickle=False) as data:
                            patch = {key: data[key] for key in data.files}
                        nodes, edges, segments = fragment_patch_geometry(table, int(index), spec, metadata)
                        preview = directory / f'{kind}_{index}.png'
                        render_preview(preview, patch, nodes, edges, segments, f'{brain} / {kind} / candidate {index}')
                        case.update(preview=str(preview.resolve()), metadata=metadata,
                                    statistics=alignment_statistics(patch, nodes, edges))
                        trace.emit('alignment_case', f'{brain}/{kind}/{index}: saved {preview}',
                                   statistics=case['statistics'])
                    except Exception as exc:
                        case['error'] = f'{type(exc).__name__}: {exc}'
                        trace.emit('alignment_case_error', case['error'], brain=brain, kind=kind, row=int(index))
                    report['cases'].append(case)
            report['status'] = 'needs_visual_review' if not any(c.get('error') for c in report['cases']) else 'incomplete'
            report['interpretation'] = 'Sampled image/fragment registration check, not a whole-brain guarantee or error-label audit'
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            trace.emit('alignment_error', report['error'], brain=brain)
        (directory / 'alignment.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        bank = fragment_store = store = source = graph = table = None
        gc.collect()
    trace.emit('alignment_audit_end', 'Saved raw measurements and overlays; visual review must be recorded separately')


if __name__ == '__main__':
    main()
