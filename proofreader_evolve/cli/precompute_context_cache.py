"""Build the persistent candidate context cache (fragment neighbourhoods and image patches).

Run with panda on an allocated compute node, for example n257:
    python -u -m proofreader_evolve.cli.precompute_context_cache --brains 794495 802449 789202 794493 794491

Launcher mode runs one brain per fresh subprocess (single-brain mode, `--brain`) so
the fragment graph of each brain is released before the next. Each entry covers the
top detector-ranked band of every kind (default 20,000 rows): geometry at 50 um /
256 nodes, images at level 1 / 30 um and level 0 / 16 um, both for the whole band
(level 0 covered only the first 4,000 rows before 2026-10-05). Existing complete entries
are reused. `--extend-tier level0` grows one image tier of complete entries in place,
keeping geometry and the other tiers. `--probe N` builds a throughput sample of N rows
into a side directory without replacing an entry.
No labels, GT or absolute coordinates are written; no LLM is called.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from . import precompute_error_scores as pc
from ..harness.context_cache import ContextCacheBuilder, IMAGE_TIERS, BAND_SPEC, GEOMETRY_SPEC
from ..harness.image_context import DEFAULT_ALIGNMENT, CandidateImageStore, digest_json
from ..harness.local_context import FragmentStore, LocalContextProvider, _anchors
from ..harness.native_pool import cache_path, ensure_native_tables


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / 'proofreader_evolve' / 'context_cache'


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def log(message):
    print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)


def available_cpu_count():
    try:
        count = len(os.sched_getaffinity(0))
        if count:
            return count
    except (AttributeError, OSError, NotImplementedError):
        pass
    return max(1, os.cpu_count() or 1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--brains', nargs='+', help='Launcher mode: one subprocess per brain')
    parser.add_argument('--brain', help='Single-brain mode (used by the launcher)')
    parser.add_argument('--mcl', type=int, default=100)
    parser.add_argument('--kinds', nargs='+', default=['merge', 'split'], choices=['merge', 'split'])
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--feature-tables-dir', type=Path, default=pc.DEFAULT_OUT)
    parser.add_argument('--image-alignment', type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument('--merge-dir', action='append')
    parser.add_argument('--split-dir', action='append')
    parser.add_argument('--band-rows', type=int, default=BAND_SPEC['max_candidates'])
    parser.add_argument('--readers', type=int, default=8, help='Concurrent image reads per brain (default: 8)')
    parser.add_argument('--no-images', action='store_true', help='Geometry contexts only')
    parser.add_argument('--probe', type=int, default=None, metavar='N',
                        help='Throughput sample of N band rows into a side directory; the entry is not replaced')
    parser.add_argument('--force', action='store_true', help='Rebuild complete entries')
    parser.add_argument('--extend-tier', action='append', choices=sorted(IMAGE_TIERS), metavar='TIER',
                        help='Grow this image tier of existing complete entries to the configured row limit '
                             '(repeatable); geometry and other tiers are untouched')
    parser.add_argument('--threads', default='1', metavar='N|all', help='Numerical-library threads per subprocess')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--log-txt', type=Path)
    args = parser.parse_args(argv)
    if bool(args.brains) == bool(args.brain):
        parser.error('Use --brains (launcher) or --brain (single brain), not both')
    for brain in (args.brains or [args.brain]):
        if not brain.isascii() or not brain.isdigit():
            parser.error('Brain IDs must contain ASCII digits only')
    if args.band_rows < 1 or args.readers < 1 or (args.probe is not None and args.probe < 1):
        parser.error('band-rows, readers and probe must be positive')
    if args.extend_tier and (args.no_images or args.probe is not None or args.force):
        parser.error('--extend-tier needs images and cannot be combined with --probe or --force')
    if args.threads == 'all':
        args.threads = str(available_cpu_count())
    elif not args.threads.isdigit() or int(args.threads) < 1:
        parser.error("threads must be a positive integer or 'all'")
    args.out_dir = args.out_dir.expanduser().resolve()
    return args


def build_brain(args):
    """Single-brain mode: load tables, one fragment graph and one image source, then build each kind."""
    brain = str(args.brain)
    selected = pc.resolve_detector_runs(args.merge_dir, args.split_dir)
    bank = ensure_native_tables(brain, cache_path(brain, args.mcl), selected, args.feature_tables_dir,
                                prepare=False, mcl=args.mcl)
    log(f'brain {brain}: tables loaded ({", ".join(f"{k}={len(t.truth)} rows" for k, t in bank.tables.items())})')
    with tempfile.TemporaryDirectory(prefix='context_cache_build_') as tmp:
        store = FragmentStore(Path(tmp) / 'fragments')
        image_store = None if args.no_images else CandidateImageStore(Path(tmp) / 'images', args.image_alignment)
        builder = ContextCacheBuilder(args.out_dir, readers=args.readers, log=log)
        band = {**BAND_SPEC, 'max_candidates': args.band_rows}
        for kind in args.kinds:
            table = bank.tables[kind]
            provider = LocalContextProvider(store, table)
            table.local_context = provider
            reader = None
            if image_store is not None:
                source, graph = image_store.source(table)

                def reader(row, request, _source=source, _graph=graph, _table=table):
                    if row is None:
                        level = _source.level(request['level'])
                        return {'identity': _source.identity, 'uri': _source.root,
                                'level_metadata_sha256': digest_json(_source._levels[request['level']][1]),
                                'geometry': level.summary()}
                    anchors, _ = _anchors(_table.kind, _table.candidates[row], 0)
                    xyz = np.asarray(_graph.node_xyz[anchors], dtype=float)
                    return _source.read(xyz, radius_um=request['radius_um'], level=request['level'],
                                        channel=request['channel'], timepoint=request['timepoint'])
                # Open each level's array before the reader pool starts (TensorStore handles are shared).
                first = int(np.flatnonzero(np.isfinite(table.features['detector_score'].to_numpy(dtype=float)))[0])
                for tier in IMAGE_TIERS.values():
                    reader(first, tier)
            started = time.monotonic()
            if args.extend_tier:
                for name in args.extend_tier:
                    manifest = builder.extend_tier(table, reader, name, IMAGE_TIERS[name])
            else:
                manifest = builder.build(table, provider, reader, band=band, geometry=GEOMETRY_SPEC, tiers=IMAGE_TIERS,
                                         probe_rows=args.probe, force=args.force)
            summary = {'brain': brain, 'kind': kind, 'band_rows': manifest['band_rows'],
                       'geometry_seconds': manifest['geometry'].get('seconds'),
                       'image_tiers': {name: {k: tier.get(k) for k in ('rows', 'failures', 'bytes', 'seconds', 'seconds_per_read')}
                                       for name, tier in manifest['image_tiers'].items()},
                       'wall_seconds': time.monotonic() - started, 'probe': args.probe,
                       'extend_tier': args.extend_tier}
            log(f'brain {brain}/{kind}: {json.dumps(summary)}')
    return 0


def launcher(args):
    commands = []
    for brain in args.brains:
        command = [sys.executable, '-u', '-m', 'proofreader_evolve.cli.precompute_context_cache', '--brain', brain,
                   '--mcl', str(args.mcl), '--kinds', *args.kinds, '--out-dir', str(args.out_dir),
                   '--feature-tables-dir', str(args.feature_tables_dir), '--image-alignment', str(args.image_alignment),
                   '--band-rows', str(args.band_rows), '--readers', str(args.readers), '--threads', str(args.threads)]
        for flag, values in (('--merge-dir', args.merge_dir), ('--split-dir', args.split_dir)):
            for value in values or []:
                command += [flag, value]
        if args.no_images:
            command.append('--no-images')
        if args.probe is not None:
            command += ['--probe', str(args.probe)]
        if args.force:
            command.append('--force')
        for name in args.extend_tier or []:
            command += ['--extend-tier', name]
        commands.append(command)
    missing = [str(cache_path(b, args.mcl)) for b in args.brains if not cache_path(b, args.mcl).is_file()]
    if missing and not args.dry_run:
        print('Missing source cache(s):\n' + '\n'.join(missing), file=sys.stderr)
        return 1
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS'):
        env[name] = str(args.threads)
    env.setdefault('AWS_EC2_METADATA_DISABLED', 'true')
    print(f'Project: {PROJECT_ROOT}\nOutput: {args.out_dir}\nThreads: {args.threads}; readers: {args.readers}; '
          f'available CPUs: {available_cpu_count()}', flush=True)
    for command in commands:
        print('\n' + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        with subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                code = process.wait()
            except BaseException:
                process.kill()
                process.wait()
                raise
        if code:
            print(f'Stopped: subprocess exit code {code}. Completed entries are reusable.', file=sys.stderr)
            return code if code > 0 else 1
    if not args.dry_run:
        print('\nContext cache complete for all requested brains.', flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.brain:
        return build_brain(args)
    if args.dry_run:
        return launcher(args)
    log_path = (args.log_txt or PROJECT_ROOT / 'proofreader_evolve' / 'log' /
                f'precompute_context_cache_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log.txt')
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    returncode = 1
    with log_path.open('a', encoding='utf-8') as log_file, \
            redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(_Tee(sys.stderr, log_file)):
        print(f'\n# argv: {shlex.join([sys.executable, "-m", "proofreader_evolve.cli.precompute_context_cache", *(sys.argv[1:] if argv is None else argv)])}')
        print(f'# started {datetime.now():%Y-%m-%d %H:%M:%S%z}\n# host {platform.node()}  python {platform.python_version()}')
        print(f'Log: {log_path}')
        try:
            returncode = launcher(args)
            return returncode
        except KeyboardInterrupt:
            returncode = 130
            print('Interrupted; no further brains will be started.', file=sys.stderr)
            return returncode
        except Exception:
            traceback.print_exc()
            return returncode
        finally:
            outcome = 'OK' if returncode == 0 else 'FAILED'
            print(f'\n# {outcome} (exit code {returncode}) after {time.monotonic() - started:.1f}s')


if __name__ == '__main__':
    raise SystemExit(main())
