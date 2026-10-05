"""Persistent, label-free candidate contexts: fragment neighbourhoods and image patches.

Built once per brain and kind over a fixed detector-ranked band, so agent-written
descriptor code can run over tens of thousands of rows inside a generation without
loading a fragment graph or reading pixels from cloud storage. Nothing here is a
feature: the cache stores inputs, and every quantity computed from them is defined
by the agent at run time. Contexts carry the same whitelisted geometry fields and
patch arrays as `analyze(context)`; labels, GT, absolute coordinates and persistent
IDs never enter the cache.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .image_context import digest_json, file_hash
from .image_coordinates import ImageGeometry
from .local_context import _anchors, cache_identity
from .local_features import select_rows


CACHE_VERSION = 'candidate-context-cache-v1'
BAND_SPEC = {'selection_feature': 'detector_score', 'selection_largest': True, 'max_candidates': 20000}
GEOMETRY_SPEC = {'radius_um': 50., 'max_nodes': 256}
IMAGE_TIERS = {'level1': {'level': 1, 'radius_um': 30., 'channel': 0, 'timepoint': 0, 'max_rows': None},
               'level0': {'level': 0, 'radius_um': 16., 'channel': 0, 'timepoint': 0, 'max_rows': 4000}}
CHUNK_ROWS = 1024
FRAGMENT_FIELDS = ('xyz_um', 'radius_um', 'degree', 'segment', 'edges', 'anchor_nodes', 'outside_radius', 'truncated')
IMPLEMENTATION_FILES = ('context_cache.py', 'local_context.py', 'image_context.py', 'image_coordinates.py')
COORDINATE_DOC = {'array_axes': 'zyx', 'anchors_and_nodes': 'patch-relative voxel centers',
                  'xyz_um': 'graph xyz microns relative to the candidate anchor midpoint',
                  'spacing_zyx_um': 'physical image spacing in zyx order',
                  'valid_zyx': 'in-domain mask, not proof of acquired signal in every chunk',
                  'intensity': 'original fused-volume pixels; no preview normalization'}


def implementation_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in IMPLEMENTATION_FILES}


def table_identity(table):
    provenance = table.meta.get('provenance', {})
    return {'brain': str(provenance.get('brain', '')), 'kind': table.kind,
            'pool_sha256': table.meta['pool_sha256'],
            'features_sha256': table.meta.get('files', {}).get('features.pkl'),
            'source_cache': provenance.get('source_cache')}


def entry_directory(root, table):
    identity = table_identity(table)
    return Path(root) / identity['brain'] / table.kind / identity['pool_sha256']


def _pack_sites(rows, sites, totals):
    nodes = [len(site['xyz_um']) for site in sites]
    edges = [len(site['edges']) for site in sites]
    anchors = [len(site['anchor_nodes']) for site in sites]
    offsets = lambda counts: np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    xyz = (np.concatenate([np.asarray(s['xyz_um'], dtype=float).reshape(-1, 3) for s in sites])
           if sum(nodes) else np.zeros((0, 3)))
    edge_array = (np.concatenate([np.asarray(s['edges'], dtype=np.int32).reshape(-1, 2) for s in sites])
                  if sum(edges) else np.zeros((0, 2), dtype=np.int32))
    return {'rows': np.asarray(rows, dtype=np.int64), 'node_offsets': offsets(nodes),
            'edge_offsets': offsets(edges), 'anchor_offsets': offsets(anchors), 'xyz_um': xyz,
            'radius_um': np.asarray([np.nan if r is None else float(r) for s in sites for r in s['radius_um']]),
            'degree': np.asarray([d for s in sites for d in s['degree']], dtype=np.int16),
            'segment': np.asarray([g for s in sites for g in s['segment']], dtype=np.int32),
            'outside_radius': np.asarray([o for s in sites for o in s['outside_radius']], dtype=bool),
            'edges': edge_array,
            'anchor_nodes': np.asarray([a for s in sites for a in s['anchor_nodes']], dtype=np.int32),
            'truncated': np.asarray([bool(s['truncated']) for s in sites], dtype=bool),
            'occurrences_total': np.asarray(totals, dtype=np.int16)}


def _unpack_site(chunk, position):
    a, b = int(chunk['node_offsets'][position]), int(chunk['node_offsets'][position + 1])
    c, d = int(chunk['edge_offsets'][position]), int(chunk['edge_offsets'][position + 1])
    e, f = int(chunk['anchor_offsets'][position]), int(chunk['anchor_offsets'][position + 1])
    radius = chunk['radius_um'][a:b]
    return {'xyz_um': chunk['xyz_um'][a:b].tolist(),
            'radius_um': [None if not np.isfinite(r) else float(r) for r in radius],
            'degree': chunk['degree'][a:b].astype(int).tolist(),
            'segment': chunk['segment'][a:b].astype(int).tolist(),
            'edges': chunk['edges'][c:d].astype(int).tolist(),
            'anchor_nodes': chunk['anchor_nodes'][e:f].astype(int).tolist(),
            'outside_radius': chunk['outside_radius'][a:b].tolist(),
            'truncated': bool(chunk['truncated'][position])}


class ContextCacheBuilder:
    """Offline construction; one brain graph and one image source at a time."""
    def __init__(self, root, *, readers=8, trace=None, log=print):
        self.root, self.readers, self.trace, self.log = Path(root), max(1, int(readers)), trace, log

    def existing(self, table):
        directory = entry_directory(self.root, table)
        manifest = directory / 'manifest.json'
        if not manifest.is_file():
            return None
        value = json.loads(manifest.read_text())
        if value.get('complete') and value.get('identity') == table_identity(table):
            return value
        return None

    def build(self, table, provider, image_reader=None, *, band=BAND_SPEC, geometry=GEOMETRY_SPEC,
              tiers=IMAGE_TIERS, probe_rows=None, force=False):
        """`provider.context(row, ...)` supplies geometry; `image_reader(anchors_xyz, tier)`
        returns (data, metadata) like ImageSource.read, or None to skip images."""
        started = time.monotonic()
        identity = table_identity(table)
        if identity['source_cache'] is not None:
            if cache_identity(identity['source_cache']['path']) != identity['source_cache']:
                raise ValueError('Fragment cache changed since native feature preparation')
        if not force and probe_rows is None:
            manifest = self.existing(table)
            if manifest is not None:
                self.log(f"[context-cache] {identity['brain']}/{table.kind}: complete entry reused")
                return manifest
        rows = select_rows(table.features, band)
        if probe_rows is not None:
            rows = rows[:int(probe_rows)]
        rows = np.asarray(rows, dtype=np.int64)
        final = entry_directory(self.root, table)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.staging_', dir=final.parent))
        try:
            manifest = {'version': CACHE_VERSION, 'identity': identity, 'band': dict(band),
                        'band_rows': int(len(rows)), 'pool_rows': int(len(table.features)),
                        'probe_rows': probe_rows, 'base_columns': [str(c) for c in table.features.columns],
                        'geometry': dict(geometry), 'image_tiers': {}, 'implementation': implementation_hashes(),
                        'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
                        'coordinates': COORDINATE_DOC, 'complete': False}
            np.save(staging / 'rows.npy', rows, allow_pickle=False)
            sites, centers, totals = self._build_geometry(table, provider, rows, geometry, staging, manifest)
            if image_reader is not None:
                for name, tier in tiers.items():
                    self._build_tier(table, image_reader, rows, sites, centers, name, tier, staging, manifest)
            manifest['build_seconds'] = time.monotonic() - started
            manifest['complete'] = True
            (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
            if probe_rows is not None:
                probe = final.parent / f'.probe_{identity["pool_sha256"][:12]}'
                if probe.exists():
                    _remove_tree(probe)
                staging.rename(probe)
                return manifest
            if final.exists():
                _remove_tree(final)
            staging.rename(final)
            return manifest
        except BaseException:
            _remove_tree(staging)
            raise

    def _build_geometry(self, table, provider, rows, geometry, staging, manifest):
        started = time.monotonic()
        directory = staging / 'geometry'
        directory.mkdir()
        sites, centers, totals, chunks, truncated = [], [], [], [], 0
        for start in range(0, len(rows), CHUNK_ROWS):
            block = rows[start:start + CHUNK_ROWS]
            block_sites, block_totals = [], []
            for row in block:
                context, center = provider.context(int(row), radius_um=geometry['radius_um'],
                                                   max_nodes=geometry['max_nodes'], max_occurrences=1)
                site = context['sites'][0]
                block_sites.append(site)
                block_totals.append(int(context['occurrences_total']))
                centers.append(np.asarray(center[0], dtype=float))
                truncated += int(site['truncated'])
            packed = _pack_sites(block, block_sites, block_totals)
            name = f'chunk_{start // CHUNK_ROWS:05d}.npz'
            np.savez(directory / name, **packed)
            chunks.append({'file': name, 'rows': int(len(block)), 'bytes': (directory / name).stat().st_size})
            sites.extend(block_sites)
            totals.extend(block_totals)
            self.log(f"[context-cache] geometry {start + len(block)}/{len(rows)} rows "
                     f"({time.monotonic() - started:.0f}s)")
        manifest['geometry'].update(chunks=chunks, truncated_sites=truncated,
                                    seconds=time.monotonic() - started, chunk_rows=CHUNK_ROWS)
        return sites, centers, totals

    def _build_tier(self, table, image_reader, rows, sites, centers, name, tier, staging, manifest):
        started = time.monotonic()
        limit = tier.get('max_rows')
        tier_rows = rows if limit is None else rows[:int(limit)]
        directory = staging / 'images' / name
        directory.mkdir(parents=True)
        request = {k: tier[k] for k in ('level', 'radius_um', 'channel', 'timepoint')}
        reader_info = image_reader(None, request)  # metadata-only call: source identity and level geometry
        chunks, failures, bytes_total, metadata_summary = [], [], 0, None
        per_read = []

        def fetch(position):
            row = int(tier_rows[position])
            anchors, _ = _anchors(table.kind, table.candidates[row], 0)
            t0 = time.monotonic()
            try:
                data, metadata = image_reader(row, request)
            except Exception as exc:  # One failed patch does not abort the band; it is recorded.
                return position, None, None, f'{type(exc).__name__}: {exc}', time.monotonic() - t0
            return position, data, metadata, None, time.monotonic() - t0

        for start in range(0, len(tier_rows), CHUNK_ROWS):
            block = list(range(start, min(start + CHUNK_ROWS, len(tier_rows))))
            with ThreadPoolExecutor(max_workers=self.readers) as pool:
                results = list(pool.map(fetch, block))
            payload = {'rows': tier_rows[block[0]:block[-1] + 1].astype(np.int64),
                       'available': np.zeros(len(block), dtype=bool),
                       'origin_zyx': np.zeros((len(block), 3), dtype=np.int64),
                       'shape_zyx': np.zeros((len(block), 3), dtype=np.int64),
                       'valid_fraction': np.zeros(len(block)),
                       'anchors_zyx': np.full((len(block), 2, 3), np.nan)}
            node_lists = []
            for position, data, metadata, error, seconds in results:
                local = position - block[0]
                per_read.append(seconds)
                if error is not None:
                    failures.append({'row': int(tier_rows[position]), 'error': error[:300]})
                    node_lists.append(np.zeros((0, 3)))
                    continue
                if metadata_summary is None:
                    metadata_summary = {k: metadata[k] for k in ('source_axes', 'source_shape', 'spacing_xyz_um',
                                        'translation_xyz_um', 'graph_to_world_um', 'dataset', 'level',
                                        'channel', 'timepoint', 'dtype')}
                image_geometry = ImageGeometry(tuple(metadata['source_axes']), tuple(metadata['source_shape']),
                    np.asarray(metadata['spacing_xyz_um']), np.asarray(metadata['translation_xyz_um']),
                    np.asarray(metadata['graph_to_world_um']), metadata['dataset'])
                relative = np.asarray(sites[position]['xyz_um'], dtype=float).reshape(-1, 3)
                nodes = image_geometry.graph_to_voxel(relative + centers[position]) - np.asarray(metadata['origin_zyx'])
                anchors_zyx = np.asarray(data['anchors_zyx'], dtype=float)
                if not np.allclose(nodes[sites[position]['anchor_nodes']], anchors_zyx, atol=1e-6, rtol=0):
                    failures.append({'row': int(tier_rows[position]), 'error': 'fragment/image anchor mismatch'})
                    node_lists.append(np.zeros((0, 3)))
                    continue
                payload[f'image_{local}'] = np.asarray(data['image_zyx'])
                payload[f'valid_{local}'] = np.asarray(data['valid_zyx'], dtype=bool)
                payload['available'][local] = True
                payload['origin_zyx'][local] = metadata['origin_zyx']
                payload['shape_zyx'][local] = metadata['shape_zyx']
                payload['valid_fraction'][local] = metadata['valid_fraction']
                payload['anchors_zyx'][local, :len(anchors_zyx)] = anchors_zyx
                node_lists.append(nodes)
            payload['node_offsets'] = np.concatenate([[0], np.cumsum([len(n) for n in node_lists])]).astype(np.int64)
            payload['nodes_zyx'] = np.concatenate(node_lists) if node_lists else np.zeros((0, 3))
            file_name = f'chunk_{start // CHUNK_ROWS:05d}.npz'
            np.savez_compressed(directory / file_name, **payload)
            size = (directory / file_name).stat().st_size
            bytes_total += size
            chunks.append({'file': file_name, 'rows': len(block), 'bytes': size})
            self.log(f"[context-cache] images/{name} {block[-1] + 1}/{len(tier_rows)} rows "
                     f"({time.monotonic() - started:.0f}s, {len(failures)} failures)")
        # Tolerate isolated read failures (recorded as unavailable rows); abort on systematic failure.
        if len(tier_rows) and len(failures) > max(50, .05 * len(tier_rows)):
            raise ValueError(f'Image tier {name}: {len(failures)} of {len(tier_rows)} patches failed')
        manifest['image_tiers'][name] = {
            **request, 'rows': int(len(tier_rows)), 'chunks': chunks, 'bytes': bytes_total,
            'failed_rows': failures[:200], 'failures': len(failures), 'metadata': metadata_summary,
            'source': reader_info, 'seconds': time.monotonic() - started,
            'seconds_per_read': {'median': float(np.median(per_read)) if per_read else None,
                                 'max': float(np.max(per_read)) if per_read else None},
            'readers': self.readers, 'chunk_rows': CHUNK_ROWS}


def _remove_tree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


class ContextEntry:
    """Read-only view of one brain/kind entry; chunks are loaded on demand."""
    def __init__(self, directory, manifest):
        self.directory, self.manifest = Path(directory), manifest
        self.rows = np.load(self.directory / 'rows.npy', allow_pickle=False)
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        self._chunks = {}

    @property
    def kind(self):
        return self.manifest['identity']['kind']

    @property
    def brain(self):
        return self.manifest['identity']['brain']

    @property
    def base_columns(self):
        return list(self.manifest['base_columns'])

    def tiers(self):
        return list(self.manifest.get('image_tiers', {}))

    def has_rows(self, rows):
        return np.asarray([int(r) in self.position for r in rows], dtype=bool)

    def _chunk(self, group, index):
        key = (group, index)
        if key not in self._chunks:
            if len(self._chunks) > 8:
                self._chunks.pop(next(iter(self._chunks)))
            subdir = self.directory / ('geometry' if group == 'geometry' else f'images/{group}')
            with np.load(subdir / f'chunk_{index:05d}.npz', allow_pickle=False) as data:
                self._chunks[key] = {name: data[name] for name in data.files}
        return self._chunks[key]

    def geometry(self, rows):
        chunk_rows = self.manifest['geometry'].get('chunk_rows', CHUNK_ROWS)
        sites, totals = [], []
        for row in rows:
            index = self.position[int(row)]
            chunk = self._chunk('geometry', index // chunk_rows)
            local = index % chunk_rows
            sites.append(_unpack_site(chunk, local))
            totals.append(int(chunk['occurrences_total'][local]))
        return sites, totals

    def image(self, rows, tier):
        info = self.manifest['image_tiers'].get(tier)
        if info is None:
            raise ValueError(f'Image tier {tier!r} is not cached for {self.brain}/{self.kind}')
        chunk_rows = info.get('chunk_rows', CHUNK_ROWS)
        spacing = np.asarray(info['metadata']['spacing_xyz_um'], dtype=float)[::-1]
        results = []
        for row in rows:
            index = self.position[int(row)]
            if index >= info['rows']:
                results.append(None)
                continue
            chunk = self._chunk(tier, index // chunk_rows)
            local = index % chunk_rows
            if not chunk['available'][local]:
                results.append(None)
                continue
            a, b = int(chunk['node_offsets'][local]), int(chunk['node_offsets'][local + 1])
            anchors = chunk['anchors_zyx'][local]
            anchors = anchors[np.isfinite(anchors).all(axis=1)]
            results.append({'image_zyx': chunk[f'image_{local}'], 'valid_zyx': chunk[f'valid_{local}'],
                            'anchors_zyx': anchors, 'spacing_zyx_um': spacing.copy(),
                            'nodes_zyx': chunk['nodes_zyx'][a:b], 'shape_zyx': chunk['shape_zyx'][local],
                            'valid_fraction': float(chunk['valid_fraction'][local])})
        return results

    def contexts(self, table, rows, *, inputs='geometry', tier='level1'):
        """Worker-ready contexts (no pixels inline) plus patch dicts for rows with images."""
        rows = [int(r) for r in rows]
        missing = [r for r in rows if r not in self.position]
        if missing:
            raise ValueError(f'{len(missing)} requested rows are outside the cached band')
        sites, totals = self.geometry(rows)
        images = self.image(rows, tier) if inputs in ('image', 'both') else [None] * len(rows)
        base = [c for c in self.base_columns if c in table.features.columns]
        frame = table.features[base]
        contexts, patches = [], []
        geometry = self.manifest['geometry']
        tier_info = self.manifest['image_tiers'].get(tier, {}) if inputs != 'geometry' else {}
        for row, site, total, patch in zip(rows, sites, totals, images):
            fragment = {name: site[name] for name in FRAGMENT_FIELDS}
            if patch is not None:
                nodes = np.asarray(patch['nodes_zyx'])
                fragment.update(nodes_zyx=nodes.tolist(),
                                inside_patch=((nodes >= 0) & (nodes < patch['shape_zyx'])).all(axis=1).tolist())
            values = frame.iloc[self._row_index(table, row)]
            contexts.append({'kind': table.kind, 'row_in_band': True,
                             'features': {str(k): float(v) if np.isfinite(v) else None for k, v in values.items()},
                             'fragment': fragment, 'radius_um': geometry['radius_um'],
                             'level': tier_info.get('level'), 'channel': tier_info.get('channel'),
                             'timepoint': tier_info.get('timepoint'), 'image_radius_um': tier_info.get('radius_um'),
                             'image_available': patch is not None, 'occurrence_index': 0,
                             'occurrences_total': total, 'coordinates': COORDINATE_DOC})
            patches.append(None if patch is None else {k: patch[k] for k in ('image_zyx', 'valid_zyx', 'anchors_zyx', 'spacing_zyx_um')})
        return contexts, patches

    @staticmethod
    def _row_index(table, row):
        return int(row)

    def summary(self):
        tiers = {name: {k: info.get(k) for k in ('level', 'radius_um', 'rows', 'failures', 'bytes')}
                 for name, info in self.manifest.get('image_tiers', {}).items()}
        return {'version': self.manifest['version'], 'brain': self.brain, 'kind': self.kind,
                'band_rows': int(len(self.rows)), 'pool_rows': self.manifest['pool_rows'],
                'band': self.manifest['band'], 'geometry': {k: self.manifest['geometry'][k] for k in ('radius_um', 'max_nodes')},
                'image_tiers': tiers, 'created_at': self.manifest.get('created_at')}


class ContextCache:
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f'Context cache directory does not exist: {self.root}')
        self._entries = {}

    def entry(self, table):
        directory = entry_directory(self.root, table)
        key = str(directory)
        if key in self._entries:
            return self._entries[key]
        manifest_path = directory / 'manifest.json'
        if not manifest_path.is_file():
            identity = table_identity(table)
            raise FileNotFoundError(f"No context cache for brain {identity['brain']} {table.kind} "
                                    f"(pool {identity['pool_sha256'][:12]}); run precompute_context_cache first")
        manifest = json.loads(manifest_path.read_text())
        identity = table_identity(table)
        if not manifest.get('complete') or manifest.get('identity') != identity:
            raise ValueError(f"Context cache identity mismatch for {identity['brain']}/{table.kind}")
        if identity['source_cache'] is not None and cache_identity(identity['source_cache']['path']) != identity['source_cache']:
            raise ValueError('Fragment cache changed since the context cache was built')
        for group, info in [('geometry', manifest['geometry']), *[(f'images/{n}', t) for n, t in manifest.get('image_tiers', {}).items()]]:
            for chunk in info.get('chunks', []):
                path = directory / group / chunk['file']
                if not path.is_file() or path.stat().st_size != chunk['bytes']:
                    raise ValueError(f'Context cache chunk missing or resized: {path}')
        entry = ContextEntry(directory, manifest)
        self._entries[key] = entry
        return entry

    def identity_key(self, table):
        entry = self.entry(table)
        return digest_json({'version': entry.manifest['version'], 'identity': entry.manifest['identity'],
                            'band': entry.manifest['band'], 'band_rows': entry.manifest['band_rows'],
                            'geometry': {k: entry.manifest['geometry'][k] for k in ('radius_um', 'max_nodes')},
                            'tiers': {n: {k: t.get(k) for k in ('level', 'radius_um', 'rows')}
                                      for n, t in entry.manifest.get('image_tiers', {}).items()},
                            'implementation': entry.manifest['implementation']})
