"""Trusted image IO, registration receipts and immutable candidate patch caching."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit

import numpy as np

from .image_contract import IMAGE_VERSION
from .image_coordinates import ImageGeometry
from .local_context import _anchors

DEFAULT_ALIGNMENT = Path(__file__).resolve().parents[2] / 'configs' / 'image_alignment.json'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def kv_spec(uri):
    parsed = urlsplit(uri)
    if parsed.scheme not in ('s3', 'gs') or not parsed.netloc:
        raise ValueError('Image sources require a host-configured s3:// or gs:// OME-Zarr URI')
    if parsed.scheme == 'gs' and not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        credential = DEFAULT_ALIGNMENT.parent / 'allen-nd-goog-f5d46dbfa2cd.json'
        if credential.is_file():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credential)
    os.environ.setdefault('AWS_EC2_METADATA_DISABLED', 'true')
    return {'driver': 'gcs' if parsed.scheme == 'gs' else 's3', 'bucket': parsed.netloc,
            'path': parsed.path.lstrip('/').rstrip('/') + '/'}


class ImageSource:
    """OME-Zarr v2 reader with explicit metadata and bounded IO waits."""
    def __init__(self, uri, *, graph_to_world=None, timeout=45.):
        import tensorstore as ts
        self.timeout = timeout
        # Native cache metadata identifies one array, e.g. fused.zarr/0.
        marker = uri.find('.zarr')
        if marker < 0:
            raise ValueError('This image adapter supports OME-Zarr only; unsupported formats fail explicitly')
        self.root = uri[:marker + 5].rstrip('/') + '/'
        self.kv = ts.KvStore.open(kv_spec(self.root)).result(timeout=timeout)
        self.attributes = self._json('.zattrs')
        scales = self.attributes.get('multiscales', [])
        if len(scales) != 1:
            raise ValueError('Image must declare one OME multiscale')
        self.datasets = [d['path'] for d in scales[0]['datasets']]
        if any(not isinstance(p, str) or p.startswith('/') or '..' in p.split('/') for p in self.datasets):
            raise ValueError('Invalid relative multiscale dataset path')
        requested = uri[marker + 5:].strip('/')
        if requested and requested not in self.datasets:
            raise ValueError('Cached image array does not exist in multiscale metadata')
        self.graph_to_world = graph_to_world
        self._levels, self._arrays = {}, {}
        self.level(0)
        self.identity = {'root': self.root, 'attributes_sha256': digest_json(self.attributes),
                         'base_array_sha256': digest_json(self._levels[0][1])}

    def _json(self, name):
        value = self.kv.read(name).result(timeout=self.timeout)
        if value.state != 'value':
            raise FileNotFoundError(f'Missing image metadata: {self.root}{name}')
        return json.loads(value.value)

    def level(self, level):
        if type(level) is not int or not 0 <= level < len(self.datasets):
            raise ValueError('Requested image resolution level is unavailable')
        if level not in self._levels:
            dataset = self.datasets[level]
            metadata = self._json(dataset + '/.zarray')
            if metadata.get('zarr_format') != 2:
                raise ValueError('Unsupported image Zarr version')
            geometry = ImageGeometry.from_ngff(self.attributes, metadata, dataset, self.graph_to_world)
            self._levels[level] = geometry, metadata
        return self._levels[level][0]

    def read(self, anchors_xyz, radius_um=40., level=0, channel=0, timepoint=0):
        import tensorstore as ts
        geometry = self.level(level)
        origin, shape, anchors = geometry.crop(anchors_xyz, radius_um)
        selection, permutation, destination = geometry.selection(origin, shape, channel, timepoint)
        if level not in self._arrays:
            # TensorStore exposes array storage axes as-is; transpose only from
            # the axes declared in the OME metadata, never from a path heuristic.
            self._arrays[level] = ts.open({'driver': 'zarr',
                'kvstore': kv_spec(self.root + geometry.dataset),
                'context': {'cache_pool': {'total_bytes_limit': 128 * 1024**2},
                            'data_copy_concurrency': {'limit': 2}}},
                read=True, write=False, open=True).result(timeout=self.timeout)
        array = self._arrays[level]
        if tuple(array.shape) != geometry.shape or any(array.domain.inclusive_min):
            raise ValueError('Image storage domain differs from the declared OME array')
        block = np.asarray(array[selection].read().result(timeout=self.timeout)).transpose(permutation)
        pixels = np.zeros(tuple(shape), dtype=block.dtype.newbyteorder('='))
        valid = np.zeros(tuple(shape), dtype=bool)
        pixels[destination], valid[destination] = block, True
        if not np.isfinite(block).all():
            raise ValueError('Image contains nonfinite values')
        return {'image_zyx': pixels, 'valid_zyx': valid, 'anchors_zyx': anchors,
                'spacing_zyx_um': geometry.scale_xyz_um[::-1].copy()}, {
                **geometry.summary(), 'origin_zyx': origin.tolist(), 'shape_zyx': shape.tolist(),
                'center_xyz_um': np.asarray(anchors_xyz).mean(axis=0).tolist(),
                'level': level, 'channel': channel, 'timepoint': timepoint,
                'valid_fraction': float(valid.mean()), 'dtype': str(pixels.dtype)}


class CandidateImageStore:
    def __init__(self, directory, alignment_path=DEFAULT_ALIGNMENT, trace=None, max_cache_bytes=20 * 1024**3):
        self.directory, self.trace = Path(directory), trace
        self.alignment_path = Path(alignment_path)
        self.reviews = json.loads(self.alignment_path.read_text()).get('brains', {}) if self.alignment_path.exists() else {}
        self.max_cache_bytes = max_cache_bytes
        self.cache_bytes = sum(p.stat().st_size for p in self.directory.glob('*/patch.npz')) if self.directory.exists() else 0
        self.sources = {}

    def source(self, table, *, reviewed=True):
        provider = getattr(table, 'local_context', None)
        if provider is None:
            raise ValueError('Image access requires the matching host fragment provider')
        provenance = table.meta['provenance']
        graph, _ = provider.store.graph(provenance)
        metadata = provider.store.image_metadata
        brain = provenance['brain']
        review = self.reviews.get(brain, {})
        if reviewed and review.get('status') != 'reviewed':
            raise ValueError(f'Image alignment has not been reviewed for brain {brain}; run check_image_alignment first')
        uri = review.get('image_uri') or metadata.get('img_path')
        if not isinstance(uri, str):
            raise ValueError('Matching cache has no image URI')
        transform = review.get('graph_to_world_um')
        key = digest_json({'uri': uri, 'transform': transform})
        if key not in self.sources:
            self.sources[key] = ImageSource(uri, graph_to_world=transform)
        source = self.sources[key]
        anisotropy = np.asarray(metadata.get('anisotropy'), dtype=float)
        if anisotropy.shape != (3,) or not np.allclose(anisotropy, graph.anisotropy):
            raise ValueError('Fragment graph and cache disagree about coordinate units')
        if transform is None and not np.allclose(anisotropy, source.level(0).scale_xyz_um):
            raise ValueError('Image base spacing and fragment cache spacing differ; an explicit registration is required')
        if reviewed and (review.get('source_cache') != provenance['source_cache']
                         or review.get('image_identity') != source.identity):
            raise ValueError('Image/cache identity changed since alignment review')
        return source, graph

    def patch(self, table, index, spec, occurrence=0, *, reviewed=True):
        started = time.monotonic()
        source, graph = self.source(table, reviewed=reviewed)
        anchors, total = _anchors(table.kind, table.candidates[index], occurrence)
        xyz = np.asarray(graph.node_xyz[anchors], dtype=float)
        geometry = source.level(spec['level'])
        identity = {'version': IMAGE_VERSION, 'image': source.identity, 'geometry': geometry.summary(),
                    'level_metadata_sha256': digest_json(source._levels[spec['level']][1]),
                    'implementation': {name: file_hash(Path(__file__).with_name(name)) for name in
                                       ('image_context.py', 'image_coordinates.py', 'image_contract.py')},
                    'source_cache': table.meta['provenance']['source_cache'], 'kind': table.kind,
                    'pool': table.meta['pool_sha256'], 'row': int(index), 'occurrence': occurrence,
                    'anchors_xyz_um': xyz.tolist(),
                    'request': {k: spec[k] for k in ('radius_um', 'level', 'channel', 'timepoint')}}
        key = digest_json(identity)
        directory = self.directory / key
        cached = directory.is_dir()
        if not cached:
            data, metadata = source.read(xyz, **identity['request'])
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.cache_bytes + sum(a.nbytes for a in data.values()) > self.max_cache_bytes:
                raise ValueError('Image cache byte budget exhausted')
            with tempfile.TemporaryDirectory(prefix='.staging_', dir=self.directory) as tmp:
                staged = Path(tmp) / 'entry'
                staged.mkdir()
                np.savez(staged / 'patch.npz', **data)
                metadata.update(identity=identity, occurrences_total=total,
                                patch_sha256=file_hash(staged / 'patch.npz'))
                (staged / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
                staged.rename(directory)
            self.cache_bytes += (directory / 'patch.npz').stat().st_size
        metadata = json.loads((directory / 'metadata.json').read_text())
        path = directory / 'patch.npz'
        if metadata['identity'] != identity or metadata['patch_sha256'] != file_hash(path):
            raise ValueError('Cached image patch identity/checksum mismatch')
        if self.trace:
            self.trace.emit('candidate_image_ready', 'Loaded candidate image patch',
                brain=table.meta['provenance']['brain'], kind=table.kind, row=int(index), occurrence=occurrence,
                cache_key=key, cached=cached, wall_seconds=time.monotonic() - started,
                shape_zyx=metadata['shape_zyx'], valid_fraction=metadata['valid_fraction'])
        return path, metadata


def attach_image_context(banks, directory, alignment_path=DEFAULT_ALIGNMENT, trace=None):
    store = CandidateImageStore(directory, alignment_path, trace)
    for bank in banks.values():
        for table in bank.tables.values():
            table.image_context = store
    return store
