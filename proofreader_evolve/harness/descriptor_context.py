"""On-demand, label-free descriptor contexts over the complete native candidate pool.

No precomputed context directory is opened. The existing host fragment provider owns
geometry access; descriptor_inputs owns reviewed image access and temporary transport.
"""
from pathlib import Path

import numpy as np

from .image_context import digest_json, file_hash
from .local_context import cache_identity


GEOMETRY_SPEC = {'radius_um': 50., 'max_nodes': 256}
FRAGMENT_FIELDS = ('xyz_um', 'radius_um', 'degree', 'segment', 'edges', 'anchor_nodes', 'outside_radius', 'truncated')
COORDINATE_DOC = {'array_axes': 'zyx', 'anchors_and_nodes': 'patch-relative voxel centers',
                  'xyz_um': 'graph xyz microns relative to the candidate anchor midpoint',
                  'spacing_zyx_um': 'physical image spacing in zyx order',
                  'valid_zyx': 'in-domain mask, not proof of acquired signal in every chunk',
                  'intensity': 'original fused-volume pixels; no preview normalization'}


def table_identity(table):
    provenance = table.meta.get('provenance', {})
    return {'brain': str(provenance.get('brain', '')), 'kind': table.kind,
            'pool_sha256': table.meta['pool_sha256'],
            'features_sha256': table.meta.get('files', {}).get('features.pkl'),
            'source_cache': provenance.get('source_cache')}


class DescriptorContextEntry:
    expose_band = False
    chunk_loads = 0

    def __init__(self, table):
        self.table = table
        self.rows = np.arange(len(table.features), dtype=np.int64)
        self.base_columns = [str(c) for c in table.features.columns if not str(c).startswith('bank_')]
        self.manifest = {'image_tiers': {}}
        self.brain = table.meta.get('provenance', {}).get('brain', '')
        self.kind = table.kind

    @property
    def band(self):
        # Only the explicitly requested boundary scope uses detector rank.
        scores = self.table.features['detector_score'].to_numpy(dtype=float)
        return np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind='stable')

    def has_rows(self, rows):
        rows = np.asarray(rows)
        return (rows >= 0) & (rows < len(self.table.features))

    def contexts(self, table, rows, *, inputs='geometry'):
        if inputs != 'geometry':
            raise ValueError('Images are assembled by descriptor_inputs')
        if not self.has_rows(rows).all():
            raise ValueError('Descriptor row is outside the candidate pool')
        provider = getattr(table, 'local_context', None)
        if provider is None:
            raise ValueError('Descriptor geometry requires the host fragment provider')
        columns = [c for c in self.base_columns if c in table.features.columns]
        columns += [c for c in table.features.columns if str(c).startswith('raw_') and c not in columns]
        frame = table.features.iloc[list(rows), table.features.columns.get_indexer(columns)]
        contexts = []
        for position, row in enumerate(rows):
            geometry, _ = provider.context(int(row), **GEOMETRY_SPEC, max_occurrences=1, occurrence_index=0)
            site = geometry['sites'][0]
            values = frame.iloc[position]
            contexts.append({'kind': table.kind,
                'features': {str(k): float(v) if np.isfinite(v) else None for k, v in values.items()},
                'fragment': {k: site[k] for k in FRAGMENT_FIELDS}, 'radius_um': GEOMETRY_SPEC['radius_um'],
                'level': None, 'channel': None, 'timepoint': None, 'image_radius_um': None,
                'image_available': False, 'occurrence_index': 0,
                'occurrences_total': geometry['occurrences_total'], 'coordinates': COORDINATE_DOC})
        return contexts, [None] * len(contexts)

    def release(self):
        # Contexts and pixels belong to a bounded batch, never to this entry.
        pass

    def summary(self):
        return {'mode': 'on_demand', 'pool_rows': len(self.rows), 'geometry': dict(GEOMETRY_SPEC),
                'candidate_selection': 'all_native_candidates', 'precomputed_context_required': False}


class DescriptorContexts:
    expose_band = False

    def __init__(self):
        self._entries = {}
        self.implementation = {name: file_hash(Path(__file__).with_name(name))
                               for name in ('descriptor_context.py', 'local_context.py', 'detector_prior.py')}

    def entry(self, table):
        key = id(table)
        if key not in self._entries:
            self._entries[key] = DescriptorContextEntry(table)
        return self._entries[key]

    def identity_key(self, table, tiers=None):
        identity = table_identity(table)
        source = identity['source_cache']
        if source is not None and cache_identity(source['path']) != source:
            raise ValueError('Fragment source changed since native feature preparation')
        entry = self.entry(table)
        return digest_json({'version': 'on-demand-descriptor-context-v1', 'table': identity,
                            'base_columns': [c for c in entry.base_columns if c in table.features.columns],
                            'geometry': GEOMETRY_SPEC, 'implementation': self.implementation})
