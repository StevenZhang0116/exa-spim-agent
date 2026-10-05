"""Host-owned, label-free fragment neighborhoods for immutable native candidates.

Only explicit geometry leaves the host. Cached graph attributes, GT, source paths
and persistent node/segment IDs are never sent to an agent-authored extractor.
"""
import hashlib
from pathlib import Path

import numpy as np

from .dataset import load_cached_graphs


CONTEXT_VERSION = 'fragment-neighborhood-v1'


def candidate_ref(table, index):
    brain = table.meta.get('provenance', {}).get('brain', '')
    tag = hashlib.sha256(f"{brain}/{table.meta['pool_sha256']}".encode()).hexdigest()[:16]
    return f"ctx:{tag}:{table.kind}:{int(index)}"


def resolve_train_candidate(train, reference):
    """Resolve against TRAIN banks only, even if a heldout handle is guessed."""
    if not isinstance(reference, str) or len(reference) > 100:
        raise ValueError('Use a candidate_ref from TRAIN feedback')
    fields = reference.split(':')
    if len(fields) != 4 or fields[0] != 'ctx' or not fields[3].isascii() or not fields[3].isdigit():
        raise ValueError('Invalid TRAIN candidate_ref')
    index = int(fields[3])
    matches = [(brain, table) for brain, bank in train.items() for table in bank.tables.values()
               if table.kind == fields[2] and candidate_ref(table, index) == reference
               and index < len(table.candidates)]
    if len(matches) != 1:
        raise ValueError('Candidate is absent or ambiguous in the TRAIN pools')
    brain, table = matches[0]
    return brain, table, index


def cache_identity(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}


class FragmentStore:
    """One lazy host graph at a time; shared by all TRAIN and validation tables."""
    def __init__(self, directory, trace=None):
        self.directory = Path(directory)
        self.trace = trace
        self._identity, self._graph, self._tree = None, None, None
        self.image_metadata = None

    def graph(self, provenance):
        identity = provenance['source_cache']
        if cache_identity(identity['path']) != identity:
            raise ValueError('Fragment cache changed since native feature preparation')
        if identity != self._identity:
            # Release the previous graph before loading another large brain.
            self._identity, self._graph, self._tree = None, None, None
            try:
                graph, gt, payload = load_cached_graphs(identity['path'],
                    expect_brain=provenance['brain'], expect_mcl=provenance['mcl'])
            except SystemExit as exc:
                raise ValueError(str(exc)) from exc
            self.image_metadata = {key: payload.get(key) for key in ('img_path', 'anisotropy', 'fragments_path')}
            del gt, payload
            if cache_identity(identity['path']) != identity:
                raise ValueError('Fragment cache changed while loading')
            from scipy.spatial import cKDTree
            tree = getattr(graph, 'kdtree', None)
            if tree is None:
                tree = cKDTree(graph.node_xyz)
            self._identity, self._graph, self._tree = dict(identity), graph, tree
            if self.trace:
                self.trace.emit('fragment_context_loaded', 'Loaded cached fragment geometry',
                                brain=provenance['brain'], nodes=len(graph.node_xyz))
        return self._graph, self._tree


def _anchors(kind, row, occurrence_index):
    if kind == 'merge':
        if occurrence_index != 0:
            raise ValueError('A merge candidate has one location (occurrence_index=0)')
        return [int(row['node_id'])], 1
    occurrences = row.get('occurrences') or [row]
    if not 0 <= occurrence_index < len(occurrences):
        raise ValueError(f'occurrence_index must be in [0, {len(occurrences) - 1}]')
    site = occurrences[occurrence_index]
    return [int(site['node_id_a']), int(site['node_id_b'])], len(occurrences)


def neighborhood(graph, tree, anchors, radius_um, max_nodes):
    """Nearest nodes in a sphere, with anchor nodes always retained.

    Coordinates are xyz microns relative to the anchor midpoint. Edges are the
    induced subgraph, so truncation can remove branches and disconnect paths.
    """
    if any(n < 0 or n >= len(graph.node_xyz) or n not in graph for n in anchors):
        raise ValueError('Candidate anchor is missing from the matching fragment graph')
    center = np.asarray(graph.node_xyz[anchors], dtype=float).mean(axis=0)
    if not np.isfinite(center).all():
        raise ValueError('Candidate location is not finite')
    distance, nodes = tree.query(center, k=max_nodes + 1, distance_upper_bound=radius_um, workers=1)
    near = [(float(d), int(n)) for d, n in zip(np.atleast_1d(distance), np.atleast_1d(nodes))
            if np.isfinite(d)]
    near.sort(key=lambda item: (item[0], item[1]))
    kept = list(dict.fromkeys(anchors))
    for _, node in near:
        if node not in kept and len(kept) < max_nodes:
            kept.append(node)
    truncated = any(node not in kept for _, node in near)
    remap = {node: i for i, node in enumerate(kept)}
    relative = np.asarray(graph.node_xyz[kept], dtype=float) - center
    if not np.isfinite(relative).all():
        raise ValueError('Nonfinite fragment coordinates')
    # IDs are local to this neighborhood. Anchors assign the first segment IDs.
    segments, segment_ids = {}, []
    for node in kept:
        segment = str(graph.node_segment_id(node))
        segment_ids.append(segments.setdefault(segment, len(segments)))
    radii = np.asarray(graph.node_radius[kept], dtype=float)
    data = {'xyz_um': relative.tolist(),
            'radius_um': [float(r) if np.isfinite(r) else None for r in radii],
            'degree': [int(graph.degree[n]) for n in kept],
            'segment': segment_ids,
            'edges': [[remap[a], remap[b]] for a in kept for b in graph.neighbors(a)
                      if b in remap and remap[a] < remap[b]],
            'anchor_nodes': [remap[n] for n in anchors],
            'outside_radius': (np.linalg.norm(relative, axis=1) > radius_um).tolist(),
            'truncated': truncated}
    return data, center.tolist()


class LocalContextProvider:
    def __init__(self, store, table):
        self.store, self.table = store, table

    def context(self, index, *, radius_um=50., max_nodes=128, max_occurrences=1, occurrence_index=0):
        if not 1 <= radius_um <= 200 or not 2 <= max_nodes <= 512 or not 1 <= max_occurrences <= 8:
            raise ValueError('Neighborhood limits: radius 1..200 um, nodes 2..512, occurrences 1..8')
        if not 0 <= index < len(self.table.candidates) or type(occurrence_index) is not int:
            raise ValueError('Invalid candidate/occurrence index')
        graph, tree = self.store.graph(self.table.meta['provenance'])
        row = self.table.candidates[index]
        _, total = _anchors(self.table.kind, row, occurrence_index)
        sites, centers = [], []
        for occurrence in range(occurrence_index, min(total, occurrence_index + max_occurrences)):
            anchors, _ = _anchors(self.table.kind, row, occurrence)
            site, center = neighborhood(graph, tree, anchors, radius_um, max_nodes)
            sites.append(site)
            centers.append(center)
        geometry = {'version': CONTEXT_VERSION, 'kind': self.table.kind, 'radius_um': radius_um,
                    'occurrences_total': total, 'occurrences_used': len(sites),
                    'occurrences_truncated': len(sites) < total, 'sites': sites}
        return geometry, centers


def attach_local_context(banks, directory, trace=None):
    """No graph IO here; ordinary table-only policies remain cheap.

    Attach at runtime instead of changing native_pool.py, whose source hash is
    part of the existing prepared-table identity.
    """
    store = FragmentStore(directory, trace)
    for bank in banks.values():
        for table in bank.tables.values():
            table.local_context = LocalContextProvider(store, table)
    return store
