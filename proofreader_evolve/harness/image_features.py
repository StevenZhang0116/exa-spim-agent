"""
Lazy, cached image-patch reader for the proofreading policy.

Why this exists
---------------
The policy decides "are these two fragments the same neuron?" from skeleton
geometry alone — it has NO access to the raw fluorescence image, which is the
most direct evidence (does signal actually connect across the gap?). The scoring
path deliberately drops the image for speed, so this provides an OPT-IN handle
the policy can call ONLY for the candidates it cares about.

Cost discipline (important): every patch read is a GCS/S3 fetch. A policy can
propose thousands of candidates, so reading a patch per candidate eagerly would
dominate runtime. This reader is therefore:
  - LAZY: the TensorStore client is opened on the first read, not at construction
    (so enabling it costs nothing if the policy never calls it);
  - CACHED: repeated reads of the same (node, shape) are memoized within a
    generation, so re-querying a node is free.
The policy is expected to gate image reads behind cheap geometric filters (read
the gap patch only for candidates that already pass gap/size/margin tests).
"""

from __future__ import annotations

import numpy as np


class LazyImagePatchReader:
    """Reads a cubic image patch centered on a fragment-graph node, on demand.

    Parameters
    ----------
    img_path : str
        Cloud path of the fused ExaSPIM image (e.g. the cache's ``img_path``).
    graph : SkeletonGraph
        The fragment graph whose ``node_voxel(i)`` gives a node's voxel centre.
    default_shape : tuple[int, int, int]
        Patch size in voxels (z, y, x) when the caller does not specify one.
    """

    def __init__(self, img_path, graph, default_shape=(48, 48, 48)):
        self._img_path = img_path
        self._graph = graph
        self._default_shape = tuple(default_shape)
        self._img = None                 # opened lazily on first read
        self._cache: dict = {}           # (node_id, shape) -> patch ndarray
        self.n_reads = 0                 # cloud reads actually performed (cost meter)

    def _ensure_open(self):
        if self._img is None:
            # Imported here so merely enabling the reader pulls in nothing until used.
            from agentic_neuron_proofreader.utils import img_util
            self._img = img_util.TensorStoreImage(self._img_path)

    def read_patch(self, node_id, shape=None) -> np.ndarray:
        """Return the raw image patch centered on ``node_id``'s voxel.

        Memoized per (node, shape); only the first call for a given key hits the
        cloud. ``shape`` defaults to ``default_shape``.
        """
        shape = tuple(shape) if shape is not None else self._default_shape
        key = (int(node_id), shape)
        if key in self._cache:
            return self._cache[key]
        self._ensure_open()
        voxel = self._graph.node_voxel(int(node_id))   # (z, y, x) voxel centre
        patch = np.asarray(self._img.read(voxel, shape))
        self._cache[key] = patch
        self.n_reads += 1
        return patch

    def gap_connectivity(self, node_a, node_b, shape=None) -> dict:
        """Cheap fluorescence summary at each tip — a 'does signal continue?' proxy.

        SPLIT-repair evidence (the merge_labels question): is there bright signal on
        BOTH sides of a gap? Reads a small patch at each endpoint and returns summary
        intensities (mean / max / 90th pct). A real continuation tends to have bright
        signal filling both patches; a spurious join across background has a dim
        side. The policy decides how to threshold; this just surfaces the numbers
        without forcing an interpretation. Returns NaNs if a read fails.
        """
        out = {}
        for tag, n in (("a", node_a), ("b", node_b)):
            try:
                p = self.read_patch(n, shape).astype(np.float32)
                out[f"mean_{tag}"] = float(p.mean())
                out[f"max_{tag}"] = float(p.max())
                out[f"p90_{tag}"] = float(np.percentile(p, 90))
            except Exception:
                out[f"mean_{tag}"] = out[f"max_{tag}"] = out[f"p90_{tag}"] = float("nan")
        return out

    def merge_cut_evidence(
        self, seed_a_node, seed_b_node, n_samples=9, shape=(16, 16, 16)
    ) -> dict:
        """Fluorescence VALLEY check across a candidate merge cut (split_label).

        MERGE-repair evidence — the opposite question to ``gap_connectivity``. A
        ``split_label`` should fire only where ONE label actually covers TWO
        neurites. The image tell is whether the signal between the two arm seeds
        DIPS through a valley near the cut (two adjacent bright structures that only
        touch) or stays bright the whole way (one continuous neuron the skeleton
        merely branched). This samples the intensity profile along the straight
        chord between ``seed_a_node`` and ``seed_b_node`` and surfaces a valley
        statistic; the policy decides how to threshold (it does NOT interpret here).

        Sampling is along the chord in VOXEL space at ``n_samples`` points, each a
        small ``shape`` patch summarized by its 90th-percentile intensity (robust to
        a few hot voxels). Because the two seeds sit deep in each arm, the chord
        passes through the contact region; a genuine merge shows a low interior
        minimum relative to the bright endpoints.

        Returns a dict:
          * ``profile``       — list of per-sample p90 intensities, seed_a → seed_b.
          * ``endpoint_mean`` — mean of the two endpoint intensities (the "bright"
            reference).
          * ``valley``        — minimum interior intensity (excludes the endpoints).
          * ``valley_ratio``  — ``valley / endpoint_mean``: LOW (≪1) ⇒ a real valley
            ⇒ merge-like; ≈1 ⇒ continuous signal ⇒ likely one neuron (do NOT split).
          * ``valley_pos``    — fractional position (0..1) of the minimum along the
            chord (≈0.5 ⇒ a valley right between the arms, the cleanest merge tell).
        Costs ``n_samples`` cloud reads (cached), so gate it behind cheap geometric
        filters exactly like ``gap_connectivity``. NaNs on read failure.
        """
        try:
            self._ensure_open()
            va = np.asarray(self._graph.node_voxel(int(seed_a_node)), dtype=float)
            vb = np.asarray(self._graph.node_voxel(int(seed_b_node)), dtype=float)
            shape = tuple(shape)
            profile = []
            for t in np.linspace(0.0, 1.0, n_samples):
                voxel = tuple(int(round(c)) for c in (va + t * (vb - va)))
                patch = np.asarray(self._img.read(voxel, shape)).astype(np.float32)
                profile.append(float(np.percentile(patch, 90)))
                self.n_reads += 1
            endpoint_mean = float(np.mean([profile[0], profile[-1]]))
            interior = profile[1:-1] if n_samples > 2 else profile
            valley = float(np.min(interior))
            valley_idx = int(np.argmin(profile))
            return {
                "profile": profile,
                "endpoint_mean": endpoint_mean,
                "valley": valley,
                "valley_ratio": (valley / endpoint_mean
                                 if endpoint_mean > 0 else float("nan")),
                "valley_pos": valley_idx / (n_samples - 1) if n_samples > 1 else 0.0,
            }
        except Exception:
            return {
                "profile": [], "endpoint_mean": float("nan"),
                "valley": float("nan"), "valley_ratio": float("nan"),
                "valley_pos": float("nan"),
            }
