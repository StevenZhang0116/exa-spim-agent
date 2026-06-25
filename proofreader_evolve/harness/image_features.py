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


# Safety rail on the per-read patch size (voxels per axis). The patch ``shape`` is a
# caller-tunable receptive field — a bigger patch sees more context but every read is
# a cloud fetch, so an unbounded shape would blow up latency and memory with no guard.
# We clamp each axis to this cap (the analogue of ENUM_PARAM_SPEC's rails for the
# enumeration knobs). 512 voxels ~ 383 µm in x/y / 512 µm in z — a generous upper
# bound that still rules out pathological whole-volume reads.
_MAX_PATCH_DIM = 512


def _clamp_shape(shape) -> tuple:
    """Clamp a requested patch shape to (1, _MAX_PATCH_DIM] per axis (>=1, int)."""
    return tuple(max(1, min(_MAX_PATCH_DIM, int(s))) for s in shape)


class LazyImagePatchReader:
    """Reads a cubic image patch centered on a fragment-graph node, on demand.

    Parameters
    ----------
    img_path : str
        Cloud path of the fused ExaSPIM image (e.g. the cache's ``img_path``).
    graph : SkeletonGraph
        The fragment graph whose ``node_voxel(i)`` gives a node's voxel centre.
    default_shape : tuple[int, int, int]
        Patch size in voxels (z, y, x) when the caller does not specify one. A
        caller may pass a larger ``shape`` to any read method for a bigger receptive
        field, but every axis is clamped to ``_MAX_PATCH_DIM`` (a cloud-cost rail).
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
        shape = _clamp_shape(shape) if shape is not None else self._default_shape
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

        LIMITATION: this only measures the two ENDPOINTS, never the region BETWEEN
        them, so two parallel-but-unconnected neurites both read bright and look
        mergeable. For an actual across-the-gap continuity test use
        ``gap_bridge_evidence``, which samples the interior of the gap.
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

    def _gap_um(self, node_a, node_b) -> float:
        """Physical straight-line distance (microns) between two graph nodes.

        Used to size the chord sampling to the gap. Prefers ``graph.node_xyz``
        (already microns); falls back to voxel distance scaled by ``anisotropy``
        (node_voxel is (z,y,x); anisotropy is (x,y,z), so it is reversed to align);
        returns NaN if neither is available so callers can pick a safe default.
        """
        g = self._graph
        xyz = getattr(g, "node_xyz", None)
        if xyz is not None:
            try:
                a = np.asarray(xyz[int(node_a)], dtype=float)
                b = np.asarray(xyz[int(node_b)], dtype=float)
                return float(np.linalg.norm(a - b))
            except Exception:
                pass
        try:
            va = np.asarray(g.node_voxel(int(node_a)), dtype=float)
            vb = np.asarray(g.node_voxel(int(node_b)), dtype=float)
            aniso = getattr(g, "anisotropy", None)
            if aniso is not None:
                aniso_zyx = np.asarray(aniso, dtype=float)[::-1]   # (x,y,z)->(z,y,x)
                return float(np.linalg.norm((va - vb) * aniso_zyx))
            return float(np.linalg.norm(va - vb))
        except Exception:
            return float("nan")

    @staticmethod
    def _auto_n_samples(gap_um, lo=5, hi=25, per_um=0.5, default=9) -> int:
        """Pick a chord sample count from the gap length (~one sample per 2 µm).

        A short gap needs few samples; a long gap needs many so the interior is
        actually traversed (the fixed-9 sampling could step right over the dark
        middle of a long gap). Clamped to ``[lo, hi]``; ``default`` on NaN gap.
        """
        if gap_um != gap_um or gap_um <= 0:   # NaN / nonpositive guard
            return default
        return int(min(hi, max(lo, round(gap_um * per_um) + 2)))

    def _chord_profile(self, node_a, node_b, n_samples, shape) -> list:
        """Sample p90 intensity at ``n_samples`` points along the chord a→b.

        Shared by ``merge_cut_evidence`` and ``gap_bridge_evidence`` — both walk the
        straight line in VOXEL space between two seed nodes and summarize each small
        patch by its 90th percentile (robust to a few hot voxels). Costs
        ``n_samples`` cloud reads (counted in ``n_reads``); not memoized (the chord
        points are unique per call), matching the original merge-cut behavior.
        """
        self._ensure_open()
        va = np.asarray(self._graph.node_voxel(int(node_a)), dtype=float)
        vb = np.asarray(self._graph.node_voxel(int(node_b)), dtype=float)
        shape = _clamp_shape(shape)
        profile = []
        for t in np.linspace(0.0, 1.0, n_samples):
            voxel = tuple(int(round(c)) for c in (va + t * (vb - va)))
            patch = np.asarray(self._img.read(voxel, shape)).astype(np.float32)
            profile.append(float(np.percentile(patch, 90)))
            self.n_reads += 1
        return profile

    def gap_bridge_evidence(
        self, node_a, node_b, n_samples=None, shape=(16, 16, 16)
    ) -> dict:
        """Fluorescence BRIDGE check across a candidate split gap (merge_labels).

        SPLIT-repair evidence — the symmetric counterpart to ``merge_cut_evidence``
        and the across-the-gap test ``gap_connectivity`` cannot do. A ``merge_labels``
        should fire only where the two fragments are ONE neuron the segmentation
        broke, i.e. there is a CONTINUOUS bright bridge of signal spanning the gap.
        This samples the intensity profile along the straight chord between the two
        tips and reports how dim the DIMMEST interior point is relative to the bright
        endpoints:

          * ``bridge_min``    — minimum interior intensity (the weakest link in the
            bridge; excludes the endpoints).
          * ``bridge_ratio``  — ``bridge_min / endpoint_mean``: HIGH (≈1) ⇒ signal
            stays bright all the way across ⇒ a real continuation ⇒ SAFE to merge;
            LOW (≪1) ⇒ the gap goes dark in the middle ⇒ two separate structures ⇒
            do NOT merge (a join here would create a merge error).
          * ``bridge_pos``    — fractional position (0..1) of the dimmest point.
          * ``endpoint_mean`` / ``profile`` — the bright reference and the raw samples.

        Note this is the same chord profile ``merge_cut_evidence`` computes; only the
        decision flips — there a LOW interior valley means "split", here a HIGH
        interior bridge means "merge". ``n_samples`` defaults to ADAPTIVE: it scales
        with the physical gap length (``_auto_n_samples``) so a long gap is actually
        traversed rather than stepped over. Costs ``n_samples`` cloud reads (cached
        machinery shared with merge_cut), so gate it behind cheap geometric filters
        exactly like the others. NaNs on read failure.
        """
        try:
            if n_samples is None:
                n_samples = self._auto_n_samples(self._gap_um(node_a, node_b))
            profile = self._chord_profile(node_a, node_b, n_samples, shape)
            endpoint_mean = float(np.mean([profile[0], profile[-1]]))
            interior = profile[1:-1] if n_samples > 2 else profile
            bridge_min = float(np.min(interior))
            bridge_idx = int(np.argmin(profile))
            # --- SHAPE features of the bridge profile (added) ---------------------
            # bridge_ratio collapses the whole profile to one point (the dimmest).
            # These describe the SHAPE so the policy can tell a single dim sample (a
            # noise dip, probably still one neuron) from a WIDE dark valley (a true
            # gap between two structures). All normalized by endpoint_mean so they are
            # brain-agnostic; the policy chooses whether to use them.
            prof = np.asarray(profile, dtype=float)
            ref = endpoint_mean if endpoint_mean > 0 else 1.0
            interior_arr = prof[1:-1] if n_samples > 2 else prof
            # valley_frac: fraction of interior samples below half the endpoint
            # brightness (how WIDE the dark stretch is, not just how deep).
            valley_frac = float(np.mean(interior_arr < 0.5 * ref)) if len(interior_arr) else float("nan")
            # bridge_mean_ratio: mean interior brightness / endpoints (a soft version
            # of bridge_ratio — robust to one outlier dip).
            bridge_mean_ratio = float(np.mean(interior_arr) / ref) if len(interior_arr) else float("nan")
            # profile_cv: coefficient of variation along the profile (flat bright
            # bridge ≈ 0; a deep narrow dip raises it). Shape, not depth.
            profile_cv = float(np.std(prof) / ref) if len(prof) else float("nan")
            return {
                "profile": profile,
                "endpoint_mean": endpoint_mean,
                "bridge_min": bridge_min,
                "bridge_ratio": (bridge_min / endpoint_mean
                                 if endpoint_mean > 0 else float("nan")),
                "bridge_pos": bridge_idx / (n_samples - 1) if n_samples > 1 else 0.0,
                "n_samples": n_samples,
                # shape features (added)
                "valley_frac": valley_frac,
                "bridge_mean_ratio": bridge_mean_ratio,
                "profile_cv": profile_cv,
            }
        except Exception:
            return {
                "profile": [], "endpoint_mean": float("nan"),
                "bridge_min": float("nan"), "bridge_ratio": float("nan"),
                "bridge_pos": float("nan"), "n_samples": 0,
                "valley_frac": float("nan"), "bridge_mean_ratio": float("nan"),
                "profile_cv": float("nan"),
            }

    def merge_cut_evidence(
        self, seed_a_node, seed_b_node, n_samples=None, shape=(16, 16, 16)
    ) -> dict:
        """Fluorescence VALLEY check across a candidate merge cut (split_label).

        MERGE-repair evidence — the opposite question to ``gap_bridge_evidence``. A
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
          * ``n_samples``     — points actually sampled (useful when adaptive).
        ``n_samples`` defaults to ADAPTIVE (scales with the seed-to-seed distance via
        ``_auto_n_samples``) so a long arm span is actually traversed; pass an int to
        force a fixed count (the old default was 9). Costs ``n_samples`` cloud reads,
        so gate it behind cheap geometric filters exactly like the others. NaNs on
        read failure.
        """
        try:
            if n_samples is None:
                n_samples = self._auto_n_samples(
                    self._gap_um(seed_a_node, seed_b_node))
            profile = self._chord_profile(seed_a_node, seed_b_node, n_samples, shape)
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
                "n_samples": n_samples,
            }
        except Exception:
            return {
                "profile": [], "endpoint_mean": float("nan"),
                "valley": float("nan"), "valley_ratio": float("nan"),
                "valley_pos": float("nan"), "n_samples": 0,
            }


class RecordingImageReader:
    """Transparent wrapper that LOGS every image-evidence call the policy makes.

    The policy already pays cloud reads via ``read_patch`` / ``gap_connectivity`` /
    ``gap_bridge_evidence`` / ``merge_cut_evidence`` for the few candidates it gates
    through its cheap geometric filters. This wrapper observes those calls — it adds
    NO reads of its own — and records ``(method, node ids, returned summary)`` for
    each, INCLUDING raw ``read_patch`` cubes (summarized into generic intensity
    scalars) so a policy that invents its own image feature still leaves a labelled
    trace. The harness then labels each record TRUE/NON by ground truth (train split
    only) and surfaces it in the failure report, turning the image evidence the
    policy ALREADY fetched into a labelled learning signal for the reviser (which
    image thresholds separate real merges/splits from false ones) — without any
    extra cost and without ever reading held-out image data the policy didn't
    already touch.

    Delegates remaining attributes (``n_reads``, ``default_shape``, …) to the wrapped
    reader, so it is a drop-in for ``ctx["read_image_patch"]``. ``read_patch`` is
    overridden (not delegated) so its cubes are recorded.
    """

    def __init__(self, inner):
        self._inner = inner
        self.records: list = []   # [{"method","node_a","node_b","result"}]

    @staticmethod
    def _patch_summary(patch) -> dict:
        """CPU-side scalar summary of a raw cube — NO extra cloud read.

        When the policy invents its OWN image feature via ``read_patch`` (e.g. a
        self-made MIP, signal-occupancy, variance), the harness has no idea what it
        computed. We can't record an arbitrary derived feature, but we CAN record a
        fixed, generic summary of the cube the policy looked at, so the failure
        report can show whether THOSE cubes separate real splits from false joins by
        train GT. These are the cheapest universally-meaningful intensity stats:
          * ``mean`` / ``max`` / ``p90`` — overall brightness.
          * ``occupancy`` — fraction of voxels above half the patch max (a
            threshold-free 'how much of this cube is signal vs background' proxy;
            a thin neurite fills few voxels, a blob fills many).
          * ``std`` — intensity spread (flat background vs structured signal).
        All float; NaNs if the patch is empty / unreadable.
        """
        try:
            p = np.asarray(patch, dtype=np.float32).ravel()
            if p.size == 0:
                raise ValueError("empty patch")
            pmax = float(p.max())
            occ = float((p > 0.5 * pmax).mean()) if pmax > 0 else 0.0
            return {
                "mean": float(p.mean()), "max": pmax,
                "p90": float(np.percentile(p, 90)),
                "occupancy": occ, "std": float(p.std()),
            }
        except Exception:
            return {"mean": float("nan"), "max": float("nan"),
                    "p90": float("nan"), "occupancy": float("nan"),
                    "std": float("nan")}

    def read_patch(self, node_id, shape=None):
        """Record a generic summary of every raw cube the policy reads itself.

        Forwards to the inner reader (the real cloud read + cache) and logs a
        fixed scalar summary keyed on the single ``node_id`` — so a policy that
        builds a custom intensity feature still leaves a labelled trace in the
        failure report. Adds no cloud read of its own (it summarizes the cube the
        inner read already returned). Overridden explicitly because ``__getattr__``
        would otherwise transparently forward and bypass recording.
        """
        patch = self._inner.read_patch(node_id, shape=shape)
        self.records.append({
            "method": "read_patch",
            "node_a": int(node_id), "node_b": None,
            "result": self._patch_summary(patch),
        })
        return patch

    def gap_connectivity(self, node_a, node_b, shape=None) -> dict:
        out = self._inner.gap_connectivity(node_a, node_b, shape=shape)
        self.records.append({
            "method": "gap_connectivity",
            "node_a": int(node_a), "node_b": int(node_b),
            "result": dict(out),
        })
        return out

    def merge_cut_evidence(self, seed_a_node, seed_b_node, **kw) -> dict:
        out = self._inner.merge_cut_evidence(seed_a_node, seed_b_node, **kw)
        # Drop the (long) per-sample profile from the record; keep the scalars.
        rec = {k: v for k, v in out.items() if k != "profile"}
        self.records.append({
            "method": "merge_cut_evidence",
            "node_a": int(seed_a_node), "node_b": int(seed_b_node),
            "result": rec,
        })
        return out

    def gap_bridge_evidence(self, node_a, node_b, **kw) -> dict:
        out = self._inner.gap_bridge_evidence(node_a, node_b, **kw)
        rec = {k: v for k, v in out.items() if k != "profile"}
        self.records.append({
            "method": "gap_bridge_evidence",
            "node_a": int(node_a), "node_b": int(node_b),
            "result": rec,
        })
        return out

    def __getattr__(self, name):
        # read_patch, n_reads, _ensure_open, default_shape, etc. -> the real reader.
        return getattr(self._inner, name)
