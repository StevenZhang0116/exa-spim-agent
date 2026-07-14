"""
Run one candidate proofreader and score it.

A "candidate" = the current state of the evolved artifacts
(``artifacts/heuristics.py`` + ``artifacts/rules.md``). This module executes the
policy to produce edits, then scores those edits on a chosen set of GT skeletons
via the real metric framework. It is the bridge between the evolved program and
the deterministic scorer, and is the unit the evolution loop calls each
generation (once on train for feedback, once on held-out for gating).

Importantly the heuristics file is (re)loaded from disk each call, so after the
agent rewrites it the next evaluation picks up the new policy with no restart.
"""

from __future__ import annotations

import heapq
import importlib.util
import json
import os
import signal
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import KDTree

from proofreader_evolve.harness import dataset as ds
from proofreader_evolve.harness import image_confidence as imgconf
from proofreader_evolve.harness import scoring


def _node_xyz(g, n):
    return np.asarray(g.node_xyz[n], dtype=float)


def _arm_inner_node(g, start, prefer_dir=None, walk_um: float = 4.0):
    """A node ~``walk_um`` of cable into the arm leaving ``start`` (for a tangent).

    ``start`` is a SplitSite endpoint. With ``prefer_dir`` given and ``start`` not a
    simple tip, follow the neighbor whose first step best aligns with ``prefer_dir``
    (so we trace the arm that CONTINUES that line, not a side branch). The walk stops
    at ``walk_um`` cable or the next tip/branch. Returns the inner node, or None if
    ``start`` is isolated. Pure fragment geometry — GT-free, safe on held-out.
    """
    nbrs = list(g.neighbors(start))
    if not nbrs:
        return None
    p0 = _node_xyz(g, start)
    if prefer_dir is not None and len(nbrs) > 1:
        def _score(nb):
            v = _node_xyz(g, nb) - p0
            nrm = float(np.linalg.norm(v))
            return float(np.dot(v / nrm, prefer_dir)) if nrm > 0 else -2.0
        first = max(nbrs, key=_score)
    else:
        first = nbrs[0]
    prev, cur = start, first
    cable = g.dist(start, first)
    while cable < walk_um:
        nxt = [n for n in g.neighbors(cur) if n != prev]
        if len(nxt) != 1:  # tip (0) or branch (>=2) — stop the walk here
            break
        prev, cur = cur, nxt[0]
        cable += g.dist(prev, cur)
    return cur


def _on_graph_path_um(g, src, dst, max_steps=400, max_um=None):
    """Shortest CABLE distance from ``src`` to ``dst`` along existing edges, or None.

    Bounded Dijkstra over the fragment graph — None if ``dst`` is not reached within
    ``max_steps`` node-visits (and ``max_um`` cable, if given). The bound is a
    PERFORMANCE rail, not a decision threshold: the dominant case (two endpoints in
    different components) fails fast, and a real reconnection is almost always LOCAL,
    mirroring ``dataset._arms_reconverge``. A path longer than the bound reports None
    (conservative miss) — the policy then sees "no short on-graph path", never a wrong
    distance. Parameter-free as a SIGNAL: it returns the measured micron distance and
    lets the evolved policy decide what (if any) threshold is meaningful.
    """
    if src is None or dst is None or src == dst:
        return 0.0 if src == dst else None
    best = {src: 0.0}
    pq = [(0.0, src)]
    steps = 0
    while pq and steps < max_steps:
        d, cur = heapq.heappop(pq)
        steps += 1
        if cur == dst:
            return float(d)
        if d > best.get(cur, float("inf")):
            continue
        if max_um is not None and d > max_um:
            continue
        for nbr in g.neighbors(cur):
            nd = d + g.dist(cur, nbr)
            if nd < best.get(nbr, float("inf")):
                best[nbr] = nd
                heapq.heappush(pq, (nd, nbr))
    return None


def _arm_cable_um(g, endpoint, max_steps=400, max_um=200.0):
    """Total cable length of the fragment reachable from ``endpoint`` along its label.

    Bounded BFS that sums edge lengths over the same-segment component containing
    ``endpoint`` (a measure of fragment MATURITY: a long mature cable ending in a tip
    is a more credible broken-neuron than a short stub). Bounds are performance rails;
    a fragment larger than the bound reports the (capped) cable seen so far. Returns
    None if the endpoint is isolated. Parameter-free as a signal — no threshold baked
    in; the policy decides what counts as "mature".
    """
    if endpoint is None:
        return None
    try:
        label = g.node_segment_id(endpoint)
    except Exception:
        return None
    seen = {endpoint}
    frontier = [endpoint]
    cable = 0.0
    steps = 0
    while frontier and steps < max_steps:
        cur = frontier.pop()
        steps += 1
        for nbr in g.neighbors(cur):
            if nbr in seen:
                continue
            if g.node_segment_id(nbr) != label:
                continue
            seen.add(nbr)
            cable += g.dist(cur, nbr)
            if cable >= max_um:
                return float(max_um)
            frontier.append(nbr)
    return float(cable) if seen != {endpoint} else 0.0


def _dist_to_branch_um(g, endpoint, max_steps=400, max_um=50.0):
    """Cable distance from ``endpoint`` to the nearest branch node (degree>=3), or None.

    Walks outward within the same label until it hits a degree>=3 node, returning the
    cable traversed. None = no branch within the bound (an effectively branch-free
    local neighbourhood). This is the RAW geometry of where on a shaft a tip attaches
    — a tip joining mid-span of a long branchless shaft vs. right at a junction — left
    un-thresholded so the policy keys on the distance itself.
    """
    if endpoint is None:
        return None
    try:
        label = g.node_segment_id(endpoint)
    except Exception:
        return None
    # Dijkstra outward; stop at first degree>=3 node encountered (excluding start).
    best = {endpoint: 0.0}
    pq = [(0.0, endpoint)]
    steps = 0
    while pq and steps < max_steps:
        d, cur = heapq.heappop(pq)
        steps += 1
        if cur != endpoint:
            try:
                if int(g.degree[cur]) >= 3:
                    return float(d)
            except Exception:
                pass
        if d > max_um:
            continue
        for nbr in g.neighbors(cur):
            if g.node_segment_id(nbr) != label:
                continue
            nd = d + g.dist(cur, nbr)
            if nd < best.get(nbr, float("inf")):
                best[nbr] = nd
                heapq.heappush(pq, (nd, nbr))
    return None


def _split_site_geom(g, s):
    """Cheap, GT-free geometry for one SplitSite, from the fragment graph.

    Returns a dict:
      colinear_cos — straightness of (arm A -> gap -> arm B): +1 is a clean colinear
                     continuation across the gap, ~0 a right-angle join, <0 the arms
                     double back. Mean of two half-cosines (arm A into the gap; the
                     gap into B's continuing arm). None if a tangent is undefined.
      deg_a, deg_b — graph degree of the two endpoints (A is always a tip = 1; B
                     tells tip(1) / shaft(2) / branch(3+)).
      rad_a, rad_b — neurite radius at each endpoint (None if no radius array).
      rad_ratio    — max/min of the two radii (None if either missing/zero); far from
                     1.0 means two different cable calibers fused (less likely ONE
                     neuron the segmentation broke).
      cos_a, cos_b — the TWO half-cosines separately (arm A vs gap; gap vs arm B).
                     colinear_cos is their mean; exposing both lets the policy see
                     ASYMMETRY (one arm aligned, the other bent — a tip grazing a
                     shaft tends to have one low cosine).
      tip_tangent_cos — cosine between the two arms' own tangents directly (not via
                     the gap direction). +1 = the two cables run parallel/continuous;
                     a different view of continuity than colinear_cos.

      --- TOPOLOGY (pure graph structure, brain-INDEPENDENT — no thresholds baked in;
          the policy / report decide what value is meaningful) ---
      same_component  — True if A and B are ALREADY connected on the graph (so merging
                     them would CLOSE A LOOP). Real neurons are ~tree-like, so a small
                     gap whose two ends already connect is usually a redundant
                     reconnection — a strong NON-merge prior. None if undetermined.
      graph_path_um   — shortest on-graph CABLE distance A->B (None if disconnected or
                     beyond the bounded search). With a small euclidean gap, a SHORT
                     graph path = the two tips are nearly adjacent on one strand
                     already (loop); a long/absent path = genuinely separate strands.
      cable_a, cable_b — total cable of each endpoint's fragment (maturity): a long
                     mature cable ending in a tip is a more credible broken neuron than
                     a tiny stub. Capped by a performance rail.
      cable_min       — min(cable_a, cable_b): the SMALLER (more fragile) fragment.
      dist_to_branch_b — for a shaft/branch endpoint B, cable distance to the nearest
                     degree>=3 node (None if branch-free locally). Where on the shaft
                     the tip attaches — raw, un-thresholded.
    Every value is a function of fragment geometry/topology only, so it is leak-free
    and the same on train and held-out — the policy may key on these directly.
    """
    out = {"colinear_cos": None, "deg_a": None, "deg_b": None,
           "rad_a": None, "rad_b": None, "rad_ratio": None,
           "cos_a": None, "cos_b": None, "tip_tangent_cos": None,
           "same_component": None, "graph_path_um": None,
           "cable_a": None, "cable_b": None, "cable_min": None,
           "dist_to_branch_b": None}
    if g is None:
        return out
    a, b = getattr(s, "node_a", None), getattr(s, "node_b", None)
    if a is None or b is None:
        return out
    try:
        out["deg_a"] = int(g.degree[a])
        out["deg_b"] = int(g.degree[b])
    except Exception:
        pass
    radius = getattr(g, "node_radius", None)
    if radius is not None:
        try:
            ra, rb = float(radius[a]), float(radius[b])
            out["rad_a"] = ra if ra > 0 else None
            out["rad_b"] = rb if rb > 0 else None
            if ra > 0 and rb > 0:
                out["rad_ratio"] = max(ra, rb) / min(ra, rb)
        except Exception:
            pass
    try:
        pa, pb = _node_xyz(g, a), _node_xyz(g, b)
        gap_vec = pb - pa
        gnorm = float(np.linalg.norm(gap_vec))
        if gnorm > 0:
            gdir = gap_vec / gnorm
            ai = _arm_inner_node(g, a)                    # A is a tip: its one arm
            bi = _arm_inner_node(g, b, prefer_dir=gdir)   # B's arm continuing the gap
            cos_a = cos_b = None
            ta_u = tb_u = None                            # unit tangents (for tip_tangent_cos)
            if ai is not None:
                ta = pa - _node_xyz(g, ai)                # outward tangent of arm A
                n = float(np.linalg.norm(ta))
                if n > 0:
                    ta_u = ta / n
                    cos_a = float(np.dot(ta_u, gdir))
            if bi is not None:
                tb = _node_xyz(g, bi) - pb                # B's arm leaving the junction
                n = float(np.linalg.norm(tb))
                if n > 0:
                    tb_u = tb / n
                    cos_b = float(np.dot(tb_u, gdir))
            out["cos_a"], out["cos_b"] = cos_a, cos_b
            if cos_a is not None and cos_b is not None:
                out["colinear_cos"] = 0.5 * (cos_a + cos_b)
            # Direct arm-to-arm tangent agreement (independent of the gap direction):
            # +1 means the two cables are parallel/continuous regardless of where the
            # gap points. ta_u points back into arm A, tb_u forward out of B, so a
            # straight pass-through has ta_u ~ tb_u -> cos ~ +1.
            if ta_u is not None and tb_u is not None:
                out["tip_tangent_cos"] = float(np.dot(ta_u, tb_u))
    except Exception:
        pass
    # --- Topology: pure graph structure, leak-free, no decision thresholds. ---
    try:
        path = _on_graph_path_um(g, a, b)
        out["graph_path_um"] = path
        out["same_component"] = (path is not None)
    except Exception:
        pass
    try:
        ca = _arm_cable_um(g, a)
        cb = _arm_cable_um(g, b)
        out["cable_a"], out["cable_b"] = ca, cb
        if ca is not None and cb is not None:
            out["cable_min"] = min(ca, cb)
    except Exception:
        pass
    try:
        if out["deg_b"] is not None and out["deg_b"] >= 2:
            out["dist_to_branch_b"] = _dist_to_branch_um(g, b)
    except Exception:
        pass
    return out


# --- Information-maximizing row selection for truncated report tables --------
# The failure-report tables are the reviser's "training set" for choosing accept/
# reject thresholds. When a bucket has more rows than fit in the prompt, the old code
# kept the FIRST ``cap`` (``rows[:cap]``) — an arbitrary, order-dependent slice that
# can hide the tail's distribution (e.g. all the wide-gap examples). Instead we keep a
# REPRESENTATIVE subset chosen from the data: spread across the quantiles of the most
# informative feature (so every regime is visible) plus the boundary extremes (min/
# max, where the accept/reject cutoff actually sits), then summarize what was omitted.
# Deterministic (no RNG), leak-free (operates only on the rows already built from
# train-only data), and no extra LLM call — it just replaces a fixed cap with a
# data-driven choice. ``DEFAULT_TABLE_BUDGET`` keeps output ~the old magnitude.
DEFAULT_TABLE_BUDGET = 60


def _select_representative(items, budget=DEFAULT_TABLE_BUDGET, key=None):
    """Pick a representative subset of ``items`` of size <= ``budget``.

    ``items`` is a list of arbitrary objects; ``key(item) -> float | None`` extracts
    the feature to spread over (e.g. gap_um). If ``key`` is None or yields too few
    finite values, falls back to a head slice (order preserved). Otherwise the return
    is order-preserved but CONTENT-selected to cover the feature's range:
      * always keep the min and max (the decision-boundary extremes);
      * fill the rest by walking evenly across the value-sorted items (quantile
        coverage), so no regime of the feature is invisible.
    Returns ``(kept_items, kept_index_set)`` — the subset in ORIGINAL order, and the
    set of original indices kept (so a caller can summarize the omitted remainder).
    """
    n = len(items)
    if n <= budget:
        return list(items), set(range(n))
    # Extract the spread key; fall back to head slice if not enough signal.
    keyed = []
    if key is not None:
        for i, it in enumerate(items):
            try:
                v = key(it)
                v = float(v) if v is not None else None
            except (TypeError, ValueError):
                v = None
            if v is not None and v == v:  # finite
                keyed.append((v, i))
    if len(keyed) < max(3, budget // 2):
        # Not enough finite feature values to spread over — keep a head slice.
        return list(items[:budget]), set(range(budget))
    keyed.sort()                      # by feature value
    order = [i for _, i in keyed]     # original indices, feature-sorted
    chosen = {order[0], order[-1]}    # boundary extremes
    remaining = budget - len(chosen)
    if remaining > 0 and len(order) > 2:
        # Evenly sample the interior (quantile coverage) without RNG.
        interior = order[1:-1]
        if remaining >= len(interior):
            chosen.update(interior)
        else:
            step = len(interior) / float(remaining)
            for k in range(remaining):
                chosen.add(interior[int(k * step)])
    # Items with a non-finite key (dropped from `keyed`) are lowest priority; only
    # backfill them if we're still under budget after the spread.
    if len(chosen) < budget:
        for i in range(n):
            if i not in chosen:
                chosen.add(i)
                if len(chosen) >= budget:
                    break
    kept_idx = set(list(chosen)[:budget])
    kept = [items[i] for i in range(n) if i in kept_idx]
    return kept, kept_idx


def _omitted_summary(items, kept_idx, feature_keys):
    """One-line aggregate of the rows NOT shown, so the tail's distribution is visible.

    ``feature_keys`` maps a short name -> ``fn(item) -> float | None``. For each, we
    report the omitted rows' count and the min/median/max of that feature, so the
    reviser knows the shape of what was truncated (e.g. "the 1076 omitted misses span
    gap 5.9–41 µm, median 12") instead of a blind "…and 1076 more." Returns "" when
    nothing was omitted.
    """
    omitted = [it for i, it in enumerate(items) if i not in kept_idx]
    if not omitted:
        return ""
    parts = []
    for name, fn in feature_keys.items():
        vals = []
        for it in omitted:
            try:
                v = fn(it)
                v = float(v) if v is not None else None
            except (TypeError, ValueError):
                v = None
            if v is not None and v == v:
                vals.append(v)
        if vals:
            vals.sort()
            med = vals[len(vals) // 2]
            parts.append(f"{name} {vals[0]:.3g}–{vals[-1]:.3g} (median {med:.3g})")
    span = ("; ".join(parts)) if parts else "no finite features"
    return (f"\n…and {len(omitted)} more not shown (representative rows kept above). "
            f"Omitted rows span: {span}.")


# --- Generic fitness-attribution over a candidate's feature space ------------
# Attribute a per-edit verdict (correct / false / unscored) to the feature space of
# the edits, with NOTHING domain-specific hardcoded:
#   * DIMENSIONS are discovered by introspecting each edit's feature dict (so any
#     numeric feature the harness exposes — present or future — is eligible);
#   * BUCKET EDGES are data-driven (quantiles for continuous features, value sets for
#     discrete ones), not hand-picked thresholds;
#   * the verdict comes from a pluggable ``classify_fn`` (so SplitSite/merge and
#     MergeSite/split, or any other action, reuse the same machinery);
#   * only the MOST DISCRIMINATIVE features are reported (ranked by how well the
#     feature separates correct from false), so the reviser sees the few axes that
#     actually explain where its repairs vs. mistakes come from — not a wall of axes.
_VERDICTS = ("correct", "false", "unscored")


def _bucketize_continuous(values, n_bins=3):
    """Quantile bin EDGES for a list of finite floats -> (n_bins-1) interior edges.

    Data-driven (tertiles by default) so the buckets adapt to THIS generation's
    distribution instead of a hand-picked threshold. Returns sorted unique interior
    edges; an empty list (constant feature) means 'do not bin'.
    """
    xs = sorted(v for v in values if v is not None and v == v and abs(v) != float("inf"))
    if len(xs) < n_bins:
        return []
    edges = []
    for k in range(1, n_bins):
        q = xs[min(len(xs) - 1, int(k * len(xs) / n_bins))]
        if not edges or q > edges[-1]:
            edges.append(q)
    return edges


def _bucket_label(v, edges, name):
    """Human-readable bucket for value ``v`` given continuous ``edges`` (may be [])."""
    if v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
        return f"{name} ?"
    if not edges:                       # discrete or constant -> bucket by value
        return f"{name}={v:g}" if isinstance(v, (int, float)) else f"{name}={v}"
    lo = "-inf"
    for e in edges:
        if v < e:
            return f"{name} [{lo},{e:g})"
        lo = f"{e:g}"
    return f"{name} [{lo},inf)"


def _auc_corr_vs_false(rows, feat):
    """Discriminative power of ``feat`` for correct-vs-false, as |AUC - 0.5| * 2.

    rows: list of (verdict, feature_dict). Uses the existing rank-AUC (_auc). 0 = no
    separation, 1 = perfectly separates correct from false. unscored rows are ignored
    (no ground-truth verdict). None if a class is empty or the feature is constant.
    """
    pos = [r[1].get(feat) for r in rows if r[0] == "correct"]
    neg = [r[1].get(feat) for r in rows if r[0] == "false"]
    pos = [v for v in pos if isinstance(v, (int, float)) and v == v and abs(v) != float("inf")]
    neg = [v for v in neg if isinstance(v, (int, float)) and v == v and abs(v) != float("inf")]
    a = _auc(pos, neg)
    if a is None:
        return None
    return abs(a - 0.5) * 2.0


def attribute_fitness(rows, top_k=3, n_bins=3):
    """Generic: attribute verdicts across the most discriminative feature axes.

    Parameters
    ----------
    rows : list[(verdict, feature_dict)]
        One per edit; ``verdict in {"correct","false","unscored"}``; ``feature_dict``
        maps feature name -> value (numbers or small discretes). Built domain-side via
        ``edit_feature_rows`` so this function stays domain-agnostic.
    top_k : int
        Show at most this many feature axes, chosen by correct-vs-false separation
        (falls back to coverage/variance when no correct/false split exists).
    n_bins : int
        Quantile bins per continuous feature.

    Returns ``{feature_name: {bucket_label: {"correct","false","unscored","n"}}}`` for
    the top-k features, or None if there is nothing to attribute.
    """
    from collections import defaultdict
    rows = [r for r in rows if r and isinstance(r[1], dict)]
    if not rows:
        return None
    feats = sorted({k for _, fd in rows for k in fd})
    if not feats:
        return None

    # Rank features: prefer discriminative power (correct vs false); when that is
    # undefined (e.g. zero false this gen), fall back to how many edits the feature
    # covers so we still surface SOMETHING informative.
    def _rank(feat):
        disc = _auc_corr_vs_false(rows, feat)
        cover = sum(1 for _, fd in rows if fd.get(feat) is not None)
        return (disc if disc is not None else -1.0, cover)
    chosen = sorted(feats, key=_rank, reverse=True)[:max(1, top_k)]

    out = {}
    for feat in chosen:
        vals = [fd.get(feat) for _, fd in rows]
        numeric = [v for v in vals if isinstance(v, (int, float))]
        edges = _bucketize_continuous(numeric, n_bins) if len(numeric) == len(
            [v for v in vals if v is not None]) and numeric else []
        cells = defaultdict(lambda: {v: 0 for v in _VERDICTS} | {"n": 0})
        for verdict, fd in rows:
            label = _bucket_label(fd.get(feat), edges, feat)
            cells[label][verdict] += 1
            cells[label]["n"] += 1
        out[feat] = dict(cells)
    return out or None


def edit_feature_rows(edits, classify_fn, feature_fn):
    """Build ``[(verdict, feature_dict)]`` for ``attribute_fitness`` — domain glue.

    ``classify_fn(edit) -> "correct"|"false"|"unscored"|None`` (None drops the edit);
    ``feature_fn(edit) -> dict`` of that edit's features. Both are supplied by the
    caller, so the generic attributor knows nothing about merges/splits/geometry.
    """
    rows = []
    for e in (edits or []):
        v = classify_fn(e)
        if v is None:
            continue
        rows.append((v, feature_fn(e) or {}))
    return rows


def split_repair_by_bucket(train_run, label_gt_map, fragments_graph, top_k=3):
    """SplitSite/merge ADAPTER over the generic ``attribute_fitness``.

    Wires the split-repair verdict (train-map dominant-neuron rule, same as the gate)
    and the SplitSite feature set (gap + EVERY key ``_split_site_geom`` exposes:
    colinear_cos, deg_b, rad_ratio, tip_tangent_cos, same_component, graph_path_um,
    cable_min, dist_to_branch_b, …) into the generic attributor. Returns
    ``{feature: {bucket: counts}}`` for the top-k most discriminative features, or
    None. Leak-free (train map only). Adding a new SplitSite feature automatically
    makes it eligible here with no change to this function.
    """
    if not label_gt_map or not train_run.edits:
        return None

    def _pk(a, b):
        a, b = str(a), str(b)
        return (a, b) if a <= b else (b, a)
    site_by_pair = {}
    for s in (train_run.split_sites or []):
        site_by_pair[_pk(s.label_a, s.label_b)] = s

    def _dominant(lbl):
        c = label_gt_map.get(str(lbl))
        return max(c, key=c.get) if c else None

    def _labels(e):
        if isinstance(e, dict):
            if e.get("kind", "merge_labels") != "merge_labels":
                return None
            return e.get("label_a"), e.get("label_b")
        if isinstance(e, (tuple, list)) and len(e) >= 2:
            return e[0], e[1]
        return None

    def classify_fn(e):
        lp = _labels(e)
        if lp is None:
            return None
        da, db = _dominant(lp[0]), _dominant(lp[1])
        if da is None or db is None:
            return "unscored"
        return "correct" if da == db else "false"

    def feature_fn(e):
        lp = _labels(e)
        if lp is None:
            return {}
        s = site_by_pair.get(_pk(lp[0], lp[1]))
        if s is None:
            return {}
        feats = {"gap_um": getattr(s, "gap_um", None)}
        if fragments_graph is not None:
            geom = _split_site_geom(fragments_graph, s)
            # Forward EVERY feature _split_site_geom exposes (numeric + bool/discrete);
            # attribute_fitness ranks them by discriminative AUC and drops Nones and
            # constants on its own. This keeps the function's promise literally true —
            # adding a feature to _split_site_geom makes it eligible here with NO edit
            # to this list (the old hardcoded 4-tuple silently dropped the rest).
            for k, v in geom.items():
                feats[k] = v
        return feats

    rows = edit_feature_rows(train_run.edits, classify_fn, feature_fn)
    return attribute_fitness(rows, top_k=top_k)


def _load_policy(heuristics_path: str):
    """Import artifacts/heuristics.py fresh from disk; return (propose_edits, module).

    The module is returned too so the caller can read an optional ``ENUM_PARAMS``
    dict (the policy's evolvable enumeration priors) off it.
    """
    spec = importlib.util.spec_from_file_location(
        f"_evolved_heuristics_{abs(hash(heuristics_path))}", heuristics_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "propose_edits"):
        raise AttributeError(
            f"{heuristics_path} must define propose_edits(sites, ctx)."
        )
    return module.propose_edits, module


# Candidate sites depend only on the fragment graph (+ max_gap_um), which is fixed
# for a whole run, but run_candidate is called (2 * generations + 1) times. The
# enumeration is a whole-brain geometric scan, so memoize it keyed by the graph's
# object identity and the gap. Keyed by id() because SkeletonGraph is unhashable
# and the loop reuses one graph instance throughout; a small dict avoids leaks.
_SITES_CACHE: dict = {}


def _enumerate_sites_cached(fragments_graph, params: dict, enumerate_merges: bool = True):
    """Return (split_sites, merge_sites, split_stats), computing once per (graph, params).

    ``split_stats`` is the GT-free enumeration-ceiling dict from
    ``candidate_split_sites(..., return_stats=True)`` — how many label pairs were
    found vs. dropped by the ``split_max_sites`` cap, and the gap range truncated.
    It rides the same cache so it is paid once per param tuple, like the sites.

    ``params`` is a resolved (validated + clamped) ENUM_PARAMS dict. The cache key
    includes EVERY enumeration param, not just the gap — otherwise a policy that
    changed an enumeration prior (e.g. min_arm_cable_um) would silently reuse sites
    enumerated under the old prior. Same params -> the whole-brain scan is paid once
    and reused across all (2*generations + 1) run_candidate calls; a new param tuple
    pays a fresh scan (and that is the intended cost of evolving the prior).

    ``enumerate_merges`` False (SPLIT-ERROR-ONLY mode) SKIPS the whole-brain merge
    scan entirely and returns ``merge_sites == []`` — the policy is then handed no
    MergeSite at all (so it cannot reason over, or pay cloud reads for, merge
    repairs this run). It is part of the cache key so a split-only run and a full
    run never share a cached entry.
    """
    key = (id(fragments_graph),
           params["max_gap_um"], params["split_max_sites"], params["tip_to_shaft"],
           params.get("split_alt_per_pair", 1), params.get("split_per_tip_k", 4),
           params["min_arm_cable_um"], params["seed_depth_um"],
           params["merge_max_sites"], params["max_per_label"],
           bool(enumerate_merges))
    cached = _SITES_CACHE.get(key)
    if cached is None:
        split_sites, split_stats = ds.candidate_split_sites(
            fragments_graph,
            max_gap_um=params["max_gap_um"],
            max_sites=params["split_max_sites"],
            tip_to_shaft=params["tip_to_shaft"],
            alt_per_pair=params.get("split_alt_per_pair", 1),
            per_tip_k=params.get("split_per_tip_k", 4),
            return_stats=True,
        )
        if enumerate_merges:
            merge_sites = ds.candidate_merge_sites(
                fragments_graph,
                min_arm_cable_um=params["min_arm_cable_um"],
                seed_depth_um=params["seed_depth_um"],
                max_sites=params["merge_max_sites"],
                max_per_label=params["max_per_label"],
            )
        else:
            merge_sites = []  # split-error-only: skip the whole-brain merge scan
        cached = (split_sites, merge_sites, split_stats)
        _SITES_CACHE[key] = cached
    return cached


@dataclass
class CandidateRun:
    """Everything produced by evaluating one candidate on one GT subset."""

    split: str                 # "train" or "heldout"
    n_sites: int               # candidate split sites considered
    n_edits: int               # edits the policy proposed
    score: scoring.ScoreResult
    edits: list                # the (a, b) pairs proposed (for the failure report)
    merge_sites: list = field(default_factory=list)  # enumerated MergeSites (for the
                               # failure report's per-site feature table); GT-free.
    split_sites: list = field(default_factory=list)  # enumerated SplitSites (for the
                               # report's SplitSite feature audit); GT-free.
    image_reads: list = field(default_factory=list)  # image-evidence calls the policy
                               # made this run (passively recorded; zero extra reads).
    enum_params: dict = field(default_factory=dict)  # resolved (validated+clamped)
                               # ENUM_PARAMS in effect this run (for the rail-
                               # sensitivity section); empty if not recorded.
    raw_enum_params: dict = field(default_factory=dict)  # the policy's RAW ENUM_PARAMS
                               # request before clamping (to flag clamped knobs).
    n_split_label_dropped: int = 0  # split_label edits discarded by SPLIT-ERROR-ONLY
                               # mode (should be 0 — non-zero means the policy emitted
                               # merge repairs that were silently dropped this run).
    split_enum_stats: dict = field(default_factory=dict)  # GT-free enumeration-ceiling
                               # stats for the SplitSite stream (pairs found vs. dropped
                               # by the split_max_sites cap, truncated gap range); drives
                               # the report's enumeration recall-ceiling section.
    # (1) PER-STEP TIMING: wall-clock of the policy's own propose_edits() call, kept
    # SEPARATE from scoring (score.seconds). Without this, a policy that writes an
    # accidentally O(N^2) feature (e.g. a per-site full-graph scan) is invisible — its
    # cost hides inside the scoring heartbeat. Recorded per run and surfaced in the
    # ledger so a runaway policy is attributable to the policy, not the scorer.
    policy_seconds: float = 0.0
    # (2) BUDGET: True iff propose_edits exceeded ``policy_time_budget`` and was
    # aborted. A timed-out policy yields NO edits (the candidate is treated as a failed
    # generation upstream, like an import/lint failure), so this run is a no-op repair.
    policy_timed_out: bool = False

    def to_json(self) -> dict:
        return {
            "split": self.split,
            "n_sites": self.n_sites,
            "n_edits": self.n_edits,
            "primary": self.score.primary,
            "metrics": self.score.metrics,
            "seconds": self.score.seconds,
            "policy_seconds": self.policy_seconds,
            "policy_timed_out": self.policy_timed_out,
        }


class PolicyTimeout(Exception):
    """Raised when ``propose_edits`` exceeds its wall-clock budget (option 2).

    A policy is arbitrary evolved code; a quadratic feature over a whole-brain
    candidate stream can run for an hour (observed: a per-site full-graph distance
    scan). The budget turns that from a silent multi-hour stall into a fast, bounded
    REJECT — the same failure class as a non-importing or lint-failing revision.
    """


class _policy_time_budget:
    """Context manager that aborts the body if it runs longer than ``seconds``.

    Uses SIGALRM on the main thread (the common case: run_candidate is called
    synchronously from the loop), which can interrupt even a tight C-level numpy loop
    at the next Python bytecode check. When not on the main thread (SIGALRM is
    unavailable there), it degrades to a NO-OP guard — the wall-clock is still recorded
    by the caller, so a slow policy is at least VISIBLE even if not interrupted. A
    non-positive or None budget disables the guard entirely.
    """

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self._armed = False
        self._old_handler = None

    def __enter__(self):
        if not self.seconds or self.seconds <= 0:
            return self
        # signal.alarm only works on the main thread; guard so a worker-thread caller
        # (or a platform without SIGALRM) degrades gracefully instead of raising.
        if threading.current_thread() is not threading.main_thread():
            return self
        if not hasattr(signal, "SIGALRM"):
            return self

        def _fire(signum, frame):
            raise PolicyTimeout(
                f"propose_edits exceeded {self.seconds:g}s budget")

        self._old_handler = signal.signal(signal.SIGALRM, _fire)
        # setitimer takes a float; alarm() would truncate a sub-second budget to 0.
        signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
        self._armed = True
        return self

    def __exit__(self, *exc):
        if self._armed:
            signal.setitimer(signal.ITIMER_REAL, 0)  # disarm
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
            self._armed = False
        return False  # never suppress (PolicyTimeout propagates to run_candidate)


# (3) SHARED SPATIAL INDEX — the systematic fix for the ROOT CAUSE of the O(N^2)
# blowup. gen12's policy hand-wrote a foreign-label density feature as a full O(N)
# numpy scan of every node coordinate, PER candidate site (the docstring even claimed
# it was bounded). It did that because the policy had no efficient radius query — the
# fragment graph exposes node_xyz but no spatial index. We build ONE KD-tree per
# fragment graph (cached by graph identity, like the site enumeration) and expose two
# helpers on ctx so the natural, least-effort way to write a spatial feature is also
# the fast one:
#   * ctx["nodes_within"](xyz, radius) -> np.ndarray of node ids within radius (µm)
#   * ctx["foreign_labels_near"](xyz, radius, exclude=(la, lb)) -> set of distinct
#     segment ids near xyz other than the excluded pair (the exact gen12 feature,
#     but O(log N + hits) instead of O(N)).
_KDTREE_CACHE: dict = {}


def _graph_kdtree(fragments_graph):
    """Return (KDTree, node_id_array) for a fragment graph, built once and cached.

    Cached by ``id(fragments_graph)`` — the loop reuses one graph instance for the
    whole run, so the tree is paid once (like ``_enumerate_sites_cached``). Returns
    ``(None, None)`` if the graph has no usable ``node_xyz`` so callers no-op safely.
    """
    key = id(fragments_graph)
    cached = _KDTREE_CACHE.get(key)
    if cached is not None:
        return cached
    xyz = getattr(fragments_graph, "node_xyz", None)
    if xyz is None:
        _KDTREE_CACHE[key] = (None, None)
        return None, None
    try:
        node_ids = np.asarray(list(fragments_graph.nodes))
        coords = np.asarray(xyz, dtype=float)[node_ids]
        tree = KDTree(coords)
    except Exception:
        _KDTREE_CACHE[key] = (None, None)
        return None, None
    _KDTREE_CACHE[key] = (tree, node_ids)
    return tree, node_ids


def _nodes_within(fragments_graph, xyz, radius_um):
    """Node ids within ``radius_um`` (µm) of ``xyz`` — KD-tree query, O(log N + hits).

    Returns an ``np.ndarray`` of fragment-graph node ids (empty on any failure or if
    the graph has no coordinates). The efficient replacement for a full-array
    ``np.linalg.norm(all_coords - xyz)`` scan.
    """
    tree, node_ids = _graph_kdtree(fragments_graph)
    if tree is None:
        return np.empty(0, dtype=int)
    try:
        pt = np.asarray(xyz, dtype=float)
        idx = tree.query_ball_point(pt, r=float(radius_um))
        return node_ids[np.asarray(idx, dtype=int)]
    except Exception:
        return np.empty(0, dtype=int)


def _foreign_labels_near(fragments_graph, xyz, radius_um, exclude=()):
    """Distinct segment ids within ``radius_um`` of ``xyz``, excluding ``exclude``.

    The crossing/tangle density cue (how many UNRELATED fragments crowd a gap),
    computed via the KD-tree instead of a per-site whole-graph scan. ``exclude`` is
    the pair whose own labels should not count (e.g. a SplitSite's ``label_a`` /
    ``label_b``). Returns a set of segment-id strings; empty on failure.
    """
    near = _nodes_within(fragments_graph, xyz, radius_um)
    if len(near) == 0:
        return set()
    own = {str(x) for x in exclude}
    out: set = set()
    for n in near:
        try:
            seg = str(fragments_graph.node_segment_id(int(n)))
        except Exception:
            continue
        if seg not in own:
            out.add(seg)
    return out


def run_candidate(
    prepared,
    fragments_graph,
    gt_swc_names: list[str],
    split_name: str,
    heuristics_path: str,
    max_gap_um: float = 15.0,
    max_class_size=None,
    image_reader=None,
    verbose: bool = False,
    splits_only: bool = False,
    policy_time_budget: float | None = None,
) -> CandidateRun:
    """Execute the evolved policy and score its edits on the given GT subset.

    Scoring uses the incremental scorer (seconds), so this is cheap enough to
    call twice per generation. The candidate-site geometry still comes from the
    fast cached fragment graph.

    Parameters
    ----------
    prepared : incremental_scoring.PreparedBrain
        The once-loaded, candidate-invariant brain state.
    fragments_graph : SkeletonGraph
        Cached fragment graph (from dataset.load_cached_graphs), used ONLY to
        enumerate candidate split sites for the policy to reason over.
    gt_swc_names : list of str
        The GT skeletons to score on (train or held-out).
    split_name : str
        "train" or "heldout" — only used for labelling.
    heuristics_path : str
        Path to the evolved artifacts/heuristics.py.
    max_gap_um : float
        Candidate-site enumeration radius (the policy sees everything below this
        and decides internally; keep generous so the policy can choose).
    splits_only : bool
        SPLIT-ERROR-ONLY fast mode. When True, NO MergeSite is enumerated (the
        whole-brain merge scan is skipped), so the policy is handed SplitSites only
        and cannot reason over — or pay cloud reads for — merge repairs this run. As
        a defensive backstop, any ``split_label`` a policy still hardcodes is dropped
        BEFORE scoring (recorded in ``n_split_label_dropped``). Only ``merge_labels``
        (split-error repairs) are scored, which skips the expensive coordinate-aware
        split path (multi-source Dijkstra + fragment-graph rebuild) in the scorer. It
        does NOT change the gate: the gate's split-repair metric already classifies
        only ``merge_labels`` edits (``classify_merge_edits`` ignores ``split_label``),
        so a split-error-only run is measured by exactly the same number.
    """
    # Imported here to avoid a hard dependency for callers that only need the
    # failure-report helper.
    from proofreader_evolve.harness import incremental_scoring as inc

    from proofreader_evolve.harness.edit_handler import normalize_edits

    propose_edits, policy_module = _load_policy(heuristics_path)

    # Evolvable enumeration priors: the policy MAY define a module-level ENUM_PARAMS
    # dict to widen/narrow the candidate stream (what counts as a candidate). It is
    # validated + clamped to a safe schema; missing keys fall back to the framework
    # defaults, so a policy that defines nothing behaves exactly as before. The
    # max_gap_um function arg is the default prior; ENUM_PARAMS["max_gap_um"], when
    # given, overrides it.
    raw_enum = getattr(policy_module, "ENUM_PARAMS", None)
    # The policy's RAW request (before defaulting/clamping), kept verbatim so the
    # report can flag which knobs the safety rail clamped. Only dict entries the
    # policy actually set are recorded (max_gap_um's framework default is not a
    # "request").
    policy_raw_enum = {k: v for k, v in raw_enum.items()} if isinstance(raw_enum, dict) else {}
    if isinstance(raw_enum, dict):
        raw_enum = {**raw_enum}
        raw_enum.setdefault("max_gap_um", max_gap_um)
    else:
        raw_enum = {"max_gap_um": max_gap_um}
    enum_params = ds.resolve_enum_params(raw_enum)

    # The policy reasons over a UNIFIED candidate stream of two site kinds:
    #   - SplitSite (kind="split"): two nearby fragments with DIFFERENT labels;
    #     valid action = merge_labels (repairs a split error).
    #   - MergeSite (kind="merge"): ONE label fused across two neurites at a branch;
    #     valid action = split_label (repairs a merge error).
    # Both are GT-free (fragment geometry only), so the same stream is used on
    # train and held-out. The policy dispatches on ``site.kind``.
    #
    # Candidate-INVARIANT: the sites depend only on ``fragments_graph`` (and
    # max_gap_um), which never changes within a run, while run_candidate is called
    # (2 * generations + 1) times. Enumerating is a whole-brain geometric scan, so
    # we cache per (graph identity, max_gap_um) and reuse across every call. This is
    # the difference between paying the scan once vs. once per candidate.
    # SPLIT-ERROR-ONLY: skip the whole-brain merge scan so the policy is handed NO
    # MergeSite — it then cannot reason over, or pay cloud reads for, merge repairs
    # this run, and the failure report's merge sections go empty on their own.
    split_sites, merge_sites, split_enum_stats = _enumerate_sites_cached(
        fragments_graph, enum_params, enumerate_merges=not splits_only)
    sites = list(split_sites) + list(merge_sites)

    # Passively record the image-evidence calls the policy makes (gap_connectivity /
    # merge_cut_evidence), so the failure report can turn the reads the policy ALREADY
    # paid for into a labelled learning signal. Adds NO extra cloud reads. Only wraps
    # when an image reader is actually present.
    if image_reader is not None:
        from proofreader_evolve.harness.image_features import RecordingImageReader
        rec_reader = RecordingImageReader(image_reader)
    else:
        rec_reader = None
    ctx = {
        "max_gap_um": enum_params["max_gap_um"],
        # The resolved (validated + clamped) enumeration priors actually in effect
        # this run — so the policy / failure report can see what candidate stream it
        # was handed (e.g. distinguish "no site here" from "my widened gap took
        # effect"). The values may differ from a policy's raw ENUM_PARAMS if a knob
        # was clamped to its safety rail.
        "enum_params": dict(enum_params),
        "fragments_graph": fragments_graph,
        # Candidate-stream composition, so the policy can tell how many of each
        # kind it was handed without re-scanning (sites carry a ``kind`` tag too).
        "n_split_sites": len(split_sites),
        "n_merge_sites": len(merge_sites),
        # (1) Cheap, in-memory signal the policy was missing: per-node neurite
        # radius (float16 array indexed by node id). Lets the policy reason about
        # fragment thickness — e.g. refuse to fuse two thick (likely-real) neurites
        # — with NO cloud read. Present on the agentic SkeletonGraph already.
        "node_radius": getattr(fragments_graph, "node_radius", None),
        # (1b) Cheap, GT-free, NO-cloud-read geometry+topology for a SplitSite: call
        # ctx["split_geom"](site) -> {colinear_cos, cos_a, cos_b, tip_tangent_cos,
        # deg_a, deg_b, rad_a, rad_b, rad_ratio,  # geometry/morphology
        #   same_component, graph_path_um, cable_a, cable_b, cable_min,
        #   dist_to_branch_b}.                     # pure graph TOPOLOGY (brain-indep.)
        # These are the deployable features (straightness, per-arm asymmetry, endpoint
        # degree, cable caliber; plus loop-closure / fragment-maturity / shaft-position
        # topology) the policy can threshold on directly, computed from the fragment
        # graph. same_component=True means merging would CLOSE A LOOP (a strong
        # non-merge prior for tree-like neurons). NO thresholds are baked in — the
        # harness exposes raw values and the policy decides which to use and where.
        "split_geom": (lambda site: _split_site_geom(fragments_graph, site)),
        # (3) EFFICIENT SPATIAL QUERIES (KD-tree backed, O(log N + hits)). Use these
        # for any "what fragments are near here" feature instead of scanning node_xyz
        # yourself — a hand-written full scan is O(N) PER site and made held-out scoring
        # run for an hour once. ``nodes_within(xyz, radius)`` -> node-id array within
        # ``radius`` µm of a point; ``foreign_labels_near(xyz, radius, exclude=(a,b))``
        # -> set of distinct segment ids near ``xyz`` other than the excluded pair (the
        # crossing/tangle density cue). Both no-op to empty if the graph has no coords.
        "nodes_within": (lambda xyz, radius: _nodes_within(fragments_graph, xyz, radius)),
        "foreign_labels_near": (
            lambda xyz, radius, exclude=(): _foreign_labels_near(
                fragments_graph, xyz, radius, exclude)),
        # (2) Optional, lazy, cached raw-image patch reader (the fluorescence
        # signal at the gap). None unless an image reader was provided — the policy
        # MUST handle ctx["read_image_patch"] is None. When present it is a
        # LazyImagePatchReader; call .read_patch(node_id[, shape]),
        # .gap_connectivity(node_a, node_b) (cheap split evidence: signal at each
        # endpoint — does NOT test the gap interior), .gap_bridge_evidence(node_a,
        # node_b) (split evidence: is there a CONTINUOUS bright bridge across the gap?
        # high bridge_ratio ⇒ safe to merge), or .merge_cut_evidence(seed_a_node,
        # seed_b_node) (merge evidence: an intensity valley between two fused arms?)
        # — each is a cloud read, so gate it behind cheap geometric filters.
        "read_image_patch": rec_reader,
        # (3) Receptive-field knob for the image reads above. Every reader method
        # takes an optional ``shape=(z,y,x)`` (voxels) — the patch read per sample.
        # Bigger = more spatial context but a larger cloud fetch; each axis is clamped
        # to image_features._MAX_PATCH_DIM (64). This is the RECOMMENDED default the
        # policy may pass, e.g. ``reader.gap_bridge_evidence(a, b, shape=ctx["image_patch_shape"])``
        # — tune it per tier (a tight window for a clean micro-gap, a wider one to
        # confirm a long faint bridge). None of the readers require it (they default
        # to the reader's own shape); it is exposed so the receptive field is an
        # explicit, documented dial rather than a buried default.
        "image_patch_shape": (16, 16, 16),
    }

    # The policy may return legacy (label_a, label_b) tuples OR typed edit dicts
    # ({"kind": "merge_labels"|"split_label"|...}). normalize_edits promotes both
    # to the typed form, so the loop accepts either without the old, lossy
    # tuple(map(str, e)) coercion (which silently corrupted dict edits into a
    # 4-tuple of their keys).
    #
    # (1) TIME the policy call separately from scoring, and (2) BUDGET it: a policy is
    # arbitrary evolved code that can run an accidental O(N^2) feature over the whole
    # candidate stream. If it exceeds ``policy_time_budget`` it is aborted and treated
    # as producing NO edits (a no-op candidate), so the generation is rejected like any
    # other failed revision rather than stalling the run for an hour.
    policy_timed_out = False
    _t_policy = time.monotonic()
    try:
        with _policy_time_budget(policy_time_budget):
            raw_edits = propose_edits(sites, ctx)
    except PolicyTimeout:
        policy_timed_out = True
        raw_edits = []
        if verbose:
            print(f"[{split_name}] propose_edits exceeded "
                  f"{policy_time_budget:g}s budget — aborted, treating as no edits")
    policy_seconds = time.monotonic() - _t_policy
    edits = normalize_edits(raw_edits)

    # SPLIT-ERROR-ONLY: defensive guard. With no MergeSite enumerated above the policy
    # has nothing to build a split_label from, but a policy could still hardcode one,
    # so drop any split_label before it reaches the (expensive) scorer. A non-zero
    # count here means the policy emitted merge repairs that were SILENTLY discarded —
    # surface it on the run so that is visible rather than mysterious.
    n_split_label_dropped = 0
    if splits_only and edits:
        kept = [e for e in edits if e.get("kind") != "split_label"]
        n_split_label_dropped = len(edits) - len(kept)
        edits = kept

    # Always route through the typed path so split_label edits actually take
    # effect. A pure-merge edit list reproduces the legacy label-pair result
    # exactly (EditHandler's merge union-find == LabelHandler's equivalence
    # classes), so split-only policies are unaffected.
    result = inc.score_incremental(
        prepared,
        edits=edits,
        gt_swc_names=gt_swc_names,
        max_class_size=max_class_size,
        verbose=verbose,
    )
    return CandidateRun(
        split=split_name,
        n_sites=len(sites),
        n_edits=len(edits),
        score=result,
        edits=edits,
        merge_sites=list(merge_sites),
        split_sites=list(split_sites),
        image_reads=list(rec_reader.records) if rec_reader is not None else [],
        enum_params=dict(enum_params),
        raw_enum_params=dict(policy_raw_enum),
        n_split_label_dropped=n_split_label_dropped,
        split_enum_stats=dict(split_enum_stats),
        policy_seconds=policy_seconds,
        policy_timed_out=policy_timed_out,
    )


def _auc(pos: list, neg: list) -> float | None:
    """Mann–Whitney AUC: P(a random REAL bridge_ratio > a random FALSE one).

    Rank-based (ties counted as 0.5), so it needs no threshold and is robust to the
    bridge_ratio scale. None when either class is empty. 0.5 == no separation; >0.5
    means REAL splits tend to have the HIGHER bridge_ratio (the expected direction:
    a real continuation stays bright across the gap).
    """
    pos = [v for v in pos if v == v]  # drop NaN
    neg = [v for v in neg if v == v]
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


def image_warmstart_probe(
    split_sites: list,
    label_gt_map: dict,
    image_reader,
    train_gt_names,
    max_sites: int = 24,
    overlap_only: bool = True,
) -> dict | None:
    """HARNESS-side, one-time DE-RISK probe of image separability for split repair.

    Breaks the image-signal cold start: the per-generation failure report can only
    REPLAY ``gap_bridge_evidence`` reads the POLICY already made, but a policy that
    has not yet learned to read image cannot produce any — and a revision that only
    "starts reading" without changing its merge decisions ties the split-repair gate
    and is reverted, discarding its reads. So the harness itself reads a small,
    budget-capped, balanced sample of REAL vs FALSE SplitSites ONCE and reports
    whether ``bridge_ratio`` separates them. This is a MEASUREMENT (à la
    ``incremental_scoring.probe_split_oracle``), not a policy and not a gate input:
    it never selects edits and never reaches held-out.

    Leak-free: REAL/FALSE is decided ONLY by the train ``label_gt_map`` dominant-
    neuron rule (same as the SplitSite audit); a label with no train entry is
    dropped. Budget: at most ``max_sites`` sites get a (cloud) ``gap_bridge_evidence``
    read; the caller caches the result for the whole run and skips this entirely when
    no image reader is present.

    ``overlap_only`` focuses the sample on the ``gap_um`` band where REAL and FALSE
    OVERLAP — the geometrically ambiguous sites where image has the most decision
    value (away from that band, gap alone already separates them).

    Returns ``{"section": <markdown lines list>, "auc": float|None, "n_real": int,
    "n_false": int}`` or None when there is nothing to probe (no image reader / no
    label map / no train-classifiable sites).
    """
    if image_reader is None or not label_gt_map or not split_sites:
        return None

    def _dominant(lbl):
        counts = label_gt_map.get(str(lbl))
        return max(counts, key=counts.get) if counts else None

    # Classify every train-visible SplitSite REAL (same dominant train neuron) vs
    # FALSE (different), carrying its gap. Drop labels with no train entry (leak-free).
    real, false = [], []  # each: (gap_um, node_a, node_b)
    for s in split_sites:
        da, db = _dominant(getattr(s, "label_a", "")), _dominant(getattr(s, "label_b", ""))
        if da is None or db is None:
            continue
        gap = getattr(s, "gap_um", None)
        na, nb = getattr(s, "node_a", None), getattr(s, "node_b", None)
        if na is None or nb is None:
            continue
        try:
            gap = float(gap)
        except (TypeError, ValueError):
            continue
        (real if da == db else false).append((gap, na, nb))
    if not real or not false:
        return None

    # Focus on the gap band where the two classes OVERLAP — the ambiguous region
    # where image is worth reading. The overlap is [max(min), min(max)] of the two
    # gap ranges; if they don't overlap, fall back to the full set (image still
    # informative, just less decisive).
    def _band(sites):
        gaps = [g for g, _, _ in sites]
        return min(gaps), max(gaps)
    if overlap_only:
        rlo, rhi = _band(real); flo, fhi = _band(false)
        lo, hi = max(rlo, flo), min(rhi, fhi)
        if lo <= hi:
            real_b = [t for t in real if lo <= t[0] <= hi] or real
            false_b = [t for t in false if lo <= t[0] <= hi] or false
        else:
            real_b, false_b = real, false
    else:
        real_b, false_b = real, false

    # Balanced budget: split max_sites evenly, take the sites CLOSEST to the shared
    # median gap first (most ambiguous), deterministically (no RNG — sort by |gap-med|).
    allg = sorted(t[0] for t in real_b + false_b)
    med = allg[len(allg) // 2]
    half = max(1, max_sites // 2)
    real_pick = sorted(real_b, key=lambda t: abs(t[0] - med))[:half]
    false_pick = sorted(false_b, key=lambda t: abs(t[0] - med))[:half]

    def _probe(sites):
        # Collect the FULL feature vector per probed site (not just bridge_ratio) so
        # the confidence model can be calibrated multi-feature and each feature's
        # separability reported. Zero extra reads vs. the old single-scalar probe —
        # gap_bridge_evidence already returns every feature in one call.
        rows, vals, feats = [], [], []
        for gap, na, nb in sites:
            res = image_reader.gap_bridge_evidence(na, nb) or {}
            br = res.get("bridge_ratio", float("nan"))
            try:
                brf = float(br)
            except (TypeError, ValueError):
                brf = float("nan")
            vals.append(brf)
            feats.append(imgconf.extract_features(res))
            br_s = "NaN" if brf != brf else f"{brf:.2f}"
            bm = res.get("bridge_min"); em = res.get("endpoint_mean")
            def _n(v, nd=2):
                try:
                    fv = float(v); return "NaN" if fv != fv else f"{fv:.{nd}f}"
                except (TypeError, ValueError):
                    return "—"
            rows.append(f"| {gap:.2f} | {br_s} | {_n(bm)} | {_n(em,1)} |")
        return rows, vals, feats

    real_rows, real_vals, real_feats = _probe(real_pick)
    false_rows, false_vals, false_feats = _probe(false_pick)
    auc = _auc(real_vals, false_vals)
    # CALIBRATE the multi-feature confidence model on these labeled examples (REAL vs
    # FALSE). Applied per-edit at the gate (option #1) and its per-feature separation
    # is reported below (option #2). Fit is leak-free: labels come only from the train
    # dominant-neuron rule. Falls back to a bridge_ratio sigmoid if too few examples.
    model = imgconf.ConfidenceModel.fit(real_feats, false_feats)
    # Per-feature separability (AUC of REAL vs FALSE for EACH feature), so the reviser
    # sees WHICH image features actually separate real from false — the signal it
    # needs to evolve an image rule beyond a single bridge_ratio threshold.
    feat_auc = {}
    for i, fname in enumerate(imgconf.FEATURES):
        rv = [f[i] for f in real_feats]
        fv = [f[i] for f in false_feats]
        feat_auc[fname] = _auc(rv, fv)

    lines = ["\n\n## Image warm-start probe: does bridge_ratio separate REAL from FALSE splits?\n"]
    lines.append(
        "HARNESS-measured (NOT your policy, NOT scored, never held-out): the harness "
        "read `gap_bridge_evidence` on a balanced, budget-capped sample of train "
        "SplitSites in the `gap_um` band where REAL and FALSE OVERLAP — i.e. where "
        "geometry alone CANNOT separate them, so image has the most decision value. "
        "Use this to decide, BEFORE writing any image rule, whether `bridge_ratio` is "
        "worth gating on, and roughly where the threshold sits. A high `bridge_ratio` "
        "(signal stays bright across the gap) should mark REAL splits you SHOULD "
        "merge.\n"
    )
    if auc is not None:
        verdict = ("strong — image is worth using" if auc >= 0.8 or auc <= 0.2 else
                   "moderate" if auc >= 0.65 or auc <= 0.35 else
                   "weak — bridge_ratio alone may not separate; do not over-invest")
        lines.append(
            f"**Separability: bridge_ratio AUC = {auc:.2f}** (1.0 = REAL always "
            f"brighter than FALSE; 0.5 = no separation). Read: {verdict}.\n")
    else:
        lines.append("_Separability AUC unavailable (a class had no valid read)._\n")

    # PER-FEATURE separation: bridge_ratio is ONE of several features the reader
    # returns in the same read. This shows how well EACH separates REAL from FALSE, so
    # a rule can combine them (a wide dark valley `valley_frac` or an uneven profile
    # `profile_cv` can flag a false join that bridge_ratio alone misses). AUC>0.5 means
    # REAL tends to have the HIGHER value of that feature; <0.5 the LOWER.
    have_feat_auc = any(v is not None for v in feat_auc.values())
    if have_feat_auc:
        lines.append("\n**Per-feature separability (REAL vs FALSE, same reads — no "
                     "extra cost):**")
        lines.append("| feature | AUC | direction (REAL is…) |")
        lines.append("|---|---|---|")
        for fname in imgconf.FEATURES:
            a = feat_auc.get(fname)
            if a is None:
                lines.append(f"| {fname} | — | (no valid pair) |")
            else:
                direction = ("higher" if a >= 0.55 else
                             "lower" if a <= 0.45 else "~no separation")
                lines.append(f"| {fname} | {a:.2f} | {direction} |")
        lines.append(f"\n**Calibrated confidence model:** {model.describe()}. The gate "
                     "uses this multi-feature model (not a single bridge_ratio "
                     "threshold) to gauge confidence on merges the held-out GT can't "
                     "verify. Combine the features above when writing your image rule "
                     "rather than thresholding bridge_ratio alone.\n")
    bh = "| gap_um | bridge_ratio | bridge_min | endpoint_mean |"
    bsep = "|---|---|---|---|"
    lines.append(f"**REAL splits — SHOULD merge** ({len(real_rows)} probed):")
    if real_rows:
        lines.append(bh); lines.append(bsep); lines.extend(real_rows)
    else:
        lines.append("_none probed._")
    lines.append(f"\n**FALSE joins — must NOT merge** ({len(false_rows)} probed):")
    if false_rows:
        lines.append(bh); lines.append(bsep); lines.extend(false_rows)
    else:
        lines.append("_none probed._")
    return {"section": lines, "auc": auc,
            "n_real": len(real_rows), "n_false": len(false_rows),
            "model": model, "feat_auc": feat_auc}


def write_failure_report(
    train_run: CandidateRun,
    baseline: scoring.ScoreResult,
    path: str,
    merge_labels: dict | None = None,
    label_gt_map: dict | None = None,
    fragments_graph=None,
    extra_sections: list | None = None,
    priors_section: list | None = None,
    merge_penalty: float = 100.0,
    table_budget: int = DEFAULT_TABLE_BUDGET,
) -> str:
    """Write the 'where you were wrong' report the agent reads to revise.

    Compares the candidate's train metrics to the no-edit baseline so the agent
    sees, per skeleton, whether its edits helped or hurt (splits repaired vs
    merges introduced). This is the feedback half of the self-improvement loop.

    Parameters
    ----------
    merge_labels : dict, optional
        Output of ``incremental_scoring.collect_merge_labels(prepared)`` — the
        BASELINE (pre-existing) merge errors by raw segment label. When given, a
        "Baseline merge errors" section lists which raw labels span multiple GT
        neurons (the ``split_label`` repair targets). DIAGNOSIS-ONLY: it is GT-
        derived, so the caller must pass only the TRAIN-split merges; it tells the
        reviser what to fix, it is never a held-out policy input.
    fragments_graph : SkeletonGraph, optional
        The run's cached fragment graph. When given, the SplitSite audit gains cheap,
        GT-free geometry per site (cross-gap colinearity, endpoint degrees, radii,
        radius ratio) so the reviser can pick a generalizable accept/reject feature
        threshold instead of gap alone. Geometry is fragment-only (leak-free).
    extra_sections : list of str, optional
        Pre-rendered markdown lines appended verbatim AFTER the body (e.g. the
        run-cached image warm-start probe from ``image_warmstart_probe``). Computed
        once per run by the caller and reused every generation, so it is independent
        of the per-generation policy/gate.
    priors_section : list of str, optional
        Pre-rendered 'grounding priors' lines, inlined near the TOP of the report
        when supplied. CURRENTLY UNUSED by run_evolution.py: the loop was rewound to
        the earlier design where the reviser is pointed at the discovery knowledge
        base and READS it itself (see ``_format_priors``), rather than the harness
        inlining a ranked menu. The parameter is retained (defaults to None => no
        section) so the inlining path can be re-enabled without a signature change.
        Population geometry only — leak-free, like every other section.
    """
    lines = ["# Candidate failure report (train split)\n"]
    if priors_section:
        lines.extend(priors_section)
    lines.extend(_failure_report_body(train_run, baseline, merge_labels, label_gt_map,
                                      fragments_graph, extra_sections, merge_penalty,
                                      table_budget=table_budget))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def write_multibrain_failure_report(per_brain: list, path: str,
                                    priors_section: list | None = None,
                                    merge_penalty: float = 100.0,
                                    table_budget: int = DEFAULT_TABLE_BUDGET) -> str:
    """Write ONE failure report aggregating several brains, each in its own section.

    ``per_brain`` is a list of ``(brain_id, train_run, baseline, merge_labels,
    label_gt_map[, fragments_graph[, extra_sections]])`` tuples — one per brain, each
    already scored on THAT brain's train split with THAT brain's merge_labels / train
    label→GT map (and optionally THAT brain's fragment graph for the SplitSite
    geometry, and pre-rendered extra sections such as its image warm-start probe). We
    never pool across brains here: raw segment-id labels are NOT unique across brains,
    so a brain's MergeSite / SplitSite TRUE/NON classification must use only its own
    maps. Each brain gets a ``# Brain <id>`` block built by the same per-brain body as
    the single-brain report; a short pooled header notes the brain set.

    ``priors_section`` (optional): pre-rendered 'grounding priors' lines, inlined ONCE
    after the pooled header (a cross-brain generality, so it is not repeated per
    brain). CURRENTLY UNUSED by run_evolution.py (the loop reads the priors file
    itself rather than inlining a menu — see ``write_failure_report``); retained for
    re-enabling without a signature change.
    """
    brain_ids = [str(b) for b, *_ in per_brain]
    lines = [
        f"# Candidate failure report (multi-brain, train split)\n",
        f"Brains in this report: {', '.join(brain_ids)}. Each brain is scored and "
        f"diagnosed SEPARATELY below (raw segment ids are not comparable across "
        f"brains). Look for FEATURE patterns that hold ACROSS brains — those "
        f"generalize; a rule that only helps one brain likely will not.\n",
    ]
    if priors_section:
        lines.extend(priors_section)
    for brain_id, train_run, baseline, merge_labels, label_gt_map, *rest in per_brain:
        fragments_graph = rest[0] if len(rest) >= 1 else None  # optional 6th element
        extra_sections = rest[1] if len(rest) >= 2 else None   # optional 7th element
        lines.append(f"\n\n{'='*60}")
        lines.append(f"# Brain {brain_id}\n")
        lines.extend(_failure_report_body(train_run, baseline, merge_labels, label_gt_map,
                                          fragments_graph, extra_sections, merge_penalty,
                                          table_budget=table_budget))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _failure_report_body(
    train_run: CandidateRun,
    baseline: scoring.ScoreResult,
    merge_labels: dict | None = None,
    label_gt_map: dict | None = None,
    fragments_graph=None,
    extra_sections: list | None = None,
    merge_penalty: float = 100.0,
    table_budget: int = DEFAULT_TABLE_BUDGET,
) -> list:
    """The body (all sections below the top header) of one brain's failure report.

    Returned as a list of markdown lines so both the single-brain and multi-brain
    writers can reuse it. All GT/merge-label use is restricted to this train split
    (``train_run.score.per_swc.index``), exactly as before.

    ``fragments_graph`` (optional) enriches the SplitSite audit with cheap, GT-free
    geometry per site (cross-gap colinearity, endpoint degrees, radii); None falls
    back to the gap-only audit. ``extra_sections`` (optional) is appended verbatim at
    the end (e.g. the run-cached image warm-start probe).

    ``merge_penalty`` is the per-false-merge penalty the gate applies (fitness =
    correct - false - merge_penalty*false); it is stated in the header so the reviser
    knows how costly a false merge is under the SMOOTH gate (not an automatic reject).
    """
    cand = train_run.score.per_swc
    base = baseline.per_swc.reindex(cand.index)
    lines = []
    # PRIMARY FITNESS = penalized split-repair score (the gate's actual keep
    # decision): classify each merge edit this generation against the TRAIN
    # label->neuron map (leak-free, same classifier the gate runs on held-out) into
    # correct (joins one neuron the segmentation broke) vs false (fuses two different
    # neurons) vs unscored (an endpoint not train-visible). The gate keeps a candidate
    # iff its FITNESS = (correct - false) - merge_penalty*false BEATS the parent's, so
    # this — not Edge Accuracy — is the number to move. Edge Accuracy is reported below
    # as a secondary diagnostic (it barely moves on a correct repair: only a bridged
    # split EDGE shifts it, which is why it is no longer the bar).
    repair = None
    if label_gt_map:
        from proofreader_evolve.harness import incremental_scoring as inc
        repair = inc.classify_merge_edits(train_run.edits, label_gt_map)
    if repair is not None:
        score = repair["correct"] - repair["false"]
        fitness = score - merge_penalty * repair["false"]
        # BLIND-SPOT HEADLINE: what fraction of THIS gen's edits the train GT can even
        # judge. In a sparsely-traced (esp. cross-brain) run this is ~1-3%, i.e. the
        # score above rests on a tiny minority of the policy's actual output; the rest
        # are graded on held-out neurons NOT visible in this report. Stated up top as a
        # headline (not buried as a parenthetical) so the reviser knows how thin the
        # ground under its fitness is before reading anything else.
        _n_edits = train_run.n_edits or 0
        _scored = repair["correct"] + repair["false"]
        _blind_frac = (repair["unscored"] / _n_edits) if _n_edits else float("nan")
        lines.append(
            f"- Proposed **{train_run.n_edits} edits** from "
            f"{train_run.n_sites} candidate sites.\n"
            f"- **BLIND SPOT: only {_scored} of your {_n_edits} edits ({(1 - _blind_frac):.0%}) "
            f"can be scored by train GT; the other {repair['unscored']} "
            f"({_blind_frac:.0%}) are `unscored` — they touch no train-traced neuron, so "
            f"this report cannot tell if they are right, and your gate fitness is decided "
            f"mostly by held-out neurons you cannot see here.** Prefer edits whose "
            f"geometry generalizes over ones that merely add unscored volume.\n"
            f"- Split-repair score = correct - false = {repair['correct']} - "
            f"{repair['false']} = {score} (train-classified; "
            f"unscored={repair['unscored']}).\n"
            f"- **FITNESS the gate keeps on = score - {merge_penalty:g}*false = "
            f"{fitness:g}.** The gate accepts iff this BEATS the parent's fitness. A "
            f"false merge (fusing two different neurons) is NOT an automatic reject "
            f"anymore, but it is HEAVILY penalized: each one costs {merge_penalty:g} "
            f"correct repairs to offset, so only accept a merge you are confident is "
            f"ONE neuron — a false merge is worth it only if the SAME revision adds "
            f">{merge_penalty:g} correct repairs per false merge.\n"
            f"- PRIMARY RECALL SIGNAL is the **'SplitSite audit'** section below "
            f"(REAL splits you MISSED = your recall headroom, with per-site GT-free "
            f"geometry to separate real splits from false joins). Read that section "
            f"FIRST after this header — it is the actionable one; the tables between "
            f"here and it are secondary diagnostics.\n"
            f"- Edge Accuracy (secondary diagnostic, NOT the bar; = 100 - %Split - "
            f"%Omit - %Merged): baseline "
            f"{scoring._weighted_avg(base, 'Edge Accuracy'):.4f} -> candidate "
            f"{train_run.score.primary:.4f}.\n"
            f"- Merge-error component: %Merged Edges baseline "
            f"{scoring._weighted_avg(base, '% Merged Edges'):.4f} -> candidate "
            f"{scoring._weighted_avg(cand, '% Merged Edges'):.4f}; "
            f"# Merges baseline {scoring._weighted_avg(base, '# Merges'):.2f} -> "
            f"candidate {scoring._weighted_avg(cand, '# Merges'):.2f}.\n"
            f"- Over-split watchdog: %Split Edges baseline "
            f"{scoring._weighted_avg(base, '% Split Edges'):.4f} -> candidate "
            f"{scoring._weighted_avg(cand, '% Split Edges'):.4f} "
            f"(a merge repair that drives this UP is over-splitting a real neuron).\n"
        )
        # WHERE the repairs (and the false merges) come from. The harness attributes
        # your accepted edits' correct/false/unscored counts across the geometric
        # FEATURES of their SplitSites, showing the FEW features that best separate
        # correct from false this generation (data-driven: feature axes are discovered
        # from the site geometry, bucket edges are quantiles of THIS gen's edits — not
        # hand-picked). These are harness-computed geometry, NOT your code's path
        # names, but a "path" is a region of this space, so this tells you which
        # geometric regime to invest in vs. prune.
        _buckets = split_repair_by_bucket(train_run, label_gt_map, fragments_graph)
        if _buckets:
            lines.append("\n### Split-repair attribution — most discriminative "
                         "feature axes (where your repairs vs. mistakes come from)\n")
            for feat, cells in _buckets.items():
                lines.append(f"**{feat}** (quantile buckets) — correct / false / "
                             f"unscored (n):\n")
                for label in sorted(cells):
                    c = cells[label]
                    lines.append(f"  - {label}: {c['correct']} / {c['false']} / "
                                 f"{c['unscored']}  (n={c['n']})\n")
            lines.append("Read: a bucket with high correct and zero false is a regime "
                         "worth EXTENDING; one with few correct but >0 false (or mostly "
                         "unscored) is a regime to TIGHTEN or prune — do not spend more "
                         "generations micro-tuning a low-yield region of feature "
                         "space.\n")
    else:
        # No label map (rare; e.g. a caller without train GT) — fall back to the
        # Edge-Accuracy header so the report still renders.
        lines.append(
            f"- Proposed **{train_run.n_edits} edits** from "
            f"{train_run.n_sites} candidate sites.\n"
            f"- Train Edge Accuracy (= 100 - %Split - %Omit - %Merged; higher is "
            f"better): baseline "
            f"{scoring._weighted_avg(base, 'Edge Accuracy'):.4f} -> candidate "
            f"{train_run.score.primary:.4f}.\n"
            f"- Merge-error component (the repair target): %Merged Edges baseline "
            f"{scoring._weighted_avg(base, '% Merged Edges'):.4f} -> candidate "
            f"{scoring._weighted_avg(cand, '% Merged Edges'):.4f}; "
            f"# Merges baseline {scoring._weighted_avg(base, '# Merges'):.2f} -> "
            f"candidate {scoring._weighted_avg(cand, '# Merges'):.2f}.\n"
            f"- Over-split watchdog: %Split Edges baseline "
            f"{scoring._weighted_avg(base, '% Split Edges'):.4f} -> candidate "
            f"{scoring._weighted_avg(cand, '% Split Edges'):.4f} "
            f"(a merge repair that drives this UP is over-splitting a real neuron).\n"
        )
    # MEGA-MERGE WATCHDOG (top-level, GT-free): the split-repair score is leak-blind
    # to fused classes whose neurons are held-out, so a union-find chain that fuses
    # dozens of fragments into ONE brain-spanning class can read "0 false / raise
    # recall" while wrecking %Merged Edges (the gen03/gen20 failure). Surface the
    # largest fused class HERE, at the top, so it is the first thing the reviser sees
    # instead of being buried in the per-merge table below. Sizes come from the SAME
    # Fused_Labels column that table uses (raw labels unified into one class id).
    ms = train_run.score.merge_sites
    if ms is not None and len(ms) and "Fused_Labels" in ms.columns:
        sizes = [len(f) for f in ms["Fused_Labels"] if isinstance(f, (list, tuple))]
        sizes = [s for s in sizes if s >= 2]  # only edit-created classes (>=2 raws)
        if sizes:
            biggest = max(sizes)
            n_big = sum(1 for s in sizes if s > 5)
            verdict = (
                "OK (no chaining)" if biggest <= 5 else
                f"**MEGA-MERGE: a single class fused {biggest} fragments** — almost "
                f"certainly an over-merge chaining distinct neurons, not one repair. "
                f"Cap it (ENUM_PARAMS/policy) before raising recall.")
            lines.append(
                f"- Largest fused class = **{biggest} raw labels** "
                f"({n_big} class(es) fuse >5 labels). {verdict}\n")
    # EDIT-CAUSED MERGE SITES (the geometric `# Merges` metric) — ADVISORY, NEVER
    # gated. The gate scores merges by ARGMAX (classify_merge_edits: do the two fused
    # labels' DOMINANT neurons match?), which is structurally blind to geometry: an
    # argmax-CORRECT repair can still drag a fragment arm into a spatial collision with
    # a NEIGHBOURING GT skeleton, which the geometric `# Merges` metric (MergeCountMetric:
    # a fragment leaf >50um from its own GT skeleton, then <6um to ANOTHER) counts as a
    # merge SITE. So edit-caused sites can be >0 while the gate's argmax `false` is 0 —
    # they are not contradictory, they measure different things. Surfacing them lets the
    # reviser PREFER repairs that don't also create a geometric collision, but this is
    # READ-ONLY: the accept/reject gate is unchanged (still split-repair score with zero
    # held-out argmax-false). Leak-safe: train_run.score.merge_sites is the TRAIN run's,
    # already restricted to the train neurons.
    if ms is not None and len(ms) and "Caused_By_Edit" in ms.columns:
        caused = ms[ms["Caused_By_Edit"].astype(bool)]
        n_caused = int(len(caused))
        if n_caused:
            gate_false = "n/a" if repair is None else str(repair["false"])
            lines.append(
                "\n## Edit-caused merge sites (geometric `# Merges`) — ADVISORY, NOT the gate\n")
            lines.append(
                f"- Your merges created **{n_caused} geometric merge SITE(s)** on the TRAIN "
                f"neurons (the `# Merges` / `% Merged Edges` metrics react to these), while "
                f"the gate's argmax `false` count is **{gate_false}**. These two numbers "
                f"measure DIFFERENT things and legitimately differ:\n"
                f"  - the **gate** asks: do the two fused labels' DOMINANT (argmax) GT "
                f"neurons match? — no geometry; this is the ONLY accept/reject signal;\n"
                f"  - **`# Merges`** asks: does a fused fragment arm physically land on a "
                f"NEIGHBOURING GT skeleton (a geometric collision site)?\n"
                f"- An argmax-correct split repair can STILL create a geometric collision "
                f"(e.g. a thin arm grazing an adjacent neuron). Such sites do NOT reject the "
                f"candidate, but they erode %Merged Edges / Edge Accuracy. Where they "
                f"cluster, PREFER a repair that fixes the split WITHOUT adding one — tighten "
                f"the geometry (continuity / caliber / endpoint degree) for that regime "
                f"rather than chasing the last bit of recall there.\n")
            if "GroundTruth_ID" in caused.columns:
                vc = caused["GroundTruth_ID"].value_counts()
                lines.append("\nEdit-caused merge sites by GT neuron (top by count):\n")
                for gt_id, n in vc.head(8).items():
                    lines.append(f"  - {gt_id}: {int(n)} site(s)\n")
                if len(vc) > 8:
                    lines.append(f"  - …and {len(vc) - 8} more neuron(s).\n")
    # rail (ds.ENUM_PARAM_SPEC). A reviser that asked for, say, max_gap_um=80 but is
    # silently capped at 40 would otherwise never learn its request had no effect —
    # this table makes the requested→in-effect→rail mapping explicit and flags any
    # knob that hit a rail (so it stops tuning a dead knob) or is parked AT a rail (so
    # it knows the candidate stream is at its widening/narrowing limit).
    enum_in = dict(getattr(train_run, "enum_params", None) or {})
    enum_raw = dict(getattr(train_run, "raw_enum_params", None) or {})
    if enum_in:
        spec = ds.ENUM_PARAM_SPEC
        rows = []
        n_clamped = 0
        for key, (default, lo, hi) in spec.items():
            eff = enum_in.get(key, default)
            requested = enum_raw.get(key, None)  # None => policy did not set it
            # Status: did the rail change the request, or is it parked at a bound?
            status = "default" if key not in enum_raw else "ok"
            if lo is not None and key in enum_raw:
                try:
                    req_f = type(default)(requested)
                    if req_f < lo:
                        status = "clamped↑ to lo"; n_clamped += 1
                    elif req_f > hi:
                        status = "clamped↓ to hi"; n_clamped += 1
                except (TypeError, ValueError):
                    status = "invalid→default"
            # Flag parked-at-rail even when not clamped (e.g. default already at a rail,
            # or an in-range request that equals a bound) — the stream is at its limit.
            if lo is not None:
                try:
                    if float(eff) <= float(lo):
                        status += " [AT lo rail]"
                    elif float(eff) >= float(hi):
                        status += " [AT hi rail]"
                except (TypeError, ValueError):
                    pass
            req_s = "—" if requested is None else str(requested)
            rail_s = "—" if lo is None else f"[{lo}, {hi}]"
            rows.append(f"| {key} | {req_s} | {eff} | {rail_s} | {status} |")
        lines.append("\n## ENUM_PARAMS rail sensitivity (candidate-stream knobs)\n")
        lines.append(
            "The candidate stream you reason over is shaped by `ENUM_PARAMS` "
            "(module-level dict in heuristics.py). Each knob is CLAMPED to a safety "
            "rail; `requested` is what your policy asked for ('—' = not set, framework "
            "default used), `in_effect` is what actually shaped THIS run's stream. If "
            "a knob is `clamped`, your request had NO effect past the rail — stop "
            "tuning it. If it is `[AT … rail]`, the stream is already at its "
            "widening/narrowing limit in that direction.\n"
        )
        if n_clamped:
            lines.append(
                f"**{n_clamped} knob(s) were CLAMPED this run — your requested value "
                f"was overridden by the rail.**\n")
        lines.append("| knob | requested | in_effect | rail [lo, hi] | status |")
        lines.append("|---|---|---|---|---|")
        lines.extend(rows)

    lines.append("\n## Per-skeleton delta (candidate - baseline)\n")
    # % Omit Edges is shown because an edit that relabels a label<->background
    # boundary node moves omit (an edge is an omit edge when EITHER endpoint is
    # background), so Edge Accuracy can shift via the omit term, not only via
    # split/merge — the column lets the reviser attribute an EdgeAcc change.
    lines.append(
        "| GT skeleton | dEdgeAcc | d%MergedEdges | d#Merges | d%SplitEdges | "
        "d#Splits | d%OmitEdges |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name in cand.index:
        def d(col):
            return cand.loc[name, col] - base.loc[name, col]
        lines.append(
            f"| {name} | {d('Edge Accuracy'):+.3f} | {d('% Merged Edges'):+.3f} | "
            f"{d('# Merges'):+.0f} | {d('% Split Edges'):+.3f} | {d('# Splits'):+.0f} | "
            f"{d('% Omit Edges'):+.3f} |"
        )
    # Reading aid (verified by experiment): a RISING d#Splits alongside a FALLING
    # d%OmitEdges is NOT you over-splitting. It is a side effect of your merges: a
    # merge heals omit (background) nodes into labels, so a GT skeleton ends up
    # crossing MORE distinct fragment labels — # Splits = (distinct non-zero labels
    # on the skeleton) - 1, so it rises. d#Splits is a GT-side DIAGNOSTIC; it does
    # NOT enter the gate (the gate is the split-repair score below). Likewise dEdgeAcc
    # is a secondary diagnostic, not the bar.
    lines.append(
        "\n_Note: a positive `d#Splits` paired with a negative `d%OmitEdges` is the "
        "footprint of your MERGES healing background (omit) nodes into labels (the "
        "skeleton then crosses more distinct labels) — not over-splitting by you. "
        "`# Splits` / Edge Accuracy are GT-side diagnostics, NOT the gate._\n"
    )
    # TRAIN-SIDE FALSE MERGES (diagnostic alert, NOT a gate failure). The gate judges
    # accept/reject on the HELD-OUT split only (kept separate so it is an honest
    # generalization signal); these false merges are scored on the TRAIN skeletons the
    # reviser can see. They do NOT by themselves reject the candidate — but each is a
    # REAL over-merge (your edit fused two DIFFERENT neurons on a train skeleton), so
    # they are precision bugs to fix even though the held-out gate did not catch them.
    # Surfacing them keeps train-side over-merges from staying invisible (the gate's
    # blind spot: it never penalizes a fusion whose partners are train / no-GT region).
    lines.append("\n## Train-side FALSE merges (precision alert — NOT a gate reject)\n")
    if repair is not None:
        if repair["false"]:
            lines.append(
                f"**{repair['false']} merge(s) fused two DIFFERENT neurons on a TRAIN "
                f"skeleton.** These are real over-merges your policy should stop "
                f"emitting (add a precision guard on these label pairs / their "
                f"geometry). NOTE: the gate scores false merges on HELD-OUT, not here, "
                f"so these did NOT reject the candidate — but they ARE genuine "
                f"over-merges (the gate's blind spot on train / no-GT regions):")
            lines.append("| label_a | label_b | neuron_a | neuron_b |")
            lines.append("|---|---|---|---|")
            for a, b, na, nb in repair["false_pairs"][:30]:
                lines.append(f"| {a} | {b} | {na} | {nb} |")
            if len(repair["false_pairs"]) > 30:
                lines.append(f"\n…and {len(repair['false_pairs']) - 30} more train-side false merges.")
        else:
            lines.append("_no train-side false merges — your merges did not fuse two "
                         "different neurons on any train skeleton._")
    # Secondary: per-skeleton Edge-Accuracy dips (diagnostic, not a gate failure).
    worse = [n for n in cand.index
             if cand.loc[n, "Edge Accuracy"] < base.loc[n, "Edge Accuracy"]]
    lines.append("\n## Skeletons with an Edge-Accuracy dip (secondary diagnostic)\n")
    lines.append(", ".join(worse) if worse else "_none — every skeleton improved or held._")

    # Baseline merge errors: which raw labels span >=2 GT neurons in the unedited
    # segmentation — the concrete repair targets for split_label. GT-derived, so
    # train-split only (diagnosis, not a held-out signal). Restricting to skeletons
    # in this report's `cand.index` keeps a train report from naming held-out GT.
    if merge_labels:
        report_gt = set(cand.index)
        rows = []
        for label, info in merge_labels.items():
            involved = [n for n in info["gt_skeletons"] if n in report_gt]
            if len(involved) < 2:
                continue  # the merge's partners are outside this split — not actionable here
            counts = "; ".join(f"{n}:{info['node_counts'][n]}" for n in involved)
            locs = " | ".join(
                f"({s['World'][0]:.1f},{s['World'][1]:.1f},{s['World'][2]:.1f})"
                for s in info["merge_sites"] if s["GroundTruth_ID"] in report_gt
            )
            rows.append((sum(info["node_counts"][n] for n in involved), label, involved, counts, locs))
        rows.sort(reverse=True)  # biggest (most nodes) merges first — best repair payoff
        lines.append("\n\n## Baseline merge errors (split_label repair targets)\n")
        if rows:
            lines.append(
                f"{len(rows)} raw label(s) span >=2 GT neurons in this split — each "
                f"is a merge a `split_label` edit should break apart (reduces "
                f"%Merged Edges / #Merges). Locations are GT-frame centroids (µm).\n"
            )
            lines.append("| raw label | # GT neurons | nodes per GT | per-GT centroid (x,y,z) |")
            lines.append("|---|---|---|---|")
            for _, label, involved, counts, locs in rows:
                lines.append(f"| {label} | {len(involved)} | {counts} | {locs} |")
        else:
            lines.append("_none — no raw label spans >=2 GT neurons in this split._")

    # MergeSite feature patterns, LABELLED by ground truth (train-only diagnosis).
    # The "Baseline merge errors" table above names WHICH raw labels are merges; this
    # one shows the GT-free GEOMETRY of the candidate MergeSites at those labels, side
    # by side with the geometry of MergeSites that are NOT merges. The reviser may
    # only key its policy on the FEATURE columns (detector/angle/radius/cable/...),
    # never the raw label — so a labelled positive/negative feature table is the
    # signal it needs to learn a GENERALIZABLE split_label threshold instead of
    # guessing.
    #
    # ISOLATION (critical): the TRUE/NON-merge label of each row must be derivable
    # from the TRAIN skeletons alone, or held-out GT leaks to the mutation operator
    # through this classification. ``merge_labels`` is computed over the FULL brain
    # (train + held-out), so we MUST NOT key the negative class on its membership —
    # doing so would let held-out GT shape the negative distribution the reviser
    # learns its threshold against (a leak, even if the dropped rows are never
    # shown). We therefore classify strictly by the TRAIN skeletons (``report_gt``):
    #   * >=2 train neurons carry the label -> TRUE merge (train-derivable) -> positive
    #   * otherwise                         -> negative
    # A merge only visible via held-out (or a train<->held-out crossing) thus lands
    # in the negative class. That injects symmetric label NOISE into the negatives
    # (some are really merges), but no held-out signal — the negative boundary the
    # reviser sees is a pure function of train. This is the deliberate purity/noise
    # trade-off: we accept noisier negatives to keep the report strictly leak-free.
    if merge_labels is not None:
        report_gt = set(cand.index)
        train_merge_labels = {
            str(label) for label, info in merge_labels.items()
            if sum(1 for n in info.get("gt_skeletons", []) if n in report_gt) >= 2
        }
        sites = list(getattr(train_run, "merge_sites", None) or [])

        def _f(v, nd=2):
            try:
                if v is None:
                    return "—"
                fv = float(v)
                return "NaN" if fv != fv else f"{fv:.{nd}f}"
            except (TypeError, ValueError):
                return str(v)

        def _site_row(s):
            lab = str(getattr(s, "label", ""))
            rec = getattr(s, "arms_reconverge", None)
            rec_s = "—" if rec is None else ("yes" if rec else "no")
            return (
                f"| {getattr(s, 'detector', '?')} | {_f(getattr(s, 'angle_deg', None))} | "
                f"{_f(getattr(s, 'radius_ratio', None))} | "
                f"{_f(getattr(s, 'cable_a_um', None),1)} | {_f(getattr(s, 'cable_b_um', None),1)} | "
                f"{getattr(s, 'branch_degree', '?')} | {rec_s} |"
            )

        # Two-way, train-derivable only: positive (label is a train merge) vs
        # negative (everything else). No held-out membership is consulted.
        pos = [s for s in sites
               if str(getattr(s, "label", "")) in train_merge_labels]
        neg = [s for s in sites
               if str(getattr(s, "label", "")) not in train_merge_labels]
        header = ("| detector | angle_deg | radius_ratio | cable_a | cable_b | "
                  "branch_degree | arms_reconverge |")
        sep = "|---|---|---|---|---|---|---|"

        lines.append("\n\n## MergeSite features at TRUE merges (split_label targets)\n")
        lines.append(
            "Each row is a candidate MergeSite whose label IS a baseline merge (a real "
            "fusion of >=2 GT neurons). These are the geometry patterns a `split_label` "
            "SHOULD fire on. Key your policy on these columns, never the raw label.\n"
        )
        # Representative selection over the merge geometry (spread by angle_deg, the
        # primary merge-vs-not axis) + an omitted-tail summary, so a large table is
        # sampled across its regimes instead of keeping only the first 60.
        def _angle_of(s):
            return getattr(s, "angle_deg", None)
        _ms_feat_keys = {
            "angle_deg": _angle_of,
            "radius_ratio": lambda s: getattr(s, "radius_ratio", None),
            "cable_a": lambda s: getattr(s, "cable_a_um", None),
        }
        if pos:
            lines.append(header); lines.append(sep)
            kept, kept_idx = _select_representative(pos, budget=table_budget, key=_angle_of)
            for s in kept:
                lines.append(_site_row(s))
            tail = _omitted_summary(pos, kept_idx, _ms_feat_keys)
            if tail:
                lines.append(tail.replace("more not shown", "more true-merge sites not shown"))
        else:
            lines.append("_no enumerated MergeSite lands on a baseline merge label "
                         "(see the recall-gap note below)._")

        lines.append("\n\n## MergeSite features at NON-(train)-merges (cutting here usually over-splits)\n")
        lines.append(
            "Same geometry for MergeSites whose label is NOT a train-visible merge — a "
            "`split_label` here would *usually* cut a single real neuron (raises "
            "%Split Edges). Use these as the negative class: pick thresholds that "
            "separate the table above from this one. (A few rows here may secretly be "
            "merges only visible on the held-out skeletons — that is intentional label "
            "noise so the table stays a pure function of TRAIN GT; do not try to "
            "second-guess individual rows.)\n"
        )
        if neg:
            lines.append(header); lines.append(sep)
            kept, kept_idx = _select_representative(neg, budget=table_budget, key=_angle_of)
            for s in kept:
                lines.append(_site_row(s))
            tail = _omitted_summary(neg, kept_idx, _ms_feat_keys)
            if tail:
                lines.append(tail.replace("more not shown", "more non-merge sites not shown"))
        else:
            lines.append("_no non-merge MergeSites enumerated._")

        # Recall gap: merge targets the enumerator produced NO MergeSite for. No policy
        # change can repair these — they need a better detector — so surface them
        # explicitly rather than letting them read as "already handled".
        if merge_labels:
            report_gt = set(cand.index)
            actionable = {
                str(label) for label, info in merge_labels.items()
                if sum(1 for n in info["gt_skeletons"] if n in report_gt) >= 2
            }
            site_labels = {str(getattr(s, "label", "")) for s in sites}
            missing = sorted(actionable - site_labels)
            lines.append("\n\n## Merge targets with NO candidate MergeSite (detector recall gap)\n")
            if missing:
                lines.append(
                    f"{len(missing)} baseline merge label(s) have no enumerated MergeSite, "
                    f"so `propose_edits` can never reach them — a `candidate_merge_sites` "
                    f"detector limitation, not a policy bug: "
                    + ", ".join(missing[:20])
                    + (f" …(+{len(missing) - 20} more)" if len(missing) > 20 else "")
                    + "\n"
                )
            else:
                lines.append("_none — every actionable merge target has at least one "
                             "candidate MergeSite._")

    # SplitSite feature audit, LABELLED by GT (train-only). Symmetric to the
    # MergeSite tables, but for the merge_labels (split-repair) lever. Without this
    # the reviser only learns precision from created-merge attribution ("stop
    # over-merging"); it has no signal for RECALL — which UNSELECTED SplitSites are
    # actually one neuron the segmentation broke and SHOULD be merged. We classify
    # each enumerated SplitSite by whether its two fragment labels map to the SAME
    # TRAIN GT neuron (a real split to repair) or DIFFERENT neurons (a join that
    # would over-merge). label_gt_map is the train-only {label: {gt_neuron: count}}
    # map (incremental_scoring.label_gt_counts), so the verdict is leak-free; a
    # label whose dominant neuron is held-out simply has no train entry and the site
    # is dropped (revealed to neither class). Features shown are the cheap, GT-free
    # ones carried on the site (gap_um + partner degree); richer geometric features
    # (colinearity/tangent) are the policy's to compute.
    split_sites = list(getattr(train_run, "split_sites", None) or [])
    if split_sites and label_gt_map:
        def _dominant(lbl):
            counts = label_gt_map.get(str(lbl))
            if not counts:
                return None
            return max(counts, key=counts.get)

        # Which label pairs did the policy ACTUALLY accept this generation? Cross
        # the enumerated audit against train_run.edits so each SplitSite can be
        # marked accepted/rejected — turning "REAL/FALSE" into the policy's own
        # confusion matrix: rejected-REAL = false NEGATIVES (recall misses), and
        # accepted-FALSE = false POSITIVES (the over-merges). Order-free pair keys.
        def _pair_key(a, b):
            a, b = str(a), str(b)
            return (a, b) if a <= b else (b, a)
        accepted_pairs = set()
        for e in (train_run.edits or []):
            if isinstance(e, dict):
                if e.get("kind", "merge_labels") != "merge_labels":
                    continue
                a, b = e.get("label_a"), e.get("label_b")
            elif isinstance(e, (tuple, list)) and len(e) >= 2:
                a, b = e[0], e[1]
            else:
                continue
            if a is not None and b is not None:
                accepted_pairs.add(_pair_key(a, b))

        # Cheap, GT-free geometry per site, from the fragment graph (when provided).
        # These are the FEATURE columns the reviser keys an accept/reject threshold on
        # — gap alone barely separates the buckets, but colinearity + degree + radius
        # ratio do. All fragment-only, so leak-free and identical on held-out.
        def _f(v, nd=2):
            try:
                if v is None:
                    return "—"
                fv = float(v)
                return "NaN" if fv != fv else f"{fv:.{nd}f}"
            except (TypeError, ValueError):
                return str(v)

        def _split_row(s, verdict):
            gap = getattr(s, "gap_um", None)
            geom = _split_site_geom(fragments_graph, s)
            # Reciprocal-neighbor signal (GT-free): "y" when the two endpoints are each
            # other's #1 partner (a strong TRUE-split cue); otherwise the (rank_a,
            # rank_b) pair so the reviser can see how far down each tip's preference
            # list the partner sits. 0 => unranked/undefined.
            ra = int(getattr(s, "recip_rank_a", 0) or 0)
            rb = int(getattr(s, "recip_rank_b", 0) or 0)
            mutual = "y" if getattr(s, "mutual_nearest", False) else f"{ra},{rb}"
            return (
                f"| {_f(gap)} | {_f(geom['colinear_cos'])} | "
                f"{geom['deg_a'] if geom['deg_a'] is not None else '—'} | "
                f"{geom['deg_b'] if geom['deg_b'] is not None else '—'} | "
                f"{_f(geom['rad_a'])} | {_f(geom['rad_b'])} | "
                f"{_f(geom['rad_ratio'])} | {mutual} | {verdict} |"
            )

        # Buckets: (real|false) x (accepted|rejected). Each entry is (site, row_str) so
        # truncation can select representative rows by the site's geometry (gap /
        # colinear_cos), not just keep the first N.
        sp = {"real_acc": [], "real_rej": [], "false_acc": [], "false_rej": []}
        sp_drop = 0
        for s in split_sites:
            la, lb = str(getattr(s, "label_a", "")), str(getattr(s, "label_b", ""))
            da, db = _dominant(la), _dominant(lb)
            accepted = _pair_key(la, lb) in accepted_pairs
            verdict = "accepted" if accepted else "rejected"
            row = _split_row(s, verdict)
            if da is None or db is None:
                sp_drop += 1            # at least one label not train-visible -> drop
            elif da == db:
                sp["real_acc" if accepted else "real_rej"].append((s, row))
            else:
                sp["false_acc" if accepted else "false_rej"].append((s, row))

        n_real = len(sp["real_acc"]) + len(sp["real_rej"])
        n_false = len(sp["false_acc"]) + len(sp["false_rej"])
        has_geom = fragments_graph is not None
        lines.append("\n\n## SplitSite audit: REAL splits vs FALSE joins, crossed with your decision\n")
        lines.append(
            "Each enumerated SplitSite is classified by whether its two fragment "
            "labels belong to the SAME train GT neuron (a REAL split your "
            "`merge_labels` SHOULD repair — the RECALL signal) or DIFFERENT neurons "
            "(a FALSE join that would CREATE a merge — the precision signal), AND "
            "crossed with whether YOUR policy accepted (emitted a merge) or rejected "
            "it. So:\n"
            "  • **rejected REAL = your false NEGATIVES** (reachable real splits you "
            "MISSED — raise recall here);\n"
            "  • **accepted FALSE = your false POSITIVES** (over-merges that fail the "
            "gate — tighten here);\n"
            "  • accepted REAL = correct repairs; rejected FALSE = correct refusals.\n"
            "Labels whose dominant neuron is held-out are omitted (leak-free).\n"
        )
        if has_geom:
            lines.append(
                "Each row carries cheap, GT-free fragment geometry so you can pick a "
                "GENERALIZABLE accept/reject rule (gap alone barely separates the "
                "buckets):\n"
                "  • `colinear_cos` — straightness of armA→gap→armB: ~+1 a clean "
                "colinear continuation (the segmentation broke ONE neuron — merge), "
                "~0 a right-angle join, <0 the arms double back (likely a FALSE "
                "join). The single strongest precision feature here.\n"
                "  • `deg_a`/`deg_b` — endpoint graph degree (A is always a tip=1; B "
                "= 1 tip / 2 shaft / 3+ branch). Joining INTO a branch/shaft is "
                "riskier than tip-to-tip.\n"
                "  • `rad_a`/`rad_b`/`rad_ratio` — neurite radius at each end and "
                "their max/min. A ratio far from 1.0 = two different cable calibers "
                "(less likely one broken neuron).\n"
                "  • `mutual (recip #1?)` — reciprocal-neighbor test: `y` iff each "
                "endpoint is the OTHER's #1 (closest, differently-labelled) "
                "reconnection partner; otherwise the raw `(rank_a,rank_b)` pair "
                "(rank 1 = this endpoint's closest other-label partner; 0 = "
                "unranked). Two fragment ends that each pick each other are far more "
                "likely ONE broken neuron than a tip grazing an unrelated neurite "
                "that does not point back — a strong, gap-independent PRECISION cue. "
                "Read it off `site.mutual_nearest` / `site.recip_rank_a` / "
                "`site.recip_rank_b` in propose_edits (no split_geom call needed).\n"
                "Find the colinear_cos / rad_ratio / mutual cutoff that separates "
                "your MISSES from your hits below.\n"
            )
        else:
            lines.append(
                "Read the `gap_um` of each bucket to find the threshold that "
                "separates your misses from your hits.\n"
            )
        lines.append(
            f"Confusion summary: REAL {n_real} (accepted {len(sp['real_acc'])} / "
            f"MISSED {len(sp['real_rej'])}); FALSE {n_false} (WRONGLY accepted "
            f"{len(sp['false_acc'])} / correctly rejected {len(sp['false_rej'])}).\n")

        _sp_header = ("| gap_um | colinear_cos | deg_a | deg_b | rad_a | rad_b | "
                      "rad_ratio | mutual (recip #1?) | your decision |")
        _sp_sep = "|---|---|---|---|---|---|---|---|---|"

        # Feature extractors for representative selection + omitted-tail summary. Rows
        # are (site, row_str); spread over gap_um (the axis the reviser thresholds on),
        # and summarize the omitted tail over gap_um + colinear_cos so its distribution
        # stays visible instead of a blind "…and N more".
        def _gap_of(entry):
            return getattr(entry[0], "gap_um", None)

        def _colinear_of(entry):
            try:
                return _split_site_geom(fragments_graph, entry[0]).get("colinear_cos")
            except Exception:
                return None
        _sp_feat_keys = {"gap_um": _gap_of, "colinear_cos": _colinear_of}

        def _emit(title, rows, cap=table_budget):
            lines.append(f"\n**{title}** ({len(rows)} sites):")
            if rows:
                lines.append(_sp_header); lines.append(_sp_sep)
                kept, kept_idx = _select_representative(rows, budget=cap, key=_gap_of)
                lines.extend(row for _s, row in kept)
                tail = _omitted_summary(rows, kept_idx, _sp_feat_keys)
                if tail:
                    lines.append(tail)
            else:
                lines.append("_none._")

        # Lead with the two ACTIONABLE error buckets — misses and over-merges.
        _emit("MISSED real splits (rejected REAL — raise recall)", sp["real_rej"])
        _emit("WRONGLY accepted false joins (accepted FALSE — over-merges to kill)",
              sp["false_acc"])
        _emit("Correctly repaired (accepted REAL)", sp["real_acc"])
        _emit("Correctly refused (rejected FALSE)", sp["false_rej"])
        if sp_drop:
            lines.append(f"\n_({sp_drop} SplitSite(s) omitted: a label's dominant "
                         f"neuron is held-out, so no train-derivable verdict — "
                         f"excluded to keep the signal leak-free.)_")

        # ----------------------------------------------------------------------
        # SplitSite ENUMERATION RECALL CEILING (the analogue of the MergeSite
        # "detector recall gap"). The audit above only scores splits the enumerator
        # PRODUCED — so a real split the geometric scan never emitted is invisible
        # there: it silently caps recall with no row and no gradient. This section
        # makes that ceiling explicit, so the reviser can tell a miss it CAN fix (a
        # site it rejected — tune thresholds) from one only ENUM_PARAMS can (a site
        # that was never enumerated — widen the stream).
        #
        # Method (leak-free, GT-free geometry + the SAME train-only dominant-neuron
        # rule as the audit): group the train-visible fragment labels by their
        # dominant TRAIN neuron. A neuron split into |F| fragments needs |F|-1 merges
        # to be made whole. Connect those fragments by the enumerated REAL SplitSite
        # edges (union-find); the resulting component count C tells us how many of
        # those merges are REACHABLE (|F|-C) vs. beyond the enumerator (C-1, because
        # no enumerated site bridges the components). Summed over neurons this is the
        # achievable-recall ceiling; isolated fragments (no same-neuron enumerated
        # edge at all) are the starkest unreachable cases.
        # Build neuron -> set(fragment labels) from the train-only map (same source
        # as _dominant, so identical leak-free scope).
        neuron_frags: dict = {}
        for lbl in label_gt_map.keys():
            dom = _dominant(lbl)
            if dom is not None:
                neuron_frags.setdefault(dom, set()).add(str(lbl))

        # Union-find over fragment labels, joined by enumerated REAL edges only.
        uf: dict = {}
        def _ufind(x):
            uf.setdefault(x, x)
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x
        def _uunion(a, b):
            ra, rb = _ufind(a), _ufind(b)
            if ra != rb:
                uf[ra] = rb
        real_edges = 0
        for s in split_sites:
            la, lb = str(getattr(s, "label_a", "")), str(getattr(s, "label_b", ""))
            da, db = _dominant(la), _dominant(lb)
            if da is not None and da == db:   # a REAL (same-neuron) enumerated edge
                _uunion(la, lb)
                real_edges += 1

        needed = reachable = ceiling = 0          # split-repairs, summed over neurons
        frag_total = 0                            # fragments in multi-fragment neurons
        frag_unreachable = 0                      # isolated same-neuron fragments
        n_neurons_fragmented = 0                  # neurons broken into >=2 fragments
        n_neurons_ceiling = 0                     # neurons with an UNreachable fragment
        for dom, frags in neuron_frags.items():
            k = len(frags)
            if k < 2:
                continue                          # not fragmented -> no split to repair
            n_neurons_fragmented += 1
            frag_total += k
            comps = len({_ufind(f) for f in frags})
            needed += (k - 1)
            reachable += (k - comps)
            ceiling += (comps - 1)
            # Isolated fragments: a same-neuron label whose UF root is reached by no
            # other fragment of this neuron (its component is a singleton within F).
            root_counts = {}
            for f in frags:
                root_counts[_ufind(f)] = root_counts.get(_ufind(f), 0) + 1
            iso = sum(1 for f in frags if root_counts[_ufind(f)] == 1)
            if comps > 1:
                n_neurons_ceiling += 1
                frag_unreachable += iso

        est = train_run.split_enum_stats or {}
        lines.append("\n\n## SplitSite enumeration recall ceiling (what no policy change can reach)\n")
        if needed > 0:
            pct_reach = 100.0 * reachable / needed
            lines.append(
                f"Across **{n_neurons_fragmented}** train neurons broken into "
                f"≥2 fragments ({frag_total} fragments total), **{needed}** "
                f"`merge_labels` repairs are needed to make them whole. Of those, "
                f"**{reachable}** ({pct_reach:.1f}%) are REACHABLE — the enumerator "
                f"produced at least one REAL SplitSite bridging them, so a good policy "
                f"CAN make them; **{ceiling}** are NOT — no enumerated SplitSite "
                f"connects the pieces, so NO accept/reject change can repair them "
                f"(an enumeration limit, like the MergeSite detector recall gap, not a "
                f"policy bug). This {pct_reach:.1f}% is your achievable-recall CEILING "
                f"on train; the SplitSite audit's MISSED bucket lives BELOW it.\n"
            )
            if ceiling > 0:
                lines.append(
                    f"- **{n_neurons_ceiling}** of those neurons have a fragment no "
                    f"enumerated site reaches; **{frag_unreachable}** fragments are "
                    f"fully ISOLATED (no same-neuron SplitSite at all). To pull these "
                    f"into reach you must WIDEN the candidate stream via `ENUM_PARAMS` "
                    f"(raise `max_gap_um` for long true gaps; set `tip_to_shaft=True` "
                    f"if a partner is mid-shaft; raise `split_max_sites` if truncation "
                    f"is dropping them — see below), NOT tune your thresholds.\n"
                )
            else:
                lines.append(
                    "- Every fragmented neuron is fully connected by enumerated sites: "
                    "your recall is bounded ONLY by your own accept/reject thresholds, "
                    "not by enumeration. Tune thresholds, not `ENUM_PARAMS`.\n"
                )
        else:
            lines.append(
                "_No train neuron is split into ≥2 fragments under the current map — "
                "no split-repair ceiling to report._\n"
            )

        # Truncation: the split_max_sites cap keeps the CLOSEST gaps, so a real split
        # at a far gap can be silently dropped. Pure geometry (GT-free), so always safe.
        if est:
            n_pairs = est.get("n_pairs_enumerated")
            n_trunc = est.get("n_truncated") or 0
            cap = est.get("max_sites")
            lines.append("\n### Candidate-stream truncation (`split_max_sites` cap)\n")
            if n_trunc > 0:
                t0 = est.get("truncated_at_gap_um")
                t1 = est.get("max_truncated_gap_um")
                lines.append(
                    f"The scan found **{n_pairs}** label pairs within "
                    f"`max_gap_um`={est.get('max_gap_um')} µm but the `split_max_sites`"
                    f"={cap} cap kept only the **{est.get('n_returned')}** closest — "
                    f"**{n_trunc}** pairs (gaps "
                    f"{t0:.2f}–{t1:.2f} µm) were DROPPED before the policy saw them. "
                    f"Any real split among them is unreachable until you raise "
                    f"`split_max_sites` (rail 100–50000). Note the dropped pairs are the "
                    f"FARTHEST gaps (most likely false joins), so widen with care.\n"
                )
            else:
                lines.append(
                    f"The scan found **{n_pairs}** label pairs, all returned (cap "
                    f"`split_max_sites`={cap} not hit) — truncation is NOT limiting "
                    f"recall this run.\n"
                )
            # Per-tip quota + reciprocal-neighbor summary. The per-tip quota runs
            # BEFORE the global cap, so a dense region can no longer starve a sparse
            # tip's one true partner; n_mutual counts returned sites that are
            # reciprocal #1 pairs (the strong true-split cue exposed per row).
            ptk = est.get("per_tip_k")
            n_mut = est.get("n_mutual_nearest")
            if ptk is not None:
                lines.append(
                    f"Per-tip quota `split_per_tip_k`={ptk}: each tip contributes only "
                    f"its {ptk} closest differently-labelled partners BEFORE the global "
                    f"cap (rail 1–32), so a dense region cannot consume the whole budget "
                    f"and starve a sparse tip's single true partner. Raise it toward the "
                    f"max to approximate the old global-only stream; lower it to "
                    f"concentrate the budget on each tip's very best partners."
                    + (f" Of the returned sites, **{n_mut}** are reciprocal #1 "
                       f"(mutual-nearest) pairs — see the `mutual` column in the "
                       f"SplitSite audit.\n" if n_mut is not None else "\n")
                )

    # Image evidence the policy ALREADY fetched, labelled by GT (train-only). The
    # policy pays cloud reads for gap_connectivity / gap_bridge_evidence /
    # merge_cut_evidence on the few candidates it gates through; we recorded those
    # (zero extra reads) and split them TRUE/NON so the reviser can SEE which image
    # thresholds separate real merges/splits from false ones — the missing learning
    # signal for image-driven rules. Two readers get a leak-free TRUE/NON verdict:
    #   * merge_cut_evidence — its seed nodes belong to a MergeSite whose label we
    #     classify against the TRAIN merge set (split_label evidence).
    #   * gap_bridge_evidence — its nodes belong to a SplitSite whose label pair we
    #     classify by dominant TRAIN neuron via label_gt_map (merge_labels evidence).
    # gap_connectivity (endpoint-only) has no interior verdict, so its reads are just
    # counted.
    img_reads = list(getattr(train_run, "image_reads", None) or [])
    # Gate on label_gt_map, NOT merge_labels: gap_bridge_evidence and the SplitSite
    # branch of read_patch are SPLIT-repair (merge_labels-lever) image signals that
    # classify via label_gt_map alone and are wanted even in split-error-only runs
    # (where merge_labels is withheld). Only the merge_cut_evidence sub-table below
    # needs merge_labels, and it is guarded separately.
    if img_reads and label_gt_map:
        # Recompute the train merge label set locally (same leak-free rule as the
        # MergeSite tables) so this block is self-contained. The classification must
        # be a pure function of TRAIN GT: positive = label is a train merge,
        # negative = everything else. A held-out-only merge therefore lands in the
        # negative class as intentional label NOISE — exactly like the geometry
        # tables above — rather than being excluded via the full-brain merge set,
        # which would let held-out GT shape the negative distribution the reviser
        # calibrates its image threshold against (a leak, even if dropped rows are
        # never shown). Empty when merge_labels is withheld (split-error-only mode);
        # the merge_cut_evidence sub-table is then skipped, but the split-signal
        # tables (gap_bridge_evidence / read_patch) still render.
        _report_gt = set(cand.index)
        train_merge_labels = {
            str(label) for label, info in (merge_labels or {}).items()
            if sum(1 for n in info.get("gt_skeletons", []) if n in _report_gt) >= 2
        }
        # Map a MergeSite seed node -> its label, to classify merge_cut_evidence.
        seed_to_label = {}
        for s in (getattr(train_run, "merge_sites", None) or []):
            lab = str(getattr(s, "label", ""))
            for nd in (getattr(s, "seed_a_node", None), getattr(s, "seed_b_node", None)):
                if nd is not None:
                    seed_to_label[int(nd)] = lab
        # Map a SplitSite tip node -> its (label_a, label_b) pair, to classify
        # gap_bridge_evidence by the SAME train-only dominant-neuron rule as the
        # SplitSite audit (leak-free): REAL split (both labels' dominant TRAIN neuron
        # is the same) ⇒ a high bridge_ratio is the correct merge signal; FALSE join
        # (different dominant neurons) ⇒ a high bridge_ratio would be a false-merge
        # trap. A label with no train entry yields no verdict and the read is dropped.
        node_to_label_pair = {}
        for s in (getattr(train_run, "split_sites", None) or []):
            pair = (str(getattr(s, "label_a", "")), str(getattr(s, "label_b", "")))
            for nd in (getattr(s, "node_a", None), getattr(s, "node_b", None)):
                if nd is not None:
                    node_to_label_pair[int(nd)] = pair

        def _dom(lbl):
            counts = (label_gt_map or {}).get(str(lbl))
            return max(counts, key=counts.get) if counts else None

        def _g(d, k, nd=2):
            v = d.get(k)
            try:
                fv = float(v)
                return "NaN" if fv != fv else f"{fv:.{nd}f}"
            except (TypeError, ValueError):
                return "—"

        cut_pos, cut_neg = [], []
        for r in img_reads:
            if r.get("method") != "merge_cut_evidence":
                continue
            lab = seed_to_label.get(r.get("node_a")) or seed_to_label.get(r.get("node_b"))
            res = r.get("result", {})
            row = (f"| {_g(res,'valley_ratio')} | {_g(res,'valley')} | "
                   f"{_g(res,'endpoint_mean',1)} | {_g(res,'valley_pos')} |")
            if lab is not None and lab in train_merge_labels:
                cut_pos.append(row)
            elif lab is not None and lab not in train_merge_labels:
                cut_neg.append(row)

        # gap_bridge_evidence: classify by dominant TRAIN neuron of the SplitSite's
        # two labels (leak-free, same rule as the SplitSite audit). REAL split (same
        # dominant neuron) -> a high bridge_ratio confirms a merge SHOULD happen;
        # FALSE join (different) -> a high bridge_ratio here is the false-merge trap.
        bridge_pos, bridge_neg = [], []
        for r in img_reads:
            if r.get("method") != "gap_bridge_evidence":
                continue
            pair = (node_to_label_pair.get(r.get("node_a"))
                    or node_to_label_pair.get(r.get("node_b")))
            if not pair:
                continue
            da, db = _dom(pair[0]), _dom(pair[1])
            if da is None or db is None:
                continue                      # not train-visible -> no verdict
            res = r.get("result", {})
            row = (f"| {_g(res,'bridge_ratio')} | {_g(res,'bridge_min')} | "
                   f"{_g(res,'endpoint_mean',1)} | {_g(res,'bridge_pos')} |")
            (bridge_pos if da == db else bridge_neg).append(row)

        # read_patch: the policy read a raw cube to compute its OWN intensity feature.
        # We can't know what it computed, but we recorded a generic summary of each
        # cube; classify the cube's node by the site it belongs to (leak-free, reusing
        # the two maps above) so the reviser can SEE whether those cubes separate
        # repair targets from non-targets by train GT — the feedback that makes a
        # self-invented image feature evolvable instead of a blind guess. A node is a
        # TARGET if it is a REAL-split tip (both labels' dominant train neuron match)
        # or a TRUE-merge seed; a NON-target if it is a false-join tip or non-merge
        # seed. Nodes with no train-derivable verdict (held-out, or not on any site)
        # are dropped.
        patch_pos, patch_neg = [], []
        for r in img_reads:
            if r.get("method") != "read_patch":
                continue
            nd = r.get("node_a")
            verdict = None
            if nd in node_to_label_pair:               # SplitSite tip
                la, lb = node_to_label_pair[nd]
                da, db = _dom(la), _dom(lb)
                if da is not None and db is not None:
                    verdict = (da == db)
            elif nd in seed_to_label:                  # MergeSite seed
                lab = seed_to_label[nd]
                verdict = lab in train_merge_labels
            if verdict is None:
                continue
            res = r.get("result", {})
            row = (f"| {_g(res,'mean',1)} | {_g(res,'max',1)} | {_g(res,'p90',1)} | "
                   f"{_g(res,'occupancy',3)} | {_g(res,'std',1)} |")
            (patch_pos if verdict else patch_neg).append(row)

        gap_reads = [r for r in img_reads if r.get("method") == "gap_connectivity"]

        # merge_cut_evidence is a MERGE-repair (split_label-lever) signal: it needs
        # merge_labels to label TRUE vs NON merges, so skip the whole sub-table when
        # merge_labels is withheld (split-error-only mode — there are no MergeSites
        # and no merge_cut reads anyway).
        if merge_labels is not None:
            lines.append("\n\n## Image evidence the policy read (merge_cut_evidence), by GT class\n")
            if cut_pos or cut_neg:
                lines.append(
                    "Valley statistics from `merge_cut_evidence` calls the policy ALREADY "
                    "made, split by whether the cut's label is a TRUE (train) merge. A LOW "
                    "`valley_ratio` (signal dips between the arms) is the merge tell; "
                    "pick a `split_label` image threshold that separates these groups.\n"
                )
                ih = "| valley_ratio | valley | endpoint_mean | valley_pos |"
                isep = "|---|---|---|---|"
                lines.append(f"**TRUE merges** ({len(cut_pos)} read):")
                if cut_pos:
                    lines.append(ih); lines.append(isep); lines.extend(cut_pos[:40])
                else:
                    lines.append("_none read on a true-merge label this generation._")
                lines.append(f"\n**NON-merges** ({len(cut_neg)} read):")
                if cut_neg:
                    lines.append(ih); lines.append(isep); lines.extend(cut_neg[:40])
                else:
                    lines.append("_none read on a non-merge label this generation._")
            else:
                lines.append("_the policy made no merge_cut_evidence reads this generation "
                             "(or none mapped to a classifiable MergeSite)._")
        if bridge_pos or bridge_neg:
            lines.append("\n\n## Image evidence the policy read (gap_bridge_evidence), by GT class\n")
            lines.append(
                "Bridge statistics from `gap_bridge_evidence` calls the policy ALREADY "
                "made, split by whether the SplitSite's two labels are the SAME train "
                "neuron. A HIGH `bridge_ratio` (signal stays bright across the gap) is "
                "the merge tell; pick a `merge_labels` image threshold that ACCEPTS the "
                "REAL splits below and REJECTS the false joins.\n"
            )
            bh = "| bridge_ratio | bridge_min | endpoint_mean | bridge_pos |"
            bsep = "|---|---|---|---|"
            lines.append(f"**REAL splits — SHOULD merge** ({len(bridge_pos)} read):")
            if bridge_pos:
                lines.append(bh); lines.append(bsep); lines.extend(bridge_pos[:40])
            else:
                lines.append("_none read on a same-neuron SplitSite this generation._")
            lines.append(f"\n**FALSE joins — must NOT merge** ({len(bridge_neg)} read):")
            if bridge_neg:
                lines.append(bh); lines.append(bsep); lines.extend(bridge_neg[:40])
            else:
                lines.append("_none read on a cross-neuron SplitSite this generation._")
        if patch_pos or patch_neg:
            lines.append("\n\n## Image evidence the policy read (raw read_patch cubes), by GT class\n")
            lines.append(
                "Generic intensity summary of the raw cubes the policy read with "
                "`read_patch` (to compute its OWN image feature), split by whether the "
                "cube's node is a repair TARGET (a real split to merge / a true merge "
                "to split) or a NON-target by train GT. If a column separates the two "
                "groups, a feature built on it will generalize; if none do, the cube "
                "alone is not enough and you need the chord-based readers above. "
                "`occupancy` = fraction of voxels brighter than half the cube max.\n"
            )
            ph = "| mean | max | p90 | occupancy | std |"
            psep = "|---|---|---|---|---|"
            lines.append(f"**Repair TARGETS** ({len(patch_pos)} read):")
            if patch_pos:
                lines.append(ph); lines.append(psep); lines.extend(patch_pos[:40])
            else:
                lines.append("_none read on a repair-target node this generation._")
            lines.append(f"\n**NON-targets** ({len(patch_neg)} read):")
            if patch_neg:
                lines.append(ph); lines.append(psep); lines.extend(patch_neg[:40])
            else:
                lines.append("_none read on a non-target node this generation._")
        if gap_reads:
            lines.append(f"\n_({len(gap_reads)} gap_connectivity read(s) also made "
                         f"(endpoint-only split evidence — does not test the gap "
                         f"interior; prefer gap_bridge_evidence); not tabulated here.)_")

    # The concrete edits the policy proposed, so the reviser can reason about
    # *which* edit to change — not just that some skeleton regressed. Edits are
    # typed dicts ({"kind": "merge_labels"|"split_label"|...}); render by kind.
    lines.append("\n\n## Edits proposed\n")
    edits = train_run.edits or []
    if edits:
        from collections import Counter
        kinds = Counter(e.get("kind", "?") for e in edits)
        lines.append(f"{len(edits)} edits: "
                     + ", ".join(f"{n}× {k}" for k, n in kinds.items()) + "\n")

        def render(e):
            k = e.get("kind")
            if k == "merge_labels":
                return f"merge({e.get('label_a')}, {e.get('label_b')})"
            if k == "split_label":
                return f"split({e.get('label')} @ {e.get('seed_a_xyz')}|{e.get('seed_b_xyz')})"
            return f"{k}({ {x: e[x] for x in e if x != 'kind'} })"

        shown = edits[:40]
        lines.append(", ".join(render(e) for e in shown))
        if len(edits) > len(shown):
            lines.append(f"\n…and {len(edits) - len(shown)} more.")
    else:
        lines.append("_none — the policy proposed no edits._")

    # Per-merge attribution: which edit caused which merge, on which GT skeleton.
    # This is the high-value signal — it points the reviser at the exact label
    # pair to guard (e.g. add a direction/continuity check before joining it).
    lines.append("\n\n## Merges attributed to edits (the ones to fix)\n")
    ms = train_run.score.merge_sites
    caused = None
    if ms is not None and len(ms) and "Caused_By_Edit" in ms.columns:
        caused = ms[ms["Caused_By_Edit"]]
    if caused is not None and len(caused):
        lines.append(
            "Each row is a merge that an edit *created* (its class fuses >=2 raw "
            "labels). Guard these label pairs in the policy.\n"
        )
        lines.append("| GT skeleton | fused labels | merge location (world) |")
        lines.append("|---|---|---|")
        for _, row in caused.iterrows():
            fused = "+".join(map(str, row.get("Fused_Labels", [])))
            world = row.get("World", "")
            lines.append(f"| {row['GroundTruth_ID']} | {fused} | {world} |")
    elif ms is not None:
        lines.append("_no merge was caused by an edit (any merges are pre-existing in "
                     "the baseline segmentation)._")
    else:
        lines.append("_merge attribution unavailable for this run._")
    lines.append("\n")

    # Pre-rendered, run-cached sections (e.g. the image warm-start probe) appended
    # verbatim. Independent of this generation's policy/gate, so they give the
    # reviser a stable signal even from gen 1.
    if extra_sections:
        lines.extend(extra_sections)
    return lines
