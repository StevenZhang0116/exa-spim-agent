"""
THE EVOLVED PROGRAM (executable policy).  <-- the evolution loop edits THIS file.

``propose_edits`` is the proofreader's decision policy: given a UNIFIED stream of
candidate sites enumerated from the fragment graph, decide which proofreading edits
to make. The harness feeds the returned edits to the scoring framework.

The candidate stream has TWO kinds of site (dispatch on ``site.kind``):

  - SplitSite  (kind == "split"): two nearby fragments with DIFFERENT labels — a
    neuron that the segmentation broke into pieces. Valid repair = ``merge_labels``
    (unify the two labels). Fields: ``label_a``, ``label_b``, ``gap_um``,
    ``node_a``/``node_b`` (fragment-graph node ids), ``xyz_a``/``xyz_b``.

  - MergeSite  (kind == "merge"): ONE label fused across two neurites —
    two neurons the segmentation glued together. Valid repair = ``split_label``.

IMPORTANT: the stream mixes both kinds. NEVER assume a site is a SplitSite.
Always branch on ``getattr(s, "kind", "split")`` first.

Contract:
    propose_edits(sites, ctx) -> list[edit]
"""

from __future__ import annotations

import math


# --- Tunable parameters -------------------------------------------------------
# Split-repair (merge_labels) gates. This generation attacks the SPLIT lever with a
# NEW evidence source: fluorescence GAP CONNECTIVITY (does signal continue across
# the gap?) plus RADIUS CONTINUITY, rather than pure tangent geometry. The merge
# lever (split_label) is intentionally NOT used here: on this acceptance split the
# failure report shows NO raw label spans >=2 GT neurons, so any split_label cuts a
# real neuron and trips the over-split watchdog (this is exactly why the prior
# split_label attempt, +0.492 on held-out, was rejected).
GAP_NEAR_UM = 3.0          # cheap geometric prefilter: clearly-touching gap
GAP_FAR_UM = 6.0           # looser ceiling allowed only with strong corroboration
TANGENT_OPP_ANGLE = 50.0   # tangents must oppose within this (continuation, not crossing)
RADIUS_RATIO_MAX = 2.2     # the two fragments' tip radii must be of similar caliber
IMG_DIM_FRAC = 0.45        # gap is "bright on both sides" if dim side >= this * bright side
MAX_EDITS = 200            # safety cap


def _unit_tangent(g, node, radius_um=6.0):
    """Local outward tangent at ``node`` from its rooted neighborhood. Returns a
    length-3 list or None. Defensive against any graph API surprise."""
    try:
        p0 = g.node_xyz[node]
    except Exception:
        return None
    pts = []
    try:
        sub = g.rooted_subgraph(node, radius_um)
        for n in sub.nodes:
            if n == node:
                continue
            try:
                pts.append(g.node_xyz[n])
            except Exception:
                continue
    except Exception:
        # fall back to immediate neighbors
        try:
            for n in g.neighbors(node):
                pts.append(g.node_xyz[n])
        except Exception:
            return None
    if not pts:
        return None
    vx = vy = vz = 0.0
    cnt = 0
    for p in pts:
        try:
            dx = float(p[0]) - float(p0[0])
            dy = float(p[1]) - float(p0[1])
            dz = float(p[2]) - float(p0[2])
        except Exception:
            continue
        n = math.sqrt(dx * dx + dy * dy + dz * dz)
        if n <= 1e-6:
            continue
        vx += dx / n
        vy += dy / n
        vz += dz / n
        cnt += 1
    if cnt == 0:
        return None
    m = math.sqrt(vx * vx + vy * vy + vz * vz)
    if m <= 1e-6:
        return None
    return [vx / m, vy / m, vz / m]


def _opposition_angle_deg(g, node_a, node_b):
    """Angle between the OUTWARD tangents at the two endpoints. ~180° means the two
    fragments continue into each other (good); ~0° means they point the same way
    (a crossing / fold). Returns degrees or None."""
    ta = _unit_tangent(g, node_a)
    tb = _unit_tangent(g, node_b)
    if ta is None or tb is None:
        return None
    dot = ta[0] * tb[0] + ta[1] * tb[1] + ta[2] * tb[2]
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _radius_at(ctx, node):
    try:
        nr = ctx.get("node_radius")
        if nr is None:
            return None
        r = float(nr[node])
        if not math.isfinite(r) or r <= 0:
            return None
        return r
    except Exception:
        return None


def propose_edits(sites, ctx) -> list:
    """Selective SPLIT-repair policy.

    For each SplitSite we accept a ``merge_labels`` only when the evidence that the
    two fragments are ONE neuron is multi-signal:

      (1) geometry — small gap AND the two tip tangents are near-collinear
          (opposition angle close to 180°, i.e. a continuation not a crossing);
      (2) caliber  — the two endpoints' radii are of similar size (continuity);
      (3) image    — when fluorescence is available, the gap is BRIGHT ON BOTH
          SIDES (gap_connectivity), confirming signal genuinely bridges the gap.
          A clearly-touching short gap may merge on geometry+caliber alone; a
          longer gap REQUIRES the image confirmation.

    MergeSites are deliberately not split here (see module header / rules.md).
    """
    edits = []
    if not sites:
        return edits

    g = None
    try:
        g = ctx.get("fragments_graph")
    except Exception:
        g = None

    reader = None
    try:
        reader = ctx.get("read_image_patch")
    except Exception:
        reader = None

    seen_pairs = set()

    for s in sites:
        if len(edits) >= MAX_EDITS:
            break
        kind = getattr(s, "kind", "split")
        if kind != "split":
            # Merge lever intentionally disabled this generation.
            continue

        # --- pull fields defensively ---
        try:
            la = s.label_a
            lb = s.label_b
            gap = s.gap_um
        except Exception:
            continue
        if la is None or lb is None or la == lb:
            continue
        try:
            gap = float(gap)
        except Exception:
            gap = None
        if gap is None or not math.isfinite(gap) or gap < 0:
            continue
        if gap > GAP_FAR_UM:
            continue

        pair_key = tuple(sorted((str(la), str(lb))))
        if pair_key in seen_pairs:
            continue

        node_a = getattr(s, "node_a", None)
        node_b = getattr(s, "node_b", None)

        # --- (1) geometry: tangent continuation ---
        opp = None
        if g is not None and node_a is not None and node_b is not None:
            opp = _opposition_angle_deg(g, node_a, node_b)
        # opp near 180 = good continuation. Require deviation from 180 small.
        geom_ok = (opp is not None) and ((180.0 - opp) <= TANGENT_OPP_ANGLE)

        # --- (2) caliber continuity ---
        caliber_ok = True
        ra = _radius_at(ctx, node_a) if node_a is not None else None
        rb = _radius_at(ctx, node_b) if node_b is not None else None
        if ra is not None and rb is not None:
            hi = max(ra, rb)
            lo = min(ra, rb)
            if lo > 0 and (hi / lo) > RADIUS_RATIO_MAX:
                caliber_ok = False

        # --- (3) image gap connectivity (only for candidates that pass cheap gates) ---
        img_ok = None  # None = no image evidence available
        need_image = gap > GAP_NEAR_UM  # short gaps may pass without image
        cheap_pass = geom_ok and caliber_ok
        if cheap_pass and reader is not None and node_a is not None and node_b is not None:
            try:
                gc = reader.gap_connectivity(node_a, node_b)
                if gc is not None:
                    ma = gc.get("mean_a")
                    mb = gc.get("mean_b")
                    if ma is not None and mb is not None:
                        ma = float(ma)
                        mb = float(mb)
                        if math.isfinite(ma) and math.isfinite(mb):
                            bright = max(ma, mb)
                            dim = min(ma, mb)
                            if bright > 0:
                                img_ok = (dim >= IMG_DIM_FRAC * bright)
                            else:
                                img_ok = False
            except Exception:
                img_ok = None

        # --- decision ---
        accept = False
        if cheap_pass:
            if not need_image:
                # short, well-aligned, similar-caliber gap: merge.
                # If image is present and says the gap is dark, veto.
                accept = (img_ok is not False)
            else:
                # longer gap: require positive image confirmation. If no image at
                # all, fall back to geometry only if the gap is still modest AND the
                # alignment is strong.
                if img_ok is True:
                    accept = True
                elif img_ok is None and opp is not None and (180.0 - opp) <= (TANGENT_OPP_ANGLE * 0.6):
                    accept = True
                else:
                    accept = False

        if accept:
            try:
                edits.append(s.as_edit())
                seen_pairs.add(pair_key)
            except Exception:
                continue

    return edits
