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

import importlib.util
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from proofreader_evolve.harness import dataset as ds
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
    Every value is a function of fragment geometry only, so it is leak-free and the
    same on train and held-out — the policy may key on these directly.
    """
    out = {"colinear_cos": None, "deg_a": None, "deg_b": None,
           "rad_a": None, "rad_b": None, "rad_ratio": None}
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
            if ai is not None:
                ta = pa - _node_xyz(g, ai)                # outward tangent of arm A
                n = float(np.linalg.norm(ta))
                if n > 0:
                    cos_a = float(np.dot(ta / n, gdir))
            if bi is not None:
                tb = _node_xyz(g, bi) - pb                # B's arm leaving the junction
                n = float(np.linalg.norm(tb))
                if n > 0:
                    cos_b = float(np.dot(tb / n, gdir))
            if cos_a is not None and cos_b is not None:
                out["colinear_cos"] = 0.5 * (cos_a + cos_b)
    except Exception:
        pass
    return out


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
    """Return (split_sites, merge_sites), computing once per (graph, params).

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
           params["min_arm_cable_um"], params["seed_depth_um"],
           params["merge_max_sites"], params["max_per_label"],
           bool(enumerate_merges))
    cached = _SITES_CACHE.get(key)
    if cached is None:
        split_sites = ds.candidate_split_sites(
            fragments_graph,
            max_gap_um=params["max_gap_um"],
            max_sites=params["split_max_sites"],
            tip_to_shaft=params["tip_to_shaft"],
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
        cached = (split_sites, merge_sites)
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

    def to_json(self) -> dict:
        return {
            "split": self.split,
            "n_sites": self.n_sites,
            "n_edits": self.n_edits,
            "primary": self.score.primary,
            "metrics": self.score.metrics,
            "seconds": self.score.seconds,
        }


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
    split_sites, merge_sites = _enumerate_sites_cached(
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
    }

    # The policy may return legacy (label_a, label_b) tuples OR typed edit dicts
    # ({"kind": "merge_labels"|"split_label"|...}). normalize_edits promotes both
    # to the typed form, so the loop accepts either without the old, lossy
    # tuple(map(str, e)) coercion (which silently corrupted dict edits into a
    # 4-tuple of their keys).
    raw_edits = propose_edits(sites, ctx)
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
    )


def write_failure_report(
    train_run: CandidateRun,
    baseline: scoring.ScoreResult,
    path: str,
    merge_labels: dict | None = None,
    label_gt_map: dict | None = None,
    fragments_graph=None,
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
    """
    lines = ["# Candidate failure report (train split)\n"]
    lines.extend(_failure_report_body(train_run, baseline, merge_labels, label_gt_map,
                                      fragments_graph))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def write_multibrain_failure_report(per_brain: list, path: str) -> str:
    """Write ONE failure report aggregating several brains, each in its own section.

    ``per_brain`` is a list of ``(brain_id, train_run, baseline, merge_labels,
    label_gt_map[, fragments_graph])`` tuples — one per brain, each already scored on
    THAT brain's train split with THAT brain's merge_labels / train label→GT map (and
    optionally THAT brain's fragment graph for the SplitSite geometry). We never pool
    across brains here: raw segment-id labels are NOT unique across brains, so a
    brain's MergeSite / SplitSite TRUE/NON classification must use only its own maps.
    Each brain gets a ``# Brain <id>`` block built by the same per-brain body as the
    single-brain report; a short pooled header notes the brain set.
    """
    brain_ids = [str(b) for b, *_ in per_brain]
    lines = [
        f"# Candidate failure report (multi-brain, train split)\n",
        f"Brains in this report: {', '.join(brain_ids)}. Each brain is scored and "
        f"diagnosed SEPARATELY below (raw segment ids are not comparable across "
        f"brains). Look for FEATURE patterns that hold ACROSS brains — those "
        f"generalize; a rule that only helps one brain likely will not.\n",
    ]
    for brain_id, train_run, baseline, merge_labels, label_gt_map, *rest in per_brain:
        fragments_graph = rest[0] if rest else None  # optional 6th tuple element
        lines.append(f"\n\n{'='*60}")
        lines.append(f"# Brain {brain_id}\n")
        lines.extend(_failure_report_body(train_run, baseline, merge_labels, label_gt_map,
                                          fragments_graph))
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
) -> list:
    """The body (all sections below the top header) of one brain's failure report.

    Returned as a list of markdown lines so both the single-brain and multi-brain
    writers can reuse it. All GT/merge-label use is restricted to this train split
    (``train_run.score.per_swc.index``), exactly as before.

    ``fragments_graph`` (optional) enriches the SplitSite audit with cheap, GT-free
    geometry per site (cross-gap colinearity, endpoint degrees, radii); None falls
    back to the gap-only audit.
    """
    cand = train_run.score.per_swc
    base = baseline.per_swc.reindex(cand.index)
    lines = []
    # PRIMARY FITNESS = split-repair score (the gate's actual keep decision):
    # classify each merge edit this generation against the TRAIN label->neuron map
    # (leak-free, same classifier the gate runs on held-out) into correct (joins one
    # neuron the segmentation broke) vs false (fuses two different neurons) vs
    # unscored (an endpoint not train-visible). The gate keeps a candidate iff it
    # makes MORE net correct repairs than the parent AND zero false merges, so this
    # — not Edge Accuracy — is the number to move. Edge Accuracy is reported below
    # as a secondary diagnostic (it barely moves on a correct repair: only a bridged
    # split EDGE shifts it, which is why it is no longer the bar).
    repair = None
    if label_gt_map:
        from proofreader_evolve.harness import incremental_scoring as inc
        repair = inc.classify_merge_edits(train_run.edits, label_gt_map)
    if repair is not None:
        score = repair["correct"] - repair["false"]
        lines.append(
            f"- Proposed **{train_run.n_edits} edits** from "
            f"{train_run.n_sites} candidate sites.\n"
            f"- **Split-repair score (THE FITNESS the gate keeps on) = correct - "
            f"false = {repair['correct']} - {repair['false']} = {score}** "
            f"(train-classified merges; unscored={repair['unscored']}). The gate "
            f"accepts only if this BEATS the parent AND false == 0 — a single false "
            f"merge (fusing two different neurons) rejects the candidate outright.\n"
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
    # ENUM_PARAMS rail sensitivity: the policy MAY define a module-level ENUM_PARAMS
    # dict to widen/narrow the candidate stream, but every knob is CLAMPED to a safety
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
    # What made the candidate WORSE in the GATE's currency: any FALSE merge (fusing
    # two different neurons) is an automatic reject, so list those edits explicitly —
    # they are the agent's first homework. (The per-skeleton Edge-Accuracy dips below
    # are a secondary diagnostic; a dip there does NOT by itself fail the gate.)
    lines.append("\n## What FAILED the gate (split-repair currency)\n")
    if repair is not None:
        if repair["false"]:
            lines.append(
                f"**{repair['false']} FALSE merge(s) — each fuses two DIFFERENT "
                f"neurons and rejects the candidate outright. Stop emitting these:**")
            lines.append("| label_a | label_b | neuron_a | neuron_b |")
            lines.append("|---|---|---|---|")
            for a, b, na, nb in repair["false_pairs"][:30]:
                lines.append(f"| {a} | {b} | {na} | {nb} |")
            if len(repair["false_pairs"]) > 30:
                lines.append(f"\n…and {len(repair['false_pairs']) - 30} more false merges.")
        else:
            lines.append("_no false merges — the no-new-merge guard is satisfied; "
                         "improve by RAISING correct repairs (recall)._")
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
        if pos:
            lines.append(header); lines.append(sep)
            for s in pos[:60]:
                lines.append(_site_row(s))
            if len(pos) > 60:
                lines.append(f"\n…and {len(pos) - 60} more true-merge sites.")
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
            for s in neg[:60]:
                lines.append(_site_row(s))
            if len(neg) > 60:
                lines.append(f"\n…and {len(neg) - 60} more non-merge sites.")
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
            return (
                f"| {_f(gap)} | {_f(geom['colinear_cos'])} | "
                f"{geom['deg_a'] if geom['deg_a'] is not None else '—'} | "
                f"{geom['deg_b'] if geom['deg_b'] is not None else '—'} | "
                f"{_f(geom['rad_a'])} | {_f(geom['rad_b'])} | "
                f"{_f(geom['rad_ratio'])} | {verdict} |"
            )

        # Buckets: (real|false) x (accepted|rejected). Each row carries gap + geometry
        # + verdict.
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
                sp["real_acc" if accepted else "real_rej"].append(row)
            else:
                sp["false_acc" if accepted else "false_rej"].append(row)

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
                "Find the colinear_cos / rad_ratio cutoff that separates your MISSES "
                "from your hits below.\n"
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
                      "rad_ratio | your decision |")
        _sp_sep = "|---|---|---|---|---|---|---|---|"

        def _emit(title, rows, cap=60):
            lines.append(f"\n**{title}** ({len(rows)} sites):")
            if rows:
                lines.append(_sp_header); lines.append(_sp_sep)
                lines.extend(rows[:cap])
                if len(rows) > cap:
                    lines.append(f"\n…and {len(rows) - cap} more.")
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
    return lines
