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

from proofreader_evolve.harness import dataset as ds
from proofreader_evolve.harness import scoring


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


def _enumerate_sites_cached(fragments_graph, params: dict):
    """Return (split_sites, merge_sites), computing once per (graph, params).

    ``params`` is a resolved (validated + clamped) ENUM_PARAMS dict. The cache key
    includes EVERY enumeration param, not just the gap — otherwise a policy that
    changed an enumeration prior (e.g. min_arm_cable_um) would silently reuse sites
    enumerated under the old prior. Same params -> the whole-brain scan is paid once
    and reused across all (2*generations + 1) run_candidate calls; a new param tuple
    pays a fresh scan (and that is the intended cost of evolving the prior).
    """
    key = (id(fragments_graph),
           params["max_gap_um"], params["split_max_sites"], params["tip_to_shaft"],
           params["min_arm_cable_um"], params["seed_depth_um"],
           params["merge_max_sites"], params["max_per_label"])
    cached = _SITES_CACHE.get(key)
    if cached is None:
        split_sites = ds.candidate_split_sites(
            fragments_graph,
            max_gap_um=params["max_gap_um"],
            max_sites=params["split_max_sites"],
            tip_to_shaft=params["tip_to_shaft"],
        )
        merge_sites = ds.candidate_merge_sites(
            fragments_graph,
            min_arm_cable_um=params["min_arm_cable_um"],
            seed_depth_um=params["seed_depth_um"],
            max_sites=params["merge_max_sites"],
            max_per_label=params["max_per_label"],
        )
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
    split_sites, merge_sites = _enumerate_sites_cached(fragments_graph, enum_params)
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
    )


def write_failure_report(
    train_run: CandidateRun,
    baseline: scoring.ScoreResult,
    path: str,
    merge_labels: dict | None = None,
    label_gt_map: dict | None = None,
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
    """
    lines = ["# Candidate failure report (train split)\n"]
    lines.extend(_failure_report_body(train_run, baseline, merge_labels, label_gt_map))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def write_multibrain_failure_report(per_brain: list, path: str) -> str:
    """Write ONE failure report aggregating several brains, each in its own section.

    ``per_brain`` is a list of ``(brain_id, train_run, baseline, merge_labels,
    label_gt_map)`` tuples — one per brain, each already scored on THAT brain's train
    split with THAT brain's merge_labels / train label→GT map. We never pool across
    brains here: raw segment-id labels are NOT unique across brains, so a brain's
    MergeSite / SplitSite TRUE/NON classification must use only its own maps. Each
    brain gets a ``# Brain <id>`` block built by the same per-brain body as the
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
    for brain_id, train_run, baseline, merge_labels, label_gt_map in per_brain:
        lines.append(f"\n\n{'='*60}")
        lines.append(f"# Brain {brain_id}\n")
        lines.extend(_failure_report_body(train_run, baseline, merge_labels, label_gt_map))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _failure_report_body(
    train_run: CandidateRun,
    baseline: scoring.ScoreResult,
    merge_labels: dict | None = None,
    label_gt_map: dict | None = None,
) -> list:
    """The body (all sections below the top header) of one brain's failure report.

    Returned as a list of markdown lines so both the single-brain and multi-brain
    writers can reuse it. All GT/merge-label use is restricted to this train split
    (``train_run.score.per_swc.index``), exactly as before.
    """
    cand = train_run.score.per_swc
    base = baseline.per_swc.reindex(cand.index)
    lines = []
    lines.append(
        f"- Proposed **{train_run.n_edits} edits** from "
        f"{train_run.n_sites} candidate sites.\n"
        f"- Train Edge Accuracy (the fitness, = 100 - %Split - %Omit - %Merged; "
        f"higher is better): baseline "
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
    # Flag the rows where the candidate made things worse — the agent's homework.
    # "Worse" is measured on the fitness (Edge Accuracy), the thing the gate uses.
    worse = [n for n in cand.index
             if cand.loc[n, "Edge Accuracy"] < base.loc[n, "Edge Accuracy"]]
    lines.append("\n## Skeletons the candidate made WORSE\n")
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

        sp_pos, sp_neg, sp_drop = [], [], 0
        for s in split_sites:
            la, lb = str(getattr(s, "label_a", "")), str(getattr(s, "label_b", ""))
            da, db = _dominant(la), _dominant(lb)
            gap = getattr(s, "gap_um", None)
            try:
                gap_s = f"{float(gap):.2f}"
            except (TypeError, ValueError):
                gap_s = "—"
            row = f"| {gap_s} |"
            if da is None or db is None:
                sp_drop += 1            # at least one label not train-visible -> drop
            elif da == db:
                sp_pos.append(row)      # same train neuron -> a real split to repair
            else:
                sp_neg.append(row)      # different neurons -> joining would over-merge

        lines.append("\n\n## SplitSite audit: which gaps are REAL splits (merge_labels targets)\n")
        lines.append(
            "Each enumerated SplitSite, classified by whether its two fragment labels "
            "belong to the SAME train GT neuron (a real split your `merge_labels` "
            "SHOULD repair — the RECALL signal) or DIFFERENT neurons (a join that "
            "would CREATE a merge — the precision signal). Use the `gap_um` "
            "distribution to separate them; richer features (colinearity, tangent "
            "agreement) you compute yourself from `ctx['fragments_graph']`. Labels "
            "whose dominant neuron is held-out are omitted (leak-free).\n"
        )
        lines.append(f"**REAL splits — SHOULD merge** ({len(sp_pos)} sites): "
                     f"gap_um distribution below.")
        if sp_pos:
            lines.append("| gap_um |"); lines.append("|---|")
            lines.extend(sp_pos[:60])
            if len(sp_pos) > 60:
                lines.append(f"\n…and {len(sp_pos) - 60} more real-split sites.")
        else:
            lines.append("_none enumerated on a same-neuron label pair._")
        lines.append(f"\n**FALSE joins — must NOT merge** ({len(sp_neg)} sites):")
        if sp_neg:
            lines.append("| gap_um |"); lines.append("|---|")
            lines.extend(sp_neg[:60])
            if len(sp_neg) > 60:
                lines.append(f"\n…and {len(sp_neg) - 60} more cross-neuron sites.")
        else:
            lines.append("_none enumerated on a cross-neuron label pair._")
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
    if img_reads and merge_labels is not None:
        # Recompute the train merge label set locally (same leak-free rule as the
        # MergeSite tables) so this block is self-contained. The classification must
        # be a pure function of TRAIN GT: positive = label is a train merge,
        # negative = everything else. A held-out-only merge therefore lands in the
        # negative class as intentional label NOISE — exactly like the geometry
        # tables above — rather than being excluded via the full-brain merge set,
        # which would let held-out GT shape the negative distribution the reviser
        # calibrates its image threshold against (a leak, even if dropped rows are
        # never shown).
        _report_gt = set(cand.index)
        train_merge_labels = {
            str(label) for label, info in merge_labels.items()
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
