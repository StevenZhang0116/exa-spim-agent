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
        # .gap_connectivity(node_a, node_b) (split evidence: signal on both sides of
        # a gap?), or .merge_cut_evidence(seed_a_node, seed_b_node) (merge evidence:
        # an intensity valley between two fused arms?) — each is a cloud read, so
        # gate it behind cheap geometric filters.
        "read_image_patch": image_reader,
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
    )


def write_failure_report(
    train_run: CandidateRun,
    baseline: scoring.ScoreResult,
    path: str,
    merge_labels: dict | None = None,
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
    cand = train_run.score.per_swc
    base = baseline.per_swc.reindex(cand.index)
    lines = ["# Candidate failure report (train split)\n"]
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
    # guessing. GT is used ONLY to assign the true/false column and is restricted to
    # this report's skeletons (train split), so it never reaches a held-out policy.
    if merge_labels is not None:
        merge_target_labels = {str(l) for l in (merge_labels or {}).keys()}
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

        pos = [s for s in sites if str(getattr(s, "label", "")) in merge_target_labels]
        neg = [s for s in sites if str(getattr(s, "label", "")) not in merge_target_labels]
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

        lines.append("\n\n## MergeSite features at NON-merges (cutting here over-splits)\n")
        lines.append(
            "Same geometry for MergeSites whose label is NOT a baseline merge — a "
            "`split_label` here would cut a single real neuron (raises %Split Edges). "
            "Use these as the negative class: pick thresholds that separate the table "
            "above from this one.\n"
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

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
