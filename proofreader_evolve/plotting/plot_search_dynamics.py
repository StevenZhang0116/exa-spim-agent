"""
Search-dynamics & generalization view of one evolution run (complements the other
two figures). Reads ``runs/<run_name>/ledger.jsonl`` and renders a two-panel figure:

  (1) SEARCH DYNAMICS — one marker per generation, ACCEPTED *and* REJECTED (the
      other figures hide rejects). x = generation, y = held-out split-repair score
      (correct − false); marker colour = outcome (green ● accepted, red ✕ rejected),
      size ∝ #edits. A step line tracks the running PARENT score (best accepted so
      far); the flat stretches between its steps are the REJECTED attempts against
      that parent, so "how many tries per parent" and the near-misses are visible.
      Each rejected generation is annotated with WHY it was rejected — reconstructed
      from the ledger (near-miss / false-merge cost / timed-out / no-gain), reusing
      the loop's own NEAR-MISS rule (correct ≥ 20 and correct > 10 × false).

  (2) TRAIN vs HELD-OUT GENERALIZATION — the overfitting view the other figures
      lack. Edge Accuracy: train line vs held-out line (markers per gen); plus, on a
      twin axis, the over-merge counts train vs held-out (false merges). If a train
      gain does not carry to held-out, or train over-merges far less than held-out,
      you see it here. In CROSS-BRAIN runs train and held-out are DIFFERENT brains,
      so absolute Edge Accuracy is not directly comparable — the panel says so and
      shades the same-brain gap only when the run is a within-brain split.

Deterministic, headless (matplotlib Agg), no model. Mirrors the style of
``plot_run_performance.py`` and reuses its ledger loader.

Usage (from the ``exa-spim-agent/`` project root)
-------------------------------------------------
    python proofreader_evolve/plotting/plot_search_dynamics.py <run_name>
    python proofreader_evolve/plotting/plot_search_dynamics.py <run_name> --out fig.png
    python proofreader_evolve/plotting/plot_search_dynamics.py <run_name> --md digest.md

``run_name`` is the directory name under ``proofreader_evolve/runs/`` (a full path
is also accepted). Default output: ``runs/<run_name>/search_dynamics.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# HERE = proofreader_evolve/ (anchors runs/). This file is in proofreader_evolve/plotting/.
HERE = Path(__file__).resolve().parent.parent
RUNS_DIR = HERE / "runs"

# Reuse the exact ledger loader the performance figure uses, so the two figures never
# disagree on what a run's rows are.
try:
    from proofreader_evolve.plotting.plot_run_performance import (
        load_ledger, _resolve_run_dir,
    )
except Exception:  # pragma: no cover — allow running as a loose script
    def _resolve_run_dir(run_name: str) -> Path:
        p = Path(run_name)
        if p.is_dir():
            return p
        cand = RUNS_DIR / run_name
        if cand.is_dir():
            return cand
        raise FileNotFoundError(f"run not found: {run_name!r}")

    def load_ledger(run_dir: Path) -> list[dict]:
        path = run_dir / "ledger.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r.get("generation", 0))
        return rows


def _outcome_reason(r: dict) -> tuple[str, str]:
    """Return (outcome, terse_reason) for one generation, from the ledger row.

    ``outcome`` is "accepted" or "rejected". ``reason`` explains a rejection using the
    SAME logic the loop logs: a timeout, a high-recall near-miss (the loop's
    NEAR-MISS rule: correct ≥ 20 and correct > 10 × false), a plain false-merge cost,
    or no net gain. Accepted rows return an empty reason.
    """
    if r.get("accepted"):
        return "accepted", ""
    if r.get("heldout_policy_timed_out"):
        return "rejected", "timed out"
    correct = int(r.get("heldout_correct_merges", 0) or 0)
    # Use POLICY-CAUSED false merges (the only ones the gate penalizes); fall back to
    # total false for older ledgers that predate the split. Pre-existing false merges
    # are the policy's fault neither in the gate nor in this reason string.
    false = int(r.get("heldout_false_merges_policy",
                      r.get("heldout_false_merges", 0)) or 0)
    # The loop's own near-miss definition (see run_evolution NEAR-MISS alert): lots of
    # real repairs lost to only a few POLICY-caused false merges — a precision-cull
    # opportunity.
    if false > 0 and correct >= 20 and correct > 10 * false:
        return "rejected", f"near-miss: {correct} correct lost to {false} policy-false"
    if false > 0:
        return "rejected", f"{false} policy-false merge(s)"
    return "rejected", "no net gain"


def _running_parent_score(rows: list[dict]) -> list[float]:
    """Best accepted held-out split-repair score so far (the parent lineage), in SCORE
    units — the y the search-dynamics markers live in. Steps up only on an accept."""
    bar, best = [], None
    for r in rows:
        val = float(r.get("heldout_split_repair_score", 0) or 0)
        bar.append(best if best is not None else val)
        if r.get("accepted"):
            best = val if best is None else max(best, val)
    return bar


def _cross_brain_info(run_dir: Path) -> dict:
    """Read split.json for the cross-brain flag + brain roles (best-effort)."""
    try:
        d = json.loads((run_dir / "split.json").read_text())
        return {
            "cross_brain": bool(d.get("cross_brain")),
            "train_brains": d.get("train_brains"),
            "test_brains": d.get("test_brains"),
        }
    except Exception:
        return {"cross_brain": False, "train_brains": None, "test_brains": None}


def make_figure(rows: list[dict], run_name: str, out_path: Path,
                run_dir: Path | None = None) -> Path:
    """Render the two-panel search-dynamics + generalization figure to ``out_path``."""
    gens = [r.get("generation", i + 1) for i, r in enumerate(rows)]
    score = [float(r.get("heldout_split_repair_score", 0) or 0) for r in rows]
    n_edits = [int(r.get("heldout_n_edits", 0) or 0) for r in rows]
    accepted = [bool(r.get("accepted")) for r in rows]
    parent_score = _running_parent_score(rows)
    outcomes = [_outcome_reason(r) for r in rows]

    # Train vs held-out (panel 2). Both Edge Accuracy fields are in the ledger; false
    # merges too. NaN-safe for import-failed gens.
    tr_ea = [float(r.get("train_edge_accuracy", float("nan"))) for r in rows]
    ho_ea = [float(r.get("heldout_edge_accuracy", float("nan"))) for r in rows]
    # False merges, split into POLICY-caused (a genuinely new fusion of two clean
    # fragments) and PRE-EXISTING (a fragment already spanning both neurons — NOT the
    # policy's fault). BOTH sides are shown split, symmetrically.
    #   NOTE the field naming: the loop records the TRAIN policy-caused count in the
    #   un-suffixed ``train_false_merges`` (NOT a total — see run_evolution:
    #   train_false += classify_merge_edits(...)["false_policy"]), and the pre-existing
    #   count separately in ``train_false_merges_preexisting``. Held-out uses the
    #   explicit ``heldout_false_merges_policy`` / ``_preexisting`` pair. Older ledgers
    #   lack the preexisting fields -> fall back to preexisting = 0 (policy = recorded).
    tr_false_pol = [int(r.get("train_false_merges", 0) or 0) for r in rows]
    tr_false_pre = [int(r.get("train_false_merges_preexisting", 0) or 0) for r in rows]
    ho_false_pol = [int(r.get("heldout_false_merges_policy",
                             r.get("heldout_false_merges", 0)) or 0) for r in rows]
    ho_false_pre = [int(r.get("heldout_false_merges_preexisting", 0) or 0) for r in rows]
    have_blame = any(("heldout_false_merges_policy" in r
                      or "train_false_merges_preexisting" in r) for r in rows)

    xb = _cross_brain_info(run_dir) if run_dir is not None else {"cross_brain": False}
    cross = xb.get("cross_brain")
    splits_only = any(r.get("splits_only") for r in rows)
    mode = " [splits-only]" if splits_only else ""
    # Blind-spot caveat: fraction of held-out edits with NO GT verdict (mean over gens
    # that scored held-out). High => the standard metrics see almost nothing the policy
    # did, so panel-2 Edge Accuracy is a thin slice. Surfaced in the title.
    _uf = [float(r.get("heldout_unscored_fraction", float("nan"))) for r in rows]
    _uf = [v for v in _uf if v == v]
    mean_unscored = (sum(_uf) / len(_uf)) if _uf else float("nan")

    # Marker sizing by #edits (sqrt so a 50k-edit gen doesn't dwarf a 200-edit one).
    _emax = max(n_edits) if n_edits else 1
    def _msize(n):
        return 40 + 260 * (np.sqrt(max(n, 0)) / np.sqrt(_emax)) if _emax > 0 else 60
    acc_color = ["#2ca02c" if a else "#d62728" for a in accepted]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 11), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    # ---- Panel 1: SEARCH DYNAMICS (all gens, accepted + rejected) ----
    # Parent-lineage step (best accepted score so far). Its flat runs = attempts
    # against the same parent; each up-step = an accept.
    ax0.step(gens, parent_score, where="mid", color="#888", lw=1.3, ls="--",
             zorder=1, label="parent score (best accepted so far)")
    ax0.plot(gens, score, "-", color="#bbbbbb", lw=0.8, zorder=1)
    # Decide which rejected gens to ANNOTATE. Annotating every reject collides badly on
    # long runs (many flat repeats of the same reason). Rule: label a reject only when
    # it is INFORMATIVE — either a distinct near-miss score band (a real attempt above
    # the parent line, so worth naming), or the first of a run of identical reasons.
    # Repeated identical low-effort rejects are left as bare ✕ markers.
    _last_reason = None
    _seen_band = set()
    to_annotate = {}  # gen -> reason
    for g, s, a, (oc, reason) in zip(gens, score, accepted, outcomes):
        if a or not reason:
            _last_reason = None
            continue
        # Band the score to ~10-unit buckets so near-identical near-misses dedupe.
        band = (round(s / 10.0), reason.split(":")[0])
        interesting = (s > (parent_score[gens.index(g)] + 5)  # notably above parent
                       or reason != _last_reason)             # a new kind of failure
        if interesting and band not in _seen_band:
            to_annotate[g] = reason
            _seen_band.add(band)
        _last_reason = reason

    for i, (g, s, c, a, n, (oc, reason)) in enumerate(zip(
            gens, score, acc_color, accepted, n_edits, outcomes)):
        ax0.scatter([g], [s], c=c, s=_msize(n), marker="o" if a else "X",
                    edgecolors="k", linewidths=0.5, zorder=3)
        if g in to_annotate:
            ax0.annotate(to_annotate[g], (g, s), textcoords="offset points",
                         xytext=(0, 11 if (i % 2 == 0) else -18), ha="center",
                         fontsize=7, color="#7f1d1d",
                         bbox=dict(boxstyle="round,pad=0.15", fc="#fde8e8",
                                   ec="none", alpha=0.85))
    ax0.axhline(0, color="k", lw=0.6, alpha=0.4, zorder=0)
    ax0.set_ylabel("held-out split-repair score\n(correct − false)")
    ax0.set_title(
        f"Run {run_name}{mode} — search dynamics across {len(rows)} generations\n"
        f"● accepted (new parent) / ✕ rejected · marker size ∝ #edits · "
        f"annotations = why a candidate was rejected"
        + (f"\n[blind spot: mean {mean_unscored:.0%} of held-out edits are UNSCORED — "
           f"standard metrics see only the GT-covered remainder]"
           if mean_unscored == mean_unscored else ""))
    ax0.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    ax0.legend(handles=[
        Line2D([0], [0], color="#888", lw=1.3, ls="--", label="parent score (best accepted)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="k", markersize=9, label="accepted"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#d62728",
               markeredgecolor="k", markersize=9, label="rejected"),
    ], loc="best", fontsize=8, framealpha=0.9)

    # ---- Panel 2: TRAIN vs HELD-OUT GENERALIZATION ----
    ax1.plot(gens, tr_ea, "-o", color="#1f77b4", ms=4, lw=1.3, label="TRAIN Edge Accuracy")
    ax1.plot(gens, ho_ea, "-s", color="#ff7f0e", ms=4, lw=1.3, label="HELD-OUT Edge Accuracy")
    # Same-brain runs: the train↔held-out gap is a real overfit signal, so shade it.
    # Cross-brain runs compare DIFFERENT brains, so a subtraction is meaningless — skip.
    if not cross:
        _t = np.array(tr_ea, float); _h = np.array(ho_ea, float)
        _ok = ~(np.isnan(_t) | np.isnan(_h))
        if _ok.any():
            ax1.fill_between(np.array(gens)[_ok], _t[_ok], _h[_ok],
                             color="#9467bd", alpha=0.15, label="train↔held-out gap")
    ax1.set_ylabel("Edge Accuracy")
    ax1.grid(True, alpha=0.3)
    # Twin axis: over-merge counts (false merges) train vs held-out. Train is
    # diagnostic-only (never gates); held-out gates. BOTH bars are STACKED, symmetrically,
    # into POLICY-caused (solid — a genuinely new fusion; what the gate penalizes on
    # held-out) and PRE-EXISTING (hatched — a fragment already spanning both neurons, NOT
    # the policy's fault). Showing the pre-existing component on BOTH sides avoids the
    # misread where train looks far cleaner than held-out only because its (often large)
    # pre-existing false merges were hidden.
    ax1b = ax1.twinx()
    _w = 0.4
    xg = np.array(gens, float)
    # TRAIN (left half-bar, blue): policy solid + pre-existing hatched.
    ax1b.bar(xg - _w / 2, tr_false_pol, width=_w, color="#1f77b4", alpha=0.45,
             label="TRAIN false: POLICY-caused", zorder=0)
    if have_blame and any(tr_false_pre):
        ax1b.bar(xg - _w / 2, tr_false_pre, width=_w, bottom=tr_false_pol,
                 color="#1f77b4", alpha=0.16, hatch="////",
                 label="TRAIN false: pre-existing (NOT policy)", zorder=0)
    # HELD-OUT (right half-bar, red): policy solid + pre-existing hatched.
    ax1b.bar(xg + _w / 2, ho_false_pol, width=_w, color="#d62728", alpha=0.45,
             label="HELD-OUT false: POLICY-caused (penalized)", zorder=0)
    if have_blame and any(ho_false_pre):
        ax1b.bar(xg + _w / 2, ho_false_pre, width=_w, bottom=ho_false_pol,
                 color="#d62728", alpha=0.18, hatch="////",
                 label="HELD-OUT false: pre-existing (NOT penalized)", zorder=0)
    ax1b.set_ylabel("false merges (over-merges)")
    _tr_tot = [p + q for p, q in zip(tr_false_pol, tr_false_pre)]
    _ho_tot = [p + q for p, q in zip(ho_false_pol, ho_false_pre)]
    _fmax = max(_tr_tot + _ho_tot) if (_tr_tot or _ho_tot) else 0
    ax1b.set_ylim(0, max(1, _fmax) * 1.3)
    _brains = ""
    if cross:
        _brains = (f"  (CROSS-BRAIN: train={xb.get('train_brains')} vs "
                   f"test={xb.get('test_brains')} — absolute EA differs by brain; "
                   f"read the TREND, not the gap)")
    ax1.set_title("Train vs held-out generalization" + _brains, fontsize=10)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, framealpha=0.9)
    ax1.set_xlabel("generation")
    ax1.set_xticks(gens)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_md(rows: list[dict], run_name: str, run_dir: Path, path: Path) -> Path:
    """A compact per-generation Markdown digest of the search (accepted + rejected)."""
    xb = _cross_brain_info(run_dir)
    lines = [f"# Search digest — {run_name}", ""]
    if xb.get("cross_brain"):
        lines.append(f"- cross-brain: train={xb.get('train_brains')} → "
                     f"test={xb.get('test_brains')}")
    # false column shows correct / policy-caused (+pre-existing) — the gate penalizes
    # only the policy-caused count; pre-existing is shown in parens as context.
    lines += ["", "| gen | outcome | reason | held-out corr / false (policy+preexist) | "
              "score | train EA | held-out EA | Δdiff |",
              "|--:|:--|:--|--:|--:|--:|--:|:--|"]
    for r in rows:
        oc, reason = _outcome_reason(r)
        _fp = r.get("heldout_false_merges_policy", r.get("heldout_false_merges", 0))
        _fx = r.get("heldout_false_merges_preexisting", 0)
        lines.append(
            f"| {r.get('generation')} | {oc} | {reason or '—'} | "
            f"{r.get('heldout_correct_merges', 0)} / {_fp} (+{_fx}) | "
            f"{r.get('heldout_split_repair_score', 0)} | "
            f"{r.get('train_edge_accuracy', float('nan')):.3f} | "
            f"{r.get('heldout_edge_accuracy', float('nan')):.3f} | "
            f"{r.get('heuristics_diffstat', '')} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_name", help="run directory under proofreader_evolve/runs/ "
                                         "(or a full path to a run dir)")
    parser.add_argument("--out", default=None,
                        help="output PNG (default: runs/<run>/search_dynamics.png)")
    parser.add_argument("--md", default=None, help="also write a Markdown digest here")
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_name)
    rows = load_ledger(run_dir)
    if not rows:
        print(f"[plot_search_dynamics] {run_dir/'ledger.jsonl'} has no generations.")
        return 1

    out_path = Path(args.out) if args.out else (run_dir / "search_dynamics.png")
    make_figure(rows, run_dir.name, out_path, run_dir=run_dir)
    n_acc = sum(1 for r in rows if r.get("accepted"))
    print(f"[plot_search_dynamics] {len(rows)} generations, {n_acc} accepted -> {out_path}")
    if args.md:
        md_path = write_md(rows, run_dir.name, run_dir, Path(args.md))
        print(f"[plot_search_dynamics] digest -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
