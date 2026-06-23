"""
Plot one evolution run's performance across generations from its ledger.

Reads ``runs/<run_name>/ledger.jsonl`` (one JSON row per generation, written by
``harness.ledger.GenerationCost``) and renders a two-panel figure tracking how
the policy improved generation over generation:

  (1) Split-repair score (THE GATE METRIC) per generation, with the running
      PARENT bar (best accepted-so-far) overlaid as a step line, and each
      generation marked accepted (green ●) or rejected (red ✕). This is the one
      figure that answers "did it get better, and which gens were kept".
  (2) Cumulative cost (wall-clock minutes and agent $), so the accuracy gain can
      be read against the compute it took.

Deterministic, headless (matplotlib Agg), no model. Mirrors the style of
``agentic/summary_heatmap.py``.

Usage (from the ``exa-spim-agent/`` project root)
-------------------------------------------------
    python proofreader_evolve/plot_run_performance.py 789202_20260623_005939
    python proofreader_evolve/plot_run_performance.py <run_name> --out perf.png
    python proofreader_evolve/plot_run_performance.py <run_name> --csv series.csv

``run_name`` is the directory name under ``proofreader_evolve/runs/`` (a full path
is also accepted). Default output: ``runs/<run_name>/performance.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"


def _resolve_run_dir(run_name: str) -> Path:
    """Accept a bare run name (dir under runs/) or a full/relative path."""
    p = Path(run_name)
    if p.is_dir():
        return p
    cand = RUNS_DIR / run_name
    if cand.is_dir():
        return cand
    raise FileNotFoundError(
        f"run not found: {run_name!r} (looked for {p} and {cand})"
    )


def load_ledger(run_dir: Path) -> list[dict]:
    """Read the run's ledger.jsonl into a list of per-generation dicts (in order)."""
    path = run_dir / "ledger.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no ledger at {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r.get("generation", 0))
    return rows


def _running_parent_bar(rows: list[dict]) -> list[float]:
    """The split-repair bar each generation had to BEAT = best accepted score so far.

    The seed's score is the bar gen 1 faces; thereafter the bar advances only on an
    accepted generation (the gate is parent-relative). We reconstruct it from the
    ledger alone: start at the first generation's parent-implied bar (its own score
    if accepted, else it tells us the parent it failed to beat is <= its score), and
    step up whenever a generation is accepted.
    """
    bar = []
    best = None
    for r in rows:
        score = r.get("heldout_split_repair_score", 0)
        # The bar in force WHEN this gen was judged is the best accepted BEFORE it.
        bar.append(best if best is not None else score)
        if r.get("accepted"):
            best = score if best is None else max(best, score)
    return bar


def make_figure(rows: list[dict], run_name: str, out_path: Path) -> Path:
    """Render the two-panel performance figure and save it to ``out_path``."""
    gens = [r.get("generation", i + 1) for i, r in enumerate(rows)]
    score = [r.get("heldout_split_repair_score", 0) for r in rows]
    accepted = [bool(r.get("accepted")) for r in rows]
    parent_bar = _running_parent_bar(rows)

    # Cumulative cost. NOTE the two ledger fields differ in kind:
    #   * wall_seconds is PER-GENERATION (time.monotonic() - gen_wall0, reset each
    #     gen), so we cumsum it to get cumulative wall-clock.
    #   * cost_usd is ALREADY CUMULATIVE: it is ResultMessage.total_cost_usd from the
    #     ONE persistent ClaudeSDKClient session shared across all generations, i.e.
    #     the running session total — so we use it directly (cumsum would double-count).
    cum_min = np.cumsum([r.get("wall_seconds", 0.0) for r in rows]) / 60.0
    cum_usd = [r.get("cost_usd", 0.0) for r in rows]

    acc_color = ["#2ca02c" if a else "#d62728" for a in accepted]
    splits_only = any(r.get("splits_only") for r in rows)
    mode = " [splits-only]" if splits_only else ""

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax0, ax3 = axes

    # (1) Split-repair score + parent bar + accept/reject markers.
    ax0.plot(gens, score, "-", color="#1f77b4", lw=1.5, zorder=1, label="candidate score")
    ax0.step(gens, parent_bar, where="mid", color="#888", lw=1.2, ls="--",
             zorder=1, label="parent bar (to beat)")
    for g, s, c, a in zip(gens, score, acc_color, accepted):
        ax0.scatter([g], [s], c=c, s=70, marker="o" if a else "X",
                    edgecolors="k", linewidths=0.5, zorder=3)
    ax0.set_ylabel("split-repair score\n(correct − false)")
    ax0.set_title(f"Run {run_name}{mode} — performance across {len(rows)} generations\n"
                  f"gate metric: held-out split-repair score "
                  f"(● accepted = new parent, ✕ rejected)")
    ax0.grid(True, alpha=0.3)
    # Legend includes the accept/reject marker meaning.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#1f77b4", lw=1.5, label="candidate score"),
        Line2D([0], [0], color="#888", lw=1.2, ls="--", label="parent bar (to beat)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="k", markersize=9, label="accepted"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#d62728",
               markeredgecolor="k", markersize=9, label="rejected"),
    ]
    ax0.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    # (2) Cumulative cost (twin axis: minutes + $).
    ax3.plot(gens, cum_min, "-o", color="#ff7f0e", ms=4, lw=1.2, label="cum. wall (min)")
    ax3.set_ylabel("cumulative\nwall-clock (min)", color="#ff7f0e")
    ax3.tick_params(axis="y", labelcolor="#ff7f0e")
    ax3.grid(True, alpha=0.3)
    ax3b = ax3.twinx()
    ax3b.plot(gens, cum_usd, "-s", color="#17becf", ms=4, lw=1.2, label="cum. $")
    ax3b.set_ylabel("cumulative\nagent cost ($)", color="#17becf")
    ax3b.tick_params(axis="y", labelcolor="#17becf")
    ax3.set_xlabel("generation")
    ax3.set_xticks(gens)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_csv(rows: list[dict], parent_bar: list[float], path: Path) -> Path:
    """Dump the plotted series to CSV for spreadsheets / further analysis."""
    # Column names mirror the ledger, EXCEPT cost is renamed to make its kind
    # explicit: wall_seconds is per-generation; cost_usd_cumulative is the running
    # session total (already cumulative in the ledger).
    cols = ["generation", "accepted", "heldout_split_repair_score", "parent_bar",
            "heldout_correct_merges", "heldout_false_merges", "heldout_n_edits",
            "wall_seconds", "cost_usd_cumulative", "splits_only"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r, bar in zip(rows, parent_bar):
            w.writerow([r.get("generation"), r.get("accepted"),
                        r.get("heldout_split_repair_score"), bar,
                        r.get("heldout_correct_merges"), r.get("heldout_false_merges"),
                        r.get("heldout_n_edits"), r.get("wall_seconds"),
                        r.get("cost_usd"), r.get("splits_only")])
    return path


def summarize(rows: list[dict]) -> str:
    """One-line text summary printed to stdout alongside the figure."""
    if not rows:
        return "no generations in ledger."
    accepted = [r for r in rows if r.get("accepted")]
    scores = [r.get("heldout_split_repair_score", 0) for r in rows]
    best = max(scores) if scores else 0
    first = scores[0] if scores else 0
    total_min = sum(r.get("wall_seconds", 0.0) for r in rows) / 60.0
    # cost_usd is the running SESSION total (cumulative), so the run total is the
    # LAST/MAX value, NOT a sum over generations.
    total_usd = max((r.get("cost_usd", 0.0) for r in rows), default=0.0)
    return (f"{len(rows)} generations, {len(accepted)} accepted. "
            f"split-repair score {first} -> best {best}. "
            f"{total_min:.0f} min wall, ${total_usd:.2f} agent cost (cumulative).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_name",
                        help="run directory name under proofreader_evolve/runs/ "
                             "(or a full path to a run dir)")
    parser.add_argument("--out", default=None,
                        help="output PNG path (default: runs/<run>/performance.png)")
    parser.add_argument("--csv", default=None,
                        help="also write the plotted series to this CSV path")
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_name)
    rows = load_ledger(run_dir)
    if not rows:
        print(f"[plot_run_performance] {run_dir/'ledger.jsonl'} has no generations.")
        return 1

    out_path = Path(args.out) if args.out else (run_dir / "performance.png")
    make_figure(rows, run_dir.name, out_path)
    print(f"[plot_run_performance] {summarize(rows)}")
    print(f"[plot_run_performance] figure -> {out_path}")

    if args.csv:
        csv_path = write_csv(rows, _running_parent_bar(rows), Path(args.csv))
        print(f"[plot_run_performance] series -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
