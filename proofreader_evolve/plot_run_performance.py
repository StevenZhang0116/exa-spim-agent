"""
Plot one evolution run's performance across generations from its ledger.

Reads ``runs/<run_name>/ledger.jsonl`` (one JSON row per generation, written by
``harness.ledger.GenerationCost``) and renders a three-panel figure tracking how
the policy improved generation over generation:

  (1) Penalized FITNESS (THE GATE METRIC) per generation, with the running PARENT
      bar (best accepted-so-far) overlaid as a step line, and each generation
      marked accepted (green ●) or rejected (red ✕). Fitness = split-repair score
      − merge_penalty × false merges, so a false-merge gen dives sharply below its
      raw score. This is the panel that answers "did it get better, which gens were
      kept" — the accept/reject markers live here because this is what the gate
      compares.
  (2) The decomposition behind the fitness: the raw split-repair score
      (correct − false) and the false-merge count that the penalty acts on.
  (3) Cumulative cost (wall-clock minutes and agent $), so the accuracy gain can
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


def _row_fitness(r: dict) -> float:
    """This generation's penalized FITNESS (the gate's decision variable).

    Prefer the recorded ``heldout_fitness``; fall back to reconstructing it from the
    raw score and false-merge count for OLD ledgers written before the field existed
    (default merge_penalty=100). Pre-smoothing runs used a hard 'false==0' gate, so
    their reconstructed fitness is a faithful post-hoc view: a rejected false-merge
    gen dives, exactly as the new gate would score it.
    """
    if "heldout_fitness" in r:
        return float(r["heldout_fitness"])
    penalty = float(r.get("merge_penalty", 100.0))
    score = r.get("heldout_split_repair_score", 0)
    false = r.get("heldout_false_merges", 0)
    return float(score) - penalty * float(false)


def _cumulative_cost(rows: list[dict]) -> list[float]:
    """Cumulative agent $ across generations, robust to BOTH ledger conventions.

    ``cost_usd`` has meant two different things over this project's history:
      * PER-GENERATION (current): each gen opens a FRESH ClaudeSDKClient, so
        ``ResultMessage.total_cost_usd`` is that gen's OWN session total — it rises
        and falls with how much work the gen did. These must be cumsum'd.
      * ALREADY-CUMULATIVE (old runs): one persistent session was shared across the
        whole loop, so ``cost_usd`` was the running session total and already
        monotonic. Cumsumming those would double-count.

    We disambiguate from the data alone: an already-cumulative series is
    non-decreasing; a per-gen series dips. So if the raw series is non-decreasing we
    take it as-is, otherwise we cumsum. (Single-gen runs coincide either way.)
    """
    raw = [float(r.get("cost_usd", 0.0) or 0.0) for r in rows]
    already_cumulative = all(b >= a for a, b in zip(raw, raw[1:]))
    return raw if already_cumulative else list(np.cumsum(raw))


def _running_parent_bar(rows: list[dict], key=None) -> list[float]:
    """The bar each generation had to BEAT = best accepted value so far.

    ``key`` maps a row to the value being tracked; default is the penalized fitness
    (the actual gate variable). The seed's value is the bar gen 1 faces; thereafter
    the bar advances only on an accepted generation (the gate is parent-relative). We
    reconstruct it from the ledger alone: start at the first generation's
    parent-implied bar, and step up whenever a generation is accepted.
    """
    if key is None:
        key = _row_fitness
    bar = []
    best = None
    for r in rows:
        val = key(r)
        # The bar in force WHEN this gen was judged is the best accepted BEFORE it.
        bar.append(best if best is not None else val)
        if r.get("accepted"):
            best = val if best is None else max(best, val)
    return bar


def make_figure(rows: list[dict], run_name: str, out_path: Path) -> Path:
    """Render the three-panel performance figure and save it to ``out_path``."""
    gens = [r.get("generation", i + 1) for i, r in enumerate(rows)]
    score = [r.get("heldout_split_repair_score", 0) for r in rows]
    fitness = [_row_fitness(r) for r in rows]
    false_merges = [r.get("heldout_false_merges", 0) for r in rows]
    accepted = [bool(r.get("accepted")) for r in rows]
    fitness_bar = _running_parent_bar(rows, key=_row_fitness)
    # The penalty in force (last row's; constant within a run). For the title only.
    penalty = float(rows[-1].get("merge_penalty", 100.0)) if rows else 100.0

    # Cumulative cost. BOTH series are per-generation in kind now, so both cumsum:
    #   * wall_seconds is PER-GENERATION (time.monotonic() - gen_wall0, reset each gen).
    #   * cost_usd is PER-GENERATION too: each gen opens a FRESH ClaudeSDKClient, so
    #     ResultMessage.total_cost_usd is that gen's own session total (it dips when a
    #     gen did less work). _cumulative_cost() cumsums it — while still handling OLD
    #     ledgers whose cost_usd was already the running total of one shared session.
    cum_min = np.cumsum([r.get("wall_seconds", 0.0) for r in rows]) / 60.0
    cum_usd = _cumulative_cost(rows)

    acc_color = ["#2ca02c" if a else "#d62728" for a in accepted]
    splits_only = any(r.get("splits_only") for r in rows)
    mode = " [splits-only]" if splits_only else ""

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    ax0, ax1, ax3 = axes

    # (1) Penalized FITNESS (the gate metric) + parent bar + accept/reject markers.
    ax0.plot(gens, fitness, "-", color="#1f77b4", lw=1.5, zorder=1,
             label="candidate fitness")
    ax0.step(gens, fitness_bar, where="mid", color="#888", lw=1.2, ls="--",
             zorder=1, label="parent bar (to beat)")
    for g, fval, c, a in zip(gens, fitness, acc_color, accepted):
        ax0.scatter([g], [fval], c=c, s=70, marker="o" if a else "X",
                    edgecolors="k", linewidths=0.5, zorder=3)
    ax0.axhline(0, color="k", lw=0.6, alpha=0.4, zorder=0)
    ax0.set_ylabel(f"penalized fitness\n(score − {penalty:g}×false)")
    ax0.set_title(f"Run {run_name}{mode} — performance across {len(rows)} generations\n"
                  f"gate metric: held-out penalized fitness "
                  f"(merge_penalty={penalty:g}; ● accepted = new parent, ✕ rejected)")
    ax0.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#1f77b4", lw=1.5, label="candidate fitness"),
        Line2D([0], [0], color="#888", lw=1.2, ls="--", label="parent bar (to beat)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="k", markersize=9, label="accepted"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#d62728",
               markeredgecolor="k", markersize=9, label="rejected"),
    ]
    ax0.legend(handles=handles, loc="best", fontsize=8, framealpha=0.9)

    # (2) Decomposition: raw split-repair score (line) + false-merge count (bars).
    # This shows WHY fitness dips — a bar of false merges is what the penalty acts on.
    ax1.plot(gens, score, "-o", color="#9467bd", ms=4, lw=1.3,
             label="split-repair score (correct − false)")
    ax1.set_ylabel("split-repair score", color="#9467bd")
    ax1.tick_params(axis="y", labelcolor="#9467bd")
    ax1.grid(True, alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.bar(gens, false_merges, width=0.6, color="#d62728", alpha=0.35,
             label="false merges", zorder=0)
    ax1b.set_ylabel("false merges", color="#d62728")
    ax1b.tick_params(axis="y", labelcolor="#d62728")
    # Integer ticks for the (small) false-merge count.
    _fmax = max(false_merges) if false_merges else 0
    ax1b.set_ylim(0, max(1, _fmax) * 1.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, framealpha=0.9)

    # (3) Cumulative cost (twin axis: minutes + $).
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
    """Dump the plotted series to CSV for spreadsheets / further analysis.

    ``parent_bar`` is the FITNESS bar (the gate variable). Column names mirror the
    ledger, EXCEPT cost is renamed to make its kind explicit: wall_seconds is
    per-generation; cost_usd_cumulative is the running total we compute here (cost_usd
    is per-generation now, so we cumsum it — see _cumulative_cost).
    """
    cols = ["generation", "accepted", "heldout_fitness", "parent_fitness_bar",
            "merge_penalty", "heldout_split_repair_score",
            "heldout_correct_merges", "heldout_false_merges", "heldout_n_edits",
            "wall_seconds", "cost_usd_cumulative", "splits_only"]
    cum_usd = _cumulative_cost(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r, bar, cum in zip(rows, parent_bar, cum_usd):
            w.writerow([r.get("generation"), r.get("accepted"),
                        _row_fitness(r), bar, r.get("merge_penalty", 100.0),
                        r.get("heldout_split_repair_score"),
                        r.get("heldout_correct_merges"), r.get("heldout_false_merges"),
                        r.get("heldout_n_edits"), r.get("wall_seconds"),
                        cum, r.get("splits_only")])
    return path


def summarize(rows: list[dict]) -> str:
    """One-line text summary printed to stdout alongside the figure."""
    if not rows:
        return "no generations in ledger."
    accepted = [r for r in rows if r.get("accepted")]
    fits = [_row_fitness(r) for r in rows]
    best = max(fits) if fits else 0
    first = fits[0] if fits else 0
    # Also report the raw split-repair score frontier as a secondary number.
    scores = [r.get("heldout_split_repair_score", 0) for r in rows]
    best_score = max(scores) if scores else 0
    total_min = sum(r.get("wall_seconds", 0.0) for r in rows) / 60.0
    # cost_usd is per-generation (fresh session each gen), so the run total is the
    # cumulative series' last value. _cumulative_cost() also handles old ledgers
    # whose cost_usd was already the running total of one shared session.
    cum_usd = _cumulative_cost(rows)
    total_usd = cum_usd[-1] if cum_usd else 0.0
    return (f"{len(rows)} generations, {len(accepted)} accepted. "
            f"fitness {first:g} -> best {best:g} (best split-repair score {best_score}). "
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
