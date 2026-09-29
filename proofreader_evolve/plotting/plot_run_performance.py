"""Plot fixed-pool Precision@K, recorded parent bars and per-generation costs.

Missing evaluations remain gaps, never zero. No LLM or brain data is loaded.
"""

import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proofreader_evolve.harness import run_records as records

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
load_ledger = records.load_ledger


def _resolve_run_dir(run_name):
    for path in (Path(run_name), RUNS_DIR / run_name):
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Run not found: {run_name}")


def _floats(values):
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def plot_candidates(ax, rows, metric_name):
    gens = [r["generation"] for r in rows]
    values = _floats([records.validation_value(r) for r in rows])
    ax.plot(gens, values, color="tab:blue", label="candidate validation")
    ax.step(gens, _floats(records.parent_values(rows)), where="mid",
            linestyle="--", color="gray", label="recorded parent bar")
    for gen, value, row in zip(gens, values, rows):
        if np.isfinite(value):
            ax.scatter(gen, value, color="tab:green" if row.get("accepted") else "tab:red",
                       marker="o" if row.get("accepted") else "x", zorder=3)
        else:
            ax.plot(gen, .04, "|", color="tab:red", transform=ax.get_xaxis_transform())
    ax.set_ylabel(metric_name)
    ax.set_title("Green: accepted; red: rejected; bottom tick: validation not measured")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)


def make_figure(rows, run_name, out_path):
    version = records.run_objective(rows=rows)
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    gens = [r["generation"] for r in rows]
    plot_candidates(axes[0], rows, records.METRIC_NAME)
    fig.suptitle(f"{run_name} — {version}\nDevelopment validation, not an untouched final test")
    cells = sorted({k for r in rows for k in (r.get("validation") or {}).get("cells", {})})
    for cell in cells:
        values = [(r.get("validation") or {}).get("cells", {}).get(cell, {}).get("precision") for r in rows]
        axes[1].plot(gens, _floats(values), "-o", label=cell)
    axes[1].set_ylabel("Precision@K by brain/kind")
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.3)
    costs = records.cumulative_accounting(rows)
    axes[2].plot(gens, _floats(costs), "-o")
    if any(v is None for v in costs):
        axes[2].text(.02, .92, "Some costs unknown: missing recorded amount",
                     transform=axes[2].transAxes, fontsize=8)
    axes[2].set_ylabel("Cumulative agent cost ($)")
    other = axes[2].twinx()
    other.plot(gens, np.cumsum(_floats([r.get("wall_seconds") for r in rows])) / 60,
               "--", color="tab:orange")
    other.set_ylabel("Cumulative wall time (minutes)")
    axes[2].set_xlabel("generation")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_csv(rows, parent_bar, path):
    version = records.run_objective(rows=rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["generation", "objective", "accepted", "validation_value",
                         "parent_value", "decision_reason", "cost_usd_cumulative"])
        for row, parent, cost in zip(rows, parent_bar, records.cumulative_accounting(rows)):
            writer.writerow([row["generation"], version, row.get("accepted"),
                             records.validation_value(row), parent, records.decision_reason(row), cost])
    return path


def summarize(rows):
    if not rows:
        return "No generations recorded."
    version = records.run_objective(rows=rows)
    accepted = [records.validation_value(r) for r in rows if r.get("accepted")]
    parents = records.parent_values(rows)
    final = accepted[-1] if accepted else parents[0]
    cost = records.cumulative_accounting(rows)[-1]
    return (f"{len(rows)} generations, {len(accepted)} accepted; {records.METRIC_NAME}: "
            f"seed={parents[0]}, final accepted={final}; total cost={cost if cost is not None else 'unknown'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--out")
    parser.add_argument("--csv")
    args = parser.parse_args(argv)
    run_dir = _resolve_run_dir(args.run_name)
    rows = load_ledger(run_dir)
    if not rows:
        print("No generations; consult seed.json for the baseline.")
        return 0
    make_figure(rows, run_dir.name, Path(args.out) if args.out else run_dir / "performance.png")
    if args.csv:
        write_csv(rows, records.parent_values(rows), Path(args.csv))
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
