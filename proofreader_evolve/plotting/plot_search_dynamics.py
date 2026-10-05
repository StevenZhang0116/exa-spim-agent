"""Plot scorer decisions and policy-matched TRAIN/development-validation precision.

Missing evaluations are not zero; repeated validation is not final testing.
"""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from proofreader_evolve.harness import run_records as records
from proofreader_evolve.plotting.plot_run_performance import (
    load_ledger, _resolve_run_dir, _floats, plot_candidates, plt,
)


def _outcome_reason(row):
    return ("accepted" if row.get("accepted") else "rejected", records.decision_reason(row))


def make_figure(rows, run_name, out_path, run_dir=None):
    records.run_objective(run_dir, rows)
    pairs = records.paired_measurements(rows, run_dir)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    plot_candidates(axes[0], rows, records.METRIC_NAME)
    if pairs:
        gens = [p["generation"] for p in pairs]
        axes[1].plot(gens, _floats([p["train"] for p in pairs]), "-o", label="TRAIN, same policy")
        axes[1].plot(gens, _floats([p["validation"] for p in pairs]), "-s", label="validation, same policy")
        axes[1].legend()
    else:
        axes[1].text(.5, .5, "No policy-matched train/validation measurements",
                     ha="center", transform=axes[1].transAxes)
    axes[1].set_ylabel(records.METRIC_NAME)
    axes[1].set_xlabel("policy generation")
    axes[1].set_title("Different data scopes: raw differences are not proof of generalization")
    axes[1].grid(alpha=.3)
    fig.suptitle(f"{run_name} — development search, not final testing")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_md(rows, run_name, run_dir, path):
    records.run_objective(run_dir, rows)
    lines = [f"# Search digest — {run_name}", "", f"Metric: {records.METRIC_NAME}", "",
             "Missing validation is not a zero score. Reasons are recorded, not inferred.", "",
             "| Generation | Outcome | Validation | Reason |", "| --- | --- | --- | --- |"]
    for row in rows:
        outcome, reason = _outcome_reason(row)
        value = records.validation_value(row)
        reason = reason.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['generation']} | {outcome} | {value if value is not None else 'not measured'} | {reason} |")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--out")
    parser.add_argument("--md")
    args = parser.parse_args(argv)
    run_dir = _resolve_run_dir(args.run_name)
    rows = load_ledger(run_dir)
    if not rows:
        print("No generations; consult seed.json for the baseline.")
        return 0
    make_figure(rows, run_dir.name, Path(args.out) if args.out else run_dir / "search_dynamics.png", run_dir)
    if args.md:
        write_md(rows, run_dir.name, run_dir, Path(args.md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
