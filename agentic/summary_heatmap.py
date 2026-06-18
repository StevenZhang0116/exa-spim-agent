"""
Render a 0–1 quality heatmap from a workflow summary Markdown report.

Reads one ``<RUN>.summary.md`` (the deliverable the summarize → rerun →
extrapolate → verify workflow writes) and turns three per-hypothesis verdicts
into a heatmap: one ROW per hypothesis (y-axis = the hypothesis's actual record
``ID``, e.g. 47, 3, 10 …) and three COLUMNS across the top — Reproduction,
Generalization, Statistics — scored 0 (worst: not reproduced / not generalized /
major statistical issue) to 1 (best).

The report is parsed deterministically (no model): each ``### N.`` entry block is
scanned for the bullets the agents fold in:

* ``- **Reproduction:** REPRODUCED | DIVERGED | FAILED``
* ``- **Generalization:** GENERALIZES | PARTIAL | DOES-NOT-GENERALIZE | INCONCLUSIVE``
* ``- **Verdict:** OK | MINOR | MAJOR``  (or SOUND | WEAK | FLAWED)

Missing/unrecognized verdicts render as a distinct "n/a" cell (NaN), not 0, so
"absent" is never confused with "bad".

Usage (from the ``exa-spim-agent/`` project root)
-------------------------------------------------
    python agentic/summary_heatmap.py autodiscovery/RUN.summary.md
    python agentic/summary_heatmap.py autodiscovery/RUN.summary.md --out heat.png
    python agentic/summary_heatmap.py autodiscovery/RUN.summary.md --csv scores.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Each metric maps a verdict word -> score in [0, 1] (0 worst, 1 best). Words are
# matched case-insensitively; anything not listed (or absent) becomes NaN ("n/a").
REPRODUCTION = {"reproduced": 1.0, "diverged": 0.5, "failed": 0.0}
GENERALIZATION = {
    "generalizes": 1.0,
    "partial": 0.5,
    "does-not-generalize": 0.0,
    "inconclusive": float("nan"),
}
# Statistics quality comes from the verifier's Verdict bullet. Two vocabularies
# appear in practice: OK/MINOR/MAJOR (severity of issues) and SOUND/WEAK/FLAWED.
STATISTICS = {
    "ok": 1.0,
    "sound": 1.0,
    "minor": 0.66,
    "weak": 0.5,
    "major": 0.33,
    "flawed": 0.0,
}

ROWS = [
    ("Reproduction", "Reproduction", REPRODUCTION),
    ("Generalization", "Generalization", GENERALIZATION),
    ("Statistics", "Verdict", STATISTICS),
]

# ``### 12. (Priority …) Title`` -> capture the leading rank number.
_HEADER = re.compile(r"^###\s+(\d+)\.\s", re.MULTILINE)
# ``- **Run:** … · **ID:** 47 · …`` -> the hypothesis's actual record ID.
_ID = re.compile(r"\*\*ID:\*\*\s*([0-9A-Za-z_-]+)")


def _bullet_value(block: str, label: str) -> str | None:
    """First ALL-CAPS-ish token after ``- **<label>:**`` in an entry block."""
    m = re.search(
        rf"^\s*-\s*\*\*{re.escape(label)}:\*\*\s*([A-Za-z][A-Za-z-]*)",
        block,
        re.MULTILINE,
    )
    return m.group(1) if m else None


def parse_entries(text: str) -> list[dict]:
    """Split the report into ``### N.`` blocks and score each metric.

    Only numbered hypothesis entries are kept; trailing summary sections
    (``### Findings that did NOT reproduce`` …) have no number and are skipped.
    """
    matches = list(_HEADER.finditer(text))
    entries = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        num = int(m.group(1))  # rank within the report (### N.)
        id_m = _ID.search(block)
        hid = id_m.group(1) if id_m else str(num)  # actual record ID; fall back to rank
        title = text[m.end() : text.find("\n", m.end())].strip()

        scores, labels = {}, {}
        for key, bullet_label, mapping in ROWS:
            raw = _bullet_value(block, bullet_label)
            labels[key] = raw
            scores[key] = mapping.get(raw.lower(), float("nan")) if raw else float("nan")
        entries.append(
            {"num": num, "id": hid, "title": title, "scores": scores, "labels": labels}
        )
    entries.sort(key=lambda e: e["num"])  # keep report (priority) order
    return entries


def build_matrix(entries: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """(rows × hypotheses) score matrix, row names, and hypothesis ID labels."""
    ids = [e["id"] for e in entries]
    row_names = [name for name, _, _ in ROWS]
    mat = np.full((len(ROWS), len(entries)), np.nan)
    for j, e in enumerate(entries):
        for i, name in enumerate(row_names):
            mat[i, j] = e["scores"][name]
    return mat, row_names, ids


def render(mat, row_names, ids, out_path: Path, title: str) -> None:
    """Draw the heatmap (green=1 best … red=0 worst, grey=n/a) and save it.

    Vertical layout: one ROW per hypothesis (y-axis = actual record ID) and one
    COLUMN per metric (Reproduction / Generalization / Statistics) across the
    top. ``mat`` is (metrics × hypotheses); we transpose it to (hypotheses ×
    metrics) for this orientation.
    """
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d9d9d9")  # NaN ("n/a") cells render grey

    matT = mat.T  # (hypotheses × metrics)
    fig_h = max(3.0, 0.40 * len(ids) + 1.5)
    fig, ax = plt.subplots(figsize=(5.0, fig_h))
    im = ax.imshow(
        np.ma.masked_invalid(matT), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto"
    )

    # Metrics across the top.
    ax.set_xticks(range(len(row_names)))
    ax.set_xticklabels(row_names, fontsize=9)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    # Hypothesis IDs down the side.
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([str(i) for i in ids], fontsize=8)
    ax.set_ylabel("Hypothesis ID")

    # Annotate each cell with its score (or "n/a") for legibility.
    for i in range(matT.shape[0]):
        for j in range(matT.shape[1]):
            v = matT[i, j]
            label = "n/a" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black")

    ax.set_title(title, pad=24)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("0 = worst   →   1 = best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(entries: list[dict], csv_path: Path) -> None:
    """Dump the scores + raw verdict labels to CSV alongside the image."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "hypothesis_id",
                "rank",
                "reproduction_score",
                "reproduction_label",
                "generalization_score",
                "generalization_label",
                "statistics_score",
                "statistics_label",
                "title",
            ]
        )
        for e in entries:
            s, lab = e["scores"], e["labels"]
            w.writerow(
                [
                    e["id"],
                    e["num"],
                    s["Reproduction"],
                    lab["Reproduction"] or "",
                    s["Generalization"],
                    lab["Generalization"] or "",
                    s["Statistics"],
                    lab["Statistics"] or "",
                    e["title"],
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary", type=Path, help="The <RUN>.summary.md report to read."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Heatmap image path. Default: <summary stem>.heatmap.png next to it.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV of the underlying scores/labels.",
    )
    args = parser.parse_args(argv)

    if not args.summary.is_file():
        parser.error(f"No such summary report: {args.summary}")
    text = args.summary.read_text(encoding="utf-8")
    entries = parse_entries(text)
    if not entries:
        parser.error(f"No '### N.' hypothesis entries found in {args.summary}")

    out_path = args.out or args.summary.with_suffix(".heatmap.png")
    mat, row_names, ids = build_matrix(entries)
    render(mat, row_names, ids, out_path, title=args.summary.stem)
    if args.csv is not None:
        write_csv(entries, args.csv)

    # Coverage log to stderr so missing verdicts are visible.
    for name, _, _ in ROWS:
        n_missing = sum(1 for e in entries if np.isnan(e["scores"][name]))
        print(
            f"[heatmap] {name}: {len(entries) - n_missing}/{len(entries)} scored"
            + (f", {n_missing} n/a" if n_missing else ""),
            file=sys.stderr,
        )
    print(out_path)
    if args.csv is not None:
        print(args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
