"""
Render a 0–1 quality heatmap from a workflow summary Markdown report.

Reads one ``<RUN>.summary.md`` (the deliverable the summarize → rerun →
extrapolate → verify → fix-tests workflow writes) and turns the per-hypothesis
verdicts into a heatmap: one ROW per hypothesis (y-axis = the hypothesis's actual
record ``ID``, e.g. 47, 3, 10 …) and one COLUMN per metric across the top, each
scored 0 (worst) to 1 (best).

The report is parsed deterministically (no model): each ``### N.`` entry block is
scanned for the bullets the agents fold in (the verdict must be the LEADING word
after the label):

* ``- **Reproduction:** REPRODUCED | DIVERGED | FAILED``
* ``- **Generalization:** GENERALIZES | PARTIAL | DOES-NOT-GENERALIZE | INCONCLUSIVE``
* ``- **Verdict:** OK | MINOR | MAJOR | CRITICAL``  (or SOUND | WEAK | FLAWED)  → "Statistics"
* ``- **Post-correction verdict:** UPHELD | WEAKENED | OVERTURNED`` (fix-tests)
* ``- **Corrected generalization:** GENERALIZES | PARTIAL | DOES-NOT-GENERALIZE``
  (fix-tests, only when extra datasets were run)

Missing/unrecognized verdicts render as a distinct "n/a" cell (NaN), not 0, so
"absent" is never confused with "bad". The last two columns are typically n/a
for hypotheses the verifier did not flag for a test fix.

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
# appear in practice: OK/MINOR/MAJOR/CRITICAL (severity of issues) and
# SOUND/WEAK/FLAWED. CRITICAL is the most severe issue tier (worse than MAJOR),
# so it maps to 0.0 — without it CRITICAL entries would render as a neutral "n/a"
# cell, hiding the worst findings rather than flagging them.
STATISTICS = {
    "ok": 1.0,
    "sound": 1.0,
    "minor": 0.66,
    "weak": 0.5,
    "major": 0.33,
    "critical": 0.0,
    "flawed": 0.0,
}
# fix-tests outcome for hypotheses whose statistical test was corrected and
# re-measured. Records that were NOT fixed have no bullet -> NaN ("n/a").
POST_CORRECTION = {"upheld": 1.0, "weakened": 0.5, "overturned": 0.0}
# Generalization re-judged with the CORRECTED test (same vocabulary as the
# original Generalization row). Only present on fixed records run with extras.
CORRECTED_GENERALIZATION = GENERALIZATION

ROWS = [
    ("Reproduction", "Reproduction", REPRODUCTION),
    ("Generalization", "Generalization", GENERALIZATION),
    ("Statistics", "Verdict", STATISTICS),
    ("Post-correction", "Post-correction verdict", POST_CORRECTION),
    ("Corrected gen.", "Corrected generalization", CORRECTED_GENERALIZATION),
]

# ``### 12. (Priority …) Title`` -> capture the leading rank number.
_HEADER = re.compile(r"^###\s+(\d+)\.\s", re.MULTILINE)
# ``- **Run:** … · **ID:** 47 · …`` -> the hypothesis's actual record ID.
_ID = re.compile(r"\*\*ID:\*\*\s*([0-9A-Za-z_-]+)")


def _metric_value(block: str, label: str, mapping: dict) -> str | None:
    """Resolve a metric's verdict word for ``label`` within an entry block.

    The verdict MUST be the first recognized word immediately after the label
    (``- **<label>:** VERDICT …`` or the inline ``<label>: VERDICT …`` form),
    optionally preceded by short filler like "the"/"is". We deliberately do NOT
    scan deep into free text for a keyword: a sentence like "the qualitative
    claim GENERALIZES but the magnitude does NOT" must be written with a leading
    verdict word (PARTIAL) to be scored, otherwise it stays n/a rather than being
    mis-scored on the first keyword that happens to appear.
    """
    m = re.search(rf"\*\*\s*{re.escape(label)}\s*:?\s*\*\*\s*", block)
    if not m:
        m = re.search(rf"{re.escape(label)}\s*:\s*", block)  # inline sub-sentence
        if not m:
            return None
    # Only look at the first few words right after the label.
    tail = re.split(r"\*\*", block[m.end() : m.end() + 80], maxsplit=1)[0]
    words = re.findall(r"[A-Za-z][A-Za-z-]*", tail)
    for word in words[:3]:  # leading verdict, allowing brief filler
        if word.lower() in mapping:
            return word
    return None


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
            raw = _metric_value(block, bullet_label, mapping)
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
    COLUMN per metric (Reproduction / Generalization / Statistics /
    Post-correction / Corrected gen.) across the top. ``mat`` is
    (metrics × hypotheses); we transpose it to (hypotheses × metrics).
    """
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#eceff1")  # NaN ("n/a") cells render a soft neutral grey

    matT = mat.T  # (hypotheses × metrics)
    fig_h = max(3.0, 0.42 * len(ids) + 1.8)
    fig_w = max(5.0, 1.05 * len(row_names) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(matT)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Thin white gridlines between cells for a cleaner, tile-like look.
    ax.set_xticks(np.arange(-0.5, len(row_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ids), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Metrics across the top (rotated so longer labels don't overlap).
    ax.set_xticks(range(len(row_names)))
    ax.set_xticklabels(row_names, fontsize=8.5, rotation=30, ha="left")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    # Hypothesis IDs down the side.
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([str(i) for i in ids], fontsize=8)
    ax.set_ylabel("Hypothesis ID", fontsize=10)
    ax.tick_params(axis="y", length=0)

    # Annotate each cell with its score (or "n/a"); white text on dark cells,
    # dark text on light cells, faint grey on n/a — for legibility on any hue.
    for i in range(matT.shape[0]):
        for j in range(matT.shape[1]):
            v = matT[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5,
                        color="#90a4ae", style="italic")
                continue
            txt_color = "white" if (v <= 0.18 or v >= 0.82) else "#222222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=txt_color, fontweight="medium")

    ax.set_title(title, pad=26, fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("0 = worst   →   1 = best", fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _csv_key(name: str) -> str:
    """A CSV-safe column prefix from a metric name, e.g. 'Corrected gen.' -> 'corrected_gen'."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def write_csv(entries: list[dict], csv_path: Path) -> None:
    """Dump the scores + raw verdict labels to CSV alongside the image.

    Generic over ROWS so new metrics (Post-correction, Corrected gen.) appear
    automatically as ``<key>_score`` / ``<key>_label`` column pairs.
    """
    metric_names = [name for name, _, _ in ROWS]
    header = ["hypothesis_id", "rank"]
    for name in metric_names:
        k = _csv_key(name)
        header += [f"{k}_score", f"{k}_label"]
    header.append("title")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for e in entries:
            row = [e["id"], e["num"]]
            for name in metric_names:
                row += [e["scores"][name], e["labels"][name] or ""]
            row.append(e["title"])
            w.writerow(row)


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
