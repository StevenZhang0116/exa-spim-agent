"""Generate agentic/discovery_workflow_diagram.png (the discovery workflow figure).

The SVG is built here with measured text fits, then rasterized with resvg
(``pip install --target <dir> resvg-py``, pure wheel, no system libraries; add
<dir> to PYTHONPATH). Text widths are measured with DejaVu Sans, which a Linux
browser and resvg use for the sans-serif fallback and which is wider than
Helvetica/Arial, so a fit here fits everywhere.

Usage (from the project root):
    PYTHONPATH=<dir> python scripts/diagrams/make_discovery_diagram.py \
        agentic/discovery_workflow_diagram.png
"""
import os
import sys
from pathlib import Path

import matplotlib
from PIL import ImageFont

FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONTS = {}


def width(text, size, bold=False):
    key = (size, bold)
    if key not in FONTS:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        FONTS[key] = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    return FONTS[key].getlength(text)


BLUE = ("#dbeafe", "#3b82f6", "#1e3a8a")     # agent judgment
GREEN = ("#dcfce7", "#16a34a", "#166534")    # driver compute
GREY = "#475569"
MUTED = "#64748b"
W, H = 1920, 1150
PAD = 14
out = []
problems = []


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _check(t, size, bold, max_w):
    if max_w is not None:
        w = width(t, size, bold)
        if w > max_w:
            problems.append(f"{t!r} at {size}px is {w:.0f}px > {max_w:.0f}px")


def text(x, y, t, size, fill, bold=False, anchor="start", max_w=None):
    _check(t, size, bold, max_w)
    weight = ' font-weight="700"' if bold else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    out.append(f'  <text x="{x}" y="{y}" font-size="{size}"{weight}{a} fill="{fill}">{esc(t)}</text>')


def semibold(x, y, t, size, fill, anchor="start", max_w=None):
    _check(t, size, True, max_w)
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    out.append(f'  <text x="{x}" y="{y}" font-size="{size}" font-weight="600"{a} fill="{fill}">{esc(t)}</text>')


def box(x, y, w, h, kind, lines, rx=12, center=False):
    """A colored box: a semibold title line, plain lines below; all must fit."""
    fill, stroke, title_fill = kind
    out.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    ty = y + 32
    for i, (t, size) in enumerate(lines):
        if ty + 4 > y + h:
            problems.append(f"{t!r} runs past the bottom of its box ({ty} > {y + h})")
        if center:
            cx = x + w / 2
            if i == 0:
                semibold(cx, ty, t, size, title_fill, anchor="middle", max_w=w - 2 * PAD)
            else:
                text(cx, ty, t, size, GREY, anchor="middle", max_w=w - 2 * PAD)
        elif i == 0:
            semibold(x + PAD + 4, ty, t, size, title_fill, max_w=w - 2 * PAD - 4)
        else:
            text(x + PAD + 4, ty, t, size, GREY, max_w=w - 2 * PAD - 4)
        ty += size + 9


def arrow(x1, y1, x2, y2, sw=3):
    out.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#64748b" stroke-width="{sw}" marker-end="url(#arrow)"/>')


def card(x, y, w, h, n, title, dashed=False):
    dash = ' stroke-dasharray="8 5"' if dashed else ""
    stroke = "#94a3b8" if dashed else "#cbd5e1"
    out.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="#ffffff" stroke="{stroke}" stroke-width="2"{dash} filter="url(#shadow)"/>')
    out.append(f'  <circle cx="{x + 44}" cy="{y + 48}" r="24" fill="#0f172a"/>')
    text(x + 44, y + 56, str(n), 22, "#ffffff", bold=True, anchor="middle")
    text(x + 84, y + 56, title, 26, "#0f172a", bold=True, max_w=w - 100)


out.append('<?xml version="1.0" encoding="UTF-8"?>')
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="\'Helvetica Neue\',Arial,sans-serif">')
out.append('''  <defs>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
      <path d="M0,0 L11,6 L0,12 z" fill="#64748b"/>
    </marker>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.10"/>
    </filter>
  </defs>''')
out.append(f'  <rect width="{W}" height="{H}" fill="#f8fafc"/>')

# Header
out.append("  <!-- Header -->")
text(80, 82, "Discovery workflow", 42, "#0f172a", bold=True)
text(80, 122, "Convert exploratory findings into reproduced, audited conclusions", 22, MUTED)
out.append('  <rect x="1440" y="68" width="28" height="20" rx="5" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>')
text(1480, 85, "agent judgment", 17, "#334155")
out.append('  <rect x="1660" y="68" width="28" height="20" rx="5" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
text(1700, 85, "driver compute", 17, "#334155")

# Contract bands
out.append("  <!-- Cross-cutting execution contract -->")
out.append('  <rect x="60" y="145" width="1800" height="34" rx="10" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5"/>')
out.append('  <line x1="690" y1="151" x2="690" y2="173" stroke="#cbd5e1" stroke-width="1.5"/>')
out.append('  <line x1="1260" y1="151" x2="1260" y2="173" stroke="#cbd5e1" stroke-width="1.5"/>')
for x, head, rest, max_w in ((82, "SESSION ISOLATION", " · fresh SDK session per agent step · file-based handoff", 590),
                             (712, "MODEL ROUTING", " · Opus 5.5 · analytical xhigh · mechanical medium", 530),
                             (1282, "OBSERVABILITY", " · per-step tokens · cache · model · time · cost", 560)):
    if width(head, 14, True) + width(rest, 14) > max_w:
        problems.append(f"band cell {head} too wide")
    out.append(f'  <text x="{x}" y="167" font-size="14" fill="#334155"><tspan font-weight="700" fill="#0f172a">{head}</tspan><tspan>{esc(rest)}</tspan></text>')
out.append('  <rect x="60" y="187" width="1800" height="34" rx="10" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5"/>')
band2 = (" · every fold step also writes <phase>-verdicts.json · the driver checks tokens, ids and agreement "
         "with the report · merges into <RUN>.verdicts.json · read by fix-tests selection and the detector build")
if width("STRUCTURED VERDICTS", 14, True) + width(band2, 14) > 1760:
    problems.append("band 2 too wide")
out.append(f'  <text x="82" y="209" font-size="14" fill="#334155"><tspan font-weight="700" fill="#0f172a">STRUCTURED VERDICTS</tspan><tspan>{esc(band2)}</tspan></text>')

Y0, CARD_H = 250, 560          # every phase card spans 250..810
ARROW_Y = 500
FIRST = Y0 + 84                # first box top: clear of the card title

# Phase 1: Summarize
out.append("  <!-- Phase 1 -->")
card(60, Y0, 300, CARD_H, 1, "Summarize")
box(88, FIRST, 244, 94, GREEN, [("Driver startup", 18), ("baseline cards · recorded-", 14), ("data audit of the export", 14)])
arrow(210, FIRST + 94, 210, FIRST + 122)
box(88, FIRST + 130, 244, 94, BLUE, [("Select hypotheses", 18), ("exclusion rules; audit flags", 14), ("are evidence, not verdicts", 14)])
arrow(210, FIRST + 224, 210, FIRST + 252)
box(88, FIRST + 260, 244, 108, BLUE, [("Rank discoveries", 18), ("evidence × surprise", 15), ("→ initial report", 15)])
arrow(360, ARROW_Y, 400, ARROW_Y)

# Phase 2: Reproduce
out.append("  <!-- Phase 2 -->")
card(410, Y0, 410, CARD_H, 2, "Reproduce")
box(438, FIRST, 354, 94, GREEN, [("Run recorded analysis as-is", 18), ("up to --jobs scripts at once · input audit", 14), ("library-skew rewrites", 14)])
arrow(615, FIRST + 94, 615, FIRST + 116)
LOOP = FIRST + 124
out.append(f'  <rect x="432" y="{LOOP}" width="366" height="156" rx="14" fill="#f1f5f9" stroke="#64748b" stroke-width="2" stroke-dasharray="8 5"/>')
text(456, LOOP + 32, "Loading repair loop · max 3 rounds", 17, "#334155", bold=True, max_w=340)
box(456, LOOP + 52, 142, 82, BLUE, [("Fix loading", 16), ("logic unchanged", 13)], rx=10, center=True)
arrow(598, LOOP + 93, 628, LOOP + 93)
box(636, LOOP + 52, 138, 82, GREEN, [("Remeasure", 16), ("changed only", 13)], rx=10, center=True)
arrow(615, LOOP + 156, 615, LOOP + 178)
DIFF = LOOP + 186
box(438, DIFF, 354, 72, GREEN, [("Numeric diff: recorded vs rerun", 17), ("the complete list of changed values", 14)])
arrow(615, DIFF + 72, 615, DIFF + 94)
box(438, DIFF + 102, 354, 52, BLUE, [("REPRODUCED · DIVERGED · FAILED", 16)], center=True)
arrow(820, ARROW_Y, 860, ARROW_Y)

# Phase 3: Generalize
out.append("  <!-- Phase 3 -->")
card(870, Y0, 300, CARD_H, 3, "Generalize", dashed=True)
text(898, FIRST + 10, "optional held-out brain", 15, MUTED)
box(898, FIRST + 28, 244, 94, GREEN, [("Run on another brain", 17), ("same code; image folder", 14), ("retargeted when needed", 14)])
arrow(1020, FIRST + 122, 1020, FIRST + 150)
box(898, FIRST + 158, 244, 118, BLUE, [("Judge transfer", 17), ("generalizes / partial / no", 14), ("each brain against its", 14), ("own baseline card", 14)])
arrow(1170, ARROW_Y, 1210, ARROW_Y)

# Phase 4: Audit
out.append("  <!-- Phase 4 -->")
card(1220, Y0, 280, CARD_H, 4, "Audit")
box(1248, FIRST, 224, 68, GREEN, [("Write slice of export", 16), ("code · output · audit flags", 13)])
arrow(1360, FIRST + 68, 1360, FIRST + 90)
B2 = FIRST + 98
box(1248, B2, 224, 146, BLUE, [("Audit in batches of 12", 15), ("test choice, assumptions", 13), ("effect vs baseline card", 13), ("deployability: GT-blind?", 13), ("→ verdicts + headline p", 13)])
arrow(1360, B2 + 146, 1360, B2 + 168)
B3 = B2 + 176
box(1248, B3, 224, 68, GREEN, [("Benjamini–Hochberg", 16), ("over headline p-values", 13)])
arrow(1360, B3 + 68, 1360, B3 + 90)
B4 = B3 + 98
box(1248, B4, 224, 68, BLUE, [("Synthesis turn", 16), ("run-wide section", 13)])
text(1360, Y0 + CARD_H - 14, "SOUND · WEAK · MINOR · MAJOR · CRITICAL", 12, MUTED, anchor="middle", max_w=270)
arrow(1500, ARROW_Y, 1540, ARROW_Y)

# Phase 5: Correct
out.append("  <!-- Phase 5 -->")
card(1550, Y0, 310, CARD_H, 5, "Correct")
box(1578, FIRST, 254, 118, BLUE, [("Write corrected test", 18), ("material issues only;", 14), ("measured with canonical_stats", 14), ("against the raw baseline", 14)])
arrow(1705, FIRST + 118, 1705, FIRST + 146)
box(1578, FIRST + 154, 254, 86, GREEN, [("Remeasure", 18), ("run corrected analysis", 15)])
arrow(1705, FIRST + 240, 1705, FIRST + 268)
box(1564, FIRST + 276, 282, 56, BLUE, [("UPHELD · WEAKENED · OVERTURNED", 12)], center=True)

# Phase 6: split-only feature applicability
out.append("  <!-- Phase 6: split-only feature applicability -->")
arrow(960, Y0 + CARD_H, 960, Y0 + CARD_H + 26)
P6 = Y0 + CARD_H + 34
out.append(f'  <rect x="420" y="{P6}" width="1080" height="116" rx="20" fill="#f5f3ff" stroke="#8b5cf6" stroke-width="2" stroke-dasharray="8 5" filter="url(#shadow)"/>')
out.append(f'  <circle cx="462" cy="{P6 + 36}" r="22" fill="#0f172a"/>')
text(462, P6 + 44, "6", 20, "#ffffff", bold=True, anchor="middle")
text(500, P6 + 44, "Label feature applicability", 23, "#0f172a", bold=True, max_w=390)
out.append(f'  <rect x="910" y="{P6 + 16}" width="92" height="28" rx="14" fill="#ede9fe" stroke="#8b5cf6" stroke-width="1.5"/>')
text(956, P6 + 35, "split only", 14, "#6d28d9", bold=True, anchor="middle", max_w=84)
out.append(f'  <rect x="480" y="{P6 + 56}" width="420" height="50" rx="9" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>')
semibold(690, P6 + 78, "Dedicated labeler: infer node-role requirement", 15, "#1e3a8a", anchor="middle", max_w=404)
text(690, P6 + 97, "both tips · tip anchor · no tip · unclear", 12.5, GREY, anchor="middle", max_w=404)
arrow(900, P6 + 80, 948, P6 + 80, sw=2.5)
out.append(f'  <rect x="956" y="{P6 + 56}" width="430" height="50" rx="9" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
semibold(1171, P6 + 87, "Driver: bind exact source path + SHA-256", 16, "#166534", anchor="middle", max_w=414)

# Final outputs
out.append("  <!-- Final outputs -->")
OUT = P6 + 148
arrow(960, P6 + 116, 960, OUT - 8)
out.append(f'  <rect x="120" y="{OUT}" width="1680" height="104" rx="22" fill="#eef2ff" stroke="#6366f1" stroke-width="2" filter="url(#shadow)"/>')
text(160, OUT + 58, "DISCOVERY OUTPUTS", 18, "#3730a3", bold=True, max_w=230)
for xd in (400, 870, 1335):
    out.append(f'  <line x1="{xd}" y1="{OUT + 18}" x2="{xd}" y2="{OUT + 86}" stroke="#c7d2fe" stroke-width="2"/>')
for cx, title, size, sub in ((635, "Final discovery report", 22, "ranked findings · reproduction · transfer · audit"),
                             (1102, "<RUN>.verdicts.json", 20, "every phase's checked verdict tokens · read by the detector build"),
                             (1567, "split-feature-applicability.json", 19, "split only · source-bound feature requirements")):
    text(cx, OUT + 50, title, size, "#3730a3", bold=True, anchor="middle", max_w=440)
    text(cx, OUT + 76, sub, 13, GREY, anchor="middle", max_w=440)

text(80, H - 20, "agentic/run_discovery_workflow.py", 15, "#94a3b8")
out.append("</svg>")

target = Path(sys.argv[1])
if problems:
    print("FIT PROBLEMS:\n  " + "\n  ".join(problems))
    sys.exit(1)
svg = "\n".join(out) + "\n"
if target.suffix == ".svg":
    target.write_text(svg, encoding="utf-8")
    print(f"wrote {target}")
else:
    import resvg_py  # noqa: E402  (see the module docstring)
    data = resvg_py.svg_to_bytes(svg_string=svg, font_dirs=["/usr/share/fonts", FONT_DIR],
                                 sans_serif_family="DejaVu Sans", font_family="DejaVu Sans")
    target.write_bytes(bytes(data))
    print(f"wrote {target} ({len(data) / 1e6:.2f} MB)")
