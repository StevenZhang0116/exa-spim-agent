# Reviser transcript

- thinking captured: no (0 chars)
- final text: 6169 chars
- tokens: in=39 out=86; cost_usd=106.07146124999998
- priors read: None

## Prompt sent to the reviser

```
Use the proofreader-reviser subagent to improve the evolved proofreading program. The current policy lives in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py and its theory in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md. Edit THOSE files in place (do not touch any other files). The failure report for the latest candidate is at /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen40/failure_report.md. Diagnose why the policy lost accuracy, make ONE concrete improvement to propose_edits (keeping its call signature), and update the rules file and its change log. Do NOT claim you ran, imported, or tested anything — you have no Bash; the harness import-checks and lint-checks your edit after you finish.

IMPORTANT — THIS RUN: merge-error repair is DISABLED. The candidate stream contains SplitSites only (no MergeSite is enumerated), and any `split_label` edit you emit is dropped before scoring. Do NOT write or tune `split_label` logic; focus entirely on the `merge_labels` (split-error) policy. The failure report's merge sections are omitted accordingly.

IMAGE CURRICULUM — couple reading image to PASSING THE GATE in ONE revision. The gate keeps a candidate only if it makes MORE net correct `merge_labels` repairs than the parent with ZERO false merges. A revision that merely STARTS calling `gap_bridge_evidence` without changing which labels you merge ties the parent's split-repair score and is REVERTED — so its reads (and any image signal) are thrown away and never reach a future report. Therefore, if you decide image is worth using, you MUST spend it to RAISE RECALL in the SAME revision: take SplitSites in the report's 'MISSED real splits (rejected REAL)' bucket — real splits your current geometry is too conservative to accept — and ACCEPT the ones a high `gap_bridge_evidence.bridge_ratio` confirms are a continuous bright bridge across the gap. Gate the (cloud) read behind your cheap geometric filters so you only read the handful of ambiguous candidates. Consult the report's 'Image warm-start probe' section FIRST: it already measured whether `bridge_ratio` separates REAL from FALSE on this brain (an AUC), so you know BEFORE writing the rule whether image is worth gating on and roughly where the threshold sits. If the AUC is weak, do not over-invest in image — improve the geometric `merge_labels` policy instead.

GENERALIZATION CHECK (aggregate over the last 5 ACCEPTED generations; no neuron identities, just scores): mean TRAIN split-repair score = 668.4, mean HELD-OUT split-repair score = 131.0, gap (train - held-out) = +537.4 — widening (train pulling ahead of held-out). The gate scores you on HELD-OUT, so a large positive gap means your recent accepted changes repair train splits that DON'T carry over. If the gap is widening, prefer improvements grounded in GENERALIZABLE geometry/topology (tangent continuity, caliber match, endpoint degree) over ones that chase train-specific recall.

```

## Stream (thinking / text / tool calls, in order)

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

The generalization aggregate shifted (held-out 127.8→131.0, train 635.8→668.4) and the do-not-repeat list reset — strong signals that **gen39 was ACCEPTED** as the new parent. Let me read the gen40 report to confirm and diagnose.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen40/failure_report.md"
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Confirmed: **gen39 was ACCEPTED** — train 692→703 (+11, all in the tip-to-tip `deg_b<2` bucket, now 182/0), held-out mean 127.8→131.0, lone false unchanged. We're on a fresh parent with no do-not-repeat list. The thin-distal-tip radius-floor signal worked and transferred.

Let me read the current `deg_b==2` branch structure (now shifted by gen39's additions) to design the next improvement precisely.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 455,
  "limit": 85
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

This is a clean opportunity. gen39's radius-floor thinness signal just proved it generalizes (accepted, held-out +). The natural next step is to extend that *same proven signal* from tip-to-tip (`deg_b==1`) to tip-to-shaft (`deg_b==2`), where it can safely open the previously-dead MID-caliber band: the lone train false there (2.19/0.39/**rad_a 1.00**/rr1.97) is a *thick* proximal endpoint, while genuine thin-distal-tip reconnections sit at the 0.75 radius floor — so a thinness gate cleanly separates them. Let me confirm `TIP_RAD_FLOOR` is available to reuse, then delegate.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 141,
  "limit": 2
}
```

### 📝 text [orchestrator]

`TIP_RAD_FLOOR = 0.80` is available to reuse (no new constant needed). I have a precise, well-grounded improvement. Let me delegate to the proofreader-reviser subagent.

### 🔧 tool call [orchestrator]: Agent

```json
{
  "description": "Revise proofreader policy for gen40",
  "subagent_type": "proofreader-reviser",
  "prompt": "You are revising an evolved split-error proofreading policy. Make ONE concrete, well-reasoned improvement to `propose_edits`, plus update the rules/changelog. I have already done the full diagnosis below \u2014 implement EXACTLY this change faithfully and explain it in the changelog. Do not invent a different change.\n\n## Files to edit IN PLACE (touch NO other files)\n- Policy: `/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py`\n- Theory/changelog: `/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md`\n\n## Hard constraints\n- You have NO Bash. Do NOT claim you ran, imported, or tested anything. The harness import-checks and lint-checks your edit after you finish, so keep it syntactically valid Python and preserve the `propose_edits(sites, ctx)` call signature.\n- THIS RUN merge-repair is DISABLED: do NOT write or tune any `split_label` logic; the candidate stream is SplitSites only. Focus entirely on the `merge_labels` (split-error) policy.\n- Image is RULED OUT this gen: the harness `bridge_ratio` warm-start AUC = 0.52 (no REAL/FALSE separation). Do NOT add any `gap_bridge_evidence` / image read.\n- The changelog in rules.md ACCRETES \u2014 append a new \"Gen 40\" entry, retain all prior entries, do not rewrite earlier sections.\n\n## How the gate scores you\nFitness = net correct `merge_labels` repairs (correct \u2212 false) on a HELD-OUT split. The candidate is accepted ONLY if it BEATS the parent's integer count AND has ZERO held-out false merges. One held-out false \u2192 outright REJECT. A tie \u2192 reverted.\n\n## CONTEXT \u2014 the immediately preceding generation (gen39) was ACCEPTED\ngen39 introduced a NEW, generalizable acceptance signal \u2014 ABSOLUTE endpoint thinness \u2014 and it WORKED: train split-repair 692\u2192703 (+11, all in the tip-to-tip `deg_b==1` bucket), and it transferred to held-out (mean held-out 127.8\u2192131.0). The key insight, now PROVEN to generalize: the segmentation pipeline's radius FLOOR is 0.75 (the minimum measurable cable radius across every report table; next value up is 1.00) \u2014 a BRAIN-INDEPENDENT pipeline constant. An endpoint with radius \u2264 0.80 sits AT that floor = a genuine thin DISTAL TIP (where a real neuron was broken), whereas two mature neurites grazing have BOTH radii \u2265 1.00. gen39 added the module constant `TIP_RAD_FLOOR = 0.80` and used it to accept caliber-MISMATCHED thin-tip reals for `deg_b==1`.\n\n## THE ONE IMPROVEMENT TO IMPLEMENT (gen40): extend the proven thinness signal from deg_b==1 to deg_b==2\nApply the SAME `TIP_RAD_FLOOR` thinness signal to the tip-to-shaft (`deg_b==2`) acceptor, to open the previously-DEAD MID-caliber band (rad_ratio between the gen28 match ceiling 1.15 and the gen30 mismatch floor 2.05) \u2014 but ONLY for thin distal tips. This is provably safe because the lone train false in that MID band is NOT a thin tip.\n\n### The exact code location\nIn `propose_edits`, tier (D) (gap (2.15, 3.0]), there is this branch (currently around lines 467\u2013506):\n\n```python\nelif deg_b == 2 and cc >= 0.0:\n    # ... (paths i/ii/iii comment block) ...\n    tl = _through_line_cos(g, s, TANGENT_WALK_UM)\n    rr = _rad_ratio(ctx, s.node_a, s.node_b)\n    if tl is not None and (\n            tl >= THROUGH_LINE_STRONG                                          # (i)  tl>=0.85 alone\n            or (tl >= THROUGH_LINE_COS and rr is not None and rr <= SHAFT_RAD_MATCH_MAX)      # (ii) tl>=0.70 & rr<=1.15 (caliber MATCH)\n            or (tl >= THROUGH_LINE_COS and rr is not None and rr >= SHAFT_RAD_MISMATCH_MIN)): # (iii) tl>=0.70 & rr>=2.05 (caliber MISMATCH)\n        accept = True\n```\n\n`THROUGH_L
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 440,
  "limit": 80
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "                                #      both in-band deg_b==2 falses (rr1.97 @2.19, rr1.20 @2.71) are BELOW\n                                #      2.05 so are excluded; the [2.04,inf)=0 false attribution corroborates.\n                                tl = _through_line_cos(g, s, TANGENT_WALK_UM)\n                                rr = _rad_ratio(ctx, s.node_a, s.node_b)\n                                if tl is not None and (\n                                        tl >= THROUGH_LINE_STRONG\n                                        or (tl >= THROUGH_LINE_COS\n                                            and rr is not None and rr <= SHAFT_RAD_MATCH_MAX)\n                                        or (tl >= THROUGH_LINE_COS\n                                            and rr is not None and rr >= SHAFT_RAD_MISMATCH_MIN)):\n                                    accept = True",
  "new_string": "                                #      both in-band deg_b==2 falses (rr1.97 @2.19, rr1.20 @2.71) are BELOW\n                                #      2.05 so are excluded; the [2.04,inf)=0 false attribution corroborates.\n                                #  (iv) Gen 40: extend gen39's PROVEN absolute-thinness tip signal from\n                                #      deg_b==1 to deg_b==2, OPENING the previously-DEAD MID-caliber band\n                                #      (1.15 < rr < 2.05) that paths (ii)/(iii) leave between the MATCH and\n                                #      MISMATCH ratio bands and that path (i) reaches only at tl>=0.85.\n                                #      Accept a THIN DISTAL TIP (min endpoint radius <= TIP_RAD_FLOOR=0.80,\n                                #      the brain-INDEPENDENT segmentation radius floor 0.75) joining a shaft\n                                #      WITH shaft-axis through-line continuity (tl >= THROUGH_LINE_COS=0.70)\n                                #      regardless of rad_ratio band. Two-signal CONJUNCTION (absolute\n                                #      thinness AND shaft-axis through-line) \u2014 a thin tip merely GRAZING the\n                                #      SIDE of a shaft would NOT align with the shaft axis (tl<0.70 rejects).\n                                #      PROVABLY FALSE-SAFE: both in-band deg_b==2 falses have a THICK minimum\n                                #      endpoint radius (2.19/rad1.00,1.97 => min 1.00 > 0.80; 2.71/rad1.67,\n                                #      2.01 => min 1.67 > 0.80), so path (iv) CANNOT fire for either \u2014 zero\n                                #      new train false (the lone 2.19 false stays caught only by path (i),\n                                #      count UNCHANGED). RECALL non-decreasing: appended OR-branch, paths\n                                #      (i)/(ii)/(iii) byte-for-byte unchanged; (iv) only ADDS thin-tip MID-\n                                #      band reals (e.g. 2.22/0.26/rad0.75,1.29/rr1.72, 2.20/0.28/rad0.75,\n                                #      1.68/rr2.24). GENERALIZES: same radius-floor key gen39 proved\n                                #      transfers to held-out.\n                                tl = _through_line_cos(g, s, TANGENT_WALK_UM)\n                                rr = _rad_ratio(ctx, s.node_a, s.node_b)\n                                ra = gd.get(\"rad_a\")\n                                rb = gd.get(\"rad_b\")\n                                if tl is not None and (\n                                        tl >= THROUGH_LINE_STRONG\n                                        or (tl >= THROUGH_LINE_COS\n                                            and rr is not None and rr <= SHAFT_RAD_MATCH_MAX)\n                                        or (tl >= THROUGH_LINE_COS\n                 
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 400,
  "limit": 50
}
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "offset": 1,
  "limit": 60
}
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "offset": 60,
  "limit": 30
}
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "offset": 88,
  "limit": 400
}
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "offset": 487,
  "limit": 400
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "  all helper bodies are untouched. This changelog accretes \u2014 all prior entries above\n  are retained, and the header / objective / criteria sections are unchanged. Mark\n  CANDIDATE pending the held-out gate.",
  "new_string": "  all helper bodies are untouched. This changelog accretes \u2014 all prior entries above\n  are retained, and the header / objective / criteria sections are unchanged. Mark\n  CANDIDATE pending the held-out gate.\n- **Gen 40 (2026-06-29) \u2014 CANDIDATE (pending the held-out gate):** EXTENDED gen39's\n  PROVEN ABSOLUTE-THINNESS tip signal from the tip-to-tip (`deg_b == 1`) acceptor to\n  the tip-to-shaft (`deg_b == 2`) `cc >= 0.0` acceptor, OPENING the previously-DEAD\n  MID-caliber band (1.15 < rad_ratio < 2.05) \u2014 the gap between the gen28 caliber-MATCH\n  ceiling `SHAFT_RAD_MATCH_MAX = 1.15` and the gen30 caliber-MISMATCH floor\n  `SHAFT_RAD_MISMATCH_MIN = 2.05` \u2014 but ONLY for genuine thin distal tips.\n  *Parent* = the gen39-accepted line: gen39 added the module constant\n  `TIP_RAD_FLOOR = 0.80` and PROVED that the segmentation pipeline's radius FLOOR\n  (0.75; the smallest measurable cable radius, next value up 1.00) is a\n  BRAIN-INDEPENDENT pipeline constant that TRANSFERS to held-out \u2014 train split-repair\n  692 \u2192 703 (+11, all in the `deg_b == 1` thin-tip bucket) and held-out mean\n  127.8 \u2192 131.0. *Diagnosis (the untapped recall):* in tier (D)'s `elif deg_b == 2\n  and cc >= 0.0:` branch the MID-caliber band (1.15 < rr < 2.05) is reachable ONLY via\n  path (i) `through_line >= THROUGH_LINE_STRONG = 0.85`; the caliber-MATCH path (ii)\n  (`rr <= 1.15`) and the caliber-MISMATCH path (iii) (`rr >= 2.05`) both leave this\n  band open, so genuine thin-tip-into-shaft reconnections in it with shaft-axis\n  through-line continuity but `through_line` only 0.70\u20130.85 are MISSED \u2014 e.g. report\n  MISSED rows `2.22/colinear 0.26/rad 0.75,1.29/rr1.72` and `2.20/colinear 0.28/rad\n  0.75,1.68/rr2.24`, both with an endpoint AT the 0.75 floor. *The ONE change:* read\n  the endpoint radii from the geom dict (`ra = gd.get(\"rad_a\")`, `rb = gd.get(\"rad_b\")`)\n  and add a FOURTH OR-branch (path iv) to the SAME `if tl is not None and (...)`\n  acceptance INSIDE the `elif deg_b == 2 and cc >= 0.0:` branch body:\n  `or (tl >= THROUGH_LINE_COS and ra is not None and rb is not None and\n  min(ra, rb) <= TIP_RAD_FLOOR)`. Path (iv) accepts a thin distal tip\n  (`min(rad_a, rad_b) <= TIP_RAD_FLOOR = 0.80`) joining a shaft WITH shaft-axis\n  through-line continuity (`tl >= THROUGH_LINE_COS = 0.70`), regardless of rad_ratio\n  band. NO new constant was introduced \u2014 `TIP_RAD_FLOOR = 0.80` (the gen39 constant)\n  is REUSED. The `elif deg_b == 2 and cc >= 0.0:` comment block gains a \"(iv) Gen 40:\"\n  note; paths (i)/(ii)/(iii) comment text is intact. *Why PROVABLY train-false-safe:*\n  both in-band (gap (2.15, 3.0]) `deg_b == 2` FALSE joins have a THICK minimum\n  endpoint radius, so path (iv)'s `min(rad_a, rad_b) <= 0.80` clause CANNOT fire for\n  either \u2014 (a) the lone accepted false `2.19 / colinear 0.39 / rad 1.00,1.97 / rr1.97`:\n  min radius 1.00 > 0.80 \u2192 path (iv) does NOT fire (it remains accepted via path (i)\n  exactly as before, so the train false count stays 1, UNCHANGED, no NEW false added);\n  (b) the correctly-refused false `2.71 / colinear 0.10 / rad 1.67,2.01 / rr1.20`: min\n  radius 1.67 > 0.80 \u2192 path (iv) does NOT fire (and paths i/ii/iii do not either), so\n  it stays correctly REJECTED. Path (iv) adds ZERO new train false. *Why recall is\n  strictly NON-DECREASING:* path (iv) is an APPENDED OR-branch; paths (i)/(ii)/(iii)\n  are byte-for-byte unchanged so no currently-accepted real is lost. Path (iv) only\n  ADDS accept
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

<analysis>
Gen 39 was ACCEPTED — the first acceptance in the visible window — by introducing the segmentation radius FLOOR as an absolute-thinness signal: an endpoint with radius ≤ 0.80 (TIP_RAD_FLOOR) sits at the pipeline's minimum measurable cable radius (0.75, the floor present in every report table; next value up is 1.00), so it is a genuine thin distal tip where a neuron broke, not a mature neurite grazing another. That signal is brain-independent (a pipeline constant, not a train-specific threshold) and it transferred to held-out. This validated the pivot away from seven straight rejections (gen32–gen38) that all chased recall by RELAXING continuity/caliber gates in the tier-(D) gap band — every one tied or admitted a held-out false because continuity-relaxation OR-branches are precision-fragile on held-out.

The widening gap this gen (mean TRAIN=668.4, HELD-OUT=131.0, +537.4) again says: do not chase train-specific recall; prefer generalizable geometry. The remaining structural opportunity is the deg_b==2 (tip-to-shaft) MID caliber band, rad_ratio ∈ (1.15, 2.05). It is a dead-zone: the deg_b==2 MATCH branch (gen28) needs rr ≤ 1.15 and the MISMATCH branch (gen30/31) needs rr ≥ 2.05, so a thin distal tip joining a shaft of intermediate caliber falls through and is never accepted. That dead-zone was deliberately left closed because the lone accepted train false (2.19 / cc 0.39 / rr 1.97 / rad_a 1.00) lives in it.

The ONE improvement for gen 40: extend gen39's PROVEN radius-floor thinness from deg_b==1 to deg_b==2, opening the MID band for THIN tips only. In the `elif deg_b == 2 and cc >= 0.0:` branch I read `ra = gd.get("rad_a")` / `rb = gd.get("rad_b")` and add a FOURTH OR-branch to the existing acceptance `if`:
`or (tl >= THROUGH_LINE_COS and ra is not None and rb is not None and min(ra, rb) <= TIP_RAD_FLOOR)`.
This reuses the existing TIP_RAD_FLOOR = 0.80 — NO new constant — and conjoins thinness with the shaft-axis through-line continuity `tl >= THROUGH_LINE_COS` (0.70). The through-line is reliable for deg_b==2 because a deg-2 node has a well-defined local axis, so a thin tip merely grazing the side of a shaft fails to align (tl < 0.70) and is rejected — this is the proven two-signal conjunction (continuity + absolute thinness + endpoint degree), not a relaxation.

Train-false-safety: both in-band deg_b==2 falses have thick min radius — the lone accepted false (rad 1.00, 1.97; min 1.00 > 0.80) and the correctly-refused false (rad 1.67, 2.01; min 1.67 > 0.80). Path-(iv) cannot fire for either; the 2.19 false stays caught only by path-(i), so its count is unchanged. Paths (i)/(ii)/(iii) are byte-for-byte unchanged, so recall is strictly non-decreasing — path-(iv) can only add accepts of genuine thin distal tips in the previously-dead band.
</analysis>

<summary>
Gen 40 makes ONE change to `propose_edits`: it extends gen39's accepted radius-floor absolute-thinness signal from the tip-to-tip case (deg_b==1) to the tip-to-shaft case (deg_b==2), opening the previously-dead MID caliber band (rad_ratio between 1.15 and 2.05) for thin distal tips only.

- Edited files (in place, no others): `artifacts/heuristics.py` and `artifacts/rules.md`.
- In `heuristics.py`, inside the `elif deg_b == 2 and cc >= 0.0:` branch, added `ra = gd.get("rad_a")` / `rb = gd.get("rad_b")` reads and a fourth OR-branch to the acceptance `if`: `or (tl >= THROUGH_LINE_COS and ra is not None and rb is not None and min(ra, rb) <= TIP_RAD_FLOOR)`. Reuses the existing `TIP_RAD_FLOOR = 0.80` — no new constant. Added a "(iv) Gen 40:" note to that branch's comment block.
- Paths (i)/(ii)/(iii), the deg_b==1 branches (including gen39's accepted thin-tip path), the other deg_b==2 branches, tiers A/B/C, the ENUM_PARAMS, and all helpers are unchanged. `split_label` logic untouched (disabled this run).
- Rationale: brain-independent pipeline constant (radius floor 0.75 → ≤0.80 = genuine thin distal tip), conjoined with reliable deg-2 shaft-axis through-line continuity (tl ≥ 0.70). Generalizable geometry/topology, not train-specific recall.
- Train-false-safety: both in-band deg_b==2 train falses have thick min radius (1.00 and 1.67, both > 0.80), so path-(iv) cannot fire for either; the lone 2.19 false stays caught only by path-(i), count unchanged. Recall is strictly non-decreasing because paths (i)/(ii)/(iii) are byte-for-byte unchanged.
- Appended a Gen 40 entry to the rules.md change log documenting the change, rationale, and false-safety argument.

Note: no tools were run; the harness import-checks and lint-checks the edit, then scores the candidate on the held-out split. Gen 40's gate outcome (net correct merge_labels vs. the gen39 parent, with zero held-out false merges required) will arrive with the next failure report.
</summary>
