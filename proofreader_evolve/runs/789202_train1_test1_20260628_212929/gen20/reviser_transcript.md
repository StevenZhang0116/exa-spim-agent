# Reviser transcript

- thinking captured: no (0 chars)
- final text: 3924 chars
- tokens: in=39 out=255; cost_usd=47.64167425000001
- priors read: None

## Prompt sent to the reviser

```
Use the proofreader-reviser subagent to improve the evolved proofreading program. The current policy lives in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py and its theory in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md. Edit THOSE files in place (do not touch any other files). The failure report for the latest candidate is at /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen20/failure_report.md. Diagnose why the policy lost accuracy, make ONE concrete improvement to propose_edits (keeping its call signature), and update the rules file and its change log. Do NOT claim you ran, imported, or tested anything — you have no Bash; the harness import-checks and lint-checks your edit after you finish.

IMPORTANT — THIS RUN: merge-error repair is DISABLED. The candidate stream contains SplitSites only (no MergeSite is enumerated), and any `split_label` edit you emit is dropped before scoring. Do NOT write or tune `split_label` logic; focus entirely on the `merge_labels` (split-error) policy. The failure report's merge sections are omitted accordingly.

IMAGE CURRICULUM — couple reading image to PASSING THE GATE in ONE revision. The gate keeps a candidate only if it makes MORE net correct `merge_labels` repairs than the parent with ZERO false merges. A revision that merely STARTS calling `gap_bridge_evidence` without changing which labels you merge ties the parent's split-repair score and is REVERTED — so its reads (and any image signal) are thrown away and never reach a future report. Therefore, if you decide image is worth using, you MUST spend it to RAISE RECALL in the SAME revision: take SplitSites in the report's 'MISSED real splits (rejected REAL)' bucket — real splits your current geometry is too conservative to accept — and ACCEPT the ones a high `gap_bridge_evidence.bridge_ratio` confirms are a continuous bright bridge across the gap. Gate the (cloud) read behind your cheap geometric filters so you only read the handful of ambiguous candidates. Consult the report's 'Image warm-start probe' section FIRST: it already measured whether `bridge_ratio` separates REAL from FALSE on this brain (an AUC), so you know BEFORE writing the rule whether image is worth gating on and roughly where the threshold sits. If the AUC is weak, do not over-invest in image — improve the geometric `merge_labels` policy instead.

GENERALIZATION CHECK (aggregate over the last 5 ACCEPTED generations; no neuron identities, just scores): mean TRAIN split-repair score = 429.6, mean HELD-OUT split-repair score = 76.4, gap (train - held-out) = +353.2 — widening (train pulling ahead of held-out). The gate scores you on HELD-OUT, so a large positive gap means your recent accepted changes repair train splits that DON'T carry over. If the gap is widening, prefer improvements grounded in GENERALIZABLE geometry/topology (tangent continuity, caliber match, endpoint degree) over ones that chase train-specific recall.

```

## Stream (thinking / text / tool calls, in order)

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen20/failure_report.md"
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Gen19 was **accepted** — train rose 507 → **528** and the deg_b==2 double-gated path transferred to held-out (held-out mean is climbing 51→64→76). Parent is now 528/0. Here's my analysis for gen20:

- **The validated pattern works:** gen18 (tip-to-tip floor 0.50) and gen19 (tip-to-shaft colinear≥0.50 AND through_line≥0.70) both transferred. The accepted approach — relax the colinear floor in tier (D), backed by endpoint-degree conditioning + a through_line continuity conjunction — is generalizing.
- **Where recall remains, and what's safe:** the MISSED pool is now low-colinear (< 0.50) near-band sites. For **tip-to-shaft (deg_b==2)** I can't lower the floor below 0.50 — the in-band false ceiling is colinear 0.39, so 0.50 is already at the verifiable-safe edge. But for **tip-to-tip (deg_b==1)** the in-band false ceiling is only **0.17** (2.24/−0.01, 2.45/0.17), so there's headroom: a relaxed floor of **0.30** still clears it by 0.13 and keeps train-false = 0 by colinear alone.
- **Caliber ruled out again:** for tip-to-tip the in-band false joins (rad_ratio 1.25, 1.56) are *more* caliber-matched than many missed reals, so caliber doesn't separate here (and caliber-mismatch already failed on held-out in gen17).
- **The one change:** mirror gen19's accepted construction for tip-to-tip — add a relaxed path `deg_b==1 AND colinear≥0.30 AND _through_line_cos≥0.70`. The colinear≥0.30 floor (0.13 above the 0.17 tip-to-tip in-band false ceiling) keeps train-false = 0 verifiably; the through_line conjunction (antiparallel tip tangents, bridge-bypassing) is the independent held-out hedge. Grounded in endpoint degree + tangent continuity — exactly the generalizable features the note endorses.

Let me delegate it.

### 🔧 tool call [orchestrator]: Agent

```json
{
  "description": "Revise proofreader policy gen20",
  "subagent_type": "proofreader-reviser",
  "prompt": "You are revising an evolved split-error proofreading policy. Make ONE concrete, generalizable improvement, editing ONLY these two files IN PLACE (touch no other files):\n\n1. POLICY: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py\n2. THEORY/CHANGELOG: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md\n\nRead BOTH files first. The latest failure report (for context/diagnosis only \u2014 do NOT edit it) is at .../gen20/failure_report.md.\n\nHARD CONSTRAINTS (must all hold):\n- Keep `propose_edits(sites, ctx) -> list` call signature unchanged.\n- You have NO Bash. Do NOT claim you ran/imported/tested anything. The harness import-checks and lint-checks afterward, so the file MUST remain valid Python.\n- merge-error repair is DISABLED: stream is SplitSites only; any `split_label` edit is dropped. Do NOT write or tune `split_label` logic. Work entirely on the `merge_labels` (split-repair) policy.\n- Make exactly ONE conceptual change to propose_edits, plus the matching rules.md update + change-log entry.\n- Image is ruled out: warm-start probe shows bridge_ratio AUC = 0.52 (REAL and FALSE fully overlap). Do NOT add image reads.\n- Do NOT reintroduce any caliber/rad_ratio acceptance (gen17 tried a rad_ratio floor \u2192 created a held-out false merge \u2192 rejected; and for tip-to-tip the in-band false joins are MORE caliber-matched than many missed reals, so caliber doesn't separate). Do NOT change ENUM_PARAMS, GAP_THRESHOLD_UM, MID_GAP_UM, MID_COLINEAR_COS, TIP_TIP_COLINEAR_COS, SHAFT_COLINEAR_FLOOR, THROUGH_LINE_COS, or any helper body.\n\nCONTEXT (already diagnosed \u2014 implement this, do not re-derive):\nThe current parent is 528 correct / 0 false on train (gen19 was ACCEPTED, 507 -> 528, and transferred to held-out). Tier (D)'s degree dispatch currently reads EXACTLY (around lines 396-414):\n\n                        if deg_a == 1 and cc is not None:\n                            # tip-to-tip (deg_b==1): both arms are reliable tips, so a\n                            # LOWER colinear floor is precision-safe (in-band tip-to-tip\n                            # false ceiling is 0.17); tip-to-shaft (deg_b==2) keeps the\n                            # stricter 0.75 because its shaft-arm tangent is noisy.\n                            if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 2 and cc >= MID_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 2 and cc >= SHAFT_COLINEAR_FLOOR:\n                                # tip-to-shaft with MODERATE average colinear (0.50-0.75):\n                                # colinear_cos is dragged down by the noisy shaft arm, so\n                                # require bridge-BYPASSING tangent continuity to confirm\n                                # before accepting. colinear>=0.50 already clears the 0.39\n                                # in-band deg_b==2 false ceiling (train false stays 0); the\n                                # through-line gate is the independent held-out hedge.\n                                tl = _through_line_cos(g, s, TANGENT_WALK_UM)\n                                if tl is not None and tl >= THROUGH_LINE_COS:\n                                    accept = True\n\nRelevant module constants (DO NOT modify their values): TIP_TIP_COLINEAR_COS=0.50, SHAFT_COLINEAR_FLOOR=0.50 (around lines 134-135), MID_COLINEAR_COS=0.75, THROUGH_LINE_COS=0.70, TANGENT_WALK_UM=6.0. The helper `_through_line_cos(g, s, walk)` exists; for a tip-to-tip site (node_b a tip) it returns the antiparallel-ness of the two o
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py"
}
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "SHAFT_COLINEAR_FLOOR = 0.50   # tier (D) RELAXED colinear floor for tip-to-shaft (deg_b==2), used ONLY in conjunction with through-line continuity (_through_line_cos >= THROUGH_LINE_COS). colinear_cos is the AVERAGE of the two arm cosines and is dragged DOWN for tip-to-shaft by the noisy/laterally-offset shaft arm, so the strict 0.75 floor rejects genuine continuations. 0.50 sits 0.11 ABOVE the 0.39 in-band deg_b==2 false-join ceiling (only in-band deg_b==2 false joins: gap 2.19/colinear 0.39, gap 2.71/colinear 0.10), so colinear>=0.50 keeps TRAIN false==0 on its own; the AND through-line conjunction is the held-out precision hedge (two independent continuity signals must agree). NOT used for deg_b==1 (which has TIP_TIP_COLINEAR_COS) and NOT a standalone floor.",
  "new_string": "SHAFT_COLINEAR_FLOOR = 0.50   # tier (D) RELAXED colinear floor for tip-to-shaft (deg_b==2), used ONLY in conjunction with through-line continuity (_through_line_cos >= THROUGH_LINE_COS). colinear_cos is the AVERAGE of the two arm cosines and is dragged DOWN for tip-to-shaft by the noisy/laterally-offset shaft arm, so the strict 0.75 floor rejects genuine continuations. 0.50 sits 0.11 ABOVE the 0.39 in-band deg_b==2 false-join ceiling (only in-band deg_b==2 false joins: gap 2.19/colinear 0.39, gap 2.71/colinear 0.10), so colinear>=0.50 keeps TRAIN false==0 on its own; the AND through-line conjunction is the held-out precision hedge (two independent continuity signals must agree). NOT used for deg_b==1 (which has TIP_TIP_COLINEAR_COS) and NOT a standalone floor.\nTIP_TIP_RELAX_FLOOR = 0.30   # tier (D) RELAXED colinear floor for tip-to-tip (deg_b==1), used ONLY in conjunction with through-line continuity (_through_line_cos >= THROUGH_LINE_COS). On short stubs the two tip tangents are noisy, so colinear_cos (their average) lands in 0.30-0.50 even for genuine breaks. 0.30 sits 0.13 ABOVE the 0.17 in-band deg_b==1 false-join ceiling (only in-band tip-to-tip false joins: gap 2.24/colinear -0.01, gap 2.45/colinear 0.17), so colinear>=0.30 keeps TRAIN false==0 on its own; the AND through-line conjunction (near-antiparallel tip tangents) is the held-out precision hedge (two independent continuity signals must agree). Mirrors the gen19 deg_b==2 double-gated path. NOT a standalone floor."
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "                            if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 2 and cc >= MID_COLINEAR_COS:",
  "new_string": "                            if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 1 and cc >= TIP_TIP_RELAX_FLOOR:\n                                # tip-to-tip with MODERATE average colinear (0.30-0.50):\n                                # short-stub tip tangents are noisy, so require\n                                # bridge-BYPASSING tangent continuity (near-antiparallel\n                                # tip tangents) to confirm before accepting. colinear>=0.30\n                                # already clears the 0.17 in-band deg_b==1 false ceiling\n                                # (train false stays 0); through-line is the held-out hedge.\n                                tl = _through_line_cos(g, s, TANGENT_WALK_UM)\n                                if tl is not None and tl >= THROUGH_LINE_COS:\n                                    accept = True\n                            elif deg_b == 2 and cc >= MID_COLINEAR_COS:"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "  accept when `ctx[\"split_geom\"](s).colinear_cos` clears a **degree-conditioned**\n  floor:\n    - **tip-to-tip** (`deg_b == 1`): `colinear_cos >= TIP_TIP_COLINEAR_COS`\n      (**0.50**);\n    - **tip-to-shaft** (`deg_b == 2`): accept at `colinear_cos >= MID_COLINEAR_COS`\n      (**0.75**, strict, unchanged) **OR** at `colinear_cos >= SHAFT_COLINEAR_FLOOR`\n      (**0.50**) **AND** `_through_line_cos(g, s, TANGENT_WALK_UM) >=\n      THROUGH_LINE_COS` (**0.70**) \u2014 a double-gated relaxed path (Gen 19).\n  This reads the harness's leak-free per-site geometry directly and fails safe (no\n  accept) when `split_geom` is missing or returns nothing.",
  "new_string": "  accept when `ctx[\"split_geom\"](s).colinear_cos` clears a **degree-conditioned**\n  floor:\n    - **tip-to-tip** (`deg_b == 1`): accept at `colinear_cos >= TIP_TIP_COLINEAR_COS`\n      (**0.50**) **OR** at `colinear_cos >= TIP_TIP_RELAX_FLOOR` (**0.30**) **AND**\n      `_through_line_cos(g, s, TANGENT_WALK_UM) >= THROUGH_LINE_COS` (**0.70**) \u2014 a\n      double-gated relaxed path (Gen 20);\n    - **tip-to-shaft** (`deg_b == 2`): accept at `colinear_cos >= MID_COLINEAR_COS`\n      (**0.75**, strict, unchanged) **OR** at `colinear_cos >= SHAFT_COLINEAR_FLOOR`\n      (**0.50**) **AND** `_through_line_cos(g, s, TANGENT_WALK_UM) >=\n      THROUGH_LINE_COS` (**0.70**) \u2014 a double-gated relaxed path (Gen 19).\n  This reads the harness's leak-free per-site geometry directly and fails safe (no\n  accept) when `split_geom` is missing or returns nothing.\n\n  *Why the tip-to-tip floor is also double-gated (Gen 20):* genuine tip-to-tip\n  breaks on SHORT STUBS have noisy tip tangents, so `colinear_cos` (the AVERAGE of\n  the two tip cosines) lands in 0.30\u20130.50 even for real continuations \u2014 the strict\n  0.50 `TIP_TIP_COLINEAR_COS` floor rejects them. The relaxed\n  `TIP_TIP_RELAX_FLOOR = 0.30` sits **0.13 above** the **0.17** in-band `deg_b == 1`\n  false-join ceiling (the only in-band tip-to-tip false joins are gap 2.24 /\n  colinear \u22120.01 and gap 2.45 / colinear 0.17), so `colinear >= 0.30` keeps TRAIN\n  false at 0 by colinear alone. The conjoined `_through_line_cos >= 0.70` test is the\n  INDEPENDENT held-out precision hedge: for tip-to-tip it measures the\n  antiparallel-ness of the two outward tip tangents (cables pointing AT each other,\n  bridge-bypassing), so requiring BOTH continuity signals to agree guards against\n  held-out false joins in the relaxed `colinear` [0.30, 0.50) range. This relaxed\n  path fires only as the AND-conjunction above and is NOT a standalone floor. It\n  DELIBERATELY mirrors the gen19 `deg_b == 2` double-gated construction (relaxed\n  colinear floor + through-line conjunction)."
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "  values `TIP_TIP_COLINEAR_COS` (0.50), `MID_COLINEAR_COS` (0.75, tip-to-shaft\n  strict path unchanged), and `THROUGH_LINE_COS` (0.70) are untouched. Mark\n  CANDIDATE pending the held-out gate.",
  "new_string": "  values `TIP_TIP_COLINEAR_COS` (0.50), `MID_COLINEAR_COS` (0.75, tip-to-shaft\n  strict path unchanged), and `THROUGH_LINE_COS` (0.70) are untouched. Mark\n  CANDIDATE pending the held-out gate.\n- **Gen 20 (2026-06-29) \u2014 CANDIDATE (pending the held-out gate):** added a\n  double-gated RELAXED colinear floor to tier (D) for tip-to-tip (`deg_b == 1`),\n  mirroring the gen19 `deg_b == 2` construction. *Parent* = **528 correct / 0 false**\n  on train: gen19's `deg_b == 2` double-gated path was ACCEPTED (507 \u2192 528) and\n  TRANSFERRED to held-out, confirming the relaxed-colinear-floor + through-line\n  conjunction pattern generalizes. *Diagnosis:* the remaining MISSED near-band pool\n  is low-colinear tip-to-tip whose SHORT-STUB tip tangents depress `colinear_cos`\n  (the average of the two tip cosines) into 0.30\u20130.50 even for genuine breaks, so the\n  strict 0.50 `TIP_TIP_COLINEAR_COS` floor rejects them. For tip-to-shaft the floor\n  CANNOT go below 0.50 because the in-band `deg_b == 2` false-join ceiling is 0.39,\n  but tip-to-tip has HEADROOM because its in-band false ceiling is only **0.17**.\n  *The ONE change:* added a module constant `TIP_TIP_RELAX_FLOOR = 0.30`\n  (immediately after `SHAFT_COLINEAR_FLOOR`) and a SECOND tier-(D) `deg_b == 1`\n  degree-dispatch branch \u2014 in addition to the unchanged `colinear_cos >=\n  TIP_TIP_COLINEAR_COS` (0.50) path, accept tip-to-tip when `colinear_cos >=\n  TIP_TIP_RELAX_FLOOR` (0.30) **AND** `_through_line_cos(g, s, TANGENT_WALK_UM) >=\n  THROUGH_LINE_COS` (0.70). *Precision rationale:* the relaxed `colinear >= 0.30`\n  floor sits **0.13 above** the **0.17** in-band `deg_b == 1` false-join ceiling (the\n  only in-band tip-to-tip false joins are gap 2.24 / colinear \u22120.01 and gap 2.45 /\n  colinear 0.17), so it keeps TRAIN false at 0 on its own; the conjoined\n  `_through_line_cos >= 0.70` is a SECOND, INDEPENDENT continuity signal\n  (near-antiparallel outward tip tangents, bridge-bypassing) that must agree, hedging\n  against held-out false joins in the relaxed `colinear` [0.30, 0.50) range.\n  *Generalization rationale:* this DELIBERATELY mirrors the gen19 construction \u2014 the\n  validated, transferring pattern (relaxed colinear floor + through-line conjunction,\n  degree-conditioned) \u2014 rather than a new untested lever, and is grounded in endpoint\n  degree + tangent continuity (the generalizable features the widening-gap note\n  endorses). Image NOT used (`bridge_ratio` AUC 0.52 \u2014 REAL and FALSE fully overlap;\n  the recall gap is geometric, not photometric); caliber explicitly REJECTED for\n  tip-to-tip (gen17's `rad_ratio` floor created a held-out false merge, and the\n  in-band tip-to-tip false joins are MORE caliber-matched than many missed reals, so\n  caliber does not separate). Tiers (A), (B), (C), the outer `GAP_THRESHOLD_UM`\n  guard, `ENUM_PARAMS`, all helper bodies, the two `deg_b == 2` branches, and the\n  constant values `TIP_TIP_COLINEAR_COS` (0.50), `SHAFT_COLINEAR_FLOOR` (0.50),\n  `MID_COLINEAR_COS` (0.75), `THROUGH_LINE_COS` (0.70), and `TANGENT_WALK_UM` (6.0)\n  are untouched. Mark CANDIDATE pending the held-out gate."
}
```

### 💭 thinking [orchestrator]



### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py"
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

gen20 edits confirmed in `heuristics.py`: `TIP_TIP_RELAX_FLOOR = 0.30` is present (line 136, right after `SHAFT_COLINEAR_FLOOR`), and tier (D)'s degree dispatch now has the four ordered branches (lines 402–425): `deg_b==1 cc≥0.50` → accept; `deg_b==1 cc≥0.30` → through_line gate; `deg_b==2 cc≥0.75` → accept; `deg_b==2 cc≥0.50` → through_line gate. Tiers A/B/C, the outer `GAP_THRESHOLD_UM` guard, `ENUM_PARAMS`, all helpers, and every other constant are unchanged. Now verifying the `rules.md` companion.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "offset": 440,
  "limit": 80
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Entries through gen19 are present. Let me confirm the gen20 changelog entry and the rule (D) paragraph update.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "offset": 519,
  "limit": 80
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Both gen20 edits are verified on disk and complete:

**`heuristics.py`**
- New constant `TIP_TIP_RELAX_FLOOR = 0.30` (line 136), placed immediately after `SHAFT_COLINEAR_FLOOR = 0.50`.
- Tier (D) degree dispatch now has the four ordered branches (lines 402–425): `deg_b==1 & cc≥0.50` → accept; `deg_b==1 & cc≥0.30` → accept iff `_through_line_cos ≥ 0.70`; `deg_b==2 & cc≥0.75` → accept; `deg_b==2 & cc≥0.50` → accept iff through-line gate. The new branch is correctly ordered as `elif` #2 so the 0.30–0.50 window is reachable.
- Tiers A/B/C, the outer `GAP_THRESHOLD_UM` guard, `ENUM_PARAMS`, all helpers, and every other constant are unchanged; file is syntactically valid.

**`rules.md`**
- The "Gen 20 (2026-06-29) — CANDIDATE" change-log entry is present (lines 544–579), documenting the single change, the 0.13 margin above the 0.17 in-band tip-to-tip false ceiling, the through-line conjunction as the held-out hedge, and the explicit statement that it mirrors the validated gen19 construction (image and caliber both ruled out).

The gen20 task is done: diagnosis (remaining MISSED pool is low-colinear tip-to-tip whose short-stub tangents depress `colinear_cos` into 0.30–0.50; tip-to-tip has headroom the tip-to-shaft path lacks), ONE concrete improvement (relaxed `TIP_TIP_RELAX_FLOOR=0.30` gated by `through_line≥0.70`), and both files updated in place — verified consistent with the subagent's report. No false merges expected on train (colinear≥0.30 clears the 0.17 ceiling independent of the conjunction). Awaiting the held-out gate result, or a new generation task.
