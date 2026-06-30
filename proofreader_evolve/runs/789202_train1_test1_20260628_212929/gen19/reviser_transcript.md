# Reviser transcript

- thinking captured: no (0 chars)
- final text: 4472 chars
- tokens: in=40 out=132; cost_usd=44.16243425
- priors read: None

## Prompt sent to the reviser

```
Use the proofreader-reviser subagent to improve the evolved proofreading program. The current policy lives in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py and its theory in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md. Edit THOSE files in place (do not touch any other files). The failure report for the latest candidate is at /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen19/failure_report.md. Diagnose why the policy lost accuracy, make ONE concrete improvement to propose_edits (keeping its call signature), and update the rules file and its change log. Do NOT claim you ran, imported, or tested anything — you have no Bash; the harness import-checks and lint-checks your edit after you finish.

IMPORTANT — THIS RUN: merge-error repair is DISABLED. The candidate stream contains SplitSites only (no MergeSite is enumerated), and any `split_label` edit you emit is dropped before scoring. Do NOT write or tune `split_label` logic; focus entirely on the `merge_labels` (split-error) policy. The failure report's merge sections are omitted accordingly.

IMAGE CURRICULUM — couple reading image to PASSING THE GATE in ONE revision. The gate keeps a candidate only if it makes MORE net correct `merge_labels` repairs than the parent with ZERO false merges. A revision that merely STARTS calling `gap_bridge_evidence` without changing which labels you merge ties the parent's split-repair score and is REVERTED — so its reads (and any image signal) are thrown away and never reach a future report. Therefore, if you decide image is worth using, you MUST spend it to RAISE RECALL in the SAME revision: take SplitSites in the report's 'MISSED real splits (rejected REAL)' bucket — real splits your current geometry is too conservative to accept — and ACCEPT the ones a high `gap_bridge_evidence.bridge_ratio` confirms are a continuous bright bridge across the gap. Gate the (cloud) read behind your cheap geometric filters so you only read the handful of ambiguous candidates. Consult the report's 'Image warm-start probe' section FIRST: it already measured whether `bridge_ratio` separates REAL from FALSE on this brain (an AUC), so you know BEFORE writing the rule whether image is worth gating on and roughly where the threshold sits. If the AUC is weak, do not over-invest in image — improve the geometric `merge_labels` policy instead.

GENERALIZATION CHECK (aggregate over the last 5 ACCEPTED generations; no neuron identities, just scores): mean TRAIN split-repair score = 387.2, mean HELD-OUT split-repair score = 63.6, gap (train - held-out) = +323.6 — widening (train pulling ahead of held-out). The gate scores you on HELD-OUT, so a large positive gap means your recent accepted changes repair train splits that DON'T carry over. If the gap is widening, prefer improvements grounded in GENERALIZABLE geometry/topology (tangent continuity, caliber match, endpoint degree) over ones that chase train-specific recall.

```

## Stream (thinking / text / tool calls, in order)

### 💭 thinking [orchestrator]



### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen19/failure_report.md"
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Gen18 was **accepted** — train split-repair rose 461 → **507** and the tip-to-tip floor-0.50 change transferred to held-out (it's now the parent). So this isn't a regression to diagnose; the gen19 report reflects the new 507/0 parent, and I need the next generalizable gain.

Here's my analysis:

- **The transferable signal is tip-to-tip.** gen18 (lowering the colinear floor for `deg_b==1` only) was accepted and *transferred* — confirming tip-to-tip continuation generalizes (both endpoints are reliable degree-1 tips). The widening +323.6 gap is driven by the **tip-to-shaft (`deg_b==2`)** near-band pool, which is train-specific noise.
- **Why `deg_b==2` resists every single-feature gate:** `colinear_cos` (the bridge-routed *average*) is dragged down because the shaft-side arm tangent is unreliable / the bridge vector is laterally offset on a tip-into-shaft join. The in-band `deg_b==2` false ceiling is colinear 0.39, so I can't safely lower the tip-to-shaft colinear floor below ~0.72 on colinear alone. Caliber (rad_ratio) already failed on held-out (gen17). `through_line` *window-extension* failed (gen8). Image is out (AUC 0.52, total REAL/FALSE overlap in the probe).
- **The one untried construction:** a *double-gated* tip-to-shaft path — relax the colinear floor to 0.50 for `deg_b==2` **but require `_through_line_cos ≥ 0.70` as an AND conjunction**. The colinear ≥ 0.50 floor sits 0.11 above the 0.39 in-band `deg_b==2` false ceiling, so it keeps **train false = 0 by itself** (verifiable from the report, independent of through_line); the through_line conjunction (tangent-to-tangent continuity that *bypasses* the offset bridge) is an independent second continuity signal that hedges against held-out false joins with colinear in [0.50, 0.75). Two unrelated neurites would have to coincidentally satisfy both — a much safer bet than either alone, and grounded in the exact features the generalization note endorses (tangent continuity + endpoint degree). This is distinct from gen8 (OR window-extension via caliber), gen17 (caliber floor), and gen18 (`deg_b==1`).

Let me delegate it.

### 🔧 tool call [orchestrator]: Agent

```json
{
  "description": "Revise proofreader policy gen19",
  "subagent_type": "proofreader-reviser",
  "prompt": "You are revising an evolved split-error proofreading policy. Make ONE concrete, generalizable improvement, editing ONLY these two files IN PLACE (touch no other files):\n\n1. POLICY: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py\n2. THEORY/CHANGELOG: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md\n\nRead BOTH files first. The latest failure report (for context/diagnosis only \u2014 do NOT edit it) is at .../gen19/failure_report.md.\n\nHARD CONSTRAINTS (must all hold):\n- Keep `propose_edits(sites, ctx) -> list` call signature unchanged.\n- You have NO Bash. Do NOT claim you ran/imported/tested anything. The harness import-checks and lint-checks afterward, so the file MUST remain valid Python.\n- merge-error repair is DISABLED: stream is SplitSites only; any `split_label` edit is dropped. Do NOT write or tune `split_label` logic. Work entirely on the `merge_labels` (split-repair) policy.\n- Make exactly ONE conceptual change to propose_edits, plus the matching rules.md update + change-log entry.\n- Image is ruled out: warm-start probe shows bridge_ratio AUC = 0.52 (REAL and FALSE fully overlap). Do NOT add image reads.\n- Do NOT reintroduce any caliber/rad_ratio acceptance (gen17 tried a rad_ratio floor \u2192 created a held-out false merge \u2192 rejected). Do NOT change ENUM_PARAMS, GAP_THRESHOLD_UM, MID_GAP_UM, MID_COLINEAR_COS, TIP_TIP_COLINEAR_COS, or any helper body.\n\nCONTEXT (already diagnosed \u2014 implement this, do not re-derive):\nThe current parent is 507 correct / 0 false on train (gen18 was accepted). The if/elif chain's tier (D) currently reads EXACTLY (around lines 387-403):\n\n            elif NEAR_GAP_UM < s.gap_um <= MID_GAP_UM:\n                geom_fn = ctx.get(\"split_geom\")\n                if geom_fn is not None:\n                    gd = geom_fn(s)\n                    if gd:\n                        deg_a = gd.get(\"deg_a\")\n                        deg_b = gd.get(\"deg_b\")\n                        cc = gd.get(\"colinear_cos\")\n                        if deg_a == 1 and cc is not None:\n                            # tip-to-tip (deg_b==1): both arms are reliable tips, so a\n                            # LOWER colinear floor is precision-safe (in-band tip-to-tip\n                            # false ceiling is 0.17); tip-to-shaft (deg_b==2) keeps the\n                            # stricter 0.75 because its shaft-arm tangent is noisy.\n                            if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 2 and cc >= MID_COLINEAR_COS:\n                                accept = True\n\nModule constants of interest (do NOT modify their values): NEAR_GAP_UM=2.15, MID_GAP_UM=3.0, MID_COLINEAR_COS=0.75, TIP_TIP_COLINEAR_COS=0.50 (around lines 128-134), THROUGH_LINE_COS=0.70 and TANGENT_WALK_UM=6.0 (defined earlier). The helper `_through_line_cos(g, s, walk)` already exists (it measures bridge-BYPASSING tangent-to-tangent continuity: for tip-to-shaft it compares arm A's outward tangent to the shaft's local axis at node_b). `g = ctx.get(\"fragments_graph\")` is in scope at the top of propose_edits.\n\nTHE ONE CHANGE \u2014 a double-gated relaxed colinear floor for tip-to-shaft (deg_b==2):\nTier (D) keeps tip-to-shaft at the strict 0.75 colinear floor because colinear_cos (the AVERAGE of the two arm cosines) is dragged DOWN for tip-to-shaft by the noisy/laterally-offset shaft-side arm. So genuine tip-to-shaft continuations in (2.15, 3.0] with colinear 0.50-0.75 are wrongly rejected. Relax the floor to 0.50 for deg_b==2 ONLY WHEN bridge-bypassing tangent continuity inde
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
  "old_string": "TIP_TIP_COLINEAR_COS = 0.50   # tier (D) colinear floor for TIP-TO-TIP (deg_b==1) only. Both endpoints are degree-1 tips with reliable tangents, so colinear_cos (the AVERAGE of the two arm cosines) is trustworthy \u2014 unlike tip-to-shaft, where the shaft arm tangent is noisy and drags the average down. Within tier (D)'s band (2.15, 3.0] the only tip-to-tip FALSE joins are at colinear -0.01 (gap 2.24) and 0.17 (gap 2.45), ceiling 0.17, so 0.50 carries a 0.33 margin (comparable to the 0.36 margin of the proven-transferable 0.75 tip-to-shaft floor) while admitting MISSED tip-to-tip reals at colinear 0.50-0.73. tip-to-shaft (deg_b==2) keeps the stricter MID_COLINEAR_COS=0.75 because its shaft-arm tangent is unreliable.",
  "new_string": "TIP_TIP_COLINEAR_COS = 0.50   # tier (D) colinear floor for TIP-TO-TIP (deg_b==1) only. Both endpoints are degree-1 tips with reliable tangents, so colinear_cos (the AVERAGE of the two arm cosines) is trustworthy \u2014 unlike tip-to-shaft, where the shaft arm tangent is noisy and drags the average down. Within tier (D)'s band (2.15, 3.0] the only tip-to-tip FALSE joins are at colinear -0.01 (gap 2.24) and 0.17 (gap 2.45), ceiling 0.17, so 0.50 carries a 0.33 margin (comparable to the 0.36 margin of the proven-transferable 0.75 tip-to-shaft floor) while admitting MISSED tip-to-tip reals at colinear 0.50-0.73. tip-to-shaft (deg_b==2) keeps the stricter MID_COLINEAR_COS=0.75 because its shaft-arm tangent is unreliable.\nSHAFT_COLINEAR_FLOOR = 0.50   # tier (D) RELAXED colinear floor for tip-to-shaft (deg_b==2), used ONLY in conjunction with through-line continuity (_through_line_cos >= THROUGH_LINE_COS). colinear_cos is the AVERAGE of the two arm cosines and is dragged DOWN for tip-to-shaft by the noisy/laterally-offset shaft arm, so the strict 0.75 floor rejects genuine continuations. 0.50 sits 0.11 ABOVE the 0.39 in-band deg_b==2 false-join ceiling (only in-band deg_b==2 false joins: gap 2.19/colinear 0.39, gap 2.71/colinear 0.10), so colinear>=0.50 keeps TRAIN false==0 on its own; the AND through-line conjunction is the held-out precision hedge (two independent continuity signals must agree). NOT used for deg_b==1 (which has TIP_TIP_COLINEAR_COS) and NOT a standalone floor."
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "                            if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 2 and cc >= MID_COLINEAR_COS:\n                                accept = True\n            if accept:",
  "new_string": "                            if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 2 and cc >= MID_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 2 and cc >= SHAFT_COLINEAR_FLOOR:\n                                # tip-to-shaft with MODERATE average colinear (0.50-0.75):\n                                # colinear_cos is dragged down by the noisy shaft arm, so\n                                # require bridge-BYPASSING tangent continuity to confirm\n                                # before accepting. colinear>=0.50 already clears the 0.39\n                                # in-band deg_b==2 false ceiling (train false stays 0); the\n                                # through-line gate is the independent held-out hedge.\n                                tl = _through_line_cos(g, s, TANGENT_WALK_UM)\n                                if tl is not None and tl >= THROUGH_LINE_COS:\n                                    accept = True\n            if accept:"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "    - **tip-to-tip** (`deg_b == 1`): `colinear_cos >= TIP_TIP_COLINEAR_COS`\n      (**0.50**);\n    - **tip-to-shaft** (`deg_b == 2`): `colinear_cos >= MID_COLINEAR_COS`\n      (**0.75**).\n  This reads the harness's leak-free per-site geometry directly and fails safe (no\n  accept) when `split_geom` is missing or returns nothing.",
  "new_string": "    - **tip-to-tip** (`deg_b == 1`): `colinear_cos >= TIP_TIP_COLINEAR_COS`\n      (**0.50**);\n    - **tip-to-shaft** (`deg_b == 2`): accept at `colinear_cos >= MID_COLINEAR_COS`\n      (**0.75**, strict, unchanged) **OR** at `colinear_cos >= SHAFT_COLINEAR_FLOOR`\n      (**0.50**) **AND** `_through_line_cos(g, s, TANGENT_WALK_UM) >=\n      THROUGH_LINE_COS` (**0.70**) \u2014 a double-gated relaxed path (Gen 19).\n  This reads the harness's leak-free per-site geometry directly and fails safe (no\n  accept) when `split_geom` is missing or returns nothing.\n\n  *Why the tip-to-shaft floor is double-gated (Gen 19):* `colinear_cos` is the\n  AVERAGE of the two arm cosines, and for tip-to-shaft it is DEPRESSED by the\n  noisy/laterally-offset shaft-side arm \u2014 so genuine tip-to-shaft continuations in\n  (2.15, 3.0] with `colinear_cos` 0.50\u20130.75 are wrongly rejected by the strict 0.75\n  floor. The relaxed `SHAFT_COLINEAR_FLOOR = 0.50` sits **0.11 above** the **0.39**\n  in-band `deg_b == 2` false-join ceiling (the only in-band `deg_b == 2` false joins\n  are gap 2.19 / colinear 0.39 and gap 2.71 / colinear 0.10), so `colinear >= 0.50`\n  keeps TRAIN false at 0 by colinear alone. The conjoined `_through_line_cos >= 0.70`\n  test is the INDEPENDENT held-out precision hedge: it is a bridge-BYPASSING\n  tangent-to-tangent continuity measure (it compares arm A's outward tangent to the\n  shaft's local axis at `node_b`, not the gap bridge vector), so requiring BOTH\n  continuity signals to agree guards against held-out false joins in the relaxed\n  `colinear` [0.50, 0.75) range. This relaxed path is NOT used for tip-to-tip\n  (`deg_b == 1`, which has its own `TIP_TIP_COLINEAR_COS`) and is NOT a standalone\n  floor \u2014 it fires only as the AND-conjunction above."
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "- **Harness:** `candidate_split_sites` broadened from tip-to-tip to",
  "new_string": "- **Gen 19 (2026-06-29) \u2014 CANDIDATE (pending the held-out gate):** added a\n  double-gated RELAXED colinear floor to tier (D) for tip-to-shaft (`deg_b == 2`).\n  *Parent* = **507 correct / 0 false** on train: gen18's degree-conditioned floor\n  (the tip-to-tip `TIP_TIP_COLINEAR_COS = 0.50` path) was ACCEPTED (461 \u2192 507) and\n  TRANSFERRED to held-out, confirming the leak-free `split_geom.colinear_cos` lever\n  generalizes. *Diagnosis:* the still-widening train/held-out gap is driven by the\n  tip-to-shaft `deg_b == 2` pool in the near band JUST above `NEAR_GAP_UM = 2.15`,\n  which `colinear_cos` UNDER-SCORES: it is the AVERAGE of the two arm cosines and is\n  dragged DOWN by the noisy / laterally-offset shaft-side arm, so genuine\n  tip-to-shaft continuations with true `colinear` 0.50\u20130.75 are rejected by the\n  strict 0.75 floor. *The ONE change:* added a module constant\n  `SHAFT_COLINEAR_FLOOR = 0.50` (immediately after `TIP_TIP_COLINEAR_COS`) and a\n  THIRD tier-(D) degree-dispatch branch \u2014 for `deg_b == 2`, in addition to the\n  unchanged `colinear_cos >= MID_COLINEAR_COS` (0.75) path, accept when\n  `colinear_cos >= SHAFT_COLINEAR_FLOOR` (0.50) **AND**\n  `_through_line_cos(g, s, TANGENT_WALK_UM) >= THROUGH_LINE_COS` (0.70). *Precision\n  rationale:* the relaxed `colinear >= 0.50` floor sits **0.11 above** the **0.39**\n  in-band `deg_b == 2` false-join ceiling (the only in-band `deg_b == 2` false joins\n  are gap 2.19 / colinear 0.39 and gap 2.71 / colinear 0.10), so it keeps TRAIN\n  false at 0 on its own; the conjoined `_through_line_cos >= 0.70` is a SECOND,\n  INDEPENDENT continuity signal (bridge-bypassing tangent-to-tangent) that must\n  agree, hedging against held-out false joins in the relaxed `colinear` [0.50, 0.75)\n  range. *Why this DIFFERS from prior rejected attempts:* gen8 was a through-line\n  GAP-WINDOW extension via an OR / caliber path (no conjunction, and a window\n  change); gen17 was a caliber `rad_ratio` floor that created a held-out false\n  merge; gen11 used a per-arm `min(cos_a, cos_b)`; gen13 used `tip_tangent_cos`.\n  Here through-line is an AND PRECISION-CONJUNCTION that ENABLES a relaxed colinear\n  floor within the EXISTING (2.15, 3.0] band \u2014 no gap-window change, no caliber, no\n  per-arm min. Image NOT used (`bridge_ratio` AUC 0.52 \u2014 REAL and FALSE fully\n  overlap; the recall gap is geometric, not photometric). Tiers (A), (B), (C), the\n  outer `GAP_THRESHOLD_UM` guard, `ENUM_PARAMS`, all helper bodies, and the constant\n  values `TIP_TIP_COLINEAR_COS` (0.50), `MID_COLINEAR_COS` (0.75, tip-to-shaft\n  strict path unchanged), and `THROUGH_LINE_COS` (0.70) are untouched. Mark\n  CANDIDATE pending the held-out gate.\n- **Harness:** `candidate_split_sites` broadened from tip-to-tip to"
}
```

### 📝 text [orchestrator]

Let me verify the edits landed correctly on disk, per my established pattern.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 133,
  "limit": 4
}
```

### 📝 text [orchestrator]

Constant landed. Now verify the tier (D) dispatch block.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 388,
  "limit": 28
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Both code edits verified and syntactically sound — the third elif is correctly ordered (a `deg_b==2` site with `cc≥0.75` accepts at the second branch; `0.50≤cc<0.75` falls through to the through-line conjunction; `cc<0.50` accepts nowhere), and `g` is in scope from the top of `propose_edits`. Let me confirm the rules.md change-log entry.

### 🔧 tool call [orchestrator]: Grep

```json
{
  "pattern": "Gen 19",
  "path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "output_mode": "content",
  "-n": true
}
```

### 💭 thinking [orchestrator]



### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "offset": 137,
  "limit": 28
}
```

### 📝 text [orchestrator]

The rule (D) section and change-log entry are both in place and coherent. All edits verified on disk.

**Gen19 revision complete.** Summary:

- **State:** gen18 was **accepted** (train 461 → 507, and it transferred to held-out), so the parent is now 507/0. The gen19 report reflects that parent — no regression to fix.
- **Diagnosis:** the widening train/held-out gap (+323.6) is driven by the **tip-to-shaft (`deg_b==2`) near-band pool**. Their `colinear_cos` (the *average* of the two arm cosines) is depressed by the noisy/laterally-offset shaft-side arm, so genuine continuations with true straightness 0.50–0.75 are wrongly rejected by the strict 0.75 floor. Single-feature gates here are exhausted: colinear can't be lowered alone (0.39 in-band false ceiling), caliber failed on held-out (gen17), through-line *window-extension* failed (gen8), image is out (AUC 0.52).
- **The one change:** a **double-gated** relaxed path for `deg_b==2` inside the existing (2.15, 3.0] band — accept at `colinear_cos ≥ 0.50` **AND** `_through_line_cos ≥ 0.70`. The colinear ≥ 0.50 floor sits 0.11 above the 0.39 in-band `deg_b==2` false ceiling (keeps **train false = 0 by colinear alone**, verifiable from the report), while the bridge-bypassing through-line conjunction is an **independent** second continuity signal hedging held-out false joins in the [0.50, 0.75) colinear range. Distinct from gen8 (OR window-extension), gen17 (caliber), gen11 (per-arm min), gen13 (tip_tangent).
- **Verified on disk:** new constant `SHAFT_COLINEAR_FLOOR = 0.50` (line 135); third `elif` branch correctly ordered in tier (D) (lines 405–414); rule (D) section + "Gen 19 — CANDIDATE" change-log entry updated in rules.md. Tiers A/B/C, the outer guard, ENUM_PARAMS, all helpers, and the constants `TIP_TIP_COLINEAR_COS`/`MID_COLINEAR_COS`/`THROUGH_LINE_COS` are untouched. Marked candidate, pending the held-out gate.
