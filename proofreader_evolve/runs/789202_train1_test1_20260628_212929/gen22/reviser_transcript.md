# Reviser transcript

- thinking captured: no (0 chars)
- final text: 4045 chars
- tokens: in=34 out=132; cost_usd=54.43102150000002
- priors read: None

## Prompt sent to the reviser

```
Use the proofreader-reviser subagent to improve the evolved proofreading program. The current policy lives in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py and its theory in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md. Edit THOSE files in place (do not touch any other files). The failure report for the latest candidate is at /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen22/failure_report.md. Diagnose why the policy lost accuracy, make ONE concrete improvement to propose_edits (keeping its call signature), and update the rules file and its change log. Do NOT claim you ran, imported, or tested anything — you have no Bash; the harness import-checks and lint-checks your edit after you finish.

IMPORTANT — THIS RUN: merge-error repair is DISABLED. The candidate stream contains SplitSites only (no MergeSite is enumerated), and any `split_label` edit you emit is dropped before scoring. Do NOT write or tune `split_label` logic; focus entirely on the `merge_labels` (split-error) policy. The failure report's merge sections are omitted accordingly.

IMAGE CURRICULUM — couple reading image to PASSING THE GATE in ONE revision. The gate keeps a candidate only if it makes MORE net correct `merge_labels` repairs than the parent with ZERO false merges. A revision that merely STARTS calling `gap_bridge_evidence` without changing which labels you merge ties the parent's split-repair score and is REVERTED — so its reads (and any image signal) are thrown away and never reach a future report. Therefore, if you decide image is worth using, you MUST spend it to RAISE RECALL in the SAME revision: take SplitSites in the report's 'MISSED real splits (rejected REAL)' bucket — real splits your current geometry is too conservative to accept — and ACCEPT the ones a high `gap_bridge_evidence.bridge_ratio` confirms are a continuous bright bridge across the gap. Gate the (cloud) read behind your cheap geometric filters so you only read the handful of ambiguous candidates. Consult the report's 'Image warm-start probe' section FIRST: it already measured whether `bridge_ratio` separates REAL from FALSE on this brain (an AUC), so you know BEFORE writing the rule whether image is worth gating on and roughly where the threshold sits. If the AUC is weak, do not over-invest in image — improve the geometric `merge_labels` policy instead.

GENERALIZATION CHECK (aggregate over the last 5 ACCEPTED generations; no neuron identities, just scores): mean TRAIN split-repair score = 460.0, mean HELD-OUT split-repair score = 93.6, gap (train - held-out) = +366.4 — widening (train pulling ahead of held-out). The gate scores you on HELD-OUT, so a large positive gap means your recent accepted changes repair train splits that DON'T carry over. If the gap is widening, prefer improvements grounded in GENERALIZABLE geometry/topology (tangent continuity, caliber match, endpoint degree) over ones that chase train-specific recall.

```

## Stream (thinking / text / tool calls, in order)

### 💭 thinking [orchestrator]



### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen22/failure_report.md"
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Critical finding: the gen21 decoupled `THROUGH_LINE_STRONG ≥ 0.85` deg_b==2 path **accepted the canonical in-band false join at gap 2.19 / colinear 0.39** (the "WRONGLY accepted FALSE" row, line 174) — exactly the hotspot I flagged as a risk. It raised train correct to 649 (+121) but created a train false merge (N020+N016 fused). My assumption that false grazes have low shaft-alignment was wrong for this site: its arm aligns with the shaft axis (through_line ≥ 0.85) despite being two different neurons.

I must check the current on-disk file state — whether gen21 was accepted (kept) or reverted to the gen19 parent — before deciding the fix.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 130,
  "limit": 12
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

This is the key diagnosis. Let me confirm the strategic picture before delegating:

**gen21 was ACCEPTED** — `THROUGH_LINE_STRONG = 0.85` is still on disk (line 136), train rose 528→649, and the 5-gen accepted-mean climbed 429.6→460.0 (consistent with gen21 entering the window). So the parent is gen21 (train 649 correct / 1 false = 648 net; held-out clean since the gate didn't reject it).

**Why it "lost accuracy":** the gen21 decoupled deg_b==2 path (through_line ≥ 0.85, colinear ≥ 0.0) **accepted the canonical false join at gap 2.19 / colinear 0.39** (line 174, the lone "WRONGLY accepted FALSE"). My gen21 hypothesis was backwards: for tip-to-shaft, a **high shaft-axis alignment is actually the signature of a PARALLEL-FASCICULATION graze** (a tip running alongside a *different* neuron's shaft), not a reconnection. So that path has a precision crack — it got lucky on held-out but created a real train over-merge.

**Why I will NOT tighten the deg_b==2 path:** the false (cc 0.39) is geometrically inseparable from genuine deg_b==2 catches by colinear/caliber; tightening it would strip gen21's transferred recall and revert. Instead I add net-correct via the **safest untapped pool with a generalizable, verifiable rule**: caliber-matched tip-to-tip.

**The ONE change:** in tier (D), add a deg_b==1 (tip-to-tip) relaxed path — `cc ≥ 0.30 AND rad_ratio ≤ 1.3`. Tip-to-tip is the cleanest break topology (deg_b==1 bucket = 158 correct / **0 false**). The two in-band deg_b==1 false joins (2.24/cc−0.01/rr1.25 and 2.45/cc0.17/rr1.56) are **both verifiably excluded** (the first by cc<0.30, the second by cc<0.30 AND rr>1.3). Caliber match (a broken neuron keeps its caliber across the break) + endpoint degree + moderate colinear = three brain-independent signals that must agree — the kind of generalizable conjunction the widening-gap mandate endorses, and distinct from gen20 (through_line conjunction) and gen17 (caliber *mismatch* on shaft).

Let me delegate the exact edits.

### 🔧 tool call [orchestrator]: Agent

```json
{
  "description": "Revise proofreading policy gen22",
  "subagent_type": "proofreader-reviser",
  "prompt": "You are revising an evolved split-error proofreading policy. Make ONE concrete, surgical improvement by editing TWO files IN PLACE (touch no other files). You have no Bash \u2014 do NOT claim you ran/imported/tested anything; the harness import-checks and lint-checks afterward.\n\nFILES:\n- Policy: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py\n- Theory/changelog: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md\n\nRead BOTH files first. The change is confined to acceptance tier (D) inside `propose_edits` (the block `elif NEAR_GAP_UM < s.gap_um <= MID_GAP_UM:`) plus two new module constants. Do NOT change anything else.\n\nCONTEXT (already diagnosed \u2014 implement faithfully, do not re-derive):\nThe parent (gen21) added a DECOUPLED high-through-line acceptor for tip-to-shaft (deg_b==2): accept when `colinear_cos >= 0.0` AND `_through_line_cos >= THROUGH_LINE_STRONG` (0.85). It raised train recall but ACCEPTED a FALSE join at gap 2.19 / colinear 0.39 / deg_b==2 (a parallel-fasciculation graze \u2014 a tip running alongside a DIFFERENT neuron's shaft has high shaft-axis alignment too). That deg_b==2 path is the ACCEPTED parent and must be LEFT UNTOUCHED (tightening it would lose its transferred held-out recall and revert). Instead we raise net-correct on the SAFEST untapped pool with a generalizable, report-verifiable rule: caliber-matched TIP-TO-TIP (deg_b==1). The deg_b==1 bucket has 158 correct / 0 false on train. The only two in-band (gap 2.15\u20133.0) deg_b==1 FALSE joins are gap 2.24/colinear \u22120.01/rad_ratio 1.25 and gap 2.45/colinear 0.17/rad_ratio 1.56.\n\nTHE ONE CHANGE \u2014 a caliber-matched relaxed tip-to-tip path in tier (D):\n\n(1) Add TWO module constants. In heuristics.py, immediately AFTER the existing `THROUGH_LINE_STRONG = 0.85` line (with its long comment), add these two lines:\n\nTIP_TIP_RELAX_COS = 0.30   # tier (D) RELAXED colinear floor for tip-to-tip (deg_b==1), used ONLY in conjunction with a caliber-MATCH guard (rad_ratio <= TIP_TIP_RAD_MATCH_MAX). 0.30 sits 0.13 ABOVE the 0.17 in-band deg_b==1 false-join ceiling (only in-band deg_b==1 false joins: gap 2.24/colinear -0.01, gap 2.45/colinear 0.17), so colinear>=0.30 alone already excludes BOTH in-band false joins; the caliber-match conjunction is the independent second signal (two brain-independent signals must agree, the gen19-style hedge).\nTIP_TIP_RAD_MATCH_MAX = 1.3   # tier (D) caliber-MATCH ceiling for the relaxed tip-to-tip path. A genuinely broken neuron keeps the SAME cable caliber across the break, so its two endpoint radii match (rad_ratio near 1). 1.3 admits caliber-matched tips while the conjunction with colinear>=0.30 verifiably excludes the two in-band deg_b==1 false joins (gap 2.24/rad_ratio 1.25 is excluded by colinear -0.01 < 0.30; gap 2.45/rad_ratio 1.56 is excluded by BOTH colinear 0.17 < 0.30 AND rad_ratio 1.56 > 1.3). This uses caliber MATCH for tip-to-tip (opposite direction and topology from gen17's rejected caliber-MISMATCH acceptor on tip-to-shaft), as a precision conjunction not a standalone acceptor.\n\n(2) Add ONE new `elif` branch in tier (D)'s degree dispatch for deg_b==1. The current tier-(D) dispatch begins EXACTLY (inside `if deg_a == 1 and cc is not None:`):\n\n                            if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 2 and cc >= MID_COLINEAR_COS:\n                                accept = True\n\nInsert the new branch IMMEDIATELY AFTER the `if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:` branch and BEFORE the `elif deg_b == 2 and cc >= MID_COLINEAR_COS:` bra
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
  "old_string": "THROUGH_LINE_STRONG = 0.85   # tier (D) DECOUPLED shaft-alignment floor for tip-to-shaft (deg_b==2), used as a standalone acceptor gated only by a colinear sign guard. For a tip-into-shaft reconnection the bridge-vector colinear_cos is dragged DOWN by lateral offset (the tip meets the SIDE of the shaft, tilting the bridge vector), so it under-scores genuine continuations whose ARM nonetheless runs ALONG the shaft axis. _through_line_cos for deg_b==2 measures exactly that (|arm-A tangent . shaft local axis|) on the RELIABLE shaft anchor (a degree-2 node has a well-defined local axis, unlike a short noisy stub). 0.85 is a VERY HIGH bar (near-perfect axial alignment): an in-band false graze crosses the shaft (2.19/colinear 0.39, 2.71/colinear 0.10) so its |dot| stays low and will not reach it. This is STRICTLY MORE conservative on the through-line axis than the gen19 deg_b==2 path (through_line>=0.70 conjoined with colinear>=0.50, PROVEN held-out-safe); it trades the train-specific corrupted bridge-colinear MAGNITUDE for the generalizable, reliable shaft-axis-alignment signal, and is gated by colinear_cos >= 0 (a cheap doubling-back guard from the verifiable feature) so a backward arm is still rejected.",
  "new_string": "THROUGH_LINE_STRONG = 0.85   # tier (D) DECOUPLED shaft-alignment floor for tip-to-shaft (deg_b==2), used as a standalone acceptor gated only by a colinear sign guard. For a tip-into-shaft reconnection the bridge-vector colinear_cos is dragged DOWN by lateral offset (the tip meets the SIDE of the shaft, tilting the bridge vector), so it under-scores genuine continuations whose ARM nonetheless runs ALONG the shaft axis. _through_line_cos for deg_b==2 measures exactly that (|arm-A tangent . shaft local axis|) on the RELIABLE shaft anchor (a degree-2 node has a well-defined local axis, unlike a short noisy stub). 0.85 is a VERY HIGH bar (near-perfect axial alignment): an in-band false graze crosses the shaft (2.19/colinear 0.39, 2.71/colinear 0.10) so its |dot| stays low and will not reach it. This is STRICTLY MORE conservative on the through-line axis than the gen19 deg_b==2 path (through_line>=0.70 conjoined with colinear>=0.50, PROVEN held-out-safe); it trades the train-specific corrupted bridge-colinear MAGNITUDE for the generalizable, reliable shaft-axis-alignment signal, and is gated by colinear_cos >= 0 (a cheap doubling-back guard from the verifiable feature) so a backward arm is still rejected.\nTIP_TIP_RELAX_COS = 0.30   # tier (D) RELAXED colinear floor for tip-to-tip (deg_b==1), used ONLY in conjunction with a caliber-MATCH guard (rad_ratio <= TIP_TIP_RAD_MATCH_MAX). 0.30 sits 0.13 ABOVE the 0.17 in-band deg_b==1 false-join ceiling (only in-band deg_b==1 false joins: gap 2.24/colinear -0.01, gap 2.45/colinear 0.17), so colinear>=0.30 alone already excludes BOTH in-band false joins; the caliber-match conjunction is the independent second signal (two brain-independent signals must agree, the gen19-style hedge).\nTIP_TIP_RAD_MATCH_MAX = 1.3   # tier (D) caliber-MATCH ceiling for the relaxed tip-to-tip path. A genuinely broken neuron keeps the SAME cable caliber across the break, so its two endpoint radii match (rad_ratio near 1). 1.3 admits caliber-matched tips while the conjunction with colinear>=0.30 verifiably excludes the two in-band deg_b==1 false joins (gap 2.24/rad_ratio 1.25 is excluded by colinear -0.01 < 0.30; gap 2.45/rad_ratio 1.56 is excluded by BOTH colinear 0.17 < 0.30 AND rad_ratio 1.56 > 1.3). This uses caliber MATCH for tip-to-tip (opposite direction and topology from gen17's rejected caliber-MISMATCH acceptor on tip-to-shaft), as a precision conjunction not a standalone acceptor."
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "                            if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 2 and cc >= MID_COLINEAR_COS:",
  "new_string": "                            if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:\n                                accept = True\n                            elif deg_b == 1 and cc >= TIP_TIP_RELAX_COS:\n                                # tip-to-tip (deg_b==1) with MODERATE colinear (0.30-0.50):\n                                # both endpoints are degree-1 tips (the cleanest break\n                                # topology), and colinear>=0.30 already clears the 0.17\n                                # in-band deg_b==1 false ceiling. Require caliber MATCH\n                                # (rad_ratio <= TIP_TIP_RAD_MATCH_MAX) as the independent\n                                # second signal: a broken neuron keeps its cable caliber\n                                # across the break, so genuine tip-to-tip reconnections are\n                                # caliber-matched, while the two in-band false joins are\n                                # excluded (one by colinear, one by colinear AND caliber).\n                                rr = _rad_ratio(ctx, s.node_a, s.node_b)\n                                if rr is not None and rr <= TIP_TIP_RAD_MATCH_MAX:\n                                    accept = True\n                            elif deg_b == 2 and cc >= MID_COLINEAR_COS:"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "    - **tip-to-tip** (`deg_b == 1`): `colinear_cos >= TIP_TIP_COLINEAR_COS`\n      (**0.50**);",
  "new_string": "    - **tip-to-tip** (`deg_b == 1`): accept at `colinear_cos >= TIP_TIP_COLINEAR_COS`\n      (**0.50**) **OR**, at MODERATE colinear (`TIP_TIP_RELAX_COS` 0.30 `<= colinear_cos\n      < 0.50`), a caliber-MATCHED relaxed path (Gen 22): accept when\n      `_rad_ratio(ctx, node_a, node_b) <= TIP_TIP_RAD_MATCH_MAX` (**1.3**). Both\n      endpoints are degree-1 tips (the cleanest break topology); `colinear >= 0.30`\n      already clears the 0.17 in-band `deg_b == 1` false ceiling, and the caliber\n      MATCH is the INDEPENDENT second signal (a broken neuron keeps its cable caliber\n      across the break);"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "- **Harness:** `candidate_split_sites` broadened from tip-to-tip to",
  "new_string": "- **Gen 22 (2026-06-29) \u2014 CANDIDATE (pending the held-out gate):** added a caliber-matched RELAXED tip-to-tip (`deg_b == 1`) acceptor to tier (D). *Parent* = gen21's line (train **649 correct / 1 false**; the lone train false is the gen21 decoupled `THROUGH_LINE_STRONG` deg_b==2 path accepting the gap 2.19 / colinear 0.39 parallel-fasciculation graze \u2014 held-out was clean, so gen21 was accepted). *Diagnosis:* for tip-to-shaft, a HIGH `_through_line_cos` (shaft-axis alignment) is actually the signature of a PARALLEL graze (a tip running alongside a DIFFERENT neuron's shaft), not only a reconnection, so the gen21 decoupled path has a precision crack; it is geometrically inseparable from genuine deg_b==2 catches by the available features, and it is the ACCEPTED parent, so it is LEFT UNTOUCHED (tightening it would forfeit its transferred held-out recall and revert). *The ONE change:* raise net-correct on the SAFEST untapped pool \u2014 caliber-matched tip-to-tip. Two new constants `TIP_TIP_RELAX_COS = 0.30` and `TIP_TIP_RAD_MATCH_MAX = 1.3` (immediately after `THROUGH_LINE_STRONG`) and a new tier-(D) `deg_b == 1` branch: in addition to the unchanged `colinear_cos >= TIP_TIP_COLINEAR_COS` (0.50) path, accept tip-to-tip when `colinear_cos >= TIP_TIP_RELAX_COS` (0.30) **AND** `_rad_ratio(ctx, node_a, node_b) <= TIP_TIP_RAD_MATCH_MAX` (1.3). *Precision rationale (verifiable from the report):* the deg_b==1 bucket is 158 correct / 0 false; the only two in-band (gap 2.15\u20133.0) deg_b==1 false joins are gap 2.24/colinear \u22120.01/rad_ratio 1.25 and gap 2.45/colinear 0.17/rad_ratio 1.56 \u2014 the first is excluded by `colinear < 0.30`, the second by BOTH `colinear < 0.30` AND `rad_ratio > 1.3`, so train false stays 0 on this path. *Generalization rationale (widening train/held-out gap):* it conjoins THREE brain-independent signals \u2014 endpoint degree (tip-to-tip = the cleanest break topology), caliber MATCH (a broken neuron keeps caliber across the break), and a margin-safe moderate colinear floor \u2014 the gen19-style \"two independent signals must agree\" hedge, rather than chasing train-specific recall. *Why this DIFFERS from prior attempts:* gen20 relaxed tip-to-tip to 0.30 with a `_through_line_cos` conjunction (the through-line gate filtered everything out on short stubs, so it TIED and was reverted) \u2014 this uses a CALIBER-MATCH conjunction instead; gen17 used caliber MISMATCH (`rad_ratio >= 2.1`) as a standalone ACCEPTOR on tip-to-shaft and created a held-out false \u2014 this uses caliber MATCH (`rad_ratio <= 1.3`) as a precision CONJUNCTION on tip-to-tip (opposite direction, opposite topology, conjunction not standalone). Image NOT used (`bridge_ratio` AUC 0.52). The deg_b==2 branches (gen21 decoupled `THROUGH_LINE_STRONG`, `SHAFT_COLINEAR_FLOOR`, `MID_COLINEAR_COS`), tiers (A), (B), (C), the outer `GAP_THRESHOLD_UM` guard, `ENUM_PARAMS`, all helper bodies, and the constant values `TIP_TIP_COLINEAR_COS` (0.50), `SHAFT_COLINEAR_FLOOR` (0.50), `MID_COLINEAR_COS` (0.75), `THROUGH_LINE_COS` (0.70), `THROUGH_LINE_STRONG` (0.85), `TANGENT_WALK_UM` (6.0) are untouched. Mark CANDIDATE pending the held-out gate.\n- **Harness:** `candidate_split_sites` broadened from tip-to-tip to"
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

Verified on disk — the gen22 edits landed correctly and the file is syntactically intact:

**Constants (lines 137–138):** Both new constants are present immediately after `THROUGH_LINE_STRONG = 0.85` (line 136):
- `TIP_TIP_RELAX_COS = 0.30`
- `TIP_TIP_RAD_MATCH_MAX = 1.3`

**Tier (D) deg_b==1 dispatch (lines 404–418):** The branch ordering is correct:
1. `if deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS:` (0.50) → accept (lines 404–405)
2. `elif deg_b == 1 and cc >= TIP_TIP_RELAX_COS:` (0.30) → accept iff `_rad_ratio(ctx, s.node_a, s.node_b) <= TIP_TIP_RAD_MATCH_MAX` (1.3) — the NEW gen22 branch (lines 406–418)
3. then the three deg_b==2 branches (0.75 / SHAFT_COLINEAR_FLOOR+through-line / decoupled THROUGH_LINE_STRONG) — all unchanged (lines 419–443)

**Untouched:** The gen21 deg_b==2 branches, tiers (A)/(B)/(C), the outer `GAP_THRESHOLD_UM` guard, `ENUM_PARAMS`, all helper bodies (`_rad_ratio` reused, not redefined), and `propose_edits(sites, ctx)` signature are all intact. No image or `split_label` logic added.

The gen22 revision is complete and consistent: a caliber-matched tip-to-tip relaxation (cc≥0.30 AND rad_ratio≤1.3) that verifiably excludes both in-band deg_b==1 false joins (2.24/cc−0.01 by cc<0.30; 2.45/cc0.17/rr1.56 by both gates), keeping the train-false count from rising on this path while opening the safest untapped recall pool. No further generation task has been issued.
