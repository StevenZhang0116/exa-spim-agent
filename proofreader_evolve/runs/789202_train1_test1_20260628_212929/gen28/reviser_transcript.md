# Reviser transcript

- thinking captured: no (0 chars)
- final text: 4567 chars
- tokens: in=50 out=159; cost_usd=69.57630475
- priors read: None

## Prompt sent to the reviser

```
Use the proofreader-reviser subagent to improve the evolved proofreading program. The current policy lives in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py and its theory in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md. Edit THOSE files in place (do not touch any other files). The failure report for the latest candidate is at /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen28/failure_report.md. Diagnose why the policy lost accuracy, make ONE concrete improvement to propose_edits (keeping its call signature), and update the rules file and its change log. Do NOT claim you ran, imported, or tested anything — you have no Bash; the harness import-checks and lint-checks your edit after you finish.

IMPORTANT — THIS RUN: merge-error repair is DISABLED. The candidate stream contains SplitSites only (no MergeSite is enumerated), and any `split_label` edit you emit is dropped before scoring. Do NOT write or tune `split_label` logic; focus entirely on the `merge_labels` (split-error) policy. The failure report's merge sections are omitted accordingly.

IMAGE CURRICULUM — couple reading image to PASSING THE GATE in ONE revision. The gate keeps a candidate only if it makes MORE net correct `merge_labels` repairs than the parent with ZERO false merges. A revision that merely STARTS calling `gap_bridge_evidence` without changing which labels you merge ties the parent's split-repair score and is REVERTED — so its reads (and any image signal) are thrown away and never reach a future report. Therefore, if you decide image is worth using, you MUST spend it to RAISE RECALL in the SAME revision: take SplitSites in the report's 'MISSED real splits (rejected REAL)' bucket — real splits your current geometry is too conservative to accept — and ACCEPT the ones a high `gap_bridge_evidence.bridge_ratio` confirms are a continuous bright bridge across the gap. Gate the (cloud) read behind your cheap geometric filters so you only read the handful of ambiguous candidates. Consult the report's 'Image warm-start probe' section FIRST: it already measured whether `bridge_ratio` separates REAL from FALSE on this brain (an AUC), so you know BEFORE writing the rule whether image is worth gating on and roughly where the threshold sits. If the AUC is weak, do not over-invest in image — improve the geometric `merge_labels` policy instead.

GENERALIZATION CHECK (aggregate over the last 5 ACCEPTED generations; no neuron identities, just scores): mean TRAIN split-repair score = 509.6, mean HELD-OUT split-repair score = 103.6, gap (train - held-out) = +406.0 — widening (train pulling ahead of held-out). The gate scores you on HELD-OUT, so a large positive gap means your recent accepted changes repair train splits that DON'T carry over. If the gap is widening, prefer improvements grounded in GENERALIZABLE geometry/topology (tangent continuity, caliber match, endpoint degree) over ones that chase train-specific recall.

Attempts already tried against the CURRENT policy (do NOT repeat these — they did not beat it). The number is the held-out SPLIT-REPAIR score delta vs the parent (correct - false; the gate's actual fitness, an integer count — NOT Edge Accuracy):
  - gen24: +22 -17 lines; gen23 was **ACCEPTED** — the parent train score rose 649→662 correct (the deg_b<2 tip-to-tip bucket went 158→171, +13 co -> held-out split-repair +7 vs parent (rejected)
  - gen25: +2 -1 lines; gen24 was **REJECTED** — held-out +7 net but it introduced a held-out false merge (gate requires zero), so it reverted;  -> held-out split-repair +0 vs parent (rejected)
  - gen26: +32 -0 lines; gen25 **TIED** (held-out +0 → reverted); parent remains gen23. Let me read the gen26 report to confirm state and look fo -> held-out split-repair +0 vs parent (rejected)
  - gen27: +19 -1 lines; The report shows train accepted REAL = 662 (identical to the gen23 parent), confirming **gen26 added zero train accepts  -> held-out split-repair -1 vs parent (rejected)

Propose a DIFFERENT improvement from any listed above.
```

## Stream (thinking / text / tool calls, in order)

### 💭 thinking [orchestrator]



### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen28/failure_report.md"
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

The gen28 report is identical to gen27/gen23-parent (train 662/1), confirming **gen27 was rejected (held-out −1: the shorter-walk max-through-line admitted a held-out false merge) and reverted to the gen23 parent**. Let me confirm the disk reverted cleanly before diagnosing.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 99,
  "limit": 4
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Confirmed — `TIP_TANGENT_WALK_UM` is gone (line 102 is back to the comment); gen27 reverted cleanly to the gen23 parent. Let me verify the deg_b==2 branches are in their parent form before delegating.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 423,
  "limit": 32
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Confirmed: the gen21 deg_b==2 branch (lines 437–449) is in clean parent form. 

**Diagnosis:** Parent is gen23 (train 662/1, held-out clean). The last four attempts all targeted tip-to-tip (deg_b==1) or estimation/band tweaks and failed: gen24 (caliber-mismatch tip-to-tip → held-out false), gen25 (lowered through-line bar → tied), gen26 (widened gap band → empty/tied), gen27 (shorter-walk max-through-line → held-out false). Image AUC is again 0.52 (ruled out). The lesson: loosening single signals on the tip-to-tip pool either ties (pool doesn't transfer) or admits held-out falses. The ONE thing that GENERALIZED was gen23's **two-signal conjunction (caliber match + tangent continuity)** — but it was only ever applied to tip-to-tip. The large deg_b==2 (tip-to-shaft) low-colinear missed pool is gated only by the single-signal gen21 path (through_line≥0.85 alone), which carries no caliber gate and produced the persistent train false. That's the untried, directive-aligned move: extend the proven gen23 caliber-matched-continuity recipe to the tip-to-shaft class. Let me delegate.

### 🔧 tool call [orchestrator]: Agent

```json
{
  "description": "Revise proofreader policy gen28",
  "subagent_type": "proofreader-reviser",
  "prompt": "You are improving an evolved split-error proofreading policy. There are EXACTLY TWO files you may edit, IN PLACE \u2014 touch no others:\n- POLICY: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py\n- THEORY/CHANGELOG: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md\n\nHARD CONSTRAINTS (do not violate):\n- Keep propose_edits(sites, ctx) -> list call signature unchanged.\n- Do NOT claim you ran, imported, or tested anything \u2014 you have no Bash. The harness import-checks and lint-checks your edit after you finish, so keep it syntactically clean and import-safe.\n- THIS RUN merge-error repair is DISABLED: any split_label edit is dropped before scoring. Do NOT write or tune split_label logic. Focus entirely on the merge_labels (split-repair) policy.\n- Read heuristics.py FRESH from disk first. It is the gen23 parent (gen27 was reverted). Confirm this exact structure before editing:\n  * Constant TANGENT_WALK_UM = 6.0; constant TIP_TIP_RAD_MATCH_MAX = 1.3; constant THROUGH_LINE_COS = 0.7; constant THROUGH_LINE_STRONG = 0.85. There must be NO constant named TIP_TANGENT_WALK_UM or SHAFT_RAD_MATCH_MAX yet.\n  * Inside tier (D) (`elif NEAR_GAP_UM < s.gap_um <= MID_GAP_UM:`), the LAST branch of the deg_a==1 dispatch is:\n        elif deg_b == 2 and cc >= 0.0:\n            # ...comment...\n            tl = _through_line_cos(g, s, TANGENT_WALK_UM)\n            if tl is not None and tl >= THROUGH_LINE_STRONG:\n                accept = True\n    This is the \"gen21 decoupled tip-to-shaft\" path. If you do not find exactly this, STOP and report what you see.\n\nTHE ONE IMPROVEMENT TO MAKE (this exact change, nothing more):\nExtend the PROVEN gen23 two-signal recipe \u2014 caliber MATCH conjoined with tangent CONTINUITY \u2014 to the tip-to-shaft (deg_b==2) low-colinear pool, which currently has NO caliber gate. Do it by adding a SECOND acceptance path INSIDE the existing gen21 `elif deg_b == 2 and cc >= 0.0:` branch body (do NOT add a new sibling elif after it \u2014 the gen21 condition already consumes every deg_b==2/cc>=0 site, so a sibling would be DEAD CODE).\n\n1. Add a module constant near TIP_TIP_RAD_MATCH_MAX, e.g.:\n   SHAFT_RAD_MATCH_MAX = 1.15   # Gen 28: caliber-match ceiling for the NEW tip-to-shaft (deg_b==2) low-colinear acceptor that conjoins caliber match with through-line continuity (the gen23 recipe that GENERALIZED to held-out, extended from tip-to-tip to tip-to-shaft via endpoint degree). TIGHTER than the tip-to-tip TIP_TIP_RAD_MATCH_MAX=1.3 because tip-to-shaft is the riskier geometry AND the in-band deg_b==2 FALSE join at gap 2.71/colinear 0.10 has rad_ratio 1.20, so a 1.15 ceiling provably EXCLUDES it (and the other in-band deg_b==2 false at gap 2.19/colinear 0.39 has rad_ratio 1.97, also excluded). A broken neuron keeps the SAME cable caliber across the break, so rad_ratio ~1; this is a brain-independent ratio so it should transfer.\n2. Replace the gen21 branch body so it accepts on EITHER the existing very-high through-line path OR the new caliber-matched moderate-through-line path:\n        elif deg_b == 2 and cc >= 0.0:\n            # tip-to-shaft, LOW bridge-vector colinear (0.0-0.50). Two acceptance\n            # paths (OR):\n            #  (i) gen21: VERY HIGH shaft-axis through-line alone (>=0.85). Kept\n            #      verbatim so recall is strictly NON-DECREASING (no currently-\n            #      accepted real is lost).\n            #  (ii) Gen 28: the PROVEN gen23 two-signal conjunction (caliber MATCH +\n            #      tangent CONTINUITY), extended from tip-to-tip to tip-to-shaft.\n            #      A genuine same-caliber tip-into-shaft reconnectio
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
  "old_string": "TIP_TIP_RAD_MATCH_MAX = 1.3  # tier (D) caliber-match ceiling for the LOW-colinear TIP-TO-TIP (deg_b==1) acceptor (Gen 23). A broken neuron keeps the SAME cable caliber across the break, so rad_ratio ~1; a parallel-graze of two different neurites has mismatched caliber. 1.3 excludes the in-band tip-to-tip false join at gap 2.45/colinear 0.17/rr1.56 (rr 1.56 > 1.3). Brain-independent ratio (not a per-brain distance), so it should transfer.",
  "new_string": "TIP_TIP_RAD_MATCH_MAX = 1.3  # tier (D) caliber-match ceiling for the LOW-colinear TIP-TO-TIP (deg_b==1) acceptor (Gen 23). A broken neuron keeps the SAME cable caliber across the break, so rad_ratio ~1; a parallel-graze of two different neurites has mismatched caliber. 1.3 excludes the in-band tip-to-tip false join at gap 2.45/colinear 0.17/rr1.56 (rr 1.56 > 1.3). Brain-independent ratio (not a per-brain distance), so it should transfer.\nSHAFT_RAD_MATCH_MAX = 1.15   # Gen 28: caliber-match ceiling for the NEW tip-to-shaft (deg_b==2) low-colinear acceptor that conjoins caliber match with through-line continuity (the gen23 recipe that GENERALIZED to held-out, extended from tip-to-tip to tip-to-shaft via endpoint degree). TIGHTER than the tip-to-tip TIP_TIP_RAD_MATCH_MAX=1.3 because tip-to-shaft is the riskier geometry AND the in-band deg_b==2 FALSE join at gap 2.71/colinear 0.10 has rad_ratio 1.20, so a 1.15 ceiling provably EXCLUDES it (and the other in-band deg_b==2 false at gap 2.19/colinear 0.39 has rad_ratio 1.97, also excluded). A broken neuron keeps the SAME cable caliber across the break, so rad_ratio ~1; this is a brain-independent ratio so it should transfer."
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "                            elif deg_b == 2 and cc >= 0.0:\n                                # tip-to-shaft with LOW bridge-vector colinear (0.0-0.50):\n                                # colinear_cos is corrupted by lateral offset for a\n                                # tip-into-the-SIDE-of-shaft join, so fall back to the\n                                # RELIABLE, orthogonal shaft-axis alignment signal. Require a\n                                # VERY HIGH through-line (arm A runs along the shaft's local\n                                # axis) to confirm a genuine continuation. The colinear>=0.0\n                                # guard rejects a backward/doubling-back arm; the 0.85 bar is\n                                # stricter on the through-line axis than the gen19 path, so it\n                                # keeps the held-out precision the gen19 conjunction proved.\n                                tl = _through_line_cos(g, s, TANGENT_WALK_UM)\n                                if tl is not None and tl >= THROUGH_LINE_STRONG:\n                                    accept = True",
  "new_string": "                            elif deg_b == 2 and cc >= 0.0:\n                                # tip-to-shaft, LOW bridge-vector colinear (0.0-0.50). Two acceptance\n                                # paths (OR):\n                                #  (i) gen21: VERY HIGH shaft-axis through-line alone (>=0.85). Kept\n                                #      verbatim so recall is strictly NON-DECREASING (no currently-\n                                #      accepted real is lost).\n                                #  (ii) Gen 28: the PROVEN gen23 two-signal conjunction (caliber MATCH +\n                                #      tangent CONTINUITY), extended from tip-to-tip to tip-to-shaft.\n                                #      A genuine same-caliber tip-into-shaft reconnection has its arm\n                                #      running along the shaft axis (through-line >= THROUGH_LINE_COS=0.70)\n                                #      AND matched caliber (rad_ratio <= SHAFT_RAD_MATCH_MAX=1.15). This\n                                #      admits same-caliber reconnections at moderate through-line (0.70-0.85)\n                                #      that path (i) misses. FALSE-SAFE: the two in-band deg_b==2 FALSE\n                                #      joins are EXCLUDED from path (ii) by caliber \u2014 gap 2.19/colinear\n                                #      0.39/rr1.97 and gap 2.71/colinear 0.10/rr1.20 both have rad_ratio\n                                #      > 1.15 \u2014 so this path adds ZERO new train false (the lone existing\n                                #      train false at 2.19 stays caught only by path (i), count unchanged).\n                                tl = _through_line_cos(g, s, TANGENT_WALK_UM)\n                                rr = _rad_ratio(ctx, s.node_a, s.node_b)\n                                if tl is not None and (\n                                        tl >= THROUGH_LINE_STRONG\n                                        or (tl >= THROUGH_LINE_COS\n                                            and rr is not None and rr <= SHAFT_RAD_MATCH_MAX)):\n                                    accept = True"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "- **Harness:** `candidate_split_sites` broadened from tip-to-tip to",
  "new_string": "- **Gen 28 (2026-06-29) \u2014 CANDIDATE (pending the held-out gate):** extended the\n  PROVEN gen23 two-signal recipe (caliber MATCH conjoined with tangent CONTINUITY)\n  from tip-to-tip to the tip-to-shaft (`deg_b == 2`) LOW-colinear pool inside tier\n  (D). *Parent* = the gen23 line, **662 correct / 1 train-false** (held-out clean).\n  *Diagnosis:* gen27 was REVERTED \u2014 its shorter-walk MAX through-line for tip-to-tip\n  admitted a held-out false; image AUC remains 0.52 (no photometric separation, so\n  the recall gap is geometric). The large remaining MISSED pool is the\n  `deg_b == 2` LOW-colinear near-band, which the gen21 branch gates ONLY on the\n  single-signal through-line `>= THROUGH_LINE_STRONG = 0.85` path with NO caliber\n  gate \u2014 so genuine same-caliber tip-into-shaft reconnections at MODERATE through-line\n  (0.70\u20130.85) are missed. *The ONE change:* (1) new module constant\n  `SHAFT_RAD_MATCH_MAX = 1.15` (immediately after `TIP_TIP_RAD_MATCH_MAX`); (2) the\n  gen21 `elif deg_b == 2 and cc >= 0.0:` branch body now accepts on EITHER the\n  existing path (i) `through_line >= THROUGH_LINE_STRONG` (0.85, kept VERBATIM) OR\n  the new path (ii) `through_line >= THROUGH_LINE_COS` (0.70) AND `rad_ratio <=\n  SHAFT_RAD_MATCH_MAX` (1.15). NO new sibling elif was added \u2014 the gen21\n  `deg_b == 2 and cc >= 0.0` condition already consumes every such site, so a sibling\n  would be dead code; the OR-path lives INSIDE that branch body. *Why it generalizes:*\n  it REUSES the only recipe that has demonstrably carried to held-out (gen23: caliber\n  match + continuity, accepted and transferred), applied to a NEW endpoint-degree\n  class via endpoint topology \u2014 the endorsed generalizable axis (tangent continuity,\n  caliber match, endpoint degree) over train-specific colinear magnitude. *Strictly\n  recall NON-DECREASING and zero new train false:* path (i) is gen21 verbatim so no\n  currently-accepted real is lost; both in-band `deg_b == 2` FALSE joins are EXCLUDED\n  from path (ii) by caliber INDEPENDENT of their through-line value \u2014 gap 2.19 /\n  colinear 0.39 / rr 1.97 and gap 2.71 / colinear 0.10 / rr 1.20 both have\n  `rad_ratio > 1.15` \u2014 so train correct can only tie or rise and train false stays 1\n  (the lone existing train false at 2.19 stays caught only by path (i), count\n  unchanged). *Why this DIFFERS from gen24/25/26/27:* gen24 was a caliber-MISMATCH\n  tip-to-tip acceptor (held-out false, REJECTED); gen25 lowered the tip-to-tip\n  through-line bar 0.70 \u2192 0.50 (TIED); gen26 widened the tip-to-tip gap band to\n  (3.0, 4.0] (empty, TIED); gen27 was a shorter-walk MAX through-line for tip-to-tip\n  (held-out false, REJECTED). All four were tip-to-tip or estimation/band tweaks;\n  NONE added a caliber gate to the tip-to-shaft (`deg_b == 2`) class. *Honest tie\n  risk:* genuine tip-to-shaft reconnections are often caliber-MISMATCHED (a thin\n  distal tip rejoining a thick parent shaft, `rad_ratio` up to ~2.8 \u2014 see the Gen-9\n  note), so the `rad_ratio <= 1.15` gate admits only the SAME-caliber minority; that\n  pool may be sparse and the candidate may TIE (held-out +0 \u2192 reverted). That is\n  acceptable and low-risk \u2014 it cannot introduce a train false and, being the proven\n  two-signal conjunction, is unlikely to introduce a held-out false. Image NOT used\n  (`bridge_ratio` AUC 0.52). Tiers (A), (B), (C), the outer `GAP_THRESHOLD_UM` guard,\n  `ENUM_PARAMS`, all helper bodies (`_through_line_cos` and `_rad_ratio` are REUSED),\n  every tip-to-tip branch, the `deg_b == 2 and cc >= MID_COLINEAR_COS` and\n  `deg_b == 2 and cc >= SHAFT_COLINEAR_FLOOR` branches, and 
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 136,
  "limit": 5
}
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 438,
  "limit": 28
}
```

### 📝 text [orchestrator]

Let me verify the edits on disk.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 136,
  "limit": 3
}
```

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 438,
  "limit": 28
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

The code edits are verified — `SHAFT_RAD_MATCH_MAX = 1.15` added (line 137) and the gen21 deg_b==2 branch now accepts on the OR of the verbatim through_line≥0.85 path and the new caliber-matched through_line≥0.70 path (lines 455–461). Let me confirm the rules.md changelog entry.

### 🔧 tool call [orchestrator]: Grep

```json
{
  "pattern": "Gen 28",
  "path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "output_mode": "content",
  "-n": true,
  "-A": 6
}
```

### 📝 text [orchestrator]

The gen28 task is complete and all edits verified on disk.

## Summary — Gen 28

**Diagnosis:** Train accepted = 662/1 (identical to the gen23 parent) → **gen27 was rejected (held-out −1: the shorter-walk max-through-line admitted a held-out false merge) and reverted**. Image AUC is again 0.52 (no photometric separation → the recall gap is geometric, image ruled out). The last four attempts (gen24–27) all targeted the tip-to-tip class or estimation/band tweaks and all failed/tied. The one thing that ever *generalized* was gen23's two-signal conjunction (**caliber match + tangent continuity**) — but it was only ever applied to tip-to-tip. The large remaining missed pool is the tip-to-shaft (deg_b==2) low-colinear near-band, gated only by the single-signal gen21 path (through_line≥0.85, no caliber gate).

**The one improvement:** Extended the proven gen23 caliber-matched-continuity recipe to the deg_b==2 class, as a **second OR-path inside the existing gen21 branch** (not a sibling elif, which would be dead code):
- New constant `SHAFT_RAD_MATCH_MAX = 1.15` (tighter than tip-to-tip's 1.3 — see below).
- The deg_b==2/cc≥0 branch now accepts on `through_line ≥ 0.85` (gen21, verbatim) **OR** (`through_line ≥ 0.70` **AND** `rad_ratio ≤ 1.15`).

**Why it's sound / generalizable:**
- Strictly **recall non-decreasing** (path (i) is gen21 verbatim — no current accept lost), so train correct can only tie or rise.
- **Zero new train false** by construction: both in-band deg_b==2 falses (2.19/0.39/rr1.97 and 2.71/0.10/rr1.20) have rad_ratio > 1.15, so the caliber gate excludes them from the new path regardless of their through-line — the 1.15 ceiling was set specifically below the 2.71 false's rr1.20.
- Reuses the only recipe demonstrated to transfer to held-out (gen23), applied to a new endpoint-degree class — exactly the directive's favored generalizable axes (continuity, caliber match, endpoint degree).
- **Distinct** from gen24 (caliber-mismatch tip-to-tip → held-out false), gen25 (lowered tip-to-tip bar → tied), gen26 (widened gap band → empty/tied), gen27 (shorter-walk → held-out false): none added a caliber gate to the tip-to-shaft class.

**Honest risk (noted in rules.md):** genuine tip-to-shaft reconnections are often caliber-mismatched (thin tip into thick shaft, rr up to ~2.8), so rr≤1.15 admits only the same-caliber minority — the pool may be sparse and the candidate may tie and revert. But it cannot introduce a train false and, being the proven two-signal conjunction, is unlikely to introduce a held-out false.

Both files edited in place; rules.md has a dated Gen 28 CANDIDATE changelog entry (header/objective/criteria and prior entries intact).
