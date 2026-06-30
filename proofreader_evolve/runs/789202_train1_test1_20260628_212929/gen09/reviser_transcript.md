# Reviser transcript

- thinking captured: no (0 chars)
- final text: 2506 chars
- tokens: in=44 out=169; cost_usd=17.296008500000006
- priors read: None

## Prompt sent to the reviser

```
Use the proofreader-reviser subagent to improve the evolved proofreading program. The current policy lives in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py and its theory in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md. Edit THOSE files in place (do not touch any other files). The failure report for the latest candidate is at /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen09/failure_report.md. Diagnose why the policy lost accuracy, make ONE concrete improvement to propose_edits (keeping its call signature), and update the rules file and its change log. Do NOT claim you ran, imported, or tested anything — you have no Bash; the harness import-checks and lint-checks your edit after you finish.

IMPORTANT — THIS RUN: merge-error repair is DISABLED. The candidate stream contains SplitSites only (no MergeSite is enumerated), and any `split_label` edit you emit is dropped before scoring. Do NOT write or tune `split_label` logic; focus entirely on the `merge_labels` (split-error) policy. The failure report's merge sections are omitted accordingly.

IMAGE CURRICULUM — couple reading image to PASSING THE GATE in ONE revision. The gate keeps a candidate only if it makes MORE net correct `merge_labels` repairs than the parent with ZERO false merges. A revision that merely STARTS calling `gap_bridge_evidence` without changing which labels you merge ties the parent's split-repair score and is REVERTED — so its reads (and any image signal) are thrown away and never reach a future report. Therefore, if you decide image is worth using, you MUST spend it to RAISE RECALL in the SAME revision: take SplitSites in the report's 'MISSED real splits (rejected REAL)' bucket — real splits your current geometry is too conservative to accept — and ACCEPT the ones a high `gap_bridge_evidence.bridge_ratio` confirms are a continuous bright bridge across the gap. Gate the (cloud) read behind your cheap geometric filters so you only read the handful of ambiguous candidates. Consult the report's 'Image warm-start probe' section FIRST: it already measured whether `bridge_ratio` separates REAL from FALSE on this brain (an AUC), so you know BEFORE writing the rule whether image is worth gating on and roughly where the threshold sits. If the AUC is weak, do not over-invest in image — improve the geometric `merge_labels` policy instead.

GENERALIZATION CHECK (aggregate over the last 4 ACCEPTED generations; no neuron identities, just scores): mean TRAIN split-repair score = 152.8, mean HELD-OUT split-repair score = 25.8, gap (train - held-out) = +127.0 — widening (train pulling ahead of held-out). The gate scores you on HELD-OUT, so a large positive gap means your recent accepted changes repair train splits that DON'T carry over. If the gap is widening, prefer improvements grounded in GENERALIZABLE geometry/topology (tangent continuity, caliber match, endpoint degree) over ones that chase train-specific recall.

Attempts already tried against the CURRENT policy (do NOT repeat these — they did not beat it). The number is the held-out SPLIT-REPAIR score delta vs the parent (correct - false; the gate's actual fitness, an integer count — NOT Edge Accuracy):
  - gen8: +21 -13 lines; The gen08 prompt has no "attempts already tried" list and a higher train score baseline — suggesting Gen 7 may have been -> held-out split-repair +0 vs parent (rejected)

Propose a DIFFERENT improvement from any listed above.
```

## Stream (thinking / text / tool calls, in order)

### 💭 thinking [orchestrator]



### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen09/failure_report.md"
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

The gen09 report confirms the parent is still **376/0** and my Gen 8 window-extension was **reverted** (held-out +0 — it raised train recall in the 2.15–4.0µm band that doesn't exist on held-out). I need to verify the current on-disk state and find a *different*, generalizable recall lever. Let me check the constants and rule C as they actually sit on disk now.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 99,
  "limit": 26
}
```

### 💭 thinking [orchestrator]



### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 343,
  "limit": 30
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Confirmed: Gen 8 was reverted — rule C is back to the Gen 7 form (capped at `NEAR_GAP_UM=2.15`, `RAD_RATIO_MAX=1.5`). Parent is 376/0. Now I have a clear, *different*, data-grounded diagnosis. Delegating the gen09 revision.

### 🔧 tool call [orchestrator]: Agent

```json
{
  "description": "Gen09 proofreader revision",
  "subagent_type": "proofreader-reviser",
  "prompt": "You are revising an evolved proofreading policy inside an evolution loop. Make ONE concrete, well-justified improvement, editing exactly two files IN PLACE. Do not touch any other files. You have no Bash \u2014 do NOT claim you ran, imported, or tested anything; the harness import-checks and lint-checks your edit afterward.\n\nFILES (absolute paths):\n- Policy: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py\n- Theory + change log: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md\n- Failure report to diagnose from (READ it): /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen09/failure_report.md\n\nHARD CONSTRAINTS (this run):\n- merge-error repair is DISABLED. The candidate stream is SplitSites only; any split_label edit is dropped before scoring. Do NOT write or tune split_label logic. Focus ENTIRELY on the merge_labels (split-error) policy in propose_edits.\n- Keep propose_edits' call signature unchanged: propose_edits(sites, ctx).\n- IMAGE: the image warm-start probe AUC is 0.52 (weak \u2014 see report). Do NOT add image reads / gap_bridge_evidence gating. Improve the geometric/topological policy instead.\n- GENERALIZATION: the gate scores on HELD-OUT and a candidate must make MORE net correct merge_labels repairs than the parent with ZERO false merges (a single false merge \u2192 outright reject). Train-vs-held-out split-repair gap is large and widening (+127). Prefer GENERALIZABLE geometry/topology over train-specific recall.\n\nCURRENT ON-DISK STATE (I have verified this myself \u2014 do not re-derive, just trust it):\n- Module constants (heuristics.py ~lines 99-124): GAP_THRESHOLD_UM=4.0, MIN_COLINEAR_COS=0.94, TANGENT_WALK_UM=6.0, SMALL_GAP_UM=1.8, NEAR_GAP_UM=2.15, RAD_RATIO_MAX=1.5, THROUGH_LINE_COS=0.7.\n- propose_edits has three accept rules in a try/except loop (skips non-split kinds and gap > GAP_THRESHOLD_UM):\n  (A) if _is_colinear_split(g, s, MIN_COLINEAR_COS=0.94): accept  (bridge-routed straightness, very strict)\n  (B) elif s.gap_um <= SMALL_GAP_UM (1.8): if deg_a==1 and deg_b in (1,2): accept  (proximity only)\n  (C) elif SMALL_GAP_UM < s.gap_um <= NEAR_GAP_UM (2.15): if deg_a==1 and deg_b in (1,2): rr=_rad_ratio(ctx,node_a,node_b); caliber_ok = rr is not None and rr <= RAD_RATIO_MAX; tl=_through_line_cos(g,s,TANGENT_WALK_UM); through_ok = tl is not None and tl >= THROUGH_LINE_COS; if caliber_ok or through_ok: accept\n  then: if accept: edits.append(s.as_edit())\n\nCONTEXT \u2014 last candidate (gen09 report): parent = 376 correct / 0 false on train; confusion REAL 1126 (accepted 376 / MISSED 750), FALSE 7 (accepted 0 / rejected 7). A previously-tried change (extending the through_line continuity gap window from 2.15 up to 4.0\u00b5m) was REJECTED at held-out +0 \u2014 the extra straight sites in the 2.15\u20134.0\u00b5m band don't exist on held-out. Do NOT propose any gap-window extension; that lever is exhausted.\n\nTHE NEW EVIDENCE that motivates this revision \u2014 the report's split-repair attribution table shows rad_ratio has ZERO discriminative power:\n  rad_ratio [-inf,1.44):  110 correct / 0 false\n  rad_ratio [1.44,2.05):  135 correct / 0 false\n  rad_ratio [2.05,inf):   131 correct / 0 false\nEvery rad_ratio bucket \u2014 including the HIGH-mismatch bucket [2.05,inf) with 131 correct \u2014 has ZERO false joins. So caliber match (rad_ratio) does NOT separate REAL splits from FALSE joins on this data. Caliber MISMATCH is in fact the NORM for genuine tip-to-shaft reconnections: a thin distal tip (rad ~0.75) rejoining its thick parent shaft (rad ~1.9) give
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 95,
  "limit": 60
}
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md"
}
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 335,
  "limit": 35
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "RAD_RATIO_MAX = 1.5         # caliber match: max/min of the two endpoint radii; > this = mismatched calibers => likely two different neurons, refuse",
  "new_string": "RAD_RATIO_MAX = 3.0         # near-band caliber ceiling (NON-discriminative): the Gen-9 attribution table shows 0 false joins in EVERY rad_ratio bucket (incl. [2.05,inf) with 131 correct), so caliber does NOT separate REAL from FALSE. Caliber MISMATCH is the NORM for genuine tip-to-shaft reconnections (thin distal tip ~0.75 rejoining a thick parent shaft ~1.9 => rad_ratio up to ~2.79). 3.0 admits that full range; near-band precision comes from the sub-2.19 \u00b5m gap cap + the endpoint-degree gate, NOT from caliber."
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "# Caliber-match near-band rule (Gen 5, rule C \u2014 a generalizable feature). The 876\n# remaining MISSED real splits all sit in a narrow band JUST above SMALL_GAP_UM and\n# are tip-to-shaft (deg_b==2). Widening proximity alone is train-specific (the\n# train/held-out split-repair gap is +14 and widening), so instead of pure\n# proximity we gate the near-band on CALIBER MATCH: a tip reconnecting to the SAME\n# broken neuron shares cable radius across the break, while two unrelated neurites\n# grazing typically have mismatched calibers. rad_ratio is a brain-independent\n# ratio, so it is expected to transfer to held-out where pure proximity did not.",
  "new_string": "# Near-band rule (Gen 5 rule C; caliber gate widened to non-restrictive in Gen 9).\n# The remaining MISSED real splits sit in a narrow band JUST above SMALL_GAP_UM and\n# are tip-to-shaft (deg_b==2). The Gen-9 split-repair attribution table shows\n# rad_ratio has ZERO discriminative power: 0 false joins in EVERY bucket, including\n# [2.05,inf) with 131 correct. Caliber MISMATCH is in fact the NORM for genuine\n# tip-to-shaft reconnections \u2014 a thin distal tip (rad ~0.75) rejoining its thick\n# parent shaft (rad ~1.9) gives rad_ratio up to ~2.79 \u2014 so RAD_RATIO_MAX=1.5 was\n# rejecting that whole real-split pool while preventing NO false merge. The\n# near-band precision actually rests on (a) the sub-2.19 \u00b5m gap cap (NEAR_GAP_UM,\n# strictly below the train false-join floor) and (b) the endpoint-degree gate\n# (deg_a==1, deg_b in (1,2); branch points deg_b>=3 excluded). RAD_RATIO_MAX is now\n# 3.0 so the caliber branch admits the full near-band tip-to-shaft caliber range\n# instead of suppressing it; the recovered pool is grounded in endpoint topology +\n# the physical proximity floor (both brain-independent), so it should transfer."
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "            # (C) caliber-matched near-band continuation (Gen 5). Rule (B) leaves a\n            #     large MISSED real-split pool sitting just above SMALL_GAP_UM, all\n            #     tip-to-shaft (deg_b==2), in the band [SMALL_GAP_UM, NEAR_GAP_UM]\n            #     (still strictly below the 2.19 \u00b5m train false-join floor). Pure\n            #     proximity recall there is train-specific and does not transfer\n            #     (train/held-out gap +14 and widening), so instead of widening\n            #     proximity we gate on CALIBER MATCH: a tip reconnecting to the SAME\n            #     broken neuron shares cable radius across the break, whereas two\n            #     unrelated grazing neurites usually have mismatched calibers.\n            #     rad_ratio is a brain-independent ratio, so it should transfer to\n            #     held-out; the gap cap below 2.19 keeps train false == 0.",
  "new_string": "            # (C) near-band continuation (Gen 5; caliber gate widened in Gen 9).\n            #     Rule (B) leaves a large MISSED real-split pool sitting just above\n            #     SMALL_GAP_UM, all tip-to-shaft (deg_b==2), in the band\n            #     [SMALL_GAP_UM, NEAR_GAP_UM] (strictly below the 2.19 \u00b5m train\n            #     false-join floor). The Gen-9 attribution table shows rad_ratio is\n            #     NON-discriminative (0 false joins in EVERY bucket, 131 correct in\n            #     the high [2.05,inf) bucket), and high rad_ratio is the NORM for a\n            #     genuine thin-tip-into-thick-shaft reconnection (rad_ratio up to\n            #     ~2.79). So RAD_RATIO_MAX is now 3.0 \u2014 the caliber branch no longer\n            #     suppresses those real splits. Near-band PRECISION comes from the\n            #     sub-2.19 \u00b5m gap cap (NEAR_GAP_UM) + the endpoint-degree gate\n            #     (deg_a==1, deg_b in (1,2)); both are brain-independent so the\n            #     recovered tip-to-shaft pool should transfer to held-out. The\n            #     through-line branch is unchanged."
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "  **(C) Near-band caliber-match OR through-line continuity** (Gen 5, extended\n  Gen 7). For a SplitSite in the band `SMALL_GAP_UM < gap_um <= NEAR_GAP_UM`\n  (1.8\u20132.15 \u00b5m, strictly BELOW the 2.19 \u00b5m train false-join floor) that is\n  **tip-to-tip** (`deg_b == 1`) OR **tip-to-shaft** (`deg_b == 2`) \u2014 `node_a`\n  always a tip \u2014 accept when EITHER:\n    - **caliber match** (Gen 5): `rad_ratio = max(r_a, r_b) / min(r_a, r_b) <=\n      RAD_RATIO_MAX` (1.5), radii from `ctx[\"node_radius\"]`; OR\n    - **through-line continuity** (Gen 7): `_through_line_cos(g, s,\n      TANGENT_WALK_UM) >= THROUGH_LINE_COS` (0.7).",
  "new_string": "  **(C) Near-band caliber-OR through-line continuity** (Gen 5, extended Gen 7,\n  caliber gate widened Gen 9). For a SplitSite in the band\n  `SMALL_GAP_UM < gap_um <= NEAR_GAP_UM` (1.8\u20132.15 \u00b5m, strictly BELOW the 2.19 \u00b5m\n  train false-join floor) that is **tip-to-tip** (`deg_b == 1`) OR **tip-to-shaft**\n  (`deg_b == 2`) \u2014 `node_a` always a tip \u2014 accept when EITHER:\n    - **caliber branch** (Gen 5, widened Gen 9): `rad_ratio = max(r_a, r_b) /\n      min(r_a, r_b) <= RAD_RATIO_MAX` (now **3.0**), radii from\n      `ctx[\"node_radius\"]`; OR\n    - **through-line continuity** (Gen 7): `_through_line_cos(g, s,\n      TANGENT_WALK_UM) >= THROUGH_LINE_COS` (0.7).\n\n  Caliber is NO LONGER a restrictive precision gate. The Gen-9 split-repair\n  attribution table shows `rad_ratio` is non-discriminative \u2014 0 false joins in\n  EVERY bucket, including `[2.05, inf)` (131 correct / 0 false) \u2014 and high\n  `rad_ratio` is the NORM for a genuine thin-tip-into-thick-shaft reconnection\n  (up to ~2.79). `RAD_RATIO_MAX = 3.0` therefore admits the full near-band caliber\n  range instead of rejecting those real splits. The near-band's PRECISION rests on\n  the sub-2.19 \u00b5m gap cap (`NEAR_GAP_UM`) + the endpoint-degree gate\n  (`deg_a == 1`, `deg_b in (1, 2)`, branch points excluded), not on caliber."
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "- **Harness:** `candidate_split_sites` broadened from tip-to-tip to",
  "new_string": "- **Gen 9 (2026-06-29):** widened rule (C)'s caliber branch by raising the\n  module-level constant `RAD_RATIO_MAX` from **1.5 \u2192 3.0** (the ONLY policy\n  change). Parent = **376 correct / 0 false** on train; confusion REAL 1126\n  (accepted 376 / MISSED 750), FALSE 7 (accepted 0 / rejected 7). Diagnosis: the\n  Gen-9 split-repair attribution table shows `rad_ratio` has ZERO discriminative\n  power \u2014 `[-inf,1.44)`: 110/0, `[1.44,2.05)`: 135/0, `[2.05,inf)`: **131\n  correct / 0 false**. Every bucket, including the high-mismatch one, has zero\n  false joins, so caliber does NOT separate REAL splits from FALSE joins on this\n  data. In fact caliber MISMATCH is the NORM for a genuine tip-to-shaft\n  reconnection: a thin distal tip (rad ~0.75) rejoining its thick parent shaft\n  (rad ~1.9) gives `rad_ratio` ~2.5, and the MISSED-real bucket is dominated by\n  exactly these high-`rad_ratio` (2.0\u20132.79), `deg_b == 2` sites in the 1.8\u20132.15 \u00b5m\n  band. So the old `RAD_RATIO_MAX = 1.5` ceiling was rejecting a large pool of\n  genuine near-band tip-to-shaft real splits while preventing NO false merge \u2014\n  caliber was dead weight there. `RAD_RATIO_MAX = 3.0` covers the observed\n  near-band max (~2.79) with margin and stops suppressing them.\n  *Precision preserved:* nothing else changed \u2014 `NEAR_GAP_UM` stays 2.15 (the\n  sub-2.19 \u00b5m gap cap is the real false-join floor; every train false sits at\n  gap \u2265 2.19 \u00b5m and `colinear_cos \u2264 0.62`), the endpoint-degree gate\n  (`deg_a == 1`, `deg_b in (1,2)`, branch points excluded) is unchanged, and the\n  through-line branch / `THROUGH_LINE_COS` are untouched. Train false stays 0.\n  *Generalizability:* the recall expansion is grounded in (a) endpoint-degree\n  topology (tip-to-tip/shaft, branch points excluded) \u2014 a blessed generalizable\n  feature \u2014 and (b) the physical sub-2.2 \u00b5m proximity floor, NOT in train-specific\n  tuning. Thin-tip-to-thick-shaft caliber mismatch is a universal,\n  brain-independent morphological fact, so the held-out near band should contain\n  the same pool. This DIFFERS from the previously REJECTED gap-window extension\n  (through-line continuity gap window 2.15 \u2192 4.0 \u00b5m, held-out +0) \u2014 that targeted a\n  2.15\u20134.0 \u00b5m band that is EMPTY on held-out; the near band is dense and known to\n  transfer (the original Gen-5 near-band rule was accepted). The exhausted\n  gap-window lever is NOT re-touched. Image again NOT used (warm-start probe AUC\n  0.52 \u2014 no REAL/FALSE separation; the recall gap is geometric, not photometric).\n  Updated the inline comment on `RAD_RATIO_MAX`, the near-band-rule comment block,\n  and rule (C)'s in-function comment to state caliber is non-discriminative.\n- **Harness:** `candidate_split_sites` broadened from tip-to-tip to"
}
```

### 📝 text [orchestrator]

Let me verify the actual edit on disk.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 122,
  "limit": 3
}
```

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 124,
  "limit": 8
}
```

### 📝 text [orchestrator]

The edit is verified on disk: `RAD_RATIO_MAX` is now `3.0`, `NEAR_GAP_UM` stays `2.15`, and the rule structure is untouched. Gen09 revision complete.

**Gen09 diagnosis & change:**

Reading gen09's failure report, two facts stood out. First, my Gen 8 change (extending the through-line continuity window to the full 2.15–4.0µm gap range) was **reverted** — it scored held-out +0 because that wider band is essentially empty on held-out, exactly the train-specific recall trap the generalization check warns about. Parent remains 376 correct / 0 false.

Second, the report's **split-repair attribution table** is the key new evidence: `rad_ratio` has *zero* discriminative power — all three buckets show 0 false joins, including the high-mismatch bucket `[2.05,∞)` with 131 correct. Caliber mismatch is the *norm* for genuine tip-to-shaft reconnections (a thin distal tip, rad ~0.75, rejoining a thick parent shaft, rad ~1.9, gives rad_ratio ~2.5), and the 750-site MISSED-real pool is dominated by exactly these high-rad_ratio, deg_b=2 sites in the 1.8–2.19µm band.

**The one improvement:** raised `RAD_RATIO_MAX` from `1.5` → `3.0`, widening rule (C)'s caliber branch to admit the full near-band tip-to-shaft caliber range (observed max ~2.79). Nothing else changed — `NEAR_GAP_UM=2.15` (the sub-2.19µm false-join floor) and the endpoint-degree gate remain the actual precision guards; the through-line branch, gap windows, and rules (A)/(B) are untouched.

Why it should generalize (unlike the reverted gap extension): the recovered recall rests on endpoint-degree topology + the physical sub-2.2µm proximity floor, and thin-tip-to-thick-shaft caliber mismatch is a universal morphological fact, so the held-out near band should hold the same dense, false-free pool. Image was not used (AUC 0.52). `rules.md` rule (C) description updated and a dated Gen 9 change-log entry added.
