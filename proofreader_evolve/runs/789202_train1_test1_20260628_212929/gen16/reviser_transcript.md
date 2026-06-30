# Reviser transcript

- thinking captured: no (0 chars)
- final text: 3969 chars
- tokens: in=32 out=129; cost_usd=35.284152000000006
- priors read: None

## Prompt sent to the reviser

```
Use the proofreader-reviser subagent to improve the evolved proofreading program. The current policy lives in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py and its theory in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md. Edit THOSE files in place (do not touch any other files). The failure report for the latest candidate is at /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen16/failure_report.md. Diagnose why the policy lost accuracy, make ONE concrete improvement to propose_edits (keeping its call signature), and update the rules file and its change log. Do NOT claim you ran, imported, or tested anything — you have no Bash; the harness import-checks and lint-checks your edit after you finish.

IMPORTANT — THIS RUN: merge-error repair is DISABLED. The candidate stream contains SplitSites only (no MergeSite is enumerated), and any `split_label` edit you emit is dropped before scoring. Do NOT write or tune `split_label` logic; focus entirely on the `merge_labels` (split-error) policy. The failure report's merge sections are omitted accordingly.

IMAGE CURRICULUM — couple reading image to PASSING THE GATE in ONE revision. The gate keeps a candidate only if it makes MORE net correct `merge_labels` repairs than the parent with ZERO false merges. A revision that merely STARTS calling `gap_bridge_evidence` without changing which labels you merge ties the parent's split-repair score and is REVERTED — so its reads (and any image signal) are thrown away and never reach a future report. Therefore, if you decide image is worth using, you MUST spend it to RAISE RECALL in the SAME revision: take SplitSites in the report's 'MISSED real splits (rejected REAL)' bucket — real splits your current geometry is too conservative to accept — and ACCEPT the ones a high `gap_bridge_evidence.bridge_ratio` confirms are a continuous bright bridge across the gap. Gate the (cloud) read behind your cheap geometric filters so you only read the handful of ambiguous candidates. Consult the report's 'Image warm-start probe' section FIRST: it already measured whether `bridge_ratio` separates REAL from FALSE on this brain (an AUC), so you know BEFORE writing the rule whether image is worth gating on and roughly where the threshold sits. If the AUC is weak, do not over-invest in image — improve the geometric `merge_labels` policy instead.

GENERALIZATION CHECK (aggregate over the last 5 ACCEPTED generations; no neuron identities, just scores): mean TRAIN split-repair score = 345.0, mean HELD-OUT split-repair score = 51.4, gap (train - held-out) = +293.6 — widening (train pulling ahead of held-out). The gate scores you on HELD-OUT, so a large positive gap means your recent accepted changes repair train splits that DON'T carry over. If the gap is widening, prefer improvements grounded in GENERALIZABLE geometry/topology (tangent continuity, caliber match, endpoint degree) over ones that chase train-specific recall.

```

## Stream (thinking / text / tool calls, in order)

### 💭 thinking [orchestrator]



### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen16/failure_report.md"
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

The gen15 tier (D) change was **accepted** — train split-repair rose **404 → 461 (+57)** and held-out improved (it's live: tier D's colinear-gated band is in effect). The streak of tied recall levers is broken. Now let me diagnose what's left.

Key facts from gen16:
- **MISSED pool = 2262 sites.** The near band (2.15–2.24) is dominated by **LOW** colinear_cos (mostly < 0.75) — tip-to-shaft reals whose bridge-averaged colinear is depressed by the lateral offset. These are *geometrically inseparable* from the near-band false joins (false at 2.19 has colinear 0.39; reals there span 0.05–0.67), so they can't be safely recovered.
- **Where clean separation still exists: longer gaps with high colinear.** The "correctly refused FALSE" table shows the global false-join colinear ceiling is **0.82** (at gaps 7.0 and 10.8); the in-band (3.0–6.0µm) false max is **0.79** (at 3.67). **No train false join anywhere reaches 0.85.**
- Tier D (average colinear via leak-free `split_geom`, ≥0.75) only covers (2.15, 3.0]. Rule A's strict 0.94 **both-arm** bridge test under-scores tip-to-shaft joins (one arm's cos depressed by shaft averaging), so the band (3.0, 6.0] is served *only* by the over-strict rule A and leaves recoverable long-gap continuations unrepaired.
- Image AUC 0.52 (weak) → ruled out again.

The clean, generalizable, untried lever: a new tier (E) extending the average-colinear acceptance into (3.0, 6.0] at a **higher** threshold (0.88) — chosen because the false-join colinear ceiling *rises with gap*, and 0.88 clears the global false max (0.82) by 0.06 and the in-band max (0.79) by 0.09 (comparable to tier D's margin), while staying below rule A's strict 0.94 both-arm bar so it recovers tip-to-shaft reals rule A misses. Let me delegate.

### 🔧 tool call [orchestrator]: Agent

```json
{
  "description": "Add long-gap colinear tier to policy",
  "subagent_type": "proofreader-reviser",
  "prompt": "You are revising an evolved proofreading policy. Edit EXACTLY TWO files in place, nothing else:\n- Policy: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py\n- Theory/changelog: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md\n\nYou have NO Bash. Do NOT claim you ran, imported, or tested anything \u2014 the harness import-checks and lint-checks the edit afterward. Make exactly ONE conceptual improvement (a new acceptance tier), keep the call signature `propose_edits(sites, ctx) -> list` unchanged.\n\nCONTEXT (already diagnosed \u2014 implement, do not re-derive):\nThis run repairs SPLIT errors only via `merge_labels`. The gate keeps a candidate only if it makes MORE net-correct merges than the parent on HELD-OUT with ZERO false merges (one false merge = outright reject). Parent now scores 461 correct / 0 false on train. The stream is un-truncated (split_max_sites=25000, 15378 pairs all surfaced).\n\nRead heuristics.py first to anchor line numbers. Inside `propose_edits` there is an if/elif chain over each SplitSite (after `if s.gap_um > GAP_THRESHOLD_UM: continue`, GAP_THRESHOLD_UM=6.0):\n  (A) `if _is_colinear_split(g, s, MIN_COLINEAR_COS)` \u2014 STRICT 0.94 BOTH-ARM bridge test (requires BOTH arm cosines high), any gap <= 6.0\n  (B) `elif s.gap_um <= SMALL_GAP_UM` (1.8) \u2014 tip-to-tip/-shaft proximity\n  (C) `elif SMALL_GAP_UM < s.gap_um <= NEAR_GAP_UM` (2.15) \u2014 deg-gated caliber-OR-through-line\n  (D) `elif NEAR_GAP_UM < s.gap_um <= MID_GAP_UM` (3.0) \u2014 deg-gated, accepts when `ctx[\"split_geom\"](s)[\"colinear_cos\"] >= MID_COLINEAR_COS` (0.75). This is the AVERAGE of the two arm cosines, leak-free, the report's single strongest precision feature. (Just added in Gen 15; ACCEPTED, raised train 404 -> 461.)\n\nTHE IMPROVEMENT \u2014 add a new tier (E): extend the SAME leak-free average-colinear acceptance into the longer-gap band (MID_GAP_UM, GAP_THRESHOLD_UM] = (3.0, 6.0], at a HIGHER threshold, for the same deg-gated tip-to-tip/-shaft sites. Data justification (from the failure report's SplitSite audit \"Correctly refused (rejected FALSE)\" table): the false-join colinear_cos ceiling RISES with gap \u2014 it is <= 0.62 at gap <= 3.16\u00b5m, 0.79 at gap 3.67\u00b5m, and the GLOBAL maximum over the ENTIRE audit (gaps 0\u201314.9\u00b5m) is 0.82 (at gaps 7.0 and 10.8). NO train false join anywhere reaches 0.85. Within the (3.0, 6.0] band the false-join colinear max is 0.79. So a threshold of 0.88 carries a 0.06 margin over the global false ceiling (0.82) and 0.09 over the in-band max (0.79) \u2014 comparable to tier D's 0.13 margin \u2014 while remaining BELOW rule A's strict 0.94 BOTH-ARM requirement, so it recovers genuine tip-to-shaft continuations whose AVERAGED colinear_cos is depressed (one arm's cosine is pulled down by the shaft-axis averaging on a laterally-offset tip-into-shaft join, which is exactly what rule A's both-arm bridge test rejects). Longer gaps are riskier, hence the higher (0.88 vs 0.75) threshold.\n\nCONCRETE EDITS to heuristics.py:\n\n1) Add ONE module-level constant immediately AFTER the existing `MID_COLINEAR_COS = 0.75 ...` line (in the tunable-parameters block, before `# --- Evolvable enumeration priors`). Name and value:\n   HIGH_COLINEAR_COS = 0.88\n   Comment it to say: tier (E) acceptance floor on ctx[\"split_geom\"].colinear_cos for the longer-gap band (MID_GAP_UM, GAP_THRESHOLD_UM] = (3.0, 6.0]. The false-join colinear_cos ceiling RISES with gap (<= 0.62 at gap <= 3.16\u00b5m, 0.79 at 3.67\u00b5m, GLOBAL max 0.82 over the whole audit); NO train false join anywhere reaches 0.85, and the in-band (3.0\u20136.0\u00b5m) false max is
```

### 🔧 tool call [subagent]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "MID_COLINEAR_COS = 0.75     # tier (D) acceptance floor on ctx[\"split_geom\"].colinear_cos. 0.75 sits 0.13 ABOVE the observed 0.62 colinear_cos ceiling of EVERY train false join at gap <= 3.16\u00b5m (the first false join above 0.62 is at gap 3.67\u00b5m, BEYOND the 3.0\u00b5m cap), so it admits genuine continuations while excluding false joins. This keys on the report's single strongest precision feature (split_geom.colinear_cos, the AVERAGE of the two arm cosines \u2014 NOT the strict both-arm bridge product of rule A), is gap-cap-bounded to where that separation is empirically clean, and is brain-independent so it should transfer to held-out.\n\n# --- Evolvable enumeration priors (optional) --------------------------------",
  "new_string": "MID_COLINEAR_COS = 0.75     # tier (D) acceptance floor on ctx[\"split_geom\"].colinear_cos. 0.75 sits 0.13 ABOVE the observed 0.62 colinear_cos ceiling of EVERY train false join at gap <= 3.16\u00b5m (the first false join above 0.62 is at gap 3.67\u00b5m, BEYOND the 3.0\u00b5m cap), so it admits genuine continuations while excluding false joins. This keys on the report's single strongest precision feature (split_geom.colinear_cos, the AVERAGE of the two arm cosines \u2014 NOT the strict both-arm bridge product of rule A), is gap-cap-bounded to where that separation is empirically clean, and is brain-independent so it should transfer to held-out.\nHIGH_COLINEAR_COS = 0.88    # tier (E) acceptance floor on ctx[\"split_geom\"].colinear_cos for the longer-gap band (MID_GAP_UM, GAP_THRESHOLD_UM] = (3.0, 6.0]. The false-join colinear_cos ceiling RISES with gap (<= 0.62 at gap <= 3.16\u00b5m, 0.79 at 3.67\u00b5m, GLOBAL max 0.82 over the whole audit); NO train false join anywhere reaches 0.85, and the in-band (3.0\u20136.0\u00b5m) false max is 0.79, so 0.88 carries a 0.06 margin over the global false ceiling and 0.09 over the in-band max. Higher than tier D's 0.75 because longer gaps are riskier; still BELOW rule A's strict 0.94 BOTH-ARM bridge test, so it recovers tip-to-shaft continuations whose AVERAGED colinear_cos is depressed by shaft-axis averaging. Brain-independent geometry, should transfer to held-out.\n\n# --- Evolvable enumeration priors (optional) --------------------------------"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "                        if (deg_a == 1 and deg_b in (1, 2)\n                                and cc is not None and cc >= MID_COLINEAR_COS):\n                            accept = True\n            if accept:",
  "new_string": "                        if (deg_a == 1 and deg_b in (1, 2)\n                                and cc is not None and cc >= MID_COLINEAR_COS):\n                            accept = True\n            # (E) long-gap colinear continuation (Gen 16). Rule (A)'s strict 0.94\n            #     BOTH-ARM bridge test rejects tip-to-shaft joins whose AVERAGED\n            #     colinear_cos is high but one arm is depressed by shaft-axis\n            #     averaging; tier (D) only reaches gap 3.0. Extend the same leak-free\n            #     average-colinear acceptance into (MID_GAP_UM, GAP_THRESHOLD_UM] at\n            #     a HIGHER threshold (longer gaps are riskier). No train false join\n            #     in the whole audit reaches 0.85 (global max 0.82; in-band max 0.79),\n            #     so 0.88 admits genuine long-gap continuations with zero train false.\n            elif MID_GAP_UM < s.gap_um <= GAP_THRESHOLD_UM:\n                geom_fn = ctx.get(\"split_geom\")\n                if geom_fn is not None:\n                    gd = geom_fn(s)\n                    if gd:\n                        deg_a = gd.get(\"deg_a\")\n                        deg_b = gd.get(\"deg_b\")\n                        cc = gd.get(\"colinear_cos\")\n                        if (deg_a == 1 and deg_b in (1, 2)\n                                and cc is not None and cc >= HIGH_COLINEAR_COS):\n                            accept = True\n            if accept:"
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
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "  zero train false. Colinear continuation is brain-independent, so it should\n  transfer to held-out.\n\n  *Why no image:* the report's warm-start probe measured `bridge_ratio` AUC = 0.52",
  "new_string": "  zero train false. Colinear continuation is brain-independent, so it should\n  transfer to held-out.\n\n  **(E) Long-gap colinear continuation** (Gen 16). For a SplitSite in the band\n  `MID_GAP_UM < gap_um <= GAP_THRESHOLD_UM` (3.0\u20136.0 \u00b5m) that is **tip-to-tip**\n  (`deg_b == 1`) OR **tip-to-shaft** (`deg_b == 2`) \u2014 `node_a` always a tip \u2014\n  accept when `ctx[\"split_geom\"](s).colinear_cos >= HIGH_COLINEAR_COS` (**0.88**).\n  Like rule (D) this reads the harness's leak-free per-site geometry directly (the\n  AVERAGE of the two arm cosines) and fails safe (no accept) when `split_geom` is\n  missing or returns nothing.\n\n  *Why (E) extends (D) at a HIGHER threshold:* tier (D) reaches only gap 3.0; the\n  recoverable pool with clean colinear separation continues at longer gaps, served\n  until now ONLY by rule (A)'s over-strict 0.94 BOTH-ARM bridge test. That test\n  rejects genuine tip-to-shaft continuations whose AVERAGED `colinear_cos` is high\n  but one arm's cosine is depressed by shaft-axis averaging on a laterally-offset\n  tip-into-shaft join. Tier (E) uses the same leak-free average-colinear feature,\n  but at **0.88** instead of (D)'s 0.75 because longer gaps are riskier \u2014 and 0.88\n  is still BELOW rule (A)'s strict 0.94 both-arm requirement, so it recovers exactly\n  the joins (A) misses.\n\n  *Why (E) is precision-safe and generalizable:* the SplitSite audit's \"Correctly\n  refused (rejected FALSE)\" table shows the false-join `colinear_cos` ceiling RISES\n  with gap \u2014 `<= 0.62` at gap `<= 3.16 \u00b5m`, `0.79` at gap 3.67 \u00b5m \u2014 and the GLOBAL\n  maximum over the ENTIRE audit (gaps 0\u201314.9 \u00b5m) is **0.82** (at gaps 7.0 and 10.8).\n  NO train false join anywhere reaches 0.85; within the (3.0, 6.0] band the\n  false-join `colinear_cos` max is **0.79**. So a 0.88 floor carries a 0.06 margin\n  over the global false ceiling (0.82) and 0.09 over the in-band max (0.79) \u2014\n  comparable to tier (D)'s 0.13 margin \u2014 admitting genuine long-gap continuations\n  with zero train false. Colinear continuation is brain-independent, so it should\n  transfer to held-out.\n\n  *Why no image:* the report's warm-start probe measured `bridge_ratio` AUC = 0.52"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "- **Harness:** `candidate_split_sites` broadened from tip-to-tip to",
  "new_string": "- **Gen 16 (2026-06-29) \u2014 CANDIDATE (pending the held-out gate):** added a NEW\n  acceptance tier (E), a long-gap colinear continuation, in the band\n  `MID_GAP_UM < gap_um <= GAP_THRESHOLD_UM` (3.0\u20136.0 \u00b5m) for `deg_a == 1`,\n  `deg_b in (1, 2)`, accepting when `ctx[\"split_geom\"](s).colinear_cos >=\n  HIGH_COLINEAR_COS` (**0.88**). One new constant `HIGH_COLINEAR_COS = 0.88`; the\n  ONLY logic change is the new `elif` branch (E) appended after tier (D). *Parent* =\n  **461 correct / 0 false** on train after Gen 15's tier (D) was accepted\n  (404 \u2192 461). The stream is UN-TRUNCATED (`split_max_sites = 25000`, all 15378\n  pairs surfaced), so stream-widening is exhausted \u2014 the binding constraint is which\n  reachable sites the policy accepts. *Diagnosis:* the remaining near-band MISSED\n  pool (gap 2.15\u20132.24) is dominated by LOW `colinear_cos` (< 0.75) tip-to-shaft\n  sites that are geometrically inseparable from the near-band false joins, so recall\n  there is capped; the recoverable pool with CLEAN colinear separation is at LONGER\n  gaps, served until now only by rule (A)'s over-strict 0.94 both-arm test (which\n  rejects tip-to-shaft continuations whose averaged `colinear_cos` is depressed by\n  shaft-axis averaging). *The ONE change* is the new tier (E) extending tier (D)'s\n  leak-free average-colinear acceptance into (3.0, 6.0] at the higher 0.88 threshold\n  (longer gaps are riskier). *Precision rationale:* the SplitSite audit's \"Correctly\n  refused (rejected FALSE)\" table shows the false-join `colinear_cos` ceiling RISES\n  with gap (`<= 0.62` at gap `<= 3.16 \u00b5m`, `0.79` at 3.67 \u00b5m, GLOBAL max **0.82**\n  over gaps 0\u201314.9 \u00b5m); NO train false join anywhere reaches 0.85, and the in-band\n  (3.0\u20136.0 \u00b5m) false max is **0.79**. So 0.88 carries a 0.06 margin over the global\n  false ceiling and 0.09 over the in-band max \u2014 comparable to tier (D)'s 0.13 margin\n  \u2014 with zero train false. 0.88 stays BELOW rule (A)'s 0.94, so it recovers exactly\n  the tip-to-shaft continuations (A) misses. Image NOT used (`bridge_ratio`\n  AUC = 0.52 \u2014 no REAL/FALSE separation; the recall gap is geometric, not\n  photometric). Rules (A), (B), (C), (D), the outer `GAP_THRESHOLD_UM` guard, all\n  existing constant values, `ENUM_PARAMS`, and the helpers are unchanged. Mark\n  CANDIDATE pending the held-out gate.\n- **Harness:** `candidate_split_sites` broadened from tip-to-tip to"
}
```

### 📝 text [orchestrator]

Let me verify the edits landed correctly on disk.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 131,
  "limit": 6
}
```

### 📝 text [orchestrator]

Constant is correctly placed. Now the tier (E) branch in `propose_edits`.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 387,
  "limit": 30
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Both edits are verified on disk and syntactically sound. Tier (E) is correctly chained as an `elif` after tier (D), uses the same null-safe `split_geom` access inside the existing `try/except`, and sits before the `if accept:` dispatch. The gen16 task is complete.

**Diagnosis:** Gen15's tier (D) was **accepted** — train split-repair rose **404 → 461** (and held-out improved, since it's live). The remaining near-band MISSED pool (gap 2.15–2.24) is dominated by **low** colinear_cos (< 0.75) tip-to-shaft sites that are geometrically inseparable from the near-band false joins (false at 2.19 has colinear 0.39; reals there span 0.05–0.67) — so recall there is geometrically capped and cannot be widened without a false merge. The recoverable pool with *clean* colinear separation lives at **longer gaps** (3.0–6.0µm), served until now only by rule (A)'s over-strict 0.94 **both-arm** bridge test, which rejects tip-to-shaft continuations whose averaged colinear_cos is depressed by shaft-axis averaging.

**The ONE improvement — new tier (E):** extends the same leak-free average-`colinear_cos` acceptance (via `ctx["split_geom"]`) into the band **(3.0, 6.0]**, deg-gated (`deg_a==1`, `deg_b in (1,2)`), at threshold **0.88**.

**Why precision-safe and generalizable:**
- The false-join colinear ceiling **rises with gap**: ≤0.62 at gap ≤3.16µm, 0.79 at 3.67µm, and the **global max over the entire audit (gaps 0–14.9µm) is 0.82**. **No train false join anywhere reaches 0.85.** So 0.88 carries a 0.06 margin over the global ceiling and 0.09 over the in-band (3.0–6.0µm) max of 0.79 — comparable safety to tier D's 0.13 margin.
- 0.88 > tier D's 0.75 because longer gaps are riskier, yet stays **below** rule A's strict 0.94 both-arm bar, so it recovers exactly the tip-to-shaft long-gap continuations rule A misses.
- Colinear continuation across a 3–6µm gap is brain-independent geometry — the kind the generalization check (gap +293.6, widening) asks to prefer over train-specific recall.

Image ruled out again (bridge_ratio AUC 0.52). Marked CANDIDATE pending the held-out gate.
