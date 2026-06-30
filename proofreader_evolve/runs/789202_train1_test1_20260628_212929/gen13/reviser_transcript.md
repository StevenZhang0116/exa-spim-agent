# Reviser transcript

- thinking captured: no (0 chars)
- final text: 4764 chars
- tokens: in=58 out=305; cost_usd=27.917849250000003
- priors read: None

## Prompt sent to the reviser

```
Use the proofreader-reviser subagent to improve the evolved proofreading program. The current policy lives in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py and its theory in /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md. Edit THOSE files in place (do not touch any other files). The failure report for the latest candidate is at /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen13/failure_report.md. Diagnose why the policy lost accuracy, make ONE concrete improvement to propose_edits (keeping its call signature), and update the rules file and its change log. Do NOT claim you ran, imported, or tested anything — you have no Bash; the harness import-checks and lint-checks your edit after you finish.

IMPORTANT — THIS RUN: merge-error repair is DISABLED. The candidate stream contains SplitSites only (no MergeSite is enumerated), and any `split_label` edit you emit is dropped before scoring. Do NOT write or tune `split_label` logic; focus entirely on the `merge_labels` (split-error) policy. The failure report's merge sections are omitted accordingly.

IMAGE CURRICULUM — couple reading image to PASSING THE GATE in ONE revision. The gate keeps a candidate only if it makes MORE net correct `merge_labels` repairs than the parent with ZERO false merges. A revision that merely STARTS calling `gap_bridge_evidence` without changing which labels you merge ties the parent's split-repair score and is REVERTED — so its reads (and any image signal) are thrown away and never reach a future report. Therefore, if you decide image is worth using, you MUST spend it to RAISE RECALL in the SAME revision: take SplitSites in the report's 'MISSED real splits (rejected REAL)' bucket — real splits your current geometry is too conservative to accept — and ACCEPT the ones a high `gap_bridge_evidence.bridge_ratio` confirms are a continuous bright bridge across the gap. Gate the (cloud) read behind your cheap geometric filters so you only read the handful of ambiguous candidates. Consult the report's 'Image warm-start probe' section FIRST: it already measured whether `bridge_ratio` separates REAL from FALSE on this brain (an AUC), so you know BEFORE writing the rule whether image is worth gating on and roughly where the threshold sits. If the AUC is weak, do not over-invest in image — improve the geometric `merge_labels` policy instead.

GENERALIZATION CHECK (aggregate over the last 5 ACCEPTED generations; no neuron identities, just scores): mean TRAIN split-repair score = 197.4, mean HELD-OUT split-repair score = 27.6, gap (train - held-out) = +169.8 — widening (train pulling ahead of held-out). The gate scores you on HELD-OUT, so a large positive gap means your recent accepted changes repair train splits that DON'T carry over. If the gap is widening, prefer improvements grounded in GENERALIZABLE geometry/topology (tangent continuity, caliber match, endpoint degree) over ones that chase train-specific recall.

Attempts already tried against the CURRENT policy (do NOT repeat these — they did not beat it). The number is the held-out SPLIT-REPAIR score delta vs the parent (correct - false; the gate's actual fitness, an integer count — NOT Edge Accuracy):
  - gen10: +1 -1 lines; Gen 9 was **ACCEPTED** — parent went 376 → **400** (the `RAD_RATIO_MAX` 1.5→3.0 change transferred). Critically, the MIS -> held-out split-repair +0 vs parent (rejected)
  - gen11: +23 -0 lines; Gen10 (NEAR_GAP_UM 2.15→2.18) was **rejected** at held-out +0 — the sliver extension didn't transfer. That lever is now  -> held-out split-repair +0 vs parent (rejected)
  - gen12: +55 -17 lines; Gen 11 (the per-arm symmetry branch using `split_geom`) was **rejected** at held-out +0 — it added train recall (+23 lin -> held-out split-repair +0 vs parent (rejected)

Propose a DIFFERENT improvement from any listed above.
```

## Stream (thinking / text / tool calls, in order)

### 💭 thinking [orchestrator]



### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/gen13/failure_report.md"
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

The gen13 report's ENUM table shows `split_alt_per_pair` requested = "—" (default 1), and the data is byte-identical to gen12 (400/0, 726 missed, 1740 edits). That signals my Gen 12 `alt_gaps` lever tied at held-out +0 and was **reverted**. I must verify the actual on-disk state before proposing anything.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 144,
  "limit": 16
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Confirmed — Gen 12 (`alt_gaps`) was reverted. The file is back to the Gen 9 form. Let me confirm the `propose_edits` body and helpers are also reverted.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 250,
  "limit": 130
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Confirmed: the entire file is reverted to the Gen 9 form (parent 400/0) — Gen 12's `alt_gaps` lever tied at held-out +0 and was rolled back.

**Diagnosis:** Four consecutive recall levers in the gap>2.15 band have now tied at held-out +0 — gen8 (through-line window), gen10 (proximity sliver), gen11 (per-arm `min(cos_a,cos_b)` symmetry), gen12 (`alt_gaps` multi-contact colinear). All four operate on the *gap-relative* colinearity of the closest contact. The held-out reachable pool above the near band is saturated for that family of features. The MISSED table confirms it: e.g. `gap 2.16, colinear_cos 0.84` and `gap 2.24, colinear_cos 0.92` are real splits we reject — these are *laterally-offset* continuations where the cables are parallel but the gap/bridge vector is tilted, so every gap-relative measure under-scores them.

The one feature no prior lever has touched is the harness's leak-free **`tip_tangent_cos`** — direct cable-to-cable parallelism, *independent of gap direction* — which is exactly the "tangent continuity" the generalization check asks for. But its sign convention is precision-critical under the zero-false gate, so let me verify it in the harness source before deploying.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/harness/candidate.py",
  "offset": 60,
  "limit": 90
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Verified the harness definition (candidate.py lines 82–141): `tip_tangent_cos = dot(ta_u, tb_u)` where `ta_u` points outward along arm A and `tb_u` points forward out of B, so **a straight pass-through gives `tip_tangent_cos ≈ +1` regardless of gap direction** — sign confirmed. Crucially, for a *laterally-offset* continuation the two cables are parallel (`tip_tangent_cos` high) while the tilted gap vector drags `cos_a`/`cos_b`/`colinear_cos` down — exactly the population the four tied gap-relative levers reject. The perpendicular side-by-side bundle (the false-positive risk) has `tip_tangent_cos` high but `cos_a ≈ 0`, so a `cos_a` forward-alignment guard cleanly separates them.

This is a genuinely new, generalizable tangent-continuity lever. Let me delegate the precise change.

### 🔧 tool call [orchestrator]: Agent

```json
{
  "subagent_type": "proofreader-reviser",
  "description": "Gen 13 tip_tangent_cos continuity lever",
  "prompt": "You are revising an evolved split-repair proofreading policy. Edit ONLY these two files, in place (touch no other files):\n\n- POLICY: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py\n- THEORY/CHANGELOG: /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md\n\nHARD CONSTRAINTS (all must hold):\n- Keep propose_edits' signature EXACTLY: propose_edits(sites, ctx) -> list.\n- This run: merge-error repair is DISABLED. Do NOT write or tune any split_label logic. Focus only on merge_labels (split-repair).\n- Do NOT claim you ran, imported, or tested anything \u2014 there is no Bash; the harness import-checks and lint-checks the edit after you finish. Just make correct, syntactically valid Python.\n- Access ctx as a dict via ctx.get(...) ONLY (mirror the existing convention; do NOT use hasattr or attribute access on ctx).\n- Do NOT add any image-reader gating (the harness probe shows bridge_ratio AUC = 0.52, useless).\n- Do NOT modify ENUM_PARAMS. Do NOT change existing branches (A)/(B)/(C), the module constants already present, or the helper functions. ONLY add the new constants and the new branch described below.\n- The fitness gate scores net correct merges on a HELD-OUT split and REJECTS the whole candidate outright on a SINGLE false merge. So the new branch must be precision-safe.\n\nCONTEXT / DIAGNOSIS (put a short version in rules.md):\nFour consecutive recall levers in the gap>2.15\u00b5m band have all tied at held-out +0 and been reverted: gen8 (through-line window), gen10 (NEAR_GAP_UM 2.15->2.18 sliver), gen11 (per-arm min(cos_a,cos_b) symmetry), gen12 (alt_gaps multi-contact colinear). Every one of them keyed on GAP-RELATIVE colinearity of the closest contact (cos_a/cos_b/colinear_cos, all measured against the gap/bridge direction). That feature family is saturated on held-out. The MISSED-real-splits table contains real splits we wrongly reject whose cables are PARALLEL but laterally OFFSET across the gap (e.g. gap 2.16 colinear_cos 0.84; gap 2.24 colinear_cos 0.92): the lateral offset tilts the gap vector, so every gap-relative cosine is dragged down and the strict colinear test rejects them \u2014 even though the two cables are clearly continuous.\n\nTHE ONE IMPROVEMENT (a NEW, gap-direction-INDEPENDENT continuity branch):\nUse the harness-provided leak-free geometry accessor ctx.get(\"split_geom\") (a function: call it on a site to get a dict). Its field `tip_tangent_cos` is the cosine between the two arms' OWN tangents directly, independent of the gap direction \u2014 a straight/parallel continuation gives ~+1 even when the gap vector is tilted by lateral offset. This is exactly the continuity signal the four tied gap-relative levers miss. The field is leak-free and identical on train and held-out (so it is generalization-grounded).\nPrecision safeguard: a high tip_tangent_cos ALSO occurs for two distinct cables running side-by-side as a bundle \u2014 but for those the gap is PERPENDICULAR to the cables, so `cos_a` (the A-tip's own tangent vs the gap direction) is ~0. Requiring cos_a forward-aligned (>= GAP_FORWARD_MIN) means the tip actually AIMS toward the partner (a continuation), not a perpendicular graze (a bundle). Requiring cos_b >= 0 rejects the \"arms double back\" false geometry.\n\nIMPLEMENTATION \u2014 make exactly these edits to heuristics.py:\n\n1) Add two module-level constants, immediately AFTER the existing `THROUGH_LINE_COS = 0.7 ...` line (around line 130), with a comment:\n       # Gap-direction-INDEPENDENT continuity branch (Gen 13). The four prior recall\n       # levers all keyed on GAP-RELATIVE cosines (cos_a/cos_b/colinear_cos) and tied\n       #
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
  "old_string": "THROUGH_LINE_COS = 0.7      # near-band continuity floor measured cable-tangent-to-cable-tangent (bypasses the bridge vector, which a laterally-offset tip-to-shaft join tilts). 0.7 sits ABOVE the 0.62 max colinear_cos of any train false join, so it adds zero train false.",
  "new_string": "THROUGH_LINE_COS = 0.7      # near-band continuity floor measured cable-tangent-to-cable-tangent (bypasses the bridge vector, which a laterally-offset tip-to-shaft join tilts). 0.7 sits ABOVE the 0.62 max colinear_cos of any train false join, so it adds zero train false.\n# Gap-direction-INDEPENDENT continuity branch (Gen 13). The four prior recall\n# levers all keyed on GAP-RELATIVE cosines (cos_a/cos_b/colinear_cos) and tied\n# on held-out. tip_tangent_cos (harness split_geom) is the cosine of the two\n# arms' OWN tangents, independent of the gap vector, so it still reads ~+1 for a\n# laterally-OFFSET continuation that the gap-relative measures under-score.\n# Bundle guard: a side-by-side bundle also has high tip_tangent_cos but the gap\n# is PERPENDICULAR to the cables (cos_a ~ 0), so require cos_a forward-aligned.\nTIP_TANGENT_COS_MIN = 0.92   # arms must be within ~23 degrees of parallel (strong direct continuity)\nGAP_FORWARD_MIN = 0.5        # A-tip's own tangent must aim within ~60 degrees of the partner (NOT a perpendicular bundle graze)"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "old_string": "                    if caliber_ok or through_ok:\n                        accept = True\n            if accept:\n                edits.append(s.as_edit())   # (label_a, label_b) merge tuple",
  "new_string": "                    if caliber_ok or through_ok:\n                        accept = True\n            # (D) Gen 13: gap-direction-INDEPENDENT continuity. The branches above all\n            #     reject laterally-OFFSET real continuations because the tilted gap\n            #     vector drags every gap-relative cosine down. tip_tangent_cos compares\n            #     the two arms' OWN tangents (no gap vector), so it still reads ~+1 for\n            #     a parallel continuation. Precision-safe: the perpendicular side-by-side\n            #     bundle (a distinct neuron pair) has high tip_tangent_cos but cos_a ~ 0,\n            #     so require cos_a >= GAP_FORWARD_MIN (the tip AIMS at the partner) and\n            #     cos_b >= 0 (B's arm continues forward, does not double back). Endpoint\n            #     degree gated like (B)/(C). All fields are the harness leak-free\n            #     split_geom geometry (identical on train and held-out).\n            if not accept:\n                sg = ctx.get(\"split_geom\")\n                if sg is not None:\n                    gd = sg(s)\n                    deg_a = gd.get(\"deg_a\")\n                    deg_b = gd.get(\"deg_b\")\n                    tt = gd.get(\"tip_tangent_cos\")\n                    ca = gd.get(\"cos_a\")\n                    cb = gd.get(\"cos_b\")\n                    if (deg_a == 1 and deg_b in (1, 2)\n                            and tt is not None and tt >= TIP_TANGENT_COS_MIN\n                            and ca is not None and ca >= GAP_FORWARD_MIN\n                            and cb is not None and cb >= 0.0):\n                        accept = True\n            if accept:\n                edits.append(s.as_edit())   # (label_a, label_b) merge tuple"
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
  "old_string": "  *Why no image:* the report's warm-start probe measured `bridge_ratio` AUC = 0.52",
  "new_string": "  **(D) Gap-direction-INDEPENDENT continuity** (Gen 13, CANDIDATE). Fires only when\n  none of (A)/(B)/(C) accepted. For a deg-gated pair (`deg_a == 1`,\n  `deg_b in (1, 2)`) accept when the harness leak-free `split_geom` reports\n  `tip_tangent_cos >= 0.92` (the two arms' OWN tangents are within ~23\u00b0 of parallel,\n  measured WITHOUT the gap vector) AND `cos_a >= 0.5` (the A-tip's own tangent aims\n  within ~60\u00b0 of the partner, so the tip continues TOWARD it rather than grazing\n  perpendicular as a side-by-side bundle would) AND `cos_b >= 0` (B's arm continues\n  forward, does not double back). Bounded by the existing `GAP_THRESHOLD_UM` = 4.0 \u00b5m\n  cap. This recovers the laterally-OFFSET parallel continuations whose tilted gap\n  vector drags every gap-relative cosine (rules A/C) below threshold.\n\n  *Why no image:* the report's warm-start probe measured `bridge_ratio` AUC = 0.52"
}
```

### 🔧 tool call [subagent]: Edit

```json
{
  "replace_all": false,
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "old_string": "- **Harness:** `candidate_split_sites` broadened from tip-to-tip to",
  "new_string": "- **Gen 13 (2026-06-29, CANDIDATE):** added rule (D), a NEW gap-direction-INDEPENDENT\n  continuity branch, with two new constants `TIP_TANGENT_COS_MIN = 0.92` and\n  `GAP_FORWARD_MIN = 0.5` (added immediately after `THROUGH_LINE_COS`). Rules\n  (A)/(B)/(C), the existing constants, helpers, and `ENUM_PARAMS` are byte-identical.\n  *Diagnosis:* four consecutive recall levers in the gap > 2.15 \u00b5m band have all tied\n  at held-out **+0** and been reverted \u2014 gen8 (through-line window), gen10\n  (`NEAR_GAP_UM` 2.15 \u2192 2.18 sliver), gen11 (per-arm `min(cos_a, cos_b)` symmetry),\n  gen12 (`alt_gaps` multi-contact colinear). Every one keyed on GAP-RELATIVE\n  colinearity of the closest contact (`cos_a`/`cos_b`/`colinear_cos`, all measured\n  against the gap/bridge direction); that feature family is SATURATED on held-out.\n  The residual MISSED-real-splits are real splits whose cables are PARALLEL but\n  laterally OFFSET across the gap (e.g. gap 2.16 `colinear_cos` 0.84; gap 2.24\n  `colinear_cos` 0.92) \u2014 the lateral offset TILTS the gap vector, dragging every\n  gap-relative cosine down so the strict colinear test wrongly rejects them even\n  though the two cables are clearly continuous. *Change:* branch (D) keys on\n  `split_geom`'s `tip_tangent_cos` \u2014 the cosine of the two arms' OWN tangents,\n  INDEPENDENT of the gap vector \u2014 which still reads ~+1 for those offset\n  continuations. *Precision rationale:* a side-by-side bundle (a distinct neuron\n  pair) ALSO has high `tip_tangent_cos`, but for it the gap is PERPENDICULAR to the\n  cables so `cos_a` ~ 0; requiring `cos_a >= 0.5` means the tip actually AIMS at the\n  partner (a continuation), excluding the bundle false positive, and `cos_b >= 0`\n  rejects an arms-double-back geometry. The deg gate (`deg_a == 1`,\n  `deg_b in (1, 2)`) matches (B)/(C). *Generalization:* `split_geom` is harness\n  leak-free and IDENTICAL on train and held-out, and (D) is a genuinely DIFFERENT\n  feature axis (gap-direction-independent) than the four tied gap-relative levers, so\n  it is not expected to tie the same way. Bounded by `GAP_THRESHOLD_UM` = 4.0 \u00b5m.\n  Image again NOT used (`bridge_ratio` AUC = 0.52). Marked CANDIDATE pending the\n  held-out gate.\n- **Harness:** `candidate_split_sites` broadened from tip-to-tip to"
}
```

### 📝 text [orchestrator]

Let me verify the actual edits landed correctly on disk.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 128,
  "limit": 16
}
```

### 📝 text [orchestrator]

Constants are in place. Now verify branch (D) in `propose_edits`.

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 372,
  "limit": 28
}
```

### 🔧 tool call [orchestrator]: Read

```json
{
  "file_path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/heuristics.py",
  "offset": 399,
  "limit": 16
}
```

### 💭 thinking [orchestrator]



### 📝 text [orchestrator]

Branch (D) is correctly placed and structured — additive `if not accept` after the A/B/C chain, inside the try wrapper, before the append. Let me confirm rules.md was updated.

### 🔧 tool call [orchestrator]: Grep

```json
{
  "pattern": "Gen 13|tip_tangent_cos|\\(D\\)|GAP_FORWARD_MIN",
  "path": "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/proofreader_evolve/runs/789202_train1_test1_20260628_212929/artifacts/rules.md",
  "output_mode": "content",
  "-n": true,
  "head_limit": 20
}
```

### 📝 text [orchestrator]

Both files are updated and verified on disk. Gen 13 is complete.

**Gen 13 change — rule (D), a gap-direction-independent continuity branch:**

**Diagnosis:** Gen 12's `alt_gaps` lever tied at held-out +0 and was reverted (file back to the Gen 9 parent, 400/0). That makes **four consecutive recall levers** — gen8 (through-line window), gen10 (proximity sliver), gen11 (per-arm symmetry), gen12 (alt_gaps) — all tied on held-out. Every one keyed on *gap-relative* colinearity (`cos_a`/`cos_b`/`colinear_cos`, all measured against the gap/bridge vector). That feature family is saturated on held-out. The MISSED table shows the residual real splits are *laterally-offset parallel continuations* (e.g. `gap 2.16, colinear_cos 0.84`; `gap 2.24, colinear_cos 0.92`): the offset tilts the gap vector, dragging every gap-relative cosine below the accept bars even though the two cables are plainly continuous.

**The one improvement:** A new additive branch (D) that keys on the harness leak-free `ctx["split_geom"]` field **`tip_tangent_cos`** — the cosine between the two arms' *own* tangents, independent of the gap direction (verified in candidate.py: a straight pass-through gives ≈ +1). It accepts a deg-gated pair (`deg_a==1`, `deg_b∈{1,2}`) when `tip_tangent_cos ≥ 0.92` (cables within ~23° of parallel) **AND** `cos_a ≥ 0.5` **AND** `cos_b ≥ 0`.

**Why it's precision-safe and generalizable:** the one false-positive geometry with high `tip_tangent_cos` — two distinct cables running side-by-side as a bundle — has a *perpendicular* gap, i.e. `cos_a ≈ 0`, so the `cos_a ≥ 0.5` "the tip aims at its partner" guard excludes it; `cos_b ≥ 0` rejects arms that double back. `tip_tangent_cos` is leak-free and identical on train/held-out (the generalization anchor the prompt asks for), and it is a genuinely *different feature axis* than the four tied gap-relative levers — so it has a real path to add held-out correct merges rather than just re-slicing the saturated band. Marked CANDIDATE pending the held-out gate.
