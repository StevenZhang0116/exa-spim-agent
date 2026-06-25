# Proofreading Criteria (the evolved natural-language program)

> The evolution loop revises THIS file alongside `heuristics.py`. It is the
> human-readable statement of *why* the policy makes the decisions it does.
> Each criterion should correspond to logic in `heuristics.py::propose_edits`.
> When the agent revises the policy, it must keep this file in sync so a
> reviewer can read the current proofreading theory in plain language.

## Objective

Maximize run-length-weighted **Edge Accuracy** on held-out ground-truth
skeletons (ERL is the tie-breaker), by repairing BOTH error classes the
segmentation makes:

- **Split errors** — one true neuron broken into several fragments. Repair by
  **unifying** fragment label pairs (`merge_labels`), without joining fragments
  that belong to different neurons.
- **Merge errors** — two (or more) true neurons fused under one segment label.
  Repair by **splitting** that label by location (`split_label`), without
  over-splitting a single real neuron (which would raise % Split Edges).

Edge Accuracy = 100 − (% Split Edges + % Omit Edges + % Merged Edges), so a good
policy must lower split AND merge errors together while not trading one for the
other. The failure report shows both components plus an over-split watchdog.

**Acceptance gate (HARD — a generation is reverted if it fails any):**
- Edge Accuracy must beat the parent by `gate_eps`.
- **No new merge error (EVERY generation):** neither `# Merges` nor
  `% Merged Edges` may rise above the parent (beyond `merge_tol`, default 0). This
  enforces the "without creating merge errors" clause directly — raising net Edge
  Accuracy by repairing splits while introducing a few merges is NOT accepted. A
  correct `merge_labels` (fusing fragments of the SAME neuron) never trips this;
  only a wrong fusion does.
- **Over-split watchdog (only when the policy emits `split_label`):** `% Split
  Edges` may not rise more than `split_tol` above the parent.

## Current criteria

The split-repair policy now has FOUR accept paths for SplitSites; MergeSites are
left alone (no `split_label`). Path 1 uses the strict WORST-arm (both-arms)
geometry test (`_is_colinear_split`); Paths 2, 3 and 4 (image-gated) use the MEAN
colinear_cos (matching the failure report's column) with a weak-arm floor
(`MIN_ARM_COS`).

  1. **Strict geometry-only accept** — emit `merge_labels` when BOTH the gap is
     small (`s.gap_um <= GAP_THRESHOLD_UM`, 4.0 µm) AND the two fragments are
     colinear across the gap via the WORST-arm test (`_is_colinear_split`, BOTH
     half-cosines `>= MIN_COLINEAR_COS`, 0.94 ≈ ≤20° bend). UNCHANGED — this is the
     original high-precision path (6 correct / 0 false on train).
  1c. **Small-gap geometry-only accept (Gen 13)** — emit `merge_labels` (pure
     geometry, NO image read, NO caliber gate) when ALL of: the gap is tiny
     (`s.gap_um <= SMALL_GAP_UM`, 1.5 µm), BOTH arms point forward across the gap
     (`min(c1, c2) >= MIN_ARM_COS`, 0.35), AND the MEAN colinearity clears a modest
     floor (`0.5 * (c1 + c2) >= SMALL_GAP_MIN_COS`, 0.50). Reuses the already-computed
     half-cosines `hc`. This reaches a NEW region NO other path touches: the image
     paths (Paths 2/3) require `gap >= IMAGE_GAP_MIN_UM` (1.0) and Path 1 requires the
     strict 0.94 worst-arm test, so small-gap reals that are merely modestly colinear
     (e.g. the report's missed real at gap 0.61 / colinear_cos 0.78) are rejected
     outright today. Rationale: when the tips are essentially adjacent, a
     forward-pointing modestly-colinear break is almost always ONE neuron the
     segmentation fragmented. SAFETY (zero false-merge risk on train): all 3 train
     FALSE joins sit at `gap >= 2.20` (2.20 / 3.00 / 3.16) — well above the 1.5 µm
     ceiling — so none can enter this path; the `MIN_ARM_COS` floor additionally
     rejects fragments that double back (negative half-cosine). Being pure geometry
     (no fluorescence-bridge brightness), it should GENERALIZE to held-out brains,
     unlike the exhausted brain-specific bridge levers.
  2. **Image-gated recall path (MEAN colinearity + weak-arm floor)** — for
     SplitSites at gap `>= IMAGE_GAP_MIN_UM` (1.0 µm) whose MEAN colinear_cos
     (`mean_cos = (c1 + c2) / 2` of the two half-cosines from
     `_colinear_half_cosines`, the SAME quantity the failure report's
     `colinear_cos` column reports) is high, `mean_cos >= IMAGE_MIN_COLINEAR_COS`
     (0.75), AND neither arm points sideways/backward across the gap
     (`min(c1, c2) >= MIN_ARM_COS`, 0.35 ≈ ≤70° bend on the weaker arm), read the
     fluorescence bridge (`gap_bridge_evidence(node_a, node_b)`) and accept ONLY
     when `bridge_ratio >= IMAGE_BRIDGE_RATIO_MIN_STRONG` (0.85, Gen 10) — signal
     stays ≥85% bright all the way across, i.e. one continuous neuron. This Path-2
     floor is RELAXED relative to Path 3 (0.90) BECAUSE the strong-colinearity gate
     (`mean_cos >= 0.75`) already supplies the precision here. This is the Gen 7
     change: the test was the WORST-arm (both-arms) `_is_colinear_split`, which
     rejected reals whose MEAN clears 0.75 but whose weaker arm dips just below it
     (one straight arm + one slightly bent arm); the MEAN test matches the report
     column while `MIN_ARM_COS` still rejects crossings/double-backs.
  3. **Caliber-gated lower-cos image tier** — an ORTHOGONAL recall lever for
     SplitSites with `IMAGE_GAP_MIN_UM <= gap_um <= 4.0` whose MEAN colinear_cos
     falls in the lower band `[IMAGE_LOWCOS_MIN_COLINEAR_COS,
     IMAGE_MIN_COLINEAR_COS)` = `[0.50, 0.75)` (Gen 16 lowered the floor 0.60 ->
     0.50; so this tier only ADDS new candidates below the Path-2 mean floor, never
     double-counts), with neither
     arm sideways/backward (`min(c1, c2) >= MIN_ARM_COS`, 0.35), the bright
     fluorescence bridge at the STRICT `bridge_ratio >= IMAGE_BRIDGE_RATIO_MIN`
     (0.90, UNCHANGED — only Path 2 was relaxed in Gen 10), AND neurite
     caliber across the gap within a DEGREE-AWARE ceiling (`rad_ratio <= cap`;
     `rad_ratio` is `max(r_a, r_b)/min(r_a, r_b)` of the two endpoint radii, an
     orthogonal feature). The ceiling depends on the partner node's degree
     (`deg_b = _deg_b(g, s)`, `int(g.degree[node_b])` or `None`): for a tip-to-TIP
     partner (`deg_b <= 1`, or unknown degree) the strict `RAD_RATIO_MAX = 1.45`
     (Gen 18: 1.4 -> 1.45) applies; for a tip-to-SHAFT partner (`deg_b >= 2`) the looser
     `RAD_RATIO_MAX_SHAFT = 2.8` applies (Gen 17: raised 2.5 -> 2.8). BIOLOGICAL RATIONALE: `node_a` is always a
     thin neurite TIP, while a shaft partner (`node_b` mid-cable, degree >= 2) is
     measured mid-shaft and is THICKER, so a large caliber mismatch is EXPECTED for a
     genuine tip-to-shaft reconnection and is a WEAK against-merge signal there;
     `RAD_RATIO_MAX_SHAFT` is still a real bound that rejects extreme (>2.8)
     mismatches. Accept ONLY when the fluorescence bridge is bright
     (`bridge_ratio >= IMAGE_BRIDGE_RATIO_MIN`, 0.90, unchanged). The radius lookup
     (`_rad_ratio`) is fully defensive: it tries `s.rad_ratio`, then `s.rad_a`/
     `s.rad_b`, then a `ctx["node_radius"]`/`g.node_radius` array indexed at the two
     endpoints, and returns a LARGE sentinel (`inf`) on any failure so an
     unavailable radius simply SKIPS this tier (degrades to parent behavior).
  4. **Tightly-matched-caliber image tier (Gen 19; extended Gen 20, Gen 21)** — a recall
     lever that lowers the MEAN-colinearity floor BELOW Path 3's 0.50, into the band
     `[IMAGE_MATCHED_MIN_COLINEAR_COS, IMAGE_LOWCOS_MIN_COLINEAR_COS)` =
     `[0.0, 0.50)` (Gen 21 lowered the floor `0.25 -> 0.0` to the perpendicular
     boundary; DISJOINT from Path 3's `[0.50, 0.75)` band, so the two tiers never
     double-count), but ONLY when neurite caliber is TIGHTLY matched across the gap:
     `_rad_ratio(g, ctx, s) <= IMAGE_MATCHED_CALIBER_MAX` (1.5). Gate: gap in
     `[IMAGE_GAP_MIN_UM, GAP_THRESHOLD_UM]` = `[1.0, 4.0]` µm AND
     `0.0 <= mean_cos < 0.50` AND BOTH arms forward-OR-perpendicular via Path 4's OWN
     weak-arm floor (`min(c1, c2) >= IMAGE_MATCHED_MIN_ARM_COS`, **0.0** — Gen 21,
     DECOUPLED from the global `MIN_ARM_COS` of 0.35 that still guards Paths 1c/2/3;
     requiring `>= 0.0` still REFUSES any arm with negative cosine, i.e. pointing
     sideways/backward — the double-back / crossing false-join signature — while now
     admitting a right-angle (perpendicular) forward bend) AND tight caliber match
     (`rad_ratio <= 1.5`).
     Then read the fluorescence bridge (`gap_bridge_evidence`) and accept ONLY when it
     is BRIGHT at the STRICT floor (`bridge_ratio >= IMAGE_BRIDGE_RATIO_MIN`, 0.90).
     (Gen 22) Path 4's caliber cap is now DEGREE-AWARE, mirroring Path 3: a tip-to-SHAFT
     partner (`db = _deg_b(g, s) >= 2`) uses the looser
     `IMAGE_MATCHED_CALIBER_MAX_SHAFT` (1.85), while a tip-to-TIP / unknown-degree
     partner keeps the strict `IMAGE_MATCHED_CALIBER_MAX` (1.5). `db` is already in
     scope (computed for Path 3 just above).
     (Gen 23; bumped Gen 25; fence extended Gen 27) Path 4's SHAFT cap is now also
     GAP-CONDITIONED: for a tip-to-SHAFT partner
     (`db >= 2`), gap `< MATCHED_SMALLGAP_UM` (**2.18** µm — Gen 27: 2.0 -> 2.18) uses
     the looser `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP` (2.8 — Gen 25: 2.5 -> 2.8,
     now equal to Path 3's `RAD_RATIO_MAX_SHAFT`), while gap `>= 2.18` keeps
     `IMAGE_MATCHED_CALIBER_MAX_SHAFT` (1.85). The boundary 2.18 sits between the
     highest MISSED shaft target (gap 2.16 -> true <= 2.165) and the only shaft false
     join FJ1 (gap 2.20 -> true >= 2.195), so the band `[2.0, 2.18)` is
     SHAFT-FALSE-JOIN-FREE and FJ1 (gap 2.20, NOT < 2.18) stays on the strict 1.85
     branch and remains rejected (rad_ratio 1.97 > 1.85).
     (Gen 24; extended Gen 26) Path 4's TIP-TIP cap is now a THREE-TIER GAP-CONDITIONED
     ladder, mirroring and extending the Gen 23 shaft treatment: for a tip-to-TIP /
     unknown-degree partner (`db <= 1` or `None`), gap `< TIP_TINYGAP_UM` (2.0 µm) uses the
     loosest `IMAGE_MATCHED_CALIBER_MAX_TIP_TINYGAP` (2.2, Gen 26); gap in
     `[2.0, TIP_SMALLGAP_UM)` = `[2.0, 2.5)` uses `IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP`
     (1.85, Gen 24); gap `>= 2.5` keeps `IMAGE_MATCHED_CALIBER_MAX` (1.5). The ladder is
     monotonic (looser cap at smaller gap). The cap is now computed as
     `if db is not None and db >= 2: cap4 = IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP if
     s.gap_um < MATCHED_SMALLGAP_UM else IMAGE_MATCHED_CALIBER_MAX_SHAFT; else: (gap <
     TIP_TINYGAP_UM -> IMAGE_MATCHED_CALIBER_MAX_TIP_TINYGAP; elif gap < TIP_SMALLGAP_UM ->
     IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP; else -> IMAGE_MATCHED_CALIBER_MAX)` and applied
     as the gate `_rad_ratio(g, ctx, s) <= cap4`. The shaft branch is byte-for-byte
     unchanged.
     (Gen 28; widened Gen 29) Path 4's caliber cap now has a LEADING mid-gap false-join-free band
     override that PREEMPTS the degree split: when `MIDGAP_FREE_LO_UM <= s.gap_um <
     MIDGAP_FREE_HI_UM` = `[2.30, 2.90)` (Gen 29: widened from `[2.35, 2.80)`), BOTH degrees use the single loosened cap
     `cap4 = IMAGE_MATCHED_CALIBER_MAX_MIDGAP` (2.8) regardless of `db`. The band
     `[2.30, 2.90)` contains ZERO train false joins for EITHER endpoint degree: FJ1
     (the only shaft false join) sits at gap 2.20 (true >= 2.195), >=0.095 µm below the
     lower edge 2.30; the lowest tip false join FJ2 sits at gap 3.00 (true >= 2.995),
     >=0.095 µm above the upper edge 2.90. The new branch is PREPENDED and the existing `if db ...`
     shaft branch is demoted to `elif db ...`, with the shaft line and the full tip
     three-tier ladder left byte-for-byte unchanged. This is purely a LOOSENING inside
     the band (2.8 >= every cap that branch previously selected there), so it only
     ADDS acceptances and removes none — a monotonic recall gain. The warm-start probe
     (AUC 0.86) shows reals in exactly this band (gap 2.58–2.67) with bridge_ratio
     >= 0.98, so the strict bright-bridge floor (0.90) confirms them.
     WHY THE OWN WEAK-ARM
     FLOOR (Gen 20): because `min(c1, c2) <= mean_cos` always, requiring
     `min_arm >= 0.35` (the shared global floor) effectively PINNED Path 4's mean
     floor at 0.35 — the cos floor could not drop below 0.35 without giving Path 4 a
     lower weak-arm floor. The new `IMAGE_MATCHED_MIN_ARM_COS = 0.25` unpins it,
     extending the band into `[0.25, 0.35)`. BIOLOGICAL RATIONALE: when two fragments
     have nearly identical neurite caliber AND a bright continuous fluorescence bridge
     spans the gap, they are very likely ONE neuron even when the tip-tangent geometry
     is only WEAKLY colinear (the break may sit at a natural bend, or a short fragment
     yields a noisy tangent). The tight caliber match + bright bridge together supply
     the precision that strong colinearity normally would; keeping the weak-arm floor
     strictly positive (0.25) still rejects arms that double back or cross sideways
     (the genuine false-join signature). Concrete reachable target: gap 1.80 /
     colinear_cos 0.29 / deg_b 2 / rad_ratio 1.25.

**Why path 4 is safe (Gen 19; reaffirmed Gen 20, Gen 21; degree-split fence Gen 22;
gap-regime fence Gen 23; tip-tip gap-regime fence Gen 24; tip-tip tiny-gap tier Gen 26;
shaft small-gap fence extended Gen 27; mid-gap false-join-free band Gen 28; widened Gen 29):**
**Gen 28 / Gen 29 mid-gap false-join-free band argument:** the train false joins BRACKET the
band `[2.30, 2.90)` (Gen 29: widened from `[2.35, 2.80)`) on BOTH sides — FJ1 (the only SHAFT false join) sits at gap 2.20
(true >= 2.195), >=0.095 µm BELOW the lower edge 2.30, and the lowest TIP false join FJ2 sits at
gap 3.00 (true >= 2.995), >=0.095 µm ABOVE the upper edge 2.90 (the other tip false join FJ3
is at gap 3.16, even further above). So the band contains
ZERO train false joins for EITHER endpoint degree, with deterministic margins
(>= 0.095 µm below FJ1, >= 0.095 µm above FJ2). The Gen 29 widening only adds the proven
sub-bands `(2.30, 2.35)` and `[2.80, 2.90)` of the same false-join-free POCKET (2.20, 3.00)
that the Gen 28 band sat inside; the lone visible lower-sub-band target at gap 2.22 stays
unreachable (too close to FJ1), so the lower edge stops at 2.30, not lower. Inside the band the loosened cap
`IMAGE_MATCHED_CALIBER_MAX_MIDGAP` (2.8) applies to BOTH degrees, and because 2.8 is
>= every cap the degree/gap ladder would otherwise select in `[2.30, 2.90)` (shaft
1.85 at gap >= 2.18; tip 1.5 at gap >= 2.5), the override only ADDS acceptances — a
monotonic recall gain with no removals. The safety rests on the gap fence (a band with
no false join of either degree), NOT on a caliber margin, so the cap is effectively
unconstrained there; 2.8 stays within observed genuine real caliber (accepted reals
reach rad_ratio ~3.0). Each newly-eligible candidate still must clear the strict
bright-bridge floor (`bridge_ratio >= 0.90`); the warm-start probe (AUC 0.86) shows
in-band reals at gap 2.58–2.67 with bridge_ratio >= 0.98. FJ1 (gap 2.20, below 2.30)
stays on the strict 1.85 shaft branch and remains rejected (1.97 > 1.85); both tip
false joins (gap >= 3.00, above 2.90) keep the 1.5 tip cap and stay rejected
(1.67 / 1.87 > 1.5).
**Gen 26 tip-tip tiny-gap argument:** both TIP-TIP (`deg_b = 1`) train false joins sit at
gap 3.00 (rad_ratio 1.67) and gap 3.16 (rad_ratio 1.87) — BOTH at gap >= 3.00. So the tip
gap regime `< TIP_TINYGAP_UM` (2.0) is TIP-FALSE-JOIN-FREE with a FULL 1.0 µm margin to the
nearest tip false join (gap 3.00). Below this fence the new looser tip cap
`IMAGE_MATCHED_CALIBER_MAX_TIP_TINYGAP` (2.2) sits ABOVE the tip false-join rad_ratios
(1.67, 1.87) — but caliber is NOT what does the rejection there; the gap fence (gap < 2.0 vs
the false joins at gap >= 3.00) is. The cap 2.2 stays well within observed GENUINE tip
caliber: the accepted bucket has tip-tip reals out to rad_ratio 2.60–3.00 (via the high-cos
paths), so wide caliber is NOT itself a false-join signal at small gap. This targets MISSED
tip-tip reals whose ONLY disqualifier under Gen 24/25 was the tip caliber cap — e.g. gap
1.00 / cos 0.06 / deg_b 1 / rad_ratio 2.00 and gap 1.00 / cos 0.08 / deg_b 1 / rad_ratio
1.95 — both currently rejected at the 1.85 tip cap. Each still must clear the strict
bright-bridge floor (`bridge_ratio >= 0.90`); the three image paths' cos band [0.0, 0.50),
`IMAGE_MATCHED_MIN_ARM_COS` (0.0), and the shaft branch are unaffected. The two tip false
joins (gap 3.00 / 3.16 >= 2.0) still use the regular 1.5 tip cap (gap >= 2.5) and stay
rejected by caliber (1.67 / 1.87 > 1.5); the shaft false join (gap 2.20) is on the unchanged
shaft branch.
**Gen 24 tip-tip gap-regime argument:** the two TIP-TIP (`deg_b = 1`) train false joins
sit at gap 3.00 (rad_ratio 1.67) and gap 3.16 (rad_ratio 1.87) — BOTH at gap >= 3.00. So
the tip-tip gap regime `< TIP_SMALLGAP_UM` (2.5) is TIP-TIP-FALSE-JOIN-FREE: there is NO
tip-tip false join below 2.5 µm (the smallest is 3.00, a 0.5 µm buffer). The new looser tip
cap `IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP` (1.85) applies ONLY for tip-to-TIP /
unknown-degree partners (`db <= 1` or `None`) at gap < 2.5 — strictly below the smallest
tip-tip false-join gap (3.00) — so loosening the tip caliber cap there fences against
NOTHING that needs fencing and adds ZERO new false-merge risk (the safety rests on a gap
regime containing no tip-tip false joins, NOT on a thin caliber margin). The two tip-tip
false joins both sit at gap >= 3.00 >= 2.5, so they still use the regular 1.5 tip cap and
stay rejected by caliber (1.67 / 1.87 > 1.5). The remaining false join (gap 2.20 /
rad_ratio 1.97) is a `deg_b = 2` SHAFT join handled by the UNCHANGED shaft branch and is
unaffected. The bright-bridge requirement (0.90) is retained. Concrete reachable MISSED
tip-tip reals this opens (deg_b 1, gap < 2.5, cos in [0, 0.5), rad in (1.5, 1.85],
currently rejected): gap 1.73 / cos 0.28 / rad 1.84; gap 2.00 / cos 0.21 / rad 1.67.
**Gen 23 / Gen 25 / Gen 27 gap-regime argument:** the smallest train false-join gap is 2.20 — ALL 3 false
joins sit at gap >= 2.20 (2.20 / 3.00 / 3.16). The ONLY shaft (`deg_b >= 2`) false join FJ1
sits at gap 2.20 (displayed -> true gap >= 2.195). **Gen 27 extends the shaft small-gap fence
2.0 -> 2.18.** The band `[2.0, 2.18)` is SHAFT-FALSE-JOIN-FREE with a DETERMINISTIC margin:
the boundary 2.18 sits cleanly between the highest MISSED shaft target (gap 2.16 -> true <=
2.165) and FJ1 (gap 2.20 -> true >= 2.195), so train precision is GUARANTEED preserved (zero
new train false merges). The looser shaft cap `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP` (2.8
— Gen 25) applies ONLY for shaft partners (`db >= 2`) at gap < 2.18 — strictly below FJ1's
gap (2.20) — so loosening caliber there fences against NOTHING that needs fencing and adds
ZERO new false-merge risk (the safety rests on a shaft-false-join-free gap regime, NOT on a
caliber margin, so the cap is effectively unconstrained there). FJ1 (gap 2.20 / cos 0.44 /
rad_ratio 1.97) is NOT < 2.18, so it still uses the regular 1.85 shaft cap and stays rejected
by caliber (1.97 > 1.85). The two tip false joins (gap 3.00 / 3.16) keep the tip cap 1.5 and
are unaffected. CRITICAL: the Gen 27 targets at gap 2.08 / 2.15 / 2.16 have rad_ratio
2.21 / 2.58 / 2.74 — all EXCEEDING FJ1's rad_ratio 1.97 — so caliber alone could NOT separate
them from FJ1; the ONLY separator is the gap fence (targets at 2.08–2.16 vs FJ1 at 2.20),
which is exactly why the fence (not a caliber margin) is the load-bearing guard here. Wide
caliber is NOT a false-join signal at small gap: genuine shaft reals with rad_ratio 2.64 /
2.69 already sit in the accepted bucket. The cap value (2.8) equals Path 3's existing shaft
cap `RAD_RATIO_MAX_SHAFT`. Concrete reachable MISSED shaft reals the Gen 27 bump opens
(deg_b 2, gap in [2.0, 2.18), cos in [0, 0.5), rad in (1.85, 2.8], currently rejected): gap
2.08 / cos 0.07 / rad 2.21; gap 2.15 / cos 0.31 / rad 2.58; gap 2.16 / cos 0.22 / rad 2.74 —
each still gated by the strict bright-bridge floor (`bridge_ratio >= 0.90`; false joins are
uniformly bright 0.97–0.99, so the bridge floor does NOT reject them — the gap+caliber
geometry does). Concrete reals the earlier Gen 25 bump opened (deg_b 2, gap < 2.0): gap 1.60
/ cos 0.24 / rad 2.64; gap 1.88 / cos 0.29 / rad 2.58; gap 1.98 / cos 0.13 / rad 2.56.
The 3 train FALSE joins are (gap 2.20 / cos 0.44 / deg_b 2 / rad_ratio 1.97),
(gap 3.00 / cos 0.06 / deg_b 1 / rad_ratio 1.67), and (gap 3.16 / cos 0.67 / deg_b 1 /
rad_ratio 1.87). **Gen 22 degree-split fence:** Path 4's caliber cap is now keyed on
`db`. The ONLY `deg_b >= 2` (shaft) false join sits at rad_ratio 1.97; the new shaft
cap 1.85 is strictly BELOW it (margin 0.12), so it stays rejected by caliber. The two
`deg_b = 1` (tip) false joins sit at rad_ratio 1.67 and 1.87, both ABOVE the unchanged
tip cap 1.5, so they stay rejected too. Thus the degree-split (shaft 1.85, tip 1.5)
still excludes ALL three false joins by caliber alone, with the tip cap's margin to
the 1.67 tip false join preserved. Under the prior flat cap (1.5) EVERY one had
`rad_ratio >= 1.67 > 1.5 = IMAGE_MATCHED_CALIBER_MAX`, so Path 4's tight caliber gate
rejects all three REGARDLESS of their cos OR weak-arm angle OR bridge brightness.
This is the structural fact that lets Path 4 lower the cos floor below 0.50 SAFELY
where it could not be lowered GLOBALLY: the deg_b 2 false join (cos 0.44 / rad_ratio
1.97) has a bright (~0.97) bridge, so globally only the 0.50 cos floor keeps it out —
but inside Path 4 the tight caliber gate (1.97 > 1.5) keeps it out instead.
**Gen 20** lowers Path 4's cos floor `0.35 -> 0.25` AND its weak-arm floor
`0.35 -> 0.25` (via the new Path-4-local `IMAGE_MATCHED_MIN_ARM_COS`, leaving the
global `MIN_ARM_COS` of 0.35 that guards Paths 1c/2/3 untouched). Because the caliber
gate rejects ALL three false joins on caliber ALONE (each rad_ratio >= 1.67 > 1.5),
neither floor change can admit any of them — the cos/weak-arm floors are NOT what
holds them out of Path 4, the caliber gate is. Gen 20 therefore opens ZERO new
false-merge risk on the train set while harvesting tightly-matched-caliber reals in
the newly-opened `[0.25, 0.35)` mean band (e.g. the report's gap 1.80 / colinear_cos
0.29 / deg_b 2 / rad_ratio 1.25 — gently curved, tightly matched caliber, and like
nearly all train reals confirmed by a bright bridge — plus similar tight-caliber
misses among the 345 not shown). The bright-bridge floor stays the strict 0.90.
**Gen 21** lowers Path 4's cos floor `0.25 -> 0.0` AND its weak-arm floor
`0.25 -> 0.0`, extending the mean-colinearity band to `[0.0, 0.50)` (the
perpendicular boundary). Because the tight caliber gate (`rad_ratio <= 1.5`) rejects
ALL three false joins on caliber ALONE (each rad_ratio >= 1.67 > 1.5), the cos and
weak-arm floors are NOT what hold them out of Path 4 — caliber is — so dropping both
floors to the perpendicular boundary opens ZERO new false-merge risk on the train
set. The `min_arm >= 0.0` floor still REFUSES any arm with a negative forward cosine
(sideways/backward — the double-back / crossing false-join signature), admitting only
forward-or-perpendicular arms. The bright-bridge floor stays the strict 0.90 (image
warm-start probe AUC 0.86, strong). Concrete reachable tight-caliber, low-but-forward
MISSED reals this opens (currently rejected at the 0.25 floor): gap 0.85 / cos 0.05 /
deg_b 2 / rad_ratio 1.22; gap 1.38 / cos 0.10 / deg_b 2 / rad_ratio 1.15; gap 1.00 /
cos 0.18 / deg_b 1 / rad_ratio 1.00; gap 1.00 / cos 0.15 / deg_b 1 / rad_ratio 1.34.

**Why path 3 is safe (with the Gen 9 degree-aware ceiling, Gen 16 floor 0.50,
Gen 17 shaft cap 2.8, Gen 18 tip-to-tip cap 1.45):** the 3 train false joins all
remain rejected. The cos 0.44 false join (`deg_b = 2`, `rad_ratio = 1.97`) is BELOW
Path 3's 0.50 mean floor (margin 0.06), so the colinearity floor excludes it — and
its `rad_ratio` 1.97 is already well INSIDE both the old 2.5 and the new 2.8 shaft
cap, so raising the shaft cap could not have admitted it even if it cleared the
floor; the cos 0.06 false join is far below the floor. The cos 0.67 / cos 0.06 false
joins are both TIP-TO-TIP (`deg_b = 1`), so they keep the tip-to-tip
`RAD_RATIO_MAX` ceiling — now `1.45` (Gen 18: 1.4 -> 1.45) — and their `rad_ratio`
1.87 / 1.67 STILL exceed it (margins 0.42 and 0.22), so they remain killed by the
caliber gate regardless of the shaft cap. The Gen 18 raise (1.4 -> 1.45) is kept
deliberately small precisely to preserve the margin to the 1.67 false join (do NOT
raise it toward 1.67).
So NONE of the 3 false joins is gated by `RAD_RATIO_MAX_SHAFT`: raising it from 2.5
to 2.8 cannot admit any of them. The relaxation to `RAD_RATIO_MAX_SHAFT = 2.8`
opens Path 3 only for tip-to-SHAFT (`deg_b >= 2`) candidates with `rad_ratio` in
`(1.4, 2.8]`, where a high caliber ratio is EXPECTED for a genuine reconnection
(thin tip vs thicker mid-shaft) — and those candidates must STILL pass the mean
band, the weak-arm floor, AND a bright (>= 0.90) fluorescence bridge. The Gen 16
MISSED bucket has reachable such reals (e.g. gap 1.65 / cos 0.78 / rad 2.69 /
deg_b 2 — high colinearity, killed ONLY by `rad_ratio` 2.69 > 2.5) blocked solely by
the old 2.5 ceiling; accepted reals already reach `rad_ratio` up to 3.00 (e.g. gap
1.73 / cos 0.77 / rad 3.00, accepted via the no-caliber-gate Path 2), confirming a
large tip-to-shaft caliber mismatch is real when colinear. So the tier now also
harvests MISSED tip-to-shaft reals in the `[0.50, 0.75)` band with a high (but
legitimate) caliber mismatch in `(2.5, 2.8]` and a bright fluorescence bridge
confirming one continuous neuron. Caliber (`rad_ratio`) is ORTHOGONAL to colinearity
(`cos`), so it adds a genuinely new discriminator rather than re-cutting the same
axis. The cloud bridge read is gated behind cheap geometry (gap band + lower
colinearity + caliber) so only a handful of ambiguous candidates trigger it.

**Why path 2:** recall is still the bottleneck. After gen03, only 30/535 real
splits are accepted and 505 are still MISSED, and the `cos >= 0.80` band is nearly
saturated (almost all reals there are already accepted). The policy read the image
on 26 REAL splits and 0 FALSE joins — no false join reaches the cos floor, because
all 3 train false joins sit at `colinear_cos <= 0.67`. So `colinear_cos`, not the
bridge gate, is the false-join guard (probe false joins are bright, 0.97–0.99, with
`bridge_ratio` AUC 0.86 driven by reals being slightly brighter — so it CANNOT
separate the bright false joins). Lowering the floor from 0.80 to 0.75 harvests the
near-colinear reals in the newly-opened `[0.75, 0.80)` cos pocket (missed reals at
cos ~0.72–0.78) whose bright fluorescence bridge confirms continuity, while staying
a clear margin (0.08 above the highest, 0.31 above the next observed false join) above
the false-join geometry (`cos <= 0.67`). The cloud read is gated behind cheap geometry
(gap band + relaxed colinearity) so only the handful of ambiguous candidates are
read. The gate is parent-relative, so each generation must beat the last accepted
policy, not the no-edit floor.

## Known failure modes to address (hypotheses for the loop)

Split-repair (SplitSite → `merge_labels`):
- **Over-merge at crossings.** Two unrelated neurites passing within a few µm get
  wrongly unified, creating a merge error. Candidate fix: require the two tips'
  local tangent directions to be roughly collinear (continuation, not crossing).
- **Under-repair of real gaps.** True splits with a gap slightly above threshold
  are missed. Candidate fix: allow larger gaps when direction agreement is high.

Merge-repair (MergeSite → `split_label`):
- **Over-split of one neuron.** Cutting at a genuine bifurcation of a SINGLE
  neuron breaks a real cable, raising % Split Edges. Candidate fix: only split
  when the branch looks like two distinct neurites — e.g. `angle_deg` far from
  180° (not a straight pass-through), `radius_ratio` far from 1 (different
  calibers), and both arms long (`cable_a_um`, `cable_b_um`).
- **Missed merges.** A real fusion left uncut keeps % Merged Edges high. The
  failure report lists the BASELINE merge labels (raw labels spanning ≥2 GT
  neurons on train) as concrete repair targets to aim a `split_label` at.

Both:
- **No image evidence.** Geometry alone ignores fluorescence. Candidate fix: read
  the image patch (`ctx["read_image_patch"]`, may be None) to test for a
  connecting signal across a gap, or an intensity valley at a suspected cut.

## Candidate space (what the policy gets to choose from)

The policy receives a UNIFIED stream of two site kinds (branch on `site.kind`):

- **SplitSite** (`kind == "split"`, from `dataset.candidate_split_sites`): for
  every fragment **tip** (`node_a`, degree 1), the nearest **differently-labelled**
  node within `max_gap_um` as `node_b` (tip, shaft, or branch — so tip-to-shaft
  and branch-point reconnections are candidates). One per unordered label pair.
- **MergeSite** (`kind == "merge"`, from `dataset.candidate_merge_sites`): a
  branch node (degree ≥ 3) inside ONE label where the two longest arms are BOTH
  long — the signature of two neurites fused at a touch/crossing. Carries the cut
  node, a seed deep in each arm, and advisory features (`branch_degree`,
  `angle_deg`, `radius_ratio`, `cable_a_um`, `cable_b_um`). All fields are derived
  from fragment geometry alone, so MergeSites are leak-free on held-out.

### Tuning the candidate stream itself (`ENUM_PARAMS`)

The thresholds above decide what to ACCEPT; the candidate stream decides what you
even SEE. `heuristics.py` may define a module-level `ENUM_PARAMS` dict to widen or
narrow that stream — turning the framework's prior on "what an error looks like"
from a fixed rail into an evolvable knob:
- `max_gap_um` (1–40) — split search radius; raise to reach longer true gaps a
  tight radius misses, lower to cut noise.
- `tip_to_shaft` — if False, split partners must be tips (legacy tip-to-tip).
- `min_arm_cable_um` (2–50) — lower to surface SHORTER merges the 10 µm floor hides
  (a detector-recall lever for the "missed merges" mode above).
- `seed_depth_um`, `max_per_label`, `split_max_sites`, `merge_max_sites` — see
  `dataset.ENUM_PARAM_SPEC`.
Values are clamped to a safe rail and unknown keys ignored, so this can't crash the
run; `ctx["enum_params"]` reports the values actually in effect (post-clamp). Widen
the stream when the failure report shows a recall gap (merge targets with no
MergeSite, or unrepaired splits with no SplitSite); narrow it when the candidate
set is noisy and precision is the bottleneck.

`ctx["n_split_sites"]` / `ctx["n_merge_sites"]` report the stream composition.

## Change log

- **Gen 0:** seed is a conservative colinear split-repair policy — `merge_labels`
  for SplitSites with `gap_um <= GAP_THRESHOLD_UM` (4.0 µm) AND colinear
  continuation (`cos >= MIN_COLINEAR_COS`, 0.94); no `split_label`.
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
- **Gen 2:** added an image-gated recall path for near-colinear SplitSites. For
  gaps `<= 4.0 µm` that fail the strict 0.94 colinear test but fall in the band
  `cos in [0.85, 0.94)` at gap `>= 1.0 µm`, accept the merge only when
  `gap_bridge_evidence.bridge_ratio >= 0.90`. Motivated by the recall bottleneck
  (only 6/535 reals accepted; cos ≈ 0.85–0.94 band of missed reals) and the strong
  bridge_ratio separator (AUC 0.86); the relaxed band stays above the observed
  false-join geometry (`colinear_cos <= 0.67`). The strict 0.94 geometry-only
  accept is retained unchanged.
- **Gen 3:** lowered `IMAGE_MIN_COLINEAR_COS` from 0.85 to 0.80, expanding the
  image-gated recall band to `cos in [0.80, 0.94)`. Motivated by gen02
  (split-repair score 6 -> 23 accepted, 23 correct / 0 false): 512 reals are still
  MISSED and the `cos >= 0.85` band is nearly exhausted, while all 3 train false
  joins sit at `cos <= 0.67` so none reach the cos floor (`bridge_ratio` AUC 0.86).
  This harvests the `[0.80, 0.85)` reals whose bright bridge confirms continuity
  while staying above the false-join geometry, so net-correct rises with zero false
  merges. Strict 0.94 geometry-only accept, `bridge_ratio >= 0.90`, and
  `gap >= 1.0` are all unchanged.
- **Gen 4:** lowered `IMAGE_MIN_COLINEAR_COS` from 0.80 to 0.75, expanding the
  image-gated recall band to `cos in [0.75, 0.94)`. Motivated by gen03 (split-repair
  score 23 -> 30 accepted, 30 correct / 0 false): 505 reals are still MISSED and the
  `cos >= 0.80` band is nearly saturated; the policy read 26 REAL and 0 FALSE joins
  (no false join reaches the cos floor — all 3 train false joins sit at `cos <= 0.67`),
  and `bridge_ratio` (AUC 0.86) cannot separate the bright false joins (0.97–0.99), so
  `colinear_cos` is the sole false-join guard. This harvests the `[0.75, 0.80)` reals
  whose bright bridge confirms continuity while staying a clear margin above the
  false-join geometry (`cos <= 0.67`), so net-correct rises with zero false merges.
  Strict 0.94 geometry-only accept, `bridge_ratio >= 0.90`, and `gap >= 1.0` are all
  unchanged.
- **Gen 6:** added a THIRD accept path — a caliber-gated lower-cos image tier — as
  an ORTHOGONAL recall lever (new constants `IMAGE_LOWCOS_MIN_COLINEAR_COS = 0.60`
  and `RAD_RATIO_MAX = 1.4`). Motivated by the gen05 report: all 3 train false
  joins have low `colinear_cos` (0.44, 0.06, 0.67) AND high endpoint caliber
  mismatch `rad_ratio` (1.97, 1.67, 1.87 — all >= 1.67), an UNUSED feature
  orthogonal to colinearity. A true single-neuron break has MATCHED caliber on both
  sides, so a caliber gate (`rad_ratio <= 1.4`) lets us safely admit reals BELOW
  the 0.75 cos floor. Tier 3 accepts a SplitSite with gap in `[1.0, 4.0]` µm that
  FAILS the Path-2 0.75 colinear test but PASSES the 0.60 floor, has matched caliber
  (`rad_ratio <= 1.4`), and a bright bridge (`bridge_ratio >= 0.90`). Every train
  false join is excluded by EITHER the 0.60 cos floor (0.44, 0.06 below it) OR the
  caliber gate (the 0.67 one has `rad_ratio` 1.87 > 1.4), so the tier harvests
  missed reals in the `[0.60, 0.75)` band with zero false merges. The radius lookup
  (`_rad_ratio`) is defensive (tries `s.rad_ratio`, then `s.rad_a`/`s.rad_b`, then
  a `node_radius` array; returns `inf` sentinel on failure to SKIP the tier). The
  bridge read stays gated behind cheap geometry so only a handful of candidates
  trigger a cloud read. Paths 1 (strict 0.94), 2 (0.75 + bridge 0.90), `gap >= 1.0`,
  and `IMAGE_BRIDGE_RATIO_MIN = 0.90` are ALL unchanged. NOTE: the previously
  rejected bridge relaxation (`IMAGE_BRIDGE_RATIO_MIN` 0.90 -> 0.83, which tied the
  parent at +0.000 held-out and was reverted) is deliberately NOT repeated — this
  generation keeps bridge at 0.90 and adds a new orthogonal feature instead.
- **Gen 7:** switched the IMAGE-gated paths (Paths 2 and 3) from the WORST-arm
  (both-arms) `_is_colinear_split` test to a MEAN colinear_cos test with a weak-arm
  floor (new constant `MIN_ARM_COS = 0.35`). DIAGNOSIS: the failure report's
  `colinear_cos` column is the MEAN of the two half-cosines (arm A into the gap;
  the gap into B's continuing arm), but the policy gated on the WORST arm, so it
  rejected real splits whose MEAN clears the floor while the weaker arm dips just
  below it (one straight arm + one slightly bent arm). The gen06 MISSED-real bucket
  is full of these — e.g. gap 1.82/cos 0.86, gap 1.86/cos 0.78, gap 1.41/cos
  0.63/rad 1.00, gap 1.43/cos 0.70/rad 1.17 — all with report colinear_cos at/above
  a path floor yet rejected by the both-arms gate. FIX: added a
  `_colinear_half_cosines(g, s)` helper returning `(c1, c2)` (or `None` on
  degeneracy) with the EXACT vector construction of `_is_colinear_split` so
  `(c1 + c2) / 2` equals the report's `colinear_cos`; refactored `_is_colinear_split`
  to call it and keep its both-arms semantics bit-for-bit (Path 1 unchanged). New
  Path 2 accepts when `mean_cos >= IMAGE_MIN_COLINEAR_COS` (0.75) AND `min(c1, c2)
  >= MIN_ARM_COS` (0.35) AND bright bridge; new Path 3 accepts when
  `0.60 <= mean_cos < 0.75` AND `min(c1, c2) >= 0.35` AND `rad_ratio <= 1.4` AND
  bright bridge. SAFETY: all 3 train false joins have MEAN colinear_cos <= 0.67, so
  Path 2's 0.75 mean floor excludes all of them; the only false join with mean in
  Path 3's `[0.60, 0.75)` band (mean 0.67) has `rad_ratio` 1.87 > 1.4 and is killed
  by the caliber gate. The `MIN_ARM_COS = 0.35` weak-arm floor additionally rejects
  sideways/backward arms (crossings / double-backs) that a pure mean could let
  through on held-out. Because both-arms `>= T` implies mean `>= T`, EVERY candidate
  the parent accepted still passes (no regression); the change ADDS reals whose
  weaker arm previously failed the both-arms gate, so net-correct rises with zero
  new false merges. This is a DIFFERENT lever than the rejected gen5 bridge
  relaxation and the gen6 caliber gate — `IMAGE_BRIDGE_RATIO_MIN` (0.90),
  `RAD_RATIO_MAX` (1.4), `IMAGE_GAP_MIN_UM` (1.0), and the strict Path-1 0.94
  both-arms test are ALL unchanged.
- **Gen 9:** made Path 3's caliber gate DEGREE-AWARE (new constant
  `RAD_RATIO_MAX_SHAFT = 2.5` and helper `_deg_b(g, s)` returning `int(g.degree[
  node_b])` or `None`). DIAGNOSIS: Path 3's flat caliber ceiling (`rad_ratio <= 1.4`)
  was blocking a real pool of MISSED tip-to-SHAFT splits. The gen08 report's MISSED
  bucket has many reals where the partner `node_b` is mid-SHAFT (`deg_b = 2`),
  moderate colinearity, and HIGH `rad_ratio` — e.g. gap 1.67/cos 0.76/rad 2.01,
  gap 1.87/cos 0.67/rad 2.26, gap 1.65/cos 0.78/rad 2.69, all in Path 3's
  `[0.60, 0.75)` mean band but killed by the 1.4 ceiling. KEY INSIGHT: a tip-to-shaft
  junction legitimately has a large radius mismatch — `node_a` is the THIN tip of a
  neurite while `node_b` is measured mid-cable and is THICKER — so a high `rad_ratio`
  is EXPECTED for a genuine tip-to-shaft reconnection and a WEAK against-merge signal
  there; two tip ENDS (`deg_b = 1`) of mismatched caliber is more suspicious. FIX:
  in Path 3 ONLY, compute `db = _deg_b(g, s)` and `cap = RAD_RATIO_MAX_SHAFT if
  (db is not None and db >= 2) else RAD_RATIO_MAX`, then require `_rad_ratio(g, ctx,
  s) <= cap`. On unknown degree (`None`) it falls back to the strict 1.4 ceiling
  (conservative). SAFETY: all 3 train false joins remain rejected — the ONLY one in
  Path 3's `[0.60, 0.75)` mean band is tip-to-tip (`deg_b = 1`, cos 0.67, rad 1.87)
  and stays gated by the tight 1.4 ceiling; the other two (cos 0.44 and 0.06) sit
  BELOW Path 3's 0.60 mean floor and never reach the caliber test. The relaxation
  opens Path 3 only for tip-to-shaft (`deg_b >= 2`) candidates with `rad_ratio` in
  `(1.4, 2.5]`, which must still pass the mean-colinearity band, the weak-arm floor
  (`MIN_ARM_COS`), AND the bright fluorescence bridge (>= 0.90). This is a DIFFERENT
  lever than gen5 (bridge threshold), gen6 (introducing the caliber gate), gen7
  (mean vs worst-arm colinearity), and gen8 (gap reach, which produced ZERO new
  correct merges and was REVERTED and is NOT repeated). `RAD_RATIO_MAX` (1.4), the
  mean test, `MIN_ARM_COS` (0.35), the bridge floor (0.90), `GAP_THRESHOLD_UM`, and
  the top-level 4 µm gap cap are all UNCHANGED. Paths 1 and 2 are byte-for-byte
  unchanged.
- **Gen 10:** made the image bridge-ratio floor GEOMETRY-CONDITIONED — relaxed it
  to `IMAGE_BRIDGE_RATIO_MIN_STRONG = 0.85` in Path 2 ONLY (the strong-colinearity
  tier, `mean_cos >= IMAGE_MIN_COLINEAR_COS = 0.75`), while Path 3 (the weaker
  `[0.60, 0.75)` tier) KEEPS the strict `IMAGE_BRIDGE_RATIO_MIN = 0.90`. DIAGNOSIS:
  the policy is recall-bound at the parent score of 95 (gen09 confusion: REAL 535 =
  accepted 95 / MISSED 440; FALSE 3, all correctly rejected). The gen09 "Image
  evidence the policy read (gap_bridge_evidence)" table lists train-REAL SplitSites
  the policy ALREADY read whose `bridge_ratio` is 0.84/0.85 — real splits it
  correctly identified as geometrically colinear (mean_cos >= 0.75, so in Path 2)
  but then REJECTED solely because the bridge fell just under the 0.90 floor.
  FIX: added the module-level constant `IMAGE_BRIDGE_RATIO_MIN_STRONG = 0.85` and
  changed Path 2's bridge check from `br >= IMAGE_BRIDGE_RATIO_MIN` to
  `br >= IMAGE_BRIDGE_RATIO_MIN_STRONG`; this flips the stuck `[0.85, 0.90)` reals
  into correct merges. `IMAGE_BRIDGE_RATIO_MIN` itself stays 0.90 and Path 3's
  check is byte-for-byte unchanged. SAFETY (zero new false merges): the 3 train
  FALSE joins are all currently REJECTED. The gen09 "Image warm-start probe" shows
  they are BRIGHT (`bridge_ratio` 0.97 / 0.97 / 0.99), so a LOWER bridge floor
  cannot newly admit them on brightness. More fundamentally, NONE of the 3 false
  joins is in Path 2: a false join with `mean_cos >= 0.75` would already be
  Path-2-accepted at today's 0.90 floor (its 0.97–0.99 bridge clears 0.90), yet all
  3 are rejected — so all 3 have policy `mean_cos < 0.75` and never enter Path 2.
  Relaxing Path 2's bridge floor does NOT change which candidates ENTER Path 2
  (that is the untouched `mean_cos >= 0.75` cos gate), so it cannot admit any train
  false join. DISTINCT from the rejected gen5 bridge relaxation: gen5 dropped
  `IMAGE_BRIDGE_RATIO_MIN` UNIFORMLY to 0.83 across ALL image paths and merely TIED
  the parent (reverted). This is (a) geometry-CONDITIONED — relaxed ONLY in the
  strong-colinearity Path 2 (`mean_cos >= 0.75`) with Path 3 kept strict at 0.90 —
  and (b) operates in today's policy where gen7's mean-cos paths and gen9's
  degree-aware caliber route many more reals through the Path-2 bridge check, so the
  marginal effect differs. It mirrors gen9's successful pattern of CONDITIONING a
  threshold on a feature (gen9 conditioned the caliber cap on `deg_b`; this
  conditions the bridge floor on the colinearity tier). Path 1 (strict 0.94
  worst-arm `_is_colinear_split`), the gap cap `GAP_THRESHOLD_UM`,
  `IMAGE_MIN_COLINEAR_COS` (0.75), `IMAGE_LOWCOS_MIN_COLINEAR_COS` (0.60),
  `MIN_ARM_COS` (0.35), `IMAGE_GAP_MIN_UM` (1.0), `RAD_RATIO_MAX` (1.4),
  `RAD_RATIO_MAX_SHAFT` (2.5), and Path 3's bridge floor (0.90) are all UNCHANGED.
- **Gen 13:** added Path 1c — a SMALL-GAP geometry-only accept (new constants
  `SMALL_GAP_UM = 1.5` and `SMALL_GAP_MIN_COS = 0.50`). DIAGNOSIS: the policy is
  RECALL-bound, stuck at split-repair score 97 (gen12 confusion: REAL 535 =
  accepted 97 / MISSED 438; FALSE 3, all correctly rejected, 0 wrongly accepted).
  The previously-tried bridge-floor levers are EXHAUSTED (they either tied or gave
  held-out +0.000 because brain-specific bridge brightness does not generalize), and
  the prior any-gap very-high-cos geometry accept TIED the parent (every site it
  reached with a bright bridge was already image-accepted). KEY INSIGHT: the image
  paths (Path 2 / Path 3) BOTH require `gap >= IMAGE_GAP_MIN_UM` (1.0) and Path 1
  requires the strict 0.94 worst-arm test, so small-gap reals that are merely
  modestly colinear are rejected by EVERY existing path — a genuinely unreached
  region of candidate space. The gen12 MISSED bucket contains exactly such reals,
  e.g. gap 0.61 / colinear_cos 0.78 (deg_b 2, rad_ratio 1.14). FIX: insert Path 1c
  IMMEDIATELY after Path 1 (reusing the already-computed `hc = _colinear_half_cosines`)
  and BEFORE the image block; accept (geometry only, no reader call, no caliber gate)
  when `s.gap_um <= SMALL_GAP_UM` (1.5) AND `min(c1, c2) >= MIN_ARM_COS` (0.35, both
  arms forward) AND `0.5 * (c1 + c2) >= SMALL_GAP_MIN_COS` (0.50, modest mean
  colinearity). SAFETY (zero new false merges on train): all 3 train FALSE joins sit
  at `gap >= 2.20` (2.20 / 3.00 / 3.16), well above the 1.5 µm ceiling, so NONE can
  enter Path 1c; the `MIN_ARM_COS` floor rejects fragments that double back (negative
  half-cosine). WHY IT SHOULD GENERALIZE: it is PURE GEOMETRY (gap + colinearity, no
  fluorescence-bridge brightness and no caliber gate), so it does not depend on the
  brain-specific bridge brightness that made the bridge levers fail on held-out. This
  is DISTINCT from the tied prior geometry accept (which was ANY-gap + very-high cos):
  Path 1c is SMALL-gap + MODEST cos, opening the small-gap pocket the gap >= 1.0 image
  paths cannot reach rather than re-cutting the already-image-accepted high-cos band.
  Path 1, the image paths (Paths 2/3), the degree-aware caliber logic, the helpers,
  `ENUM_PARAMS`, and every other constant are byte-for-byte UNCHANGED.
- **Gen 16:** lowered Path 3's colinearity floor `IMAGE_LOWCOS_MIN_COLINEAR_COS`
  from 0.60 to 0.50, widening the caliber-gated, bright-bridge-confirmed image tier
  DOWN into the `[0.50, 0.60)` mean-colinearity band (Path 3's accept band is now
  `[0.50, 0.75)`). This is the ONLY functional change — one numeric literal — plus
  comment/criteria text updates. DIAGNOSIS: the policy is RECALL-bound (gen15
  confusion: REAL 535 = accepted 99 / MISSED 436; FALSE 3, all correctly rejected,
  0 wrongly accepted; split-repair score 99). Three pure-geometry attempts to relax
  the mean-colinearity floor in the (1.5, 2.0] band (Gen 12 mean >= 0.85; Gen 14
  mean >= 0.70; Gen 15 caliber-matched mean >= 0.55) ALL TIED the parent and were
  reverted, because the policy's RUNTIME mean colinearity (from
  `_colinear_half_cosines`) runs well BELOW the failure report's `colinear_cos`
  column for those medium-gap sites, so geometry-only cos thresholds fire on nothing
  new. The bridge-floor relaxations (Gen 10/11) are also exhausted (held-out
  +0.000). KEY OBSERVATION: the IMAGE side is reliable and report-aligned — all 90
  reals the policy read have a bright bridge (0.84–1.02) and the warm-start probe
  reports `bridge_ratio` AUC = 0.86 (strong) — BUT Path 3 only ever READS image
  where mean colinearity is already >= 0.60, so real splits sitting just below that
  floor are never looked at even though a bright bridge would confirm them. FIX:
  lower the Path-3 floor to 0.50, opening a NEW colinearity band `[0.50, 0.60)` the
  policy has never read image in; Path 3 otherwise unchanged — still requires gap in
  `[IMAGE_GAP_MIN_UM, GAP_THRESHOLD_UM]`, `min_arm >= MIN_ARM_COS` (0.35), the
  degree-aware caliber cap (`RAD_RATIO_MAX_SHAFT = 2.5` if `deg_b >= 2` else
  `RAD_RATIO_MAX = 1.4`), and a BRIGHT bridge (`bridge_ratio >= IMAGE_BRIDGE_RATIO_MIN`
  = 0.90). REACHABLE MISSED targets: gap 1.96 / cos 0.57 (rad_ratio 2.50, deg_b 2 ->
  shaft cap 2.5, passes) and gap 1.37 / cos 0.50 (rad_ratio 1.94, deg_b 2 -> passes)
  — both below the old 0.60 floor, inside the shaft caliber cap, and (like nearly all
  train reals) bright-bridged. SAFETY (all 3 train FALSE joins stay rejected): the
  gap 2.20 / cos 0.44 / deg_b 2 / rad_ratio 1.97 join has cos 0.44 still BELOW the
  new 0.50 floor (margin 0.06), so the colinearity floor excludes it (do NOT lower
  the floor to <= 0.44); the gap 3.00 / cos 0.06 / deg_b 1 / rad_ratio 1.67 and gap
  3.16 / cos 0.67 / deg_b 1 / rad_ratio 1.87 joins are both tip-to-tip (cap 1.4) with
  rad_ratio >= 1.67 > 1.4, excluded by the caliber gate regardless of cos — zero new
  false-merge risk on train. WHY THIS IS DIFFERENT from the exhausted/tied levers:
  unlike Gen 10/11 (which relaxed the BRIDGE floor on geometry the policy already
  accepted) and Gen 12/14/15 (geometry-only cos tiers that tied because runtime mean
  << report cos), this opens a NEW colinearity band the policy has never spent a
  gated image read in — exactly the image curriculum's intent: spend a gated cloud
  read to RAISE RECALL on geometry-conservative reals confirmed by a bright bridge.
  Path 1, Path 1c, Path 2, the bridge floors (0.90 / 0.85), the caliber caps
  (1.4 / 2.5), `IMAGE_GAP_MIN_UM`, `MIN_ARM_COS`, the helpers, `ENUM_PARAMS`, and
  every other constant are byte-for-byte UNCHANGED.
- **Gen 17:** raised Path 3's tip-to-SHAFT caliber ceiling `RAD_RATIO_MAX_SHAFT`
  from 2.5 to 2.8 — a SINGLE numeric-literal change at module level. This is the
  ONLY functional change (plus comment/criteria text updates). `RAD_RATIO_MAX`
  (1.4, the tip-to-TIP cap) and every other constant are byte-for-byte UNCHANGED.
  DIAGNOSIS: the stream is RECALL-bound. Gen 16 confusion: REAL 535 = accepted 111 /
  MISSED 424; FALSE 3 (0 wrongly accepted / 3 correctly rejected); split-repair score
  111 (it PASSED the gate, raising correct from 99 to 111). To gain we must raise
  recall with ZERO new false merges. Path 3 is the ONLY accept path with a caliber
  gate, and in its `[0.50, 0.75)` runtime-mean band, tip-to-shaft reals (`deg_b >= 2`)
  are rejected purely on the 2.5 shaft cap. A concrete reachable MISSED real in the
  Gen 16 report is gap 1.65 / colinear_cos 0.78 / deg_b 2 / rad_ratio 2.69 — high
  colinearity, killed ONLY by `rad_ratio` 2.69 > 2.5. Raising the shaft cap to 2.8
  lets Path 3 read its (bright) bridge and merge it. (424 missed reals total, only 60
  shown, so additional `deg_b >= 2` reals with `rad_ratio` in `(2.5, 2.8]` and mean in
  `[0.50, 0.75)` are likely present too.) RATIONALE: a tip-to-shaft reconnection
  legitimately has a large caliber mismatch (thin tip into a thicker mid-shaft);
  accepted reals already reach `rad_ratio` up to 3.00 (e.g. gap 1.73 / cos 0.77 /
  rad_ratio 3.00, accepted via the no-caliber-gate Path 2), confirming large
  tip-to-shaft mismatch is real when colinear. 2.8 stays a real, bounded ceiling
  (still rejects extreme >2.8 mismatches). SAFETY (all 3 train FALSE joins stay
  rejected): NONE is gated by the shaft cap. The only `deg_b >= 2` false join (gap
  2.20 / cos 0.44 / rad_ratio 1.97) has rad_ratio 1.97 already well inside 2.5 and is
  held out by the 0.50 cos floor (its cos 0.44 < 0.50); the two `deg_b = 1` false
  joins (gap 3.00 / cos 0.06 / rad_ratio 1.67 and gap 3.16 / cos 0.67 / rad_ratio
  1.87) are gated by the UNCHANGED tip-to-tip cap (1.4 < 1.67 / 1.87). Raising the
  shaft cap from 2.5 to 2.8 cannot admit any of them. WHY THIS IS DIFFERENT from
  every prior reverted attempt: Gen 12/14/15 relaxed cos floors (tied — runtime mean
  ran below report cos), Gen 10/11 relaxed bridge floors (held-out +0.000), Gen 13
  was small-gap geometry. The caliber CEILING has never been raised (Gen 9 introduced
  it at 2.5; Gen 16 lowered the cos floor to 0.50). Path 1, Path 1c, Path 2, the
  bridge floors (0.90 / 0.85), `RAD_RATIO_MAX` (1.4), `IMAGE_LOWCOS_MIN_COLINEAR_COS`
  (0.50), `IMAGE_GAP_MIN_UM`, `MIN_ARM_COS`, the helpers, `ENUM_PARAMS`, and every
  other constant are byte-for-byte UNCHANGED.
- **Gen 18:** raised Path 3's tip-to-TIP caliber ceiling `RAD_RATIO_MAX` from 1.4 to
  1.45 — a SINGLE numeric-literal change at module level. This is the ONLY functional
  change (plus comment/criteria text updates). `RAD_RATIO_MAX_SHAFT` (2.8, the
  tip-to-shaft cap) and every other constant are byte-for-byte UNCHANGED. This is the
  SYMMETRIC partner to Gen 17 (which raised the tip-to-SHAFT cap 2.5 -> 2.8 and
  PASSED); the tip-to-TIP ceiling had been frozen at 1.4 since Gen 9. DIAGNOSIS: the
  stream is RECALL-bound. Gen 17 confusion: REAL 535 = accepted 123 / MISSED 412;
  FALSE 3 (0 wrongly accepted / 3 correctly rejected); split-repair score 123 (it
  PASSED the gate, up from 111). To gain we must raise recall with ZERO new false
  merges. Path 3 is the ONLY accept path with a caliber gate; for tip-to-tip partners
  (`deg_b <= 1`) it applies the strict `RAD_RATIO_MAX = 1.4`. Concrete reachable
  MISSED reals in the Gen 17 report blocked SOLELY by this cap: gap 2.00 /
  colinear_cos 0.70 / deg_b 1 / rad_ratio 1.44, and gap 2.00 / colinear_cos 0.58 /
  deg_b 1 / rad_ratio 1.41 — both decently colinear, rejected only because rad_ratio
  nudges just over 1.4. Raising the cap to 1.45 lets Path 3 read their (bright)
  bridge and merge them. (412 missed reals total, only 60 shown, so more `deg_b <= 1`
  reals with `rad_ratio` in `(1.4, 1.45]` are likely present.) RATIONALE: the caliber
  ratio `_rad_ratio` is report-aligned and reliable (unlike the runtime mean
  colinearity, which runs below the report `colinear_cos` and made the Gen 12/14/15
  cos-floor levers TIE) — which is exactly why the Gen 17 caliber lever fired cleanly.
  The accepted-REAL list confirms `deg_b = 1` reals at report cos 0.45–0.46 already
  reach Path 3 (their runtime mean clears 0.50), so a `deg_b = 1` site at report cos
  0.58–0.70 will too. SAFETY (all 3 train FALSE joins stay rejected): the 3 train
  FALSE joins are (gap 2.20 / cos 0.44 / deg_b 2 / rad_ratio 1.97), (gap 3.00 / cos
  0.06 / deg_b 1 / rad_ratio 1.67), (gap 3.16 / cos 0.67 / deg_b 1 / rad_ratio 1.87).
  The two tip-to-tip (`deg_b 1`) false joins are at rad_ratio 1.67 and 1.87 — BOTH
  still exceed the new 1.45 cap (margins 0.22 and 0.42), so they stay rejected by the
  caliber gate; the `deg_b 2` false join uses the UNCHANGED shaft cap (2.8) and is
  held out by the 0.50 cos floor (its cos 0.44 < 0.50). Raising the tip-to-tip cap
  from 1.4 to 1.45 cannot admit any of them. The raise is kept small (1.45, not
  higher) precisely to preserve the margin to the 1.67 false join — do NOT raise it
  toward 1.67. WHY DIFFERENT from every prior reverted/exhausted attempt: Gen 12/14/15
  relaxed cos floors (tied — runtime mean ran below report cos), Gen 10/11 relaxed
  bridge floors (held-out +0.000), Gen 13 was small-gap geometry, Gen 16 lowered the
  cos floor to 0.50, Gen 17 raised the SHAFT cap. The tip-to-TIP caliber ceiling has
  never been moved. Path 1, Path 1c, Path 2, the bridge floors (0.90 / 0.85),
  `RAD_RATIO_MAX_SHAFT` (2.8), `IMAGE_LOWCOS_MIN_COLINEAR_COS` (0.50),
  `IMAGE_GAP_MIN_UM`, `MIN_ARM_COS`, the helpers, `ENUM_PARAMS`, and every other
  constant are byte-for-byte UNCHANGED.
- **Gen 19:** added Path 4 — a TIGHTLY-MATCHED-CALIBER image tier — with two new
  module-level constants (`IMAGE_MATCHED_CALIBER_MAX = 1.5` and
  `IMAGE_MATCHED_MIN_COLINEAR_COS = 0.35`). It is the FIRST tier to lower the
  mean-colinearity floor BELOW 0.50, made safe by a NEW tight caliber gate. DIAGNOSIS:
  the stream is RECALL-bound. Gen 18 confusion: REAL 535 = accepted 126 / MISSED 409;
  FALSE 3 (0 wrongly accepted / 3 correctly rejected); split-repair score 126 (PASSED
  the gate, up from 123). Precision is perfect; the only gain is more recall with zero
  new false merges. The caliber-CAP family is tapped (Gen 17/18 raised the shaft/tip
  caps); the remaining `deg_b >= 2` misses already sit inside the 2.8 shaft cap and
  the `deg_b <= 1` misses above rad 1.45 are low-cos or above the rad-1.67 false join,
  so those misses are NOT blocked by caliber — they are blocked by the 0.50 mean
  colinearity floor (their runtime mean_cos < 0.50). KEY STRUCTURAL FACT: all 3 train
  FALSE joins have HIGH caliber mismatch — rad_ratio 1.97 / 1.67 / 1.87, ALL >= 1.67.
  So a TIGHTLY caliber-matched gate (rad_ratio <= 1.5) excludes ALL THREE by caliber
  ALONE, independent of cos; WITHIN that tightly-matched tier the cos floor can be
  safely lowered below 0.50. FIX: inserted Path 4 AFTER Path 3, inside the same
  `if reader is not None and hc is not None:` block, gated on `s.gap_um in [1.0, 4.0]`
  AND `IMAGE_MATCHED_MIN_COLINEAR_COS <= mean_cos < IMAGE_LOWCOS_MIN_COLINEAR_COS`
  (i.e. mean_cos in `[0.35, 0.50)`, DISJOINT from Path 3's `[0.50, 0.75)` band so no
  double-count) AND `min_arm >= MIN_ARM_COS` (0.35, both arms gently forward) AND
  `_rad_ratio(g, ctx, s) <= IMAGE_MATCHED_CALIBER_MAX` (1.5); it then reads
  `gap_bridge_evidence` and merges only when `bridge_ratio` is non-NaN and
  `>= IMAGE_BRIDGE_RATIO_MIN` (the strict 0.90 bright-bridge floor). Per the image
  curriculum, the cloud read is gated behind cheap geometry (gap + matched mean band +
  weak-arm + tight caliber) so only a handful of candidates trigger it, and it is
  spent to RAISE RECALL (accepting MISSED reals confirmed by a bright continuous
  bridge — e.g. the report's gap 1.41 / colinear_cos 0.37 / deg_b 1 / rad_ratio 1.34,
  and similar tightly-matched-caliber misses among the 349 not shown). BIOLOGICAL
  RATIONALE: nearly-identical neurite caliber + a bright continuous fluorescence
  bridge together supply the precision that strong colinearity normally would, so a
  gently-colinear break (a natural bend, or a short fragment yielding a noisy tangent)
  is very likely ONE neuron. TRAIN-SAFETY (zero false-merge risk): the 3 train FALSE
  joins are (gap 2.20 / cos 0.44 / deg_b 2 / rad_ratio 1.97), (gap 3.00 / cos 0.06 /
  deg_b 1 / rad_ratio 1.67), (gap 3.16 / cos 0.67 / deg_b 1 / rad_ratio 1.87). EVERY
  one has rad_ratio >= 1.67 > 1.5 = `IMAGE_MATCHED_CALIBER_MAX`, so Path 4's tight
  caliber gate rejects all three REGARDLESS of cos or bridge brightness. (Note the
  0.50 cos floor could NOT be lowered globally: the deg_b 2 false join at cos 0.44 /
  rad_ratio 1.97 has a bright ~0.97 bridge, so only the cos floor keeps it out
  globally — but in Path 4 the tight caliber gate keeps it out instead.) WHY DIFFERENT
  from every prior reverted/exhausted lever: Gen 12/14/15 relaxed GEOMETRY-ONLY cos
  tiers (tied — runtime mean ran below report cos); Gen 10/11 relaxed the BRIDGE floor
  (held-out +0.000); Gen 13 was small-gap geometry; Gen 16 lowered the Path-3 cos
  floor GLOBALLY to 0.50 (pinned there by the deg_b 2 false join); Gen 17/18 raised the
  caliber CAPS (2.8 shaft / 1.45 tip-tip). Path 4 is the FIRST tier to lower the cos
  floor below 0.50, made safe by a NEW tight caliber gate that excludes all 3 false
  joins by caliber alone — a structurally new use of caliber, not a cap raise. Path 1,
  Path 1c, Path 2, Path 3, the bridge floors (0.90 / 0.85), the caliber caps
  (1.45 / 2.8), `IMAGE_LOWCOS_MIN_COLINEAR_COS` (0.50), `IMAGE_GAP_MIN_UM`,
  `MIN_ARM_COS`, the helpers, `ENUM_PARAMS`, and every other constant are byte-for-byte
  UNCHANGED.
- **Gen 20:** EXTENDED Path 4 downward into the `[0.25, 0.50)` mean-colinearity band
  by giving it its OWN, lower weak-arm floor — one new module-level constant
  (`IMAGE_MATCHED_MIN_ARM_COS = 0.25`), one numeric-literal change
  (`IMAGE_MATCHED_MIN_COLINEAR_COS` `0.35 -> 0.25`), and one in-gate comparison swap
  (Path 4's weak-arm test `min_arm >= MIN_ARM_COS` `-> min_arm >=
  IMAGE_MATCHED_MIN_ARM_COS`). DIAGNOSIS: the stream is RECALL-bound. Gen 19
  confusion: REAL 535 = accepted 130 / MISSED 405; FALSE 3 (0 wrongly accepted / 3
  correctly rejected); split-repair score 130 (PASSED the gate, up from 126).
  Precision is perfect; the only gain is more recall with zero new false merges.
  Path 4 (the tight-caliber image tier) currently bottoms out at mean_cos 0.35, but
  that floor is effectively PINNED by the SHARED weak-arm floor: Path 4 required
  `min_arm >= MIN_ARM_COS = 0.35`, and since `min_arm <= mean_cos` ALWAYS, that
  already forces `mean_cos >= 0.35`. So the `[0.35, 0.50)` band could not be extended
  downward without giving Path 4 its OWN weak-arm floor. The remaining reachable
  tight-caliber (rad_ratio <= 1.5) misses sit JUST below 0.35 — e.g. the report's
  gap 1.80 / colinear_cos 0.29 / deg_b 2 / rad_ratio 1.25 (gently curved, tightly
  matched caliber), plus similar tight-caliber misses among the 345 not shown — all
  real splits confirmed (on this brain, all reals) by bright bridges. FIX: added
  `IMAGE_MATCHED_MIN_ARM_COS = 0.25` (Path-4-LOCAL, decoupled from the global
  `MIN_ARM_COS` of 0.35 that still guards Paths 1c/2/3), changed
  `IMAGE_MATCHED_MIN_COLINEAR_COS` `0.35 -> 0.25`, and in Path 4's gate ONLY replaced
  `min_arm >= MIN_ARM_COS` with `min_arm >= IMAGE_MATCHED_MIN_ARM_COS`. The
  `IMAGE_MATCHED_CALIBER_MAX` (1.5), the strict bridge floor `IMAGE_BRIDGE_RATIO_MIN`
  (0.90), and the disjoint upper bound `< IMAGE_LOWCOS_MIN_COLINEAR_COS` (0.50) are
  UNCHANGED. After this, Path 4 admits tightly-matched-caliber reals with mean_cos in
  `[0.25, 0.50)` and BOTH arms gently forward (`min_arm >= 0.25`), confirmed by a
  bright bridge. KEY STRUCTURAL FACT (unchanged): all 3 train FALSE joins have
  rad_ratio 1.97 / 1.67 / 1.87, ALL >= 1.67 > 1.5 = `IMAGE_MATCHED_CALIBER_MAX`, so
  Path 4's tight caliber gate excludes EVERY false join by CALIBER ALONE — regardless
  of cos OR weak-arm angle OR bridge brightness. TRAIN-SAFETY (zero false-merge
  risk): lowering Path 4's cos floor (`0.35 -> 0.25`) and its weak-arm floor
  (`0.35 -> 0.25`) therefore opens ZERO new false-merge risk on the train set — the
  cos/weak-arm floors are NOT what holds the false joins out of Path 4, the tight
  caliber gate is. The change is confined to Path 4; the global `MIN_ARM_COS` (0.35)
  used by Paths 1c/2/3 is UNTOUCHED. BIOLOGICAL RATIONALE: when neurite caliber is
  tightly matched across the gap AND a bright continuous fluorescence bridge spans it,
  the fragments are very likely ONE neuron even when the tip-tangent geometry is only
  WEAKLY colinear (a natural bend at the break, or a short fragment giving a noisy
  tangent); the tight caliber + bright bridge supply the precision that strong
  colinearity normally would, while the strictly-positive 0.25 weak-arm floor still
  rejects arms that double back or cross sideways (the genuine false-join signature).
  WHY DIFFERENT from prior gens: Gen 16 lowered the Path-3 cos floor GLOBALLY to 0.50;
  Gen 17/18 raised the caliber CAPS (2.8 shaft / 1.45 tip-tip); Gen 19 CREATED Path 4
  at floor 0.35 (PINNED there by the shared 0.35 weak-arm floor). Gen 20 is the FIRST
  to give Path 4 its own DECOUPLED weak-arm floor, unpinning the cos floor and
  extending the tight-caliber tier into `[0.25, 0.35)`. (Gen 10/11 bridge-floor
  relaxations are NOT repeated — the bridge floor stays 0.90.) Path 1, Path 1c,
  Path 2, Path 3, the bridge floors (0.90 / 0.85), the caliber caps (1.45 / 2.8 / 1.5),
  `IMAGE_LOWCOS_MIN_COLINEAR_COS` (0.50), `IMAGE_GAP_MIN_UM`, the global `MIN_ARM_COS`
  (0.35), the helpers, `ENUM_PARAMS`, and every other constant are byte-for-byte
  UNCHANGED.
- **Gen 22:** made Path 4's caliber cap DEGREE-AWARE, mirroring Path 3 — one new
  module-level constant (`IMAGE_MATCHED_CALIBER_MAX_SHAFT = 1.85`), one new in-gate
  line (`cap4 = IMAGE_MATCHED_CALIBER_MAX_SHAFT if (db is not None and db >= 2) else
  IMAGE_MATCHED_CALIBER_MAX`), and one gate-clause swap (Path 4's
  `_rad_ratio(g, ctx, s) <= IMAGE_MATCHED_CALIBER_MAX` `-> <= cap4`). This is the ONLY
  functional change. DIAGNOSIS: the stream is RECALL-bound with perfect train precision
  (170 correct / 0 false merges; 365 real splits still MISSED; all 3 false joins
  correctly rejected). Path 4 used a single FLAT caliber cap of 1.5 for BOTH endpoint
  kinds, but a tip-to-SHAFT partner (`deg_b >= 2`) legitimately has a LARGER caliber
  mismatch (thin tip vs thicker mid-shaft) — the same biological fact that justified
  Path 3's degree-aware cap — so the flat 1.5 was over-rejecting low-cos tight-ish
  shaft reals. FIX: in Path 4 ONLY, key the cap on the partner degree `db` (already
  computed for Path 3 just above, in the same try-block scope): `db >= 2` (shaft) uses
  `IMAGE_MATCHED_CALIBER_MAX_SHAFT = 1.85`, `db <= 1` or unknown keeps
  `IMAGE_MATCHED_CALIBER_MAX = 1.5`. TRAIN-SAFETY (zero new false-merge risk): from the
  Gen 21 SplitSite audit, FALSE = 3, all correctly rejected. Crossed with endpoint
  degree and Path 4's `[0.0, 0.50)` cos band: the ONLY `deg_b >= 2` (shaft) false join
  sits at rad_ratio 1.97 (gap 2.20 / cos 0.44 / deg_b 2) — the new shaft cap 1.85 is
  strictly below it (margin 0.12), so it stays rejected by caliber; the `deg_b = 1`
  (tip) false join inside this cos band sits at rad_ratio 1.67 (gap 3.00 / cos 0.06 /
  deg_b 1) and the tip cap stays at 1.5 < 1.67; the third false join (cos 0.67) is in
  Path 3's band, not Path 4's, and is unaffected. So raising ONLY the shaft cap to 1.85
  adds ZERO new false-merge risk on train. Concrete reachable MISSED shaft reals this
  opens (deg_b 2, cos in [0, 0.5), rad in (1.5, 1.85], currently rejected): gap 1.35 /
  cos 0.07 / rad 1.82; gap 1.76 / cos 0.15 / rad 1.84; gap 1.92 / cos 0.20 / rad 1.81;
  gap 1.22 / cos 0.03 / rad 1.52; gap 1.72 / cos 0.41 / rad 1.53 — each still gated by
  the strict bright-bridge floor (`bridge_ratio >= 0.90`). WHY DIFFERENT from prior
  gens: Gen 19 created Path 4 with a flat tight cap; Gen 20/21 lowered Path 4's
  cos/weak-arm floors to the perpendicular boundary; Gen 22 is the FIRST to give Path 4
  a degree-aware caliber cap (the same lever Gen 9 applied to Path 3), harvesting the
  bucket of low-cos tight-ish shaft reals. Paths 1/1c/2/3, the global `MIN_ARM_COS`
  (0.35), `RAD_RATIO_MAX` (1.45), `RAD_RATIO_MAX_SHAFT` (2.8),
  `IMAGE_MATCHED_CALIBER_MAX` (1.5, now the tip/unknown cap),
  `IMAGE_MATCHED_MIN_COLINEAR_COS` (0.0), `IMAGE_MATCHED_MIN_ARM_COS` (0.0),
  `IMAGE_BRIDGE_RATIO_MIN` (0.90), the bridge floors (0.90 / 0.85), the helpers,
  `ENUM_PARAMS`, and every other constant are byte-for-byte UNCHANGED.
- **Gen 21:** EXTENDED Path 4's mean-colinearity band down to the PERPENDICULAR
  boundary by lowering BOTH of its Path-4-local floors from 0.25 to 0.0 — two
  numeric-literal changes at module level (`IMAGE_MATCHED_MIN_COLINEAR_COS` `0.25 ->
  0.0` and `IMAGE_MATCHED_MIN_ARM_COS` `0.25 -> 0.0`). This is the ONLY functional
  change (plus comment/criteria text updates); the gate body and Paths 1/1c/2/3 are
  byte-for-byte UNCHANGED. DIAGNOSIS: the stream is RECALL-bound and precision is
  perfect (Gen 20 SplitSite audit: FALSE = 3, all correctly rejected). Path 4's
  `[0.25, 0.50)` band leaves tight-caliber reals JUST below 0.25 unreached — e.g. the
  report's gap 0.85 / cos 0.05 / deg_b 2 / rad_ratio 1.22; gap 1.38 / cos 0.10 /
  deg_b 2 / rad_ratio 1.15; gap 1.00 / cos 0.18 / deg_b 1 / rad_ratio 1.00; gap 1.00 /
  cos 0.15 / deg_b 1 / rad_ratio 1.34 — all tightly-matched caliber, low-but-FORWARD
  cos, and (on this brain, like nearly all reals) bright-bridged, yet rejected solely
  by the 0.25 cos/weak-arm floors. FIX: lowered both Path-4-local floors to 0.0,
  extending the band to `[0.0, 0.50)`. KEY STRUCTURAL FACT (unchanged): the Gen 20
  SplitSite audit's 3 FALSE joins all have rad_ratio >= 1.67 (1.97 / 1.67 / 1.87),
  EVERY one ABOVE Path 4's tight caliber cap of 1.5 — so within the `rad_ratio <= 1.5`
  regime there are ZERO false joins on the train set. TRAIN-SAFETY (zero new
  false-merge risk): CALIBER, not colinearity, is the precision guard in this tier;
  the tight caliber gate excludes all 3 false joins by caliber ALONE, so lowering the
  cos/weak-arm floors to the perpendicular boundary adds ZERO new false-merge risk on
  the train set. The `min_arm >= 0.0` floor still REFUSES any arm with negative
  forward cosine (sideways/backward — the double-back / crossing false-join
  signature), admitting only forward-or-perpendicular arms — so it is NOT an
  unconstrained accept. The strict bright-bridge requirement
  (`IMAGE_BRIDGE_RATIO_MIN = 0.90`) is RETAINED (image warm-start probe AUC 0.86,
  strong). WHY DIFFERENT from prior gens: Gen 19 created Path 4 at cos floor 0.35;
  Gen 20 gave it its own decoupled weak-arm floor and dropped both to 0.25; Gen 21
  reaches the perpendicular boundary (0.0), the structural limit of "forward-or-
  perpendicular arms." `IMAGE_MATCHED_CALIBER_MAX` (1.5), `IMAGE_BRIDGE_RATIO_MIN`
  (0.90), the upper bound `< IMAGE_LOWCOS_MIN_COLINEAR_COS` (0.50), Paths 1/1c/2/3,
  the bridge floors (0.90 / 0.85), the caliber caps (1.45 / 2.8), the global
  `MIN_ARM_COS` (0.35), the helpers, `ENUM_PARAMS`, and every other constant are
  byte-for-byte UNCHANGED.
- **Gen 23:** made Path 4's tip-to-SHAFT caliber cap GAP-CONDITIONED — two new
  module-level constants (`MATCHED_SMALLGAP_UM = 2.0` and
  `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP = 2.5`) and a replacement of the single
  `cap4 = ...` ternary with an if/else block. This is the ONLY functional change (plus
  comment/criteria text updates). DIAGNOSIS: the stream is RECALL-bound with perfect
  precision (Gen 22 SplitSite audit: 180 correct / 0 false; 355 real splits still
  MISSED; all 3 false joins correctly rejected; image warm-start probe AUC 0.86,
  strong). Path 4's shaft cap (1.85) exists ONLY to fence against the single deg_b 2
  false join (gap 2.20 / cos 0.44 / rad_ratio 1.97), so its margin (0.12) is what
  blocks tightly-but-not-tightly-matched shaft reals. KEY STRUCTURAL FACT (Gen 22
  audit "Correctly refused" table): ALL 3 train false joins sit at gap >= 2.20 (2.20 /
  3.00 / 3.16) — the smallest false-join gap is 2.20. So the gap regime `< 2.0` is
  FALSE-JOIN-FREE, and the shaft caliber cap can be loosened there freely with no fence
  to maintain. FIX: added `MATCHED_SMALLGAP_UM = 2.0` and
  `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP = 2.5`, then replaced the single
  `cap4 = IMAGE_MATCHED_CALIBER_MAX_SHAFT if (db is not None and db >= 2) else
  IMAGE_MATCHED_CALIBER_MAX` line with:
  `if db is not None and db >= 2: cap4 = IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP if
  s.gap_um < MATCHED_SMALLGAP_UM else IMAGE_MATCHED_CALIBER_MAX_SHAFT; else: cap4 =
  IMAGE_MATCHED_CALIBER_MAX`. So a shaft partner (`db >= 2`) at gap < 2.0 uses the
  looser 2.5 cap; at gap >= 2.0 it keeps the regular 1.85 cap; tip-to-tip / unknown
  keeps 1.5. TRAIN-SAFETY (zero new false-merge risk): the new 2.5 cap applies ONLY at
  gap < 2.0, strictly BELOW the smallest false-join gap (2.20), so NO train false join
  is in that regime — loosening caliber there adds ZERO new false-merge risk (the
  safety no longer rests on the thin 0.12 caliber margin, but on a gap regime that
  contains no false joins at all). The deg_b 2 false join (gap 2.20 / cos 0.44 /
  rad_ratio 1.97) sits ABOVE 2.0, so it still uses the 1.85 shaft cap and stays
  rejected (1.97 > 1.85); the tip cap (1.5) and the two tip false joins (gap 3.00 /
  3.16) are unaffected; the bright-bridge floor (0.90) is retained. Concrete reachable
  MISSED shaft reals this opens (deg_b 2, gap < 2.0, cos in [0, 0.5), rad in
  (1.85, 2.5], currently rejected): gap 1.33 / cos 0.35 / rad 2.41; gap 1.68 / cos
  0.34 / rad 2.01. WHY DIFFERENT from prior gens: Gen 19 created Path 4 with a flat
  tight cap; Gen 20/21 lowered Path 4's cos/weak-arm floors; Gen 22 gave Path 4 a
  degree-aware caliber cap (shaft 1.85). Gen 23 is the FIRST to make the shaft cap
  GAP-conditioned, exploiting the false-join-free small-gap regime. Paths 1/1c/2/3,
  the global `MIN_ARM_COS` (0.35), `RAD_RATIO_MAX` (1.45), `RAD_RATIO_MAX_SHAFT` (2.8),
  the Path 4 gate body, the `[0.0, 0.50)` cos band, `IMAGE_MATCHED_MIN_ARM_COS` (0.0),
  `IMAGE_BRIDGE_RATIO_MIN` (0.90), `IMAGE_MATCHED_CALIBER_MAX` (1.5),
  `IMAGE_MATCHED_CALIBER_MAX_SHAFT` (1.85), the bridge floors (0.90 / 0.85), the
  helpers, `ENUM_PARAMS`, and every other constant are byte-for-byte UNCHANGED.
- **Gen 24:** made Path 4's tip-to-TIP caliber cap GAP-CONDITIONED, mirroring Gen 23's
  shaft treatment — two new module-level constants (`TIP_SMALLGAP_UM = 2.5` and
  `IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP = 1.85`) and a replacement of ONLY the `else:`
  branch's `cap4 = IMAGE_MATCHED_CALIBER_MAX` line with a gap ternary. This is the ONLY
  functional change (plus comment/criteria text updates). DIAGNOSIS: the stream is
  RECALL-bound with perfect train precision (185 correct / 0 false merges; 350 real
  splits still MISSED; all 3 false joins correctly rejected; image warm-start probe AUC
  0.86, strong). Path 4's tip-tip cap (a flat 1.5) blocks tightly-but-not-tightly-matched
  tip-tip reals. KEY STRUCTURAL FACT (Gen 23 SplitSite audit "Correctly refused" table):
  the two TIP-TIP (`deg_b = 1`) false joins sit at gap 3.00 (rad 1.67) and gap 3.16
  (rad 1.87) — BOTH at gap >= 3.00. So the tip-tip gap regime `< 2.5` is
  TIP-TIP-FALSE-JOIN-FREE (smallest tip-tip false-join gap 3.00, a 0.5 µm buffer), and
  the tip caliber cap can be loosened there freely with no fence to maintain. FIX: added
  `TIP_SMALLGAP_UM = 2.5` and `IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP = 1.85`, then
  replaced the `else:` body `cap4 = IMAGE_MATCHED_CALIBER_MAX` with
  `cap4 = IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP if s.gap_um < TIP_SMALLGAP_UM else
  IMAGE_MATCHED_CALIBER_MAX`. So a tip-to-tip / unknown-degree partner (`db <= 1` or
  `None`) at gap < 2.5 uses the looser 1.85 cap; at gap >= 2.5 it keeps the regular 1.5
  cap; the shaft branch (`db >= 2`) is byte-for-byte UNCHANGED. TRAIN-SAFETY (zero new
  false-merge risk): the new 1.85 tip cap applies ONLY at gap < 2.5, strictly BELOW the
  smallest tip-tip false-join gap (3.00), so NO tip-tip false join is in that regime —
  loosening the tip caliber cap there adds ZERO new false-merge risk (the safety rests on
  a tip-tip-false-join-free gap regime, not a thin caliber margin). The two tip false
  joins (gap 3.00 / rad 1.67; gap 3.16 / rad 1.87) sit ABOVE 2.5, so they keep the 1.5
  tip cap and stay rejected (1.67 / 1.87 > 1.5); the remaining false join (gap 2.20 /
  rad 1.97) is a `deg_b = 2` SHAFT join on the unchanged shaft branch and is unaffected;
  the bright-bridge floor (0.90) is retained. Concrete reachable MISSED tip-tip reals
  this opens (deg_b 1, gap < 2.5, cos in [0, 0.5), rad in (1.5, 1.85], currently
  rejected): gap 1.73 / cos 0.28 / rad 1.84; gap 2.00 / cos 0.21 / rad 1.67. WHY
  DIFFERENT from prior gens: Gen 22 split Path 4's cap by degree (shaft 1.85, tip 1.5);
  Gen 23 made the SHAFT cap gap-conditioned; Gen 24 makes the TIP-TIP cap gap-conditioned
  too, exploiting the same false-join-free small-gap structure for the tip-tip class.
  Paths 1/1c/2/3, the shaft branch, the global `MIN_ARM_COS` (0.35), `RAD_RATIO_MAX`
  (1.45), `RAD_RATIO_MAX_SHAFT` (2.8), the Path 4 gate body, the `[0.0, 0.50)` cos band,
  `IMAGE_MATCHED_MIN_ARM_COS` (0.0), `IMAGE_BRIDGE_RATIO_MIN` (0.90),
  `IMAGE_MATCHED_CALIBER_MAX` (1.5), `IMAGE_MATCHED_CALIBER_MAX_SHAFT` (1.85),
  `MATCHED_SMALLGAP_UM` (2.0), `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP` (2.5), the
  bridge floors (0.90 / 0.85), the helpers, `ENUM_PARAMS`, and every other constant are
  byte-for-byte UNCHANGED.
- **Gen 25:** raised the Path 4 small-gap SHAFT caliber cap
  `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP` from 2.5 to 2.8 — a SINGLE-CONSTANT value
  change. DIAGNOSIS: the policy is RECALL-bound with perfect train precision (192 correct /
  0 false merges; 343 real splits still MISSED; all 3 false joins correctly rejected;
  image warm-start probe AUC 0.86, strong). Path 4's small-gap shaft cap (gap < 2.0,
  `deg_b >= 2`) was holding out tightly-matched, low-cos shaft reals with caliber in
  `(2.5, 2.8]`. KEY STRUCTURAL FACT (gen24 SplitSite audit): of the 3 train FALSE joins
  (all correctly rejected), the ONLY `deg_b >= 2` (shaft) one is at gap 2.20 (rad 1.97);
  the other two are tip-tip at gap 3.00 / 3.16. The looser small-gap shaft cap applies
  ONLY at gap < `MATCHED_SMALLGAP_UM` (2.0), strictly BELOW the shaft false join's gap of
  2.20 — so the small-gap shaft regime is SHAFT-FALSE-JOIN-FREE and the cap is
  effectively unconstrained there. FIX: `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP`
  2.5 -> 2.8 (now equal to Path 3's `RAD_RATIO_MAX_SHAFT`); nothing else touched. TRAIN-
  SAFETY (zero new false-merge risk): the cap change touches only gap < 2.0, where there
  is NO shaft false join; the shaft false join at gap 2.20 sits ABOVE 2.0 so it still
  uses the regular 1.85 shaft cap and stays rejected (1.97 > 1.85); the two tip-tip false
  joins keep the tip cap and are unaffected. The new value 2.8 matches Path 3's existing
  shaft cap and is consistent with observed REAL shaft calibers (accepted shaft reals
  reach rad_ratio ~2.69); the bright-bridge floor (0.90) is retained. Concrete reachable
  MISSED shaft reals this opens (deg_b 2, gap < 2.0, cos in [0, 0.5), rad in (2.5, 2.8],
  currently rejected): gap 1.60 / cos 0.24 / rad 2.64; gap 1.88 / cos 0.29 / rad 2.58;
  gap 1.98 / cos 0.13 / rad 2.56. WHY DIFFERENT from prior gens: Gen 22 split Path 4's cap
  by degree; Gen 23 made the SHAFT cap gap-conditioned (introducing the small-gap shaft
  cap at 2.5); Gen 24 made the TIP-TIP cap gap-conditioned; Gen 25 simply WIDENS the
  already-gap-fenced small-gap shaft cap into the unconstrained false-join-free regime,
  harvesting low-cos wide-caliber small-gap shaft reals. The tip-tip cap, `MATCHED_SMALLGAP_UM`
  (2.0), `TIP_SMALLGAP_UM` (2.5), the Path 4 gate body, the `[0.0, 0.50)` cos band,
  `IMAGE_MATCHED_MIN_ARM_COS` (0.0), `IMAGE_BRIDGE_RATIO_MIN` (0.90), Paths 1/1c/2/3, the
  global `MIN_ARM_COS` (0.35), `RAD_RATIO_MAX` (1.45), `RAD_RATIO_MAX_SHAFT` (2.8), and
  every other constant are byte-for-byte UNCHANGED.
- **Gen 26:** added a tighter-gap, looser-caliber TIP tier to Path 4 — two new
  module-level constants (`TIP_TINYGAP_UM = 2.0` and
  `IMAGE_MATCHED_CALIBER_MAX_TIP_TINYGAP = 2.2`) and a replacement of ONLY the `else:`
  (tip / unknown-degree) branch's single `cap4 = ...` ternary with a three-tier ladder.
  The shaft branch is byte-for-byte UNCHANGED. DIAGNOSIS: the policy is RECALL-bound with
  perfect train precision (Gen 25 parent: 196 correct / 0 false merges; REAL 535 /
  accepted 196 / MISSED 339; all 3 false joins correctly rejected; image warm-start probe
  AUC 0.86, strong). Among the MISSED reals is a clear tip-tip (`deg_b = 1`) cluster at
  very small gap whose ONLY disqualifier is the tip caliber cap — e.g. gap 1.00 / cos 0.06
  / deg_b 1 / rad_ratio 2.00 and gap 1.00 / cos 0.08 / deg_b 1 / rad_ratio 1.95 — both
  blocked by the Gen 24 tip cap (`IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP = 1.85` for gap <
  2.5). KEY STRUCTURAL FACT: BOTH tip-tip false joins are at gap >= 3.00, so the gap < 2.0
  regime is entirely TIP-FALSE-JOIN-FREE with a 1.0 µm margin to the nearest tip false
  join — and genuine tip reals with caliber THIS wide DO exist (the accepted bucket has
  tip-tip reals out to rad_ratio 2.60–3.00 via the high-cos paths), so wide caliber is NOT
  itself a false-join signal at small gap. FIX: added `TIP_TINYGAP_UM = 2.0` and
  `IMAGE_MATCHED_CALIBER_MAX_TIP_TINYGAP = 2.2`, and changed ONLY the tip/unknown-degree
  `else:` branch to the monotonic ladder
  `if s.gap_um < TIP_TINYGAP_UM: cap4 = IMAGE_MATCHED_CALIBER_MAX_TIP_TINYGAP;
  elif s.gap_um < TIP_SMALLGAP_UM: cap4 = IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP;
  else: cap4 = IMAGE_MATCHED_CALIBER_MAX` (looser cap at smaller gap). The shaft branch,
  the Path 4 gate body, the cos band `[0.0, 0.50)`, and the bright-bridge floor are
  byte-for-byte UNCHANGED. TRAIN-SAFETY (zero new false-merge risk): the new 2.2 cap
  applies ONLY at gap < 2.0, strictly BELOW the smallest tip-tip false-join gap (3.00,
  a 1.0 µm buffer), so NO tip-tip false join is in that regime — the gap fence, not
  caliber, does the rejection there. The two tip false joins (gap 3.00 / rad 1.67; gap
  3.16 / rad 1.87) sit at gap >= 2.5 so they keep the regular 1.5 tip cap and stay
  rejected (1.67 / 1.87 > 1.5); the shaft false join (gap 2.20 / rad 1.97) is a
  `deg_b = 2` join on the UNCHANGED shaft branch and is unaffected; each newly-eligible
  candidate still must clear the strict bright-bridge floor (`bridge_ratio >= 0.90`).
  Although the cap 2.2 sits above the tip false-join rad_ratios (1.67 / 1.87), those joins
  are excluded by the gap fence, not by caliber — exactly the Gen 23/24/25 small-gap-regime
  pattern. WHY DIFFERENT from prior gens: Gen 24 made the tip-tip cap gap-conditioned in
  TWO tiers (gap < 2.5 -> 1.85, else 1.5); Gen 26 inserts a THIRD, tighter-gap tier below
  it (gap < 2.0 -> 2.2), exploiting the 1.0 µm false-join-free margin that the two-tier
  ladder did not yet use. Paths 1/1c/2/3, the shaft branch, the global `MIN_ARM_COS`
  (0.35), `RAD_RATIO_MAX` (1.45), `RAD_RATIO_MAX_SHAFT` (2.8), `IMAGE_MATCHED_CALIBER_MAX`
  (1.5), `IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP` (1.85), `TIP_SMALLGAP_UM` (2.5),
  `MATCHED_SMALLGAP_UM` (2.0), `IMAGE_MATCHED_CALIBER_MAX_SHAFT` (1.85),
  `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP` (2.8), the cos band `[0.0, 0.50)`,
  `IMAGE_MATCHED_MIN_ARM_COS` (0.0), `IMAGE_BRIDGE_RATIO_MIN` (0.90), the bridge floors
  (0.90 / 0.85), the helpers, `ENUM_PARAMS`, and every other constant are byte-for-byte
  UNCHANGED.
- **Gen 27:** bumped the SHAFT small-gap gap fence `MATCHED_SMALLGAP_UM` from 2.0 to
  2.18 (single-constant change). DIAGNOSIS: the policy is RECALL-bound at parent score
  197 (gen26 confusion: REAL 535 = accepted 197 / MISSED 338; FALSE 3, all correctly
  rejected). Among the MISSED reals there is a clear SHAFT (`deg_b = 2`) cluster in the
  gap band `[2.0, 2.20)` whose ONLY disqualifier is the strict 1.85 shaft caliber cap
  that applies at gap >= `MATCHED_SMALLGAP_UM` (2.0): e.g. gap 2.08 / cos 0.07 / deg_b 2
  / rad_ratio 2.21; gap 2.15 / cos 0.31 / deg_b 2 / rad_ratio 2.58; gap 2.16 / cos 0.22
  / deg_b 2 / rad_ratio 2.74 — all rejected. Their rad_ratio (2.21 / 2.58 / 2.74)
  EXCEEDS both the 1.85 cap AND the only shaft false join FJ1's rad_ratio 1.97, so
  caliber alone CANNOT separate them from FJ1 (gap 2.20 / cos 0.44 / rad_ratio 1.97) —
  the ONLY separator is the gap fence (targets at 2.08–2.16 vs FJ1 at 2.20). FIX:
  raise `MATCHED_SMALLGAP_UM` 2.0 -> 2.18, so these gap-2.08/2.15/2.16 shaft targets
  fall under the looser `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP` (2.8) cap and become
  eligible. TRAIN-SAFETY (GUARANTEED zero new false merges): the boundary 2.18 sits
  cleanly BETWEEN the highest target (gap 2.16 -> true <= 2.165) and FJ1 (gap 2.20 ->
  true >= 2.195), so the band `[2.0, 2.18)` is SHAFT-FALSE-JOIN-FREE with a
  deterministic margin; FJ1 at gap 2.20 is NOT < 2.18, so it stays on the strict 1.85
  shaft branch and remains rejected (1.97 > 1.85). The two tip false joins (gap 3.00 /
  3.16) are unaffected (tip branch). Wide caliber is NOT a false-join signal at small
  gap — genuine shaft reals with rad_ratio 2.64 / 2.69 already sit in the accepted
  bucket, and the gap < 2.0 looser-cap regime (Gen 25) already passed the held-out
  check. Path 4's bright-bridge floor stays the strict `IMAGE_BRIDGE_RATIO_MIN` (0.90;
  warm-start AUC 0.86) — false joins are uniformly bright 0.97–0.99 so the bridge floor
  does NOT reject them; the gap+caliber geometry does, which this change preserves. ONLY
  `MATCHED_SMALLGAP_UM` (2.0 -> 2.18) changed: the shaft cap-selection line, the tip
  three-tier ladder (`TIP_TINYGAP_UM` 2.0, `IMAGE_MATCHED_CALIBER_MAX_TIP_TINYGAP` 2.2,
  `TIP_SMALLGAP_UM` 2.5, `IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP` 1.85,
  `IMAGE_MATCHED_CALIBER_MAX` 1.5), `IMAGE_MATCHED_CALIBER_MAX_SHAFT` (1.85),
  `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP` (2.8), the Path 4 gate body, the cos band
  `[0.0, 0.50)`, `IMAGE_MATCHED_MIN_ARM_COS` (0.0 — NOT lowered, negative cos =
  double-back = false-join signature, off-limits), Paths 1/1c/2/3, the helpers,
  `ENUM_PARAMS`, and every other constant are byte-for-byte UNCHANGED.
- **Gen 28:** added a MID-GAP false-join-free band override to Path 4's caliber-cap
  selection — three new module-level constants (`MIDGAP_FREE_LO_UM = 2.35`,
  `MIDGAP_FREE_HI_UM = 2.80`, `IMAGE_MATCHED_CALIBER_MAX_MIDGAP = 2.8`) and a LEADING
  branch PREPENDED to the cap-selection block (the existing `if db is not None and
  db >= 2:` shaft branch demoted to `elif`, with the shaft line and the full tip
  three-tier ladder left byte-for-byte unchanged). This is the ONLY functional change
  (plus comment/criteria text updates). DIAGNOSIS: the policy is RECALL-bound at parent
  score 198 (gen27 confusion: REAL 535 / accepted 198 / MISSED 337; FALSE 3, all
  correctly rejected; perfect precision). The geometric caliber/gap levers are largely
  exhausted, but the 3 train false joins BRACKET a mid-gap pocket: FJ1 (the ONLY shaft
  false join) is at gap 2.20 (true >= 2.195) and BOTH tip false joins are at gap >= 3.00
  (true >= 2.995), so the band `[2.35, 2.80)` is FALSE-JOIN-FREE for BOTH endpoint
  degrees. Yet Path 4's caliber caps in that band are tight (shaft 1.85 at gap >= 2.18;
  tip 1.5 at gap >= 2.5), so wide-caliber low-cos reals there — exactly the ambiguous,
  bridge-bright population the warm-start probe (bridge_ratio AUC 0.86) samples (in-band
  reals at gap 2.58–2.67, ALL with bridge_ratio >= 0.98) — are rejected. FIX: PREPEND
  `if MIDGAP_FREE_LO_UM <= s.gap_um < MIDGAP_FREE_HI_UM: cap4 =
  IMAGE_MATCHED_CALIBER_MAX_MIDGAP` (2.8) ahead of the degree split, so both degrees use
  the looser cap inside the false-join-free band. TRAIN-SAFETY (GUARANTEED zero new false
  merges): the band is bracketed by FJ1 at gap 2.20 (0.15 µm below the lower edge 2.35)
  and the tip false joins at gap >= 3.00 (0.20 µm above the upper edge 2.80), so NO train
  false join of either degree lies in `[2.35, 2.80)` — deterministic margins. FJ1 (gap
  2.20, below 2.35) stays on the strict 1.85 shaft branch and remains rejected
  (1.97 > 1.85); both tip false joins (gap >= 3.00, above 2.80) keep the 1.5 tip cap and
  stay rejected (1.67 / 1.87 > 1.5). The override is purely a LOOSENING inside the band
  (2.8 >= every cap that branch previously selected there: shaft 1.85, tip 1.5), so it
  ADDS acceptances and removes none — a monotonic recall gain. Each newly-eligible
  candidate still must clear the strict bright-bridge floor (`bridge_ratio >= 0.90`);
  2.8 stays within observed genuine real caliber (accepted reals reach rad_ratio ~3.0).
  WHY DIFFERENT from prior gens: Gen 23/25/27 widened/extended the SHAFT small-gap fence
  (gap < 2.18); Gen 24/26 gap-conditioned the TIP cap (gap < 2.5 / < 2.0); those all
  exploit the small-gap (below FJ1) regime. Gen 28 is the FIRST to exploit the MID-gap
  pocket ABOVE FJ1 and BELOW the tip false joins, loosening BOTH degrees at once inside a
  band that is false-join-free for either degree. The Path 4 gate body (gap in
  `[IMAGE_GAP_MIN_UM, GAP_THRESHOLD_UM]`, cos band `[0.0, 0.50)`, `min_arm >=
  IMAGE_MATCHED_MIN_ARM_COS` 0.0, `_rad_ratio <= cap4`, `bridge_ratio >=
  IMAGE_BRIDGE_RATIO_MIN` 0.90), the shaft cap-selection line and its constants
  (`IMAGE_MATCHED_CALIBER_MAX_SHAFT` 1.85, `IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP` 2.8,
  `MATCHED_SMALLGAP_UM` 2.18), the tip three-tier ladder and its constants
  (`TIP_TINYGAP_UM` 2.0, `IMAGE_MATCHED_CALIBER_MAX_TIP_TINYGAP` 2.2, `TIP_SMALLGAP_UM`
  2.5, `IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP` 1.85, `IMAGE_MATCHED_CALIBER_MAX` 1.5),
  the cos floor (0.0 — negative cos off-limits), `IMAGE_MATCHED_MIN_ARM_COS` (0.0), the
  0.90 bridge floor, Paths 1/1c/2/3, the helpers, `ENUM_PARAMS`, and every other constant
  are byte-for-byte UNCHANGED.
- **Gen 29:** WIDENED the Gen 28 mid-gap false-join-free band from `[2.35, 2.80)` to
  `[2.30, 2.90)` — a TWO-CONSTANT change: `MIDGAP_FREE_LO_UM` 2.35 -> 2.30 and
  `MIDGAP_FREE_HI_UM` 2.80 -> 2.90. DIAGNOSIS: the parent (Gen 28) scored 204
  (correct=204, false=0) — recall-bound with perfect precision; Gen 28's mid-gap band
  override added +6 correct merges, proving the pocket it sits in is RICH. But that band
  only covered `[2.35, 2.80)` and left the adjacent sub-bands `(2.30, 2.35)` and
  `[2.80, 2.90)` of the SAME false-join-free pocket on the tight degree caps (shaft 1.85
  / tip 1.5). The whole pocket `(2.20, 3.00)` — bracketed by FJ1 (the only SHAFT false
  join, gap 2.20 -> true >= 2.195) below and FJ2/FJ3 (TIP false joins, gap >= 3.00 ->
  true >= 2.995) above — is false-join-free on train. CHANGE: drop the lower edge to 2.30
  (keeps a >=0.095 µm deterministic margin above FJ1; the lone visible lower-sub-band
  target at gap 2.22 stays unreachable — too close to FJ1 — so 2.30 is the floor) and
  raise the upper edge to 2.90 (keeps a >=0.095 µm deterministic margin below FJ2). The
  widened band `[2.30, 2.90)` STILL contains ZERO train false joins for either degree, so
  applying the loosened `IMAGE_MATCHED_CALIBER_MAX_MIDGAP` (2.8) there stays
  train-precision-safe. This is a PURE LOOSENING (2.8 >= every cap the degree/gap ladder
  would otherwise select in the band: shaft 1.85, tip 1.5), so it only ADDS acceptances
  and removes none — a monotonic recall gain that extends the Gen 28 +6 lever into the
  rest of the proven-rich pocket. Each newly-eligible candidate still must clear the
  strict bright-bridge floor (`bridge_ratio >= 0.90`). NOTHING else changed:
  `IMAGE_MATCHED_CALIBER_MAX_MIDGAP` (2.8), the leading band-override branch in
  `propose_edits`, the shaft branch and its constants (1.85 / 2.8 / 2.18), the tip
  three-tier ladder and its constants (2.0 / 2.2 / 2.5 / 1.85 / 1.5), the cos band
  `[0.0, 0.50)` and the 0.0 floor, `IMAGE_MATCHED_MIN_ARM_COS` (0.0), the 0.90 bridge
  floor, and Paths 1/1c/2/3 are byte-for-byte UNCHANGED.
