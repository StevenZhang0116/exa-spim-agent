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

## Current criteria (distance-graded colinear split-repair)

A **distance-graded colinear split-repair** policy. For each SplitSite it emits a
`merge_labels` edit only when the gap is small — `s.gap_um <= GAP_THRESHOLD_UM`
(6.5 µm) — AND a colinearity gate passes whose strictness DEPENDS on the gap:
  1. **MICRO regime** — `s.gap_um <= MICRO_GAP_UM` (2.0 µm): distance ALONE is
     decisive, so the colinearity gate is **WAIVED entirely** (not merely loosened).
     Accept the merge on gap alone. Justification: inter-neuron gaps essentially
     never fall below ~7 µm, so a sub-2 µm gap is overwhelmingly one broken neuron;
     at these lattice-scale gaps (√1..√4 voxels = 1.0, 1.41, 1.73, 2.0 µm) the
     tip-tangent angle is dominated by skeletonization jitter and is uninformative.
     2.0 µm sits 0.2 µm below the nearest reachable FALSE join (2.20 µm), so every
     reachable false join is out of band.
  2. **TIP-TO-TIP regime** — `MICRO_GAP_UM < s.gap_um <= TIP_TIP_GAP_UM` (2.3 µm)
     AND BOTH fragment endpoints are tips (`deg_a == 1 AND deg_b == 1`): distance +
     topology are decisive, so the colinearity gate is **WAIVED entirely** (same
     lattice-jitter rationale as MICRO — the dense missed-real cluster sits at the
     √5 = 2.24 µm voxel-lattice diagonal where the tip-tangent angle is
     uninformative). Two genuine TERMINI meeting across a sub-2.3 µm gap is the
     canonical "one neuron broken into two fragments" signature; joining INTO a
     shaft/branch is the error-prone topology, so it is excluded. `deg_a`/`deg_b`
     are computed as the number of graph neighbors of `node_a`/`node_b` in the
     fragments graph (`len(list(g.neighbors(node)))`). 2.3 µm sits 0.15 µm BELOW
     the nearest TIP-TO-TIP false join (2.45 µm), so BOTH tip-to-tip false joins
     (gap 2.45, 3.00 µm) stay out of band; the false join at 2.20 µm is
     tip-into-shaft (`deg_b == 2`) and is excluded by the tip-to-tip requirement,
     falling through to the FAR floor (cos 0.44 < 0.60 → rejected).
  3. **EXTENDED TIP-TO-TIP regime (caliber-gated)** — `TIP_TIP_GAP_UM (2.3) <
     s.gap_um <= TIP_TIP_EXT_GAP_UM (2.5)` AND BOTH endpoints are tips
     (`deg_a == 1 AND deg_b == 1`) AND the endpoint caliber is matched
     (`_rad_ratio(g, ctx, s) <= TIP_TIP_RAD_MATCH`, 1.50): distance + topology +
     caliber are decisive, so the colinearity gate is **WAIVED entirely** (same
     lattice-jitter rationale as the MICRO / TIP-TO-TIP tiers). This recovers the
     dense cluster of MISSED tip-to-tip reals piled up at gap 2.45 (scattered
     `colinear_cos` ≈ −0.81..+0.53, so the colinearity tier rejects them; many also
     have `rad_ratio > 1.30` so the general caliber OR-clause misses them too).
     Distance alone could NOT extend the prior tier here (the tip-to-tip FALSE @2.45
     sits in this band), but CALIBER separates: the captured tip-to-tip reals have
     `rad_ratio <= 1.46` while the only in-band tip-to-tip FALSE has `rad_ratio
     1.56`, so the 1.50 ceiling sits 0.06 BELOW the FALSE and 0.04 ABOVE the reals.
     The ceiling is LOOSER than the general colinearity-tier caliber path's 1.30
     precisely because `deg_a == deg_b == 1` already excludes the high-caliber
     `deg_b == 2` FALSE @2.20 — a topology-specific relaxation. The FALSE @2.45
     (rad 1.56 > 1.50) and FALSE @3.00 (gap > 2.5) both stay out of band → zero new
     train-false merges. Placed immediately AFTER the TIP-TO-TIP tier and BEFORE the
     colinearity tier; purely ADDITIVE (a site it accepts would otherwise fall to
     the colinearity tier), so train-correct cannot drop below the parent's 256.
  4. **TIP-INTO-SHAFT regime (caliber-gated)** — `MICRO_GAP_UM (2.0) < s.gap_um <=
     TIP_SHAFT_GAP_UM (2.75)` AND `deg_a == 1 AND deg_b == 2` (a fragment tip joining
     INTO a shaft node) AND the endpoint caliber is matched (`_rad_ratio(g, ctx, s)
     <= TIP_SHAFT_RAD_MATCH`, 1.50), colinearity WAIVED: distance + topology +
     caliber are decisive. The gap ceiling is a DEDICATED constant
     `TIP_SHAFT_GAP_UM` (2.75), DECOUPLED from the colinearity-tier's
     `RAD_MATCH_GAP_UM` (2.5) which is left unchanged. This widens the band from 2.5
     to 2.75 to capture the dense matched-caliber missed-real cluster at gap
     2.5–2.72; it is FALSE-free because the only reachable `deg_b == 2` FALSE is
     @2.20 (caliber-excluded, rad 1.97 > 1.50) and the next reachable FALSE @3.00 is
     `deg_b == 1` (never seen by this `deg_b == 2` tier), so there is a FALSE-free
     gap window from ~2.2 up to 3.0; 2.75 stays conservatively below 3.0. So
     the colinearity gate is **WAIVED entirely**. The missed-real bucket is now
     dominated by tip-into-shaft sites at gap 2.0–2.65 (`deg_a == 1, deg_b == 2`)
     that fall through EVERY existing path: gap `> TIP_TIP_GAP_UM` (so both tip-to-tip
     tiers skip them on the `deg_b == 1` requirement), low/negative `colinear_cos`
     (so the FAR 0.60 / averaged 0.55 floors reject them), and `rad_ratio` often just
     ABOVE the colinearity-tier caliber cap of 1.30 (so that OR-clause misses them
     too). Colinearity is actively MISLEADING here: the ONLY reachable `deg_b == 2`
     FALSE join (gap 2.20) has `colinear_cos 0.44`, which is HIGHER than roughly half
     the missed reals in this band — so no colinearity floor can recover these reals
     without also admitting it. CALIBER is the better discriminator: that FALSE has
     `rad_ratio 1.97` (clearly MISMATCHED caliber), while the matched-caliber missed
     reals to capture have `rad_ratio <= ~1.49`. The 1.50 ceiling sits 0.47 BELOW the
     FALSE @2.20 — a WIDE margin → zero new train-false merges; the other two
     reachable FALSE joins are `deg_b == 1`, so this `deg_b == 2` tier never sees
     them. Restricted to `deg_b == 2` ONLY (shaft); `deg_b >= 3` branch points are
     rarer/riskier and the missed cluster is all `deg_b == 2`, so they are excluded.
     The lattice-jitter rationale for waiving colinearity is topology-independent
     (tip-arm jitter corrupts the tip direction regardless of `node_b`'s degree),
     exactly as in the MICRO / TIP-TO-TIP tiers. Relative caliber only (no absolute
     radius), so brain-agnostic. Placed immediately AFTER the EXTENDED TIP-TO-TIP
     tier and BEFORE the colinearity tier; PURELY ADDITIVE (a site it accepts
     currently fails all existing paths), so train-correct cannot drop below the
     parent's 267 — it can only ADD net-new correct repairs.
  4b. **HIGH-rad-ratio TIP-INTO-SHAFT regime (caliber FLOOR)** — `MICRO_GAP_UM (2.0)
     < s.gap_um <= TIP_SHAFT_GAP_UM (2.75)` AND `deg_a == 1 AND deg_b == 2` AND the
     endpoint caliber is HIGHLY MISMATCHED (`_rad_ratio(g, ctx, s) >=
     TIP_SHAFT_HIGH_RAD_RATIO`, 2.0), colinearity WAIVED. This is the INVERSE of tier
     4 — a rad_ratio FLOOR rather than a ceiling. `deg_b == 2` means a tip joining the
     MIDDLE of a cable = a branch/T-junction; a real T-junction is a THIN daughter
     neurite emanating from a THICK parent shaft, so it is naturally VERY mismatched in
     caliber (high rad_ratio). Tier 4 captures the MATCHED-caliber tip-into-shaft reals
     (rad_ratio ≤ 1.50); this tier 4b captures the high-caliber-MISMATCH tip-into-shaft
     reals (rad_ratio ≥ 2.0, ranging up to ~2.75 — a thin tip ~0.75–1.06 reconnecting
     to a thick shaft ~1.5–2.0). The gap prior does the heavy lifting (at gap ≤ 2.75 ≪
     7 µm the site is overwhelmingly a REAL break by population statistics); the
     rad_ratio ≥ 2.0 FLOOR is the cheap precision guard that keeps the 3 known FALSE
     joins out. FALSE-free: all 3 reachable FALSE joins have rad_ratio < 2.0 (1.97,
     1.56, 1.67); the only `deg_b == 2` FALSE (@gap 2.20) is at 1.97, so a 2.0 floor
     leaves a 0.03 margin above it, AND the `(1.50, 2.00)` caliber band is deliberately
     left as a no-go zone (between tier 4's ceiling and this floor) protecting exactly
     that FALSE; the other two reachable FALSE are `deg_b == 1`, so this `deg_b == 2`
     tier never sees them → zero new train-false merges. Relative caliber only (no
     absolute radius), so brain-agnostic. Placed immediately AFTER tier 4 and BEFORE the
     colinearity tier; PURELY ADDITIVE (a site it accepts currently fails all existing
     paths), so train-correct cannot drop below the parent's 286 — it can only ADD
     net-new correct repairs (the report shows dozens of qualifying missed reals).
  4d. **MID-CALIBER TIP-INTO-SHAFT regime (gap-floored band re-fill)** —
     `DEG2_MID_GAP_FLOOR (2.40) < s.gap_um <= TIP_SHAFT_GAP_UM (2.75)` AND
     `deg_a == 1 AND deg_b == 2` AND the endpoint caliber is in the MIDDLE band
     (`TIP_SHAFT_RAD_MATCH < _rad_ratio(g, ctx, s) < TIP_SHAFT_HIGH_RAD_RATIO`, i.e.
     `1.50 < rad_ratio < 2.0`), colinearity WAIVED. This RE-FILLS the middle caliber
     band that tier 4 (ceiling, rad ≤ 1.50) and tier 4b (floor, rad ≥ 2.0)
     deliberately left as a NO-GO zone. That band was carved out for ONE reason: the
     FALSE join @gap 2.20 has rad_ratio 1.97, which sits inside it — and it is the
     ONLY reachable `deg_b == 2` FALSE join AT ANY GAP (the other two reachable FALSE
     are `deg_b == 1`, never seen by a `deg_b == 2` tier). So the band can be safely
     re-opened by gating on gap ABOVE that single FALSE: a new constant
     `DEG2_MID_GAP_FLOOR = 2.40` sits 0.20 µm above the FALSE @2.20, so no reachable
     `deg_b == 2` FALSE can enter → ZERO new train-false merges. The floor
     intentionally sacrifices a few mid-caliber reals at gap 2.21–2.35 (too close to
     the FALSE @2.20 to separate safely) for a wide, generalizable margin. The
     dominant remaining missed cluster is exactly these mid-caliber `deg_b == 2`
     reals, densely populated at gaps 2.49–2.73 (e.g. (gap, rad_ratio): (2.49, 1.86),
     (2.50, 1.90), (2.57, 1.82), (2.70, 1.94), (2.73, 1.76)). The gap prior does the
     heavy lifting (gap 2.40–2.75 ≪ 7 µm ⇒ overwhelmingly a REAL break, gap-distance
     AUC 0.998; true split gaps ~99% under ~6.5 µm). This is a NEW kind of lever — a
     GAP-floor precision gate that re-fills a previously carved-out caliber band,
     distinct from the rad-ceiling (tier 4) / rad-floor (tiers 4b, 4c) tiers. Placed
     immediately AFTER tier 4b and BEFORE tier 4c; PURELY ADDITIVE (new tier +
     continue) and mutually exclusive by caliber band with the two existing
     `deg_b == 2` tiers (rad ≤ 1.50 and rad ≥ 2.0): a site it accepts currently fails
     all existing paths, so train-correct cannot drop below the parent's 323 — it can
     only ADD net-new correct repairs. Relative caliber only (no absolute radius), so
     brain-agnostic.
  4e. **EXTENDED-GAP TIP-INTO-SHAFT regime (gap-ceiling extension to (2.75, 3.0])** —
     `TIP_SHAFT_GAP_UM (2.75) < s.gap_um <= TIP_SHAFT_EXT_GAP_UM (3.0)` AND
     `deg_a == 1 AND deg_b == 2`, accepting across the FULL rad_ratio range and
     WAIVING colinearity — exactly the UNION of the three existing `deg_b == 2` tiers
     (tier 4 matched-caliber rad ≤ 1.50, tier 4d mid-caliber 1.50 < rad < 2.0 gated
     above gap 2.40, tier 4b high-mismatch rad ≥ 2.0), which together accept
     essentially ALL tip-into-shaft sites at gap ≤ TIP_SHAFT_GAP_UM (2.75). All three
     of those tiers cap at the SAME 2.75 ceiling, leaving the dominant remaining
     missed cluster — `deg_b == 2` reals densely packed at gap 2.76–2.98 across the
     full rad_ratio range (e.g. (gap, rad_ratio): (2.76, 1.23), (2.81, 2.08),
     (2.85, 2.06), (2.90, 2.63), (2.97, 2.75)) — uncaptured ONLY because gap > 2.75.
     This tier extends the SAME topology one step further, to (2.75, 3.0]. FALSE-free
     with a WIDE margin: the FALSE @gap 2.20 is the ONLY reachable `deg_b == 2` FALSE
     join at ANY gap, and the new floor (2.75) sits 0.55 µm above it; there is NO
     `deg_b == 2` FALSE anywhere in (2.75, 3.0]. The other two reachable FALSE joins
     (incl. the one @gap 3.00) are `deg_b == 1`, NEVER seen by this `deg_b == 2` tier —
     which is also why `deg_b == 1` (tip-to-tip) is deliberately NOT extended here.
     The gap prior does the heavy lifting (findings #1, #21): at gap 2.75–3.0 ≪ 7 µm a
     tip-into-shaft site is overwhelmingly a REAL break (gap-distance AUC 0.998,
     F1-optimal 6.84 µm; ~99% of true split gaps under ~6.5 µm), so even held-out a
     `deg_b == 2` FALSE in this window is extraordinarily rare. Placed immediately
     AFTER tier 4d and BEFORE tier 4c; PURELY ADDITIVE (new tier + continue): these
     gap-2.76–2.98 `deg_b == 2` sites currently fail all paths (gap > 2.75 so all three
     `deg_b == 2` tiers skip; scattered/negative colinear_cos fails the colinearity
     tier), so train-correct cannot drop below the parent's 336 — it can only ADD
     net-new correct repairs. Relative caliber only (no absolute radius), so
     brain-agnostic.
  4g. **SECOND EXTENDED-GAP TIP-INTO-SHAFT regime (gap-ceiling extension to
     (3.0, 3.25])** — `TIP_SHAFT_EXT_GAP_UM (3.0) < s.gap_um <= TIP_SHAFT_EXT2_GAP_UM
     (3.25)` AND `deg_a == 1 AND deg_b == 2`, accepting across the FULL rad_ratio
     range and WAIVING colinearity. MIRRORS tier 4e's structure exactly, one further
     step beyond its 3.0 ceiling. Tier 4e (and the three caliber-banded tiers 4 / 4b
     / 4d it unions) all cap at gap 3.0, leaving the dominant remaining missed cluster
     — `deg_b == 2` reals densely packed at gap 3.01–3.12 across the full rad_ratio
     range (e.g. (gap, rad_ratio): (3.01, 1.38), (3.01, 2.54), (3.02, 1.13),
     (3.03, 1.88), (3.07, 1.70), (3.10, 2.69), (3.12, 2.66)) — uncaptured ONLY because
     gap > 3.0. Accept on DISTANCE + TOPOLOGY (caliber and colinearity WAIVED).
     FALSE-free: the FALSE @gap 2.20 is the ONLY reachable `deg_b == 2` FALSE join at
     ANY gap, and the new floor (`TIP_SHAFT_EXT_GAP_UM` = 3.0) sits 0.8 µm above it, so
     the nearest `deg_b == 2` FALSE is 0.8 µm below the floor and there is NO
     `deg_b == 2` FALSE anywhere in (3.0, 3.25]. The other two reachable FALSE joins
     (`deg_b == 1` @gap 2.45 and @gap 3.00) are NEVER seen by this `deg_b == 2` tier —
     so the gap-3.00 FALSE is irrelevant here, which is also why `deg_b == 1` is
     deliberately NOT extended. The gap prior does the heavy lifting (findings #1, #21):
     at gap 3.0–3.25 µm, far below the F1-optimal ~6.84 µm split/non-split gap
     threshold and the ~6.5 µm at which ~99% of true split gaps fall, a tip-into-shaft
     site is overwhelmingly a REAL break (gap-distance AUC 0.998). Adds a NEW tier +
     constant (`TIP_SHAFT_EXT2_GAP_UM` = 3.25) ALONGSIDE tier 4e's `TIP_SHAFT_EXT_GAP_UM`
     — gen16's tier 4e and its 3.0 ceiling are UNTOUCHED, so all currently-accepted
     (2.75, 3.0] sites remain accepted. Placed immediately AFTER tier 4e and BEFORE
     tier 4c; PURELY ADDITIVE (new tier + continue): these gap-3.01–3.12 `deg_b == 2`
     sites currently fall through to the colinearity tier and are mostly rejected
     (scattered colinear_cos), so train-correct cannot drop below the parent's 365 — it
     can only ADD net-new correct repairs. Relative caliber/topology only (no absolute
     radius), so brain-agnostic.
  4c. **HIGH-rad-ratio TIP-TO-TIP regime (caliber FLOOR)** — `MICRO_GAP_UM (2.0) <
     s.gap_um <= TIP_TIP_HIGH_RAD_GAP_UM (2.9)` AND `deg_a == 1 AND deg_b == 1` (two
     free ENDS meeting) AND the endpoint caliber is HIGHLY MISMATCHED
     (`_rad_ratio(g, ctx, s) >= TIP_SHAFT_HIGH_RAD_RATIO`, 2.0, the same floor reused
     from tier 4b), colinearity WAIVED. This is the TIP-TO-TIP analogue of the gen13
     tip-into-shaft caliber FLOOR (tier 4b). The matched-caliber EXTENDED tip-to-tip
     tier (tier 3, rad_ratio ≤ 1.50) captures the symmetric-caliber tip-to-tip reals;
     this tier 4c captures the HIGH-mismatch ones — a thin distal end (~0.75–1.0)
     meeting a thick end (~1.5–2.4), rad_ratio ≥ 2.0 — which the gen14 report shows
     are DENSE in the missed bucket at gaps 2.45–2.83 (e.g. (gap, colinear_cos,
     rad_ratio): (2.45, 0.04, 2.00), (2.45, −0.34, 2.41), (2.83, −0.11, 2.00),
     (2.83, −0.09, 2.00), (2.83, 0.37, 2.12), (2.83, 0.52, 2.00)). These are
     currently REJECTED because the EXTENDED tip-to-tip tier only accepts deg_b == 1
     when rad_ratio ≤ TIP_TIP_RAD_MATCH (1.50), and their colinearity is
     low/scattered so the 0.55/0.60 floors miss them too. The gap prior does the
     heavy lifting (at gap ≤ 2.9 ≪ 7 µm a site is overwhelmingly a REAL break by
     population statistics); the rad_ratio ≥ 2.0 FLOOR is the cheap precision guard.
     FALSE-free with DOUBLE protection: both reachable deg_b == 1 FALSE joins have
     rad_ratio < 2.0 (1.56, 1.67), so the 2.0 floor alone excludes them (margin ≥
     0.33); AND the only deg_b == 1 FALSE near this window (@gap 3.00) is also beyond
     the 2.9 gap ceiling. The deg_b == 2 FALSE (@gap 2.20) is never seen by this
     deg_b == 1 tier. Relative caliber only (no absolute radius), so brain-agnostic.
     Placed immediately AFTER tier 4b and BEFORE the colinearity tier; PURELY
     ADDITIVE (a site it accepts currently fails all existing paths), so train-correct
     cannot drop below the parent's 317 — it can only ADD net-new correct repairs.
  4f. **MID-GAP TIP-TO-TIP regime (distance + topology — caliber AND colinearity
     WAIVED)** — `TIP_TIP_EXT_GAP_UM (2.5) < s.gap_um <= TIP_TIP_HIGH_RAD_GAP_UM
     (2.9)` AND `deg_a == 1 AND deg_b == 1` (two free ENDS meeting), accept on
     distance + topology ALONE across the FULL rad_ratio range, WAIVING both caliber
     and colinearity. This GENERALIZES the HIGH-rad-ratio tip-to-tip tier 4c (which
     required `rad_ratio >= TIP_SHAFT_HIGH_RAD_RATIO = 2.0`) to the entire rad_ratio
     range within the FALSE-free `(2.5, 2.9]` window. It recovers the dominant
     remaining missed cluster: tip-to-tip reals (`deg_a == 1 AND deg_b == 1`) piled at
     gap ≈ 2.83 (= √8 ≈ 2.828, a voxel-lattice diagonal) with `rad_ratio < 2.0`
     (so tier 4c's 2.0 floor skips them) and scattered `colinear_cos < 0.6` (so the
     colinearity tier rejects them) — at least 9 visible in the report's missed-real
     bucket (e.g. (gap, cos, rad): (2.83, 0.39, 1.44), (2.83, −0.20, 1.00),
     (2.83, 0.48, 1.95), (2.83, 0.50, 1.41), (2.83, −0.38, 1.67), (2.83, 0.14, 1.58)).
     Colinearity is WAIVED for the same lattice-jitter rationale as the MICRO /
     TIP-TO-TIP tiers (at sub-3 µm scales the tip-tangent angle is skeletonization
     jitter and is uninformative). FALSE-free: the window `(2.5, 2.9]` contains NO
     reachable `deg_b == 1` FALSE join — the two reachable `deg_b == 1` FALSE sit at
     gap 2.45 (BELOW the 2.5 floor, owned by the EXTENDED tip-to-tip tier 3) and gap
     3.00 (ABOVE the 2.9 ceiling); the floor 2.5 sits 0.05 µm above the 2.45 FALSE and
     the ceiling 2.9 sits 0.10 µm below the 3.00 FALSE. The `deg_b == 2` FALSE @gap
     2.20 is never seen by this `deg_b == 1` tier. The gap prior does the heavy lifting
     (findings #1, #21): at gap 2.5–2.9 ≪ 7 µm a tip-to-tip site is overwhelmingly a
     REAL break (gap-distance AUC 0.998, F1-optimal 6.84 µm; ~99% of true split gaps
     under ~6.5 µm). REUSES existing constants (`TIP_TIP_EXT_GAP_UM`,
     `TIP_TIP_HIGH_RAD_GAP_UM`) — no new constant. Placed immediately AFTER tier 4c
     and BEFORE the colinearity tier; PURELY ADDITIVE (new tier + continue): a site it
     accepts currently fails EVERY existing path (gap > 2.5 so both lower tip-to-tip
     tiers skip on gap; rad < 2.0 so tier 4c skips; cos < 0.6 / scattered so the
     colinearity tier rejects), so train-correct cannot drop below the parent's 356 —
     it can only ADD net-new correct repairs.
  5. **NEAR regime** — `MICRO_GAP_UM < s.gap_um <= NEAR_GAP_UM`: colinearity floor
     `NEAR_COLINEAR_COS` (0.0) — accept everything except joins that double back
     (anti-parallel / obtuse, `cos < 0`). (Currently EMPTY because `MICRO_GAP_UM`
     (2.0) > `NEAR_GAP_UM` (1.5); kept for clarity / future re-tuning.)
  6. **FAR regime** — `MICRO_GAP_UM < s.gap_um <= 6.5 µm` (i.e. `> 2.0 µm`), for
     sites NOT accepted by the tip-to-tip tiers: accept if EITHER of two paths passes
     (an OR — the second is purely additive and can only ADD accepts):
       - **(a) hard single-arm gate** — `_is_colinear_split(g, s, min_cos)` with
         `min_cos = MIN_COLINEAR_COS` (0.60): requires the tip-arm half-cosine
         `cos_a >= 0.60` as a SEPARATE floor (6.0 µm tip-tangent walk) AND the
         partner continuation `>= 0.60`. This is the original gate, kept unchanged.
       - **(b) averaged cross-gap colinearity** — `_avg_cross_gap_colinearity(g, s)
         >= COLINEAR_AVG_ACCEPT` (0.55). This FAITHFULLY reproduces the harness's
         validated `colinear_cos` (candidate.py::_split_site_geom): `gdir =
         unit(xyz[node_b]-xyz[node_a])`; `cos_a = dot(arm-A outward tangent, gdir)`
         using a SHORT 4.0 µm walk (`HARNESS_TANGENT_WALK_UM`, matching the harness,
         NOT the 6.0 µm the hard gate uses); `cos_b = max over node_b's same-segment
         neighbors of dot(gdir, neighbor_dir)`; the measure is the MEAN
         `0.5*(cos_a + cos_b)`. Because it AVERAGES the two half-cosines instead of
         gating `cos_a` separately, it recovers reals whose tip arm bends slightly
         (`cos_a` ~0.4) but whose partner continues straight (`cos_b` ~0.9; avg
         ~0.65) — exactly the missed-real cluster path (a) wrongly rejects.
       - **(c) caliber-matched recall path** — `_rad_ratio(g, ctx, s) <=
         RAD_RATIO_MATCH` (1.30) AND `avg_col >= 0.0` (not anti-parallel) AND
         `s.gap_um <= RAD_MATCH_GAP_UM` (2.5). `_rad_ratio` reads the endpoint
         radii from `ctx["node_radius"]` (falling back to `g.node_radius`) at
         `s.node_a`/`s.node_b` and returns the RELATIVE ratio `max(ra,rb)/min(ra,rb)`
         (None if no radius array or either radius <= 0). A real "one neuron broken
         in two" has nearly-equal endpoint calibers (ratio ~1.0–1.3); a false join
         fuses two unrelated neurites of DIFFERENT caliber. This recovers the dense
         band of reals just above 2.0 µm whose tip-tangent angle is lattice jitter
         (colinear_cos scattered ~ −0.8..+0.5, so paths (a)/(b) reject them) but whose
         two broken ends have MATCHED caliber. It is RELATIVE caliber only — no
         absolute-radius threshold — so it carries no brain-specific assumption.
     The averaged floor 0.55 sits 0.11 ABOVE the largest reachable FALSE-join
     `colinear_cos` (the 3 false joins are at cos 0.44/0.19/0.06), so every reachable
     false join stays rejected with ~0.11 margin. The caliber ceiling 1.30 sits
     ~0.26 BELOW the smallest reachable FALSE-join `rad_ratio` (the 3 false joins are
     at 1.97/1.56/1.67), so no reachable false join can enter path (c) either. All
     three paths are joined by OR, so each is purely additive (no currently-accepted
     site is dropped) and each still recovers clean mid-gap / caliber-matched
     continuations while staying above the false-join population.
MergeSites are left alone (no `split_label`) — splitting is the riskier edit, left
for the loop to add once it can measure the trade-off. This widens recall (the
prior 0.94 gate accepted only 6 of 540 reachable real splits) while still rejecting
the few false joins, which sit at larger gaps with low colinearity. The gate is
parent-relative, so each generation must beat the last accepted policy, not the
no-edit floor.

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
- **Gen 1 → candidate (distance-graded colinearity gate):** Diagnosis was a RECALL
  problem, not precision. The gen01 policy scored 6 correct − 0 false; among
  reachable SplitSites the class balance is 540 REAL : 3 FALSE, yet it accepted only
  6/540 reals. The binding constraint was `MIN_COLINEAR_COS = 0.94`, NOT the gap:
  hundreds of missed real splits sit at tiny gaps (0.18–1.5 µm) but have low/noisy
  local colinearity (cos 0.0–0.7), so the strict 0.94 gate rejected them. The 3
  reachable FALSE joins sit at gap 2.2–3.0 µm with cos 0.06–0.44. Change: raised
  `GAP_THRESHOLD_UM` 4.0→6.5, lowered `MIN_COLINEAR_COS` 0.94→0.85, and added a
  distance-graded near-gap regime (`NEAR_GAP_UM = 1.5`, `NEAR_COLINEAR_COS = 0.0`)
  so at ≤1.5 µm distance decides and only anti-parallel joins are vetoed. This keeps
  all 6 current hits, rejects all 3 reachable FALSE joins (each at gap > 1.5 µm with
  cos < 0.85), and newly captures the small-gap real splits the 0.94 gate missed.
  Image was deliberately NOT used: the warm-start bridge_ratio AUC was only 0.69 and
  the 3 FALSE joins have HIGH bridge_ratio (0.97–0.99) fully overlapping the REAL
  range (0.94–1.02), so gating on it would not exclude them and would risk a false
  merge (auto-revert). AutoDiscovery support: finding #1 (id 30 — gap-distance AUC
  0.998, inter-neuron gaps rarely < 7 µm, F1-optimal ~6.84 µm), finding #21 (id 63 —
  true split gaps 99% under 6.5 µm), finding #16 (id 33 — angular alignment AUC 0.93,
  true continuation ~153° vs false ~90°); all three GENERALIZE + UPHELD/OK. Avoided
  the non-generalizing distance-angle INTERACTION finding (#25, id 47) — instead
  combined the two independently-robust discriminators (gap and angle) separately.
- **Gen 2 → candidate (lower the FAR-regime colinearity floor):** Diagnosis was
  RECALL-bound, not precision: gen02 scored 31 correct − 0 train-false (= 31, up
  from the parent's 6) with the no-new-merge guard satisfied, yet only 31 of 540
  reachable real splits were accepted — 509 still missed. The binding constraint
  was the FAR floor `MIN_COLINEAR_COS = 0.85`: many missed reals sit at gap
  1.5–2.0 µm with clean colinearity just under 0.85 (e.g. (gap,cos) =
  (1.55,0.62), (1.65,0.78), (1.87,0.67), (1.91,0.82), (1.94,0.76)), while accepted
  FAR reals all had cos >= 0.88. Change: lowered `MIN_COLINEAR_COS` 0.85→0.60
  (FAR regime ONLY; NEAR regime, `NEAR_GAP_UM`, `NEAR_COLINEAR_COS` and
  `GAP_THRESHOLD_UM` all UNCHANGED). Safety: all 3 reachable FALSE joins have
  cos <= 0.44 (0.44, 0.19, 0.06), so a 0.60 floor still rejects every reachable
  false join with ~0.16 margin; and high-colinearity mid-gap pairs are the
  lowest-chaining-risk recall gain, so this was the safe lever. Did NOT loosen the
  NEAR regime: the report shows held-out mega-merges from union-find chaining (a
  5-way fusion on N013, a 4-way on N018 costing −0.530 EdgeAcc / +0.677
  %MergedEdges) that originated in the loose short-gap NEAR regime, so requiring
  strong colinearity beyond 1.5 µm limits new chaining. Image deliberately NOT
  used: warm-start bridge_ratio AUC is only 0.69 and the 3 FALSE joins have HIGH
  bridge_ratio (0.97–0.99) overlapping REAL, so gating on it cannot exclude them.
  AutoDiscovery support: primary finding #16 (id 33 — angular alignment AUC 0.93,
  true continuation ~153° vs false ~90°/cos≈0; a 0.60 cos ≈ 53° bend sits
  comfortably above the false population), with finding #1 (id 30 — gap-distance
  AUC 0.998) and finding #21 (id 63 — true split gaps 99% under 6.5 µm) for
  gap-scale context; all GENERALIZE + UPHELD/OK. Considered but did NOT adopt
  finding #17 (split-on-thin-process radius — magnitude brain-specific) or finding
  #15 (split clustering — chaining-risky).
- **Gen 5 → candidate (micro-gap distance-decisive tier — WAIVE colinearity below
  2.0 µm):** Diagnosis was RECALL-bound, not precision: gen05 scored 91 correct − 0
  train-false (= 91), but of 540 reachable REAL splits only 91 were accepted — 449
  MISSED, and all 3 reachable FALSE joins were correctly rejected (0 wrongly
  accepted). The missed reals cluster heavily at MICRO gaps (0.18–~2.0 µm) where
  `colinear_cos` is scattered across −0.76..+0.77 (e.g. gap 1.41/cos −0.54,
  1.00/cos −0.14, 0.18/cos 0.02): the tip-tangent angle at lattice-scale gaps
  (√1..√4 voxels = 1.0, 1.41, 1.73, 2.0 µm) is dominated by skeletonization jitter
  and is essentially uninformative, so the existing colinearity floor
  (`NEAR_COLINEAR_COS = 0.0`) rejected them on a meaningless `cos < 0`. Colinearity
  does NOT separate these missed reals from the 3 FALSE (their cos ranges overlap);
  the clean separator is GAP DISTANCE — every reachable FALSE sits at gap >= 2.20 µm
  (2.20/cos 0.44, 2.45/cos 0.19, 3.00/cos 0.06). Change: added a new
  `MICRO_GAP_UM = 2.0` constant (0.2 µm below the nearest reachable false join) and a
  MICRO fast-accept tier in `propose_edits` — for `s.gap_um <= MICRO_GAP_UM`, accept
  the merge on DISTANCE ALONE and skip `_is_colinear_split` entirely (the FAR
  cos >= 0.60 path is unchanged for gaps above 2.0 µm). This is a genuinely NEW
  mechanism: prior gens (gen1 NEAR regime, gen2 FAR floor) always kept a colinearity
  floor of at least 0.0 — they never WAIVED it. It converts the large micro-gap
  missed-real bucket into correct repairs (lifting train correct above 91) while
  keeping train false == 0, because all 3 reachable false joins are >= 2.20 µm > 2.0
  µm and thus out of band. NOT re-tuning the colinearity floors (gen4 tied 91 doing
  that) and NOT a component-size cap (gen3 lost recall, held-out −2.000). Image
  deliberately NOT used: warm-start bridge_ratio AUC is only 0.69 and the 3 FALSE
  joins (bridge_ratio 0.97–0.99) overlap REAL (0.94–1.02), so it cannot exclude them.
  AutoDiscovery support: finding #1 (id 30 — gap-distance AUC 0.9979, inter-neuron
  gaps RARELY below ~7 µm, so a sub-2 µm gap is overwhelmingly one broken neuron;
  GENERALIZES + OK) and finding #21 (id 63 — true split gaps 99% under 6.5 µm, tight
  characteristic scale; GENERALIZES + OK). Both qualify (GENERALIZES + UPHELD/OK).
- **Gen 6 → candidate (tip-to-tip distance+topology tier — WAIVE colinearity below
  2.3 µm when BOTH endpoints are tips):** Diagnosis was RECALL-bound, NOT precision:
  the parent scored 163 correct − 0 train-false (= 163), but of 540 reachable REAL
  splits only 163 were accepted — 377 MISSED, all 3 reachable FALSE joins correctly
  rejected. The missed reals now pile up JUST ABOVE the 2.0 µm MICRO cutoff, with a
  dominant cluster at gap = 2.24 µm (= √5, a voxel-lattice diagonal) where
  `colinear_cos` is scattered across ≈ −0.88..+0.55 (lattice-scale skeletonization
  jitter — the SAME noise that justified waiving colinearity in the MICRO tier).
  Distance ALONE could not be extended: the three reachable FALSE joins are at gap
  2.20 (deg_b=2, tip-into-SHAFT), 2.45 (deg_b=1, tip-to-tip), 3.00 (deg_b=1,
  tip-to-tip), and the FALSE @ 2.20 sits INSIDE the next missed-real band, so
  simply raising `MICRO_GAP_UM` would accept a false join. But ENDPOINT DEGREE
  separates cleanly: the big missed-real cluster at gap 2.0–2.24 is TIP-TO-TIP
  (deg_b=1), the canonical "one neuron broken into two fragments" signature; the
  FALSE @ 2.20 is tip-INTO-shaft (deg_b=2); the two tip-to-tip FALSE are farther
  out at 2.45 and 3.00 µm. Lever (a genuinely NEW mechanism — endpoint degree, a
  feature no prior gen has touched): added `TIP_TIP_GAP_UM = 2.3` (0.15 µm below
  the nearest tip-to-tip false join at 2.45 µm) and a TIP-TO-TIP fast-accept tier in
  `propose_edits`, placed AFTER the MICRO fast-accept and BEFORE the NEAR/FAR
  `min_cos` logic: compute `deg_a = len(list(g.neighbors(s.node_a)))` and `deg_b =
  len(list(g.neighbors(s.node_b)))`, and if `s.gap_um <= TIP_TIP_GAP_UM AND
  deg_a == 1 AND deg_b == 1`, accept on distance+topology alone (waiving
  colinearity, same lattice-jitter rationale). This converts the dense gap-2.0–2.24
  tip-to-tip missed-real bucket into correct repairs (lifting train correct above
  163) while keeping train false == 0: the FALSE @ 2.20 is deg_b=2 (excluded → falls
  to FAR floor cos 0.44 < 0.60 → rejected); the FALSE @ 2.45 and @ 3.00 are
  tip-to-tip but at gap > 2.3 (out of band → still rejected). NOT raising
  `MICRO_GAP_UM` (would accept the FALSE @ 2.20) and NOT a colinearity re-tune
  (gen4 tied 91 doing that) — the lever is the new endpoint-degree gate. Image
  deliberately NOT used: warm-start bridge_ratio AUC is only 0.69 and the 3 FALSE
  joins (bridge_ratio 0.97–0.99) overlap REAL (0.94–1.02), so it cannot exclude
  them. AutoDiscovery support: finding #1 (id 30 — gap-distance AUC 0.9979,
  inter-neuron gaps RARELY below ~7 µm, so a sub-2.3 µm gap is overwhelmingly one
  broken neuron; GENERALIZES + OK), with finding #21 (id 63 — true split gaps 99%
  under 6.5 µm, tight characteristic scale; GENERALIZES + OK) for gap-scale context.
  The topology half (tip-to-tip = clean broken-neuron signature; joining into a
  shaft/branch is the error-prone topology) is grounded on the report's own
  SplitSite audit (the deg_b=1 cluster is the missed-real population; the only
  in-band false join is deg_b=2 tip-into-shaft) rather than on any non-generalizing
  KB finding.
- **Gen 7 → candidate (additive averaged cross-gap colinearity accept path):**
  Diagnosis was RECALL-bound, NOT precision: the parent scored 198 correct − 0
  train-false (= 198), but of 540 reachable REAL splits only 198 were accepted —
  342 MISSED, all 3 reachable FALSE joins correctly rejected (precision perfect).
  The FAR-regime gate `_is_colinear_split` is MISMATCHED with the harness's
  validated `colinear_cos` (candidate.py::_split_site_geom) in two ways: (a) it
  REQUIRES the tip-arm half-cosine `cos_a >= 0.60` as a SEPARATE hard floor, whereas
  the harness uses the AVERAGE `0.5*(cos_a + cos_b)`, so a real split whose tip arm
  bends slightly (`cos_a` ~0.4) but whose partner continues straight (`cos_b` ~0.9;
  avg ~0.65, clearly colinear) is REJECTED; (b) it walks 6.0 µm vs the harness's
  4.0 µm, over-smoothing the tangent. Evidence in the SplitSite audit: many MISSED
  reals have reported `colinear_cos` >= 0.55 yet were rejected (e.g. gap/cos
  2.13/0.73, 2.05/0.56, 2.22/0.55, 2.03/0.52), while the 3 FALSE joins have
  `colinear_cos` = 0.44/0.19/0.06 — well below — so the AVERAGED measure separates
  REAL from FALSE cleanly. Lever (a genuinely NEW mechanism — a faithful averaged
  measure, not a threshold re-tune): added `_avg_cross_gap_colinearity(g, s)` that
  reproduces the harness measure exactly (gdir = unit(xyz[node_b]-xyz[node_a]);
  cos_a from a 4.0 µm `_walk_tangent`; cos_b = best same-segment neighbor dot at
  node_b; return `0.5*(cos_a + cos_b)`), a constant `COLINEAR_AVG_ACCEPT = 0.55`
  (0.11 above the largest reachable false-join cos of 0.44), and a `HARNESS_TANGENT_
  WALK_UM = 4.0` constant. In `propose_edits` the FAR colinearity acceptance became
  an OR: accept if `_is_colinear_split(g, s, min_cos)` OR
  `(_avg_cross_gap_colinearity(g, s) >= COLINEAR_AVG_ACCEPT)`. Because the new clause
  is purely ADDITIVE (OR), no currently-accepted repair is lost (correct cannot drop
  below 198); it ADDS the missed reals whose averaged cross-gap colinearity >= 0.55,
  lifting train correct ABOVE 198; and the 3 FALSE joins (cos <= 0.44 < 0.55) stay
  rejected, so train false stays 0. Did NOT modify `_is_colinear_split`, MICRO/
  TIP-TO-TIP tiers, or any of MIN_COLINEAR_COS / NEAR_COLINEAR_COS / MICRO_GAP_UM /
  TIP_TIP_GAP_UM. AVOIDED: lowering the cos floor (gen2 already did that and gen4
  tied 91 re-tuning floors — the lever here is a DIFFERENT averaged measure, not a
  lower threshold) and a component-size cap (gen3, reverted). Image deliberately NOT
  used: the warm-start `bridge_ratio` AUC is only 0.69 and the FALSE joins
  (bridge_ratio 0.97–0.99) overlap the REAL range (0.94–1.02), so it cannot exclude
  them and would risk an auto-reverting false merge. AutoDiscovery support: primary
  finding #16 (id 33 — endpoint angular alignment is the validated discriminator,
  AUC 0.93: true continuation aligned vs false ~right-angle; the averaged two-half-
  cosine measure IS that discriminator computed faithfully, so gating on it is the
  correct implementation), with finding #1 (id 30 — gap-distance AUC 0.998, true
  split gaps tightly bounded) for band context; both GENERALIZE + UPHELD/OK.
- **Gen 8 → candidate (additive caliber-matched recall path):** Diagnosis was that
  the parent (248 correct − 0 train-false, 292 missed reals) OVER-MERGED on
  held-out: Edge Accuracy collapsed 72.97 → 66.17 (−6.8), %Merged Edges jumped
  24.34 → 31.56 (+7.2), and union-find transitive closure fused components of 20–33
  raw labels into brain-spanning mega-labels (e.g. N016 Edge Accuracy −14.78, N013
  many 24-label fusions). The remaining 292 missed reals now OVERLAP the 3 reachable
  FALSE joins in BOTH gap and colinear_cos, so those two features are EXHAUSTED as
  separators — no colinearity re-tune can lift recall without admitting a false
  join. The UNUSED clean separator is CABLE CALIBER: the two endpoint radii of a
  real "one neuron broken in two" are nearly equal (rad_ratio ≈ 1.0–1.3), whereas
  all 3 reachable FALSE joins fuse different calibers (rad_ratio 1.97/1.56/1.67, all
  ≥ 1.56). No prior generation has used caliber. Lever (a genuinely NEW feature —
  endpoint radius ratio): added `RAD_RATIO_MATCH = 1.30` and `RAD_MATCH_GAP_UM =
  2.5` constants, a `_rad_ratio(g, ctx, s)` helper (reads `ctx["node_radius"]`,
  falls back to `g.node_radius`, guards None/zero/negative, returns
  `max(ra,rb)/min(ra,rb)`), and a THIRD OR clause in the FAR colinearity tier:
  accept also when `rr is not None and rr <= RAD_RATIO_MATCH and avg_col is not None
  and avg_col >= 0.0 and s.gap_um <= RAD_MATCH_GAP_UM` (reusing the
  already-computed `avg_col`; the `>= 0.0` guard rejects anti-parallel doubling-back
  joins). The clause is PURELY ADDITIVE (OR), so it cannot drop any of the parent's
  248 accepts and adds the dense caliber-matched missed-real band just above 2.0 µm
  (lifting correct above 248); the 1.30 ceiling sits ~0.26 BELOW the smallest FALSE
  rad_ratio (1.56), so all 3 reachable false joins stay excluded with margin →
  zero new train-false merges. KEPT MICRO, TIP-TO-TIP, the two existing colinearity
  OR clauses, `_is_colinear_split`, and `_avg_cross_gap_colinearity` intact. Did NOT
  re-add a union-find component-size cap (a prior gen's MAX_MERGE_COMPONENT was
  rejected, held-out −2.000), did NOT re-tune the colinearity floor as the lever (a
  prior gen tied and was reverted), and did NOT raise MICRO_GAP_UM (would catch the
  FALSE @ 2.20 µm). Image deliberately NOT used: the warm-start bridge_ratio AUC is
  only 0.69 and the 3 FALSE joins (bridge_ratio 0.97–0.99) overlap REAL (0.94–1.02),
  so image cannot separate the reachable false joins. AutoDiscovery support: finding
  #1 (id 30 — gap-distance AUC 0.998, inter-neuron gaps rarely below ~7 µm, so a
  sub-2.5 µm gap is overwhelmingly one broken neuron; GENERALIZES + UPHELD/OK) plus
  the failure report's cable-caliber semantics (matched endpoint caliber = one
  broken neuron; mismatched = two fused neurites). DISTRUSTS the brain-specific
  absolute-radius / split-on-thin-process finding (#17, DOES-NOT-GENERALIZE) — this
  path uses only the RELATIVE rad_ratio of the two endpoints, never any absolute
  radius threshold.
- **Gen 9 → candidate (caliber-gated EXTENDED tip-to-tip gap-extension tier):**
  Diagnosis was RECALL-bound, NOT precision: the parent scored 256 correct − 0
  train-false (= 256), but of 540 reachable REAL splits only 256 were accepted —
  284 MISSED, all 3 reachable FALSE joins correctly rejected. The clean unused
  recall opportunity is a DENSE CLUSTER of MISSED reals at gap 2.45 that are
  TIP-TO-TIP (deg_a == 1 AND deg_b == 1): e.g. (cos, rad_ratio) =
  (−0.07, 1.18), (−0.00, 1.34), (−0.13, 1.41), (0.30, 1.34), (−0.03, 1.00),
  (0.52, 1.46), (−0.15, 1.41), (−0.35, 1.00), (−0.26, 1.46), (−0.48, 1.41). These
  sit JUST ABOVE the TIP_TIP_GAP_UM = 2.3 cutoff, so they fall through to the
  colinearity tier and are rejected (near-zero/negative `colinear_cos`, and many
  have `rad_ratio > 1.30` so the general caliber OR-clause misses them too). The
  ACCEPTED tip-to-tip reals at gap 1.41/1.73 already PROVE colinearity is pure
  lattice jitter for tip-to-tip topology — many were accepted with strongly
  NEGATIVE `colinear_cos` (−0.73, −0.76, −0.68, −0.54) — so the recovery tier must
  WAIVE colinearity, exactly as the existing TIP-TO-TIP tier does. Distance alone
  could NOT extend the prior tier: the only tip-to-tip FALSE inside the (2.3, 2.5]
  band is gap 2.45 / rad_ratio 1.56 / deg_b == 1 (the 3 reachable FALSE are
  constant across gens: gap 2.20 rad 1.97 deg_b==2; gap 2.45 rad 1.56 deg_b==1;
  gap 3.00 rad 1.67 deg_b==1). Lever (a NEW tier — a caliber-gated tip-to-tip gap
  extension): added `TIP_TIP_EXT_GAP_UM = 2.5` and `TIP_TIP_RAD_MATCH = 1.50`, and
  a tier placed immediately AFTER the existing TIP-TO-TIP tier and BEFORE the
  colinearity tier (reusing the already-computed `deg_a`/`deg_b` and the existing
  `_rad_ratio` helper): if `TIP_TIP_GAP_UM < s.gap_um <= TIP_TIP_EXT_GAP_UM AND
  deg_a == 1 AND deg_b == 1 AND rr_tt <= TIP_TIP_RAD_MATCH`, accept on
  distance + topology + matched caliber, WAIVING colinearity. The caliber ceiling
  1.50 sits 0.06 BELOW the in-band tip-to-tip FALSE rad_ratio 1.56 and 0.04 ABOVE
  the captured reals' 1.46 — and it can be LOOSER than the general path's 1.30
  precisely BECAUSE the `deg_a == deg_b == 1` requirement already removes the
  high-caliber `deg_b == 2` FALSE @2.20. So all 3 reachable FALSE stay excluded:
  @2.20 (deg_b == 2), @2.45 (rad 1.56 > 1.50), @3.00 (gap > 2.5) → zero new
  train-false merges. The tier is PURELY ADDITIVE (a site it accepts would
  otherwise fall to the colinearity tier), so train-correct cannot drop below the
  parent's 256, and it recovers ~10 tip-to-tip reals → score > 256. KEPT MICRO,
  the existing TIP-TO-TIP tier, the full colinearity tier (all three OR clauses),
  `_is_colinear_split`, `_avg_cross_gap_colinearity`, and `_rad_ratio` intact. Did
  NOT widen `RAD_RATIO_MATCH` (1.30) or `RAD_MATCH_GAP_UM` in the colinearity tier,
  did NOT re-add a union-find component-size cap (a prior gen's MAX_MERGE_COMPONENT
  was rejected, held-out −2.000), did NOT re-tune the colinearity floor as the
  lever (a prior gen tied 256 and was reverted), and did NOT raise MICRO_GAP_UM or
  the existing TIP_TIP_GAP_UM (raising TIP_TIP_GAP_UM ungated would admit the 2.45
  FALSE). Image deliberately NOT used: the warm-start probe AUC is only 0.69 and
  the 3 FALSE bridge_ratios (0.97–0.99) OVERLAP the REAL range (0.94–1.02), so
  image cannot separate the reachable false joins in this band. AutoDiscovery
  support: finding #1 (id 30 — gap-distance AUC 0.998, inter-neuron gaps rarely
  below ~7 µm, so a sub-2.5 µm gap is overwhelmingly one broken neuron;
  GENERALIZES + OK), paired with the failure report's own tip-to-tip topology
  evidence (two genuine termini at a tiny gap = one neuron broken in two) and
  cable-caliber semantics (matched endpoint caliber = one broken neuron; mismatched
  = two fused neurites). DISTRUSTS the brain-specific absolute-radius /
  split-on-thin-process finding (#17, DOES-NOT-GENERALIZE) — this tier uses only
  the RELATIVE rad_ratio of the two endpoints, never any absolute radius threshold.
- **Gen 10 → candidate (caliber-gated TIP-INTO-SHAFT tier):** Diagnosis was
  RECALL-bound, NOT precision: the parent scored 267 correct − 0 train-false
  (= 267), but of 540 reachable REAL splits only 267 were accepted — 273 MISSED, all
  3 reachable FALSE joins correctly rejected (precision perfect, no over-merges). The
  MISSED bucket is now DOMINATED by TIP-INTO-SHAFT sites: `deg_a == 1 AND deg_b == 2`
  at gap 2.0–2.65 (e.g. `2.01 / cos −0.01 / rad 1.43`, `2.22 / 0.16 / 1.47`,
  `2.25 / −0.18 / 1.43`, `2.35 / 0.24 / 1.31`, `2.35 / −0.05 / 1.09`,
  `2.39 / 0.12 / 1.33`, `2.41 / 0.08 / 1.39`, `2.47 / −0.02 / 1.49`). They fall
  through EVERY existing path: gap `> TIP_TIP_GAP_UM` (2.3) so both tip-to-tip tiers
  skip them on the `deg_b == 1` requirement; low/negative `colinear_cos` fails the
  FAR 0.60 floor and the averaged 0.55 floor; and many have `rad_ratio` just ABOVE
  the colinearity-tier caliber cap of 1.30, so that OR-clause misses them too. KEY
  INSIGHT: in this `deg_b == 2` band colinearity is actively MISLEADING — the ONLY
  reachable `deg_b == 2` FALSE join (gap 2.20) has `colinear_cos 0.44`, HIGHER than
  roughly half the missed reals, so no colinearity floor can recover the reals
  without admitting the false join (exactly why the 0.55/0.60 gates cannot reach
  these). But that FALSE has clearly MISMATCHED caliber `rad_ratio 1.97`, whereas the
  matched-caliber missed reals have `rad_ratio <= ~1.49`. So CALIBER beats
  colinearity here. The lattice-jitter argument that already justified WAIVING
  colinearity in the MICRO and TIP-TO-TIP tiers is topology-independent (tip-arm
  jitter corrupts the tip direction regardless of what `node_b` is), so it applies to
  tip-into-shaft too. Lever (a NEW caliber-gated tier): added
  `TIP_SHAFT_RAD_MATCH = 1.50` (reusing `RAD_MATCH_GAP_UM` 2.5 as the gap ceiling),
  and a tier placed immediately AFTER the EXTENDED TIP-TO-TIP tier and BEFORE the
  colinearity tier (reusing the already-computed `deg_a`/`deg_b` and the existing
  `_rad_ratio` helper): if `MICRO_GAP_UM < s.gap_um <= RAD_MATCH_GAP_UM AND
  deg_a == 1 AND deg_b == 2 AND rr_ts <= TIP_SHAFT_RAD_MATCH`, accept on
  distance + topology + matched caliber, WAIVING colinearity. The 1.50 ceiling sits
  0.47 BELOW the only reachable `deg_b == 2` FALSE's `rad_ratio` 1.97 — a WIDE margin
  — while admitting the matched-caliber missed reals at `rad_ratio <= ~1.49`. The
  other two reachable FALSE joins are `deg_b == 1` (so this `deg_b == 2` tier never
  sees them) → zero new train-false merges. Restricted to `deg_b == 2` ONLY (shaft):
  `deg_b >= 3` branch points are rarer/riskier and the missed cluster is all
  `deg_b == 2`. The tier is PURELY ADDITIVE — a site it accepts currently fails all
  existing paths (gap > 2.3, deg_b == 2, low/negative cos, rad_ratio > 1.30) — so
  train-correct cannot drop below the parent's 267; it recovers ~8–12 matched-caliber
  tip-into-shaft reals → score > 267. KEPT MICRO, both tip-to-tip tiers, the full
  colinearity tier (all three OR clauses), `_is_colinear_split`,
  `_avg_cross_gap_colinearity`, and `_rad_ratio` intact. Did NOT widen
  `RAD_RATIO_MATCH` (1.30) or `RAD_MATCH_GAP_UM` in the colinearity tier, did NOT
  widen the tip-to-tip constants, did NOT raise `MICRO_GAP_UM` / `TIP_TIP_GAP_UM` /
  `TIP_TIP_EXT_GAP_UM`, did NOT re-add a union-find component-size cap (a prior gen's
  MAX_MERGE_COMPONENT was rejected, held-out −2.000), and did NOT re-tune the
  colinearity floor as the lever (a prior gen tied and was reverted). Image
  deliberately NOT used: the warm-start probe AUC is only 0.69 and the 3 reachable
  FALSE bridge_ratios (0.97–0.99) OVERLAP the REAL range (0.94–1.02), so image cannot
  separate the reachable false joins in this band. AutoDiscovery support: finding #1
  (id 30 — gap-distance AUC 0.998, inter-neuron gaps rarely below ~7 µm, so a
  sub-2.5 µm gap is overwhelmingly one broken neuron; GENERALIZES + OK), paired with
  the failure report's own cable-caliber semantics (matched endpoint caliber = one
  neuron broken; mismatched = two fused neurites) and the observation that the
  `deg_b == 2` FALSE @2.20 has misleadingly HIGH `colinear_cos` 0.44 but mismatched
  caliber 1.97. DISTRUSTS the brain-specific absolute-radius / split-on-thin-process
  finding (#17, DOES-NOT-GENERALIZE) — this tier uses only the RELATIVE rad_ratio of
  the two endpoints, never any absolute radius threshold.
- **Gen 11 → candidate (extend the TIP-INTO-SHAFT gap ceiling 2.5 → 2.75, decoupled
  from RAD_MATCH_GAP_UM):** Diagnosis was RECALL-bound, NOT precision: the parent
  scored 275 correct − 0 train-false (= 275), but of 540 reachable REAL splits only
  275 were accepted — 265 MISSED, all 3 reachable FALSE joins correctly rejected
  (precision perfect, no over-merges). The MISSED bucket splits into two groups.
  GROUP (1), the recoverable one: CALIBER-MATCHED tip-into-shaft sites (`deg_a == 1,
  deg_b == 2, rad_ratio <= 1.50`) at gaps JUST ABOVE the tier's 2.5 ceiling — e.g.
  `2.51 / rad 1.19`, `2.55 / 1.44`, `2.56 / 1.47`, `2.60 / 1.13`, `2.64 / 1.17`,
  `2.67 / 1.11`, `2.71 / 1.13`, `2.72 / 1.33`, `2.72 / 1.25`. These are EXACTLY what
  the gen10 TIP-INTO-SHAFT tier already accepts, except they fall just outside its
  gap ceiling (`RAD_MATCH_GAP_UM` = 2.5, which the tier was sharing with the
  colinearity-tier caliber clause). GROUP (2), the un-recoverable one: caliber-
  MISMATCHED high-rad `deg_b == 2` reals (rad_ratio ~1.9–2.7, thin tip onto thick
  shaft) that OVERLAP the only reachable `deg_b == 2` FALSE @2.20 (rad 1.97,
  colinear_cos 0.44) in BOTH caliber and colinearity, and the image probe cannot
  separate them either (the FALSE bridge_ratio 0.97 lies inside the REAL range
  0.94–1.02, probe AUC only 0.69) — so chasing group (2) cannot be done without
  admitting the false join. The lever therefore targets ONLY group (1). SAFETY (the
  FALSE-free-gap-window argument): the only reachable `deg_b == 2` FALSE is @2.20
  and it is excluded by the caliber ceiling (rad 1.97 > 1.50); the next reachable
  FALSE is @3.00 but it is `deg_b == 1`, so a `deg_b == 2` tier NEVER sees it. Hence
  caliber-matched tip-into-shaft is FALSE-free across the whole gap window from ~2.2
  up to 3.0, and the tier's gap ceiling can be pushed past 2.5 without admitting any
  reachable false join. Change (decouple, then extend): added a DEDICATED constant
  `TIP_SHAFT_GAP_UM = 2.75` and, in the TIP-INTO-SHAFT tier ONLY, changed the gap
  upper bound from `RAD_MATCH_GAP_UM` to `TIP_SHAFT_GAP_UM`. This is PURELY ADDITIVE
  — it only WIDENS the gap band from 2.5 to 2.75, so every site the tier already
  accepted still is, and it newly admits ~9+ caliber-matched tip-into-shaft reals in
  (2.5, 2.75]; train-correct cannot drop below the parent's 275, it can only rise.
  No reachable FALSE is in this `deg_b == 2` band (the 2.20 FALSE is rad 1.97 > 1.50;
  the 2.45 and 3.00 FALSE are `deg_b == 1`) → zero new train-false merges → score >
  275. Held conservatively below 3.0 (the `deg_b == 1` FALSE sits at 3.00 and the
  held-out population near it is unseen). KEPT everything else byte-for-byte: did NOT
  touch `RAD_MATCH_GAP_UM` (2.5) in the colinearity-tier caliber clause, did NOT
  touch the tip-to-tip tiers/constants, `TIP_SHAFT_RAD_MATCH` (1.50), the helpers,
  MICRO, or the colinearity tier. Did NOT extend to/past 3.0, did NOT widen
  `TIP_SHAFT_RAD_MATCH` (group (2) high-rad reals are unseparable from the FALSE
  @2.20 — raising the caliber ceiling would admit it), did NOT re-add a union-find
  component-size cap (a prior gen's MAX_MERGE_COMPONENT was rejected, held-out
  −2.000), and did NOT re-tune the colinearity floor (a prior gen tied and was
  reverted). Image deliberately NOT used: the warm-start probe AUC is only 0.69 and
  the 3 reachable FALSE bridge_ratios (0.97–0.99) OVERLAP the REAL range (0.94–1.02),
  so image cannot separate the reachable false joins in this band. AutoDiscovery
  support: finding #1 (id 30 — gap-distance AUC 0.9979, inter-neuron gaps rarely
  below ~7 µm, so a sub-2.75 µm gap is overwhelmingly one broken neuron; GENERALIZES
  + OK) and the tight-split-gap finding (id 63 — true split gaps 99% under ~6.5 µm,
  GENERALIZES + OK), paired with the failure report's own cable-caliber semantics
  (matched endpoint caliber = one broken neuron; mismatched = two fused neurites).
  DISTRUSTS the brain-specific absolute-radius / split-on-thin-process finding (#17,
  DOES-NOT-GENERALIZE) — this tier uses only the RELATIVE rad_ratio of the two
  endpoints, never any absolute radius threshold.
- **Gen 13 → candidate (HIGH-rad-ratio TIP-INTO-SHAFT recall tier — a caliber
  FLOOR, the inverse of the matched-caliber ceiling tiers):** Diagnosis was purely
  RECALL-bound, NOT precision: the parent scored 286 correct − 0 train-false (= 286),
  but of 540 reachable REAL splits 254 are still MISSED, all 3 reachable FALSE joins
  correctly rejected (precision perfect, no over-merges). The MISSED bucket is now
  DOMINATED by tip-into-shaft sites (`deg_a == 1, deg_b == 2`) at gap 2.0–2.75 whose
  rad_ratio is HIGH (≥ 2.0, ranging up to ~2.75): a THIN neurite tip (rad_a ~0.75–1.06)
  reconnecting to the MIDDLE of a THICK shaft (rad_b ~1.5–2.0). Examples from the
  report (gap, colinear_cos, rad_ratio): `2.06 / −0.18 / 2.27`, `2.08 / 0.07 / 2.21`,
  `2.14 / −0.11 / 2.17`, `2.15 / 0.31 / 2.58`, `2.16 / 0.22 / 2.74`, `2.18 / −0.23 /
  2.44`, `2.55 / 0.04 / 2.66`, `2.62 / 0.12 / 2.69`, `2.74 / 0.49 / 2.73`. They are
  currently REJECTED because the gen10/11 TIP-INTO-SHAFT tier is a caliber CEILING
  (`rad_ratio <= TIP_SHAFT_RAD_MATCH`, 1.50), and their colinearity is low/scattered so
  the 0.55/0.60 floors miss them too. KEY INSIGHT: `deg_b == 2` means a tip joining the
  MIDDLE of a cable = a branch/T-junction, and a real T-junction is a thin daughter
  neurite emanating from a thick parent shaft → naturally VERY mismatched caliber (high
  rad_ratio). So these high-rad_ratio sites are not noise — they are the canonical
  high-caliber-mismatch T-junction reals that a CEILING tier structurally excludes.
  Lever (a genuinely NEW mechanism — a rad_ratio FLOOR, the INVERSE of every prior
  caliber tier, never tried before): added `TIP_SHAFT_HIGH_RAD_RATIO = 2.0` and a
  sibling tier placed immediately AFTER the matched-caliber TIP-INTO-SHAFT tier and
  BEFORE the colinearity tier (reusing the already-computed `deg_a`/`deg_b` and the
  already-computed `rr_ts`): if `MICRO_GAP_UM < s.gap_um <= TIP_SHAFT_GAP_UM AND
  deg_a == 1 AND deg_b == 2 AND rr_ts_hi >= TIP_SHAFT_HIGH_RAD_RATIO`, accept on
  distance + topology + high-caliber-mismatch, WAIVING colinearity (the lattice-jitter
  rationale is topology-independent, exactly as in the MICRO / tip-to-tip / tip-into-
  shaft tiers). The gap prior does the heavy lifting (gap ≤ 2.75 ≪ 7 µm ⇒ overwhelmingly
  a REAL break by population statistics); the 2.0 FLOOR is the cheap precision guard.
  FALSE-free: all 3 reachable FALSE joins have rad_ratio < 2.0 (1.97, 1.56, 1.67); the
  only `deg_b == 2` FALSE (@gap 2.20) is at 1.97, so a 2.0 floor leaves a 0.03 margin
  above it, AND the `(1.50, 2.00)` caliber band is deliberately left as a protective
  no-go zone (between the matched-tier ceiling and this floor) bracketing exactly that
  FALSE; the other two reachable FALSE are `deg_b == 1`, so this `deg_b == 2` tier never
  sees them → zero new train-false merges. PURELY ADDITIVE (new tier + `continue`): a
  site this accepts currently fails ALL existing paths, so train-correct cannot drop
  below the parent's 286 — it can only ADD net-new correct repairs (the report shows
  dozens of qualifying missed reals) → score > 286. KEPT everything else byte-for-byte:
  did NOT touch `_is_colinear_split`, `COLINEAR_AVG_ACCEPT`, `GAP_THRESHOLD_UM`,
  `TIP_SHAFT_RAD_MATCH` (1.50), `TIP_SHAFT_GAP_UM`, `MICRO_GAP_UM`, the tip-to-tip
  tiers/constants, the colinearity tier, the helpers, or the `ENUM_PARAMS` dict; did NOT
  re-add a union-find component-size cap (a prior gen's MAX_MERGE_COMPONENT was rejected,
  held-out −2.000); did NOT raise `MICRO_GAP_UM`. Image deliberately NOT used: the
  warm-start probe AUC is only 0.69 and REAL bridge_ratio (0.94–1.02) OVERLAPS FALSE
  (0.97–0.99); the only separating column is absolute `bridge_min` (FALSE 173–259 vs
  REAL 27–158), which is acquisition/brain-specific and non-generalizable, so an image
  gate would not separate the reachable false joins. AutoDiscovery support: finding #1
  (id 30 — gap-distance AUC 0.998, inter-neuron gaps rarely below ~7 µm, so a sub-2.75 µm
  gap is overwhelmingly one broken neuron; GENERALIZES + UPHELD/OK) and finding #21
  (id 63 — true split gaps tightly bounded ~99% under ~6.5 µm; GENERALIZES + OK), which
  together justify trusting the gap prior to carry these gap ≤ 2.75 sites as REAL.
  DISTRUSTS the brain-specific absolute-radius / split-on-thin-process finding (#17,
  DOES-NOT-GENERALIZE) and absolute intensity — this tier uses only the RELATIVE
  rad_ratio of the two endpoints, never any absolute radius or intensity threshold.
- **Gen 14 → candidate (HIGH-rad-ratio TIP-TO-TIP recall tier — a caliber FLOOR,
  the tip-to-tip analogue of the gen13 tip-into-shaft floor):** Diagnosis was purely
  RECALL-bound, NOT precision: the parent scored 317 correct − 0 train-false (= 317),
  with 223 of the reachable REAL splits still MISSED and all 3 reachable FALSE joins
  correctly rejected (precision perfect, no over-merges). Context: gen13 had added a
  HIGH-rad-ratio tip-INTO-shaft floor (deg_b == 2, rad_ratio ≥ 2.0) that was ACCEPTED
  and lifted the parent's correct count from 286 → 317. The MISSED bucket now contains
  a large, cleanly-separable cluster of TIP-TO-TIP reals (`deg_a == 1 AND deg_b == 1`)
  with HIGH rad_ratio (≥ 2.0) at gaps 2.45–2.83 — a thin distal end (~0.75–1.0)
  meeting a thick end (~1.5–2.4). Examples from the report (gap, colinear_cos,
  rad_ratio): `2.45 / 0.04 / 2.00`, `2.45 / −0.34 / 2.41`, `2.83 / −0.11 / 2.00`,
  `2.83 / −0.09 / 2.00`, `2.83 / 0.37 / 2.12`, `2.83 / 0.52 / 2.00`. They are
  currently REJECTED because the EXTENDED TIP-TO-TIP tier (gen09) only accepts
  deg_b == 1 when rad_ratio ≤ TIP_TIP_RAD_MATCH (1.50), and their colinearity is
  low/scattered so the 0.55/0.60 floors miss them too. KEY INSIGHT: deg_b == 1 = two
  free ENDS meeting (tip-to-tip); the matched-caliber EXTENDED tier already captures
  the symmetric-caliber tip-to-tip reals, so these high-MISMATCH ones are the
  structural complement a CEILING tier excludes — exactly mirroring why gen13's
  tip-into-shaft FLOOR was needed. Lever (a NEW tier — a rad_ratio FLOOR for
  tip-to-tip, the direct analogue of gen13's tip-into-shaft floor): added
  `TIP_TIP_HIGH_RAD_GAP_UM = 2.9` and a sibling tier placed immediately AFTER the
  gen13 HIGH-rad-ratio tip-into-shaft tier and BEFORE the colinearity tier (reusing
  the already-computed `rr_ts` as `rr_tt_hi` and the existing
  `TIP_SHAFT_HIGH_RAD_RATIO` = 2.0 as the floor): if `MICRO_GAP_UM < s.gap_um <=
  TIP_TIP_HIGH_RAD_GAP_UM AND deg_a == 1 AND deg_b == 1 AND rr_tt_hi >=
  TIP_SHAFT_HIGH_RAD_RATIO`, accept on distance + topology + high-caliber-mismatch,
  WAIVING colinearity (the lattice-jitter rationale is topology-independent, exactly
  as in the MICRO / tip-to-tip / tip-into-shaft tiers). The gap prior does the heavy
  lifting (gap ≤ 2.9 ≪ 7 µm ⇒ overwhelmingly a REAL break by population statistics);
  the 2.0 FLOOR is the cheap precision guard. FALSE-free with DOUBLE protection: both
  reachable deg_b == 1 FALSE joins have rad_ratio < 2.0 (1.56 and 1.67), so the 2.0
  floor alone excludes them (margin ≥ 0.33); AND the only deg_b == 1 FALSE near this
  window (@gap 3.00) is also beyond the 2.9 gap ceiling — a double margin. The
  deg_b == 2 FALSE (@gap 2.20) is never seen by this deg_b == 1 tier. PURELY ADDITIVE
  (new tier + `continue`): a site this accepts currently fails ALL existing paths, so
  train-correct cannot drop below the parent's 317 — it can only ADD net-new correct
  repairs (the report shows a dense cluster of qualifying missed reals at gap
  2.45–2.83) → score > 317. KEPT everything else byte-for-byte: did NOT touch the
  gen13 deg_b == 2 tier, `TIP_SHAFT_GAP_UM`, `TIP_SHAFT_HIGH_RAD_RATIO`,
  `TIP_TIP_RAD_MATCH`, `COLINEAR_AVG_ACCEPT`, `GAP_THRESHOLD_UM`, `MICRO_GAP_UM`, the
  colinearity tier, the helpers, or the `ENUM_PARAMS` dict; did NOT re-add a
  union-find component-size cap (a prior gen's MAX_MERGE_COMPONENT was rejected,
  held-out −2.000). Did NOT chase the deg_b == 2 reals stuck in the rad_ratio
  (1.50, 2.00) band — those genuinely overlap the FALSE @2.20 in gap/colinear/rad and
  have no clean separator, so that band is a deliberate no-go zone. Image deliberately
  NOT used: the warm-start probe AUC is only 0.69 and REAL bridge_ratio (0.94–1.02)
  OVERLAPS FALSE (0.97–0.99); the only separating column is absolute `bridge_min`
  (FALSE 173–259 vs REAL 27–158), which is acquisition/brain-specific and
  non-generalizable, so an image gate would not separate the reachable false joins.
  AutoDiscovery support: finding #1 (id 30 — gap-distance AUC 0.998, inter-neuron gaps
  rarely below ~7 µm, so a sub-2.9 µm gap is overwhelmingly one broken neuron;
  GENERALIZES + UPHELD/OK) and finding #21 (id 63 — true split gaps tightly bounded
  ~99% under ~6.5 µm, so 2.9 µm is well inside; GENERALIZES + OK), which together
  justify trusting the gap prior to carry these gap ≤ 2.9 sites as REAL. DISTRUSTS the
  brain-specific absolute-radius / split-on-thin-process finding (#17,
  DOES-NOT-GENERALIZE) and absolute intensity — this tier uses only the RELATIVE
  rad_ratio of the two endpoints, never any absolute radius or intensity threshold.
- **Gen 15 → candidate (MID-CALIBER TIP-INTO-SHAFT recall tier — a gap-floored
  band re-fill, a NEW kind of lever):** Diagnosis was purely RECALL-bound, NOT
  precision: the parent scored 323 correct − 0 train-false (= 323), with 217 of the
  reachable REAL splits still MISSED and all 3 reachable FALSE joins correctly
  rejected (precision perfect, no over-merges). Context: gen14 had added a HIGH-rad-
  ratio tip-to-tip floor (deg_b == 1, rad_ratio ≥ 2.0, gap ≤ 2.9) that was ACCEPTED
  and lifted the parent's correct count 317 → 323. The two existing `deg_b == 2`
  (tip-into-shaft) tiers leave a DELIBERATE GAP in caliber space: the matched-caliber
  tier (gen10/11) accepts rad_ratio ≤ TIP_SHAFT_RAD_MATCH (1.50) and the gen13 high-
  mismatch tier accepts rad_ratio ≥ TIP_SHAFT_HIGH_RAD_RATIO (2.0), so the MIDDLE band
  `1.50 < rad_ratio < 2.0` was left a NO-GO zone for ONE reason only: the FALSE join
  @gap 2.20 has rad_ratio 1.97, which sits in that band. The dominant remaining missed
  cluster is exactly these mid-caliber tip-into-shaft reals (deg_a == 1, deg_b == 2,
  1.50 < rad_ratio < 2.0), densely populated at gaps 2.49–2.73. Examples from the
  report (gap, colinear_cos, rad_ratio): `2.49 / 0.33 / 1.86`, `2.50 / 0.15 / 1.90`,
  `2.51 / 0.03 / 1.89`, `2.51 / 0.20 / 1.66`, `2.57 / 0.02 / 1.82`, `2.58 / 0.36 /
  1.65`, `2.69 / 0.25 / 1.51`, `2.70 / 0.39 / 1.94`, `2.72 / 0.06 / 1.69`, `2.72 /
  −0.02 / 1.60`, `2.72 / 0.46 / 1.78`, `2.73 / 0.17 / 1.76`. They fall through EVERY
  existing path: gap > TIP_TIP_GAP_UM (so both tip-to-tip tiers skip them on the
  deg_b == 1 requirement); deg_b == 2 with rad_ratio in (1.50, 2.0) so both deg_b == 2
  caliber tiers skip them; low/scattered colinear_cos so the 0.55/0.60 floors miss
  them too. KEY PRECISION INSIGHT: the FALSE @2.20 (rad 1.97) is the ONLY reachable
  deg_b == 2 FALSE join AT ANY GAP — the other two reachable FALSE (@gap 2.45 rad 1.56
  and @gap 3.00 rad 1.67) are deg_b == 1, so a deg_b == 2 tier NEVER sees them.
  Therefore the mid-caliber band can be safely re-opened PROVIDED we gate on gap ABOVE
  that single FALSE. Lever (a genuinely NEW kind of lever — a GAP-floor precision gate
  that re-fills a previously carved-out caliber band, distinct from the rad-ceiling /
  rad-floor tiers): added `DEG2_MID_GAP_FLOOR = 2.40` and a sibling tier placed
  immediately AFTER the gen13 HIGH-rad-ratio tip-into-shaft tier and BEFORE the gen14
  HIGH-rad tip-to-tip tier (reusing the already-computed `rr_ts` as `rr_ts_mid` and
  the existing TIP_SHAFT_RAD_MATCH = 1.50 / TIP_SHAFT_HIGH_RAD_RATIO = 2.0 as the band
  edges and TIP_SHAFT_GAP_UM = 2.75 as the upper gap bound): if `DEG2_MID_GAP_FLOOR <
  s.gap_um <= TIP_SHAFT_GAP_UM AND deg_a == 1 AND deg_b == 2 AND TIP_SHAFT_RAD_MATCH <
  rr_ts_mid < TIP_SHAFT_HIGH_RAD_RATIO`, accept on distance + topology + mid-caliber,
  WAIVING colinearity. The gap prior does the heavy lifting (gap 2.40–2.75 ≪ 7 µm ⇒
  overwhelmingly a REAL break by population statistics); the 2.40 FLOOR is the cheap
  precision guard. FALSE-free: the 2.40 floor sits 0.20 µm ABOVE the only reachable
  deg_b == 2 FALSE (@gap 2.20), so no reachable deg_b == 2 FALSE can enter; the 2.40
  floor intentionally sacrifices a few mid-caliber reals at gap 2.21–2.35 (too close
  to the FALSE @2.20 to separate safely) in exchange for a wide, generalizable margin;
  the other two reachable FALSE are deg_b == 1, never seen by this deg_b == 2 tier →
  zero new train-false merges. PURELY ADDITIVE (new tier + `continue`) and mutually
  exclusive by caliber band with the two existing deg_b == 2 tiers (they handle
  rad ≤ 1.50 and rad ≥ 2.0; this handles strictly between): a site this accepts
  currently fails ALL existing paths, so train-correct cannot drop below the parent's
  323 — it can only ADD net-new correct repairs (the report shows ~13+ visible
  qualifying missed reals at gap 2.49–2.73, plus more in the hidden rows) → score >
  323. KEPT everything else byte-for-byte: did NOT touch the gen13 / gen14 tiers,
  `TIP_SHAFT_RAD_MATCH`, `TIP_SHAFT_HIGH_RAD_RATIO`, `TIP_SHAFT_GAP_UM`,
  `TIP_TIP_HIGH_RAD_GAP_UM`, `COLINEAR_AVG_ACCEPT`, `GAP_THRESHOLD_UM`, `MICRO_GAP_UM`,
  the colinearity tier, the helpers, or the `ENUM_PARAMS` dict; did NOT re-add a
  union-find component-size cap (a prior gen's MAX_MERGE_COMPONENT was rejected,
  held-out −2.000); did NOT raise `TIP_SHAFT_GAP_UM` (there are missed reals above
  2.75 but that is a separate lever for a later generation). Image deliberately NOT
  used: the warm-start probe AUC is only 0.69 and REAL bridge_ratio (0.94–1.02)
  OVERLAPS FALSE (0.97–0.99); the only separating column is absolute `bridge_min`
  (FALSE 173–259 vs REAL 27–158), which is acquisition/brain-specific and
  non-generalizable, so an image gate would not separate the reachable false joins.
  AutoDiscovery support: finding #1 (id 30 — gap-distance AUC 0.998, inter-neuron
  gaps rarely below ~7 µm, so all these gap 2.40–2.75 sites are overwhelmingly REAL
  by population statistics; a deg_b == 2 FALSE above gap 2.40 would be extraordinarily
  rare; GENERALIZES + UPHELD/OK) and finding #21 (id 63 — true split gaps tightly
  bounded ~99% under ~6.5 µm, so 2.40–2.75 µm is well inside; GENERALIZES + OK), which
  together justify trusting the gap prior to carry these gap ≤ 2.75 sites as REAL.
  DISTRUSTS the brain-specific absolute-radius / split-on-thin-process finding (#17,
  DOES-NOT-GENERALIZE) and absolute intensity — this tier uses only the RELATIVE
  rad_ratio of the two endpoints, never any absolute radius or intensity threshold.
- **Gen 16 → candidate (extended-gap tip-into-shaft gap-ceiling extension to
  (2.75, 3.0]):** Diagnosis was RECALL-bound, NOT precision: the parent scored 336
  correct − 0 train-false (= 336), but of 540 reachable REAL splits only 336 were
  accepted — 204 MISSED, all 3 reachable FALSE joins correctly rejected (precision
  perfect, no over-merges). Gen15's MID-CALIBER tip-into-shaft tier was accepted
  (323 → 336) and CLEARED the gap 2.49–2.73 deg_b == 2 cluster — the missed list now
  jumps from gap 2.45 directly to 2.76. The dominant remaining missed cluster is
  TIP-INTO-SHAFT (deg_a == 1, deg_b == 2) reals JUST ABOVE the 2.75 ceiling, densely
  packed at gap 2.76–2.98 across the FULL rad_ratio range (examples (gap,
  colinear_cos, rad_ratio): (2.76, 0.52, 1.23), (2.81, 0.13, 2.08), (2.85, 0.12,
  2.06), (2.90, 0.07, 2.63), (2.97, 0.38, 2.75), (2.98, 0.32, 2.61)). These are
  missed ONLY because gap > 2.75: all three deg_b == 2 tiers (matched-caliber
  rad ≤ 1.50, mid-caliber 1.50 < rad < 2.0 gated above gap 2.40, high-mismatch
  rad ≥ 2.0) cap at the SAME TIP_SHAFT_GAP_UM = 2.75 ceiling, and their scattered/
  negative colinear_cos fails the colinearity tier. KEY PRECISION INSIGHT: the FALSE
  @gap 2.20 is the ONLY reachable deg_b == 2 FALSE join AT ANY GAP — the gap-3.00
  FALSE is deg_b == 1 (never seen by a deg_b == 2 tier) and the gap-2.45 FALSE is
  also deg_b == 1. So the deg_b == 2 gap ceiling can be extended above 2.75 with a
  WIDE precision margin (≥ 0.55 µm above the only deg_b == 2 FALSE @2.20), and there
  is NO deg_b == 2 FALSE anywhere in (2.75, 3.0]. Lever (a single gap-ceiling
  extension for the deg_b == 2 topology, mirroring the three existing tiers' union):
  added `TIP_SHAFT_EXT_GAP_UM = 3.0` and a new tier placed immediately AFTER the
  gen15 MID-CALIBER tip-into-shaft block and BEFORE the gen14 HIGH-rad tip-to-tip
  tier (keeping all deg_b == 2 tiers grouped, reusing the already-computed
  `deg_a`/`deg_b`): if `TIP_SHAFT_GAP_UM < s.gap_um <= TIP_SHAFT_EXT_GAP_UM AND
  deg_a == 1 AND deg_b == 2`, accept across the FULL rad range and WAIVE colinearity
  — exactly as the union of the three existing deg_b == 2 tiers does for gap ≤ 2.75.
  FALSE-free with a WIDE margin: the new floor (2.75) sits 0.55 µm above the only
  deg_b == 2 FALSE (@gap 2.20) and there is no deg_b == 2 FALSE in (2.75, 3.0]; the
  gap-3.00 FALSE is deg_b == 1 and is NEVER seen by this deg_b == 2 tier → zero new
  train-false merges. deg_b == 1 (tip-to-tip) was DELIBERATELY NOT extended: a
  deg_b == 1 FALSE sits at gap 3.00, so extending tip-to-tip here would catch it —
  this lever is deg_b == 2 ONLY. The gap prior does the heavy lifting (findings #1,
  #21): at gap 2.75–3.0 ≪ 7 µm a tip-into-shaft site is overwhelmingly a REAL break
  (gap-distance AUC 0.998, F1-optimal 6.84 µm; ~99% of true split gaps under
  ~6.5 µm), so even held-out a deg_b == 2 FALSE in this window is extraordinarily
  rare. PURELY ADDITIVE (new tier + `continue`): these gap-2.76–2.98 deg_b == 2 sites
  currently fail ALL existing paths, so train-correct cannot drop below the parent's
  336 — it can only ADD net-new correct repairs (the report shows ~14+ visible
  qualifying missed reals at gap 2.76–2.98, plus more in the 144 hidden rows) →
  score > 336. KEPT everything else byte-for-byte: did NOT touch `TIP_SHAFT_GAP_UM`,
  `DEG2_MID_GAP_FLOOR`, `TIP_SHAFT_RAD_MATCH`, `TIP_SHAFT_HIGH_RAD_RATIO`,
  `TIP_TIP_HIGH_RAD_GAP_UM`, `COLINEAR_AVG_ACCEPT`, `GAP_THRESHOLD_UM`,
  `MICRO_GAP_UM`, any existing tier, the helpers, or the `ENUM_PARAMS` dict; did NOT
  re-add a union-find component-size cap (a prior gen's MAX_MERGE_COMPONENT was
  rejected, held-out −2.000); did NOT extend the deg_b == 1 gap ceiling (a deg_b == 1
  FALSE @gap 3.00 makes tip-to-tip unsafe to extend here). Image deliberately NOT
  used: the warm-start probe AUC is only 0.69 and REAL bridge_ratio (0.94–1.02)
  OVERLAPS FALSE (0.97–0.99); the only separating column is absolute `bridge_min`
  (FALSE 173–259 vs REAL 27–158), which is acquisition/brain-specific and
  non-generalizable, so an image gate would not separate the reachable false joins.
  AutoDiscovery support: finding #1 (id 30 — gap-distance AUC 0.998, inter-neuron
  gaps rarely below ~7 µm, F1-optimal 6.84 µm, so all these gap 2.75–3.0 sites are
  overwhelmingly REAL by population statistics and a deg_b == 2 FALSE in this window
  would be extraordinarily rare; GENERALIZES + UPHELD/OK) and finding #21 (id 63 —
  true split gaps tightly bounded ~99% under ~6.5 µm, so 2.75–3.0 µm is well inside;
  GENERALIZES + OK). DISTRUSTS the brain-specific absolute-radius / split-on-thin-
  process finding (#17, DOES-NOT-GENERALIZE) and absolute intensity — this tier uses
  only the RELATIVE rad_ratio (and here only the topology/gap), never any absolute
  radius or intensity threshold.
- **Gen 17 → candidate (MID-GAP TIP-TO-TIP recall tier — distance + topology over the
  FALSE-free (2.5, 2.9] window, generalizing the gen14 HIGH-rad tip-to-tip tier to the
  FULL rad range):** Diagnosis was purely RECALL-bound, NOT precision: the gen17 parent
  scored 356 correct − 0 train-false (= 356), with 184 of the reachable REAL splits
  still MISSED and all 3 reachable FALSE joins correctly rejected (precision perfect,
  no over-merges). The 3 reachable FALSE joins are constant across gens: (gap 2.20,
  deg_b 2, rad 1.97), (gap 2.45, deg_b 1, rad 1.56), (gap 3.00, deg_b 1, rad 1.67).
  The dominant untapped missed cluster is TIP-TO-TIP reals (`deg_a == 1 AND deg_b == 1`)
  at gap ≈ 2.83 (= √8 ≈ 2.828, the voxel-lattice diagonal), all with rad_ratio < 2.0
  and colinear_cos < 0.6 — examples from the report (gap, cos, rad): (2.83, 0.39, 1.44),
  (2.83, −0.20, 1.00), (2.83, 0.48, 1.95), (2.83, 0.50, 1.41), (2.83, 0.42, 1.17),
  (2.83, −0.38, 1.67), (2.83, 0.14, 1.58) — at least 9 visible in the missed bucket.
  They fall through EVERY existing path: the gen14 HIGH-rad tip-to-tip tier (tier 4c)
  requires rad_ratio ≥ 2.0 so it SKIPS them, and the colinearity tier rejects them
  (cos < 0.6 / scattered); gap > 2.5 so both lower tip-to-tip tiers skip on gap. KEY
  PRECISION INSIGHT (the FALSE-free window): both reachable deg_b == 1 FALSE joins sit
  at gap 2.45 and gap 3.00. The window (2.5, 2.9] therefore contains NO reachable
  deg_b == 1 FALSE — the FALSE @2.45 is BELOW 2.5 (and the EXTENDED tip-to-tip tier 3
  already owns gap ≤ 2.5), the FALSE @3.00 is ABOVE 2.9; the floor 2.5 sits 0.05 µm
  above the @2.45 FALSE and the ceiling 2.9 sits 0.10 µm below the @3.00 FALSE. The
  deg_b == 2 FALSE @2.20 is never seen by this deg_b == 1-gated tier. Lever (a NEW
  tier that GENERALIZES the gen14 HIGH-rad tip-to-tip tier from rad_ratio ≥ 2.0 to the
  FULL rad range within the FALSE-free window, REUSING existing constants — floor
  `TIP_TIP_EXT_GAP_UM` = 2.5, ceiling `TIP_TIP_HIGH_RAD_GAP_UM` = 2.9, no new
  constant): inserted immediately AFTER the gen14 HIGH-rad tip-to-tip tier's `continue`
  and BEFORE the distance-graded colinearity gate (reusing the already-computed
  `deg_a`/`deg_b`): if `TIP_TIP_EXT_GAP_UM < s.gap_um <= TIP_TIP_HIGH_RAD_GAP_UM AND
  deg_a == 1 AND deg_b == 1`, accept on distance + topology alone, WAIVING both caliber
  and colinearity (the latter is lattice jitter at these sub-3 µm scales, exactly as in
  the MICRO / TIP-TO-TIP tiers). PURELY ADDITIVE (new tier + `continue`): any site this
  accepts currently fails EVERY existing path (gap > 2.5 so both lower tip-to-tip tiers
  skip on gap; rad < 2.0 so tier 4c skips; cos < 0.6 / scattered so the colinearity
  tier rejects), so train-correct cannot drop below the parent's 356 — it can only ADD
  net-new correct repairs (the report shows ≥ 9 visible qualifying missed reals at gap
  ≈ 2.83, plus more in the hidden rows) → score > 356, with zero new train-false merges
  (the FALSE-free window guarantees it). KEPT everything else byte-for-byte: did NOT
  touch the gen14 deg_b == 1 tier, any deg_b == 2 tier, `TIP_SHAFT_HIGH_RAD_RATIO`,
  `TIP_TIP_RAD_MATCH`, `COLINEAR_AVG_ACCEPT`, `GAP_THRESHOLD_UM`, `MICRO_GAP_UM`, the
  colinearity tier (no per-arm colinearity floor decoupling), the helpers, or the
  `ENUM_PARAMS` dict; did NOT re-add a union-find component-size cap (a prior gen's
  MAX_MERGE_COMPONENT was rejected, held-out −2.000); did NOT raise `MICRO_GAP_UM`.
  Image deliberately NOT used: the warm-start probe AUC is only 0.69 (moderate) and the
  FALSE bridge_ratios (0.97–0.99) OVERLAP the REAL range (0.94–1.02); only acquisition-
  specific absolute brightness separates them, so geometry is the better lever here and
  no `gap_bridge_evidence` image read was added. AutoDiscovery support: finding #1 (id
  30 — gap-distance AUC 0.998, inter-neuron gaps rarely below ~7 µm, F1-optimal 6.84 µm,
  so a gap 2.5–2.9 µm tip-to-tip site is overwhelmingly a REAL break; GENERALIZES +
  UPHELD/OK) and finding #21 (id 63 — true split gaps tightly bounded ~99% under
  ~6.5 µm, so 2.5–2.9 µm is well inside; GENERALIZES + OK), which together justify
  accepting on distance + topology in this FALSE-free window. DISTRUSTS the brain-
  specific absolute-radius / split-on-thin-process finding (#17, DOES-NOT-GENERALIZE),
  absolute radius/intensity, Z-axis anisotropy, centrifugal branch-order, and omit/
  split-near-merge co-location (all non-generalizing) — this tier uses only gap + tip
  topology, never any absolute radius or intensity threshold.
- **Gen 18 → candidate (2026-06-23) (SECOND EXTENDED-GAP TIP-INTO-SHAFT recall tier —
  extend deg_b==2 acceptance one further step into (3.0, 3.25]):** Diagnosis was purely
  RECALL-bound, NOT precision: the gen18 parent scored 365 correct − 0 train-false
  (= 365), with 175 of the reachable REAL splits still MISSED and all 3 reachable FALSE
  joins correctly rejected. The 3 reachable FALSE joins are constant across gens:
  (gap 2.20, deg_b 2, rad 1.97), (gap 2.45, deg_b 1, rad 1.56), (gap 3.00, deg_b 1,
  rad 1.67). The dominant CLEANLY-SEPARABLE remaining missed cluster is TIP-INTO-SHAFT
  reals (`deg_a == 1 AND deg_b == 2`) at gap just ABOVE 3.0, across the FULL rad_ratio
  range — at least 13 visible in the report's missed bucket (gap, rad_ratio):
  (3.01, 1.38), (3.01, 2.54), (3.02, 1.13), (3.02, 2.33), (3.03, 1.88), (3.04, 2.66),
  (3.05, 2.77), (3.07, 1.70), (3.10, 2.69), (3.10, 1.47), (3.11, 1.48), (3.12, 2.66),
  (3.12, 1.32). They sit immediately above the gen16 EXTENDED-GAP tip-into-shaft tier's
  ceiling (`TIP_SHAFT_EXT_GAP_UM` = 3.0), so they are missed ONLY because gap > 3.0.
  KEY PRECISION INSIGHT (the FALSE-free window): the FALSE @gap 2.20 is the ONLY
  reachable `deg_b == 2` FALSE join AT ANY GAP — there is NO `deg_b == 2` FALSE anywhere
  above 2.20. The other two reachable FALSE joins are `deg_b == 1` (@2.45 and @3.00) and
  are NEVER seen by a `deg_b == 2`-gated tier. So the window (3.0, 3.25] is provably
  FALSE-free for `deg_b == 2`, with the nearest `deg_b == 2` FALSE 0.8 µm below the new
  floor. Lever (a NEW tier that MIRRORS gen16's tier 4e structure one step further,
  adding a new constant `TIP_SHAFT_EXT2_GAP_UM` = 3.25 ALONGSIDE gen16's
  `TIP_SHAFT_EXT_GAP_UM` = 3.0, which is left UNCHANGED): inserted immediately AFTER the
  gen16 EXTENDED-GAP tip-into-shaft tier's `continue` and BEFORE the gen14 HIGH-rad
  tip-to-tip tier (reusing the already-computed `deg_a`/`deg_b`): if
  `TIP_SHAFT_EXT_GAP_UM < s.gap_um <= TIP_SHAFT_EXT2_GAP_UM AND deg_a == 1 AND
  deg_b == 2`, accept across the FULL rad_ratio range, WAIVING colinearity (lattice
  jitter at these sub-3.25 µm scales, exactly as in the MICRO / TIP-TO-TIP / tier-4e
  tiers). PURELY ADDITIVE (new tier + `continue`): the (3.0, 3.25] `deg_b == 2` sites
  currently fall through to the colinearity tier and are mostly rejected (scattered
  colinear_cos), so train-correct cannot drop below the parent's 365 — it can only ADD
  net-new correct repairs (≥ 13 visible qualifying missed reals at gap 3.01–3.12) →
  score > 365, with zero new train-false merges (the FALSE-free window guarantees it).
  Does NOT touch the gen16 tier 4e: `TIP_SHAFT_EXT_GAP_UM` and its block are untouched,
  so all currently-accepted (2.75, 3.0] sites remain accepted. KEPT everything else
  byte-for-byte: did NOT add a union-find component-size cap, did NOT decouple the
  per-arm colinearity floor, did NOT raise `MICRO_GAP_UM`, did NOT change
  `TIP_SHAFT_EXT_GAP_UM` (gen16's ceiling), and did NOT touch any other tier, helper,
  constant, or the `ENUM_PARAMS` dict. Image deliberately NOT used: the report's
  warm-start probe shows bridge_ratio AUC = 0.69 (moderate), the 3 FALSE joins all have
  bridge_ratio 0.97–0.99 sitting INSIDE the REAL range (0.94–1.02), and the probe was
  only measured at gap ~2.55–2.62 (NOT the 3.0–3.25 band), so there is no basis to gate
  on image here — geometry is the better lever. AutoDiscovery support: finding #1 (id
  30 — gap-distance AUC 0.998, inter-neuron gaps rarely below ~7 µm, F1-optimal 6.84 µm,
  so a gap 3.0–3.25 µm tip-into-shaft site is overwhelmingly a REAL break; GENERALIZES +
  UPHELD/OK) and finding #21 (id 63 — true split gaps tightly bounded ~99% under
  ~6.5 µm, so 3.0–3.25 µm is well inside; GENERALIZES + OK), which together justify
  accepting on distance + topology in this FALSE-free `deg_b == 2` window. Uses ONLY
  GENERALIZES + UPHELD/OK findings; relies on no DOES-NOT-GENERALIZE / PARTIAL /
  WEAKENED / OVERTURNED finding (Z-axis anisotropy, centrifugal branch-order, absolute
  radius/intensity, omit/split-near-merge co-location) — this tier uses only gap + tip
  topology, never any absolute radius or intensity threshold.
