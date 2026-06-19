# Evolution Run Summary — `789202_20260616_225052`

- **Run id:** `789202_20260616_225052`
- **Brain id:** `789202`
- **Generations:** 10
- **Accepted:** 1 (gen 3)
- **Net held-out Edge Accuracy gain:** baseline `71.45397` → final `71.52989` = **+0.07593** (percentage points)
- **Total agent cost:** **$59.10**

This run learned a single improvement — a **distance-dependent colinearity floor**
for split-repair merges — accepted at generation 3. Every one of the nine other
generations scored exactly +0.000 on held-out and was reverted. The net gain is
very small (about +0.076 pp of Edge Accuracy).

## Trajectory

Held-out Edge Accuracy is the selection metric. The parent bar is the last accepted
policy's held-out score; a generation is accepted only if it beats that bar (and
passes the no-new-merge / over-split gates).

| Gen | What it tried | Train EdgeAcc | Held-out EdgeAcc | Δ vs parent | Result |
|----:|---------------|--------------:|-----------------:|------------:|--------|
| 1 | Radius (caliber) continuity guard on split-repair accept | 87.52734 | 71.45397 | +0.00000 | reverted |
| 2 | Image gap-connectivity guard (fluorescence across gap) | 87.52734 | 71.45397 | +0.00000 | reverted |
| 3 | **Distance-dependent colinearity floor** (0.94/0.97/0.992 over 2/4/8 µm bands) | 87.52734 | 71.52989 | **+0.07593** | **accepted** |
| 4 | Node-degree penalty on the accept score | 87.52734 | 71.52989 | +0.00000 | reverted |
| 5 | New far band: floor 0.997 over 8–15 µm | 87.52734 | 71.52989 | +0.00000 | reverted |
| 6 | Branch-tangent (multi-edge) colinearity instead of single-edge | 87.52734 | 71.52989 | +0.00000 | reverted |
| 7 | First `split_label` (MergeSite) merge-repair path + accept gate | 87.52734 | 71.52989 | +0.00000 | reverted |
| 8 | `ENUM_PARAMS` enumeration lever (surface more SplitSites) | 87.52734 | 71.52989 | +0.00000 | reverted |
| 9 | Composition flip (`tip_to_shaft=False`, drop `split_max_sites`) | 87.52734 | 71.52989 | +0.00000 | reverted |
| 10 | Proximity-override OR branch (recall-expanding accept) | 87.52734 | 71.52989 | +0.00000 | reverted |

Note: train Edge Accuracy is identical (`87.52734`) across all ten rows — the
collector reports the same train primary for every generation.

## Learned policy (final accepted, gen 3)

A **colinear split-repair** policy with no `split_label`. For each candidate
SplitSite it merges two fragment labels (`merge_labels`) only when the worst-case
colinearity across the gap (`min(cos_in, cos_out)`, via `_colinearity_score`)
clears a floor that **tightens as the gap grows** (`_required_cos`). Anchor:
`gen03/heuristics.accepted.py`.

| Parameter | Value | Why it exists |
|-----------|------:|---------------|
| `GAP_TIGHT_UM` | **2.0** | Boundary of the very-near band. |
| `MIN_COLINEAR_COS` | **0.94** | Floor for `gap ≤ 2.0 µm`; loosest, since very-near pairs are almost certainly the same neurite. |
| `GAP_THRESHOLD_UM` | **4.0** | Boundary of the mid band. |
| `COS_NEAR` | **0.97** | Floor for `2.0 < gap ≤ 4.0 µm`; tighter than the seed to drop marginal mid-gap pairs that raised % Split Edges and caused one wrong fusion. |
| `GAP_FAR_UM` | **8.0** | Boundary of the new far band. |
| `COS_FAR` | **0.992** | Floor for `4.0 < gap ≤ 8.0 µm`; near-perfect continuation only, to bridge longer TRUE gaps the seed's 4 µm box never reached (the omit-reducing merges that drive the gain). Beyond `GAP_FAR_UM`, reject. |
| `TANGENT_WALK_UM` | **6.0** | Arm length over which the local tangent direction is measured for the colinearity test. |

In plain terms: replace the seed's single rectangular accept box
(`gap ≤ 4 AND cos ≥ 0.94`) with a sloped boundary that **both removes** marginal
mid-gap edits **and adds** high-precision far-gap edits — a genuine change to the
proposed edit set, not a fail-open filter.

## What the failures taught

The change log and orchestrator diagnoses converge on one lesson: **on this
held-out set the score barely moves, and only adding true high-precision merges
ever moved it.** Specifically:

- **Gen 1 (radius-continuity guard) and Gen 2 (image gap-connectivity guard):**
  both were *acceptance-side filters that failed open* — on the geometry-only
  default run they never fired, so they left the proposed edit set unchanged
  (+0.000). Lesson: a guard that only rejects can't help if it never triggers; the
  winning gen 3 rule instead *changes* the edit set in both directions.
- **Gens 4–6 (degree penalty, 8–15 µm far band, branch-tangent colinearity):**
  variations on the gen 3 mechanism that produced no held-out delta — the run had
  already converged at the gen 3 optimum on a low-sensitivity held-out set.
- **Gen 7 (`split_label` merge-repair) and Gen 8 (`ENUM_PARAMS` recall lever):**
  the first structurally novel attempts — a whole new edit type and a wider
  candidate stream — still scored +0.000. The fitness gain is carried entirely by
  dropping omitted edges (split-repair), and merge-repair found no net-positive cut.
- **Gens 9–10 (composition flip, proximity-override OR branch):** by this point
  nine consecutive candidates had scored exactly +0.000; the orchestrator noted the
  run was "converged at the gen3 optimum on a low-sensitivity held-out set" and that
  "adding true accepts is the only direction that ever moved the score." Neither
  flip changed the held-out edit set enough to register.

## Caveats

- **Held-out is reused for selection every generation.** The final `71.52989` is a
  *selection* metric, not an unbiased estimate of generalization; it is the score
  the loop was optimizing against directly.
- **The gain is tiny.** +0.07593 pp of Edge Accuracy from a single accepted
  generation, with all nine other generations flat at +0.000. The held-out set is
  low-sensitivity here: structurally distinct attempts (a new edit type, a wider
  candidate stream) registered no change at all.
- **Diagnosis text is orchestrator-narration, not subagent reasoning.** The
  collector's `diagnosis` lines are the orchestrator verifying the subagent's work
  ("Let me verify the import...", "The subagent made the change..."). The
  trustworthy, plain-language account of *why* each guard exists is the **Change
  log in `gen03/rules.accepted.md`**, which this summary follows.
- **No merge-repair is live.** Despite the harness adding a `split_label`
  MergeSite path (gen 7), it was reverted; the accepted policy emits only
  `merge_labels`.

## Pointers (relative to the run dir)

- Final accepted heuristics: `gen03/heuristics.accepted.py`
- Final accepted rules / change log: `gen03/rules.accepted.md`
- Ledger: `ledger.jsonl`
- Orchestrator attempts log: `attempts.md`
