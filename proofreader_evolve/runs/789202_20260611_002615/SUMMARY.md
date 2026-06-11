# Evolution Run Summary — `789202_20260611_002615`

## Header

| Field | Value |
|---|---|
| Run id | `789202_20260611_002615` |
| Brain id | `789202` |
| Generations | 5 |
| Accepted | 2 (gen 3, gen 4) |
| Baseline held-out Edge Accuracy | 76.171 |
| Final held-out Edge Accuracy | 76.778 |
| **Net held-out gain (final − baseline)** | **+0.607** |
| Total agent cost | **$13.49** |

Task: repair **split errors** (one true neuron broken into fragments) by
unifying fragment label pairs, **without** introducing merge errors. Fitness is
run-length-weighted **Edge Accuracy** on held-out skeletons; ERL is the
tie-breaker.

## Trajectory

| Gen | What it tried | Train EdgeAcc | Held-out EdgeAcc | Δ vs parent | Outcome |
|---|---|---|---|---|---|
| 1 | Proximity + PCA tangent-direction / collinearity | 79.801 | 76.171 | +0.000 | reverted |
| 2 | Mutual-NN tip-to-tip pairing (gap ≤ 4 µm, union-find cap 3) | 79.801 | 73.363 | −2.808 | reverted |
| 3 | Size gate (absorb small fragment ≤ 25 nodes) + Lowe-style ambiguity-margin ratio | 79.801 | 76.705 | +0.533 | **accepted** |
| 4 | Add transitive-fusion guard (union-find, `MERGE_LABEL_CAP = 2`) | 80.808 | 76.778 | +0.074 | **accepted** |
| 5 | Detect parallel/adjacent neurites via multi-location signature | 80.808 | 76.778 | +0.000 | reverted |

(Parent bars: gens 1–3 compared against the baseline 76.171; gen 4 against
76.705; gen 5 against 76.778.)

## Learned policy (final = gen 4)

Anchor: `gen04/heuristics.accepted.py`. A candidate split site (a fragment tip
`node_a` near a differently-labelled partner `node_b`) is unified into a
`merge_labels` edit ONLY if ALL four guards hold:

1. **Proximity** — physical gap `GAP_THRESHOLD_UM` = **5.0 µm**. Hard cap; basic
   distance prerequisite for any reconnection.
2. **Bounded-downside size gate** — `SMALL_FRAGMENT_MAX_NODES` = **25**. At least
   one fragment must be small, so the policy only ever ABSORBS a small fragment
   into a larger neighbour. Even a wrong merge then costs little run-length, so
   the catastrophic downside is removed by construction.
3. **Ambiguity margin (Lowe-style ratio)** — `MARGIN_MIN_SECOND_UM` = **4.0 µm**
   OR `MARGIN_RATIO` = **0.5**. Accept only if the chosen partner is
   unambiguously the tip's closest differently-labelled fragment (runner-up far
   away, or best/second gap < 0.5). Crossings, where many neurites sit at similar
   distance, have a small margin and are rejected.
4. **Transitive-fusion guard** — `MERGE_LABEL_CAP` = **2**. Surviving candidates
   are admitted most-confident-first (ascending gap) through a union-find; a
   merge is refused if it would grow a merged class beyond two raw labels. Each
   accepted class holds exactly two labels, so no chain or multi-absorption blob
   can form.

The three principles, per the rules.md: the size gate bounds the **cost** of a
bad merge, the margin test reduces its **probability** at crossings, and the
transitive guard caps the **span** of any merged class.

## What the failures taught

- **Gen 1 (held-out +0.000, reverted).** Proximity + PCA tangent/collinearity.
  Geometric direction agreement did not separate good joins from bad; no net
  improvement, reverted to the no-op seed.
- **Gen 2 (held-out −2.808, reverted).** Mutual nearest-neighbour tip-to-tip
  pairing. It **over-merged and actively hurt**. Key lesson driving the rest of
  the run: because fitness is run-length-weighted, a single wrong merge fusing
  two LONG neurites destroys a large fraction of correctly-traced path length and
  dwarfs many correct small repairs. Reciprocity + small gap does not rule out
  two unrelated neurites crossing. This motivated gen 3's precision-first,
  bounded-downside redesign.
- **Gen 5 (held-out +0.000, reverted).** Targeted residual 2-label over-merges
  (e.g. `553856478+744503863`) flagged at multiple separated world locations —
  the signature of two parallel/adjacent neurites that a cap-of-2 cannot catch.
  Produced no held-out improvement over gen 4 and was reverted.

Note also gen 4's own motivation (it was accepted): gen 3 deduplicated only by
label pair with no transitive guard, so chains of small fragments fused into
multi-label classes (e.g. `375149638+746076720+746076722`) that span long paths
and collapsed ERL on several skeletons (N016 ΔERL −13934, N020 −8859, N010
−2059, N023 −1637, N006 −908) even as Edge Accuracy rose. Gen 4's cap-of-2 guard
removed those long-span over-merges while keeping the legitimate two-label
repairs.

## Caveats

- **Held-out is reused for selection every generation**, so the final 76.778 is
  a selection metric, not an unbiased held-out estimate. The reported net gain is
  optimistic in that sense.
- **The gain is small: +0.607 held-out Edge Accuracy** over five generations and
  $13.49 of agent spend. Most of the lift came from gen 3 (+0.533); gen 4 added
  only +0.074 held-out (its value was chiefly protecting ERL, the tie-breaker,
  not raising Edge Accuracy). Gens 1, 2, and 5 contributed nothing or hurt.
- **Diagnosis text is orchestrator narration, not subagent reasoning.** Every
  generation's `diagnosis` field in the ledger begins with orchestrator chatter
  (e.g. "I'll delegate this to the proofreader-reviser subagent", "Let me verify
  the actual changes"). The trustworthy, plain-language account of why each guard
  exists and what each rejected attempt taught lives in the **rules.md change
  log**, which this summary draws from.

## Pointers

- Final accepted heuristics: `gen04/heuristics.accepted.py`
- Final accepted rules (criteria + change log): `gen04/rules.accepted.md`
- Ledger: `ledger.jsonl`
- Attempts log: `attempts.md`
