"""
Compare POLICY SIMILARITY between generations with sentence-transformer embeddings.

For each generation of a ``proofreader_evolve`` run this embeds the candidate policy's
natural-language theory (``rules.candidate.md``) — or its code (``heuristics.candidate.py``)
— with a sentence-transformers model, then measures how the policy TEXT drifts across
the search:

  * a full generation×generation cosine-similarity HEATMAP (which gens are alike),
  * CONSECUTIVE similarity sim(gen_i, gen_{i-1}) — the per-step "revision magnitude":
    dips = a big rewrite, plateaus near 1.0 = fine-tuning the same policy,
  * similarity to the SEED (gen 1) and to the FINAL accepted policy — convergence.

Accepted generations are marked, so you can see whether an accepted revision was a
large semantic jump or a small tweak, and whether rejected candidates clustered.

Why chunk-and-mean-pool: rules.md is ~10–20 KB with a large identical boilerplate
header, far beyond a sentence model's ~256-token window. Embedding it raw would encode
only the shared preamble and make every generation look identical. Instead we split
each doc into sections/paragraphs, embed each chunk, and mean-pool the (normalized)
chunk vectors into one document vector — so the whole doc contributes and the EVOLVING
sections (Current criteria / Change log) drive the contrast.

NOT wired into the evolution workflow — run it standalone:

    conda run -n panda python proofreader_evolve/plotting/analyze_policy_similarity.py \\
        794491_train1_test1_20260714_011120

    ... [--source rules|code|both] [--model all-MiniLM-L6-v2] [--out fig.png] [--csv m.csv]

``run_name`` is the directory under ``proofreader_evolve/runs/`` (a full path also
works). Default output: ``runs/<run>/policy_similarity.png`` (+ ``.csv`` alongside).
The model downloads from https://huggingface.co/sentence-transformers on first use.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# HERE = proofreader_evolve/. This file is in proofreader_evolve/plotting/.
HERE = Path(__file__).resolve().parent.parent
RUNS_DIR = HERE / "runs"


def _resolve_run_dir(run_name: str) -> Path:
    p = Path(run_name)
    if p.is_dir():
        return p
    cand = RUNS_DIR / run_name
    if cand.is_dir():
        return cand
    raise FileNotFoundError(f"run not found: {run_name!r} (looked for {p} and {cand})")


def _gen_dirs(run_dir: Path) -> list[Path]:
    """Every ``gen<NN>/`` under the run, in generation order."""
    gens = [d for d in run_dir.glob("gen*") if d.is_dir()
            and re.fullmatch(r"gen\d+", d.name)]
    return sorted(gens, key=lambda d: int(d.name[3:]))


def _accepted_gens(run_dir: Path) -> set[int]:
    """Generation numbers the gate ACCEPTED, from the ledger (empty if unreadable)."""
    try:
        rows = [json.loads(l) for l in (run_dir / "ledger.jsonl").read_text().splitlines()
                if l.strip()]
        return {int(r["generation"]) for r in rows if r.get("accepted")}
    except Exception:
        return set()


def _policy_text(gen_dir: Path, source: str) -> str | None:
    """The candidate policy TEXT for one generation.

    ``source``:
      * "rules" — rules.candidate.md (the natural-language theory; the default, cleanest
        semantic signal — it is prose the reviser writes to describe the policy).
      * "code"  — heuristics.candidate.py (the executable policy; a sentence model reads
        its docstrings/comments/identifiers, so it captures intent more than syntax).
      * "both"  — rules.md text followed by the code, concatenated.
    Uses the ``.candidate`` files so EVERY generation (accepted or rejected) has one —
    that is the policy that generation actually PROPOSED. Returns None if absent.
    """
    rules = gen_dir / "rules.candidate.md"
    code = gen_dir / "heuristics.candidate.py"
    parts = []
    if source in ("rules", "both") and rules.exists():
        parts.append(rules.read_text())
    if source in ("code", "both") and code.exists():
        parts.append(code.read_text())
    if not parts:
        return None
    return "\n\n".join(parts)


def _chunks(text: str, min_chars: int = 40, max_chars: int = 800) -> list[str]:
    """Split a doc into embeddable chunks (roughly section / paragraph sized).

    Split on blank lines (paragraphs / markdown blocks), drop trivially short chunks,
    and hard-wrap any over-long chunk so no single piece blows past the model's window.
    Keeping chunks paragraph-sized means each is embedded whole rather than truncated.
    """
    raw = re.split(r"\n\s*\n", text)
    out: list[str] = []
    for c in raw:
        c = c.strip()
        if len(c) < min_chars:
            continue
        if len(c) <= max_chars:
            out.append(c)
        else:
            # hard-wrap long blocks into <=max_chars slices on whitespace boundaries
            words, cur = c.split(), ""
            for w in words:
                if len(cur) + len(w) + 1 > max_chars:
                    if cur:
                        out.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                out.append(cur)
    return out or [text.strip()[:max_chars]]


def _drop_common_chunks(doc_chunks: list[list[str]], keep_below: float = 0.6
                        ) -> list[list[str]]:
    """Remove BOILERPLATE: chunks that appear verbatim in >= ``keep_below`` of the docs.

    rules.md / heuristics.py share a large, identical preamble (objective, gate
    description, API contract) across every generation. If those chunks are kept, they
    dominate the mean-pool and drag every doc's vector to the same centroid — the
    signal (the EVOLVING 'Current criteria' / 'Change log' / constants) gets washed out
    and all similarities read ~0.99. Dropping chunks present in a majority of docs
    leaves only the content that actually varies, so the embedding reflects the policy
    DIFFERENCES. A doc left with nothing (rare) falls back to its full chunk set.
    """
    from collections import Counter
    n = len(doc_chunks)
    # Count in how many DOCS each exact chunk text appears (dedupe within a doc first).
    df = Counter()
    for chunks in doc_chunks:
        for ch in set(chunks):
            df[ch] += 1
    thresh = keep_below * n
    kept = []
    for chunks in doc_chunks:
        keep = [ch for ch in chunks if df[ch] < thresh]
        kept.append(keep if keep else chunks)  # never leave a doc empty
    return kept


def embed_docs(texts: list[str], model_name: str, batch_size: int = 64,
               drop_common: bool = True) -> tuple[np.ndarray, list[list[str]]]:
    """Chunk each doc, drop shared boilerplate, then mean-pool into one unit vector.

    Every surviving chunk of every doc is embedded in ONE batched ``model.encode`` call
    (fast); each doc's chunk vectors are mean-pooled and renormalized. Returns
    ``(docs, retained_chunks)`` where ``docs`` is an ``(n_docs, dim)`` array of unit
    vectors (a plain dot product is cosine similarity) and ``retained_chunks[i]`` is the
    exact list of chunks that were embedded for doc ``i`` — i.e. what actually fed the
    model after boilerplate removal, so the input is auditable.
    ``drop_common`` removes chunks shared across a majority of generations (boilerplate)
    so the similarity reflects the EVOLVING policy content, not the shared preamble.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    doc_chunks = [_chunks(t) for t in texts]
    if drop_common:
        doc_chunks = _drop_common_chunks(doc_chunks)
    flat, owner = [], []
    for i, chunks in enumerate(doc_chunks):
        for ch in chunks:
            flat.append(ch)
            owner.append(i)
    # Normalized chunk embeddings (so mean-pool is a spherical average).
    chunk_vecs = model.encode(flat, batch_size=batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
    dim = chunk_vecs.shape[1]
    docs = np.zeros((len(texts), dim), dtype=np.float32)
    owner = np.asarray(owner)
    for i in range(len(texts)):
        m = owner == i
        v = chunk_vecs[m].mean(axis=0) if m.any() else np.zeros(dim)
        n = np.linalg.norm(v)
        docs[i] = v / n if n > 0 else v
    return docs, doc_chunks


def make_figure(gens: list[int], sim: np.ndarray, accepted: set[int],
                run_name: str, source: str, model_name: str, out_path: Path) -> Path:
    """Heatmap + consecutive-similarity + similarity-to-seed/final, one figure."""
    n = len(gens)
    acc_mask = [g in accepted for g in gens]
    # consecutive sim(gen_i, gen_{i-1}); first gen has no predecessor -> NaN
    consec = [float("nan")] + [float(sim[i, i - 1]) for i in range(1, n)]
    to_seed = [float(sim[i, 0]) for i in range(n)]
    # final = last ACCEPTED gen if any, else last gen
    final_idx = max((i for i, a in enumerate(acc_mask) if a), default=n - 1)
    to_final = [float(sim[i, final_idx]) for i in range(n)]

    fig = plt.figure(figsize=(13, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    # --- (left) similarity heatmap ---
    axh = fig.add_subplot(gs[0, 0])
    im = axh.imshow(sim, cmap="viridis", vmin=float(np.min(sim)), vmax=1.0,
                    origin="upper")
    axh.set_xticks(range(n)); axh.set_yticks(range(n))
    axh.set_xticklabels(gens, fontsize=6, rotation=90)
    axh.set_yticklabels(gens, fontsize=6)
    axh.set_xlabel("generation"); axh.set_ylabel("generation")
    axh.set_title("policy text cosine similarity (gen × gen)")
    # mark accepted gens on both axes
    for i, a in enumerate(acc_mask):
        if a:
            axh.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, n, fill=False,
                                        edgecolor="#2ca02c", lw=0.8, alpha=0.5))
    cbar = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity", fontsize=8)

    # --- (right) trajectory lines ---
    axl = fig.add_subplot(gs[0, 1])
    x = np.arange(n)
    axl.plot(x, consec, "-o", color="#d62728", ms=4, lw=1.3,
             label="consecutive: sim(gen, gen−1)  [revision magnitude]")
    axl.plot(x, to_seed, "-s", color="#1f77b4", ms=3, lw=1.1,
             label="similarity to seed (gen 1)")
    axl.plot(x, to_final, "-^", color="#2ca02c", ms=3, lw=1.1,
             label=f"similarity to final ({gens[final_idx]})")
    # accepted-gen vertical markers
    for i, a in enumerate(acc_mask):
        if a:
            axl.axvline(i, color="#2ca02c", lw=0.8, ls=":", alpha=0.6)
    axl.set_xticks(x); axl.set_xticklabels(gens, fontsize=6, rotation=90)
    axl.set_xlabel("generation"); axl.set_ylabel("cosine similarity")
    axl.set_ylim(min(0.5, np.nanmin(consec + to_seed) - 0.02), 1.005)
    axl.grid(True, alpha=0.3)
    axl.set_title("policy drift across generations\n(low consecutive = big rewrite; "
                  "green dotted = accepted)", fontsize=9)
    axl.legend(fontsize=7, loc="lower right", framealpha=0.9)

    fig.suptitle(f"Run {run_name} — policy similarity across {n} generations "
                 f"(source={source}, model={model_name})", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_csv(gens: list[int], sim: np.ndarray, path: Path) -> Path:
    """Full similarity matrix as CSV (row/col header = generation number)."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen"] + list(gens))
        for i, g in enumerate(gens):
            w.writerow([g] + [f"{sim[i, j]:.4f}" for j in range(len(gens))])
    return path


def dump_retained_chunks(gens: list[int], retained: list[list[str]], source: str,
                         model_name: str, drop_common: bool, out_path: Path) -> Path:
    """Write the EXACT text embedded for each generation (post-boilerplate-drop).

    Makes the similarity input fully auditable: for each generation, the chunks that
    actually fed the model, in order. This is what the cosine numbers are computed over
    — a reader can see precisely which policy content drove each gen's vector.
    """
    lines = [
        f"# Policy-similarity input — retained chunks per generation",
        "",
        f"- source: `{source}`  ·  model: `{model_name}`  ·  "
        f"boilerplate dropped: {drop_common}",
        f"- Each section below is the EXACT text embedded for that generation, AFTER "
        f"chunking and (by default) removing chunks shared across a majority of "
        f"generations. The cosine similarities are computed over these chunks only.",
        "",
    ]
    for g, chunks in zip(gens, retained):
        total = sum(len(c) for c in chunks)
        lines.append(f"\n## gen{g:02d} — {len(chunks)} chunk(s), {total} chars embedded\n")
        for j, ch in enumerate(chunks):
            lines.append(f"### gen{g:02d} · chunk {j}")
            lines.append("```")
            lines.append(ch)
            lines.append("```")
            lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def analyze(run_dir: Path, source: str, model_name: str, drop_common: bool = True):
    """Return (gens, similarity_matrix, accepted_set, retained_chunks) for a run."""
    gen_dirs = _gen_dirs(run_dir)
    gens, texts = [], []
    for gd in gen_dirs:
        t = _policy_text(gd, source)
        if t is not None:
            gens.append(int(gd.name[3:]))
            texts.append(t)
    if len(gens) < 2:
        raise SystemExit(f"need >=2 generations with policy text (found {len(gens)}).")
    docs, retained = embed_docs(texts, model_name, drop_common=drop_common)
    sim = docs @ docs.T                      # unit vectors -> dot product = cosine
    sim = np.clip(sim, -1.0, 1.0)
    return gens, sim, _accepted_gens(run_dir), retained


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_name", help="run dir under proofreader_evolve/runs/ (or a path)")
    ap.add_argument("--source", default="rules", choices=["rules", "code", "both"],
                    help="what to embed per generation (default: rules = rules.candidate.md)")
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="sentence-transformers model (default: all-MiniLM-L6-v2, fast/384d; "
                         "try all-mpnet-base-v2 for higher quality/768d)")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: runs/<run>/policy_similarity.png)")
    ap.add_argument("--csv", default=None,
                    help="also write the similarity matrix CSV (default: alongside PNG)")
    ap.add_argument("--keep-boilerplate", action="store_true",
                    help="do NOT drop chunks shared across a majority of generations. "
                         "By default the shared preamble is removed so similarity "
                         "reflects the EVOLVING policy content; keeping it makes every "
                         "generation look ~identical (~0.99) because the boilerplate "
                         "dominates the mean-pool.")
    args = ap.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_name)
    drop_common = not args.keep_boilerplate
    gens, sim, accepted, retained = analyze(run_dir, args.source, args.model,
                                            drop_common=drop_common)

    out_path = Path(args.out) if args.out else (run_dir / "policy_similarity.png")
    make_figure(gens, sim, accepted, run_dir.name, args.source, args.model, out_path)
    csv_path = Path(args.csv) if args.csv else out_path.with_suffix(".csv")
    write_csv(gens, sim, csv_path)
    # Dump the exact embedded text per generation (auditable input) next to the figure.
    chunks_path = out_path.with_name(out_path.stem + "_input_chunks.md")
    dump_retained_chunks(gens, retained, args.source, args.model, drop_common, chunks_path)

    # Terse stdout summary: the biggest revision (lowest consecutive sim) + mean drift.
    consec = [(gens[i], float(sim[i, i - 1])) for i in range(1, len(gens))]
    if consec:
        bg, bs = min(consec, key=lambda t: t[1])
        mean_consec = sum(s for _, s in consec) / len(consec)
        print(f"[policy_similarity] {len(gens)} gens ({len(accepted)} accepted), "
              f"source={args.source}, model={args.model}")
        print(f"[policy_similarity] mean consecutive similarity = {mean_consec:.3f} "
              f"(1.0 = identical); biggest rewrite at gen {bg} (sim {bs:.3f} vs prev)")
    print(f"[policy_similarity] figure -> {out_path}")
    print(f"[policy_similarity] matrix -> {csv_path}")
    print(f"[policy_similarity] input chunks -> {chunks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
