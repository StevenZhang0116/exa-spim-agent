"""Trusted labeled-cache loading for native candidate preparation.

No candidate enumeration, graph edits, image loading or within-brain splitting.
Run in panda so pickled SkeletonGraph classes and binary dependencies match.
"""

import os
import pickle


def default_cache_path(brain_id: str, min_cable_length: int = 100) -> str:
    """Path to the BrainDataset cache for a brain, relative to the project root."""
    here = os.path.dirname(__file__)
    return os.path.abspath(
        os.path.join(
            here, "..", "..", "cache",
            f"dataset_cache_{brain_id}_mcl{min_cable_length}_add.pkl",
        )
    )


def load_cached_graphs(cache_path: str, expect_brain=None, expect_mcl=None):
    """Load just the two SkeletonGraphs from a BrainDataset cache pickle.

    We read the pickle payload directly (rather than via BrainDataset.load_from_cache)
    so this module has no dependency on the TensorStore image reader — the
    native preparation process only needs the cached graphs and annotations.

    Content validation (opt-in): the path follows the
    ``dataset_cache_<brain>_mcl<mcl>.pkl`` filename CONVENTION, but nothing
    guarantees the file's CONTENTS match that name — a wrong-but-existing pickle (a
    renamed/stale build, a swapped brain) would otherwise load silently and the run
    would train/gate on the wrong data. When ``expect_brain`` / ``expect_mcl`` are
    given we cross-check them against the payload's own metadata and FAIL FAST:
      * ``expect_mcl`` vs the payload's ``min_cable_length`` (stored at build time);
      * ``expect_brain`` vs the brain id embedded in the payload's
        ``fragments_path`` / ``gt_path`` / ``img_path`` (e.g. ``/789202/``).
    Either is skipped if the corresponding metadata is absent (older caches), so the
    check only ever tightens, never breaks, an otherwise-valid load.
    """
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    # --- mcl content check: payload's own min_cable_length must match the request.
    if expect_mcl is not None and payload.get("min_cable_length") is not None:
        got_mcl = payload["min_cable_length"]
        if int(got_mcl) != int(expect_mcl):
            raise SystemExit(
                f"cache mcl mismatch: {cache_path} was built with "
                f"min_cable_length={got_mcl}, but the run requested mcl={expect_mcl}. "
                f"The filename convention (dataset_cache_<brain>_mcl<mcl>.pkl) and the "
                f"file CONTENTS disagree — refusing to enumerate candidates over the "
                f"wrong cache. Load the matching cache or rebuild it."
            )

    # --- brain content check: the brain id embedded in the source paths must match.
    if expect_brain is not None:
        eb = str(expect_brain)
        srcs = [payload.get(k) for k in ("fragments_path", "gt_path", "img_path")]
        srcs = [str(s) for s in srcs if s]
        # Only assert when at least one source path is present AND some path actually
        # encodes a brain id, so older/atypical caches don't trip a false alarm.
        if srcs and not any(f"/{eb}/" in s or f"_{eb}_" in s for s in srcs):
            raise SystemExit(
                f"cache brain mismatch: {cache_path} requested brain={eb}, but none "
                f"of the payload's source paths reference it "
                f"({'; '.join(srcs) or 'no source paths in payload'}). The filename "
                f"says brain {eb} but the contents look like a different brain — "
                f"refusing to train/gate on mismatched data."
            )

    return payload["fragments_graph"], payload["gt_graph"], payload
