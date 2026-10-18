"""
Dataset configuration helpers shared across notebooks/scripts.

Resolves per-brain paths from the files in ``configs/``:
- ``segmentation_datasets.rtf`` -> brain_id -> segmentation_id
- ``exaspim_image_prefixes.json`` -> brain_id -> image prefix
- ``segmentation_paths.json`` -> brain_id -> explicit segmentation/fragment
  URIs for datasets outside the legacy GCS layout (takes precedence over the RTF)
- ``image_alignment.json`` -> brain_id -> the image URI reviewed for alignment; image
  reads go through ``get_reviewed_img_path`` / ``get_cache_img_path``, not a cache's
  stored ``img_path``, which goes stale when S3 volumes are deleted

These were originally inlined in ``load_skeletons.py``; they live here so
other notebooks can reuse them via::

    import sys; sys.path.insert(0, "../scripts")
    from dataset_config import get_segmentation_id, get_img_path
    from dataset_config import get_segmentation_path, get_fragments_path
"""

from __future__ import annotations

import json
import os
import re

import pandas as pd

# Default config locations, relative to a notebook running in ``notebooks/`` or
# a script running in ``scripts/`` (both are one level below the project root).
DEFAULT_RTF_PATH = "../configs/segmentation_datasets.rtf"
DEFAULT_IMAGE_PREFIXES_PATH = "../configs/exaspim_image_prefixes.json"
# Explicit paths are resolved from this file's location, so callers that only
# pass ``rtf_path`` still see them.
DEFAULT_SEGMENTATION_PATHS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "segmentation_paths.json"
)
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
DEFAULT_IMAGE_ALIGNMENT_PATH = os.path.join(_CONFIG_DIR, "image_alignment.json")
_IMAGE_PREFIXES_ABSOLUTE = os.path.join(_CONFIG_DIR, "exaspim_image_prefixes.json")
_IDENTITY_4X4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
_CACHE_BRAIN = re.compile(r"dataset_cache_(\d+)_mcl")
LEGACY_SEGMENTATION_TEMPLATE = (
    "gs://allen-nd-goog/from_google/{brain_id}/whole_brain/{segmentation_id}/"
)


def load_segmentation_datasets(rtf_path: str = DEFAULT_RTF_PATH) -> pd.DataFrame:
    """Parse the segmentation_datasets RTF into a brain_id -> segmentation_id table.

    The file lists one ``"<brain_id>, <segmentation_id>"`` entry per line (some
    segmentation ids are wrapped in an RTF hyperlink). We strip the RTF markup
    and return a DataFrame indexed by the 6-digit brain id (as a string).
    """
    with open(rtf_path) as f:
        rtf = f.read()
    # Drop hyperlink instruction groups (keep only the visible link text).
    text = re.sub(r"\{\\\*\\fldinst\{HYPERLINK[^}]*\}\}", "", rtf)
    # Strip RTF control words and group braces / escapes.
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("\\", " ")
    # Each entry: "<6-digit brain_id>, <segmentation_id token>".
    rows = re.findall(r"(\d{6})\s*,\s*([A-Za-z0-9_.]+)", text)
    df = pd.DataFrame(rows, columns=["brain_id", "segmentation_id"])
    return df.drop_duplicates("brain_id").set_index("brain_id")


def load_segmentation_paths(paths_path: str = DEFAULT_SEGMENTATION_PATHS) -> dict:
    """brain_id -> {"segmentation_id", "segmentation", "fragments"} explicit entries."""
    if not os.path.exists(paths_path):
        return {}
    with open(paths_path) as f:
        entries = json.load(f)
    return {k: v for k, v in entries.items() if not k.startswith("_")}


def get_segmentation_id(brain_id, rtf_path: str = DEFAULT_RTF_PATH,
                        paths_path: str = DEFAULT_SEGMENTATION_PATHS) -> str:
    """Look up the segmentation id for a brain (explicit paths file, then the RTF)."""
    explicit = load_segmentation_paths(paths_path).get(str(brain_id))
    if explicit is not None:
        return explicit["segmentation_id"]
    df = load_segmentation_datasets(rtf_path)
    brain_id = str(brain_id)
    if brain_id not in df.index:
        raise KeyError(
            f"brain_id {brain_id} not found in {rtf_path}; "
            f"available: {sorted(df.index)}"
        )
    return df.loc[brain_id, "segmentation_id"]


def get_segmentation_path(brain_id, rtf_path: str = DEFAULT_RTF_PATH,
                          paths_path: str = DEFAULT_SEGMENTATION_PATHS) -> str:
    """Dense segmentation volume (neuroglancer precomputed root, ends with '/')."""
    explicit = load_segmentation_paths(paths_path).get(str(brain_id))
    if explicit is not None:
        return explicit["segmentation"].rstrip("/") + "/"
    segmentation_id = get_segmentation_id(brain_id, rtf_path, paths_path)
    return LEGACY_SEGMENTATION_TEMPLATE.format(
        brain_id=brain_id, segmentation_id=segmentation_id
    )


def get_fragments_path(brain_id, rtf_path: str = DEFAULT_RTF_PATH,
                       paths_path: str = DEFAULT_SEGMENTATION_PATHS) -> str:
    """Fragment SWC directory (zip shards). Not always ``<segmentation>/swcs``."""
    explicit = load_segmentation_paths(paths_path).get(str(brain_id))
    if explicit is not None:
        return explicit["fragments"].rstrip("/")
    return get_segmentation_path(brain_id, rtf_path, paths_path) + "swcs"


def get_img_path(brain_id, prefixes_path: str = DEFAULT_IMAGE_PREFIXES_PATH) -> str:
    """Look up the ExaSPIM image path (resolution level 0) for a brain."""
    with open(prefixes_path) as f:
        prefixes = json.load(f)
    # JSON object keys are strings; accept an int/str brain_id either way.
    img_prefix = prefixes.get(str(brain_id))
    if img_prefix is None:
        img_prefix = prefixes[int(brain_id)]  # raises KeyError if truly absent
    return os.path.join(img_prefix, "0")


def get_reviewed_img_path(brain_id, fallback: str | None = None,
                          alignment_path: str = DEFAULT_IMAGE_ALIGNMENT_PATH) -> str:
    """Image path (resolution level 0) to read patches from for a brain.

    The URI reviewed in ``configs/image_alignment.json`` comes first: it is what the
    context cache and the companions read, and it stays valid when a cache's stored
    ``img_path`` points at a deleted S3 volume (794491's 2025-10-26 volume). Without a
    reviewed entry, ``fallback`` (typically the cache's ``img_path``) is returned, else
    the ``exaspim_image_prefixes.json`` entry. A review that needs a non-identity
    ``graph_to_world_um`` raises: an untransformed patch read would be misplaced.
    """
    review = {}
    if os.path.exists(alignment_path):
        with open(alignment_path) as f:
            review = json.load(f).get("brains", {}).get(str(brain_id), {})
    uri = review.get("image_uri") if review.get("status") == "reviewed" else None
    if uri is None:
        return fallback if fallback is not None else get_img_path(brain_id, _IMAGE_PREFIXES_ABSOLUTE)
    transform = review.get("graph_to_world_um")
    if transform is not None and [[float(v) for v in row] for row in transform] != _IDENTITY_4X4:
        raise ValueError(f"Brain {brain_id}: the reviewed image needs a graph_to_world_um registration; "
                         "read it through proofreader_evolve.harness.image_context instead")
    return uri.rstrip("/") + "/0"


def get_cache_img_path(cache_path, payload: dict, alignment_path: str = DEFAULT_IMAGE_ALIGNMENT_PATH) -> str:
    """``get_reviewed_img_path`` for a ``dataset_cache_<brain>_mcl<N>*.pkl`` file, falling
    back to the payload's ``img_path``; the brain id comes from the file name."""
    match = _CACHE_BRAIN.search(os.path.basename(str(cache_path)))
    if match is None:
        return payload["img_path"]
    return get_reviewed_img_path(match.group(1), fallback=payload.get("img_path"), alignment_path=alignment_path)
