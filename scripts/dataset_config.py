"""
Dataset configuration helpers shared across notebooks/scripts.

Resolves per-brain paths from the files in ``configs/``:
- ``segmentation_datasets.rtf`` -> brain_id -> segmentation_id
- ``exaspim_image_prefixes.json`` -> brain_id -> image prefix

These were originally inlined in ``load_skeletons.ipynb``; they live here so
other notebooks can reuse them via::

    import sys; sys.path.insert(0, "../scripts")
    from dataset_config import get_segmentation_id, get_img_path
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


def get_segmentation_id(brain_id, rtf_path: str = DEFAULT_RTF_PATH) -> str:
    """Look up the segmentation id for a brain from the segmentation_datasets RTF."""
    df = load_segmentation_datasets(rtf_path)
    brain_id = str(brain_id)
    if brain_id not in df.index:
        raise KeyError(
            f"brain_id {brain_id} not found in {rtf_path}; "
            f"available: {sorted(df.index)}"
        )
    return df.loc[brain_id, "segmentation_id"]


def get_img_path(brain_id, prefixes_path: str = DEFAULT_IMAGE_PREFIXES_PATH) -> str:
    """Look up the ExaSPIM image path (resolution level 0) for a brain."""
    with open(prefixes_path) as f:
        prefixes = json.load(f)
    # JSON object keys are strings; accept an int/str brain_id either way.
    img_prefix = prefixes.get(str(brain_id))
    if img_prefix is None:
        img_prefix = prefixes[int(brain_id)]  # raises KeyError if truly absent
    return os.path.join(img_prefix, "0")
