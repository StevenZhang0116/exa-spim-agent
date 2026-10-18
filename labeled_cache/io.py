"""File helpers of the labeled-cache builder: names, signatures, atomic writes, source paths."""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_RTF = CONFIG_DIR / "segmentation_datasets.rtf"
IMAGE_PREFIXES = CONFIG_DIR / "exaspim_image_prefixes.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_LOG_DIR = PROJECT_ROOT / "notebooks" / "log"
GCS_CREDENTIALS = CONFIG_DIR / "allen-nd-goog-f5d46dbfa2cd.json"
GT_PATH_TEMPLATE = "gs://allen-nd-goog/ground_truth_tracings/{brain}/voxel"

_CACHE_NAME = re.compile(r"dataset_cache_(\d+)_mcl(\d+)(_add)?\.pkl$")


def base_cache_path(cache_dir, brain, mcl) -> Path:
    return Path(cache_dir) / f"dataset_cache_{brain}_mcl{int(mcl)}.pkl"


def labeled_cache_path(cache_dir, brain, mcl) -> Path:
    return Path(cache_dir) / f"dataset_cache_{brain}_mcl{int(mcl)}_add.pkl"


def parse_cache_name(path):
    """``(brain, mcl, labeled)`` of a cache file name, or None."""
    match = _CACHE_NAME.search(Path(path).name)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), match.group(3) is not None


def cache_signature(path):
    """``(size, mtime_ns, inode)``: detects a file replaced while it was being rewritten."""
    info = Path(path).stat()
    return info.st_size, info.st_mtime_ns, info.st_ino


def atomic_cache_write(path, writer, expected_signature=None):
    """Replace a cache only after a complete write; preserve it on failure.

    ``writer(temporary_path)`` writes the new file next to ``path``; the file is
    fsynced, then moved over ``path`` in one rename. With ``expected_signature``
    (from ``cache_signature``) the replacement is refused when ``path`` changed
    since it was read.
    """
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if expected_signature is not None and cache_signature(path) != expected_signature:
            raise RuntimeError("Cache changed during regeneration; refusing to replace it")
        if path.exists():
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def set_cloud_environment():
    """GCS credentials and the S3 metadata guard, unless the caller set them."""
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(GCS_CREDENTIALS))
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")


def image_path(brain):
    """Level-0 fused image of one brain, from ``configs/exaspim_image_prefixes.json``."""
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from dataset_config import get_img_path  # noqa: E402

    return get_img_path(brain, prefixes_path=str(IMAGE_PREFIXES))


def dataset_sources(brain):
    """Source paths of one brain: fragment SWCs, GT SWCs, fused image, dense segmentation.

    Resolved through ``scripts/dataset_config.py`` (``configs/segmentation_paths.json``
    first, then the legacy RTF table); the GT path is the fixed GCS layout.
    """
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from dataset_config import get_fragments_path, get_segmentation_path  # noqa: E402

    return {
        "fragments_path": get_fragments_path(brain, rtf_path=str(CONFIG_RTF)),
        "gt_path": GT_PATH_TEMPLATE.format(brain=brain),
        "img_path": image_path(brain),
        "segmentation_path": get_segmentation_path(brain, rtf_path=str(CONFIG_RTF)),
    }
