#!/usr/bin/env python3
"""Build missing skeleton caches and bake segmentation-derived GT error labels.

Run in the panda environment on a compute node. By default, process brains
709221, 718162, and 794492 with MCL 100 and node spacing 5. Existing _add.pkl
files are skipped; compatible base .pkl files are reused. Each completed cache
is saved atomically, so a failed labeling step can resume from the base cache.
Console output is also saved to notebooks/log/generate_add_caches_<timestamp>_<pid>.log.
Use --log-txt to append to a custom log file instead.

Examples (from the repository root):
    python -u notebooks/generate_add_caches.py
    python -u notebooks/generate_add_caches.py --brains 709221 718162

Paths to configs and the default cache directory are relative to this script,
independent of the working directory. Cloud reads use the existing GCS
credentials environment variable, or configs/allen-nd-goog-f5d46dbfa2cd.json by default.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import gc
import math
import os
import platform
import shlex
import sys
import time
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_BRAINS = ("718162", "794492")


class _Tee:
    """Mirror Python console output and flush the log after every write."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, data):
        self.log_file.write(data)
        self.log_file.flush()
        self.console.write(data)
        self.console.flush()
        return len(data)

    def flush(self):
        self.log_file.flush()
        self.console.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.console, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--brains", nargs="+", default=DEFAULT_BRAINS,
        help="Brain IDs from configs/segmentation_paths.json or segmentation_datasets.rtf (default: %(default)s).",
    )
    parser.add_argument(
        "--mcl", type=int, default=100,
        help="Integer minimum cable length (default: %(default)s).",
    )
    parser.add_argument("--node-spacing", type=float, default=5.0)
    parser.add_argument(
        "--anisotropy", type=float, nargs=3, default=(0.748, 0.748, 1.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / "cache",
        help="Output directory (default: the repository's cache/ directory).",
    )
    parser.add_argument(
        "--log-txt", type=Path,
        help="Append console output to this file (default: a new file in notebooks/log/).",
    )
    args = parser.parse_args()
    if args.mcl < 0:
        parser.error("--mcl must be nonnegative")
    for name, values in (
        ("--node-spacing", [args.node_spacing]),
        ("--anisotropy", args.anisotropy),
    ):
        if any(not math.isfinite(value) or value <= 0 for value in values):
            parser.error(f"{name} must contain finite positive values")
    return args


def validate_base_cache(dataset, expected: dict, path: Path) -> None:
    """Avoid reusing skeletons built from different sources or parameters."""
    for name, value in expected.items():
        actual = getattr(dataset, name)
        if name == "anisotropy":
            actual = tuple(actual)
        if actual != value:
            raise ValueError(
                f"Cannot reuse {path}: {name}={actual!r}, expected {value!r}. "
                "Use matching parameters or a different --cache-dir."
            )


def generate_caches(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(CONFIG_DIR / "allen-nd-goog-f5d46dbfa2cd.json"),
    )
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    print(f"GCS credentials: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}", flush=True)

    from dataset_config import get_img_path
    from relabel_cache import atomic_cache_write, fragments_path_for, segmentation_path_for
    from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

    cache = args.cache_dir.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache}", flush=True)

    for brain in dict.fromkeys(args.brains):
        base = cache / f"dataset_cache_{brain}_mcl{args.mcl}.pkl"
        labeled = cache / f"dataset_cache_{brain}_mcl{args.mcl}_add.pkl"
        if labeled.exists():
            print(f"Skip existing: {labeled}", flush=True)
            continue

        print(f"\nProcessing brain {brain}", flush=True)
        segmentation = segmentation_path_for(brain)
        metadata = dict(
            fragments_path=fragments_path_for(brain),
            gt_path=f"gs://allen-nd-goog/ground_truth_tracings/{brain}/voxel",
            img_path=get_img_path(
                brain, prefixes_path=str(CONFIG_DIR / "exaspim_image_prefixes.json"),
            ),
            anisotropy=tuple(args.anisotropy),
            min_cable_length=args.mcl,
            node_spacing=args.node_spacing,
        )
        print(f"Segmentation: {segmentation}", flush=True)
        print(f"Fragments: {metadata['fragments_path']}", flush=True)

        if base.exists():
            print(f"Loading existing base cache: {base}", flush=True)
            dataset = BrainDataset.load_from_cache(str(base))
            validate_base_cache(dataset, metadata, base)
            # Legacy loaders saved 100.0; feature extraction requires integer 100.
            # Validation above ensures the value matches before normalizing it.
            dataset.min_cable_length = args.mcl
        else:
            print("Loading fragment and GT skeletons...", flush=True)
            dataset = BrainDataset(**metadata)
            atomic_cache_write(base, dataset.save)
            print(f"Saved: {base}", flush=True)

        print("Labeling GT from segmentation...", flush=True)
        dataset.label_gt_from_segmentation(segmentation)
        atomic_cache_write(labeled, dataset.save)
        print(f"Saved: {labeled}", flush=True)
        del dataset
        gc.collect()

    print("Finished processing requested brains.", flush=True)


def main() -> int:
    args = parse_args()
    log_path = args.log_txt or (
        PROJECT_ROOT / "notebooks" / "log"
        / f"generate_add_caches_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log"
    )
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    returncode = 1
    with log_path.open("a", encoding="utf-8") as log_file, \
            redirect_stdout(_Tee(sys.stdout, log_file)), \
            redirect_stderr(_Tee(sys.stderr, log_file)):
        print(f"\n# Started: {datetime.now().astimezone().isoformat(timespec='seconds')}")
        print(f"# Command: {shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])}")
        print(f"# Host: {platform.node()}  PID: {os.getpid()}  Python: {platform.python_version()}")
        print(f"# Brains: {', '.join(args.brains)}  MCL: {args.mcl}  "
              f"Node spacing: {args.node_spacing}  Anisotropy: {tuple(args.anisotropy)}")
        print(f"Log: {log_path}")
        try:
            generate_caches(args)
            returncode = 0
        except KeyboardInterrupt:
            returncode = 130
            traceback.print_exc()
            print("Interrupted; no further brains will be started.", file=sys.stderr)
        except Exception:
            traceback.print_exc()
        finally:
            outcome = "OK" if returncode == 0 else "FAILED"
            print(f"\n# {outcome} (exit code {returncode}) after {time.monotonic() - started:.1f}s")
            print(f"# Ended: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
