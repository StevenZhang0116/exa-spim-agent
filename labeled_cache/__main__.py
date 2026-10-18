"""Command line of the labeled-cache builder.

    python -m labeled_cache --brains 794495 --mcl 100 --policy length_v1 --check-reproduce
    python -m labeled_cache --brains 718162 794492                 # new brains: base + labels
    python -m labeled_cache --brains 794495 --dry-run              # show the stages, load nothing
    python -m labeled_cache --brains 794491 --refresh-image-path   # only replace a stale img_path

Run in the panda environment on a compute node, one process per brain (each brain
needs more than 20 GB). Console output is also appended to
``notebooks/log/labeled_cache_<timestamp>_<pid>.log`` (or ``--log-txt``).
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import time
import traceback

from labeled_cache import io
from labeled_cache.build import BuildRequest, plan, run
from labeled_cache.policy import DEFAULT_LENGTH_UM, POLICIES, POLICY_LENGTH_V1, describe


class _Tee:
    def __init__(self, console, log_file):
        self.console, self.log_file = console, log_file

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m labeled_cache", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", required=True,
                        help="Brain ids from configs/segmentation_paths.json or segmentation_datasets.rtf.")
    parser.add_argument("--mcl", type=int, default=100, help="Minimum cable length (default %(default)s).")
    parser.add_argument("--policy", choices=POLICIES, default=POLICY_LENGTH_V1,
                        help="Merge-threshold policy (default %(default)s).")
    parser.add_argument("--threshold-um", type=float, default=DEFAULT_LENGTH_UM,
                        help="Cable length of the length_v1 policy (default %(default)s).")
    parser.add_argument("--cache-dir", type=Path, default=io.DEFAULT_CACHE_DIR)
    parser.add_argument("--node-spacing", type=float, default=5.0)
    parser.add_argument("--anisotropy", type=float, nargs=3, default=(0.748, 0.748, 1.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--no-verify-gt", action="store_true",
                        help="Do not compare a reused cache's GT with the current GCS tracings.")
    parser.add_argument("--gt-workers", type=int, default=4, help="Processes for the GT comparison load.")
    parser.add_argument("--rebuild-base", action="store_true",
                        help="Delete both pkl files and rebuild from GCS (also when the GT check finds them stale).")
    parser.add_argument("--force-relabel", action="store_true",
                        help="Apply the policy even when the cache already records it.")
    parser.add_argument("--check-reproduce", action="store_true",
                        help="Before relabeling, recompute with the original thresholds and require the stored labels back.")
    parser.add_argument("--refresh-image-path", action="store_true",
                        help="Only give the existing pkl files the img_path of configs/exaspim_image_prefixes.json "
                             "(after an S3 volume moved); no GT check, no other stage.")
    parser.add_argument("--dry-run", action="store_true", help="Print the stages per brain and exit.")
    parser.add_argument("--quiet", action="store_true", help="No per-step progress bars.")
    parser.add_argument("--log-txt", type=Path, help="Append console output to this file.")
    args = parser.parse_args(argv)
    if args.mcl < 0 or args.threshold_um <= 0 or args.node_spacing <= 0 or any(a <= 0 for a in args.anisotropy):
        parser.error("--mcl must be nonnegative; --threshold-um, --node-spacing and --anisotropy must be positive")
    if args.gt_workers < 1:
        parser.error("--gt-workers must be at least 1")
    if args.refresh_image_path and (args.rebuild_base or args.force_relabel or args.check_reproduce):
        parser.error("--refresh-image-path runs alone; drop --rebuild-base, --force-relabel and --check-reproduce")
    return args


def requests_from(args):
    for brain in dict.fromkeys(args.brains):
        yield BuildRequest(brain=str(brain), mcl=args.mcl, cache_dir=args.cache_dir.expanduser().resolve(),
                           policy=args.policy, length_um=args.threshold_um, node_spacing=args.node_spacing,
                           anisotropy=tuple(args.anisotropy), verify_gt=not args.no_verify_gt,
                           gt_workers=args.gt_workers, rebuild_base=args.rebuild_base,
                           force_relabel=args.force_relabel, check_reproduce=args.check_reproduce,
                           refresh_image_path=args.refresh_image_path, verbose=not args.quiet)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(f"policy {describe(args.policy, args.threshold_um)}; cache dir {args.cache_dir}")
        for request in requests_from(args):
            print(f"  {request.brain} mcl{request.mcl}: {' -> '.join(plan(request))}"
                  + ("" if request.verify_gt or request.refresh_image_path else " (GT check off)"))
        return 0
    log_path = args.log_txt or io.DEFAULT_LOG_DIR / f"labeled_cache_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log"
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    returncode = 1
    with log_path.open("a", encoding="utf-8") as log_file, \
            redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(_Tee(sys.stderr, log_file)):
        print(f"\n# Started: {datetime.now().astimezone().isoformat(timespec='seconds')}")
        print(f"# Command: {shlex.join([sys.executable, '-m', 'labeled_cache', *sys.argv[1:]])}")
        print(f"# Host: {platform.node()}  PID: {os.getpid()}  Python: {platform.python_version()}")
        print(f"# Log: {log_path}")
        summaries = []
        try:
            for request in requests_from(args):
                summary = run(request, log=lambda message: print(message, flush=True))
                summaries.append(summary)
                print(json.dumps(summary, indent=1, default=str), flush=True)
            returncode = 0
        except KeyboardInterrupt:
            returncode = 130
            print("Interrupted; no further brains will be started.", file=sys.stderr)
        except Exception:
            traceback.print_exc()
            print("Stopping at the first failure; later brains were not touched.", file=sys.stderr)
        finally:
            outcome = "OK" if returncode == 0 else "FAILED"
            print(f"\n# {outcome} (exit code {returncode}) after {time.monotonic() - started:.1f}s; "
                  f"{len(summaries)} brain(s) completed")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
