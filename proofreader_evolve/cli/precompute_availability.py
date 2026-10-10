"""Write the label-availability sidecar for every detector-native table (required by evolution).

Run with panda on an allocated compute node, for example n268:
    python -u -m proofreader_evolve.cli.precompute_availability --brains 794495 802449 789202 794493 794491

Launcher mode runs one brain per fresh subprocess (single-brain mode, `--brain`), so each
`_add.pkl` is released before the next. For each brain the existing merge and split tables
are loaded and validated exactly as evolution loads them, availability is computed from the
cache's GT node labels (and GT node coordinates for the merge `near_gt_150` definition), every
binding is checked (row identity, labels, source cache, positives available) and the sidecar
is written beside the table directory (see harness/label_availability.py). Tables, caches and
context caches are only read. Existing valid sidecars are reused unless `--force`.
No LLM is called.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import os
from pathlib import Path
import pickle
import platform
import shlex
import socket
import subprocess
import sys
import time
import traceback

from . import precompute_error_scores as pc
from ..harness import label_availability as la
from ..harness.native_pool import cache_path, ensure_native_tables

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", help="Launcher mode: one subprocess per brain")
    parser.add_argument("--brain", help="Single-brain mode (used by the launcher)")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--feature-tables-dir", type=Path, default=pc.DEFAULT_OUT)
    parser.add_argument("--merge-dir", action="append")
    parser.add_argument("--split-dir", action="append")
    parser.add_argument("--force", action="store_true", help="Rewrite valid existing sidecars")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-txt", type=Path)
    args = parser.parse_args(argv)
    if bool(args.brains) == bool(args.brain):
        parser.error("Give exactly one of --brains (launcher) or --brain (single brain)")
    return args


def single(args):
    brain = str(args.brain)
    selected = pc.resolve_detector_runs(args.merge_dir, args.split_dir)
    source = cache_path(brain, args.mcl)
    bank = ensure_native_tables(brain, source, selected, args.feature_tables_dir, prepare=False, mcl=args.mcl)
    pending = {}
    for kind, table in bank.tables.items():
        table_dir = bank.meta["table_paths"][kind]
        try:
            existing = la.load(table, table_dir)
        except la.AvailabilityMismatch as exc:
            log(f"{brain}/{kind}: existing sidecar is stale ({exc}); rewriting")
            existing = None
        if existing is not None and not args.force:
            log(f"{brain}/{kind}: valid sidecar reused ({existing['path']})")
            continue
        pending[kind] = (table, table_dir)
    if not pending:
        return 0

    identities = {t.meta["provenance"]["source_cache"]["path"] for t, _ in pending.values()}
    if len(identities) != 1:
        raise RuntimeError(f"{brain}: tables were prepared from different caches {identities}")
    expected = next(iter(pending.values()))[0].meta["provenance"]["source_cache"]
    if pc._cache_identity(source) != expected:
        raise RuntimeError(f"{brain}: {source} changed since the tables were prepared; rebuild the tables first")
    log(f"{brain}: loading GT from {source}")
    with open(source, "rb") as stream:
        payload = pickle.load(stream)
    gt_label = payload["gt_node_canonical_label"]
    gt_xyz = payload["gt_graph"].node_xyz
    del payload
    if pc._cache_identity(source) != expected:
        raise RuntimeError(f"{brain}: {source} changed while it was being read")

    for kind, (table, table_dir) in pending.items():
        arrays = la.compute(kind, table.candidates, gt_label, gt_xyz)
        target = la.write(table_dir, kind, arrays, truth=table.truth, keys=table.keys,
                          pool_sha256=table.meta["pool_sha256"], source_cache=expected,
                          provenance={"host": socket.gethostname(),
                                      "created": datetime.now().astimezone().isoformat(),
                                      "detector": table.meta["provenance"]["detector"]["run_id"]})
        fractions = ", ".join(f"{name}={arrays[name].mean():.4f}" for name in arrays)
        log(f"{brain}/{kind}: {len(table.truth)} rows, {int(table.truth.sum())} positives, "
            f"available {fractions} -> {target}")
    return 0


def launcher(args):
    for brain in dict.fromkeys(args.brains):
        command = [sys.executable, "-u", "-m", "proofreader_evolve.cli.precompute_availability",
                   "--brain", str(brain), "--mcl", str(args.mcl),
                   "--feature-tables-dir", str(args.feature_tables_dir)]
        for flag, values in (("--merge-dir", args.merge_dir), ("--split-dir", args.split_dir)):
            for value in values or []:
                command += [flag, value]
        if args.force:
            command.append("--force")
        print("\n" + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        with subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                code = process.wait()
            except BaseException:
                process.kill()
                process.wait()
                raise
        if code:
            print(f"Stopped: subprocess exit code {code}. Completed sidecars are reusable.", file=sys.stderr)
            return code if code > 0 else 1
    if not args.dry_run:
        print("\nAvailability sidecars complete for all requested brains.", flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.brain:
        return single(args)
    if args.dry_run:
        return launcher(args)
    log_path = (args.log_txt or PROJECT_ROOT / "proofreader_evolve" / "log" /
                f"precompute_availability_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log.txt").expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started, returncode = time.monotonic(), 1
    with log_path.open("a", encoding="utf-8") as log_file, \
            redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(_Tee(sys.stderr, log_file)):
        print(f"\n# argv: {shlex.join([sys.executable, '-m', 'proofreader_evolve.cli.precompute_availability', *(sys.argv[1:] if argv is None else argv)])}")
        print(f"# started {datetime.now():%Y-%m-%d %H:%M:%S}  host {platform.node()}\nLog: {log_path}", flush=True)
        try:
            returncode = launcher(args)
        except Exception:
            traceback.print_exc()
        finally:
            print(f"\n# {'OK' if returncode == 0 else 'FAILED'} (exit code {returncode}) after "
                  f"{time.monotonic() - started:.1f}s", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
