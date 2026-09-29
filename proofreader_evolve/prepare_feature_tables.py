"""Generate evolution-matching detector tables, then run read-only validation.

Run with panda on an allocated compute node (e.g. n244):
    python -u proofreader_evolve/prepare_feature_tables.py

Defaults: brains 789202 and 794491, MCL 100, labeled _add caches, and the exact
native candidate pools of the current default frozen models. Each brain runs in a fresh subprocess to
release memory before the next. Existing matching tables are reused. The first
failure stops the script with a nonzero exit code. No LLM calls or evolution.

Use --dry-run to print commands without importing models or loading any data.
Use --threads all to give numerical libraries all CPUs available to this process
(respecting CPU affinity). Serial detector loops remain serial.
Merge extraction can scan the entire graph; this is not a quick smoke test.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def available_cpu_count():
    """Honor the process CPU affinity, including scheduler CPU binding."""
    try:
        count = len(os.sched_getaffinity(0))
        if count:
            return count
    except (AttributeError, OSError, NotImplementedError):
        pass
    return max(1, os.cpu_count() or 1)


def thread_argument(value):
    if value == "all":
        return value
    try:
        count = int(value)
    except ValueError:
        count = 0
    if count < 1:
        raise argparse.ArgumentTypeError("threads must be a positive integer or 'all'")
    return count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", default=["789202", "794491"])
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--threads", type=thread_argument, default=1, metavar="N|all",
                        help="Numerical-library threads: positive integer or all available CPUs (default: 1)")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "proofreader_evolve" / "feature_tables")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if any(not b.isascii() or not b.isdigit() for b in args.brains):
        parser.error("Brain IDs must contain ASCII digits only")
    if len(set(args.brains)) != len(args.brains):
        parser.error("Brain IDs must not repeat")
    if args.mcl < 0:
        parser.error("MCL must be >= 0")
    args.requested_threads = args.threads
    args.available_cpus = available_cpu_count()
    if args.threads == "all":
        args.threads = args.available_cpus
    args.out_dir = args.out_dir.expanduser().resolve()
    return args


def commands(args):
    """Use existing entry points and their defaults; do not duplicate feature math."""
    result = []
    for brain in args.brains:
        cache = PROJECT_ROOT / "cache" / f"dataset_cache_{brain}_mcl{args.mcl}_add.pkl"
        result.append([sys.executable, "-u", "-m", "proofreader_evolve.cli.precompute_error_scores",
                       "--brain", brain, "--mcl", str(args.mcl), "--pkl", str(cache),
                       "--out-dir", str(args.out_dir)])
    result.append([sys.executable, "-u", "-m", "proofreader_evolve.cli.preflight",
                   "--brains", *args.brains, "--mcl", str(args.mcl),
                   "--feature-tables-dir", str(args.out_dir)])
    return result


def main(argv=None):
    args = parse_args(argv)
    planned = commands(args)
    if not args.dry_run:
        # Check all inputs before doing expensive work on the first brain.
        missing = [cmd[cmd.index("--pkl") + 1] for cmd in planned[:-1]
                   if not Path(cmd[cmd.index("--pkl") + 1]).is_file()]
        if missing:
            print("Missing source cache(s):\n" + "\n".join(missing), file=sys.stderr)
            return 1
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        env[name] = str(args.threads)
    env["JOBLIB_MULTIPROCESSING"] = "0"
    print(f"Project: {PROJECT_ROOT}\nOutput: {args.out_dir}\n"
          f"Threads: {args.threads} (requested: {args.requested_threads}; "
          f"available CPUs: {args.available_cpus})", flush=True)
    for command in planned:
        print("\n" + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Stopped: subprocess exit code {exc.returncode}. Completed matching tables are reusable.",
                  file=sys.stderr)
            return exc.returncode if exc.returncode > 0 else 1
        except KeyboardInterrupt:
            print("Interrupted; no further brains will be started.", file=sys.stderr)
            return 130
    if not args.dry_run:
        print("\nAll requested tables generated and validated. No evolution was started.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
