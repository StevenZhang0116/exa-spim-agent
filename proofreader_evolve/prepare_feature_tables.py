"""Generate evolution-matching detector tables, then run read-only validation.

Run with panda on an allocated compute node (e.g. n257):
    python -u proofreader_evolve/prepare_feature_tables.py --brains 794495 789202 794491 794493 802449

Without --brains, the preparation default is 789202 and 794491; include 794495
explicitly to prepare the current default evolution TRAIN brain. MCL defaults
to 100, with labeled _add caches and the exact native candidate pools of the
current default frozen models. The shared merge
default is autodiscovery-application/merge-error-794495-mcl100_2026-08-04;
precompute, preflight and evolution all resolve it through precompute_error_scores.
Each brain runs in a fresh subprocess to release memory before the next.
Existing matching tables are reused. The first failure stops the script with a
nonzero exit code. No LLM calls or evolution.

Use --dry-run to print commands without importing models or loading any data.
Real runs mirror console output to proofreader_evolve/log/prepare_feature_tables_<timestamp>_<pid>.log.txt.
Use --log-txt to choose a log file (appended to if it already exists).
Use --threads all to give numerical libraries all CPUs available to this process
(respecting CPU affinity). Serial detector loops remain serial.
Merge extraction can scan the entire graph; this is not a quick smoke test.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Tee:
    """Mirror output to the console and flush the log on every write."""

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


def run_command(command, *, cwd, env):
    """Relay child output through the tee; inherited descriptors bypass it."""
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1) as process:
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            returncode = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


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
    parser.add_argument("--log-txt", type=Path,
                        help="Append console output to this file (default: a new file in proofreader_evolve/log/)")
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
    if args.dry_run:
        return run(args)
    log_path = (args.log_txt or PROJECT_ROOT / "proofreader_evolve" / "log" /
                f"prepare_feature_tables_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log.txt")
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    returncode = 1
    with log_path.open("a", encoding="utf-8") as log_file, \
            redirect_stdout(_Tee(sys.stdout, log_file)), \
            redirect_stderr(_Tee(sys.stderr, log_file)):
        print(f"\n# argv: {shlex.join([sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])}")
        print(f"# started {datetime.now():%Y-%m-%d %H:%M:%S%z}")
        print(f"# host {platform.node()}  python {platform.python_version()}")
        print(f"Log: {log_path}")
        try:
            returncode = run(args)
            return returncode
        except KeyboardInterrupt:
            returncode = 130
            print("Interrupted; no further brains will be started.", file=sys.stderr)
            return returncode
        except Exception:
            traceback.print_exc()
            return returncode
        finally:
            outcome = "OK" if returncode == 0 else "FAILED"
            print(f"\n# {outcome} (exit code {returncode}) after {time.monotonic() - started:.1f}s "
                  f"(ended {datetime.now():%Y-%m-%d %H:%M:%S%z})")


def run(args):
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
            run_command(command, cwd=PROJECT_ROOT, env=env)
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
