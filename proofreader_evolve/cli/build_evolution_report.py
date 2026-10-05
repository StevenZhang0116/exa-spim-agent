"""Build an offline report from saved precision-run artifacts; never run a policy."""

import argparse
from pathlib import Path
import time

from ..harness.evolution_report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', help='Run directory or name under proofreader_evolve/runs')
    parser.add_argument('--out', type=Path, help='Output HTML (default: RUN/report.html)')
    parser.add_argument('--watch', action='store_true', help='Rebuild while an experiment writes artifacts')
    parser.add_argument('--interval', type=float, default=10, help='Watch interval in seconds (minimum 2)')
    args = parser.parse_args(argv)
    if not 2 <= args.interval <= 3600:
        parser.error('--interval must be between 2 and 3600 seconds')
    directory = Path(args.run)
    if not directory.is_dir():
        directory = Path(__file__).resolve().parents[1] / 'runs' / args.run
    if not directory.is_dir():
        parser.error(f'Run directory does not exist: {args.run}')
    try:
        while True:
            print(f'Report: {write_report(directory, args.out).resolve()}', flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
