"""Reconcile a run's saved status with its recorded Slurm step; run on a compute node."""
import argparse
import json
from pathlib import Path

from ..harness.run_lifecycle import reconcile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args(argv)
    if not args.run.is_dir():
        parser.error(f'Run directory does not exist: {args.run}')
    print(json.dumps(reconcile(args.run), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
