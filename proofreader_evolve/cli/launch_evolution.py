"""Launch evolution in a separate Slurm step and reconcile its exit from the caller step.

Run this lightweight parent on an allocated compute node. If the parent/allocation
also dies, check_run_status can reconcile later from another compute allocation.

Resources: the driver does not size itself to the allocation. Its defaults (8 threads,
16 GiB and 300 s per fitting worker, 24 descriptor workers, 3600 s of classifier search)
are hard limits inside the reviser's tools, and a run that keeps them on a large node is
silently restricted: the agent answers cut fits and refused plans with smaller models and
cheaper descriptors. Pass the resource options after ``--`` sized to the allocation and
run one evolution run per allocation; for a dedicated 40-CPU / 600 GB job:

    --classifier-memory-mb 98304 --classifier-time-budget 3600 --policy-time-budget 2400
    --classifier-threads 32 --classifier-search-seconds 7200
    --descriptor-workers auto --reviser-max-turns 80

The table and its rationale are in docs/stage3_evolution.md ("Give the run the
allocation's resources").
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ..harness.run_lifecycle import atomic_json, reconcile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--job-id', default=os.environ.get('SLURM_JOB_ID'))
    parser.add_argument('--node', required=True)
    parser.add_argument('--cpus', type=int, required=True)
    parser.add_argument('--runs-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'runs')
    parser.add_argument('--python', default=sys.executable, help='Absolute panda interpreter path')
    parser.add_argument('evolution_args', nargs=argparse.REMAINDER, help='Arguments after -- go to run_evolution')
    args = parser.parse_args(argv)
    if not args.job_id or args.cpus < 1 or not Path(args.python).is_absolute():
        parser.error('Need --job-id (or SLURM_JOB_ID), positive --cpus and an absolute --python path')
    forwarded = args.evolution_args
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    if any(a.split('=', 1)[0] in {'--run-dir', '--runs-dir'} for a in forwarded):
        parser.error('Set --runs-dir on the launcher; it owns the unique --run-dir')
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='precision_' + datetime.now().strftime('%Y%m%d_%H%M%S_'),
                                      dir=args.runs_dir)).resolve()
    atomic_json(directory / 'launch.json', {'slurm_job_id': args.job_id, 'node': args.node,
                'cpus': args.cpus, 'python': args.python, 'evolution_args': forwarded})
    # Open OnDemand JupyterLab terminals carry SLURM_EXPORT_ENV=NONE, under which srun gives the step
    # an empty environment (no PATH, no ANTHROPIC_API_KEY; 2026-11-02): export the caller's explicitly.
    command = ['srun', f'--jobid={args.job_id}', '--overlap', '--exact', '--nodes=1', '--ntasks=1',
               f'--cpus-per-task={args.cpus}', f'--nodelist={args.node}', '--kill-on-bad-exit=1',
               '--export=ALL', args.python, '-u', '-m', 'proofreader_evolve.cli.run_evolution',
               '--run-dir', str(directory), *forwarded]
    print(f'Run directory: {directory}', flush=True)
    try:
        code = subprocess.call(command, cwd=Path(__file__).resolve().parents[2])
    except KeyboardInterrupt:
        # The caller was interrupted; without terminal Slurm evidence do not claim the child stopped.
        reconcile(directory)
        raise
    except OSError:
        reconcile(directory, srun_exit=127)
        raise
    reconcile(directory, srun_exit=code)
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
