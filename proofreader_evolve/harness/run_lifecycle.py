"""Host-only execution identity and evidence-based reconciliation after a Slurm step exits."""
from datetime import datetime
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile


TERMINAL = {'COMPLETED', 'OUT_OF_MEMORY', 'TIMEOUT', 'CANCELLED', 'FAILED', 'NODE_FAIL',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def atomic_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def record_execution(run_dir):
    record = {'hostname': socket.gethostname(), 'pid': os.getpid(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'slurm_step_id': os.environ.get('SLURM_STEP_ID'),
              'started_at': datetime.now().astimezone().isoformat(timespec='seconds')}
    atomic_json(Path(run_dir) / 'execution.json', record)
    return record


def slurm_record(execution):
    job, step = execution.get('slurm_job_id', ''), execution.get('slurm_step_id', '')
    if not re.fullmatch(r'\d+(?:_\d+)?', job or '') or not re.fullmatch(r'\d+', step or ''):
        return {'query_error': 'No numeric Slurm job/step identity; cannot confirm termination'}
    target = f'{job}.{step}'
    try:
        result = subprocess.run(['sacct', '-n', '-P', '-j', target, '--format=JobIDRaw,State%32,ExitCode'],
                                text=True, capture_output=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return {'query_error': f'{type(exc).__name__}: {exc}'}
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if len(fields) >= 3 and fields[0] == target and fields[1].strip():
            return {'job_step': target, 'state': fields[1].split('+')[0].split()[0], 'exit_code': fields[2]}
    return {'query_error': f'Accounting has no record for {target} yet'}


def reconcile(run_dir, *, srun_exit=None):
    """Only a terminal step or an observed srun exit can end a stale running state.

    No age-based guesses, no implicit retries, and never override a completed run.
    An explicit later check may refine an unconfirmed exit after accounting catches up.
    """
    root = Path(run_dir)
    def read(name):
        path = root / name
        return json.loads(path.read_text()) if path.exists() else {}
    state = read('run_status.json')
    if state.get('status') == 'completed':
        return state
    evidence = slurm_record(read('execution.json'))
    terminal = evidence.get('state')
    if terminal not in TERMINAL and srun_exit is None:
        return {**state, 'slurm_observation': evidence, 'status_changed': False}
    if terminal == 'OUT_OF_MEMORY':
        status, kind, reason = 'failed', 'oom', 'Slurm reported OUT_OF_MEMORY'
    elif terminal in {'TIMEOUT', 'CANCELLED', 'PREEMPTED', 'DEADLINE'}:
        status, kind, reason = 'interrupted', terminal.lower(), f'Slurm reported {terminal}'
    else:
        status, kind = 'failed', 'unconfirmed_exit' if terminal not in TERMINAL else terminal.lower()
        reason = f'Execution ended without a completed run; Slurm={terminal or "unknown"}, srun_exit={srun_exit}'
    # Preserve useful Python failure details; append the independently observed termination.
    from .run_records import PRECISION
    result = {**state, 'objective': state.get('objective', PRECISION),
              'status': status, 'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
              'reason': reason,
              'termination': {'kind': kind, 'reason': reason, 'slurm': evidence, 'srun_exit': srun_exit}}
    if state.get('reason') and not state.get('termination'):
        result['execution_error'] = state['reason']
    # Re-read immediately before writing, so a completed result wins over reconciliation.
    latest = read('run_status.json')
    if latest.get('status') == 'completed':
        return latest
    atomic_json(root / 'run_status.json', result)
    try:
        from .evolution_report import write_report
        write_report(root)
    except Exception as exc:
        result['report_warning'] = f'{type(exc).__name__}: {exc}'
    return result
