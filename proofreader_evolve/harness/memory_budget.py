"""Conservative admission estimates for host input preparation and descriptor workers.

This is not an allocator or an OOM guarantee. In Slurm, require a readable cgroup
v2 limit rather than treating the entire node's memory as the job's allowance.
"""
import os
from pathlib import Path

from .model_execution import ModelExecutionError


GIB = 1024**3
MIN_RESERVE_BYTES = 8 * GIB
PREPARATION_BYTES = 512 * 1024**2  # two 128 MiB patch buffers plus geometry/transport allowance


def memory_snapshot(cgroup_file='/proc/self/cgroup', cgroup_root='/sys/fs/cgroup',
                    meminfo='/proc/meminfo'):
    """Use the least headroom of all finite cgroup ancestors, without reclaim estimates."""
    limits = []
    try:
        relative = next(line.split('::', 1)[1] for line in Path(cgroup_file).read_text().splitlines()
                        if line.startswith('0::'))
        root = Path(cgroup_root).resolve()
        node = (root / relative.lstrip('/')).resolve()
        while node.is_relative_to(root):
            try:
                raw = (node / 'memory.max').read_text().strip()
                if raw != 'max':
                    limit = int(raw)
                    used = int((node / 'memory.current').read_text())
                    limits.append((limit, used))
            except (OSError, ValueError):
                pass
            if node == root:
                break
            node = node.parent
    except (OSError, StopIteration):
        pass
    if limits:
        reserve = max(MIN_RESERVE_BYTES, min(limit for limit, _ in limits) // 10)
        return {'source': 'cgroup_v2', 'limit_bytes': min(limit for limit, _ in limits),
                'headroom_bytes': min(limit - used for limit, used in limits),
                'reserve_bytes': reserve}
    if os.environ.get('SLURM_JOB_ID'):
        raise ModelExecutionError('Cannot read a finite Slurm cgroup v2 memory limit; '
                                  'descriptor execution requires a measurable job memory allowance')
    try:
        fields = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in Path(meminfo).read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
        return {'source': 'host_memavailable', 'limit_bytes': fields['MemTotal'],
                'headroom_bytes': fields['MemAvailable'],
                'reserve_bytes': max(MIN_RESERVE_BYTES, fields['MemTotal'] // 10)}
    except (OSError, ValueError, KeyError) as exc:
        raise ModelExecutionError('Cannot determine descriptor memory headroom') from exc


def admission(memory_mb, workers, *, pending=0):
    """Reserve a whole worker cap plus host preparation per admitted batch.

Pending batches are charged in full, even when some memory is already resident;
this deliberately errs on the conservative side and avoids simultaneous launch races.
"""
    snapshot = memory_snapshot()
    batch_bytes = int(memory_mb) * 1024**2 + PREPARATION_BYTES
    slots = max(0, (snapshot['headroom_bytes'] - snapshot['reserve_bytes']) // batch_bytes - pending)
    return {**snapshot, 'batch_reservation_bytes': batch_bytes,
            'pending_batches': pending, 'available_workers': min(max(0, workers - pending), slots)}


def require_capacity(snapshot):
    if snapshot['available_workers'] < 1:
        raise ModelExecutionError('Insufficient descriptor memory headroom for one batch: '
                                  f"{snapshot['headroom_bytes'] / GIB:.2f} GiB available, "
                                  f"{snapshot['reserve_bytes'] / GIB:.2f} GiB reserve, "
                                  f"{snapshot['batch_reservation_bytes'] / GIB:.2f} GiB per batch; "
                                  'request more Slurm memory or lower --descriptor-memory-mb '
                                  'if the descriptor can run within the smaller worker cap')
