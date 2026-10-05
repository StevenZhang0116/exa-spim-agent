"""Standalone Linux scorer worker. No labels or credentials enter this process.

Load numeric inputs and libraries, then enforce a seccomp syscall allowlist
before executing any candidate code. Unsupported kernels fail closed.
"""

import ctypes
import errno
import json
import math
from pathlib import Path
import resource
import sys
import traceback

import numpy as np
import pandas as pd
# NumPy exposes these lazily; pandas missing-value checks use np.rec.
# Load numerical dependencies before the syscall filter denies file reads.
import numpy.rec
import numpy.linalg
import numpy.fft
import numpy.ma
import numpy.random


def restrict_syscalls():
    lib = ctypes.CDLL('libseccomp.so.2')
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x00050000 | errno.EPERM)  # SCMP_ACT_ERRNO
    if not context:
        raise RuntimeError('Cannot create seccomp scorer filter')
    try:
        # No open/stat-by-path, network, process creation, ptrace, kill, exec,
        # namespace, or filesystem mutation calls. Only bootstrap-owned FDs.
        allowed = ('read write readv writev close fstat lseek ftruncate fsync fdatasync '
                   'mmap mprotect munmap mremap madvise brk futex '
                   'rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack '
                   'clock_gettime clock_getres gettimeofday time nanosleep clock_nanosleep '
                   'getpid gettid getuid geteuid getgid getegid sched_yield sched_getaffinity '
                   'getrandom restart_syscall exit exit_group').split()
        for name in allowed:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add(context, 0x7fff0000, number, 0) != 0:
                raise RuntimeError('Cannot allow scorer syscall: ' + name)
        if lib.seccomp_load(context) != 0:
            raise RuntimeError('Cannot enforce seccomp scorer filter')
    finally:
        lib.seccomp_release(context)


def invoke(source, frame, kind):
    namespace = {}
    exec(compile(source, '<scorer>', 'exec'), namespace)
    if 'ENUM_PARAMS' in namespace or 'propose_edits' in namespace:
        raise ValueError('Fixed-pool policies define score_candidates, not ENUM_PARAMS/propose_edits')
    scorer = namespace.get('score_candidates')
    if not callable(scorer):
        raise ValueError('Policy must define score_candidates(features, ctx)')
    output = []
    for _ in range(2):
        values = np.asarray(scorer(frame.copy(deep=True), {'kind': kind}), dtype=float).copy()
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError('Return one finite score per candidate in original row order')
        output.append(values)
    if not np.array_equal(*output):
        raise ValueError('Scorer must be deterministic')
    return output[0]


def main():
    root = Path(sys.argv[1])
    request = json.loads((root / 'request.json').read_text())
    source = (root / 'scorer.py').read_text()
    data = np.load(root / 'features.npy', allow_pickle=False)
    frame = pd.DataFrame(data, columns=request['columns'], copy=False)
    output = (root / 'scores.npy').open('wb')
    status = (root / 'status.json').open('w')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (request['memory_bytes'], request['memory_bytes']))
    cpu = math.ceil(request['timeout']) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    size = max(1 << 20, len(frame) * 8 + 4096)
    resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
    # Ensure numpy's serialization imports are loaded before file access stops.
    import io
    np.save(io.BytesIO(), np.zeros(1), allow_pickle=False)
    try:
        restrict_syscalls()
    except Exception as exc:
        json.dump({'status': 'sandbox_error', 'error': str(exc)}, status)
        return 2
    try:
        scores = invoke(source, frame, request['kind'])
        buffer = io.BytesIO()
        np.save(buffer, scores, allow_pickle=False)
        output.write(buffer.getvalue())
        output.flush()
        json.dump({'status': 'ok'}, status)
        return 0
    except BaseException as exc:
        # Traceback formatting must not try to read source files after filtering.
        location = traceback.extract_tb(exc.__traceback__, limit=8)
        json.dump({'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                   'frames': [{'function': f.name, 'line': f.lineno} for f in location]}, status)
        return 1
    finally:
        output.close()
        status.close()


if __name__ == '__main__':
    raise SystemExit(main())
