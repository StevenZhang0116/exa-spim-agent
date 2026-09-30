"""Linux Landlock filesystem boundary plus seccomp: applied before any agent code.

Supports library imports and CPU threads, but no network or child processes.
Unsupported kernels fail closed; never fall back to an unrestricted worker.
"""
import ctypes
import errno
import os
from pathlib import Path
import platform
import sys
import sysconfig


def restrict_filesystem(read_paths, write_paths):
    if platform.machine() not in ('x86_64', 'aarch64') or sys.platform != 'linux':
        raise RuntimeError('Agent model workers require Linux x86_64/aarch64 with Landlock')
    libc = ctypes.CDLL(None, use_errno=True)
    # These numbers are shared by x86_64 and aarch64.
    create, add, restrict = 444, 445, 446
    abi = libc.syscall(create, 0, 0, 1)
    if abi < 3:
        raise RuntimeError('Agent model workers require Landlock ABI >= 3; isolation unavailable')
    # Handle all filesystem operations supported by this ABI, including truncate.
    rights = (1 << 15) - 1
    if abi >= 5:
        rights |= 1 << 15  # IOCTL_DEV
    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    config = Ruleset(rights)
    fd = libc.syscall(create, ctypes.byref(config), ctypes.sizeof(config), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'Landlock ruleset creation failed')
    read = (1 << 2) | (1 << 3)  # READ_FILE | READ_DIR; execution disallowed.
    write = read | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 13) | (1 << 14)
    try:
        for paths, permissions in ((read_paths, read), (write_paths, write)):
            for raw in paths:
                path = Path(raw).resolve()
                if not path.exists():
                    continue
                allowed = permissions if path.is_dir() else permissions & ((1 << 1) | (1 << 2) | (1 << 14))
                target = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    rule = PathRule(allowed, target)
                    if libc.syscall(add, fd, 1, ctypes.byref(rule), 0) < 0:
                        raise OSError(ctypes.get_errno(), 'Landlock path rule failed')
                finally:
                    os.close(target)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(restrict, fd, 0):  # NO_NEW_PRIVS
            raise OSError(ctypes.get_errno(), 'Landlock enforcement failed')
    finally:
        os.close(fd)


def runtime_paths():
    # Do not grant the repository, home, /tmp, /proc, or environment prefix wholesale.
    paths = [sysconfig.get_path(k) for k in ('stdlib', 'platstdlib', 'purelib', 'platlib')]
    paths += [str(Path(sys.prefix) / 'lib'), '/usr/lib', '/usr/lib64', '/lib', '/lib64',
              '/etc/ld.so.cache', '/etc/localtime', '/dev/urandom', '/dev/random', '/dev/zero',
              '/proc/cpuinfo', '/proc/meminfo', '/sys/devices/system/cpu']
    return list(dict.fromkeys(p for p in paths if p))


def restrict_syscalls():
    lib = ctypes.CDLL('libseccomp.so.2')
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    class Compare(ctypes.Structure):
        _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_int),
                    ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                         ctypes.c_uint, ctypes.POINTER(Compare)]
    context = lib.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError('Cannot create model seccomp filter')
    try:
        allowed = ('read write readv writev pread64 pwrite64 close close_range '
                   'open openat stat lstat fstat newfstatat statx access faccessat faccessat2 '
                   'readlink readlinkat getdents getdents64 lseek ftruncate truncate fsync fdatasync '
                   'mkdir mkdirat rmdir unlink unlinkat rename renameat renameat2 '
                   'getcwd chdir fchdir dup dup2 dup3 fcntl pipe pipe2 poll ppoll select pselect6 '
                   'mmap mprotect munmap mremap madvise brk futex futex_waitv membarrier '
                   'rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack '
                   'clock_gettime clock_getres gettimeofday time nanosleep clock_nanosleep '
                   'getpid getppid gettid getuid geteuid getgid getegid uname sysinfo getrlimit ugetrlimit '
                   'sched_yield sched_getaffinity getrandom getrusage times '
                   'set_tid_address set_robust_list rseq arch_prctl '
                   'restart_syscall exit exit_group').split()
        for name in allowed:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add(context, 0x7fff0000, number, 0):
                raise RuntimeError('Cannot allow syscall: ' + name)
        # A library may pin its calling thread, never change the host's affinity.
        number = lib.seccomp_syscall_resolve_name(b'sched_setaffinity')
        compare = Compare(0, 4, 0, 0)  # pid == 0: calling thread
        if number >= 0 and lib.seccomp_rule_add_array(context, 0x7fff0000, number, 1, ctypes.byref(compare)):
            raise RuntimeError('Cannot restrict sched_setaffinity to the calling thread')
        # Libraries may inspect limits, but cannot raise or change them.
        number = lib.seccomp_syscall_resolve_name(b'prlimit64')
        compare = Compare(2, 4, 0, 0)  # SCMP_CMP_EQ: new_limit == NULL
        if number >= 0 and lib.seccomp_rule_add_array(context, 0x7fff0000, number, 1, ctypes.byref(compare)):
            raise RuntimeError('Cannot restrict prlimit64 to reads')
        # glibc falls back from clone3(ENOSYS) to clone. Permit threads only.
        number = lib.seccomp_syscall_resolve_name(b'clone3')
        if number >= 0 and lib.seccomp_rule_add(context, 0x00050000 | errno.ENOSYS, number, 0):
            raise RuntimeError('Cannot restrict clone3')
        number = lib.seccomp_syscall_resolve_name(b'clone')
        flags = 0x00010000 | 0x00000100 | 0x00000800  # THREAD | VM | SIGHAND
        compare = Compare(0, 7, flags, flags)  # SCMP_CMP_MASKED_EQ
        if number < 0 or lib.seccomp_rule_add_array(context, 0x7fff0000, number, 1, ctypes.byref(compare)):
            raise RuntimeError('Cannot restrict clone to threads')
        if lib.seccomp_load(context):
            raise RuntimeError('Cannot enforce model seccomp filter')
    finally:
        lib.seccomp_release(context)


def isolate(read_paths, write_paths):
    # Resolve/load the seccomp library before Landlock; /lib is also allowed for imports.
    restrict_filesystem([*runtime_paths(), *read_paths], [*write_paths, '/dev/null'])
    restrict_syscalls()
