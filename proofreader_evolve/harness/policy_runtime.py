"""Execution time guard for generated ranking scorers."""

import signal
import threading


class PolicyTimeout(Exception):
    """Raised when evolved policy code exceeds its wall-clock budget.

    A policy is arbitrary evolved code; a quadratic feature over a whole-brain
    candidate stream can run for an hour (observed: a per-site full-graph distance
    scan). The budget turns that from a silent multi-hour stall into a fast, bounded
    REJECT — the same failure class as a non-importing or lint-failing revision.
    """


class policy_time_budget:
    """Context manager that aborts the body if it runs longer than ``seconds``.

    Uses SIGALRM on the main thread (the common case: run_candidate is called
    synchronously from the loop), which can interrupt even a tight C-level numpy loop
    at the next Python bytecode check. When not on the main thread (SIGALRM is
    unavailable there), it degrades to a NO-OP guard — the wall-clock is still recorded
    by the caller, so a slow policy is at least VISIBLE even if not interrupted. A
    non-positive or None budget disables the guard entirely.
    """

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self._armed = False
        self._old_handler = None

    def __enter__(self):
        if not self.seconds or self.seconds <= 0:
            return self
        # signal.alarm only works on the main thread; guard so a worker-thread caller
        # (or a platform without SIGALRM) degrades gracefully instead of raising.
        if threading.current_thread() is not threading.main_thread():
            return self
        if not hasattr(signal, "SIGALRM"):
            return self

        def _fire(signum, frame):
            raise PolicyTimeout(
                f"Policy exceeded {self.seconds:g}s budget")

        self._old_handler = signal.signal(signal.SIGALRM, _fire)
        # setitimer takes a float; alarm() would truncate a sub-second budget to 0.
        signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
        self._armed = True
        return self

    def __exit__(self, *exc):
        if self._armed:
            signal.setitimer(signal.ITIMER_REAL, 0)  # disarm
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
            self._armed = False
        return False  # never suppress (PolicyTimeout propagates to the evaluator)
