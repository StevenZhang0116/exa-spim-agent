# Parallel compute steps

Status: implemented 2026-10-20 (`agentic/rerun_experiments.py`, `agentic/run_discovery_workflow.py`, tests in `agentic/tests/`). Not yet run on a full workflow. See "As built" at the end.

## Why

The compute steps ran one script at a time. On 2026-10-19, the new-merge-1 run on job
27344406 (16 CPUs, 200 GiB) showed one child at 96% CPU with one thread, while 15 cores
sat idle.

| Part | Wall time | Note |
|---|---|---|
| reproduce-run (794495, 91 scripts) | 1.63 h | median 61 s, longest 208 s |
| extrapolate-run (802449, 91 scripts) | 4.01 h | median 149 s, longest 425 s |
| fix-tests-measure (estimated) | about 1.6 h | 27 corrected scripts on both brains |
| agent steps | about 4.2 h | verify 47 min, fix-tests about 1 h 50 min |
| whole run | about 11.5 h | compute is about 62% |

Longest-first scheduling of the recorded per-script runtimes, ignoring contention:

| Concurrent scripts | reproduce | extrapolate |
|---|---|---|
| 1 (today) | 1.63 h | 4.01 h |
| 4 | 0.41 h | 1.00 h |
| 6 | 0.27 h | 0.67 h |
| 8 | 0.21 h | 0.50 h |

## Invariants: results do not change

- Same work items, code (after compat rewrites), data, per-script timeout and
  checkpoint keys. The results JSON differs only in `runtime_ms`, the log order, and the
  new top-level `jobs` field.
- `--jobs 1` keeps today's behavior: one child at a time, and no thread caps.
- A result never comes from a run that parallelism disturbed. A child killed by the
  kernel is rerun alone (see 1.2).
- Scripts do not share files. All 91 new-merge-1 scripts have no absolute `/tmp` path
  and no savefig, to_csv, np.save or write-mode open, and each one runs in its own
  sandbox directory.

## Design

### 1. Runner: `--jobs N` (`agentic/rerun_experiments.py`)

**1.1 Start and poll instead of blocking.**
- `start_run(code, pkl, timeout, companions)` builds the sandbox (keeping the
  `TemporaryDirectory` on the handle), the env and the links. It then forks the child
  (`_fork_child`, the body of the former `_run_forked`), or on the `--no-preload` path starts
  `subprocess.Popen` with stdout and stderr going to files.
- `RunHandle.poll()` returns `None` while the child runs, kills it at its deadline
  (`timed_out`), and otherwise returns the same run dict as today, with `input_audit`
  attached. It samples the child's private memory (`Private_Clean + Private_Dirty` from
  `/proc/<pid>/smaps_rollup`) and keeps the peak.
- `rerun_one` becomes start + wait, so every existing caller and test keeps its meaning.

**1.2 Work-item pool (`_run_pool`).**
- An item is a record id, a dataset, the code after `apply_compat_rewrites`, the
  companions and a predicted runtime.
- The checkpoint is consulted first with the same `_work_key`. Reused items never start.
- **Order:** longest predicted runtime first. An origin item takes its prediction from
  `rerun_runtime_ms` in `--base-results`. An extra item takes its origin runtime, which
  in `--extra-only` mode comes from `--origin-results`. Items with no prediction keep
  rank order.
- **Admission:** start an item while fewer than N children run and the memory headroom
  covers the reserve (1.3).
- **On completion:** the parent writes the checkpoint atomically, as today, and prints
  the same `[rerun k/n] id=X ... OK in Ns` line.
- **Missing package:** for an environment failure naming an `INSTALLABLE_MODULES` entry,
  the parent installs it (existing lock, once per module) and requeues the item, as
  `_run_installing_missing` does today.
- **Killed by the kernel:** a child ended by SIGKILL that the runner did not send (an
  OOM kill) is discarded and requeued to run alone once the pool drains. A second kill
  is recorded as is, as a serial run would record it.

**1.3 Memory gate.**
- **Headroom:** walk the ancestors of `/proc/self/cgroup` (cgroup v2) and take the
  minimum of `memory.max - (memory.current - inactive_file)` over those with a finite
  `memory.max`. Fall back to `MemAvailable` from `/proc/meminfo`.
  - On job 27344406 the job cgroup has `memory.max` 200 GiB and `memory.current`
    35.9 GB, of which 34.8 GB is inactive file cache. The task-level cgroups say `max`.
- **Reserve for a new child:** `max(1.25 × the largest child peak seen so far, 2 GB)`.
  Until a child has finished, use the parent's RSS, the worst case of a child copying
  the whole payload.
- The gate only delays starts. It never changes what runs.

**1.4 Thread caps (only when N > 1).**
- Each child caps BLAS and OpenMP threads at `max(1, cpus // N)`, where
  `cpus = len(os.sched_getaffinity(0))`.
- **Subprocess path:** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
  `NUMEXPR_NUM_THREADS` in the env.
- **Fork path:** `threadpoolctl.threadpool_limits` (3.6.0 is in panda), because the
  parent loaded the libraries before forking, so env vars come too late.
- **Effect on numbers:** multithreaded BLAS reductions may change in the last bit.
  Most scripts are single-threaded (nlwp 1 when observed). The 26 scripts with a
  32–64-thread `ThreadPoolExecutor` use it for S3 reads, which these caps do not touch.

**1.5 Phases in one invocation.**
- Origin items for all records run first. Extra items then run for records whose
  origin result is USABLE, the same rule as today.
- `--extra-only` (the extrapolate step) has only extra items.
- Results are assembled in rank order, with extrapolations in `--extra-pkl` order.
- The payload gains `jobs`.

### 2. Driver (`agentic/run_discovery_workflow.py`)

- **`--jobs N`, default 6.** `rerun_argv` passes it to every non-export compute step.
  `--jobs 1` gives today's serial run.
- **Skip round-1 fix-loading when nothing is broken.** Round 1 gets
  `skip_if_no_unusable_in: <.reproduce-raw.json>`. Its instruction already forbids
  touching USABLE scripts, so skipping it when nothing is UNUSABLE cannot change a
  script. On 2026-10-19 it ran 334 s and cost $0.85 with nothing to fix. The remeasure
  after it still runs and writes `.reproduce.json` (8 s, no dataset load).

### 3. Docs

- the CLAUDE.md resident-payload paragraph (jobs, memory gate, thread caps, rerun alone
  after a kill), plus a `--jobs` row in the flag table;
- the `rerun_experiments.py` module docstring.

## Not in this revision

- **Overlapping agent and compute steps** (summarize beside the reproduce chain, and
  reproduce-fold beside extrapolate-run). After parallel compute this saves about
  30 min. It needs a second SDK client (one client streams one turn), compute in a
  thread, interleaved logs and a new `--resume-from` model.
- **Fix-tests batches in parallel.** This saves about 1 h, but weakens cross-batch
  consistency: batch 3 reused batch 2's statistics block on 2026-10-19.
- **Sharding verify or summarize, or cheaper models.** Both change results (run-wide BH,
  cross-entry synthesis).
- **A resident pkl server.** A load takes only 54–89 s per invocation.

## Tests (`agentic/tests`, no dataset loads, no network)

1. `--jobs 1` and `--jobs 3` write identical results JSON, apart from `runtime_ms` and
   `jobs`. The set mixes OK, non-zero exit, timeout, swallowed input and the "dataset not
   found" gate, and runs on both the fork and subprocess paths.
2. Scheduler with fake handles of set durations:
   - never more than N children at once;
   - longest predicted first;
   - results in rank order;
   - a checkpoint write after each completion;
   - reused items never start.
3. Memory gate:
   - a mocked headroom below the reserve keeps one child at a time;
   - cgroup parsing from a fake tree (nested `max`, finite `memory.max`,
     `inactive_file`), with the `/proc/meminfo` fallback.
4. A script that SIGKILLs itself once (marker file) is requeued alone and ends OK. One
   killed twice is recorded as a failure.
5. An environment failure on an allowlisted module installs once and requeues the item.
6. Extra items start only after their origin is USABLE, and `--extra-only` runs only
   extra items.
7. Thread caps:
   - the subprocess child sees the env vars;
   - the fork child reports the `threadpoolctl` limits;
   - `--jobs 1` sets neither.
8. Driver:
   - `rerun_argv` passes `--jobs` to run steps, not to the export;
   - round-1 fix-loading is skipped when the raw results hold no UNUSABLE result.

## Expected effect

| | Today | `--jobs 6` (4–5× after contention) |
|---|---|---|
| reproduce-run | 1.6 h | about 0.35 h |
| extrapolate-run | 4.0 h | about 0.9 h |
| fix-tests-measure | about 1.6 h | about 0.35 h |
| whole run | about 11.5 h | about 6 h |

Agent cost does not change.

## Rollout

1. Implement and test.
2. Rerun new-merge-1 from scratch with the default `--jobs 6`, and watch:
   - the startup line `[rerun] jobs 6, memory headroom ...`;
   - any `requeued alone` line;
   - the peak child memory in the log;
   - the step durations.
3. With no requeues and enough headroom, try `--jobs 8`.

## Risks

- **Memory on 802449**, whose parent payload is larger than 794495's 13.8 GB. The gate
  and the run-alone requeue keep results intact, at the cost of speed.
- **S3 bandwidth.** Concurrent image scripts share the node's bandwidth, so each one
  runs slower, with the same numbers.
- **`runtime_ms`** includes contention and is no longer comparable with serial runs.
- **Log order.** Lines come out of order, but each carries its id.

## As built

- **`--jobs 1` keeps the old code path.** Each item goes through
  `_run_work_item` → `rerun_one` (now `start_run(...).wait()`). Only `--jobs` above 1
  uses `_run_pool`.
- **Origin work first at every N.** With `--jobs 1` too, all origin work now runs
  before any extrapolation. This changes the log order only.
- **No prediction for reproduce-run.** It has no `--base-results`, so its items keep
  rank order. Checkpointed runtimes are not used as predictions.
- **A rerun after a kill gets no thread cap**, the same as a serial run.
- **The forked child opens fresh `sys.stdout` and `sys.stderr` on fds 1 and 2.**
  Before, it reused the parent's stream objects, which lose output when the parent's
  streams are redirected (as in tests).
- **Log lines:**
  - `[rerun] jobs N (T thread(s) per script) for K script(s); memory headroom ...`
    when a pool starts;
  - `[rerun] pool finished ...; largest script private memory ...; run alone after a
    kill: ...` when it ends;
  - `... killed (SIGKILL) while ...; requeued alone` for each rerun after a kill.
- **Tests:**
  - `ParallelPoolTests` (7 tests);
  - in `test_discovery_smoke`, `--jobs` forwarding and the round-1 skip.
  - The `--jobs 1` vs `--jobs 3` test uses an 8 s script timeout. A 3 s timeout
    flaked once on a busy login node.

