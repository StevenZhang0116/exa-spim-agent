# Swallowed input failures and library version skew

Status: implemented 2026-10-20 in `agentic/rerun_experiments.py`,
`agentic/run_discovery_workflow.py`, the discovery-reproducer and discovery-extrapolator
specs, and `agentic/tests/`. Motivating case: new-merge-1 ID 33.

## Problem

ID 33 (`hypo_33.py`) opens the S3 image with

```python
'aws_credentials': {'type': 'anonymous'}
```

This is the newer tensorstore spec form. panda's tensorstore 0.1.71 accepts only
`{'anonymous': True}`, `{}` or no `aws_credentials` (checked with `ts.Spec`). `ts.open`
raises `ValueError` synchronously, the script catches it, reads 0 patches, computes
all-NaN features and a delta AP of +0.00000, and exits 0. The runner called the result
USABLE, so:

- fix-loading was told not to touch USABLE scripts, and nothing repaired it;
- reproduction was FAILED (the fold agent noticed it by reading the output), and
  extrapolation was INCONCLUSIVE;
- fix-tests authored a corrected script that would read nothing again.

The other 38 tensorstore scripts omit `aws_credentials` and read real data on both
brains, so this was one script. The gap is general, though:

- **A. Swallowed input failure.** A data source fails, the script catches the error,
  continues on empty or fallback data, and exits 0. Causes include version skew,
  credentials, network, wrong paths and truncated companions (the split AD npz).
- **B. Library version skew.** The AD sandbox and panda differ in library versions,
  and recorded code uses a form panda rejects. It fails loudly (non-zero exit, already
  routed to fix-loading) or silently (class A).

`result_usability` checked only timeout, exit code, empty stdout and a "dataset not
found" regex. The spec's `INPUTS READ` line is printed by 29 of 91 scripts, mostly
free-form, so it cannot be the main signal.

## Goals and non-goals

Goals:
- detect class A in the runner, deterministically, in every compute step (reproduce,
  extrapolate, corrected);
- send it into the existing fix-loading → remeasure loop;
- repair class B generically (agent) and recurring cases deterministically (table);
- record all of it in the results JSON.

Non-goals:
- exceptions swallowed inside analysis code (the verifier's job);
- failed imports (see "Changes from the first draft");
- upgrading panda packages;
- parallel execution.

## Design

### 1. Input audit (`_INPUT_AUDIT` in `rerun_experiments.py`)

One source string runs in the script's process on both paths. `_RUNNER` (subprocess) and
`_apply_inprocess_patches` (fork) both execute it, so they cannot drift.

| Source key | Hook |
|---|---|
| `tensorstore` | `ts.open` (synchronous errors), `ts.Future.result`, `ts.TensorStore.__array__` |
| `np.load:<file>` | `numpy.load` |
| `npz:<file>[<key>]` | `numpy.lib.npyio.NpzFile.__getitem__`, one key per array, so one corrupt companion array is caught while its siblings read fine |
| `pickle.load` | `pickle.load` (on the fork path it wraps the resident-payload shortcut) |

How it records:
- Each source counts `ok`, `failed` (first lines of up to 3 errors) and `absent`.
  `KeyError` and `FileNotFoundError` count as `absent`, because probing for an optional
  key or file is normal.
- numpy and tensorstore are hooked by a `sys.meta_path` finder when the script first
  imports them, or at once if they are already loaded. The parent forks every script
  and never imports tensorstore itself.
- The counts are written once to `$RERUN_INPUT_AUDIT`: by atexit on the subprocess path,
  and by an explicit call before `os._exit` in the forked child.
- The run handle (`_with_input_audit`) reads the file into `run["input_audit"]`, which
  the checkpoint persists.

### 2. Usability gate

`result_usability`, after the timeout and exit-code checks:
- a source with `failed > 0 and ok == 0` → `UNUSABLE` (`swallowed-input-error`); partial
  failures stay USABLE with their counts;
- an `INPUTS READ` line with zero image patches read (`image_patches=0/<n>` with n > 0,
  or `0 image patches read`) → `UNUSABLE` (`zero-inputs`). This is a secondary signal
  for readers outside the hooks.

For each swallowed source, the runner appends
`[runner] swallowed input error: <source>: <error>` to stderr, so the driver digest
(`_failure_brief`, last stderr line) shows the cause. Both reasons are in
`_DATA_LOADING_REASONS` (failure kind `data-loading`).

`_RUNNER_VERSION` is `work-items-v2`, so v1 checkpoint items, which have no audit, are
recomputed.

### 3. Repair through the existing loop

An UNUSABLE result enters fix-loading, and remeasure reruns the changed script.

`VERSION_SKEW_RULE` sits next to `MISSING_PACKAGE_RULE` in both fix-loading
instructions. `_REVISION_GUIDE` and `discovery-reproducer.md` carry the same rule:

- a swallowed read is a loading fault;
- when the cause is a call or spec form this library version rejects, make the smallest
  compatible edit to that one call;
- never install, upgrade or vendor the library.

A missing package stays the runner's job (`INSTALLABLE_MODULES`).

Extrapolation and corrected runs have no repair loop. Their UNUSABLE results fold as
INCONCLUSIVE or as having no post-correction verdict.

### 4. Compat rewrite table (`COMPAT_REWRITES`)

Each entry has a `name`, a `pattern`, a `replacement` and a `probe`:

- The probe is Python source, run once per invocation in a fresh interpreter. It exits
  0 only when the installed library rejects the original form and accepts the
  replacement, so an upgrade turns the entry off by itself.
- First entry, `tensorstore-anonymous-credentials`: rewrites
  `aws_credentials: {'type': 'anonymous'}` to `{'anonymous': True}`, in dict-key or
  keyword form and with either quote style.

`_run_work_item` applies the table to every executed code (origin, extra after
`retarget_image`, corrected). The run records `compat_rewrites`, and the log prints
`[runner] compat rewrite <name> (id <id>)`.

The work key and the checkpoint's `code_sha256` hash the rewritten code. The result's
`executed_code_sha256` stays on the script, so extra-only fingerprint checks, change
detection and the stall check are unaffected.

An entry is added only for a skew seen in more than one script or run, each with a unit
test. A one-off goes through fix-loading.

## Tests (`agentic/tests`, no dataset loads, no network)

`InputAuditTests` (with a fake `tensorstore` package):
- a swallowed open is `swallowed-input-error` on both paths, and stderr ends with the
  runner line;
- one failed open beside two good ones stays USABLE (6 ok, 1 failed);
- a corrupt npz array is swallowed, while a missing key counts as absent;
- `INPUTS READ` zero parsing.

`CompatRewriteTests`:
- a rewrite is applied only when the probe says so;
- the work item runs and keys the rewritten code;
- the result keeps the script hash and lists the rewrites;
- the rewritten spec parses with the installed tensorstore;
- v1 work items are not reused.

`test_discovery_smoke`:
- every fix round carries `VERSION_SKEW_RULE`;
- `_failure_brief` shows the runner line.

The fork-path tests hide an already-imported real tensorstore from `sys.modules`, so the
child loads the fake.

## Rollout

1. Delete all new-merge-1 artifacts, both checkpoints included (the runner version
   changed), and rerun from scratch.
2. Expected in the log:
   - `[runner] compat rewrite tensorstore-anonymous-credentials (id 33)`, with ID 33
     reading about 11,245 patches on 794495;
   - no `swallowed-input-error` unless something else is broken. Any that appear go
     through fix-loading.
3. A skew that fix-loading repairs more than once becomes a table entry.

## Limits

- Hook overhead is one Python call per `Future.result` or `__array__`, which is
  negligible at about 30k reads.
- Only zero successes count as swallowed. A script that tries several sources until one
  works stays USABLE, and so does a script whose tensorstore open succeeds but every
  read fails.
- Readers outside the hooks (zarr, h5py, s3fs; no current script uses them) are covered
  only by the `zero-inputs` line until hooks are added.

## Changes from the first draft

- **No import audit.** Successful imports would always give `ok > 0`, so the
  zero-success rule could never fire. Flagging every failed import would also catch
  legitimate fallbacks, such as `cannot import name` for a function a newer library
  adds.
- **No incremental failure log.** A run killed by its timeout is UNUSABLE anyway.
- **`zero-inputs` reads image patches only.** Zero fragment nodes or companion rows can
  be legitimate for a script that does not use that input.
- **The probe checks both forms.** The first draft used an invalid bucket name, which
  fails on any version.
