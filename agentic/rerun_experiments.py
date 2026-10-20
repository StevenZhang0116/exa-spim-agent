"""
Re-execute AutoDiscovery experiments against a provided dataset pkl.

Deterministic helper for the ``discovery-reproducer`` agent
(``.claude/agents/discovery-reproducer.md``). Not a general-purpose tool — its
record schema and stdout contract are what that agent reads. If you change the
output shape here, update the agent prompt to match.

Each AutoDiscovery run export (one JSON file) is a list of tested-hypothesis
objects. Each carries a ``code`` field: a self-contained Python script that
loads a dataset ``.pkl``, runs a statistical test, and prints its results (the
recorded ``codeOutput``). This helper takes ONE run JSON and ONE dataset pkl,
selects the SAME records the summarizer reported (using the same ranking,
direction, optional top-K, and predictive manifest). Predictive mode forbids
top-K and therefore re-runs every non-excluded record. It re-runs each record's
``code`` against the given pkl, and prints fresh and recorded output as JSON.
It does NOT judge reproduction; it only re-executes deterministically.

Two execution paths
-------------------
Run directly (forced-load runner): each record's recorded ``code`` runs in a
fresh temp dir through a runner that monkeypatches ``open`` / ``os.path.exists``
/ ``glob.glob`` so any ``.pkl`` reference resolves to the provided pkl. This
works when the host environment can unpickle the dataset and the script's
bootstrap is benign.

Export → revise → rerun (agent-in-the-loop): the recorded scripts often fail to
reproduce because they cannot LOCATE the dataset — a "dataset not found" gate the
monkeypatch doesn't intercept, or a hardcoded path/glob that finds nothing. For
those, ``--export-dir`` dumps each selected record's code as an editable
``hypo_<id>.py`` (plus ``MANIFEST.json`` and ``REVISION_GUIDE.md``) so the agent
can revise ONLY the loading part — load the pkl directly from ``$RERUN_PKL`` —
keeping the analysis identical. ``--code-dir`` then executes those revised
scripts (falling back to the recorded ``code`` for any record without a revised
file) and reports which source each result used.

The driver owns the ENVIRONMENT, not the scripts. The host interpreter is the
single source of truth and is pre-provisioned with the scientific stack; the
runner neutralizes any ``pip``/``apt``/``conda`` install the recorded code
attempts (they become logged no-ops) so a script can neither swap the NumPy
version mid-run nor hang in an install retry loop. Before running anything, a
one-time preflight imports numpy/pandas/scipy/statsmodels and aborts loudly
(exit 3) if the env itself is broken. With ``--install-missing`` a script that
dies on an absent module from ``INSTALLABLE_MODULES`` gets that package
pip-installed (every installed distribution pinned, so nothing already there
changes version) and is run again, and a checkpointed environment failure is
never reused. If every script fails with an
import/native-load error and none succeed, the run is flagged
``environment_failure`` and also exits 3 — so a broken env can never be folded
as a batch of "could not run" statistical verdicts.

Corrected statistical test (fix a wrong test, then re-measure): when the
verifier flags a hypothesis whose STATISTICAL TEST is wrong (wrong test for the
data, violated assumptions, p-value misread, huge-n significance of a trivial
effect), ``--corrected-dir`` runs ``hypo_<id>.py`` scripts whose ANALYSIS has
been rewritten to the correct test (built on the loading-fixed script, keeping
the data and quantities the same). It takes precedence over ``--code-dir`` per
record, so the corrected result is real measured numbers, not a paper proposal.

Usage (paths relative to the ``exa-spim-agent/`` project root)
--------------------------------------------------------------
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl ../data/RUN.pkl
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --rank-by posterior-surprise --top 20
    # Predictive selection (the workflow creates this manifest):
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --direction predictive \
        --predictive-manifest autodiscovery/.predictive/RUN.selection.json
    # Agent-in-the-loop: export editable scripts, revise them, then rerun:
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --top 20 --export-dir autodiscovery/RUN.rerun
    #   (agent edits autodiscovery/RUN.rerun/hypo_<id>.py loading sections)
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --top 20 --code-dir autodiscovery/RUN.rerun
    # Selective loading-fix rerun; unchanged records are merged from the first pass:
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --code-dir autodiscovery/RUN.rerun --only-changed \
        --base-results autodiscovery/RUN.reproduce-raw.json
    # Extrapolate: run the reproduced code on OTHER datasets to test generalization
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --code-dir autodiscovery/RUN.rerun \
        --extra-only --origin-results autodiscovery/RUN.reproduce.json \
        --extra-pkl OTHER1.pkl --extra-pkl OTHER2.pkl
    # Corrected tests: re-measure the flagged hypotheses with the right test
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --code-dir autodiscovery/RUN.rerun --corrected-dir autodiscovery/RUN.fixed \
        --only-corrected

Extrapolation (generalization to other datasets)
------------------------------------------------
The dataset a hypothesis was generated on is the ORIGIN pkl (``--pkl``). Pass
one or more OTHER datasets via ``--extra-pkl`` (repeatable) to test whether each
finding GENERALIZES. In workflow use, ``--extra-only --origin-results`` reuses
the completed origin result and executes the SAME code only on each extra pkl.
All pkls are assumed to share the same payload structure (same keys). Each
result then carries an ``extrapolations`` list — one entry per extra pkl with
its own fresh output — so the agent can compare datasets without recomputing the
origin.

A script that hardcodes the origin brain's S3 image folder instead of reading
the companion manifest would read the origin image at the extra brain's
coordinates. When both pkls have a companion, the runner replaces that folder
with the extra brain's (the extrapolation then carries ``image_retargeted``) if
both images share bucket and level geometry, and otherwise does not run the
script there (UNUSABLE, reason ``origin-image-path``).

Every code×dataset execution can be persisted with ``--checkpoint``. Its key
includes the source-run hash, record id, exact code hash (after compat
rewrites), dataset identity, timeout, runner version, Python, and
scientific-package versions. A changed input invalidates only the affected work
item; completed matching items resume without execution, except an environment
failure, which is always run again.

Parallel execution
------------------
All origin work runs first, then the extrapolation work of records whose origin
result is USABLE. ``--jobs 1`` runs each phase one script at a time. ``--jobs N``
runs up to N at once, longest predicted runtime first. A script starts only when
the job's cgroup can hold it, gets cpus // N BLAS/OpenMP threads, and is run again
alone if the kernel kills it. Results, checkpoint keys and the results JSON order
do not depend on N; only ``runtime_ms`` and the log order do.

Swallowed input failures and version skew
-----------------------------------------
A script that catches a failed data read and carries on with no data exits 0.
The runner counts, inside the script's process, every tensorstore open and read,
``np.load`` and npz array read, and ``pickle.load`` (``input_audit`` per
source). When every access to one source failed, the result is UNUSABLE
(``swallowed-input-error``) and a ``[runner] swallowed input error`` line closes
its stderr; an ``INPUTS READ`` line reporting zero image patches gives
``zero-inputs``. Both go to the loading-repair agents. A library form this
environment rejects that recurs across scripts gets a probed source rewrite in
``COMPAT_REWRITES`` instead (``compat_rewrites`` in the result).
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import NamedTuple

# Reuse the exact ranking the summarizer uses so the rerun set matches the
# report set. rank_by_surprise.py lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_by_surprise import (  # noqa: E402
    load_predictive_ids,
    load_records,
    rank_records,
)

# Every ``*.pkl`` filename literal a recorded script might open.
_PKL_LITERAL = re.compile(r"""["']([^"']*?\.pkl)["']""")

# GCS credential for cloud-image hypotheses. Recorded scripts follow the
# AutoDiscovery delivery-dir convention: they look this file up BY NAME next to
# the dataset pkl / in their cwd (or its parent) and export
# GOOGLE_APPLICATION_CREDENTIALS themselves; tensorstore then opens that path
# at the C++ level, bypassing the runner's Python ``open`` monkeypatch — so the
# file must genuinely exist wherever a script computes it. ``start_run``
# therefore links this token into each candidate location and presets the
# standard env vars as a default for scripts that don't set them.
GCS_TOKEN_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "allen-nd-goog-f5d46dbfa2cd.json"
)
# Older delivery markdowns and recorded scripts look the token up as
# ``zihan_gcs_token.json``. That key is rejected by Google since 2026-10-06, so
# the working token above is also exposed under the legacy name.
LEGACY_GCS_TOKEN_NAMES = ("zihan_gcs_token.json",)

# Companion files of the proofreader_evolve feature specifications
# (markdowns/{merge,split}_features_for_proofreader_evolve.md):
# ``rerank_companion_<brain>_<kind>.{npz,json}``. Recorded scripts open them by
# name. ``start_run`` links the companion bound to the pkl being run under every
# companion name the code mentions, so a script written for one brain reads the
# other brain's candidates when it is re-run there (``--companion``).
_COMPANION_NAME = re.compile(r"rerank_companion_[A-Za-z0-9-]+_(merge|split)\.(npz|json)")


def bind_companions(paths, pkls) -> dict:
    """Bind each companion (an npz/json pair) to the pkl its manifest names.

    Returns ``{pkl: [companion files]}`` for every pkl in ``pkls``. Raises
    ``ValueError`` for a malformed name, a missing half of a pair, a manifest
    whose ``source_cache.file`` matches none of the pkls, an npz that differs
    from its manifest digest, or two companions of one kind for one pkl.
    """
    by_name = {Path(p).name: Path(p) for p in pkls}
    bound = {Path(p): [] for p in pkls}
    seen = set()
    for path in paths:
        path = Path(path).resolve()
        match = _COMPANION_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Not a companion file name: {path.name}")
        stem = path.with_suffix("")
        if stem in seen:
            continue
        seen.add(stem)
        npz, manifest_path = stem.with_suffix(".npz"), stem.with_suffix(".json")
        for f in (npz, manifest_path):
            if not f.is_file():
                raise ValueError(f"Missing companion file: {f}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = (manifest.get("source_cache") or {}).get("file")
        if source not in by_name:
            raise ValueError(
                f"{manifest_path.name} belongs to {source!r}, which is not among "
                f"the dataset pkls {sorted(by_name)}"
            )
        if hashlib.sha256(npz.read_bytes()).hexdigest() != manifest.get("npz_sha256"):
            raise ValueError(f"{npz.name} does not match the digest in its manifest")
        target = bound[by_name[source]]
        kind = match.group(1)
        if any(_COMPANION_NAME.fullmatch(f.name).group(1) == kind for f in target):
            raise ValueError(f"Two {kind} companions for {source}")
        target += [npz, manifest_path]
    return bound


def companion_image(companions) -> tuple[str, str, dict] | None:
    """(bucket, dataset folder, level geometry) of the image a companion manifest names."""
    for path in companions:
        if Path(path).suffix != ".json":
            continue
        image = json.loads(Path(path).read_text(encoding="utf-8")).get("image") or {}
        uri = str(image.get("uri") or "")
        if not uri.startswith("s3://"):
            continue
        bucket, _, key = uri[len("s3://"):].partition("/")
        folder = key.split("/", 1)[0]
        if bucket and folder:
            return bucket, folder, image.get("levels") or {}
    return None


def retarget_image(code: str, origin_companions, extra_companions):
    """Code to run on an extra dataset, the folder rewrite made, and a refusal reason.

    Returns ``(code, None, None)`` when ``code`` does not name the origin image
    folder, ``(rewritten, "<origin> -> <extra>", None)`` when the extra image can
    stand in (same bucket and level geometry), else ``(code, None, reason)``.
    """
    origin = companion_image(origin_companions)
    if origin is None or origin[1] not in (code or ""):
        return code, None, None
    extra = companion_image(extra_companions)
    if extra is None:
        return code, None, (
            f"the script hardcodes the origin image folder {origin[1]} and no "
            "companion names the extra dataset's image"
        )
    if (extra[0], extra[2]) != (origin[0], origin[2]):
        return code, None, (
            f"the script hardcodes the origin image folder {origin[1]}, and the "
            f"extra image {extra[1]} differs in bucket or level geometry"
        )
    return code.replace(origin[1], extra[1]), f"{origin[1]} -> {extra[1]}", None


def companion_links(code: str, companions) -> dict:
    """Companion name -> bound file: each file's own name plus every companion
    name of the same kind and extension that ``code`` mentions."""
    files = {_COMPANION_NAME.fullmatch(Path(c).name).groups(): str(c) for c in companions}
    links = {Path(c).name: str(c) for c in companions}
    for match in _COMPANION_NAME.finditer(code or ""):
        target = files.get(match.groups())
        if target is not None:
            links[match.group(0)] = target
    return links

# Resident-payload execution. Historically every script ran in a fresh
# subprocess that re-deserialized the multi-GB dataset itself (~minutes per
# script; 42 scripts ≈ hours of pure pickle.load). The driver now loads each
# pkl ONCE and forks a child per script: copy-on-write shares the payload for
# reads while keeping any in-place mutation private to that child — the same
# isolation the subprocess gave, without the reload. ``--no-preload`` restores
# the old behavior; non-POSIX platforms fall back automatically.
_PRELOAD_ENABLED = hasattr(os, "fork")
_PAYLOAD_CACHE: dict[str, object] = {}


def _preload_payload(pkl: Path):
    """Load (once) and cache a dataset payload; ``None`` disables the fork path."""
    key = str(pkl.resolve())
    if key in _PAYLOAD_CACHE:
        return _PAYLOAD_CACHE[key]
    start = time.monotonic()
    print(f"[rerun] preloading dataset payload: {pkl}", file=sys.stderr, flush=True)
    try:
        import importlib
        try:
            importlib.import_module("agentic_neuron_proofreader")
        except ImportError:
            pass  # payloads without SkeletonGraph objects unpickle fine anyway
        with open(pkl, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        print(
            f"[rerun] preload failed ({type(exc).__name__}: {exc}); falling "
            "back to per-script subprocess loading.",
            file=sys.stderr, flush=True,
        )
        _PAYLOAD_CACHE[key] = None
        return None
    print(
        f"[rerun] payload resident in {time.monotonic() - start:.0f}s; scripts "
        "now fork from it instead of re-deserializing per script.",
        file=sys.stderr, flush=True,
    )
    _PAYLOAD_CACHE[key] = payload
    return payload
_UNUSABLE_OUTPUT_RE = re.compile(
    r"(?:^|\n)\s*(?:no dataset(?: files?)? found|dataset not found|"
    r"could not find (?:a )?dataset|record has no ['\"]code['\"] to execute)"
    r"[.!]?\s*(?:\n|$)",
    re.IGNORECASE,
)
# The specs ask for ``INPUTS READ ... image_patches=<read>/<requested>``; recorded
# scripts also write it free-form ("0 image patches read").
_ZERO_IMAGE_INPUTS_RE = re.compile(
    r"INPUTS READ\b[^\n]*?(?:image_patches\s*=\s*0\s*/\s*0*[1-9]|\b0 image patches read)",
    re.IGNORECASE,
)
# Exit-0 UNUSABLE reasons that a loading repair can fix.
_DATA_LOADING_REASONS = frozenset({"dataset-loading", "swallowed-input-error", "zero-inputs"})
_CHECKPOINT_SCHEMA_VERSION = 1
# v2 (2026-10-20): runs carry ``input_audit``, which v1 work items lack.
_RUNNER_VERSION = "work-items-v2"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_sha256(code: str) -> str:
    """Stable identity for the exact experiment code being executed."""
    return _sha256_text(code or "")


def file_fingerprint(path: Path) -> dict:
    """Cheap dataset identity suitable for invalidating local checkpoints.

    Dataset pkls can be many GB, so hashing their full contents on every resume
    would itself be expensive.  A resolved path plus size and nanosecond mtime
    detects replacement/editing while keeping resume startup constant-time.
    """
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def swallowed_inputs(audit) -> list[str]:
    """Audited input sources whose every access failed (absent files and keys aside)."""
    return sorted(
        source for source, counts in (audit or {}).items()
        if isinstance(counts, dict) and counts.get("failed") and not counts.get("ok")
    )


def result_usability(run: dict) -> tuple[str, str | None]:
    """Separate process success from whether an experiment produced a result."""
    if run.get("timed_out"):
        return "UNUSABLE", "timeout"
    if run.get("exitcode") != 0:
        return "UNUSABLE", "nonzero-exit"
    if swallowed_inputs(run.get("input_audit")):
        return "UNUSABLE", "swallowed-input-error"
    stdout = str(run.get("stdout") or "")
    if not stdout.strip():
        return "UNUSABLE", "empty-output"
    if _UNUSABLE_OUTPUT_RE.search(stdout):
        return "UNUSABLE", "dataset-loading"
    if _ZERO_IMAGE_INPUTS_RE.search(stdout):
        return "UNUSABLE", "zero-inputs"
    return "USABLE", None


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically replace a checkpoint so interruption never truncates it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _load_checkpoint(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {"schema_version": _CHECKPOINT_SCHEMA_VERSION, "work_items": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": _CHECKPOINT_SCHEMA_VERSION, "work_items": {}}
    if (
        payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
        or not isinstance(payload.get("work_items"), dict)
    ):
        return {"schema_version": _CHECKPOINT_SCHEMA_VERSION, "work_items": {}}
    return payload


@functools.lru_cache(maxsize=1)
def _environment_fingerprint() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "networkx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": {"executable": sys.executable, "version": sys.version},
        "packages": packages,
    }


def _work_key(
    *, run_sha256: str, record_id, code_hash: str, dataset: Path, timeout: int,
    companions=(),
) -> str:
    identity = {
        "runner_version": _RUNNER_VERSION,
        "environment": _environment_fingerprint(),
        "run_sha256": run_sha256,
        "record_id": str(record_id),
        "code_sha256": code_hash,
        "dataset": file_fingerprint(dataset),
        "timeout": timeout,
    }
    if companions:  # absent key keeps checkpoints written without companions valid
        identity["companions"] = [file_fingerprint(Path(c)) for c in companions]
    return _sha256_text(json.dumps(identity, sort_keys=True, separators=(",", ":")))


def _work_item(
    code: str, dataset: Path, timeout: int, *, run_sha256: str, record_id, companions=(),
) -> tuple[str, list[str], str]:
    """The code to execute (``COMPAT_REWRITES`` applied), its rewrites, and its key."""
    code, rewrites = apply_compat_rewrites(code)
    if rewrites:
        print(f"[runner] compat rewrite {', '.join(rewrites)} (id {record_id})",
              file=sys.stderr, flush=True)
    key = _work_key(
        run_sha256=run_sha256,
        record_id=record_id,
        code_hash=code_sha256(code),
        dataset=dataset,
        timeout=timeout,
        companions=companions,
    )
    return code, rewrites, key


def _checkpointed_run(checkpoint: dict, key: str) -> dict | None:
    """The stored run for ``key``, unless absent or an environment failure.

    A cached environment failure (missing package, broken native import) is run
    again rather than reused: the work key does not cover every package a script
    may import, so a package installed since then would otherwise never be seen.
    """
    cached = checkpoint["work_items"].get(key)
    if isinstance(cached, dict) and isinstance(cached.get("run"), dict) and not (
        _is_environment_failure(cached["run"])
    ):
        return cached["run"]
    return None


def _store_run(
    checkpoint: dict, checkpoint_path: Path | None, key: str, *, record_id, code: str,
    dataset: Path, run: dict,
) -> None:
    checkpoint["work_items"][key] = {
        "record_id": record_id,
        "code_sha256": code_sha256(code),
        "dataset": file_fingerprint(dataset),
        "run": run,
    }
    if checkpoint_path is not None:
        _atomic_write_json(checkpoint_path, checkpoint)


def _run_work_item(
    code: str,
    dataset: Path,
    timeout: int,
    *,
    run_sha256: str,
    record_id,
    checkpoint: dict,
    checkpoint_path: Path | None,
    companions=(),
) -> tuple[dict, bool]:
    """Run or reuse one code×dataset work item and persist it immediately.

    The code runs, and is keyed, with its ``COMPAT_REWRITES`` applied.
    """
    code, rewrites, key = _work_item(
        code, dataset, timeout, run_sha256=run_sha256, record_id=record_id,
        companions=companions,
    )
    cached = _checkpointed_run(checkpoint, key)
    if cached is not None:
        return cached, True
    run = _run_installing_missing(code, dataset, timeout, companions)
    run["compat_rewrites"] = rewrites
    _store_run(checkpoint, checkpoint_path, key, record_id=record_id, code=code,
               dataset=dataset, run=run)
    return run, False


def pkl_basenames(code: str) -> list[str]:
    """Distinct ``*.pkl`` basenames referenced as literals in a record's code."""
    names = {Path(m).name for m in _PKL_LITERAL.findall(code or "")}
    return sorted(n for n in names if n)


def hypo_filename(rid) -> str:
    """Stable per-record script filename for the export/code dir."""
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", str(rid))
    return f"hypo_{safe}.py"


# Guidance written into each export dir so the agent knows exactly what it may
# change (the data-loading/bootstrap) and what it must NOT (the analysis).
_REVISION_GUIDE = """\
# Rerun revision guide

Each `hypo_<id>.py` here is one hypothesis's recorded experiment script. The
reruns fail mostly because the script cannot LOCATE the dataset (hardcoded paths
/ globs / "dataset not found" gates), not for analysis reasons. Revise ONLY the
data-loading part of each script; keep the statistical ANALYSIS and what it
prints byte-for-byte identical so the rerun is a faithful reproduction.

You MAY change, near the top of each script:
- Replace the dataset search (hardcoded paths, `os.path.exists`, `glob.glob`,
  "dataset not found" gates) with a DIRECT load of the provided pkl:

      import os, pickle
      with open(os.environ["RERUN_PKL"], "rb") as f:
          payload = pickle.load(f)

  Keep the SAME variable name the rest of the script uses (e.g. `payload`,
  `data`) and the same downstream keys (`fragments_graph`, `gt_graph`,
  `gt_edge_error`, `gt_node_canonical_label`, `gt_merge_sites`, ...).
- Cloud-image credentials are ALSO driver-owned: the runner presets
  `GOOGLE_APPLICATION_CREDENTIALS` and `AWS_EC2_METADATA_DISABLED=true`, links
  the GCS token (`allen-nd-goog-f5d46dbfa2cd.json`, also under the legacy name
  `zihan_gcs_token.json`) into the
  script's cwd, the cwd's parent, and the pkl's directory, and exports its
  absolute path as `$RERUN_GCS_TOKEN`. If a script's credential search still
  fails, replace it with:

      os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["RERUN_GCS_TOKEN"]

  Do NOT hardcode any other credential path and do NOT weaken the analysis to
  avoid image access.
- A call or spec form that this environment's library version rejects (for
  example a tensorstore spec field it does not know) is a loading fault too:
  make the smallest compatible edit to that one call. The runner reports a read
  the script caught and survived as `swallowed-input-error` (or `zero-inputs`),
  with a `[runner] swallowed input error: <source>: <error>` line at the end of
  stderr; fix that cause. Never install, upgrade or vendor the library.

The ENVIRONMENT is owned by the driver, NOT by your script. Do NOT manage
packages or interpreters from inside the script:
- Do NOT `pip install` / `apt install` / `conda install` anything, and do NOT
  add or keep install retry loops — the runner already turns every install
  command into a logged no-op, so they only waste time. The host env is
  pre-provisioned with numpy, pandas, scipy, statsmodels, sklearn, networkx,
  matplotlib, tensorstore and the proofreader package; just `import` them.
- Do NOT touch NumPy: never `pip install numpy...`, never `del sys.modules[...]`
  to reload it, and NEVER put a numpy/site-packages/source directory on
  `sys.path` (e.g. `sys.path.insert(0, "/tmp/np2")`). Importing numpy from a
  source tree raises "you should not try to import numpy from its source
  directory" and breaks the whole script. Just `import numpy as np`.
- If an import genuinely fails, that is an ENVIRONMENT problem for the driver to
  fix (provision the package once, globally) — not something to patch per
  script. Leave it: the runner installs a missing package that is on its
  allowlist and runs the script again, and reports anything else as an
  environment failure. Never vendor a package into a directory of your own or
  add one to `sys.path`.

You MUST NOT change:
- The statistical test, its parameters, the sampling/grouping logic, the effect
  the script computes, or the lines it prints. The whole point is to re-measure
  the SAME analysis on the real data.

Print a clear `Loading dataset from: $RERUN_PKL` line so the log shows the fix
took effect. `MPLBACKEND=Agg` is already set (headless plotting is fine).
"""


def export_scripts(ranked: list[dict], pkl: Path, export_dir: Path) -> dict:
    """Write each ranked record's code as an editable script + manifest/guide.

    Returns a small summary dict for stdout. The agent edits the ``hypo_<id>.py``
    files (loading section only) and then a ``--code-dir`` rerun executes them.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "REVISION_GUIDE.md").write_text(_REVISION_GUIDE, encoding="utf-8")

    # An export directory represents exactly this selection. Remove only the
    # workflow-owned hypothesis scripts that are no longer selected; otherwise a
    # changed predictive manifest leaves loose stale code beside the new manifest.
    selected_names = {hypo_filename(record.get("id")) for record in ranked}
    for old_script in export_dir.glob("hypo_*.py"):
        if old_script.name not in selected_names:
            old_script.unlink()

    manifest = []
    for rank, r in enumerate(ranked, start=1):
        rid = r.get("id")
        fname = hypo_filename(rid)
        code = r.get("code") or ""
        (export_dir / fname).write_text(code, encoding="utf-8")
        manifest.append(
            {
                "rank": rank,
                "id": rid,
                "file": fname,
                "status": r.get("status", ""),
                "surprisal": round(r["_surprisal"], 4),
                "priority_score": round(r.get("_priority", 0.0), 4),
                "hypothesis": (r.get("hypothesis") or "").strip(),
                "pkl_basenames_in_code": pkl_basenames(code),
                "has_code": bool(code.strip()),
                "recorded_code_sha256": code_sha256(code),
            }
        )
    (export_dir / "MANIFEST.json").write_text(
        json.dumps(
            {"pkl": str(pkl), "n": len(manifest), "records": manifest},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "export_dir": str(export_dir),
        "n_exported": len(manifest),
        "guide": str(export_dir / "REVISION_GUIDE.md"),
        "manifest": str(export_dir / "MANIFEST.json"),
    }


def resolve_code(
    r: dict, code_dir: Path | None, corrected_dir: Path | None = None
) -> tuple[str, str]:
    """Return ``(code, source)`` for a record, by precedence.

    ``corrected_dir`` (the test-fixer's rewritten-analysis scripts) wins when it
    has a ``hypo_<id>.py`` (``source='corrected'``); else ``code_dir`` (the
    loading-revised scripts) is used (``source='revised'``); else the recorded
    JSON ``code`` (``source='recorded'``). This lets a corrected statistical test
    supersede the loading-only revision, which supersedes the raw recorded code.
    """
    rid = r.get("id")
    for d, src in ((corrected_dir, "corrected"), (code_dir, "revised")):
        if d is not None:
            f = d / hypo_filename(rid)
            if f.is_file():
                text = f.read_text(encoding="utf-8")
                if text.strip():
                    return text, src
    return r.get("code") or "", "recorded"


# Input audit, run in the script's process by both execution paths. A recorded
# script that catches a failed data read and carries on with no data exits 0 and
# looks usable (2026-10-19: new-merge-1 id 33 read 0 image patches because this
# environment's tensorstore rejected its S3 spec, and printed delta AP +0.00000).
# The audit counts every tensorstore open and read, np.load, npz array read and
# pickle.load per source, keeps absent files and keys apart (probing for them is
# normal), and writes the counts to $RERUN_INPUT_AUDIT at exit. numpy and
# tensorstore are hooked when the script imports them, so the parent, which
# forks every script, never imports tensorstore itself.
_INPUT_AUDIT = r'''
import atexit as _audit_atexit
import importlib.machinery as _audit_machinery
import io as _audit_io
import json as _audit_json
import os as _audit_os
import pickle as _audit_pickle
import sys as _audit_sys

_AUDIT = {}
_AUDIT_PATH = _audit_os.environ.get("RERUN_INPUT_AUDIT")
_AUDIT_PID = _audit_os.getpid()
_AUDIT_ABSENT = (FileNotFoundError, KeyError)


def _audit_note(source, exc=None):
    entry = _AUDIT.setdefault(source, {"ok": 0, "failed": 0, "absent": 0, "errors": []})
    if exc is None:
        entry["ok"] += 1
    elif isinstance(exc, _AUDIT_ABSENT):
        entry["absent"] += 1
    else:
        entry["failed"] += 1
        if len(entry["errors"]) < 3:
            first_line = (str(exc).splitlines() or [""])[0]
            entry["errors"].append("%s: %s" % (type(exc).__name__, first_line[:300]))


def _audit_basename(obj):
    obj = getattr(obj, "name", obj)
    try:
        return _audit_os.path.basename(_audit_os.fsdecode(obj))
    except TypeError:
        return type(obj).__name__


def _audited(fn, source):
    """``fn`` counting each call under ``source`` (a name, or a function of the call)."""
    def wrapper(*args, **kwargs):
        try:
            name = source(*args, **kwargs) if callable(source) else source
        except Exception:
            name = "?"
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:
            _audit_note(name, exc)
            raise
        _audit_note(name)
        return out
    wrapper.__wrapped__ = fn
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def _audit_numpy(np):
    np.load = _audited(np.load, lambda file, *a, **k: "np.load:" + _audit_basename(file))
    npz = getattr(getattr(np.lib, "npyio", None), "NpzFile", None)
    if npz is not None:
        npz.__getitem__ = _audited(
            npz.__getitem__,
            lambda self, key: "npz:%s[%s]" % (_audit_basename(getattr(self, "fid", None)), key),
        )


def _audit_tensorstore(ts):
    ts.open = _audited(ts.open, "tensorstore")
    ts.Future.result = _audited(ts.Future.result, "tensorstore")
    ts.TensorStore.__array__ = _audited(ts.TensorStore.__array__, "tensorstore")


_AUDIT_HOOKS = {"numpy": _audit_numpy, "tensorstore": _audit_tensorstore}


def _audit_hook(name, module):
    try:
        _AUDIT_HOOKS.pop(name)(module)
    except Exception as exc:  # the audit must never break the script
        print("[runner] input audit could not hook %s: %s" % (name, exc),
              file=_audit_sys.stderr, flush=True)


class _AuditFinder:
    """Hook a module of ``_AUDIT_HOOKS`` right after the script first imports it."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name not in _AUDIT_HOOKS:
            return None
        spec = _audit_machinery.PathFinder.find_spec(name, path)
        loader = getattr(spec, "loader", None)
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        exec_module = loader.exec_module

        def _exec_module(module):
            exec_module(module)
            _audit_hook(name, module)

        loader.exec_module = _exec_module
        return spec


def _audit_flush():
    if not _AUDIT_PATH or _audit_os.getpid() != _AUDIT_PID:
        return
    try:
        with _audit_io.open(_AUDIT_PATH, "w", encoding="utf-8") as handle:
            _audit_json.dump(_AUDIT, handle)
    except (OSError, TypeError, ValueError):
        pass


_audit_pickle.load = _audited(_audit_pickle.load, "pickle.load")
for _audit_name in list(_AUDIT_HOOKS):
    if _audit_name in _audit_sys.modules:
        _audit_hook(_audit_name, _audit_sys.modules[_audit_name])
if _AUDIT_HOOKS:
    _audit_sys.meta_path.insert(0, _AuditFinder)
_audit_atexit.register(_audit_flush)
'''


# The recorded scripts locate the dataset through a wide variety of hardcoded
# paths and relative globs (``../data/<name>.pkl``, ``../*_add.pkl``,
# ``**/*_add.pkl``, ``cache/*_add.pkl``, ``/data/...`` …) — there is no single
# path or env hook to set. So rather than trying to make every path-search
# *find* the provided pkl, we force the load: a small runner monkeypatches the
# handful of filesystem calls the scripts use to locate and open a ``.pkl`` so
# that ANY ``.pkl`` reference resolves to the provided dataset, whatever path
# logic the script runs. All 100 records in the run export load via
# ``pickle.load(open(<path>, "rb"))`` after an ``os.path.exists`` / ``glob.glob``
# search, which these patches cover.
_RUNNER = r'''
import builtins, glob, os, runpy, subprocess, sys

_PKL = os.environ["RERUN_PKL"]
import json as _json
_COMPANIONS = _json.loads(os.environ.get("RERUN_COMPANIONS") or "{}")


def _companion(x):
    """The bound companion file for a path whose basename is a linked companion name."""
    try:
        s = os.fspath(x)
    except TypeError:
        return None
    return _COMPANIONS.get(os.path.basename(s)) if isinstance(s, str) else None


# --- Driver owns the environment --------------------------------------------
# The host env is the single source of truth: it is pre-provisioned with the
# scientific stack the recorded scripts need (numpy, pandas, scipy, statsmodels,
# sklearn, networkx, matplotlib, tensorstore, the proofreader package). The
# recorded scripts, however, routinely try to `pip install` those at run time
# (sometimes in retry loops that hang until the timeout) or to SWAP the NumPy
# version mid-script — both mutate the interpreter environment unpredictably and
# were a real source of failures (a NumPy source tree shoved onto sys.path made
# `import numpy` fail for every corrected script in one run). So we neutralize
# package installs HERE, in the driver's runner, rather than trusting each
# script (or an agent) to get the bootstrap right: pip / apt / conda install
# commands become logged no-ops, and the script runs against the one known-good
# env. A package that is genuinely missing then surfaces as a loud import error
# (which the driver classifies as an environment failure, and which the runner
# itself repairs with --install-missing when the package is allowlisted), not as
# a silent version swap or a half-hour hang.
def _is_install_cmd(a):
    if isinstance(a, str):
        s = a.split()
    elif isinstance(a, (list, tuple)):
        s = [str(x) for x in a]
    else:
        return False
    if not s:
        return False
    prog = os.path.basename(s[0])
    if prog.startswith("pip") and "install" in s:
        return True
    if "pip" in s and "install" in s:          # python -m pip install ...
        return True
    if prog in ("apt", "apt-get", "conda", "mamba", "sudo") and "install" in s:
        return True
    return False


_orig_check_call = subprocess.check_call
def _check_call(a, *args, **kw):
    if _is_install_cmd(a):
        print("[runner] suppressed package install:", a, file=sys.stderr, flush=True)
        return 0
    return _orig_check_call(a, *args, **kw)
subprocess.check_call = _check_call

_orig_check_output = subprocess.check_output
def _check_output(a, *args, **kw):
    if _is_install_cmd(a):
        print("[runner] suppressed package install:", a, file=sys.stderr, flush=True)
        return "" if (kw.get("text") or kw.get("encoding")) else b""
    return _orig_check_output(a, *args, **kw)
subprocess.check_output = _check_output

_orig_run = subprocess.run
def _run(a, *args, **kw):
    if _is_install_cmd(a):
        print("[runner] suppressed package install:", a, file=sys.stderr, flush=True)
        return subprocess.CompletedProcess(a, 0, "" if kw.get("text") else b"",
                                           "" if kw.get("text") else b"")
    return _orig_run(a, *args, **kw)
subprocess.run = _run

_orig_popen = subprocess.Popen
class _Popen(_orig_popen):
    def __init__(self, a, *args, **kw):
        if _is_install_cmd(a):
            print("[runner] suppressed package install:", a, file=sys.stderr, flush=True)
            a = [sys.executable, "-c", "pass"]
        super().__init__(a, *args, **kw)
subprocess.Popen = _Popen


def _is_pkl(x):
    try:
        s = os.fspath(x)
    except TypeError:
        return False
    return isinstance(s, str) and s.endswith(".pkl")


# Any open() of a *.pkl path -> open the provided dataset instead.
_orig_open = builtins.open
def _open(file, *a, **k):
    return _orig_open(_PKL if _is_pkl(file) else (_companion(file) or file), *a, **k)
builtins.open = _open

# A *.pkl or companion path always "exists" so the script takes its load branch.
_orig_exists = os.path.exists
os.path.exists = lambda p: True if _is_pkl(p) or _companion(p) else _orig_exists(p)
_orig_isfile = os.path.isfile
os.path.isfile = lambda p: True if _is_pkl(p) or _companion(p) else _orig_isfile(p)

# Any glob for *.pkl resolves to exactly the provided dataset.
_orig_glob = glob.glob
def _glob(pathname, *a, **k):
    if isinstance(pathname, str) and pathname.endswith(".pkl"):
        return [_PKL]
    return _orig_glob(pathname, *a, **k)
glob.glob = _glob

# pathlib.Path(...).exists()/is_file() for *.pkl, in case a script uses it.
from pathlib import Path as _Path
_orig_p_exists, _orig_p_isfile = _Path.exists, _Path.is_file
_Path.exists = lambda self: True if str(self).endswith(".pkl") or _companion(self) else _orig_p_exists(self)
_Path.is_file = lambda self: True if str(self).endswith(".pkl") or _companion(self) else _orig_p_isfile(self)
''' + _INPUT_AUDIT + '''
runpy.run_path("experiment.py", run_name="__main__")
'''


def _link_gcs_token(dest_dir: Path, announce: bool = False) -> None:
    """Expose the GCS token in ``dest_dir`` under its current and legacy names."""
    for name in dict.fromkeys((GCS_TOKEN_PATH.name, *LEGACY_GCS_TOKEN_NAMES)):
        dest = dest_dir / name
        if dest.is_symlink():
            # Re-point stale links (e.g. to the revoked legacy key); never touch
            # regular files, which may be user-provided credentials.
            if os.path.realpath(dest) == os.path.realpath(GCS_TOKEN_PATH):
                continue
            try:
                dest.unlink()
            except OSError:
                continue
        elif dest.exists():
            continue
        try:
            dest.symlink_to(GCS_TOKEN_PATH)
        except OSError:
            try:
                shutil.copy2(GCS_TOKEN_PATH, dest)
            except OSError:
                continue  # best effort — the env-var default may still suffice
        if announce:
            print(f"[rerun] exposed GCS token at {dest}", file=sys.stderr, flush=True)


def _apply_inprocess_patches(payload):
    """Child-side twin of ``_RUNNER`` for the forked execution path.

    Applies the same neutralizations and ``*.pkl`` redirections as the
    subprocess runner (keep the two in sync), plus one preload shortcut:
    ``pickle.load`` on any ``*.pkl``-named file returns the parent's resident
    payload instead of deserializing the multi-GB cache again. Returns the
    input audit's flush, which the child must call itself because it leaves
    through ``os._exit`` and so never runs atexit handlers.
    """
    import builtins
    import glob as glob_module
    from pathlib import Path as _Path

    pkl_path = os.environ["RERUN_PKL"]
    companions = json.loads(os.environ.get("RERUN_COMPANIONS") or "{}")

    def _companion(x):
        try:
            s = os.fspath(x)
        except TypeError:
            return None
        return companions.get(os.path.basename(s)) if isinstance(s, str) else None

    def _is_install_cmd(a):
        if isinstance(a, str):
            s = a.split()
        elif isinstance(a, (list, tuple)):
            s = [str(x) for x in a]
        else:
            return False
        if not s:
            return False
        prog = os.path.basename(s[0])
        if prog.startswith("pip") and "install" in s:
            return True
        if "pip" in s and "install" in s:
            return True
        if prog in ("apt", "apt-get", "conda", "mamba", "sudo") and "install" in s:
            return True
        return False

    orig_check_call = subprocess.check_call

    def _check_call(a, *args, **kw):
        if _is_install_cmd(a):
            print("[runner] suppressed package install:", a,
                  file=sys.stderr, flush=True)
            return 0
        return orig_check_call(a, *args, **kw)

    subprocess.check_call = _check_call

    orig_check_output = subprocess.check_output

    def _check_output(a, *args, **kw):
        if _is_install_cmd(a):
            print("[runner] suppressed package install:", a,
                  file=sys.stderr, flush=True)
            return "" if (kw.get("text") or kw.get("encoding")) else b""
        return orig_check_output(a, *args, **kw)

    subprocess.check_output = _check_output

    orig_run = subprocess.run

    def _run(a, *args, **kw):
        if _is_install_cmd(a):
            print("[runner] suppressed package install:", a,
                  file=sys.stderr, flush=True)
            return subprocess.CompletedProcess(
                a, 0, "" if kw.get("text") else b"", "" if kw.get("text") else b"")
        return orig_run(a, *args, **kw)

    subprocess.run = _run

    orig_popen = subprocess.Popen

    class _Popen(orig_popen):
        def __init__(self, a, *args, **kw):
            if _is_install_cmd(a):
                print("[runner] suppressed package install:", a,
                      file=sys.stderr, flush=True)
                a = [sys.executable, "-c", "pass"]
            super().__init__(a, *args, **kw)

    subprocess.Popen = _Popen

    def _is_pkl(x):
        try:
            s = os.fspath(x)
        except TypeError:
            return False
        return isinstance(s, str) and s.endswith(".pkl")

    orig_open = builtins.open

    def _open(file, *a, **k):
        return orig_open(pkl_path if _is_pkl(file) else (_companion(file) or file), *a, **k)

    builtins.open = _open

    orig_exists = os.path.exists
    os.path.exists = lambda p: True if _is_pkl(p) or _companion(p) else orig_exists(p)
    orig_isfile = os.path.isfile
    os.path.isfile = lambda p: True if _is_pkl(p) or _companion(p) else orig_isfile(p)

    orig_glob = glob_module.glob

    def _glob(pathname, *a, **k):
        if isinstance(pathname, str) and pathname.endswith(".pkl"):
            return [pkl_path]
        return orig_glob(pathname, *a, **k)

    glob_module.glob = _glob

    orig_p_exists, orig_p_isfile = _Path.exists, _Path.is_file
    _Path.exists = lambda self: True if str(self).endswith(".pkl") or _companion(self) else orig_p_exists(self)
    _Path.is_file = lambda self: True if str(self).endswith(".pkl") or _companion(self) else orig_p_isfile(self)

    orig_pickle_load = pickle.load

    def _pickle_load(file, *a, **k):
        name = getattr(file, "name", "")
        if isinstance(name, str) and name.endswith(".pkl"):
            return payload
        return orig_pickle_load(file, *a, **k)

    pickle.load = _pickle_load

    audit = {"__name__": "_rerun_input_audit"}
    exec(_INPUT_AUDIT, audit)
    return audit["_audit_flush"]


_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")
_MEMORY_SAMPLE_S = 2.0  # how often a running child's private memory is read


def _limit_threads(threads: int) -> None:
    """Cap the native thread pools already loaded in this process.

    A forked child inherits libraries the parent loaded, which read their
    thread-count env vars at load time, so only threadpoolctl reaches them. The
    limits hold until restored explicitly, which this process never does.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(limits=threads)


def _fork_child(run_dir: Path, env: dict, payload, threads: int | None) -> int:
    """Fork a child that runs ``experiment.py`` against the resident payload.

    fork gives the child a copy-on-write view of the parent's memory: the
    multi-GB payload is shared for reads and any in-place mutation stays
    private to that child — the same isolation the per-script subprocess gave,
    without a fresh deserialization per script. The child leads its own
    session so a timeout kills its whole process group. Returns the child pid.
    """
    stdout_path = run_dir.parent / "stdout.txt"
    stderr_path = run_dir.parent / "stderr.txt"
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:  # child — must never return into the parent's control flow
        exit_code = 1
        flush_audit = None
        try:
            os.setsid()
            out_fd = os.open(str(stdout_path),
                             os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            err_fd = os.open(str(stderr_path),
                             os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            # Fresh streams on fds 1 and 2, in case the parent's sys.stdout or
            # sys.stderr is not a file on them (redirected, as in tests).
            sys.stdout = open(1, "w", closefd=False)
            sys.stderr = open(2, "w", errors="backslashreplace", closefd=False)
            os.chdir(str(run_dir))
            os.environ.clear()
            os.environ.update(env)
            sys.argv = ["experiment.py"]
            flush_audit = _apply_inprocess_patches(payload)
            if threads:
                _limit_threads(threads)
            import runpy
            runpy.run_path("experiment.py", run_name="__main__")
            exit_code = 0
        except SystemExit as exc:
            code = exc.code
            if code in (None, 0):
                exit_code = 0
            else:
                exit_code = code if isinstance(code, int) else 1
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        finally:
            try:
                if flush_audit is not None:
                    flush_audit()
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(exit_code)
    return pid


def _private_bytes(pid: int) -> int | None:
    """Unshared resident memory of ``pid`` (copy-on-write copies included)."""
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except OSError:
        return None
    return sum(
        int(line.split()[1]) * 1024
        for line in text.splitlines()
        if line.startswith(("Private_Clean:", "Private_Dirty:"))
    )


class RunHandle:
    """One started script; ``poll()`` returns its run dict once it has ended.

    The sandbox lives until then. A child past its deadline is killed with its
    whole process group. While it runs, its private memory is sampled every
    ``_MEMORY_SAMPLE_S`` seconds into ``peak_private_bytes``.
    """

    def __init__(self, sandbox, pid: int, proc, timeout: int, start: float) -> None:
        self._sandbox = sandbox
        self._root = Path(sandbox.name)
        self._pid = pid
        self._proc = proc  # the Popen on the subprocess path, else None (forked)
        self._timeout = timeout
        self._start = start
        self._next_sample = start
        self.peak_private_bytes = 0

    def _reap(self) -> int | None:
        """The exit code (negative signal number) once the child has ended."""
        if self._proc is not None:
            return self._proc.poll()
        done_pid, status = os.waitpid(self._pid, os.WNOHANG)
        if done_pid != self._pid:
            return None
        if os.WIFSIGNALED(status):
            return -os.WTERMSIG(status)
        return os.WEXITSTATUS(status)

    def _kill(self) -> None:
        try:
            os.killpg(self._pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(self._pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._proc is not None:
            self._proc.wait()
            return
        try:
            os.waitpid(self._pid, 0)
        except ChildProcessError:
            pass

    def poll(self) -> dict | None:
        exitcode = self._reap()
        now = time.monotonic()
        if exitcode is None:
            if now < self._start + self._timeout:
                if now >= self._next_sample:
                    sample = _private_bytes(self._pid) or 0
                    self.peak_private_bytes = max(self.peak_private_bytes, sample)
                    self._next_sample = now + _MEMORY_SAMPLE_S
                return None
            self._kill()
        return self._finish(timed_out=exitcode is None, exitcode=exitcode)

    def wait(self) -> dict:
        while (run := self.poll()) is None:
            time.sleep(0.05)
        return run

    def _finish(self, *, timed_out: bool, exitcode: int | None) -> dict:
        def _read(name: str) -> str:
            try:
                return (self._root / name).read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""

        stderr_text = _read("stderr.txt")
        if timed_out:
            stderr_text += f"\n[rerun] timed out after {self._timeout}s"
        run = _with_input_audit({
            "timed_out": timed_out,
            "exitcode": None if timed_out else exitcode,
            "stdout": _read("stdout.txt"),
            "stderr": stderr_text,
            "runtime_ms": round((time.monotonic() - self._start) * 1000),
        }, self._root / "input_audit.json")
        self._sandbox.cleanup()
        return run


def start_run(
    code: str, pkl: Path, timeout: int, companions=(), *, threads: int | None = None,
) -> RunHandle:
    """Start one record's ``code`` with every pkl load forced to ``pkl``.

    The recorded script is written verbatim to ``experiment.py`` and launched
    through ``_runner.py``, which monkeypatches ``open`` / ``os.path.exists`` /
    ``glob.glob`` (and the ``pathlib`` equivalents) so the script's own
    path-finding always resolves to the provided dataset (passed via the
    ``RERUN_PKL`` env var) regardless of the hardcoded paths or globs it uses.

    Cloud credentials get the same driver-owned treatment: when the repo token
    at ``GCS_TOKEN_PATH`` exists, it is linked under its current and legacy names into
    the script's cwd, the cwd's parent, and the pkl's directory (the places the
    recorded delivery-dir conventions look), and the standard
    ``GOOGLE_APPLICATION_CREDENTIALS`` / ``AWS_EC2_METADATA_DISABLED`` env vars
    are preset so image-reading hypotheses can reach GCS / public S3.

    Companion files bound to ``pkl`` (``--companion``) are linked into the cwd
    under their own names and under every companion name the code mentions
    (``companion_links``); opens of those names elsewhere are redirected too.

    Execution strategy: when the payload preloads (POSIX, ``--no-preload`` not
    given), the script runs in a forked child sharing the resident payload via
    copy-on-write and ``pickle.load`` short-circuits to it — one dataset
    deserialization per invocation instead of one per script. Otherwise it
    falls back to the original fresh-subprocess-per-script path. Either way the
    run carries the script's ``input_audit``. ``threads`` caps the script's
    BLAS/OpenMP thread pools (``--jobs`` above 1).
    """
    payload = _preload_payload(pkl) if _PRELOAD_ENABLED else None
    sandbox = tempfile.TemporaryDirectory(prefix="rerun_")
    try:
        tmp = Path(sandbox.name)
        # The cwd is nested one level so scripts probing ``Path.cwd().parent``
        # still land inside the disposable sandbox (where a token link can
        # live) instead of the shared $TMPDIR.
        run = tmp / "run"
        run.mkdir()
        (run / "experiment.py").write_text(code, encoding="utf-8")
        env = {
            **os.environ,
            "MPLBACKEND": "Agg",  # headless plotting
            "RERUN_PKL": str(pkl),
            "RERUN_INPUT_AUDIT": str(tmp / "input_audit.json"),
        }
        env.setdefault("AWS_EC2_METADATA_DISABLED", "true")
        if threads:
            env.update({name: str(threads) for name in _THREAD_ENV})
        links = companion_links(code, companions)
        if links:
            env["RERUN_COMPANIONS"] = json.dumps(links)
            for name, target in links.items():
                (run / name).symlink_to(target)
        if GCS_TOKEN_PATH.is_file():
            env.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(GCS_TOKEN_PATH))
            env["RERUN_GCS_TOKEN"] = str(GCS_TOKEN_PATH)
            _link_gcs_token(run)
            _link_gcs_token(tmp)
            _link_gcs_token(pkl.parent, announce=True)
        start = time.monotonic()
        if payload is not None:
            pid = _fork_child(run, env, payload, threads)
            return RunHandle(sandbox, pid, None, timeout, start)
        (run / "_runner.py").write_text(_RUNNER, encoding="utf-8")
        with open(tmp / "stdout.txt", "wb") as out, open(tmp / "stderr.txt", "wb") as err:
            proc = subprocess.Popen(
                [sys.executable, "_runner.py"],
                cwd=str(run),
                stdout=out,
                stderr=err,
                env=env,
                start_new_session=True,
            )
        return RunHandle(sandbox, proc.pid, proc, timeout, start)
    except BaseException:
        sandbox.cleanup()
        raise


def rerun_one(code: str, pkl: Path, timeout: int, companions=()) -> dict:
    """Run one record's ``code`` against ``pkl`` to completion (see ``start_run``)."""
    return start_run(code, pkl, timeout, companions).wait()


def _with_input_audit(run: dict, audit_path: Path) -> dict:
    """Attach the script's input audit; name each swallowed input failure in stderr."""
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        audit = {}
    run["input_audit"] = audit if isinstance(audit, dict) else {}
    if run.get("exitcode") == 0:
        for source in swallowed_inputs(run["input_audit"]):
            error = (run["input_audit"][source].get("errors") or ["unknown error"])[0]
            run["stderr"] = (
                (run.get("stderr") or "").rstrip("\n")
                + f"\n[runner] swallowed input error: {source}: {error}"
            )
    return run


def truncate(text: str | None, limit: int) -> str:
    """Trim long captured output to keep the stdout payload manageable."""
    text = text or ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n…[{len(text) - limit} chars omitted]…\n{tail}"


# A failure is an ENVIRONMENT failure (the interpreter could not import the
# scientific stack) rather than an ANALYSIS failure (the script ran but the
# statistics/data logic raised, or a "dataset not found" gate exited) when its
# stderr carries an import/native-load signature. We classify these distinctly so
# a broken env can NEVER again masquerade as a batch of INCONCLUSIVE statistical
# verdicts: the driver aborts on a systematic env failure instead of letting an
# agent fold per-hypothesis "could not run" calls. The patterns are deliberately
# narrow — a "dataset not found" gate or a genuine analysis exception is NOT an
# environment failure (the reproducer is meant to fix loading and rerun those).
_ENV_FAILURE_RE = re.compile(
    r"ModuleNotFoundError"
    r"|ImportError"
    r"|cannot import name"
    r"|you should not try to import numpy from its source directory"
    r"|numpy\._core"
    r"|_multiarray_umath"
    r"|undefined symbol"
    r"|DLL load failed"
    r"|cannot open shared object file"
    r"|\.so: cannot open",
    re.IGNORECASE,
)


def _is_environment_failure(run: dict) -> bool:
    return (
        not run.get("timed_out")
        and run.get("exitcode") not in (0, None)
        and classify_failure(run.get("stderr")) == "environment"
    )


def classify_failure(stderr: str | None) -> str | None:
    """Tag a non-zero run as ``"environment"`` vs ``"analysis"`` (None if clean).

    Used both per-result (so the agent sees why a script failed) and to decide
    whether a whole compute step is a systematic environment failure worth
    aborting on, rather than folding as statistical verdicts.
    """
    if not stderr:
        return "analysis"
    return "environment" if _ENV_FAILURE_RE.search(stderr) else "analysis"


def run_outcome(run: dict) -> tuple[str, str | None, str, str | None]:
    """(log outcome, failure kind, ``result_status``, ``result_failure_reason``) of a run."""
    usability, reason = result_usability(run)
    if run.get("timed_out"):
        return "TIMEOUT", None, usability, reason
    if run.get("exitcode") == 0:
        if usability == "USABLE":
            return "OK", None, usability, reason
        kind = "data-loading" if reason in _DATA_LOADING_REASONS else "analysis"
        return f"UNUSABLE({reason})", kind, usability, reason
    kind = classify_failure(run.get("stderr"))
    label = "ENV-FAILED" if kind == "environment" else "FAILED"
    return f"{label}(exit={run.get('exitcode')})", kind, usability, reason


# --- Missing packages ---------------------------------------------------------
# Recorded scripts import whatever the AutoDiscovery sandbox had installed. With
# --install-missing, a run that fails because a top-level module listed here is
# absent gets its package installed into this interpreter's environment (once per
# invocation) and is run again. Every installed distribution is passed to pip as a
# constraint, so an install may add packages but never change one: numpy, scipy
# and the rest of the checkpoint fingerprint stay as they are, and an install that
# would need a different version fails, leaving the environment failure in place.
# Only names in this table are fetched, so a module name invented by recorded code
# never reaches PyPI. To allow another package, add its import name -> pip name.
INSTALLABLE_MODULES: dict[str, str] = {
    "catboost": "catboost",
    "community": "python-louvain",
    "cv2": "opencv-python-headless",
    "h5py": "h5py",
    "hdbscan": "hdbscan",
    "hmmlearn": "hmmlearn",
    "igraph": "igraph",
    "imblearn": "imbalanced-learn",
    "joblib": "joblib",
    "kneed": "kneed",
    "leidenalg": "leidenalg",
    "lifelines": "lifelines",
    "lightgbm": "lightgbm",
    "numba": "numba",
    "optuna": "optuna",
    "pingouin": "pingouin",
    "plotly": "plotly",
    "polars": "polars",
    "psutil": "psutil",
    "pyarrow": "pyarrow",
    "pygam": "pygam",
    "pynndescent": "pynndescent",
    "pywt": "PyWavelets",
    "ripser": "ripser",
    "seaborn": "seaborn",
    "shap": "shap",
    "skimage": "scikit-image",
    "tifffile": "tifffile",
    "tqdm": "tqdm",
    "umap": "umap-learn",
    "xgboost": "xgboost",
    "zarr": "zarr",
}
PIP_TIMEOUT_S: int = 900
_MISSING_MODULE_RE = re.compile(r"No module named '([A-Za-z_][\w.]*)'")
_INSTALL_MISSING = False          # set from --install-missing in main()
_INSTALLED: dict[str, dict] = {}  # module -> install record, this invocation


def missing_module(stderr: str | None) -> str | None:
    """The absent top-level module a run died on, or None.

    The last ``No module named`` wins (earlier ones may be caught fallbacks). A
    missing submodule of an installed package (``numpy._core``) is a version or
    ABI problem, not an absent package, so it gives None.
    """
    for name in reversed(_MISSING_MODULE_RE.findall(stderr or "")):
        top = name.split(".")[0]
        try:
            if importlib.util.find_spec(top) is None:
                return top
        except (ImportError, ValueError):
            return top
    return None


def _installed_pins() -> list[str]:
    """``name==version`` for every distribution visible to this interpreter."""
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:  # pragma: no cover - pip environments ship packaging
        Version = None
    pins = {}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").strip()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if not name or key in pins:  # the first on sys.path is the one imported
            continue
        if Version is not None:
            try:
                Version(dist.version)
            except InvalidVersion:
                continue
        pins[key] = f"{name}=={dist.version}"
    return sorted(pins.values())


def _pip_install(package: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        constraints = Path(tmp) / "constraints.txt"
        constraints.write_text("\n".join(_installed_pins()) + "\n", encoding="utf-8")
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
               "--no-input", "--prefer-binary", "-c", str(constraints), package]
        try:
            # Two workflows sharing one environment must not run pip at once.
            with open(Path(sys.prefix) / ".rerun-install.lock", "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=PIP_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    importlib.invalidate_caches()
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return {"error": f"pip exit {proc.returncode}: " + " | ".join(tail)}
    try:
        return {"version": importlib.metadata.version(package)}
    except importlib.metadata.PackageNotFoundError:
        return {"version": None}


def install_missing(module: str) -> dict:
    """Install the package providing ``module`` once per invocation; return the record."""
    if module not in _INSTALLED:
        package = INSTALLABLE_MODULES.get(module)
        record = {"module": module, "package": package}
        record.update(
            _pip_install(package) if package else
            {"error": "not in INSTALLABLE_MODULES (agentic/rerun_experiments.py)"}
        )
        _INSTALLED[module] = record
        outcome = (f"installed {package}=={record.get('version')}"
                   if "error" not in record else f"not installed: {record['error']}")
        print(f"[runner] missing module {module!r}: {outcome}", file=sys.stderr, flush=True)
    return _INSTALLED[module]


def _run_installing_missing(code: str, dataset: Path, timeout: int, companions=()) -> dict:
    """Run once; while it dies on an installable absent module, install it and rerun."""
    run = rerun_one(code, dataset, timeout, companions)
    tried = set()
    while _INSTALL_MISSING and _is_environment_failure(run):
        module = missing_module(run.get("stderr"))
        if module is None or module in tried or "error" in install_missing(module):
            break
        tried.add(module)
        run = rerun_one(code, dataset, timeout, companions)
    return run


# --- Library version skew -----------------------------------------------------
# Recorded scripts were written against the AutoDiscovery sandbox's library
# versions. A form this environment rejects is repaired script by script by the
# loading-repair agents; one that recurs gets a deterministic source rewrite here.
# Each entry's probe is Python source, run once per invocation in a fresh
# interpreter (this process forks every script, so it never imports the
# library), that exits 0 only when the installed version rejects the original
# form and accepts the replacement; upgrading the library turns the rewrite off
# by itself. The work key
# hashes the rewritten code, while ``executed_code_sha256`` stays on the script.
class CompatRewrite(NamedTuple):
    name: str
    pattern: re.Pattern
    replacement: str
    probe: str


COMPAT_REWRITES: tuple[CompatRewrite, ...] = (
    # Newer tensorstore names the S3 credential source with "type"; panda's
    # 0.1.71 accepts only {"anonymous": true} (new-merge-1 id 33, 2026-10-19).
    CompatRewrite(
        name="tensorstore-anonymous-credentials",
        pattern=re.compile(
            r"""(aws_credentials['"]?\s*[:=]\s*)\{\s*(['"])type\2\s*:\s*(['"])anonymous\3\s*,?\s*\}"""
        ),
        replacement=r"\1{\2anonymous\2: True}",
        probe=(
            "import sys\n"
            "import tensorstore as ts\n"
            "def spec(credentials):\n"
            "    return {'driver': 'zarr', 'kvstore': {'driver': 's3',\n"
            "            'bucket': 'aind-open-data', 'path': 'p/',\n"
            "            'aws_credentials': credentials}}\n"
            "ts.Spec(spec({'anonymous': True}))\n"
            "try:\n"
            "    ts.Spec(spec({'type': 'anonymous'}))\n"
            "except ValueError:\n"
            "    sys.exit(0)\n"
            "sys.exit(1)\n"
        ),
    ),
)
COMPAT_PROBE_TIMEOUT_S: int = 120


@functools.lru_cache(maxsize=None)
def _compat_needed(rule: CompatRewrite) -> bool:
    """Whether this interpreter rejects the form ``rule`` rewrites (and takes its replacement)."""
    try:
        proc = subprocess.run([sys.executable, "-c", rule.probe], capture_output=True,
                              timeout=COMPAT_PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def apply_compat_rewrites(code: str) -> tuple[str, list[str]]:
    """``code`` with every needed ``COMPAT_REWRITES`` entry applied, and their names."""
    applied = []
    for rule in COMPAT_REWRITES:
        if rule.pattern.search(code or "") and _compat_needed(rule):
            code = rule.pattern.sub(rule.replacement, code)
            applied.append(rule.name)
    return code, applied


# --- Parallel execution (--jobs) ----------------------------------------------
# Scripts are independent: each runs in its own sandbox against a copy-on-write
# view of the payload, and only this process writes the checkpoint. With --jobs N
# above 1, up to N run at once, longest predicted runtime first. A new script
# starts only when the job's cgroup can still hold it: the reserve is 1.25 x the
# largest private memory a finished script used (this process's RSS before any
# has finished, the worst case of a script copying the whole payload). A script
# the kernel killed (SIGKILL the runner did not send, e.g. the OOM killer) is
# discarded and run again alone once the others have finished, so a crowded
# node never turns into a recorded failure. Each script gets cpus // N BLAS and
# OpenMP threads. --jobs 1 runs the items one at a time, as before.
_GiB = 1 << 30
_MIN_RESERVE_BYTES = 2 * _GiB


def _memory_headroom(
    cgroup_file: str = "/proc/self/cgroup",
    cgroup_root: str = "/sys/fs/cgroup",
    meminfo: str = "/proc/meminfo",
) -> int | None:
    """Bytes this process may still allocate, or None when unknown.

    The tightest cgroup v2 ancestor with a finite ``memory.max`` decides (a Slurm
    job sets its limit on the job cgroup, above the step and task ones); its
    inactive file cache counts as free. Without one, ``MemAvailable``.
    """
    room = None
    try:
        rel = next(line.split("::", 1)[1].strip()
                   for line in Path(cgroup_file).read_text().splitlines()
                   if line.startswith("0::"))
    except (OSError, StopIteration):
        rel = None
    if rel is not None:
        root = Path(cgroup_root)
        node = root / rel.lstrip("/")
        while True:
            try:
                limit = (node / "memory.max").read_text().strip()
                if limit != "max":
                    used = int((node / "memory.current").read_text())
                    stat = dict(line.split()[:2] for line in
                                (node / "memory.stat").read_text().splitlines())
                    free = int(limit) - used + int(stat.get("inactive_file", 0))
                    room = free if room is None else min(room, free)
            except (OSError, ValueError):
                pass
            if node == root or node == node.parent:
                break
            node = node.parent
    if room is not None:
        return room
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _rss_bytes() -> int:
    """This process's resident memory."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def _log_done(entry: dict) -> None:
    run = entry["run"]
    print(f"{entry['label']} {run_outcome(run)[0]} in {run['runtime_ms'] / 1000:.0f}s",
          file=sys.stderr, flush=True)


def execute_work_items(
    work: list[dict], *, jobs: int, timeout: int, run_sha256: str, checkpoint: dict,
    checkpoint_path: Path | None,
) -> None:
    """Fill each work entry's ``run`` and ``reused``.

    An entry holds ``code``, ``dataset``, ``companions``, ``record_id``, a log
    ``label`` and an optional ``predicted_s``. One job runs the entries one at a
    time in order, as before; more go through ``_run_pool``.
    """
    if jobs > 1:
        _run_pool(work, jobs=jobs, timeout=timeout, run_sha256=run_sha256,
                  checkpoint=checkpoint, checkpoint_path=checkpoint_path)
        return
    for entry in work:
        print(f"{entry['label']} …", file=sys.stderr, flush=True)
        entry["run"], entry["reused"] = _run_work_item(
            entry["code"], entry["dataset"], timeout, run_sha256=run_sha256,
            record_id=entry["record_id"], checkpoint=checkpoint,
            checkpoint_path=checkpoint_path, companions=entry["companions"],
        )
        _log_done(entry)


def _run_pool(
    work: list[dict], *, jobs: int, timeout: int, run_sha256: str, checkpoint: dict,
    checkpoint_path: Path | None,
) -> None:
    """Run the entries up to ``jobs`` at a time (see the section comment)."""
    pending = []
    for entry in work:
        code, rewrites, key = _work_item(
            entry["code"], entry["dataset"], timeout, run_sha256=run_sha256,
            record_id=entry["record_id"], companions=entry["companions"],
        )
        cached = _checkpointed_run(checkpoint, key)
        if cached is not None:
            entry["run"], entry["reused"] = cached, True
            _log_done(entry)
            continue
        pending.append({"entry": entry, "code": code, "rewrites": rewrites, "key": key,
                        "tried": set(), "alone": False})
    if not pending:
        return
    # Longest predicted runtime first; entries without a prediction keep their order.
    queue = sorted(pending, key=lambda item: (item["entry"].get("predicted_s") is None,
                                              -(item["entry"].get("predicted_s") or 0)))
    threads = max(1, _cpu_count() // jobs)
    first_reserve = max(_rss_bytes(), _MIN_RESERVE_BYTES)
    headroom = _memory_headroom()
    print(
        f"[rerun] jobs {jobs} ({threads} thread(s) per script) for {len(queue)} "
        f"script(s); memory headroom "
        f"{'unknown' if headroom is None else f'{headroom / _GiB:.1f} GB'}, "
        f"first reserve {first_reserve / _GiB:.1f} GB",
        file=sys.stderr, flush=True,
    )
    largest = 0  # largest private memory of a finished script
    running: list[tuple[dict, RunHandle]] = []
    alone: list[dict] = []
    exclusive = False

    def admit() -> bool:
        if not running:
            return True
        room = _memory_headroom()
        reserve = max(int(1.25 * largest), _MIN_RESERVE_BYTES) if largest else first_reserve
        return room is None or room >= reserve

    def start(item: dict, cap: int | None, note: str = "") -> None:
        entry = item["entry"]
        print(f"{entry['label']} …{note}", file=sys.stderr, flush=True)
        handle = start_run(item["code"], entry["dataset"], timeout, entry["companions"],
                           threads=cap)
        running.append((item, handle))

    while queue or running or alone:
        while queue and not exclusive and len(running) < jobs and admit():
            start(queue.pop(0), threads)
        if not running and alone:
            exclusive = True
            start(alone.pop(0), None, " (alone)")
        finished = False
        for pair in list(running):
            item, handle = pair
            run = handle.poll()
            if run is None:
                continue
            finished = True
            running.remove(pair)
            exclusive = False
            largest = max(largest, handle.peak_private_bytes)
            entry = item["entry"]
            if _INSTALL_MISSING and _is_environment_failure(run):
                module = missing_module(run.get("stderr"))
                if (module is not None and module not in item["tried"]
                        and "error" not in install_missing(module)):
                    item["tried"].add(module)
                    queue.insert(0, item)
                    continue
            if (run.get("exitcode") == -signal.SIGKILL and not run.get("timed_out")
                    and not item["alone"]):
                item["alone"] = True
                alone.append(item)
                print(f"{entry['label']} killed (SIGKILL) while {len(running)} other "
                      "script(s) ran; requeued alone", file=sys.stderr, flush=True)
                continue
            run["compat_rewrites"] = item["rewrites"]
            _store_run(checkpoint, checkpoint_path, item["key"],
                       record_id=entry["record_id"], code=item["code"],
                       dataset=entry["dataset"], run=run)
            entry["run"], entry["reused"] = run, False
            _log_done(entry)
        if not finished:
            time.sleep(0.05)
    requeued = [item["entry"]["record_id"] for item in pending if item["alone"]]
    print(
        f"[rerun] pool finished {len(pending)} script(s); largest script private "
        f"memory {largest / _GiB:.1f} GB; run alone after a kill: "
        f"{requeued or 'none'}",
        file=sys.stderr, flush=True,
    )


def preflight_env(python: str, *, timeout: int = 60) -> tuple[bool, str]:
    """Cheap one-time check that the run interpreter can import the core stack.

    Catches a broken base environment ONCE, up front, with a clear message —
    instead of discovering it as N identical per-script import failures after
    spending time launching each. Returns ``(ok, detail)``; ``detail`` is the
    captured error when not ok. The recorded scripts also use statsmodels for the
    corrected tests, so it is included.
    """
    probe = (
        "import importlib, sys\n"
        "mods = ['numpy', 'pandas', 'scipy', 'statsmodels']\n"
        "bad = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as e:\n"
        "        bad.append('%s: %s: %s' % (m, type(e).__name__, e))\n"
        "if bad:\n"
        "    sys.stderr.write('\\n'.join(bad))\n"
        "    sys.exit(1)\n"
        "import numpy\n"
        "print(numpy.__version__)\n"
    )
    try:
        proc = subprocess.run(
            [python, "-c", probe],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"preflight could not launch the interpreter: {e}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "unknown import error").strip()
    return True, (proc.stdout or "").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file", type=Path, help="The run JSON export to re-execute."
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        required=True,
        help="The ORIGIN dataset .pkl this run's experiments load.",
    )
    parser.add_argument(
        "--extra-pkl",
        type=Path,
        action="extend",
        nargs="+",
        default=[],
        metavar="PKL",
        help=(
            "Other dataset .pkl(s) to EXTRAPOLATE onto. Accepts several paths "
            "after one flag and/or the flag repeated. The same code is re-run "
            "with its load redirected to each, to test generalization. Assumes "
            "the same payload structure as --pkl."
        ),
    )
    parser.add_argument(
        "--companion",
        type=Path,
        action="extend",
        nargs="+",
        default=[],
        metavar="FILE",
        help=(
            "Companion files (rerank_companion_<brain>_<kind>.npz or .json; the "
            "other half of the pair is found next to it). Each is bound to the "
            "--pkl / --extra-pkl its manifest names and linked into the script "
            "cwd under the companion names the code uses."
        ),
    )
    parser.add_argument(
        "--rank-by",
        choices=["surprise", "posterior-surprise"],
        default="posterior-surprise",
        help="Ranking key; must match the summarizer's selection command.",
    )
    parser.add_argument(
        "--direction",
        choices=["both", "positive", "predictive"],
        default="both",
        help=(
            "Direction filter (must match the summarizer's): 'positive' re-runs only "
            "hypotheses where the experiment raised belief (posterior > prior by >0.02); "
            "'both' allows either belief direction; 'predictive' re-runs "
            "the IDs in --predictive-manifest."
        ),
    )
    parser.add_argument(
        "--predictive-manifest",
        type=Path,
        default=None,
        help=(
            "JSON manifest with a selected_ids list. Required when "
            "--direction predictive."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help=(
            "Re-run only the N top-ranked records. Not allowed with "
            "--direction predictive, which re-runs every non-excluded record."
        ),
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help=(
            "Write each selected record's code as an editable hypo_<id>.py (plus "
            "MANIFEST.json and REVISION_GUIDE.md) into this dir, then exit "
            "without running. Edit the loading sections and rerun with --code-dir."
        ),
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=None,
        help=(
            "Execute the revised hypo_<id>.py from this dir instead of the JSON "
            "code (per record; falls back to the recorded code when no revised "
            "file exists). Typically the same dir a prior --export-dir wrote."
        ),
    )
    parser.add_argument(
        "--corrected-dir",
        type=Path,
        default=None,
        help=(
            "Dir of hypo_<id>.py with a CORRECTED statistical test (the "
            "test-fixer's output). Takes precedence over --code-dir per record, "
            "so a fixed test supersedes the loading-only revision. Records with "
            "no corrected file fall back to --code-dir then the recorded code."
        ),
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help=(
            "Execute only records whose resolved --code-dir code differs from "
            "the recorded code, or whose --base-results entry is an environment "
            "failure, then merge them into --base-results."
        ),
    )
    parser.add_argument(
        "--only-corrected",
        action="store_true",
        help="Execute only records that have a non-empty script in --corrected-dir.",
    )
    parser.add_argument(
        "--base-results",
        type=Path,
        default=None,
        help="Complete prior result JSON used as the merge base for selective reruns.",
    )
    parser.add_argument(
        "--extra-only",
        action="store_true",
        help=(
            "Do not rerun the origin dataset; reuse --origin-results and execute "
            "only the requested --extra-pkl work items."
        ),
    )
    parser.add_argument(
        "--origin-results",
        type=Path,
        default=None,
        help="Final origin result JSON required by --extra-only.",
    )
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help=(
            "When a script fails on an absent module listed in INSTALLABLE_MODULES, "
            "pip-install its package into this environment (every installed "
            "distribution pinned as a constraint) and run the script again."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Atomic per-work-item checkpoint. Matching code/dataset/config work "
            "is reused after interruption."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-experiment timeout in seconds (default 3600 = 60 min).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Run up to this many scripts at once (default 1, one at a time). Above "
            "1, a script starts only when the job's memory can hold it, gets "
            "cpus // jobs BLAS/OpenMP threads, and is run again alone if the kernel "
            "kills it."
        ),
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=6000,
        help="Truncate each recorded/fresh output to this many chars.",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help=(
            "Disable the resident-payload fork runner and reload the pkl in a "
            "fresh subprocess per script (slower; the pre-2026-08 behavior)."
        ),
    )
    args = parser.parse_args(argv)
    global _INSTALL_MISSING
    _INSTALL_MISSING = args.install_missing

    global _PRELOAD_ENABLED
    if args.no_preload or not hasattr(os, "fork"):
        _PRELOAD_ENABLED = False

    if args.jobs < 1:
        parser.error("--jobs must be at least 1.")
    if args.direction == "predictive" and args.predictive_manifest is None:
        parser.error("--direction predictive requires --predictive-manifest PATH.")
    if args.direction == "predictive" and args.top is not None:
        parser.error("--direction predictive does not allow --top.")
    try:
        predictive_ids = (
            load_predictive_ids(args.predictive_manifest)
            if args.predictive_manifest is not None
            else None
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"Invalid predictive manifest: {exc}")

    if not args.json_file.is_file():
        parser.error(f"No such run JSON: {args.json_file}")
    if not args.pkl.is_file():
        parser.error(f"No such dataset pkl: {args.pkl}")
    if args.export_dir is not None and (
        args.code_dir is not None or args.corrected_dir is not None
    ):
        parser.error("--export-dir cannot combine with --code-dir/--corrected-dir.")
    if args.export_dir is not None and args.extra_pkl:
        parser.error("--extra-pkl has no effect with --export-dir (export only).")
    if args.only_changed and (args.code_dir is None or args.base_results is None):
        parser.error("--only-changed requires --code-dir and --base-results.")
    if args.only_corrected and args.corrected_dir is None:
        parser.error("--only-corrected requires --corrected-dir.")
    if args.only_changed and args.only_corrected:
        parser.error("--only-changed and --only-corrected are mutually exclusive.")
    if args.extra_only and (args.origin_results is None or not args.extra_pkl):
        parser.error("--extra-only requires --origin-results and --extra-pkl.")
    if args.extra_only and (args.only_changed or args.only_corrected or args.base_results):
        parser.error("--extra-only cannot combine with selective origin rerun options.")
    pkl = args.pkl.resolve()
    extra_pkls = []
    for ep in args.extra_pkl:
        if not ep.is_file():
            parser.error(f"No such extra dataset pkl: {ep}")
        extra_pkls.append(ep.resolve())
    try:
        companions = bind_companions(args.companion, [pkl, *extra_pkls])
    except ValueError as exc:
        parser.error(f"--companion: {exc}")

    def load_result_payload(path: Path | None, option: str) -> dict | None:
        if path is None:
            return None
        if not path.is_file():
            parser.error(f"{option} does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Invalid {option} JSON: {exc}")
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            parser.error(f"{option} must contain a results list.")
        return payload

    base_payload = load_result_payload(args.base_results, "--base-results")
    origin_payload = load_result_payload(args.origin_results, "--origin-results")

    records = load_records(args.json_file)
    ranked, _, _excluded = rank_records(
        records,
        rank_by=args.rank_by,
        direction=args.direction,
        predictive_ids=predictive_ids,
    )
    if args.top is not None:
        ranked = ranked[: args.top]

    run_sha256 = hashlib.sha256(args.json_file.read_bytes()).hexdigest()
    expected_ids = [str(r.get("id")) for r in ranked]
    if len(expected_ids) != len(set(expected_ids)):
        parser.error("Selected records contain duplicate ids; cannot merge safely.")

    def validate_prior_payload(payload: dict | None, option: str) -> None:
        if payload is None:
            return
        if payload.get("run_sha256") != run_sha256:
            parser.error(f"{option} was produced from a different run JSON.")
        if payload.get("pkl_fingerprint") != file_fingerprint(pkl):
            parser.error(f"{option} was produced from a different origin dataset.")
        ids = [str(item.get("id")) for item in payload["results"]]
        if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
            parser.error(
                f"{option} result ids do not exactly match the current selection."
            )

    validate_prior_payload(base_payload, "--base-results")
    validate_prior_payload(origin_payload, "--origin-results")

    # Export mode: dump editable scripts for the agent to revise, then exit.
    if args.export_dir is not None:
        summary = export_scripts(ranked, pkl, args.export_dir)
        print(
            f"[rerun] exported {summary['n_exported']} script(s) to "
            f"{summary['export_dir']}",
            file=sys.stderr,
            flush=True,
        )
        print(json.dumps({"mode": "export", **summary}, indent=2, ensure_ascii=False))
        return 0

    # Preflight (#4 guardrail): verify ONCE, up front, that the run interpreter
    # can import the core scientific stack. A globally broken env then fails here
    # with one clear message instead of as N identical per-script import errors,
    # and the non-zero exit stops the workflow before any agent folds the result.
    ok, detail = preflight_env(sys.executable)
    if not ok:
        print(
            f"[rerun] PREFLIGHT FAILED: the interpreter ({sys.executable}) cannot "
            f"import the core scientific stack — aborting before running any "
            f"script.\n{detail}",
            file=sys.stderr,
            flush=True,
        )
        print(
            json.dumps(
                {
                    "json_file": str(args.json_file),
                    "pkl": str(pkl),
                    "preflight_ok": False,
                    "environment_failure": True,
                    "preflight_error": detail,
                    "results": [],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 3
    print(f"[rerun] preflight OK (numpy {detail})", file=sys.stderr, flush=True)

    checkpoint_path = args.checkpoint.resolve() if args.checkpoint is not None else None
    checkpoint = _load_checkpoint(checkpoint_path)
    base_by_id = {
        str(item.get("id")): item
        for item in ((base_payload or {}).get("results") or [])
    }
    origin_by_id = {
        str(item.get("id")): item
        for item in ((origin_payload or {}).get("results") or [])
    }

    # Plan every record first, then run all origin work and then all
    # extrapolation work, so --jobs can spread each phase over the cores.
    results: list[dict | None] = []
    n_base_reused = 0
    selected_execution_ids = []
    plans = []
    for rank, r in enumerate(ranked, start=1):
        code, source = resolve_code(r, args.code_dir, args.corrected_dir)
        rid = r.get("id")
        rid_key = str(rid)
        recorded_code = r.get("code") or ""
        changed = code_sha256(code) != code_sha256(recorded_code)

        should_execute = True
        if args.only_changed:
            # An environment failure is retried with unchanged code, so a
            # package installed after the first pass is picked up.
            base = base_by_id.get(rid_key) or {}
            should_execute = changed or base.get("rerun_failure_kind") == "environment"
        elif args.only_corrected:
            corrected = args.corrected_dir / hypo_filename(rid)
            should_execute = corrected.is_file() and bool(
                corrected.read_text(encoding="utf-8").strip()
            )

        if not should_execute:
            if args.only_corrected:
                continue
            cached_result = base_by_id.get(rid_key)
            if cached_result is None:
                parser.error(
                    f"--base-results is missing selected record id={rid}; "
                    "cannot construct a complete merged result."
                )
            results.append(cached_result)
            n_base_reused += 1
            continue
        selected_execution_ids.append(rid)

        origin_result = origin_by_id.get(rid_key) if args.extra_only else None
        if args.extra_only and origin_result is None:
            parser.error(
                f"--origin-results is missing selected record id={rid}; "
                "cannot extrapolate without its origin result."
            )
        if args.extra_only and origin_result.get("executed_code_sha256") != code_sha256(code):
            parser.error(
                f"--origin-results code fingerprint for id={rid} does not match "
                "the code selected for extrapolation."
            )
        plan = {
            "rank": rank, "record": r, "rid": rid, "code": code, "source": source,
            "recorded_code": recorded_code, "changed": changed,
            "origin_result": origin_result, "slot": len(results),
            "failure_kind": None,  # None | "environment" | "data-loading" | "analysis"
        }
        results.append(None)  # filled in once the record's work has run
        plans.append(plan)
        label = f"[rerun {rank}/{len(ranked)}] id={rid}"
        if args.extra_only:
            plan["run"] = {
                "timed_out": bool(origin_result.get("rerun_timed_out")),
                "exitcode": origin_result.get("rerun_exitcode"),
                "stdout": origin_result.get("rerun_stdout") or "",
                "stderr": origin_result.get("rerun_stderr") or "",
                "runtime_ms": origin_result.get("rerun_runtime_ms") or 0,
            }
            usability = origin_result.get("result_status")
            usability_reason = origin_result.get("result_failure_reason")
            if usability not in {"USABLE", "UNUSABLE"}:
                usability, usability_reason = result_usability(plan["run"])
            plan.update(usability=usability, reason=usability_reason)
        elif not code.strip():
            plan["run"] = {
                "timed_out": False,
                "exitcode": None,
                "stdout": "",
                "stderr": "[rerun] record has no 'code' to execute",
                "runtime_ms": 0,
            }
            plan.update(failure_kind="analysis", usability="UNUSABLE", reason="no-code")
            print(f"{label} ({source}) origin NO-CODE in 0s", file=sys.stderr, flush=True)
        else:
            prior = base_by_id.get(rid_key) or {}
            plan["work"] = {
                "code": code, "dataset": pkl, "companions": companions[pkl],
                "record_id": rid, "label": f"{label} ({source}) origin",
                "predicted_s": (prior.get("rerun_runtime_ms") or 0) / 1000 or None,
            }

    def execute(work: list[dict]) -> tuple[int, int]:
        execute_work_items(work, jobs=args.jobs, timeout=args.timeout,
                           run_sha256=run_sha256, checkpoint=checkpoint,
                           checkpoint_path=checkpoint_path)
        reused = sum(1 for entry in work if entry["reused"])
        return len(work) - reused, reused

    n_executed, n_checkpoint_reused = execute(
        [plan["work"] for plan in plans if "work" in plan]
    )
    for plan in plans:
        if "work" in plan:
            plan["run"] = plan["work"]["run"]
            _, plan["failure_kind"], plan["usability"], plan["reason"] = run_outcome(
                plan["run"]
            )

    # Extrapolation: run the SAME code on each other dataset to test whether
    # the finding generalizes. Skip when the code didn't even run on origin
    # (no point — there's no analysis to carry over).
    extra_work = []
    for plan in plans:
        plan["extrapolations"] = []
        if not (extra_pkls and plan["code"].strip() and plan["usability"] == "USABLE"):
            continue
        for ei, ep in enumerate(extra_pkls, start=1):
            label = (f"[rerun {plan['rank']}/{len(ranked)}] id={plan['rid']} "
                     f"extrapolate {ei}/{len(extra_pkls)} {ep.name}")
            xcode, retargeted, refusal = retarget_image(
                plan["code"], companions[pkl], companions[ep]
            )
            if refusal:
                print(f"{label} UNUSABLE(origin-image-path), not run",
                      file=sys.stderr, flush=True)
                plan["extrapolations"].append(
                    {
                        "pkl": str(ep),
                        "pkl_name": ep.name,
                        "exitcode": None,
                        "timed_out": False,
                        "runtime_ms": 0,
                        "result_status": "UNUSABLE",
                        "result_failure_reason": "origin-image-path",
                        "stdout": "",
                        "stderr": refusal,
                    }
                )
                continue
            if retargeted:
                print(f"{label}: image folder {retargeted}", file=sys.stderr, flush=True)
            work = {
                "code": xcode, "dataset": ep, "companions": companions[ep],
                "record_id": plan["rid"], "label": label,
                "predicted_s": (plan["run"].get("runtime_ms") or 0) / 1000 or None,
                "retargeted": retargeted,
            }
            plan["extrapolations"].append(work)
            extra_work.append(work)
    executed, reused = execute(extra_work)
    n_executed += executed
    n_checkpoint_reused += reused

    for plan in plans:
        extrapolations = []
        for item in plan["extrapolations"]:
            if "run" not in item:  # refused before running
                extrapolations.append(item)
                continue
            xrun = item["run"]
            xusability, xreason = result_usability(xrun)
            extrapolations.append(
                {
                    "pkl": str(item["dataset"]),
                    "pkl_name": item["dataset"].name,
                    "exitcode": xrun["exitcode"],
                    "timed_out": xrun["timed_out"],
                    "runtime_ms": xrun["runtime_ms"],
                    "result_status": xusability,
                    "result_failure_reason": xreason,
                    "stdout": truncate(xrun["stdout"], args.max_output_chars),
                    "stderr": truncate(xrun["stderr"], args.max_output_chars),
                    "input_audit": xrun.get("input_audit") or {},
                    "compat_rewrites": xrun.get("compat_rewrites") or [],
                    **({"image_retargeted": item["retargeted"]}
                       if item["retargeted"] else {}),
                }
            )
        if args.extra_only:
            result = dict(plan["origin_result"])
            result["extrapolations"] = extrapolations
        else:
            r, run, code = plan["record"], plan["run"], plan["code"]
            result = {
                "rank": plan["rank"],
                "id": plan["rid"],
                "status": r.get("status", ""),
                "code_source": plan["source"],
                "surprisal": round(r["_surprisal"], 4),
                "priority_score": round(r.get("_priority", 0.0), 4),
                "hypothesis": (r.get("hypothesis") or "").strip(),
                "pkl_basenames_in_code": pkl_basenames(code),
                "recorded_output": truncate(
                    str(r.get("codeOutput") or ""), args.max_output_chars
                ),
                "rerun_exitcode": run["exitcode"],
                "rerun_timed_out": run["timed_out"],
                "rerun_failure_kind": plan["failure_kind"],
                "result_status": plan["usability"],
                "result_failure_reason": plan["reason"],
                "rerun_runtime_ms": run["runtime_ms"],
                "rerun_stdout": truncate(run["stdout"], args.max_output_chars),
                "rerun_stderr": truncate(run["stderr"], args.max_output_chars),
                "input_audit": run.get("input_audit") or {},
                "compat_rewrites": run.get("compat_rewrites") or [],
                "extrapolations": extrapolations,
                "recorded_code_sha256": code_sha256(plan["recorded_code"]),
                "executed_code_sha256": code_sha256(code),
                "code_changed": plan["changed"],
            }
        results[plan["slot"]] = result

    n_ok = sum(1 for item in results if item.get("result_status") == "USABLE")
    n_timeout = sum(1 for item in results if item.get("rerun_timed_out"))
    n_env_failed = sum(
        1 for item in results if item.get("rerun_failure_kind") == "environment"
    )
    n_failed = len(results) - n_ok - n_timeout
    n_revised = sum(1 for item in results if item.get("code_source") == "revised")
    n_corrected = sum(1 for item in results if item.get("code_source") == "corrected")

    payload = {
        "json_file": str(args.json_file),
        "pkl": str(pkl),
        "extra_pkls": [str(p) for p in extra_pkls],
        "companions": {str(p): [str(f) for f in fs] for p, fs in companions.items() if fs},
        "rank_by": args.rank_by,
        "top": args.top,
        "timeout_s": args.timeout,
        "jobs": args.jobs,
        "code_dir": str(args.code_dir) if args.code_dir is not None else None,
        "corrected_dir": str(args.corrected_dir)
        if args.corrected_dir is not None
        else None,
        "n_rerun": len(results),
        "n_revised": n_revised,
        "n_corrected": n_corrected,
        "n_recorded": len(results) - n_revised - n_corrected,
        "n_executed": n_executed,
        "n_checkpoint_reused": n_checkpoint_reused,
        "n_base_reused": n_base_reused,
        "selected_execution_ids": selected_execution_ids,
        "changed_ids": [item.get("id") for item in results if item.get("code_changed")],
        "run_sha256": run_sha256,
        "pkl_fingerprint": file_fingerprint(pkl),
        "runner_version": _RUNNER_VERSION,
        "extra_only": args.extra_only,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_env_failed": n_env_failed,
        "n_timeout": n_timeout,
        "installed_packages": list(_INSTALLED.values()),
        "preflight_ok": True,
        # A systematic environment failure: nothing ran AND every failure is an
        # import/native-load error. This is the signature of a broken env (the
        # NumPy-source-tree regression), NOT a batch of legitimately-inconclusive
        # statistics — flag it so the downstream fold step / orchestrator does not
        # record N "could not run" verdicts as if they were real findings.
        "environment_failure": (
            n_ok == 0 and n_env_failed > 0 and n_env_failed == n_failed
        ),
        "results": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    # Exit 3 (distinct from argparse's 2) on a systematic env failure so the
    # driver's run_compute() aborts loudly BEFORE an agent folds a bad result,
    # rather than treating it as a normal completion.
    if payload["environment_failure"]:
        print(
            f"[rerun] ENVIRONMENT FAILURE: {n_env_failed}/{len(results)} scripts "
            f"failed to import the scientific stack and none succeeded. This is an "
            f"environment problem, not a statistical result — aborting so it is not "
            f"folded as 'could not run' verdicts. Inspect a rerun_stderr above.",
            file=sys.stderr,
            flush=True,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
