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
includes the source-run hash, record id, exact code hash, dataset identity,
timeout, runner version, Python, and scientific-package versions. A changed
input invalidates only the affected work item; completed matching items resume
without execution, except an environment failure, which is always run again.
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
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

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
# file must genuinely exist wherever a script computes it. ``rerun_one``
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
# name. ``rerun_one`` links the companion bound to the pkl being run under every
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
_CHECKPOINT_SCHEMA_VERSION = 1
_RUNNER_VERSION = "work-items-v1"


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


def result_usability(run: dict) -> tuple[str, str | None]:
    """Separate process success from whether an experiment produced a result."""
    if run.get("timed_out"):
        return "UNUSABLE", "timeout"
    if run.get("exitcode") != 0:
        return "UNUSABLE", "nonzero-exit"
    stdout = str(run.get("stdout") or "")
    if not stdout.strip():
        return "UNUSABLE", "empty-output"
    if _UNUSABLE_OUTPUT_RE.search(stdout):
        return "UNUSABLE", "dataset-loading"
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

    A cached environment failure (missing package, broken native import) is run
    again rather than reused: the work key does not cover every package a script
    may import, so a package installed since then would otherwise never be seen.
    """
    key = _work_key(
        run_sha256=run_sha256,
        record_id=record_id,
        code_hash=code_sha256(code),
        dataset=dataset,
        timeout=timeout,
        companions=companions,
    )
    cached = checkpoint["work_items"].get(key)
    if isinstance(cached, dict) and isinstance(cached.get("run"), dict) and not (
        _is_environment_failure(cached["run"])
    ):
        return cached["run"], True
    run = _run_installing_missing(code, dataset, timeout, companions)
    checkpoint["work_items"][key] = {
        "record_id": record_id,
        "code_sha256": code_sha256(code),
        "dataset": file_fingerprint(dataset),
        "run": run,
    }
    if checkpoint_path is not None:
        _atomic_write_json(checkpoint_path, checkpoint)
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


def _apply_inprocess_patches(payload) -> None:
    """Child-side twin of ``_RUNNER`` for the forked execution path.

    Applies the same neutralizations and ``*.pkl`` redirections as the
    subprocess runner (keep the two in sync), plus one preload shortcut:
    ``pickle.load`` on any ``*.pkl``-named file returns the parent's resident
    payload instead of deserializing the multi-GB cache again.
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


def _run_forked(run_dir: Path, env: dict, timeout: int, payload) -> dict:
    """Execute ``experiment.py`` in a forked child sharing the resident payload.

    fork gives the child a copy-on-write view of the parent's memory: the
    multi-GB payload is shared for reads and any in-place mutation stays
    private to that child — the same isolation the per-script subprocess gave,
    without a fresh deserialization per script. The child leads its own
    session so a timeout kills its whole process group.
    """
    import signal

    stdout_path = run_dir.parent / "stdout.txt"
    stderr_path = run_dir.parent / "stderr.txt"
    start = time.monotonic()
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:  # child — must never return into the parent's control flow
        exit_code = 1
        try:
            os.setsid()
            out_fd = os.open(str(stdout_path),
                             os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            err_fd = os.open(str(stderr_path),
                             os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            os.chdir(str(run_dir))
            os.environ.clear()
            os.environ.update(env)
            sys.argv = ["experiment.py"]
            _apply_inprocess_patches(payload)
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
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(exit_code)

    timed_out = False
    exitcode = None
    deadline = start + timeout
    while True:
        done_pid, status = os.waitpid(pid, os.WNOHANG)
        if done_pid == pid:
            if os.WIFEXITED(status):
                exitcode = os.WEXITSTATUS(status)
            elif os.WIFSIGNALED(status):
                exitcode = -os.WTERMSIG(status)
            break
        if time.monotonic() >= deadline:
            timed_out = True
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            break
        time.sleep(0.05)

    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    stderr_text = _read(stderr_path)
    if timed_out:
        stderr_text += f"\n[rerun] timed out after {timeout}s"
    return {
        "timed_out": timed_out,
        "exitcode": None if timed_out else exitcode,
        "stdout": _read(stdout_path),
        "stderr": stderr_text,
        "runtime_ms": round((time.monotonic() - start) * 1000),
    }


def rerun_one(code: str, pkl: Path, timeout: int, companions=()) -> dict:
    """Execute one record's recorded ``code``, forcing every pkl load to ``pkl``.

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
    falls back to the original fresh-subprocess-per-script path.
    """
    payload = _preload_payload(pkl) if _PRELOAD_ENABLED else None
    with tempfile.TemporaryDirectory(prefix="rerun_") as tmp:
        # The cwd is nested one level so scripts probing ``Path.cwd().parent``
        # still land inside the disposable sandbox (where a token link can
        # live) instead of the shared $TMPDIR.
        run = Path(tmp) / "run"
        run.mkdir()
        (run / "experiment.py").write_text(code, encoding="utf-8")
        env = {
            **os.environ,
            "MPLBACKEND": "Agg",  # headless plotting
            "RERUN_PKL": str(pkl),
        }
        env.setdefault("AWS_EC2_METADATA_DISABLED", "true")
        links = companion_links(code, companions)
        if links:
            env["RERUN_COMPANIONS"] = json.dumps(links)
            for name, target in links.items():
                (run / name).symlink_to(target)
        if GCS_TOKEN_PATH.is_file():
            env.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(GCS_TOKEN_PATH))
            env["RERUN_GCS_TOKEN"] = str(GCS_TOKEN_PATH)
            _link_gcs_token(run)
            _link_gcs_token(Path(tmp))
            _link_gcs_token(pkl.parent, announce=True)
        if payload is not None:
            return _run_forked(run, env, timeout, payload)
        (run / "_runner.py").write_text(_RUNNER, encoding="utf-8")
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "_runner.py"],
                cwd=str(run),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return {
                "timed_out": False,
                "exitcode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "runtime_ms": round((time.monotonic() - start) * 1000),
            }
        except subprocess.TimeoutExpired as e:
            # On timeout the partial stdout/stderr may come back as bytes even
            # with text=True; decode defensively before concatenating.
            def _txt(x):
                if isinstance(x, bytes):
                    return x.decode("utf-8", "replace")
                return x or ""

            return {
                "timed_out": True,
                "exitcode": None,
                "stdout": _txt(e.stdout),
                "stderr": _txt(e.stderr) + f"\n[rerun] timed out after {timeout}s",
                "runtime_ms": round((time.monotonic() - start) * 1000),
            }


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

    results = []
    n_executed = n_checkpoint_reused = n_base_reused = 0
    selected_execution_ids = []
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
        print(
            f"[rerun {rank}/{len(ranked)}] id={rid} ({source}) "
            f"{'extra-only' if args.extra_only else 'origin'} …",
            file=sys.stderr,
            flush=True,
        )
        failure_kind = None  # None | "environment" | "analysis"
        if args.extra_only:
            run = {
                "timed_out": bool(origin_result.get("rerun_timed_out")),
                "exitcode": origin_result.get("rerun_exitcode"),
                "stdout": origin_result.get("rerun_stdout") or "",
                "stderr": origin_result.get("rerun_stderr") or "",
                "runtime_ms": origin_result.get("rerun_runtime_ms") or 0,
            }
            usability = origin_result.get("result_status")
            usability_reason = origin_result.get("result_failure_reason")
            if usability not in {"USABLE", "UNUSABLE"}:
                usability, usability_reason = result_usability(run)
        elif not code.strip():
            run = {
                "timed_out": False,
                "exitcode": None,
                "stdout": "",
                "stderr": "[rerun] record has no 'code' to execute",
                "runtime_ms": 0,
            }
            outcome = "NO-CODE"
            failure_kind = "analysis"
            usability, usability_reason = "UNUSABLE", "no-code"
        else:
            run, reused = _run_work_item(
                code,
                pkl,
                args.timeout,
                run_sha256=run_sha256,
                record_id=rid,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                companions=companions[pkl],
            )
            n_checkpoint_reused += int(reused)
            n_executed += int(not reused)
            usability, usability_reason = result_usability(run)
            if run["timed_out"]:
                outcome = "TIMEOUT"
            elif run["exitcode"] == 0 and usability == "USABLE":
                outcome = "OK"
            elif run["exitcode"] == 0:
                outcome = f"UNUSABLE({usability_reason})"
                failure_kind = (
                    "data-loading" if usability_reason == "dataset-loading" else "analysis"
                )
            else:
                failure_kind = classify_failure(run["stderr"])
                if failure_kind == "environment":
                    outcome = f"ENV-FAILED(exit={run['exitcode']})"
                else:
                    outcome = f"FAILED(exit={run['exitcode']})"

        if not args.extra_only:
            print(
                f"[rerun {rank}/{len(ranked)}] id={rid} {outcome} "
                f"in {run['runtime_ms'] / 1000:.0f}s",
                file=sys.stderr,
                flush=True,
            )

        # Extrapolation: run the SAME code on each other dataset to test whether
        # the finding generalizes. Skip when the code didn't even run on origin
        # (no point — there's no analysis to carry over).
        extrapolations = []
        if extra_pkls and code.strip() and usability == "USABLE":
            for ei, ep in enumerate(extra_pkls, start=1):
                xcode, retargeted, refusal = retarget_image(
                    code, companions[pkl], companions[ep]
                )
                if refusal:
                    print(
                        f"[rerun {rank}/{len(ranked)}] id={rid} extrapolate "
                        f"{ei}/{len(extra_pkls)} {ep.name} "
                        "UNUSABLE(origin-image-path), not run",
                        file=sys.stderr,
                        flush=True,
                    )
                    extrapolations.append(
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
                    print(
                        f"[rerun {rank}/{len(ranked)}] id={rid} extrapolate "
                        f"{ei}/{len(extra_pkls)} {ep.name}: image folder "
                        f"{retargeted}",
                        file=sys.stderr,
                        flush=True,
                    )
                xrun, reused = _run_work_item(
                    xcode,
                    ep,
                    args.timeout,
                    run_sha256=run_sha256,
                    record_id=rid,
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                    companions=companions[ep],
                )
                n_checkpoint_reused += int(reused)
                n_executed += int(not reused)
                xusability, xreason = result_usability(xrun)
                xoutcome = (
                    "TIMEOUT"
                    if xrun["timed_out"]
                    else (
                        "OK"
                        if xrun["exitcode"] == 0 and xusability == "USABLE"
                        else (
                            f"UNUSABLE({xreason})"
                            if xrun["exitcode"] == 0
                            else f"FAILED(exit={xrun['exitcode']})"
                        )
                    )
                )
                print(
                    f"[rerun {rank}/{len(ranked)}] id={rid} extrapolate "
                    f"{ei}/{len(extra_pkls)} {ep.name} {xoutcome} "
                    f"in {xrun['runtime_ms'] / 1000:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                extrapolations.append(
                    {
                        "pkl": str(ep),
                        "pkl_name": ep.name,
                        "exitcode": xrun["exitcode"],
                        "timed_out": xrun["timed_out"],
                        "runtime_ms": xrun["runtime_ms"],
                        "result_status": xusability,
                        "result_failure_reason": xreason,
                        "stdout": truncate(xrun["stdout"], args.max_output_chars),
                        "stderr": truncate(xrun["stderr"], args.max_output_chars),
                        **({"image_retargeted": retargeted} if retargeted else {}),
                    }
                )

        if args.extra_only:
            result = dict(origin_result)
            result["extrapolations"] = extrapolations
        else:
            result = {
                "rank": rank,
                "id": rid,
                "status": r.get("status", ""),
                "code_source": source,
                "surprisal": round(r["_surprisal"], 4),
                "priority_score": round(r.get("_priority", 0.0), 4),
                "hypothesis": (r.get("hypothesis") or "").strip(),
                "pkl_basenames_in_code": pkl_basenames(code),
                "recorded_output": truncate(
                    str(r.get("codeOutput") or ""), args.max_output_chars
                ),
                "rerun_exitcode": run["exitcode"],
                "rerun_timed_out": run["timed_out"],
                "rerun_failure_kind": failure_kind,
                "result_status": usability,
                "result_failure_reason": usability_reason,
                "rerun_runtime_ms": run["runtime_ms"],
                "rerun_stdout": truncate(run["stdout"], args.max_output_chars),
                "rerun_stderr": truncate(run["stderr"], args.max_output_chars),
                "extrapolations": extrapolations,
                "recorded_code_sha256": code_sha256(recorded_code),
                "executed_code_sha256": code_sha256(code),
                "code_changed": changed,
            }
        results.append(result)

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
