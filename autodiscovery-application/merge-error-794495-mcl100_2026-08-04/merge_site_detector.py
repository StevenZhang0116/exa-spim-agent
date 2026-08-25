#!/usr/bin/env python
"""Merge-site detector for ExaSPIM segment adjudication.

ONE self-contained CLI that:
  * loads an `_add.pkl` dataset cache once,
  * extracts audited segment-level geometry features in shared traversal passes,
  * compares several regularized/boosted model families with nested CV,
  * selects and fits the winning family with a leak-free one-standard-error rule,
  * scores every adjudicable segment out-of-fold,
  * optionally transfers the fitted pipeline to a different held-out brain,
  * profiles hypothesis cost on a small sample with --measuretime and exits pre-model, or
  * skips explicitly excluded computation with --hypothesis-selection or
    --exclude-hypotheses.

Feature semantics are copied verbatim (constants, reductions, aggregations)
from the audited hypothesis source scripts named in `feature_inventory.json`;
each source's SHA-256 is re-verified at those files' original locations when the
inventory is present.  Generated detectors are not fully validated on a real
pkl during the build; successful end-to-end real-data verification remains an
operator step.  Run on a compute node with >=80 GB RAM under `panda`.

NOTE ON MISSINGNESS: definedness is decided by dictionary/set membership during
extraction, NOT by comparing a value to an old sentinel.  Undefined numeric
features are emitted as NaN with a companion `<feature>_is_defined` flag.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import socket
import sys
import time
import traceback
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------- #
# Embedded declarative model policy (SHA-256                                   #
# 4aa9c7a9411f73438faeb74cb6459db084db4aabd82cfa9861328e32c9b22f3c).           #
# This embedded copy is the runtime trust anchor. The build driver validates   #
# the on-disk policy; the deployed detector validates model_candidates.json    #
# against this exact policy hash without requiring that policy file.           #
# --------------------------------------------------------------------------- #
EMBEDDED_MODEL_POLICY = {
    "schema_version": 1,
    "selection": {
        "max_optional_models": 2,
        "max_grid_combinations": 24,
        "primary_metric": "average_precision",
        "selection_rule": "one_standard_error",
        "outer_folds": 5,
        "inner_folds": 3,
        "random_seed": 42,
        "review_budget": 100,
        "simplicity_order": [
            "logistic_l2",
            "logistic_elasticnet",
            "spline_logistic",
            "explainable_boosting",
            "hist_gradient_boosting",
            "extra_trees",
            "random_forest",
            "xgboost",
        ],
    },
    "families": {
        "logistic_l2": {
            "required": True,
            "native_nan": False,
            "requires_package": None,
            "parameters": {"C": "positive_number"},
        },
        "logistic_elasticnet": {
            "required": True,
            "native_nan": False,
            "requires_package": None,
            "parameters": {"C": "positive_number", "l1_ratio": "unit_interval"},
        },
        "hist_gradient_boosting": {
            "required": True,
            "native_nan": True,
            "requires_package": None,
            "parameters": {
                "learning_rate": "positive_number",
                "max_depth": "nullable_positive_integer",
                "max_leaf_nodes": "integer_min_2",
                "min_samples_leaf": "positive_integer",
                "l2_regularization": "nonnegative_number",
            },
        },
        "spline_logistic": {
            "required": False,
            "native_nan": False,
            "requires_package": None,
            "parameters": {
                "n_knots": "integer_min_2",
                "degree": "positive_integer",
                "C": "positive_number",
            },
        },
        "explainable_boosting": {
            "required": False,
            "native_nan": True,
            "requires_package": "interpret",
            "parameters": {
                "max_bins": "integer_min_2",
                "interactions": "nonnegative_integer",
                "learning_rate": "positive_number",
                "max_rounds": "positive_integer",
                "min_samples_leaf": "positive_integer",
            },
        },
        "extra_trees": {
            "required": False,
            "native_nan": False,
            "requires_package": None,
            "parameters": {
                "n_estimators": "positive_integer",
                "max_depth": "nullable_positive_integer",
                "min_samples_leaf": "positive_integer",
                "max_features": "max_features",
            },
        },
        "random_forest": {
            "required": False,
            "native_nan": False,
            "requires_package": None,
            "parameters": {
                "n_estimators": "positive_integer",
                "max_depth": "nullable_positive_integer",
                "min_samples_leaf": "positive_integer",
                "max_features": "max_features",
            },
        },
        "xgboost": {
            "required": False,
            "native_nan": True,
            "requires_package": "xgboost",
            "parameters": {
                "n_estimators": "positive_integer",
                "max_depth": "positive_integer",
                "learning_rate": "positive_number",
                "min_child_weight": "nonnegative_number",
                "subsample": "positive_unit_interval",
                "colsample_bytree": "positive_unit_interval",
                "reg_lambda": "nonnegative_number",
            },
        },
    },
}
EMBEDDED_MODEL_POLICY_SHA256 = (
    "4aa9c7a9411f73438faeb74cb6459db084db4aabd82cfa9861328e32c9b22f3c"
)

RANDOM_SEED = 42
ALLOWED_FAMILIES = set(EMBEDDED_MODEL_POLICY["families"].keys())
NATIVE_NAN_FAMILIES = {
    n for n, m in EMBEDDED_MODEL_POLICY["families"].items() if m["native_nan"]
}


# --------------------------------------------------------------------------- #
# Tee logging: flush every write; isatty()==False so libs don't emit progress  #
# bars / ANSI escapes into the log file.                                       #
# --------------------------------------------------------------------------- #
class _Tee(object):
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._fh.write(data)
        self._fh.flush()

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self._stream.fileno()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_canonical_json(obj):
    """SHA-256 of a canonicalized JSON encoding (sort keys, tight separators)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# Parameter value-rule validation (mirrors the policy vocabulary exactly).     #
# --------------------------------------------------------------------------- #
def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and np.isfinite(x))


def _validate_param_value(rule, value):
    """Return True iff `value` satisfies the policy `rule` token."""
    if rule == "positive_number":
        return _is_num(value) and value > 0
    if rule == "nonnegative_number":
        return _is_num(value) and value >= 0
    if rule == "unit_interval":
        return _is_num(value) and 0.0 <= value <= 1.0
    if rule == "positive_unit_interval":
        return _is_num(value) and 0.0 < value <= 1.0
    if rule == "positive_integer":
        return _is_int(value) and value > 0
    if rule == "nonnegative_integer":
        return _is_int(value) and value >= 0
    if rule == "integer_min_2":
        return _is_int(value) and value >= 2
    if rule == "nullable_positive_integer":
        return value is None or (_is_int(value) and value > 0)
    if rule == "max_features":
        if value is None:
            return True
        if isinstance(value, str):
            return value in ("sqrt", "log2")
        if _is_int(value):
            return value > 0
        if _is_num(value):
            return 0.0 < value <= 1.0
        return False
    raise ValueError("unknown parameter rule token: %r" % (rule,))


def _grid_size(grid):
    n = 1
    for vals in grid.values():
        n *= max(1, len(vals))
    return n


def validate_model_config(cfg, inventory_sha256):
    """Re-validate the driver-produced model_candidates.json against the policy.

    Raises ValueError on any violation.  This is the single trust boundary for
    the JSON config: after this returns, only allowlisted families with
    value-checked grids are constructed.
    """
    policy = EMBEDDED_MODEL_POLICY
    families = policy["families"]
    sel = policy["selection"]

    if cfg.get("schema_version") != 2:
        raise ValueError("model config schema_version must be 2")

    inv_sha = cfg.get("feature_inventory_sha256")
    if inventory_sha256 is not None and inv_sha != inventory_sha256:
        raise ValueError(
            "model config feature_inventory_sha256 %r != inventory %r"
            % (inv_sha, inventory_sha256))

    pol_sha = cfg.get("model_policy_sha256")
    if pol_sha != EMBEDDED_MODEL_POLICY_SHA256:
        raise ValueError(
            "model config model_policy_sha256 %r != embedded policy %r"
            % (pol_sha, EMBEDDED_MODEL_POLICY_SHA256))

    candidates = cfg.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("model config has no candidates")

    seen = set()
    n_optional = 0
    required_present = set()
    for cand in candidates:
        name = cand.get("name")
        if name not in ALLOWED_FAMILIES:
            raise ValueError("unknown / non-allowlisted family: %r" % (name,))
        if name in seen:
            raise ValueError("duplicate family in config: %r" % (name,))
        seen.add(name)
        fam = families[name]

        role = cand.get("role")
        expected_role = "baseline" if fam["required"] else "optional"
        if role != expected_role:
            raise ValueError(
                "family %r role=%r contradicts policy role %r"
                % (name, role, expected_role))
        if fam["required"]:
            required_present.add(name)
        else:
            n_optional += 1

        declared_nan = bool(cand.get("native_nan", False))
        if declared_nan != bool(fam["native_nan"]):
            raise ValueError(
                "family %r native_nan=%r contradicts policy %r"
                % (name, declared_nan, fam["native_nan"]))

        declared_pkg = cand.get("requires_package", None)
        if declared_pkg != fam["requires_package"]:
            raise ValueError(
                "family %r requires_package=%r contradicts policy %r"
                % (name, declared_pkg, fam["requires_package"]))

        grid = cand.get("grid", {})
        if not isinstance(grid, dict):
            raise ValueError("family %r grid must be an object" % (name,))
        allowed_params = fam["parameters"]
        for pname, pvals in grid.items():
            if pname not in allowed_params:
                raise ValueError(
                    "family %r has non-allowlisted parameter %r"
                    % (name, pname))
            if not isinstance(pvals, list) or not pvals:
                raise ValueError(
                    "family %r parameter %r must be a non-empty list"
                    % (name, pname))
            rule = allowed_params[pname]
            for v in pvals:
                if not _validate_param_value(rule, v):
                    raise ValueError(
                        "family %r parameter %r value %r violates rule %r"
                        % (name, pname, v, rule))
        gsize = _grid_size(grid)
        if gsize > sel["max_grid_combinations"]:
            raise ValueError(
                "family %r grid has %d combinations > cap %d"
                % (name, gsize, sel["max_grid_combinations"]))

    missing_required = [
        n for n, m in families.items() if m["required"] and n not in required_present
    ]
    if missing_required:
        raise ValueError("required baseline(s) missing: %r" % (missing_required,))

    if n_optional > sel["max_optional_models"]:
        raise ValueError(
            "%d optional models > cap %d"
            % (n_optional, sel["max_optional_models"]))

    return True


# --------------------------------------------------------------------------- #
# Cache loading. Prefer the real package (its SkeletonGraph carries helper      #
# methods the mock lacks); fall back to a mock unpickler only on ImportError.   #
# --------------------------------------------------------------------------- #
def load_payload(pkl_path):
    import importlib
    import pickle

    try:
        importlib.import_module("agentic_neuron_proofreader")  # registers classes
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except (ImportError, ModuleNotFoundError):
        import types
        import networkx as nx

        class SkeletonGraph(nx.Graph):
            pass

        for mod in [
            "agentic_neuron_proofreader",
            "agentic_neuron_proofreader.data_modules",
            "agentic_neuron_proofreader.data_modules.canonical_labeling",
        ]:
            if mod not in sys.modules:
                sys.modules[mod] = types.ModuleType(mod)
        sys.modules["agentic_neuron_proofreader"].SkeletonGraph = SkeletonGraph

        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("agentic_neuron_proofreader"):
                    if name == "SkeletonGraph":
                        return SkeletonGraph
                    return type(name, (object,), {})
                return super().find_class(module, name)

        with open(pkl_path, "rb") as f:
            return CustomUnpickler(f).load()


def brain_id_from_path(pkl_path):
    import re

    base = os.path.basename(pkl_path)
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", base)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4,})", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def mcl_from_payload(payload, pkl_path):
    try:
        return int(payload.get("min_cable_length"))
    except Exception:
        import re

        m = re.search(r"_mcl(\d+)_", os.path.basename(pkl_path))
        return int(m.group(1)) if m else None


def build_segment_universe(payload):
    """One row per adjudicable segment.

    Universe = non-zero canonical labels.  Components map to segments through the
    part of component_id_to_swc_id before the '.'.  is_merge = membership in
    gt_merge_labels.  A segment with no fragment component still gets a row.
    """
    node_label = np.asarray(payload["gt_node_canonical_label"])
    adjudicable = sorted(int(x) for x in np.unique(node_label) if int(x) != 0)
    gt_merge_labels = set(int(x) for x in payload["gt_merge_labels"])
    is_merge = np.array(
        [1 if s in gt_merge_labels else 0 for s in adjudicable], dtype=np.int64
    )
    return adjudicable, is_merge


def build_comp_to_seg(frag):
    """component_id -> segment id (part of the swc id before the '.')."""
    out = {}
    for comp_id, swc_id in frag.component_id_to_swc_id.items():
        out[int(comp_id)] = int(str(swc_id).split(".")[0])
    return out


# --------------------------------------------------------------------------- #
# Hypothesis-cost profiling and final-run selection. Feature code supplies     #
# declarative groups; runtime owns validation, artifacts, and CLI semantics.   #
# --------------------------------------------------------------------------- #
def validate_analysis_timing_groups(groups):
    """Require one selectable timing-group owner per feature and hypothesis."""
    if not isinstance(groups, list) or not groups:
        raise ValueError("ANALYSIS_TIMING_GROUPS must be a non-empty list")
    keys = set()
    covered = []
    hypothesis_owner = {}
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("each analysis timing group must be an object")
        key = group.get("key")
        ids = group.get("hypothesis_ids")
        names = group.get("feature_names")
        phase = group.get("phase")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("analysis timing group keys must be unique non-empty strings")
        keys.add(key)
        if (not isinstance(ids, list) or not ids or
                any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in ids)):
            raise ValueError("timing group %r has invalid hypothesis_ids" % key)
        if len(ids) != len(set(ids)):
            raise ValueError("timing group %r repeats hypothesis_ids" % key)
        for hypothesis_id in ids:
            if hypothesis_id in hypothesis_owner:
                raise ValueError(
                    "hypothesis %d belongs to multiple timing groups: %r and %r"
                    % (hypothesis_id, hypothesis_owner[hypothesis_id], key))
            hypothesis_owner[hypothesis_id] = key
        if (not isinstance(names, list) or not names or
                any(not isinstance(x, str) or not x for x in names)):
            raise ValueError("timing group %r has invalid feature_names" % key)
        if not isinstance(phase, str) or not phase:
            raise ValueError("timing group %r has invalid phase" % key)
        covered.extend(names)
    expected = list(FEATURE_NAMES)
    duplicates = sorted({name for name in covered if covered.count(name) > 1})
    missing = sorted(set(expected) - set(covered))
    unknown = sorted(set(covered) - set(expected))
    if duplicates or missing or unknown or len(covered) != len(expected):
        raise ValueError(
            "analysis timing coverage mismatch: duplicate=%r missing=%r unknown=%r"
            % (duplicates, missing, unknown))
    return True


def load_hypothesis_selection(path, groups, excluded_hypothesis_ids=None):
    """Validate a final-run selection and resolve it to executable group keys.

    Selection is deliberately expressed using excluded hypothesis ids.  A
    multi-hypothesis timing group is an indivisible computation unit, so partial
    exclusion is rejected instead of claiming savings that cannot occur.
    """
    validate_analysis_timing_groups(groups)
    all_ids = sorted({int(i) for group in groups for i in group["hypothesis_ids"]})
    all_keys = [group["key"] for group in groups]
    if path is not None and excluded_hypothesis_ids is not None:
        raise ValueError(
            "--hypothesis-selection and --exclude-hypotheses are mutually exclusive")
    if path is None and excluded_hypothesis_ids is None:
        return {
            "path": None, "sha256": None,
            "selection_source": "none",
            "source_timing_artifact": None,
            "source_timing_sha256": None,
            "source_timing_feature_inventory_sha256": None,
            "reasons": {},
            "excluded_hypothesis_ids": [],
            "included_hypothesis_ids": all_ids,
            "enabled_analysis_keys": all_keys,
            "excluded_feature_names": [],
        }
    if excluded_hypothesis_ids is not None:
        selection = {
            "schema_version": 1,
            "excluded_hypothesis_ids": list(excluded_hypothesis_ids),
            "reasons": {
                str(i): "Manual CLI exclusion after cost-report review."
                for i in excluded_hypothesis_ids
            },
        }
        selection_source = "manual_cli"
    else:
        with open(path, "r") as fh:
            selection = json.load(fh)
        selection_source = "json_file"
    if not isinstance(selection, dict) or selection.get("schema_version") != 1:
        raise ValueError("hypothesis selection must be a schema_version 1 object")
    excluded = selection.get("excluded_hypothesis_ids")
    if (not isinstance(excluded, list) or
            any(not isinstance(i, int) or isinstance(i, bool) for i in excluded)):
        raise ValueError("excluded_hypothesis_ids must be a list of integers")
    if len(excluded) != len(set(excluded)):
        raise ValueError("excluded_hypothesis_ids contains duplicates")
    unknown = sorted(set(excluded) - set(all_ids))
    if unknown:
        raise ValueError("unknown excluded hypothesis ids: %r" % unknown)

    reasons = selection.get("reasons", {})
    if not isinstance(reasons, dict):
        raise ValueError("hypothesis selection reasons must be an object")
    normalized_reasons = {}
    for raw_id, reason in reasons.items():
        try:
            hypothesis_id = int(raw_id)
        except (TypeError, ValueError):
            raise ValueError("hypothesis selection reason key %r is not an integer" % raw_id)
        if hypothesis_id <= 0 or str(hypothesis_id) != str(raw_id):
            raise ValueError("hypothesis selection reason key %r is invalid" % raw_id)
        if hypothesis_id not in excluded:
            raise ValueError(
                "hypothesis selection reason supplied for non-excluded id %d"
                % hypothesis_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "hypothesis selection reason for id %d must be non-empty"
                % hypothesis_id)
        normalized_reasons[str(hypothesis_id)] = reason.strip()

    timing_path = selection.get("source_timing_artifact")
    timing_sha = selection.get("source_timing_sha256")
    if (timing_path is None) != (timing_sha is None):
        raise ValueError(
            "source_timing_artifact and source_timing_sha256 must be supplied together")
    timing_inventory_sha = None
    if timing_path is not None:
        if not isinstance(timing_path, str) or not timing_path:
            raise ValueError("source_timing_artifact must be a non-empty path")
        if not isinstance(timing_sha, str) or len(timing_sha) != 64:
            raise ValueError("source_timing_sha256 must be a SHA-256 hex digest")
        timing_path = os.path.abspath(timing_path)
        actual_timing_sha = _sha256_file(timing_path)
        if actual_timing_sha != timing_sha:
            raise ValueError(
                "source timing SHA-256 %r != selection provenance %r"
                % (actual_timing_sha, timing_sha))
        with open(timing_path, "r") as fh:
            timing = json.load(fh)
        if (not isinstance(timing, dict) or timing.get("schema_version") != 2 or
                timing.get("mode") != "measuretime" or
                timing.get("status") != "complete"):
            raise ValueError(
                "source timing artifact must be a complete schema-v2 measuretime run")
        timing_metadata = timing.get("metadata")
        if not isinstance(timing_metadata, dict):
            raise ValueError("source timing artifact metadata must be an object")
        if timing_metadata.get("detector_sha256") != _sha256_file(__file__):
            raise ValueError(
                "source timing artifact was produced by a different detector")
        timing_inventory_sha = timing_metadata.get("feature_inventory_sha256")
    excluded_set = set(excluded)
    enabled_keys = []
    excluded_features = []
    for group in groups:
        members = set(group["hypothesis_ids"])
        removed = members & excluded_set
        if removed and removed != members:
            raise ValueError(
                "partial exclusion of shared selection unit %r is invalid; "
                "exclude all hypothesis ids %r or none"
                % (group["key"], group["hypothesis_ids"]))
        if removed:
            excluded_features.extend(group["feature_names"])
        else:
            enabled_keys.append(group["key"])
    included = sorted(set(all_ids) - excluded_set)
    if not included:
        raise ValueError("hypothesis selection cannot exclude every hypothesis")
    return {
        "path": os.path.abspath(path) if path is not None else None,
        "sha256": _sha256_file(path) if path is not None else None,
        "selection_source": selection_source,
        "source_timing_artifact": timing_path,
        "source_timing_sha256": timing_sha,
        "source_timing_feature_inventory_sha256": timing_inventory_sha,
        "reasons": normalized_reasons,
        "excluded_hypothesis_ids": sorted(excluded_set),
        "included_hypothesis_ids": included,
        "enabled_analysis_keys": enabled_keys,
        "excluded_feature_names": excluded_features,
    }


class AnalysisTimingRecorder(object):
    """Streaming aggregate profiler with bounded top-K and atomic checkpoints."""

    def __init__(self, path, groups, metadata, checkpoint_interval_s=60.0,
                 top_k=20):
        validate_analysis_timing_groups(groups)
        self.path = os.path.abspath(path)
        self.groups = [dict(g) for g in groups]
        self.group_by_key = {g["key"]: g for g in self.groups}
        self.metadata = dict(metadata)
        self.checkpoint_interval_s = float(checkpoint_interval_s)
        self.top_k = int(top_k)
        self.started_wall = time.time()
        self.started_perf = time.perf_counter()
        self.last_checkpoint_perf = 0.0
        self.status = "running"
        self.error = None
        self.active_analysis = None
        self.wall_seconds = {}
        self.phases = {}
        self.analyses = {}
        for group in self.groups:
            self.analyses[group["key"]] = {
                "considered_calls": 0,
                "eligible_calls": 0,
                "total_seconds": 0.0,
                "min_seconds": None,
                "max_seconds": None,
                "slowest": [],
            }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.checkpoint(force=True)

    def update_metadata(self, **values):
        self.metadata.update(values)
        self.checkpoint()

    def record_wall_seconds(self, name, seconds):
        self.wall_seconds[str(name)] = float(seconds)
        self.checkpoint(force=True)

    def start_phase(self, name):
        name = str(name)
        state = self.phases.setdefault(name, {"calls": 0, "total_seconds": 0.0})
        state["calls"] += 1
        return (name, time.perf_counter())

    def stop_phase(self, token):
        if token is None:
            return
        name, started = token
        self.phases[name]["total_seconds"] += time.perf_counter() - started
        self.checkpoint(force=True)

    def start_analysis(self, key, eligible=True, context=None):
        if key not in self.group_by_key:
            raise KeyError("unknown analysis timing group: %r" % key)
        agg = self.analyses[key]
        agg["considered_calls"] += 1
        if not eligible:
            self.checkpoint()
            return None
        agg["eligible_calls"] += 1
        token = {
            "key": key,
            "started": time.perf_counter(),
            "context": dict(context or {}),
        }
        self.active_analysis = {
            "key": key,
            "hypothesis_ids": self.group_by_key[key]["hypothesis_ids"],
            "feature_names": self.group_by_key[key]["feature_names"],
            "context": token["context"],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.checkpoint()
        return token

    def stop_analysis(self, token):
        if token is None:
            return
        elapsed = time.perf_counter() - token["started"]
        agg = self.analyses[token["key"]]
        agg["total_seconds"] += elapsed
        agg["min_seconds"] = (elapsed if agg["min_seconds"] is None
                              else min(agg["min_seconds"], elapsed))
        agg["max_seconds"] = (elapsed if agg["max_seconds"] is None
                              else max(agg["max_seconds"], elapsed))
        item = dict(token["context"])
        item["seconds"] = elapsed
        agg["slowest"].append(item)
        agg["slowest"].sort(key=lambda x: x["seconds"], reverse=True)
        del agg["slowest"][self.top_k:]
        self.active_analysis = None
        self.checkpoint()

    def finish(self):
        self.status = "complete"
        self.active_analysis = None
        self.checkpoint(force=True)

    def fail(self, error):
        self.status = "failed"
        self.error = str(error)
        self.checkpoint(force=True)

    def _snapshot(self):
        analyses = []
        phase_analysis_totals = {}
        for group in self.groups:
            agg = self.analyses[group["key"]]
            eligible = agg["eligible_calls"]
            row = dict(group)
            row.update({
                "considered_calls": agg["considered_calls"],
                "eligible_calls": eligible,
                "total_seconds": agg["total_seconds"],
                "mean_seconds": (agg["total_seconds"] / eligible
                                 if eligible else None),
                "min_seconds": agg["min_seconds"],
                "max_seconds": agg["max_seconds"],
                "slowest": list(agg["slowest"]),
            })
            analyses.append(row)
            phase_analysis_totals[group["phase"]] = (
                phase_analysis_totals.get(group["phase"], 0.0)
                + agg["total_seconds"])
        phases = []
        for name in sorted(self.phases):
            phase = self.phases[name]
            attributed = phase_analysis_totals.get(name, 0.0)
            phases.append({
                "phase": name,
                "calls": phase["calls"],
                "total_seconds": phase["total_seconds"],
                "analysis_seconds": attributed,
                "shared_unattributed_seconds": max(
                    0.0, phase["total_seconds"] - attributed),
            })
        hypothesis_costs = []
        selection_units = []
        for row in analyses:
            ids = list(row["hypothesis_ids"])
            shared = len(ids) > 1
            selection_units.append({
                "key": row["key"],
                "hypothesis_ids": ids,
                "feature_names": list(row["feature_names"]),
                "phase": row["phase"],
                "shared": shared,
                "estimated_removable_seconds": row["total_seconds"],
                "cost_scope": self.metadata.get(
                    "cost_scope", "sample_observed_not_full_run"),
                "eligible_calls": row["eligible_calls"],
            })
            for hypothesis_id in ids:
                hypothesis_costs.append({
                    "hypothesis_id": hypothesis_id,
                    "selection_unit": row["key"],
                    "selection_unit_hypothesis_ids": ids,
                    "feature_names": list(row["feature_names"]),
                    "phase": row["phase"],
                    "exclusive_seconds": (None if shared else row["total_seconds"]),
                    "shared_selection_unit_seconds": (
                        row["total_seconds"] if shared else 0.0),
                    "estimated_removable_seconds_if_excluded_alone": (
                        0.0 if shared else row["total_seconds"]),
                    "estimated_removable_seconds_if_unit_excluded": row["total_seconds"],
                    "eligible_calls": row["eligible_calls"],
                    "considered_calls": row["considered_calls"],
                })
        hypothesis_costs.sort(key=lambda row: row["hypothesis_id"])
        return {
            "schema_version": 2,
            "mode": "measuretime",
            "status": self.status,
            "error": self.error,
            "started_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_wall)),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_wall_seconds": time.perf_counter() - self.started_perf,
            "metadata": self.metadata,
            "wall_seconds": dict(self.wall_seconds),
            "active_analysis": self.active_analysis,
            "phases": phases,
            "analyses": analyses,
            "selection_units": selection_units,
            "hypothesis_costs": hypothesis_costs,
            "selection_guidance": {
                "decision_metric": "estimated_removable_seconds",
                "estimate_scope": (
                    "Observed time in the small profiling sample only, not a "
                    "full-run projection. Phase-level shared_unattributed_seconds "
                    "is not allocated to units; use rankings as a rough guide."),
                "shared_unit_rule": (
                    "A shared selection unit saves its measured time only when all "
                    "member hypotheses are excluded."),
                "warning": (
                    "Cost is not scientific or predictive value; review both before "
                    "creating the final hypothesis selection."),
            },
        }

    def checkpoint(self, force=False):
        now = time.perf_counter()
        if (not force and self.last_checkpoint_perf and
                now - self.last_checkpoint_perf < self.checkpoint_interval_s):
            return
        tmp = self.path + ".tmp.%d" % os.getpid()
        with open(tmp, "w") as fh:
            json.dump(self._snapshot(), fh, indent=2, default=_json_default)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        self.last_checkpoint_perf = now


def write_hypothesis_cost_artifacts(timing_path, out_dir, brain):
    """Write a ranked human report and an editable final-run selection file."""
    with open(timing_path, "r") as fh:
        timing = json.load(fh)
    units = sorted(
        timing["selection_units"],
        key=lambda row: row["estimated_removable_seconds"], reverse=True)
    report_path = os.path.join(
        out_dir, "hypothesis_cost_report_%s.md" % brain)
    lines = [
        "# Hypothesis computational-cost report: %s" % brain,
        "",
        "This is a small-sample profile (%s of %s adjudicable segments; limit %s). "
        "Seconds are observed sample cost, not projected full-run savings." % (
            timing.get("metadata", {}).get("n_profiled_segments", "?"),
            timing.get("metadata", {}).get("n_total_adjudicable_segments", "?"),
            timing.get("metadata", {}).get("profile_occurrence_limit", "?")),
        "Use the ranking only as a rough computational-cost guide. It is not a "
        "measure of scientific or predictive value, and phase-level shared "
        "overhead is not assigned to individual units.",
        "",
        "| selection unit | hypothesis ids | phase | observed sample seconds | "
        "eligible / considered | features |",
        "|---|---:|---|---:|---:|---|",
    ]
    for unit in units:
        lines.append(
            "| %s | %s | %s | %.6f | %d / %d | %s |" % (
                unit["key"],
                ", ".join(str(i) for i in unit["hypothesis_ids"]),
                unit["phase"], unit["estimated_removable_seconds"],
                unit["eligible_calls"],
                next(row["considered_calls"] for row in timing["analyses"]
                     if row["key"] == unit["key"]),
                ", ".join(unit["feature_names"])))
    lines.extend([
        "",
        "A row containing multiple hypothesis ids is indivisible: excluding only "
        "one member does not remove the shared computation. Exclude all ids in "
        "that row or none.",
        "",
        "Edit the generated `hypothesis_selection_template_%s.json`, then pass "
        "it to the final run with `--hypothesis-selection`." % brain,
        "For a quick manual choice, pass ids directly as "
        "`--exclude-hypotheses <ID1> <ID2>`. The same shared-unit validation applies.",
        "",
    ])
    tmp = report_path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, report_path)

    selection_path = os.path.join(
        out_dir, "hypothesis_selection_template_%s.json" % brain)
    selection = {
        "schema_version": 1,
        "source_timing_artifact": os.path.abspath(timing_path),
        "source_timing_sha256": _sha256_file(timing_path),
        "excluded_hypothesis_ids": [],
        "reasons": {},
    }
    tmp = selection_path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump(selection, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, selection_path)
    return report_path, selection_path


# ------------------------------------------------------------------------- #
# FEATURE IMPLEMENTATION FRAGMENT (spliced at __DETECTOR_FEATURE_IMPLEMENTATION__)
#
# Owns ONLY feature semantics: geometry/data helpers, FEATURE_REGISTRY (in
# feature_inventory.json order), FEATURE_NAMES, SegmentAccumulator, and
# extract_features(payload, verbose=True, timing=None,
# enabled_analysis_keys=None).  Model selection, CLI, main,
# figures, output writers and the audit driver live in the runtime template
# and are NOT redefined here.
#
# Feature math, constants, reductions, and measurable conditions are copied
# verbatim from each hypothesis's SHA-256-verified source script named in the
# inventory (hypo_18 -> FIXED, all others -> rerun).  Install/sandbox/cloud
# scaffolding from those scripts is removed.  Definedness is decided by explicit
# measured-membership during extraction; undefined numeric features are emitted
# as NaN with a companion <feature>_is_defined flag.  A real measurement that
# equals a historical sentinel (0.0/1.0/180.0/-1.0/2.0) stays DEFINED.
# ------------------------------------------------------------------------- #

import heapq
import itertools
import math
from collections import deque

# numpy is imported as np at module scope by the runtime template.


# ===================================================================== #
# Shared geometry / graph helpers.                                      #
# ===================================================================== #
def _dist(g, i, j):
    """Euclidean distance between two node coordinates."""
    return float(np.linalg.norm(g.node_xyz[i] - g.node_xyz[j]))


def _branch_direction(g, node, nbr, reach_um):
    """Unit direction of the branch leaving `node` toward `nbr`, walking the
    degree-2 chain until `reach_um` accumulated length (hypo_3/25/26 form:
    return v/nrm, else the raw v)."""
    prev, cur = node, nbr
    acc = _dist(g, node, nbr)
    steps = 0
    while acc < reach_um and steps < 1000:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        acc += _dist(g, prev, cur)
        steps += 1
    v = g.node_xyz[cur] - g.node_xyz[node]
    nrm = np.linalg.norm(v)
    if nrm > 0:
        return v / nrm
    return v


def _branch_radius_from_nbr(g, start_node, nbr_node, reach_um):
    """Mean node radius along the branch leaving `start_node` toward `nbr_node`,
    seeding radii=[node_radius[nbr_node]] (hypo_4/49 form)."""
    prev, cur = start_node, nbr_node
    acc = _dist(g, start_node, nbr_node)
    radii = [g.node_radius[cur]]
    steps = 0
    while acc < reach_um and steps < 1000:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        acc += _dist(g, prev, cur)
        radii.append(g.node_radius[cur])
        steps += 1
    return float(np.mean(radii))


def _branch_mean_radius(g, start_node, nbr_node, reach_um):
    """Mean node radius along the branch, seeding radii=[r_start, r_nbr]
    (hypo_35 form: get_branch_mean_radius)."""
    prev, cur = start_node, nbr_node
    acc = _dist(g, start_node, nbr_node)
    radii = [g.node_radius[start_node], g.node_radius[nbr_node]]
    steps = 0
    while acc < reach_um and steps < 1000:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        acc += _dist(g, prev, cur)
        radii.append(g.node_radius[cur])
        steps += 1
    return float(np.mean(radii))


def _branch_direction_and_radius(g, node, nbr, reach_um):
    """hypo_47 form: return (unit_direction, mean_radius) where the radii are
    taken over the interior path nodes (path_list[1:]) with a node_radius
    fallback and NaN filtering."""
    prev, cur = node, nbr
    path = {node, cur}
    path_list = [node, cur]
    acc = _dist(g, node, cur)
    steps = 0
    while acc < reach_um and steps < 1000:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        if nxt[0] in path:
            break
        prev, cur = cur, nxt[0]
        acc += _dist(g, prev, cur)
        path.add(cur)
        path_list.append(cur)
        steps += 1
    v = g.node_xyz[cur] - g.node_xyz[node]
    nrm = np.linalg.norm(v)
    direction = v / nrm if nrm > 0 else np.zeros(3)
    radii = [g.node_radius[n] for n in path_list[1:]]
    valid_radii = [r for r in radii if not np.isnan(r)]
    mean_radius = float(np.mean(valid_radii)) if valid_radii else float(g.node_radius[node])
    return direction, mean_radius


def _get_branch_direction(g, node, nbr, reach_um):
    """hypo_46 form: unit direction, walking degree-2 chain to reach_um; returns
    a zero vector when degenerate."""
    prev, cur = node, nbr
    acc = _dist(g, node, nbr)
    steps = 0
    while acc < reach_um and steps < 1000:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        acc += _dist(g, prev, cur)
        steps += 1
    v = g.node_xyz[cur] - g.node_xyz[node]
    nrm = np.linalg.norm(v)
    if nrm > 0:
        return v / nrm
    return np.zeros(3)


def _compute_tortuosity(g, node, nbr, reach_um):
    """hypo_13 form: geodesic/euclidean tortuosity of the branch leaving `node`
    toward `nbr`; returns 1.0 when the euclidean span is zero."""
    prev, cur = node, nbr
    geo = _dist(g, node, nbr)
    steps = 0
    while geo < reach_um and steps < 1000:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        geo += _dist(g, prev, cur)
        steps += 1
    euclid = _dist(g, node, cur)
    if euclid == 0:
        return 1.0
    return geo / euclid


def _comp_nodes_map(frag):
    """component_id -> list of node ids (single shared pass)."""
    comp_nodes = defaultdict(list)
    for n in frag.nodes:
        comp_nodes[int(frag.node_component_id[n])].append(n)
    return comp_nodes


def _bbox_diag(coords):
    mn = np.min(coords, axis=0)
    mx = np.max(coords, axis=0)
    return float(np.linalg.norm(mx - mn))


def _intra_comp_cable_length(g, nodes):
    """Sum of euclidean edge lengths among edges of the induced subgraph on
    `nodes`."""
    node_set = set(nodes)
    total = 0.0
    for u in nodes:
        for w in g.neighbors(u):
            if w in node_set and u < w:
                total += _dist(g, u, w)
    return total


# ---- unweighted double-BFS pseudo-diameter (hypo_19) ---- #
def _get_diameter_path_unweighted(g, comp_nodes):
    node_set = set(comp_nodes)
    if len(node_set) < 2:
        return list(node_set)

    def bfs(start):
        visited = {start: None}
        q = deque([start])
        last = start
        while q:
            cur = q.popleft()
            last = cur
            for nb in g.neighbors(cur):
                if nb in node_set and nb not in visited:
                    visited[nb] = cur
                    q.append(nb)
        return last, visited

    far1, _ = bfs(comp_nodes[0])
    far2, parents = bfs(far1)
    path = []
    cur = far2
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


# ---- weighted double-Dijkstra pseudo-diameter length (hypo_20) ---- #
def _double_dijkstra_length(g, comp_nodes):
    node_set = set(comp_nodes)
    if len(node_set) < 2:
        return 0.0, comp_nodes[0], comp_nodes[0]

    def dijkstra(src):
        dist_map = {src: 0.0}
        pq = [(0.0, src)]
        far = src
        far_d = 0.0
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist_map.get(u, float("inf")):
                continue
            if d > far_d:
                far_d = d
                far = u
            for w in g.neighbors(u):
                if w not in node_set:
                    continue
                nd = d + _dist(g, u, w)
                if nd < dist_map.get(w, float("inf")):
                    dist_map[w] = nd
                    heapq.heappush(pq, (nd, w))
        return far, far_d

    a, _ = dijkstra(comp_nodes[0])
    b, graph_dist = dijkstra(a)
    return graph_dist, a, b


# ---- hop-count farthest BFS + longest path (hypo_29) ---- #
def _bfs_farthest(g, start, allowed):
    visited = {start: None}
    q = deque([start])
    last = start
    while q:
        cur = q.popleft()
        last = cur
        for nb in g.neighbors(cur):
            if nb in allowed and nb not in visited:
                visited[nb] = cur
                q.append(nb)
    return last, visited


def _get_longest_path(g, comp_nodes):
    allowed = set(comp_nodes)
    if len(allowed) < 2:
        return list(allowed)
    far1, _ = _bfs_farthest(g, comp_nodes[0], allowed)
    far2, parents = _bfs_farthest(g, far1, allowed)
    path = []
    cur = far2
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


# ---- weighted farthest (hypo_37 BFSFinder over an adjacency list) ---- #
def _weighted_farthest(adj, start):
    dist_map = {start: 0.0}
    pq = [(0.0, start)]
    far = start
    far_d = 0.0
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist_map.get(u, float("inf")):
            continue
        if d > far_d:
            far_d = d
            far = u
        for w, wt in adj.get(u, ()):  # (neighbor, weight)
            nd = d + wt
            if nd < dist_map.get(w, float("inf")):
                dist_map[w] = nd
                heapq.heappush(pq, (nd, w))
    return far, far_d


# ---- edge-betweenness helpers ---- #
def _get_max_ebc_istree(subg, nx):
    """hypo_21: exact tree betweenness for trees, else k-sampled EBC."""
    N = subg.number_of_nodes()
    if N < 2 or subg.number_of_edges() == 0:
        return 0.0
    if nx.is_tree(subg):
        # exact edge betweenness on a tree: for a bridge splitting into
        # sizes n1,n2 the number of shortest paths across it is n1*n2,
        # normalized by N*(N-1)/2.
        best = 0.0
        norm = N * (N - 1) / 2.0
        for u, v in subg.edges():
            subg.remove_edge(u, v)
            comp = None
            for cc in nx.connected_components(subg):
                if u in cc:
                    comp = cc
                    break
            n1 = len(comp) if comp is not None else 1
            n2 = N - n1
            subg.add_edge(u, v)
            ebc = (n1 * n2) / norm if norm > 0 else 0.0
            if ebc > best:
                best = ebc
        return best
    if N > 10000:
        k = 10
    elif N > 1000:
        k = 50
    elif N > 200:
        k = min(100, N)
    else:
        k = None
    ebc = nx.edge_betweenness_centrality(subg, k=k, normalized=True, seed=RANDOM_SEED)
    return max(ebc.values()) if ebc else 0.0


def _get_max_ebc_contracted(subg, nx):
    """hypo_40: contract all degree-2 chains, keep degree!=2 anchors, then
    k-sampled edge betweenness on the reduced graph; None when too small."""
    keep_nodes = [n for n in subg.nodes if subg.degree(n) != 2]
    if len(keep_nodes) < 10:
        return None
    H = nx.Graph()
    H.add_nodes_from(keep_nodes)
    keep_set = set(keep_nodes)
    visited_edges = set()
    for anchor in keep_nodes:
        for start in subg.neighbors(anchor):
            ekey = tuple(sorted((anchor, start)))
            if ekey in visited_edges:
                continue
            prev, cur = anchor, start
            steps = 0
            while cur not in keep_set and steps < 100000:
                nxts = [k for k in subg.neighbors(cur) if k != prev]
                if len(nxts) != 1:
                    break
                prev, cur = cur, nxts[0]
                steps += 1
            visited_edges.add(ekey)
            visited_edges.add(tuple(sorted((prev, cur))))
            if cur in keep_set and cur != anchor:
                # hypo_40 contracts topology only and calls unweighted edge
                # betweenness.  Do not calculate geometry from this plain
                # networkx subgraph: unlike the original SkeletonGraph, it has
                # no node_xyz attribute, and the unused weight would not affect
                # the source feature anyway.
                H.add_edge(anchor, cur)
    if H.number_of_nodes() < 10 or H.number_of_edges() < 1:
        return None
    k = 300 if H.number_of_nodes() > 300 else None
    ebc = nx.edge_betweenness_centrality(H, k=k, normalized=True, seed=RANDOM_SEED)
    return max(ebc.values()) if ebc else None


# ---- branching-frequency mismatch (hypo_27) ---- #
def _compute_mismatch_score(g, comp_nodes, nx):
    if len(comp_nodes) < 5:
        return 0.0
    subg = g.subgraph(comp_nodes)
    if subg.number_of_edges() == 0:
        return 0.0
    root = comp_nodes[0]
    tree = nx.bfs_tree(subg, root)
    order = list(nx.dfs_postorder_nodes(tree, root))
    children = {n: list(tree.successors(n)) for n in tree.nodes}
    subtree_len = defaultdict(float)
    subtree_branch = defaultdict(int)
    for n in order:
        deg = subg.degree(n)
        if deg > 2:
            subtree_branch[n] += 1
        for c in children[n]:
            subtree_len[n] += subtree_len[c] + _dist(g, n, c)
            subtree_branch[n] += subtree_branch[c]
    freqs = []
    for n in tree.nodes:
        for c in children[n]:
            c_length = subtree_len[c] + _dist(g, n, c)
            if c_length >= 15.0:
                freqs.append(subtree_branch[c] / c_length)
    total_len = subtree_len[root]
    if total_len >= 15.0:
        freqs.append(subtree_branch[root] / total_len)
    if len(freqs) >= 2:
        return float(max(freqs) - min(freqs))
    return 0.0


# ---- best bridge radius-variance ratio (hypo_39) ---- #
def _get_best_bridge_var_ratio(g, comp_nodes, nx):
    if len(comp_nodes) < 20:
        return None
    subg = g.subgraph(comp_nodes)
    try:
        bridges = list(nx.bridges(subg))
    except Exception:
        return None
    if not bridges:
        return None
    N = len(comp_nodes)
    candidates = []
    for (u, v) in bridges:
        subg2 = nx.Graph(subg)
        subg2.remove_edge(u, v)
        side_u = None
        for cc in nx.connected_components(subg2):
            if u in cc:
                side_u = cc
                break
        if side_u is None:
            continue
        n1 = len(side_u)
        n2 = N - n1
        if n1 >= 10 and n2 >= 10:
            balance = min(n1, n2)
            candidates.append((balance, u, v, side_u))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    best_ratio = None
    for balance, u, v, side_u in candidates[:10]:
        side_v = [n for n in comp_nodes if n not in side_u]
        r_u = [g.node_radius[n] for n in side_u if not np.isnan(g.node_radius[n])]
        r_v = [g.node_radius[n] for n in side_v if not np.isnan(g.node_radius[n])]
        if len(r_u) < 2 or len(r_v) < 2:
            continue
        var_u = float(np.var(r_u))
        var_v = float(np.var(r_v))
        hi = max(var_u, var_v, 1e-6)
        lo = min(var_u, var_v)
        lo = max(lo, 1e-6)
        ratio = hi / lo
        if best_ratio is None or ratio > best_ratio:
            best_ratio = ratio
    return best_ratio


# ---- intra-component branch tortuosity variance (hypo_45) ---- #
def _extract_branches(g, comp_nodes):
    """Split a component into branches at terminals (degree != 2); cycles get
    one arbitrary terminal (hypo_45 form)."""
    node_set = set(comp_nodes)
    deg = {n: sum(1 for w in g.neighbors(n) if w in node_set) for n in comp_nodes}
    terminals = [n for n in comp_nodes if deg[n] != 2]
    if not terminals:
        terminals = [comp_nodes[0]]
    branches = []
    visited_edges = set()
    for t in terminals:
        for start in g.neighbors(t):
            if start not in node_set:
                continue
            ekey = tuple(sorted((t, start)))
            if ekey in visited_edges:
                continue
            branch = [t, start]
            visited_edges.add(ekey)
            prev, cur = t, start
            steps = 0
            while deg.get(cur, 0) == 2 and steps < 100000:
                nxts = [k for k in g.neighbors(cur)
                        if k in node_set and k != prev]
                if len(nxts) != 1:
                    break
                prev, cur = cur, nxts[0]
                branch.append(cur)
                visited_edges.add(tuple(sorted((prev, cur))))
                steps += 1
            branches.append(branch)
    return branches


# ---- x-crossing disjoint pair helpers (hypo_41) ---- #
def _best_two_disjoint_pairs(dirs):
    """hypo_41: over >=4 directions find the two disjoint pairs whose combined
    anti-parallel-ness is best; return (best_val, min_dot) with default
    (2.0, 2.0)."""
    n = len(dirs)
    if n < 4:
        return 2.0, 2.0
    best_val = 2.0
    best_min_dot = 2.0
    idx = list(range(n))
    for combo in itertools.combinations(idx, 4):
        a, b, c, d = combo
        pairings = [
            ((a, b), (c, d)),
            ((a, c), (b, d)),
            ((a, d), (b, c)),
        ]
        for (p1, p2) in pairings:
            dot1 = float(np.dot(dirs[p1[0]], dirs[p1[1]]))
            dot2 = float(np.dot(dirs[p2[0]], dirs[p2[1]]))
            val = max(dot1, dot2)  # both must be anti-parallel => minimize the worst
            if val < best_val:
                best_val = val
                best_min_dot = min(dot1, dot2)
    return best_val, best_min_dot


# ===================================================================== #
# Per-feature reducers.                                                 #
# Each returns (value, defined) for one segment.  None columns keep     #
# undefined -> (nan, False).                                            #
# ===================================================================== #
class SegmentAccumulator:
    """Collects one row per adjudicable segment.  For every feature it stores
    the numeric value (NaN when undefined) and an explicit is_defined flag
    derived from measured membership, NOT from any sentinel comparison."""

    def __init__(self, seg_ids):
        self._seg_ids = list(seg_ids)
        self._pos = {s: i for i, s in enumerate(self._seg_ids)}
        n = len(self._seg_ids)
        self._values = {f.name: np.full(n, np.nan, dtype=float)
                        for f in FEATURE_REGISTRY}
        self._defined = {f.name: np.zeros(n, dtype=bool)
                         for f in FEATURE_REGISTRY}

    def set(self, name, seg_id, value):
        """Record a genuinely measured value for one segment/feature."""
        i = self._pos.get(seg_id)
        if i is None:
            return
        self._values[name][i] = float(value)
        self._defined[name][i] = True

    def has(self, name, seg_id):
        i = self._pos.get(seg_id)
        if i is None:
            return False
        return bool(self._defined[name][i])

    def get(self, name, seg_id):
        i = self._pos.get(seg_id)
        if i is None:
            return np.nan
        return self._values[name][i]

    def to_frame(self):
        import pandas as pd

        data = {}
        for f in FEATURE_REGISTRY:
            data[f.name] = self._values[f.name]
            data[f.name + "_is_defined"] = self._defined[f.name].astype(bool)
        return pd.DataFrame(data, index=range(len(self._seg_ids)))


class FeatureSpec:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


# FEATURE_REGISTRY owns public names and inventory order only. Execution phases
# and selectable computation ownership live solely in ANALYSIS_TIMING_GROUPS.
FEATURE_REGISTRY = [
    FeatureSpec("max_windowed_path_tortuosity"),
    FeatureSpec("max_euclidean_geodesic_wraparound_ratio"),
    FeatureSpec("min_junction_branch_angle_deg"),
    FeatureSpec("max_degree3_radius_asymmetry"),
    FeatureSpec("max_normalized_edge_betweenness_bridge"),
    FeatureSpec("max_euclidean_edge_jump"),
    FeatureSpec("box_counting_fractal_dimension"),
    FeatureSpec("component_aspect_ratio"),
    FeatureSpec("component_spatial_density"),
    FeatureSpec("max_junction_tortuosity_variance"),
    FeatureSpec("gmm_bimodality_bic_gain"),
    FeatureSpec("p95_internode_radius_cv"),
    FeatureSpec("max_leaf_cluster_silhouette"),
    FeatureSpec("max_radius_step_along_pseudo_diameter"),
    FeatureSpec("pseudo_diameter_tortuosity"),
    FeatureSpec("max_edge_betweenness_centrality"),
    FeatureSpec("neg_min_component_spatial_density"),
    FeatureSpec("max_high_degree_node_density"),
    FeatureSpec("neg_min_uturn_angle_deg"),
    FeatureSpec("min_junction_cosine_similarity"),
    FeatureSpec("max_branching_frequency_mismatch"),
    FeatureSpec("curvature_spike_density"),
    FeatureSpec("neg_min_radius_assortativity"),
    FeatureSpec("neg_min_log_convexhull_cable_density"),
    FeatureSpec("max_tapering_reversal_rate"),
    FeatureSpec("neg_min_normalized_leaf_nn_distance"),
    FeatureSpec("max_junction_radius_ratio"),
    FeatureSpec("max_thickness_distance_spearman"),
    FeatureSpec("diameter_to_leaf_ratio"),
    FeatureSpec("max_bridge_radius_variance_ratio"),
    FeatureSpec("max_contracted_normalized_edge_betweenness"),
    FeatureSpec("min_xcrossing_disjoint_pair_dot"),
    FeatureSpec("spatial_bimodality_silhouette"),
    FeatureSpec("max_intra_component_tortuosity_variance"),
    FeatureSpec("min_xcrossing_antiparallel_pair_dot"),
    FeatureSpec("t_merge_geometry_score"),
    FeatureSpec("max_local_branch_density"),
    FeatureSpec("max_rall_ratio"),
    FeatureSpec("log_bbox_cable_density"),
]

FEATURE_NAMES = [f.name for f in FEATURE_REGISTRY]

# Stable profiling and final-run selection units. A group may own multiple
# features when the implementation deliberately computes them in one shared
# traversal; every FEATURE_REGISTRY entry appears exactly once.
ANALYSIS_TIMING_GROUPS = [
    {"key": "hypo_8", "hypothesis_ids": [8], "phase": "edge",
     "feature_names": ["max_euclidean_edge_jump"]},
    {"key": "hypo_48", "hypothesis_ids": [48], "phase": "junction",
     "feature_names": ["max_local_branch_density"]},
    {"key": "shared_junction_nodes", "hypothesis_ids": [3, 4, 25, 26, 35, 46, 47, 49],
     "phase": "junction", "feature_names": [
         "min_junction_branch_angle_deg", "max_degree3_radius_asymmetry",
         "neg_min_uturn_angle_deg", "min_junction_cosine_similarity",
         "max_junction_radius_ratio", "min_xcrossing_antiparallel_pair_dot",
         "t_merge_geometry_score", "max_rall_ratio"]},
    {"key": "hypo_41", "hypothesis_ids": [41], "phase": "x_crossing",
     "feature_names": ["min_xcrossing_disjoint_pair_dot"]},
    {"key": "shared_chain_1_15", "hypothesis_ids": [1, 15], "phase": "chain",
     "feature_names": ["max_windowed_path_tortuosity", "p95_internode_radius_cv"]},
    {"key": "hypo_37", "hypothesis_ids": [37], "phase": "component",
     "feature_names": ["diameter_to_leaf_ratio"]},
    {"key": "hypo_6", "hypothesis_ids": [6], "phase": "component",
     "feature_names": ["max_normalized_edge_betweenness_bridge"]},
    {"key": "hypo_2", "hypothesis_ids": [2], "phase": "component",
     "feature_names": ["max_euclidean_geodesic_wraparound_ratio"]},
    {"key": "hypo_13", "hypothesis_ids": [13], "phase": "component",
     "feature_names": ["max_junction_tortuosity_variance"]},
    {"key": "hypo_19", "hypothesis_ids": [19], "phase": "component",
     "feature_names": ["max_radius_step_along_pseudo_diameter"]},
    {"key": "hypo_20", "hypothesis_ids": [20], "phase": "component",
     "feature_names": ["pseudo_diameter_tortuosity"]},
    {"key": "hypo_21", "hypothesis_ids": [21], "phase": "component",
     "feature_names": ["max_edge_betweenness_centrality"]},
    {"key": "hypo_22", "hypothesis_ids": [22], "phase": "component",
     "feature_names": ["neg_min_component_spatial_density"]},
    {"key": "hypo_23", "hypothesis_ids": [23], "phase": "component",
     "feature_names": ["max_high_degree_node_density"]},
    {"key": "hypo_27", "hypothesis_ids": [27], "phase": "component",
     "feature_names": ["max_branching_frequency_mismatch"]},
    {"key": "hypo_29", "hypothesis_ids": [29], "phase": "component",
     "feature_names": ["curvature_spike_density"]},
    {"key": "hypo_30", "hypothesis_ids": [30], "phase": "component",
     "feature_names": ["neg_min_radius_assortativity"]},
    {"key": "hypo_31", "hypothesis_ids": [31], "phase": "component",
     "feature_names": ["neg_min_log_convexhull_cable_density"]},
    {"key": "hypo_32", "hypothesis_ids": [32], "phase": "component",
     "feature_names": ["max_tapering_reversal_rate"]},
    {"key": "hypo_34", "hypothesis_ids": [34], "phase": "component",
     "feature_names": ["neg_min_normalized_leaf_nn_distance"]},
    {"key": "hypo_36", "hypothesis_ids": [36], "phase": "component",
     "feature_names": ["max_thickness_distance_spearman"]},
    {"key": "hypo_39", "hypothesis_ids": [39], "phase": "component",
     "feature_names": ["max_bridge_radius_variance_ratio"]},
    {"key": "hypo_45", "hypothesis_ids": [45], "phase": "component",
     "feature_names": ["max_intra_component_tortuosity_variance"]},
    {"key": "hypo_40", "hypothesis_ids": [40], "phase": "component",
     "feature_names": ["max_contracted_normalized_edge_betweenness"]},
    {"key": "hypo_11", "hypothesis_ids": [11], "phase": "component",
     "feature_names": ["component_aspect_ratio", "component_spatial_density"]},
    {"key": "hypo_18", "hypothesis_ids": [18], "phase": "component",
     "feature_names": ["max_leaf_cluster_silhouette"]},
    {"key": "hypo_9", "hypothesis_ids": [9], "phase": "segment",
     "feature_names": ["box_counting_fractal_dimension"]},
    {"key": "hypo_14", "hypothesis_ids": [14], "phase": "segment",
     "feature_names": ["gmm_bimodality_bic_gain"]},
    {"key": "hypo_44", "hypothesis_ids": [44], "phase": "segment",
     "feature_names": ["spatial_bimodality_silhouette"]},
    {"key": "hypo_50", "hypothesis_ids": [50], "phase": "segment",
     "feature_names": ["log_bbox_cable_density"]},
]


# ===================================================================== #
# extract_features — shared traversal passes over the fragments graph.  #
# ===================================================================== #
def extract_features(payload, verbose=True, timing=None,
                     enabled_analysis_keys=None, profile_segment_limit=None):
    import networkx as nx
    from scipy.spatial import KDTree, ConvexHull
    from scipy.stats import spearmanr, linregress
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    def _phase_start(name):
        return timing.start_phase(name) if timing is not None else None

    def _phase_stop(token):
        if timing is not None:
            timing.stop_phase(token)

    enabled_analysis_keys = (None if enabled_analysis_keys is None else
                             frozenset(enabled_analysis_keys))

    def _analysis_enabled(key):
        return enabled_analysis_keys is None or key in enabled_analysis_keys

    def _analysis_start(key, eligible=True, context=None):
        if timing is None or not _analysis_enabled(key):
            return None
        return timing.start_analysis(key, eligible=eligible, context=context)

    def _analysis_stop(token):
        if timing is not None:
            timing.stop_analysis(token)

    phase_token = _phase_start("indexing")
    frag = payload["fragments_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    adjudicable_set = set(int(x) for x in np.unique(node_label) if int(x) != 0)
    adjudicable = sorted(adjudicable_set)
    gt_merge_labels = set(int(x) for x in payload["gt_merge_labels"])

    try:
        min_cable_length = int(payload.get("min_cable_length", 100))
    except Exception:
        min_cable_length = 100

    comp_to_seg = build_comp_to_seg(frag)
    degrees = dict(frag.degree())

    _log("[extract] indexing components and segments ...")
    comp_nodes = _comp_nodes_map(frag)
    # component ids that belong to an adjudicable segment
    seg_to_comps = defaultdict(list)
    for comp_id, seg_id in comp_to_seg.items():
        if seg_id in adjudicable_set:
            seg_to_comps[seg_id].append(comp_id)

    if profile_segment_limit is not None:
        if (not isinstance(profile_segment_limit, int) or
                isinstance(profile_segment_limit, bool) or
                profile_segment_limit <= 0):
            raise ValueError("profile_segment_limit must be a positive integer")
        component_bearing = [
            seg_id for seg_id in adjudicable if seg_to_comps.get(seg_id)
        ]
        adjudicable = component_bearing[:profile_segment_limit]
        adjudicable_set = set(adjudicable)
        seg_to_comps = defaultdict(list, {
            seg_id: seg_to_comps[seg_id] for seg_id in adjudicable
        })
        _log("[extract] measuretime sample: %d component-bearing segments: %s"
             % (len(adjudicable), adjudicable))

    is_merge = np.array(
        [1 if s in gt_merge_labels else 0 for s in adjudicable], dtype=np.int64)
    acc = SegmentAccumulator(adjudicable)
    sampled_nodes = None
    if profile_segment_limit is not None:
        sampled_nodes = []
        for seg_id in adjudicable:
            for comp_id in seg_to_comps[seg_id]:
                sampled_nodes.extend(comp_nodes.get(comp_id, ()))
        sampled_nodes = tuple(sampled_nodes)
    pass_nodes = frag.nodes if sampled_nodes is None else sampled_nodes

    # nodes grouped by segment (for per-segment features)
    seg_to_nodes = defaultdict(list)
    for n in pass_nodes:
        c = int(frag.node_component_id[n])
        seg = comp_to_seg.get(c)
        if seg is not None and seg in adjudicable_set:
            seg_to_nodes[seg].append(n)

    node_xyz = frag.node_xyz
    node_radius = frag.node_radius

    # ---- running per-segment reductions (scratch) ---- #
    max_wraparound = {}
    min_branch_angle = {}
    max_deg3_asym = {}
    max_bridge_ebc = {}
    max_edge_jump = {}
    max_junc_tort_var = {}
    max_radius_step = {}
    max_pdiam_tort = {}
    max_ebc = {}
    min_comp_density = {}
    max_hdn_density = {}
    min_uturn_angle = {}
    min_junc_cos = {}
    max_branch_mismatch = {}
    curvature_spike = {}
    min_radius_assort = {}
    min_log_ch_density = {}
    max_taper_rev = {}
    min_leaf_nn = {}
    max_junc_radius_ratio = {}
    max_thick_spearman = {}
    max_bridge_var_ratio = {}
    max_intra_tort_var = {}
    min_xantiparallel = {}
    t_merge_score = {}
    max_rall = {}
    _phase_stop(phase_token)

    # ================================================================= #
    # EDGE PASS: single-edge euclidean jump (hypo_8).                   #
    # ================================================================= #
    _log("[extract] edge pass ...")
    phase_token = _phase_start("edge")
    profile_edges = (frag.edges if sampled_nodes is None
                     else frag.edges(sampled_nodes))
    analysis_token = _analysis_start(
        "hypo_8", context={"edges": (frag.number_of_edges()
                                      if sampled_nodes is None else "sampled")})
    for u, v in (profile_edges if _analysis_enabled("hypo_8") else ()):
        cu = int(frag.node_component_id[u])
        seg = comp_to_seg.get(cu)
        if seg is None or seg not in adjudicable_set:
            continue
        d = _dist(frag, u, v)
        if seg not in max_edge_jump or d > max_edge_jump[seg]:
            max_edge_jump[seg] = d
    for seg, val in max_edge_jump.items():
        acc.set("max_euclidean_edge_jump", seg, val)
    _analysis_stop(analysis_token)
    _phase_stop(phase_token)

    # ================================================================= #
    # JUNCTION / CHAIN PASS over whole-graph nodes by degree.          #
    # Covers hypo_3, 4, 25, 26, 35, 46, 47, 48, 49, and 41 seeds.       #
    # ================================================================= #
    _log("[extract] junction/chain node pass ...")
    phase_token = _phase_start("junction")

    # Full runs use all branch nodes; measuretime uses only sampled segments.
    branch_nodes = ([n for n in pass_nodes if degrees[n] >= 3]
                    if _analysis_enabled("hypo_48") else [])
    analysis_token = _analysis_start(
        "hypo_48", eligible=bool(branch_nodes),
        context={"branch_nodes": len(branch_nodes)})
    max_local_branch_density = {}
    if branch_nodes:
        branch_coords = np.array([node_xyz[n] for n in branch_nodes])
        branch_comp_ids = np.array([int(frag.node_component_id[n]) for n in branch_nodes])
        tree = KDTree(branch_coords)
        try:
            counts = tree.query_ball_point(branch_coords, r=5.0, return_length=True) - 1
        except TypeError:
            indices = tree.query_ball_point(branch_coords, r=5.0)
            counts = np.array([len(idx) - 1 for idx in indices])
        for i in range(len(branch_nodes)):
            seg = comp_to_seg.get(int(branch_comp_ids[i]))
            if seg is None or seg not in adjudicable_set:
                continue
            c = int(counts[i])
            if seg not in max_local_branch_density or c > max_local_branch_density[seg]:
                max_local_branch_density[seg] = c
    for seg, val in max_local_branch_density.items():
        acc.set("max_local_branch_density", seg, val)
    _analysis_stop(analysis_token)

    # per-node junction/chain features
    analysis_token = _analysis_start(
        "shared_junction_nodes", context={"nodes": frag.number_of_nodes()})
    for n in (pass_nodes if _analysis_enabled("shared_junction_nodes") else ()):
        deg = degrees[n]
        comp_id = int(frag.node_component_id[n])
        seg = comp_to_seg.get(comp_id)
        if seg is None or seg not in adjudicable_set:
            continue
        nbrs = list(frag.neighbors(n))

        # --- degree-2 chain-node features (hypo_25 u-turn) --- #
        if deg == 2 and len(nbrs) == 2:
            d1 = _branch_direction(frag, n, nbrs[0], 15.0)
            d2 = _branch_direction(frag, n, nbrs[1], 15.0)
            nrm1 = np.linalg.norm(d1)
            nrm2 = np.linalg.norm(d2)
            if nrm1 != 0 and nrm2 != 0:
                dot = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
                angle = math.degrees(math.acos(dot))
                if seg not in min_uturn_angle or angle < min_uturn_angle[seg]:
                    min_uturn_angle[seg] = angle

        # --- degree-3 junction features (hypo_4 asymmetry, hypo_49 Rall) --- #
        if deg == 3 and len(nbrs) == 3:
            rads4 = [_branch_radius_from_nbr(frag, n, nb, 15.0) for nb in nbrs]
            rads4 = sorted(rads4)
            if rads4[2] > 1e-5:
                asym = rads4[1] / rads4[2]
                if seg not in max_deg3_asym or asym > max_deg3_asym[seg]:
                    max_deg3_asym[seg] = asym

            rads49 = []
            for nb in nbrs:
                r = _branch_radius_from_nbr(frag, n, nb, 5.0)
                if np.isnan(r):
                    r = 0.0
                rads49.append(r)
            rads49.sort(reverse=True)
            r1 = rads49[0]
            ratio = float(rads49[1] / r1) if r1 > 0 else 0.0
            if seg not in max_rall or ratio > max_rall[seg]:
                max_rall[seg] = ratio

        # --- degree>=3 junction features (hypo_3 angle, hypo_26 cosine) --- #
        if deg >= 3:
            dirs = [_branch_direction(frag, n, nb, 15.0) for nb in nbrs]

            # hypo_3: uses ALL neighbor vectors (no norm>0 filter), per-junction
            # min_angle init 180.0, always recorded (defined whenever seg has a
            # degree>=3 junction).
            local_min_angle = 180.0
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    dot = float(np.dot(dirs[i], dirs[j]))
                    cdot = max(min(dot, 1.0), -1.0)
                    ang = math.degrees(math.acos(cdot))
                    if ang < local_min_angle:
                        local_min_angle = ang
            if seg not in min_branch_angle or local_min_angle < min_branch_angle[seg]:
                min_branch_angle[seg] = local_min_angle

            # hypo_26: filters to norm>0 dirs, requires >=2, records only then.
            valid = [d for d in dirs if np.linalg.norm(d) > 0]
            if len(valid) >= 2:
                local_min_cos = 1.0
                for i in range(len(valid)):
                    for j in range(i + 1, len(valid)):
                        cos_sim = float(np.dot(valid[i], valid[j]))
                        if cos_sim < local_min_cos:
                            local_min_cos = cos_sim
                if seg not in min_junc_cos or local_min_cos < min_junc_cos[seg]:
                    min_junc_cos[seg] = local_min_cos

        # --- degree>=4 junction features (hypo_35 radius ratio, hypo_46) --- #
        if deg >= 4:
            radii35 = [_branch_mean_radius(frag, n, nb, 10.0) for nb in nbrs]
            if len(radii35) >= 2:
                max_r = max(radii35)
                min_r = min(radii35)
                ratio = max_r / min_r if min_r > 1e-5 else max_r / 1e-5
                if seg not in max_junc_radius_ratio or ratio > max_junc_radius_ratio[seg]:
                    max_junc_radius_ratio[seg] = ratio

            dirs46 = [_get_branch_direction(frag, n, nb, 10.0) for nb in nbrs]
            dirs46 = [d for d in dirs46 if np.linalg.norm(d) > 0]
            dots46 = []
            for i in range(len(dirs46)):
                for j in range(i + 1, len(dirs46)):
                    dots46.append(float(np.dot(dirs46[i], dirs46[j])))
            if len(dots46) >= 6:
                dots46.sort()
                min_dot = dots46[0]
                if seg not in min_xantiparallel or min_dot < min_xantiparallel[seg]:
                    min_xantiparallel[seg] = min_dot

        # --- degree-3 T-merge geometry (hypo_47) --- #
        if deg == 3 and len(nbrs) == 3:
            branches = [_branch_direction_and_radius(frag, n, nb, 15.0) for nb in nbrs]
            bdirs = [b[0] for b in branches]
            brads = [b[1] for b in branches]
            dot_products = []
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                dot_products.append((float(np.dot(bdirs[i], bdirs[j])), i, j))
            dot_products.sort(key=lambda x: x[0])
            min_dot, idx1, idx2 = dot_products[0]
            r1 = brads[idx1]
            r2 = brads[idx2]
            max_r = max(r1, r2)
            radius_ratio = min(r1, r2) / max_r if max_r > 0 else 1.0
            if min_dot < -0.85 and radius_ratio > 0.8 and dot_products[1][0] >= -0.85:
                score = -min_dot  # more anti-parallel => stronger T-merge signal
                if seg not in t_merge_score or score > t_merge_score[seg]:
                    t_merge_score[seg] = score

    for seg, val in min_uturn_angle.items():
        acc.set("neg_min_uturn_angle_deg", seg, -val)
    for seg, val in max_deg3_asym.items():
        acc.set("max_degree3_radius_asymmetry", seg, val)
    for seg, val in max_rall.items():
        acc.set("max_rall_ratio", seg, val)
    for seg, val in min_branch_angle.items():
        acc.set("min_junction_branch_angle_deg", seg, val)
    for seg, val in min_junc_cos.items():
        acc.set("min_junction_cosine_similarity", seg, val)
    for seg, val in max_junc_radius_ratio.items():
        acc.set("max_junction_radius_ratio", seg, val)
    for seg, val in min_xantiparallel.items():
        acc.set("min_xcrossing_antiparallel_pair_dot", seg, val)
    for seg, val in t_merge_score.items():
        acc.set("t_merge_geometry_score", seg, val)
    _analysis_stop(analysis_token)
    _phase_stop(phase_token)

    # ================================================================= #
    # X-CROSSING DISJOINT-PAIR (hypo_41): per-component junction dirs.  #
    # ================================================================= #
    _log("[extract] x-crossing disjoint-pair pass ...")
    phase_token = _phase_start("x_crossing")
    analysis_token = _analysis_start(
        "hypo_41", context={"segments": len(adjudicable)})
    for seg_id in (adjudicable if _analysis_enabled("hypo_41") else []):
        seg_best = None
        for comp_id in seg_to_comps.get(seg_id, []):
            nodes = comp_nodes.get(comp_id, [])
            node_set = set(nodes)
            for p in nodes:
                pnbrs = list(frag.neighbors(p))
                if degrees[p] >= 4:
                    dirs = [_branch_direction(frag, p, nb, 15.0) for nb in pnbrs]
                    dirs = [d for d in dirs if np.linalg.norm(d) > 0]
                    if len(dirs) >= 4:
                        best_val, _ = _best_two_disjoint_pairs(dirs)
                        if best_val <= 1.0 and (seg_best is None or best_val < seg_best):
                            seg_best = best_val
                elif degrees[p] >= 3:
                    for q in pnbrs:
                        if q in node_set and degrees[q] >= 3 and q > p:
                            dirs = []
                            for nb in frag.neighbors(p):
                                if nb != q:
                                    dirs.append(_branch_direction(frag, p, nb, 15.0))
                            for nb in frag.neighbors(q):
                                if nb != p:
                                    dirs.append(_branch_direction(frag, q, nb, 15.0))
                            dirs = [d for d in dirs if np.linalg.norm(d) > 0]
                            if len(dirs) >= 4:
                                best_val, _ = _best_two_disjoint_pairs(dirs)
                                if best_val <= 1.0 and (seg_best is None or best_val < seg_best):
                                    seg_best = best_val
        if seg_best is not None:
            acc.set("min_xcrossing_disjoint_pair_dot", seg_id, seg_best)
    _analysis_stop(analysis_token)
    _phase_stop(phase_token)

    # ================================================================= #
    # CHAIN PASS: degree-2 path enumeration shared by hypo_1 and 15.    #
    # ================================================================= #
    _log("[extract] chain / internode pass ...")
    phase_token = _phase_start("chain")
    analysis_token = _analysis_start(
        "shared_chain_1_15", context={"segments": len(adjudicable)})
    for seg_id in (adjudicable
                   if _analysis_enabled("shared_chain_1_15") else []):
        seg_max_wt = None
        cv_list = []
        for comp_id in seg_to_comps.get(seg_id, []):
            nodes = comp_nodes.get(comp_id, [])
            node_set = set(nodes)
            local_deg = {n: sum(1 for w in frag.neighbors(n) if w in node_set)
                         for n in nodes}
            visited_edges = set()
            # enumerate maximal degree-2 chains (internodes): start at any node
            # whose local degree != 2, walk along degree-2 interior.
            endpoints = [n for n in nodes if local_deg[n] != 2]
            starts = endpoints if endpoints else nodes[:1]
            paths = []
            for s in starts:
                for first in frag.neighbors(s):
                    if first not in node_set:
                        continue
                    ekey = tuple(sorted((s, first)))
                    if ekey in visited_edges:
                        continue
                    path = [s, first]
                    visited_edges.add(ekey)
                    prev, cur = s, first
                    steps = 0
                    while local_deg.get(cur, 0) == 2 and steps < 100000:
                        nxts = [k for k in frag.neighbors(cur)
                                if k in node_set and k != prev]
                        if len(nxts) != 1:
                            break
                        prev, cur = cur, nxts[0]
                        path.append(cur)
                        visited_edges.add(tuple(sorted((prev, cur))))
                        steps += 1
                    paths.append(path)
            for path in paths:
                if len(path) < 2:
                    continue
                pts = np.array([node_xyz[n] for n in path])
                seg_d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                cum = np.concatenate([[0.0], np.cumsum(seg_d)])
                n_pts = len(path)
                # hypo_1 sliding-window tortuosity
                i = 0
                while i < n_pts - 1:
                    j = i
                    while j < n_pts and (cum[j] - cum[i]) < 15.0:
                        j += 1
                    if j == n_pts:
                        j = n_pts - 1
                    L = cum[j] - cum[i]
                    if L >= 10.0:
                        E = float(np.linalg.norm(pts[j] - pts[i]))
                        if E > 1e-5:
                            T = L / E
                            if T > 1.0 and (seg_max_wt is None or T > seg_max_wt):
                                seg_max_wt = T
                    i += 1
                # hypo_15 internode radius CV (path length > 10um)
                length = float(cum[-1])
                if length > 10.0:
                    radii = np.array([node_radius[n] for n in path], dtype=float)
                    finite = radii[np.isfinite(radii)]
                    if finite.size > 0:
                        mean_r = float(np.mean(finite))
                        if mean_r > 0:
                            cv = float(np.std(finite) / mean_r)
                            cv_list.append(cv)
        if seg_max_wt is not None:
            acc.set("max_windowed_path_tortuosity", seg_id, seg_max_wt)
        if cv_list:
            acc.set("p95_internode_radius_cv", seg_id, float(np.percentile(cv_list, 95)))
    _analysis_stop(analysis_token)
    _phase_stop(phase_token)

    # ================================================================= #
    # COMPONENT PASS: everything that walks a whole component.          #
    # ================================================================= #
    _log("[extract] component pass ...")
    phase_token = _phase_start("component")

    # Component extraction can take hours on the real cache.  Keep this
    # instrumentation beside the feature code so a durable tee log identifies
    # the exact segment, component, and feature currently being evaluated.
    # Timing calls and prints are observational only: they do not alter feature
    # values, traversal order, or random state.
    component_pass_started = time.perf_counter()
    component_feature_seconds = defaultdict(float)
    component_feature_calls = defaultdict(int)

    def _component_feature_start(name, seg_id, comp_id=None, eligible=True,
                                 n_nodes=None, n_edges=None):
        scope = f"seg_id={seg_id}"
        context = {"segment_id": int(seg_id)}
        if comp_id is not None:
            scope += f" comp_id={comp_id}"
            context["component_id"] = int(comp_id)
        if n_nodes is not None:
            context["nodes"] = int(n_nodes)
        if n_edges is not None:
            context["edges"] = int(n_edges)
        key = "hypo_" + name.split("[hypo_", 1)[1].split("]", 1)[0]
        if not _analysis_enabled(key):
            return None
        _log(f"[extract][component][feature:start] {scope} feature={name}")
        analysis_token = _analysis_start(
            key, eligible=eligible, context=context)
        return time.perf_counter(), analysis_token

    def _component_feature_done(name, started, seg_id, comp_id=None):
        if started is None:
            return
        feature_clock, analysis_token = started
        elapsed = time.perf_counter() - feature_clock
        _analysis_stop(analysis_token)
        component_feature_seconds[name] += elapsed
        component_feature_calls[name] += 1
        scope = f"seg_id={seg_id}"
        if comp_id is not None:
            scope += f" comp_id={comp_id}"
        _log(
            f"[extract][component][feature:done] {scope} feature={name} "
            f"elapsed_s={elapsed:.3f} "
            f"cumulative_s={component_feature_seconds[name]:.3f}"
        )

    component_keys = {
        group["key"] for group in ANALYSIS_TIMING_GROUPS
        if group["phase"] == "component"
    }
    component_pass_enabled = any(
        _analysis_enabled(key) for key in component_keys)
    per_component_keys = component_keys - {
        "hypo_37", "hypo_40", "hypo_11", "hypo_18"
    }
    per_component_pass_enabled = any(
        _analysis_enabled(key) for key in per_component_keys)
    per_component_total = (
        sum(len(seg_to_comps.get(seg_id, [])) for seg_id in adjudicable)
        if per_component_pass_enabled else 0)
    per_component_index = 0

    # weighted adjacency scratch for hypo_37 (adjudicable comps only)
    component_segments = adjudicable if component_pass_enabled else []
    for seg_index, seg_id in enumerate(component_segments, start=1):
        comps = seg_to_comps.get(seg_id, [])
        segment_started = time.perf_counter()
        _log(
            f"[extract][component][segment:start] {seg_index}/{len(adjudicable)} "
            f"seg_id={seg_id} components={len(comps)}"
        )

        # ---- hypo_37 diameter/leaf ratio (needs all comps of the seg) ---- #
        feature_started = _component_feature_start(
            "diameter_to_leaf_ratio[hypo_37]", seg_id,
            eligible=bool(comps))
        total_leaves = 0
        comp_cables = []
        adj_lists = {}
        for comp_id in (comps if _analysis_enabled("hypo_37") else []):
            nodes = comp_nodes.get(comp_id, [])
            node_set = set(nodes)
            adj = defaultdict(list)
            cable = 0.0
            for u in nodes:
                for w in frag.neighbors(u):
                    if w in node_set:
                        d = _dist(frag, u, w)
                        adj[u].append((w, d))
                        if u < w:
                            cable += d
            adj_lists[comp_id] = adj
            comp_cables.append((cable, comp_id))
            local_deg = {u: len(adj[u]) for u in nodes}
            total_leaves += sum(1 for u in nodes if local_deg.get(u, 0) == 1)
        max_diam = 0.0
        comp_cables.sort(key=lambda x: -x[0])
        for cable, comp_id in comp_cables:
            if cable <= max_diam:
                break
            nodes = comp_nodes.get(comp_id, [])
            if len(nodes) < 2:
                continue
            adj = adj_lists[comp_id]
            a, _ = _weighted_farthest(adj, nodes[0])
            _, diam = _weighted_farthest(adj, a)
            if diam > max_diam:
                max_diam = diam
        if comps and total_leaves > 0:
            acc.set("diameter_to_leaf_ratio", seg_id, max_diam / total_leaves)
        _component_feature_done(
            "diameter_to_leaf_ratio[hypo_37]", feature_started, seg_id)

        # ---- per-component reductions ---- #
        for comp_id in (comps if per_component_pass_enabled else []):
            per_component_index += 1
            component_started = time.perf_counter()
            nodes = comp_nodes.get(comp_id, [])
            n_nodes = len(nodes)
            if n_nodes == 0:
                _log(
                    f"[extract][component][component:skip] "
                    f"{per_component_index}/{per_component_total} seg_id={seg_id} "
                    f"comp_id={comp_id} reason=no_nodes"
                )
                continue
            node_set = set(nodes)
            coords = np.array([node_xyz[n] for n in nodes])
            subg = frag.subgraph(nodes)
            n_edges = subg.number_of_edges()
            cable = (_intra_comp_cable_length(frag, nodes)
                     if any(_analysis_enabled(key) for key in
                            ("hypo_22", "hypo_23", "hypo_31")) else 0.0)
            _log(
                f"[extract][component][component:start] "
                f"{per_component_index}/{per_component_total} seg_id={seg_id} "
                f"comp_id={comp_id} nodes={n_nodes} edges={n_edges}"
            )

            # hypo_6 normalized bridge betweenness (comp >= 50 nodes)
            feature_started = _component_feature_start(
                "max_normalized_edge_betweenness_bridge[hypo_6]",
                seg_id, comp_id, eligible=n_nodes >= 50,
                n_nodes=n_nodes, n_edges=n_edges)
            if _analysis_enabled("hypo_6") and n_nodes >= 50:
                try:
                    bridges = list(nx.bridges(subg))
                except Exception:
                    bridges = []
                best = None
                denom = (n_nodes / 2.0) ** 2
                for (u, v) in bridges:
                    subg2 = nx.Graph(subg)
                    subg2.remove_edge(u, v)
                    side = None
                    for cc in nx.connected_components(subg2):
                        if u in cc:
                            side = cc
                            break
                    n1 = len(side) if side is not None else 1
                    n2 = n_nodes - n1
                    val = (n1 * n2) / denom if denom > 0 else 0.0
                    if best is None or val > best:
                        best = val
                if best is not None:
                    prev = max_bridge_ebc.get(seg_id, 0.0)
                    max_bridge_ebc[seg_id] = max(prev, best)
                elif seg_id not in max_bridge_ebc:
                    max_bridge_ebc[seg_id] = 0.0
            _component_feature_done(
                "max_normalized_edge_betweenness_bridge[hypo_6]",
                feature_started, seg_id, comp_id)

            # hypo_2 euclidean/geodesic wraparound (KDTree pairs)
            feature_started = _component_feature_start(
                "max_euclidean_geodesic_wraparound_ratio[hypo_2]",
                seg_id, comp_id, eligible=n_nodes >= 2,
                n_nodes=n_nodes, n_edges=n_edges)
            if (_analysis_enabled("hypo_2") and
                    comp_id in comp_to_seg and n_nodes >= 2):
                subw = nx.Graph(subg)
                for (u, v) in subw.edges():
                    subw[u][v]["weight"] = _dist(frag, u, v)
                ktree = KDTree(coords)
                pairs = list(ktree.query_pairs(10.0))
                rng = np.random.RandomState(42 + comp_id)
                if len(pairs) > 5000:
                    sel = rng.choice(len(pairs), size=5000, replace=False)
                    pairs = [pairs[k] for k in sel]
                us = defaultdict(list)
                for (a_i, b_i) in pairs:
                    us[a_i].append(b_i)
                u_keys = list(us.keys())
                if len(u_keys) > 100:
                    sel = rng.choice(len(u_keys), size=100, replace=False)
                    u_keys = [u_keys[k] for k in sel]
                best_ratio = None
                for a_i in u_keys:
                    src = nodes[a_i]
                    lengths = nx.single_source_dijkstra_path_length(
                        subw, src, cutoff=150.0, weight="weight")
                    for b_i in us[a_i]:
                        tgt = nodes[b_i]
                        euc = float(np.linalg.norm(coords[a_i] - coords[b_i]))
                        if euc <= 0.01:
                            continue
                        if tgt in lengths:
                            geo = lengths[tgt]
                        else:
                            try:
                                geo = nx.dijkstra_path_length(
                                    subw, src, tgt, weight="weight")
                            except Exception:
                                continue
                        if geo > 150.0:
                            ratio = geo / euc
                            if best_ratio is None or ratio > best_ratio:
                                best_ratio = ratio
                if best_ratio is not None:
                    prev = max_wraparound.get(seg_id, 0.0)
                    max_wraparound[seg_id] = max(prev, best_ratio)
                elif seg_id not in max_wraparound:
                    max_wraparound[seg_id] = 0.0
            _component_feature_done(
                "max_euclidean_geodesic_wraparound_ratio[hypo_2]",
                feature_started, seg_id, comp_id)

            # hypo_13 junction tortuosity variance (deg>=3, all nbrs)
            feature_started = _component_feature_start(
                "max_junction_tortuosity_variance[hypo_13]",
                seg_id, comp_id, n_nodes=n_nodes, n_edges=n_edges)
            local_deg = {u: subg.degree(u) for u in nodes}
            for u in (nodes if _analysis_enabled("hypo_13") else []):
                if local_deg[u] >= 3:
                    torts = [_compute_tortuosity(frag, u, nb, 20.0)
                             for nb in frag.neighbors(u) if nb in node_set]
                    if len(torts) >= 3:
                        var = float(np.var(torts))
                        prev = max_junc_tort_var.get(seg_id)
                        if prev is None or var > prev:
                            max_junc_tort_var[seg_id] = var
            _component_feature_done(
                "max_junction_tortuosity_variance[hypo_13]",
                feature_started, seg_id, comp_id)

            # hypo_19 max radius step along unweighted pseudo-diameter
            feature_started = _component_feature_start(
                "max_radius_step_along_pseudo_diameter[hypo_19]",
                seg_id, comp_id, n_nodes=n_nodes, n_edges=n_edges)
            path = (_get_diameter_path_unweighted(frag, nodes)
                    if _analysis_enabled("hypo_19") else [])
            if len(path) >= 2:
                pradii = np.array([node_radius[n] for n in path], dtype=float)
                dr = np.abs(np.diff(pradii))
                dr = dr[np.isfinite(dr)]
                if dr.size > 0:
                    step = float(np.max(dr))
                    prev = max_radius_step.get(seg_id, 0.0)
                    max_radius_step[seg_id] = max(prev, step)
                elif seg_id not in max_radius_step:
                    max_radius_step[seg_id] = 0.0
            elif (_analysis_enabled("hypo_19") and
                  seg_id not in max_radius_step):
                max_radius_step[seg_id] = 0.0
            _component_feature_done(
                "max_radius_step_along_pseudo_diameter[hypo_19]",
                feature_started, seg_id, comp_id)

            # hypo_20 pseudo-diameter tortuosity (weighted double-Dijkstra)
            feature_started = _component_feature_start(
                "pseudo_diameter_tortuosity[hypo_20]", seg_id, comp_id,
                n_nodes=n_nodes, n_edges=n_edges)
            if _analysis_enabled("hypo_20"):
                graph_dist, a_node, b_node = _double_dijkstra_length(frag, nodes)
            else:
                graph_dist, a_node, b_node = 0.0, None, None
            if graph_dist > 0:
                euc = float(np.linalg.norm(node_xyz[a_node] - node_xyz[b_node]))
                tort = graph_dist / max(euc, 1e-5)
                prev = max_pdiam_tort.get(seg_id, 1.0)
                max_pdiam_tort[seg_id] = max(prev, tort)
            _component_feature_done(
                "pseudo_diameter_tortuosity[hypo_20]",
                feature_started, seg_id, comp_id)

            # hypo_21 max edge betweenness (comp >= 3 nodes)
            feature_started = _component_feature_start(
                "max_edge_betweenness_centrality[hypo_21]",
                seg_id, comp_id, eligible=n_nodes >= 3,
                n_nodes=n_nodes, n_edges=n_edges)
            if _analysis_enabled("hypo_21") and n_nodes >= 3:
                ebc_val = _get_max_ebc_istree(nx.Graph(subg), nx)
                prev = max_ebc.get(seg_id, 0.0)
                max_ebc[seg_id] = max(prev, ebc_val)
            _component_feature_done(
                "max_edge_betweenness_centrality[hypo_21]",
                feature_started, seg_id, comp_id)

            # hypo_22 min component spatial density (bbox)
            feature_started = _component_feature_start(
                "neg_min_component_spatial_density[hypo_22]",
                seg_id, comp_id, n_nodes=n_nodes, n_edges=n_edges)
            dims = (np.maximum(np.max(coords, axis=0) - np.min(coords, axis=0), 1e-3)
                    if _analysis_enabled("hypo_22") else None)
            vol = float(np.prod(dims)) if dims is not None else 0.0
            if _analysis_enabled("hypo_22") and cable > 0 and vol > 0:
                density = cable / vol
                prev = min_comp_density.get(seg_id)
                if prev is None or density < prev:
                    min_comp_density[seg_id] = density
            _component_feature_done(
                "neg_min_component_spatial_density[hypo_22]",
                feature_started, seg_id, comp_id)

            # hypo_23 max high-degree-node density (deg>=4 / cable length)
            feature_started = _component_feature_start(
                "max_high_degree_node_density[hypo_23]",
                seg_id, comp_id, n_nodes=n_nodes, n_edges=n_edges)
            hdn = (sum(1 for u in nodes if local_deg[u] >= 4)
                   if _analysis_enabled("hypo_23") else 0)
            if _analysis_enabled("hypo_23") and cable > 0:
                density = hdn / cable
                prev = max_hdn_density.get(seg_id, 0.0)
                max_hdn_density[seg_id] = max(prev, density)
            _component_feature_done(
                "max_high_degree_node_density[hypo_23]",
                feature_started, seg_id, comp_id)

            # hypo_27 branching frequency mismatch (comp >= 5 nodes)
            feature_started = _component_feature_start(
                "max_branching_frequency_mismatch[hypo_27]",
                seg_id, comp_id, eligible=n_nodes >= 5,
                n_nodes=n_nodes, n_edges=n_edges)
            if _analysis_enabled("hypo_27") and n_nodes >= 5:
                mm = _compute_mismatch_score(frag, nodes, nx)
                prev = max_branch_mismatch.get(seg_id, 0.0)
                max_branch_mismatch[seg_id] = max(prev, mm)
            _component_feature_done(
                "max_branching_frequency_mismatch[hypo_27]",
                feature_started, seg_id, comp_id)

            # hypo_29 curvature spike density along longest path (comp >= 3)
            feature_started = _component_feature_start(
                "curvature_spike_density[hypo_29]", seg_id, comp_id,
                eligible=n_nodes >= 3, n_nodes=n_nodes, n_edges=n_edges)
            if _analysis_enabled("hypo_29") and n_nodes >= 3:
                lpath = _get_longest_path(frag, nodes)
                if len(lpath) >= 3:
                    pts = np.array([node_xyz[n] for n in lpath])
                    dvecs = np.diff(pts, axis=0)
                    dlens = np.linalg.norm(dvecs, axis=1)
                    path_um = float(np.sum(dlens))
                    spikes = 0
                    for k in range(len(dvecs) - 1):
                        if dlens[k] > 0 and dlens[k + 1] > 0:
                            dot = float(np.dot(dvecs[k], dvecs[k + 1]))
                            if dot < 0:
                                spikes += 1
                    if path_um > 0:
                        rate = (spikes / path_um) * 100.0
                        prev_um = curvature_spike.get(seg_id, (0.0, -1.0))
                        if path_um > prev_um[1]:
                            curvature_spike[seg_id] = (rate, path_um)
            _component_feature_done(
                "curvature_spike_density[hypo_29]",
                feature_started, seg_id, comp_id)

            # hypo_30 radius assortativity (comp > 10 edges)
            feature_started = _component_feature_start(
                "neg_min_radius_assortativity[hypo_30]",
                seg_id, comp_id, eligible=n_edges > 10,
                n_nodes=n_nodes, n_edges=n_edges)
            if _analysis_enabled("hypo_30") and n_edges > 10:
                ra = []
                rb = []
                for (u, v) in subg.edges():
                    ru = node_radius[u]
                    rv = node_radius[v]
                    ra.append(ru)
                    rb.append(rv)
                    ra.append(rv)
                    rb.append(ru)
                ra = np.array(ra, dtype=float)
                rb = np.array(rb, dtype=float)
                mask = np.isfinite(ra) & np.isfinite(rb)
                ra = ra[mask]
                rb = rb[mask]
                if ra.size > 0:
                    va = max(float(np.var(ra)), 0.0)
                    vb = max(float(np.var(rb)), 0.0)
                    if va <= 0 or vb <= 0:
                        assort = 1.0
                    else:
                        cov = float(np.mean((ra - ra.mean()) * (rb - rb.mean())))
                        denom = math.sqrt(va * vb)
                        assort = cov / denom if denom > 0 else 1.0
                    if not np.isfinite(assort):
                        assort = 1.0
                    prev = min_radius_assort.get(seg_id)
                    if prev is None or assort < prev:
                        min_radius_assort[seg_id] = assort
            _component_feature_done(
                "neg_min_radius_assortativity[hypo_30]",
                feature_started, seg_id, comp_id)

            # hypo_31 min log convexhull cable density (comp > 20 nodes)
            feature_started = _component_feature_start(
                "neg_min_log_convexhull_cable_density[hypo_31]",
                seg_id, comp_id,
                eligible=n_nodes > 20 and n_edges > 0 and cable > 0,
                n_nodes=n_nodes, n_edges=n_edges)
            if (_analysis_enabled("hypo_31") and
                    n_nodes > 20 and n_edges > 0 and cable > 0):
                try:
                    hull = ConvexHull(coords)
                    chvol = float(hull.volume)
                except Exception:
                    dims_ch = np.maximum(
                        np.max(coords, axis=0) - np.min(coords, axis=0), 0.1)
                    chvol = float(np.prod(dims_ch))
                if chvol <= 1e-5:
                    chvol = 1e-3
                density = cable / chvol
                prev = min_log_ch_density.get(seg_id)
                if prev is None or density < prev:
                    min_log_ch_density[seg_id] = density
            _component_feature_done(
                "neg_min_log_convexhull_cable_density[hypo_31]",
                feature_started, seg_id, comp_id)

            # hypo_32 max tapering reversal rate (BFS from max-radius root).
            # Root chosen exactly as the source: plain max over node_radius (no
            # NaN handling), matching Python's order-dependent comparison.
            feature_started = _component_feature_start(
                "max_tapering_reversal_rate[hypo_32]",
                seg_id, comp_id, eligible=n_nodes >= 2 and n_edges > 0,
                n_nodes=n_nodes, n_edges=n_edges)
            if (_analysis_enabled("hypo_32") and
                    n_nodes >= 2 and n_edges > 0):
                root = max(nodes, key=lambda nn: node_radius[nn])
                reversals = 0
                edges_traversed = 0
                for (u, v) in nx.bfs_edges(subg, root):
                    edges_traversed += 1
                    if (node_radius[v] - node_radius[u]) > 0.5:
                        reversals += 1
                if edges_traversed > 0:
                    rate = reversals / edges_traversed
                    prev = max_taper_rev.get(seg_id, 0.0)
                    max_taper_rev[seg_id] = max(prev, rate)
            _component_feature_done(
                "max_tapering_reversal_rate[hypo_32]",
                feature_started, seg_id, comp_id)

            # hypo_34 min normalized leaf NN distance
            feature_started = _component_feature_start(
                "neg_min_normalized_leaf_nn_distance[hypo_34]",
                seg_id, comp_id, n_nodes=n_nodes, n_edges=n_edges)
            leaves = ([u for u in nodes if local_deg[u] == 1]
                      if _analysis_enabled("hypo_34") else [])
            if len(leaves) > 3:
                bb_diag = _bbox_diag(coords)
                if bb_diag != 0:
                    leaf_xyz = np.array([node_xyz[u] for u in leaves])
                    ltree = KDTree(leaf_xyz)
                    k = min(4, len(leaves))
                    dists, _ = ltree.query(leaf_xyz, k=k)
                    if dists.ndim == 1:
                        dists = dists[:, None]
                    neighbor_dists = dists[:, 1:]
                    norm = float(np.mean(neighbor_dists)) / bb_diag
                    prev = min_leaf_nn.get(seg_id)
                    if prev is None or norm < prev:
                        min_leaf_nn[seg_id] = norm
            _component_feature_done(
                "neg_min_normalized_leaf_nn_distance[hypo_34]",
                feature_started, seg_id, comp_id)

            # hypo_36 thickness-distance spearman (comp > 20 nodes)
            feature_started = _component_feature_start(
                "max_thickness_distance_spearman[hypo_36]",
                seg_id, comp_id, eligible=n_nodes > 20,
                n_nodes=n_nodes, n_edges=n_edges)
            if _analysis_enabled("hypo_36") and n_nodes > 20:
                comp_radii = np.array([node_radius[n] for n in nodes], dtype=float)
                comp_radii = np.nan_to_num(comp_radii, nan=0.0)
                if float(np.std(comp_radii)) >= 1e-6:
                    root_idx = int(np.argmax(comp_radii))
                    root_xyz = coords[root_idx]
                    dists = np.linalg.norm(coords - root_xyz, axis=1)
                    if float(np.std(dists)) >= 1e-6:
                        r_use = comp_radii
                        d_use = dists
                        if n_nodes > 3000:
                            sel = np.random.choice(n_nodes, size=3000, replace=False)
                            r_use = comp_radii[sel]
                            d_use = dists[sel]
                        corr, _ = spearmanr(r_use, d_use)
                        if corr is not None and np.isfinite(corr):
                            prev = max_thick_spearman.get(seg_id)
                            if prev is None or corr > prev:
                                max_thick_spearman[seg_id] = float(corr)
            _component_feature_done(
                "max_thickness_distance_spearman[hypo_36]",
                feature_started, seg_id, comp_id)

            # hypo_39 bridge radius variance ratio (comp >= 100 nodes, spread>=50)
            feature_started = _component_feature_start(
                "max_bridge_radius_variance_ratio[hypo_39]",
                seg_id, comp_id, eligible=n_nodes >= 100,
                n_nodes=n_nodes, n_edges=n_edges)
            if _analysis_enabled("hypo_39") and n_nodes >= 100:
                spread = float(np.max(np.max(coords, axis=0) - np.min(coords, axis=0)))
                if spread >= 50.0:
                    ratio = _get_best_bridge_var_ratio(frag, nodes, nx)
                    if ratio is not None:
                        prev = max_bridge_var_ratio.get(seg_id)
                        if prev is None or ratio > prev:
                            max_bridge_var_ratio[seg_id] = ratio
            _component_feature_done(
                "max_bridge_radius_variance_ratio[hypo_39]",
                feature_started, seg_id, comp_id)

            # hypo_45 intra-component branch tortuosity variance (comp >= 10)
            feature_started = _component_feature_start(
                "max_intra_component_tortuosity_variance[hypo_45]",
                seg_id, comp_id, eligible=n_nodes >= 10,
                n_nodes=n_nodes, n_edges=n_edges)
            if _analysis_enabled("hypo_45") and n_nodes >= 10:
                branches = _extract_branches(frag, nodes)
                torts = []
                for br in branches:
                    if len(br) < 2:
                        continue
                    pts = np.array([node_xyz[n] for n in br])
                    c_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                    if c_len >= 10.0:
                        euc = float(np.linalg.norm(pts[-1] - pts[0]))
                        t = c_len / euc if euc > 0.1 else c_len / 0.1
                        t = min(t, 20.0)
                        torts.append(t)
                if len(torts) >= 2:
                    var = float(np.var(torts))
                    prev = max_intra_tort_var.get(seg_id, 0.0)
                    max_intra_tort_var[seg_id] = max(prev, var)
            _component_feature_done(
                "max_intra_component_tortuosity_variance[hypo_45]",
                feature_started, seg_id, comp_id)
            _log(
                f"[extract][component][component:done] "
                f"{per_component_index}/{per_component_total} seg_id={seg_id} "
                f"comp_id={comp_id} nodes={n_nodes} edges={n_edges} "
                f"elapsed_s={time.perf_counter() - component_started:.3f}"
            )

        # ---- hypo_40 contracted edge betweenness (largest comp >= 10 nodes) ---- #
        feature_started = _component_feature_start(
            "max_contracted_normalized_edge_betweenness[hypo_40]", seg_id,
            eligible=bool(comps))
        if _analysis_enabled("hypo_40") and comps:
            largest = max(comps, key=lambda c: len(comp_nodes.get(c, [])))
            lnodes = comp_nodes.get(largest, [])
            if len(lnodes) >= 10:
                subg = nx.Graph(frag.subgraph(lnodes))
                val = _get_max_ebc_contracted(subg, nx)
                if val is not None:
                    acc.set("max_contracted_normalized_edge_betweenness", seg_id, val)
        _component_feature_done(
            "max_contracted_normalized_edge_betweenness[hypo_40]",
            feature_started, seg_id)

        # ---- hypo_11 aspect ratio + spatial density (largest comp >= 20) ---- #
        feature_started = _component_feature_start(
            "component_aspect_ratio+component_spatial_density[hypo_11]",
            seg_id, eligible=bool(comps))
        if _analysis_enabled("hypo_11") and comps:
            eligible = [c for c in comps if len(comp_nodes.get(c, [])) >= 20]
            if eligible:
                main_comp = max(eligible, key=lambda c: len(comp_nodes.get(c, [])))
                mnodes = comp_nodes.get(main_comp, [])
                mcoords = np.array([node_xyz[n] for n in mnodes])
                cov = np.cov(mcoords.T)
                eig = np.linalg.eigvalsh(cov)
                L1 = max(float(eig[2]), 1e-6)
                L3 = max(float(eig[0]), 0.0)
                aspect = L3 / L1
                acc.set("component_aspect_ratio", seg_id, aspect)
                ranges = np.max(mcoords, axis=0) - np.min(mcoords, axis=0)
                vol = float(np.prod(np.maximum(ranges, 1e-3)))
                length = len(mnodes) * 5.0
                density = length / vol if vol > 0 else 0.0
                acc.set("component_spatial_density", seg_id, density)
        _component_feature_done(
            "component_aspect_ratio+component_spatial_density[hypo_11]",
            feature_started, seg_id)

        # ---- hypo_18 (FIXED) leaf-cluster silhouette (measured-only) ---- #
        feature_started = _component_feature_start(
            "max_leaf_cluster_silhouette[hypo_18]", seg_id,
            eligible=bool(comps))
        if _analysis_enabled("hypo_18") and comps:
            seg_measured = False
            seg_max_sil = -1.0
            for comp_id in comps:
                nodes = comp_nodes.get(comp_id, [])
                node_set = set(nodes)
                local_deg = {u: sum(1 for w in frag.neighbors(u) if w in node_set)
                             for u in nodes}
                leaves = [u for u in nodes if local_deg[u] == 1]
                if len(leaves) >= 3:
                    leaf_xyz = np.array([node_xyz[u] for u in leaves])
                    if len(np.unique(leaf_xyz, axis=0)) >= 2:
                        km = KMeans(n_clusters=2, n_init=1, max_iter=100,
                                    random_state=42)
                        labels = km.fit_predict(leaf_xyz)
                        if len(np.unique(labels)) == 2:
                            score = float(silhouette_score(leaf_xyz, labels))
                            seg_measured = True
                            if score > seg_max_sil:
                                seg_max_sil = score
            if seg_measured:
                acc.set("max_leaf_cluster_silhouette", seg_id, seg_max_sil)
        _component_feature_done(
            "max_leaf_cluster_silhouette[hypo_18]", feature_started, seg_id)
        _log(
            f"[extract][component][segment:done] "
            f"{seg_index}/{len(adjudicable)} seg_id={seg_id} "
            f"components={len(comps)} "
            f"elapsed_s={time.perf_counter() - segment_started:.3f}"
        )

    component_elapsed = time.perf_counter() - component_pass_started
    if component_pass_enabled:
        _log(
            f"[extract][component][pass:done] "
            f"segments_processed={len(component_segments)} "
            f"per_component_reductions="
            f"{per_component_index}/{per_component_total} "
            f"elapsed_s={component_elapsed:.3f}"
        )
    else:
        _log(
            "[extract][component][pass:skip] "
            "reason=no_enabled_analysis_groups"
        )
    for feature_name in sorted(component_feature_seconds):
        _log(
            f"[extract][component][timing] feature={feature_name} "
            f"calls={component_feature_calls[feature_name]} "
            f"total_s={component_feature_seconds[feature_name]:.3f}"
        )
    # flush component-pass reductions
    for seg, val in max_bridge_ebc.items():
        acc.set("max_normalized_edge_betweenness_bridge", seg, val)
    for seg, val in max_wraparound.items():
        acc.set("max_euclidean_geodesic_wraparound_ratio", seg, val)
    for seg, val in max_junc_tort_var.items():
        acc.set("max_junction_tortuosity_variance", seg, val)
    for seg, val in max_radius_step.items():
        acc.set("max_radius_step_along_pseudo_diameter", seg, val)
    for seg, val in max_pdiam_tort.items():
        acc.set("pseudo_diameter_tortuosity", seg, val)
    for seg, val in max_ebc.items():
        acc.set("max_edge_betweenness_centrality", seg, val)
    for seg, val in min_comp_density.items():
        acc.set("neg_min_component_spatial_density", seg, -val)
    for seg, val in max_hdn_density.items():
        acc.set("max_high_degree_node_density", seg, val)
    for seg, val in max_branch_mismatch.items():
        acc.set("max_branching_frequency_mismatch", seg, val)
    for seg, (rate, path_um) in curvature_spike.items():
        if path_um > 0:
            acc.set("curvature_spike_density", seg, rate)
    for seg, val in min_radius_assort.items():
        acc.set("neg_min_radius_assortativity", seg, -val)
    for seg, val in min_log_ch_density.items():
        acc.set("neg_min_log_convexhull_cable_density", seg,
                -math.log(val + 1e-12))
    for seg, val in max_taper_rev.items():
        acc.set("max_tapering_reversal_rate", seg, val)
    for seg, val in min_leaf_nn.items():
        acc.set("neg_min_normalized_leaf_nn_distance", seg, -val)
    for seg, val in max_thick_spearman.items():
        acc.set("max_thickness_distance_spearman", seg, val)
    for seg, val in max_bridge_var_ratio.items():
        acc.set("max_bridge_radius_variance_ratio", seg, val)
    for seg, val in max_intra_tort_var.items():
        acc.set("max_intra_component_tortuosity_variance", seg, val)
    _phase_stop(phase_token)

    # ================================================================= #
    # SEGMENT PASS: features aggregating ALL of a segment's nodes.      #
    # Covers hypo_9, 14, 44, 50.                                        #
    # ================================================================= #
    _log("[extract] segment pass ...")
    phase_token = _phase_start("segment")
    segment_keys = {
        group["key"] for group in ANALYSIS_TIMING_GROUPS
        if group["phase"] == "segment"
    }
    segment_pass_enabled = any(
        _analysis_enabled(key) for key in segment_keys)
    for seg_id in (adjudicable if segment_pass_enabled else []):
        nodes = seg_to_nodes.get(seg_id, [])
        n_nodes = len(nodes)
        if n_nodes == 0:
            for timing_key in ("hypo_9", "hypo_14", "hypo_44", "hypo_50"):
                if _analysis_enabled(timing_key):
                    _analysis_start(
                        timing_key, eligible=False,
                        context={"segment_id": int(seg_id), "nodes": 0})
            continue
        coords = np.array([node_xyz[n] for n in nodes])
        timing_context = {"segment_id": int(seg_id), "nodes": int(n_nodes)}

        # hypo_9 box-counting fractal dimension (>=50 nodes, spread>=50)
        analysis_token = _analysis_start(
            "hypo_9", eligible=n_nodes >= 50, context=timing_context)
        if _analysis_enabled("hypo_9") and n_nodes >= 50:
            shifted = coords - np.min(coords, axis=0)
            max_spread = float(np.max(np.max(shifted, axis=0)))
            if max_spread >= 50.0:
                box_sizes = np.logspace(math.log10(2.0), math.log10(50.0), 10)
                logs_inv = []
                log_counts = []
                for s in box_sizes:
                    keys = np.floor(shifted / s).astype(np.int64)
                    n_boxes = len(np.unique(keys, axis=0))
                    if n_boxes > 0:
                        logs_inv.append(math.log(1.0 / s))
                        log_counts.append(math.log(n_boxes))
                if len(logs_inv) >= 2:
                    slope, _, _, _, _ = linregress(logs_inv, log_counts)
                    acc.set("box_counting_fractal_dimension", seg_id, float(slope))
        _analysis_stop(analysis_token)

        # hypo_14 GMM bimodality BIC gain over per-node distances-from-centroid
        analysis_token = _analysis_start(
            "hypo_14", eligible=n_nodes >= 50, context=timing_context)
        if _analysis_enabled("hypo_14") and n_nodes >= 50:
            centroid = np.mean(coords, axis=0)
            dists = np.linalg.norm(coords - centroid, axis=1).reshape(-1, 1)
            if n_nodes < 20:
                gain = 0.0
            else:
                try:
                    g1 = GaussianMixture(n_components=1, random_state=42).fit(dists)
                    g2 = GaussianMixture(n_components=2, random_state=42).fit(dists)
                    gain = float(g1.bic(dists) - g2.bic(dists))
                except Exception:
                    gain = 0.0
            acc.set("gmm_bimodality_bic_gain", seg_id, gain)
        _analysis_stop(analysis_token)

        # hypo_44 spatial bimodality silhouette (>1000 nodes)
        analysis_token = _analysis_start(
            "hypo_44", eligible=n_nodes > 1000, context=timing_context)
        if _analysis_enabled("hypo_44") and n_nodes > 1000:
            np.random.seed(42)
            pts = coords
            if n_nodes > 1000:
                sel = np.random.choice(n_nodes, size=1000, replace=False)
                pts = coords[sel]
            if len(np.unique(pts, axis=0)) < 2:
                sil = 0.0
            else:
                km = KMeans(n_clusters=2, n_init=10, random_state=42)
                labels = km.fit_predict(pts)
                if len(np.unique(labels)) < 2:
                    sil = 0.0
                else:
                    sil = float(silhouette_score(pts, labels))
            acc.set("spatial_bimodality_silhouette", seg_id, sil)
        _analysis_stop(analysis_token)

        # hypo_50 log bbox cable density (length>=5*mcl, >=10 nodes)
        analysis_token = _analysis_start(
            "hypo_50", context=timing_context)
        seg_cable = (_intra_comp_cable_length(frag, nodes)
                     if _analysis_enabled("hypo_50") else 0.0)
        if (_analysis_enabled("hypo_50") and
                seg_cable >= 5 * min_cable_length and n_nodes >= 10):
            dims = np.maximum(np.max(coords, axis=0) - np.min(coords, axis=0), 1.0)
            vol = float(np.prod(dims))
            if vol > 0:
                acc.set("log_bbox_cable_density", seg_id,
                        float(math.log10(seg_cable / vol)))
        _analysis_stop(analysis_token)

    _phase_stop(phase_token)
    _log("[extract] done: %d adjudicable segments" % len(adjudicable))
    return adjudicable, is_merge, acc


# --------------------------------------------------------------------------- #
# SAFE MODEL FACTORY.  Every family is built through an explicit, named branch. #
# No eval, no dynamic import of configured paths.  JSON controls only which     #
# allowlisted family is included and which allowlisted grid values are tried.   #
# --------------------------------------------------------------------------- #
def _package_available(pkg):
    if pkg is None:
        return True
    import importlib

    try:
        importlib.import_module(pkg)
        return True
    except Exception:
        return False


def _logistic_regularization_kwargs(estimator_class, penalty, l1_ratio=None):
    """Keep one regularization meaning across old and new sklearn APIs."""
    import inspect

    penalty_param = inspect.signature(estimator_class).parameters.get("penalty")
    uses_l1_ratio_api = (
        penalty_param is None or penalty_param.default == "deprecated")
    if uses_l1_ratio_api:
        return {"l1_ratio": 0.0 if penalty == "l2" else float(l1_ratio)}
    options = {"penalty": penalty}
    if penalty == "elasticnet":
        options["l1_ratio"] = float(l1_ratio)
    return options


def build_estimator(family, params, native_nan_features_present=True):
    """Return a fresh sklearn-compatible pipeline/estimator for `family` with
    `params`.  `params` keys are already validated against the policy."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, SplineTransformer
    from sklearn.linear_model import LogisticRegression

    seed = RANDOM_SEED

    if family == "logistic_l2":
        return Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=float(params["C"]), solver="lbfgs", class_weight="balanced",
                max_iter=5000, random_state=seed,
                **_logistic_regularization_kwargs(LogisticRegression, "l2"))),
        ])

    if family == "logistic_elasticnet":
        return Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=float(params["C"]), solver="saga", class_weight="balanced",
                max_iter=10000, random_state=seed,
                **_logistic_regularization_kwargs(
                    LogisticRegression, "elasticnet", params["l1_ratio"]))),
        ])

    if family == "spline_logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("spline", SplineTransformer(
                n_knots=int(params["n_knots"]), degree=int(params["degree"]),
                include_bias=False)),
            ("scale", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(
                C=float(params["C"]), solver="lbfgs", class_weight="balanced",
                max_iter=5000, random_state=seed,
                **_logistic_regularization_kwargs(LogisticRegression, "l2"))),
        ])

    if family == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_depth=(None if params.get("max_depth") is None
                       else int(params["max_depth"])),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            class_weight="balanced", random_state=seed)

    if family == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("clf", ExtraTreesClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=(None if params.get("max_depth") is None
                           else int(params["max_depth"])),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=params.get("max_features"),
                class_weight="balanced", n_jobs=1, random_state=seed)),
        ])

    if family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("clf", RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=(None if params.get("max_depth") is None
                           else int(params["max_depth"])),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=params.get("max_features"),
                class_weight="balanced", n_jobs=1, random_state=seed)),
        ])

    if family == "explainable_boosting":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(
            max_bins=int(params["max_bins"]),
            interactions=int(params["interactions"]),
            learning_rate=float(params["learning_rate"]),
            max_rounds=int(params["max_rounds"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed)

    if family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            tree_method="hist", eval_metric="aucpr",
            random_state=seed, n_jobs=1)

    raise ValueError("unhandled family in build_estimator: %r" % (family,))


def iter_param_grid(grid):
    """Yield param dicts (Cartesian product) in a deterministic order."""
    import itertools

    keys = list(grid.keys())
    if not keys:
        yield {}
        return
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


# --------------------------------------------------------------------------- #
# NESTED MODEL SELECTION.                                                       #
# average_precision is the PRIMARY selection score.  one_standard_error rule    #
# in the policy simplicity order.  Every preprocessing / hyperparameter choice  #
# is fitted from inner-training data only.                                      #
# --------------------------------------------------------------------------- #
def _score_matrix(estimator, X):
    """Return positive-class scores (class-weighted, NOT calibrated probs)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def inner_cv_family(family, grid, X, y, inner_folds, seed):
    """3-fold inner CV over a family's grid.  Returns list of dicts:
    {params, mean_ap, std_ap, mean_se}."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score

    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    results = []
    for params in iter_param_grid(grid):
        fold_ap = []
        ok = True
        for tr, va in skf.split(X, y):
            try:
                est = build_estimator(family, params)
                est.fit(X[tr], y[tr])
                sc = _score_matrix(est, X[va])
                fold_ap.append(average_precision_score(y[va], sc))
            except Exception:
                ok = False
                break
        if not ok or not fold_ap:
            continue
        fold_ap = np.asarray(fold_ap, dtype=float)
        results.append({
            "params": params,
            "mean_ap": float(np.mean(fold_ap)),
            "std_ap": float(np.std(fold_ap, ddof=1)) if len(fold_ap) > 1 else 0.0,
            "n": len(fold_ap),
        })
    return results


def best_params_1se(results):
    """One-standard-error rule WITHIN a family: among configs within 1 SE of the
    best mean AP, pick the one with the highest mean AP (families themselves are
    ordered by simplicity later)."""
    if not results:
        return None
    best = max(results, key=lambda r: r["mean_ap"])
    se = best["std_ap"] / np.sqrt(max(best["n"], 1))
    threshold = best["mean_ap"] - se
    eligible = [r for r in results if r["mean_ap"] >= threshold]
    return max(eligible, key=lambda r: r["mean_ap"]) if eligible else best


def select_family_1se(family_scores, simplicity_order):
    """Across families: one-standard-error rule using each family's best inner
    mean AP and its SE; among families whose best mean AP is within 1 SE of the
    global best, choose the SIMPLEST in policy order."""
    if not family_scores:
        return None
    glob = max(family_scores.items(), key=lambda kv: kv[1]["mean_ap"])
    gname, ginfo = glob
    se = ginfo["std_ap"] / np.sqrt(max(ginfo["n"], 1))
    threshold = ginfo["mean_ap"] - se
    eligible = [f for f, info in family_scores.items()
                if info["mean_ap"] >= threshold]
    order = {name: i for i, name in enumerate(simplicity_order)}
    eligible.sort(key=lambda f: order.get(f, 10 ** 6))
    return eligible[0] if eligible else gname


def run_nested_selection(X, y, configured, policy, log):
    """Full nested CV.

    Returns dict with per-family OOF score arrays, selector OOF array, per-fold
    inner choices, selection frequencies, and per-family outer metrics.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score, roc_auc_score

    sel = policy["selection"]
    outer_folds = sel["outer_folds"]
    inner_folds = sel["inner_folds"]
    seed = sel["random_seed"]
    simplicity = sel["simplicity_order"]

    n = len(y)
    families = list(configured.keys())
    oof = {f: np.full(n, np.nan) for f in families}
    oof_selector = np.full(n, np.nan)
    per_fold = []
    selection_freq = defaultdict(int)

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True,
                            random_state=seed)
    for k, (tr, te) in enumerate(outer.split(X, y)):
        log("outer fold %d/%d (train=%d test=%d)"
            % (k + 1, outer_folds, len(tr), len(te)))
        fold_record = {"fold": k, "family_choice": {}, "selected_family": None}
        family_best = {}
        for f in families:
            grid = configured[f]["grid"]
            results = inner_cv_family(f, grid, X[tr], y[tr], inner_folds, seed)
            choice = best_params_1se(results)
            if choice is None:
                fold_record["family_choice"][f] = None
                continue
            fold_record["family_choice"][f] = {
                "params": choice["params"],
                "inner_mean_ap": choice["mean_ap"],
                "inner_std_ap": choice["std_ap"],
            }
            family_best[f] = {
                "mean_ap": choice["mean_ap"],
                "std_ap": choice["std_ap"],
                "n": choice["n"],
                "params": choice["params"],
            }
            # fit on all outer-train rows, score outer-test rows
            try:
                est = build_estimator(f, choice["params"])
                est.fit(X[tr], y[tr])
                oof[f][te] = _score_matrix(est, X[te])
            except Exception:
                pass
        # selector: apply the full policy in this fold
        chosen = select_family_1se(family_best, simplicity)
        fold_record["selected_family"] = chosen
        if chosen is not None:
            selection_freq[chosen] += 1
            oof_selector[te] = oof[chosen][te]
        per_fold.append(fold_record)

    # per-family + selector outer metrics on the pooled OOF predictions
    metrics = {}
    for f in families:
        mask = ~np.isnan(oof[f])
        if mask.sum() > 0 and len(np.unique(y[mask])) > 1:
            metrics[f] = {
                "oof_average_precision": float(
                    average_precision_score(y[mask], oof[f][mask])),
                "oof_roc_auc": float(roc_auc_score(y[mask], oof[f][mask])),
                "n_scored": int(mask.sum()),
            }
        else:
            metrics[f] = {"oof_average_precision": None,
                          "oof_roc_auc": None,
                          "n_scored": int(mask.sum())}
    smask = ~np.isnan(oof_selector)
    if smask.sum() > 0 and len(np.unique(y[smask])) > 1:
        selector_metrics = {
            "oof_average_precision": float(
                average_precision_score(y[smask], oof_selector[smask])),
            "oof_roc_auc": float(roc_auc_score(y[smask], oof_selector[smask])),
            "n_scored": int(smask.sum()),
        }
    else:
        selector_metrics = {"oof_average_precision": None,
                            "oof_roc_auc": None,
                            "n_scored": int(smask.sum())}

    return {
        "oof": oof,
        "oof_selector": oof_selector,
        "per_fold": per_fold,
        "selection_frequency": dict(selection_freq),
        "family_metrics": metrics,
        "selector_metrics": selector_metrics,
    }


def fit_final_winner(X, y, configured, policy, log):
    """Repeat the same inner search on ALL rows, apply the one-standard-error
    rule with the policy simplicity order, fit the winner on all rows."""
    sel = policy["selection"]
    inner_folds = sel["inner_folds"]
    seed = sel["random_seed"]
    simplicity = sel["simplicity_order"]

    family_best = {}
    per_family_choice = {}
    for f in configured:
        grid = configured[f]["grid"]
        results = inner_cv_family(f, grid, X, y, inner_folds, seed)
        choice = best_params_1se(results)
        if choice is None:
            continue
        per_family_choice[f] = choice
        family_best[f] = {
            "mean_ap": choice["mean_ap"], "std_ap": choice["std_ap"],
            "n": choice["n"], "params": choice["params"],
        }
    winner = select_family_1se(family_best, simplicity)
    if winner is None:
        raise RuntimeError("no family produced a usable inner-CV result")
    win_params = per_family_choice[winner]["params"]
    log("final winner=%s params=%s" % (winner, win_params))
    est = build_estimator(winner, win_params)
    est.fit(X, y)
    return winner, win_params, est, family_best


# --------------------------------------------------------------------------- #
# AUDIT.  Printed BEFORE the statistical fill destroys the missingness         #
# evidence.  Uses is_defined flags, never value-vs-sentinel comparisons.       #
# --------------------------------------------------------------------------- #
def undefined_audit(df, y, log):
    n = len(y)
    n_merge = int(np.sum(y == 1))
    n_clean = int(np.sum(y == 0))
    print("\n==================== UNDEFINED / COVERAGE AUDIT ====================")
    print("segments: %d  merges: %d  clean: %d  prevalence: %.4f"
          % (n, n_merge, n_clean, n_merge / n if n else float("nan")))

    all_undef = np.ones(n, dtype=bool)
    print("\nper-feature undefined share (overall / merges undefined):")
    for name in FEATURE_NAMES:
        fc = name + "_is_defined"
        if fc not in df.columns:
            continue
        flag = df[fc].values.astype(bool)
        all_undef &= ~flag
        n_undef = int(np.sum(~flag))
        n_undef_merge = int(np.sum((~flag) & (y == 1)))
        share = n_undef / n if n else float("nan")
        share_m = (np.sum((~flag) & (y == 1)) / n_merge) if n_merge else float("nan")
        print("  %-42s undef=%6d (%.3f)  undef&merge=%4d (%.3f of merges)"
              % (name, n_undef, share, n_undef_merge, share_m))

    n_allundef = int(np.sum(all_undef))
    n_allundef_merge = int(np.sum(all_undef & (y == 1)))
    print("\nALL-UNDEFINED GROUP (every feature undefined): %d segments, "
          "%d of them merges" % (n_allundef, n_allundef_merge))
    if n_allundef > 0 and n_allundef_merge == 0:
        print("  WARNING: a large all-undefined group with ZERO merges is the "
              "classic confound — the model can score it trivially clean.")

    print("\nsingle-feature AUC  (full set  vs  defined-subset):")
    print("  the DEFINED-SUBSET number is the honest one and should match the "
          "AUC the report recorded for that hypothesis;")
    print("  a full-set AUC well above it is inflation the combination "
          "introduced.")
    for name in FEATURE_NAMES:
        fc = name + "_is_defined"
        if fc not in df.columns:
            continue
        flag = df[fc].values.astype(bool)
        vals = df[name].values.astype(float)
        # full set: undefined -> sentinel-free neutral fill for AUC only
        full_auc = _single_feature_auc(vals, flag, y, scope="full")
        sub_auc = _single_feature_auc(vals, flag, y, scope="defined")
        print("  %-42s full=%s  defined=%s"
              % (name, _fmt(full_auc), _fmt(sub_auc)))
    print("===================================================================\n")
    return {
        "n_segments": n, "n_merge": n_merge, "prevalence":
        (n_merge / n if n else None),
        "all_undefined_count": n_allundef,
        "all_undefined_merges": n_allundef_merge,
    }


def _fmt(x):
    return "  nan " if x is None or not np.isfinite(x) else "%.4f" % x


def _single_feature_auc(vals, flag, y, scope):
    """Directionless single-feature AUC (max of auc and 1-auc), so a feature that
    separates in either direction is credited.  For 'full', undefined rows use
    the median of the defined values as a neutral score; for 'defined', restrict
    to defined rows only."""
    from sklearn.metrics import roc_auc_score

    if scope == "defined":
        m = flag
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            return None
        v = vals[m]
        if not np.all(np.isfinite(v)):
            return None
        try:
            a = roc_auc_score(y[m], v)
        except Exception:
            return None
        return max(a, 1.0 - a)
    # full
    v = vals.copy()
    if flag.sum() == 0:
        return None
    med = float(np.median(v[flag]))
    v[~flag] = med
    if len(np.unique(y)) < 2 or not np.all(np.isfinite(v)):
        return None
    try:
        a = roc_auc_score(y, v)
    except Exception:
        return None
    return max(a, 1.0 - a)


# --------------------------------------------------------------------------- #
# FIGURES (headless).  Agg backend selected BEFORE importing pyplot; a missing  #
# matplotlib costs the figures, not the extraction.                            #
# --------------------------------------------------------------------------- #
def make_figures(fig_dir, prefix, df, y, nested, winner_oof, selector_oof,
                 winner, final_est, feature_order, heldout=False, log=None,
                 winner_params=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        if log:
            log("matplotlib unavailable, skipping figures: %r" % (exc,))
        return []
    from sklearn.metrics import (precision_recall_curve, roc_curve,
                                 average_precision_score, roc_auc_score)

    os.makedirs(fig_dir, exist_ok=True)
    written = []

    def save(fig, nn, tag):
        title_kind = "held-out" if heldout else "out-of-fold"
        name = "%s_%02d_%s.png" % (prefix, nn, tag)
        path = os.path.join(fig_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
        return title_kind

    prevalence = float(np.mean(y == 1)) if len(y) else float("nan")

    # 01: per-family OOF PR + ROC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for f, oof in nested["oof"].items():
        m = ~np.isnan(oof)
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            continue
        pr, rc, _ = precision_recall_curve(y[m], oof[m])
        ap = average_precision_score(y[m], oof[m])
        ax1.plot(rc, pr, label="%s (AP=%.3f)" % (f, ap))
        fpr, tpr, _ = roc_curve(y[m], oof[m])
        auc = roc_auc_score(y[m], oof[m])
        ax2.plot(fpr, tpr, label="%s (AUC=%.3f)" % (f, auc))
    ax1.axhline(prevalence, ls="--", color="gray",
                label="prevalence=%.3f" % prevalence)
    ax1.set_xlabel("recall"); ax1.set_ylabel("precision"); ax1.legend(fontsize=7)
    ax1.set_title("per-family PR")
    ax2.plot([0, 1], [0, 1], ls="--", color="gray")
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.legend(fontsize=7)
    ax2.set_title("per-family ROC")
    tk = save(fig, 1, "family_pr_roc")

    # 02: winner threshold / workload curves
    m = ~np.isnan(winner_oof)
    if m.sum() > 0 and len(np.unique(y[m])) > 1:
        order = np.argsort(-winner_oof[m])
        ys = y[m][order]
        k = np.arange(1, len(ys) + 1)
        tp = np.cumsum(ys == 1)
        prec = tp / k
        total_pos = max(int(np.sum(ys == 1)), 1)
        rec = tp / total_pos
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(k, prec, label="precision@k")
        ax.plot(k, rec, label="recall@k")
        ax.plot(k, tp / max(k.max(), 1), alpha=0)  # keep axis
        ax.set_xlabel("review queue size k")
        ax.set_ylabel("precision / recall")
        ax.set_title("winner (%s) %s workload" % (winner, tk))
        ax.legend()
        save(fig, 2, "winner_workload")

    if not heldout:
        # 03: winner OOF score by class
        if m.sum() > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(winner_oof[m][y[m] == 0], bins=40, alpha=0.5,
                    density=True, label="clean")
            ax.hist(winner_oof[m][y[m] == 1], bins=40, alpha=0.5,
                    density=True, label="merge")
            ax.set_xlabel("winner OOF score"); ax.set_ylabel("density")
            ax.set_title("score separation by class"); ax.legend()
            save(fig, 3, "score_by_class")

    # 04: raw-feature ECDFs + Spearman correlation (both scopes get it)
    from scipy.stats import spearmanr
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(14, 6))
    show = feature_order[:8]
    for name in show:
        v = df[name].values.astype(float)
        vf = v[np.isfinite(v)]
        if vf.size:
            xs = np.sort(vf)
            axa.plot(xs, np.linspace(0, 1, len(xs)), label=name)
    axa.set_title("raw-feature ECDFs (first 8)"); axa.legend(fontsize=6)
    axa.set_xlabel("value"); axa.set_ylabel("F(x)")
    mat = df[feature_order].values.astype(float)
    try:
        corr, _ = spearmanr(mat, nan_policy="omit")
        if np.ndim(corr) == 0:
            corr = np.array([[1.0]])
        im = axb.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(im, ax=axb)
    except Exception:
        pass
    axb.set_title("feature Spearman correlation")
    save(fig, 4, "feature_ecdf_corr")

    if heldout:
        return written

    # 05: undefined share by class
    fig, ax = plt.subplots(figsize=(11, 6))
    names = []
    sh_all = []
    sh_m = []
    for name in feature_order:
        fc = name + "_is_defined"
        flag = df[fc].values.astype(bool)
        names.append(name)
        sh_all.append(1.0 - float(np.mean(flag)))
        mm = (y == 1)
        sh_m.append(1.0 - float(np.mean(flag[mm])) if mm.sum() else np.nan)
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.2, sh_all, width=0.4, label="all segments")
    ax.bar(xpos + 0.2, sh_m, width=0.4, label="merges")
    ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_ylabel("undefined share"); ax.legend()
    ax.set_title("undefined share by class")
    save(fig, 5, "undefined_by_class")

    # 06: winner-aware model explanation. Linear models expose signed
    # standardized coefficients; EBM exposes additive term importance and shape
    # functions; tree models may expose native unsigned importance. A model
    # without a reliable native summary falls back to validation permutation
    # importance, using the same cached calculation as figure 07.
    permutation_cache = {}

    def validation_permutation():
        if "result" in permutation_cache:
            return permutation_cache["result"]
        if "error" in permutation_cache:
            raise permutation_cache["error"]
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.model_selection import train_test_split
            Xdf = df[feature_order]
            Xtr, Xva, ytr, yva = train_test_split(
                Xdf.values, y, test_size=0.25, random_state=RANDOM_SEED,
                stratify=y)
            wp = winner_params if winner_params is not None \
                else _winner_params_placeholder(nested, winner)
            est = build_estimator(winner, wp)
            est.fit(Xtr, ytr)
            result = permutation_importance(
                est, Xva, yva, n_repeats=5, random_state=RANDOM_SEED,
                scoring="average_precision")
            permutation_cache["result"] = result
            return result
        except Exception as exc:
            permutation_cache["error"] = exc
            raise

    clf = final_est.named_steps["clf"] if hasattr(
        final_est, "named_steps") else final_est
    fig6_written = False

    if winner in {"logistic_l2", "logistic_elasticnet"}:
        try:
            coef = np.ravel(clf.coef_)
            if len(coef) != len(feature_order):
                raise ValueError("coefficient count does not match feature count")
            order = np.argsort(np.abs(coef))[::-1]
            fig, ax = plt.subplots(figsize=(11, 7))
            pos = np.arange(len(order))
            ax.barh(pos, coef[order])
            ax.set_yticks(pos)
            ax.set_yticklabels([feature_order[i] for i in order], fontsize=6)
            ax.invert_yaxis()
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.set_xlabel("standardized signed coefficient")
            ax.set_title("winner model explanation: signed linear coefficients\n"
                         "credit may split across correlated features")
            save(fig, 6, "winner_model_explanation")
            fig6_written = True
        except Exception as exc:
            if log:
                log("linear winner explanation unavailable; using fallback: %r"
                    % (exc,))

    if not fig6_written and winner == "explainable_boosting":
        try:
            importance = np.asarray(clf.term_importances(), dtype=float)
            term_features = list(clf.term_features_)

            def ebm_term_name(index):
                return " x ".join(
                    feature_order[i] if 0 <= i < len(feature_order)
                    else "feature_%d" % i
                    for i in term_features[index])

            finite = np.flatnonzero(np.isfinite(importance))
            if finite.size == 0:
                raise ValueError("EBM reported no finite term importance")
            ranked = finite[np.argsort(importance[finite])[::-1]]
            top_terms = ranked[:min(20, len(ranked))]
            main_terms = [i for i in ranked if len(term_features[i]) == 1][:5]
            n_shapes = max(1, len(main_terms))
            fig = plt.figure(figsize=(17, max(7, 2.2 * n_shapes)))
            grid = fig.add_gridspec(n_shapes, 2, width_ratios=[1.15, 1.85])
            ax_imp = fig.add_subplot(grid[:, 0])
            pos = np.arange(len(top_terms))
            ax_imp.barh(pos, importance[top_terms])
            ax_imp.set_yticks(pos)
            ax_imp.set_yticklabels(
                [ebm_term_name(i) for i in top_terms], fontsize=6)
            ax_imp.invert_yaxis()
            ax_imp.set_xlabel("mean absolute additive contribution")
            ax_imp.set_title("EBM global term importance\n"
                             "main effects and interactions; unsigned")

            explanation = clf.explain_global()
            if not main_terms:
                ax_shape = fig.add_subplot(grid[:, 1])
                ax_shape.text(0.5, 0.5, "No univariate EBM terms available",
                              ha="center", va="center")
                ax_shape.set_axis_off()
            for row, term_index in enumerate(main_terms):
                ax_shape = fig.add_subplot(grid[row, 1])
                data = explanation.data(int(term_index))
                scores = np.asarray(data["scores"], dtype=float).reshape(-1)
                raw_names = list(data["names"])
                x = None
                try:
                    numeric_names = np.asarray(raw_names, dtype=float)
                    if (len(numeric_names) == len(scores) + 1 and
                            np.all(np.isfinite(numeric_names))):
                        x = (numeric_names[:-1] + numeric_names[1:]) / 2.0
                    elif (len(numeric_names) == len(scores) and
                          np.all(np.isfinite(numeric_names))):
                        x = numeric_names
                except (TypeError, ValueError):
                    x = None
                if x is None:
                    x = np.arange(len(scores))
                    if len(raw_names) == len(scores):
                        shown = np.linspace(
                            0, max(len(scores) - 1, 0),
                            min(len(scores), 8), dtype=int)
                        ax_shape.set_xticks(shown)
                        ax_shape.set_xticklabels(
                            [str(raw_names[i]) for i in shown],
                            rotation=30, ha="right", fontsize=6)
                ax_shape.plot(x, scores, linewidth=1.5)
                lower = data.get("lower_bounds")
                upper = data.get("upper_bounds")
                if lower is not None and upper is not None:
                    lower = np.asarray(lower, dtype=float).reshape(-1)
                    upper = np.asarray(upper, dtype=float).reshape(-1)
                    if len(lower) == len(x) and len(upper) == len(x):
                        ax_shape.fill_between(x, lower, upper, alpha=0.18)
                ax_shape.axhline(0.0, color="black", linewidth=0.7)
                feature_index = term_features[term_index][0]
                ax_shape.set_title(feature_order[feature_index], fontsize=8)
                ax_shape.set_ylabel("additive score", fontsize=7)
            fig.suptitle("winner model explanation: EBM terms and top shapes",
                         fontsize=12)
            save(fig, 6, "winner_model_explanation")
            fig6_written = True
        except Exception as exc:
            if log:
                log("EBM winner explanation unavailable; using fallback: %r"
                    % (exc,))

    if not fig6_written:
        try:
            native = np.asarray(clf.feature_importances_, dtype=float).reshape(-1)
            if len(native) != len(feature_order):
                raise ValueError("native importance count does not match features")
            finite = np.flatnonzero(np.isfinite(native))
            order = finite[np.argsort(native[finite])[::-1]][:25]
            fig, ax = plt.subplots(figsize=(11, 7))
            pos = np.arange(len(order))
            ax.barh(pos, native[order])
            ax.set_yticks(pos)
            ax.set_yticklabels([feature_order[i] for i in order], fontsize=6)
            ax.invert_yaxis()
            ax.set_xlabel("native model-specific importance (unsigned)")
            ax.set_title("winner model explanation: %s native importance\n"
                         "values are not comparable across model families"
                         % winner)
            save(fig, 6, "winner_model_explanation")
            fig6_written = True
        except Exception as exc:
            if log:
                log("native winner explanation unavailable; using permutation "
                    "fallback: %r" % (exc,))

    if not fig6_written:
        try:
            r = validation_permutation()
            order = np.argsort(r.importances_mean)[::-1]
            fig, ax = plt.subplots(figsize=(11, 7))
            pos = np.arange(len(order))
            ax.barh(pos, r.importances_mean[order],
                    xerr=r.importances_std[order])
            ax.set_yticks(pos)
            ax.set_yticklabels([feature_order[i] for i in order], fontsize=6)
            ax.invert_yaxis()
            ax.set_xlabel("validation AP decrease after permutation")
            ax.set_title("winner model explanation: permutation fallback\n"
                         "no reliable native global importance was available")
            save(fig, 6, "winner_model_explanation")
            fig6_written = True
        except Exception as exc:
            if log:
                log("winner-explanation figure skipped: %r" % (exc,))

    # 07: common model-agnostic permutation importance on validation data.
    try:
        r = validation_permutation()
        fig, ax = plt.subplots(figsize=(11, 6))
        order = np.argsort(r.importances_mean)[::-1]
        ax.bar(range(len(feature_order)), r.importances_mean[order],
               yerr=r.importances_std[order])
        ax.set_xticks(range(len(feature_order)))
        ax.set_xticklabels([feature_order[i] for i in order],
                           rotation=90, fontsize=5)
        ax.set_title("permutation importance (validation, AP-scored)")
        save(fig, 7, "permutation_importance")
    except Exception as exc:
        if log:
            log("permutation-importance figure skipped: %r" % (exc,))

    return written


def _winner_params_placeholder(nested, winner):
    """Best-effort winner params for figure 06 fallback and figure 07.

    These figures train their own validation split and need one valid grid point.
    """
    for rec in reversed(nested["per_fold"]):
        ch = rec["family_choice"].get(winner)
        if ch:
            return ch["params"]
    return {}


# --------------------------------------------------------------------------- #
# CONFIG LOADING + candidate assembly (skips unavailable optional packages).    #
# --------------------------------------------------------------------------- #
def load_and_validate_config(config_path, inventory_sha256):
    with open(config_path, "r") as fh:
        cfg = json.load(fh)
    validate_model_config(cfg, inventory_sha256)
    return cfg


def assemble_configured(cfg, log):
    """Return (configured, skipped).  configured maps family->{grid,...} for
    families whose optional dependency is importable; unavailable optional
    families are recorded as skipped (never installed)."""
    configured = {}
    skipped = []
    for cand in cfg["candidates"]:
        name = cand["name"]
        pkg = EMBEDDED_MODEL_POLICY["families"][name]["requires_package"]
        if not _package_available(pkg):
            skipped.append({"family": name, "reason":
                            "optional package %r unavailable" % (pkg,)})
            log("SKIP family %s: optional package %r not importable"
                % (name, pkg))
            continue
        configured[name] = {"grid": cand.get("grid", {}), "role": cand.get("role")}
    return configured, skipped


# --------------------------------------------------------------------------- #
# SYNTHETIC SMOKE TEST.  No data files, no outputs; prints DETECTOR_SMOKE_OK    #
# ONLY after every assertion passes.  An invalid/unknown family in the config   #
# must already have failed validation (nonzero exit, no marker, no file).       #
# --------------------------------------------------------------------------- #
def run_smoke_test(config_path, inventory_sha256, log):
    log("synthetic smoke test: validating config %s" % config_path)
    cfg = load_and_validate_config(config_path, inventory_sha256)
    configured, skipped = assemble_configured(cfg, log)
    assert configured, "no configured families after availability filtering"

    # small synthetic frame with NaNs and known missingness structure
    rng = np.random.RandomState(RANDOM_SEED)
    n = 240
    p = len(FEATURE_NAMES)
    y = (rng.rand(n) < 0.15).astype(np.int64)
    base = rng.randn(n, p)
    base[:, 0] += 1.5 * y  # give one feature real signal
    # inject structural missingness in some columns
    miss = rng.rand(n, p) < 0.3
    X = base.copy()
    X[miss] = np.nan
    # an all-undefined group with zero merges
    allmiss_idx = np.where(y == 0)[0][:20]
    X[allmiss_idx, :] = np.nan

    policy = EMBEDDED_MODEL_POLICY
    nested = run_nested_selection(X, y, configured, policy, log)

    # assert: every expected OOF column is filled exactly once per outer row
    for f in configured:
        col = nested["oof"][f]
        assert col.shape == (n,), "family %s OOF wrong shape" % f
        filled = np.sum(~np.isnan(col))
        assert filled == n, ("family %s filled %d/%d OOF rows"
                             % (f, filled, n))
    sel = nested["oof_selector"]
    assert np.sum(~np.isnan(sel)) == n, "selector OOF not fully filled"

    # assert: fold-local preprocessing handles NaNs (a linear family produced
    # finite scores despite NaN inputs)
    lin = "logistic_l2"
    assert lin in configured, "logistic_l2 must be configured"
    assert np.all(np.isfinite(nested["oof"][lin])), \
        "logistic_l2 produced non-finite OOF scores (NaN handling failed)"

    # fit final winner, then prove held-out prediction does NOT mutate the fitted
    # preprocessing (imputer statistics unchanged after scoring a second frame).
    winner, win_params, est, _ = fit_final_winner(X, y, configured, policy, log)
    pre_state = _snapshot_preprocessing(est)
    X_test = rng.randn(50, p)
    X_test[rng.rand(50, p) < 0.4] = np.nan
    _ = _score_matrix(est, X_test)
    post_state = _snapshot_preprocessing(est)
    assert _states_equal(pre_state, post_state), \
        "held-out scoring mutated fitted preprocessing statistics"

    # optional-dependency-backed candidate skipping is exercised: if any optional
    # family was configured-but-unavailable it must be in `skipped`.
    for s in skipped:
        assert s["family"] in ALLOWED_FAMILIES

    # ensure NO output files were created by the smoke path
    print("DETECTOR_SMOKE_OK")
    return 0


def _snapshot_preprocessing(est):
    """Capture imputer/scaler learned statistics for mutation-detection."""
    import numpy as _np

    snap = {}
    steps = getattr(est, "named_steps", None)
    if steps is None:
        return snap
    for sname, step in steps.items():
        for attr in ("statistics_", "mean_", "scale_", "var_", "n_features_in_"):
            if hasattr(step, attr):
                val = getattr(step, attr)
                snap["%s.%s" % (sname, attr)] = _np.asarray(val).copy() \
                    if hasattr(val, "__len__") else val
    return snap


def _states_equal(a, b):
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if hasattr(va, "shape"):
            if not np.array_equal(np.asarray(va), np.asarray(vb),
                                  equal_nan=True):
                return False
        elif va != vb:
            return False
    return True


# --------------------------------------------------------------------------- #
# CLI / MAIN.                                                                   #
# --------------------------------------------------------------------------- #
def build_arg_parser():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="ExaSPIM merge-site segment detector (extract once, "
                    "compare families, score out-of-fold).")
    ap.add_argument("pkl", nargs="?", default=None,
                    help="training _add.pkl (positional). Omit only for "
                         "--synthetic-smoke-test.")
    ap.add_argument("--test-pkl", default=None,
                    help="a DIFFERENT brain's _add.pkl for held-out transfer.")
    ap.add_argument("--model-config", default=os.path.join(
        here, "model_candidates.json"),
        help="model_candidates.json (default beside the script).")
    ap.add_argument("--inventory", default=os.path.join(
        here, "feature_inventory.json"),
        help="feature_inventory.json (for SHA cross-check).")
    ap.add_argument("--out-csv", default=None,
                    help="feature/score CSV (default merge_detector_<brain>.csv "
                         "beside the script).")
    ap.add_argument("--out-dir", default=here,
                    help="directory for JSON/joblib/log outputs.")
    ap.add_argument("--fig-dir", default=None,
                    help="figure directory (default <out-dir>/figures).")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--exclude-empty", action="store_true",
                    help="drop rows whose is_defined flags are ALL zero "
                         "(applied identically to train and held-out).")
    ap.add_argument("--review-budget", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--log-txt", default=None)
    ap.add_argument("--measuretime", action="store_true",
                    help="profile hypothesis computation cost on a small "
                         "segment sample, write cost artifacts, and exit "
                         "before models.")
    ap.add_argument("--measuretime-occurrences", type=int, default=3,
                    help="maximum component-bearing segments sampled by "
                         "--measuretime (default: 3; must be positive).")
    ap.add_argument("--hypothesis-selection", default=None,
                    help="schema-v1 JSON selecting hypotheses to exclude from "
                         "feature computation in the final train/test run.")
    ap.add_argument("--exclude-hypotheses", nargs="+", type=int, metavar="ID",
                    help="manually exclude hypothesis IDs after reviewing the "
                         "cost report; cannot "
                         "be combined with --hypothesis-selection.")
    ap.add_argument("--synthetic-smoke-test", action="store_true")
    return ap


def _inventory_sha(inventory_path):
    if inventory_path and os.path.exists(inventory_path):
        return _sha256_file(inventory_path)
    return None


def _drop_all_undefined(df, adjudicable, y):
    flag_cols = [c for c in df.columns if c.endswith("_is_defined")]
    flags = df[flag_cols].values
    keep = flags.sum(axis=1) > 0
    return (df.loc[keep].reset_index(drop=True),
            [s for s, k in zip(adjudicable, keep) if k],
            y[keep], keep)


def _apply_scope(feat_df, adjudicable, is_merge, exclude_empty, log):
    if not exclude_empty:
        return feat_df, adjudicable, is_merge
    df2, adj2, y2, _ = _drop_all_undefined(feat_df, adjudicable, is_merge)
    log("--exclude-empty: dropped %d all-undefined rows (kept %d)"
        % (len(adjudicable) - len(adj2), len(adj2)))
    return df2, adj2, y2


def _print_queue(adjudicable, is_merge, scores, budget, title):
    order = np.argsort(-scores)
    print("\n%s (top %d by score):" % (title, min(budget, len(order))))
    print("  rank  segment_id     score  is_merge")
    for rank, idx in enumerate(order[:budget], 1):
        print("  %4d  %10d  %.5f  %d"
              % (rank, adjudicable[idx], scores[idx], int(is_merge[idx])))
        if rank >= 20:
            break


def _collect_measuretime_versions():
    """Record extraction dependencies without importing plotting/model extras."""
    versions = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "sklearn", "networkx"):
        try:
            module = __import__(mod)
            versions[mod] = getattr(module, "__version__", "?")
        except Exception:
            versions[mod] = None
    return versions


def run_measuretime(args, log):
    """Profile all hypotheses on a small sample; write rough cost artifacts."""
    if args.test_pkl:
        raise ValueError("--measuretime cannot be combined with --test-pkl")
    if args.out_csv:
        raise ValueError("--measuretime does not write --out-csv")
    if args.exclude_empty:
        raise ValueError("--measuretime cannot be combined with --exclude-empty")
    if args.hypothesis_selection or args.exclude_hypotheses:
        raise ValueError(
            "--measuretime profiles all hypotheses and cannot use "
            "--hypothesis-selection or --exclude-hypotheses")
    if args.measuretime_occurrences <= 0:
        raise ValueError("--measuretime-occurrences must be positive")

    validate_analysis_timing_groups(ANALYSIS_TIMING_GROUPS)
    os.makedirs(args.out_dir, exist_ok=True)
    brain = brain_id_from_path(args.pkl)
    artifact = os.path.join(args.out_dir, "analysis_timing_%s.json" % brain)
    inventory_sha = _inventory_sha(args.inventory)
    if inventory_sha is None:
        raise ValueError("--measuretime requires a readable --inventory file")
    metadata = {
        "detector_path": os.path.abspath(__file__),
        "detector_sha256": _sha256_file(__file__),
        "feature_inventory_path": os.path.abspath(args.inventory),
        "feature_inventory_sha256": inventory_sha,
        "input_pkl": os.path.abspath(args.pkl),
        "input_size_bytes": os.path.getsize(args.pkl),
        "brain": brain,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "checkpoint_min_interval_seconds": 60.0,
        "top_slowest_limit": 20,
        "sampled": True,
        "profile_occurrence_limit": int(args.measuretime_occurrences),
        "profile_sampling_method": "first_component_bearing_segments",
        "cost_scope": "sample_observed_not_full_run",
    }
    recorder = AnalysisTimingRecorder(artifact, ANALYSIS_TIMING_GROUPS, metadata)
    total_started = time.perf_counter()
    payload = None
    try:
        log("measuretime: loading payload: %s" % args.pkl)
        started = time.perf_counter()
        payload = load_payload(args.pkl)
        recorder.record_wall_seconds(
            "payload_load", time.perf_counter() - started)
        recorder.update_metadata(min_cable_length=mcl_from_payload(payload, args.pkl))

        total_adjudicable, total_is_merge = build_segment_universe(payload)
        recorder.update_metadata(
            n_total_adjudicable_segments=len(total_adjudicable),
            n_total_merge_labels=int(np.sum(total_is_merge == 1)),
        )
        log("measuretime: profiling at most %d component-bearing segments only"
            % args.measuretime_occurrences)
        started = time.perf_counter()
        adjudicable, is_merge, _ = extract_features(
            payload, verbose=True, timing=recorder,
            profile_segment_limit=args.measuretime_occurrences)
        recorder.record_wall_seconds(
            "feature_extraction", time.perf_counter() - started)
        recorder.update_metadata(
            n_adjudicable_segments=len(adjudicable),
            n_merge_labels=int(np.sum(is_merge == 1)),
            n_profiled_segments=len(adjudicable),
            profiled_segment_ids=[int(x) for x in adjudicable],
            dependency_versions=_collect_measuretime_versions(),
        )
        recorder.record_wall_seconds(
            "total", time.perf_counter() - total_started)
        recorder.finish()
        log("measuretime: wrote timing artifact: %s" % artifact)
        report_path, selection_path = write_hypothesis_cost_artifacts(
            artifact, args.out_dir, brain)
        log("measuretime: wrote hypothesis cost report: %s" % report_path)
        log("measuretime: wrote editable selection template: %s" % selection_path)
        return 0
    except BaseException as exc:
        recorder.record_wall_seconds(
            "total", time.perf_counter() - total_started)
        recorder.fail("%s: %s" % (type(exc).__name__, exc))
        log("measuretime: wrote partial failed artifact: %s" % artifact)
        raise
    finally:
        if payload is not None:
            del payload
        gc.collect()


def run_detector(args, log):
    import pandas as pd
    import joblib

    inventory_sha = _inventory_sha(args.inventory)
    cfg = load_and_validate_config(args.model_config, inventory_sha)
    config_sha = _sha256_file(args.model_config)
    configured, skipped = assemble_configured(cfg, log)
    if not configured:
        raise RuntimeError("no configured families available")

    policy = EMBEDDED_MODEL_POLICY
    hypothesis_selection = load_hypothesis_selection(
        args.hypothesis_selection, ANALYSIS_TIMING_GROUPS,
        excluded_hypothesis_ids=args.exclude_hypotheses)
    timing_inventory_sha = hypothesis_selection[
        "source_timing_feature_inventory_sha256"]
    if (timing_inventory_sha is not None and
            timing_inventory_sha != inventory_sha):
        raise ValueError(
            "source timing feature inventory SHA-256 %r != current inventory %r"
            % (timing_inventory_sha, inventory_sha))
    enabled_analysis_keys = set(
        hypothesis_selection["enabled_analysis_keys"])
    excluded_feature_names = set(
        hypothesis_selection["excluded_feature_names"])
    model_feature_names = [
        name for name in FEATURE_NAMES if name not in excluded_feature_names
    ]
    if hypothesis_selection["excluded_hypothesis_ids"]:
        log("hypothesis selection: excluded ids=%r; enabled groups=%d/%d"
            % (hypothesis_selection["excluded_hypothesis_ids"],
               len(enabled_analysis_keys), len(ANALYSIS_TIMING_GROUPS)))
    train_brain = brain_id_from_path(args.pkl)
    scope = "excludeempty" if args.exclude_empty else "full"
    prefix = "%s_%s" % (train_brain, scope)

    log("loading training payload: %s" % args.pkl)
    payload = load_payload(args.pkl)
    train_mcl = mcl_from_payload(payload, args.pkl)
    adjudicable, is_merge, acc = extract_features(
        payload, verbose=True, enabled_analysis_keys=enabled_analysis_keys)
    feat_df = acc.to_frame()
    # release the big payload before anything else
    del payload
    gc.collect()

    feat_df, adjudicable, is_merge = _apply_scope(
        feat_df, adjudicable, is_merge, args.exclude_empty, log)
    y = np.asarray(is_merge, dtype=np.int64)

    audit = undefined_audit(feat_df, y, log)

    X = feat_df[model_feature_names].values.astype(float)
    log("running nested model selection (outer=%d inner=%d)"
        % (policy["selection"]["outer_folds"],
           policy["selection"]["inner_folds"]))
    nested = run_nested_selection(X, y, configured, policy, log)

    print("\n==================== CANDIDATE COMPARISON ====================")
    for f, m in nested["family_metrics"].items():
        print("  %-24s OOF AP=%s  OOF ROC-AUC=%s  (n=%d)"
              % (f, _fmt(m["oof_average_precision"]),
                 _fmt(m["oof_roc_auc"]), m["n_scored"]))
    sm = nested["selector_metrics"]
    print("  %-24s OOF AP=%s  OOF ROC-AUC=%s  (n=%d)"
          % ("SELECTOR (policy)", _fmt(sm["oof_average_precision"]),
             _fmt(sm["oof_roc_auc"]), sm["n_scored"]))
    print("  selection frequency:", nested["selection_frequency"])

    winner, win_params, final_est, _ = fit_final_winner(
        X, y, configured, policy, log)
    print("\nFINAL WINNER: %s   params=%s" % (winner, win_params))
    print("  (chosen by one-standard-error rule in simplicity order %s)"
          % (policy["selection"]["simplicity_order"],))

    winner_oof = nested["oof"][winner]
    selector_oof = nested["oof_selector"]
    in_sample = _score_matrix(final_est, X)  # reference-only

    # threshold sweep + review-budget metrics on winner OOF
    _report_thresholds(y, winner_oof, args.review_budget)
    _print_queue(adjudicable, is_merge, np.where(
        np.isnan(winner_oof), -np.inf, winner_oof),
        args.review_budget, "TOP REVIEW QUEUE (winner OOF)")

    # ---- write CSV ---- #
    out_csv = args.out_csv or os.path.join(
        args.out_dir, "merge_detector_%s.csv" % train_brain)
    csv_df = pd.DataFrame({"segment_id": adjudicable, "is_merge": y})
    for name in FEATURE_NAMES:
        csv_df[name] = feat_df[name].values
        csv_df[name + "_is_defined"] = feat_df[name + "_is_defined"].values
    for f in configured:
        csv_df["merge_probability_oof_%s" % f] = nested["oof"][f]
    csv_df["merge_probability_oof_selector"] = selector_oof
    csv_df["merge_probability_oof"] = winner_oof
    csv_df["merge_probability_in_sample"] = in_sample
    csv_df.to_csv(out_csv, index=False)
    log("wrote CSV: %s" % out_csv)

    # ---- held-out transfer ---- #
    heldout_report = None
    if args.test_pkl:
        heldout_report = _run_heldout(
            args, final_est, winner, win_params, configured, policy,
            train_brain, train_mcl, scope, hypothesis_selection,
            model_feature_names, log)

    # ---- model-selection result JSON ---- #
    versions = _collect_versions()
    model_selection_record = {
        "model_config_path": os.path.abspath(args.model_config),
        "model_config_sha256": config_sha,
        "model_policy_sha256": EMBEDDED_MODEL_POLICY_SHA256,
        "feature_inventory_sha256": inventory_sha,
        "train_pkl": os.path.abspath(args.pkl),
        "train_brain": train_brain,
        "train_mcl": train_mcl,
        "scope": scope,
        "seeds": {"random_seed": RANDOM_SEED},
        "registry": {f: configured[f]["grid"] for f in configured},
        "skipped_candidates": skipped,
        "dependency_versions": versions,
        "per_fold_inner_choices": nested["per_fold"],
        "per_family_outer_metrics": nested["family_metrics"],
        "selector_metrics": nested["selector_metrics"],
        "selection_frequency": nested["selection_frequency"],
        "final_winner": winner,
        "final_params": win_params,
        "feature_order": model_feature_names,
        "all_feature_order": FEATURE_NAMES,
        "hypothesis_selection": hypothesis_selection,
        "audit": audit,
        "heldout": heldout_report,
    }
    out_json = os.path.join(args.out_dir, "model_selection_%s.json" % train_brain)
    with open(out_json, "w") as fh:
        json.dump(model_selection_record, fh, indent=2, default=_json_default)
    log("wrote model-selection JSON: %s" % out_json)

    # ---- joblib ---- #
    out_joblib = os.path.join(args.out_dir,
                              "merge_detector_%s.joblib" % train_brain)
    joblib.dump({"pipeline": final_est, "feature_order": model_feature_names,
                 "all_feature_order": FEATURE_NAMES,
                 "winner": winner, "params": win_params,
                 "scope": scope,
                 "hypothesis_selection": hypothesis_selection}, out_joblib)
    log("wrote joblib: %s" % out_joblib)

    # ---- figures ---- #
    if not args.no_figures:
        fig_dir = args.fig_dir or os.path.join(args.out_dir, "figures")
        make_figures(fig_dir, prefix, feat_df, y, nested, winner_oof,
                     selector_oof, winner, final_est, model_feature_names,
                     heldout=False, log=log, winner_params=win_params)
        log("wrote training figures to %s" % fig_dir)

    return 0


def _report_thresholds(y, oof, budget):
    from sklearn.metrics import average_precision_score, roc_auc_score
    m = ~np.isnan(oof)
    print("\n==================== OOF THRESHOLD / WORKLOAD ====================")
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        print("  insufficient OOF coverage for threshold sweep.")
        return
    yy = y[m]
    ss = oof[m]
    ap = average_precision_score(yy, ss)
    auc = roc_auc_score(yy, ss)
    prevalence = float(np.mean(yy == 1))
    print("  OOF average precision (PRIMARY): %.4f   [prevalence=%.4f]"
          % (ap, prevalence))
    print("  OOF ROC-AUC (secondary under imbalance): %.4f" % auc)
    order = np.argsort(-ss)
    ys = yy[order]
    total_pos = max(int(np.sum(ys == 1)), 1)
    for k in sorted(set([budget, 50, 100, 200, 500])):
        if k > len(ys):
            continue
        tp = int(np.sum(ys[:k] == 1))
        print("  @%-4d  precision=%.4f  recall=%.4f  merges_found=%d"
              % (k, tp / k, tp / total_pos, tp))


def _run_heldout(args, final_est, winner, win_params, configured, policy,
                 train_brain, train_mcl, scope, hypothesis_selection,
                 model_feature_names, log):
    from sklearn.metrics import (average_precision_score, roc_auc_score)

    test_brain = brain_id_from_path(args.test_pkl)
    log("loading held-out payload: %s" % args.test_pkl)
    payload = load_payload(args.test_pkl)
    test_mcl = mcl_from_payload(payload, args.test_pkl)
    if train_mcl is not None and test_mcl is not None and train_mcl != test_mcl:
        log("WARNING: mcl mismatch train=%s test=%s — the filter changes both "
            "the dataless fraction and the fill-factor gate." % (train_mcl,
                                                                 test_mcl))
    adj_t, is_merge_t, acc_t = extract_features(
        payload, verbose=True,
        enabled_analysis_keys=set(hypothesis_selection["enabled_analysis_keys"]))
    df_t = acc_t.to_frame()
    del payload
    gc.collect()

    df_t, adj_t, is_merge_t = _apply_scope(
        df_t, adj_t, is_merge_t, args.exclude_empty, log)
    y_t = np.asarray(is_merge_t, dtype=np.int64)

    # test brain's own all-undefined share (context for the transfer gap)
    flag_cols = [c for c in df_t.columns if c.endswith("_is_defined")]
    all_undef_share = float(np.mean(df_t[flag_cols].values.sum(axis=1) == 0))

    X_t = df_t[model_feature_names].values.astype(float)
    # score with the fitted pipeline; TRAINING imputation constant is applied
    # (SimpleImputer inside the pipeline uses its fitted statistics — the test
    #  brain's own statistics never enter).
    scores_t = _score_matrix(final_est, X_t)

    print("\n==================== HELD-OUT TRANSFER (%s from %s) ============"
          % (test_brain, train_brain))
    ok = len(np.unique(y_t)) > 1
    if ok:
        ap_t = float(average_precision_score(y_t, scores_t))
        auc_t = float(roc_auc_score(y_t, scores_t))
    else:
        ap_t = auc_t = float("nan")
    print("  held-out AP=%.4f  ROC-AUC=%.4f  (prevalence=%.4f)"
          % (ap_t, auc_t, float(np.mean(y_t == 1))))
    print("  test brain all-undefined share: %.4f" % all_undef_share)
    order = np.argsort(-scores_t)
    ys = y_t[order]
    total_pos = max(int(np.sum(ys == 1)), 1)
    for k in [args.review_budget, 100]:
        if k <= len(ys):
            tp = int(np.sum(ys[:k] == 1))
            print("  @%-4d precision=%.4f recall=%.4f"
                  % (k, tp / k, tp / total_pos))

    heldout_report = {
        "test_pkl": os.path.abspath(args.test_pkl),
        "test_brain": test_brain, "test_mcl": test_mcl,
        "mcl_mismatch": (train_mcl != test_mcl),
        "held_out_average_precision": ap_t if ok else None,
        "held_out_roc_auc": auc_t if ok else None,
        "test_all_undefined_share": all_undef_share,
        "n_test_segments": int(len(y_t)),
        "n_test_merges": int(np.sum(y_t == 1)),
    }

    # held-out figures: ONLY 01, 02, 04 (03 is a training-fit property, 05-07
    # describe the training data). Tagged heldout-from-<train>.
    if not args.no_figures:
        fig_dir = args.fig_dir or os.path.join(args.out_dir, "figures")
        ho_prefix = "%s_%s_heldout-from-%s" % (test_brain, scope, train_brain)
        nested_ho = {"oof": {winner: scores_t}}
        make_figures(fig_dir, ho_prefix, df_t, y_t, nested_ho, scores_t,
                     scores_t, winner, final_est, model_feature_names,
                     heldout=True, log=log)
        log("wrote held-out figures (01,02,04) to %s" % fig_dir)
    return heldout_report


def _collect_versions():
    v = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "networkx",
                "matplotlib", "joblib"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "?")
        except Exception:
            v[mod] = None
    return v


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.synthetic_smoke_test and (args.measuretime or
                                      args.hypothesis_selection or
                                      args.exclude_hypotheses):
        parser.error(
            "--synthetic-smoke-test cannot use --measuretime, "
            "--hypothesis-selection, or --exclude-hypotheses")

    # --- smoke test: no data, no outputs, no log file. --- #
    if args.synthetic_smoke_test:
        inv_sha = _inventory_sha(args.inventory)

        def _log(msg):
            print("[smoke] " + msg, flush=True)

        return run_smoke_test(args.model_config, inv_sha, _log)

    if not args.pkl:
        parser.error(
            "a positional _add.pkl is required unless "
            "--synthetic-smoke-test is given")

    # --- set up tee logging --- #
    brain = brain_id_from_path(args.pkl)
    log_path = args.log_txt or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "merge_detector_%s.log.txt" % brain)
    real_out, real_err = sys.stdout, sys.stderr
    fh = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(real_out, fh)
    sys.stderr = _Tee(real_err, fh)
    start = time.time()

    def _log(msg):
        print("[detector] " + msg, flush=True)

    ok = False
    try:
        print("# merge_site_detector run")
        print("# argv: %s" % " ".join(sys.argv))
        print("# start: %s" % time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(start)))
        print("# host: %s" % socket.gethostname())
        print("# python: %s" % platform.python_version())
        rc = run_measuretime(args, _log) if args.measuretime else run_detector(args, _log)
        ok = (rc == 0)
        return rc
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        elapsed = time.time() - start
        print("# %s after %.1fs" % ("OK" if ok else "FAILED", elapsed))
        sys.stdout = real_out
        sys.stderr = real_err
        try:
            fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
