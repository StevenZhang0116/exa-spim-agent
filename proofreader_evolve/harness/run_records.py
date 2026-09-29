"""Read fixed-pool scorer records; missing measurements remain unknown.

No graph loading, policy execution or SDK imports. Other objectives are rejected,
not reinterpreted as precision runs.
"""

import json
import math
from pathlib import Path

PRECISION = "native-precision-at-k-v1"
METRIC_NAME = "macro native-label Precision@K"


def objective(row):
    version = row.get("objective", row.get("scoring_version"))
    if version is None and "validation" in row and "train" in row:
        version = PRECISION
    if version != PRECISION:
        raise ValueError(f"Only fixed-pool precision runs are supported, got {version!r}")
    return version


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def load_ledger(run_dir):
    path = Path(run_dir) / "ledger.jsonl"
    if not path.exists():
        if (Path(run_dir) / "seed.json").exists():
            run_objective(run_dir)
            return []
        raise FileNotFoundError(f"No ledger at {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["generation"])
    if len({r["generation"] for r in rows}) != len(rows):
        raise ValueError("Duplicate generation in ledger")
    run_objective(run_dir, rows)
    return rows


def run_objective(run_dir=None, rows=()):
    versions = {objective(r) for r in rows}
    if run_dir is not None:
        metadata = read_json(Path(run_dir) / "manifest.json", {})
        if "objective" in metadata:
            versions.add(metadata["objective"])
    if versions != {PRECISION}:
        raise ValueError("Only fixed-pool precision runs with a known objective are supported")
    return PRECISION


def number(value):
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def measurement_value(measurement):
    return None if measurement is None else number(measurement.get("macro_precision"))


def validation_value(row):
    objective(row)
    return measurement_value(row.get("validation"))


def parent_values(rows, seed=None):
    current = number(seed)
    result = []
    for row in rows:
        objective(row)
        recorded = number(row.get("parent_validation_precision"))
        if recorded is not None:
            current = recorded
        result.append(current)
        if row.get("accepted"):
            current = validation_value(row)
    return result


def decision_reason(row):
    return row.get("reason") or "reason not recorded"


def paired_measurements(rows, run_dir=None):
    """Pair completed TRAIN/validation measurements of the same scorer."""
    run_objective(run_dir, rows)
    pairs = []
    for row in rows:
        train, validation = row.get("train"), row.get("validation")
        if train is None or validation is None:
            continue
        if not train.get("policy_sha256") or train["policy_sha256"] != validation.get("policy_sha256"):
            raise ValueError("Mismatched policies in train/validation pair")
        pairs.append({"generation": row["generation"], "policy_sha256": train["policy_sha256"],
                      "train": measurement_value(train),
                      "validation": measurement_value(validation)})
    return pairs


def cumulative_accounting(rows):
    """Sum fresh-session generation costs; missing amounts propagate as unknown."""
    total, values = 0.0, []
    for row in rows:
        objective(row)
        if row.get("accounting_semantics", "per_generation") != "per_generation":
            raise ValueError("Precision runs require per_generation accounting")
        value = number((row.get("reviser") or {}).get("cost_usd"))
        if value is not None and value < 0:
            raise ValueError("Negative accounting value")
        total = None if value is None or total is None else total + value
        values.append(total)
    return values
