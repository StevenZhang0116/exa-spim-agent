"""Read-only launch checks; no brain loading, table building or evolution.

Optional --check-api sends one tiny text-only request to the configured model.
It verifies direct Anthropic access, not the full reviser SDK/tool workflow.
"""

import argparse
import json
import os
import socket

from . import precompute_error_scores as pc
from ..harness.reviser_session import DEFAULT_MODEL
from ..harness.native_pool import ensure_native_tables, cache_path as native_cache_path


def api_probe(model):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"status": "missing_key", "model": model}
    try:
        from anthropic import Anthropic
        # Pin the same provider as the reviser; never inherit a proxy base URL.
        with Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                       base_url="https://api.anthropic.com", timeout=20, max_retries=0) as client:
            reply = client.messages.create(model=model, max_tokens=16,
                                           messages=[{"role": "user", "content": "Reply OK."}])
        return {"status": "passed", "model": reply.model,
                "input_tokens": reply.usage.input_tokens,
                "output_tokens": reply.usage.output_tokens}
    except Exception as exc:
        # No response body/headers/request object: they may contain credentials
        # or provider diagnostics inappropriate for a durable public report.
        return {"status": "failed", "model": model, "error_type": type(exc).__name__,
                "http_status": getattr(exc, "status_code", None)}


def selection_readiness(report, train_brains, selection_brains, protocol):
    """Brain-role check for the selection protocol from table provenance only.

    The fragment graph needed for split spatial folds on a selection brain is
    loaded at run start, not here (it requires the full brain cache).
    """
    train_brains = [str(b) for b in train_brains or []]
    if not train_brains:
        return None
    fitted, missing = set(), []
    for brain in train_brains:
        entry = report["brains"].get(brain, {})
        provenance = entry.get("detector_training_brains")
        if entry.get("status") == "passed" and not provenance:
            missing.append(brain)
        fitted.update(provenance or [])
    if protocol == "grouped_oof":
        chosen = [str(b) for b in selection_brains] if selection_brains else [b for b in train_brains if b not in fitted]
    else:
        chosen = list(train_brains)
    conflicts = sorted(set(chosen) & fitted) if protocol == "grouped_oof" else []
    outside = sorted(set(chosen) - set(train_brains))
    blocked = protocol == "grouped_oof" and (not chosen or conflicts or missing or outside)
    return {"protocol": protocol, "train_brains": train_brains, "detector_training_brains": sorted(fitted),
            "selection_brains": chosen, "auxiliary_train_brains": [b for b in train_brains if b not in chosen],
            "missing_provenance": missing, "conflicts": conflicts, "not_train": outside,
            "status": "blocked" if blocked else "passed",
            "note": "Selection brains must not be detector training brains; fragment-graph availability "
                    "for split folds is verified when the run starts"}


def context_cache_readiness(banks, context_cache):
    """Entries must exist and match each table identity; no chunk is read here."""
    from ..harness.context_cache import ContextCache
    report = {"directory": str(context_cache), "brains": {}}
    try:
        cache = ContextCache(context_cache)
    except (FileNotFoundError, ValueError) as exc:
        report.update(status="blocked", reason=str(exc))
        return report
    blocked = False
    for brain, bank in banks.items():
        cells = {}
        for kind, table in (getattr(bank, "tables", None) or {}).items():
            try:
                entry = cache.entry(table)
                cells[kind] = {"status": "passed", "band_rows": int(len(entry.rows)),
                               "image_tiers": entry.tiers()}
            except (FileNotFoundError, ValueError) as exc:
                cells[kind] = {"status": "blocked", "reason": str(exc)}
                blocked = True
        report["brains"][brain] = cells
    report["status"] = "blocked" if blocked or not banks else "passed"
    return report


def inspect_readiness(brains, mcl=100, tables_dir=pc.DEFAULT_OUT,
                      merge_dirs=None, split_dirs=None, check_api=False, model=DEFAULT_MODEL,
                      train_brains=None, selection_brains=None, selection_protocol="grouped_oof",
                      context_cache=None):
    report = {"host": socket.gethostname(), "mcl": mcl, "brains": {},
              "scope": "No brain-cache deserialization, table preparation, scoring or evolution",
              "api": {"status": "not_checked", "key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
                      "model": model},
              "warnings": ["Native mode evolves scores on fixed candidates, not graph edits or candidate expansion.",
                           "Raw brain-cache contents and full reviser SDK/tool execution are not checked here."]}
    try:
        selected = pc.resolve_detector_runs(merge_dirs, split_dirs)
    except (Exception, SystemExit) as exc:
        report.update(status="blocked", detector_error=str(exc))
        return report
    report["detectors"] = selected
    loaded = {}
    for brain in brains:
        brain = str(brain)
        try:
            bank = ensure_native_tables(brain, native_cache_path(brain, mcl), selected,
                                         tables_dir, prepare=False, mcl=mcl)
            loaded[brain] = bank
            tables = getattr(bank, "tables", None) or {}
            report["brains"][brain] = {
                "status": "passed", "table_paths": bank.meta["table_paths"],
                "detector_training_brains": sorted({str(t.meta["training_brain"]) for t in tables.values()
                                                    if getattr(t, "meta", {}).get("training_brain") is not None})}
        except (Exception, SystemExit) as exc:
            report["brains"][brain] = {"status": "blocked", "reason": str(exc)}
    if check_api:
        report["api"] = api_probe(model)
    report["selection"] = selection_readiness(report, train_brains, selection_brains, selection_protocol)
    report["context_cache"] = (context_cache_readiness(loaded, context_cache) if context_cache else None)
    blocked = (not brains or any(v["status"] != "passed" for v in report["brains"].values())
               or (check_api and report["api"]["status"] != "passed")
               or (report["selection"] is not None and report["selection"]["status"] != "passed")
               or (report["context_cache"] is not None and report["context_cache"]["status"] != "passed"))
    report["status"] = "blocked" if blocked else "checks_passed"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brains", nargs="+", required=True)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--feature-tables-dir", default=str(pc.DEFAULT_OUT))
    parser.add_argument("--merge-dir", action="append")
    parser.add_argument("--split-dir", action="append")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument("--train-brains", nargs="*", default=None,
                        help="TRAIN brains whose roles to check (must be among --brains)")
    parser.add_argument("--selection-brains", nargs="*", default=None,
                        help="Explicit selection brains (default: TRAIN brains the detectors were not fitted on)")
    parser.add_argument("--selection-protocol", choices=("grouped_oof", "in_sample"), default="grouped_oof")
    parser.add_argument("--context-cache", default=None,
                        help="Verify that every brain/kind has a complete context cache entry in this directory")
    args = parser.parse_args(argv)
    report = inspect_readiness(args.brains, args.mcl, args.feature_tables_dir,
                               args.merge_dir, args.split_dir, args.check_api, args.model,
                               train_brains=args.train_brains, selection_brains=args.selection_brains,
                               selection_protocol=args.selection_protocol, context_cache=args.context_cache)
    print(json.dumps(report, indent=2))
    return int(report["status"] != "checks_passed")


if __name__ == "__main__":
    raise SystemExit(main())
