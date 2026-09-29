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


def inspect_readiness(brains, mcl=100, tables_dir=pc.DEFAULT_OUT,
                      merge_dirs=None, split_dirs=None, check_api=False, model=DEFAULT_MODEL):
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
    for brain in brains:
        brain = str(brain)
        try:
            bank = ensure_native_tables(brain, native_cache_path(brain, mcl), selected,
                                         tables_dir, prepare=False, mcl=mcl)
            report["brains"][brain] = {"status": "passed", "table_paths": bank.meta["table_paths"]}
        except (Exception, SystemExit) as exc:
            report["brains"][brain] = {"status": "blocked", "reason": str(exc)}
    if check_api:
        report["api"] = api_probe(model)
    blocked = (not brains or any(v["status"] != "passed" for v in report["brains"].values())
               or (check_api and report["api"]["status"] != "passed"))
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
    args = parser.parse_args(argv)
    report = inspect_readiness(args.brains, args.mcl, args.feature_tables_dir,
                               args.merge_dir, args.split_dir, args.check_api, args.model)
    print(json.dumps(report, indent=2))
    return int(report["status"] != "checks_passed")


if __name__ == "__main__":
    raise SystemExit(main())
