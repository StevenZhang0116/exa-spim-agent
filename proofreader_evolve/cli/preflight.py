"""Read-only launch checks; no brain loading, table building or evolution.

Optional --check-api sends one tiny tool-free request to the configured model through the
reviser's own SDK path (bundled Claude Code CLI, API env, model id); it does not exercise the
tool workflow. Run `precision_20261009_235526_ispymdwn` lost every generation because the
bundled CLI was too old for the model, which a direct API request cannot detect.
Contexts are read on demand during evolution. These offline checks do not exercise
fragment loading or cloud image access. Under --detector-prior raw the checked tables are
the detector-free raw pools (2026-11-06); no detector is resolved and --merge-dir/--split-dir
are ignored. The label counts also give the GT-positive floor check (2026-11-08): pools below
--min-positives are listed, since a run scores and reports them but leaves them out of the
objective, the bootstrap and kind headroom, and a TRAIN kind without a counted pool blocks,
as the driver refuses it.
"""

import argparse
import json
import os
import socket

from . import precompute_error_scores as pc
from ..harness.reviser_session import DEFAULT_MODEL
from ..harness.native_pool import ensure_native_tables, cache_path as native_cache_path
from ..harness.raw_pool import ensure_raw_tables
from ..harness.fixed_pool_scoring import DEFAULT_MIN_POSITIVES
from ..harness import label_availability


def api_probe(model, timeout_s=120):
    """One-turn, tool-free request through the reviser's SDK path (bundled CLI + model)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"status": "missing_key", "model": model}
    import asyncio
    try:
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
        try:
            from claude_agent_sdk._cli_version import __cli_version__ as cli_version
        except ImportError:  # pragma: no cover - older SDK layout
            cli_version = None
        from ..harness.reviser_session import anthropic_api_env
        options = ClaudeAgentOptions(model=model, env=anthropic_api_env(), max_turns=1, tools=[],
                                     allowed_tools=[], setting_sources=[], skills=[], strict_mcp_config=True,
                                     permission_mode="default", system_prompt="Answer with the single word OK.")

        async def run():
            async for message in query(prompt="Reply OK.", options=options):
                if isinstance(message, ResultMessage):
                    return message
            raise RuntimeError("SDK session ended without a result message")

        result = asyncio.run(asyncio.wait_for(run(), timeout=timeout_s))
        report = {"model": model, "cli_version": cli_version, "subtype": result.subtype}
        if result.is_error or result.subtype != "success":
            # The CLI's error text (e.g. an unsupported model) is the actionable part; no headers/bodies.
            return {"status": "failed", **report, "detail": str(result.result)[:300]}
        return {"status": "passed", **report, "models_used": sorted((result.model_usage or {}).keys()),
                "cost_usd": result.total_cost_usd}
    except Exception as exc:
        return {"status": "failed", "model": model, "error_type": type(exc).__name__, "detail": str(exc)[:300]}


def selection_readiness(report, train_brains, selection_brains, protocol, detector_prior="frozen"):
    """Brain-role check for the selection protocol from table provenance only.

    The fragment graph needed for split spatial folds on a selection brain is
    loaded at run start, not here (it requires the full brain cache).
    """
    train_brains = [str(b) for b in train_brains or []]
    if not train_brains:
        return None
    # Under --detector-prior raw no detector-derived input remains, so fitted brains may be selection
    # brains, and raw pools (2026-11-06) carry no training provenance to check.
    withheld = detector_prior == "raw"
    fitted, missing = set(), []
    for brain in train_brains:
        entry = report["brains"].get(brain, {})
        provenance = entry.get("detector_training_brains")
        if entry.get("status") == "passed" and not provenance and not withheld:
            missing.append(brain)
        fitted.update(provenance or [])
    if protocol != "grouped_oof" or list(selection_brains or []) == ["all"] or (not selection_brains and withheld):
        chosen = list(train_brains)
    elif selection_brains:
        chosen = [str(b) for b in selection_brains]
    else:
        chosen = [b for b in train_brains if b not in fitted]
    conflicts = sorted(set(chosen) & fitted) if protocol == "grouped_oof" and not withheld else []
    outside = sorted(set(chosen) - set(train_brains))
    blocked = protocol == "grouped_oof" and (not chosen or conflicts or missing or outside)
    return {"protocol": protocol, "train_brains": train_brains, "detector_training_brains": sorted(fitted),
            "selection_brains": chosen, "auxiliary_train_brains": [b for b in train_brains if b not in chosen],
            "missing_provenance": missing, "conflicts": conflicts, "not_train": outside,
            "status": "blocked" if blocked else "passed", "detector_prior": detector_prior,
            "note": ("Every detector-derived input is withheld under --detector-prior raw, so detector training "
                     "brains may be selection brains" if withheld else
                     "Selection brains must not be detector training brains") +
                    "; fragment-graph availability for split folds is verified when the run starts"}


def positive_floor(report, min_positives, train_brains=None):
    """Pools below the GT-positive floor, from the label counts already in the report.

    Mirrors the driver's ``Budgets.in_mean``: a pool with fewer than ``min_positives`` label-1
    rows is scored and reported by a run but excluded from the objective, the paired bootstrap
    and kind headroom. The driver refuses to start when every pool of a kind in a role is below
    the floor; TRAIN is checked here (validation brains are not named to preflight).
    """
    positives = {}
    for entry in report["brains"].values():
        for cell, counts in (entry.get("label_counts") or {}).items():
            positives[cell] = int(counts.get("1", 0))
    below = {cell: n for cell, n in sorted(positives.items()) if n < min_positives}
    train = [str(b) for b in train_brains or []]
    uncounted_kinds = []
    for kind in ("merge", "split"):
        cells = [f"{b}/{kind}" for b in train if f"{b}/{kind}" in positives]
        if cells and all(cell in below for cell in cells):
            uncounted_kinds.append(kind)
    return {"min_positives": min_positives, "positives": positives, "below_floor": below,
            "train_kinds_without_counted_pool": uncounted_kinds,
            "status": "blocked" if uncounted_kinds else "passed",
            "rule": (f"pools with fewer than {min_positives} GT positives are scored and reported but excluded "
                     "from the objective, the bootstrap and kind headroom (--min-positives of the driver)"
                     if min_positives > 0 else "every pool counts toward the objective (--min-positives 0)")}


def inspect_readiness(brains, mcl=100, tables_dir=pc.DEFAULT_OUT,
                      merge_dirs=None, split_dirs=None, check_api=False, model=DEFAULT_MODEL,
                      train_brains=None, selection_brains=None, selection_protocol="grouped_oof",
                      detector_prior="frozen", min_positives=DEFAULT_MIN_POSITIVES):
    report = {"host": socket.gethostname(), "mcl": mcl, "brains": {},
              "scope": "No brain-cache deserialization, table preparation, scoring or evolution",
              "api": {"status": "not_checked", "key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
                      "model": model},
              "warnings": ["Native mode evolves scores on fixed candidates, not graph edits or candidate expansion.",
                           "Raw brain-cache contents and the reviser tool workflow are not checked here; --check-api covers the SDK session path only."]}
    if detector_prior == "raw":
        # Raw pools carry no detector (2026-11-06): nothing to resolve.
        selected = None
        report["detectors"] = None
        report["candidate_pools"] = "raw (detector-free pools; harness/raw_pool.py)"
    else:
        try:
            selected = pc.resolve_detector_runs(merge_dirs, split_dirs)
        except (Exception, SystemExit) as exc:
            report.update(status="blocked", detector_error=str(exc))
            return report
        report["detectors"] = selected
        report["candidate_pools"] = "detector-native tables"
    for brain in brains:
        brain = str(brain)
        try:
            source = native_cache_path(brain, mcl)
            bank = (ensure_raw_tables(brain, source, tables_dir, prepare=False, mcl=mcl) if selected is None
                    else ensure_native_tables(brain, source, selected, tables_dir, prepare=False, mcl=mcl))
            tables = getattr(bank, "tables", None) or {}
            # Evolution requires the three-valued labels; a missing/mismatched sidecar blocks.
            labels = label_availability.apply({brain: bank}) if tables else {}
            report["brains"][brain] = {
                "status": "passed", "table_paths": bank.meta["table_paths"],
                "label_counts": {cell: info["counts"] for cell, info in labels.items()},
                "detector_training_brains": sorted({str(t.meta["training_brain"]) for t in tables.values()
                                                    if getattr(t, "meta", {}).get("training_brain") is not None})}
        except (Exception, SystemExit) as exc:
            report["brains"][brain] = {"status": "blocked", "reason": str(exc)}
    if check_api:
        report["api"] = api_probe(model)
    report["selection"] = selection_readiness(report, train_brains, selection_brains, selection_protocol,
                                              detector_prior)
    report["descriptor_context"] = {"mode": "on_demand", "precomputed_context_required": False,
                                    "source_access": "not_exercised"}
    report["positive_floor"] = positive_floor(report, min_positives, train_brains)
    below = report["positive_floor"]["below_floor"]
    if below:
        report["warnings"].append(
            f"{len(below)} pool(s) below the GT-positive floor ({min_positives}) are scored and reported but "
            "excluded from the objective, the bootstrap and kind headroom: "
            + ", ".join(f"{cell} ({n})" for cell, n in below.items()))
    blocked = (not brains or any(v["status"] != "passed" for v in report["brains"].values())
               or (check_api and report["api"]["status"] != "passed")
               or (report["selection"] is not None and report["selection"]["status"] != "passed")
               or report["positive_floor"]["status"] != "passed")
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
                        help="Explicit selection brains, or 'all' for every TRAIN brain (default: every TRAIN "
                             "brain under --detector-prior raw; otherwise TRAIN brains the detectors were "
                             "not fitted on). Detector-fitted selection brains require raw")
    parser.add_argument("--selection-protocol", choices=("grouped_oof", "in_sample"), default="grouped_oof")
    parser.add_argument("--detector-prior", choices=("frozen", "no-score", "raw"), default="frozen",
                        help="raw withholds every detector-derived input, so detector training brains may be "
                             "selection brains (--selection-brains all)")
    parser.add_argument("--min-positives", type=int, default=DEFAULT_MIN_POSITIVES,
                        help="GT-positive floor of the run to list the pools it will not count; must match the "
                             f"driver's --min-positives (default: {DEFAULT_MIN_POSITIVES})")
    args = parser.parse_args(argv)
    if args.min_positives < 0:
        parser.error("min-positives must be >= 0 (0 counts every pool)")
    report = inspect_readiness(args.brains, args.mcl, args.feature_tables_dir,
                               args.merge_dir, args.split_dir, args.check_api, args.model,
                               train_brains=args.train_brains, selection_brains=args.selection_brains,
                               selection_protocol=args.selection_protocol, detector_prior=args.detector_prior,
                               min_positives=args.min_positives)
    print(json.dumps(report, indent=2))
    return int(report["status"] != "checks_passed")


if __name__ == "__main__":
    raise SystemExit(main())
