"""Three-valued label counts (1 / 0 / NaN) per brain and kind, straight from the labeled caches.

No feature tables or sidecars are needed: the candidate rows and labels come from
``proofreader_evolve.cli.export_rerank_companion.pool_only_bank``, the same code path as the
pool-only companion export (frozen detector enumeration with the pool declared in
``proofreader_evolve/candidate_pools.json``; availability from ``label_availability``: merge
``near_gt_150``, split ``segment_on_gt``). Counts therefore match the companions and the
tables built later from the same caches.

Output: ``notebooks/label_counts/<brain>_mcl<N>.json`` (one per brain; existing files are kept
unless --force), read by ``notebooks/plot_label_availability.py``. Where a companion of the
same pool exists in ``markdowns/``, its label counts are compared and the result is printed.

Run with panda on a compute node, one process per group of brains (each loads one cache;
the split enumeration of the largest brains takes 15-25 min):
    python -u notebooks/label_counts.py --brains 794495 802449
    python -u notebooks/label_counts.py --dry-run          # which brains would be computed
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALL_BRAINS = ["794495", "802449", "789202", "794493", "794491", "747807", "750318",
              "751473", "754610", "754612", "754613"]
OUT_DIR = ROOT / "notebooks" / "label_counts"
COMPANION_DIR = ROOT / "markdowns"
KINDS = ("merge", "split")


def count_brain(brain: str, mcl: int) -> dict:
    from proofreader_evolve.cli import export_rerank_companion as ex
    from proofreader_evolve.harness import label_availability as la

    started = time.monotonic()
    bank, _graph = ex.pool_only_bank(brain, mcl, list(KINDS))
    kinds = {}
    for kind in KINDS:
        table = bank.tables[kind]
        counts = la.label_counts(table.truth)
        kinds[kind] = {"rows": int(len(table.truth)), **counts,
                       "definition": table.availability["definition"],
                       "candidate_policy": table.meta.get("candidate_policy"),
                       "candidate_policy_source": table.meta.get("candidate_policy_source"),
                       "pool_sha256": table.meta["pool_sha256"],
                       "labels_sha256": la.labels_digest(table.truth)}
    source = bank.tables[KINDS[0]].meta["provenance"]["source_cache"]
    del bank, _graph
    gc.collect()
    return {"brain": brain, "mcl": mcl, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_cache": source, "kinds": kinds, "seconds": round(time.monotonic() - started, 1)}


def compare_with_companions(result: dict) -> None:
    for kind, record in result["kinds"].items():
        path = COMPANION_DIR / f"rerank_companion_{result['brain']}_{kind}.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        if manifest.get("pool_sha256") != record["pool_sha256"]:
            print(f"  {kind}: companion pool differs (companion built for another pool)", flush=True)
            continue
        same = all(int(manifest["label_counts"][k]) == record[k] for k in ("1", "0", "nan"))
        print(f"  {kind}: companion labels {'match' if same else 'differ (companion predates the current labels)'}"
              f" ({manifest['label_counts']} vs 1={record['1']} 0={record['0']} nan={record['nan']})", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", default=ALL_BRAINS)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true", help="Recompute brains that already have a JSON")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    todo = [b for b in dict.fromkeys(map(str, args.brains))
            if args.force or not (args.out_dir / f"{b}_mcl{args.mcl}.json").exists()]
    print(f"brains to compute: {todo or 'none'} (output {args.out_dir})", flush=True)
    if args.dry_run:
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for brain in todo:
        result = count_brain(brain, args.mcl)
        path = args.out_dir / f"{brain}_mcl{args.mcl}.json"
        path.write_text(json.dumps(result, indent=1))
        line = " | ".join(f"{k}: rows {v['rows']:,}, 1 {v['1']:,}, 0 {v['0']:,}, NaN {v['nan']:,} ({v['candidate_policy']})"
                          for k, v in result["kinds"].items())
        print(f"[{brain}] {line} | {result['seconds']:.0f}s -> {path}", flush=True)
        compare_with_companions(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
