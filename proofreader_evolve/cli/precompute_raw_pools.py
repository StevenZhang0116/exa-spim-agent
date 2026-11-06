"""Prepare raw candidate pools (no detector) from trusted labeled _add.pkl caches.

``--detector-prior raw`` evolution loads these pools instead of the detector-native tables
(2026-11-06): the declared candidate rule enumerated by candidate_rules.py, binary evaluator
labels and the label-blind raw candidate attributes as the only feature frame
(harness/raw_pool.py). No frozen detector is resolved, imported or run, so one brain takes a
cache load plus the enumeration. Run in panda on a compute node (the cache needs more than
20 GB). Existing matching pools are reused; corrupt or mismatched pools are never silently
replaced. prepare_feature_tables.py --detector-prior raw runs this per brain, then the
availability sidecars and the preflight.

    python -u -m proofreader_evolve.cli.precompute_raw_pools --brain 802449 --mcl 100 \\
        --pkl cache/dataset_cache_802449_mcl100_add.pkl
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from proofreader_evolve.cli.precompute_error_scores import DEFAULT_OUT, log  # noqa: E402
from proofreader_evolve.harness.raw_pool import ensure_raw_tables  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--pkl", required=True, help="Matching trusted labeled _add.pkl cache")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    bank = ensure_raw_tables(args.brain, args.pkl, args.out_dir, prepare=True, mcl=args.mcl)
    log(json.dumps(bank.meta, indent=2))
    log("# OK (raw candidate pool)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
