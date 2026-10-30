#!/usr/bin/env bash
# Rebuild the derived Stage 3 artifacts on this node, one brain at a time (rebuild plan,
# section 4): detector-native tables of both kinds with their availability sidecars, the
# companion pool check (794495 and 802449); then one preflight over this call.
# Evolution reads geometry and images on demand; no context patches are precomputed.
#
# Run it inside the allocation (a terminal of the interactive node's JupyterLab); the steps use the CPUs
# this process may run on. Every step reuses finished work (tables per brain and kind, valid
# sidecars), so the same command after an interruption resumes at the
# first unfinished brain and kind; a kind interrupted mid-way is built again from its start.
# The first failure stops the script. Never run two of these on one node: 200 GB held one brain
# (802449 in the detector build), two brains together are untested. Calls on different nodes with
# different brains are independent (per-brain directories, per-call logs). The TRAIN/selection
# role check of the closing preflight runs only when the call holds both 794495 and 802449.
#
#   bash proofreader_evolve/rebuild_brains.sh [brain ...]          # default: all eleven brains
#   nohup bash proofreader_evolve/rebuild_brains.sh >/dev/null 2>&1 &   # keeps running when the terminal closes
#
# Environment: IMAGE_WORKERS (detector image pass, default -1 = one per CPU).
# The console output is mirrored to proofreader_evolve/log/rebuild_brains_<timestamp>_<pid>.log.txt;
# each finished brain prints "# OK brain <b>", the end "# OK" or "# FAILED".

source /shared/utils.x86_64/anaconda3-2024.10/etc/profile.d/conda.sh
conda activate panda
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
# Evolution brains first (TRAIN 794495 802449, validation 789202 794493 794491), then the
# older-microscope brains, shortest image pass first.
DEFAULT_BRAINS=(794495 802449 789202 794493 794491 754613 754612 754610 750318 751473 747807)
COMPANION_BRAINS=" 794495 802449 "
MCL=100
IMAGE_WORKERS=${IMAGE_WORKERS:--1}
if [ "$#" -eq 0 ]; then
    set -- "${DEFAULT_BRAINS[@]}"
fi

LOG=proofreader_evolve/log/rebuild_brains_$(date +%Y%m%d_%H%M%S)_$$.log.txt
mkdir -p proofreader_evolve/log
exec > >(tee -a "$LOG") 2>&1
finish() {
    local status=$?
    if [ "$status" -eq 0 ]; then
        echo "# OK after ${SECONDS}s (ended $(date '+%F %T'))"
    else
        echo "# FAILED (exit code $status) after ${SECONDS}s (ended $(date '+%F %T')); rerun the same command to resume"
    fi
}
trap finish EXIT
echo "# argv: $0 $*"
echo "# started $(date '+%F %T')  host $(hostname)  CPUs $(nproc)  python $(command -v python)"
echo "# image workers $IMAGE_WORKERS  log $LOG"
export AWS_EC2_METADATA_DISABLED=true

check_companion() {
    # The tables' rows must be the companions' rows (pool_sha256 hashes the rows only).
    python - "$1" "$MCL" <<'PY'
import json, sys
from proofreader_evolve.cli import precompute_error_scores as pc
from proofreader_evolve.harness.native_pool import cache_path, ensure_native_tables
brain, mcl = sys.argv[1], int(sys.argv[2])
bank = ensure_native_tables(brain, cache_path(brain, mcl), pc.resolve_detector_runs(), pc.DEFAULT_OUT, mcl=mcl)
matches = []
for kind, table in sorted(bank.tables.items()):
    with open(f"markdowns/rerank_companion_{brain}_{kind}.json") as stream:
        expected = json.load(stream)["pool_sha256"]
    matches.append(table.meta["pool_sha256"] == expected)
    print(f"[companion] {brain}/{kind} pool matches the companion: {matches[-1]}", flush=True)
sys.exit(0 if all(matches) else 1)
PY
}

for b in "$@"; do
    echo; echo "# brain $b: tables and sidecars ($(date '+%F %T'))"
    python -u proofreader_evolve/prepare_feature_tables.py --brains "$b" --mcl "$MCL" \
        --threads all --image-workers "$IMAGE_WORKERS"
    if [[ "$COMPANION_BRAINS" == *" $b "* ]]; then
        echo "# brain $b: companion pool check"
        check_companion "$b"
    fi
    echo "# OK brain $b ($(date '+%F %T'))"
done

roles=()
if [[ " $* " == *" 794495 "* && " $* " == *" 802449 "* ]]; then
    roles=(--train-brains 794495 802449)
fi
echo; echo "# preflight"
python -u -m proofreader_evolve.cli.preflight --brains "$@" --mcl "$MCL" "${roles[@]}"
