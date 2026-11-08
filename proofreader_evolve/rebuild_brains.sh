#!/usr/bin/env bash
# Rebuild the derived Stage 3 artifacts on this node, one brain at a time (rebuild plan,
# section 4): candidate pools of both kinds with their availability sidecars, the
# companion pool check (802449; 794495 only when passed explicitly); then one preflight over this call.
# DETECTOR_PRIOR=raw (default since 2026-11-06) builds detector-free raw pools (minutes per
# brain, no detector, no image pass; what --detector-prior raw evolution loads).
# DETECTOR_PRIOR=frozen builds the detector-native tables with the default frozen detectors
# (hours per brain), needed only for frozen / no-score runs; every brain's native tables were
# deleted on 2026-11-07, so frozen mode starts from nothing.
# Evolution reads geometry and images on demand; no context patches are precomputed.
#
# Run it inside the allocation (a terminal of the interactive node's JupyterLab); the steps use the CPUs
# this process may run on. Every step reuses finished work (tables per brain and kind, valid
# sidecars), so the same command after an interruption resumes at the
# first unfinished brain and kind; a kind interrupted mid-way is built again from its start.
# The first failure stops the script. Never run two of these on one node: 200 GB held one brain
# (802449 in the detector build), two brains together are untested. Calls on different nodes with
# different brains are independent (per-brain directories, per-call logs). The closing preflight
# receives the current TRAIN brains among the call's brains as --train-brains (selection all).
#
# 2026-11-05: the segmentation sources of 789202 and 802449 were replaced by the denoised Google
# segmentations and 841260 was added (explicit entries in configs/segmentation_paths.json). Their
# caches (2026-11-06), raw pools and 802449 companions (2026-11-06/07) and alignment receipts
# (2026-11-07) are rebuilt. After any future graph reload repeat that order first: rebuild the cache,
# regenerate the brain's companions if it has any (otherwise the companion pool check below fails
# against the old file), run this script, then re-record the receipt (docs/dataset_artifact_dependencies.md).
#
#   bash proofreader_evolve/rebuild_brains.sh [brain ...]          # default: the eleven development brains, 841260 last
#   nohup bash proofreader_evolve/rebuild_brains.sh >/dev/null 2>&1 &   # keeps running when the terminal closes
#
# Environment: DETECTOR_PRIOR (raw | frozen, default raw); IMAGE_WORKERS (frozen only: detector
# image pass, default -1 = one per CPU).
# The console output is mirrored to proofreader_evolve/log/rebuild_brains_<timestamp>_<pid>.log.txt;
# each finished brain prints "# OK brain <b>", the end "# OK" or "# FAILED".

source /shared/utils.x86_64/anaconda3-2024.10/etc/profile.d/conda.sh
conda activate panda
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
# Current roles (2026-11-04): TRAIN/selection 802449 747807 750318 754610 754612 754613, validation
# 789202 794493 794491 751473 841260 (841260 added 2026-11-05, validation since 2026-11-07); 794495 is
# reserved (no development role, so not in the default list; pass it explicitly if its pools are ever
# needed). Evolution brains first, then the older-microscope brains, shortest image pass first.
DEFAULT_BRAINS=(802449 789202 794493 794491 754613 754612 754610 750318 751473 747807 841260)
TRAIN_BRAINS=" 802449 747807 750318 754610 754612 754613 "
COMPANION_BRAINS=" 794495 802449 "
MCL=100
IMAGE_WORKERS=${IMAGE_WORKERS:--1}
DETECTOR_PRIOR=${DETECTOR_PRIOR:-raw}
case "$DETECTOR_PRIOR" in raw|frozen) ;; *) echo "DETECTOR_PRIOR must be raw or frozen" >&2; exit 2 ;; esac
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
echo "# detector prior $DETECTOR_PRIOR  image workers $IMAGE_WORKERS  log $LOG"
export AWS_EC2_METADATA_DISABLED=true

check_companion() {
    # The tables' rows must be the companions' rows (pool_sha256 hashes the rows only).
    python - "$1" "$MCL" "$DETECTOR_PRIOR" <<'PY'
import json, sys
from proofreader_evolve.cli import precompute_error_scores as pc
from proofreader_evolve.harness.native_pool import cache_path, ensure_native_tables
from proofreader_evolve.harness.raw_pool import ensure_raw_tables
brain, mcl, prior = sys.argv[1], int(sys.argv[2]), sys.argv[3]
source = cache_path(brain, mcl)
bank = (ensure_raw_tables(brain, source, pc.DEFAULT_OUT, mcl=mcl) if prior == "raw"
        else ensure_native_tables(brain, source, pc.resolve_detector_runs(), pc.DEFAULT_OUT, mcl=mcl))
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
        --threads all --image-workers "$IMAGE_WORKERS" --detector-prior "$DETECTOR_PRIOR"
    if [[ "$COMPANION_BRAINS" == *" $b "* ]]; then
        echo "# brain $b: companion pool check"
        check_companion "$b"
    fi
    echo "# OK brain $b ($(date '+%F %T'))"
done

roles=()
train_present=()
for b in "$@"; do
    if [[ "$TRAIN_BRAINS" == *" $b "* ]]; then train_present+=("$b"); fi
done
if [ "${#train_present[@]}" -gt 0 ]; then
    roles=(--train-brains "${train_present[@]}" --selection-brains all)
fi
echo; echo "# preflight"
python -u -m proofreader_evolve.cli.preflight --brains "$@" --mcl "$MCL" --detector-prior "$DETECTOR_PRIOR" "${roles[@]}"
