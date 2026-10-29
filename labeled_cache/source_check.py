"""Check that a brain's fragment SWCs and dense segmentation come from the same run.

The builder consumes the two sources at different stages: the base stage loads the
fragment skeletons (``fragments_path``), the label stage reads the dense segmentation
(``segmentation_path``) at every GT node. Nothing in the build compares the two, and a
mismatched pair (a re-exported segmentation with old skeletons, a wrong folder in
``configs/segmentation_paths.json``) would only show up later as near-zero GT label
coverage. This check samples interior fragment nodes, converts them to voxels exactly as
the builder does (``swc_util.Reader`` with the cache anisotropy, then
``img_util.to_voxels``), reads the segmentation at those voxels through the same
``TensorStoreImage`` reader and compares the voxel's segment id with the fragment's own id
(the SWC name). Matched sources give close to 100% agreement (resampled skeleton nodes sit
within a voxel of the centre line); an unrelated segmentation gives close to 0.

    python -m labeled_cache.source_check --brains 794495 802449
    python -m labeled_cache.source_check --brains 751473 --record docs/fragment_segmentation_match.md

Reads a few SWC zip shards and a few hundred 3x3x3 voxel patches per brain (seconds,
GCS credentials needed); no cache is loaded, so it runs on a login node.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import io as stringio
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import zipfile

import numpy as np

from labeled_cache import io

DEFAULT_ANISOTROPY = (0.748, 0.748, 1.0)
DEFAULT_RECORD = io.PROJECT_ROOT / "docs" / "fragment_segmentation_match.md"
PATCH = 3  # odd patch edge: the centre voxel is the node, the rest absorbs 1-voxel resampling offsets


# --- pure parts (unit tested) --------------------------------------------------------------
def fragment_id(swc_name):
    """Integer segment id of an SWC name (``"123"``, ``"123.swc"``, ``"123.1"``), or None."""
    try:
        return int(str(swc_name).split(".")[0])
    except ValueError:
        return None


def sample_fragment_nodes(fragments, nodes_per_fragment, min_nodes, anisotropy, rng):
    """``[(segment id, (z, y, x) voxel)]`` over interior nodes of the parsed fragments.

    ``fragments`` are ``swc_util.Reader.parse`` dictionaries (``swc_name``, ``xyz`` in
    microns). The two nodes at either end are skipped (tips are the least reliable voxels)
    and fragments shorter than ``min_nodes`` or without an integer name are ignored.
    """
    from agentic_neuron_proofreader.utils import img_util

    samples = []
    for fragment in fragments:
        segment = fragment_id(fragment.get("swc_name"))
        xyz = np.asarray(fragment["xyz"], dtype=np.float64)
        if segment is None or len(xyz) < max(int(min_nodes), 5):
            continue
        interior = range(2, len(xyz) - 2)
        for index in sorted(rng.sample(interior, min(int(nodes_per_fragment), len(interior)))):
            samples.append((segment, tuple(int(v) for v in img_util.to_voxels(xyz[index], anisotropy))))
    return samples


def tally(samples, read_patch):
    """Agreement counts of ``samples`` against ``read_patch(voxel) -> (PATCH, PATCH, PATCH)`` ids."""
    counts = {"nodes": len(samples), "exact_match": 0, "match_within_1vox": 0,
              "background": 0, "other_id": 0, "read_failed": 0}
    other_ids = {}
    centre = PATCH // 2
    for segment, voxel in samples:
        try:
            patch = np.asarray(read_patch(voxel)).reshape(PATCH, PATCH, PATCH)
        except Exception:  # noqa: BLE001 - out of bounds or a transient read error
            counts["read_failed"] += 1
            continue
        at_node = int(patch[centre, centre, centre])
        if at_node == segment:
            counts["exact_match"] += 1
        elif (patch == segment).any():
            counts["match_within_1vox"] += 1
        elif at_node == 0:
            counts["background"] += 1
        else:
            counts["other_id"] += 1
            other_ids[at_node] = other_ids.get(at_node, 0) + 1
    read = counts["nodes"] - counts["read_failed"]
    counts["agreement_pct"] = round(100.0 * (counts["exact_match"] + counts["match_within_1vox"]) / read, 1) if read else 0.0
    counts["top_other_ids"] = sorted(other_ids.items(), key=lambda item: -item[1])[:3]
    return counts


def verdict(report, min_agreement_pct, min_nodes):
    """``"matched"``, ``"MISMATCH"`` or ``"insufficient"`` for one brain's report."""
    if "error" in report:
        return "error"
    read = report["nodes"] - report["read_failed"]
    if read < int(min_nodes):
        return "insufficient"
    return "matched" if report["agreement_pct"] >= float(min_agreement_pct) else "MISMATCH"


def markdown_table(reports):
    """One Markdown row per brain, in the given order."""
    lines = ["| brain | segmentation_id | fragments | nodes | exact | within 1 vox | background | other id | failed | agreement | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for brain, report in reports.items():
        if "error" in report:
            lines.append(f"| {brain} | {report.get('segmentation_id', '')} | | | | | | | | | error: {report['error']} |")
            continue
        lines.append(f"| {brain} | `{report['segmentation_id']}` | {report['fragments']} | {report['nodes']} | "
                     f"{report['exact_match']} | {report['match_within_1vox']} | {report['background']} | "
                     f"{report['other_id']} | {report['read_failed']} | {report['agreement_pct']}% | {report['verdict']} |")
    return "\n".join(lines)


# --- cloud parts -----------------------------------------------------------------------------
def segmentation_id_of(sources):
    """The segmentation id of the resolved sources (the folder before ``label_mask`` or the legacy id)."""
    parts = [p for p in sources["segmentation_path"].rstrip("/").split("/") if p]
    if parts and parts[-1] == "label_mask":
        parts = parts[:-1]
    return parts[-1] if parts else ""


def read_fragments(fragments_path, n_zips, rng, anisotropy, max_fragments):
    """Parsed fragments from ``n_zips`` random zip shards (or loose SWCs) under ``fragments_path``."""
    from google.cloud import storage
    from agentic_neuron_proofreader.utils import swc_util

    bucket_name, prefix = fragments_path.replace("gs://", "", 1).split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix, max_results=50000))
    zips = [b for b in blobs if b.name.endswith(".zip")]
    swcs = [b for b in blobs if b.name.endswith(".swc")]
    reader = swc_util.Reader(anisotropy=tuple(anisotropy), verbose=False)
    fragments = []
    if zips:
        chosen = rng.sample(zips, min(int(n_zips), len(zips)))
        for blob in chosen:
            archive = zipfile.ZipFile(stringio.BytesIO(blob.download_as_bytes()))
            names = [n for n in archive.namelist() if n.endswith(".swc")]
            rng.shuffle(names)
            for name in names:
                parsed = reader.parse(archive.read(name).decode().splitlines(), os.path.basename(name))
                if parsed is not None:
                    fragments.append(parsed)
                if len(fragments) >= int(max_fragments):
                    break
        layout = f"{len(zips)} zip shards, {len(chosen)} read"
    elif swcs:
        for blob in rng.sample(swcs, min(int(max_fragments), len(swcs))):
            parsed = reader.parse(blob.download_as_text().splitlines(), os.path.basename(blob.name))
            if parsed is not None:
                fragments.append(parsed)
        layout = f"{len(swcs)} loose SWC files"
    else:
        raise FileNotFoundError(f"no .zip or .swc objects under {fragments_path}")
    rng.shuffle(fragments)
    return fragments, layout


def check_brain(brain, fragments=60, nodes_per_fragment=4, zips=6, min_fragment_nodes=30,
                anisotropy=DEFAULT_ANISOTROPY, seed=0, min_agreement_pct=95.0, min_read_nodes=100):
    """Agreement report of one brain (``verdict`` included); never raises on a source error."""
    from agentic_neuron_proofreader.utils import img_util

    io.set_cloud_environment()
    started = time.monotonic()
    sources = io.dataset_sources(brain)
    report = {"segmentation_id": segmentation_id_of(sources), "segmentation_path": sources["segmentation_path"],
              "fragments_path": sources["fragments_path"]}
    rng = random.Random(f"{seed}:{brain}")
    try:
        parsed, layout = read_fragments(sources["fragments_path"], zips, rng, anisotropy, 3 * int(fragments))
        usable = [f for f in parsed if fragment_id(f.get("swc_name")) is not None and len(f["xyz"]) >= int(min_fragment_nodes)]
        chosen = usable[:int(fragments)]
        samples = sample_fragment_nodes(chosen, nodes_per_fragment, min_fragment_nodes, anisotropy, rng)
        segmentation = img_util.TensorStoreImage(sources["segmentation_path"])
    except Exception as exc:  # noqa: BLE001 - reported per brain, the other brains continue
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["verdict"] = "error"
        return report
    # Issue every read first so TensorStore resolves them concurrently; tally consumes the
    # futures in the same order it walks the samples.
    half = PATCH // 2
    futures = iter([segmentation.img[(0, 0) + tuple(slice(v - half, v + half + 1) for v in voxel)].read()
                    for _, voxel in samples])
    counts = tally(samples, lambda voxel: next(futures).result())
    report.update({"fragments_layout": layout, "fragments": len(chosen),
                   "volume_shape_zyx": [int(s) for s in segmentation.img.shape[-3:]],
                   **counts, "seconds": round(time.monotonic() - started, 1)})
    report["verdict"] = verdict(report, min_agreement_pct, min_read_nodes)
    return report


def record(reports, path, command):
    """Append a dated section with the Markdown table to ``path`` (created with a header if absent)."""
    path = Path(path)
    header = ("# Fragment / segmentation source agreement\n\n"
              "Receipts of `python -m labeled_cache.source_check`: sampled interior fragment nodes read back from the\n"
              "dense segmentation, agreement = exact + within-1-voxel matches over nodes read. Matched sources give\n"
              "close to 100%; an unrelated segmentation run gives close to 0%.\n")
    section = (f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} on {platform.node()}\n\n"
               f"`{command}`\n\n{markdown_table(reports)}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text((existing or header) + section, encoding="utf-8")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m labeled_cache.source_check", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", required=True,
                        help="Brain ids from configs/segmentation_paths.json or segmentation_datasets.rtf.")
    parser.add_argument("--fragments", type=int, default=60, help="Fragments sampled per brain (default %(default)s).")
    parser.add_argument("--nodes-per-fragment", type=int, default=4, help="Interior nodes per fragment (default %(default)s).")
    parser.add_argument("--zips", type=int, default=6, help="SWC zip shards read per brain (default %(default)s).")
    parser.add_argument("--min-fragment-nodes", type=int, default=30, help="Skip shorter fragments (default %(default)s).")
    parser.add_argument("--anisotropy", type=float, nargs=3, default=DEFAULT_ANISOTROPY, metavar=("X", "Y", "Z"),
                        help="Voxel size in microns, as given to python -m labeled_cache (default 0.748 0.748 1.0).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-agreement", type=float, default=95.0,
                        help="Agreement percentage below which a brain is reported as MISMATCH (default %(default)s).")
    parser.add_argument("--min-read-nodes", type=int, default=100,
                        help="Fewer successfully read nodes than this gives 'insufficient' (default %(default)s).")
    parser.add_argument("--record", nargs="?", const=str(DEFAULT_RECORD), default=None, metavar="PATH",
                        help=f"Append the table to this Markdown file (default when given alone: {DEFAULT_RECORD}).")
    parser.add_argument("--json", type=Path, help="Also write the per-brain reports to this JSON file.")
    args = parser.parse_args(argv)
    if args.fragments < 1 or args.nodes_per_fragment < 1 or args.zips < 1 or args.min_fragment_nodes < 5:
        parser.error("--fragments, --nodes-per-fragment and --zips must be positive; --min-fragment-nodes at least 5")
    if any(a <= 0 for a in args.anisotropy):
        parser.error("--anisotropy must be positive")
    if not 0 <= args.min_agreement <= 100:
        parser.error("--min-agreement is a percentage")
    if args.fragments * args.nodes_per_fragment < args.min_read_nodes:
        parser.error(f"--fragments x --nodes-per-fragment ({args.fragments * args.nodes_per_fragment}) is below "
                     f"--min-read-nodes ({args.min_read_nodes}); every brain would be 'insufficient'")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    reports = {}
    for brain in dict.fromkeys(str(b) for b in args.brains):
        reports[brain] = check_brain(brain, fragments=args.fragments, nodes_per_fragment=args.nodes_per_fragment,
                                     zips=args.zips, min_fragment_nodes=args.min_fragment_nodes,
                                     anisotropy=tuple(args.anisotropy), seed=args.seed,
                                     min_agreement_pct=args.min_agreement, min_read_nodes=args.min_read_nodes)
        r = reports[brain]
        if "error" in r:
            print(f"{brain}: error: {r['error']}", flush=True)
        else:
            print(f"{brain}: {r['verdict']} {r['agreement_pct']}% ({r['exact_match']} exact + {r['match_within_1vox']} "
                  f"within 1 vox of {r['nodes']} nodes, {r['background']} background, {r['other_id']} other id, "
                  f"{r['read_failed']} failed; {r['fragments']} fragments from {r['fragments_layout']}; "
                  f"{r['segmentation_id']}; {r['seconds']} s)", flush=True)
    print()
    print(markdown_table(reports))
    command = " ".join(["python -m labeled_cache.source_check", *map(str, (argv if argv is not None else sys.argv[1:]))])
    if args.record:
        print(f"\nrecorded in {record(reports, args.record, command)}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(f"json in {args.json}")
    return 0 if all(r.get("verdict") == "matched" for r in reports.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
