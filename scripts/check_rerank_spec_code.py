"""End-to-end check of markdowns/{merge,split}_features_for_proofreader_evolve.md.

Simulates one AutoDiscovery submission: run inside a directory that holds only the
four submitted files, in a fresh environment set up with the spec's install line.
Executes the spec's code blocks as written, then compares them with evolve's
context cache (geometry and image patches) and computes the baselines.

    python scripts/check_rerank_spec_code.py --kind merge --workdir <dir with the 4 files>
"""
import argparse
import glob
import json
import os
from pathlib import Path
import re
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = ("DESCRIPTOR = {", "context = {")  # illustrative blocks, not runnable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["merge", "split"], required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--rows", type=int, default=200)
    p.add_argument("--image-rows", type=int, default=10)
    a = p.parse_args()
    kind = a.kind
    spec = (a.workdir / f"{kind}_features_for_proofreader_evolve.md").read_text()
    os.chdir(a.workdir)
    print(f"[{kind}] files: {sorted(os.listdir('.'))}", flush=True)

    # 1. Every runnable python block, in document order (the loading block loads the pkl).
    ns = {}
    blocks = [b for b in re.findall(r"```python\n(.*?)```", spec, re.S) if not b.lstrip().startswith(TEMPLATES)]
    t0 = time.time()
    for b in blocks:
        exec(b, ns)
    print(f"[{kind}] executed {len(blocks)} code blocks ({time.time() - t0:.0f}s incl. pkl load)", flush=True)
    payload, frag = ns["payload"], ns["frag"]
    for key in ("gt_node_canonical_label", "gt_edge_error", "gt_merge_labels", "gt_merge_sites"):
        if f'payload["{key}"]' in spec:
            assert payload.get(key) is not None, key
    if getattr(frag, "kdtree", None) is None:
        frag.set_kdtree()
    print(f"[{kind}] fragments: {len(frag.node_xyz)} nodes; node_segment_id -> {type(frag.node_segment_id(0)).__name__}")

    # 2. Companion and manifest, as the spec says to load them.
    comp = dict(np.load(f"rerank_companion_794495_{kind}.npz", allow_pickle=False))
    manifest = json.load(open(f"rerank_companion_794495_{kind}.json"))
    assert set(comp) == set(manifest["columns"]), "npz columns differ from manifest"
    lab = comp["label"]
    print(f"[{kind}] companion rows {len(lab)}, labels 1/0/NaN = {int((lab == 1).sum())}/{int((lab == 0).sum())}/"
          f"{int(np.isnan(lab).sum())}")

    # 3. neighbourhood() vs evolve geometry cache.
    entry = Path(glob.glob(str(REPO / f"proofreader_evolve/context_cache/794495/{kind}/*/manifest.json"))[0]).parent
    g = np.load(sorted((entry / "geometry").glob("chunk_*.npz"))[0])
    mismatch = 0
    for i in range(a.rows):
        row = int(g["rows"][i])
        ctx = ns["make_context"](frag, comp, row)["fragment"]
        lo, hi = g["node_offsets"][i], g["node_offsets"][i + 1]
        if (ctx["xyz_um"].shape != g["xyz_um"][lo:hi].shape or not np.array_equal(g["segment"][lo:hi], ctx["segment"])
                or np.abs(ctx["xyz_um"] - g["xyz_um"][lo:hi]).max() > 1e-9):
            mismatch += 1
    print(f"[{kind}] neighbourhood vs evolve cache: {mismatch}/{a.rows} mismatched", flush=True)

    # 4. Image: manifest geometry vs zarr metadata, then image_patch() vs evolve image cache.
    import tensorstore as ts
    bucket, key = manifest["image"]["uri"][len("s3://"):].split("/", 1)
    kv = ts.KvStore.open({"driver": "s3", "bucket": bucket, "path": key.rstrip("/") + "/"}).result()
    zattrs = json.loads(kv.read(".zattrs").result().value)
    for ds in zattrs["multiscales"][0]["datasets"][:2]:
        t = {c["type"]: c for c in ds["coordinateTransformations"]}
        want = manifest["image"]["levels"][ds["path"]]
        assert np.allclose(t["scale"]["scale"][-3:][::-1], want["scale_xyz_um"]), ds
        assert np.allclose(t.get("translation", {"translation": [0] * 5})["translation"][-3:][::-1],
                           want["translation_xyz_um"]), ds
    print(f"[{kind}] manifest level geometry matches .zattrs", flush=True)
    for tier, level in (("level0", 0), ("level1", 1)):
        chunk = sorted((entry / "images" / tier).glob("chunk_*.npz"))
        if not chunk:
            print(f"[{kind}] no evolve image cache for {tier}; skipped")
            continue
        c = np.load(chunk[0])
        array = ns["open_level"](manifest, level)
        bad, checked, t0 = 0, 0, time.time()
        for local in np.flatnonzero(c["available"])[:a.image_rows]:
            ctx = ns["make_context"](frag, comp, int(c["rows"][local]), array, manifest, level)
            same = (ctx["image_available"] and np.array_equal(ctx["image_zyx"], c[f"image_{local}"])
                    and np.array_equal(ctx["valid_zyx"], c[f"valid_{local}"])
                    and np.allclose(ctx["anchors_zyx"], c["anchors_zyx"][local], equal_nan=True))
            bad += not same
            checked += 1
        print(f"[{kind}] image_patch {tier} vs evolve cache: {bad}/{checked} mismatched "
              f"({(time.time() - t0) / max(checked, 1):.1f}s per patch)", flush=True)

    # 5. A toy feature module through the contract, then the scoring code and baselines.
    def describe(context):
        f = context["fragment"]
        img = context.get("image_zyx")
        return {"n_nodes": float(len(f["xyz_um"])),
                "patch_mean": float(img[context["valid_zyx"]].mean()) if img is not None and context["image_available"] else float("nan")}
    out = describe(ns["make_context"](frag, comp, int(np.flatnonzero(lab == 1)[0])))
    assert set(out) == {"n_nodes", "patch_mean"} and all(isinstance(v, float) for v in out.values())
    keys = ns["candidate_keys"](comp)
    rng = np.random.default_rng(0)
    baselines = [("random", rng.random(len(lab)))]
    if kind == "split":
        baselines.append(("gap_um ascending", -comp["gap_um"]))
    else:
        u, v = np.array(list(frag.edges)).T
        seg_of = np.array([int(frag.node_segment_id(n)) for n in range(len(frag.node_xyz))])
        segs, inv = np.unique(seg_of[u], return_inverse=True)
        total = dict(zip(segs.tolist(), np.bincount(inv, weights=np.linalg.norm(frag.node_xyz[u] - frag.node_xyz[v], axis=1)).tolist()))
        cable = np.array([total[int(s)] for s in comp["segment_id"]])
        baselines += [("degree descending", comp["degree"].astype(float)), ("segment cable descending", cable)]
    for name, scores in baselines:
        curve = [round(ns["precision_at_fraction"](comp, scores, f, keys), 3) for f in (.01, .025, .05, .10)]
        print(f"[{kind}]   {name}: P@1/2.5/5/10% = {curve}, AP = {ns['average_precision'](comp, scores):.3f}", flush=True)
    print(f"[{kind}] DONE")


if __name__ == "__main__":
    main()
