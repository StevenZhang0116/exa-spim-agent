#!/usr/bin/env python3
"""Build one self-contained merge-review index.html per run.

Reads each run's *_merge_sites.csv and SWC zips, pulls raw MIPs from the zarr
volume, renders one JPEG per site, and injects everything into template.html.

    python build_review.py --out /root/capsule/results/merge_review
    python build_review.py --cfgs it2 --steps step4_learned_split_correction --limit 6 --preview
"""
import argparse
import base64
import csv
import glob
import hashlib
import io
import json
import os
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/root/capsule/data/aggs08_arborist_v2"
BRAIN = "794495"
VOLUME = ("gs://allen-nd-goog/from_aind/exaSPIM_794495_2026-01-21_14-25-07_training-data/"
          "whole-brain-denoised/fused.zarr/0/")
VOX_XYZ = np.array([0.748, 0.748, 1.0])  # µm per voxel

CFGS = ["it2", "agg_mt08_st08", "agg_mt08_st08_dt5k", "agg_mt08_st08_dt20k"]
STEPS = [
    ("original", "Original"),
    ("step1_learned_merge_correction", "1 · Learned merge"),
    ("step2_somas_merge_correction", "2 · Soma merge"),
    ("step3_soma_split_correction", "3 · Soma split"),
    ("step4_learned_split_correction", "4 · Learned split"),
]

PATCH_VOX = 96               # patch edge length in voxels (96**3 patch)
HALF_VOX = PATCH_VOX // 2
HALF_UM = HALF_VOX * VOX_XYZ  # half extent per axis in µm (x, y, z)
PPV = 3                      # rendered pixels per voxel
PANEL = PATCH_VOX * PPV
GAP = 4
SCALE_BAR_UM = 10
RING_UM = 6
ALSO_IN_UM = 15.0
RADIUS_SCALE = 0.25          # node radius (file units) -> drawn radius, as a fraction of r*PPU
MIN_WIDTH = 2
RAW_PCT = (1.0, 99.5)        # raw MIP clipped to these percentiles (pooled over the three panels)
BUDGET_BYTES = 14_000_000    # image payload per page (base64 size)

BG = (12, 15, 20)
C_FRAGMENT = (255, 70, 70)
C_GT = (40, 205, 225)
C_RING = (255, 0, 0)


# ---------------------------------------------------------------- SWC geometry
def parse_swc(buf):
    a = np.loadtxt(io.BytesIO(buf), comments="#", ndmin=2)
    return a


class Skel:
    """Edges (child -> parent) with radius, in world µm. Columns 2:5 of the SWC are x, y, z."""

    def __init__(self, arrays):
        e0, e1, er, es = [], [], [], []
        for k, a in enumerate(arrays):
            if a.size == 0:
                continue
            ids = a[:, 0].astype(np.int64)
            pid = a[:, 6].astype(np.int64)
            xyz = a[:, 2:5]
            r = a[:, 5]
            order = np.argsort(ids)
            sid = ids[order]
            j = np.clip(np.searchsorted(sid, pid), 0, len(sid) - 1)
            has_parent = (sid[j] == pid) & (pid > 0)
            par = order[j]
            # roots / single nodes become zero-length edges so they still draw as dots
            p1 = np.where(has_parent[:, None], xyz[par], xyz)
            e0.append(xyz)
            e1.append(p1)
            er.append(r)
            es.append(np.full(len(xyz), k, dtype=np.int32))
        if e0:
            self.E0 = np.concatenate(e0)
            self.E1 = np.concatenate(e1)
            self.ER = np.concatenate(er)
            self.ES = np.concatenate(es)
        else:
            self.E0 = self.E1 = np.zeros((0, 3))
            self.ER = np.zeros(0)
            self.ES = np.zeros(0, dtype=np.int32)
        self.mn = np.minimum(self.E0, self.E1)
        self.mx = np.maximum(self.E0, self.E1)

    def window(self, c, h):
        m = np.all((self.mx >= c - h) & (self.mn <= c + h), axis=1)
        return self.E0[m], self.E1[m], self.ER[m], self.ES[m]


class RunData:
    """Zip access with a parse cache shared across steps (GT files are identical between steps)."""

    def __init__(self, base):
        self.base = base
        self.lock = threading.Lock()
        self.skel_cache = {}
        self.zips = {}

    def step_dir(self, step):
        return os.path.join(self.base, step)

    def _zip(self, path):
        with self.lock:
            if path not in self.zips:
                self.zips[path] = zipfile.ZipFile(path)
            return self.zips[path]

    def read_swc(self, zpath, name):
        z = self._zip(zpath)
        info = z.getinfo(name)
        key = (name, info.CRC, info.file_size)
        with self.lock:
            hit = self.skel_cache.get(key)
        if hit is not None:
            return hit
        with self.lock:  # zipfile handles are not thread-safe
            buf = z.read(name)
        a = parse_swc(buf)
        with self.lock:
            self.skel_cache[key] = a
        return a

    def merge_zip(self, step):
        return glob.glob(os.path.join(self.step_dir(step), "*_fragments_with_merges.zip"))[0]


def load_sites(base, step):
    path = glob.glob(os.path.join(base, step, "*_merge_sites.csv"))[0]
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            if not row:
                continue
            vox = [int(float(v)) for v in row[5].strip("() ").split(",")]
            wld = [float(v) for v in row[6].strip("() ").split(",")]
            out.append({
                "id": row[0].replace(".swc", ""),
                "frag": row[1],
                "seg": row[2],
                "gt": row[3],
                "vox": vox,           # z, y, x
                "wld": wld,           # x, y, z
                "cable": float(row[7]),
            })
    return out


# ---------------------------------------------------------------- raw image
_vol = None
_vol_lock = threading.Lock()


def volume():
    global _vol
    with _vol_lock:
        if _vol is None:
            import tensorstore as ts
            _vol = ts.open({"driver": "zarr", "kvstore": VOLUME}).result()
        return _vol


def raw_mips(vox_zyx):
    """Three normalised MIP panels (XY, XZ, YZ), each PANEL x PANEL uint8."""
    z, y, x = vox_zyx
    vol = volume()
    shape = vol.shape[2:]
    lo = np.array([z - HALF_VOX, y - HALF_VOX, x - HALF_VOX])
    hi = np.array([z + HALF_VOX + 1, y + HALF_VOX + 1, x + HALF_VOX + 1])  # odd crop: site voxel is the exact centre
    clo = np.maximum(lo, 0)
    chi = np.minimum(hi, shape)
    crop = vol[0, 0, clo[0]:chi[0], clo[1]:chi[1], clo[2]:chi[2]].read().result()
    pad = [(clo[i] - lo[i], hi[i] - chi[i]) for i in range(3)]
    if any(p != (0, 0) for p in pad):
        crop = np.pad(crop, pad)
    mips = [crop.max(0), crop.max(1), crop.max(2)]   # (y,x), (z,x), (z,y)
    pooled = np.concatenate([m.ravel() for m in mips]).astype(np.float32)
    vlo, vhi = np.percentile(pooled, RAW_PCT[0]), np.percentile(pooled, RAW_PCT[1])
    vhi = max(vhi, vlo + 1)
    out = []
    for m in mips:
        g = np.clip((m.astype(np.float32) - vlo) / (vhi - vlo), 0, 1) ** 0.85
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((PANEL, PANEL), Image.BILINEAR, box=(0.5, 0.5, PATCH_VOX + 0.5, PATCH_VOX + 0.5))
        out.append(np.asarray(im))
    return out


# ---------------------------------------------------------------- drawing
AXES = [(0, 1), (0, 2), (1, 2)]   # XY, XZ, YZ


def project(P, c, a, b):
    return ((P[:, a] - c[a]) / VOX_XYZ[a] + HALF_VOX) * PPV, ((P[:, b] - c[b]) / VOX_XYZ[b] + HALF_VOX) * PPV


def draw_edges(draw, E0, E1, ER, c, a, b, color):
    if len(E0) == 0:
        return
    u0, v0 = project(E0, c, a, b)
    u1, v1 = project(E1, c, a, b)
    w = np.maximum(MIN_WIDTH, np.rint(2 * ER / VOX_XYZ[0] * PPV * RADIUS_SCALE)).astype(int)
    for i in range(len(u0)):
        wi = int(w[i])
        if u0[i] == u1[i] and v0[i] == v1[i]:
            r = wi / 2
            draw.ellipse([u0[i] - r, v0[i] - r, u0[i] + r, v0[i] + r], fill=color)
            continue
        draw.line([(u0[i], v0[i]), (u1[i], v1[i])], fill=color, width=wi)
        if wi >= 3:
            r = wi / 2 - 0.5
            draw.ellipse([u0[i] - r, v0[i] - r, u0[i] + r, v0[i] + r], fill=color)


def decorate(draw):
    r = RING_UM / VOX_XYZ[0] * PPV
    cx = cy = PANEL / 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=C_RING, width=2)
    y = PANEL - 9
    draw.line([(9, y), (9 + SCALE_BAR_UM / VOX_XYZ[0] * PPV, y)], fill=(235, 238, 242), width=2)


def skeleton_panels(c, gt_skel, frag_arr):
    panels = []
    g = gt_skel.window(c, HALF_UM + 2)
    f = Skel([frag_arr]).window(c, HALF_UM + 2) if frag_arr is not None else None
    for a, b in AXES:
        im = Image.new("RGB", (PANEL, PANEL), BG)
        d = ImageDraw.Draw(im)
        draw_edges(d, g[0], g[1], g[2], c, a, b, C_GT)
        if f is not None:
            draw_edges(d, f[0], f[1], f[2], c, a, b, C_FRAGMENT)
        decorate(d)
        panels.append(im)
    return panels


def _viridis_lut():
    import matplotlib
    return (matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


VIRIDIS = _viridis_lut()


def compose(raw, skel):
    W = 3 * PANEL + 2 * GAP
    H = 2 * PANEL + GAP
    canvas = Image.new("RGB", (W, H), (8, 10, 13))
    for i in range(3):
        x0 = i * (PANEL + GAP)
        r = Image.fromarray(VIRIDIS[raw[i]])
        decorate(ImageDraw.Draw(r))
        canvas.paste(r, (x0, 0))
        canvas.paste(skel[i], (x0, PANEL + GAP))
    return canvas


# ---------------------------------------------------------------- orchestration
def match_also_in(all_sites):
    """all_sites[cfg][step] -> list of sites; fill site['also_in'] with other cfgs within ALSO_IN_UM."""
    trees = {}
    for cfg, steps in all_sites.items():
        for step, sites in steps.items():
            if sites:
                trees[(cfg, step)] = cKDTree(np.array([s["wld"] for s in sites]))
    for cfg, steps in all_sites.items():
        for step, sites in steps.items():
            for s in sites:
                hits = []
                for other in all_sites:
                    if other == cfg or (other, step) not in trees:
                        continue
                    d, _ = trees[(other, step)].query(s["wld"])
                    if d <= ALSO_IN_UM:
                        hits.append(other)
                s["also_in"] = hits


def render_run(cfg, steps, sites_by_step, limit=None, workers=12):
    base = [p for p in glob.glob(os.path.join(DATA, f"eval_{BRAIN}_{cfg}", "*")) if not p.endswith("output")][0]
    run = RunData(base)
    jobs = []
    for step in steps:
        for s in sites_by_step[step][:limit]:
            jobs.append((step, s))

    raw_cache, raw_lock = {}, threading.Lock()

    def get_raw(vox):
        key = tuple(vox)
        with raw_lock:
            if key in raw_cache:
                return raw_cache[key]
        r = raw_mips(vox)
        with raw_lock:
            raw_cache[key] = r
        return r

    def one(job):
        step, s = job
        mz = run.merge_zip(step)
        gt_arr = run.read_swc(mz, f"{s['gt']}.0.swc")
        frag_name = f"{s['frag']}.0.swc"
        frag_arr = run.read_swc(mz, frag_name) if frag_name in run._zip(mz).namelist() else None
        c = np.array(s["wld"])
        skel = skeleton_panels(c, Skel([gt_arr]), frag_arr)
        return compose(get_raw(s["vox"]), skel)

    with ThreadPoolExecutor(workers) as pool:
        imgs = list(pool.map(one, jobs))
    return jobs, imgs


def encode_budgeted(images, budget=BUDGET_BYTES):
    """Dedupe identical renders, then pick the highest JPEG quality that fits the budget."""
    uniq, index, seen = [], [], {}
    for im in images:
        h = hashlib.md5(im.tobytes()).hexdigest()
        if h not in seen:
            seen[h] = len(uniq)
            uniq.append(im)
        index.append(seen[h])
    scale = 1.0
    while True:
        work = uniq if scale == 1.0 else [
            im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS) for im in uniq]
        for q in range(86, 39, -4):
            blobs = []
            for im in work:
                bio = io.BytesIO()
                im.save(bio, "JPEG", quality=q, optimize=True, subsampling=0 if q > 78 else 2)
                blobs.append(bio.getvalue())
            if sum(len(b) for b in blobs) * 4 / 3 <= budget:
                uris = ["data:image/jpeg;base64," + base64.b64encode(b).decode() for b in blobs]
                return uris, index, q, scale
        scale *= 0.85


def font_css():
    from fontTools import subset
    import matplotlib
    d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    faces = [("Review Sans", "DejaVuSans.ttf", 400), ("Review Sans", "DejaVuSans-Bold.ttf", 700),
             ("Review Mono", "DejaVuSansMono.ttf", 400), ("Review Mono", "DejaVuSansMono-Bold.ttf", 700)]
    text = "".join(chr(c) for c in range(32, 127)) + "µ±·→←↑↓×–—…✓•▸"
    css = []
    for fam, fn, wt in faces:
        opts = subset.Options()
        opts.flavor = "woff"
        opts.layout_features = ["kern", "liga"]
        font = subset.load_font(os.path.join(d, fn), opts)
        sub = subset.Subsetter(opts)
        sub.populate(text=text)
        sub.subset(font)
        bio = io.BytesIO()
        subset.save_font(font, bio, opts)
        b64 = base64.b64encode(bio.getvalue()).decode()
        css.append(f"@font-face{{font-family:'{fam}';font-weight:{wt};font-style:normal;"
                   f"src:url(data:font/woff;base64,{b64}) format('woff');}}")
    return "\n".join(css)


def build_page(cfg, jobs, imgs, all_sites, out_dir, template):
    uris, index, q, scale = encode_budgeted(imgs)
    sites = []
    for (step, s), ix in zip(jobs, index):
        sites.append({"cfg": cfg, "step": step, "id": s["id"], "seg": s["seg"], "gt": s["gt"],
                      "vox": s["vox"], "wld": s["wld"], "cable": s["cable"],
                      "also_in": s.get("also_in", []), "img": ix})
    used_steps = [k for k, _ in STEPS if any(s["step"] == k for s in sites)]
    D = {
        "dataset": f"merge_review_{BRAIN}_{cfg}",
        "brain": BRAIN,
        "cfgs": [cfg],
        "steps": [{"id": k, "label": dict(STEPS)[k]} for k in used_steps],
        "patch_vox": PATCH_VOX,
        "scale_bar_um": SCALE_BAR_UM,
        "ring_um": RING_UM,
        "img_w": 3 * PANEL + 2 * GAP,
        "img_h": 2 * PANEL + GAP,
        "panel": PANEL,
        "gap": GAP,
        "imgs": uris,
        "sites": sites,
    }
    html = template.replace("/*__FONTS__*/", font_css()).replace("/*__DATA__*/", json.dumps(D, separators=(",", ":")))
    os.makedirs(os.path.join(out_dir, cfg), exist_ok=True)
    path = os.path.join(out_dir, cfg, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    mb = os.path.getsize(path) / 1e6
    print(f"[{cfg}] {len(sites)} sites, {len(uris)} unique images, jpeg q={q} scale={scale:.2f} -> {path} ({mb:.1f} MB)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/capsule/results/merge_review")
    ap.add_argument("--cfgs", nargs="*", default=CFGS)
    ap.add_argument("--steps", nargs="*", default=[k for k, _ in STEPS])
    ap.add_argument("--limit", type=int, default=None, help="sites per step (for quick tests)")
    ap.add_argument("--preview", action="store_true", help="write the first few JPEGs next to the output and stop")
    ap.add_argument("--template", default=os.path.join(HERE, "template.html"))
    a = ap.parse_args()

    bases = {}
    for cfg in CFGS:
        bases[cfg] = [p for p in glob.glob(os.path.join(DATA, f"eval_{BRAIN}_{cfg}", "*")) if not p.endswith("output")][0]
    all_sites = {cfg: {k: load_sites(bases[cfg], k) for k, _ in STEPS} for cfg in CFGS}
    match_also_in(all_sites)

    template = open(a.template, encoding="utf-8").read() if not a.preview else ""
    for cfg in a.cfgs:
        jobs, imgs = render_run(cfg, a.steps, all_sites[cfg], limit=a.limit)
        if a.preview:
            os.makedirs(a.out, exist_ok=True)
            for n, ((step, s), im) in enumerate(zip(jobs, imgs)):
                im.save(os.path.join(a.out, f"preview_{cfg}_{step[:5]}_{s['id']}.png"))
            print(f"wrote {len(imgs)} previews to {a.out}")
            continue
        build_page(cfg, jobs, imgs, all_sites, a.out, template)


if __name__ == "__main__":
    sys.exit(main())
