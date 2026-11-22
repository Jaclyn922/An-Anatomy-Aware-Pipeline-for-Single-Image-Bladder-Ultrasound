"""Step 6c: sanity check on the REAL frames (one file per source frame, 67 frames, 9 videos).

Part A, no reference needed (is the behaviour on real data consistent with the simulation?):
  lumen_noise   fine-scale noise in the lumen core, processed / raw
  tissue_tex    speckle texture of nearby tissue, processed / raw
  edge_keep     mean gradient magnitude on the lumen boundary band, processed / raw (the masks are hand-drawn and
                loose, so this is indicative only)
  lumen_shift   change of mean grey level in the lumen core
Part B, device composites: the simulator's device effect (device echo, shadow, reverberation = clean image with
device minus clean image without) is added to the REAL frame, and retention of the device contrast is measured on
that real background. No clean reference exists for a real frame, so retention = contrast(processed) /
contrast(composite before processing).

Usage (project root):  python src/run_real_check.py --per-frame 5
Outputs: outputs/real_noref.csv, outputs/real_composite.csv
"""
import argparse
import csv
import dataclasses
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS  # noqa: E402
from device import make_device_mask  # noqa: E402
from metrics import contrast, hp_noise, regions  # noqa: E402
from roi import build_rois  # noqa: E402
from simulate import build_scene, fov_mask, load_calibration, render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SP, RP = load_calibration(ROOT / "outputs/sim_calibration.json")
CONDS = ["raw", "nlm_global", "srad_global", "srad_lumen", "roi_full", "roi_i100", "roi_i200", "roi_d30", "roi_d100", "roi_d200"]
PER_FRAME = 5


def grad_mag(img):
    f = img.astype(np.float64)
    return np.hypot(ndi.sobel(f, 0), ndi.sobel(f, 1))


def one_frame(row):
    img = np.array(Image.open(ROOT / "data/roboflow" / row["file"]).convert("L"))
    mask = np.array(Image.open(ROOT / "data/masks" / (Path(row["file"]).stem + "_mask.png"))) > 127
    fov = fov_mask(img)
    core, wall, outer = build_rois(mask, 3, 3)
    Lc = ndi.binary_erosion(mask, iterations=5)
    d_out = ndi.distance_transform_edt(~ndi.binary_dilation(mask, iterations=3))
    tis = ndi.binary_erosion(outer & fov & (d_out < 25), iterations=2)
    band = ndi.binary_dilation(mask, iterations=2) & ~ndi.binary_erosion(mask, iterations=2)
    if Lc.sum() < 50 or tis.sum() < 200:
        return [], []
    g_raw = grad_mag(img)[band].mean()
    hp_t = lambda x: float((x.astype(float) - ndi.gaussian_filter(x.astype(float), 3.0))[tis].std())
    noref, comp = [], []
    procs = {}
    for c in CONDS:
        procs[c] = CONDITIONS[c](img, mask, fov)
        p = procs[c]
        noref.append(dict(frame=row["frame"], video=row["video"], condition=c,
                          lumen_noise=hp_noise(p, Lc) / max(hp_noise(img, Lc), 1e-6),
                          tissue_tex=hp_t(p) / max(hp_t(img), 1e-6),
                          edge_keep=float(grad_mag(p)[band].mean() / max(g_raw, 1e-6)),
                          lumen_shift=float(p[core].mean() - img[core].mean())))
    rng = np.random.default_rng(zlib.crc32(row["frame"].encode()))  # deterministic (hash() changes every run)
    for j in range(PER_FRAME):
        touch = bool(j % 2)
        dev, _ = make_device_mask(mask, rng, touch_wall=touch)
        if dev is None:
            continue
        g = float(np.exp(rng.uniform(np.log(0.5), np.log(6.0))))
        sp = dataclasses.replace(SP, device_gain=g)
        seed = int(rng.integers(0, 2**31 - 1))
        _, clean_p = render(build_scene(mask, fov, sp, seed, device=dev), 1, RP)
        _, clean_a = render(build_scene(mask, fov, sp, seed, device=None), 1, RP)
        composite = np.clip(img.astype(float) + (clean_p - clean_a) * 255.0, 0, 255).astype(np.uint8)
        Lc2, B = regions(mask, dev)
        if B.sum() < 30:
            continue
        c0 = contrast(composite, dev, B)
        if c0 < 8.0:
            continue
        for c in CONDS:
            p = CONDITIONS[c](composite, mask, fov)
            comp.append(dict(frame=row["frame"], video=row["video"], condition=c, device_gain=g, touch_wall=int(touch),
                             contrast_before=c0, retention=contrast(p, dev, B) / c0))
    return noref, comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-frame", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    PER_FRAME = a.per_frame
    seen, frames = set(), []
    for r in csv.DictReader(open(ROOT / "data/manifest.csv")):
        if r["frame"] not in seen:
            seen.add(r["frame"])
            frames.append(r)
    t0, A, Bc = time.time(), [], []
    with Pool(a.workers) as pool:
        for i, (n, c) in enumerate(pool.imap(one_frame, frames, chunksize=1), 1):
            A.extend(n)
            Bc.extend(c)
            if i % 10 == 0:
                print(f"  {i}/{len(frames)} frames, {time.time() - t0:.0f}s", flush=True)
    for name, rows in (("real_noref", A), ("real_composite", Bc)):
        with open(ROOT / f"outputs/{name}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print(f"wrote {len(A)} + {len(Bc)} rows in {time.time() - t0:.0f}s")
