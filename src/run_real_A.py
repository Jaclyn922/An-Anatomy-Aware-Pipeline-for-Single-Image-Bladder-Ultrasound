"""Experiment A (real frames): the noise-versus-edge trade-off of every despeckling family, with strength sweeps.

No clean reference exists for a real frame, so three reference-free measurements are used, all computed from the same
hand-drawn lumen mask. Every number is reported as processed / raw, so mask looseness cancels out.
  noise_removed   1 - (fine-scale std in the lumen core, processed / raw)            (higher = smoother lumen)
  std_removed     1 - (total plane-detrended std in the lumen core, processed / raw)  (also catches coarse blobs)
  edge_keep       steepness of the mean intensity profile across the lumen boundary, processed / raw
                  The profile is averaged over every boundary pixel (signed distance to the mask boundary, 1-px bins), so speckle
                  averages out; a blurred wall edge gives a lower maximum slope. (higher = sharper wall kept)
  tissue_keep     fine-scale std of nearby outer tissue, processed / raw             (1 = outer texture untouched)
Families: global Gaussian, global bilateral, global NLM, global SRAD, ROI v1 (three regions), ROI v2 (barrier at lumen core).

Usage (project root):  python src/run_real_A.py --workers 2
Output: outputs/realA_rows.csv  (one row per frame x method x strength)
"""
import argparse
import csv
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.restoration import denoise_nl_means

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi import build_rois, roi_filter, srad  # noqa: E402
from roi_nlm import roi_nlm  # noqa: E402
from simulate import fov_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def hp(img, sigma=3.0):
    f = img.astype(np.float64)
    return f - ndi.gaussian_filter(f, sigma)


def lumen_std(img, Lc):
    """Total grey-level spread in the lumen core after removing a fitted plane (catches coarse blobs the 3-px high-pass misses)."""
    yy, xx = np.nonzero(Lc)
    v = img.astype(np.float64)[yy, xx]
    A = np.c_[np.ones_like(xx), xx, yy].astype(float)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return float((v - A @ coef).std())


def profile(img, sd, lo=-10, hi=10):
    """Mean intensity by signed distance (px) from the mask boundary, 1-px bins. sd < 0 inside the lumen."""
    f = img.astype(np.float64)
    bins = np.round(sd).astype(int)
    out = []
    for b in range(lo, hi + 1):
        m = bins == b
        out.append(f[m].mean() if m.sum() >= 20 else np.nan)
    return np.array(out)


def max_slope(p):
    d = np.diff(p)
    return float(np.nanmax(d)) if np.isfinite(d).any() else np.nan


def methods():
    m = []
    for s in (0.7, 1.0, 1.5, 2.0):
        m.append(("gauss_global", s, lambda img, mask, fov, s=s: np.clip(ndi.gaussian_filter(img.astype(float), s), 0, 255).astype(np.uint8)))
    for sc in (20, 40, 80, 160):
        m.append(("bilateral_global", sc, lambda img, mask, fov, sc=sc: cv2.bilateralFilter(img, 7, sc, 3)))
    for hm in (0.5, 1.0, 1.5, 2.5):
        def nlm(img, mask, fov, hm=hm):
            f = img.astype(np.float64) / 255.0
            h = hp(img) / 255.0
            sig = float(1.4826 * np.median(np.abs(h[fov] - np.median(h[fov]))))
            out = denoise_nl_means(f, patch_size=5, patch_distance=6, h=hm * sig, sigma=sig, fast_mode=True)
            return np.clip(np.round(out * 255), 0, 255).astype(np.uint8) * fov.astype(np.uint8)
        m.append(("nlm_global", hm, nlm))
    for it in (10, 30, 100, 200, 400):
        m.append(("srad_global", it, lambda img, mask, fov, it=it: np.clip(np.round(srad(img.astype(float) / 255.0, iters=it, q0_region=None) * 255), 0, 255).astype(np.uint8) * fov.astype(np.uint8)))
    for it in (30, 100, 200, 400):
        m.append(("roi_v1", it, lambda img, mask, fov, it=it: roi_filter(img, mask.astype(np.uint8), srad_iters=it)))
        m.append(("roi_v2", it, lambda img, mask, fov, it=it: roi_filter(img, mask.astype(np.uint8), srad_iters=it, guard=True)))
    for hm in (1.0, 2.0, 4.0, 8.0):
        m.append(("roi_nlm", hm, lambda img, mask, fov, hm=hm: roi_nlm(img, mask, hmult=hm)))
    return m


METHODS = methods()


def one_frame(row):
    img = np.array(Image.open(ROOT / "data/roboflow" / row["file"]).convert("L"))
    mask = np.array(Image.open(ROOT / "data/masks" / (Path(row["file"]).stem + "_mask.png"))) > 127
    fov = fov_mask(img)
    core, wall, outer = build_rois(mask, 3, 3)
    Lc = ndi.binary_erosion(mask, iterations=5)
    d_out = ndi.distance_transform_edt(~ndi.binary_dilation(mask, iterations=3))
    tis = ndi.binary_erosion(outer & fov & (d_out < 25), iterations=2)
    if Lc.sum() < 50 or tis.sum() < 200:
        return []
    sd = ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)
    p0 = profile(img, sd)
    s0 = max_slope(p0)
    n0 = float(hp(img)[Lc].std())
    u0 = lumen_std(img, Lc)
    t0 = float(hp(img)[tis].std())
    rows = []
    for fam, strength, fn in METHODS:
        out = fn(img, mask, fov)
        rows.append(dict(frame=row["frame"], video=row["video"], family=fam, strength=strength,
                         noise_removed=1.0 - float(hp(out)[Lc].std()) / max(n0, 1e-6),
                         std_removed=1.0 - lumen_std(out, Lc) / max(u0, 1e-6),
                         edge_keep=max_slope(profile(out, sd)) / max(s0, 1e-6),
                         tissue_keep=float(hp(out)[tis].std()) / max(t0, 1e-6)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    seen, frames = set(), []
    for r in csv.DictReader(open(ROOT / "data/manifest.csv")):
        if r["frame"] not in seen:
            seen.add(r["frame"])
            frames.append(r)
    t0, out = time.time(), []
    with get_context("spawn").Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap(one_frame, frames, chunksize=1), 1):
            out.extend(rows)
            if i % 10 == 0:
                print(f"  {i}/{len(frames)} frames {time.time() - t0:.0f}s", flush=True)
    with open(ROOT / "outputs/realA_rows.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    print(f"wrote {len(out)} rows {time.time() - t0:.0f}s")
