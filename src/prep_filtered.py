"""Experiment B, step 1: filter every real image once per condition and cache it (so training does not repeat the slow SRAD).

Conditions are fixed BEFORE looking at any result (pre-specified):
  raw         no processing
  nlm         global non-local means, h = 1.0 x noise sigma
  srad30      global SRAD, 30 iterations
  srad100     global SRAD, 100 iterations
  roi30       the course-project ROI filter (three regions, 30 iterations, no guard)
  roi_v2_100  ROI with the barrier at the lumen core, 100 iterations
  roi_nlm2    NLM (h = 2 x sigma) inside the lumen core only, wall and outer untouched (h chosen on the dev fold 0 by max std_removed with edge_keep >= 0.95)
ROI filters here use the HAND-DRAWN mask (oracle). The deployment version (mask predicted by a segmenter) is built in run_real_seg.py.

Usage (project root):  python src/prep_filtered.py --workers 2      Output: outputs/cache_filtered.npz
"""
import argparse
import csv
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.restoration import denoise_nl_means

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi import roi_filter, srad  # noqa: E402
from roi_nlm import roi_nlm  # noqa: E402
from simulate import fov_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONDS = ["raw", "nlm", "srad30", "srad100", "roi30", "roi_v2_100", "roi_nlm2"]


def nlm(img, fov):
    f = img.astype(np.float64) / 255.0
    h = (img.astype(float) - ndi.gaussian_filter(img.astype(float), 3.0)) / 255.0
    sig = float(1.4826 * np.median(np.abs(h[fov] - np.median(h[fov]))))
    out = denoise_nl_means(f, patch_size=5, patch_distance=6, h=1.0 * sig, sigma=sig, fast_mode=True)
    return np.clip(np.round(out * 255), 0, 255).astype(np.uint8) * fov.astype(np.uint8)


def sradg(img, fov, it):
    return np.clip(np.round(srad(img.astype(float) / 255.0, iters=it, q0_region=None) * 255), 0, 255).astype(np.uint8) * fov.astype(np.uint8)


def apply(name, img, mask, fov):
    if name == "raw":
        return img
    if name == "nlm":
        return nlm(img, fov)
    if name == "srad30":
        return sradg(img, fov, 30)
    if name == "srad100":
        return sradg(img, fov, 100)
    if name == "roi30":
        return roi_filter(img, mask.astype(np.uint8), srad_iters=30)
    if name == "roi_v2_100":
        return roi_filter(img, mask.astype(np.uint8), srad_iters=100, guard=True)
    if name == "roi_nlm2":
        return roi_nlm(img, mask, hmult=2.0)
    raise ValueError(name)


def one(row):
    img = np.array(Image.open(ROOT / "data/roboflow" / row["file"]).convert("L"))
    mask = np.array(Image.open(ROOT / "data/masks" / (Path(row["file"]).stem + "_mask.png"))) > 127
    fov = fov_mask(img)
    return row["file"], {c: apply(c, img, mask, fov) for c in CONDS}, mask


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    t0 = time.time()
    store = {}
    with get_context("spawn").Pool(a.workers) as pool:
        for i, (f, d, m) in enumerate(pool.imap(one, rows, chunksize=2), 1):
            for c, v in d.items():
                store[f"{c}|{f}"] = v
            store[f"mask|{f}"] = m.astype(np.uint8)
            if i % 20 == 0:
                print(f"  {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(ROOT / "outputs/cache_filtered.npz", **store)
    print(f"saved {len(store)} arrays {time.time() - t0:.0f}s")
