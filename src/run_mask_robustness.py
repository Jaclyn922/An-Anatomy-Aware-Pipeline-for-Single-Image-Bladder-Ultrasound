"""Step 6a: what if the segmentation mask the method uses is wrong?

The scene is built from the TRUE lumen mask; the method only gets a perturbed copy. Metrics are still computed
against the truth (device mask, clean reference, true lumen). Perturbations: boundary moved out/in by k px,
the whole mask shifted by 4 px, and a smooth random warp of the boundary (about 3 px).

Usage (project root):
    python src/run_mask_robustness.py --n 300 --conds srad_lumen roi_full --out outputs/results_mask.csv
"""
import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS  # noqa: E402
from degrade import realise, sample_specs  # noqa: E402
from metrics import contrast, detect_stat_dog, hallucination, regions  # noqa: E402
from simulate import load_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SP, RP = load_calibration(ROOT / "outputs/sim_calibration.json")
CONDS = ["srad_lumen", "roi_full"]
PERTURBATIONS = ["none", "dilate1", "dilate2", "dilate4", "erode1", "erode2", "erode4", "shift4", "warp3"]


def perturb(mask, kind, rng):
    m = mask.astype(bool)
    if kind == "none":
        return m
    if kind.startswith("dilate"):
        return ndi.binary_dilation(m, iterations=int(kind[6:]))
    if kind.startswith("erode"):
        out = ndi.binary_erosion(m, iterations=int(kind[5:]))
        return out if out.sum() > 60 else m
    if kind == "shift4":
        ang = rng.uniform(0, 2 * np.pi)
        return ndi.shift(m.astype(np.uint8), (4 * np.sin(ang), 4 * np.cos(ang)), order=0).astype(bool)
    if kind == "warp3":
        H, W = m.shape
        dy = ndi.gaussian_filter(rng.standard_normal((H, W)), 6.0)
        dx = ndi.gaussian_filter(rng.standard_normal((H, W)), 6.0)
        dy *= 3.0 / (np.abs(dy).max() + 1e-9)
        dx *= 3.0 / (np.abs(dx).max() + 1e-9)
        yy, xx = np.mgrid[:H, :W].astype(float)
        out = ndi.map_coordinates(m.astype(np.uint8), [yy + dy, xx + dx], order=0, mode="nearest").astype(bool)
        return out if out.sum() > 60 else m
    raise ValueError(kind)


def one_spec(spec):
    d = realise(spec, SP, RP)
    if d is None:
        return []
    dev, lumen, fov = d["device"], d["lumen"], d["fov"]
    Lc, B = regions(lumen, dev)
    Lc_a = ndi.binary_erosion(lumen, iterations=4)
    ok = bool(B.sum() >= 30)
    c_clean = contrast(d["clean_p"], dev, B) if ok else np.nan
    base = dict(spec_id=spec["spec_id"], frame=spec["frame"], video=spec["video"], device_gain=spec["device_gain"],
                gain_offset_db=spec["gain_offset_db"], visible=int(ok and c_clean >= 8.0))
    vis = bool(base["visible"])
    raw_H = hallucination(d["noisy_a"], d["clean_a"], Lc_a)
    rng = np.random.default_rng(spec["spec_id"] + 12345)
    rows = [dict(base, perturb="-", condition="raw", retention=(contrast(d["noisy_p"], dev, B) / c_clean) if vis else np.nan,
                 dH=0.0, S_dog_present=detect_stat_dog(d["noisy_p"], Lc), S_dog_absent=detect_stat_dog(d["noisy_a"], Lc_a),
                 mask_dice=1.0)]
    for kind in PERTURBATIONS:
        pm = perturb(lumen, kind, rng)
        dice = 2.0 * (pm & lumen).sum() / (pm.sum() + lumen.sum())
        for name in CONDS:
            fn = CONDITIONS[name]
            proc_p = fn(d["noisy_p"], pm, fov)
            proc_a = fn(d["noisy_a"], pm, fov)
            rows.append(dict(base, perturb=kind, condition=name,
                             retention=(contrast(proc_p, dev, B) / c_clean) if vis else np.nan,
                             dH=hallucination(proc_a, d["clean_a"], Lc_a) - raw_H,
                             S_dog_present=detect_stat_dog(proc_p, Lc), S_dog_absent=detect_stat_dog(proc_a, Lc_a),
                             mask_dice=float(dice)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--conds", nargs="+", default=CONDS)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/results_mask.csv")
    a = ap.parse_args()
    CONDS[:] = a.conds
    seen, frames = set(), []
    for r in csv.DictReader(open(ROOT / "data/manifest.csv")):
        if r["frame"] not in seen:
            seen.add(r["frame"])
            frames.append(r)
    specs = sample_specs(frames, 600, seed=0)[: a.n]  # the same first scenes as the main experiment
    t0, out = time.time(), []
    with Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap(one_spec, specs, chunksize=2), 1):
            out.extend(rows)
            if i % 25 == 0:
                print(f"  {i}/{len(specs)} scenes, {time.time() - t0:.0f}s", flush=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    print(f"wrote {len(out)} rows to {a.out} in {time.time() - t0:.0f}s")
