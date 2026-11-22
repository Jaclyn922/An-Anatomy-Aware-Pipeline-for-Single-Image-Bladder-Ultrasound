"""Step 3d: run every processing condition on every matched present/absent scene and save per-image metrics.

Usage (project root):  python src/run_experiment.py --n 240 --workers 2
Output: outputs/results.csv  (one row per scene x condition)
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
from conditions import CONDITIONS, SETS  # noqa: E402
from degrade import realise, sample_specs  # noqa: E402
from metrics import contrast, detect_stat, detect_stat_dog, hallucination, hp_noise, psnr_ssim, regions  # noqa: E402
from simulate import load_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SP, RP = load_calibration(ROOT / "outputs/sim_calibration.json")
VISIBLE_MIN = 8.0  # grey levels of clean-reference contrast below which the device is treated as not visible


NAMES = list(CONDITIONS)


def one_spec(spec):
    d = realise(spec, SP, RP)
    if d is None:
        return []
    dev, lumen, fov = d["device"], d["lumen"], d["fov"]
    Lc, B = regions(lumen, dev)
    Lc_a = ndi.binary_erosion(lumen, iterations=4)
    ok = bool(B.sum() >= 30)
    c_clean = contrast(d["clean_p"], dev, B) if ok else np.nan
    rows = []
    for name in NAMES:
        fn = CONDITIONS[name]
        proc_p = fn(d["noisy_p"], lumen, fov)
        proc_a = fn(d["noisy_a"], lumen, fov)
        c_proc = contrast(proc_p, dev, B) if ok else np.nan
        ps, ss = psnr_ssim(proc_p, d["clean_p"], fov)
        rows.append(
            dict(
                **{k: spec[k] for k in ("spec_id", "frame", "video", "device_gain", "shadow_strength", "gain_offset_db",
                                          "dr_scale", "psf_scale", "looks", "touch_wall")},
                condition=name,
                device_area=int(dev.sum()),
                visible=int(ok and c_clean >= VISIBLE_MIN),
                contrast_clean=c_clean,
                contrast_proc=c_proc,
                retention=(c_proc / c_clean) if ok and abs(c_clean) > 1e-6 else np.nan,
                S_present=detect_stat(proc_p, Lc),
                S_absent=detect_stat(proc_a, Lc_a),
                S_dog_present=detect_stat_dog(proc_p, Lc),
                S_dog_absent=detect_stat_dog(proc_a, Lc_a),
                lumen_hp_std_absent=hp_noise(proc_a, Lc_a),
                H_absent=hallucination(proc_a, d["clean_a"], Lc_a),
                psnr=ps,
                ssim=ss,
            )
        )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/results.csv")
    ap.add_argument("--set", default="main", choices=list(SETS))
    a = ap.parse_args()
    NAMES[:] = SETS[a.set]

    frame_rows, seen = [], set()
    for r in csv.DictReader(open(ROOT / "data/manifest.csv")):
        if r["frame"] not in seen:
            seen.add(r["frame"])
            frame_rows.append(r)
    specs = sample_specs(frame_rows, a.n, seed=a.seed)
    t0 = time.time()
    out = []
    with Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap(one_spec, specs, chunksize=2), 1):
            out.extend(rows)
            if i % 20 == 0:
                print(f"  {i}/{len(specs)} scenes, {time.time() - t0:.0f}s", flush=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    print(f"wrote {len(out)} rows ({len(out) // len(NAMES)} scenes) to {a.out} in {time.time() - t0:.0f}s")
