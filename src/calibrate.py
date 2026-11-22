"""Compare simulated frames with real frames using the same measurements on both.

Measured per frame, all in 8-bit display units so real and simulated are directly comparable:
  lumen   : mean grey level in the eroded lumen core
  wall    : 90th percentile grey level in the wall band (peak of the bright wall line)
  tissue  : mean grey level of tissue within `near` px outside the wall band, inside the fan
  grain   : std of the high-passed tissue texture (how strong the speckle looks)
  corr_x / corr_y : speckle size = lag (px) at which the high-passed autocorrelation drops below 0.5

Usage (project root):  python src/calibrate.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi import build_rois  # noqa: E402
from simulate import RenderParams, SceneParams, build_scene, fov_mask, render, to_u8  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def autocorr_width(x, axis):
    """Lag (px) where the normalised autocorrelation of a zero-mean patch first falls below 0.5."""
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * x.shape[axis], axis=axis)
    ac = np.fft.irfft(f * np.conj(f), axis=axis)
    ac = ac.mean(axis=1 - axis)
    ac = ac[: x.shape[axis] // 2] / ac[0]
    below = np.nonzero(ac < 0.5)[0]
    if not below.size:
        return float(len(ac))
    i = int(below[0])  # >= 1 because ac[0] == 1
    return float((i - 1) + (ac[i - 1] - 0.5) / (ac[i - 1] - ac[i]))  # linear sub-pixel interpolation


def measure(img_u8, lumen, near=25):
    fov = fov_mask(img_u8)
    core, wall, outer = build_rois(lumen, wall_in=3, wall_out=3)
    core = ndi.binary_erosion(core, iterations=2)
    d_out = ndi.distance_transform_edt(~ndi.binary_dilation(lumen, iterations=3))
    tis = outer & fov & (d_out < near)
    tis = ndi.binary_erosion(tis, iterations=2)
    if core.sum() < 30 or wall.sum() < 30 or tis.sum() < 200:
        return None
    img = img_u8.astype(float)
    hp = img - ndi.gaussian_filter(img, 3.0)
    ys, xs = np.nonzero(tis)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = np.where(tis, hp, 0.0)[y0:y1, x0:x1]
    return dict(
        lumen=img[core].mean(),
        wall=np.percentile(img[wall], 90),
        tissue=img[tis].mean(),
        grain=hp[tis].std(),
        corr_x=autocorr_width(patch, 1),
        corr_y=autocorr_width(patch, 0),
    )


def load_frame(row):
    img = np.array(Image.open(ROOT / "data/roboflow" / row["file"]).convert("L"))
    m = np.array(Image.open(ROOT / "data/masks" / (Path(row["file"]).stem + "_mask.png"))) > 127
    return img, m


def real_stats(rows):
    per = {}
    for r in rows:
        img, m = load_frame(r)
        s = measure(img, m)
        if s:
            per.setdefault(r["frame"], []).append(s)
    keys = ["lumen", "wall", "tissue", "grain", "corr_x", "corr_y"]
    frames = [{k: np.median([s[k] for s in v]) for k in keys} for v in per.values()]
    return {k: (np.median([f[k] for f in frames]), np.percentile([f[k] for f in frames], 25), np.percentile([f[k] for f in frames], 75)) for k in keys}, len(frames)


def sim_stats(rows, sp, rp, seeds=(0, 1, 2)):
    out = []
    for r in rows:
        img, m = load_frame(r)
        fov = fov_mask(img)
        for s in seeds:
            sc = build_scene(m, fov, sp, scene_seed=s)
            noisy, _ = render(sc, noise_seed=100 + s, rp=rp)
            st = measure(to_u8(noisy), m)
            if st:
                out.append(st)
    keys = ["lumen", "wall", "tissue", "grain", "corr_x", "corr_y"]
    return {k: np.median([o[k] for o in out]) for k in keys}


def report(rows, sp, rp, title):
    real, nfr = real_stats(rows)
    sim = sim_stats(rows, sp, rp)
    print(f"\n{title}   (real: {nfr} source frames; median [IQR])")
    print(f"{'measure':>8} | {'real':>22} | {'sim':>8}")
    for k in ["lumen", "wall", "tissue", "grain", "corr_x", "corr_y"]:
        m, q1, q3 = real[k]
        print(f"{k:>8} | {m:8.1f} [{q1:6.1f},{q3:6.1f}] | {sim[k]:8.1f}")
    return real, sim


KEYS = ["lumen", "wall", "tissue", "grain", "corr_x", "corr_y"]


def params_from_vector(v, looks=1, floor_u8=0.0, jitter_u8=0.8):
    gain, dr, wl, plat, pax, tl = v
    # lumen_floor is fixed very low: the lumen is shown at the scanner's black level (floor_u8), not as speckle
    sp = SceneParams(lumen_floor=0.003, wall_level=float(np.exp(wl)), tissue_level=float(np.exp(tl)))
    rp = RenderParams(gain_db=float(gain), dr_db=float(dr), psf_lat_px=float(plat), psf_ax_px=float(pax),
                      looks=int(looks), floor_u8=float(floor_u8), jitter_u8=float(jitter_u8))
    return sp, rp


def fit(fit_rows, target, scale, x0, looks, floor_u8, seeds=(0,)):
    from scipy.optimize import minimize

    def loss(v):
        gain, dr, wl, plat, pax, tl = v
        if dr < 20 or dr > 140 or plat < 0.4 or pax < 0.4 or plat > 8 or pax > 8:
            return 1e6
        sp, rp = params_from_vector(v, looks, floor_u8)
        st = sim_stats(fit_rows, sp, rp, seeds=seeds)
        # the lumen level is set directly by floor_u8, so it is not part of the fitted loss
        return sum(((st[k] - target[k]) / scale[k]) ** 2 for k in KEYS if k != "lumen")

    res = minimize(loss, x0, method="Nelder-Mead", options=dict(maxiter=260, xatol=1e-2, fatol=1e-2))
    return res


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--looks", type=int, default=2)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    apache = [r for r in rows if r["video"].startswith("Apache")]
    other = [r for r in rows if not r["video"].startswith("Apache")]
    print(f"fit set: {len({r['frame'] for r in apache})} Apache frames | held-out: {len({r['frame'] for r in other})} frames from {len({r['video'] for r in other})} other sources")

    real_fit, _ = real_stats(apache)
    target = {k: real_fit[k][0] for k in KEYS}
    scale = {k: max((real_fit[k][2] - real_fit[k][1]) / 1.35, 1.0 if k in ("lumen", "wall", "tissue", "grain") else 0.25) for k in KEYS}

    # subsample the fit set (one file per frame, every 3rd frame) so each evaluation is fast
    seen, sub = set(), []
    for r in apache:
        if r["frame"] not in seen:
            seen.add(r["frame"])
            sub.append(r)
    sub = sub[::3]
    floor_u8 = target["lumen"]
    x0 = [-41.0, 62.0, np.log(1.0), 3.0, 1.5, np.log(0.25)]
    res = fit(sub, target, scale, x0, args.looks, floor_u8)
    sp, rp = params_from_vector(res.x, args.looks, floor_u8)
    print("\nfitted:", dict(gain_db=round(rp.gain_db, 1), dr_db=round(rp.dr_db, 1), looks=rp.looks, floor_u8=round(rp.floor_u8, 1),
                          wall_level=round(sp.wall_level, 3), tissue_level=round(sp.tissue_level, 4), psf_lat_px=round(rp.psf_lat_px, 2), psf_ax_px=round(rp.psf_ax_px, 2)), " loss", round(res.fun, 2))
    report(apache, sp, rp, "FIT SET (Apache video)")
    report(other, sp, rp, "HELD-OUT sources (not used for fitting)")
    with open(ROOT / f"outputs/sim_calibration_looks{args.looks}.json", "w") as f:
        json.dump(dict(gain_db=rp.gain_db, dr_db=rp.dr_db, lumen_floor=sp.lumen_floor, wall_level=sp.wall_level,
                       tissue_level=sp.tissue_level, psf_lat_px=rp.psf_lat_px, psf_ax_px=rp.psf_ax_px,
                       looks=rp.looks, floor_u8=rp.floor_u8, jitter_u8=rp.jitter_u8, fit_loss=float(res.fun)), f, indent=2)
