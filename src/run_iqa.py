"""Step 6b: IQA stability. Does despeckling make a no-reference quality score more stable, and does it stay sensitive?

Quality target (objective, no human rating): device visibility index = contrast of the device in the CLEAN
reference image, in grey levels (clipped to 0..150, scaled to 0..1). It is a property of the scene and the
acquisition settings, not of the noise.

Model: for every processing condition, a separate gradient-boosting regressor (scikit-learn) is trained on the
condition's output images, as the original project did (denoise, then classify). Features are hand-made
(lumen/wall/tissue levels, fine noise, blob statistics). The original used a ResNet; PyTorch could not be installed in
this environment, so this is a stated deviation.

Split by SOURCE VIDEO: train on scenes built on the Apache video, test on scenes built on the 8 other sources.

Stability: every test scene is rendered with K different noise seeds. Within-scene spread of the predictions is the
instability; the share of total variance that is between scenes (ICC) combines stability and sensitivity.

Usage (project root):  python src/run_iqa.py --ntrain 1200 --ntest 240 --k 4
Outputs: outputs/iqa_predictions.csv, outputs/iqa_summary.csv, outputs/iqa_figure.png
"""
import argparse
import csv
import dataclasses
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS  # noqa: E402
from degrade import realise, sample_specs  # noqa: E402
from metrics import contrast, regions  # noqa: E402
from roi import build_rois  # noqa: E402
from simulate import load_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SP, RP = load_calibration(ROOT / "outputs/sim_calibration.json")
IQA_CONDS = ["raw", "nlm_global", "srad_global", "roi_full", "roi_i100"]


def features(img, lumen, fov):
    f = img.astype(np.float64)
    core, wall, outer = build_rois(lumen, 3, 3)
    Lc = ndi.binary_erosion(lumen, iterations=4)
    d_out = ndi.distance_transform_edt(~ndi.binary_dilation(lumen, iterations=3))
    tis = ndi.binary_erosion(outer & fov & (d_out < 25), iterations=2)
    hp = f - ndi.gaussian_filter(f, 3.0)
    dog = ndi.gaussian_filter(f, 1.5) - ndi.gaussian_filter(f, 5.0)
    lap = ndi.laplace(f)
    lm, tm = f[core].mean(), f[tis].mean() if tis.sum() > 20 else f[fov].mean()
    blobs = ndi.label((dog > 8) & Lc)[1]
    out = [
        lm, np.percentile(f[core], 95), hp[Lc].std(), np.abs(hp[Lc]).max(),
        np.percentile(f[wall], 90), f[wall].mean(), tm, hp[tis].std() if tis.sum() > 20 else 0.0,
        dog[Lc].max(), np.percentile(dog[Lc], 99), np.percentile(dog[Lc], 95), np.clip(dog[Lc], 0, None).mean(), blobs,
        lap[fov].var(), *np.percentile(f[fov], [10, 50, 90, 99]),
        tm - lm, np.percentile(f[wall], 90) - lm,
    ]
    return np.array(out, dtype=np.float64)


def render_k(spec, seeds):
    """Present images for several noise seeds; returns (list of images, clean contrast, geometry) or None."""
    out = []
    d0 = None
    for s in seeds:
        sp = dict(spec, noise_seed_present=int(s))
        d = realise(sp, SP, RP)
        if d is None:
            return None
        out.append(d["noisy_p"])
        d0 = d
    Lc, B = regions(d0["lumen"], d0["device"])
    if B.sum() < 30:
        return None
    y = float(np.clip(contrast(d0["clean_p"], d0["device"], B), 0.0, 150.0) / 150.0)
    return out, y, d0["lumen"], d0["fov"]


def work(arg):
    spec, k = arg
    rng = np.random.default_rng(spec["noise_seed_present"])
    seeds = [spec["noise_seed_present"]] + [int(x) for x in rng.integers(0, 2**31 - 1, size=k - 1)]
    r = render_k(spec, seeds)
    if r is None:
        return None
    imgs, y, lumen, fov = r
    feats = np.stack([[features(CONDITIONS[c](im, lumen, fov), lumen, fov) for c in IQA_CONDS] for im in imgs])  # [k, cond, feat]
    return dict(spec_id=spec["spec_id"], frame=spec["frame"], video=spec["video"], y=y, X=feats)


def icc1(pred):
    """pred [scenes, k]. One-way random-effects ICC(1): share of variance that is between scenes."""
    n, k = pred.shape
    gm = pred.mean()
    ms_b = k * ((pred.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_w = ((pred - pred.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    den = ms_b + (k - 1) * ms_w
    return float((ms_b - ms_w) / den) if den > 0 else float("nan")


def stats(pred, y):
    within = float(pred.std(axis=1, ddof=1).mean())
    total = float(pred.std())
    pm = pred.mean(1)
    rho = float(spearmanr(pm, y).statistic) if pm.std() > 0 else float("nan")
    return dict(within_sd=within, total_sd=total, instability=(within / total) if total > 0 else float("nan"), icc=icc1(pred), spearman=rho)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=1200)
    ap.add_argument("--ntest", type=int, default=240)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    seen, frames = set(), []
    for r in rows:
        if r["frame"] not in seen:
            seen.add(r["frame"])
            frames.append(r)
    f_train = [r for r in frames if r["video"].startswith("Apache")]
    f_test = [r for r in frames if not r["video"].startswith("Apache")]
    print(f"train frames {len(f_train)} (Apache) | test frames {len(f_test)} from {len({r['video'] for r in f_test})} other videos", flush=True)
    tr_specs = sample_specs(f_train, a.ntrain, seed=100)
    te_specs = sample_specs(f_test, a.ntest, seed=200)

    t0 = time.time()
    with Pool(a.workers) as pool:
        tr = [r for r in pool.imap(work, [(s, 1) for s in tr_specs], chunksize=4) if r]
        print(f"  train features done: {len(tr)} scenes, {time.time() - t0:.0f}s", flush=True)
        te = [r for r in pool.imap(work, [(s, a.k) for s in te_specs], chunksize=2) if r]
        print(f"  test features done: {len(te)} scenes x {a.k} seeds, {time.time() - t0:.0f}s", flush=True)

    Ytr = np.array([r["y"] for r in tr])
    Yte = np.array([r["y"] for r in te])
    Xte = np.stack([r["X"] for r in te])  # [scenes, k, cond, feat]
    print(f"visibility index: train median {np.median(Ytr):.2f}, test median {np.median(Yte):.2f}", flush=True)

    summary, pred_rows = [], []
    boot_frames = np.array([r["frame"] for r in te])
    ufr = np.unique(boot_frames)
    for ci, c in enumerate(IQA_CONDS):
        Xtr = np.stack([r["X"][0, ci] for r in tr])
        model = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05, l2_regularization=1.0, random_state=0)
        model.fit(Xtr, Ytr)
        pred = np.stack([model.predict(Xte[:, k, ci]) for k in range(Xte.shape[1])], axis=1)  # [scenes, k]
        s = stats(pred, Yte)
        rng = np.random.default_rng(0)
        bs = {m: [] for m in ("instability", "icc", "spearman")}
        for _ in range(500):
            pick = rng.choice(ufr, size=len(ufr), replace=True)
            idx = np.concatenate([np.nonzero(boot_frames == f)[0] for f in pick])
            if len(idx) < 10:
                continue
            sb = stats(pred[idx], Yte[idx])
            for m in bs:
                bs[m].append(sb[m])
        row = dict(condition=c, **s)
        for m in bs:
            row[m + "_lo"], row[m + "_hi"] = float(np.nanpercentile(bs[m], 2.5)), float(np.nanpercentile(bs[m], 97.5))
        summary.append(row)
        for i, r in enumerate(te):
            for k in range(pred.shape[1]):
                pred_rows.append(dict(condition=c, spec_id=r["spec_id"], frame=r["frame"], seed_idx=k, y=r["y"], pred=float(pred[i, k])))
        print(f"{c:>12}: instability {s['instability']:.3f} [{row['instability_lo']:.3f},{row['instability_hi']:.3f}] | "
              f"ICC {s['icc']:.3f} [{row['icc_lo']:.3f},{row['icc_hi']:.3f}] | Spearman {s['spearman']:.3f} [{row['spearman_lo']:.3f},{row['spearman_hi']:.3f}]", flush=True)

    with open(ROOT / "outputs/iqa_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        wr.writerows(summary)
    with open(ROOT / "outputs/iqa_predictions.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        wr.writeheader()
        wr.writerows(pred_rows)
    print(f"done in {time.time() - t0:.0f}s")
