"""Step 7b: can the device be measured automatically, how accurate is it, and can a 1-5 quality grade tell
which frames to trust?

Pipeline tested (one per processing condition):
    frame -> [optional despeckling] -> (a) device segmentation -> length / position
                                       (b) quality grade 1-5 -> accept / reject flag

(a) Segmentation: a per-pixel gradient-boosting classifier (scikit-learn) on multi-scale intensity features inside the
    lumen. A plain threshold rule was tried first and failed in noisy scenes (documented in README), so a learned
    pixel classifier is used. PyTorch could not be installed here; a small CNN/U-Net is the natural upgrade.
(b) Quality grade: a gradient-boosting regressor predicts the device contrast of the CLEAN image from the processed
    image; the prediction is binned into five grades that mirror the five-level protocol of the original quality
    study (1 Poor ... 5 Excellent). Bins (grey levels of clean device contrast): <8, 8-20, 20-40, 40-70, >=70.
    This is a simulation-defined grade, NOT the expert-labelled USQNet grade. A trained USQNet/ResNet grader can be
    swapped in for it (see validate_real.py).

Training scenes come from the Apache video only; test scenes come from the 8 other source videos and use fresh seeds
(not used in any earlier experiment). Each classifier is trained separately on each condition's output images.

Usage (project root):   python src/run_measure.py --ntrain 700 --ntest 320 --workers 2
Output: outputs/measure_results.csv  (one row per test scene and condition)
"""
import argparse
import csv
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS  # noqa: E402
from degrade import realise, sample_specs  # noqa: E402
from measure import dice, skeleton_length, span_px, truth  # noqa: E402
from metrics import contrast, regions  # noqa: E402
from run_iqa import features as scene_features  # noqa: E402
from simulate import load_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SP, RP = load_calibration(ROOT / "outputs/sim_calibration.json")
CONDS = ["raw", "nlm_global", "srad_global", "roi_full", "roi_g100"]
GRADE_EDGES = [8.0, 20.0, 40.0, 70.0]
MIN_BLOB = 15
SUB_PER_SCENE = 2500


def grade_of(c):
    return int(1 + np.searchsorted(GRADE_EDGES, c, side="right"))


def pixel_features(img, lumen):
    """Features for every pixel of the eroded lumen. Returns (Lc mask, [n_pixels, n_features])."""
    f = img.astype(np.float32)
    Lc = ndi.binary_erosion(lumen.astype(bool), iterations=4)
    bg = float(np.median(f[Lc]))
    sc = 1.4826 * float(np.median(np.abs(f[Lc] - bg))) + 1.0
    g = {s: ndi.gaussian_filter(f, s) for s in (1.5, 2.5, 3.5, 5.0, 7.0)}
    dog1, dog2 = g[1.5] - g[5.0], g[2.5] - g[7.0]
    m1 = ndi.uniform_filter(f, 5)
    lstd = np.sqrt(np.maximum(ndi.uniform_filter(f * f, 5) - m1 * m1, 0.0))
    dist = ndi.distance_transform_edt(lumen.astype(bool)).astype(np.float32)
    ridge = []
    for sg in (2.0, 3.0):
        ev = hessian_matrix_eigvals(hessian_matrix(f, sigma=sg, order="rc", use_gaussian_derivatives=False), )
        ridge += [-ev[1] * sg * sg / sc, -ev[0] * sg * sg / sc]  # bright blob / ridge response, scale-normalised
    maps = [*ridge, f - bg, g[1.5] - bg, g[2.5] - bg, g[3.5] - bg, g[5.0] - bg, dog1, dog2, lstd, dist, (f - bg) / sc, dog1 / sc, dog2 / sc,
            np.full_like(f, bg), np.full_like(f, sc)]
    return Lc, np.stack([m[Lc] for m in maps], axis=1).astype(np.float32)


def work(arg):
    spec, keep_full = arg
    d = realise(spec, SP, RP)
    if d is None:
        return None
    Lc0, B = regions(d["lumen"], d["device"])
    if B.sum() < 30:
        return None
    c_clean = float(contrast(d["clean_p"], d["device"], B))
    out = dict(spec_id=spec["spec_id"], frame=spec["frame"], video=spec["video"], c_clean=c_clean, conds={})
    rng = np.random.default_rng(spec["spec_id"] + 99)
    for name in CONDS:
        img = CONDITIONS[name](d["noisy_p"], d["lumen"], d["fov"])
        Lc, X = pixel_features(img, d["lumen"])
        y = d["device"][Lc]
        entry = dict(scene=scene_features(img, d["lumen"], d["fov"]))
        if keep_full:
            entry.update(X=X, Lc=Lc)
        else:
            pos, neg = np.nonzero(y)[0], np.nonzero(~y)[0]
            k = min(SUB_PER_SCENE, len(neg))
            idx = np.concatenate([pos, rng.choice(neg, size=k, replace=False)])
            entry.update(X=X[idx], y=y[idx])
        out["conds"][name] = entry
    if keep_full:
        out["device"], out["lumen"] = d["device"], d["lumen"]
    return out


def postprocess(prob, Lc):
    m = np.zeros(Lc.shape, bool)
    m[Lc] = prob > 0.5
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        m = np.isin(lab, 1 + np.nonzero(sizes >= MIN_BLOB)[0])
    if m.any():
        m = ndi.binary_closing(m, iterations=2) & Lc
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=700)
    ap.add_argument("--ntest", type=int, default=320)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--test_seed", type=int, default=500)
    ap.add_argument("--results", type=Path, default=ROOT / "outputs/measure_results.csv")
    ap.add_argument("--save_dir", type=Path, default=ROOT / "outputs/segmenters")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    seen, frames = set(), []
    for r in rows:
        if r["frame"] not in seen:
            seen.add(r["frame"])
            frames.append(r)
    f_train = [r for r in frames if r["video"].startswith("Apache")]
    f_test = [r for r in frames if not r["video"].startswith("Apache")]
    tr_specs = sample_specs(f_train, a.ntrain, seed=300)
    te_specs = sample_specs(f_test, a.ntest, seed=a.test_seed)
    t0 = time.time()
    with Pool(a.workers) as pool:
        tr = [r for r in pool.imap(work, [(s, False) for s in tr_specs], chunksize=4) if r]
        print(f"train scenes {len(tr)}  {time.time() - t0:.0f}s", flush=True)
        te = [r for r in pool.imap(work, [(s, True) for s in te_specs], chunksize=2) if r]
        print(f"test scenes {len(te)}  {time.time() - t0:.0f}s", flush=True)

    c_tr = np.array([r["c_clean"] for r in tr])
    out_rows = []
    for name in CONDS:
        Xp = np.concatenate([r["conds"][name]["X"] for r in tr])
        yp = np.concatenate([r["conds"][name]["y"] for r in tr])
        seg = HistGradientBoostingClassifier(max_depth=5, max_iter=200, learning_rate=0.08, l2_regularization=1.0, random_state=0).fit(Xp, yp)
        a.save_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(seg, open(a.save_dir / f"segmenter_{name}.pkl", "wb"))  # use with validate_real.py --segmenter
        S_tr = np.stack([r["conds"][name]["scene"] for r in tr])
        grader = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05, l2_regularization=1.0, random_state=0).fit(
            S_tr, np.clip(c_tr, 0, 150))
        for r in te:
            e = r["conds"][name]
            prob = seg.predict_proba(e["X"])[:, 1]
            m = postprocess(prob, e["Lc"])
            t = truth(r["device"])
            det = bool(m.any())
            est = skeleton_length(m) if det else np.nan
            est_span = span_px(m) if det else np.nan
            cy, cx = (np.nonzero(m)[0].mean(), np.nonzero(m)[1].mean()) if det else (np.nan, np.nan)
            pc = float(grader.predict(e["scene"][None])[0])
            out_rows.append(dict(spec_id=r["spec_id"], frame=r["frame"], video=r["video"], condition=name, c_clean=r["c_clean"],
                                 grade_true=grade_of(r["c_clean"]), pred_c=pc, grade_pred=grade_of(pc),
                                 **{f"sf{i}": float(v) for i, v in enumerate(e["scene"])}, true_len=t["length_px"], est_len=est, true_span=t["span_px"], est_span=est_span, detected=int(det), dice=dice(m, r["device"]) if det else 0.0,
                                 centroid_err=float(np.hypot(cy - t["cy"], cx - t["cx"])) if det else np.nan))
        print(f"  {name} done {time.time() - t0:.0f}s", flush=True)
    with open(a.results, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows in {time.time() - t0:.0f}s")
