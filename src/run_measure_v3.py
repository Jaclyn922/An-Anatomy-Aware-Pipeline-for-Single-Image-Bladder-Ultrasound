"""Step 7e (v3): a stronger measurement model, tuned only on validation frames, scored once on fresh test scenes.

Changes relative to run_measure.py (all decided before looking at the v3 test scenes):
  1. 1600 training scenes instead of 900; a deeper gradient-boosting model.
  2. Extra context features: very coarse smoothing (sigma 10, 14) and the position of each pixel relative to the lumen.
  3. Post-processing: smooth the probability map, keep the dominant cluster of device pixels plus nearby clusters, drop far-away
     blobs (a false blob far from the device used to inflate the measured span).
  4. The probability threshold tau is chosen per condition on VALIDATION frames (10 Apache frames that are NOT used for
     training); the test videos never influence it.
Test scenes: the 8 other videos, new seed (600). Same endpoints as before, plus localisation (centre within 8 px).

Usage (project root):  python src/run_measure_v3.py --ntrain 1600 --nval 200 --ntest 400 --workers 2
Output: outputs/measure_results_v3.csv, outputs/segmenters_v3/*.pkl
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_measure as R  # noqa: E402
from conditions import CONDITIONS  # noqa: E402
from degrade import realise, sample_specs  # noqa: E402
from measure import dice, skeleton_length, span_px, truth  # noqa: E402
from metrics import contrast, regions  # noqa: E402
from run_iqa import features as scene_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SP, RP, CONDS = R.SP, R.RP, R.CONDS
TAUS = (0.10, 0.15, 0.20, 0.25, 0.35, 0.45)


def pixel_features3(img, lumen):
    Lc, X = R.pixel_features(img, lumen)
    f = img.astype(np.float32)
    bg = float(np.median(f[Lc]))
    ys, xs = np.nonzero(lumen)
    cy, cx = ys.mean(), xs.mean()
    ext = max(ys.max() - ys.min(), xs.max() - xs.min(), 1.0)
    yy, xx = np.nonzero(Lc)
    extra = [ndi.gaussian_filter(f, 10.0)[Lc] - bg, ndi.gaussian_filter(f, 14.0)[Lc] - bg,
             (yy - cy) / ext, (xx - cx) / ext, np.hypot(yy - cy, xx - cx) / ext]
    return Lc, np.concatenate([X, np.stack(extra, 1).astype(np.float32)], axis=1)


def work3(arg):
    spec, keep_full = arg
    d = realise(spec, SP, RP)
    if d is None:
        return None
    _, B = regions(d["lumen"], d["device"])
    if B.sum() < 30:
        return None
    out = dict(spec_id=spec["spec_id"], frame=spec["frame"], video=spec["video"], c_clean=float(contrast(d["clean_p"], d["device"], B)), conds={})
    rng = np.random.default_rng(spec["spec_id"] + 77)
    for name in CONDS:
        img = CONDITIONS[name](d["noisy_p"], d["lumen"], d["fov"])
        Lc, X = pixel_features3(img, d["lumen"])
        y = d["device"][Lc]
        e = dict(scene=scene_features(img, d["lumen"], d["fov"]))
        if keep_full:
            e.update(X=X, Lc=Lc)
        else:
            pos, neg = np.nonzero(y)[0], np.nonzero(~y)[0]
            idx = np.concatenate([pos, rng.choice(neg, size=min(R.SUB_PER_SCENE, len(neg)), replace=False)])
            e.update(X=X[idx], y=y[idx])
        out["conds"][name] = e
    if keep_full:
        out["device"], out["lumen"] = d["device"], d["lumen"]
    return out


def post3(prob, Lc, lumen, tau):
    P = np.zeros(Lc.shape, np.float32)
    P[Lc] = prob
    P = ndi.gaussian_filter(P, 1.0)
    m = (P > tau) & Lc
    lab, n = ndi.label(m)
    if n == 0:
        return m
    idx = np.arange(1, n + 1)
    area = ndi.sum(m, lab, idx)
    mass = ndi.sum(P, lab, idx)
    ok = area >= R.MIN_BLOB
    if not ok.any():
        return np.zeros_like(m)
    cents = np.array(ndi.center_of_mass(m, lab, idx))
    main = int(np.argmax(np.where(ok, mass, -1)))
    eq_diam = 2.0 * np.sqrt(lumen.sum() / np.pi)
    keep = [i for i in range(n) if ok[i] and np.hypot(*(cents[i] - cents[main])) <= 0.5 * eq_diam]
    m = np.isin(lab, [i + 1 for i in keep])
    return ndi.binary_closing(m, iterations=2) & Lc


def evaluate_rows(seg, grader, tau, name, scenes):
    rows = []
    for r in scenes:
        e = r["conds"][name]
        m = post3(seg.predict_proba(e["X"])[:, 1], e["Lc"], r["lumen"], tau)
        t = truth(r["device"])
        det = bool(m.any())
        if det:
            ys, xs = np.nonzero(m)
            cy, cx = ys.mean(), xs.mean()
        pc = float(grader.predict(e["scene"][None])[0]) if grader is not None else np.nan
        rows.append(dict(spec_id=r["spec_id"], frame=r["frame"], video=r["video"], condition=name, c_clean=r["c_clean"],
                         grade_true=R.grade_of(r["c_clean"]), pred_c=pc, grade_pred=R.grade_of(pc) if grader is not None else 0,
                         **{f"sf{i}": float(v) for i, v in enumerate(e["scene"])},
                         true_len=t["length_px"], est_len=skeleton_length(m) if det else np.nan, true_span=t["span_px"],
                         est_span=span_px(m) if det else np.nan, detected=int(det), dice=dice(m, r["device"]) if det else 0.0,
                         centroid_err=float(np.hypot(cy - t["cy"], cx - t["cx"])) if det else np.nan, tau=tau))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=1600)
    ap.add_argument("--nval", type=int, default=200)
    ap.add_argument("--ntest", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--test_seed", type=int, default=600)
    ap.add_argument("--results", type=Path, default=ROOT / "outputs/measure_results_v3.csv")
    ap.add_argument("--save_dir", type=Path, default=ROOT / "outputs/segmenters_v3")
    a = ap.parse_args()
    rows_m = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    seen, frames = set(), []
    for r in rows_m:
        if r["frame"] not in seen:
            seen.add(r["frame"])
            frames.append(r)
    apache = sorted([r for r in frames if r["video"].startswith("Apache")], key=lambda r: r["frame"])
    f_val = apache[::5]
    f_train = [r for r in apache if r not in f_val]
    f_test = [r for r in frames if not r["video"].startswith("Apache")]
    print(f"train frames {len(f_train)}, validation frames {len(f_val)}, test frames {len(f_test)}", flush=True)
    tr_specs = sample_specs(f_train, a.ntrain, seed=301)
    va_specs = sample_specs(f_val, a.nval, seed=302)
    te_specs = sample_specs(f_test, a.ntest, seed=a.test_seed)
    t0 = time.time()
    with Pool(a.workers) as pool:
        tr = [r for r in pool.imap(work3, [(s, False) for s in tr_specs], chunksize=4) if r]
        print(f"train {len(tr)} scenes {time.time() - t0:.0f}s", flush=True)
        va = [r for r in pool.imap(work3, [(s, True) for s in va_specs], chunksize=2) if r]
        print(f"validation {len(va)} scenes {time.time() - t0:.0f}s", flush=True)
        te = [r for r in pool.imap(work3, [(s, True) for s in te_specs], chunksize=2) if r]
        print(f"test {len(te)} scenes {time.time() - t0:.0f}s", flush=True)

    c_tr = np.array([r["c_clean"] for r in tr])
    a.save_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for name in CONDS:
        Xp = np.concatenate([r["conds"][name]["X"] for r in tr])
        yp = np.concatenate([r["conds"][name]["y"] for r in tr])
        seg = HistGradientBoostingClassifier(max_depth=6, max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=0).fit(Xp, yp)
        pickle.dump(seg, open(a.save_dir / f"segmenter_{name}.pkl", "wb"))
        # threshold chosen on validation frames only
        score = {}
        for tau in TAUS:
            score[tau] = float(np.mean([x["dice"] for x in evaluate_rows(seg, None, tau, name, va)]))
        tau = max(score, key=score.get)
        print(f"  {name}: validation mean Dice by tau {({k: round(v, 3) for k, v in score.items()})} -> tau {tau}", flush=True)
        grader = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05, l2_regularization=1.0, random_state=0).fit(
            np.stack([r["conds"][name]["scene"] for r in tr]), np.clip(c_tr, 0, 150))
        out_rows += evaluate_rows(seg, grader, tau, name, te)
        print(f"  {name} done {time.time() - t0:.0f}s", flush=True)
    with open(a.results, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows in {time.time() - t0:.0f}s")
