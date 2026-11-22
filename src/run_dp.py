"""Step 8: "detect, then protect" ROI filter (ROI v3).

Problem found earlier: the guarded ROI filter (v2) stops the wall from bleeding into the lumen, but strong smoothing still wears
away the device itself (400 iterations: 42% of devices nearly erased). Idea: find the device candidate first, then wall it off
from the diffusion so the rest of the lumen can be smoothed hard without touching it.

Pipeline for one frame:  raw frame -> stage-1 detector (raw-image pixel classifier) -> candidate region (lenient threshold, dilated 3 px)
                         -> roi_filter_protected(frame, lumen mask, candidate, iterations) -> measurement.
Both stages are evaluated honestly:
  * stage-1 detector is trained on Apache frames set A only;
  * the stage-2 measurement models (one per processing condition) are trained on Apache frames set B (candidates for B come from a
    detector that never saw B), thresholds are chosen on validation frames V, and everything is scored once on fresh test scenes from
    the 8 other videos. Absent-device scenes get the same detector, so false candidates are paid for (hallucination H is measured).

Conditions: raw, roi_g100, roi_g200 (guard only, v2), roi_dp100, roi_dp200, roi_dp400 (v3).
Endpoints on test scenes: device contrast retention, noise removed in the lumen, false structure H (no-device scenes), and the same
measurement endpoints as before (within 20% span, Dice, centre error).

Usage (project root):  python src/run_dp.py --nA 800 --nB 700 --nval 120 --ntest 300 --workers 2
Outputs: outputs/dp_results.csv (measurement rows), outputs/dp_extra.csv (retention / noise / H rows), outputs/segmenters_dp/*.pkl
"""
import argparse
import csv
import pickle
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_measure as R  # noqa: E402
import run_measure_v3 as V3  # noqa: E402
from degrade import realise, sample_specs  # noqa: E402
from metrics import contrast, hallucination, hp_noise, regions  # noqa: E402
from roi import roi_filter, roi_filter_protected  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SP, RP = R.SP, R.RP
CONDS = ["raw", "roi_g100", "roi_g200", "roi_dp100", "roi_dp200", "roi_dp400"]
TAU_CAND = 0.20
D1 = None


def _init(path):
    global D1
    D1 = pickle.load(open(path, "rb")) if path else None


def candidate(img_raw, lumen):
    """Stage-1: device candidate region (bool) from a raw image."""
    Lc, X = V3.pixel_features3(img_raw, lumen)
    m = V3.post3(D1.predict_proba(X)[:, 1], Lc, lumen, TAU_CAND)
    if not m.any():
        return m
    return ndi.binary_dilation(m, iterations=3) & lumen


def apply_cond(name, img, lumen, fov, cand):
    if name == "raw":
        return img
    n = int(name[-3:])
    if name.startswith("roi_g"):
        return roi_filter(img, lumen, srad_iters=n, guard=True)
    return roi_filter_protected(img, lumen, cand, srad_iters=n)


def work_a(spec):
    d = realise(spec, SP, RP)
    if d is None:
        return None
    Lc, X = V3.pixel_features3(d["noisy_p"], d["lumen"])
    y = d["device"][Lc]
    rng = np.random.default_rng(spec["spec_id"] + 5)
    pos, neg = np.nonzero(y)[0], np.nonzero(~y)[0]
    idx = np.concatenate([pos, rng.choice(neg, size=min(R.SUB_PER_SCENE, len(neg)), replace=False)])
    return X[idx], y[idx]


def work_b(arg):
    spec, mode = arg                       # mode: "train" (subsampled pixels), "val" or "test" (full)
    d = realise(spec, SP, RP)
    if d is None:
        return None
    _, B = regions(d["lumen"], d["device"])
    if B.sum() < 30:
        return None
    c_clean = float(contrast(d["clean_p"], d["device"], B))
    out = dict(spec_id=spec["spec_id"], frame=spec["frame"], video=spec["video"], c_clean=c_clean, conds={}, extra=[])
    rng = np.random.default_rng(spec["spec_id"] + 9)
    cand_p = candidate(d["noisy_p"], d["lumen"])
    cand_a = candidate(d["noisy_a"], d["lumen"]) if mode == "test" else None
    Lc_a = ndi.binary_erosion(d["lumen"], iterations=4)
    raw_hp = hp_noise(d["noisy_a"], Lc_a) if mode == "test" else None
    raw_H = hallucination(d["noisy_a"], d["clean_a"], Lc_a) if mode == "test" else None
    for name in CONDS:
        img = apply_cond(name, d["noisy_p"], d["lumen"], d["fov"], cand_p)
        Lc, X = V3.pixel_features3(img, d["lumen"])
        e = dict(scene=np.zeros(1))
        y = d["device"][Lc]
        if mode == "train":
            pos, neg = np.nonzero(y)[0], np.nonzero(~y)[0]
            idx = np.concatenate([pos, rng.choice(neg, size=min(R.SUB_PER_SCENE, len(neg)), replace=False)])
            e.update(X=X[idx], y=y[idx])
        else:
            e.update(X=X, Lc=Lc)
        out["conds"][name] = e
        if mode == "test":
            proc_a = apply_cond(name, d["noisy_a"], d["lumen"], d["fov"], cand_a)
            out["extra"].append(dict(spec_id=spec["spec_id"], frame=spec["frame"], video=spec["video"], condition=name, c_clean=c_clean,
                                     retention=contrast(img, d["device"], B) / c_clean if c_clean >= 8.0 else np.nan,
                                     noise_cut=1.0 - hp_noise(proc_a, Lc_a) / max(raw_hp, 1e-6),
                                     dH=hallucination(proc_a, d["clean_a"], Lc_a) - raw_H,
                                     protected_px=int(cand_p.sum()) if name.startswith("roi_dp") else 0,
                                     protect_overlap=float((cand_p & d["device"]).sum() / max(d["device"].sum(), 1)) if name.startswith("roi_dp") else np.nan,
                                     false_protect_px=int(cand_a.sum()) if name.startswith("roi_dp") else 0))
    if mode != "train":
        out["device"], out["lumen"] = d["device"], d["lumen"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nA", type=int, default=800)
    ap.add_argument("--nB", type=int, default=700)
    ap.add_argument("--nval", type=int, default=120)
    ap.add_argument("--ntest", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_dir", type=Path, default=ROOT / "outputs/segmenters_dp")
    a = ap.parse_args()
    rows_m = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    seen, frames = set(), []
    for r in rows_m:
        if r["frame"] not in seen:
            seen.add(r["frame"])
            frames.append(r)
    apache = sorted([r for r in frames if r["video"].startswith("Apache")], key=lambda r: r["frame"])
    f_val = apache[::5]
    rest = [r for r in apache if r not in f_val]
    f_A, f_B = rest[::2], rest[1::2]
    f_test = [r for r in frames if not r["video"].startswith("Apache")]
    print(f"frames: A {len(f_A)}, B {len(f_B)}, validation {len(f_val)}, test {len(f_test)}", flush=True)
    t0 = time.time()
    a.save_dir.mkdir(parents=True, exist_ok=True)

    # stage 1: detector trained on A only
    with mp.get_context("spawn").Pool(a.workers) as pool:
        parts = [r for r in pool.imap(work_a, sample_specs(f_A, a.nA, seed=701), chunksize=4) if r]
    d1 = HistGradientBoostingClassifier(max_depth=6, max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=0).fit(
        np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    d1_path = a.save_dir / "stage1_detector.pkl"
    pickle.dump(d1, open(d1_path, "wb"))
    print(f"stage-1 detector trained on {len(parts)} scenes {time.time() - t0:.0f}s", flush=True)

    # stage 2
    # spawn, not fork: the main process has just used OpenMP (gradient boosting), and forked workers would deadlock
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(str(d1_path),)) as pool:
        tr = [r for r in pool.imap(work_b, [(s, "train") for s in sample_specs(f_B, a.nB, seed=702)], chunksize=4) if r]
        print(f"train {len(tr)} scenes {time.time() - t0:.0f}s", flush=True)
        va = [r for r in pool.imap(work_b, [(s, "val") for s in sample_specs(f_val, a.nval, seed=703)], chunksize=2) if r]
        print(f"validation {len(va)} scenes {time.time() - t0:.0f}s", flush=True)
        te = [r for r in pool.imap(work_b, [(s, "test") for s in sample_specs(f_test, a.ntest, seed=704)], chunksize=2) if r]
        print(f"test {len(te)} scenes {time.time() - t0:.0f}s", flush=True)

    extra = [x for r in te for x in r["extra"]]
    with open(ROOT / "outputs/dp_extra.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(extra[0].keys()))
        wr.writeheader()
        wr.writerows(extra)
    out_rows = []
    for name in CONDS:
        seg = HistGradientBoostingClassifier(max_depth=6, max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=0).fit(
            np.concatenate([r["conds"][name]["X"] for r in tr]), np.concatenate([r["conds"][name]["y"] for r in tr]))
        pickle.dump(seg, open(a.save_dir / f"segmenter_{name}.pkl", "wb"))
        score = {tau: float(np.mean([x["dice"] for x in V3.evaluate_rows(seg, None, tau, name, va)])) for tau in V3.TAUS}
        tau = max(score, key=score.get)
        print(f"  {name}: validation mean Dice by tau {({k: round(v, 3) for k, v in score.items()})} -> tau {tau}  {time.time() - t0:.0f}s", flush=True)
        out_rows += V3.evaluate_rows(seg, None, tau, name, te)
    with open(ROOT / "outputs/dp_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"wrote {len(out_rows)} + {len(extra)} rows in {time.time() - t0:.0f}s")
