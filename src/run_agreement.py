"""Step 9a: technical performance of the measurement tool in the vocabulary of quantitative imaging biomarkers
(bias, repeatability, linearity; QIBA-style metrology, e.g. Raunig et al. 2015, Stat Methods Med Res).

Design: every test scene is imaged K times with different noise realisations (same device, same anatomy, new speckle), each image is
processed by each condition and measured. This gives
  * bias and limits of agreement against the true span (Bland & Altman 1986),
  * repeatability: within-scene spread across the K repeat images (within-subject CV, repeatability coefficient RC = 2.77 * wSD),
  * linearity: does the measured span track the true span,
  * failure rate: how often nothing is found.
Models: the segmenters trained by run_dp.py (trained on Apache frames; this script uses the 8 other videos with new seeds).

Usage (project root, after run_dp.py):   python src/run_agreement.py --n 150 --k 4 --workers 2
Output: outputs/agreement_rows.csv
"""
import argparse
import csv
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dp as D  # noqa: E402
import run_measure as R  # noqa: E402
import run_measure_v3 as V3  # noqa: E402
from degrade import realise, sample_specs  # noqa: E402
from measure import span_px, truth  # noqa: E402
from metrics import contrast, regions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {}
TAUS = {}


def _init(dir_, taus):
    D.D1 = pickle.load(open(Path(dir_) / "stage1_detector.pkl", "rb"))
    for c in D.CONDS:
        MODELS[c] = pickle.load(open(Path(dir_) / f"segmenter_{c}.pkl", "rb"))
    TAUS.update(taus)


def work(arg):
    spec, k = arg
    rng = np.random.default_rng(spec["spec_id"] + 31)
    seeds = [spec["noise_seed_present"]] + [int(x) for x in rng.integers(0, 2**31 - 1, size=k - 1)]
    rows = []
    for ki, s in enumerate(seeds):
        d = realise(dict(spec, noise_seed_present=int(s)), D.SP, D.RP)
        if d is None:
            return []
        _, B = regions(d["lumen"], d["device"])
        if B.sum() < 30:
            return []
        c_clean = float(contrast(d["clean_p"], d["device"], B))
        t = truth(d["device"])
        cand = D.candidate(d["noisy_p"], d["lumen"])
        for name in D.CONDS:
            img = D.apply_cond(name, d["noisy_p"], d["lumen"], d["fov"], cand)
            Lc, X = V3.pixel_features3(img, d["lumen"])
            m = V3.post3(MODELS[name].predict_proba(X)[:, 1], Lc, d["lumen"], TAUS[name])
            rows.append(dict(spec_id=spec["spec_id"], frame=spec["frame"], video=spec["video"], condition=name, k=ki, c_clean=c_clean,
                             true_span=t["span_px"], est_span=span_px(m) if m.any() else np.nan, detected=int(m.any())))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model_dir", type=Path, default=ROOT / "outputs/segmenters_dp")
    a = ap.parse_args()
    res = pd.read_csv(ROOT / "outputs/dp_results.csv")
    taus = {c: float(res[res.condition == c].tau.iloc[0]) for c in D.CONDS}
    seen, frames = set(), []
    for r in csv.DictReader(open(ROOT / "data/manifest.csv")):
        if r["frame"] not in seen:
            seen.add(r["frame"])
            frames.append(r)
    f_test = [r for r in frames if not r["video"].startswith("Apache")]
    specs = sample_specs(f_test, a.n, seed=705)
    t0, out = time.time(), []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(str(a.model_dir), taus)) as pool:
        for i, rows in enumerate(pool.imap(work, [(s, a.k) for s in specs], chunksize=2), 1):
            out.extend(rows)
            if i % 25 == 0:
                print(f"  {i}/{len(specs)} scenes {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(out).to_csv(ROOT / "outputs/agreement_rows.csv", index=False)
    print(f"wrote {len(out)} rows in {time.time() - t0:.0f}s")
