"""Do the metrics detect what they claim to? We feed them deliberately wrong 'filters':
  an eraser that removes the device     -> retention must collapse
  a hallucinator that paints a fake blob -> H must jump
  the identity                           -> retention near 1, H small

Run (project root):  python tests/test_metrics.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from degrade import realise, sample_specs  # noqa: E402
from metrics import auc, contrast, detect_stat, hallucination, regions  # noqa: E402
from simulate import load_calibration  # noqa: E402


def frames():
    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    seen, out = set(), []
    for r in rows:
        if r["frame"] not in seen:
            seen.add(r["frame"])
            out.append(r)
    return out


def strong_specs(n):
    specs = sample_specs(frames(), 400, seed=1)
    keep = [s for s in specs if s["device_gain"] > 3.0 and -2 < s["gain_offset_db"] < 8 and s["looks"] == 2 and not s["touch_wall"]]
    return keep[:n]


def test_eraser_and_identity():
    sp, rp = load_calibration(ROOT / "outputs/sim_calibration.json")
    ret_id, ret_er = [], []
    for s in strong_specs(12):
        d = realise(s, sp, rp)
        if d is None:
            continue
        Lc, B = regions(d["lumen"], d["device"])
        if B.sum() < 30:
            continue
        c_clean = contrast(d["clean_p"], d["device"], B)
        if c_clean < 15:
            continue
        ret_id.append(contrast(d["noisy_p"], d["device"], B) / c_clean)
        erased = d["noisy_p"].copy()
        erased[ndi.binary_dilation(d["device"], iterations=3)] = int(np.median(d["noisy_p"][B]))
        ret_er.append(contrast(erased, d["device"], B) / c_clean)
    print(f"retention  : identity median {np.median(ret_id):.2f}, eraser median {np.median(ret_er):.2f} (n={len(ret_id)})")
    assert 0.6 < np.median(ret_id) < 1.5
    assert np.median(ret_er) < 0.2


def test_hallucination_metric():
    sp, rp = load_calibration(ROOT / "outputs/sim_calibration.json")
    h_id, h_fake = [], []
    for s in strong_specs(12):
        d = realise(s, sp, rp)
        if d is None:
            continue
        Lc = ndi.binary_erosion(d["lumen"], iterations=4)
        ys, xs = np.nonzero(Lc)
        k = len(ys) // 2
        fake = d["noisy_a"].astype(float)
        yy, xx = np.mgrid[: fake.shape[0], : fake.shape[1]]
        fake += 120.0 * np.exp(-((yy - ys[k]) ** 2 + (xx - xs[k]) ** 2) / (2 * 2.0**2))
        h_id.append(hallucination(d["noisy_a"], d["clean_a"], Lc))
        h_fake.append(hallucination(np.clip(fake, 0, 255).astype(np.uint8), d["clean_a"], Lc))
    print(f"hallucin.  : identity median H {np.median(h_id):.1f}, painted blob median H {np.median(h_fake):.1f}")
    assert np.median(h_fake) > np.median(h_id) + 20


def test_auc_function():
    assert auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert abs(auc([1, 2, 3], [1, 2, 3]) - 0.5) < 1e-9
    print("auc        : known cases ok")


def test_present_vs_absent_detectable():
    sp, rp = load_calibration(ROOT / "outputs/sim_calibration.json")
    sp_, sa_ = [], []
    for s in strong_specs(20):
        d = realise(s, sp, rp)
        if d is None:
            continue
        Lc = ndi.binary_erosion(d["lumen"], iterations=4)
        sp_.append(detect_stat(d["noisy_p"], Lc))
        sa_.append(detect_stat(d["noisy_a"], Lc))
    a = auc(sp_, sa_)
    print(f"detection  : strong devices, raw images, AUC {a:.2f} (n={len(sp_)})")
    assert a > 0.8


if __name__ == "__main__":
    test_auc_function()
    test_eraser_and_identity()
    test_hallucination_metric()
    test_present_vs_absent_detectable()
    print("all metric tests passed")
