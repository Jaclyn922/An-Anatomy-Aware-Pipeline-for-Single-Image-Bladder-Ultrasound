"""Experiment M (real frames): what happens when the mask that guides the ROI filter is wrong?

The filter is guided by a DELIBERATELY WRONG mask (dilated or eroded by k px, or shifted by 4 px), but the result is scored against the
hand-drawn mask (the best available truth): lumen smoothing, wall-edge kept, outer texture kept (same definitions as experiment A).
Filters: lumen-only NLM (h = 2) and the guarded ROI v2 (SRAD 100). Output: outputs/maskrob_rows.csv, outputs/maskrob_summary.txt, maskrob_figure.png
Usage (project root):  nice python src/run_real_maskrob.py
"""
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi import build_rois, roi_filter  # noqa: E402
from roi_nlm import nlm_full, roi_nlm  # noqa: E402
from run_real_A import hp, lumen_std, max_slope, profile  # noqa: E402
from simulate import fov_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PERT = [("exact", 0, 0, 0), ("erode 6", -6, 0, 0), ("erode 3", -3, 0, 0), ("dilate 3", 3, 0, 0), ("dilate 6", 6, 0, 0),
        ("shift x 4", 0, 4, 0), ("shift y 4", 0, 0, 4)]


def perturb(mask, k, dx, dy):
    m = mask.astype(bool)
    if k > 0:
        m = ndi.binary_dilation(m, iterations=k)
    elif k < 0:
        m = ndi.binary_erosion(m, iterations=-k)
    return np.roll(m, (dy, dx), (0, 1))


def main():
    seen, frames = set(), []
    for r in csv.DictReader(open(ROOT / "data/manifest.csv")):
        if r["frame"] not in seen:
            seen.add(r["frame"])
            frames.append(r)
    rows, t0 = [], time.time()
    for i, row in enumerate(frames, 1):
        img = np.array(Image.open(ROOT / "data/roboflow" / row["file"]).convert("L"))
        mask = np.array(Image.open(ROOT / "data/masks" / (Path(row["file"]).stem + "_mask.png"))) > 127
        fov = fov_mask(img)
        core, wall, outer = build_rois(mask, 3, 3)
        Lc = ndi.binary_erosion(mask, iterations=5)
        d_out = ndi.distance_transform_edt(~ndi.binary_dilation(mask, iterations=3))
        tis = ndi.binary_erosion(outer & fov & (d_out < 25), iterations=2)
        if Lc.sum() < 50 or tis.sum() < 200:
            continue
        sd = ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)
        s0, n0, u0, t_0 = max_slope(profile(img, sd)), float(hp(img)[Lc].std()), lumen_std(img, Lc), float(hp(img)[tis].std())
        sm = nlm_full(img, 2.0)
        for name, k, dx, dy in PERT:
            pm = perturb(mask, k, dx, dy)
            outs = {"lumen-only NLM": roi_nlm(img, pm, 2.0, sm=sm), "ROI v2 (SRAD 100)": roi_filter(img, pm.astype(np.uint8), srad_iters=100, guard=True)}
            for fam, o in outs.items():
                rows.append(dict(frame=row["frame"], video=row["video"], perturbation=name, family=fam,
                                 noise_removed=1.0 - float(hp(o)[Lc].std()) / max(n0, 1e-6),
                                 std_removed=1.0 - lumen_std(o, Lc) / max(u0, 1e-6),
                                 edge_keep=max_slope(profile(o, sd)) / max(s0, 1e-6),
                                 tissue_keep=float(hp(o)[tis].std()) / max(t_0, 1e-6)))
        if i % 10 == 0:
            print(f"  {i}/{len(frames)} {time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "maskrob_rows.csv", index=False)
    g = d.groupby(["family", "perturbation"])[["std_removed", "noise_removed", "edge_keep", "tissue_keep"]].median().round(3)
    order = [p[0] for p in PERT]
    g = g.reindex(pd.MultiIndex.from_product([d.family.unique(), order]))
    txt = ["MASK-ERROR ROBUSTNESS (median over %d real frames; scored against the hand-drawn mask)" % d.frame.nunique(), g.to_string()]
    (OUT / "maskrob_summary.txt").write_text("\n".join(txt))
    print("\n".join(txt))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for fam, col in (("lumen-only NLM", "#17becf"), ("ROI v2 (SRAD 100)", "#2ca02c")):
        for a, key in zip(ax, ("edge_keep", "noise_removed", "tissue_keep")):
            ys = [g.loc[(fam, p), key] for p in order]
            a.plot(range(len(order)), ys, "o-", color=col, label=fam)
            a.set_xticks(range(len(order)))
            a.set_xticklabels(order, rotation=30, fontsize=8)
            a.grid(alpha=0.25)
    ax[0].set_title("Wall edge kept (1 = unchanged)", fontsize=10)
    ax[1].set_title("Fine-scale lumen noise removed", fontsize=10)
    ax[2].set_title("Outer texture kept", fontsize=10)
    ax[0].legend(fontsize=8)
    fig.suptitle("When the guiding mask is wrong (real frames)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "maskrob_figure.png", dpi=120)


if __name__ == "__main__":
    main()
