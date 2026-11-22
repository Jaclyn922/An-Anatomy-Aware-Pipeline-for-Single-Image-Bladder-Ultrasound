"""Compare raw vs cleaned masks: numbers plus outputs/mask_cleaning.png (zoomed, side by side).

Usage (from the project root):  python src/compare_masks.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import perimeter

ROOT = Path(__file__).resolve().parents[1]


def load(folder, file):
    return np.array(Image.open(ROOT / folder / (Path(file).stem + "_mask.png"))) > 127


def main():
    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    area_chg, per_ratio, comps_raw, shift = [], [], [], []
    for r in rows:
        raw, cl = load("data/masks_raw", r["file"]), load("data/masks", r["file"])
        area_chg.append(cl.sum() / raw.sum() - 1.0)
        per_ratio.append(perimeter(cl) / perimeter(raw))
        comps_raw.append(ndi.label(raw)[1])
        # how far the boundary moved: max distance from any raw pixel to the cleaned mask and back
        d_to_cl = ndi.distance_transform_edt(~cl)[raw].max()
        d_to_raw = ndi.distance_transform_edt(~raw)[cl].max()
        shift.append(max(d_to_cl, d_to_raw))
    area_chg, per_ratio, shift = map(np.array, (area_chg, per_ratio, shift))
    print(f"files: {len(rows)}")
    print(f"area change      : mean {area_chg.mean():+.2%}, worst {area_chg[np.abs(area_chg).argmax()]:+.2%}")
    print(f"boundary length  : cleaned / raw = {per_ratio.mean():.2f} on average (lower = smoother)")
    print(f"max boundary move: median {np.median(shift):.1f} px, worst {shift.max():.1f} px")
    print(f"raw masks with more than one piece: {sum(c > 1 for c in comps_raw)}")

    # figure: the 3 frames whose raw boundary was longest relative to its area (roughest)
    rough = [perimeter(load("data/masks_raw", r["file"])) / np.sqrt(load("data/masks_raw", r["file"]).sum()) for r in rows]
    picks, seen = [], set()
    for i in np.argsort(rough)[::-1]:
        if rows[i]["frame"] not in seen:
            seen.add(rows[i]["frame"])
            picks.append(rows[i])
        if len(picks) == 3:
            break
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax_row, r in zip(axes, picks):
        img = np.array(Image.open(ROOT / "data/roboflow" / r["file"]).convert("L"))
        raw, cl = load("data/masks_raw", r["file"]), load("data/masks", r["file"])
        ys, xs = np.nonzero(raw)
        y0, y1, x0, x1 = max(0, ys.min() - 8), ys.max() + 8, max(0, xs.min() - 8), xs.max() + 8
        ax_row[0].imshow(raw[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
        ax_row[0].set_title("raw mask", fontsize=9)
        ax_row[1].imshow(cl[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
        ax_row[1].set_title("cleaned mask", fontsize=9)
        ax_row[2].imshow(img[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
        ax_row[2].contour(raw[y0:y1, x0:x1], levels=[0.5], colors="r", linewidths=0.8)
        ax_row[2].contour(cl[y0:y1, x0:x1], levels=[0.5], colors="c", linewidths=0.8)
        ax_row[2].set_title("red = raw, cyan = cleaned", fontsize=9)
        ax_row[0].set_ylabel(r["frame"].split("-")[-1], fontsize=9)
        for a in ax_row:
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout()
    out = ROOT / "outputs" / "mask_cleaning.png"
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
