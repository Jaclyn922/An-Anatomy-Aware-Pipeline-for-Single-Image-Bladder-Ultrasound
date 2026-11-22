"""Make two sanity-check figures:
  outputs/mask_check.png     real frames with their lumen mask contours (one frame per source)
  outputs/device_preview.png the 3D pretzel centre-line and random plane slices placed in a real lumen
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import make_device_mask, pretzel_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load(row):
    img = np.array(Image.open(ROOT / "data/roboflow" / row["file"]).convert("L"))
    m = np.array(Image.open(ROOT / "data/masks" / (Path(row["file"]).stem + "_mask.png"))) > 127
    return img, m


def crop_box(mask, pad=12):
    ys, xs = np.nonzero(mask)
    return max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad


def mask_check(rows):
    seen, picks = set(), []
    for r in rows:
        if r["frame"] not in seen and len(picks) < 12:
            seen.add(r["frame"])
            picks.append(r)
    # make sure other sources are included
    for r in rows:
        if r["video"] not in {p["video"] for p in picks} and len(picks) < 12:
            picks.append(r)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, r in zip(axes.ravel(), picks):
        img, m = load(r)
        ax.imshow(img, cmap="gray")
        ax.contour(m, levels=[0.5], colors="r", linewidths=1)
        ax.set_title(r["video"][:26] + "\n" + r["frame"].split("-")[-1], fontsize=8)
        ax.axis("off")
    fig.suptitle("Roboflow frames with lumen mask (red)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "mask_check.png", dpi=110)
    plt.close(fig)


def device_preview(rows):
    row = next(r for r in rows if r["video"].startswith("Apache") and 0.06 < float(r["lumen_frac"]) < 0.09)
    img, m = load(row)
    y0, y1, x0, x1 = crop_box(m)
    rng = np.random.default_rng(7)

    fig = plt.figure(figsize=(14, 7))
    ax3 = fig.add_subplot(2, 4, 1, projection="3d")
    c = pretzel_curve()
    ax3.plot(c[:, 0], c[:, 1], c[:, 2], lw=2, color="goldenrod")
    ax3.set_title("3D centre-line (pretzel-like 8)", fontsize=9)
    ax3.set_box_aspect((2, 1.2, 0.6))

    slots = [(2, False), (3, False), (4, False), (5, True), (6, True), (7, True), (8, True)]
    for pos, touch in slots:
        ax = fig.add_subplot(2, 4, pos)
        dev, p = make_device_mask(m, rng, touch_wall=touch)
        ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        if dev is not None:
            ov = np.zeros(dev[y0:y1, x0:x1].shape + (4,))
            ov[dev[y0:y1, x0:x1]] = (1.0, 0.85, 0.0, 0.9)
            ax.imshow(ov, interpolation="nearest")
            ax.set_title(f"{'touching wall' if touch else 'free'} | {p['area_px']} px", fontsize=9)
        ax.contour(m[y0:y1, x0:x1], levels=[0.5], colors="c", linewidths=0.7)
        ax.axis("off")
    fig.suptitle(f"Device slices placed in a real lumen ({row['video']}); cyan = lumen mask, yellow = device cross-section", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "device_preview.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    mask_check(rows)
    device_preview(rows)
    print("saved", OUT / "mask_check.png", "and", OUT / "device_preview.png")
