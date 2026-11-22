"""Same frame, different filters (+ wall zoom) and the method-flow schematic. Uses outputs/cache_filtered.npz (hand-drawn masks)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
C = np.load(OUT / "cache_filtered.npz")
rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
seen, files = set(), []
for r in rows:
    if r["frame"] not in seen:
        seen.add(r["frame"]); files.append(r)
conds = [("raw", "raw"), ("nlm", "global NLM"), ("srad100", "global SRAD 100"), ("roi30", "course ROI (3 regions)"), ("roi_nlm2", "lumen-only NLM")]
# pick frames: largest lumen areas from 3 different videos
by = {}
for r in files:
    m = C["mask|" + r["file"]] > 0
    by.setdefault(r["video"], []).append((m.sum(), r))
picks = [max(v, key=lambda t: t[0])[1] for v in list(by.values())[:3]]
fig, ax = plt.subplots(len(picks) * 2, len(conds), figsize=(3.2 * len(conds), 3.2 * len(picks) * 2))
for i, r in enumerate(picks):
    m = C["mask|" + r["file"]] > 0
    ys, xs = np.where(ndi.binary_dilation(m, iterations=12) & ~ndi.binary_erosion(m, iterations=12))
    yb = np.where(m)[0]
    # zoom box on the wall: around the lumen top boundary centre
    cy, cx = int(np.median(ys)), int(np.median(xs))
    top = np.where(m.any(1))[0].min(); cx = int(np.median(np.where(m[top + 3])[0])) if m[top + 3].any() else cx
    y0, x0 = max(top - 30, 0), max(cx - 40, 0)
    for j, (k, name) in enumerate(conds):
        img = C[f"{k}|{r['file']}"]
        a = ax[2 * i, j]; a.imshow(img, cmap="gray", vmin=0, vmax=255)
        a.contour(m, levels=[0.5], colors="#ffcc00", linewidths=0.7)
        a.set_xticks([]); a.set_yticks([]); a.set_title(name if i == 0 else "", fontsize=11)
        a.add_patch(plt.Rectangle((x0, y0), 80, 60, fill=False, ec="#00e5ff", lw=1))
        b = ax[2 * i + 1, j]; b.imshow(img[y0:y0 + 60, x0:x0 + 80], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        b.set_xticks([]); b.set_yticks([])
    ax[2 * i, 0].set_ylabel(r["video"][:22], fontsize=8); ax[2 * i + 1, 0].set_ylabel("zoom: wall", fontsize=8)
fig.suptitle("Same frames, different filters. Yellow = hand-drawn lumen outline; cyan box = zoom on the bladder wall", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "qualitative_figure.png", dpi=110); plt.close(fig)

# schematic
fig, a = plt.subplots(figsize=(13, 3.4)); a.axis("off"); a.set_xlim(0, 13); a.set_ylim(0, 3.4)
boxes = [("1  Find the lumen\n(segmenter or hand mask)", 0.2), ("2  Erode 3 px\n= lumen core", 3.4), ("3  NLM only inside core\nh = 2 x noise sigma", 6.6), ("4  Feather 1.5 px seam,\nwall + tissue untouched", 9.8)]
for t, x in boxes:
    a.add_patch(plt.Rectangle((x, 1.0), 2.9, 1.5, fc="#e8f1fb", ec="#1f4e79", lw=1.5)); a.text(x + 1.45, 1.75, t, ha="center", va="center", fontsize=10)
for x in (3.1, 6.3, 9.5):
    a.annotate("", xy=(x + 0.3, 1.75), xytext=(x, 1.75), arrowprops=dict(arrowstyle="->", lw=1.8))
a.text(6.5, 0.4, "Hypothesis: strong smoothing only where tissue is uniform removes speckle, keeps edges, and does not hurt automatic segmentation", ha="center", fontsize=10)
a.text(6.5, 3.0, "Anatomy-aware despeckling: where to filter, with what, and how much", ha="center", fontsize=13, weight="bold")
fig.savefig(OUT / "method_flow.png", dpi=120, bbox_inches="tight")
