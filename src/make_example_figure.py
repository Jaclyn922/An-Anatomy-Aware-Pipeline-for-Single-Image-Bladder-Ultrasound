"""One scene through the processing conditions, plus residual maps of the no-device image.

Top row   : image WITH the device (yellow outline = ground truth): raw, global NLM, ROI 30 it, ROI 200 it
Bottom row: image WITHOUT the device: raw, ROI 200 it, then (processed - clean reference) for raw and for ROI 200 it.
            Red = brighter than the truth. Structure in these maps that is not noise is false structure.
Output: outputs/example_conditions.png   (also prints real-vs-simulated lumen noise)
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS  # noqa: E402
from degrade import realise, sample_specs  # noqa: E402
from metrics import hp_noise  # noqa: E402
from simulate import load_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    sp, rp = load_calibration(ROOT / "outputs/sim_calibration.json")
    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    seen, fr = set(), []
    for r in rows:
        if r["frame"] not in seen:
            seen.add(r["frame"])
            fr.append(r)
    spec = next(s for s in sample_specs(fr, 600, seed=0)
                if 2.0 < s["device_gain"] < 5.0 and 4.0 < s["gain_offset_db"] < 10.0 and s["looks"] == 2 and not s["touch_wall"])
    d = realise(spec, sp, rp)
    lumen, fov, dev = d["lumen"], d["fov"], d["device"]
    ys, xs = np.nonzero(lumen)
    pad = 25
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
    sl = (slice(y0, y1), slice(x0, x1))

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    top = [("raw", "raw"), ("nlm_global", "global NLM"), ("roi_full", "ROI method, 30 it"), ("roi_i200", "ROI method, 200 it")]
    for a, (c, title) in zip(ax[0], top):
        im = CONDITIONS[c](d["noisy_p"], lumen, fov)
        a.imshow(im[sl], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        a.contour(dev[sl], levels=[0.5], colors="yellow", linewidths=0.8)
        a.set_title(title + "  (device present)", fontsize=10)
    raw_a = d["noisy_a"]
    roi200_a = CONDITIONS["roi_i200"](raw_a, lumen, fov)
    clean_a = d["clean_a"].astype(float)
    ax[1, 0].imshow(raw_a[sl], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax[1, 0].set_title("raw  (no device)", fontsize=10)
    ax[1, 1].imshow(roi200_a[sl], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax[1, 1].set_title("ROI 200 it  (no device)", fontsize=10)
    for a, im, t in [(ax[1, 2], raw_a, "raw - clean reference"), (ax[1, 3], roi200_a, "ROI 200 it - clean reference")]:
        res = ndi.gaussian_filter(im.astype(float) - clean_a, 1.5)
        h = a.imshow(res[sl], cmap="coolwarm", vmin=-60, vmax=60, interpolation="nearest")
        a.contour(lumen[sl], levels=[0.5], colors="k", linewidths=0.6)
        a.set_title(t + "  (red = brighter than truth)", fontsize=10)
    fig.colorbar(h, ax=ax[1, 3], fraction=0.046)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"One scene: device echo gain {spec['device_gain']:.1f}, gain offset {spec['gain_offset_db']:+.0f} dB, {spec['looks']} looks", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/example_conditions.png", dpi=100)
    print("saved outputs/example_conditions.png")

    # real vs simulated fine noise inside the lumen core (grey levels)
    real, sim = [], []
    for r in fr:
        img = np.array(Image.open(ROOT / "data/roboflow" / r["file"]).convert("L"))
        m = np.array(Image.open(ROOT / "data/masks" / (Path(r["file"]).stem + "_mask.png"))) > 127
        core = ndi.binary_erosion(m, iterations=5)
        if core.sum() > 50:
            real.append(hp_noise(img, core))
    for s in sample_specs(fr, 120, seed=3):
        if abs(s["gain_offset_db"]) < 1.5:
            dd = realise(s, sp, rp)
            if dd is not None:
                sim.append(hp_noise(dd["noisy_a"], ndi.binary_erosion(dd["lumen"], iterations=5)))
    print(f"fine noise in lumen (grey levels, std of high-passed image): real median {np.median(real):.2f} "
          f"[{np.percentile(real, 25):.2f}, {np.percentile(real, 75):.2f}] over {len(real)} frames | "
          f"simulated at calibrated gain median {np.median(sim):.2f} (n={len(sim)})")


if __name__ == "__main__":
    main()
