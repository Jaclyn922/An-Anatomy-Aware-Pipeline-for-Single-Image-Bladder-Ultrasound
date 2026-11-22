"""Figure: real frame vs simulated frames built on the same lumen shape.

Columns: real | simulated noisy (no device) | simulated clean reference |
         simulated noisy + strong-echo device | simulated noisy + weak-echo device.
Rows: three different frames (one from the fit video, two from other sources); the last row puts the
device against the wall. Output: outputs/sim_vs_real.png
"""
import csv
import dataclasses
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device import make_device_mask  # noqa: E402
from simulate import build_scene, fov_mask, load_calibration, render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    sp, rp = load_calibration(ROOT / "outputs/sim_calibration.json")
    picks = [
        next(r for r in rows if r["video"].startswith("Apache") and 0.05 < float(r["lumen_frac"]) < 0.08),
        next(r for r in rows if not r["video"].startswith("Apache") and 0.03 < float(r["lumen_frac"]) < 0.1),
        [r for r in rows if not r["video"].startswith("Apache") and 0.03 < float(r["lumen_frac"]) < 0.1][-1],
    ]
    titles = ["real frame", "simulated (noisy)", "simulated clean reference", "+ device, strong echo (gain 6)", "+ device, weak echo (gain 0.3)"]
    sp_strong = dataclasses.replace(sp, device_gain=6.0)
    sp_weak = dataclasses.replace(sp, device_gain=0.3, shadow_strength=0.4)

    fig, axes = plt.subplots(3, 5, figsize=(17, 9.5))
    rng = np.random.default_rng(11)
    for i, r in enumerate(picks):
        img = np.array(Image.open(ROOT / "data/roboflow" / r["file"]).convert("L"))
        m = np.array(Image.open(ROOT / "data/masks" / (Path(r["file"]).stem + "_mask.png"))) > 127
        fov = fov_mask(img)
        dev, _ = make_device_mask(m, rng, touch_wall=(i == 2))

        sc0 = build_scene(m, fov, sp, scene_seed=3)
        noisy, clean = render(sc0, noise_seed=7, rp=rp)
        noisy_s, _ = render(build_scene(m, fov, sp_strong, 3, device=dev), 7, rp)
        noisy_w, _ = render(build_scene(m, fov, sp_weak, 3, device=dev), 7, rp)

        ys, xs = np.nonzero(m)
        pad = 30
        y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
        for j, im in enumerate([img / 255.0, noisy, clean, noisy_s, noisy_w]):
            ax = axes[i, j]
            ax.imshow(im[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if j >= 3 and dev is not None:
                ax.contour(dev[y0:y1, x0:x1], levels=[0.5], colors="yellow", linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=11)
        axes[i, 0].set_ylabel(r["video"][:22] + ("\n(device touches wall)" if i == 2 else ""), fontsize=9)
    fig.suptitle("Real vs simulated bladder ultrasound (same lumen shape; yellow outline = device cross-section)", fontsize=12)
    fig.tight_layout()
    out = ROOT / "outputs/sim_vs_real.png"
    fig.savefig(out, dpi=100)
    print("saved", out)


if __name__ == "__main__":
    main()
