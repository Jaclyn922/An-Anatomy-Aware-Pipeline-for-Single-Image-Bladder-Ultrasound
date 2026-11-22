"""Step 7d: validate automatic measurement and the quality gate on REAL images with a reference length.

Inputs (you provide, e.g. after approval from your mentor):
  a CSV with columns:  image_path, mask_path, reference_span_mm, pixel_mm
      image_path          8-bit grayscale frame (png/jpg)
      mask_path           binary lumen mask for that frame (white = lumen)
      reference_span_mm   independently measured device span (caliper, photograph, other modality)
      pixel_mm            pixel spacing of the scanner for that frame (mm per pixel)
  --segmenter pkl         optional: a pickled scikit-learn classifier trained with run_measure.py features;
                          if omitted the simple threshold baseline from measure.py is used.
Outputs: per-frame errors, mean bias, 95% limits of agreement (Bland-Altman), share within 20%, and a Bland-Altman plot.

Usage:  python src/validate_real.py --csv my_reference_measurements.csv --out outputs/real_validation
Nothing here is a clinical claim; it is the script that turns the simulation study into a real validation.
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import measure_device, span_px  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs/real_validation")
    ap.add_argument("--segmenter", default=None)
    a = ap.parse_args()
    seg = pickle.load(open(a.segmenter, "rb")) if a.segmenter else None
    df = pd.read_csv(a.csv)
    rows = []
    for _, r in df.iterrows():
        img = np.array(Image.open(r.image_path).convert("L"))
        lumen = np.array(Image.open(r.mask_path)) > 127
        if seg is None:
            m = measure_device(img, lumen)
            est_px = span_px(m["mask"]) if m["detected"] else np.nan
        else:
            from run_measure import pixel_features, postprocess
            Lc, X = pixel_features(img, lumen)
            m = postprocess(seg.predict_proba(X)[:, 1], Lc)
            est_px = span_px(m) if m.any() else np.nan
        rows.append(dict(image=r.image_path, ref_mm=r.reference_span_mm, est_mm=est_px * r.pixel_mm))
    out = pd.DataFrame(rows)
    out["diff_mm"] = out.est_mm - out.ref_mm
    ok = out.dropna()
    bias, sd = ok.diff_mm.mean(), ok.diff_mm.std(ddof=1)
    within = float(((ok.diff_mm.abs() / ok.ref_mm) <= 0.2).sum() / len(out))
    print(f"frames {len(out)}, measured {len(ok)}")
    print(f"bias {bias:+.2f} mm, 95% limits of agreement [{bias - 1.96 * sd:+.2f}, {bias + 1.96 * sd:+.2f}] mm")
    print(f"within 20% of reference: {within:.2f} (not-measured frames count as failures)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(str(a.out) + ".csv", index=False)
    mean = (ok.est_mm + ok.ref_mm) / 2
    plt.figure(figsize=(5, 4))
    plt.scatter(mean, ok.diff_mm)
    for v, ls in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
        plt.axhline(v, color="k", ls=ls, lw=0.8)
    plt.xlabel("mean of estimate and reference (mm)")
    plt.ylabel("estimate - reference (mm)")
    plt.title("Bland-Altman")
    plt.tight_layout()
    plt.savefig(str(a.out) + "_bland_altman.png", dpi=120)


if __name__ == "__main__":
    main()
