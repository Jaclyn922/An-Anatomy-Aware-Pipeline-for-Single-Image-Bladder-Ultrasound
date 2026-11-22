"""Step 3e: summary table and figures from outputs/results.csv.

Detection uses a difference-of-Gaussians blob observer (chosen after checking it on raw images only) and the matched design: for every scene we have a PRESENT and an ABSENT image rendered with the same
anatomy and acquisition settings but different noise. The score is the two-alternative forced-choice accuracy
  2AFC = mean over scenes of [ S_present > S_absent ]   (ties count 0.5),   0.5 = chance, 1 = always right.
(Pooling all scenes into one ROC would mix scene-to-scene differences in noise level with the device signal.)

Confidence intervals: cluster bootstrap over SOURCE FRAMES (scenes on the same frame share anatomy),
1000 resamples, 95% percentile interval.

Outputs: outputs/summary.csv, outputs/sweep.csv, outputs/results_figure.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ["raw", "nlm_global", "srad_global", "srad_lumen", "roi_full"]
LABEL = {"raw": "raw", "nlm_global": "global NLM", "srad_global": "global SRAD", "srad_lumen": "lumen-only SRAD", "roi_full": "ROI method",
         "roi_i10": "ROI 10 it", "roi_i100": "ROI 100 it", "roi_i200": "ROI 200 it"}
COLOR = {"raw": "#7f7f7f", "nlm_global": "#1f77b4", "srad_global": "#ff7f0e", "srad_lumen": "#2ca02c", "roi_full": "#d62728"}
NB = 1000


def boot_median(df, col, seed=0):
    rng = np.random.default_rng(seed)
    groups = {f: g[col].values for f, g in df.groupby("frame")}
    keys = list(groups)
    vals = []
    for _ in range(NB):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        vals.append(np.nanmedian(np.concatenate([groups[keys[i]] for i in pick])))
    return float(np.nanmedian(df[col])), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_mean(df, values, seed=0):
    """values: per-row array aligned with df. Cluster bootstrap of the mean via per-frame sums."""
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"frame": df["frame"].values, "v": values}).groupby("frame")["v"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    vals = []
    for _ in range(NB):
        pick = rng.integers(0, len(s), size=len(s))
        vals.append(s[pick].sum() / c[pick].sum())
    return float(np.mean(values)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def afc(d):
    p, n = d["S_dog_present"].values, d["S_dog_absent"].values
    return np.where(p > n, 1.0, np.where(p == n, 0.5, 0.0))


def main():
    df = pd.read_csv(ROOT / "outputs/results.csv")
    raw = df[df.condition == "raw"].set_index("spec_id")
    df["dH"] = df["H_absent"] - df["spec_id"].map(raw["H_absent"])
    df["noise_cut"] = 1.0 - df["lumen_hp_std_absent"] / df["spec_id"].map(raw["lumen_hp_std_absent"])  # share of fine lumen noise removed

    afc_raw = pd.Series(afc(df[df.condition == "raw"]), index=df[df.condition == "raw"]["spec_id"].values)
    print("Paired change in 2AFC detection vs raw (same scenes; negative = device harder to detect):")
    delta = {}
    for c in MAIN[1:] + ["roi_i10", "roi_i100", "roi_i200"]:
        d = df[df.condition == c]
        delta[c] = boot_mean(d, afc(d) - afc_raw.loc[d["spec_id"].values].values, seed=2)
        print(f"  {LABEL[c]:>16}: {delta[c][0]:+.3f} [{delta[c][1]:+.3f}, {delta[c][2]:+.3f}]")
    print()
    rows = []
    print(f"{'condition':>16} | {'retention':>20} | {'mostly erased':>20} | {'2AFC detection':>20} | {'extra false struct.':>22} | lumen noise removed | PSNR  SSIM")
    for c in MAIN + ["roi_i10", "roi_i100", "roi_i200"]:
        d = df[df.condition == c]
        v = d[d.visible == 1]
        r = boot_median(v, "retention")
        er = boot_mean(v, (v["retention"].values < 0.5).astype(float))
        a = boot_mean(d, afc(d))
        h = boot_median(d, "dH")
        nc = float(np.nanmedian(d.noise_cut))
        dd_ = delta.get(c, (0.0, 0.0, 0.0))
        rows.append(dict(condition=c, d_afc=dd_[0], d_afc_lo=dd_[1], d_afc_hi=dd_[2], retention=r[0], ret_lo=r[1], ret_hi=r[2], erased=er[0], er_lo=er[1], er_hi=er[2],
                         afc=a[0], afc_lo=a[1], afc_hi=a[2], dH=h[0], dH_lo=h[1], dH_hi=h[2], noise_cut=nc,
                         psnr=d.psnr.median(), ssim=d.ssim.median()))
        print(f"{LABEL[c]:>16} | {r[0]:5.2f} [{r[1]:4.2f},{r[2]:4.2f}]    | {er[0]:5.1%} [{er[1]:4.1%},{er[2]:4.1%}] | {a[0]:5.3f} [{a[1]:4.3f},{a[2]:4.3f}] | {h[0]:+6.1f} [{h[1]:+5.1f},{h[2]:+5.1f}] | {nc:17.0%} | {d.psnr.median():5.1f} {d.ssim.median():.3f}")
    S = pd.DataFrame(rows)
    S[S.condition.isin(MAIN)].to_csv(ROOT / "outputs/summary.csv", index=False)
    S[~S.condition.isin(["nlm_global", "srad_global", "srad_lumen"])].to_csv(ROOT / "outputs/sweep.csv", index=False)

    bins = [(-7, 4, "low gain\n(<4 dB)"), (4, 11, "mid gain\n(4-11 dB)"), (11, 19, "high gain\n(>11 dB)")]
    print("\n2AFC detection by gain bin:")
    ab = {}
    for lo, hi, name in bins:
        line = f"{name.replace(chr(10), ' '):>22}: "
        for c in MAIN:
            d = df[(df.condition == c) & (df.gain_offset_db >= lo) & (df.gain_offset_db < hi)]
            ab[(name, c)] = boot_mean(d, afc(d), seed=1)
            line += f"{LABEL[c]} {ab[(name, c)][0]:.3f} | "
        print(line, f"(n={len(d)} scenes)")

    print("\nRetention by device echo strength (visible devices, median):")
    for lo, hi, name in [(0.3, 1.0, "weak (0.3-1)"), (1.0, 2.5, "medium (1-2.5)"), (2.5, 6.01, "strong (2.5-6)")]:
        line = f"{name:>16}: "
        for c in MAIN:
            d = df[(df.condition == c) & (df.visible == 1) & (df.device_gain >= lo) & (df.device_gain < hi)]
            line += f"{LABEL[c]} {np.nanmedian(d.retention):.2f} | "
        print(line)

    print("\nRetention when the device touches the wall vs floats free (ROI method vs lumen-only SRAD):")
    for tw in (False, True):
        for c in ("srad_lumen", "roi_full"):
            d = df[(df.condition == c) & (df.visible == 1) & (df.touch_wall == tw)]
            print(f"  touch_wall={tw!s:5} {LABEL[c]:>16}: median retention {np.nanmedian(d.retention):.3f}  (n={len(d)})")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    x = np.arange(len(MAIN))
    M = S.set_index("condition").loc[MAIN]
    for a_, key, lo, hi, title, yl, ref in [
        (ax[0], "retention", "ret_lo", "ret_hi", "Device contrast kept", "retention (median, 1 = untouched)", 1.0),
        (ax[3], "dH", "dH_lo", "dH_hi", "Extra false structure (no-device images)", "grey levels vs raw (median, lower = safer)", 0.0),
    ]:
        y = M[key].values
        a_.bar(x, y, color=[COLOR[c] for c in MAIN], yerr=[y - M[lo].values, M[hi].values - y], capsize=4)
        a_.set_xticks(x)
        a_.set_xticklabels([LABEL[c] for c in MAIN], rotation=20, fontsize=9)
        a_.set_title(title, fontsize=11)
        a_.set_ylabel(yl, fontsize=9)
        a_.axhline(ref, color="k", lw=0.8)
    for j, (lo, hi, name) in enumerate(bins):
        for i, c in enumerate(MAIN):
            p, l, h = ab[(name, c)]
            ax[1].errorbar(j + (i - 2) * 0.13, p, yerr=[[p - l], [h - p]], fmt="o", color=COLOR[c], capsize=3, label=LABEL[c] if j == 0 else None)
    ax[1].axhline(0.5, color="k", lw=0.8, ls="--")
    ax[1].set_xticks(range(len(bins)))
    ax[1].set_xticklabels([b[2] for b in bins], fontsize=9)
    ax[1].set_title("Device detection (2AFC, 0.5 = chance)", fontsize=11)
    ax[1].legend(fontsize=8, loc="lower left")
    sw = S.set_index("condition").loc[["raw", "roi_i10", "roi_full", "roi_i100", "roi_i200"]]
    ax[2].plot(sw.noise_cut * 100, sw.retention, "o-", color="#d62728")
    for k, (c, r_) in enumerate(sw.iterrows()):
        ax[2].annotate({"raw": "raw", "roi_i10": "10", "roi_full": "30", "roi_i100": "100", "roi_i200": "200"}[c] + (" it" if c != "raw" else ""),
                       (r_.noise_cut * 100, r_.retention), textcoords="offset points", xytext=(5, 6), fontsize=9)
    ax[2].set_xlabel("lumen noise removed (%)")
    ax[2].set_ylabel("device contrast kept (median)")
    ax[2].set_title("ROI method: stronger smoothing vs device", fontsize=11)
    fig.suptitle("Does despeckling hurt device visibility?  600 matched simulated scenes, 95% CI over source frames", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/results_figure.png", dpi=110)
    print("saved outputs/results_figure.png, summary.csv, sweep.csv")


if __name__ == "__main__":
    main()
