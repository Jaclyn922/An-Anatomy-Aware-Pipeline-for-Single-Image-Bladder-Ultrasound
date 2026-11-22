"""Experiment A summary: noise-versus-edge frontier on real frames.

Primary pre-specified comparison: at the SAME amount of lumen smoothing (std_removed = 0.20, 0.30, 0.40), how much wall edge is kept
(edge_keep) by each family? Per frame, each family's sweep is linearly interpolated at the target smoothing (only if the sweep reaches it).
Paired differences ROI-v2 minus family, bootstrap over source frames (2000 resamples). Secondary: same with noise_removed.
Writes outputs/realA_summary.txt and outputs/realA_figure.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LINES = []
FAMS = ["gauss_global", "bilateral_global", "nlm_global", "srad_global", "roi_v1", "roi_v2", "roi_nlm"]
COL = dict(gauss_global="#8c564b", bilateral_global="#9467bd", nlm_global="#ff7f0e", srad_global="#d62728", roi_v1="#1f77b4", roi_v2="#2ca02c", roi_nlm="#17becf")


def say(s=""):
    print(s)
    LINES.append(s)


def at_target(g, xcol, ycol, target):
    g = g.sort_values(xcol)
    x, y = g[xcol].values, g[ycol].values
    if x.max() < target or x.min() > target:
        return np.nan
    # first crossing, linear interpolation on the monotone envelope
    xs = np.maximum.accumulate(x)
    return float(np.interp(target, xs, y))


def main():
    d = pd.read_csv(OUT / "realA_rows.csv")
    say(f"REAL-FRAME NOISE-VS-EDGE FRONTIER: {d.frame.nunique()} frames from {d.video.nunique()} videos "
        f"({d[d.video.str.startswith('Apache')].frame.nunique()} Apache frames, {d[~d.video.str.startswith('Apache')].frame.nunique()} others)")
    say("edge_keep and tissue_keep are processed/raw; std_removed and noise_removed = 1 - processed/raw lumen spread (higher = smoother).")
    say()
    say("Median over frames for every setting:")
    say(d.groupby(["family", "strength"])[["std_removed", "noise_removed", "edge_keep", "tissue_keep"]].median().round(2).to_string())
    say()
    rng = np.random.default_rng(0)
    frames = d.frame.unique()
    for xcol in ("std_removed", "noise_removed"):
        say(f"MATCHED-SMOOTHING COMPARISON on {xcol} (edge_keep at equal smoothing; ROI-v2 minus other family; paired per frame)")
        for target in (0.2, 0.3, 0.4):
            tab = {}
            for fam in FAMS:
                tab[fam] = {f: at_target(g, xcol, "edge_keep", target) for f, g in d[d.family == fam].groupby("frame")}
            tab = pd.DataFrame(tab)
            line = f"  target {target:.1f}: " + " | ".join(f"{fam} {tab[fam].mean():.2f} (n={tab[fam].notna().sum()})" for fam in FAMS)
            say(line)
            for fam in FAMS[:-1]:
                dd = (tab["roi_v2"] - tab[fam]).dropna()
                if len(dd) < 10:
                    say(f"      roi_v2 - {fam:<17}: sweep does not reach {target:.1f} in most frames (n={len(dd)})")
                    continue
                bs = [dd.iloc[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
                say(f"      roi_v2 - {fam:<17}: {dd.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] (n={len(dd)})")
        say()
    (OUT / "realA_summary.txt").write_text("\n".join(LINES))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for fam in FAMS:
        g = d[d.family == fam].groupby("strength")[["std_removed", "noise_removed", "edge_keep", "tissue_keep"]].median().sort_index()
        ax[0].plot(g.std_removed, g.edge_keep, "o-", color=COL[fam], label=fam)
        ax[1].plot(g.noise_removed, g.edge_keep, "o-", color=COL[fam], label=fam)
        ax[2].plot(g.std_removed, g.tissue_keep, "o-", color=COL[fam], label=fam)
    ax[0].set_xlabel("lumen smoothing: total spread removed (median)")
    ax[0].set_ylabel("wall edge kept (1 = same as raw)")
    ax[0].set_title("Smoothing vs wall edge (up and right is better)", fontsize=10)
    ax[1].set_xlabel("lumen smoothing: fine-scale noise removed (median)")
    ax[1].set_title("Same, with fine-scale noise as the x-axis", fontsize=10)
    ax[2].set_xlabel("lumen smoothing: total spread removed (median)")
    ax[2].set_ylabel("outer tissue texture kept")
    ax[2].set_title("Surrounding tissue texture (1 = untouched)", fontsize=10)
    for a in ax:
        a.axhline(1.0, color="#bbbbbb", lw=0.8)
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=8)
    fig.suptitle("Real bladder frames: how much lumen speckle each filter removes, and what it costs", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "realA_figure.png", dpi=120)


if __name__ == "__main__":
    main()
