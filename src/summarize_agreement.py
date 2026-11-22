"""Step 9b: bias, limits of agreement, repeatability, linearity and failure rate per processing condition.

Definitions (relative, so no pixel size is needed):
  relative difference  e = (estimate - truth) / truth
  bias                 mean of e;  95% limits of agreement = bias +/- 1.96 SD(e)   (Bland & Altman 1986)
  wCV                  within-scene coefficient of variation across the K repeat images
                       = sqrt(mean over scenes of var_within / mean^2);  RC% = 2.77 * wCV  (95% repeatability coefficient)
  ICC(1)               share of variance of the measured span that is between scenes (one-way random effects)
  linearity            slope and R^2 of scene-mean estimate against truth
  failure rate         images where nothing was found
Only devices that are truly visible (clean contrast >= 8 grey levels) are used. Statistics use images where the device was found;
scenes need at least 3 of K successful measurements for the repeatability numbers. CIs: bootstrap over source frames.
Writes outputs/agreement_summary.txt and outputs/agreement_figure.png
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


def say(s=""):
    print(s)
    LINES.append(s)


def stats(d):
    d = d[d.detected == 1]
    if len(d) < 10:
        return dict(bias=np.nan, loa_lo=np.nan, loa_hi=np.nan, wcv=np.nan, icc=np.nan, slope=np.nan, r2=np.nan)
    e = (d.est_span - d.true_span) / d.true_span
    g = d.groupby("spec_id")
    n_ok = g.est_span.count()
    keep = n_ok[n_ok >= 3].index
    dd = d[d.spec_id.isin(keep)]
    gm = dd.groupby("spec_id").est_span
    mean, var = gm.mean(), gm.var(ddof=1)
    wcv = float(np.sqrt(np.mean(var / mean**2))) if len(mean) > 3 else np.nan
    k = dd.groupby("spec_id").size().mean()
    ms_w = float(var.mean())
    ms_b = float(k * ((mean - mean.mean()) ** 2).sum() / max(len(mean) - 1, 1))
    icc = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w) if (ms_b + (k - 1) * ms_w) > 0 else np.nan
    tr = dd.groupby("spec_id").true_span.first().loc[mean.index]
    slope, icpt = np.polyfit(tr.values, mean.values, 1) if len(mean) > 3 else (np.nan, np.nan)
    r2 = float(np.corrcoef(tr.values, mean.values)[0, 1] ** 2) if len(mean) > 3 else np.nan
    return dict(bias=float(e.mean()), loa_lo=float(e.mean() - 1.96 * e.std()), loa_hi=float(e.mean() + 1.96 * e.std()),
                wcv=wcv, icc=float(icc), slope=float(slope), r2=r2)


def boot(df, key, nb=500, seed=0):
    rng = np.random.default_rng(seed)
    groups = {f: g for f, g in df.groupby("frame")}
    ks = list(groups)
    vals = []
    for _ in range(nb):
        sub = pd.concat([groups[ks[i]] for i in rng.choice(len(ks), len(ks))])
        v = stats(sub)[key]
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    df = pd.read_csv(OUT / "agreement_rows.csv")
    df = df[df.c_clean >= 8.0]
    conds = list(df.condition.unique())
    say(f"AGREEMENT AND REPEATABILITY: {df.spec_id.nunique()} visible-device scenes x {df.k.nunique()} repeat images, {df.frame.nunique()} source frames, {df.video.nunique()} videos")
    say(f"{'condition':>10} | {'failure rate':>12} | {'bias (rel.)':>20} | {'95% limits of agreement':>26} | {'wCV':>20} | {'RC (95%)':>9} | {'ICC(1)':>7} | {'slope':>6} | {'R^2':>5}")
    res = {}
    for c in conds:
        d = df[df.condition == c]
        s = stats(d)
        b = boot(d, "bias")
        w = boot(d, "wcv")
        fail = 1.0 - d.detected.mean()
        res[c] = s
        say(f"{c:>10} | {fail:12.2f} | {s['bias']:+.2f} [{b[0]:+.2f},{b[1]:+.2f}] | [{s['loa_lo']:+.2f}, {s['loa_hi']:+.2f}]".ljust(80) +
            f" | {s['wcv']:.2f} [{w[0]:.2f},{w[1]:.2f}] | {2.77 * s['wcv']:9.2f} | {s['icc']:7.2f} | {s['slope']:6.2f} | {s['r2']:5.2f}")
    say()
    say("Reading guide: wCV is the typical run-to-run spread of the tool on the SAME device. If human readers measuring the same")
    say("frames disagree with themselves by less than this wCV, the tool is not more repeatable than a reader; this number is the bar")
    say("to compare with a real reader study (it is not a claim about human readers).")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for c, col in (("raw", "#7f7f7f"), ("roi_g100", "#1f77b4"), ("roi_dp200", "#2ca02c")):
        if c not in conds:
            continue
        d = df[(df.condition == c) & (df.detected == 1)]
        mean = (d.est_span + d.true_span) / 2
        e = (d.est_span - d.true_span) / d.true_span
        ax[0].scatter(mean, e, s=6, alpha=0.25, color=col, label=c)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("mean of estimate and truth (px)")
    ax[0].set_ylabel("relative difference (estimate - truth)/truth")
    ax[0].set_title("Bland-Altman (each dot = one image)", fontsize=10)
    ax[0].legend(fontsize=8)
    x = np.arange(len(conds))
    ax[1].bar(x, [res[c]["wcv"] for c in conds], color="#4c78a8")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(conds, rotation=25, fontsize=9)
    ax[1].set_title("Repeatability: within-scene CV (lower = better)", fontsize=10)
    for c, col in (("raw", "#7f7f7f"), ("roi_g100", "#1f77b4"), ("roi_dp200", "#2ca02c")):
        if c not in conds:
            continue
        d = df[(df.condition == c) & (df.detected == 1)].groupby("spec_id").agg(t=("true_span", "first"), m=("est_span", "mean"))
        ax[2].scatter(d.t, d.m, s=10, alpha=0.5, color=col, label=c)
    lim = [df.true_span.min(), df.true_span.max()]
    ax[2].plot(lim, lim, "k--", lw=0.8)
    ax[2].set_xlabel("true span (px)")
    ax[2].set_ylabel("mean measured span (px)")
    ax[2].set_title("Linearity", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.suptitle("Technical performance of the automatic span measurement (simulation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "agreement_figure.png", dpi=110)
    (OUT / "agreement_summary.txt").write_text("\n".join(LINES))


if __name__ == "__main__":
    main()
