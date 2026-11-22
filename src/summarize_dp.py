"""Step 8b: tables and figure for the "detect, then protect" ROI filter (outputs/dp_results.csv, outputs/dp_extra.csv).
All intervals: bootstrap over source frames. Paired differences are on the same scenes. Writes outputs/dp_summary.txt, outputs/dp_figure.png
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
TOL = 0.2


def say(s=""):
    print(s)
    LINES.append(s)


def boot_mean(df, col, nb=1000, seed=0, fn=np.nanmean):
    rng = np.random.default_rng(seed)
    groups = {f: g[col].values for f, g in df.groupby("frame")}
    ks = list(groups)
    vals = [fn(np.concatenate([groups[ks[i]] for i in rng.choice(len(ks), len(ks))])) for _ in range(nb)]
    return float(fn(df[col].values)), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def fmt(t, f="{:.2f}"):
    return (f + " [" + f + ", " + f + "]").format(*t)


def main():
    r = pd.read_csv(OUT / "dp_results.csv")
    x = pd.read_csv(OUT / "dp_extra.csv")
    r = r[r.grade_true >= 2].copy()
    r["good"] = ((((r.est_span - r.true_span) / r.true_span).abs() <= TOL) & (r.detected == 1)).astype(float)
    v = x[x.c_clean >= 8.0]
    conds = ["raw", "roi_g100", "roi_g200", "roi_dp100", "roi_dp200", "roi_dp400"]
    say(f"DETECT-THEN-PROTECT: {r.spec_id.nunique()} fresh test scenes (visible devices), {r.video.nunique()} videos, {r.frame.nunique()} frames")
    say("guard-only = ROI v2 (barrier at lumen core); dp = v2 + detected device candidate walled off from the diffusion.")
    say("Detector and measurement models were trained on disjoint Apache frames; thresholds chosen on validation frames.")
    say()
    say(f"{'condition':>10} | {'contrast kept (median)':>24} | {'noise removed':>15} | {'false structure dH':>20} | {'within 20% span':>20} | {'Dice':>18}")
    stat = {}
    for c in conds:
        vx = v[v.condition == c]
        rr = r[r.condition == c]
        ret = boot_mean(vx.dropna(subset=["retention"]), "retention", fn=np.nanmedian)
        nc = boot_mean(vx, "noise_cut", fn=np.nanmedian)
        dh = boot_mean(vx, "dH", fn=np.nanmedian)
        gd = boot_mean(rr, "good")
        dc = boot_mean(rr, "dice")
        stat[c] = dict(ret=ret, nc=nc, dh=dh, gd=gd, dc=dc)
        say(f"{c:>10} | {fmt(ret):>24} | {fmt(nc):>15} | {fmt(dh, '{:+.1f}'):>20} | {fmt(gd):>20} | {fmt(dc):>18}")
    say()
    say("Paired differences on the same scenes (first minus second):")
    wide = r.pivot(index="spec_id", columns="condition")
    fr = r.drop_duplicates("spec_id").set_index("spec_id").frame
    rng = np.random.default_rng(1)

    def paired(a, b, col):
        d = (wide[col][a] - wide[col][b]).dropna()
        f = fr.loc[d.index]
        ks = f.unique()
        grp = {k: d[f == k].values for k in ks}
        bs = [np.concatenate([grp[ks[i]] for i in rng.choice(len(ks), len(ks))]).mean() for _ in range(2000)]
        return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    for a, b in [("roi_dp200", "roi_g200"), ("roi_dp100", "roi_g100"), ("roi_dp200", "raw"), ("roi_g200", "raw"), ("roi_g100", "raw")]:
        say(f"  {a:>10} - {b:<9} | within-20% {fmt(paired(a, b, 'good'), '{:+.3f}')} | Dice {fmt(paired(a, b, 'dice'), '{:+.3f}')}")
    say()
    pr = v[v.condition == "roi_dp200"]
    say(f"Detector quality on test scenes: share of true device pixels inside the protected region (median) {pr.protect_overlap.median():.2f}; "
        f"average false protected pixels in device-free images {pr.false_protect_px.mean():.1f}.")
    (OUT / "dp_summary.txt").write_text("\n".join(LINES))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    its = [100, 200, 400]
    for lab, names, col in (("v2: barrier at lumen core", ["roi_g100", "roi_g200"], "#d62728"), ("v3: detect, then protect", ["roi_dp100", "roi_dp200", "roi_dp400"], "#2ca02c")):
        xs = [int(n[-3:]) for n in names]
        for a_, key, ttl in ((ax[0], "ret", "Device contrast kept (median, 1 = all)"), (ax[1], "nc", "Speckle noise removed in lumen (median)")):
            y = [stat[n][key][0] for n in names]
            lo = [stat[n][key][1] for n in names]
            hi = [stat[n][key][2] for n in names]
            a_.errorbar(xs, y, yerr=[np.array(y) - lo, np.array(hi) - np.array(y)], fmt="o-", color=col, capsize=3, label=lab)
            a_.set_title(ttl, fontsize=10)
        y = [stat[n]["gd"][0] for n in names]
        lo = [stat[n]["gd"][1] for n in names]
        hi = [stat[n]["gd"][2] for n in names]
        ax[2].errorbar(xs, y, yerr=[np.array(y) - lo, np.array(hi) - np.array(y)], fmt="o-", color=col, capsize=3, label=lab)
    ax[2].axhline(stat["raw"]["gd"][0], color="#7f7f7f", ls="--", label="raw")
    ax[2].set_title("Device span measured within 20% of truth", fontsize=10)
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xticks(its)
        a_.set_xticklabels([str(i) for i in its])
        a_.set_xlabel("SRAD iterations")
    ax[0].legend(fontsize=8)
    fig.suptitle("Detect, then protect: strong smoothing without erasing the device", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "dp_figure.png", dpi=110)


if __name__ == "__main__":
    main()
