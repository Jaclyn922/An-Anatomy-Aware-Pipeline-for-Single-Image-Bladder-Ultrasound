"""Experiment B summary: does despeckling change lumen segmentation on unseen videos?  (outputs/realSeg_rows.csv)

Pre-specified rules (written before any result):
  * headline mode = 'predicted' (the ROI filters are guided by the mask the raw-image segmenter found, i.e. what deployment would do);
    'oracle' (hand-drawn mask guides the filter) is reported as the upper bound.
  * metric = Dice per test frame, averaged over seeds; paired difference to RAW on the same frame.
  * non-inferiority margin = 0.02 Dice: a condition is 'not worse than raw' if the lower 95% bound of (condition - raw) is above -0.02.
  * intervals: bootstrap over test frames (2000 resamples). Apache (53 frames, one video, 4 time blocks) and the 14 frames of the other
    8 videos are also reported separately, because the pooled number is dominated by Apache.
  * area: relative area error (pred - true)/true; bias and 95% limits of agreement (mean +/- 1.96 SD), median absolute error.
Writes outputs/realSeg_summary.txt and outputs/realSeg_figure.png
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
MARGIN = 0.02
CONDS = ["raw", "nlm", "srad30", "srad100", "roi30", "roi_v2_100", "roi_nlm2"]
NAMES = dict(raw="raw", nlm="global NLM", srad30="global SRAD 30", srad100="global SRAD 100", roi30="ROI v1 (3 regions, SRAD 30)",
             roi_v2_100="ROI v2 (SRAD 100, barrier)", roi_nlm2="lumen-only NLM (h=2)")


def say(s=""):
    print(s)
    LINES.append(s)


def boot(v, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(nb)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def f(t, p="{:.3f}"):
    return (p + " [" + p + ", " + p + "]").format(*t)


def main():
    d = pd.read_csv(OUT / "realSeg_rows.csv")
    seeds = sorted(d.seed.unique())
    g = d.groupby(["frame", "video", "condition", "mode"], as_index=False)[["dice", "hd95", "area_pred", "area_true"]].mean()
    g["apache"] = g.video.str.startswith("Apache")
    nfr = g.frame.nunique()
    say(f"REAL-DATA LUMEN SEGMENTATION: {nfr} test frames ({g[g.apache].frame.nunique()} Apache, {g[~g.apache].frame.nunique()} other videos), "
        f"4 folds held out by video/time block, {len(seeds)} seed(s) {seeds}, same tiny U-Net for every input.")
    say(f"Non-inferiority margin {MARGIN} Dice, headline mode = predicted mask.")
    say()
    for mode in ("predicted", "oracle"):
        gm = g[g["mode"] == mode]
        wide = gm.pivot(index="frame", columns="condition", values="dice")
        wh = gm.pivot(index="frame", columns="condition", values="hd95")
        isap = gm.drop_duplicates("frame").set_index("frame").apparent if False else gm.drop_duplicates("frame").set_index("frame")["apache"]
        say(f"MODE = {mode}")
        say(f"{'input to the segmenter':>30} | {'Dice (all frames)':>26} | {'Dice Apache':>26} | {'Dice other videos':>26} | {'HD95 px (median)':>16}")
        for c in CONDS:
            if c not in wide:
                continue
            a = boot(wide[c])
            b = boot(wide[c][isap.loc[wide.index]])
            o = boot(wide[c][~isap.loc[wide.index]])
            say(f"{NAMES[c]:>30} | {f(a):>26} | {f(b):>26} | {f(o):>26} | {wh[c].median():16.1f}")
        say(f"  paired difference to raw (same frames), Dice; 'NI' = lower bound above -{MARGIN}")
        for c in CONDS[1:]:
            if c not in wide:
                continue
            dd = (wide[c] - wide["raw"]).dropna()
            t = boot(dd, seed=1)
            ni = "NI yes" if t[1] > -MARGIN else "NI NO"
            better = "  better" if t[1] > 0 else ("  WORSE" if t[2] < 0 else "")
            ap = boot(dd[isap.loc[dd.index]], seed=2)
            ot = boot(dd[~isap.loc[dd.index]], seed=3)
            say(f"{NAMES[c]:>30} - raw : {f(t, '{:+.3f}')}  {ni}{better}  | Apache {f(ap, '{:+.3f}')} | others {f(ot, '{:+.3f}')}")
        say()
    gm = g[(g["mode"] == "predicted") & g.apache]
    say("AREA OF THE LUMEN (cross-section, a proxy for bladder volume), Apache frames, relative error (pred - true)/true")
    say(f"{'input':>30} | {'bias':>8} | {'95% limits of agreement':>26} | {'median |error|':>14}")
    for c in CONDS:
        x = gm[gm.condition == c]
        if x.empty:
            continue
        e = (x.area_pred - x.area_true) / x.area_true
        say(f"{NAMES[c]:>30} | {e.mean():+8.3f} | [{e.mean() - 1.96 * e.std():+.3f}, {e.mean() + 1.96 * e.std():+.3f}]".ljust(75) + f" | {e.abs().median():14.3f}")
    (OUT / "realSeg_summary.txt").write_text("\n".join(LINES))

    gm = g[g["mode"] == "predicted"]
    wide = gm.pivot(index="frame", columns="condition", values="dice")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    cs = [c for c in CONDS[1:] if c in wide]
    ys, lo, hi = [], [], []
    for c in cs:
        t = boot((wide[c] - wide["raw"]).dropna(), seed=1)
        ys.append(t[0]), lo.append(t[0] - t[1]), hi.append(t[2] - t[0])
    pos = np.arange(len(cs))
    ax[0].errorbar(ys, pos, xerr=[lo, hi], fmt="o", color="#1f77b4", capsize=4)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].axvline(-MARGIN, color="#d62728", ls="--", lw=1, label=f"non-inferiority margin (-{MARGIN})")
    ax[0].set_yticks(pos)
    ax[0].set_yticklabels([NAMES[c] for c in cs], fontsize=9)
    ax[0].set_xlabel("Dice difference to raw input (right = better), 95% CI")
    ax[0].set_title("Does despeckling change lumen segmentation on unseen videos?", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=0.25)
    x = gm[(gm.apache)]
    for c, col in (("raw", "#7f7f7f"), ("roi_nlm2", "#17becf")):
        xx = x[x.condition == c]
        ax[1].scatter(xx.area_true, xx.area_pred, s=14, alpha=0.7, color=col, label=NAMES[c])
    lim = [x.area_true.min(), x.area_true.max()]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set_xlabel("hand-drawn lumen area (px)")
    ax[1].set_ylabel("segmented area (px)")
    ax[1].set_title("Lumen area tracking, Apache video", fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)
    fig.suptitle("Real bladder frames, held-out videos", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "realSeg_figure.png", dpi=120)


if __name__ == "__main__":
    main()
