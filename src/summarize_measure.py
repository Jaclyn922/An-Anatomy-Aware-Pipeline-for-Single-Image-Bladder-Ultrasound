"""Step 7c: tables and figure for the measurement + quality-grade experiment (outputs/measure_results.csv).

Errors are relative (percent of the true device span), so no pixel size in mm is needed. To get mm, multiply the span
by the pixel spacing of the real scanner.
Confidence intervals: bootstrap over source frames (test scenes of the same frame are not independent).
Writes outputs/measure_summary.txt and outputs/measure_figure.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LINES = []
INFILE = sys.argv[1] if len(sys.argv) > 1 else "measure_results.csv"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
TOL = 0.20


def say(s=""):
    print(s)
    LINES.append(s)


def boot(df, fn, nb=1000, seed=0):
    """fn(sub_df) -> float; bootstrap over source frames. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby("frame")}
    keys = list(groups)
    pt = fn(df)
    vals = []
    for _ in range(nb):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sub = pd.concat([groups[keys[i]] for i in pick])
        v = fn(sub)
        if np.isfinite(v):
            vals.append(v)
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fmt(t, f="{:.2f}"):
    return (f + " [" + f + ", " + f + "]").format(*t)


def rel_err(d):
    return (d.est_span - d.true_span) / d.true_span


def within(d):
    e = rel_err(d)
    return float(((e.abs() <= TOL) & (d.detected == 1)).mean()) if len(d) else np.nan


def main():
    df = pd.read_csv(OUT / INFILE)
    vis = df[df.grade_true >= 2]
    say(f"MEASUREMENT EXPERIMENT: {df.spec_id.nunique()} test scenes from {df.video.nunique()} source videos, {df.frame.nunique()} frames.")
    say(f"Scenes where the device is truly visible (clean contrast >= 8 grey levels): {vis.spec_id.nunique()}.")
    say()
    say(f"A) Device found and measured, visible devices. 'within 20%' = detected AND span error <= 20% of the true span")
    say(f"{'condition':>12} | {'detected':>20} | {'Dice':>20} | {'within 20%':>20} | {'median |span err|':>22} | centre error px")
    conds = list(df.condition.unique())
    res = {}
    for c in conds:
        d = vis[vis.condition == c]
        det = boot(d, lambda x: float(x.detected.mean()))
        dc = boot(d, lambda x: float(x.dice.median()))
        w = boot(d, within)
        me = boot(d[d.detected == 1], lambda x: float(rel_err(x).abs().median()))
        ce = float(d.centroid_err.median())
        res[c] = dict(w=w, det=det, dc=dc, me=me)
        say(f"{c:>12} | {fmt(det):>20} | {fmt(dc):>20} | {fmt(w):>20} | {fmt(me):>22} | {ce:5.1f}")
    say()
    say("A2) Paired comparison on the same scenes (difference of means, bootstrap over frames). Positive = first condition is better.")
    wide = vis.pivot(index="spec_id", columns="condition")
    fr = vis.drop_duplicates("spec_id").set_index("spec_id").frame
    good = ((rel_err(vis).abs() <= TOL) & (vis.detected == 1)).astype(float)
    vis = vis.assign(good=good.values)
    wide = vis.pivot(index="spec_id", columns="condition")
    rng = np.random.default_rng(0)

    def paired(a, b, col):
        x = (wide[col][a] - wide[col][b]).dropna()
        keys = fr.loc[x.index].unique()
        grp = {k: x[fr.loc[x.index] == k].values for k in keys}
        bs = [np.concatenate([grp[keys[i]] for i in rng.choice(len(keys), len(keys))]).mean() for _ in range(2000)]
        return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    for a_, b_ in [("roi_g100", "raw"), ("roi_full", "raw"), ("srad_global", "raw"), ("nlm_global", "raw"), ("roi_g100", "roi_full"), ("roi_g100", "srad_global")]:
        if a_ in conds and b_ in conds:
            say(f"   {a_:>11} vs {b_:<11} | within-20% {fmt(paired(a_, b_, 'good'), '{:+.3f}')} | Dice {fmt(paired(a_, b_, 'dice'), '{:+.3f}')}")
    say()
    say("B) By true visibility grade (processing = raw): how the measurement degrades as the device gets harder to see")
    say(f"{'true grade':>10} {'n':>5} | {'detected':>9} | {'within 20%':>11} | {'median |span err|':>18}")
    for g in range(1, 6):
        d = df[(df.condition == "raw") & (df.grade_true == g)]
        if len(d) == 0:
            continue
        dd = d[d.detected == 1]
        say(f"{g:>10} {len(d):5d} | {d.detected.mean():9.2f} | {within(d):11.2f} | {rel_err(dd).abs().median() if len(dd) else float('nan'):18.2f}")
    say()
    say("C) Quality grader (predicts the grade from the image it is given), per condition")
    say(f"{'condition':>12} | {'Spearman pred vs true contrast':>32} | {'exact grade':>12} | {'within one grade':>16}")
    for c in conds:
        d = df[df.condition == c]
        rho = boot(d, lambda x: float(spearmanr(x.pred_c, x.c_clean).statistic), nb=500)
        ex = float((d.grade_pred == d.grade_true).mean())
        w1 = float(((d.grade_pred - d.grade_true).abs() <= 1).mean())
        say(f"{c:>12} | {fmt(rho):>32} | {ex:12.2f} | {w1:16.2f}")
    say()
    say("D) Reliability grade. The visibility grade above says how visible the device is in the CLEAN image; what matters for a")
    say("   user is whether THIS measurement can be trusted. A second grader predicts P(measurement within 20%) from the image,")
    say("   trained leave-one-video-out (the model never sees the video it scores). P is binned into grades 1-5:")
    say("   P<0.10 -> 1, <0.20 -> 2, <0.35 -> 3, <0.50 -> 4, >=0.50 -> 5.")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    sf = [c_ for c_ in df.columns if c_.startswith("sf")]
    rel_edges = [0.10, 0.20, 0.35, 0.50]
    parts = []
    for c in conds:
        d = df[df.condition == c].copy()
        d["good"] = ((rel_err(d).abs() <= TOL) & (d.detected == 1)).astype(int)
        d["P"] = np.nan
        for v in d.video.unique():
            tr_, te_ = d[d.video != v], d[d.video == v]
            if tr_.good.nunique() < 2:
                continue
            m = HistGradientBoostingClassifier(max_depth=2, max_iter=120, learning_rate=0.05, l2_regularization=2.0, random_state=0)
            m.fit(tr_[sf].values, tr_.good.values)
            d.loc[te_.index, "P"] = m.predict_proba(te_[sf].values)[:, 1]
        d["rel_grade"] = 1 + np.searchsorted(rel_edges, d.P.values, side="right")
        parts.append(d)
    dd_all = pd.concat(parts)
    say(f"{'condition':>12} | {'AUC (reliability vs good)':>27} | {'rule':>9} | {'frames kept':>11} | {'good among kept':>20} | {'good among rejected':>19}")
    for c in conds:
        d = dd_all[dd_all.condition == c]
        auc = boot(d, lambda x: float(roc_auc_score(x.good, x.P)) if x.good.nunique() == 2 else np.nan, nb=500)
        say(f"{c:>12} | {fmt(auc):>27} | {'no gate':>9} | {1.0:11.2f} | {d.good.mean():20.2f} | {'-':>19}")
        for thr in (3, 4, 5):
            k, r = d[d.rel_grade >= thr], d[d.rel_grade < thr]
            if len(k) < 15:
                say(f"{'':>12} | {'':>27} | {'grade>=' + str(thr):>9} | {len(k) / len(d):11.2f} | {'(too few frames)':>20} |")
                continue
            kg = boot(k, lambda x: float(x.good.mean()))
            say(f"{'':>12} | {'':>27} | {'grade>=' + str(thr):>9} | {len(k) / len(d):11.2f} | {fmt(kg):>20} | {r.good.mean() if len(r) else float('nan'):19.2f}")
    say()
    say("E) Reliability grade table (condition roi_g100 and raw): what each grade means for the user")
    for c in ("raw", "roi_g100"):
        if c not in conds:
            continue
        d = dd_all[dd_all.condition == c]
        say(f"  {c}:")
        for g in range(1, 6):
            k = d[d.rel_grade == g]
            ke = k[k.detected == 1]
            say(f"    grade {g}: {len(k):4d} frames | measured within 20%: {k.good.mean() if len(k) else float('nan'):.2f} | median |span err|: {rel_err(ke).abs().median() if len(ke) else float('nan'):.2f}")
    say()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.arange(len(conds))
    y = [res[c]["w"][0] for c in conds]
    ax[0].bar(x, y, yerr=[[res[c]["w"][0] - res[c]["w"][1] for c in conds], [res[c]["w"][2] - res[c]["w"][0] for c in conds]], capsize=4,
              color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"][: len(conds)])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(conds, rotation=20, fontsize=9)
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Device measured within 20% of true span\n(visible devices, higher = better)", fontsize=10)
    for c, col in (("raw", "#7f7f7f"), ("roi_g100", "#2ca02c")):
        if c not in conds:
            continue
        d = dd_all[dd_all.condition == c]
        ys = [d[d.rel_grade == g].good.mean() if (d.rel_grade == g).sum() >= 8 else np.nan for g in range(1, 6)]
        ax[1].plot(range(1, 6), ys, "o-", color=col, label=c)
    ax[1].set_xticks(range(1, 6))
    ax[1].set_xlabel("reliability grade (1 low ... 5 high)")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Share of frames measured within 20%\nby predicted reliability grade", fontsize=10)
    ax[1].legend(fontsize=8)
    for c, col in (("raw", "#7f7f7f"), ("roi_g100", "#2ca02c")):
        if c not in conds:
            continue
        d = dd_all[dd_all.condition == c]
        cov, acc = [], []
        for thr in (1, 2, 3, 4, 5):
            k = d[d.rel_grade >= thr]
            if len(k) >= 15:
                cov.append(len(k) / len(d))
                acc.append(k.good.mean())
        ax[2].plot(cov, acc, "o-", color=col, label=c)
    ax[2].set_xlabel("share of frames kept")
    ax[2].set_ylabel("share measured within 20%")
    ax[2].set_title("Keep only high-grade frames:\ncoverage vs reliability", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.suptitle("Automatic device measurement and the 1-5 quality gate (simulation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"measure_figure{TAG}.png", dpi=110)
    (OUT / f"measure_summary{TAG}.txt").write_text("\n".join(LINES))


if __name__ == "__main__":
    main()
