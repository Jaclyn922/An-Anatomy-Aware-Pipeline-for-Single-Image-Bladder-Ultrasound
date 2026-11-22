"""Step 6d: tables and figures for the guard variant, mask robustness, IQA stability and the real-frame check.

Run after run_experiment.py --set guard, run_mask_robustness.py, run_iqa.py and run_real_check.py.
Everything is a cluster bootstrap over source frames. Output text is also saved to outputs/extra_summary.txt.
Figures: guard_figure.png, mask_figure.png, iqa_figure.png, real_figure.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def prep(df):
    raw = df[df.condition == "raw"].set_index("spec_id")
    df = df.copy()
    df["dH"] = df["H_absent"] - df["spec_id"].map(raw["H_absent"])
    df["noise_cut"] = 1.0 - df["lumen_hp_std_absent"] / df["spec_id"].map(raw["lumen_hp_std_absent"])
    return df


def ci_txt(t, fmt="{:.2f}"):
    return (fmt + " [" + fmt + ", " + fmt + "]").format(*t)


# ---------------------------------------------------------------- guard
def guard():
    f = OUT / "results_guard.csv"
    if not f.exists():
        say("(results_guard.csv missing, skipped)")
        return
    df = prep(pd.read_csv(f))
    raw = df[df.condition == "raw"]
    afc_raw = pd.Series(S.afc(raw), index=raw["spec_id"].values)
    fam = {"original ROI (barrier at none)": ["roi_full", "roi_i100", "roi_i200"],
           "barrier at lumen core": ["roi_g30", "roi_g100", "roi_g200", "roi_g400"],
           "barrier 8 px inside lumen": ["roi_d30", "roi_d100", "roi_d200", "roi_d400"]}
    its = {"roi_full": 30, "roi_i100": 100, "roi_i200": 200, "roi_g30": 30, "roi_g100": 100, "roi_g200": 200, "roi_g400": 400,
           "roi_d30": 30, "roi_d100": 100, "roi_d200": 200, "roi_d400": 400}
    say(f"GUARD VARIANT ({df.spec_id.nunique()} scenes). Diffusion stopped at a barrier inside the lumen.")
    say(f"{'condition':>10} {'iters':>5} | {'retention':>22} | {'mostly erased':>20} | {'extra false struct. (grey)':>26} | {'paired 2AFC change vs raw':>28} | noise removed")
    res = {}
    for name, conds in fam.items():
        for c in conds:
            d = df[df.condition == c]
            v = d[d.visible == 1]
            r = S.boot_median(v, "retention")
            er = S.boot_mean(v, (v["retention"].values < 0.5).astype(float))
            h = S.boot_median(d, "dH")
            da = S.boot_mean(d, S.afc(d) - afc_raw.loc[d["spec_id"].values].values, seed=2)
            nc = float(np.nanmedian(d.noise_cut))
            res[c] = dict(ret=r, h=h, da=da, nc=nc, er=er)
            say(f"{c:>10} {its[c]:5d} | {ci_txt(r):>22} | {er[0]:5.1%} [{er[1]:4.1%},{er[2]:4.1%}] | {ci_txt(h, '{:+.1f}'):>26} | {ci_txt(da, '{:+.3f}'):>28} | {nc:5.0%}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for (name, conds), col in zip(fam.items(), colors):
        xs = [its[c] for c in conds]
        for a_, key, idx in [(ax[0], "ret", 0), (ax[1], "da", 0), (ax[2], "h", 0)]:
            y = [res[c][key][0] for c in conds]
            lo = [res[c][key][1] for c in conds]
            hi = [res[c][key][2] for c in conds]
            a_.errorbar(xs, y, yerr=[np.array(y) - lo, np.array(hi) - np.array(y)], fmt="o-", color=col, capsize=3, label=name)
    ax[0].set_title("Device contrast kept (median)")
    ax[1].set_title("Detection change vs raw (paired 2AFC)")
    ax[2].set_title("Extra false structure, no-device images (grey levels)")
    ax[0].axhline(1.0, color="k", lw=0.7)
    ax[1].axhline(0.0, color="k", lw=0.7)
    ax[2].axhline(0.0, color="k", lw=0.7)
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xticks([30, 100, 200, 400])
        a_.set_xticklabels(["30", "100", "200", "400"])
        a_.set_xlabel("SRAD iterations")
    ax[0].legend(fontsize=8)
    fig.suptitle("Does stopping diffusion at the lumen boundary make strong despeckling safe?", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "guard_figure.png", dpi=110)
    say()


# ---------------------------------------------------------------- mask robustness
def mask():
    f = OUT / "results_mask.csv"
    if not f.exists():
        say("(results_mask.csv missing, skipped)")
        return
    df = pd.read_csv(f)
    raw = df[df.condition == "raw"].set_index("spec_id")
    raw_afc = pd.Series(S.afc(raw.reset_index()), index=raw.index)
    order = ["erode4", "erode2", "erode1", "none", "dilate1", "dilate2", "dilate4", "shift4", "warp3"]
    conds = [c for c in ["srad_lumen", "roi_full", "roi_g30", "roi_d30"] if c in set(df.condition)]
    say(f"MASK ROBUSTNESS ({df.spec_id.nunique()} scenes). The scene uses the true mask; the method gets a perturbed one.")
    say(f"{'perturbation':>12} {'mask Dice':>9} | " + " | ".join(f"{c:>34}" for c in conds))
    say(f"{'':>12} {'':>9} | " + " | ".join(f"{'retention / extra false / d2AFC':>34}" for _ in conds))
    out = {}
    for p in order:
        parts = []
        for c in conds:
            d = df[(df.condition == c) & (df.perturb == p)]
            v = d[d.visible == 1]
            r = S.boot_median(v, "retention")[0]
            h = S.boot_median(d, "dH")[0]
            da = S.boot_mean(d, S.afc(d) - raw_afc.loc[d["spec_id"].values].values, seed=3)
            out[(p, c)] = (r, h, da)
            parts.append(f"{r:6.2f} / {h:+6.1f} / {da[0]:+.3f} [{da[1]:+.3f},{da[2]:+.3f}]")
        dice = df[(df.perturb == p) & (df.condition == conds[0])].mask_dice.median()
        say(f"{p:>12} {dice:9.2f} | " + " | ".join(f"{x:>34}" for x in parts))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    col = {"srad_lumen": "#2ca02c", "roi_full": "#d62728", "roi_g30": "#9467bd", "roi_d30": "#1f77b4"}
    x = np.arange(len(order))
    for c in conds:
        ax[0].plot(x, [out[(p, c)][0] for p in order], "o-", color=col[c], label=c)
        ax[1].errorbar(x, [out[(p, c)][2][0] for p in order],
                       yerr=[[out[(p, c)][2][0] - out[(p, c)][2][1] for p in order], [out[(p, c)][2][2] - out[(p, c)][2][0] for p in order]],
                       fmt="o-", color=col[c], capsize=3, label=c)
        ax[2].plot(x, [out[(p, c)][1] for p in order], "o-", color=col[c], label=c)
    ax[0].set_title("Device contrast kept (median)")
    ax[1].set_title("Detection change vs raw (paired 2AFC)")
    ax[2].set_title("Extra false structure (grey levels)")
    ax[0].axhline(1.0, color="k", lw=0.7)
    ax[1].axhline(0.0, color="k", lw=0.7)
    ax[2].axhline(0.0, color="k", lw=0.7)
    for a_ in ax:
        a_.set_xticks(x)
        a_.set_xticklabels(order, rotation=35, fontsize=9)
    ax[0].legend(fontsize=8)
    fig.suptitle("Wrong segmentation mask: how much does it matter? (30 iterations)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "mask_figure.png", dpi=110)
    say()


# ---------------------------------------------------------------- IQA
def iqa():
    f = OUT / "iqa_summary.csv"
    if not f.exists():
        say("(iqa_summary.csv missing, skipped)")
        return
    d = pd.read_csv(f)
    say("IQA STABILITY (train: Apache video, test: 8 other sources; same scene rendered with several noise seeds)")
    say(f"{'condition':>12} | {'instability (within sd / total sd, lower better)':>50} | {'ICC (higher better)':>26} | {'Spearman with true visibility':>32}")
    for _, r in d.iterrows():
        say(f"{r.condition:>12} | {r.instability:.3f} [{r.instability_lo:.3f}, {r.instability_hi:.3f}]".ljust(66) +
            f" | {r.icc:.3f} [{r.icc_lo:.3f}, {r.icc_hi:.3f}]".ljust(29) + f" | {r.spearman:.3f} [{r.spearman_lo:.3f}, {r.spearman_hi:.3f}]")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(d))
    for a_, key, title, ref in [(ax[0], "instability", "Prediction instability across noise (lower = more stable)", None),
                               (ax[1], "icc", "ICC: share of variance between scenes (higher = better)", None),
                               (ax[2], "spearman", "Sensitivity: Spearman with true device visibility", None)]:
        y = d[key].values
        a_.bar(x, y, color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728", "#8c564b"][: len(d)],
               yerr=[y - d[key + "_lo"].values, d[key + "_hi"].values - y], capsize=4)
        a_.set_xticks(x)
        a_.set_xticklabels(d.condition, rotation=20, fontsize=9)
        a_.set_title(title, fontsize=10)
    fig.suptitle("Does despeckling make a no-reference quality score more stable?", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "iqa_figure.png", dpi=110)
    say()


# ---------------------------------------------------------------- real frames
def real():
    fa, fb = OUT / "real_noref.csv", OUT / "real_composite.csv"
    if not fa.exists():
        say("(real_noref.csv missing, skipped)")
        return
    a = pd.read_csv(fa)
    say(f"REAL FRAMES, no reference ({a.frame.nunique()} frames, {a.video.nunique()} videos), processed / raw, median [95% CI over frames]")
    say(f"{'condition':>12} | {'lumen noise':>22} | {'tissue texture':>22} | {'boundary gradient':>22} | lumen mean shift (grey)")
    for c in a.condition.unique():
        d = a[a.condition == c]
        ln, tt, ek = (S.boot_median(d, k) for k in ("lumen_noise", "tissue_tex", "edge_keep"))
        say(f"{c:>12} | {ci_txt(ln):>22} | {ci_txt(tt):>22} | {ci_txt(ek):>22} | {d.lumen_shift.median():+.2f}")
    if fb.exists():
        b = pd.read_csv(fb)
        say()
        say(f"DEVICE COMPOSITES on real backgrounds ({b.groupby(['frame']).ngroups} frames, {len(b[b.condition == 'raw'])} composites): retention of device contrast")
        sim = None
        for fn in ("results_guard.csv", "results.csv"):
            if (OUT / fn).exists():
                t = pd.read_csv(OUT / fn)
                t = t[t.visible == 1]
                sim = pd.concat([sim, t]) if sim is not None else t
        for c in b.condition.unique():
            d = b[b.condition == c]
            r = S.boot_median(d, "retention")
            s_ = sim[sim.condition == c].retention.median() if sim is not None and c in set(sim.condition) else np.nan
            say(f"{c:>12} | real background {ci_txt(r):>22} | simulated background {s_:.2f}")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        conds = list(a.condition.unique())
        for a_, key, title in [(ax[0], "lumen_noise", "Lumen fine noise (processed / raw)"), (ax[1], "tissue_tex", "Tissue texture (processed / raw)"),
                               (ax[2], "edge_keep", "Boundary gradient (processed / raw)")]:
            data = [a[a.condition == c][key].values for c in conds]
            a_.boxplot(data, tick_labels=conds, showfliers=False)
            a_.set_title(title, fontsize=10)
            a_.tick_params(axis="x", rotation=40, labelsize=8)
            a_.axhline(1.0, color="k", lw=0.6)
        fig.suptitle("Real frames: what each processing does (no reference needed)", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / "real_figure.png", dpi=110)
    say()


if __name__ == "__main__":
    S.NB = 1000
    guard()
    mask()
    iqa()
    real()
    (OUT / "extra_summary.txt").write_text("\n".join(LINES))
    print("saved outputs/extra_summary.txt")
