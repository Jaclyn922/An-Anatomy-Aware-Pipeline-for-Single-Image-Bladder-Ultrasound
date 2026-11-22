"""Experiment B (real data): does despeckling change how well a segmenter finds the bladder lumen on UNSEEN videos?

Same tiny U-Net, same training recipe, same folds; only the INPUT image differs (6 pre-specified conditions, see prep_filtered.py).
Folds (4): the 53 Apache frames are cut into 4 contiguous time blocks; the 14 frames of the other 8 videos are spread over the 4 folds
BY VIDEO. A held-out fold's neighbours in time (3 frames each side of an Apache block) are dropped from training so near-identical
neighbouring frames cannot leak. Test uses one file per source frame (the Roboflow copies of a frame are never split).
Two ways to feed the ROI filters at test time:
  oracle     the hand-drawn mask guides the filter (upper bound; what a lab with annotations could do)
  predicted  the mask comes from the raw-image segmenter of the same fold (what deployment would do)
Metrics per test frame: Dice, 95% Hausdorff distance (px, 256x256 images), relative area error.
Usage (project root):  python src/run_real_seg.py --seeds 0 --epochs 40
Output: outputs/realSeg_rows.csv (appended after every model so a crash loses nothing)
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi import roi_filter  # noqa: E402
from roi_nlm import roi_nlm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONDS = ["raw", "nlm", "srad30", "srad100", "roi30", "roi_v2_100", "roi_nlm2"]
ROI_CONDS = {"roi30": dict(srad_iters=30, guard=False), "roi_v2_100": dict(srad_iters=100, guard=True), "roi_nlm2": None}
VID_FOLD = {"Bladder_Trans_Female_Video_mp4": 0, "Bladder_Trans_Male_mp4": 1, "Bladder_Sag_Male": 2, "Bladder_Trans_Female_2": 2,
            "Bladder_Trans_Female": 2, "Bladder_gain30": 3, "Img_20210720_181354": 3, "Img_20210720_181407": 3}
S = 128  # network input size


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.e1, self.e2, self.e3, self.b = block(1, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), block(8 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), block(2 * c, c)
        self.o = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.o(d1)


def to_net(img):
    return cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def fold_split(rows):
    """returns dict file -> (frame, video), frames list, fold of each frame, ordered Apache frames"""
    fr = {}
    for r in rows:
        fr.setdefault(r["frame"], []).append(r)
    ap = sorted([f for f in fr if fr[f][0]["video"].startswith("Apache")], key=lambda f: int(re.search(r"-(\d+)$", f).group(1)))
    fold = {}
    n = len(ap)
    cuts = [round(i * n / 4) for i in range(5)]
    for k in range(4):
        for f in ap[cuts[k]:cuts[k + 1]]:
            fold[f] = k
    for f in fr:
        if not fr[f][0]["video"].startswith("Apache"):
            fold[f] = VID_FOLD[fr[f][0]["video"]]
    buffer = {k: set(ap[max(0, cuts[k] - 3):cuts[k]] + ap[cuts[k + 1]:cuts[k + 1] + 3]) for k in range(4)}
    return fr, fold, buffer


def augment(x, y, rng):
    if rng.random() < 0.5:
        x, y = x[:, ::-1], y[:, ::-1]
    dx, dy = rng.integers(-8, 9, 2)
    x, y = np.roll(x, (dy, dx), (0, 1)), np.roll(y, (dy, dx), (0, 1))
    g, b = rng.uniform(0.9, 1.1), rng.uniform(-0.05, 0.05)
    return np.clip(x * g + b, 0, 1), y


def train(X, Y, epochs, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = UNet()
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=epochs * int(np.ceil(len(X) / 8)))
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(len(X))
        for i in range(0, len(X), 8):
            ids = perm[i:i + 8]
            xs, ys = zip(*[augment(X[j], Y[j], rng) for j in ids])
            xb = torch.from_numpy(np.ascontiguousarray(np.stack(xs)))[:, None]
            yb = torch.from_numpy(np.ascontiguousarray(np.stack(ys)))[:, None]
            lo = m(xb)
            p = torch.sigmoid(lo)
            loss = F.binary_cross_entropy_with_logits(lo, yb) + 1 - (2 * (p * yb).sum() + 1) / (p.sum() + yb.sum() + 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    m.eval()
    return m


@torch.no_grad()
def predict(m, img_u8):
    p = torch.sigmoid(m(torch.from_numpy(to_net(img_u8))[None, None]))
    p = F.interpolate(p, size=img_u8.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    mk = p > 0.5
    mk = ndi.binary_fill_holes(mk)
    lab, n = ndi.label(mk)
    if n > 1:
        sz = ndi.sum(mk, lab, range(1, n + 1))
        mk = lab == (1 + int(np.argmax(sz)))
    return mk


def hd95(a, b):
    if not a.any() or not b.any():
        return np.nan
    sa = a ^ ndi.binary_erosion(a)
    sb = b ^ ndi.binary_erosion(b)
    da, db = ndi.distance_transform_edt(~sb)[sa], ndi.distance_transform_edt(~sa)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def score(pred, true):
    inter = (pred & true).sum()
    return dict(dice=float(2 * inter / max(pred.sum() + true.sum(), 1)), hd95=hd95(pred, true),
                area_pred=int(pred.sum()), area_true=int(true.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", default="outputs/realSeg_rows.csv")
    a = ap.parse_args()
    torch.set_num_threads(2)
    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    cache = np.load(ROOT / "outputs/cache_filtered.npz")
    fr, fold, buffer = fold_split(rows)
    outp = ROOT / a.out
    new = not outp.exists()
    fh = open(outp, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=["seed", "fold", "frame", "video", "condition", "mode", "dice", "hd95", "area_pred", "area_true"])
    if new:
        wr.writeheader()
    t0 = time.time()
    for seed in a.seeds:
        for k in a.folds:
            tr_files = [r["file"] for r in rows if fold[r["frame"]] != k and r["frame"] not in buffer[k]]
            te_frames = [f for f in fr if fold[f] == k]
            te_rows = [fr[f][0] for f in te_frames]
            Ytr = np.stack([cv2.resize(cache[f"mask|{f}"], (S, S), interpolation=cv2.INTER_NEAREST).astype(np.float32) for f in tr_files])
            pred_mask = {}
            for cond in CONDS:
                Xtr = np.stack([to_net(cache[f"{cond}|{f}"]) for f in tr_files])
                m = train(Xtr, Ytr, a.epochs, seed)
                for r in te_rows:
                    true = cache[f"mask|{r['file']}"] > 0
                    modes = [("oracle", cache[f"{cond}|{r['file']}"])]
                    if cond == "raw":
                        raw_img = cache[f"raw|{r['file']}"]
                        pm = predict(m, raw_img)
                        pred_mask[r["file"]] = pm
                    if cond in ROI_CONDS:
                        raw_img = cache[f"raw|{r['file']}"]
                        pm = pred_mask[r["file"]]
                        if pm.sum() <= 200:
                            img_p = raw_img
                        elif cond == "roi_nlm2":
                            img_p = roi_nlm(raw_img, pm, hmult=2.0)
                        else:
                            img_p = roi_filter(raw_img, pm.astype(np.uint8), **ROI_CONDS[cond])
                        modes.append(("predicted", img_p))
                    elif cond != "raw":
                        modes.append(("predicted", cache[f"{cond}|{r['file']}"]))
                    else:
                        modes.append(("predicted", cache[f"{cond}|{r['file']}"]))
                    for mode, img in modes:
                        pm = predict(m, img)
                        wr.writerow(dict(seed=seed, fold=k, frame=r["frame"], video=r["video"], condition=cond, mode=mode, **score(pm, true)))
                fh.flush()
                print(f"seed {seed} fold {k} {cond} done {time.time() - t0:.0f}s ({len(tr_files)} train files, {len(te_rows)} test frames)", flush=True)
    fh.close()
