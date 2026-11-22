"""Step 7a: automatic measurement of a device-like object inside the lumen.

Input : 8-bit image (already filtered or not), lumen mask (from the segmentation), imaging fan.
Output: detected device mask, visible length (skeleton length in pixels), centroid, and a detection flag.

What is measured: the VISIBLE EXTENT (span = largest distance between two device pixels) of the device in the imaging
plane, plus its centre. This is not the true 3D length: a 2D cut through a 3D object shows only part of it.

measure_device() below is the simple threshold baseline. It was tried first (parameters scanned on a separate development
set) and found poor in noisy scenes (about 0.3 Dice at best), so the experiment uses a learned pixel classifier
(run_measure.py). span_px, skeleton_length, truth and dice from this file are used by that experiment.

Threshold baseline method:
  1. Difference-of-Gaussians (sigma 1.5 - 5) turns small bright structures into peaks.
  2. Noise level sigma_n = 1.4826 * MAD of the DoG inside the eroded lumen (robust to the device itself).
  3. Hysteresis threshold: seeds above median + HI*sigma_n, grown down to median + LO*sigma_n.
  4. Remove blobs smaller than MIN_AREA px, close small gaps, thin to a centre-line (skeleton).
  5. Length = skeleton length with 1 px for straight and sqrt(2) px for diagonal steps.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize

HI, LO, MIN_AREA = 4.5, 2.5, 20


def skeleton_length(mask):
    """Centre-line length in pixels of a binary mask."""
    sk = skeletonize(mask.astype(bool))
    if sk.sum() == 0:
        return 0.0
    # count 4-neighbour and diagonal links once each
    s = sk.astype(np.uint8)
    straight = (s[:, 1:] & s[:, :-1]).sum() + (s[1:, :] & s[:-1, :]).sum()
    diag = (s[1:, 1:] & s[:-1, :-1]).sum() + (s[1:, :-1] & s[:-1, 1:]).sum()
    # a skeleton pixel with no links still has length ~1
    return float(straight + np.sqrt(2.0) * diag + 1.0)


def span_px(mask):
    """Largest distance between any two pixels of the mask (Feret diameter): the device extent in the image.
    Robust to a device that shows up as several separate echoes."""
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return float("nan")
    pts = np.stack([ys, xs], 1).astype(float)
    if len(pts) > 3:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:
            pass
    return float(cdist(pts, pts).max()) + 1.0


def measure_device(img, lumen, hi=HI, lo=LO, min_area=MIN_AREA):
    f = img.astype(np.float64)
    Lc = ndi.binary_erosion(lumen.astype(bool), iterations=4)
    out = dict(detected=False, mask=np.zeros(img.shape, bool), length_px=np.nan, cy=np.nan, cx=np.nan, area=0)
    if Lc.sum() < 100:
        return out
    dog = ndi.gaussian_filter(f, 1.5) - ndi.gaussian_filter(f, 5.0)
    med = float(np.median(dog[Lc]))
    sig = 1.4826 * float(np.median(np.abs(dog[Lc] - med))) + 1e-6
    cand = apply_hysteresis_threshold(np.where(Lc, dog, -1e9), med + lo * sig, med + hi * sig)
    lab, n = ndi.label(cand)
    if n == 0:
        return out
    sizes = ndi.sum(cand, lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes >= min_area)[0])
    if keep.sum() == 0:
        return out
    keep = ndi.binary_closing(keep, iterations=2) & Lc
    ys, xs = np.nonzero(keep)
    out.update(detected=True, mask=keep, length_px=skeleton_length(keep), cy=float(ys.mean()), cx=float(xs.mean()), area=int(keep.sum()))
    return out


def truth(device):
    ys, xs = np.nonzero(device)
    return dict(length_px=skeleton_length(device), span_px=span_px(device), cy=float(ys.mean()), cx=float(xs.mean()), area=int(device.sum()))


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s > 0 else float("nan")
