"""Step 3c: metrics. All in 8-bit grey levels; the device mask (ground truth) is known.

contrast / retention   How much of the device's true contrast survives processing.
                       contrast(img) = mean grey level on the device minus mean of a local lumen background
                       (shadow columns below the device excluded). retention = contrast(processed) / contrast(clean
                       reference). Below 1 means the device was attenuated or erased.
detect_stat S          Detection statistic of a simple fixed observer: brightest blob (Gaussian sigma 1.5) inside the
                       lumen minus the lumen median. Computed on present and on absent images.
hallucination H        On ABSENT images: brightest blob of (processed - clean) inside the lumen. This is structure the
                       truth does not contain. For a harmless filter H does not grow compared with raw.
psnr / ssim            Against the clean reference, inside the imaging fan.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

SIGMA = 1.5


def regions(lumen, device):
    Lc = ndi.binary_erosion(lumen, iterations=4)
    near = ndi.binary_dilation(device, iterations=14)
    excl = ndi.binary_dilation(device, iterations=4)
    shadow = ndi.binary_dilation(np.maximum.accumulate(device, axis=0), structure=np.ones((1, 5), bool), iterations=2)
    B = Lc & near & ~excl & ~shadow
    return Lc, B


def contrast(img, device, B):
    return float(img[device].mean() - img[B].mean())


def detect_stat(img, Lc):
    g = ndi.gaussian_filter(img.astype(np.float64), SIGMA)
    return float(g[Lc].max() - np.median(img[Lc]))


def detect_stat_dog(img, Lc, s1=1.5, s2=5.0):
    """Small-bright-blob detector (difference of Gaussians). Unlike detect_stat it ignores slow brightness ramps,
    such as wall echo leaking into the lumen edge, so it responds to the device and not to the wall."""
    f = img.astype(np.float64)
    return float((ndi.gaussian_filter(f, s1) - ndi.gaussian_filter(f, s2))[Lc].max())


def hp_noise(img, Lc, sigma=3.0):
    """Fine-scale noise in the lumen: std of the high-passed image."""
    f = img.astype(np.float64)
    return float((f - ndi.gaussian_filter(f, sigma))[Lc].std())


def hallucination(proc, clean, Lc):
    d = ndi.gaussian_filter(proc.astype(np.float64) - clean.astype(np.float64), SIGMA)
    return float(d[Lc].max())


def psnr_ssim(proc, clean, fov):
    err = (proc.astype(np.float64) - clean.astype(np.float64))[fov]
    mse = float(np.mean(err**2)) + 1e-9
    _, smap = structural_similarity(clean, proc, data_range=255, full=True)
    return float(10 * np.log10(255.0**2 / mse)), float(smap[fov].mean())


def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie): the area under the ROC curve."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)
