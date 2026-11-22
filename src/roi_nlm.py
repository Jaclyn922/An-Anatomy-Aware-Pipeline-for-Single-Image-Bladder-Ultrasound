"""ROI-NLM: smooth ONLY the lumen core with non-local means; wall band and outer tissue are left exactly as acquired.

Why: on real frames (experiment A) the course-project ROI filter pays a fixed price outside the lumen (bilateral on the wall, Gaussian on the
outer tissue: 13% of the wall edge and 31% of the outer texture lost) while global NLM smooths the lumen more and keeps both. The idea of
the ROI method (strong smoothing only where the tissue is uniform) stays; the operator inside the lumen becomes NLM, and nothing is done outside.
The lumen core is the mask eroded by `wall_in` px; the seam is feathered with a Gaussian of 1.5 px so there is no visible step.
Strength: h = hmult x noise sigma (sigma = robust MAD of the 3-px high-pass over the imaging fan, as in the global NLM baseline).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_nl_means

from simulate import fov_mask


def nlm_full(img_u8, hmult=2.0):
    """Whole-frame NLM at strength hmult x sigma (float in [0, 1]); computed once and reused by roi_nlm(sm=...)."""
    fov = fov_mask(img_u8)
    f = img_u8.astype(np.float64) / 255.0
    hp = f - ndi.gaussian_filter(f, 3.0)
    sig = float(1.4826 * np.median(np.abs(hp[fov] - np.median(hp[fov]))))
    return denoise_nl_means(f, patch_size=5, patch_distance=6, h=hmult * sig, sigma=sig, fast_mode=True)


def roi_nlm(img_u8, mask, hmult=2.0, wall_in=3, feather=1.5, sm=None):
    f = img_u8.astype(np.float64) / 255.0
    if sm is None:
        sm = nlm_full(img_u8, hmult)
    core = ndi.binary_erosion(mask.astype(bool), iterations=wall_in)
    w = ndi.gaussian_filter(core.astype(np.float64), feather)
    w = np.where(core, np.maximum(w, 0.0), w)
    out = w * sm + (1.0 - w) * f
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
