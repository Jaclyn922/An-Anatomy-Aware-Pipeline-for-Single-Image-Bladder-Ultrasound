"""Step 3b: the five processing conditions. Each takes (img_u8, lumen_mask, fov) and returns uint8.

  raw         no processing (the reference every other condition is compared with)
  nlm_global  non-local means over the whole frame (replaces the repo's log-NLM, see note)
  srad_global SRAD over the whole frame, speckle scale q0 estimated from the whole frame
  srad_lumen  SRAD only inside the lumen (q0 from the lumen core), everything else untouched
  roi_full    the three-region method: SRAD in lumen, bilateral on the wall band, light Gaussian outside

Note: the simulated and real images are already log-compressed display images, so noise is roughly additive
and NLM is applied directly (the original script took a second log). SRAD is applied to the display image
as in the original project; this is a stated simplification, not a claim that it is the optimal domain.
OBNLM is not included: it is not in the repository and was not re-implemented.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_nl_means

from roi import build_rois, roi_filter, srad


def raw(img, lumen, fov):
    return img.copy()


def nlm_global(img, lumen, fov):
    f = img.astype(np.float64) / 255.0
    # Speckle is spatially correlated, so wavelet-based estimate_sigma badly underestimates it (it made NLM a no-op).
    # Use the robust spread (MAD) of the high-passed image inside the imaging fan instead.
    hp = f - ndi.gaussian_filter(f, 3.0)
    sig = float(1.4826 * np.median(np.abs(hp[fov] - np.median(hp[fov]))))
    out = denoise_nl_means(f, patch_size=5, patch_distance=6, h=1.0 * sig, sigma=sig, fast_mode=True)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8) * fov.astype(np.uint8)


def srad_global(img, lumen, fov):
    out = srad(img.astype(np.float64) / 255.0, iters=30, q0_region=None)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8) * fov.astype(np.uint8)


def srad_lumen(img, lumen, fov):
    core, _, _ = build_rois(lumen, 3, 3)
    out = srad(img.astype(np.float64) / 255.0, iters=30, q0_region=core)
    res = img.copy()
    res[core] = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)[core]
    return res


def roi_full(img, lumen, fov):
    return roi_filter(img, lumen.astype(np.uint8), srad_iters=30)


def _roi_with(iters):
    def f(img, lumen, fov):
        return roi_filter(img, lumen.astype(np.uint8), srad_iters=iters)

    return f


MAIN = ["raw", "nlm_global", "srad_global", "srad_lumen", "roi_full"]
CONDITIONS = dict(raw=raw, nlm_global=nlm_global, srad_global=srad_global, srad_lumen=srad_lumen, roi_full=roi_full)
# strength sweep of the ROI method (roi_full is the 30-iteration setting used in the original project)
SWEEP = dict(roi_i10=_roi_with(10), roi_i100=_roi_with(100), roi_i200=_roi_with(200))
CONDITIONS.update(SWEEP)


def _roi_guard(iters):
    def f(img, lumen, fov):
        return roi_filter(img, lumen.astype(np.uint8), srad_iters=iters, guard=True)

    return f


# device-friendly variant: diffusion stops at the lumen boundary
def _roi_guard_deep(iters, guard_in):
    def f(img, lumen, fov):
        return roi_filter(img, lumen.astype(np.uint8), srad_iters=iters, guard=True, guard_in=guard_in)

    return f


GUARD = {f"roi_g{n}": _roi_guard(n) for n in (30, 100, 200, 400)}
GUARD.update({f"roi_d{n}": _roi_guard_deep(n, 8) for n in (30, 100, 200, 400)})
CONDITIONS.update(GUARD)

SETS = dict(
    main=["raw", "nlm_global", "srad_global", "srad_lumen", "roi_full", "roi_i10", "roi_i100", "roi_i200"],
    guard=["raw", "roi_full", "roi_i100", "roi_i200", "roi_g30", "roi_g100", "roi_g200", "roi_g400",
           "roi_d30", "roi_d100", "roi_d200", "roi_d400"],
)
