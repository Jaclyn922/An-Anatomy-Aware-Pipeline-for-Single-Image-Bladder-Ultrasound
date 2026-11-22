"""Region-adaptive despeckling (the 'ROI-guided' method) -- rewritten from roi_denoise_single.py.

What changed relative to the original script and why:
  1. SRAD sign fixed. The original computed d = I_i - I_neighbour and then did I + lam*div,
     which is anti-diffusion; on real frames it diverged to NaN. Here d = I_neighbour - I_i.
  2. q0 (the speckle reference of SRAD) is estimated inside the lumen core, as in Yu & Acton,
     not as the mean over the whole frame (which includes walls and bright tissue).
  3. Regions are disjoint. Originally `wall` partly sat inside `lumen`, and the lumen
     assignment then overwrote it, so the inner half of the wall band was never filtered as wall.
  4. No hard-coded paths, works on arrays, SRAD can be restricted to a bounding box for speed.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi


def build_rois(mask, wall_in=3, wall_out=3):
    """Split a binary lumen mask into three disjoint regions.

    lumen : lumen eroded by `wall_in` px (the homogeneous interior)
    wall  : thin band around the lumen boundary, `wall_in` inside to `wall_out` outside
    outer : everything beyond the band (surrounding pelvic tissue)
    """
    m = mask.astype(bool)
    inner = ndi.binary_erosion(m, iterations=wall_in) if wall_in > 0 else m
    outer_edge = ndi.binary_dilation(m, iterations=wall_out) if wall_out > 0 else m
    lumen = inner
    wall = outer_edge & ~inner
    outer = ~outer_edge
    return lumen, wall, outer


def srad(img, iters=30, dt=0.15, q0_region=None, eps=1e-3, barrier=None):
    """Speckle-reducing anisotropic diffusion (Yu & Acton 2002) on an image in [0, 1].

    q0_region: boolean mask of a homogeneous area (the lumen core) used to estimate the
    speckle scale q0^2. If None or too small, the whole frame is used.
    """
    I = np.clip(img.astype(np.float64), eps, 1.0)
    use_roi = q0_region is not None and q0_region.sum() > 50
    if barrier is not None:
        lab = barrier.astype(np.int8)
        Pb = np.pad(lab, 1, mode="edge")
        sN = (Pb[:-2, 1:-1] == lab).astype(np.float64)
        sS = (Pb[2:, 1:-1] == lab).astype(np.float64)
        sW = (Pb[1:-1, :-2] == lab).astype(np.float64)
        sE = (Pb[1:-1, 2:] == lab).astype(np.float64)
    for _ in range(iters):
        P = np.pad(I, 1, mode="edge")
        dN = P[:-2, 1:-1] - I
        dS = P[2:, 1:-1] - I
        dW = P[1:-1, :-2] - I
        dE = P[1:-1, 2:] - I
        G2 = (dN**2 + dS**2 + dW**2 + dE**2) / (I**2)
        L = (dN + dS + dW + dE) / I
        q2 = (0.5 * G2 - (1.0 / 16.0) * L**2) / ((1.0 + 0.25 * L) ** 2 + 1e-12)
        q02 = np.mean(q2[q0_region]) if use_roi else np.mean(q2)
        c = 1.0 / (1.0 + (q2 - q02) / (q02 * (1.0 + q02) + 1e-12))
        c = np.clip(c, 0.0, 1.0)
        Pc = np.pad(c, 1, mode="edge")
        if barrier is not None:
            # no flux between pixels with different barrier labels (keeps the wall from diffusing into the lumen)
            div = (Pc[:-2, 1:-1] * dN * sN + Pc[2:, 1:-1] * dS * sS + Pc[1:-1, :-2] * dW * sW + Pc[1:-1, 2:] * dE * sE)
        else:
            div = Pc[:-2, 1:-1] * dN + Pc[2:, 1:-1] * dS + Pc[1:-1, :-2] * dW + Pc[1:-1, 2:] * dE
        I = np.clip(I + (dt / 4.0) * div, 0.0, 1.0)
    return I


def roi_filter(img_u8, mask, srad_iters=30, wall_in=3, wall_out=3, guard=False, guard_in=None):
    """ROI-guided despeckling: SRAD in lumen, bilateral on wall, light Gaussian outside.

    img_u8: uint8 grayscale image. mask: binary lumen mask. Returns uint8.
    guard : stop the diffusion at the lumen boundary (SRAD cannot pull wall brightness into the lumen).
    """
    lumen, wall, outer = build_rois(mask, wall_in, wall_out)
    img_f = img_u8.astype(np.float64) / 255.0

    # guard_in: diffuse only inside the lumen eroded by this many px (wider than the wall echo that leaks inward)
    region = lumen
    if guard and guard_in is not None and guard_in > wall_in:
        region = ndi.binary_erosion(mask.astype(bool), iterations=guard_in)
    lumen_out = (srad(img_f, iters=srad_iters, q0_region=lumen, barrier=region if guard else None) * 255.0).astype(np.uint8)
    if guard and region is not lumen:
        # pixels between the guard region and the lumen core are left as they were
        lumen_out = np.where(region, lumen_out, img_u8)
    wall_out_img = cv2.bilateralFilter(img_u8, d=5, sigmaColor=40, sigmaSpace=40)
    outer_out = cv2.GaussianBlur(img_u8, (5, 5), sigmaX=1.0)

    result = outer_out.copy()
    result[wall] = wall_out_img[wall]
    result[lumen] = lumen_out[lumen]
    return cv2.GaussianBlur(result, (3, 3), sigmaX=0.5)


def roi_filter_protected(img_u8, mask, protect, srad_iters=100, wall_in=3, wall_out=3):
    """ROI v3 ("detect, then protect"): like the guarded ROI filter, but pixels in `protect` (a detected device candidate)
    are walled off from the diffusion and keep their original values, so strong smoothing of the rest of the lumen cannot erase them.
    With an empty `protect` this is identical to roi_filter(..., guard=True).
    """
    lumen, wall, outer = build_rois(mask, wall_in, wall_out)
    prot = protect.astype(bool) & lumen
    img_f = img_u8.astype(np.float64) / 255.0
    labels = lumen.astype(np.int8) + 2 * prot.astype(np.int8)
    q0 = lumen & ~prot
    if q0.sum() < 50:
        q0 = lumen
    lumen_out = (srad(img_f, iters=srad_iters, q0_region=q0, barrier=labels) * 255.0).astype(np.uint8)
    lumen_out = np.where(prot, img_u8, lumen_out)
    wall_out_img = cv2.bilateralFilter(img_u8, d=5, sigmaColor=40, sigmaSpace=40)
    outer_out = cv2.GaussianBlur(img_u8, (5, 5), sigmaX=1.0)
    result = outer_out.copy()
    result[wall] = wall_out_img[wall]
    result[lumen] = lumen_out[lumen]
    return cv2.GaussianBlur(result, (3, 3), sigmaX=0.5)
