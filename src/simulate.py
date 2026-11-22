"""Step 2: a phenomenological B-mode simulator with a known clean reference.

Two stages, kept apart on purpose:

  build_scene()  anatomy + device + acoustics  ->  an expected-reflectivity map R_eff
                 (what the tissue would return on average). Fixed by the scene seed.
  render()       acquisition settings + noise  ->  a noisy image and its clean reference.
                 The noise realisation is fixed by the noise seed.

Because they are separate, one scene can be rendered many times with different noise seeds
(the 'same scene, different speckle' test) or with different gain / dynamic range / resolution.

Model in one paragraph. Every pixel is a tiny scatterer whose strength is sqrt(R). A complex
Gaussian random number per pixel gives fully developed speckle; blurring I and Q with the
system PSF gives the speckle grain; |I + iQ| is the envelope (Rayleigh distributed), and
20*log10 maps it to the display. The CLEAN image uses the same PSF and the same log mapping
but replaces the random envelope by its geometric mean, so noisy and clean agree on average
in the log domain. Beam direction is simplified to vertical (the real frames are sectors).
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from scipy.special import digamma


def _log_bias(looks):
    """E[ln env] - ln(s) for the envelope of `looks` independent speckle patterns averaged in power.
    looks=1 is the Rayleigh case: (ln 2 - gamma) / 2."""
    return 0.5 * (np.log(2.0) + digamma(looks) - np.log(looks))


@dataclass
class SceneParams:
    # anatomy
    lumen_floor: float = 0.004       # reflectivity inside the (almost anechoic) lumen
    tissue_level: float = 0.10       # mean reflectivity of the surrounding tissue
    tissue_log_std: float = 0.6      # strength of the large-scale tissue texture (log-normal)
    tissue_corr_px: float = 5.0      # correlation length of that texture
    wall_level: float = 1.0          # peak reflectivity of the bladder wall
    wall_offset_px: float = 1.5      # wall peak sits this far outside the lumen boundary
    wall_width_px: float = 1.6
    # acoustics
    alpha: float = 0.004             # amplitude attenuation per px inside tissue
    tgc: float = 0.7                 # fraction of that attenuation compensated by the machine
    # device (only used when a device mask is given)
    device_gain: float = 1.5         # edge reflectivity relative to the wall (1.5 strong, 0.6 weak)
    device_interior: float = 0.03
    device_edge_px: float = 1.5
    shadow_strength: float = 0.85    # 0 = no shadow, 1 = complete
    reverb_n: int = 3
    reverb_decay: float = 0.45
    reverb_spacing_px: float = 5.0


@dataclass
class RenderParams:
    gain_db: float = 0.0
    dr_db: float = 60.0
    psf_lat_px: float = 1.5          # Gaussian sigma across the beam (columns)
    psf_ax_px: float = 1.0           # Gaussian sigma along the beam (rows)
    looks: int = 1                   # independent speckle patterns averaged (spatial compounding); 1 = fully developed
    floor_u8: float = 0.0            # black level: the scanner shows nothing darker than this grey level (0-255)
    jitter_u8: float = 0.0           # small electronic/quantisation jitter added to the noisy image (grey levels)


@dataclass
class Scene:
    R_eff: np.ndarray                # expected reflectivity incl. attenuation, shadow, reverberation
    lumen: np.ndarray
    fov: np.ndarray
    device: object                   # bool mask or None


def fov_mask(gray_u8, thr=8):
    """Valid imaging region (the fan) of a real frame; everything else is black padding."""
    m = ndi.binary_closing(gray_u8 > thr, iterations=3)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def build_scene(lumen, fov, p=None, scene_seed=0, device=None):
    p = p or SceneParams()
    rng = np.random.default_rng(scene_seed)
    lumen = lumen.astype(bool)
    fov = fov.astype(bool)

    # signed distance to the lumen boundary: > 0 outside, < 0 inside. Using a distance (not the 0/1
    # mask) makes the wall a smooth profile, so no pixel staircase is baked into the phantom.
    sd = ndi.distance_transform_edt(~lumen) - ndi.distance_transform_edt(lumen)
    w_out = 1.0 / (1.0 + np.exp(-(sd - 0.5) / 1.0))

    tex = ndi.gaussian_filter(rng.standard_normal(lumen.shape), p.tissue_corr_px)
    tex /= tex.std()
    tissue = p.tissue_level * np.exp(p.tissue_log_std * tex - 0.5 * p.tissue_log_std**2)
    wall = p.wall_level * np.exp(-0.5 * ((sd - p.wall_offset_px) / p.wall_width_px) ** 2)
    R0 = p.lumen_floor * (1.0 - w_out) + tissue * w_out + wall

    # two-way attenuation along the beam, partly compensated by TGC. Water (lumen) does not
    # attenuate, so tissue behind the lumen comes out brighter (posterior enhancement).
    depth = np.cumsum(fov, axis=0)
    alpha_map = p.alpha * w_out * fov
    att = np.exp(-2.0 * (np.cumsum(alpha_map, axis=0) - p.tgc * p.alpha * depth))
    R_eff = R0 * att

    dev = None
    if device is not None and np.any(device):
        dev = device.astype(bool) & fov
        dist_in = ndi.distance_transform_edt(dev)
        ring = dev & (dist_in <= p.device_edge_px)
        R_dev = np.where(ring, p.device_gain * p.wall_level, p.device_interior) * att

        # acoustic shadow: everything below the device in its columns is dimmed
        below = np.maximum.accumulate(dev, axis=0)
        sh = ndi.gaussian_filter(below.astype(float), sigma=(0.0, 1.0))
        shadow = 1.0 - p.shadow_strength * np.clip(sh, 0.0, 1.0)
        R_eff = R_eff * shadow

        # reverberation: weaker copies of the bright edge, stacked below the device
        rev = np.zeros_like(R_eff)
        ring_f = ring.astype(float)
        for k in range(1, p.reverb_n + 1):
            s = int(round(k * p.reverb_spacing_px))
            if 0 < s < ring_f.shape[0]:
                shifted = np.zeros_like(ring_f)
                shifted[s:] = ring_f[:-s]
                rev += (p.reverb_decay**k) * shifted
        rev = ndi.gaussian_filter(rev, sigma=0.8)
        R_eff = R_eff + 0.5 * p.device_gain * p.wall_level * rev * att * (~dev)

        R_eff = np.where(dev, R_dev, R_eff)

    return Scene(R_eff=R_eff * fov, lumen=lumen, fov=fov, device=dev)


def _to_display(env, rp):
    db = 20.0 * np.log10(env + 1e-9) + rp.gain_db
    return np.clip((db + rp.dr_db) / rp.dr_db, 0.0, 1.0)


def render(scene, noise_seed, rp=None):
    """Return (noisy, clean) float images in [0, 1] for one noise realisation."""
    rp = rp or RenderParams()
    rng = np.random.default_rng(noise_seed)
    R = scene.R_eff
    sig = (rp.psf_ax_px, rp.psf_lat_px)  # (rows, cols)
    k = np.sqrt(4.0 * np.pi * rp.psf_ax_px * rp.psf_lat_px)  # makes var(I) = blur(R, sigma/sqrt2)
    amp = np.sqrt(R) * k
    power = np.zeros_like(R)
    for _ in range(max(1, int(rp.looks))):
        I = ndi.gaussian_filter(amp * rng.standard_normal(R.shape), sigma=sig)
        Q = ndi.gaussian_filter(amp * rng.standard_normal(R.shape), sigma=sig)
        power += I * I + Q * Q
    env = np.sqrt(power / max(1, int(rp.looks)))

    var = ndi.gaussian_filter(R, sigma=(sig[0] / np.sqrt(2.0), sig[1] / np.sqrt(2.0)))
    env_clean = np.exp(_log_bias(max(1, int(rp.looks)))) * np.sqrt(var)

    f = scene.fov
    floor = rp.floor_u8 / 255.0
    noisy = np.maximum(_to_display(env, rp), floor)
    clean = np.maximum(_to_display(env_clean, rp), floor)
    if rp.jitter_u8 > 0:
        noisy = np.clip(noisy + rng.normal(0.0, rp.jitter_u8 / 255.0, noisy.shape), 0.0, 1.0)
    return noisy * f, clean * f


def to_u8(img01):
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def load_calibration(path):
    """Read the parameters fitted by calibrate.py -> (SceneParams, RenderParams)."""
    import json

    with open(path) as f:
        P = json.load(f)
    sp = SceneParams(lumen_floor=P["lumen_floor"], wall_level=P["wall_level"], tissue_level=P["tissue_level"])
    rp = RenderParams(gain_db=P["gain_db"], dr_db=P["dr_db"], psf_lat_px=P["psf_lat_px"], psf_ax_px=P["psf_ax_px"],
                      looks=int(P.get("looks", 1)), floor_u8=P.get("floor_u8", 0.0), jitter_u8=P.get("jitter_u8", 0.0))
    return sp, rp
