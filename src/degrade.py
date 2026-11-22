"""Step 3a: sample random acquisition conditions and render matched present/absent scene pairs.

For every spec we render TWO images of the same anatomy, same tissue texture, same acquisition settings:
  present : with the device-like object (device mask = ground truth)
  absent  : without it, rendered with a DIFFERENT noise seed (a real single frame has no pairing)
plus the noise-free clean reference of each.

Degradations (all sampled, never a fixed grid):
  gain_offset_db  raises/lowers the overall gain. Turning gain up lifts lumen clutter above the black level,
                  turning it down sinks the device below it.
  dr_scale        dynamic range multiplier (compresses or stretches contrast)
  psf_scale       worse resolution = larger speckle grain
  looks           1 = fully developed speckle (worst), 2 = calibrated, 3 = smoother
  device_gain     echo strength of the device (log-uniform 0.3 .. 6)
  shadow_strength acoustic shadow behind the device
  touch_wall      device sits against the wall (stress test for the ROI split)
"""
import dataclasses
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from device import make_device_mask
from simulate import build_scene, fov_mask, render, to_u8

ROOT = Path(__file__).resolve().parents[1]


def sample_specs(frame_rows, n, seed=0):
    """frame_rows: one manifest row per source frame. Returns a list of spec dicts."""
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n):
        r = frame_rows[int(rng.integers(len(frame_rows)))]
        specs.append(
            dict(
                spec_id=i,
                file=r["file"],
                frame=r["frame"],
                video=r["video"],
                device_gain=float(np.exp(rng.uniform(np.log(0.3), np.log(6.0)))),
                shadow_strength=float(rng.uniform(0.3, 0.95)),
                gain_offset_db=float(rng.uniform(-6.0, 18.0)),
                dr_scale=float(rng.uniform(0.85, 1.3)),
                psf_scale=float(rng.uniform(1.0, 2.0)),
                looks=int(rng.choice([1, 2, 3], p=[0.25, 0.5, 0.25])),
                touch_wall=bool(rng.random() < 0.3),
                device_seed=int(rng.integers(0, 2**31 - 1)),
                scene_seed=int(rng.integers(0, 2**31 - 1)),
                noise_seed_present=int(rng.integers(0, 2**31 - 1)),
                noise_seed_absent=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return specs


@lru_cache(maxsize=128)
def _load(file):
    img = np.array(Image.open(ROOT / "data/roboflow" / file).convert("L"))
    m = np.array(Image.open(ROOT / "data/masks" / (Path(file).stem + "_mask.png"))) > 127
    return img, m, fov_mask(img)


def realise(spec, base_sp, base_rp):
    """Return a dict of arrays for one spec, or None if no device pose fitted inside the lumen."""
    img, lumen, fov = _load(spec["file"])
    dev, dparams = make_device_mask(lumen, np.random.default_rng(spec["device_seed"]), touch_wall=spec["touch_wall"])
    if dev is None:
        return None
    sp = dataclasses.replace(base_sp, device_gain=spec["device_gain"], shadow_strength=spec["shadow_strength"])
    rp = dataclasses.replace(
        base_rp,
        gain_db=base_rp.gain_db + spec["gain_offset_db"],
        dr_db=base_rp.dr_db * spec["dr_scale"],
        psf_lat_px=base_rp.psf_lat_px * spec["psf_scale"],
        psf_ax_px=base_rp.psf_ax_px * spec["psf_scale"],
        looks=spec["looks"],
    )
    sc_p = build_scene(lumen, fov, sp, spec["scene_seed"], device=dev)
    sc_a = build_scene(lumen, fov, sp, spec["scene_seed"], device=None)
    noisy_p, clean_p = render(sc_p, spec["noise_seed_present"], rp)
    noisy_a, clean_a = render(sc_a, spec["noise_seed_absent"], rp)
    return dict(
        noisy_p=to_u8(noisy_p), clean_p=to_u8(clean_p),
        noisy_a=to_u8(noisy_a), clean_a=to_u8(clean_a),
        device=sc_p.device if sc_p.device is not None else dev, lumen=lumen, fov=fov, device_params=dparams,
    )
