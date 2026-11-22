"""Checks that the simulator behaves like ultrasound should.

Run (from the project root):  python tests/test_simulate.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import RenderParams, Scene, SceneParams, build_scene, render  # noqa: E402


def flat_scene(n=256, level=0.1):
    fov = np.ones((n, n), bool)
    return Scene(R_eff=np.full((n, n), level), lumen=np.zeros((n, n), bool), fov=fov, device=None)


def test_rayleigh_statistics():
    # in a uniform medium the envelope must be Rayleigh: coefficient of variation sqrt((4-pi)/pi)
    sc = flat_scene()
    rp = RenderParams()
    rng = np.random.default_rng(0)
    from scipy import ndimage as ndi

    R = sc.R_eff
    k = np.sqrt(4 * np.pi * rp.psf_ax_px * rp.psf_lat_px)
    sig = (rp.psf_ax_px, rp.psf_lat_px)
    I = ndi.gaussian_filter(np.sqrt(R) * k * rng.standard_normal(R.shape), sigma=sig)
    Q = ndi.gaussian_filter(np.sqrt(R) * k * rng.standard_normal(R.shape), sigma=sig)
    env = np.hypot(I, Q)[20:-20, 20:-20]
    cv = env.std() / env.mean()
    print(f"rayleigh   : CV {cv:.3f} (theory 0.523)")
    assert abs(cv - 0.5227) < 0.03


def test_clean_is_unbiased_in_log_domain():
    sc = flat_scene()
    rp = RenderParams(gain_db=0.0, dr_db=80.0)
    noisy, clean = render(sc, noise_seed=1, rp=rp)
    a, b = noisy[20:-20, 20:-20].mean(), clean[20:-20, 20:-20].mean()
    print(f"log bias   : noisy mean {a:.4f}, clean {b:.4f}, diff {abs(a - b) * 80:.2f} dB")
    assert abs(a - b) * 80 < 0.4, "clean reference should match the noisy image on average (dB)"


def test_clean_unbiased_with_compounding():
    sc = flat_scene()
    for looks in (2, 3, 4):
        rp = RenderParams(gain_db=0.0, dr_db=80.0, looks=looks)
        noisy, clean = render(sc, noise_seed=2, rp=rp)
        d = abs(noisy[20:-20, 20:-20].mean() - clean[20:-20, 20:-20].mean()) * 80
        sd = noisy[20:-20, 20:-20].std() * 80
        print(f"looks={looks}   : log bias {d:.2f} dB, speckle std {sd:.2f} dB (1 look = 5.57)")
        assert d < 0.4
        assert sd < 5.57


def test_seeds():
    sc = flat_scene(64)
    n1, c1 = render(sc, 5)
    n2, c2 = render(sc, 5)
    n3, c3 = render(sc, 6)
    print(f"seeds      : same seed identical = {np.array_equal(n1, n2)}, different seed differs = {not np.allclose(n1, n3)}, clean equal = {np.array_equal(c1, c3)}")
    assert np.array_equal(n1, n2)
    assert not np.allclose(n1, n3)
    assert np.array_equal(c1, c3), "clean image must not depend on the noise seed"


def test_device_shadow_and_brightness():
    n = 128
    lumen = np.zeros((n, n), bool)
    lumen[30:100, 30:100] = True
    fov = np.ones((n, n), bool)
    dev = np.zeros((n, n), bool)
    dev[50:55, 60:66] = True
    below = (slice(60, 90), slice(61, 65))

    # shadow alone (reverberation switched off): echo below the device must drop clearly
    p = SceneParams(reverb_n=0)
    s0 = build_scene(lumen, fov, p, scene_seed=0, device=None)
    s1 = build_scene(lumen, fov, p, scene_seed=0, device=dev)
    shadow_ratio = s1.R_eff[below].mean() / s0.R_eff[below].mean()
    dev_bright = s1.R_eff[dev].max()
    wall_bright = s0.R_eff[28:32, 40:90].max()
    print(f"shadow     : echo below device = {shadow_ratio:.2f}x no-device; device peak {dev_bright:.2f} vs wall peak {wall_bright:.2f}")
    assert shadow_ratio < 0.5
    assert dev_bright > wall_bright

    # reverberation alone (shadow switched off): the dark lumen below the device gets extra echo
    pr = SceneParams(shadow_strength=0.0)
    r0 = build_scene(lumen, fov, pr, scene_seed=0, device=None)
    r1 = build_scene(lumen, fov, pr, scene_seed=0, device=dev)
    rev_ratio = r1.R_eff[below].mean() / r0.R_eff[below].mean()
    print(f"reverb     : echo below device = {rev_ratio:.2f}x no-device")
    assert rev_ratio > 1.5

    # no device -> identical to a scene built with an empty mask
    s2 = build_scene(lumen, fov, p, scene_seed=0, device=np.zeros_like(dev))
    assert np.array_equal(s0.R_eff, s2.R_eff)


if __name__ == "__main__":
    test_rayleigh_statistics()
    test_clean_is_unbiased_in_log_domain()
    test_clean_unbiased_with_compounding()
    test_seeds()
    test_device_shadow_and_brightness()
    print("all simulator tests passed")
