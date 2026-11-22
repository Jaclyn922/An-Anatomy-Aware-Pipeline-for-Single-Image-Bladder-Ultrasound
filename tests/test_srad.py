"""Sanity tests for srad(): it must smooth speckle, keep the mean, and keep a step edge.

Run:  python tests/test_srad.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roi import srad  # noqa: E402


def speckle(shape, mean, rng):
    # fully developed speckle amplitude is Rayleigh; normalise to mean 1 then scale
    n = rng.rayleigh(scale=1.0, size=shape)
    return mean * n / n.mean()


def cv(x):
    return x.std() / x.mean()


def test_flat_patch():
    rng = np.random.default_rng(0)
    img = np.clip(speckle((128, 128), 0.3, rng), 0, 1)
    out = srad(img, iters=50)
    core = (slice(16, -16), slice(16, -16))
    ratio = cv(out[core]) / cv(img[core])
    drift = abs(out[core].mean() - img[core].mean()) / img[core].mean()
    print(f"flat patch : CV {cv(img[core]):.3f} -> {cv(out[core]):.3f} (x{ratio:.2f}), mean drift {drift:.1%}")
    assert not np.isnan(out).any()
    assert ratio < 0.7, "SRAD should cut the coefficient of variation clearly"
    assert drift < 0.05, "SRAD should preserve the mean"


def test_step_edge():
    rng = np.random.default_rng(1)
    clean = np.full((128, 128), 0.2)
    clean[:, 64:] = 0.6
    img = np.clip(clean * speckle(clean.shape, 1.0, rng), 0, 1)
    out = srad(img, iters=50)
    left = out[16:-16, 40:56].mean()
    right = out[16:-16, 72:88].mean()
    contrast_in = img[16:-16, 72:88].mean() - img[16:-16, 40:56].mean()
    contrast_out = right - left
    print(f"step edge  : contrast {contrast_in:.3f} -> {contrast_out:.3f}")
    assert contrast_out > 0.8 * contrast_in, "SRAD should not wash out a clear step edge"


def test_no_nan_with_black_background():
    rng = np.random.default_rng(2)
    img = np.zeros((128, 128))
    img[32:96, 32:96] = np.clip(speckle((64, 64), 0.3, rng), 0, 1)
    out = srad(img, iters=30)
    print(f"black bg   : NaN present = {np.isnan(out).any()}")
    assert not np.isnan(out).any()


def test_barrier_limits_exchange():
    """With a barrier no brightness crosses it, so each side changes far less than without one."""
    rng = np.random.default_rng(3)
    n = 96
    left = speckle((n, 48), 0.08, rng)
    right = speckle((n, 48), 0.6, rng)
    img = np.clip(np.concatenate([left, right], axis=1), 0, 1)
    lab = np.zeros((n, n), bool)
    lab[:, :48] = True
    before = img[:, :48].mean()
    held = srad(img, iters=150, q0_region=lab, barrier=lab)
    free = srad(img, iters=150, q0_region=lab)
    d_held = abs(held[:, :48].mean() - before) / before
    d_free = abs(free[:, :48].mean() - before) / before
    print(f"barrier    : mean of the dark side changes by {d_held:.4%} with barrier, {d_free:.4%} without")
    # SRAD's own discretisation is not exactly conservative (the flat-patch test shows ~2.5% mean drift),
    # so the barrier is judged against that floor: it must remove most of the exchange across the boundary.
    assert d_held < 0.04
    assert d_held < 0.5 * d_free


if __name__ == "__main__":
    test_flat_patch()
    test_step_edge()
    test_no_nan_with_black_background()
    test_barrier_limits_exchange()
    print("all SRAD tests passed")
