"""Tests for the detect-then-protect ROI filter."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roi import roi_filter, roi_filter_protected  # noqa: E402


def _scene():
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(40, 8, (100, 100)), 0, 255).astype(np.uint8)
    mask = np.zeros((100, 100), bool)
    mask[15:85, 15:85] = True
    dev = np.zeros_like(mask)
    dev[45:52, 45:52] = True
    img[dev] = 200
    return img, mask, dev


def test_empty_protect_equals_guarded_roi():
    img, mask, _ = _scene()
    a = roi_filter(img, mask, srad_iters=20, guard=True)
    b = roi_filter_protected(img, mask, np.zeros_like(mask), srad_iters=20)
    assert np.array_equal(a, b)


def test_protected_region_keeps_its_values():
    img, mask, dev = _scene()
    out = roi_filter_protected(img, mask, dev, srad_iters=200)
    assert abs(float(out[48, 48]) - 200.0) < 5.0          # centre of the protected block is untouched
    assert out[dev].mean() > 0.9 * img[dev].mean()
