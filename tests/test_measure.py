"""Tests for the measurement helpers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from measure import dice, skeleton_length, span_px  # noqa: E402


def test_span_of_line_and_two_blobs():
    m = np.zeros((60, 60), bool)
    m[30, 10:41] = True                      # horizontal line, 31 px
    assert abs(span_px(m) - 31.0) < 1.5
    m2 = np.zeros((60, 60), bool)
    m2[10:13, 10:13] = True
    m2[40:43, 50:53] = True                  # two separate echoes: span covers both
    assert span_px(m2) > 50


def test_skeleton_length_of_line():
    m = np.zeros((40, 40), bool)
    m[20, 5:35] = True
    assert abs(skeleton_length(m) - 30.0) < 3.0


def test_dice_identity_and_disjoint():
    a = np.zeros((20, 20), bool)
    a[5:10, 5:10] = True
    b = np.zeros_like(a)
    b[15:18, 15:18] = True
    assert dice(a, a) == 1.0
    assert dice(a, b) == 0.0
