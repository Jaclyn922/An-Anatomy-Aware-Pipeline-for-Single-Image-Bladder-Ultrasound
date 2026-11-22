"""A TAR-210-inspired intravesical test object: a pretzel-like 8-shaped tube, sliced by the imaging plane.

The shape follows the public schematic only (two interlocking loops). Its true size, material and
acoustic appearance are NOT known here, so size and tube radius are sampled from ranges and the
object is called 'device-like', never 'TAR-210'.

How it works (all in pixel units of the target image):
  1. pretzel_curve(): a closed 3D centre-line. Its two crossings are separated in z so the tube
     never intersects itself.
  2. The curve is scaled, given a random 3D rotation, and moved to a random position inside the lumen.
     The imaging plane is z = 0, so a random z-offset decides how much of the object the plane cuts.
  3. A pixel (x, y) belongs to the device if the 3D point (x, y, 0) is closer than the tube radius to
     the centre-line. That is the exact intersection of a plane with a tube.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def pretzel_curve(n=1500, z_amp=0.25):
    """Closed 8-shaped centre-line, half-width 1 along x. Shape: Gerono lemniscate, crossings split in z."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(t)
    y = 1.2 * np.sin(t) * np.cos(t)
    z = z_amp * np.sin(t)
    return np.stack([x, y, z], axis=1)


def _equivalent_diameter(mask):
    return 2.0 * np.sqrt(mask.sum() / np.pi)


def make_device_mask(
    lumen_mask,
    rng,
    size_frac=(0.35, 0.7),
    tube_frac=(0.025, 0.05),
    touch_wall=False,
    min_px=25,
    max_tries=40,
):
    """Return (device_mask[bool HxW], params) for one random pose, or (None, None) if every try missed.

    size_frac : device width as a fraction of the lumen's equivalent diameter
    tube_frac : tube radius as a fraction of the lumen's equivalent diameter
    touch_wall: put the centre so the device reaches the wall (stress test for the ROI split)
    """
    lumen = lumen_mask.astype(bool)
    H, W = lumen.shape
    D = _equivalent_diameter(lumen)
    dt = ndi.distance_transform_edt(lumen)
    ys, xs = np.nonzero(lumen)
    curve = pretzel_curve()
    lumen_grown = ndi.binary_dilation(lumen, iterations=2)

    for attempt in range(1, max_tries + 1):
        L = rng.uniform(*size_frac) * D
        scale = L / 2.0
        r = max(1.5, rng.uniform(*tube_frac) * D)

        if touch_wall:
            ok = (dt >= 0.25 * L) & (dt <= 0.45 * L)
        else:
            ok = dt >= 0.5 * L * 0.9
        cand = np.nonzero(ok[ys, xs])[0]
        if cand.size == 0:
            cand = np.arange(ys.size)
        k = rng.choice(cand)
        cx, cy = xs[k], ys[k]
        cz = rng.uniform(-0.35, 0.35) * L

        R = Rotation.random(random_state=int(rng.integers(0, 2**31 - 1)))
        pts = R.apply(curve * scale) + np.array([cx, cy, cz])

        x0, x1 = int(max(0, pts[:, 0].min() - r - 1)), int(min(W - 1, pts[:, 0].max() + r + 1))
        y0, y1 = int(max(0, pts[:, 1].min() - r - 1)), int(min(H - 1, pts[:, 1].max() + r + 1))
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        query = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        dist, _ = cKDTree(pts).query(query)
        inside = (dist < r).reshape(gx.shape)

        mask = np.zeros((H, W), dtype=bool)
        mask[y0 : y1 + 1, x0 : x1 + 1] = inside
        mask &= lumen_grown  # a device cannot sit inside the wall tissue
        if mask.sum() >= min_px:
            params = dict(
                tries=attempt,
                diameter_px=float(L),
                tube_radius_px=float(r),
                center=(int(cx), int(cy), float(cz)),
                rotvec=R.as_rotvec().round(3).tolist(),
                touch_wall=bool(touch_wall),
                area_px=int(mask.sum()),
            )
            return mask, params
    return None, None
