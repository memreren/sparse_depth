"""Direct gradient-based time-to-contact (TTC) estimation.

This is the *direct* branch of the TTC-vs-triangulation study: it estimates
time-to-contact tau = Z / Vz straight from image brightness gradients, with NO
feature correspondences, NO triangulation, and NO metric scale. It is a
deliberately different information source from ``triangulation.py`` (which needs
tracked matches and a known metric pose), so the two can be compared on a common
ground (tau error vs LiDAR-derived GT tau) rather than being reparameterizations
of each other.

Method (Horn / Fang / Masaki, "Time to Contact Relative to a Planar Surface").
Every pixel obeys the brightness-constancy constraint

    E_x * u + E_y * v + E_t = 0,

where (E_x, E_y) are spatial gradients and E_t is the temporal gradient between
two frames. For motion toward a locally fronto-parallel surface the image motion
is a radial expansion about the focus of expansion (FOE) (x0, y0):

    u = (x - x0) / tau,     v = (y - y0) / tau.

Substituting collapses the constraint to a single unknown ``1/tau``:

    (1/tau) * G + E_t = 0,     G := E_x (x - x0) + E_y (y - y0),

so a least-squares fit over a patch gives, in one shot and without ever forming
the flow field,

    1/tau = sum(G * E_t) / sum(G * G),     with E_t = I_curr - I_prev.

Two practical points handled here:
  * Rotation is removed up front by warping the previous frame through the known
    infinite homography ``K R K^-1`` (GT rotation), leaving purely translational
    (expansion) residual motion. Without this the FOE-centered expansion model
    does not hold.
  * The linearized brightness constraint assumes small (sub-pixel-ish) motion.
    The expansion rate ``1/tau`` is *invariant to image downscaling* (halving the
    image halves positions and displacements together, leaving the ratio fixed),
    so we simply evaluate on a downsampled pyramid level where the inter-frame
    motion is small. That is why ``pyramid_level`` exists.

The estimator is pure (no file IO); the eval script owns KITTI loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DirectTTCConfig:
    patch_radius_px: int = 7          # half-window (in pyramid-level pixels)
    pyramid_level: int = 2            # downsample factor 2**level for gradient validity
    min_structure: float = 1e3        # min sum(G^2) in a patch to trust the fit
    min_valid_frac: float = 0.85      # min fraction of de-rotated (in-view) patch pixels
    min_tau_frames: float = 0.5       # reject implausibly small/negative tau
    max_tau_frames: float = 400.0     # reject implausibly large (no-expansion) tau


def image_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference spatial gradients (E_x, E_y) of a float image."""
    gy, gx = np.gradient(img.astype(np.float64))
    return gx, gy


def infinite_homography(K: np.ndarray, K_inv: np.ndarray, R_prev_to_curr: np.ndarray) -> np.ndarray:
    """Homography mapping infinitely-far (rotation-only) content prev -> curr."""
    return K @ R_prev_to_curr @ K_inv


def foe_pixel(K: np.ndarray, t_prev_to_curr: np.ndarray) -> np.ndarray:
    """Focus of expansion = projection of the translation direction, in pixels."""
    f = K @ np.asarray(t_prev_to_curr, dtype=np.float64).reshape(3)
    return f[:2] / (f[2] + 1e-12)


def derotate_previous(
    prev_gray: np.ndarray,
    K: np.ndarray,
    K_inv: np.ndarray,
    R_prev_to_curr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp the previous frame into the current orientation.

    Returns the de-rotated previous image (float) and a [0,1] validity mask that
    is 0 where the warp pulled in out-of-view (border) content.
    """
    h, w = prev_gray.shape[:2]
    H = infinite_homography(K, K_inv, R_prev_to_curr)
    prev_f = prev_gray.astype(np.float64)
    warped = cv2.warpPerspective(prev_f, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    valid = cv2.warpPerspective(np.ones((h, w), dtype=np.float64), H, (w, h),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0.0)
    return warped, valid


def _downsample(img: np.ndarray, level: int) -> np.ndarray:
    out = img
    for _ in range(int(level)):
        out = cv2.pyrDown(out)
    return out


class DirectTTCFrame:
    """Precomputed per-frame-pair fields for repeated point queries.

    Building this once per (prev, curr) pair and then querying many pixels keeps
    the per-point cost to a windowed sum. All fields live at the chosen pyramid
    level; the FOE and query points are scaled into that level on the fly.
    """

    def __init__(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        K: np.ndarray,
        R_prev_to_curr: np.ndarray,
        t_prev_to_curr: np.ndarray,
        cfg: DirectTTCConfig,
    ):
        self.cfg = cfg
        self.scale = float(2 ** int(cfg.pyramid_level))
        K_inv = np.linalg.inv(K)

        prev_derot, valid = derotate_previous(prev_gray, K, K_inv, R_prev_to_curr)
        curr_f = curr_gray.astype(np.float64)

        prev_s = _downsample(prev_derot, cfg.pyramid_level)
        curr_s = _downsample(curr_f, cfg.pyramid_level)
        valid_s = _downsample(valid, cfg.pyramid_level)
        # align shapes (pyrDown rounding can differ by a row/col)
        h = min(prev_s.shape[0], curr_s.shape[0], valid_s.shape[0])
        w = min(prev_s.shape[1], curr_s.shape[1], valid_s.shape[1])
        prev_s, curr_s, valid_s = prev_s[:h, :w], curr_s[:h, :w], valid_s[:h, :w]

        avg = 0.5 * (prev_s + curr_s)
        self.Ex, self.Ey = image_gradients(avg)
        self.Et = curr_s - prev_s
        self.valid = valid_s
        self.h, self.w = h, w

        foe = foe_pixel(K, t_prev_to_curr) / self.scale
        self.x0, self.y0 = float(foe[0]), float(foe[1])
        xs = np.arange(w, dtype=np.float64)[None, :]
        ys = np.arange(h, dtype=np.float64)[:, None]
        # radial gradient field G = E_x (x - x0) + E_y (y - y0), computed once
        self.G = self.Ex * (xs - self.x0) + self.Ey * (ys - self.y0)

    def estimate_points(self, uv_full: np.ndarray) -> np.ndarray:
        """Return tau (frames) for each full-resolution pixel; NaN where untrusted."""
        cfg = self.cfg
        r = int(cfg.patch_radius_px)
        uv = np.asarray(uv_full, dtype=np.float64).reshape(-1, 2)
        out = np.full(uv.shape[0], np.nan, dtype=np.float64)
        cx = uv[:, 0] / self.scale
        cy = uv[:, 1] / self.scale
        for i in range(uv.shape[0]):
            xc = int(round(cx[i]))
            yc = int(round(cy[i]))
            if xc - r < 0 or yc - r < 0 or xc + r + 1 > self.w or yc + r + 1 > self.h:
                continue
            sl = (slice(yc - r, yc + r + 1), slice(xc - r, xc + r + 1))
            vwin = self.valid[sl]
            if float(np.mean(vwin)) < cfg.min_valid_frac:
                continue
            Gw = self.G[sl]
            Etw = self.Et[sl]
            gg = float(np.sum(Gw * Gw))
            if gg < cfg.min_structure:
                continue
            # Sign convention verified against a known synthetic looming warp:
            # for E_t = I_curr - I_prev and G the radial-gradient field, the
            # expansion rate is +sum(G*E_t)/sum(G^2). (A residual few-percent
            # underestimate remains from the small-motion linearization; it
            # shrinks as inter-frame motion -> 0, which is what the pyramid buys.)
            inv_tau = float(np.sum(Gw * Etw)) / gg
            if abs(inv_tau) < 1e-9:
                continue
            tau = 1.0 / inv_tau
            if tau < cfg.min_tau_frames or tau > cfg.max_tau_frames:
                continue
            out[i] = tau
        return out
