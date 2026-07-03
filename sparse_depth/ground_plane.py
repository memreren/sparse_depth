"""Local ground-plane geometry and direct-warp diagnostics.

The convention follows :func:`sparse_depth.geometry.relative_pose`: ``R_ts, t_ts``
map a target-camera 3-D point to its source-camera coordinates,
``X_s = R_ts @ X_t + t_ts``.  Planes are represented in target-camera
coordinates by ``n.T @ X + d = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Plane:
    """A normalized plane, ``normal.T @ X + offset = 0``."""

    normal: np.ndarray
    offset: float

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            raise ValueError("Plane normal must be non-zero")
        object.__setattr__(self, "normal", n / norm)
        object.__setattr__(self, "offset", float(self.offset) / norm)


def level_ground_plane(camera_height_m: float) -> Plane:
    """Return a level road plane for KITTI camera coordinates.

    KITTI rectified camera coordinates are right/down/forward.  Hence road
    points below a camera at height ``h`` satisfy ``Y = h``.
    """
    if camera_height_m <= 0.0:
        raise ValueError("camera_height_m must be positive")
    return Plane(np.array([0.0, 1.0, 0.0]), -float(camera_height_m))


def plane_homography(K: np.ndarray, R_ts: np.ndarray, t_ts: np.ndarray, plane: Plane) -> np.ndarray:
    """Return target-to-source homography induced by ``plane``.

    For ``X_s = R_ts X_t + t_ts`` and ``n.T X_t + d = 0``,
    ``H_ts = K (R_ts - t_ts n.T / d) K^-1``.
    """
    if abs(plane.offset) < 1e-12:
        raise ValueError("Plane offset must be non-zero for a camera-plane homography")
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R_ts = np.asarray(R_ts, dtype=np.float64).reshape(3, 3)
    t_ts = np.asarray(t_ts, dtype=np.float64).reshape(3)
    H = K @ (R_ts - np.outer(t_ts, plane.normal) / plane.offset) @ np.linalg.inv(K)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def ray_plane_depth(K: np.ndarray, image_shape: Tuple[int, int], plane: Plane) -> np.ndarray:
    """Compute target-camera Z depth for every pixel's ray-plane intersection.

    Invalid rays (parallel, behind camera) are returned as NaN.  Since the
    pixel ray is ``K^-1 [u,v,1]``, its scalar intersection parameter equals
    camera Z depth.
    """
    h, w = image_shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float64)
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=0).reshape(3, -1)
    rays = np.linalg.inv(np.asarray(K, dtype=np.float64).reshape(3, 3)) @ pixels
    denom = plane.normal @ rays
    lam = -plane.offset / np.where(np.abs(denom) > 1e-12, denom, np.nan)
    valid = np.isfinite(lam) & (lam > 0.0) & (rays[2] > 0.0)
    depth = np.full(lam.shape, np.nan, dtype=np.float64)
    depth[valid] = lam[valid] * rays[2, valid]
    return depth.reshape(h, w)


def road_trapezoid_mask(
    image_shape: Tuple[int, int], top_y_frac: float, bottom_left_frac: float,
    bottom_right_frac: float, top_left_frac: float, top_right_frac: float,
) -> np.ndarray:
    """Build a conservative normalized trapezoid support mask for road tests."""
    h, w = image_shape[:2]
    for name, value in {
        "top_y_frac": top_y_frac, "bottom_left_frac": bottom_left_frac,
        "bottom_right_frac": bottom_right_frac, "top_left_frac": top_left_frac,
        "top_right_frac": top_right_frac,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if top_left_frac >= top_right_frac or bottom_left_frac >= bottom_right_frac:
        raise ValueError("Trapezoid left edge must be left of right edge")
    y_top = int(round(top_y_frac * (h - 1)))
    poly = np.array([
        [round(top_left_frac * (w - 1)), y_top],
        [round(top_right_frac * (w - 1)), y_top],
        [round(bottom_right_frac * (w - 1)), h - 1],
        [round(bottom_left_frac * (w - 1)), h - 1],
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255, lineType=cv2.LINE_AA)
    return mask.astype(bool)


def warp_source_to_target(source: np.ndarray, H_target_to_source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a source image into target pixels and return its coverage mask."""
    h, w = source.shape[:2]
    H_source_to_target = np.linalg.inv(H_target_to_source)
    warped = cv2.warpPerspective(source, H_source_to_target, (w, h), flags=cv2.INTER_LINEAR)
    support = cv2.warpPerspective(
        np.full((h, w), 255, dtype=np.uint8), H_source_to_target, (w, h), flags=cv2.INTER_NEAREST,
    )
    return warped, support > 0


def robust_warp_residual(
    target: np.ndarray, warped_source: np.ndarray, support: np.ndarray,
    candidate_mask: np.ndarray, mad_scale: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Brightness-offset-corrected residual and a robust inlier mask.

    The offset accommodates small global exposure shifts.  The output inlier
    mask is only meaningful inside ``support & candidate_mask``; it does not
    claim that all matching pixels are road, only that they do not contradict
    the current plane warp.
    """
    target_f = np.asarray(target, dtype=np.float32)
    source_f = np.asarray(warped_source, dtype=np.float32)
    valid = np.asarray(support, dtype=bool) & np.asarray(candidate_mask, dtype=bool)
    residual = np.full(target_f.shape, np.nan, dtype=np.float32)
    inliers = np.zeros(target_f.shape, dtype=bool)
    if not np.any(valid):
        return residual, inliers, np.nan, np.nan, np.nan
    offset = float(np.median(target_f[valid] - source_f[valid]))
    residual = np.abs(target_f - (source_f + offset))
    vals = residual[valid]
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    threshold = median + float(mad_scale) * 1.4826 * max(mad, 1.0)
    inliers = valid & (residual <= threshold)
    residual[~valid] = np.nan
    return residual, inliers, offset, median, threshold


def local_zncc(
    target: np.ndarray, warped_source: np.ndarray, patch_radius_px: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local zero-mean normalized cross-correlation and patch stddevs.

    ZNCC compares local appearance after removing each patch's mean, so it is
    insensitive to a local additive brightness shift.  It is intentionally
    reported alongside the raw residual: a flat asphalt patch has undefined or
    unstable correlation, which is useful information rather than a reason to
    call it plane-consistent.
    """
    radius = int(patch_radius_px)
    if radius < 1:
        raise ValueError("patch_radius_px must be at least 1")
    target_f = np.asarray(target, dtype=np.float32)
    source_f = np.asarray(warped_source, dtype=np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_t = cv2.boxFilter(target_f, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_s = cv2.boxFilter(source_f, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_tt = cv2.boxFilter(target_f * target_f, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_ss = cv2.boxFilter(source_f * source_f, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_ts = cv2.boxFilter(target_f * source_f, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    var_t = np.maximum(mean_tt - mean_t * mean_t, 0.0)
    var_s = np.maximum(mean_ss - mean_s * mean_s, 0.0)
    denom = np.sqrt(var_t * var_s)
    zncc = np.full(target_f.shape, np.nan, dtype=np.float32)
    valid = denom > 1e-6
    zncc[valid] = np.clip((mean_ts[valid] - mean_t[valid] * mean_s[valid]) / denom[valid], -1.0, 1.0)
    return zncc, np.sqrt(var_t), np.sqrt(var_s)
