"""Evaluation metrics for sparse depth ablations."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def finite_float(x) -> float:
    try:
        x = float(x)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def safe_median(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def safe_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def compute_depth_metrics(z_pred, z_gt):
    """Pointwise sparse-depth error metrics against matched LiDAR depths."""
    z_pred = np.asarray(z_pred, dtype=np.float64)
    z_gt = np.asarray(z_gt, dtype=np.float64)
    valid = np.isfinite(z_pred) & np.isfinite(z_gt) & (z_pred > 0) & (z_gt > 0)
    z_pred = z_pred[valid]
    z_gt = z_gt[valid]
    if z_pred.size == 0:
        return {
            "median_abs_err_m": float("nan"),
            "mean_abs_err_m": float("nan"),
            "rmse_m": float("nan"),
            "median_rel_err": float("nan"),
            "mean_rel_err": float("nan"),
            "delta_10": float("nan"),
            "delta_20": float("nan"),
            "delta_30": float("nan"),
            "delta_125": float("nan"),
        }
    err = z_pred - z_gt
    abs_err = np.abs(err)
    rel_err = abs_err / np.maximum(z_gt, 1e-12)
    ratio = np.maximum(z_pred / np.maximum(z_gt, 1e-12), z_gt / np.maximum(z_pred, 1e-12))
    return {
        "median_abs_err_m": float(np.median(abs_err)),
        "mean_abs_err_m": float(np.mean(abs_err)),
        "rmse_m": float(np.sqrt(np.mean(err * err))),
        "median_rel_err": float(np.median(rel_err)),
        "mean_rel_err": float(np.mean(rel_err)),
        "delta_10": float(np.mean(rel_err <= 0.10)),
        "delta_20": float(np.mean(rel_err <= 0.20)),
        "delta_30": float(np.mean(rel_err <= 0.30)),
        "delta_125": float(np.mean(ratio < 1.25)),
    }


def compute_scale_factors(z_tri, z_lidar):
    """Return least-squares and median scale factors mapping triangulated depth to LiDAR depth."""
    z_tri = np.asarray(z_tri, dtype=np.float64)
    z_lidar = np.asarray(z_lidar, dtype=np.float64)
    valid = np.isfinite(z_tri) & np.isfinite(z_lidar) & (z_tri > 1e-9) & (z_lidar > 1e-9)
    zt = z_tri[valid]
    zl = z_lidar[valid]
    if zt.size == 0:
        return float("nan"), float("nan")
    alpha_l2 = float(np.sum(zt * zl) / (np.sum(zt * zt) + 1e-12))
    alpha_med = float(np.median(zl / zt))
    return alpha_l2, alpha_med


def in_bounds_mask(pts: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """Return a mask for points that actually lie inside the image.

    Coverage should not be inflated by predicted points that fall just outside
    the image. This is the same basic guard used by the match diagnostics script:
    first keep only valid image-plane points, then map them into coarse cells.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    h, w = img_shape[:2]
    return (
        np.isfinite(pts[:, 0])
        & np.isfinite(pts[:, 1])
        & (pts[:, 0] >= 0)
        & (pts[:, 0] < w)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < h)
    )


def grid_occupancy(pts: np.ndarray, img_shape: Tuple[int, int], cols: int, rows: int) -> np.ndarray:
    """Count how many sparse points land in each image-space grid cell.

    This deliberately measures spatial spread of the *final point set* rather
    than track-manager bucket health. A frame with 800 points all clustered in
    one tree gets poor occupancy; a frame with fewer points distributed across
    the image can score better. That is the notion of sparse-depth coverage we
    want for ablations.
    """
    h, w = img_shape[:2]
    occ = np.zeros((rows, cols), dtype=np.int32)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return occ

    pts_v = pts[in_bounds_mask(pts, img_shape)]
    if pts_v.size == 0:
        return occ

    xs = np.clip((pts_v[:, 0] / max(w, 1) * cols).astype(int), 0, cols - 1)
    ys = np.clip((pts_v[:, 1] / max(h, 1) * rows).astype(int), 0, rows - 1)
    for x, y in zip(xs, ys):
        occ[y, x] += 1
    return occ


def coverage_stats(prefix: str, pts: np.ndarray, img_shape: Tuple[int, int], cols: int, rows: int) -> Dict[str, float]:
    """Report coarse scene coverage for a sparse point cloud in image space.

    The headline field is `<prefix>_coverage`: occupied grid cells divided by
    total grid cells. Entropy is normalized to [0, 1], where 1 means the points
    are evenly spread across occupied cells and lower values mean clumping.
    """
    occ = grid_occupancy(pts, img_shape, cols, rows)
    total_cells = max(int(cols * rows), 1)
    occupied = int(np.sum(occ > 0))
    vals = occ[occ > 0].astype(np.float64)
    total_pts = float(np.sum(vals))

    if total_pts > 0:
        p = vals / total_pts
        entropy = -float(np.sum(p * np.log(p + 1e-12)))
        max_entropy = np.log(max(occupied, 1))
        entropy_norm = float(entropy / max(max_entropy, 1e-12)) if occupied > 1 else 0.0
    else:
        entropy_norm = 0.0

    return {
        f"{prefix}_coverage": float(occupied / total_cells),
        f"{prefix}_occupied_cells": float(occupied),
        f"{prefix}_mean_pts_per_occ_cell": float(np.mean(vals)) if vals.size else 0.0,
        f"{prefix}_max_cell_count": float(np.max(vals)) if vals.size else 0.0,
        f"{prefix}_spatial_entropy": entropy_norm,
    }
