"""Lightweight estimated-pose backend for the depth-from-motion study.

Produces an *estimated* global pose per frame from the feature manager's own
correspondences (essential matrix + recoverPose), so the triangulation/TTC
pipeline can run on estimated geometry instead of GT poses. It is deliberately
NOT a VO/SLAM system:

  * only frame-to-frame (gap-1) essential matrices are ever fit -- the pairs with
    abundant, reliable correspondences -- never wide, few-match pairs;
  * per-step translation *magnitude* is taken from GT, so chaining the steps is
    metric and there is no scale drift (the one thing monocular cannot recover);
  * rotation and epipole *direction* stay estimated -- that is exactly the error
    whose effect on depth we want to measure.

Wide-baseline relative poses used by the selector come for free by chaining these
reliable steps: ``relative_pose(est_poses, i, j)`` composes them. Local relative
poses are drift-immune (common accumulated drift cancels in ``inv(T_j) T_i``).

The tracker fills ``self.poses`` causally (entry ``k`` set before frame ``k`` is
triangulated). It is initialised from GT so any not-yet-estimated future entry is
harmless (never read before it is overwritten).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparse_depth.geometry import (
    relative_pose,
    rotation_error_deg,
    translation_direction_error_deg,
)


@dataclass
class StepEstimate:
    """Per-frame diagnostics for the (frame-1 -> frame) pose estimate."""
    frame: int
    fallback: bool
    reason: str
    pairs: int = 0
    inliers: int = 0
    rot_err_deg: float = float("nan")
    t_err_deg: float = float("nan")


def _gather_correspondences(tracks: Iterable[Any], i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    pts_i: List[np.ndarray] = []
    pts_j: List[np.ndarray] = []
    for tr in tracks:
        if tr.hit_count < 2:
            continue
        oi = tr.observation_at(i)
        oj = tr.observation_at(j)
        if oi is None or oj is None:
            continue
        pts_i.append(np.asarray(oi.pt, dtype=np.float32).reshape(2))
        pts_j.append(np.asarray(oj.pt, dtype=np.float32).reshape(2))
    if not pts_i:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    return np.asarray(pts_i, np.float32), np.asarray(pts_j, np.float32)


def estimate_from_matches(
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    i: int,
    j: int,
    K: np.ndarray,
    poses_gt: Sequence[np.ndarray],
    cfg: Any,
) -> Tuple[np.ndarray, np.ndarray, StepEstimate]:
    """Estimate cam_i -> cam_j relative pose (``X_j = R X_i + t``) from raw matches.

    Rotation and translation direction come from ``findEssentialMat`` +
    ``recoverPose`` on the given correspondences (the RANSAC inlier mask is used
    only internally to fit E robustly -- it is NOT used to filter tracks). The
    translation *magnitude* is taken from GT so the step is metric. On any
    failure the GT relative pose is returned (``fallback=True``), so a bad frame
    degrades one step rather than breaking the chain.
    """
    R_gt, t_gt = relative_pose(poses_gt, i, j)
    scale = float(np.linalg.norm(t_gt))
    pts_i = np.asarray(pts_i, dtype=np.float32).reshape(-1, 2)
    pts_j = np.asarray(pts_j, dtype=np.float32).reshape(-1, 2)

    def _fallback(reason: str, pairs: int = 0) -> Tuple[np.ndarray, np.ndarray, StepEstimate]:
        return R_gt, t_gt, StepEstimate(j, True, reason, pairs=pairs)

    if scale < 1e-6:
        return _fallback("stationary", pts_i.shape[0])
    if pts_i.shape[0] < cfg.pose_min_pairs:
        return _fallback("too_few_pairs", pts_i.shape[0])

    try:
        cv2.setRNGSeed(12345)
        E, mask = cv2.findEssentialMat(
            pts_i, pts_j, K, method=cv2.RANSAC,
            prob=cfg.pose_ransac_prob,
            threshold=cfg.pose_ransac_thresh_px,
            maxIters=cfg.pose_ransac_max_iters,
        )
    except cv2.error:
        return _fallback("E_error", pts_i.shape[0])
    if E is None or mask is None:
        return _fallback("E_failed", pts_i.shape[0])
    if E.shape[0] > 3:  # findEssentialMat can stack multiple candidate solutions
        E = E[:3, :3]

    try:
        _, R_est, t_est, _ = cv2.recoverPose(E, pts_i, pts_j, K, mask=mask.copy())
    except cv2.error:
        return _fallback("recoverPose_error", pts_i.shape[0])

    t_dir = np.asarray(t_est, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(t_dir))
    if nrm < 1e-9 or not np.all(np.isfinite(R_est)):
        return _fallback("degenerate", pts_i.shape[0])
    t_dir = t_dir / nrm

    est = StepEstimate(
        frame=j, fallback=False, reason="ok",
        pairs=int(pts_i.shape[0]), inliers=int(np.sum(mask)),
        rot_err_deg=rotation_error_deg(R_est, R_gt),
        t_err_deg=translation_direction_error_deg(t_dir, t_gt),
    )
    return np.asarray(R_est, dtype=np.float64), t_dir * scale, est


def estimate_step(
    tracks: Iterable[Any],
    i: int,
    j: int,
    K: np.ndarray,
    poses_gt: Sequence[np.ndarray],
    cfg: Any,
) -> Tuple[np.ndarray, np.ndarray, StepEstimate]:
    """Tracks-based wrapper for :func:`estimate_from_matches`.

    Gathers the (i, j) correspondences from tracks observed at both frames, then
    estimates the step. Used for the LK-off / triangulation-time path where the
    current-frame observations already live on the tracks.
    """
    pts_i, pts_j = _gather_correspondences(tracks, i, j)
    return estimate_from_matches(pts_i, pts_j, i, j, K, poses_gt, cfg)


def summarize_steps(diag: Sequence[StepEstimate]) -> dict:
    """Headline pose-estimation diagnostics over a run's per-step estimates."""
    if not diag:
        return {"pose_steps": 0}
    fb = np.array([d.fallback for d in diag], dtype=bool)
    rot = np.array([d.rot_err_deg for d in diag], dtype=np.float64)
    td = np.array([d.t_err_deg for d in diag], dtype=np.float64)
    rot = rot[np.isfinite(rot)]
    td = td[np.isfinite(td)]
    return {
        "pose_steps": len(diag),
        "pose_fallback_rate": float(np.mean(fb)),
        "pose_median_rot_err_deg": float(np.median(rot)) if rot.size else float("nan"),
        "pose_median_t_err_deg": float(np.median(td)) if td.size else float("nan"),
    }
