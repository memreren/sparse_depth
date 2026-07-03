"""Relative-pose RANSAC diagnostics for sparse feature tracks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence

import cv2
import numpy as np

from sparse_depth.geometry import (
    relative_pose,
    rotation_error_deg,
    translation_direction_error_deg,
)


@dataclass
class PoseEval:
    ok: bool = False
    reason: str = "not computed"
    gap: int = 1
    pairs: int = 0
    inliers: int = 0
    inlier_mask_by_track: Dict[int, bool] = field(default_factory=dict)
    rot_err_deg: float = np.nan
    t_err_deg: float = np.nan


def evaluate_pose_gap_from_tracks(
    tracks: Iterable[Any],
    frame: int,
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
) -> PoseEval:
    """Estimate relative pose for a fixed frame gap and compare it to GT pose.

    The input tracks only need the attributes/methods used by the current
    managers: ``id``, ``hit_count``, and ``observation_at(frame)``. This keeps
    pose diagnostics independent from whether the track came from SIFT-only,
    LK-first, or a future tracker.
    """
    gap = cfg.pose_eval_gap
    i = frame - gap
    j = frame
    if i < 0 or j >= len(poses):
        return PoseEval(False, "invalid gap", gap)

    pts_i = []
    pts_j = []
    track_ids = []
    for tr in tracks:
        if tr.hit_count < 2:
            continue
        oi = tr.observation_at(i)
        oj = tr.observation_at(j)
        if oi is None or oj is None:
            continue
        pts_i.append(oi.pt)
        pts_j.append(oj.pt)
        track_ids.append(tr.id)

    if len(pts_i) < cfg.pose_min_pairs:
        return PoseEval(False, "too few pairs", gap, pairs=len(pts_i))

    pts_i_arr = np.asarray(pts_i, dtype=np.float32)
    pts_j_arr = np.asarray(pts_j, dtype=np.float32)
    try:
        cv2.setRNGSeed(12345)
        E, mask = cv2.findEssentialMat(
            pts_i_arr,
            pts_j_arr,
            K,
            method=cv2.RANSAC,
            prob=cfg.pose_ransac_prob,
            threshold=cfg.pose_ransac_thresh_px,
            maxIters=cfg.pose_ransac_max_iters,
        )
    except cv2.error as e:
        return PoseEval(False, f"findEssentialMat error: {e}", gap, pairs=len(pts_i))

    if E is None or mask is None:
        return PoseEval(False, "E failed", gap, pairs=len(pts_i))
    if E.shape[0] > 3:
        E = E[:3, :3]

    mask_bool = mask.ravel().astype(bool)
    inlier_mask_by_track = {tid: bool(mask_bool[k]) for k, tid in enumerate(track_ids)}
    inliers = int(np.sum(mask_bool))
    rot_err = np.nan
    t_err = np.nan
    if inliers >= 5:
        try:
            _, R_est, t_est, _ = cv2.recoverPose(E, pts_i_arr, pts_j_arr, K, mask=mask.astype(np.uint8))
            R_gt, t_gt = relative_pose(poses, i, j)
            rot_err = rotation_error_deg(R_est, R_gt)
            t_err = translation_direction_error_deg(t_est.reshape(3), t_gt.reshape(3))
        except cv2.error:
            pass

    return PoseEval(
        True,
        "ok",
        gap,
        pairs=len(pts_i),
        inliers=inliers,
        inlier_mask_by_track=inlier_mask_by_track,
        rot_err_deg=rot_err,
        t_err_deg=t_err,
    )
