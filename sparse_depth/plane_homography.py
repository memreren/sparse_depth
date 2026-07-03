"""2-D plane-homography RANSAC over feature-manager track correspondences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from sparse_depth.ground_plane import Plane, road_trapezoid_mask
from sparse_depth.track_types import Track


@dataclass
class HomographyResult:
    source_frame: int = -1
    candidate_ids: set[int] = field(default_factory=set)
    inlier_ids: set[int] = field(default_factory=set)
    H_target_to_source: Optional[np.ndarray] = None
    reproj_error_by_id: Dict[int, float] = field(default_factory=dict)
    median_reproj_px: float = np.nan
    reason: str = "not_run"


@dataclass
class GroundScaleResult:
    R_target_to_source: Optional[np.ndarray] = None
    t_direction_target_to_source: Optional[np.ndarray] = None
    scale_m: float = np.nan
    candidate_ids: set[int] = field(default_factory=set)
    inlier_ids: set[int] = field(default_factory=set)
    median_reproj_px: float = np.nan
    pose_pairs: int = 0
    pose_inliers: int = 0
    reason: str = "not_run"


def estimate_lower_image_homography(
    tracks: Iterable[Track], current_frame: int, image_shape: Tuple[int, int], cfg,
) -> HomographyResult:
    """Fit current-to-past homography from eligible lower-image track pairs.

    This deliberately estimates an unconstrained 2-D homography first.  It is
    a diagnostic of the dominant eligible plane, not yet a claim that the
    winning plane is ground.
    """
    gap = int(cfg.homography_gap)
    source_frame = current_frame - gap
    result = HomographyResult(source_frame=source_frame)
    if source_frame < 0:
        result.reason = "source_before_zero"
        return result
    mask = road_trapezoid_mask(
        image_shape, cfg.homography_roi_top_y_frac, cfg.homography_roi_bottom_left_frac,
        cfg.homography_roi_bottom_right_frac, cfg.homography_roi_top_left_frac,
        cfg.homography_roi_top_right_frac,
    )
    ids, target_pts, source_pts = [], [], []
    h, w = image_shape[:2]
    for tr in tracks:
        if tr.dead or not tr.active_in(current_frame):
            continue
        if cfg.homography_confirmed_only and not tr.confirmed:
            continue
        if (not cfg.homography_include_reacquired) and tr.reacq_probation:
            continue
        previous = tr.observation_at(source_frame)
        if previous is None:
            continue
        point = tr.last_pt()
        x, y = np.rint(point).astype(int)
        if not (0 <= x < w and 0 <= y < h and mask[y, x]):
            continue
        ids.append(tr.id)
        target_pts.append(point)
        source_pts.append(previous.pt)
    result.candidate_ids = set(ids)
    if len(ids) < 4:
        result.reason = f"only_{len(ids)}_candidates"
        return result
    target = np.asarray(target_pts, dtype=np.float32).reshape(-1, 1, 2)
    source = np.asarray(source_pts, dtype=np.float32).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(
        target, source, method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg.homography_ransac_thresh_px),
        maxIters=int(cfg.homography_ransac_max_iters), confidence=float(cfg.homography_ransac_confidence),
    )
    if H is None or inlier_mask is None:
        result.reason = "ransac_failed"
        return result
    projected = cv2.perspectiveTransform(target, H).reshape(-1, 2)
    errors = np.linalg.norm(projected - source.reshape(-1, 2), axis=1)
    inlier_mask = inlier_mask.reshape(-1).astype(bool)
    result.H_target_to_source = H
    result.inlier_ids = {track_id for track_id, good in zip(ids, inlier_mask) if good}
    result.reproj_error_by_id = {track_id: float(error) for track_id, error in zip(ids, errors)}
    result.median_reproj_px = float(np.median(errors[inlier_mask])) if np.any(inlier_mask) else np.nan
    result.reason = "ok"
    return result


def plane_from_homography_known_pose(H: np.ndarray, K: np.ndarray, R_ts: np.ndarray, t_ts: np.ndarray):
    """Recover target-frame plane from H when metric relative pose is known.

    Solves ``R - alpha*K^-1 H K = t q.T`` jointly for the arbitrary
    homography scale ``alpha`` and ``q=n/d``.  Returns ``(Plane, rmse)`` or
    ``(None, nan)`` for degenerate translation/homography cases.
    """
    t = np.asarray(t_ts, dtype=np.float64).reshape(3)
    if np.linalg.norm(t) < 1e-9:
        return None, np.nan
    B = np.linalg.inv(K) @ np.asarray(H, dtype=np.float64).reshape(3, 3) @ K
    R = np.asarray(R_ts, dtype=np.float64).reshape(3, 3)
    columns = [B.reshape(-1)]
    columns.extend(np.outer(t, np.eye(3)[j]).reshape(-1) for j in range(3))
    A = np.column_stack(columns)
    # vec(R) = alpha vec(B) + vec(t q.T)
    solution, *_ = np.linalg.lstsq(A, R.reshape(-1), rcond=None)
    alpha, q = float(solution[0]), solution[1:]
    q_norm = float(np.linalg.norm(q))
    if not np.isfinite(q_norm) or q_norm < 1e-9 or not np.isfinite(alpha):
        return None, np.nan
    reconstructed = alpha * B + np.outer(t, q)
    rmse = float(np.sqrt(np.mean((R - reconstructed) ** 2)))
    # Choose the equivalent representation whose plane offset is negative;
    # for a road below the camera this normally yields a down-facing normal.
    plane = Plane(-q / q_norm, -1.0 / q_norm)
    return plane, rmse


def estimate_ground_scale_ransac(tracks: Iterable[Track], current_frame: int, image_shape: Tuple[int, int], K: np.ndarray, cfg, plane: Plane) -> GroundScaleResult:
    """Estimate Essential pose then metric translation scale on lower-road pairs.

    Essential RANSAC uses broadly distributed active track pairs.  The second
    stage enumerates one-correspondence scale hypotheses and scores their
    known-plane homographies, which is exhaustive 1-D RANSAC for this small
    candidate pool.
    """
    gap, source_frame = int(cfg.homography_gap), current_frame - int(cfg.homography_gap)
    result = GroundScaleResult()
    if source_frame < 0: result.reason = "source_before_zero"; return result
    all_ids, all_src, all_tgt = [], [], []
    road_ids, road_src, road_tgt = [], [], []
    mask = road_trapezoid_mask(image_shape, cfg.homography_roi_top_y_frac, cfg.homography_roi_bottom_left_frac, cfg.homography_roi_bottom_right_frac, cfg.homography_roi_top_left_frac, cfg.homography_roi_top_right_frac)
    h, w = image_shape[:2]
    for tr in tracks:
        if tr.dead or not tr.active_in(current_frame) or tr.observation_at(source_frame) is None: continue
        previous, current = tr.observation_at(source_frame), tr.last_pt()
        all_ids.append(tr.id); all_src.append(previous.pt); all_tgt.append(current)
        x, y = np.rint(current).astype(int)
        if tr.confirmed and not tr.reacq_probation and 0 <= x < w and 0 <= y < h and mask[y, x]:
            road_ids.append(tr.id); road_src.append(previous.pt); road_tgt.append(current)
    result.pose_pairs = len(all_ids)
    result.candidate_ids = set(road_ids)
    if len(all_ids) < max(5, int(cfg.pose_min_pairs)) or len(road_ids) < 4:
        result.reason = "too_few_pairs"; return result
    src, tgt = np.asarray(all_src, np.float32), np.asarray(all_tgt, np.float32)
    try:
        cv2.setRNGSeed(12345)
        E, pose_mask = cv2.findEssentialMat(src, tgt, K, method=cv2.RANSAC, prob=float(cfg.pose_ransac_prob), threshold=float(cfg.pose_ransac_thresh_px), maxIters=int(cfg.pose_ransac_max_iters))
        if E is None or pose_mask is None: result.reason = "essential_failed"; return result
        if E.shape[0] > 3: E = E[:3, :3]
        _, R_s_to_t, t_s_to_t, pose_mask = cv2.recoverPose(E, src, tgt, K, mask=pose_mask.astype(np.uint8))
    except cv2.error:
        result.reason = "pose_recovery_failed"; return result
    # Invert source->target recoverPose output to target->source convention.
    R = R_s_to_t.T; tdir = -R_s_to_t.T @ t_s_to_t.reshape(3); tdir /= np.linalg.norm(tdir)
    result.R_target_to_source, result.t_direction_target_to_source = R, tdir
    result.pose_inliers = int(np.sum(pose_mask.ravel() > 0))
    target, source = np.asarray(road_tgt, float), np.asarray(road_src, float)
    Kinv = np.linalg.inv(K); n_over_d = plane.normal / plane.offset
    xh = np.hstack([target, np.ones((len(target), 1))]); yh = np.hstack([source, np.ones((len(source), 1))])
    A = (K @ R @ Kinv @ xh.T).T
    B = -(K @ tdir.reshape(3, 1) @ n_over_d.reshape(1, 3) @ Kinv @ xh.T).T
    c, e = np.cross(yh, A), np.cross(yh, B)
    denom = np.sum(e * e, axis=1); scales = -np.sum(c * e, axis=1) / np.maximum(denom, 1e-12)
    scales = scales[np.isfinite(scales) & (scales > 1e-4) & (scales < 20.0)]
    if scales.size == 0: result.reason = "no_scale_hypotheses"; return result
    best_scale, best_inliers, best_errors = np.nan, None, None
    for scale in scales[:int(cfg.ground_scale_max_hypotheses)]:
        H = K @ (R - scale * np.outer(tdir, n_over_d)) @ Kinv
        projected = cv2.perspectiveTransform(target.astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(projected - source, axis=1); inliers = errors <= float(cfg.ground_scale_ransac_thresh_px)
        if best_inliers is None or np.sum(inliers) > np.sum(best_inliers) or (np.sum(inliers) == np.sum(best_inliers) and np.median(errors[inliers]) < np.median(best_errors[best_inliers])):
            best_scale, best_inliers, best_errors = scale, inliers, errors
    if best_inliers is None or np.sum(best_inliers) < 4: result.reason = "scale_ransac_failed"; return result
    # Least-squares refinement of the scalar using all consensus correspondences.
    ei, ci = e[best_inliers], c[best_inliers]
    scale = -float(np.sum(ei * ci) / max(np.sum(ei * ei), 1e-12))
    H = K @ (R - scale * np.outer(tdir, n_over_d)) @ Kinv
    projected = cv2.perspectiveTransform(target.astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - source, axis=1); inliers = errors <= float(cfg.ground_scale_ransac_thresh_px)
    result.scale_m = scale; result.inlier_ids = {track_id for track_id, ok in zip(road_ids, inliers) if ok}; result.median_reproj_px = float(np.median(errors[inliers])) if np.any(inliers) else np.nan; result.reason = "ok"
    return result
