"""Pure camera-geometry helpers for sparse temporal depth.

This module deliberately contains no feature-manager state, no KITTI file IO,
and no visualization. Keeping these functions isolated makes it easier to add
new triangulation methods later while preserving the current best-pair baseline.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


def relative_pose(poses: Sequence[np.ndarray], i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return R,t satisfying X_j = R X_i + t for KITTI odometry poses T_0i.

    KITTI odometry pose files store each camera pose in a common frame. For an
    arbitrary pair (i, j), we compute the direct pairwise transform from those
    two global poses instead of incrementally chaining frame-to-frame transforms.
    """
    T_0_i = poses[i]
    T_0_j = poses[j]
    T_j_i = np.linalg.inv(T_0_j) @ T_0_i
    return T_j_i[:3, :3], T_j_i[:3, 3]


def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def fundamental_from_R_t(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build the fundamental matrix for two calibrated views."""
    E = skew(t) @ R
    K_inv = np.linalg.inv(K)
    return K_inv.T @ E @ K_inv


def sampson_error_px(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Sampson approximation to epipolar error, in pixel units."""
    n = pts1.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    x1 = np.hstack([pts1, np.ones((n, 1))]).astype(np.float64)
    x2 = np.hstack([pts2, np.ones((n, 1))]).astype(np.float64)
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    x2tFx1 = np.sum(x2 * Fx1, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(np.maximum((x2tFx1 ** 2) / (denom + 1e-12), 0.0))


def parallax_angle_deg(K: np.ndarray, R: np.ndarray, pts_i: np.ndarray, pts_j: np.ndarray) -> np.ndarray:
    """Ray angle in camera-i coordinates between ray_i and R^T ray_j."""
    n = pts_i.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    K_inv = np.linalg.inv(K)
    x_i = np.hstack([pts_i, np.ones((n, 1))]).T.astype(np.float64)
    x_j = np.hstack([pts_j, np.ones((n, 1))]).T.astype(np.float64)
    r_i = K_inv @ x_i
    r_j_camj = K_inv @ x_j
    r_j_cami = R.T @ r_j_camj
    r_i = r_i / (np.linalg.norm(r_i, axis=0, keepdims=True) + 1e-12)
    r_j_cami = r_j_cami / (np.linalg.norm(r_j_cami, axis=0, keepdims=True) + 1e-12)
    dots = np.clip(np.sum(r_i * r_j_cami, axis=0), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R_delta = R_est @ R_gt.T
    val = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def translation_direction_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    a = np.asarray(t_est, dtype=np.float64).reshape(3)
    b = np.asarray(t_gt, dtype=np.float64).reshape(3)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return np.nan
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot1 = np.clip(float(np.dot(a, b)), -1.0, 1.0)
    dot2 = np.clip(float(np.dot(-a, b)), -1.0, 1.0)
    # Essential decomposition sign may flip; use smaller direction error.
    return float(min(np.degrees(np.arccos(dot1)), np.degrees(np.arccos(dot2))))


def project_points(K: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, 3)
    x = (K @ X.T).T
    return x[:, :2] / (x[:, 2:3] + 1e-12)


def triangulate_dlt(K: np.ndarray, R: np.ndarray, t: np.ndarray, pt_i: np.ndarray, pt_j: np.ndarray):
    """Triangulate one correspondence. Returns X_i, X_j, depth_i, depth_j, reproj_rmse."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    p1 = np.asarray(pt_i, dtype=np.float64).reshape(2)
    p2 = np.asarray(pt_j, dtype=np.float64).reshape(2)
    X_h = cv2.triangulatePoints(P1, P2, p1.reshape(2, 1), p2.reshape(2, 1)).reshape(4)
    if abs(X_h[3]) < 1e-12:
        return None
    X_i = X_h[:3] / X_h[3]
    X_j = R @ X_i + t.reshape(3)
    depth_i = float(X_i[2])
    depth_j = float(X_j[2])
    p1_hat = project_points(K, X_i)[0]
    p2_hat = project_points(K, X_j)[0]
    reproj = float(np.sqrt(0.5 * (np.sum((p1_hat - p1) ** 2) + np.sum((p2_hat - p2) ** 2))))
    return X_i, X_j, depth_i, depth_j, reproj
