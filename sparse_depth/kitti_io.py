"""KITTI file IO and LiDAR projection helpers.

These utilities are shared by the interactive managers and the headless
evaluators.  They are intentionally small and explicit: each function maps one
KITTI file convention to arrays used by the feature/depth pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def read_kitti_calib_file(path: Path):
    """Parse numeric entries from a KITTI calibration text file."""
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            vals = value.strip().split()
            if vals:
                try:
                    data[key] = np.array([float(v) for v in vals], dtype=np.float64)
                except ValueError:
                    # Raw KITTI files also contain metadata such as dates.
                    continue
    return data


def load_kitti_K(calib_path: Path) -> np.ndarray:
    """Load camera-0 intrinsic matrix from an odometry-style calib file."""
    data = read_kitti_calib_file(calib_path)
    if "P0" not in data:
        raise RuntimeError(f"Could not find P0 in calibration file: {calib_path}")
    return np.asarray(data["P0"], dtype=np.float64).reshape(3, 4)[:, :3]


def load_kitti_poses(poses_path: Path) -> List[np.ndarray]:
    """Load KITTI odometry poses as 4x4 transforms."""
    poses: List[np.ndarray] = []
    with open(poses_path, "r") as f:
        for line in f:
            vals = np.array(line.strip().split(), dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = vals.reshape(3, 4)
            poses.append(T)
    return poses


def load_gray(img_dir: Path, frame: int, digits: int) -> Optional[np.ndarray]:
    """Read one KITTI grayscale PNG by frame number."""
    path = img_dir / f"{frame:0{digits}d}.png"
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def make_4x4_from_R_T(R_flat, T_flat) -> np.ndarray:
    """Build a homogeneous transform from flattened KITTI R/T entries."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R_flat, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(T_flat, dtype=np.float64).reshape(3)
    return T


def load_raw_kitti_velo_to_cam(calib_velo_to_cam_path: Path) -> np.ndarray:
    """Load raw KITTI Velodyne-to-camera-0 transform."""
    data = read_kitti_calib_file(calib_velo_to_cam_path)
    if "R" not in data or "T" not in data:
        raise KeyError(f"Expected R and T in {calib_velo_to_cam_path}")
    return make_4x4_from_R_T(data["R"], data["T"])


def load_raw_kitti_cam_calib(calib_cam_to_cam_path: Path, camera: str = "00") -> Tuple[np.ndarray, np.ndarray]:
    """Load raw KITTI rectified projection and rectification transform."""
    data = read_kitti_calib_file(calib_cam_to_cam_path)
    p_key_candidates = [f"P_rect_{camera}", f"P{int(camera)}", f"P{camera}"]
    r_key_candidates = [f"R_rect_{camera}", "R_rect_00", "R_rect"]

    P_rect = None
    for key in p_key_candidates:
        if key in data:
            P_rect = np.asarray(data[key], dtype=np.float64).reshape(3, 4)
            break
    if P_rect is None:
        raise KeyError(f"Could not find projection matrix for camera {camera}")

    R_rect_3 = None
    for key in r_key_candidates:
        if key in data:
            R_rect_3 = np.asarray(data[key], dtype=np.float64).reshape(3, 3)
            break
    if R_rect_3 is None:
        R_rect_3 = np.eye(3, dtype=np.float64)

    R_rect_4 = np.eye(4, dtype=np.float64)
    R_rect_4[:3, :3] = R_rect_3
    return P_rect, R_rect_4


def load_odometry_lidar_projection(calib_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load LiDAR projection from an odometry calib file when Tr is present."""
    data = read_kitti_calib_file(calib_path)
    if "P0" not in data:
        raise KeyError(f"Could not find P0 in {calib_path}")
    P_rect = np.asarray(data["P0"], dtype=np.float64).reshape(3, 4)
    R_rect_4 = np.eye(4, dtype=np.float64)

    tr_key = None
    for key in ("Tr", "Tr_velo_to_cam", "Tr_velo_cam"):
        if key in data:
            tr_key = key
            break
    if tr_key is None:
        raise KeyError(
            f"{calib_path} has no Tr/Tr_velo_to_cam. Provide raw calib files "
            "with --calib-velo-to-cam and --calib-cam-to-cam, or skip LiDAR."
        )

    vals = np.asarray(data[tr_key], dtype=np.float64)
    T_cam0_velo = np.eye(4, dtype=np.float64)
    T_cam0_velo[:3, :] = vals.reshape(3, 4)
    return T_cam0_velo, R_rect_4, P_rect


def load_velodyne_bin(path: Path) -> np.ndarray:
    """Load one KITTI Velodyne .bin as N x 4 float64 points."""
    pts = np.fromfile(str(path), dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"Velodyne file has invalid size: {path}")
    return pts.reshape(-1, 4).astype(np.float64)


def project_velodyne_to_image(
    velo_points: np.ndarray,
    T_cam0_velo: np.ndarray,
    R_rect_4: np.ndarray,
    P_rect: np.ndarray,
    image_shape,
    min_depth_m: float = 1.0,
    max_depth_m: float = 120.0,
):
    """Project Velodyne points into the rectified image."""
    h, w = image_shape[:2]
    xyz = velo_points[:, :3]
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)]).T
    rect_h = R_rect_4 @ (T_cam0_velo @ xyz_h)
    rect_xyz = rect_h[:3, :].T
    z_rect = rect_xyz[:, 2]
    proj = P_rect @ rect_h
    u = proj[0, :] / (proj[2, :] + 1e-12)
    v = proj[1, :] / (proj[2, :] + 1e-12)
    valid = (
        np.isfinite(u) & np.isfinite(v) & np.isfinite(z_rect)
        & (z_rect > min_depth_m) & (z_rect < max_depth_m)
        & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    )
    return np.vstack([u[valid], v[valid]]).T, z_rect[valid], rect_xyz[valid]


def match_sparse_to_lidar_with_radius(
    sparse_uv: np.ndarray,
    lidar_uv: np.ndarray,
    lidar_z: np.ndarray,
    radius_px: float,
    mode: str = "nearest",
):
    """Match each sparse depth point to nearby projected LiDAR depth."""
    sparse_uv = np.asarray(sparse_uv, dtype=np.float64)
    lidar_uv = np.asarray(lidar_uv, dtype=np.float64)
    lidar_z = np.asarray(lidar_z, dtype=np.float64)
    n = sparse_uv.shape[0]
    matched = np.zeros(n, dtype=bool)
    matched_z = np.full(n, np.nan, dtype=np.float64)
    matched_dist = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or lidar_uv.shape[0] == 0:
        return matched, matched_z, matched_dist

    try:
        from scipy.spatial import cKDTree  # type: ignore

        neighbors = cKDTree(lidar_uv).query_ball_point(sparse_uv, r=radius_px)
        for i, inds in enumerate(neighbors):
            if not inds:
                continue
            inds = np.asarray(inds, dtype=int)
            d = np.linalg.norm(lidar_uv[inds] - sparse_uv[i], axis=1)
            if mode == "nearest":
                k = int(np.argmin(d))
                z = lidar_z[inds[k]]
                dist = d[k]
            elif mode == "min_depth":
                k = int(np.argmin(lidar_z[inds]))
                z = lidar_z[inds[k]]
                dist = d[k]
            elif mode == "median_depth":
                z = float(np.median(lidar_z[inds]))
                dist = float(np.min(d))
            else:
                raise ValueError(f"Unknown lidar match mode: {mode}")
            matched[i] = True
            matched_z[i] = z
            matched_dist[i] = dist
        return matched, matched_z, matched_dist
    except Exception:
        pass

    cell = max(float(radius_px), 1.0)
    grid = {}
    ij = np.floor(lidar_uv / cell).astype(int)
    for idx, key in enumerate(map(tuple, ij)):
        grid.setdefault(key, []).append(idx)
    r2 = radius_px * radius_px
    for i, p in enumerate(sparse_uv):
        ci, cj = np.floor(p / cell).astype(int)
        cand = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                cand.extend(grid.get((ci + di, cj + dj), []))
        if not cand:
            continue
        cand = np.asarray(cand, dtype=int)
        diff = lidar_uv[cand] - p
        d2 = np.sum(diff * diff, axis=1)
        keep = d2 <= r2
        if not np.any(keep):
            continue
        cand = cand[keep]
        d = np.sqrt(d2[keep])
        if mode == "nearest":
            k = int(np.argmin(d))
            z = lidar_z[cand[k]]
            dist = d[k]
        elif mode == "min_depth":
            k = int(np.argmin(lidar_z[cand]))
            z = lidar_z[cand[k]]
            dist = d[k]
        elif mode == "median_depth":
            z = float(np.median(lidar_z[cand]))
            dist = float(np.min(d))
        else:
            raise ValueError(f"Unknown lidar match mode: {mode}")
        matched[i] = True
        matched_z[i] = z
        matched_dist[i] = dist
    return matched, matched_z, matched_dist
