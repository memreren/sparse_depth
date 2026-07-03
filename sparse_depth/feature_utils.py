"""Feature detection and small track-association utilities."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def make_sift(nfeatures: int, n_octave_layers: int, contrast: float, edge: float, sigma: float):
    """Create an OpenCV SIFT detector with explicit numeric arguments."""
    return cv2.SIFT_create(
        nfeatures=int(nfeatures),
        nOctaveLayers=int(n_octave_layers),
        contrastThreshold=float(contrast),
        edgeThreshold=float(edge),
        sigma=float(sigma),
    )


def detect_sift(img: np.ndarray, sift, mask: np.ndarray | None = None) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detect SIFT keypoints/descriptors, always returning a float32 descriptor array."""
    kps, desc = sift.detectAndCompute(img, mask)
    if desc is None:
        desc = np.empty((0, 128), dtype=np.float32)
        kps = []
    return kps, desc.astype(np.float32)


def detect_shi_sift(
    img: np.ndarray, sift, max_corners: int, quality_level: float,
    min_distance_px: float, block_size: int, mask: np.ndarray | None = None,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detect Shi--Tomasi corners, then compute SIFT descriptors at them."""
    corners = cv2.goodFeaturesToTrack(
        img, maxCorners=int(max_corners), qualityLevel=float(quality_level),
        minDistance=float(min_distance_px), mask=mask, blockSize=int(block_size),
        useHarrisDetector=False,
    )
    if corners is None or len(corners) == 0:
        return [], np.empty((0, 128), dtype=np.float32)
    size = max(float(block_size), 3.0)
    keypoints = [cv2.KeyPoint(float(x), float(y), size) for x, y in corners.reshape(-1, 2)]
    keypoints, desc = sift.compute(img, keypoints)
    if desc is None or not keypoints:
        return [], np.empty((0, 128), dtype=np.float32)
    return keypoints, desc.astype(np.float32)


def detect_xfeat_sift(
    img: np.ndarray, sift, xfeat, top_k: int, desc_size_px: float,
    mask: np.ndarray | None = None,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detect XFeat keypoints, then compute SIFT descriptors at those locations.

    XFeat's own 64-d descriptors are intentionally discarded: the association,
    soft-rescue, and quality thresholds throughout the manager are calibrated to
    128-d SIFT L2 distances, so we keep the descriptor stage identical to
    ``detect_shi_sift`` and only swap the detector. ``mask`` filters keypoints to
    the allowed image region (XFeat itself has no mask input).
    """
    pts = xfeat.detect_points(img)
    if pts.shape[0] == 0:
        return [], np.empty((0, 128), dtype=np.float32)
    if pts.shape[0] > int(top_k):
        pts = pts[: int(top_k)]
    if mask is not None:
        h, w = mask.shape[:2]
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        keep = mask[yi, xi] > 0
        pts = pts[keep]
        if pts.shape[0] == 0:
            return [], np.empty((0, 128), dtype=np.float32)
    size = max(float(desc_size_px), 3.0)
    keypoints = [cv2.KeyPoint(float(x), float(y), size) for x, y in pts]
    keypoints, desc = sift.compute(img, keypoints)
    if desc is None or not keypoints:
        return [], np.empty((0, 128), dtype=np.float32)
    return keypoints, desc.astype(np.float32)


def detect_xfeat_native(
    img: np.ndarray, xfeat, top_k: int, desc_size_px: float, desc_scale: float,
    mask: np.ndarray | None = None,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detect XFeat keypoints AND keep XFeat's own learned descriptors.

    Unlike ``detect_xfeat_sift`` (which discards XFeat descriptors and recomputes
    SIFT), this returns XFeat's 64-d L2-normalized descriptors, so matching runs
    on the learned features end to end. XFeat descriptors are unit-norm, giving
    L2 distances in [0, 2]; every association/quality gate in the manager is
    calibrated to SIFT L2 magnitudes (~150-280). We therefore scale the unit
    descriptors by ``desc_scale`` so those existing gates apply unchanged. The
    Lowe ratio test is scale-invariant and remains the primary gate; the scale
    only affects the absolute-distance ceilings.
    """
    pts, scores, desc = xfeat.detect_points_desc(img)
    if pts.shape[0] == 0:
        return [], np.empty((0, 64), dtype=np.float32)
    if pts.shape[0] > int(top_k):
        pts, scores, desc = pts[: int(top_k)], scores[: int(top_k)], desc[: int(top_k)]
    if mask is not None:
        h, w = mask.shape[:2]
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        keep = mask[yi, xi] > 0
        pts, scores, desc = pts[keep], scores[keep], desc[keep]
        if pts.shape[0] == 0:
            return [], np.empty((0, 64), dtype=np.float32)
    size = max(float(desc_size_px), 3.0)
    keypoints = [
        cv2.KeyPoint(float(x), float(y), size, -1.0, float(s))
        for (x, y), s in zip(pts, scores)
    ]
    desc = (desc * float(desc_scale)).astype(np.float32)
    return keypoints, desc


def keypoint_points(kps: Sequence[cv2.KeyPoint]) -> np.ndarray:
    """Convert OpenCV keypoints to an N x 2 float32 point array."""
    if not kps:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([kp.pt for kp in kps], dtype=np.float32)


def bucket_index(pt: np.ndarray, w: int, h: int, cols: int, rows: int) -> Tuple[int, int, int]:
    """Return row, col, flat-index for an image-space point in a regular grid."""
    x = int(np.clip(pt[0], 0, w - 1))
    y = int(np.clip(pt[1], 0, h - 1))
    c = min(cols - 1, max(0, int(x * cols / w)))
    r = min(rows - 1, max(0, int(y * rows / h)))
    return r, c, r * cols + c


def nearest_dist_to_points(pt: np.ndarray, pts: np.ndarray) -> float:
    """Distance from one point to the nearest point in an array."""
    if pts.size == 0:
        return float("inf")
    d = np.linalg.norm(pts - pt.reshape(1, 2), axis=1)
    return float(np.min(d))


class SpatialHash:
    """Tiny grid hash for approximate duplicate/spacing checks."""

    def __init__(self, radius: float):
        self.radius = max(float(radius), 1e-6)
        self.radius2 = self.radius * self.radius
        self.cells: Dict[Tuple[int, int], List[np.ndarray]] = {}

    def _cell(self, pt: np.ndarray) -> Tuple[int, int]:
        return int(np.floor(float(pt[0]) / self.radius)), int(np.floor(float(pt[1]) / self.radius))

    def add(self, pt: np.ndarray):
        key = self._cell(pt)
        self.cells.setdefault(key, []).append(np.asarray(pt, dtype=np.float32).copy())

    def add_many(self, pts: np.ndarray):
        if pts.size == 0:
            return
        for pt in pts:
            self.add(pt)

    def too_close(self, pt: np.ndarray) -> bool:
        if not self.cells:
            return False
        cx, cy = self._cell(pt)
        p = np.asarray(pt, dtype=np.float32)
        for yy in range(cy - 1, cy + 2):
            for xx in range(cx - 1, cx + 2):
                for q in self.cells.get((xx, yy), []):
                    d = p - q
                    if float(d[0] * d[0] + d[1] * d[1]) < self.radius2:
                        return True
        return False


def predict_track(track, target_frame: int, mode: str) -> np.ndarray:
    """Predict a track point at target_frame using the current simple motion model."""
    last = track.last_pt().astype(np.float64)
    dt_pred = max(1, target_frame - track.last_seen_frame())
    if mode == "constant_velocity" and len(track.observations) >= 2:
        obs2 = track.observations[-1]
        obs1 = track.observations[-2]
        dt = max(1, obs2.frame - obs1.frame)
        v = (obs2.pt.astype(np.float64) - obs1.pt.astype(np.float64)) / float(dt)
        return (last + v * dt_pred).astype(np.float32)
    return last.astype(np.float32)


def descriptor_distance(descs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Euclidean descriptor distance from every descriptor to one reference."""
    if descs.size == 0:
        return np.empty((0,), dtype=np.float32)
    diff = descs.astype(np.float32) - ref.reshape(1, -1).astype(np.float32)
    return np.linalg.norm(diff, axis=1)
