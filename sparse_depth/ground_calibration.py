"""Ground-plane calibration: the constant camera-vs-road orientation offset.

Per-frame LiDAR road-plane fits give the road normal n*(i) in camera coordinates.
Averaging over a drive cancels the zero-mean local effects (suspension, and grade/
bank on a varied route) and leaves the constant offset -- dominated by the camera
mounting + rectification, plus any systematic road geometry the ROI happens to
sample.  This module is the single source of truth for that calibration's format,
shared by the calibrator tool, the headless evaluator, and the interactive viewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np

from sparse_depth.ground_plane import Plane


def normal_pitch_roll_deg(normal: np.ndarray) -> Tuple[float, float, float]:
    """Signed pitch (tips y->z), roll (tips y->x) and angle-to-down of a near-[0,1,0] normal."""
    n = np.asarray(normal, dtype=np.float64)
    pitch = float(np.degrees(np.arctan2(n[2], n[1])))
    roll = float(np.degrees(np.arctan2(n[0], n[1])))
    angle_down = float(np.degrees(np.arccos(np.clip(n[1], -1.0, 1.0))))
    return pitch, roll, angle_down


def select_road_points(uv, z, roi, image_shape, min_depth_m, max_depth_m) -> np.ndarray:
    """Boolean mask of points whose projection lands in the road ROI + depth band."""
    ij = np.rint(uv).astype(int)
    h, w = image_shape[:2]
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    keep = np.zeros(len(uv), dtype=bool)
    keep[inside] = roi[ij[inside, 1], ij[inside, 0]]
    keep &= (z >= min_depth_m) & (z <= max_depth_m)
    return keep


@dataclass
class GroundCalibration:
    """Constant camera-vs-road plane: unit normal (camera frame, n_y>0) + height."""

    normal: np.ndarray
    height_m: float
    pitch_deg: float
    roll_deg: float
    angle_down_deg: float
    n_frames: int = 0
    meta: dict = field(default_factory=dict)

    def plane(self) -> Plane:
        return Plane(np.asarray(self.normal, dtype=np.float64), -float(self.height_m))

    @classmethod
    def from_normals(cls, normals, heights, meta=None) -> "GroundCalibration":
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        heights = np.asarray(heights, dtype=np.float64).reshape(-1)
        n = normals.mean(axis=0)
        n = n / np.linalg.norm(n)
        pitch, roll, angle = normal_pitch_roll_deg(n)
        per = np.array([normal_pitch_roll_deg(m)[:2] for m in normals])
        info = {
            "pitch_std_deg": float(per[:, 0].std()),
            "roll_std_deg": float(per[:, 1].std()),
            "height_std_m": float(heights.std()),
        }
        info.update(meta or {})
        return cls(n, float(np.median(heights)), pitch, roll, angle, len(normals), info)

    def to_dict(self) -> dict:
        return {
            "normal": [float(v) for v in self.normal],
            "height_m": float(self.height_m),
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "angle_down_deg": self.angle_down_deg,
            "n_frames": self.n_frames,
            "meta": self.meta,
        }

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "GroundCalibration":
        d = json.loads(Path(path).read_text())
        return cls(
            np.asarray(d["normal"], dtype=np.float64), float(d["height_m"]),
            float(d["pitch_deg"]), float(d["roll_deg"]), float(d["angle_down_deg"]),
            int(d.get("n_frames", 0)), d.get("meta", {}),
        )
