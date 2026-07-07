#!/usr/bin/env python3
"""Standalone 2-D plane-homography inspector driven by SIFT+LK tracks.

The feature manager is used only as a correspondence provider.  This viewer
does not change its association, spawning, or triangulation behaviour.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from sparse_depth.config_io import load_argparse_defaults
from sparse_depth.ground_plane import (
    Plane, fit_plane_ransac, level_ground_plane, plane_homography, ray_plane_depth,
    road_trapezoid_mask, robust_warp_residual, warp_source_to_target,
)
from sparse_depth.ground_calibration import GroundCalibration
from sparse_depth.plane_validation import ValidationPolicy, symmetric_photometric_gate
from sparse_depth.interactive_feature_manager_kitti import parse_args as parse_manager_args
from sparse_depth.kitti_io import (
    load_gray, load_kitti_K, load_kitti_poses, load_raw_kitti_cam_calib,
    load_raw_kitti_velo_to_cam, load_velodyne_bin, project_velodyne_to_image,
)
from sparse_depth.geometry import relative_pose
from sparse_depth.geometry import rotation_error_deg, translation_direction_error_deg
from sparse_depth.plane_homography import GroundScaleResult, HomographyResult, estimate_ground_scale_ransac, estimate_lower_image_homography, plane_from_homography_known_pose
from sparse_depth.feature_manager import FeatureManager


WINDOW_NAME = "Plane homography inspector - KITTI"
LIDAR_WINDOW_NAME = "Projected LiDAR depth - plane homography"


@dataclass
class PlaneConfig:
    camera_height_m: float = 1.65
    min_depth_m: float = 2.0
    max_depth_m: float = 45.0
    ground_scale_ransac_thresh_px: float = 2.0
    ground_scale_max_hypotheses: int = 3000
    cache_size: int = 80
    evaluation_cache_size: int = 4
    homography_gap: int = 1
    homography_confirmed_only: bool = True
    homography_include_reacquired: bool = False
    homography_ransac_thresh_px: float = 2.0
    homography_ransac_confidence: float = 0.999
    homography_ransac_max_iters: int = 3000
    homography_roi_top_y_frac: float = 0.58
    homography_roi_bottom_left_frac: float = 0.02
    homography_roi_bottom_right_frac: float = 0.98
    homography_roi_top_left_frac: float = 0.25
    homography_roi_top_right_frac: float = 0.75
    residual_mad_scale: float = 3.5
    lidar_plane_thresh_m: float = 0.06
    lidar_plane_min_inliers: int = 40
    calibration_path: Optional[str] = None
    # On-plane membership (ZNCC) gate; tunable from TOML.
    gate_zncc_threshold: float = 0.35
    gate_patch_radius_px: int = 4
    gate_min_patch_std: float = 5.0
    gate_sparse_reproj_thresh_px: float = 2.0
    gate_min_sparse_inliers: int = 12


def depth_colormap(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth >= lo) & (depth <= hi)
    values = np.zeros(depth.shape, dtype=np.uint8)
    values[valid] = np.clip(255 * (1.0 - (depth[valid] - lo) / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(values, cv2.COLORMAP_JET)
    out[~valid] = (25, 25, 25)
    return out


class Viewer:
    def __init__(self, manager_cfg, plane_cfg: PlaneConfig):
        self.cfg, self.plane_cfg = manager_cfg, plane_cfg
        self.K, self.poses = load_kitti_K(manager_cfg.calib_path), load_kitti_poses(manager_cfg.poses_path)
        self.manager = FeatureManager(manager_cfg, self.K, self.poses)
        self.frame = manager_cfg.start_frame
        self.warmup_start = max(0, manager_cfg.start_frame - max(plane_cfg.homography_gap, manager_cfg.min_hits_to_confirm - 1))
        self.mode = 0  # inliers / blend / raw residual / patch ZNCC / depth
        self.image = None
        self.result = HomographyResult()
        self.selected_id = None
        self.method_index = 0  # GT fixed / generic H / constrained scale-RANSAC
        self.methods = {}
        self.lidar_projection = self._prepare_lidar()
        self.lidar_uv = np.empty((0, 2), dtype=np.float64)
        self.lidar_z = np.empty((0,), dtype=np.float64)
        self.lidar_xyz = np.empty((0, 3), dtype=np.float64)
        self.lidar_plane: Optional[Plane] = None
        self.lidar_plane_stats: dict = {}
        self.calibration: Optional[GroundCalibration] = None
        if getattr(plane_cfg, "calibration_path", None):
            cpath = Path(plane_cfg.calibration_path)
            if cpath.exists():
                self.calibration = GroundCalibration.load(cpath)
                print(f"[calib] loaded {cpath}: pitch {self.calibration.pitch_deg:+.3f} "
                      f"roll {self.calibration.roll_deg:+.3f} h {self.calibration.height_m:.3f}")
            else:
                print(f"[calib] calibration_path not found: {cpath}")
        self.selected_lidar = None
        self.validation_policy = ValidationPolicy(
            patch_radius_px=int(plane_cfg.gate_patch_radius_px),
            min_patch_std=float(plane_cfg.gate_min_patch_std),
            zncc_threshold=float(plane_cfg.gate_zncc_threshold),
            sparse_reproj_thresh_px=float(plane_cfg.gate_sparse_reproj_thresh_px),
            min_sparse_inliers=int(plane_cfg.gate_min_sparse_inliers),
        )
        self.cache = OrderedDict()
        self.evaluation_cache = OrderedDict()
        self.timings = {"load_ms": np.nan, "manager_ms": np.nan, "homography_ms": np.nan, "scale_ms": np.nan, "lidar_ms": np.nan, "photometric_ms": np.nan, "cache_ms": np.nan, "total_ms": np.nan}
        self.gt_baseline_m = self.scale_rot_err_deg = self.scale_t_err_deg = np.nan
        self.last_action = "ready"

    def _prepare_lidar(self):
        if self.cfg.velodyne_dir is None or not self.cfg.velodyne_dir.exists():
            return None
        try:
            T = load_raw_kitti_velo_to_cam(self.cfg.calib_velo_to_cam)
            P, R = load_raw_kitti_cam_calib(self.cfg.calib_cam_to_cam, camera=self.cfg.camera)
            return T, R, P
        except Exception as exc:
            print(f"[warn] LiDAR disabled: {exc}")
            return None

    def _load_lidar(self):
        if self.lidar_projection is None or self.cfg.velodyne_dir is None:
            return np.empty((0, 2)), np.empty((0,)), np.empty((0, 3))
        digits = self.cfg.lidar_digits if self.cfg.lidar_digits is not None else self.cfg.image_digits
        path = self.cfg.velodyne_dir / f"{self.frame:0{digits}d}.bin"
        if not path.exists():
            return np.empty((0, 2)), np.empty((0,)), np.empty((0, 3))
        T, R, P = self.lidar_projection
        return project_velodyne_to_image(load_velodyne_bin(path), T, R, P, self.image.shape, self.cfg.min_lidar_depth_m, self.cfg.max_lidar_depth_m)[:3]

    def _fit_lidar_plane(self):
        """RANSAC-fit the road plane to camera-frame LiDAR points inside the ROI.

        This is the per-frame ground-truth road normal n*(i) and height h*(i):
        select LiDAR returns whose projection lands in the road trapezoid and the
        trusted depth band, then robustly fit a plane to their 3-D camera-frame
        coordinates (rect_xyz from project_velodyne_to_image).
        """
        if self.lidar_xyz.size == 0 or self.lidar_uv.size == 0:
            return None, {"reason": "no_lidar", "n_road": 0, "n_inliers": 0, "rms": np.nan}
        roi = self._roi()
        h, w = self.image.shape[:2]
        uv = np.rint(self.lidar_uv).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        road = np.zeros(len(uv), dtype=bool)
        road[inside] = roi[uv[inside, 1], uv[inside, 0]]
        road &= (self.lidar_z >= self.plane_cfg.min_depth_m) & (self.lidar_z <= self.plane_cfg.max_depth_m)
        plane, inliers, rms = fit_plane_ransac(
            self.lidar_xyz[road], thresh_m=self.plane_cfg.lidar_plane_thresh_m,
            min_inliers=int(self.plane_cfg.lidar_plane_min_inliers),
        )
        stats = {
            "reason": "ok" if plane is not None else "fit_failed",
            "n_road": int(np.sum(road)),
            "n_inliers": int(np.sum(inliers)) if plane is not None else 0,
            "rms": rms,
        }
        return plane, stats

    def _image(self, frame: int):
        image = load_gray(self.cfg.img_dir, frame, self.cfg.image_digits)
        if image is None:
            raise FileNotFoundError(f"Could not read frame {frame}")
        return image

    def _save_snapshot(self):
        start = time.perf_counter()
        self.cache[self.frame] = (copy.deepcopy(self.manager.tracks), self.manager.next_id, self.manager.prev_image.copy() if self.manager.prev_image is not None else None, self.manager.prev_frame, copy.deepcopy(self.manager.last_stats), self.image.copy())
        self.cache.move_to_end(self.frame)
        while len(self.cache) > self.plane_cfg.cache_size: self.cache.popitem(last=False)
        # Evaluation products are replaced wholesale on every _evaluate() and
        # never mutated in place. Keep references: deep-copying several dense
        # maps per method made every forward frame needlessly expensive.
        self.evaluation_cache[self.frame] = (self.result, self.methods, self.lidar_uv, self.lidar_z, self.scale_result, self.ransac_plane_raw, self.gt_baseline_m, self.scale_rot_err_deg, self.scale_t_err_deg, self.lidar_plane, self.lidar_plane_stats, dict(self.timings))
        self.evaluation_cache.move_to_end(self.frame)
        while len(self.evaluation_cache) > self.plane_cfg.evaluation_cache_size: self.evaluation_cache.popitem(last=False)
        self.timings["cache_ms"] = 1000 * (time.perf_counter() - start)

    def _restore_snapshot(self, frame: int):
        tracks, next_id, prev_image, prev_frame, last_stats, image = self.cache[frame]
        self.manager.tracks = copy.deepcopy(tracks); self.manager.next_id = next_id; self.manager.prev_image = prev_image.copy() if prev_image is not None else None; self.manager.prev_frame = prev_frame; self.manager.current_frame = frame; self.manager.last_stats = copy.deepcopy(last_stats)
        self.frame, self.image = frame, image.copy(); self.cache.move_to_end(frame)
        if frame in self.evaluation_cache:
            (self.result, self.methods, self.lidar_uv, self.lidar_z, self.scale_result, self.ransac_plane_raw, self.gt_baseline_m, self.scale_rot_err_deg, self.scale_t_err_deg, self.lidar_plane, self.lidar_plane_stats, self.timings) = self.evaluation_cache[frame]
            self.evaluation_cache.move_to_end(frame); self.timings["manager_ms"] = 0.0
        else:
            self._evaluate(); self._save_snapshot(); self.timings["manager_ms"] = 0.0

    def reset_and_run_to(self, target_frame: int):
        start_total = time.perf_counter()
        image = self._image(self.warmup_start)
        self.manager.close()
        self.manager = FeatureManager(self.cfg, self.K, self.poses)
        self.manager.reset_at(self.warmup_start, image, update_geom_stats=False)
        for frame in range(self.warmup_start + 1, target_frame + 1):
            image = self._image(frame)
            self.manager.step(frame, image, update_geom_stats=False)
        self.frame, self.image = target_frame, image
        self._evaluate()
        self._save_snapshot(); self.timings["total_ms"] = 1000 * (time.perf_counter() - start_total)

    def _evaluate(self):
        total = time.perf_counter(); start = time.perf_counter()
        self.result = estimate_lower_image_homography(self.manager.tracks, self.frame, self.image.shape, self.plane_cfg)
        self.timings["homography_ms"] = 1000 * (time.perf_counter() - start)
        start = time.perf_counter()
        self.lidar_uv, self.lidar_z, self.lidar_xyz = self._load_lidar()
        self.timings["lidar_ms"] = 1000 * (time.perf_counter() - start)
        self.selected_lidar = None
        source = self._image(self.result.source_frame) if self.result.source_frame >= 0 else None
        R, t = relative_pose(self.poses, self.frame, self.result.source_frame)
        fixed = level_ground_plane(self.plane_cfg.camera_height_m)
        ransac_plane, decomp_rmse = (None, np.nan)
        self.ransac_plane_raw = None
        if self.result.H_target_to_source is not None:
            ransac_plane, decomp_rmse = plane_from_homography_known_pose(self.result.H_target_to_source, self.K, R, t)
            self.ransac_plane_raw = ransac_plane
            # RANSAC H supplies the plane *orientation*.  The user-specified
            # metric camera-to-road distance supplies the offset, so both
            # depth methods share the same height prior and react to j/k.
            if ransac_plane is not None:
                ransac_plane = Plane(ransac_plane.normal, -self.plane_cfg.camera_height_m)
        start = time.perf_counter()
        self.methods = {
            "gt": self._make_method("GT fixed plane", plane_homography(self.K, R, t, fixed), fixed, source, np.nan),
            "ransac": self._make_method("RANSAC H plane", self.result.H_target_to_source, ransac_plane, source, decomp_rmse),
        }
        scale_cfg = SimpleNamespace(**{**vars(self.cfg), **vars(self.plane_cfg)})
        start = time.perf_counter(); self.scale_result: GroundScaleResult = estimate_ground_scale_ransac(self.manager.tracks, self.frame, self.image.shape, self.K, scale_cfg, fixed)
        self.timings["scale_ms"] = 1000 * (time.perf_counter() - start)
        self.gt_baseline_m = float(np.linalg.norm(t))
        if self.scale_result.reason == "ok":
            self.scale_rot_err_deg = rotation_error_deg(self.scale_result.R_target_to_source, R)
            self.scale_t_err_deg = translation_direction_error_deg(self.scale_result.t_direction_target_to_source, t)
        else: self.scale_rot_err_deg = self.scale_t_err_deg = np.nan
        if self.scale_result.reason == "ok":
            H_scale = self.K @ (self.scale_result.R_target_to_source - self.scale_result.scale_m * np.outer(self.scale_result.t_direction_target_to_source, fixed.normal / fixed.offset)) @ np.linalg.inv(self.K)
            self.methods["scale"] = self._make_method("Estimated pose + scale RANSAC", H_scale, fixed, source, np.nan)
        else:
            self.methods["scale"] = {"name": "Estimated pose + scale RANSAC", "reason": self.scale_result.reason, "plane": fixed}
        self.lidar_plane, self.lidar_plane_stats = self._fit_lidar_plane()
        if self.lidar_plane is not None:
            self.methods["lidar"] = self._make_method("LiDAR road-plane fit", plane_homography(self.K, R, t, self.lidar_plane), self.lidar_plane, source, np.nan)
        else:
            self.methods["lidar"] = {"name": "LiDAR road-plane fit", "reason": self.lidar_plane_stats.get("reason", "no_lidar"), "plane": fixed}
        self.methods["lidar"]["fit_stats"] = self.lidar_plane_stats
        if self.calibration is not None:
            calib_plane = self.calibration.plane()
            self.methods["calib"] = self._make_method("Calibrated offset plane", plane_homography(self.K, R, t, calib_plane), calib_plane, source, np.nan)
        else:
            self.methods["calib"] = {"name": "Calibrated offset plane", "reason": "no_calibration", "plane": fixed}
        self._attach_sparse_stats("gt", self.result.candidate_ids)
        self._attach_sparse_stats("ransac", self.result.candidate_ids)
        self._attach_sparse_stats("scale", self.scale_result.candidate_ids)
        self._attach_sparse_stats("lidar", self.result.candidate_ids)
        self._attach_sparse_stats("calib", self.result.candidate_ids)
        self.timings["photometric_ms"] = 1000 * (time.perf_counter() - start)
        self.timings["total_ms"] = 1000 * (time.perf_counter() - total)

    def _make_method(self, name, H, plane, source, decomp_rmse):
        roi = self._roi()
        if H is None or plane is None or source is None:
            return {"name": name, "H": H, "plane": plane, "reason": "no_valid_plane", "rmse": decomp_rmse}
        validation = symmetric_photometric_gate(self.image, source, H, roi, self.validation_policy)
        warped, base_gate = validation["warped"], validation["base_gate"]
        residual, gate, offset, median, threshold = robust_warp_residual(self.image, warped, base_gate, roi, self.plane_cfg.residual_mad_scale)
        depth = ray_plane_depth(self.K, self.image.shape, plane)
        return {"name": name, "H": H, "plane": plane, "warped": warped, "residual": residual, "raw_gate": gate, "base_gate": base_gate, "zncc": validation["zncc_forward"], "informative": validation["informative"], "patch_gate": validation["green"], "red": validation["red"], "photometric_gate": validation["keep_gate"], "depth": depth, "offset": offset, "median": median, "threshold": threshold, "rmse": decomp_rmse, "reason": "ok"}

    def _attach_sparse_stats(self, key, candidate_ids):
        data = self.methods[key]
        if data.get("reason") != "ok": return
        ids, current, source = [], [], []
        for tr in self.manager.tracks:
            if tr.id not in candidate_ids or not tr.active_in(self.frame): continue
            previous = tr.observation_at(self.result.source_frame)
            if previous is None: continue
            ids.append(tr.id); current.append(tr.last_pt()); source.append(previous.pt)
        if not ids:
            data.update(sparse_inliers=0, sparse_median=np.nan, geometry_ok=False); data["photometric_gate"][:] = False; return
        projected = cv2.perspectiveTransform(np.asarray(current, np.float32).reshape(-1, 1, 2), data["H"]).reshape(-1, 2)
        errors = np.linalg.norm(projected - np.asarray(source), axis=1)
        inliers = errors <= self.validation_policy.sparse_reproj_thresh_px
        data.update(sparse_inliers=int(np.sum(inliers)), sparse_median=float(np.median(errors[inliers])) if np.any(inliers) else np.nan, geometry_ok=bool(np.sum(inliers) >= self.validation_policy.min_sparse_inliers))
        if not data["geometry_ok"]: data["photometric_gate"][:] = False

    def _metrics(self, data, photometric: bool):
        if data.get("reason") != "ok" or self.lidar_uv.size == 0:
            return 0, np.nan, np.nan
        xy = np.rint(self.lidar_uv).astype(int); h, w = self.image.shape[:2]
        good = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xy, z = xy[good], self.lidar_z[good]
        gate = data["photometric_gate"] if photometric else data["base_gate"]
        keep = gate[xy[:, 1], xy[:, 0]] & np.isfinite(data["depth"][xy[:, 1], xy[:, 0]])
        if not np.any(keep): return 0, np.nan, np.nan
        pred = data["depth"][xy[keep, 1], xy[keep, 0]]; rel = np.abs(pred - z[keep]) / np.maximum(z[keep], 1e-9)
        return int(rel.size), float(np.mean(rel)), float(np.median(rel))

    def step(self, delta: int):
        target = self.frame + delta
        maximum = min(len(self.poses) - 1, self.cfg.end_frame) if self.cfg.end_frame is not None else len(self.poses) - 1
        if self.cfg.start_frame <= target <= maximum:
            if target in self.cache:
                self._restore_snapshot(target)
                self.last_action = f"restored cached frame {target}"
            elif delta == 1:
                start = time.perf_counter(); self.image = self._image(target); self.timings["load_ms"] = 1000 * (time.perf_counter() - start)
                start = time.perf_counter(); self.manager.step(target, self.image, update_geom_stats=False); self.frame = target; self.timings["manager_ms"] = 1000 * (time.perf_counter() - start); self._evaluate(); self._save_snapshot(); self.last_action = f"computed frame {target}"
            else:
                self.reset_and_run_to(target); self.last_action = f"rebuilt frame {target}"
        else:
            self.last_action = f"frame {target} outside [{self.cfg.start_frame}, {maximum}]"

    def _roi(self):
        return road_trapezoid_mask(
            self.image.shape, self.plane_cfg.homography_roi_top_y_frac,
            self.plane_cfg.homography_roi_bottom_left_frac, self.plane_cfg.homography_roi_bottom_right_frac,
            self.plane_cfg.homography_roi_top_left_frac, self.plane_cfg.homography_roi_top_right_frac,
        )

    def _render(self):
        target = self.image
        target_bgr = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
        roi = self._roi()
        source_frame = self.result.source_frame
        method_key = ("gt", "ransac", "scale", "lidar", "calib")[self.method_index]
        method = self.methods[method_key]
        warped = method.get("warped", target)
        residual = method.get("residual", np.full(target.shape, np.nan))
        raw_ok = method.get("raw_gate", np.zeros(target.shape, bool))
        residual_median, residual_threshold = method.get("median", np.nan), method.get("threshold", np.nan)
        if self.mode == 0:
            view = target_bgr
            title = "RANSAC inliers: green=inlier, red=eligible outlier"
        elif self.mode == 1:
            view = cv2.addWeighted(target_bgr, 0.5, cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR), 0.5, 0)
            title = "blend: target + source warped by feature-homography"
        elif self.mode == 2:
            scale = max(float(residual_threshold) * 1.5 if np.isfinite(residual_threshold) else 1.0, 1.0)
            view = cv2.applyColorMap(np.clip(np.nan_to_num(residual) * 255.0 / scale, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            view[~roi] = target_bgr[~roi] // 3
            view[raw_ok] = (50, 180, 50)
            title = "raw intensity residual (display only; not evaluation gate)"
        elif self.mode == 3:
            informative = method.get("informative", np.zeros(target.shape, bool)); patch_gate = method.get("patch_gate", np.zeros(target.shape, bool))
            view = target_bgr // 3
            view[roi] = (80, 80, 80)
            view[informative & ~patch_gate] = (30, 30, 230)
            view[patch_gate] = (50, 180, 50)
            title = "patch ZNCC: gray=uninformative, green=agree, red=disagree"
        else:
            depth = method.get("depth", np.full(target.shape, np.nan))
            view = depth_colormap(depth, self.plane_cfg.min_depth_m, self.plane_cfg.max_depth_m)
            view[~roi] = target_bgr[~roi] // 3
            title = "metric depth of selected plane"
        contours, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (0, 220, 255), 1, cv2.LINE_AA)
        if self.mode == 0:
            for tr in self.manager.active_tracks(self.frame, confirmed_only=False):
                if method_key == "scale":
                    candidate_ids, inlier_ids = self.scale_result.candidate_ids, self.scale_result.inlier_ids
                else:
                    candidate_ids, inlier_ids = self.result.candidate_ids, self.result.inlier_ids
                if tr.id not in candidate_ids:
                    continue
                color = (60, 230, 60) if tr.id in inlier_ids else (40, 40, 255)
                cv2.circle(view, tuple(np.rint(tr.last_pt()).astype(int)), 3, color, -1, cv2.LINE_AA)
        h, w = view.shape[:2]
        panel = np.zeros((max(h, self.cfg.side_panel_height), self.cfg.side_panel_width, 3), dtype=np.uint8)
        all_metrics = {key: (self._metrics(data, False), self._metrics(data, True)) for key, data in self.methods.items()}
        raw_plane = self.ransac_plane_raw
        if raw_plane is None:
            normal_line, raw_plane_line = "RANSAC H plane: unavailable", ""
        else:
            down = np.array([0.0, 1.0, 0.0])
            angle = np.degrees(np.arccos(np.clip(float(np.dot(raw_plane.normal, down)), -1.0, 1.0)))
            normal_line = f"RANSAC n(raw): [{raw_plane.normal[0]:+.3f}, {raw_plane.normal[1]:+.3f}, {raw_plane.normal[2]:+.3f}] | angle-to-down {angle:.2f} deg"
            raw_plane_line = f"RANSAC d(raw from H+GT pose): {raw_plane.offset:+.3f} m | depth uses d=-h"
        if self.lidar_plane is not None:
            down = np.array([0.0, 1.0, 0.0])
            lidar_angle = np.degrees(np.arccos(np.clip(float(np.dot(self.lidar_plane.normal, down)), -1.0, 1.0)))
            lidar_fit_line = (
                f"LiDAR-fit n*: [{self.lidar_plane.normal[0]:+.3f}, {self.lidar_plane.normal[1]:+.3f}, {self.lidar_plane.normal[2]:+.3f}] "
                f"angle-down {lidar_angle:.2f} deg | h*={-self.lidar_plane.offset:.3f} m | "
                f"inl {self.lidar_plane_stats.get('n_inliers', 0)}/{self.lidar_plane_stats.get('n_road', 0)} rms {self.lidar_plane_stats.get('rms', float('nan')):.3f} m"
            )
        else:
            lidar_fit_line = f"LiDAR-fit n*: {self.lidar_plane_stats.get('reason', 'n/a')}"
        if self.calibration is not None:
            calib_line = (
                f"Calib plane: pitch {self.calibration.pitch_deg:+.3f} roll {self.calibration.roll_deg:+.3f} deg | "
                f"h {self.calibration.height_m:.3f} m | {self.calibration.n_frames} frames"
            )
        else:
            calib_line = "Calib plane: none (set calibration_path in config)"
        lines = [
            "2-D plane-homography inspector", f"target/source: {self.frame} / {source_frame}", f"mode: {title}",
            f"method: {method['name']} | z/x/c/v/g select GT/H/scale/LiDAR/Calib", f"method status: {method.get('reason')} | decomp RMSE {method.get('rmse', np.nan):.3g}",
            f"sparse validation: {method.get('sparse_inliers', 0)} inliers | median {method.get('sparse_median', np.nan):.2f}px | healthy={method.get('geometry_ok', False)}",
            f"camera height h: {self.plane_cfg.camera_height_m:.3f} m (j/k changes by 0.020 m)",
            normal_line, raw_plane_line, lidar_fit_line, calib_line,
            f"scale RANSAC: {self.scale_result.reason} | s={self.scale_result.scale_m:.3f}m | road inl {len(self.scale_result.inlier_ids)} | med {self.scale_result.median_reproj_px:.2f}px",
            f"Essential pose: {self.scale_result.pose_inliers}/{self.scale_result.pose_pairs} | R err {self.scale_rot_err_deg:.2f} deg | t-dir err {self.scale_t_err_deg:.2f} deg",
            f"GT baseline: {self.gt_baseline_m:.3f} m",
            f"RANSAC: {len(self.result.inlier_ids)}/{len(self.result.candidate_ids)} inliers", f"median reproj: {self.result.median_reproj_px:.2f}px | {self.result.reason}",
            f"threshold: {self.plane_cfg.homography_ransac_thresh_px:.2f}px | gap {self.plane_cfg.homography_gap}",
            f"raw residual median/threshold: {residual_median:.1f}/{residual_threshold:.1f}", "",
            *[f"LiDAR {key} ROI: n={all_metrics[key][0][0]} | mean/med AbsRel {all_metrics[key][0][1]:.3f}/{all_metrics[key][0][2]:.3f}" for key in ("gt", "ransac", "scale", "lidar", "calib")],
            *[f"LiDAR {key} ZNCC keep gray+green: n={all_metrics[key][1][0]} | mean/med AbsRel {all_metrics[key][1][1]:.3f}/{all_metrics[key][1][2]:.3f}" for key in ("gt", "ransac", "scale", "lidar", "calib")],
            f"depth scale {self.plane_cfg.min_depth_m:.0f}m red -> {self.plane_cfg.max_depth_m:.0f}m blue", "",
            f"timing ms: load {self.timings['load_ms']:.1f} | mgr {self.timings['manager_ms']:.1f} | H {self.timings['homography_ms']:.1f} | scale {self.timings['scale_ms']:.1f}",
            f"timing ms: LiDAR {self.timings['lidar_ms']:.1f} | dense-photo {self.timings['photometric_ms']:.1f} | cache {self.timings['cache_ms']:.1f} | eval-total {self.timings['total_ms']:.1f}",
            f"cache: {len(self.cache)}/{self.plane_cfg.cache_size} frames",
            f"last command: {self.last_action}",
            "1..5: inliers/blend/raw/ZNCC/depth | m: next",
            "z/x/c/v/g: GT-H / generic-H / scale / LiDAR-fit / Calib | j/k: height",
            "n/b: frame | click main/LiDAR: diagnostic | s: screenshot | q: quit",
        ]
        y = 22
        max_chars = max(28, int((self.cfg.side_panel_width - 20) / 7.4))
        for line in lines:
            words, row = line.split(), ""
            for word in words or [""]:
                candidate = f"{row} {word}".strip()
                if row and len(candidate) > max_chars:
                    cv2.putText(panel, row, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .40, (235, 235, 235), 1, cv2.LINE_AA); y += 18; row = word
                else: row = candidate
            cv2.putText(panel, row, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .40, (235, 235, 235), 1, cv2.LINE_AA); y += 18
        view = cv2.copyMakeBorder(view, 0, panel.shape[0] - h, 0, 0, cv2.BORDER_CONSTANT)
        return np.hstack([view, panel])

    def _click(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or self.image is None:
            return
        h, w = self.image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        self._inspect_pixel(np.array([x, y], dtype=float))
        tracks = self.manager.active_tracks(self.frame, confirmed_only=False)
        if not tracks:
            return
        distances = np.array([np.linalg.norm(tr.last_pt() - np.array([x, y])) for tr in tracks])
        k = int(np.argmin(distances))
        if distances[k] > self.cfg.inspect_radius_px:
            return
        tr = tracks[k]
        if self.method_index == 2:
            print(f"[scale click] id={tr.id} point={tr.last_pt()} candidate={tr.id in self.scale_result.candidate_ids} scaleInlier={tr.id in self.scale_result.inlier_ids}")
        else:
            print(f"[homography click] id={tr.id} point={tr.last_pt()} candidate={tr.id in self.result.candidate_ids} inlier={tr.id in self.result.inlier_ids} reproj={self.result.reproj_error_by_id.get(tr.id, np.nan):.2f}px")

    def _inspect_pixel(self, point):
        x, y = np.rint(point).astype(int); h, w = self.image.shape[:2]
        if not (0 <= x < w and 0 <= y < h): return
        data = self.methods[("gt", "ransac", "scale", "lidar", "calib")[self.method_index]]
        depth = data.get("depth", np.full(self.image.shape, np.nan)); gate = data.get("photometric_gate", np.zeros(self.image.shape, bool))
        if self.lidar_uv.size:
            distances = np.linalg.norm(self.lidar_uv - np.array([[x, y]]), axis=1); i = int(np.argmin(distances))
            if distances[i] <= self.cfg.lidar_click_radius_px:
                self.selected_lidar = i; z = self.lidar_z[i]; pred = depth[y, x]
                diff = pred - z if np.isfinite(pred) else np.nan
                print(f"[plane click] pixel=({x},{y}) {data['name']} Z={pred:.2f}m keepGrayOrGreen={bool(gate[y,x])} green={bool(data.get('patch_gate', np.zeros(self.image.shape,bool))[y,x])} ZNCC={data.get('zncc', np.full(self.image.shape,np.nan))[y,x]:.3f} | LiDAR Z={z:.2f}m at {distances[i]:.2f}px | diff={diff:+.2f}m")
                return
        print(f"[plane click] pixel=({x},{y}) {data['name']} Z={depth[y,x]:.2f}m gate={bool(gate[y,x])} | no LiDAR within {self.cfg.lidar_click_radius_px:.1f}px")

    def _draw_lidar(self):
        if self.lidar_projection is None: return
        canvas = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR); n = len(self.lidar_z)
        if n:
            idx = np.arange(n) if n <= self.cfg.max_lidar_vis_points else np.linspace(0, n - 1, self.cfg.max_lidar_vis_points).astype(int)
            z = self.lidar_z[idx]; value = np.clip(255 * (1 - (z - self.cfg.min_lidar_depth_m) / max(self.cfg.max_lidar_depth_m - self.cfg.min_lidar_depth_m, 1e-6)), 0, 255).astype(np.uint8)
            colors = cv2.applyColorMap(value.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
            for p, c in zip(self.lidar_uv[idx], colors): cv2.circle(canvas, tuple(np.rint(p).astype(int)), 2, tuple(int(v) for v in c), -1, cv2.LINE_AA)
        if self.selected_lidar is not None and self.selected_lidar < n:
            p = self.lidar_uv[self.selected_lidar]; cv2.circle(canvas, tuple(np.rint(p).astype(int)), 7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(canvas, f"Projected LiDAR | {n} points | click for selected plane depth", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (235,235,235), 1, cv2.LINE_AA)
        cv2.putText(canvas, "red=near, blue=far", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,220,0), 1, cv2.LINE_AA)
        cv2.imshow(LIDAR_WINDOW_NAME, canvas)

    def _lidar_click(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or self.lidar_uv.size == 0: return
        d = np.linalg.norm(self.lidar_uv - np.array([[x, y]]), axis=1); i = int(np.argmin(d))
        if d[i] <= self.cfg.lidar_click_radius_px:
            self.selected_lidar = i; self._inspect_pixel(self.lidar_uv[i])

    def run(self):
        self.reset_and_run_to(self.frame)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
        cv2.setMouseCallback(WINDOW_NAME, self._click)
        if self.lidar_projection is not None:
            cv2.namedWindow(LIDAR_WINDOW_NAME, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0)); cv2.setMouseCallback(LIDAR_WINDOW_NAME, self._lidar_click)
        while True:
            canvas = self._render(); cv2.imshow(WINDOW_NAME, canvas)
            self._draw_lidar()
            cv2.resizeWindow(WINDOW_NAME, int(canvas.shape[1] * self.cfg.resize), int(canvas.shape[0] * self.cfg.resize))
            raw_key = cv2.waitKeyEx(0)
            key = raw_key & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                self.manager.close(); cv2.destroyAllWindows(); return
            if key in (ord("n"), ord("N"), ord("d"), ord("D"), 83): self.step(1)
            elif key in (ord("b"), ord("B"), ord("a"), ord("A"), 81): self.step(-1)
            elif key in (ord("m"), ord("M")): self.mode = (self.mode + 1) % 5; self.last_action = f"mode {self.mode + 1}"
            elif key in [ord(str(i)) for i in range(1, 6)]: self.mode = int(chr(key)) - 1
            elif key in (ord("z"), ord("Z")): self.method_index = 0; self.last_action = "method GT"
            elif key in (ord("x"), ord("X")): self.method_index = 1; self.last_action = "method generic H"
            elif key in (ord("c"), ord("C")): self.method_index = 2; self.last_action = "method scale RANSAC"
            elif key in (ord("v"), ord("V")): self.method_index = 3; self.last_action = "method LiDAR fit"
            elif key in (ord("g"), ord("G")): self.method_index = 4; self.last_action = "method calibrated"
            elif key in (ord("j"), ord("J")):
                self.plane_cfg.camera_height_m = max(0.10, self.plane_cfg.camera_height_m - 0.02); self.evaluation_cache.clear(); self._evaluate(); self._save_snapshot()
                self.last_action = f"height {self.plane_cfg.camera_height_m:.3f} m"
            elif key in (ord("k"), ord("K")):
                self.plane_cfg.camera_height_m += 0.02; self.evaluation_cache.clear(); self._evaluate(); self._save_snapshot()
                self.last_action = f"height {self.plane_cfg.camera_height_m:.3f} m"
            elif key in (ord("s"), ord("S")):
                self.cfg.output_root.mkdir(parents=True, exist_ok=True)
                path = self.cfg.output_root / f"plane_homography_{self.frame:06d}_mode{self.mode}.png"; cv2.imwrite(str(path), canvas); print(f"saved {path}")
        self.manager.close(); cv2.destroyAllWindows()


def parse_configs():
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--config", type=Path, action="append", default=[])
    ns, _ = pre.parse_known_args()
    values = load_argparse_defaults(ns.config)
    fields = {name: values.get(name, field.default) for name, field in PlaneConfig.__dataclass_fields__.items()}
    # parse_manager_args reads the same --config values and builds the existing
    # manager Config; plane-only TOML keys are harmless defaults to that parser.
    return parse_manager_args(), PlaneConfig(**fields)


if __name__ == "__main__":
    Viewer(*parse_configs()).run()
