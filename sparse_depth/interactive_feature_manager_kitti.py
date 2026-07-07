#!/usr/bin/env python3
r"""
interactive_feature_manager_kitti.py

Interactive hybrid feature-manager prototype for KITTI.

This version starts from the SIFT v3 manager and adds an LK-first continuation
path. SIFT still births tracks, rescues LK failures, refreshes descriptors, and
spawns new bucket-fill tracks.

Compared with the earlier prototype, this version adds a practical subset of the
front-end improvements discussed in chat, while deliberately keeping dynamic-object
reasoning and keyframe/direct-keyframe matching out for now.

Main additions
--------------
1.  Feature-manager population is separated from output selection.
2.  Triangulation candidates are not confirmed-only by default. Tracks with at
    least --triang-min-hits observations are evaluated.
3.  Actual GT-pose DLT triangulation preview is computed for candidate pairs.
4.  Estimated-pose/RANSAC diagnostics are added for a fixed current-vs-past gap.
5.  Track quality scores are maintained for display/selection.
6.  Bucket diagnostics distinguish active / confirmed / triangulation-eligible tracks.
7.  Soft per-bucket selection is used for display/output instead of blindly showing all tracks.
8.  Duplicate track suppression is available.
9.  Missed-track prediction uncertainty is visualized.
10. Reacquired tracks can be treated as probationary for triangulation.
11. Track gaps are represented as missing observations; paths are not drawn through missing frames.
12. Adaptive soft SIFT spawning/rescue is supported. Soft detections can rescue unmatched tracks, then unused detections may spawn new tracks.
13. Per-frame CSV logging is available.
14. A dashboard-style side panel avoids bottom-text overflow.
15. Optional mouse-click track inspection prints detailed track info.

Geometry in triang/depth/pose views is still diagnostic. Association/spawning does not use GT.

Controls
--------
n/d/right  : next frame
b/a/left   : previous cached frame
r          : reset manager at current frame
R          : reset manager at original --start frame
m          : cycle view mode
1..7       : status / buckets / age / triang / depth / quality / pose
p          : toggle track paths
v          : toggle velocity arrows
l          : toggle lost/predicted tracks
G          : toggle bucket grid
+ / -      : increase/decrease displayed track cap
s          : save screenshot
h          : print controls
q / ESC    : quit
mouse left : inspect nearest active track in console
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparse_depth.config_io import load_argparse_defaults
from sparse_depth.feature_utils import predict_track
from sparse_depth.kitti_io import (
    load_gray,
    load_kitti_K,
    load_kitti_poses,
    load_odometry_lidar_projection,
    load_raw_kitti_cam_calib,
    load_raw_kitti_velo_to_cam,
    load_velodyne_bin,
    match_sparse_to_lidar_with_radius,
    project_velodyne_to_image,
    read_kitti_calib_file,
)
from sparse_depth.pose_eval import PoseEval
from sparse_depth.manager_config import (
    Config,
    DEFAULT_CALIB_PATH,
    DEFAULT_IMAGE_DIGITS,
    DEFAULT_IMG_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POSES_PATH,
    resolve_detector_descriptor,
    validate_feature_axes,
)
from sparse_depth.feature_manager import FeatureManager
from sparse_depth.track_types import FrameStats, Track
from sparse_depth.triangulation import GOOD_TRIANG_LABELS, TriangInfo


WINDOW_NAME = "SIFT+LK feature manager v1 - KITTI"
LIDAR_WINDOW_NAME = "Projected LiDAR depth - KITTI"

VIEW_STATUS = "status"
VIEW_BUCKETS = "buckets"
VIEW_AGE = "age"
VIEW_TRIANG = "triang"
VIEW_DEPTH = "depth"
VIEW_QUALITY = "quality"
VIEW_POSE = "pose"
VIEW_ORDER = [VIEW_STATUS, VIEW_BUCKETS, VIEW_AGE, VIEW_TRIANG, VIEW_DEPTH, VIEW_QUALITY, VIEW_POSE]

# OpenCV BGR colors
CLR_GREEN = (60, 230, 60)
CLR_DARK_GREEN = (40, 150, 40)
CLR_YELLOW = (0, 220, 255)
CLR_BLUE = (255, 120, 40)
CLR_CYAN = (255, 220, 0)
CLR_ORANGE = (0, 150, 255)
CLR_RED = (40, 40, 255)
CLR_GRAY = (145, 145, 145)
CLR_DARK_GRAY = (80, 80, 80)
CLR_WHITE = (235, 235, 235)
CLR_BLACK = (0, 0, 0)
CLR_MAGENTA = (255, 0, 255)


def choose_display_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int32)
    return np.linspace(0, n - 1, max_n).astype(np.int32)


def put_text_lines(img: np.ndarray, lines: Sequence[str], x: int, y: int, color=CLR_WHITE, scale=0.43, dy=17, max_width_chars: int = 80):
    yy = y
    for line in lines:
        if len(line) > max_width_chars:
            line = line[: max_width_chars - 1] + "…"
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        yy += dy


def draw_cross(img: np.ndarray, pt: np.ndarray, color, size=5, thickness=1):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness, cv2.LINE_AA)


def nearest_point_within(pts: np.ndarray, query: np.ndarray, radius_px: float):
    if pts.size == 0:
        return None
    d = np.linalg.norm(pts - np.asarray(query, dtype=np.float64).reshape(1, 2), axis=1)
    k = int(np.argmin(d))
    if float(d[k]) <= radius_px:
        return k, float(d[k])
    return None


def effective_lidar_click_radius(cfg: Config) -> float:
    """Use the match radius for viewer clicks unless an explicit larger click radius is set."""
    return max(float(cfg.lidar_radius_px), float(cfg.lidar_click_radius_px))


def finite_or_nan(x: float) -> float:
    return float(x) if np.isfinite(x) else np.nan


def prepare_lidar_projection_for_viewer(cfg: Config):
    if cfg.velodyne_dir is None:
        return None
    if not cfg.velodyne_dir.exists():
        print(f"[warn] Velodyne dir not found, LiDAR viewer disabled: {cfg.velodyne_dir}")
        return None
    try:
        if cfg.calib_velo_to_cam is not None and cfg.calib_cam_to_cam is not None:
            T_cam0_velo = load_raw_kitti_velo_to_cam(cfg.calib_velo_to_cam)
            P_rect, R_rect_4 = load_raw_kitti_cam_calib(cfg.calib_cam_to_cam, camera=cfg.camera)
        elif cfg.calib_velo_to_cam is not None:
            print("[warn] calib_velo_to_cam provided without calib_cam_to_cam; LiDAR viewer uses P0 from --calib and identity rectification.")
            T_cam0_velo = load_raw_kitti_velo_to_cam(cfg.calib_velo_to_cam)
            data = read_kitti_calib_file(cfg.calib_path)
            P_rect = np.asarray(data["P0"], dtype=np.float64).reshape(3, 4)
            R_rect_4 = np.eye(4, dtype=np.float64)
        else:
            T_cam0_velo, R_rect_4, P_rect = load_odometry_lidar_projection(cfg.calib_path)
    except Exception as e:
        print(f"[warn] LiDAR viewer unavailable: {e}")
        return None
    return T_cam0_velo, R_rect_4, P_rect


# ============================================================
# Visualization
# ============================================================

class Viewer:
    def __init__(self, cfg: Config, K: np.ndarray, poses: Sequence[np.ndarray]):
        self.cfg = cfg
        self.K = K
        self.poses = poses
        self.manager = FeatureManager(cfg, K, poses)
        self.frame = cfg.start_frame
        self.mode_idx = 0
        self.show_paths = True
        self.show_vectors = False
        self.show_lost = True
        self.show_grid = True
        self.draw_max_tracks = cfg.draw_max_tracks
        self.cache: List[Tuple[int, List[Track], int, FrameStats]] = []
        self.cache_index = 0
        self.last_image: Optional[np.ndarray] = None
        self.last_display_shape: Optional[Tuple[int, int]] = None
        self.selected_track_id: Optional[int] = None
        self.last_pose_eval = PoseEval()
        self.lidar_projection = prepare_lidar_projection_for_viewer(cfg)
        self.last_lidar_uv = np.empty((0, 2), dtype=np.float64)
        self.last_lidar_z = np.empty((0,), dtype=np.float64)
        self.last_lidar_canvas_shape: Optional[Tuple[int, int]] = None
        self.selected_lidar_index: Optional[int] = None
        cfg.output_root.mkdir(parents=True, exist_ok=True)

    @property
    def mode(self):
        return VIEW_ORDER[self.mode_idx]

    def initialize(self):
        img = load_gray(self.cfg.img_dir, self.frame, self.cfg.image_digits)
        if img is None:
            raise FileNotFoundError(f"Could not read start image {self.frame}")
        stats = self.manager.reset_at(self.frame, img, update_geom_stats=self.mode in [VIEW_TRIANG, VIEW_DEPTH, VIEW_POSE])
        self.last_image = img
        self.cache = [(self.frame, copy.deepcopy(self.manager.tracks), self.manager.next_id, copy.deepcopy(stats))]
        self.cache_index = 0
        self._show(img, stats)

    def run(self):
        window_flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0)
        cv2.namedWindow(WINDOW_NAME, window_flags)
        if self.lidar_projection is not None:
            cv2.namedWindow(LIDAR_WINDOW_NAME, window_flags)
        self.initialize()
        h, w = self.last_image.shape[:2]
        display_h = max(h, self.cfg.side_panel_height)
        cv2.resizeWindow(WINDOW_NAME, int((w + self.cfg.side_panel_width) * self.cfg.resize), int(display_h * self.cfg.resize))
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        if self.lidar_projection is not None:
            cv2.resizeWindow(LIDAR_WINDOW_NAME, int(w * self.cfg.resize), int(h * self.cfg.resize))
            cv2.setMouseCallback(LIDAR_WINDOW_NAME, self._on_lidar_mouse)
        print_controls()
        try:
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in [ord('q'), 27]:
                    break
                elif key in [ord('n'), ord('d'), 83]:
                    self.next_frame()
                elif key in [ord('b'), ord('a'), 81]:
                    self.prev_frame()
                elif key == ord('r'):
                    self.reset_at_current()
                elif key == ord('R'):
                    self.frame = self.cfg.start_frame
                    self.manager.close()
                    self.manager = FeatureManager(self.cfg, self.K, self.poses)
                    self.initialize()
                elif key == ord('m'):
                    self.mode_idx = (self.mode_idx + 1) % len(VIEW_ORDER)
                    self.refresh()
                elif key in [ord(str(i)) for i in range(1, 8)]:
                    self.mode_idx = int(chr(key)) - 1
                    self.refresh()
                elif key == ord('p'):
                    self.show_paths = not self.show_paths
                    self.refresh()
                elif key == ord('v'):
                    self.show_vectors = not self.show_vectors
                    self.refresh()
                elif key == ord('l'):
                    self.show_lost = not self.show_lost
                    self.refresh()
                elif key == ord('G'):
                    self.show_grid = not self.show_grid
                    self.refresh()
                elif key in [ord('+'), ord('=')]:
                    self.draw_max_tracks = min(10000, self.draw_max_tracks + 250)
                    self.refresh()
                elif key in [ord('-'), ord('_')]:
                    self.draw_max_tracks = max(100, self.draw_max_tracks - 250)
                    self.refresh()
                elif key == ord('s'):
                    self.save_screenshot()
                elif key == ord('h'):
                    print_controls()
                else:
                    print(f"Unhandled key: {key}")
        finally:
            self.manager.close()
            cv2.destroyAllWindows()

    def next_frame(self):
        if self.cache_index < len(self.cache) - 1:
            self.cache_index += 1
            self._restore_cache()
            self.refresh()
            return
        next_f = self.frame + 1
        if self.cfg.end_frame is not None and next_f > self.cfg.end_frame:
            print("Reached --end-frame")
            return
        if next_f >= len(self.poses):
            print("Reached end of pose file")
            return
        t0 = time.perf_counter()
        img = load_gray(self.cfg.img_dir, next_f, self.cfg.image_digits)
        t_load = time.perf_counter()
        if img is None:
            print(f"Could not read frame {next_f}")
            return
        stats = self.manager.step(next_f, img, update_geom_stats=self.mode in [VIEW_TRIANG, VIEW_DEPTH, VIEW_POSE])
        t_done = time.perf_counter()
        stats.time_load_ms = 1000.0 * (t_load - t0)
        stats.time_manage_depth_ms = 1000.0 * (t_done - t_load)
        stats.time_no_draw_ms = 1000.0 * (t_done - t0)
        print(
            f"[timing] frame {next_f:06d} no_draw={stats.time_no_draw_ms:.1f} ms "
            f"(load={stats.time_load_ms:.1f}, manage+depth={stats.time_manage_depth_ms:.1f}) "
            f"lk={stats.time_lk_track_ms:.1f} det={stats.time_primary_detect_ms:.1f} assoc={stats.time_primary_assoc_ms:.1f} "
            f"softDet={stats.time_soft_detect_ms:.1f} softRescue={stats.time_soft_rescue_ms:.1f} "
            f"spawn={stats.time_spawn_ms:.1f} dup={stats.time_duplicate_ms:.1f} "
            f"counts={stats.time_count_stats_ms:.1f} geom={stats.time_geom_depth_pose_ms:.1f} "
            f"lk={stats.lk_accepted}/{stats.lk_attempted} depth_valid={stats.depth_valid} tracks={stats.active}"
        )
        self.frame = next_f
        self.last_image = img
        self.selected_lidar_index = None
        self.cache.append((self.frame, copy.deepcopy(self.manager.tracks), self.manager.next_id, copy.deepcopy(stats)))
        self.cache_index = len(self.cache) - 1
        self._show(img, stats)

    def prev_frame(self):
        if self.cache_index <= 0:
            print("Already at first cached frame")
            return
        self.cache_index -= 1
        self._restore_cache()
        self.refresh()

    def reset_at_current(self):
        img = load_gray(self.cfg.img_dir, self.frame, self.cfg.image_digits)
        if img is None:
            return
        stats = self.manager.reset_at(self.frame, img, update_geom_stats=self.mode in [VIEW_TRIANG, VIEW_DEPTH, VIEW_POSE])
        self.last_image = img
        self.selected_lidar_index = None
        self.cache = [(self.frame, copy.deepcopy(self.manager.tracks), self.manager.next_id, copy.deepcopy(stats))]
        self.cache_index = 0
        self._show(img, stats)

    def _restore_cache(self):
        frame, tracks, next_id, stats = self.cache[self.cache_index]
        self.frame = frame
        self.manager.tracks = copy.deepcopy(tracks)
        self.manager.next_id = next_id
        self.manager.current_frame = frame
        self.manager.last_stats = copy.deepcopy(stats)
        self.last_image = load_gray(self.cfg.img_dir, frame, self.cfg.image_digits)
        self.selected_lidar_index = None
        self.manager.prev_image = self.last_image.copy() if self.last_image is not None else None
        self.manager.prev_frame = frame

    def refresh(self):
        if self.last_image is not None:
            self._show(self.last_image, self.manager.last_stats)

    @staticmethod
    def _pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
        if img.shape[0] >= height:
            return img
        out = np.zeros((height, img.shape[1], img.shape[2]), dtype=img.dtype)
        out[: img.shape[0], :, :] = img
        return out

    def _map_window_click_to_canvas(self, window_name: str, display_shape: Optional[Tuple[int, int]], x: int, y: int) -> Optional[Tuple[float, float]]:
        """Map an OpenCV-window click to the displayed canvas coordinates.

        HighGUI backends disagree here. On many Windows builds the callback
        already reports canvas coordinates even when the window is resized. On
        others it reports scaled-window coordinates. Use one explicit mode from
        config so dense LiDAR clicks do not silently snap via a wrong mapping.
        """
        if display_shape is None:
            return float(x), float(y)
        disp_h, disp_w = display_shape
        mode = str(getattr(self.cfg, "mouse_coordinate_mode", "raw")).lower()

        if mode == "raw":
            cx, cy = float(x), float(y)
        elif mode == "resize":
            scale = max(float(self.cfg.resize), 1e-6)
            cx, cy = float(x) / scale, float(y) / scale
        elif mode == "window":
            try:
                _rx, _ry, rw, rh = cv2.getWindowImageRect(window_name)
            except cv2.error:
                rw, rh = disp_w, disp_h
            if rw <= 1 or rh <= 1:
                cx, cy = float(x), float(y)
            else:
                cx = float(x) * float(disp_w) / float(rw)
                cy = float(y) * float(disp_h) / float(rh)
        else:
            # Conservative auto: raw first because it is the observed behavior
            # on common Windows/OpenCV builds. Fall back only if raw is outside.
            cx, cy = float(x), float(y)
            if not (0.0 <= cx < disp_w and 0.0 <= cy < disp_h):
                scale = max(float(self.cfg.resize), 1e-6)
                cx, cy = float(x) / scale, float(y) / scale

        if not (0.0 <= cx < disp_w and 0.0 <= cy < disp_h):
            return None
        return cx, cy

    def _mouse_to_image_point(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        if self.last_image is None:
            return None
        h, w = self.last_image.shape[:2]
        mapped = self._map_window_click_to_canvas(WINDOW_NAME, self.last_display_shape, x, y)
        if mapped is None:
            return None
        ix, iy = mapped
        if 0.0 <= ix < w and 0.0 <= iy < h:
            return ix, iy
        return None

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.last_image is None:
            return
        mapped = self._mouse_to_image_point(x, y)
        if mapped is None:
            return
        active = self.manager.active_tracks(self.frame, confirmed_only=False)
        if not active:
            return
        pts = np.array([tr.last_pt() for tr in active], dtype=np.float32)
        ix, iy = mapped
        d = np.linalg.norm(pts - np.array([[ix, iy]], dtype=np.float32), axis=1)
        best_k = int(np.argmin(d))
        best_d = float(d[best_k])
        if best_k >= 0 and best_d <= self.cfg.inspect_radius_px:
            tr = active[best_k]
            self.selected_track_id = tr.id
            self.print_track_info(tr)
            self.refresh()
        else:
            print(f"[click] manager mapped=({ix:.1f},{iy:.1f}) nearestTrackDist={best_d:.1f}px mode={self.cfg.mouse_coordinate_mode}")

    def _mouse_to_lidar_point(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        if self.last_image is None:
            return None
        h, w = self.last_image.shape[:2]
        mapped = self._map_window_click_to_canvas(LIDAR_WINDOW_NAME, self.last_lidar_canvas_shape, x, y)
        if mapped is None:
            return None
        ix, iy = mapped
        if 0.0 <= ix < w and 0.0 <= iy < h:
            return ix, iy
        return None

    def _on_lidar_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.last_lidar_uv.size == 0:
            return
        mapped = self._mouse_to_lidar_point(x, y)
        if mapped is None:
            return
        ix, iy = mapped
        d = np.linalg.norm(self.last_lidar_uv - np.array([[ix, iy]], dtype=np.float64), axis=1)
        if d.size == 0:
            return
        best_i = int(np.argmin(d))
        best_d = float(d[best_i])
        click_radius = effective_lidar_click_radius(self.cfg)
        if best_i >= 0 and best_d <= click_radius:
            self.selected_lidar_index = best_i
            u, v = self.last_lidar_uv[best_i]
            z = self.last_lidar_z[best_i]
            print("\n================ LiDAR INSPECTION ================")
            print(f"frame={self.frame} idx={best_i} u={u:.2f} v={v:.2f} depthZ={z:.2f}m clickDist={best_d:.2f}px")
            print("==================================================\n")
            self.refresh()
        else:
            print(f"[click] lidar mapped=({ix:.1f},{iy:.1f}) nearestLidarDist={best_d:.1f}px radius={click_radius:.1f}px mode={self.cfg.mouse_coordinate_mode}")

    def print_track_info(self, tr: Track):
        tri = self.manager.evaluate_triang_candidates(self.frame, compute_dlt=True).get(tr.id)
        print("\n================ TRACK INSPECTION ================")
        print(f"id={tr.id} birth={tr.birth_frame} last={tr.last_seen_frame()} active={tr.active_in(self.frame)}")
        print(f"confirmed={tr.confirmed} source={tr.last_source} hits={tr.hit_count} misses={tr.miss_count} reqConfirm={tr.required_hits_to_confirm}")
        print(f"quality={tr.quality_score:.2f} reacqCount={tr.reacq_count} probation={tr.reacq_probation}")
        print(f"lastDesc={tr.last_desc_dist:.2f} anchorDesc={tr.last_anchor_desc_dist:.2f} ratio={tr.last_ratio:.3f} spatial={tr.last_spatial_err:.2f}px")
        print("obs frames:", [o.frame for o in tr.observations[-20:]])
        if tri:
            print(f"triang: method={tri.method} label={tri.label} past={tri.past_frame} gap={tri.gap} views={tri.inlier_views}/{tri.used_views} epi={tri.epi_px:.3f}px par={tri.parallax_deg:.3f}deg depth={tri.depth_m:.2f}m reproj={tri.reproj_px:.3f}px")
            if tri.method in ("refined_pair_dlt", "refined_multiview_dlt"):
                print(
                    f"refine: initZ={tri.init_depth_m:.2f}m refinedZ={tri.depth_m:.2f}m "
                    f"rmse {tri.rmse_before_px:.3f}->{tri.rmse_after_px:.3f}px "
                    f"curRes={tri.current_reproj_px:.3f}px maxRes={tri.max_residual_px:.3f}px worst={tri.worst_frame} "
                    f"shift={100.0 * tri.depth_shift_ratio:.1f}% success={tri.optimize_success} "
                    f"landmarkCand={tri.landmark_candidate}"
                )
        lidar_hit = nearest_point_within(self.last_lidar_uv, tr.last_pt(), self.cfg.lidar_radius_px)
        if lidar_hit is not None:
            li, ld = lidar_hit
            lz = float(self.last_lidar_z[li])
            extra = ""
            if tri and np.isfinite(tri.depth_m):
                extra = f" | tri-lidar {tri.depth_m - lz:+.2f}m ({abs(tri.depth_m - lz) / max(lz, 1e-12):.1%})"
            print(f"nearest LiDAR at track: idx={li} dist={ld:.2f}px depthZ={lz:.2f}m{extra}")
        elif self.lidar_projection is not None:
            print(f"nearest LiDAR at track: none within {self.cfg.lidar_radius_px:.1f}px")
        print("==================================================\n")

    def _show(self, img_gray: np.ndarray, stats: FrameStats):
        canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        mode = self.mode
        if self.lidar_projection is not None:
            self._refresh_lidar_data(img_gray.shape)
        triang_infos = self.manager.evaluate_triang_candidates(self.frame, compute_dlt=(mode in [VIEW_TRIANG, VIEW_DEPTH]), fast=False) if mode in [VIEW_TRIANG, VIEW_DEPTH] else {}
        pose_eval = self.manager.evaluate_pose_gap(self.frame) if mode == VIEW_POSE else PoseEval()
        lidar_depth_metrics = self._triangulated_lidar_metrics(triang_infos) if mode == VIEW_DEPTH else (0, np.nan, np.nan)
        self.last_pose_eval = pose_eval

        if self.show_grid or mode == VIEW_BUCKETS:
            self._draw_grid(canvas, mode == VIEW_BUCKETS)

        if mode == VIEW_BUCKETS:
            self._draw_bucket_counts(canvas)

        self._draw_tracks(canvas, mode, triang_infos, pose_eval)
        display_h = max(canvas.shape[0], self.cfg.side_panel_height)
        panel = self._make_side_panel(display_h, stats, mode, triang_infos, pose_eval, lidar_depth_metrics)
        out = np.hstack([self._pad_to_height(canvas, display_h), panel])
        self.last_display_shape = out.shape[:2]
        cv2.imshow(WINDOW_NAME, out)
        self._show_lidar_window(img_gray)

    def _refresh_lidar_data(self, image_shape):
        self.last_lidar_uv, self.last_lidar_z = self._load_lidar_projection_for_frame(image_shape)

    def _triangulated_lidar_metrics(self, triang_infos: Dict[int, TriangInfo]):
        """Compare current-frame valid sparse triangulated Z to nearby LiDAR Z."""
        if self.last_lidar_uv.size == 0:
            return 0, np.nan, np.nan
        points, depths = [], []
        for tr in self.manager.active_tracks(self.frame, confirmed_only=False):
            info = triang_infos.get(tr.id)
            if info is None or info.label not in GOOD_TRIANG_LABELS or not np.isfinite(info.depth_m):
                continue
            points.append(tr.last_pt())
            depths.append(float(info.depth_m))
        if not points:
            return 0, np.nan, np.nan
        matched, lidar_z, _dist = match_sparse_to_lidar_with_radius(
            np.asarray(points, dtype=np.float64), self.last_lidar_uv, self.last_lidar_z,
            self.cfg.lidar_radius_px, mode=self.cfg.lidar_match_mode,
        )
        pred = np.asarray(depths, dtype=np.float64)
        valid = matched & np.isfinite(lidar_z) & (lidar_z > 1e-9)
        if not np.any(valid):
            return 0, np.nan, np.nan
        rel = np.abs(pred[valid] - lidar_z[valid]) / lidar_z[valid]
        return int(rel.size), float(np.mean(rel)), float(np.median(rel))

    def _load_lidar_projection_for_frame(self, image_shape) -> Tuple[np.ndarray, np.ndarray]:
        if self.lidar_projection is None or self.cfg.velodyne_dir is None:
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
        digits = self.cfg.lidar_digits if self.cfg.lidar_digits is not None else self.cfg.image_digits
        lidar_path = self.cfg.velodyne_dir / f"{self.frame:0{digits}d}.bin"
        if not lidar_path.exists():
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
        T_cam0_velo, R_rect_4, P_rect = self.lidar_projection
        velo = load_velodyne_bin(lidar_path)
        uv, z, _xyz = project_velodyne_to_image(
            velo,
            T_cam0_velo,
            R_rect_4,
            P_rect,
            image_shape,
            min_depth_m=self.cfg.min_lidar_depth_m,
            max_depth_m=self.cfg.max_lidar_depth_m,
        )
        return uv, z

    def _show_lidar_window(self, img_gray: np.ndarray):
        if self.lidar_projection is None:
            return
        uv, z = self.last_lidar_uv, self.last_lidar_z
        canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        n = len(z)
        if n > 0:
            idx = choose_display_indices(n, self.cfg.max_lidar_vis_points)
            z_norm = np.clip(
                (z[idx] - self.cfg.min_lidar_depth_m)
                / max(1e-6, self.cfg.max_lidar_depth_m - self.cfg.min_lidar_depth_m),
                0.0,
                1.0,
            )
            color_vals = (255 * (1.0 - z_norm)).astype(np.uint8).reshape(-1, 1)
            colors = cv2.applyColorMap(color_vals, cv2.COLORMAP_JET).reshape(-1, 3)
            for p, col in zip(uv[idx], colors):
                x, y = int(round(p[0])), int(round(p[1]))
                cv2.circle(canvas, (x, y), 2, tuple(int(c) for c in col), -1, cv2.LINE_AA)
        if self.selected_lidar_index is not None and 0 <= self.selected_lidar_index < len(z):
            p = uv[self.selected_lidar_index]
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(canvas, (x, y), 7, CLR_MAGENTA, 2, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 2, CLR_WHITE, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"{z[self.selected_lidar_index]:.2f}m", (x + 8, max(14, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_MAGENTA, 1, cv2.LINE_AA)
        digits = self.cfg.lidar_digits if self.cfg.lidar_digits is not None else self.cfg.image_digits
        lidar_path = self.cfg.velodyne_dir / f"{self.frame:0{digits}d}.bin" if self.cfg.velodyne_dir is not None else None
        status = "ok" if n > 0 else "missing/no projected points"
        if lidar_path is not None and not lidar_path.exists():
            status = f"missing {lidar_path.name}"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(canvas, f"Projected LiDAR | frame {self.frame:0{self.cfg.image_digits}d} | points {n} | {status}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"click point for Z | scale {self.cfg.min_lidar_depth_m:.0f}-{self.cfg.max_lidar_depth_m:.0f}m", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_CYAN, 1, cv2.LINE_AA)
        self.last_lidar_canvas_shape = canvas.shape[:2]
        cv2.imshow(LIDAR_WINDOW_NAME, canvas)

    def _draw_grid(self, canvas: np.ndarray, color_by_occupancy: bool):
        h, w = canvas.shape[:2]
        counts, confirmed, triang = self.manager.bucket_counts(self.frame, canvas.shape[:2])
        for r in range(self.cfg.grid_rows):
            for c in range(self.cfg.grid_cols):
                x0 = int(c * w / self.cfg.grid_cols)
                x1 = int((c + 1) * w / self.cfg.grid_cols)
                y0 = int(r * h / self.cfg.grid_rows)
                y1 = int((r + 1) * h / self.cfg.grid_rows)
                if color_by_occupancy:
                    n = counts[r, c]
                    if n < self.cfg.target_per_bucket:
                        color = (60, 60, 255)
                    elif n > self.cfg.max_per_bucket:
                        color = (255, 120, 0)
                    else:
                        color = (80, 180, 80)
                else:
                    color = (70, 70, 70)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1)

    def _draw_bucket_counts(self, canvas: np.ndarray):
        h, w = canvas.shape[:2]
        counts, confirmed, triang = self.manager.bucket_counts(self.frame, canvas.shape[:2])
        for r in range(self.cfg.grid_rows):
            for c in range(self.cfg.grid_cols):
                x0 = int(c * w / self.cfg.grid_cols)
                y0 = int(r * h / self.cfg.grid_rows)
                n = int(counts[r, c])
                conf = int(confirmed[r, c])
                tri = int(triang[r, c])
                color = CLR_RED if n < self.cfg.target_per_bucket else CLR_GREEN
                txt = f"{n}/{conf}/{tri}"
                cv2.putText(canvas, txt, (x0 + 2, y0 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.31, color, 1, cv2.LINE_AA)

    def _track_color(self, tr: Track, mode: str, triang_infos: Dict[int, TriangInfo], pose_eval: PoseEval) -> Tuple[int, int, int]:
        if tr.id == self.selected_track_id:
            return CLR_MAGENTA
        if mode == VIEW_STATUS:
            if tr.newborn_last_step:
                return CLR_BLUE
            if tr.last_source == "lk" and tr.matched_last_step:
                return (255, 80, 255)
            if tr.reacquired_last_step or tr.reacq_probation:
                return CLR_CYAN
            if tr.confirmed:
                return CLR_GREEN
            return CLR_YELLOW
        if mode == VIEW_BUCKETS:
            return CLR_GREEN if tr.confirmed else CLR_YELLOW
        if mode == VIEW_AGE:
            val = min(1.0, tr.hit_count / 25.0)
            if val < 0.33:
                return (255, int(255 * val / 0.33), 0)
            if val < 0.66:
                t = (val - 0.33) / 0.33
                return (int(255 * (1 - t)), 255, int(255 * t))
            t = (val - 0.66) / 0.34
            return (0, int(255 * (1 - 0.5 * t)), 255)
        if mode == VIEW_TRIANG:
            info = triang_infos.get(tr.id)
            if info is None or info.label == "no_pair":
                return CLR_GRAY
            if info.label == "confirmed_good":
                return CLR_GREEN
            if info.label == "candidate_good":
                return CLR_DARK_GREEN
            if info.label == "reacq_good":
                return CLR_CYAN
            if info.label == "low_parallax":
                return CLR_ORANGE
            if info.label == "bad_epi":
                return CLR_RED
            if info.label in ("bad_reproj", "bad_depth"):
                return CLR_MAGENTA
        if mode == VIEW_DEPTH:
            info = triang_infos.get(tr.id)
            if info is None or info.label not in ("confirmed_good", "candidate_good", "reacq_good") or not np.isfinite(info.depth_m):
                return CLR_GRAY
            # Near = red/orange, far = blue-ish using a simple normalized scale.
            z = np.clip((info.depth_m - self.cfg.min_depth_m) / max(1e-6, self.cfg.max_depth_m - self.cfg.min_depth_m), 0, 1)
            col = cv2.applyColorMap(np.array([[int(255 * (1 - z))]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            return tuple(int(x) for x in col.tolist())
        if mode == VIEW_QUALITY:
            q = np.clip((tr.quality_score + 20.0) / 80.0, 0, 1)
            if q < 0.5:
                return (0, int(255 * q * 2), 255)
            return (0, 255, int(255 * (1 - q) * 2))
        if mode == VIEW_POSE:
            if tr.id in pose_eval.inlier_mask_by_track:
                return CLR_GREEN if pose_eval.inlier_mask_by_track[tr.id] else CLR_RED
            return CLR_GRAY
        return CLR_WHITE

    def _draw_tracks(self, canvas: np.ndarray, mode: str, triang_infos: Dict[int, TriangInfo], pose_eval: PoseEval):
        active = self.manager.select_tracks_for_display(
            self.frame,
            canvas.shape[:2],
            confirmed_only=False,
            max_tracks=self.draw_max_tracks,
        )
        for tr in active:
            color = self._track_color(tr, mode, triang_infos, pose_eval)
            pt = tr.last_pt()
            x, y = int(round(pt[0])), int(round(pt[1]))

            if self.show_paths and len(tr.observations) >= 2:
                obs = tr.observations[-self.cfg.path_len:] if self.cfg.path_len > 0 else tr.observations
                # Do not connect through missing frame gaps.
                segment: List[np.ndarray] = []
                prev_frame = None
                for o in obs:
                    if prev_frame is None or o.frame == prev_frame + 1:
                        segment.append(o.pt)
                    else:
                        if len(segment) >= 2:
                            pts_seg = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
                            cv2.polylines(canvas, [pts_seg], False, color, 1, cv2.LINE_AA)
                        segment = [o.pt]
                    prev_frame = o.frame
                if len(segment) >= 2:
                    pts_seg = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [pts_seg], False, color, 1, cv2.LINE_AA)

            if self.show_vectors and len(tr.observations) >= 2:
                prev = tr.observations[-2].pt
                cv2.arrowedLine(canvas, tuple(np.round(prev).astype(int)), (x, y), color, 1, cv2.LINE_AA, tipLength=0.25)

            rad = self.cfg.point_radius + (1 if tr.confirmed else 0)
            if tr.id == self.selected_track_id:
                rad += 3
            cv2.circle(canvas, (x, y), rad, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), rad + 1, CLR_BLACK, 1, cv2.LINE_AA)

        if self.show_lost:
            lost = [tr for tr in self.manager.tracks if (not tr.dead) and tr.miss_count > 0]
            lost.sort(key=lambda tr: (tr.miss_count, -tr.quality_score))
            for tr in lost[: min(len(lost), 700)]:
                pred = predict_track(tr, self.frame, self.cfg.prediction_mode)
                color = CLR_ORANGE if tr.confirmed else CLR_DARK_GRAY
                size = 5 + 3 * tr.miss_count
                draw_cross(canvas, pred, color, size=size, thickness=1)
                # Uncertainty circle grows with miss_count.
                cv2.circle(canvas, tuple(np.round(pred).astype(int)), int(self.cfg.reacq_search_radius_px * min(1.0, 0.25 * tr.miss_count)), color, 1, cv2.LINE_AA)

    def _draw_depth_colorbar(self, panel: np.ndarray, x: int, y: int, width: int, height: int):
        if width <= 0 or height <= 0:
            return
        grad = np.linspace(255, 0, width, dtype=np.uint8).reshape(1, width)
        bar = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
        bar = np.repeat(bar, height, axis=0)
        panel[y:y + height, x:x + width] = bar
        cv2.rectangle(panel, (x, y), (x + width, y + height), (90, 90, 90), 1)
        z_min = self.cfg.min_depth_m
        z_max = self.cfg.max_depth_m
        z_mid = 0.5 * (z_min + z_max)
        cv2.putText(panel, f"{z_min:.0f}m", (x, y + height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, CLR_CYAN, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{z_mid:.0f}m", (x + width // 2 - 14, y + height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, CLR_CYAN, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{z_max:.0f}m", (x + width - 34, y + height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, CLR_CYAN, 1, cv2.LINE_AA)

    def _make_side_panel(self, height: int, stats: FrameStats, mode: str, triang_infos: Dict[int, TriangInfo], pose_eval: PoseEval, lidar_depth_metrics=(0, np.nan, np.nan)) -> np.ndarray:
        width = self.cfg.side_panel_width
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (15, 15, 15), -1)
        y = 20
        title_lines = [
            f"SIFT+LK manager v1 | frame {self.frame}",
            f"mode {mode} | LK {'on' if self.cfg.lk_on else 'off'} | soft {'on' if self.cfg.soft_spawn else 'off'} | log {'on' if self.cfg.log_csv else 'off'}",
        ]
        put_text_lines(panel, title_lines, 8, y, CLR_WHITE, scale=0.48, dy=20, max_width_chars=62)
        y += 48

        lines = [
            f"Detect: primary {stats.primary_kp} | soft {stats.soft_kp}",
            f"LK: acc {stats.lk_accepted}/{stats.lk_attempted} | fbMed {stats.lk_median_fb_err:.2f}px | errMed {stats.lk_median_err:.1f}",
            f"LK reject: fwd {stats.lk_reject_forward} | bwd {stats.lk_reject_backward} | fb {stats.lk_reject_fb} | step {stats.lk_reject_step} | err {stats.lk_reject_error}",
            f"LK gates: epi {stats.lk_reject_epi} | flow {stats.lk_reject_flow} | desc {stats.lk_reject_desc}",
            f"LK desc refresh: {stats.lk_desc_refreshed} | medD {stats.lk_median_refresh_desc:.1f} | require {'yes' if self.cfg.lk_require_desc_refresh else 'no'}",
            f"Tracks: active {stats.active} | conf {stats.confirmed_active} | cand {stats.candidate_active}",
            f"Lost/dead: temp {stats.temp_lost} | dead {stats.dead_total} | maxMiss {self.cfg.max_misses}",
            f"Step: match {stats.matched} (lk {stats.lk_accepted}, pri {stats.matched_primary}, soft {stats.soft_rescued}) | reacq {stats.reacquired}",
            f"Miss/kill: miss {stats.missed} | kill {stats.killed}+dup{stats.dup_killed}",
            f"Spawn: primary {stats.spawned_primary} | soft {stats.spawned_soft}",
            f"Soft gates: rescue r{self.cfg.soft_rescue_search_radius_px:.0f}/ratio{self.cfg.soft_rescue_ratio:.2f}/D{self.cfg.soft_rescue_max_desc_dist:.0f}",
            f"Assoc: cand {stats.assoc_candidates} | medD {stats.median_desc:.1f} | medPx {stats.median_spatial:.1f}",
            f"Buckets: {self.cfg.grid_cols}x{self.cfg.grid_rows} target {self.cfg.target_per_bucket} max {self.cfg.max_per_bucket}",
            f"Bucket stats: cov {100*stats.bucket_coverage:.1f}% | under {stats.underfilled_buckets} | over {stats.overfilled_buckets}",
            f"Display selection: {stats.selected_display} tracks | perBucket {self.cfg.output_per_bucket}",
            f"Tri: good {stats.triang_good} (conf {stats.triang_confirmed_good}, cand {stats.triang_candidate_good}, reacq {stats.triang_reacq_good})",
            f"Tri rejects: lowPar {stats.triang_low_parallax} | epi {stats.triang_bad_epi} | reproj {stats.triang_bad_reproj} | depth {stats.triang_bad_depth}",
            f"Depth: valid {stats.depth_valid} | medZ {stats.median_depth:.1f}m | medRepr {stats.median_reproj:.2f}px",
            f"Tri-vs-LiDAR: n {lidar_depth_metrics[0]} | mean/med AbsRel {lidar_depth_metrics[1]:.3f}/{lidar_depth_metrics[2]:.3f}" if mode == VIEW_DEPTH else "Tri-vs-LiDAR: switch to DEPTH view for metrics",
            f"Pose gap {self.cfg.pose_eval_gap}: pairs {stats.pose_pairs} | inl {stats.pose_inliers} | rot {stats.pose_rot_err_deg:.2f} | t {stats.pose_t_err_deg:.2f}",
        ]
        put_text_lines(panel, lines, 8, y, CLR_WHITE, scale=0.39, dy=17, max_width_chars=70)
        y += 17 * len(lines) + 8

        if mode == VIEW_STATUS:
            legend = ["STATUS colors:", "purple=LK this frame", "green=confirmed, yellow=candidate", "blue=newborn, cyan=reacquired/probation", "orange/gray crosses=missed predictions"]
        elif mode == VIEW_BUCKETS:
            legend = ["BUCKET view:", "cell text active/confirmed/triang", "red=under target, green=ok, blue=overfull"]
        elif mode == VIEW_AGE:
            legend = ["AGE view:", "blue/cyan=newer, yellow/red=older/more hits", "paths do not connect across missed frames"]
        elif mode == VIEW_TRIANG:
            legend = ["TRIANG candidate view:", "bright green=confirmed+good", "dark green=candidate+good", "cyan=reacquired+good", "orange=low parallax, red=bad epipolar", "magenta=bad depth/reprojection"]
        elif mode == VIEW_DEPTH:
            legend = ["DEPTH preview view:", "uses GT pose + DLT for best current-past pair", "colors encode approximate depth", "gray=no valid triangulation"]
        elif mode == VIEW_QUALITY:
            legend = ["QUALITY view:", "green/yellow=higher score", "red/orange=lower score", "score uses hits/conf/misses/desc/spatial"]
        else:
            legend = ["POSE view:", "fixed gap current-vs-past Essential RANSAC", "green=RANSAC inlier, red=outlier", "gray=no observation pair for chosen gap"]
        put_text_lines(panel, legend, 8, y, CLR_CYAN, scale=0.39, dy=17, max_width_chars=68)
        y += 17 * len(legend) + 12

        if mode == VIEW_DEPTH:
            bar_h = 16
            block_h = 48
            if y + block_h <= height - 6:
                cv2.putText(panel, "depth color scale", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, CLR_CYAN, 1, cv2.LINE_AA)
                self._draw_depth_colorbar(panel, 8, y + 10, width - 24, bar_h)
                y += block_h

        # Selected track info compact. Keep it in flow so it does not cover the
        # dashboard/legend when the panel height changes.
        if self.selected_track_id is not None:
            tr = next((t for t in self.manager.tracks if t.id == self.selected_track_id), None)
            if tr is not None:
                box_h = 86
                if y + box_h <= height - 6:
                    y2 = y
                    cv2.rectangle(panel, (4, y2 - 12), (width - 6, y2 + box_h - 4), (35, 35, 35), -1)
                    sel = [
                        f"selected id {tr.id}: hits {tr.hit_count}, miss {tr.miss_count}, conf {tr.confirmed}, q {tr.quality_score:.1f}",
                        f"birth {tr.birth_frame}, last {tr.last_seen_frame()}, src {tr.last_source}, reacq {tr.reacq_count}",
                        f"desc {tr.last_desc_dist:.1f}, spat {tr.last_spatial_err:.1f}px, obs {len(tr.observations)}",
                        f"LK fb {tr.last_lk_fb_err:.2f}, err {tr.last_lk_err:.1f}, refreshD {tr.last_desc_refresh_dist:.1f}, stale {tr.frames_since_desc_refresh}",
                    ]
                    put_text_lines(panel, sel, 8, y2 + 5, CLR_MAGENTA, scale=0.36, dy=16, max_width_chars=70)

        return panel

    def save_screenshot(self):
        if self.last_image is None:
            return
        path = self.cfg.output_root / f"sift_manager_v3_frame{self.frame:06d}_{self.mode}.png"
        canvas = cv2.cvtColor(self.last_image, cv2.COLOR_GRAY2BGR)
        if self.lidar_projection is not None:
            self._refresh_lidar_data(self.last_image.shape)
        triang_infos = self.manager.evaluate_triang_candidates(self.frame, compute_dlt=(self.mode in [VIEW_TRIANG, VIEW_DEPTH]), fast=False) if self.mode in [VIEW_TRIANG, VIEW_DEPTH] else {}
        pose_eval = self.manager.evaluate_pose_gap(self.frame) if self.mode == VIEW_POSE else PoseEval()
        lidar_depth_metrics = self._triangulated_lidar_metrics(triang_infos) if self.mode == VIEW_DEPTH else (0, np.nan, np.nan)
        if self.show_grid or self.mode == VIEW_BUCKETS:
            self._draw_grid(canvas, self.mode == VIEW_BUCKETS)
        if self.mode == VIEW_BUCKETS:
            self._draw_bucket_counts(canvas)
        self._draw_tracks(canvas, self.mode, triang_infos, pose_eval)
        display_h = max(canvas.shape[0], self.cfg.side_panel_height)
        panel = self._make_side_panel(display_h, self.manager.last_stats, self.mode, triang_infos, pose_eval, lidar_depth_metrics)
        out = np.hstack([self._pad_to_height(canvas, display_h), panel])
        cv2.imwrite(str(path), out)
        print(f"Saved screenshot: {path}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> Config:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="TOML config file. Later config files and explicit CLI flags override earlier values.",
    )
    cfg_ns, remaining = config_parser.parse_known_args()

    p = argparse.ArgumentParser(
        description="Interactive hybrid SIFT+LK feature manager viewer for KITTI.",
        parents=[config_parser],
    )
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    p.add_argument("--poses", type=Path, default=DEFAULT_POSES_PATH)
    p.add_argument("--image-digits", type=int, default=DEFAULT_IMAGE_DIGITS)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--detector", choices=["shi", "sift", "xfeat"], default=None, help="Keypoint detector (new axis).")
    p.add_argument("--descriptor", choices=["sift", "xfeat"], default=None, help="Descriptor type (new axis); xfeat requires --detector xfeat.")
    p.add_argument("--detector-mode", choices=["sift_lk", "shi_sift_lk", "xfeat_sift_lk", "xfeat_native"], default=None, help="[legacy] bundled detector+descriptor; decomposed into --detector/--descriptor.")
    p.add_argument("--shi-max-corners", type=int, default=6000)
    p.add_argument("--shi-quality-level", type=float, default=0.005)
    p.add_argument("--shi-min-distance", type=float, default=7.0)
    p.add_argument("--shi-block-size", type=int, default=7)
    p.add_argument("--xfeat-top-k", type=int, default=4096, help="Max XFeat keypoints per frame (xfeat modes).")
    p.add_argument("--xfeat-detection-threshold", type=float, default=0.05, help="XFeat keypoint score NMS threshold.")
    p.add_argument("--xfeat-desc-size", type=float, default=7.0, help="SIFT descriptor patch size at each XFeat keypoint (xfeat_sift_lk).")
    p.add_argument("--xfeat-native-desc-scale", type=float, default=250.0, help="Scale applied to unit XFeat descriptors so SIFT-tuned distance gates apply (xfeat_native).")
    p.add_argument("--matcher", "--association-matcher", dest="matcher", choices=["radius_lowe", "xfeat_mnn"], default="radius_lowe", help="radius_lowe = per-track spatial+Lowe NN; xfeat_mnn = XFeat global mutual-NN + cosine.")
    p.add_argument("--xfeat-mnn-min-cossim", type=float, default=0.82, help="Cosine-similarity threshold for xfeat_mnn matches.")
    p.add_argument("--xfeat-mnn-max-step", type=float, default=0.0, help="Max displacement for an MNN match; 0 uses max_step_px.")
    p.add_argument("--velodyne-dir", type=Path, default=None, help="KITTI velodyne_points/data directory; enables synced LiDAR projection window.")
    p.add_argument("--lidar-digits", type=int, default=None, help="Digits in Velodyne .bin filenames; defaults to image_digits.")
    p.add_argument("--calib-velo-to-cam", type=Path, default=None, help="Raw KITTI calib_velo_to_cam.txt for LiDAR projection.")
    p.add_argument("--calib-cam-to-cam", type=Path, default=None, help="Raw KITTI calib_cam_to_cam.txt for rectified LiDAR projection.")
    p.add_argument("--camera", type=str, default="00", help="Raw KITTI camera id used for P_rect/R_rect, usually 00 for image_00.")
    p.add_argument("--lidar-radius-px", type=float, default=3.0, help="Evaluator sparse-to-LiDAR match radius; stored here for shared config consistency.")
    p.add_argument("--lidar-match-mode", choices=["nearest", "min_depth", "median_depth"], default="nearest", help="Evaluator sparse-to-LiDAR match mode; stored here for shared config consistency.")
    p.add_argument("--min-lidar-depth", type=float, default=1.0, help="Minimum projected LiDAR Z in meters.")
    p.add_argument("--max-lidar-depth", type=float, default=120.0, help="Maximum projected LiDAR Z in meters and viewer color scale top.")
    p.add_argument("--max-lidar-vis-points", type=int, default=12000, help="Maximum projected LiDAR points drawn in the viewer window.")
    p.add_argument("--lidar-click-radius", type=float, default=2.5, help="Click-selection radius for projected LiDAR points.")

    # Primary SIFT.
    p.add_argument("--sift-nfeatures", type=int, default=8000)
    p.add_argument("--sift-n-octave-layers", type=int, default=3)
    p.add_argument("--contrast", type=float, default=0.01)
    p.add_argument("--edge", type=float, default=10.0)
    p.add_argument("--sigma", type=float, default=1.6)

    # Soft detector / rescue / spawn.
    # --soft-spawn enables a second, more permissive SIFT detector. In v3, its
    # detections first try to rescue still-unmatched tracks, then unused soft
    # detections may spawn new tracks in under-filled buckets.
    p.add_argument("--soft-spawn", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable (--no-soft-spawn) softened SIFT detector for soft-rescue and bucketed spawning.")
    p.add_argument("--no-soft-rescue", dest="soft_rescue", action="store_false", default=True, help="Disable soft second-pass matching; soft detections will only be used for spawning.")
    p.add_argument("--soft-nfeatures", type=int, default=14000, help="Maximum keypoints requested from the softened detector.")
    p.add_argument("--soft-contrast", type=float, default=0.004, help="Lower contrast threshold for softened SIFT; lower gives more weak features.")
    p.add_argument("--soft-edge", type=float, default=15.0, help="Edge threshold for softened SIFT; larger admits more edge-like features.")
    p.add_argument("--soft-sigma", type=float, default=1.6, help="Initial Gaussian sigma for softened SIFT.")
    p.add_argument("--soft-confirm-hits", type=int, default=3, help="Hits needed to confirm soft-born tracks; default equals primary confirmation.")
    p.add_argument("--soft-rescue-search-radius", type=float, default=65.0, help="Search radius for soft second-pass rescue of tracks that primary matching missed.")
    p.add_argument("--soft-rescue-ratio", type=float, default=0.75, help="Ratio-test threshold for soft rescue; slightly stricter than normal matching.")
    p.add_argument("--soft-rescue-max-desc-dist", type=float, default=280.0, help="Max SIFT L2 descriptor distance for soft rescue.")
    p.add_argument("--soft-detect-mask-used-primary", type=float, default=0.0, help="Mask soft SIFT detection around primary detections that already won this frame; 0 disables detector masking.")
    p.add_argument("--soft-dup-used-primary", type=float, default=3.0, help="Reject soft-rescue detections within this px distance of already-used primary detections.")
    p.add_argument("--soft-spawn-primary-sep", type=float, default=4.0, help="Reject soft-spawn detections within this px distance of any primary detection.")
    p.add_argument("--soft-source-penalty", type=float, default=0.0, help="Optional quality-score penalty for observations/tracks sourced from soft detections; default 0.")

    # Association. These control how existing tracks are linked to current-frame
    # SIFT detections. Normal rules are used for visible tracks; reacq rules are
    # used for tracks that have already missed one or more frames.
    p.add_argument("--descriptor-mode", choices=["latest", "anchor"], default="latest", help="Descriptor used for matching: latest adapts, anchor resists drift.")
    p.add_argument("--prediction", choices=["constant_position", "constant_velocity"], default="constant_velocity", help="Image-plane prediction used to center the local search gate.")
    p.add_argument("--search-radius", type=float, default=55.0, help="Normal local association radius in pixels.")
    p.add_argument("--ratio", type=float, default=0.80, help="Normal Lowe ratio threshold; lower is stricter.")
    p.add_argument("--max-desc-dist", type=float, default=320.0, help="Normal maximum SIFT L2 descriptor distance.")
    p.add_argument("--reacq-search-radius", type=float, default=90.0, help="Larger radius for reacquiring tracks that missed previous frame(s).")
    p.add_argument("--reacq-ratio", type=float, default=0.70, help="Stricter ratio threshold for reacquisition.")
    p.add_argument("--reacq-max-desc-dist", type=float, default=240.0, help="Stricter max descriptor distance for reacquisition.")
    p.add_argument("--no-single-candidate", dest="allow_single_candidate", action="store_false", default=True)
    p.add_argument("--spatial-weight", type=float, default=0.20)
    p.add_argument("--miss-cost", type=float, default=30.0)
    p.add_argument("--max-step-px", type=float, default=250.0)

    # LK continuation. With --lk-tracking, previous-frame active tracks are first
    # propagated by pyramidal Lucas-Kanade optical flow. Descriptor matching then
    # rescues only the tracks LK failed to continue.
    p.add_argument("--lk-on", "--lk-tracking", dest="lk_on", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable (--no-lk-on) LK-first continuation before descriptor association.")
    p.add_argument("--lk-require-desc-refresh", action="store_true", help="Strict mode: LK points must also find a nearby acceptable SIFT descriptor refresh.")
    p.add_argument("--lk-win-size", type=int, default=21, help="LK window side length in pixels. Larger handles more motion but can smear across edges.")
    p.add_argument("--lk-max-level", type=int, default=3, help="Number of pyramid levels for LK. Higher helps larger motion.")
    p.add_argument("--lk-max-iter", type=int, default=30, help="Maximum LK iterations per pyramid level.")
    p.add_argument("--lk-eps", type=float, default=0.01, help="LK convergence epsilon.")
    p.add_argument("--lk-fb-thresh", type=float, default=1.5, help="Forward-backward consistency threshold in pixels.")
    p.add_argument("--lk-max-step-px", type=float, default=80.0, help="Reject LK tracks moving more than this many pixels in one frame.")
    p.add_argument("--lk-max-error", type=float, default=35.0, help="Reject LK tracks with OpenCV LK error above this value.")
    p.add_argument("--lk-min-eig-thresh", type=float, default=1e-4, help="LK minimum eigenvalue threshold; higher rejects weaker/aperture-prone points.")
    p.add_argument("--lk-desc-refresh-radius", type=float, default=4.0, help="Search radius for SIFT descriptor refresh around accepted LK point.")
    p.add_argument("--lk-desc-refresh-max-dist", type=float, default=320.0, help="Max SIFT descriptor distance accepted for LK descriptor refresh.")
    p.add_argument("--detection-period", "--lk-sift-period", dest="detection_period", type=int, default=1, help="Run the primary detector every N frames; 1 = every frame.")
    p.add_argument("--soft-sift-period", "--lk-soft-period", dest="soft_sift_period", type=int, default=1, help="Run soft SIFT every N primary-detection frames.")
    p.add_argument("--spawn-period", "--lk-spawn-period", dest="spawn_period", type=int, default=1, help="Spawn new tracks every N primary-detection frames.")
    p.add_argument("--force-detection-active-below", "--lk-force-sift-active-below", dest="force_detection_active_below", type=int, default=0, help="Force primary detection when active tracks drop below this count; 0 disables.")
    p.add_argument("--force-detection-coverage-below", "--lk-force-sift-bucket-coverage-below", dest="force_detection_coverage_below", type=float, default=0.0, help="Force primary detection when bucket coverage drops below this fraction; 0 disables.")
    p.add_argument("--force-soft-underfilled-buckets", "--lk-force-soft-underfilled-buckets", dest="force_soft_underfilled_buckets", type=int, default=0, help="Only run soft SIFT if at least this many buckets are underfilled; 0 disables.")
    p.add_argument("--lk-epipolar-thresh", type=float, default=0.0, help="Reject LK points with GT Sampson epipolar error above this px threshold; 0 disables.")
    p.add_argument("--lk-flow-consistency", action="store_true", help="Reject LK flow vectors that are local neighborhood outliers.")
    p.add_argument("--lk-flow-radius", type=float, default=45.0, help="Neighborhood radius in pixels for local LK flow consistency.")
    p.add_argument("--lk-flow-min-neighbors", type=int, default=5, help="Minimum local neighbors before flow consistency is tested.")
    p.add_argument("--lk-flow-mad-k", type=float, default=3.5, help="Robust MAD multiplier for local LK flow outlier rejection.")
    p.add_argument("--lk-flow-abs-thresh", type=float, default=8.0, help="Minimum absolute flow outlier threshold in pixels.")

    # Management.
    p.add_argument("--min-hits-confirm", type=int, default=3)
    p.add_argument("--max-misses", type=int, default=2)
    p.add_argument("--max-active-tracks", type=int, default=3000)
    p.add_argument("--no-duplicate-suppression", dest="duplicate_suppression", action="store_false", default=True)
    p.add_argument("--duplicate-dist", type=float, default=4.0)

    # Quality scoring. This affects duplicate suppression, display selection,
    # and depth-output selection, but it does not alter raw association gates.
    p.add_argument("--quality-confirmed-bonus", type=float, default=40.0)
    p.add_argument("--quality-hit-weight", type=float, default=1.5)
    p.add_argument("--quality-hit-cap", type=float, default=25.0)
    p.add_argument("--quality-miss-penalty", type=float, default=18.0)
    p.add_argument("--quality-reacq-penalty", type=float, default=8.0)
    p.add_argument("--quality-probation-penalty", type=float, default=8.0)
    p.add_argument("--quality-lk-bonus", type=float, default=3.0)
    p.add_argument("--quality-lk-stale-after", type=int, default=5)
    p.add_argument("--quality-lk-stale-penalty", type=float, default=0.8)
    p.add_argument("--quality-lk-stale-cap", type=float, default=10.0)
    p.add_argument("--quality-desc-penalty", type=float, default=0.03)
    p.add_argument("--quality-spatial-penalty", type=float, default=0.25)
    p.add_argument("--quality-active-bonus", type=float, default=10.0)

    # Buckets/spawn. The image is split into grid cells. After association,
    # unused detections are spawned only in cells below target occupancy.
    p.add_argument("--grid-cols", type=int, default=20, help="Number of bucket columns.")
    p.add_argument("--grid-rows", type=int, default=8, help="Number of bucket rows.")
    p.add_argument("--target-per-bucket", type=int, default=12, help="Spawn until each cell roughly reaches this many active tracks.")
    p.add_argument("--max-per-bucket", type=int, default=16, help="Overfull diagnostic threshold for bucket view.")
    p.add_argument("--output-per-bucket", "--depth-output-per-bucket", dest="output_per_bucket", type=int, default=12, help="Max quality-ranked depth-output tracks per cell.")
    p.add_argument("--output-max-tracks", "--depth-output-max-tracks", dest="output_max_tracks", type=int, default=2200, help="Global cap for depth-output track selection.")
    p.add_argument("--min-spawn-distance", type=float, default=7.0, help="Minimum distance from active tracks for creating a new spawned track.")
    p.add_argument("--spawn-count-confirmed-only", dest="spawn_count_candidates", action="store_false", default=True)

    # Geometry/triangulation diagnostics. GT geometry is used only to assess
    # whether current-past observations are good candidates for triangulation.
    p.add_argument("--triangulation-method", choices=["best_pair_dlt", "flow_depth_pair", "ttc_expansion", "ttc_expansion_norot", "refined_pair_dlt", "corrected_pair_dlt", "windowed_multiview_dlt", "refined_multiview_dlt", "hybrid_pair_multiview"], default="best_pair_dlt", help="Triangulation backend used by depth/triang views.")
    p.add_argument("--multiview-min-views", type=int, default=3, help="Minimum current+past observations for windowed multiview DLT.")
    p.add_argument("--hybrid-pair-min-parallax-deg", type=float, default=0.50, help="Stricter minimum parallax for hybrid_pair_multiview pair fallback.")
    p.add_argument("--hybrid-pair-min-baseline", type=float, default=0.5, help="Stricter minimum baseline for hybrid_pair_multiview pair fallback.")
    p.add_argument("--hybrid-pair-max-history", "--hybrid-pair-max-gap", dest="hybrid_pair_max_pair_history", type=int, default=3, help="Recent-frame search limit for hybrid_pair_multiview pair fallback.")
    p.add_argument("--hybrid-pair-reproj-thresh", type=float, default=1.5, help="Stricter reprojection threshold for hybrid_pair_multiview pair fallback.")
    p.add_argument("--refine-min-views", type=int, default=3, help="Minimum current+past observations for refined_multiview_dlt.")
    p.add_argument("--refine-max-iters", type=int, default=6, help="Maximum fixed-pose point-refinement iterations.")
    p.add_argument("--refine-huber-px", type=float, default=2.0, help="Huber robust-loss transition in pixels; 0 disables robust weighting.")
    p.add_argument("--refine-rmse-thresh", type=float, default=2.0, help="Maximum refined reprojection RMSE in pixels.")
    p.add_argument("--refine-max-depth-shift-ratio", type=float, default=0.5, help="Reject if refined depth moves this fraction away from DLT initialization.")
    p.add_argument("--current-reproj-thresh", type=float, default=2.0, help="Maximum reprojection error at the current-frame pixel for multiview depth.")
    p.add_argument("--max-reproj-thresh", type=float, default=3.0, help="Maximum single-view reprojection error allowed inside multiview fits.")
    p.add_argument("--gt-epi-thresh", type=float, default=1.0, help="GT Sampson epipolar threshold in pixels.")
    p.add_argument("--min-parallax-deg", type=float, default=0.10, help="Minimum ray parallax angle for a useful triangulation pair.")
    p.add_argument("--min-baseline", type=float, default=0.5, help="Minimum pose baseline in meters for candidate pairs.")
    p.add_argument("--max-pair-history", type=int, default=8, help="How many past frames to search for the best current-past pair.")
    p.add_argument("--triang-confirmed-only", action="store_true", default=False, help="Only evaluate confirmed tracks for triangulation; off by default.")
    p.add_argument("--triang-min-hits", type=int, default=2, help="Minimum observations/hits before a track can be triangulation-evaluated.")
    p.add_argument("--no-triang-reacquired", dest="triang_include_reacquired", action="store_false", default=True)
    p.add_argument("--min-depth", type=float, default=1.0)
    p.add_argument("--max-depth", type=float, default=120.0)
    p.add_argument("--reproj-thresh", type=float, default=2.0)


    # Pose eval.
    p.add_argument("--pose-eval-gap", type=int, default=2)
    p.add_argument("--pose-min-pairs", type=int, default=30)
    p.add_argument("--pose-ransac-thresh", type=float, default=0.75)
    p.add_argument("--pose-ransac-prob", type=float, default=0.999)
    p.add_argument("--pose-ransac-max-iters", type=int, default=2000)

    # Logging.
    p.add_argument("--no-log-csv", dest="log_csv", action="store_false", default=True)
    p.add_argument("--log-path", type=Path, default=None)

    # Display.
    p.add_argument("--draw-max-tracks", type=int, default=2200)
    p.add_argument("--point-radius", type=int, default=2)
    p.add_argument("--side-panel-width", type=int, default=400, help="Width of the black right-side dashboard panel; increase if text is clipped.")
    p.add_argument("--side-panel-height", type=int, default=550, help="Height of the dashboard canvas; image is padded, not stretched.")
    p.add_argument("--resize", type=float, default=1.2, help="OpenCV display scaling factor for the full canvas.")
    p.add_argument("--mouse-coordinate-mode", choices=["raw", "window", "resize", "auto"], default="raw", help="How OpenCV mouse callback coordinates are mapped to image/canvas pixels.")
    p.add_argument("--path-len", type=int, default=15)
    p.add_argument("--inspect-radius", type=float, default=12.0)

    p.set_defaults(**load_argparse_defaults(cfg_ns.config))
    a = p.parse_args(remaining)
    a.detector, a.descriptor = resolve_detector_descriptor(a.detector, a.descriptor, a.detector_mode)
    validate_feature_axes(a.detector, a.descriptor, a.matcher)
    if not a.lk_on and a.detection_period != 1:
        print("[note] LK tracking off -> forcing detection_period=1.")
        a.detection_period = 1
    return Config(
        img_dir=a.img_dir,
        calib_path=a.calib,
        poses_path=a.poses,
        velodyne_dir=a.velodyne_dir,
        calib_velo_to_cam=a.calib_velo_to_cam,
        calib_cam_to_cam=a.calib_cam_to_cam,
        camera=a.camera,
        image_digits=a.image_digits,
        lidar_digits=a.lidar_digits,
        start_frame=a.start,
        end_frame=a.end_frame,
        output_root=a.output_root,
        detector=a.detector,
        descriptor=a.descriptor,
        shi_max_corners=a.shi_max_corners,
        shi_quality_level=a.shi_quality_level,
        shi_min_distance_px=a.shi_min_distance,
        shi_block_size=a.shi_block_size,
        xfeat_top_k=a.xfeat_top_k,
        xfeat_detection_threshold=a.xfeat_detection_threshold,
        xfeat_desc_size_px=a.xfeat_desc_size,
        xfeat_native_desc_scale=a.xfeat_native_desc_scale,
        matcher=a.matcher,
        xfeat_mnn_min_cossim=a.xfeat_mnn_min_cossim,
        xfeat_mnn_max_step_px=a.xfeat_mnn_max_step,
        sift_nfeatures=a.sift_nfeatures,
        sift_n_octave_layers=a.sift_n_octave_layers,
        sift_contrast_threshold=a.contrast,
        sift_edge_threshold=a.edge,
        sift_sigma=a.sigma,
        soft_spawn=a.soft_spawn,
        soft_rescue=a.soft_rescue,
        soft_nfeatures=a.soft_nfeatures,
        soft_contrast_threshold=a.soft_contrast,
        soft_edge_threshold=a.soft_edge,
        soft_sigma=a.soft_sigma,
        soft_confirm_hits=a.soft_confirm_hits,
        soft_rescue_search_radius_px=a.soft_rescue_search_radius,
        soft_rescue_ratio=a.soft_rescue_ratio,
        soft_rescue_max_desc_dist=a.soft_rescue_max_desc_dist,
        soft_detect_mask_used_primary_px=a.soft_detect_mask_used_primary,
        soft_dup_used_primary_px=a.soft_dup_used_primary,
        soft_spawn_primary_sep_px=a.soft_spawn_primary_sep,
        soft_source_penalty=a.soft_source_penalty,
        descriptor_mode=a.descriptor_mode,
        prediction_mode=a.prediction,
        search_radius_px=a.search_radius,
        ratio=a.ratio,
        max_desc_dist=a.max_desc_dist,
        reacq_search_radius_px=a.reacq_search_radius,
        reacq_ratio=a.reacq_ratio,
        reacq_max_desc_dist=a.reacq_max_desc_dist,
        allow_single_candidate=a.allow_single_candidate,
        spatial_weight=a.spatial_weight,
        miss_cost=a.miss_cost,
        max_step_px=a.max_step_px,
        lk_on=a.lk_on,
        lk_require_desc_refresh=a.lk_require_desc_refresh,
        lk_win_size=a.lk_win_size,
        lk_max_level=a.lk_max_level,
        lk_max_iter=a.lk_max_iter,
        lk_eps=a.lk_eps,
        lk_fb_thresh_px=a.lk_fb_thresh,
        lk_max_step_px=a.lk_max_step_px,
        lk_max_error=a.lk_max_error,
        lk_min_eig_thresh=a.lk_min_eig_thresh,
        lk_desc_refresh_radius_px=a.lk_desc_refresh_radius,
        lk_desc_refresh_max_dist=a.lk_desc_refresh_max_dist,
        detection_period=a.detection_period,
        soft_sift_period=a.soft_sift_period,
        spawn_period=a.spawn_period,
        force_detection_active_below=a.force_detection_active_below,
        force_detection_coverage_below=a.force_detection_coverage_below,
        force_soft_underfilled_buckets=a.force_soft_underfilled_buckets,
        lk_epipolar_thresh_px=a.lk_epipolar_thresh,
        lk_flow_consistency=a.lk_flow_consistency,
        lk_flow_radius_px=a.lk_flow_radius,
        lk_flow_min_neighbors=a.lk_flow_min_neighbors,
        lk_flow_mad_k=a.lk_flow_mad_k,
        lk_flow_abs_thresh_px=a.lk_flow_abs_thresh,
        min_hits_to_confirm=a.min_hits_confirm,
        max_misses=a.max_misses,
        max_active_tracks=a.max_active_tracks,
        duplicate_suppression=a.duplicate_suppression,
        duplicate_dist_px=a.duplicate_dist,
        quality_confirmed_bonus=a.quality_confirmed_bonus,
        quality_hit_weight=a.quality_hit_weight,
        quality_hit_cap=a.quality_hit_cap,
        quality_miss_penalty=a.quality_miss_penalty,
        quality_reacq_penalty=a.quality_reacq_penalty,
        quality_probation_penalty=a.quality_probation_penalty,
        quality_lk_bonus=a.quality_lk_bonus,
        quality_lk_stale_after=a.quality_lk_stale_after,
        quality_lk_stale_penalty=a.quality_lk_stale_penalty,
        quality_lk_stale_cap=a.quality_lk_stale_cap,
        quality_desc_penalty=a.quality_desc_penalty,
        quality_spatial_penalty=a.quality_spatial_penalty,
        quality_active_bonus=a.quality_active_bonus,
        grid_cols=a.grid_cols,
        grid_rows=a.grid_rows,
        target_per_bucket=a.target_per_bucket,
        max_per_bucket=a.max_per_bucket,
        output_per_bucket=a.output_per_bucket,
        output_max_tracks=a.output_max_tracks,
        min_spawn_distance_px=a.min_spawn_distance,
        spawn_count_candidates=a.spawn_count_candidates,
        triangulation_method=a.triangulation_method,
        multiview_min_views=a.multiview_min_views,
        hybrid_pair_min_parallax_deg=a.hybrid_pair_min_parallax_deg,
        hybrid_pair_min_baseline_m=a.hybrid_pair_min_baseline,
        hybrid_pair_max_pair_history=a.hybrid_pair_max_pair_history,
        hybrid_pair_reproj_thresh_px=a.hybrid_pair_reproj_thresh,
        refine_min_views=a.refine_min_views,
        refine_max_iters=a.refine_max_iters,
        refine_huber_px=a.refine_huber_px,
        refine_rmse_thresh_px=a.refine_rmse_thresh,
        refine_max_depth_shift_ratio=a.refine_max_depth_shift_ratio,
        current_reproj_thresh_px=a.current_reproj_thresh,
        max_reproj_thresh_px=a.max_reproj_thresh,
        gt_epi_thresh_px=a.gt_epi_thresh,
        min_parallax_deg=a.min_parallax_deg,
        min_baseline_m=a.min_baseline,
        max_pair_history=a.max_pair_history,
        triang_confirmed_only=a.triang_confirmed_only,
        triang_min_hits=a.triang_min_hits,
        triang_include_reacquired=a.triang_include_reacquired,
        min_depth_m=a.min_depth,
        max_depth_m=a.max_depth,
        reproj_thresh_px=a.reproj_thresh,
        lidar_radius_px=a.lidar_radius_px,
        lidar_match_mode=a.lidar_match_mode,
        min_lidar_depth_m=a.min_lidar_depth,
        max_lidar_depth_m=a.max_lidar_depth,
        max_lidar_vis_points=a.max_lidar_vis_points,
        lidar_click_radius_px=a.lidar_click_radius,
        pose_eval_gap=a.pose_eval_gap,
        pose_min_pairs=a.pose_min_pairs,
        pose_ransac_thresh_px=a.pose_ransac_thresh,
        pose_ransac_prob=a.pose_ransac_prob,
        pose_ransac_max_iters=a.pose_ransac_max_iters,
        log_csv=a.log_csv,
        log_path=a.log_path,
        draw_max_tracks=a.draw_max_tracks,
        point_radius=a.point_radius,
        side_panel_width=a.side_panel_width,
        side_panel_height=a.side_panel_height,
        resize=a.resize,
        mouse_coordinate_mode=a.mouse_coordinate_mode,
        path_len=a.path_len,
        inspect_radius_px=a.inspect_radius,
    )


def print_controls():
    print("\nControls:")
    print("  n/d/right  : next frame")
    print("  b/a/left   : previous cached frame")
    print("  r          : reset manager at current frame")
    print("  R          : reset at original --start frame")
    print("  m          : cycle mode")
    print("  1..7       : status / buckets / age / triang / depth / quality / pose")
    print("  p          : toggle paths")
    print("  v          : toggle velocity arrows")
    print("  l          : toggle lost predicted crosses")
    print("  G          : toggle grid")
    print("  +/-        : change display track cap")
    print("  s          : save screenshot")
    print("  mouse left : inspect nearest active track in manager window")
    print("  mouse left : inspect nearest projected LiDAR point in LiDAR window")
    print("  q/ESC      : quit\n")


def main():
    cfg = parse_args()
    if not cfg.img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {cfg.img_dir}")
    if not cfg.calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {cfg.calib_path}")
    if not cfg.poses_path.exists():
        raise FileNotFoundError(f"Pose file not found: {cfg.poses_path}")
    K = load_kitti_K(cfg.calib_path)
    poses = load_kitti_poses(cfg.poses_path)
    print("\n================ SIFT+LK FEATURE MANAGER v1 ================")
    print("Image dir:", cfg.img_dir)
    print("Calib:", cfg.calib_path)
    print("Poses:", cfg.poses_path)
    print("K:\n", K)
    print("Start:", cfg.start_frame)
    print("Soft spawn:", cfg.soft_spawn)
    print("LK tracking:", cfg.lk_on)
    print("LK require descriptor refresh:", cfg.lk_require_desc_refresh)
    print("CSV log:", cfg.log_csv)
    print("Buckets:", cfg.grid_cols, "x", cfg.grid_rows, "target", cfg.target_per_bucket)
    print("Excluded for now: dynamic-object filtering and keyframe/direct-keyframe matching")
    print("=========================================================\n")
    viewer = Viewer(cfg, K, poses)
    viewer.run()


if __name__ == "__main__":
    main()
