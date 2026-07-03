#!/usr/bin/env python3
"""Interactive GT-pose, fixed-height road-plane warp viewer for KITTI.

This intentionally precedes plane fitting: it makes the assumed plane visible
and gives a direct, conservative residual mask for inspecting where the model
is contradicted by cars, curbs, slopes, occlusions, and non-road regions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from sparse_depth.config_io import load_argparse_defaults
from sparse_depth.geometry import relative_pose
from sparse_depth.ground_plane import (
    level_ground_plane, plane_homography, ray_plane_depth, road_trapezoid_mask,
    local_zncc, robust_warp_residual, warp_source_to_target,
)
from sparse_depth.kitti_io import (
    load_gray, load_kitti_K, load_kitti_poses, load_odometry_lidar_projection,
    load_raw_kitti_cam_calib, load_raw_kitti_velo_to_cam, load_velodyne_bin,
    project_velodyne_to_image, read_kitti_calib_file,
)


WINDOW_NAME = "Ground-plane warp - KITTI"
LIDAR_WINDOW_NAME = "Projected LiDAR depth - ground plane"


@dataclass
class Config:
    img_dir: Path
    calib: Path
    poses: Path
    image_digits: int
    velodyne_dir: Optional[Path]
    lidar_digits: Optional[int]
    calib_velo_to_cam: Optional[Path]
    calib_cam_to_cam: Optional[Path]
    camera: str
    min_lidar_depth_m: float
    max_lidar_depth_m: float
    max_lidar_vis_points: int
    lidar_click_radius_px: float
    start: int
    end_frame: Optional[int]
    output_root: Path
    camera_height_m: float
    source_gap: int
    min_depth_m: float
    max_depth_m: float
    roi_top_y_frac: float
    roi_bottom_left_frac: float
    roi_bottom_right_frac: float
    roi_top_left_frac: float
    roi_top_right_frac: float
    residual_mad_scale: float
    patch_radius_px: int
    patch_min_std: float
    patch_zncc_thresh: float
    resize: float
    side_panel_width: int
    side_panel_height: int


def color_depth(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth >= lo) & (depth <= hi)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[valid] = np.clip(255 * (1.0 - (depth[valid] - lo) / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    out[~valid] = (25, 25, 25)
    return out


def draw_depth_scale(panel: np.ndarray, x: int, y: int, width: int, lo: float, hi: float) -> None:
    """Draw the exact JET mapping used by :func:`color_depth`."""
    height = 16
    # color_depth maps near depth to input 255 and far depth to input 0.
    grad = np.linspace(255, 0, width, dtype=np.uint8).reshape(1, width)
    bar = np.repeat(cv2.applyColorMap(grad, cv2.COLORMAP_JET), height, axis=0)
    panel[y:y + height, x:x + width] = bar
    cv2.rectangle(panel, (x, y), (x + width - 1, y + height - 1), (150, 150, 150), 1)
    mid = 0.5 * (lo + hi)
    labels = [(f"{lo:.0f}m near", x), (f"{mid:.1f}m", x + width // 2 - 16), (f"{hi:.0f}m far", x + width - 52)]
    for text, xx in labels:
        cv2.putText(panel, text, (xx, y + height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1, cv2.LINE_AA)


def prepare_lidar_projection(cfg: Config):
    """Prepare raw/odometry calibration once; return None when LiDAR is off."""
    if cfg.velodyne_dir is None:
        return None
    if not cfg.velodyne_dir.exists():
        print(f"[warn] LiDAR directory not found; LiDAR viewer disabled: {cfg.velodyne_dir}")
        return None
    try:
        if cfg.calib_velo_to_cam is not None and cfg.calib_cam_to_cam is not None:
            T_cam0_velo = load_raw_kitti_velo_to_cam(cfg.calib_velo_to_cam)
            P_rect, R_rect_4 = load_raw_kitti_cam_calib(cfg.calib_cam_to_cam, camera=cfg.camera)
        elif cfg.calib_velo_to_cam is not None:
            T_cam0_velo = load_raw_kitti_velo_to_cam(cfg.calib_velo_to_cam)
            P_rect = np.asarray(read_kitti_calib_file(cfg.calib)["P0"], dtype=np.float64).reshape(3, 4)
            R_rect_4 = np.eye(4, dtype=np.float64)
        else:
            T_cam0_velo, R_rect_4, P_rect = load_odometry_lidar_projection(cfg.calib)
        return T_cam0_velo, R_rect_4, P_rect
    except Exception as exc:
        print(f"[warn] LiDAR viewer disabled: {exc}")
        return None


def nearest_projected_point(uv: np.ndarray, query: np.ndarray, radius_px: float):
    if uv.size == 0:
        return None
    distances = np.linalg.norm(uv - np.asarray(query, dtype=np.float64).reshape(1, 2), axis=1)
    index = int(np.argmin(distances))
    return (index, float(distances[index])) if distances[index] <= radius_px else None


class Viewer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.K = load_kitti_K(cfg.calib)
        self.poses = load_kitti_poses(cfg.poses)
        self.frame = max(cfg.start, cfg.source_gap)
        self.mode = 0  # blend, raw residual, patch ZNCC, depth
        self.last = None
        self.last_canvas: Optional[np.ndarray] = None
        self.lidar_projection = prepare_lidar_projection(cfg)
        self.last_lidar_uv = np.empty((0, 2), dtype=np.float64)
        self.last_lidar_z = np.empty((0,), dtype=np.float64)
        self.selected_pixel: Optional[np.ndarray] = None
        self.selected_lidar_index: Optional[int] = None

    def compute(self):
        source_frame = self.frame - self.cfg.source_gap
        target = load_gray(self.cfg.img_dir, self.frame, self.cfg.image_digits)
        source = load_gray(self.cfg.img_dir, source_frame, self.cfg.image_digits)
        if target is None or source is None:
            raise FileNotFoundError(f"Missing target/source image for frames {self.frame}/{source_frame}")
        R_ts, t_ts = relative_pose(self.poses, self.frame, source_frame)
        plane = level_ground_plane(self.cfg.camera_height_m)
        H_ts = plane_homography(self.K, R_ts, t_ts, plane)
        warped, support = warp_source_to_target(source, H_ts)
        roi = road_trapezoid_mask(
            target.shape, self.cfg.roi_top_y_frac, self.cfg.roi_bottom_left_frac,
            self.cfg.roi_bottom_right_frac, self.cfg.roi_top_left_frac, self.cfg.roi_top_right_frac,
        )
        depth = ray_plane_depth(self.K, target.shape, plane)
        candidate = roi & np.isfinite(depth) & (depth >= self.cfg.min_depth_m) & (depth <= self.cfg.max_depth_m)
        residual, inliers, offset, median, threshold = robust_warp_residual(
            target, warped, support, candidate, self.cfg.residual_mad_scale,
        )
        zncc, std_target, std_warped = local_zncc(target, warped, self.cfg.patch_radius_px)
        informative = candidate & support & (std_target >= self.cfg.patch_min_std) & (std_warped >= self.cfg.patch_min_std)
        patch_inliers = informative & np.isfinite(zncc) & (zncc >= self.cfg.patch_zncc_thresh)
        self.last = (
            target, source, warped, roi, candidate, depth, residual, inliers, offset, median, threshold,
            zncc, std_target, std_warped, informative, patch_inliers, R_ts, t_ts,
        )
        self.last_lidar_uv, self.last_lidar_z = self._load_lidar(target.shape)
        self.selected_lidar_index = None

    def _load_lidar(self, image_shape):
        if self.lidar_projection is None or self.cfg.velodyne_dir is None:
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
        digits = self.cfg.lidar_digits if self.cfg.lidar_digits is not None else self.cfg.image_digits
        path = self.cfg.velodyne_dir / f"{self.frame:0{digits}d}.bin"
        if not path.exists():
            print(f"[warn] LiDAR frame missing: {path}")
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
        T_cam0_velo, R_rect_4, P_rect = self.lidar_projection
        velo = load_velodyne_bin(path)
        uv, z, _ = project_velodyne_to_image(
            velo, T_cam0_velo, R_rect_4, P_rect, image_shape,
            min_depth_m=self.cfg.min_lidar_depth_m, max_depth_m=self.cfg.max_lidar_depth_m,
        )
        return uv, z

    def draw(self):
        if self.last is None:
            self.compute()
        (target, _source, warped, roi, candidate, depth, residual, inliers, offset, median, threshold,
         zncc, _std_target, _std_warped, informative, patch_inliers, R, t) = self.last
        target_bgr = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
        warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        if self.mode == 0:
            view = cv2.addWeighted(target_bgr, 0.5, warped_bgr, 0.5, 0.0)
            title = "blend: target + plane-warped source"
        elif self.mode == 1:
            residual_vis = np.nan_to_num(residual, nan=0.0)
            scale = max(float(threshold) * 1.5 if np.isfinite(threshold) else 1.0, 1.0)
            view = cv2.applyColorMap(np.clip(residual_vis * 255.0 / scale, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            view[~candidate] = target_bgr[~candidate] // 3
            view[inliers] = (50, 180, 50)
            title = "residual: green = not contradicted by plane warp"
        elif self.mode == 2:
            # Gray = insufficient texture to evaluate; green/red = textured
            # patch agrees/disagrees with the plane warp according to ZNCC.
            view = target_bgr // 3
            view[candidate] = (80, 80, 80)
            view[informative & ~patch_inliers] = (30, 30, 230)
            view[patch_inliers] = (50, 180, 50)
            title = "patch ZNCC: gray=uninformative, green=agreement, red=disagreement"
        else:
            view = color_depth(depth, self.cfg.min_depth_m, self.cfg.max_depth_m)
            view[~candidate] = target_bgr[~candidate] // 3
            title = "metric plane depth; dark = outside trusted road support"
        contours, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (0, 220, 255), 1, cv2.LINE_AA)
        if self.selected_pixel is not None:
            cv2.drawMarker(view, tuple(self.selected_pixel.astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        image_h, w = view.shape[:2]
        panel_h = max(image_h, self.cfg.side_panel_height)
        panel = np.zeros((panel_h, self.cfg.side_panel_width, 3), dtype=np.uint8)
        baseline = float(np.linalg.norm(t))
        support_count = int(np.sum(candidate))
        inlier_count = int(np.sum(inliers))
        informative_count = int(np.sum(informative))
        patch_inlier_count = int(np.sum(patch_inliers))
        lines = [
            "GT pose + fixed road plane",
            f"target/source: {self.frame} / {self.frame - self.cfg.source_gap}",
            f"mode: {title}",
            f"height: {self.cfg.camera_height_m:.3f} m", f"baseline: {baseline:.3f} m",
            f"t_ts: [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]",
            f"ROI candidate pixels: {support_count}",
            f"warp-consistent: {inlier_count} ({inlier_count / max(support_count, 1):.1%})",
            f"brightness offset: {offset:+.1f}",
            f"residual median / threshold: {median:.1f} / {threshold:.1f}",
            f"patch: r={self.cfg.patch_radius_px}px std>={self.cfg.patch_min_std:.1f} | informative {informative_count} ({informative_count / max(support_count, 1):.1%})",
            f"patch ZNCC >= {self.cfg.patch_zncc_thresh:.2f}: {patch_inlier_count} ({patch_inlier_count / max(informative_count, 1):.1%} of informative)",
            f"LiDAR: {'off' if self.lidar_projection is None else f'{len(self.last_lidar_z)} projected points'}",
            "", "controls: n/b frame, m mode", "j/k height -/+ 2 cm", "s screenshot, q quit",
            "click main/LiDAR image for plane-vs-LiDAR inspection",
        ]
        y = 22
        for line in lines:
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1, cv2.LINE_AA)
            y += 20
        if self.mode == 3 and y + 36 < panel_h:
            cv2.putText(panel, "plane-depth scale (red=near, blue=far)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 220, 0), 1, cv2.LINE_AA)
            draw_depth_scale(panel, 10, y + 7, self.cfg.side_panel_width - 20, self.cfg.min_depth_m, self.cfg.max_depth_m)
        if panel_h > image_h:
            view = cv2.copyMakeBorder(view, 0, panel_h - image_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        out = np.hstack([view, panel])
        self.last_canvas = out
        cv2.imshow(WINDOW_NAME, out)

    def draw_lidar(self):
        if self.lidar_projection is None or self.last is None:
            return
        target = self.last[0]
        canvas = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
        n = len(self.last_lidar_z)
        if n:
            indices = np.arange(n) if n <= self.cfg.max_lidar_vis_points else np.linspace(0, n - 1, self.cfg.max_lidar_vis_points).astype(int)
            z = self.last_lidar_z[indices]
            value = np.clip(255 * (1.0 - (z - self.cfg.min_lidar_depth_m) / max(self.cfg.max_lidar_depth_m - self.cfg.min_lidar_depth_m, 1e-6)), 0, 255).astype(np.uint8)
            colors = cv2.applyColorMap(value.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
            for point, color in zip(self.last_lidar_uv[indices], colors):
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, tuple(int(v) for v in color), -1, cv2.LINE_AA)
        if self.selected_lidar_index is not None and self.selected_lidar_index < n:
            point = self.last_lidar_uv[self.selected_lidar_index]
            z = self.last_lidar_z[self.selected_lidar_index]
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 7, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{z:.2f} m", tuple(np.rint(point + [8, -8]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(canvas, f"Projected LiDAR | frame {self.frame} | points {n} | click point to compare plane depth", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"scale: {self.cfg.min_lidar_depth_m:.0f}m near (red) to {self.cfg.max_lidar_depth_m:.0f}m far (blue)", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 220, 0), 1, cv2.LINE_AA)
        cv2.imshow(LIDAR_WINDOW_NAME, canvas)

    def _inspect_pixel(self, point: np.ndarray):
        if self.last is None:
            return
        target, _source, _warped, _roi, candidate, depth, residual, inliers, offset, median, threshold, zncc, std_target, std_warped, informative, patch_inliers, *_ = self.last
        x, y = np.rint(point).astype(int)
        if not (0 <= x < target.shape[1] and 0 <= y < target.shape[0]):
            return
        self.selected_pixel = np.array([x, y])
        lidar = nearest_projected_point(self.last_lidar_uv, self.selected_pixel, self.cfg.lidar_click_radius_px)
        message = (
            f"[plane click] pixel=({x},{y}) planeZ={depth[y, x]:.2f}m candidate={bool(candidate[y, x])} "
            f"rawResidual={residual[y, x]:.1f} rawGreen={bool(inliers[y, x])} "
            f"patchStd(t/w)={std_target[y, x]:.1f}/{std_warped[y, x]:.1f} ZNCC={zncc[y, x]:.3f} "
            f"informative={bool(informative[y, x])} patchGreen={bool(patch_inliers[y, x])}"
        )
        if lidar is None:
            print(message + f" | LiDAR: none within {self.cfg.lidar_click_radius_px:.1f}px")
        else:
            index, distance = lidar
            lidar_z = self.last_lidar_z[index]
            if np.isfinite(depth[y, x]):
                delta = depth[y, x] - lidar_z
                comparison = f"plane-LiDAR={delta:+.2f}m ({delta / max(lidar_z, 1e-12):+.1%})"
            else:
                comparison = "plane depth unavailable (ray does not meet the assumed road plane ahead)"
            print(message + f" | LiDAR Z={lidar_z:.2f}m at {distance:.2f}px | {comparison}")
            self.selected_lidar_index = index

    def _on_main_mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or self.last is None:
            return
        h, w = self.last[0].shape[:2]
        if 0 <= x < w and 0 <= y < h:  # Ignore the dashboard panel.
            self._inspect_pixel(np.array([x, y]))

    def _on_lidar_mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        nearest = nearest_projected_point(self.last_lidar_uv, np.array([x, y]), self.cfg.lidar_click_radius_px)
        if nearest is None:
            print(f"[LiDAR click] no projected point within {self.cfg.lidar_click_radius_px:.1f}px")
            return
        index, distance = nearest
        self.selected_lidar_index = index
        point = self.last_lidar_uv[index]
        self._inspect_pixel(point)
        print(f"[LiDAR click] selected index={index} pixel=({point[0]:.1f},{point[1]:.1f}) click-distance={distance:.2f}px")

    def step(self, delta: int):
        candidate = self.frame + delta
        minimum = self.cfg.source_gap
        maximum = min(len(self.poses) - 1, self.cfg.end_frame) if self.cfg.end_frame is not None else len(self.poses) - 1
        if minimum <= candidate <= maximum:
            self.frame = candidate
            self.compute()

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
        if self.lidar_projection is not None:
            cv2.namedWindow(LIDAR_WINDOW_NAME, cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_KEEPRATIO", 0))
        self.compute()
        h, w = self.last[0].shape[:2]
        display_h = max(h, self.cfg.side_panel_height)
        cv2.resizeWindow(WINDOW_NAME, int((w + self.cfg.side_panel_width) * self.cfg.resize), int(display_h * self.cfg.resize))
        cv2.setMouseCallback(WINDOW_NAME, self._on_main_mouse)
        if self.lidar_projection is not None:
            cv2.resizeWindow(LIDAR_WINDOW_NAME, int(w * self.cfg.resize), int(h * self.cfg.resize))
            cv2.setMouseCallback(LIDAR_WINDOW_NAME, self._on_lidar_mouse)
        while True:
            self.draw()
            self.draw_lidar()
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("n"), ord("d"), 83): self.step(1)
            elif key in (ord("b"), ord("a"), 81): self.step(-1)
            elif key == ord("m"): self.mode = (self.mode + 1) % 4
            elif key == ord("j"):
                self.cfg.camera_height_m = max(0.1, self.cfg.camera_height_m - 0.02); self.compute()
            elif key == ord("k"):
                self.cfg.camera_height_m += 0.02; self.compute()
            elif key == ord("s"):
                self.cfg.output_root.mkdir(parents=True, exist_ok=True)
                path = self.cfg.output_root / f"ground_plane_{self.frame:0{self.cfg.image_digits}d}_mode{self.mode}.png"
                if self.last_canvas is not None:
                    cv2.imwrite(str(path), self.last_canvas)
                    print(f"saved {path}")
        cv2.destroyAllWindows()


def parse_args() -> Config:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, action="append", default=[])
    ns, remaining = pre.parse_known_args()
    p = argparse.ArgumentParser(description=__doc__, parents=[pre])
    p.add_argument("--img-dir", type=Path, default=Path("data/2011_09_30_drive_0016_sync/image_00/data"))
    p.add_argument("--calib", type=Path, default=Path("data/sequences/04/calib.txt"))
    p.add_argument("--poses", type=Path, default=Path("data/data_odometry_poses/dataset/poses/04.txt"))
    p.add_argument("--image-digits", type=int, default=10)
    p.add_argument("--velodyne-dir", type=Path, default=None)
    p.add_argument("--lidar-digits", type=int, default=None)
    p.add_argument("--calib-velo-to-cam", type=Path, default=None)
    p.add_argument("--calib-cam-to-cam", type=Path, default=None)
    p.add_argument("--camera", type=str, default="00")
    p.add_argument("--min-lidar-depth-m", type=float, default=1.0)
    p.add_argument("--max-lidar-depth-m", type=float, default=120.0)
    p.add_argument("--max-lidar-vis-points", type=int, default=12000)
    p.add_argument("--lidar-click-radius-px", type=float, default=5.0)
    p.add_argument("--start", type=int, default=25)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--output-root", type=Path, default=Path("outputs/ground_plane_seq04"))
    p.add_argument("--camera-height-m", type=float, default=1.65)
    p.add_argument("--source-gap", type=int, default=1)
    p.add_argument("--min-depth-m", type=float, default=2.0)
    p.add_argument("--max-depth-m", type=float, default=45.0)
    p.add_argument("--roi-top-y-frac", type=float, default=0.54)
    p.add_argument("--roi-bottom-left-frac", type=float, default=0.08)
    p.add_argument("--roi-bottom-right-frac", type=float, default=0.92)
    p.add_argument("--roi-top-left-frac", type=float, default=0.36)
    p.add_argument("--roi-top-right-frac", type=float, default=0.64)
    p.add_argument("--residual-mad-scale", type=float, default=3.5)
    p.add_argument("--patch-radius-px", type=int, default=3, help="ZNCC half-window radius; 3 means 7x7 patches.")
    p.add_argument("--patch-min-std", type=float, default=5.0, help="Minimum target and warped patch grayscale stddev to call a patch informative.")
    p.add_argument("--patch-zncc-thresh", type=float, default=0.75, help="Minimum local ZNCC for a textured patch to agree with the plane warp.")
    p.add_argument("--resize", type=float, default=1.1)
    p.add_argument("--side-panel-width", type=int, default=380)
    p.add_argument("--side-panel-height", type=int, default=520, help="Minimum dashboard height; image is padded beside a taller panel.")
    p.set_defaults(**load_argparse_defaults(ns.config))
    a = p.parse_args(remaining)
    values = vars(a)
    values.pop("config", None)
    return Config(**values)


if __name__ == "__main__":
    Viewer(parse_args()).run()
