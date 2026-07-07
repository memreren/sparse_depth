#!/usr/bin/env python3
r"""
ground_plane_calibrate_kitti.py

Standalone calibrator for the constant camera-vs-road plane offset.

Over a frame range it RANSAC-fits the road plane to camera-frame LiDAR points
inside the road ROI (n*(i), h*(i)), averages them -- cancelling the zero-mean
suspension / grade / bank so the constant offset (mounting + rectification, plus
any systematic geometry the ROI samples) survives -- and writes a small JSON
calibration consumed by the evaluator and the interactive viewer.

Calibrate over the SAME ROI you will use the plane on, and prefer a route with
varied grade and a laterally centred ROI so road camber cancels left/right.

Usage:
  python ground_plane_calibrate_kitti.py --config configs/default.toml \
      --config configs/plane_homography_seq04.toml --start 20 --num-frames 150 \
      --output configs/ground_calib_seq04.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sparse_depth.config_io import load_argparse_defaults
from sparse_depth.ground_calibration import GroundCalibration, select_road_points
from sparse_depth.ground_plane import fit_plane_ransac, road_trapezoid_mask
from sparse_depth.kitti_io import (
    load_gray, load_kitti_poses, load_raw_kitti_cam_calib,
    load_raw_kitti_velo_to_cam, load_velodyne_bin, project_velodyne_to_image,
)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, action="append", default=[])
    ns, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description=__doc__, parents=[pre],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-dir", type=Path)
    p.add_argument("--calib", type=Path)
    p.add_argument("--poses", type=Path)
    p.add_argument("--image-digits", type=int, default=6)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=150)
    p.add_argument("--velodyne-dir", type=Path, default=None)
    p.add_argument("--lidar-digits", type=int, default=None)
    p.add_argument("--calib-velo-to-cam", type=Path, default=None)
    p.add_argument("--calib-cam-to-cam", type=Path, default=None)
    p.add_argument("--camera", type=str, default="00")
    p.add_argument("--output", type=Path, required=True, help="JSON calibration output path.")

    p.add_argument("--min-depth-m", dest="min_depth_m", type=float, default=2.0)
    p.add_argument("--max-depth-m", dest="max_depth_m", type=float, default=45.0)
    p.add_argument("--lidar-plane-thresh-m", dest="lidar_plane_thresh_m", type=float, default=0.06)
    p.add_argument("--lidar-plane-min-inliers", dest="lidar_plane_min_inliers", type=int, default=40)
    p.add_argument("--homography-roi-top-y-frac", dest="homography_roi_top_y_frac", type=float, default=0.58)
    p.add_argument("--homography-roi-bottom-left-frac", dest="homography_roi_bottom_left_frac", type=float, default=0.02)
    p.add_argument("--homography-roi-bottom-right-frac", dest="homography_roi_bottom_right_frac", type=float, default=0.98)
    p.add_argument("--homography-roi-top-left-frac", dest="homography_roi_top_left_frac", type=float, default=0.25)
    p.add_argument("--homography-roi-top-right-frac", dest="homography_roi_top_right_frac", type=float, default=0.75)

    known = {a.dest for a in p._actions}
    values = load_argparse_defaults(ns.config)
    p.set_defaults(**{k: v for k, v in values.items() if k in known})
    args = p.parse_args()
    for req in ("img_dir", "calib", "poses"):
        if getattr(args, req) is None:
            p.error(f"--{req.replace('_', '-')} is required (flag or config file)")
    if args.velodyne_dir is None or args.calib_velo_to_cam is None or args.calib_cam_to_cam is None:
        p.error("LiDAR paths (velodyne_dir, calib_velo_to_cam, calib_cam_to_cam) are required.")
    return args


def main():
    args = parse_args()
    poses = load_kitti_poses(args.poses)
    T_cv = load_raw_kitti_velo_to_cam(args.calib_velo_to_cam)
    P_rect, R_rect = load_raw_kitti_cam_calib(args.calib_cam_to_cam, camera=args.camera)
    lidar = (T_cv, R_rect, P_rect)

    first = load_gray(args.img_dir, args.start, args.image_digits)
    if first is None:
        raise SystemExit(f"cannot load start frame {args.start}")
    roi = road_trapezoid_mask(first.shape, args.homography_roi_top_y_frac,
                              args.homography_roi_bottom_left_frac, args.homography_roi_bottom_right_frac,
                              args.homography_roi_top_left_frac, args.homography_roi_top_right_frac)
    digits = args.lidar_digits if args.lidar_digits is not None else args.image_digits
    end = min(args.start + args.num_frames, len(poses)) - 1

    normals, heights, rmss, n_fit = [], [], [], 0
    for f in range(args.start, end + 1):
        img = load_gray(args.img_dir, f, args.image_digits)
        path = Path(args.velodyne_dir) / f"{f:0{digits}d}.bin"
        if img is None or not path.exists():
            continue
        uv, z, xyz = project_velodyne_to_image(load_velodyne_bin(path), *lidar, img.shape,
                                               min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m)
        if uv.size == 0:
            continue
        keep = select_road_points(uv, z, roi, img.shape, args.min_depth_m, args.max_depth_m)
        plane, inliers, rms = fit_plane_ransac(xyz[keep], thresh_m=args.lidar_plane_thresh_m,
                                               min_inliers=args.lidar_plane_min_inliers)
        if plane is None:
            continue
        normals.append(plane.normal)
        heights.append(-plane.offset)
        rmss.append(rms)
        n_fit += 1

    if not normals:
        raise SystemExit("no successful LiDAR plane fits; check ROI / depth band / paths.")

    meta = {
        "img_dir": str(args.img_dir),
        "start": int(args.start), "num_frames": int(args.num_frames),
        "roi": {"top_y": args.homography_roi_top_y_frac,
                "bottom": [args.homography_roi_bottom_left_frac, args.homography_roi_bottom_right_frac],
                "top": [args.homography_roi_top_left_frac, args.homography_roi_top_right_frac]},
        "depth_band_m": [args.min_depth_m, args.max_depth_m],
        "mean_fit_rms_m": float(np.mean(rmss)),
    }
    calib = GroundCalibration.from_normals(normals, heights, meta)
    calib.save(args.output)

    print("===== GROUND-PLANE CALIBRATION =====")
    print(f"frames fit       : {n_fit}")
    print(f"normal (mean n*) : [{calib.normal[0]:+.4f}, {calib.normal[1]:+.4f}, {calib.normal[2]:+.4f}]")
    print(f"constant offset  : pitch {calib.pitch_deg:+.3f} deg | roll {calib.roll_deg:+.3f} deg | angle-to-down {calib.angle_down_deg:.3f} deg")
    print(f"per-frame spread : pitch std {calib.meta['pitch_std_deg']:.3f} | roll std {calib.meta['roll_std_deg']:.3f} deg (local grade/bank + suspension)")
    print(f"height           : {calib.height_m:.3f} m (std {calib.meta['height_std_m']:.3f})")
    print(f"mean fit RMS     : {calib.meta['mean_fit_rms_m']:.3f} m")
    print(f"saved            : {args.output}")


if __name__ == "__main__":
    main()
