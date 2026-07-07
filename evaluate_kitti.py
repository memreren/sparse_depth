#!/usr/bin/env python3
r"""
Simple sparse-depth evaluator for KITTI.

Runs the feature-manager sparse-depth pipeline over the frames named by a
``--config`` TOML file and reports how good the depths are against projected
Velodyne LiDAR. It is deliberately small: one per-frame CSV, one summary row,
and (optionally) one MP4 showing the colored sparse depth over the image. For
the full diagnostic evaluator (per-point dumps, error maps, many plots) see the
experimental branch.

Usage
-----
    # Metrics only
    python evaluate_kitti.py --config configs/default.toml --num-frames 250

    # Metrics + a depth video
    python evaluate_kitti.py --config configs/default.toml --num-frames 250 --save-video

Outputs (under --output-root)
-----------------------------
    per_frame.csv   one row per frame: frame, n_points, img_cov, gt_cov,
                    medRel, meanRel, delta20
    summary.csv     one row: medians/means of the above over the run
    depth.mp4       (only with --save-video) colored sparse depth over the frame
    config.json     the resolved run configuration
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from sparse_depth.cli_config import (
    add_manager_args,
    build_config_parser,
    make_manager_config,
)
from sparse_depth.config_io import load_argparse_defaults
from sparse_depth.feature_manager import FeatureManager
from sparse_depth.eval_metrics import (
    compute_depth_metrics,
    coverage_stats,
    finite_float,
    safe_mean,
    safe_median,
)
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

# Triangulation labels that count as a usable depth estimate.
GOOD_LABELS = {"confirmed_good", "candidate_good", "reacq_good"}


def json_safe(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    return v


def parse_args() -> argparse.Namespace:
    config_parser = build_config_parser()
    cfg_ns, remaining = config_parser.parse_known_args()

    p = argparse.ArgumentParser(
        description="Simple KITTI sparse-depth evaluator (metrics + optional video).",
        parents=[config_parser],
    )
    add_manager_args(p)
    # Run-specific options that are not part of the manager Config.
    p.add_argument("--num-frames", type=int, default=250,
                   help="Number of frames to process from --start.")
    p.add_argument("--all-active-depths", action="store_true",
                   help="Use every active good depth instead of the per-bucket selection.")
    p.add_argument("--save-video", action="store_true",
                   help="Write depth.mp4: colored sparse depth over the grayscale frame.")
    p.add_argument("--video-fps", type=float, default=10.0,
                   help="Frames per second for the depth video.")

    p.set_defaults(**load_argparse_defaults(cfg_ns.config))
    args = p.parse_args(remaining)
    if args.lidar_digits is None:
        args.lidar_digits = args.image_digits
    return args


# --------------------------------------------------------------------------- #
# LiDAR ground-truth projection
# --------------------------------------------------------------------------- #
def prepare_lidar_projection(args: argparse.Namespace):
    """Return (T_cam0_velo, R_rect_4, P_rect) for GT projection, or None."""
    if args.velodyne_dir is None:
        return None
    if not args.velodyne_dir.exists():
        print(f"[warn] Velodyne dir not found, LiDAR metrics disabled: {args.velodyne_dir}")
        return None
    try:
        if args.calib_velo_to_cam is not None and args.calib_cam_to_cam is not None:
            T_cam0_velo = load_raw_kitti_velo_to_cam(args.calib_velo_to_cam)
            P_rect, R_rect_4 = load_raw_kitti_cam_calib(args.calib_cam_to_cam, camera=args.camera)
        elif args.calib_velo_to_cam is not None:
            print("[warn] --calib-velo-to-cam without --calib-cam-to-cam; using P0 and identity rectification.")
            T_cam0_velo = load_raw_kitti_velo_to_cam(args.calib_velo_to_cam)
            data = read_kitti_calib_file(args.calib)
            P_rect = np.asarray(data["P0"], dtype=np.float64).reshape(3, 4)
            R_rect_4 = np.eye(4, dtype=np.float64)
        else:
            T_cam0_velo, R_rect_4, P_rect = load_odometry_lidar_projection(args.calib)
    except Exception as e:
        print(f"[warn] LiDAR projection unavailable: {e}")
        return None
    return T_cam0_velo, R_rect_4, P_rect


# --------------------------------------------------------------------------- #
# Per-frame sparse depth extraction
# --------------------------------------------------------------------------- #
def sparse_depth_points(manager, frame, image_shape, triang_infos, all_active_depths):
    """Return (uv[N,2], z[N]) for the accepted sparse depths at this frame."""
    if all_active_depths:
        tracks = manager.active_tracks(frame, confirmed_only=False)
    else:
        tracks = manager.select_tracks_for_display(frame, image_shape, confirmed_only=False)
    uv: List[np.ndarray] = []
    z: List[float] = []
    for tr in tracks:
        info = triang_infos.get(tr.id)
        if info is None or info.label not in GOOD_LABELS or not np.isfinite(info.depth_m):
            continue
        uv.append(tr.last_pt().astype(np.float64))
        z.append(float(info.depth_m))
    if not uv:
        return np.empty((0, 2), np.float64), np.empty((0,), np.float64)
    return np.asarray(uv, np.float64).reshape(-1, 2), np.asarray(z, np.float64)


def frame_metrics(args, lidar_projection, frame, image_shape, sparse_uv, sparse_z):
    """Compute the headline metrics for one frame.

    img_cov  : fraction of image grid cells occupied by sparse depth.
    gt_cov   : fraction of projected LiDAR points that have a sparse depth nearby.
    medRel/meanRel/delta20 : depth error vs matched LiDAR (raw, unscaled).
    """
    img_cov = coverage_stats("sparse", sparse_uv, image_shape,
                             args.grid_cols, args.grid_rows)["sparse_coverage"]
    out = {
        "n_points": int(len(sparse_z)),
        "img_cov": float(img_cov),
        "gt_cov": float("nan"),
        "medRel": float("nan"),
        "meanRel": float("nan"),
        "delta20": float("nan"),
    }
    if lidar_projection is None:
        return out, None

    lidar_path = args.velodyne_dir / f"{frame:0{args.lidar_digits}d}.bin"
    if not lidar_path.exists():
        return out, None

    T_cam0_velo, R_rect_4, P_rect = lidar_projection
    velo = load_velodyne_bin(lidar_path)
    lidar_uv, lidar_z, _ = project_velodyne_to_image(
        velo, T_cam0_velo, R_rect_4, P_rect,
        image_shape=image_shape,
        min_depth_m=args.min_lidar_depth,
        max_depth_m=args.max_lidar_depth,
    )
    if len(lidar_z) == 0:
        return out, None

    # Forward match: each sparse point -> nearest LiDAR depth within radius.
    matched, matched_z, _ = match_sparse_to_lidar_with_radius(
        sparse_uv, lidar_uv, lidar_z,
        radius_px=args.lidar_radius_px, mode=args.lidar_match_mode)
    # Reverse match: which LiDAR points are covered by a sparse estimate (gt_cov).
    lidar_covered, _, _ = match_sparse_to_lidar_with_radius(
        lidar_uv, sparse_uv, sparse_z,
        radius_px=args.lidar_radius_px, mode=args.lidar_match_mode)
    out["gt_cov"] = float(np.sum(lidar_covered) / max(len(lidar_z), 1))

    z_pred = sparse_z[matched]
    z_gt = matched_z[matched]
    m = compute_depth_metrics(z_pred, z_gt)
    out["medRel"] = m["median_rel_err"]
    out["meanRel"] = m["mean_rel_err"]
    out["delta20"] = m["delta_20"]
    return out, (matched, matched_z)


# --------------------------------------------------------------------------- #
# Depth video rendering
# --------------------------------------------------------------------------- #
_TURBO_LUT = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
).reshape(256, 3)


def _depth_colors(z: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    """Map depths to BGR colors (near = warm) via a fixed Turbo LUT."""
    t = np.clip((np.asarray(z) - zmin) / max(zmax - zmin, 1e-6), 0.0, 1.0)
    idx = (255 * (1.0 - t)).astype(np.int32)  # invert: near -> red end
    return _TURBO_LUT[idx]


def render_depth_frame(gray, sparse_uv, sparse_z, frame, zmin, zmax):
    """Grayscale frame with colored sparse-depth dots and a small colorbar."""
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = (0.55 * canvas).astype(np.uint8)  # dim so colored dots pop
    if len(sparse_z) > 0:
        colors = _depth_colors(sparse_z, zmin, zmax)
        for (u, v), c in zip(sparse_uv, colors):
            cv2.circle(canvas, (int(round(u)), int(round(v))), 3,
                       (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
    _draw_colorbar(canvas, zmin, zmax)
    cv2.putText(canvas, f"frame {frame}  n={len(sparse_z)}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _draw_colorbar(canvas, zmin, zmax):
    h, w = canvas.shape[:2]
    bar_h, bar_w, x0, y0 = 120, 14, w - 60, h - 140
    for i in range(bar_h):
        t = i / max(bar_h - 1, 1)
        c = _TURBO_LUT[int(255 * (1.0 - t))]  # top = near
        canvas[y0 + i, x0:x0 + bar_w] = c
    cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
    cv2.putText(canvas, f"{zmin:.0f}m", (x0 - 4, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{zmax:.0f}m", (x0 - 4, y0 + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    K = load_kitti_K(args.calib)
    poses = load_kitti_poses(args.poses)
    cfg = make_manager_config(args)
    manager = FeatureManager(cfg, K, poses)
    lidar_projection = prepare_lidar_projection(args)
    if args.pose_source == "estimated":
        manager.enable_estimated_pose(args.start)

    with open(args.output_root / "config.json", "w") as f:
        json.dump({k: json_safe(v) for k, v in vars(args).items()}, f, indent=2)

    end_exclusive = min(args.start + max(args.num_frames, 0), len(poses))
    if args.end_frame is not None:
        end_exclusive = min(end_exclusive, args.end_frame + 1)

    print("============== SPARSE DEPTH EVAL ==============")
    print("Images :", args.img_dir)
    print("Frames :", args.start, "to", end_exclusive - 1)
    print("LiDAR  :", args.velodyne_dir if lidar_projection is not None else "disabled/unavailable")
    print("Method :", args.triangulation_method, "| pose:", args.pose_source,
          "| detector:", args.detector, "| LK:", args.lk_on)
    print("Output :", args.output_root)
    print("Video  :", "on" if args.save_video else "off")
    print("===============================================")

    first_image = load_gray(args.img_dir, args.start, args.image_digits)
    if first_image is None:
        raise FileNotFoundError(f"Could not load start frame {args.start}")

    video: Optional[cv2.VideoWriter] = None
    if args.save_video:
        h, w = first_image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(str(args.output_root / "depth.mp4"), fourcc,
                                args.video_fps, (w, h))

    rows: List[dict] = []
    t_start = time.perf_counter()

    manager.reset_at(args.start, first_image, update_geom_stats=False)
    for idx, frame in enumerate(range(args.start, end_exclusive)):
        if idx == 0:
            image = first_image
        else:
            image = load_gray(args.img_dir, frame, args.image_digits)
            if image is None:
                print(f"[stop] could not load frame {frame:0{args.image_digits}d}")
                break
            manager.step(frame, image, update_geom_stats=False)

        triang_infos = manager.evaluate_triang_candidates(frame, compute_dlt=True, fast=False)
        sparse_uv, sparse_z = sparse_depth_points(
            manager, frame, image.shape, triang_infos, args.all_active_depths)
        m, _ = frame_metrics(args, lidar_projection, frame, image.shape, sparse_uv, sparse_z)
        m["frame"] = int(frame)
        rows.append(m)

        if video is not None:
            video.write(render_depth_frame(
                image, sparse_uv, sparse_z, frame,
                args.min_lidar_depth, args.max_lidar_depth))

        if idx % 25 == 0:
            print(f"  frame {frame:>5}  n={m['n_points']:>4}  "
                  f"img_cov={m['img_cov']:.2f}  gt_cov={finite_float(m['gt_cov']):.2f}  "
                  f"medRel={finite_float(m['medRel']):.3f}")

    if video is not None:
        video.release()

    elapsed = time.perf_counter() - t_start
    n_frames = max(len(rows), 1)

    # --- write per-frame CSV ---
    fields = ["frame", "n_points", "img_cov", "gt_cov", "medRel", "meanRel", "delta20"]
    with open(args.output_root / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    # --- write one-line summary CSV ---
    def col(name):
        return np.array([r[name] for r in rows], dtype=np.float64)

    pose = manager.pose_summary() if args.pose_source == "estimated" else {}
    summary = {
        "frames": len(rows),
        "triangulation_method": args.triangulation_method,
        "pose_source": args.pose_source,
        "median_n_points": safe_median(col("n_points")),
        "mean_img_cov": safe_mean(col("img_cov")),
        "mean_gt_cov": safe_mean(col("gt_cov")),
        "median_medRel": safe_median(col("medRel")),
        "mean_meanRel": safe_mean(col("meanRel")),
        "mean_delta20": safe_mean(col("delta20")),
        "ms_per_frame": 1000.0 * elapsed / n_frames,
        "pose_median_rot_err_deg": pose.get("pose_median_rot_err_deg", float("nan")),
        "pose_median_t_err_deg": pose.get("pose_median_t_err_deg", float("nan")),
        "pose_fallback_rate": pose.get("pose_fallback_rate", float("nan")),
    }
    with open(args.output_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print("=================== SUMMARY ===================")
    print(f"frames        : {summary['frames']}")
    print(f"median points : {summary['median_n_points']:.0f}")
    print(f"img_cov       : {summary['mean_img_cov']:.3f}")
    print(f"gt_cov        : {finite_float(summary['mean_gt_cov']):.3f}")
    print(f"median medRel : {finite_float(summary['median_medRel']):.3f}")
    print(f"mean meanRel  : {finite_float(summary['mean_meanRel']):.3f}")
    print(f"mean delta20  : {finite_float(summary['mean_delta20']):.3f}")
    print(f"ms/frame      : {summary['ms_per_frame']:.1f}")
    print(f"CSV           : {args.output_root / 'per_frame.csv'}")
    if args.save_video:
        print(f"Video         : {args.output_root / 'depth.mp4'}")
    print("===============================================")


if __name__ == "__main__":
    main()
