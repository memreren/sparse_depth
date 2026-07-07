#!/usr/bin/env python3
r"""
depth_filter_eval_kitti.py

Headless evaluator for the keyframe inverse-depth filter (sparse_depth.depth_filter).
Runs mapper.process() over a frame range and scores ALL visible depth points
(persistent map + current keyframe) against projected LiDAR each frame, with
the same headline metrics as the feature-pipeline ablation harness.

Usage:
  python depth_filter_eval_kitti.py --config configs/default.toml --start 100 --num-frames 100
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from sparse_depth.config_io import load_argparse_defaults
from sparse_depth.depth_filter import DepthFilterConfig, DepthFilterMapper
from sparse_depth.eval_metrics import coverage_stats, safe_mean, safe_median
from sparse_depth.kitti_io import (
    load_gray,
    load_kitti_K,
    load_kitti_poses,
    load_raw_kitti_cam_calib,
    load_raw_kitti_velo_to_cam,
    load_velodyne_bin,
    match_sparse_to_lidar_with_radius,
    project_velodyne_to_image,
    read_kitti_calib_file,
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
    p.add_argument("--num-frames", type=int, default=100)
    p.add_argument("--velodyne-dir", type=Path, default=None)
    p.add_argument("--lidar-digits", type=int, default=None)
    p.add_argument("--calib-velo-to-cam", type=Path, default=None)
    p.add_argument("--calib-cam-to-cam", type=Path, default=None)
    p.add_argument("--camera", type=str, default="00")
    p.add_argument("--lidar-radius-px", type=float, default=3.0)
    p.add_argument("--output-root", type=Path, default=None,
                   help="Optional dir for per_frame_metrics.csv")

    p.add_argument("--seed-mode", choices=["gradient", "corners"], default="gradient")
    p.add_argument("--max-seeds", type=int, default=2500)
    p.add_argument("--grad-thresh", type=float, default=20.0)
    p.add_argument("--converge-ratio", type=float, default=0.10)
    p.add_argument("--converge-min-obs", type=int, default=4)
    p.add_argument("--zncc-min", type=float, default=0.85)
    p.add_argument("--validate-map", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--beta-inlier", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--photometric-sigma", action=argparse.BooleanOptionalAction, default=True)
    # deliberately NOT named min-parallax-deg: the shared TOML has an unrelated
    # triangulation key of that name which would silently override this default.
    p.add_argument("--seed-min-parallax", type=float, default=1.0,
                   help="Min accumulated parallax (deg) before a seed may be promoted; 0 disables.")

    known = {a.dest for a in p._actions}
    values = load_argparse_defaults(ns.config)
    p.set_defaults(**{k: v for k, v in values.items() if k in known})
    args = p.parse_args()
    for req in ("img_dir", "calib", "poses"):
        if getattr(args, req) is None:
            p.error(f"--{req.replace('_', '-')} is required (flag or config file)")
    return args


def main():
    args = parse_args()
    K = load_kitti_K(args.calib)
    poses = load_kitti_poses(args.poses)

    lidar = None
    if args.velodyne_dir is not None and Path(args.velodyne_dir).exists():
        if args.calib_velo_to_cam is not None and args.calib_cam_to_cam is not None:
            T_cv = load_raw_kitti_velo_to_cam(args.calib_velo_to_cam)
            P_rect, R_rect = load_raw_kitti_cam_calib(args.calib_cam_to_cam, camera=args.camera)
            lidar = (T_cv, R_rect, P_rect)
    if lidar is None:
        print("[warn] no LiDAR; only counts/coverage will be reported")

    cfg = DepthFilterConfig(
        seed_mode=args.seed_mode,
        max_seeds=args.max_seeds,
        grad_thresh=args.grad_thresh,
        converge_sigma_ratio=args.converge_ratio,
        converge_min_obs=args.converge_min_obs,
        zncc_min=args.zncc_min,
        validate_map=args.validate_map,
        use_beta_inlier=args.beta_inlier,
        photometric_sigma=args.photometric_sigma,
        min_parallax_deg=args.seed_min_parallax,
    )
    mapper = DepthFilterMapper(K, poses, cfg)

    frame = args.start
    img = load_gray(args.img_dir, frame, args.image_digits)
    if img is None:
        raise SystemExit(f"cannot load frame {frame}")
    n0, _, _ = mapper.set_reference(frame, img)
    end = min(args.start + args.num_frames, len(poses)) - 1
    print(f"depth-filter eval: frames {args.start}..{end}, {n0} seeds "
          f"({args.seed_mode}), validate_map={args.validate_map} "
          f"beta={args.beta_inlier} photSigma={args.photometric_sigma}")

    digits = args.lidar_digits if args.lidar_digits is not None else args.image_digits
    rows = []
    t_all = time.time()
    kfs = 0
    for f in range(args.start + 1, end + 1):
        img = load_gray(args.img_dir, f, args.image_digits)
        if img is None:
            break
        t0 = time.time()
        st = mapper.process(f, img)
        ms = (time.time() - t0) * 1000
        kfs += int(st.kf_switched)

        uv, z, sig, _kf = mapper.visible_depth_points(f, img.shape)
        row = {"frame": f, "n_pts": len(uv), "map_size": st.map_size, "ms": ms,
               "kf": int(st.kf_switched)}
        row.update({k: v for k, v in coverage_stats("img", uv, img.shape, 20, 8).items()
                    if k == "img_coverage"})
        if lidar is not None:
            path = Path(args.velodyne_dir) / f"{f:0{digits}d}.bin"
            if path.exists():
                pts = load_velodyne_bin(path)
                luv, lz, _ = project_velodyne_to_image(pts, *lidar, img.shape)
                mt, zl, _ = match_sparse_to_lidar_with_radius(uv, luv, lz, args.lidar_radius_px)
                if np.any(mt):
                    diff = z[mt] - zl[mt]
                    rel = np.abs(diff) / zl[mt]
                    row.update(n_matched=int(mt.sum()),
                               med_rel=float(np.median(rel)), mean_rel=float(np.mean(rel)),
                               med_abs=float(np.median(np.abs(diff))),
                               mean_abs=float(np.mean(np.abs(diff))),
                               d20=float(np.mean(rel < 0.2)))
                    if np.max(rel) > 2.0:
                        k = int(np.argmax(rel))
                        gi = np.where(mt)[0][k]
                        print(f"  [monster] f{f} rel={rel[k]:.1f} z_est={z[gi]:.1f} "
                              f"z_lidar={zl[gi]:.1f} uv=({uv[gi,0]:.0f},{uv[gi,1]:.0f}) "
                              f"sig={sig[gi]:.3f}")
                # gt coverage: fraction of projected LiDAR covered by an estimate
                rev, _, _ = match_sparse_to_lidar_with_radius(luv, uv, z, args.lidar_radius_px)
                row["gt_coverage"] = float(np.mean(rev)) if rev.size else 0.0
        rows.append(row)

    def agg(key, fn=safe_median):
        return fn([r.get(key, float("nan")) for r in rows])

    print(f"\n===== depth-filter summary ({len(rows)} frames, {kfs} keyframes, "
          f"{time.time()-t_all:.0f}s total) =====")
    print(f"n_pts    med {agg('n_pts'):6.0f}   mean {agg('n_pts', safe_mean):6.0f}")
    print(f"map size final {rows[-1]['map_size']}")
    print(f"img_cov  med {agg('img_coverage'):.3f}")
    print(f"gt_cov   med {agg('gt_coverage'):.3f}")
    print(f"medRel   med {agg('med_rel'):.4f}   meanRel mean {agg('mean_rel', safe_mean):.4f}")
    print(f"medAbs   med {agg('med_abs'):.3f}m  meanAbs mean {agg('mean_abs', safe_mean):.3f}m")
    print(f"d<20%    med {agg('d20'):.3f}")
    print(f"update   mean {agg('ms', safe_mean):.0f} ms/frame")

    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=True)
        out = args.output_root / "per_frame_metrics.csv"
        keys = sorted({k for r in rows for k in r})
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"per-frame metrics: {out}")


if __name__ == "__main__":
    main()
