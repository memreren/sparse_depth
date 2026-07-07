#!/usr/bin/env python3
r"""
tti_direct_eval_kitti.py

Headless evaluator for the direct gradient-based time-to-contact estimator
(sparse_depth.tti_direct). For each consecutive frame pair it estimates tau at
every projected-LiDAR pixel straight from image gradients (no correspondences,
no triangulation), then scores it two ways against the SAME LiDAR:

  * tau error   : tau_est  vs  tau_gt = Z_lidar / Vz         (native, units=frames)
  * depth error : Z_est = tau_est * Vz  vs  Z_lidar          (comparable to the
                  triangulation runs, using GT forward speed Vz for the lift)

Vz (GT forward speed per frame) and the FOE come from GT poses, so this isolates
the estimator from motion estimation. This is the "direct TTC" arm of the
TTC-vs-triangulation comparison.

Usage:
  python tti_direct_eval_kitti.py --config configs/default.toml --start 0 --num-frames 200 \
      --output-root outputs/tti_direct_seq10
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from sparse_depth.config_io import load_argparse_defaults
from sparse_depth.geometry import relative_pose
from sparse_depth.tti_direct import DirectTTCConfig, DirectTTCFrame
from sparse_depth.eval_metrics import compute_depth_metrics, safe_mean, safe_median
from sparse_depth.kitti_io import (
    load_gray,
    load_kitti_K,
    load_kitti_poses,
    load_raw_kitti_cam_calib,
    load_raw_kitti_velo_to_cam,
    load_velodyne_bin,
    project_velodyne_to_image,
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
    p.add_argument("--num-frames", type=int, default=200)
    p.add_argument("--gap", type=int, default=1,
                   help="Frame gap prev->curr. Direct TTC assumes small motion; 1 is standard.")
    p.add_argument("--velodyne-dir", type=Path, default=None)
    p.add_argument("--lidar-digits", type=int, default=None)
    p.add_argument("--calib-velo-to-cam", type=Path, default=None)
    p.add_argument("--calib-cam-to-cam", type=Path, default=None)
    p.add_argument("--camera", type=str, default="00")
    p.add_argument("--min-lidar-depth", type=float, default=1.0)
    p.add_argument("--max-lidar-depth", type=float, default=120.0)
    p.add_argument("--min-speed-m", type=float, default=0.05,
                   help="Skip frame pairs whose GT forward motion is below this (near-stationary).")
    p.add_argument("--output-root", type=Path, default=None)

    # direct TTC estimator knobs
    p.add_argument("--patch-radius", type=int, default=7)
    p.add_argument("--pyramid-level", type=int, default=2)
    p.add_argument("--min-structure", type=float, default=1e3)
    p.add_argument("--min-valid-frac", type=float, default=0.85)
    p.add_argument("--min-tau", type=float, default=0.5)
    p.add_argument("--max-tau", type=float, default=400.0)

    known = {a.dest for a in p._actions}
    values = load_argparse_defaults(ns.config)
    p.set_defaults(**{k: v for k, v in values.items() if k in known})
    args = p.parse_args()
    for req in ("img_dir", "calib", "poses"):
        if getattr(args, req) is None:
            p.error(f"--{req.replace('_', '-')} is required (flag or config file)")
    return args


def prepare_lidar(args):
    if args.velodyne_dir is None or not Path(args.velodyne_dir).exists():
        print("[warn] no LiDAR dir; nothing to score against")
        return None
    if args.calib_velo_to_cam is None or args.calib_cam_to_cam is None:
        print("[warn] need --calib-velo-to-cam and --calib-cam-to-cam for LiDAR projection")
        return None
    T_cv = load_raw_kitti_velo_to_cam(args.calib_velo_to_cam)
    P_rect, R_rect = load_raw_kitti_cam_calib(args.calib_cam_to_cam, camera=args.camera)
    return T_cv, R_rect, P_rect


def main():
    args = parse_args()
    K = load_kitti_K(args.calib)
    poses = load_kitti_poses(args.poses)
    lidar = prepare_lidar(args)
    if lidar is None:
        raise SystemExit("LiDAR projection is required to score tau; aborting.")

    cfg = DirectTTCConfig(
        patch_radius_px=args.patch_radius,
        pyramid_level=args.pyramid_level,
        min_structure=args.min_structure,
        min_valid_frac=args.min_valid_frac,
        min_tau_frames=args.min_tau,
        max_tau_frames=args.max_tau,
    )

    digits = args.lidar_digits if args.lidar_digits is not None else args.image_digits
    end = min(args.start + args.num_frames, len(poses)) - 1
    print(f"direct-TTC eval: frames {args.start}..{end} gap={args.gap} "
          f"level={args.pyramid_level} patch={args.patch_radius}")

    rows = []
    t_all = time.time()
    for f in range(args.start + args.gap, end + 1):
        prev_i = f - args.gap
        prev = load_gray(args.img_dir, prev_i, args.image_digits)
        curr = load_gray(args.img_dir, f, args.image_digits)
        if prev is None or curr is None:
            continue
        R, t = relative_pose(poses, prev_i, f)
        Vz = -float(t[2])  # forward speed per gap; >0 when approaching
        if abs(Vz) < args.min_speed_m:
            continue

        lidar_path = Path(args.velodyne_dir) / f"{f:0{digits}d}.bin"
        if not lidar_path.exists():
            continue
        velo = load_velodyne_bin(lidar_path)
        lidar_uv, lidar_z, _ = project_velodyne_to_image(
            velo, *lidar, image_shape=curr.shape,
            min_depth_m=args.min_lidar_depth, max_depth_m=args.max_lidar_depth)
        if lidar_uv.shape[0] == 0:
            continue

        t0 = time.time()
        frame = DirectTTCFrame(prev, curr, K, R, t, cfg)
        tau_est = frame.estimate_points(lidar_uv)
        ms = (time.time() - t0) * 1000.0

        tau_gt = lidar_z / Vz
        z_est = tau_est * Vz  # metric depth via GT scale, comparable to triangulation

        finite = np.isfinite(tau_est)
        tau_m = compute_depth_metrics(tau_est, tau_gt)
        z_m = compute_depth_metrics(z_est, lidar_z)
        row = {
            "frame": f, "prev": prev_i, "Vz_m": Vz,
            "n_lidar": int(lidar_uv.shape[0]),
            "n_tau": int(np.sum(finite)),
            "coverage": float(np.mean(finite)),
            "ms": ms,
            "tau_medRel": tau_m["median_rel_err"], "tau_meanRel": tau_m["mean_rel_err"],
            "tau_medAbs": tau_m["median_abs_err_m"], "tau_d20": tau_m["delta_20"],
            "z_medRel": z_m["median_rel_err"], "z_meanRel": z_m["mean_rel_err"],
            "z_medAbs": z_m["median_abs_err_m"], "z_meanAbs": z_m["mean_abs_err_m"],
            "z_d20": z_m["delta_20"], "z_medZ_est": safe_median(z_est[finite]),
            "z_medZ_lidar": safe_median(lidar_z),
        }
        rows.append(row)

    if not rows:
        raise SystemExit("no scorable frame pairs produced")

    def agg(key, fn=safe_median):
        return fn([r.get(key, float("nan")) for r in rows])

    print(f"\n===== direct-TTC summary ({len(rows)} pairs, {time.time()-t_all:.0f}s) =====")
    print(f"coverage   med {agg('coverage'):.3f}   (finite tau / projected LiDAR)")
    print(f"n_tau      med {agg('n_tau'):.0f}")
    print(f"-- tau (frames) --")
    print(f"tau medRel med {agg('tau_medRel'):.4f}   meanRel mean {agg('tau_meanRel', safe_mean):.4f}")
    print(f"tau d<20%  med {agg('tau_d20'):.3f}")
    print(f"-- depth Z = tau*Vz (m), comparable to triangulation --")
    print(f"z medRel   med {agg('z_medRel'):.4f}   meanRel mean {agg('z_meanRel', safe_mean):.4f}")
    print(f"z medAbs   med {agg('z_medAbs'):.3f}m  meanAbs mean {agg('z_meanAbs', safe_mean):.3f}m")
    print(f"z d<20%    med {agg('z_d20'):.3f}")
    print(f"update     mean {agg('ms', safe_mean):.0f} ms/pair")

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
