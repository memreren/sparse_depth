#!/usr/bin/env python3
r"""
ground_plane_eval_kitti.py

Phase-0 substrate for the ground-plane road-depth work. Over a frame range it:

  1. CALIBRATES: per frame, RANSAC-fits the road plane to camera-frame LiDAR
     points inside the road ROI (giving the ground-truth normal n*(i) and height
     h*(i)); the mean over frames cancels road grade/bank + suspension and leaves
     the constant camera mounting + rectification offset. Mean h* calibrates h.

  2. SCORES: for each frame it compares plane-predicted depth to projected LiDAR
     on the same road-ROI points, bucketed by range, for several plane hypotheses:
       - level        : [0,1,0], d = -h_nominal          (Phase 1 baseline)
       - level_hcal   : [0,1,0], d = -h_cal              (height-calibrated)
       - offset       : n_offset (mean n*), d = -h_cal   (Method 1', Phase 2)
       - oracle       : per-frame n*(i), h*(i)           (single-plane floor)

Depth for a road pixel uses only K + plane (no pose): Z = -d * r_z / (n . r),
r = K^-1 [u,v,1]. Same convention as sparse_depth.ground_plane.ray_plane_depth.

Usage:
  python ground_plane_eval_kitti.py --config configs/default.toml \
      --config configs/plane_homography_seq04.toml --start 0 --num-frames 150
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sparse_depth.config_io import load_argparse_defaults
from sparse_depth.geometry import relative_pose
from sparse_depth.ground_calibration import GroundCalibration, normal_pitch_roll_deg, select_road_points
from sparse_depth.ground_plane import Plane, fit_plane_ransac, plane_homography, road_trapezoid_mask
from sparse_depth.plane_validation import ValidationPolicy, symmetric_photometric_gate
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
    p.add_argument("--num-frames", type=int, default=150)
    p.add_argument("--velodyne-dir", type=Path, default=None)
    p.add_argument("--lidar-digits", type=int, default=None)
    p.add_argument("--calib-velo-to-cam", type=Path, default=None)
    p.add_argument("--calib-cam-to-cam", type=Path, default=None)
    p.add_argument("--camera", type=str, default="00")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Optional dir for per_frame_ground_plane.csv")

    p.add_argument("--camera-height-m", dest="camera_height_m", type=float, default=1.65,
                   help="Nominal h for the 'level' baseline hypothesis.")
    p.add_argument("--min-depth-m", dest="min_depth_m", type=float, default=2.0)
    p.add_argument("--max-depth-m", dest="max_depth_m", type=float, default=45.0)
    p.add_argument("--range-edges", type=str, default="0,15,30,45",
                   help="Comma-separated range-bucket edges in metres.")
    p.add_argument("--lidar-plane-thresh-m", dest="lidar_plane_thresh_m", type=float, default=0.06)
    p.add_argument("--lidar-plane-min-inliers", dest="lidar_plane_min_inliers", type=int, default=40)
    p.add_argument("--zncc-gate", action=argparse.BooleanOptionalAction, default=True,
                   help="Also report AbsRel after the symmetric-ZNCC on-plane gate (keep gray+green).")
    p.add_argument("--gate-gap", type=int, default=1, help="Frame gap for the gate's plane homography.")
    p.add_argument("--gate-zncc-thresh", type=float, default=0.35)
    p.add_argument("--gate-patch-radius-px", type=int, default=4)
    p.add_argument("--gate-min-patch-std", type=float, default=5.0)
    # Membership ground truth: LiDAR residual to the per-frame oracle road plane.
    p.add_argument("--member-on-thresh-m", dest="member_on_thresh_m", type=float, default=0.10,
                   help="|residual| below this = truly on-plane (road).")
    p.add_argument("--member-off-thresh-m", dest="member_off_thresh_m", type=float, default=0.30,
                   help="|residual| above this = truly off-plane (car/curb/fence). Between = ignored.")

    # ROI trapezoid; names match the viewer's PlaneConfig so a shared TOML applies.
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
    return args


def plane_depth_at_pixels(K: np.ndarray, plane: Plane, uv: np.ndarray) -> np.ndarray:
    """Per-pixel ray-plane depth Z, NaN where the ray misses the plane ahead."""
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    pix = np.hstack([uv, np.ones((len(uv), 1))])
    rays = (Kinv @ pix.T).T
    denom = rays @ plane.normal
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -plane.offset / denom
        Z = lam * rays[:, 2]
    bad = ~np.isfinite(Z) | (lam <= 0) | (rays[:, 2] <= 0)
    Z[bad] = np.nan
    return Z


def road_lidar(frame, args, lidar, roi, digits):
    """Return (uv, z, xyz) of LiDAR points inside the road ROI + depth band."""
    path = Path(args.velodyne_dir) / f"{frame:0{digits}d}.bin"
    img = load_gray(args.img_dir, frame, args.image_digits)
    if img is None or not path.exists():
        return None
    uv, z, xyz = project_velodyne_to_image(load_velodyne_bin(path), *lidar, img.shape,
                                           min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m)
    if uv.size == 0:
        return None
    keep = select_road_points(uv, z, roi, img.shape, args.min_depth_m, args.max_depth_m)
    return uv[keep], z[keep], xyz[keep]


def main():
    args = parse_args()
    K = load_kitti_K(args.calib)
    poses = load_kitti_poses(args.poses)
    if args.velodyne_dir is None or not Path(args.velodyne_dir).exists():
        raise SystemExit("LiDAR is required for ground-plane eval (set velodyne_dir in config).")
    if args.calib_velo_to_cam is None or args.calib_cam_to_cam is None:
        raise SystemExit("Need --calib-velo-to-cam and --calib-cam-to-cam for raw LiDAR projection.")
    # load_raw_kitti_cam_calib returns (P_rect, R_rect_4); project wants (T, R_rect_4, P_rect)
    T_cv = load_raw_kitti_velo_to_cam(args.calib_velo_to_cam)
    P_rect, R_rect = load_raw_kitti_cam_calib(args.calib_cam_to_cam, camera=args.camera)
    lidar = (T_cv, R_rect, P_rect)

    first = load_gray(args.img_dir, args.start, args.image_digits)
    if first is None:
        raise SystemExit(f"cannot load start frame {args.start}")
    roi = road_trapezoid_mask(first.shape, args.homography_roi_top_y_frac,
                              args.homography_roi_bottom_left_frac, args.homography_roi_bottom_right_frac,
                              args.homography_roi_top_left_frac, args.homography_roi_top_right_frac)
    edges = [float(x) for x in args.range_edges.split(",")]
    buckets = list(zip(edges[:-1], edges[1:]))
    digits = args.lidar_digits if args.lidar_digits is not None else args.image_digits
    end = min(args.start + args.num_frames, len(poses)) - 1
    print(f"ground-plane eval: seq frames {args.start}..{end} | ROI top_y={args.homography_roi_top_y_frac} "
          f"| band [{args.min_depth_m},{args.max_depth_m}]m | buckets {buckets}")

    # ---- Pass 1: per-frame LiDAR road-plane fit (ground-truth normal/height) ----
    per_frame = []
    normals, heights = [], []
    for f in range(args.start, end + 1):
        data = road_lidar(f, args, lidar, roi, digits)
        if data is None:
            continue
        uv, z, xyz = data
        plane, inliers, rms = fit_plane_ransac(xyz, thresh_m=args.lidar_plane_thresh_m,
                                               min_inliers=args.lidar_plane_min_inliers)
        rec = {"frame": f, "n_road": len(uv), "uv": uv, "z": z, "xyz": xyz, "plane": plane,
               "n_inliers": int(inliers.sum()) if plane is not None else 0, "rms": rms}
        if plane is not None:
            normals.append(plane.normal)
            heights.append(-plane.offset)
            rec["pitch"], rec["roll"], rec["angle_down"] = normal_pitch_roll_deg(plane.normal)
        per_frame.append(rec)

    if not normals:
        raise SystemExit("no successful LiDAR plane fits; check ROI/paths.")
    normals = np.array(normals)
    heights = np.array(heights)
    n_offset = normals.mean(axis=0)
    n_offset = n_offset / np.linalg.norm(n_offset)
    h_cal = float(np.median(heights))
    off_pitch, off_roll, off_angle = normal_pitch_roll_deg(n_offset)
    per_pitch = np.array([r["pitch"] for r in per_frame if "pitch" in r])
    per_roll = np.array([r["roll"] for r in per_frame if "roll" in r])

    print("\n===== CALIBRATION (mean of per-frame LiDAR road-plane fits) =====")
    print(f"frames fit         : {len(normals)} / {len(per_frame)}")
    print(f"n_offset (mean n*) : [{n_offset[0]:+.4f}, {n_offset[1]:+.4f}, {n_offset[2]:+.4f}]")
    print(f"constant offset    : pitch {off_pitch:+.3f} deg | roll {off_roll:+.3f} deg | angle-to-down {off_angle:.3f} deg")
    print(f"per-frame spread   : pitch {per_pitch.mean():+.3f}+/-{per_pitch.std():.3f} | roll {per_roll.mean():+.3f}+/-{per_roll.std():.3f} deg")
    print(f"                     (std = road grade/bank + suspension; mean = mounting offset)")
    print(f"height h*          : median {h_cal:.3f} m | mean {heights.mean():.3f} +/- {heights.std():.3f} m (nominal {args.camera_height_m:.3f})")
    print(f"fit inliers/road   : {np.mean([r['n_inliers'] for r in per_frame]):.0f}/{np.mean([r['n_road'] for r in per_frame]):.0f} | rms {np.nanmean([r['rms'] for r in per_frame if r['plane'] is not None]):.3f} m")

    # ---- Pass 2: score hypotheses vs LiDAR, bucketed by range ----
    hyps = {
        "level":      lambda rec: Plane(np.array([0.0, 1.0, 0.0]), -args.camera_height_m),
        "level_hcal": lambda rec: Plane(np.array([0.0, 1.0, 0.0]), -h_cal),
        "offset":     lambda rec: Plane(n_offset, -h_cal),
        "oracle":     lambda rec: rec["plane"],
    }
    # accumulators: hyp -> bucket_index (or 'all') -> list of abs-rel
    def new_acc():
        a = {name: {i: [] for i in range(len(buckets))} for name in hyps}
        for name in hyps:
            a[name]["all"] = []
        return a

    acc = new_acc()          # ungated: every road-ROI LiDAR point
    acc_g = new_acc()        # ZNCC-gated: only points the plane's gate keeps (gray+green)
    kept = {name: [0, 0] for name in hyps}   # [kept, eligible]
    # Membership: does the gate keep true road and reject true off-plane?
    member = {name: {"on_keep": 0, "on_tot": 0, "off_rej": 0, "off_tot": 0} for name in hyps}
    policy = ValidationPolicy(patch_radius_px=args.gate_patch_radius_px,
                              min_patch_std=args.gate_min_patch_std,
                              zncc_threshold=args.gate_zncc_thresh)

    for rec in per_frame:
        if rec["plane"] is None:
            continue
        uv, z = rec["uv"], rec["z"]
        ij = np.rint(uv).astype(int)
        # Ground-truth membership label from residual to the oracle (LiDAR-fit) plane.
        signed = rec["xyz"] @ rec["plane"].normal + rec["plane"].offset
        on_lbl = np.abs(signed) < args.member_on_thresh_m
        off_lbl = np.abs(signed) > args.member_off_thresh_m
        gates = {}
        if args.zncc_gate:
            f, sf = rec["frame"], rec["frame"] - args.gate_gap
            target = load_gray(args.img_dir, f, args.image_digits)
            source = load_gray(args.img_dir, sf, args.image_digits) if sf >= 0 else None
            if target is not None and source is not None:
                R_ts, t_ts = relative_pose(poses, f, sf)
                for name, make in hyps.items():
                    plane = make(rec)
                    if plane is None:
                        continue
                    try:
                        H = plane_homography(K, R_ts, t_ts, plane)
                    except ValueError:
                        continue
                    gates[name] = symmetric_photometric_gate(target, source, H, roi, policy)["keep_gate"]
        for name, make in hyps.items():
            plane = make(rec)
            if plane is None:
                continue
            pred = plane_depth_at_pixels(K, plane, uv)
            ok = np.isfinite(pred)
            rel = np.abs(pred[ok] - z[ok]) / np.maximum(z[ok], 1e-9)
            zk = z[ok]
            acc[name]["all"].extend(rel.tolist())
            for bi, (lo, hi) in enumerate(buckets):
                acc[name][bi].extend(rel[(zk >= lo) & (zk < hi)].tolist())
            rec[f"mean_rel_{name}"] = float(np.mean(rel)) if rel.size else float("nan")
            if name in gates:
                gm = gates[name][ij[ok, 1], ij[ok, 0]]
                kept[name][0] += int(gm.sum())
                kept[name][1] += int(gm.size)
                relg, zg = rel[gm], zk[gm]
                acc_g[name]["all"].extend(relg.tolist())
                for bi, (lo, hi) in enumerate(buckets):
                    acc_g[name][bi].extend(relg[(zg >= lo) & (zg < hi)].tolist())
                # Membership: gate keep vs GT label, over ALL road-ROI points (not depth-ok only).
                keep_all = gates[name][ij[:, 1], ij[:, 0]]
                member[name]["on_keep"] += int(keep_all[on_lbl].sum())
                member[name]["on_tot"] += int(on_lbl.sum())
                member[name]["off_rej"] += int((~keep_all[off_lbl]).sum())
                member[name]["off_tot"] += int(off_lbl.sum())

    def summ(vals):
        v = np.array(vals)
        return (len(v), float(np.mean(v)) if v.size else float("nan"),
                float(np.median(v)) if v.size else float("nan"))

    col_labels = [f"{lo:.0f}-{hi:.0f}m" for lo, hi in buckets] + ["all"]

    def print_table(title, A):
        print(title)
        header = f"{'hypothesis':<12}" + "".join(f"{c:>18}" for c in col_labels)
        print(header)
        print("-" * len(header))
        for name in ("level", "level_hcal", "offset", "oracle"):
            cells = []
            for key in list(range(len(buckets))) + ["all"]:
                _, mean, med = summ(A[name][key])
                cells.append(f"{mean:6.3f}|{med:6.3f}")
            print(f"{name:<12}" + "".join(f"{c:>18}" for c in cells))

    print("\n===== DEPTH AbsRel vs LiDAR (mean | median), road ROI =====")
    print_table("[ungated: all road-ROI LiDAR points]", acc)
    n_all, _, _ = summ(acc["level"]["all"])
    print(f"(points scored per hypothesis overall: {n_all})")
    if args.zncc_gate:
        print()
        print_table("[ZNCC-gated: keep gray (uninformative) + green (textured agreement)]", acc_g)
        print("kept fraction (gated/eligible): " +
              " | ".join(f"{name} {kept[name][0] / max(kept[name][1], 1):.1%}"
                         for name in ("level", "offset", "oracle")))
        print(f"\n===== MEMBERSHIP (gate vs LiDAR label: on-plane<{args.member_on_thresh_m}m, off-plane>{args.member_off_thresh_m}m) =====")
        print(f"{'gate plane':<12}{'road retention':>16}{'off-plane reject':>18}{'(n_on / n_off)':>20}")
        print("-" * 66)
        for name in ("level", "offset", "oracle"):
            m = member[name]
            ret = m["on_keep"] / max(m["on_tot"], 1)
            rej = m["off_rej"] / max(m["off_tot"], 1)
            counts = f"{m['on_tot']} / {m['off_tot']}"
            print(f"{name:<12}{ret:>15.1%}{rej:>18.1%}{counts:>20}")
        print("road retention = kept true road (want high); off-plane reject = rejected true")
        print("obstacles (want high). This is the direct membership scoreboard.")
    print("\nread: 'offset' closing the level->oracle gap = value of the constant calibration;")
    print("residual oracle error = single-plane floor (road non-planarity + LiDAR noise).")
    print("gated vs ungated = value of the on-plane membership test (should help most on")
    print("frames where off-plane objects, e.g. cars, sit inside the ROI).")

    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=True)
        out = args.output_root / "per_frame_ground_plane.csv"
        keys = ["frame", "n_road", "n_inliers", "rms", "pitch", "roll", "angle_down",
                "mean_rel_level", "mean_rel_level_hcal", "mean_rel_offset", "mean_rel_oracle"]
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for rec in per_frame:
                w.writerow(rec)
        # also stash the calibration itself
        with open(args.output_root / "calibration.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["n_offset_x", "n_offset_y", "n_offset_z", "pitch_deg", "roll_deg", "angle_down_deg", "h_cal_m"])
            w.writerow([*n_offset, off_pitch, off_roll, off_angle, h_cal])
        print(f"\nper-frame: {out}")


if __name__ == "__main__":
    main()
