#!/usr/bin/env python3
r"""
interactive_depth_filter_viewer_kitti.py

Interactive debug viewer for the keyframe inverse-depth filter (depth_filter.py).

Run:
  python -m sparse_depth.interactive_depth_filter_viewer_kitti --config configs/default.toml

Keys:
  n / space   advance one frame (filter update + automatic keyframing)
  a           toggle autoplay
  r           force a keyframe NOW (harvests converged seeds into the map,
              re-seeds uncovered regions; the map is kept)
  s           toggle active-seed dots
  c           toggle depth points (persistent map + current keyframe)
  l           toggle projected LiDAR dots
  q / Esc     quit
  left-click  on a depth point: prints its estimated depth vs the nearest
              LiDAR return (error in m and %) to the console.
              elsewhere: selects the nearest active seed; its epipolar search
              segment (yellow), best match (magenta x), and state are printed
  d           deselect

Reading the overlay:
  - small dots       = active seeds, projected at their current mean depth;
                       red = uncertain, green = nearly converged
  - filled circles   = depth points (world map + current keyframe's converged
                       seeds), colored by current depth (turbo, log scale)
  - cyan specks      = projected LiDAR (ground truth) if available
  - HUD              = seed/map counts, per-frame match stats, LiDAR error of
                       all visible depth points, and update time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from sparse_depth.config_io import load_argparse_defaults
from sparse_depth.depth_filter import (
    DepthFilterConfig,
    DepthFilterMapper,
    SEED_ACTIVE,
    SEED_CONVERGED,
    SEED_DEAD,
)
from sparse_depth.eval_metrics import safe_median
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
    p.add_argument("--img-dir", type=Path, required=False)
    p.add_argument("--calib", type=Path, required=False)
    p.add_argument("--poses", type=Path, required=False)
    p.add_argument("--image-digits", type=int, default=6)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--velodyne-dir", type=Path, default=None)
    p.add_argument("--lidar-digits", type=int, default=None)
    p.add_argument("--calib-velo-to-cam", type=Path, default=None)
    p.add_argument("--calib-cam-to-cam", type=Path, default=None)
    p.add_argument("--camera", type=str, default="00")
    p.add_argument("--lidar-radius-px", type=float, default=3.0)
    p.add_argument("--resize", type=float, default=1.4)

    p.add_argument("--seed-mode", choices=["gradient", "corners"], default="gradient",
                   help="gradient = semi-dense (LSD-style); corners = sparse Shi-Tomasi.")
    p.add_argument("--max-seeds", type=int, default=2500)
    p.add_argument("--grad-thresh", type=float, default=20.0)
    p.add_argument("--seeds-per-cell", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--zncc-min", type=float, default=0.85)
    p.add_argument("--z-seed-min", type=float, default=2.0, help="Near end of the depth prior.")
    p.add_argument("--z-seed-max", type=float, default=80.0, help="Far end of the depth prior.")
    p.add_argument("--converge-ratio", type=float, default=0.10,
                   help="Promote a seed when sigma_d/mu_d falls below this (lower = stricter/sparser).")
    p.add_argument("--converge-min-obs", type=int, default=4)
    p.add_argument("--max-search-px", type=float, default=300.0)
    p.add_argument("--kf-max-baseline", type=float, default=8.0,
                   help="Force a new keyframe when baseline from reference exceeds this (m).")
    p.add_argument("--kf-visible-frac", type=float, default=0.5,
                   help="Force a new keyframe when active in-view fraction drops below this.")

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
        return None
    try:
        if args.calib_velo_to_cam is not None and args.calib_cam_to_cam is not None:
            T_cam0_velo = load_raw_kitti_velo_to_cam(args.calib_velo_to_cam)
            P_rect, R_rect_4 = load_raw_kitti_cam_calib(args.calib_cam_to_cam, camera=args.camera)
        elif args.calib_velo_to_cam is not None:
            T_cam0_velo = load_raw_kitti_velo_to_cam(args.calib_velo_to_cam)
            data = read_kitti_calib_file(args.calib)
            P_rect = np.asarray(data["P0"], dtype=np.float64).reshape(3, 4)
            R_rect_4 = np.eye(4, dtype=np.float64)
        else:
            return None
    except Exception as e:
        print(f"[warn] LiDAR projection unavailable: {e}")
        return None
    return T_cam0_velo, R_rect_4, P_rect


def lidar_uv_z(args, lidar_proj, frame, img_shape):
    digits = args.lidar_digits if args.lidar_digits is not None else args.image_digits
    path = Path(args.velodyne_dir) / f"{frame:0{digits}d}.bin"
    if not path.exists():
        return None
    T_cam0_velo, R_rect_4, P_rect = lidar_proj
    pts = load_velodyne_bin(path)
    uv, z, _ = project_velodyne_to_image(pts, T_cam0_velo, R_rect_4, P_rect, img_shape)
    return uv, z


def depth_color(z, z_lo=2.0, z_hi=80.0):
    """Log-scaled turbo colormap value for one depth (BGR tuple)."""
    n = (np.log(max(z, 1e-3)) - np.log(z_lo)) / (np.log(z_hi) - np.log(z_lo))
    n = float(np.clip(n, 0.0, 1.0))
    v = np.array([[int(round((1.0 - n) * 255))]], dtype=np.uint8)  # near = hot end
    c = cv2.applyColorMap(v, cv2.COLORMAP_TURBO)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def main():
    args = parse_args()
    K = load_kitti_K(args.calib)
    poses = load_kitti_poses(args.poses)
    lidar_proj = prepare_lidar(args)

    cfg = DepthFilterConfig(
        seed_mode=args.seed_mode,
        max_seeds=args.max_seeds,
        grad_thresh=args.grad_thresh,
        seeds_per_cell=args.seeds_per_cell,
        patch_size=args.patch_size,
        zncc_min=args.zncc_min,
        z_min=args.z_seed_min,
        z_max=args.z_seed_max,
        converge_sigma_ratio=args.converge_ratio,
        converge_min_obs=args.converge_min_obs,
        max_search_px=args.max_search_px,
        kf_max_baseline_m=args.kf_max_baseline,
        kf_visible_frac=args.kf_visible_frac,
    )
    mapper = DepthFilterMapper(K, poses, cfg)

    frame = args.start
    ref_img = load_gray(args.img_dir, frame, args.image_digits)
    if ref_img is None:
        raise SystemExit(f"Could not load frame {frame} from {args.img_dir}")
    n_seeds, _, _ = mapper.set_reference(frame, ref_img)
    print(f"[depth-filter] keyframe {frame}: {n_seeds} seeds ({args.seed_mode} mode)")

    last_frame = len(poses) - 1
    if args.end_frame is not None:
        last_frame = min(last_frame, args.end_frame)

    show_seeds, show_points, show_lidar = True, True, lidar_proj is not None
    autoplay = False
    selected = -1
    stats = None
    lidar_line = "LiDAR: (unavailable)" if lidar_proj is None else "LiDAR: -"
    win = "depth filter"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    # click selection state shared with the mouse callback
    seed_draw = {"idx": np.zeros(0, dtype=int), "uv": np.zeros((0, 2))}
    depth_draw = {"uv": np.zeros((0, 2)), "z": np.zeros(0), "sig": np.zeros(0),
                  "kf": np.zeros(0, dtype=int)}
    cur_lidar = None  # (uv, z) of projected LiDAR for the current frame
    picked = -1       # index into depth_draw of the last clicked depth point

    def on_mouse(event, x, y, flags, param):
        nonlocal selected, picked
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pt = np.array([x / args.resize, y / args.resize])
        # Depth points first: click one to get its depth-vs-LiDAR report.
        if depth_draw["uv"].shape[0]:
            d = np.linalg.norm(depth_draw["uv"] - pt[None, :], axis=1)
            k = int(np.argmin(d))
            if d[k] < 10:
                picked = k
                z_est = float(depth_draw["z"][k])
                sig = float(depth_draw["sig"][k])
                kf = int(depth_draw["kf"][k])
                line = (f"[depth pt] px=({depth_draw['uv'][k,0]:.0f},{depth_draw['uv'][k,1]:.0f}) "
                        f"z_est={z_est:.2f}m (+/-{sig*z_est:.2f}m, kf {kf})")
                if cur_lidar is not None:
                    dl = np.linalg.norm(cur_lidar[0] - depth_draw["uv"][k][None, :], axis=1)
                    j = int(np.argmin(dl)) if dl.size else -1
                    if j >= 0 and dl[j] <= args.lidar_radius_px:
                        z_l = float(cur_lidar[1][j])
                        line += (f"  LiDAR={z_l:.2f}m  err={z_est - z_l:+.2f}m "
                                 f"({abs(z_est - z_l) / z_l * 100:.1f}%)")
                    else:
                        line += f"  (no LiDAR return within {args.lidar_radius_px:.0f}px)"
                print(line)
                return
        # Otherwise select the nearest active seed for epipolar inspection.
        if seed_draw["uv"].shape[0]:
            d = np.linalg.norm(seed_draw["uv"] - pt[None, :], axis=1)
            k = int(np.argmin(d))
            if d[k] < 25:
                selected = int(seed_draw["idx"][k])
                i = selected
                z = 1.0 / max(mapper.mu[i], 1e-12)
                ratio = float(np.sqrt(mapper.sigma2[i]) / max(mapper.mu[i], 1e-12))
                print(f"[seed {i}] ref px=({mapper.uv[i,0]:.1f},{mapper.uv[i,1]:.1f}) "
                      f"z={z:.2f}m sigma/mu={ratio:.3f} in={mapper.n_in[i]} out={mapper.n_out[i]} "
                      f"attempts={mapper.attempts[i]}")

    cv2.setMouseCallback(win, on_mouse)

    img = ref_img
    if lidar_proj is not None:
        cur_lidar = lidar_uv_z(args, lidar_proj, frame, img.shape)
    while True:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if show_lidar and cur_lidar is not None:
            uv, _z = cur_lidar
            step = max(1, uv.shape[0] // 4000)
            for u, v in uv[::step]:
                vis[int(v), int(u)] = (200, 255, 0)

        uv_d, z_d, sig_d, kf_d = mapper.visible_depth_points(frame, img.shape)
        depth_draw.update(uv=uv_d, z=z_d, sig=sig_d, kf=kf_d)
        if show_points:
            for (u, v), z in zip(uv_d, z_d):
                cv2.circle(vis, (int(u), int(v)), 2, depth_color(z), -1, cv2.LINE_AA)
        if 0 <= picked < uv_d.shape[0]:
            cv2.circle(vis, (int(uv_d[picked, 0]), int(uv_d[picked, 1])), 6,
                       (255, 255, 255), 1, cv2.LINE_AA)

        if show_seeds:
            idx, uv_s, ratio = mapper.seed_positions_in(frame)
            seed_draw["idx"], seed_draw["uv"] = idx, uv_s
            for (u, v), r in zip(uv_s, ratio):
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    g = float(np.clip(1.0 - r / 0.5, 0.0, 1.0))  # certain -> green
                    cv2.circle(vis, (int(u), int(v)), 1, (0, int(255 * g), int(255 * (1 - g))), -1)
        else:
            seed_draw["idx"], seed_draw["uv"] = np.zeros(0, dtype=int), np.zeros((0, 2))

        if selected >= 0 and selected in mapper.last_segments:
            p0, p1, match, zncc = mapper.last_segments[selected]
            cv2.line(vis, tuple(np.int32(p0)), tuple(np.int32(p1)), (0, 255, 255), 1, cv2.LINE_AA)
            if match is not None:
                cv2.drawMarker(vis, tuple(np.int32(match)), (255, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 9, 2)
            cv2.putText(vis, f"seed {selected} zncc={zncc:.2f}", tuple(np.int32(p1) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # LiDAR check of ALL visible depth points (map + current keyframe).
        if cur_lidar is not None and uv_d.shape[0] > 0:
            matched, z_l, _ = match_sparse_to_lidar_with_radius(
                uv_d, cur_lidar[0], cur_lidar[1], args.lidar_radius_px)
            if np.any(matched):
                diff = z_d[matched] - z_l[matched]
                rel = np.abs(diff) / z_l[matched]
                lidar_line = (f"LiDAR: {int(np.sum(matched))}/{uv_d.shape[0]} matched  "
                              f"medRel={safe_median(rel):.3f} meanRel={np.mean(rel):.3f}  "
                              f"medAbs={safe_median(np.abs(diff)):.2f}m meanAbs={np.mean(np.abs(diff)):.2f}m")
            else:
                lidar_line = f"LiDAR: 0/{uv_d.shape[0]} matched"

        n_act = int(np.sum(mapper.status == SEED_ACTIVE))
        n_con = int(np.sum(mapper.status == SEED_CONVERGED))
        n_dead = int(np.sum(mapper.status == SEED_DEAD))
        hud = [
            f"frame {frame}  (kf {mapper.ref_frame}, +{frame - mapper.ref_frame})",
            f"seeds  active {n_act}  converged {n_con}  dead {n_dead}",
            f"map {mapper.map_xyz.shape[0]} pts  |  visible depth pts {uv_d.shape[0]}",
        ]
        if stats is not None:
            hud.append(f"this frame: searched {stats.searched} matched {stats.matched} "
                       f"fused {stats.fused} outliers {stats.outliers} "
                       f"(not-in-view {stats.no_view})")
            hud.append(f"median sigma/mu {stats.median_sigma_ratio:.3f}   update {stats.update_ms:.0f} ms")
        hud.append(lidar_line)
        hud.append("[n]ext [a]uto [r]=force keyframe [s]eeds [c]points [l]idar click=inspect [q]uit")
        y = 18
        for line in hud:
            cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

        disp = cv2.resize(vis, None, fx=args.resize, fy=args.resize,
                          interpolation=cv2.INTER_NEAREST) if args.resize != 1.0 else vis
        cv2.imshow(win, disp)
        key = cv2.waitKey(30 if autoplay else 0) & 0xFF

        advance = autoplay
        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), ord(" ")):
            advance = True
        elif key == ord("a"):
            autoplay = not autoplay
        elif key == ord("s"):
            show_seeds = not show_seeds
        elif key == ord("c"):
            show_points = not show_points
        elif key == ord("l"):
            show_lidar = not show_lidar
        elif key == ord("d"):
            selected = -1
        elif key == ord("r"):
            n, harv, dropped = mapper.set_reference(frame, img)
            selected = -1
            picked = -1
            stats = None
            print(f"[KF] forced keyframe {frame}: harvested {harv} -> map "
                  f"{mapper.map_xyz.shape[0]}, dropped {dropped} unconverged, "
                  f"reseeded {n}")
            continue

        if advance and frame < last_frame:
            frame += 1
            nxt = load_gray(args.img_dir, frame, args.image_digits)
            if nxt is None:
                print(f"[depth-filter] missing frame {frame}; stopping")
                frame -= 1
                autoplay = False
                continue
            img = nxt
            picked = -1
            selected = -1 if selected >= 0 and mapper.frames_since_ref == 0 else selected
            stats = mapper.process(frame, img)
            if lidar_proj is not None:
                cur_lidar = lidar_uv_z(args, lidar_proj, frame, img.shape)
            print(f"[f{frame:04d}] searched {stats.searched:3d} matched {stats.matched:3d} "
                  f"fused {stats.fused:3d} out {stats.outliers:2d} conv +{stats.new_converged:2d} "
                  f"(cur {stats.total_converged}, map {stats.map_size})  "
                  f"med sig/mu {stats.median_sigma_ratio:.3f}  {stats.update_ms:.0f} ms")
            if stats.kf_switched:
                selected = -1
                print(f"[KF] auto keyframe at frame {frame}: harvested {stats.harvested} "
                      f"-> map {stats.map_size}, reseeded {stats.reseeded} "
                      f"(baseline was {stats.baseline_m:.1f}m)")
        elif advance:
            autoplay = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
