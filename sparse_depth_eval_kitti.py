#!/usr/bin/env python3
r"""
Headless SIFT+LK feature-manager evaluator for KITTI sparse temporal depth.

This is intentionally close to interactive_feature_manager_kitti.py:
LK-first tracking, SIFT rescue/spawn, soft rescue/spawn, pair choice, and
triangulation gates come from that file. The difference is that this script does
no drawing and computes triangulation once per frame, then reuses that same
result for depth metrics, pose metrics, CSV output, and frame-wise plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparse_depth.config_io import load_argparse_defaults
from sparse_depth.pose_eval import PoseEval
from sparse_depth.manager_config import (
    Config,
    DEFAULT_CALIB_PATH,
    DEFAULT_IMAGE_DIGITS,
    DEFAULT_IMG_DIR,
    DEFAULT_POSES_PATH,
    resolve_detector_descriptor,
    validate_feature_axes,
)
from sparse_depth.feature_manager import FeatureManager
from sparse_depth.track_types import FrameStats
from sparse_depth.triangulation import TriangInfo
from sparse_depth.eval_metrics import (
    compute_depth_metrics,
    compute_scale_factors,
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

GOOD_LABELS = {"confirmed_good", "candidate_good", "reacq_good"}
DEFAULT_OUTPUT_ROOT = Path("outputs/sift_lk_feature_depth_eval")


def json_safe_value(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, list):
        return [json_safe_value(x) for x in v]
    if isinstance(v, tuple):
        return [json_safe_value(x) for x in v]
    if isinstance(v, dict):
        return {k: json_safe_value(x) for k, x in v.items()}
    return v


def make_manager_config(args: argparse.Namespace) -> Config:
    # Resolve the feature axes (new --detector/--descriptor win; else legacy
    # --detector-mode is decomposed) and write them back so config.json records
    # the resolved values.
    args.detector, args.descriptor = resolve_detector_descriptor(
        args.detector, args.descriptor, args.detector_mode)
    validate_feature_axes(args.detector, args.descriptor, args.matcher)
    # LK off has no frame-to-frame carrier, so detection must run every frame or
    # non-detection frames produce nothing. Enforce it here so it can't be misset.
    if not args.lk_on and args.detection_period != 1:
        print("[note] LK tracking off -> forcing detection_period=1.")
        args.detection_period = 1
    return Config(
        img_dir=args.img_dir,
        calib_path=args.calib,
        poses_path=args.poses,
        velodyne_dir=args.velodyne_dir,
        calib_velo_to_cam=args.calib_velo_to_cam,
        calib_cam_to_cam=args.calib_cam_to_cam,
        camera=args.camera,
        image_digits=args.image_digits,
        lidar_digits=args.lidar_digits,
        start_frame=args.start,
        end_frame=args.end_frame,
        output_root=args.output_root,
        detector=args.detector,
        descriptor=args.descriptor,
        shi_max_corners=args.shi_max_corners,
        shi_quality_level=args.shi_quality_level,
        shi_min_distance_px=args.shi_min_distance,
        shi_block_size=args.shi_block_size,
        xfeat_top_k=args.xfeat_top_k,
        xfeat_detection_threshold=args.xfeat_detection_threshold,
        xfeat_desc_size_px=args.xfeat_desc_size,
        xfeat_native_desc_scale=args.xfeat_native_desc_scale,
        sift_nfeatures=args.sift_nfeatures,
        sift_n_octave_layers=args.sift_n_octave_layers,
        sift_contrast_threshold=args.contrast,
        sift_edge_threshold=args.edge,
        sift_sigma=args.sigma,
        soft_spawn=args.soft_spawn,
        soft_rescue=args.soft_rescue,
        soft_nfeatures=args.soft_nfeatures,
        soft_contrast_threshold=args.soft_contrast,
        soft_edge_threshold=args.soft_edge,
        soft_sigma=args.soft_sigma,
        soft_confirm_hits=args.soft_confirm_hits,
        soft_rescue_search_radius_px=args.soft_rescue_search_radius,
        soft_rescue_ratio=args.soft_rescue_ratio,
        soft_rescue_max_desc_dist=args.soft_rescue_max_desc_dist,
        soft_detect_mask_used_primary_px=args.soft_detect_mask_used_primary,
        soft_dup_used_primary_px=args.soft_dup_used_primary,
        soft_spawn_primary_sep_px=args.soft_spawn_primary_sep,
        soft_source_penalty=args.soft_source_penalty,
        descriptor_mode=args.descriptor_mode,
        matcher=args.matcher,
        xfeat_mnn_min_cossim=args.xfeat_mnn_min_cossim,
        xfeat_mnn_max_step_px=args.xfeat_mnn_max_step,
        prediction_mode=args.prediction,
        search_radius_px=args.search_radius,
        ratio=args.ratio,
        max_desc_dist=args.max_desc_dist,
        reacq_search_radius_px=args.reacq_search_radius,
        reacq_ratio=args.reacq_ratio,
        reacq_max_desc_dist=args.reacq_max_desc_dist,
        allow_single_candidate=args.allow_single_candidate,
        spatial_weight=args.spatial_weight,
        miss_cost=args.miss_cost,
        max_step_px=args.max_step_px,
        lk_on=args.lk_on,
        lk_require_desc_refresh=args.lk_require_desc_refresh,
        lk_win_size=args.lk_win_size,
        lk_max_level=args.lk_max_level,
        lk_max_iter=args.lk_max_iter,
        lk_eps=args.lk_eps,
        lk_fb_thresh_px=args.lk_fb_thresh,
        lk_max_step_px=args.lk_max_step_px,
        lk_max_error=args.lk_max_error,
        lk_min_eig_thresh=args.lk_min_eig_thresh,
        lk_desc_refresh_radius_px=args.lk_desc_refresh_radius,
        lk_desc_refresh_max_dist=args.lk_desc_refresh_max_dist,
        detection_period=args.detection_period,
        soft_sift_period=args.soft_sift_period,
        spawn_period=args.spawn_period,
        force_detection_active_below=args.force_detection_active_below,
        force_detection_coverage_below=args.force_detection_coverage_below,
        force_soft_underfilled_buckets=args.force_soft_underfilled_buckets,
        lk_epipolar_thresh_px=args.lk_epipolar_thresh,
        lk_flow_consistency=args.lk_flow_consistency,
        lk_flow_radius_px=args.lk_flow_radius,
        lk_flow_min_neighbors=args.lk_flow_min_neighbors,
        lk_flow_mad_k=args.lk_flow_mad_k,
        lk_flow_abs_thresh_px=args.lk_flow_abs_thresh,
        min_hits_to_confirm=args.min_hits_confirm,
        max_misses=args.max_misses,
        max_active_tracks=args.max_active_tracks,
        duplicate_suppression=args.duplicate_suppression,
        duplicate_dist_px=args.duplicate_dist,
        quality_confirmed_bonus=args.quality_confirmed_bonus,
        quality_hit_weight=args.quality_hit_weight,
        quality_hit_cap=args.quality_hit_cap,
        quality_miss_penalty=args.quality_miss_penalty,
        quality_reacq_penalty=args.quality_reacq_penalty,
        quality_probation_penalty=args.quality_probation_penalty,
        quality_lk_bonus=args.quality_lk_bonus,
        quality_lk_stale_after=args.quality_lk_stale_after,
        quality_lk_stale_penalty=args.quality_lk_stale_penalty,
        quality_lk_stale_cap=args.quality_lk_stale_cap,
        quality_desc_penalty=args.quality_desc_penalty,
        quality_spatial_penalty=args.quality_spatial_penalty,
        quality_active_bonus=args.quality_active_bonus,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        target_per_bucket=args.target_per_bucket,
        max_per_bucket=args.max_per_bucket,
        output_per_bucket=args.output_per_bucket,
        output_max_tracks=args.output_max_tracks,
        min_spawn_distance_px=args.min_spawn_distance,
        spawn_count_candidates=args.spawn_count_candidates,
        triangulation_method=args.triangulation_method,
        multiview_min_views=args.multiview_min_views,
        hybrid_pair_min_parallax_deg=args.hybrid_pair_min_parallax_deg,
        hybrid_pair_min_baseline_m=args.hybrid_pair_min_baseline,
        hybrid_pair_max_pair_history=args.hybrid_pair_max_pair_history,
        hybrid_pair_reproj_thresh_px=args.hybrid_pair_reproj_thresh,
        refine_min_views=args.refine_min_views,
        refine_max_iters=args.refine_max_iters,
        refine_huber_px=args.refine_huber_px,
        refine_rmse_thresh_px=args.refine_rmse_thresh,
        refine_max_depth_shift_ratio=args.refine_max_depth_shift_ratio,
        current_reproj_thresh_px=args.current_reproj_thresh,
        max_reproj_thresh_px=args.max_reproj_thresh,
        gt_epi_thresh_px=args.gt_epi_thresh,
        min_parallax_deg=args.min_parallax_deg,
        min_baseline_m=args.min_baseline,
        max_pair_history=args.max_pair_history,
        triang_confirmed_only=args.triang_confirmed_only,
        triang_min_hits=args.triang_min_hits,
        triang_include_reacquired=args.triang_include_reacquired,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        reproj_thresh_px=args.reproj_thresh,
        lidar_radius_px=args.lidar_radius_px,
        lidar_match_mode=args.lidar_match_mode,
        min_lidar_depth_m=args.min_lidar_depth,
        max_lidar_depth_m=args.max_lidar_depth,
        max_lidar_vis_points=args.max_lidar_vis_points,
        lidar_click_radius_px=args.lidar_click_radius,
        pose_eval_gap=args.pose_eval_gap,
        pose_min_pairs=args.pose_min_pairs,
        pose_ransac_thresh_px=args.pose_ransac_thresh,
        pose_ransac_prob=args.pose_ransac_prob,
        pose_ransac_max_iters=args.pose_ransac_max_iters,
        log_csv=False,
        log_path=None,
        draw_max_tracks=args.draw_max_tracks,
        point_radius=2,
        side_panel_width=400,
        side_panel_height=450,
        resize=1.0,
        mouse_coordinate_mode="raw",
        path_len=15,
        inspect_radius_px=12.0,
    )


def prepare_lidar_projection(args: argparse.Namespace):
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
            # Useful for quick odometry-style runs where --calib has P0/P1/P2/P3
            # but the raw cam-to-cam rectification file was not supplied. The
            # preferred raw KITTI path is still to pass --calib-cam-to-cam too,
            # because that gives the true rectification matrix and P_rect_00.
            print("[warn] --calib-velo-to-cam was provided without --calib-cam-to-cam; using P0 from --calib and identity rectification.")
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


def active_depth_points(
    manager: FeatureManager,
    frame: int,
    image_shape: Tuple[int, int],
    triang_infos: Dict[int, TriangInfo],
    all_active_depths: bool,
):
    if all_active_depths:
        tracks = manager.active_tracks(frame, confirmed_only=False)
    else:
        tracks = manager.select_tracks_for_display(frame, image_shape, confirmed_only=False)

    rows = []
    uv = []
    z = []
    for tr in tracks:
        info = triang_infos.get(tr.id)
        if info is None or info.label not in GOOD_LABELS or not np.isfinite(info.depth_m):
            continue
        pt = tr.last_pt()
        uv.append(pt.astype(np.float64))
        z.append(float(info.depth_m))
        rows.append({
            "track_id": tr.id,
            "u": float(pt[0]),
            "v": float(pt[1]),
            "depth_m": float(info.depth_m),
            "past_frame": -1 if info.past_frame is None else int(info.past_frame),
            "gap": int(info.gap),
            "triang_label": info.label,
            "confirmed": bool(info.confirmed),
            "reacquired": bool(info.reacquired),
            "hit_count": int(info.hit_count),
            "source": info.source,
            "epi_px": finite_float(info.epi_px),
            "parallax_deg": finite_float(info.parallax_deg),
            "baseline_m": finite_float(info.baseline_m),
            "reproj_px": finite_float(info.reproj_px),
            "triang_method": info.method,
            "triang_used_views": int(info.used_views),
            "triang_inlier_views": int(info.inlier_views),
            "last_desc_dist": finite_float(tr.last_desc_dist),
            "last_ratio": finite_float(tr.last_ratio),
            "last_lk_fb_err": finite_float(getattr(tr, "last_lk_fb_err", np.nan)),
            "last_lk_err": finite_float(getattr(tr, "last_lk_err", np.nan)),
            "last_desc_refresh_dist": finite_float(getattr(tr, "last_desc_refresh_dist", np.nan)),
            "last_desc_refresh_spatial": finite_float(getattr(tr, "last_desc_refresh_spatial", np.nan)),
            "frames_since_desc_refresh": int(getattr(tr, "frames_since_desc_refresh", 0)),
            "quality_score": finite_float(tr.quality_score),
        })

    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2) if uv else np.empty((0, 2), dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64) if z else np.empty((0,), dtype=np.float64)
    return uv_arr, z_arr, rows


def summarize_triang(triang_infos: Dict[int, TriangInfo]) -> Dict[str, float]:
    labels = {}
    good_par = []
    good_rep = []
    good_gap = []
    for info in triang_infos.values():
        labels[info.label] = labels.get(info.label, 0) + 1
        if info.label in GOOD_LABELS:
            good_par.append(info.parallax_deg)
            good_rep.append(info.reproj_px)
            good_gap.append(info.gap)
    return {
        "triang_good": int(sum(labels.get(k, 0) for k in GOOD_LABELS)),
        "triang_confirmed_good": int(labels.get("confirmed_good", 0)),
        "triang_candidate_good": int(labels.get("candidate_good", 0)),
        "triang_reacq_good": int(labels.get("reacq_good", 0)),
        "triang_bad_epi": int(labels.get("bad_epi", 0)),
        "triang_low_parallax": int(labels.get("low_parallax", 0)),
        "triang_bad_reproj": int(labels.get("bad_reproj", 0)),
        "triang_bad_depth": int(labels.get("bad_depth", 0)),
        "triang_no_pair": int(labels.get("no_pair", 0)),
        "median_parallax_deg": safe_median(good_par),
        "median_reproj_px": safe_median(good_rep),
        "median_gap": safe_median(good_gap),
    }


def compute_lidar_metrics(args, lidar_projection, frame: int, image_shape, sparse_uv, sparse_z):
    empty = {
        "num_projected_lidar_points": 0,
        "num_lidar_matched": 0,
        "num_lidar_covered": 0,
        "gt_coverage": 0.0,
        "lidar_match_rate": 0.0,
        "median_lidar_pixel_distance_px": float("nan"),
        "median_Z_tri_m": float("nan"),
        "median_Z_lidar_m": float("nan"),
        "raw_median_abs_err_m": float("nan"),
        "raw_mean_abs_err_m": float("nan"),
        "raw_rmse_m": float("nan"),
        "raw_median_rel_err": float("nan"),
        "raw_mean_rel_err": float("nan"),
        "raw_delta_10": float("nan"),
        "raw_delta_20": float("nan"),
        "raw_delta_30": float("nan"),
        "raw_delta_125": float("nan"),
        "alpha_l2": float("nan"),
        "alpha_median_ratio": float("nan"),
        "median_scaled_median_abs_err_m": float("nan"),
        "median_scaled_mean_abs_err_m": float("nan"),
        "median_scaled_rmse_m": float("nan"),
        "median_scaled_median_rel_err": float("nan"),
        "median_scaled_mean_rel_err": float("nan"),
        "median_scaled_delta_10": float("nan"),
        "median_scaled_delta_20": float("nan"),
        "median_scaled_delta_30": float("nan"),
        "median_scaled_delta_125": float("nan"),
        "lidar_status": "disabled",
    }
    empty.update(coverage_stats("lidar_matched_depth", np.empty((0, 2), dtype=np.float64), image_shape, args.grid_cols, args.grid_rows))
    if lidar_projection is None:
        return empty, None

    lidar_path = args.velodyne_dir / f"{frame:0{args.lidar_digits}d}.bin"
    if not lidar_path.exists():
        empty["lidar_status"] = f"missing:{lidar_path}"
        return empty, None

    T_cam0_velo, R_rect_4, P_rect = lidar_projection
    velo = load_velodyne_bin(lidar_path)
    lidar_uv, lidar_z, _ = project_velodyne_to_image(
        velo,
        T_cam0_velo,
        R_rect_4,
        P_rect,
        image_shape=image_shape,
        min_depth_m=args.min_lidar_depth,
        max_depth_m=args.max_lidar_depth,
    )

    matched, matched_z, matched_dist = match_sparse_to_lidar_with_radius(
        sparse_uv,
        lidar_uv,
        lidar_z,
        radius_px=args.lidar_radius_px,
        mode=args.lidar_match_mode,
    )
    # Reverse query for GT coverage: which projected LiDAR points have a sparse
    # depth estimate within radius. Answers "what fraction of measurable ground
    # truth did we actually produce depth for", independent of image-grid coverage.
    lidar_covered, _, _ = match_sparse_to_lidar_with_radius(
        lidar_uv,
        sparse_uv,
        sparse_z,
        radius_px=args.lidar_radius_px,
        mode=args.lidar_match_mode,
    )
    num_lidar_covered = int(np.sum(lidar_covered))
    matched_uv = sparse_uv[matched]
    z_pred = sparse_z[matched]
    z_gt = matched_z[matched]
    raw = compute_depth_metrics(z_pred, z_gt)
    alpha_l2, alpha_med = compute_scale_factors(z_pred, z_gt)
    med_scaled = compute_depth_metrics(alpha_med * z_pred, z_gt) if np.isfinite(alpha_med) else compute_depth_metrics([], [])

    out = {
        "num_projected_lidar_points": int(len(lidar_z)),
        "num_lidar_matched": int(np.sum(matched)),
        "num_lidar_covered": num_lidar_covered,
        "gt_coverage": float(num_lidar_covered / max(len(lidar_z), 1)),
        "lidar_match_rate": float(np.sum(matched) / max(len(sparse_z), 1)),
        "median_lidar_pixel_distance_px": safe_median(matched_dist[matched]),
        "median_Z_tri_m": safe_median(z_pred),
        "median_Z_lidar_m": safe_median(z_gt),
        "raw_median_abs_err_m": raw["median_abs_err_m"],
        "raw_mean_abs_err_m": raw["mean_abs_err_m"],
        "raw_rmse_m": raw["rmse_m"],
        "raw_median_rel_err": raw["median_rel_err"],
        "raw_mean_rel_err": raw["mean_rel_err"],
        "raw_delta_10": raw["delta_10"],
        "raw_delta_20": raw["delta_20"],
        "raw_delta_30": raw["delta_30"],
        "raw_delta_125": raw["delta_125"],
        "alpha_l2": alpha_l2,
        "alpha_median_ratio": alpha_med,
        "median_scaled_median_abs_err_m": med_scaled["median_abs_err_m"],
        "median_scaled_mean_abs_err_m": med_scaled["mean_abs_err_m"],
        "median_scaled_rmse_m": med_scaled["rmse_m"],
        "median_scaled_median_rel_err": med_scaled["median_rel_err"],
        "median_scaled_mean_rel_err": med_scaled["mean_rel_err"],
        "median_scaled_delta_10": med_scaled["delta_10"],
        "median_scaled_delta_20": med_scaled["delta_20"],
        "median_scaled_delta_30": med_scaled["delta_30"],
        "median_scaled_delta_125": med_scaled["delta_125"],
        "lidar_status": "ok" if np.sum(matched) > 0 else "no_matches",
    }
    # This coverage is intentionally computed only on points that could be
    # compared against LiDAR. It answers: "where in the image do I have measured
    # depth quality feedback?", which can be much narrower than final depth map
    # coverage because KITTI LiDAR projections are mostly lower-half/road-biased.
    out.update(coverage_stats("lidar_matched_depth", matched_uv, image_shape, args.grid_cols, args.grid_rows))
    matched_payload = {
        "matched": matched,
        "matched_z": matched_z,
        "matched_dist": matched_dist,
        "sparse_uv": sparse_uv,
        "sparse_z": sparse_z,
        "lidar_uv": lidar_uv,
        "lidar_z": lidar_z,
    }
    return out, matched_payload


def pose_row(pose: PoseEval) -> Dict[str, float]:
    return {
        "pose_ok": bool(pose.ok),
        "pose_reason": pose.reason,
        "pose_gap": int(pose.gap),
        "pose_pairs": int(pose.pairs),
        "pose_inliers": int(pose.inliers),
        "pose_inlier_ratio": float(pose.inliers / pose.pairs) if pose.pairs > 0 else float("nan"),
        "pose_rot_err_deg": finite_float(pose.rot_err_deg),
        "pose_t_dir_err_deg": finite_float(pose.t_err_deg),
    }


def write_csv_rows(path: Path, rows: List[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows: List[dict]) -> Dict[str, float]:
    def col(name):
        return [r.get(name, np.nan) for r in rows if np.isfinite(finite_float(r.get(name, np.nan)))]

    keys = [
        "num_sparse_depth",
        "num_lidar_matched",
        "num_projected_lidar_points",
        "num_lidar_covered",
        "gt_coverage",
        "lidar_match_rate",
        "depth_coverage",
        "depth_occupied_cells",
        "depth_mean_pts_per_occ_cell",
        "depth_max_cell_count",
        "depth_spatial_entropy",
        "lidar_matched_depth_coverage",
        "lidar_matched_depth_occupied_cells",
        "lidar_matched_depth_spatial_entropy",
        "raw_median_abs_err_m",
        "raw_mean_abs_err_m",
        "raw_rmse_m",
        "raw_median_rel_err",
        "raw_mean_rel_err",
        "raw_delta_10",
        "raw_delta_20",
        "raw_delta_30",
        "depthbin_0_10m_count",
        "depthbin_0_10m_median_abs_err_m",
        "depthbin_0_10m_median_rel_err",
        "depthbin_10_20m_count",
        "depthbin_10_20m_median_abs_err_m",
        "depthbin_10_20m_median_rel_err",
        "depthbin_20_40m_count",
        "depthbin_20_40m_median_abs_err_m",
        "depthbin_20_40m_median_rel_err",
        "depthbin_40_80m_count",
        "depthbin_40_80m_median_abs_err_m",
        "depthbin_40_80m_median_rel_err",
        "depthbin_80_120m_count",
        "depthbin_80_120m_median_abs_err_m",
        "depthbin_80_120m_median_rel_err",
        "median_scaled_median_abs_err_m",
        "median_scaled_median_rel_err",
        "pose_pairs",
        "pose_inliers",
        "pose_inlier_ratio",
        "pose_rot_err_deg",
        "pose_t_dir_err_deg",
        "lk_attempted",
        "lk_accepted",
        "lk_accept_rate",
        "lk_desc_refreshed",
        "lk_median_fb_err",
        "lk_median_err",
        "lk_median_refresh_desc",
        "time_lk_track_ms",
        "time_step_ms",
        "time_manage_depth_ms",
        "time_triang_ms",
        "time_total_no_draw_ms",
    ]
    out = {"num_frames": len(rows)}
    for k in keys:
        out[f"median_{k}"] = safe_median(col(k))
        out[f"mean_{k}"] = safe_mean(col(k))
    return out


def save_metric_plot(rows: List[dict], output_dir: Path, metrics: Sequence[Tuple[str, str]], filename: str, title: str):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray([r["frame"] for r in rows], dtype=np.int32)
    series = [(key, label, np.asarray([finite_float(r.get(key, np.nan)) for r in rows], dtype=np.float64)) for key, label in metrics]

    if not getattr(save_metric_plot, "_matplotlib_broken", False):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 4.8))
            for _, label, vals in series:
                plt.plot(frames, vals, marker=".", linewidth=1.2, label=label)
            plt.xlabel("frame")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=160)
            plt.close()
            return
        except Exception as e:
            print(f"[warn] Matplotlib unavailable; using simple OpenCV plots ({e})")
            setattr(save_metric_plot, "_matplotlib_broken", True)

    w, h = 1100, 520
    margin_l, margin_r, margin_t, margin_b = 82, 28, 48, 68
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b
    cv2.rectangle(img, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (220, 220, 220), 1)
    cv2.putText(img, title, (margin_l, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
    if frames.size == 0:
        cv2.imwrite(str(plots_dir / filename), img)
        return

    all_vals = np.concatenate([vals[np.isfinite(vals)] for _, _, vals in series if np.any(np.isfinite(vals))]) if any(np.any(np.isfinite(vals)) for _, _, vals in series) else np.array([])
    if all_vals.size == 0:
        cv2.putText(img, "no finite values", (margin_l + 18, margin_t + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.imwrite(str(plots_dir / filename), img)
        return
    y_min = float(np.nanmin(all_vals))
    y_max = float(np.nanmax(all_vals))
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1.0
        y_max += 1.0
    x_min = int(frames.min())
    x_max = int(frames.max())
    if x_max == x_min:
        x_max = x_min + 1

    for k in range(5):
        y = margin_t + int(round(k * plot_h / 4))
        cv2.line(img, (margin_l, y), (margin_l + plot_w, y), (235, 235, 235), 1)
    cv2.putText(img, f"{y_max:.3g}", (8, margin_t + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(img, f"{y_min:.3g}", (8, margin_t + plot_h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(img, f"frame {x_min}", (margin_l, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(img, f"{x_max}", (margin_l + plot_w - 48, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)

    colors = [(30, 90, 220), (30, 160, 30), (190, 80, 25), (160, 40, 160)]
    for s_i, (_, label, vals) in enumerate(series):
        pts = []
        for f, v in zip(frames, vals):
            if not np.isfinite(v):
                continue
            x = margin_l + int(round((int(f) - x_min) / max(1, x_max - x_min) * plot_w))
            y = margin_t + plot_h - int(round((float(v) - y_min) / (y_max - y_min) * plot_h))
            pts.append((x, y))
        color = colors[s_i % len(colors)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)
        lx = margin_l + 16 + s_i * 245
        ly = margin_t + plot_h + 34
        cv2.line(img, (lx, ly), (lx + 22, ly), color, 3, cv2.LINE_AA)
        cv2.putText(img, label[:24], (lx + 30, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (45, 45, 45), 1, cv2.LINE_AA)

    cv2.imwrite(str(plots_dir / filename), img)


def save_relative_error_visualization(
    image_gray: np.ndarray,
    frame: int,
    matched_payload: Optional[dict],
    output_dir: Path,
    rel_vmax: float,
    abs_vmax: float,
    max_points: int,
    show_lidar_points: bool,
):
    """Save image-space relative and absolute error maps for one frame.

    The sparse depth result is already a set of isolated image points. A normal
    dense heatmap would be misleading, so this visualization keeps the geometry
    honest:
      - faint gray dots: accepted final sparse depth points, even if no LiDAR
        match was found nearby;
      - colored dots: sparse depth points with a LiDAR match, colored by
        either |Z_pred - Z_lidar| / Z_lidar or |Z_pred - Z_lidar|;
      - optional tiny cyan dots: projected raw LiDAR samples, to make the
        evaluation support visible.
    """
    if matched_payload is None:
        return

    sparse_uv = np.asarray(matched_payload["sparse_uv"], dtype=np.float64)
    sparse_z = np.asarray(matched_payload["sparse_z"], dtype=np.float64)
    matched = np.asarray(matched_payload["matched"], dtype=bool)
    matched_z = np.asarray(matched_payload["matched_z"], dtype=np.float64)
    lidar_uv = np.asarray(matched_payload.get("lidar_uv", np.empty((0, 2))), dtype=np.float64)

    if sparse_uv.size == 0:
        return

    rel_dir = output_dir / "visualizations" / "relative_error_maps"
    abs_dir = output_dir / "visualizations" / "absolute_error_maps"
    rel_dir.mkdir(parents=True, exist_ok=True)
    abs_dir.mkdir(parents=True, exist_ok=True)

    if max_points > 0 and len(sparse_uv) > max_points:
        draw_idx = np.linspace(0, len(sparse_uv) - 1, max_points).astype(int)
    else:
        draw_idx = np.arange(len(sparse_uv), dtype=int)

    matched_idx = np.where(matched)[0]
    if max_points > 0 and len(matched_idx) > max_points:
        matched_idx = matched_idx[np.linspace(0, len(matched_idx) - 1, max_points).astype(int)]

    rel = np.full(len(sparse_uv), np.nan, dtype=np.float64)
    abs_err = np.full(len(sparse_uv), np.nan, dtype=np.float64)
    valid_rel = matched & np.isfinite(sparse_z) & np.isfinite(matched_z) & (matched_z > 0)
    abs_err[valid_rel] = np.abs(sparse_z[valid_rel] - matched_z[valid_rel])
    rel[valid_rel] = abs_err[valid_rel] / np.maximum(matched_z[valid_rel], 1e-12)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def save_one(values: np.ndarray, vmax: float, label: str, out_dir: Path, suffix: str):
            plt.figure(figsize=(12, 4.2))
            plt.imshow(image_gray, cmap="gray", vmin=0, vmax=255)

            if show_lidar_points and lidar_uv.size:
                lid = lidar_uv
                if max_points > 0 and len(lid) > max_points:
                    lid = lid[np.linspace(0, len(lid) - 1, max_points).astype(int)]
                plt.scatter(lid[:, 0], lid[:, 1], s=1.0, c="#22d3ee", alpha=0.22, linewidths=0, label="projected LiDAR")

            unjudged = draw_idx[~matched[draw_idx]]
            if len(unjudged):
                plt.scatter(sparse_uv[unjudged, 0], sparse_uv[unjudged, 1], s=5.0, c="white", alpha=0.35, linewidths=0, label="no LiDAR match")

            judged = matched_idx[np.isfinite(values[matched_idx])]
            if len(judged):
                sc = plt.scatter(
                    sparse_uv[judged, 0],
                    sparse_uv[judged, 1],
                    c=np.clip(values[judged], 0.0, vmax),
                    s=11.0,
                    cmap="turbo",
                    vmin=0.0,
                    vmax=vmax,
                    edgecolors="black",
                    linewidths=0.2,
                    label="LiDAR-matched sparse depth",
                )
                cbar = plt.colorbar(sc, fraction=0.024, pad=0.012)
                cbar.set_label(label)

            plt.title(f"frame {frame:06d} sparse-depth {suffix.replace('_', ' ')}")
            plt.xlim(0, image_gray.shape[1])
            plt.ylim(image_gray.shape[0], 0)
            plt.tight_layout()
            plt.savefig(out_dir / f"frame{frame:06d}_{suffix}.png", dpi=170)
            plt.close()

        save_one(rel, rel_vmax, "relative depth error |Z_pred - Z_lidar| / Z_lidar", rel_dir, "relative_error")
        save_one(abs_err, abs_vmax, "absolute depth error |Z_pred - Z_lidar| [m]", abs_dir, "absolute_error")
        return
    except Exception as e:
        print(f"[warn] Could not save Matplotlib error maps for frame {frame}: {e}")

    # Minimal fallback: draw colored circles directly with OpenCV. This path is
    # mainly for broken plotting environments; normal cv-env runs use Matplotlib.
    def save_cv(values: np.ndarray, vmax: float, out_dir: Path, suffix: str):
        canvas = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        for idx in draw_idx:
            x, y = np.round(sparse_uv[idx]).astype(int)
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.circle(canvas, (x, y), 2, (220, 220, 220), -1, cv2.LINE_AA)
        for idx in matched_idx:
            if not np.isfinite(values[idx]):
                continue
            x, y = np.round(sparse_uv[idx]).astype(int)
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                val = int(np.clip(values[idx] / max(vmax, 1e-12), 0.0, 1.0) * 255)
                color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0].tolist()
                cv2.circle(canvas, (x, y), 3, tuple(int(c) for c in color), -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"frame{frame:06d}_{suffix}.png"), canvas)

    save_cv(rel, rel_vmax, rel_dir, "relative_error")
    save_cv(abs_err, abs_vmax, abs_dir, "absolute_error")


def collect_matched_point_diagnostics(frame: int, frame_point_rows: List[dict], matched_payload: Optional[dict]) -> List[dict]:
    """Build the matched-point table used by threshold-selection diagnostics.

    Each row is one accepted sparse-depth point that also found a nearby LiDAR
    projection. These rows are the raw material for plots like error-vs-parallax
    and error-vs-reprojection. Keeping them as rows, rather than only per-frame
    medians, lets us ask threshold questions directly from the data.
    """
    if matched_payload is None or not frame_point_rows:
        return []

    matched = np.asarray(matched_payload["matched"], dtype=bool)
    matched_z = np.asarray(matched_payload["matched_z"], dtype=np.float64)
    matched_dist = np.asarray(matched_payload["matched_dist"], dtype=np.float64)
    out = []
    for p_i, base in enumerate(frame_point_rows):
        if p_i >= len(matched) or not matched[p_i]:
            continue
        row = dict(base)
        row["frame"] = int(frame)
        row["lidar_matched"] = True
        row["lidar_depth_m"] = finite_float(matched_z[p_i])
        row["lidar_pixel_dist_px"] = finite_float(matched_dist[p_i])
        row["depth_abs_err_m"] = abs(row["depth_m"] - row["lidar_depth_m"])
        row["depth_rel_err"] = (
            row["depth_abs_err_m"] / max(row["lidar_depth_m"], 1e-12)
            if np.isfinite(row["lidar_depth_m"]) and row["lidar_depth_m"] > 0
            else float("nan")
        )
        out.append(row)
    return out


def bin_label(lo: float, hi: float) -> str:
    lo_s = f"{lo:g}"
    hi_s = "inf" if np.isinf(hi) else f"{hi:g}"
    return f"{lo_s}_{hi_s}"


def point_metric_stats(rows: List[dict]) -> Dict[str, float]:
    """Summarize a point subset using both typical and outlier-sensitive errors."""
    abs_err = np.asarray([finite_float(r.get("depth_abs_err_m", np.nan)) for r in rows], dtype=np.float64)
    rel_err = np.asarray([finite_float(r.get("depth_rel_err", np.nan)) for r in rows], dtype=np.float64)
    pred = np.asarray([finite_float(r.get("depth_m", np.nan)) for r in rows], dtype=np.float64)
    lidar = np.asarray([finite_float(r.get("lidar_depth_m", np.nan)) for r in rows], dtype=np.float64)
    par = np.asarray([finite_float(r.get("parallax_deg", np.nan)) for r in rows], dtype=np.float64)
    rep = np.asarray([finite_float(r.get("reproj_px", np.nan)) for r in rows], dtype=np.float64)
    gap = np.asarray([finite_float(r.get("gap", np.nan)) for r in rows], dtype=np.float64)

    abs_f = abs_err[np.isfinite(abs_err)]
    rel_f = rel_err[np.isfinite(rel_err)]
    return {
        "count": float(len(rows)),
        "median_abs_err_m": safe_median(abs_f),
        "mean_abs_err_m": safe_mean(abs_f),
        "rmse_m": float(np.sqrt(np.mean(abs_f * abs_f))) if abs_f.size else float("nan"),
        "median_rel_err": safe_median(rel_f),
        "mean_rel_err": safe_mean(rel_f),
        "delta_10": float(np.mean(rel_f <= 0.10)) if rel_f.size else float("nan"),
        "delta_20": float(np.mean(rel_f <= 0.20)) if rel_f.size else float("nan"),
        "delta_30": float(np.mean(rel_f <= 0.30)) if rel_f.size else float("nan"),
        "median_depth_m": safe_median(pred),
        "median_lidar_depth_m": safe_median(lidar),
        "median_parallax_deg": safe_median(par),
        "median_reproj_px": safe_median(rep),
        "median_gap": safe_median(gap),
    }


def summarize_point_bins(point_rows: List[dict], output_dir: Path) -> List[dict]:
    """Write binned point diagnostics for choosing gates and thresholds.

    These summaries are deliberately run-level rather than per-frame. For tuning
    gates like min parallax or reprojection threshold, the question is usually:
    "Across all LiDAR-evaluable points, where does error start getting bad?"
    """
    if not point_rows:
        return []

    specs = [
        ("lidar_depth_m", "lidar_depth_m", [(0, 10), (10, 20), (20, 40), (40, 80), (80, 120), (120, np.inf)]),
        ("parallax_deg", "parallax_deg", [(0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.0), (1.0, np.inf)]),
        ("reproj_px", "reproj_px", [(0, 0.25), (0.25, 0.50), (0.50, 1.0), (1.0, 2.0), (2.0, np.inf)]),
        ("last_desc_dist", "desc_dist", [(0, 100), (100, 200), (200, 280), (280, 320), (320, np.inf)]),
        ("last_ratio", "lowe_ratio", [(0, 0.50), (0.50, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.0), (1.0, np.inf)]),
    ]

    summary_rows: List[dict] = []
    for key, family, bins in specs:
        values = np.asarray([finite_float(r.get(key, np.nan)) for r in point_rows], dtype=np.float64)
        for lo, hi in bins:
            if np.isinf(hi):
                mask = np.isfinite(values) & (values >= lo)
            else:
                mask = np.isfinite(values) & (values >= lo) & (values < hi)
            subset = [r for r, keep in zip(point_rows, mask) if keep]
            row = {
                "family": family,
                "bin": bin_label(lo, hi),
                "lo": float(lo),
                "hi": float(hi),
            }
            row.update(point_metric_stats(subset))
            summary_rows.append(row)

    gaps = sorted({int(r["gap"]) for r in point_rows if np.isfinite(finite_float(r.get("gap", np.nan)))})
    for gap in gaps:
        subset = [r for r in point_rows if int(r.get("gap", -999)) == gap]
        row = {"family": "gap", "bin": str(gap), "lo": float(gap), "hi": float(gap)}
        row.update(point_metric_stats(subset))
        summary_rows.append(row)

    sources = sorted({str(r.get("source", "unknown")) for r in point_rows})
    for source in sources:
        subset = [r for r in point_rows if str(r.get("source", "unknown")) == source]
        row = {"family": "source", "bin": source, "lo": float("nan"), "hi": float("nan")}
        row.update(point_metric_stats(subset))
        summary_rows.append(row)

    write_csv_rows(output_dir / "point_diagnostics_binned_summary.csv", summary_rows)
    return summary_rows


def save_point_scatter_plots(point_rows: List[dict], output_dir: Path):
    """Save run-level point diagnostic plots.

    These plots answer threshold questions:
      - parallax vs error -> where should --min-parallax-deg be?
      - reprojection vs error -> is --reproj-thresh too loose/tight?
      - descriptor distance / ratio vs error -> are matching gates admitting bad
        depth points?
    """
    if not point_rows:
        return
    plots_dir = output_dir / "plots" / "point_diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] Could not save point diagnostic plots because Matplotlib is unavailable: {e}")
        return

    def arr(key: str) -> np.ndarray:
        return np.asarray([finite_float(r.get(key, np.nan)) for r in point_rows], dtype=np.float64)

    x_specs = [
        ("lidar_depth_m", "LiDAR depth [m]", "error_vs_lidar_depth"),
        ("parallax_deg", "selected pair parallax [deg]", "error_vs_parallax"),
        ("reproj_px", "DLT reprojection RMSE [px]", "error_vs_reprojection"),
        ("last_desc_dist", "accepted SIFT descriptor distance", "error_vs_descriptor_distance"),
        ("last_ratio", "accepted Lowe ratio", "error_vs_lowe_ratio"),
        ("gap", "selected triangulation frame gap", "error_vs_pair_gap"),
        ("last_lk_fb_err", "LK forward-backward error [px]", "error_vs_lk_fb_err"),
        ("last_lk_err", "LK OpenCV error", "error_vs_lk_error"),
        ("last_desc_refresh_dist", "LK descriptor refresh distance", "error_vs_lk_desc_refresh_dist"),
        ("frames_since_desc_refresh", "frames since descriptor refresh", "error_vs_desc_stale_age"),
    ]

    y_specs = [
        ("depth_abs_err_m", "absolute depth error [m]", "abs"),
        ("depth_rel_err", "relative depth error", "rel"),
    ]

    for x_key, x_label, stem in x_specs:
        x = arr(x_key)
        for y_key, y_label, y_stem in y_specs:
            y = arr(y_key)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 2:
                continue
            plt.figure(figsize=(8, 5))
            if np.sum(mask) >= 400:
                hb = plt.hexbin(x[mask], y[mask], gridsize=45, bins="log", mincnt=1, cmap="viridis")
                plt.colorbar(hb, label="log10(point count)")
            else:
                plt.scatter(x[mask], y[mask], s=9, alpha=0.45)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"{y_label} vs {x_label}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{stem}_{y_stem}.png", dpi=160)
            plt.close()


def depth_bin_frame_metrics(point_rows: List[dict]) -> Dict[str, float]:
    """Per-frame depth-bin errors for quick near/mid/far inspection.

    The bins use LiDAR depth as the reference, because it is the measured target
    we compare against. These columns are intentionally compact and repeated per
    frame, so plots/spreadsheets can show whether the near range behaves
    differently from far points over time.
    """
    bins = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 120)]
    out: Dict[str, float] = {}
    if not point_rows:
        for lo, hi in bins:
            prefix = f"depthbin_{lo:g}_{hi:g}m"
            out[f"{prefix}_count"] = 0.0
            out[f"{prefix}_median_abs_err_m"] = float("nan")
            out[f"{prefix}_mean_abs_err_m"] = float("nan")
            out[f"{prefix}_median_rel_err"] = float("nan")
        return out

    z = np.asarray([finite_float(r.get("lidar_depth_m", np.nan)) for r in point_rows], dtype=np.float64)
    for lo, hi in bins:
        mask = np.isfinite(z) & (z >= lo) & (z < hi)
        subset = [r for r, keep in zip(point_rows, mask) if keep]
        stats = point_metric_stats(subset)
        prefix = f"depthbin_{lo:g}_{hi:g}m"
        out[f"{prefix}_count"] = stats["count"]
        out[f"{prefix}_median_abs_err_m"] = stats["median_abs_err_m"]
        out[f"{prefix}_mean_abs_err_m"] = stats["mean_abs_err_m"]
        out[f"{prefix}_median_rel_err"] = stats["median_rel_err"]
    return out


def save_frame_tradeoff_plots(rows: List[dict], output_dir: Path):
    """Plot frame-level coverage/quality tradeoffs.

    This answers a different question than the point diagnostics. Instead of
    "which points are bad?", it asks: "when a frame covers more of the image,
    does the depth quality stay acceptable?"
    """
    if not rows:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] Could not save coverage-quality plots because Matplotlib is unavailable: {e}")
        return

    x = np.asarray([finite_float(r.get("depth_coverage", np.nan)) for r in rows], dtype=np.float64)
    y_abs = np.asarray([finite_float(r.get("raw_median_abs_err_m", np.nan)) for r in rows], dtype=np.float64)
    y_rel = np.asarray([finite_float(r.get("raw_median_rel_err", np.nan)) for r in rows], dtype=np.float64)
    c = np.asarray([finite_float(r.get("num_lidar_matched", np.nan)) for r in rows], dtype=np.float64)

    for y, ylabel, name in [
        (y_abs, "median absolute depth error [m]", "coverage_vs_median_abs_error.png"),
        (y_rel, "median relative depth error", "coverage_vs_median_rel_error.png"),
    ]:
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            continue
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(x[mask], y[mask], c=c[mask], s=38, cmap="viridis", edgecolors="black", linewidths=0.25)
        plt.colorbar(sc, label="LiDAR-matched depth points")
        plt.xlabel("final sparse-depth coverage")
        plt.ylabel(ylabel)
        plt.title("Frame coverage vs depth quality")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / name, dpi=160)
        plt.close()


def save_all_plots(rows: List[dict], output_dir: Path):
    save_metric_plot(
        rows,
        output_dir,
        [
            ("raw_median_abs_err_m", "median abs [m]"),
            ("raw_mean_abs_err_m", "mean abs [m]"),
            ("raw_rmse_m", "RMSE [m]"),
        ],
        "depth_abs_error_by_frame.png",
        "Depth Error vs LiDAR",
    )
    save_metric_plot(
        rows,
        output_dir,
        [
            ("raw_median_rel_err", "median rel"),
            ("raw_mean_rel_err", "mean rel"),
            ("median_scaled_median_rel_err", "median-scale median rel"),
        ],
        "depth_rel_error_by_frame.png",
        "Relative Depth Error vs LiDAR",
    )
    save_metric_plot(
        rows,
        output_dir,
        [
            ("depthbin_0_10m_median_abs_err_m", "0-10 m"),
            ("depthbin_10_20m_median_abs_err_m", "10-20 m"),
            ("depthbin_20_40m_median_abs_err_m", "20-40 m"),
            ("depthbin_40_80m_median_abs_err_m", "40-80 m"),
            ("depthbin_80_120m_median_abs_err_m", "80-120 m"),
        ],
        "depth_bin_abs_error_by_frame.png",
        "Median Absolute Error by LiDAR Depth Bin",
    )
    save_metric_plot(
        rows,
        output_dir,
        [
            ("depthbin_0_10m_count", "0-10 m"),
            ("depthbin_10_20m_count", "10-20 m"),
            ("depthbin_20_40m_count", "20-40 m"),
            ("depthbin_40_80m_count", "40-80 m"),
            ("depthbin_80_120m_count", "80-120 m"),
        ],
        "depth_bin_counts_by_frame.png",
        "LiDAR-Matched Point Counts by Depth Bin",
    )
    save_metric_plot(
        rows,
        output_dir,
        [
            ("num_sparse_depth", "sparse depths"),
            ("num_lidar_matched", "LiDAR matched"),
        ],
        "point_counts_by_frame.png",
        "Sparse Depth and LiDAR-Matched Point Counts",
    )
    save_metric_plot(
        rows,
        output_dir,
        [
            ("depth_coverage", "final depth coverage"),
            ("lidar_matched_depth_coverage", "LiDAR-matched coverage"),
            ("depth_spatial_entropy", "depth entropy"),
        ],
        "spatial_coverage_by_frame.png",
        "Sparse Depth Spatial Coverage",
    )
    save_metric_plot(
        rows,
        output_dir,
        [
            ("pose_rot_err_deg", "rotation err [deg]"),
            ("pose_t_dir_err_deg", "translation direction err [deg]"),
        ],
        "pose_error_by_frame.png",
        "Strict Essential-Matrix RANSAC Pose Error",
    )
    save_metric_plot(
        rows,
        output_dir,
        [
            ("pose_pairs", "pose pairs"),
            ("pose_inliers", "RANSAC inliers"),
        ],
        "pose_inliers_by_frame.png",
        "Pose RANSAC Inliers",
    )
    save_metric_plot(
        rows,
        output_dir,
        [
            ("lk_attempted", "LK attempted"),
            ("lk_accepted", "LK accepted"),
            ("lk_desc_refreshed", "descriptor refreshed"),
        ],
        "lk_counts_by_frame.png",
        "LK Continuation Counts",
    )
    save_metric_plot(
        rows,
        output_dir,
        [
            ("lk_median_fb_err", "median FB [px]"),
            ("lk_median_err", "median LK err"),
            ("lk_median_refresh_desc", "median refresh desc"),
        ],
        "lk_quality_by_frame.png",
        "LK Tracking Quality",
    )


def parse_args():
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
        description="Headless SIFT+LK manager sparse-depth evaluator for KITTI.",
        parents=[config_parser],
    )
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    p.add_argument("--poses", type=Path, default=DEFAULT_POSES_PATH)
    p.add_argument("--image-digits", type=int, default=DEFAULT_IMAGE_DIGITS)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=50)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p.add_argument("--detector", choices=["shi", "sift", "xfeat"], default=None, help="Keypoint detector (new axis).")
    p.add_argument("--descriptor", choices=["sift", "xfeat"], default=None, help="Descriptor/fingerprint type (new axis); xfeat requires --detector xfeat.")
    p.add_argument("--detector-mode", choices=["sift_lk", "shi_sift_lk", "xfeat_sift_lk", "xfeat_native"], default=None, help="[legacy] bundled detector+descriptor; decomposed into --detector/--descriptor.")
    p.add_argument("--shi-max-corners", type=int, default=6000)
    p.add_argument("--shi-quality-level", type=float, default=0.005)
    p.add_argument("--shi-min-distance", type=float, default=7.0)
    p.add_argument("--shi-block-size", type=int, default=7)
    p.add_argument("--xfeat-top-k", type=int, default=4096, help="Max XFeat keypoints per frame (xfeat modes).")
    p.add_argument("--xfeat-detection-threshold", type=float, default=0.05, help="XFeat keypoint score NMS threshold.")
    p.add_argument("--xfeat-desc-size", type=float, default=7.0, help="SIFT descriptor patch size at each XFeat keypoint (xfeat_sift_lk).")
    p.add_argument("--xfeat-native-desc-scale", type=float, default=250.0, help="Scale applied to unit XFeat descriptors so SIFT-tuned distance gates apply (xfeat_native).")

    p.add_argument("--velodyne-dir", type=Path, default=None)
    p.add_argument("--lidar-digits", type=int, default=None)
    p.add_argument("--calib-velo-to-cam", type=Path, default=None)
    p.add_argument("--calib-cam-to-cam", type=Path, default=None)
    p.add_argument("--camera", type=str, default="00")
    p.add_argument("--lidar-radius-px", type=float, default=3.0)
    p.add_argument("--lidar-match-mode", choices=["nearest", "min_depth", "median_depth"], default="nearest")
    p.add_argument("--min-lidar-depth", type=float, default=1.0)
    p.add_argument("--max-lidar-depth", type=float, default=120.0)
    p.add_argument("--max-lidar-vis-points", type=int, default=12000)
    p.add_argument("--lidar-click-radius", type=float, default=2.5)

    p.add_argument("--all-active-depths", action="store_true", help="Use every active good triangulation instead of per-bucket output selection.")
    p.add_argument("--minimal-output", action="store_true", help="Write only summary_metrics.csv, per_frame_metrics.csv, and config.json; skip all plots, per-point diagnostics, and point-bin files. Intended for ablation sweeps.")
    p.add_argument("--save-per-point", action="store_true")
    p.add_argument("--plot-backend", choices=["opencv", "matplotlib", "auto"], default="matplotlib")
    p.add_argument("--save-visualizations", action="store_true", help="Save image-space relative-error overlays for selected frames.")
    p.add_argument("--save-vis-every", type=int, default=10, help="Save one relative-error overlay every N processed frames when --save-visualizations is set.")
    p.add_argument("--error-vmax-rel", type=float, default=1.0, help="Relative-error value mapped to the top of the visualization colorbar.")
    p.add_argument("--error-vmax-abs", type=float, default=10.0, help="Absolute-error value in meters mapped to the top of the visualization colorbar.")
    p.add_argument("--max-vis-points", type=int, default=6000, help="Maximum sparse/LiDAR points drawn per visualization.")
    p.add_argument("--show-lidar-vis-points", action="store_true", help="Also draw projected LiDAR samples as tiny cyan dots in relative-error maps.")

    p.add_argument("--sift-nfeatures", type=int, default=8000)
    p.add_argument("--sift-n-octave-layers", type=int, default=3)
    p.add_argument("--contrast", type=float, default=0.01)
    p.add_argument("--edge", type=float, default=10.0)
    p.add_argument("--sigma", type=float, default=1.6)

    p.add_argument("--soft-spawn", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable (--no-soft-spawn) the soft-SIFT rescue+fill layer (sift_lk only).")
    p.add_argument("--no-soft-rescue", dest="soft_rescue", action="store_false", default=True)
    p.add_argument("--soft-nfeatures", type=int, default=14000)
    p.add_argument("--soft-contrast", type=float, default=0.004)
    p.add_argument("--soft-edge", type=float, default=15.0)
    p.add_argument("--soft-sigma", type=float, default=1.6)
    p.add_argument("--soft-confirm-hits", type=int, default=3)
    p.add_argument("--soft-rescue-search-radius", type=float, default=65.0)
    p.add_argument("--soft-rescue-ratio", type=float, default=0.75)
    p.add_argument("--soft-rescue-max-desc-dist", type=float, default=280.0)
    p.add_argument("--soft-detect-mask-used-primary", type=float, default=0.0)
    p.add_argument("--soft-dup-used-primary", type=float, default=3.0)
    p.add_argument("--soft-spawn-primary-sep", type=float, default=4.0)
    p.add_argument("--soft-source-penalty", type=float, default=0.0)

    p.add_argument("--descriptor-mode", choices=["latest", "anchor"], default="latest")
    p.add_argument("--matcher", "--association-matcher", dest="matcher", choices=["radius_lowe", "xfeat_mnn"], default="radius_lowe", help="radius_lowe = per-track spatial+Lowe NN; xfeat_mnn = XFeat global mutual-NN + cosine.")
    p.add_argument("--xfeat-mnn-min-cossim", type=float, default=0.82, help="Cosine-similarity threshold for xfeat_mnn matches.")
    p.add_argument("--xfeat-mnn-max-step", type=float, default=0.0, help="Max displacement for an MNN match; 0 uses max_step_px.")
    p.add_argument("--prediction", choices=["constant_position", "constant_velocity"], default="constant_velocity")
    p.add_argument("--search-radius", type=float, default=55.0)
    p.add_argument("--ratio", type=float, default=0.80)
    p.add_argument("--max-desc-dist", type=float, default=320.0)
    p.add_argument("--reacq-search-radius", type=float, default=90.0)
    p.add_argument("--reacq-ratio", type=float, default=0.70)
    p.add_argument("--reacq-max-desc-dist", type=float, default=240.0)
    p.add_argument("--no-single-candidate", dest="allow_single_candidate", action="store_false", default=True)
    p.add_argument("--spatial-weight", type=float, default=0.20)
    p.add_argument("--miss-cost", type=float, default=30.0)
    p.add_argument("--max-step-px", type=float, default=250.0)

    # LK continuation. In hybrid mode, LK tries to continue active tracks before
    # descriptor association. SIFT then rescues LK failures and spawns new tracks.
    p.add_argument("--lk-on", "--lk-tracking", dest="lk_on", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable (--no-lk-on) LK-first continuation before descriptor association.")
    p.add_argument("--lk-require-desc-refresh", action="store_true", help="Strict mode: LK points must also find a nearby acceptable SIFT descriptor refresh.")
    p.add_argument("--lk-win-size", type=int, default=21, help="LK window side length in pixels. Larger handles more motion but may smear across edges.")
    p.add_argument("--lk-max-level", type=int, default=3, help="Number of pyramid levels for LK. Higher helps larger motion.")
    p.add_argument("--lk-max-iter", type=int, default=30, help="Maximum LK iterations per pyramid level.")
    p.add_argument("--lk-eps", type=float, default=0.01, help="LK convergence epsilon.")
    p.add_argument("--lk-fb-thresh", type=float, default=1.5, help="Forward-backward consistency threshold in pixels.")
    p.add_argument("--lk-max-step-px", type=float, default=80.0, help="Reject LK tracks moving more than this many pixels in one frame.")
    p.add_argument("--lk-max-error", type=float, default=35.0, help="Reject LK tracks with OpenCV LK error above this value.")
    p.add_argument("--lk-min-eig-thresh", type=float, default=1e-4, help="LK minimum eigenvalue threshold; higher rejects weak/aperture-prone points.")
    p.add_argument("--lk-desc-refresh-radius", type=float, default=4.0, help="Search radius for SIFT descriptor refresh around accepted LK point.")
    p.add_argument("--lk-desc-refresh-max-dist", type=float, default=320.0, help="Max SIFT descriptor distance accepted for LK descriptor refresh.")
    p.add_argument("--detection-period", "--lk-sift-period", dest="detection_period", type=int, default=1)
    p.add_argument("--soft-sift-period", "--lk-soft-period", dest="soft_sift_period", type=int, default=1)
    p.add_argument("--spawn-period", "--lk-spawn-period", dest="spawn_period", type=int, default=1)
    p.add_argument("--force-detection-active-below", "--lk-force-sift-active-below", dest="force_detection_active_below", type=int, default=0)
    p.add_argument("--force-detection-coverage-below", "--lk-force-sift-bucket-coverage-below", dest="force_detection_coverage_below", type=float, default=0.0)
    p.add_argument("--force-soft-underfilled-buckets", "--lk-force-soft-underfilled-buckets", dest="force_soft_underfilled_buckets", type=int, default=0)
    p.add_argument("--lk-epipolar-thresh", type=float, default=0.0)
    p.add_argument("--lk-flow-consistency", action="store_true")
    p.add_argument("--lk-flow-radius", type=float, default=45.0)
    p.add_argument("--lk-flow-min-neighbors", type=int, default=5)
    p.add_argument("--lk-flow-mad-k", type=float, default=3.5)
    p.add_argument("--lk-flow-abs-thresh", type=float, default=8.0)

    p.add_argument("--min-hits-confirm", type=int, default=3)
    p.add_argument("--max-misses", type=int, default=2)
    p.add_argument("--max-active-tracks", type=int, default=3000)
    p.add_argument("--no-duplicate-suppression", dest="duplicate_suppression", action="store_false", default=True)
    p.add_argument("--duplicate-dist", type=float, default=4.0)

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

    p.add_argument("--grid-cols", type=int, default=20)
    p.add_argument("--grid-rows", type=int, default=8)
    p.add_argument("--target-per-bucket", type=int, default=12)
    p.add_argument("--max-per-bucket", type=int, default=16)
    p.add_argument("--output-per-bucket", "--depth-output-per-bucket", dest="output_per_bucket", type=int, default=12)
    p.add_argument("--output-max-tracks", "--depth-output-max-tracks", dest="output_max_tracks", type=int, default=2200)
    p.add_argument("--min-spawn-distance", type=float, default=7.0)
    p.add_argument("--spawn-count-confirmed-only", dest="spawn_count_candidates", action="store_false", default=True)

    p.add_argument("--triangulation-method", choices=["best_pair_dlt", "flow_depth_pair", "refined_pair_dlt", "corrected_pair_dlt", "windowed_multiview_dlt", "refined_multiview_dlt", "hybrid_pair_multiview"], default="best_pair_dlt")
    p.add_argument("--multiview-min-views", type=int, default=3)
    p.add_argument("--hybrid-pair-min-parallax-deg", type=float, default=0.50)
    p.add_argument("--hybrid-pair-min-baseline", type=float, default=0.5)
    p.add_argument("--hybrid-pair-max-history", "--hybrid-pair-max-gap", dest="hybrid_pair_max_pair_history", type=int, default=3)
    p.add_argument("--hybrid-pair-reproj-thresh", type=float, default=1.5)
    p.add_argument("--refine-min-views", type=int, default=3)
    p.add_argument("--refine-max-iters", type=int, default=6)
    p.add_argument("--refine-huber-px", type=float, default=2.0)
    p.add_argument("--refine-rmse-thresh", type=float, default=2.0)
    p.add_argument("--refine-max-depth-shift-ratio", type=float, default=0.5)
    p.add_argument("--current-reproj-thresh", type=float, default=2.0)
    p.add_argument("--max-reproj-thresh", type=float, default=3.0)
    p.add_argument("--gt-epi-thresh", type=float, default=1.0)
    p.add_argument("--min-parallax-deg", type=float, default=0.10)
    p.add_argument("--min-baseline", type=float, default=0.5)
    p.add_argument("--max-pair-history", type=int, default=8)
    p.add_argument("--triang-confirmed-only", action="store_true", default=False)
    p.add_argument("--triang-min-hits", type=int, default=2)
    p.add_argument("--no-triang-reacquired", dest="triang_include_reacquired", action="store_false", default=True)
    p.add_argument("--min-depth", type=float, default=1.0)
    p.add_argument("--max-depth", type=float, default=120.0)
    p.add_argument("--reproj-thresh", type=float, default=2.0)

    p.add_argument("--pose-eval-gap", type=int, default=2)
    p.add_argument("--pose-min-pairs", type=int, default=30)
    p.add_argument("--pose-ransac-thresh", type=float, default=0.5)
    p.add_argument("--pose-ransac-prob", type=float, default=0.999)
    p.add_argument("--pose-ransac-max-iters", type=int, default=3000)

    p.add_argument("--draw-max-tracks", type=int, default=2200)
    p.set_defaults(**load_argparse_defaults(cfg_ns.config))
    args = p.parse_args(remaining)
    if args.lidar_digits is None:
        args.lidar_digits = args.image_digits
    return args


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    K = load_kitti_K(args.calib)
    poses = load_kitti_poses(args.poses)
    cfg = make_manager_config(args)
    manager = FeatureManager(cfg, K, poses)
    lidar_projection = prepare_lidar_projection(args)

    with open(args.output_root / "config.json", "w") as f:
        json.dump({k: json_safe_value(v) for k, v in vars(args).items()}, f, indent=2)

    rows: List[dict] = []
    point_rows: List[dict] = []
    diagnostic_point_rows: List[dict] = []

    end_exclusive = args.start + max(args.num_frames, 0)
    if args.end_frame is not None:
        end_exclusive = min(end_exclusive, args.end_frame + 1)
    end_exclusive = min(end_exclusive, len(poses))

    print("============== SIFT+LK FEATURE DEPTH EVAL ==============")
    print("Image dir:", args.img_dir)
    print("Calib:", args.calib)
    print("Poses:", args.poses)
    print("Frames:", args.start, "to", end_exclusive - 1)
    print("LiDAR:", args.velodyne_dir if lidar_projection is not None else "disabled/unavailable")
    print("Detector/descriptor:", args.detector, "/", args.descriptor, "| LK on:", args.lk_on, "| matcher:", args.matcher)
    print("LK require descriptor refresh:", args.lk_require_desc_refresh)
    print("Pose RANSAC threshold px:", args.pose_ransac_thresh)
    print("Plot backend:", args.plot_backend)
    print("Output:", args.output_root)
    print("=====================================================")
    if args.plot_backend == "opencv":
        setattr(save_metric_plot, "_matplotlib_broken", True)
    elif args.plot_backend == "matplotlib":
        setattr(save_metric_plot, "_matplotlib_broken", False)

    first_image = load_gray(args.img_dir, args.start, args.image_digits)
    if first_image is None:
        raise FileNotFoundError(f"Could not load start frame {args.start}")

    t0 = time.perf_counter()
    stats = manager.reset_at(args.start, first_image, update_geom_stats=False)
    triang_t0 = time.perf_counter()
    triang_infos = manager.evaluate_triang_candidates(args.start, compute_dlt=True, fast=False)
    stats.time_geom_depth_pose_ms = 1000.0 * (time.perf_counter() - triang_t0)
    pose_t0 = time.perf_counter()
    pose = manager.evaluate_pose_gap(args.start)
    pose_ms = 1000.0 * (time.perf_counter() - pose_t0)
    stats.time_no_draw_ms = 1000.0 * (time.perf_counter() - t0)
    stats.time_manage_depth_ms = stats.time_no_draw_ms

    frames_to_process = list(range(args.start, end_exclusive))
    images = {args.start: first_image}

    for idx, frame in enumerate(frames_to_process):
        if idx > 0:
            t_load = time.perf_counter()
            image = load_gray(args.img_dir, frame, args.image_digits)
            load_ms = 1000.0 * (time.perf_counter() - t_load)
            if image is None:
                print(f"[stop] could not load frame {frame:0{args.image_digits}d}")
                break
            t_step = time.perf_counter()
            stats = manager.step(frame, image, update_geom_stats=False)
            step_ms = 1000.0 * (time.perf_counter() - t_step)
            triang_t0 = time.perf_counter()
            triang_infos = manager.evaluate_triang_candidates(frame, compute_dlt=True, fast=False)
            triang_ms = 1000.0 * (time.perf_counter() - triang_t0)
            pose_t0 = time.perf_counter()
            pose = manager.evaluate_pose_gap(frame)
            pose_ms = 1000.0 * (time.perf_counter() - pose_t0)
            stats.time_load_ms = load_ms
            stats.time_manage_depth_ms = step_ms + triang_ms + pose_ms
            stats.time_geom_depth_pose_ms = triang_ms + pose_ms
            stats.time_no_draw_ms = load_ms + stats.time_manage_depth_ms
        else:
            image = images[args.start]
            load_ms = 0.0
            triang_ms = stats.time_geom_depth_pose_ms
            step_ms = stats.time_manage_depth_ms - triang_ms - pose_ms

        sparse_uv, sparse_z, frame_point_rows = active_depth_points(
            manager,
            frame,
            image.shape,
            triang_infos,
            all_active_depths=args.all_active_depths,
        )
        triang_summary = summarize_triang(triang_infos)
        lidar_metrics, matched_payload = compute_lidar_metrics(
            args,
            lidar_projection,
            frame,
            image.shape,
            sparse_uv,
            sparse_z,
        )
        frame_diagnostic_points = collect_matched_point_diagnostics(frame, frame_point_rows, matched_payload)
        diagnostic_point_rows.extend(frame_diagnostic_points)

        row = {
            "frame": int(frame),
            "num_sparse_depth": int(len(sparse_z)),
            "num_active_tracks": int(stats.active),
            "num_confirmed_active": int(stats.confirmed_active),
            "bucket_coverage": finite_float(stats.bucket_coverage),
            "median_depth_m": safe_median(sparse_z),
            "time_load_ms": finite_float(stats.time_load_ms),
            "time_lk_track_ms": finite_float(getattr(stats, "time_lk_track_ms", np.nan)),
            "time_step_ms": finite_float(step_ms),
            "time_triang_ms": finite_float(triang_ms),
            "time_pose_ms": finite_float(pose_ms),
            "time_manage_depth_ms": finite_float(stats.time_manage_depth_ms),
            "time_total_no_draw_ms": finite_float(stats.time_no_draw_ms),
            "primary_kp": int(stats.primary_kp),
            "soft_kp": int(stats.soft_kp),
            "matched": int(stats.matched),
            "soft_rescued": int(stats.soft_rescued),
            "spawned_primary": int(stats.spawned_primary),
            "spawned_soft": int(stats.spawned_soft),
            "lk_attempted": int(getattr(stats, "lk_attempted", 0)),
            "lk_accepted": int(getattr(stats, "lk_accepted", 0)),
            "lk_accept_rate": float(getattr(stats, "lk_accepted", 0) / max(getattr(stats, "lk_attempted", 0), 1)),
            "lk_reject_forward": int(getattr(stats, "lk_reject_forward", 0)),
            "lk_reject_backward": int(getattr(stats, "lk_reject_backward", 0)),
            "lk_reject_fb": int(getattr(stats, "lk_reject_fb", 0)),
            "lk_reject_bounds": int(getattr(stats, "lk_reject_bounds", 0)),
            "lk_reject_step": int(getattr(stats, "lk_reject_step", 0)),
            "lk_reject_error": int(getattr(stats, "lk_reject_error", 0)),
            "lk_reject_epi": int(getattr(stats, "lk_reject_epi", 0)),
            "lk_reject_flow": int(getattr(stats, "lk_reject_flow", 0)),
            "lk_reject_desc": int(getattr(stats, "lk_reject_desc", 0)),
            "lk_desc_refreshed": int(getattr(stats, "lk_desc_refreshed", 0)),
            "lk_median_fb_err": finite_float(getattr(stats, "lk_median_fb_err", np.nan)),
            "lk_median_err": finite_float(getattr(stats, "lk_median_err", np.nan)),
            "lk_median_refresh_desc": finite_float(getattr(stats, "lk_median_refresh_desc", np.nan)),
        }
        # Final-depth coverage is the spatial spread of the accepted sparse depth
        # map itself. It is intentionally separate from manager bucket_coverage,
        # which measures active tracks before/independent of triangulation.
        row.update(coverage_stats("depth", sparse_uv, image.shape, args.grid_cols, args.grid_rows))
        row.update(depth_bin_frame_metrics(frame_diagnostic_points))
        row.update(triang_summary)
        row.update(lidar_metrics)
        row.update(pose_row(pose))
        rows.append(row)

        if (
            args.save_visualizations
            and matched_payload is not None
            and args.save_vis_every > 0
            and (idx % args.save_vis_every == 0)
        ):
            save_relative_error_visualization(
                image_gray=image,
                frame=frame,
                matched_payload=matched_payload,
                output_dir=args.output_root,
                rel_vmax=args.error_vmax_rel,
                abs_vmax=args.error_vmax_abs,
                max_points=args.max_vis_points,
                show_lidar_points=args.show_lidar_vis_points,
            )

        if args.save_per_point:
            matched = matched_payload["matched"] if matched_payload is not None else np.zeros(len(frame_point_rows), dtype=bool)
            matched_z = matched_payload["matched_z"] if matched_payload is not None else np.full(len(frame_point_rows), np.nan)
            matched_dist = matched_payload["matched_dist"] if matched_payload is not None else np.full(len(frame_point_rows), np.nan)
            for p_i, pr in enumerate(frame_point_rows):
                pr = dict(pr)
                pr["frame"] = int(frame)
                pr["lidar_matched"] = bool(matched[p_i]) if p_i < len(matched) else False
                pr["lidar_depth_m"] = finite_float(matched_z[p_i]) if p_i < len(matched_z) else float("nan")
                pr["lidar_pixel_dist_px"] = finite_float(matched_dist[p_i]) if p_i < len(matched_dist) else float("nan")
                pr["depth_abs_err_m"] = abs(pr["depth_m"] - pr["lidar_depth_m"]) if pr["lidar_matched"] else float("nan")
                pr["depth_rel_err"] = (
                    pr["depth_abs_err_m"] / max(pr["lidar_depth_m"], 1e-12)
                    if pr["lidar_matched"] and np.isfinite(pr["lidar_depth_m"]) and pr["lidar_depth_m"] > 0
                    else float("nan")
                )
                point_rows.append(pr)

        print(
            f"[eval] frame {frame:0{args.image_digits}d} "
            f"depth={len(sparse_z)} matched={row['num_lidar_matched']} "
            f"lk={row['lk_accepted']}/{row['lk_attempted']} "
            f"cov={row['depth_coverage']:.2f} matchCov={row['lidar_matched_depth_coverage']:.2f} "
            f"medAbs={row['raw_median_abs_err_m']:.3f}m medRel={row['raw_median_rel_err']:.3f} "
            f"poseInl={row['pose_inliers']}/{row['pose_pairs']} "
            f"rot={row['pose_rot_err_deg']:.2f}deg t={row['pose_t_dir_err_deg']:.2f}deg "
            f"time={row['time_total_no_draw_ms']:.1f}ms"
        )

    write_csv_rows(args.output_root / "per_frame_metrics.csv", rows)
    if args.save_per_point:
        write_csv_rows(args.output_root / "per_point_depths.csv", point_rows)
    if not args.minimal_output:
        write_csv_rows(args.output_root / "matched_point_diagnostics.csv", diagnostic_point_rows)
        summarize_point_bins(diagnostic_point_rows, args.output_root)
    summary = make_summary(rows)
    with open(args.output_root / "summary_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    if not args.minimal_output:
        save_all_plots(rows, args.output_root)
        save_point_scatter_plots(diagnostic_point_rows, args.output_root)
        save_frame_tradeoff_plots(rows, args.output_root)
    manager.close()

    print("Done.")
    print("Per-frame metrics:", args.output_root / "per_frame_metrics.csv")
    print("Summary metrics:", args.output_root / "summary_metrics.csv")
    if not args.minimal_output:
        print("Plots:", args.output_root / "plots")


if __name__ == "__main__":
    main()
