"""Shared CLI -> Config plumbing for the sparse-depth entry points.

Both the interactive viewer and the headless evaluator need to turn a large set
of command-line flags (layered on top of TOML ``--config`` files) into a single
:class:`~sparse_depth.manager_config.Config`. This module centralises that so the
entry-point scripts stay thin:

    p = argparse.ArgumentParser(parents=[config_parser])
    add_manager_args(p)
    p.set_defaults(**load_argparse_defaults(cfg_ns.config))
    args = p.parse_args(remaining)
    cfg = make_manager_config(args)

``add_manager_args`` registers every flag consumed by ``make_manager_config``.
Scripts add their own run-specific flags (frame count, output style, ...) on top.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sparse_depth.manager_config import (
    Config,
    DEFAULT_CALIB_PATH,
    DEFAULT_IMAGE_DIGITS,
    DEFAULT_IMG_DIR,
    DEFAULT_POSES_PATH,
    resolve_detector_descriptor,
    validate_feature_axes,
)

# Default output location for the simple evaluator; scripts may override.
DEFAULT_OUTPUT_ROOT = Path("outputs/sparse_depth_eval")


def build_config_parser() -> argparse.ArgumentParser:
    """A parent parser that only knows ``--config`` (repeatable, layered).

    Parse this first with ``parse_known_args`` to discover the TOML config files,
    load their values as argparse defaults, then build the full parser with this
    as a ``parents=[...]`` entry so ``--config`` shows up in ``--help`` too.
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="TOML config file. Repeatable; later configs and explicit CLI flags "
             "override earlier values.",
    )
    return config_parser


def add_manager_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register every flag that :func:`make_manager_config` reads.

    Grouped roughly by subsystem to keep ``--help`` navigable. Defaults match the
    tuned baseline; a TOML ``--config`` normally supplies the dataset paths and
    any overrides.
    """
    # --- dataset IO / run scope -------------------------------------------------
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    p.add_argument("--poses", type=Path, default=DEFAULT_POSES_PATH)
    p.add_argument("--image-digits", type=int, default=DEFAULT_IMAGE_DIGITS)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    # --- detector / descriptor axes --------------------------------------------
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
    p.add_argument("--xfeat-native-desc-scale", type=float, default=250.0, help="Scale for unit XFeat descriptors so SIFT-tuned distance gates apply (xfeat_native).")

    # --- LiDAR ground truth -----------------------------------------------------
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

    # --- SIFT primary detector --------------------------------------------------
    p.add_argument("--sift-nfeatures", type=int, default=8000)
    p.add_argument("--sift-n-octave-layers", type=int, default=3)
    p.add_argument("--contrast", type=float, default=0.01)
    p.add_argument("--edge", type=float, default=10.0)
    p.add_argument("--sigma", type=float, default=1.6)

    # --- soft SIFT rescue/fill layer -------------------------------------------
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

    # --- descriptor association -------------------------------------------------
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

    # --- LK continuation --------------------------------------------------------
    p.add_argument("--lk-on", "--lk-tracking", dest="lk_on", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable (--no-lk-on) LK-first continuation before descriptor association.")
    p.add_argument("--lk-require-desc-refresh", action="store_true", help="Strict mode: LK points must also find a nearby acceptable SIFT descriptor refresh.")
    p.add_argument("--lk-win-size", type=int, default=21, help="LK window side length in pixels.")
    p.add_argument("--lk-max-level", type=int, default=3, help="Number of pyramid levels for LK.")
    p.add_argument("--lk-max-iter", type=int, default=30, help="Maximum LK iterations per pyramid level.")
    p.add_argument("--lk-eps", type=float, default=0.01, help="LK convergence epsilon.")
    p.add_argument("--lk-fb-thresh", type=float, default=1.5, help="Forward-backward consistency threshold in pixels.")
    p.add_argument("--lk-max-step-px", type=float, default=80.0, help="Reject LK tracks moving more than this many pixels in one frame.")
    p.add_argument("--lk-max-error", type=float, default=35.0, help="Reject LK tracks with OpenCV LK error above this value.")
    p.add_argument("--lk-min-eig-thresh", type=float, default=1e-4, help="LK minimum eigenvalue threshold.")
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

    # --- track lifecycle --------------------------------------------------------
    p.add_argument("--min-hits-confirm", type=int, default=3)
    p.add_argument("--max-misses", type=int, default=2)
    p.add_argument("--max-active-tracks", type=int, default=3000)
    p.add_argument("--no-duplicate-suppression", dest="duplicate_suppression", action="store_false", default=True)
    p.add_argument("--duplicate-dist", type=float, default=4.0)

    # --- quality scoring --------------------------------------------------------
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

    # --- spatial bucketing / output selection ----------------------------------
    p.add_argument("--grid-cols", type=int, default=20)
    p.add_argument("--grid-rows", type=int, default=8)
    p.add_argument("--target-per-bucket", type=int, default=12)
    p.add_argument("--max-per-bucket", type=int, default=16)
    p.add_argument("--output-per-bucket", "--depth-output-per-bucket", dest="output_per_bucket", type=int, default=12)
    p.add_argument("--output-max-tracks", "--depth-output-max-tracks", dest="output_max_tracks", type=int, default=2200)
    p.add_argument("--min-spawn-distance", type=float, default=7.0)
    p.add_argument("--spawn-count-confirmed-only", dest="spawn_count_candidates", action="store_false", default=True)

    # --- triangulation / pose ---------------------------------------------------
    p.add_argument("--triangulation-method", choices=["best_pair_dlt", "flow_depth_pair", "ttc_expansion", "ttc_expansion_norot", "refined_pair_dlt", "corrected_pair_dlt", "windowed_multiview_dlt", "refined_multiview_dlt", "hybrid_pair_multiview"], default="windowed_multiview_dlt")
    p.add_argument("--pose-source", choices=["gt", "estimated"], default="estimated",
                   help="Pose used for triangulation/TTC and the LK epipolar gate. "
                        "'estimated' runs a frame-to-frame essential-matrix backend "
                        "(GT translation magnitude per step); 'gt' uses ground-truth poses.")
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

    # --- pose backend (essential-matrix estimation) ----------------------------
    p.add_argument("--pose-eval-gap", type=int, default=2)
    p.add_argument("--pose-min-pairs", type=int, default=30)
    p.add_argument("--pose-ransac-thresh", type=float, default=0.5)
    p.add_argument("--pose-ransac-prob", type=float, default=0.999)
    p.add_argument("--pose-ransac-max-iters", type=int, default=3000)

    p.add_argument("--draw-max-tracks", type=int, default=2200)
    return p


def make_manager_config(args: argparse.Namespace) -> Config:
    """Turn parsed CLI args into a :class:`Config` for the feature manager."""
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
