"""Configuration for the hybrid SIFT+LK sparse-depth manager."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_IMG_DIR = Path("data/2011_09_30_drive_0016_sync/image_00/data")
DEFAULT_CALIB_PATH = Path("data/sequences/04/calib.txt")
DEFAULT_POSES_PATH = Path("data/data_odometry_poses/dataset/poses/04.txt")
DEFAULT_IMAGE_DIGITS = 10
DEFAULT_OUTPUT_ROOT = Path("outputs/sift_lk_feature_manager_seq04_v1")


@dataclass
class Config:
    img_dir: Path
    calib_path: Path
    poses_path: Path
    velodyne_dir: Optional[Path]
    calib_velo_to_cam: Optional[Path]
    calib_cam_to_cam: Optional[Path]
    camera: str
    image_digits: int
    lidar_digits: Optional[int]
    start_frame: int
    end_frame: Optional[int]
    output_root: Path

    # Orthogonal feature axes (replacing the bundled legacy `detector_mode`):
    #   detector   = keypoint locations: "shi" | "sift" | "xfeat"
    #   descriptor = fingerprint type:   "sift" | "xfeat"  (distinct from
    #                descriptor_mode below, which is the latest/anchor reference)
    detector: str
    descriptor: str
    shi_max_corners: int
    shi_quality_level: float
    shi_min_distance_px: float
    shi_block_size: int

    # SIFT primary detector.
    sift_nfeatures: int
    sift_n_octave_layers: int
    sift_contrast_threshold: float
    sift_edge_threshold: float
    sift_sigma: float

    # Optional softened detector. In v3 it can rescue unmatched old tracks first,
    # then fill under-populated buckets with unused detections.
    soft_spawn: bool
    soft_rescue: bool
    soft_nfeatures: int
    soft_contrast_threshold: float
    soft_edge_threshold: float
    soft_sigma: float
    soft_confirm_hits: int
    soft_rescue_search_radius_px: float
    soft_rescue_ratio: float
    soft_rescue_max_desc_dist: float
    soft_detect_mask_used_primary_px: float
    soft_dup_used_primary_px: float
    soft_spawn_primary_sep_px: float
    soft_source_penalty: float

    # Association.
    descriptor_mode: str
    prediction_mode: str
    search_radius_px: float
    ratio: float
    max_desc_dist: float
    reacq_search_radius_px: float
    reacq_ratio: float
    reacq_max_desc_dist: float
    allow_single_candidate: bool
    spatial_weight: float
    miss_cost: float
    max_step_px: float

    # LK continuation. LK is tried before descriptor association. Accepted LK
    # points append normal observations, so the existing triangulation logic can
    # use them without special cases.
    lk_on: bool
    lk_require_desc_refresh: bool
    lk_win_size: int
    lk_max_level: int
    lk_max_iter: int
    lk_eps: float
    lk_fb_thresh_px: float
    lk_max_step_px: float
    lk_max_error: float
    lk_min_eig_thresh: float
    lk_desc_refresh_radius_px: float
    lk_desc_refresh_max_dist: float
    detection_period: int
    soft_sift_period: int
    spawn_period: int
    force_detection_active_below: int
    force_detection_coverage_below: float
    force_soft_underfilled_buckets: int
    lk_epipolar_thresh_px: float
    lk_flow_consistency: bool
    lk_flow_radius_px: float
    lk_flow_min_neighbors: int
    lk_flow_mad_k: float
    lk_flow_abs_thresh_px: float

    # Track management.
    min_hits_to_confirm: int
    max_misses: int
    max_active_tracks: int
    duplicate_suppression: bool
    duplicate_dist_px: float

    # Quality score used for duplicate suppression and output/display selection.
    # Formula:
    # score = confirmed_bonus * confirmed
    #       + min(hit_cap, hit_weight * hit_count)
    #       - miss_penalty * miss_count
    #       - reacq_penalty * reacq_count
    #       - probation_penalty * reacq_probation
    #       - soft_source_penalty * last_source_is_soft
    #       - lk_staleness_penalty * stale_descriptor_frames_after_threshold
    #       - desc_penalty * last_desc_dist
    #       - spatial_penalty * last_spatial_err
    #       + lk_bonus * last_source_is_lk
    #       + active_bonus * active_in_current_frame
    quality_confirmed_bonus: float
    quality_hit_weight: float
    quality_hit_cap: float
    quality_miss_penalty: float
    quality_reacq_penalty: float
    quality_probation_penalty: float
    quality_lk_bonus: float
    quality_lk_stale_after: int
    quality_lk_stale_penalty: float
    quality_lk_stale_cap: float
    quality_desc_penalty: float
    quality_spatial_penalty: float
    quality_active_bonus: float

    # Bucketing/spawning.
    grid_cols: int
    grid_rows: int
    target_per_bucket: int
    max_per_bucket: int
    output_per_bucket: int
    output_max_tracks: int
    min_spawn_distance_px: float
    spawn_count_candidates: bool

    # Geometry/triangulation diagnostic.
    triangulation_method: str
    multiview_min_views: int
    hybrid_pair_min_parallax_deg: float
    hybrid_pair_min_baseline_m: float
    hybrid_pair_max_pair_history: int
    hybrid_pair_reproj_thresh_px: float
    refine_min_views: int
    refine_max_iters: int
    refine_huber_px: float
    refine_rmse_thresh_px: float
    refine_max_depth_shift_ratio: float
    current_reproj_thresh_px: float
    max_reproj_thresh_px: float
    gt_epi_thresh_px: float
    min_parallax_deg: float
    min_baseline_m: float
    max_pair_history: int
    triang_confirmed_only: bool
    triang_min_hits: int
    triang_include_reacquired: bool
    min_depth_m: float
    max_depth_m: float
    reproj_thresh_px: float


    # LiDAR projection/evaluation.
    lidar_radius_px: float
    lidar_match_mode: str
    min_lidar_depth_m: float
    max_lidar_depth_m: float
    max_lidar_vis_points: int
    lidar_click_radius_px: float

    # Estimated-pose diagnostic.
    pose_eval_gap: int
    pose_min_pairs: int
    pose_ransac_thresh_px: float
    pose_ransac_prob: float
    pose_ransac_max_iters: int

    # Logging.
    log_csv: bool
    log_path: Optional[Path]

    # Display / output selection.
    draw_max_tracks: int
    point_radius: int
    side_panel_width: int
    side_panel_height: int
    resize: float
    mouse_coordinate_mode: str
    path_len: int
    inspect_radius_px: float

    # XFeat detector (detector_mode == "xfeat_sift_lk"). XFeat supplies keypoint
    # locations only; SIFT descriptors are still computed at those points, so the
    # rest of the pipeline is unchanged. Defaulted so older Config callers and
    # non-XFeat runs need not set them.
    xfeat_top_k: int = 4096
    xfeat_detection_threshold: float = 0.05
    xfeat_desc_size_px: float = 7.0
    # xfeat_native mode only: XFeat descriptors are unit-norm (L2 in [0,2]); this
    # scale maps them into the SIFT L2 range so existing max_desc_dist gates apply.
    xfeat_native_desc_scale: float = 250.0

    # Association matcher: "radius_lowe" = per-track spatial-gated nearest neighbor
    # with Lowe ratio (the default pipeline). "xfeat_mnn" = XFeat's own matching
    # rule: global mutual nearest neighbor on cosine similarity with a min-cossim
    # threshold (no spatial search gate; only a max-displacement sanity cap).
    matcher: str = "radius_lowe"
    xfeat_mnn_min_cossim: float = 0.82
    # Max accepted displacement for an MNN match; 0 falls back to max_step_px.
    xfeat_mnn_max_step_px: float = 0.0


# Legacy bundled detector_mode -> (detector, descriptor). Kept for back-compat so
# old configs/CLI (detector_mode="sift_lk" etc.) still resolve to the new axes.
LEGACY_DETECTOR_MODE = {
    "sift_lk": ("sift", "sift"),
    "shi_sift_lk": ("shi", "sift"),
    "xfeat_sift_lk": ("xfeat", "sift"),
    "xfeat_native": ("xfeat", "xfeat"),
}


def resolve_detector_descriptor(detector, descriptor, detector_mode):
    """Return (detector, descriptor) from the new axes or a legacy detector_mode.

    A legacy `detector_mode` wins whenever it is explicitly present: it is only
    ever non-None for genuinely legacy input (old config key / --detector-mode
    flag), while the new config always leaves it unset. This keeps old configs
    and scripts working even when merged on top of a new base config that already
    supplies `detector`/`descriptor` defaults. New usage (detector_mode=None)
    falls through to the `detector`/`descriptor` axes.
    """
    if detector_mode:
        if detector_mode not in LEGACY_DETECTOR_MODE:
            raise ValueError(f"Unknown legacy detector_mode: {detector_mode!r}")
        return LEGACY_DETECTOR_MODE[detector_mode]
    if detector:
        return detector, (descriptor or "sift")
    return "sift", "sift"


def validate_feature_axes(detector, descriptor, matcher):
    """Warn (do not crash) on unusual/unsupported feature-axis combinations."""
    if descriptor == "xfeat" and detector != "xfeat":
        print(f"[warn] descriptor='xfeat' currently requires detector='xfeat' (got detector={detector!r}); "
              "XFeat descriptors are only sampled at XFeat keypoints.")
    if matcher == "xfeat_mnn" and descriptor != "xfeat":
        print(f"[warn] matcher='xfeat_mnn' (cosine mutual-NN) is designed for XFeat unit descriptors, "
              f"but descriptor={descriptor!r}; results may be poor.")


