"""Dataclasses for the hybrid SIFT+LK sparse-depth manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Observation:
    frame: int
    pt: np.ndarray  # shape (2,), float32/float64


@dataclass
class Track:
    id: int
    birth_frame: int
    anchor_pt: np.ndarray
    anchor_desc: np.ndarray
    latest_desc: np.ndarray
    required_hits_to_confirm: int
    observations: List[Observation] = field(default_factory=list)
    hit_count: int = 1
    miss_count: int = 0
    confirmed: bool = False
    dead: bool = False
    last_desc_dist: float = np.nan
    last_anchor_desc_dist: float = np.nan
    last_ratio: float = np.nan
    last_spatial_err: float = np.nan
    last_cost: float = np.nan
    last_source: str = "primary"
    last_lk_fb_err: float = np.nan
    last_lk_err: float = np.nan
    last_desc_refresh_dist: float = np.nan
    last_desc_refresh_spatial: float = np.nan
    frames_since_desc_refresh: int = 0
    newborn_last_step: bool = True
    reacquired_last_step: bool = False
    reacq_probation: bool = False
    reacq_count: int = 0
    matched_last_step: bool = False
    missed_last_step: bool = False
    killed_last_step: bool = False
    duplicate_killed_last_step: bool = False
    quality_score: float = 0.0

    def last_obs(self) -> Observation:
        return self.observations[-1]

    def last_pt(self) -> np.ndarray:
        return self.observations[-1].pt

    def last_seen_frame(self) -> int:
        return self.observations[-1].frame

    def active_in(self, frame: int) -> bool:
        return (not self.dead) and self.last_seen_frame() == frame

    def age_frames(self, current_frame: int) -> int:
        return max(0, current_frame - self.birth_frame + 1)

    def observed_span(self) -> int:
        if not self.observations:
            return 0
        return self.observations[-1].frame - self.observations[0].frame + 1

    def has_observation_at(self, frame: int) -> bool:
        return any(o.frame == frame for o in self.observations)

    def observation_at(self, frame: int) -> Optional[Observation]:
        for o in reversed(self.observations):
            if o.frame == frame:
                return o
        return None


@dataclass
class AssignmentCandidate:
    track_index: int
    det_index: int
    desc_dist: float
    second_dist: float
    ratio_value: float
    spatial_err: float
    cost: float
    mode: str  # normal or reacq


@dataclass
class FrameStats:
    frame: int
    primary_kp: int = 0
    soft_kp: int = 0
    before_tracks: int = 0
    before_active: int = 0
    assoc_candidates: int = 0
    matched: int = 0
    matched_primary: int = 0
    soft_rescued: int = 0
    reacquired: int = 0
    missed: int = 0
    killed: int = 0
    dup_killed: int = 0
    spawned_primary: int = 0
    spawned_soft: int = 0
    active: int = 0
    confirmed_active: int = 0
    candidate_active: int = 0
    temp_lost: int = 0
    dead_total: int = 0
    bucket_coverage: float = 0.0
    underfilled_buckets: int = 0
    overfilled_buckets: int = 0
    mean_bucket_active: float = 0.0
    min_bucket_active: int = 0
    median_desc: float = np.nan
    median_spatial: float = np.nan
    selected_display: int = 0
    triang_good: int = 0
    triang_candidate_good: int = 0
    triang_confirmed_good: int = 0
    triang_reacq_good: int = 0
    triang_bad_epi: int = 0
    triang_low_parallax: int = 0
    triang_bad_reproj: int = 0
    triang_bad_depth: int = 0
    triang_no_pair: int = 0
    depth_valid: int = 0
    median_depth: float = np.nan
    median_reproj: float = np.nan
    pose_pairs: int = 0
    pose_inliers: int = 0
    pose_inlier_ratio: float = np.nan
    pose_rot_err_deg: float = np.nan
    pose_t_err_deg: float = np.nan
    time_load_ms: float = np.nan
    time_manage_depth_ms: float = np.nan
    time_no_draw_ms: float = np.nan
    time_primary_detect_ms: float = np.nan
    time_lk_track_ms: float = np.nan
    lk_attempted: int = 0
    lk_accepted: int = 0
    lk_reject_forward: int = 0
    lk_reject_backward: int = 0
    lk_reject_fb: int = 0
    lk_reject_bounds: int = 0
    lk_reject_step: int = 0
    lk_reject_error: int = 0
    lk_reject_epi: int = 0
    lk_reject_flow: int = 0
    lk_reject_desc: int = 0
    lk_desc_refreshed: int = 0
    lk_median_fb_err: float = np.nan
    lk_median_err: float = np.nan
    lk_median_refresh_desc: float = np.nan
    time_primary_assoc_ms: float = np.nan
    time_soft_detect_ms: float = np.nan
    time_soft_rescue_ms: float = np.nan
    time_miss_ms: float = np.nan
    time_spawn_ms: float = np.nan
    time_duplicate_ms: float = np.nan
    time_count_stats_ms: float = np.nan
    time_geom_depth_pose_ms: float = np.nan
    time_log_ms: float = np.nan


