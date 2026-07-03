"""Core hybrid SIFT+LK feature manager.

The interactive viewer and the headless evaluator both use this class. Keeping
LK tracking, SIFT rescue/spawn, quality scoring, and triangulation diagnostics
here prevents the two front ends from drifting apart.
"""

from __future__ import annotations

import csv
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparse_depth.feature_utils import (
    SpatialHash,
    bucket_index,
    descriptor_distance,
    detect_shi_sift,
    detect_sift,
    detect_xfeat_native,
    detect_xfeat_sift,
    keypoint_points,
    make_sift,
    predict_track,
)
from sparse_depth.geometry import fundamental_from_R_t, relative_pose, sampson_error_px
from sparse_depth.pose_eval import PoseEval, evaluate_pose_gap_from_tracks
from sparse_depth.manager_config import Config
from sparse_depth.track_types import AssignmentCandidate, FrameStats, Observation, Track
from sparse_depth.triangulation import TriangInfo, evaluate_best_pair_triangulation


class FeatureManager:
    def __init__(self, cfg: Config, K: np.ndarray, poses: Sequence[np.ndarray]):
        self.cfg = cfg
        self.K = K
        self.poses = poses
        self.primary_sift = make_sift(
            cfg.sift_nfeatures, cfg.sift_n_octave_layers,
            cfg.sift_contrast_threshold, cfg.sift_edge_threshold, cfg.sift_sigma,
        )
        self.soft_sift = make_sift(
            cfg.soft_nfeatures, cfg.sift_n_octave_layers,
            cfg.soft_contrast_threshold, cfg.soft_edge_threshold, cfg.soft_sigma,
        )
        # Soft-SIFT rescue/fill only applies to the plain SIFT detector+descriptor.
        self.soft_enabled = bool(cfg.soft_spawn and cfg.detector == "sift" and cfg.descriptor == "sift")
        self.xfeat = None
        if cfg.detector == "xfeat":
            from sparse_depth.xfeat_detector import XFeatDetector
            self.xfeat = XFeatDetector(cfg.xfeat_top_k, cfg.xfeat_detection_threshold)
        self.tracks: List[Track] = []
        self.next_id = 0
        self.current_frame = cfg.start_frame
        self.prev_image: Optional[np.ndarray] = None
        self.prev_frame: Optional[int] = None
        self.last_stats = FrameStats(frame=cfg.start_frame)
        self.log_file = None
        self.log_writer = None
        self.log_header_written = False
        if cfg.log_csv:
            cfg.output_root.mkdir(parents=True, exist_ok=True)
            log_path = cfg.log_path if cfg.log_path is not None else (cfg.output_root / "sift_feature_manager_log.csv")
            self.log_file = open(log_path, "w", newline="")
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=list(FrameStats.__dataclass_fields__.keys()))
            self.log_writer.writeheader()
            self.log_header_written = True

    def _detect_primary(self, image: np.ndarray):
        det, desc = self.cfg.detector, self.cfg.descriptor
        if det == "shi":
            return detect_shi_sift(
                image, self.primary_sift, self.cfg.shi_max_corners,
                self.cfg.shi_quality_level, self.cfg.shi_min_distance_px,
                self.cfg.shi_block_size,
            )
        if det == "xfeat":
            if desc == "xfeat":
                return detect_xfeat_native(
                    image, self.xfeat, self.cfg.xfeat_top_k,
                    self.cfg.xfeat_desc_size_px, self.cfg.xfeat_native_desc_scale,
                )
            return detect_xfeat_sift(
                image, self.primary_sift, self.xfeat,
                self.cfg.xfeat_top_k, self.cfg.xfeat_desc_size_px,
            )
        return detect_sift(image, self.primary_sift)

    def close(self):
        if self.log_file is not None:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None

    def _soft_detection_mask(self, image_shape: Tuple[int, int], primary_pts: np.ndarray, used_primary: set) -> Optional[np.ndarray]:
        """Mask softened SIFT away from primary detections that already won.

        Soft SIFT is still allowed near unused primary detections, because an
        unused primary point may have failed descriptor matching even though a
        nearby soft-scale/orientation keypoint can rescue the track. The mask is
        therefore keyed to used primary detections only.
        """
        radius = float(self.cfg.soft_detect_mask_used_primary_px)
        if radius <= 0.0 or not used_primary or primary_pts.size == 0:
            return None
        h, w = image_shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        for idx in used_primary:
            if int(idx) < 0 or int(idx) >= primary_pts.shape[0]:
                continue
            x, y = primary_pts[int(idx)]
            cv2.circle(mask, (int(round(x)), int(round(y))), int(round(radius)), 0, thickness=-1)
        return mask

    def _period_due(self, frame: int, period: int) -> bool:
        period = int(period)
        if period <= 1:
            return True
        return ((int(frame) - int(self.cfg.start_frame)) % period) == 0

    def _bucket_fill_snapshot(self, frame: int, image_shape: Tuple[int, int]) -> Tuple[int, float, int]:
        counts, _ = self.basic_bucket_counts(frame, image_shape)
        active = int(np.sum(counts))
        occupied = int(np.sum(counts > 0))
        coverage = float(occupied / max(1, self.cfg.grid_rows * self.cfg.grid_cols))
        underfilled = int(np.sum(counts < self.cfg.target_per_bucket))
        return active, coverage, underfilled

    def reset_at(self, frame: int, image: np.ndarray, update_geom_stats: bool = True) -> FrameStats:
        self.tracks = []
        self.next_id = 0
        self.current_frame = frame
        self.prev_image = image.copy()
        self.prev_frame = frame
        kps, desc = self._detect_primary(image)
        soft_kps, soft_desc = ([], np.empty((0, 128), dtype=np.float32))
        if self.soft_enabled:
            soft_kps, soft_desc = detect_sift(image, self.soft_sift)
        stats = FrameStats(frame=frame, primary_kp=len(kps), soft_kp=len(soft_kps))
        self._spawn_tracks(frame, image, kps, desc, set(), soft_kps, soft_desc, set(), stats)
        self._update_quality_scores(frame)
        self._refresh_counts(frame, image, stats)
        if update_geom_stats:
            self._update_geom_stats(frame, stats)
        self._log_stats(stats)
        self.last_stats = stats
        return stats

    def _track_with_lk(self, frame: int, image: np.ndarray, stats: FrameStats) -> set:
        """Continue previous-frame active tracks with pyramidal LK.

        This is the first stage of the hybrid manager. A track accepted here gets
        a normal Observation at the current frame. Descriptor matching later only
        handles tracks LK failed to continue. This keeps the triangulation code
        unchanged: it still sees a list of 2D observations per track.
        """
        cfg = self.cfg
        assigned_lk = set()
        if (not cfg.lk_on) or self.prev_image is None or self.prev_frame is None:
            return assigned_lk
        if frame != self.prev_frame + 1:
            # LK is a frame-to-frame tracker; if the caller jumps, fall back to
            # descriptor association for this step.
            return assigned_lk

        track_indices = [
            ti for ti, tr in enumerate(self.tracks)
            if (not tr.dead) and tr.active_in(self.prev_frame)
        ]
        stats.lk_attempted = len(track_indices)
        if not track_indices:
            return assigned_lk

        prev_pts = np.asarray([self.tracks[ti].last_pt() for ti in track_indices], dtype=np.float32).reshape(-1, 1, 2)
        win = (int(cfg.lk_win_size), int(cfg.lk_win_size))
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(cfg.lk_max_iter), float(cfg.lk_eps))

        try:
            next_pts, st_f, err_f = cv2.calcOpticalFlowPyrLK(
                self.prev_image,
                image,
                prev_pts,
                None,
                winSize=win,
                maxLevel=int(cfg.lk_max_level),
                criteria=criteria,
                flags=0,
                minEigThreshold=float(cfg.lk_min_eig_thresh),
            )
            back_pts, st_b, _ = cv2.calcOpticalFlowPyrLK(
                image,
                self.prev_image,
                next_pts,
                None,
                winSize=win,
                maxLevel=int(cfg.lk_max_level),
                criteria=criteria,
                flags=0,
                minEigThreshold=float(cfg.lk_min_eig_thresh),
            )
        except cv2.error:
            stats.lk_reject_forward = len(track_indices)
            return assigned_lk

        if next_pts is None or st_f is None or back_pts is None or st_b is None:
            stats.lk_reject_forward = len(track_indices)
            return assigned_lk

        h, w = image.shape[:2]
        fb_vals = []
        err_vals = []
        candidates = []
        F_lk = None
        if cfg.lk_epipolar_thresh_px > 0.0:
            try:
                R_lk, t_lk = relative_pose(self.poses, self.prev_frame, frame)
                F_lk = fundamental_from_R_t(self.K, R_lk, t_lk)
            except Exception:
                F_lk = None
        for local_i, ti in enumerate(track_indices):
            tr = self.tracks[ti]
            if int(st_f[local_i, 0]) == 0:
                stats.lk_reject_forward += 1
                continue
            if int(st_b[local_i, 0]) == 0:
                stats.lk_reject_backward += 1
                continue

            p0 = prev_pts[local_i, 0].astype(np.float32)
            p1 = next_pts[local_i, 0].astype(np.float32)
            p_back = back_pts[local_i, 0].astype(np.float32)
            fb = float(np.linalg.norm(p_back - p0))
            step = float(np.linalg.norm(p1 - p0))
            lk_err = float(err_f[local_i, 0]) if err_f is not None else np.nan

            if not (0.0 <= p1[0] < w and 0.0 <= p1[1] < h):
                stats.lk_reject_bounds += 1
                continue
            if fb > cfg.lk_fb_thresh_px:
                stats.lk_reject_fb += 1
                continue
            if step > cfg.lk_max_step_px:
                stats.lk_reject_step += 1
                continue
            if np.isfinite(lk_err) and lk_err > cfg.lk_max_error:
                stats.lk_reject_error += 1
                continue
            if F_lk is not None:
                epi = float(sampson_error_px(F_lk, p0.reshape(1, 2), p1.reshape(1, 2))[0])
                if epi > cfg.lk_epipolar_thresh_px:
                    stats.lk_reject_epi += 1
                    continue

            candidates.append((ti, p0.copy(), p1.copy(), fb, lk_err))

        if cfg.lk_flow_consistency and len(candidates) >= max(1, cfg.lk_flow_min_neighbors + 1):
            cur_pts = np.asarray([c[2] for c in candidates], dtype=np.float32)
            flows = np.asarray([c[2] - c[1] for c in candidates], dtype=np.float32)
            keep = np.ones(len(candidates), dtype=bool)
            radius = float(cfg.lk_flow_radius_px)
            for i in range(len(candidates)):
                d = np.linalg.norm(cur_pts - cur_pts[i].reshape(1, 2), axis=1)
                neigh = np.where((d > 1e-6) & (d <= radius))[0]
                if neigh.size < cfg.lk_flow_min_neighbors:
                    continue
                med = np.median(flows[neigh], axis=0)
                residuals = np.linalg.norm(flows[neigh] - med.reshape(1, 2), axis=1)
                mad = float(np.median(np.abs(residuals - np.median(residuals))))
                thresh = max(float(cfg.lk_flow_abs_thresh_px), float(cfg.lk_flow_mad_k) * 1.4826 * mad)
                err = float(np.linalg.norm(flows[i] - med))
                if err > thresh:
                    keep[i] = False
            if not np.all(keep):
                stats.lk_reject_flow += int(np.sum(~keep))
                candidates = [c for c, ok in zip(candidates, keep) if ok]

        for ti, _p0, p1, fb, lk_err in candidates:
            tr = self.tracks[ti]
            tr.observations.append(Observation(frame=frame, pt=p1.copy()))
            tr.hit_count += 1
            tr.confirmed = tr.confirmed or (tr.hit_count >= tr.required_hits_to_confirm)
            tr.miss_count = 0
            tr.last_source = "lk"
            tr.last_lk_fb_err = fb
            tr.last_lk_err = lk_err
            tr.last_desc_refresh_dist = np.nan
            tr.last_desc_refresh_spatial = np.nan
            tr.frames_since_desc_refresh += 1
            tr.matched_last_step = True
            tr.reacquired_last_step = False
            tr.reacq_probation = False
            tr.missed_last_step = False
            assigned_lk.add(ti)
            fb_vals.append(fb)
            if np.isfinite(lk_err):
                err_vals.append(lk_err)

        stats.lk_accepted = len(assigned_lk)
        stats.lk_median_fb_err = float(np.median(fb_vals)) if fb_vals else np.nan
        stats.lk_median_err = float(np.median(err_vals)) if err_vals else np.nan
        return assigned_lk

    def _refresh_lk_descriptors(self, frame: int, det_pts: np.ndarray, det_desc: np.ndarray,
                                assigned_lk: set, stats: FrameStats) -> set:
        """Opportunistically refresh descriptors for LK-accepted tracks.

        In the default hybrid mode this is not a hard gate: LK already accepted
        the geometry observation. A nearby SIFT descriptor simply keeps the
        track's descriptor fresh for future rescue. If --lk-require-desc-refresh
        is set, failing this refresh kills the LK acceptance for this frame and
        the track falls through to SIFT rescue.
        """
        cfg = self.cfg
        used_refresh = set()
        if not assigned_lk or det_pts.shape[0] == 0 or det_desc.shape[0] == 0:
            if cfg.lk_require_desc_refresh:
                stats.lk_reject_desc += len(assigned_lk)
                failed = list(assigned_lk)
                for ti in failed:
                    tr = self.tracks[ti]
                    if tr.observations and tr.observations[-1].frame == frame and tr.last_source == "lk":
                        tr.observations.pop()
                        tr.hit_count = max(0, tr.hit_count - 1)
                        tr.matched_last_step = False
                        tr.last_source = "lk_desc_reject"
                assigned_lk.clear()
                stats.lk_accepted = 0
            return used_refresh

        desc_vals = []
        failed = []
        for ti in sorted(assigned_lk):
            tr = self.tracks[ti]
            pt = tr.last_pt()
            spatial = np.linalg.norm(det_pts - pt.reshape(1, 2), axis=1)
            cand_idx = np.where(spatial <= cfg.lk_desc_refresh_radius_px)[0]
            if cand_idx.size == 0:
                failed.append(ti)
                continue

            ref_desc = tr.latest_desc if cfg.descriptor_mode == "latest" else tr.anchor_desc
            dists = descriptor_distance(det_desc[cand_idx], ref_desc)
            best_local = int(np.argmin(dists))
            best_det = int(cand_idx[best_local])
            best_dist = float(dists[best_local])
            if best_dist > cfg.lk_desc_refresh_max_dist:
                failed.append(ti)
                continue

            tr.latest_desc = det_desc[best_det].copy()
            tr.last_desc_refresh_dist = best_dist
            tr.last_desc_refresh_spatial = float(spatial[best_det])
            tr.frames_since_desc_refresh = 0
            used_refresh.add(best_det)
            desc_vals.append(best_dist)

        stats.lk_desc_refreshed = len(desc_vals)
        stats.lk_median_refresh_desc = float(np.median(desc_vals)) if desc_vals else np.nan

        if cfg.lk_require_desc_refresh and failed:
            # Remove the just-appended LK observation so the normal SIFT rescue
            # path can try to continue these tracks. This is the strict "Mode C"
            # behavior discussed in chat.
            for ti in failed:
                tr = self.tracks[ti]
                if tr.observations and tr.observations[-1].frame == frame and tr.last_source == "lk":
                    tr.observations.pop()
                    tr.hit_count = max(0, tr.hit_count - 1)
                    tr.miss_count = 0
                    tr.matched_last_step = False
                    tr.last_source = "lk_desc_reject"
            assigned_lk.difference_update(failed)
            stats.lk_accepted = len(assigned_lk)
            stats.lk_reject_desc = len(failed)

        return used_refresh

    def step(self, frame: int, image: np.ndarray, update_geom_stats: bool = True) -> FrameStats:
        cfg = self.cfg
        self.current_frame = frame
        stats = FrameStats(frame=frame)
        for tr in self.tracks:
            tr.newborn_last_step = False
            tr.reacquired_last_step = False
            tr.matched_last_step = False
            tr.missed_last_step = False
            tr.killed_last_step = False
            tr.duplicate_killed_last_step = False

        before_alive = [tr for tr in self.tracks if not tr.dead]
        stats.before_tracks = len(before_alive)
        stats.before_active = sum(1 for tr in before_alive if tr.miss_count == 0)

        # Hybrid stage 1: try LK on previous-frame active tracks. LK accepted
        # tracks are already continued before descriptor association runs.
        t_phase = time.perf_counter()
        assigned_lk = self._track_with_lk(frame, image, stats)
        stats.time_lk_track_ms = 1000.0 * (time.perf_counter() - t_phase)

        lk_active, lk_bucket_cov, lk_underfilled = self._bucket_fill_snapshot(frame, image.shape)
        run_primary = (
            self._period_due(frame, cfg.detection_period)
            or (cfg.force_detection_active_below > 0 and lk_active < cfg.force_detection_active_below)
            or (cfg.force_detection_coverage_below > 0.0 and lk_bucket_cov < cfg.force_detection_coverage_below)
        )
        run_spawn = run_primary and self._period_due(frame, cfg.spawn_period)
        run_soft = (
            run_primary
            and self.soft_enabled
            and self._period_due(frame, cfg.soft_sift_period)
            and (cfg.force_soft_underfilled_buckets <= 0 or lk_underfilled >= cfg.force_soft_underfilled_buckets)
        )

        # Primary detection is the normal, stronger SIFT pool. Existing tracks try
        # to continue into this pool only if LK did not already continue them.
        kps: List[cv2.KeyPoint] = []
        desc = np.empty((0, 128), dtype=np.float32)
        pts = np.empty((0, 2), dtype=np.float32)
        if run_primary:
            t_phase = time.perf_counter()
            kps, desc = self._detect_primary(image)
            stats.primary_kp = len(kps)
            pts = keypoint_points(kps)
            stats.time_primary_detect_ms = 1000.0 * (time.perf_counter() - t_phase)
        else:
            stats.time_primary_detect_ms = 0.0

        used_primary_refresh = self._refresh_lk_descriptors(frame, pts, desc, assigned_lk, stats)

        t_phase = time.perf_counter()
        if run_primary:
            primary_track_indices = [
                ti for ti, tr in enumerate(self.tracks)
                if (not tr.dead) and (ti not in assigned_lk) and tr.miss_count <= cfg.max_misses
            ]
            if cfg.matcher == "xfeat_mnn":
                assigned_primary, used_primary, primary_stats = self._associate_mnn(
                    frame, pts, desc, primary_track_indices, source="primary",
                    disallowed_dets=used_primary_refresh,
                )
            else:
                assigned_primary, used_primary, primary_stats = self._associate_existing_tracks(
                    frame, pts, desc, track_indices=primary_track_indices, source="primary",
                    disallowed_dets=used_primary_refresh,
                )
            used_primary = set(used_primary) | set(used_primary_refresh)
        else:
            assigned_primary = []
            used_primary = set()
            primary_stats = {"assoc_candidates": 0, "matched": 0, "reacquired": 0, "desc_vals": [], "spatial_vals": []}
        stats.time_primary_assoc_ms = 1000.0 * (time.perf_counter() - t_phase)

        # Optional soft detector. In v3 soft detections are not only for spawning:
        # after primary association, unmatched tracks may be rescued by soft detections
        # using slightly stricter descriptor gates. Only then do leftover detections spawn.
        soft_kps: List[cv2.KeyPoint] = []
        soft_desc = np.empty((0, 128), dtype=np.float32)
        soft_pts = np.empty((0, 2), dtype=np.float32)
        assigned_soft: List[int] = []
        used_soft: set = set()
        soft_stats = {
            "assoc_candidates": 0, "matched": 0, "reacquired": 0,
            "desc_vals": [], "spatial_vals": [],
        }

        if run_soft:
            t_phase = time.perf_counter()
            soft_mask = self._soft_detection_mask(image.shape, pts, used_primary)
            soft_kps, soft_desc = detect_sift(image, self.soft_sift, mask=soft_mask)
            soft_pts = keypoint_points(soft_kps)
            stats.soft_kp = len(soft_kps)
            stats.time_soft_detect_ms = 1000.0 * (time.perf_counter() - t_phase)

            if cfg.soft_rescue and len(soft_kps) > 0:
                t_phase = time.perf_counter()
                # Do not let a soft duplicate of an already-used primary detection
                # become a second observation of another track. Soft detections near
                # unused primary detections are still allowed for rescue, because the
                # nearby primary keypoint may have had a worse descriptor/scale/orientation.
                used_primary_pts = pts[list(used_primary)] if used_primary else np.empty((0, 2), dtype=np.float32)
                disallowed_soft = set()
                if used_primary_pts.size and cfg.soft_dup_used_primary_px > 0:
                    used_primary_hash = SpatialHash(cfg.soft_dup_used_primary_px)
                    used_primary_hash.add_many(used_primary_pts)
                    for si, sp in enumerate(soft_pts):
                        if used_primary_hash.too_close(sp):
                            disallowed_soft.add(si)

                already_assigned = set(assigned_lk) | set(assigned_primary)
                unmatched_indices = [
                    ti for ti, tr in enumerate(self.tracks)
                    if (not tr.dead) and (ti not in already_assigned) and tr.miss_count <= cfg.max_misses
                ]
                assigned_soft, used_soft, soft_stats = self._associate_existing_tracks(
                    frame, soft_pts, soft_desc,
                    track_indices=unmatched_indices,
                    source="soft_rescue",
                    normal_radius=cfg.soft_rescue_search_radius_px,
                    normal_ratio=cfg.soft_rescue_ratio,
                    normal_max_desc=cfg.soft_rescue_max_desc_dist,
                    # If the track was already lost before this frame, keep reacquisition
                    # at least as strict as both the reacq and soft-rescue settings.
                    reacq_radius=cfg.reacq_search_radius_px,
                    reacq_ratio=min(cfg.reacq_ratio, cfg.soft_rescue_ratio),
                    reacq_max_desc=min(cfg.reacq_max_desc_dist, cfg.soft_rescue_max_desc_dist),
                    disallowed_dets=disallowed_soft,
                )
                stats.time_soft_rescue_ms = 1000.0 * (time.perf_counter() - t_phase)

        assigned_all = set(assigned_lk) | set(assigned_primary) | set(assigned_soft)
        desc_vals = list(primary_stats.get("desc_vals", [])) + list(soft_stats.get("desc_vals", []))
        spatial_vals = list(primary_stats.get("spatial_vals", [])) + list(soft_stats.get("spatial_vals", []))
        stats.assoc_candidates = int(primary_stats["assoc_candidates"] + soft_stats["assoc_candidates"])
        stats.matched_primary = int(primary_stats["matched"])
        stats.soft_rescued = int(soft_stats["matched"])
        stats.matched = int(stats.lk_accepted + stats.matched_primary + stats.soft_rescued)
        stats.reacquired = int(primary_stats["reacquired"] + soft_stats["reacquired"])
        stats.median_desc = float(np.median(desc_vals)) if desc_vals else np.nan
        stats.median_spatial = float(np.median(spatial_vals)) if spatial_vals else np.nan

        missed = 0
        killed = 0
        t_phase = time.perf_counter()
        for ti, tr in enumerate(self.tracks):
            if tr.dead:
                continue
            if ti in assigned_all:
                continue
            tr.miss_count += 1
            tr.missed_last_step = True
            missed += 1
            if tr.miss_count > cfg.max_misses:
                tr.dead = True
                tr.killed_last_step = True
                killed += 1
        stats.missed = missed
        stats.killed = killed
        stats.time_miss_ms = 1000.0 * (time.perf_counter() - t_phase)

        # After both association passes, count active tracks and spawn only unused
        # primary/soft detections in under-filled buckets.
        t_phase = time.perf_counter()
        if run_spawn:
            self._spawn_tracks(frame, image, kps, desc, used_primary, soft_kps, soft_desc, used_soft, stats)
        stats.time_spawn_ms = 1000.0 * (time.perf_counter() - t_phase)

        if cfg.duplicate_suppression:
            t_phase = time.perf_counter()
            stats.dup_killed = self._suppress_duplicate_tracks(frame)
            stats.time_duplicate_ms = 1000.0 * (time.perf_counter() - t_phase)

        self._update_quality_scores(frame)
        t_phase = time.perf_counter()
        self._refresh_counts(frame, image, stats)
        stats.time_count_stats_ms = 1000.0 * (time.perf_counter() - t_phase)
        t_phase = time.perf_counter()
        if update_geom_stats:
            self._update_geom_stats(frame, stats)
        stats.time_geom_depth_pose_ms = 1000.0 * (time.perf_counter() - t_phase)
        t_phase = time.perf_counter()
        self._log_stats(stats)
        stats.time_log_ms = 1000.0 * (time.perf_counter() - t_phase)
        self.last_stats = stats
        self.prev_image = image.copy()
        self.prev_frame = frame
        return stats

    def _associate_existing_tracks(
        self,
        frame: int,
        det_pts: np.ndarray,
        det_desc: np.ndarray,
        track_indices: Optional[Sequence[int]] = None,
        source: str = "primary",
        normal_radius: Optional[float] = None,
        normal_ratio: Optional[float] = None,
        normal_max_desc: Optional[float] = None,
        reacq_radius: Optional[float] = None,
        reacq_ratio: Optional[float] = None,
        reacq_max_desc: Optional[float] = None,
        disallowed_dets: Optional[set] = None,
    ):
        """Associate existing tracks to a detection pool.

        This function is used twice in v3:
        1. primary pass: all old tracks -> primary SIFT detections;
        2. soft-rescue pass: still-unmatched old tracks -> soft SIFT detections.

        It only updates tracks that receive an accepted one-to-one assignment.
        Miss counts are updated later, after all association passes have failed.
        """
        cfg = self.cfg
        candidates: List[AssignmentCandidate] = []
        disallowed_dets = disallowed_dets or set()

        if det_pts.shape[0] == 0 or det_desc.shape[0] == 0:
            return [], set(), {
                "assoc_candidates": 0, "matched": 0, "reacquired": 0,
                "median_desc": np.nan, "median_spatial": np.nan,
                "desc_vals": [], "spatial_vals": [],
            }

        if track_indices is None:
            track_indices = range(len(self.tracks))

        normal_radius = cfg.search_radius_px if normal_radius is None else normal_radius
        normal_ratio = cfg.ratio if normal_ratio is None else normal_ratio
        normal_max_desc = cfg.max_desc_dist if normal_max_desc is None else normal_max_desc
        reacq_radius = cfg.reacq_search_radius_px if reacq_radius is None else reacq_radius
        reacq_ratio = cfg.reacq_ratio if reacq_ratio is None else reacq_ratio
        reacq_max_desc = cfg.reacq_max_desc_dist if reacq_max_desc is None else reacq_max_desc
        for ti in track_indices:
            tr = self.tracks[ti]
            if tr.dead or tr.miss_count > cfg.max_misses:
                continue

            pred = predict_track(tr, frame, cfg.prediction_mode)
            spatial = np.linalg.norm(det_pts - pred.reshape(1, 2), axis=1)

            is_reacq = tr.miss_count > 0
            radius = reacq_radius if is_reacq else normal_radius
            ratio_thr = reacq_ratio if is_reacq else normal_ratio
            max_desc = reacq_max_desc if is_reacq else normal_max_desc

            cand_idx = np.where(spatial <= radius)[0]
            if cand_idx.size == 0:
                continue
            if disallowed_dets:
                cand_idx = np.array([idx for idx in cand_idx if int(idx) not in disallowed_dets], dtype=int)
                if cand_idx.size == 0:
                    continue

            ref_desc = tr.latest_desc if cfg.descriptor_mode == "latest" else tr.anchor_desc
            dists = descriptor_distance(det_desc[cand_idx], ref_desc)
            order = np.argsort(dists)
            best_local = int(order[0])
            best_det = int(cand_idx[best_local])
            best_dist = float(dists[best_local])
            second_dist = float(dists[int(order[1])]) if len(order) >= 2 else float("inf")
            ratio_val = best_dist / (second_dist + 1e-12)
            spatial_err = float(spatial[best_det])

            if spatial_err > cfg.max_step_px:
                continue
            if best_dist > max_desc:
                continue
            if len(order) >= 2:
                if not (best_dist < ratio_thr * second_dist):
                    continue
            elif not cfg.allow_single_candidate:
                continue

            normalized_spatial = spatial_err / max(radius, 1e-9)
            cost = best_dist + cfg.spatial_weight * normalized_spatial * max_desc + cfg.miss_cost * tr.miss_count
            mode = "reacq" if is_reacq else source
            candidates.append(AssignmentCandidate(ti, best_det, best_dist, second_dist, ratio_val, spatial_err, float(cost), mode))

        candidates.sort(key=lambda a: (a.cost, a.desc_dist, a.spatial_err))
        used_tracks = set()
        used_dets = set()
        accepted: List[AssignmentCandidate] = []
        for a in candidates:
            if a.track_index in used_tracks or a.det_index in used_dets:
                continue
            used_tracks.add(a.track_index)
            used_dets.add(a.det_index)
            accepted.append(a)

        desc_vals = []
        spatial_vals = []
        reacq = 0
        for a in accepted:
            tr = self.tracks[a.track_index]
            was_reacq = tr.miss_count > 0
            if was_reacq:
                reacq += 1
                tr.reacq_count += 1
            pt = det_pts[a.det_index].astype(np.float32)
            tr.observations.append(Observation(frame=frame, pt=pt))
            tr.latest_desc = det_desc[a.det_index].copy()
            tr.frames_since_desc_refresh = 0
            tr.hit_count += 1
            tr.confirmed = tr.confirmed or (tr.hit_count >= tr.required_hits_to_confirm)
            tr.miss_count = 0
            tr.last_desc_dist = a.desc_dist
            tr.last_anchor_desc_dist = float(descriptor_distance(det_desc[a.det_index].reshape(1, -1), tr.anchor_desc)[0])
            tr.last_ratio = a.ratio_value
            tr.last_spatial_err = a.spatial_err
            tr.last_cost = a.cost
            tr.last_source = source
            tr.matched_last_step = True
            tr.reacquired_last_step = was_reacq
            # Probation is for true reacquisition after a miss, not simply for soft-rescue.
            # A clean successful match in the next frame clears previous probation.
            tr.reacq_probation = was_reacq
            tr.missed_last_step = False
            desc_vals.append(a.desc_dist)
            spatial_vals.append(a.spatial_err)

        return sorted(list(used_tracks)), used_dets, {
            "assoc_candidates": len(candidates),
            "matched": len(accepted),
            "reacquired": reacq,
            "median_desc": float(np.median(desc_vals)) if desc_vals else np.nan,
            "median_spatial": float(np.median(spatial_vals)) if spatial_vals else np.nan,
            "desc_vals": desc_vals,
            "spatial_vals": spatial_vals,
        }

    def _associate_mnn(self, frame: int, det_pts: np.ndarray, det_desc: np.ndarray,
                       track_indices: Sequence[int], source: str,
                       disallowed_dets: Optional[set] = None):
        """XFeat-style global mutual-NN + cosine-threshold association.

        Replaces the per-track radius+Lowe matcher: all eligible track
        descriptors are matched against all current detections by mutual nearest
        neighbor on cosine similarity, gated by ``xfeat_mnn_min_cossim`` and a
        max-displacement sanity cap. This is XFeat's own ``match()`` rule (not the
        LighterGlue attention matcher); there is no spatial *search* gate.
        """
        cfg = self.cfg
        empty = {"assoc_candidates": 0, "matched": 0, "reacquired": 0,
                 "median_desc": np.nan, "median_spatial": np.nan,
                 "desc_vals": [], "spatial_vals": []}
        disallowed_dets = disallowed_dets or set()
        elig = [ti for ti in track_indices
                if (not self.tracks[ti].dead) and self.tracks[ti].miss_count <= cfg.max_misses]
        if det_pts.shape[0] == 0 or det_desc.shape[0] == 0 or not elig:
            return [], set(), empty

        T = np.array([self.tracks[ti].latest_desc for ti in elig], dtype=np.float64)
        Dm = det_desc.astype(np.float64)
        Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
        Dn = Dm / (np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12)
        cossim = Tn @ Dn.T                       # (n_tracks, n_dets)
        match12 = np.argmax(cossim, axis=1)      # best detection per track
        match21 = np.argmax(cossim, axis=0)      # best track per detection
        max_step = cfg.xfeat_mnn_max_step_px if cfg.xfeat_mnn_max_step_px > 0 else cfg.max_step_px

        used_tracks: set = set()
        used_dets: set = set()
        reacq = 0
        desc_vals: List[float] = []
        spatial_vals: List[float] = []
        for k, ti in enumerate(elig):
            d = int(match12[k])
            if d in disallowed_dets or d in used_dets:
                continue
            if match21[d] != k:                   # mutual nearest-neighbor check
                continue
            cs = float(cossim[k, d])
            if cs < cfg.xfeat_mnn_min_cossim:
                continue
            tr = self.tracks[ti]
            pred = predict_track(tr, frame, cfg.prediction_mode)
            spatial = float(np.linalg.norm(det_pts[d].astype(np.float64) - pred.astype(np.float64)))
            if max_step > 0 and spatial > max_step:
                continue
            was_reacq = tr.miss_count > 0
            if was_reacq:
                reacq += 1
                tr.reacq_count += 1
            pt = det_pts[d].astype(np.float32)
            desc_dist = float(1.0 - cs)            # cosine distance as a pseudo desc-dist
            tr.observations.append(Observation(frame=frame, pt=pt))
            tr.latest_desc = det_desc[d].copy()
            tr.frames_since_desc_refresh = 0
            tr.hit_count += 1
            tr.confirmed = tr.confirmed or (tr.hit_count >= tr.required_hits_to_confirm)
            tr.miss_count = 0
            tr.last_desc_dist = desc_dist
            tr.last_anchor_desc_dist = desc_dist
            tr.last_ratio = np.nan
            tr.last_spatial_err = spatial
            tr.last_cost = desc_dist
            tr.last_source = source
            tr.matched_last_step = True
            tr.reacquired_last_step = was_reacq
            tr.reacq_probation = was_reacq
            tr.missed_last_step = False
            used_tracks.add(ti)
            used_dets.add(d)
            desc_vals.append(desc_dist)
            spatial_vals.append(spatial)

        return sorted(list(used_tracks)), used_dets, {
            "assoc_candidates": len(elig),
            "matched": len(used_tracks),
            "reacquired": reacq,
            "median_desc": float(np.median(desc_vals)) if desc_vals else np.nan,
            "median_spatial": float(np.median(spatial_vals)) if spatial_vals else np.nan,
            "desc_vals": desc_vals,
            "spatial_vals": spatial_vals,
        }

    def _spawn_tracks(self, frame: int, image: np.ndarray, primary_kps: Sequence[cv2.KeyPoint], primary_desc: np.ndarray,
                      used_primary: set, soft_kps: Sequence[cv2.KeyPoint], soft_desc: np.ndarray,
                      used_soft: set, stats: FrameStats):
        cfg = self.cfg
        h, w = image.shape[:2]
        counts_active, counts_confirmed = self.basic_bucket_counts(frame, image.shape)
        counts = counts_active if cfg.spawn_count_candidates else counts_confirmed

        active_pts = np.array([tr.last_pt() for tr in self.tracks if tr.active_in(frame) and not tr.dead], dtype=np.float32)
        if active_pts.size == 0:
            active_pts = np.empty((0, 2), dtype=np.float32)
        active_hash = SpatialHash(cfg.min_spawn_distance_px)
        active_hash.add_many(active_pts)

        candidates_by_cell: Dict[int, List[Tuple[str, int, cv2.KeyPoint, np.ndarray]]] = {i: [] for i in range(cfg.grid_rows * cfg.grid_cols)}

        for idx, kp in enumerate(primary_kps):
            if idx in used_primary:
                continue
            pt = np.array(kp.pt, dtype=np.float32)
            if active_hash.too_close(pt):
                continue
            _, _, bi = bucket_index(pt, w, h, cfg.grid_cols, cfg.grid_rows)
            candidates_by_cell[bi].append(("primary", idx, kp, primary_desc[idx]))

        if cfg.soft_spawn:
            primary_pts_all = keypoint_points(primary_kps)
            primary_hash = SpatialHash(cfg.soft_spawn_primary_sep_px) if cfg.soft_spawn_primary_sep_px > 0 else None
            if primary_hash is not None:
                primary_hash.add_many(primary_pts_all)
            for idx, kp in enumerate(soft_kps):
                if idx in used_soft:
                    continue
                pt = np.array(kp.pt, dtype=np.float32)
                # Soft spawn candidates must not duplicate an already-active track
                # and should also not duplicate a primary detection. This is only
                # for spawning; soft rescue association above may still use soft
                # detections near unused primary detections.
                if active_hash.too_close(pt):
                    continue
                if primary_hash is not None and primary_hash.too_close(pt):
                    continue
                _, _, bi = bucket_index(pt, w, h, cfg.grid_cols, cfg.grid_rows)
                candidates_by_cell[bi].append(("soft", idx, kp, soft_desc[idx]))

        for bi, lst in candidates_by_cell.items():
            # Primary first; for equal source, higher SIFT response.
            lst.sort(key=lambda x: (0 if x[0] == "primary" else 1, -float(x[2].response)))

        active_total = sum(1 for tr in self.tracks if tr.active_in(frame) and not tr.dead)
        spawned_primary = 0
        spawned_soft = 0

        for r in range(cfg.grid_rows):
            for c in range(cfg.grid_cols):
                if active_total >= cfg.max_active_tracks:
                    break
                need = max(0, cfg.target_per_bucket - int(counts[r, c]))
                if need <= 0:
                    continue
                bi = r * cfg.grid_cols + c
                accepted_in_cell = 0
                for source, idx, kp, desc in candidates_by_cell[bi]:
                    if accepted_in_cell >= need or active_total >= cfg.max_active_tracks:
                        break
                    pt = np.array(kp.pt, dtype=np.float32)
                    if active_hash.too_close(pt):
                        continue
                    req_hits = cfg.soft_confirm_hits if source == "soft" else cfg.min_hits_to_confirm
                    tr = Track(
                        id=self.next_id,
                        birth_frame=frame,
                        anchor_pt=pt.copy(),
                        anchor_desc=desc.copy(),
                        latest_desc=desc.copy(),
                        required_hits_to_confirm=req_hits,
                        observations=[Observation(frame=frame, pt=pt.copy())],
                        hit_count=1,
                        miss_count=0,
                        confirmed=False,
                        dead=False,
                        last_source=source,
                        newborn_last_step=True,
                        matched_last_step=True,
                    )
                    self.next_id += 1
                    self.tracks.append(tr)
                    if source == "primary":
                        spawned_primary += 1
                    else:
                        spawned_soft += 1
                    active_hash.add(pt)
                    counts[r, c] += 1
                    active_total += 1
                    accepted_in_cell += 1

        stats.spawned_primary = spawned_primary
        stats.spawned_soft = spawned_soft

    def _suppress_duplicate_tracks(self, frame: int) -> int:
        cfg = self.cfg
        active = self.active_tracks(frame, confirmed_only=False)
        if len(active) <= 1:
            return 0
        self._update_quality_scores(frame)
        active.sort(key=lambda tr: (-tr.quality_score, -tr.hit_count, tr.id))
        accepted_hash = SpatialHash(cfg.duplicate_dist_px)
        confirmed_tight_hash = SpatialHash(max(2.0, 0.5 * cfg.duplicate_dist_px))
        killed = 0
        for tr in active:
            pt = tr.last_pt()
            is_duplicate = accepted_hash.too_close(pt)
            # Prefer not to hard-kill very stable confirmed tracks unless duplicate is very close.
            if is_duplicate and ((not tr.confirmed) or confirmed_tight_hash.too_close(pt)):
                tr.dead = True
                tr.killed_last_step = True
                tr.duplicate_killed_last_step = True
                killed += 1
                continue
            accepted_hash.add(pt)
            confirmed_tight_hash.add(pt)
        return killed

    def track_quality(self, tr: Track, current_frame: int) -> float:
        score = 0.0
        score += self.cfg.quality_confirmed_bonus if tr.confirmed else 0.0
        score += min(self.cfg.quality_hit_cap, self.cfg.quality_hit_weight * tr.hit_count)
        score -= self.cfg.quality_miss_penalty * tr.miss_count
        score -= self.cfg.quality_reacq_penalty * tr.reacq_count
        score -= self.cfg.quality_probation_penalty if tr.reacq_probation else 0.0
        if tr.last_source in ("soft", "soft_rescue"):
            score -= self.cfg.soft_source_penalty
        if tr.last_source == "lk":
            score += self.cfg.quality_lk_bonus
        if tr.frames_since_desc_refresh > self.cfg.quality_lk_stale_after:
            stale_frames = tr.frames_since_desc_refresh - self.cfg.quality_lk_stale_after
            score -= min(self.cfg.quality_lk_stale_cap, self.cfg.quality_lk_stale_penalty * stale_frames)
        if np.isfinite(tr.last_desc_dist):
            score -= self.cfg.quality_desc_penalty * tr.last_desc_dist
        if np.isfinite(tr.last_spatial_err):
            score -= self.cfg.quality_spatial_penalty * tr.last_spatial_err
        # Prefer currently active tracks.
        score += self.cfg.quality_active_bonus if tr.active_in(current_frame) else 0.0
        return float(score)

    def _update_quality_scores(self, frame: int):
        for tr in self.tracks:
            if tr.dead:
                tr.quality_score = -1e9
            else:
                tr.quality_score = self.track_quality(tr, frame)

    def basic_bucket_counts(self, frame: int, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = shape[:2]
        active_counts = np.zeros((self.cfg.grid_rows, self.cfg.grid_cols), dtype=np.int32)
        confirmed_counts = np.zeros_like(active_counts)
        for tr in self.tracks:
            if tr.dead or not tr.active_in(frame):
                continue
            r, c, _ = bucket_index(tr.last_pt(), w, h, self.cfg.grid_cols, self.cfg.grid_rows)
            active_counts[r, c] += 1
            if tr.confirmed:
                confirmed_counts[r, c] += 1
        return active_counts, confirmed_counts

    def bucket_counts(self, frame: int, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = shape[:2]
        active_counts, confirmed_counts = self.basic_bucket_counts(frame, shape)
        triang_counts = np.zeros_like(active_counts)
        triang_infos = self.evaluate_triang_candidates(frame, compute_dlt=False, fast=True)
        for tr in self.tracks:
            if tr.dead or not tr.active_in(frame):
                continue
            r, c, _ = bucket_index(tr.last_pt(), w, h, self.cfg.grid_cols, self.cfg.grid_rows)
            info = triang_infos.get(tr.id)
            if info is not None and info.label in ("good", "candidate_good", "reacq_good", "confirmed_good"):
                triang_counts[r, c] += 1
        return active_counts, confirmed_counts, triang_counts

    def _refresh_counts(self, frame: int, image: np.ndarray, stats: FrameStats):
        active = [tr for tr in self.tracks if tr.active_in(frame) and not tr.dead]
        confirmed_active = [tr for tr in active if tr.confirmed]
        candidate_active = [tr for tr in active if not tr.confirmed]
        temp_lost = [tr for tr in self.tracks if (not tr.dead) and tr.miss_count > 0]
        dead = [tr for tr in self.tracks if tr.dead]
        stats.active = len(active)
        stats.confirmed_active = len(confirmed_active)
        stats.candidate_active = len(candidate_active)
        stats.temp_lost = len(temp_lost)
        stats.dead_total = len(dead)
        counts, _ = self.basic_bucket_counts(frame, image.shape)
        occupied = np.sum(counts > 0)
        stats.bucket_coverage = float(occupied / max(1, self.cfg.grid_rows * self.cfg.grid_cols))
        stats.underfilled_buckets = int(np.sum(counts < self.cfg.target_per_bucket))
        stats.overfilled_buckets = int(np.sum(counts > self.cfg.max_per_bucket))
        stats.mean_bucket_active = float(np.mean(counts)) if counts.size else 0.0
        stats.min_bucket_active = int(np.min(counts)) if counts.size else 0
        selected = self.select_tracks_for_display(frame, image.shape)
        stats.selected_display = len(selected)

    def _update_geom_stats(self, frame: int, stats: FrameStats):
        infos = self.evaluate_triang_candidates(frame, compute_dlt=True, fast=False)
        vals_depth = []
        vals_reproj = []
        for tr in self.active_tracks(frame, confirmed_only=False):
            info = infos.get(tr.id)
            if info is None:
                continue
            if info.label == "confirmed_good":
                stats.triang_good += 1
                stats.triang_confirmed_good += 1
            elif info.label == "candidate_good":
                stats.triang_good += 1
                stats.triang_candidate_good += 1
            elif info.label == "reacq_good":
                stats.triang_good += 1
                stats.triang_reacq_good += 1
            elif info.label == "low_parallax":
                stats.triang_low_parallax += 1
            elif info.label == "bad_epi":
                stats.triang_bad_epi += 1
            elif info.label == "bad_reproj":
                stats.triang_bad_reproj += 1
            elif info.label == "bad_depth":
                stats.triang_bad_depth += 1
            else:
                stats.triang_no_pair += 1
            if info.label in ("confirmed_good", "candidate_good", "reacq_good") and np.isfinite(info.depth_m):
                vals_depth.append(info.depth_m)
                vals_reproj.append(info.reproj_px)
        stats.depth_valid = len(vals_depth)
        stats.median_depth = float(np.median(vals_depth)) if vals_depth else np.nan
        stats.median_reproj = float(np.median(vals_reproj)) if vals_reproj else np.nan
        pose_eval = self.evaluate_pose_gap(frame)
        stats.pose_pairs = pose_eval.pairs
        stats.pose_inliers = pose_eval.inliers
        stats.pose_inlier_ratio = float(pose_eval.inliers / pose_eval.pairs) if pose_eval.pairs > 0 else np.nan
        stats.pose_rot_err_deg = pose_eval.rot_err_deg
        stats.pose_t_err_deg = pose_eval.t_err_deg

    def active_tracks(self, frame: int, confirmed_only=False) -> List[Track]:
        return [tr for tr in self.tracks if (not tr.dead) and tr.active_in(frame) and ((not confirmed_only) or tr.confirmed)]

    def select_tracks_for_display(
        self,
        frame: int,
        shape: Tuple[int, int],
        confirmed_only: bool = False,
        per_bucket: Optional[int] = None,
        max_tracks: Optional[int] = None,
    ) -> List[Track]:
        # Soft per-bucket selection. Does not kill tracks; just decides what to draw/output.
        per_bucket = self.cfg.output_per_bucket if per_bucket is None else per_bucket
        max_tracks = self.cfg.output_max_tracks if max_tracks is None else max_tracks
        h, w = shape[:2]
        by_cell: Dict[int, List[Track]] = {i: [] for i in range(self.cfg.grid_rows * self.cfg.grid_cols)}
        for tr in self.active_tracks(frame, confirmed_only=confirmed_only):
            _, _, bi = bucket_index(tr.last_pt(), w, h, self.cfg.grid_cols, self.cfg.grid_rows)
            by_cell[bi].append(tr)
        selected: List[Track] = []
        for cell_tracks in by_cell.values():
            cell_tracks.sort(key=lambda tr: (-tr.quality_score, -tr.hit_count, tr.id))
            selected.extend(cell_tracks[: max(1, per_bucket)])
        selected.sort(key=lambda tr: (-tr.quality_score, tr.id))
        if len(selected) > max_tracks:
            selected = selected[: max_tracks]
        return selected

    def evaluate_triang_candidates(self, frame: int, compute_dlt: bool = True, fast: bool = False) -> Dict[int, TriangInfo]:
        return evaluate_best_pair_triangulation(
            self.active_tracks(frame, confirmed_only=False),
            frame,
            self.K,
            self.poses,
            self.cfg,
            compute_dlt=compute_dlt,
            fast=fast,
        )

    def evaluate_pose_gap(self, frame: int) -> PoseEval:
        return evaluate_pose_gap_from_tracks(
            self.active_tracks(frame, confirmed_only=False),
            frame,
            self.K,
            self.poses,
            self.cfg,
        )

    def _log_stats(self, stats: FrameStats):
        if self.log_writer is None:
            return
        row = {k: getattr(stats, k) for k in FrameStats.__dataclass_fields__.keys()}
        self.log_writer.writerow(row)
        if self.log_file is not None:
            self.log_file.flush()




# Backwards-compatible aliases for scripts that still import the old class names.
SiftLkFeatureManager = SiftFeatureManager = FeatureManager
