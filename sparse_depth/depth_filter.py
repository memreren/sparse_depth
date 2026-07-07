"""Keyframe inverse-depth filter (SVO/LSD-inspired mapping, known poses).

Each *seed* is one pixel in a reference (key)frame whose inverse depth d = 1/z
is modeled as a Gaussian N(mu, sigma2). For every new frame with known relative
pose, the seed's current belief mu +/- 2*sigma bounds a short segment of the
epipolar line in the new image; a ZNCC patch search along that segment yields a
depth measurement, which is fused into the belief with a 1-D Kalman/Gaussian
update. Converged seeds (low relative uncertainty) are promoted to output
points anchored in the reference frame.

This module has no KITTI IO and no drawing: it consumes grayscale images,
an intrinsic matrix K, and a pose list compatible with geometry.relative_pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from sparse_depth.geometry import relative_pose


@dataclass
class DepthFilterConfig:
    # Seeding. "corners" = Shi-Tomasi (sparse, strong texture); "gradient" =
    # every local gradient maximum above grad_thresh (semi-dense, LSD-style).
    seed_mode: str = "gradient"
    max_seeds: int = 2500
    seed_quality_level: float = 0.005
    seed_min_distance_px: float = 10.0
    seed_block_size: int = 7
    grad_thresh: float = 20.0          # Sobel-magnitude floor for gradient seeds
    seed_cell_px: int = 16             # bucketing cell for seeding + coverage
    seeds_per_cell: int = 3            # max new gradient seeds per cell
    cell_covered_count: int = 2        # skip seeding cells with this many map points

    # Keyframe policy (process()): switch reference when too few active seeds
    # still see the current frame, or the baseline has grown too long.
    kf_min_gap: int = 3                # never switch faster than this many frames
    kf_visible_frac: float = 0.5       # switch when active in-view fraction drops below
    kf_max_baseline_m: float = 8.0     # or when baseline from reference exceeds this
    kf_min_active: int = 50            # or when almost no seeds remain active

    # Scene depth prior: the initial belief spans [z_min, z_max].
    z_min: float = 2.0
    z_max: float = 80.0

    # Patch matching.
    patch_size: int = 8               # square patch side in pixels
    search_step_px: float = 0.7       # sample spacing along the epipolar segment
    # The epipolar segment is CLIPPED to the image rectangle (never shrunk
    # around the belief mean: for an uninformed seed the mean is the near-biased
    # prior, and re-centering the search on it loses the true depth entirely).
    # max_search_px is retained for CLI compatibility but no longer used.
    max_search_px: float = 300.0
    max_search_samples: int = 640     # sample cap; step adapts between 0.7-2 px
    min_search_px: float = 1.0        # below this the frame carries ~no depth info
    # Never let the searched window shrink below this: once +/-2 sigma maps to a
    # couple of pixels, the window IS the prior and the filter self-confirms its
    # own early noise (matches clamp to the window edge and re-fuse the belief).
    min_search_window_px: float = 8.0
    zncc_min: float = 0.85            # accept a match only above this score
    # Peak distinctiveness: reject when a second correlation peak (outside the
    # best peak's own lobe) comes within this margin of the best. Repetitive
    # structure (lane dashes, guardrails, fence posts) produces near-equal
    # peaks at the wrong period; such CONSISTENTLY wrong matches pass every
    # temporal-consistency gate, so they must be rejected at match time.
    peak_margin: float = 0.08
    border_px: int = 8                # keep patches away from the image border

    # Measurement noise / fusion.
    pixel_sigma: float = 1.0          # assumed matching noise in pixels
    # Floor on per-measurement noise, relative to the measured inverse depth.
    # Models un-modeled correlated error (patch appearance drift, subpixel bias)
    # so a single geometrically-sharp observation cannot collapse the belief.
    min_rel_meas_sigma: float = 0.02
    outlier_gate_sigmas: float = 2.5  # innovation gate in combined sigmas
    # Per-frame process noise (relative to mu). Early measurements are strongly
    # correlated (same appearance, similar geometry), so pure Gaussian fusion
    # collapses sigma far below the true error; the belief then gates out the
    # *sharper* wide-baseline measurements that arrive later. A small inflation
    # each frame keeps the filter permanently able to accept new evidence.
    process_noise_rel: float = 0.01

    # Seed lifecycle. (0.10/4 trades ~0.7pp median accuracy for ~75% more
    # density vs the strict 0.05/5 — the right default for a depth MAP.)
    converge_sigma_ratio: float = 0.10  # promote when sigma_d / mu_d < this
    converge_min_obs: int = 4
    # A seed must accumulate real triangulation geometry before promotion.
    # Points near the focus of expansion have ~zero parallax under forward
    # motion: their tiny search windows self-confirm, sigma shrinks on no real
    # information, and they converge to confident garbage (typically far
    # points overestimated 30-50%). Parallax is the honest information gauge.
    min_parallax_deg: float = 1.0
    max_attempts: int = 40              # kill un-converged seeds after this many update tries
    kill_outlier_margin: int = 5        # kill when n_out - n_in > this
    # Keep refining converged seeds instead of freezing them: later frames have
    # larger baselines and much sharper measurements than the ones that first
    # pushed the seed over the convergence threshold.
    refine_converged: bool = True

    # Beta inlier model (SVO): each seed tracks a running inlier/outlier tally.
    # Seeds must be predominantly-inlier to be promoted; chronically-outlier
    # seeds die early instead of converging onto repeated mismatches.
    use_beta_inlier: bool = True
    beta_prior_a: float = 10.0
    beta_prior_b: float = 10.0
    beta_harvest_min: float = 0.55    # min inlier ratio to allow promotion
    beta_kill_below: float = 0.35     # kill when ratio drops below (after 6 tries)

    # Photometric measurement noise: matching precision along the epipolar line
    # is limited by image gradient there (LSD-style). sigma_px ~ sigma_I / |grad|.
    photometric_sigma: bool = True
    sigma_intensity: float = 4.0      # assumed intensity noise (0-255 scale)

    # Map hygiene: re-verify visible map points photometrically each frame and
    # prune those that keep failing (occluded / dynamic / wrong depth). Without
    # this, occluded points are scored against their occluder's LiDAR and
    # dominate the MEAN error tail.
    validate_map: bool = True
    map_zncc_min: float = 0.5         # below this counts as a miss
    map_zncc_refresh: float = 0.7     # above this the stored patch is refreshed
    map_max_misses: int = 3           # prune after this many consecutive misses
    # Self-occlusion consistency: a projected map point much deeper than its
    # image-cell neighbors is behind them (occluded or plain wrong) — e.g. a
    # bogus 60 m point drifting across 8 m ground at the image corners, where
    # low-texture patches can pass the photometric check by luck.
    map_occl_cell_px: int = 12
    map_occl_ratio: float = 2.0       # miss when z > ratio * cell minimum


# Seed status codes (kept as ints so state lives in numpy arrays).
SEED_ACTIVE = 0
SEED_CONVERGED = 1
SEED_DEAD = 2


def _in_box(p: np.ndarray, w: int, h: int, margin: float) -> bool:
    return bool(margin <= p[0] < w - margin and margin <= p[1] < h - margin)


def _clip_segment(p0: np.ndarray, p1: np.ndarray, w: int, h: int, margin: float):
    """Liang-Barsky clip of segment [p0, p1] to the margin-inset image rect."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], p0[0] - margin), (d[0], (w - margin) - p0[0]),
                 (-d[1], p0[1] - margin), (d[1], (h - margin) - p0[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    if t0 > t1:
        return None
    return p0 + t0 * d, p0 + t1 * d


@dataclass
class FrameUpdateStats:
    frame: int = -1
    alive: int = 0
    searched: int = 0
    active_searched: int = 0
    matched: int = 0
    fused: int = 0
    outliers: int = 0
    no_view: int = 0
    tiny_segment: int = 0
    new_converged: int = 0
    total_converged: int = 0
    median_sigma_ratio: float = float("nan")
    update_ms: float = 0.0
    baseline_m: float = 0.0
    # Keyframe bookkeeping (filled by process()).
    kf_switched: bool = False
    harvested: int = 0
    reseeded: int = 0
    map_size: int = 0


class DepthFilterMapper:
    """Holds one reference frame's seeds and refines them against new frames."""

    def __init__(self, K: np.ndarray, poses, cfg: Optional[DepthFilterConfig] = None):
        self.K = np.asarray(K, dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.poses = poses
        self.cfg = cfg or DepthFilterConfig()

        self.ref_frame: int = -1
        self.ref_img: Optional[np.ndarray] = None

        # Per-seed parallel arrays (allocated in set_reference).
        self.uv = np.zeros((0, 2), dtype=np.float64)       # ref-frame pixel
        self.f_ref = np.zeros((0, 3), dtype=np.float64)    # K^-1 [u v 1] (z=1 bearing)
        self.mu = np.zeros((0,), dtype=np.float64)         # inverse-depth mean
        self.sigma2 = np.zeros((0,), dtype=np.float64)     # inverse-depth variance
        self.n_in = np.zeros((0,), dtype=np.int32)
        self.n_out = np.zeros((0,), dtype=np.int32)
        self.attempts = np.zeros((0,), dtype=np.int32)
        self.status = np.zeros((0,), dtype=np.int32)
        self.patches = np.zeros((0, 0, 0), dtype=np.float32)

        # Persistent world map: converged seeds harvested at keyframe switches.
        self.map_xyz = np.zeros((0, 3), dtype=np.float64)      # world coordinates
        self.map_sigma_rel = np.zeros((0,), dtype=np.float64)  # sigma_z / z at harvest
        self.map_kf = np.zeros((0,), dtype=np.int32)           # source keyframe
        p = self.cfg.patch_size
        self.map_patch = np.zeros((0, p, p), dtype=np.float32)  # appearance at harvest
        self.map_miss = np.zeros((0,), dtype=np.int32)          # consecutive photometric misses
        self.map_alive = np.zeros((0,), dtype=bool)
        self.frames_since_ref: int = 0

        # Debug info for the most recent update, keyed by seed index.
        self.last_segments: Dict[int, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float]] = {}

    # ------------------------------------------------------------------ setup

    def set_reference(self, frame: int, img: np.ndarray) -> Tuple[int, int, int]:
        """Make `img` the new reference (keyframe).

        Converged seeds are first harvested into the persistent world map;
        un-converged seeds are dropped. New seeds are placed only in image
        cells not already covered by projected map points.

        Returns (n_new_seeds, n_harvested, n_dropped).
        """
        cfg = self.cfg
        harvested = self._harvest_converged()
        dropped = int(np.sum(self.status == SEED_ACTIVE))

        self.ref_frame = int(frame)
        self.ref_img = img
        self.frames_since_ref = 0

        covered = self._covered_cells(frame, img.shape)
        if cfg.seed_mode == "gradient":
            uv = self._seed_gradient(img, covered)
        else:
            uv = self._seed_corners(img, covered)
        n = uv.shape[0]

        ones = np.ones((n, 1), dtype=np.float64)
        self.uv = uv
        self.f_ref = (self.K_inv @ np.hstack([uv, ones]).T).T  # z-component == 1

        d_lo = 1.0 / cfg.z_max
        d_hi = 1.0 / cfg.z_min
        self.mu = np.full(n, 0.5 * (d_lo + d_hi), dtype=np.float64)
        # +/- 2 sigma covers the whole prior range.
        self.sigma2 = np.full(n, ((d_hi - d_lo) / 4.0) ** 2, dtype=np.float64)
        self.n_in = np.zeros(n, dtype=np.int32)
        self.n_out = np.zeros(n, dtype=np.int32)
        self.attempts = np.zeros(n, dtype=np.int32)
        self.status = np.full(n, SEED_ACTIVE, dtype=np.int32)
        self.beta_a = np.full(n, cfg.beta_prior_a, dtype=np.float64)
        self.beta_b = np.full(n, cfg.beta_prior_b, dtype=np.float64)
        self.parallax_deg = np.zeros(n, dtype=np.float64)  # max parallax seen

        p = cfg.patch_size
        self.patches = np.zeros((n, p, p), dtype=np.float32)
        for i in range(n):
            self.patches[i] = cv2.getRectSubPix(img, (p, p), tuple(uv[i]))
        self.last_segments = {}
        return n, harvested, dropped

    def _seed_corners(self, img: np.ndarray, covered: np.ndarray) -> np.ndarray:
        """Shi-Tomasi seeding, skipping map-covered cells."""
        cfg = self.cfg
        corners = cv2.goodFeaturesToTrack(
            img,
            maxCorners=cfg.max_seeds,
            qualityLevel=cfg.seed_quality_level,
            minDistance=cfg.seed_min_distance_px,
            blockSize=cfg.seed_block_size,
        )
        if corners is None:
            return np.zeros((0, 2), dtype=np.float64)
        uv = corners.reshape(-1, 2).astype(np.float64)
        return self._filter_seed_candidates(uv, img.shape, covered, per_cell=None)

    def _seed_gradient(self, img: np.ndarray, covered: np.ndarray) -> np.ndarray:
        """Semi-dense seeding: local Sobel-magnitude maxima above grad_thresh."""
        cfg = self.cfg
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # Local maxima (5x5) so seeds do not stack on one edge pixel run.
        localmax = (mag >= cv2.dilate(mag, np.ones((5, 5), np.uint8)) - 1e-6) & (mag > cfg.grad_thresh)
        ys, xs = np.nonzero(localmax)
        if xs.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        order = np.argsort(-mag[ys, xs])  # strongest gradient first
        uv = np.stack([xs[order], ys[order]], axis=1).astype(np.float64)
        return self._filter_seed_candidates(uv, img.shape, covered, per_cell=cfg.seeds_per_cell)

    def _filter_seed_candidates(self, uv: np.ndarray, shape, covered: np.ndarray,
                                per_cell: Optional[int]) -> np.ndarray:
        """Border filter + skip covered cells + optional per-cell cap + global cap."""
        cfg = self.cfg
        h, w = shape[:2]
        b = cfg.border_px
        keep = (uv[:, 0] >= b) & (uv[:, 0] < w - b) & (uv[:, 1] >= b) & (uv[:, 1] < h - b)
        uv = uv[keep]
        c = cfg.seed_cell_px
        rows, cols = covered.shape
        out = []
        counts = np.zeros_like(covered, dtype=np.int32)
        for u, v in uv:
            r, k = min(int(v // c), rows - 1), min(int(u // c), cols - 1)
            if covered[r, k]:
                continue
            if per_cell is not None and counts[r, k] >= per_cell:
                continue
            counts[r, k] += 1
            out.append((u, v))
            if len(out) >= cfg.max_seeds:
                break
        return np.asarray(out, dtype=np.float64).reshape(-1, 2)

    def _covered_cells(self, frame: int, shape) -> np.ndarray:
        """Boolean cell grid: True where projected map points already cover."""
        cfg = self.cfg
        h, w = shape[:2]
        c = cfg.seed_cell_px
        rows, cols = (h + c - 1) // c, (w + c - 1) // c
        counts = np.zeros((rows, cols), dtype=np.int32)
        uv, _z, _idx = self.map_points_in(frame, shape)
        for u, v in uv:
            counts[min(int(v // c), rows - 1), min(int(u // c), cols - 1)] += 1
        return counts >= cfg.cell_covered_count

    # ----------------------------------------------------------------- update

    def update(self, frame: int, img: np.ndarray) -> FrameUpdateStats:
        """Run one epipolar-search + fusion pass of all active seeds against `img`."""
        t0 = cv2.getTickCount()
        cfg = self.cfg
        stats = FrameUpdateStats(frame=int(frame))
        self.last_segments = {}
        if self.ref_img is None:
            return stats

        R, t = relative_pose(self.poses, self.ref_frame, int(frame))
        stats.baseline_m = float(np.linalg.norm(t))
        Kt = self.K @ t.reshape(3)
        KR = self.K @ R
        h, w = img.shape[:2]
        p = cfg.patch_size
        half = p / 2.0
        d_floor = 1.0 / cfg.z_max
        d_ceil = 1.0 / cfg.z_min

        if cfg.refine_converged:
            active = np.where(self.status != SEED_DEAD)[0]
        else:
            active = np.where(self.status == SEED_ACTIVE)[0]
        stats.alive = int(np.sum(self.status == SEED_ACTIVE))

        # Process noise: keep beliefs receptive to new (sharper) measurements.
        if cfg.process_noise_rel > 0 and active.size:
            self.sigma2[active] += (cfg.process_noise_rel * self.mu[active]) ** 2

        for i in active:
            is_active = self.status[i] == SEED_ACTIVE
            if is_active:
                self.attempts[i] += 1
            A = KR @ self.f_ref[i]            # projective point at d -> A + d * Kt
            sig = np.sqrt(self.sigma2[i])
            d_lo = max(self.mu[i] - 2.0 * sig, d_floor)
            d_hi = min(self.mu[i] + 2.0 * sig, d_ceil)

            # Keep the candidate 3D point in front of the current camera.
            tz = Kt[2]
            if abs(tz) > 1e-12:
                d_at_zmin = (0.1 - A[2]) / tz
                if tz < 0:
                    d_hi = min(d_hi, d_at_zmin)
                else:
                    d_lo = max(d_lo, d_at_zmin)
            if not (d_hi > d_lo > 0):
                stats.no_view += 1
                self._age_kill(i)
                continue

            m = half + 1.0
            seg = self._segment_pixels(A, Kt, d_lo, d_hi, w, h, m)
            if seg is None:
                stats.no_view += 1
                self._age_kill(i)
                continue
            p0, p1 = seg
            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len < cfg.min_search_px:
                stats.tiny_segment += 1
                self._age_kill(i)
                continue
            stats.searched += 1
            if is_active:
                stats.active_searched += 1

            # Affine warp of the patch grid: forward motion magnifies the scene,
            # and matching an unwarped patch biases the ZNCC peak toward the
            # near-depth end of the segment (radially outward from the FOE).
            # Only warp once the belief has been informed by at least one
            # measurement: an unfused seed's mean is still the (near-biased)
            # prior, and a warp computed at that bogus depth implies absurd
            # magnification and would wrongly disqualify the seed.
            W = None
            if self.n_in[i] > 0:
                W = self._patch_warp(KR, Kt, i)
                if W is None:
                    stats.no_view += 1
                    self._age_kill(i)
                    continue

            match = self._search_segment(img, self.patches[i], p0, p1, w, h, m, W)
            self.last_segments[int(i)] = (p0, p1, match[0] if match else None,
                                          match[1] if match else float("nan"))
            if match is None:
                self._age_kill(i)
                continue
            uv_cur, zncc = match
            stats.matched += 1

            # Photometric pixel noise: matching precision along the epipolar
            # direction is limited by the image gradient there (LSD-style).
            px_sigma = cfg.pixel_sigma
            if cfg.photometric_sigma:
                unit = (p1 - p0) / (seg_len + 1e-12)
                ip = cv2.getRectSubPix(img, (1, 1), tuple(uv_cur + unit))[0, 0]
                im = cv2.getRectSubPix(img, (1, 1), tuple(uv_cur - unit))[0, 0]
                g = abs(float(ip) - float(im)) / 2.0
                px_sigma = float(np.clip(cfg.sigma_intensity / max(g, 1e-3), 0.7, 4.0))

            tri = self._triangulate_inverse_depth(R, t, self.f_ref[i], uv_cur, p1 - p0,
                                                  pixel_sigma=px_sigma)
            if tri is None:
                self._age_kill(i)
                continue
            d_meas, sigma_meas = tri

            # Innovation gate: reject measurements wildly outside current belief.
            gate = cfg.outlier_gate_sigmas * np.sqrt(self.sigma2[i] + sigma_meas ** 2)
            if abs(d_meas - self.mu[i]) > gate:
                self.n_out[i] += 1
                self.beta_b[i] += 1.0
                stats.outliers += 1
                if self.status[i] == SEED_ACTIVE:
                    if cfg.use_beta_inlier:
                        tries = self.n_in[i] + self.n_out[i]
                        ratio = self.beta_a[i] / (self.beta_a[i] + self.beta_b[i])
                        if tries >= 6 and ratio < cfg.beta_kill_below:
                            self.status[i] = SEED_DEAD
                        else:
                            self._age_kill(i)
                    elif self.n_out[i] - self.n_in[i] > cfg.kill_outlier_margin:
                        self.status[i] = SEED_DEAD
                    else:
                        self._age_kill(i)
                continue

            # Gaussian product update on inverse depth.
            s2, m2 = sigma_meas ** 2, d_meas
            denom = self.sigma2[i] + s2
            self.mu[i] = (self.mu[i] * s2 + m2 * self.sigma2[i]) / denom
            self.sigma2[i] = (self.sigma2[i] * s2) / denom
            self.n_in[i] += 1
            self.beta_a[i] += 1.0
            stats.fused += 1

            # Accumulated parallax: angle subtended by the baseline at the
            # point (perpendicular baseline over depth). The honest measure of
            # how much triangulation information this seed has ever received.
            r_hat = self.f_ref[i] / (np.linalg.norm(self.f_ref[i]) + 1e-12)
            t_perp = t - np.dot(t, r_hat) * r_hat
            z_now = 1.0 / max(self.mu[i], 1e-12)
            par = np.degrees(np.arctan2(np.linalg.norm(t_perp), z_now))
            if par > self.parallax_deg[i]:
                self.parallax_deg[i] = par

            beta_ok = (not cfg.use_beta_inlier
                       or self.beta_a[i] / (self.beta_a[i] + self.beta_b[i]) >= cfg.beta_harvest_min)
            if (self.status[i] == SEED_ACTIVE
                    and self.n_in[i] >= cfg.converge_min_obs
                    and beta_ok
                    and self.parallax_deg[i] >= cfg.min_parallax_deg
                    and np.sqrt(self.sigma2[i]) / max(self.mu[i], 1e-12) < cfg.converge_sigma_ratio):
                self.status[i] = SEED_CONVERGED
                stats.new_converged += 1

        stats.total_converged = int(np.sum(self.status == SEED_CONVERGED))
        act = self.status == SEED_ACTIVE
        if np.any(act):
            stats.median_sigma_ratio = float(np.median(np.sqrt(self.sigma2[act]) / np.maximum(self.mu[act], 1e-12)))
        stats.update_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
        return stats

    # ---------------------------------------------------------------- helpers

    def _age_kill(self, i: int) -> None:
        if self.status[i] == SEED_ACTIVE and self.attempts[i] >= self.cfg.max_attempts:
            self.status[i] = SEED_DEAD

    def _segment_pixels(self, A, Kt, d_lo, d_hi, w, h, margin) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Project the inverse-depth interval to a pixel segment clipped to the image.

        Never shrunk around the belief mean: for an uninformed seed the mean is
        the near-biased prior, so re-centering the search on it would lose the
        true depth. What falls outside the image is unobservable anyway.
        """
        q0 = A + d_lo * Kt
        q1 = A + d_hi * Kt
        if q0[2] <= 1e-9 or q1[2] <= 1e-9:
            return None
        p0 = q0[:2] / q0[2]
        p1 = q1[:2] / q1[2]
        clipped = _clip_segment(p0, p1, w, h, margin)
        if clipped is None:
            return None
        p0, p1 = clipped
        # Enforce a minimum window so the search never collapses onto the prior.
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        min_len = self.cfg.min_search_window_px
        if 1e-9 < length < min_len:
            grow = 0.5 * (min_len - length) / length
            p0 = p0 - seg * grow
            p1 = p1 + seg * grow
        return p0, p1

    def _patch_warp(self, KR, Kt, i: int) -> Optional[np.ndarray]:
        """2x2 affine mapping ref-patch pixel offsets to current-frame offsets.

        Assumes a fronto-parallel patch at the seed's current mean depth (SVO's
        getWarpMatrixAffine). Columns come from projecting the patch center and
        one-pixel steps in u and v through the relative pose.
        """
        z0 = 1.0 / max(self.mu[i], 1e-12)
        u0, v0 = self.uv[i]
        eps = 3.0
        cols = []
        base = None
        for du, dv in ((0.0, 0.0), (eps, 0.0), (0.0, eps)):
            f = self.K_inv @ np.array([u0 + du, v0 + dv, 1.0])
            q = KR @ (f * z0) + Kt
            if q[2] <= 1e-9:
                return None
            px = q[:2] / q[2]
            if base is None:
                base = px
            else:
                cols.append((px - base) / eps)
        W = np.stack(cols, axis=1)
        det = abs(np.linalg.det(W))
        if not (0.2 < det < 5.0):
            return None
        return W

    def _search_segment(self, img, ref_patch, p0, p1, w, h, margin,
                        warp: Optional[np.ndarray] = None) -> Optional[Tuple[np.ndarray, float]]:
        """ZNCC patch search along the pixel segment [p0, p1]; subpixel refined."""
        cfg = self.cfg
        p = cfg.patch_size
        direction = p1 - p0
        seg_len = float(np.linalg.norm(direction))
        direction = direction / (seg_len + 1e-12)
        # Step adapts between search_step_px and ~2 px so long (uninformed)
        # segments stay affordable without the ZNCC peak slipping between samples.
        num = min(int(seg_len / cfg.search_step_px) + 2, cfg.max_search_samples)
        num = max(num, min(int(seg_len / 2.0) + 2, cfg.max_search_samples))
        s = np.linspace(0.0, seg_len, num)
        centers = p0[None, :] + s[:, None] * direction[None, :]

        inside = ((centers[:, 0] >= margin) & (centers[:, 0] < w - margin)
                  & (centers[:, 1] >= margin) & (centers[:, 1] < h - margin))
        if not np.any(inside):
            return None
        centers = centers[inside]
        s = s[inside]
        L = centers.shape[0]

        # Bilinear-sample all candidate patches with one remap call. The offset
        # grid is warped so candidates are compared in ref-patch geometry.
        offs = np.arange(p, dtype=np.float32) - (p - 1) / 2.0
        gx, gy = np.meshgrid(offs, offs)                      # (p, p)
        if warp is not None:
            wx = warp[0, 0] * gx + warp[0, 1] * gy
            wy = warp[1, 0] * gx + warp[1, 1] * gy
            gx, gy = wx.astype(np.float32), wy.astype(np.float32)
        map_x = (centers[:, 0][:, None, None] + gx[None]).astype(np.float32).reshape(L * p, p)
        map_y = (centers[:, 1][:, None, None] + gy[None]).astype(np.float32).reshape(L * p, p)
        cand = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR).reshape(L, p * p).astype(np.float32)

        r = ref_patch.reshape(-1).astype(np.float32)
        r = r - r.mean()
        r_norm = np.linalg.norm(r) + 1e-6
        c = cand - cand.mean(axis=1, keepdims=True)
        c_norm = np.linalg.norm(c, axis=1) + 1e-6
        zncc = (c @ r) / (c_norm * r_norm)

        k = int(np.argmax(zncc))
        best = float(zncc[k])
        if best < cfg.zncc_min:
            return None
        # A peak on the window boundary usually means the true correspondence
        # lies outside the searched segment; fusing it would clamp the belief.
        if k == 0 or k == L - 1:
            return None
        # Distinctiveness: exclude the best peak's own lobe (~patch width) and
        # require every other candidate to trail by a clear margin.
        step = s[1] - s[0] if L > 1 else 1.0
        lobe = max(int(p / max(step, 1e-6)), 2)
        outside = np.ones(L, dtype=bool)
        outside[max(0, k - lobe):k + lobe + 1] = False
        if np.any(outside) and float(np.max(zncc[outside])) > best - cfg.peak_margin:
            return None

        # Parabolic subpixel refinement along the segment.
        s_best = s[k]
        if 0 < k < L - 1:
            y0, y1, y2 = zncc[k - 1], zncc[k], zncc[k + 1]
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-9:
                delta = 0.5 * (y0 - y2) / denom
                delta = float(np.clip(delta, -1.0, 1.0))
                step = (s[k + 1] - s[k - 1]) / 2.0
                s_best = s[k] + delta * step
        uv_cur = p0 + s_best * direction
        return uv_cur, best

    def _triangulate_inverse_depth(self, R, t, f_ref, uv_cur, seg_dir,
                                   pixel_sigma: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """Two-ray triangulation -> (d_meas, sigma_meas) in inverse depth."""
        def depth_from_pixel(uv):
            f_cur = self.K_inv @ np.array([uv[0], uv[1], 1.0])
            Amat = np.stack([R @ f_ref, -f_cur], axis=1)      # 3x2
            sol, *_ = np.linalg.lstsq(Amat, -t.reshape(3), rcond=None)
            z_ref, z_cur = float(sol[0]), float(sol[1])
            if z_ref <= 1e-3 or z_cur <= 1e-3:
                return None
            return z_ref

        z = depth_from_pixel(uv_cur)
        if z is None:
            return None
        d_meas = 1.0 / z

        # Geometric measurement noise: perturb the match 1 px along the epipolar
        # direction and see how much the inverse depth moves.
        n = np.linalg.norm(seg_dir) + 1e-12
        unit = np.asarray(seg_dir, dtype=np.float64) / n
        z_p = depth_from_pixel(np.asarray(uv_cur) + unit)
        if z_p is None:
            z_p = depth_from_pixel(np.asarray(uv_cur) - unit)
        if z_p is None:
            return None
        ps = self.cfg.pixel_sigma if pixel_sigma is None else pixel_sigma
        sigma_meas = abs(1.0 / z_p - d_meas) * ps
        sigma_meas = max(sigma_meas, self.cfg.min_rel_meas_sigma * d_meas, 1e-6)
        return d_meas, sigma_meas

    # ------------------------------------------------------------- keyframing

    def process(self, frame: int, img: np.ndarray) -> FrameUpdateStats:
        """update() + automatic keyframe switching. Use this as the main loop."""
        cfg = self.cfg
        stats = self.update(frame, img)
        self.frames_since_ref += 1
        if cfg.validate_map:
            self.validate_map(frame, img)

        in_view_frac = stats.active_searched / max(stats.alive, 1)
        need_kf = (self.frames_since_ref >= cfg.kf_min_gap and (
            in_view_frac < cfg.kf_visible_frac
            or stats.baseline_m > cfg.kf_max_baseline_m
            or stats.alive < cfg.kf_min_active))
        if need_kf:
            n_new, harvested, _dropped = self.set_reference(frame, img)
            stats.kf_switched = True
            stats.harvested = harvested
            stats.reseeded = n_new
        stats.map_size = int(self.map_xyz.shape[0])
        return stats

    def _harvest_converged(self) -> int:
        """Move converged seeds of the current reference into the world map."""
        idx = np.where(self.status == SEED_CONVERGED)[0]
        if idx.size == 0 or self.ref_frame < 0:
            return 0
        mu = self.mu[idx]
        z = 1.0 / np.maximum(mu, 1e-12)
        X_ref = self.f_ref[idx] * z[:, None]
        T = self.poses[self.ref_frame]
        Xw = (T[:3, :3] @ X_ref.T).T + T[:3, 3]
        self.map_xyz = np.vstack([self.map_xyz, Xw])
        self.map_sigma_rel = np.concatenate(
            [self.map_sigma_rel, np.sqrt(self.sigma2[idx]) / mu])
        self.map_kf = np.concatenate(
            [self.map_kf, np.full(idx.size, self.ref_frame, dtype=np.int32)])
        self.map_patch = np.vstack([self.map_patch, self.patches[idx]])
        self.map_miss = np.concatenate([self.map_miss, np.zeros(idx.size, dtype=np.int32)])
        self.map_alive = np.concatenate([self.map_alive, np.ones(idx.size, dtype=bool)])
        return int(idx.size)

    def validate_map(self, frame: int, img: np.ndarray) -> int:
        """Photometric re-verification of visible map points; prune repeat failures.

        An occluded or dynamic point no longer looks like its harvest patch at
        its projected position. Left unpruned, such points get scored against
        the occluder's LiDAR and dominate the MEAN error tail. On success the
        stored patch is refreshed so gradual scale change does not cause false
        pruning far from the source keyframe.
        """
        cfg = self.cfg
        uv, _z, idx = self.map_points_in(frame, img.shape)
        pruned = 0
        p = cfg.patch_size
        h, w = img.shape[:2]
        m = p / 2.0 + 1.0

        # Self-occlusion: per image cell, points much deeper than the cell's
        # nearest point are behind it and cannot really be visible here.
        if uv.shape[0]:
            c = cfg.map_occl_cell_px
            cols = (w + c - 1) // c
            cell = (uv[:, 1] // c).astype(int) * cols + (uv[:, 0] // c).astype(int)
            zmin = {}
            for ci, zi in zip(cell, _z):
                if ci not in zmin or zi < zmin[ci]:
                    zmin[ci] = zi
            for k, gi in enumerate(idx):
                if _z[k] > cfg.map_occl_ratio * zmin[cell[k]]:
                    self.map_miss[gi] += 1
                    if self.map_miss[gi] >= cfg.map_max_misses:
                        self.map_alive[gi] = False
                        pruned += 1

        for k, gi in enumerate(idx):
            u, v = uv[k]
            if not (m <= u < w - m and m <= v < h - m):
                continue
            cur = cv2.getRectSubPix(img, (p, p), (float(u), float(v))).astype(np.float32).ravel()
            ref = self.map_patch[gi].ravel()
            a = cur - cur.mean()
            b = ref - ref.mean()
            z = float(a @ b / ((np.linalg.norm(a) + 1e-6) * (np.linalg.norm(b) + 1e-6)))
            if z < cfg.map_zncc_min:
                self.map_miss[gi] += 1
                if self.map_miss[gi] >= cfg.map_max_misses:
                    self.map_alive[gi] = False
                    pruned += 1
            else:
                self.map_miss[gi] = 0
                if z > cfg.map_zncc_refresh:
                    self.map_patch[gi] = cur.reshape(p, p)
        return pruned

    # ---------------------------------------------------------------- outputs

    def map_points_in(self, frame: int, shape=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World-map points projected into `frame`: (uv, z_cur, map_indices)."""
        if self.map_xyz.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=int)
        T_c0 = np.linalg.inv(self.poses[int(frame)])
        Xc = (T_c0[:3, :3] @ self.map_xyz.T).T + T_c0[:3, 3]
        z = Xc[:, 2]
        good = (z > 0.5) & self.map_alive
        idx = np.where(good)[0]
        proj = (self.K @ Xc[good].T).T
        uv = proj[:, :2] / proj[:, 2:3]
        if shape is not None:
            h, w = shape[:2]
            inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            uv, idx = uv[inb], idx[inb]
            z = z[good][inb]
        else:
            z = z[good]
        return uv, z, idx

    def visible_depth_points(self, frame: int, shape) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """All depth output visible in `frame`: harvested map + current converged.

        Returns (uv (M,2), z (M,), sigma_rel (M,), src_kf (M,)).
        """
        uv_m, z_m, idx_m = self.map_points_in(frame, shape)
        sig_m = self.map_sigma_rel[idx_m]
        kf_m = self.map_kf[idx_m]

        cidx = np.where(self.status == SEED_CONVERGED)[0]
        uv_c = np.zeros((0, 2))
        z_c = np.zeros((0,))
        sig_c = np.zeros((0,))
        if cidx.size:
            z_ref = 1.0 / np.maximum(self.mu[cidx], 1e-12)
            X_ref = self.f_ref[cidx] * z_ref[:, None]
            R, t = relative_pose(self.poses, self.ref_frame, int(frame))
            Xc = (R @ X_ref.T).T + t.reshape(1, 3)
            sig_all = np.sqrt(self.sigma2[cidx]) / np.maximum(self.mu[cidx], 1e-12)
            good = Xc[:, 2] > 0.5
            proj = (self.K @ Xc[good].T).T
            uv_c = proj[:, :2] / proj[:, 2:3]
            z_c = Xc[good, 2]
            sig_c = sig_all[good]
            h, w = shape[:2]
            inb = (uv_c[:, 0] >= 0) & (uv_c[:, 0] < w) & (uv_c[:, 1] >= 0) & (uv_c[:, 1] < h)
            uv_c, z_c, sig_c = uv_c[inb], z_c[inb], sig_c[inb]
        kf_c = np.full(uv_c.shape[0], self.ref_frame, dtype=np.int32)

        uv = np.vstack([uv_m, uv_c])
        z = np.concatenate([z_m, z_c])
        sig = np.concatenate([sig_m, sig_c])
        kf = np.concatenate([kf_m, kf_c])
        return uv, z, sig, kf

    def converged_ref_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(uv_ref, z_ref, sigma_z) of converged seeds, in the reference frame."""
        idx = np.where(self.status == SEED_CONVERGED)[0]
        if idx.size == 0:
            return (np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,)))
        mu = self.mu[idx]
        z = 1.0 / np.maximum(mu, 1e-12)
        sigma_z = np.sqrt(self.sigma2[idx]) / np.maximum(mu, 1e-12) ** 2
        return self.uv[idx].copy(), z, sigma_z

    def converged_points_in(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Converged points projected into `frame`: (uv (M,2), z_cur (M,))."""
        uv_ref, z_ref, _ = self.converged_ref_points()
        if uv_ref.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros((0,))
        idx = np.where(self.status == SEED_CONVERGED)[0]
        X_ref = self.f_ref[idx] * z_ref[:, None]
        R, t = relative_pose(self.poses, self.ref_frame, int(frame))
        X_cur = (R @ X_ref.T).T + t.reshape(1, 3)
        z_cur = X_cur[:, 2]
        good = z_cur > 1e-3
        proj = (self.K @ X_cur[good].T).T
        uv = proj[:, :2] / proj[:, 2:3]
        return uv, z_cur[good]

    def seed_positions_in(self, frame: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Active seeds projected into `frame` at their mean depth.

        Returns (indices, uv (M,2), sigma_ratio (M,)) for drawing.
        """
        idx = np.where(self.status == SEED_ACTIVE)[0]
        if idx.size == 0:
            return idx, np.zeros((0, 2)), np.zeros((0,))
        z = 1.0 / np.maximum(self.mu[idx], 1e-12)
        X_ref = self.f_ref[idx] * z[:, None]
        R, t = relative_pose(self.poses, self.ref_frame, int(frame))
        X_cur = (R @ X_ref.T).T + t.reshape(1, 3)
        good = X_cur[:, 2] > 1e-3
        idx = idx[good]
        proj = (self.K @ X_cur[good].T).T
        uv = proj[:, :2] / proj[:, 2:3]
        ratio = np.sqrt(self.sigma2[idx]) / np.maximum(self.mu[idx], 1e-12)
        return idx, uv, ratio
