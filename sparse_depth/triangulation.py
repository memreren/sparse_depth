"""Best-pair triangulation selection for sparse temporal depth tracks.

The feature managers keep track state and visualization state; this module owns
the geometric decision for "which previous observation should triangulate the
current point?"  It intentionally accepts track/config objects by attribute so
the SIFT-only, SIFT+LK, and evaluator scripts can share the same implementation
without inheriting from a common manager class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparse_depth.geometry import (
    fundamental_from_R_t,
    parallax_angle_deg,
    project_points,
    relative_pose,
    sampson_error_px,
    triangulate_dlt,
)


# --- Geometry memoization ---------------------------------------------------
# relative_pose(i, j), its underlying pose inverse inv(T_0j), and the projection
# matrix K[R|t] for a frame pair depend ONLY on the frame indices, which are
# constant for a whole run. Recomputing them per-track / per-point / per-LM-
# iteration is the dominant triangulation cost, especially for refined_multiview
# (its solver hammers relative_pose every iteration). These caches make each a
# one-time computation. Keyed by id(poses) so a new sequence (new poses object,
# always the case across ablation subprocesses) gets a fresh namespace; entries
# are bounded by frame-pair count and use only past/current frames, so this is
# fully causal and safe for the live viewer too.
_POSE_INV_CACHE: Dict[Tuple[int, int], np.ndarray] = {}
_REL_CACHE: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
_PROJ_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def _pose_inv(poses: Sequence[np.ndarray], j: int) -> np.ndarray:
    key = (id(poses), int(j))
    v = _POSE_INV_CACHE.get(key)
    if v is None:
        v = np.linalg.inv(poses[j])
        _POSE_INV_CACHE[key] = v
    return v


def _relative_pose(poses: Sequence[np.ndarray], i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cached relative_pose; matches geometry.relative_pose exactly, no inverse per call."""
    key = (id(poses), int(i), int(j))
    v = _REL_CACHE.get(key)
    if v is None:
        T_j_i = _pose_inv(poses, j) @ poses[i]
        v = (T_j_i[:3, :3], T_j_i[:3, 3])
        _REL_CACHE[key] = v
    return v


@dataclass
class TriangInfo:
    """One track's selected pair and geometric accept/reject diagnostics."""

    label: str
    past_frame: Optional[int]
    current_frame: int
    gap: int
    epi_px: float
    parallax_deg: float
    baseline_m: float
    depth_m: float = np.nan
    reproj_px: float = np.nan
    cheirality_ok: bool = False
    confirmed: bool = False
    reacquired: bool = False
    hit_count: int = 0
    source: str = "primary"
    method: str = "best_pair_dlt"
    used_views: int = 0
    inlier_views: int = 0
    init_depth_m: float = np.nan
    rmse_before_px: float = np.nan
    rmse_after_px: float = np.nan
    max_residual_px: float = np.nan
    current_reproj_px: float = np.nan
    worst_frame: Optional[int] = None
    depth_shift_ratio: float = np.nan
    landmark_candidate: bool = False
    optimize_success: bool = False


TRIANG_LABEL_RANK = {
    "confirmed_good": 6,
    "candidate_good": 5,
    "reacq_good": 5,
    "bad_reproj": 4,
    "bad_depth": 3,
    "low_parallax": 2,
    "bad_epi": 1,
    "no_pair": 0,
}

GOOD_TRIANG_LABELS = {"confirmed_good", "candidate_good", "reacq_good"}


class _ConfigOverride:
    """Read-through config wrapper used for local method-specific gate changes."""

    def __init__(self, base: Any, **overrides: Any):
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def _no_pair_info(track: Any, frame: int, *, reacquired: Optional[bool] = None) -> TriangInfo:
    """Return the standard no-pair record while preserving track metadata."""
    return TriangInfo(
        "no_pair",
        None,
        frame,
        0,
        np.nan,
        np.nan,
        np.nan,
        confirmed=bool(track.confirmed),
        reacquired=bool(track.reacq_probation if reacquired is None else reacquired),
        hit_count=int(track.hit_count),
        source=str(track.last_source),
    )


def _prefer_info(candidate: TriangInfo, current_best: TriangInfo) -> bool:
    """Ranking/tie-break policy for pair selection.

    This mirrors the original manager behavior: first prefer the highest label
    rank, then choose lower epipolar error among bad-epipolar pairs and higher
    parallax everywhere else.
    """
    cand_rank = TRIANG_LABEL_RANK.get(candidate.label, 0)
    best_rank = TRIANG_LABEL_RANK.get(current_best.label, 0)
    if cand_rank != best_rank:
        return cand_rank > best_rank
    if candidate.label == "bad_epi":
        return candidate.epi_px < current_best.epi_px
    return candidate.parallax_deg > current_best.parallax_deg


def _good_label(track: Any) -> str:
    label = "candidate_good"
    if track.confirmed:
        label = "confirmed_good"
    if track.reacq_probation or track.reacquired_last_step:
        label = "reacq_good"
    return label


def _pair_geom(
    K: np.ndarray,
    K_inv: np.ndarray,
    poses: Sequence[np.ndarray],
    geom_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float, np.ndarray]],
    past_frame: int,
    current_frame: int,
    past_pt: np.ndarray,
    curr_pt: np.ndarray,
):
    key = (past_frame, current_frame)
    cached = geom_cache.get(key)
    if cached is None:
        R, t = _relative_pose(poses, past_frame, current_frame)
        baseline = float(np.linalg.norm(t))
        F = fundamental_from_R_t(K, R, t)
        geom_cache[key] = (R, t, baseline, F)
    else:
        R, t, baseline, F = cached

    past_pt_2 = past_pt.reshape(1, 2)
    curr_pt_2 = curr_pt.reshape(1, 2)
    epi = float(sampson_error_px(F, past_pt_2, curr_pt_2)[0])
    ray_i = K_inv @ np.array([past_pt_2[0, 0], past_pt_2[0, 1], 1.0], dtype=np.float64)
    ray_j = K_inv @ np.array([curr_pt_2[0, 0], curr_pt_2[0, 1], 1.0], dtype=np.float64)
    ray_j_i = R.T @ ray_j
    ray_i = ray_i / (np.linalg.norm(ray_i) + 1e-12)
    ray_j_i = ray_j_i / (np.linalg.norm(ray_j_i) + 1e-12)
    par = float(np.degrees(np.arccos(np.clip(float(np.dot(ray_i, ray_j_i)), -1.0, 1.0))))
    return R, t, baseline, F, epi, par


def _precompute_pair_gates(
    tracks: Sequence[Any],
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]]:
    """Batch the per-(track, past_view) epipolar+parallax gating by frame pair.

    The multiview backends otherwise call ``_pair_geom`` once per (track, past
    view), i.e. a single-point ``sampson_error_px``/parallax each — thousands of
    tiny numpy calls per frame that all share one fundamental matrix per frame
    pair. Here we group every candidate pair by (past_frame, current_frame),
    compute F/baseline once, and run ONE vectorized Sampson + parallax over all
    tracks sharing that pair. The result is keyed by ``(id(track), past_frame)``
    and returned in exactly the tuple order ``_pair_geom`` yields, so the caller
    loops are unchanged. Deterministic: identical math, just batched.
    """
    groups: Dict[Tuple[int, int], List[Tuple[int, int, np.ndarray, np.ndarray]]] = {}
    for tr in tracks:
        obs = tr.observations
        if len(obs) < 2:
            continue
        curr_obs = obs[-1]
        j = int(curr_obs.frame)
        curr_pt = curr_obs.pt.reshape(2)
        candidates = obs[:-1]
        if cfg.max_pair_history > 0:
            candidates = [o for o in candidates if j - o.frame <= cfg.max_pair_history]
        for past_obs in candidates:
            i = int(past_obs.frame)
            if i == j or i < 0 or j >= len(poses):
                continue
            groups.setdefault((i, j), []).append((id(tr), i, past_obs.pt.reshape(2), curr_pt))

    gates: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]] = {}
    for (i, j), items in groups.items():
        R, t = _relative_pose(poses, i, j)
        baseline = float(np.linalg.norm(t))
        F = fundamental_from_R_t(K, R, t)
        past_pts = np.array([it[2] for it in items], dtype=np.float64)
        curr_pts = np.array([it[3] for it in items], dtype=np.float64)
        epis = sampson_error_px(F, past_pts, curr_pts)
        pars = parallax_angle_deg(K, R, past_pts, curr_pts)
        for k, it in enumerate(items):
            gates[(it[0], it[1])] = (R, t, baseline, F, float(epis[k]), float(pars[k]))
    return gates


def _projection_current_to_frame(K: np.ndarray, poses: Sequence[np.ndarray], current_frame: int, view_frame: int) -> np.ndarray:
    key = (id(poses), int(current_frame), int(view_frame))
    P = _PROJ_CACHE.get(key)
    if P is None:
        if view_frame == current_frame:
            R = np.eye(3, dtype=np.float64)
            t = np.zeros(3, dtype=np.float64)
        else:
            R, t = _relative_pose(poses, current_frame, view_frame)
        P = K @ np.hstack([R, t.reshape(3, 1)])
        _PROJ_CACHE[key] = P
    return P


def _triangulate_multiview_current(
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    current_frame: int,
    view_frames: Sequence[int],
    points: Sequence[np.ndarray],
):
    """Linear DLT for one 3D point expressed in current camera coordinates."""
    A = []
    projections = []
    for fr, pt in zip(view_frames, points):
        P = _projection_current_to_frame(K, poses, current_frame, int(fr))
        projections.append(P)
        x, y = np.asarray(pt, dtype=np.float64).reshape(2)
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.asarray(A, dtype=np.float64)
    if A.shape[0] < 4:
        return None
    try:
        _u, _s, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    X_h = vt[-1, :]
    if abs(X_h[3]) < 1e-12:
        return None
    X_cur = X_h[:3] / X_h[3]

    reproj = []
    depths = []
    for P, pt in zip(projections, points):
        xh = P @ np.array([X_cur[0], X_cur[1], X_cur[2], 1.0], dtype=np.float64)
        depths.append(float(xh[2]))
        uv = xh[:2] / (xh[2] + 1e-12)
        reproj.append(float(np.linalg.norm(uv - np.asarray(pt, dtype=np.float64).reshape(2))))
    return X_cur, np.asarray(depths, dtype=np.float64), np.asarray(reproj, dtype=np.float64)


def _project_current_point_to_views(
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    current_frame: int,
    view_frames: Sequence[int],
    X_cur: np.ndarray,
):
    X_h = np.array([X_cur[0], X_cur[1], X_cur[2], 1.0], dtype=np.float64)
    uv = []
    depths = []
    for fr in view_frames:
        P = _projection_current_to_frame(K, poses, current_frame, int(fr))
        xh = P @ X_h
        depths.append(float(xh[2]))
        uv.append(xh[:2] / (xh[2] + 1e-12))
    return np.asarray(uv, dtype=np.float64), np.asarray(depths, dtype=np.float64)


def _multiview_residuals(
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    current_frame: int,
    view_frames: Sequence[int],
    points: Sequence[np.ndarray],
    X_cur: np.ndarray,
):
    pred, depths = _project_current_point_to_views(K, poses, current_frame, view_frames, X_cur)
    obs = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return (pred - obs).reshape(-1), depths


def _residual_stats(residuals: np.ndarray, view_frames: Sequence[int]):
    if residuals.size == 0:
        return np.nan, np.nan, None
    per_view = np.linalg.norm(residuals.reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(per_view ** 2)))
    worst_i = int(np.argmax(per_view))
    return rmse, float(per_view[worst_i]), int(view_frames[worst_i])


def _current_reproj_value(reproj_each: np.ndarray) -> float:
    vals = np.asarray(reproj_each, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return np.nan
    return float(vals[-1])


def _max_reproj_value(reproj_each: np.ndarray) -> float:
    vals = np.asarray(reproj_each, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return np.nan
    return float(np.max(vals))


def _current_residual_norm(residuals: np.ndarray) -> float:
    vals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if vals.size < 2:
        return np.nan
    return float(np.linalg.norm(vals[-2:]))


def _max_residual_norm(residuals: np.ndarray) -> float:
    vals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if vals.size < 2:
        return np.nan
    return float(np.max(np.linalg.norm(vals.reshape(-1, 2), axis=1)))


def _robustify_residuals(residuals: np.ndarray, huber_px: float) -> np.ndarray:
    if huber_px <= 0:
        return residuals
    out = residuals.copy()
    for i in range(0, len(out), 2):
        r = out[i:i + 2]
        norm = float(np.linalg.norm(r))
        if norm > huber_px:
            out[i:i + 2] = r * np.sqrt(huber_px / (norm + 1e-12))
    return out


def _numerical_jacobian(fun, X: np.ndarray) -> np.ndarray:
    base = fun(X)
    J = np.zeros((base.size, 3), dtype=np.float64)
    for k in range(3):
        step = 1e-4 * max(1.0, abs(float(X[k])))
        Xp = X.copy()
        Xm = X.copy()
        Xp[k] += step
        Xm[k] -= step
        J[:, k] = (fun(Xp) - fun(Xm)) / (2.0 * step)
    return J


def _robust_residuals_and_jacobian_current(
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    current_frame: int,
    view_frames: Sequence[int],
    points: Sequence[np.ndarray],
    X_cur: np.ndarray,
    huber_px: float,
):
    """Analytic residual/Jacobian for fixed-pose point reprojection refinement."""
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    residuals = []
    jac_rows = []
    depths = []

    for fr, pt in zip(view_frames, points):
        if int(fr) == current_frame:
            R = np.eye(3, dtype=np.float64)
            t = np.zeros(3, dtype=np.float64)
        else:
            R, t = _relative_pose(poses, current_frame, int(fr))
        X_view = R @ X_cur + t.reshape(3)
        x, y, z = float(X_view[0]), float(X_view[1]), float(X_view[2])
        depths.append(z)
        if abs(z) < 1e-12:
            z = 1e-12 if z >= 0.0 else -1e-12

        u = fx * x / z + cx
        v = fy * y / z + cy
        obs = np.asarray(pt, dtype=np.float64).reshape(2)
        r = np.array([u - obs[0], v - obs[1]], dtype=np.float64)

        J_view = np.array(
            [
                [fx / z, 0.0, -fx * x / (z * z)],
                [0.0, fy / z, -fy * y / (z * z)],
            ],
            dtype=np.float64,
        ) @ R

        if huber_px > 0.0:
            norm = float(np.linalg.norm(r))
            if norm > huber_px:
                scale = float(np.sqrt(huber_px / (norm + 1e-12)))
                r *= scale
                J_view *= scale

        residuals.append(r)
        jac_rows.append(J_view)

    return (
        np.asarray(residuals, dtype=np.float64).reshape(-1),
        np.vstack(jac_rows) if jac_rows else np.zeros((0, 3), dtype=np.float64),
        np.asarray(depths, dtype=np.float64),
    )


def _refine_multiview_point_current(
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    current_frame: int,
    view_frames: Sequence[int],
    points: Sequence[np.ndarray],
    X_init: np.ndarray,
    *,
    max_iters: int,
    huber_px: float,
):
    """Small fixed-pose point-only LM refinement for one current-frame 3D point."""
    X = np.asarray(X_init, dtype=np.float64).reshape(3).copy()
    r0, depths0 = _multiview_residuals(K, poses, current_frame, view_frames, points, X)
    rmse0, _max0, _worst0 = _residual_stats(r0, view_frames)
    damping = 1e-3
    success = False

    r_robust, _J, _depths = _robust_residuals_and_jacobian_current(
        K, poses, current_frame, view_frames, points, X, huber_px
    )
    best_cost = float(np.dot(r_robust, r_robust))
    for _iter in range(max(0, int(max_iters))):
        r, J, _depths = _robust_residuals_and_jacobian_current(
            K, poses, current_frame, view_frames, points, X, huber_px
        )
        A = J.T @ J + damping * np.eye(3, dtype=np.float64)
        b = -J.T @ r
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        if np.linalg.norm(delta) < 1e-7 * max(1.0, np.linalg.norm(X)):
            success = True
            break

        X_try = X + delta
        if not np.all(np.isfinite(X_try)):
            damping *= 10.0
            continue
        r_try, _J_try, _depths_try = _robust_residuals_and_jacobian_current(
            K, poses, current_frame, view_frames, points, X_try, huber_px
        )
        cost_try = float(np.dot(r_try, r_try))
        if cost_try < best_cost:
            X = X_try
            best_cost = cost_try
            damping = max(damping * 0.5, 1e-9)
            success = True
        else:
            damping = min(damping * 10.0, 1e9)

    r_final, depths = _multiview_residuals(K, poses, current_frame, view_frames, points, X)
    rmse_final, max_resid, worst_frame = _residual_stats(r_final, view_frames)
    return X, depths, rmse0, rmse_final, max_resid, worst_frame, success


def _flow_depth_pair_current(
    K: np.ndarray,
    K_inv: np.ndarray,
    R_past_to_current: np.ndarray,
    t_past_to_current: np.ndarray,
    past_pt: np.ndarray,
    curr_pt: np.ndarray,
):
    """Closed-form pairwise depth from current pixel and one source-frame match.

    The target frame is the current frame because the sparse depth map is always
    reported on the current image. Given the current pixel ray ``q`` and the
    current-to-past transform ``X_past = R_cp X_current + t_cp``, solve the
    least-squares scalar depth ``d`` in:

        normalize(R_cp * (d * q) + t_cp) ~= normalize(K^-1 * past_pixel)

    The returned 3D point/depth are in current-camera coordinates.
    """
    q = K_inv @ np.array([curr_pt[0], curr_pt[1], 1.0], dtype=np.float64)
    source = K_inv @ np.array([past_pt[0], past_pt[1], 1.0], dtype=np.float64)
    source_xy = source[:2] / (source[2] + 1e-12)

    R_current_to_past = R_past_to_current.T
    t_current_to_past = -R_current_to_past @ np.asarray(t_past_to_current, dtype=np.float64).reshape(3)
    a = R_current_to_past @ q
    b = t_current_to_past

    m = a[:2] - a[2] * source_xy
    n = b[2] * source_xy - b[:2]
    denom = float(np.dot(m, m))
    if denom < 1e-12:
        return None
    depth = float(np.dot(m, n) / denom)
    X_current = depth * q
    X_past = R_current_to_past @ X_current + t_current_to_past

    if abs(X_current[2]) < 1e-12 or abs(X_past[2]) < 1e-12:
        return None

    curr_hat = project_points(K, X_current)[0]
    past_hat = project_points(K, X_past)[0]
    curr_err = float(np.linalg.norm(curr_hat - np.asarray(curr_pt, dtype=np.float64).reshape(2)))
    past_err = float(np.linalg.norm(past_hat - np.asarray(past_pt, dtype=np.float64).reshape(2)))
    reproj_rmse = float(np.sqrt(0.5 * (curr_err ** 2 + past_err ** 2)))
    return X_current, X_past, float(X_current[2]), float(X_past[2]), reproj_rmse


def _ttc_expansion_pair_current(
    K: np.ndarray,
    K_inv: np.ndarray,
    R_past_to_current: np.ndarray,
    t_past_to_current: np.ndarray,
    past_pt: np.ndarray,
    curr_pt: np.ndarray,
    use_rotation: bool = True,
):
    """Discrete time-to-contact (radial expansion) depth for one correspondence.

    This is the TTC counterpart of ``_flow_depth_pair_current``. Instead of a
    least-squares ray intersection, it uses the focus-of-expansion (FOE, i.e. the
    epipole = image of the translation direction) and the *radial* motion of the
    feature away from it. With the known GT rotation the past feature is first
    de-rotated into the current frame, so the residual displacement is purely
    translational and streams radially out of the FOE.

    Geometry (current-camera frame, ``X_current = R X_past + t``):
      * ``g   = R * ray_past``  -> de-rotated past bearing;
      * ``e   = pi(t)``         -> FOE, where translational flow vanishes;
      * ``pinf= pi(g)``         -> where the point would sit if infinitely far;
      * ``pj  = pi(ray_curr)``  -> the measured current feature.
    All three of ``e, pinf, pj`` are colinear on the epipolar line, so the signed
    radial ratio ``rho = (pj - e) / (pinf - e)`` obeys ``rho = 1 - t_z / Z``.
    Hence the scale-free time-to-contact and metric depth are

        tau = 1 / (rho - 1)            # image-only, no metric scale
        Z   = t_z / (1 - rho) = Vz*tau # metric, Vz = t_z is this pair's
                                       #   forward baseline component

    ``rho`` is measured by projecting ``pj - e`` onto the (ideally identical)
    direction ``pinf - e``, which is the least-squares radial coordinate when
    pixel noise pushes ``pj`` slightly off the epipolar line. Returns the same
    5-tuple as the other pair solvers (or ``None`` on a degenerate configuration:
    near-lateral motion, a bearing pointing straight at the FOE, or no
    measurable expansion -- all cases where depth is unobservable).

    NOTE: because ``t_z`` (metric forward baseline) is used to lift ``tau`` to
    metric depth, with GT poses this is numerically a triangulation estimator
    expressed in the radial/FOE parameterization. Its distinct value shows up
    under the forward-motion conditioning analysis and the input ablations where
    scale/pose are withheld; ``tau`` itself is the scale-free quantity to expose
    there.
    """
    t = np.asarray(t_past_to_current, dtype=np.float64).reshape(3)
    t_z = float(t[2])
    if abs(t_z) < 1e-9:
        return None  # near-lateral motion: FOE at infinity, TTC undefined

    # Rotation-free arm (use_rotation=False): keep the epipole and t_z from t, but
    # DROP de-rotation (R := I). The estimator then consumes epipole + forward
    # scale only -- immune to rotation-estimation error, at the cost of a bias
    # that grows with real inter-frame rotation. Used to test whether TTC degrades
    # differently from triangulation under estimated pose.
    R_eff = R_past_to_current if use_rotation else np.eye(3, dtype=np.float64)

    q = K_inv @ np.array([curr_pt[0], curr_pt[1], 1.0], dtype=np.float64)
    ray_past = K_inv @ np.array([past_pt[0], past_pt[1], 1.0], dtype=np.float64)
    g = R_eff @ ray_past
    if abs(g[2]) < 1e-12 or abs(q[2]) < 1e-12:
        return None

    e = t[:2] / t_z                 # FOE (normalized image coords)
    p_inf = g[:2] / g[2]            # de-rotated past feature (point at infinity)
    p_j = q[:2] / q[2]              # current feature

    radial = p_inf - e             # epipolar-line direction from the FOE
    den = float(np.dot(radial, radial))
    if den < 1e-12:
        return None  # bearing points straight at the FOE: no parallax, unobservable

    rho = float(np.dot(p_j - e, radial) / den)
    one_minus_rho = 1.0 - rho
    if abs(one_minus_rho) < 1e-9:
        return None  # no measurable expansion: depth -> infinity

    depth = t_z / one_minus_rho
    X_current = depth * q          # q has z==1, so X_current[2] == depth
    R_current_to_past = R_eff.T
    t_current_to_past = -R_current_to_past @ t
    X_past = R_current_to_past @ X_current + t_current_to_past

    if abs(X_current[2]) < 1e-12 or abs(X_past[2]) < 1e-12:
        return None

    curr_hat = project_points(K, X_current)[0]
    past_hat = project_points(K, X_past)[0]
    curr_err = float(np.linalg.norm(curr_hat - np.asarray(curr_pt, dtype=np.float64).reshape(2)))
    past_err = float(np.linalg.norm(past_hat - np.asarray(past_pt, dtype=np.float64).reshape(2)))
    reproj_rmse = float(np.sqrt(0.5 * (curr_err ** 2 + past_err ** 2)))
    return X_current, X_past, float(X_current[2]), float(X_past[2]), reproj_rmse


def _corrected_pair_dlt_current(
    K: np.ndarray,
    F: np.ndarray,
    R_past_to_current: np.ndarray,
    t_past_to_current: np.ndarray,
    past_pt: np.ndarray,
    curr_pt: np.ndarray,
):
    """Hartley-style corrected two-view correspondence followed by DLT.

    ``cv2.correctMatches`` moves the measured pixels as little as possible while
    enforcing the epipolar constraint exactly. We triangulate those corrected
    pixels, but still report reprojection RMSE against the original measured
    pixels so acceptance does not hide a large correction.
    """
    past = np.asarray(past_pt, dtype=np.float64).reshape(1, 1, 2)
    curr = np.asarray(curr_pt, dtype=np.float64).reshape(1, 1, 2)
    try:
        past_corr, curr_corr = cv2.correctMatches(F.astype(np.float64), past, curr)
    except cv2.error:
        return None

    p_corr = past_corr.reshape(2)
    c_corr = curr_corr.reshape(2)
    tri = triangulate_dlt(K, R_past_to_current, t_past_to_current, p_corr, c_corr)
    if tri is None:
        return None

    X_past, X_current, z_past, z_current, _corr_reproj = tri
    past_hat = project_points(K, X_past)[0]
    curr_hat = project_points(K, X_current)[0]
    orig_reproj = float(
        np.sqrt(
            0.5
            * (
                np.sum((past_hat - np.asarray(past_pt, dtype=np.float64).reshape(2)) ** 2)
                + np.sum((curr_hat - np.asarray(curr_pt, dtype=np.float64).reshape(2)) ** 2)
            )
        )
    )
    return X_past, X_current, float(z_past), float(z_current), orig_reproj


def _evaluate_best_pair_triangulation_impl(
    tracks: Iterable[Any],
    frame: int,
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
    *,
    compute_dlt: bool = True,
    fast: bool = False,
) -> Dict[int, TriangInfo]:
    """Select the best past/current triangulation pair for every active track.

    Inputs are deliberately manager-neutral. Each track is expected to expose the
    same attributes already used by the prototype managers: ``id``,
    ``observations``, ``confirmed``, ``hit_count``, ``reacq_probation``,
    ``reacquired_last_step``, and ``last_source``. Each observation must expose
    ``frame`` and ``pt``.

    Depth is reported in the current camera coordinates because the generated
    sparse depth map is visualized/evaluated on the current frame.
    """
    infos: Dict[int, TriangInfo] = {}
    geom_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    K_inv = np.linalg.inv(K)
    tracks = list(tracks)
    pair_gates = _precompute_pair_gates(tracks, K, poses, cfg)

    for tr in tracks:
        if cfg.triang_confirmed_only and not tr.confirmed:
            continue
        if tr.hit_count < cfg.triang_min_hits or len(tr.observations) < 2:
            infos[tr.id] = _no_pair_info(tr, frame)
            continue
        if (not cfg.triang_include_reacquired) and tr.reacq_probation:
            infos[tr.id] = _no_pair_info(tr, frame, reacquired=True)
            continue

        curr_obs = tr.observations[-1]
        curr_pt = curr_obs.pt.reshape(1, 2)
        best_info: Optional[TriangInfo] = None
        obs_candidates = tr.observations[:-1]
        if cfg.max_pair_history > 0:
            obs_candidates = [o for o in obs_candidates if curr_obs.frame - o.frame <= cfg.max_pair_history]

        for past_obs in obs_candidates:
            i, j = past_obs.frame, curr_obs.frame
            if i == j or i < 0 or j >= len(poses):
                continue

            gate = pair_gates.get((id(tr), int(i)))
            if gate is None:
                gate = _pair_geom(K, K_inv, poses, geom_cache, i, j, past_obs.pt, curr_obs.pt)
            R, t, baseline, _F, epi, par = gate
            if baseline < cfg.min_baseline_m:
                continue

            past_pt = past_obs.pt.reshape(1, 2)
            depth = np.nan
            reproj = np.nan
            cheir = False

            if epi > cfg.gt_epi_thresh_px:
                label = "bad_epi"
            elif par < cfg.min_parallax_deg:
                label = "low_parallax"
            else:
                label = _good_label(tr)
                if compute_dlt and not fast:
                    tri = triangulate_dlt(K, R, t, past_pt.reshape(2), curr_pt.reshape(2))
                    if tri is None:
                        label = "bad_depth"
                    else:
                        _X_i, _X_j, z_i, z_j, reproj = tri
                        depth = z_j
                        cheir = (z_i > 0.0 and z_j > 0.0)
                        if (not cheir) or depth < cfg.min_depth_m or depth > cfg.max_depth_m:
                            label = "bad_depth"
                        elif reproj > cfg.reproj_thresh_px:
                            label = "bad_reproj"

            info = TriangInfo(
                label,
                i,
                j,
                j - i,
                epi,
                par,
                baseline,
                depth,
                reproj,
                cheir,
                tr.confirmed,
                tr.reacq_probation or tr.reacquired_last_step,
                tr.hit_count,
                tr.last_source,
                "best_pair_dlt",
                2 if np.isfinite(depth) else 0,
                2 if label in ("confirmed_good", "candidate_good", "reacq_good") and np.isfinite(depth) else 0,
            )
            if best_info is None or _prefer_info(info, best_info):
                best_info = info

        if best_info is None:
            best_info = _no_pair_info(tr, frame)
        infos[tr.id] = best_info

    return infos


def _evaluate_refined_pair_dlt(
    tracks: Iterable[Any],
    frame: int,
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
    *,
    compute_dlt: bool = True,
    fast: bool = False,
) -> Dict[int, TriangInfo]:
    """Best-pair DLT initialization plus fixed-pose two-view point refinement.

    This is the pairwise counterpart of ``refined_multiview_dlt``. Pair
    selection is intentionally identical to ``best_pair_dlt``: every usable
    past/current observation pair is gated by baseline, GT epipolar error, and
    parallax, then ranked by the same label/parallax policy. The only added
    work is after a pair survives those gates:

    1. Triangulate a current-frame 3D point with linear two-view DLT.
    2. Keep the two camera poses fixed.
    3. Refine only the 3D point by minimizing pixel reprojection residuals in
       the selected past frame and the current frame.

    The reported depth remains the refined Z coordinate in the current camera
    frame. The measured track pixels are not moved or propagated.
    """
    max_iters = int(getattr(cfg, "refine_max_iters", 15))
    huber_px = float(getattr(cfg, "refine_huber_px", 2.0))
    rmse_thresh = float(getattr(cfg, "refine_rmse_thresh_px", cfg.reproj_thresh_px))
    current_thresh = float(getattr(cfg, "current_reproj_thresh_px", cfg.reproj_thresh_px))
    max_reproj_thresh = float(getattr(cfg, "max_reproj_thresh_px", max(rmse_thresh, current_thresh)))
    max_shift = float(getattr(cfg, "refine_max_depth_shift_ratio", 0.5))

    infos: Dict[int, TriangInfo] = {}
    geom_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    K_inv = np.linalg.inv(K)
    tracks = list(tracks)
    pair_gates = _precompute_pair_gates(tracks, K, poses, cfg)

    for tr in tracks:
        if cfg.triang_confirmed_only and not tr.confirmed:
            continue
        if tr.hit_count < cfg.triang_min_hits or len(tr.observations) < 2:
            infos[tr.id] = _no_pair_info(tr, frame)
            infos[tr.id].method = "refined_pair_dlt"
            continue
        if (not cfg.triang_include_reacquired) and tr.reacq_probation:
            infos[tr.id] = _no_pair_info(tr, frame, reacquired=True)
            infos[tr.id].method = "refined_pair_dlt"
            continue

        curr_obs = tr.observations[-1]
        curr_frame = int(curr_obs.frame)
        curr_pt = curr_obs.pt.reshape(2)
        good_candidates = []
        best_reject: Optional[TriangInfo] = None
        obs_candidates = tr.observations[:-1]
        if cfg.max_pair_history > 0:
            obs_candidates = [o for o in obs_candidates if curr_frame - int(o.frame) <= cfg.max_pair_history]

        for past_obs in obs_candidates:
            i, j = int(past_obs.frame), curr_frame
            if i == j or i < 0 or j >= len(poses):
                continue

            gate = pair_gates.get((id(tr), int(i)))
            if gate is None:
                gate = _pair_geom(K, K_inv, poses, geom_cache, i, j, past_obs.pt, curr_pt)
            R, t, baseline, _F, epi, par = gate
            if baseline < cfg.min_baseline_m:
                continue

            past_pt = past_obs.pt.reshape(2)

            if epi > cfg.gt_epi_thresh_px:
                label = "bad_epi"
            elif par < cfg.min_parallax_deg:
                label = "low_parallax"
            else:
                label = _good_label(tr)

            info = TriangInfo(
                label,
                i,
                j,
                j - i,
                epi,
                par,
                baseline,
                confirmed=tr.confirmed,
                reacquired=tr.reacq_probation or tr.reacquired_last_step,
                hit_count=tr.hit_count,
                source=tr.last_source,
                method="refined_pair_dlt",
                used_views=2 if label in GOOD_TRIANG_LABELS else 0,
                inlier_views=2 if label in GOOD_TRIANG_LABELS else 0,
            )
            if label in GOOD_TRIANG_LABELS:
                good_candidates.append((info, R, t, past_pt))
            elif best_reject is None or _prefer_info(info, best_reject):
                best_reject = info

        if (not compute_dlt) or fast:
            best_info: Optional[TriangInfo] = None
            for info, _R, _t, _past_pt in good_candidates:
                if best_info is None or _prefer_info(info, best_info):
                    best_info = info
            if best_info is None:
                best_info = best_reject
            if best_info is None:
                best_info = _no_pair_info(tr, frame)
                best_info.method = "refined_pair_dlt"
            infos[tr.id] = best_info
            continue

        best_evaluated: Optional[TriangInfo] = best_reject
        while good_candidates:
            best_idx = 0
            for idx in range(1, len(good_candidates)):
                if _prefer_info(good_candidates[idx][0], good_candidates[best_idx][0]):
                    best_idx = idx
            cheap_info, R, t, past_pt = good_candidates.pop(best_idx)

            label = cheap_info.label
            depth = np.nan
            init_depth = np.nan
            reproj = np.nan
            cheir = False
            inlier_views = 0
            rmse_before = np.nan
            rmse_after = np.nan
            max_resid = np.nan
            worst_frame = None
            depth_shift = np.nan
            success = False

            tri = triangulate_dlt(K, R, t, past_pt, curr_pt)
            if tri is None:
                label = "bad_depth"
            else:
                _X_past, X_current_init, z_past, z_current, reproj_dlt = tri
                init_depth = float(z_current)
                frames = [cheap_info.past_frame, cheap_info.current_frame]
                points = [past_pt, curr_pt]
                X_ref, depths_ref, rmse_before, rmse_after, max_resid, worst_frame, success = (
                    _refine_multiview_point_current(
                        K,
                        poses,
                        cheap_info.current_frame,
                        frames,
                        points,
                        X_current_init,
                        max_iters=max_iters,
                        huber_px=huber_px,
                    )
                )
                depth = float(X_ref[2])
                reproj = rmse_after
                cheir = bool(np.all(depths_ref > 0.0))
                depth_shift = abs(depth - init_depth) / max(abs(init_depth), 1e-12)

                if (not cheir) or depth < cfg.min_depth_m or depth > cfg.max_depth_m:
                    label = "bad_depth"
                elif (not np.isfinite(rmse_after)) or rmse_after > rmse_thresh:
                    label = "bad_reproj"
                elif np.isfinite(depth_shift) and depth_shift > max_shift:
                    label = "bad_depth"
                elif reproj_dlt > cfg.reproj_thresh_px and rmse_after > cfg.reproj_thresh_px:
                    label = "bad_reproj"

                if label in GOOD_TRIANG_LABELS:
                    inlier_views = 2

            info = TriangInfo(
                label,
                cheap_info.past_frame,
                cheap_info.current_frame,
                cheap_info.gap,
                cheap_info.epi_px,
                cheap_info.parallax_deg,
                cheap_info.baseline_m,
                depth,
                reproj,
                cheir,
                tr.confirmed,
                tr.reacq_probation or tr.reacquired_last_step,
                tr.hit_count,
                tr.last_source,
                "refined_pair_dlt",
                2 if np.isfinite(depth) else 0,
                inlier_views,
                init_depth_m=init_depth,
                rmse_before_px=rmse_before,
                rmse_after_px=rmse_after,
                max_residual_px=max_resid,
                worst_frame=worst_frame,
                depth_shift_ratio=depth_shift,
                landmark_candidate=False,
                optimize_success=success,
            )
            if info.label in GOOD_TRIANG_LABELS:
                infos[tr.id] = info
                break
            if best_evaluated is None or _prefer_info(info, best_evaluated):
                best_evaluated = info
        else:
            if best_evaluated is None:
                best_evaluated = _no_pair_info(tr, frame)
                best_evaluated.method = "refined_pair_dlt"
            infos[tr.id] = best_evaluated

    return infos


def _evaluate_flow_depth_pair(
    tracks: Iterable[Any],
    frame: int,
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
    *,
    compute_dlt: bool = True,
    fast: bool = False,
) -> Dict[int, TriangInfo]:
    """Select the best pair, then solve depth with the flow-to-depth formula.

    This keeps the same gating and ranking logic as ``best_pair_dlt`` so the
    comparison is clean: epipolar error, parallax, baseline, depth range, and
    reprojection threshold mean the same thing. The only changed component is
    the pairwise depth estimator itself.
    """
    infos: Dict[int, TriangInfo] = {}
    geom_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    K_inv = np.linalg.inv(K)
    tracks = list(tracks)
    pair_gates = _precompute_pair_gates(tracks, K, poses, cfg)

    for tr in tracks:
        if cfg.triang_confirmed_only and not tr.confirmed:
            continue
        if tr.hit_count < cfg.triang_min_hits or len(tr.observations) < 2:
            infos[tr.id] = _no_pair_info(tr, frame)
            infos[tr.id].method = "flow_depth_pair"
            continue
        if (not cfg.triang_include_reacquired) and tr.reacq_probation:
            infos[tr.id] = _no_pair_info(tr, frame, reacquired=True)
            infos[tr.id].method = "flow_depth_pair"
            continue

        curr_obs = tr.observations[-1]
        curr_pt = curr_obs.pt.reshape(1, 2)
        best_info: Optional[TriangInfo] = None
        obs_candidates = tr.observations[:-1]
        if cfg.max_pair_history > 0:
            obs_candidates = [o for o in obs_candidates if curr_obs.frame - o.frame <= cfg.max_pair_history]

        for past_obs in obs_candidates:
            i, j = int(past_obs.frame), int(curr_obs.frame)
            if i == j or i < 0 or j >= len(poses):
                continue

            gate = pair_gates.get((id(tr), int(i)))
            if gate is None:
                gate = _pair_geom(K, K_inv, poses, geom_cache, i, j, past_obs.pt, curr_obs.pt)
            R, t, baseline, _F, epi, par = gate
            if baseline < cfg.min_baseline_m:
                continue

            past_pt = past_obs.pt.reshape(1, 2)
            depth = np.nan
            reproj = np.nan
            cheir = False

            if epi > cfg.gt_epi_thresh_px:
                label = "bad_epi"
            elif par < cfg.min_parallax_deg:
                label = "low_parallax"
            else:
                label = _good_label(tr)
                if compute_dlt and not fast:
                    solved = _flow_depth_pair_current(
                        K,
                        K_inv,
                        R,
                        t,
                        past_pt.reshape(2),
                        curr_pt.reshape(2),
                    )
                    if solved is None:
                        label = "bad_depth"
                    else:
                        _X_cur, _X_past, z_cur, z_past, reproj = solved
                        depth = z_cur
                        cheir = (z_cur > 0.0 and z_past > 0.0)
                        if (not cheir) or depth < cfg.min_depth_m or depth > cfg.max_depth_m:
                            label = "bad_depth"
                        elif reproj > cfg.reproj_thresh_px:
                            label = "bad_reproj"

            info = TriangInfo(
                label,
                i,
                j,
                j - i,
                epi,
                par,
                baseline,
                depth,
                reproj,
                cheir,
                tr.confirmed,
                tr.reacq_probation or tr.reacquired_last_step,
                tr.hit_count,
                tr.last_source,
                "flow_depth_pair",
                2 if np.isfinite(depth) else 0,
                2 if label in GOOD_TRIANG_LABELS and np.isfinite(depth) else 0,
            )
            if best_info is None or _prefer_info(info, best_info):
                best_info = info

        if best_info is None:
            best_info = _no_pair_info(tr, frame)
            best_info.method = "flow_depth_pair"
        infos[tr.id] = best_info

    return infos


def _evaluate_ttc_expansion(
    tracks: Iterable[Any],
    frame: int,
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
    *,
    compute_dlt: bool = True,
    fast: bool = False,
    use_rotation: bool = True,
) -> Dict[int, TriangInfo]:
    """Select the best pair, then solve depth with the discrete TTC formula.

    Pair selection and every gate (baseline, GT epipolar error, parallax, depth
    range, reprojection threshold) are intentionally identical to
    ``best_pair_dlt`` and ``flow_depth_pair`` so the only variable is the
    pairwise depth estimator itself. The estimator is
    ``_ttc_expansion_pair_current`` (radial expansion from the focus of
    expansion), making this the time-to-contact arm of the method comparison.
    With ``use_rotation=False`` it becomes the rotation-free arm
    (``ttc_expansion_norot``): epipole + forward scale, no de-rotation.
    """
    method_name = "ttc_expansion" if use_rotation else "ttc_expansion_norot"
    infos: Dict[int, TriangInfo] = {}
    geom_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    K_inv = np.linalg.inv(K)
    tracks = list(tracks)
    pair_gates = _precompute_pair_gates(tracks, K, poses, cfg)

    for tr in tracks:
        if cfg.triang_confirmed_only and not tr.confirmed:
            continue
        if tr.hit_count < cfg.triang_min_hits or len(tr.observations) < 2:
            infos[tr.id] = _no_pair_info(tr, frame)
            infos[tr.id].method = method_name
            continue
        if (not cfg.triang_include_reacquired) and tr.reacq_probation:
            infos[tr.id] = _no_pair_info(tr, frame, reacquired=True)
            infos[tr.id].method = method_name
            continue

        curr_obs = tr.observations[-1]
        curr_pt = curr_obs.pt.reshape(1, 2)
        best_info: Optional[TriangInfo] = None
        obs_candidates = tr.observations[:-1]
        if cfg.max_pair_history > 0:
            obs_candidates = [o for o in obs_candidates if curr_obs.frame - o.frame <= cfg.max_pair_history]

        for past_obs in obs_candidates:
            i, j = int(past_obs.frame), int(curr_obs.frame)
            if i == j or i < 0 or j >= len(poses):
                continue

            gate = pair_gates.get((id(tr), int(i)))
            if gate is None:
                gate = _pair_geom(K, K_inv, poses, geom_cache, i, j, past_obs.pt, curr_obs.pt)
            R, t, baseline, _F, epi, par = gate
            if baseline < cfg.min_baseline_m:
                continue

            past_pt = past_obs.pt.reshape(1, 2)
            depth = np.nan
            reproj = np.nan
            cheir = False

            if epi > cfg.gt_epi_thresh_px:
                label = "bad_epi"
            elif par < cfg.min_parallax_deg:
                label = "low_parallax"
            else:
                label = _good_label(tr)
                if compute_dlt and not fast:
                    solved = _ttc_expansion_pair_current(
                        K,
                        K_inv,
                        R,
                        t,
                        past_pt.reshape(2),
                        curr_pt.reshape(2),
                        use_rotation=use_rotation,
                    )
                    if solved is None:
                        label = "bad_depth"
                    else:
                        _X_cur, _X_past, z_cur, z_past, reproj = solved
                        depth = z_cur
                        cheir = (z_cur > 0.0 and z_past > 0.0)
                        if (not cheir) or depth < cfg.min_depth_m or depth > cfg.max_depth_m:
                            label = "bad_depth"
                        elif reproj > cfg.reproj_thresh_px:
                            label = "bad_reproj"

            info = TriangInfo(
                label,
                i,
                j,
                j - i,
                epi,
                par,
                baseline,
                depth,
                reproj,
                cheir,
                tr.confirmed,
                tr.reacq_probation or tr.reacquired_last_step,
                tr.hit_count,
                tr.last_source,
                method_name,
                2 if np.isfinite(depth) else 0,
                2 if label in GOOD_TRIANG_LABELS and np.isfinite(depth) else 0,
            )
            if best_info is None or _prefer_info(info, best_info):
                best_info = info

        if best_info is None:
            best_info = _no_pair_info(tr, frame)
            best_info.method = method_name
        infos[tr.id] = best_info

    return infos


def _evaluate_corrected_pair_dlt(
    tracks: Iterable[Any],
    frame: int,
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
    *,
    compute_dlt: bool = True,
    fast: bool = False,
) -> Dict[int, TriangInfo]:
    """Best-pair DLT after epipolar correction of the two image points."""
    infos: Dict[int, TriangInfo] = {}
    geom_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    K_inv = np.linalg.inv(K)
    tracks = list(tracks)
    pair_gates = _precompute_pair_gates(tracks, K, poses, cfg)

    for tr in tracks:
        if cfg.triang_confirmed_only and not tr.confirmed:
            continue
        if tr.hit_count < cfg.triang_min_hits or len(tr.observations) < 2:
            infos[tr.id] = _no_pair_info(tr, frame)
            infos[tr.id].method = "corrected_pair_dlt"
            continue
        if (not cfg.triang_include_reacquired) and tr.reacq_probation:
            infos[tr.id] = _no_pair_info(tr, frame, reacquired=True)
            infos[tr.id].method = "corrected_pair_dlt"
            continue

        curr_obs = tr.observations[-1]
        curr_pt = curr_obs.pt.reshape(1, 2)
        best_info: Optional[TriangInfo] = None
        obs_candidates = tr.observations[:-1]
        if cfg.max_pair_history > 0:
            obs_candidates = [o for o in obs_candidates if curr_obs.frame - o.frame <= cfg.max_pair_history]

        for past_obs in obs_candidates:
            i, j = int(past_obs.frame), int(curr_obs.frame)
            if i == j or i < 0 or j >= len(poses):
                continue

            gate = pair_gates.get((id(tr), int(i)))
            if gate is None:
                gate = _pair_geom(K, K_inv, poses, geom_cache, i, j, past_obs.pt, curr_obs.pt)
            R, t, baseline, F, epi, par = gate
            if baseline < cfg.min_baseline_m:
                continue

            past_pt = past_obs.pt.reshape(1, 2)
            depth = np.nan
            reproj = np.nan
            cheir = False

            if epi > cfg.gt_epi_thresh_px:
                label = "bad_epi"
            elif par < cfg.min_parallax_deg:
                label = "low_parallax"
            else:
                label = _good_label(tr)
                if compute_dlt and not fast:
                    tri = _corrected_pair_dlt_current(
                        K,
                        F,
                        R,
                        t,
                        past_pt.reshape(2),
                        curr_pt.reshape(2),
                    )
                    if tri is None:
                        label = "bad_depth"
                    else:
                        _X_past, _X_current, z_past, z_current, reproj = tri
                        depth = z_current
                        cheir = (z_past > 0.0 and z_current > 0.0)
                        if (not cheir) or depth < cfg.min_depth_m or depth > cfg.max_depth_m:
                            label = "bad_depth"
                        elif reproj > cfg.reproj_thresh_px:
                            label = "bad_reproj"

            info = TriangInfo(
                label,
                i,
                j,
                j - i,
                epi,
                par,
                baseline,
                depth,
                reproj,
                cheir,
                tr.confirmed,
                tr.reacq_probation or tr.reacquired_last_step,
                tr.hit_count,
                tr.last_source,
                "corrected_pair_dlt",
                2 if np.isfinite(depth) else 0,
                2 if label in GOOD_TRIANG_LABELS and np.isfinite(depth) else 0,
            )
            if best_info is None or _prefer_info(info, best_info):
                best_info = info

        if best_info is None:
            best_info = _no_pair_info(tr, frame)
            best_info.method = "corrected_pair_dlt"
        infos[tr.id] = best_info

    return infos


def _evaluate_windowed_multiview_dlt(
    tracks: Iterable[Any],
    frame: int,
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
    *,
    compute_dlt: bool = True,
    fast: bool = False,
) -> Dict[int, TriangInfo]:
    """Triangulate current-frame depth from several recent observations.

    Candidate past observations must individually pass the same GT epipolar,
    baseline, and parallax gates used by best-pair DLT. A single 3D point is
    then fit in current-camera coordinates from the current observation plus all
    surviving past observations. Reprojection outliers are removed once and the
    point is refit from the inlier views.
    """
    min_views = int(getattr(cfg, "multiview_min_views", 3))
    current_thresh = float(getattr(cfg, "current_reproj_thresh_px", cfg.reproj_thresh_px))
    max_reproj_thresh = float(getattr(cfg, "max_reproj_thresh_px", max(cfg.reproj_thresh_px, current_thresh)))
    infos: Dict[int, TriangInfo] = {}
    geom_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    K_inv = np.linalg.inv(K)
    tracks = list(tracks)
    pair_gates = _precompute_pair_gates(tracks, K, poses, cfg)

    for tr in tracks:
        if cfg.triang_confirmed_only and not tr.confirmed:
            continue
        if tr.hit_count < cfg.triang_min_hits or len(tr.observations) < 2:
            infos[tr.id] = _no_pair_info(tr, frame)
            continue
        if (not cfg.triang_include_reacquired) and tr.reacq_probation:
            infos[tr.id] = _no_pair_info(tr, frame, reacquired=True)
            continue

        curr_obs = tr.observations[-1]
        curr_frame = int(curr_obs.frame)
        curr_pt = curr_obs.pt.reshape(2)
        obs_candidates = tr.observations[:-1]
        if cfg.max_pair_history > 0:
            obs_candidates = [o for o in obs_candidates if curr_frame - o.frame <= cfg.max_pair_history]

        valid_obs = []
        best_reject: Optional[TriangInfo] = None
        max_par = np.nan
        max_baseline = np.nan
        best_past_frame: Optional[int] = None
        best_epi = np.nan
        for past_obs in obs_candidates:
            i, j = int(past_obs.frame), curr_frame
            if i == j or i < 0 or j >= len(poses):
                continue
            gate = pair_gates.get((id(tr), i))
            if gate is None:  # safety fallback; the pre-pass should cover every pair
                gate = _pair_geom(K, K_inv, poses, geom_cache, i, j, past_obs.pt, curr_pt)
            _R, _t, baseline, _F, epi, par = gate
            if baseline < cfg.min_baseline_m:
                continue
            if epi > cfg.gt_epi_thresh_px:
                reject = TriangInfo("bad_epi", i, j, j - i, epi, par, baseline, confirmed=tr.confirmed,
                                    reacquired=tr.reacq_probation or tr.reacquired_last_step,
                                    hit_count=tr.hit_count, source=tr.last_source, method="windowed_multiview_dlt")
                if best_reject is None or _prefer_info(reject, best_reject):
                    best_reject = reject
                continue
            if par < cfg.min_parallax_deg:
                reject = TriangInfo("low_parallax", i, j, j - i, epi, par, baseline, confirmed=tr.confirmed,
                                    reacquired=tr.reacq_probation or tr.reacquired_last_step,
                                    hit_count=tr.hit_count, source=tr.last_source, method="windowed_multiview_dlt")
                if best_reject is None or _prefer_info(reject, best_reject):
                    best_reject = reject
                continue

            valid_obs.append(past_obs)
            if not np.isfinite(max_par) or par > max_par:
                max_par = par
                max_baseline = baseline
                best_past_frame = i
                best_epi = epi

        view_count = len(valid_obs) + 1
        if view_count < min_views:
            if best_reject is not None:
                infos[tr.id] = best_reject
            else:
                infos[tr.id] = _no_pair_info(tr, frame)
            continue

        if (not compute_dlt) or fast:
            infos[tr.id] = TriangInfo(
                _good_label(tr),
                best_past_frame,
                curr_frame,
                curr_frame - int(best_past_frame) if best_past_frame is not None else 0,
                best_epi,
                max_par,
                max_baseline,
                confirmed=tr.confirmed,
                reacquired=tr.reacq_probation or tr.reacquired_last_step,
                hit_count=tr.hit_count,
                source=tr.last_source,
                method="windowed_multiview_dlt",
                used_views=view_count,
                inlier_views=view_count,
            )
            continue

        frames: List[int] = [int(o.frame) for o in valid_obs] + [curr_frame]
        points: List[np.ndarray] = [o.pt.reshape(2) for o in valid_obs] + [curr_pt]
        tri = _triangulate_multiview_current(K, poses, curr_frame, frames, points)
        if tri is None:
            label = "bad_depth"
            depth = np.nan
            reproj_rmse = np.nan
            current_reproj = np.nan
            max_reproj = np.nan
            cheir = False
            inlier_frames = frames
            inlier_points = points
        else:
            X_cur, depths, reproj_each = tri
            inlier_mask = reproj_each <= cfg.reproj_thresh_px
            # The sparse depth is reported at the current pixel, so the current
            # observation must never be silently trimmed from a multiview refit.
            inlier_mask[-1] = True
            inlier_frames = [fr for fr, keep in zip(frames, inlier_mask) if keep]
            inlier_points = [pt for pt, keep in zip(points, inlier_mask) if keep]

            if len(inlier_frames) >= min_views and len(inlier_frames) < len(frames):
                refit = _triangulate_multiview_current(K, poses, curr_frame, inlier_frames, inlier_points)
                if refit is not None:
                    X_cur, depths, reproj_each = refit

            depth = float(X_cur[2])
            reproj_rmse = float(np.sqrt(np.mean(reproj_each ** 2))) if reproj_each.size else np.nan
            current_reproj = _current_reproj_value(reproj_each)
            max_reproj = _max_reproj_value(reproj_each)
            cheir = bool(np.all(depths > 0.0))
            if (
                len(inlier_frames) < min_views
                or reproj_rmse > cfg.reproj_thresh_px
                or current_reproj > current_thresh
                or max_reproj > max_reproj_thresh
            ):
                label = "bad_reproj"
            elif (not cheir) or depth < cfg.min_depth_m or depth > cfg.max_depth_m:
                label = "bad_depth"
            else:
                label = _good_label(tr)

        infos[tr.id] = TriangInfo(
            label,
            best_past_frame,
            curr_frame,
            curr_frame - int(best_past_frame) if best_past_frame is not None else 0,
            best_epi,
            max_par,
            max_baseline,
            depth,
            reproj_rmse,
            cheir,
            tr.confirmed,
            tr.reacq_probation or tr.reacquired_last_step,
            tr.hit_count,
            tr.last_source,
            "windowed_multiview_dlt",
            view_count,
            len(inlier_frames),
            max_residual_px=max_reproj,
            current_reproj_px=current_reproj,
        )

    return infos


def _evaluate_refined_multiview_dlt(
    tracks: Iterable[Any],
    frame: int,
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
    *,
    compute_dlt: bool = True,
    fast: bool = False,
) -> Dict[int, TriangInfo]:
    """Windowed multiview DLT initialization plus fixed-pose 3D point refinement."""
    min_views = int(getattr(cfg, "refine_min_views", getattr(cfg, "multiview_min_views", 3)))
    max_iters = int(getattr(cfg, "refine_max_iters", 15))
    huber_px = float(getattr(cfg, "refine_huber_px", 2.0))
    rmse_thresh = float(getattr(cfg, "refine_rmse_thresh_px", cfg.reproj_thresh_px))
    current_thresh = float(getattr(cfg, "current_reproj_thresh_px", cfg.reproj_thresh_px))
    max_reproj_thresh = float(getattr(cfg, "max_reproj_thresh_px", max(rmse_thresh, current_thresh)))
    max_shift = float(getattr(cfg, "refine_max_depth_shift_ratio", 0.5))

    infos: Dict[int, TriangInfo] = {}
    geom_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    K_inv = np.linalg.inv(K)
    tracks = list(tracks)
    pair_gates = _precompute_pair_gates(tracks, K, poses, cfg)

    for tr in tracks:
        if cfg.triang_confirmed_only and not tr.confirmed:
            continue
        if tr.hit_count < cfg.triang_min_hits or len(tr.observations) < 2:
            infos[tr.id] = _no_pair_info(tr, frame)
            infos[tr.id].method = "refined_multiview_dlt"
            continue
        if (not cfg.triang_include_reacquired) and tr.reacq_probation:
            infos[tr.id] = _no_pair_info(tr, frame, reacquired=True)
            infos[tr.id].method = "refined_multiview_dlt"
            continue

        curr_obs = tr.observations[-1]
        curr_frame = int(curr_obs.frame)
        curr_pt = curr_obs.pt.reshape(2)
        obs_candidates = tr.observations[:-1]
        if cfg.max_pair_history > 0:
            obs_candidates = [o for o in obs_candidates if curr_frame - o.frame <= cfg.max_pair_history]

        valid_obs = []
        best_reject: Optional[TriangInfo] = None
        max_par = np.nan
        max_baseline = np.nan
        best_past_frame: Optional[int] = None
        best_epi = np.nan
        for past_obs in obs_candidates:
            i, j = int(past_obs.frame), curr_frame
            if i == j or i < 0 or j >= len(poses):
                continue
            gate = pair_gates.get((id(tr), i))
            if gate is None:  # safety fallback; the pre-pass should cover every pair
                gate = _pair_geom(K, K_inv, poses, geom_cache, i, j, past_obs.pt, curr_pt)
            _R, _t, baseline, _F, epi, par = gate
            if baseline < cfg.min_baseline_m:
                continue
            if epi > cfg.gt_epi_thresh_px:
                reject = TriangInfo("bad_epi", i, j, j - i, epi, par, baseline, confirmed=tr.confirmed,
                                    reacquired=tr.reacq_probation or tr.reacquired_last_step,
                                    hit_count=tr.hit_count, source=tr.last_source, method="refined_multiview_dlt")
                if best_reject is None or _prefer_info(reject, best_reject):
                    best_reject = reject
                continue
            if par < cfg.min_parallax_deg:
                reject = TriangInfo("low_parallax", i, j, j - i, epi, par, baseline, confirmed=tr.confirmed,
                                    reacquired=tr.reacq_probation or tr.reacquired_last_step,
                                    hit_count=tr.hit_count, source=tr.last_source, method="refined_multiview_dlt")
                if best_reject is None or _prefer_info(reject, best_reject):
                    best_reject = reject
                continue

            valid_obs.append(past_obs)
            if not np.isfinite(max_par) or par > max_par:
                max_par = par
                max_baseline = baseline
                best_past_frame = i
                best_epi = epi

        view_count = len(valid_obs) + 1
        if view_count < min_views:
            if best_reject is not None:
                infos[tr.id] = best_reject
            else:
                infos[tr.id] = _no_pair_info(tr, frame)
            infos[tr.id].method = "refined_multiview_dlt"
            continue

        if (not compute_dlt) or fast:
            infos[tr.id] = TriangInfo(
                _good_label(tr),
                best_past_frame,
                curr_frame,
                curr_frame - int(best_past_frame) if best_past_frame is not None else 0,
                best_epi,
                max_par,
                max_baseline,
                confirmed=tr.confirmed,
                reacquired=tr.reacq_probation or tr.reacquired_last_step,
                hit_count=tr.hit_count,
                source=tr.last_source,
                method="refined_multiview_dlt",
                used_views=view_count,
                inlier_views=view_count,
            )
            continue

        frames: List[int] = [int(o.frame) for o in valid_obs] + [curr_frame]
        points: List[np.ndarray] = [o.pt.reshape(2) for o in valid_obs] + [curr_pt]
        tri = _triangulate_multiview_current(K, poses, curr_frame, frames, points)
        if tri is None:
            infos[tr.id] = TriangInfo(
                "bad_depth", best_past_frame, curr_frame,
                curr_frame - int(best_past_frame) if best_past_frame is not None else 0,
                best_epi, max_par, max_baseline, confirmed=tr.confirmed,
                reacquired=tr.reacq_probation or tr.reacquired_last_step,
                hit_count=tr.hit_count, source=tr.last_source, method="refined_multiview_dlt",
                used_views=view_count, inlier_views=0,
            )
            continue

        X_init, depths_init, reproj_each = tri
        inlier_mask = reproj_each <= cfg.reproj_thresh_px
        # The current observation defines where the sparse depth is reported.
        # Keep it in every refit and reject later if it is not explained well.
        inlier_mask[-1] = True
        inlier_frames = [fr for fr, keep in zip(frames, inlier_mask) if keep]
        inlier_points = [pt for pt, keep in zip(points, inlier_mask) if keep]

        if len(inlier_frames) >= min_views and len(inlier_frames) < len(frames):
            refit = _triangulate_multiview_current(K, poses, curr_frame, inlier_frames, inlier_points)
            if refit is not None:
                X_init, depths_init, reproj_each = refit

        if len(inlier_frames) < min_views:
            label = "bad_reproj"
            depth = float(X_init[2]) if X_init is not None else np.nan
            reproj_rmse = float(np.sqrt(np.mean(reproj_each ** 2))) if reproj_each.size else np.nan
            cheir = bool(np.all(depths_init > 0.0)) if depths_init.size else False
            inlier_views = len(inlier_frames)
            rmse_before = reproj_rmse
            rmse_after = np.nan
            current_reproj = _current_reproj_value(reproj_each)
            max_resid = _max_reproj_value(reproj_each)
            worst_frame = None
            depth_shift = np.nan
            success = False
            landmark_candidate = False
        else:
            init_depth = float(X_init[2])
            X_ref, depths_ref, rmse_before, rmse_after, max_resid, worst_frame, success = _refine_multiview_point_current(
                K,
                poses,
                curr_frame,
                inlier_frames,
                inlier_points,
                X_init,
                max_iters=max_iters,
                huber_px=huber_px,
            )
            depth = float(X_ref[2])
            depth_shift = abs(depth - init_depth) / max(abs(init_depth), 1e-12)
            r_final, _depths_final = _multiview_residuals(K, poses, curr_frame, inlier_frames, inlier_points, X_ref)
            current_reproj = _current_residual_norm(r_final)
            max_resid = _max_residual_norm(r_final)
            cheir = bool(np.all(depths_ref > 0.0))
            reproj_rmse = rmse_after
            if (not cheir) or depth < cfg.min_depth_m or depth > cfg.max_depth_m:
                label = "bad_depth"
            elif (not np.isfinite(rmse_after)) or rmse_after > rmse_thresh:
                label = "bad_reproj"
            elif current_reproj > current_thresh or max_resid > max_reproj_thresh:
                label = "bad_reproj"
            elif np.isfinite(depth_shift) and depth_shift > max_shift:
                label = "bad_depth"
            else:
                label = _good_label(tr)
            inlier_views = len(inlier_frames)
            landmark_candidate = label in GOOD_TRIANG_LABELS and inlier_views >= min_views and success

        init_depth_value = float(X_init[2]) if X_init is not None and np.isfinite(X_init[2]) else np.nan
        info = TriangInfo(
            label,
            best_past_frame,
            curr_frame,
            curr_frame - int(best_past_frame) if best_past_frame is not None else 0,
            best_epi,
            max_par,
            max_baseline,
            depth,
            reproj_rmse,
            cheir,
            tr.confirmed,
            tr.reacq_probation or tr.reacquired_last_step,
            tr.hit_count,
            tr.last_source,
            "refined_multiview_dlt",
            view_count,
            inlier_views,
            init_depth_m=init_depth_value,
            rmse_before_px=rmse_before,
            rmse_after_px=rmse_after,
            max_residual_px=max_resid,
            current_reproj_px=current_reproj,
            worst_frame=worst_frame,
            depth_shift_ratio=depth_shift,
            landmark_candidate=landmark_candidate,
            optimize_success=success,
        )
        infos[tr.id] = info

    return infos


def _evaluate_hybrid_pair_multiview(
    tracks: Iterable[Any],
    frame: int,
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
    *,
    compute_dlt: bool = True,
    fast: bool = False,
) -> Dict[int, TriangInfo]:
    """Use multiview DLT when possible, otherwise fall back to stricter pair DLT.

    This is meant for the SIFT+LK manager's current workflow: short tracks are
    valuable for close road points, but they should face tighter pairwise gates
    than mature multiview tracks. If the current track has enough accepted views
    for windowed multiview DLT, the multiview result is authoritative. If it does
    not, a best-pair DLT pass is retried with ``hybrid_pair_*`` thresholds.
    """
    tracks = list(tracks)
    min_views = int(getattr(cfg, "multiview_min_views", 3))
    multiview_infos = _evaluate_windowed_multiview_dlt(
        tracks,
        frame,
        K,
        poses,
        cfg,
        compute_dlt=compute_dlt,
        fast=fast,
    )

    pair_cfg = _ConfigOverride(
        cfg,
        min_parallax_deg=float(getattr(cfg, "hybrid_pair_min_parallax_deg", cfg.min_parallax_deg)),
        min_baseline_m=float(getattr(cfg, "hybrid_pair_min_baseline_m", cfg.min_baseline_m)),
        max_pair_history=int(getattr(cfg, "hybrid_pair_max_pair_history", cfg.max_pair_history)),
        reproj_thresh_px=float(getattr(cfg, "hybrid_pair_reproj_thresh_px", cfg.reproj_thresh_px)),
    )
    pair_infos = _evaluate_best_pair_triangulation_impl(
        tracks,
        frame,
        K,
        poses,
        pair_cfg,
        compute_dlt=compute_dlt,
        fast=fast,
    )

    infos: Dict[int, TriangInfo] = {}
    all_ids = set(multiview_infos) | set(pair_infos)
    for track_id in all_ids:
        mv = multiview_infos.get(track_id)
        pair = pair_infos.get(track_id)

        if mv is None:
            chosen = pair
        elif mv.label in GOOD_TRIANG_LABELS:
            chosen = mv
            chosen.method = "hybrid_multiview"
        elif mv.used_views >= min_views:
            chosen = mv
            chosen.method = "hybrid_multiview_reject"
        elif pair is not None and pair.label in GOOD_TRIANG_LABELS:
            chosen = pair
            chosen.method = "hybrid_pair_fallback"
        elif pair is not None and _prefer_info(pair, mv):
            chosen = pair
            chosen.method = "hybrid_pair_reject"
        else:
            chosen = mv
            chosen.method = "hybrid_multiview_insufficient"

        if chosen is not None:
            infos[track_id] = chosen

    return infos


def evaluate_best_pair_triangulation(
    tracks: Iterable[Any],
    frame: int,
    K: np.ndarray,
    poses: Sequence[np.ndarray],
    cfg: Any,
    *,
    compute_dlt: bool = True,
    fast: bool = False,
) -> Dict[int, TriangInfo]:
    method = str(getattr(cfg, "triangulation_method", "best_pair_dlt"))
    if method == "best_pair_dlt":
        return _evaluate_best_pair_triangulation_impl(tracks, frame, K, poses, cfg, compute_dlt=compute_dlt, fast=fast)
    if method == "flow_depth_pair":
        return _evaluate_flow_depth_pair(tracks, frame, K, poses, cfg, compute_dlt=compute_dlt, fast=fast)
    if method == "ttc_expansion":
        return _evaluate_ttc_expansion(tracks, frame, K, poses, cfg, compute_dlt=compute_dlt, fast=fast, use_rotation=True)
    if method == "ttc_expansion_norot":
        return _evaluate_ttc_expansion(tracks, frame, K, poses, cfg, compute_dlt=compute_dlt, fast=fast, use_rotation=False)
    if method == "refined_pair_dlt":
        return _evaluate_refined_pair_dlt(tracks, frame, K, poses, cfg, compute_dlt=compute_dlt, fast=fast)
    if method == "corrected_pair_dlt":
        return _evaluate_corrected_pair_dlt(tracks, frame, K, poses, cfg, compute_dlt=compute_dlt, fast=fast)
    if method == "windowed_multiview_dlt":
        return _evaluate_windowed_multiview_dlt(tracks, frame, K, poses, cfg, compute_dlt=compute_dlt, fast=fast)
    if method == "refined_multiview_dlt":
        return _evaluate_refined_multiview_dlt(tracks, frame, K, poses, cfg, compute_dlt=compute_dlt, fast=fast)
    if method == "hybrid_pair_multiview":
        return _evaluate_hybrid_pair_multiview(tracks, frame, K, poses, cfg, compute_dlt=compute_dlt, fast=fast)
    raise ValueError(f"Unknown triangulation_method: {method}")
