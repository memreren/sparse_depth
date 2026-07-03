"""Compact validation policy for road-plane homography warps.

This module deliberately keeps the policy small: sparse geometry decides
whether a frame/method is credible at all; dense symmetric photometrics reject
only textured contradictions while leaving textureless asphalt prior-supported.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from sparse_depth.ground_plane import local_zncc, warp_source_to_target


@dataclass(frozen=True)
class ValidationPolicy:
    patch_radius_px: int = 4       # 9x9: stable, still reasonably local
    min_patch_std: float = 5.0
    zncc_threshold: float = 0.35   # reject only clear textured disagreement
    sparse_reproj_thresh_px: float = 2.0
    min_sparse_inliers: int = 12


def symmetric_photometric_gate(target, source, H_target_to_source, roi, policy: ValidationPolicy):
    """Return gray/green/red classification using both warp directions.

    ``keep`` retains gray (uninformative) and green (textured agreement), and
    rejects only red patches that are informative in both directions yet fail
    either directional ZNCC check.  Bidirectionality mainly improves boundary
    and visibility handling; inverse homographies alone do not create a true
    optical-flow forward/backward test.
    """
    warped_source, support_t = warp_source_to_target(source, H_target_to_source)
    H_source_to_target = np.linalg.inv(H_target_to_source)
    warped_target, support_s = warp_source_to_target(target, H_source_to_target)
    zncc_t, std_tt, std_st = local_zncc(target, warped_source, policy.patch_radius_px)
    zncc_s, std_ss, std_ts = local_zncc(source, warped_target, policy.patch_radius_px)
    h, w = target.shape[:2]
    # Map source-domain reverse statistics back to target coordinates.
    reverse_valid_s = (support_s & np.isfinite(zncc_s)).astype(np.uint8)
    zncc_s_safe = np.nan_to_num(zncc_s, nan=0.0)
    zncc_back = cv2.warpPerspective(zncc_s_safe, H_source_to_target, (w, h), flags=cv2.INTER_LINEAR)
    std_ss_back = cv2.warpPerspective(std_ss, H_source_to_target, (w, h), flags=cv2.INTER_LINEAR)
    std_ts_back = cv2.warpPerspective(std_ts, H_source_to_target, (w, h), flags=cv2.INTER_LINEAR)
    support_back = cv2.warpPerspective(reverse_valid_s, H_source_to_target, (w, h), flags=cv2.INTER_NEAREST) > 0
    base = np.asarray(roi, bool) & support_t & support_back
    info_f = base & (std_tt >= policy.min_patch_std) & (std_st >= policy.min_patch_std) & np.isfinite(zncc_t)
    info_b = base & (std_ss_back >= policy.min_patch_std) & (std_ts_back >= policy.min_patch_std) & support_back
    informative = info_f & info_b
    green = informative & (zncc_t >= policy.zncc_threshold) & (zncc_back >= policy.zncc_threshold)
    red = informative & ~green
    keep = base & ~red
    return {
        "warped": warped_source, "base_gate": base, "keep_gate": keep,
        "informative": informative, "green": green, "red": red,
        "zncc_forward": zncc_t, "zncc_backward": zncc_back,
    }
