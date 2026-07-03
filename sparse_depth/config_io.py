"""TOML config loading for command-line scripts.

Config files are intentionally permissive: values may be placed in sections for
readability, but the final key must match an argparse destination or one of the
aliases below. Explicit CLI arguments still override config-file defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback path.
    tomllib = None


PATH_DESTS = {
    "img_dir",
    "calib",
    "poses",
    "output_root",
    "velodyne_dir",
    "calib_velo_to_cam",
    "calib_cam_to_cam",
    "calib_imu_to_velo",
    "oxts_dir",
    "gt_depth_dir",
    "fusion_net_ckpt",
    "log_path",
}


ALIASES = {
    "nfeatures": "sift_nfeatures",
    "sift_nfeatures": "sift_nfeatures",
    "contrast": "contrast",
    "soft_nfeatures": "soft_nfeatures",
    "soft_contrast": "soft_contrast",
    "spawn_target_per_bucket": "target_per_bucket",
    "depth_output_per_bucket": "output_per_bucket",
    "depth_output_max_tracks": "output_max_tracks",
    "display_max_tracks": "draw_max_tracks",
    "hybrid_pair_max_history": "hybrid_pair_max_pair_history",
    "calib_path": "calib",
    "poses_path": "poses",
    # Back-compat: legacy parameter names -> new orthogonal / de-"lk"-ified names.
    # (detector_mode is NOT aliased here; it is resolved into detector+descriptor.)
    "lk_tracking": "lk_on",
    "association_matcher": "matcher",
    "lk_sift_period": "detection_period",
    "lk_spawn_period": "spawn_period",
    "lk_soft_period": "soft_sift_period",
    "lk_force_sift_active_below": "force_detection_active_below",
    "lk_force_sift_bucket_coverage_below": "force_detection_coverage_below",
    "lk_force_soft_underfilled_buckets": "force_soft_underfilled_buckets",
}


def _load_one_toml(path: Path) -> Dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("TOML config files require Python 3.11+ tomllib support.")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _flatten_config(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def visit(obj: Dict[str, Any]):
        for key, value in obj.items():
            if isinstance(value, dict):
                visit(value)
            else:
                dest = ALIASES.get(key, key).replace("-", "_")
                out[dest] = value

    visit(data)
    return out


def load_argparse_defaults(config_paths: Iterable[Path]) -> Dict[str, Any]:
    """Load one or more TOML files as argparse defaults.

    Later files override earlier files. Values for path-like argparse
    destinations are converted to ``Path`` because argparse type conversion is
    not applied to defaults set programmatically.
    """
    merged: Dict[str, Any] = {}
    for path in config_paths:
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        merged.update(_flatten_config(_load_one_toml(path)))

    for key in list(merged.keys()):
        if key in PATH_DESTS and merged[key] is not None:
            merged[key] = Path(merged[key])
    return merged
