#!/usr/bin/env python3
r"""Factorial ablation runner for the SIFT+LK sparse-depth eval.

Core grid = detector {shi, sift, xfeat} x tracking {LK-on, LK-off} x
triangulation {best_pair, windowed_mv, hybrid} = 18 runs, plus OFAT extras
(target_per_bucket, Lowe ratio) at the baseline cell, plus one fully-learned
xfeat_native line. ~23 runs. A compact headline row per run is collected into
one table (printed + outputs/ablation/ablation_summary.csv).

Design notes
------------
- Only the interacting factors are crossed (detector x tracking x triangulation);
  independent knobs (density, ratio) are swept OFAT to keep the grid report-sized.
- LK-off has no frame-to-frame carrier, so those runs force lk_sift_period=1
  (detect + match every frame) or all tracks would die.
- Every run is forced CPU-only (CUDA_VISIBLE_DEVICES=-1) so timings are
  comparable across variants, including the xfeat ones.
- Each run uses --minimal-output: only summary_metrics.csv, per_frame_metrics.csv,
  and config.json (no plots, no per-point diagnostics).

Usage
-----
  python run_ablation.py                       # seq04, 50 frames, all ~23 runs
  python run_ablation.py --num-frames 100
  python run_ablation.py --only core-xfeat learned   # run chosen groups only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
EVAL_SCRIPT = REPO / "sparse_depth_eval_kitti.py"
DEFAULT_CONFIG = REPO / "configs" / "default.toml"

# CONTROLLED baseline: every factor pinned explicitly and soft-SIFT OFF, so the
# detector comparison is clean (soft is a sift_lk-only layer; leaving it on for
# sift but off for shi/xfeat silently conflates "detector" with "soft layer").
# Baseline = sift / soft-off / LK-on / period 3 / windowed / tpb 8 / radius 55 /
# ratio 0.8 / radius_lowe. Runs override by appending tokens (last flag wins).
BASE_TOKENS = [
    "--detector", "sift", "--descriptor", "sift",
    "--no-soft-spawn",
    "--lk-on", "--detection-period", "3",
    "--triangulation-method", "windowed_multiview_dlt",
    "--target-per-bucket", "8",
    "--search-radius", "55",
    "--ratio", "0.8",
    "--matcher", "radius_lowe",
]

# Detector axis: keypoint locations (descriptor stays sift from BASE_TOKENS).
DETECTORS = [("shi", ["--detector", "shi"]),
             ("sift", ["--detector", "sift"]),
             ("xfeat", ["--detector", "xfeat"])]
TRIANG = [("bestpair", "best_pair_dlt"),
          ("windowed", "windowed_multiview_dlt"),
          ("hybrid", "hybrid_pair_multiview"),
          ("ttc", "ttc_expansion")]
# LK-off has no frame-to-frame carrier, so it MUST detect+match every frame
# (period 1) or tracks die. LK-on keeps the baseline period 3.
LKOFF = ["--no-lk-on", "--detection-period", "1"]
TRACKING = [("lkon", []), ("lkoff", LKOFF)]


def build_runs():
    """Return the full run list as (group, label, extra_cli_tokens)."""
    runs = []
    # A. Core factorial (18): detector x tracking x triangulation, soft OFF.
    for dshort, dtoks in DETECTORS:
        for tshort, ttoks in TRACKING:
            for gshort, gmode in TRIANG:
                toks = dtoks + ["--triangulation-method", gmode] + ttoks
                runs.append((f"core-{dshort}", f"{dshort}|{tshort}|{gshort}", toks))
    # B. Detection period (sift / LK-on / windowed). period 3 is the baseline.
    runs.append(("period", "period=1", ["--detection-period", "1"]))
    runs.append(("period", "period=5", ["--detection-period", "5"]))
    # C. Soft-SIFT layer effect (sift / LK-on / windowed). off is the baseline.
    runs.append(("soft", "soft=on", ["--soft-spawn"]))
    # D. Association params, LK-off (pure matching is where they bite). base r55/0.8.
    for r in ("30", "90"):
        runs.append(("assoc", f"lkoff|radius={r}", LKOFF + ["--search-radius", r]))
    for ra in ("0.7", "0.9"):
        runs.append(("assoc", f"lkoff|ratio={ra}", LKOFF + ["--ratio", ra]))
    # E. Density (sift / LK-on / windowed). tpb 8 is the baseline.
    runs.append(("density", "tpb=4", ["--target-per-bucket", "4"]))
    runs.append(("density", "tpb=12", ["--target-per-bucket", "12"]))
    # F. Learned features + learned matcher (xfeat_native / LK-off / windowed):
    #    radius+Lowe vs XFeat's own global mutual-NN matcher, and a cossim sweep.
    xn = ["--detector", "xfeat", "--descriptor", "xfeat"] + LKOFF
    runs.append(("learned", "xnative|radius", xn))
    runs.append(("learned", "xnative|mnn", xn + ["--matcher", "xfeat_mnn"]))
    runs.append(("learned", "xnative|mnn|cos0.7", xn + ["--matcher", "xfeat_mnn", "--xfeat-mnn-min-cossim", "0.7"]))
    runs.append(("learned", "xnative|mnn|cos0.9", xn + ["--matcher", "xfeat_mnn", "--xfeat-mnn-min-cossim", "0.9"]))
    return runs

# Headline columns from each run's summary_metrics.csv. median_ = robust per-frame
# central value; mean_ = mean over frames. Timings are means (feature-managing vs
# triangulation vs total). Errors reported both median and mean, relative and abs.
HEADLINE = [
    ("n_pts", "median_num_sparse_depth"),
    ("n_pts_mn", "mean_num_sparse_depth"),
    ("img_cov", "median_depth_coverage"),
    ("gt_cov", "median_gt_coverage"),
    ("lidarN", "median_num_lidar_matched"),
    ("lidarNmn", "mean_num_lidar_matched"),
    ("match_r", "median_lidar_match_rate"),
    ("medRel", "median_raw_median_rel_err"),
    ("meanRel", "mean_raw_mean_rel_err"),
    ("medAbs", "median_raw_median_abs_err_m"),
    ("meanAbs", "mean_raw_mean_abs_err_m"),
    ("d20", "median_raw_delta_20"),
    ("feat_ms", "mean_time_step_ms"),
    ("tri_ms", "mean_time_triang_ms"),
    ("tot_ms", "mean_time_total_no_draw_ms"),
]


# Resolved-parameter columns, pulled verbatim from each run's config.json so every
# row is self-describing (records the value of every factor that varies in the grid,
# not just the one named in the label). Column name == config.json key; full values.
PARAM_COLS = [
    "detector",
    "descriptor",
    "lk_on",
    "detection_period",
    "triangulation_method",
    "target_per_bucket",
    "search_radius",
    "ratio",
    "soft_spawn",
    "matcher",
    "xfeat_mnn_min_cossim",
]


def read_config(out_dir: Path) -> dict:
    path = out_dir / "config.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_command(config: Path, out_dir: Path, num_frames: int, tokens: list) -> list:
    return [
        sys.executable, str(EVAL_SCRIPT),
        "--config", str(config),
        "--start", "0",
        "--num-frames", str(num_frames),
        "--output-root", str(out_dir),
        "--minimal-output",
    ] + BASE_TOKENS + tokens


def read_summary(out_dir: Path) -> dict:
    path = out_dir / "summary_metrics.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def fmt(value: str) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if x != x:  # NaN
        return "n/a"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def safe_name(label: str) -> str:
    for ch in "|=/\\ ":
        label = label.replace(ch, "_")
    return label


def main():
    ap = argparse.ArgumentParser(description="Factorial ablation runner for sparse-depth eval.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--num-frames", type=int, default=50)
    ap.add_argument("--out-root", type=Path, default=REPO / "outputs" / "ablation")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to run groups by prefix (e.g. core, core-xfeat, ofat, learned).")
    ap.add_argument("--collect-only", action="store_true",
                    help="Skip running; rebuild the CSV/table from existing run dirs.")
    args = ap.parse_args()

    variants = build_runs()
    if args.only:
        prefixes = tuple(args.only)
        variants = [v for v in variants if v[0].startswith(prefixes)]

    args.out_root.mkdir(parents=True, exist_ok=True)

    # Force CPU-only for every child process so timings stay comparable.
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = "-1"

    collected = []  # (group, label, params, summary)
    mode = "collect-only" if args.collect_only else "run"
    print(f"Ablation ({mode}): {len(variants)} runs, seq from {args.config.name}, {args.num_frames} frames each, CPU-only.\n")
    for i, (axis, label, tokens) in enumerate(variants, 1):
        out_dir = args.out_root / safe_name(label)
        if args.collect_only:
            summary = read_summary(out_dir)
            print(f"[{i}/{len(variants)}] {axis:11s} {label:26s} ... {'found' if summary else 'MISSING'}")
            collected.append((axis, label, read_config(out_dir), summary))
            continue
        cmd = build_command(args.config, out_dir, args.num_frames, tokens)
        print(f"[{i}/{len(variants)}] {axis:11s} {label:26s} ...", end=" ", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, env=child_env, cwd=str(REPO),
                              capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"FAILED ({dt:.1f}s)")
            print("  last stderr:", proc.stderr.strip().splitlines()[-1:] or proc.stdout.strip().splitlines()[-1:])
            collected.append((axis, label, {}, {}))
            continue
        collected.append((axis, label, read_config(out_dir), read_summary(out_dir)))
        print(f"ok ({dt:.1f}s)")

    # Build the combined CSV: group, label, resolved parameters, then metrics.
    combined_path = args.out_root / "ablation_summary.csv"
    fieldnames = ["group", "label"] + PARAM_COLS + [name for name, _ in HEADLINE]
    with open(combined_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for axis, label, params, summary in collected:
            row = {"group": axis, "label": label}
            for c in PARAM_COLS:
                row[c] = params.get(c, "")
            for name, col in HEADLINE:
                row[name] = summary.get(col, "")
            w.writerow(row)

    # Print an aligned table (metrics view; full resolved params live in the CSV).
    print("\n================ ABLATION SUMMARY ================")
    cols = ["label"] + [name for name, _ in HEADLINE]
    widths = {c: len(c) for c in cols}
    table = []
    for axis, label, params, summary in collected:
        r = {"label": label}
        for name, col in HEADLINE:
            r[name] = fmt(summary.get(col, "")) if summary else "n/a"
        for c in cols:
            widths[c] = max(widths[c], len(str(r[c])))
        table.append((axis, r))
    header = "  ".join(c.rjust(widths[c]) if c != "label" else c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    last_axis = None
    for axis, r in table:
        if last_axis is not None and axis != last_axis:
            print()  # blank line between axes
        last_axis = axis
        line = "  ".join(
            (str(r[c]).ljust(widths[c]) if c == "label" else str(r[c]).rjust(widths[c]))
            for c in cols
        )
        print(line)
    print("\nSaved:", combined_path)


if __name__ == "__main__":
    main()
