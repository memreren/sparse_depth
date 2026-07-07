# sparse_depth

Monocular **sparse depth from motion** on the KITTI odometry benchmark. The
pipeline detects keypoints, tracks/matches them across frames, and triangulates
per-point depth. It is a research harness for studying the
**coverage ↔ accuracy ↔ compute** trade-offs of different feature front-ends,
with an interactive viewer and a headless evaluator that share the exact same
feature-manager code.

The pipeline is a *hybrid* front-end built from four orthogonal axes, so you can
mix and match:

| Axis | Options | What it controls |
|------|---------|------------------|
| **detector**   | `shi` · `sift` · `xfeat` | where keypoints are placed |
| **descriptor** | `sift` · `xfeat`         | the fingerprint used for matching |
| **matcher**    | `radius_lowe` · `xfeat_mnn` | how detections are associated to tracks |
| **LK tracking**| `--lk-on` / `--no-lk-on` | optical-flow continuation before matching |

Depth is triangulated from **estimated** camera poses by default: a lightweight
frame-to-frame essential-matrix backend recovers rotation and translation
*direction*, with the per-step translation *magnitude* taken from the
ground-truth poses so the metric scale is fixed (monocular vision cannot recover
absolute scale on its own). Pass `--pose-source gt` to use ground-truth poses
directly.

Key empirical findings from the ablations (KITTI seq04/seq10):

- The dominant effect is the **detector × tracking interaction**: Shi–Tomasi
  ("good features to *track*") is strongest with LK **on**; SIFT ("good features
  to *match*") is strongest with LK **off** (pure descriptor matching), where it
  gives the best drift-free median relative error.
- **Triangulation method** is a coverage/outlier/compute knob, not a
  median-accuracy knob.
- **XFeat** (learned detector+descriptor, CPU-friendly) with its own mutual-NN
  matcher recovers far more points than radius+Lowe on native XFeat descriptors,
  reaching roughly LK-on-classical quality.

## Two entry points

| Script | Purpose |
|--------|---------|
| `python -m sparse_depth.interactive_feature_manager_kitti` | **Interactive viewer** — page through frames, inspect tracks/geometry/depth by eye. |
| `python evaluate_kitti.py` | **Headless evaluator** — run a sequence, write a small CSV of depth metrics, optionally a depth video. |

Both read `configs/default.toml` for data paths and baseline parameters; CLI
flags override individual values.

## Repository layout

```
sparse_depth/                          the package (front-end + geometry + I/O)
  feature_manager.py                   core hybrid manager (class FeatureManager)
  manager_config.py                    Config dataclass + axis resolution
  cli_config.py                        CLI flags -> Config (shared by both entry points)
  track_types.py                       Track / Observation / FrameStats
  interactive_feature_manager_kitti.py interactive OpenCV viewer  (entry point 1)
  feature_utils.py                     detectors/descriptors (Shi/SIFT/XFeat)
  xfeat_detector.py                    XFeat wrapper (CPU-forced)
  triangulation.py                     DLT / multiview / pair / TTC backends
  pose_backend.py                      frame-to-frame essential-matrix pose estimation
  geometry.py, pose_eval.py            epipolar geometry & pose diagnostics
  eval_metrics.py, kitti_io.py         metrics & KITTI data loaders
  config_io.py                         TOML config + argparse layering
evaluate_kitti.py                      headless evaluator                    (entry point 2)
configs/default.toml                   unified config (paths + baseline params)
```

## Installation

Python 3.11+ (the config loader uses the stdlib `tomllib`).

```bash
pip install numpy opencv-python matplotlib scipy
# torch is only needed for the XFeat detector/descriptor modes:
pip install torch
```

XFeat is loaded on first use via `torch.hub` (`verlab/accelerated_features`) and
runs on CPU by default, so no GPU is required.

## Data setup

KITTI data is **not** included (it is gitignored). Point the config at your local
copy. `configs/default.toml` holds the shared paths and one `[sequenceNN]` block
per sequence — uncomment the one you want and comment out the others:

```toml
[sequence10]
img_dir  = "data/data_odometry_gray/dataset/sequences/10/image_0"
calib    = "data/data_odometry_gray/dataset/sequences/10/calib.txt"
poses    = "data/data_odometry_poses/dataset/poses/10.txt"
image_digits = 6
# optional LiDAR (for ground-truth depth metrics):
velodyne_dir       = "data/2011_09_30_drive_0034_sync/velodyne_points/data"
calib_velo_to_cam  = "data/2011_09_30_calib/2011_09_30/calib_velo_to_cam.txt"
calib_cam_to_cam   = "data/2011_09_30_calib/2011_09_30/calib_cam_to_cam.txt"
```

You need the KITTI odometry **grayscale images**, **ground-truth poses**, and
(for depth metrics) the raw **Velodyne** scans + calibration.

## Headless evaluation

`evaluate_kitti.py` runs the pipeline over the configured sequence and writes, to
`--output-root`:

- `per_frame.csv` — one row per frame: `frame, n_points, img_cov, gt_cov, medRel, meanRel, delta20`
- `summary.csv` — one row of run-level medians/means (plus pose-estimation error)
- `depth.mp4` — only with `--save-video`: colored sparse depth over the frame
- `config.json` — the resolved run configuration

```powershell
# Metrics only (250 frames)
python evaluate_kitti.py --config configs/default.toml `
  --num-frames 250 --output-root outputs/eval_seq04

# Metrics + depth video
python evaluate_kitti.py --config configs/default.toml `
  --num-frames 250 --save-video --output-root outputs/eval_seq04

# Swap the triangulation backend or use GT poses
python evaluate_kitti.py --config configs/default.toml `
  --triangulation-method best_pair_dlt --pose-source gt `
  --num-frames 250 --output-root outputs/eval_seq04_bestpair_gt
```

### Metrics reported

- **n_points** — accepted sparse depth points that frame.
- **img_cov** — fraction of image grid cells that hold a depth point (spatial spread).
- **gt_cov** — fraction of projected LiDAR points that have a sparse estimate nearby.
- **medRel / meanRel** — median / mean relative depth error vs. matched LiDAR (`|z−z_gt|/z_gt`).
- **delta20** — fraction of points within 20 % of LiDAR depth (higher is better).
- **ms/frame** — wall-clock throughput.
- **pose_median_rot_err_deg / pose_median_t_err_deg** — pose-backend accuracy vs. GT (estimated mode only).

## Interactive viewer

Launch as a module so package imports resolve. Same axis flags as the evaluator;
you step through frames by hand.

```powershell
# Baseline
python -m sparse_depth.interactive_feature_manager_kitti --config configs/default.toml --start 0

# Shi-Tomasi + LK
python -m sparse_depth.interactive_feature_manager_kitti --config configs/default.toml `
  --detector shi --descriptor sift --lk-on --start 0

# XFeat end-to-end + MNN matcher
python -m sparse_depth.interactive_feature_manager_kitti --config configs/default.toml `
  --detector xfeat --descriptor xfeat --no-lk-on --matcher xfeat_mnn --start 0
```

### View modes (keys `1`–`7`, or `m` to cycle)

| Key | Mode | Shows |
|-----|------|-------|
| `1` | **status**  | tracks colored by state (confirmed / candidate / reacquired / lost) |
| `2` | **buckets** | the spatial grid and per-bucket population used for spawning |
| `3` | **age**     | tracks colored by how many frames they have survived |
| `4` | **triang**  | triangulation gate outcome per track (good / bad epipolar / low parallax / …) |
| `5` | **depth**   | per-point triangulated depth, color-coded, with LiDAR comparison in the panel |
| `6` | **quality** | tracks colored by their quality score (the selection ranking) |
| `7` | **pose**    | estimated-vs-GT relative-pose diagnostics for the current frame gap |

### Keybindings

| Key | Action |
|-----|--------|
| `n` / `d` / `→` | next frame |
| `b` / `a` / `←` | previous cached frame |
| `m` | cycle view mode |
| `1`–`7` | jump to a view mode |
| `p` | toggle track paths |
| `v` | toggle velocity arrows |
| `l` | toggle lost / predicted tracks |
| `G` | toggle bucket grid overlay |
| `+` / `-` | increase / decrease displayed track cap |
| `r` | reset the manager at the current frame |
| `R` | reset the manager at the original `--start` frame |
| `s` | save a screenshot to `--output-root` |
| `h` | print the controls to the console |
| `q` / `Esc` | quit |
| mouse left-click | print the nearest active track's details to the console |

### Color meaning

In the **depth** view points are colored by distance (Turbo colormap, warm =
near, cool = far); the side panel reports matched-LiDAR error for the frame. In
**status**/**age**/**quality** the color encodes the named attribute rather than
depth. Use `--end-frame N` to cap the range; `--resize` scales the window.

## Configuration

`configs/default.toml` is organized into sections (`[frontend]`, `[lk]`,
`[association]`, `[triangulation]`, `[lidar]`, …). Section names are cosmetic —
each key maps to a CLI flag. Precedence is: **CLI flag > config file > built-in
default**. You can pass `--config` more than once; later files override earlier
ones. Old bundled names (`--detector-mode`, `--lk-tracking`, `--lk-sift-period`,
`--association-matcher`, …) still work as back-compat aliases.

## Notes

- Absolute scale comes from GT translation magnitude per step; everything else in
  the pose (rotation, translation direction) is estimated. This isolates the
  *feature front-end + triangulation* quality from monocular scale ambiguity.
- Frames at the very start of a run report zero points until a track has enough
  views for the chosen triangulation method (e.g. 3 for windowed multiview).
