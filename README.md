# sparse_depth

Monocular **sparse depth from motion** on the KITTI odometry benchmark. The
pipeline detects keypoints, tracks/matches them across frames, and triangulates
per-point depth using the dataset's ground-truth camera poses. It is a research
harness for studying the **coverage ↔ accuracy ↔ compute** trade-offs of
different feature front-ends, with an interactive viewer and a headless
evaluator that share the exact same feature-manager code.

The pipeline is a *hybrid* front-end built from four orthogonal axes, so you can
mix and match:

| Axis | Options | What it controls |
|------|---------|------------------|
| **detector**   | `shi` · `sift` · `xfeat` | where keypoints are placed |
| **descriptor** | `sift` · `xfeat`         | the fingerprint used for matching |
| **matcher**    | `radius_lowe` · `xfeat_mnn` | how detections are associated to tracks |
| **LK tracking**| `--lk-on` / `--no-lk-on` | optical-flow continuation before matching |

Key empirical findings from the built-in ablations (KITTI seq04/seq10):

- The dominant effect is the **detector × tracking interaction**: Shi–Tomasi
  ("good features to *track*") is strongest with LK **on**; SIFT ("good features
  to *match*") is strongest with LK **off** (pure descriptor matching), where it
  gives the best drift-free median relative error.
- **Triangulation method** is a coverage/outlier/compute knob, not a
  median-accuracy knob.
- **XFeat** (learned detector+descriptor, CPU-friendly) with its own mutual-NN
  matcher recovers far more points than radius+Lowe on native XFeat descriptors,
  reaching roughly LK-on-classical quality.

## Repository layout

```
sparse_depth/                          the package (front-end + geometry + I/O)
  feature_manager.py                   core hybrid manager (class FeatureManager)
  manager_config.py                    Config dataclass + axis resolution
  track_types.py                       Track / Observation / FrameStats
  interactive_feature_manager_kitti.py interactive OpenCV viewer
  feature_utils.py                     detectors/descriptors (Shi/SIFT/XFeat)
  xfeat_detector.py                    XFeat wrapper (CPU-forced)
  triangulation.py                     DLT / multiview / pair triangulation backends
  geometry.py, pose_eval.py            epipolar geometry & pose diagnostics
  eval_metrics.py, kitti_io.py         metrics & KITTI data loaders
  config_io.py                         TOML config + argparse layering
  ground_plane.py, plane_homography.py, plane_validation.py,
  interactive_ground_plane_viewer_kitti.py,
  interactive_plane_homography_viewer_kitti.py   ground/plane subsystem
sparse_depth_eval_kitti.py             headless evaluator (CSV metrics)
run_ablation.py                        factorial ablation runner
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
per sequence — uncomment the one you want:

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

## Usage

All commands read `configs/default.toml` for data paths and baseline parameters;
the CLI flags below just override individual axes.

### Headless evaluation

Writes `summary_metrics.csv`, `per_frame_metrics.csv`, `config.json`, and (unless
`--minimal-output`) diagnostic plots to `--output-root`.

```powershell
# 1. Baseline — hybrid SIFT + LK, windowed multiview
python sparse_depth_eval_kitti.py --config configs/default.toml `
  --detector sift --descriptor sift --lk-on `
  --triangulation-method windowed_multiview_dlt `
  --start 0 --num-frames 50 --output-root outputs/eval_sift_lk_windowed

# 2. Accuracy champion — SIFT, LK off, pure descriptor matching, tight ratio
python sparse_depth_eval_kitti.py --config configs/default.toml `
  --detector sift --descriptor sift --no-lk-on `
  --matcher radius_lowe --ratio 0.70 --search-radius 30 `
  --triangulation-method best_pair_dlt `
  --start 0 --num-frames 50 --output-root outputs/eval_sift_lkoff_ratio70

# 3. Shi-Tomasi + LK — tracking-friendly detector
python sparse_depth_eval_kitti.py --config configs/default.toml `
  --detector shi --descriptor sift --lk-on `
  --triangulation-method windowed_multiview_dlt `
  --start 0 --num-frames 50 --output-root outputs/eval_shi_lk

# 4. XFeat end-to-end — learned detector + descriptor + its own MNN matcher
python sparse_depth_eval_kitti.py --config configs/default.toml `
  --detector xfeat --descriptor xfeat --no-lk-on `
  --matcher xfeat_mnn --xfeat-mnn-min-cossim 0.82 --xfeat-top-k 4096 `
  --triangulation-method best_pair_dlt `
  --start 0 --num-frames 50 --output-root outputs/eval_xfeat_native_mnn

# 5. Coverage-oriented — denser buckets
python sparse_depth_eval_kitti.py --config configs/default.toml `
  --detector sift --descriptor sift --lk-on `
  --target-per-bucket 20 --triangulation-method windowed_multiview_dlt `
  --start 0 --num-frames 50 --output-root outputs/eval_sift_lk_dense
```

Add `--minimal-output` for fast sweeps (CSV + `config.json` only, no plots).

### Interactive viewer

Same axis flags as the evaluator; page through frames by hand. Launch as a
module so package imports resolve:

```powershell
# 1. Baseline — hybrid SIFT + LK
python -m sparse_depth.interactive_feature_manager_kitti --config configs/default.toml `
  --detector sift --descriptor sift --lk-on `
  --triangulation-method windowed_multiview_dlt --start 0

# 2. SIFT, LK off, pure matching
python -m sparse_depth.interactive_feature_manager_kitti --config configs/default.toml `
  --detector sift --descriptor sift --no-lk-on `
  --matcher radius_lowe --ratio 0.70 --search-radius 30 `
  --triangulation-method best_pair_dlt --start 0

# 3. Shi-Tomasi + LK
python -m sparse_depth.interactive_feature_manager_kitti --config configs/default.toml `
  --detector shi --descriptor sift --lk-on `
  --triangulation-method windowed_multiview_dlt --start 0

# 4. XFeat end-to-end + MNN matcher
python -m sparse_depth.interactive_feature_manager_kitti --config configs/default.toml `
  --detector xfeat --descriptor xfeat --no-lk-on `
  --matcher xfeat_mnn --xfeat-mnn-min-cossim 0.82 --xfeat-top-k 4096 `
  --triangulation-method best_pair_dlt --start 0

# 5. Coverage-oriented — denser buckets
python -m sparse_depth.interactive_feature_manager_kitti --config configs/default.toml `
  --detector sift --descriptor sift --lk-on `
  --target-per-bucket 20 --triangulation-method windowed_multiview_dlt --start 0
```

Use `--end-frame N` to cap the range; omit it to walk the whole sequence.
`--resize` scales the display window.

### Ablation sweeps

`run_ablation.py` runs a predefined factorial grid over the axes and writes one
comparison CSV (`ablation_summary.csv`) with per-run resolved parameters and
median+mean metrics:

```powershell
python run_ablation.py --num-frames 50                 # full grid
python run_ablation.py --only period core --num-frames 50   # selected groups
```

## Metrics

The evaluator reports, per run: **image coverage** (fraction of grid buckets
with depth), **GT coverage** (fraction of projected LiDAR points a sparse
estimate covers), **matched point counts** (median + mean), **relative and
absolute depth error** (median + mean, robust to startup zeros vs. tail
throughput), and **categorized timings** (feature front-end vs. triangulation).

## Notes

- Depth is triangulated from **ground-truth poses**, so this measures the
  *feature front-end + triangulation* quality, not pose estimation. A separate
  pose-eval diagnostic (essential-matrix RANSAC on the tracks) is reported
  alongside for reference.
- Old bundled names (`--detector-mode`, `--lk-tracking`, `--lk-sift-period`,
  `--association-matcher`, …) still work as back-compat aliases for the new
  orthogonal flags.
