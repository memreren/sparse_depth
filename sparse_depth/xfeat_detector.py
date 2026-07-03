"""CPU-only XFeat keypoint detector wrapper.

XFeat (Potje et al., CVPR 2024, ``verlab/accelerated_features``) is a learned
lightweight local-feature network. Here it is used purely as a *detector*: we
take its keypoint locations and hand them to the existing SIFT descriptor +
LK-tracking pipeline (see ``feature_utils.detect_xfeat_sift``). This mirrors the
``shi_sift_lk`` mode, so every SIFT-calibrated association/quality threshold in
the manager keeps working unchanged; only the detection stage differs.

The model is loaded once via ``torch.hub`` and forced onto the CPU. We set
``CUDA_VISIBLE_DEVICES=-1`` before importing torch so this process never touches
the GPU (which may be busy with another job).
"""

from __future__ import annotations

import os

# Force CPU before torch initializes its CUDA context. "-1" hides all GPUs;
# an empty string is treated as "unset" on Windows and does NOT hide them.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from typing import Optional

import numpy as np


class XFeatDetector:
    """Lazy, CPU-pinned XFeat detector returning keypoint pixel locations."""

    def __init__(self, top_k: int = 4096, detection_threshold: float = 0.05):
        self.top_k = int(top_k)
        self.detection_threshold = float(detection_threshold)
        self._xf = None
        self._torch = None

    def _ensure_loaded(self):
        if self._xf is not None:
            return
        import torch

        if torch.cuda.is_available():  # pragma: no cover - safety net only
            raise RuntimeError(
                "XFeatDetector expects a CPU-only process; CUDA is visible. "
                "Ensure CUDA_VISIBLE_DEVICES=-1 is set before torch is imported."
            )
        torch.set_grad_enabled(False)
        xf = torch.hub.load(
            "verlab/accelerated_features", "XFeat",
            pretrained=True, top_k=self.top_k, trust_repo=True,
        )
        # The hub entrypoint auto-selects a device; pin everything to CPU.
        xf.dev = "cpu"
        xf.net = xf.net.to("cpu")
        self._xf = xf
        self._torch = torch

    def _detect(self, img_gray: np.ndarray):
        self._ensure_loaded()
        torch = self._torch
        if img_gray.ndim != 2:
            raise ValueError(f"XFeatDetector expects a 2D grayscale image, got {img_gray.shape}")
        ten = torch.from_numpy(np.ascontiguousarray(img_gray)).float()[None, None]
        return self._xf.detectAndCompute(
            ten, top_k=self.top_k, detection_threshold=self.detection_threshold
        )[0]

    def detect_points(self, img_gray: np.ndarray) -> np.ndarray:
        """Return an (N, 2) float32 array of (x, y) keypoint locations.

        ``img_gray`` is a HxW uint8 grayscale image (the manager works in
        grayscale). Keypoints come back ordered by detector score.
        """
        out = self._detect(img_gray)
        kpts = out["keypoints"].cpu().numpy()
        if kpts.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return kpts.astype(np.float32)

    def detect_points_desc(self, img_gray: np.ndarray):
        """Return (points Nx2, scores N, descriptors Nx64) as float32 arrays.

        Descriptors are XFeat's own L2-normalized 64-d features. Used by the
        xfeat_native detector mode, where matching runs on these learned
        descriptors instead of SIFT.
        """
        out = self._detect(img_gray)
        kpts = out["keypoints"].cpu().numpy()
        if kpts.size == 0:
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0, 64), dtype=np.float32))
        scores = out["scores"].cpu().numpy().astype(np.float32)
        desc = out["descriptors"].cpu().numpy().astype(np.float32)
        return kpts.astype(np.float32), scores, desc
