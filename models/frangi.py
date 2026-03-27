# frangi_baseline.py
"""==================
Frangi Vesselness Baseline with Topological Pruning.

Workflow:
  1. Multi-scale Frangi filter (Hessian-based enhancement)
  2. Gaussian smoothing to suppress background noise before thresholding
  3. Morphological cleanup (Binary closing + size filtering)
  4. Skeletonization (1-pixel centerline extraction)
  5. Skan Pruning: Removes 'Type 1' spurs (Tip-to-Junction) below prune_length
==================
"""

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from skan import Skeleton as SkanSkeleton
from skan import summarize
from skimage import filters, morphology
from skimage.morphology import remove_small_objects, skeletonize


class FrangiBaseline:
    """Classical vessel segmentation using Frangi filter, with graph-based pruning."""

    def __init__(
        self,
        sigma_min: float = 1.0,
        sigma_max: float = 3.0,
        num_scales: int = 5,
        threshold: float = 0.05,
        min_size: int = 50,
        prune_length: int = 10,
        gauss_sigma: float = 1.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales
        self.threshold = threshold
        self.min_size = min_size
        self.prune_length = prune_length
        self.gauss_sigma = gauss_sigma

    def extract_centerline(
        self,
        preprocessed: np.ndarray,
        fov_mask: Optional[np.ndarray] = None,
        return_vesselness: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Extract a 1-pixel skeleton from a preprocessed fundus image.

        Args:
            preprocessed: (H, W) float32 CLAHE-enhanced grayscale [0, 1]
            fov_mask:     (H, W) uint8 FOV mask {0, 255}
        """
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.frangi(
            preprocessed.astype(np.float64), sigmas=sigmas, black_ridges=True
        )
        vesselness = (vesselness - vesselness.min()) / (
            vesselness.max() - vesselness.min() + 1e-8
        )

        if self.gauss_sigma > 0:
            vesselness = gaussian_filter(vesselness, sigma=self.gauss_sigma)

        if fov_mask is not None:
            vesselness *= fov_mask > 0

        binary = vesselness > self.threshold
        binary = morphology.binary_closing(binary, morphology.disk(1))
        binary = remove_small_objects(binary.astype(bool), min_size=self.min_size)

        skeleton = skeletonize(binary)

        if np.any(skeleton):
            skeleton = self._prune_with_skan(skeleton)

        if not return_vesselness:
            vesselness = None

        return skeleton.astype(np.float32), vesselness, binary.astype(np.uint8)

    def _prune_with_skan(self, skeleton_img):
        try:
            skel = SkanSkeleton(skeleton_img)
            stats = summarize(skel, separator="_")
            short_tips = stats[
                (stats["branch-type"] == 1)
                & (stats["branch-distance"] < self.prune_length)
            ]
            pruned_skeleton = skeleton_img.copy()
            for edge_idx in short_tips.index:
                coords = skel.path_coordinates(edge_idx)
                for r, c in coords.astype(int):
                    pruned_skeleton[r, c] = 0
            return skeletonize(pruned_skeleton)
        except Exception:
            return skeleton_img
