# frangi.py
"""
==================
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

# Local imports
from data.fundus_preprocessor import FundusPreprocessor


class FrangiBaseline:
    """Classical vessel segmentation using Frangi filter, with graph-based pruning.

    Pipeline:
    1. Preprocessing (CLAHE, Green channel, Masking)
    2. Frangi vesselness filter
    3. Gaussian smoothing (suppresses background noise before thresholding)
    4. Binary Thresholding
    5. Small object removal
    6. Skeletonization (Source of Truth)
    7. SKAN Pruning (Graph-based removal of spurious tips)
    """

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

        self.preprocessor = FundusPreprocessor()

    def extract_centerline(
        self,
        image: np.ndarray,
        return_vesselness: bool = False,
        external_fov_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Extract a 1-pixel skeleton from a fundus image.

        Returns:
            skeleton (np.ndarray): 1-pixel centerline of vessels
            vesselness (np.ndarray or None): Frangi vesselness map if requested
            binary_mask (np.ndarray): Binary vessel mask after thresholding and cleanup

        """
        preprocessed, _, _, _, mask = self.preprocessor.preprocess(
            image, external_mask=external_fov_mask, return_intermediate=True
        )

        # 1. Frangi vesselness
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.frangi(
            preprocessed.astype(np.float64), sigmas=sigmas, black_ridges=True
        )
        vesselness = (vesselness - vesselness.min()) / (
            vesselness.max() - vesselness.min() + 1e-8
        )

        # 2. Gaussian smoothing — suppresses background noise before thresholding
        #    Keep gauss_sigma in range 1.0–1.5; higher values blur vessel edges
        if self.gauss_sigma > 0:
            vesselness = gaussian_filter(vesselness, sigma=self.gauss_sigma)

        vesselness *= mask > 0

        # 3. Binary segmentation + morphological cleanup
        binary = vesselness > self.threshold
        binary = morphology.binary_closing(binary, morphology.disk(1))
        binary = remove_small_objects(binary.astype(bool), min_size=self.min_size)

        # 4. Skeletonization
        skeleton = skeletonize(binary)

        # 5. Graph-based pruning with SKAN
        if np.any(skeleton):
            skeleton = self._prune_with_skan(skeleton)

        # 6. Return tuple (skeleton, vesselness, binary)
        if not return_vesselness:
            vesselness = None

        return skeleton.astype(np.float32), vesselness, binary.astype(np.uint8)

    def _prune_with_skan(self, skeleton_img: np.ndarray) -> np.ndarray:
        """Remove short spur branches ending in tips using SKAN graph.
        """
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
            # fallback if SKAN fails
            return skeleton_img
