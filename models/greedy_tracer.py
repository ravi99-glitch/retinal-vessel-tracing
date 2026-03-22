# greedy_tracer_baseline.py
"""==================
Greedy Tracer Baseline for Vessel Extraction

Workflow:
  1. Preprocessing & Multi-scale Frangi filtering.
  2. Gaussian smoothing to stabilize steepest-ascent.
  3. FOV erosion (3 iterations) to eliminate boundary halo artifacts.
  4. Greedy steepest-ascent tracing from local maxima seeds.
  5. Post-trace object removal for noise cleanup.
  6. SKAN Pruning (New): Graph-based removal of spurious tips.
==================
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
from skan import Skeleton as SkanSkeleton
from skan import summarize
from skimage import filters
from skimage.morphology import remove_small_objects, skeletonize

from data.fundus_preprocessor import FundusPreprocessor

# ==========================================
# TRAJECTORY-RECORDING GREEDY TRACER
# ==========================================


class GreedyTracer:
    """Steepest-ascent greedy tracer on a soft probability/vesselness map.

    Returns both the binary skeleton AND full trajectory data so the
    evaluation script can render per-trace visualizations.

    NOTE: This class only handles tracing. Post-processing like
    remove_small_objects lives in GreedyTracerBaseline.extract_centerline().

    Args:
        seed_thresh : minimum vesselness to start a new trace
        step_thresh : minimum vesselness to continue stepping
        min_length  : discard traces shorter than this (pixels)
        thin_output : apply skeletonize to the final binary output

    """

    def __init__(
        self,
        seed_thresh: float = 0.15,
        step_thresh: float = 0.08,
        min_length: int = 10,
        thin_output: bool = True,
    ):
        self.seed_thresh = seed_thresh
        self.step_thresh = step_thresh
        self.min_length = min_length
        self.thin_output = thin_output

        # 8-connected neighbour offsets
        self._offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    def _local_maxima(self, prob: np.ndarray) -> np.ndarray:
        """Boolean mask of strict 8-neighbour local maxima."""
        padded = np.pad(prob, 1, mode="constant", constant_values=0)
        lm = np.ones_like(prob, dtype=bool)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[
                    1 + dy : 1 + dy + prob.shape[0], 1 + dx : 1 + dx + prob.shape[1]
                ]
                lm &= prob >= shifted
        return lm

    def _trace_from(
        self,
        prob: np.ndarray,
        visited: np.ndarray,
        start_r: int,
        start_c: int,
    ) -> List[Tuple[int, int]]:
        """Greedy steepest-ascent from a seed. Returns ordered (r, c) path."""
        H, W = prob.shape
        path = [(start_r, start_c)]
        visited[start_r, start_c] = True
        r, c = start_r, start_c

        while True:
            best_val = self.step_thresh
            best_rc = None
            for dr, dc in self._offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                    if prob[nr, nc] > best_val:
                        best_val = prob[nr, nc]
                        best_rc = (nr, nc)
            if best_rc is None:
                break
            r, c = best_rc
            visited[r, c] = True
            path.append((r, c))

        return path

    def trace(
        self,
        prob_map: np.ndarray,
        fov_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """Args:
            prob_map  : (H, W) float32 vesselness/probability map
            fov_mask  : (H, W) uint8 — trace only inside FOV

        Returns:
            skeleton  : (H, W) uint8 binary centerline {0, 255}
            traces    : list of paths, each path = ordered list of (r, c) tuples
                        sorted by visit order (trace 0 = first/strongest drawn)

        """
        prob = prob_map.copy().astype(np.float32)
        if fov_mask is not None:
            prob[fov_mask == 0] = 0.0

        H, W = prob.shape
        skeleton = np.zeros((H, W), dtype=np.uint8)
        visited = np.zeros((H, W), dtype=bool)

        # Seeds: above threshold AND local maxima
        candidates = (prob >= self.seed_thresh) & self._local_maxima(prob)
        seed_coords = np.argwhere(candidates)

        if len(seed_coords) == 0:
            return skeleton, []

        # Sort by descending vesselness (strongest ridges traced first)
        seed_probs = prob[seed_coords[:, 0], seed_coords[:, 1]]
        order = np.argsort(-seed_probs)
        seed_coords = seed_coords[order]

        traces = []

        for sr, sc in seed_coords:
            if visited[sr, sc]:
                continue
            path = self._trace_from(prob, visited, sr, sc)
            if len(path) >= self.min_length:
                for r, c in path:
                    skeleton[r, c] = 255
                traces.append(path)

        if self.thin_output and skeleton.any():
            skeleton_bool = skeletonize(skeleton > 0)

            # --- SKAN Pruning Step ---
            try:
                skel = SkanSkeleton(skeleton_bool)
                stats = summarize(skel, separator="_")

                # Find short Type 1 (tip-to-junction) branches
                short_tips = stats[
                    (stats["branch-type"] == 1)
                    & (stats["branch-distance"] < self.min_length)
                ]

                pruned = skeleton_bool.copy()
                for edge_idx in short_tips.index:
                    coords = skel.path_coordinates(edge_idx)
                    for r, c in coords.astype(int):
                        pruned[r, c] = False

                # Re-skeletonize to ensure perfect 1-pixel junctions
                skeleton_bool = skeletonize(pruned)
            except Exception:
                pass  # Fallback if graph is too small

            skeleton = (skeleton_bool * 255).astype(np.uint8)

        return skeleton, traces


# ==========================================
# GREEDY TRACER BASELINE
# ==========================================


class GreedyTracerBaseline:
    """Full pipeline: image → preprocessing → vesselness → greedy tracing → skeleton.

    Args:
        sigma_min    : smallest Frangi scale (thin capillaries)
        sigma_max    : largest  Frangi scale (wide vessels)
        num_scales   : number of scales between min and max
        gauss_sigma  : Gaussian smoothing applied to vesselness before tracing
                       suppresses noisy local maxima in background texture
        seed_thresh  : minimum vesselness to start a trace
        step_thresh  : minimum vesselness to continue stepping
        min_length   : discard traces shorter than this (pixels)
        thin_output  : apply skeletonize to the final binary output
        min_obj_size : remove isolated skeleton blobs smaller than this after
                       tracing — set to 0 to disable

    """

    def __init__(
        self,
        sigma_min: float = 0.5,
        sigma_max: float = 3.0,
        num_scales: int = 5,
        gauss_sigma: float = 1.0,
        seed_thresh: float = 0.15,
        step_thresh: float = 0.08,
        min_length: int = 10,
        thin_output: bool = True,
        min_obj_size: int = 0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales
        self.gauss_sigma = gauss_sigma
        self.min_obj_size = min_obj_size  # lives here, NOT in GreedyTracer

        self.preprocessor = FundusPreprocessor()

        self.tracer = GreedyTracer(
            seed_thresh=seed_thresh,
            step_thresh=step_thresh,
            min_length=min_length,
            thin_output=thin_output,
        )

    def _compute_vesselness(
        self,
        preprocessed: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Multi-scale Frangi filter → normalize → Gaussian smooth → FOV mask.

        Gaussian smoothing suppresses noisy local maxima in background texture
        while preserving the strong ridges of real vessel centerlines.
        """
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.frangi(
            preprocessed.astype(np.float64),
            sigmas=sigmas,
            black_ridges=True,
        )

        # Normalize to [0, 1]
        vmin, vmax = vesselness.min(), vesselness.max()
        vesselness = (vesselness - vmin) / (vmax - vmin + 1e-8)

        # Smooth to suppress noisy local maxima in background
        if self.gauss_sigma > 0:
            vesselness = gaussian_filter(vesselness, sigma=self.gauss_sigma)

        # Erode the mask to kill the Frangi edge halo artifacts
        safe_mask = binary_erosion(mask > 0, iterations=3)

        # Zero outside the NEW, slightly smaller FOV
        vesselness *= safe_mask.astype(np.float32)

        return vesselness.astype(np.float32)

    def extract_centerline(
        self,
        image: np.ndarray,
        external_fov_mask: Optional[np.ndarray] = None,
        return_vesselness: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List]:
        """Args:
            image             : (H, W, 3) RGB uint8 fundus image
            external_fov_mask : (H, W) uint8 FOV mask, or None for auto-detection
            return_vesselness : if True, also return the vesselness map

        Returns:
            skeleton          : (H, W) uint8 binary centerline {0, 255}
            vesselness        : (H, W) float32 [0,1], or None
            traces            : list of paths for trajectory visualization
                                each path = ordered list of (r, c) tuples
                                sorted by visit order (strongest vessel first)

        """
        # 1. Preprocessing
        preprocessed, _, _, _, mask = self.preprocessor.preprocess(
            image,
            external_mask=external_fov_mask,
            return_intermediate=True,
        )

        # 2. Frangi vesselness + Gaussian smoothing + Eroded FOV masking
        vesselness = self._compute_vesselness(preprocessed, mask)

        # 3. Greedy tracing — returns skeleton + full trajectory data
        # Note: We pass the UNERODED mask here so the trace can still reach
        # the real edges of the vessels, but the vesselness map itself
        # already has the halo zeroed out.
        skeleton, traces = self.tracer.trace(vesselness, fov_mask=mask)

        # 4. Remove isolated small blobs (post-trace noise cleanup)
        #    Kills scattered false-positive dots without hard thresholding.
        #    min_obj_size=0 disables this step entirely.
        if skeleton.any() and self.min_obj_size > 0:
            skeleton_bool = remove_small_objects(
                skeleton > 0, min_size=self.min_obj_size
            )
            skeleton = (skeleton_bool * 255).astype(np.uint8)

        if return_vesselness:
            return skeleton, vesselness, traces
        return skeleton, None, traces
