# evaluation/metrics.py
"""Evaluation metrics for retinal vessel centerline extraction.

Includes:
- Centerline F1 at multiple tolerances (distance-based)
- clDice (mask-based, hard skeletonization — for evaluation only)
- cl_dice_from_probs (threshold prob map → binary → clDice)
- Betti-0 error (connected component difference)
- HD95 (95th percentile Hausdorff distance)
- IoU (Intersection over Union for binary vessel masks)

Note on clDice:
    Training uses a differentiable soft-skeleton approximation (see CenterlineLoss).
    Evaluation uses skimage.skeletonize on the thresholded probability map, which
    is the correct hard metric to report in results. The two are intentionally
    different: the soft version exists only to make gradients flow during training.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize


class CenterlineMetrics:
    """Compute evaluation metrics for predicted vs ground-truth skeletons
    and vessel masks.
    """

    def __init__(self, tolerance_levels: List[int] = [1, 2, 3]):
        self.tolerance_levels = tolerance_levels

    # ============================================================
    # MAIN ENTRY
    # ============================================================

    def compute_all_metrics(
        self,
        pred_skeleton: np.ndarray,
        gt_skeleton: np.ndarray,
        pred_vessel_mask: Optional[np.ndarray] = None,
        gt_vessel_mask: Optional[np.ndarray] = None,
        pred_prob: Optional[np.ndarray] = None,  # raw model prob map (H, W) float [0,1]
        fov_mask: Optional[
            np.ndarray
        ] = None,  # FOV mask — metrics computed inside ROI only
    ) -> Dict[str, float]:
        """Compute all metrics for a single prediction.

        Args:
            pred_skeleton    : (H, W) uint8  — predicted binary centerline
            gt_skeleton      : (H, W) uint8  — ground-truth binary centerline
            pred_vessel_mask : (H, W) uint8  — predicted binary vessel mask (optional)
            gt_vessel_mask   : (H, W) uint8  — GT binary vessel mask (optional)
            pred_prob        : (H, W) float  — raw sigmoid output; if provided,
                               clDice is computed via thresholding rather than
                               from pred_vessel_mask (preferred)
            fov_mask         : (H, W) uint8/bool — if provided, all metrics are
                               restricted to the FOV region (excludes black padding)

        """
        metrics = {}

        # --------------------------------------------------------
        # Apply FOV mask — zero out everything outside the retina
        # --------------------------------------------------------
        if fov_mask is not None:
            fov = fov_mask > 0
            pred_skeleton = pred_skeleton * fov
            gt_skeleton = gt_skeleton * fov
            if pred_vessel_mask is not None:
                pred_vessel_mask = pred_vessel_mask * fov
            if gt_vessel_mask is not None:
                gt_vessel_mask = gt_vessel_mask * fov
            if pred_prob is not None:
                pred_prob = pred_prob * fov

        # --------------------------------------------------------
        # 1. Centerline F1 Scores (skeleton-based)
        # --------------------------------------------------------
        for tau in self.tolerance_levels:
            precision, recall, f1 = self.centerline_f1(
                pred_skeleton,
                gt_skeleton,
                tau,
            )
            metrics[f"precision@{tau}px"] = precision
            metrics[f"recall@{tau}px"] = recall
            metrics[f"f1@{tau}px"] = f1

        # --------------------------------------------------------
        # 2. clDice (mask-based)
        #    Prefer pred_prob (thresholded) over pred_vessel_mask
        #    because using the full vessel mask — not the skeleton —
        #    is what clDice was designed for.
        # --------------------------------------------------------
        if gt_vessel_mask is not None:
            if pred_prob is not None:
                metrics["clDice"] = self.cl_dice_from_probs(pred_prob, gt_vessel_mask)
            elif pred_vessel_mask is not None:
                metrics["clDice"] = self.cl_dice(pred_vessel_mask, gt_vessel_mask)

        # --------------------------------------------------------
        # 3. IoU (mask-based)
        # --------------------------------------------------------
        if pred_vessel_mask is not None and gt_vessel_mask is not None:
            metrics["iou"] = self.iou(pred_vessel_mask, gt_vessel_mask)

        # --------------------------------------------------------
        # 4. Topology Metrics
        # --------------------------------------------------------
        metrics["betti_0_error"] = self.betti_0_error(pred_skeleton, gt_skeleton)
        metrics["hd95"] = self.hd95(pred_skeleton, gt_skeleton)

        return metrics

    # ============================================================
    # CENTERLINE F1 (Distance-Tolerant, Vectorized)
    # ============================================================

    def centerline_f1(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        tolerance: int = 2,
    ) -> Tuple[float, float, float]:
        """Compute centerline F1 with Euclidean distance tolerance.

        A predicted pixel is a true positive if it lies within
        `tolerance` pixels of any GT centerline pixel, and vice versa.
        """
        pred_bin = pred > 0
        gt_bin = gt > 0

        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            return 1.0, 1.0, 1.0
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return 0.0, 0.0, 0.0

        gt_dist = ndimage.distance_transform_edt(~gt_bin)
        pred_dist = ndimage.distance_transform_edt(~pred_bin)

        tp_precision = int((gt_dist[pred_bin] <= tolerance).sum())
        tp_recall = int((pred_dist[gt_bin] <= tolerance).sum())

        precision = tp_precision / float(pred_bin.sum())
        recall = tp_recall / float(gt_bin.sum())

        if precision + recall == 0:
            return 0.0, 0.0, 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    # ============================================================
    # clDice (Hard, Mask-Based — evaluation only)
    # ============================================================

    def cl_dice(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> float:
        """Hard clDice from binary vessel masks.

        Tprec = |S(P) ∩ G| / |S(P)|
        Tsens = |S(G) ∩ P| / |S(G)|
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens)

        P = predicted vessel mask
        G = ground-truth vessel mask
        S(.) = skimage.skeletonize  (hard, non-differentiable)

        Use this for evaluation. For training, see CenterlineLoss
        which uses a differentiable soft-skeleton approximation.
        """
        pred_bin = pred_mask > 0
        gt_bin = gt_mask > 0

        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            return 1.0
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return 0.0

        skel_pred = skeletonize(pred_bin)
        skel_gt = skeletonize(gt_bin)

        if skel_pred.sum() == 0 or skel_gt.sum() == 0:
            return 0.0

        tprec = np.logical_and(skel_pred, gt_bin).sum() / float(skel_pred.sum())
        tsens = np.logical_and(skel_gt, pred_bin).sum() / float(skel_gt.sum())

        if tprec + tsens == 0:
            return 0.0

        return float(2 * tprec * tsens / (tprec + tsens))

    def cl_dice_from_probs(
        self,
        pred_prob: np.ndarray,
        gt_vessel_mask: np.ndarray,
        prob_threshold: float = 0.5,
    ) -> float:
        """Hard clDice computed from a raw probability map.

        Thresholds pred_prob → binary vessel mask → skeletonize → clDice.
        This is the correct evaluation path: the full thresholded vessel
        mask (not the post-processed skeleton) is what clDice operates on.

        Use this for evaluation, NOT for training.
        """
        pred_bin = (pred_prob >= prob_threshold).astype(np.uint8)
        return self.cl_dice(pred_bin, gt_vessel_mask)

    # ============================================================
    # IoU (Mask-Based)
    # ============================================================

    def iou(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> float:
        """Intersection over Union for binary vessel masks.

        IoU = |P ∩ G| / |P ∪ G|

        Returns 1.0 if both masks are empty (perfect agreement),
        0.0 if exactly one is empty.
        """
        pred_bin = pred_mask > 0
        gt_bin = gt_mask > 0

        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            return 1.0
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return 0.0

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()

        return float(intersection / union)

    # ============================================================
    # BETTI-0 ERROR
    # ============================================================

    def betti_0_error(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> float:
        """Absolute difference in number of connected components (8-connectivity).
        Lower is better; 0 means topology matches GT exactly.
        """
        _, pred_b0 = measure.label(pred > 0, return_num=True, connectivity=2)
        _, gt_b0 = measure.label(gt > 0, return_num=True, connectivity=2)

        return float(abs(int(pred_b0) - int(gt_b0)))

    # ============================================================
    # HD95 (Symmetric)
    # ============================================================

    def hd95(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> float:
        """95th percentile symmetric Hausdorff distance (pixels).

        When one input is empty, returns the image diagonal as a
        worst-case penalty (conventional choice — document in thesis).
        """
        p_bin = pred > 0
        g_bin = gt > 0

        if p_bin.sum() == 0 and g_bin.sum() == 0:
            return 0.0

        if p_bin.sum() == 0 or g_bin.sum() == 0:
            # Worst-case penalty: image diagonal
            return float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))

        p_dist_map = ndimage.distance_transform_edt(~p_bin)
        g_dist_map = ndimage.distance_transform_edt(~g_bin)

        hd95_p_g = float(np.percentile(g_dist_map[p_bin], 95))
        hd95_g_p = float(np.percentile(p_dist_map[g_bin], 95))

        return max(hd95_p_g, hd95_g_p)
