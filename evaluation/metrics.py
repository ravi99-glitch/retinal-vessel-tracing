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

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize
from typing import Dict, List, Optional, Tuple

# --- Standalone helper for use in RL Rewards ---
def compute_betti0(binary_mask: np.ndarray, connectivity: int = 2) -> int:
    """Count connected components (Betti-0) of a binary mask."""
    if binary_mask.sum() == 0:
        return 0
    _, n_components = measure.label(
        binary_mask > 0, return_num=True, connectivity=connectivity
    )
    return int(n_components)

class CenterlineMetrics:
    def __init__(self, tolerance_levels: List[int] = [1, 2, 3]):
        self.tolerance_levels = tolerance_levels

    def compute_all_metrics(self, pred_skeleton, gt_skeleton, pred_vessel_mask=None, 
                           gt_vessel_mask=None, pred_prob=None, fov_mask=None) -> Dict[str, float]:
        metrics = {}
        # Apply FOV mask
        if fov_mask is not None:
            fov = fov_mask > 0
            pred_skeleton = pred_skeleton * fov
            gt_skeleton = gt_skeleton * fov
            if pred_vessel_mask is not None: pred_vessel_mask = pred_vessel_mask * fov
            if gt_vessel_mask is not None: gt_vessel_mask = gt_vessel_mask * fov

        # 1. Centerline F1
        for tau in self.tolerance_levels:
            precision, recall, f1 = self.centerline_f1(pred_skeleton, gt_skeleton, tau)
            metrics[f"precision@{tau}px"] = precision
            metrics[f"recall@{tau}px"] = recall
            metrics[f"f1@{tau}px"] = f1

        # 2. clDice
        if gt_vessel_mask is not None:
            if pred_prob is not None:
                metrics["clDice"] = self.cl_dice_from_probs(pred_prob, gt_vessel_mask)
            elif pred_vessel_mask is not None:
                metrics["clDice"] = self.cl_dice(pred_vessel_mask, gt_vessel_mask)

        # 3. IoU
        if pred_vessel_mask is not None and gt_vessel_mask is not None:
            metrics["iou"] = self.iou(pred_vessel_mask, gt_vessel_mask)

        # 4. Topology
        metrics["betti_0_error"] = self.betti_0_error(pred_skeleton, gt_skeleton)
        metrics["hd95"] = self.hd95(pred_skeleton, gt_skeleton)
        return metrics

    def centerline_f1(self, pred, gt, tolerance=2) -> Tuple[float, float, float]:
        p_bin, g_bin = pred > 0, gt > 0
        if p_bin.sum() == 0 and g_bin.sum() == 0: return 1.0, 1.0, 1.0 # YOUR CORRECT LOGIC
        if p_bin.sum() == 0 or g_bin.sum() == 0: return 0.0, 0.0, 0.0
        
        gt_dist = ndimage.distance_transform_edt(~g_bin)
        pred_dist = ndimage.distance_transform_edt(~p_bin)
        tp_p = int((gt_dist[p_bin] <= tolerance).sum())
        tp_r = int((pred_dist[g_bin] <= tolerance).sum())
        
        prec = tp_p / float(p_bin.sum())
        rec = tp_r / float(g_bin.sum())
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    def cl_dice(self, pred_mask, gt_mask) -> float:
        p_bin, g_bin = pred_mask > 0, gt_mask > 0
        if p_bin.sum() == 0 and g_bin.sum() == 0: return 1.0 # YOUR CORRECT LOGIC
        if p_bin.sum() == 0 or g_bin.sum() == 0: return 0.0
        
        s_p, s_g = skeletonize(p_bin), skeletonize(g_bin)
        tprec = np.logical_and(s_p, g_bin).sum() / float(s_p.sum())
        tsens = np.logical_and(s_g, p_bin).sum() / float(s_g.sum())
        return float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) > 0 else 0.0

    def cl_dice_from_probs(self, pred_prob, gt_vessel_mask, threshold=0.5) -> float:
        return self.cl_dice((pred_prob >= threshold).astype(np.uint8), gt_vessel_mask)

    def iou(self, pred_mask, gt_mask) -> float:
        p_bin, g_bin = pred_mask > 0, gt_mask > 0
        if p_bin.sum() == 0 and g_bin.sum() == 0: return 1.0 # YOUR CORRECT LOGIC
        if p_bin.sum() == 0 or g_bin.sum() == 0: return 0.0
        inter = np.logical_and(p_bin, g_bin).sum()
        union = np.logical_or(p_bin, g_bin).sum()
        return float(inter / union)

    def betti_0_error(self, pred, gt) -> float:
        # USE THE NEW STANDALONE HELPER
        return float(abs(compute_betti0(pred) - compute_betti0(gt)))

    def hd95(self, pred, gt) -> float:
        p_bin, g_bin = pred > 0, gt > 0
        if p_bin.sum() == 0 and g_bin.sum() == 0: return 0.0
        if p_bin.sum() == 0 or g_bin.sum() == 0:
            return float(np.sqrt(pred.shape[0]**2 + pred.shape[1]**2))
        p_dist = ndimage.distance_transform_edt(~p_bin)
        g_dist = ndimage.distance_transform_edt(~g_bin)
        return max(float(np.percentile(g_dist[p_bin], 95)), float(np.percentile(p_dist[g_bin], 95)))
