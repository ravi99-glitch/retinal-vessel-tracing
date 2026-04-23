# models/seed_detector.py
"""Seed detector — predicts a sparse heatmap of vessel endpoints and junctions.

Architecture: CenterlineUNet (same as baseline, see centerline_unet_baseline.py)
  - Depthwise-separable convolutions (~0.5M params)
  - 4-level encoder/decoder with skip connections
  - in_channels=3 (RGB with enhanced green)
  - Two output heads:
      1. endpoint_heatmap: sparse peaks at endpoints/junctions (focal loss)
      2. vessel_prob:      dense vessel probability (BCE) — replaces external
                           vessel_mask dependency (eliminates chicken-and-egg filtering)

Differences from original:
  - Two-head output (endpoint heatmap + vessel probability)
  - Asymmetric focal loss (recall-focused, gamma_neg > gamma_pos)
  - Inverse-thickness weighted training targets (upweights thin vessels)
  - Auxiliary seeds on long unbranched segments (improves coverage)
  - Coverage-aware seed sampling (greedy novelty × confidence scoring)
  - Stochastic seed sampling during training (temperature-based exploration)
  - Curriculum-based seed difficulty filtering
  - Reduced confidence_threshold (0.1), increased top_k (120), tighter nms_radius (8)
  - External vessel_mask filter removed from detect_seeds (was chicken-and-egg)

Training logic lives in training/seed_detector_trainer.py.
Used at inference time by scripts/drive_rl_tracing.py via FrontierTracer.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_blocks import DownBlock, DSConvBlock, UpBlock


# ==========================================
# TRAINING TARGET CONSTRUCTION
# ==========================================

def detect_endpoints_and_junctions(skeleton: np.ndarray):
    """Return (endpoints, junctions) as lists of (y, x) tuples.

    A skeleton pixel is an endpoint  if it has exactly 1 neighbour.
    A skeleton pixel is a junction   if it has 3 or more neighbours.
    """
    from scipy.ndimage import convolve

    kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.uint8)

    skel_bin = (skeleton > 0).astype(np.uint8)
    neighbour_count = convolve(skel_bin, kernel, mode="constant", cval=0)

    endpoint_mask = skel_bin & (neighbour_count == 1)
    junction_mask = skel_bin & (neighbour_count >= 3)

    endpoints = list(zip(*np.where(endpoint_mask)))
    junctions = list(zip(*np.where(junction_mask)))

    return [(int(y), int(x)) for y, x in endpoints], \
           [(int(y), int(x)) for y, x in junctions]


def build_seed_targets(
    skeleton: np.ndarray,
    vessel_mask: np.ndarray,
    sigma: float = 3.0,
    aux_spacing: int = 40,
) -> np.ndarray:
    """Build inverse-thickness weighted heatmap targets.

    Thin-vessel seeds get higher amplitude than thick-vessel seeds so the
    network is forced to attend to capillaries rather than arterioles.
    Auxiliary seeds are placed every aux_spacing pixels along long unbranched
    segments to ensure full tree coverage even without junction detection.

    Args:
        skeleton:    (H, W) skeletonised vessel mask (binary)
        vessel_mask: (H, W) full vessel segmentation mask (binary)
        sigma:       Gaussian blob radius in pixels
        aux_spacing: minimum spacing for auxiliary seeds on long segments

    Returns:
        heatmap: (H, W) float32 in [0, 1]
    """
    from scipy.ndimage import distance_transform_edt

    H, W = skeleton.shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    # Vessel thickness at every pixel (Euclidean distance to background)
    dist_to_bg = distance_transform_edt(vessel_mask > 0).astype(np.float32)
    dist_to_bg = np.clip(dist_to_bg, 1.0, 15.0)

    endpoints, junctions = detect_endpoints_and_junctions(skeleton)
    primary_seeds = endpoints + junctions

    # Auxiliary seeds on long unbranched segments
    aux_seeds = _generate_auxiliary_seeds(skeleton, primary_seeds, aux_spacing)
    all_seeds = primary_seeds + aux_seeds

    for y, x in all_seeds:
        thickness = dist_to_bg[y, x]
        # Thin vessels (thickness≈1) → amplitude≈0.77
        # Thick vessels (thickness≈15) → amplitude≈0.17
        amplitude = 1.0 / (1.0 + 0.3 * thickness)

        y0, y1 = max(0, y - 3 * int(sigma)), min(H, y + 3 * int(sigma) + 1)
        x0, x1 = max(0, x - 3 * int(sigma)), min(W, x + 3 * int(sigma) + 1)

        yy = np.arange(y0, y1).reshape(-1, 1) - y
        xx = np.arange(x0, x1).reshape(1, -1) - x
        gaussian = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))

        heatmap[y0:y1, x0:x1] = np.maximum(
            heatmap[y0:y1, x0:x1],
            amplitude * gaussian,
        )

    return heatmap


def _generate_auxiliary_seeds(
    skeleton: np.ndarray,
    existing_seeds: List[Tuple[int, int]],
    min_spacing: int = 20,
) -> List[Tuple[int, int]]:
    """Place auxiliary seeds on skeleton segments far from any existing seed.

    Uses a greedy exclusion mask that grows as seeds are placed, so the
    spacing guarantee holds against ALL placed seeds (primary + auxiliary),
    not just primary seeds.  Parallel unbranched branches both receive seeds.

    Walks skeleton in raster order (deterministic, O(N)); for 512×512 images
    this completes in <1 ms.
    """
    H, W = skeleton.shape
    half = min_spacing // 2

    excluded = np.zeros((H, W), dtype=bool)
    for y, x in existing_seeds:
        if 0 <= y < H and 0 <= x < W:
            excluded[max(0, y - half):y + half + 1,
                     max(0, x - half):x + half + 1] = True

    skel_points = np.argwhere(skeleton > 0)
    if len(skel_points) == 0:
        return []

    aux_seeds = []
    for y, x in skel_points:
        y, x = int(y), int(x)
        if excluded[y, x]:
            continue
        aux_seeds.append((y, x))
        excluded[max(0, y - half):y + half + 1,
                 max(0, x - half):x + half + 1] = True

    return aux_seeds


# ==========================================
# LOSS FUNCTIONS
# ==========================================

def asymmetric_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma_pos: float = 2.0,
    gamma_neg: float = 4.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Asymmetric focal loss for recall-focused seed detection.

    gamma_neg > gamma_pos: the loss on easy background pixels vanishes
    faster, forcing the network to spend capacity on hard positives
    (thin-vessel seeds it currently misses).

    alpha > 0.5: further upweights the positive (seed) class.

    Args:
        pred:      (B, 1, H, W) sigmoid output in [0, 1]
        target:    (B, 1, H, W) heatmap targets in [0, 1]
        gamma_pos: focusing exponent for positives (seed pixels)
        gamma_neg: focusing exponent for negatives (background pixels)
        alpha:     class balance weight for positives

    Returns:
        scalar loss
    """
    is_pos = (target >= 0.5).float()
    is_neg = 1.0 - is_pos

    pt_pos = pred
    pt_neg = 1.0 - pred

    # Focal weights
    w_pos = alpha * (1.0 - pt_pos) ** gamma_pos
    w_neg = (1.0 - alpha) * (1.0 - pt_neg) ** gamma_neg

    bce = F.binary_cross_entropy(pred, target, reduction="none")
    loss = (is_pos * w_pos + is_neg * w_neg) * bce

    return loss.mean()


def seed_detector_loss(
    endpoint_pred: torch.Tensor,
    vessel_pred: torch.Tensor,
    endpoint_target: torch.Tensor,
    vessel_target: torch.Tensor,
    gamma_pos: float = 2.0,
    gamma_neg: float = 4.0,
    alpha: float = 0.75,
    vessel_weight: float = 0.3,
    fp_penalty_weight: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined loss for two-head seed detector.

    endpoint loss: asymmetric focal — recall-focused keypoint detection
    vessel loss:   standard BCE     — dense vessel probability for internal
                                      vessel mask (replaces external dependency)
    fp_penalty:    off-vessel endpoint suppression — penalises endpoint
                   predictions at locations where vessel_target is background,
                   directly reducing false-positive off-vessel seeds.

    Args:
        endpoint_pred:     (B, 1, H, W) endpoint heatmap predictions
        vessel_pred:       (B, 1, H, W) vessel probability predictions
        endpoint_target:   (B, 1, H, W) inverse-thickness weighted heatmap targets
        vessel_target:     (B, 1, H, W) binary vessel segmentation targets
        vessel_weight:     weight of vessel BCE loss
        fp_penalty_weight: weight of the off-vessel false-positive penalty

    Returns:
        total_loss, dict of component losses for logging
    """
    focal = asymmetric_focal_loss(
        endpoint_pred, endpoint_target, gamma_pos, gamma_neg, alpha
    )
    vessel_bce = F.binary_cross_entropy(vessel_pred, vessel_target)
    fp_penalty = (endpoint_pred * (1.0 - vessel_target)).mean()
    total = focal + vessel_weight * vessel_bce + fp_penalty_weight * fp_penalty

    return total, {
        "loss/focal":       focal.item(),
        "loss/vessel_bce":  vessel_bce.item(),
        "loss/fp_penalty":  fp_penalty.item(),
        "loss/total":       total.item(),
    }


# ==========================================
# SEED SAMPLING UTILITIES
# ==========================================

def sample_seeds_coverage_aware(
    seeds: List[Tuple[int, int, float]],
    traced_mask: np.ndarray,
    vessel_mask: np.ndarray,
    n_seeds: int = 80,
    min_spacing: int = 15,
) -> List[Tuple[int, int, float]]:
    """Select seeds that maximise expected NEW coverage.

    Scores each candidate by confidence × novelty × vessel membership,
    where novelty = distance from already-traced regions.  Farther from
    traced = more likely to reach unvisited vessel pixels.

    A false positive on a vessel is harmless (agent traces and gets reward);
    a missed seed loses an entire subtree.  The scoring deliberately
    tolerates low-confidence seeds in novel regions.

    Args:
        seeds:       list of (y, x, confidence) from _extract_seeds_nms
        traced_mask: (H, W) binary — pixels already traced this inference run
        vessel_mask: (H, W) binary — vessel probability > threshold
        n_seeds:     maximum seeds to return
        min_spacing: minimum pixel distance between returned seeds

    Returns:
        selected seeds sorted by coverage score (descending)
    """
    from scipy.ndimage import distance_transform_edt

    if traced_mask.any():
        dist_from_traced = distance_transform_edt(~traced_mask).astype(np.float32)
    else:
        dist_from_traced = np.ones_like(traced_mask, dtype=np.float32) * 999.0

    scored = []
    for y, x, conf in seeds:
        novelty = float(dist_from_traced[y, x])
        on_vessel = float(vessel_mask[y, x] > 0)
        # sqrt dampens novelty so high-confidence nearby seeds still win
        score = conf * np.sqrt(max(novelty, 1.0)) * (0.3 + 0.7 * on_vessel)
        scored.append((y, x, conf, score))

    scored.sort(key=lambda s: s[3], reverse=True)

    selected = []
    used = np.zeros_like(traced_mask, dtype=bool)

    for y, x, conf, _score in scored:
        if len(selected) >= n_seeds:
            break
        half = min_spacing
        if used[max(0, y - half):y + half, max(0, x - half):x + half].any():
            continue
        selected.append((y, x, conf))
        used[max(0, y - half):y + half, max(0, x - half):x + half] = True

    return selected


def sample_seeds_stochastic(
    seeds: List[Tuple[int, int, float]],
    n_seeds: int = 32,
    temperature: float = 2.0,
) -> List[Tuple[int, int, float]]:
    """Sample seeds proportional to confidence^(1/temperature).

    Higher temperature → more uniform → more exploration of low-confidence
    thin-vessel seeds during PPO training.  Used only during training;
    inference uses coverage-aware deterministic sampling.

    Args:
        seeds:       list of (y, x, confidence)
        n_seeds:     how many to return
        temperature: softmax temperature (1.0 = proportional to conf,
                     higher = more uniform, lower = more greedy)

    Returns:
        sampled seeds (unordered)
    """
    if len(seeds) <= n_seeds:
        return seeds

    confs = np.array([s[2] for s in seeds], dtype=np.float32)
    logits = np.log(confs + 1e-8) / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    indices = np.random.choice(len(seeds), size=n_seeds, replace=False, p=probs)
    return [seeds[i] for i in indices]


def filter_seeds_by_difficulty(
    seeds: List[Tuple[int, int, float]],
    vessel_mask: np.ndarray,
    difficulty: float,
) -> List[Tuple[int, int, float]]:
    """Curriculum filtering: at low difficulty return only thick-vessel seeds.

    At difficulty 0.3 only seeds on vessels thicker than 70% of the maximum
    vessel radius are returned.  At difficulty 1.0 all seeds are returned.
    Integrates directly with the existing curriculum stages.

    Args:
        seeds:       list of (y, x, confidence)
        vessel_mask: (H, W) binary vessel mask
        difficulty:  float in [0, 1] from curriculum stage

    Returns:
        filtered seed list (falls back to top-5 if everything is filtered)
    """
    from scipy.ndimage import distance_transform_edt

    thickness = distance_transform_edt(vessel_mask > 0).astype(np.float32)
    max_thickness = thickness.max()
    if max_thickness < 1.0:
        return seeds

    thickness_threshold = max_thickness * (1.0 - difficulty)

    filtered = [
        (y, x, conf) for y, x, conf in seeds
        if thickness[y, x] >= thickness_threshold
    ]

    if not filtered:
        # fallback: take top-5 by confidence
        return sorted(seeds, key=lambda s: s[2], reverse=True)[:5]

    return filtered


# ==========================================
# SEED DETECTOR MODEL
# ==========================================

class SeedDetector(nn.Module):
    """UNet-based seed point heatmap predictor.

    Input : (B, 3, H, W)  full RGB fundus image, float32 in [0, 1]
    Output: tuple of
        endpoint_heatmap: (B, 1, H, W) in [0, 1] — sparse peaks at seeds
        vessel_prob:      (B, 1, H, W) in [0, 1] — dense vessel probability

    The vessel_prob head replaces the external vessel_mask dependency in
    detect_seeds, eliminating the chicken-and-egg filtering problem where
    a separate segmentation model could not detect thin vessels that the
    seed detector needed to propose.

    Channel layout (base_ch=16):
        enc0 →  16   enc1 →  32   enc2 →  64   enc3 → 128
        bot  → 256
        up3  → 128   up2  →  64   up1  →  32   up0  →  16
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        seed_cfg = config.get("seed_detector", {})
        self.nms_radius = seed_cfg.get("nms_radius", 8)
        self.confidence_threshold = seed_cfg.get("confidence_threshold", 0.1)
        self.top_k = seed_cfg.get("top_k_seeds", 120)
        self.vessel_gate_threshold  = seed_cfg.get("vessel_gate_threshold", 0.0)
        self.snap_radius            = seed_cfg.get("snap_radius", 5)
        self.use_frangi_supplement  = seed_cfg.get("use_frangi_supplement", False)
        self.frangi_spacing         = seed_cfg.get("frangi_spacing", 20)
        self.seeds_per_skeleton_px  = seed_cfg.get("seeds_per_skeleton_px", 15)
        self.max_adaptive_seeds     = seed_cfg.get("max_adaptive_seeds", 600)
        base_ch = seed_cfg.get("base_ch", 16)

        ch = [base_ch * (2 ** i) for i in range(5)]
        # ch = [16, 32, 64, 128, 256] for base_ch=16

        # Encoder — in_channels=3 for RGB
        self.enc0 = DSConvBlock(3, ch[0])
        self.enc1 = DownBlock(ch[0], ch[1])
        self.enc2 = DownBlock(ch[1], ch[2])
        self.enc3 = DownBlock(ch[2], ch[3])

        # Bottleneck
        self.bot = DownBlock(ch[3], ch[4])

        # Decoder with skip connections
        self.up3 = UpBlock(ch[4], ch[3], ch[3])
        self.up2 = UpBlock(ch[3], ch[2], ch[2])
        self.up1 = UpBlock(ch[2], ch[1], ch[1])
        self.up0 = UpBlock(ch[1], ch[0], ch[0])

        # Head 1: sparse endpoint/junction heatmap (trained with asymmetric focal loss)
        self.endpoint_head = nn.Sequential(
            nn.Conv2d(ch[0], ch[0] // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0] // 2, 1, 1),
        )

        # Head 2: dense vessel probability (trained with BCE, replaces external vessel_mask)
        self.vessel_head = nn.Sequential(
            nn.Conv2d(ch[0], ch[0] // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0] // 2, 1, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            endpoint_heatmap: (B, 1, H, W) in [0, 1]
            vessel_prob:      (B, 1, H, W) in [0, 1]
        """
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        b = self.bot(s3)

        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        endpoint_heatmap = torch.sigmoid(self.endpoint_head(d0))
        vessel_prob = torch.sigmoid(self.vessel_head(d0))

        return endpoint_heatmap, vessel_prob

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect_seeds(
        self,
        image: torch.Tensor,
        obs_half: int = 32,
        return_heatmap: bool = False,
        fov_mask: Optional[torch.Tensor] = None,
        traced_mask: Optional[np.ndarray] = None,
        difficulty: Optional[float] = None,
        training_mode: bool = False,
        stochastic_temperature: float = 2.0,
        n_seeds: int = 80,
    ) -> Tuple[List[List[Tuple[int, int, float]]], Optional[torch.Tensor]]:
        """Run inference and return ranked seed points per image.

        Replaces the original detect_seeds.  Key differences:
        - No external vessel_mask argument: vessel probability comes from
          the internal vessel_head (eliminates chicken-and-egg filtering).
        - Coverage-aware sampling at inference time.
        - Stochastic sampling during PPO training.
        - Optional curriculum difficulty filtering.

        Args:
            image:                (B, 3, H, W) float32 tensor in [0, 1]
            obs_half:             half-size of observation window for border suppression
            return_heatmap:       if True, also return the post-processed heatmap
            fov_mask:             (B, 1, H, W) float32 FOV mask — suppresses FOV rim
            traced_mask:          (H, W) numpy bool — pixels already traced
                                  (used for coverage-aware sampling)
            difficulty:           float in [0, 1] for curriculum filtering,
                                  or None to skip filtering
            training_mode:        if True use stochastic sampling, else coverage-aware
            stochastic_temperature: temperature for training-mode sampling
            n_seeds:              maximum seeds to return per image

        Returns:
            batch_seeds: list of lists of (y, x, confidence) tuples
            heatmap:     endpoint heatmap tensor if return_heatmap else None
        """
        self.eval()

        endpoint_heatmap, vessel_prob = self.forward(image)

        h, w = image.shape[-2], image.shape[-1]
        margin = obs_half + 5

        # --- FOV boundary suppression (eroded_fov stored for Frangi branch) ---
        eroded_fov = None
        if fov_mask is not None:
            kernel_size = 35
            pad = kernel_size // 2
            eroded_fov = -F.max_pool2d(
                -fov_mask, kernel_size=kernel_size, stride=1, padding=pad
            )
            endpoint_heatmap = endpoint_heatmap * eroded_fov

        # --- Rectangular border suppression ---
        endpoint_heatmap[:, :, :margin, :] = 0.0
        endpoint_heatmap[:, :, -margin:, :] = 0.0
        endpoint_heatmap[:, :, :, :margin] = 0.0
        endpoint_heatmap[:, :, :, -margin:] = 0.0

        batch_seeds = []
        for b_idx in range(endpoint_heatmap.shape[0]):
            hmap = endpoint_heatmap[b_idx, 0].cpu().numpy()
            vmap = vessel_prob[b_idx, 0].cpu().numpy()

            # Extract candidates via dual-scale NMS with adaptive threshold
            raw_seeds = self._extract_seeds_nms(hmap, h, w)

            # Hard vessel gate: discard seeds where vessel_prob is below threshold.
            # Only meaningful after vessel head is properly trained.
            if self.vessel_gate_threshold > 0:
                gated = [(y, x, c) for y, x, c in raw_seeds
                         if vmap[y, x] >= self.vessel_gate_threshold]
                raw_seeds = gated if gated else raw_seeds[:3]

            # Skeleton-snap: move each seed to the peak vessel-probability pixel
            # within a small neighbourhood to correct sub-pixel Gaussian diffusion offsets.
            if self.snap_radius > 0:
                snapped = []
                for y, x, c in raw_seeds:
                    y0 = max(0, y - self.snap_radius)
                    y1 = min(h, y + self.snap_radius + 1)
                    x0 = max(0, x - self.snap_radius)
                    x1 = min(w, x + self.snap_radius + 1)
                    patch = vmap[y0:y1, x0:x1]
                    by, bx = np.unravel_index(np.argmax(patch), patch.shape)
                    snapped.append((y0 + by, x0 + bx, c))
                raw_seeds = snapped

            # Frangi supplement: contrast-invariant seeds merged with UNet seeds.
            # Frangi seeds are snapped to vmap before merging so both branches
            # share the same spatial reference frame.
            frangi_skeleton = None
            if self.use_frangi_supplement:
                img_np = image[b_idx].cpu().numpy()
                eroded_fov_np = eroded_fov[b_idx, 0].cpu().numpy() \
                                if eroded_fov is not None else None
                frangi_seeds_raw, frangi_skeleton = self._frangi_supplement(
                    img_np, eroded_fov_np, min_spacing=self.frangi_spacing
                )
                if self.snap_radius > 0 and frangi_seeds_raw:
                    snapped_f = []
                    for fy, fx, fc in frangi_seeds_raw:
                        y0 = max(0, fy - self.snap_radius)
                        y1 = min(h, fy + self.snap_radius + 1)
                        x0 = max(0, fx - self.snap_radius)
                        x1 = min(w, fx + self.snap_radius + 1)
                        patch = vmap[y0:y1, x0:x1]
                        by, bx = np.unravel_index(np.argmax(patch), patch.shape)
                        snapped_f.append((y0 + by, x0 + bx, fc))
                    frangi_seeds_raw = snapped_f

                if frangi_seeds_raw:
                    half_r = float(max(5, self.nms_radius // 2))
                    occ_arr = np.array([(y, x) for y, x, c in raw_seeds], dtype=float) \
                              if raw_seeds else np.empty((0, 2))
                    for fy, fx, fc in frangi_seeds_raw:
                        if len(occ_arr) > 0:
                            dists = np.sqrt(
                                ((occ_arr - np.array([fy, fx])) ** 2).sum(axis=1)
                            )
                            if dists.min() < half_r:
                                continue
                        raw_seeds.append((fy, fx, fc))
                        new_pt = np.array([[fy, fx]], dtype=float)
                        occ_arr = np.vstack([occ_arr, new_pt]) \
                                  if len(occ_arr) > 0 else new_pt

            # Adaptive seed count: scale with Frangi skeleton complexity
            if frangi_skeleton is not None and frangi_skeleton.any():
                skeleton_px = int(frangi_skeleton.sum())
                n_eff = min(
                    max(n_seeds, skeleton_px // self.seeds_per_skeleton_px),
                    self.max_adaptive_seeds,
                )
            else:
                n_eff = n_seeds

            # Curriculum difficulty filtering
            if difficulty is not None:
                vessel_binary = (vmap > 0.5).astype(np.float32)
                raw_seeds = filter_seeds_by_difficulty(
                    raw_seeds, vessel_binary, difficulty
                )

            # Sampling strategy
            if training_mode:
                # Stochastic: explore low-confidence thin-vessel seeds during PPO
                seeds = sample_seeds_stochastic(
                    raw_seeds,
                    n_seeds=n_eff,
                    temperature=stochastic_temperature,
                )
            else:
                # Coverage-aware: maximise novelty × confidence at inference
                vessel_binary = (vmap > 0.5).astype(np.float32)
                _traced = (
                    traced_mask
                    if traced_mask is not None
                    else np.zeros((h, w), dtype=bool)
                )
                seeds = sample_seeds_coverage_aware(
                    raw_seeds,
                    traced_mask=_traced,
                    vessel_mask=vessel_binary,
                    n_seeds=n_eff,
                )

            batch_seeds.append(seeds)

        return batch_seeds, (endpoint_heatmap if return_heatmap else None)

    def _extract_seeds_nms(
        self,
        heatmap: np.ndarray,
        h: int,
        w: int,
    ) -> List[Tuple[int, int, float]]:
        """Dual-scale NMS on heatmap → candidate seed list.

        Coarse pass (nms_radius) captures thick-vessel peaks.
        Fine pass (nms_radius//3) recovers thin-vessel peaks that coarse
        suppression would merge.  Adaptive threshold adjusts to the
        image's own heatmap scale so low-contrast images still fire.
        Falls back to image centre if no peak exceeds the threshold.
        """
        from skimage.feature import peak_local_max

        # Adaptive threshold: 60th-percentile of non-zero values (top 40%)
        nonzero = heatmap[heatmap > 0]
        if len(nonzero) > 100:
            adaptive = float(np.clip(np.percentile(nonzero, 60), 0.05, 0.5))
        else:
            adaptive = self.confidence_threshold
        thresh = min(adaptive, self.confidence_threshold)

        if not (heatmap > thresh).any():
            return [(h // 2, w // 2, 0.5)]

        # Coarse pass — wide suppression for thick vessels
        coarse = peak_local_max(
            heatmap, min_distance=self.nms_radius,
            threshold_abs=thresh, num_peaks=self.top_k,
        )

        # Fine pass — tight suppression to recover thin-vessel peaks
        fine_r = max(3, self.nms_radius // 3)
        fine = peak_local_max(
            heatmap, min_distance=fine_r,
            threshold_abs=thresh, num_peaks=self.top_k * 3,
        )

        # Merge: keep fine peaks not within (nms_radius // 2) of any coarse peak
        half_r = max(5, self.nms_radius // 2)
        coarse_arr = np.array(coarse, dtype=float) if len(coarse) else np.empty((0, 2))
        merged = list(coarse)
        for fy, fx in fine:
            if len(coarse_arr) > 0:
                dists = np.sqrt(((coarse_arr - np.array([fy, fx])) ** 2).sum(axis=1))
                if dists.min() < half_r:
                    continue
            merged.append((fy, fx))

        seeds = [(int(y), int(x), float(heatmap[y, x])) for y, x in merged]
        seeds.sort(key=lambda s: s[2], reverse=True)
        return seeds

    @staticmethod
    def _frangi_supplement(
        image_np: np.ndarray,
        eroded_fov_np: Optional[np.ndarray] = None,
        min_spacing: int = 20,
    ) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
        """Frangi vesselness + skeletonisation → seeds.

        Contrast-invariant supplement to the UNet branch.  Illumination
        correction (background subtraction) is applied before Frangi so
        peripheral vessels are not suppressed by the radial illumination
        gradient common in fundus images.

        Args:
            image_np:      (3, H, W) float32 in [0, 1]
            eroded_fov_np: (H, W) float32 FOV mask, already eroded to the
                           same extent as the UNet branch FOV mask, so rim
                           seeds are consistently suppressed in both branches.
            min_spacing:   minimum pixel distance between auxiliary seeds.

        Returns:
            (seeds, skeleton) — seeds as (y, x, confidence),
            skeleton as (H, W) uint8.
        """
        from skimage import filters as skfilters
        from skimage.filters import threshold_otsu
        from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk
        from scipy.ndimage import gaussian_filter

        # Green channel — best vessel contrast in fundus images
        if image_np.ndim == 3 and image_np.shape[0] == 3:
            green = image_np[1]
        elif image_np.ndim == 3:
            green = image_np[:, :, 1]
        else:
            green = image_np
        green = green.astype(np.float64)

        # Illumination correction: subtract smooth background (sigma=50 px)
        # removes the radial gradient that suppresses peripheral vessel response
        background = gaussian_filter(green, sigma=50)
        green = green - background
        g_min, g_max = green.min(), green.max()
        if g_max > g_min:
            green = (green - g_min) / (g_max - g_min)

        # Multi-scale Frangi (same params as FrangiBaseline in models/frangi.py)
        sigmas = np.linspace(1.0, 3.0, 5)
        vessel_enh = skfilters.frangi(green, sigmas=sigmas, black_ridges=True)
        vessel_enh = gaussian_filter(vessel_enh, sigma=1.0)  # suppress noise
        if vessel_enh.max() > 0:
            vessel_enh = vessel_enh / vessel_enh.max()

        # Apply the same eroded FOV mask used by the UNet branch
        if eroded_fov_np is not None:
            vessel_enh = vessel_enh * (eroded_fov_np > 0).astype(np.float32)

        # Adaptive Otsu threshold on vesselness values
        pos_vals = vessel_enh[vessel_enh > 0]
        try:
            thresh = float(np.clip(threshold_otsu(pos_vals), 0.1, 0.5)) \
                     if len(pos_vals) > 10 else 0.15
        except Exception:
            thresh = 0.15

        binary = remove_small_objects((vessel_enh > thresh).astype(bool), min_size=50)
        binary = binary_closing(binary, disk(1))
        skeleton = skeletonize(binary).astype(np.uint8)

        endpoints, junctions = detect_endpoints_and_junctions(skeleton)
        primary = endpoints + junctions
        aux = _generate_auxiliary_seeds(skeleton, primary, min_spacing=min_spacing)

        H, W = skeleton.shape
        seeds = [
            (y, x, float(vessel_enh[y, x]))
            for y, x in (primary + aux)
            if 0 <= y < H and 0 <= x < W
        ]
        return seeds, skeleton