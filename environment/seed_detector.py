# models/seed_detector.py
"""Seed detector — predicts a sparse heatmap of vessel endpoints and junctions.

Architecture: CenterlineUNet (same as baseline, see centerline_unet_baseline.py)
  - Depthwise-separable convolutions (~0.5M params)
  - 4-level encoder/decoder with skip connections  ← key upgrade vs ResNet+plain decoder
  - in_channels=3 (RGB with enhanced green)
  - Sigmoid output → heatmap in [0, 1]

Difference from centerline baseline:
  - Input: 3-channel RGB (not 1-channel greyscale)
  - GT targets: sparse Gaussian blobs at endpoints+junctions only (not full vessel mask)
  - Loss: focal loss (not BCE+clDice) — clDice is for connected segments, not sparse keypoints

Provides:
    SeedDetector  — UNet → (B,1,H,W) heatmap
                    call .detect_seeds() to get ranked (y, x, confidence) lists

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
# SEED DETECTOR  (UNet backbone)
# ==========================================


class SeedDetector(nn.Module):
    """UNet-based seed point heatmap predictor.

    Input : (B, 3, H, W)  full RGB fundus image, float32 in [0, 1]
    Output: (B, 1, H, W)  heatmap in [0, 1], peaks = endpoints / junctions

    Architecture is identical to CenterlineUNet(in_channels=3, base_ch=16)
    from centerline_unet_baseline.py. Skip connections give the decoder
    access to full-resolution spatial features at every level — critical
    for localising sparse keypoints precisely on thin vessels.

    Channel layout (base_ch=16):
        enc0 →  16   enc1 →  32   enc2 →  64   enc3 → 128
        bot  → 256
        up3  → 128   up2  →  64   up1  →  32   up0  →  16
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        seed_cfg = config.get("seed_detector", {})
        self.nms_radius = seed_cfg.get("nms_radius", 10)
        self.confidence_threshold = seed_cfg.get("confidence_threshold", 0.3)
        self.top_k = seed_cfg.get("top_k_seeds", 50)
        base_ch = seed_cfg.get("base_ch", 16)

        ch = [base_ch * (2**i) for i in range(5)]
        # ch = [16, 32, 64, 128, 256] for base_ch=16

        # Encoder — same as CenterlineUNet, in_channels=3 for RGB
        self.enc0 = DSConvBlock(3, ch[0])
        self.enc1 = DownBlock(ch[0], ch[1])
        self.enc2 = DownBlock(ch[1], ch[2])
        self.enc3 = DownBlock(ch[2], ch[3])

        # Bottleneck
        self.bot = DownBlock(ch[3], ch[4])

        # Decoder with skip connections — same as CenterlineUNet
        self.up3 = UpBlock(ch[4], ch[3], ch[3])  # 256 + 128 → 128
        self.up2 = UpBlock(ch[3], ch[2], ch[2])  # 128 +  64 →  64
        self.up1 = UpBlock(ch[2], ch[1], ch[1])  #  64 +  32 →  32
        self.up0 = UpBlock(ch[1], ch[0], ch[0])  #  32 +  16 →  16

        # Head — same as CenterlineUNet
        self.head = nn.Sequential(
            nn.Conv2d(ch[0], ch[0] // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0] // 2, 1, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Same initialisation as CenterlineUNet."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — save skip tensors
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        # Bottleneck
        b = self.bot(s3)

        # Decoder — skip connections from encoder
        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        return torch.sigmoid(self.head(d0))  # (B, 1, H, W) in [0, 1]

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def detect_seeds(
        self,
        image: torch.Tensor,
        obs_half: int = 32,
        return_heatmap: bool = False,
        fov_mask: Optional[torch.Tensor] = None,
        vessel_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[List[Tuple[int, int, float]]], Optional[torch.Tensor]]:
        """Run inference and return ranked seed points per image,
        masking out the FOV border.

        Args:
            vessel_mask: (H, W) numpy uint8 — seeds outside vessel regions
                         are discarded post-NMS; heatmap is also suppressed
                         in non-vessel regions before NMS.
        """
        self.eval()
        with torch.no_grad():
            heatmap = self.forward(image)

        h, w = image.shape[-2], image.shape[-1]
        margin = obs_half + 5

        # 1. Silence the circular FOV boundary
        if fov_mask is not None:
            kernel_size = 35
            pad = kernel_size // 2
            eroded_fov = -F.max_pool2d(
                -fov_mask, kernel_size=kernel_size, stride=1, padding=pad
            )
            heatmap = heatmap * eroded_fov

        # 2. Silence the outer rectangular margin
        heatmap[:, :, :margin, :] = 0
        heatmap[:, :, -margin:, :] = 0
        heatmap[:, :, :, :margin] = 0
        heatmap[:, :, :, -margin:] = 0

        # 3. Suppress non-vessel regions in the heatmap before NMS
        if vessel_mask is not None:
            from scipy.ndimage import binary_dilation
            vessel_dilated = binary_dilation(vessel_mask > 0, iterations=3)
            vessel_t = torch.from_numpy(
                vessel_dilated.astype(np.float32)
            ).to(heatmap.device)
            heatmap = heatmap * vessel_t.unsqueeze(0).unsqueeze(0)

        batch_seeds = []
        for b in range(heatmap.shape[0]):
            hmap = heatmap[b, 0].cpu().numpy()
            seeds = self._extract_seeds_nms(hmap, h, w, vessel_mask=vessel_mask)
            batch_seeds.append(seeds)

        return batch_seeds, (heatmap if return_heatmap else None)

    def _extract_seeds_nms(
        self,
        heatmap: np.ndarray,
        h: int,
        w: int,
        vessel_mask: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int, float]]:
        """Two-pass NMS: coarse pass for thick/central vessels, fine pass for thin.

        A single NMS radius of 15px suppresses all peaks within a 30px diameter
        area, causing thin peripheral vessel endpoints to lose out to the nearest
        thick vessel peak. Two-pass NMS avoids this:

        Pass 1 (coarse): large radius, high threshold → thick/central vessels
        Pass 2 (fine):   small radius, lower threshold → thin/peripheral vessels,
                         with fine seeds within `coarse_radius` of any coarse seed
                         suppressed to avoid duplicates.
        """
        from skimage.feature import peak_local_max

        if not (heatmap > self.confidence_threshold).any():
            return [(h // 2, w // 2, 0.5)]

        half_k = max(self.top_k // 2, 1)
        coarse_radius = self.nms_radius          # e.g. 15 at inference
        fine_radius   = max(coarse_radius // 3, 3)  # e.g. 5 at inference
        coarse_thresh = max(self.confidence_threshold * 2.0, 0.1)
        fine_thresh   = self.confidence_threshold

        # Pass 1 — coarse
        coords_coarse = peak_local_max(
            heatmap,
            min_distance=coarse_radius,
            threshold_abs=coarse_thresh,
            num_peaks=half_k,
        )

        # Pass 2 — fine (suppressed near any coarse seed)
        coords_fine = peak_local_max(
            heatmap,
            min_distance=fine_radius,
            threshold_abs=fine_thresh,
            num_peaks=self.top_k,
        )

        # Convert coarse to set for fast proximity check
        coarse_set = [(int(y), int(x)) for y, x in coords_coarse]

        seeds = []
        seen_fine = set()
        for y, x in coords_fine:
            iy, ix = int(y), int(x)
            # Suppress fine seeds too close to any coarse seed
            too_close = any(
                abs(iy - cy) + abs(ix - cx) < coarse_radius
                for cy, cx in coarse_set
            )
            if not too_close and (iy, ix) not in seen_fine:
                seeds.append((iy, ix, float(heatmap[iy, ix])))
                seen_fine.add((iy, ix))

        # Add coarse seeds
        for y, x in coords_coarse:
            seeds.append((int(y), int(x), float(heatmap[int(y), int(x)])))

        # Post-NMS vessel mask filter
        if vessel_mask is not None:
            seeds = [
                (y, x, conf) for y, x, conf in seeds if vessel_mask[y, x] > 0
            ]
            if not seeds:
                return [(h // 2, w // 2, 0.5)]

        seeds.sort(key=lambda s: s[2], reverse=True)
        return seeds[: self.top_k]
