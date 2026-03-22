# seed_detector.py
"""
Seed detector — predicts a sparse heatmap of vessel endpoints and junctions.

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
    ) -> Tuple[List[List[Tuple[int, int, float]]], Optional[torch.Tensor]]:
        """Run inference and return ranked seed points per image,
        masking out the FOV border.
        """
        self.eval()
        with torch.no_grad():
            heatmap = self.forward(image)

        h, w = image.shape[-2], image.shape[-1]
        margin = obs_half + 5

        # 1. Silence the circular FOV boundary using the mask
        if fov_mask is not None:
            # Erode the mask to remove the bright edge rim
            kernel_size = 35
            pad = kernel_size // 2
            # Max pooling on the negative mask acts as a quick morphological erosion in PyTorch
            eroded_fov = -F.max_pool2d(
                -fov_mask, kernel_size=kernel_size, stride=1, padding=pad
            )
            heatmap = heatmap * eroded_fov

        # 2. Silence the outer rectangular margin completely so we don't pick up seeds there
        heatmap[:, :, :margin, :] = 0
        heatmap[:, :, -margin:, :] = 0
        heatmap[:, :, :, :margin] = 0
        heatmap[:, :, :, -margin:] = 0

        batch_seeds = []
        for b in range(heatmap.shape[0]):
            hmap = heatmap[b, 0].cpu().numpy()
            seeds = self._extract_seeds_nms(hmap, h, w)
            batch_seeds.append(seeds)

        return batch_seeds, (heatmap if return_heatmap else None)

    def _extract_seeds_nms(
        self,
        heatmap: np.ndarray,
        h: int,
        w: int,
    ) -> List[Tuple[int, int, float]]:
        """Non-maximum suppression on heatmap → top-k seed list.
        """
        from skimage.feature import peak_local_max

        if not (heatmap > self.confidence_threshold).any():
            return [(h // 2, w // 2, 0.5)]

        coords = peak_local_max(
            heatmap,
            min_distance=self.nms_radius,
            threshold_abs=self.confidence_threshold,
            num_peaks=self.top_k,
        )

        seeds = []
        for y, x in coords:
            seeds.append((int(y), int(x), float(heatmap[y, x])))

        seeds.sort(key=lambda s: s[2], reverse=True)
        return seeds[: self.top_k]
