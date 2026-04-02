# models/unet.py
"""==================
Lightweight Centerline UNet for retinal vessel centerline probability estimation.

Architecture:
  - Depthwise-Separable (3x3 DW -> 1x1 PW) convolutions for parameter efficiency (~0.5M params)
  - ReLU activation and Batch Normalization throughout
  - 4-level encoder/decoder with skip connections
  - Single-channel sigmoid output → centerline probability map

Loss:
  - clDice: Topology-aware loss using a differentiable soft-skeleton proxy
  - Binary Cross-Entropy: Pixel-level supervision (with optional pos_weight for imbalance)
  - Combined: Total = w_bce * BCE + w_cl * (1 - clDice)

Extras:
  - GreedyTracer: Seeded steepest-ascent traversal to extract 1-pixel binary skeletons
  - Patched Inference: Sliding-window prediction with Gaussian window blending to eliminate edge artifacts
==================
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.greedy_tracer import GreedyTracer
from models.unet_blocks import DownBlock, DSConvBlock, UpBlock

# ==========================================
# CENTERLINE UNET
# ==========================================

class CenterlineUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_ch: int = 16):
        super().__init__()
        ch = [base_ch * (2**i) for i in range(5)]

        self.enc0 = DSConvBlock(in_channels, ch[0])
        self.enc1 = DownBlock(ch[0], ch[1])
        self.enc2 = DownBlock(ch[1], ch[2])
        self.enc3 = DownBlock(ch[2], ch[3])
        self.bot = DownBlock(ch[3], ch[4])

        self.up3 = UpBlock(ch[4], ch[3], ch[3])
        self.up2 = UpBlock(ch[3], ch[2], ch[2])
        self.up1 = UpBlock(ch[2], ch[1], ch[1])
        self.up0 = UpBlock(ch[1], ch[0], ch[0])

        self.head = nn.Sequential(
            nn.Conv2d(ch[0], ch[0] // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0] // 2, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bot(s3)
        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        return self.head(d0)

# ==========================================
# SOFT SKELETONISATION & clDice LOSS
# ==========================================

def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))

def soft_skeleton(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    skel = F.relu(img - _soft_open(img))
    for _ in range(num_iter):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel

# ==========================================
# Topology-aware Loss
# ==========================================

class CenterlineLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.4, cl_weight: float = 0.6,
                 skeleton_iter: int = 10, pos_weight: Optional[float] = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.cl_weight = cl_weight
        self.skeleton_iter = skeleton_iter
        if pos_weight is not None:
            self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight]))
        else:
            self.pos_weight_tensor = None

    def _soft_cl_dice(self, pred_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        skel_pred = soft_skeleton(pred_probs, self.skeleton_iter)
        skel_target = soft_skeleton(target, self.skeleton_iter)

        tprec = (skel_pred * target).sum(dim=[1, 2, 3]) / (skel_pred.sum(dim=[1, 2, 3]) + 1e-6)
        tsens = (skel_target * pred_probs).sum(dim=[1, 2, 3]) / (skel_target.sum(dim=[1, 2, 3]) + 1e-6)

        cl_dice = (2 * tprec * tsens) / (tprec + tsens + 1e-6)
        return cl_dice.mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        if mask is not None:
            bce = F.binary_cross_entropy_with_logits(
                pred_logits, target,
                pos_weight=self.pos_weight_tensor,
                reduction='none'
            )
            bce = (bce * mask).sum() / (mask.sum() + 1e-7)
        else:
            bce = F.binary_cross_entropy_with_logits(
                pred_logits, target, pos_weight=self.pos_weight_tensor, reduction='mean'
            )

        probs = torch.sigmoid(pred_logits)
        cl_d = self._soft_cl_dice(probs, target)

        total = (self.bce_weight * bce) + (self.cl_weight * (1.0 - cl_d))

        if torch.isnan(total):
            total = bce

        return total, {"bce": bce.item(), "cl_dice": cl_d.item(), "total": total.item()}

# unet.py

# ... [CenterlineUNet, soft_skeleton, and CenterlineLoss classes remain the same] ...

# ==========================================
# PREDICTOR
# ==========================================

class CenterlinePredictor:
    def __init__(self, model: CenterlineUNet, tracer: Optional[GreedyTracer] = None,
                 device: str = "cpu", patch_size: Optional[int] = None, patch_stride: Optional[int] = None):
        self.model = model.to(device).eval()
        self.tracer = tracer or GreedyTracer()
        self.device = device
        self.patch_size = patch_size
        self.patch_stride = patch_stride or (patch_size // 2 if patch_size else None)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu", **kwargs) -> "CenterlinePredictor":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("model_cfg", {"in_channels": 1, "base_ch": 16})
        model = CenterlineUNet(**cfg)
        model.load_state_dict(ckpt["model_state"])
        return cls(model, device=device, **kwargs)

    @torch.no_grad()
    def _infer_full(self, img_t: torch.Tensor) -> torch.Tensor:
        logits = self.model(img_t.unsqueeze(0).to(self.device))[0, 0]
        return torch.sigmoid(logits).cpu()

    @torch.no_grad()
    def _infer_patched(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        ps, st = self.patch_size, self.patch_stride
        prob = torch.zeros(H, W)
        count = torch.zeros(H, W)
        lin = torch.linspace(-1, 1, ps)
        gauss = torch.exp(-2 * (lin**2))
        win = gauss[:, None] * gauss[None, :]

        for y in range(0, H - ps + 1, st):
            for x in range(0, W - ps + 1, st):
                patch = img_t[:, y : y + ps, x : x + ps].unsqueeze(0).to(self.device)
                out = torch.sigmoid(self.model(patch)[0, 0]).cpu()
                prob[y : y + ps, x : x + ps] += out * win
                count[y : y + ps, x : x + ps] += win
        return prob / (count + 1e-8)

    def predict(self, image: np.ndarray, fov_mask: Optional[np.ndarray] = None):
        import cv2
        from skimage import morphology

        # 1. CNN Inference
        img_t = torch.from_numpy(image).float().unsqueeze(0)
        prob = self._infer_patched(img_t) if self.patch_size else self._infer_full(img_t)
        prob_np = prob.numpy().astype(np.float32)

        # 2. Gaussian "Guide Rail" Construction
        # Instead of distanceTransform, we smooth the U-Net probability map.
        # This creates a 'ramp' that guides the greedy tracer to the ridge.
        vesselness = cv2.GaussianBlur(prob_np, (0, 0), sigmaX=1.5)

        # 3. FOV Masking
        if fov_mask is not None:
            # Normalise mask to binary 0/1 for math
            mask_bin = (fov_mask > 0).astype(np.uint8)
            # Erode slightly to prevent tracing the edge of the FOV circle
            kernel = np.ones((5, 5), np.uint8)
            eroded_mask = cv2.erode(mask_bin, kernel, iterations=2)
            
            vesselness *= eroded_mask.astype(np.float32)
            # Re-assign fov_mask as 0/255 for the Tracer's internal logic
            fov_mask_for_tracer = (eroded_mask * 255).astype(np.uint8)
        else:
            fov_mask_for_tracer = None

        # 4. Greedy Tracing
        # Ensure the tracer passed into this class has seed_thresh > step_thresh!
        skeleton, _ = self.tracer.trace(vesselness, fov_mask=fov_mask_for_tracer)

        # 5. Small Object Removal (using connectivity=2 for diagonal skeletons)
        binary_skel = (skeleton > 0)
        if binary_skel.any():
            clean_skel = morphology.remove_small_objects(binary_skel, min_size=15, connectivity=2)
            skeleton = (clean_skel.astype(np.uint8) * 255)
        else:
            skeleton = np.zeros_like(prob_np, dtype=np.uint8)

        return prob_np, skeleton
