# centerline_unet_baseline.py
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

from baselines.greedy_tracer_baseline import GreedyTracer
from baselines.unet_blocks import DownBlock, DSConvBlock, UpBlock

# ==========================================
# CENTERLINE UNET
# ==========================================


class CenterlineUNet(nn.Module):
    """Lightweight UNet for centerline probability estimation.

    Input:  (B, in_channels, H, W)  - float32, normalised 0-1
    Output: (B, 1, H, W)            - sigmoid probability map

    Default channel widths keep the model ~0.5 M parameters.

    Channel layout (base_ch=16):
        enc0 →  16   enc1 →  32   enc2 →  64   enc3 → 128
        bot  → 256
        up3  → 128   up2  →  64   up1  →  32   up0  →  16
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 16,
    ):
        super().__init__()

        ch = [base_ch * (2**i) for i in range(5)]
        # ch = [16, 32, 64, 128, 256] for base_ch=16

        # Encoder
        self.enc0 = DSConvBlock(in_channels, ch[0])  # (B, 16,  H,    W)
        self.enc1 = DownBlock(ch[0], ch[1])  # (B, 32,  H/2,  W/2)
        self.enc2 = DownBlock(ch[1], ch[2])  # (B, 64,  H/4,  W/4)
        self.enc3 = DownBlock(ch[2], ch[3])  # (B, 128, H/8,  W/8)

        # Bottleneck
        self.bot = DownBlock(ch[3], ch[4])  # (B, 256, H/16, W/16)

        # Decoder channels are now explicit and index-driven
        #   UpBlock(in_from_below, skip_from_encoder, out)
        self.up3 = UpBlock(ch[4], ch[3], ch[3])  # 256 + 128 → 128
        self.up2 = UpBlock(ch[3], ch[2], ch[2])  # 128 +  64 →  64
        self.up1 = UpBlock(ch[2], ch[1], ch[1])  #  64 +  32 →  32
        self.up0 = UpBlock(ch[1], ch[0], ch[0])  #  32 +  16 →  16

        # Head
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        # Bottleneck
        b = self.bot(s3)

        # Decoder
        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        return torch.sigmoid(self.head(d0))


# ==========================================
# SOFT SKELETONISATION & clDice LOSS
# ==========================================


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Morphological min-pool (2-connectivity)."""
    if img.ndim == 4:  # (B, 1, H, W)
        return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
    raise ValueError("Expected 4-D tensor.")


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeleton(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    """Differentiable skeleton approximation via iterative soft-erosion.
    Reference: Shit et al., "clDice - a Novel Topology-Preserving Loss Function
               for Tubular Structure Segmentation", CVPR 2021.
    """
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
    """Combined loss:
        L = w_bce * BCE + w_cl * (1 - soft_clDice)

    soft_clDice uses a differentiable skeleton proxy so gradients
    flow back into the network through both terms.

    Args:
        bce_weight   : weight for BCE term
        cl_weight    : weight for clDice term
        skeleton_iter: soft-skeleton iterations (more → sharper, slower)
        pos_weight   : optional positive-class weight for BCE (handles imbalance)

    """

    def __init__(
        self,
        bce_weight: float = 0.4,
        cl_weight: float = 0.6,
        skeleton_iter: int = 10,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.cl_weight = cl_weight
        self.skeleton_iter = skeleton_iter
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None

    def _soft_cl_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft clDice ∈ [0, 1], higher is better.
        pred, target: (B, 1, H, W) float, [0, 1]
        """
        skel_pred = soft_skeleton(pred, self.skeleton_iter)
        skel_target = soft_skeleton(target, self.skeleton_iter)

        # Topology-Precision: how much of pred-skeleton lies on target
        tprec = (skel_pred * target).sum(dim=[1, 2, 3]) / (
            skel_pred.sum(dim=[1, 2, 3]) + 1e-5
        )
        # Topology-Sensitivity: how much of gt-skeleton is covered by pred
        tsens = (skel_target * pred).sum(dim=[1, 2, 3]) / (
            skel_target.sum(dim=[1, 2, 3]) + 1e-5
        )

        cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-5)
        return cl_dice.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Args:
            pred   : (B, 1, H, W) sigmoid output
            target : (B, 1, H, W) binary GT centerline, float
            mask   : (B, 1, H, W) optional FOV mask - loss only inside ROI

        Returns:
            total_loss, {'bce': ..., 'cl_dice': ..., 'total': ...}

        """
        if mask is not None:
            p = pred[mask > 0]
            t = target[mask > 0]
        else:
            p = pred.reshape(-1)
            t = target.reshape(-1)

        # Single clean BCE path — no silent overwrite
        pw = self.pos_weight.to(pred.device) if self.pos_weight is not None else None
        if pw is not None:
            # Weighted BCE: penalises false negatives on rare centerline pixels
            bce = -(
                pw * t * torch.log(p + 1e-5) + (1 - t) * torch.log(1 - p + 1e-5)
            ).mean()
        else:
            bce = F.binary_cross_entropy(p, t, reduction="mean")

        cl_d = self._soft_cl_dice(pred, target)
        total = self.bce_weight * bce + self.cl_weight * (1.0 - cl_d)

        return total, {
            "bce": bce.item(),
            "cl_dice": cl_d.item(),
            "total": total.item(),
        }


# ==========================================
# FULL PIPELINE
# ==========================================


class CenterlinePredictor:
    """Wraps model + tracer for end-to-end inference.

    Usage:
        predictor = CenterlinePredictor.from_checkpoint('weights.pt')
        skeleton  = predictor.predict(image_np, fov_mask_np)
    """

    def __init__(
        self,
        model: CenterlineUNet,
        tracer: Optional[GreedyTracer] = None,
        device: str = "cpu",
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None,
    ):
        self.model = model.to(device).eval()
        self.tracer = tracer or GreedyTracer()
        self.device = device
        self.patch_size = patch_size
        self.patch_stride = patch_stride or (patch_size // 2 if patch_size else None)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "CenterlinePredictor":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("model_cfg", {})
        model = CenterlineUNet(**cfg)
        model.load_state_dict(ckpt["model_state"])
        return cls(model, device=device, **kwargs)

    @torch.no_grad()
    def _infer_full(self, img_t: torch.Tensor) -> torch.Tensor:
        return self.model(img_t.unsqueeze(0).to(self.device))[0, 0].cpu()

    @torch.no_grad()
    def _infer_patched(self, img_t: torch.Tensor) -> torch.Tensor:
        """Sliding-window inference with Gaussian blend weights."""
        C, H, W = img_t.shape
        ps = self.patch_size
        st = self.patch_stride

        prob = torch.zeros(H, W)
        count = torch.zeros(H, W)

        # Gaussian weight window
        lin = torch.linspace(-1, 1, ps)
        gauss = torch.exp(-2 * (lin**2))
        win = gauss[:, None] * gauss[None, :]

        ys = list(range(0, H - ps + 1, st)) + [max(0, H - ps)]
        xs = list(range(0, W - ps + 1, st)) + [max(0, W - ps)]

        for y in set(ys):
            for x in set(xs):
                patch = img_t[:, y : y + ps, x : x + ps].unsqueeze(0).to(self.device)
                out = self.model(patch)[0, 0].cpu()
                prob[y : y + ps, x : x + ps] += out * win
                count[y : y + ps, x : x + ps] += win

        return prob / (count + 1e-8)

    def predict(
        self,
        image: np.ndarray,  # (H, W) float32 pre-processed
        fov_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns:
        prob_map : (H, W) float32  raw probability
        skeleton : (H, W) uint8    binarised centerline

        """
        img_t = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)

        if self.patch_size is not None:
            prob = self._infer_patched(img_t)
        else:
            prob = self._infer_full(img_t)

        prob_np = prob.numpy()
        skeleton, _ = self.tracer.trace(prob_np, fov_mask)
        return prob_np, skeleton


# ==========================================
# Sanity check
# ==========================================

if __name__ == "__main__":
    print("=== CenterlineUNet Sanity Check ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model ──
    model = CenterlineUNet(in_channels=1, base_ch=16).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total:,}  (~{total/1e6:.2f}M)")

    # ── Forward pass ──
    x = torch.rand(2, 1, 512, 512, device=device)
    target = torch.zeros(2, 1, 512, 512, device=device)
    target[:, :, 100:400, 254:258] = 1.0  # thin vertical line

    pred = model(x)
    print(f"Input      : {tuple(x.shape)}  →  Output: {tuple(pred.shape)}")
    print(f"Pred range : [{pred.min():.3f}, {pred.max():.3f}]")

    # ── Loss ──
    criterion = CenterlineLoss(bce_weight=0.4, cl_weight=0.6, pos_weight=10.0)
    loss, breakdown = criterion(pred, target)
    print(f"Loss       : {loss.item():.4f}  |  {breakdown}")

    # ── Greedy Tracer ──
    tracer = GreedyTracer(seed_thresh=0.5, step_thresh=0.3, min_length=5)
    prob_np = pred[0, 0].detach().cpu().numpy()
    skeleton, _ = tracer.trace(prob_np)
    print(f"Skeleton   : {skeleton.shape}, nonzero pixels: {skeleton.sum() // 255}")

    print("=== All OK ===")
