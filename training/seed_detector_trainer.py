# training/seed_detector_trainer.py
"""Seed detector training logic

Provides:
    create_seed_heatmap()   — GT heatmap with Gaussians at endpoints + junctions
    SeedDetectorTrainer     — full train loop, validation, checkpoint saving

Used by:
    scripts/train_seed_detector.py  (DRIVE)
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ==========================================
# GT HEATMAP GENERATION
# ==========================================
def create_seed_heatmap(
    centerline: np.ndarray, sigma: float = 3.0, n_seeds: int = 50
) -> np.ndarray:
    """Build ground-truth seed heatmap using coverage-based farthest-point sampling.

    Instead of placing blobs at every endpoint and junction (which produces
    500+ densely clustered targets), this selects ~n_seeds points that are
    maximally spread along the vessel tree. The seed detector then learns
    to predict good starting points for tracing, not topological features.

    Args:
        centerline: binary centerline mask (H, W), float32
        sigma:      Gaussian blob radius in pixels
        n_seeds:    number of seed points to place

    Returns:
        heatmap (H, W), float32, normalised to [0, 1]
    """
    from scipy.ndimage import gaussian_filter

    points = np.argwhere(centerline > 0)
    if len(points) == 0:
        return np.zeros_like(centerline, dtype=np.float32)

    # Farthest-point sampling for maximum coverage
    n_seeds = min(n_seeds, len(points))
    h, w = centerline.shape

    # First seed: closest to image center
    center = np.array([h / 2, w / 2])
    dists_to_center = np.linalg.norm(points - center, axis=1)
    seeds = [points[np.argmin(dists_to_center)]]

    # Remaining seeds: each maximally far from all existing seeds
    min_dists = np.full(len(points), np.inf)
    for _ in range(n_seeds - 1):
        # Update minimum distance to nearest existing seed
        last_seed = seeds[-1]
        d = np.linalg.norm(points - last_seed, axis=1)
        min_dists = np.minimum(min_dists, d)

        seeds.append(points[np.argmax(min_dists)])

    # Place Gaussian blobs at each seed
    heatmap = np.zeros_like(centerline, dtype=np.float32)
    for y, x in seeds:
        if 0 <= y < h and 0 <= x < w:
            heatmap[y, x] = 1.0

    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


# def create_seed_heatmap(centerline: np.ndarray, sigma: float = 3.0) -> np.ndarray:
#     """Build ground-truth seed heatmap for one image.
#     Places a Gaussian blob (sigma=3px) at every endpoint and junction
#     of the GT centerline skeleton.

#     Args:
#         centerline: binary centerline mask (H, W), float32
#         sigma:      Gaussian sigma in pixels

#     Returns:
#         heatmap (H, W), float32, normalised to [0, 1]

#     """
#     from scipy.ndimage import gaussian_filter

#     from data.centerline_extraction import CenterlineExtractor

#     extractor = CenterlineExtractor()
#     endpoints = extractor._find_endpoints(centerline)
#     junctions = extractor._find_junctions(centerline)

#     heatmap = np.zeros_like(centerline, dtype=np.float32)
#     for y, x in endpoints + junctions:
#         if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
#             heatmap[y, x] = 1.0

#     heatmap = gaussian_filter(heatmap, sigma=sigma)
#     if heatmap.max() > 0:
#         heatmap /= heatmap.max()
#     return heatmap


# ==========================================
# DATASET
# ==========================================
class SeedDataset(Dataset):
    """Each item: (image_tensor, gt_heatmap_tensor, fov_mask_tensor)
    image      : (3, H, W)  float32
    gt_heatmap : (1, H, W)  float32
    fov_mask   : (1, H, W)  float32
    """

    def __init__(
        self, samples: List[Dict], sigma: float = 3.0, resize: tuple = (512, 512)
    ):
        self.items = []
        for s in samples:
            gt_hm = create_seed_heatmap(s["centerline"], sigma=sigma)
            img_t = torch.from_numpy(s["image"].transpose(2, 0, 1)).float()
            hm_t = torch.from_numpy(gt_hm).unsqueeze(0).float()
            fov_t = torch.from_numpy(s["fov_mask"]).unsqueeze(0).float()

            if resize is not None:
                h, w = resize
                img_t = F.interpolate(
                    img_t.unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                hm_t = F.interpolate(
                    hm_t.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
                ).squeeze(0)
                fov_t = F.interpolate(
                    fov_t.unsqueeze(0), size=(h, w), mode="nearest"
                ).squeeze(0)

            self.items.append((img_t, hm_t, fov_t))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ==========================================
# LOSS
# ==========================================


def focal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for highly imbalanced heatmap regression.
    Downweights easy background pixels so the network focuses on seed regions.

    pred, gt, mask : (B, 1, H, W)
    """
    bce = F.binary_cross_entropy(pred, gt, reduction="none")
    pt = torch.where(gt > 0.5, pred, 1.0 - pred)
    loss = alpha * (1.0 - pt) ** gamma * bce

    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


# ==========================================
# TRAINER
# ==========================================


class SeedDetectorTrainer:
    """Full training loop for SeedDetector.

    Usage:
        trainer = SeedDetectorTrainer(model, device, lr=1e-4, num_epochs=30)
        trainer.train(train_samples, val_samples, save_path, config)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-4,
        batch_size: int = 4,
        num_epochs: int = 30,
        sigma: float = 3.0,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sigma = sigma

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

    def train(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        save_path: str,
        config: dict,
    ) -> None:
        """Run full training loop and save best checkpoint.

        Args:
            train_samples: list of sample dicts (image, centerline, fov_mask, ...)
            val_samples:   list of sample dicts for validation
            save_path:     where to save best model (.pt)
            config:        full CONFIG dict stored in checkpoint

        """
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        train_loader = DataLoader(
            SeedDataset(train_samples, sigma=self.sigma),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            SeedDataset(val_samples, sigma=self.sigma),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

        best_val_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch:3d}/{self.num_epochs}  "
                f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": config,
                    },
                    save_path,
                )
                print(f"  ✓ Saved best model (val_loss={val_loss:.5f})")

        print(f"\nDone. Best val_loss={best_val_loss:.5f}  →  {save_path}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        """Run one epoch. Returns mean loss."""
        self.model.train() if train else self.model.eval()
        total_loss = 0.0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for img_batch, hm_batch, fov_batch in loader:
                img_batch = img_batch.to(self.device)
                hm_batch = hm_batch.to(self.device)
                fov_batch = fov_batch.to(self.device)

                pred = self.model(img_batch)
                loss = focal_loss(pred, hm_batch, mask=fov_batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)
