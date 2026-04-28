# training/seed_detector_trainer.py
"""Seed detector training logic with Recall-Focused Focal Loss.

- Seed Head: Focal Loss configured via config.py (targeting high FN).
- Vessel Head: Focal Loss + Soft Dice for structural connectivity.
- Visualization: Generates Epoch Error Maps and a 2-panel Training Summary.
"""

from typing import Dict, List, Optional
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist

from models.seed_detector import build_seed_targets
from config import SEED_CONFIG


# ==========================================
# GT HEATMAP GENERATION
# ==========================================
def create_seed_heatmap(centerline: np.ndarray, sigma: float = 3.0, n_seeds: int = 50) -> np.ndarray:
    """Build ground-truth seed heatmap using farthest-point sampling for coverage."""
    points = np.argwhere(centerline > 0)
    if len(points) == 0:
        return np.zeros_like(centerline, dtype=np.float32)

    n_seeds = min(n_seeds, len(points))
    h, w = centerline.shape
    center = np.array([h / 2, w / 2])
    dists_to_center = np.linalg.norm(points - center, axis=1)
    seeds = [points[np.argmin(dists_to_center)]]

    min_dists = np.full(len(points), np.inf)
    for _ in range(n_seeds - 1):
        last_seed = seeds[-1]
        d = np.linalg.norm(points - last_seed, axis=1)
        min_dists = np.minimum(min_dists, d)
        seeds.append(points[np.argmax(min_dists)])

    heatmap = np.zeros_like(centerline, dtype=np.float32)
    for y, x in seeds:
        if 0 <= y < h and 0 <= x < w:
            heatmap[y, x] = 1.0

    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

# ==========================================
# DATASET
# ==========================================
class SeedDataset(Dataset):
    def __init__(self, samples: List[Dict], sigma: float = 3.0, resize: tuple = (512, 512), aux_spacing: int = 20):
        self.items = []
        for s in samples:
            vessel_mask = s.get("vessel_mask", s["centerline"])
            gt_hm = build_seed_targets(s["centerline"], vessel_mask, sigma=sigma, aux_spacing=aux_spacing)

            img_t = torch.from_numpy(s["image"].transpose(2, 0, 1)).float()
            hm_t = torch.from_numpy(gt_hm).unsqueeze(0).float()
            fov_t = torch.from_numpy(s["fov_mask"]).unsqueeze(0).float()
            vessel_t = torch.from_numpy(vessel_mask.astype(np.float32)).unsqueeze(0).float()

            if resize is not None:
                h, w = resize
                img_t = F.interpolate(img_t.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
                hm_t = F.interpolate(hm_t.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
                fov_t = F.interpolate(fov_t.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
                vessel_t = F.interpolate(vessel_t.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)

            self.items.append((img_t, hm_t, fov_t, vessel_t))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


# ==========================================
# LOSS FUNCTIONS
# ==========================================

def focal_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal Loss to handle extreme class imbalance."""
    bce = F.binary_cross_entropy(pred, gt, reduction="none")
    pt = torch.where(gt > 0.5, pred, 1.0 - pred)
    loss = alpha * (1.0 - pt) ** gamma * bce

    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()

def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Soft Dice loss for tubular structure overlap."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


# ==========================================
# TRAINER
# ==========================================

class SeedDetectorTrainer:
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
        
        # --- Config Link: Read Focal Loss parameters from config.py ---
        train_cfg = SEED_CONFIG.get("training", {})
        self.ep_alpha = train_cfg.get("ep_alpha", 0.80)
        self.ep_gamma = train_cfg.get("ep_gamma", 4.0)
        self.v_alpha  = train_cfg.get("vessel_alpha", 0.75)
        self.v_gamma  = train_cfg.get("vessel_gamma", 2.0)
        self.aux_spacing = train_cfg.get("aux_spacing", 20)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

    def train(self, train_samples: List[Dict], val_samples: List[Dict], save_path: str, config: dict) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        train_loader = DataLoader(
            SeedDataset(train_samples, sigma=self.sigma, aux_spacing=self.aux_spacing),
            batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            SeedDataset(val_samples, sigma=self.sigma, aux_spacing=self.aux_spacing),
            batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

        best_val_loss = float("inf")
        # 5 lists to track history for the final summary
        history = {"tp": [], "fp": [], "fn": [], "t_loss": [], "v_loss": []}

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch:3d}/{self.num_epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f}", flush=True)

            # Capture spatial metrics and save per-epoch image
            tp, fp, fn = self.visualize_performance(val_loader, epoch)
            
            history["tp"].append(tp)
            history["fp"].append(fp)
            history["fn"].append(fn)
            history["t_loss"].append(train_loss)
            history["v_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                }, save_path)
                print(f"  ✓ Saved best model (val_loss={val_loss:.5f})", flush=True)

        # Generate the final summary with metrics and loss curves
        self._plot_training_summary(history)

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        
        with ctx:
            for img, hm, fov, vessel in loader:
                img, hm, fov, vessel = img.to(self.device), hm.to(self.device), fov.to(self.device), vessel.to(self.device)
                
                ep_pred, v_pred = self.model(img)

                # Seed Loss (Focal) - Targeting high recall
                ep_loss = focal_loss(ep_pred, hm, mask=fov, alpha=self.ep_alpha, gamma=self.ep_gamma)

                # Vessel Loss (Focal + Dice)
                v_foc = focal_loss(v_pred, vessel, alpha=self.v_alpha, gamma=self.v_gamma)
                v_dice = soft_dice_loss(v_pred, vessel)
                v_loss = v_foc + v_dice

                # Penalty for seeds in non-vessel regions
                fp_penalty = (ep_pred * (1.0 - vessel)).mean()

                loss = ep_loss + 0.3 * v_loss + 0.2 * fp_penalty

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()

        return total_loss / len(loader)


    def _plot_training_summary(self, history, save_dir="training_seed_detector"):
        """Plots a two-panel summary: TP/FP/FN Metrics and Loss Curves."""
        epochs = range(1, len(history["tp"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Panel 1: Metrics
        ax1.plot(epochs, history["tp"], label='True Positives', color='lime', marker='o', markersize=4)
        ax1.plot(epochs, history["fp"], label='False Positives', color='red', marker='x', markersize=4)
        ax1.plot(epochs, history["fn"], label='False Negatives', color='cyan', marker='s', markersize=4)
        ax1.set_title('Performance Metrics Progression', fontsize=14); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)
        
        # Panel 2: Loss Curves
        ax2.plot(epochs, history["t_loss"], label='Train Loss', color='orange', linewidth=2)
        ax2.plot(epochs, history["v_loss"], label='Val Loss', color='blue', linewidth=2, linestyle='--')
        ax2.set_title('Loss Curves', fontsize=14); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True)
        
        # Visual Warning if overfitting occurs
        if history["v_loss"][-1] > min(history["v_loss"]) * 1.1:
            ax2.text(0.5, 0.5, 'OVERFITTING DETECTED', transform=ax2.transAxes, 
                     fontsize=20, color='red', alpha=0.4, ha='center', fontweight='bold')

        summary_path = os.path.join(save_dir, "Training_Summary.png")
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150)
        plt.close()
        print(f"Saved complete training summary to: {summary_path}")
