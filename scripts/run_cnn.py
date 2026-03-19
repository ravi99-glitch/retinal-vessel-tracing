"""
run_cnn.py
=========================
Centerline UNet CNN Baseline — dataset-agnostic.

Three-way split:
  train → gradient updates
  val   → checkpoint selection
  test  → final metric reporting (never seen during training)

For datasets with an official test directory (e.g. DRIVE), the test
split is loaded from there automatically.  For single-folder datasets
(e.g. STARE), a ratio-based 3-way split is used.
"""

import os, sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import albumentations as A
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.centerline_unet_baseline import (
    CenterlineUNet, CenterlineLoss, CenterlinePredictor,
)
from data.dataloader import load_dataset
from data.dataset_paths import get_root, WEIGHTS_DIR, OUTPUT_DIR as _OUTPUT_BASE
from evaluation.metrics import CenterlineMetrics

# ==========================================
# CONFIG
# ==========================================
DATASET_NAME = "DRIVE"
DATA_ROOT    = str(get_root(DATASET_NAME))
SAVE_PATH    = str(WEIGHTS_DIR / "centerline_unet.pt")
OUTPUT_DIR   = str(_OUTPUT_BASE / "unet")

EPOCHS       = 10
BATCH_SIZE   = 2
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CFG    = dict(in_channels=1, base_ch=16)

# For DRIVE (has official test dir): controls train/val split only.
# For STARE etc.: controls full train/val/test split.
SPLIT_RATIOS = (0.7, 0.15, 0.15)

# Standardised metric columns — shared across all baseline scripts
METRIC_COLS = [
    "iou",
    "clDice",
    "betti_0_error",
    "hd95",
    "f1@1px",    "precision@1px", "recall@1px",
    "f1@2px",    "precision@2px", "recall@2px",
    "f1@3px",    "precision@3px", "recall@3px",
]

# ==========================================
# AUGMENTATION
# ==========================================
def _spatial_normalize():
    return [
        A.LongestMaxSize(max_size=584),
        A.PadIfNeeded(min_height=584, min_width=584, border_mode=0),
    ]

def get_train_transforms():
    return A.Compose([
        *_spatial_normalize(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.GridDistortion(p=1.0),
        ], p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ], additional_targets={'fov': 'mask', 'thick_gt': 'mask'})

def get_val_transforms():
    return A.Compose(
        _spatial_normalize(),
        additional_targets={'fov': 'mask', 'thick_gt': 'mask'},
    )

# ==========================================
# TRAINING HELPERS
# ==========================================
def run_epoch(model, loader, criterion, optimizer=None, device="cpu", desc=""):
    model.train() if optimizer else model.eval()
    total_loss = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in tqdm(loader, desc=desc, leave=False):
            img    = batch['image'].to(device)
            target = batch['centerline'].to(device)
            mask   = batch['fov_mask'].to(device)
            if optimizer:
                optimizer.zero_grad()
            pred = model(img)
            loss, _ = criterion(pred, target, mask=mask)
            if optimizer:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)

def plot_loss_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses,
             label='Train Loss', color='#1f77b4', lw=2)
    plt.plot(range(1, len(val_losses)+1), val_losses,
             label='Val Loss', color='#ff7f0e', lw=2, linestyle='--')
    plt.title("CNN Training Progression", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ==========================================
# EVALUATION
# ==========================================
@torch.no_grad()
def evaluate_and_visualize(checkpoint_path, test_dataset, split_label="Test"):
    print(f"\nEvaluating on {split_label} set ({len(test_dataset)} images)")
    panels_dir = os.path.join(OUTPUT_DIR, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor = CenterlinePredictor.from_checkpoint(
            checkpoint_path, device=DEVICE,
        )

    metrics_fn  = CenterlineMetrics(tolerance_levels=[1, 2, 3])
    all_metrics = []
    mosaic_data = []

    for batch in tqdm(loader, desc=f"Evaluating ({split_label})"):
        img_np      = batch['image'][0, 0].numpy()
        mask_np     = (batch['fov_mask'][0, 0].numpy() * 255).astype(np.uint8)
        gt_skel     = (batch['centerline'][0, 0].numpy() * 255).astype(np.uint8)
        vessel_mask = (batch['vessel_mask'][0, 0].numpy() * 255).astype(np.uint8)
        image_id    = batch['id'][0]

        prob_map, pred_skel = predictor.predict(img_np, fov_mask=mask_np)

        # Threshold prob map → binary vessel mask for IoU + clDice
        pred_vessel_mask = (prob_map >= 0.5).astype(np.uint8) * 255

        res = metrics_fn.compute_all_metrics(
            pred_skeleton    = pred_skel,
            gt_skeleton      = gt_skel,
            pred_vessel_mask = pred_vessel_mask,
            gt_vessel_mask   = vessel_mask,
            pred_prob        = prob_map,
            fov_mask         = mask_np,
        )
        res['image_id'] = image_id
        all_metrics.append(res)

        gt_vis   = (gt_skel > 0).astype(np.uint8) * 255
        pred_vis = (pred_skel > 0).astype(np.uint8) * 255
        mosaic_data.append(dict(
            image_id=image_id, gt_skeleton=gt_vis,
            pred_skeleton=pred_vis, metrics=res,
        ))

        # Per-image panel
        fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='white')
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title(f"Preprocessed ({image_id})")
        axes[1].imshow(prob_map * (mask_np > 0), cmap='gray')
        axes[1].set_title("Probability Map")
        axes[2].imshow(np.concatenate([gt_vis, pred_vis], axis=1), cmap='gray')
        axes[2].set_title("GT | Pred")
        overlay = np.zeros((*img_np.shape[:2], 3), dtype=np.uint8)
        overlay[..., 1] = gt_vis; overlay[..., 0] = pred_vis
        axes[3].imshow(overlay)
        axes[3].set_title(
            f"F1@2px: {res.get('f1@2px', 0):.3f} | IoU: {res.get('iou', 0):.3f}"
        )
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(panels_dir, f"{image_id}_panel.png"), dpi=150)
        plt.close()

    # Mosaic
    if mosaic_data:
        n      = len(mosaic_data)
        n_cols = min(n, 4)
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
        axes = np.array(axes).reshape(-1)
        for i, d in enumerate(mosaic_data):
            h, w = d['gt_skeleton'].shape
            ov = np.zeros((h, w, 3), dtype=np.uint8)
            ov[..., 1] = d['gt_skeleton']; ov[..., 0] = d['pred_skeleton']
            axes[i].imshow(ov)
            axes[i].set_title(
                f"{d['image_id']}\n"
                f"F1@2px: {d['metrics'].get('f1@2px', 0):.3f} | "
                f"IoU: {d['metrics'].get('iou', 0):.3f}",
                fontweight='bold',
            )
            axes[i].axis('off')
        for j in range(i+1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"mosaic_{split_label.lower()}.png"), dpi=200)
        plt.close()

    # Summary table + CSV
    df = pd.DataFrame(all_metrics)
    summary_rows = [
        {"Metric": c, "Mean +/- Std": f"{df[c].mean():.4f} +/- {df[c].std():.4f}"}
        for c in METRIC_COLS if c in df.columns
    ]
    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "="*50)
    print(f"   FINAL RESULTS — {DATASET_NAME} ({split_label})")
    print("="*50)
    print(summary_df.to_string(index=False))
    print("="*50)

    df.to_csv(os.path.join(OUTPUT_DIR, "metrics_per_image.csv"), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    print(f"CSVs saved → {OUTPUT_DIR}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    RUN_TRAINING = True

    common = dict(
        target="unet",
        split_ratios=SPLIT_RATIOS,
    )

    train_ds, train_loader = load_dataset(
        DATA_ROOT, DATASET_NAME, split="train",
        batch_size=BATCH_SIZE, shuffle=True,
        transform=get_train_transforms(), **common,
    )

    val_ds, val_loader = load_dataset(
        DATA_ROOT, DATASET_NAME, split="val",
        batch_size=1, shuffle=False,
        transform=get_val_transforms(), **common,
    )

    test_ds, _ = load_dataset(
        DATA_ROOT, DATASET_NAME, split="test",
        batch_size=1, shuffle=False,
        transform=get_val_transforms(), **common,
    )

    print(f"[{DATASET_NAME}]  train={len(train_ds)}  "
          f"val={len(val_ds)}  test={len(test_ds)}")

    if RUN_TRAINING:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        model     = CenterlineUNet(**MODEL_CFG).to(DEVICE)
        criterion = CenterlineLoss(0.4, 0.6, pos_weight=10.0)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        train_hist, val_hist = [], []
        best_val_loss = float('inf')

        print(f"--- Training ({EPOCHS} epochs) ---")
        for epoch in range(1, EPOCHS + 1):
            t_loss = run_epoch(
                model, train_loader, criterion, optimizer, DEVICE, "Train",
            )
            v_loss = run_epoch(
                model, val_loader, criterion, None, DEVICE, "Val",
            )
            train_hist.append(t_loss)
            val_hist.append(v_loss)
            scheduler.step()

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                torch.save(
                    {'model_state': model.state_dict(), 'model_cfg': MODEL_CFG},
                    SAVE_PATH,
                )

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3}/{EPOCHS}  "
                      f"train={t_loss:.4f}  val={v_loss:.4f}")

        plot_loss_curves(
            train_hist, val_hist,
            os.path.join(OUTPUT_DIR, "training_val_curves.png"),
        )
        print(f"Best val loss: {best_val_loss:.4f}")

    if os.path.exists(SAVE_PATH):
        evaluate_and_visualize(SAVE_PATH, test_ds, split_label="Test")
