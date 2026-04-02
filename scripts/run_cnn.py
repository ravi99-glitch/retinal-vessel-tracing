"""
run_cnn.py
=========================
Centerline UNet CNN Baseline — dataset-agnostic.
Aligned with Greedy Tracer Visualization & HPC Logging.
"""

import os
import sys
import warnings
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from matplotlib.patches import Patch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet import (CenterlineLoss,
                         CenterlinePredictor,
                         CenterlineUNet)
from models.greedy_tracer import GreedyTracer 
from data.dataloader import OUTPUT_DIR as _OUTPUT_BASE
from data.dataloader import TEST_DATASETS, WEIGHTS_DIR, get_data, get_test_data
from evaluation.metrics import CenterlineMetrics

# ==========================================
# CONFIG & HPC OPTIMIZATIONS
# ==========================================
SAVE_PATH = str(WEIGHTS_DIR / "centerline_unet.pt")
OUTPUT_DIR = str(_OUTPUT_BASE / "unet")

EPOCHS = 100
BATCH_SIZE = 4
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CFG = dict(in_channels=1, base_ch=64)
RESIZE = (512, 512)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True 

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
PIN_MEMORY = True if DEVICE == "cuda" else False

METRIC_COLS = [
    "iou", "clDice", "betti_0_error", "hd95",
    "f1@2px", "precision@2px", "recall@2px",
]

# VISUALISATION CONSTANTS
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10
DPI = 200

# ==========================================
# AUGMENTATION
# ==========================================
def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.OneOf([A.ElasticTransform(alpha=1, sigma=50, p=1.0), A.GridDistortion(p=1.0)], p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ],
        additional_targets={"fov": "mask", "thick_gt": "mask"},
    )

# ==========================================
# HELPERS
# ==========================================
def save_standard_panel(img_gray, prob_map, gt_skel, pred_skel, mask, res, image_id, panels_dir, dataset_name=""):
    """Aligned with Greedy Tracer visualization style."""
    img_vis = np.clip(img_gray, 0, 1)
    
    # Overlay logic: Green (Missed), Red (Wrong), Yellow (Correct)
    gt_bool = gt_skel > 0
    pred_bool = pred_skel > 0
    tp = gt_bool & pred_bool
    fn = gt_bool & ~pred_bool
    fp = ~gt_bool & pred_bool

    overlay = np.zeros((*img_gray.shape, 3), dtype=np.uint8)
    overlay[fn] = [0, 200, 80]    # Green
    overlay[fp] = [220, 50, 50]   # Red
    overlay[tp] = [255, 220, 0]   # Yellow

    fig = plt.figure(figsize=(26, 7), facecolor="white")
    gs = fig.add_gridspec(1, 4, wspace=0.04, left=0.01, right=0.99, top=0.88, bottom=0.08)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    TITLE_KW = dict(fontweight="bold", fontsize=FONT_SIZE_TITLE, color="#111111", pad=8)

    axes[0].imshow(img_vis, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Fundus Image · {image_id}", **TITLE_KW)

    axes[1].imshow(prob_map, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("CNN Probability Map", **TITLE_KW)

    side_by_side = np.concatenate([(gt_skel > 0)*255, (pred_skel > 0)*255], axis=1)
    axes[2].imshow(side_by_side, cmap="gray")
    axes[2].set_title("Skeletons · GT | Pred", **TITLE_KW)
    axes[2].axvline(x=img_gray.shape[1], color="#555555", linewidth=1.0, linestyle="--")

    axes[3].imshow(overlay)
    axes[3].set_title(
        f"F1@2px: {res.get('f1@2px', 0):.3f}   Prec: {res.get('precision@2px', 0):.3f}   Rec: {res.get('recall@2px', 0):.3f}\n"
        f"clDice: {res.get('clDice', 0):.3f}   IoU: {res.get('iou', 0):.3f}",
        **TITLE_KW,
    )

    legend_elements = [
        Patch(facecolor="#00c850", label="GT Missed"),
        Patch(facecolor="#ffdc00", label="Correct (TP)"),
        Patch(facecolor="#dc3232", label="Wrong (FP)"),
    ]
    axes[3].legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=FONT_SIZE_LEGEND)

    for ax in axes: ax.axis("off")
    plt.savefig(os.path.join(panels_dir, f"{dataset_name}_{image_id}_centreline_analysis.png"), bbox_inches="tight", dpi=DPI)
    plt.close()

# ==========================================
# TRAINING & EVALUATION
# ==========================================
def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    model.train() if optimizer else model.eval()
    total_loss = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            img, target, mask = batch["image"].to(device), batch["centerline"].to(device), batch["fov_mask"].to(device)
            if optimizer: optimizer.zero_grad(set_to_none=True)
            pred = model(img)
            loss, _ = criterion(pred, target, mask=mask)
            if optimizer:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_and_visualize(checkpoint_path, test_dataset, split_label="Test", output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    panels_dir = os.path.join(output_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    predictor = CenterlinePredictor.from_checkpoint(checkpoint_path, 
                                                  device=DEVICE, 
                                                  tracer=GreedyTracer(seed_thresh=0.335, step_thresh=0.15))
    metrics_fn = CenterlineMetrics(tolerance_levels=[1, 2, 3])
    loader = DataLoader(test_dataset, batch_size=1, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    print(f"\n[{split_label}] Evaluating {len(test_dataset)} images...")
    all_metrics = []
    
    for i, batch in enumerate(loader):
        img_np, mask_np = batch["image"][0, 0].numpy(), (batch["fov_mask"][0, 0].numpy() * 255).astype(np.uint8)
        gt_skel, v_mask, image_id = (batch["centerline"][0, 0].numpy() * 255).astype(np.uint8), (batch["vessel_mask"][0, 0].numpy() * 255).astype(np.uint8), batch["id"][0]

        prob_map, pred_skel = predictor.predict(img_np, fov_mask=mask_np)
        res = metrics_fn.compute_all_metrics(pred_skeleton=pred_skel, gt_skeleton=gt_skel, pred_vessel_mask=(prob_map >= 0.5).astype(np.uint8)*255, gt_vessel_mask=v_mask, fov_mask=mask_np)
        res["image_id"] = image_id
        all_metrics.append(res)

        save_standard_panel(img_np, prob_map, gt_skel, pred_skel, mask_np, res, image_id, panels_dir, dataset_name=split_label)
        if (i + 1) % 5 == 0 or (i + 1) == len(test_dataset):
            print(f" -> Processed {i + 1}/{len(test_dataset)} images")

    # 1. Convert list of dicts to a DataFrame
    df = pd.DataFrame(all_metrics)
    
    # 2. Save the full per-image breakdown 
    df.to_csv(os.path.join(output_dir, "metrics_per_image.csv"), index=False)

    # 3. Calculate Mean +/- Std for each metric column
    summary_rows = []
    for col in METRIC_COLS:
        if col in df.columns:
            m = df[col].mean()
            s = df[col].std()
            summary_rows.append({"Metric": col, "Mean +/- Std": f"{m:.4f} +/- {s:.4f}"})
    
    # 4. Save and Print the summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    
    print("\n" + "="*50)
    print(f"   FINAL SUMMARY: {split_label}")
    print("="*50)
    print(summary_df.to_string(index=False))
    print("="*50 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if not any([args.train, args.eval, args.test]): args.train = args.eval = args.test = True

    if args.train:
        train_ds, _ = get_data("unet", "train", batch_size=BATCH_SIZE, resize=RESIZE, transform=get_train_transforms())
        val_ds, _   = get_data("unet", "val", batch_size=1, resize=RESIZE)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader   = DataLoader(val_ds, batch_size=1, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        model, criterion = CenterlineUNet(**MODEL_CFG).to(DEVICE), CenterlineLoss(0.1, 0.9, pos_weight=2.0).to(DEVICE)
        optimizer, scheduler = optim.AdamW(model.parameters(), lr=LR), CosineAnnealingLR(optim.AdamW(model.parameters(), lr=LR), T_max=EPOCHS)

        best_loss, train_hist, val_hist = float("inf"), [], []
        print(f"--- Training ({EPOCHS} epochs) ---")
        for epoch in range(1, EPOCHS + 1):
            t_loss, v_loss = run_epoch(model, train_loader, criterion, optimizer, DEVICE), run_epoch(model, val_loader, criterion, None, DEVICE)
            train_hist.append(t_loss); val_hist.append(v_loss); scheduler.step()
            
            is_best = v_loss < best_loss
            if is_best or epoch == 1:
                best_loss = v_loss
                os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                torch.save({"model_state": model.state_dict(), "model_cfg": MODEL_CFG}, SAVE_PATH)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}{' *Best*' if is_best else ''}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Assuming plot_loss_curves exists as previously defined
    
    if args.eval and os.path.exists(SAVE_PATH):
        val_ds, _ = get_data("unet", "val", batch_size=1, resize=RESIZE)
        evaluate_and_visualize(SAVE_PATH, val_ds, split_label="Val", output_dir=os.path.join(OUTPUT_DIR, "val"))

    if args.test and os.path.exists(SAVE_PATH):
        for name in TEST_DATASETS:
            try:
                ds, _ = get_test_data(name, "unet", batch_size=1, resize=RESIZE)
                evaluate_and_visualize(SAVE_PATH, ds, split_label=name, output_dir=os.path.join(OUTPUT_DIR, name))
            except Exception as e: print(f"Skipping {name}: {e}")
