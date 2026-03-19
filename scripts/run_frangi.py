"""
run_frangi.py
=========================
Frangi Vesselness Baseline — dataset-agnostic.

Change DATASET_NAME and DATA_ROOT below to run on any supported dataset.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.frangi_baseline import FrangiBaseline
from data.dataloader import RetinalFundusDataset
from data.dataloader import load_dataset
from data.dataset_paths import get_root, OUTPUT_DIR as _OUTPUT_BASE
from evaluation.metrics import CenterlineMetrics

# ==========================================
# CONFIG — change these to switch dataset
# ==========================================
DATASET_NAME = "DRIVE"
DATA_ROOT    = str(get_root(DATASET_NAME))
OUTPUT_DIR   = str(_OUTPUT_BASE / "frangi")

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
# MAIN
# ==========================================
if __name__ == "__main__":
    panels_dir = os.path.join(OUTPUT_DIR, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    # ── Data & model ────────────────────────────────────
    dataset, loader = load_dataset(DATA_ROOT, DATASET_NAME, target="frangi", batch_size=1)
    model = FrangiBaseline()
    metrics_calculator = CenterlineMetrics(tolerance_levels=[1, 2, 3])

    print(f"[{DATASET_NAME}]  {len(dataset)} images\n")

    all_metrics = []
    mosaic_data = []

    # ── Per-image evaluation ────────────────────────────
    for i in tqdm(range(len(dataset)), desc="Evaluating Frangi Baseline"):
        sample   = dataset[i]
        image_id = sample["id"]

        # Run Frangi baseline
        pred_skeleton, vesselness, _ = model.extract_centerline(
            sample["image"],
            return_vesselness=True,
            external_fov_mask=sample["fov_mask"],
        )

        # Ground truth
        fov_mask_bool = sample["fov_mask"] > 128
        gt_binary     = (sample["vessel_mask"] > 128) & fov_mask_bool
        gt_skeleton   = skeletonize(gt_binary)

        # Predicted vessel mask: threshold vesselness map at 0.5
        pred_vessel_mask = (vesselness >= 0.5).astype(np.uint8)

        # Metrics
        raw_metrics = metrics_calculator.compute_all_metrics(
            pred_skeleton    = pred_skeleton,
            gt_skeleton      = gt_skeleton,
            pred_vessel_mask = pred_vessel_mask,
            gt_vessel_mask   = gt_binary.astype(np.uint8),
            fov_mask         = sample["fov_mask"],
        )
        metrics_entry = {"image_id": image_id}
        metrics_entry.update(raw_metrics)
        all_metrics.append(metrics_entry)

        mosaic_data.append({
            "image_id":      image_id,
            "gt_skeleton":   gt_skeleton,
            "pred_skeleton": pred_skeleton,
            "metrics":       metrics_entry,
        })

        # ── Panel visualisation ─────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='white')

        axes[0].imshow(sample["image"])
        axes[0].set_title(f"Original Image (ID: {image_id})", fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(vesselness, cmap='gray')
        axes[1].set_title("Frangi Vesselness", fontsize=14, fontweight='bold')
        axes[1].axis('off')

        combined_skel = np.hstack((
            gt_skeleton.astype(np.uint8) * 255,
            pred_skeleton.astype(np.uint8) * 255,
        ))
        axes[2].imshow(combined_skel, cmap='gray')
        axes[2].set_title("1px Skeletons\n(Left: GT | Right: Pred)", fontsize=14, fontweight='bold')
        axes[2].axis('off')

        h, w = pred_skeleton.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 1] = gt_skeleton.astype(np.uint8) * 255
        overlay[:, :, 0] = pred_skeleton.astype(np.uint8) * 255
        axes[3].imshow(overlay)
        axes[3].set_title(
            f"Overlay Analysis\n"
            f"F1@2px: {raw_metrics.get('f1@2px', 0):.3f} | "
            f"clDice: {raw_metrics.get('clDice', 0):.3f} | "
            f"IoU: {raw_metrics.get('iou', 0):.3f}",
            fontsize=14, fontweight='bold', color='darkblue',
        )
        axes[3].axis('off')

        legend_elements = [
            Patch(facecolor='green',  edgecolor='black', label='GT'),
            Patch(facecolor='red',    edgecolor='black', label='Pred'),
            Patch(facecolor='yellow', edgecolor='black', label='Match'),
        ]
        axes[3].legend(handles=legend_elements, loc='lower center',
                       bbox_to_anchor=(0.5, -0.2), ncol=3,
                       frameon=False, fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(panels_dir, f"{image_id}_comparison.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # ── Mosaic overview ─────────────────────────────────
    if mosaic_data:
        n = len(mosaic_data)
        n_cols = 4
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        axes = np.array(axes).flatten()

        for i, data in enumerate(mosaic_data):
            h, w = data['pred_skeleton'].shape
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            overlay[:, :, 1] = data['gt_skeleton'].astype(np.uint8) * 255
            overlay[:, :, 0] = data['pred_skeleton'].astype(np.uint8) * 255
            axes[i].imshow(overlay)
            axes[i].set_title(
                f"[{data['image_id']}]\n"
                f"clDice: {data['metrics'].get('clDice', 0):.3f} | "
                f"IoU: {data['metrics'].get('iou', 0):.3f}\n"
                f"F1@2px: {data['metrics'].get('f1@2px', 0):.3f}",
                fontsize=9, fontweight='bold',
            )
            axes[i].axis('off')
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "mosaic_overview.png"),
                    dpi=200, bbox_inches='tight')
        plt.close()

    # ── Summary table + CSV ─────────────────────────────
    df = pd.DataFrame(all_metrics)
    summary_rows = [
        {"Metric": c, "Mean +/- Std": f"{df[c].mean():.4f} +/- {df[c].std():.4f}"}
        for c in METRIC_COLS if c in df.columns
    ]
    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 45)
    print(f"   FRANGI BASELINE — {DATASET_NAME}")
    print("=" * 45)
    print(summary_df.to_string(index=False))
    print("=" * 45)

    df.to_csv(os.path.join(OUTPUT_DIR, "metrics_per_image.csv"), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    print(f"CSVs saved → {OUTPUT_DIR}")
