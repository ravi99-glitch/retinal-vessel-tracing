"""
run_greedytracer.py
=========================
Greedy Tracer Baseline — dataset-agnostic.

Change DATASET_NAME and DATA_ROOT below to run on any supported dataset.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm
from skimage.morphology import skeletonize
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.greedy_tracer_baseline import GreedyTracerBaseline
from data.dataloader import RetinalFundusDataset
from data.dataloader import load_dataset
from data.dataset_paths import get_root, OUTPUT_DIR as _OUTPUT_BASE
from evaluation.metrics import CenterlineMetrics

# ==========================================
# CONFIG — change these to switch dataset
# ==========================================
DATASET_NAME = "DRIVE"
DATA_ROOT    = str(get_root(DATASET_NAME))
OUTPUT_DIR   = str(_OUTPUT_BASE / "greedy_tracer")

MODEL_CFG = dict(
    sigma_min    = 0.5,
    sigma_max    = 3.0,
    num_scales   = 5,
    gauss_sigma  = 1.5,
    seed_thresh  = 0.25,
    step_thresh  = 0.15,
    min_length   = 15,
    thin_output  = True,
    min_obj_size = 0,
)

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
# VISUALISATION SETTINGS
# ==========================================
FONT_SIZE_TITLE  = 14
FONT_SIZE_LABEL  = 12
FONT_SIZE_LEGEND = 10
TOP_N_ORDER      = 50
DPI              = 200

# ==========================================
# HELPERS
# ==========================================
def save_standard_panel(img_rgb, vesselness, gt_skel_vis, pred_skel_vis,
                        mask, res, image_id, panels_dir):
    fov_bin    = (mask > 0).astype(np.float32)
    vessel_vis = vesselness * fov_bin

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='white')

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original Image (ID: {image_id})",
                      fontweight='bold', fontsize=FONT_SIZE_TITLE)

    axes[1].imshow(vessel_vis, cmap='gray')
    axes[1].set_title("Frangi Vesselness Map",
                      fontweight='bold', fontsize=FONT_SIZE_TITLE)

    side_by_side = np.concatenate([gt_skel_vis, pred_skel_vis], axis=1)
    axes[2].imshow(side_by_side, cmap='gray')
    axes[2].set_title("1px Skeletons\n(Left: GT | Right: Pred)",
                      fontweight='bold', fontsize=FONT_SIZE_TITLE)

    overlay = np.zeros((*img_rgb.shape[:2], 3), dtype=np.uint8)
    overlay[..., 1] = gt_skel_vis
    overlay[..., 0] = pred_skel_vis
    axes[3].imshow(overlay)
    axes[3].set_title(
        f"Overlay Analysis\n"
        f"F1@2px: {res.get('f1@2px', 0):.3f} | "
        f"clDice: {res.get('clDice', 0):.3f} | "
        f"IoU: {res.get('iou', 0):.3f}",
        fontweight='bold', color='darkblue', fontsize=FONT_SIZE_TITLE,
    )

    legend_elements = [
        Patch(facecolor='green',  edgecolor='black', label='GT'),
        Patch(facecolor='red',    edgecolor='black', label='Pred'),
        Patch(facecolor='yellow', edgecolor='black', label='Match'),
    ]
    axes[3].legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, -0.15), ncol=3,
                   frameon=False, fontsize=FONT_SIZE_LEGEND)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(panels_dir, f"{image_id}_greedy_panel.png"),
                bbox_inches='tight', dpi=DPI)
    plt.close()


def save_trajectory_panel(vesselness, mask, traces, image_id, traj_dir):
    if len(traces) == 0:
        return

    fov_bin       = (mask > 0).astype(np.float32)
    vessel_bg     = vesselness * fov_bin
    trace_lengths = np.array([len(p) for p in traces])
    N             = len(traces)
    seeds         = np.array([p[0] for p in traces])

    BG = '#0d0d0d'
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis('off')

    axes[0].imshow(vessel_bg, cmap='gray', vmin=0, vmax=1)
    axes[0].scatter(seeds[:, 1], seeds[:, 0], c='cyan', s=12, alpha=0.8)
    axes[0].set_title(f"Vesselness + {N} Seeds", color='white',
                      fontsize=FONT_SIZE_TITLE)

    n_show     = min(TOP_N_ORDER, N)
    cmap_order = plt.cm.plasma
    order_norm = mcolors.Normalize(vmin=0, vmax=max(n_show - 1, 1))
    axes[1].imshow(vessel_bg, cmap='gray', alpha=0.2)
    for idx in range(n_show):
        coords = np.array(traces[idx])
        color  = cmap_order(order_norm(idx))
        axes[1].plot(coords[:, 1], coords[:, 0], color=color, linewidth=1.2)
    axes[1].set_title(f"Top-{n_show} Visit Order", color='white',
                      fontsize=FONT_SIZE_TITLE)

    sm = plt.cm.ScalarMappable(cmap=cmap_order, norm=order_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Visit Order (0 = First/Strongest)', color='white',
                   fontsize=FONT_SIZE_LABEL)
    cbar.ax.yaxis.set_tick_params(colors='white')

    axes[2].axis('on')
    axes[2].set_facecolor('#1a1a1a')
    log_bins = np.logspace(
        np.log10(max(trace_lengths.min(), 1)),
        np.log10(trace_lengths.max()), 40,
    )
    axes[2].hist(trace_lengths, bins=log_bins, color='#f07f2a', alpha=0.85)
    axes[2].set_xscale('log')
    axes[2].set_title("Length Distribution (log x)", color='white',
                      fontsize=FONT_SIZE_TITLE)
    axes[2].tick_params(colors='white')
    axes[2].set_xlabel('Trace Length (pixels)', color='white',
                       fontsize=FONT_SIZE_LABEL)
    axes[2].set_ylabel('Count (Number of Traces)', color='white',
                       fontsize=FONT_SIZE_LABEL)

    plt.suptitle(
        f"Greedy Tracer Trajectory Analysis — {DATASET_NAME} — Image {image_id}",
        color='white', fontsize=FONT_SIZE_TITLE + 4, fontweight='bold', y=1.02,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(traj_dir, f"{image_id}_trajectory.png"),
                facecolor=BG, dpi=DPI, bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN
# ==========================================
def main():
    panels_dir = os.path.join(OUTPUT_DIR, "panels")
    traj_dir   = os.path.join(OUTPUT_DIR, "trajectories")
    os.makedirs(panels_dir, exist_ok=True)
    os.makedirs(traj_dir,   exist_ok=True)

    # ── Data & model ────────────────────────────────────
    dataset, loader = load_dataset(DATA_ROOT, DATASET_NAME, target="greedy_tracer", batch_size=1)
    model      = GreedyTracerBaseline(**MODEL_CFG)
    metrics_fn = CenterlineMetrics(tolerance_levels=[1, 2, 3])

    num_total = len(dataset)
    print(f"[{DATASET_NAME}]  {num_total} images\n")

    all_metrics = []

    # ── Per-image evaluation ────────────────────────────
    for i in tqdm(range(num_total), desc="Evaluating Greedy Tracer"):
        sample   = dataset[i]
        image_id = sample["id"]

        img_rgb     = sample["image"]           # (H,W,3) uint8
        fov_mask    = sample["fov_mask"]         # (H,W) uint8 {0,255}
        vessel_mask = sample["vessel_mask"]      # (H,W) uint8 {0,255}

        gt_skel = (skeletonize(vessel_mask > 128) * 255).astype(np.uint8)

        pred_skel, vesselness, traces = model.extract_centerline(
            img_rgb, external_fov_mask=fov_mask, return_vesselness=True,
        )

        # Predicted vessel mask: threshold vesselness map at 0.5
        pred_vessel_mask = (vesselness >= 0.5).astype(np.uint8) * 255

        res = metrics_fn.compute_all_metrics(
            pred_skeleton    = pred_skel,
            gt_skeleton      = gt_skel,
            pred_vessel_mask = pred_vessel_mask,
            gt_vessel_mask   = vessel_mask,
            fov_mask         = fov_mask,
        )
        res.update({
            'image_id':   image_id,
            'num_traces': len(traces),
            'median_len': float(np.median([len(t) for t in traces])) if traces else 0.0,
        })
        all_metrics.append(res)

        gt_skel_vis   = (gt_skel > 0).astype(np.uint8) * 255
        pred_skel_vis = (pred_skel > 0).astype(np.uint8) * 255

        save_standard_panel(img_rgb, vesselness, gt_skel_vis, pred_skel_vis,
                            fov_mask, res, image_id, panels_dir)
        save_trajectory_panel(vesselness, fov_mask, traces, image_id, traj_dir)

    # ── Summary table + CSV ─────────────────────────────
    df = pd.DataFrame(all_metrics)
    summary_rows = [
        {"Metric": c, "Mean +/- Std": f"{df[c].mean():.4f} +/- {df[c].std():.4f}"}
        for c in METRIC_COLS if c in df.columns
    ]
    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 55)
    print(f"   GREEDY TRACER — {DATASET_NAME} (N={num_total})")
    print("=" * 55)
    print(summary_df.to_string(index=False))
    print("=" * 55)

    summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "metrics_per_image.csv"), index=False)
    print(f"CSVs saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
