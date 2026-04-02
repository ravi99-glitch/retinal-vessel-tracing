"""
run_greedytracer.py
=========================
Greedy Tracer Baseline — dataset-specific configurations.
"""

import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.greedy_tracer import GreedyTracerBaseline
from data.dataloader import OUTPUT_DIR as _OUTPUT_BASE
from data.dataloader import TEST_DATASETS, get_test_data
from evaluation.metrics import CenterlineMetrics

# ==========================================
# METRIC SETTINGS
# ==========================================
METRIC_COLS = [
    "iou",
    "clDice",
    "betti_0_error",
    "hd95",
    "f1@1px",
    "precision@1px",
    "recall@1px",
    "f1@2px",
    "precision@2px",
    "recall@2px",
    "f1@3px",
    "precision@3px",
    "recall@3px",
]

# ==========================================
# PER-DATASET GREEDY PARAMETERS
# ==========================================
# UPDATED WITH GRID SEARCH OPTIMALS
GREEDY_PARAMS = {
    "DRIVE": dict(
        sigma_min=0.5,
        sigma_max=2.5,
        num_scales=5,
        gauss_sigma=1.5,     # Tuned
        seed_thresh=0.4,     # Tuned
        step_thresh=0.2,     # Tuned
        min_length=10.0,
        thin_output=True,  
        min_obj_size=0,      # Tuned
    ),
    
    "DRHAGIS": dict(
        sigma_min=0.5,
        sigma_max=2.5,
        num_scales=5,
        gauss_sigma=1.5,     # Tuned
        seed_thresh=0.3,     # Tuned
        step_thresh=0.1,     # Tuned
        min_length=10.0,
        thin_output=True,
        min_obj_size=0,      # Tuned
    ),
}

DEFAULT_GREEDY_PARAMS = dict(
    sigma_min=0.5,
    sigma_max=3.0,
    num_scales=5,
    gauss_sigma=1.5,
    seed_thresh=0.25,
    step_thresh=0.15,
    min_length=15,
    thin_output=True,
    min_obj_size=0,
)

# ==========================================
# VISUALISATION SETTINGS
# ==========================================
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10
TOP_N_ORDER = 50
DPI = 200

# ==========================================
# HELPERS
# ==========================================
def save_standard_panel(
    img_rgb, vesselness, gt_skel_vis, pred_skel_vis, mask, res, image_id, panels_dir, dataset_name=""
):
    fov_bin = (mask > 0).astype(np.float32)
    vessel_vis = vesselness * fov_bin

    # --- Overlay with correct semantics ---
    gt_bool   = gt_skel_vis > 0
    pred_bool = pred_skel_vis > 0
    tp = gt_bool & pred_bool       # GT + correct pred  → yellow
    fn = gt_bool & ~pred_bool      # GT not predicted   → green
    fp = ~gt_bool & pred_bool      # pred not in GT     → red

    overlay = np.zeros((*img_rgb.shape[:2], 3), dtype=np.uint8)
    overlay[fn] = [0,   200, 80]   # green  — GT missed
    overlay[fp] = [220, 50,  50]   # red    — wrong prediction
    overlay[tp] = [255, 220, 0]    # yellow — correct prediction

    fig = plt.figure(figsize=(26, 7), facecolor="white")
    gs  = fig.add_gridspec(1, 4, wspace=0.04, left=0.01, right=0.99, top=0.88, bottom=0.08)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    TITLE_KW = dict(fontweight="bold", fontsize=FONT_SIZE_TITLE, color="#111111", pad=8)

    # Panel 0 — Fundus
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Fundus Image · {image_id}", **TITLE_KW)

    # Panel 1 — Vesselness
    axes[1].imshow(vessel_vis, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Multi-Scale Vesselness", **TITLE_KW)

    # Panel 2 — Skeletons side by side
    side_by_side = np.concatenate([gt_skel_vis, pred_skel_vis], axis=1)
    axes[2].imshow(side_by_side, cmap="gray")
    axes[2].set_title("Skeletons · GT (left) | Pred (right)", **TITLE_KW)
    axes[2].axvline(x=gt_skel_vis.shape[1], color="#555555", linewidth=1.0, linestyle="--")

    # Panel 3 — Overlap
    axes[3].imshow(np.zeros((*img_rgb.shape[:2], 3), dtype=np.uint8))
    axes[3].imshow(overlay)
    axes[3].set_title(
        f"F1@2px: {res.get('f1@2px', 0):.3f}   Prec: {res.get('precision@2px', 0):.3f}   Rec: {res.get('recall@2px', 0):.3f}\n"
        f"clDice: {res.get('clDice', 0):.3f}   IoU: {res.get('iou', 0):.3f}",
        **TITLE_KW,
    )

    legend_elements = [
        Patch(facecolor="#00c850", edgecolor="none", label="GT"),
        Patch(facecolor="#ffdc00", edgecolor="none", label="Correct (TP)"),
        Patch(facecolor="#dc3232", edgecolor="none", label="Wrong (FP)"),
    ]
    axes[3].legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
        labelcolor="#111111",
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{dataset_name}  ·  {image_id}", color="#aaaaaa", fontsize=11, y=0.97)

    plt.savefig(
        os.path.join(panels_dir, f"{dataset_name}_{image_id}_centreline_analysis.png"),
        bbox_inches="tight", dpi=DPI, facecolor=fig.get_facecolor(),
    )
    plt.close()


def save_trajectory_panel(vesselness, mask, traces, image_id, traj_dir, dataset_name=""):
    if len(traces) == 0: return

    fov_bin = (mask > 0).astype(np.float32)
    vessel_bg = vesselness * fov_bin
    trace_lengths = np.array([len(p) for p in traces])
    seeds = np.array([p[0] for p in traces])

    BG = "#0d0d0d"
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis("off")

    axes[0].imshow(vessel_bg, cmap="gray", vmin=0, vmax=1)
    axes[0].scatter(seeds[:, 1], seeds[:, 0], c="cyan", s=12, alpha=0.8)
    axes[0].set_title(f"Vesselness + {len(traces)} Seeds", color="white", fontsize=FONT_SIZE_TITLE)

    n_show = min(TOP_N_ORDER, len(traces))
    cmap_order = plt.cm.plasma
    order_norm = mcolors.Normalize(vmin=0, vmax=max(n_show - 1, 1))
    axes[1].imshow(vessel_bg, cmap="gray", alpha=0.2)
    for idx in range(n_show):
        coords = np.array(traces[idx])
        axes[1].plot(coords[:, 1], coords[:, 0], color=cmap_order(order_norm(idx)), linewidth=1.2)
    axes[1].set_title(f"Top-{n_show} Visit Order", color="white", fontsize=FONT_SIZE_TITLE)

    sm = plt.cm.ScalarMappable(cmap=cmap_order, norm=order_norm)
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Visit Order (0 = First)", color="white", fontsize=FONT_SIZE_LABEL)
    cbar.ax.yaxis.set_tick_params(colors="white")

    axes[2].axis("on")
    axes[2].set_facecolor("#1a1a1a")
    log_bins = np.logspace(np.log10(max(trace_lengths.min(), 1)), np.log10(trace_lengths.max()), 40)
    axes[2].hist(trace_lengths, bins=log_bins, color="#f07f2a", alpha=0.85)
    axes[2].set_xscale("log")
    axes[2].set_title("Length Distribution (log x)", color="white", fontsize=FONT_SIZE_TITLE)
    axes[2].tick_params(colors="white")
    axes[2].set_xlabel("Trace Length (pixels)", color="white", fontsize=FONT_SIZE_LABEL)
    axes[2].set_ylabel("Count", color="white", fontsize=FONT_SIZE_LABEL)

    plt.suptitle(f"Greedy Tracer Trajectory Analysis — {dataset_name} — {image_id}", color="white", fontsize=FONT_SIZE_TITLE + 4, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(traj_dir, f"{dataset_name}_{image_id}_trace_analysis.png"), facecolor=BG, dpi=DPI, bbox_inches="tight")
    plt.close()

# ==========================================
# EVALUATE
# ==========================================
def evaluate(dataset_name):
    output_dir = str(_OUTPUT_BASE / "greedy_tracer" / dataset_name)
    panels_dir = os.path.join(output_dir, "panels")
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(panels_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    dataset, _ = get_test_data(dataset_name, "greedy_tracer", batch_size=1, resize=None)
    
    # FETCH DATASET SPECIFIC PARAMS
    params = GREEDY_PARAMS.get(dataset_name, DEFAULT_GREEDY_PARAMS)
    model = GreedyTracerBaseline(**params)
    metrics_fn = CenterlineMetrics(tolerance_levels=[1, 2, 3])

    print(f"\n[{dataset_name}] Evaluating {len(dataset)} images (HPC Mode)...")
    all_metrics = []

    # Removed tqdm for clean SLURM logs
    for i in range(len(dataset)):
        sample = dataset[i]
        image_id, img_rgb, fov_mask, vessel_mask = sample["id"], sample["image"], sample["fov_mask"], sample["vessel_mask"]

        gt_skel = (skeletonize(vessel_mask > 128) * 255).astype(np.uint8)

        pred_skel, vesselness, traces = model.extract_centerline(
            sample["preprocessed"], fov_mask=fov_mask, return_vesselness=True,
        )

        res = metrics_fn.compute_all_metrics(
            pred_skeleton=pred_skel, gt_skeleton=gt_skel,
            pred_vessel_mask=(vesselness >= 0.5).astype(np.uint8) * 255,
            gt_vessel_mask=vessel_mask, fov_mask=fov_mask,
        )
        
        res.update({
            "image_id": image_id,
            "num_traces": len(traces),
            "median_len": float(np.median([len(t) for t in traces])) if traces else 0.0,
        })
        all_metrics.append(res)

        # Visuals
        save_standard_panel(img_rgb, vesselness, (gt_skel > 0) * 255, (pred_skel > 0) * 255, fov_mask, res, image_id, panels_dir, dataset_name=dataset_name)
        save_trajectory_panel(vesselness, fov_mask, traces, image_id, traj_dir, dataset_name=dataset_name)

        if (i + 1) % 5 == 0 or (i + 1) == len(dataset):
            print(f" -> Processed {i + 1}/{len(dataset)} images")

    # Summary and CSVs
    df = pd.DataFrame(all_metrics)
    summary_rows = [{"Metric": c, "Mean +/- Std": f"{df[c].mean():.4f} +/- {df[c].std():.4f}"} for c in METRIC_COLS if c in df.columns]
    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 55 + f"\n   GREEDY TRACER — {dataset_name} (N={len(dataset)})\n" + "=" * 55)
    print(summary_df.to_string(index=False))
    print("=" * 55)

    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "metrics_per_image.csv"), index=False)
    return df

if __name__ == "__main__":
    for name in TEST_DATASETS:
        evaluate(name)
