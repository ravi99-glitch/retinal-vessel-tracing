"""
scripts/drive_rl_tracing.py
=========================
End-to-end inference: SeedDetector → FrontierTracer → F1 evaluation.
"""

import os
import sys
import csv
from tqdm import tqdm
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import skeletonize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_models.policy_network import ActorCriticNetwork
from rl_models.seed_detector import SeedDetector
from rl_environment.vessel_env import VesselTracingEnv
from rl_environment.frontier_tracer import FrontierTracer
from rl_environment.seeding_utils import merge_seeds

from data.dataloader import load_dataset
from data.dataset_paths import get_root, WEIGHTS_DIR, OUTPUT_DIR as _OUTPUT_BASE
from evaluation.metrics import CenterlineMetrics

# ==========================================
# MODE — switch between gt and e2e
# ==========================================
MODE = 'e2e'   # 'gt' | 'e2e'

# ==========================================
# PATHS
# ==========================================
DRIVE_ROOT      = str(get_root("DRIVE"))
PPO_WEIGHTS     = str(WEIGHTS_DIR / "ppo_policy.pt")
SEED_WEIGHTS    = str(WEIGHTS_DIR / "seed_detector.pt")
OUTPUT_DIR      = str(_OUTPUT_BASE / "RL_tracing_seeddetector_DRIVE")

TOLERANCE       = 2.0
OBS_SIZE        = 65
MAX_STEPS       = 2000
MAX_TRACES      = 80
MIN_COV_GAIN    = 0.001
TEST_IDS = ["38_training", "39_training", "40_training"]

# Morphological post-processing params
DILATION_RADIUS = 3

# FOV-ring peripheral seeding params
N_RING_SEEDS  = 24
RING_INSET_PX = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# METRICS
# ==========================================
metrics_calc = CenterlineMetrics(tolerance_levels=[1, 2, 3])

# Standardised metric columns — shared across all baseline scripts
METRIC_COLS = [
    "iou",
    "clDice",
    "betti_0_error_raw",
    "betti_0_error_postproc",
    "hd95",
    "f1@1px",    "precision@1px", "recall@1px",
    "f1@2px",    "precision@2px", "recall@2px",
    "f1@3px",    "precision@3px", "recall@3px",
]

CSV_COLUMNS = ["image_id"] + METRIC_COLS

# ==========================================
# CONFIG
# ==========================================

PPO_CONFIG = {
    'policy': {
        'hidden_dim':    128,
        'lstm_hidden':   128,
        'use_lstm':      False,
        'dropout':       0.0,
        'encoder_type':  'cnn',
    },
    'environment': {
        'observation_size':      OBS_SIZE,
        'tolerance':             TOLERANCE,
        'use_vesselness':        False,
        'max_steps_per_episode': 600,
        'max_off_track_streak':  5,
        'step_size':             2,
    },
    'reward': {
        'alpha_near':            0.1,
        'beta_coverage':         1.0,
        'gamma_off':            -0.5,
        'lambda_revisit':       -2.0,
        'step_cost':            -0.01,
        'direction_bonus':       0.05,
        'terminal_f1_weight':    5.0,
        'use_potential_shaping': False,
    },
    'training': {'ppo': {'gamma': 0.99}},
}

SEED_CONFIG = {
    'seed_detector': {
        'base_ch':              16,
        'nms_radius':           15,
        'confidence_threshold': 0.3,
        'top_k_seeds':          MAX_TRACES,
    }
}


# ==========================================
# DATA LOADING
# ==========================================

def _load_all_samples():
    """Load all samples via the unified dataloader, return dict keyed by image ID."""
    ds, _ = load_dataset(DRIVE_ROOT, "DRIVE", target="rl_agent", tolerance=TOLERANCE)
    samples = {}
    for i in range(len(ds)):
        s = ds[i]
        samples[s['id']] = {
            'id':                 s['id'],
            'image_orig':         s['image_orig'].permute(1, 2, 0).numpy(),
            'image':              s['image'].permute(1, 2, 0).numpy(),
            'vessel_mask':        (s['vessel_mask'].squeeze(0).numpy() > 0).astype(np.uint8),
            'centerline':         s['centerline'].squeeze(0).numpy(),
            'distance_transform': s['distance_transform'].squeeze(0).numpy(),
            'fov_mask':           s['fov_mask'].squeeze(0).numpy(),
        }
    return samples


# ==========================================
# POST-PROCESSING (BETTI-0 ONLY)
# ==========================================

def postprocess_skeleton(traced: np.ndarray, dilation_radius: int = DILATION_RADIUS) -> np.ndarray:
    """
    Merge disconnected RL path segments into a cleaner skeleton.

    Steps:
      1. Binary dilation — bridges small gaps between nearby path endpoints.
      2. Re-skeletonize — thins the dilated mask back to 1px-wide centerlines.

    This does NOT affect F1 / HD95 / IoU (computed on the raw traced map).
    It is used only for Betti-0 post-processed reporting.
    """
    binary = (traced > 0).astype(np.uint8)
    r  = dilation_radius
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dilated = cv2.dilate(binary, se, iterations=1)
    thinned = skeletonize(dilated > 0).astype(np.uint8)
    return thinned


# ==========================================
# GT-BASED SEED PICKING  (MODE='gt')
# ==========================================

def _pick_frontier_seed_gt(gt_centerline, covered, half):
    uncovered = (gt_centerline > 0) & (covered == 0)
    if not uncovered.any():
        return None

    uncovered_pts = np.argwhere(uncovered)
    h, w = gt_centerline.shape
    covered_bin = (covered > 0).astype(np.uint8)

    if covered_bin.any():
        dist   = cv2.distanceTransform(1 - covered_bin, cv2.DIST_L2, 5)
        scores = dist[uncovered_pts[:, 0], uncovered_pts[:, 1]]
        best   = uncovered_pts[np.argmax(scores)]
    else:
        centre = np.array([h // 2, w // 2])
        dists  = np.linalg.norm(uncovered_pts - centre, axis=1)
        best   = uncovered_pts[np.argmin(dists)]

    y = int(np.clip(best[0], half + 5, h - half - 6))
    x = int(np.clip(best[1], half + 5, w - half - 6))
    return (y, x)


# ==========================================
# GT MODE TRACING
# ==========================================

def trace_gt_mode(ppo_model, sample):
    env = VesselTracingEnv(PPO_CONFIG)
    env.set_data(
        image=sample['image'],
        centerline=sample['centerline'],
        distance_transform=sample['distance_transform'],
        fov_mask=sample['fov_mask'],
    )

    h, w     = sample['image'].shape[:2]
    half     = OBS_SIZE // 2
    combined = np.zeros((h, w), dtype=np.float32)
    paths    = []
    gt_total = float(max(sample['centerline'].sum(), 1))

    ppo_model.eval()
    with torch.no_grad():
        for trace_idx in tqdm(range(MAX_TRACES), desc=f"Img {sample['id']} Tracing", unit="trace", leave=False):
            start = _pick_frontier_seed_gt(sample['centerline'], combined, half)
            if start is None:
                tqdm.write(f"    Full GT coverage after {trace_idx} traces.")
                break

            obs, _         = env.reset(start_position=start)
            path           = [start]
            covered_before = combined.sum()
            done           = False

            while not done:
                obs_t        = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                logits, _, _ = ppo_model(obs_t)
                action       = logits.argmax(dim=-1).item()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                y, x = env.position
                path.append((y, x))
                combined[y, x] = 1.0

            gain         = (combined.sum() - covered_before) / gt_total
            coverage_pct = combined.sum() / gt_total

            tqdm.write(f"    Trace {trace_idx+1:3d} gain={gain:.3f} coverage={coverage_pct:.3f}")
            paths.append(path)

            if trace_idx >= 3 and gain < MIN_COV_GAIN:
                tqdm.write(f"    Early stop: gain {gain:.4f} < {MIN_COV_GAIN}")
                break

    return combined, paths


# ==========================================
# E2E MODE TRACING
# ==========================================

def trace_e2e_mode(ppo_model, seed_model, sample):
    img_t = torch.from_numpy(
        sample['image'].transpose(2, 0, 1)
    ).unsqueeze(0).float().to(DEVICE)

    fov_t = torch.from_numpy(sample['fov_mask']).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    batch_seeds, _ = seed_model.detect_seeds(
        img_t,
        obs_half=OBS_SIZE // 2,
        return_heatmap=True,
        fov_mask=fov_t
    )
    seeds = batch_seeds[0]
    tqdm.write(f"    Seed detector: {len(seeds)} seeds predicted")

    if not seeds:
        tqdm.write("    WARNING: No seeds found, falling back to image centre")
        h, w = sample['image'].shape[:2]
        seeds = [(h // 2, w // 2, 0.5)]

    merged, n_ring_added = merge_seeds(
        detector_seeds = seeds,
        fov_mask       = sample["fov_mask"],
        max_traces     = MAX_TRACES,
        n_ring_seeds   = N_RING_SEEDS,
        inset_px       = RING_INSET_PX,
        obs_half       = OBS_SIZE // 2,
    )
    tqdm.write(
        f"    Detector seeds (capped): {MAX_TRACES - N_RING_SEEDS}  "
        f"Ring seeds added: {n_ring_added}  "
        f"Total: {len(merged)}"
    )

    env = VesselTracingEnv(PPO_CONFIG)
    tracer = FrontierTracer(env, ppo_model, DEVICE, obs_size=OBS_SIZE)

    combined, paths = tracer.trace_from_seeds(sample, merged)

    return combined, paths


# ==========================================
# VISUALISATION
# ==========================================

def make_overlay(image_orig, gt_centerline, traced, paths):
    """Creates a darkened grayscale background with colored traces over it."""
    gray      = cv2.cvtColor((image_orig * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dark_gray = (gray * 0.4).astype(np.uint8)
    overlay   = cv2.cvtColor(dark_gray, cv2.COLOR_GRAY2RGB)

    overlay[gt_centerline > 0]                         = [0,   200,  0]
    overlay[traced > 0]                                = [220,  50, 50]
    overlay[(gt_centerline > 0) & (traced > 0)]        = [255, 220,  0]
    for path in paths:
        if path:
            y, x = path[0]
            cv2.circle(overlay, (x, y), 4, (0, 255, 255), -1)
    return overlay


def visualize_sample(ppo_model, seed_model, sample, output_dir):
    img_id = sample['id']
    tqdm.write(f"\nProcessing Image {img_id} [Mode: {MODE}]")

    if MODE == 'gt':
        traced, paths = trace_gt_mode(ppo_model, sample)
    else:
        traced, paths = trace_e2e_mode(ppo_model, seed_model, sample)

    pred_skel = (traced > 0).astype(np.uint8)
    gt_skel   = (sample['centerline'] > 0).astype(np.uint8)

    # ----------------------------------------------------------
    # Metrics on RAW traced skeleton
    # pred_vessel_mask = pred_skel (RL produces a skeleton, not a filled mask)
    # ----------------------------------------------------------
    metrics = metrics_calc.compute_all_metrics(
        pred_skeleton    = pred_skel,
        gt_skeleton      = gt_skel,
        pred_vessel_mask = pred_skel,
        gt_vessel_mask   = sample['vessel_mask'],
        fov_mask         = sample['fov_mask'],
    )
    metrics['image_id']          = img_id
    metrics['betti_0_error_raw'] = metrics.pop('betti_0_error')

    # ----------------------------------------------------------
    # Post-processed skeleton — Betti-0 only
    # ----------------------------------------------------------
    postproc_skel = postprocess_skeleton(traced)
    if sample['fov_mask'] is not None:
        postproc_skel = postproc_skel * (sample['fov_mask'] > 0)

    metrics['betti_0_error_postproc'] = metrics_calc.betti_0_error(postproc_skel, gt_skel)

    n_traces_used = len(paths)
    tqdm.write(
        f"  F1@2={metrics['f1@2px']:.3f}  "
        f"P@2={metrics['precision@2px']:.3f}  "
        f"R@2={metrics['recall@2px']:.3f}  "
        f"IoU={metrics.get('iou', 0):.3f}  "
        f"HD95={metrics['hd95']:.1f}px  "
        f"Betti0(raw)={metrics['betti_0_error_raw']:.0f}  "
        f"Betti0(post)={metrics['betti_0_error_postproc']:.0f}"
    )

    overlay = make_overlay(sample['image_orig'], sample['centerline'], traced, paths)

    # 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    title_str = (
        f"Image {img_id}  |  "
        f"F1@2={metrics['f1@2px']:.3f}  "
        f"P@2={metrics['precision@2px']:.3f}  "
        f"R@2={metrics['recall@2px']:.3f}  "
        f"IoU={metrics.get('iou', 0):.3f}  "
        f"HD95={metrics['hd95']:.1f}px  "
        f"Betti0 raw={metrics['betti_0_error_raw']:.0f} → post={metrics['betti_0_error_postproc']:.0f}  "
        f"({n_traces_used} traces)"
    )
    fig.suptitle(title_str, fontsize=12, fontweight='bold')

    axes[0].imshow(sample['image_orig'])
    axes[0].set_title("(a) Original RGB Fundus", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(sample['centerline'], cmap='gray')
    axes[1].set_title("(b) GT Centerline", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(traced, cmap='gray')
    axes[2].set_title(f"(c) Agent Traced ({n_traces_used} paths)", fontsize=10)
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title("(d) Overlay (TP / GT-miss / FP / Seeds)", fontsize=10)
    axes[3].axis('off')

    legend = [
        mpatches.Patch(color='#00C800', label='GT only (miss)'),
        mpatches.Patch(color='#FFDC00', label='True positive'),
        mpatches.Patch(color='#DC3232', label='Traced only (FP)'),
        mpatches.Patch(color='#00FFFF', label='Seed point'),
    ]
    axes[3].legend(handles=legend, loc='lower right', fontsize=8,
                   framealpha=0.8, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"trace_{img_id}_{MODE}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    tqdm.write(f"  Saved → {out_path}")

    return metrics


# ==========================================
# CSV HELPERS
# ==========================================

def init_csv(csv_path: str):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()


def append_csv(csv_path: str, metrics: dict):
    row = {col: metrics.get(col, '') for col in CSV_COLUMNS}
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


# ==========================================
# MAIN
# ==========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}  |  Mode: {MODE}  |  Dilation radius: {DILATION_RADIUS}px")

    csv_path = os.path.join(OUTPUT_DIR, f"metrics_{MODE}.csv")
    init_csv(csv_path)
    print(f"CSV → {csv_path}")

    # Load PPO model
    ppo_ckpt  = torch.load(PPO_WEIGHTS, map_location=DEVICE, weights_only=True)
    ppo_model = ActorCriticNetwork(PPO_CONFIG).to(DEVICE)
    ppo_model.load_state_dict(ppo_ckpt['model_state_dict'])
    ppo_model.eval()

    # Load seed detector
    seed_model = None
    if MODE == 'e2e':
        seed_ckpt  = torch.load(SEED_WEIGHTS, map_location=DEVICE, weights_only=True)
        seed_model = SeedDetector(SEED_CONFIG).to(DEVICE)
        seed_model.load_state_dict(seed_ckpt['model_state_dict'])
        seed_model.eval()

    all_samples = _load_all_samples()
    all_metrics = []

    for img_id in tqdm(TEST_IDS, desc="Total Benchmark", unit="img"):
        if img_id not in all_samples:
            tqdm.write(f"  Skipping {img_id} — not found in dataset.")
            continue
        sample  = all_samples[img_id]
        metrics = visualize_sample(ppo_model, seed_model, sample, OUTPUT_DIR)

        append_csv(csv_path, metrics)
        all_metrics.append(metrics)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    if all_metrics:
        print("\n" + "=" * 65)
        print(f"SUMMARY  ({len(all_metrics)} images, mode={MODE}, dilation={DILATION_RADIUS}px)")
        print("=" * 65)
        for k in METRIC_COLS:
            vals = [m[k] for m in all_metrics if k in m]
            if vals:
                print(f"  {k:<28s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
        print("=" * 65)

        # Summary CSV (mean ± std)
        summary_rows = [
            {"Metric": k,
             "Mean +/- Std": f"{np.mean([m[k] for m in all_metrics if k in m]):.4f} +/- "
                             f"{np.std([m[k] for m in all_metrics if k in m]):.4f}"}
            for k in METRIC_COLS
            if any(k in m for m in all_metrics)
        ]
        import pandas as pd
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(OUTPUT_DIR, f"metrics_summary_{MODE}.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Summary CSV → {summary_csv}")

    print(f"\nOutput directory : {OUTPUT_DIR}")
    print(f"Per-image CSV    : {csv_path}")


if __name__ == "__main__":
    main()
