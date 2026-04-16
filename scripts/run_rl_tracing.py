# scripts/run_rl_tracing_resnet.py
"""
End-to-end inference: SeedDetector → FrontierTracer → F1 evaluation.
(ResNet Version)
"""

import csv
import os
import sys
import cv2
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import OUTPUT_DIR as _OUTPUT_BASE
from data.dataloader import TEST_DATASETS, WEIGHTS_DIR, get_test_data
from evaluation.metrics import CenterlineMetrics
from environment.frontier_tracer import FrontierTracer
from environment.seeding_utils import merge_seeds
from environment.vessel_env import VesselTracingEnv
from models.policy_network import ActorCriticNetwork
from models.seed_detector import SeedDetector

# ==========================================
# MODE — switch between gt and e2e
# ==========================================
MODE = "e2e"  # 'gt' | 'e2e'

# ==========================================
# PATHS
# ==========================================
PPO_WEIGHTS = str(WEIGHTS_DIR / "ppo_policy_resnet.pt")
SEED_WEIGHTS = str(WEIGHTS_DIR / "seed_detector.pt")

TOLERANCE = 2.0
OBS_SIZE = 65
MAX_STEPS = 2000

MAX_TRACES = 80
MIN_PATH_LENGTH = 15
MIN_COV_GAIN = 0.001

# FOV-ring peripheral seeding params
N_RING_SEEDS = 0
RING_INSET_PX = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# METRICS
# ==========================================
metrics_calc = CenterlineMetrics(tolerance_levels=[1, 2, 3])

METRIC_COLS = [
    "iou", "clDice", "betti_0_error_raw", "hd95",
    "f1@1px", "precision@1px", "recall@1px",
    "f1@2px", "precision@2px", "recall@2px",
    "f1@3px", "precision@3px", "recall@3px",
]
CSV_COLUMNS = ["image_id"] + METRIC_COLS

# ==========================================
# CONFIG
# ==========================================
PPO_CONFIG = {
    "policy": {
        "hidden_dim": 128,
        "dropout": 0.0,
        "encoder_type": "resnet",
    },
    "environment": {
        "observation_size": OBS_SIZE,
        "tolerance": TOLERANCE,
        "max_steps_per_episode": 2000,
        "max_off_track_streak": 5,      # <--- RESTORED BASELINE
        "step_size": 1,
    },
    "reward": {
        "alpha_near": 0.5,
        "beta_coverage": 2.0,
        "gamma_off": -1.0,
        "lambda_revisit": -5.0,
        "step_cost": -0.01,             # <--- RESTORED BASELINE
        "direction_bonus": 0.05,
        "terminal_f1_weight": 2.5,
        "terminal_cldice_weight": 5.0,
        "smoothness_penalty": -0.05,
        "use_potential_shaping": False,
    },
    "training": {"ppo": {"gamma": 0.99}},
}

SEED_CONFIG = {
    "seed_detector": {
        "base_ch": 16,
        "nms_radius": 5,
        "confidence_threshold": 0.05,
        "top_k_seeds": MAX_TRACES,
    }
}

# ==========================================
# DATA LOADING
# ==========================================
def _load_all_samples(dataset_name):
    from data.dataloader import DATASET_REGISTRY
    from data.fundus_preprocessor import FundusPreprocessor 
    
    ds, _ = get_test_data(dataset_name, "rl_agent", tolerance=TOLERANCE)
    cfg = DATASET_REGISTRY.get(dataset_name.upper(), None)
    no_fov = cfg.no_fov if cfg else False

    preprocessor = FundusPreprocessor() 

    samples = {}
    for i in range(len(ds)):
        s = ds[i]
        
        # 1. raw RGB numpy array
        raw_rgb = s["image_orig"].permute(1, 2, 0).numpy()
        fov_mask = s["fov_mask"].squeeze(0).numpy()
        # 2. Apply the same CLAHE-based green channel enhancement as in training
        enhanced_green = preprocessor.preprocess(raw_rgb, external_mask=fov_mask)
        # 3. Create the 3-channel input for the agent, normalizing to [0, 1]
        img_f = raw_rgb.astype(np.float32) / 255.0 
        # Overwrite the green channel with the CLAHE version
        img_f[:, :, 1] = enhanced_green

        samples[s["id"]] = {
            "id": s["id"],
            "image_orig": raw_rgb,
            "image": img_f,
            "vessel_mask": (s["vessel_mask"].squeeze(0).numpy() > 0).astype(np.uint8),
            "centerline": s["centerline"].squeeze(0).numpy(),
            "distance_transform": s["distance_transform"].squeeze(0).numpy(),
            "fov_mask": fov_mask,
            "vessel_orientation": s["vessel_orientation"].numpy(),
            "dt_gradient": s["dt_gradient"].numpy(),
        }
    return samples, no_fov

# ==========================================
# PATH SMOOTHING
# ==========================================
def smooth_paths_and_redraw(paths, shape, window=7):
    h, w = shape
    smooth_traced = np.zeros((h, w), dtype=np.float32)
    smoothed_paths = []
    
    for p in paths:
        if len(p) < window:
            smoothed_paths.append(p)
            if p:
                coords = np.array(p, dtype=np.int32)
                smooth_traced[coords[:, 0], coords[:, 1]] = 1.0
            continue
            
        pts = np.array(p, dtype=np.float32)
        pad_start = np.repeat(pts[0:1], window//2, axis=0)
        pad_end = np.repeat(pts[-1:], window - 1 - window//2, axis=0)
        padded = np.vstack((pad_start, pts, pad_end))
        
        y_sm = np.convolve(padded[:, 0], np.ones(window)/window, mode='valid')
        x_sm = np.convolve(padded[:, 1], np.ones(window)/window, mode='valid')
        
        sm_path = np.column_stack((y_sm, x_sm)).astype(np.int32)
        sm_path[:, 0] = np.clip(sm_path[:, 0], 0, h - 1)
        sm_path[:, 1] = np.clip(sm_path[:, 1], 0, w - 1)
        
        smoothed_paths.append(sm_path)
        
        pts_cv = sm_path[:, ::-1].reshape((-1, 1, 2))
        cv2.polylines(smooth_traced, [pts_cv], isClosed=False, color=1.0, thickness=1)
        
    return smooth_traced, smoothed_paths

def _pick_frontier_seed_gt(gt_centerline, covered, half):
    uncovered = (gt_centerline > 0) & (covered == 0)
    if not uncovered.any():
        return None

    uncovered_pts = np.argwhere(uncovered)
    h, w = gt_centerline.shape
    covered_bin = (covered > 0).astype(np.uint8)

    if covered_bin.any():
        dist = cv2.distanceTransform(1 - covered_bin, cv2.DIST_L2, 5)
        scores = dist[uncovered_pts[:, 0], uncovered_pts[:, 1]]
        best = uncovered_pts[np.argmax(scores)]
    else:
        centre = np.array([h // 2, w // 2])
        dists = np.linalg.norm(uncovered_pts - centre, axis=1)
        best = uncovered_pts[np.argmin(dists)]

    y = int(np.clip(best[0], half + 5, h - half - 6))
    x = int(np.clip(best[1], half + 5, w - half - 6))
    return (y, x)

def trace_gt_mode(ppo_model, sample):
    env = VesselTracingEnv(PPO_CONFIG)
    env.set_data(
        image=sample["image"],
        centerline=sample["centerline"],
        distance_transform=sample["distance_transform"],
        fov_mask=sample["fov_mask"],
        vesselness=sample["vessel_mask"],
        vessel_orientation=sample.get("vessel_orientation"),
        dt_gradient=sample.get("dt_gradient"),
    )

    h, w = sample["image"].shape[:2]
    half = OBS_SIZE // 2
    combined = np.zeros((h, w), dtype=np.float32)
    paths = []
    gt_total = float(max(sample["centerline"].sum(), 1))

    ppo_model.eval()
    with torch.no_grad():
        for trace_idx in tqdm(range(MAX_TRACES), desc=f"Img {sample['id']} Tracing", unit="trace", leave=False):
            start = _pick_frontier_seed_gt(sample["centerline"], combined, half)
            if start is None:
                tqdm.write(f"    Full GT coverage after {trace_idx} traces.")
                break

            obs, _ = env.reset(start_position=start)
            path = [start]
            covered_before = combined.sum()
            done = False

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                if hasattr(ppo_model, 'get_action_and_value'):
                    out = ppo_model.get_action_and_value(obs_t)
                    action = out[0].item()
                else:
                    logits, _ = ppo_model(obs_t)
                    logits[0, 8] = -float("inf")  # mask out "stay put" action
                    action = logits.argmax(dim=-1).item()
                    
                obs, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated
                y, x = env.position
                path.append((y, x))
                combined[y, x] = 1.0

            gain = (combined.sum() - covered_before) / gt_total
            coverage_pct = combined.sum() / gt_total

            tqdm.write(f"    Trace {trace_idx+1:3d} gain={gain:.3f} coverage={coverage_pct:.3f}")
            paths.append(path)

            if trace_idx >= 3 and gain < MIN_COV_GAIN:
                tqdm.write(f"    Early stop: gain {gain:.4f} < {MIN_COV_GAIN}")
                break

    return combined, paths

def trace_e2e_mode(ppo_model, seed_model, sample, no_fov=False):
    mask_coverage = sample["fov_mask"].mean()
    if mask_coverage < 0.25:
        sample["fov_mask"] = (np.ones_like(sample["fov_mask"])).astype(np.uint8)
        no_fov = True 

    # --- APPLY DEATH ZONE IN EVAL ---
    sample["distance_transform"][sample["fov_mask"] == 0] = 100.0
    # --------------------------------

    img_t = torch.from_numpy(sample["image"].transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    fov_t = torch.from_numpy(sample["fov_mask"]).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    safe_fov_t = None if no_fov else fov_t

    batch_seeds, _ = seed_model.detect_seeds(
        img_t, obs_half=OBS_SIZE // 2, return_heatmap=True, fov_mask=safe_fov_t
    )
    seeds = batch_seeds[0]
    
    n_ring = 0 if no_fov else N_RING_SEEDS
    
    merged, n_ring_added = merge_seeds(
        detector_seeds=seeds, 
        fov_mask=sample["fov_mask"], 
        max_traces=MAX_TRACES,
        n_ring_seeds=n_ring, 
        inset_px=RING_INSET_PX, 
        obs_half=OBS_SIZE // 2,
    )

    # --- HARD FILTER: DELETE SEEDS IN THE VOID ---
    valid_merged = []
    for y, x in merged:
        if sample["fov_mask"][int(y), int(x)] > 0:
            valid_merged.append((y, x))
    merged = valid_merged
    # ---------------------------------------------

    env = VesselTracingEnv(PPO_CONFIG)
    tracer = FrontierTracer(env, ppo_model, DEVICE, obs_size=OBS_SIZE)
    combined, paths = tracer.trace_from_seeds(sample, merged)

    filtered_paths = [p for p in paths if len(p) >= MIN_PATH_LENGTH]
    
    h, w = sample["image"].shape[:2]
    filtered_combined = np.zeros((h, w), dtype=np.float32)
    
    if filtered_paths:
        # Stack all path coordinates into a single fast numpy array
        all_coords = np.vstack(filtered_paths).astype(np.int32)
        filtered_combined[all_coords[:, 0], all_coords[:, 1]] = 1.0

    return filtered_combined, filtered_paths, merged

# ==========================================
# TEST-TIME AUGMENTATION (TTA) WRAPPER
# ==========================================
def trace_e2e_tta(ppo_model, seed_model, sample, no_fov=False):
    """Runs the agent 3 times (Normal, H-Flip, V-Flip) and takes a majority vote."""
    h, w = sample["image"].shape[:2]

    def create_flipped_sample(s, flip_code):
        """flip_code: 1 = Horizontal, 0 = Vertical"""
        s_out = s.copy()
        
        if flip_code == 1:  # Horizontal
            s_out["image"] = s["image"][:, ::-1, :].copy()
            s_out["fov_mask"] = s["fov_mask"][:, ::-1].copy()
            s_out["vessel_mask"] = s["vessel_mask"][:, ::-1].copy()
            s_out["distance_transform"] = s["distance_transform"][:, ::-1].copy()
            s_out["centerline"] = s["centerline"][:, ::-1].copy()
        elif flip_code == 0:  # Vertical
            s_out["image"] = s["image"][::-1, :, :].copy()
            s_out["fov_mask"] = s["fov_mask"][::-1, :].copy()
            s_out["vessel_mask"] = s["vessel_mask"][::-1, :].copy()
            s_out["distance_transform"] = s["distance_transform"][::-1, :].copy()
            s_out["centerline"] = s["centerline"][::-1, :].copy()
            
        # Delete vectors so the environment automatically recomputes them for the flipped image!
        s_out.pop("vessel_orientation", None)
        s_out.pop("dt_gradient", None)
        return s_out

    # --- 1. Normal Run ---
    tqdm.write("      → [TTA 1/3] Standard orientation...")
    mask1, paths1, seeds1 = trace_e2e_mode(ppo_model, seed_model, sample, no_fov)

    # --- 2. Horizontal Flip Run ---
    tqdm.write("      → [TTA 2/3] Horizontal flip...")
    sample_h = create_flipped_sample(sample, 1)
    mask_h, paths_h, seeds_h = trace_e2e_mode(ppo_model, seed_model, sample_h, no_fov)
    
    # Flip the results back (Horizontal)
    mask2 = mask_h[:, ::-1].copy()
    paths2 = []
    for p in paths_h:
        arr = np.array(p, dtype=np.int32)
        arr[:, 1] = w - 1 - arr[:, 1]
        paths2.append(list(map(tuple, arr)))
    seeds2 = list(map(tuple, np.array(seeds_h, dtype=np.int32) * [1, -1] + [0, w - 1])) if seeds_h else []

    # --- 3. Vertical Flip Run ---
    tqdm.write("      → [TTA 3/3] Vertical flip...")
    sample_v = create_flipped_sample(sample, 0)
    mask_v, paths_v, seeds_v = trace_e2e_mode(ppo_model, seed_model, sample_v, no_fov)
    
    # Flip the results back (Vertical)
    mask3 = mask_v[::-1, :].copy()
    paths3 = []
    for p in paths_v:
        arr = np.array(p, dtype=np.int32)
        arr[:, 0] = h - 1 - arr[:, 0]
        paths3.append(list(map(tuple, arr)))
    seeds3 = list(map(tuple, np.array(seeds_v, dtype=np.int32) * [-1, 1] + [h - 1, 0])) if seeds_v else []

    # --- MAJORITY VOTE (The Golden Mean) ---
    final_mask = ((mask1 + mask2 + mask3) >= 2.0).astype(np.float32)
    
    # --- DILATED GEOGRAPHIC FILTER ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_vessel_map = cv2.dilate(sample["vessel_mask"], kernel, iterations=1)
    
    # Apply the forgiving gate
    final_mask = np.logical_and(final_mask, dilated_vessel_map > 0).astype(np.float32)
    
    # --- MORPHOLOGICAL BRIDGE BUILDER (THE 0.60+ CLDICE KEY) ---
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, close_kernel)
    
    # Combine paths and seeds for the visualization overlay
    final_paths = paths1 + paths2 + paths3
    final_seeds = seeds1 + seeds2 + seeds3

    return final_mask, final_paths, final_seeds

def make_overlay(image_orig, gt_centerline, traced, paths, all_seeds=None):
    gray = cv2.cvtColor((image_orig * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dark_gray = (gray * 0.4).astype(np.uint8)
    overlay = cv2.cvtColor(dark_gray, cv2.COLOR_GRAY2RGB)

    overlay[gt_centerline > 0] = [0, 200, 0]
    overlay[traced > 0] = [220, 50, 50]
    overlay[(gt_centerline > 0) & (traced > 0)] = [255, 220, 0]
    
    if all_seeds is not None:
        for y, x in all_seeds:
            cv2.circle(overlay, (int(x), int(y)), 2, (255, 0, 255), -1)
            
    for path in paths:
        if len(path) > 0:
            y, x = path[0]
            cv2.circle(overlay, (int(x), int(y)), 4, (0, 255, 255), -1)
    return overlay

def visualize_sample(ppo_model, seed_model, sample, output_dir, no_fov=False):
    img_id = sample["id"]
    tqdm.write(f"\nProcessing Image {img_id} [Mode: {MODE} w/ TTA]")
    
    inv_mask = (sample["vessel_mask"] == 0).astype(np.uint8)
    pixel_dt = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 3)
    sample["distance_transform"][pixel_dt > 4.0] = 100.0 

    all_seeds = None
    if MODE == "gt":
        traced, paths = trace_gt_mode(ppo_model, sample)
        traced, paths = smooth_paths_and_redraw(paths, sample["image"].shape[:2], window=7)
        all_seeds = [] 
    else:
        # 1. Get the clean, majority-voted mask from TTA
        traced, paths, all_seeds = trace_e2e_tta(ppo_model, seed_model, sample, no_fov=no_fov)
        
        # 2. Smooth the paths ONLY for the overlay dots, but DO NOT overwrite 'traced'!
        _, paths = smooth_paths_and_redraw(paths, sample["image"].shape[:2], window=7)
        
        # 3. SKELETONIZE: Shave the thick majority mask down to a perfect 1-pixel centerline!
        from skimage.morphology import skeletonize
        traced = skeletonize(traced > 0).astype(np.float32)

    pred_skel = (traced > 0).astype(np.uint8)
    gt_skel = (sample["centerline"] > 0).astype(np.uint8)

    metrics = metrics_calc.compute_all_metrics(
        pred_skeleton=pred_skel,
        gt_skeleton=gt_skel,
        pred_vessel_mask=pred_skel,
        gt_vessel_mask=sample["vessel_mask"],
        fov_mask=sample["fov_mask"],
    )
    metrics["image_id"] = img_id
    metrics["betti_0_error_raw"] = metrics.pop("betti_0_error")

    n_traces_used = len(paths)
    tqdm.write(
        f"  F1@2={metrics['f1@2px']:.3f}  "
        f"P@2={metrics['precision@2px']:.3f}  "
        f"R@2={metrics['recall@2px']:.3f}  "
        f"IoU={metrics.get('iou', 0):.3f}  "
        f"clDice={metrics.get('clDice', 0):.3f}  "
        f"HD95={metrics['hd95']:.1f}px  "
        f"Betti0={metrics['betti_0_error_raw']:.0f}"
    )

    overlay = make_overlay(sample["image_orig"], sample["centerline"], traced, paths, all_seeds=all_seeds)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    title_str = (
        f"Image {img_id}  |  "
        f"F1@2={metrics['f1@2px']:.3f}  "
        f"P@2={metrics['precision@2px']:.3f}  "
        f"R@2={metrics['recall@2px']:.3f}  "
        f"IoU={metrics.get('iou', 0):.3f}  "
        f"clDice={metrics.get('clDice', 0):.3f}  "
        f"HD95={metrics['hd95']:.1f}px  "
        f"Betti0={metrics['betti_0_error_raw']:.0f}  "
        f"({n_traces_used} traces)"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold")

    axes[0].imshow(sample["image_orig"])
    axes[0].set_title("(a) Original RGB Fundus", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(sample["centerline"], cmap="gray")
    axes[1].set_title("(b) GT Centerline", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(traced, cmap="gray")
    axes[2].set_title(f"(c) Agent Traced TTA ({n_traces_used} paths)", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("(d) Overlay (TP / GT-miss / FP / Seeds)", fontsize=10)
    axes[3].axis("off")

    legend = [
        mpatches.Patch(color="#00C800", label="GT only (miss)"),
        mpatches.Patch(color="#FFDC00", label="True positive"),
        mpatches.Patch(color="#DC3232", label="Traced only (FP)"),
        mpatches.Patch(color="#00FFFF", label="Seed Used"),
        mpatches.Patch(color="#FF00FF", label="Seed Predicted"),
    ]
    axes[3].legend(
        handles=legend, loc="lower right", fontsize=8, framealpha=0.8, ncol=2
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"trace_{img_id}_{MODE}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    tqdm.write(f"  Saved → {out_path}")

    return metrics

def init_csv(csv_path: str):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

def append_csv(csv_path: str, metrics: dict):
    row = {col: metrics.get(col, "") for col in CSV_COLUMNS}
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate on val set")
    parser.add_argument("--test", action="store_true", help="Test on external datasets")
    args = parser.parse_args()

    if not args.eval and not args.test:
        args.eval = args.test = True

    print(f"Device: {DEVICE}  |  Mode: {MODE}  |  Running Path Smoothing (w=7)")
    torch.set_float32_matmul_precision("high")

    ppo_ckpt = torch.load(PPO_WEIGHTS, map_location=DEVICE, weights_only=True)
    ppo_model = ActorCriticNetwork(PPO_CONFIG).to(DEVICE)
    
    state_dict = ppo_ckpt["model_state_dict"]
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    ppo_model.load_state_dict(clean_state_dict)
    
    ppo_model.eval()
    ppo_model = torch.compile(ppo_model)

    seed_model = None
    if MODE == "e2e":
        seed_ckpt = torch.load(SEED_WEIGHTS, map_location=DEVICE, weights_only=True)
        seed_model = SeedDetector(SEED_CONFIG).to(DEVICE)
        seed_model.load_state_dict(seed_ckpt["model_state_dict"])
        seed_model.eval()

    if args.eval:
        _run_on_datasets(ppo_model, seed_model, ("val",), label="val")

    if args.test:
        _run_on_datasets(ppo_model, seed_model, TEST_DATASETS, label="test")

def _run_on_datasets(ppo_model, seed_model, dataset_names, label="test"):
    for dataset_name in dataset_names:
        if dataset_name == "val":
            from data.dataloader import get_data
            ds, _ = get_data("rl_agent", "val", tolerance=TOLERANCE, resize=(512, 512))
            no_fov = False
            samples = {}
            for i in range(len(ds)):
                s = ds[i]
                samples[s["id"]] = {
                    "id": s["id"],
                    "image_orig": s["image_orig"].permute(1, 2, 0).numpy(),
                    "image": s["image"].permute(1, 2, 0).numpy(),
                    "vessel_mask": (s["vessel_mask"].squeeze(0).numpy() > 0).astype(np.uint8),
                    "centerline": s["centerline"].squeeze(0).numpy(),
                    "distance_transform": s["distance_transform"].squeeze(0).numpy(),
                    "fov_mask": s["fov_mask"].squeeze(0).numpy(),
                    "vessel_orientation": s["vessel_orientation"].numpy(),
                    "dt_gradient": s["dt_gradient"].numpy(),
                }
        else:
            samples, no_fov = _load_all_samples(dataset_name)

        output_dir = str(_OUTPUT_BASE / f"RL_tracing_{MODE}_resnet" / dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, f"metrics_{MODE}_resnet.csv")
        init_csv(csv_path)
        print(f"\n[{dataset_name}] CSV → {csv_path}")

        all_metrics = []

        for img_id in tqdm(samples.keys(), desc=f"RL Tracing — {dataset_name}", unit="img"):
            sample = samples[img_id]
            metrics = visualize_sample(ppo_model, seed_model, sample, output_dir, no_fov=no_fov)
            append_csv(csv_path, metrics)
            all_metrics.append(metrics)

        if all_metrics:
            print("\n" + "=" * 65)
            print(f"SUMMARY — {dataset_name}  ({len(all_metrics)} images, mode={MODE})")
            print("=" * 65)
            for k in METRIC_COLS:
                vals = [m[k] for m in all_metrics if k in m]
                if vals:
                    print(f"  {k:<28s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
            print("=" * 65)

            import pandas as pd
            summary_rows = [
                {
                    "Metric": k,
                    "Mean +/- Std": f"{np.mean([m[k] for m in all_metrics if k in m]):.4f} +/- "
                    f"{np.std([m[k] for m in all_metrics if k in m]):.4f}",
                }
                for k in METRIC_COLS
                if any(k in m for m in all_metrics)
            ]
            summary_df = pd.DataFrame(summary_rows)
            summary_csv = os.path.join(output_dir, f"metrics_summary_{MODE}_resnet.csv")
            summary_df.to_csv(summary_csv, index=False)
            print(f"Summary CSV → {summary_csv}")

if __name__ == "__main__":
    main()
