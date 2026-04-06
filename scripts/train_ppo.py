# scripts/train_ppo_resnet.py
"""PPO Training — Step 2 of 2. Run AFTER train_imitation.py.

All PPO logic lives in rl_training/ppo.py.
This script handles: paths, config, and data loading via the unified dataloader.
"""

import os
import sys
import re
from typing import Dict, List

import torch
import numpy as np
import cv2 
import matplotlib
matplotlib.use("Agg")  # Use headless backend for SLURM
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import WEIGHTS_DIR, get_data
from models.policy_network import ActorCriticNetwork
from training.ppo import PPOTrainer

# ==========================================
# CONFIG
# ==========================================
SAVE_PATH = str(WEIGHTS_DIR / "ppo_policy_resnet.pt")
IMITATION_WEIGHTS = str(WEIGHTS_DIR / "imitation_policy_resnet.pt")
LOG_PATH = str(WEIGHTS_DIR / "ppo_training_log.txt")
PLOT_PATH = str(WEIGHTS_DIR / "ppo_learning_curve.png")

TOLERANCE = 2.0
OBS_SIZE = 65
MAX_STEPS = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "policy": {
        "hidden_dim": 128,
        "lstm_hidden": 128,
        "use_lstm": False,
        "dropout": 0.05,
        "encoder_type": "resnet",
    },
    "environment": {
        "observation_size": OBS_SIZE,
        "tolerance": TOLERANCE,
        "max_steps_per_episode": MAX_STEPS,
        "max_off_track_streak": 8,
        "step_size": 1,
    },
    "reward": {
        "alpha_near": 0.5,
        "beta_coverage": 2.0,
        "gamma_off": -1.0,
        "lambda_revisit": -0.5,     
        "step_cost": -0.01,
        "direction_bonus": 0.05,    
        "terminal_f1_weight": 5.0,
        "smoothness_penalty": -0.05,
        "use_potential_shaping": False,
    },
    "training": {"ppo": {"gamma": 0.99}},
}

# ==========================================
# DATA LOADING (DataLoader)
# ==========================================

def dataloader_to_env_samples(dataset, limit: int = None) -> List[Dict]:
    """Convert dataloader samples (torch tensors) to env-compatible numpy dicts."""
    samples = []
    
    n_samples = len(dataset) if limit is None else min(limit, len(dataset))
    
    for i in range(n_samples):
        s = dataset[i]
        
        fov_mask = (s["fov_mask"].squeeze(0).numpy() > 0).astype(np.uint8)
        dt = s["distance_transform"].squeeze(0).numpy()
        vessel_mask = s["vessel_mask"].squeeze(0).numpy()

        inv_mask = (vessel_mask == 0).astype(np.uint8)
        pixel_dt = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 3)
        dt[pixel_dt > 4.0] = 100.0  # Instant death if 4 pixels off-track
        dt[fov_mask == 0] = 100.0   # Instant death if outside FOV
        
        samples.append(
            {
                "id": s["id"],
                "image": s["image"].permute(1, 2, 0).numpy(), 
                "centerline": s["centerline"].squeeze(0).numpy(), 
                "distance_transform": dt, 
                "fov_mask": fov_mask, 
                "vessel_mask": vessel_mask,
                "vessel_orientation": s["vessel_orientation"].numpy(),
                "dt_gradient": s["dt_gradient"].numpy(),
            }
        )

    return samples

    
def load_samples(split: str, limit: int = None) -> List[Dict]:
    ds, _ = get_data(
        "rl_agent",
        split,
        tolerance=TOLERANCE,
    )
    samples = dataloader_to_env_samples(ds, limit=limit)
    print(f"Loaded {len(samples)} {split} samples (combined dataset).")
    for s in samples:
        print(f"  [{s['id']}] centerline px: {int(s['centerline'].sum())}")
    return samples


# ==========================================
# VISUALIZATION
# ==========================================
def plot_training_curve(log_file: str, save_file: str):
    """Parses the PPO log file and generates a learning curve graph."""
    print(f"\nGenerating training curve graph from {log_file}...")
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    iters, rewards, val_iters, val_f1s = [], [], [], []

    iter_re = re.compile(r"Iter\s+(\d+)/")
    reward_re = re.compile(r"reward=\s*(-?\d+\.\d+)")
    f1_re = re.compile(r"val_f1=\s*(\d+\.\d+)")

    with open(log_file, "r") as f:
        for line in f:
            i_match = iter_re.search(line)
            r_match = reward_re.search(line)
            f_match = f1_re.search(line)

            if i_match and r_match:
                iteration = int(i_match.group(1))
                iters.append(iteration)
                rewards.append(float(r_match.group(1)))

                if f_match:
                    val_iters.append(iteration)
                    val_f1s.append(float(f_match.group(1)))

    if not iters:
        print("No data found in logs to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("PPO Iterations", fontweight='bold')
    ax1.set_ylabel("Mean Episode Reward", color="tab:blue", fontweight='bold')
    ax1.plot(iters, rewards, color="tab:blue", alpha=0.7, label="Reward")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.6)

    if val_f1s:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Validation F1 Score (@2px)", color="tab:red", fontweight='bold')
        ax2.plot(val_iters, val_f1s, color="tab:red", marker="o", linewidth=2, label="Val F1")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("PPO Agent Training Progress", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    plt.savefig(save_file, dpi=150)
    print(f"Graph successfully saved to: {save_file}")


# ==========================================
# MAIN
# ==========================================
def main():
    print(f"Device: {DEVICE}")

    train_samples = load_samples("train")  
    val_samples = load_samples("val")      

    if not train_samples:
        print("ERROR: No training samples loaded.")
        return

    model = ActorCriticNetwork(CONFIG).to(DEVICE)
    trainer = PPOTrainer(
        model,
        CONFIG,
        DEVICE,
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.1,
        entropy_coef=0.05,
        value_coef=0.5,
        max_grad_norm=1.0,
        ppo_epochs=4,
        mini_batch_size=1024,
        steps_per_iter=2048,
        num_iterations=1000,
        eval_every=25,
        save_every=50,
        tolerance=TOLERANCE,
    )

    trainer.train(
        train_samples=train_samples,
        val_samples=val_samples,
        save_path=SAVE_PATH,
        log_path=LOG_PATH,
        imitation_path=IMITATION_WEIGHTS,
    )

    # --- Generate the visualization after training finishes ---
    plot_training_curve(LOG_PATH, PLOT_PATH)


if __name__ == "__main__":
    main()
