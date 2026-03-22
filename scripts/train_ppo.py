# scripts/train_ppo.py
"""PPO Training — Step 2 of 2. Run AFTER train_imitation.py.

All PPO logic lives in rl_training/ppo.py.
This script handles: paths, config, and data loading via the unified dataloader.
"""

import os
import sys
from typing import Dict, List

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import WEIGHTS_DIR, get_data
from models.policy_network import ActorCriticNetwork
from training.ppo import PPOTrainer

# ==========================================
# CONFIG
# ==========================================
SAVE_PATH = str(WEIGHTS_DIR / "ppo_policy.pt")
IMITATION_WEIGHTS = str(WEIGHTS_DIR / "imitation_policy.pt")
LOG_PATH = str(WEIGHTS_DIR / "ppo_log.txt")

TOLERANCE = 2.0
OBS_SIZE = 65
MAX_STEPS = 2000
USE_VESSELNESS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "policy": {
        "hidden_dim": 128,
        "lstm_hidden": 128,
        "use_lstm": False,
        "dropout": 0.05,
        "encoder_type": "cnn",
    },
    "environment": {
        "observation_size": OBS_SIZE,
        "tolerance": TOLERANCE,
        "use_vesselness": USE_VESSELNESS,
        "max_steps_per_episode": MAX_STEPS,
        "max_off_track_streak": 8,
        "step_size": 1,
    },
    "reward": {
        "alpha_near": 0.1,
        "beta_coverage": 1.0,
        "gamma_off": -0.5,
        "lambda_revisit": -1.0,
        "step_cost": -0.01,
        "direction_bonus": 0.05,
        "terminal_f1_weight": 5.0,
        "use_potential_shaping": False,
    },
    "training": {"ppo": {"gamma": 0.99}},
}


# ==========================================
# DATA LOADING (unified dataloader)
# ==========================================


def dataloader_to_env_samples(dataset) -> List[Dict]:
    """Convert dataloader samples (torch tensors) to env-compatible numpy dicts."""
    samples = []
    for i in range(len(dataset)):
        s = dataset[i]
        samples.append(
            {
                "id": s["id"],
                "image": s["image"].permute(1, 2, 0).numpy(),  # (3,H,W) → (H,W,3)
                "centerline": s["centerline"].squeeze(0).numpy(),  # (1,H,W) → (H,W)
                "distance_transform": s["distance_transform"]
                .squeeze(0)
                .numpy(),  # (1,H,W) → (H,W)
                "fov_mask": s["fov_mask"].squeeze(0).numpy(),  # (1,H,W) → (H,W)
            }
        )
    return samples


def load_samples(split: str) -> List[Dict]:
    ds, _ = get_data(
        "rl_agent",
        split,
        tolerance=TOLERANCE,
    )
    samples = dataloader_to_env_samples(ds)
    print(f"Loaded {len(samples)} {split} samples (combined dataset).")
    for s in samples:
        print(f"  [{s['id']}] centerline px: {int(s['centerline'].sum())}")
    return samples


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
        mini_batch_size=256,
        steps_per_iter=4096,
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


if __name__ == "__main__":
    main()
