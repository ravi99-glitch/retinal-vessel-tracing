"""PPO Training — Step 2 of 2. Run AFTER train_imitation.py.

All PPO logic lives in training/ppo.py.
This script handles: paths, config, and data loading via the unified dataloader.

Supports both feedforward and LSTM policies via CONFIG["policy"]["use_lstm"].
"""

import os
import sys
import numpy as np
from typing import Dict, List

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE
from config import IMITATION_WEIGHTS_PATH as IMITATION_WEIGHTS
from config import LSTM_CHUNK_LENGTH, MAX_STEPS
from config import MODEL_CONFIG as CONFIG
from config import (
    OBS_SIZE,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_EVAL_EVERY,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
)
from config import PPO_LOG_PATH as LOG_PATH
from config import (
    PPO_LR,
    PPO_MAX_GRAD_NORM,
    PPO_MINI_BATCH_SIZE,
    PPO_NUM_ITERATIONS,
    PPO_SAVE_EVERY,
    PPO_STEPS_PER_ITER,
    PPO_VALUE_COEF,
)
from config import PPO_WEIGHTS_PATH as SAVE_PATH
from config import TOLERANCE
from data.dataloader import get_data
from models.policy_network import ActorCriticNetwork
from training.ppo import PPOTrainer

# ==========================================
# DATA LOADING (unified dataloader)
# ==========================================


# def dataloader_to_env_samples(dataset) -> List[Dict]:
#     """Convert dataloader samples (torch tensors) to env-compatible numpy dicts."""
#     samples = []
#     for i in range(len(dataset)):
#         s = dataset[i]
#         samples.append(
#             {
#                 "id": s["id"],
#                 "image": s["image"].permute(1, 2, 0).numpy(),  # (3,H,W) → (H,W,3)
#                 "centerline": s["centerline"].squeeze(0).numpy(),  # (1,H,W) → (H,W)
#                 "distance_transform": s["distance_transform"]
#                 .squeeze(0)
#                 .numpy(),  # (1,H,W) → (H,W)
#                 "fov_mask": s["fov_mask"].squeeze(0).numpy(),  # (1,H,W) → (H,W)
#             }
#         )
#     return samples


# def load_samples(split: str) -> List[Dict]:
#     ds, _ = get_data(
#         "rl_agent",
#         split,
#         tolerance=TOLERANCE,
#     )
#     samples = dataloader_to_env_samples(ds)
#     print(f"Loaded {len(samples)} {split} samples (combined dataset).")
#     for s in samples:
#         print(f"  [{s['id']}] centerline px: {int(s['centerline'].sum())}")
#     return samples


class LazyEnvDataset:
    """Thin wrapper: keeps the torch Dataset, converts to numpy on demand."""

    def __init__(self, dataset):
        self.dataset = dataset
        self._cache: Dict[int, Dict] = {}  # small LRU if needed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]
        if idx in self._cache:
            return self._cache[idx]
        s = self.dataset[idx]
        sample = {
            "id": s["id"],
            "image": s["image"].permute(1, 2, 0).numpy(),
            "centerline": s["centerline"].squeeze(0).numpy(),
            "distance_transform": s["distance_transform"].squeeze(0).numpy(),
            "fov_mask": s["fov_mask"].squeeze(0).numpy(),
        }
        if "vessel_orientation" in s:
            sample["vessel_orientation"] = s["vessel_orientation"].numpy()
        if "dt_gradient" in s:
            sample["dt_gradient"] = s["dt_gradient"].numpy()
        self._cache[idx] = sample
        return sample

    def random_sample(self, rng=None) -> Dict:
        idx = (rng or np.random).randint(len(self))
        return self[idx]


def load_samples(split: str) -> LazyEnvDataset:
    ds, _ = get_data("rl_agent", split, tolerance=TOLERANCE)
    print(f"Registered {len(ds)} {split} samples (lazy, not materialized).")
    return LazyEnvDataset(ds)


# ==========================================
# MAIN
# ==========================================


def main():
    use_lstm = CONFIG["policy"]["use_lstm"]

    wandb_mode = CONFIG.get("training", {}).get("wandb_mode", "online")
    os.environ["WANDB_MODE"] = wandb_mode
    use_wandb = CONFIG.get("training", {}).get("use_wandb", False)

    print(f"Device: {DEVICE}")
    print(f"LSTM:   {'ON chunk_len=' + str(LSTM_CHUNK_LENGTH) if use_lstm else 'OFF'}")

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
        use_wandb=use_wandb,
        lr=PPO_LR,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_eps=PPO_CLIP_EPS,
        entropy_coef=PPO_ENTROPY_COEF,
        value_coef=PPO_VALUE_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=PPO_MINI_BATCH_SIZE,
        steps_per_iter=PPO_STEPS_PER_ITER,
        num_iterations=PPO_NUM_ITERATIONS,
        eval_every=PPO_EVAL_EVERY,
        save_every=PPO_SAVE_EVERY,
        tolerance=TOLERANCE,
        lstm_chunk_length=LSTM_CHUNK_LENGTH,
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
