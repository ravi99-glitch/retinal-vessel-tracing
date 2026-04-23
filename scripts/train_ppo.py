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
from config import MODEL_CONFIG as CONFIG
from config import PPO_LOG_PATH as LOG_PATH
from config import PPO_WEIGHTS_PATH as SAVE_PATH
from config import TOLERANCE

_ppo = CONFIG["training"]["ppo"]
PPO_LR              = _ppo["lr"]
PPO_GAMMA           = _ppo["gamma"]
PPO_GAE_LAMBDA      = _ppo["gae_lambda"]
PPO_CLIP_EPS        = _ppo["clip_eps"]
PPO_ENTROPY_COEF    = _ppo["entropy_coef"]
PPO_VALUE_COEF      = _ppo["value_coef"]
PPO_MAX_GRAD_NORM   = _ppo["max_grad_norm"]
PPO_EPOCHS          = _ppo["epochs"]
PPO_MINI_BATCH_SIZE = _ppo["mini_batch_size"]
PPO_STEPS_PER_ITER  = _ppo["steps_per_iter"]
PPO_NUM_ITERATIONS  = _ppo["num_iterations"]
PPO_EVAL_EVERY      = _ppo["eval_every"]
PPO_SAVE_EVERY      = _ppo["save_every"]
LSTM_CHUNK_LENGTH   = _ppo["lstm_chunk_length"]
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
        if "unet_prior" in s:
            sample["unet_prior"] = s["unet_prior"].squeeze(0).numpy()
        if "vessel_mask" in s:
            sample["vessel_mask"] = s["vessel_mask"].squeeze(0).numpy()
        self._cache[idx] = sample
        return sample

    def random_sample(self, rng=None) -> Dict:
        idx = (rng or np.random).randint(len(self))
        return self[idx]


def load_samples(split: str) -> LazyEnvDataset:
    use_unet_prior = CONFIG.get("environment", {}).get("use_unet_prior", False)
    ds, _ = get_data(
        "rl_agent", split, tolerance=TOLERANCE, use_unet_prior=use_unet_prior,
    )
    print(f"Registered {len(ds)} {split} samples (lazy, not materialized).")
    return LazyEnvDataset(ds)


# ==========================================
# MAIN
# ==========================================


def main():
    use_lstm = CONFIG["policy"]["use_lstm"]

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
