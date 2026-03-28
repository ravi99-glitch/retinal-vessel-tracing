"""Imitation Learning — Step 1 of 2 before PPO.
Run this BEFORE train_ppo.py.

All logic lives in training/imitation.py.
This script handles: paths, config, data loading via the unified dataloader,
and wiring.

Supports both feedforward and LSTM modes via CONFIG["policy"]["use_lstm"].
"""

import os
import sys

import numpy as np
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import DEVICE
from config import IMITATION_BATCH_SIZE as BATCH_SIZE
from config import IMITATION_LR as LEARNING_RATE
from config import IMITATION_LSTM_BATCH_SIZE as LSTM_BATCH_SIZE
from config import IMITATION_NUM_EPOCHS as NUM_EPOCHS
from config import IMITATION_USE_AUGMENT as USE_AUGMENT
from config import IMITATION_WEIGHTS_PATH as SAVE_PATH
from config import MODEL_CONFIG as CONFIG
from config import OBS_SIZE, TOLERANCE
from data.centerline_extraction import CenterlineExtractor
from data.dataloader import get_data
# from data.dataloader import WEIGHTS_DIR, get_data
from models.policy_network import ActorCriticNetwork
from training.imitation import (ImitationTrainer, augment_sample,
                                generate_expert_pairs,
                                generate_expert_sequences)

USE_LSTM = CONFIG["policy"]["use_lstm"]

# ==========================================
# CONFIG
# ==========================================
# SAVE_PATH = str(WEIGHTS_DIR / "imitation_policy.pt")

# LEARNING_RATE = 3e-4
# BATCH_SIZE = 128
# LSTM_BATCH_SIZE = 16
# NUM_EPOCHS = 30
# TOLERANCE = 2.0
# OBS_SIZE = 65
# USE_AUGMENT = False

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONFIG = {
#     "policy": {
#         "hidden_dim": 128,
#         "lstm_hidden": 128,
#         "use_lstm": True,
#         "dropout": 0.0,
#         "encoder_type": "cnn",
#     },
#     "environment": {
#         "observation_size": OBS_SIZE,
#         "tolerance": TOLERANCE,
#         "use_vesselness": False,
#     },
#     "training": {"ppo": {"gamma": 0.99}},
# }

# USE_LSTM = CONFIG["policy"]["use_lstm"]


# ==========================================
# DATA LOADING (unified dataloader)
# ==========================================
process = psutil.Process()


def load_training_samples():
    """Load combined dataset training samples via the unified dataloader
    and generate expert traces for imitation learning.
    """
    ds, _ = get_data(
        "rl_agent",
        "train",
        tolerance=TOLERANCE,
    )

    extractor = CenterlineExtractor(min_branch_length=10, prune_iterations=5)
    samples = []

    for i in range(len(ds)):
        s = ds[i]
        sid = s["id"]
        centerline = s["centerline"].squeeze(0).numpy()
        expert_traces = extractor.generate_expert_traces(centerline)

        sample = {
            "image": s["image"].permute(1, 2, 0).numpy(),  # (3,H,W) → (H,W,3)
            "centerline": centerline,  # (H,W)
            "distance_transform": s["distance_transform"].squeeze(0).numpy(),  # (H,W)
            "fov_mask": s["fov_mask"].squeeze(0).numpy(),  # (H,W)
            "expert_traces": expert_traces,
        }
        samples.append(sample)

        print(f"  [{sid}] Memory: {process.memory_info().rss / 1e9:.1f} GB")

    print(f"Loaded {len(samples)} training samples (combined dataset).\n")
    return samples


# ==========================================
# MAIN
# ==========================================


def main():
    print(f"Device: {DEVICE}")
    print(f"LSTM:   {'ON' if USE_LSTM else 'OFF'}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("\nLoading combined training samples...")
    all_samples = load_training_samples()

    # ------------------------------------------------------------------
    # Generate training data: pairs (FF) and/or sequences (LSTM)
    # ------------------------------------------------------------------
    all_pairs = []
    all_sequences = []

    for sample in all_samples:
        # Always generate pairs (used for FF, also useful for statistics)
        pairs = generate_expert_pairs(sample, CONFIG, OBS_SIZE)
        all_pairs.extend(pairs)

        if USE_LSTM:
            seqs = generate_expert_sequences(sample, CONFIG, OBS_SIZE)
            all_sequences.extend(seqs)

        n_info = f"{len(pairs)} pairs"
        if USE_LSTM:
            # n_seqs = len(generate_expert_sequences(sample, CONFIG, OBS_SIZE))
            # Use already-computed seqs count
            n_info += f", {len(seqs)} sequences"
        print(f"  -> {n_info}")

        if USE_AUGMENT:
            for aug in augment_sample(sample, TOLERANCE):
                aug_pairs = generate_expert_pairs(aug, CONFIG, OBS_SIZE)
                all_pairs.extend(aug_pairs)
                if USE_LSTM:
                    aug_seqs = generate_expert_sequences(aug, CONFIG, OBS_SIZE)
                    all_sequences.extend(aug_seqs)

    # ------------------------------------------------------------------
    # Train/val split
    # ------------------------------------------------------------------
    if USE_LSTM:
        print(
            f"\nTotal sequences: {len(all_sequences)}  "
            f"(avg length {np.mean([s['length'] for s in all_sequences]):.1f})"
        )

        # Shuffle and split sequences
        indices = np.random.permutation(len(all_sequences))
        split = int(len(all_sequences) * 0.9)
        train_sequences = [all_sequences[i] for i in indices[:split]]
        val_sequences = [all_sequences[i] for i in indices[split:]]

        # Also split pairs for logging / potential hybrid use
        split_p = int(len(all_pairs) * 0.9)
        train_pairs = all_pairs[:split_p]
        val_pairs = all_pairs[split_p:]

        print(f"Train: {len(train_sequences)} seqs  |  Val: {len(val_sequences)} seqs")
    else:
        print(f"\nTotal (obs, action) pairs: {len(all_pairs)}")
        if not all_pairs:
            print("ERROR: No pairs generated. Check data paths.")
            return

        split = int(len(all_pairs) * 0.9)
        train_pairs = all_pairs[:split]
        val_pairs = all_pairs[split:]
        train_sequences = None
        val_sequences = None

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model = ActorCriticNetwork(CONFIG).to(DEVICE)
    trainer = ImitationTrainer(
        model,
        DEVICE,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lstm_batch_size=LSTM_BATCH_SIZE,
    )
    trainer.train(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        save_path=SAVE_PATH,
        config=CONFIG,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
    )


if __name__ == "__main__":
    main()
