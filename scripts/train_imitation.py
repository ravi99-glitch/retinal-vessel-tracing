# scripts/train_imitation.py
"""
Imitation Learning — Step 1 of 2 before PPO.
Run this BEFORE train_ppo.py.

All logic lives in rl_training/imitation.py.
This script handles: paths, config, data loading via the unified dataloader,
and wiring.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rl_models.policy_network import ActorCriticNetwork
from data.centerline_extraction import CenterlineExtractor
from data.dataloader import load_dataset
from data.dataset_paths import get_root, WEIGHTS_DIR
from rl_training.imitation import (
    ImitationTrainer,
    augment_sample,
    generate_expert_pairs,
)

# ==========================================
# CONFIG
# ==========================================
SAVE_PATH   = str(WEIGHTS_DIR / "imitation_policy.pt")

LEARNING_RATE = 3e-4
BATCH_SIZE    = 128
NUM_EPOCHS    = 30
TOLERANCE     = 2.0
OBS_SIZE      = 65
USE_AUGMENT   = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'policy': {
        'hidden_dim':   128,
        'lstm_hidden':  128,
        'use_lstm':     False,
        'dropout':      0.0,
        'encoder_type': 'cnn',
    },
    'environment': {
        'observation_size': OBS_SIZE,
        'tolerance':        TOLERANCE,
        'use_vesselness':   False,
    },
    'training': {'ppo': {'gamma': 0.99}},
}


# ==========================================
# DATA LOADING (unified dataloader)
# ==========================================

def load_training_samples():
    """Load DRIVE training samples via the unified dataloader
    and generate expert traces for imitation learning."""
    drive_root = get_root("DRIVE")
    ds, _ = load_dataset(
        str(drive_root), "DRIVE",
        target="rl_agent",
        split="train",
        tolerance=TOLERANCE,
    )

    extractor = CenterlineExtractor(min_branch_length=10, prune_iterations=5)
    samples = []

    for i in range(len(ds)):
        s = ds[i]
        sid = s['id']
        centerline = s['centerline'].squeeze(0).numpy()
        expert_traces = extractor.generate_expert_traces(centerline)

        sample = {
            'image':          s['image'].permute(1, 2, 0).numpy(),        # (3,H,W) → (H,W,3)
            'centerline':     centerline,                                 # (H,W)
            'distance_transform': s['distance_transform'].squeeze(0).numpy(), # (H,W)
            'fov_mask':       s['fov_mask'].squeeze(0).numpy(),           # (H,W)
            'expert_traces':  expert_traces,
        }
        samples.append(sample)

        print(f"  [{sid}] centerline px: {int(centerline.sum())}  "
              f"traces: {len(expert_traces)}")

    print(f"Loaded {len(samples)} training samples from {drive_root}.\n")
    return samples


# ==========================================
# MAIN
# ==========================================

def main():
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("\nLoading DRIVE training samples...")
    all_pairs = []

    for sample in load_training_samples():
        pairs = generate_expert_pairs(sample, CONFIG, OBS_SIZE)
        all_pairs.extend(pairs)
        print(f"  -> {len(pairs)} pairs")

        if USE_AUGMENT:
            for aug in augment_sample(sample, TOLERANCE):
                all_pairs.extend(generate_expert_pairs(aug, CONFIG, OBS_SIZE))

    print(f"\nTotal (obs, action) pairs: {len(all_pairs)}")
    if not all_pairs:
        print("ERROR: No pairs generated. Check DRIVE paths.")
        return

    split       = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[:split]
    val_pairs   = all_pairs[split:]

    model   = ActorCriticNetwork(CONFIG).to(DEVICE)
    trainer = ImitationTrainer(
        model, DEVICE,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
    )
    trainer.train(train_pairs, val_pairs, SAVE_PATH, CONFIG)


if __name__ == "__main__":
    main()
