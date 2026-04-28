# scripts/train_seed_detector.py
"""Seed Detector Training — Step 0

Trains a UNet-based heatmap predictor to detect vessel endpoints and junctions.
Architecture: same UNet backbone as centerline_unet_baseline.py (DSConv blocks,
skip connections, ~0.5M params) with in_channels=3 for RGB input.

Its output replaces the GT-dependent _pick_frontier_seed() at inference time,
making the pipeline fully end-to-end.

Run BEFORE or independently of train_imitation.py / train_ppo.py.

All logic lives in training/seed_detector_trainer.py.
This script handles: paths, config, and data loading via the unified dataloader.
"""

import os
import sys
from typing import Dict, List

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE
from config import SEED_CONFIG as CONFIG
from config import SEED_WEIGHTS_PATH as SAVE_PATH
from config import TOLERANCE

_seed_train = CONFIG["training"]
SIGMA      = _seed_train["sigma"]
NUM_EPOCHS = _seed_train["num_epochs"]
BATCH_SIZE = _seed_train["batch_size"]
LR         = _seed_train["lr"]
from data.centerline_extraction import CenterlineExtractor
from data.dataloader import get_data
from models.seed_detector import SeedDetector
from training.seed_detector_trainer import SeedDetectorTrainer


# ==========================================
# DATA LOADING (unified dataloader)
# ==========================================


def load_samples(split: str) -> List[Dict]:
    """Load combined dataset samples for seed detector training."""
    ds, _ = get_data(
        "rl_agent",
        split,
        tolerance=TOLERANCE,
    )

    extractor = CenterlineExtractor()
    samples = []

    for i in range(len(ds)):
        s = ds[i]
        sid = s["id"]
        centerline = s["centerline"].squeeze(0).numpy()

        sample = {
            "id": sid,
            "image": s["image"].permute(1, 2, 0).numpy(),
            "centerline": centerline,
            "fov_mask": s["fov_mask"].squeeze(0).numpy(),
        }
        samples.append(sample)
        
        # We removed the individual sample prints from here 
        # to keep the log clean.

    print(f"Loaded {len(samples)} {split} samples (combined dataset).", flush=True)
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

    model = SeedDetector(CONFIG).to(DEVICE)
    trainer = SeedDetectorTrainer(
        model,
        DEVICE,
        lr=LR,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        sigma=SIGMA,
    )

    trainer.train(
        train_samples=train_samples,
        val_samples=val_samples,
        save_path=SAVE_PATH,
        config=CONFIG,
    )


if __name__ == "__main__":
    main()
