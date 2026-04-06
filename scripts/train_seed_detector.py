"""
scripts/train_seed_detector.py
==============================
Training script for the sparse keypoint (seeds) detector.
Targets: Endpoints and Junctions of the retinal vessel tree.
"""

import os
import sys
import torch
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import WEIGHTS_DIR, get_data
from models.seed_detector import SeedDetector
from training.seed_detector_trainer import SeedDetectorTrainer

# ==========================================
# CONFIGURATION
# ==========================================
SAVE_PATH = str(WEIGHTS_DIR / "seed_detector.pt")

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8       # Adjusted for 32GB RAM / GPU memory
NUM_EPOCHS = 40      # Heatmap regression takes a bit longer to converge
SIGMA = 3.0          # Radius of the Gaussian blobs in the GT heatmap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "seed_detector": {
        "base_ch": 16,
        "nms_radius": 10,
        "confidence_threshold": 0.3,
        "top_k_seeds": 400,
    }
}

# ==========================================
# DATA LOADING
# ==========================================

def dataloader_to_list(dataloader):
    """
    Convert torch dataloader batches into a list of dictionaries 
    for the SeedDetectorTrainer's internal dataset format.
    """
    samples = []
    print("Pre-processing training samples into heatmap-ready format...")
    
    for batch in dataloader:
        # Batch is typically size 1 from the unified get_data call
        # We extract the components needed for heatmap generation
        samples.append({
            "image": batch["image"].squeeze(0).permute(1, 2, 0).numpy(), # (H,W,3), preprocessed image
            "centerline": batch["centerline"].squeeze(0).squeeze(0).numpy(), # (H,W)
            "fov_mask": batch["fov_mask"].squeeze(0).squeeze(0).numpy()      # (H,W)
        })
    return samples

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # 1. Load Data via Unified Dataloader
    # Note: Using 'rl_agent' target because it provides raw image + centerline
    _, train_loader = get_data(
        target="rl_agent", 
        split="train", 
        batch_size=1, 
        num_workers=4
    )
    _, val_loader = get_data(
        target="rl_agent", 
        split="val", 
        batch_size=1, 
        num_workers=4
    )

    train_samples = dataloader_to_list(train_loader)
    val_samples = dataloader_to_list(val_loader)

    # 2. Initialize Model and Trainer
    model = SeedDetector(CONFIG).to(DEVICE)
    
    trainer = SeedDetectorTrainer(
        model=model,
        device=DEVICE,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        sigma=SIGMA
    )

    # 3. Start Training
    print("\n--- Starting Seed Detector Training ---")
    trainer.train(
        train_samples=train_samples,
        val_samples=val_samples,
        save_path=SAVE_PATH,
        config=CONFIG
    )
    print(f"\nTraining Complete. Best model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
