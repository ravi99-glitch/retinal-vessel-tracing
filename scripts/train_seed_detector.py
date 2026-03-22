# scripts/train_seed_detector.py
"""Seed Detector Training — Step 0 (optional but recommended for end-to-end inference).

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

from data.centerline_extraction import CenterlineExtractor
from data.dataloader import WEIGHTS_DIR, get_data
from models.seed_detector import SeedDetector
from training.seed_detector_trainer import SeedDetectorTrainer

# ==========================================
# CONFIG
# ==========================================
SAVE_PATH = str(WEIGHTS_DIR / "seed_detector.pt")

TOLERANCE = 2.0
SIGMA = 3.0  # Gaussian blob size around each endpoint/junction
NUM_EPOCHS = 30
BATCH_SIZE = 4  # full 565x584 images — keep small to fit VRAM
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "seed_detector": {
        "base_ch": 16,  # UNet channel width — 16 → ~0.5M params
        "nms_radius": 10,  # min distance between peaks after NMS
        "confidence_threshold": 0.3,  # min heatmap value to count as a seed
        "top_k_seeds": 50,  # max seeds returned per image
    }
}

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
        print(f"  [{s['id']}] image shape: {s['image'].shape}")
        centerline = s["centerline"].squeeze(0).numpy()

        sample = {
            "id": sid,
            "image": s["image"].permute(1, 2, 0).numpy(),  # (3,H,W) → (H,W,3)
            "centerline": centerline,
            "fov_mask": s["fov_mask"].squeeze(0).numpy(),
        }
        samples.append(sample)

        n_ep = len(extractor._find_endpoints(centerline))
        n_jn = len(extractor._find_junctions(centerline))
        print(f"  [{sid}]  endpoints={n_ep}  junctions={n_jn}  seeds={n_ep + n_jn}")

    print(f"Loaded {len(samples)} {split} samples (combined dataset).\n")
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
