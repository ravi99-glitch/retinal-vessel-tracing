# scripts/train_imitation_resnet.py
"""Imitation Learning — Step 1 of 2 before PPO.
Run this BEFORE train_ppo.py.
"""

import os
import sys
import psutil
import torch
import csv
import matplotlib
matplotlib.use("Agg")  # Use headless backend for SLURM
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader  # <-- ADDED THIS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.centerline_extraction import CenterlineExtractor
from data.dataloader import WEIGHTS_DIR, get_data
from models.policy_network import ActorCriticNetwork
from training.imitation import (ImitationTrainer, augment_sample,
                                generate_expert_metadata, ImitationDataset)

# ==========================================
# CONFIG
# ==========================================
SAVE_PATH = str(WEIGHTS_DIR / "imitation_policy_resnet.pt")
PLOT_PATH = str(WEIGHTS_DIR / "imitation_learning_curve.png")
LOG_PATH = str(WEIGHTS_DIR / "imitation_log.csv")

LEARNING_RATE = 3e-4
BATCH_SIZE = 128
NUM_EPOCHS = 30
TOLERANCE = 2.0
OBS_SIZE = 65
USE_AUGMENT = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "policy": {
        "hidden_dim": 128,
        "dropout": 0.0,
        "encoder_type": "resnet",
    },
    "environment": {
        "observation_size": OBS_SIZE,
        "tolerance": TOLERANCE,
        "max_steps_per_episode": 2000,
        "max_off_track_streak": 8,
        "step_size": 1,
    },
    "reward": {
        "alpha_near": 0.5,           
        "beta_coverage": 1.0,
        "gamma_off": -1.0,           
        "lambda_revisit": -5.0,      
        "step_cost": -0.01,
        "direction_bonus": 0.05,
        "terminal_f1_weight": 2.5,
        "terminal_cldice_weight": 5.0,
        "smoothness_penalty": -0.05,
        "use_potential_shaping": False,
    },
    "training": {"ppo": {"gamma": 0.99}},
}

# ==========================================
# DATA LOADING (unified dataloader)
# ==========================================
process = psutil.Process()

def load_training_samples():
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
            "image": s["image"].permute(1, 2, 0).numpy(),
            "centerline": centerline,
            "distance_transform": s["distance_transform"].squeeze(0).numpy(),
            "fov_mask": s["fov_mask"].squeeze(0).numpy(),
            "expert_traces": expert_traces,
            "vessel_mask": s["vessel_mask"].squeeze(0).numpy(),
            "vessel_orientation": s["vessel_orientation"].numpy(),
            "dt_gradient": s["dt_gradient"].numpy(),
        }
        samples.append(sample)

        if i % 50 == 0:
            print(f"  [{sid}] Memory: {process.memory_info().rss / 1e9:.1f} GB")

    print(f"Loaded {len(samples)} training samples (combined dataset).\n")
    return samples


# ==========================================
# VISUALIZATION
# ==========================================
def plot_imitation_curve(log_file: str, save_file: str):
    """Parses the Imitation log CSV and generates a learning curve graph."""
    print(f"\nGenerating training curve graph from {log_file}...")
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    epochs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []

    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            train_accs.append(float(row["train_acc"]))
            val_accs.append(float(row["val_acc"]))

    if not epochs:
        print("No data found in logs to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Loss
    ax1.set_xlabel("Epochs", fontweight='bold')
    ax1.set_ylabel("Cross Entropy Loss", fontweight='bold')
    ax1.plot(epochs, train_losses, color="tab:blue", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, color="tab:red", label="Validation Loss", linewidth=2, linestyle='--')
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    ax1.set_title("Imitation Loss Curve")

    # Plot Accuracy
    ax2.set_xlabel("Epochs", fontweight='bold')
    ax2.set_ylabel("Accuracy", fontweight='bold')
    ax2.plot(epochs, train_accs, color="tab:green", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, val_accs, color="tab:orange", label="Validation Accuracy", linewidth=2, linestyle='--')
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    ax2.set_title("Imitation Accuracy Curve")

    fig.suptitle("Imitation Agent Training Progress", fontsize=16, fontweight='bold')
    fig.tight_layout()
    
    plt.savefig(save_file, dpi=150)
    print(f"Graph successfully saved to: {save_file}")


# ==========================================
# MAIN
# ==========================================
def main():
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    samples = load_training_samples()
    all_metadata = []

    print("\nGenerating Step Metadata (RAM Efficient)...")
    for i, sample in enumerate(samples):
        meta = generate_expert_metadata(sample, i, OBS_SIZE)
        all_metadata.extend(meta)
        if i % 50 == 0:
            print(f"  Processed {i}/{len(samples)} images...")

    print(f"\nTotal Expert Steps: {len(all_metadata)}")
    
    import random
    random.shuffle(all_metadata)
    
    split = int(len(all_metadata) * 0.9)
    train_meta = all_metadata[:split]
    val_meta = all_metadata[split:]

    train_ds = ImitationDataset(samples, train_meta, CONFIG)
    val_ds = ImitationDataset(samples, val_meta, CONFIG)

    # <-- ADDED DATALOADERS HERE -->
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ActorCriticNetwork(CONFIG).to(DEVICE)
    trainer = ImitationTrainer(
        model,
        DEVICE,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
    )
    
    # Initialize CSV Log
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

    best_val_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        # <-- PASSED DATALOADERS INSTEAD OF DATASETS -->
        train_loss, train_acc = trainer._run_epoch(train_loader, train=True)
        val_loss, val_acc = trainer._run_epoch(val_loader, train=False)
        
        # Log to CSV
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": trainer.model.state_dict(),
                    "val_loss": val_loss,
                    "config": CONFIG,
                },
                SAVE_PATH,
            )
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    print(f"\nDone. Best val_loss={best_val_loss:.4f}  →  {SAVE_PATH}")
    
    # --- Generate the visualization after training finishes ---
    plot_imitation_curve(LOG_PATH, PLOT_PATH)


if __name__ == "__main__":
    main()
