# rl_training/imitation.py
"""Imitation learning (behaviour cloning)

Provides:
    augment_sample()        — 9 geometric/photometric variants per sample
    generate_expert_pairs() — walks GT traces → (observation, action) pairs
    ImitationDataset        — PyTorch dataset wrapper
    ImitationTrainer        — train loop, validation, checkpoint saving

Used by:
    scripts/train_imitation.py  (DRIVE)
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ==========================================
# ACTION CONSTANTS
# N=0 NE=1 E=2 SE=3 S=4 SW=5 W=6 NW=7 STOP=8
# ==========================================
DIRECTION_MAP = {
    (-1, 0): 0,
    (-1, 1): 1,
    (0, 1): 2,
    (1, 1): 3,
    (1, 0): 4,
    (1, -1): 5,
    (0, -1): 6,
    (-1, -1): 7,
}

_FLIP_H_REMAP = {0: 0, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 8}
_FLIP_V_REMAP = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5, 8: 8}
_ROT90_REMAP = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 0, 7: 1, 8: 8}
_ROT180_REMAP = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3, 8: 8}
_ROT270_REMAP = {0: 6, 1: 7, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 8}


# ==========================================
# AUGMENTATION
# ==========================================


def _remap_traces(traces, transform_fn):
    return [[transform_fn(y, x) for y, x in trace] for trace in traces]


def augment_sample(sample: Dict, tolerance: float) -> List[Dict]:
    """Return augmented copies of a sample (original not included).
    5 geometric transforms + 4 brightness/contrast jitters = 9 variants.

    Args:
        sample: dict with keys image, centerline, distance_transform, fov_mask, expert_traces
        tolerance: centerline tolerance for recomputing distance transform

    Returns:
        List of augmented sample dicts in the same format as the input

    """
    from data.centerline_extraction import CenterlineExtractor

    img, cl, fov = sample["image"], sample["centerline"], sample["fov_mask"]
    traces = sample["expert_traces"]
    h, w = img.shape[:2]

    def make(new_img, new_cl, new_fov, new_traces):
        ext = CenterlineExtractor(min_branch_length=10, prune_iterations=5)
        new_dt = ext.compute_distance_transform(new_cl, tolerance=tolerance)
        return {
            "image": new_img,
            "centerline": new_cl,
            "distance_transform": new_dt,
            "fov_mask": new_fov,
            "expert_traces": new_traces,
        }

    aug = []

    # Horizontal flip
    aug.append(
        make(
            img[:, ::-1, :].copy(),
            cl[:, ::-1].copy(),
            fov[:, ::-1].copy(),
            _remap_traces(traces, lambda y, x: (y, w - 1 - x)),
        )
    )

    # Vertical flip
    aug.append(
        make(
            img[::-1, :, :].copy(),
            cl[::-1, :].copy(),
            fov[::-1, :].copy(),
            _remap_traces(traces, lambda y, x: (h - 1 - y, x)),
        )
    )

    # Rotation 90° CW
    aug.append(
        make(
            np.rot90(img, k=3).copy(),
            np.rot90(cl, k=3).copy(),
            np.rot90(fov, k=3).copy(),
            _remap_traces(traces, lambda y, x: (x, h - 1 - y)),
        )
    )

    # Rotation 180°
    aug.append(
        make(
            np.rot90(img, k=2).copy(),
            np.rot90(cl, k=2).copy(),
            np.rot90(fov, k=2).copy(),
            _remap_traces(traces, lambda y, x: (h - 1 - y, w - 1 - x)),
        )
    )

    # Rotation 270° CW
    aug.append(
        make(
            np.rot90(img, k=1).copy(),
            np.rot90(cl, k=1).copy(),
            np.rot90(fov, k=1).copy(),
            _remap_traces(traces, lambda y, x: (w - 1 - x, y)),
        )
    )

    # Brightness / contrast jitter — geometry unchanged, no dt recompute needed
    dt = sample["distance_transform"]
    for brightness, contrast in [(0.8, 1.0), (1.2, 1.0), (1.0, 0.8), (1.0, 1.2)]:
        img_jit = np.clip(img * contrast + (brightness - 1.0) * 0.5, 0.0, 1.0).astype(
            np.float32
        )
        aug.append(
            {
                "image": img_jit,
                "centerline": cl,
                "distance_transform": dt,
                "fov_mask": fov,
                "expert_traces": traces,
            }
        )

    return aug


# ==========================================
# EXPERT PAIR GENERATION
# ==========================================


def direction_to_action(dy: int, dx: int) -> int:
    """Convert (dy, dx) step to discrete action index (0–8)."""
    return DIRECTION_MAP.get((dy, dx), 8)


def generate_expert_pairs(
    sample: Dict, config: dict, obs_size: int
) -> List[Tuple[np.ndarray, int]]:
    """Walk expert traces and return (observation, action) pairs.

    Args:
        sample: dict with image, distance_transform, expert_traces
        config: full CONFIG dict (for ObservationBuilder)
        obs_size: observation patch size (e.g. 65)

    Returns:
        List of (obs_array, action_int) tuples

    """
    from rl_environment.observation import ObservationBuilder

    obs_builder = ObservationBuilder(config)
    image, dt = sample["image"], sample["distance_transform"]
    h, w = image.shape[:2]
    half = obs_size // 2
    visited_mask = np.zeros((h, w), dtype=np.float32)
    pairs = []

    for trace in sample["expert_traces"]:
        if len(trace) < 2:
            continue
        for i in range(len(trace) - 1):
            y, x = trace[i]
            ny, nx = trace[i + 1]

            if y < half or y >= h - half or x < half or x >= w - half:
                continue

            action = direction_to_action(int(ny) - int(y), int(nx) - int(x))
            if action == 8:
                continue

            prev_dir = (
                direction_to_action(
                    int(y) - int(trace[i - 1][0]), int(x) - int(trace[i - 1][1])
                )
                if i > 0
                else None
            )

            obs = obs_builder.build(
                image=image,
                visited_mask=visited_mask,
                vesselness=None,
                position=np.array([y, x]),
                prev_direction=prev_dir,
                distance_transform=dt,
            )
            pairs.append((obs, action))
            visited_mask[y, x] = 1.0

    return pairs


# ==========================================
# DATASET
# ==========================================


class ImitationDataset(Dataset):
    """PyTorch dataset of (observation, action) pairs."""

    def __init__(self, pairs: List[Tuple[np.ndarray, int]]):
        self.obs = [p[0] for p in pairs]
        self.actions = [p[1] for p in pairs]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]).float(),
            torch.tensor(self.actions[idx], dtype=torch.long),
        )


# ==========================================
# TRAINER
# ==========================================


class ImitationTrainer:
    """Behavior cloning trainer.

    Usage:
        trainer = ImitationTrainer(model, device, lr=3e-4, batch_size=128, num_epochs=30)
        trainer.train(train_pairs, val_pairs, save_path, config)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        batch_size: int = 128,
        num_epochs: int = 30,
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        train_pairs: List[Tuple],
        val_pairs: List[Tuple],
        save_path: str,
        config: dict,
    ) -> None:
        """Run the full training loop and save best weights.

        Args:
            train_pairs: list of (obs, action) for training
            val_pairs:   list of (obs, action) for validation
            save_path:   where to save best checkpoint (.pt)
            config:      full CONFIG dict (stored in checkpoint)

        """
        train_loader = DataLoader(
            ImitationDataset(train_pairs),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            ImitationDataset(val_pairs),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

        best_val_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            self.scheduler.step()

            print(
                f"Epoch {epoch:3d}/{self.num_epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "config": config,
                    },
                    save_path,
                )
                print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        print(f"\nDone. Best val_loss={best_val_loss:.4f}  →  {save_path}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        """Run one epoch. Returns (loss, accuracy)."""
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for obs_batch, action_batch in loader:
                obs_batch = obs_batch.to(self.device)
                action_batch = action_batch.to(self.device)

                logits, _, _ = self.model(obs_batch)
                loss = self.criterion(logits, action_batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(action_batch)
                correct += (logits.argmax(-1) == action_batch).sum().item()
                total += len(action_batch)

        return total_loss / total, correct / total
