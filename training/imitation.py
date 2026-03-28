"""Imitation learning (behaviour cloning)

Provides:
    augment_sample()        — 9 geometric/photometric variants per sample
    generate_expert_pairs() — walks GT traces → (observation, action) pairs  [FF]
    generate_expert_sequences() — walks GT traces → episode sequences        [LSTM]
    ImitationDataset        — PyTorch dataset wrapper (feedforward)
    ImitationSequenceDataset— PyTorch dataset of variable-length episodes    [LSTM]
    sequence_collate_fn()   — pads episodes to equal length for batching     [LSTM]
    ImitationTrainer        — train loop, validation, checkpoint saving

Used by:
    scripts/train_imitation.py  (DRIVE)
"""

from typing import Any, Dict, List, Optional, Tuple

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
# EXPERT PAIR GENERATION  (feedforward)
# ==========================================


def direction_to_action(dy: int, dx: int) -> int:
    """Convert (dy, dx) step to discrete action index (0–8)."""
    return DIRECTION_MAP.get((dy, dx), 8)


def generate_expert_pairs(
    sample: Dict, config: dict, obs_size: int
) -> List[Tuple[np.ndarray, int]]:
    """Walk expert traces and return (observation, action) pairs.

    Each step is independent — used for feedforward (non-LSTM) training.

    Args:
        sample: dict with image, distance_transform, expert_traces
        config: full CONFIG dict (for ObservationBuilder)
        obs_size: observation patch size (e.g. 65)

    Returns:
        List of (obs_array, action_int) tuples

    """
    from environment.observation import ObservationBuilder

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
# EXPERT SEQUENCE GENERATION  (LSTM)
# ==========================================


def generate_expert_sequences(
    sample: Dict, config: dict, obs_size: int
) -> List[Dict[str, Any]]:
    """Walk expert traces and return full episode sequences.

    Unlike generate_expert_pairs(), this preserves temporal ordering
    and gives each trace its own visited mask (simulating a fresh episode),
    so the LSTM can learn sequential dependencies.

    Args:
        sample: dict with image, distance_transform, expert_traces
        config: full CONFIG dict (for ObservationBuilder)
        obs_size: observation patch size (e.g. 65)

    Returns:
        List of sequence dicts, each with:
            observations : list of np.ndarray (C, H, W)
            actions      : list of int
            length       : int
    """
    from environment.observation import ObservationBuilder

    obs_builder = ObservationBuilder(config)
    image, dt = sample["image"], sample["distance_transform"]
    h, w = image.shape[:2]
    half = obs_size // 2
    sequences = []

    for trace in sample["expert_traces"]:
        if len(trace) < 2:
            continue

        # Each trace gets its own visited mask (like a fresh episode)
        visited_mask = np.zeros((h, w), dtype=np.float32)
        seq_obs: List[np.ndarray] = []
        seq_actions: List[int] = []

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
            seq_obs.append(obs)
            seq_actions.append(action)
            visited_mask[y, x] = 1.0

        if len(seq_obs) >= 2:
            sequences.append(
                {
                    "observations": seq_obs,
                    "actions": seq_actions,
                    "length": len(seq_obs),
                }
            )

    return sequences


# ==========================================
# DATASETS
# ==========================================


class ImitationDataset(Dataset):
    """PyTorch dataset of (observation, action) pairs — feedforward mode."""

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


class ImitationSequenceDataset(Dataset):
    """PyTorch dataset of variable-length episode sequences — LSTM mode.

    Each item is a dict with:
        observations : torch.Tensor (T, C, H, W)
        actions      : torch.Tensor (T,)
        length       : int
    """

    def __init__(self, sequences: List[Dict[str, Any]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        obs = np.stack(seq["observations"], axis=0)  # (T, C, H, W)
        actions = np.array(seq["actions"], dtype=np.int64)  # (T,)
        return {
            "observations": torch.from_numpy(obs).float(),
            "actions": torch.from_numpy(actions).long(),
            "length": seq["length"],
        }


def sequence_collate_fn(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collate variable-length sequences by padding to max length in batch.

    Returns:
        observations : (T_max, B, C, H, W)  — time-first for forward_sequence()
        actions      : (T_max, B)
        mask         : (T_max, B)  — 1.0 for valid steps, 0.0 for padding
        dones        : (T_max, B)  — 1.0 at last valid step of each sequence
        lengths      : list of int
    """
    lengths = [item["length"] for item in batch]
    T_max = max(lengths)
    B = len(batch)
    C, H, W = batch[0]["observations"].shape[1:]

    obs_padded = torch.zeros(T_max, B, C, H, W)
    act_padded = torch.zeros(T_max, B, dtype=torch.long)
    mask = torch.zeros(T_max, B)
    dones = torch.zeros(T_max, B)

    for b, item in enumerate(batch):
        L = item["length"]
        obs_padded[:L, b] = item["observations"]  # already (T, C, H, W)
        act_padded[:L, b] = item["actions"]
        mask[:L, b] = 1.0
        # Mark last valid step so LSTM resets hidden state for padding region
        dones[L - 1, b] = 1.0

    return {
        "observations": obs_padded,
        "actions": act_padded,
        "mask": mask,
        "dones": dones,
        "lengths": lengths,
    }


# ==========================================
# TRAINER  (supports both FF and LSTM)
# ==========================================


class ImitationTrainer:
    """Behavior cloning trainer — supports feedforward and LSTM modes.

    Feedforward (use_lstm=False):
        - Uses ImitationDataset with shuffled (obs, action) pairs
        - Standard mini-batch cross-entropy
        - train() signature: train(train_pairs, val_pairs, save_path, config)

    LSTM (use_lstm=True):
        - Uses ImitationSequenceDataset with full episode sequences
        - Batches are padded variable-length sequences
        - Masked cross-entropy loss over valid timesteps
        - forward_sequence() handles sequential LSTM context
        - train() signature: train(train_pairs, val_pairs, save_path, config,
                                   train_sequences=..., val_sequences=...)

    Usage:
        trainer = ImitationTrainer(model, device, lr=3e-4, batch_size=128, num_epochs=30)
        trainer.train(train_pairs, val_pairs, save_path, config,
                      train_sequences=train_seqs,   # only needed for LSTM
                      val_sequences=val_seqs)        # only needed for LSTM
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        batch_size: int = 128,
        num_epochs: int = 30,
        lstm_batch_size: int = 16,
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lstm_batch_size = lstm_batch_size
        self.use_lstm = getattr(model, "use_lstm", False)

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
        train_sequences: Optional[List[Dict[str, Any]]] = None,
        val_sequences: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Run the full training loop and save best weights.

        Args:
            train_pairs:     list of (obs, action) for FF training
            val_pairs:       list of (obs, action) for FF validation
            save_path:       where to save best checkpoint (.pt)
            config:          full CONFIG dict (stored in checkpoint)
            train_sequences: list of sequence dicts for LSTM training (required if use_lstm)
            val_sequences:   list of sequence dicts for LSTM validation (required if use_lstm)
        """
        if self.use_lstm:
            if train_sequences is None or val_sequences is None:
                raise ValueError(
                    "LSTM mode requires train_sequences and val_sequences. "
                    "Use generate_expert_sequences() to create them."
                )
            self._train_lstm(train_sequences, val_sequences, save_path, config)
        else:
            self._train_ff(train_pairs, val_pairs, save_path, config)

    # ------------------------------------------------------------------
    # Feedforward training (original path, unchanged)
    # ------------------------------------------------------------------

    def _train_ff(
        self,
        train_pairs: List[Tuple],
        val_pairs: List[Tuple],
        save_path: str,
        config: dict,
    ) -> None:
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
            train_loss, train_acc = self._run_epoch_ff(train_loader, train=True)
            val_loss, val_acc = self._run_epoch_ff(val_loader, train=False)
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

    def _run_epoch_ff(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        """Run one feedforward epoch. Returns (loss, accuracy)."""
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

        return total_loss / max(total, 1), correct / max(total, 1)

    # ------------------------------------------------------------------
    # LSTM sequential training
    # ------------------------------------------------------------------

    def _train_lstm(
        self,
        train_sequences: List[Dict[str, Any]],
        val_sequences: List[Dict[str, Any]],
        save_path: str,
        config: dict,
    ) -> None:
        train_loader = DataLoader(
            ImitationSequenceDataset(train_sequences),
            batch_size=self.lstm_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=sequence_collate_fn,
        )
        val_loader = DataLoader(
            ImitationSequenceDataset(val_sequences),
            batch_size=self.lstm_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=sequence_collate_fn,
        )

        train_steps = sum(s["length"] for s in train_sequences)
        val_steps = sum(s["length"] for s in val_sequences)
        print(
            f"LSTM imitation: {len(train_sequences)} train seqs ({train_steps} steps)  |  "
            f"{len(val_sequences)} val seqs ({val_steps} steps)"
        )
        print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

        best_val_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._run_epoch_lstm(train_loader, train=True)
            val_loss, val_acc = self._run_epoch_lstm(val_loader, train=False)
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

    def _run_epoch_lstm(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        """Run one LSTM epoch over padded sequence batches.

        Each batch from the loader (via sequence_collate_fn) contains:
            observations : (T_max, B, C, H, W)
            actions      : (T_max, B)
            mask         : (T_max, B)  — valid-step mask
            dones        : (T_max, B)  — end-of-sequence markers

        Uses model.forward_sequence() for sequential processing
        and masked cross-entropy for the loss.
        """
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                obs_seq = batch["observations"].to(self.device)  # (T, B, C, H, W)
                actions = batch["actions"].to(self.device)  # (T, B)
                mask = batch["mask"].to(self.device)  # (T, B)
                dones = batch["dones"].to(self.device)  # (T, B)

                T, B = obs_seq.shape[:2]

                # Fresh hidden state for each batch
                init_state = self.model.init_hidden(batch_size=B, device=self.device)

                # Sequential forward through the whole padded sequence
                logits_seq, _ = self.model.forward_sequence(obs_seq, init_state, dones)
                # logits_seq: (T, B, N_ACTIONS)

                # Masked cross-entropy
                logits_flat = logits_seq.reshape(T * B, -1)
                actions_flat = actions.reshape(T * B)
                mask_flat = mask.reshape(T * B)

                per_step_loss = nn.functional.cross_entropy(
                    logits_flat, actions_flat, reduction="none"
                )
                loss = (per_step_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                # Masked accuracy
                preds = logits_seq.argmax(dim=-1)  # (T, B)
                valid_steps = mask.sum().item()
                correct += ((preds == actions) * mask).sum().item()
                total_loss += loss.item() * valid_steps
                total += valid_steps

        return total_loss / max(total, 1), correct / max(total, 1)
