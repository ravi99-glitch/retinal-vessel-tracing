"""Imitation learning (behaviour cloning)

Provides:
    augment_sample()            — 9 geometric/photometric variants per sample
    generate_expert_metadata()  — walks GT traces → lightweight metadata     [FF]
    generate_expert_sequences() — walks GT traces → episode sequences        [LSTM]
    ImitationDataset            — on-the-fly patch cropping dataset          [FF]
    ImitationSequenceDataset    — PyTorch dataset of variable-length episodes [LSTM]
    sequence_collate_fn()       — pads episodes to equal length for batching [LSTM]
    ImitationTrainer            — train loop, validation, checkpoint saving

Used by:
    scripts/train_imitation.py
"""

import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ==========================================
# ACTION CONSTANTS
# N=0 NE=1 E=2 SE=3 S=4 SW=5 W=6 NW=7  STOP=8
# ==========================================
STOP_ACTION = 8

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

# Remaps for symmetry-augmented expert traces (STOP is invariant).
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
    """Convert (dy, dx) step to discrete movement action index (0–7).

    Returns -1 for steps that don't fall on the 8-neighbour grid (e.g.
    multi-pixel jumps from skeleton kinks). Callers must skip these — note
    that 8 is the STOP action and **must not** be produced from a movement.
    """
    return DIRECTION_MAP.get((dy, dx), -1)


def generate_expert_metadata(
    sample: Dict, sample_idx: int, obs_size: int
) -> List[Dict]:
    """Walk expert traces and return lightweight metadata instead of full patches.

    This replaces generate_expert_pairs() to avoid OOM when training on ~1000 images.
    Patches are cropped on-the-fly by ImitationDataset.__getitem__().

    Args:
        sample: dict with image, distance_transform, expert_traces
        sample_idx: index of this sample in the full dataset list
        obs_size: observation patch size (e.g. 65)

    Returns:
        List of metadata dicts: {sample_idx, pos, action, prev_dir}
    """
    h, w = sample["image"].shape[:2]
    half = obs_size // 2
    steps_meta = []

    for trace in sample["expert_traces"]:
        if len(trace) < 2:
            continue
        last_valid = None  # (pos, prev_dir) for the final STOP supervision
        for i in range(len(trace) - 1):
            y, x = trace[i]
            ny, nx = trace[i + 1]

            if y < half or y >= h - half or x < half or x >= w - half:
                continue

            action = direction_to_action(int(ny) - int(y), int(nx) - int(x))
            if action < 0:
                continue

            raw_prev = (
                direction_to_action(
                    int(y) - int(trace[i - 1][0]), int(x) - int(trace[i - 1][1])
                )
                if i > 0
                else None
            )
            prev_dir = raw_prev if (raw_prev is not None and raw_prev >= 0) else None

            steps_meta.append({
                "sample_idx": sample_idx,
                "pos": (y, x),
                "action": action,
                "prev_dir": prev_dir,
            })
            last_valid = ((ny, nx), action)

        # Append a final STOP supervision at the end of each trace, so
        # imitation teaches the agent to terminate at the actual vessel end.
        if last_valid is not None:
            (sy, sx), last_action = last_valid
            if half <= sy < h - half and half <= sx < w - half:
                steps_meta.append({
                    "sample_idx": sample_idx,
                    "pos": (sy, sx),
                    "action": STOP_ACTION,
                    "prev_dir": last_action,
                })

    return steps_meta


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
        last_valid = None  # ((ny, nx), action) for STOP supervision

        for i in range(len(trace) - 1):
            y, x = trace[i]
            ny, nx = trace[i + 1]

            if y < half or y >= h - half or x < half or x >= w - half:
                continue

            action = direction_to_action(int(ny) - int(y), int(nx) - int(x))
            if action < 0:
                continue

            raw_prev = (
                direction_to_action(
                    int(y) - int(trace[i - 1][0]), int(x) - int(trace[i - 1][1])
                )
                if i > 0
                else None
            )
            prev_dir = raw_prev if (raw_prev is not None and raw_prev >= 0) else None

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
            last_valid = ((ny, nx), action)

        # Append a final STOP step at the trace endpoint.
        if last_valid is not None:
            (sy, sx), last_action = last_valid
            if half <= sy < h - half and half <= sx < w - half:
                stop_obs = obs_builder.build(
                    image=image,
                    visited_mask=visited_mask,
                    vesselness=None,
                    position=np.array([sy, sx]),
                    prev_direction=last_action,
                    distance_transform=dt,
                )
                seq_obs.append(stop_obs)
                seq_actions.append(STOP_ACTION)

        if len(seq_obs) >= 2:
            sequences.append(
                {
                    "observations": seq_obs,
                    "actions": seq_actions,
                    "length": len(seq_obs),
                }
            )

    return sequences


def generate_expert_sequence_metadata(
    sample_idx: int, sample: Dict, obs_size: int
) -> List[Dict[str, Any]]:
    """Return lightweight sequence metadata for LSTM training.

    Same trace-walking logic as generate_expert_sequences() but stores only
    positions and actions (~100 bytes per step) instead of full observation
    tensors (~170KB per step).  Observations are built on-the-fly by
    ImitationSequenceDataset.__getitem__().
    """
    traces = sample.get("expert_traces", [])
    h, w = sample["image"].shape[:2]
    half = obs_size // 2
    sequences = []

    for trace in traces:
        if len(trace) < 2:
            continue
        steps: List[Dict[str, Any]] = []
        last_valid = None
        for i in range(len(trace) - 1):
            y, x = trace[i]
            ny, nx = trace[i + 1]
            if y < half or y >= h - half or x < half or x >= w - half:
                continue
            action = direction_to_action(int(ny) - int(y), int(nx) - int(x))
            if action < 0:
                continue
            raw_prev = (
                direction_to_action(
                    int(y) - int(trace[i - 1][0]), int(x) - int(trace[i - 1][1])
                )
                if i > 0
                else None
            )
            prev_dir = raw_prev if (raw_prev is not None and raw_prev >= 0) else None
            steps.append({"pos": (y, x), "action": action, "prev_dir": prev_dir})
            last_valid = ((ny, nx), action)
        # Append STOP supervision at the endpoint of each trace.
        if last_valid is not None:
            (sy, sx), last_action = last_valid
            if half <= sy < h - half and half <= sx < w - half:
                steps.append({
                    "pos": (sy, sx),
                    "action": STOP_ACTION,
                    "prev_dir": last_action,
                })
        if len(steps) >= 2:
            sequences.append(
                {"sample_idx": sample_idx, "steps": steps, "length": len(steps)}
            )
    return sequences


# ==========================================
# OBSERVATION SOURCE PRECOMPUTATION HELPERS
# ==========================================


def _build_stacked_sources(samples, obs_builder, unet_priors=None):
    """Pre-stack static channels (DT, grads, centerline, tangent, [curv], [junc], [unet])
    via the same code path the env uses, so layout matches PPO observations."""
    from environment.observation import ObservationBuilder

    stacks: List[np.ndarray] = []
    for i, s in enumerate(samples):
        dt_grad = ObservationBuilder.compute_dt_gradient(s["distance_transform"])
        orient = ObservationBuilder.compute_vessel_orientation(s["image"])
        prior = unet_priors[i] if unet_priors is not None else None
        obs_builder.prepare_stacked_sources(
            distance_transform=s["distance_transform"],
            dt_gradient=dt_grad,
            centerline=s["centerline"],
            vessel_orientation=orient,
            unet_prior=prior,
        )
        # prepare_stacked_sources mutates obs_builder._stacked_sources;
        # grab the array now before the next iteration overwrites it.
        stacks.append(obs_builder._stacked_sources)
    return stacks


def _build_unet_priors(samples, enabled: bool):
    """Compute the centerline-UNet prior once per sample if enabled.

    Returns a list of (H, W) float32 arrays, or list of None when disabled
    or the predictor checkpoint is unavailable.
    """
    if not enabled:
        return [None] * len(samples)
    from data.dataloader import compute_unet_prior

    print(f"Computing UNet centerline prior for {len(samples)} samples...")
    out: List[Optional[np.ndarray]] = []
    for s in samples:
        out.append(compute_unet_prior(s["image"]))
    return out


def _build_vesselness_maps(samples, enabled: bool):
    """Compute Frangi vesselness once per sample if enabled.

    Skipped (returns list of None) when ``enabled`` is False so we don't
    pay the per-sample frangi cost on imitation when vesselness is off.
    """
    if not enabled:
        return [None] * len(samples)
    from skimage.filters import frangi

    print(f"Computing Frangi vesselness for {len(samples)} samples...")
    maps: List[Optional[np.ndarray]] = []
    sigmas = np.linspace(1.0, 3.0, 5)
    for s in samples:
        img = s["image"]
        gray = img[:, :, 1] if img.ndim == 3 else img
        v = frangi(gray.astype(np.float64), sigmas=sigmas, black_ridges=True)
        maps.append(v.astype(np.float32))
    return maps


# ==========================================
# DATASETS
# ==========================================


class ImitationDataset(Dataset):
    """PyTorch dataset that crops observation patches on-the-fly to save RAM.

    Instead of storing millions of pre-computed patches, stores only lightweight
    metadata and builds each patch when the DataLoader requests it.
    """

    def __init__(self, samples: List[Dict], metadata: List[Dict], config: dict):
        """
        Args:
            samples: list of full image dicts (the ~1000 images)
            metadata: list of step metadata dicts from generate_expert_metadata()
            config: full CONFIG dict for ObservationBuilder
        """
        self.samples = samples
        self.metadata = metadata

        from environment.observation import ObservationBuilder

        self.obs_builder = ObservationBuilder(config)

        # Pre-compute static channels (DT, gradients, centerline, orientation,
        # plus optional curvature/junction/unet_prior) via the same path the
        # env uses, so the channel layout stays in sync with PPO observations.
        env_cfg = config.get("environment", {})
        self.use_vesselness = env_cfg.get("use_vesselness", False)
        self.use_unet_prior = env_cfg.get("use_unet_prior", False)
        print(f"Pre-computing static observation channels for {len(samples)} samples...")
        self.unet_priors = _build_unet_priors(samples, enabled=self.use_unet_prior)
        self.stacked_sources = _build_stacked_sources(
            samples, self.obs_builder, unet_priors=self.unet_priors,
        )
        self.vesselness_maps = _build_vesselness_maps(
            samples, enabled=self.use_vesselness
        )
        print("  Done.")

        self.visited_masks = [
            np.zeros(s["image"].shape[:2], dtype=np.float32) for s in samples
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        m = self.metadata[idx]
        s = self.samples[m["sample_idx"]]
        sidx = m["sample_idx"]

        # Point obs_builder at this sample's pre-computed static channels
        self.obs_builder._stacked_sources = self.stacked_sources[sidx]

        obs = self.obs_builder.build(
            image=s["image"],
            visited_mask=self.visited_masks[sidx],
            vesselness=self.vesselness_maps[sidx] if self.use_vesselness else None,
            position=np.array(m["pos"]),
            prev_direction=m["prev_dir"],
        )

        return (
            torch.from_numpy(obs).float(),
            torch.tensor(m["action"], dtype=torch.long),
        )


class ImitationSequenceDataset(Dataset):
    """PyTorch dataset of variable-length episode sequences — LSTM mode.

    Builds observations on-the-fly from lightweight metadata to avoid
    storing ~170KB per step in RAM (which causes OOM on large datasets).

    Each item is a dict with:
        observations : torch.Tensor (T, C, H, W)
        actions      : torch.Tensor (T,)
        length       : int
    """

    def __init__(
        self,
        sequences: List[Dict[str, Any]],
        samples: List[Dict],
        config: dict,
        stacked_sources: Optional[List[np.ndarray]] = None,
        vesselness_maps: Optional[List[np.ndarray]] = None,
        unet_priors: Optional[List[Optional[np.ndarray]]] = None,
    ):
        self.sequences = sequences
        self.samples = samples

        from environment.observation import ObservationBuilder

        self.obs_builder = ObservationBuilder(config)
        env_cfg = config.get("environment", {})
        self.use_vesselness = env_cfg.get("use_vesselness", False)
        self.use_unet_prior = env_cfg.get("use_unet_prior", False)

        if unet_priors is not None:
            self.unet_priors = unet_priors
        else:
            self.unet_priors = _build_unet_priors(
                samples, enabled=self.use_unet_prior
            )

        # Re-use pre-computed stacked sources if provided, else compute them
        if stacked_sources is not None:
            self.stacked_sources = stacked_sources
        else:
            print(
                f"Pre-computing static observation channels for {len(samples)} samples (seq)..."
            )
            self.stacked_sources = _build_stacked_sources(
                samples, self.obs_builder, unet_priors=self.unet_priors,
            )
            print("  Done.")

        if vesselness_maps is not None:
            self.vesselness_maps = vesselness_maps
        else:
            self.vesselness_maps = _build_vesselness_maps(
                samples, enabled=self.use_vesselness
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        sample = self.samples[seq["sample_idx"]]
        sidx = seq["sample_idx"]
        self.obs_builder._stacked_sources = self.stacked_sources[sidx]

        h, w = sample["image"].shape[:2]
        visited_mask = np.zeros((h, w), dtype=np.float32)
        obs_list = []
        actions = []
        vmap = self.vesselness_maps[sidx] if self.use_vesselness else None

        for step in seq["steps"]:
            obs = self.obs_builder.build(
                image=sample["image"],
                visited_mask=visited_mask,
                vesselness=vmap,
                position=np.array(step["pos"]),
                prev_direction=step["prev_dir"],
            )
            obs_list.append(obs)
            actions.append(step["action"])
            visited_mask[step["pos"][0], step["pos"][1]] = 1.0

        return {
            "observations": torch.from_numpy(np.stack(obs_list)).float(),
            "actions": torch.tensor(actions, dtype=torch.long),
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
        config: dict,
        lr: float = 3e-4,
        batch_size: int = 128,
        num_epochs: int = 30,
        lstm_batch_size: int = 16,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lstm_batch_size = lstm_batch_size
        self.use_lstm = getattr(model, "use_lstm", False)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get("training", {}).get("imitation", {}).get("lr_step_size", 10),
            gamma=config.get("training", {}).get("imitation", {}).get("lr_gamma", 0.5),
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        save_path: str,
        config: dict,
        log_path: Optional[str] = None,
        train_sequences: Optional[List[Dict[str, Any]]] = None,
        val_sequences: Optional[List[Dict[str, Any]]] = None,
        samples: Optional[List[Dict]] = None,
        stacked_sources: Optional[List[np.ndarray]] = None,
        vesselness_maps: Optional[List[np.ndarray]] = None,
        unet_priors: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """Run the full training loop and save best weights.

        Args:
            train_ds:        FF training Dataset (ImitationDataset)
            val_ds:          FF validation Dataset (ImitationDataset)
            save_path:       where to save best checkpoint (.pt)
            config:          full CONFIG dict (stored in checkpoint)
            train_sequences: sequence metadata for LSTM training (required if use_lstm)
            val_sequences:   sequence metadata for LSTM validation (required if use_lstm)
            samples:         full image dicts (required if use_lstm, for on-the-fly obs)
            stacked_sources: pre-computed static channels (optional, avoids recomputation)
            vesselness_maps: pre-computed Frangi maps (optional, avoids recomputation)
        """
        _log = log_path or save_path.replace(".pt", "_log.csv")
        if self.use_lstm:
            if train_sequences is None or val_sequences is None or samples is None:
                raise ValueError(
                    "LSTM mode requires train_sequences, val_sequences, and samples. "
                    "Use generate_expert_sequence_metadata() to create sequence metadata."
                )
            self._train_lstm(
                train_sequences, val_sequences, save_path, config, _log,
                samples, stacked_sources, vesselness_maps, unet_priors,
            )
        else:
            self._train_ff(train_ds, val_ds, save_path, config, _log)

    # ------------------------------------------------------------------
    # Feedforward training (original path, unchanged)
    # ------------------------------------------------------------------

    def _train_ff(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        save_path: str,
        config: dict,
        log_path: str,
    ) -> None:
        n_workers = config.get("training", {}).get("imitation", {}).get("num_workers", 4)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
        )

        print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

        best_val_loss = float("inf")
        _csv_fields = ["epoch", "train_loss", "train_acc", "train_grad_norm",
                       "val_loss", "val_acc", "lr"]
        _csv_file = open(log_path, "w", newline="", encoding="utf-8")
        _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
        _csv_writer.writeheader()

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc, train_gn = self._run_epoch_ff(
                train_loader, train=True
            )
            val_loss, val_acc, _ = self._run_epoch_ff(val_loader, train=False)
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            print(
                f"Epoch {epoch:3d}/{self.num_epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            )
            _csv_writer.writerow({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "train_grad_norm": train_gn, "val_loss": val_loss,
                "val_acc": val_acc, "lr": current_lr,
            })
            _csv_file.flush()

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

        _csv_file.close()
        print(f"\nDone. Best val_loss={best_val_loss:.4f}  →  {save_path}")

    def _run_epoch_ff(
        self, loader: DataLoader, train: bool
    ) -> Tuple[float, float, float]:
        """Run one feedforward epoch. Returns (loss, accuracy, mean_grad_norm)."""
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        total_gn, n_updates = 0.0, 0

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
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("training", {}).get(
                            "imitation", {}).get("max_grad_norm", 1.0
                        ),
                    )
                    self.optimizer.step()
                    total_gn += grad_norm.item()
                    n_updates += 1

                total_loss += loss.item() * len(action_batch)
                correct += (logits.argmax(-1) == action_batch).sum().item()
                total += len(action_batch)

        return (
            total_loss / max(total, 1),
            correct / max(total, 1),
            total_gn / max(n_updates, 1),
        )

    # ------------------------------------------------------------------
    # LSTM sequential training
    # ------------------------------------------------------------------

    def _train_lstm(
        self,
        train_sequences: List[Dict[str, Any]],
        val_sequences: List[Dict[str, Any]],
        save_path: str,
        config: dict,
        log_path: str,
        samples: Optional[List[Dict]] = None,
        stacked_sources: Optional[List[np.ndarray]] = None,
        vesselness_maps: Optional[List[np.ndarray]] = None,
        unet_priors: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        train_loader = DataLoader(
            ImitationSequenceDataset(
                train_sequences, samples, config,
                stacked_sources=stacked_sources,
                vesselness_maps=vesselness_maps,
                unet_priors=unet_priors,
            ),
            batch_size=self.lstm_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=sequence_collate_fn,
        )
        val_loader = DataLoader(
            ImitationSequenceDataset(
                val_sequences, samples, config,
                stacked_sources=stacked_sources,
                vesselness_maps=vesselness_maps,
                unet_priors=unet_priors,
            ),
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
        _csv_fields = ["epoch", "train_loss", "train_acc", "train_grad_norm",
                       "val_loss", "val_acc", "lr"]
        _csv_file = open(log_path, "w", newline="", encoding="utf-8")
        _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
        _csv_writer.writeheader()

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc, train_gn = self._run_epoch_lstm(
                train_loader, train=True
            )
            val_loss, val_acc, _ = self._run_epoch_lstm(val_loader, train=False)
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            print(
                f"Epoch {epoch:3d}/{self.num_epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            )
            _csv_writer.writerow({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "train_grad_norm": train_gn, "val_loss": val_loss,
                "val_acc": val_acc, "lr": current_lr,
            })
            _csv_file.flush()

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

        _csv_file.close()
        print(f"\nDone. Best val_loss={best_val_loss:.4f}  →  {save_path}")

    def _run_epoch_lstm(
        self, loader: DataLoader, train: bool
    ) -> Tuple[float, float, float]:
        """Run one LSTM epoch over padded sequence batches.
        Returns (loss, accuracy, mean_grad_norm).
        """
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        total_gn, n_updates = 0.0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                obs_seq = batch["observations"].to(self.device)  # (T, B, C, H, W)
                actions = batch["actions"].to(self.device)  # (T, B)
                mask = batch["mask"].to(self.device)  # (T, B)
                dones = batch["dones"].to(self.device)  # (T, B)

                T, B = obs_seq.shape[:2]

                init_state = self.model.init_hidden(batch_size=B, device=self.device)
                logits_seq, _ = self.model.forward_sequence(obs_seq, init_state, dones)
                # logits_seq: (T, B, N_ACTIONS)

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
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("training", {}).get(
                            "imitation", {}).get("max_grad_norm", 1.0
                        ),
                    )
                    self.optimizer.step()
                    total_gn += grad_norm.item()
                    n_updates += 1

                preds = logits_seq.argmax(dim=-1)  # (T, B)
                valid_steps = mask.sum().item()
                correct += ((preds == actions) * mask).sum().item()
                total_loss += loss.item() * valid_steps
                total += valid_steps

        return (
            total_loss / max(total, 1),
            correct / max(total, 1),
            total_gn / max(n_updates, 1),
        )
