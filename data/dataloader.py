"""Unified dataloader for retinal vessel segmentation.

Datasets
--------
Training / validation (combined, balanced):
    FIVES, STARE, CHASE_DB1, HRF, LES_AV

External test (used in full, no split):
    DRIVE, DR_HAGIS

Supported targets
-----------------
unet           – (1,H,W) CLAHE-preprocessed grayscale + skeleton GT
frangi         – (H,W,3) raw RGB uint8 + binary annotations (numpy)
greedy_tracer  – (H,W,3) raw RGB uint8 + binary annotations (numpy)
rl_agent       – (3,H,W) float32 RGB + centerline, distance transform, …

Usage
-----
    from data.dataloader import get_data, get_test_data

    # Training & validation (balanced across 5 datasets)
    train_ds, train_loader = get_data("unet", "train", batch_size=4)
    val_ds,   val_loader   = get_data("unet", "val",   batch_size=1)

    # External test sets (entire dataset, no split)
    test_ds, test_loader = get_test_data("AV_WIDE",  "unet", batch_size=1)
    test_ds, test_loader = get_test_data("DR_HAGIS", "unet", batch_size=1)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import (ConcatDataset, DataLoader, Dataset,
                              WeightedRandomSampler)

from .centerline_extraction import CenterlineExtractor
from .fundus_preprocessor import FundusPreprocessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset root resolution
# ---------------------------------------------------------------------------
_DATA_BASE = Path(
    "/cfs/earth/scratch/icls/shared/icls-retinal-vessel-tracing/retinal-vessel-tracing/data"
)

_PROJECT_ROOT = _DATA_BASE.parent
WEIGHTS_DIR = _PROJECT_ROOT / "weights"
OUTPUT_DIR = _PROJECT_ROOT / "results"


def get_root(dataset_name: str) -> Path:
    """Return the root directory for a dataset."""
    canon = dataset_name.upper()
    if canon not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Known: {sorted(set(DATASET_REGISTRY))}"
        )

    env_key = f"RETINAL_DATA_{canon}"
    if env_key in os.environ:
        root = Path(os.environ[env_key])
    else:
        root = _DATA_BASE / canon

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root for {canon} does not exist: {root}\n")
    return root


# ---------------------------------------------------------------------------
# Dataset layout descriptors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    """File-system layout for one retinal fundus dataset."""

    image_dir: str
    vessel_dir: str
    image_glob: str
    vessel_suffix: str
    mask_dir: Optional[str] = None
    mask_suffix: Optional[str] = None
    stem_rule: Optional[str] = None
    train_subdir: Optional[str] = None

    def vessel_filename(self, image_stem: str) -> str:
        stem = self._transform_stem(image_stem)
        return f"{stem}{self.vessel_suffix}"

    def mask_filename(self, image_stem: str) -> str:
        if self.mask_suffix is None:
            raise ValueError("No mask_suffix configured.")
        return f"{image_stem}{self.mask_suffix}"

    def _transform_stem(self, stem: str) -> str:
        if self.stem_rule == "drive":
            return stem.replace("_training", "_manual1").replace("_test", "_manual1")
        return stem


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "DRIVE": DatasetConfig(
        image_dir="images",
        vessel_dir="1st_manual",
        image_glob="*.tif",
        vessel_suffix=".gif",
        mask_dir="mask",
        mask_suffix="_mask.gif",
        stem_rule="drive",
        train_subdir="training",
    ),
    "STARE": DatasetConfig(
        image_dir=".",
        vessel_dir=".",
        image_glob="*.ppm",
        vessel_suffix=".vk.ppm",
    ),
    "CHASEDB1": DatasetConfig(
        image_dir=".",
        vessel_dir=".",
        image_glob="*.jpg",
        vessel_suffix="_1stHO.png",
    ),
    "HRF": DatasetConfig(
        image_dir="images",
        vessel_dir="manual1",
        image_glob="*.[jJ][pP][gG]",
        vessel_suffix=".tif",
        mask_dir="mask",
        mask_suffix="_mask.tif",
    ),
    "LES-AV": DatasetConfig(
        image_dir="images",
        vessel_dir="vessel-segmentations",
        image_glob="*.png",
        vessel_suffix=".png",
        mask_dir="masks",
        mask_suffix="_mask.gif",
    ),
    "DRHAGIS": DatasetConfig(
        image_dir="Fundus_Images",
        vessel_dir="Manual_Segmentations",
        image_glob="*.jpg",
        vessel_suffix="_manual_orig.png",
        mask_dir="Mask_images",
        mask_suffix="_mask_orig.png",
    ),
    "FIVES": DatasetConfig(
        image_dir="Original",
        vessel_dir="Ground truth",
        image_glob="*.png",
        vessel_suffix=".png",
        train_subdir="train",
    ),
}


TRAIN_DATASETS = ("FIVES", "STARE", "CHASEDB1", "HRF", "LES-AV")
TEST_DATASETS = ("DRIVE", "DRHAGIS")
VALID_TARGETS = ("unet", "frangi", "greedy_tracer", "rl_agent")


# ---------------------------------------------------------------------------
# Collate for numpy-dict targets (frangi / greedy_tracer)
# ---------------------------------------------------------------------------
def _list_collate(batch: list) -> list:
    return batch


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------
class RetinalFundusDataset(Dataset):
    """PyTorch Dataset for a single retinal fundus dataset.

    Parameters
    ----------
    root_dir      : top-level dataset directory (e.g. "data/DRIVE")
    dataset_name  : key in DATASET_REGISTRY
    target        : output format ("unet", "frangi", "greedy_tracer", "rl_agent")
    split         : "train", "val", or None (= return all samples)
    train_frac    : fraction of samples used for training (rest → val)
    resize        : (H, W) to resize all images/masks, or None
    tolerance     : distance-transform radius for rl_agent target
    cache_centerlines : persist skeletons to disk
    transform     : optional albumentations pipeline (unet target only)
    preprocessor  : shared FundusPreprocessor instance
    centerline_extractor : shared CenterlineExtractor instance

    """

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        target: str = "rl_agent",
        split: Optional[str] = None,
        train_frac: float = 0.8,
        resize: Optional[Tuple[int, int]] = None,
        tolerance: float = 2.0,
        cache_centerlines: bool = True,
        transform=None,
        fundus_preprocessor: Optional[FundusPreprocessor] = None,
        centerline_extractor: Optional[CenterlineExtractor] = None,
    ):
        if target not in VALID_TARGETS:
            raise ValueError(f"target must be one of {VALID_TARGETS}, got '{target}'")

        canon = dataset_name.upper()
        if canon not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Known: {sorted(set(DATASET_REGISTRY))}"
            )

        self.dataset_name = canon
        self.cfg = DATASET_REGISTRY[canon]
        self.target = target
        self.resize = resize
        self.tolerance = tolerance
        self.transform = transform
        self.fundus_preprocessor = fundus_preprocessor or FundusPreprocessor()
        self.cl_extractor = centerline_extractor or CenterlineExtractor()

        # Resolve root directory
        self.root = self._resolve_root(Path(root_dir))

        # Discover and split samples
        self.samples = self._discover_samples()
        if split is not None:
            self.samples = self._apply_split(self.samples, split, train_frac)

        if not self.samples:
            raise FileNotFoundError(
                f"No samples found for {canon} in {self.root}. "
                f"Expected images in '{self.cfg.image_dir}/' matching "
                f"'{self.cfg.image_glob}' with vessels in '{self.cfg.vessel_dir}/'."
            )

        # Centerline cache
        self._cl_mem: Dict[str, np.ndarray] = {}
        self._cache_dir: Optional[Path] = None
        if cache_centerlines and self.resize is None:
            self._cache_dir = self.root / "centerlines_cache"
            self._cache_dir.mkdir(exist_ok=True)

        logger.info(
            "%s  %d samples  target=%s  split=%s",
            canon,
            len(self.samples),
            target,
            split,
        )

    # -- Root resolution ---------------------------------------------------
    def _resolve_root(self, root_dir: Path) -> Path:
        """Find the directory that actually contains images."""
        if self.cfg.train_subdir is not None:
            subdir = root_dir / self.cfg.train_subdir
            if subdir.is_dir():
                return subdir
            if root_dir.name == self.cfg.train_subdir:
                return root_dir  # user passed the subdir directly
        return root_dir

    # -- Sample discovery --------------------------------------------------
    def _discover_samples(self) -> List[Dict[str, Any]]:
        image_dir = self.root / self.cfg.image_dir
        vessel_dir = self.root / self.cfg.vessel_dir
        mask_dir = (self.root / self.cfg.mask_dir) if self.cfg.mask_dir else None

        samples: List[Dict[str, Any]] = []
        for img_path in sorted(image_dir.glob(self.cfg.image_glob)):
            vessel_path = vessel_dir / self.cfg.vessel_filename(img_path.stem)
            if not vessel_path.exists():
                continue

            entry: Dict[str, Any] = {
                "id": img_path.stem,
                "image": img_path,
                "vessel": vessel_path,
            }
            if mask_dir is not None and self.cfg.mask_suffix is not None:
                mask_path = mask_dir / self.cfg.mask_filename(img_path.stem)
                if mask_path.exists():
                    entry["mask"] = mask_path

            samples.append(entry)
        return samples

    # -- Train/val split ---------------------------------------------------
    @staticmethod
    def _apply_split(samples: List[Dict], split: str, train_frac: float) -> List[Dict]:
        """Deterministic train/val split (sorted filenames → reproducible)."""
        n = len(samples)
        t = max(1, min(int(train_frac * n), n - 1))
        if split == "train":
            return samples[:t]
        elif split == "val":
            return samples[t:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    # -- I/O ---------------------------------------------------------------
    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))

    @staticmethod
    def _load_gray(path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert("L"))

    def _load_vessel(self, path: Path) -> np.ndarray:
        return (self._load_gray(path) > 127).astype(np.float32)

    def _load_fov(self, path: Path) -> np.ndarray:
        return (self._load_gray(path) > 127).astype(np.uint8) * 255

    # -- Centerline cache --------------------------------------------------
    def _get_centerline(self, sid: str, vessel: np.ndarray) -> np.ndarray:
        if sid in self._cl_mem:
            return self._cl_mem[sid]
        if self._cache_dir is not None:
            cache_file = self._cache_dir / f"{sid}_cl.npy"
            if cache_file.exists():
                cl = np.load(cache_file)
                self._cl_mem[sid] = cl
                return cl
        cl = self.cl_extractor.extract_centerline(vessel)
        self._cl_mem[sid] = cl
        if self._cache_dir is not None:
            np.save(self._cache_dir / f"{sid}_cl.npy", cl)
        return cl

    # -- FOV mask ----------------------------------------------------------

    def _get_fov(self, sample: Dict, rgb: np.ndarray) -> np.ndarray:
        if "mask" in sample:
            return self._load_fov(sample["mask"])
        green = self.fundus_preprocessor.extract_green_channel(rgb)
        if green.dtype != np.uint8:
            green = np.clip(green * 255, 0, 255).astype(np.uint8)
        return self.fundus_preprocessor.create_fov_mask(green)

    # -- Resize ------------------------------------------------------------
    def _maybe_resize(self, *arrays: np.ndarray, interp: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
        if self.resize is None:
            return arrays
        target_h, target_w = self.resize
        src_h, src_w = arrays[0].shape[:2]

        # Scale to fit within target while preserving aspect ratio
        scale = min(target_h / src_h, target_w / src_w)
        new_h, new_w = int(src_h * scale), int(src_w * scale)

        # Padding offsets (center the image)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2

        out = []
        for i, arr in enumerate(arrays):
            if interp is not None and i < len(interp):
                flag = interp[i]
            elif arr.ndim == 2:
                flag = cv2.INTER_NEAREST
            else:
                flag = cv2.INTER_LINEAR

            resized = cv2.resize(arr, (new_w, new_h), interpolation=flag)

            # Create padded output (black/zero padding)
            if arr.ndim == 3:
                padded = np.zeros((target_h, target_w, arr.shape[2]), dtype=arr.dtype)
            else:
                padded = np.zeros((target_h, target_w), dtype=arr.dtype)

            padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
            out.append(padded)

        return tuple(out)
    
    # def _maybe_resize(
    #     self, *arrays: np.ndarray, interp: Optional[List[int]] = None
    # ) -> Tuple[np.ndarray, ...]:
    #     if self.resize is None:
    #         return arrays
    #     h, w = self.resize
    #     out = []
    #     for i, arr in enumerate(arrays):
    #         if interp is not None and i < len(interp):
    #             flag = interp[i]
    #         elif arr.ndim == 2:
    #             flag = cv2.INTER_NEAREST
    #         else:
    #             flag = cv2.INTER_LINEAR
    #         out.append(cv2.resize(arr, (w, h), interpolation=flag))
    #     return tuple(out)

    # -- __len__ / __getitem__ ---------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sid = sample["id"]

        rgb = self._load_rgb(sample["image"])
        vessel = self._load_vessel(sample["vessel"])
        fov = self._get_fov(sample, rgb)

        if self.resize is not None:
            rgb, vessel, fov = self._maybe_resize(
                rgb,
                vessel,
                fov,
                interp=[cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_NEAREST],
            )
            vessel = (vessel > 0.5).astype(np.float32)

        return getattr(self, f"_fmt_{self.target}")(sid, rgb, vessel, fov)

    # -- Target formatters -------------------------------------------------
    def _fmt_unet(
        self, sid: str, rgb: np.ndarray, vessel: np.ndarray, fov: np.ndarray
    ) -> Dict[str, Any]:
        ext_mask = fov if fov.max() > 0 else None
        preprocessed = self.fundus_preprocessor.preprocess(rgb, external_mask=ext_mask)
        cl = self._get_centerline(sid, vessel)
        fov_f = (fov > 0).astype(np.float32)

        if self.transform is not None:
            img_u8 = np.clip(preprocessed * 255, 0, 255).astype(np.uint8)
            assert (
                img_u8.shape == cl.shape == fov_f.shape == vessel.shape
            ), f"Shape mismatch: img={img_u8.shape} cl={cl.shape} fov={fov_f.shape} vessel={vessel.shape}"
            aug = self.transform(image=img_u8, mask=cl, fov=fov_f, thick_gt=vessel)
            preprocessed = aug["image"].astype(np.float32) / 255.0
            cl = aug["mask"]
            fov_f = aug["fov"]
            vessel = aug["thick_gt"]

        return {
            "id": sid,
            "image": torch.from_numpy(preprocessed).unsqueeze(0).float(),
            "centerline": torch.from_numpy(cl).unsqueeze(0).float(),
            "vessel_mask": torch.from_numpy(vessel).unsqueeze(0).float(),
            "fov_mask": torch.from_numpy(fov_f).unsqueeze(0).float(),
        }

    def _fmt_frangi(self, sid, rgb, vessel, fov):
        ext_mask = fov if fov.max() > 0 else None
        preprocessed = self.fundus_preprocessor.preprocess(rgb, external_mask=ext_mask)
        cl = self._get_centerline(sid, vessel)
        return {
            "id": sid,
            "image": rgb,                                      
            "preprocessed": preprocessed,                     
            "vessel_mask": (vessel * 255).astype(np.uint8),
            "centerline": (cl * 255).astype(np.uint8),
            "fov_mask": fov,
        }

    def _fmt_greedy_tracer(
        self, sid: str, rgb: np.ndarray, vessel: np.ndarray, fov: np.ndarray
    ) -> Dict[str, Any]:
        return self._fmt_frangi(sid, rgb, vessel, fov)

    def _fmt_rl_agent(
        self, sid: str, rgb: np.ndarray, vessel: np.ndarray, fov: np.ndarray
    ) -> Dict[str, Any]:
        img_f = rgb.astype(np.float32) / 255.0
        img_orig = img_f.copy()

        ext_mask = fov if fov.max() > 0 else None
        enhanced_green = self.fundus_preprocessor.preprocess(rgb, external_mask=ext_mask)
        img_f[:, :, 1] = enhanced_green

        cl = self._get_centerline(sid, vessel)
        dt = self.cl_extractor.compute_distance_transform(cl, self.tolerance)
        fov_f = (fov > 0).astype(np.float32)

        return {
            "id": sid,
            "image": torch.from_numpy(img_f).permute(2, 0, 1).float(),
            "image_orig": torch.from_numpy(img_orig).permute(2, 0, 1).float(),
            "vessel_mask": torch.from_numpy(vessel).unsqueeze(0).float(),
            "centerline": torch.from_numpy(cl).unsqueeze(0).float(),
            "fov_mask": torch.from_numpy(fov_f).unsqueeze(0).float(),
            "distance_transform": torch.from_numpy(dt).unsqueeze(0).float(),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_data(
    target: str = "rl_agent",
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 0,
    resize: Tuple[int, int] = (512, 512),
    train_frac: float = 0.8,
    balance: bool = True,
    **kwargs,
) -> Tuple[ConcatDataset, DataLoader]:
    """Load the combined train/val set (DRIVE + STARE + CHASE_DB1 + HRF + LES_AV).

    For training with ``balance=True``, a WeightedRandomSampler ensures
    each dataset contributes equally per epoch despite different sizes.

    Parameters
    ----------
    target      : output format
    split       : "train" or "val"
    batch_size  : batch size
    num_workers : DataLoader workers
    resize      : (H, W) — required for cross-dataset batching
    train_frac  : fraction of each dataset used for training (rest → val)
    balance     : use inverse-frequency weighted sampling for training
    **kwargs    : forwarded to RetinalFundusDataset (transform, tolerance, …)

    Returns
    -------
    (ConcatDataset, DataLoader)

    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    shared_pre = kwargs.pop("fundus_preprocessor", None) or FundusPreprocessor()
    shared_ext = kwargs.pop("centerline_extractor", None) or CenterlineExtractor()

    sub_datasets: List[RetinalFundusDataset] = []
    for name in TRAIN_DATASETS:
        try:
            root = get_root(name)
        except (KeyError, FileNotFoundError) as exc:
            logger.warning("Skipping %s: %s", name, exc)
            continue
        try:
            ds = RetinalFundusDataset(
                str(root),
                name,
                target=target,
                split=split,
                train_frac=train_frac,
                resize=resize,
                fundus_preprocessor=shared_pre,
                centerline_extractor=shared_ext,
                **kwargs,
            )
            sub_datasets.append(ds)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", name, exc)

    if not sub_datasets:
        raise FileNotFoundError(
            f"No datasets loaded. Check that at least one of {TRAIN_DATASETS} "
            "exists under the data root."
        )

    combined = ConcatDataset(sub_datasets)
    parts = [f"{ds.dataset_name}={len(ds)}" for ds in sub_datasets]
    logger.info("Combined %s: %s  total=%d", split, "  ".join(parts), len(combined))

    # Balanced sampling for training
    sampler = None
    shuffle = False
    if split == "train" and balance:
        weights: List[float] = []
        for ds in sub_datasets:
            w = 1.0 / len(ds)
            weights.extend([w] * len(ds))
        sampler = WeightedRandomSampler(
            weights, num_samples=len(combined), replacement=True
        )
    elif split == "train":
        shuffle = True

    collate_fn = _list_collate if target in ("frangi", "greedy_tracer") else None
    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=target in ("unet", "rl_agent"),
    )
    return combined, loader


def get_test_data(
    dataset_name: str,
    target: str = "rl_agent",
    batch_size: int = 1,
    num_workers: int = 0,
    resize: Optional[Tuple[int, int]] = (512, 512),
    **kwargs,
) -> Tuple[RetinalFundusDataset, DataLoader]:
    """Load an external test dataset in full (no split).

    Parameters
    ----------
    dataset_name : "AV_WIDE" or "DR_HAGIS"
    target       : output format
    batch_size   : batch size
    num_workers  : DataLoader workers
    resize       : (H, W) or None
    **kwargs     : forwarded to RetinalFundusDataset

    Returns
    -------
    (RetinalFundusDataset, DataLoader)

    """
    root = get_root(dataset_name)
    ds = RetinalFundusDataset(
        str(root),
        dataset_name,
        target=target,
        split=None,
        resize=resize,
        **kwargs,
    )
    collate_fn = _list_collate if target in ("frangi", "greedy_tracer") else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=target in ("unet", "rl_agent"),
    )
    return ds, loader
