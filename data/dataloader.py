"""
Unified dataloader for retinal fundus image datasets.

Supported datasets
------------------
DRIVE, STARE, CHASE_DB1, HRF, DR_HAGIS, FIVES, LES_AV, AV_WIDE, IOSTAR

Supported target models
-----------------------
unet           – (1,H,W) CLAHE-preprocessed grayscale + skeleton GT
frangi         – (H,W,3) raw RGB uint8 + binary annotations (numpy)
greedy_tracer  – same as frangi
rl_agent       – (3,H,W) float32 RGB + centerline, distance transform, …

Split logic
-----------
Datasets that ship with an official test directory (e.g. DRIVE, FIVES)
use that directory when ``split="test"`` is requested.  ``split="train"``
and ``split="val"`` partition the training directory only.

If the official test directory exists but contains no annotated samples
(missing ground-truth files — as is the case for DRIVE/test which has
images and FOV masks but **no** manual vessel segmentations), the loader
falls back to a ratio-based split from the training directory with a
warning.

Datasets without an official test directory (e.g. STARE) use a 3-way
ratio-based split over all discovered samples.

root_dir convention
-------------------
Always pass the **top-level** dataset directory::

    load_dataset("data/DRIVE", "DRIVE", split="train")   # ✓
    load_dataset("data/STARE", "STARE", split="train")   # ✓

For backward compatibility, passing the training sub-directory directly
is auto-detected and handled::

    load_dataset("data/DRIVE/training", "DRIVE", split="train")  # also works

Usage
-----
    from data.dataloader import RetinalFundusDataset, load_dataset

    # Single dataset
    ds, loader = load_dataset("data/DRIVE", "DRIVE",
                              target="unet", split="train",
                              batch_size=2, shuffle=True)

    # Test split — DRIVE has no GT in test/, so this falls back
    # to a ratio-based split from training/
    test_ds, test_loader = load_dataset("data/DRIVE", "DRIVE",
                                        target="unet", split="test",
                                        batch_size=1)

    # Flat-directory dataset
    ds, loader = load_dataset("data/STARE", "STARE",
                              target="unet", split="train",
                              batch_size=2)

    # Combine multiple datasets
    from torch.utils.data import ConcatDataset, DataLoader
    drive_ds, _ = load_dataset("data/DRIVE", "DRIVE",
                                target="rl_agent", split="train",
                                resize=(512, 512))
    stare_ds, _ = load_dataset("data/STARE", "STARE",
                                target="rl_agent", split="train",
                                resize=(512, 512))
    combined = DataLoader(ConcatDataset([drive_ds, stare_ds]),
                          batch_size=4, shuffle=True)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .centerline_extraction import CenterlineExtractor
from .fundus_preprocessor import FundusPreprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    """File-system layout descriptor for one retinal fundus dataset.

    Attributes
    ----------
    image_dir     : sub-directory containing fundus images
                    (use ``"."`` when images live in the dataset root)
    vessel_dir    : sub-directory containing vessel ground-truth masks
                    (use ``"."`` when annotations live in the dataset root)
    image_glob    : glob pattern to discover images (e.g. ``"*.tif"``)
    vessel_suffix : appended to the (optionally transformed) image stem
                    to produce the vessel annotation filename
    mask_dir      : optional sub-directory with FOV masks
    mask_suffix   : appended to image stem → FOV mask filename
    stem_rule     : named rule for mapping image stem → vessel stem
                    (currently only ``"drive"`` is special-cased)
    train_subdir  : sub-directory under the dataset root that contains
                    training data.  ``None`` means training data lives
                    directly in the dataset root.
    test_subdir   : sub-directory under the dataset root that contains
                    an official test set.  ``None`` means no official
                    test directory — a ratio-based split is used instead.
                    NOTE: the test directory may lack vessel annotations
                    (e.g. DRIVE/test has images + FOV masks only).

    Note — flat-directory datasets
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``image_dir`` and ``vessel_dir`` both point to the same
    directory (e.g. ``"."``), the glob will match annotation files too.
    These are automatically filtered out because their stems do not
    produce a valid vessel filename (the double-suffix trick).

    Example: STARE has ``im0001.ppm`` and ``im0001.vk.ppm`` side by
    side.  ``im0001.vk.ppm`` has stem ``im0001.vk``; the expected
    vessel file ``im0001.vk.vk.ppm`` does not exist → skipped.
    """

    image_dir: str
    vessel_dir: str
    image_glob: str
    vessel_suffix: str
    mask_dir: Optional[str] = None
    mask_suffix: Optional[str] = None
    stem_rule: Optional[str] = None
    train_subdir: Optional[str] = None
    test_subdir: Optional[str] = None

    # -- helpers ----------------------------------------------------------
    def vessel_filename(self, image_stem: str) -> str:
        """Return the expected vessel annotation filename."""
        return f"{self._transform_stem(image_stem)}{self.vessel_suffix}"

    def mask_filename(self, image_stem: str) -> str:
        """Return the expected FOV mask filename."""
        if self.mask_suffix is None:
            raise ValueError("No mask_suffix configured for this dataset.")
        return f"{image_stem}{self.mask_suffix}"

    def _transform_stem(self, stem: str) -> str:
        if self.stem_rule == "drive":
            return (
                stem.replace("_training", "_manual1")
                    .replace("_test", "_manual1")
            )
        return stem

    @property
    def has_official_test(self) -> bool:
        """Whether this dataset ships with a separate test directory."""
        return self.test_subdir is not None


# ---------------------------------------------------------------------------
# Pre-defined dataset registry
# ---------------------------------------------------------------------------

# DRIVE — training/ has 1st_manual annotations; test/ has images + FOV
# masks only (no vessel ground truth).  Requesting split="test" will
# detect the missing annotations and fall back to a ratio-based split
# from training/.
_DRIVE_CFG = DatasetConfig(
    image_dir="images",
    vessel_dir="1st_manual",
    image_glob="*.tif",
    vessel_suffix=".gif",
    mask_dir="mask",
    mask_suffix="_mask.gif",
    stem_rule="drive",
    train_subdir="training",
    test_subdir="test",
)

# STARE — flat directory: im0001.ppm (image), im0001.vk.ppm (annotation)
_STARE_CFG = DatasetConfig(
    image_dir=".",
    vessel_dir=".",
    image_glob="*.ppm",
    vessel_suffix=".vk.ppm",
)

# CHASEDB1 — flat directory: Image_01L.png (fundus),
# Image_01L_1stHO.png / Image_01L_2ndHO.png (vessel annotations)
_CHASEDB1_CFG = DatasetConfig(
    image_dir=".",
    vessel_dir=".",
    image_glob="*.png",
    vessel_suffix="_1stHO.png",
)

# HRF — images/*.jpg, manual1/*.tif, mask/*.tif
_HRF_CFG = DatasetConfig(
    image_dir="images",
    vessel_dir="manual1",
    image_glob="*.jpg",
    vessel_suffix=".tif",
    mask_dir="mask",
    mask_suffix="_mask.tif",
)

# DRHAGIS — Fundus_Images/*.jpg, Manual_Segmentations/*.png,
# Mask_images/*.png
_DRHAGIS_CFG = DatasetConfig(
    image_dir="Fundus_Images",
    vessel_dir="Manual_Segmentations",
    image_glob="*.jpg",
    vessel_suffix=".png",
    mask_dir="Mask_images",
    mask_suffix=".png",
)

# FIVES — train/ and test/ each with Original/*.png, Ground truth/*.png
_FIVES_CFG = DatasetConfig(
    image_dir="Original",
    vessel_dir="Ground truth",
    image_glob="*.png",
    vessel_suffix=".png",
    train_subdir="train",
    test_subdir="test",
)

# LES-AV — images/*.png, vessel-segmentations/*.png, mask/*.gif
_LES_AV_CFG = DatasetConfig(
    image_dir="images",
    vessel_dir="vessel-segmentations",
    image_glob="*.png",
    vessel_suffix=".png",
    mask_dir="mask",
    mask_suffix=".gif",
)

# AV-WIDE — images/*.png, manual/*.png
_AV_WIDE_CFG = DatasetConfig(
    image_dir="images",
    vessel_dir="manual",
    image_glob="*.png",
    vessel_suffix="_vessels.png",
)

# IOSTAR — image/*.jpg, GT/*.tif
_IOSTAR_CFG = DatasetConfig(
    image_dir="image",
    vessel_dir="GT",
    image_glob="*.jpg",
    vessel_suffix=".tif",
)

DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    # Primary keys
    "DRIVE":     _DRIVE_CFG,
    "STARE":     _STARE_CFG,
    "CHASE_DB1": _CHASEDB1_CFG,
    "HRF":       _HRF_CFG,
    "DR_HAGIS":  _DRHAGIS_CFG,
    "FIVES":     _FIVES_CFG,
    "LES_AV":    _LES_AV_CFG,
    "AV_WIDE":   _AV_WIDE_CFG,
    "IOSTAR":    _IOSTAR_CFG,
    # Aliases (common alternate spellings)
    "CHASEDB1":  _CHASEDB1_CFG,
    "DRHAGIS":   _DRHAGIS_CFG,
}


def register_dataset(name: str, config: DatasetConfig) -> None:
    """Add (or overwrite) a dataset configuration at runtime.

    >>> register_dataset("MY_DATA", DatasetConfig(
    ...     image_dir="img", vessel_dir="gt",
    ...     image_glob="*.png", vessel_suffix="_gt.png",
    ... ))
    """
    DATASET_REGISTRY[name.upper().replace("-", "_")] = config


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------
def _list_collate(batch: list) -> list:
    """Identity collate — keeps numpy dicts as a plain list."""
    return batch


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------
class RetinalFundusDataset(Dataset):
    """Unified PyTorch ``Dataset`` for retinal fundus images.

    Parameters
    ----------
    root_dir : str
        Path to the **top-level** dataset directory (e.g. ``"data/DRIVE"``).
        For datasets with ``train_subdir`` / ``test_subdir`` the loader
        resolves the correct sub-directory automatically.

        For backward compatibility, passing the training sub-directory
        directly (e.g. ``"data/DRIVE/training"``) is auto-detected.
    dataset_name : str
        One of the keys in :data:`DATASET_REGISTRY` (case-insensitive,
        hyphens / spaces accepted, e.g. ``"CHASE-DB1"``).
    target : str
        Output format — ``"unet"``, ``"frangi"``, ``"greedy_tracer"``,
        or ``"rl_agent"``.
    split : str or None
        ``"train"`` / ``"val"`` / ``"test"``.  ``None`` returns all
        samples from the training directory.

        * Datasets **with** ``test_subdir`` (DRIVE, FIVES):
          ``"test"`` loads from the test sub-directory;
          ``"train"``/``"val"`` split the training sub-directory.
        * Datasets **without** a separate test directory:
          all three splits use a ratio-based partition.
        * If the official test directory exists but has no annotated
          samples, the loader falls back with a warning.
    split_ratios : tuple
        ``(train, val, test)`` fractions (must sum to 1).
    resize : tuple or None
        Optional ``(H, W)`` to resize every image/mask pair.
    preprocessor : FundusPreprocessor or None
        Shared preprocessor instance (created with defaults if ``None``).
    centerline_extractor : CenterlineExtractor or None
        Shared extractor (created with defaults if ``None``).
    tolerance : float
        Distance-transform clipping radius (used by ``rl_agent`` target).
    cache_centerlines : bool
        Persist skeletonised centerlines to ``<train_root>/centerlines_cache/``.
    transform : albumentations.Compose or None
        Optional augmentation pipeline.  For ``target="unet"`` this is
        applied after CLAHE preprocessing (on uint8) and should declare
        ``additional_targets={'fov': 'mask', 'thick_gt': 'mask'}``.
    """

    VALID_TARGETS = ("unet", "frangi", "greedy_tracer", "rl_agent")

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        target: str = "rl_agent",
        split: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        resize: Optional[Tuple[int, int]] = None,
        preprocessor: Optional[FundusPreprocessor] = None,
        centerline_extractor: Optional[CenterlineExtractor] = None,
        tolerance: float = 2.0,
        cache_centerlines: bool = True,
        transform=None,
    ):
        if target not in self.VALID_TARGETS:
            raise ValueError(
                f"target must be one of {self.VALID_TARGETS}, got '{target}'"
            )

        canon = dataset_name.upper().replace("-", "_").replace(" ", "_")
        if canon not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Registered: {sorted(set(DATASET_REGISTRY))}"
            )

        self.dataset_name = canon
        self.cfg = DATASET_REGISTRY[canon]
        self.target = target
        self.resize = resize
        self.tolerance = tolerance

        self.preprocessor = preprocessor or FundusPreprocessor()
        self.cl_extractor = centerline_extractor or CenterlineExtractor()
        self.transform = transform

        # ----------------------------------------------------------
        # Resolve base / train / test roots
        # ----------------------------------------------------------
        self._base_root, self._train_root, self._test_root = \
            self._resolve_roots(Path(root_dir), self.cfg)

        # ----------------------------------------------------------
        # Discover samples & apply split
        # ----------------------------------------------------------
        if split == "test" and self.cfg.has_official_test:
            self.root, self.samples = self._load_test_split(split_ratios)
        else:
            self.root = self._train_root
            self.samples = self._discover_samples()

            if split is not None:
                if self.cfg.has_official_test:
                    # Test lives in separate dir → only train/val here
                    self.samples = self._apply_train_val_split(
                        self.samples, split, split_ratios,
                    )
                else:
                    # No official test → 3-way ratio split
                    self.samples = self._apply_split(
                        self.samples, split, split_ratios,
                    )

        if not self.samples:
            raise FileNotFoundError(
                f"No valid samples found for {canon} in {self.root}. "
                f"Expected images in '{self.cfg.image_dir}/' matching "
                f"'{self.cfg.image_glob}' with vessel annotations in "
                f"'{self.cfg.vessel_dir}/'."
            )

        # ----------------------------------------------------------
        # Centerline caches (always stored under the training root)
        # ----------------------------------------------------------
        self._cl_mem: Dict[str, np.ndarray] = {}
        self._cache_dir: Optional[Path] = None
        if cache_centerlines:
            self._cache_dir = self._train_root / "centerlines_cache"
            self._cache_dir.mkdir(exist_ok=True)

        logger.info(
            "%s  %d samples  target=%s  split=%s  root=%s",
            canon, len(self.samples), target, split, self.root,
        )

    # ------------------------------------------------------------------
    # Root resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_roots(
        root_dir: Path, cfg: DatasetConfig,
    ) -> Tuple[Path, Path, Optional[Path]]:
        """Return ``(base_root, train_root, test_root)``.

        Handles two calling conventions:
          1. ``root_dir`` is the top-level dataset dir (preferred)
             e.g. ``data/DRIVE``
          2. ``root_dir`` is the training sub-dir (backward compat)
             e.g. ``data/DRIVE/training``
        """
        if cfg.train_subdir is not None:
            subdir_path = root_dir / cfg.train_subdir
            if subdir_path.is_dir():
                # Convention 1: user passed "data/DRIVE"
                base_root = root_dir
                train_root = subdir_path
            elif root_dir.name == cfg.train_subdir:
                # Convention 2: user passed "data/DRIVE/training"
                base_root = root_dir.parent
                train_root = root_dir
                logger.info(
                    "Auto-detected: root_dir points to '%s/' sub-directory. "
                    "Preferred usage: pass the parent directory '%s'.",
                    cfg.train_subdir, base_root,
                )
            else:
                # Best effort: treat root_dir as-is
                base_root = root_dir
                train_root = root_dir
                logger.warning(
                    "Expected train sub-directory '%s' under %s — not found. "
                    "Using root_dir directly.",
                    cfg.train_subdir, root_dir,
                )
        else:
            # No train_subdir → training data lives in root
            base_root = root_dir
            train_root = root_dir

        test_root: Optional[Path] = None
        if cfg.test_subdir is not None:
            test_root = base_root / cfg.test_subdir

        return base_root, train_root, test_root

    # ------------------------------------------------------------------
    # Load test split (from official test dir or fallback)
    # ------------------------------------------------------------------
    def _load_test_split(
        self,
        split_ratios: Tuple[float, float, float],
    ) -> Tuple[Path, List[Dict[str, Any]]]:
        """Try official test directory; fall back to ratio split."""
        if self._test_root is not None and self._test_root.is_dir():
            self.root = self._test_root
            samples = self._discover_samples()

            if samples:
                logger.info(
                    "Using official test directory: %s (%d samples)",
                    self._test_root, len(samples),
                )
                return self._test_root, samples

            # Test dir exists but no annotated samples
            logger.warning(
                "Official test dir %s has no annotated samples "
                "(missing ground truth?) — falling back to "
                "ratio-based split from training data.",
                self._test_root,
            )

        # Fallback: ratio-based split from training data
        self.root = self._train_root
        samples = self._discover_samples()
        samples = self._apply_split(samples, "test", split_ratios)
        return self._train_root, samples

    # ------------------------------------------------------------------
    # Sample discovery
    # ------------------------------------------------------------------
    def _discover_samples(self) -> List[Dict[str, Any]]:
        image_dir = self.root / self.cfg.image_dir
        vessel_dir = self.root / self.cfg.vessel_dir
        mask_dir = (
            (self.root / self.cfg.mask_dir) if self.cfg.mask_dir else None
        )

        samples: List[Dict[str, Any]] = []
        for img_path in sorted(image_dir.glob(self.cfg.image_glob)):
            stem = img_path.stem
            vessel_path = vessel_dir / self.cfg.vessel_filename(stem)

            if not vessel_path.exists():
                logger.debug(
                    "Vessel annotation missing for %s — skipped", img_path.name
                )
                continue

            entry: Dict[str, Any] = {
                "id": stem,
                "image": img_path,
                "vessel": vessel_path,
            }

            if mask_dir is not None and self.cfg.mask_suffix is not None:
                mask_path = mask_dir / self.cfg.mask_filename(stem)
                if mask_path.exists():
                    entry["mask"] = mask_path

            samples.append(entry)

        return samples

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_split(
        samples: List[Dict],
        split: str,
        ratios: Tuple[float, float, float],
    ) -> List[Dict]:
        """3-way ratio split for datasets without an official test dir."""
        n = len(samples)
        t = int(ratios[0] * n)
        v = int((ratios[0] + ratios[1]) * n)
        if split == "train":
            return samples[:t]
        if split == "val":
            return samples[t:v]
        if split == "test":
            return samples[v:]
        raise ValueError(f"split must be 'train'/'val'/'test', got '{split}'")

    @staticmethod
    def _apply_train_val_split(
        samples: List[Dict],
        split: str,
        ratios: Tuple[float, float, float],
    ) -> List[Dict]:
        """Train/val split only — used when an official test dir exists.

        Re-normalises ``ratios[0]`` and ``ratios[1]`` so they span the
        full sample list (the test portion lives in a separate directory).
        """
        if split not in ("train", "val"):
            raise ValueError(
                f"This dataset has an official test split — "
                f"use split='train' or 'val' here, got '{split}'"
            )
        n = len(samples)
        train_frac = ratios[0] / (ratios[0] + ratios[1])
        # Ensure at least 1 sample in each split
        t = max(1, min(int(train_frac * n), n - 1))
        if split == "train":
            return samples[:t]
        return samples[t:]

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        """Load image as ``(H, W, 3)`` uint8 RGB."""
        return np.array(Image.open(path).convert("RGB"))

    @staticmethod
    def _load_gray(path: Path) -> np.ndarray:
        """Load image as ``(H, W)`` uint8 grayscale."""
        return np.array(Image.open(path).convert("L"))

    def _load_vessel(self, path: Path) -> np.ndarray:
        """Binary vessel mask — ``(H, W)`` float32 {0, 1}."""
        return (self._load_gray(path) > 127).astype(np.float32)

    def _load_fov(self, path: Path) -> np.ndarray:
        """FOV mask — ``(H, W)`` uint8 {0, 255}."""
        return (self._load_gray(path) > 127).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # Centerline extraction (memory + disk cache)
    # ------------------------------------------------------------------
    def _get_centerline(self, sid: str, vessel: np.ndarray) -> np.ndarray:
        """Return ``(H, W)`` float32 skeleton, using cache when possible."""
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

    # ------------------------------------------------------------------
    # FOV mask (load or derive)
    # ------------------------------------------------------------------
    def _get_fov(self, sample: Dict, rgb: np.ndarray) -> np.ndarray:
        """Return ``(H, W)`` uint8 {0, 255} FOV mask."""
        if "mask" in sample:
            return self._load_fov(sample["mask"])
        # Derive automatically from the green channel
        green = self.preprocessor.extract_green_channel(rgb)
        if green.dtype != np.uint8:
            green = np.clip(green * 255, 0, 255).astype(np.uint8)
        return self.preprocessor.create_fov_mask(green)

    # ------------------------------------------------------------------
    # Optional resize
    # ------------------------------------------------------------------
    def _maybe_resize(
        self,
        *arrays: np.ndarray,
        interp: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, ...]:
        if self.resize is None:
            return arrays
        h, w = self.resize
        out = []
        for i, arr in enumerate(arrays):
            if interp is not None and i < len(interp):
                flag = interp[i]
            elif arr.ndim == 2:
                flag = cv2.INTER_NEAREST
            else:
                flag = cv2.INTER_LINEAR
            out.append(cv2.resize(arr, (w, h), interpolation=flag))
        return tuple(out)

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sid = sample["id"]

        rgb = self._load_rgb(sample["image"])
        vessel = self._load_vessel(sample["vessel"])
        fov = self._get_fov(sample, rgb)

        # Resize if requested
        if self.resize is not None:
            rgb, vessel, fov = self._maybe_resize(
                rgb, vessel, fov,
                interp=[cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_NEAREST],
            )
            vessel = (vessel > 0.5).astype(np.float32)

        # Dispatch to target-specific formatter
        return getattr(self, f"_fmt_{self.target}")(sid, rgb, vessel, fov)

    # ------------------------------------------------------------------
    # Target-specific formatters
    # ------------------------------------------------------------------
    def _fmt_unet(
        self, sid: str, rgb: np.ndarray, vessel: np.ndarray, fov: np.ndarray,
    ) -> Dict[str, Any]:
        """U-Net CNN baseline format.

        Returns single-channel CLAHE-preprocessed image and the
        skeletonised centerline as training target.

        When a ``transform`` (albumentations Compose) is set, it is
        applied after preprocessing but before tensor conversion.  The
        transform should declare ``additional_targets`` for ``fov``
        and ``thick_gt`` (both as ``'mask'``).
        """
        ext_mask = fov if fov.max() > 0 else None
        preprocessed = self.preprocessor.preprocess(rgb, external_mask=ext_mask)
        cl = self._get_centerline(sid, vessel)
        fov_f = (fov > 0).astype(np.float32)

        # Apply augmentation (operates on uint8, returns uint8)
        if self.transform is not None:
            img_u8 = np.clip(preprocessed * 255, 0, 255).astype(np.uint8)
            aug = self.transform(
                image=img_u8, mask=cl, fov=fov_f, thick_gt=vessel,
            )
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

    def _fmt_frangi(
        self, sid: str, rgb: np.ndarray, vessel: np.ndarray, fov: np.ndarray,
    ) -> Dict[str, Any]:
        """Frangi vesselness filter baseline format.

        Returns raw RGB image (the baseline handles its own preprocessing)
        and uint8 binary annotations for evaluation.
        """
        cl = self._get_centerline(sid, vessel)
        return {
            "id": sid,
            "image": rgb,
            "vessel_mask": (vessel * 255).astype(np.uint8),
            "centerline": (cl * 255).astype(np.uint8),
            "fov_mask": fov,
        }

    def _fmt_greedy_tracer(
        self, sid: str, rgb: np.ndarray, vessel: np.ndarray, fov: np.ndarray,
    ) -> Dict[str, Any]:
        """Greedy tracer baseline format (identical to Frangi)."""
        return self._fmt_frangi(sid, rgb, vessel, fov)

    def _fmt_rl_agent(
        self, sid: str, rgb: np.ndarray, vessel: np.ndarray, fov: np.ndarray,
    ) -> Dict[str, Any]:
        """RL agent format.

        Returns normalised RGB tensor (with CLAHE-enhanced green channel)
        and all annotation channels needed by
        :class:`rl_environment.vessel_env.VesselTracingEnv`.

        Also returns ``image_orig`` — the normalised RGB *before* green-
        channel enhancement — for visualisation overlays.
        """
        img_f = rgb.astype(np.float32) / 255.0

        # Save original RGB before green-channel enhancement
        img_orig = img_f.copy()

        # Enhance green channel with CLAHE for better vessel contrast
        ext_mask = fov if fov.max() > 0 else None
        enhanced_green = self.preprocessor.preprocess(rgb, external_mask=ext_mask)
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
# Loader factory
# ---------------------------------------------------------------------------
def load_dataset(
    root_dir: str,
    dataset_name: str,
    target: str = "rl_agent",
    split: Optional[str] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    **dataset_kwargs,
) -> Tuple[RetinalFundusDataset, DataLoader]:
    """Create a dataset and its DataLoader in one call.

    All ``**dataset_kwargs`` are forwarded to
    :class:`RetinalFundusDataset` (e.g. ``transform``,
    ``split_ratios``, ``resize``).

    Returns
    -------
    (dataset, loader) : tuple
    """
    ds = RetinalFundusDataset(
        root_dir, dataset_name,
        target=target, split=split,
        **dataset_kwargs,
    )
    collate_fn = _list_collate if target in ("frangi", "greedy_tracer") else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=target in ("unet", "rl_agent"),
    )
    return ds, loader