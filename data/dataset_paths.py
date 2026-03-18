# data/dataset_paths.py
"""
Central registry of dataset root directories.

Edit the paths below to match your local / cluster file system.
Alternatively, override any path via environment variable:

    export RETINAL_DATA_DRIVE="/my/custom/path/DRIVE"

Every ``run_*.py`` script imports from here instead of
hardcoding its own ``DATA_ROOT``.
"""

import os
from pathlib import Path

# ── Base directory (parent of all dataset folders) ───────
# Override with:  export RETINAL_DATA_ROOT="/scratch/data"
_DEFAULT_BASE = Path(__file__).resolve().parent  # …/data/

DATA_BASE = Path(os.environ.get("RETINAL_DATA_ROOT", _DEFAULT_BASE))

# ── Per-dataset roots ───────────────────────────────────
# Each entry points to the top-level dataset directory.
# The dataloader resolves train/test sub-dirs internally.
#
# Override any single dataset with:
#   export RETINAL_DATA_DRIVE="/other/path/DRIVE"

DATASET_ROOTS = {
    "DRIVE":     DATA_BASE / "DRIVE",
    "STARE":     DATA_BASE / "STARE",
    "CHASE_DB1": DATA_BASE / "CHASEDB1",
    "HRF":       DATA_BASE / "HRF",
    "DR_HAGIS":  DATA_BASE / "DRHAGIS",
    "FIVES":     DATA_BASE / "FIVES",
    "LES_AV":    DATA_BASE / "LES-AV",
    "AV_WIDE":   DATA_BASE / "AV-WIDE",
    "IOSTAR":    DATA_BASE / "IOSTAR",
}

# Apply per-dataset env-var overrides
for key in list(DATASET_ROOTS):
    env_key = f"RETINAL_DATA_{key}"
    if env_key in os.environ:
        DATASET_ROOTS[key] = Path(os.environ[env_key])


# ── Project directories (weights, results) ────────────
# Override with:  export RETINAL_WEIGHTS_DIR="/scratch/weights"
#                 export RETINAL_OUTPUT_DIR="/scratch/results"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # …/retinal-vessel-tracing-main/

WEIGHTS_DIR = Path(os.environ.get("RETINAL_WEIGHTS_DIR", _PROJECT_ROOT / "weights"))
OUTPUT_DIR  = Path(os.environ.get("RETINAL_OUTPUT_DIR",  _PROJECT_ROOT / "results"))


def get_root(dataset_name: str) -> Path:
    """Return the root directory for a dataset.

    Parameters
    ----------
    dataset_name : str
        Registry key (case-insensitive, hyphens accepted).

    Raises
    ------
    KeyError
        If the dataset is not in ``DATASET_ROOTS``.
    FileNotFoundError
        If the resolved path does not exist on disk.
    """
    canon = dataset_name.upper().replace("-", "_").replace(" ", "_")
    if canon not in DATASET_ROOTS:
        raise KeyError(
            f"No root configured for '{dataset_name}'. "
            f"Known datasets: {sorted(DATASET_ROOTS)}"
        )
    root = DATASET_ROOTS[canon]
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root for {canon} does not exist: {root}\n"
            f"Set RETINAL_DATA_ROOT or RETINAL_DATA_{canon} "
            f"environment variable, or edit data/dataset_paths.py."
        )
    return root