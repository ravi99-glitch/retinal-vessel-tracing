"""config.py — Unified project configuration.

Single source of truth for all hyperparameters, paths, and architecture
configs used across training, evaluation, and inference scripts.

Usage:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import MODEL_CONFIG, DEVICE, ...
"""

import copy

import torch

from data.dataloader import OUTPUT_DIR as OUTPUT_BASE  # re-exported
from data.dataloader import WEIGHTS_DIR

# ═══════════════════════════════════════════════════════════════════════
# DEVICE
# ═══════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════
# COMMON CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
TOLERANCE = 2.0
OBS_SIZE  = 65
MAX_STEPS = 2000                       # max steps per PPO training episode

# ═══════════════════════════════════════════════════════════════════════
# WEIGHT / CHECKPOINT PATHS
# ═══════════════════════════════════════════════════════════════════════
PPO_WEIGHTS_PATH       = str(WEIGHTS_DIR / "ppo_policy.pt")
IMITATION_WEIGHTS_PATH = str(WEIGHTS_DIR / "imitation_policy.pt")
SEED_WEIGHTS_PATH      = str(WEIGHTS_DIR / "seed_detector.pt")
PPO_LOG_PATH           = str(WEIGHTS_DIR / "ppo_log.txt")

# ═══════════════════════════════════════════════════════════════════════
# MASTER MODEL / ENVIRONMENT / REWARD CONFIG
# ═══════════════════════════════════════════════════════════════════════
# Consumed by ActorCriticNetwork, VesselTracingEnv, reward shaping,
# curriculum scheduler, imitation trainer, and PPO trainer.

MODEL_CONFIG = {
    "policy": {
        "hidden_dim":   128,
        "lstm_hidden":  128,
        "use_lstm":     True,
        "dropout":      0.05,
        "encoder_type": "cnn",
    },
    "environment": {
        "observation_size":      OBS_SIZE,
        "tolerance":             TOLERANCE,
        "use_vesselness":        False,
        "max_steps_per_episode": MAX_STEPS,
        "max_off_track_streak":  8,
        "step_size":             1,
        "momentum":              0.0,
    },
    "reward": {
        "alpha_near":                 0.1,
        "beta_coverage":              1.0,
        "gamma_off":                 -0.5,
        "lambda_revisit":            -0.3,
        "step_cost":                 -0.01,
        "direction_bonus":            0.05,
        "terminal_f1_weight":         5.0,
        "use_potential_shaping":      False,
        "smoothness_weight":          0.3,
        "oscillation_weight":         0.6,
        "oscillation_window":         6,
        "off_vessel_distance_weight": 0.3,
        "bridge_penalty":            -3.0,
        "betti0_episode_weight":      2.0,
        "local_merge_reward":         1.5,
        "local_merge_radius":         5,
        "betti0_check_interval":      50,
        "betti0_delta_weight":        0.5,
    },
    "training": {
        "ppo": {"gamma": 0.99},
        "patience": 100,
        # "entropy_coef": PPO_ENTROPY_COEF,       # default; overridden per stage
    },
    "curriculum": {
        "start_difficulty": 0.2,
        "end_difficulty":   1.0,
        "warmup_steps":     500_000,
        "stages": [
            {
                "name":                "thick_straight",
                "difficulty":          0.2,
                "min_success_rate":    0.6,
                "min_episodes":        100,
                "description":         "Thick straight vessels — learn smooth following",
                "smoothness_weight":   0.6,
                "max_off_track_streak": 2,
                "max_steps_per_episode": 300,
                "entropy_coef":        0.08,
            },
            {
                "name":                "medium_branching",
                "difficulty":          0.4,
                "min_success_rate":    0.5,
                "min_episodes":        200,
                "description":         "Medium vessels with branches",
                "smoothness_weight":   0.4,
                "max_off_track_streak": 3,
                "max_steps_per_episode": 400,
                "entropy_coef":        0.05,
            },
            {
                "name":                "thin_vessels",
                "difficulty":          0.6,
                "min_success_rate":    0.4,
                "min_episodes":        300,
                "description":         "Thin vessels and junctions",
                "smoothness_weight":   0.3,
                "max_off_track_streak": 3,
                "max_steps_per_episode": 500,
                "entropy_coef":        0.03,
            },
            {
                "name":                "capillaries",
                "difficulty":          0.8,
                "min_success_rate":    0.35,
                "min_episodes":        400,
                "description":         "Capillaries and low-contrast regions",
                "smoothness_weight":   0.2,
                "max_off_track_streak": 4,
                "max_steps_per_episode": 600,
                "entropy_coef":        0.02,
            },
            {
                "name":                "full",
                "difficulty":          1.0,
                "min_success_rate":    0.3,
                "min_episodes":        500,
                "description":         "Full difficulty",
                "smoothness_weight":   0.2,
                "max_off_track_streak": 5,
                "max_steps_per_episode": 600,
                "entropy_coef":        0.01,
            },
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════
# SEED DETECTOR CONFIG  (architecture + training-time detection defaults)
# ═══════════════════════════════════════════════════════════════════════
SEED_CONFIG = {
    "seed_detector": {
        "base_ch":              16,
        "nms_radius":           10,
        "confidence_threshold": 0.3,
        "top_k_seeds":          50,
    }
}

# ═══════════════════════════════════════════════════════════════════════
# INFERENCE / EVALUATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════
INFERENCE_MODE  = "e2e"                # 'gt' | 'e2e'
MAX_TRACES      = 80
MIN_COV_GAIN    = 0.001
DILATION_RADIUS = 3
N_RING_SEEDS    = 24
RING_INSET_PX   = 40

# Environment overrides at inference (shorter episodes, bigger steps)
_INFERENCE_ENV_OVERRIDES = {
    "max_steps_per_episode": 700,
    "max_off_track_streak":  3,
    "step_size":             2,
}

# Seed detection overrides at inference (wider NMS, more seeds allowed)
_INFERENCE_SEED_OVERRIDES = {
    "nms_radius":  15,
    "top_k_seeds": MAX_TRACES,
}


def get_inference_config():
    """Return MODEL_CONFIG with inference-time environment overrides applied."""
    cfg = copy.deepcopy(MODEL_CONFIG)
    cfg["environment"].update(_INFERENCE_ENV_OVERRIDES)
    cfg["policy"]["dropout"] = 0.0
    return cfg


def get_seed_inference_config():
    """Return SEED_CONFIG with inference-time detection overrides applied."""
    cfg = copy.deepcopy(SEED_CONFIG)
    cfg["seed_detector"].update(_INFERENCE_SEED_OVERRIDES)
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION METRIC COLUMNS  (shared across all eval / baseline scripts)
# ═══════════════════════════════════════════════════════════════════════
METRIC_COLS = [
    "iou", "clDice",
    "betti_0_error_raw", "betti_0_error_postproc",
    "hd95",
    "f1@1px",  "precision@1px",  "recall@1px",
    "f1@2px",  "precision@2px",  "recall@2px",
    "f1@3px",  "precision@3px",  "recall@3px",
]
CSV_COLUMNS = ["image_id"] + METRIC_COLS

# ═══════════════════════════════════════════════════════════════════════
# IMITATION LEARNING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════
IMITATION_LR              = 3e-4
IMITATION_BATCH_SIZE      = 128
IMITATION_LSTM_BATCH_SIZE = 16
IMITATION_NUM_EPOCHS      = 30
IMITATION_USE_AUGMENT     = False

# ═══════════════════════════════════════════════════════════════════════
# PPO TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════
PPO_LR              = 1e-4
PPO_GAMMA           = 0.99
PPO_GAE_LAMBDA      = 0.95
PPO_CLIP_EPS        = 0.1
PPO_ENTROPY_COEF    = 0.05
PPO_VALUE_COEF      = 0.5
PPO_MAX_GRAD_NORM   = 1.0
PPO_EPOCHS          = 4
PPO_MINI_BATCH_SIZE = 256
PPO_STEPS_PER_ITER  = 4096
PPO_NUM_ITERATIONS  = 1000
PPO_EVAL_EVERY      = 25
PPO_SAVE_EVERY      = 50
LSTM_CHUNK_LENGTH   = 32

# ═══════════════════════════════════════════════════════════════════════
# SEED DETECTOR TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════
SEED_SIGMA      = 3.0
SEED_NUM_EPOCHS = 30
SEED_BATCH_SIZE = 4
SEED_LR         = 1e-4