"""config.py — Unified project configuration.

Single source of truth: all hyperparameters live inside MODEL_CONFIG or
SEED_CONFIG.  Scripts unpack what they need at import time.
"""

import copy

import torch

from data.dataloader import OUTPUT_DIR as OUTPUT_BASE
from data.dataloader import WEIGHTS_DIR

# ═══════════════════════════════════════════════════════════════════════
# DEVICE
# ═══════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════
# WEIGHT / CHECKPOINT PATHS
# ═══════════════════════════════════════════════════════════════════════
PPO_WEIGHTS_PATH       = str(WEIGHTS_DIR / "ppo_policy.pt")
IMITATION_WEIGHTS_PATH = str(WEIGHTS_DIR / "imitation_policy.pt")
SEED_WEIGHTS_PATH      = str(WEIGHTS_DIR / "seed_detector.pt")
PPO_LOG_PATH           = str(WEIGHTS_DIR / "ppo_log.csv")
IMITATION_LOG_PATH     = str(WEIGHTS_DIR / "imitation_log.csv")

# ═══════════════════════════════════════════════════════════════════════
# MASTER MODEL / ENVIRONMENT / REWARD CONFIG
# ═══════════════════════════════════════════════════════════════════════
MODEL_CONFIG = {
    "policy": {
        "hidden_dim": 256,
        "lstm_hidden": 256,
        "head_hidden": 128,
        "use_lstm": False,
        "use_junction_aux": True,
        "dropout": 0.05,
        "encoder_type": "cnn",
    },
    "environment": {
        "observation_size": 65,
        "tolerance": 2.0,
        "use_vesselness": False,
        "use_unet_prior": False,
        "use_curvature": True,
        "use_junction": True,
        "use_prev_action": True,
        "use_global_visited": True,
        "use_prior_coverage": True,
        "max_steps_per_episode": 500,
        "max_off_track_streak": 3,
        "step_size": 1,
        "momentum": 0.0,
    },
    "reward": {
        # ── Coverage ──────────────────────────────────────────────────────
        # β × raw new-pixel count.  The dominant DENSE signal.
        # Typical 2–4 new px/step → 0.6–1.2/step → ~270 per 500-step episode.
        # Naturally zero on revisits — rewards forward progress without
        # rewarding loitering.  (Earlier normalisation by total_gt crushed
        # this to 0.006/step and the agent had nothing to learn from.)
        "beta_coverage": 0.3,

        # ── Proximity (continuous) ────────────────────────────────────────
        # α × max(0, 1 − D(p)/τ): positive level signal within tolerance,
        # peaks at α on the skeleton, zero at / beyond τ.  Complements the
        # shaping term (which is a delta) with a continuous "be near the
        # centerline" signal.  Paper eq. 1.
        "alpha_near": 0.01,

        # ── Off-vessel penalty ────────────────────────────────────────────
        # Flat, gentle (-0.2): brief off-track detours to reach a branching
        # point stay profitable.  Harsher values (e.g. -1.0) collapse training
        # by making ANY exploration unprofitable.
        "gamma_off": -0.2,

        # ── Revisit penalty ───────────────────────────────────────────────
        # −λ per step landing on an already-visited pixel.  Discourages loops
        # and backtracking.  Small magnitude so valid backtracks (e.g. past a
        # dead end) stay affordable.  Paper eq. 4.
        "lambda_revisit": 0.02,

        # ── Step cost ─────────────────────────────────────────────────────
        # Small constant cost to discourage dithering / standing still.
        "step_cost": -0.01,

        # ── Potential-based shaping ───────────────────────────────────────
        # Ng et al. (1999): γ·Φ(s') − Φ(s),  Φ(s) = −min(dt(s), ε) / ε.
        # Full weight — |Δφ|≈0.01/step, small enough to never compete with
        # coverage, consistent enough to keep the agent centred.
        # shaping_gamma MUST equal training.ppo.gamma (enforced in PPOTrainer).
        "shaping_weight": 1.0,
        "shaping_gamma": 0.99,

        # ── Terminal F-β reward ───────────────────────────────────────────
        # Computed against ``covered_centerline`` (NOT ``trajectory_mask``) so
        # the training signal is a monotonic proxy for clDice.  See
        # ``VesselTracingEnv._compute_fbeta``.  β²=4 weights recall 4×
        # precision, rewarding completeness (what clDice punishes gaps in).
        "terminal_f1_weight": 8.0,
        "terminal_recall_beta_sq": 4.0,

        # ── Stop gate ─────────────────────────────────────────────────────
        "min_stop_coverage": 0.05,
        "early_stop_penalty": -1.0,

        # ── Out-of-bounds penalty ─────────────────────────────────────────
        # Softened from -10: a single OOB should not destabilise training.
        "oob_penalty": -5.0,
    },
    "training": {
        "patience": 100,
        "reward_norm_clip": 10.0,
        "terminal_norm_clip": 5.0,
        "lr_end_factor": 0.1,
        "value_clamp": 10.0,
        "ppo": {
            "lr": 1e-4,
            "lr_warmup_iters": 50,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "entropy_coef": 0.05,
            "value_coef": 0.5,
            "max_grad_norm": 2.0,
            "epochs": 3,
            "mini_batch_size": 512,
            "steps_per_iter": 8192,
            "num_iterations": 1000,
            "eval_every": 20,
            "save_every": 50,
            "lstm_chunk_length": 128,
            "n_envs": 32,
            "target_kl": 0.08,
        },
        "imitation": {
            "lr": 3e-4,
            "batch_size": 512,
            "lstm_batch_size": 16,
            "num_epochs": 15,
            "max_grad_norm": 1.0,
            "use_augment": False,
            "lr_step_size": 5,
            "lr_gamma": 0.5,
            "num_workers": 16,
        },
    },
    "curriculum": {
        "start_difficulty": 0.3,
        "end_difficulty": 1.0,
        "warmup_steps": 50_000,
        "advancement_window": 200,
        "success_min_coverage_base": 0.02,
        "success_min_precision": 0.5,
        "stages": [
            {
                "name": "easy",
                "difficulty": 0.3,
                "min_success_rate": 0.3,
                "min_episodes": 50,
                "max_off_track_streak": 15,
                "max_steps_per_episode": 300,
                "entropy_coef": 0.05,
            },
            {
                "name": "medium",
                "difficulty": 0.6,
                "min_success_rate": 0.2,
                "min_episodes": 100,
                "max_off_track_streak": 12,
                "max_steps_per_episode": 500,
                "entropy_coef": 0.03,
            },
            {
                "name": "full",
                "difficulty": 1.0,
                "min_success_rate": 0.1,
                "min_episodes": 200,
                "max_off_track_streak": 10,
                "max_steps_per_episode": 700,
                "entropy_coef": 0.015,
            },
        ],
    },
    "inference": {
        "mode": "e2e",
        "max_traces": 80,
        "min_cov_gain": 0.0001,
        "dilation_radius": 5,
        "n_ring_seeds": 0,
        "ring_inset_px": 40,
        # Environment overrides applied by get_inference_config()
        "max_steps_per_episode": 700,
        "max_off_track_streak": 3,
    },
}

# ═══════════════════════════════════════════════════════════════════════
# SEED DETECTOR CONFIG
# ═══════════════════════════════════════════════════════════════════════
SEED_CONFIG = {
    "seed_detector": {
        "base_ch": 16,
        "nms_radius": 10,
        "confidence_threshold": 0.3,
        "top_k_seeds": 80,
        "vessel_gate_threshold": 0.35,
        "snap_radius": 5,
        "use_frangi_supplement": True,
        "frangi_spacing": 20,
        "seeds_per_skeleton_px": 15,
        "max_adaptive_seeds": 600,
    },
    "training": {
        "sigma": 2.0,
        "num_epochs": 30,
        "batch_size": 4,
        "lr": 1e-4,
        "aux_spacing": 20,
    },
}

# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE ALIASES
# ═══════════════════════════════════════════════════════════════════════
TOLERANCE = MODEL_CONFIG["environment"]["tolerance"]
OBS_SIZE  = MODEL_CONFIG["environment"]["observation_size"]


def get_config() -> dict:
    """Return a deep copy of MODEL_CONFIG for safe mutation (e.g. sweeps)."""
    return copy.deepcopy(MODEL_CONFIG)


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_inference_config() -> dict:
    """Return a policy config tuned for inference (no dropout, longer episodes).

    Differences from the training config:
    - ``policy.dropout`` set to 0.0 — disables regularisation at test time.
    - ``environment.max_steps_per_episode`` raised to the inference value so
      the agent can complete long vessel paths without early truncation.
    - ``environment.max_off_track_streak`` kept at 3 (same as training).

    Used by ``scripts/run_rl_tracing.py`` and ``scripts/diagnose_seed_detector.py``.
    """
    cfg = copy.deepcopy(MODEL_CONFIG)
    inf = cfg["inference"]
    cfg["environment"]["max_steps_per_episode"] = inf["max_steps_per_episode"]
    cfg["environment"]["max_off_track_streak"]  = inf["max_off_track_streak"]
    cfg["environment"]["step_size"]             = 1
    cfg["policy"]["dropout"]                    = 0.0
    return cfg


def get_seed_inference_config() -> dict:
    """Return a seed-detector config tuned for inference.

    Differences from the training config:
    - ``nms_radius`` = 10 (was 15 "for cleaner selection", but that filter was too
      aggressive: two valid seeds on parallel vessels 15px apart were merged into
      one, leaving some images with only 20-30 seeds).  10 px still suppresses
      duplicate peaks but keeps seeds on nearby branches distinct.
    - ``frangi_spacing`` = 12 (was 20): denser auxiliary seeds on long unbranched
      segments.  A 100 px branch now gets ~8 supplementary seeds instead of ~5.
    - ``top_k_seeds`` set to ``max_traces`` — matches the inference budget.
    - ``confidence_threshold`` = 0.08: catches low-confidence thin-vessel seeds.

    Used by ``scripts/run_rl_tracing.py``.
    """
    cfg = copy.deepcopy(SEED_CONFIG)
    cfg["seed_detector"].update({
        "nms_radius": 10,
        "frangi_spacing": 12,
        "top_k_seeds": MODEL_CONFIG["inference"]["max_traces"],
        "confidence_threshold": 0.08,
    })
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION METRIC COLUMNS
# ═══════════════════════════════════════════════════════════════════════
METRIC_COLS = [
    "iou",
    "clDice",
    "betti_0_error_raw",
    "betti_0_error_postproc",
    "hd95",
    "f1@1px",
    "precision@1px",
    "recall@1px",
    "f1@2px",
    "precision@2px",
    "recall@2px",
    "f1@3px",
    "precision@3px",
    "recall@3px",
]
CSV_COLUMNS = ["image_id"] + METRIC_COLS
