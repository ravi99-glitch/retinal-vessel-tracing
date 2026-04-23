# training/curriculum.py
"""Curriculum learning for progressive training difficulty."""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from data.centerline_extraction import CenterlineExtractor
import numpy as np


@dataclass
class CurriculumStage:
    """A stage in the curriculum."""

    name: str
    difficulty: float
    min_success_rate: float
    min_episodes: int
    description: str = ""
    smoothness_weight: float = 0.0
    max_off_track_streak: int = 3
    max_steps_per_episode: int = 600
    entropy_coef: float = 0.05
    # Optional intra-stage entropy annealing. When ``entropy_coef_end`` is
    # set and ``entropy_anneal_iters > 0``, the trainer linearly interpolates
    # ``entropy_coef`` → ``entropy_coef_end`` over ``entropy_anneal_iters``
    # outer PPO iterations spent inside the stage.
    entropy_coef_end: Optional[float] = None
    entropy_anneal_iters: int = 0
    # Soft off-track tolerance.  When True the per-step off-track penalty
    # ramps linearly with the streak (mild at 1, full at max_off_track)
    # instead of a flat penalty every step.  Set False (default) to keep
    # the original hard penalty (e.g. for the "full" curriculum stage).
    off_track_penalty_ramp: bool = False


class CurriculumManager:
    """Manages curriculum learning for vessel tracing.

    Progressively increases difficulty from easy cases (large, well-defined
    vessels) to hard cases (thin capillaries, pathologies, poor contrast).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialise from the top-level MODEL_CONFIG dict.

        Stage definitions are read from
        ``config["curriculum"]["stages"]`` — a list of dicts whose keys
        match :class:`CurriculumStage` fields.  If the key is missing a
        single default stage is created so nothing breaks.

        Args:
            config: The full MODEL_CONFIG dictionary.
        """
        curriculum_config = config.get("curriculum", {})

        self.start_difficulty = curriculum_config.get("start_difficulty", 0.2)
        self.end_difficulty = curriculum_config.get("end_difficulty", 1.0)
        self.warmup_steps = curriculum_config.get("warmup_steps", 500_000)

        self._cfg = curriculum_config

        self.current_difficulty = self.start_difficulty
        self.total_steps = 0

        # ------------------------------------------------------------------
        # Build CurriculumStage objects from the config dicts
        # ------------------------------------------------------------------
        stage_dicts = curriculum_config.get("stages", None)

        if stage_dicts:
            self.stages: List[CurriculumStage] = [
                CurriculumStage(**sd) for sd in stage_dicts
            ]
        else:
            # Minimal fallback — single stage, full difficulty
            self.stages = [
                CurriculumStage(
                    name="default",
                    difficulty=1.0,
                    min_success_rate=0.3,
                    min_episodes=100,
                    description="Single default stage (no stages configured)",
                )
            ]

        self.current_stage_idx = 0
        self._window_size = curriculum_config.get("advancement_window", 200)
        self._recent_successes: deque = deque(maxlen=self._window_size)

    # ==================================================================
    # Public API
    # ==================================================================

    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty

    def get_current_stage(self) -> CurriculumStage:
        """Return the active curriculum stage."""
        return self.stages[self.current_stage_idx]

    def step(self, success: bool = False):
        """Update curriculum state after one episode.

        Args:
            success: Whether the episode was successful.
        """
        self.total_steps += 1
        self._recent_successes.append(1 if success else 0)

        # Linear warmup of difficulty
        if self.total_steps < self.warmup_steps:
            progress = self.total_steps / self.warmup_steps
            self.current_difficulty = self.start_difficulty + progress * (
                self.end_difficulty - self.start_difficulty
            )
        else:
            self.current_difficulty = self.end_difficulty

        # Check whether we should move to the next stage
        self._check_stage_advancement()

    def is_episode_successful(self, info: Dict[str, Any]) -> bool:
        """Determine whether an episode counts as a success for curriculum advancement.

        Prefers the tolerance-aware centerline F1 stored in ``info["episode_f1"]``
        (computed in VesselTracingEnv.step at episode end) because F1 jointly
        captures precision AND recall and therefore cannot be gamed by traces that
        merely cover a small fraction of the centerline at high local precision.

        Falls back to the old coverage+precision check when episode_f1 is absent
        (e.g. mid-episode info dicts from truncated episodes before the first eval).

        Thresholds scale with stage difficulty so the bar rises progressively:
          easy   (diff=0.3): min_f1 ≈ 0.10 + 0.15×0.3 ≈ 0.145
          medium (diff=0.6): min_f1 ≈ 0.10 + 0.15×0.6 ≈ 0.190
          full   (diff=1.0): min_f1 ≈ 0.10 + 0.15×1.0 ≈ 0.250
        """
        stage = self.get_current_stage()

        ep_f1 = info.get("episode_f1", None)
        if ep_f1 is not None:
            min_f1 = (
                self._cfg.get("success_min_f1_base", 0.10)
                + self._cfg.get("success_min_f1_scale", 0.15) * stage.difficulty
            )
            min_cov = self._cfg.get("success_min_coverage_base", 0.02) * (
                1 + stage.difficulty
            )
            coverage = info.get("coverage_ratio", 0.0)
            return float(ep_f1) >= min_f1 and coverage >= min_cov

        # Fallback: old criterion for info dicts that lack episode_f1
        ep_len = info.get("step_count", 0)
        base = self._cfg.get("success_min_steps_base", 20)
        scale = self._cfg.get("success_min_steps_scale", 30)
        min_length = base + int(stage.difficulty * scale)
        coverage = info.get("coverage_ratio", 0.0)
        precision = info.get("precision", 0.0)
        min_coverage = self._cfg.get("success_min_coverage_base", 0.02) * (
            1 + stage.difficulty
        )
        min_precision = self._cfg.get("success_min_precision", 0.5)
        return ep_len >= min_length and coverage >= min_coverage and precision >= min_precision

    def get_stage_overrides(self) -> Dict[str, Any]:
        """Return config overrides for the current stage.

        PPOTrainer applies these each iteration to dynamically adjust
        reward weights, episode length, and entropy coefficient.

        Returns:
            Nested dict mirroring MODEL_CONFIG structure with only the
            keys that the current stage overrides.
        """
        stage = self.get_current_stage()
        return {
            "reward": {
                "smoothness_weight": stage.smoothness_weight,
            },
            "environment": {
                "max_off_track_streak": stage.max_off_track_streak,
                "max_steps_per_episode": stage.max_steps_per_episode,
                "off_track_penalty_ramp": stage.off_track_penalty_ramp,
            },
            "training": {
                "entropy_coef": stage.entropy_coef,
            },
        }

    def filter_samples(
        self, samples: List[Dict], get_difficulty: Callable
    ) -> List[Dict]:
        """Filter samples based on current difficulty level.

        Args:
            samples: List of data samples.
            get_difficulty: Function that returns a float difficulty
                for a single sample dict.

        Returns:
            Filtered list of samples appropriate for the current
            difficulty.
        """
        difficulty = self.get_difficulty()
        filtered = [s for s in samples if get_difficulty(s) <= difficulty]

        # Always return at least some samples
        if len(filtered) < 10:
            filtered = samples[:10]

        return filtered

    def compute_sample_difficulty(
        self, centerline: np.ndarray, vessel_mask: np.ndarray
    ) -> float:
        """Compute difficulty score for a sample.

        Factors:
          1. Average vessel width  (thinner → harder)
          2. Junction density      (more → harder)
          3. Vessel pixel density  (sparser → harder)

        Args:
            centerline: Binary centerline image.
            vessel_mask: Binary vessel mask.

        Returns:
            Difficulty score in ``[0, 1]``.
        """

        extractor = CenterlineExtractor()

        # --- average vessel width ---
        centerline_pixels = centerline.sum()
        vessel_pixels = vessel_mask.sum()

        if centerline_pixels > 0:
            avg_width = vessel_pixels / centerline_pixels
        else:
            avg_width = 1.0

        width_difficulty = 1.0 - min(avg_width / 10.0, 1.0)

        # --- junction density ---
        junctions = extractor._find_junctions(centerline)
        junction_density = len(junctions) / max(centerline_pixels, 1) * 1000
        junction_difficulty = min(junction_density / 10.0, 1.0)

        # --- vessel pixel density ---
        total_pixels = centerline.shape[0] * centerline.shape[1]
        vessel_density = vessel_pixels / total_pixels
        density_difficulty = 1.0 - min(vessel_density * 20, 1.0)

        weights = self._cfg.get("difficulty_weights", [0.4, 0.3, 0.3])
        difficulty = (
            weights[0] * width_difficulty
            + weights[1] * junction_difficulty
            + weights[2] * density_difficulty
        )

        return difficulty

    @property
    def recent_success_rate(self) -> float:
        """Rolling success rate over the current window. 0.0 if window is empty."""
        if not self._recent_successes:
            return 0.0
        return sum(self._recent_successes) / len(self._recent_successes)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _check_stage_advancement(self):
        """Advance to the next stage if success-rate threshold is met.

        ``min_episodes`` acts as a minimum observation count before judging.
        It is capped to ``_window_size`` so it can never exceed the deque
        capacity (which would make advancement permanently impossible).
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return  # already at the last stage

        current_stage = self.stages[self.current_stage_idx]

        # Guard: min_episodes cannot be larger than the window itself
        min_obs = min(current_stage.min_episodes, self._window_size)

        if len(self._recent_successes) < min_obs:
            return  # not enough observations yet

        success_rate = sum(self._recent_successes) / len(self._recent_successes)

        if success_rate >= current_stage.min_success_rate:
            self.current_stage_idx += 1
            self._recent_successes.clear()
            logging.info(
                "Advancing to curriculum stage: %s",
                self.stages[self.current_stage_idx].name,
            )
