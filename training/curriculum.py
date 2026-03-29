# training/curriculum.py
"""Curriculum learning for progressive training difficulty."""

import logging
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
    description: str
    smoothness_weight: float = 0.4
    max_off_track_streak: int = 3
    max_steps_per_episode: int = 600
    entropy_coef: float = 0.05


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
        self.stage_episodes = 0
        self.stage_successes = 0

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
        self.stage_episodes += 1

        if success:
            self.stage_successes += 1

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
        coverage = info.get("coverage_ratio", 0.0)
        ep_len = info.get("step_count", 0)
        stage = self.get_current_stage()

        base = self._cfg.get("success_min_steps_base", 20)
        scale = self._cfg.get("success_min_steps_scale", 30)
        min_length = base + int(stage.difficulty * scale)
        min_coverage = 0.005 * (1 + stage.difficulty)

        return ep_len >= min_length and coverage >= min_coverage

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

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _check_stage_advancement(self):
        """Advance to the next stage if success-rate threshold is met."""
        if self.current_stage_idx >= len(self.stages) - 1:
            return  # already at the last stage

        current_stage = self.stages[self.current_stage_idx]

        if self.stage_episodes < current_stage.min_episodes:
            return  # not enough episodes yet

        success_rate = self.stage_successes / self.stage_episodes

        if success_rate >= current_stage.min_success_rate:
            self.current_stage_idx += 1
            self.stage_episodes = 0
            self.stage_successes = 0
            logging.info(
                "Advancing to curriculum stage: %s",
                self.stages[self.current_stage_idx].name,
            )
