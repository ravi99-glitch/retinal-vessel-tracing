# training/curriculum.py
"""Curriculum learning for progressive training difficulty.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np


@dataclass
class CurriculumStage:
    """A stage in the curriculum."""

    name: str
    difficulty: float
    min_success_rate: float
    min_episodes: int
    description: str


class CurriculumManager:
    """Manages curriculum learning for vessel tracing.

    Progressively increases difficulty from easy cases (large, well-defined vessels)
    to hard cases (thin capillaries, pathologies, poor contrast).
    """

    def __init__(self, config: Dict[str, Any]):
        curriculum_config = config.get("training", {}).get("curriculum", {})

        self.start_difficulty = curriculum_config.get("start_difficulty", 0.2)
        self.end_difficulty = curriculum_config.get("end_difficulty", 1.0)
        self.warmup_steps = curriculum_config.get("warmup_steps", 500_000)

        self.current_difficulty = self.start_difficulty
        self.total_steps = 0

        # Define curriculum stages
        self.stages = [
            CurriculumStage(
                name="large_vessels",
                difficulty=0.2,
                min_success_rate=0.7,
                min_episodes=100,
                description="Large, high-contrast vessels only",
            ),
            CurriculumStage(
                name="medium_vessels",
                difficulty=0.4,
                min_success_rate=0.6,
                min_episodes=200,
                description="Medium-sized vessels added",
            ),
            CurriculumStage(
                name="small_vessels",
                difficulty=0.6,
                min_success_rate=0.5,
                min_episodes=300,
                description="Small vessels and some capillaries",
            ),
            CurriculumStage(
                name="capillaries",
                difficulty=0.8,
                min_success_rate=0.4,
                min_episodes=400,
                description="All capillaries included",
            ),
            CurriculumStage(
                name="full_difficulty",
                difficulty=1.0,
                min_success_rate=0.3,
                min_episodes=500,
                description="Full difficulty with pathologies and artifacts",
            ),
        ]

        self.current_stage_idx = 0
        self.stage_episodes = 0
        self.stage_successes = 0

    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty

    def step(self, success: bool = False):
        """Update curriculum state.

        Args:
            success: Whether the episode was successful

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

        # Check for stage advancement
        self._check_stage_advancement()

    def _check_stage_advancement(self):
        """Check if we should advance to the next curriculum stage."""
        if self.current_stage_idx >= len(self.stages) - 1:
            return

        current_stage = self.stages[self.current_stage_idx]

        if self.stage_episodes < current_stage.min_episodes:
            return

        success_rate = self.stage_successes / self.stage_episodes

        if success_rate >= current_stage.min_success_rate:
            self.current_stage_idx += 1
            self.stage_episodes = 0
            self.stage_successes = 0
            print(
                f"Advancing to curriculum stage: {self.stages[self.current_stage_idx].name}"
            )

    def filter_samples(
        self, samples: List[Dict], get_difficulty: Callable
    ) -> List[Dict]:
        """Filter samples based on current difficulty level.

        Args:
            samples: List of data samples
            get_difficulty: Function to get difficulty of a sample

        Returns:
            Filtered list of samples

        """
        difficulty = self.get_difficulty()

        filtered = []
        for sample in samples:
            sample_difficulty = get_difficulty(sample)
            if sample_difficulty <= difficulty:
                filtered.append(sample)

        # Always return at least some samples
        if len(filtered) < 10:
            filtered = samples[:10]

        return filtered

    def compute_sample_difficulty(
        self, centerline: np.ndarray, vessel_mask: np.ndarray
    ) -> float:
        """Compute difficulty score for a sample.

        Args:
            centerline: Binary centerline
            vessel_mask: Binary vessel mask

        Returns:
            Difficulty score in [0, 1]

        """
        # Factors affecting difficulty:
        # 1. Average vessel width (thinner = harder)
        # 2. Number of junctions (more = harder)
        # 3. Vessel density (lower = harder)
        # 4. Contrast (lower = harder, not computed here)

        from data.centerline_extraction import CenterlineExtractor

        extractor = CenterlineExtractor()

        # Compute average vessel width
        centerline_pixels = centerline.sum()
        vessel_pixels = vessel_mask.sum()

        if centerline_pixels > 0:
            avg_width = vessel_pixels / centerline_pixels
        else:
            avg_width = 1.0

        # Normalize width (assume max width of ~10 pixels)
        width_difficulty = 1.0 - min(avg_width / 10.0, 1.0)

        # Count junctions
        junctions = extractor._find_junctions(centerline)
        junction_density = len(junctions) / max(centerline_pixels, 1) * 1000
        junction_difficulty = min(junction_density / 10.0, 1.0)

        # Vessel density
        total_pixels = centerline.shape[0] * centerline.shape[1]
        vessel_density = vessel_pixels / total_pixels
        density_difficulty = 1.0 - min(vessel_density * 20, 1.0)

        # Combined difficulty
        difficulty = (
            0.4 * width_difficulty
            + 0.3 * junction_difficulty
            + 0.3 * density_difficulty
        )

        return difficulty
