# environment/vessel_env.py
"""RL Environment for vessel tracing."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .observation import ObservationBuilder
from .reward import RewardCalculator

@dataclass
class EnvConfig:
    """Environment configuration."""
    observation_size: int = 65
    step_size: int = 1
    tolerance: float = 2.0
    max_off_track_streak: int = 5
    max_steps_per_episode: int = 2000

class VesselTracingEnv(gym.Env):
    """RL Environment for tracing vessel centerlines."""

    DIRECTIONS = np.array([
        [-1, 0], [-1, 1], [0, 1], [1, 1], 
        [1, 0], [1, -1], [0, -1], [-1, -1]
    ])

    def __init__(
        self,
        config: Dict[str, Any],
        image: Optional[np.ndarray] = None,
        centerline: Optional[np.ndarray] = None,
        distance_transform: Optional[np.ndarray] = None,
        fov_mask: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.config = config
        env_config = config.get("environment", {})

        self.obs_size = env_config.get("observation_size", 65)
        self.step_size = env_config.get("step_size", 1)
        self.tolerance = env_config.get("tolerance", 2.0)
        self.max_off_track = env_config.get("max_off_track_streak", 3)
        self.max_steps = env_config.get("max_steps_per_episode", 2000)

        self.image = image
        self.centerline = centerline
        self.distance_transform = distance_transform
        self.fov_mask = fov_mask

        self.vessel_orientation = None
        self.dt_gradient = None

        if image is not None:
            self.height, self.width = image.shape[:2]
        else:
            self.height, self.width = 512, 512

        self.action_space = spaces.Discrete(9)
        self._setup_observation_space()

        self.reward_calculator = RewardCalculator(config)
        self.observation_builder = ObservationBuilder(config)

        # Episode state
        self.position = None
        self.visited_mask = None
        self.trajectory = None
        self.step_count = 0
        self.off_track_streak = 0
        self.prev_direction = None
        self.covered_centerline = None

    def _setup_observation_space(self):
        n_channels = 9
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(n_channels, self.obs_size, self.obs_size),
            dtype=np.float32,
        )

    def set_data(
        self,
        image: np.ndarray,
        centerline: np.ndarray,
        distance_transform: np.ndarray,
        fov_mask: Optional[np.ndarray] = None,
        vessel_orientation: Optional[np.ndarray] = None,
        dt_gradient: Optional[np.ndarray] = None,
    ):
        self.image = image
        self.centerline = centerline
        self.distance_transform = distance_transform
        self.fov_mask = fov_mask if fov_mask is not None else np.ones_like(centerline)
        self.height, self.width = image.shape[:2]

        # --- PRECOMPUTE THE COMPASS TANGENTS ---
        self.vessel_orientation = (
            vessel_orientation if vessel_orientation is not None
            else self.observation_builder.compute_vessel_orientation(image)
        )
        self.dt_gradient = (
            dt_gradient if dt_gradient is not None
            else self.observation_builder.compute_dt_gradient(distance_transform)
        )

        self.observation_builder.prepare_stacked_sources(
            distance_transform=distance_transform,
            dt_gradient=self.dt_gradient,
            vessel_orientation=self.vessel_orientation,
        )

    def reset(
        self, seed: Optional[int] = None, start_position: Optional[Tuple[int, int]] = None, **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if self.image is None:
            raise ValueError("No image data set. Call set_data() first.")

        self.visited_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.trajectory = []
        self.step_count = 0
        self.off_track_streak = 0
        self.prev_direction = None
        self.covered_centerline = np.zeros_like(self.centerline, dtype=np.float32)

        if start_position is not None:
            self.position = np.array(start_position, dtype=np.int32)
        else:
            self.position = self._sample_start_position()

        self.visited_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory.append(tuple(self.position))
        self._update_coverage()

        return self._get_observation(), self._get_info()

    def _sample_start_position(self) -> np.ndarray:
        centerline_points = np.argwhere(self.centerline > 0)
        if len(centerline_points) == 0:
            fov_points = np.argwhere(self.fov_mask > 0)
            if len(fov_points) == 0:
                return np.array([self.height // 2, self.width // 2])
            idx = self.np_random.integers(len(fov_points))
            return fov_points[idx]

        from data.centerline_extraction import CenterlineExtractor
        extractor = CenterlineExtractor()
        endpoints = extractor._find_endpoints(self.centerline)

        if endpoints:
            idx = self.np_random.integers(len(endpoints))
            return np.array(endpoints[idx])

        idx = self.np_random.integers(len(centerline_points))
        return centerline_points[idx]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1

        if action == 8:
            reward = self.reward_calculator.compute_terminal_reward(
                self.covered_centerline, self.centerline
            )
            return self._get_observation(), reward, True, False, self._get_info()

        direction = self.DIRECTIONS[action] * self.step_size
        new_position = self.position + direction

        if not self._is_valid_position(new_position):
            reward = self.reward_calculator.compute_out_of_bounds_penalty()
            return self._get_observation(), reward, True, False, self._get_info()

        old_position = self.position.copy()
        self.position = new_position

        y, x = self.position
        is_revisit = self.visited_mask[y, x] > 0
        
        # Draw a 3x3 square on the visited mask to repel parallel tracking
        y_min, y_max = max(0, y - 1), min(self.height, y + 2)
        x_min, x_max = max(0, x - 1), min(self.width, x + 2)
        self.visited_mask[y_min:y_max, x_min:x_max] = 1.0
        
        self.trajectory.append(tuple(self.position))

        distance = self.distance_transform[self.position[0], self.position[1]]
        is_on_track = distance <= self.tolerance

        if is_on_track:
            self.off_track_streak = 0
        else:
            self.off_track_streak += 1

        prev_coverage = self.covered_centerline.sum()
        self._update_coverage()
        new_coverage = float(self.covered_centerline.sum() - prev_coverage)

        # =================================================================
        # CONTINUOUS BULLSEYE REWARD
        # Reward exploration based on how perfectly centered the agent is
        # =================================================================
        if distance <= 4.0 and new_coverage > 0:
            # Gaussian drop-off formula: e^(-(x^2) / sigma^2)
            bullseye_multiplier = math.exp(-(distance ** 2) / 2.0)
            new_coverage = new_coverage * bullseye_multiplier
        # =================================================================

        reward = self.reward_calculator.compute_step_reward(
            distance=distance,
            is_revisit=is_revisit,
            is_on_track=is_on_track,
            new_coverage=new_coverage,
            prev_distance=self.distance_transform[old_position[0], old_position[1]],
            action=action,
            prev_action=self.prev_direction,
        )

        self.prev_direction = action
        terminated = self.off_track_streak >= self.max_off_track
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward += self.reward_calculator.compute_off_track_termination_penalty()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _is_valid_position(self, position: np.ndarray) -> bool:
        y, x = position
        half = self.obs_size // 2
        if y < half or y >= self.height - half:
            return False
        if x < half or x >= self.width - half:
            return False
        return True

    def _update_coverage(self):
        y, x = self.position
        tol_i = int(self.tolerance)
        
        # Bounding box coordinates
        y_min = max(0, y - tol_i - 1)
        y_max = min(self.height, y + tol_i + 2)
        x_min = max(0, x - tol_i - 1)
        x_max = min(self.width, x + tol_i + 2)

        # Slice the local regions
        local_cl = self.centerline[y_min:y_max, x_min:x_max]
        local_covered = self.covered_centerline[y_min:y_max, x_min:x_max]

        # Create a fast coordinate grid
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        
        # Calculate squared distances (avoids the slow np.sqrt)
        distances_sq = (yy - y) ** 2 + (xx - x) ** 2
        
        # Vectorized boolean mask: within tolerance AND is a centerline pixel
        mask = (distances_sq <= self.tolerance ** 2) & (local_cl > 0)
        
        # Apply the mask instantly
        local_covered[mask] = 1.0

    def _get_observation(self) -> np.ndarray:
        return self.observation_builder.build(
            image=self.image,
            visited_mask=self.visited_mask,
            position=self.position,
            prev_direction=self.prev_direction,
            distance_transform=self.distance_transform,
            vessel_orientation=self.vessel_orientation,
            dt_gradient=self.dt_gradient,               
        )

    def _get_info(self) -> Dict[str, Any]:
        total = self.centerline.sum()
        covered = self.covered_centerline.sum()
        return {
            "position": tuple(self.position),
            "step_count": self.step_count,
            "trajectory_length": len(self.trajectory),
            "off_track_streak": self.off_track_streak,
            "coverage_ratio": covered / max(total, 1),
            "covered_pixels": int(covered),
            "total_centerline_pixels": int(total),
        }

    def render(self) -> np.ndarray:
        vis = (self.image.copy() * 255).astype(np.uint8)
        vis[self.centerline > 0] = [0, 0, 255]
        vis[self.covered_centerline > 0] = [0, 255, 0]
        for y, x in self.trajectory:
            vis[
                max(0, y - 1) : min(self.height, y + 2),
                max(0, x - 1) : min(self.width, x + 2),
            ] = [255, 0, 0]
        y, x = self.position
        vis[
            max(0, y - 2) : min(self.height, y + 3),
            max(0, x - 2) : min(self.width, x + 3),
        ] = [255, 255, 0]
        return vis

class VectorizedVesselEnv:
    """Vectorized environment for parallel training."""

    def __init__(self, config, num_envs=8, dataset=None):
        self.config = config
        self.num_envs = num_envs
        self.dataset = dataset
        self.envs = [VesselTracingEnv(config) for _ in range(num_envs)]
        self.current_samples = [None] * num_envs

    def reset(self):
        observations, infos = [], []
        for i, env in enumerate(self.envs):
            sample = self._get_random_sample()
            self.current_samples[i] = sample
            env.set_data(
                image=sample["image"].permute(1, 2, 0).numpy(),
                centerline=sample["centerline"].squeeze().numpy(),
                distance_transform=sample["distance_transform"].squeeze().numpy(),
                fov_mask=sample["fov_mask"].squeeze().numpy(),
                vessel_orientation=(sample["vessel_orientation"].numpy() if "vessel_orientation" in sample else None),
                dt_gradient=(sample["dt_gradient"].numpy() if "dt_gradient" in sample else None),
            )
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), infos

    def step(self, actions):
        observations, rewards, terminateds, truncateds, infos = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                sample = self._get_random_sample()
                self.current_samples[i] = sample
                env.set_data(
                    image=sample["image"].permute(1, 2, 0).numpy(),
                    centerline=sample["centerline"].squeeze().numpy(),
                    distance_transform=sample["distance_transform"].squeeze().numpy(),
                    fov_mask=sample["fov_mask"].squeeze().numpy(),
                    vessel_orientation=(sample["vessel_orientation"].numpy() if "vessel_orientation" in sample else None),
                    dt_gradient=(sample["dt_gradient"].numpy() if "dt_gradient" in sample else None),
                )
                obs, _ = env.reset()
                info["terminal_observation"] = obs
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        return (
            np.stack(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos,
        )

    def _get_random_sample(self):
        idx = np.random.randint(len(self.dataset))
        return self.dataset[idx]
