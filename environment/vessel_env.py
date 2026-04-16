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
    """
    Configuration parameters for the Vessel Tracing Environment.
    
    Attributes:
        observation_size: The pixel width/height of the local patch provided to the agent.
        step_size: Number of pixels the agent moves per action.
        tolerance: Radius (in pixels) for considering a point 'on-track' or 'covered'.
        max_off_track_streak: Maximum consecutive steps allowed outside the tolerance radius.
        max_steps_per_episode: Hard limit on steps to prevent infinite loops.
    """
    observation_size: int = 65
    step_size: int = 1
    tolerance: float = 2.0
    max_off_track_streak: int = 5
    max_steps_per_episode: int = 2000

class VesselTracingEnv(gym.Env):
    """
    An RL Environment for tracing centerlines in medical images.
    
    The agent learns to navigate along vessel structures by receiving local image patches
    and rewards based on centerline coverage and topological accuracy.
    """

    # 8-neighbor movement directions
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
        """
        Initializes the environment and precomputes static masks for performance.
        """
        super().__init__()
        self.config = config
        env_config = config.get("environment", {})

        self.obs_size = env_config.get("observation_size", 65)
        self.step_size = env_config.get("step_size", 1)
        self.tolerance = env_config.get("tolerance", 2.0)
        self.max_off_track = env_config.get("max_off_track_streak", 3)
        self.max_steps = env_config.get("max_steps_per_episode", 2000)

        # --- PRECOMPUTE CIRCULAR COVERAGE MASK ONCE ---
        # This prevents redundant math during the step() loop.
        tol_i = int(self.tolerance)
        self.pad = tol_i + 1
        yy, xx = np.ogrid[-self.pad : self.pad + 1, -self.pad : self.pad + 1]
        self.circle_mask = (yy**2 + xx**2 <= self.tolerance**2)

        self.image = image
        self.centerline = centerline
        self.distance_transform = distance_transform
        self.fov_mask = fov_mask
        self.vesselness = None

        self.vessel_orientation = None
        self.dt_gradient = None

        if image is not None:
            self.height, self.width = image.shape[:2]
        else:
            self.height, self.width = 512, 512

        # Action 0-7: Directions, Action 8: Stop
        self.action_space = spaces.Discrete(9)
        self._setup_observation_space()

        self.reward_calculator = RewardCalculator(config)
        self.observation_builder = ObservationBuilder(config)

        # Episode state variables
        self.position = None
        self.visited_mask = None
        self.trajectory = None
        self.step_count = 0
        self.off_track_streak = 0
        self.prev_direction = None
        self.covered_centerline = None
        self.covered_centerline_pixels = 0

    def _setup_observation_space(self):
        """Defines the shape and range of the multi-channel state patch."""
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
        vesselness: Optional[np.ndarray] = None,
        vessel_orientation: Optional[np.ndarray] = None,
        dt_gradient: Optional[np.ndarray] = None,
    ):
        """Injects a new image and precalculates features for observations."""
        self.image = image
        self.centerline = centerline
        self.distance_transform = distance_transform
        self.fov_mask = fov_mask if fov_mask is not None else np.ones_like(centerline)
        self.vesselness = vesselness if vesselness is not None else (distance_transform <= self.tolerance).astype(np.float32)

        self.height, self.width = image.shape[:2]

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
        """Resets the episode state and samples a starting point."""
        super().reset(seed=seed)
        if self.image is None:
            raise ValueError("No image data set. Call set_data() first.")

        self.visited_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.trajectory = []
        self.step_count = 0
        self.off_track_streak = 0
        self.prev_direction = None
        
        self.covered_centerline = np.zeros_like(self.centerline, dtype=np.float32)
        self.covered_centerline_pixels = 0

        if start_position is not None:
            self.position = np.array(start_position, dtype=np.int32)
        else:
            self.position = self._sample_start_position()

        self.visited_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory.append(tuple(self.position))
        self._update_coverage()

        return self._get_observation(), self._get_info()

    def _sample_start_position(self) -> np.ndarray:
        """Finds a valid starting point, prioritizing vessel endpoints."""
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
        """Executes a movement, updates coverage, and calculates rewards."""
        self.step_count += 1
        reward = self.config["reward"].get("step_cost", -0.01)
        terminated = False
        truncated = False

        if action == 8:  # STOP
            terminated = True
        else:
            direction = self.DIRECTIONS[action] * self.step_size
            new_position = self.position + direction

            if not self._is_valid_position(new_position):
                reward += self.reward_calculator.compute_out_of_bounds_penalty()
                terminated = True
            else:
                old_position = self.position.copy()
                self.position = new_position
                
                y, x = self.position
                is_revisit = self.visited_mask[y, x] > 0
                
                if is_revisit:
                    terminated = True
                    reward += self.config["reward"].get("lambda_revisit", -5.0)
                else:
                    self.visited_mask[y, x] = 1.0
                    self.trajectory.append(tuple(self.position))

                    distance = self.distance_transform[y, x]
                    is_on_track = distance <= self.tolerance

                    if is_on_track:
                        self.off_track_streak = 0
                    else:
                        self.off_track_streak += 1

                    # Update coverage and calculate reward for discovery
                    prev_coverage = float(self.covered_centerline_pixels)
                    self._update_coverage()
                    new_coverage = float(self.covered_centerline_pixels - prev_coverage)

                    if distance <= 4.0 and new_coverage > 0:
                        alpha = self.config["reward"].get("alpha_near", 0.5)
                        bullseye_multiplier = math.exp(-(distance ** 2) / 8.0)
                        reward += alpha * bullseye_multiplier

                    reward += self.reward_calculator.compute_step_reward(
                        distance=distance,
                        is_revisit=False,
                        is_on_track=is_on_track,
                        new_coverage=new_coverage,
                        prev_distance=self.distance_transform[old_position[0], old_position[1]],
                        action=action,
                        prev_action=self.prev_direction,
                    )

                    self.prev_direction = action
                    if self.off_track_streak >= self.max_off_track:
                        terminated = True
                        reward += self.reward_calculator.compute_off_track_termination_penalty()

        if self.step_count >= self.max_steps:
            truncated = True

        done = terminated or truncated

        if done:
            # F1 and clDice Jackpots
            f1_weight = self.config["reward"].get("terminal_f1_weight", 0.0)
            cldice_weight = self.config["reward"].get("terminal_cldice_weight", 0.0)
            agent_path_pixels = np.sum(self.visited_mask)
            gt_centerline_pixels = max(np.sum(self.centerline), 1)

            if agent_path_pixels > 0:
                precision = np.sum(self.visited_mask * self.vesselness) / agent_path_pixels
                recall = self.covered_centerline_pixels / gt_centerline_pixels

                if f1_weight > 0 and (precision + recall) > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    reward += f1_score * f1_weight

                if cldice_weight > 0 and (precision + recall) > 0:
                    cldice_score = 2 * (precision * recall) / (precision + recall)
                    reward += cldice_score * cldice_weight

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _is_valid_position(self, position: np.ndarray) -> bool:
        """Checks if the position is within image bounds and padding."""
        y, x = position
        half = self.obs_size // 2
        if y < half or y >= self.height - half:
            return False
        if x < half or x >= self.width - half:
            return False
        return True

    def _update_coverage(self):
        """Updates the coverage mask efficiently using the precomputed circle."""
        y, x = self.position
        
        # Slicing coordinates for image
        y_min, y_max = max(0, y - self.pad), min(self.height, y + self.pad + 1)
        x_min, x_max = max(0, x - self.pad), min(self.width, x + self.pad + 1)

        # Slicing coordinates for the static circle mask (handles edges)
        my_min, my_max = max(0, self.pad - y), self.circle_mask.shape[0] - max(0, y + self.pad + 1 - self.height)
        mx_min, mx_max = max(0, self.pad - x), self.circle_mask.shape[1] - max(0, x + self.pad + 1 - self.width)

        local_cl = self.centerline[y_min:y_max, x_min:x_max]
        local_covered = self.covered_centerline[y_min:y_max, x_min:x_max]
        local_circle = self.circle_mask[my_min:my_max, mx_min:mx_max]
        
        mask = local_circle & (local_cl > 0)
        new_coverage_mask = mask & (local_covered == 0)
        self.covered_centerline_pixels += np.sum(new_coverage_mask)
        local_covered[mask] = 1.0

    def _get_observation(self) -> np.ndarray:
        """Constructs the multi-channel state patch."""
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
        """Returns episode metrics."""
        total = self.centerline.sum()
        return {
            "position": tuple(self.position),
            "step_count": self.step_count,
            "trajectory_length": len(self.trajectory),
            "off_track_streak": self.off_track_streak,
            "coverage_ratio": self.covered_centerline_pixels / max(total, 1),
            "covered_pixels": int(self.covered_centerline_pixels),
            "total_centerline_pixels": int(total),
        }

    def render(self) -> np.ndarray:
        """Renders an RGB visualization of the environment state."""
        vis = (self.image.copy() * 255).astype(np.uint8)
        vis[self.centerline > 0] = [0, 0, 255] # Blue GT
        vis[self.covered_centerline > 0] = [0, 255, 0] # Green Covered
        for y, x in self.trajectory:
            vis[max(0, y-1):min(self.height, y+2), max(0, x-1):min(self.width, x+2)] = [255, 0, 0] # Red path
        return vis

class VectorizedVesselEnv:
    """Manages multiple environments in parallel."""

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
            
            def to_np(item):
                if item is None: return None
                return item.numpy() if hasattr(item, "numpy") else item
                
            env.set_data(
                image=to_np(sample["image"]).transpose(1, 2, 0) if hasattr(sample["image"], "permute") else sample["image"],
                centerline=np.squeeze(to_np(sample["centerline"])),
                distance_transform=np.squeeze(to_np(sample["distance_transform"])),
                fov_mask=np.squeeze(to_np(sample["fov_mask"])),
                vesselness=np.squeeze(to_np(sample.get("vessel_mask", None))),
                vessel_orientation=to_np(sample.get("vessel_orientation", None)),
                dt_gradient=to_np(sample.get("dt_gradient", None)),
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
                
                def to_np(item):
                    if item is None: return None
                    return item.numpy() if hasattr(item, "numpy") else item
                    
                env.set_data(
                    image=to_np(sample["image"]).transpose(1, 2, 0) if hasattr(sample["image"], "permute") else sample["image"],
                    centerline=np.squeeze(to_np(sample["centerline"])),
                    distance_transform=np.squeeze(to_np(sample["distance_transform"])),
                    fov_mask=np.squeeze(to_np(sample["fov_mask"])),
                    vesselness=np.squeeze(to_np(sample.get("vessel_mask", None))),
                    vessel_orientation=to_np(sample.get("vessel_orientation", None)),
                    dt_gradient=to_np(sample.get("dt_gradient", None)),
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
