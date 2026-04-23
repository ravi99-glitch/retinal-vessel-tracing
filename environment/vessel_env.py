"""RL Environment for vessel tracing."""

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .observation import ObservationBuilder
from .reward import RewardCalculator, RewardState


@dataclass
class EnvConfig:
    observation_size: int = 65
    step_size: int = 1
    tolerance: float = 2.0
    max_off_track_streak: int = 5
    max_steps_per_episode: int = 2000
    use_vesselness: bool = False


class VesselTracingEnv(gym.Env):

    N_ACTIONS = 9  # 8 directional moves + STOP (index 8)
    STOP_ACTION = 8

    DIRECTIONS = np.array(
        [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    )

    def __init__(
        self,
        config,
        image=None,
        centerline=None,
        distance_transform=None,
        vesselness=None,
        fov_mask=None,
    ):
        super().__init__()

        self.config = config
        env_config = config.get("environment", {})

        self.obs_size = env_config.get("observation_size", 65)
        self.step_size = env_config.get("step_size", 1)
        self.tolerance = env_config.get("tolerance", 2.0)
        self.max_off_track = env_config.get("max_off_track_streak", 3)
        self.max_steps = env_config.get("max_steps_per_episode", 2000)
        # Soft off-track tolerance — when True the per-step off-track
        # penalty ramps linearly with the streak instead of a flat penalty.
        # Toggled dynamically per curriculum stage via apply_overrides.
        self.off_track_ramp = env_config.get("off_track_penalty_ramp", False)

        # Precompute circular coverage template (reused every step)
        tol_i = int(self.tolerance)
        r = np.arange(-tol_i - 1, tol_i + 2)
        self._cov_template = (r[:, None] ** 2 + r[None, :] ** 2) <= self.tolerance ** 2
        self._cov_half = tol_i + 1  # half-size of template

        # Momentum blending
        self.momentum = env_config.get("momentum", 0.0)
        # 0.0 = no momentum (pure discrete), 0.3 = mild smoothing

        self.image = image
        self.centerline = centerline
        self.distance_transform = distance_transform
        self.vesselness = vesselness
        self.unet_prior = None  # (H, W) float32 in [0, 1], set in set_data()
        self.fov_mask = fov_mask

        self.vessel_orientation = None  # precomputed (H,W,2), set in set_data()

        if image is not None:
            self.height, self.width = image.shape[:2]
        else:
            self.height, self.width = 512, 512

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self._setup_observation_space()

        self.reward_calculator = RewardCalculator(config)
        self.observation_builder = ObservationBuilder(config)

        # Episode state
        self.position = None
        self.visited_mask = None
        self.trajectory = None
        self.trajectory_mask = None   # all visited pixels (on + off vessel)
        self.step_count = 0
        self.off_track_streak = 0
        self.on_track_streak = 0
        self.prev_direction = None
        self.covered_centerline = None
        self.prior_coverage = None    # accumulated mask from earlier traces
        self._momentum_vec: Optional[np.ndarray] = None  # running direction

    def _setup_observation_space(self):
        from models.policy_network import _compute_in_channels
        n_channels = _compute_in_channels(self.config)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_channels, self.obs_size, self.obs_size),
            dtype=np.float32,
        )

    def set_data(
        self,
        image,
        centerline,
        distance_transform,
        vesselness=None,
        fov_mask=None,
        vessel_orientation=None,
        dt_gradient=None,
        unet_prior=None,
        prior_coverage=None,
    ):
        self.image = image
        self.centerline = centerline
        self.distance_transform = distance_transform
        # Lazily compute Frangi vesselness if it's enabled in config but the
        # caller didn't provide it. The dataloader doesn't supply this field,
        # so this keeps the env self-sufficient. One frangi() per sample load.
        env_cfg = self.config.get("environment", {})
        if vesselness is None and env_cfg.get("use_vesselness", False):
            from skimage.filters import frangi
            gray = image[:, :, 1] if image.ndim == 3 else image
            vesselness = frangi(
                gray.astype(np.float64),
                sigmas=np.linspace(1.0, 3.0, 5),
                black_ridges=True,
            ).astype(np.float32)
        self.vesselness = vesselness
        # Lazily compute UNet prior if enabled but not supplied. Called once
        # per sample load — the predictor caches the model itself.
        if unet_prior is None and env_cfg.get("use_unet_prior", False):
            from data.dataloader import compute_unet_prior
            unet_prior = compute_unet_prior(image)
        self.unet_prior = unet_prior
        self.prior_coverage = prior_coverage  # accumulated mask from earlier traces
        self.fov_mask = fov_mask if fov_mask is not None else np.ones_like(centerline)
        self.height, self.width = image.shape[:2]

        # Use precomputed if provided, else fall back to computing
        self.vessel_orientation = (
            vessel_orientation
            if vessel_orientation is not None
            else self.observation_builder.compute_vessel_orientation(image)
        )
        self.dt_gradient = (
            dt_gradient
            if dt_gradient is not None
            else self.observation_builder.compute_dt_gradient(distance_transform)
        )

        self.observation_builder.prepare_stacked_sources(
            distance_transform=distance_transform,
            dt_gradient=self.dt_gradient,
            centerline=centerline,
            vessel_orientation=self.vessel_orientation,
            unet_prior=self.unet_prior,
        )

    def reset(self, seed=None, start_position=None, **kwargs):
        super().reset(seed=seed)

        if self.image is None:
            raise ValueError("No image data set. Call set_data() first.")

        self.visited_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.trajectory_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.trajectory = []
        self.step_count = 0
        self.off_track_streak = 0
        self.on_track_streak = 0
        self.prev_direction = None
        self.covered_centerline = np.zeros_like(self.centerline, dtype=np.float32)
        self._total_visited = 0
        self._total_visited_on_track = 0
        self._momentum_vec = None

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

    def step(self, action: int):
        self.step_count += 1

        # ─── Explicit STOP action ─────────────────────────────────────────
        if action == self.STOP_ACTION:
            f_beta = self._compute_fbeta()
            pos = np.array(self.position)
            dist = float(self.distance_transform[self.position[0], self.position[1]])
            cov_ratio = self.covered_centerline.sum() / max(self.centerline.sum(), 1.0)
            state = RewardState(
                is_terminal=True,
                terminal_reason="stop",
                new_coverage=0.0,
                is_on_track=dist <= self.tolerance,
                distance=dist,
                prev_distance=dist,
                coverage=cov_ratio,
                f_beta_score=f_beta,
                position=pos,
                step_number=self.step_count,
                junction_map_value=self._junction_val_at(self.position),
            )
            reward, bd = self.reward_calculator.compute(state)
            info = self._get_info()
            info.update(bd)
            info["stopped"] = True
            info["episode_f1"] = float(f_beta)
            return self._get_observation(), reward, True, False, info

        # ─── Movement ────────────────────────────────────────────────────────
        prev_pos = np.array(self.position)
        prev_distance = float(self.distance_transform[self.position[0], self.position[1]])

        raw_direction = self.DIRECTIONS[action].astype(np.float64) * self.step_size
        if self.momentum > 0 and self._momentum_vec is not None:
            blended = (1.0 - self.momentum) * raw_direction + self.momentum * self._momentum_vec
            new_position = self.position + np.round(blended).astype(np.int32)
            if np.array_equal(new_position, self.position):
                new_position = self.position + raw_direction.astype(np.int32)
            self._momentum_vec = blended / (np.linalg.norm(blended) + 1e-8)
        else:
            new_position = self.position + raw_direction.astype(np.int32)
            norm = np.linalg.norm(raw_direction)
            self._momentum_vec = raw_direction / (norm + 1e-8) if norm > 0 else None

        # ─── Out of bounds ───────────────────────────────────────────────────
        if not self._is_valid_position(new_position):
            state = RewardState(
                is_terminal=True,
                terminal_reason="oob",
                new_coverage=0.0,
                is_on_track=False,
                distance=float(self.distance_transform[prev_pos[0], prev_pos[1]]),
                prev_distance=prev_distance,
                coverage=self.covered_centerline.sum() / max(self.centerline.sum(), 1.0),
                f_beta_score=0.0,
                position=prev_pos,
                step_number=self.step_count,
            )
            reward, bd = self.reward_calculator.compute(state)
            info = self._get_info()
            info.update(bd)
            return self._get_observation(), reward, True, False, info

        # ─── Apply move ──────────────────────────────────────────────────────
        self.position = new_position

        is_revisit = self.visited_mask[self.position[0], self.position[1]] > 0
        self.visited_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory.append(tuple(self.position))

        distance = float(self.distance_transform[self.position[0], self.position[1]])
        is_on_track = distance <= self.tolerance

        if not is_revisit:
            self._total_visited += 1
            if is_on_track:
                self._total_visited_on_track += 1

        if is_on_track:
            self.off_track_streak = 0
            self.on_track_streak += 1
        else:
            self.off_track_streak += 1
            self.on_track_streak = 0

        total_gt = max(float(self.centerline.sum()), 1.0)
        prev_coverage_sum = self.covered_centerline.sum()
        prev_coverage_ratio = prev_coverage_sum / total_gt
        self._update_coverage()
        new_coverage = self.covered_centerline.sum() - prev_coverage_sum
        current_coverage_ratio = self.covered_centerline.sum() / total_gt

        junction_val = self._junction_val_at(self.position)

        # Double off-track limit at junction pixels so the agent can try a
        # branch direction without immediate termination.
        effective_off_track = (
            self.max_off_track * 2 if junction_val >= 0.8 else self.max_off_track
        )
        terminated = self.off_track_streak >= effective_off_track
        truncated = self.step_count >= self.max_steps

        terminal_reason = ""
        f_beta = 0.0
        if terminated or truncated:
            terminal_reason = "off_track" if terminated else "max_steps"
            f_beta = self._compute_fbeta()

        state = RewardState(
            is_terminal=terminated or truncated,
            terminal_reason=terminal_reason,
            new_coverage=new_coverage,
            is_on_track=is_on_track,
            distance=distance,
            prev_distance=prev_distance,
            coverage=current_coverage_ratio,
            f_beta_score=f_beta,
            position=np.array(self.position),
            step_number=self.step_count,
            junction_map_value=junction_val,
            is_revisit=is_revisit,
        )

        reward, bd = self.reward_calculator.compute(state)
        self.prev_direction = action

        info = self._get_info()
        info.update(bd)
        if terminated or truncated:
            info["episode_f1"] = float(f_beta)

        return self._get_observation(), reward, terminated, truncated, info

    def _compute_fbeta(self) -> float:
        """Compute F-beta against ``covered_centerline`` — the SAME mask the
        eval loop uses for clDice.  This aligns the training terminal signal
        with the eval metric: a policy that maximises this F_β also maximises
        Tsens/Tprec in clDice.
        """
        from data.centerline_extraction import compute_centerline_f1
        rc = self.config.get("reward", {})
        beta_sq = float(rc.get("terminal_recall_beta_sq", 4.0))
        metrics = compute_centerline_f1(
            self.covered_centerline, self.centerline, tolerance=self.tolerance
        )
        recall = metrics["recall"]
        precision = metrics["precision"]
        denom = beta_sq * precision + recall
        return (1.0 + beta_sq) * precision * recall / denom if denom > 0 else 0.0

    def _junction_val_at(self, position) -> float:
        """Return junction-map value at the given position (0.0 if not built)."""
        if self.observation_builder.junction_map is not None:
            return float(self.observation_builder.junction_map[position[0], position[1]])
        return 0.0

    def _is_valid_position(self, position):
        y, x = position
        half = self.obs_size // 2
        if y < half or y >= self.height - half:
            return False
        if x < half or x >= self.width - half:
            return False
        if self.fov_mask[y, x] == 0:
            return False
        return True

    def _update_coverage(self):
        y, x = self.position
        h = self._cov_half

        # Image-space bounds
        y_min = max(0, y - h)
        y_max = min(self.height, y + h + 1)
        x_min = max(0, x - h)
        x_max = min(self.width, x + h + 1)

        patch = self.centerline[y_min:y_max, x_min:x_max]
        if not patch.any():
            return

        # Slice the precomputed template to match boundary clipping
        ty_min = y_min - (y - h)
        ty_max = ty_min + (y_max - y_min)
        tx_min = x_min - (x - h)
        tx_max = tx_min + (x_max - x_min)
        within = self._cov_template[ty_min:ty_max, tx_min:tx_max]

        self.covered_centerline[y_min:y_max, x_min:x_max] = np.where(
            within & (patch > 0),
            1.0,
            self.covered_centerline[y_min:y_max, x_min:x_max],
        )

    def _get_observation(self):
        return self.observation_builder.build(
            image=self.image,
            visited_mask=self.visited_mask,
            vesselness=self.vesselness,
            position=self.position,
            prev_direction=self.prev_direction,
            distance_transform=self.distance_transform,
            centerline=self.centerline,
            vessel_orientation=self.vessel_orientation,
            dt_gradient=self.dt_gradient,
            unet_prior=self.unet_prior,
            prior_coverage=self.prior_coverage,
        )

    def _get_info(self):
        total = self.centerline.sum()
        covered = self.covered_centerline.sum()
        info = {
            "position": tuple(self.position),
            "step_count": self.step_count,
            "trajectory_length": len(self.trajectory),
            "off_track_streak": self.off_track_streak,
            "coverage_ratio": covered / max(total, 1),
            "covered_pixels": int(covered),
            "total_centerline_pixels": int(total),
        }
        # Precision: fraction of unique visited positions that were on-track
        if self._total_visited > 0:
            info["precision"] = self._total_visited_on_track / self._total_visited
        else:
            info["precision"] = 0.0
        return info

    def render(self):
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

    def _apply_sample(self, env, sample):
        """Unpack a dataset sample and call env.set_data()."""
        env.set_data(
            image=sample["image"].permute(1, 2, 0).numpy(),
            centerline=sample["centerline"].squeeze().numpy(),
            distance_transform=sample["distance_transform"].squeeze().numpy(),
            fov_mask=sample["fov_mask"].squeeze().numpy(),
            vessel_orientation=(
                sample["vessel_orientation"].numpy()
                if "vessel_orientation" in sample
                else None
            ),
            dt_gradient=(
                sample["dt_gradient"].numpy() if "dt_gradient" in sample else None
            ),
            unet_prior=(
                sample["unet_prior"].squeeze(0).numpy()
                if "unet_prior" in sample
                else None
            ),
        )

    def reset(self):
        observations, infos = [], []
        for i, env in enumerate(self.envs):
            sample = self._get_random_sample()
            self.current_samples[i] = sample
            self._apply_sample(env, sample)
            # env.set_data(
            #     image=sample["image"].permute(1, 2, 0).numpy(),
            #     centerline=sample["centerline"].squeeze().numpy(),
            #     distance_transform=sample["distance_transform"].squeeze().numpy(),
            #     fov_mask=sample["fov_mask"].squeeze().numpy(),
            # )
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
                self._apply_sample(env, sample)
                # env.set_data(
                #     image=sample["image"].permute(1, 2, 0).numpy(),
                #     centerline=sample["centerline"].squeeze().numpy(),
                #     distance_transform=sample["distance_transform"].squeeze().numpy(),
                #     fov_mask=sample["fov_mask"].squeeze().numpy(),
                # )
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
