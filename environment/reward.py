"""Reward calculation for vessel tracing."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as ndimage_label


class RewardCalculator:
    """Calculates rewards for the vessel tracing agent.

    Reward components:
    - Proximity reward (staying near centerline, only on NEW positions)
    - Coverage bonus (discovering new centerline segments)
    - Off-track penalty
    - Revisit penalty
    - Step cost
    - Smoothness penalty (penalise sharp turns)            # ← NEW
    - Oscillation penalty (penalise back-and-forth zigzag)  # ← NEW
    - Potential-based shaping (optional)
    """

    # ── precomputed normalised direction vectors for actions 0-7 ──
    _RAW_DIRS = np.array(
        [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]],
        dtype=np.float64,
    )
    _DIR_NORMS = _RAW_DIRS / np.linalg.norm(_RAW_DIRS, axis=1, keepdims=True)

    def __init__(self, config: Dict[str, Any]):
        reward_config = config.get("reward", {})

        self.alpha = reward_config.get("alpha_near", 1.0)
        self.beta = reward_config.get("beta_coverage", 2.0)
        self.gamma_off = reward_config.get("gamma_off", -1.0)
        self.lambda_revisit = reward_config.get("lambda_revisit", -0.5)
        self.step_cost = reward_config.get("step_cost", -0.01)
        self.terminal_f1_weight = reward_config.get("terminal_f1_weight", 10.0)
        self.use_potential_shaping = reward_config.get("use_potential_shaping", True)

        # smoothness
        self.smoothness_weight = reward_config.get("smoothness_weight", 0.4)
        self.oscillation_weight = reward_config.get("oscillation_weight", 0.6)
        self.oscillation_window = reward_config.get("oscillation_window", 6)

        # reduce bridging
        self.off_vessel_distance_weight = reward_config.get(
            "off_vessel_distance_weight", 0.3
        )
        self.bridge_penalty = reward_config.get("bridge_penalty", -3.0)

        # Betti-0 / topology
        self.betti0_episode_weight = reward_config.get("betti0_episode_weight", 2.0)
        self.local_merge_reward = reward_config.get("local_merge_reward", 1.5)
        self.local_merge_radius = reward_config.get("local_merge_radius", 5)
        self.betti0_check_interval = reward_config.get("betti0_check_interval", 50)
        self.betti0_delta_weight = reward_config.get("betti0_delta_weight", 0.5)

        self.tolerance = config.get("environment", {}).get("tolerance", 2.0)
        self.gamma = config.get("training", {}).get("ppo", {}).get("gamma", 0.99)

        self.out_of_bounds_penalty = -10.0
        self.off_track_termination_penalty = -5.0

    def _smoothness_penalty(self, action: int, prev_action: Optional[int]) -> float:
        """Penalise sharp single-step turns.

        Uses cosine similarity between consecutive direction vectors:
            cos = 1  → straight ahead    → penalty = 0
            cos = 0  → 90° turn          → penalty = −weight × 0.5
            cos = −1 → 180° reversal     → penalty = −weight × 1.0
        """
        if prev_action is None or action >= 8 or prev_action >= 8:
            return 0.0

        cos_angle = float(np.dot(self._DIR_NORMS[prev_action], self._DIR_NORMS[action]))
        return -self.smoothness_weight * (1.0 - cos_angle) / 2.0

    def _oscillation_penalty(self, action_history: List[int]) -> float:
        """Detect back-and-forth zigzag over a short window.

        Opposite action pairs on the 8-direction ring are exactly 4 apart
        (N↔S, NE↔SW, …).  If the agent alternates between roughly
        opposite directions, it's oscillating.

        We count the number of "reversals" (direction change ≥ 135°)
        in the last `oscillation_window` actions and penalise
        proportionally.
        """
        window = self.oscillation_window
        if len(action_history) < 3:
            return 0.0

        recent = action_history[-window:]
        # only consider movement actions (0-7)
        recent = [a for a in recent if a < 8]
        if len(recent) < 3:
            return 0.0

        reversals = 0
        for i in range(1, len(recent)):
            cos_angle = float(
                np.dot(self._DIR_NORMS[recent[i - 1]], self._DIR_NORMS[recent[i]])
            )
            if cos_angle < -0.3:  # ≥ ~107° turn
                reversals += 1

        # fraction of steps that are reversals
        reversal_ratio = reversals / (len(recent) - 1)
        return -self.oscillation_weight * reversal_ratio

    def _graduated_off_track_penalty(self, distance: float) -> float:
        """Scale the off-track penalty by HOW FAR off-track the agent is.

        Replaces the flat gamma_off for off-track steps.
        Close to centerline (distance ~ tolerance) → mild penalty.
        Far from centerline → harsh penalty.
        """
        if distance <= self.tolerance:
            return 0.0
        excess = distance - self.tolerance
        return self.gamma_off - self.off_vessel_distance_weight * excess

    def _bridge_penalty(self, off_track_streak: int, is_on_track: bool) -> float:
        """Penalise when agent returns on-track AFTER being off-track.

        This is exactly the bridging pattern: off-vessel → back on-vessel
        at a disconnected segment.
        """
        if is_on_track and off_track_streak > 0:
            # Agent just came back on track after wandering off
            return self.bridge_penalty * off_track_streak
        return 0.0

    def compute_local_merge_reward(
        self,
        position: Tuple[int, int],
        traced_mask: np.ndarray,
    ) -> float:
        y, x = int(position[0]), int(position[1])
        r = self.local_merge_radius
        h, w = traced_mask.shape

        y_min, y_max = max(0, y - r), min(h, y + r + 1)
        x_min, x_max = max(0, x - r), min(w, x + r + 1)

        patch = traced_mask[y_min:y_max, x_min:x_max]
        local_y = y - y_min
        local_x = x - x_min

        # Skip both labels entirely if current pixel was already visited,
        # or if there are fewer than 2 filled pixels in the neighbourhood —
        # merging two components requires at least 2 others to exist.
        if patch[local_y, local_x] == 0:
            return 0.0
        if (patch > 0).sum() < 3:  # current pixel + at least 2 others
            return 0.0

        # Label the patch WITHOUT the current pixel
        patch_without = patch.copy()
        patch_without[local_y, local_x] = 0
        _, n_before = ndimage_label(patch_without > 0)

        if n_before < 2:
            return 0.0  # nothing to merge — skip second label entirely

        # Only now pay for the second label
        patch_with = patch_without.copy()
        patch_with[local_y, local_x] = 1
        _, n_after = ndimage_label(patch_with > 0)

        merged = n_before - n_after
        return self.local_merge_reward * merged if merged > 0 else 0.0

    def compute_betti0_episode_reward(
        self,
        traced_mask: np.ndarray,
        gt_centerline: np.ndarray,
    ) -> float:
        """Episode-end reward penalising Betti-0 deviation from GT.

        Negative reward proportional to |B0_pred - B0_gt|.
        """
        from evaluation.metrics import compute_betti0

        b0_pred = compute_betti0(traced_mask)
        b0_gt = compute_betti0(gt_centerline)

        return -self.betti0_episode_weight * abs(b0_pred - b0_gt)

    def compute_betti0_delta_reward(
        self,
        prev_betti0: int,
        current_betti0: int,
    ) -> float:
        """Periodic reward for REDUCING the number of connected components.

        Positive when components merge, negative when they fragment.
        """
        delta = prev_betti0 - current_betti0  # positive = merged
        return self.betti0_delta_weight * delta

    def compute_step_reward(
        self,
        distance: float,
        is_revisit: bool,
        is_on_track: bool,
        new_coverage: float,
        prev_distance: float,
        action: int,
        prev_action: Optional[int],
        action_history: Optional[List[int]] = None,
        off_track_streak: int = 0,
    ) -> float:
        reward = 0.0

        # 1. Proximity reward — only on unvisited positions
        if not is_revisit:
            reward += self.alpha * max(0.0, 1.0 - distance / self.tolerance)

        # 2. Coverage bonus
        if new_coverage > 0:
            reward += self.beta * new_coverage

        # 3. Off-track penalty
        if not is_on_track:
            reward += self._graduated_off_track_penalty(distance)

        # 3b. Bridge penalty (returning on-track after wandering)
        reward += self._bridge_penalty(off_track_streak, is_on_track)

        # 4. Revisit penalty
        if is_revisit:
            reward += self.lambda_revisit

        # 5. Step cost
        reward += self.step_cost

        # 6. Potential-based shaping
        if self.use_potential_shaping:
            reward += self.gamma * (-distance) - (-prev_distance)

        # 7. Smoothness penalty (pairwise)
        reward += self._smoothness_penalty(action, prev_action)

        # 8. Oscillation penalty (window-based)
        if action_history is not None:
            reward += self._oscillation_penalty(action_history)

        return reward

    def compute_terminal_reward(
        self, covered_centerline: np.ndarray, gt_centerline: np.ndarray
    ) -> float:
        covered = covered_centerline > 0
        gt = gt_centerline > 0

        tp = np.logical_and(covered, gt).sum()
        pp = covered.sum()
        ap = gt.sum()

        precision = tp / max(pp, 1)
        recall = tp / max(ap, 1)

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return self.terminal_f1_weight * f1

    def compute_out_of_bounds_penalty(self) -> float:
        return self.out_of_bounds_penalty

    def compute_off_track_termination_penalty(self) -> float:
        return self.off_track_termination_penalty
