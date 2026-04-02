# environment/reward.py
"""Reward calculation for vessel tracing.
"""

from typing import Any, Dict, Optional

import numpy as np


class RewardCalculator:
    """Calculates rewards for the vessel tracing agent.

    Reward components:
    - Proximity reward (staying near centerline, only on NEW positions)
    - Coverage bonus (discovering new centerline segments)
    - Off-track penalty
    - Revisit penalty
    - Step cost
    - Smoothness penalty (NEW: penalizes sharp turns to prevent 'hairy' traces)
    - Potential-based shaping (optional)
    """

    def __init__(self, config: Dict[str, Any]):
        reward_config = config.get("reward", {})

        self.alpha = reward_config.get("alpha_near", 1.0)
        self.beta = reward_config.get("beta_coverage", 2.0)
        self.gamma_off = reward_config.get("gamma_off", -1.0)
        self.lambda_revisit = reward_config.get("lambda_revisit", -0.5)
        self.step_cost = reward_config.get("step_cost", -0.01)
        
        # --- NEW: Smoothness Penalty ---
        # A small penalty applied when the agent turns >= 90 degrees
        self.smoothness_penalty = reward_config.get("smoothness_penalty", -0.05)
        
        self.terminal_f1_weight = reward_config.get("terminal_f1_weight", 10.0)
        self.use_potential_shaping = reward_config.get("use_potential_shaping", True)

        self.tolerance = config.get("environment", {}).get("tolerance", 2.0)
        self.gamma = config.get("training", {}).get("ppo", {}).get("gamma", 0.99)

        self.out_of_bounds_penalty = -10.0
        self.off_track_termination_penalty = -5.0

    def compute_step_reward(
        self,
        distance: float,
        is_revisit: bool,
        is_on_track: bool,
        new_coverage: float,
        prev_distance: float,
        action: int,
        prev_action: Optional[int],
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
            reward += self.gamma_off

        # 4. Revisit penalty
        if is_revisit:
            reward += self.lambda_revisit

        # 5. Step cost
        reward += self.step_cost

        # --- 6. NEW: Smoothness penalty (Anti-Jitter) ---
        if prev_action is not None and action != 8 and prev_action != 8:
            # Calculate the turn angle on an 8-direction circle
            # 0 = straight, 1 = 45 deg, 2 = 90 deg, 3 = 135 deg, 4 = 180 deg (U-turn)
            turn_angle = abs(action - prev_action)
            if turn_angle > 4:
                turn_angle = 8 - turn_angle
            
            # Penalize any turn that is 90 degrees or sharper
            if turn_angle >= 2:
                # Multiply penalty by the severity of the turn
                reward += self.smoothness_penalty * (turn_angle - 1)

        # 7. Potential-based shaping
        if self.use_potential_shaping:
            reward += self.gamma * (-distance) - (-prev_distance)

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
