"""Reward calculation logic for vessel tracing agent."""

from typing import Any, Dict, Optional

class RewardCalculator:
    """
    Calculates step-wise rewards for the RL agent.
    
    Includes standard proximity/coverage rewards and the 'Topology Magnet'
    to encourage bridging gaps between vessel segments.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes reward weights from the provided configuration.
        """
        reward_cfg = config.get("reward", {})
        env_cfg = config.get("environment", {})

        # Standard Weights
        self.alpha = reward_cfg.get("alpha", 1.0)              # Proximity weight
        self.beta = reward_cfg.get("beta", 0.1)                # Coverage weight
        self.gamma_off = reward_cfg.get("gamma_off", -0.5)     # Off-track penalty
        self.lambda_revisit = reward_cfg.get("lambda_revisit", -1.0) # Revisit penalty
        self.step_cost = reward_cfg.get("step_cost", -0.01)    # Base step cost
        
        # Anti-Jitter Weights
        self.smoothness_penalty = reward_cfg.get("smoothness_penalty", -0.05)
        
        # Topology/Magnet Weights
        self.magnet_strength = reward_cfg.get("magnet_strength", 0.05)
        self.tolerance = env_cfg.get("tolerance", 2.0)
        
        # Potential Shaping
        self.use_potential_shaping = reward_cfg.get("use_potential_shaping", False)
        self.gamma_shaping = reward_cfg.get("gamma_shaping", 0.99)

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
        """
        Computes the final reward for a single step.
        """
        reward = 0.0

        # 1. Proximity reward (dense) — only rewarded on unvisited pixels
        if not is_revisit:
            reward += self.alpha * max(0.0, 1.0 - distance / self.tolerance)

        # 2. Coverage bonus (dense) — rewarded for discovering new centerline pixels
        if new_coverage > 0:
            reward += self.beta * new_coverage
            
        # 3. Off-track penalty — applied when agent drifts outside tolerance
        if not is_on_track:
            reward += self.gamma_off

        # 4. Revisit penalty — applied when agent loops back on itself
        if is_revisit:
            reward += self.lambda_revisit

        # 5. Base step cost — constant pressure to be efficient
        reward += self.step_cost

        # 6. Smoothness penalty (Anti-Jitter)
        # Penalizes sharp turns (e.g., immediate 180-degree turns)
        if prev_action is not None and action != 8 and prev_action != 8:
            turn_angle = abs(action - prev_action)
            if turn_angle > 4:
                turn_angle = 8 - turn_angle
            if turn_angle >= 2:
                reward += self.smoothness_penalty * (turn_angle - 1)

        # 7. Potential-based shaping
        # Uses the difference in distance transform to guide the agent
        if self.use_potential_shaping:
            reward += self.gamma_shaping * (-distance) - (-prev_distance)

        # 8. TOPOLOGY MAGNET (clDice Helper)
        # Pulls the agent across gaps by rewarding distance reduction to the next segment
        if not is_on_track and not is_revisit and action != 8:
             # If distance is decreasing, the agent is moving toward an unexplored branch
             if distance < prev_distance:
                 reward += self.magnet_strength
             else:
                 # Penalize moving deeper into 'unvessel' space
                 reward -= self.magnet_strength
                 
        return reward

    def compute_out_of_bounds_penalty(self) -> float:
        """Penalty for hitting the edge of the image/FOV."""
        return -2.0

    def compute_off_track_termination_penalty(self) -> float:
        """Penalty for exceeding the max off-track streak."""
        return -1.0
