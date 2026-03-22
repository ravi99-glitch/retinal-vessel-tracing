# rl_training/ppo.py
"""PPO algorithm with GAE for retinal vessel tracing

Provides:
    RolloutBuffer   — stores transitions, computes GAE returns + advantages
    evaluate()      — runs n greedy episodes on val samples, returns mean F1
    PPOTrainer      — rollout collection, PPO update, training loop, checkpointing

Used by:
    scripts/train_ppo.py  (DRIVE)
"""

import os
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# ROLLOUT BUFFER
# ==========================================


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = self.rewards
        values = self.values + [last_value]
        dones = self.dones

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(self.values, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = torch.tensor(np.array(self.obs), dtype=torch.float32)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32)
        return obs, actions, log_probs


# ==========================================
# EVALUATION
# ==========================================


def evaluate(
    model: nn.Module,
    val_samples: List[Dict],
    config: dict,
    device: torch.device,
    tolerance: float,
    n_episodes: int = 5,
) -> Dict[str, float]:
    """Run n greedy episodes per val sample.
    Returns mean_coverage and mean_f1.
    """
    from data.centerline_extraction import compute_centerline_f1
    from rl_environment.vessel_env import VesselTracingEnv

    model.eval()
    coverages = []
    f1_scores = []

    with torch.no_grad():
        for sample in val_samples:
            env = VesselTracingEnv(config)
            env.set_data(
                image=sample["image"],
                centerline=sample["centerline"],
                distance_transform=sample["distance_transform"],
                fov_mask=sample["fov_mask"],
            )

            cl_points = np.argwhere(sample["centerline"] > 0)
            if len(cl_points) == 0:
                continue

            for _ in range(n_episodes):
                idx = np.random.randint(len(cl_points))
                start = tuple(cl_points[idx])
                obs, _ = env.reset(start_position=start)

                done = False
                while not done:
                    obs_t = (
                        torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    )
                    logits, _, _ = model(obs_t)
                    action = logits.argmax(dim=-1).item()
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                coverages.append(info["coverage_ratio"])
                metrics = compute_centerline_f1(
                    env.covered_centerline,
                    sample["centerline"],
                    tolerance=tolerance,
                )
                f1_scores.append(metrics["f1"])

    model.train()
    return {
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
    }


# ==========================================
# PPO TRAINER
# ==========================================


class PPOTrainer:
    """PPO trainer with GAE.

    Usage:
        trainer = PPOTrainer(model, config, device)
        trainer.train(train_samples, val_samples, save_path, log_path)
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.1,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
        mini_batch_size: int = 256,
        steps_per_iter: int = 4096,
        num_iterations: int = 1000,
        eval_every: int = 25,
        save_every: int = 50,
        tolerance: float = 2.0,
    ):

        self.model = model
        self.config = config
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.steps_per_iter = steps_per_iter
        self.num_iterations = num_iterations
        self.eval_every = eval_every
        self.save_every = save_every
        self.tolerance = tolerance

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_iterations,
        )

    def _ppo_update(self, buffer: RolloutBuffer, last_value: float) -> Dict[str, float]:
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        obs, actions, old_log_probs = buffer.get_tensors()

        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        total_p, total_v, total_e, n = 0.0, 0.0, 0.0, 0
        dataset_size = len(obs)

        for _ in range(self.ppo_epochs):
            for start in range(0, dataset_size, self.mini_batch_size):
                idx = torch.randperm(dataset_size)[start : start + self.mini_batch_size]

                logits, values, _ = self.model(obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_prob - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * advantages[idx]
                )
                p_loss = -torch.min(surr1, surr2).mean()

                v_loss = nn.functional.mse_loss(
                    torch.clamp(values, -10.0, 10.0),
                    torch.clamp(returns[idx], -10.0, 10.0),
                )

                loss = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += entropy.item()
                n += 1

        return {
            "policy_loss": total_p / n,
            "value_loss": total_v / n,
            "entropy": total_e / n,
        }

    def load_checkpoint(self, save_path: str, imitation_path: str) -> Tuple[int, float]:
        """Resume from PPO checkpoint if it exists,
        otherwise load imitation weights.
        Returns (start_iteration, best_f1).
        """
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start = ckpt.get("iteration", 0) + 1
            best = ckpt.get("best_f1", 0.0)
            print(f"Resumed from PPO checkpoint  iter={start-1}  best_F1={best:.3f}")
            return start, best

        if os.path.exists(imitation_path):
            ckpt = torch.load(
                imitation_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded imitation weights  val_acc={ckpt.get('val_acc', 0):.3f}")
            # Reset value head — never trained during imitation
            for layer in self.model.value_head:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)
            print("Value head re-initialized.")
            return 1, 0.0

        print("WARNING: No weights found, training from scratch.")
        return 1, 0.0

    def train(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        save_path: str,
        log_path: str,
        imitation_path: str = "",
    ) -> None:
        """Full PPO training loop.

        Args:
            train_samples:   list of sample dicts (image, centerline, distance_transform, fov_mask)
            val_samples:     list of sample dicts for periodic evaluation
            save_path:       where to save best checkpoint
            log_path:        where to write training log
            imitation_path:  fallback weights if no PPO checkpoint exists

        """
        from rl_environment.vessel_env import VesselTracingEnv

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        start_iteration, best_f1 = self.load_checkpoint(save_path, imitation_path)

        env = VesselTracingEnv(self.config)
        buffer = RolloutBuffer()
        episode_rewards = deque(maxlen=50)
        episode_lengths = deque(maxlen=50)
        log_lines = []

        # Initial episode
        current = np.random.choice(train_samples)
        env.set_data(
            image=current["image"],
            centerline=current["centerline"],
            distance_transform=current["distance_transform"],
            fov_mask=current["fov_mask"],
        )
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_length = 0

        print(
            f"\nStarting PPO — iters {start_iteration}–{self.num_iterations} "
            f"× {self.steps_per_iter} steps\n"
        )

        for iteration in range(start_iteration, self.num_iterations + 1):
            buffer.reset()
            self.model.eval()

            # --- Collect rollout ---
            for _ in range(self.steps_per_iter):
                obs_t = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                with torch.no_grad():
                    action, log_prob, _, value, _ = self.model.get_action_and_value(
                        obs_t
                    )

                next_obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

                buffer.add(
                    obs,
                    action.item(),
                    log_prob.item(),
                    np.clip(reward, -1.0, 1.0),
                    value.item(),
                    float(done),
                )

                ep_reward += reward
                ep_length += 1
                obs = next_obs

                if done:
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    ep_reward = 0.0
                    ep_length = 0
                    current = np.random.choice(train_samples)
                    env.set_data(
                        image=current["image"],
                        centerline=current["centerline"],
                        distance_transform=current["distance_transform"],
                        fov_mask=current["fov_mask"],
                    )
                    obs, _ = env.reset()

            # Bootstrap
            with torch.no_grad():
                obs_t = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                last_value = self.model.get_value(obs_t).item()

            # --- Update ---
            self.model.train()
            stats = self._ppo_update(buffer, last_value)
            self.scheduler.step()

            # --- Log ---
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
            log = (
                f"Iter {iteration:4d}/{self.num_iterations}  "
                f"reward={mean_reward:7.3f}  ep_len={mean_length:6.1f}  "
                f"p_loss={stats['policy_loss']:7.4f}  "
                f"v_loss={stats['value_loss']:6.4f}  "
                f"entropy={stats['entropy']:.3f}"
            )

            # --- Eval ---
            if iteration % self.eval_every == 0 and val_samples:
                ev = evaluate(
                    self.model, val_samples, self.config, self.device, self.tolerance
                )
                log += (
                    f"  |  val_coverage={ev['mean_coverage']:.3f}"
                    f"  val_f1={ev['mean_f1']:.3f}"
                )

                if ev["mean_f1"] > best_f1:
                    best_f1 = ev["mean_f1"]
                    torch.save(
                        {
                            "iteration": iteration,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "best_f1": best_f1,
                            "config": self.config,
                        },
                        save_path,
                    )
                    log += f"  ✓ saved (best F1={best_f1:.3f})"

            print(log)
            log_lines.append(log)

            # --- Periodic checkpoint ---
            if iteration % self.save_every == 0:
                ckpt_path = save_path.replace(".pt", f"_iter{iteration}.pt")
                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "config": self.config,
                    },
                    ckpt_path,
                )

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

        print(f"\nDone. Best F1: {best_f1:.3f}")
        print(f"Weights: {save_path}")
        print(f"Log:     {log_path}")
