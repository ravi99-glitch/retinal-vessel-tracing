"""PPO algorithm with GAE for retinal vessel tracing

Provides:
    RolloutBuffer   — stores transitions (including LSTM states), computes GAE
    evaluate()      — runs n greedy episodes on val samples, returns mean F1
    PPOTrainer      — rollout collection, PPO update, training loop, checkpointing

Supports both feedforward and recurrent (LSTM) policies.
"""

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ==========================================
# ROLLOUT BUFFER
# ==========================================

class RolloutBuffer:
    """Stores rollout transitions."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[float] = []
        self.lstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: float,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        if lstm_state is not None:
            self.lstm_states.append(
                (lstm_state[0].detach().cpu(), lstm_state[1].detach().cpu())
            )
        else:
            self.lstm_states.append(None)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(self.rewards)
        advantages = np.empty(n, dtype=np.float32)

        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        gae = 0.0
        next_value = last_value
        for t in range(n - 1, -1, -1):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
            next_value = values[t]

        advantages_t = torch.from_numpy(advantages)
        returns = advantages_t + torch.from_numpy(values)
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )
        return returns, advantages_t

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
    n_episodes: int = 1,
) -> Dict[str, float]:
    """Run n greedy episodes per val sample."""
    from data.centerline_extraction import compute_centerline_f1
    from environment.vessel_env import VesselTracingEnv

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

                lstm_state = model.init_hidden(batch_size=1, device=device)
                done = False
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    # Support for both ResNet and LSTM via conditional unpacking
                    out = model(obs_t, lstm_state)
                    if len(out) == 3:
                        logits, _, lstm_state = out
                    else:
                        logits, _ = out
                        
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


class RunningRewardNormalizer:
    """Track running mean/std of rewards for normalisation."""
    def __init__(self, clip: float = 10.0, gamma: float = 0.99):
        self.clip = clip
        self.gamma = gamma
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 1e-4

    def update(self, reward: float):
        self.count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        delta2 = reward - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count

    def normalize(self, reward: float) -> float:
        std = max(np.sqrt(self.running_var), 1e-8)
        return np.clip((reward - self.running_mean) / std, -self.clip, self.clip)


# ==========================================
# PPO TRAINER 
# ==========================================

class PPOTrainer:
    """PPO trainer with GAE."""

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
        lstm_chunk_length: int = 32,
        use_wandb: bool = False,
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
        self.lstm_chunk_length = lstm_chunk_length
        self.use_lstm = getattr(model, "use_lstm", False)
        self.value_clamp = config.get("training", {}).get("value_clamp", 10.0)

        self.reward_normalizer = RunningRewardNormalizer(
            clip=config.get("training", {}).get("reward_norm_clip", 10.0)
        )

        self.patience = config.get("training", {}).get("patience", 100)
        self.no_improve_count = 0

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=config.get("training", {}).get("lr_end_factor", 0.1),
            total_iters=num_iterations,
        )

        self.use_wandb = use_wandb and _WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project="vessel-tracing", config=config, resume="allow")

    def _ppo_update_ff(self, buffer: RolloutBuffer, last_value: float) -> Dict[str, float]:
        """Standard feedforward PPO update."""
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        obs, actions, old_log_probs = buffer.get_tensors()

        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        total_p, total_v, total_e, total_kl, total_gn, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        dataset_size = len(obs)

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.mini_batch_size):
                idx = perm[start : start + self.mini_batch_size]

                out = self.model(obs[idx])
                logits = out[0]
                values = out[1]
                
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_prob - old_log_probs[idx])

                with torch.no_grad():
                    approx_kl = (((ratio - 1) - (log_prob - old_log_probs[idx])).mean().item())
                    total_kl += approx_kl

                surr1 = ratio * advantages[idx]
                surr2 = (torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx])
                p_loss = -torch.min(surr1, surr2).mean()

                v_loss = nn.functional.mse_loss(
                    torch.clamp(values, -self.value_clamp, self.value_clamp),
                    torch.clamp(returns[idx], -self.value_clamp, self.value_clamp),
                )

                loss = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                total_gn += grad_norm.item()
                self.optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += entropy.item()
                n += 1

        with torch.no_grad():
            values_all = torch.tensor(buffer.values, dtype=torch.float32)
            ev = (1 - (returns.cpu() - values_all).var() / (returns.cpu().var() + 1e-8)).item()

        return {
            "policy_loss": total_p / max(n, 1),
            "value_loss": total_v / max(n, 1),
            "entropy": total_e / max(n, 1),
            "approx_kl": total_kl / max(n, 1),
            "grad_norm": total_gn / max(n, 1),
            "explained_variance": ev,
        }

    def load_checkpoint(self, save_path: str, imitation_path: str) -> Tuple[int, float]:
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
            ckpt = torch.load(imitation_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Loaded imitation weights  val_acc={ckpt.get('val_acc', 0):.3f}")
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
        from environment.vessel_env import VesselTracingEnv
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        start_iteration, best_f1 = self.load_checkpoint(save_path, imitation_path)

        env = VesselTracingEnv(self.config)
        buffer = RolloutBuffer()
        episode_rewards: deque = deque(maxlen=50)
        episode_lengths: deque = deque(maxlen=50)
        log_lines: List[str] = []

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
        lstm_state = self.model.init_hidden(batch_size=1, device=self.device)

        print(f"\nStarting PPO — iters {start_iteration}–{self.num_iterations} × {self.steps_per_iter} steps\n")

        for iteration in range(start_iteration, self.num_iterations + 1):
            buffer.reset()
            self.model.eval()

            for _ in range(self.steps_per_iter):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    # Handle both ResNet and LSTM modes smoothly
                    if hasattr(self.model, 'get_action_and_value'):
                        out = self.model.get_action_and_value(obs_t, lstm_state)
                        if len(out) == 5:
                            action, log_prob, _, value, new_lstm_state = out
                        else:
                            action, log_prob, _, value = out
                            new_lstm_state = None
                    else:
                        raise AttributeError("Model must implement get_action_and_value")

                next_obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

                self.reward_normalizer.update(reward)
                norm_reward = self.reward_normalizer.normalize(reward)

                buffer.add(
                    obs, action.item(), log_prob.item(), norm_reward, value.item(), float(done), lstm_state,
                )

                ep_reward += reward
                ep_length += 1
                obs = next_obs
                lstm_state = new_lstm_state

                if done:
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    ep_reward = 0.0
                    ep_length = 0
                    lstm_state = self.model.init_hidden(batch_size=1, device=self.device)

                    current = np.random.choice(train_samples)
                    env.set_data(
                        image=current["image"],
                        centerline=current["centerline"],
                        distance_transform=current["distance_transform"],
                        fov_mask=current["fov_mask"],
                    )
                    obs, _ = env.reset()

            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                last_value = self.model.get_value(obs_t, lstm_state).item()

            self.model.train()
            stats = self._ppo_update_ff(buffer, last_value)
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
            log = (
                f"Iter {iteration:4d}/{self.num_iterations}  "
                f"reward={mean_reward:7.3f}  ep_len={mean_length:6.1f}  "
                f"p_loss={stats['policy_loss']:7.4f}  v_loss={stats['value_loss']:6.4f}  "
                f"entropy={stats['entropy']:.3f}"
            )

            if self.use_wandb:
                wandb.log({
                    "train/policy_loss": stats["policy_loss"],
                    "train/value_loss": stats["value_loss"],
                    "train/entropy": stats["entropy"],
                    "train/approx_kl": stats["approx_kl"],
                    "train/explained_variance": stats["explained_variance"],
                    "train/mean_reward": mean_reward,
                    "train/mean_ep_length": mean_length,
                    "train/lr": current_lr,
                }, step=iteration)

            if iteration % self.eval_every == 0 and val_samples:
                ev = evaluate(self.model, val_samples, self.config, self.device, self.tolerance)
                log += f"  |  val_cov={ev['mean_coverage']:.3f}  val_f1={ev['mean_f1']:.3f}"

                if self.use_wandb:
                    wandb.log({"eval/mean_f1": ev["mean_f1"], "eval/mean_coverage": ev["mean_coverage"]}, step=iteration)

                if ev["mean_f1"] > best_f1:
                    best_f1 = ev["mean_f1"]
                    self.no_improve_count = 0
                    torch.save({
                        "iteration": iteration,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "best_f1": best_f1,
                        "config": self.config,
                    }, save_path)
                    log += f"  ✓ saved (best F1={best_f1:.3f})"
                else:
                    self.no_improve_count += 1

                if self.no_improve_count >= self.patience:
                    print(log)
                    print(f"\nEarly stopping: no improvement for {self.patience} eval cycles.")
                    break

            print(log)
            log_lines.append(log)

            if iteration % self.save_every == 0:
                ckpt_path = save_path.replace(".pt", f"_iter{iteration}.pt")
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                }, ckpt_path)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

        if self.use_wandb:
            wandb.finish()

        print(f"\nDone. Best F1: {best_f1:.3f}")
        print(f"Weights: {save_path}")
