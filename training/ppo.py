# training/ppo.py
"""PPO algorithm with GAE for retinal vessel tracing.
Now featuring Vectorized Swarm Environments and Automatic Mixed Precision (AMP).
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
    """Stores rollout transitions for multiple environments simultaneously."""

    def __init__(self, steps: int, num_envs: int, obs_shape: tuple):
        self.steps = steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.steps, self.num_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.steps, self.num_envs), dtype=np.int64)
        self.log_probs = np.zeros((self.steps, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.steps, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.steps, self.num_envs), dtype=np.float32)
        self.step = 0

    def add(self, obs, actions, log_probs, rewards, values, dones):
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = rewards
        self.values[self.step] = values
        self.dones[self.step] = dones
        self.step += 1

    def compute_returns_and_advantages(
        self, last_values: np.ndarray, gamma: float, gae_lambda: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = np.zeros((self.steps, self.num_envs), dtype=np.float32)
        lastgaelam = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                nextnonterminal = 1.0 - self.dones[t]
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - self.dones[t]
                nextvalues = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + self.values

        # Flatten the batches for the GPU
        adv_flat = torch.from_numpy(advantages.reshape(-1))
        ret_flat = torch.from_numpy(returns.reshape(-1))

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        return ret_flat, adv_flat

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_flat = torch.tensor(self.obs.reshape(-1, *self.obs_shape), dtype=torch.float32)
        act_flat = torch.tensor(self.actions.reshape(-1), dtype=torch.long)
        log_flat = torch.tensor(self.log_probs.reshape(-1), dtype=torch.float32)
        return obs_flat, act_flat, log_flat

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

                done = False
                while not done:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    logits, _ = model(obs_t)
                    
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
# PPO TRAINER (AMP + VECTORIZED)
# ==========================================
class PPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        num_envs: int = 8,
        use_amp: bool = True,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.1,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
        mini_batch_size: int = 256,
        steps_per_iter: int = 2048,
        num_iterations: int = 400,
        eval_every: int = 25,
        save_every: int = 50,
        tolerance: float = 2.0,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.num_envs = num_envs
        self.use_amp = use_amp
        
        # AMP Scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
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

    def _ppo_update_ff(self, buffer: RolloutBuffer, last_values: np.ndarray) -> Dict[str, float]:
        returns, advantages = buffer.compute_returns_and_advantages(
            last_values, self.gamma, self.gae_lambda
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

                # AMP Forward Pass
                with torch.autocast('cuda', enabled=self.use_amp):
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

                # AMP Backward Pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                total_gn += grad_norm.item()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += entropy.item()
                n += 1

        with torch.no_grad():
            values_all = torch.tensor(buffer.values.reshape(-1), dtype=torch.float32)
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
            print(f"Loaded imitation weights")
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
        from environment.vessel_env import VectorizedVesselEnv
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        start_iteration, best_f1 = self.load_checkpoint(save_path, imitation_path)

        # Create vectorized environments for parallel data collection
        env = VectorizedVesselEnv(self.config, num_envs=self.num_envs, dataset=train_samples)
        
        # Calculate matrix dimensions
        obs_shape = (9, self.config["environment"]["observation_size"], self.config["environment"]["observation_size"])
        steps_per_env = self.steps_per_iter // self.num_envs
        
        buffer = RolloutBuffer(steps=steps_per_env, num_envs=self.num_envs, obs_shape=obs_shape)
        
        episode_rewards: deque = deque(maxlen=50)
        episode_lengths: deque = deque(maxlen=50)
        
        ep_rewards = np.zeros(self.num_envs, dtype=np.float32)
        ep_lengths = np.zeros(self.num_envs, dtype=np.int32)
        log_lines: List[str] = []

        obs, _ = env.reset()

        print(f"\nStarting Swarm PPO | {self.num_envs} Envs | AMP: {self.use_amp}")
        print(f"Iters {start_iteration}–{self.num_iterations} × {self.steps_per_iter} steps/iter\n")

        for iteration in range(start_iteration, self.num_iterations + 1):
            buffer.reset()
            self.model.eval()

            for _ in range(steps_per_env):
                obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
                
                # Get actions from the network for ALL 8 environments simultaneously
                with torch.no_grad():
                    with torch.autocast('cuda', enabled=self.use_amp):
                        out = self.model.get_action_and_value(obs_t)
                    action, log_prob, _, value = out

                actions_np = action.cpu().numpy()
                
                # Step the Swarm
                next_obs, rewards, terminateds, truncateds, infos = env.step(actions_np)
                dones = np.logical_or(terminateds, truncateds).astype(np.float32)

                norm_rewards = np.zeros_like(rewards, dtype=np.float32)
                for i in range(self.num_envs):
                    self.reward_normalizer.update(rewards[i])
                    norm_rewards[i] = self.reward_normalizer.normalize(rewards[i])
                    
                    ep_rewards[i] += rewards[i]
                    ep_lengths[i] += 1
                    
                    if dones[i]:
                        episode_rewards.append(ep_rewards[i])
                        episode_lengths.append(ep_lengths[i])
                        ep_rewards[i] = 0.0
                        ep_lengths[i] = 0

                buffer.add(
                    obs, 
                    actions_np, 
                    log_prob.cpu().numpy(), 
                    norm_rewards, 
                    value.cpu().numpy(), 
                    dones
                )
                obs = next_obs

            # Get the final boot-strap value
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
                with torch.autocast('cuda', enabled=self.use_amp):
                    last_values = self.model.get_value(obs_t).cpu().numpy()

            self.model.train()
            stats = self._ppo_update_ff(buffer, last_values)
            self.scheduler.step()

            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
            log = (
                f"Iter {iteration:4d}/{self.num_iterations}  "
                f"reward={mean_reward:7.3f}  ep_len={mean_length:6.1f}  "
                f"p_loss={stats['policy_loss']:7.4f}  v_loss={stats['value_loss']:6.4f}  "
                f"entropy={stats['entropy']:.3f}"
            )

            if iteration % self.eval_every == 0 and val_samples:
                ev = evaluate(self.model, val_samples, self.config, self.device, self.tolerance)
                log += f"  |  val_cov={ev['mean_coverage']:.3f}  val_f1={ev['mean_f1']:.3f}"

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

        print(f"\nDone. Best F1: {best_f1:.3f}")
        print(f"Weights: {save_path}")
