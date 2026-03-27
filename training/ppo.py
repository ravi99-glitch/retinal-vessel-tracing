"""PPO algorithm with GAE for retinal vessel tracing

Provides:
    RolloutBuffer   — stores transitions (including LSTM states), computes GAE
    evaluate()      — runs n greedy episodes on val samples, returns mean F1
    PPOTrainer      — rollout collection, PPO update, training loop, checkpointing

Supports both feedforward and recurrent (LSTM) policies:
  - Feedforward: standard random mini-batch PPO
  - LSTM: sequential chunk-based PPO with hidden state management

Used by:
    scripts/train_ppo.py  (DRIVE)
"""

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# ROLLOUT BUFFER  (LSTM-aware)
# ==========================================


class RolloutBuffer:
    """Stores rollout transitions.

    When the policy uses an LSTM, additionally stores per-step LSTM
    hidden states (captured *before* the action was taken) so that
    chunk-based recurrent training can reconstruct the correct context.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[float] = []
        # LSTM bookkeeping — stored on CPU to save GPU memory
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
# EVALUATION  (LSTM-aware)
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
    Properly passes LSTM hidden state step-by-step when use_lstm=True.
    """
    from data.centerline_extraction import compute_centerline_f1
    from environment.vessel_env import VesselTracingEnv

    model.eval()
    # use_lstm = getattr(model, "use_lstm", False)
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

                # Fresh LSTM state per episode
                lstm_state = model.init_hidden(batch_size=1, device=device)

                done = False
                while not done:
                    obs_t = (
                        torch.tensor(obs, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )
                    logits, _, lstm_state = model(obs_t, lstm_state)
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
# PPO TRAINER  (LSTM-aware)
# ==========================================


class PPOTrainer:
    """PPO trainer with GAE, supporting both feedforward and LSTM policies.

    When the policy has use_lstm=True:
      - Rollout collection passes hidden state step-by-step, resets on done
      - Each step stores its LSTM state (before the action) for training
      - PPO update uses forward_sequence() on sequential chunks
      - Chunks are contiguous slices of the rollout; forward_sequence()
        handles hidden state resets at episode boundaries via done masks

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
        lstm_chunk_length: int = 32,
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

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_iterations,
        )

    # ------------------------------------------------------------------
    # PPO update — feedforward (original path, unchanged logic)
    # ------------------------------------------------------------------

    def _ppo_update_ff(
        self, buffer: RolloutBuffer, last_value: float
    ) -> Dict[str, float]:
        """Standard feedforward PPO update with random mini-batches."""
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
            perm = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.mini_batch_size):
                idx = perm[start : start + self.mini_batch_size]

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
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += entropy.item()
                n += 1

        return {
            "policy_loss": total_p / max(n, 1),
            "value_loss": total_v / max(n, 1),
            "entropy": total_e / max(n, 1),
        }

    # ------------------------------------------------------------------
    # PPO update — recurrent (sequential chunks)
    # ------------------------------------------------------------------

    def _ppo_update_lstm(
        self, buffer: RolloutBuffer, last_value: float
    ) -> Dict[str, float]:
        """Recurrent PPO update using sequential chunks.

        1. Compute GAE returns/advantages for the whole rollout (same as FF).
        2. Split the rollout into contiguous chunks of lstm_chunk_length.
        3. For each PPO epoch, shuffle the *chunks* (not individual steps)
           and run forward_sequence() on each.

        Each chunk uses the LSTM state stored at its first timestep as
        init_state, and passes done masks so forward_sequence() resets
        the hidden state at episode boundaries.
        """
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Full rollout tensors (stay on CPU until chunk extraction)
        all_obs = torch.tensor(np.array(buffer.obs), dtype=torch.float32)
        all_actions = torch.tensor(buffer.actions, dtype=torch.long)
        all_old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
        all_dones = torch.tensor(buffer.dones, dtype=torch.float32)

        # Build chunk index list
        T_total = len(buffer.obs)
        chunk_length = self.lstm_chunk_length
        chunk_indices: List[Tuple[int, int]] = []
        for s in range(0, T_total, chunk_length):
            e = min(s + chunk_length, T_total)
            if e - s >= 2:
                chunk_indices.append((s, e))

        total_p, total_v, total_e, n = 0.0, 0.0, 0.0, 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(len(chunk_indices))

            for ci in perm:
                s, e = chunk_indices[ci.item()]
                B = 1  # single-env rollout

                # Shape: (T_chunk, C, H, W) → (T_chunk, 1, C, H, W)
                chunk_obs = all_obs[s:e].unsqueeze(1).to(self.device)
                chunk_actions = all_actions[s:e].to(self.device)
                chunk_old_lp = all_old_log_probs[s:e].to(self.device)
                chunk_returns = returns[s:e].to(self.device)
                chunk_advs = advantages[s:e].to(self.device)
                chunk_dones = all_dones[s:e].unsqueeze(1).to(self.device)  # (T, 1)

                # Retrieve initial hidden state stored during rollout
                init_state = buffer.lstm_states[s]
                if init_state is not None:
                    init_state = (
                        init_state[0].to(self.device),
                        init_state[1].to(self.device),
                    )
                else:
                    init_state = self.model.init_hidden(
                        batch_size=B, device=self.device
                    )

                # Forward through the chunk sequentially
                logits_seq, values_seq = self.model.forward_sequence(
                    chunk_obs, init_state, chunk_dones
                )
                # logits_seq: (T, 1, N_ACTIONS),  values_seq: (T, 1)

                logits_flat = logits_seq.squeeze(1)  # (T, N_ACTIONS)
                values_flat = values_seq.squeeze(1)  # (T,)

                dist = torch.distributions.Categorical(logits=logits_flat)
                log_prob = dist.log_prob(chunk_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_prob - chunk_old_lp)
                surr1 = ratio * chunk_advs
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * chunk_advs
                )
                p_loss = -torch.min(surr1, surr2).mean()

                v_loss = nn.functional.mse_loss(
                    torch.clamp(values_flat, -10.0, 10.0),
                    torch.clamp(chunk_returns, -10.0, 10.0),
                )

                loss = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += entropy.item()
                n += 1

        return {
            "policy_loss": total_p / max(n, 1),
            "value_loss": total_v / max(n, 1),
            "entropy": total_e / max(n, 1),
        }

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _ppo_update(
        self, buffer: RolloutBuffer, last_value: float
    ) -> Dict[str, float]:
        if self.use_lstm:
            return self._ppo_update_lstm(buffer, last_value)
        return self._ppo_update_ff(buffer, last_value)

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
            # strict=False: imitation checkpoint may lack LSTM / value head weights
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
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
        """Full PPO training loop with LSTM support.

        When use_lstm=True:
          - Hidden state is carried step-by-step during rollout
          - Hidden state is reset when episodes end (done=True)
          - Each step stores the LSTM state *before* the action
          - PPO update splits rollout into sequential chunks and uses
            forward_sequence() with done masks for hidden state resets
        """
        from environment.vessel_env import VesselTracingEnv

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        start_iteration, best_f1 = self.load_checkpoint(save_path, imitation_path)

        env = VesselTracingEnv(self.config)
        buffer = RolloutBuffer()
        episode_rewards: deque = deque(maxlen=50)
        episode_lengths: deque = deque(maxlen=50)
        log_lines: List[str] = []

        # Initial episode setup
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

        # Initialize LSTM hidden state (returns None when use_lstm=False)
        lstm_state = self.model.init_hidden(batch_size=1, device=self.device)

        print(
            f"\nStarting PPO — iters {start_iteration}–{self.num_iterations} "
            f"× {self.steps_per_iter} steps"
            f"  LSTM={'ON chunk_len=' + str(self.lstm_chunk_length) if self.use_lstm else 'OFF'}\n"
        )

        for iteration in range(start_iteration, self.num_iterations + 1):
            buffer.reset()
            self.model.eval()

            # --- Collect rollout ---
            for _ in range(self.steps_per_iter):
                obs_t = (
                    torch.tensor(obs, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                with torch.no_grad():
                    action, log_prob, _, value, new_lstm_state = (
                        self.model.get_action_and_value(obs_t, lstm_state)
                    )

                next_obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

                # Store transition with LSTM state *before* this step
                buffer.add(
                    obs,
                    action.item(),
                    log_prob.item(),
                    np.clip(reward, -1.0, 1.0),
                    value.item(),
                    float(done),
                    lstm_state,
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

                    # Reset LSTM state for new episode
                    lstm_state = self.model.init_hidden(
                        batch_size=1, device=self.device
                    )

                    current = np.random.choice(train_samples)
                    env.set_data(
                        image=current["image"],
                        centerline=current["centerline"],
                        distance_transform=current["distance_transform"],
                        fov_mask=current["fov_mask"],
                    )
                    obs, _ = env.reset()

            # Bootstrap value for GAE
            with torch.no_grad():
                obs_t = (
                    torch.tensor(obs, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                last_value = self.model.get_value(obs_t, lstm_state).item()

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
                    self.model,
                    val_samples,
                    self.config,
                    self.device,
                    self.tolerance,
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