# training/ppo.py
"""PPO algorithm with GAE for retinal vessel tracing

Provides:
    RolloutBuffer   — stores transitions (including LSTM states), computes GAE
    evaluate()      — runs n greedy episodes on val samples, returns mean F1
    PPOTrainer      — rollout collection, PPO update, training loop, checkpointing

Supports both feedforward and recurrent (LSTM) policies:
  - Feedforward: standard random mini-batch PPO
  - LSTM: sequential chunk-based PPO with hidden state management

Used by:
    scripts/train_ppo.py
"""

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import csv

from environment.reward import RewardCalculator
from training.curriculum import CurriculumManager

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
        return returns, advantages_t

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
    n_episodes: int = 1,
    n_parallel: int = 32,
) -> Dict[str, float]:
    """Run n greedy episodes per val sample with batched GPU inference."""
    from data.centerline_extraction import compute_centerline_f1
    from environment.vessel_env import VesselTracingEnv

    model.eval()
    use_lstm = getattr(model, "use_lstm", False)
    coverages, f1_scores, cldice_scores = [], [], []
    per_sample_coverage: Dict[int, np.ndarray] = {}
    per_sample_vessel: Dict[int, np.ndarray] = {}

    work_queue: deque = deque()
    for sample in val_samples:
        cl_points = np.argwhere(sample["centerline"] > 0)
        if len(cl_points) == 0:
            continue
        for _ in range(n_episodes):
            idx = np.random.randint(len(cl_points))
            work_queue.append((sample, tuple(cl_points[idx])))

    if not work_queue:
        model.train()
        return {"mean_coverage": 0.0, "mean_f1": 0.0, "mean_cldice": 0.0}

    n_slots = min(n_parallel, len(work_queue))
    envs: List[Optional[object]] = [None] * n_slots
    samples_ref: List[Optional[Dict]] = [None] * n_slots
    obs_list: List[Optional[np.ndarray]] = [None] * n_slots
    lstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_slots
    active = [False] * n_slots

    def _start_slot(slot, sample, start_pos):
        env = VesselTracingEnv(config)
        env.set_data(
            image=sample["image"], centerline=sample["centerline"],
            distance_transform=sample["distance_transform"], fov_mask=sample["fov_mask"],
            vessel_orientation=sample.get("vessel_orientation"), dt_gradient=sample.get("dt_gradient"),
            vesselness=sample.get("vesselness"), unet_prior=sample.get("unet_prior"),
        )
        obs, _ = env.reset(start_position=start_pos)
        envs[slot] = env
        samples_ref[slot] = sample
        obs_list[slot] = obs
        lstm_states[slot] = model.init_hidden(batch_size=1, device=device)
        active[slot] = True

    for i in range(n_slots):
        if work_queue:
            sample, start = work_queue.popleft()
            _start_slot(i, sample, start)

    with torch.no_grad():
        while any(active):
            active_idx = [i for i in range(n_slots) if active[i]]
            if not active_idx: break

            # [SPEEDUP]: non_blocking=True
            obs_batch = torch.from_numpy(np.stack([obs_list[i] for i in active_idx])).float().to(device, non_blocking=True)

            if use_lstm:
                h_cat = torch.cat([lstm_states[i][0].to(device, non_blocking=True) for i in active_idx], dim=0)
                c_cat = torch.cat([lstm_states[i][1].to(device, non_blocking=True) for i in active_idx], dim=0)
                batched_lstm = (h_cat, c_cat)
            else:
                batched_lstm = None

            # [SPEEDUP]: Evaluate in half precision
            with torch.amp.autocast("cuda") if device.type == "cuda" else torch.autocast("cpu", enabled=False):
                logits, _, new_lstm = model(obs_batch, batched_lstm)
            actions = logits.argmax(dim=-1)

            for j, i in enumerate(active_idx):
                obs, _, terminated, truncated, info = envs[i].step(actions[j].item())
                obs_list[i] = obs

                if use_lstm and new_lstm is not None:
                    lstm_states[i] = (
                        new_lstm[0][j:j+1, :].detach().cpu(),
                        new_lstm[1][j:j+1, :].detach().cpu(),
                    )

                if terminated or truncated:
                    coverages.append(info["coverage_ratio"])
                    metrics = compute_centerline_f1(envs[i].covered_centerline, samples_ref[i]["centerline"], tolerance=tolerance)
                    f1_scores.append(metrics["f1"])

                    cov = envs[i].covered_centerline
                    if cov is not None:
                        s_key = id(samples_ref[i])
                        if s_key not in per_sample_coverage:
                            per_sample_coverage[s_key] = (cov > 0).astype(np.float32)
                            per_sample_vessel[s_key] = samples_ref[i].get("vessel_mask", samples_ref[i]["centerline"])
                        else:
                            per_sample_coverage[s_key] = np.where(cov > 0, 1.0, per_sample_coverage[s_key])

                    if work_queue:
                        sample, start = work_queue.popleft()
                        _start_slot(i, sample, start)
                    else:
                        active[i] = False

    from evaluation.metrics import CenterlineMetrics
    _cm = CenterlineMetrics()
    for s_key, cov in per_sample_coverage.items():
        cldice_scores.append(_cm.cl_dice(cov, per_sample_vessel[s_key]))

    model.train()
    return {
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "mean_cldice": float(np.mean(cldice_scores)) if cldice_scores else 0.0,
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
# PPO TRAINER
# ==========================================


class PPOTrainer:
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
        self.value_clamp = config.get("training", {}).get("value_clamp", 10.0)

        # [SPEEDUP]: Initialize AMP Scaler for mixed precision training
        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        ppo_cfg = config.get("training", {}).get("ppo", {})
        target_kl = ppo_cfg.get("target_kl", None)
        self.target_kl: Optional[float] = float(target_kl) if target_kl else None

        self._stage_iter: int = 0
        self._eval_cldice_window: deque = deque(maxlen=3)
        self._entropy_frozen: bool = False
        self._last_stage_idx: int = 0

        shaping_gamma = config.get("reward", {}).get("shaping_gamma", gamma)
        if abs(shaping_gamma - gamma) > 1e-6:
            raise ValueError("shaping_gamma must equal ppo.gamma")

        from models.policy_network import _junction_channel_idx
        self._junction_ch_idx: Optional[int] = _junction_channel_idx(config)
        obs_size = config.get("environment", {}).get("observation_size", 65)
        self._obs_center: int = obs_size // 2

        self.curriculum = CurriculumManager(config)
        self.reward_normalizer = RunningRewardNormalizer(clip=config.get("training", {}).get("reward_norm_clip", 10.0))
        self.terminal_reward_normalizer = RunningRewardNormalizer(clip=config.get("training", {}).get("terminal_norm_clip", 5.0))

        self.patience = config.get("training", {}).get("patience", 100)
        self.no_improve_count = 0

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        lr_end_factor = config.get("training", {}).get("lr_end_factor", 0.1)
        warmup_iters = ppo_cfg.get("lr_warmup_iters", 0)

        if warmup_iters > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
            decay_sched = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=max(num_iterations - warmup_iters, 1))
            self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_sched, decay_sched], milestones=[warmup_iters])
        else:
            self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=num_iterations)

    def _get_curriculum_overrides_dict(self) -> dict:
        overrides = self.curriculum.get_stage_overrides()
        result = {}
        env_ov = overrides.get("environment", {})
        if "max_off_track_streak" in env_ov: result["max_off_track"] = env_ov["max_off_track_streak"]
        if "max_steps_per_episode" in env_ov: result["max_steps"] = env_ov["max_steps_per_episode"]
        if "off_track_penalty_ramp" in env_ov: result["off_track_ramp"] = env_ov["off_track_penalty_ramp"]
        reward_ov = overrides.get("reward", {})
        if "smoothness_weight" in reward_ov: result["smoothness_weight"] = reward_ov["smoothness_weight"]

        train_ov = overrides.get("training", {})
        if "entropy_coef" in train_ov:
            stage = self.curriculum.get_current_stage()
            ec_start = float(train_ov["entropy_coef"])
            ec_end = getattr(stage, "entropy_coef_end", None)
            ec_iters = getattr(stage, "entropy_anneal_iters", 0) or 0
            if ec_end is not None and ec_iters > 0:
                if not self._entropy_frozen:
                    t = min(self._stage_iter, ec_iters) / float(ec_iters)
                    self.entropy_coef = ec_start + t * (float(ec_end) - ec_start)
            else:
                self.entropy_coef = ec_start

        return result

    # ------------------------------------------------------------------
    # PPO update — feedforward
    # ------------------------------------------------------------------

    def _ppo_update_ff(self, buffers: List[RolloutBuffer], last_values: List[float]) -> Dict[str, float]:
        all_returns, all_advantages = [], []
        all_obs, all_actions, all_old_log_probs = [], [], []
        all_values_raw = [] 

        for buf, lv in zip(buffers, last_values):
            if len(buf.rewards) == 0: continue
            ret, adv = buf.compute_returns_and_advantages(lv, self.gamma, self.gae_lambda)
            ep_w = torch.from_numpy(self._ep_length_weights(buf.dones))
            adv = adv * ep_w

            obs, actions, log_probs = buf.get_tensors()
            all_returns.append(ret)
            all_advantages.append(adv)
            all_obs.append(obs)
            all_actions.append(actions)
            all_old_log_probs.append(log_probs)
            all_values_raw.extend(buf.values)

        # [SPEEDUP]: non_blocking transfers for large concatenated buffers
        returns = torch.cat(all_returns).to(self.device, non_blocking=True)
        advantages = torch.cat(all_advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device, non_blocking=True)
        
        obs = torch.cat(all_obs).to(self.device, non_blocking=True)
        actions = torch.cat(all_actions).to(self.device, non_blocking=True)
        old_log_probs = torch.cat(all_old_log_probs).to(self.device, non_blocking=True)

        total_p = total_v = total_e = total_kl = total_gn = 0.0
        max_gn = 0.0
        epochs_run = 0
        dataset_size = len(obs)

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(dataset_size)
            epoch_kl_sum, epoch_kl_n = 0.0, 0
            
            for start in range(0, dataset_size, self.mini_batch_size):
                idx = perm[start : start + self.mini_batch_size]

                # [SPEEDUP]: AMP Autocast + Optimized backward pass
                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits, values, _ = self.model(obs[idx])
                        dist = torch.distributions.Categorical(logits=logits)
                        log_prob = dist.log_prob(actions[idx])
                        entropy = dist.entropy().mean()
                        ratio = torch.exp(log_prob - old_log_probs[idx])

                        with torch.no_grad():
                            approx_kl = ((ratio - 1) - (log_prob - old_log_probs[idx])).mean().item()
                            total_kl += approx_kl
                            epoch_kl_sum += approx_kl
                            epoch_kl_n += 1

                        surr1 = ratio * advantages[idx]
                        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                        p_loss = -torch.min(surr1, surr2).mean()

                        v_loss = nn.functional.mse_loss(values, torch.clamp(returns[idx], -self.value_clamp, self.value_clamp))
                        loss = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                        if self.model.junction_head is not None and self._junction_ch_idx is not None:
                            enc_feats = self.model.encode(obs[idx])
                            j_logits = self.model.junction_head(enc_feats)
                            j_vals = obs[idx][:, self._junction_ch_idx, self._obs_center, self._obs_center]
                            j_class = torch.zeros(len(idx), dtype=torch.long, device=self.device)
                            j_class[j_vals > 0.7] = 2
                            j_class[(j_vals > 0.3) & (j_vals <= 0.7)] = 1
                            j_loss = nn.functional.cross_entropy(j_logits, j_class)
                            loss = loss + 0.1 * j_loss

                    # Fast zero_grad
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    gn_val = grad_norm.item()
                    total_gn += gn_val
                    if gn_val > max_gn: max_gn = gn_val
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    # STANDARD FALLBACK (CPU / No-AMP)
                    logits, values, _ = self.model(obs[idx])
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(actions[idx])
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(log_prob - old_log_probs[idx])

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (log_prob - old_log_probs[idx])).mean().item()
                        total_kl += approx_kl
                        epoch_kl_sum += approx_kl
                        epoch_kl_n += 1

                    surr1 = ratio * advantages[idx]
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                    p_loss = -torch.min(surr1, surr2).mean()

                    v_loss = nn.functional.mse_loss(values, torch.clamp(returns[idx], -self.value_clamp, self.value_clamp))
                    loss = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                    if self.model.junction_head is not None and self._junction_ch_idx is not None:
                        enc_feats = self.model.encode(obs[idx])
                        j_logits = self.model.junction_head(enc_feats)
                        j_vals = obs[idx][:, self._junction_ch_idx, self._obs_center, self._obs_center]
                        j_class = torch.zeros(len(idx), dtype=torch.long, device=self.device)
                        j_class[j_vals > 0.7] = 2
                        j_class[(j_vals > 0.3) & (j_vals <= 0.7)] = 1
                        j_loss = nn.functional.cross_entropy(j_logits, j_class)
                        loss = loss + 0.1 * j_loss

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    gn_val = grad_norm.item()
                    total_gn += gn_val
                    if gn_val > max_gn: max_gn = gn_val
                    self.optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += entropy.item()

            epochs_run += 1

            if self.target_kl is not None and epoch_kl_n > 0:
                epoch_kl_mean = epoch_kl_sum / epoch_kl_n
                if epoch_kl_mean > self.target_kl:
                    break

        with torch.no_grad():
            values_all = torch.tensor(all_values_raw, dtype=torch.float32)
            ev = (1 - (returns.cpu() - values_all).var() / (returns.cpu().var() + 1e-8)).item()

        n = max(dataset_size * epochs_run // self.mini_batch_size, 1)
        return {
            "policy_loss": total_p / n,
            "value_loss": total_v / n,
            "entropy": total_e / n,
            "approx_kl": total_kl / n,
            "grad_norm": total_gn / n,
            "grad_norm_max": max_gn,
            "epochs_run": epochs_run,
            "explained_variance": ev,
        }

    # ------------------------------------------------------------------
    # PPO update — recurrent 
    # ------------------------------------------------------------------

    def _ppo_update_lstm(self, buffers: List[RolloutBuffer], last_values: List[float]) -> Dict[str, float]:
        T_chunk = self.lstm_chunk_length

        per_buf_returns, per_buf_advantages, all_values_raw = [], [], []

        for buf, lv in zip(buffers, last_values):
            if len(buf.rewards) == 0:
                per_buf_returns.append(None); per_buf_advantages.append(None); continue
            ret, adv = buf.compute_returns_and_advantages(lv, self.gamma, self.gae_lambda)
            ep_w = torch.from_numpy(self._ep_length_weights(buf.dones))
            adv = adv * ep_w
            per_buf_returns.append(ret); per_buf_advantages.append(adv); all_values_raw.extend(buf.values)

        non_empty = [a for a in per_buf_advantages if a is not None]
        if non_empty:
            cat = torch.cat(non_empty)
            mean, std = cat.mean(), cat.std() + 1e-8
            per_buf_advantages = [None if a is None else (a - mean) / std for a in per_buf_advantages]

        chunk_specs: List[Tuple[int, int, int]] = []
        for buf_idx, buf in enumerate(buffers):
            T = len(buf.obs)
            for s in range(0, T, T_chunk):
                valid = min(T_chunk, T - s)
                if valid >= 2: chunk_specs.append((buf_idx, s, valid))

        if not chunk_specs:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "grad_norm": 0.0, "grad_norm_max": 0.0, "epochs_run": 0, "explained_variance": 0.0}

        chunks_per_batch = max(1, self.mini_batch_size // T_chunk)
        total_p = total_v = total_e = total_kl = total_gn = max_gn = 0.0
        n_updates = epochs_run = 0

        obs_shape = buffers[0].obs[0].shape
        C, H, W = obs_shape

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(len(chunk_specs)).tolist()
            epoch_kl_sum = epoch_kl_n = 0.0

            for batch_start in range(0, len(perm), chunks_per_batch):
                batch_idx = perm[batch_start:batch_start + chunks_per_batch]
                B = len(batch_idx)

                obs_np = np.zeros((T_chunk, B, C, H, W), dtype=np.float32)
                actions_np = np.zeros((T_chunk, B), dtype=np.int64)
                old_lp_np = np.zeros((T_chunk, B), dtype=np.float32)
                returns_np = np.zeros((T_chunk, B), dtype=np.float32)
                advs_np = np.zeros((T_chunk, B), dtype=np.float32)
                dones_np = np.zeros((T_chunk, B), dtype=np.float32)
                mask_np = np.zeros((T_chunk, B), dtype=np.float32)

                init_h_list, init_c_list = [], []

                for j, ci in enumerate(batch_idx):
                    buf_idx, s, valid = chunk_specs[ci]
                    buf = buffers[buf_idx]

                    obs_np[:valid, j] = np.asarray(buf.obs[s:s + valid], dtype=np.float32)
                    actions_np[:valid, j] = np.asarray(buf.actions[s:s + valid], dtype=np.int64)
                    old_lp_np[:valid, j] = np.asarray(buf.log_probs[s:s + valid], dtype=np.float32)
                    returns_np[:valid, j] = per_buf_returns[buf_idx][s:s + valid].numpy()
                    advs_np[:valid, j] = per_buf_advantages[buf_idx][s:s + valid].numpy()
                    dones_np[:valid, j] = np.asarray(buf.dones[s:s + valid], dtype=np.float32)
                    mask_np[:valid, j] = 1.0

                    init = buf.lstm_states[s]
                    if init is not None:
                        init_h_list.append(init[0].squeeze(0))
                        init_c_list.append(init[1].squeeze(0))
                    else:
                        zero = self.model.init_hidden(batch_size=1, device="cpu")
                        init_h_list.append(zero[0].squeeze(0))
                        init_c_list.append(zero[1].squeeze(0))

                # non_blocking=True
                obs_t = torch.from_numpy(obs_np).to(self.device, non_blocking=True)
                actions_t = torch.from_numpy(actions_np).to(self.device, non_blocking=True)
                old_lp_t = torch.from_numpy(old_lp_np).to(self.device, non_blocking=True)
                returns_t = torch.from_numpy(returns_np).to(self.device, non_blocking=True)
                advs_t = torch.from_numpy(advs_np).to(self.device, non_blocking=True)
                dones_t = torch.from_numpy(dones_np).to(self.device, non_blocking=True)
                mask_t = torch.from_numpy(mask_np).to(self.device, non_blocking=True)

                init_h = torch.stack(init_h_list, dim=0).to(self.device, non_blocking=True)
                init_c = torch.stack(init_c_list, dim=0).to(self.device, non_blocking=True)
                init_state = (init_h, init_c)

                use_junction_aux = (self.model.junction_head is not None and self._junction_ch_idx is not None)

                # AMP Autocast + Optimized backward pass
                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        fwd_result = self.model.forward_sequence(obs_t, init_state, dones_t, return_enc_features=use_junction_aux)
                        if use_junction_aux:
                            logits_seq, values_seq, enc_feats_flat = fwd_result
                        else:
                            logits_seq, values_seq = fwd_result

                        dist = torch.distributions.Categorical(logits=logits_seq)
                        log_prob = dist.log_prob(actions_t)
                        entropy = dist.entropy()
                        ratio = torch.exp(log_prob - old_lp_t)
                        mask_sum = mask_t.sum().clamp(min=1.0)

                        with torch.no_grad():
                            kl_per = ((ratio - 1) - (log_prob - old_lp_t)) * mask_t
                            approx_kl = (kl_per.sum() / mask_sum).item()
                            total_kl += approx_kl; epoch_kl_sum += approx_kl; epoch_kl_n += 1

                        surr1 = ratio * advs_t
                        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs_t
                        per_step_p = -torch.min(surr1, surr2)
                        p_loss = (per_step_p * mask_t).sum() / mask_sum

                        target_t = torch.clamp(returns_t, -self.value_clamp, self.value_clamp)
                        per_step_v = (values_seq - target_t).pow(2)
                        v_loss = (per_step_v * mask_t).sum() / mask_sum

                        ent_loss = (entropy * mask_t).sum() / mask_sum
                        loss = p_loss + self.value_coef * v_loss - self.entropy_coef * ent_loss

                        if use_junction_aux:
                            j_logits = self.model.junction_head(enc_feats_flat)
                            obs_flat = obs_t.reshape(T_chunk * B, *obs_t.shape[2:])
                            j_vals = obs_flat[:, self._junction_ch_idx, self._obs_center, self._obs_center]
                            j_class = torch.zeros(T_chunk * B, dtype=torch.long, device=self.device)
                            j_class[j_vals > 0.7] = 2
                            j_class[(j_vals > 0.3) & (j_vals <= 0.7)] = 1
                            mask_flat = mask_t.reshape(-1)
                            j_loss_per = nn.functional.cross_entropy(j_logits, j_class, reduction="none")
                            j_loss = (j_loss_per * mask_flat).sum() / mask_sum
                            loss = loss + 0.1 * j_loss

                    # Fast zero_grad
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    gn_val = grad_norm.item()
                    total_gn += gn_val
                    if gn_val > max_gn: max_gn = gn_val
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    # STANDARD FALLBACK (CPU / No-AMP)
                    fwd_result = self.model.forward_sequence(obs_t, init_state, dones_t, return_enc_features=use_junction_aux)
                    if use_junction_aux: logits_seq, values_seq, enc_feats_flat = fwd_result
                    else: logits_seq, values_seq = fwd_result

                    dist = torch.distributions.Categorical(logits=logits_seq)
                    log_prob = dist.log_prob(actions_t)
                    entropy = dist.entropy()
                    ratio = torch.exp(log_prob - old_lp_t)
                    mask_sum = mask_t.sum().clamp(min=1.0)

                    with torch.no_grad():
                        kl_per = ((ratio - 1) - (log_prob - old_lp_t)) * mask_t
                        approx_kl = (kl_per.sum() / mask_sum).item()
                        total_kl += approx_kl; epoch_kl_sum += approx_kl; epoch_kl_n += 1

                    surr1 = ratio * advs_t
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs_t
                    per_step_p = -torch.min(surr1, surr2)
                    p_loss = (per_step_p * mask_t).sum() / mask_sum

                    target_t = torch.clamp(returns_t, -self.value_clamp, self.value_clamp)
                    per_step_v = (values_seq - target_t).pow(2)
                    v_loss = (per_step_v * mask_t).sum() / mask_sum

                    ent_loss = (entropy * mask_t).sum() / mask_sum
                    loss = p_loss + self.value_coef * v_loss - self.entropy_coef * ent_loss

                    if use_junction_aux:
                        j_logits = self.model.junction_head(enc_feats_flat)
                        obs_flat = obs_t.reshape(T_chunk * B, *obs_t.shape[2:])
                        j_vals = obs_flat[:, self._junction_ch_idx, self._obs_center, self._obs_center]
                        j_class = torch.zeros(T_chunk * B, dtype=torch.long, device=self.device)
                        j_class[j_vals > 0.7] = 2; j_class[(j_vals > 0.3) & (j_vals <= 0.7)] = 1
                        mask_flat = mask_t.reshape(-1)
                        j_loss_per = nn.functional.cross_entropy(j_logits, j_class, reduction="none")
                        j_loss = (j_loss_per * mask_flat).sum() / mask_sum
                        loss = loss + 0.1 * j_loss

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    gn_val = grad_norm.item()
                    total_gn += gn_val
                    if gn_val > max_gn: max_gn = gn_val
                    self.optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += ent_loss.item()
                n_updates += 1

            epochs_run += 1

            if self.target_kl is not None and epoch_kl_n > 0:
                epoch_kl_mean = epoch_kl_sum / epoch_kl_n
                if epoch_kl_mean > self.target_kl: break

        with torch.no_grad():
            all_returns = torch.cat([r for r in per_buf_returns if r is not None])
            values_all = torch.tensor(all_values_raw, dtype=torch.float32)
            ev = (1 - (all_returns.cpu() - values_all).var() / (all_returns.cpu().var() + 1e-8)).item()

        return {
            "policy_loss": total_p / max(n_updates, 1),
            "value_loss": total_v / max(n_updates, 1),
            "entropy": total_e / max(n_updates, 1),
            "approx_kl": total_kl / max(n_updates, 1),
            "grad_norm": total_gn / max(n_updates, 1),
            "grad_norm_max": max_gn,
            "epochs_run": epochs_run,
            "explained_variance": ev,
        }

    # ------------------------------------------------------------------
    # Dispatch & Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ep_length_weights(dones: list) -> np.ndarray:
        dones_arr = np.asarray(dones, dtype=np.float32)
        n = len(dones_arr)
        if n == 0: return np.ones(0, dtype=np.float32)

        weights = np.ones(n, dtype=np.float32)
        ep_lengths, start = [], 0
        for t in range(n):
            if dones_arr[t] > 0 or t == n - 1:
                ep_len = t - start + 1
                ep_lengths.append(ep_len)
                weights[start:t + 1] = float(ep_len)
                start = t + 1

        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 1.0
        return np.sqrt(np.maximum(weights / max(mean_len, 1.0), 0.0)).astype(np.float32)

    def _ppo_update(self, buffers: List[RolloutBuffer], last_values: List[float]) -> Dict[str, float]:
        if self.use_lstm: return self._ppo_update_lstm(buffers, last_values)
        return self._ppo_update_ff(buffers, last_values)

    def load_checkpoint(self, save_path: str, imitation_path: str) -> Tuple[int, float]:
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                try: self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except: print("  Scheduler state incompatible, starting fresh.")

            if "curriculum_stage" in ckpt:
                self.curriculum.current_stage_idx = ckpt["curriculum_stage"]
                print(f"  Restored curriculum stage: {self.curriculum.get_current_stage().name}")

            start = ckpt.get("iteration", 0) + 1
            best = ckpt.get("best_cldice", ckpt.get("best_f1", 0.0))
            print(f"Resumed from PPO checkpoint  iter={start-1}  best_clDice={best:.3f}")
            return start, best

        if os.path.exists(imitation_path):
            ckpt = torch.load(imitation_path, map_location=self.device, weights_only=True)
            state = dict(ckpt["model_state_dict"])
            model_state = self.model.state_dict()
            stripped = []
            for k in list(state.keys()):
                if k in model_state and state[k].shape != model_state[k].shape:
                    stripped.append((k, tuple(state[k].shape), tuple(model_state[k].shape)))
                    del state[k]
            self.model.load_state_dict(state, strict=False)
            print(f"Loaded imitation weights  val_acc={ckpt.get('val_acc', 0):.3f}")
            
            for layer in self.model.value_head:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)
            print("Value head re-initialized.")
            
            actor_last_keys = {"actor_head.3.weight", "actor_head.3.bias"}
            if any(k in actor_last_keys for k, _, _ in stripped):
                last = self.model.actor_head[-1]
                nn.init.orthogonal_(last.weight, gain=0.01)
                nn.init.zeros_(last.bias)
                if self.model.N_ACTIONS == 9:
                    with torch.no_grad(): last.bias[8] = -1.0
                print(f"Actor head last layer re-initialised for N_ACTIONS={self.model.N_ACTIONS}.")
            return 1, 0.0

        print("WARNING: No weights found, training from scratch.")
        return 1, 0.0

    def _pick_sample_index(self, train_samples) -> int:
        difficulty = self.curriculum.get_difficulty()
        if not hasattr(self, "_sample_difficulties"):
            self._sample_difficulties = []
            for s in train_samples:
                self._sample_difficulties.append(self.curriculum.compute_sample_difficulty(s["centerline"], s.get("vessel_mask", s["centerline"])))
        valid = [i for i, d in enumerate(self._sample_difficulties) if d <= difficulty]
        if len(valid) < 10: valid = list(range(min(10, len(train_samples))))
        return int(np.random.choice(valid))

    def _collect_rollout_vec(self, vec_env, buffers, train_samples, obs_list, lstm_states_list, ep_rewards, ep_lengths, episode_rewards, episode_lengths, current_sample_ids, accumulated_coverage):
        n_envs = vec_env.n_envs
        steps_collected = 0

        while steps_collected < self.steps_per_iter:
            obs_batch = torch.from_numpy(np.stack(obs_list)).float().to(self.device)
            if self.use_lstm:
                h_cat = torch.cat([s[0].to(self.device) for s in lstm_states_list], dim=0)
                h_c_cat = torch.cat([s[1].to(self.device) for s in lstm_states_list], dim=0)
                batched_lstm = (h_cat, h_c_cat)
            else:
                batched_lstm = None

            # [SPEEDUP]: Evaluated in Autocast during Collection
            with torch.no_grad():
                with torch.amp.autocast("cuda") if self.device.type == "cuda" else torch.autocast("cpu", enabled=False):
                    actions, log_probs, _, values, new_lstm = self.model.get_action_and_value(obs_batch, batched_lstm)

            action_list = [actions[i].item() for i in range(n_envs)]
            all_obs, all_rewards, all_terminated, all_truncated, all_infos = vec_env.step(action_list)

            for i in range(n_envs):
                action_i, log_prob_i, value_i = action_list[i], log_probs[i].item(), values[i].item()
                next_obs, reward, done = all_obs[i], all_rewards[i], all_terminated[i] or all_truncated[i]
                lstm_state_i = (lstm_states_list[i][0], lstm_states_list[i][1]) if self.use_lstm else None

                if done:
                    self.terminal_reward_normalizer.update(reward)
                    norm_reward = self.terminal_reward_normalizer.normalize(reward)
                else:
                    self.reward_normalizer.update(reward)
                    norm_reward = self.reward_normalizer.normalize(reward)

                buffers[i].add(obs_list[i], action_i, log_prob_i, norm_reward, value_i, float(done), lstm_state_i)

                ep_rewards[i] += reward; ep_lengths[i] += 1; steps_collected += 1
                info_i = all_infos[i] if isinstance(all_infos, (list, tuple)) else {}
                if isinstance(info_i, dict):
                    for _key in RewardCalculator.BREAKDOWN_KEYS:
                        _val = info_i.get(_key)
                        if _val is not None:
                            self._rwrd_sums[_key] = self._rwrd_sums.get(_key, 0.0) + _val
                            self._rwrd_counts[_key] = self._rwrd_counts.get(_key, 0) + 1

                if self.use_lstm: lstm_states_list[i] = (new_lstm[0][i:i+1, :].detach().cpu(), new_lstm[1][i:i+1, :].detach().cpu())

                if done:
                    episode_rewards.append(ep_rewards[i])
                    episode_lengths.append(ep_lengths[i])
                    self.curriculum.step(success=self.curriculum.is_episode_successful(all_infos[i]))
                    ep_rewards[i] = ep_lengths[i] = 0

                    if self.use_lstm:
                        fresh = self.model.init_hidden(batch_size=1, device=self.device)
                        lstm_states_list[i] = (fresh[0].detach().cpu(), fresh[1].detach().cpu())

                    finished_sample_id = current_sample_ids[i]
                    cov_mask = vec_env.get_coverage_mask(i)
                    if cov_mask is not None and finished_sample_id >= 0:
                        prev = accumulated_coverage.get(finished_sample_id)
                        if prev is None: accumulated_coverage[finished_sample_id] = (cov_mask > 0).astype(np.float32)
                        else: accumulated_coverage[finished_sample_id] = np.where(cov_mask > 0, 1.0, prev)
                        if len(accumulated_coverage) > 512: accumulated_coverage.pop(next(iter(accumulated_coverage)))

                    sample_idx = self._pick_sample_index(train_samples)
                    current_sample_ids[i] = sample_idx
                    vec_env.set_sample(i, sample_idx, prior_coverage=accumulated_coverage.get(sample_idx))
                    
                    prev_stage = self.curriculum.current_stage_idx
                    vec_env.apply_overrides(i, self._get_curriculum_overrides_dict())
                    next_obs = vec_env.reset(i)

                    if self.curriculum.current_stage_idx != prev_stage:
                        print(f"  → Curriculum stage: {self.curriculum.get_current_stage().name}")
                        new_overrides = self._get_curriculum_overrides_dict()
                        for j in range(n_envs): vec_env.apply_overrides(j, new_overrides)

                obs_list[i] = next_obs

        obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if self.use_lstm:
                h_cat = torch.cat([s[0] for s in lstm_states_list], dim=0).to(self.device)
                h_c_cat = torch.cat([s[1] for s in lstm_states_list], dim=0).to(self.device)
                last_values = self.model.get_value(obs_batch, (h_cat, h_c_cat))
            else:
                last_values = self.model.get_value(obs_batch, None)
        return [last_values[i].item() for i in range(n_envs)]

    def train(self, train_samples, val_samples, save_path: str, log_path: str, imitation_path: str = "") -> None:
        from environment.vec_env import SubprocVecEnv

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        start_iteration, best_cldice = self.load_checkpoint(save_path, imitation_path)

        N_ENVS = self.config.get("training", {}).get("ppo", {}).get("n_envs", 8)
        vec_env = SubprocVecEnv(self.config, n_envs=N_ENVS)

        current_sample_ids = [-1] * N_ENVS
        accumulated_coverage = {}
        obs_list, lstm_states_list = [], []
        ep_rewards, ep_lengths = [0.0] * N_ENVS, [0] * N_ENVS

        overrides = self._get_curriculum_overrides_dict()
        for i in range(N_ENVS):
            sample_idx = self._pick_sample_index(train_samples)
            current_sample_ids[i] = sample_idx
            vec_env.set_sample(i, sample_idx)
            vec_env.apply_overrides(i, overrides)
            obs_list.append(vec_env.reset(i))
            hidden = self.model.init_hidden(batch_size=1, device=self.device)
            lstm_states_list.append(tuple(t.detach().cpu() for t in hidden) if hidden is not None else None)

        buffers = [RolloutBuffer() for _ in range(N_ENVS)]
        episode_rewards, episode_lengths = deque(maxlen=50), deque(maxlen=50)

        _csv_fields = ["iteration", "mean_reward", "mean_ep_length", "policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance", "grad_norm", "lr", "stage"] + list(RewardCalculator.BREAKDOWN_KEYS) + ["val_coverage", "val_f1", "val_cldice"]
        _csv_file = open(log_path, "w", newline="", encoding="utf-8")
        _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields, extrasaction="ignore")
        _csv_writer.writeheader()

        print(f"Starting curriculum stage: {self.curriculum.get_current_stage().name}")
        print(f"\nStarting PPO — iters {start_iteration}–{self.num_iterations} × {self.steps_per_iter} steps  {N_ENVS} envs  LSTM={'ON chunk_len=' + str(self.lstm_chunk_length) if self.use_lstm else 'OFF'}\n")

        self._last_stage_idx = self.curriculum.current_stage_idx

        for iteration in range(start_iteration, self.num_iterations + 1):
            if self.curriculum.current_stage_idx != self._last_stage_idx:
                self._stage_iter = 0
                self._last_stage_idx = self.curriculum.current_stage_idx
                self._entropy_frozen = False
                self._eval_cldice_window.clear()
            self._stage_iter += 1

            for buf in buffers: buf.reset()
            self._rwrd_sums, self._rwrd_counts = {}, {}
            
            self.model.eval()
            last_values = self._collect_rollout_vec(vec_env, buffers, train_samples, obs_list, lstm_states_list, ep_rewards, ep_lengths, episode_rewards, episode_lengths, current_sample_ids, accumulated_coverage)

            self.model.train()
            stats = self._ppo_update(buffers, last_values)
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
            log = f"Iter {iteration:4d}/{self.num_iterations}  reward={mean_reward:7.3f}  ep_len={mean_length:6.1f}  p_loss={stats['policy_loss']:7.4f}  v_loss={stats['value_loss']:6.4f}  entropy={stats['entropy']:.3f}"

            if iteration % self.eval_every == 0 and val_samples:
                ev = evaluate(self.model, val_samples, self.config, self.device, self.tolerance, 
                n_episodes=4,
                n_parallel=16
                )
                stage = self.curriculum.get_current_stage()
                log += f"  |  val_cov={ev['mean_coverage']:.3f}  val_f1={ev['mean_f1']:.3f}  val_clDice={ev['mean_cldice']:.3f}  stage={stage.name}  ent_c={self.entropy_coef:.3f}"

                if ev["mean_cldice"] > best_cldice:
                    best_cldice = ev["mean_cldice"]
                    self.no_improve_count = 0
                    torch.save({"iteration": iteration, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.scheduler.state_dict(), "best_cldice": best_cldice, "config": self.config, "curriculum_stage": self.curriculum.current_stage_idx}, save_path)
                    log += f"  ✓ saved (best clDice={best_cldice:.3f})"
                else:
                    self.no_improve_count += 1

                self._eval_cldice_window.append(ev["mean_cldice"])
                if len(self._eval_cldice_window) >= 2:
                    recent_improvement = max(self._eval_cldice_window) - min(self._eval_cldice_window)
                    was_frozen = self._entropy_frozen
                    self._entropy_frozen = recent_improvement < 0.005
                    if self._entropy_frozen and not was_frozen: log += f"  [entropy frozen @ {self.entropy_coef:.4f}]"
                    elif not self._entropy_frozen and was_frozen: log += "  [entropy unfrozen]"
                if self.no_improve_count >= self.patience:
                    print(log)
                    print(f"\nEarly stopping: no improvement for {self.patience} eval cycles.")
                    break

            print(log)
            _csv_row = {"iteration": iteration, "mean_reward": mean_reward, "mean_ep_length": mean_length, "policy_loss": stats["policy_loss"], "value_loss": stats["value_loss"], "entropy": stats["entropy"], "approx_kl": stats["approx_kl"], "explained_variance": stats["explained_variance"], "grad_norm": stats["grad_norm"], "lr": current_lr, "stage": self.curriculum.get_current_stage().name}
            for _k in RewardCalculator.BREAKDOWN_KEYS: _csv_row[_k] = self._rwrd_sums.get(_k, 0.0) / max(self._rwrd_counts.get(_k, 1), 1)
            if iteration % self.eval_every == 0 and val_samples and "ev" in dir(): _csv_row.update({"val_coverage": ev["mean_coverage"], "val_f1": ev["mean_f1"], "val_cldice": ev["mean_cldice"]})
            _csv_writer.writerow(_csv_row)
            _csv_file.flush()

            if iteration % self.save_every == 0:
                ckpt_path = save_path.replace(".pt", f"_iter{iteration}.pt")
                torch.save({"iteration": iteration, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.scheduler.state_dict(), "config": self.config}, ckpt_path)

        vec_env.close()
        _csv_file.close()

        print(f"\nDone. Best clDice: {best_cldice:.3f}")
        print(f"Weights: {save_path}")
