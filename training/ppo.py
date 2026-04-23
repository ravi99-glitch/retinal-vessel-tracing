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
        # Do NOT normalise per-buffer here.  Normalising individually corrupts the
        # relative advantage ordering across environments before the global
        # re-normalisation in _ppo_update_ff / _ppo_update_lstm.  A single global
        # norm (applied there after concatenating all buffers) is both correct and
        # consistent with the standard PPO implementation.
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
    """Run n greedy episodes per val sample with batched GPU inference.

    Runs up to n_parallel envs simultaneously in the main process,
    batching their observations into a single GPU forward pass per step.
    LSTM hidden states are managed per-slot when use_lstm=True.

    Returns mean_coverage and mean_f1.
    """
    from data.centerline_extraction import compute_centerline_f1
    from environment.vessel_env import VesselTracingEnv

    model.eval()
    use_lstm = getattr(model, "use_lstm", False)
    coverages = []
    f1_scores = []
    cldice_scores = []
    # Accumulate covered_centerline across all episodes for the same sample.
    # Single-episode clDice ≈ 2×coverage_ratio ≈ 0.06 and never improves even
    # as the policy learns — it is a monitoring artifact, not a real metric.
    # Multi-episode union coverage gives the real clDice signal.
    per_sample_coverage: Dict[int, np.ndarray] = {}
    per_sample_vessel: Dict[int, np.ndarray] = {}

    # Build work queue: (sample, start_position) pairs
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

    # Slot arrays
    n_slots = min(n_parallel, len(work_queue))
    envs: List[Optional[object]] = [None] * n_slots
    samples_ref: List[Optional[Dict]] = [None] * n_slots
    obs_list: List[Optional[np.ndarray]] = [None] * n_slots
    lstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_slots
    active = [False] * n_slots

    def _start_slot(slot, sample, start_pos):
        env = VesselTracingEnv(config)
        env.set_data(
            image=sample["image"],
            centerline=sample["centerline"],
            distance_transform=sample["distance_transform"],
            fov_mask=sample["fov_mask"],
            vessel_orientation=sample.get("vessel_orientation"),
            dt_gradient=sample.get("dt_gradient"),
            vesselness=sample.get("vesselness"),
            unet_prior=sample.get("unet_prior"),
        )
        obs, _ = env.reset(start_position=start_pos)
        envs[slot] = env
        samples_ref[slot] = sample
        obs_list[slot] = obs
        lstm_states[slot] = model.init_hidden(batch_size=1, device=device)
        active[slot] = True

    # Fill initial slots
    for i in range(n_slots):
        if work_queue:
            sample, start = work_queue.popleft()
            _start_slot(i, sample, start)

    with torch.no_grad():
        while any(active):
            active_idx = [i for i in range(n_slots) if active[i]]
            if not active_idx:
                break

            # Batch observations
            obs_batch = torch.from_numpy(
                np.stack([obs_list[i] for i in active_idx])
            ).float().to(device)

            # Batch LSTM states if needed
            if use_lstm:
                h_cat = torch.cat(
                    [lstm_states[i][0].to(device) for i in active_idx], dim=0
                )
                c_cat = torch.cat(
                    [lstm_states[i][1].to(device) for i in active_idx], dim=0
                )
                batched_lstm = (h_cat, c_cat)
            else:
                batched_lstm = None

            logits, _, new_lstm = model(obs_batch, batched_lstm)
            actions = logits.argmax(dim=-1)  # greedy

            # Step each active env
            for j, i in enumerate(active_idx):
                obs, _, terminated, truncated, info = envs[i].step(actions[j].item())
                obs_list[i] = obs

                # Update LSTM state for this slot
                if use_lstm and new_lstm is not None:
                    lstm_states[i] = (
                        new_lstm[0][j:j+1, :].detach().cpu(),
                        new_lstm[1][j:j+1, :].detach().cpu(),
                    )

                if terminated or truncated:
                    coverages.append(info["coverage_ratio"])
                    metrics = compute_centerline_f1(
                        envs[i].covered_centerline,
                        samples_ref[i]["centerline"],
                        tolerance=tolerance,
                    )
                    f1_scores.append(metrics["f1"])

                    # Accumulate coverage per sample; compute clDice after all
                    # episodes finish.  Per-episode clDice ≈ 2×coverage_ratio
                    # and is always ~0.06 regardless of policy quality.
                    cov = envs[i].covered_centerline
                    if cov is not None:
                        s_key = id(samples_ref[i])
                        if s_key not in per_sample_coverage:
                            per_sample_coverage[s_key] = (cov > 0).astype(np.float32)
                            per_sample_vessel[s_key] = samples_ref[i].get(
                                "vessel_mask", samples_ref[i]["centerline"]
                            )
                        else:
                            per_sample_coverage[s_key] = np.where(
                                cov > 0, 1.0, per_sample_coverage[s_key]
                            )

                    # Refill slot from queue
                    if work_queue:
                        sample, start = work_queue.popleft()
                        _start_slot(i, sample, start)
                    else:
                        active[i] = False

    # Compute clDice on the union of coverage across all episodes per sample.
    # This reflects true multi-episode performance (what the full inference
    # pipeline achieves), not the misleading single-episode artifact.
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
    """Track running mean/std of rewards for normalisation.

    Stabilises training when reward scale changes across curriculum stages.
    """

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
        self.value_clamp = config.get("training", {}).get("value_clamp", 10.0)

        # Adaptive KL early-stopping target. None / 0 disables the check.
        ppo_cfg = config.get("training", {}).get("ppo", {})
        target_kl = ppo_cfg.get("target_kl", None)
        self.target_kl: Optional[float] = (
            float(target_kl) if target_kl else None
        )

        # Per-stage iteration counter for entropy annealing inside a
        # curriculum stage. Reset whenever the curriculum advances.
        self._stage_iter: int = 0
        # Rolling window of eval clDice scores — used to gate entropy annealing.
        # Annealing only advances when the policy is actively improving.
        self._eval_cldice_window: deque = deque(maxlen=3)
        self._entropy_frozen: bool = False
        self._last_stage_idx: int = 0

        # Potential-based shaping requires shaping_gamma == ppo.gamma to
        # preserve the optimal policy (Ng et al. 1999).
        shaping_gamma = config.get("reward", {}).get("shaping_gamma", gamma)
        if abs(shaping_gamma - gamma) > 1e-6:
            raise ValueError(
                f"reward.shaping_gamma ({shaping_gamma}) must equal "
                f"training.ppo.gamma ({gamma}) for potential-based "
                f"shaping to be policy-invariant."
            )

        # Junction auxiliary loss: index of the junction channel in obs tensors.
        # Used to extract per-step supervision targets for the junction aux head
        # without re-running the environment.
        from models.policy_network import _junction_channel_idx
        self._junction_ch_idx: Optional[int] = _junction_channel_idx(config)
        # Half-size of the observation patch — center pixel index for GT lookup.
        obs_size = config.get("environment", {}).get("observation_size", 65)
        self._obs_center: int = obs_size // 2

        # curriculum manager
        self.curriculum = CurriculumManager(config)
        self.reward_normalizer = RunningRewardNormalizer(
            clip=config.get("training", {}).get("reward_norm_clip", 10.0)
        )
        # Separate normalizer for terminal rewards (fired once per episode).
        # The step-reward normalizer is dominated by dense per-step signals
        # (scale ~10–100 per episode); mixing terminal rewards (scale 0–5)
        # into the same running statistics compresses the topology signal to
        # near zero after normalisation.  Using a dedicated normalizer with a
        # tighter clip preserves the relative magnitude of the terminal bonus.
        # See: Peng et al. (2018) DeepMimic; Schulman et al. (2017) PPO.
        self.terminal_reward_normalizer = RunningRewardNormalizer(
            clip=config.get("training", {}).get("terminal_norm_clip", 5.0)
        )

        # early stopping patience
        self.patience = config.get("training", {}).get("patience", 100)
        self.no_improve_count = 0

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # LR schedule: optional linear warmup then linear decay.
        lr_end_factor = config.get("training", {}).get("lr_end_factor", 0.1)
        warmup_iters = ppo_cfg.get("lr_warmup_iters", 0)

        if warmup_iters > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_iters,
            )
            decay_sched = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=lr_end_factor,
                total_iters=max(num_iterations - warmup_iters, 1),
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, decay_sched],
                milestones=[warmup_iters],
            )
        else:
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=lr_end_factor,
                total_iters=num_iterations,
            )


    def _get_curriculum_overrides_dict(self) -> dict:
        """Return current curriculum overrides as a flat dict for SubprocVecEnv.
        Also applies training-side overrides (entropy_coef) directly on self.

        Entropy coefficient is linearly annealed inside a stage when the
        stage defines ``entropy_coef_end`` and ``entropy_anneal_iters``.
        Annealing progress is driven by ``self._stage_iter``, which the
        outer training loop ticks once per PPO iteration.
        """
        overrides = self.curriculum.get_stage_overrides()
        result = {}
        env_ov = overrides.get("environment", {})
        if "max_off_track_streak" in env_ov:
            result["max_off_track"] = env_ov["max_off_track_streak"]
        if "max_steps_per_episode" in env_ov:
            result["max_steps"] = env_ov["max_steps_per_episode"]
        if "off_track_penalty_ramp" in env_ov:
            result["off_track_ramp"] = env_ov["off_track_penalty_ramp"]
        reward_ov = overrides.get("reward", {})
        if "smoothness_weight" in reward_ov:
            result["smoothness_weight"] = reward_ov["smoothness_weight"]

        # Training overrides (entropy coef) — applied on trainer, not env.
        # Linearly anneal toward ``entropy_coef_end`` over the configured
        # number of iterations spent inside the stage.
        train_ov = overrides.get("training", {})
        if "entropy_coef" in train_ov:
            stage = self.curriculum.get_current_stage()
            ec_start = float(train_ov["entropy_coef"])
            ec_end = getattr(stage, "entropy_coef_end", None)
            ec_iters = getattr(stage, "entropy_anneal_iters", 0) or 0
            if ec_end is not None and ec_iters > 0:
                # Performance-gated annealing: only advance the timer when
                # val_clDice is improving. When it plateaus, freeze entropy
                # at its current value so exploration is preserved.
                if not self._entropy_frozen:
                    t = min(self._stage_iter, ec_iters) / float(ec_iters)
                    self.entropy_coef = ec_start + t * (float(ec_end) - ec_start)
                # If frozen, leave self.entropy_coef unchanged
            else:
                self.entropy_coef = ec_start

        return result

    # ------------------------------------------------------------------
    # PPO update — feedforward (original path, unchanged logic)
    # ------------------------------------------------------------------

    def _ppo_update_ff(
        self, buffers: List[RolloutBuffer], last_values: List[float]
    ) -> Dict[str, float]:
        """Standard feedforward PPO update with random mini-batches.
        
        Accepts per-environment buffers and last values so that GAE
        is computed independently per environment (correct bootstrapping).
        Tensors are then concatenated for standard random-shuffle training.
        """
        # Compute GAE per-env, then concatenate
        all_returns, all_advantages = [], []
        all_obs, all_actions, all_old_log_probs = [], [], []
        all_values_raw = []  # for explained variance

        for buf, lv in zip(buffers, last_values):
            if len(buf.rewards) == 0:
                continue
            ret, adv = buf.compute_returns_and_advantages(
                lv, self.gamma, self.gae_lambda
            )
            # Episode-length weighting: amplify advantages from long episodes
            # relative to short ones before global normalization so the
            # optimizer does not over-weight frequent short-episode gradients.
            ep_w = torch.from_numpy(self._ep_length_weights(buf.dones))
            adv = adv * ep_w

            obs, actions, log_probs = buf.get_tensors()
            all_returns.append(ret)
            all_advantages.append(adv)
            all_obs.append(obs)
            all_actions.append(actions)
            all_old_log_probs.append(log_probs)
            all_values_raw.extend(buf.values)

        returns = torch.cat(all_returns).to(self.device)
        # Re-normalize advantages across the full dataset after concat
        advantages = torch.cat(all_advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)
        obs = torch.cat(all_obs).to(self.device)
        actions = torch.cat(all_actions).to(self.device)
        old_log_probs = torch.cat(all_old_log_probs).to(self.device)

        total_p, total_v, total_e, total_kl, total_gn, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        max_gn = 0.0
        epochs_run = 0
        dataset_size = len(obs)

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(dataset_size)
            epoch_kl_sum = 0.0
            epoch_kl_n = 0
            for start in range(0, dataset_size, self.mini_batch_size):
                idx = perm[start : start + self.mini_batch_size]

                logits, values, _ = self.model(obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_prob - old_log_probs[idx])

                with torch.no_grad():
                    approx_kl = (
                        ((ratio - 1) - (log_prob - old_log_probs[idx])).mean().item()
                    )
                    total_kl += approx_kl
                    epoch_kl_sum += approx_kl
                    epoch_kl_n += 1

                surr1 = ratio * advantages[idx]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * advantages[idx]
                )
                p_loss = -torch.min(surr1, surr2).mean()

                # Clamp targets only, never the value-head output. Clamping
                # `values` zeroes the gradient whenever the network predicts
                # outside [-value_clamp, value_clamp], which can stall the
                # critic. Returns are already bounded because rewards pass
                # through RunningRewardNormalizer (clip ±10).
                v_loss = nn.functional.mse_loss(
                    values,
                    torch.clamp(returns[idx], -self.value_clamp, self.value_clamp),
                )

                loss = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                # Junction auxiliary loss — forces encoder to build junction-
                # discriminative features via supervised multi-class prediction.
                # GT class derived from the center pixel of the junction-map
                # channel already present in the stored observation batch.
                if (
                    self.model.junction_head is not None
                    and self._junction_ch_idx is not None
                ):
                    enc_feats = self.model.encode(obs[idx])  # (B, hidden_dim)
                    j_logits = self.model.junction_head(enc_feats)  # (B, 3)
                    j_vals = obs[idx][:, self._junction_ch_idx,
                                      self._obs_center, self._obs_center]  # (B,)
                    j_class = torch.zeros(len(idx), dtype=torch.long, device=self.device)
                    j_class[j_vals > 0.7] = 2   # junction  (~1.0)
                    j_class[(j_vals > 0.3) & (j_vals <= 0.7)] = 1  # endpoint (~0.5)
                    j_loss = nn.functional.cross_entropy(j_logits, j_class)
                    loss = loss + 0.1 * j_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                gn_val = grad_norm.item()
                total_gn += gn_val
                if gn_val > max_gn:
                    max_gn = gn_val
                self.optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += entropy.item()
                n += 1

            epochs_run += 1

            # Adaptive KL early-stopping: if the running approx-KL of this
            # epoch exceeded the target, stop further passes over the same
            # rollout (the policy has already moved enough that additional
            # gradient steps would train on stale data).
            if self.target_kl is not None and epoch_kl_n > 0:
                epoch_kl_mean = epoch_kl_sum / epoch_kl_n
                if epoch_kl_mean > self.target_kl:
                    break

        with torch.no_grad():
            values_all = torch.tensor(all_values_raw, dtype=torch.float32)
            ev = (
                1 - (returns.cpu() - values_all).var() / (returns.cpu().var() + 1e-8)
            ).item()

        return {
            "policy_loss": total_p / max(n, 1),
            "value_loss": total_v / max(n, 1),
            "entropy": total_e / max(n, 1),
            "approx_kl": total_kl / max(n, 1),
            "grad_norm": total_gn / max(n, 1),
            "grad_norm_max": max_gn,
            "epochs_run": epochs_run,
            "explained_variance": ev,
        }

    # ------------------------------------------------------------------
    # PPO update — recurrent (sequential chunks)
    # ------------------------------------------------------------------

    def _ppo_update_lstm(
        self, buffers: List[RolloutBuffer], last_values: List[float]
    ) -> Dict[str, float]:
        """Recurrent PPO update using batched fixed-length chunks.

        Each chunk is a contiguous slice of one buffer with length =
        ``lstm_chunk_length`` (right-padded with zeros if shorter). Multiple
        chunks are stacked along the batch dim so each optimizer step sees
        ``mini_batch_size`` transitions, matching the feedforward path.

        Steps:
        1. Compute GAE per buffer; concatenate advantages and re-normalize
           globally so the loss scale matches ``_ppo_update_ff``.
        2. Build fixed-length chunks with a valid-mask, dropping chunks with
           fewer than 2 valid steps.
        3. Shuffle chunks per epoch and process them in groups of
           ``chunks_per_batch`` so each optimizer step has ~mini_batch_size
           valid transitions.
        4. ``forward_sequence`` accepts the (T, B) layout natively; the
           per-step loss is averaged using the valid mask so padded steps
           contribute zero gradient.
        """
        T_chunk = self.lstm_chunk_length

        # ---- 1. GAE per buffer + global advantage normalization ----
        per_buf_returns: List[Optional[torch.Tensor]] = []
        per_buf_advantages: List[Optional[torch.Tensor]] = []
        all_values_raw: List[float] = []

        for buf, lv in zip(buffers, last_values):
            if len(buf.rewards) == 0:
                per_buf_returns.append(None)
                per_buf_advantages.append(None)
                continue
            ret, adv = buf.compute_returns_and_advantages(
                lv, self.gamma, self.gae_lambda
            )
            # Episode-length weighting before global normalization
            ep_w = torch.from_numpy(self._ep_length_weights(buf.dones))
            adv = adv * ep_w

            per_buf_returns.append(ret)
            per_buf_advantages.append(adv)
            all_values_raw.extend(buf.values)

        # Global re-normalization across all transitions, matching FF path
        non_empty = [a for a in per_buf_advantages if a is not None]
        if non_empty:
            cat = torch.cat(non_empty)
            mean = cat.mean()
            std = cat.std() + 1e-8
            per_buf_advantages = [
                None if a is None else (a - mean) / std
                for a in per_buf_advantages
            ]

        # ---- 2. Build fixed-length chunks with a valid mask ----
        # Each entry: (buf_idx, start, valid_len)
        chunk_specs: List[Tuple[int, int, int]] = []
        for buf_idx, buf in enumerate(buffers):
            T = len(buf.obs)
            for s in range(0, T, T_chunk):
                valid = min(T_chunk, T - s)
                if valid >= 2:
                    chunk_specs.append((buf_idx, s, valid))

        if not chunk_specs:
            return {
                "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "approx_kl": 0.0, "grad_norm": 0.0, "grad_norm_max": 0.0,
                "epochs_run": 0, "explained_variance": 0.0,
            }

        chunks_per_batch = max(1, self.mini_batch_size // T_chunk)
        total_p = total_v = total_e = total_kl = total_gn = 0.0
        max_gn = 0.0
        n_updates = 0
        epochs_run = 0

        # Pre-allocate per-chunk numpy buffers we'll fill in the loop.
        obs_shape = buffers[0].obs[0].shape  # (C, H, W)
        C, H, W = obs_shape

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(len(chunk_specs)).tolist()
            epoch_kl_sum = 0.0
            epoch_kl_n = 0

            for batch_start in range(0, len(perm), chunks_per_batch):
                batch_idx = perm[batch_start:batch_start + chunks_per_batch]
                B = len(batch_idx)

                # Pre-allocate (T, B, ...) tensors. Padding values are zero;
                # the valid mask zeros their loss contribution.
                obs_np = np.zeros((T_chunk, B, C, H, W), dtype=np.float32)
                actions_np = np.zeros((T_chunk, B), dtype=np.int64)
                old_lp_np = np.zeros((T_chunk, B), dtype=np.float32)
                returns_np = np.zeros((T_chunk, B), dtype=np.float32)
                advs_np = np.zeros((T_chunk, B), dtype=np.float32)
                dones_np = np.zeros((T_chunk, B), dtype=np.float32)
                mask_np = np.zeros((T_chunk, B), dtype=np.float32)

                init_h_list: List[torch.Tensor] = []
                init_c_list: List[torch.Tensor] = []

                for j, ci in enumerate(batch_idx):
                    buf_idx, s, valid = chunk_specs[ci]
                    buf = buffers[buf_idx]

                    obs_np[:valid, j] = np.asarray(
                        buf.obs[s:s + valid], dtype=np.float32
                    )
                    actions_np[:valid, j] = np.asarray(
                        buf.actions[s:s + valid], dtype=np.int64
                    )
                    old_lp_np[:valid, j] = np.asarray(
                        buf.log_probs[s:s + valid], dtype=np.float32
                    )
                    returns_np[:valid, j] = (
                        per_buf_returns[buf_idx][s:s + valid].numpy()
                    )
                    advs_np[:valid, j] = (
                        per_buf_advantages[buf_idx][s:s + valid].numpy()
                    )
                    dones_np[:valid, j] = np.asarray(
                        buf.dones[s:s + valid], dtype=np.float32
                    )
                    mask_np[:valid, j] = 1.0

                    init = buf.lstm_states[s]
                    if init is not None:
                        init_h_list.append(init[0].squeeze(0))  # (hidden,)
                        init_c_list.append(init[1].squeeze(0))
                    else:
                        zero = self.model.init_hidden(
                            batch_size=1, device="cpu"
                        )
                        init_h_list.append(zero[0].squeeze(0))
                        init_c_list.append(zero[1].squeeze(0))

                obs_t = torch.from_numpy(obs_np).to(self.device)
                actions_t = torch.from_numpy(actions_np).to(self.device)
                old_lp_t = torch.from_numpy(old_lp_np).to(self.device)
                returns_t = torch.from_numpy(returns_np).to(self.device)
                advs_t = torch.from_numpy(advs_np).to(self.device)
                dones_t = torch.from_numpy(dones_np).to(self.device)
                mask_t = torch.from_numpy(mask_np).to(self.device)

                init_h = torch.stack(init_h_list, dim=0).to(self.device)  # (B, hidden)
                init_c = torch.stack(init_c_list, dim=0).to(self.device)
                init_state = (init_h, init_c)

                # Forward (T, B, ...) once.
                # Request encoder features when junction aux head is active so
                # we can compute the auxiliary loss without a second encoder pass.
                use_junction_aux = (
                    self.model.junction_head is not None
                    and self._junction_ch_idx is not None
                )
                fwd_result = self.model.forward_sequence(
                    obs_t, init_state, dones_t,
                    return_enc_features=use_junction_aux,
                )
                if use_junction_aux:
                    logits_seq, values_seq, enc_feats_flat = fwd_result
                    # enc_feats_flat: (T*B, hidden_dim)
                else:
                    logits_seq, values_seq = fwd_result
                # logits_seq: (T, B, N_ACTIONS), values_seq: (T, B)

                dist = torch.distributions.Categorical(logits=logits_seq)
                log_prob = dist.log_prob(actions_t)  # (T, B)
                entropy = dist.entropy()             # (T, B)

                ratio = torch.exp(log_prob - old_lp_t)

                mask_sum = mask_t.sum().clamp(min=1.0)

                with torch.no_grad():
                    kl_per = ((ratio - 1) - (log_prob - old_lp_t)) * mask_t
                    approx_kl = (kl_per.sum() / mask_sum).item()
                    total_kl += approx_kl
                    epoch_kl_sum += approx_kl
                    epoch_kl_n += 1

                surr1 = ratio * advs_t
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * advs_t
                )
                per_step_p = -torch.min(surr1, surr2)
                p_loss = (per_step_p * mask_t).sum() / mask_sum

                # Clamp targets only (matches FF path).
                target_t = torch.clamp(
                    returns_t, -self.value_clamp, self.value_clamp
                )
                per_step_v = (values_seq - target_t).pow(2)
                v_loss = (per_step_v * mask_t).sum() / mask_sum

                ent_loss = (entropy * mask_t).sum() / mask_sum

                loss = p_loss + self.value_coef * v_loss - self.entropy_coef * ent_loss

                # Junction auxiliary loss — uses encoder features already computed
                # above; no second encoder pass needed.
                if use_junction_aux:
                    j_logits = self.model.junction_head(enc_feats_flat)  # (T*B, 3)
                    obs_flat = obs_t.reshape(T_chunk * B, *obs_t.shape[2:])
                    j_vals = obs_flat[:, self._junction_ch_idx,
                                      self._obs_center, self._obs_center]  # (T*B,)
                    j_class = torch.zeros(T_chunk * B, dtype=torch.long, device=self.device)
                    j_class[j_vals > 0.7] = 2   # junction
                    j_class[(j_vals > 0.3) & (j_vals <= 0.7)] = 1  # endpoint
                    mask_flat = mask_t.reshape(-1)
                    j_loss_per = nn.functional.cross_entropy(
                        j_logits, j_class, reduction="none"
                    )
                    j_loss = (j_loss_per * mask_flat).sum() / mask_sum
                    loss = loss + 0.1 * j_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                gn_val = grad_norm.item()
                total_gn += gn_val
                if gn_val > max_gn:
                    max_gn = gn_val
                self.optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += ent_loss.item()
                n_updates += 1

            epochs_run += 1

            # Adaptive KL early-stopping (mirrors the FF path).
            if self.target_kl is not None and epoch_kl_n > 0:
                epoch_kl_mean = epoch_kl_sum / epoch_kl_n
                if epoch_kl_mean > self.target_kl:
                    break

        # Explained variance across all envs
        with torch.no_grad():
            all_returns = torch.cat(
                [r for r in per_buf_returns if r is not None]
            )
            values_all = torch.tensor(all_values_raw, dtype=torch.float32)
            ev = (
                1
                - (all_returns.cpu() - values_all).var()
                / (all_returns.cpu().var() + 1e-8)
            ).item()

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
    # Dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _ep_length_weights(dones: list) -> np.ndarray:
        """Per-step weight = sqrt(episode_length / mean_episode_length).

        Counteracts the frequency bias from short episodes: with many short
        episodes per iteration, their cumulative gradient weight dominates
        even though individual long episodes carry more learning signal.
        Weighting by sqrt(L / mean_L) before global normalisation amplifies
        long-episode advantages without collapsing the normalised scale.

        Incomplete trailing episodes (no done at buffer end) are assigned
        their observed length so they are still up-weighted relative to
        single-step episodes.
        """
        dones_arr = np.asarray(dones, dtype=np.float32)
        n = len(dones_arr)
        if n == 0:
            return np.ones(0, dtype=np.float32)

        weights = np.ones(n, dtype=np.float32)
        ep_lengths = []
        start = 0
        for t in range(n):
            if dones_arr[t] > 0 or t == n - 1:
                ep_len = t - start + 1
                ep_lengths.append(ep_len)
                weights[start:t + 1] = float(ep_len)
                start = t + 1

        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 1.0
        return np.sqrt(np.maximum(weights / max(mean_len, 1.0), 0.0)).astype(np.float32)

    def _ppo_update(
        self, buffers: List[RolloutBuffer], last_values: List[float]
    ) -> Dict[str, float]:
        if self.use_lstm:
            return self._ppo_update_lstm(buffers, last_values)
        return self._ppo_update_ff(buffers, last_values)

    def load_checkpoint(self, save_path: str, imitation_path: str) -> Tuple[int, float]:
        """Resume from PPO checkpoint if it exists,
        otherwise load imitation weights.
        Returns (start_iteration, best_cldice).
        """
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                try:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except (KeyError, TypeError, ValueError):
                    # Scheduler type changed (e.g. LinearLR → SequentialLR
                    # after adding warmup). Safe to skip — the scheduler
                    # restarts fresh; the LR will ramp from scratch.
                    print("  Scheduler state incompatible, starting fresh.")

            if "curriculum_stage" in ckpt:
                self.curriculum.current_stage_idx = ckpt["curriculum_stage"]
                print(
                    f"  Restored curriculum stage: "
                    f"{self.curriculum.get_current_stage().name}"
                )

            start = ckpt.get("iteration", 0) + 1
            best = ckpt.get("best_cldice", ckpt.get("best_f1", 0.0))
            print(f"Resumed from PPO checkpoint  iter={start-1}  best_clDice={best:.3f}")
            return start, best

        if os.path.exists(imitation_path):
            ckpt = torch.load(
                imitation_path, map_location=self.device, weights_only=True
            )
            # Strip incompatible heads (e.g. old N_ACTIONS=8 actor) before
            # loading so a stale imitation ckpt does not block PPO startup.
            state = dict(ckpt["model_state_dict"])
            model_state = self.model.state_dict()
            stripped = []
            for k in list(state.keys()):
                if k in model_state and state[k].shape != model_state[k].shape:
                    stripped.append((k, tuple(state[k].shape), tuple(model_state[k].shape)))
                    del state[k]
            # strict=False: imitation checkpoint may lack LSTM / value head weights
            self.model.load_state_dict(state, strict=False)
            print(f"Loaded imitation weights  val_acc={ckpt.get('val_acc', 0):.3f}")
            if stripped:
                print(
                    "  Skipped mismatched tensors (will be re-initialised): "
                    + ", ".join(f"{k} ckpt={a} model={b}" for k, a, b in stripped)
                )
            # Reset value head — never trained during imitation
            for layer in self.model.value_head:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)
            print("Value head re-initialized.")
            # If the actor's last layer was stripped (N_ACTIONS changed),
            # re-initialise it with the same scheme used at construction:
            # near-uniform logits + slight negative STOP bias.
            actor_last_keys = {"actor_head.3.weight", "actor_head.3.bias"}
            if any(k in actor_last_keys for k, _, _ in stripped):
                last = self.model.actor_head[-1]
                nn.init.orthogonal_(last.weight, gain=0.01)
                nn.init.zeros_(last.bias)
                if self.model.N_ACTIONS == 9:
                    with torch.no_grad():
                        last.bias[8] = -1.0
                print("Actor head last layer re-initialised for N_ACTIONS="
                      f"{self.model.N_ACTIONS}.")
            return 1, 0.0

        print("WARNING: No weights found, training from scratch.")
        return 1, 0.0

    
    def _pick_sample_index(self, train_samples) -> int:
        """Pick a random sample index filtered by current curriculum difficulty."""
        difficulty = self.curriculum.get_difficulty()
        if not hasattr(self, "_sample_difficulties"):
            # Pre-compute once, cache for all future calls
            self._sample_difficulties = []
            for i in range(len(train_samples)):
                s = train_samples[i]
                d = self.curriculum.compute_sample_difficulty(
                    s["centerline"], s.get("vessel_mask", s["centerline"])
                )
                self._sample_difficulties.append(d)

        valid = [i for i, d in enumerate(self._sample_difficulties) if d <= difficulty]
        if len(valid) < 10:
            valid = list(range(min(10, len(train_samples))))
        return int(np.random.choice(valid))

    def _collect_rollout_vec(
        self,
        vec_env,
        buffers: List[RolloutBuffer],
        train_samples,
        obs_list,
        lstm_states_list,
        ep_rewards,
        ep_lengths,
        episode_rewards: deque,
        episode_lengths: deque,
        current_sample_ids: List[int],
        accumulated_coverage: dict,
    ):
        """Collect self.steps_per_iter steps across multiple envs.

        Uses SubprocVecEnv for parallel env stepping and batched policy inference.
        One RolloutBuffer per environment for clean temporal sequences.
        """
        n_envs = vec_env.n_envs
        steps_collected = 0

        while steps_collected < self.steps_per_iter:
            # Batch observations for all envs
            obs_batch = torch.from_numpy(
                np.stack(obs_list)
            ).float().to(self.device)  # (n_envs, C, H, W)

            # Batch LSTM states: (1, n_envs, hidden)
            if self.use_lstm:
                h_cat = torch.cat([s[0].to(self.device) for s in lstm_states_list], dim=0)
                h_c_cat = torch.cat([s[1].to(self.device) for s in lstm_states_list], dim=0)
                batched_lstm = (h_cat, h_c_cat)
            else:
                batched_lstm = None

            with torch.no_grad():
                actions, log_probs, _, values, new_lstm = (
                    self.model.get_action_and_value(obs_batch, batched_lstm))

            # Step ALL envs in parallel
            action_list = [actions[i].item() for i in range(n_envs)]
            all_obs, all_rewards, all_terminated, all_truncated, all_infos = (
                vec_env.step(action_list))

            # Process results (bookkeeping only — fast)
            for i in range(n_envs):
                action_i = action_list[i]
                log_prob_i = log_probs[i].item()
                value_i = values[i].item()
                next_obs = all_obs[i]
                reward = all_rewards[i]
                done = all_terminated[i] or all_truncated[i]

                # 1) Capture LSTM state BEFORE this action (for buffer)
                if self.use_lstm:
                    lstm_state_i = (
                        lstm_states_list[i][0],
                        lstm_states_list[i][1],
                    )
                else:
                    lstm_state_i = None

                # Normalise: terminal steps (done=True) carry the topology signal
                # (terminal F1 + connectivity penalty) which is sparse and has a
                # different scale than dense per-step rewards.  Using a separate
                # normalizer for terminal steps preserves the relative magnitude
                # of the topology bonus instead of drowning it in step statistics.
                if done:
                    self.terminal_reward_normalizer.update(reward)
                    norm_reward = self.terminal_reward_normalizer.normalize(reward)
                else:
                    self.reward_normalizer.update(reward)
                    norm_reward = self.reward_normalizer.normalize(reward)

                # 2) Store transition with OLD lstm state
                buffers[i].add(
                    obs_list[i],
                    action_i,
                    log_prob_i,
                    norm_reward,
                    value_i,
                    float(done),
                    lstm_state_i,
                )

                ep_rewards[i] += reward
                ep_lengths[i] += 1
                steps_collected += 1

                # Accumulate per-component reward means for W&B logging
                info_i = all_infos[i] if isinstance(all_infos, (list, tuple)) else {}
                if isinstance(info_i, dict):
                    for _key in RewardCalculator.BREAKDOWN_KEYS:
                        _val = info_i.get(_key)
                        if _val is not None:
                            self._rwrd_sums[_key] = self._rwrd_sums.get(_key, 0.0) + _val
                            self._rwrd_counts[_key] = self._rwrd_counts.get(_key, 0) + 1

                # 3) Update LSTM state to NEW state from forward pass
                if self.use_lstm:
                    lstm_states_list[i] = (
                        new_lstm[0][i:i+1, :].detach().cpu(),
                        new_lstm[1][i:i+1, :].detach().cpu(),
                    )

                if done:
                    episode_rewards.append(ep_rewards[i])
                    episode_lengths.append(ep_lengths[i])

                    success = self.curriculum.is_episode_successful(all_infos[i])
                    prev_stage = self.curriculum.current_stage_idx
                    self.curriculum.step(success=success)

                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0

                    # Reset LSTM on episode boundary
                    if self.use_lstm:
                        fresh = self.model.init_hidden(
                            batch_size=1, device=self.device
                        )
                        lstm_states_list[i] = (
                            fresh[0].detach().cpu(),
                            fresh[1].detach().cpu(),
                        )

                    # Accumulate centerline coverage for multi-episode training.
                    # Retrieve the mask before loading the next sample — it is
                    # still valid in the worker until set_sample overwrites it.
                    # This closes the train-eval gap: the frontier_tracer
                    # provides prior_coverage at inference but training never
                    # did, making the gated connectivity bonus always zero.
                    finished_sample_id = current_sample_ids[i]
                    cov_mask = vec_env.get_coverage_mask(i)
                    if cov_mask is not None and finished_sample_id >= 0:
                        prev = accumulated_coverage.get(finished_sample_id)
                        if prev is None:
                            accumulated_coverage[finished_sample_id] = (
                                cov_mask > 0
                            ).astype(np.float32)
                        else:
                            accumulated_coverage[finished_sample_id] = np.where(
                                cov_mask > 0, 1.0, prev
                            )
                        # Cap memory: discard the oldest entry when the dict
                        # grows too large (worst case: n_envs × max_traces images).
                        if len(accumulated_coverage) > 512:
                            accumulated_coverage.pop(
                                next(iter(accumulated_coverage))
                            )

                    # Sample new episode via curriculum — send index only (no large IPC)
                    sample_idx = self._pick_sample_index(train_samples)
                    current_sample_ids[i] = sample_idx
                    prior_cov = accumulated_coverage.get(sample_idx)
                    vec_env.set_sample(i, sample_idx, prior_coverage=prior_cov)
                    overrides = self._get_curriculum_overrides_dict()
                    vec_env.apply_overrides(i, overrides)
                    next_obs = vec_env.reset(i)

                    if self.curriculum.current_stage_idx != prev_stage:
                        stage = self.curriculum.get_current_stage()
                        print(f"  → Curriculum stage: {stage.name}")
                        # Apply new stage to all envs
                        new_overrides = self._get_curriculum_overrides_dict()
                        for j in range(n_envs):
                            vec_env.apply_overrides(j, new_overrides)

                obs_list[i] = next_obs

        # Return last values for GAE bootstrap
        obs_batch = torch.tensor(
            np.array(obs_list), dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            if self.use_lstm:
                h_cat = torch.cat([s[0] for s in lstm_states_list], dim=0).to(self.device)
                h_c_cat = torch.cat([s[1] for s in lstm_states_list], dim=0).to(self.device)
                last_values = self.model.get_value(obs_batch, (h_cat, h_c_cat))
            else:
                last_values = self.model.get_value(obs_batch, None)
        return [last_values[i].item() for i in range(n_envs)]

    def train(
        self,
        train_samples,
        val_samples,
        save_path: str,
        log_path: str,
        imitation_path: str = "",
    ) -> None:
        from environment.vec_env import SubprocVecEnv

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        start_iteration, best_cldice = self.load_checkpoint(save_path, imitation_path)

        N_ENVS = self.config.get("training", {}).get("ppo", {}).get("n_envs", 8)
        vec_env = SubprocVecEnv(self.config, n_envs=N_ENVS)

        # Track current sample index per env so we can look up/update
        # the per-image accumulated coverage after each episode.
        current_sample_ids: List[int] = [-1] * N_ENVS
        # Accumulated centerline coverage per dataset sample index.
        # Enables multi-episode coverage training: when the same image is
        # traced in successive episodes, the agent sees which segments have
        # already been covered and earns the gated connectivity bonus.
        accumulated_coverage: dict = {}

        # Initialize all envs
        obs_list = []
        lstm_states_list = []
        ep_rewards = [0.0] * N_ENVS
        ep_lengths = [0] * N_ENVS

        overrides = self._get_curriculum_overrides_dict()
        for i in range(N_ENVS):
            sample_idx = self._pick_sample_index(train_samples)
            current_sample_ids[i] = sample_idx
            vec_env.set_sample(i, sample_idx)
            vec_env.apply_overrides(i, overrides)
            obs = vec_env.reset(i)
            obs_list.append(obs)
            hidden = self.model.init_hidden(batch_size=1, device=self.device)
            lstm_states_list.append(tuple(t.detach().cpu() for t in hidden) if hidden is not None else None)

        buffers = [RolloutBuffer() for _ in range(N_ENVS)]
        episode_rewards: deque = deque(maxlen=50)
        episode_lengths: deque = deque(maxlen=50)

        _csv_fields = [
            "iteration", "mean_reward", "mean_ep_length",
            "policy_loss", "value_loss", "entropy", "approx_kl",
            "explained_variance", "grad_norm", "lr", "stage",
        ] + list(RewardCalculator.BREAKDOWN_KEYS) + [
            "val_coverage", "val_f1", "val_cldice",
        ]
        _csv_file = open(log_path, "w", newline="", encoding="utf-8")
        _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields, extrasaction="ignore")
        _csv_writer.writeheader()

        print(f"Starting curriculum stage: {self.curriculum.get_current_stage().name}")
        print(
            f"\nStarting PPO — iters {start_iteration}–{self.num_iterations} "
            f"× {self.steps_per_iter} steps  {N_ENVS} envs"
            f"  LSTM={'ON chunk_len=' + str(self.lstm_chunk_length) if self.use_lstm else 'OFF'}\n"
        )

        # Initial sync for the per-stage iteration counter (in case the
        # checkpoint resumed mid-curriculum).
        self._last_stage_idx = self.curriculum.current_stage_idx

        for iteration in range(start_iteration, self.num_iterations + 1):
            # Tick the per-stage iteration counter, resetting on stage change
            # so intra-stage entropy annealing restarts at each new stage.
            if self.curriculum.current_stage_idx != self._last_stage_idx:
                self._stage_iter = 0
                self._last_stage_idx = self.curriculum.current_stage_idx
                # New stage — unfreeze entropy so annealing restarts fresh
                self._entropy_frozen = False
                self._eval_cldice_window.clear()
            self._stage_iter += 1

            for buf in buffers:
                buf.reset()
            self._rwrd_sums: dict = {}
            self._rwrd_counts: dict = {}
            self.model.eval()

            last_values = self._collect_rollout_vec(
                vec_env, buffers, train_samples,
                obs_list, lstm_states_list,
                ep_rewards, ep_lengths,
                episode_rewards, episode_lengths,
                current_sample_ids=current_sample_ids,
                accumulated_coverage=accumulated_coverage,
            )

            # --- Update ---
            self.model.train()
            stats = self._ppo_update(buffers, last_values)
            current_lr = self.scheduler.get_last_lr()[0]
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
                    self.model, val_samples, self.config, self.device, self.tolerance,
                    n_episodes=4,
                )
                stage = self.curriculum.get_current_stage()
                log += (
                    f"  |  val_cov={ev['mean_coverage']:.3f}"
                    f"  val_f1={ev['mean_f1']:.3f}"
                    f"  val_clDice={ev['mean_cldice']:.3f}"
                    f"  stage={stage.name}"
                    f"  ent_c={self.entropy_coef:.3f}"
                )

                if ev["mean_cldice"] > best_cldice:
                    best_cldice = ev["mean_cldice"]
                    self.no_improve_count = 0
                    torch.save(
                        {
                            "iteration": iteration,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "best_cldice": best_cldice,
                            "config": self.config,
                            "curriculum_stage": self.curriculum.current_stage_idx,
                        },
                        save_path,
                    )
                    log += f"  ✓ saved (best clDice={best_cldice:.3f})"
                else:
                    self.no_improve_count += 1

                # Performance-gated entropy annealing: update the freeze state
                # based on whether val_clDice is improving over the last 3 evals.
                self._eval_cldice_window.append(ev["mean_cldice"])
                if len(self._eval_cldice_window) >= 2:
                    recent_improvement = (
                        max(self._eval_cldice_window)
                        - min(self._eval_cldice_window)
                    )
                    was_frozen = self._entropy_frozen
                    self._entropy_frozen = recent_improvement < 0.005
                    if self._entropy_frozen and not was_frozen:
                        log += f"  [entropy frozen @ {self.entropy_coef:.4f}]"
                    elif not self._entropy_frozen and was_frozen:
                        log += "  [entropy unfrozen]"
                if self.no_improve_count >= self.patience:
                    print(log)
                    print(
                        f"\nEarly stopping: no improvement for "
                        f"{self.patience} eval cycles."
                    )
                    break

            print(log)
            _csv_row = {
                "iteration": iteration,
                "mean_reward": mean_reward,
                "mean_ep_length": mean_length,
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "entropy": stats["entropy"],
                "approx_kl": stats["approx_kl"],
                "explained_variance": stats["explained_variance"],
                "grad_norm": stats["grad_norm"],
                "lr": current_lr,
                "stage": self.curriculum.get_current_stage().name,
            }
            for _k in RewardCalculator.BREAKDOWN_KEYS:
                _csv_row[_k] = (
                    self._rwrd_sums.get(_k, 0.0) / max(self._rwrd_counts.get(_k, 1), 1)
                )
            if iteration % self.eval_every == 0 and val_samples and "ev" in dir():
                _csv_row.update({
                    "val_coverage": ev["mean_coverage"],
                    "val_f1": ev["mean_f1"],
                    "val_cldice": ev["mean_cldice"],
                })
            _csv_writer.writerow(_csv_row)
            _csv_file.flush()

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

        vec_env.close()
        _csv_file.close()

        print(f"\nDone. Best clDice: {best_cldice:.3f}")
        print(f"Weights: {save_path}")
        print(f"Log:     {log_path}")