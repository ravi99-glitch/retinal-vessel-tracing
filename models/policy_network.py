# models/policy_network.py
"""Policy network for vessel tracing RL agent.

Actor-Critic with CNN encoder → optional LSTMCell → actor + critic heads.

Two forward modes:
  forward()          — single timestep, used during rollout collection
  forward_sequence() — T-step chunk with done masks, used during PPO training
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_in_channels(config: Dict[str, Any]) -> int:
    """Single source of truth for observation channel count.

    Base channels (always present, 10):
      RGB                : 3
      visited mask       : 1
      distance transform : 1
      vessel grad dy     : 1
      vessel grad dx     : 1
      centerline mask    : 1
      vessel tangent dy  : 1
      vessel tangent dx  : 1

    Optional channels (in this order):
      curvature   (use_curvature,   default True)
      junction    (use_junction,    default True)
      vesselness  (use_vesselness,  default False)
      unet_prior  (use_unet_prior,  default False)
      prev_action (use_prev_action, default False) — adds 2 channels

    Must stay in sync with ObservationBuilder in environment/observation.py.
    """
    env = config.get("environment", {})
    n = 10
    if env.get("use_curvature", True):
        n += 1
    if env.get("use_junction", True):
        n += 1
    if env.get("use_vesselness", False):
        n += 1
    if env.get("use_unet_prior", False):
        n += 1
    if env.get("use_global_visited", False):
        n += 1
    if env.get("use_prior_coverage", False):
        n += 1
    if env.get("use_prev_action", False):
        n += 2
    return n


def _junction_channel_idx(config: Dict[str, Any]) -> Optional[int]:
    """Return the observation channel index of the junction map, or None.

    Used by the PPO trainer to extract per-step junction supervision targets
    from the stored observation batch without re-running the environment.
    """
    env = config.get("environment", {})
    if not env.get("use_junction", True):
        return None
    n = 10  # base channels
    if env.get("use_curvature", True):
        n += 1  # curvature precedes junction in stacked sources
    return n  # junction channel index


# ------------------------------------------------------------------
# Encoders
# ------------------------------------------------------------------


class CNNEncoder(nn.Module):
    """Deeper CNN encoder with BatchNorm + Global Average Pooling.

    5-layer conv stack with dilated convolutions in the last two blocks.
    Receptive field covers the full 65px crop (~81px theoretical) so
    branch-point context is visible even at the edge of the observation.
    GAP replaces the 2.3M-param Linear bottleneck so total encoder
    params drop by ~6× while feature extraction capacity goes up.

    RF derivation (k=kernel, s=stride, d=dilation, j=jump):
      L1  k5 s2 d1  → j=2,  RF=5
      L2  k3 s2 d1  → j=4,  RF=9
      L3  k3 s1 d1  → j=4,  RF=17
      L4  k3 s2 d2  → j=8,  RF=33
      L5  k3 s1 d3  → j=8,  RF=81
    """

    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 65 -> 33,  RF=5
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 33 -> 17,  RF=9
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 17 -> 17,  RF=17
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 17 -> 9,   RF=33  (dilation=2)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 9  -> 9,   RF=81  (dilation=3)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # Soft spatial attention + global-max readout.
        # Pure GAP loses the spatial arrangement of features — at a bifurcation
        # the agent needs to know which direction each branch goes, but GAP only
        # knows "branches exist somewhere."  A 1×1 attention conv learns WHERE
        # in the 9×9 feature map to focus; global-max retains the peak signal.
        # At initialisation (attn_conv weights = 0) attention is uniform, which
        # matches the behaviour of the old avg pool so training starts stably.
        self.attn_conv = nn.Conv2d(128, 1, kernel_size=1, bias=True)
        self.gap_max = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),  # 128 attended + 128 max
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)              # (B, 128, 9, 9)
        B = x.shape[0]
        # Soft spatial attention: learn which spatial positions are relevant.
        attn = self.attn_conv(x)             # (B, 1, 9, 9)
        attn = F.softmax(attn.view(B, -1), dim=-1).view(B, 1, 9, 9)
        x_attn = (x * attn).sum(dim=(2, 3)) # (B, 128) — spatially attended mean
        x_max = self.gap_max(x).flatten(1)  # (B, 128)
        x = torch.cat([x_attn, x_max], dim=1)  # (B, 256)
        x = self.dropout(x)
        return self.fc(x)                    # (B, hidden_dim)


class ResNetEncoder(nn.Module):
    """Lightweight ResNet-style encoder with residual blocks."""

    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.relu(x + self.block(x))

    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = self.ResBlock(32)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.layer2 = self.ResBlock(64)
        # Soft spatial attention + global-max (same rationale as CNNEncoder).
        self.attn_conv = nn.Conv2d(64, 1, kernel_size=1, bias=True)
        self.gap_max = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),  # 64 attended + 64 max
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        B = x.shape[0]
        attn = self.attn_conv(x)             # (B, 1, H', W')
        attn = F.softmax(attn.view(B, -1), dim=-1).view(B, 1, *x.shape[2:])
        x_attn = (x * attn).sum(dim=(2, 3)) # (B, 64)
        x_max = self.gap_max(x).flatten(1)  # (B, 64)
        x = torch.cat([x_attn, x_max], dim=1)  # (B, 128)
        x = self.dropout(x)
        return self.fc(x)


# ------------------------------------------------------------------
# LSTM head
# ------------------------------------------------------------------


class LSTMHead(nn.Module):
    """Single-layer LSTMCell with layer norm on input.

    Uses LSTMCell (not nn.LSTM) because RL rollout is step-by-step.
    For training on T-step chunks, the caller loops explicitly
    via ActorCriticNetwork.forward_sequence().

    Owns a learnable initial hidden state ``(init_h, init_c)`` that is
    used whenever an episode begins or a ``done`` event resets the state
    inside ``forward_sequence``. Gradients flow into these parameters
    through every reset event during a chunked PPO update.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        # Learnable initial hidden state, broadcast to the batch dim at use.
        self.init_h = nn.Parameter(torch.zeros(hidden_dim))
        self.init_c = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.ln(x)
        if state is None:
            B = x.size(0)
            h = self.init_h.unsqueeze(0).expand(B, -1).contiguous()
            c = self.init_c.unsqueeze(0).expand(B, -1).contiguous()
        else:
            h, c = state
        h, c = self.lstm(x, (h, c))
        return h, (h, c)


# ------------------------------------------------------------------
# Actor-Critic
# ------------------------------------------------------------------


class ActorCriticNetwork(nn.Module):
    """Actor-Critic policy network.

    Architecture:
      encoder (CNN | ResNet) → optional LSTMCell → actor head + critic head

    Two forward paths:
      forward()          — one timestep  (rollout collection)
      forward_sequence() — T-step chunk  (PPO training with done masks)
    """

    N_ACTIONS = 9  # N, NE, E, SE, S, SW, W, NW + STOP (index 8)

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        policy_cfg = config.get("policy", {})

        hidden_dim = policy_cfg.get("hidden_dim", 128)
        lstm_hidden = policy_cfg.get("lstm_hidden", 128)
        head_hidden = policy_cfg.get("head_hidden", 128)
        self.use_lstm = policy_cfg.get("use_lstm", False)
        dropout = policy_cfg.get("dropout", 0.0)
        encoder_type = policy_cfg.get("encoder_type", "cnn")

        in_channels = _compute_in_channels(config)

        # ---- encoder ----
        if encoder_type == "resnet":
            self.encoder = ResNetEncoder(in_channels, hidden_dim, dropout)
        else:
            self.encoder = CNNEncoder(in_channels, hidden_dim, dropout)

        # ---- optional recurrence ----
        if self.use_lstm:
            self.lstm_head = LSTMHead(hidden_dim, lstm_hidden)
            feature_dim = lstm_hidden
        else:
            self.lstm_head = None
            feature_dim = hidden_dim

        # ---- heads ----
        # LayerNorm + Tanh after the head bottleneck stabilises PPO updates
        # against the intra-epoch feature distribution drift caused by
        # multiple gradient passes over the same rollout buffer.
        self.actor_head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, self.N_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1),
        )

        # ---- junction auxiliary head ----
        # Attached to ENCODER features (before LSTM) so it forces the CNN to
        # build junction-discriminative representations.  Predicts 3 classes:
        #   0 = background / straight vessel
        #   1 = endpoint   (~0.5 on junction map)
        #   2 = junction   (~1.0 on junction map)
        # Supervised by the junction-map value at the center pixel of each obs.
        env_cfg = config.get("environment", {})
        use_junction_aux = (
            policy_cfg.get("use_junction_aux", True)
            and env_cfg.get("use_junction", True)
        )
        if use_junction_aux:
            self.junction_head: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
            )
        else:
            self.junction_head = None

        self._init_weights()

    # ---- initialisation ----

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

        # Near-uniform initial policy
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        # Bias the STOP logit (index 8) slightly negative so the agent does
        # not collapse to "always STOP" before discovering it pays off.
        if self.N_ACTIONS == 9:
            with torch.no_grad():
                self.actor_head[-1].bias[8] = -1.0

        # Attention conv starts with zero weights → uniform softmax over the
        # 9×9 spatial grid, equivalent to the original average pooling.
        # The network diverges from uniform attention as it encounters junctions.
        if hasattr(self.encoder, "attn_conv"):
            nn.init.zeros_(self.encoder.attn_conv.weight)
            nn.init.zeros_(self.encoder.attn_conv.bias)

    def init_hidden(
        self, batch_size: int = 1, device: Union[torch.device, str] = "cpu"
    ):
        """Return the learnable initial hidden state, broadcast over batch.

        Detached so rollout collection (which only needs forward inference)
        does not pin a gradient graph. The same parameters are referenced
        directly inside ``forward_sequence`` on done-reset events, where
        the gradient path remains intact for training.
        Returns ``None`` when LSTM is disabled.
        """
        if not self.use_lstm:
            return None
        h = (
            self.lstm_head.init_h.detach()
            .to(device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        c = (
            self.lstm_head.init_c.detach()
            .to(device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        return h, c

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Return CNN encoder features (before LSTM) for auxiliary losses."""
        return self.encoder(obs)

    # ---- single-step forward (rollout) ----

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Process one timestep.

        Args:
            obs:   (B, C, H, W)
            state: (h, c) each (B, lstm_hidden) or None

        Returns:
            logits: (B, N_ACTIONS)
            values: (B,)
            state:  updated (h, c) or None
        """
        features = self.encoder(obs)

        if self.lstm_head is not None:
            features, state = self.lstm_head(features, state)

        logits = self.actor_head(features)
        # 10% gradient leak: critic can weakly shape the encoder without dominating actor
        values = self.value_head(features * 0.1 + features.detach() * 0.9).squeeze(-1)
        return logits, values, state

    # ---- sequence forward (PPO training) ----

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        init_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        dones: torch.Tensor,
        return_enc_features: bool = False,
    ):
        """Process a T-step chunk for recurrent PPO training.

        Loops over timesteps, resetting hidden state wherever
        dones[t]==1 (episode ended at step t, so step t+1 starts fresh).

        Args:
            obs_seq:             (T, B, C, H, W)  — sequential observations
            init_state:          (h, c) each (B, lstm_hidden)
            dones:               (T, B) float — 1.0 at timesteps where episode ended
            return_enc_features: if True, also return encoder features shaped
                                 (T*B, hidden_dim) so callers can compute
                                 auxiliary losses without a second encoder pass.

        Returns:
            all_logits: (T, B, N_ACTIONS)
            all_values: (T, B)
            enc_features: (T*B, hidden_dim) — only when return_enc_features=True

        When use_lstm=False, simply batch-forwards all T*B observations
        in one pass (no sequential dependency).
        """
        T, B = obs_seq.shape[:2]

        if not self.use_lstm:
            flat = obs_seq.reshape(T * B, *obs_seq.shape[2:])
            features = self.encoder(flat)
            logits = self.actor_head(features).view(T, B, -1)
            values = self.value_head(features * 0.1 + features.detach() * 0.9).squeeze(-1).view(T, B)
            if return_enc_features:
                return logits, values, features  # (T*B, hidden_dim)
            return logits, values

        # Recurrent path: encode all frames in one batch, then loop LSTM
        flat = obs_seq.reshape(T * B, *obs_seq.shape[2:])
        all_features = self.encoder(flat).view(T, B, -1)  # (T, B, hidden_dim)

        h, c = init_state
        all_logits = []
        all_values = []

        for t in range(T):
            # Reset hidden state for environments that ended at previous
            # step. The reset target is the *learnable* initial state, so
            # gradients propagate into ``lstm_head.init_h/init_c`` through
            # every done event observed during the chunk.
            if t > 0:
                done_mask = dones[t - 1].unsqueeze(-1)  # (B, 1)
                init_h_b = self.lstm_head.init_h.unsqueeze(0).expand_as(h)
                init_c_b = self.lstm_head.init_c.unsqueeze(0).expand_as(c)
                h = h * (1.0 - done_mask) + init_h_b * done_mask
                c = c * (1.0 - done_mask) + init_c_b * done_mask

            features_t, (h, c) = self.lstm_head(all_features[t], (h, c))
            all_logits.append(self.actor_head(features_t))
            all_values.append(self.value_head(features_t * 0.1 + features_t.detach() * 0.9).squeeze(-1))

        if return_enc_features:
            # Return pre-LSTM encoder features (T*B, hidden_dim) for aux losses
            return torch.stack(all_logits), torch.stack(all_values), all_features.reshape(T * B, -1)
        return torch.stack(all_logits), torch.stack(all_values)

    # ---- convenience methods ----

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Sample action + compute log_prob, entropy, value. Used during rollout."""
        logits, values, state = self.forward(obs, state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values, state

    def get_value(
        self,
        obs: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Value only — used for GAE bootstrap."""
        _, values, _ = self.forward(obs, state)
        return values