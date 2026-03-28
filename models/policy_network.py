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

    Channels:
      RGB                : 3
      visited mask       : 1
      distance transform : 1
      vessel grad dy     : 1
      vessel grad dx     : 1
      centerline mask    : 1    ← NEW
      vessel tangent dy  : 1    ← NEW
      vessel tangent dx  : 1    ← NEW

    Total (default): 10
    """
    n = 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # 10
    if config.get("environment", {}).get("use_vesselness", False):
        n += 1
    return n


# ------------------------------------------------------------------
# Encoders
# ------------------------------------------------------------------


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

        dummy = torch.zeros(1, in_channels, 65, 65)
        with torch.no_grad():
            flat_size = self.conv_layers(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


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
        self.dropout = nn.Dropout(dropout)

        # Mirror actual forward path exactly
        dummy = torch.zeros(1, in_channels, 65, 65)
        with torch.no_grad():
            out = self.layer2(self.down1(self.layer1(self.stem(dummy))))
            flat_size = out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ------------------------------------------------------------------
# LSTM head
# ------------------------------------------------------------------


class LSTMHead(nn.Module):
    """Single-layer LSTMCell with layer norm on input.

    Uses LSTMCell (not nn.LSTM) because RL rollout is step-by-step.
    For training on T-step chunks, the caller loops explicitly
    via ActorCriticNetwork.forward_sequence().
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.ln(x)
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            c = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
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

    N_ACTIONS = 9  # N, NE, E, SE, S, SW, W, NW, STOP

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        policy_cfg = config.get("policy", {})

        hidden_dim = policy_cfg.get("hidden_dim", 128)
        lstm_hidden = policy_cfg.get("lstm_hidden", 128)
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
        self.actor_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.N_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

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

    def init_hidden(
        self, batch_size: int = 1, device: Union[torch.device, str] = "cpu"
    ):
        """Create zero hidden state. Returns None when LSTM is disabled."""
        if not self.use_lstm:
            return None
        return (
            torch.zeros(batch_size, self.lstm_head.hidden_dim, device=device),
            torch.zeros(batch_size, self.lstm_head.hidden_dim, device=device),
        )

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
        values = self.value_head(features).squeeze(-1)
        return logits, values, state

    # ---- sequence forward (PPO training) ----

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        init_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a T-step chunk for recurrent PPO training.

        Loops over timesteps, resetting hidden state wherever
        dones[t]==1 (episode ended at step t, so step t+1 starts fresh).

        Args:
            obs_seq:    (T, B, C, H, W)  — sequential observations
            init_state: (h, c) each (B, lstm_hidden), hidden state at
                        the START of this chunk (stored during rollout)
            dones:      (T, B) float — 1.0 at timesteps where episode ended

        Returns:
            all_logits: (T, B, N_ACTIONS)
            all_values: (T, B)

        When use_lstm=False, simply batch-forwards all T*B observations
        in one pass (no sequential dependency).
        """
        T, B = obs_seq.shape[:2]

        if not self.use_lstm:
            flat = obs_seq.reshape(T * B, *obs_seq.shape[2:])
            features = self.encoder(flat)
            logits = self.actor_head(features).view(T, B, -1)
            values = self.value_head(features).squeeze(-1).view(T, B)
            return logits, values

        # Recurrent path: encode all frames in one batch, then loop LSTM
        flat = obs_seq.reshape(T * B, *obs_seq.shape[2:])
        all_features = self.encoder(flat).view(T, B, -1)  # (T, B, hidden_dim)

        h, c = init_state
        all_logits = []
        all_values = []

        for t in range(T):
            # Reset hidden state for environments that ended at previous step
            if t > 0:
                done_mask = dones[t - 1].unsqueeze(-1)  # (B, 1)
                h = h * (1.0 - done_mask)
                c = c * (1.0 - done_mask)

            features_t, (h, c) = self.lstm_head(all_features[t], (h, c))
            all_logits.append(self.actor_head(features_t))
            all_values.append(self.value_head(features_t).squeeze(-1))

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
